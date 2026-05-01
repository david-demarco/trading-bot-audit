#!/usr/bin/env python3
"""
CC Lifecycle Manager — automated covered call management.

Rules:
1. SELL: When RSI5>70 + RSI14>60 (per-ticker thresholds), sell 30% OTM ~60d call
2. BUYBACK: Place GTC buy-to-close at 25% profit immediately after selling
3. ASSIGNMENT RISK: If stock within 10% of strike with <14 DTE, close the CC
4. NEAR EXPIRY: If <7 DTE and profitable, let expire. If <7 DTE and losing, evaluate.
5. RE-SELL: After buyback fills, wait for RSI conditions to re-trigger

Run: python3 ~/trading_bot/cc_manager.py [--check] [--place-buybacks]
"""
import sys, os, json, logging
from datetime import datetime, timedelta

sys.path.insert(0, "/opt/jarvis-utils/lib")
from jarvis_utils.secrets import get
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger("cc_manager")

API_KEY = get("Alpaca", "api_key_id", user="a4dc8459-608d-49f5-943e-e5e105ed5207")
SECRET = get("Alpaca", "secret_key", user="a4dc8459-608d-49f5-943e-e5e105ed5207")
BASE = "https://paper-api.alpaca.markets"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET}

BUYBACK_TARGET = 0.75  # buy back at 75% of sold price (= 25% profit)
ASSIGNMENT_RISK_PCT = 0.10  # warn if stock within 10% of strike
CLOSE_DTE = 7  # evaluate at 7 DTE


def get_positions():
    return requests.get(f"{BASE}/v2/positions", headers=HEADERS, timeout=5).json()


def get_open_orders():
    return requests.get(f"{BASE}/v2/orders?status=open", headers=HEADERS, timeout=5).json()


def get_stock_price(symbol):
    """Get live stock price for underlying."""
    try:
        r = requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
                         headers=HEADERS, params={"feed": "iex"}, timeout=5)
        return r.json().get("trade", {}).get("p", 0)
    except:
        return 0


def parse_option_symbol(sym):
    """Parse option symbol like KGC260417C00027000 → (ticker, date, type, strike)."""
    # Format: TICKER + YYMMDD + C/P + strike*1000
    import re
    m = re.match(r'([A-Z]+)(\d{6})([CP])(\d+)', sym)
    if not m:
        return None
    ticker = m.group(1)
    date_str = m.group(2)
    opt_type = 'call' if m.group(3) == 'C' else 'put'
    strike = int(m.group(4)) / 1000
    expiry = datetime.strptime('20' + date_str, '%Y%m%d')
    dte = (expiry - datetime.now()).days
    return {'ticker': ticker, 'expiry': expiry, 'type': opt_type, 'strike': strike, 'dte': dte}


def place_buyback(symbol, qty, target_price):
    """Place GTC limit buy-to-close order."""
    order = {
        "symbol": symbol,
        "qty": str(abs(qty)),
        "side": "buy",
        "type": "limit",
        "limit_price": str(round(target_price, 2)),
        "time_in_force": "gtc",
    }
    r = requests.post(f"{BASE}/v2/orders", headers=HEADERS, json=order, timeout=5)
    result = r.json()
    log.info(f"BUYBACK: buy {symbol} @ ${target_price:.2f} → {result.get('status', 'error')}")
    return result


def check_positions():
    """Check all CC positions and manage lifecycle."""
    positions = get_positions()
    orders = get_open_orders()
    order_syms = {o['symbol'] for o in orders if o['side'] == 'buy'}

    short_calls = []
    for p in positions:
        qty = float(p['qty'])
        if qty < 0:
            parsed = parse_option_symbol(p['symbol'])
            if parsed and parsed['type'] == 'call':
                short_calls.append({
                    'symbol': p['symbol'],
                    'qty': qty,
                    'entry_price': float(p['avg_entry_price']),
                    'current_price': float(p['current_price']),
                    'pnl': float(p['unrealized_pl']),
                    **parsed,
                })

    if not short_calls:
        log.info("No short calls found.")
        return

    log.info(f"Found {len(short_calls)} short calls:")
    for sc in short_calls:
        sym = sc['symbol']
        entry = sc['entry_price']
        current = sc['current_price']
        dte = sc['dte']
        strike = sc['strike']
        stock_price = get_stock_price(sc['ticker'])
        pct_to_strike = (strike - stock_price) / stock_price * 100 if stock_price > 0 else 999

        # Status
        profitable = current < entry
        target = entry * BUYBACK_TARGET
        has_buyback = sym in order_syms

        log.info(f"  {sym}: sold ${entry:.2f}, now ${current:.2f}, DTE={dte}, "
                 f"stock=${stock_price:.2f}, {pct_to_strike:.0f}% to strike")

        # Rule 0: Validate strike vs entry (should never have been sold below entry)
        stock_positions = {p['symbol']: float(p['avg_entry_price']) for p in positions if len(p['symbol']) <= 6}
        stock_entry = stock_positions.get(sc['ticker'], 0)
        if stock_entry > 0 and strike < stock_entry:
            log.info(f"    ❌ BAD CC: strike ${strike:.2f} < stock entry ${stock_entry:.2f} — assignment = net loss")

        # Rule 1: Place buyback if missing
        if not has_buyback and dte > CLOSE_DTE:
            log.info(f"    → MISSING buyback. Target: ${target:.2f}")
            if current <= target:
                log.info(f"    → Already at target! Should buy back NOW")
            else:
                place_buyback(sym, sc['qty'], target)

        # Rule 2: Assignment risk warning
        if pct_to_strike < ASSIGNMENT_RISK_PCT * 100 and dte < 14:
            log.info(f"    ⚠️ ASSIGNMENT RISK: stock {pct_to_strike:.0f}% from strike, {dte} DTE")

        # Rule 3: Near expiry evaluation
        if dte <= CLOSE_DTE:
            if profitable:
                log.info(f"    → {dte} DTE, profitable — LET EXPIRE for max profit")
            else:
                log.info(f"    → {dte} DTE, LOSING ${abs(sc['pnl']):.0f} — evaluate close vs hold")
                if pct_to_strike < 5:
                    log.info(f"    → CLOSE recommended (near strike, assignment likely)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CC Lifecycle Manager")
    parser.add_argument("--check", action="store_true", help="Check all CC positions")
    parser.add_argument("--place-buybacks", action="store_true", help="Place missing buyback orders")
    args = parser.parse_args()

    check_positions()
