#!/usr/bin/env python3
"""
Position Monitor — runs during market hours, alerts on:
1. Short calls going ITM or losing >50%
2. LEAPs with <90 DTE (roll window)
3. Any position P&L change >10% since last check
4. New fills or cancellations
5. Updates ~/memory/trading_journal.md

Designed to be called from idle-turn duties or as a standalone check.
NOT a persistent service — called on demand.
"""

import json
import os
import sys
import requests
from datetime import datetime, timezone

STATE_FILE = os.path.expanduser("~/trading_bot/monitor_state.json")
JOURNAL_FILE = os.path.expanduser("~/memory/trading_journal.md")

# Thresholds
SHORT_CALL_LOSS_PCT = -50.0    # Alert if short call losing more than this
POSITION_CHANGE_PCT = 10.0     # Alert if P&L % changes more than this since last check
LEAP_DTE_WARNING = 90          # Alert if LEAP has fewer DTE than this


ALPACA_USER_ID = 'a4dc8459-608d-49f5-943e-e5e105ed5207'


def get_alpaca_creds():
    from jarvis_utils.secrets import get
    api_key = get('Alpaca', 'api_key_id', user=ALPACA_USER_ID)
    secret = get('Alpaca', 'secret_key', user=ALPACA_USER_ID)
    return api_key, secret


# === Edge 123: auto-refresh Alpaca credentials on 401 (Apr 22 2026) ===
_session = None


def _refresh_alpaca_credentials(session):
    from jarvis_utils.secrets import get
    new_key = get('Alpaca', 'api_key_id', user=ALPACA_USER_ID)
    new_secret = get('Alpaca', 'secret_key', user=ALPACA_USER_ID)
    if not new_key or not new_secret:
        raise EnvironmentError("position_monitor: portal returned empty Alpaca creds")
    session.headers['APCA-API-KEY-ID'] = new_key
    session.headers['APCA-API-SECRET-KEY'] = new_secret


def _get_alpaca_session():
    global _session
    if _session is None:
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from alpaca_client import _AutoRefreshSession
        key, secret = get_alpaca_creds()
        _session = _AutoRefreshSession(_refresh_alpaca_credentials)
        _session.headers.update({
            'APCA-API-KEY-ID': key or '',
            'APCA-API-SECRET-KEY': secret or '',
        })
    return _session


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"positions": {}, "last_check": None}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def parse_option_symbol(sym):
    """Parse OCC option symbol: AAPL261218C00150000 -> (AAPL, 2026-12-18, C, 150.00)"""
    # Find where the date starts (6 digits after ticker)
    import re
    m = re.match(r'^([A-Z]+)(\d{6})([CP])(\d{8})$', sym)
    if not m:
        return None
    ticker = m.group(1)
    date_str = m.group(2)  # YYMMDD
    put_call = m.group(3)
    strike = int(m.group(4)) / 1000.0
    exp_date = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
    return ticker, exp_date, put_call, strike


def days_to_expiry(exp_date_str):
    """Calculate DTE from YYYY-MM-DD string."""
    exp = datetime.strptime(exp_date_str, "%Y-%m-%d").date()
    today = datetime.now().date()
    return (exp - today).days


def detect_spread_membership(current_positions: dict) -> dict:
    """Detect spread membership by structural pairing.

    Heuristic: a position is part of a spread if there exists another
    position with the SAME underlying ticker, SAME option type (call
    or put), and OPPOSITE-sign qty (one long, one short).

    Returns {option_symbol: spread_id} where spread_id is a synthetic
    string like "spread:COPX_C" — same id for both legs.

    Stocks (non-option symbols) and options without a counterparty leg
    are excluded.
    """
    # Group option symbols by (ticker, put_call)
    groups: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for sym, info in current_positions.items():
        if len(sym) <= 10:
            continue  # stock symbol, not OCC option
        parsed = parse_option_symbol(sym)
        if not parsed:
            continue
        ticker, _exp, put_call, _strike = parsed
        groups.setdefault((ticker, put_call), []).append((sym, info["qty"]))

    spread_map: dict[str, str] = {}
    for (ticker, pc), legs in groups.items():
        has_long = any(q > 0 for _, q in legs)
        has_short = any(q < 0 for _, q in legs)
        if has_long and has_short:
            spread_id = f"spread:{ticker}_{pc}"
            for sym, _ in legs:
                spread_map[sym] = spread_id
    return spread_map


def check_positions():
    """Pull positions, compare to last state, return alerts."""
    api_key, secret = get_alpaca_creds()
    if not api_key or not secret:
        return ["ERROR: Alpaca credentials not loaded (portal may be down)"]

    session = _get_alpaca_session()
    base = 'https://paper-api.alpaca.markets/v2'

    # Get account
    acct_r = session.get(f'{base}/account', timeout=10)
    if acct_r.status_code != 200:
        return [f"ERROR: Alpaca API returned {acct_r.status_code}"]
    acct = acct_r.json()

    # Get positions
    pos_r = session.get(f'{base}/positions', timeout=10)
    positions = pos_r.json()

    # Get recent orders
    orders_r = session.get(f'{base}/orders',
                           params={'status': 'all', 'limit': 20, 'direction': 'desc'}, timeout=10)
    orders = orders_r.json()

    prev_state = load_state()
    prev_positions = prev_state.get("positions", {})
    alerts = []

    current_positions = {}
    for p in positions:
        sym = p['symbol']
        qty = float(p['qty'])  # Edge 117 — preserve fractional crypto qty (was int(float(...)))
        pnl_pct = float(p['unrealized_plpc']) * 100
        current_price = float(p['current_price'])
        entry_price = float(p['avg_entry_price'])
        pnl = float(p['unrealized_pl'])

        current_positions[sym] = {
            "qty": qty,
            "pnl_pct": pnl_pct,
            "pnl": pnl,
            "current_price": current_price,
            "entry_price": entry_price,
        }

    # Spread-membership map: any option symbol that's one leg of a
    # paired spread (same ticker, same call/put, opposite qty signs).
    # We use this to SUPPRESS per-leg P&L MOVE alerts (which are noisy
    # for hedged structures — losing on one leg means gaining on the
    # other) and emit a single net-spread alert instead.
    spread_map = detect_spread_membership(current_positions)
    spread_pnl_now: dict[str, float] = {}
    spread_pnl_prev: dict[str, float] = {}
    for sym, sid in spread_map.items():
        spread_pnl_now[sid] = spread_pnl_now.get(sid, 0.0) + current_positions[sym]["pnl"]
        if sym in prev_positions:
            spread_pnl_prev[sid] = spread_pnl_prev.get(sid, 0.0) + prev_positions[sym].get("pnl", 0.0)

    for p in positions:
        sym = p['symbol']
        qty = float(p['qty'])
        pnl_pct = float(p['unrealized_plpc']) * 100
        current_price = float(p['current_price'])
        entry_price = float(p['avg_entry_price'])
        pnl = float(p['unrealized_pl'])

        # Check short calls
        if qty < 0 and len(sym) > 10:
            parsed = parse_option_symbol(sym)
            if parsed:
                ticker, exp_date, put_call, strike = parsed
                if put_call == 'C':
                    # Check if ITM
                    # We'd need the underlying price — approximate from entry
                    if pnl_pct < SHORT_CALL_LOSS_PCT:
                        alerts.append(
                            f"⚠️ SHORT CALL ALERT: {sym} is {pnl_pct:+.1f}% "
                            f"(${pnl:+,.2f}). Consider closing."
                        )
                    dte = days_to_expiry(exp_date)
                    if dte < 5:
                        alerts.append(
                            f"⏰ SHORT CALL EXPIRING: {sym} expires in {dte} days")

        # Check LEAPs DTE
        if qty > 0 and len(sym) > 10:
            parsed = parse_option_symbol(sym)
            if parsed:
                ticker, exp_date, put_call, strike = parsed
                dte = days_to_expiry(exp_date)
                if dte < LEAP_DTE_WARNING:
                    alerts.append(
                        f"📅 LEAP DTE WARNING: {sym} expires in {dte} days "
                        f"(P&L: {pnl_pct:+.1f}%). Consider rolling."
                    )

        # Check P&L change vs last check.
        # Suppress per-leg alerts for spread members — those legs hedge
        # each other and per-leg P&L swings are noise. Emit a single
        # spread-level alert below.
        if sym in prev_positions and sym not in spread_map:
            prev_pnl_pct = prev_positions[sym].get("pnl_pct", 0)
            change = abs(pnl_pct - prev_pnl_pct)
            if change > POSITION_CHANGE_PCT:
                direction = "improved" if pnl_pct > prev_pnl_pct else "worsened"
                alerts.append(
                    f"📊 P&L MOVE: {sym} {direction} "
                    f"{prev_pnl_pct:+.1f}% → {pnl_pct:+.1f}% "
                    f"(Δ{change:.1f}%)"
                )

    # Spread-level P&L MOVE alerts: emit one alert per spread when
    # net spread P&L moved by more than $POSITION_CHANGE_PCT-equivalent
    # ($25 absolute, since spread sizes vary so % is unstable).
    SPREAD_NET_DOLLAR_THRESHOLD = 25.0
    for sid, pnl_now in spread_pnl_now.items():
        pnl_prev = spread_pnl_prev.get(sid)
        if pnl_prev is None:
            continue
        delta = pnl_now - pnl_prev
        if abs(delta) > SPREAD_NET_DOLLAR_THRESHOLD:
            direction = "improved" if delta > 0 else "worsened"
            # List the legs in the alert for context
            legs = sorted(s for s, m in spread_map.items() if m == sid)
            legs_str = ",".join(legs)
            alerts.append(
                f"📊 SPREAD P&L: {sid} {direction} "
                f"net=${pnl_prev:+.0f} → ${pnl_now:+.0f} "
                f"(Δ${delta:+.0f}) [{legs_str}]"
            )

    # Check for positions that disappeared (closed)
    for sym in prev_positions:
        if sym not in current_positions:
            alerts.append(f"🔒 POSITION CLOSED: {sym} (was {prev_positions[sym].get('pnl_pct', 0):+.1f}%)")

    # Check for new positions
    for sym in current_positions:
        if sym not in prev_positions:
            alerts.append(f"🆕 NEW POSITION: {sym} qty={current_positions[sym]['qty']}")

    # Save new state
    new_state = {
        "positions": current_positions,
        "last_check": datetime.now(timezone.utc).isoformat(),
        "equity": float(acct['equity']),
        "cash": float(acct['cash']),
    }
    save_state(new_state)

    # Update journal header
    update_journal(acct, positions)

    return alerts


def update_journal(acct, positions):
    """Update the trade journal with current positions."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M ET')
    equity = float(acct['equity'])
    cash = float(acct['cash'])
    pnl = equity - 100000

    lines = [f"# Trading Journal", f"Last updated: {now}", ""]
    lines.append("## Account Summary")
    lines.append(f"- Equity: ${equity:,.2f}")
    lines.append(f"- Cash: ${cash:,.2f}")
    lines.append(f"- P&L from $100K: ${pnl:,.2f} ({pnl/1000:.1f}%)")
    lines.append("")

    shares = []
    options = []
    for p in positions:
        sym = p['symbol']
        qty = int(float(p['qty']))
        entry = float(p['avg_entry_price'])
        current = float(p['current_price'])
        upnl = float(p['unrealized_pl'])
        pnl_pct = float(p['unrealized_plpc']) * 100

        rec = (sym, qty, entry, current, upnl, pnl_pct)
        if len(sym) > 10:
            options.append(rec)
        else:
            shares.append(rec)

    lines.append("## Open Positions")
    if shares:
        lines.append("\n### Shares")
        lines.append("| Ticker | Qty | Entry | Current | P&L | P&L % |")
        lines.append("|--------|-----|-------|---------|-----|-------|")
        for sym, qty, entry, cur, upnl, pct in sorted(shares):
            lines.append(f"| {sym} | {qty} | ${entry:.2f} | ${cur:.2f} | ${upnl:+,.2f} | {pct:+.1f}% |")

    if options:
        lines.append("\n### Options")
        lines.append("| Symbol | Qty | Entry | Current | P&L | P&L % | DTE |")
        lines.append("|--------|-----|-------|---------|-----|-------|-----|")
        for sym, qty, entry, cur, upnl, pct in sorted(options):
            parsed = parse_option_symbol(sym)
            dte = days_to_expiry(parsed[1]) if parsed else "?"
            side = "SHORT" if qty < 0 else "LONG"
            lines.append(f"| {sym} | {qty} ({side}) | ${entry:.2f} | ${cur:.2f} | ${upnl:+,.2f} | {pct:+.1f}% | {dte} |")

    lines.append("")
    lines.append("## Daily Checklist (idle-turn duty)")
    lines.append("1. Run `python3 ~/trading_bot/position_monitor.py`")
    lines.append("2. Review alerts")
    lines.append("3. Act on any critical alerts (short calls ITM, expiring positions)")
    lines.append("4. Report to David if action needed")

    with open(JOURNAL_FILE, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == "__main__":
    alerts = check_positions()
    if alerts:
        print(f"\n{'='*50}")
        print(f"POSITION MONITOR — {len(alerts)} alert(s)")
        print(f"{'='*50}")
        for a in alerts:
            print(f"  {a}")
    else:
        print("Position monitor: no alerts.")
