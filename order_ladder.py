#!/usr/bin/env python3
"""
Order Price Ladder — works the bid/ask spread for options orders.

Instead of submitting one limit order and hoping, this module:
1. Starts at a favorable price (mid or better)
2. Steps toward the unfavorable side in small increments
3. Waits between steps for fills
4. Stops at a hard max price (never chases past limit)

For SELLING (to open): start at ask, walk down toward mid
For BUYING (to close): start at bid, walk up toward mid
"""

import time
import requests
import logging

logger = logging.getLogger(__name__)

# Default config
DEFAULT_STEP_SIZE = 0.05       # $0.05 per step
DEFAULT_WAIT_SECONDS = 30      # 30 seconds between steps
DEFAULT_MAX_STEPS = 10         # Max 10 steps before giving up


def get_option_quote(symbol, headers):
    """Get current bid/ask for an option."""
    # Use Alpaca options data API
    url = f"https://data.alpaca.markets/v1beta1/options/quotes/latest"
    params = {"symbols": symbol, "feed": "indicative"}
    r = requests.get(url, headers=headers, params=params, timeout=10)
    if r.status_code != 200:
        logger.error(f"Quote fetch failed for {symbol}: {r.status_code}")
        return None

    data = r.json()
    quotes = data.get("quotes", {})
    if symbol not in quotes:
        logger.error(f"No quote data for {symbol}")
        return None

    q = quotes[symbol]
    return {
        "bid": float(q.get("bp", 0)),
        "ask": float(q.get("ap", 0)),
        "bid_size": int(q.get("bs", 0)),
        "ask_size": int(q.get("as", 0)),
        "mid": (float(q.get("bp", 0)) + float(q.get("ap", 0))) / 2,
    }


def submit_limit_order(symbol, qty, side, limit_price, headers,
                       time_in_force="day"):
    """Submit a limit order and return the order object."""
    url = "https://paper-api.alpaca.markets/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": str(abs(qty)),
        "side": side,  # "buy" or "sell"
        "type": "limit",
        "time_in_force": time_in_force,
        "limit_price": str(round(limit_price, 2)),
    }
    r = requests.post(url, headers=headers, json=payload, timeout=10)
    if r.status_code in (200, 201):
        order = r.json()
        logger.info(f"Order submitted: {side} {qty} {symbol} @ ${limit_price:.2f} "
                    f"[{order['id']}]")
        return order
    else:
        logger.error(f"Order failed: {r.status_code} — {r.text}")
        return None


def cancel_order(order_id, headers):
    """Cancel an open order."""
    url = f"https://paper-api.alpaca.markets/v2/orders/{order_id}"
    r = requests.delete(url, headers=headers, timeout=10)
    if r.status_code in (200, 204):
        logger.info(f"Order {order_id} canceled")
        return True
    else:
        logger.warning(f"Cancel failed for {order_id}: {r.status_code}")
        return False


def check_order_status(order_id, headers):
    """Check if an order has been filled."""
    url = f"https://paper-api.alpaca.markets/v2/orders/{order_id}"
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 200:
        return r.json()
    return None


def ladder_buy_to_close(symbol, qty, max_price, headers,
                        step_size=DEFAULT_STEP_SIZE,
                        wait_seconds=DEFAULT_WAIT_SECONDS,
                        max_steps=DEFAULT_MAX_STEPS):
    """
    Buy to close a short option position, walking up from bid toward max_price.

    Args:
        symbol: Option symbol (e.g., "KGC260417C00027000")
        qty: Number of contracts to buy (positive)
        max_price: Maximum price willing to pay (hard cap)
        headers: Alpaca API headers
        step_size: Price increment per step
        wait_seconds: Seconds to wait between steps
        max_steps: Maximum number of price steps

    Returns:
        dict with 'filled', 'fill_price', 'steps_taken', 'order'
    """
    quote = get_option_quote(symbol, headers)
    if not quote:
        return {"filled": False, "error": "Could not get quote"}

    bid = quote["bid"]
    ask = quote["ask"]
    mid = quote["mid"]
    spread = ask - bid

    # Start at natural price (slightly above bid)
    start_price = round(bid + step_size, 2)
    if start_price > max_price:
        start_price = max_price

    logger.info(f"LADDER BUY {symbol}: bid=${bid:.2f} ask=${ask:.2f} "
               f"mid=${mid:.2f} spread=${spread:.2f} | "
               f"start=${start_price:.2f} max=${max_price:.2f}")

    current_price = start_price
    for step in range(max_steps):
        if current_price > max_price:
            logger.info(f"LADDER STOP: price ${current_price:.2f} exceeds "
                       f"max ${max_price:.2f}")
            break

        # Submit order
        order = submit_limit_order(symbol, qty, "buy", current_price, headers)
        if not order:
            return {"filled": False, "error": "Order submission failed",
                    "steps_taken": step + 1}

        # Wait for fill
        logger.info(f"LADDER STEP {step+1}: ${current_price:.2f} — "
                   f"waiting {wait_seconds}s...")
        time.sleep(wait_seconds)

        # Check status
        status = check_order_status(order["id"], headers)
        if status and status["status"] == "filled":
            fill_price = float(status.get("filled_avg_price", current_price))
            logger.info(f"LADDER FILLED at ${fill_price:.2f} on step {step+1}")
            return {
                "filled": True,
                "fill_price": fill_price,
                "steps_taken": step + 1,
                "order": status,
            }

        # Not filled — cancel and step up
        cancel_order(order["id"], headers)
        current_price = round(current_price + step_size, 2)

    logger.info(f"LADDER EXHAUSTED after {max_steps} steps. Not filled.")
    return {"filled": False, "steps_taken": max_steps,
            "last_price": current_price - step_size}


def ladder_sell_to_open(symbol, qty, min_price, headers,
                        step_size=DEFAULT_STEP_SIZE,
                        wait_seconds=DEFAULT_WAIT_SECONDS,
                        max_steps=DEFAULT_MAX_STEPS):
    """
    Sell to open a covered call, walking down from ask toward min_price.

    Args:
        symbol: Option symbol
        qty: Number of contracts to sell (positive)
        min_price: Minimum price willing to accept (hard floor)
        headers: Alpaca API headers
        step_size: Price decrement per step
        wait_seconds: Seconds to wait between steps
        max_steps: Maximum number of price steps

    Returns:
        dict with 'filled', 'fill_price', 'steps_taken', 'order'
    """
    quote = get_option_quote(symbol, headers)
    if not quote:
        return {"filled": False, "error": "Could not get quote"}

    bid = quote["bid"]
    ask = quote["ask"]
    mid = quote["mid"]
    spread = ask - bid

    # Start at natural price (slightly below ask)
    start_price = round(ask - step_size, 2)
    if start_price < min_price:
        start_price = min_price

    logger.info(f"LADDER SELL {symbol}: bid=${bid:.2f} ask=${ask:.2f} "
               f"mid=${mid:.2f} spread=${spread:.2f} | "
               f"start=${start_price:.2f} min=${min_price:.2f}")

    current_price = start_price
    for step in range(max_steps):
        if current_price < min_price:
            logger.info(f"LADDER STOP: price ${current_price:.2f} below "
                       f"min ${min_price:.2f}")
            break

        # Submit order
        order = submit_limit_order(symbol, qty, "sell", current_price, headers)
        if not order:
            return {"filled": False, "error": "Order submission failed",
                    "steps_taken": step + 1}

        # Wait for fill
        logger.info(f"LADDER STEP {step+1}: ${current_price:.2f} — "
                   f"waiting {wait_seconds}s...")
        time.sleep(wait_seconds)

        # Check status
        status = check_order_status(order["id"], headers)
        if status and status["status"] == "filled":
            fill_price = float(status.get("filled_avg_price", current_price))
            logger.info(f"LADDER FILLED at ${fill_price:.2f} on step {step+1}")
            return {
                "filled": True,
                "fill_price": fill_price,
                "steps_taken": step + 1,
                "order": status,
            }

        # Not filled — cancel and step down
        cancel_order(order["id"], headers)
        current_price = round(current_price - step_size, 2)

    logger.info(f"LADDER EXHAUSTED after {max_steps} steps. Not filled.")
    return {"filled": False, "steps_taken": max_steps,
            "last_price": current_price + step_size}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s | %(levelname)s | %(message)s')

    # Test: get a quote for the KGC short call
    from jarvis_utils.secrets import get
    api_key = get('Alpaca', 'api_key_id', user='a4dc8459-608d-49f5-943e-e5e105ed5207')
    secret = get('Alpaca', 'secret_key', user='a4dc8459-608d-49f5-943e-e5e105ed5207')
    headers = {'APCA-API-KEY-ID': api_key, 'APCA-API-SECRET-KEY': secret}

    # Just test quote fetching
    quote = get_option_quote("KGC260417C00027000", headers)
    if quote:
        print(f"\nKGC Apr17 $27C quote:")
        print(f"  Bid: ${quote['bid']:.2f} x {quote['bid_size']}")
        print(f"  Ask: ${quote['ask']:.2f} x {quote['ask_size']}")
        print(f"  Mid: ${quote['mid']:.2f}")
        print(f"  Spread: ${quote['ask'] - quote['bid']:.2f}")
    else:
        print("Could not get quote")
