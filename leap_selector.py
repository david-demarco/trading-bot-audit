"""
leap_selector.py - LEAP selection and purchase module for the PMCC trading bot.

Handles finding deep ITM LEAP calls that qualify as long legs for Poor Man's
Covered Call (diagonal spread) strategies, and executing purchases.

A LEAP is selected based on:
  - Delta between 0.65 and 0.85 (deep ITM, behaves like stock ownership)
  - DTE between 180 and 365 days (6-12 months, sweet spot ~270 days)
  - Tight bid-ask spread (max 15% of mid-price for illiquid mining options)
  - Adequate open interest for liquidity

Capital allocation is enforced before any purchase:
  - Single LEAP cost must not exceed 5% of portfolio equity
  - Total LEAP allocation must not exceed 20% of portfolio equity

Usage:
    from leap_selector import select_leap, execute_leap_purchase, check_leap_sizing

    client = AlpacaOptionsClient()
    candidate = select_leap("GDX", client)
    if candidate:
        ok, reason = check_leap_sizing(candidate, equity=100_000)
        if ok:
            result = execute_leap_purchase(candidate, client, dry_run=False)

Author: PMCC Integration Module
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytz

# Black-Scholes helpers from the existing pmcc_manager module.
# These are module-level functions (prefixed with _ in pmcc_manager.py but
# still importable).
from pmcc_manager import (
    _bs_greeks as bs_greeks,
    _bs_call_price as bs_call_price,
    _implied_volatility as implied_volatility,
)

logger = logging.getLogger(__name__)

_ET = pytz.timezone("US/Eastern")

# Default risk-free rate (mirrors slvr_cc_config.RISK_FREE_RATE)
_DEFAULT_RISK_FREE_RATE = 0.045


def _today_et() -> date:
    """Return the current calendar date in US/Eastern time."""
    return datetime.now(_ET).date()


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class LeapCandidate:
    """Represents a single LEAP call option that qualifies for PMCC long leg.

    All price fields are per-share (not per-contract).
    """
    symbol: str            # OCC option symbol, e.g. "GDX261218C00080000"
    ticker: str            # Underlying ticker, e.g. "GDX"
    strike: float          # Strike price
    expiry: str            # Expiration date YYYY-MM-DD
    dte: int               # Days to expiration
    estimated_delta: float # Black-Scholes estimated delta
    bid: float             # Best bid price (per share)
    ask: float             # Best ask price (per share)
    mid_price: float       # (bid + ask) / 2
    cost_per_contract: float  # mid_price * 100
    open_interest: int     # Open interest
    spread_pct: float      # (ask - bid) / mid as a decimal (e.g. 0.03 = 3%)


# =============================================================================
# LEAP SELECTION
# =============================================================================

def select_leap(
    ticker: str,
    options_client,  # AlpacaOptionsClient instance
    min_dte: int = 180,
    max_dte: int = 365,
    min_delta: float = 0.65,
    max_delta: float = 0.95,
    max_spread_pct: float = 0.15,
) -> Optional[LeapCandidate]:
    """Select the best deep ITM LEAP call for a PMCC long leg.

    Searches the option chain for the given ticker and returns the single
    best candidate that meets all criteria, or None if nothing qualifies.

    Selection process:
      1. Get the current underlying price.
      2. Query Alpaca option chain API for calls in the DTE window.
      3. Fetch snapshots for quote data (bid/ask/open interest).
      4. Estimate delta via Black-Scholes using historical volatility.
      5. Filter on delta, spread, and liquidity.
      6. Score and rank candidates.

    Args:
        ticker: Underlying ticker symbol (e.g. "GDX").
        options_client: An AlpacaOptionsClient instance for API calls.
        min_dte: Minimum days to expiration (default 180).
        max_dte: Maximum days to expiration (default 365).
        min_delta: Minimum acceptable delta (default 0.70).
        max_delta: Maximum acceptable delta (default 0.80).
        max_spread_pct: Maximum bid-ask spread as fraction of mid (default 0.05).

    Returns:
        The best LeapCandidate, or None if no qualifying candidate is found.
    """
    logger.info(
        "LEAP SELECT: searching %s | DTE=%d-%d | delta=%.2f-%.2f | max_spread=%.0f%%",
        ticker, min_dte, max_dte, min_delta, max_delta, max_spread_pct * 100,
    )

    # ---- Step 1: Get current underlying price ----
    underlying_price = _get_underlying_price(ticker, options_client)
    if underlying_price is None or underlying_price <= 0:
        logger.error("LEAP SELECT: cannot get price for %s -- aborting", ticker)
        return None

    logger.info("LEAP SELECT: %s underlying price = $%.2f", ticker, underlying_price)

    # ---- Step 2: Compute date window ----
    today = _today_et()
    exp_gte = (today + timedelta(days=min_dte)).strftime("%Y-%m-%d")
    exp_lte = (today + timedelta(days=max_dte)).strftime("%Y-%m-%d")

    # For deep ITM calls, strike should be well below current price.
    # A delta of 0.70-0.80 typically means strike is 10-30% below spot
    # depending on volatility and time.  Use a generous range.
    strike_lte = round(underlying_price * 0.95, 2)  # at most 5% below spot
    strike_gte = round(underlying_price * 0.50, 2)  # at least 50% below spot

    # ---- Step 3: Estimate historical volatility for BS calculations ----
    hist_vol = _estimate_historical_vol(ticker, options_client)
    sigma = hist_vol * 1.10 if hist_vol and hist_vol > 0 else 0.50
    logger.info("LEAP SELECT: using sigma=%.2f (HV=%.2f)", sigma, hist_vol or 0.0)

    # ---- Step 4: Query option chain (Alpaca primary, yfinance fallback) ----
    candidates = _query_alpaca_chain(
        ticker, options_client, underlying_price, sigma,
        exp_gte, exp_lte, strike_gte, strike_lte,
        min_dte, max_dte, min_delta, max_delta, max_spread_pct,
    )

    if not candidates:
        logger.info(
            "LEAP SELECT: Alpaca chain returned no candidates for %s, trying yfinance fallback",
            ticker,
        )
        candidates = _query_yfinance_chain(
            ticker, underlying_price, sigma,
            min_dte, max_dte, strike_gte, strike_lte,
            min_delta, max_delta, max_spread_pct,
        )

    if not candidates:
        logger.warning("LEAP SELECT: no qualifying LEAP candidates for %s", ticker)
        return None

    # ---- Step 5: Score and rank ----
    for c in candidates:
        c._score = _score_candidate(c)

    candidates.sort(key=lambda c: c._score, reverse=True)

    # Log top candidates
    logger.info("LEAP SELECT: %d candidate(s) for %s (showing top 5):", len(candidates), ticker)
    for i, c in enumerate(candidates[:5]):
        logger.info(
            "  #%d: %s | $%.0f exp=%s DTE=%d | delta=%.3f | mid=$%.2f "
            "(cost=$%.0f) | spread=%.1f%% | OI=%d | score=%.2f",
            i + 1, c.symbol, c.strike, c.expiry, c.dte, c.estimated_delta,
            c.mid_price, c.cost_per_contract, c.spread_pct * 100,
            c.open_interest, c._score,
        )

    best = candidates[0]
    # Clean up the temporary score attribute
    for c in candidates:
        if hasattr(c, "_score"):
            delattr(c, "_score")

    logger.info(
        "LEAP SELECT: BEST for %s -> %s $%.0f exp=%s DTE=%d | "
        "delta=%.3f | mid=$%.2f | cost=$%.0f",
        ticker, best.symbol, best.strike, best.expiry, best.dte,
        best.estimated_delta, best.mid_price, best.cost_per_contract,
    )

    return best


def _get_underlying_price(ticker: str, options_client) -> Optional[float]:
    """Get the current price of the underlying via Alpaca, with yfinance fallback."""
    # Try Alpaca first
    try:
        price = options_client.get_latest_price(ticker)
        if price and price > 0:
            return price
    except Exception as e:
        logger.debug("Alpaca price fetch failed for %s: %s", ticker, e)

    # Fallback to yfinance
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if hist is not None and len(hist) > 0:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        logger.debug("yfinance price fetch failed for %s: %s", ticker, e)

    return None


def _estimate_historical_vol(ticker: str, options_client) -> Optional[float]:
    """Estimate annualized historical volatility from daily bars.

    Uses 20-day realized volatility (HV-20) from Alpaca daily bars.
    Falls back to a default of 0.50 if data is unavailable.
    """
    try:
        bars_dict = options_client.get_daily_bars([ticker], days=60)
        df = bars_dict.get(ticker)
        if df is not None and len(df) >= 20:
            log_returns = df["close"].pct_change().dropna()
            if len(log_returns) >= 20:
                hv20 = float(log_returns.tail(20).std() * math.sqrt(252))
                if hv20 > 0:
                    return hv20
    except Exception as e:
        logger.debug("HV estimation failed for %s: %s", ticker, e)

    return None


def _query_alpaca_chain(
    ticker: str,
    options_client,
    underlying_price: float,
    sigma: float,
    exp_gte: str,
    exp_lte: str,
    strike_gte: float,
    strike_lte: float,
    min_dte: int,
    max_dte: int,
    min_delta: float,
    max_delta: float,
    max_spread_pct: float,
) -> List[LeapCandidate]:
    """Query the Alpaca option chain API and filter for LEAP candidates."""
    candidates: List[LeapCandidate] = []
    today = _today_et()

    try:
        contracts = options_client.get_option_contracts(
            underlying=ticker,
            option_type="call",
            expiration_gte=exp_gte,
            expiration_lte=exp_lte,
            strike_gte=strike_gte,
            strike_lte=strike_lte,
            limit=100,
        )
    except Exception as e:
        logger.warning("LEAP SELECT: Alpaca chain query failed for %s: %s", ticker, e)
        return []

    if not contracts:
        logger.debug("LEAP SELECT: Alpaca returned no contracts for %s", ticker)
        return []

    logger.info("LEAP SELECT: Alpaca returned %d contracts for %s", len(contracts), ticker)

    # Collect OCC symbols for bulk snapshot fetch
    symbol_map: Dict[str, Dict] = {}
    for contract in contracts:
        occ_symbol = contract.get("symbol", "")
        if not occ_symbol:
            continue

        # Validate expiration and compute DTE
        expiry_str = contract.get("expiration_date", "")
        if not expiry_str:
            continue

        try:
            exp_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        dte = (exp_date - today).days
        if dte < min_dte or dte > max_dte:
            continue

        strike = float(contract.get("strike_price", 0))
        if strike <= 0:
            continue

        symbol_map[occ_symbol] = {
            "strike": strike,
            "expiry": expiry_str,
            "dte": dte,
        }

    if not symbol_map:
        return []

    # Fetch snapshots in bulk for bid/ask/OI data
    occ_symbols = list(symbol_map.keys())
    snapshots = {}
    try:
        snapshots = options_client.get_option_snapshots_bulk(occ_symbols)
    except Exception as e:
        logger.warning("LEAP SELECT: bulk snapshot fetch failed: %s", e)

    # If bulk fetch returned nothing, try individual snapshots for a sample
    if not snapshots:
        logger.debug("LEAP SELECT: falling back to individual snapshots (sample of 20)")
        for sym in occ_symbols[:20]:
            try:
                snap = options_client.get_option_snapshot(sym)
                if snap:
                    snapshots[sym] = snap
            except Exception:
                pass
            time.sleep(0.05)

    # Process each contract with its snapshot data
    for occ_symbol, meta in symbol_map.items():
        strike = meta["strike"]
        expiry = meta["expiry"]
        dte = meta["dte"]
        T = dte / 365.0

        # Extract bid/ask from snapshot
        snap = snapshots.get(occ_symbol, {})
        quote = snap.get("latestQuote", snap.get("quote", {}))
        bid = float(quote.get("bp", quote.get("bid", 0)) or 0)
        ask = float(quote.get("ap", quote.get("ask", 0)) or 0)
        oi = int(snap.get("openInterest", quote.get("openInterest", 0)) or 0)

        # If we have greeks from the snapshot, use them directly
        greeks_data = snap.get("greeks", {})
        snapshot_delta = float(greeks_data.get("delta", 0) or 0)

        # Compute mid price
        if bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        elif bid > 0:
            mid = bid
        elif ask > 0:
            mid = ask
        else:
            # No quote data -- estimate from BS model
            mid = bs_call_price(underlying_price, strike, T, _DEFAULT_RISK_FREE_RATE, sigma)
            if mid <= 0:
                continue

        # Compute spread percentage
        if bid > 0 and ask > 0 and mid > 0:
            spread_pct = (ask - bid) / mid
        else:
            # No reliable spread data -- set to a high value that can be
            # overridden by the max_spread_pct filter if we have no quote
            spread_pct = 1.0

        # Filter: bid-ask spread
        if spread_pct > max_spread_pct and bid > 0 and ask > 0:
            logger.debug(
                "LEAP REJECT %s: spread=%.1f%% > max %.1f%% | $%.0f strike | bid=%.2f ask=%.2f",
                occ_symbol, spread_pct * 100, max_spread_pct * 100, strike, bid, ask,
            )
            continue

        # Estimate delta via Black-Scholes
        if snapshot_delta > 0:
            delta = snapshot_delta
        else:
            greeks = bs_greeks(underlying_price, strike, T, _DEFAULT_RISK_FREE_RATE, sigma)
            delta = greeks["delta"]

        # Filter: delta range
        if delta < min_delta or delta > max_delta:
            logger.debug(
                "LEAP REJECT %s: delta=%.3f outside [%.2f, %.2f] | $%.0f strike DTE=%d",
                occ_symbol, delta, min_delta, max_delta, strike, dte,
            )
            continue

        cost_per_contract = mid * 100.0

        candidate = LeapCandidate(
            symbol=occ_symbol,
            ticker=ticker,
            strike=strike,
            expiry=expiry,
            dte=dte,
            estimated_delta=round(delta, 4),
            bid=bid,
            ask=ask,
            mid_price=round(mid, 2),
            cost_per_contract=round(cost_per_contract, 2),
            open_interest=oi,
            spread_pct=round(spread_pct, 4),
        )
        candidates.append(candidate)

    return candidates


def _query_yfinance_chain(
    ticker: str,
    underlying_price: float,
    sigma: float,
    min_dte: int,
    max_dte: int,
    strike_gte: float,
    strike_lte: float,
    min_delta: float,
    max_delta: float,
    max_spread_pct: float,
) -> List[LeapCandidate]:
    """Fallback: query yfinance for LEAP call candidates."""
    candidates: List[LeapCandidate] = []
    today = _today_et()

    try:
        import yfinance as yf
    except ImportError:
        logger.error("LEAP SELECT: yfinance not installed -- cannot use fallback")
        return []

    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            logger.debug("LEAP SELECT: yfinance returned no expirations for %s", ticker)
            return []
    except Exception as e:
        logger.warning("LEAP SELECT: yfinance options fetch failed for %s: %s", ticker, e)
        return []

    for exp_str in expirations:
        try:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        dte = (exp_date - today).days
        if dte < min_dte or dte > max_dte:
            continue

        try:
            chain = t.option_chain(exp_str)
            calls = chain.calls
            if calls is None or len(calls) == 0:
                continue
        except Exception as e:
            logger.debug("LEAP SELECT: yfinance chain fetch failed for %s/%s: %s", ticker, exp_str, e)
            continue

        # Filter to strike range
        mask = (calls["strike"] >= strike_gte) & (calls["strike"] <= strike_lte)
        filtered = calls[mask]
        if len(filtered) == 0:
            continue

        T = dte / 365.0

        for _, row in filtered.iterrows():
            strike = float(row.get("strike", 0))
            bid = float(row.get("bid", 0) or 0)
            ask = float(row.get("ask", 0) or 0)
            oi = int(row.get("openInterest", 0) or 0)
            contract_symbol = str(row.get("contractSymbol", ""))

            # Mid price
            if bid > 0 and ask > 0:
                mid = (bid + ask) / 2.0
            else:
                last_price = float(row.get("lastPrice", 0) or 0)
                if last_price > 0:
                    mid = last_price
                else:
                    mid = bs_call_price(underlying_price, strike, T, _DEFAULT_RISK_FREE_RATE, sigma)
                    if mid <= 0:
                        continue

            # Spread percentage
            if bid > 0 and ask > 0 and mid > 0:
                spread_pct = (ask - bid) / mid
            else:
                spread_pct = 1.0

            # Filter: spread
            if spread_pct > max_spread_pct and bid > 0 and ask > 0:
                continue

            # Estimate delta
            greeks = bs_greeks(underlying_price, strike, T, _DEFAULT_RISK_FREE_RATE, sigma)
            delta = greeks["delta"]

            # Filter: delta
            if delta < min_delta or delta > max_delta:
                continue

            cost_per_contract = mid * 100.0

            candidate = LeapCandidate(
                symbol=contract_symbol,
                ticker=ticker,
                strike=strike,
                expiry=exp_str,
                dte=dte,
                estimated_delta=round(delta, 4),
                bid=bid,
                ask=ask,
                mid_price=round(mid, 2),
                cost_per_contract=round(cost_per_contract, 2),
                open_interest=oi,
                spread_pct=round(spread_pct, 4),
            )
            candidates.append(candidate)

    return candidates


def _score_candidate(candidate: LeapCandidate) -> float:
    """Score a LEAP candidate for ranking.  Higher score = better.

    Scoring criteria (weighted):
      a) Delta closest to 0.75 (ideal balance of ITM depth and capital efficiency)
      b) DTE closest to 270 days (sweet spot for time value vs. cost)
      c) Tightest bid-ask spread (better fills, less slippage)
      d) Highest open interest (liquidity, ability to exit)
    """
    score = 0.0

    # (a) Delta proximity to 0.75 -- weight 30
    #     Max penalty at the edges of the range (0.70 or 0.80)
    delta_distance = abs(candidate.estimated_delta - 0.75)
    score += max(0, 30.0 - delta_distance * 600.0)  # 0.05 distance -> 0 points

    # (b) DTE proximity to 270 days -- weight 25
    dte_distance = abs(candidate.dte - 270)
    score += max(0, 25.0 - dte_distance * 0.15)  # ~167 day distance -> 0 points

    # (c) Tightest spread -- weight 25
    #     0% spread = 25 points, 5% spread = 0 points
    if candidate.spread_pct >= 0:
        score += max(0, 25.0 - candidate.spread_pct * 500.0)

    # (d) Open interest -- weight 20
    #     Logarithmic scale: OI of 1 = 0 pts, OI of 1000+ = 20 pts
    if candidate.open_interest > 0:
        oi_score = min(20.0, math.log10(max(candidate.open_interest, 1)) * 6.67)
        score += oi_score

    return round(score, 2)


# =============================================================================
# LEAP PURCHASE EXECUTION
# =============================================================================

def execute_leap_purchase(
    candidate: LeapCandidate,
    options_client,  # AlpacaOptionsClient
    contracts: int = 1,
    dry_run: bool = True,
) -> Optional[dict]:
    """Execute a LEAP purchase order.

    Places a limit buy order at the mid-price via the Alpaca options API.
    In dry-run mode, logs what would happen and returns a mock result.

    Args:
        candidate: The LeapCandidate to purchase.
        options_client: An AlpacaOptionsClient instance.
        contracts: Number of contracts to buy (default 1).
        dry_run: If True, do not place a real order (default True).

    Returns:
        A dict with order details on success, or None on failure.
    """
    # For paper trading, use ask price to ensure fills on illiquid options.
    # Mid-price limit orders on 0-OI contracts often never fill in paper sim.
    if not dry_run and candidate.ask > 0:
        limit_price = candidate.ask
    else:
        limit_price = candidate.mid_price
    total_cost = limit_price * 100.0 * contracts

    logger.info(
        "LEAP PURCHASE %s: %s | %d contract(s) @ $%.2f (limit, %s) | "
        "total=$%.0f | $%.0f strike exp=%s DTE=%d delta=%.3f | dry_run=%s",
        "DRY-RUN" if dry_run else "LIVE",
        candidate.symbol, contracts, limit_price,
        "ask" if (not dry_run and candidate.ask > 0) else "mid",
        total_cost, candidate.strike, candidate.expiry,
        candidate.dte, candidate.estimated_delta, dry_run,
    )

    if dry_run:
        mock_result = {
            "id": f"dry-run-{int(time.time())}",
            "status": "dry_run",
            "symbol": candidate.symbol,
            "side": "buy",
            "qty": str(contracts),
            "type": "limit",
            "limit_price": f"{limit_price:.2f}",
            "filled_avg_price": None,
            "ticker": candidate.ticker,
            "strike": candidate.strike,
            "expiry": candidate.expiry,
            "dte": candidate.dte,
            "estimated_delta": candidate.estimated_delta,
            "cost_per_contract": candidate.cost_per_contract,
            "total_estimated_cost": total_cost,
            "dry_run": True,
        }
        logger.info(
            "LEAP PURCHASE DRY-RUN: would buy %d x %s @ $%.2f limit | "
            "estimated cost $%.0f",
            contracts, candidate.symbol, limit_price, total_cost,
        )
        return mock_result

    # ---- Live execution ----
    try:
        order_result = options_client.place_option_order(
            option_symbol=candidate.symbol,
            qty=contracts,
            side="buy",
            order_type="limit",
            limit_price=limit_price,
            time_in_force="day",
        )
    except Exception as e:
        logger.error("LEAP PURCHASE: order placement failed for %s: %s", candidate.symbol, e)
        return None

    if order_result is None:
        logger.error("LEAP PURCHASE: order rejected or failed for %s", candidate.symbol)
        return None

    order_id = order_result.get("id", "")
    logger.info(
        "LEAP PURCHASE: order submitted for %s | order_id=%s | waiting for fill...",
        candidate.symbol, order_id,
    )

    # ---- Wait for fill ----
    fill_timeout = 120  # seconds
    filled_order = None
    try:
        filled_order = options_client.wait_for_fill(order_id, timeout=fill_timeout)
    except Exception as e:
        logger.error("LEAP PURCHASE: error waiting for fill on %s: %s", order_id, e)

    if filled_order is None:
        logger.warning(
            "LEAP PURCHASE: order %s did not fill within %d seconds -- "
            "it was cancelled. Consider adjusting the limit price.",
            order_id, fill_timeout,
        )
        return None

    filled_price = float(filled_order.get("filled_avg_price", 0) or 0)
    filled_qty = int(filled_order.get("filled_qty", 0) or 0)

    logger.info(
        "LEAP PURCHASE FILLED: %s | %d contract(s) @ $%.2f | order_id=%s",
        candidate.symbol, filled_qty, filled_price, order_id,
    )

    return {
        "id": order_id,
        "status": "filled",
        "symbol": candidate.symbol,
        "side": "buy",
        "qty": str(filled_qty),
        "type": "limit",
        "limit_price": f"{limit_price:.2f}",
        "filled_avg_price": f"{filled_price:.2f}" if filled_price else None,
        "ticker": candidate.ticker,
        "strike": candidate.strike,
        "expiry": candidate.expiry,
        "dte": candidate.dte,
        "estimated_delta": candidate.estimated_delta,
        "cost_per_contract": filled_price * 100 if filled_price else candidate.cost_per_contract,
        "total_cost": filled_price * 100 * filled_qty if filled_price else total_cost,
        "dry_run": False,
    }


# =============================================================================
# CAPITAL SIZING CHECK
# =============================================================================

def check_leap_sizing(
    candidate: LeapCandidate,
    equity: float,
    existing_leap_capital: float = 0,
    max_per_leap_pct: float = 0.05,
    max_total_pct: float = 0.20,
) -> Tuple[bool, str]:
    """Check whether a LEAP purchase fits within capital allocation limits.

    Enforces two rules from the PMCC integration plan:
      1. A single LEAP's cost must not exceed max_per_leap_pct of equity
         (default 5%).
      2. Total LEAP allocation (existing + new) must not exceed max_total_pct
         of equity (default 20%).

    Args:
        candidate: The LeapCandidate being evaluated.
        equity: Current portfolio equity (total account value).
        existing_leap_capital: Capital already deployed in existing LEAPs.
        max_per_leap_pct: Maximum fraction of equity for a single LEAP.
        max_total_pct: Maximum fraction of equity for all LEAPs combined.

    Returns:
        A tuple of (allowed, reason). If allowed is True, the purchase is
        within limits. If False, reason explains why it was rejected.
    """
    cost = candidate.cost_per_contract

    if equity <= 0:
        return False, f"Invalid equity value: ${equity:.2f}"

    # Rule 1: Single LEAP cost check
    max_per_leap = equity * max_per_leap_pct
    if cost > max_per_leap:
        reason = (
            f"LEAP cost ${cost:,.0f} exceeds {max_per_leap_pct:.0%} of equity "
            f"(${max_per_leap:,.0f} max). "
            f"Equity=${equity:,.0f}, ticker={candidate.ticker}, "
            f"strike=${candidate.strike:.0f}"
        )
        logger.warning("LEAP SIZING REJECTED: %s", reason)
        return False, reason

    # Rule 2: Total allocation check
    max_total = equity * max_total_pct
    new_total = existing_leap_capital + cost
    if new_total > max_total:
        reason = (
            f"Total LEAP allocation ${new_total:,.0f} would exceed "
            f"{max_total_pct:.0%} of equity (${max_total:,.0f} max). "
            f"Existing=${existing_leap_capital:,.0f}, new=${cost:,.0f}, "
            f"equity=${equity:,.0f}"
        )
        logger.warning("LEAP SIZING REJECTED: %s", reason)
        return False, reason

    logger.info(
        "LEAP SIZING OK: %s $%.0f | cost=$%.0f (%.1f%% of equity) | "
        "total after=$%.0f (%.1f%% of equity, max %.0f%%)",
        candidate.ticker, candidate.strike, cost,
        (cost / equity) * 100,
        new_total, (new_total / equity) * 100,
        max_total_pct * 100,
    )

    return True, "OK"
