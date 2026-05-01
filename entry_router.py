"""
entry_router.py - Intelligent LEAP vs Shares routing for RSI2 swing entries.

Replaces the old PMCC_MODE global boolean with per-signal, per-ticker decisions.
For each buy signal, the router evaluates:
  1. Is ticker options-eligible? (CC_OPTIONS_ELIGIBLE)
  2. Is a qualifying LEAP available? (delta, DTE, spread, sizing)
  3. Does the LEAP save >30% capital vs 100 shares?

If all three pass: route to LEAP entry.
Otherwise: route to 100-share entry.
Signals are NEVER silently dropped.

Author: Entry Router Module (Mar 16, 2026)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from combined_config import (
    CC_OPTIONS_ELIGIBLE,
    PMCC_ENABLED,
    PMCC_LONG_LEG_MIN_DTE,
    PMCC_LONG_LEG_MAX_DTE,
    PMCC_LONG_LEG_MIN_DELTA,
    PMCC_LONG_LEG_MAX_DELTA,
    PMCC_LEAP_MAX_SPREAD_PCT,
    PMCC_MAX_CONCURRENT_SPREADS,
    PMCC_MAX_LEAP_COST_PCT,
    PMCC_TOTAL_ALLOCATION_PCT,
    LEAP_CAPITAL_EFFICIENCY_THRESHOLD,
)
from leap_selector import LeapCandidate, select_leap, check_leap_sizing

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Result of the entry routing decision."""
    use_leap: bool                        # True = buy LEAP, False = buy shares
    ticker: str
    reason: str                           # Human-readable explanation
    leap_candidate: Optional[LeapCandidate] = None  # Set if use_leap is True
    leap_cost: float = 0.0               # LEAP cost per contract
    shares_cost: float = 0.0             # 100 shares cost
    saving_pct: float = 0.0             # Capital savings percentage


def route_entry(
    ticker: str,
    signal_price: float,
    equity: float,
    existing_leap_capital: float,
    active_spread_count: int,
    options_client=None,
) -> RoutingDecision:
    """Decide whether to buy a LEAP or 100 shares for a given RSI2 signal.

    This function NEVER returns a decision that would drop the signal.
    If LEAP is not viable, it falls back to shares.

    Args:
        ticker: The ticker symbol.
        signal_price: Current price of the underlying.
        equity: Current portfolio equity.
        existing_leap_capital: Capital already deployed in LEAPs.
        active_spread_count: Number of currently active PMCC spreads.
        options_client: AlpacaOptionsClient instance (lazy-created if None).

    Returns:
        RoutingDecision with use_leap=True/False and explanation.
    """
    shares_cost = signal_price * 100.0

    # --- Gate 1: Is PMCC subsystem enabled? ---
    if not PMCC_ENABLED:
        reason = f"{ticker}: PMCC disabled globally -> buying 100 shares"
        logger.info("ROUTE: %s", reason)
        return RoutingDecision(
            use_leap=False, ticker=ticker, reason=reason,
            shares_cost=shares_cost,
        )

    # --- Gate 2: Is ticker options-eligible? ---
    if ticker not in CC_OPTIONS_ELIGIBLE:
        reason = (
            f"{ticker}: not in CC_OPTIONS_ELIGIBLE -> buying 100 shares "
            f"(cost ${shares_cost:,.0f})"
        )
        logger.info("ROUTE: %s", reason)
        return RoutingDecision(
            use_leap=False, ticker=ticker, reason=reason,
            shares_cost=shares_cost,
        )

    # --- Gate 3: Spread count limit ---
    if active_spread_count >= PMCC_MAX_CONCURRENT_SPREADS:
        reason = (
            f"{ticker}: max PMCC spreads ({PMCC_MAX_CONCURRENT_SPREADS}) reached "
            f"-> buying 100 shares (cost ${shares_cost:,.0f})"
        )
        logger.info("ROUTE: %s", reason)
        return RoutingDecision(
            use_leap=False, ticker=ticker, reason=reason,
            shares_cost=shares_cost,
        )

    # --- Gate 4: Attempt LEAP lookup ---
    if options_client is None:
        try:
            from options_overlay import AlpacaOptionsClient
            options_client = AlpacaOptionsClient()
        except Exception as e:
            reason = (
                f"{ticker}: cannot create options client ({e}) "
                f"-> buying 100 shares (cost ${shares_cost:,.0f})"
            )
            logger.warning("ROUTE: %s", reason)
            return RoutingDecision(
                use_leap=False, ticker=ticker, reason=reason,
                shares_cost=shares_cost,
            )

    try:
        candidate = select_leap(
            ticker, options_client,
            min_dte=PMCC_LONG_LEG_MIN_DTE,
            max_dte=PMCC_LONG_LEG_MAX_DTE,
            min_delta=PMCC_LONG_LEG_MIN_DELTA,
            max_delta=PMCC_LONG_LEG_MAX_DELTA,
            max_spread_pct=PMCC_LEAP_MAX_SPREAD_PCT,
        )
    except Exception as e:
        reason = (
            f"{ticker}: LEAP selection error ({e}) "
            f"-> buying 100 shares (cost ${shares_cost:,.0f})"
        )
        logger.warning("ROUTE: %s", reason)
        return RoutingDecision(
            use_leap=False, ticker=ticker, reason=reason,
            shares_cost=shares_cost,
        )

    if candidate is None:
        reason = (
            f"{ticker}: no qualifying LEAP found "
            f"-> buying 100 shares (cost ${shares_cost:,.0f})"
        )
        logger.info("ROUTE: %s", reason)
        return RoutingDecision(
            use_leap=False, ticker=ticker, reason=reason,
            shares_cost=shares_cost,
        )

    # --- Gate 5: Sizing check ---
    allowed, sizing_reason = check_leap_sizing(
        candidate, equity, existing_leap_capital,
        max_per_leap_pct=PMCC_MAX_LEAP_COST_PCT,
        max_total_pct=PMCC_TOTAL_ALLOCATION_PCT,
    )
    if not allowed:
        reason = (
            f"{ticker}: LEAP sizing rejected ({sizing_reason}) "
            f"-> buying 100 shares (cost ${shares_cost:,.0f})"
        )
        logger.info("ROUTE: %s", reason)
        return RoutingDecision(
            use_leap=False, ticker=ticker, reason=reason,
            shares_cost=shares_cost,
        )

    # --- Gate 6: Capital efficiency comparison ---
    leap_cost = candidate.cost_per_contract
    if shares_cost > 0:
        saving_pct = (shares_cost - leap_cost) / shares_cost
    else:
        saving_pct = 0.0

    if saving_pct <= LEAP_CAPITAL_EFFICIENCY_THRESHOLD:
        reason = (
            f"{ticker}: LEAP saves only {saving_pct:.0%} "
            f"(${leap_cost:,.0f} vs ${shares_cost:,.0f} shares, "
            f"threshold {LEAP_CAPITAL_EFFICIENCY_THRESHOLD:.0%}) "
            f"-> buying 100 shares"
        )
        logger.info("ROUTE: %s", reason)
        return RoutingDecision(
            use_leap=False, ticker=ticker, reason=reason,
            leap_cost=leap_cost, shares_cost=shares_cost,
            saving_pct=saving_pct,
        )

    # --- All gates passed: prefer LEAP ---
    reason = (
        f"{ticker}: LEAP preferred (cost ${leap_cost:,.0f} vs "
        f"${shares_cost:,.0f} shares, saving {saving_pct:.0%}) "
        f"-> buying LEAP {candidate.symbol}"
    )
    logger.info("ROUTE: %s", reason)
    return RoutingDecision(
        use_leap=True, ticker=ticker, reason=reason,
        leap_candidate=candidate,
        leap_cost=leap_cost, shares_cost=shares_cost,
        saving_pct=saving_pct,
    )
