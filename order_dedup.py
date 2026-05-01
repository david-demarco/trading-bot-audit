"""
order_dedup.py - Unified order deduplication layer across all trading engines.

Prevents multiple engines (CC Scalper, PMCC Manager, Options Overlay) from
simultaneously submitting sell orders on the same underlying ticker.

Problem (from H2 audit):
  - CC Scalper has no duplicate sell prevention. Pending orders aren't counted
    toward position limits, so it could double-sell calls on the same ticker.
  - PMCC Manager and CC Scalper could both try to sell calls on the same
    underlying, creating overlapping short call exposure.

Solution:
  A single OrderDeduplicator instance is shared by all engines. Before any
  engine submits a sell order, it calls has_pending_or_active_sell(ticker)
  to check whether ANY engine already has a pending or active short call
  on that ticker.

Usage:
    dedup = OrderDeduplicator()
    # Register data sources
    dedup.register_cc_scalper(cc_scalper)
    dedup.register_pmcc_manager(pmcc_manager)
    dedup.register_options_overlay(options_overlay)

    # Before selling:
    if dedup.has_pending_or_active_sell("PAAS"):
        logger.info("DEDUP: skipping sell on PAAS -- another engine has it")
        return None
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

logger = logging.getLogger("order_dedup")


class OrderDeduplicator:
    """Unified order deduplication layer for all trading engines.

    Checks three sources before allowing a new sell order:
      1. CC Scalper's OrderManager pending orders + open positions
      2. PMCC Manager's active short legs + pending orders
      3. Legacy Options Overlay positions

    All checks are FAIL-CLOSED: if any check raises an exception,
    we assume a sell IS active to prevent naked short risk.
    """

    def __init__(self):
        self._cc_scalper = None          # CCScalper instance
        self._pmcc_manager = None        # PMCCManager instance
        self._options_overlay = None     # OptionsOverlay instance
        self._call_buyer = None          # Layer 4 CallBuyerManager instance
        self._pmcc_adapter_orders: Dict[str, str] = {}  # contract_symbol -> ticker for adapter-placed orders

    def register_cc_scalper(self, cc_scalper) -> None:
        """Register the CC Scalper engine for dedup checks."""
        self._cc_scalper = cc_scalper
        logger.info("OrderDeduplicator: CC Scalper registered")

    def register_pmcc_manager(self, pmcc_manager) -> None:
        """Register the PMCC Manager engine for dedup checks."""
        self._pmcc_manager = pmcc_manager
        logger.info("OrderDeduplicator: PMCC Manager registered")

    def register_options_overlay(self, options_overlay) -> None:
        """Register the legacy Options Overlay engine for dedup checks."""
        self._options_overlay = options_overlay
        logger.info("OrderDeduplicator: Options Overlay registered")

    def register_call_buyer(self, call_buyer) -> None:
        """Register the Layer 4 call buyer for dedup checks."""
        self._call_buyer = call_buyer
        logger.info("OrderDeduplicator: Layer 4 Call Buyer registered")

    def has_active_long_call(self, ticker: str) -> bool:
        """Return True if Layer 4 has an open long call on ticker.

        Used by Layer 3 to avoid selling calls on a ticker where Layer 4
        holds long calls (never be long AND short calls on the same ticker).

        FAIL-CLOSED: returns True on exception to prevent conflicting positions.
        """
        if self._call_buyer is None:
            return False
        try:
            return ticker in self._call_buyer.active_tickers()
        except Exception as e:
            logger.warning(
                "DEDUP FAIL-CLOSED: has_active_long_call(%s) exception -- "
                "assuming long call IS active: %s", ticker, e,
            )
            return True  # fail-closed

    def record_adapter_order(self, contract_symbol: str, ticker: str) -> None:
        """Record an order placed through the PMCC adapter (which has no OrderManager).

        The CombinedOrderManagerAdapter is synchronous and doesn't track pending
        orders. We record them here so cross-engine dedup can see them.
        """
        self._pmcc_adapter_orders[contract_symbol] = ticker
        logger.debug("OrderDeduplicator: recorded adapter order %s for %s",
                      contract_symbol, ticker)

    def clear_adapter_order(self, contract_symbol: str) -> None:
        """Clear an adapter order after it fills or aborts."""
        self._pmcc_adapter_orders.pop(contract_symbol, None)

    def has_pending_or_active_sell(self, ticker: str) -> bool:
        """Return True if ANY engine has a pending or active short call on ticker.

        This is the main dedup gate. Call this before submitting any sell order.

        FAIL-CLOSED: returns True on any exception to prevent duplicate sells.
        """
        try:
            # Source 1: CC Scalper - pending sell orders
            if self._cc_scalper:
                if self._check_cc_scalper_pending_sell(ticker):
                    logger.info(
                        "DEDUP BLOCK: CC Scalper has pending sell order for %s", ticker,
                    )
                    return True
                # CC Scalper - open (filled) positions
                if self._check_cc_scalper_open_position(ticker):
                    logger.info(
                        "DEDUP BLOCK: CC Scalper has open short call on %s", ticker,
                    )
                    return True

            # Source 2: PMCC Manager - active short legs
            if self._pmcc_manager:
                if self._check_pmcc_active_short(ticker):
                    logger.info(
                        "DEDUP BLOCK: PMCC Manager has active short leg on %s", ticker,
                    )
                    return True
                # PMCC Manager - pending sell orders (via OrderManager)
                if self._check_pmcc_pending_sell(ticker):
                    logger.info(
                        "DEDUP BLOCK: PMCC Manager has pending sell order for %s", ticker,
                    )
                    return True

            # Source 3: Legacy Options Overlay - open covered calls
            if self._options_overlay:
                if self._check_overlay_active_cc(ticker):
                    logger.info(
                        "DEDUP BLOCK: Options Overlay has active CC on %s", ticker,
                    )
                    return True

            # Source 4: Adapter-placed orders (combined_runner PMCC via adapter)
            if any(t == ticker for t in self._pmcc_adapter_orders.values()):
                logger.info(
                    "DEDUP BLOCK: PMCC adapter has in-flight order for %s", ticker,
                )
                return True

            # Source 5: Layer 4 Call Buyer - open long call positions.
            # If Layer 4 holds a long call, Layer 3 must NOT sell a short call
            # on the same ticker (never be long AND short calls simultaneously).
            if self._call_buyer:
                try:
                    if ticker in self._call_buyer.active_tickers():
                        logger.info(
                            "DEDUP BLOCK: Layer 4 Call Buyer has open long call on %s",
                            ticker,
                        )
                        return True
                except Exception:
                    return True  # fail-closed

            return False

        except Exception as e:
            logger.warning(
                "DEDUP FAIL-CLOSED: has_pending_or_active_sell(%s) exception -- "
                "assuming sell IS active to prevent duplicate: %s", ticker, e,
            )
            return True

    def has_conflicting_direction(self, ticker: str, proposed_side: str) -> bool:
        """Return True if there's an open Alpaca order on ticker in the OPPOSITE direction.

        Prevents simultaneous buy+sell on the same underlying, which brokers
        flag as wash trading or self-dealing. (David's Schwab account was flagged
        for this exact issue, Apr 6 2026.)

        Args:
            ticker: Underlying ticker (e.g., "KGC")
            proposed_side: "buy" or "sell" — the side we want to submit

        FAIL-CLOSED: returns True on exception to prevent conflicting orders.
        """
        opposite = "sell" if proposed_side == "buy" else "buy"
        try:
            # Check CC Scalper pending orders
            if self._cc_scalper:
                try:
                    for mo in self._cc_scalper.order_manager.get_pending_orders():
                        if mo.side == opposite and self._contract_matches_ticker(mo.contract_symbol, ticker):
                            logger.info(
                                "CONFLICT BLOCK: %s order on %s blocked — CC Scalper has pending %s",
                                proposed_side, ticker, opposite,
                            )
                            return True
                except Exception:
                    return True  # fail-closed

            # Check PMCC Manager pending orders
            if self._pmcc_manager:
                try:
                    if hasattr(self._pmcc_manager, 'order_manager'):
                        for mo in self._pmcc_manager.order_manager.get_pending_orders():
                            if mo.side == opposite and self._contract_matches_ticker(mo.contract_symbol, ticker):
                                logger.info(
                                    "CONFLICT BLOCK: %s order on %s blocked — PMCC has pending %s",
                                    proposed_side, ticker, opposite,
                                )
                                return True
                except Exception:
                    return True  # fail-closed

            # Check adapter in-flight orders (these are sells)
            if proposed_side == "buy" and any(t == ticker for t in self._pmcc_adapter_orders.values()):
                logger.info(
                    "CONFLICT BLOCK: buy on %s blocked — adapter has in-flight sell", ticker,
                )
                return True

            return False

        except Exception as e:
            logger.warning(
                "CONFLICT FAIL-CLOSED: has_conflicting_direction(%s, %s) exception: %s",
                ticker, proposed_side, e,
            )
            return True

    def has_any_pending_for_ticker(self, ticker: str) -> bool:
        """Return True if ANY engine has any pending order (sell or buy) for ticker.

        Broader check than has_pending_or_active_sell -- includes buy-backs.
        Used to prevent order spam on the same ticker.
        """
        try:
            # CC Scalper pending orders (any side)
            if self._cc_scalper:
                try:
                    for mo in self._cc_scalper.order_manager.get_pending_orders():
                        if self._contract_matches_ticker(mo.contract_symbol, ticker):
                            return True
                except Exception:
                    pass

            # PMCC Manager pending buy-back order IDs
            if self._pmcc_manager:
                try:
                    for spread in self._pmcc_manager.get_active_spreads():
                        if spread.ticker == ticker and spread.pending_buyback_order_id:
                            return True
                except Exception:
                    pass

            # Adapter in-flight orders
            if any(t == ticker for t in self._pmcc_adapter_orders.values()):
                return True

            return False

        except Exception as e:
            logger.warning(
                "DEDUP FAIL-CLOSED: has_any_pending_for_ticker(%s) exception: %s",
                ticker, e,
            )
            return True

    # ------------------------------------------------------------------
    # Internal check methods
    # ------------------------------------------------------------------

    def _check_cc_scalper_pending_sell(self, ticker: str) -> bool:
        """Check if CC Scalper's OrderManager has a pending SELL for ticker."""
        try:
            for mo in self._cc_scalper.order_manager.get_pending_orders():
                if mo.side == "sell" and self._contract_matches_ticker(mo.contract_symbol, ticker):
                    return True
        except Exception:
            return True  # fail-closed
        return False

    def _check_cc_scalper_open_position(self, ticker: str) -> bool:
        """Check if CC Scalper has a filled/open short call on ticker."""
        try:
            for pos in self._cc_scalper.state.open_positions():
                if pos.ticker == ticker and pos.status == "open":
                    return True
        except Exception:
            return True  # fail-closed
        return False

    def _check_pmcc_active_short(self, ticker: str) -> bool:
        """Check if PMCC Manager has an active short leg on ticker."""
        try:
            for spread in self._pmcc_manager.get_active_spreads():
                if spread.ticker == ticker and spread.short_leg is not None:
                    return True
        except Exception:
            return True  # fail-closed
        return False

    def _check_pmcc_pending_sell(self, ticker: str) -> bool:
        """Check if PMCC Manager has a pending sell via its OrderManager."""
        try:
            om = self._pmcc_manager.order_manager
            # Check if the order manager has a has_pending_for method
            if hasattr(om, 'get_pending_orders'):
                for mo in om.get_pending_orders():
                    if mo.side == "sell" and self._contract_matches_ticker(mo.contract_symbol, ticker):
                        return True
            # Also check pending_buyback_order_id on spreads (though this is a buy, not sell)
            # We skip buys here -- this method is specifically for sell dedup.
        except Exception:
            return True  # fail-closed
        return False

    def _check_overlay_active_cc(self, ticker: str) -> bool:
        """Check if Options Overlay has an open covered call on ticker."""
        try:
            for p in self._options_overlay.state.positions:
                if (p.underlying == ticker
                        and getattr(p, "strategy", "") == "covered_call"
                        and getattr(p, "status", "") == "open"):
                    return True
        except Exception:
            return True  # fail-closed
        return False

    @staticmethod
    def _contract_matches_ticker(contract_symbol: str, ticker: str) -> bool:
        """Check if an OCC option symbol corresponds to a given ticker.

        OCC format: TICKER + YYMMDD + C/P + strike*1000 (e.g., PAAS250620C00025000)
        The ticker is the alphabetic prefix before the first digit.
        """
        if not contract_symbol or not ticker:
            return False
        # Extract the alphabetic prefix from the contract symbol
        prefix = ""
        for ch in contract_symbol:
            if ch.isalpha():
                prefix += ch
            else:
                break
        return prefix.upper() == ticker.upper()

    def status_summary(self) -> str:
        """Return a human-readable summary of dedup state."""
        parts = ["OrderDeduplicator:"]

        if self._cc_scalper:
            try:
                pending_sells = sum(
                    1 for mo in self._cc_scalper.order_manager.get_pending_orders()
                    if mo.side == "sell"
                )
                open_pos = len(self._cc_scalper.state.open_positions())
                parts.append(f"  CC Scalper: {pending_sells} pending sells, {open_pos} open positions")
            except Exception:
                parts.append("  CC Scalper: error reading state")

        if self._pmcc_manager:
            try:
                active = self._pmcc_manager.get_active_spreads()
                with_short = sum(1 for s in active if s.short_leg is not None)
                parts.append(f"  PMCC: {len(active)} active spreads, {with_short} with short legs")
            except Exception:
                parts.append("  PMCC: error reading state")

        if self._options_overlay:
            parts.append("  Options Overlay: registered")

        adapter_count = len(self._pmcc_adapter_orders)
        if adapter_count:
            parts.append(f"  Adapter in-flight: {adapter_count}")

        if self._call_buyer:
            try:
                active = self._call_buyer.active_tickers()
                parts.append(f"  Layer 4 Call Buyer: {len(active)} open long calls")
            except Exception:
                parts.append("  Layer 4 Call Buyer: error reading state")

        return "\n".join(parts)
