"""
pmcc_manager.py - Poor Man's Covered Call (PMCC) / diagonal spread manager.

Manages diagonal spreads where the long leg is a deep ITM LEAP call and the
short leg is a near-term OTM call sold for recurring premium.

Key insight: "You don't have to have shares if you have LEAPS" -- David.
This lets the bot sell covered calls without owning 100 shares per contract,
using LEAPS as the collateral instead.

IMPORTANT super-cycle considerations:
  - Assignment on PMCC is MORE expensive than on shares because it burns
    LEAP time value (extrinsic).  The bot must exercise the LEAP or close
    it at a loss, forfeiting months of remaining time value.
  - Therefore the short leg must be even MORE conservative: lower delta,
    wider OTM than regular covered calls.
  - In a rally the long LEAP appreciates (good), but if the short leg
    gets tested, rolling costs increase because the short call gains
    intrinsic value faster than the LEAP (LEAP delta < 1.0).
  - Net position delta is lower with PMCC (LEAP delta 0.70-0.85 vs 1.0
    for shares), meaning less directional exposure and less upside cap.

Architecture:
  - DiagonalSpread dataclass tracks each spread's full state.
  - PMCCManager orchestrates detection, evaluation, execution, and risk.
  - Integrates with the existing DataLayer, SignalEngine, and OrderManager.
  - Persists state alongside regular CC positions in the same state file.

Usage:
  Instantiated by CCScalper and called during the run_once() cycle as
  Phase 0.5 (between order management and buy-back evaluation).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pytz

if TYPE_CHECKING:
    from order_manager import OrderManager, ManagedOrder

_ET = pytz.timezone("US/Eastern")


def _today_et() -> date:
    """Return the current calendar date in US/Eastern time.

    Using _today_et() returns the server's local date (UTC on most cloud
    hosts), which differs from the ET trading date after 8 PM UTC / before
    midnight ET.  All DTE calculations must use the ET date to avoid
    off-by-one errors on DTE and incorrect 'new day' comparisons.
    """
    return datetime.now(_ET).date()

from slvr_cc_config import (
    PMCC_ENABLED,
    PMCC_LONG_LEG_MIN_DTE,
    PMCC_LONG_LEG_MIN_DELTA,
    PMCC_LONG_LEG_MAX_DELTA,
    PMCC_MAX_SHORT_DELTA,
    PMCC_SHORT_MIN_DELTA,
    PMCC_SHORT_REQUIRE_LIVE_QUOTE,
    PMCC_ASSIGNMENT_BUFFER_PCT,
    PMCC_MAX_RISK_RATIO,
    PMCC_MIN_NET_CREDIT_TARGET,
    PMCC_SHORT_DELTA_WARN,
    PMCC_SHORT_DELTA_DANGER,
    PMCC_SHORT_DTE_MIN,
    PMCC_SHORT_DTE_MAX,
    PMCC_SHORT_DTE_OPTIMAL,
    PMCC_MAX_CONCURRENT_SPREADS,
    ALPACA_BASE_URL,
    RISK_FREE_RATE,
    SELL_OFFSET_FROM_MID,
    BUYBACK_OFFSET_FROM_MID,
    PMCC_SHORT_PROFIT_TARGET_PCT,
    PMCC_AUTO_RESELL,
    PMCC_BUYBACK_INITIAL_WAIT_MINUTES,
    PMCC_BUYBACK_MID_WAIT_MINUTES,
    PMCC_BUYBACK_MAX_PCT_OF_SOLD,
    MIN_SELL_SIGNALS,
    MIN_OPEN_INTEREST,
    MAX_BID_ASK_SPREAD_PCT,
    # v2 buyback params (Mar 19)
    PMCC_PROFIT_TARGET_EARLY,
    PMCC_PROFIT_TARGET_LATE,
    PMCC_CLOSE_DTE,
    PMCC_LET_EXPIRE_DTE,
    PMCC_LET_EXPIRE_DELTA,
    PMCC_LET_EXPIRE_VALUE,
    PMCC_BUYBACK_MAX_FIRST_HALF,
    PMCC_BUYBACK_MAX_SECOND_HALF,
    PMCC_CRASH_MODE_VIX,
    BREADTH_GATE_ENABLED,
    # Cheap BTC stepping params (Mar 25)
    PMCC_BTC_CHEAP_ENABLED,
    PMCC_BTC_CHEAP_MAX_DELTA,
    PMCC_BTC_CHEAP_MAX_VALUE,
    PMCC_BTC_CHEAP_START_PRICE,
    PMCC_BTC_CHEAP_STEP,
    PMCC_BTC_STEP_INTERVAL_SEC,
    PMCC_BTC_STEP_CAP_PCT,
)

logger = logging.getLogger("cc_scalper.pmcc")

# --- Edge 107 PR2 — feature flag: route leg.contracts into order placements ---
# False  = legacy behavior (always submit 1 contract per order). SAFE.
# True   = pull from spread.long_leg.contracts / .short_leg.contracts. Flip
#          ONLY after one paper trading day with no state-divergence alerts.
USE_LEG_CONTRACTS_IN_ORDERS = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LegInfo:
    """Describes one leg of a diagonal spread (either long LEAP or short call).

    All price fields are per-share (not per-contract).
    """
    symbol: str              # OCC option symbol, e.g. "SIL270115C00060000"
    strike: float            # strike price
    expiry: str              # YYYY-MM-DD expiration date
    delta: float             # BS delta at the time of entry or last update
    cost_basis: float        # per-share cost (what we paid for long / received for short)
    current_value: float     # per-share current market value (mid-price)
    contracts: int = 1       # Edge 107 — number of OCC contracts (default 1 for backwards compat).
                             # Backfilled from Alpaca on startup via PMCCManager.reconcile_contracts().

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LegInfo":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DiagonalSpread:
    """Tracks a single PMCC / diagonal spread position.

    A diagonal spread consists of:
      - long_leg: deep ITM LEAP call (the "covered" collateral)
      - short_leg: near-term OTM call sold for premium (nullable when idle)

    The spread generates income by repeatedly selling and buying back the
    short leg against the same long LEAP, just like selling covered calls
    against shares but with the LEAP as the underlying position.
    """
    ticker: str                        # underlying ETF symbol, e.g. "SIL"
    spread_id: str                     # unique identifier for this spread
    long_leg: LegInfo                  # the LEAP (always present)
    short_leg: Optional[LegInfo]       # the sold call (None when no short active)
    net_greeks: Dict[str, float] = field(default_factory=lambda: {
        "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0,
    })
    max_loss: float = 0.0              # worst case: LEAP cost - all credits received
    total_credits_received: float = 0.0  # cumulative short premium collected (per-share)
    num_short_cycles: int = 0          # how many times we have sold a short leg
    status: str = "active"             # active, idle_after_profit, short_expired_worthless, assigned, closed
    pending_buyback_order_id: Optional[str] = None  # GTC buy-to-close order ID (if working)
    pending_buyback_limit: float = 0.0              # limit price of pending GTC order
    pending_buyback_submitted_at: Optional[str] = None  # ISO timestamp when buy-back order was submitted
    short_leg_original_dte: int = 0    # DTE of short leg when first sold (for first/second half calc)
    # Cheap BTC stepping state (Mar 25)
    btc_step_start_ts: Optional[float] = None      # epoch timestamp when current stepping sequence began
    btc_step_last_price: float = 0.0               # last limit price used for cheap BTC stepping
    created_at: str = ""               # ISO timestamp of registration
    last_updated: str = ""             # ISO timestamp of last state change

    @property
    def long_leg_dte(self) -> int:
        """Current DTE for the long LEAP."""
        try:
            exp = datetime.strptime(self.long_leg.expiry, "%Y-%m-%d").date()
            return (exp - _today_et()).days
        except (ValueError, TypeError):
            return 0

    @property
    def short_leg_dte(self) -> int:
        """Current DTE for the short leg (0 if no short leg active)."""
        if self.short_leg is None:
            return 0
        try:
            exp = datetime.strptime(self.short_leg.expiry, "%Y-%m-%d").date()
            return (exp - _today_et()).days
        except (ValueError, TypeError):
            return 0

    @property
    def has_short_leg(self) -> bool:
        """True if there is an active short call against this LEAP."""
        return self.short_leg is not None

    @property
    def net_delta(self) -> float:
        """Net delta of the spread (long delta - short delta)."""
        return self.net_greeks.get("delta", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # asdict handles nested dataclasses, but Optional[LegInfo] needs care
        if self.short_leg is not None:
            d["short_leg"] = self.short_leg.to_dict()
        else:
            d["short_leg"] = None
        d["long_leg"] = self.long_leg.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DiagonalSpread":
        mapped = dict(d)
        if mapped.get("long_leg") and isinstance(mapped["long_leg"], dict):
            mapped["long_leg"] = LegInfo.from_dict(mapped["long_leg"])
        if mapped.get("short_leg") and isinstance(mapped["short_leg"], dict):
            mapped["short_leg"] = LegInfo.from_dict(mapped["short_leg"])
        elif mapped.get("short_leg") is None:
            mapped["short_leg"] = None
        return cls(**{k: v for k, v in mapped.items() if k in cls.__dataclass_fields__})


# =============================================================================
# HELPER FUNCTIONS — Black-Scholes Greeks for spread-level calculations
# =============================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
) -> Dict[str, float]:
    """Compute delta, gamma, theta (per calendar day), and vega (per 1% IV)
    for a European call option via Black-Scholes.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free rate (annualized).
        sigma: Implied volatility (annualized, as decimal).

    Returns:
        Dict with keys: delta, gamma, theta, vega.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        delta = 1.0 if S > K else 0.0
        return {"delta": delta, "gamma": 0.0, "theta": 0.0, "vega": 0.0}

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    delta = _norm_cdf(d1)
    gamma = _norm_pdf(d1) / (S * sigma * sqrt_T)
    theta_ann = (
        -(S * _norm_pdf(d1) * sigma) / (2 * sqrt_T)
        - r * K * math.exp(-r * T) * _norm_cdf(d2)
    )
    theta = theta_ann / 365.0  # per calendar day
    vega = S * _norm_pdf(d1) * sqrt_T / 100.0  # per 1% IV move

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}


def _bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def _implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float,
    max_iter: int = 100, tol: float = 1e-6,
) -> float:
    """Newton-Raphson IV solver. Returns annualized IV as a decimal."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.50
    sigma = 0.50
    for _ in range(max_iter):
        price = _bs_call_price(S, K, T, r, sigma)
        vega_full = _bs_greeks(S, K, T, r, sigma)["vega"] * 100.0
        if abs(vega_full) < 1e-10:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega_full
        sigma = max(sigma, 0.01)
        sigma = min(sigma, 5.0)
    return sigma


# =============================================================================
# PMCC MANAGER
# =============================================================================

class PMCCManager:
    """Manages Poor Man's Covered Call (diagonal spread) positions.

    Responsibilities:
      - Register existing LEAP positions (manual or auto-detected).
      - Evaluate when to sell a new short leg against an idle LEAP.
      - Monitor active short legs for assignment risk.
      - Roll short legs up and out when they approach danger zones.
      - Close entire spreads when needed.
      - Persist spread state for restart recovery.

    The manager works alongside regular CC selling -- they are complementary
    strategies.  A ticker can have both regular CCs (against shares) and
    PMCC spreads (against LEAPS) simultaneously.

    Args:
        api: Reference to the ExecutionLayer for Alpaca REST calls.
        data_layer: Shared DataLayer for market data and option chains.
        order_manager: Shared OrderManager for active order management.
        signal_engine: Shared SignalEngine for sell signal evaluation.
        config: Not used directly (we import from slvr_cc_config), but
                reserved for future per-instance overrides.
    """

    def __init__(
        self,
        api: Any,
        data_layer: Any,
        order_manager: "OrderManager",
        signal_engine: Any,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.api = api
        self.data = data_layer
        self.order_manager = order_manager
        self.signals = signal_engine
        self._config = config or {}
        self._spreads: List[DiagonalSpread] = []
        self._spread_counter: int = 0  # for generating unique IDs
        # Cross-engine order deduplication layer (set by combined_runner or standalone)
        self._order_dedup = None
        # Crash mode state — set externally by combined_runner each cycle
        self._crash_mode: bool = False
        self._vix_level: Optional[float] = None
        self._breadth_gate_active: bool = False

    # ------------------------------------------------------------------
    # Crash Mode Detection
    # ------------------------------------------------------------------

    def update_market_state(
        self,
        vix_level: Optional[float] = None,
        breadth_gate_active: bool = False,
    ) -> None:
        """Update market state for crash mode detection.

        Called by combined_runner each cycle to pass in VIX and breadth gate
        status.  PMCCManager uses this to decide whether to aggressively
        close all short calls (crash protocol).

        Args:
            vix_level: Current VIX level (None if unavailable).
            breadth_gate_active: True if the sector breadth gate has triggered.
        """
        self._vix_level = vix_level
        self._breadth_gate_active = breadth_gate_active
        old_crash = self._crash_mode
        self._crash_mode = self._detect_crash_mode()
        if self._crash_mode and not old_crash:
            logger.warning(
                "PMCC CRASH MODE ACTIVATED: VIX=%.1f breadth_gate=%s | "
                "will close all short calls aggressively",
                vix_level or 0.0, breadth_gate_active,
            )
        elif not self._crash_mode and old_crash:
            logger.info(
                "PMCC CRASH MODE DEACTIVATED: VIX=%.1f breadth_gate=%s",
                vix_level or 0.0, breadth_gate_active,
            )

    def _detect_crash_mode(self) -> bool:
        """Return True if conditions warrant crash-mode buyback behavior.

        Crash mode fires when:
          - VIX > PMCC_CRASH_MODE_VIX (30), OR
          - Sector breadth gate is active (>70% of universe oversold)
        """
        if self._vix_level is not None and self._vix_level > PMCC_CRASH_MODE_VIX:
            return True
        if BREADTH_GATE_ENABLED and self._breadth_gate_active:
            return True
        return False

    # ------------------------------------------------------------------
    # LEAP Registration
    # ------------------------------------------------------------------

    def register_leap(
        self,
        ticker: str,
        contract_symbol: str,
        cost_basis: float,
        strike: Optional[float] = None,
        expiry: Optional[str] = None,
        delta: Optional[float] = None,
        contracts: int = 1,
    ) -> DiagonalSpread:
        """Manually register an existing LEAP position as the long leg of
        a new diagonal spread.

        Use this for positions like Tom's Jan '27 SIL LEAPS that were
        purchased outside of this bot.

        Args:
            ticker: Underlying ETF symbol (e.g. "SIL").
            contract_symbol: Full OCC option symbol (e.g. "SIL270115C00060000").
            cost_basis: Per-share price paid for the LEAP.
            strike: Strike price (auto-parsed from symbol if omitted).
            expiry: Expiration date YYYY-MM-DD (auto-parsed from symbol if omitted).
            delta: Current delta (estimated via BS if omitted).

        Returns:
            The newly created DiagonalSpread.
        """
        # Parse strike and expiry from OCC symbol if not provided
        if strike is None or expiry is None:
            parsed_strike, parsed_expiry = self._parse_occ_symbol(contract_symbol)
            if strike is None:
                strike = parsed_strike
            if expiry is None:
                expiry = parsed_expiry

        # Validate DTE
        try:
            exp_date = datetime.strptime(expiry, "%Y-%m-%d").date()
            dte = (exp_date - _today_et()).days
        except (ValueError, TypeError):
            dte = 0

        if dte < PMCC_LONG_LEG_MIN_DTE:
            logger.warning(
                "PMCC register_leap: %s has only %d DTE (min=%d) -- registering "
                "anyway but flag for monitoring",
                contract_symbol, dte, PMCC_LONG_LEG_MIN_DTE,
            )

        # Estimate delta if not provided
        if delta is None:
            underlying_price = self.data.get_price(ticker)
            if underlying_price and underlying_price > 0 and dte > 0:
                T = dte / 365.0
                # Use HV-20 as a rough IV estimate with a 10% premium
                hv20 = self.data.compute_hv(ticker, 20)
                sigma = (hv20 * 1.10) if hv20 and hv20 > 0 else 0.75
                greeks = _bs_greeks(underlying_price, strike, T, RISK_FREE_RATE, sigma)
                delta = greeks["delta"]
            else:
                delta = 0.80  # conservative default for a deep ITM LEAP

        # Estimate current value
        current_value = cost_basis  # default to cost basis until we can price it
        underlying_price = self.data.get_price(ticker)
        if underlying_price and underlying_price > 0 and dte > 0:
            T = dte / 365.0
            hv20 = self.data.compute_hv(ticker, 20)
            sigma = (hv20 * 1.10) if hv20 and hv20 > 0 else 0.75
            model_price = _bs_call_price(underlying_price, strike, T, RISK_FREE_RATE, sigma)
            if model_price > 0:
                current_value = model_price

        self._spread_counter += 1
        spread_id = f"PMCC_{ticker}_{self._spread_counter}_{int(time.time())}"

        long_leg = LegInfo(
            symbol=contract_symbol,
            strike=strike,
            expiry=expiry,
            delta=delta,
            cost_basis=cost_basis,
            current_value=current_value,
            contracts=contracts,  # Edge 107 PR1b — from new register_leap arg
        )

        spread = DiagonalSpread(
            ticker=ticker,
            spread_id=spread_id,
            long_leg=long_leg,
            short_leg=None,
            max_loss=cost_basis * 100,  # per contract, worst case is losing the LEAP
            status="active",
            created_at=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        # Calculate initial greeks
        self.calculate_net_greeks(spread)
        self.calculate_max_loss(spread)

        self._spreads.append(spread)

        logger.info(
            "PMCC LEAP REGISTERED: %s | %s $%.0f exp=%s | cost=$%.2f/sh | "
            "delta=%.2f | DTE=%d | spread_id=%s",
            ticker, contract_symbol, strike, expiry,
            cost_basis, delta, dte, spread_id,
        )

        return spread

    def auto_detect_leaps(self) -> List[DiagonalSpread]:
        """Scan Alpaca positions for LEAP calls that qualify as PMCC long legs.

        Looks for long call positions where:
          - DTE > PMCC_LONG_LEG_MIN_DTE
          - Estimated delta is between PMCC_LONG_LEG_MIN_DELTA and MAX_DELTA
          - The position is not already registered as a spread

        Returns:
            List of newly registered DiagonalSpreads.
        """
        if self.api.dry_run:
            logger.info(
                "PMCC auto_detect_leaps: dry-run mode -- cannot scan Alpaca "
                "positions, use register_leap() for manual registration"
            )
            return []

        new_spreads: List[DiagonalSpread] = []

        try:
            session = self.api._get_session()
            resp = session.get(f"{ALPACA_BASE_URL}/v2/positions", timeout=15)
            resp.raise_for_status()
            positions = resp.json()
        except Exception as e:
            logger.error("PMCC auto_detect_leaps: failed to fetch positions: %s", e)
            return []

        # Track symbols already registered to avoid duplicates
        registered_symbols = {s.long_leg.symbol for s in self._spreads}

        today = _today_et()

        for pos in positions:
            # Alpaca option positions have asset_class = "us_option"
            if pos.get("asset_class") != "us_option":
                continue

            symbol = pos.get("symbol", "")
            side = pos.get("side", "")
            qty = int(pos.get("qty", 0))

            # We only care about long calls
            if side != "long" or qty <= 0:
                continue

            # Must be a call (symbol contains 'C' between date and strike)
            if "C" not in symbol:
                continue

            # Skip if already registered
            if symbol in registered_symbols:
                continue

            # Parse the OCC symbol to get strike and expiry
            try:
                parsed_strike, parsed_expiry = self._parse_occ_symbol(symbol)
            except (ValueError, IndexError):
                logger.debug("PMCC auto_detect: could not parse symbol %s", symbol)
                continue

            # Check DTE
            try:
                exp_date = datetime.strptime(parsed_expiry, "%Y-%m-%d").date()
                dte = (exp_date - today).days
            except (ValueError, TypeError):
                continue

            if dte < PMCC_LONG_LEG_MIN_DTE:
                logger.debug(
                    "PMCC auto_detect: %s has %d DTE < %d minimum -- skipping",
                    symbol, dte, PMCC_LONG_LEG_MIN_DTE,
                )
                continue

            # Extract ticker from position
            ticker = pos.get("symbol_root", "")
            if not ticker:
                # Parse from OCC symbol prefix (letters before digits)
                for i, ch in enumerate(symbol):
                    if ch.isdigit():
                        ticker = symbol[:i]
                        break

            if not ticker:
                continue

            # Estimate delta
            underlying_price = self.data.get_price(ticker)
            delta = 0.0
            if underlying_price and underlying_price > 0 and dte > 0:
                T = dte / 365.0
                hv20 = self.data.compute_hv(ticker, 20)
                sigma = (hv20 * 1.10) if hv20 and hv20 > 0 else 0.75
                greeks = _bs_greeks(underlying_price, parsed_strike, T, RISK_FREE_RATE, sigma)
                delta = greeks["delta"]

            # Check if delta qualifies as deep ITM
            if delta < PMCC_LONG_LEG_MIN_DELTA:
                logger.debug(
                    "PMCC auto_detect: %s delta=%.2f < %.2f minimum -- skipping",
                    symbol, delta, PMCC_LONG_LEG_MIN_DELTA,
                )
                continue

            if delta > PMCC_LONG_LEG_MAX_DELTA:
                logger.debug(
                    "PMCC auto_detect: %s delta=%.2f > %.2f maximum -- skipping "
                    "(too deep ITM, consider shares instead)",
                    symbol, delta, PMCC_LONG_LEG_MAX_DELTA,
                )
                continue

            # Get cost basis from Alpaca position
            cost_basis = float(pos.get("avg_entry_price", 0))
            if cost_basis <= 0:
                cost_basis = float(pos.get("cost_basis", 0)) / (qty * 100)

            # Register the LEAP
            spread = self.register_leap(
                ticker=ticker,
                contract_symbol=symbol,
                cost_basis=cost_basis,
                strike=parsed_strike,
                expiry=parsed_expiry,
                delta=delta,
            )
            new_spreads.append(spread)

            logger.info(
                "PMCC AUTO-DETECTED: %s | %s $%.0f exp=%s | delta=%.2f | "
                "DTE=%d | cost=$%.2f",
                ticker, symbol, parsed_strike, parsed_expiry,
                delta, dte, cost_basis,
            )

        if new_spreads:
            logger.info(
                "PMCC auto_detect_leaps: found %d new qualifying LEAP(s)",
                len(new_spreads),
            )
        else:
            logger.info("PMCC auto_detect_leaps: no new qualifying LEAPs found")

        # Phase 2: Detect existing short call positions and link them to
        # the registered spreads.  Without this, the manager would try to
        # sell new short calls even when we already have them on Alpaca.
        self._link_existing_short_calls(positions, new_spreads)

        return new_spreads

    def _link_existing_short_calls(
        self,
        alpaca_positions: list,
        spreads: List[DiagonalSpread],
    ) -> None:
        """Scan Alpaca positions for short calls that belong to registered spreads.

        If a short call position matches a spread's ticker, populate the
        spread's short_leg so run_cycle() doesn't try to sell again.
        """
        # Build ticker → spread mapping
        ticker_spread: Dict[str, DiagonalSpread] = {}
        for sp in spreads:
            if not sp.has_short_leg and sp.status == "active":
                ticker_spread[sp.ticker.upper()] = sp

        if not ticker_spread:
            return

        today = _today_et()

        for pos in alpaca_positions:
            if pos.get("asset_class") != "us_option":
                continue
            side = pos.get("side", "")
            qty = int(pos.get("qty", 0))
            symbol = pos.get("symbol", "")

            # We only care about short calls
            if side != "short" or qty >= 0:
                continue
            if "C" not in symbol:
                continue

            # Parse ticker from OCC symbol
            ticker = ""
            for i, ch in enumerate(symbol):
                if ch.isdigit():
                    ticker = symbol[:i]
                    break
            if not ticker:
                continue

            ticker = ticker.upper()
            if ticker not in ticker_spread:
                continue

            spread = ticker_spread[ticker]

            # Parse strike and expiry
            try:
                parsed_strike, parsed_expiry = self._parse_occ_symbol(symbol)
            except (ValueError, IndexError):
                continue

            # Get entry price
            cost_basis = abs(float(pos.get("avg_entry_price", 0)))

            # Estimate delta
            underlying_price = self.data.get_price(ticker)
            delta = 0.0
            try:
                exp_date = datetime.strptime(parsed_expiry, "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if underlying_price and underlying_price > 0 and dte > 0:
                    T = dte / 365.0
                    hv20 = self.data.compute_hv(ticker, 20)
                    sigma = (hv20 * 1.10) if hv20 and hv20 > 0 else 0.75
                    greeks = _bs_greeks(underlying_price, parsed_strike, T,
                                        RISK_FREE_RATE, sigma)
                    delta = greeks["delta"]
            except (ValueError, TypeError):
                dte = 0

            short_leg = LegInfo(
                symbol=symbol,
                strike=parsed_strike,
                expiry=parsed_expiry,
                delta=delta,
                cost_basis=cost_basis,
                current_value=cost_basis,
                contracts=abs(qty),  # Edge 107 PR1b — from Alpaca position iter
            )
            spread.short_leg = short_leg
            # Set original DTE if not already tracked (e.g. first detection)
            if spread.short_leg_original_dte == 0:
                spread.short_leg_original_dte = max(dte, 1)
            spread.total_credits_received += cost_basis
            spread.num_short_cycles = max(spread.num_short_cycles, 1)
            spread.last_updated = datetime.now(timezone.utc).isoformat()

            # Recalculate greeks and risk
            self.calculate_net_greeks(spread)
            self.calculate_max_loss(spread)

            logger.info(
                "PMCC SHORT LEG LINKED: %s | %s $%.0f exp=%s | "
                "delta=%.3f | premium=$%.2f | original_dte=%d | spread=%s",
                ticker, symbol, parsed_strike, parsed_expiry,
                delta, cost_basis, spread.short_leg_original_dte,
                spread.spread_id,
            )

            # Remove from candidates so we don't match again
            del ticker_spread[ticker]
            if not ticker_spread:
                break

    # ------------------------------------------------------------------
    # Short Leg Evaluation and Execution
    # ------------------------------------------------------------------

    def evaluate_short_leg(self, spread: DiagonalSpread) -> Optional[Dict[str, Any]]:
        """Evaluate whether to sell a new short leg against an active spread.

        Uses the same SignalEngine signals as regular CC selling but applies
        tighter PMCC-specific constraints:
          - Short delta must be < PMCC_MAX_SHORT_DELTA
          - Short strike must be above the LEAP strike
          - Premium / LEAP cost ratio check (PMCC_MAX_RISK_RATIO)
          - Extra OTM buffer (PMCC_ASSIGNMENT_BUFFER_PCT)
          - Minimum credit target (PMCC_MIN_NET_CREDIT_TARGET)

        Args:
            spread: An active DiagonalSpread with no current short leg.

        Returns:
            A dict describing the best candidate short leg, or None if
            no suitable candidate is found or signals are not aligned.
        """
        if spread.has_short_leg:
            logger.debug(
                "PMCC evaluate_short_leg: %s already has active short leg -- skip",
                spread.spread_id,
            )
            return None

        if spread.status != "active":
            return None

        ticker = spread.ticker

        # Skip tickers with known issues
        # SILJ: options too illiquid, never fills within 120s
        PMCC_SKIP_TICKERS = {"SILJ"}
        if ticker in PMCC_SKIP_TICKERS:
            logger.debug(
                "PMCC evaluate_short_leg: %s in PMCC_SKIP_TICKERS -- skip",
                ticker,
            )
            return None

        # Check if the long LEAP still has enough DTE
        if spread.long_leg_dte < PMCC_LONG_LEG_MIN_DTE:
            logger.warning(
                "PMCC evaluate_short_leg: %s LEAP DTE=%d < %d -- LEAP is "
                "getting short, consider rolling the long leg",
                spread.spread_id, spread.long_leg_dte, PMCC_LONG_LEG_MIN_DTE,
            )
            # Still allow selling shorts but log the warning

        # Evaluate sell signals via the shared SignalEngine
        signal_result = self.signals.evaluate_sell(ticker)
        if not signal_result.triggered:
            logger.debug(
                "PMCC evaluate_short_leg: %s sell signal not triggered "
                "(%d/%d, need %d)",
                ticker, signal_result.score, signal_result.max_score,
                MIN_SELL_SIGNALS,
            )
            return None

        # Get underlying price
        underlying_price = self.data.get_price(ticker)
        if underlying_price is None or underlying_price <= 0:
            logger.warning(
                "PMCC evaluate_short_leg: %s price unavailable", ticker,
            )
            return None

        # Calculate the minimum acceptable short strike:
        # Must be above the LEAP strike AND far enough OTM to reduce
        # assignment risk.  PMCC_ASSIGNMENT_BUFFER_PCT (10%) is the OTM
        # floor; the delta cap (0.15) does the rest of the heavy lifting.
        #
        # NOTE: We intentionally do NOT add STRIKE_OTM_MIN (35%) here --
        # that constant is for the SLVR CC scalper on a $5 stock.  At 35%+
        # OTM on a $59 stock there are zero listed options with any OI.
        leap_strike = spread.long_leg.strike
        pmcc_min_otm = PMCC_ASSIGNMENT_BUFFER_PCT   # 10% OTM floor
        min_short_strike = max(
            leap_strike + 0.01,  # must be above LEAP strike
            underlying_price * (1 + pmcc_min_otm),
        )

        logger.info(
            "PMCC short leg search: %s underlying=$%.2f, LEAP strike=$%.0f, "
            "min short strike=$%.2f (%.0f%% OTM buffer)",
            ticker, underlying_price, leap_strike, min_short_strike,
            pmcc_min_otm * 100,
        )

        # Search option chains for suitable short leg candidates
        candidates = self._find_short_leg_candidates(
            ticker, underlying_price, min_short_strike, spread,
        )

        if not candidates:
            logger.info(
                "PMCC evaluate_short_leg: no eligible short leg candidates for %s",
                spread.spread_id,
            )
            return None

        # Return the best candidate
        best = candidates[0]
        logger.info(
            "PMCC SHORT LEG CANDIDATE: %s | $%.0f exp=%s DTE=%d | "
            "delta=%.3f | mid=$%.2f | risk_ratio=%.1f%%",
            ticker, best["strike"], best["expiration"], best["dte"],
            best["delta"], best["mid_price"],
            best["risk_ratio"] * 100,
        )

        return best

    def _find_short_leg_candidates(
        self,
        ticker: str,
        underlying_price: float,
        min_short_strike: float,
        spread: DiagonalSpread,
    ) -> List[Dict[str, Any]]:
        """Search the option chain for short leg candidates meeting PMCC criteria.

        Filters:
          - Strike >= min_short_strike
          - DTE within PMCC_SHORT_DTE_MIN to PMCC_SHORT_DTE_MAX
          - Delta <= PMCC_MAX_SHORT_DELTA
          - Premium >= PMCC_MIN_NET_CREDIT_TARGET
          - Risk ratio (premium / LEAP cost) <= PMCC_MAX_RISK_RATIO
          - Adequate open interest
          - Acceptable bid-ask spread

        Returns:
            Sorted list of candidate dicts (best first).
        """
        # We need to fetch option chains that include the short-term DTE range.
        # The DataLayer's fetch_option_chain uses the config DTE_MIN/MAX which
        # is for regular CCs (120-220).  For PMCC short legs we need 21-60 DTE.
        # We will query yfinance directly for the short-term expirations.
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed -- cannot search short leg candidates")
            return []

        candidates = []
        today = _today_et()

        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                return []

            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except ValueError:
                    continue

                dte = (exp_date - today).days
                if dte < PMCC_SHORT_DTE_MIN or dte > PMCC_SHORT_DTE_MAX:
                    continue

                try:
                    chain = t.option_chain(exp_str)
                    calls = chain.calls
                    if calls is None or len(calls) == 0:
                        continue
                except Exception as e:
                    logger.debug("PMCC chain fetch failed for %s/%s: %s", ticker, exp_str, e)
                    continue

                # Filter strikes
                mask = calls["strike"] >= min_short_strike
                filtered = calls[mask]
                if len(filtered) == 0:
                    continue

                T = dte / 365.0

                for _, row in filtered.iterrows():
                    strike = float(row.get("strike", 0))
                    _bid = row.get("bid", 0)
                    bid = 0.0 if (isinstance(_bid, float) and math.isnan(_bid)) else float(_bid)
                    _ask = row.get("ask", 0)
                    ask = 0.0 if (isinstance(_ask, float) and math.isnan(_ask)) else float(_ask)
                    # yfinance returns NaN for missing OI/volume — coerce safely
                    _oi_raw = row.get("openInterest", 0)
                    oi = 0 if (isinstance(_oi_raw, float) and math.isnan(_oi_raw)) else int(_oi_raw)
                    _vol_raw = row.get("volume", 0)
                    volume = 0 if (isinstance(_vol_raw, float) and math.isnan(_vol_raw)) else int(_vol_raw)
                    contract_symbol = str(row.get("contractSymbol", ""))

                    # Compute mid price
                    if bid > 0 and ask > 0:
                        mid = (bid + ask) / 2.0
                    else:
                        mid = float(row.get("lastPrice", 0))

                    # Filter: minimum open interest
                    if oi < MIN_OPEN_INTEREST:
                        continue

                    # Filter: live two-sided market required (Apr 30 2026 fix)
                    # Without this, stale yfinance mid lets dead contracts through;
                    # Alpaca then rejects with code 42210000 "contract not active"
                    # in a tight retry loop. See friction_log.md 2026-04-30 18:47.
                    if PMCC_SHORT_REQUIRE_LIVE_QUOTE and (bid <= 0 or ask <= 0):
                        continue

                    # Filter: bid-ask spread
                    if bid > 0 and ask > 0:
                        mid_calc = (bid + ask) / 2.0
                        if mid_calc > 0:
                            spread_pct = (ask - bid) / mid_calc
                            if spread_pct > MAX_BID_ASK_SPREAD_PCT:
                                continue

                    # Filter: minimum credit
                    if mid < PMCC_MIN_NET_CREDIT_TARGET:
                        continue

                    # Estimate delta via BS
                    hv20 = self.data.compute_hv(ticker, 20)
                    sigma = (hv20 * 1.10) if hv20 and hv20 > 0 else 0.75
                    greeks = _bs_greeks(underlying_price, strike, T, RISK_FREE_RATE, sigma)
                    delta = greeks["delta"]

                    # Filter: max delta (must be low probability of being ITM)
                    if delta > PMCC_MAX_SHORT_DELTA:
                        continue

                    # Filter: min delta floor (Apr 30 2026 fix)
                    # Zero/near-zero delta strikes are deep OTM with no realistic
                    # premium AND typically have no live Alpaca market — they slip
                    # through other gates and cause 422 retry storms. Belt-and-
                    # suspenders with REQUIRE_LIVE_QUOTE above.
                    if delta < PMCC_SHORT_MIN_DELTA:
                        continue

                    # Filter: risk ratio (premium / LEAP cost)
                    leap_cost = spread.long_leg.cost_basis
                    if leap_cost > 0:
                        risk_ratio = mid / leap_cost
                    else:
                        risk_ratio = 0.0

                    if risk_ratio > PMCC_MAX_RISK_RATIO:
                        logger.debug(
                            "PMCC risk ratio %.1f%% > %.1f%% max for %s $%.0f -- skip",
                            risk_ratio * 100, PMCC_MAX_RISK_RATIO * 100,
                            ticker, strike,
                        )
                        continue

                    # Score the candidate (higher = better)
                    score = 0.0

                    # Prefer higher premium (more income)
                    score += min(mid * 10, 10.0)

                    # Prefer lower delta (safer)
                    score += (PMCC_MAX_SHORT_DELTA - delta) * 50

                    # Prefer DTE near optimal
                    dte_dist = abs(dte - PMCC_SHORT_DTE_OPTIMAL)
                    score += max(0, 10 - dte_dist * 0.5)

                    # Prefer higher OI (liquidity)
                    score += min(oi / 100.0, 5.0)

                    # OTM percentage
                    otm_pct = (strike - underlying_price) / underlying_price

                    candidates.append({
                        "strike": strike,
                        "expiration": exp_str,
                        "dte": dte,
                        "delta": delta,
                        "mid_price": mid,
                        "bid": bid,
                        "ask": ask,
                        "oi": oi,
                        "volume": volume,
                        "contract_symbol": contract_symbol,
                        "otm_pct": otm_pct,
                        "risk_ratio": risk_ratio,
                        "score": score,
                        "greeks": greeks,
                    })

        except Exception as e:
            logger.error("PMCC short leg candidate search failed for %s: %s", ticker, e)
            return []

        # Sort by score descending
        candidates.sort(key=lambda c: c["score"], reverse=True)

        if candidates:
            logger.info(
                "PMCC short leg candidates for %s: %d found (top 3):",
                ticker, len(candidates),
            )
            for i, c in enumerate(candidates[:3]):
                logger.info(
                    "  #%d: $%.0f %s (DTE=%d, delta=%.3f, mid=$%.2f, "
                    "OTM=%.0f%%, risk=%.1f%%, score=%.1f)",
                    i + 1, c["strike"], c["expiration"], c["dte"],
                    c["delta"], c["mid_price"], c["otm_pct"] * 100,
                    c["risk_ratio"] * 100, c["score"],
                )

        return candidates

    def sell_short_leg(
        self,
        spread: DiagonalSpread,
        candidate: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Execute the sale of a short leg against an existing LEAP.

        Posts a limit sell order via the OrderManager for active management.

        Args:
            spread: The DiagonalSpread to sell the short leg on.
            candidate: A candidate dict from evaluate_short_leg().

        Returns:
            Action summary dict, or None if the order could not be submitted.
        """
        # --- RSI14 MOMENTUM GATE (Apr 15 audit) ---
        # Only sell short calls when RSI14 > 70 (sustained uptrend confirmed).
        # Knowledge base: "RSI14>70 alone is best single filter (Sharpe 1.5, 94% WR)"
        ticker = spread.ticker
        try:
            import yfinance as _yf
            _hist = _yf.Ticker(ticker).history(period='30d')
            if len(_hist) >= 14:
                _close = _hist['Close']
                _delta = _close.diff()
                _gain = _delta.clip(lower=0).ewm(span=14, adjust=False).mean()
                _loss = (-_delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
                _rsi14 = float(100 - (100 / (1 + _gain.iloc[-1] / _loss.iloc[-1])))
                if _rsi14 < 70:
                    logger.info("RSI14 GATE: %s RSI14=%.1f < 70 — skipping short call sell", ticker, _rsi14)
                    return None
                logger.info("RSI14 GATE: %s RSI14=%.1f >= 70 — OK to sell", ticker, _rsi14)
        except Exception as _e:
            logger.warning("RSI14 gate check failed for %s: %s — proceeding", ticker, _e)

        # --- SAFETY: Verify covering LEAP exists in Alpaca before selling ---
        # Without this check, we could submit a naked short call if our state
        # is stale or Alpaca doesn't see the LEAP. (PAAS incident Mar 27.)
        leap_symbol = spread.long_leg.symbol if spread.long_leg else None
        if leap_symbol and self.api:
            try:
                # Use _get_session() method (matches all other call sites in
                # this file); self.api is a CombinedApiAdapter which exposes
                # the session via method, not attribute. Previous direct
                # attribute access (self.api._session) broke after the
                # Edge 123 _AutoRefreshSession wrapper was introduced —
                # bled 57+ log ERRORs Mar 31–Apr 20 before being caught.
                session = self.api._get_session()
                positions_resp = session.get(
                    "https://paper-api.alpaca.markets/v2/positions"
                )
                positions = positions_resp.json()
                if not isinstance(positions, list):
                    raise RuntimeError(
                        f"Unexpected /v2/positions response shape "
                        f"(status={positions_resp.status_code}): "
                        f"{str(positions)[:200]}"
                    )
                leap_found = any(
                    isinstance(p, dict)
                    and p.get("symbol") == leap_symbol
                    and int(p.get("qty", 0)) > 0
                    for p in positions
                )
                if not leap_found:
                    logger.error(
                        "PMCC SAFETY: LEAP %s NOT FOUND in Alpaca positions! "
                        "Refusing to sell naked short call on %s | spread=%s",
                        leap_symbol, ticker, spread.spread_id,
                    )
                    return None
                else:
                    logger.debug(
                        "PMCC SAFETY: Confirmed LEAP %s exists in Alpaca positions",
                        leap_symbol,
                    )
            except Exception as e:
                logger.error(
                    "PMCC SAFETY: Could not verify LEAP %s in Alpaca -- "
                    "refusing to sell short call (fail-closed) | %s",
                    leap_symbol, e,
                )
                return None

        # --- DEDUP CHECK: prevent duplicate sells across all engines ---
        if self._order_dedup and self._order_dedup.has_pending_or_active_sell(ticker):
            logger.info(
                "PMCC DEDUP SKIP: another engine already has a pending or active "
                "short call on %s -- skipping short leg sell | spread=%s",
                ticker, spread.spread_id,
            )
            return None

        # --- CONFLICT CHECK: no simultaneous buy+sell on same ticker ---
        if self._order_dedup and self._order_dedup.has_conflicting_direction(ticker, "sell"):
            logger.info(
                "PMCC CONFLICT SKIP: pending buy on %s -- cannot sell simultaneously | spread=%s",
                ticker, spread.spread_id,
            )
            return None

        contract_symbol = candidate["contract_symbol"]
        strike = candidate["strike"]
        expiration = candidate["expiration"]
        dte = candidate["dte"]
        mid = candidate["mid_price"]
        bid = candidate["bid"]
        ask = candidate["ask"]
        delta = candidate["delta"]

        # Calculate limit price: sell above mid (into urgency)
        if bid > 0 and ask > 0:
            mid_calc = (bid + ask) / 2.0
            limit_price = round(mid_calc + SELL_OFFSET_FROM_MID, 2)
        else:
            limit_price = round(mid + SELL_OFFSET_FROM_MID, 2)

        # One contract per LEAP -- now matches LEAP count when flag is on
        contracts = (
            spread.long_leg.contracts
            if USE_LEG_CONTRACTS_IN_ORDERS else 1
        )

        logger.info(
            "PMCC SELL SHORT LEG: %s | $%.0f %s DTE=%d | delta=%.3f | "
            "limit=$%.2f (mid=$%.2f) | spread=%s",
            spread.ticker, strike, expiration, dte, delta,
            limit_price, mid, spread.spread_id,
        )

        # Build the short leg info (will be finalized on fill)
        short_leg = LegInfo(
            symbol=contract_symbol,
            strike=strike,
            expiry=expiration,
            delta=delta,
            cost_basis=limit_price,  # what we expect to receive
            current_value=mid,
            contracts=spread.long_leg.contracts,  # Edge 107 PR1b — match LEAP count
        )

        # Callbacks for OrderManager
        def _on_short_fill(mo: "ManagedOrder") -> None:
            """Called when the short leg sell order fills."""
            actual_price = mo.fill_price if mo.fill_price else mo.current_price
            short_leg.cost_basis = actual_price
            short_leg.current_value = actual_price
            spread.short_leg = short_leg
            spread.short_leg_original_dte = dte  # Track original DTE for first/second half calc
            spread.total_credits_received += actual_price
            spread.num_short_cycles += 1
            spread.last_updated = datetime.now(timezone.utc).isoformat()
            self.calculate_net_greeks(spread)
            self.calculate_max_loss(spread)
            logger.info(
                "PMCC SHORT LEG FILLED: %s | $%.0f %s @ $%.2f | "
                "total credits=$%.2f | cycle #%d | original_dte=%d | spread=%s",
                spread.ticker, strike, expiration, actual_price,
                spread.total_credits_received, spread.num_short_cycles,
                dte, spread.spread_id,
            )

        def _on_short_abort(mo: "ManagedOrder") -> None:
            """Called when the short leg order is aborted."""
            logger.warning(
                "PMCC SHORT LEG ABORTED: %s | $%.0f %s -- order failed "
                "after %d attempts | spread=%s",
                spread.ticker, strike, expiration,
                mo.attempts, spread.spread_id,
            )

        # Submit via OrderManager
        managed = self.order_manager.submit_sell(
            contract_symbol=contract_symbol,
            contracts=contracts,
            limit_price=limit_price,
            mid_price=mid,
            bid=bid,
            ask=ask,
            on_fill=_on_short_fill,
            on_abort=_on_short_abort,
        )

        if managed is None:
            logger.error(
                "PMCC SELL SHORT LEG FAILED: could not submit order for %s",
                spread.spread_id,
            )
            return None

        return {
            "type": "pmcc_short_leg_submitted",
            "spread_id": spread.spread_id,
            "ticker": spread.ticker,
            "long_strike": spread.long_leg.strike,
            "short_strike": strike,
            "short_expiration": expiration,
            "short_dte": dte,
            "short_delta": delta,
            "limit_price": limit_price,
            "mid_price": mid,
            "order_id": managed.order_id,
        }

    # ------------------------------------------------------------------
    # Profit-Taking (Premium Scalping Loop)
    # ------------------------------------------------------------------

    def _get_option_bid_ask(self, option_symbol: str) -> Tuple[float, float]:
        """Fetch live bid/ask for an option contract via Alpaca snapshot.

        Returns:
            (bid, ask) tuple. Either may be 0.0 if unavailable.
        """
        try:
            session = self.api._get_session()
            r = session.get(
                f"https://data.alpaca.markets/v1beta1/options/snapshots/{option_symbol}",
                params={"feed": "indicative"},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                quote = data.get("latestQuote", {})
                bid = float(quote.get("bp", 0.0))
                ask = float(quote.get("ap", 0.0))
                return (bid, ask)
        except Exception as e:
            logger.debug("_get_option_bid_ask(%s) failed: %s", option_symbol, e)
        return (0.0, 0.0)

    def _confirm_position_closed(self, option_symbol: str) -> bool:
        """Check if a short option position is gone from Alpaca.

        Returns True if position is confirmed closed (404 or qty==0).
        """
        try:
            session = self.api._get_session()
            resp = session.get(
                f"{ALPACA_BASE_URL}/v2/positions/{option_symbol}",
                timeout=10,
            )
            if resp.status_code == 200:
                remaining = resp.json()
                remaining_qty = abs(int(remaining.get("qty", 0)))
                if remaining_qty > 0:
                    logger.error(
                        "PMCC BUYBACK: position %s still has %d contracts on Alpaca",
                        option_symbol, remaining_qty,
                    )
                    return False
            # 404 = position gone = confirmed closed
            return True
        except Exception as e:
            logger.warning(
                "PMCC BUYBACK: could not verify position closure for %s: %s -- assuming closed",
                option_symbol, e,
            )
            return True

    def _compute_hard_cap(self, spread: DiagonalSpread) -> float:
        """Compute the DTE-dependent max buyback price for a short leg.

        First half of contract life (DTE > original_DTE/2): 30% of sell price
        Second half (DTE <= original_DTE/2): 20% of sell price
        Crash mode: no cap (returns a very large number).

        Returns:
            Hard cap price (per share).
        """
        short = spread.short_leg
        if short is None:
            return 0.01

        if self._crash_mode:
            # Crash mode: cap at 50% of sell price (aggressive but not unlimited)
            return round(short.cost_basis * 0.50, 2)

        original_dte = spread.short_leg_original_dte
        current_dte = spread.short_leg_dte

        # Determine first vs second half
        if original_dte > 0 and current_dte > original_dte / 2.0:
            cap_pct = PMCC_BUYBACK_MAX_FIRST_HALF  # 30%
        else:
            cap_pct = PMCC_BUYBACK_MAX_SECOND_HALF  # 20%

        hard_cap = round(short.cost_basis * cap_pct, 2)
        return max(hard_cap, 0.01)

    # ------------------------------------------------------------------
    # Cheap BTC Stepping — helpers
    # ------------------------------------------------------------------

    def _is_btc_step_candidate(self, spread: DiagonalSpread, current_delta: float, current_value: float) -> bool:
        """Return True if the short leg qualifies for cheap BTC stepping."""
        if not PMCC_BTC_CHEAP_ENABLED:
            return False
        if spread.short_leg is None:
            return False
        return current_delta <= PMCC_BTC_CHEAP_MAX_DELTA and current_value <= PMCC_BTC_CHEAP_MAX_VALUE

    def _compute_btc_step_price(self, spread: DiagonalSpread) -> float:
        """Compute the current stepping price based on elapsed time.

        Steps up $0.01 every PMCC_BTC_STEP_INTERVAL_SEC seconds from
        btc_step_start_ts.  Caps at PMCC_BTC_STEP_CAP_PCT of the original
        premium received (short_leg.cost_basis).

        Uses DAY orders, so each morning starts fresh at $0.01.
        """
        now = time.time()

        # Initialize stepping timestamp if not set
        if spread.btc_step_start_ts is None:
            spread.btc_step_start_ts = now

        elapsed = now - spread.btc_step_start_ts
        steps = int(elapsed / PMCC_BTC_STEP_INTERVAL_SEC)
        price = round(PMCC_BTC_CHEAP_START_PRICE + (steps * PMCC_BTC_CHEAP_STEP), 2)

        # Cap at configured percentage of original premium received
        if spread.short_leg is not None and spread.short_leg.cost_basis > 0:
            max_price = round(spread.short_leg.cost_basis * PMCC_BTC_STEP_CAP_PCT, 2)
        else:
            max_price = PMCC_BTC_CHEAP_START_PRICE  # fallback

        return min(price, max(max_price, 0.01))

    def _clear_btc_step_state(self, spread: DiagonalSpread) -> None:
        """Reset cheap BTC stepping state on a spread."""
        if spread.btc_step_start_ts is not None:
            logger.debug(
                "PMCC BTC STEP: clearing state for %s", spread.spread_id,
            )
        spread.btc_step_start_ts = None
        spread.btc_step_last_price = 0.0

    def check_profit_target(self, spread: DiagonalSpread) -> Optional[Dict[str, Any]]:
        """Check if the short leg should be bought back.

        Dual-trigger system (v2):

          Trigger 1 (50% profit): If the short call has decayed to 50% of
            sell price, buy it back regardless of DTE.

          Trigger 2 (21 DTE checkpoint): If DTE <= PMCC_CLOSE_DTE (21),
            close the position regardless of profit level.  If underwater,
            this still fires (assignment risk management handles evaluation).

          Let-expire zone: If DTE <= PMCC_LET_EXPIRE_DTE (5) AND
            delta < PMCC_LET_EXPIRE_DELTA (0.10) AND
            current_value < PMCC_LET_EXPIRE_VALUE ($0.05),
            skip the buyback -- let it expire worthless.

          Crash mode: If VIX > 30 or breadth gate triggered, immediately
            close ALL short calls at ask price, ignoring max buyback caps
            and the price ladder schedule.

        Uses a price-ladder system to escalate the buy-back limit price
        over time if the initial order does not fill:

          Step 1 (0 min): submit at theoretical BS value (min $0.01)
          Step 2 (15 min): bump to mid price
          Step 3 (30 min): bump to ask price
          Hard cap: DTE-dependent (30% first half, 20% second half of contract)
          Crash mode: skip ladder, go straight to ask, no cap.

        Tracks pending orders across cycles via spread.pending_buyback_order_id
        and spread.pending_buyback_submitted_at.

        Returns:
            Action dict if a buy-back was initiated/bumped, None otherwise.
        """
        if not spread.has_short_leg:
            return None

        ticker = spread.ticker
        short = spread.short_leg
        underlying_price = self.data.get_price(ticker)

        if underlying_price is None or underlying_price <= 0:
            return None

        # Calculate current theoretical value of the short call
        short_dte = spread.short_leg_dte
        if short_dte <= 0:
            # Expired -- manage_assignment_risk handles this
            return None

        T_short = max(short_dte / 365.0, 0.001)
        hv20 = self.data.compute_hv(ticker, 20)
        sigma = (hv20 * 1.10) if hv20 and hv20 > 0 else 0.75
        current_value = _bs_call_price(
            underlying_price, short.strike, T_short, RISK_FREE_RATE, sigma,
        )

        # Estimate current delta for let-expire check
        current_greeks = _bs_greeks(
            underlying_price, short.strike, T_short, RISK_FREE_RATE, sigma,
        )
        current_delta = current_greeks["delta"]
        short.delta = current_delta
        short.current_value = current_value

        # Compute DTE-dependent hard cap
        hard_cap = self._compute_hard_cap(spread)

        # ── Handle existing pending buy-back order ──────────────────────
        if spread.pending_buyback_order_id:
            return self._manage_pending_buyback(spread, current_value, hard_cap, current_delta)

        # ── Cheap BTC stepping: actively close near-worthless shorts ───
        # Instead of passively letting near-worthless calls expire, actively
        # try to close them cheap so the LEAP is free to resell on the next spike.
        # Routes through the normal pending_buyback flow with stepping logic.
        if self._is_btc_step_candidate(spread, current_delta, current_value):
            btc_price = self._compute_btc_step_price(spread)
            logger.info(
                "PMCC BTC STEP INITIATE: %s | delta=%.3f value=$%.3f | "
                "submitting BTC @ $%.2f (cap $%.2f = %.0f%% of $%.2f premium) | spread=%s",
                spread.ticker, current_delta, current_value, btc_price,
                round(short.cost_basis * PMCC_BTC_STEP_CAP_PCT, 2),
                PMCC_BTC_STEP_CAP_PCT * 100, short.cost_basis, spread.spread_id,
            )
            order_id = self.order_manager.submit_buyback_nonblocking(
                contract_symbol=short.symbol,
                contracts=(short.contracts if USE_LEG_CONTRACTS_IN_ORDERS else 1),
                limit_price=btc_price,
                tif="day",
            )
            if order_id is None:
                logger.error(
                    "PMCC BTC STEP: failed to submit BTC order for %s @ $%.2f",
                    spread.spread_id, btc_price,
                )
                return None
            spread.pending_buyback_order_id = order_id
            spread.pending_buyback_limit = btc_price
            spread.pending_buyback_submitted_at = datetime.now(timezone.utc).isoformat()
            spread.btc_step_last_price = btc_price
            spread.last_updated = datetime.now(timezone.utc).isoformat()
            return {
                "type": "pmcc_cheap_btc_submitted",
                "spread_id": spread.spread_id,
                "ticker": spread.ticker,
                "btc_price": btc_price,
                "max_price": round(short.cost_basis * PMCC_BTC_STEP_CAP_PCT, 2),
                "delta": round(current_delta, 4),
                "value": round(current_value, 4),
                "order_id": order_id,
            }

        # ── Let-expire zone: don't bother buying back ──────────────────
        # Only reached if cheap BTC is disabled or the call doesn't qualify
        if (short_dte <= PMCC_LET_EXPIRE_DTE
                and current_delta < PMCC_LET_EXPIRE_DELTA
                and current_value < PMCC_LET_EXPIRE_VALUE):
            logger.info(
                "PMCC LET-EXPIRE ZONE: %s | DTE=%d delta=%.3f value=$%.3f | "
                "all below thresholds (DTE<=%d, delta<%.2f, val<$%.2f) -- "
                "letting expire worthless | spread=%s",
                ticker, short_dte, current_delta, current_value,
                PMCC_LET_EXPIRE_DTE, PMCC_LET_EXPIRE_DELTA,
                PMCC_LET_EXPIRE_VALUE, spread.spread_id,
            )
            return None

        # ── Determine trigger reason ───────────────────────────────────
        trigger_reason = None
        original_dte = spread.short_leg_original_dte
        in_first_half = (original_dte > 0 and short_dte > original_dte / 2.0)

        # Pick profit target based on contract half
        profit_target = PMCC_PROFIT_TARGET_EARLY if in_first_half else PMCC_PROFIT_TARGET_LATE
        profit_threshold = short.cost_basis * (1.0 - profit_target)

        # Crash mode: close ALL short calls aggressively
        if self._crash_mode:
            trigger_reason = "crash_mode"
        # Trigger 1: 50% profit
        elif current_value <= profit_threshold:
            trigger_reason = "profit_target"
        # Trigger 2: 21 DTE checkpoint
        elif short_dte <= PMCC_CLOSE_DTE:
            trigger_reason = "dte_checkpoint"

        if trigger_reason is None:
            return None

        # Calculate P&L
        profit_pct = 1.0 - (current_value / short.cost_basis) if short.cost_basis > 0 else 0
        # Edge 107 — scale by short-leg contracts
        profit_dollars = (short.cost_basis - current_value) * 100 * short.contracts

        logger.info(
            "BUYBACK SIGNAL GENERATED [%s]: %s | sold @ $%.2f, now $%.2f | "
            "%.0f%% profit ($%.0f) | DTE=%d | delta=%.3f | crash=%s | spread=%s | "
            "NOTE: signal only, order not yet submitted",
            trigger_reason.upper(), ticker, short.cost_basis, current_value,
            profit_pct * 100, profit_dollars, short_dte, current_delta,
            self._crash_mode, spread.spread_id,
        )

        # ── Determine initial buyback price ────────────────────────────
        if self._crash_mode:
            # Crash mode: skip ladder, go straight to ask or midpoint
            bid, ask = self._get_option_bid_ask(short.symbol)
            if ask > 0 and bid > 0:
                # Use midpoint of bid/ask
                buyback_limit = round((bid + ask) / 2, 2)
                buyback_limit = max(buyback_limit, 0.01)
            elif ask > 0:
                buyback_limit = round(ask, 2)
            else:
                # Fallback: midpoint between 0 and entry premium
                buyback_limit = round(max(short.cost_basis * 0.25, 0.01), 2)
                logger.warning(
                    "PMCC CRASH: no bid/ask data for %s, using 25%% of entry ($%.2f) as fallback",
                    ticker, buyback_limit,
                )
            logger.info(
                "PMCC CRASH MODE: %s | going straight to ask=$%.2f | "
                "crash cap=50%% of sold ($%.2f) | spread=%s",
                round(spread.short_leg.cost_basis * 0.50, 2) if spread.short_leg else 0,
                ticker, buyback_limit, spread.spread_id,
            )
        else:
            # Normal mode: Step 1 of price ladder -- submit at theoretical value
            buyback_limit = round(max(current_value, 0.01), 2)

            # Enforce hard cap
            if buyback_limit > hard_cap:
                logger.info(
                    "PMCC BUYBACK: theo $%.2f exceeds hard cap $%.2f -- capping",
                    buyback_limit, hard_cap,
                )
                buyback_limit = hard_cap

        # Submit non-blocking buy-back order
        order_id = self.order_manager.submit_buyback_nonblocking(
            contract_symbol=short.symbol,
            contracts=(short.contracts if USE_LEG_CONTRACTS_IN_ORDERS else 1),
            limit_price=buyback_limit,
            tif="day",
        )

        if order_id is None:
            logger.error(
                "PMCC PROFIT: could not submit buy-back for %s", spread.spread_id,
            )
            return None

        # Track the order on the spread
        spread.pending_buyback_order_id = order_id
        spread.pending_buyback_limit = buyback_limit
        spread.pending_buyback_submitted_at = datetime.now(timezone.utc).isoformat()
        spread.last_updated = datetime.now(timezone.utc).isoformat()

        ladder_step = "ask_crash" if self._crash_mode else "theo"
        logger.info(
            "PMCC BUYBACK LADDER [%s]: %s | limit=$%.2f | "
            "hard_cap=$%.2f | trigger=%s | order=%s",
            ladder_step.upper(), spread.spread_id, buyback_limit,
            hard_cap, trigger_reason, order_id[:8],
        )

        return {
            "type": "pmcc_profit_buyback",
            "spread_id": spread.spread_id,
            "ticker": ticker,
            "short_strike": short.strike,
            "entry_premium": short.cost_basis,
            "current_value": current_value,
            "buyback_limit": buyback_limit,
            "profit_pct": round(profit_pct * 100, 1),
            "estimated_profit": round(profit_dollars, 2),
            "order_id": order_id,
            "ladder_step": ladder_step,
            "trigger_reason": trigger_reason,
            "crash_mode": self._crash_mode,
        }

    def _manage_pending_buyback(
        self,
        spread: DiagonalSpread,
        current_value: float,
        hard_cap: float,
        current_delta: float = 1.0,
    ) -> Optional[Dict[str, Any]]:
        """Manage a pending buy-back order: check fill, bump price if stale.

        Called from check_profit_target() when spread.pending_buyback_order_id
        is set.  Checks the order status on Alpaca and escalates the limit
        price according to either:
          - The normal price ladder (theo -> mid -> ask over 15/30 min), OR
          - BTC stepping ($0.01 increments every 5 min) when the short call
            is near-worthless (delta < 0.05, value < $0.10).

        BTC stepping uses DAY orders; unfilled orders expire at close and the
        next morning starts fresh at $0.01.

        Returns:
            Action dict if order was bumped or filled, None if no action needed.
        """
        short = spread.short_leg
        order_id = spread.pending_buyback_order_id

        # Determine if this spread is in BTC stepping mode
        is_stepping = self._is_btc_step_candidate(spread, current_delta, current_value)

        # If no longer a step candidate but was stepping, clear step state
        if not is_stepping and spread.btc_step_start_ts is not None:
            logger.info(
                "PMCC BTC STEP: %s no longer qualifies (delta=%.3f value=$%.3f) -- "
                "reverting to normal ladder | spread=%s",
                spread.ticker, current_delta, current_value, spread.spread_id,
            )
            self._clear_btc_step_state(spread)

        # ── Check order status on Alpaca ────────────────────────────────
        order_status = self.order_manager.get_order_status(order_id)

        if order_status is None:
            # Could not reach Alpaca -- leave tracking in place, retry next cycle
            logger.warning(
                "PMCC BUYBACK: could not fetch order %s status -- retry next cycle",
                order_id[:8] if order_id else "None",
            )
            return None

        alpaca_status = order_status.get("status", "")

        # ── Handle filled order ─────────────────────────────────────────
        if alpaca_status == "filled":
            actual = float(order_status.get("filled_avg_price", spread.pending_buyback_limit))
            # Edge 107 — scale by short-leg contracts
            realized_pnl = (short.cost_basis - actual) * 100 * short.contracts

            position_closed = self._confirm_position_closed(short.symbol)

            fill_type = "pmcc_cheap_btc_filled" if is_stepping else "pmcc_profit_buyback_filled"

            if position_closed:
                logger.info(
                    "PMCC %s FILLED: %s | bought back @ $%.2f | "
                    "realized P&L: $%.0f | short leg cleared | status -> idle_after_profit",
                    "BTC STEP" if is_stepping else "BUYBACK",
                    spread.ticker, actual, realized_pnl,
                )
                spread.short_leg = None
                spread.short_leg_original_dte = 0
                spread.pending_buyback_order_id = None
                spread.pending_buyback_limit = 0.0
                spread.pending_buyback_submitted_at = None
                self._clear_btc_step_state(spread)
                spread.status = "idle_after_profit"
                spread.last_updated = datetime.now(timezone.utc).isoformat()
                self.calculate_net_greeks(spread)
            else:
                logger.warning(
                    "PMCC %s FILLED (UNCONFIRMED): %s | "
                    "bought back @ $%.2f | short_leg NOT cleared -- re-check next cycle",
                    "BTC STEP" if is_stepping else "BUYBACK",
                    spread.ticker, actual,
                )
                spread.pending_buyback_order_id = None
                spread.pending_buyback_submitted_at = None
                self._clear_btc_step_state(spread)

            return {
                "type": fill_type,
                "spread_id": spread.spread_id,
                "ticker": spread.ticker,
                "fill_price": actual,
                "realized_pnl": round(realized_pnl, 2),
            }

        # ── Handle terminal states (cancelled, expired, rejected) ───────
        if alpaca_status in ("cancelled", "canceled", "expired", "rejected"):
            logger.info(
                "PMCC BUYBACK: order %s is %s -- clearing tracker, will re-evaluate",
                order_id[:8], alpaca_status,
            )
            spread.pending_buyback_order_id = None
            spread.pending_buyback_limit = 0.0
            spread.pending_buyback_submitted_at = None
            # For BTC stepping: DAY order expired at close; clear step state
            # so next morning starts fresh at $0.01
            if is_stepping:
                self._clear_btc_step_state(spread)
            # Return None so check_profit_target re-evaluates from scratch next cycle
            return None

        # ── Order still open -- check if we need to bump the price ──────
        if alpaca_status not in ("new", "accepted", "pending_new", "partially_filled"):
            logger.debug(
                "PMCC BUYBACK: order %s in unexpected status '%s' -- skipping",
                order_id[:8], alpaca_status,
            )
            return None

        # ── BTC stepping mode: $0.01 increments every 5 minutes ────────
        if is_stepping:
            new_step_price = self._compute_btc_step_price(spread)
            current_limit = spread.pending_buyback_limit

            if abs(new_step_price - current_limit) < 0.005:
                # Same price -- let the order work
                logger.debug(
                    "PMCC BTC STEP: %s order working at $%.2f | spread=%s",
                    spread.ticker, current_limit, spread.spread_id,
                )
                return None

            # Price needs to step up -- cancel and resubmit
            logger.info(
                "PMCC BTC STEP UP: %s | $%.2f -> $%.2f (every %ds) | "
                "cap $%.2f (%.0f%% of $%.2f) | order=%s | spread=%s",
                spread.ticker, current_limit, new_step_price,
                PMCC_BTC_STEP_INTERVAL_SEC,
                round(short.cost_basis * PMCC_BTC_STEP_CAP_PCT, 2),
                PMCC_BTC_STEP_CAP_PCT * 100, short.cost_basis,
                order_id[:8], spread.spread_id,
            )

            new_order_id = self.order_manager.submit_buyback_nonblocking(
                contract_symbol=short.symbol,
                contracts=(short.contracts if USE_LEG_CONTRACTS_IN_ORDERS else 1),
                limit_price=new_step_price,
                tif="day",
            )
            if new_order_id is None:
                logger.error(
                    "PMCC BTC STEP: failed to submit stepped order for %s",
                    spread.spread_id,
                )
                spread.pending_buyback_order_id = None
                spread.pending_buyback_limit = 0.0
                spread.pending_buyback_submitted_at = None
                return None

            spread.pending_buyback_order_id = new_order_id
            spread.pending_buyback_limit = new_step_price
            spread.btc_step_last_price = new_step_price
            spread.last_updated = datetime.now(timezone.utc).isoformat()

            return {
                "type": "pmcc_cheap_btc_stepped",
                "spread_id": spread.spread_id,
                "ticker": spread.ticker,
                "old_limit": current_limit,
                "new_limit": new_step_price,
                "cap": round(short.cost_basis * PMCC_BTC_STEP_CAP_PCT, 2),
                "order_id": new_order_id,
            }

        # ── Normal ladder progression (non-stepping) ────────────────────

        # Calculate how long this order has been working
        submitted_at = spread.pending_buyback_submitted_at
        if not submitted_at:
            # Legacy spread without timestamp -- set it now, bump next cycle
            spread.pending_buyback_submitted_at = datetime.now(timezone.utc).isoformat()
            return None

        try:
            submit_time = datetime.fromisoformat(submitted_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            submit_time = datetime.now(timezone.utc)
            spread.pending_buyback_submitted_at = submit_time.isoformat()

        age_minutes = (datetime.now(timezone.utc) - submit_time).total_seconds() / 60.0

        # Determine current ladder step and whether a bump is needed
        current_limit = spread.pending_buyback_limit

        # Get live bid/ask for the option
        bid, ask = self._get_option_bid_ask(short.symbol)
        mid = round((bid + ask) / 2.0, 2) if (bid > 0 and ask > 0) else current_value

        new_limit = None
        ladder_step = None

        # ── Crash mode: skip ladder, go straight to ask ──────────────
        if self._crash_mode:
            target = round(max(ask, current_value * 1.10, 0.01), 2) if ask > 0 else round(max(current_value * 1.10, 0.01), 2)
            if target > current_limit + 0.005:
                new_limit = target
                ladder_step = "ask_crash"
                logger.info(
                    "PMCC CRASH MODE BUMP: %s | bypassing ladder, going to ask=$%.2f | "
                    "spread=%s",
                    spread.ticker, target, spread.spread_id,
                )
        # ── Normal ladder progression ────────────────────────────────
        elif age_minutes >= PMCC_BUYBACK_MID_WAIT_MINUTES:
            # Step 3: bump to ask
            target = round(max(ask, current_value, 0.01), 2) if ask > 0 else round(max(current_value * 1.1, 0.01), 2)
            if target > current_limit + 0.005:  # Only bump if meaningfully higher
                new_limit = target
                ladder_step = "ask"
        elif age_minutes >= PMCC_BUYBACK_INITIAL_WAIT_MINUTES:
            # Step 2: bump to mid
            target = round(max(mid, current_value, 0.01), 2)
            if target > current_limit + 0.005:
                new_limit = target
                ladder_step = "mid"

        if new_limit is None:
            # No bump needed yet -- order is still working at current price
            logger.debug(
                "PMCC BUYBACK: order %s still working at $%.2f (age=%.0fm, "
                "next bump at %dm) | crash=%s | spread=%s",
                order_id[:8], current_limit, age_minutes,
                PMCC_BUYBACK_INITIAL_WAIT_MINUTES if age_minutes < PMCC_BUYBACK_INITIAL_WAIT_MINUTES else PMCC_BUYBACK_MID_WAIT_MINUTES,
                self._crash_mode, spread.spread_id,
            )
            return None

        # Enforce hard cap (bypassed in crash mode since hard_cap=999.99)
        if new_limit > hard_cap:
            if current_limit >= hard_cap - 0.005:
                # Already at cap -- don't bump, let it ride or expire
                logger.info(
                    "PMCC BUYBACK: %s already at hard cap $%.2f (ask=$%.2f) -- "
                    "letting order ride at $%.2f",
                    spread.spread_id, hard_cap, ask, current_limit,
                )
                return None
            new_limit = hard_cap
            logger.info(
                "PMCC BUYBACK: capping bump at $%.2f (hard cap for current DTE half)",
                hard_cap,
            )

        # Cancel old order and submit at new price
        logger.info(
            "PMCC BUYBACK LADDER [STEP %s]: %s | bumping $%.2f -> $%.2f | "
            "age=%.0fm | bid=$%.2f ask=$%.2f | order=%s",
            "2/3 MID" if ladder_step == "mid" else "3/3 ASK",
            spread.spread_id, current_limit, new_limit,
            age_minutes, bid, ask, order_id[:8],
        )

        new_order_id = self.order_manager.submit_buyback_nonblocking(
            contract_symbol=short.symbol,
            contracts=(short.contracts if USE_LEG_CONTRACTS_IN_ORDERS else 1),
            limit_price=new_limit,
            tif="day",
        )

        if new_order_id is None:
            logger.error(
                "PMCC BUYBACK: failed to submit bumped order for %s -- "
                "old order was already canceled",
                spread.spread_id,
            )
            # Old order was canceled by submit_buyback_nonblocking -- clear tracker
            spread.pending_buyback_order_id = None
            spread.pending_buyback_limit = 0.0
            spread.pending_buyback_submitted_at = None
            return None

        # Update tracking -- keep original submitted_at so age accumulates
        spread.pending_buyback_order_id = new_order_id
        spread.pending_buyback_limit = new_limit
        spread.last_updated = datetime.now(timezone.utc).isoformat()

        return {
            "type": "pmcc_profit_buyback_bump",
            "spread_id": spread.spread_id,
            "ticker": spread.ticker,
            "old_limit": current_limit,
            "new_limit": new_limit,
            "ladder_step": ladder_step,
            "age_minutes": round(age_minutes, 1),
            "hard_cap": hard_cap,
            "order_id": new_order_id,
        }

    # ------------------------------------------------------------------
    # Assignment Risk Management
    # ------------------------------------------------------------------

    def manage_assignment_risk(self, spread: DiagonalSpread) -> Optional[Dict[str, Any]]:
        """Monitor the short leg for assignment risk and take action.

        Escalating response based on short leg delta:
          - delta > PMCC_SHORT_DELTA_WARN (0.40): log WARNING
          - delta > PMCC_SHORT_DELTA_DANGER (0.50): evaluate rolling
          - underlying > short strike: URGENT -- roll or close to avoid
            burning LEAP time value

        Assignment on a PMCC is more expensive than on shares because:
          1. You must exercise the LEAP to deliver shares, losing all
             remaining extrinsic (time) value on the LEAP.
          2. Or you close the short at a loss + close the LEAP at reduced
             value -- either way you lose the LEAP's time premium.

        Args:
            spread: An active spread with a short leg.

        Returns:
            Action dict if action was taken, None otherwise.
        """
        if not spread.has_short_leg:
            return None

        ticker = spread.ticker
        short = spread.short_leg
        underlying_price = self.data.get_price(ticker)

        if underlying_price is None or underlying_price <= 0:
            logger.warning(
                "PMCC assignment risk: cannot check %s -- price unavailable",
                spread.spread_id,
            )
            return None

        # Estimate current short leg delta
        short_dte = spread.short_leg_dte
        if short_dte <= 0:
            # Short leg has expired -- mark as expired worthless if OTM
            if underlying_price < short.strike:
                logger.info(
                    "PMCC SHORT EXPIRED WORTHLESS: %s | $%.0f | "
                    "premium kept: $%.2f | spread=%s",
                    ticker, short.strike, short.cost_basis,
                    spread.spread_id,
                )
                spread.short_leg = None
                spread.last_updated = datetime.now(timezone.utc).isoformat()
                self.calculate_net_greeks(spread)
                return {
                    "type": "pmcc_short_expired_worthless",
                    "spread_id": spread.spread_id,
                    "ticker": ticker,
                    "short_strike": short.strike,
                    "premium_kept": short.cost_basis,
                }
            else:
                # Expired ITM -- assignment happened
                logger.error(
                    "PMCC ASSIGNMENT: %s | short $%.0f expired ITM "
                    "(underlying=$%.2f) | LEAP time value burned! | spread=%s",
                    ticker, short.strike, underlying_price, spread.spread_id,
                )
                spread.status = "assigned"
                spread.last_updated = datetime.now(timezone.utc).isoformat()
                return {
                    "type": "pmcc_assignment",
                    "spread_id": spread.spread_id,
                    "ticker": ticker,
                    "short_strike": short.strike,
                    "underlying_price": underlying_price,
                    "severity": "CRITICAL",
                }

        T_short = short_dte / 365.0
        hv20 = self.data.compute_hv(ticker, 20)
        sigma = (hv20 * 1.10) if hv20 and hv20 > 0 else 0.75
        greeks = _bs_greeks(underlying_price, short.strike, T_short, RISK_FREE_RATE, sigma)
        current_delta = greeks["delta"]

        # Update the stored delta
        short.delta = current_delta

        # Estimate current short leg value (for P&L tracking)
        short_value = _bs_call_price(
            underlying_price, short.strike, T_short, RISK_FREE_RATE, sigma,
        )
        short.current_value = short_value

        # Update spread greeks
        self.calculate_net_greeks(spread)

        # --- Check danger levels ---

        # Level 3 (URGENT): underlying above short strike
        if underlying_price >= short.strike:
            intrinsic = underlying_price - short.strike
            # Calculate cost to exercise LEAP vs roll
            leap_dte = spread.long_leg_dte
            T_leap = leap_dte / 365.0 if leap_dte > 0 else 0.001
            leap_value = _bs_call_price(
                underlying_price, spread.long_leg.strike, T_leap,
                RISK_FREE_RATE, sigma,
            )
            leap_intrinsic = max(underlying_price - spread.long_leg.strike, 0)
            leap_extrinsic = leap_value - leap_intrinsic

            logger.error(
                "PMCC URGENT: %s | underlying $%.2f >= short strike $%.0f | "
                "short ITM by $%.2f | LEAP extrinsic at risk: $%.2f | "
                "delta=%.2f | DTE=%d | spread=%s",
                ticker, underlying_price, short.strike, intrinsic,
                leap_extrinsic, current_delta, short_dte, spread.spread_id,
            )

            return {
                "type": "pmcc_assignment_urgent",
                "spread_id": spread.spread_id,
                "ticker": ticker,
                "underlying_price": underlying_price,
                "short_strike": short.strike,
                "short_delta": current_delta,
                "intrinsic": intrinsic,
                "leap_extrinsic_at_risk": leap_extrinsic,
                "recommendation": "ROLL_OR_CLOSE_IMMEDIATELY",
                "severity": "URGENT",
            }

        # Level 2 (DANGER): delta > 0.50
        if current_delta > PMCC_SHORT_DELTA_DANGER:
            logger.warning(
                "PMCC DANGER: %s | short delta=%.2f > %.2f danger threshold | "
                "$%.0f strike, underlying=$%.2f | DTE=%d | "
                "CONSIDER ROLLING | spread=%s",
                ticker, current_delta, PMCC_SHORT_DELTA_DANGER,
                short.strike, underlying_price, short_dte, spread.spread_id,
            )

            return {
                "type": "pmcc_assignment_danger",
                "spread_id": spread.spread_id,
                "ticker": ticker,
                "short_delta": current_delta,
                "short_strike": short.strike,
                "underlying_price": underlying_price,
                "recommendation": "EVALUATE_ROLL",
                "severity": "HIGH",
            }

        # Level 1 (WARNING): delta > 0.40
        if current_delta > PMCC_SHORT_DELTA_WARN:
            logger.warning(
                "PMCC WARNING: %s | short delta=%.2f > %.2f warn threshold | "
                "$%.0f strike, underlying=$%.2f | DTE=%d | "
                "monitoring closely | spread=%s",
                ticker, current_delta, PMCC_SHORT_DELTA_WARN,
                short.strike, underlying_price, short_dte, spread.spread_id,
            )

            return {
                "type": "pmcc_assignment_warning",
                "spread_id": spread.spread_id,
                "ticker": ticker,
                "short_delta": current_delta,
                "short_strike": short.strike,
                "underlying_price": underlying_price,
                "recommendation": "MONITOR",
                "severity": "MEDIUM",
            }

        # All clear
        logger.debug(
            "PMCC OK: %s | short delta=%.3f | $%.0f strike | "
            "underlying=$%.2f | DTE=%d | spread=%s",
            ticker, current_delta, short.strike,
            underlying_price, short_dte, spread.spread_id,
        )
        return None

    # ------------------------------------------------------------------
    # Rolling and Closing
    # ------------------------------------------------------------------

    def roll_short_leg(
        self,
        spread: DiagonalSpread,
        new_strike: float,
        new_expiry: str,
    ) -> Optional[Dict[str, Any]]:
        """Roll the short leg to a new strike and/or expiration.

        Executes a two-step process:
          1. Buy back the current short leg (close it).
          2. Sell a new short leg at the new strike/expiry.

        In practice this is two separate limit orders managed by the
        OrderManager (not a spread order, because Alpaca paper trading
        does not support multi-leg options).

        Args:
            spread: The spread whose short leg to roll.
            new_strike: New strike for the replacement short leg.
            new_expiry: New expiration for the replacement short leg (YYYY-MM-DD).

        Returns:
            Action summary dict, or None if the roll could not be initiated.
        """
        if not spread.has_short_leg:
            logger.warning(
                "PMCC roll_short_leg: %s has no short leg to roll",
                spread.spread_id,
            )
            return None

        ticker = spread.ticker
        old_short = spread.short_leg
        old_symbol = old_short.symbol
        old_strike = old_short.strike
        old_expiry = old_short.expiry

        logger.info(
            "PMCC ROLL INITIATED: %s | closing $%.0f %s -> opening $%.0f %s | spread=%s",
            ticker, old_strike, old_expiry, new_strike, new_expiry, spread.spread_id,
        )

        # Step 1: Buy back the current short leg
        # Estimate current value for the buy-back limit price
        underlying_price = self.data.get_price(ticker)
        if underlying_price is None or underlying_price <= 0:
            logger.error("PMCC roll: cannot get underlying price for %s", ticker)
            return None

        short_dte = spread.short_leg_dte
        T_short = max(short_dte / 365.0, 0.001)
        hv20 = self.data.compute_hv(ticker, 20)
        sigma = (hv20 * 1.10) if hv20 and hv20 > 0 else 0.75

        current_short_value = _bs_call_price(
            underlying_price, old_strike, T_short, RISK_FREE_RATE, sigma,
        )
        buyback_limit = round(max(current_short_value - BUYBACK_OFFSET_FROM_MID, 0.01), 2)

        # Build the new short leg contract symbol
        # We need to find the actual OCC symbol for the new strike/expiry
        new_contract_symbol = self._find_contract_symbol(
            ticker, new_strike, new_expiry,
        )
        if new_contract_symbol is None:
            logger.error(
                "PMCC roll: cannot find contract symbol for %s $%.0f %s",
                ticker, new_strike, new_expiry,
            )
            return None

        # Estimate new short leg value for the sell limit price
        try:
            new_exp_date = datetime.strptime(new_expiry, "%Y-%m-%d").date()
            new_dte = (new_exp_date - _today_et()).days
        except (ValueError, TypeError):
            new_dte = PMCC_SHORT_DTE_OPTIMAL

        T_new = new_dte / 365.0
        new_short_value = _bs_call_price(
            underlying_price, new_strike, T_new, RISK_FREE_RATE, sigma,
        )
        sell_limit = round(new_short_value + SELL_OFFSET_FROM_MID, 2)

        # Callback: when buy-back fills, immediately submit the new sell
        def _on_buyback_fill(mo: "ManagedOrder") -> None:
            """Buy-back of old short leg filled -- now sell the new one."""
            actual_buyback = mo.fill_price if mo.fill_price else mo.current_price
            cost_to_close = actual_buyback
            # Edge 107 — scale by old-short-leg contracts
            pnl_on_close = (old_short.cost_basis - cost_to_close) * 100 * old_short.contracts

            logger.info(
                "PMCC ROLL BUY-BACK FILLED: %s | $%.0f %s @ $%.2f | "
                "P&L on close: $%.0f | now selling $%.0f %s",
                ticker, old_strike, old_expiry, actual_buyback,
                pnl_on_close, new_strike, new_expiry,
            )

            # Clear the old short leg
            spread.short_leg = None
            spread.last_updated = datetime.now(timezone.utc).isoformat()

            # Build the new short leg
            new_greeks = _bs_greeks(
                underlying_price, new_strike, T_new, RISK_FREE_RATE, sigma,
            )

            new_short_leg = LegInfo(
                symbol=new_contract_symbol,
                strike=new_strike,
                expiry=new_expiry,
                delta=new_greeks["delta"],
                cost_basis=sell_limit,
                current_value=new_short_value,
                contracts=spread.long_leg.contracts,  # Edge 107 PR1b — match LEAP count
            )

            # Submit new sell order
            def _on_new_sell_fill(mo2: "ManagedOrder") -> None:
                actual_sell = mo2.fill_price if mo2.fill_price else mo2.current_price
                new_short_leg.cost_basis = actual_sell
                spread.short_leg = new_short_leg
                spread.total_credits_received += actual_sell
                spread.num_short_cycles += 1
                spread.last_updated = datetime.now(timezone.utc).isoformat()
                self.calculate_net_greeks(spread)
                self.calculate_max_loss(spread)
                logger.info(
                    "PMCC ROLL COMPLETE: %s | new short $%.0f %s @ $%.2f | "
                    "net roll credit/debit: $%.2f | spread=%s",
                    ticker, new_strike, new_expiry, actual_sell,
                    actual_sell - actual_buyback, spread.spread_id,
                )

            def _on_new_sell_abort(mo2: "ManagedOrder") -> None:
                logger.warning(
                    "PMCC ROLL SELL FAILED: %s | $%.0f %s -- could not fill "
                    "after %d attempts | spread=%s (now has no short leg!)",
                    ticker, new_strike, new_expiry,
                    mo2.attempts, spread.spread_id,
                )

            self.order_manager.submit_sell(
                contract_symbol=new_contract_symbol,
                contracts=(spread.long_leg.contracts if USE_LEG_CONTRACTS_IN_ORDERS else 1),
                limit_price=sell_limit,
                mid_price=new_short_value,
                on_fill=_on_new_sell_fill,
                on_abort=_on_new_sell_abort,
            )

        def _on_buyback_abort(mo: "ManagedOrder") -> None:
            logger.warning(
                "PMCC ROLL BUY-BACK FAILED: %s | $%.0f %s -- could not close "
                "after %d attempts | roll aborted, position unchanged | spread=%s",
                ticker, old_strike, old_expiry,
                mo.attempts, spread.spread_id,
            )

        # Submit buy-back order
        managed = self.order_manager.submit_buy_back(
            contract_symbol=old_symbol,
            contracts=(old_short.contracts if USE_LEG_CONTRACTS_IN_ORDERS else 1),
            limit_price=buyback_limit,
            mid_price=current_short_value,
            on_fill=_on_buyback_fill,
            on_abort=_on_buyback_abort,
        )

        if managed is None:
            logger.error("PMCC ROLL FAILED: could not submit buy-back for %s", spread.spread_id)
            return None

        return {
            "type": "pmcc_roll_initiated",
            "spread_id": spread.spread_id,
            "ticker": ticker,
            "old_short_strike": old_strike,
            "old_short_expiry": old_expiry,
            "new_short_strike": new_strike,
            "new_short_expiry": new_expiry,
            "buyback_limit": buyback_limit,
            "sell_limit": sell_limit,
            "order_id": managed.order_id,
        }

    def close_spread(self, spread: DiagonalSpread) -> Optional[Dict[str, Any]]:
        """Close both legs of a diagonal spread.

        Closes the short leg first (buy back), then closes the long LEAP
        (sell to close).  Both orders go through the OrderManager.

        Args:
            spread: The spread to close entirely.

        Returns:
            Action summary dict, or None if closing could not be initiated.
        """
        ticker = spread.ticker

        logger.info(
            "PMCC CLOSE SPREAD: %s | long $%.0f exp=%s | "
            "short $%.0f exp=%s | spread=%s",
            ticker, spread.long_leg.strike, spread.long_leg.expiry,
            spread.short_leg.strike if spread.short_leg else 0,
            spread.short_leg.expiry if spread.short_leg else "N/A",
            spread.spread_id,
        )

        underlying_price = self.data.get_price(ticker)
        if underlying_price is None or underlying_price <= 0:
            logger.error("PMCC close_spread: cannot get price for %s", ticker)
            return None

        hv20 = self.data.compute_hv(ticker, 20)
        sigma = (hv20 * 1.10) if hv20 and hv20 > 0 else 0.75

        def _close_long_leg() -> None:
            """Sell the LEAP to close after the short leg is handled."""
            leap_dte = spread.long_leg_dte
            T_leap = max(leap_dte / 365.0, 0.001)
            leap_value = _bs_call_price(
                underlying_price, spread.long_leg.strike, T_leap,
                RISK_FREE_RATE, sigma,
            )
            sell_limit = round(max(leap_value - 0.10, 0.01), 2)  # slight discount for fill

            def _on_leap_sell_fill(mo: "ManagedOrder") -> None:
                actual = mo.fill_price if mo.fill_price else mo.current_price
                spread.status = "closed"
                spread.last_updated = datetime.now(timezone.utc).isoformat()
                # Edge 107 — scale by long-leg contracts
                long_n = spread.long_leg.contracts
                total_pnl = (
                    actual - spread.long_leg.cost_basis + spread.total_credits_received
                ) * 100 * long_n
                logger.info(
                    "PMCC SPREAD CLOSED: %s | LEAP sold @ $%.2f | "
                    "total P&L: $%.0f | credits collected: $%.2f | spread=%s",
                    ticker, actual, total_pnl,
                    spread.total_credits_received, spread.spread_id,
                )

            def _on_leap_sell_abort(mo: "ManagedOrder") -> None:
                logger.error(
                    "PMCC LEAP CLOSE FAILED: %s | could not sell LEAP after "
                    "%d attempts | spread=%s (short leg already closed!)",
                    ticker, mo.attempts, spread.spread_id,
                )

            self.order_manager.submit_sell(
                contract_symbol=spread.long_leg.symbol,
                contracts=(spread.long_leg.contracts if USE_LEG_CONTRACTS_IN_ORDERS else 1),
                limit_price=sell_limit,
                mid_price=leap_value,
                on_fill=_on_leap_sell_fill,
                on_abort=_on_leap_sell_abort,
            )

        if spread.has_short_leg:
            # Buy back the short leg first, then close the LEAP
            short = spread.short_leg
            short_dte = spread.short_leg_dte
            T_short = max(short_dte / 365.0, 0.001)
            short_value = _bs_call_price(
                underlying_price, short.strike, T_short, RISK_FREE_RATE, sigma,
            )
            buyback_limit = round(max(short_value - BUYBACK_OFFSET_FROM_MID, 0.01), 2)

            def _on_short_buyback_fill(mo: "ManagedOrder") -> None:
                actual = mo.fill_price if mo.fill_price else mo.current_price
                logger.info(
                    "PMCC CLOSE: short leg bought back @ $%.2f | now closing LEAP",
                    actual,
                )
                spread.short_leg = None
                spread.last_updated = datetime.now(timezone.utc).isoformat()
                _close_long_leg()

            def _on_short_buyback_abort(mo: "ManagedOrder") -> None:
                logger.error(
                    "PMCC CLOSE: short leg buy-back failed after %d attempts | "
                    "spread=%s (spread remains open!)",
                    mo.attempts, spread.spread_id,
                )

            managed = self.order_manager.submit_buy_back(
                contract_symbol=short.symbol,
                contracts=(short.contracts if USE_LEG_CONTRACTS_IN_ORDERS else 1),
                limit_price=buyback_limit,
                mid_price=short_value,
                on_fill=_on_short_buyback_fill,
                on_abort=_on_short_buyback_abort,
            )

            if managed is None:
                logger.error("PMCC close_spread: could not submit short buy-back")
                return None

            return {
                "type": "pmcc_close_spread_initiated",
                "spread_id": spread.spread_id,
                "ticker": ticker,
                "order_id": managed.order_id,
            }
        else:
            # No short leg -- just close the LEAP directly
            _close_long_leg()
            return {
                "type": "pmcc_close_spread_initiated",
                "spread_id": spread.spread_id,
                "ticker": ticker,
                "long_only": True,
            }

    # ------------------------------------------------------------------
    # Greeks and Risk Calculations
    # ------------------------------------------------------------------

    def calculate_net_greeks(self, spread: DiagonalSpread) -> Dict[str, float]:
        """Calculate position-level net Greeks for a diagonal spread.

        Net Greeks = Long LEAP Greeks - Short Call Greeks (since we are
        long the LEAP and short the near-term call).

        Updates spread.net_greeks in place and returns the result.
        """
        ticker = spread.ticker
        underlying_price = self.data.get_price(ticker)

        if underlying_price is None or underlying_price <= 0:
            return spread.net_greeks

        hv20 = self.data.compute_hv(ticker, 20)
        sigma = (hv20 * 1.10) if hv20 and hv20 > 0 else 0.75

        # Long LEAP Greeks
        leap_dte = spread.long_leg_dte
        T_leap = max(leap_dte / 365.0, 0.001)
        long_greeks = _bs_greeks(
            underlying_price, spread.long_leg.strike, T_leap, RISK_FREE_RATE, sigma,
        )

        # Update long leg delta
        spread.long_leg.delta = long_greeks["delta"]

        # Update long leg current value
        spread.long_leg.current_value = _bs_call_price(
            underlying_price, spread.long_leg.strike, T_leap, RISK_FREE_RATE, sigma,
        )

        if spread.has_short_leg:
            short_dte = spread.short_leg_dte
            T_short = max(short_dte / 365.0, 0.001)
            short_greeks = _bs_greeks(
                underlying_price, spread.short_leg.strike, T_short, RISK_FREE_RATE, sigma,
            )

            # Update short leg delta and current value
            spread.short_leg.delta = short_greeks["delta"]
            spread.short_leg.current_value = _bs_call_price(
                underlying_price, spread.short_leg.strike, T_short, RISK_FREE_RATE, sigma,
            )

            # Net = long - short (Edge 107 — multiply each leg by its contract count)
            long_n = spread.long_leg.contracts
            short_n = spread.short_leg.contracts
            spread.net_greeks = {
                "delta": long_greeks["delta"] * long_n - short_greeks["delta"] * short_n,
                "gamma": long_greeks["gamma"] * long_n - short_greeks["gamma"] * short_n,
                "theta": long_greeks["theta"] * long_n - short_greeks["theta"] * short_n,
                "vega":  long_greeks["vega"]  * long_n - short_greeks["vega"]  * short_n,
            }
        else:
            # No short leg -- net greeks are just the LEAP (Edge 107 — scaled by contracts)
            long_n = spread.long_leg.contracts
            spread.net_greeks = {k: v * long_n for k, v in long_greeks.items()}

        return spread.net_greeks

    def calculate_max_loss(self, spread: DiagonalSpread) -> float:
        """Calculate the worst-case loss for this diagonal spread.

        Worst case: the LEAP expires worthless and all credits are lost.
        Max loss = LEAP cost basis - total credits received from all short
        leg cycles.

        This is per-share; multiply by 100 for per-contract.

        Updates spread.max_loss in place and returns the value.
        """
        max_loss_per_share = spread.long_leg.cost_basis - spread.total_credits_received
        # Edge 107 — scale by long-leg contract count (each LEAP = 100 shares)
        n = spread.long_leg.contracts
        spread.max_loss = max(max_loss_per_share, 0.0) * 100 * n
        return spread.max_loss

    # ------------------------------------------------------------------
    # Phase 0.5: Run PMCC Management Cycle
    # ------------------------------------------------------------------

    def run_cycle(self) -> List[Dict[str, Any]]:
        """Run a complete PMCC management cycle.

        Called during CCScalper.run_once() as Phase 0.5.

        Steps:
          1. For each active spread with a short leg, check if the profit
             target has been hit (buy back at 50% profit for the scalping
             loop).  If not, manage assignment risk.
          2. For each active spread without a short leg, evaluate selling a
             new short leg (this auto-reopens after a profit buy-back).
          3. Handle any urgent risk situations (auto-roll if configured).

        Returns:
            List of action dicts for logging/reporting.
        """
        if not PMCC_ENABLED:
            return []

        actions: List[Dict[str, Any]] = []

        active_spreads = self.get_active_spreads()
        if not active_spreads:
            logger.debug("PMCC run_cycle: no active spreads")
            return actions

        logger.info(
            "PMCC run_cycle: managing %d active spread(s)", len(active_spreads),
        )

        # Edge 118 — orphan buyback-tracker sweep.
        # Three of four short_leg-nulling sites in this file (expired-worthless
        # at line 1988, roll at 2224, close at 2397) leave pending_buyback_*
        # fields populated. The cleanup at line 1735 is gated behind
        # has_short_leg via check_profit_target's early-return, so once
        # short_leg=None the orphan reference persists indefinitely. This
        # sweep clears them defensively at the top of every cycle. Idempotent
        # (only clears spreads that match the orphan pattern).
        orphans_cleared = 0
        for spread in active_spreads:
            if spread.has_short_leg:
                continue
            if spread.pending_buyback_order_id is None:
                continue
            logger.info(
                "PMCC ORPHAN SWEEP: clearing stale buyback tracker on %s "
                "(short_leg=None, oid=%s) — Edge 118",
                spread.spread_id,
                (spread.pending_buyback_order_id or "?")[:8],
            )
            spread.pending_buyback_order_id = None
            spread.pending_buyback_limit = 0.0
            spread.pending_buyback_submitted_at = None
            self._clear_btc_step_state(spread)
            orphans_cleared += 1
        if orphans_cleared:
            logger.info(
                "PMCC ORPHAN SWEEP: cleared %d orphan buyback tracker(s) — Edge 118",
                orphans_cleared,
            )

        for spread in active_spreads:
            try:
                if spread.has_short_leg:
                    # --- Check profit target first (premium scalping loop) ---
                    profit_action = self.check_profit_target(spread)
                    if profit_action:
                        actions.append(profit_action)
                        # Buy-back initiated -- skip assignment risk check
                        # (next cycle will sell a new short leg once filled)
                        continue

                    # --- Manage existing short leg ---
                    risk_action = self.manage_assignment_risk(spread)
                    if risk_action:
                        actions.append(risk_action)

                        # If urgent or danger, attempt auto-roll
                        severity = risk_action.get("severity", "")
                        if severity in ("URGENT", "HIGH"):
                            roll_action = self._auto_roll(spread)
                            if roll_action:
                                actions.append(roll_action)
                else:
                    # --- Evaluate selling a new short leg ---

                    # If spread is idle after profit buy-back and auto-resell
                    # is disabled, do NOT automatically sell a new short leg.
                    # Wait for explicit instruction from David.
                    if spread.status == "idle_after_profit" and not PMCC_AUTO_RESELL:
                        logger.debug(
                            "PMCC: spread %s is idle after profit (auto-resell OFF) "
                            "-- waiting for manual trigger",
                            spread.spread_id,
                        )
                        continue

                    # Check if we have too many active short legs already
                    active_shorts = sum(
                        1 for s in active_spreads if s.has_short_leg
                    )
                    if active_shorts >= PMCC_MAX_CONCURRENT_SPREADS:
                        logger.debug(
                            "PMCC: at max concurrent short legs (%d/%d) -- skip",
                            active_shorts, PMCC_MAX_CONCURRENT_SPREADS,
                        )
                        continue

                    candidate = self.evaluate_short_leg(spread)
                    if candidate:
                        sell_action = self.sell_short_leg(spread, candidate)
                        if sell_action:
                            actions.append(sell_action)

            except Exception as e:
                logger.error(
                    "PMCC run_cycle error for %s: %s", spread.spread_id, e,
                    exc_info=True,
                )

        return actions

    def _auto_roll(self, spread: DiagonalSpread) -> Optional[Dict[str, Any]]:
        """Attempt to automatically roll a short leg that is in danger.

        Finds the next suitable short leg candidate that is further OTM and/or
        further out in time, then initiates the roll.

        Returns:
            Roll action dict, or None if no suitable roll target found.
        """
        ticker = spread.ticker
        underlying_price = self.data.get_price(ticker)
        if underlying_price is None or underlying_price <= 0:
            return None

        # The new strike should be at least 5% above the current short strike
        # AND meet all PMCC short leg criteria
        old_strike = spread.short_leg.strike
        pmcc_min_otm = PMCC_ASSIGNMENT_BUFFER_PCT   # 10% OTM floor (see evaluate_short_leg)
        min_new_strike = max(
            old_strike * 1.05,  # at least 5% higher
            underlying_price * (1 + pmcc_min_otm),
            spread.long_leg.strike + 0.01,
        )

        candidates = self._find_short_leg_candidates(
            ticker, underlying_price, min_new_strike, spread,
        )

        if not candidates:
            logger.warning(
                "PMCC auto_roll: no suitable roll targets for %s -- manual "
                "intervention may be needed",
                spread.spread_id,
            )
            return None

        best = candidates[0]

        logger.info(
            "PMCC AUTO-ROLL: %s | $%.0f -> $%.0f %s (DTE=%d, delta=%.3f) | spread=%s",
            ticker, old_strike, best["strike"], best["expiration"],
            best["dte"], best["delta"], spread.spread_id,
        )

        return self.roll_short_leg(spread, best["strike"], best["expiration"])

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_spreads(self) -> List[DiagonalSpread]:
        """Return all diagonal spreads (active and closed)."""
        return list(self._spreads)

    def get_active_spreads(self) -> List[DiagonalSpread]:
        """Return only active or idle-after-profit diagonal spreads."""
        return [s for s in self._spreads if s.status in ("active", "idle_after_profit")]

    def get_spread_by_id(self, spread_id: str) -> Optional[DiagonalSpread]:
        """Look up a spread by its unique ID."""
        for s in self._spreads:
            if s.spread_id == spread_id:
                return s
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize all PMCC state for JSON persistence."""
        return {
            "spreads": [s.to_dict() for s in self._spreads],
            "spread_counter": self._spread_counter,
        }

    def reconcile_contracts(self) -> int:
        """Edge 107 — Backfill leg.contracts from Alpaca for every active spread.

        Closes the structural gap where pre-Edge-107 state files have no
        `contracts` field (LegInfo.from_dict defaults to 1) even though
        Alpaca holds 30/10/10 contracts. Run after every state restore
        (called from from_dict()) and is safe to re-run idempotently.

        Returns the number of legs whose contracts field was corrected.
        Read-only against Alpaca; never places orders.
        """
        if getattr(self.api, "dry_run", False):
            logger.info("reconcile_contracts: dry-run mode -- skipping Alpaca lookup")
            return 0
        try:
            session = self.api._get_session()
            resp = session.get(f"{ALPACA_BASE_URL}/v2/positions", timeout=15)
            resp.raise_for_status()
            positions = resp.json()
        except Exception as e:
            logger.warning("reconcile_contracts: failed to fetch positions: %s", e)
            return 0
        by_symbol = {p.get("symbol", ""): p for p in positions}
        fixed = 0
        for spread in self._spreads:
            for leg in (spread.long_leg, spread.short_leg):
                if leg is None:
                    continue
                pos = by_symbol.get(leg.symbol)
                if not pos:
                    continue
                try:
                    broker_qty = abs(int(float(pos.get("qty", 1))))
                except (ValueError, TypeError):
                    continue
                if broker_qty > 0 and leg.contracts != broker_qty:
                    logger.warning(
                        "EDGE-107 RECONCILE: %s contracts %d -> %d (Alpaca)",
                        leg.symbol, leg.contracts, broker_qty,
                    )
                    leg.contracts = broker_qty
                    fixed += 1
            # Recompute max_loss with corrected counts
            try:
                self.calculate_max_loss(spread)
            except Exception as e:
                logger.debug("reconcile_contracts: max_loss recompute failed: %s", e)
        if fixed:
            logger.info("reconcile_contracts: corrected %d leg(s)", fixed)
        return fixed

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore PMCC state from a previously persisted dict.

        Called during CCScalper initialization to restore state from the
        shared state file.
        """
        spreads_raw = data.get("spreads", [])
        self._spreads = []
        for s_dict in spreads_raw:
            try:
                spread = DiagonalSpread.from_dict(s_dict)
                self._spreads.append(spread)
            except Exception as e:
                logger.error("Failed to restore PMCC spread: %s", e)

        self._spread_counter = data.get("spread_counter", len(self._spreads))

        logger.info(
            "PMCC state restored: %d spread(s) (%d active)",
            len(self._spreads),
            len(self.get_active_spreads()),
        )

        # Edge 107 — backfill missing/stale contracts from Alpaca
        try:
            self.reconcile_contracts()
        except Exception as e:
            logger.warning("Edge 107 reconcile_contracts failed at load: %s", e)

    # ------------------------------------------------------------------
    # Status Display
    # ------------------------------------------------------------------

    def status_report(self) -> str:
        """Generate a human-readable PMCC status report.

        Returns:
            Multi-line string suitable for logging or CLI display.
        """
        active = self.get_active_spreads()
        all_spreads = self.get_spreads()
        closed = [s for s in all_spreads if s.status == "closed"]

        lines = [
            "",
            "PMCC / DIAGONAL SPREAD STATUS:",
            f"  Total spreads: {len(all_spreads)} ({len(active)} active, "
            f"{len(closed)} closed)",
            f"  PMCC enabled: {PMCC_ENABLED}",
        ]

        if not active:
            lines.append("  No active PMCC spreads")
        else:
            for spread in active:
                # Long leg info
                lines.append(
                    f"  [{spread.ticker}] spread={spread.spread_id}"
                )
                lines.append(
                    f"    LONG: {spread.long_leg.symbol} $%.0f exp=%s | "
                    f"delta=%.2f | cost=$%.2f | value=$%.2f | DTE=%d" % (
                        spread.long_leg.strike, spread.long_leg.expiry,
                        spread.long_leg.delta, spread.long_leg.cost_basis,
                        spread.long_leg.current_value, spread.long_leg_dte,
                    )
                )

                # Short leg info
                if spread.has_short_leg:
                    s = spread.short_leg
                    lines.append(
                        f"    SHORT: {s.symbol} $%.0f exp=%s | "
                        f"delta=%.3f | credit=$%.2f | value=$%.2f | DTE=%d" % (
                            s.strike, s.expiry, s.delta,
                            s.cost_basis, s.current_value, spread.short_leg_dte,
                        )
                    )
                else:
                    lines.append("    SHORT: (none -- ready to sell)")

                # Net greeks and P&L
                ng = spread.net_greeks
                lines.append(
                    f"    NET GREEKS: delta=%.3f gamma=%.4f theta=$%.2f vega=$%.2f" % (
                        ng.get("delta", 0), ng.get("gamma", 0),
                        ng.get("theta", 0), ng.get("vega", 0),
                    )
                )
                lines.append(
                    f"    Credits collected: $%.2f | Short cycles: %d | "
                    f"Max loss: $%.0f" % (
                        spread.total_credits_received,
                        spread.num_short_cycles,
                        spread.max_loss,
                    )
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_occ_symbol(symbol: str) -> Tuple[float, str]:
        """Parse an OCC option symbol into (strike, expiry_yyyy_mm_dd).

        OCC format: TICKER + YYMMDD + C/P + strike*1000 (8 digits)
        Example: SIL270115C00060000
          -> ticker=SIL, date=2027-01-15, type=C, strike=60.00

        Returns:
            (strike, expiry) tuple.
        """
        # Find where the digits start (end of ticker)
        digit_start = 0
        for i, ch in enumerate(symbol):
            if ch.isdigit():
                digit_start = i
                break

        date_str = symbol[digit_start:digit_start + 6]  # YYMMDD
        # type_char = symbol[digit_start + 6]  # C or P
        strike_str = symbol[digit_start + 7:]  # 8 digits, strike * 1000

        yy = int(date_str[0:2])
        mm = int(date_str[2:4])
        dd = int(date_str[4:6])
        year = 2000 + yy
        expiry = f"{year:04d}-{mm:02d}-{dd:02d}"

        strike = int(strike_str) / 1000.0

        return strike, expiry

    def _find_contract_symbol(
        self, ticker: str, strike: float, expiry: str,
    ) -> Optional[str]:
        """Look up the OCC contract symbol for a given ticker/strike/expiry
        from the option chain.

        Returns:
            The contract symbol string, or None if not found.
        """
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            try:
                chain = t.option_chain(expiry)
                calls = chain.calls
                if calls is not None and len(calls) > 0:
                    match = calls[calls["strike"] == strike]
                    if len(match) > 0:
                        return str(match.iloc[0]["contractSymbol"])
            except Exception:
                pass
        except ImportError:
            pass

        # Fallback: construct the OCC symbol manually
        try:
            exp_date = datetime.strptime(expiry, "%Y-%m-%d")
            date_part = exp_date.strftime("%y%m%d")
            strike_part = f"{int(strike * 1000):08d}"
            return f"{ticker}{date_part}C{strike_part}"
        except (ValueError, TypeError):
            return None
