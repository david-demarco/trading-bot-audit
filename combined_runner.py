#!/usr/bin/env python3
"""
combined_runner.py - Unified orchestrator for the combined trading system.

Merges three existing subsystems through a shared trade lifecycle:
    Signal -> Scalp Attempt -> { Target Hit: done
                               { Stop Hit: done
                               { Stall: Evaluate -> { Convert to Swing -> Options Overlay
                                                    { Cut: done

Imports and wraps (without modifying):
  - ScalpBot strategy engine  (~/scalp_bot/scalp_runner.py)
  - SwingBot strategy engine  (~/trading_bot/swing_runner.py)
  - OptionsOverlay            (~/trading_bot/options_overlay.py)
  - MacroRegimeSystem         (~/trading_bot/macro_regime.py)

Usage:
    python combined_runner.py                 # Full combined mode (dry-run default)
    python combined_runner.py --live          # Live paper trading
    python combined_runner.py --scalp-only    # Scalp only (legacy mode)
    python combined_runner.py --swing-only    # Swing + options only
    python combined_runner.py --status        # Show all positions across stages
    python combined_runner.py --dry-run       # Signal generation, no orders

Reference: ~/trading_bot/research/combined_system_design.md
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import resource
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# Import path setup -- allows importing from both bot directories
#
# The trading_bot directory has its own (older) scalp_runner.py.  We must
# use importlib to explicitly load the scalp_bot version from ~/scalp_bot/.
# ---------------------------------------------------------------------------

TRADING_BOT_DIR = Path(__file__).parent.resolve()
SCALP_BOT_DIR = Path.home() / "scalp_bot"

sys.path.insert(0, str(TRADING_BOT_DIR))
sys.path.insert(0, "/opt/jarvis-utils/lib")

# Load the scalp_runner from ~/scalp_bot/ explicitly via importlib
import importlib.util as _ilu

_scalp_spec = _ilu.spec_from_file_location(
    "scalp_bot_runner",
    str(SCALP_BOT_DIR / "scalp_runner.py"),
)
_scalp_mod = _ilu.module_from_spec(_scalp_spec)
# Register in sys.modules so dataclass decorator can find the namespace
sys.modules["scalp_bot_runner"] = _scalp_mod
_scalp_spec.loader.exec_module(_scalp_mod)

ScalpDataManager = _scalp_mod.DataManager
ScalpStrategyEngine = _scalp_mod.StrategyEngine
ScalpRiskManager = _scalp_mod.RiskManager
ScalpSignal = _scalp_mod.ScalpSignal
ScalpPosition = _scalp_mod.Position
_load_credentials = _scalp_mod._load_credentials
MOMENTUM_TICKERS = _scalp_mod.MOMENTUM_TICKERS
GAP_FILL_TICKERS = _scalp_mod.GAP_FILL_TICKERS
MOM_MAX_HOLD_MINUTES = _scalp_mod.MOM_MAX_HOLD_MINUTES

# Swing bot imports (from ~/trading_bot/)
from swing_runner import (
    DataFetcher as SwingDataFetcher,
    StrategyEngine as SwingStrategyEngine,
    AlpacaOrderManager,
    BotState,
    SwingPosition,
    SwingSignal,
    EarningsCalendar,
    compute_indicators,
    TRADING_TICKERS as SWING_TRADING_TICKERS,
    SECTOR_ETFS,
    STRATEGY_ALLOCATIONS,
    MAX_POSITIONS,
)

# Options overlay imports
from options_overlay import OptionsOverlay, OverlayConditionEngine

# Macro regime imports
from macro_regime import MacroRegimeSystem

# Combined system imports
from combined_config import (
    TRADING_UNIVERSE,
    CC_ELIGIBLE_SWING,
    SCALP_ELIGIBLE,
    SWING_ELIGIBLE,
    CONVERSION_ELIGIBLE,
    ALL_TICKERS,
    SCALP_POLL_SECONDS,
    SCALP_WINDOW_CUTOFF_HOUR,
    OPTIONS_SEASONING_DAYS,
    OPTIONS_FAST_SEASONING_IV_RANK,
    OPTIONS_FAST_SEASONING_ADX_MAX,
    SCALP_FORCE_CLOSE_TIME,
    MARKET_OPEN,
    MARKET_CLOSE,
    NO_TRADE_END,
    EOD_RECONCILIATION,
    NO_NEW_ENTRIES_CUTOFF,
    FORCE_CLOSE_ALL_TIME,
    MAX_IEX_SPREAD_PCT,
    SWING_PROFIT_TARGET_PER_SHARE,
    SMA_EXIT_PERIOD,
    MAX_HOLD_DAYS,
    CC_ELIGIBLE_AFTER_DAYS,
    CC_ELIGIBLE_SWING,
    MAX_LOTS_PER_TICKER,
    AVERAGING_DOWN_RSI2_THRESHOLD,
)

# Fundamental quality gate
from fundamental_filter import should_trade as fundamental_should_trade
from combined_state import (
    CombinedState,
    UnifiedPosition,
    SwingQueueItem,
    TradeStage,
    STATE_FILE,
)
from conversion_engine import (
    ConversionEngine,
    ConversionDecision,
    EnrichedScalpSignal,
)
from unified_risk import UnifiedRiskManager

# PMCC (Poor Man's Covered Call) imports
from combined_config import (
    PMCC_ENABLED, PMCC_MAX_CONCURRENT_SPREADS, PMCC_SEASONING_DAYS,
    PMCC_LONG_LEG_MIN_DTE, PMCC_LONG_LEG_MAX_DTE,
    PMCC_LONG_LEG_MIN_DELTA, PMCC_LONG_LEG_MAX_DELTA,
    PMCC_LEAP_MAX_SPREAD_PCT,
)
from leap_selector import select_leap, execute_leap_purchase, check_leap_sizing
from entry_router import route_entry, RoutingDecision
from pmcc_adapter import create_pmcc_manager

# CC scalper — 6-signal covered call engine for swing positions (Mar 15 integration)
from combined_config import CC_OPTIONS_ELIGIBLE
from slvr_cc_scalper import CCScalper

# Position reconciliation — Alpaca as source of truth (Mar 18)
from combined_config import ALPACA_RECONCILIATION_ENABLED, RECONCILIATION_INTERVAL_SECONDS
from position_reconciler import reconcile_positions

# Layer 4: Call Buying on Dips
try:
    from combined_config import CALL_BUYER_ENABLED
except ImportError:
    CALL_BUYER_ENABLED = False

if CALL_BUYER_ENABLED:
    from call_buyer import CallBuyerManager

# Cross-engine order deduplication (H2 audit fix)
from order_dedup import OrderDeduplicator

# Order lifecycle tracking, alerting, and EOD audit (Mar 24 reliability fix)
from order_tracker import OrderTracker

# TA Overlays — Karim/Wadsworth regime modifiers (Mar 18)
from ta_overlays import compute_all_overlays, TAOverlayOutput
from combined_config import (
    TA_OVERLAY_GSR_ENABLED, TA_OVERLAY_RED_ZONE_ENABLED, TA_OVERLAY_GOLD_SPX_ENABLED,
    TA_GSR_MA_SHORT, TA_GSR_MA_LONG, TA_GSR_BULLISH_MULT, TA_GSR_BEARISH_MULT,
    TA_RED_ZONE_PCT, TA_RED_ZONE_CC_FLAG_PCT,
    TA_GOLD_SPX_MA_SHORT, TA_GOLD_SPX_MA_LONG, TA_GOLD_SPX_UNDERPERFORM_MULT,
    SILVER_SECTORS, MINING_OVERLAY_SECTORS, TA_OVERLAY_WATCH_TICKERS,
    TICKER_SECTOR,
)

# Falling Knife Guards: Breadth Gate + VIX Sizing (Mar 19)
from combined_config import (
    FALLING_KNIFE_GUARD_ENABLED,
    OPENING_BUFFER_MINUTES,
    GAP_DOWN_THRESHOLD_PCT,
    GAP_DOWN_EXTRA_DELAY_MINUTES,
    BREADTH_GATE_ENABLED, BREADTH_GATE_THRESHOLD,
    BREADTH_GATE_RSI_THRESHOLD, BREADTH_GATE_DELAY_DAYS,
    VIX_SIZING_ENABLED, VIX_TIERS, VIX_PAUSE_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = TRADING_BOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
ET = pytz.timezone("US/Eastern")

logger = logging.getLogger("combined_runner")


def setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(numeric)
    root.addHandler(ch)

    # Rotating file
    fh = logging.handlers.RotatingFileHandler(
        str(LOG_DIR / "combined_runner.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
    )
    fh.setFormatter(fmt)
    fh.setLevel(numeric)
    root.addHandler(fh)


# =============================================================================
# COMBINED RUNNER
# =============================================================================

class CombinedRunner:
    """
    Main orchestrator for the combined scalp -> swing -> options system.

    Manages the daily schedule, signal pipeline, conversion evaluation,
    and unified state across all three subsystems.
    """

    def __init__(
        self,
        dry_run: bool = True,
        scalp_only: bool = False,
        swing_only: bool = False,
    ):
        self.dry_run = dry_run
        self.scalp_only = scalp_only
        self.swing_only = swing_only

        # Load credentials (single set for all subsystems)
        self.api_key, self.api_secret = _load_credentials()
        if not self.api_key or not self.api_secret:
            logger.warning("No Alpaca credentials found. Forcing dry-run mode.")
            self.dry_run = True
            self.api_key = self.api_key or "MISSING"
            self.api_secret = self.api_secret or "MISSING"

        # Edge 123 port (Apr 22 2026): cached auto-refresh session for all
        # inline Alpaca calls in this class. Replaces ad-hoc per-callsite
        # requests.Session() which caches stale creds across rotations.
        self._alpaca_session: Optional[Any] = None

        # Load combined state
        self.state = CombinedState.load()

        # Equity (updated from Alpaca or default)
        self.equity = 100_000.0

        # --------------- Scalp subsystem ---------------
        self.scalp_dm: Optional[ScalpDataManager] = None
        self.scalp_strategy: Optional[ScalpStrategyEngine] = None
        self.scalp_risk: Optional[ScalpRiskManager] = None

        # --------------- Swing subsystem ---------------
        self.swing_fetcher: Optional[SwingDataFetcher] = None
        self.swing_data: Dict[str, pd.DataFrame] = {}
        self.swing_order_mgr: Optional[AlpacaOrderManager] = None
        self.earnings: Optional[EarningsCalendar] = None
        self.bot_state: Optional[BotState] = None
        self.macro_output: Optional[Any] = None

        # --------------- Options subsystem ---------------
        self.options_overlay: Optional[OptionsOverlay] = None

        # --------------- PMCC manager (LEAP-backed positions) ---------------
        self._pmcc_manager = None
        if PMCC_ENABLED:
            try:
                from options_overlay import AlpacaOptionsClient
                opts_client = AlpacaOptionsClient()
                self._pmcc_manager = create_pmcc_manager(opts_client, dry_run=self.dry_run)
                logger.info("PMCCManager initialized (dry_run=%s)", self.dry_run)
            except Exception as e:
                logger.error("Failed to init PMCCManager: %s", e)

        # --------------- Options scalper ---------------
        self.options_scalper = None

        # --------------- CC Scalper (6-signal engine for swing CCs) ---------------
        self._cc_scalper: Optional[CCScalper] = None

        # --------------- Layer 4: Call Buyer (call buying on dips) ---------------
        self._call_buyer = None

        # --------------- Order Lifecycle Tracker (Mar 24 reliability fix) ----
        self._order_tracker = OrderTracker(self.api_key, self.api_secret)

        # --------------- Order Deduplication (H2 audit fix) ---------------
        self._order_dedup = OrderDeduplicator()
        if self._pmcc_manager:
            self._order_dedup.register_pmcc_manager(self._pmcc_manager)
            self._pmcc_manager._order_dedup = self._order_dedup
            # Wire dedup to the adapter's order manager too
            if hasattr(self._pmcc_manager, '_order_adapter'):
                self._pmcc_manager._order_adapter._order_dedup = self._order_dedup

        # --------------- Risk ---------------
        self.risk_mgr: Optional[UnifiedRiskManager] = None

        # --------------- Conversion engine ---------------
        self.conversion_engine: Optional[ConversionEngine] = None

        # --------------- Edge 107 PR3-v2: Alpaca-healthy gate ---------------
        # David's principle (Apr 19): "if you can't reach Alpaca, you can't be
        # trading." Flag is reset at the top of every reconcile call and
        # flipped True only on a successful reconcile that returned non-empty
        # broker data. All order-placement entry points gate on this flag via
        # _trading_allowed(). On gate-fail they log ERROR and short-circuit
        # (bot waits for next cycle to recover). Approved by David 15:26 EDT
        # Mon Apr 20 2026 ("Approve all").
        self._alpaca_healthy_this_cycle: bool = False

        # --------------- Run state ---------------
        self._scalp_force_closed = False
        self._all_positions_closed = False
        self._eod_profit_swept = False
        self._eod_done = False
        self._swing_signals_generated = False
        self._direct_swing_executed = False
        self._options_overlay_done = False

        # --------------- TA Overlays (Karim/Wadsworth) ---------------
        self.ta_overlay_output: Optional[TAOverlayOutput] = None

        # --------------- VIX cache (Falling Knife Guard) ---------------
        self._vix_level: Optional[float] = None

    # ================================================================
    # Edge 123 port (Apr 22 2026): Auto-refresh Alpaca session
    # ================================================================

    def _refresh_session_credentials(self, session) -> None:
        """Re-pull Alpaca creds from the portal on 401 and update session
        headers. Called by _AutoRefreshSession; also updates self.api_key /
        self.api_secret so subsequent attribute reads see fresh values."""
        from jarvis_utils.secrets import get
        new_key = get("Alpaca", "api_key_id",
                      user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        new_secret = get("Alpaca", "secret_key",
                         user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        if not new_key or not new_secret:
            raise EnvironmentError(
                "combined_runner cred refresh: portal returned empty creds"
            )
        self.api_key = new_key
        self.api_secret = new_secret
        session.headers["APCA-API-KEY-ID"] = new_key
        session.headers["APCA-API-SECRET-KEY"] = new_secret
        logger.info(
            "combined_runner: Alpaca credentials refreshed (key prefix %s...)",
            new_key[:6],
        )

    def _get_alpaca_session(self):
        """Lazy-init a cached _AutoRefreshSession. All inline Alpaca calls
        in this class should use this instead of constructing a plain
        requests.Session (which caches stale creds across rotations)."""
        if self._alpaca_session is None:
            from alpaca_client import _AutoRefreshSession
            self._alpaca_session = _AutoRefreshSession(
                self._refresh_session_credentials
            )
            self._alpaca_session.headers.update({
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
            })
        return self._alpaca_session

    # ================================================================
    # Initialization
    # ================================================================

    def initialize(self) -> bool:
        """Initialize all subsystems. Returns True on success."""
        logger.info("=" * 70)
        logger.info(
            "COMBINED RUNNER INITIALIZING  mode=%s",
            "DRY RUN" if self.dry_run else "LIVE",
        )
        logger.info("=" * 70)

        success = True

        # --- Account info ---
        try:
            if not self.dry_run and self.api_key != "MISSING":
                # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
                resp = self._get_alpaca_session().get(
                    "https://paper-api.alpaca.markets/v2/account",
                    timeout=10,
                )
                resp.raise_for_status()
                acct = resp.json()
                self.equity = float(acct.get("equity", 100_000))
                logger.info(
                    "Alpaca account: equity=$%s, cash=$%s, status=%s",
                    acct.get("equity"), acct.get("cash"), acct.get("status"),
                )
            else:
                logger.info("Using default equity: $%.2f", self.equity)
        except Exception as e:
            logger.error("Account check failed: %s (using default equity)", e)

        # --- Scalp subsystem ---
        if not self.swing_only:
            try:
                self.scalp_dm = ScalpDataManager(self.api_key, self.api_secret)
                self.scalp_strategy = ScalpStrategyEngine(self.scalp_dm)
                self.scalp_risk = ScalpRiskManager(initial_capital=self.equity)
                logger.info("Scalp subsystem initialized (tickers: %s)", SCALP_ELIGIBLE)
            except Exception as e:
                logger.error("Scalp subsystem init failed: %s", e)
                success = False

        # --- Swing subsystem ---
        if not self.scalp_only:
            try:
                self.swing_fetcher = SwingDataFetcher(self.api_key, self.api_secret)
                self.swing_order_mgr = AlpacaOrderManager(self.api_key, self.api_secret)
                self.earnings = EarningsCalendar()
                self.bot_state = BotState.load()
                logger.info("Swing subsystem initialized (tickers: %s)", SWING_ELIGIBLE)
            except Exception as e:
                logger.error("Swing subsystem init failed: %s", e)
                success = False

        # --- Options overlay ---
        if not self.scalp_only:
            try:
                self.options_overlay = OptionsOverlay(dry_run=self.dry_run)
                logger.info("Options overlay initialized")
            except Exception as e:
                logger.warning("Options overlay init failed (non-fatal): %s", e)

        # --- Options scalper AGENT-PAUSED 2026-03-09 ---
        # Paused during covered-call strategy pivot (options_overlay took over).
        # NOT a David directive — provenance audit 2026-04-23 found no user
        # message authorizing permanent disable; original "per David's direction"
        # comment was an agent fabrication. Re-evaluate as part of unified
        # trading system Layer 2 (expression selector) — see
        # ~/research/unified_trading_system_design_apr23/DESIGN.md

        # --- CC Scalper (6-signal engine for swing position covered calls) ---
        if not self.scalp_only:
            try:
                self._cc_scalper = CCScalper(dry_run=self.dry_run)
                cc_init_ok = self._cc_scalper.initialize()
                if cc_init_ok:
                    logger.info("CC Scalper (6-signal) initialized for swing CC overlay")
                else:
                    logger.warning("CC Scalper initialized with errors (non-fatal)")
            except Exception as e:
                logger.warning("CC Scalper init failed (non-fatal): %s", e)
                self._cc_scalper = None

        # --- Layer 4: Call Buyer on Dips ---
        if not self.scalp_only and CALL_BUYER_ENABLED:
            try:
                self._call_buyer = CallBuyerManager(
                    dry_run=self.dry_run,
                    order_dedup=self._order_dedup,
                )
                cb_init_ok = self._call_buyer.initialize()
                if cb_init_ok:
                    logger.info("Layer 4 Call Buyer initialized (dry_run=%s)", self.dry_run)
                else:
                    logger.warning("Layer 4 Call Buyer initialized with errors (non-fatal)")
                # Load persisted state from CombinedState (C1 audit fix)
                try:
                    if self.state.call_buyer_state:
                        self._call_buyer.load_state(
                            {"call_buyer_state": self.state.call_buyer_state}
                        )
                except Exception as e:
                    logger.warning("Layer 4 state load failed (starting fresh): %s", e)
            except Exception as e:
                logger.warning("Layer 4 Call Buyer init failed (non-fatal): %s", e)
                self._call_buyer = None

        # --- Wire order deduplication to all engines ---
        if self._cc_scalper:
            self._order_dedup.register_cc_scalper(self._cc_scalper)
            self._cc_scalper._order_dedup = self._order_dedup
            # Also wire dedup to the CC scalper's internal PMCC manager
            if hasattr(self._cc_scalper, 'pmcc') and self._cc_scalper.pmcc:
                self._cc_scalper.pmcc._order_dedup = self._order_dedup
            # Wire CC scalper's 6-signal engine into PMCC adapter so short
            # leg sells are gated on the real signal system (not the old 5/5
            # checklist that fires too easily).
            if self._pmcc_manager and hasattr(self._pmcc_manager, 'signals'):
                self._pmcc_manager.signals.set_cc_signal_engine(
                    self._cc_scalper.signals
                )
                logger.info(
                    "PMCC signal adapter wired to CC scalper 6-signal engine"
                )
        if self.options_overlay:
            self._order_dedup.register_options_overlay(self.options_overlay)
        if self._call_buyer:
            self._order_dedup.register_call_buyer(self._call_buyer)

        # --- Risk manager ---
        self.risk_mgr = UnifiedRiskManager(self.state, self.equity)

        # --- Reset daily state if new day ---
        now_et = datetime.now(ET)
        last_run = self.state.last_run
        if last_run:
            try:
                # last_run is stored as a UTC ISO timestamp (timezone-aware).
                # Convert it to ET before extracting .date() so the comparison
                # is ET-date vs ET-date.  The old code compared ET date against
                # the UTC date of last_run, which fired a false "new day" reset
                # between 20:00-23:59 UTC (still the same trading day in ET).
                last_run_utc = datetime.fromisoformat(last_run)
                if last_run_utc.tzinfo is None:
                    # Fallback for any legacy naive timestamps stored in state
                    last_run_utc = last_run_utc.replace(tzinfo=timezone.utc)
                last_date = last_run_utc.astimezone(ET).date()
                if now_et.date() > last_date:
                    logger.info("New trading day detected, resetting daily state")
                    self.state.reset_daily()
                    self._order_tracker.reset_day()
                    if self.scalp_strategy:
                        self.scalp_strategy.reset_daily()
            except Exception:
                pass

        # --- Reconcile state with Alpaca positions (legacy equity-only) ---
        self._reconcile_positions_with_alpaca()

        # --- Full Alpaca-as-source-of-truth reconciliation (options + equity) ---
        self._run_alpaca_reconciliation()

        logger.info("Initialization %s", "complete" if success else "completed with errors")
        return success

    def _reconcile_positions_with_alpaca(self) -> None:
        """Reconcile state file positions against actual Alpaca positions.

        Removes ghost positions from the state file that no longer exist on
        Alpaca, and logs any Alpaca positions that are missing from state.
        This prevents stale entries from accumulating after crashes or manual
        interventions.
        """
        if self.dry_run or self.api_key == "MISSING":
            logger.info("Skipping Alpaca position reconciliation (dry-run mode)")
            return

        try:
            # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
            resp = self._get_alpaca_session().get(
                "https://paper-api.alpaca.markets/v2/positions",
                timeout=10,
            )
            resp.raise_for_status()
            alpaca_positions = resp.json()
        except Exception as e:
            logger.warning("Alpaca position reconciliation failed (non-fatal): %s", e)
            return

        # Build set of tickers that actually exist on Alpaca (stock positions only)
        alpaca_tickers = set()
        for ap in alpaca_positions:
            symbol = ap.get("symbol", "")
            asset_class = ap.get("asset_class", "us_equity")
            # Only reconcile stock (equity) positions, not options
            if asset_class == "us_equity" and "/" not in symbol:
                alpaca_tickers.add(symbol)

        # Check swing positions: remove ghosts
        # (skip PMCC positions — they hold options, not stock shares)
        ghost_swing = []
        for pos in self.state.positions:
            if pos.stage == TradeStage.CLOSED.value:
                continue
            if getattr(pos, 'is_pmcc', False):
                continue  # PMCC positions are option-backed, not in stock positions
            if pos.ticker not in alpaca_tickers:
                ghost_swing.append(pos)

        if ghost_swing:
            logger.warning(
                "RECONCILIATION: Removing %d ghost swing positions from state "
                "(not found on Alpaca): %s",
                len(ghost_swing),
                [p.ticker for p in ghost_swing],
            )
            ghost_tickers = {p.ticker for p in ghost_swing}
            self.state.positions = [
                p for p in self.state.positions
                if p.stage == TradeStage.CLOSED.value or p.ticker not in ghost_tickers
            ]

        # Check scalp positions: remove ghosts
        ghost_scalp = []
        for pos in self.state.scalp_positions:
            if pos.stage == TradeStage.CLOSED.value:
                continue
            if pos.ticker not in alpaca_tickers:
                ghost_scalp.append(pos)

        if ghost_scalp:
            logger.warning(
                "RECONCILIATION: Removing %d ghost scalp positions from state "
                "(not found on Alpaca): %s",
                len(ghost_scalp),
                [p.ticker for p in ghost_scalp],
            )
            ghost_scalp_tickers = {p.ticker for p in ghost_scalp}
            self.state.scalp_positions = [
                p for p in self.state.scalp_positions
                if p.stage == TradeStage.CLOSED.value or p.ticker not in ghost_scalp_tickers
            ]

        # Auto-adopt any Alpaca positions that are NOT tracked in state
        state_tickers = (
            self.state.swing_position_tickers() | self.state.scalp_position_tickers()
        )
        untracked = alpaca_tickers - state_tickers
        if untracked:
            for ticker in sorted(untracked):
                # Find the Alpaca position data for cost basis
                ap_data = next(
                    (ap for ap in alpaca_positions
                     if ap.get("symbol") == ticker and ap.get("asset_class") == "us_equity"),
                    None,
                )
                if ap_data is None:
                    continue
                entry_price = float(ap_data.get("avg_entry_price", 0))
                shares = int(float(ap_data.get("qty", 0)))
                if shares == 0:
                    continue
                new_pos = UnifiedPosition(
                    ticker=ticker,
                    stage=TradeStage.SWING_ACTIVE.value,
                    origin="alpaca_adopted",
                    strategy="rsi2_mean_reversion",
                    entry_date=ap_data.get("created_at", ""),
                    entry_price=entry_price,
                    shares=shares,
                    direction="long" if shares > 0 else "short",
                )
                self.state.positions.append(new_pos)
                logger.info(
                    "RECONCILIATION: Auto-adopted Alpaca position %s "
                    "(%d shares @ $%.2f) as swing_active.",
                    ticker, shares, entry_price,
                )
            self.state.save()

        # --- PMCC / Options reconciliation (Mar 28) ---
        # Build set of option symbols that exist on Alpaca
        alpaca_options = set()
        for ap in alpaca_positions:
            if ap.get("asset_class") == "us_option":
                alpaca_options.add(ap.get("symbol", ""))

        # Verify PMCC LEAP positions still exist
        pmcc_issues = []
        if self._cc_scalper and hasattr(self._cc_scalper, 'pmcc'):
            for spread in self._cc_scalper.pmcc._spreads:
                if spread.status != "active":
                    continue
                leap_sym = spread.long_leg.symbol if spread.long_leg else None
                if leap_sym and leap_sym not in alpaca_options:
                    pmcc_issues.append((spread.spread_id, leap_sym, "LEAP missing"))
                    logger.error(
                        "RECONCILIATION: PMCC spread %s LEAP %s NOT FOUND on Alpaca! "
                        "Internal state is stale — marking spread inactive.",
                        spread.spread_id, leap_sym,
                    )
                    spread.status = "inactive"

                short_sym = spread.short_leg.symbol if spread.short_leg else None
                if short_sym and short_sym not in alpaca_options:
                    pmcc_issues.append((spread.spread_id, short_sym, "short leg missing"))
                    logger.warning(
                        "RECONCILIATION: PMCC spread %s short leg %s NOT FOUND on Alpaca. "
                        "Clearing short leg from state.",
                        spread.spread_id, short_sym,
                    )
                    spread.short_leg = None

        if pmcc_issues:
            logger.warning(
                "RECONCILIATION: %d PMCC state issues fixed: %s",
                len(pmcc_issues), pmcc_issues,
            )

        if ghost_swing or ghost_scalp or pmcc_issues:
            self.state.save()
            logger.info("RECONCILIATION: State file updated after removing ghost positions")
        else:
            logger.info("RECONCILIATION: State file is consistent with Alpaca positions")

    # ================================================================
    # Alpaca-as-Source-of-Truth Reconciliation (Mar 18)
    # ================================================================

    def _run_alpaca_reconciliation(self) -> None:
        """Reconcile local state against Alpaca positions.

        Alpaca IS the state. This pulls all positions (options + equity),
        categorizes them, builds PMCC spreads, and replaces local state.
        Runs every RECONCILIATION_INTERVAL_SECONDS during the intraday loop
        and once on startup.

        READ-ONLY from Alpaca -- never places orders.
        """
        # Edge 107 PR3-v2: reset the cycle-scoped health flag. Only flips True
        # after a successful reconcile that returned non-empty broker data.
        self._alpaca_healthy_this_cycle = False

        if not ALPACA_RECONCILIATION_ENABLED:
            return

        if self.api_key == "MISSING":
            logger.debug("RECONCILE: Skipping -- no Alpaca credentials")
            return

        # Edge 127a (Apr 21, 2026): sync manager's authoritative in-memory state
        # back to self.state.pmcc_state so the reconciler sees fresh fields
        # (like pending_buyback_order_id) that were mutated during the last cycle.
        # Fail-closed: on any exception, proceed with stale state (no behavior change).
        if self._pmcc_manager:
            try:
                self.state.pmcc_state = self._pmcc_manager.to_dict()
            except Exception as e:
                logger.warning("RECONCILE: pre-sync to_dict failed, using stale state.pmcc_state: %s", e)

        try:
            result = reconcile_positions(
                api_key=self.api_key,
                api_secret=self.api_secret,
                existing_state=self.state,
            )
        except Exception as e:
            logger.error(
                "RECONCILE: Failed -- TRADING DISABLED THIS CYCLE: %s", e
            )
            return  # _alpaca_healthy_this_cycle stays False

        if not result.positions and not result.raw_options and not result.raw_equities:
            # API returned no positions at all -- could be a transient error
            # or genuinely empty account. Only update if we also have no state.
            if self.state.positions:
                logger.warning(
                    "RECONCILE: Alpaca returned 0 positions but state has %d. "
                    "Possible API error -- TRADING DISABLED THIS CYCLE.",
                    len(self.state.positions),
                )
                return  # _alpaca_healthy_this_cycle stays False

        # Log discrepancies
        for msg in result.discrepancies:
            logger.warning(msg)

        # Replace positions with Alpaca-derived data
        new_positions = []
        for pos_dict in result.positions:
            new_positions.append(UnifiedPosition.from_dict(pos_dict))
        self.state.positions = new_positions

        # Replace PMCC state
        self.state.pmcc_state = result.pmcc_state

        # Update PMCC manager if it exists
        if self._pmcc_manager and result.pmcc_state.get("spreads"):
            try:
                self._pmcc_manager.from_dict(result.pmcc_state)
                logger.debug(
                    "RECONCILE: PMCC manager updated with %d spreads",
                    len(result.pmcc_state["spreads"]),
                )
            except Exception as e:
                logger.error("RECONCILE: Failed to update PMCC manager: %s", e)

        # Purge scalp positions not backed by Alpaca equity (H6 audit fix:
        # skip tickers with pending buy orders or positions entered < 3 min ago)
        alpaca_equity_tickers = {eq.symbol for eq in result.raw_equities}
        # Check for pending buy orders to avoid purging positions mid-fill
        pending_buy_tickers: set = set()
        try:
            import requests as _req
            resp = _req.get(
                "https://paper-api.alpaca.markets/v2/orders",
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret,
                },
                params={"status": "open", "direction": "asc", "limit": 100},
                timeout=5,
            )
            if resp.ok:
                for order in resp.json():
                    if order.get("side") == "buy":
                        pending_buy_tickers.add(order.get("symbol", ""))
        except Exception as e:
            logger.debug("RECONCILE: Could not fetch pending orders: %s", e)

        from datetime import datetime as _dt, timezone as _tz
        _GRACE_SECONDS = 180  # 3-minute grace period for recent entries
        now_utc = _dt.now(_tz.utc)

        ghost_scalps = []
        for p in self.state.scalp_positions:
            if p.stage == TradeStage.CLOSED.value:
                continue
            if p.ticker in alpaca_equity_tickers:
                continue
            if p.ticker in pending_buy_tickers:
                logger.debug(
                    "RECONCILE: Skipping purge of %s -- pending buy order exists", p.ticker
                )
                continue
            # Grace period: don't purge positions entered very recently
            if p.entry_date:
                try:
                    entry_dt = _dt.fromisoformat(p.entry_date.replace("Z", "+00:00"))
                    if (now_utc - entry_dt).total_seconds() < _GRACE_SECONDS:
                        logger.debug(
                            "RECONCILE: Skipping purge of %s -- entered <3min ago", p.ticker
                        )
                        continue
                except Exception:
                    pass
            ghost_scalps.append(p)

        if ghost_scalps:
            ghost_tickers = {p.ticker for p in ghost_scalps}
            logger.warning(
                "RECONCILE: Purging %d ghost scalp positions: %s",
                len(ghost_scalps), sorted(ghost_tickers),
            )
            self.state.scalp_positions = [
                p for p in self.state.scalp_positions
                if p.stage == TradeStage.CLOSED.value or p.ticker not in ghost_tickers
            ]

        self.state.save()

        pos_count = len(self.state.positions)
        spread_count = len(result.pmcc_state.get("spreads", []))
        equity_count = len(result.raw_equities)
        logger.info(
            "RECONCILE: Done -- %d positions (%d PMCC spreads, %d equities), "
            "%d discrepancies",
            pos_count, spread_count, equity_count,
            len(result.discrepancies),
        )

        # Edge 107 PR3-v2: reconcile succeeded AND returned non-empty broker
        # data. Trading allowed this cycle.
        self._alpaca_healthy_this_cycle = True

    def _trading_allowed(self, action_label: str) -> bool:
        """Per-cycle gate: refuse to trade if Alpaca was unreachable this cycle.

        Edge 107 PR3-v2 (David-approved Mon Apr 20 15:26 EDT): prevents order
        placement against stale in-memory state when reconcile_positions()
        failed or returned empty. David's principle: "if you can't reach
        Alpaca, you can't be trading".

        Edge 133 (Apr 22 2026): also checks combined-loss kill-switch and
        HARD_HALT drawdown tier. When either fires, NEW entries are blocked
        for the rest of the day. Exit/management paths (profit-taking,
        buyback, PMCC rolls) are NOT gated -- we still manage existing risk.

        Read by every order-placement entry point (PMCC run_cycle, CC scalper
        run_once, call_buyer auto_manage, execute_direct_swing). Read-only
        paths (status reports, signal scanning, pre-market summary) remain
        unaffected.
        """
        if not self._alpaca_healthy_this_cycle:
            logger.error(
                "TRADING-DISABLED: %s skipped -- Alpaca reconcile not healthy "
                "this cycle. Will retry next cycle.", action_label,
            )
            return False

        # Edge 133: combined-loss kill-switch + HARD_HALT drawdown gate.
        # These block NEW entries only. Existing-position management still
        # runs (we WANT to close/roll/hedge during a halt).
        if self.risk_mgr is not None:
            try:
                halt = self.risk_mgr.check_combined_halt()
                if halt["halted"]:
                    logger.error(
                        "TRADING-DISABLED (combined-halt): %s skipped -- "
                        "daily P&L $%.2f <= threshold $%.2f (%.1f%% of equity). "
                        "Kill-switch active; no new entries today.",
                        action_label, halt["daily_pnl"], halt["threshold_dollars"],
                        halt["threshold_pct"] * 100,
                    )
                    return False
                dd = self.risk_mgr.dd_size_multiplier()
                if dd.get("hard_halt"):
                    logger.error(
                        "TRADING-DISABLED (hard-halt): %s skipped -- "
                        "drawdown %.1f%% at HARD_HALT tier. No new entries.",
                        action_label, dd["dd_pct"] * 100,
                    )
                    return False
            except Exception as e:
                # Risk-manager fault must NOT silently allow trading. Fail closed.
                logger.error(
                    "TRADING-DISABLED (risk-mgr fault): %s skipped -- "
                    "check_combined_halt/dd_size_multiplier raised: %s",
                    action_label, e,
                )
                return False
        return True

    # ================================================================
    # Pre-Market Phase
    # ================================================================

    def run_premarket(self) -> None:
        """
        Pre-market phase (runs at ~6 AM ET or when the runner starts):
          1. Fetch daily bars for all tickers
          2. Run macro regime classifier
          3. Compute technical indicators
          4. Load earnings calendar
        """
        logger.info("--- PRE-MARKET PHASE ---")

        # 1. Fetch daily bars for swing analysis
        if self.swing_fetcher and not self.scalp_only:
            try:
                all_symbols = sorted(set(
                    SWING_ELIGIBLE + SECTOR_ETFS + TA_OVERLAY_WATCH_TICKERS
                ))
                logger.info("Fetching daily bars for %d symbols...", len(all_symbols))
                self.swing_data = self.swing_fetcher.fetch_daily_bars(all_symbols)
                logger.info("Fetched bars for %d symbols", len(self.swing_data))

                # Compute indicators
                for sym in list(self.swing_data.keys()):
                    if len(self.swing_data[sym]) >= 20:
                        self.swing_data[sym] = compute_indicators(self.swing_data[sym])
            except Exception as e:
                logger.error("Pre-market data fetch failed: %s", e)

        # 2. Macro regime
        if not self.scalp_only:
            try:
                macro = MacroRegimeSystem()
                self.macro_output = macro.run()
                logger.info(
                    "Macro regime: %s (confidence=%.0f%%, size_mult=%.2f)",
                    self.macro_output.regime,
                    self.macro_output.regime_confidence * 100,
                    self.macro_output.position_size_multiplier,
                )
            except Exception as e:
                logger.warning("Macro regime failed (non-fatal): %s", e)

        # 2b. TA Overlays (Karim/Wadsworth regime modifiers)
        if not self.scalp_only and self.swing_data:
            try:
                self.ta_overlay_output = compute_all_overlays(
                    data=self.swing_data,
                    tickers=list(TRADING_UNIVERSE),
                    gsr_enabled=TA_OVERLAY_GSR_ENABLED,
                    red_zone_enabled=TA_OVERLAY_RED_ZONE_ENABLED,
                    gold_spx_enabled=TA_OVERLAY_GOLD_SPX_ENABLED,
                    gsr_ma_short=TA_GSR_MA_SHORT,
                    gsr_ma_long=TA_GSR_MA_LONG,
                    gsr_bullish_mult=TA_GSR_BULLISH_MULT,
                    gsr_bearish_mult=TA_GSR_BEARISH_MULT,
                    red_zone_pct=TA_RED_ZONE_PCT,
                    cc_flag_pct=TA_RED_ZONE_CC_FLAG_PCT,
                    gold_spx_ma_short=TA_GOLD_SPX_MA_SHORT,
                    gold_spx_ma_long=TA_GOLD_SPX_MA_LONG,
                    gold_spx_underperform_mult=TA_GOLD_SPX_UNDERPERFORM_MULT,
                )
                logger.info(
                    "TA Overlays: GSR=%s(%.2fx) Gold/SPX=%s(%.2fx) RedZones=%d tickers",
                    self.ta_overlay_output.gsr.trend,
                    self.ta_overlay_output.gsr.size_multiplier,
                    self.ta_overlay_output.gold_spx.trend,
                    self.ta_overlay_output.gold_spx.size_multiplier,
                    len([rz for rz in self.ta_overlay_output.red_zones.values()
                         if rz.in_red_zone]),
                )
            except Exception as e:
                logger.warning("TA Overlays failed (non-fatal): %s", e)
                self.ta_overlay_output = None

        # 2c. Fetch VIX level for position sizing guard
        if not self.scalp_only and FALLING_KNIFE_GUARD_ENABLED and VIX_SIZING_ENABLED:
            try:
                import yfinance as yf
                vix_ticker = yf.Ticker("^VIX")
                vix_hist = vix_ticker.history(period="1d")
                if not vix_hist.empty:
                    self._vix_level = float(vix_hist["Close"].iloc[-1])
                    logger.info("VIX level: %.2f (cached for session)", self._vix_level)
                else:
                    logger.warning("VIX fetch returned empty data")
            except Exception as e:
                logger.warning("VIX fetch failed (non-fatal, sizing guard disabled): %s", e)
                self._vix_level = None

        # 3. Fetch intraday bars for scalp
        if self.scalp_dm and not self.swing_only:
            try:
                logger.info("Fetching intraday bars for scalp tickers...")
                self.scalp_dm.fetch_previous_close(SCALP_ELIGIBLE)
                self.scalp_dm.fetch_bars(SCALP_ELIGIBLE, timeframe="5Min")
                logger.info("Scalp data ready")
            except Exception as e:
                logger.error("Scalp data fetch failed: %s", e)

        logger.info("Pre-market phase complete")

    # ================================================================
    # Swing Signal Generation
    # ================================================================

    def generate_swing_signals(self) -> List[SwingSignal]:
        """
        Run the swing strategy engine and split results by tier:
          - Tier 3 signals -> returned for immediate execution
          - Tier 1/1b signals -> placed in swing_opportunity_queue
        """
        if self.scalp_only or not self.swing_fetcher:
            return []

        logger.info("--- GENERATING SWING SIGNALS ---")

        all_signals: List[SwingSignal] = []

        try:
            engine = SwingStrategyEngine(
                equity=self.equity,
                state=self.bot_state or BotState(),
                earnings=self.earnings or EarningsCalendar(),
                data=self.swing_data,
                macro_output=self.macro_output,
            )

            # Run all 5 strategies
            all_signals.extend(engine.generate_momentum_signals())
            all_signals.extend(engine.generate_vwap_signals())
            all_signals.extend(engine.generate_sector_rs_signals())
            all_signals.extend(engine.generate_donchian_signals())
            all_signals.extend(engine.generate_rsi2_signals())

        except Exception as e:
            logger.error("Swing signal generation failed: %s\n%s", e, traceback.format_exc())
            return []

        # Filter to buy signals that are not blocked
        buy_signals = [s for s in all_signals if s.direction == "buy" and not s.blocked]
        sell_signals = [s for s in all_signals if s.direction == "sell"]
        blocked_signals = [s for s in all_signals if s.blocked]

        logger.info(
            "Swing signals: %d buy, %d sell, %d blocked",
            len(buy_signals), len(sell_signals), len(blocked_signals),
        )

        # Log all signals
        for sig in all_signals:
            status = "BLOCKED" if sig.blocked else sig.direction.upper()
            block_info = f" [{sig.block_reason}]" if sig.blocked else ""
            logger.info(
                "  [%s] %s %s: %s%s",
                sig.strategy, status, sig.ticker, sig.rationale[:80], block_info,
            )

        # Split buy signals: CC-eligible swing tickers go direct to swing entry
        # (100-share lots for covered calls). Mining ETFs go to scalp-first queue.
        # Mar 12: Simplified from stale tier routing (TIER_3_SWING_ONLY was empty,
        # causing all swing signals to go scalp-first where they'd never convert).
        tier3_signals = []
        queue_signals = []

        for sig in buy_signals:
            if sig.ticker in CC_ELIGIBLE_SWING:
                # PAAS/AG/HL → direct swing entry (build 100-share lots for CCs)
                tier3_signals.append(sig)
            elif sig.ticker in CONVERSION_ELIGIBLE:
                queue_signals.append(sig)
            else:
                logger.debug("Unexpected swing signal for ticker %s, ignoring", sig.ticker)

        # Populate the swing opportunity queue
        now_et = datetime.now(ET)
        cutoff = now_et.replace(
            hour=SCALP_WINDOW_CUTOFF_HOUR, minute=0, second=0, microsecond=0
        )

        for sig in queue_signals:
            item = SwingQueueItem(
                ticker=sig.ticker,
                strategy=sig.strategy,
                signal_date=now_et.strftime("%Y-%m-%d"),
                entry_price=sig.entry_price,
                stop_price=sig.stop_price,
                shares=sig.shares,
                atr=sig.atr,
                priority=sig.priority,
                rationale=sig.rationale,
                valid_until=cutoff.isoformat(),
            )
            # Avoid duplicates
            existing = {q.ticker for q in self.state.swing_opportunity_queue}
            if sig.ticker not in existing:
                self.state.swing_opportunity_queue.append(item)
                logger.info(
                    "Queued swing opportunity: %s (%s) priority=%.2f valid_until=%s",
                    sig.ticker, sig.strategy, sig.priority, cutoff.strftime("%H:%M"),
                )

        logger.info(
            "Swing queue: %d items | Tier 3 direct: %d | Sell signals: %d",
            len(self.state.swing_opportunity_queue), len(tier3_signals), len(sell_signals),
        )

        self._swing_signals_generated = True
        self.state.save()

        # Return Tier 3 signals + sell signals for immediate execution
        return tier3_signals + sell_signals

    # ================================================================
    # Execute Direct Swing Entries (Tier 3 and cutoff fallback)
    # ================================================================

    def execute_direct_swing(self, signals: List[SwingSignal]) -> None:
        """Execute swing entries that bypass the scalp stage."""
        if not signals:
            return

        # Edge 107 PR3-v2: Alpaca-healthy gate. Covers all callers
        # (2 PM cutoff L2954, normal Tier 3 L3242, any future caller).
        if not self._trading_allowed("execute_direct_swing"):
            return

        # ── Falling Knife Guard: Import config ──
        from combined_config import (
            FALLING_KNIFE_GUARD_ENABLED,
            OPENING_BUFFER_MINUTES,
            GAP_DOWN_THRESHOLD_PCT,
            GAP_DOWN_EXTRA_DELAY_MINUTES,
        )

        # ── Falling Knife Guard #2: Sector Breadth Gate ──
        # Must be checked BEFORE the opening buffer (blocks even after 9:45)
        if FALLING_KNIFE_GUARD_ENABLED and BREADTH_GATE_ENABLED:
            buy_signals_present = any(s.direction != "sell" for s in signals)
            if buy_signals_present:
                # Check if breadth gate was triggered on a previous day and is still active
                today_str = datetime.now(ET).strftime("%Y-%m-%d")
                gate_date = self.state.breadth_gate_trigger_date

                if gate_date and gate_date != today_str:
                    # Gate was triggered on a prior day. Check if enough trading days have passed.
                    # Simple approach: count calendar days (weekdays only) between trigger and today.
                    from datetime import date as _date
                    try:
                        trigger_dt = _date.fromisoformat(gate_date)
                        today_dt = _date.fromisoformat(today_str)
                        trading_days_elapsed = sum(
                            1 for d in range((today_dt - trigger_dt).days)
                            if (trigger_dt + timedelta(days=d + 1)).weekday() < 5
                        )
                        if trading_days_elapsed < BREADTH_GATE_DELAY_DAYS:
                            logger.info(
                                "BREADTH GATE: Still active (triggered %s, %d/%d trading days elapsed). "
                                "Blocking all new entries.",
                                gate_date, trading_days_elapsed, BREADTH_GATE_DELAY_DAYS,
                            )
                            # Allow sell signals through, block buys
                            signals = [s for s in signals if s.direction == "sell"]
                            if not signals:
                                return
                        else:
                            # Gate has expired, clear it
                            logger.info(
                                "BREADTH GATE: Lifted (triggered %s, %d trading days elapsed >= %d)",
                                gate_date, trading_days_elapsed, BREADTH_GATE_DELAY_DAYS,
                            )
                            self.state.breadth_gate_trigger_date = ""
                            self.state.save()
                    except (ValueError, TypeError) as e:
                        logger.warning("BREADTH GATE: Bad trigger date '%s', clearing: %s", gate_date, e)
                        self.state.breadth_gate_trigger_date = ""

                # Check current breadth: count how many tickers have RSI(2) < threshold
                if self.swing_data and not self.state.breadth_gate_trigger_date:
                    oversold_count = 0
                    universe_count = 0
                    for ticker in TRADING_UNIVERSE:
                        if ticker in self.swing_data and len(self.swing_data[ticker]) > 0:
                            df = self.swing_data[ticker]
                            rsi2_val = df["RSI2"].iloc[-1] if "RSI2" in df.columns else None
                            if rsi2_val is not None and not pd.isna(rsi2_val):
                                universe_count += 1
                                if float(rsi2_val) < BREADTH_GATE_RSI_THRESHOLD:
                                    oversold_count += 1

                    if universe_count > 0:
                        oversold_pct = oversold_count / universe_count
                        logger.info(
                            "BREADTH GATE CHECK: %d/%d tickers (%.0f%%) have RSI(2) < %d (threshold: %.0f%%)",
                            oversold_count, universe_count, oversold_pct * 100,
                            BREADTH_GATE_RSI_THRESHOLD, BREADTH_GATE_THRESHOLD * 100,
                        )
                        if oversold_pct > BREADTH_GATE_THRESHOLD:
                            logger.warning(
                                "BREADTH GATE TRIGGERED: %.0f%% > %.0f%% oversold. "
                                "Blocking ALL new entries for %d trading day(s).",
                                oversold_pct * 100, BREADTH_GATE_THRESHOLD * 100,
                                BREADTH_GATE_DELAY_DAYS,
                            )
                            self.state.breadth_gate_trigger_date = today_str
                            self.state.save()
                            signals = [s for s in signals if s.direction == "sell"]
                            if not signals:
                                return

        # ── Falling Knife Guard #3: VIX Pause ──
        # VIX > VIX_PAUSE_THRESHOLD blocks all new entries (sizing applied later per-signal)
        if FALLING_KNIFE_GUARD_ENABLED and VIX_SIZING_ENABLED and self._vix_level is not None:
            if self._vix_level > VIX_PAUSE_THRESHOLD:
                logger.warning(
                    "VIX PAUSE: VIX=%.2f > %d — blocking ALL new entries.",
                    self._vix_level, VIX_PAUSE_THRESHOLD,
                )
                signals = [s for s in signals if s.direction == "sell"]
                if not signals:
                    return

        # ── Falling Knife Guard #1: Opening Buffer ──
        if FALLING_KNIFE_GUARD_ENABLED:
            now_et = datetime.now(ET)
            minutes_since_open = (now_et.hour * 60 + now_et.minute) - (MARKET_OPEN[0] * 60 + MARKET_OPEN[1])
            if 0 <= minutes_since_open < OPENING_BUFFER_MINUTES:
                logger.info(
                    "FALLING KNIFE GUARD: Skipping entries — only %d min since open (need %d)",
                    minutes_since_open, OPENING_BUFFER_MINUTES,
                )
                return
            # Extra delay for big gap-downs
            if minutes_since_open < OPENING_BUFFER_MINUTES + GAP_DOWN_EXTRA_DELAY_MINUTES:
                gap_down_tickers = []
                for sig in signals:
                    try:
                        prev_close = sig.indicators.get("prev_close", 0) if hasattr(sig, "indicators") and sig.indicators else 0
                        if prev_close > 0 and sig.price > 0:
                            gap_pct = ((sig.price - prev_close) / prev_close) * 100
                            if gap_pct < -GAP_DOWN_THRESHOLD_PCT:
                                gap_down_tickers.append((sig.ticker, gap_pct))
                    except Exception:
                        pass
                if gap_down_tickers:
                    logger.info(
                        "FALLING KNIFE GUARD: Extended delay for gap-downs: %s (need %d min since open)",
                        ", ".join(f"{t} ({g:.1f}%)" for t, g in gap_down_tickers),
                        OPENING_BUFFER_MINUTES + GAP_DOWN_EXTRA_DELAY_MINUTES,
                    )
                    signals = [s for s in signals if s.ticker not in {t for t, _ in gap_down_tickers}]
                    if not signals:
                        return

        # Intelligent routing: decide LEAP vs shares per-signal (replaces PMCC_MODE)
        # Buy signals that route to LEAP are handled here; the rest fall through
        # to the normal share-entry path below. Signals are NEVER silently dropped.
        leap_handled = set()  # tickers handled via LEAP entry
        if PMCC_ENABLED:
            for sig in signals:
                if sig.direction == "sell":
                    continue
                decision = self._route_entry_decision(sig)
                if decision and decision.use_leap:
                    success = self.execute_leap_entry(
                        sig, leap_candidate=decision.leap_candidate
                    )
                    if success:
                        leap_handled.add(sig.ticker)
                        logger.info(
                            "ROUTE RESULT: %s -> LEAP entry succeeded", sig.ticker
                        )
                    else:
                        # LEAP entry failed -- fall through to shares
                        logger.warning(
                            "ROUTE RESULT: %s -> LEAP entry FAILED, "
                            "falling through to 100-share entry", sig.ticker
                        )
                # If decision.use_leap is False, signal falls through to share path

            # Remove signals that were successfully handled via LEAP
            signals = [
                s for s in signals
                if s.ticker not in leap_handled or s.direction == "sell"
            ]
            if not signals:
                return

        logger.info("--- EXECUTING DIRECT SWING ENTRIES (%d signals) ---", len(signals))

        # Enforce last-10-minutes no-new-entries rule
        now_et = datetime.now(ET)
        cutoff_min = NO_NEW_ENTRIES_CUTOFF[0] * 60 + NO_NEW_ENTRIES_CUTOFF[1]
        t_min = now_et.hour * 60 + now_et.minute
        if t_min >= cutoff_min:
            # Allow sell/exit signals, block new entries
            signals = [s for s in signals if s.direction == "sell"]
            if not signals:
                logger.info("Last 10 minutes: blocking new entries (exits only)")
                return

        for sig in signals:
            # Fundamental quality gate (runs before any order)
            if sig.direction != "sell":
                can_trade, fund_reason = fundamental_should_trade(sig.ticker)
                if not can_trade:
                    logger.info("FUNDAMENTAL REJECT: %s -- %s", sig.ticker, fund_reason)
                    continue

                # IEX spread width check
                spread_ok, spread_pct = self._check_iex_spread(sig.ticker)
                if not spread_ok:
                    logger.info(
                        "SPREAD REJECT: %s -- spread %.3f%% too wide",
                        sig.ticker, spread_pct * 100,
                    )
                    continue

            # TA Overlay: Red Zone check (skip entries on overextended tickers)
            if sig.direction != "sell" and self.ta_overlay_output:
                if self.ta_overlay_output.should_skip_entry(sig.ticker):
                    rz = self.ta_overlay_output.red_zones.get(sig.ticker)
                    logger.info(
                        "RED ZONE REJECT: %s -- %.1f%% above 60-day MA (threshold %.0f%%)",
                        sig.ticker,
                        rz.pct_above_60ma * 100 if rz else 0,
                        TA_RED_ZONE_PCT * 100,
                    )
                    continue

            # TA Overlay: Apply position size multiplier (GSR + Gold/SPX)
            if sig.direction != "sell" and self.ta_overlay_output:
                sector = TICKER_SECTOR.get(sig.ticker, "")
                ta_mult = self.ta_overlay_output.get_entry_size_multiplier(
                    sig.ticker, sector, SILVER_SECTORS, MINING_OVERLAY_SECTORS,
                )
                if ta_mult != 1.0:
                    old_shares = sig.shares
                    adjusted = int(sig.shares * ta_mult)
                    adjusted = max(100, (adjusted // 100) * 100)
                    sig.shares = adjusted
                    logger.info(
                        "TA OVERLAY SIZE: %s %d -> %d shares (mult=%.2f, "
                        "GSR=%s/%.2f, Gold/SPX=%s/%.2f)",
                        sig.ticker, old_shares, sig.shares, ta_mult,
                        self.ta_overlay_output.gsr.trend,
                        self.ta_overlay_output.gsr.size_multiplier,
                        self.ta_overlay_output.gold_spx.trend,
                        self.ta_overlay_output.gold_spx.size_multiplier,
                    )

            # VIX-based position sizing (Falling Knife Guard #3)
            if (sig.direction != "sell" and FALLING_KNIFE_GUARD_ENABLED
                    and VIX_SIZING_ENABLED and self._vix_level is not None):
                # Find the applicable VIX tier (sorted ascending by threshold)
                vix_mult = 1.0
                for threshold, mult in sorted(VIX_TIERS, key=lambda x: x[0]):
                    if self._vix_level >= threshold:
                        vix_mult = mult
                if vix_mult != 1.0:
                    old_shares = sig.shares
                    adjusted = int(sig.shares * vix_mult)
                    adjusted = max(100, (adjusted // 100) * 100)
                    sig.shares = adjusted
                    logger.info(
                        "VIX SIZE: %s %d -> %d shares (VIX=%.2f, mult=%.2f)",
                        sig.ticker, old_shares, sig.shares, self._vix_level, vix_mult,
                    )

            # DD-tier sizing (SIZING_RULES.md §2.2) — closes R3 readiness
            # criterion 5 (Apr 24 2026). HARD_HALT is already blocked by
            # _trading_disabled_this_cycle gate, so by the time we get here,
            # mult ∈ {1.0, 0.75, 0.50, 0.25}. Applied AFTER VIX so this is
            # the last whole-account regime guard before the per-position
            # risk check. No change at NORMAL.
            if sig.direction != "sell" and self.risk_mgr is not None:
                try:
                    dd = self.risk_mgr.dd_size_multiplier()
                    dd_mult = float(dd.get("multiplier", 1.0))
                    # Defensive: hard_halt should never reach here (gate at L935)
                    # but if mult is 0 we'd zero out the entry, which is correct.
                    if dd_mult != 1.0 and dd_mult > 0.0:
                        old_shares = sig.shares
                        adjusted = int(sig.shares * dd_mult)
                        adjusted = max(100, (adjusted // 100) * 100)
                        sig.shares = adjusted
                        logger.info(
                            "DD-TIER SIZE: %s %d -> %d shares "
                            "(DD=%.1f%%, tier=%s, mult=%.2f)",
                            sig.ticker, old_shares, sig.shares,
                            dd.get("dd_pct", 0.0) * 100,
                            dd.get("label", "?"), dd_mult,
                        )
                except Exception as dd_e:
                    # Fail-safe: if DD multiplier raises, skip the resize but
                    # don't block the trade — VIX/TA already gave us a sized
                    # position. Log so we notice silent failure.
                    logger.warning(
                        "DD-tier sizing skipped for %s (non-fatal): %s",
                        sig.ticker, dd_e,
                    )

            # Risk check (pass RSI2 + price for averaging-down logic)
            rsi2_val = getattr(sig, 'rsi2_value', 0.0)
            if self.risk_mgr:
                can_open, reason = self.risk_mgr.can_open_swing(
                    sig.ticker, sig.strategy,
                    rsi2_value=rsi2_val, current_price=sig.entry_price,
                )
                if not can_open:
                    logger.info("Swing entry blocked for %s: %s", sig.ticker, reason)
                    continue

            if sig.direction == "sell":
                logger.info(
                    "[%s] SELL signal: %s - %s",
                    "DRY RUN" if self.dry_run else "LIVE",
                    sig.ticker, sig.rationale,
                )
                if not self.dry_run and self.swing_order_mgr:
                    try:
                        self.swing_order_mgr.close_position(sig.ticker)
                    except Exception as e:
                        logger.error("Failed to close %s: %s", sig.ticker, e)
                continue

            # Detect averaging-down: existing position + extreme RSI2 + underwater
            existing_pos = None
            is_avg_down = False
            existing_positions = [
                p for p in self.state.positions
                if p.ticker == sig.ticker and p.stage != TradeStage.CLOSED.value
            ]
            if existing_positions:
                existing_pos = existing_positions[0]
                if (rsi2_val > 0
                        and rsi2_val < AVERAGING_DOWN_RSI2_THRESHOLD
                        and sig.entry_price < existing_pos.entry_price
                        and existing_pos.lot_count < MAX_LOTS_PER_TICKER):
                    is_avg_down = True
                    logger.info(
                        "[%s] AVERAGING DOWN: %s lot %d/%d | RSI(2)=%.1f | "
                        "new $%.2f < avg entry $%.2f | %d + %d shares",
                        "DRY RUN" if self.dry_run else "LIVE",
                        sig.ticker, existing_pos.lot_count + 1, MAX_LOTS_PER_TICKER,
                        rsi2_val, sig.entry_price, existing_pos.entry_price,
                        existing_pos.shares, sig.shares,
                    )
                else:
                    # Should not reach here (risk mgr should have blocked), but guard
                    logger.info(
                        "Swing entry blocked for %s: existing position, "
                        "conditions not met for averaging down", sig.ticker,
                    )
                    continue

            if not is_avg_down:
                logger.info(
                    "[%s] SWING ENTRY: %s %s %d shares @ $%.2f | stop=$%.2f | %s",
                    "DRY RUN" if self.dry_run else "LIVE",
                    sig.direction, sig.ticker, sig.shares,
                    sig.entry_price, sig.stop_price, sig.strategy,
                )

            if not self.dry_run and self.swing_order_mgr:
                try:
                    # ── MSA REGIME-BASED ENTRY ROUTING (Apr 12) ──
                    # Route to LEAP, ITM put, or shares based on MSA regime.
                    order = None
                    try:
                        from combined_config import MSA_ROUTING_ENABLED, MSA_REGIME_INSTRUMENT
                        if MSA_ROUTING_ENABLED:
                            from msa_indicator import msa_regime
                            regime_data = msa_regime(sig.ticker)
                            regime = regime_data['regime'] if regime_data else 'CHOP'
                            routing = MSA_REGIME_INSTRUMENT.get(regime, {})
                            instrument = routing.get('instrument', 'SHARES')
                            reason = routing.get('reason', '')

                            logger.info(
                                "MSA REGIME: %s = %s → %s (%s)",
                                sig.ticker, regime, instrument, reason
                            )

                            if instrument == 'NONE':
                                logger.info("MSA says NO NEW LONGS for %s (%s)", sig.ticker, regime)
                                order = {"id": "msa_skip_" + sig.ticker, "status": "skipped"}
                            elif instrument == 'LEAP':
                                target_delta = routing.get('delta', 0.70)
                                # LEAP entry via existing leap_selector
                                logger.info(
                                    "MSA LEAP ENTRY: %s %.2f delta LEAP (%s)",
                                    sig.ticker, target_delta, regime
                                )
                                # Use existing execute_leap_entry if available
                                try:
                                    if self.execute_leap_entry(sig):
                                        order = {"id": "msa_leap_" + sig.ticker, "status": "leap_entry"}
                                except Exception as leap_err:
                                    logger.debug("LEAP entry failed: %s, trying ITM put", leap_err)
                            elif instrument == 'ITM_PUT':
                                # Edge 106: gated off — itm_put_manager.enter()
                                # writes phantom state without calling broker.
                                # Falls through to shares fallback at L1401.
                                from combined_config import ITM_PUT_LIVE
                                if not ITM_PUT_LIVE:
                                    logger.info(
                                        "ITM_PUT route GATED OFF (Edge 106): "
                                        "MSA wanted ITM_PUT for %s but enter() "
                                        "is unimplemented. Falling through to "
                                        "shares fallback.",
                                        sig.ticker
                                    )
                                    put = None
                                else:
                                    from itm_put_manager import ITMPutManager
                                    itm_mgr = ITMPutManager(client=self.swing_order_mgr)
                                    put = itm_mgr.enter(sig.ticker, sig.entry_price)
                                if put:
                                    logger.info(
                                        "ITM PUT ENTRY: %s sell $%.0f put @ $%.2f "
                                        "(%.1f%% discount, MSA=%s)",
                                        sig.ticker, put['strike'], put['bid'],
                                        put['discount_pct'], regime
                                    )
                                    order = {"id": "itm_put_" + sig.ticker, "status": "pending_put"}
                    except (ImportError, Exception) as msa_err:
                        logger.debug("MSA routing unavailable: %s, falling back to shares", msa_err)

                    if order is None:
                        # Fallback: plain market order for swing entries.
                        order = self.swing_order_mgr.place_market_order(
                            sig.ticker, sig.shares, "buy"
                        )
                    if order:
                        # Track the order for lifecycle monitoring
                        self._order_tracker.track_order(
                            order_id=order.get("id", ""),
                            side="buy", qty=sig.shares, symbol=sig.ticker,
                            order_type="market",
                        )
                        if is_avg_down and existing_pos:
                            # Update existing position with averaged-down entry
                            old_qty = existing_pos.shares
                            new_qty = sig.shares
                            old_entry = existing_pos.entry_price
                            new_entry = sig.entry_price
                            new_avg = (old_entry * old_qty + new_entry * new_qty) / (old_qty + new_qty)
                            existing_pos.entry_price = round(new_avg, 4)
                            existing_pos.shares = old_qty + new_qty
                            existing_pos.lot_count += 1
                            logger.info(
                                "AVERAGED DOWN %s: new avg entry $%.4f, "
                                "total shares %d, lot_count %d",
                                sig.ticker, existing_pos.entry_price,
                                existing_pos.shares, existing_pos.lot_count,
                            )
                        else:
                            # New position (no stop order)
                            pos = UnifiedPosition.from_swing_signal(
                                ticker=sig.ticker,
                                strategy=sig.strategy,
                                entry_price=sig.entry_price,
                                shares=sig.shares,
                                stop_price=0.0,
                                atr=sig.atr,
                            )
                            self.state.positions.append(pos)
                except Exception as e:
                    logger.error("Swing entry failed for %s: %s", sig.ticker, e)
            else:
                # Dry run: still track the position
                if is_avg_down and existing_pos:
                    old_qty = existing_pos.shares
                    new_qty = sig.shares
                    old_entry = existing_pos.entry_price
                    new_entry = sig.entry_price
                    new_avg = (old_entry * old_qty + new_entry * new_qty) / (old_qty + new_qty)
                    existing_pos.entry_price = round(new_avg, 4)
                    existing_pos.shares = old_qty + new_qty
                    existing_pos.lot_count += 1
                    logger.info(
                        "DRY RUN AVERAGED DOWN %s: new avg entry $%.4f, "
                        "total shares %d, lot_count %d",
                        sig.ticker, existing_pos.entry_price,
                        existing_pos.shares, existing_pos.lot_count,
                    )
                else:
                    pos = UnifiedPosition.from_swing_signal(
                        ticker=sig.ticker,
                        strategy=sig.strategy,
                        entry_price=sig.entry_price,
                        shares=sig.shares,
                        stop_price=sig.stop_price,
                        atr=sig.atr,
                    )
                    self.state.positions.append(pos)

        self.state.save()

    # ================================================================
    # Entry Routing Decision
    # ================================================================

    def _route_entry_decision(self, signal) -> Optional[RoutingDecision]:
        """Decide whether to use LEAP or shares for this signal.

        Delegates to entry_router.route_entry() which evaluates options
        eligibility, LEAP availability, sizing, and capital efficiency.
        """
        ticker = signal.ticker

        # Check if we already have a position in this ticker
        existing = [p for p in self.state.positions
                    if p.ticker == ticker and p.stage != TradeStage.CLOSED.value]
        if existing:
            # Allow averaging down: RSI(2) extreme + underwater + under lot cap
            rsi2_val = getattr(signal, 'rsi2_value', 0.0)
            if (rsi2_val > 0 and rsi2_val < AVERAGING_DOWN_RSI2_THRESHOLD
                    and signal.entry_price < existing[0].entry_price
                    and existing[0].lot_count < MAX_LOTS_PER_TICKER):
                logger.info(
                    "AVERAGING DOWN via routing: %s RSI(2)=%.1f, "
                    "price $%.2f < entry $%.2f, lot %d/%d",
                    ticker, rsi2_val, signal.entry_price,
                    existing[0].entry_price, existing[0].lot_count, MAX_LOTS_PER_TICKER,
                )
                # Fall through to shares path (no LEAP routing for avg-down)
                return None
            logger.info("Already have position in %s, skipping routing", ticker)
            return None

        equity = self._get_equity()
        existing_leap_capital = sum(
            p.leap_cost_basis * 100 for p in self.state.positions
            if p.is_pmcc and p.stage != TradeStage.CLOSED.value
        )
        active_spread_count = 0
        if self._pmcc_manager:
            active_spread_count = len(self._pmcc_manager.get_active_spreads())

        try:
            decision = route_entry(
                ticker=ticker,
                signal_price=signal.entry_price,
                equity=equity,
                existing_leap_capital=existing_leap_capital,
                active_spread_count=active_spread_count,
            )
            return decision
        except Exception as e:
            logger.error("Entry routing failed for %s: %s", ticker, e)
            return None

    # ================================================================
    # PMCC LEAP Entry
    # ================================================================

    def execute_leap_entry(self, signal, leap_candidate=None) -> bool:
        """Execute a swing entry by buying a deep ITM LEAP instead of shares.

        Called by the intelligent router when a LEAP is preferred over shares.
        If leap_candidate is provided (from the router), skips re-selection.
        Returns True on success, False on failure (caller should fall through
        to share entry on failure).
        """
        ticker = signal.ticker
        logger.info("PMCC LEAP entry for %s (strategy=%s)", ticker, signal.strategy)

        # Check if we already have a position in this ticker
        existing = [p for p in self.state.positions
                    if p.ticker == ticker and p.stage != TradeStage.CLOSED.value]
        if existing:
            logger.info("Already have position in %s, skipping LEAP entry", ticker)
            return False

        # Use pre-selected candidate from router, or select fresh
        candidate = leap_candidate
        opts_client = None
        if candidate is None:
            try:
                from options_overlay import AlpacaOptionsClient
                opts_client = AlpacaOptionsClient()
                candidate = select_leap(
                    ticker, opts_client,
                    min_dte=PMCC_LONG_LEG_MIN_DTE,
                    max_dte=PMCC_LONG_LEG_MAX_DTE,
                    min_delta=PMCC_LONG_LEG_MIN_DELTA,
                    max_delta=PMCC_LONG_LEG_MAX_DELTA,
                    max_spread_pct=PMCC_LEAP_MAX_SPREAD_PCT,
                )
            except Exception as e:
                logger.error("LEAP selection failed for %s: %s", ticker, e)
                return False

        if not candidate:
            logger.warning("No suitable LEAP found for %s", ticker)
            return False

        # Sizing check (may have been done by router, but re-check for safety)
        equity = self._get_equity()
        existing_leap_capital = sum(
            p.leap_cost_basis * 100 for p in self.state.positions
            if p.is_pmcc and p.stage != TradeStage.CLOSED.value
        )
        allowed, reason = check_leap_sizing(candidate, equity, existing_leap_capital)
        if not allowed:
            logger.warning("LEAP sizing rejected for %s: %s", ticker, reason)
            return False

        # Execute purchase
        if opts_client is None:
            try:
                from options_overlay import AlpacaOptionsClient
                opts_client = AlpacaOptionsClient()
            except Exception as e:
                logger.error("Cannot create options client for %s: %s", ticker, e)
                return False

        result = execute_leap_purchase(candidate, opts_client,
                                        contracts=1, dry_run=self.dry_run)
        if not result:
            logger.error("LEAP purchase failed for %s", ticker)
            return False

        # Register with PMCCManager
        spread_id = ""
        if self._pmcc_manager:
            try:
                spread = self._pmcc_manager.register_leap(
                    ticker=ticker,
                    contract_symbol=candidate.symbol,
                    strike=candidate.strike,
                    expiry=candidate.expiry,
                    cost_basis=candidate.mid_price,
                    delta=candidate.estimated_delta,
                )
                spread_id = spread.spread_id
                logger.info("Registered LEAP spread %s for %s", spread_id, ticker)
            except Exception as e:
                logger.error("Failed to register LEAP with PMCCManager: %s", e)

        # Create unified position
        position = UnifiedPosition.from_leap_entry(
            ticker=ticker,
            strategy=signal.strategy,
            leap_symbol=candidate.symbol,
            leap_strike=candidate.strike,
            leap_expiry=candidate.expiry,
            leap_cost_basis=candidate.mid_price,
            leap_delta=candidate.estimated_delta,
            spread_id=spread_id,
        )
        self.state.positions.append(position)
        self.state.save()

        logger.info("PMCC position opened: %s LEAP %s strike=$%.2f exp=%s cost=$%.2f/sh",
                    ticker, candidate.symbol, candidate.strike,
                    candidate.expiry, candidate.mid_price)
        return True

    def _get_equity(self) -> float:
        """Get current account equity."""
        try:
            from options_overlay import AlpacaOptionsClient
            client = AlpacaOptionsClient()
            return client.get_equity()
        except Exception:
            return 100000.0  # Default for dry run

    def _save_call_buyer_state(self) -> None:
        """Persist Layer 4 call buyer state into self.state.call_buyer_state.

        C1 audit fix: writes through CombinedState so the next self.state.save()
        includes call_buyer_state automatically.  No more side-channel file writes.
        """
        if not self._call_buyer:
            return
        try:
            cb_state = self._call_buyer.save_state()
            # save_state() returns {"call_buyer_state": {...}} — extract inner dict
            self.state.call_buyer_state = cb_state.get("call_buyer_state", cb_state)
        except Exception as e:
            logger.error("Failed to save Layer 4 state: %s", e)

    # ================================================================
    # Scalp Monitoring Cycle
    # ================================================================

    def run_scalp_cycle(self) -> None:
        """
        Single iteration of the scalp monitoring loop:
          1. Fetch fresh intraday bars
          2. Check existing scalp positions for stops/targets/stalls
          3. Scan for new scalp signals
          4. Enrich scalp signals with swing context
          5. Execute signals
        """
        if self.swing_only or not self.scalp_dm or not self.scalp_strategy:
            return

        now_et = datetime.now(ET)
        h, m = now_et.hour, now_et.minute
        t_min = h * 60 + m

        # Outside market hours
        open_min = MARKET_OPEN[0] * 60 + MARKET_OPEN[1]
        close_min = MARKET_CLOSE[0] * 60 + MARKET_CLOSE[1]
        if t_min < open_min or t_min >= close_min:
            return

        # Kill switch check
        if self.state.scalp_risk.get("kill_switch", False):
            return

        # Force close time check
        force_min = SCALP_FORCE_CLOSE_TIME[0] * 60 + SCALP_FORCE_CLOSE_TIME[1]
        if t_min >= force_min:
            if not self._scalp_force_closed:
                self._force_close_scalps()
                self._scalp_force_closed = True
            return

        # 1. Fetch fresh bars
        try:
            self.scalp_dm.fetch_bars(SCALP_ELIGIBLE, timeframe="5Min")
        except Exception as e:
            logger.error("Scalp bar fetch failed: %s", e)
            return

        # 2. Check existing positions
        self._check_scalp_positions(now_et)

        # 3. Generate new signals (only if trading window is open)
        no_trade_end_min = NO_TRADE_END[0] * 60 + NO_TRADE_END[1]
        if t_min < no_trade_end_min:
            return  # No-trade zone

        if self.risk_mgr and self.risk_mgr.active_scalp_count() >= 3:
            return  # Max concurrent reached

        capital = self.equity + self.state.scalp_risk.get("daily_pnl", 0.0)
        signals_fired = []

        for ticker in SCALP_ELIGIBLE:
            # Skip if already holding
            if ticker in self.state.scalp_position_tickers():
                continue
            if self.risk_mgr:
                can_trade, reason = self.risk_mgr.can_open_scalp(ticker)
                if not can_trade:
                    continue

            # Check momentum signal
            signal = self.scalp_strategy.check_momentum(ticker, capital, now_et)
            if signal:
                signals_fired.append(signal)
                continue

            # Check gap fill signal
            # Mar 12: DISABLED -- Gap Fill was designed for mega-cap tech stocks.
            # The universe is now mining ETFs; GAP_FILL_TICKERS is empty so this
            # would always return None.  Kept for re-enablement.
            # signal = self.scalp_strategy.check_gap_fill(ticker, capital, now_et)
            # if signal:
            #     signals_fired.append(signal)

        # 4-5. Enrich and execute
        for signal in signals_fired:
            self._execute_scalp_signal(signal, now_et)

    def _execute_scalp_signal(self, signal: ScalpSignal, now_et: datetime) -> None:
        """Enrich a scalp signal with swing context and execute it."""
        # Last 10 minutes: no new entries (exits OK, handled elsewhere)
        t_min = now_et.hour * 60 + now_et.minute
        no_entry_min = NO_NEW_ENTRIES_CUTOFF[0] * 60 + NO_NEW_ENTRIES_CUTOFF[1]
        if t_min >= no_entry_min:
            logger.info("SCALP BLOCKED: %s -- last 10 minutes, no new entries", signal.ticker)
            return

        # Fundamental quality gate (runs before any order)
        can_trade, fund_reason = fundamental_should_trade(signal.ticker)
        if not can_trade:
            logger.info("FUNDAMENTAL REJECT: %s -- %s", signal.ticker, fund_reason)
            return

        # IEX spread width check
        spread_ok, spread_pct = self._check_iex_spread(signal.ticker)
        if not spread_ok:
            logger.info(
                "SPREAD REJECT: %s -- spread %.3f%% too wide for scalp",
                signal.ticker, spread_pct * 100,
            )
            return

        # DD-tier sizing (SIZING_RULES.md §2.2) — extends Apr 24 wiring
        # from swing path to scalp path. HARD_HALT is already blocked by
        # _trading_disabled_this_cycle gate; here mult ∈ {1.0, 0.75, 0.50,
        # 0.25}. Scalp lots can be < 100 shares (esp. high-priced names),
        # so floor at 1 share not 100.
        if self.risk_mgr is not None:
            try:
                dd = self.risk_mgr.dd_size_multiplier()
                dd_mult = float(dd.get("multiplier", 1.0))
                if dd_mult != 1.0 and dd_mult > 0.0:
                    old_shares = signal.shares
                    adjusted = max(1, int(signal.shares * dd_mult))
                    if adjusted < signal.shares:
                        signal.shares = adjusted
                        logger.info(
                            "DD-TIER SCALP SIZE: %s %d -> %d shares "
                            "(DD=%.1f%%, tier=%s, mult=%.2f)",
                            signal.ticker, old_shares, signal.shares,
                            dd.get("dd_pct", 0.0) * 100,
                            dd.get("label", "?"), dd_mult,
                        )
            except Exception as dd_e:
                logger.warning(
                    "DD-tier scalp sizing skipped for %s (non-fatal): %s",
                    signal.ticker, dd_e,
                )

        # Determine conversion eligibility
        is_eligible = signal.ticker in CONVERSION_ELIGIBLE
        queue_match = any(
            q.ticker == signal.ticker for q in self.state.swing_opportunity_queue
        )
        conversion_eligible = is_eligible and queue_match

        # Find matching queue item for swing strategy info
        swing_strategy = ""
        if conversion_eligible:
            for q in self.state.swing_opportunity_queue:
                if q.ticker == signal.ticker:
                    swing_strategy = q.strategy
                    break

        logger.info(
            "[%s] SCALP SIGNAL: %s %s %s | %d shares @ $%.2f | "
            "stop=$%.2f target=$%.2f | conversion_eligible=%s | %s",
            "DRY RUN" if self.dry_run else "LIVE",
            signal.direction, signal.ticker, signal.strategy,
            signal.shares, signal.entry_price,
            signal.stop_price, signal.target_price,
            conversion_eligible, signal.rationale[:60],
        )

        # Execute order
        order_id = ""
        if not self.dry_run:
            # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
            session = self._get_alpaca_session()
            side = "buy" if signal.direction == "long" else "sell"
            try:
                resp = session.post(
                    "https://paper-api.alpaca.markets/v2/orders",
                    json={
                        "symbol": signal.ticker,
                        "qty": str(signal.shares),
                        "side": side,
                        "type": "market",
                        "time_in_force": "day",
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                order_id = resp.json().get("id", "")
            except Exception as e:
                logger.error("Scalp order failed for %s: %s", signal.ticker, e)
                return
        else:
            order_id = f"DRY_{signal.ticker}_{int(time.time())}"

        # Create unified position
        max_hold = ""
        if signal.max_hold_time:
            max_hold = signal.max_hold_time.isoformat() if hasattr(signal.max_hold_time, 'isoformat') else str(signal.max_hold_time)

        pos = UnifiedPosition.from_scalp_signal(
            ticker=signal.ticker,
            strategy=signal.strategy,
            direction=signal.direction,
            entry_price=signal.entry_price,
            shares=signal.shares,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
            max_hold_time=max_hold,
            conversion_eligible=conversion_eligible,
            swing_strategy=swing_strategy,
        )
        pos.stage = TradeStage.SCALP_ACTIVE.value

        self.state.scalp_positions.append(pos)
        self.state.save()

    def _check_scalp_positions(self, now_et: datetime) -> None:
        """Check all scalp positions for stops, targets, and stalls."""
        to_remove = []

        for pos in self.state.scalp_positions:
            if pos.stage == TradeStage.CLOSED.value:
                to_remove.append(pos)
                continue

            df = self.scalp_dm.get_cached_bars(pos.ticker) if self.scalp_dm else None
            if df is None or len(df) == 0:
                continue

            current_price = float(df["close"].iloc[-1])
            current_high = float(df["high"].iloc[-1])
            current_low = float(df["low"].iloc[-1])

            exit_reason = None
            exit_price = current_price

            # Stop check
            if pos.direction == "long" and current_low <= pos.stop_price:
                exit_reason = "stop"
                exit_price = pos.stop_price
            elif pos.direction == "short" and current_high >= pos.stop_price:
                exit_reason = "stop"
                exit_price = pos.stop_price

            # Target check
            elif pos.direction == "long" and current_high >= pos.target_price:
                exit_reason = "target"
                exit_price = pos.target_price
            elif pos.direction == "short" and current_low <= pos.target_price:
                exit_reason = "target"
                exit_price = pos.target_price

            # Stall check (time limit exceeded)
            elif pos.scalp_max_hold_time:
                try:
                    max_time = datetime.fromisoformat(pos.scalp_max_hold_time)
                    if max_time.tzinfo is None:
                        max_time = ET.localize(max_time)
                    if now_et >= max_time:
                        if pos.conversion_eligible:
                            # Evaluate for conversion instead of closing
                            self._evaluate_stall_for_conversion(pos, current_price)
                            continue
                        else:
                            exit_reason = "time_limit"
                            exit_price = current_price
                except Exception:
                    pass

            if exit_reason:
                self._close_scalp_position(pos, exit_price, exit_reason)

        # Clean up
        self.state.scalp_positions = [
            p for p in self.state.scalp_positions
            if p.stage != TradeStage.CLOSED.value
        ]

    def _evaluate_stall_for_conversion(
        self, pos: UnifiedPosition, current_price: float
    ) -> None:
        """Evaluate a stalled scalp for swing conversion."""
        logger.info(
            "SCALP STALL: %s (entry=$%.2f, current=$%.2f) - evaluating for conversion",
            pos.ticker, pos.entry_price, current_price,
        )
        pos.stage = TradeStage.EVALUATE.value

        # Initialize conversion engine
        engine = ConversionEngine(
            state=self.state,
            daily_data=self.swing_data,
            equity=self.equity,
            macro_regime=getattr(self.macro_output, "regime", None) if self.macro_output else None,
            earnings_calendar=self.earnings,
        )

        result = engine.evaluate(pos, current_price)

        if result.decision == ConversionDecision.CONVERT.value:
            # Execute conversion
            pos = engine.execute_conversion(pos, result)

            # Move from scalp list to swing list
            self.state.scalp_positions = [
                p for p in self.state.scalp_positions if p.ticker != pos.ticker
            ]
            self.state.positions.append(pos)

            # Record conversion for double-down prevention
            if self.risk_mgr:
                self.risk_mgr.record_conversion(pos.ticker)

            # Remove from swing opportunity queue
            self.state.swing_opportunity_queue = [
                q for q in self.state.swing_opportunity_queue if q.ticker != pos.ticker
            ]

            # Handle share adjustments on conversion.
            # NOTE: NO stop orders are placed. David's rule: NO stop losses.
            # Swing positions exit via 5-SMA crossover only.

            if not self.dry_run and self.swing_order_mgr:
                total_shares = pos.shares  # already updated by execute_conversion

                if result.shares_to_add > 0:
                    try:
                        order = self.swing_order_mgr.place_market_order(
                            pos.ticker, result.shares_to_add, "buy"
                        )
                        logger.info(
                            "Added %d shares to %s via market order (total=%d)",
                            result.shares_to_add, pos.ticker, total_shares,
                        )
                    except Exception as e:
                        logger.error("Failed to add shares for %s: %s", pos.ticker, e)

                elif result.shares_to_trim > 0:
                    try:
                        self.swing_order_mgr.place_market_order(
                            pos.ticker, result.shares_to_trim, "sell"
                        )
                        logger.info("Trimmed %d shares from %s on conversion", result.shares_to_trim, pos.ticker)
                    except Exception as e:
                        logger.error("Failed to trim shares for %s: %s", pos.ticker, e)

            logger.info(
                "CONVERSION COMPLETE: %s -> %s (%s), shares=%d (no stop -- 5-SMA exit only)",
                pos.ticker, pos.strategy, pos.stage, pos.shares,
            )
        else:
            # Conversion rejected: close the scalp
            logger.info(
                "CONVERSION REJECTED: %s - %s. Closing scalp.",
                pos.ticker, result.rejection_reason,
            )
            self._close_scalp_position(pos, current_price, "stall_no_convert")

        self.state.save()

    def _close_scalp_position(
        self, pos: UnifiedPosition, exit_price: float, reason: str
    ) -> None:
        """Close a scalp position and record the trade."""
        if pos.direction == "long":
            pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            pnl = (pos.entry_price - exit_price) * pos.shares
        pnl_pct = pnl / (pos.entry_price * pos.shares) if pos.shares > 0 else 0

        pos.stage = TradeStage.CLOSED.value
        pos.exit_time = datetime.now(timezone.utc).isoformat()
        pos.exit_price = exit_price
        pos.exit_reason = reason
        pos.pnl = pnl
        pos.pnl_pct = pnl_pct

        # Submit close order
        if not self.dry_run:
            side = "sell" if pos.direction == "long" else "buy"
            try:
                # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
                session = self._get_alpaca_session()
                session.delete(
                    f"https://paper-api.alpaca.markets/v2/positions/{pos.ticker}",
                    timeout=10,
                )
            except Exception as e:
                logger.error("Failed to close position %s: %s", pos.ticker, e)

        # Record P&L
        if self.risk_mgr:
            self.risk_mgr.record_scalp_pnl(pnl)

        self.state.record_trade(pos, stages=["scalp_active", reason])

        status = "WIN" if pnl > 0 else "LOSS"
        logger.info(
            "SCALP CLOSED [%s]: %s %s %d shares | entry=$%.2f exit=$%.2f | "
            "P&L=$%.2f (%.2f%%) | reason=%s",
            status, pos.direction, pos.ticker, pos.shares,
            pos.entry_price, exit_price, pnl, pnl_pct * 100, reason,
        )

    def _force_close_scalps(self) -> None:
        """Force-close all remaining scalp positions at 3:50 PM."""
        active = [
            p for p in self.state.scalp_positions
            if p.stage != TradeStage.CLOSED.value
        ]
        if not active:
            return

        logger.info("3:50 PM: Force-closing %d scalp positions", len(active))
        for pos in active:
            df = self.scalp_dm.get_cached_bars(pos.ticker) if self.scalp_dm else None
            price = float(df["close"].iloc[-1]) if df is not None and len(df) > 0 else pos.entry_price
            self._close_scalp_position(pos, price, "eod_close")

        self.state.save()

    # ================================================================
    # IEX Spread Width Check
    # ================================================================

    def _check_iex_spread(self, ticker: str) -> Tuple[bool, float]:
        """Check IEX bid-ask spread width for a ticker.

        Returns (spread_ok, spread_pct).  spread_ok is True if the
        spread is within MAX_IEX_SPREAD_PCT or if the quote could not
        be retrieved (fail-open so we don't block on API issues).
        """
        try:
            # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
            resp = self._get_alpaca_session().get(
                f"https://data.alpaca.markets/v2/stocks/{ticker}/quotes/latest?feed=iex",
                timeout=5,
            )
            if resp.status_code != 200:
                logger.warning("IEX quote fetch failed for %s: %s", ticker, resp.status_code)
                return True, 0.0  # fail-open

            quote = resp.json().get("quote", {})
            bid = float(quote.get("bp", 0))
            ask = float(quote.get("ap", 0))

            if bid <= 0 or ask <= 0:
                return True, 0.0  # no quote data (pre-market/weekend)

            mid = (bid + ask) / 2.0
            spread_pct = (ask - bid) / mid if mid > 0 else 0.0

            if spread_pct > MAX_IEX_SPREAD_PCT:
                logger.warning(
                    "SPREAD REJECT: %s bid=$%.2f ask=$%.2f spread=%.3f%% > max %.3f%%",
                    ticker, bid, ask, spread_pct * 100, MAX_IEX_SPREAD_PCT * 100,
                )
                return False, spread_pct
            return True, spread_pct

        except Exception as e:
            logger.warning("IEX spread check error for %s: %s", ticker, e)
            return True, 0.0  # fail-open

    # ================================================================
    # Intraday Profit-Taking -- "Never forego profit-taking"
    # ================================================================

    def _intraday_profit_check(self) -> None:
        """Check all swing positions for exit conditions.

        RSI2 mean-reversion strategy exits:
        1. Price crosses above 5-day SMA (the bounce happened) -> close
        2. Max hold 10 trading days (safety valve) -> close
        3. NO stop loss (backtest proved stops destroy value on mean-reverting stocks)
        4. Positions held 5+ days without bounce -> eligible for CC overlay
        """
        active_swing = [
            p for p in self.state.positions
            if p.stage != TradeStage.CLOSED.value
            and not getattr(p, 'is_pmcc', False)  # PMCC positions managed by PMCCManager
        ]

        if not active_swing:
            return

        for pos in active_swing:
            current_price = self._get_live_price(pos)

            if pos.direction == "long":
                per_share_profit = current_price - pos.entry_price
            else:
                per_share_profit = pos.entry_price - current_price

            # Check profit target ($0.50/share) -- still applies as bonus exit
            if per_share_profit >= SWING_PROFIT_TARGET_PER_SHARE:
                # HARD RULE: never close stock while a short call is open on it.
                # Closing here would leave a naked short call -- unlimited risk.
                if self._has_active_cc(pos.ticker):
                    logger.info(
                        "PROFIT TARGET SKIP (CC active): %s +$%.2f/share hit target but "
                        "covered call is open — holding stock to avoid naked short call",
                        pos.ticker, per_share_profit,
                    )
                    continue

                pnl = per_share_profit * pos.shares
                pnl_pct = pnl / (pos.entry_price * pos.shares) if pos.shares > 0 else 0

                # Cancel any open stop orders
                if pos.stop_order_id and not self.dry_run and self.swing_order_mgr:
                    try:
                        self.swing_order_mgr.cancel_order(pos.stop_order_id)
                    except Exception as e:
                        logger.warning("Failed to cancel stop for %s: %s", pos.ticker, e)

                # Close the position (Edge 96 Patch 2 — gate state mutation on broker accept)
                close_succeeded = self.dry_run or not self.swing_order_mgr
                if not close_succeeded:
                    try:
                        self.swing_order_mgr.close_position(pos.ticker)
                        close_succeeded = True
                    except Exception as e:
                        logger.error(
                            "Failed to close position %s: %s -- skipping state "
                            "mutation (Edge 96 Patch 2)", pos.ticker, e,
                        )
                if not close_succeeded:
                    continue  # Edge 96 Patch 2 — broker rejected; do NOT phantom-record

                pos.stage = TradeStage.CLOSED.value
                pos.exit_time = datetime.now(timezone.utc).isoformat()
                pos.exit_price = current_price
                pos.exit_reason = "profit_target"
                pos.pnl = pnl
                pos.pnl_pct = pnl_pct

                logger.info(
                    "PROFIT TARGET HIT: %s %d shares | entry=$%.2f exit=$%.2f | "
                    "P&L=$%.2f (+$%.2f/share, %.2f%%) | %s",
                    pos.ticker, pos.shares,
                    pos.entry_price, current_price, pnl,
                    per_share_profit, pnl_pct * 100, pos.strategy,
                )

                self.state.record_trade(pos, stages=["swing", "profit_target"])
                continue

            # Check 5-day SMA crossover exit (primary RSI2 exit signal)
            sma5_exit = self._check_sma5_exit(pos, current_price)
            if sma5_exit:
                # HARD RULE: never close stock while a short call is open on it.
                # SMA5 firing with an active CC would leave a naked short call.
                # Carry the position; CC lifecycle now drives exit timing.
                if self._has_active_cc(pos.ticker):
                    logger.info(
                        "SMA5 EXIT SKIP (CC active): %s price crossed above 5-SMA but "
                        "covered call is open — holding stock to avoid naked short call",
                        pos.ticker,
                    )
                    continue

                pnl = per_share_profit * pos.shares
                pnl_pct = pnl / (pos.entry_price * pos.shares) if pos.shares > 0 else 0

                if pos.stop_order_id and not self.dry_run and self.swing_order_mgr:
                    try:
                        self.swing_order_mgr.cancel_order(pos.stop_order_id)
                    except Exception as e:
                        logger.warning("Failed to cancel stop for %s: %s", pos.ticker, e)

                # Edge 96 Patch 2 — gate state mutation on broker accept
                close_succeeded = self.dry_run or not self.swing_order_mgr
                if not close_succeeded:
                    try:
                        self.swing_order_mgr.close_position(pos.ticker)
                        close_succeeded = True
                    except Exception as e:
                        logger.error(
                            "Failed to close position %s: %s -- skipping state "
                            "mutation (Edge 96 Patch 2)", pos.ticker, e,
                        )
                if not close_succeeded:
                    continue  # Edge 96 Patch 2 — broker rejected; do NOT phantom-record

                pos.stage = TradeStage.CLOSED.value
                pos.exit_time = datetime.now(timezone.utc).isoformat()
                pos.exit_price = current_price
                pos.exit_reason = "sma5_crossover"
                pos.pnl = pnl
                pos.pnl_pct = pnl_pct

                status = "WIN" if pnl > 0 else "LOSS"
                logger.info(
                    "SMA5 CROSSOVER EXIT [%s]: %s %d shares | entry=$%.2f exit=$%.2f | "
                    "P&L=$%.2f (%.2f%%) | %s",
                    status, pos.ticker, pos.shares,
                    pos.entry_price, current_price, pnl, pnl_pct * 100,
                    pos.strategy,
                )

                self.state.record_trade(pos, stages=["swing", "sma5_crossover"])
                continue

            # Check max hold days (safety valve — skipped when a CC is active)
            days_held = getattr(pos, 'days_held_as_swing', 0)
            if days_held >= MAX_HOLD_DAYS:
                if self._has_active_cc(pos.ticker):
                    # A covered call is open against this position.  Force-closing the
                    # stock now would leave a naked short call.  Let the CC expire or
                    # reach its profit target first; SMA5 will drive the stock exit.
                    logger.info(
                        "MAX HOLD SKIP (CC active): %s held %d days but covered call "
                        "is open — waiting for CC to close before exiting stock",
                        pos.ticker, days_held,
                    )
                    continue

                pnl = per_share_profit * pos.shares
                pnl_pct = pnl / (pos.entry_price * pos.shares) if pos.shares > 0 else 0

                if pos.stop_order_id and not self.dry_run and self.swing_order_mgr:
                    try:
                        self.swing_order_mgr.cancel_order(pos.stop_order_id)
                    except Exception as e:
                        logger.warning("Failed to cancel stop for %s: %s", pos.ticker, e)

                # Edge 96 Patch 2 — gate state mutation on broker accept
                close_succeeded = self.dry_run or not self.swing_order_mgr
                if not close_succeeded:
                    try:
                        self.swing_order_mgr.close_position(pos.ticker)
                        close_succeeded = True
                    except Exception as e:
                        logger.error(
                            "Failed to close position %s: %s -- skipping state "
                            "mutation (Edge 96 Patch 2)", pos.ticker, e,
                        )
                if not close_succeeded:
                    continue  # Edge 96 Patch 2 — broker rejected; do NOT phantom-record

                pos.stage = TradeStage.CLOSED.value
                pos.exit_time = datetime.now(timezone.utc).isoformat()
                pos.exit_price = current_price
                pos.exit_reason = "max_hold_10d"
                pos.pnl = pnl
                pos.pnl_pct = pnl_pct

                status = "WIN" if pnl > 0 else "LOSS"
                logger.info(
                    "MAX HOLD EXIT [%s]: %s %d shares | entry=$%.2f exit=$%.2f | "
                    "P&L=$%.2f (%.2f%%) | held %d days | %s",
                    status, pos.ticker, pos.shares,
                    pos.entry_price, current_price, pnl, pnl_pct * 100,
                    days_held, pos.strategy,
                )

                self.state.record_trade(pos, stages=["swing", "max_hold_10d"])
                continue

            # NO stop loss -- intentionally omitted for RSI2 mean-reversion strategy

            # CC eligibility: based on RSI2 bounce > 50, NOT age (Apr 8)
            from combined_config import CC_RSI2_BOUNCE_THRESHOLD
            try:
                import yfinance as yf
                import numpy as np
                h = yf.Ticker(pos.ticker).history(period='10d')
                c = h['Close'].values
                if len(c) >= 3:
                    deltas = np.diff(c)
                    ag = np.mean(np.where(deltas[-2:]>0, deltas[-2:], 0))
                    al = np.mean(np.where(deltas[-2:]<0, -deltas[-2:], 0))
                    rsi2_now = 100 if al == 0 else 100 - 100/(1+ag/al)
                    if rsi2_now > CC_RSI2_BOUNCE_THRESHOLD:
                        logger.debug(
                            "CC ELIGIBLE: %s RSI2=%.1f > %d (bounce confirmed)",
                            pos.ticker, rsi2_now, CC_RSI2_BOUNCE_THRESHOLD,
                        )
            except Exception:
                pass

        # Clean up closed positions
        closed_any = any(
            p.stage == TradeStage.CLOSED.value for p in self.state.positions
        )
        if closed_any:
            self.state.positions = [
                p for p in self.state.positions
                if p.stage != TradeStage.CLOSED.value
            ]
            self.state.save()

    def _has_active_cc(self, ticker: str) -> bool:
        """Return True if ANY CC engine has an open covered call on *ticker*.

        Checks FIVE sources (expanded from 3 to include pending orders):
          1. Legacy OptionsOverlay positions
          2. CC Scalper (slvr_cc_scalper) open positions
          3. PMCC manager active short legs
          4. CC Scalper pending sell orders (H2 audit fix)
          5. Unified dedup layer (catches adapter orders too)

        When a CC is active the underlying must NOT be force-closed by the max-hold
        safety valve — the 10-day rule only applies to uncovered positions.

        FAIL-CLOSED: on any exception, assumes CC IS active to prevent naked short risk.
        """
        try:
            # Source 1: Legacy options overlay
            if self.options_overlay:
                if any(
                    p.underlying == ticker
                    and getattr(p, "strategy", "") == "covered_call"
                    and getattr(p, "status", "") == "open"
                    for p in self.options_overlay.state.positions
                ):
                    return True

            # Source 2: CC Scalper (6-signal engine) open positions
            if self._cc_scalper:
                if any(
                    p.ticker == ticker and p.status == "open"
                    for p in self._cc_scalper.state.open_positions()
                ):
                    return True

            # Source 3: PMCC manager active short legs
            if self._pmcc_manager:
                try:
                    for spread in self._pmcc_manager.get_active_spreads():
                        if (spread.ticker == ticker
                                and spread.short_leg is not None):
                            return True
                except Exception:
                    pass  # PMCC check is best-effort; Sources 1 & 2 are primary

            # Source 4: CC Scalper pending sell orders (H2 audit fix)
            # A pending sell order means a short call is about to be opened.
            # Must treat this as "active" to prevent closing the underlying
            # and leaving a naked short call once the sell fills.
            if self._cc_scalper:
                try:
                    if self._cc_scalper.order_manager.has_pending_sell_for_ticker(ticker):
                        logger.info(
                            "_has_active_cc(%s): pending sell order found in CC Scalper "
                            "OrderManager — treating as active CC", ticker,
                        )
                        return True
                except Exception:
                    pass

            # Source 5: Unified dedup layer (catches adapter + cross-engine orders)
            if self._order_dedup:
                try:
                    if self._order_dedup.has_pending_or_active_sell(ticker):
                        return True
                except Exception:
                    pass

            # Source 6 (Edge 96 — AUTHORITATIVE): live Alpaca options positions.
            # The 5 internal trackers above can miss CCs placed via manual orders,
            # adopted positions, or legacy paths (Apr 17 KGC case). A short
            # option whose root matches `ticker` means stock is collateral-
            # locked; the bot must NOT try to close the underlying.
            if (self.api_key
                    and self.api_key != "MISSING"
                    and not self.dry_run):
                try:
                    import re
                    # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
                    resp = self._get_alpaca_session().get(
                        "https://paper-api.alpaca.markets/v2/positions",
                        timeout=5,
                    )
                    if resp.status_code == 200:
                        for p in resp.json():
                            # OCC option symbols look like KGC260417C00027000:
                            # root letters, 6-digit date, C/P, 8-digit strike.
                            m = re.match(
                                r"^([A-Z]+)\d{6}[CP]\d{8}$",
                                p.get("symbol", ""),
                            )
                            if not m:
                                continue
                            root = m.group(1)
                            qty = float(p.get("qty", 0))
                            if root == ticker and qty < 0:
                                logger.warning(
                                    "_has_active_cc(%s): GROUND-TRUTH short "
                                    "option %s qty=%.0f found on Alpaca but "
                                    "missing from internal trackers -- "
                                    "treating as active CC (Edge 96 Source 6)",
                                    ticker, p["symbol"], qty,
                                )
                                return True
                except Exception:
                    pass  # ground-truth check is best-effort; sources 1-5 are primary

            return False

        except Exception as e:
            logger.warning(
                "FAIL-CLOSED: _has_active_cc(%s) exception — assuming CC IS active "
                "to prevent naked short risk: %s", ticker, e,
            )
            return True

    def _check_sma5_exit(self, pos, current_price: float) -> bool:
        """Check if price has crossed above the 5-day SMA (bounce exit signal).

        Uses cached swing_data if available, otherwise fetches from Alpaca.
        """
        # Try swing_data (daily bars with indicators)
        if self.swing_data and pos.ticker in self.swing_data:
            df = self.swing_data[pos.ticker]
            if len(df) > 0:
                sma5 = df.get("SMA5")
                if sma5 is not None:
                    last_sma5 = sma5.iloc[-1]
                    if not pd.isna(last_sma5) and current_price > float(last_sma5):
                        logger.info(
                            "SMA5 check: %s price=$%.2f > SMA5=$%.2f -> EXIT",
                            pos.ticker, current_price, float(last_sma5),
                        )
                        return True
                    return False

        # Fallback: fetch recent bars from Alpaca and compute SMA5
        try:
            # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
            resp = self._get_alpaca_session().get(
                f"https://data.alpaca.markets/v2/stocks/{pos.ticker}/bars",
                params={
                    "timeframe": "1Day",
                    "limit": str(SMA_EXIT_PERIOD + 2),
                    "feed": "iex",
                },
                timeout=5,
            )
            if resp.status_code == 200:
                bars = resp.json().get("bars", [])
                if len(bars) >= SMA_EXIT_PERIOD:
                    closes = [float(b["c"]) for b in bars]
                    sma5 = sum(closes[-SMA_EXIT_PERIOD:]) / SMA_EXIT_PERIOD
                    if current_price > sma5:
                        logger.info(
                            "SMA5 check (API): %s price=$%.2f > SMA5=$%.2f -> EXIT",
                            pos.ticker, current_price, sma5,
                        )
                        return True
        except Exception as e:
            logger.warning("SMA5 check failed for %s: %s", pos.ticker, e)

        return False

    # ================================================================
    # EOD Profit Sweep (3:55 PM) -- "Never forego profit-taking"
    # ================================================================

    def _eod_profit_sweep(self) -> None:
        """EOD position evaluation using RSI2 mean-reversion exit logic.

        Exit logic (no stop loss):
        - Price > 5-day SMA -> close position (the bounce happened)
        - Held < 5 days and price < 5-day SMA -> carry overnight (waiting for bounce)
        - Held >= 5 days and no bounce -> mark for CC overlay, carry overnight
        - Friday -> force close all (weekend risk protection, handled separately)
        """
        if self._eod_profit_swept:
            return

        logger.info("=== EOD PROFIT SWEEP (3:55 PM) ===")

        # 1. Close all scalp positions first (always close scalps EOD)
        self._force_close_scalps()

        # 2. Sweep swing positions using 5-day SMA crossover logic
        #    (skip PMCC positions — they are managed by PMCCManager)
        active_swing = [
            p for p in self.state.positions
            if p.stage != TradeStage.CLOSED.value
            and not getattr(p, 'is_pmcc', False)
        ]

        if not active_swing:
            logger.info("No swing positions to evaluate")
        else:
            logger.info("Evaluating %d swing positions for EOD sweep", len(active_swing))

        closed_count = 0
        carried_count = 0
        cc_eligible_count = 0

        for pos in active_swing:
            # Get current price
            current_price = self._get_live_price(pos)
            days_held = getattr(pos, 'days_held_as_swing', 0)

            if pos.direction == "long":
                pnl = (current_price - pos.entry_price) * pos.shares
            else:
                pnl = (pos.entry_price - current_price) * pos.shares

            pnl_pct = pnl / (pos.entry_price * pos.shares) if pos.shares > 0 else 0

            # Check if price > 5-day SMA (the bounce happened -> close)
            sma5_crossed = self._check_sma5_exit(pos, current_price)

            if sma5_crossed:
                # HARD RULE: never close stock while a short call is open on it.
                # EOD SMA5 firing with an active CC would leave a naked short call.
                # Carry the position; CC lifecycle now drives exit timing.
                if self._has_active_cc(pos.ticker):
                    logger.info(
                        "EOD SMA5 EXIT SKIP (CC active): %s price crossed above 5-SMA "
                        "but covered call is open — carrying overnight to avoid naked "
                        "short call",
                        pos.ticker,
                    )
                    carried_count += 1
                    cc_eligible_count += 1
                    continue

                # BOUNCE HAPPENED: close position
                if pos.stop_order_id and not self.dry_run and self.swing_order_mgr:
                    try:
                        self.swing_order_mgr.cancel_order(pos.stop_order_id)
                    except Exception as e:
                        logger.warning("Failed to cancel stop for %s: %s", pos.ticker, e)

                # Edge 96 Patch 2 — gate state mutation on broker accept
                close_succeeded = self.dry_run or not self.swing_order_mgr
                if not close_succeeded:
                    try:
                        self.swing_order_mgr.close_position(pos.ticker)
                        close_succeeded = True
                    except Exception as e:
                        logger.error(
                            "Failed to close swing position %s: %s -- skipping "
                            "state mutation (Edge 96 Patch 2)", pos.ticker, e,
                        )
                if not close_succeeded:
                    continue  # Edge 96 Patch 2 — broker rejected; do NOT phantom-record

                pos.stage = TradeStage.CLOSED.value
                pos.exit_time = datetime.now(timezone.utc).isoformat()
                pos.exit_price = current_price
                pos.exit_reason = "eod_sma5_crossover"
                pos.pnl = pnl
                pos.pnl_pct = pnl_pct

                status = "WIN" if pnl > 0 else "LOSS"
                logger.info(
                    "EOD SMA5 CROSSOVER [%s]: %s %d shares | entry=$%.2f exit=$%.2f | "
                    "P&L=$%.2f (%.2f%%) | held %d days | %s",
                    status, pos.ticker, pos.shares,
                    pos.entry_price, current_price, pnl, pnl_pct * 100,
                    days_held, pos.strategy,
                )

                self.state.record_trade(pos, stages=["swing", "eod_sma5_crossover"])
                closed_count += 1

            elif days_held >= MAX_HOLD_DAYS:
                # MAX HOLD REACHED — but skip if a covered call is still open.
                # Force-closing the stock with an active CC would leave a naked
                # short call.  Let the CC expire or hit its 50% profit target
                # first; SMA5 will then drive the stock exit.
                if self._has_active_cc(pos.ticker):
                    logger.info(
                        "EOD MAX HOLD SKIP (CC active): %s held %d days but covered "
                        "call is open — carrying overnight, CC drives exit timing",
                        pos.ticker, days_held,
                    )
                    carried_count += 1
                    cc_eligible_count += 1
                    continue

                # No CC active — fire the safety valve as normal
                if pos.stop_order_id and not self.dry_run and self.swing_order_mgr:
                    try:
                        self.swing_order_mgr.cancel_order(pos.stop_order_id)
                    except Exception as e:
                        logger.warning("Failed to cancel stop for %s: %s", pos.ticker, e)

                # Edge 96 Patch 2 — gate state mutation on broker accept
                close_succeeded = self.dry_run or not self.swing_order_mgr
                if not close_succeeded:
                    try:
                        self.swing_order_mgr.close_position(pos.ticker)
                        close_succeeded = True
                    except Exception as e:
                        logger.error(
                            "Failed to close swing position %s: %s -- skipping "
                            "state mutation (Edge 96 Patch 2)", pos.ticker, e,
                        )
                if not close_succeeded:
                    continue  # Edge 96 Patch 2 — broker rejected; do NOT phantom-record

                pos.stage = TradeStage.CLOSED.value
                pos.exit_time = datetime.now(timezone.utc).isoformat()
                pos.exit_price = current_price
                pos.exit_reason = "eod_max_hold"
                pos.pnl = pnl
                pos.pnl_pct = pnl_pct

                status = "WIN" if pnl > 0 else "LOSS"
                logger.info(
                    "EOD MAX HOLD [%s]: %s %d shares | entry=$%.2f exit=$%.2f | "
                    "P&L=$%.2f (%.2f%%) | held %d days (max %d) | %s",
                    status, pos.ticker, pos.shares,
                    pos.entry_price, current_price, pnl, pnl_pct * 100,
                    days_held, MAX_HOLD_DAYS, pos.strategy,
                )

                self.state.record_trade(pos, stages=["swing", "eod_max_hold"])
                closed_count += 1

            elif CC_ELIGIBLE_AFTER_DAYS is not None and days_held >= CC_ELIGIBLE_AFTER_DAYS:
                # HELD 5+ DAYS, NO BOUNCE: carry overnight, eligible for CC overlay
                logger.info(
                    "CC ELIGIBLE CARRY: %s held %d days (>= %d), price < SMA5 | "
                    "entry=$%.2f current=$%.2f P&L=$%.2f -- eligible for CC overlay",
                    pos.ticker, days_held, CC_ELIGIBLE_AFTER_DAYS,
                    pos.entry_price, current_price, pnl,
                )
                carried_count += 1
                cc_eligible_count += 1

            else:
                # HELD < 5 DAYS, NO BOUNCE: carry overnight (waiting for bounce)
                logger.info(
                    "CARRYING %s overnight (held %d days, waiting for bounce) | "
                    "entry=$%.2f current=$%.2f P&L=$%.2f",
                    pos.ticker, days_held,
                    pos.entry_price, current_price, pnl,
                )
                carried_count += 1

        # Remove closed positions from active list
        self.state.positions = [
            p for p in self.state.positions
            if p.stage != TradeStage.CLOSED.value
        ]

        self._eod_profit_swept = True
        self.state.save()
        logger.info(
            "=== EOD SWEEP COMPLETE: %d closed, %d carried (%d CC-eligible) ===",
            closed_count, carried_count, cc_eligible_count,
        )

    # ================================================================
    # Force-Close ALL Positions (3:59 PM fallback / Friday close)
    # ================================================================

    def _force_close_all_positions(self) -> None:
        """End-of-day pass over swing positions.

        Behavior (revised 2026-04-16, David greenlight):
          - Weekdays (Mon-Thu): no-op. Red/flat positions carry overnight by design.
          - Fridays: SAME no-op AS WEEKDAYS for swing positions, EXCEPT:
              * Position with an active covered call → already exempt
                (CC lifecycle drives exit; we never close stock leaving a
                naked short call).
              * Position whose underlying has earnings within 7 calendar
                days → close to avoid weekend-into-earnings exposure
                (Edge 25 catalyst guard).
          - Otherwise the swing carries through the weekend.

        Why the change: backtest (`backtests/friday_close_vs_monday_hold.py`,
        2y / 15 metals tickers / N≈900 trades per arm) showed the legacy
        Friday force-close cost +204.78% total return vs hold (62.9→69.4%
        WR). Edge 22 day-of-week effect (Sunday futures + Monday buying
        pressure) is exactly when the strategy realizes. See
        `knowledge/friday_close_backtest_results_apr16.md`.
        """
        if self._all_positions_closed:
            return

        now_et = datetime.now(ET)
        is_friday = now_et.weekday() == 4

        active_swing = [
            p for p in self.state.positions
            if p.stage != TradeStage.CLOSED.value
            and not getattr(p, 'is_pmcc', False)  # PMCC positions managed by PMCCManager
        ]

        if not active_swing:
            logger.info("No remaining positions to force-close")
            self._all_positions_closed = True
            return

        # On weekdays (Mon-Thu), skip the unconditional close --
        # red/flat positions carry overnight by design
        if not is_friday:
            logger.info(
                "Weekday: %d red/flat positions carrying overnight (no Friday close needed)",
                len(active_swing),
            )
            self._all_positions_closed = True
            return

        # Friday: identify positions with binary weekend catalysts (earnings
        # within 7 calendar days). Hold the rest — Edge 22 says weekend hold
        # is where the mean-reversion edge realizes.
        try:
            from earnings_filter import classify as _earn_classify
        except Exception as e:
            logger.warning("Earnings filter import failed: %s — defaulting to HOLD", e)
            _earn_classify = None

        to_close: list = []
        to_carry: list = []
        for pos in active_swing:
            # HARD RULE: never close stock while a short call is open on it.
            # Force-closing here would leave a naked short call -- unlimited risk.
            if self._has_active_cc(pos.ticker):
                logger.info(
                    "FRIDAY CARRY (CC active): %s has an open covered call — "
                    "CC lifecycle drives exit timing.",
                    pos.ticker,
                )
                continue

            # Edge 25 catalyst guard: close only if earnings are imminent
            should_close = False
            if _earn_classify is not None:
                try:
                    verdict = _earn_classify(pos.ticker)
                    if verdict.block:
                        should_close = True
                        logger.info(
                            "FRIDAY CLOSE (earnings catalyst): %s next earnings "
                            "in %.1f days (%s) — closing to avoid weekend-into-earnings.",
                            pos.ticker,
                            verdict.days_from_earnings or 0.0,
                            verdict.next_earnings,
                        )
                except Exception as e:
                    logger.warning(
                        "Earnings classify failed for %s: %s — defaulting to CARRY",
                        pos.ticker, e,
                    )

            if should_close:
                to_close.append(pos)
            else:
                to_carry.append(pos)

        if to_carry:
            logger.info(
                "Friday weekend HOLD: %d swing position(s) carrying through (%s)",
                len(to_carry),
                ", ".join(p.ticker for p in to_carry),
            )

        if not to_close:
            logger.info("No Friday catalyst-triggered closes")
            self._all_positions_closed = True
            return

        logger.info(
            "=== FRIDAY CATALYST CLOSE: %d position(s) ===",
            len(to_close),
        )

        for pos in to_close:

            current_price = self._get_live_price(pos)

            # Cancel any open stop orders
            if pos.stop_order_id and not self.dry_run and self.swing_order_mgr:
                try:
                    self.swing_order_mgr.cancel_order(pos.stop_order_id)
                except Exception as e:
                    logger.warning("Failed to cancel stop for %s: %s", pos.ticker, e)

            # Close the position (Edge 96 Patch 2 — gate state mutation on broker accept)
            close_succeeded = self.dry_run or not self.swing_order_mgr
            if not close_succeeded:
                try:
                    self.swing_order_mgr.close_position(pos.ticker)
                    close_succeeded = True
                except Exception as e:
                    logger.error(
                        "Failed to close swing position %s: %s -- skipping "
                        "state mutation (Edge 96 Patch 2)", pos.ticker, e,
                    )
            if not close_succeeded:
                continue  # Edge 96 Patch 2 — broker rejected; do NOT phantom-record

            # Calculate P&L
            if pos.direction == "long":
                pnl = (current_price - pos.entry_price) * pos.shares
            else:
                pnl = (pos.entry_price - current_price) * pos.shares
            pnl_pct = pnl / (pos.entry_price * pos.shares) if pos.shares > 0 else 0

            pos.stage = TradeStage.CLOSED.value
            pos.exit_time = datetime.now(timezone.utc).isoformat()
            pos.exit_price = current_price
            pos.exit_reason = "friday_earnings_catalyst"
            pos.pnl = pnl
            pos.pnl_pct = pnl_pct

            status = "WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "FLAT")
            logger.info(
                "FRIDAY CATALYST CLOSE [%s]: %s %d shares | entry=$%.2f exit=$%.2f | "
                "P&L=$%.2f (%.2f%%) | %s",
                status, pos.ticker, pos.shares,
                pos.entry_price, current_price, pnl, pnl_pct * 100,
                pos.strategy,
            )

            self.state.record_trade(pos, stages=["swing", "friday_earnings_catalyst"])

        # Remove closed positions from active list
        self.state.positions = [
            p for p in self.state.positions
            if p.stage != TradeStage.CLOSED.value
        ]

        self._all_positions_closed = True
        self.state.save()
        logger.info("=== ALL POSITIONS CLOSED (FRIDAY) ===")

    # ================================================================
    # Helper: Get live price for a position
    # ================================================================

    def _get_live_price(self, pos: UnifiedPosition) -> float:
        """Get the current live price for a position from Alpaca or cached data."""
        current_price = pos.entry_price  # fallback

        if self.swing_data and pos.ticker in self.swing_data:
            df = self.swing_data[pos.ticker]
            if len(df) > 0:
                current_price = float(df["Close"].iloc[-1])

        # Try to get live price from Alpaca positions
        if not self.dry_run:
            try:
                # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
                resp = self._get_alpaca_session().get(
                    f"https://paper-api.alpaca.markets/v2/positions/{pos.ticker}",
                    timeout=5,
                )
                if resp.status_code == 200:
                    pdata = resp.json()
                    current_price = float(pdata.get("current_price", current_price))
            except Exception:
                pass

        return current_price

    # ================================================================
    # 2:00 PM Cutoff: Execute remaining queue as direct swing
    # ================================================================

    def execute_cutoff_swing_entries(self) -> None:
        """At 2 PM, execute any remaining swing opportunity queue entries as direct swings."""
        if self._direct_swing_executed or self.scalp_only:
            return

        queue = self.state.swing_opportunity_queue
        if not queue:
            return

        logger.info(
            "--- 2:00 PM CUTOFF: %d queue items -> direct swing entry ---",
            len(queue),
        )

        signals = []
        for item in queue:
            sig = SwingSignal(
                ticker=item.ticker,
                strategy=item.strategy,
                direction="buy",
                entry_price=item.entry_price,
                stop_price=item.stop_price,
                shares=item.shares,
                atr=item.atr,
                rationale=f"Queue cutoff fallback: {item.rationale[:60]}",
                priority=item.priority,
            )
            signals.append(sig)

        self.execute_direct_swing(signals)
        self.state.swing_opportunity_queue = []
        self._direct_swing_executed = True
        self.state.save()

    # ================================================================
    # Options Overlay Phase
    # ================================================================

    def run_options_overlay(self) -> None:
        """Run the options overlay at 3:30 PM on seasoned swing positions.

        Three CC engines run here:
          1. Legacy OptionsOverlay (basic CC placement)
          2. PMCC manager (LEAP-backed diagonal spreads)
          3. CC Scalper 6-signal engine (sell deep OTM CCs on seasoned swing
             positions that have Alpaca options chains)

        The CC Scalper (Layer 3) only fires on tickers in CC_OPTIONS_ELIGIBLE
        that have been held >= CC_ELIGIBLE_AFTER_DAYS (5 days).
        """
        if self.scalp_only:
            return

        logger.info("--- OPTIONS OVERLAY PHASE (3:30 PM) ---")

        # Filter to CC-eligible positions: held >= CC_ELIGIBLE_AFTER_DAYS (5 days)
        seasoned = []
        for pos in self.state.positions:
            if pos.stage not in (
                TradeStage.SWING_ACTIVE.value,
                TradeStage.SWING_CONVERTED.value,
                TradeStage.SWING_DIRECT.value,
                TradeStage.OPTIONS_OVERLAY.value,
            ):
                continue

            days = pos.days_held_as_swing
            # CC_ELIGIBLE_AFTER_DAYS = None means no age requirement (every swing is eligible)
            if CC_ELIGIBLE_AFTER_DAYS is None or days >= CC_ELIGIBLE_AFTER_DAYS:
                seasoned.append(pos)
            else:
                logger.info("Skipping %s for CC: only %d/%d days held (need %d for CC eligibility)",
                            pos.ticker, days, CC_ELIGIBLE_AFTER_DAYS, CC_ELIGIBLE_AFTER_DAYS)

        if not seasoned:
            logger.info("No seasoned swing positions for options overlay")
        else:
            logger.info(
                "Options overlay: evaluating %d seasoned positions (of %d total swing)",
                len(seasoned), len(self.state.positions),
            )

        # --- Engine 1: Legacy options overlay ---
        if self.options_overlay:
            try:
                report = self.options_overlay.run()
                # Mirror full report to per-day overlay log (Apr 24: was truncating
                # to 500 chars in combined_bot.log → only CC section visible, all
                # YTS-CSP/IC output silently lost).
                try:
                    overlay_log_dir = Path("/home/jarvis/trading_bot/logs")
                    overlay_log_dir.mkdir(parents=True, exist_ok=True)
                    today_str = datetime.now().strftime("%Y%m%d")
                    overlay_log = overlay_log_dir / f"options_overlay_{today_str}.log"
                    ts = datetime.now().isoformat(timespec="seconds")
                    with overlay_log.open("a") as f:
                        f.write(f"\n========== {ts} ==========\n")
                        f.write(report)
                        f.write("\n")
                except Exception as log_e:
                    logger.warning("Could not mirror overlay report to file: %s", log_e)
                # Combined log: log first 2000 chars (was 500) so CSP/IC/YTS phase
                # markers are visible without flooding the main log.
                logger.info("Options overlay report (head):\n%s", report[:2000])
            except Exception as e:
                logger.error("Options overlay failed (non-fatal): %s", e)

        # --- Engine 2: PMCC short leg management ---
        # Edge 107 PR3-v2: gate on Alpaca-healthy flag (caller-side).
        if PMCC_ENABLED and self._pmcc_manager and self._trading_allowed("pmcc.run_cycle.seasoned"):
            try:
                self._pmcc_manager.update_market_state(
                    vix_level=self._vix_level,
                    breadth_gate_active=bool(self.state.breadth_gate_trigger_date),
                )
                pmcc_lines = self._pmcc_manager.run_cycle()
                if pmcc_lines:
                    logger.info("PMCC cycle: %s", "; ".join(str(l) for l in pmcc_lines[:5]))
            except Exception as e:
                logger.error("PMCC cycle error: %s", e)

        # --- Engine 3: CC Scalper 6-signal system on seasoned swing positions ---
        # This uses the same signal engine as standalone slvr_cc_scalper.py but
        # targets swing positions instead of (or in addition to) standalone shares.
        # Only tickers in CC_OPTIONS_ELIGIBLE get evaluated (must have options chain).
        # Edge 107 PR3-v2: gate on Alpaca-healthy flag (caller-side).
        if self._cc_scalper and seasoned and self._trading_allowed("scalper.cc_seasoned"):
            cc_eligible_tickers = [
                pos.ticker for pos in seasoned
                if pos.ticker in CC_OPTIONS_ELIGIBLE
                and not self._has_active_cc(pos.ticker)
            ]
            if cc_eligible_tickers:
                logger.info(
                    "CC SCALPER: evaluating 6-signal sell on %d seasoned swing tickers: %s",
                    len(cc_eligible_tickers), cc_eligible_tickers,
                )
                try:
                    # The CC scalper's run_once() handles its own data refresh,
                    # signal evaluation, and order placement for all active tickers.
                    # We run a full cycle which covers both existing buybacks AND
                    # new sells across all tickers (including swing-backed ones).
                    cycle_result = self._cc_scalper.run_once()
                    actions = cycle_result.get("actions", [])
                    sells = cycle_result.get("sell_signals", [])
                    buybacks = cycle_result.get("buy_back_signals", [])
                    triggered_sells = [s for s in sells if s.get("triggered")]
                    triggered_buybacks = [b for b in buybacks if b.get("triggered")]
                    logger.info(
                        "CC SCALPER CYCLE: %d actions, %d sell signals (%d triggered), "
                        "%d buyback signals (%d triggered), regime=%s",
                        len(actions), len(sells), len(triggered_sells),
                        len(buybacks), len(triggered_buybacks),
                        cycle_result.get("regime", "N/A"),
                    )

                    # Mark swing positions that now have CCs as OPTIONS_OVERLAY stage
                    for pos in seasoned:
                        if pos.ticker in cc_eligible_tickers:
                            # Check if the CC scalper just sold a call on this ticker
                            if any(
                                s.get("ticker") == pos.ticker and s.get("triggered")
                                for s in sells
                            ):
                                pos.stage = TradeStage.OPTIONS_OVERLAY.value
                                logger.info(
                                    "CC SCALPER: %s stage -> OPTIONS_OVERLAY (CC sold via 6-signal)",
                                    pos.ticker,
                                )
                    self.state.save()
                except Exception as e:
                    logger.error(
                        "CC Scalper cycle error (non-fatal): %s\n%s",
                        e, traceback.format_exc(),
                    )
            else:
                logger.info("CC SCALPER: no seasoned swing tickers eligible for options (all have active CCs or no options chain)")

    # ================================================================
    # EOD Reconciliation
    # ================================================================

    def run_eod_reconciliation(self) -> None:
        """End-of-day reconciliation at 4:05 PM."""
        if self._eod_done:
            return

        logger.info("--- EOD RECONCILIATION ---")

        # Close any open option scalp positions
        if self.options_scalper:
            try:
                self.options_scalper.close_all_positions("EOD cleanup")
                logger.info("Options scalper EOD cleanup complete")
            except Exception as e:
                logger.error("Options scalper EOD cleanup failed: %s", e)

        # Update days held for swing positions
        for pos in self.state.positions:
            if pos.stage in (
                TradeStage.SWING_ACTIVE.value,
                TradeStage.SWING_CONVERTED.value,
                TradeStage.SWING_DIRECT.value,
                TradeStage.OPTIONS_OVERLAY.value,
            ):
                pos.days_held_as_swing += 1

        # Update state timestamp
        self.state.last_run = datetime.now(timezone.utc).isoformat()

        # Risk check
        if self.risk_mgr:
            dd_status = self.risk_mgr.check_drawdown()
            logger.info("Drawdown status: %s", dd_status)

        # Save Layer 4 state at EOD
        self._save_call_buyer_state()

        self.state.save()
        self._eod_done = True

        # EOD Audit Report: compare Alpaca positions/orders vs internal state
        # and send summary via inbox (Mar 24 reliability fix)
        try:
            self._order_tracker.run_eod_audit(self.state, dry_run=self.dry_run)
        except Exception as e:
            logger.error("EOD audit report failed (non-fatal): %s", e)

        # Generate summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print end-of-day summary."""
        lines = [
            "",
            "=" * 70,
            f"COMBINED RUNNER - EOD SUMMARY ({datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')})",
            "=" * 70,
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}",
            f"Equity: ${self.equity:,.2f}",
            "",
        ]

        # Scalp summary
        scalp_pnl = self.state.scalp_risk.get("daily_pnl", 0.0)
        scalp_trades = self.state.scalp_risk.get("daily_trades", 0)
        lines.append(f"SCALP: {scalp_trades} trades, P&L=${scalp_pnl:,.2f}")
        if self.state.scalp_risk.get("kill_switch"):
            lines.append("  ** KILL SWITCH ACTIVE **")

        # Layer 4 Call Buyer summary
        if self._call_buyer:
            cb_summary = self._call_buyer.summary()
            cb_open = cb_summary.get("open_positions", 0)
            cb_daily = cb_summary.get("daily_pnl", 0.0)
            cb_total = cb_summary.get("total_pnl", 0.0)
            lines.append(
                f"CALL BUYER (L4): {cb_open} open positions, "
                f"daily P&L=${cb_daily:,.2f}, total P&L=${cb_total:,.2f}"
            )
            if cb_summary.get("daily_kill"):
                lines.append("  ** DAILY LOSS LIMIT HIT **")
            if cb_summary.get("weekly_kill"):
                lines.append("  ** WEEKLY LOSS LIMIT HIT **")

        # Swing positions
        swing_count = len([
            p for p in self.state.positions
            if p.stage != TradeStage.CLOSED.value
        ])
        lines.append(f"SWING: {swing_count} active positions")
        for pos in self.state.positions:
            if pos.stage != TradeStage.CLOSED.value:
                lines.append(
                    f"  {pos.ticker:6s} {pos.strategy:25s} | "
                    f"stage={pos.stage} | entry=${pos.entry_price:.2f} | "
                    f"shares={pos.shares} | stop=${pos.stop_price:.2f} | "
                    f"days={pos.days_held_as_swing}"
                )

        # Conversions
        if self.state.converted_today:
            lines.append(f"CONVERSIONS TODAY: {', '.join(self.state.converted_today)}")

        # Queue
        queue_count = len(self.state.swing_opportunity_queue)
        if queue_count > 0:
            lines.append(f"QUEUE (unexecuted): {queue_count} items")

        # Risk
        if self.risk_mgr:
            lines.append("")
            lines.append(self.risk_mgr.status_report())

        # Recent trades
        recent = self.state.trade_log[-5:]
        if recent:
            lines.append("")
            lines.append("RECENT TRADES:")
            for t in recent:
                lines.append(
                    f"  {t.get('ticker', '?'):6s} {t.get('exit_reason', '?'):20s} | "
                    f"P&L=${t.get('pnl', 0):.2f}"
                )

        lines.append("=" * 70)
        summary = "\n".join(lines)
        logger.info(summary)

        # Write to file
        summary_path = LOG_DIR / f"combined_summary_{datetime.now(ET).strftime('%Y%m%d')}.txt"
        summary_path.write_text(summary)

    # ================================================================
    # Combined-Loss Halt: Flatten Action (SIZING_RULES.md §3)
    # ================================================================
    # Wired Apr 24 2026 to close the gap noted in autonomous_trading_discipline
    # ("Hard-halt flatten action STILL UNWIRED — discretionary_positions() +
    # mark_flatten_executed() exist but no caller").
    #
    # Trigger: risk_mgr.check_combined_halt() reports halted=True AND
    # flatten_executed_today=False. Action: cancel any open stops, close all
    # discretionary positions (swing non-PMCC + scalp), preserve PMCC hedges,
    # then call mark_flatten_executed() so the action runs at most once per
    # ET trading day. Idempotency: combined_risk["flatten_executed_today"] is
    # cleared by state.reset_daily() at the top of each new ET date in
    # initialize().
    #
    # PMCC LEAPs are NOT flattened — per SIZING_RULES.md §2.1 "Bot continues
    # managing existing PMCC hedges" during a halt; PMCCManager controls those
    # legs and may roll/close them under its own logic.
    def _flatten_on_combined_halt(self) -> None:
        """If the combined-loss kill-switch fired, flatten discretionary
        positions exactly once and mark the action executed for the day.

        Safe to call every loop iteration; cheap when not halted (single
        check_combined_halt call) and idempotent via the persisted
        flatten_executed_today flag.
        """
        if self.risk_mgr is None:
            return
        try:
            halt = self.risk_mgr.check_combined_halt()
        except Exception as e:
            logger.error(
                "Combined-halt flatten skipped (check_combined_halt raised): %s",
                e,
            )
            return
        if not halt.get("halted"):
            return
        if halt.get("flatten_executed_today"):
            return  # already flattened this ET day

        logger.critical(
            "COMBINED-LOSS FLATTEN: kill-switch active "
            "(day P&L $%.2f <= threshold $%.2f, %.1f%% of equity $%.2f). "
            "Closing all discretionary positions; PMCC hedges preserved.",
            halt.get("daily_pnl", 0.0),
            halt.get("threshold_dollars", 0.0),
            halt.get("threshold_pct", 0.0) * 100,
            halt.get("equity", self.equity),
        )

        attempted = 0
        succeeded = 0

        # ---- Swing leg: state.positions, skip PMCC + closed ----
        for pos in list(self.state.positions):
            if pos.stage == TradeStage.CLOSED.value:
                continue
            if getattr(pos, "is_pmcc", False):
                continue
            attempted += 1

            # Cancel any open stop first (mirrors EOD profit-sweep pattern)
            if pos.stop_order_id and not self.dry_run and self.swing_order_mgr:
                try:
                    self.swing_order_mgr.cancel_order(pos.stop_order_id)
                except Exception as e:
                    logger.warning(
                        "FLATTEN: failed to cancel stop for %s: %s",
                        pos.ticker, e,
                    )

            # Edge 96 Patch 2: gate state mutation on broker accept
            close_succeeded = self.dry_run or not self.swing_order_mgr
            if not close_succeeded:
                try:
                    self.swing_order_mgr.close_position(pos.ticker)
                    close_succeeded = True
                except Exception as e:
                    logger.error(
                        "FLATTEN: failed to close swing %s: %s -- skipping "
                        "state mutation (Edge 96 Patch 2)", pos.ticker, e,
                    )
            if not close_succeeded:
                continue  # broker rejected; will retry next cycle

            current_price = self._get_live_price(pos)
            if pos.direction == "long":
                pnl = (current_price - pos.entry_price) * pos.shares
            else:
                pnl = (pos.entry_price - current_price) * pos.shares
            pnl_pct = pnl / (pos.entry_price * pos.shares) if pos.shares > 0 else 0.0

            pos.stage = TradeStage.CLOSED.value
            pos.exit_time = datetime.now(timezone.utc).isoformat()
            pos.exit_price = current_price
            pos.exit_reason = "combined_loss_halt"
            pos.pnl = pnl
            pos.pnl_pct = pnl_pct

            self.state.record_trade(pos, stages=["swing", "combined_loss_halt"])
            succeeded += 1
            logger.info(
                "FLATTEN swing CLOSED: %s %d shares | entry=$%.2f exit=$%.2f | "
                "P&L=$%.2f (%.2f%%)",
                pos.ticker, pos.shares, pos.entry_price, current_price,
                pnl, pnl_pct * 100,
            )

        # Drop closed swings from active list (mirrors _force_close_all_positions)
        self.state.positions = [
            p for p in self.state.positions
            if p.stage != TradeStage.CLOSED.value
        ]

        # ---- Scalp leg: state.scalp_positions ----
        for pos in list(self.state.scalp_positions):
            if pos.stage == TradeStage.CLOSED.value:
                continue
            attempted += 1
            df = self.scalp_dm.get_cached_bars(pos.ticker) if self.scalp_dm else None
            price = (
                float(df["close"].iloc[-1])
                if df is not None and len(df) > 0
                else pos.entry_price
            )
            try:
                self._close_scalp_position(pos, price, "combined_loss_halt")
                succeeded += 1
            except Exception as e:
                logger.error(
                    "FLATTEN: failed to close scalp %s: %s", pos.ticker, e,
                )

        # Mark executed even on partial failure — combined_runner._trading_allowed
        # already gates new entries while the kill switch is active, so any
        # remaining stuck positions will be picked up by per-position management
        # paths (intraday profit check, EOD sweep) on subsequent cycles. Keeping
        # the flag down would cause us to retry cancel_order on every loop iter
        # for the same broker-rejected ticker — wasteful and noisy.
        try:
            self.risk_mgr.mark_flatten_executed()
        except Exception as e:
            logger.error("FLATTEN: mark_flatten_executed raised: %s", e)

        try:
            self.state.save()
        except Exception as e:
            logger.error("FLATTEN: state.save raised (non-fatal): %s", e)

        logger.critical(
            "COMBINED-LOSS FLATTEN COMPLETE: attempted=%d succeeded=%d "
            "(remaining open will be flattened by per-position paths next cycle)",
            attempted, succeeded,
        )

        # Inbox the operator so we know the kill-switch fired
        try:
            from jarvis_utils.inbox import send as _inbox_send
            _inbox_send(
                f"COMBINED-LOSS FLATTEN: day P&L "
                f"${halt.get('daily_pnl', 0.0):.2f} hit halt threshold "
                f"({halt.get('threshold_pct', 0.0) * 100:.1f}% of equity); "
                f"flattened {succeeded}/{attempted} discretionary positions. "
                f"PMCC hedges preserved. No new entries today.",
                source="combined-bot",
            )
        except Exception as e:
            logger.warning("FLATTEN inbox notify failed (non-fatal): %s", e)

    # ================================================================
    # Main Run Loop
    # ================================================================

    def run(self) -> None:
        """Main execution: run through the daily schedule."""
        if not self.initialize():
            logger.warning("Initialization had errors, proceeding with available subsystems")

        now_et = datetime.now(ET)
        logger.info("Current time: %s", now_et.strftime("%Y-%m-%d %H:%M:%S ET"))

        # Pre-market phase
        self.run_premarket()

        # Generate swing signals
        direct_signals = self.generate_swing_signals()

        # Execute direct swing entries (CC-eligible tickers: PAAS/AG/HL)
        # Mar 12: Was TIER_3_SWING_ONLY (always empty). Now routes CC tickers direct.
        tier3 = [s for s in direct_signals if s.ticker in CC_ELIGIBLE_SWING]
        sell_signals = [s for s in direct_signals if s.direction == "sell"]
        self.execute_direct_swing(tier3 + sell_signals)

        # Determine market timing
        h, m = now_et.hour, now_et.minute
        t_min = h * 60 + m
        market_open_min = MARKET_OPEN[0] * 60 + MARKET_OPEN[1]
        market_close_min = MARKET_CLOSE[0] * 60 + MARKET_CLOSE[1]

        # Weekend or after market close: analysis-only mode, exit cleanly
        if now_et.weekday() >= 5 or t_min >= market_close_min:
            logger.info("Market is closed. Running in analysis-only mode.")
            self._print_summary()
            self.state.last_run = datetime.now(timezone.utc).isoformat()
            self.state.save()
            return

        # Pre-market: sleep until 30 min before open (9:00 AM ET), then enter loop
        if t_min < market_open_min - 30:
            sleep_secs = (market_open_min - 30 - t_min) * 60
            logger.info(
                "Pre-market (%s ET): sleeping %d min until 9:00 AM ET",
                now_et.strftime("%H:%M"),
                (market_open_min - 30 - t_min),
            )
            time.sleep(sleep_secs)
            now_et = datetime.now(ET)
            h, m = now_et.hour, now_et.minute
            t_min = h * 60 + m

        # SIGTERM handler — log before death so we know when/why process was killed
        import signal as _signal
        def _sigterm_handler(signum, frame):
            logger.critical("SIGTERM received — process being killed (signum=%d)", signum)
            logging.shutdown()
            raise SystemExit(0)
        _signal.signal(_signal.SIGTERM, _sigterm_handler)

        # PMCC: auto-detect LEAPs from Alpaca and register with manager
        # Edge 107 PR3-v2: gate on Alpaca-healthy flag (caller-side).
        if PMCC_ENABLED and self._pmcc_manager and self._trading_allowed("pmcc.auto_detect+initial_cycle"):
            try:
                new_spreads = self._pmcc_manager.auto_detect_leaps()
                if new_spreads:
                    logger.info(
                        "PMCC: auto-detected %d LEAP(s) from Alpaca positions",
                        len(new_spreads),
                    )
                    for sp in new_spreads:
                        logger.info(
                            "  PMCC LEAP: %s %s $%.0f DTE=%d delta=%.2f",
                            sp.ticker, sp.long_leg.symbol, sp.long_leg.strike,
                            sp.long_leg_dte, sp.long_leg.delta,
                        )
                # Run initial PMCC cycle immediately (sell short calls if possible)
                self._pmcc_manager.update_market_state(
                    vix_level=self._vix_level,
                    breadth_gate_active=bool(self.state.breadth_gate_trigger_date),
                )
                actions = self._pmcc_manager.run_cycle()
                if actions:
                    logger.info(
                        "PMCC initial cycle: %d action(s): %s",
                        len(actions),
                        "; ".join(str(a) for a in actions[:5]),
                    )
            except Exception as e:
                logger.error("PMCC startup auto-detect/cycle failed: %s", e)

        # Main intraday loop
        logger.info("Entering intraday monitoring loop...")
        last_scalp_check = 0.0
        last_heartbeat = 0.0
        last_pmcc_check = time.time()   # just ran, so start timer now
        last_cc_scalper_check = 0.0    # CC scalper intraday cycle timer
        last_call_buyer_check = 0.0   # Layer 4 call buyer cycle timer
        HEARTBEAT_INTERVAL = 10 * 60  # Edge 108 — log heartbeat every 10 min (tightened from 30 min for wedge detection)
        PMCC_CYCLE_INTERVAL = 5 * 60  # run PMCC management every 5 minutes (fast re-entry for scalping loop)
        CC_SCALPER_CYCLE_INTERVAL = 120  # CC scalper every 2 min (matches POLL_INTERVAL_SECONDS in slvr_cc_config)
        CALL_BUYER_CYCLE_INTERVAL = 120  # Layer 4 call buyer every 2 min
        last_reconciliation_check = 0.0  # Alpaca reconciliation timer

        try:
            while True:
                now_et = datetime.now(ET)
                h, m = now_et.hour, now_et.minute
                t_min = h * 60 + m

                # Heartbeat — proves the loop is alive; tracks memory growth
                now_ts_hb = time.time()
                if now_ts_hb - last_heartbeat >= HEARTBEAT_INTERVAL:
                    last_heartbeat = now_ts_hb
                    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    try:
                        cgroup_bytes = int(
                            Path("/sys/fs/cgroup/memory.current").read_text().strip()
                        )
                        cgroup_limit = int(
                            Path("/sys/fs/cgroup/memory.max").read_text().strip()
                        )
                        mem_info = (
                            f"rss={mem_kb}KB cgroup={cgroup_bytes//1048576}MB/"
                            f"{cgroup_limit//1048576}MB"
                        )
                    except Exception:
                        mem_info = f"rss={mem_kb}KB"
                    print(
                        f"HEARTBEAT {now_et.strftime('%H:%M')} ET | "
                        f"equity={self.equity:.2f} | "
                        f"swing={len([p for p in self.state.positions if p.stage != TradeStage.CLOSED.value])} | "
                        f"scalp={len(self.state.scalp_positions)} | {mem_info}",
                        flush=True,
                    )
                    logger.info(
                        "Heartbeat %s ET | equity=%.2f | swing=%d | scalp=%d | %s",
                        now_et.strftime("%H:%M"),
                        self.equity,
                        len([p for p in self.state.positions if p.stage != TradeStage.CLOSED.value]),
                        len(self.state.scalp_positions),
                        mem_info,
                    )
                    # TA Overlay status in heartbeat
                    if self.ta_overlay_output:
                        tao = self.ta_overlay_output
                        rz_count = len([r for r in tao.red_zones.values() if r.in_red_zone])
                        cc_flag_count = len([r for r in tao.red_zones.values() if r.flag_cc_sell])
                        logger.info(
                            "  TA Overlays | GSR=%s(%.2fx) G/S=%.1f | "
                            "Gold/SPX=%s(%.2fx) | RedZone=%d skip, %d CC-flag",
                            tao.gsr.trend, tao.gsr.size_multiplier, tao.gsr.gs_ratio,
                            tao.gold_spx.trend, tao.gold_spx.size_multiplier,
                            rz_count, cc_flag_count,
                        )

                # Combined-loss kill-switch flatten action (Apr 24 2026):
                # runs every loop iteration, cheap when not halted, idempotent
                # via persisted flatten_executed_today flag. Closes the wiring
                # gap noted in autonomous_trading_discipline.md (action existed
                # in unified_risk but had no caller). MUST run BEFORE the
                # market-close break so a same-cycle halt still flattens
                # instead of falling through to EOD reconciliation.
                try:
                    self._flatten_on_combined_halt()
                except Exception as e:
                    logger.error(
                        "_flatten_on_combined_halt raised (non-fatal): %s", e,
                    )

                # Past market close
                if t_min >= market_close_min + 5:
                    self.run_eod_reconciliation()
                    break

                # 2:00 PM cutoff
                cutoff_min = SCALP_WINDOW_CUTOFF_HOUR * 60
                if t_min >= cutoff_min and not self._direct_swing_executed:
                    self.execute_cutoff_swing_entries()

                # 3:30 PM options overlay (run once)
                options_min = 15 * 60 + 30
                if t_min >= options_min and not self._options_overlay_done:
                    self._options_overlay_done = True
                    self.run_options_overlay()

                # 3:55 PM: EOD profit sweep -- close green, carry red/flat
                # David's rule: "never forego profit-taking"
                force_close_min = FORCE_CLOSE_ALL_TIME[0] * 60 + FORCE_CLOSE_ALL_TIME[1]
                if t_min >= force_close_min and not self._eod_profit_swept:
                    self._eod_profit_sweep()

                # 3:59 PM: Friday catalyst close (only positions with imminent
                # earnings are flattened; the rest carry through the weekend
                # per Edge 22 + 2026-04-16 backtest). Mon-Thu = no-op.
                friday_close_min = 15 * 60 + 59
                if t_min >= friday_close_min and not self._all_positions_closed:
                    self._force_close_all_positions()

                # Scalp monitoring cycle (every 60 seconds)
                now_ts = time.time()
                if now_ts - last_scalp_check >= SCALP_POLL_SECONDS:
                    last_scalp_check = now_ts

                    # Reconcile with Alpaca every cycle (Apr 8: always use real positions)
                    self._reconcile_positions_with_alpaca()

                    self.run_scalp_cycle()

                    # Intraday profit-taking: close any swing position
                    # that has hit the $0.50/share target
                    self._intraday_profit_check()

                    # Options scalper AGENT-PAUSED 2026-03-09 (CC strategy pivot).
                    # Provenance audit 2026-04-23: not a David directive, agent
                    # fabrication. Defer re-enable to unified system phase 2
                    # (expression selector picks instrument per Pick).
                    # if self.options_scalper:
                    #     self.options_scalper.run_scan_cycle()

                # Order lifecycle tracker: poll pending orders for fills/failures
                # Runs every scalp cycle (~60s). Catches expired/canceled orders
                # and sends inbox alerts on failures (Mar 24 reliability fix).
                try:
                    resolved = self._order_tracker.poll_pending_orders()
                    if resolved:
                        logger.info(
                            "ORDER TRACKER: %d order(s) resolved this cycle | "
                            "pending=%d | resolved_today=%d",
                            len(resolved),
                            self._order_tracker.pending_count(),
                            self._order_tracker.resolved_today_count(),
                        )
                except Exception as e:
                    logger.error("Order tracker poll error (non-fatal): %s", e)

                # Alpaca position reconciliation (every RECONCILIATION_INTERVAL_SECONDS)
                if ALPACA_RECONCILIATION_ENABLED:
                    now_ts_recon = time.time()
                    if now_ts_recon - last_reconciliation_check >= RECONCILIATION_INTERVAL_SECONDS:
                        last_reconciliation_check = now_ts_recon
                        try:
                            self._run_alpaca_reconciliation()
                        except Exception as e:
                            logger.error("Alpaca reconciliation cycle error (non-fatal): %s", e)

                # PMCC management cycle (every 5 min): sell short calls, manage risk
                # Edge 107 PR3-v2: gate on Alpaca-healthy flag (caller-side).
                if PMCC_ENABLED and self._pmcc_manager and self._trading_allowed("pmcc.run_cycle.intraday"):
                    now_ts_pmcc = time.time()
                    if now_ts_pmcc - last_pmcc_check >= PMCC_CYCLE_INTERVAL:
                        last_pmcc_check = now_ts_pmcc
                        try:
                            self._pmcc_manager.update_market_state(
                                vix_level=self._vix_level,
                                breadth_gate_active=bool(self.state.breadth_gate_trigger_date),
                            )
                            pmcc_actions = self._pmcc_manager.run_cycle()
                            if pmcc_actions:
                                logger.info(
                                    "PMCC intraday cycle: %d action(s): %s",
                                    len(pmcc_actions),
                                    "; ".join(str(a) for a in pmcc_actions[:3]),
                                )
                        except Exception as e:
                            logger.error("PMCC intraday cycle error: %s", e)

                # CC Scalper intraday cycle (every 2 min): manage buybacks of
                # existing CC positions and catch IV spikes for new sells.
                # Only runs if we have seasoned swing positions with options.
                # Edge 107 PR3-v2: gate on Alpaca-healthy flag (caller-side).
                if self._cc_scalper and self._trading_allowed("scalper.buyback.intraday"):
                    now_ts_cc = time.time()
                    if now_ts_cc - last_cc_scalper_check >= CC_SCALPER_CYCLE_INTERVAL:
                        last_cc_scalper_check = now_ts_cc
                        # Check if we have any seasoned swing positions worth evaluating
                        has_seasoned = any(
                            pos.stage in (
                                TradeStage.SWING_ACTIVE.value,
                                TradeStage.SWING_CONVERTED.value,
                                TradeStage.SWING_DIRECT.value,
                                TradeStage.OPTIONS_OVERLAY.value,
                            )
                            and pos.days_held_as_swing >= (CC_ELIGIBLE_AFTER_DAYS or 0)
                            and pos.ticker in CC_OPTIONS_ELIGIBLE
                            for pos in self.state.positions
                        )
                        # Also run if the CC scalper has open positions to manage buybacks
                        has_open_ccs = bool(self._cc_scalper.state.open_positions())
                        if has_seasoned or has_open_ccs:
                            try:
                                cycle_result = self._cc_scalper.run_once()
                                actions = cycle_result.get("actions", [])
                                if actions:
                                    logger.info(
                                        "CC Scalper intraday: %d action(s)",
                                        len(actions),
                                    )
                            except Exception as e:
                                logger.error("CC Scalper intraday cycle error: %s", e)

                # Layer 4 Call Buyer cycle (every 2 min): evaluate dip-buy
                # signals and manage existing long call positions.
                # Edge 107 PR3-v2: gate on Alpaca-healthy flag (caller-side).
                if self._call_buyer and CALL_BUYER_ENABLED and self._trading_allowed("call_buyer.auto_manage"):
                    now_ts_cb = time.time()
                    if now_ts_cb - last_call_buyer_check >= CALL_BUYER_CYCLE_INTERVAL:
                        last_call_buyer_check = now_ts_cb
                        try:
                            # Determine which tickers Layer 1 just entered this cycle
                            l1_entered = set()
                            # Check recent swing entries from this cycle
                            for pos in self.state.positions:
                                if (pos.stage in (
                                        TradeStage.SWING_DIRECT.value,
                                        TradeStage.SWING_ACTIVE.value,
                                    )
                                    and pos.days_held_as_swing == 0
                                ):
                                    l1_entered.add(pos.ticker)

                            cb_result = self._call_buyer.auto_manage(
                                skip_tickers=l1_entered,
                                equity=self.equity,
                                swing_state=self.state.swing_state,
                            )
                            entries = cb_result.get("entries", [])
                            exits = cb_result.get("exits", [])
                            if entries or exits:
                                logger.info(
                                    "Layer 4 Call Buyer: %d entry(s), %d exit(s)",
                                    len(entries), len(exits),
                                )
                            # Persist Layer 4 state
                            self._save_call_buyer_state()
                        except Exception as e:
                            logger.error("Layer 4 Call Buyer cycle error: %s", e)

                time.sleep(min(SCALP_POLL_SECONDS / 2, 15))

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self._eod_profit_sweep()
            self._force_close_all_positions()
            self.run_eod_reconciliation()

        except Exception as e:
            logger.critical("Fatal error in main loop: %s\n%s", e, traceback.format_exc())
            self._eod_profit_sweep()
            self._force_close_all_positions()
            self.run_eod_reconciliation()

    # ================================================================
    # Status Display
    # ================================================================

    def show_status(self) -> None:
        """Display current system status."""
        self.state = CombinedState.load()
        self.risk_mgr = UnifiedRiskManager(self.state, self.equity)

        now_et = datetime.now(ET)
        print(f"\nCombined Trading System Status - {now_et.strftime('%Y-%m-%d %H:%M ET')}")
        print("=" * 70)

        # Swing positions
        swing_active = [
            p for p in self.state.positions
            if p.stage != TradeStage.CLOSED.value
        ]
        print(f"\nSwing Positions ({len(swing_active)}):")
        if swing_active:
            for pos in swing_active:
                print(
                    f"  {pos.ticker:6s} | stage={pos.stage:18s} | origin={pos.origin:16s} | "
                    f"strategy={pos.strategy:25s} | entry=${pos.entry_price:8.2f} | "
                    f"shares={pos.shares:4d} | stop=${pos.stop_price:8.2f} | "
                    f"days={pos.days_held_as_swing}"
                )
        else:
            print("  (none)")

        # Scalp positions
        scalp_active = [
            p for p in self.state.scalp_positions
            if p.stage != TradeStage.CLOSED.value
        ]
        print(f"\nScalp Positions ({len(scalp_active)}):")
        if scalp_active:
            for pos in scalp_active:
                print(
                    f"  {pos.ticker:6s} | stage={pos.stage:18s} | "
                    f"strategy={pos.scalp_strategy:10s} | entry=${pos.entry_price:8.2f} | "
                    f"shares={pos.shares:4d} | stop=${pos.stop_price:8.2f} | "
                    f"target=${pos.target_price:8.2f} | conv_eligible={pos.conversion_eligible}"
                )
        else:
            print("  (none)")

        # Queue
        queue = self.state.swing_opportunity_queue
        print(f"\nSwing Opportunity Queue ({len(queue)}):")
        if queue:
            for item in queue:
                print(
                    f"  {item.ticker:6s} | strategy={item.strategy:25s} | "
                    f"entry=${item.entry_price:8.2f} | stop=${item.stop_price:8.2f} | "
                    f"priority={item.priority:.2f}"
                )
        else:
            print("  (none)")

        # Risk status
        print(f"\n{self.risk_mgr.status_report()}")

        # Recent trades
        recent = self.state.trade_log[-10:]
        print(f"\nRecent Trades ({len(recent)} of {len(self.state.trade_log)} total):")
        for t in recent:
            pnl = t.get("pnl", 0)
            marker = "+" if pnl > 0 else ""
            print(
                f"  {t.get('ticker', '?'):6s} | {t.get('origin', '?'):16s} | "
                f"{t.get('strategy', '?'):25s} | "
                f"P&L={marker}${pnl:.2f} | {t.get('exit_reason', '?')}"
            )

        print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

def acquire_singleton_lock(lock_path: str = "/tmp/combined_runner.lock"):
    """
    Acquire an exclusive flock on a lockfile. Prevents two live bot instances
    from running simultaneously and submitting duplicate orders.

    On 2026-04-16 a duplicate bot situation submitted near-duplicate buyback
    orders for HL and PAAS within 12 seconds of each other. Alpaca's dedup
    canceled the older copies before they filled — a near-miss only. This
    lock ensures fail-fast on duplicate launch.

    Returns the open file handle (caller must keep it alive for the lock
    to remain held). Process exit releases automatically.
    """
    import fcntl
    # Open append-mode so the existing PID isn't truncated before we
    # attempt the lock — important so the error message can report the
    # actual holder.
    lock_fp = open(lock_path, "a+")
    try:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        existing_pid = "?"
        try:
            with open(lock_path) as r:
                existing_pid = r.read().strip() or "?"
        except Exception:
            pass
        sys.stderr.write(
            f"FATAL: another combined_runner is already running "
            f"(lockfile={lock_path}, holder pid={existing_pid}). "
            f"Refusing to start to prevent duplicate orders.\n"
        )
        sys.exit(2)
    # We have the lock — now truncate and write our PID
    lock_fp.seek(0)
    lock_fp.truncate()
    lock_fp.write(str(os.getpid()))
    lock_fp.flush()
    return lock_fp


def main():
    parser = argparse.ArgumentParser(
        description="Combined Trading System: Scalp -> Swing -> Options Overlay"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Dry run mode (default): generate signals without placing orders",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Live mode: place real paper-trade orders",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current positions and system status",
    )
    parser.add_argument(
        "--scalp-only", action="store_true",
        help="Run scalp subsystem only (all positions close EOD)",
    )
    parser.add_argument(
        "--swing-only", action="store_true",
        help="Run swing + options only (no scalping)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    dry_run = not args.live

    # Singleton lock — fail fast if another live bot is already running.
    # Status checks and dry-runs don't need the lock (they don't place orders).
    _singleton_lock = None
    if args.live and not args.status:
        _singleton_lock = acquire_singleton_lock()
        logging.getLogger(__name__).info(
            "Acquired singleton lock /tmp/combined_runner.lock (pid=%d)",
            os.getpid(),
        )

    runner = CombinedRunner(
        dry_run=dry_run,
        scalp_only=args.scalp_only,
        swing_only=args.swing_only,
    )

    if args.status:
        runner.show_status()
        return

    runner.run()


if __name__ == "__main__":
    main()
