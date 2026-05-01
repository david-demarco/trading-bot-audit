"""
combined_state.py - Unified state management for the combined trading system.

Provides:
  - TradeStage enum tracking position lifecycle
  - UnifiedPosition dataclass for all stages (scalp, swing, options)
  - CombinedState as the single persisted state file (combined_state.json)
  - Trade log for full lifecycle auditing
  - Migration utility from legacy swing_state.json

Reference: ~/trading_bot/research/combined_system_design.md  Section 5, 12
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("combined_state")

STATE_FILE = Path(__file__).parent.resolve() / "combined_state.json"


# =============================================================================
# TRADE STAGE ENUM
# =============================================================================

class TradeStage(str, Enum):
    """Lifecycle stage for any position managed by the combined system."""
    SIGNAL_PENDING  = "signal_pending"
    SCALP_ENTRY     = "scalp_entry"
    SCALP_ACTIVE    = "scalp_active"
    SCALP_STALL     = "scalp_stall"
    EVALUATE        = "evaluate"
    SWING_DIRECT    = "swing_direct"
    SWING_CONVERTED = "swing_converted"
    SWING_ACTIVE    = "swing_active"
    OPTIONS_OVERLAY = "options_overlay"
    EXITING         = "exiting"
    CLOSED          = "closed"


# =============================================================================
# UNIFIED POSITION
# =============================================================================

@dataclass
class UnifiedPosition:
    """
    A single position tracked through every stage of the combined system.

    Fields are progressively populated as the position moves through stages.
    Scalp-only fields are None for swing-direct positions and vice versa.
    """
    # --- Identity ---
    ticker: str
    stage: str                            # TradeStage value
    origin: str = "unknown"               # "scalp", "scalp_conversion", "swing_direct"

    # --- Swing context ---
    strategy: str = ""                    # swing strategy name (if applicable)
    swing_signal_date: str = ""           # ISO date of the swing signal

    # --- Entry ---
    entry_date: str = ""                  # ISO timestamp
    entry_price: float = 0.0
    shares: int = 0
    direction: str = "long"               # "long" or "short"

    # --- Stops and targets ---
    stop_price: float = 0.0
    target_price: float = 0.0
    atr_at_entry: float = 0.0
    highest_price: float = 0.0            # For trailing stops
    stop_order_id: str = ""               # Alpaca GTC stop order

    # --- Scalp-specific ---
    scalp_strategy: str = ""              # "momentum" or "gap_fill"
    scalp_entry_time: str = ""            # ISO timestamp of scalp fill
    scalp_entry_price: float = 0.0
    scalp_max_hold_time: str = ""         # ISO timestamp deadline
    conversion_eligible: bool = False

    # --- Conversion ---
    conversion_time: str = ""             # ISO timestamp when converted
    conversion_reason: str = ""

    # --- Swing tracking ---
    days_held_as_swing: int = 0

    # --- Options overlay ---
    options_overlay: Optional[Dict[str, Any]] = None

    # --- PMCC-specific ---
    is_pmcc: bool = False                    # True if backed by LEAP, not shares
    leap_symbol: str = ""                    # OCC symbol of the LEAP
    leap_strike: float = 0.0
    leap_expiry: str = ""
    leap_cost_basis: float = 0.0            # Per-share cost of LEAP
    leap_delta: float = 0.0
    pmcc_spread_id: str = ""                # Links to DiagonalSpread in PMCCManager

    # --- Averaging down ---
    lot_count: int = 1                       # Number of entry lots (1 = initial, 2 = averaged down)

    # --- Exit ---
    exit_time: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnifiedPosition":
        valid = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid})

    @classmethod
    def from_scalp_signal(
        cls,
        ticker: str,
        strategy: str,
        direction: str,
        entry_price: float,
        shares: int,
        stop_price: float,
        target_price: float,
        max_hold_time: str = "",
        conversion_eligible: bool = False,
        swing_strategy: str = "",
    ) -> "UnifiedPosition":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            ticker=ticker,
            stage=TradeStage.SCALP_ENTRY.value,
            origin="scalp",
            strategy=swing_strategy,
            entry_date=now,
            entry_price=entry_price,
            shares=shares,
            direction=direction,
            stop_price=stop_price,
            target_price=target_price,
            scalp_strategy=strategy,
            scalp_entry_time=now,
            scalp_entry_price=entry_price,
            scalp_max_hold_time=max_hold_time,
            conversion_eligible=conversion_eligible,
        )

    @classmethod
    def from_swing_signal(
        cls,
        ticker: str,
        strategy: str,
        entry_price: float,
        shares: int,
        stop_price: float,
        atr: float,
    ) -> "UnifiedPosition":
        now = datetime.now(timezone.utc).isoformat()
        return cls(
            ticker=ticker,
            stage=TradeStage.SWING_DIRECT.value,
            origin="swing_direct",
            strategy=strategy,
            entry_date=now,
            entry_price=entry_price,
            shares=shares,
            direction="long",
            stop_price=stop_price,
            atr_at_entry=atr,
            highest_price=entry_price,
        )

    @classmethod
    def from_leap_entry(
        cls,
        ticker: str,
        strategy: str,
        leap_symbol: str,
        leap_strike: float,
        leap_expiry: str,
        leap_cost_basis: float,
        leap_delta: float,
        spread_id: str,
    ) -> "UnifiedPosition":
        """Create a position backed by a LEAP (PMCC) instead of shares."""
        pos = cls(
            ticker=ticker,
            strategy=strategy,
            entry_price=leap_cost_basis,  # Use LEAP cost as entry price
            shares=0,                      # No shares — LEAP is the position
            stage=TradeStage.SWING_DIRECT.value,
            is_pmcc=True,
            leap_symbol=leap_symbol,
            leap_strike=leap_strike,
            leap_expiry=leap_expiry,
            leap_cost_basis=leap_cost_basis,
            leap_delta=leap_delta,
            pmcc_spread_id=spread_id,
        )
        return pos


# =============================================================================
# SWING OPPORTUNITY QUEUE ITEM
# =============================================================================

@dataclass
class SwingQueueItem:
    """An entry in the swing opportunity queue held for scalp-first entry."""
    ticker: str
    strategy: str
    signal_date: str
    entry_price: float
    stop_price: float
    shares: int
    atr: float
    priority: float = 0.0
    rationale: str = ""
    valid_until: str = ""               # ISO timestamp cutoff

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SwingQueueItem":
        valid = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid})


# =============================================================================
# TRADE LOG ENTRY
# =============================================================================

@dataclass
class TradeLogEntry:
    """A completed trade for the full-lifecycle audit log."""
    ticker: str
    origin: str
    strategy: str
    direction: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    stages: List[str] = field(default_factory=list)  # stages traversed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# COMBINED STATE
# =============================================================================

@dataclass
class CombinedState:
    """
    Single persisted state for the combined system.

    Replaces separate scalp in-memory state and swing_state.json.
    Written atomically to combined_state.json after every significant event.
    """
    # Active positions across all stages
    positions: List[UnifiedPosition] = field(default_factory=list)

    # Active scalp positions (Tier 2 and unconverted Tier 1/1b)
    scalp_positions: List[UnifiedPosition] = field(default_factory=list)

    # Swing opportunity queue (Tier 1/1b signals awaiting scalp attempt)
    swing_opportunity_queue: List[SwingQueueItem] = field(default_factory=list)

    # Options overlay state (preserved for OptionsOverlay compatibility)
    options_state: Dict[str, Any] = field(default_factory=lambda: {
        "positions": [],
        "total_premium_collected": 0.0,
        "total_realized_pnl": 0.0,
    })

    # PMCC state (preserved for PMCCManager persistence)
    pmcc_state: dict = field(default_factory=lambda: {
        "spreads": [],
        "spread_counter": 0,
    })

    # Tickers converted from scalp to swing today (double-down prevention)
    converted_today: List[str] = field(default_factory=list)

    # Scalp risk tracking
    scalp_risk: Dict[str, Any] = field(default_factory=lambda: {
        "daily_pnl": 0.0,
        "daily_trades": 0,
        "kill_switch": False,
    })

    # Combined risk tracking — combined day P&L across scalp+swing+options.
    # Wired Apr 22 2026 to close the gap flagged in SIZING_RULES.md §2.1.
    # When daily_pnl <= -DAILY_COMBINED_LOSS_LIMIT * equity, kill_switch_active
    # is set to True. flatten_executed_today gates the actual flatten action
    # so it runs at most once per day even if the loss deepens further.
    combined_risk: Dict[str, Any] = field(default_factory=lambda: {
        "daily_pnl": 0.0,
        "kill_switch_active": False,
        "flatten_executed_today": False,
        "halt_threshold_pct": 0.0,  # populated when kill switch fires
        "halt_fired_at": "",         # ISO timestamp when first triggered
    })

    # Swing state (backward compatible with swing_state.json fields)
    swing_state: Dict[str, Any] = field(default_factory=lambda: {
        "strategy_pnl": {
            "momentum_rotation": 0.0,
            "vwap_mean_reversion": 0.0,
            "sector_relative_strength": 0.0,
            "donchian_breakout": 0.0,
            "rsi2_mean_reversion": 0.0,
        },
        "momentum_last_rebalance_date": "",
        "momentum_trading_days_since_rebal": 0,
        "high_water_mark": 0.0,
        "circuit_breaker_active": False,
        "circuit_breaker_days_remaining": 0,
        "drawdown_reduction_active": False,
    })

    # Trade log (completed trades)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)

    # Layer 4 call buyer state (C1 fix: first-class field, not side-channel write)
    call_buyer_state: Dict[str, Any] = field(default_factory=dict)

    # Breadth gate trigger date (YYYY-MM-DD of the trading day when gate fired)
    breadth_gate_trigger_date: str = ""

    # Metadata
    last_run: str = ""
    version: str = "1.0"

    # ----------------------------------------------------------------
    # Persistence
    # ----------------------------------------------------------------

    def save(self, path: Path = STATE_FILE) -> None:
        """Atomically save state to JSON."""
        def _json_default(obj):
            if hasattr(obj, "item"):
                return obj.item()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return str(obj)

        data = {
            "positions": [p.to_dict() for p in self.positions],
            "scalp_positions": [p.to_dict() for p in self.scalp_positions],
            "swing_opportunity_queue": [q.to_dict() for q in self.swing_opportunity_queue],
            "options_state": self.options_state,
            "pmcc_state": self.pmcc_state,
            "converted_today": self.converted_today,
            "scalp_risk": self.scalp_risk,
            "combined_risk": self.combined_risk,
            "swing_state": self.swing_state,
            "trade_log": self.trade_log[-200:],  # Keep last 200 trades
            "call_buyer_state": self.call_buyer_state,
            "breadth_gate_trigger_date": self.breadth_gate_trigger_date,
            "last_run": self.last_run,
            "version": self.version,
        }

        path = Path(path)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        tmp.rename(path)
        logger.debug("Combined state saved to %s", path)

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> "CombinedState":
        """Load state from JSON file."""
        if not path.exists():
            logger.info("No combined state file found at %s, starting fresh", path)
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)

            state = cls()
            state.positions = [
                UnifiedPosition.from_dict(p) for p in data.get("positions", [])
            ]
            state.scalp_positions = [
                UnifiedPosition.from_dict(p) for p in data.get("scalp_positions", [])
            ]
            state.swing_opportunity_queue = [
                SwingQueueItem.from_dict(q) for q in data.get("swing_opportunity_queue", [])
            ]
            state.options_state = data.get("options_state", state.options_state)
            state.pmcc_state = data.get("pmcc_state", state.pmcc_state)
            state.converted_today = data.get("converted_today", [])
            state.scalp_risk = data.get("scalp_risk", state.scalp_risk)
            state.combined_risk = data.get("combined_risk", state.combined_risk)
            state.swing_state = data.get("swing_state", state.swing_state)
            state.trade_log = data.get("trade_log", [])
            state.call_buyer_state = data.get("call_buyer_state", {})
            state.breadth_gate_trigger_date = data.get("breadth_gate_trigger_date", "")
            state.last_run = data.get("last_run", "")
            state.version = data.get("version", "1.0")

            logger.info(
                "Combined state loaded: %d swing positions, %d scalp positions, "
                "%d in queue, %d trades logged",
                len(state.positions), len(state.scalp_positions),
                len(state.swing_opportunity_queue), len(state.trade_log),
            )
            return state

        except Exception as e:
            logger.error("Failed to load combined state from %s: %s", path, e)
            return cls()

    # ----------------------------------------------------------------
    # Convenience helpers
    # ----------------------------------------------------------------

    def all_active_positions(self) -> List[UnifiedPosition]:
        """Return all positions that are not closed."""
        return [
            p for p in (self.positions + self.scalp_positions)
            if p.stage != TradeStage.CLOSED.value
        ]

    def swing_position_tickers(self) -> set:
        """Return the set of tickers held as swing positions.

        Includes PMCC positions (shares=0 but is_pmcc=True).
        """
        return {
            p.ticker for p in self.positions
            if p.stage in (
                TradeStage.SWING_ACTIVE.value,
                TradeStage.SWING_CONVERTED.value,
                TradeStage.SWING_DIRECT.value,
                TradeStage.OPTIONS_OVERLAY.value,
            )
            or p.is_pmcc
        }

    def scalp_position_tickers(self) -> set:
        """Return the set of tickers held as scalp positions."""
        return {
            p.ticker for p in self.scalp_positions
            if p.stage in (
                TradeStage.SCALP_ACTIVE.value,
                TradeStage.SCALP_ENTRY.value,
                TradeStage.SCALP_STALL.value,
                TradeStage.EVALUATE.value,
            )
        }

    def record_trade(self, position: UnifiedPosition, stages: List[str] = None) -> None:
        """Record a completed trade in the log."""
        entry = TradeLogEntry(
            ticker=position.ticker,
            origin=position.origin,
            strategy=position.strategy or position.scalp_strategy,
            direction=position.direction,
            entry_time=position.entry_date,
            entry_price=position.entry_price,
            exit_time=position.exit_time,
            exit_price=position.exit_price,
            shares=position.shares,
            pnl=position.pnl,
            pnl_pct=position.pnl_pct,
            exit_reason=position.exit_reason,
            stages=stages or [],
        )
        self.trade_log.append(entry.to_dict())

    def reset_daily(self) -> None:
        """Reset intraday-only state at the start of a new trading day."""
        self.converted_today = []
        self.scalp_risk = {
            "daily_pnl": 0.0,
            "daily_trades": 0,
            "kill_switch": False,
        }
        self.combined_risk = {
            "daily_pnl": 0.0,
            "kill_switch_active": False,
            "flatten_executed_today": False,
            "halt_threshold_pct": 0.0,
            "halt_fired_at": "",
        }
        self.swing_opportunity_queue = []
        # Clear closed scalp positions from the list
        self.scalp_positions = [
            p for p in self.scalp_positions
            if p.stage != TradeStage.CLOSED.value
        ]
        logger.info("Daily state reset complete")


# =============================================================================
# MIGRATION UTILITY
# =============================================================================

def migrate_swing_state(
    old_path: Path,
    new_path: Path = STATE_FILE,
) -> CombinedState:
    """
    One-time migration from swing_state.json to combined_state.json.

    Converts existing SwingPosition records into UnifiedPosition records
    and preserves all swing bot state fields.
    """
    if not old_path.exists():
        logger.warning("No swing state to migrate at %s", old_path)
        return CombinedState()

    with open(old_path) as f:
        old_data = json.load(f)

    combined = CombinedState()

    # Migrate swing state fields
    combined.swing_state = {
        "strategy_pnl": old_data.get("strategy_pnl", combined.swing_state["strategy_pnl"]),
        "momentum_last_rebalance_date": old_data.get("momentum_last_rebalance_date", ""),
        "momentum_trading_days_since_rebal": old_data.get("momentum_trading_days_since_rebal", 0),
        "high_water_mark": old_data.get("high_water_mark", 0.0),
        "circuit_breaker_active": old_data.get("circuit_breaker_active", False),
        "circuit_breaker_days_remaining": old_data.get("circuit_breaker_days_remaining", 0),
        "drawdown_reduction_active": old_data.get("drawdown_reduction_active", False),
    }

    # Migrate existing positions
    for pos_data in old_data.get("positions", []):
        unified = UnifiedPosition(
            ticker=pos_data.get("ticker", ""),
            stage=TradeStage.SWING_ACTIVE.value,
            origin="swing_direct",
            strategy=pos_data.get("strategy", ""),
            entry_date=pos_data.get("entry_date", ""),
            entry_price=pos_data.get("entry_price", 0.0),
            shares=pos_data.get("shares", 0),
            direction="long",
            stop_price=pos_data.get("stop_price", 0.0),
            atr_at_entry=pos_data.get("atr_at_entry", 0.0),
            highest_price=pos_data.get("highest_price", 0.0),
            stop_order_id=pos_data.get("stop_order_id", ""),
            days_held_as_swing=pos_data.get("days_held", 0),
        )
        combined.positions.append(unified)

    combined.save(new_path)
    logger.info(
        "Migration complete: %d positions migrated from %s to %s",
        len(combined.positions), old_path, new_path,
    )
    return combined
