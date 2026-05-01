#!/usr/bin/env python3
"""
slvr_cc_scalper.py - Deep OTM covered call premium scalping module for metals
mining ETFs (multi-ticker).

Strategy:
  1. Hold shares of metals mining ETFs (SLVR, SGDM, GDX, SIL, etc.).
  2. Sell DEEP OTM, LONG-DATED covered calls (35-50% OTM, 120-220 DTE) into
     IV/price spikes.
  3. Buy them back when IV crushes or the underlying pulls back.
  4. Sell again on the next spike.  Near-instant back-and-forth.  Shares never
     move.

Key insight from research:
  - GLD Granger-causes miners (p=0.0135) -- gold moves predict mining ETFs.
  - This is a VEGA trade: 1% IV drop on deep OTM calls = many days of theta.
  - Correlation intensifies during spikes (GLD 0.62 -> 0.91).
  - Vol clustering: high IV persists for days/weeks.
  - Sweet spot: 35-50% OTM strikes, 120-220 DTE.

Architecture:
  - Data layer:    yfinance for prices, option chains, and intraday data.
  - Signal engine: Composite sell/buy-back signals from multiple indicators.
  - Execution:     Alpaca REST API for limit-order placement (paper trading).
  - Risk:          Hard daily contract limits, volume participation caps.
  - State:         JSON persistence across restarts.

Usage:
    python slvr_cc_scalper.py                 # Dry-run (default)
    python slvr_cc_scalper.py --live          # Live paper-trade orders
    python slvr_cc_scalper.py --status        # Show open positions / state
    python slvr_cc_scalper.py --once          # Run one cycle then exit

Reference: ~/trading_bot/research/deep_otm_cc_strategy.md
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# ET date helper -- all DTE and "today" comparisons must use ET date, not
# the server's local date (which is UTC on most cloud hosts and diverges from
# the ET trading calendar after 20:00 UTC / before midnight ET).

_ET_TZ = pytz.timezone("US/Eastern")


def _today_et() -> "date":
    """Return the current calendar date in US/Eastern time."""
    return datetime.now(_ET_TZ).date()


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, "/opt/jarvis-utils/lib")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from slvr_cc_config import (
    TRADE_TICKERS, LEVERAGED_TICKERS, ENABLE_LEVERAGED,
    GLD_TICKER, UUP_TICKER, SLV_TICKER, USO_TICKER, WATCH_TICKERS,
    STRIKE_OTM_MIN, STRIKE_OTM_MAX, STRIKE_OTM_TARGET,
    DTE_MIN, DTE_MAX, DTE_OPTIMAL_MIN, DTE_OPTIMAL_MAX,
    MIN_PREMIUM, MIN_OPEN_INTEREST, MIN_VOLUME, MAX_BID_ASK_SPREAD_PCT,
    SELL_SLVR_UP_PCT, SELL_RSI_THRESHOLD, RSI_PERIOD,
    HV_SHORT_WINDOW, HV_LONG_WINDOW,
    GLD_RALLY_PCT, UUP_WEAK_PCT,
    GLD_SHORT_TERM_UP_THRESHOLD, MIN_SELL_SIGNALS,
    USO_RALLY_PCT, USO_SHORT_TERM_UP_THRESHOLD,
    BUYBACK_PROFIT_TARGET, BUYBACK_SLVR_DROP_PCT,
    BUYBACK_RSI_THRESHOLD, BUYBACK_IV_CRUSH_POINTS,
    BUYBACK_DTE_REMAINING, GLD_TURNING_UP_THRESHOLD,
    USO_TURNING_UP_THRESHOLD,
    MAX_CONTRACTS_PER_CHAIN_PER_DAY,
    VOLUME_WARN_PCT, VOLUME_HARD_STOP_PCT,
    DEFAULT_CONTRACTS, MAX_CONTRACTS_PER_TRADE, MAX_CONCURRENT_POSITIONS,
    ALPACA_BASE_URL, ALPACA_DATA_URL,
    SELL_OFFSET_FROM_MID, BUYBACK_OFFSET_FROM_MID,
    ORDER_TIF, ALPACA_USER_ID,
    STATE_FILE, LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    MARKET_OPEN, MARKET_CLOSE,
    POLL_INTERVAL_SECONDS, MIN_TRADE_INTERVAL_SECONDS,
    RISK_FREE_RATE,
    YF_HISTORY_PERIOD, YF_INTRADAY_PERIOD, YF_INTRADAY_INTERVAL,
    ALPACA_FEED,
    ORDER_CHECK_INTERVAL, ORDER_MAX_WAIT, ORDER_MAX_ATTEMPTS,
    ORDER_PRICE_STEP, ORDER_MIN_INTERVAL,
    RATE_LIMIT_WINDOW, RATE_LIMIT_MAX_REQUESTS,
    PMCC_ENABLED,
)

# Regime detection config (import with safe defaults for backwards compat)
try:
    from slvr_cc_config import (
        REGIME_ENABLED, REGIME_LOOKBACK_DAYS, REGIME_CACHE_MINUTES,
        REGIME_CONFIDENCE_THRESHOLD, REGIME_WEIGHTS, REGIME_ADJUSTMENTS,
    )
except ImportError:
    REGIME_ENABLED = False
    REGIME_LOOKBACK_DAYS = 90
    REGIME_CACHE_MINUTES = 60
    REGIME_CONFIDENCE_THRESHOLD = 0.60
    REGIME_WEIGHTS = {}
    REGIME_ADJUSTMENTS = {}

from order_manager import OrderManager, ManagedOrder
from pmcc_manager import PMCCManager
from regime_detector import RegimeDetector, Regime

# Energy sector detection (Mar 18) — determines USO vs GLD for macro signals
try:
    from combined_config import ENERGY_SECTORS, TICKER_SECTOR, get_macro_indicator
except ImportError:
    ENERGY_SECTORS = set()
    TICKER_SECTOR = {}
    def get_macro_indicator(ticker: str) -> str:
        return "GLD"

def _is_energy_ticker(ticker: str) -> bool:
    """Return True if ticker is in the energy/oil sector."""
    return TICKER_SECTOR.get(ticker, "") in ENERGY_SECTORS

# ---------------------------------------------------------------------------
# Active ticker helper
# ---------------------------------------------------------------------------

def get_active_tickers() -> List[str]:
    """Return the list of tickers we're actively trading."""
    tickers = list(TRADE_TICKERS)
    if ENABLE_LEVERAGED:
        tickers.extend(LEVERAGED_TICKERS)
    return tickers

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("cc_scalper")


def setup_logging(level: str = "INFO") -> None:
    """Configure console + rotating file logging."""
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
    log_path = BASE_DIR / LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
    )
    fh.setFormatter(fmt)
    fh.setLevel(numeric)
    root.addHandler(fh)


# =============================================================================
# BLACK-SCHOLES PRICING & GREEKS
# =============================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Black-Scholes call option price.

    Args:
        S: Spot price of the underlying.
        K: Strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Annualized volatility.

    Returns:
        Theoretical call option price.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_call_delta(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def bs_call_theta(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Black-Scholes call theta (per calendar day). Negative for long calls."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    term1 = -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
    term2 = -r * K * math.exp(-r * T) * _norm_cdf(d2)
    return (term1 + term2) / 365.0


def bs_call_vega(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Black-Scholes call vega (dollar change per 1% IV move)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * _norm_pdf(d1) * math.sqrt(T) / 100.0


def bs_call_gamma(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """Black-Scholes call gamma."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_pdf(d1) / (S * sigma * math.sqrt(T))


def implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float,
    max_iter: int = 100, tol: float = 1e-6,
) -> float:
    """Newton-Raphson implied volatility solver for a call option.

    Returns annualized IV as a decimal (e.g. 0.75 = 75%).
    Falls back to 0.50 if it cannot converge.
    """
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.50

    sigma = 0.50  # initial guess
    for _ in range(max_iter):
        price = bs_call_price(S, K, T, r, sigma)
        vega_100 = bs_call_vega(S, K, T, r, sigma) * 100.0  # full vega
        if abs(vega_100) < 1e-10:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega_100
        sigma = max(sigma, 0.01)
        sigma = min(sigma, 5.0)
    return sigma


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CCPosition:
    """Tracks a sold covered call position."""
    ticker: str                     # underlying ETF symbol, e.g. "SLVR", "GDX"
    option_symbol: str              # e.g. "SLVR261016C00115000"
    strike: float                   # strike price
    expiration: str                 # YYYY-MM-DD
    dte_at_entry: int               # DTE when sold
    contracts: int                  # number of contracts sold
    sell_price: float               # per-share option price we sold at
    sell_date: str                  # ISO timestamp of sell
    sell_underlying_price: float    # underlying price at time of sell
    sell_iv: float                  # IV at time of sell
    sell_signal_score: int          # number of signals that aligned
    sell_signal_details: str        # human-readable signal breakdown
    status: str = "open"            # open, closed, expired, rolled
    buy_back_price: Optional[float] = None
    buy_back_date: Optional[str] = None
    buy_back_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    alpaca_order_id: Optional[str] = None

    @property
    def days_to_expiration(self) -> int:
        """Current DTE based on today's date."""
        try:
            exp = datetime.strptime(self.expiration, "%Y-%m-%d").date()
            return (exp - _today_et()).days
        except (ValueError, TypeError):
            return 0

    @property
    def premium_collected(self) -> float:
        """Total premium collected in dollars."""
        return self.sell_price * self.contracts * 100

    def unrealized_pnl(self, current_option_price: float) -> float:
        """Unrealized P&L given current option price (per-share)."""
        return (self.sell_price - current_option_price) * self.contracts * 100

    def profit_pct(self, current_option_price: float) -> float:
        """Percentage of max profit captured so far."""
        if self.sell_price <= 0:
            return 0.0
        return (self.sell_price - current_option_price) / self.sell_price

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CCPosition":
        # Handle legacy positions that used sell_slvr_price instead of
        # sell_underlying_price, and that lacked a ticker field.
        mapped = dict(d)
        if "sell_slvr_price" in mapped and "sell_underlying_price" not in mapped:
            mapped["sell_underlying_price"] = mapped.pop("sell_slvr_price")
        elif "sell_slvr_price" in mapped:
            mapped.pop("sell_slvr_price")
        if "ticker" not in mapped:
            # Infer ticker from option_symbol prefix for legacy positions
            sym = mapped.get("option_symbol", "")
            # OCC symbols start with the ticker (padded to 6 chars for standard,
            # but yfinance uses variable length).  Try to extract.
            ticker = ""
            for i, ch in enumerate(sym):
                if ch.isdigit():
                    ticker = sym[:i]
                    break
            mapped["ticker"] = ticker if ticker else "SLVR"
        return cls(**{k: v for k, v in mapped.items() if k in cls.__dataclass_fields__})


@dataclass
class DailyTradeCounter:
    """Tracks daily contract counts per option chain for compliance."""
    date: str                                           # YYYY-MM-DD
    chain_counts: Dict[str, int] = field(default_factory=dict)
    chain_volumes: Dict[str, int] = field(default_factory=dict)

    def contracts_traded(self, chain_key: str) -> int:
        """Number of contracts traded today on the given chain."""
        return self.chain_counts.get(chain_key, 0)

    def record_trade(self, chain_key: str, contracts: int, chain_volume: int = 0) -> None:
        """Record a trade on a chain."""
        self.chain_counts[chain_key] = self.chain_counts.get(chain_key, 0) + contracts
        if chain_volume > 0:
            self.chain_volumes[chain_key] = chain_volume

    def volume_pct(self, chain_key: str) -> float:
        """Our traded contracts as a percentage of chain daily volume."""
        vol = self.chain_volumes.get(chain_key, 0)
        if vol <= 0:
            return 0.0
        return self.chain_counts.get(chain_key, 0) / vol

    def can_trade(self, chain_key: str, contracts: int, chain_volume: int = 0) -> Tuple[bool, str]:
        """Check if a trade is allowed under daily limits.

        Returns (allowed, reason) tuple.
        """
        current = self.contracts_traded(chain_key)
        proposed = current + contracts

        # Hard stop: 50 contracts per chain per day
        if proposed > MAX_CONTRACTS_PER_CHAIN_PER_DAY:
            return False, (
                f"HARD STOP: {proposed} contracts would exceed "
                f"{MAX_CONTRACTS_PER_CHAIN_PER_DAY}/day limit "
                f"(already traded {current} today)"
            )

        # Volume participation check
        if chain_volume > 0:
            pct = proposed / chain_volume
            if pct > VOLUME_HARD_STOP_PCT:
                return False, (
                    f"VOLUME STOP: {proposed} contracts = {pct:.1%} of chain "
                    f"volume ({chain_volume}), exceeds {VOLUME_HARD_STOP_PCT:.0%} limit"
                )
            if pct > VOLUME_WARN_PCT:
                logger.warning(
                    "VOLUME WARNING: %d contracts = %.1f%% of chain volume (%d) "
                    "on %s -- approaching %.0f%% limit",
                    proposed, pct * 100, chain_volume, chain_key,
                    VOLUME_HARD_STOP_PCT * 100,
                )

        return True, "OK"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalperState:
    """Full persistent state for the CC scalper."""
    positions: List[CCPosition] = field(default_factory=list)
    daily_counter: Optional[DailyTradeCounter] = None
    signal_history: List[Dict[str, Any]] = field(default_factory=list)
    total_premium_collected: float = 0.0
    total_realized_pnl: float = 0.0
    total_trades: int = 0
    last_run: Optional[str] = None
    # Edge 104(c) — separates dry-run state from live state. Set on load,
    # consulted by save/load to route to slvr_cc_state.dryrun.json when True.
    _dry_run: bool = field(default=False, repr=False)

    @staticmethod
    def _resolve_path(base: Path, dry_run: bool) -> Path:
        """Edge 104(c): suffix .dryrun before .json when in dry-run mode."""
        if dry_run:
            return base.with_name(base.stem + ".dryrun" + base.suffix)
        return base

    def open_positions(self) -> List[CCPosition]:
        """Return only open positions."""
        return [p for p in self.positions if p.status == "open"]

    def get_daily_counter(self) -> DailyTradeCounter:
        """Get or create the daily trade counter for today."""
        today = _today_et().isoformat()
        if self.daily_counter is None or self.daily_counter.date != today:
            self.daily_counter = DailyTradeCounter(date=today)
        return self.daily_counter

    # PMCC state -- stored alongside regular CC state for unified persistence.
    # This is a dict managed by PMCCManager.to_dict() / from_dict().
    pmcc_state: Optional[Dict[str, Any]] = None

    def save(self, path: Optional[Path] = None) -> None:
        """Persist state to JSON (includes PMCC diagonal spread state)."""
        if path is None:
            path = BASE_DIR / STATE_FILE
        # Edge 104(c) — quarantine dry-run state to a separate file
        path = self._resolve_path(path, self._dry_run)
        data = {
            "positions": [p.to_dict() for p in self.positions],
            "daily_counter": self.daily_counter.to_dict() if self.daily_counter else None,
            "signal_history": self.signal_history[-500:],  # keep last 500
            "total_premium_collected": self.total_premium_collected,
            "total_realized_pnl": self.total_realized_pnl,
            "total_trades": self.total_trades,
            "last_run": self.last_run,
            "pmcc_state": self.pmcc_state,
        }
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, default=str))
            tmp.replace(path)
        except Exception as e:
            logger.error("Failed to save state: %s", e)

    @classmethod
    def load(cls, path: Optional[Path] = None, dry_run: bool = False) -> "ScalperState":
        """Load state from JSON, returning fresh state if file is missing.

        Edge 104(c): when ``dry_run=True`` the path is rewritten to
        ``slvr_cc_state.dryrun.json`` so dry-run cycles don't contaminate
        the live state file.
        """
        if path is None:
            path = BASE_DIR / STATE_FILE
        # Edge 104(c) — route to per-mode file
        path = cls._resolve_path(path, dry_run)
        if not path.exists():
            return cls(_dry_run=dry_run)
        try:
            raw = json.loads(path.read_text())
            positions = [CCPosition.from_dict(p) for p in raw.get("positions", [])]
            dc_raw = raw.get("daily_counter")
            daily_counter = None
            if dc_raw:
                daily_counter = DailyTradeCounter(
                    date=dc_raw.get("date", ""),
                    chain_counts=dc_raw.get("chain_counts", {}),
                    chain_volumes=dc_raw.get("chain_volumes", {}),
                )
            return cls(
                positions=positions,
                daily_counter=daily_counter,
                signal_history=raw.get("signal_history", []),
                total_premium_collected=raw.get("total_premium_collected", 0.0),
                total_realized_pnl=raw.get("total_realized_pnl", 0.0),
                total_trades=raw.get("total_trades", 0),
                last_run=raw.get("last_run"),
                pmcc_state=raw.get("pmcc_state"),
                _dry_run=dry_run,
            )
        except Exception as e:
            logger.error("Failed to load state from %s: %s -- starting fresh", path, e)
            return cls(_dry_run=dry_run)


# =============================================================================
# DATA LAYER
# =============================================================================

class DataLayer:
    """Fetches and caches market data for the multi-ticker CC strategy.

    Uses yfinance for price data, option chains, and intraday bars.
    Computes technical indicators: RSI(14), HV-10, HV-20.
    Monitors metals direction via GLD price movement.
    """

    def __init__(self):
        self._price_cache: Dict[str, float] = {}
        self._daily_data: Dict[str, pd.DataFrame] = {}
        self._intraday_data: Dict[str, pd.DataFrame] = {}
        self._prev_close: Dict[str, float] = {}
        # Option chains keyed by ticker -> {expiration: DataFrame}
        self._option_chains: Dict[str, Dict[str, Any]] = {}
        self._last_fetch_time: float = 0.0
        self._last_intraday_fetch: float = 0.0

    def refresh(self) -> bool:
        """Fetch/refresh all market data. Returns True on success."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed. Run: pip install yfinance")
            return False

        success = True

        # Fetch daily history for all watch tickers (for RSI, HV)
        for ticker in WATCH_TICKERS:
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period=YF_HISTORY_PERIOD)
                if hist is not None and len(hist) > 0:
                    self._daily_data[ticker] = hist
                    self._price_cache[ticker] = float(hist["Close"].iloc[-1])
                    if len(hist) >= 2:
                        self._prev_close[ticker] = float(hist["Close"].iloc[-2])
                    logger.info(
                        "Fetched daily data for %s: %d bars, last=$%.2f",
                        ticker, len(hist), self._price_cache[ticker],
                    )
                else:
                    logger.warning("No daily data returned for %s", ticker)
                    success = False
            except Exception as e:
                logger.error("Failed to fetch daily data for %s: %s", ticker, e)
                success = False

        # Fetch intraday data for GLD (metals direction indicator)
        try:
            gld = yf.Ticker(GLD_TICKER)
            intra = gld.history(period=YF_INTRADAY_PERIOD, interval=YF_INTRADAY_INTERVAL)
            if intra is not None and len(intra) > 0:
                self._intraday_data[GLD_TICKER] = intra
                logger.info("Fetched intraday GLD data: %d bars", len(intra))
            else:
                logger.warning("No intraday data returned for GLD")
        except Exception as e:
            logger.error("Failed to fetch intraday GLD data: %s", e)

        # Fetch intraday data for USO (oil/energy direction indicator, Mar 18)
        try:
            uso = yf.Ticker(USO_TICKER)
            intra_uso = uso.history(period=YF_INTRADAY_PERIOD, interval=YF_INTRADAY_INTERVAL)
            if intra_uso is not None and len(intra_uso) > 0:
                self._intraday_data[USO_TICKER] = intra_uso
                logger.info("Fetched intraday USO data: %d bars", len(intra_uso))
            else:
                logger.warning("No intraday data returned for USO")
        except Exception as e:
            logger.error("Failed to fetch intraday USO data: %s", e)

        self._last_fetch_time = time.time()
        return success

    def refresh_intraday(self) -> bool:
        """Refresh only intraday data (lighter weight, for frequent polling)."""
        try:
            import yfinance as yf
        except ImportError:
            return False

        try:
            gld = yf.Ticker(GLD_TICKER)
            intra = gld.history(period="1d", interval=YF_INTRADAY_INTERVAL)
            if intra is not None and len(intra) > 0:
                self._intraday_data[GLD_TICKER] = intra

            # Also refresh USO intraday (Mar 18)
            uso = yf.Ticker(USO_TICKER)
            intra_uso = uso.history(period="1d", interval=YF_INTRADAY_INTERVAL)
            if intra_uso is not None and len(intra_uso) > 0:
                self._intraday_data[USO_TICKER] = intra_uso

            # Refresh current prices for all active trade tickers + signal tickers
            refresh_tickers = get_active_tickers() + [GLD_TICKER, UUP_TICKER, USO_TICKER]
            for ticker in refresh_tickers:
                t = yf.Ticker(ticker)
                fast = t.history(period="1d")
                if fast is not None and len(fast) > 0:
                    self._price_cache[ticker] = float(fast["Close"].iloc[-1])

            self._last_intraday_fetch = time.time()
            return True
        except Exception as e:
            logger.error("Intraday refresh failed: %s", e)
            return False

    # --- Price accessors ---

    def get_price(self, ticker: str) -> Optional[float]:
        """Get the most recent price for a ticker."""
        return self._price_cache.get(ticker)

    def get_prev_close(self, ticker: str) -> Optional[float]:
        """Get the previous trading day close for a ticker."""
        return self._prev_close.get(ticker)

    def get_daily_change_pct(self, ticker: str) -> Optional[float]:
        """Get the percentage change from previous close to current price."""
        price = self.get_price(ticker)
        prev = self.get_prev_close(ticker)
        if price is None or prev is None or prev <= 0:
            return None
        return (price - prev) / prev

    # --- Technical indicators ---

    def compute_rsi(self, ticker: str, period: int = RSI_PERIOD) -> Optional[float]:
        """Compute RSI(period) for the given ticker from daily closes."""
        df = self._daily_data.get(ticker)
        if df is None or len(df) < period + 1:
            return None
        closes = df["Close"]
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        last_rsi = rsi.iloc[-1]
        if pd.isna(last_rsi):
            return None
        return float(last_rsi)

    def compute_hv(self, ticker: str, window: int) -> Optional[float]:
        """Compute annualized historical volatility over a given window.

        Returns the value as a decimal (e.g. 0.75 = 75%).
        """
        df = self._daily_data.get(ticker)
        if df is None or len(df) < window + 1:
            return None
        closes = df["Close"]
        log_returns = np.log(closes / closes.shift(1)).dropna()
        if len(log_returns) < window:
            return None
        rv = float(log_returns.tail(window).std()) * math.sqrt(252)
        return rv if not math.isnan(rv) else None

    # --- GLD direction (metals leading indicator) ---

    def gld_short_term_change(self, minutes: int = 15) -> Optional[float]:
        """Compute GLD percentage change over the last N minutes of intraday data.

        Uses 5-minute bars. A value > 0 means GLD has been moving up recently.
        """
        df = self._intraday_data.get(GLD_TICKER)
        if df is None or len(df) < 2:
            return None
        bars_needed = max(1, minutes // 5)
        if len(df) < bars_needed + 1:
            bars_needed = len(df) - 1
        recent_close = float(df["Close"].iloc[-1])
        past_close = float(df["Close"].iloc[-(bars_needed + 1)])
        if past_close <= 0:
            return None
        return (recent_close - past_close) / past_close

    def gld_60min_change(self) -> Optional[float]:
        """Compute GLD percentage change over the last 60 minutes."""
        return self.gld_short_term_change(minutes=60)

    def macro_short_term_change(self, indicator: str, minutes: int = 15) -> Optional[float]:
        """Compute percentage change over the last N minutes for any macro indicator.

        Works for GLD, USO, or any ticker with intraday data loaded.
        Added Mar 18 for energy sector support.
        """
        df = self._intraday_data.get(indicator)
        if df is None or len(df) < 2:
            return None
        bars_needed = max(1, minutes // 5)
        if len(df) < bars_needed + 1:
            bars_needed = len(df) - 1
        recent_close = float(df["Close"].iloc[-1])
        past_close = float(df["Close"].iloc[-(bars_needed + 1)])
        if past_close <= 0:
            return None
        return (recent_close - past_close) / past_close

    # --- Option chains ---

    def fetch_option_chain(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch option chain for a specific ticker via yfinance.

        Returns a dict with keys mapping expiration dates to DataFrames of call
        options, filtered to the target strike/DTE range.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return None

        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                logger.warning("No option expirations found for %s", ticker)
                return None

            underlying_price = self.get_price(ticker)
            if underlying_price is None or underlying_price <= 0:
                logger.warning("Cannot fetch option chain: %s price unavailable", ticker)
                return None

            # Calculate strike range
            min_strike = underlying_price * (1 + STRIKE_OTM_MIN)
            max_strike = underlying_price * (1 + STRIKE_OTM_MAX)

            result = {}
            today = _today_et()

            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except ValueError:
                    continue

                dte = (exp_date - today).days
                if dte < DTE_MIN or dte > DTE_MAX:
                    continue

                try:
                    chain = t.option_chain(exp_str)
                    calls = chain.calls
                    if calls is None or len(calls) == 0:
                        continue

                    # Filter by strike range
                    mask = (calls["strike"] >= min_strike) & (calls["strike"] <= max_strike)
                    filtered = calls[mask].copy()

                    if len(filtered) == 0:
                        continue

                    # Add computed columns
                    filtered["dte"] = dte
                    filtered["otm_pct"] = (filtered["strike"] - underlying_price) / underlying_price
                    filtered["expiration"] = exp_str

                    # Compute mid price
                    if "bid" in filtered.columns and "ask" in filtered.columns:
                        filtered["mid_price"] = (
                            filtered["bid"].fillna(0) + filtered["ask"].fillna(0)
                        ) / 2.0
                    else:
                        filtered["mid_price"] = filtered.get(
                            "lastPrice", pd.Series(0.0, index=filtered.index)
                        )

                    result[exp_str] = filtered
                    logger.debug(
                        "Option chain %s/%s (DTE=%d): %d strikes in range $%.0f-$%.0f",
                        ticker, exp_str, dte, len(filtered), min_strike, max_strike,
                    )

                except Exception as e:
                    logger.warning("Failed to fetch chain for %s/%s: %s", ticker, exp_str, e)
                    continue

            self._option_chains[ticker] = result
            logger.info(
                "Fetched option chains for %s: %d expirations with eligible strikes",
                ticker, len(result),
            )
            return result

        except Exception as e:
            logger.error("Option chain fetch failed for %s: %s", ticker, e)
            return None

    def fetch_all_option_chains(self) -> None:
        """Fetch option chains for all active trade tickers."""
        active = get_active_tickers()
        for ticker in active:
            self.fetch_option_chain(ticker)

    def get_option_chains(self, ticker: str) -> Dict[str, Any]:
        """Return cached option chains for a specific ticker."""
        return self._option_chains.get(ticker, {})


# =============================================================================
# SIGNAL ENGINE
# =============================================================================

@dataclass
class SignalResult:
    """Result of a signal evaluation."""
    signal_type: str             # "sell" or "buy_back"
    triggered: bool              # whether the signal fired
    score: int                   # number of sub-signals that aligned
    max_score: int               # total possible sub-signals
    details: Dict[str, Any]      # per-component details
    timestamp: str = ""          # ISO timestamp

    def summary(self) -> str:
        """Human-readable summary of the signal."""
        parts = []
        for key, val in self.details.items():
            if isinstance(val, dict):
                status = "YES" if val.get("triggered", False) else "NO"
                reason = val.get("reason", "")
                parts.append(f"  {key}: {status} -- {reason}")
            else:
                parts.append(f"  {key}: {val}")
        return (
            f"{self.signal_type.upper()} signal: {'TRIGGERED' if self.triggered else 'NOT triggered'} "
            f"(score={self.score}/{self.max_score})\n" + "\n".join(parts)
        )


class SignalEngine:
    """Evaluates sell and buy-back signals for the multi-ticker CC strategy.

    Sell signals (all or most must align):
      1. Ticker up 3%+ on the day (per-ticker)
      2. RSI > 65 (per-ticker)
      3. HV-10 > HV-20 (vol expanding, per-ticker)
      4. GLD rallying (global sector momentum)
      5. UUP weakening (global dollar down)
      6. GLD NOT trending up in last 15-30 min (global -- don't sell into active rip)

    Buy-back signals (any one triggers):
      1. 50% profit target hit
      2. Underlying drops 5%+ from sell date
      3. RSI < 35 (per-ticker)
      4. IV crush > 15 points
      5. 60 DTE remaining
      6. GLD turning up after pullback (global)
    """

    def __init__(self, data: DataLayer):
        self.data = data

    def evaluate_sell(
        self, ticker: str, regime_adjustments: Optional[Dict[str, Any]] = None,
    ) -> SignalResult:
        """Evaluate whether conditions are right to sell covered calls on a ticker.

        Args:
            ticker: The ticker to evaluate sell signals for.
            regime_adjustments: Optional dict from RegimeDetector.get_regime_adjustments()
                with 'signal_threshold_boost' to adjust the minimum-signals threshold.

        Returns a SignalResult indicating if enough sell conditions are met.
        """
        details = {}
        score = 0
        max_score = 6

        # 1. Ticker up 3%+ on the day
        ticker_change = self.data.get_daily_change_pct(ticker)
        if ticker_change is not None:
            triggered = ticker_change >= SELL_SLVR_UP_PCT
            details["ticker_up_3pct"] = {
                "triggered": triggered,
                "value": f"{ticker_change:.2%}",
                "threshold": f"{SELL_SLVR_UP_PCT:.2%}",
                "reason": f"{ticker} {'up' if ticker_change >= 0 else 'down'} {abs(ticker_change):.2%} "
                          f"(need {SELL_SLVR_UP_PCT:.0%}+)",
            }
            if triggered:
                score += 1
        else:
            details["ticker_up_3pct"] = {
                "triggered": False, "reason": f"{ticker} daily change unavailable",
            }

        # 2. RSI > 65
        rsi = self.data.compute_rsi(ticker)
        if rsi is not None:
            triggered = rsi > SELL_RSI_THRESHOLD
            details["rsi_overbought"] = {
                "triggered": triggered,
                "value": f"{rsi:.1f}",
                "threshold": f"{SELL_RSI_THRESHOLD}",
                "reason": f"RSI({RSI_PERIOD})={rsi:.1f} (need >{SELL_RSI_THRESHOLD})",
            }
            if triggered:
                score += 1
        else:
            details["rsi_overbought"] = {
                "triggered": False, "reason": "RSI unavailable",
            }

        # 3. HV-10 > HV-20 (vol expanding)
        hv10 = self.data.compute_hv(ticker, HV_SHORT_WINDOW)
        hv20 = self.data.compute_hv(ticker, HV_LONG_WINDOW)
        if hv10 is not None and hv20 is not None:
            triggered = hv10 > hv20
            details["vol_expanding"] = {
                "triggered": triggered,
                "value": f"HV10={hv10:.1%} HV20={hv20:.1%}",
                "reason": f"HV10 {'>' if hv10 > hv20 else '<='} HV20 "
                          f"({hv10:.1%} vs {hv20:.1%})",
            }
            if triggered:
                score += 1
        else:
            details["vol_expanding"] = {
                "triggered": False, "reason": "HV data unavailable",
            }

        # 4. Macro indicator rallying (sector momentum) -- global signal
        # Energy tickers use USO; mining tickers use GLD (Mar 18)
        if _is_energy_ticker(ticker):
            macro_ticker = USO_TICKER
            macro_rally_pct = USO_RALLY_PCT
            macro_label = "USO"
        else:
            macro_ticker = GLD_TICKER
            macro_rally_pct = GLD_RALLY_PCT
            macro_label = "GLD"
        macro_change = self.data.get_daily_change_pct(macro_ticker)
        if macro_change is not None:
            triggered = macro_change >= macro_rally_pct
            details["macro_rallying"] = {
                "triggered": triggered,
                "value": f"{macro_change:.2%}",
                "threshold": f"{macro_rally_pct:.2%}",
                "reason": f"{macro_label} {'up' if macro_change >= 0 else 'down'} {abs(macro_change):.2%} "
                          f"(need {macro_rally_pct:.2%}+)",
            }
            if triggered:
                score += 1
        else:
            details["macro_rallying"] = {
                "triggered": False, "reason": f"{macro_label} daily change unavailable",
            }

        # 5. UUP weakening (dollar down) -- global signal
        uup_change = self.data.get_daily_change_pct(UUP_TICKER)
        if uup_change is not None:
            triggered = uup_change <= UUP_WEAK_PCT
            details["uup_weakening"] = {
                "triggered": triggered,
                "value": f"{uup_change:.2%}",
                "threshold": f"<={UUP_WEAK_PCT:.2%}",
                "reason": f"UUP {'down' if uup_change < 0 else 'up'} {abs(uup_change):.2%} "
                          f"(need <={UUP_WEAK_PCT:.2%})",
            }
            if triggered:
                score += 1
        else:
            details["uup_weakening"] = {
                "triggered": False, "reason": "UUP daily change unavailable",
            }

        # 6. Macro indicator NOT trending up in last 15-30 minutes -- global signal
        # This prevents selling into an active rip -- wait for the spike to peak
        # Energy tickers use USO; mining tickers use GLD (Mar 18)
        if _is_energy_ticker(ticker):
            macro_15m = self.data.macro_short_term_change(USO_TICKER, minutes=15)
            macro_threshold = USO_SHORT_TERM_UP_THRESHOLD
            macro_label_6 = "USO"
        else:
            macro_15m = self.data.gld_short_term_change(minutes=15)
            macro_threshold = GLD_SHORT_TERM_UP_THRESHOLD
            macro_label_6 = "GLD"
        if macro_15m is not None:
            # Signal fires when macro indicator is NOT still ripping
            triggered = macro_15m < macro_threshold
            details["macro_not_ripping"] = {
                "triggered": triggered,
                "value": f"{macro_15m:.3%}",
                "threshold": f"<{macro_threshold:.3%}",
                "reason": (
                    f"{macro_label_6} 15-min change={macro_15m:.3%} "
                    f"({'safe to sell' if triggered else 'STILL RIPPING -- wait'})"
                ),
            }
            if triggered:
                score += 1
        else:
            # If we cannot check intraday, assume it is fine (fail-open)
            details["macro_not_ripping"] = {
                "triggered": True,
                "reason": f"{macro_label_6} intraday data unavailable -- assuming safe (fail-open)",
            }
            score += 1

        now_iso = datetime.now(timezone.utc).isoformat()

        # Apply regime-based threshold adjustment
        effective_threshold = MIN_SELL_SIGNALS
        if regime_adjustments:
            boost = regime_adjustments.get("signal_threshold_boost", 0.0)
            # boost is a fraction of max_score (e.g. 0.15 -> +0.9 on a 6-point scale)
            effective_threshold = max(1, min(
                max_score, MIN_SELL_SIGNALS + round(boost * max_score)
            ))
            if boost != 0.0:
                details["regime_threshold_adj"] = {
                    "triggered": True,
                    "reason": (
                        f"Regime threshold: {MIN_SELL_SIGNALS} -> {effective_threshold} "
                        f"(boost={boost:+.2f})"
                    ),
                }

        triggered_overall = score >= effective_threshold

        return SignalResult(
            signal_type="sell",
            triggered=triggered_overall,
            score=score,
            max_score=max_score,
            details=details,
            timestamp=now_iso,
        )

    def evaluate_buy_back(
        self, position: CCPosition, current_option_price: float,
    ) -> SignalResult:
        """Evaluate whether to buy back an existing covered call position.

        Any single buy-back condition triggers a buy-back.

        Args:
            position: The open CCPosition to evaluate.
            current_option_price: Current per-share price of the option.

        Returns:
            SignalResult with triggered=True if any buy-back condition is met.
        """
        details = {}
        score = 0
        max_score = 6
        ticker = position.ticker

        # 1. 50% profit target hit
        profit_pct = position.profit_pct(current_option_price)
        triggered = profit_pct >= BUYBACK_PROFIT_TARGET
        details["profit_target"] = {
            "triggered": triggered,
            "value": f"{profit_pct:.1%}",
            "threshold": f"{BUYBACK_PROFIT_TARGET:.0%}",
            "reason": (
                f"Captured {profit_pct:.1%} of premium "
                f"(sell=${position.sell_price:.2f} current=${current_option_price:.2f})"
            ),
        }
        if triggered:
            score += 1

        # 2. Underlying drops 5%+ from sell date price
        underlying_price = self.data.get_price(ticker)
        if underlying_price is not None and position.sell_underlying_price > 0:
            drop = (underlying_price - position.sell_underlying_price) / position.sell_underlying_price
            triggered = drop <= -BUYBACK_SLVR_DROP_PCT
            details["underlying_pullback"] = {
                "triggered": triggered,
                "value": f"{drop:.2%}",
                "threshold": f"<=-{BUYBACK_SLVR_DROP_PCT:.0%}",
                "reason": (
                    f"{ticker} at ${underlying_price:.2f}, "
                    f"{'down' if drop < 0 else 'up'} {abs(drop):.2%} from "
                    f"sell price ${position.sell_underlying_price:.2f}"
                ),
            }
            if triggered:
                score += 1
        else:
            details["underlying_pullback"] = {
                "triggered": False, "reason": f"{ticker} price data unavailable",
            }

        # 3. RSI < 35 (oversold)
        rsi = self.data.compute_rsi(ticker)
        if rsi is not None:
            triggered = rsi < BUYBACK_RSI_THRESHOLD
            details["rsi_oversold"] = {
                "triggered": triggered,
                "value": f"{rsi:.1f}",
                "threshold": f"<{BUYBACK_RSI_THRESHOLD}",
                "reason": f"RSI({RSI_PERIOD})={rsi:.1f} (need <{BUYBACK_RSI_THRESHOLD})",
            }
            if triggered:
                score += 1
        else:
            details["rsi_oversold"] = {
                "triggered": False, "reason": "RSI unavailable",
            }

        # 4. IV crush > 15 points
        current_iv = self._estimate_current_iv(position, current_option_price)
        if current_iv is not None and position.sell_iv > 0:
            iv_change = (position.sell_iv - current_iv) * 100  # to percentage points
            triggered = iv_change >= BUYBACK_IV_CRUSH_POINTS
            details["iv_crush"] = {
                "triggered": triggered,
                "value": f"{iv_change:.1f} pts",
                "threshold": f">={BUYBACK_IV_CRUSH_POINTS} pts",
                "reason": (
                    f"IV dropped {iv_change:.1f} pts "
                    f"(sell IV={position.sell_iv:.1%}, current IV={current_iv:.1%})"
                ),
            }
            if triggered:
                score += 1
        else:
            details["iv_crush"] = {
                "triggered": False, "reason": "IV estimation unavailable",
            }

        # 5. 60 DTE remaining (force close / roll window)
        dte = position.days_to_expiration
        triggered = dte <= BUYBACK_DTE_REMAINING
        details["dte_threshold"] = {
            "triggered": triggered,
            "value": f"{dte} days",
            "threshold": f"<={BUYBACK_DTE_REMAINING} days",
            "reason": f"{dte} DTE remaining (close at <={BUYBACK_DTE_REMAINING})",
        }
        if triggered:
            score += 1

        # 6. Macro indicator turning up after pullback -> buy back before premium rises (global)
        # Energy tickers use USO; mining tickers use GLD (Mar 18)
        if _is_energy_ticker(position.ticker):
            bb_macro_ticker = USO_TICKER
            bb_macro_label = "USO"
            bb_turning_threshold = USO_TURNING_UP_THRESHOLD
            bb_macro_change = self.data.get_daily_change_pct(USO_TICKER)
            bb_macro_15m = self.data.macro_short_term_change(USO_TICKER, minutes=15)
        else:
            bb_macro_ticker = GLD_TICKER
            bb_macro_label = "GLD"
            bb_turning_threshold = GLD_TURNING_UP_THRESHOLD
            bb_macro_change = self.data.get_daily_change_pct(GLD_TICKER)
            bb_macro_15m = self.data.gld_short_term_change(minutes=15)
        if bb_macro_change is not None and bb_macro_15m is not None:
            # Macro was down on the day but turning up in last 15 min
            macro_was_down = bb_macro_change < 0
            macro_turning_up = bb_macro_15m >= bb_turning_threshold
            triggered = macro_was_down and macro_turning_up
            details["macro_turning_up"] = {
                "triggered": triggered,
                "value": f"daily={bb_macro_change:.2%}, 15m={bb_macro_15m:.3%}",
                "reason": (
                    f"{bb_macro_label} daily={bb_macro_change:.2%}, 15m={bb_macro_15m:.3%} "
                    f"({'turning up after pullback' if triggered else 'no reversal'})"
                ),
            }
            if triggered:
                score += 1
        else:
            details["macro_turning_up"] = {
                "triggered": False, "reason": f"{bb_macro_label} data unavailable",
            }

        now_iso = datetime.now(timezone.utc).isoformat()
        # ANY single buy-back signal triggers
        triggered_overall = score >= 1

        return SignalResult(
            signal_type="buy_back",
            triggered=triggered_overall,
            score=score,
            max_score=max_score,
            details=details,
            timestamp=now_iso,
        )

    def _estimate_current_iv(
        self, position: CCPosition, current_option_price: float,
    ) -> Optional[float]:
        """Estimate current implied volatility using Newton-Raphson on BS model."""
        underlying_price = self.data.get_price(position.ticker)
        if underlying_price is None or underlying_price <= 0:
            return None
        dte = position.days_to_expiration
        if dte <= 0:
            return None
        T = dte / 365.0
        iv = implied_volatility(
            market_price=current_option_price,
            S=underlying_price,
            K=position.strike,
            T=T,
            r=RISK_FREE_RATE,
        )
        return iv


# =============================================================================
# STRIKE SELECTOR
# =============================================================================

class StrikeSelector:
    """Selects optimal strikes from an option chain for a given ticker.

    Filters:
      - 35-50% OTM
      - 120-220 DTE (prefers 150-180)
      - Minimum premium $3.00/contract
      - Adequate OI and volume
      - Bid-ask spread < MAX_BID_ASK_SPREAD_PCT (filters illiquid options)
    """

    def __init__(self, data: DataLayer):
        self.data = data

    def select_strikes(
        self, ticker: str, regime_adjustments: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return a ranked list of candidate strikes for selling covered calls.

        Each candidate is a dict with keys: strike, expiration, dte, otm_pct,
        mid_price, bid, ask, oi, volume, score.

        Args:
            ticker: The ticker symbol to select strikes for.
            regime_adjustments: Optional dict from RegimeDetector.get_regime_adjustments()
                with 'min_otm_pct' to override the minimum OTM percentage filter.
        """
        chains = self.data.get_option_chains(ticker)
        if not chains:
            logger.warning("No option chains available for %s strike selection", ticker)
            return []

        underlying_price = self.data.get_price(ticker)
        if underlying_price is None or underlying_price <= 0:
            return []

        # Apply regime-based OTM override if provided
        effective_otm_min = STRIKE_OTM_MIN
        if regime_adjustments:
            regime_otm = regime_adjustments.get("min_otm_pct")
            if regime_otm is not None:
                effective_otm_min = regime_otm
                logger.info(
                    "Strike selection [%s]: regime override min OTM %.0f%% -> %.0f%%",
                    ticker, STRIKE_OTM_MIN * 100, effective_otm_min * 100,
                )

        candidates = []

        for exp_str, df in chains.items():
            if df is None or len(df) == 0:
                continue

            for _, row in df.iterrows():
                import math
                strike = float(row.get("strike", 0))
                mid = float(row.get("mid_price", 0))
                bid = float(row.get("bid", 0))
                ask = float(row.get("ask", 0))
                _oi = row.get("openInterest", 0)
                oi = int(_oi) if _oi is not None and not (isinstance(_oi, float) and math.isnan(_oi)) else 0
                _vol = row.get("volume", 0)
                vol = int(_vol) if _vol is not None and not (isinstance(_vol, float) and math.isnan(_vol)) else 0
                _dte = row.get("dte", 0)
                dte = int(_dte) if _dte is not None and not (isinstance(_dte, float) and math.isnan(_dte)) else 0
                otm_pct = float(row.get("otm_pct", 0))
                contract_symbol = str(row.get("contractSymbol", ""))

                # Filter: regime-adjusted minimum OTM
                if otm_pct < effective_otm_min:
                    continue

                # Filter: minimum premium
                effective_price = mid if mid > 0 else float(row.get("lastPrice", 0))
                if effective_price < MIN_PREMIUM:
                    continue

                # Filter: open interest
                if oi < MIN_OPEN_INTEREST:
                    continue

                # Filter: volume (if configured)
                if MIN_VOLUME > 0 and vol < MIN_VOLUME:
                    continue

                # Filter: bid-ask spread -- skip illiquid options
                if bid > 0 and ask > 0:
                    mid_calc = (bid + ask) / 2.0
                    if mid_calc > 0:
                        spread_pct = (ask - bid) / mid_calc
                        if spread_pct > MAX_BID_ASK_SPREAD_PCT:
                            logger.debug(
                                "Skipping %s %s $%.0f: spread=%.0f%% > %.0f%% max",
                                ticker, exp_str, strike,
                                spread_pct * 100, MAX_BID_ASK_SPREAD_PCT * 100,
                            )
                            continue

                # Score the candidate (higher = better)
                score = 0.0

                # Prefer strikes near the target OTM %
                otm_dist = abs(otm_pct - STRIKE_OTM_TARGET)
                score += max(0, 10 - otm_dist * 50)  # up to 10 points

                # Prefer optimal DTE range
                if DTE_OPTIMAL_MIN <= dte <= DTE_OPTIMAL_MAX:
                    score += 10
                elif DTE_MIN <= dte <= DTE_MAX:
                    score += 5

                # Prefer higher premium
                score += min(effective_price, 15.0)  # up to 15 points

                # Prefer higher OI (liquidity)
                score += min(oi / 100.0, 5.0)  # up to 5 points

                # Prefer some volume
                score += min(vol / 50.0, 5.0)  # up to 5 points

                candidates.append({
                    "strike": strike,
                    "expiration": exp_str,
                    "dte": dte,
                    "otm_pct": otm_pct,
                    "mid_price": mid,
                    "effective_price": effective_price,
                    "bid": bid,
                    "ask": ask,
                    "oi": oi,
                    "volume": vol,
                    "contract_symbol": contract_symbol,
                    "score": score,
                })

        # Sort by score descending
        candidates.sort(key=lambda c: c["score"], reverse=True)

        logger.info(
            "Strike selection for %s: %d candidates from %d expirations ($%.2f)",
            ticker, len(candidates), len(chains), underlying_price,
        )
        for i, c in enumerate(candidates[:5]):
            logger.info(
                "  #%d: $%.0f %s (DTE=%d, OTM=%.0f%%, mid=$%.2f, OI=%d, score=%.1f)",
                i + 1, c["strike"], c["expiration"], c["dte"],
                c["otm_pct"] * 100, c["mid_price"], c["oi"], c["score"],
            )

        return candidates


# =============================================================================
# EXECUTION LAYER
# =============================================================================

def _http_detail(e: Exception) -> str:
    """Extract HTTP status + response body from a requests exception.

    Apr 21 2026 — cc_scalper error logs were stripping the Alpaca response
    body, which hid structured error codes (40310000 insufficient-qty,
    40310001 naked-short-call rejection, etc.). `requests.HTTPError.__str__()`
    only includes the status line and URL. This helper appends the response
    body (truncated) when available so grep-for-error surfaces the
    actual reason without re-running the order.

    Returns the exception string unchanged if no response is attached.
    """
    resp = getattr(e, "response", None)
    if resp is None:
        return str(e)
    try:
        body = (resp.text or "")[:300]
    except Exception:
        body = ""
    if body:
        return f"{e} | body={body}"
    return str(e)


class ExecutionLayer:
    """Handles order placement via Alpaca REST API.

    ALWAYS uses limit orders. Sells above mid into spikes, buys back below mid
    on slams.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None
        self._session = None

    def initialize(self) -> bool:
        """Load Alpaca credentials and create a requests session."""
        try:
            from jarvis_utils.secrets import get
            self.api_key = get("Alpaca", "api_key_id", user=ALPACA_USER_ID)
            self.api_secret = get("Alpaca", "secret_key", user=ALPACA_USER_ID)
            if not self.api_key or not self.api_secret:
                logger.warning("Alpaca credentials not found -- forcing dry-run")
                self.dry_run = True
                return False
            logger.info("Alpaca credentials loaded for user %s", ALPACA_USER_ID)
            return True
        except Exception as e:
            logger.error("Failed to load Alpaca credentials: %s", e)
            self.dry_run = True
            return False

    def _refresh_session_credentials(self, session):
        """Edge 123 root fix (ported to cc_scalper Apr 22 2026):
        Re-pull Alpaca credentials from the portal and update session
        headers. Called when a 401 is observed. Safe to call repeatedly.

        Root cause of incident Apr 22 13:00 UTC: Alpaca rotated paper-key
        server-side; cc_scalper's plain requests.Session cached the stale
        key for hours until manual restart. Edge 129 fail-closed prevented
        bad trades, but sells were blocked entirely. Migrating to the same
        auto-refresh wrapper that alpaca_client.py uses.
        """
        from jarvis_utils.secrets import get
        new_key = get("Alpaca", "api_key_id", user=ALPACA_USER_ID)
        new_secret = get("Alpaca", "secret_key", user=ALPACA_USER_ID)
        if not new_key or not new_secret:
            raise EnvironmentError(
                "cc_scalper credential refresh: portal returned empty creds"
            )
        self.api_key = new_key
        self.api_secret = new_secret
        session.headers["APCA-API-KEY-ID"] = new_key
        session.headers["APCA-API-SECRET-KEY"] = new_secret
        logger.info(
            "cc_scalper: Alpaca credentials refreshed (key prefix %s...)",
            new_key[:6],
        )

    def _get_session(self):
        """Get or create an auto-refresh requests session with Alpaca headers.

        Edge 123 port (Apr 22 2026): uses the alpaca_client._AutoRefreshSession
        wrapper so we transparently re-fetch creds on 401 and retry once.

        Edge 132 v2 (Apr 22 2026): also invalidates the cached session if the
        cached header key differs from current self.api_key. Pre-fix bug:
        if _get_session() was called BEFORE initialize() populated
        self.api_key (e.g. PMCCManager.from_dict() invokes
        reconcile_contracts() during constructor, before
        AlpacaCCScalper.initialize() runs), the session got cached with
        empty-string auth headers. With Edge 123 alone, the FIRST request
        would 401 and trigger a refresh-and-retry — correct but wasteful.
        With this invalidation, the second _get_session() call (after
        initialize) rebuilds the session with valid headers and skips the
        wasted 401 round-trip.
        """
        cached_key = (
            self._session.headers.get("APCA-API-KEY-ID", "")
            if self._session is not None
            else ""
        )
        current_key = self.api_key or ""
        if self._session is None or cached_key != current_key:
            # Lazy import to avoid circular deps at module load.
            from alpaca_client import _AutoRefreshSession
            self._session = _AutoRefreshSession(self._refresh_session_credentials)
            self._session.headers.update({
                "APCA-API-KEY-ID": current_key,
                "APCA-API-SECRET-KEY": self.api_secret or "",
            })
        return self._session

    def get_position(self, symbol: str):
        """Fetch a single position from Alpaca.

        Returns the position dict on 200, None on 404 (no position),
        and raises on any other status. Used by Edge 104 buy-back
        guard to verify a real short exists before submitting a BUY.
        """
        sess = self._get_session()
        r = sess.get(
            f"https://paper-api.alpaca.markets/v2/positions/{symbol}",
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None
        raise RuntimeError(
            f"Alpaca position lookup HTTP {r.status_code}: {r.text[:200]}"
        )

    # ------------------------------------------------------------------
    # Edge 129 (Apr 21 2026) — short-call coverage pre-check.
    #
    # Prior behavior: sell_call() posted SELL orders to Alpaca without
    # verifying that we hold the underlying. Alpaca's "uncovered option"
    # permission check was the ONLY thing preventing naked-short-call
    # execution. If those permissions ever flip on a live account,
    # naked shorts would execute immediately.
    #
    # This guard fails CLOSED: any uncertainty (parse error, API error,
    # unexpected side) returns False and blocks the submit.
    #
    # Coverage rules (must match Alpaca's own logic):
    #   1. Equity-covered call: long shares of underlying >= contracts*100
    #   2. PMCC / diagonal: long call on same underlying with
    #      strike_long <= strike_short AND expiry_long >= expiry_short
    #      AND long qty >= contracts
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_occ(symbol: str):
        """Parse an OCC option symbol into (root, expiry_YYYY-MM-DD, side, strike).

        Example: 'COP261120C00170000' -> ('COP', '2026-11-20', 'C', 170.0)
        Raises ValueError if the symbol does not parse cleanly.
        """
        # Find first digit — root ends there
        root_end = -1
        for i, ch in enumerate(symbol):
            if ch.isdigit():
                root_end = i
                break
        if root_end <= 0:
            raise ValueError(f"no ticker root in {symbol}")
        root = symbol[:root_end]
        rest = symbol[root_end:]
        if len(rest) < 15:
            raise ValueError(f"symbol too short after root: {symbol}")
        date_part = rest[:6]
        side = rest[6]
        if side not in ("C", "P"):
            raise ValueError(f"invalid side '{side}' in {symbol}")
        strike_raw = rest[7:15]
        if not strike_raw.isdigit():
            raise ValueError(f"invalid strike in {symbol}")
        strike = int(strike_raw) / 1000.0
        expiry = f"20{date_part[0:2]}-{date_part[2:4]}-{date_part[4:6]}"
        return root, expiry, side, strike

    def _is_short_call_covered(
        self, contract_symbol: str, contracts: int,
    ) -> bool:
        """Return True iff we have equity or PMCC/diagonal coverage.

        Fails CLOSED — any parse error, API error, or unexpected condition
        returns False and blocks the submit. A False return is SAFE even if
        we happen to have coverage (we just miss a legit SELL); a True
        return on a genuinely uncovered short is the catastrophic failure
        mode this guard exists to prevent.
        """
        try:
            root, short_exp, short_side, short_strike = self._parse_occ(
                contract_symbol,
            )
        except Exception as e:
            logger.error(
                "COVERAGE CHECK FAILED (parse): %s -- %s -- fail-closed",
                contract_symbol, e,
            )
            return False

        if short_side != "C":
            # This guard is for short CALLS only. Short puts have
            # different coverage semantics (cash-secured, etc.) — don't
            # accidentally approve one here.
            logger.error(
                "COVERAGE CHECK FAILED: %s is not a call -- fail-closed",
                contract_symbol,
            )
            return False

        # Pull all positions in one call
        try:
            sess = self._get_session()
            r = sess.get(
                f"{ALPACA_BASE_URL}/v2/positions",
                timeout=15,
            )
            if r.status_code != 200:
                logger.error(
                    "COVERAGE CHECK FAILED: positions HTTP %d -- fail-closed",
                    r.status_code,
                )
                return False
            positions = r.json()
        except Exception as e:
            logger.error(
                "COVERAGE CHECK FAILED: positions fetch exception: %s"
                " -- fail-closed",
                e,
            )
            return False

        # Rule 1: equity coverage — long shares of underlying >= contracts*100
        for p in positions:
            try:
                sym = p.get("symbol", "")
                if sym != root:
                    continue
                # Defensive: equity symbols are short (<=6 chars typical);
                # options are longer. Belt-and-suspenders.
                if len(sym) > 6:
                    continue
                qty = int(float(p.get("qty", 0)))
                if p.get("side", "") == "long" and qty >= contracts * 100:
                    logger.info(
                        "COVERAGE OK (equity): %d x %s covered by %d shares of %s",
                        contracts, contract_symbol, qty, root,
                    )
                    return True
            except Exception:
                continue

        # Rule 2: PMCC / diagonal coverage — long call on same underlying
        # with strike_long <= strike_short AND expiry_long >= expiry_short.
        for p in positions:
            try:
                sym = p.get("symbol", "")
                if not sym.startswith(root):
                    continue
                if len(sym) <= len(root):
                    continue
                # Prevent prefix-matching traps: 'COPX...' must not be
                # counted as coverage for a short on 'COP'. The char right
                # after the root MUST be a digit for this to be an OCC
                # option whose underlying is exactly `root`.
                if not sym[len(root)].isdigit():
                    continue
                try:
                    p_root, p_exp, p_side, p_strike = self._parse_occ(sym)
                except Exception:
                    continue
                if p_root != root or p_side != "C":
                    continue
                p_qty = int(float(p.get("qty", 0)))
                if p.get("side", "") != "long" or p_qty < contracts:
                    continue
                # Coverage: long strike <= short strike AND
                # long expiry >= short expiry (ISO strings compare correctly).
                if p_strike <= short_strike and p_exp >= short_exp:
                    logger.info(
                        "COVERAGE OK (PMCC): %d x %s covered by long %s"
                        " (qty=%d, strike=%.2f, exp=%s)",
                        contracts, contract_symbol, sym, p_qty, p_strike, p_exp,
                    )
                    return True
            except Exception:
                continue

        logger.warning(
            "COVERAGE BLOCKED: %d x %s (underlying=%s strike=%.2f exp=%s)"
            " -- no equity and no PMCC/diagonal long-call coverage found."
            " Submit skipped to prevent naked-short-call execution.",
            contracts, contract_symbol, root, short_strike, short_exp,
        )
        return False

    def sell_call(
        self,
        contract_symbol: str,
        contracts: int,
        limit_price: float,
    ) -> Optional[str]:
        """Place a limit sell order for covered calls.

        Args:
            contract_symbol: OCC option symbol (e.g. "SLVR261016C00115000").
            contracts: Number of contracts to sell.
            limit_price: Limit price per share (sell above mid).

        Returns:
            Alpaca order ID on success, None on failure.
        """
        if self.dry_run:
            order_id = f"DRY_SELL_{contract_symbol}_{int(time.time())}"
            logger.info(
                "[DRY RUN] SELL %d x %s @ $%.2f (limit) -- order_id=%s",
                contracts, contract_symbol, limit_price, order_id,
            )
            return order_id

        # Edge 129 guard — block naked short calls before submit.
        if not self._is_short_call_covered(contract_symbol, contracts):
            logger.error(
                "SELL BLOCKED (UNCOVERED): %d x %s @ $%.2f"
                " -- coverage check failed (Edge 129).",
                contracts, contract_symbol, limit_price,
            )
            return None

        try:
            session = self._get_session()
            payload = {
                "symbol": contract_symbol,
                "qty": str(contracts),
                "side": "sell",
                "type": "limit",
                "time_in_force": ORDER_TIF,
                "limit_price": str(round(limit_price, 2)),
            }
            resp = session.post(
                f"{ALPACA_BASE_URL}/v2/orders",
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            order = resp.json()
            order_id = order.get("id", "")
            logger.info(
                "SELL ORDER PLACED: %d x %s @ $%.2f -- order_id=%s, status=%s",
                contracts, contract_symbol, limit_price,
                order_id, order.get("status", "?"),
            )
            return order_id
        except Exception as e:
            logger.error(
                "SELL ORDER FAILED: %d x %s @ $%.2f -- %s",
                contracts, contract_symbol, limit_price, _http_detail(e),
            )
            return None

    def buy_back_call(
        self,
        contract_symbol: str,
        contracts: int,
        limit_price: float,
        tif: Optional[str] = None,
    ) -> Optional[str]:
        """Place a limit buy order to close (buy back) sold calls.

        Args:
            contract_symbol: OCC option symbol.
            contracts: Number of contracts to buy back.
            limit_price: Limit price per share (buy below mid).
            tif: Time-in-force override (default: ORDER_TIF from config).
                 Use "gtc" for PMCC profit-taking buy-backs to avoid
                 cancel/replace spam.

        Returns:
            Alpaca order ID on success, None on failure.
        """
        if self.dry_run:
            order_id = f"DRY_BUY_{contract_symbol}_{int(time.time())}"
            logger.info(
                "[DRY RUN] BUY BACK %d x %s @ $%.2f (limit, tif=%s) -- order_id=%s",
                contracts, contract_symbol, limit_price, tif or ORDER_TIF, order_id,
            )
            return order_id

        try:
            session = self._get_session()
            payload = {
                "symbol": contract_symbol,
                "qty": str(contracts),
                "side": "buy",
                "type": "limit",
                "time_in_force": tif or ORDER_TIF,
                "limit_price": str(round(limit_price, 2)),
            }
            resp = session.post(
                f"{ALPACA_BASE_URL}/v2/orders",
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            order = resp.json()
            order_id = order.get("id", "")
            logger.info(
                "BUY BACK ORDER PLACED: %d x %s @ $%.2f -- order_id=%s, status=%s",
                contracts, contract_symbol, limit_price,
                order_id, order.get("status", "?"),
            )
            return order_id
        except Exception as e:
            logger.error(
                "BUY BACK ORDER FAILED: %d x %s @ $%.2f -- %s",
                contracts, contract_symbol, limit_price, _http_detail(e),
            )
            return None

    def check_account(self) -> Optional[Dict[str, Any]]:
        """Fetch Alpaca account info."""
        if self.dry_run:
            return {"equity": "100000", "status": "ACTIVE (dry-run)"}
        try:
            session = self._get_session()
            resp = session.get(f"{ALPACA_BASE_URL}/v2/account", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Account check failed: %s", _http_detail(e))
            return None


# =============================================================================
# MAIN SCALPER CLASS
# =============================================================================

class CCScalper:
    """Deep OTM covered call premium scalper for metals mining ETFs (multi-ticker).

    Orchestrates the full sell-into-spike / buy-back-on-crush cycle across all
    active tickers in TRADE_TICKERS (and optionally LEVERAGED_TICKERS).

    Usage:
        scalper = CCScalper(dry_run=True)
        scalper.initialize()
        scalper.run()           # continuous loop
        scalper.run_once()      # single cycle
        scalper.show_status()   # display state
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.data = DataLayer()
        self.signals = SignalEngine(self.data)
        self.strikes = StrikeSelector(self.data)
        self.executor = ExecutionLayer(dry_run=dry_run)
        self.order_manager = OrderManager(self.executor, dry_run=dry_run)
        # Edge 104(c) — load state from per-mode file
        self.state = ScalperState.load(dry_run=dry_run)

        # Regime detector -- correlation-based regime prediction layer
        import slvr_cc_config as _cfg
        self.regime_detector = RegimeDetector(_cfg)
        self._current_regime: Optional[Regime] = None
        self._regime_adjustments: Dict[str, Any] = {}

        # PMCC / diagonal spread manager
        self.pmcc = PMCCManager(
            api=self.executor,
            data_layer=self.data,
            order_manager=self.order_manager,
            signal_engine=self.signals,
        )
        # Restore PMCC state from the shared state file
        if self.state.pmcc_state:
            self.pmcc.from_dict(self.state.pmcc_state)

        # Cross-engine order deduplication layer (set by combined_runner or standalone)
        self._order_dedup = None

    def initialize(self) -> bool:
        """Initialize all subsystems. Returns True on success."""
        active = get_active_tickers()
        logger.info("=" * 70)
        logger.info(
            "CC SCALPER INITIALIZING  mode=%s  tickers=%d (%s)",
            "DRY RUN" if self.dry_run else "LIVE",
            len(active),
            ", ".join(active[:5]) + ("..." if len(active) > 5 else ""),
        )
        logger.info("=" * 70)

        # Load credentials
        creds_ok = self.executor.initialize()
        if not creds_ok and not self.dry_run:
            logger.warning("No credentials -- switching to dry-run mode")
            self.dry_run = True
            self.executor.dry_run = True

        # Check account
        acct = self.executor.check_account()
        if acct:
            logger.info(
                "Account: equity=$%s, status=%s",
                acct.get("equity", "?"), acct.get("status", "?"),
            )

        # Fetch market data
        data_ok = self.data.refresh()
        if not data_ok:
            logger.warning("Data refresh had errors -- some indicators may be unavailable")

        # Log current prices for all watch tickers
        for ticker in WATCH_TICKERS:
            price = self.data.get_price(ticker)
            change = self.data.get_daily_change_pct(ticker)
            if price is not None:
                change_str = f" ({change:+.2%})" if change is not None else ""
                logger.info("  %s: $%.2f%s", ticker, price, change_str)

        # Fetch option chains for all active trade tickers
        self.data.fetch_all_option_chains()

        # Log open positions
        open_pos = self.state.open_positions()
        logger.info("Open positions: %d", len(open_pos))
        for pos in open_pos:
            logger.info(
                "  [%s] $%.0f %s | %d contracts @ $%.2f | DTE=%d | status=%s",
                pos.ticker, pos.strike, pos.expiration,
                pos.contracts, pos.sell_price, pos.days_to_expiration, pos.status,
            )

        # PMCC initialization
        if PMCC_ENABLED:
            logger.info("PMCC module ENABLED -- initializing diagonal spread manager")
            # Auto-detect LEAPS from Alpaca positions (live mode only)
            if not self.dry_run:
                new_leaps = self.pmcc.auto_detect_leaps()
                if new_leaps:
                    logger.info("PMCC auto-detected %d new LEAP(s)", len(new_leaps))
            # Log existing PMCC spreads
            active_spreads = self.pmcc.get_active_spreads()
            logger.info("PMCC active spreads: %d", len(active_spreads))
            for spread in active_spreads:
                short_info = (
                    f"short=$%.0f exp=%s delta=%.3f" % (
                        spread.short_leg.strike, spread.short_leg.expiry,
                        spread.short_leg.delta,
                    ) if spread.has_short_leg else "no short leg"
                )
                logger.info(
                    "  [%s] LEAP $%.0f exp=%s delta=%.2f | %s | "
                    "credits=$%.2f cycles=%d | id=%s",
                    spread.ticker, spread.long_leg.strike,
                    spread.long_leg.expiry, spread.long_leg.delta,
                    short_info, spread.total_credits_received,
                    spread.num_short_cycles, spread.spread_id,
                )
        else:
            logger.info("PMCC module DISABLED")

        # Regime detector initialization
        if REGIME_ENABLED:
            logger.info("Regime detector ENABLED -- running initial prediction")
            try:
                self._current_regime = self.regime_detector.predict_regime()
                self._regime_adjustments = self.regime_detector.get_regime_adjustments()
                logger.info(
                    "Initial regime: %s  (adjustments: otm=%s, threshold=%+.2f, positions=%.1fx)",
                    self._current_regime.value.upper(),
                    f"{self._regime_adjustments.get('min_otm_pct', 'default')}",
                    self._regime_adjustments.get("signal_threshold_boost", 0.0),
                    self._regime_adjustments.get("max_positions_mult", 1.0),
                )
            except Exception as e:
                logger.error("Regime detector initialization failed: %s", e)
                self._current_regime = Regime.UNCERTAIN
                self._regime_adjustments = {}
        else:
            logger.info("Regime detector DISABLED")

        logger.info("Initialization complete")
        return data_ok

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def run_once(self) -> Dict[str, Any]:
        """Run a single evaluation cycle.

        1. Refresh data.
        2. Check existing positions for buy-back signals.
        3. If we have capacity, check for sell signals across all active tickers.
        4. Persist state.

        Returns a summary dict of actions taken.
        """
        cycle_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actions": [],
            "sell_signals": [],
            "buy_back_signals": [],
            "regime": None,
        }

        # Refresh data
        self.data.refresh_intraday()

        # --- Regime prediction (cached, not re-fetched every cycle) ---
        if REGIME_ENABLED:
            try:
                self._current_regime = self.regime_detector.predict_regime()
                self._regime_adjustments = self.regime_detector.get_regime_adjustments()
                cycle_summary["regime"] = self._current_regime.value
                logger.info(
                    "REGIME: %s  (adjustments: otm=%s, threshold=%+.2f, positions=%.1fx)",
                    self._current_regime.value.upper(),
                    self._regime_adjustments.get("min_otm_pct", "default"),
                    self._regime_adjustments.get("signal_threshold_boost", 0.0),
                    self._regime_adjustments.get("max_positions_mult", 1.0),
                )
            except Exception as e:
                logger.error("Regime prediction failed: %s -- using UNCERTAIN", e)
                self._current_regime = Regime.UNCERTAIN
                self._regime_adjustments = {}

        # --- Phase 0: Manage working orders (cancel/replace loop) ---
        pending = self.order_manager.get_pending_orders()
        if pending:
            logger.info(
                "Managing %d pending order(s) before signal evaluation",
                len(pending),
            )
            # Update market data for pending orders from latest option chains
            for mo in pending:
                self._update_order_market_data(mo)
            completed = self.order_manager.manage_orders()
            for mo in completed:
                cycle_summary["actions"].append({
                    "type": f"order_{mo.status}",
                    "side": mo.side,
                    "contract": mo.contract_symbol,
                    "contracts": mo.contracts,
                    "price": mo.fill_price or mo.current_price,
                    "attempts": mo.attempts,
                    "elapsed": f"{mo.elapsed:.0f}s",
                    "status": mo.status,
                })
            logger.info(
                "Order management: %d completed this cycle | %s",
                len(completed), self.order_manager.status_summary(),
            )

        # --- Phase 0.5: PMCC management ---
        # Check existing diagonal spreads, evaluate new short legs, manage
        # assignment risk.  This runs alongside regular CC selling -- they
        # are complementary strategies.
        if PMCC_ENABLED:
            try:
                pmcc_actions = self.pmcc.run_cycle()
                for pa in pmcc_actions:
                    cycle_summary["actions"].append(pa)
                    logger.info("PMCC ACTION: %s", json.dumps(pa, default=str))
                # Persist PMCC state after the cycle
                self.state.pmcc_state = self.pmcc.to_dict()
            except Exception as e:
                logger.error(
                    "PMCC cycle error: %s\n%s", e, traceback.format_exc(),
                )

        # --- Phase 1: Check existing positions for buy-back ---
        open_pos = self.state.open_positions()
        for pos in open_pos:
            try:
                current_price = self._get_current_option_price(pos)
                if current_price is None:
                    logger.debug(
                        "Cannot get current price for %s -- skipping buy-back check",
                        pos.option_symbol,
                    )
                    continue

                result = self.signals.evaluate_buy_back(pos, current_price)
                logger.info(
                    "BUY-BACK EVAL [%s/%s $%.0f %s]: %s",
                    pos.ticker, pos.option_symbol, pos.strike, pos.expiration,
                    result.summary().replace("\n", " | "),
                )

                cycle_summary["buy_back_signals"].append({
                    "ticker": pos.ticker,
                    "position": pos.option_symbol,
                    "triggered": result.triggered,
                    "score": result.score,
                    "details": result.details,
                })

                if result.triggered:
                    action = self._execute_buy_back(pos, current_price, result)
                    if action:
                        cycle_summary["actions"].append(action)

            except Exception as e:
                logger.error(
                    "Error evaluating buy-back for %s: %s\n%s",
                    pos.option_symbol, e, traceback.format_exc(),
                )

        # --- Phase 2: Check for new sell signals across all active tickers ---
        # Apply regime-adjusted max positions
        effective_max_positions = MAX_CONCURRENT_POSITIONS
        if REGIME_ENABLED and self._regime_adjustments:
            pos_mult = self._regime_adjustments.get("max_positions_mult", 1.0)
            effective_max_positions = max(1, int(MAX_CONCURRENT_POSITIONS * pos_mult))
            if pos_mult != 1.0:
                logger.info(
                    "Regime position limit: %d -> %d (%.1fx)",
                    MAX_CONCURRENT_POSITIONS, effective_max_positions, pos_mult,
                )

        open_count = len(self.state.open_positions())
        # Count pending sell orders toward position limits to prevent over-selling
        pending_sell_count = sum(
            1 for mo in self.order_manager.get_pending_orders() if mo.side == "sell"
        )
        effective_open = open_count + pending_sell_count
        if pending_sell_count > 0:
            logger.info(
                "Position count: %d open + %d pending sells = %d effective (max %d)",
                open_count, pending_sell_count, effective_open, effective_max_positions,
            )
        if effective_open < effective_max_positions:
            active = get_active_tickers()
            for ticker in active:
                # Re-check capacity each iteration (a sell in prev ticker consumes a slot)
                current_open = len(self.state.open_positions())
                current_pending = sum(
                    1 for mo in self.order_manager.get_pending_orders() if mo.side == "sell"
                )
                if current_open + current_pending >= effective_max_positions:
                    logger.info(
                        "At max concurrent positions (%d open + %d pending = %d/%d) "
                        "-- stopping sell evaluation",
                        current_open, current_pending,
                        current_open + current_pending, effective_max_positions,
                    )
                    break

                try:
                    sell_result = self.signals.evaluate_sell(
                        ticker,
                        regime_adjustments=self._regime_adjustments if REGIME_ENABLED else None,
                    )
                    logger.info(
                        "SELL SIGNAL EVAL [%s]: %s",
                        ticker, sell_result.summary().replace("\n", " | "),
                    )

                    cycle_summary["sell_signals"].append({
                        "ticker": ticker,
                        "triggered": sell_result.triggered,
                        "score": sell_result.score,
                        "max_score": sell_result.max_score,
                        "details": sell_result.details,
                    })

                    # Record signal in history
                    self.state.signal_history.append({
                        "timestamp": sell_result.timestamp,
                        "type": "sell",
                        "ticker": ticker,
                        "triggered": sell_result.triggered,
                        "score": sell_result.score,
                        "max_score": sell_result.max_score,
                    })

                    if sell_result.triggered:
                        action = self._execute_sell(ticker, sell_result)
                        if action:
                            cycle_summary["actions"].append(action)

                except Exception as e:
                    logger.error(
                        "Error evaluating sell signal for %s: %s\n%s",
                        ticker, e, traceback.format_exc(),
                    )
        else:
            logger.info(
                "At max concurrent positions (%d/%d) -- skipping sell evaluation",
                open_count, effective_max_positions,
            )

        # Persist state (including PMCC diagonal spread state)
        if PMCC_ENABLED:
            self.state.pmcc_state = self.pmcc.to_dict()
        self.state.last_run = datetime.now(timezone.utc).isoformat()
        self.state.save()

        return cycle_summary

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    def _execute_sell(self, ticker: str, signal: SignalResult) -> Optional[Dict[str, Any]]:
        """Select a strike and place a sell order for a specific ticker.

        Returns a summary dict of the action taken, or None if no trade.
        """
        # --- DEDUP CHECK: prevent duplicate sells across all engines ---
        # Check 1: Does our own OrderManager have a pending sell for this ticker?
        if self.order_manager.has_pending_sell_for_ticker(ticker):
            logger.info(
                "DEDUP SKIP: already have a pending sell order for %s -- "
                "waiting for it to fill/abort before submitting another",
                ticker,
            )
            return None

        # Check 2: Cross-engine dedup (CC Scalper + PMCC + Options Overlay)
        if self._order_dedup and self._order_dedup.has_pending_or_active_sell(ticker):
            logger.info(
                "DEDUP SKIP (cross-engine): another engine already has a pending "
                "or active short call on %s -- skipping sell",
                ticker,
            )
            return None

        # Check 3: No simultaneous buy+sell on same ticker (broker compliance)
        if self._order_dedup and self._order_dedup.has_conflicting_direction(ticker, "sell"):
            logger.info(
                "CONFLICT SKIP: pending buy order on %s -- cannot sell simultaneously",
                ticker,
            )
            return None

        # Refresh option chains for latest pricing
        self.data.fetch_option_chain(ticker)

        candidates = self.strikes.select_strikes(
            ticker,
            regime_adjustments=self._regime_adjustments if REGIME_ENABLED else None,
        )
        if not candidates:
            logger.warning("No eligible strikes found for %s -- cannot sell", ticker)
            return None

        # Take the top-ranked candidate
        best = candidates[0]
        strike = best["strike"]
        expiration = best["expiration"]
        dte = best["dte"]
        contract_symbol = best["contract_symbol"]
        mid = best["mid_price"]
        bid = best["bid"]
        ask = best["ask"]
        oi = best["oi"]
        chain_volume = best["volume"]

        # Determine number of contracts
        contracts = DEFAULT_CONTRACTS

        # Check daily limits
        chain_key = f"{contract_symbol}_{expiration}"
        counter = self.state.get_daily_counter()
        can_trade, reason = counter.can_trade(chain_key, contracts, chain_volume)
        if not can_trade:
            logger.warning("TRADE BLOCKED for %s: %s", ticker, reason)
            return None

        # Calculate limit price: sell ABOVE mid (into urgency)
        if bid > 0 and ask > 0:
            mid_calc = (bid + ask) / 2.0
            limit_price = round(mid_calc + SELL_OFFSET_FROM_MID, 2)
        elif mid > 0:
            limit_price = round(mid + SELL_OFFSET_FROM_MID, 2)
        else:
            limit_price = round(best["effective_price"] + SELL_OFFSET_FROM_MID, 2)

        # Minimum price floor
        if limit_price < MIN_PREMIUM:
            logger.warning(
                "Calculated limit price $%.2f below minimum premium $%.2f for %s -- skipping",
                limit_price, MIN_PREMIUM, ticker,
            )
            return None

        # Estimate IV at time of sale
        underlying_price = self.data.get_price(ticker)
        sell_iv = 0.0
        if underlying_price and underlying_price > 0 and dte > 0:
            T = dte / 365.0
            sell_iv = implied_volatility(limit_price, underlying_price, strike, T, RISK_FREE_RATE)

        logger.info(
            "SELL DECISION: %d x %s $%.0f %s @ $%.2f (mid=$%.2f, bid=$%.2f, ask=$%.2f) "
            "DTE=%d OTM=%.0f%% IV=%.1f%%",
            contracts, ticker, strike, expiration, limit_price,
            mid, bid, ask, dte, best["otm_pct"] * 100, sell_iv * 100,
        )

        # Build position record (will be finalized on fill)
        pos = CCPosition(
            ticker=ticker,
            option_symbol=contract_symbol,
            strike=strike,
            expiration=expiration,
            dte_at_entry=dte,
            contracts=contracts,
            sell_price=limit_price,
            sell_date=datetime.now(timezone.utc).isoformat(),
            sell_underlying_price=underlying_price or 0.0,
            sell_iv=sell_iv,
            sell_signal_score=signal.score,
            sell_signal_details=signal.summary(),
            status="open",
        )

        # Callbacks for the OrderManager
        def _on_sell_fill(mo: ManagedOrder) -> None:
            """Called when the sell order fills."""
            actual_price = mo.fill_price if mo.fill_price else mo.current_price
            pos.sell_price = actual_price
            pos.alpaca_order_id = mo.order_id
            self.state.positions.append(pos)
            self.state.total_premium_collected += pos.premium_collected
            self.state.total_trades += 1
            counter.record_trade(chain_key, mo.filled_qty or contracts, chain_volume)
            self.state.save()
            logger.info(
                "SELL FILL RECORDED: %s %d x $%.0f %s @ $%.2f avg = $%.0f premium | "
                "IV=%.1f%% | signal=%d/%d",
                ticker, pos.contracts, strike, expiration, actual_price,
                pos.premium_collected, sell_iv * 100,
                signal.score, signal.max_score,
            )

        def _on_sell_abort(mo: ManagedOrder) -> None:
            """Called when the sell order is aborted."""
            logger.warning(
                "SELL ABORTED: %s %d x $%.0f %s -- order could not be filled "
                "after %d attempts",
                ticker, contracts, strike, expiration, mo.attempts,
            )

        # Submit via OrderManager (active management begins)
        mid_calc = (bid + ask) / 2.0 if bid > 0 and ask > 0 else mid
        managed = self.order_manager.submit_sell(
            contract_symbol=contract_symbol,
            contracts=contracts,
            limit_price=limit_price,
            mid_price=mid_calc,
            bid=bid,
            ask=ask,
            on_fill=_on_sell_fill,
            on_abort=_on_sell_abort,
        )
        if managed is None:
            return None

        action = {
            "type": "sell_submitted",
            "ticker": ticker,
            "contract": contract_symbol,
            "strike": strike,
            "expiration": expiration,
            "contracts": contracts,
            "limit_price": limit_price,
            "mid_price": mid_calc,
            "iv": sell_iv,
            "signal_score": signal.score,
            "order_id": managed.order_id,
            "managed": True,
        }

        logger.info(
            "SELL SUBMITTED (managed): %s %d x $%.0f %s @ $%.2f (mid=$%.2f) | "
            "IV=%.1f%% | signal=%d/%d | order=%s",
            ticker, contracts, strike, expiration, limit_price, mid_calc,
            sell_iv * 100, signal.score, signal.max_score, managed.order_id,
        )

        return action

    def _execute_buy_back(
        self,
        position: CCPosition,
        current_price: float,
        signal: SignalResult,
    ) -> Optional[Dict[str, Any]]:
        """Place a buy-back order for an existing position.

        Returns a summary dict, or None if no trade.
        """
        contract_symbol = position.option_symbol
        contracts = position.contracts
        ticker = position.ticker

        # === Edge 104(a) guard — verify Alpaca holds the short before BUY ===
        try:
            broker_pos = self.executor.get_position(contract_symbol)
            broker_qty = float(broker_pos.get("qty", 0)) if broker_pos else 0.0
        except Exception as e:
            logger.warning(
                "Edge 104 guard: position lookup failed for %s: %s -- "
                "skipping cycle, will retry next pass",
                contract_symbol, e,
            )
            return None
        if broker_qty >= 0:
            logger.error(
                "BUY-BACK ABORTED (Edge 104 guard): %s has no real SHORT on "
                "Alpaca (broker_qty=%.0f, state_qty=-%d). Marking position as "
                "ORPHANED to stop the loop. Manual review required.",
                contract_symbol, broker_qty, contracts,
            )
            position.status = "orphaned"
            position.buy_back_reason = "edge104_phantom_guard"
            position.buy_back_date = datetime.now(timezone.utc).isoformat()
            self.state.save()
            try:
                from jarvis_utils.inbox import send
                send(
                    f"Edge 104 guard caught phantom short {contract_symbol} "
                    f"(state thinks short {contracts}, broker has "
                    f"{broker_qty:.0f}). Marked ORPHANED in slvr_cc_state.json. "
                    f"Investigate origin.",
                    source="cc-scalper",
                )
            except Exception:
                pass
            return None
        # === End Edge 104(a) guard ===

        # Determine trigger reason (first triggered sub-signal)
        trigger_reason = "unknown"
        for key, detail in signal.details.items():
            if isinstance(detail, dict) and detail.get("triggered", False):
                trigger_reason = key
                break

        # Check for conflicting sell order on same ticker (broker compliance)
        if self._order_dedup and self._order_dedup.has_conflicting_direction(ticker, "buy"):
            logger.info(
                "CONFLICT SKIP: pending sell order on %s -- cannot buy back simultaneously",
                ticker,
            )
            return None

        # Check daily limits
        chain_key = f"{contract_symbol}_{position.expiration}"
        counter = self.state.get_daily_counter()
        can_trade, reason = counter.can_trade(chain_key, contracts)
        if not can_trade:
            logger.warning("BUY-BACK BLOCKED for %s: %s", ticker, reason)
            return None

        # Calculate limit price: buy BELOW mid (into panic)
        limit_price = round(max(current_price - BUYBACK_OFFSET_FROM_MID, 0.01), 2)
        mid_price = current_price  # mid is our best estimate of current mid

        logger.info(
            "BUY-BACK DECISION: %s %d x %s @ $%.2f (current=$%.2f) | reason=%s | "
            "sold @ $%.2f | P&L=$%.0f (%.1f%%)",
            ticker, contracts, contract_symbol, limit_price, current_price, trigger_reason,
            position.sell_price, position.unrealized_pnl(current_price),
            position.profit_pct(current_price) * 100,
        )

        # Try to get bid/ask from chain data
        bb_bid, bb_ask = 0.0, 0.0
        chains = self.data.get_option_chains(ticker)
        if position.expiration in chains:
            df = chains[position.expiration]
            if df is not None and len(df) > 0:
                match = df[df["strike"] == position.strike]
                if len(match) > 0:
                    row = match.iloc[0]
                    bb_bid = float(row.get("bid", 0))
                    bb_ask = float(row.get("ask", 0))
                    if bb_bid > 0 and bb_ask > 0:
                        mid_price = (bb_bid + bb_ask) / 2.0

        # Callbacks for the OrderManager
        def _on_buyback_fill(mo: ManagedOrder) -> None:
            """Called when the buy-back order fills."""
            actual_price = mo.fill_price if mo.fill_price else mo.current_price
            pnl = (position.sell_price - actual_price) * contracts * 100
            position.status = "closed"
            position.buy_back_price = actual_price
            position.buy_back_date = datetime.now(timezone.utc).isoformat()
            position.buy_back_reason = trigger_reason
            position.realized_pnl = pnl
            self.state.total_realized_pnl += pnl
            counter.record_trade(chain_key, mo.filled_qty or contracts)
            self.state.save()
            logger.info(
                "BUY-BACK FILL RECORDED: %s %d x $%.0f %s @ $%.2f avg | "
                "sold @ $%.2f | P&L=$%.0f (%.1f%%) | reason=%s",
                ticker, contracts, position.strike, position.expiration,
                actual_price, position.sell_price, pnl,
                position.profit_pct(actual_price) * 100, trigger_reason,
            )

        def _on_buyback_abort(mo: ManagedOrder) -> None:
            """Called when the buy-back order is aborted."""
            logger.warning(
                "BUY-BACK ABORTED: %s %d x $%.0f %s -- order could not be "
                "filled after %d attempts, position remains open",
                ticker, contracts, position.strike, position.expiration,
                mo.attempts,
            )

        # Submit via OrderManager (active management begins)
        managed = self.order_manager.submit_buy_back(
            contract_symbol=contract_symbol,
            contracts=contracts,
            limit_price=limit_price,
            mid_price=mid_price,
            bid=bb_bid,
            ask=bb_ask,
            on_fill=_on_buyback_fill,
            on_abort=_on_buyback_abort,
        )
        if managed is None:
            return None

        action = {
            "type": "buy_back_submitted",
            "ticker": ticker,
            "contract": contract_symbol,
            "strike": position.strike,
            "expiration": position.expiration,
            "contracts": contracts,
            "limit_price": limit_price,
            "mid_price": mid_price,
            "sold_at": position.sell_price,
            "unrealized_pnl": position.unrealized_pnl(current_price),
            "reason": trigger_reason,
            "order_id": managed.order_id,
            "managed": True,
        }

        logger.info(
            "BUY-BACK SUBMITTED (managed): %s %d x $%.0f %s @ $%.2f (mid=$%.2f) | "
            "sold @ $%.2f | reason=%s | order=%s",
            ticker, contracts, position.strike, position.expiration,
            limit_price, mid_price, position.sell_price,
            trigger_reason, managed.order_id,
        )

        return action

    def _get_current_option_price(self, position: CCPosition) -> Optional[float]:
        """Get the current market price for an option position.

        Tries the cached option chain first, then falls back to BS model estimate.
        """
        ticker = position.ticker
        chains = self.data.get_option_chains(ticker)

        # Try to find in cached chain data
        if position.expiration in chains:
            df = chains[position.expiration]
            if df is not None and len(df) > 0:
                match = df[df["strike"] == position.strike]
                if len(match) > 0:
                    row = match.iloc[0]
                    bid = float(row.get("bid", 0))
                    ask = float(row.get("ask", 0))
                    if bid > 0 and ask > 0:
                        return (bid + ask) / 2.0
                    last = float(row.get("lastPrice", 0))
                    if last > 0:
                        return last

        # Fallback: BS model estimate using current HV
        underlying_price = self.data.get_price(ticker)
        if underlying_price is None or underlying_price <= 0:
            return None
        dte = position.days_to_expiration
        if dte <= 0:
            return max(underlying_price - position.strike, 0.0)
        T = dte / 365.0

        # Use HV-20 as vol estimate, but adjust for IV premium
        hv20 = self.data.compute_hv(ticker, HV_LONG_WINDOW)
        sigma = hv20 if hv20 and hv20 > 0 else 0.75  # default 75%

        # IV is typically higher than HV -- use a conservative 10% premium
        sigma *= 1.10

        price = bs_call_price(underlying_price, position.strike, T, RISK_FREE_RATE, sigma)
        return price if price > 0 else None

    def _update_order_market_data(self, mo: "ManagedOrder") -> None:
        """Push current bid/ask from cached chain data into a ManagedOrder.

        Called before manage_orders() so the price-adjustment logic uses
        the freshest available quotes.
        """
        # We need to find the ticker for this contract.  Option symbols
        # start with the ticker (variable length) followed by digits.
        ticker = ""
        for i, ch in enumerate(mo.contract_symbol):
            if ch.isdigit():
                ticker = mo.contract_symbol[:i]
                break
        if not ticker:
            return

        chains = self.data.get_option_chains(ticker)
        for _exp_str, df in chains.items():
            if df is None or len(df) == 0:
                continue
            match = df[df.get("contractSymbol", pd.Series()) == mo.contract_symbol]
            if len(match) > 0:
                row = match.iloc[0]
                bid = float(row.get("bid", 0))
                ask = float(row.get("ask", 0))
                if bid > 0 and ask > 0:
                    self.order_manager.update_market_data(
                        mo.contract_symbol, bid, ask,
                    )
                return

    # ------------------------------------------------------------------
    # Continuous run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main continuous execution loop.

        Runs cycles during market hours, sleeps between cycles.
        """
        logger.info("Starting continuous run loop (poll every %ds)", POLL_INTERVAL_SECONDS)

        try:
            while True:
                try:
                    # Always use ET for all schedule/weekday decisions.
                    # datetime.now() (naive) was previously used for weekday --
                    # that produced wrong results when the server is in UTC and
                    # it is between 20:00-23:59 UTC (still "today" in UTC but
                    # already "tomorrow" in ET after midnight ET). Fixed: derive
                    # everything from the single ET-aware timestamp.
                    import pytz
                    et = pytz.timezone("US/Eastern")
                    now_et = datetime.now(et)
                    weekday = now_et.weekday()

                    # Skip weekends
                    if weekday >= 5:
                        logger.info("Weekend -- sleeping 1 hour")
                        time.sleep(3600)
                        continue

                    # Check market hours using ET
                    h, m = now_et.hour, now_et.minute
                    t_min = h * 60 + m
                    open_min = MARKET_OPEN[0] * 60 + MARKET_OPEN[1]
                    close_min = MARKET_CLOSE[0] * 60 + MARKET_CLOSE[1]

                    if t_min < open_min - 15 or t_min > close_min + 15:
                        logger.info(
                            "Outside market hours (%s ET) -- sleeping 5 minutes",
                            now_et.strftime("%H:%M"),
                        )
                        time.sleep(300)
                        continue

                    # Run a cycle
                    logger.info(
                        "--- CYCLE START (%s ET) ---", now_et.strftime("%H:%M:%S"),
                    )
                    summary = self.run_once()

                    actions = summary.get("actions", [])
                    if actions:
                        for act in actions:
                            logger.info("ACTION: %s", json.dumps(act, default=str))
                    else:
                        logger.info("No actions this cycle")

                    logger.info("--- CYCLE END ---")

                except Exception as e:
                    logger.error(
                        "Cycle error (will retry): %s\n%s", e, traceback.format_exc(),
                    )

                time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt -- shutting down")
            self.state.save()

    # ------------------------------------------------------------------
    # Status display
    # ------------------------------------------------------------------

    def show_status(self) -> str:
        """Generate and return a human-readable status report."""
        active = get_active_tickers()
        lines = [
            "",
            "=" * 70,
            f"CC SCALPER STATUS ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}",
            f"Active tickers: {len(active)}",
            "=" * 70,
        ]

        # Current prices for all active tickers
        lines.append("\nTRADE TICKER PRICES:")
        for ticker in active:
            price = self.data.get_price(ticker)
            change = self.data.get_daily_change_pct(ticker)
            if price is not None:
                change_str = f" ({change:+.2%})" if change is not None else ""
                lines.append(f"  {ticker:6s}: ${price:>8.2f}{change_str}")
            else:
                lines.append(f"  {ticker:6s}: N/A")

        # Signal ticker prices
        lines.append("\nSIGNAL TICKER PRICES:")
        for ticker in [GLD_TICKER, UUP_TICKER, SLV_TICKER, USO_TICKER]:
            price = self.data.get_price(ticker)
            change = self.data.get_daily_change_pct(ticker)
            if price is not None:
                change_str = f" ({change:+.2%})" if change is not None else ""
                lines.append(f"  {ticker:6s}: ${price:>8.2f}{change_str}")

        # Per-ticker indicators (RSI & HV) -- one line each
        lines.append(f"\nPER-TICKER INDICATORS (RSI-{RSI_PERIOD}, HV-{HV_SHORT_WINDOW}, HV-{HV_LONG_WINDOW}):")
        for ticker in active:
            rsi = self.data.compute_rsi(ticker)
            hv10 = self.data.compute_hv(ticker, HV_SHORT_WINDOW)
            hv20 = self.data.compute_hv(ticker, HV_LONG_WINDOW)
            rsi_str = f"RSI={rsi:.1f}" if rsi else "RSI=N/A"
            hv_str = (
                f"HV10={hv10:.1%} HV20={hv20:.1%}"
                if hv10 and hv20 else "HV=N/A"
            )
            lines.append(f"  {ticker:6s}: {rsi_str:12s} {hv_str}")

        # GLD direction (metals signal)
        gld_15m = self.data.gld_short_term_change(minutes=15)
        gld_60m = self.data.gld_60min_change()
        lines.append(f"\nGLD DIRECTION (metals signal):")
        lines.append(
            f"  15-min: {gld_15m:+.3%}" if gld_15m is not None else "  15-min: N/A"
        )
        lines.append(
            f"  60-min: {gld_60m:+.3%}" if gld_60m is not None else "  60-min: N/A"
        )

        # USO direction (oil/energy signal, Mar 18)
        uso_15m = self.data.macro_short_term_change(USO_TICKER, minutes=15)
        uso_60m = self.data.macro_short_term_change(USO_TICKER, minutes=60)
        lines.append(f"\nUSO DIRECTION (energy signal):")
        lines.append(
            f"  15-min: {uso_15m:+.3%}" if uso_15m is not None else "  15-min: N/A"
        )
        lines.append(
            f"  60-min: {uso_60m:+.3%}" if uso_60m is not None else "  60-min: N/A"
        )

        # Sell signal scores per ticker
        lines.append(f"\nSELL SIGNAL SCORES (need {MIN_SELL_SIGNALS}/6):")
        for ticker in active:
            result = self.signals.evaluate_sell(ticker)
            status = "TRIGGERED" if result.triggered else "not triggered"
            lines.append(f"  {ticker:6s}: {result.score}/{result.max_score} ({status})")

        # Open positions grouped by ticker
        open_pos = self.state.open_positions()
        lines.append(f"\nOPEN POSITIONS ({len(open_pos)}):")
        if open_pos:
            # Group by ticker
            by_ticker: Dict[str, List[CCPosition]] = {}
            for pos in open_pos:
                by_ticker.setdefault(pos.ticker, []).append(pos)
            for tk in sorted(by_ticker.keys()):
                for pos in by_ticker[tk]:
                    current = self._get_current_option_price(pos)
                    pnl_str = ""
                    if current is not None:
                        pnl = pos.unrealized_pnl(current)
                        pct = pos.profit_pct(current)
                        pnl_str = f" | P&L=${pnl:+.0f} ({pct:.0%})"
                    lines.append(
                        f"  [{pos.ticker}] ${pos.strike:.0f} {pos.expiration} | "
                        f"{pos.contracts} contracts @ ${pos.sell_price:.2f} | "
                        f"DTE={pos.days_to_expiration}{pnl_str}"
                    )
        else:
            lines.append("  (none)")

        # Daily limits
        counter = self.state.get_daily_counter()
        lines.append(f"\nDAILY LIMITS ({counter.date}):")
        if counter.chain_counts:
            for chain, count in counter.chain_counts.items():
                vol = counter.chain_volumes.get(chain, 0)
                vol_pct = counter.volume_pct(chain)
                lines.append(
                    f"  {chain}: {count}/{MAX_CONTRACTS_PER_CHAIN_PER_DAY} contracts "
                    f"({vol_pct:.0%} of volume={vol})"
                )
        else:
            lines.append("  No trades today")

        # Regime detector status
        if REGIME_ENABLED:
            lines.append(f"\nREGIME DETECTOR:")
            if self._current_regime:
                lines.append(f"  Current regime: {self._current_regime.value.upper()}")
                adj = self._regime_adjustments
                if adj:
                    otm = adj.get("min_otm_pct")
                    lines.append(
                        f"  Adjustments: OTM={f'{otm:.0%}' if otm else 'default'}, "
                        f"threshold={adj.get('signal_threshold_boost', 0.0):+.2f}, "
                        f"positions={adj.get('max_positions_mult', 1.0):.1f}x"
                    )
            else:
                lines.append("  No regime prediction yet")
        else:
            lines.append(f"\nREGIME DETECTOR: DISABLED")

        # PMCC diagonal spread status
        if PMCC_ENABLED:
            lines.append(self.pmcc.status_report())

        # Order Manager status
        lines.append(f"\nORDER MANAGER:")
        lines.append(f"  {self.order_manager.status_summary()}")

        # Cumulative stats
        lines.append(f"\nCUMULATIVE:")
        lines.append(f"  Total trades: {self.state.total_trades}")
        lines.append(f"  Total premium collected: ${self.state.total_premium_collected:,.0f}")
        lines.append(f"  Total realized P&L: ${self.state.total_realized_pnl:,.0f}")

        # Recent closed positions
        closed = [p for p in self.state.positions if p.status == "closed"]
        if closed:
            recent = closed[-5:]
            lines.append(f"\nRECENT CLOSED ({len(recent)} of {len(closed)}):")
            for pos in recent:
                lines.append(
                    f"  [{pos.ticker}] ${pos.strike:.0f} {pos.expiration} | "
                    f"sold=${pos.sell_price:.2f} bought=${pos.buy_back_price:.2f} | "
                    f"P&L=${pos.realized_pnl:+.0f} | reason={pos.buy_back_reason}"
                )

        lines.append("=" * 70)

        report = "\n".join(lines)
        logger.info(report)
        return report

    def daily_summary(self) -> str:
        """Generate a daily summary for logging (multi-ticker aware)."""
        active = get_active_tickers()
        open_pos = self.state.open_positions()
        counter = self.state.get_daily_counter()
        total_today = sum(counter.chain_counts.values()) if counter.chain_counts else 0

        lines = [
            "",
            "=" * 50,
            f"DAILY SUMMARY ({_today_et().isoformat()})",
            "=" * 50,
            f"Active tickers: {len(active)}",
            f"Trades placed today: {total_today}",
            f"Open positions: {len(open_pos)}",
            f"Total realized P&L: ${self.state.total_realized_pnl:,.0f}",
            f"Total premium collected: ${self.state.total_premium_collected:,.0f}",
        ]

        for pos in open_pos:
            current = self._get_current_option_price(pos)
            pnl_str = ""
            if current is not None:
                pnl = pos.unrealized_pnl(current)
                pnl_str = f" P&L=${pnl:+.0f}"
            lines.append(
                f"  OPEN: [{pos.ticker}] ${pos.strike:.0f} {pos.expiration} "
                f"{pos.contracts}x @ ${pos.sell_price:.2f} DTE={pos.days_to_expiration}"
                f"{pnl_str}"
            )

        lines.append("=" * 50)
        summary = "\n".join(lines)
        logger.info(summary)
        return summary


# Backward-compatible alias for the old class name
SLVRCCScalper = CCScalper


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Deep OTM Covered Call Premium Scalper (multi-ticker)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Dry-run mode (default): evaluate signals without placing orders",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Live mode: place real paper-trade orders via Alpaca",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current state and open positions",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single evaluation cycle and exit",
    )
    parser.add_argument(
        "--pmcc-status", action="store_true",
        help="Show PMCC diagonal spread positions and exit",
    )
    parser.add_argument(
        "--pmcc-register", nargs=3, metavar=("TICKER", "SYMBOL", "COST"),
        help="Manually register a LEAP for PMCC: --pmcc-register SIL SIL270115C00060000 30.00",
    )
    parser.add_argument(
        "--regime", action="store_true",
        help="Show current regime prediction with all correlation signals and exit",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    dry_run = not args.live

    scalper = CCScalper(dry_run=dry_run)

    if args.pmcc_register:
        scalper.initialize()
        ticker, symbol, cost_str = args.pmcc_register
        cost = float(cost_str)
        spread = scalper.pmcc.register_leap(ticker, symbol, cost)
        scalper.state.pmcc_state = scalper.pmcc.to_dict()
        scalper.state.save()
        print(f"\nPMCC LEAP registered: {spread.spread_id}")
        print(f"  Ticker: {spread.ticker}")
        print(f"  LEAP: {spread.long_leg.symbol} ${spread.long_leg.strike:.0f} "
              f"exp={spread.long_leg.expiry}")
        print(f"  Delta: {spread.long_leg.delta:.2f}")
        print(f"  Cost basis: ${spread.long_leg.cost_basis:.2f}/sh")
        print(f"  DTE: {spread.long_leg_dte}")
        return

    if args.pmcc_status:
        scalper.initialize()
        report = scalper.pmcc.status_report()
        print(report)
        return

    if args.regime:
        # Regime-only mode: fetch data, predict regime, print report, exit
        # Skip full initialization (no Alpaca creds needed)
        import slvr_cc_config as _cfg
        detector = RegimeDetector(_cfg)
        detector.fetch_correlation_data()
        detector.print_regime_report()
        return

    if args.status:
        scalper.initialize()
        scalper.show_status()
        return

    if not scalper.initialize():
        logger.warning("Initialization had errors -- proceeding with available data")

    if args.once:
        summary = scalper.run_once()
        scalper.daily_summary()
        logger.info("Single cycle complete. Actions: %d", len(summary.get("actions", [])))
        return

    scalper.run()


if __name__ == "__main__":
    main()
