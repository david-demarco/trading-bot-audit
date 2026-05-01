#!/usr/bin/env python3
"""
csp_seller.py - Cash-Secured Put (CSP) seller for tail-index-eligible tickers.

Strategy:
  1. Target tickers where power-law α < CC_MIN_TAIL_INDEX (currently WPM, GDX).
     These are EXCLUDED from call-selling by the tail-index gate but ideal for
     put-selling because puts are MORE overpriced than calls on fat-tailed names.
  2. Sell OTM puts (~25-delta, 30-45 DTE) on pullbacks using a 7-signal inverse
     system. 5-of-7 signals must align.
  3. Cash-secured by design — no naked-short risk. Assignment is acceptable
     (aligns with the ownership thesis for WPM/GDX).
  4. Exit at 50% profit, 2x entry stop, or 7 DTE.

Eligible tickers: [t for t in CC_OPTIONS_ELIGIBLE if TAIL_INDEX.get(t,99) < CC_MIN_TAIL_INDEX]
Currently: WPM (α=2.92), GDX (α=2.96)

Spec: https://github.com/david-demarco/trading-bot-audit/blob/main/CSP_MODULE_SPEC_v0.md
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import math
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz

# ---------------------------------------------------------------------------
# ET date helper — all DTE comparisons use ET, not server UTC.
_ET_TZ = pytz.timezone("US/Eastern")


def _today_et() -> date:
    return datetime.now(_ET_TZ).date()


# ---------------------------------------------------------------------------
# Path setup
BASE_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, "/opt/jarvis-utils/lib")

# ---------------------------------------------------------------------------
# Configuration imports from existing modules

from slvr_cc_config import (
    GLD_TICKER,
    UUP_TICKER,
    SLV_TICKER,
    ALPACA_BASE_URL,
    ORDER_TIF,
    ALPACA_USER_ID,
    LOG_MAX_BYTES,
    LOG_BACKUP_COUNT,
    MARKET_OPEN,
    MARKET_CLOSE,
    RISK_FREE_RATE,
    YF_HISTORY_PERIOD,
    YF_INTRADAY_PERIOD,
    YF_INTRADAY_INTERVAL,
    ALPACA_FEED,
)

from combined_config import (
    TAIL_INDEX,
    CC_MIN_TAIL_INDEX,
    CC_OPTIONS_ELIGIBLE,
)

# Eligible tickers for CSP: must be in CC_OPTIONS_ELIGIBLE AND have α < CC_MIN_TAIL_INDEX.
# CC scalper blocks these; CSP is their designated alternative.
CSP_ELIGIBLE_TICKERS: List[str] = [
    t for t in CC_OPTIONS_ELIGIBLE
    if TAIL_INDEX.get(t, 99.0) < CC_MIN_TAIL_INDEX
]

# =============================================================================
# CSP-SPECIFIC PARAMETERS (all spec-resolved 2026-05-01)
# =============================================================================

# --- Strike selection ---
CSP_DTE_MIN = 30
CSP_DTE_MAX = 45
CSP_OTM_MIN_PCT = 0.08          # minimum 8% OTM below spot
CSP_OTM_MAX_PCT = 0.20          # maximum 20% OTM below spot
CSP_TARGET_DELTA = 0.25         # target put delta (~25-delta per spec)
CSP_MIN_OPEN_INTEREST = 100
CSP_MAX_BID_ASK_SPREAD_PCT = 0.30  # 30% spread cap (mining chains can be wide)

# --- Premium floor (spec: 2.0% of strike) ---
CSP_PREMIUM_FLOOR_PCT = 0.020   # bid must be >= 2.0% of strike

# --- IV-rank gates (spec: all three must pass) ---
CSP_IV_RANK_MIN_TICKER = 30     # WPM/GDX IV-rank must exceed 30
CSP_IV_RANK_MIN_CROSS = 30      # SLV OR GLD IV-rank must exceed 30

# --- Position caps (spec-resolved 2026-05-01) ---
CSP_MAX_PER_TICKER = 1          # 1 active CSP per ticker
CSP_MAX_TOTAL_EQUITY_PCT = 0.15 # <= 15% of account equity across all CSPs
CSP_MAX_CONCURRENT = 3          # <= 3 hard ceiling

# --- Stock+CSP combined cap (spec: USE STRIKE, not spot) ---
CSP_COMBINED_CAP_PCT = 0.08     # (stock_$ + strike*contracts*100) <= 8% equity

# --- Exit rules ---
CSP_PROFIT_TARGET = 0.50        # close at 50% of premium collected
CSP_STOP_LOSS_MULT = 2.0        # close when option price >= 2x entry
CSP_DTE_TRIGGER = 7             # close at 7 DTE

# --- Signal thresholds ---
CSP_MIN_SELL_SIGNALS = 5        # 5-of-7 required (spec)
CSP_RSI_PERIOD = 14
CSP_HV_SHORT = 10
CSP_HV_LONG = 20

# Signal 1: ticker down >=3%
CSP_DOWN_PCT = -0.03

# Signal 2: RSI oversold
CSP_RSI_OVERSOLD = 35

# Signal 4: GLD/SLV macro selling (either down >=0.5%)
CSP_MACRO_SELLING_PCT = -0.005

# Signal 5: UUP rallying (dollar bounce precedes metal reversal)
CSP_UUP_RALLY_PCT = 0.001       # up >= 0.10%

# Signal 6: macro NOT crashing (don't sell puts into freefall)
CSP_MACRO_NOTCRASH_15M = -0.001  # GLD 15-min change must be > -0.10%
CSP_MACRO_NOTCRASH_60M = -0.004  # GLD 60-min cumulative must be > -0.40%

# Signal 7: vol ratio not accelerating vs prior cycle
# HV10/HV20 ratio increase > 5% = "accelerating" = signal FAILS
CSP_VOL_ACCEL_THRESHOLD = 0.05

# --- Order execution ---
CSP_SELL_OFFSET_FROM_MID = 0.02    # place limit $0.02 above mid
CSP_BUYBACK_OFFSET_FROM_MID = 0.05 # buy back $0.05 above mid

# --- State / logging ---
CSP_STATE_FILE = "csp_state.json"
CSP_LOG_FILE = "logs/csp_seller.log"

# =============================================================================
# LOGGING
# =============================================================================

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("csp_seller")


def setup_logging(level: str = "INFO") -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(numeric)
    root.addHandler(ch)
    log_path = BASE_DIR / CSP_LOG_FILE
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.handlers.RotatingFileHandler(
        str(log_path), maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT,
    )
    fh.setFormatter(fmt)
    fh.setLevel(numeric)
    root.addHandler(fh)


# =============================================================================
# BLACK-SCHOLES PUT PRICING & GREEKS
# =============================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put option price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put delta magnitude (returns positive value [0, 1])."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return abs(_norm_cdf(d1) - 1.0)


def implied_volatility_put(
    market_price: float, S: float, K: float, T: float, r: float,
    max_iter: int = 100, tol: float = 1e-6,
) -> float:
    """Newton-Raphson IV solver for a put. Returns annualized IV as decimal."""
    if market_price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.50
    sigma = 0.50
    for _ in range(max_iter):
        price = bs_put_price(S, K, T, r, sigma)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        vega = S * _norm_pdf(d1) * math.sqrt(T)
        if abs(vega) < 1e-10:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        sigma = max(sigma, 0.01)
        sigma = min(sigma, 5.0)
    return sigma


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CSPPosition:
    """Tracks a sold cash-secured put position."""
    ticker: str
    option_symbol: str
    strike: float
    expiration: str                  # YYYY-MM-DD
    dte_at_entry: int
    contracts: int
    sell_price: float                # per-share option price collected
    sell_date: str                   # ISO timestamp
    sell_underlying_price: float
    sell_iv: float
    sell_signal_score: int
    sell_signal_details: str
    cash_reserved: float             # strike * contracts * 100 (assignment exposure)
    status: str = "open"             # open, closed, expired, assigned
    buy_back_price: Optional[float] = None
    buy_back_date: Optional[str] = None
    buy_back_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    alpaca_order_id: Optional[str] = None

    @property
    def days_to_expiration(self) -> int:
        try:
            exp = datetime.strptime(self.expiration, "%Y-%m-%d").date()
            return (exp - _today_et()).days
        except (ValueError, TypeError):
            return 0

    @property
    def premium_collected(self) -> float:
        return self.sell_price * self.contracts * 100

    def unrealized_pnl(self, current_option_price: float) -> float:
        return (self.sell_price - current_option_price) * self.contracts * 100

    def profit_pct(self, current_option_price: float) -> float:
        if self.sell_price <= 0:
            return 0.0
        return (self.sell_price - current_option_price) / self.sell_price

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CSPPosition":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CSPState:
    """Full persistent state for the CSP seller."""
    positions: List[CSPPosition] = field(default_factory=list)
    total_premium_collected: float = 0.0
    total_realized_pnl: float = 0.0
    total_trades: int = 0
    last_run: Optional[str] = None
    # Signal 7: HV10/HV20 ratio from the prior cycle per ticker.
    prev_hv_ratios: Dict[str, float] = field(default_factory=dict)
    _dry_run: bool = field(default=False, repr=False)

    @staticmethod
    def _resolve_path(base: Path, dry_run: bool) -> Path:
        if dry_run:
            return base.with_name(base.stem + ".dryrun" + base.suffix)
        return base

    def open_positions(self) -> List[CSPPosition]:
        return [p for p in self.positions if p.status == "open"]

    def save(self, path: Optional[Path] = None) -> None:
        if path is None:
            path = BASE_DIR / CSP_STATE_FILE
        path = self._resolve_path(path, self._dry_run)
        data = {
            "positions": [p.to_dict() for p in self.positions],
            "total_premium_collected": self.total_premium_collected,
            "total_realized_pnl": self.total_realized_pnl,
            "total_trades": self.total_trades,
            "last_run": self.last_run,
            "prev_hv_ratios": self.prev_hv_ratios,
        }
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, default=str))
            tmp.replace(path)
        except Exception as e:
            logger.error("Failed to save CSP state: %s", e)

    @classmethod
    def load(cls, path: Optional[Path] = None, dry_run: bool = False) -> "CSPState":
        if path is None:
            path = BASE_DIR / CSP_STATE_FILE
        path = cls._resolve_path(path, dry_run)
        if not path.exists():
            return cls(_dry_run=dry_run)
        try:
            raw = json.loads(path.read_text())
            positions = [CSPPosition.from_dict(p) for p in raw.get("positions", [])]
            return cls(
                positions=positions,
                total_premium_collected=raw.get("total_premium_collected", 0.0),
                total_realized_pnl=raw.get("total_realized_pnl", 0.0),
                total_trades=raw.get("total_trades", 0),
                last_run=raw.get("last_run"),
                prev_hv_ratios=raw.get("prev_hv_ratios", {}),
                _dry_run=dry_run,
            )
        except Exception as e:
            logger.error("Failed to load CSP state from %s: %s -- starting fresh", path, e)
            return cls(_dry_run=dry_run)


# =============================================================================
# DATA LAYER
# =============================================================================

class CSPDataLayer:
    """Fetches and caches market data for the CSP strategy.

    Mirrors DataLayer in slvr_cc_scalper.py but adds:
    - SLV intraday data (Signal 4: cross-asset selling confirmation)
    - IV-rank estimation via rolling HV percentile (IV-rank gates)
    """

    # Tickers the data layer must track
    _WATCH = CSP_ELIGIBLE_TICKERS + [GLD_TICKER, UUP_TICKER, SLV_TICKER]

    def __init__(self) -> None:
        self._price_cache: Dict[str, float] = {}
        self._daily_data: Dict[str, pd.DataFrame] = {}
        self._intraday_data: Dict[str, pd.DataFrame] = {}
        self._prev_close: Dict[str, float] = {}
        self._option_chains: Dict[str, Dict[str, Any]] = {}

    def refresh(self) -> bool:
        """Full refresh of daily + intraday data. Returns True on full success."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return False

        success = True
        for ticker in list(set(self._WATCH)):
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period=YF_HISTORY_PERIOD)
                if hist is not None and len(hist) > 0:
                    self._daily_data[ticker] = hist
                    self._price_cache[ticker] = float(hist["Close"].iloc[-1])
                    if len(hist) >= 2:
                        self._prev_close[ticker] = float(hist["Close"].iloc[-2])
                    logger.info(
                        "CSP data: %s %d bars last=$%.2f",
                        ticker, len(hist), self._price_cache[ticker],
                    )
                else:
                    logger.warning("CSP data: no bars for %s", ticker)
                    success = False
            except Exception as e:
                logger.error("CSP data: failed %s: %s", ticker, e)
                success = False

        # Intraday for GLD (Signal 6) and SLV (Signal 4)
        for indicator in [GLD_TICKER, SLV_TICKER]:
            try:
                import yfinance as yf
                t = yf.Ticker(indicator)
                intra = t.history(period=YF_INTRADAY_PERIOD, interval=YF_INTRADAY_INTERVAL)
                if intra is not None and len(intra) > 0:
                    self._intraday_data[indicator] = intra
                    logger.info("CSP intraday: %s %d bars", indicator, len(intra))
            except Exception as e:
                logger.error("CSP intraday: failed %s: %s", indicator, e)

        return success

    def refresh_intraday(self) -> bool:
        """Light refresh of intraday prices for the poll cycle."""
        try:
            import yfinance as yf
        except ImportError:
            return False
        try:
            for indicator in [GLD_TICKER, SLV_TICKER]:
                t = yf.Ticker(indicator)
                intra = t.history(period="1d", interval=YF_INTRADAY_INTERVAL)
                if intra is not None and len(intra) > 0:
                    self._intraday_data[indicator] = intra
            for ticker in list(set(self._WATCH)):
                t = yf.Ticker(ticker)
                fast = t.history(period="1d")
                if fast is not None and len(fast) > 0:
                    self._price_cache[ticker] = float(fast["Close"].iloc[-1])
            return True
        except Exception as e:
            logger.error("CSP intraday refresh failed: %s", e)
            return False

    # --- Price accessors ---

    def get_price(self, ticker: str) -> Optional[float]:
        return self._price_cache.get(ticker)

    def get_prev_close(self, ticker: str) -> Optional[float]:
        return self._prev_close.get(ticker)

    def get_daily_change_pct(self, ticker: str) -> Optional[float]:
        price = self.get_price(ticker)
        prev = self.get_prev_close(ticker)
        if price is None or prev is None or prev <= 0:
            return None
        return (price - prev) / prev

    # --- Technical indicators ---

    def compute_rsi(self, ticker: str, period: int = CSP_RSI_PERIOD) -> Optional[float]:
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
        return float(last_rsi) if not pd.isna(last_rsi) else None

    def compute_hv(self, ticker: str, window: int) -> Optional[float]:
        """Annualized historical volatility over window trading days."""
        df = self._daily_data.get(ticker)
        if df is None or len(df) < window + 1:
            return None
        closes = df["Close"]
        log_returns = np.log(closes / closes.shift(1)).dropna()
        if len(log_returns) < window:
            return None
        rv = float(log_returns.tail(window).std()) * math.sqrt(252)
        return rv if not math.isnan(rv) else None

    def estimate_iv_rank(self, ticker: str, lookback: int = 252) -> Optional[float]:
        """Estimate IV rank as the HV-30 percentile over the past `lookback` days.

        Returns 0-100.  Uses rolling 30-day HV as a proxy for IV because
        Alpaca free tier does not supply historical implied-volatility.
        Fails conservatively: returns None when data is insufficient.
        """
        df = self._daily_data.get(ticker)
        if df is None or len(df) < 60:
            return None
        closes = df["Close"]
        log_returns = np.log(closes / closes.shift(1)).dropna()
        rolling_hv = log_returns.rolling(30).std() * math.sqrt(252)
        valid = rolling_hv.dropna()
        if len(valid) < 30:
            return None
        current = float(valid.iloc[-1])
        history = valid.tail(lookback).values
        rank = float(np.mean(history <= current)) * 100.0
        return rank

    # --- Intraday macro direction ---

    def macro_short_term_change(self, indicator: str, minutes: int = 15) -> Optional[float]:
        """Percentage change over the last N minutes of intraday 5-min bars."""
        df = self._intraday_data.get(indicator)
        if df is None or len(df) < 2:
            return None
        bars_needed = max(1, minutes // 5)
        if len(df) < bars_needed + 1:
            bars_needed = len(df) - 1
        recent = float(df["Close"].iloc[-1])
        past = float(df["Close"].iloc[-(bars_needed + 1)])
        if past <= 0:
            return None
        return (recent - past) / past

    # --- Option chains (put side) ---

    def fetch_option_chain_puts(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch OTM PUT chain for ticker filtered to CSP DTE / OTM range."""
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed")
            return None
        try:
            t = yf.Ticker(ticker)
            expirations = t.options
            if not expirations:
                logger.warning("No option expirations for %s", ticker)
                return None
            underlying = self.get_price(ticker)
            if underlying is None or underlying <= 0:
                return None

            # OTM puts are below spot
            min_strike = underlying * (1 - CSP_OTM_MAX_PCT)
            max_strike = underlying * (1 - CSP_OTM_MIN_PCT)

            result: Dict[str, Any] = {}
            today = _today_et()

            for exp_str in expirations:
                try:
                    exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
                except ValueError:
                    continue
                dte = (exp_date - today).days
                if dte < CSP_DTE_MIN or dte > CSP_DTE_MAX:
                    continue
                try:
                    chain = t.option_chain(exp_str)
                    puts = chain.puts
                    if puts is None or len(puts) == 0:
                        continue
                    mask = (puts["strike"] >= min_strike) & (puts["strike"] <= max_strike)
                    filtered = puts[mask].copy()
                    if len(filtered) == 0:
                        continue
                    filtered["dte"] = dte
                    filtered["otm_pct"] = (underlying - filtered["strike"]) / underlying
                    filtered["expiration"] = exp_str
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
                        "Put chain %s/%s DTE=%d: %d strikes",
                        ticker, exp_str, dte, len(filtered),
                    )
                except Exception as e:
                    logger.warning("Put chain fetch %s/%s: %s", ticker, exp_str, e)
                    continue

            self._option_chains[ticker] = result
            logger.info("Put chains for %s: %d expirations", ticker, len(result))
            return result
        except Exception as e:
            logger.error("Put chain fetch failed %s: %s", ticker, e)
            return None

    def get_option_chains(self, ticker: str) -> Dict[str, Any]:
        return self._option_chains.get(ticker, {})


# =============================================================================
# SIGNAL ENGINE
# =============================================================================

@dataclass
class CSPSignalResult:
    """Result of a CSP 7-signal evaluation."""
    triggered: bool
    score: int
    max_score: int
    details: Dict[str, Any]
    timestamp: str = ""

    def summary(self) -> str:
        parts = [
            f"CSP SELL {'TRIGGERED' if self.triggered else 'NOT triggered'} "
            f"(score={self.score}/{self.max_score})"
        ]
        for key, val in self.details.items():
            if isinstance(val, dict):
                status = "YES" if val.get("triggered", False) else "NO"
                parts.append(f"  {key}: {status} -- {val.get('reason', '')}")
            else:
                parts.append(f"  {key}: {val}")
        return "\n".join(parts)


class CSPSignalEngine:
    """Evaluates the 7-signal pullback system for CSP put-selling.

    Signals (inverse of CC scalper — fire on PULLBACKS, not rallies):
      1. ticker_down_3pct      — underlying down >=3% intraday
      2. rsi_oversold          — RSI(14) < 35
      3. vol_expanding         — HV10 > HV20
      4. macro_selling         — GLD or SLV down >=0.5% on the day
      5. uup_rallying          — UUP up >=0.10% (dollar bounce)
      6. macro_not_crashing    — GLD 15-min > -0.10% AND 60-min > -0.40%
                                 (not in freefall — avoids waterfall acceleration)
      7. vol_not_accelerating  — HV10/HV20 ratio change < 5% vs prior cycle

    Trigger: 5-of-7.
    """

    def __init__(self, data: CSPDataLayer) -> None:
        self.data = data

    def evaluate_sell(
        self,
        ticker: str,
        prev_hv_ratios: Optional[Dict[str, float]] = None,
    ) -> CSPSignalResult:
        """Evaluate whether to sell a CSP on ticker. Returns CSPSignalResult."""
        details: Dict[str, Any] = {}
        score = 0
        max_score = 7

        # --- Signal 1: ticker down >=3% intraday ---
        ticker_change = self.data.get_daily_change_pct(ticker)
        if ticker_change is not None:
            s1 = ticker_change <= CSP_DOWN_PCT
            details["ticker_down_3pct"] = {
                "triggered": s1,
                "value": f"{ticker_change:.2%}",
                "threshold": f"<={CSP_DOWN_PCT:.2%}",
                "reason": (
                    f"{ticker} {'down' if ticker_change < 0 else 'up'} "
                    f"{abs(ticker_change):.2%} (need <={CSP_DOWN_PCT:.2%})"
                ),
            }
            if s1:
                score += 1
        else:
            details["ticker_down_3pct"] = {
                "triggered": False,
                "reason": f"{ticker} daily change unavailable",
            }

        # --- Signal 2: RSI(14) < 35 ---
        rsi = self.data.compute_rsi(ticker)
        if rsi is not None:
            s2 = rsi < CSP_RSI_OVERSOLD
            details["rsi_oversold"] = {
                "triggered": s2,
                "value": f"{rsi:.1f}",
                "threshold": f"<{CSP_RSI_OVERSOLD}",
                "reason": f"RSI({CSP_RSI_PERIOD})={rsi:.1f} (need <{CSP_RSI_OVERSOLD})",
            }
            if s2:
                score += 1
        else:
            details["rsi_oversold"] = {"triggered": False, "reason": "RSI unavailable"}

        # --- Signal 3: HV10 > HV20 (vol expanding) ---
        hv10 = self.data.compute_hv(ticker, CSP_HV_SHORT)
        hv20 = self.data.compute_hv(ticker, CSP_HV_LONG)
        if hv10 is not None and hv20 is not None:
            s3 = hv10 > hv20
            details["vol_expanding"] = {
                "triggered": s3,
                "value": f"HV10={hv10:.1%} HV20={hv20:.1%}",
                "reason": (
                    f"HV10 {'>' if hv10 > hv20 else '<='} HV20 "
                    f"({hv10:.1%} vs {hv20:.1%})"
                ),
            }
            if s3:
                score += 1
        else:
            details["vol_expanding"] = {"triggered": False, "reason": "HV data unavailable"}

        # --- Signal 4: GLD or SLV down >=0.5% (macro selling confirmation) ---
        gld_chg = self.data.get_daily_change_pct(GLD_TICKER)
        slv_chg = self.data.get_daily_change_pct(SLV_TICKER)
        if gld_chg is not None or slv_chg is not None:
            gld_selling = gld_chg is not None and gld_chg <= CSP_MACRO_SELLING_PCT
            slv_selling = slv_chg is not None and slv_chg <= CSP_MACRO_SELLING_PCT
            s4 = gld_selling or slv_selling
            gld_str = f"GLD={gld_chg:.2%}" if gld_chg is not None else "GLD=N/A"
            slv_str = f"SLV={slv_chg:.2%}" if slv_chg is not None else "SLV=N/A"
            details["macro_selling"] = {
                "triggered": s4,
                "value": f"{gld_str} {slv_str}",
                "threshold": f"<={CSP_MACRO_SELLING_PCT:.2%}",
                "reason": (
                    ("GLD selling " if gld_selling else "")
                    + ("SLV selling" if slv_selling else "")
                    + ("no macro selling" if not s4 else "")
                ),
            }
            if s4:
                score += 1
        else:
            details["macro_selling"] = {
                "triggered": False,
                "reason": "GLD and SLV data unavailable",
            }

        # --- Signal 5: UUP rallying >=0.10% (dollar bounce) ---
        uup_chg = self.data.get_daily_change_pct(UUP_TICKER)
        if uup_chg is not None:
            s5 = uup_chg >= CSP_UUP_RALLY_PCT
            details["uup_rallying"] = {
                "triggered": s5,
                "value": f"{uup_chg:.2%}",
                "threshold": f">={CSP_UUP_RALLY_PCT:.2%}",
                "reason": (
                    f"UUP {'up' if uup_chg >= 0 else 'down'} "
                    f"{abs(uup_chg):.2%} (need >={CSP_UUP_RALLY_PCT:.2%})"
                ),
            }
            if s5:
                score += 1
        else:
            details["uup_rallying"] = {"triggered": False, "reason": "UUP data unavailable"}

        # --- Signal 6: macro NOT crashing ---
        # GLD 15-min change > -0.10% AND GLD 60-min cumulative > -0.40%
        # Prevents selling puts into a waterfall / momentum continuation.
        gld_15m = self.data.macro_short_term_change(GLD_TICKER, minutes=15)
        gld_60m = self.data.macro_short_term_change(GLD_TICKER, minutes=60)

        if gld_15m is not None and gld_60m is not None:
            ok_15m = gld_15m > CSP_MACRO_NOTCRASH_15M
            ok_60m = gld_60m > CSP_MACRO_NOTCRASH_60M
            s6 = ok_15m and ok_60m
            details["macro_not_crashing"] = {
                "triggered": s6,
                "value": f"GLD 15m={gld_15m:.3%} 60m={gld_60m:.3%}",
                "threshold": (
                    f"15m>{CSP_MACRO_NOTCRASH_15M:.3%} "
                    f"AND 60m>{CSP_MACRO_NOTCRASH_60M:.3%}"
                ),
                "reason": (
                    f"GLD 15m={gld_15m:.3%} "
                    f"({'OK' if ok_15m else 'CRASHING'}), "
                    f"60m={gld_60m:.3%} "
                    f"({'OK' if ok_60m else 'WATERFALL'})"
                ),
            }
            if s6:
                score += 1
        elif gld_15m is not None:
            # Have 15m but not 60m: apply 15m gate only (conservative)
            ok_15m = gld_15m > CSP_MACRO_NOTCRASH_15M
            details["macro_not_crashing"] = {
                "triggered": ok_15m,
                "value": f"GLD 15m={gld_15m:.3%} 60m=N/A",
                "reason": (
                    f"GLD 15m={gld_15m:.3%} "
                    f"({'OK' if ok_15m else 'CRASHING'}); 60m unavailable"
                ),
            }
            if ok_15m:
                score += 1
        else:
            # No intraday data: fail-open (assume not crashing)
            details["macro_not_crashing"] = {
                "triggered": True,
                "reason": "GLD intraday unavailable -- assuming not crashing (fail-open)",
            }
            score += 1

        # --- Signal 7: vol ratio not accelerating vs prior cycle ---
        # HV10/HV20 ratio increasing >5% signals accelerating vol expansion —
        # selling puts into accelerating vol risks getting run over.
        if hv10 is not None and hv20 is not None and hv20 > 0:
            current_ratio = hv10 / hv20
            prior_ratio = (prev_hv_ratios or {}).get(ticker)
            if prior_ratio is not None and prior_ratio > 0:
                ratio_change = (current_ratio - prior_ratio) / prior_ratio
                s7 = ratio_change < CSP_VOL_ACCEL_THRESHOLD
                details["vol_not_accelerating"] = {
                    "triggered": s7,
                    "value": (
                        f"ratio={current_ratio:.3f} "
                        f"prior={prior_ratio:.3f} "
                        f"chg={ratio_change:+.1%}"
                    ),
                    "threshold": f"change<{CSP_VOL_ACCEL_THRESHOLD:.0%}",
                    "reason": (
                        f"HV10/HV20 ratio change {ratio_change:+.1%} "
                        f"({'stable' if s7 else 'ACCELERATING -- skip'})"
                    ),
                }
            else:
                # No prior: assume stable (first cycle)
                s7 = True
                details["vol_not_accelerating"] = {
                    "triggered": True,
                    "value": f"ratio={current_ratio:.3f} (no prior)",
                    "reason": "No prior HV ratio -- assuming stable (first cycle)",
                }
            if s7:
                score += 1
        else:
            # HV unavailable: fail-open
            details["vol_not_accelerating"] = {
                "triggered": True,
                "reason": "HV unavailable for ratio -- assuming stable (fail-open)",
            }
            score += 1

        triggered = score >= CSP_MIN_SELL_SIGNALS
        now_iso = datetime.now(timezone.utc).isoformat()
        return CSPSignalResult(
            triggered=triggered,
            score=score,
            max_score=max_score,
            details=details,
            timestamp=now_iso,
        )

    def evaluate_buy_back(
        self, position: CSPPosition, current_option_price: float,
    ) -> Dict[str, Any]:
        """Evaluate whether to close an existing CSP position.

        Returns dict: triggered, reasons, profit_pct, current_price, dte.
        Any single exit condition fires.
        """
        reasons: List[str] = []
        profit_pct = position.profit_pct(current_option_price)
        dte = position.days_to_expiration

        if profit_pct >= CSP_PROFIT_TARGET:
            reasons.append(
                f"profit target {profit_pct:.1%} >= {CSP_PROFIT_TARGET:.0%}"
            )

        if current_option_price >= position.sell_price * CSP_STOP_LOSS_MULT:
            reasons.append(
                f"stop loss: current=${current_option_price:.2f} "
                f">= {CSP_STOP_LOSS_MULT:.0f}x entry=${position.sell_price:.2f}"
            )

        if dte <= CSP_DTE_TRIGGER:
            reasons.append(f"DTE trigger: {dte} DTE <= {CSP_DTE_TRIGGER}")

        return {
            "triggered": len(reasons) > 0,
            "reasons": reasons,
            "profit_pct": profit_pct,
            "current_price": current_option_price,
            "dte": dte,
        }


# =============================================================================
# STRIKE SELECTOR
# =============================================================================

class CSPStrikeSelector:
    """Selects optimal OTM put strikes from the fetched chain.

    Applies:
      - OI >= 100
      - bid >= 2.0% of strike (premium floor)
      - bid-ask spread < 30%
      - DTE 30-45
      - Scoring: delta closeness to 25-delta, DTE quality, premium quality
    """

    def __init__(self, data: CSPDataLayer) -> None:
        self.data = data

    def select_strikes(self, ticker: str) -> List[Dict[str, Any]]:
        """Return candidates sorted by score (best first)."""
        chains = self.data.get_option_chains(ticker)
        if not chains:
            logger.warning("No put chains for %s strike selection", ticker)
            return []

        underlying = self.data.get_price(ticker)
        if underlying is None or underlying <= 0:
            return []

        # Use HV20 as volatility estimate for delta calculation
        hv20 = self.data.compute_hv(ticker, 20)
        sigma_est = hv20 if (hv20 is not None and hv20 > 0) else 0.35

        dte_mid = (CSP_DTE_MIN + CSP_DTE_MAX) / 2.0
        candidates: List[Dict[str, Any]] = []

        for exp_str, df in chains.items():
            if df is None or len(df) == 0:
                continue
            for _, row in df.iterrows():
                strike = float(row.get("strike", 0))
                bid = float(row.get("bid", 0))
                ask = float(row.get("ask", 0))
                mid = float(row.get("mid_price", 0))
                if mid <= 0:
                    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else float(row.get("lastPrice", 0))

                _oi = row.get("openInterest", 0)
                oi = (
                    int(_oi)
                    if _oi is not None and not (isinstance(_oi, float) and math.isnan(_oi))
                    else 0
                )
                _dte = row.get("dte", 0)
                dte = (
                    int(_dte)
                    if _dte is not None and not (isinstance(_dte, float) and math.isnan(_dte))
                    else 0
                )
                contract_symbol = str(row.get("contractSymbol", ""))
                otm_pct = float(row.get("otm_pct", 0))

                # Gate: OI >= 100
                if oi < CSP_MIN_OPEN_INTEREST:
                    continue

                # Gate: bid >= 2.0% of strike (premium floor per spec)
                if strike <= 0 or bid < strike * CSP_PREMIUM_FLOOR_PCT:
                    logger.debug(
                        "CSP floor: %s %s $%.0f bid=%.2f < %.0f%% floor",
                        ticker, exp_str, strike, bid, CSP_PREMIUM_FLOOR_PCT * 100,
                    )
                    continue

                # Gate: bid-ask spread
                if bid > 0 and ask > 0:
                    spread_pct = (ask - bid) / ((bid + ask) / 2.0)
                    if spread_pct > CSP_MAX_BID_ASK_SPREAD_PCT:
                        continue

                # Compute BS put delta for ranking
                T = max(dte, 1) / 365.0
                delta = bs_put_delta(underlying, strike, T, RISK_FREE_RATE, sigma_est)

                # Score: delta closeness, DTE quality, premium quality
                delta_score = max(0.0, 1.0 - abs(delta - CSP_TARGET_DELTA) / CSP_TARGET_DELTA)
                dte_score = max(0.0, 1.0 - abs(dte - dte_mid) / (dte_mid - CSP_DTE_MIN))
                # Premium quality: how far above floor?
                floor_val = strike * CSP_PREMIUM_FLOOR_PCT
                premium_excess = min((bid / floor_val) - 1.0, 2.0) if floor_val > 0 else 0.0

                score = delta_score + dte_score * 0.5 + premium_excess * 0.3

                candidates.append({
                    "ticker": ticker,
                    "contract_symbol": contract_symbol,
                    "strike": strike,
                    "expiration": exp_str,
                    "dte": dte,
                    "otm_pct": otm_pct,
                    "mid_price": mid,
                    "bid": bid,
                    "ask": ask,
                    "oi": oi,
                    "delta": delta,
                    "score": score,
                })

        candidates.sort(key=lambda c: c["score"], reverse=True)
        if candidates:
            best = candidates[0]
            logger.info(
                "Strike selection %s: %d candidates — best strike=$%.1f "
                "delta=%.2f DTE=%d bid=%.2f",
                ticker, len(candidates),
                best["strike"], best["delta"], best["dte"], best["bid"],
            )
        return candidates


# =============================================================================
# EXECUTION LAYER
# =============================================================================

def _http_detail(e: Exception) -> str:
    body = ""
    try:
        body = e.response.text[:300]  # type: ignore[attr-defined]
    except Exception:
        pass
    return f"{e} | body={body}" if body else str(e)


class CSPExecutionLayer:
    """Handles order placement for CSP positions via Alpaca REST.

    Key difference from CC scalper ExecutionLayer:
      - No coverage check (cash-secured by design)
      - Cash verification: queries Alpaca directly for available cash BEFORE
        submit (fail-closed — mirrors pmcc_manager direct-Alpaca-query pattern).
        An order is rejected if available cash < required_cash.
    """

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.api_key: Optional[str] = None
        self.api_secret: Optional[str] = None
        self._session = None

    def initialize(self) -> bool:
        try:
            from jarvis_utils.secrets import get
            self.api_key = get("Alpaca", "api_key_id", user=ALPACA_USER_ID)
            self.api_secret = get("Alpaca", "secret_key", user=ALPACA_USER_ID)
            if not self.api_key or not self.api_secret:
                logger.warning("CSP: Alpaca creds not found -- forcing dry-run")
                self.dry_run = True
                return False
            logger.info("CSP: Alpaca creds loaded for %s", ALPACA_USER_ID)
            return True
        except Exception as e:
            logger.error("CSP: credential load failed: %s", e)
            self.dry_run = True
            return False

    def _refresh_session_credentials(self, session) -> None:
        from jarvis_utils.secrets import get
        new_key = get("Alpaca", "api_key_id", user=ALPACA_USER_ID)
        new_secret = get("Alpaca", "secret_key", user=ALPACA_USER_ID)
        if not new_key or not new_secret:
            raise EnvironmentError("csp_seller cred refresh: portal returned empty creds")
        self.api_key = new_key
        self.api_secret = new_secret
        session.headers["APCA-API-KEY-ID"] = new_key
        session.headers["APCA-API-SECRET-KEY"] = new_secret
        logger.info("CSP: creds refreshed (prefix %s...)", new_key[:6])

    def _get_session(self):
        cached_key = (
            self._session.headers.get("APCA-API-KEY-ID", "")
            if self._session is not None else ""
        )
        current_key = self.api_key or ""
        if self._session is None or cached_key != current_key:
            from alpaca_client import _AutoRefreshSession
            self._session = _AutoRefreshSession(self._refresh_session_credentials)
            self._session.headers.update({
                "APCA-API-KEY-ID": current_key,
                "APCA-API-SECRET-KEY": self.api_secret or "",
            })
        return self._session

    def get_available_cash(self) -> Optional[float]:
        """Query Alpaca directly for available cash (buying power for CSP).

        FAIL-CLOSED: returns None on any error so caller must treat None as
        insufficient cash and block the submit.
        """
        if self.dry_run:
            return 100_000.0
        try:
            sess = self._get_session()
            r = sess.get(f"{ALPACA_BASE_URL}/v2/account", timeout=10)
            if r.status_code != 200:
                logger.error("CSP cash check HTTP %d -- fail-closed", r.status_code)
                return None
            acct = r.json()
            cash = float(acct.get("cash", 0) or 0)
            logger.debug("CSP: Alpaca cash=%.2f", cash)
            return cash
        except Exception as e:
            logger.error("CSP cash check exception -- fail-closed: %s", e)
            return None

    def sell_put(
        self,
        contract_symbol: str,
        contracts: int,
        limit_price: float,
        required_cash: float,
    ) -> Optional[str]:
        """Place a limit sell order for a cash-secured put.

        Verifies available cash before submitting (fail-closed).
        Returns Alpaca order ID on success, None on failure.
        """
        if self.dry_run:
            order_id = f"DRY_CSP_SELL_{contract_symbol}_{int(time.time())}"
            logger.info(
                "[DRY RUN] CSP SELL %d x %s @ $%.2f (cash_needed=$%.0f) id=%s",
                contracts, contract_symbol, limit_price, required_cash, order_id,
            )
            return order_id

        # Cash-secured check: query Alpaca directly (fail-closed)
        available = self.get_available_cash()
        if available is None:
            logger.error(
                "CSP SELL BLOCKED: cash query failed (fail-closed) %s",
                contract_symbol,
            )
            return None
        if available < required_cash:
            logger.error(
                "CSP SELL BLOCKED: insufficient cash $%.0f < required $%.0f for %s",
                available, required_cash, contract_symbol,
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
            resp = session.post(f"{ALPACA_BASE_URL}/v2/orders", json=payload, timeout=15)
            resp.raise_for_status()
            order = resp.json()
            order_id = order.get("id", "")
            logger.info(
                "CSP SELL PLACED: %d x %s @ $%.2f (cash_rsv=$%.0f) id=%s status=%s",
                contracts, contract_symbol, limit_price,
                required_cash, order_id, order.get("status", "?"),
            )
            return order_id
        except Exception as e:
            logger.error(
                "CSP SELL FAILED: %d x %s @ $%.2f -- %s",
                contracts, contract_symbol, limit_price, _http_detail(e),
            )
            return None

    def buy_back_put(
        self,
        contract_symbol: str,
        contracts: int,
        limit_price: float,
    ) -> Optional[str]:
        """Place a limit buy order to close a sold put position."""
        if self.dry_run:
            order_id = f"DRY_CSP_BUY_{contract_symbol}_{int(time.time())}"
            logger.info(
                "[DRY RUN] CSP BUY-BACK %d x %s @ $%.2f id=%s",
                contracts, contract_symbol, limit_price, order_id,
            )
            return order_id
        try:
            session = self._get_session()
            payload = {
                "symbol": contract_symbol,
                "qty": str(contracts),
                "side": "buy",
                "type": "limit",
                "time_in_force": ORDER_TIF,
                "limit_price": str(round(limit_price, 2)),
            }
            resp = session.post(f"{ALPACA_BASE_URL}/v2/orders", json=payload, timeout=15)
            resp.raise_for_status()
            order = resp.json()
            order_id = order.get("id", "")
            logger.info(
                "CSP BUY-BACK PLACED: %d x %s @ $%.2f id=%s status=%s",
                contracts, contract_symbol, limit_price,
                order_id, order.get("status", "?"),
            )
            return order_id
        except Exception as e:
            logger.error(
                "CSP BUY-BACK FAILED: %d x %s @ $%.2f -- %s",
                contracts, contract_symbol, limit_price, _http_detail(e),
            )
            return None

    def get_current_option_price(self, contract_symbol: str) -> Optional[float]:
        """Fetch current mid-price for an options contract via Alpaca snapshot."""
        if self.dry_run:
            return None
        try:
            sess = self._get_session()
            r = sess.get(
                f"https://data.alpaca.markets/v1beta1/options/snapshots/{contract_symbol}",
                params={"feed": ALPACA_FEED},
                timeout=10,
            )
            if r.status_code == 200:
                snap = r.json()
                quote = snap.get("latestQuote", {})
                bid = float(quote.get("bp", 0) or 0)
                ask = float(quote.get("ap", 0) or 0)
                if bid > 0 and ask > 0:
                    return (bid + ask) / 2.0
                trade = snap.get("latestTrade", {})
                price = float(trade.get("p", 0) or 0)
                return price if price > 0 else None
        except Exception as e:
            logger.debug("CSP: option price fetch failed %s: %s", contract_symbol, e)
        return None

    def check_account(self) -> Optional[Dict[str, Any]]:
        if self.dry_run:
            return {"equity": "100000", "status": "ACTIVE (dry-run)"}
        try:
            session = self._get_session()
            resp = session.get(f"{ALPACA_BASE_URL}/v2/account", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("CSP account check failed: %s", _http_detail(e))
            return None


# =============================================================================
# MAIN CSP SELLER CLASS
# =============================================================================

class CSPSeller:
    """Cash-Secured Put seller for tail-index-eligible tickers (WPM, GDX).

    Orchestrates:
      - 7-signal pullback detection (5-of-7 trigger)
      - Strike selection (25-delta, 30-45 DTE, bid >= 2% of strike)
      - Position cap enforcement (1/ticker, <=15% equity, <=3 total)
      - Combined stock+CSP cap (strike-based, not spot)
      - Exit management (50% profit, 2x stop, 7 DTE)
      - Cross-engine dedup via order_dedup.py CSP namespace

    Usage:
        seller = CSPSeller(dry_run=True)
        seller.initialize()
        result = seller.run_once()     # one cycle
        seller.run()                   # continuous loop
        print(seller.show_status())    # display positions
    """

    def __init__(self, dry_run: bool = True) -> None:
        self.dry_run = dry_run
        self.data = CSPDataLayer()
        self.signals = CSPSignalEngine(self.data)
        self.strikes = CSPStrikeSelector(self.data)
        self.executor = CSPExecutionLayer(dry_run=dry_run)
        self.state = CSPState.load(dry_run=dry_run)
        self._order_dedup = None        # injected by combined_runner
        self._equity: float = 100_000.0  # updated on initialize / each cycle

    def initialize(self) -> bool:
        logger.info("=" * 60)
        logger.info(
            "CSP SELLER INITIALIZING  mode=%s  eligible=%s",
            "DRY RUN" if self.dry_run else "LIVE",
            CSP_ELIGIBLE_TICKERS,
        )
        logger.info("=" * 60)

        if not CSP_ELIGIBLE_TICKERS:
            logger.warning(
                "CSP: no eligible tickers "
                "(no CC_OPTIONS_ELIGIBLE ticker has α < %.1f)",
                CC_MIN_TAIL_INDEX,
            )
            return False

        creds_ok = self.executor.initialize()
        if not creds_ok and not self.dry_run:
            logger.warning("CSP: no creds -- switching to dry-run")
            self.dry_run = True
            self.executor.dry_run = True

        acct = self.executor.check_account()
        if acct:
            try:
                self._equity = float(acct.get("equity", self._equity))
            except (ValueError, TypeError):
                pass
            logger.info(
                "CSP: equity=$%.2f status=%s",
                self._equity, acct.get("status", "?"),
            )

        data_ok = self.data.refresh()
        if not data_ok:
            logger.warning("CSP: data refresh had errors")

        return True

    # ------------------------------------------------------------------
    # Position cap helpers
    # ------------------------------------------------------------------

    def _csp_positions_for_ticker(self, ticker: str) -> List[CSPPosition]:
        return [p for p in self.state.open_positions() if p.ticker == ticker]

    def _total_csp_cash_reserved(self) -> float:
        return sum(p.cash_reserved for p in self.state.open_positions())

    def _stock_exposure(
        self, ticker: str, alpaca_positions: Optional[List[Dict]]
    ) -> float:
        """Dollar value of existing stock position in ticker from Alpaca."""
        if not alpaca_positions:
            return 0.0
        for p in alpaca_positions:
            if p.get("symbol", "") == ticker:
                try:
                    return float(p.get("market_value", 0) or 0)
                except (ValueError, TypeError):
                    return 0.0
        return 0.0

    def _check_caps(
        self,
        ticker: str,
        strike: float,
        contracts: int,
        alpaca_positions: Optional[List[Dict]] = None,
    ) -> Tuple[bool, str]:
        """Verify all four position caps.

        Cap 1: 1 active CSP per ticker
        Cap 2: total CSP cash <= 15% of equity
        Cap 3: hard ceiling <= 3 concurrent CSPs
        Cap 4: (stock_$ + strike*contracts*100) <= 8% equity
               *** strike price, NOT spot (per spec) ***
        """
        # Cap 1
        if self._csp_positions_for_ticker(ticker):
            return False, f"cap1: already 1 CSP on {ticker}"

        # Cap 2
        new_cash = strike * contracts * 100
        total_cash = self._total_csp_cash_reserved() + new_cash
        max_cash = self._equity * CSP_MAX_TOTAL_EQUITY_PCT
        if total_cash > max_cash:
            return False, (
                f"cap2: total CSP cash ${total_cash:.0f} > "
                f"${max_cash:.0f} ({CSP_MAX_TOTAL_EQUITY_PCT:.0%} of equity)"
            )

        # Cap 3
        if len(self.state.open_positions()) >= CSP_MAX_CONCURRENT:
            return False, (
                f"cap3: {CSP_MAX_CONCURRENT} concurrent CSPs open (hard ceiling)"
            )

        # Cap 4: combined stock + CSP at STRIKE (not spot)
        stock_exp = self._stock_exposure(ticker, alpaca_positions)
        csp_exp = strike * contracts * 100  # assignment exposure at strike
        combined = stock_exp + csp_exp
        max_combined = self._equity * CSP_COMBINED_CAP_PCT
        if combined > max_combined:
            return False, (
                f"cap4: stock=${stock_exp:.0f} + csp_strike=${csp_exp:.0f} = "
                f"${combined:.0f} > ${max_combined:.0f} "
                f"({CSP_COMBINED_CAP_PCT:.0%} of equity)"
            )

        return True, "OK"

    def _check_premium_gates(self, ticker: str) -> Tuple[bool, str]:
        """Check the three IV-rank gates that must ALL pass before evaluating strikes.

        Gate 1: ticker IV-rank > 30
        Gate 2: SLV or GLD IV-rank > 30
        """
        ticker_rank = self.data.estimate_iv_rank(ticker)
        if ticker_rank is None:
            return False, f"{ticker} IV-rank unavailable -- fail-closed"
        if ticker_rank <= CSP_IV_RANK_MIN_TICKER:
            return False, (
                f"{ticker} IV-rank={ticker_rank:.0f} "
                f"<= {CSP_IV_RANK_MIN_TICKER} (low volatility environment)"
            )

        slv_rank = self.data.estimate_iv_rank(SLV_TICKER)
        gld_rank = self.data.estimate_iv_rank(GLD_TICKER)
        slv_str = f"{slv_rank:.0f}" if slv_rank is not None else "N/A"
        gld_str = f"{gld_rank:.0f}" if gld_rank is not None else "N/A"

        cross_ok = (
            (slv_rank is not None and slv_rank > CSP_IV_RANK_MIN_CROSS)
            or (gld_rank is not None and gld_rank > CSP_IV_RANK_MIN_CROSS)
        )
        if not cross_ok:
            return False, (
                f"cross-asset IV-rank gate: SLV={slv_str} GLD={gld_str} "
                f"-- need SLV or GLD > {CSP_IV_RANK_MIN_CROSS}"
            )

        return True, (
            f"{ticker} IV-rank={ticker_rank:.0f} OK; "
            f"SLV={slv_str} GLD={gld_str}"
        )

    # ------------------------------------------------------------------
    # Dedup check (CSP namespace)
    # ------------------------------------------------------------------

    def _dedup_blocked(self, ticker: str) -> bool:
        """Return True if another engine OR the CSP itself has a sell on ticker.

        Checks both the CC-side dedup (prevent CC+CSP on same ticker) and the
        CSP-specific dedup (prevent double CSP on same ticker).
        FAIL-CLOSED on exception.
        """
        if self._order_dedup is None:
            return False
        try:
            # Block if any CC engine has a sell (cross-engine safety)
            if self._order_dedup.has_pending_or_active_sell(ticker):
                logger.info(
                    "CSP DEDUP: %s blocked -- CC engine has active position", ticker,
                )
                return True
            # Block if another CSP is already in flight
            if self._order_dedup.has_pending_or_active_csp(ticker):
                logger.info(
                    "CSP DEDUP: %s blocked -- CSP already active", ticker,
                )
                return True
            return False
        except Exception as e:
            logger.warning(
                "CSP DEDUP FAIL-CLOSED: exception checking %s -- assuming blocked: %s",
                ticker, e,
            )
            return True

    # ------------------------------------------------------------------
    # Alpaca positions (for combined-cap math)
    # ------------------------------------------------------------------

    def _fetch_alpaca_positions(self) -> Optional[List[Dict]]:
        if self.dry_run:
            return []
        try:
            sess = self.executor._get_session()
            r = sess.get(f"{ALPACA_BASE_URL}/v2/positions", timeout=10)
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            logger.warning("CSP: positions fetch failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Main cycle
    # ------------------------------------------------------------------

    def run_once(self) -> Dict[str, Any]:
        """Single evaluation cycle.

        Returns: { sell_signals, buy_back_actions, errors, eligible_tickers }
        """
        result: Dict[str, Any] = {
            "sell_signals": [],
            "buy_back_actions": [],
            "errors": [],
            "eligible_tickers": CSP_ELIGIBLE_TICKERS,
        }

        if not CSP_ELIGIBLE_TICKERS:
            return result

        try:
            self.data.refresh_intraday()
        except Exception as e:
            logger.warning("CSP: intraday refresh error (proceeding): %s", e)

        # Refresh equity
        acct = self.executor.check_account()
        if acct:
            try:
                self._equity = float(acct.get("equity", self._equity))
            except (ValueError, TypeError):
                pass

        alpaca_positions = self._fetch_alpaca_positions()

        # Phase 1: Manage open positions (exits)
        for pos in list(self.state.open_positions()):
            try:
                self._manage_open_position(pos, result)
            except Exception as e:
                err = f"Error managing {pos.option_symbol}: {e}"
                logger.error(err)
                result["errors"].append(err)

        # Phase 2: Evaluate new sells
        for ticker in CSP_ELIGIBLE_TICKERS:
            try:
                sig = self._evaluate_new_sell(ticker, alpaca_positions, result)
                result["sell_signals"].append(sig)
            except Exception as e:
                err = f"Error evaluating {ticker}: {e}"
                logger.error(err)
                result["errors"].append(err)

        # Persist HV ratios for Signal 7 next cycle
        for ticker in CSP_ELIGIBLE_TICKERS:
            hv10 = self.data.compute_hv(ticker, CSP_HV_SHORT)
            hv20 = self.data.compute_hv(ticker, CSP_HV_LONG)
            if hv10 is not None and hv20 is not None and hv20 > 0:
                self.state.prev_hv_ratios[ticker] = hv10 / hv20

        self.state.last_run = datetime.now(timezone.utc).isoformat()
        self.state.save()
        return result

    def _manage_open_position(
        self, pos: CSPPosition, result: Dict[str, Any],
    ) -> None:
        """Check exit conditions for an open CSP and buy back if triggered."""
        current_price = self.executor.get_current_option_price(pos.option_symbol)
        if current_price is None:
            # Fall back to Black-Scholes estimate
            underlying = self.data.get_price(pos.ticker)
            if underlying is not None:
                dte = pos.days_to_expiration
                hv20 = self.data.compute_hv(pos.ticker, 20)
                sigma = max(hv20 or 0.35, 0.10)
                T = max(dte, 0) / 365.0
                current_price = bs_put_price(underlying, pos.strike, T, RISK_FREE_RATE, sigma)
            else:
                logger.debug("CSP: cannot price %s -- skipping exit check", pos.option_symbol)
                return

        exit_check = self.signals.evaluate_buy_back(pos, current_price)
        if not exit_check["triggered"]:
            logger.debug(
                "CSP hold: %s profit=%.1f%% DTE=%d",
                pos.option_symbol,
                exit_check["profit_pct"] * 100,
                exit_check["dte"],
            )
            return

        logger.info(
            "CSP EXIT: %s (%s) -- %s",
            pos.option_symbol, pos.ticker,
            "; ".join(exit_check["reasons"]),
        )

        # Buy back slightly above mid to take the fill
        limit_price = round(current_price + CSP_BUYBACK_OFFSET_FROM_MID, 2)
        order_id = self.executor.buy_back_put(pos.option_symbol, pos.contracts, limit_price)
        if order_id:
            pnl = pos.unrealized_pnl(current_price)
            pos.status = "closed"
            pos.buy_back_price = current_price
            pos.buy_back_date = datetime.now(timezone.utc).isoformat()
            pos.buy_back_reason = "; ".join(exit_check["reasons"])
            pos.realized_pnl = pnl
            self.state.total_realized_pnl += pnl
            self.state.total_trades += 1
            result["buy_back_actions"].append({
                "ticker": pos.ticker,
                "option_symbol": pos.option_symbol,
                "buy_back_price": current_price,
                "pnl": pnl,
                "reason": pos.buy_back_reason,
                "order_id": order_id,
            })
            logger.info(
                "CSP CLOSED: %s P&L=$%.2f (%.1f%%) order_id=%s",
                pos.option_symbol, pnl, exit_check["profit_pct"] * 100, order_id,
            )
        else:
            logger.error("CSP: buy-back order failed for %s", pos.option_symbol)

    def _evaluate_new_sell(
        self,
        ticker: str,
        alpaca_positions: Optional[List[Dict]],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate and potentially execute a new CSP sell on ticker."""
        sig_info: Dict[str, Any] = {
            "ticker": ticker,
            "triggered": False,
            "signal_score": 0,
            "reason": "",
        }

        # Cross-engine dedup check (CC + CSP mutual exclusion)
        if self._dedup_blocked(ticker):
            sig_info["reason"] = "dedup blocked"
            return sig_info

        # Per-ticker concurrent cap (quick check before signal eval)
        if self._csp_positions_for_ticker(ticker):
            sig_info["reason"] = f"cap: already 1 CSP on {ticker}"
            return sig_info

        # Signal evaluation (7-signal, 5-of-7 required)
        signal = self.signals.evaluate_sell(
            ticker, prev_hv_ratios=self.state.prev_hv_ratios
        )
        sig_info["signal_score"] = signal.score
        sig_info["signal_details"] = signal.details

        if not signal.triggered:
            sig_info["reason"] = (
                f"signals {signal.score}/{signal.max_score} "
                f"(need {CSP_MIN_SELL_SIGNALS})"
            )
            logger.info(
                "CSP %s: %d/%d signals -- skip",
                ticker, signal.score, signal.max_score,
            )
            return sig_info

        logger.info(
            "CSP %s: SIGNAL TRIGGERED %d/%d -- checking IV-rank gates",
            ticker, signal.score, signal.max_score,
        )

        # IV-rank gates (all three must pass)
        iv_ok, iv_reason = self._check_premium_gates(ticker)
        if not iv_ok:
            sig_info["reason"] = f"IV-rank gate: {iv_reason}"
            logger.info("CSP %s: IV-rank gate blocked: %s", ticker, iv_reason)
            return sig_info

        # Fetch put chain
        chains = self.data.fetch_option_chain_puts(ticker)
        if not chains:
            sig_info["reason"] = f"no put chains for {ticker}"
            logger.warning("CSP %s: no put chains", ticker)
            return sig_info

        # Strike selection
        candidates = self.strikes.select_strikes(ticker)
        if not candidates:
            sig_info["reason"] = f"no qualifying strikes for {ticker}"
            logger.info("CSP %s: no qualifying put strikes", ticker)
            return sig_info

        best = candidates[0]
        strike = best["strike"]
        contracts = 1  # conservative: 1 contract per position
        required_cash = strike * contracts * 100

        # Position cap check (all four caps)
        allowed, cap_reason = self._check_caps(
            ticker, strike, contracts, alpaca_positions
        )
        if not allowed:
            sig_info["reason"] = f"cap: {cap_reason}"
            logger.info("CSP %s: blocked by cap: %s", ticker, cap_reason)
            return sig_info

        # Compute limit price (bid + small offset, but never below premium floor)
        limit_price = round(best["bid"] + CSP_SELL_OFFSET_FROM_MID, 2)
        floor_price = round(strike * CSP_PREMIUM_FLOOR_PCT, 2)
        limit_price = max(limit_price, floor_price)

        # Estimate entry IV for record-keeping
        underlying_price = self.data.get_price(ticker) or 0.0
        T = best["dte"] / 365.0
        sell_iv = (
            implied_volatility_put(
                market_price=best["mid_price"],
                S=underlying_price,
                K=strike,
                T=T,
                r=RISK_FREE_RATE,
            )
            if best["mid_price"] > 0 and underlying_price > 0
            else 0.35
        )

        order_id = self.executor.sell_put(
            best["contract_symbol"], contracts, limit_price, required_cash,
        )

        if order_id:
            pos = CSPPosition(
                ticker=ticker,
                option_symbol=best["contract_symbol"],
                strike=strike,
                expiration=best["expiration"],
                dte_at_entry=best["dte"],
                contracts=contracts,
                sell_price=limit_price,
                sell_date=datetime.now(timezone.utc).isoformat(),
                sell_underlying_price=underlying_price,
                sell_iv=sell_iv,
                sell_signal_score=signal.score,
                sell_signal_details=signal.summary(),
                cash_reserved=required_cash,
                alpaca_order_id=order_id,
            )
            self.state.positions.append(pos)
            self.state.total_premium_collected += pos.premium_collected

            sig_info.update({
                "triggered": True,
                "option_symbol": best["contract_symbol"],
                "strike": strike,
                "expiration": best["expiration"],
                "dte": best["dte"],
                "limit_price": limit_price,
                "order_id": order_id,
                "reason": (
                    f"sold: signals={signal.score}/{signal.max_score} "
                    f"strike=${strike:.1f} ({best['otm_pct']:.1%} OTM) "
                    f"DTE={best['dte']} premium=${limit_price:.2f} "
                    f"delta={best['delta']:.2f}"
                ),
            })
            logger.info(
                "CSP SOLD: %s %s | strike=$%.1f (%.1f%% OTM) DTE=%d "
                "prem=$%.2f/sh ($%.0f total) | sigs=%d/%d | id=%s",
                ticker, best["contract_symbol"],
                strike, best["otm_pct"] * 100, best["dte"],
                limit_price, limit_price * contracts * 100,
                signal.score, signal.max_score, order_id,
            )
        else:
            sig_info["reason"] = f"order submission failed ({best['contract_symbol']})"

        return sig_info

    # ------------------------------------------------------------------
    # Continuous loop (standalone use)
    # ------------------------------------------------------------------

    def run(self) -> None:
        logger.info("CSP SELLER: starting continuous loop (2-min poll)")
        mo_h, mo_m = MARKET_OPEN
        mc_h, mc_m = MARKET_CLOSE
        while True:
            try:
                et = datetime.now(_ET_TZ)
                h, m = et.hour, et.minute
                in_hours = (
                    (h > mo_h or (h == mo_h and m >= mo_m))
                    and (h < mc_h or (h == mc_h and m < mc_m))
                )
                if in_hours:
                    self.run_once()
                else:
                    logger.debug("CSP: market closed")
                time.sleep(120)
            except KeyboardInterrupt:
                logger.info("CSP SELLER: interrupted")
                break
            except Exception as e:
                logger.error("CSP cycle error: %s\n%s", e, traceback.format_exc())
                time.sleep(30)

    # ------------------------------------------------------------------
    # Status / public interface
    # ------------------------------------------------------------------

    def show_status(self) -> str:
        open_pos = self.state.open_positions()
        lines = [
            "=" * 60,
            f"CSP SELLER  mode={'DRY RUN' if self.dry_run else 'LIVE'}",
            f"Eligible tickers: {CSP_ELIGIBLE_TICKERS}",
            f"Open positions: {len(open_pos)} / {CSP_MAX_CONCURRENT} max",
            f"Total premium collected: ${self.state.total_premium_collected:,.2f}",
            f"Total realized P&L: ${self.state.total_realized_pnl:,.2f}",
            f"Total trades: {self.state.total_trades}",
            "",
        ]
        for pos in open_pos:
            lines.append(
                f"  {pos.ticker:6s} | {pos.option_symbol:25s} | "
                f"strike=${pos.strike:.1f} exp={pos.expiration} "
                f"DTE={pos.days_to_expiration} "
                f"entry=${pos.sell_price:.2f} "
                f"cash_rsv=${pos.cash_reserved:,.0f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)

    def active_tickers(self) -> List[str]:
        """Tickers with open CSP positions (used by combined_runner dedup)."""
        return list(set(p.ticker for p in self.state.open_positions()))

    def summary(self) -> Dict[str, Any]:
        open_pos = self.state.open_positions()
        return {
            "open_positions": len(open_pos),
            "total_premium_collected": self.state.total_premium_collected,
            "total_realized_pnl": self.state.total_realized_pnl,
            "total_trades": self.state.total_trades,
            "eligible_tickers": CSP_ELIGIBLE_TICKERS,
        }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="CSP Seller — cash-secured puts on WPM/GDX")
    parser.add_argument("--live", action="store_true", help="Live paper-trade orders")
    parser.add_argument("--once", action="store_true", help="Run one cycle then exit")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    seller = CSPSeller(dry_run=not args.live)
    ok = seller.initialize()
    if not ok:
        logger.error("CSP: initialization failed")
        sys.exit(1)

    if args.status:
        print(seller.show_status())
        return

    if args.once:
        result = seller.run_once()
        logger.info("CSP cycle: %s", result)
        return

    seller.run()


if __name__ == "__main__":
    main()
