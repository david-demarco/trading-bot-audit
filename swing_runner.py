#!/usr/bin/env python3
"""
swing_runner.py - Multi-Strategy Swing Trading Bot for Alpaca Paper Trading.

Implements 5 concurrent strategies across CC-eligible silver mining tickers
(PAAS, AG, HL -- sourced from combined_config.CC_ELIGIBLE_SWING):
  1. Momentum Rotation (25% allocation) - Vol-adjusted momentum, top-5, bi-weekly rebalance
  2. VWAP Mean Reversion (20%) - VWAP z-score < -2, exit at mean
  3. Sector Relative Strength (20%) - Outperformance vs sector ETF
  4. Donchian Breakout (20%) - 20-day high breakout + volume confirmation
  5. RSI(2) Mean Reversion (15%) - Ultra-short-term dip buying in uptrends

Integrated with macro_regime.py for regime overlay (position sizing, sector bias).

Usage:
    python swing_runner.py              # Full run: pull data, compute signals, execute
    python swing_runner.py --dry-run    # Dry run: compute signals but do not place orders
    python swing_runner.py --status     # Show current positions and P&L
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
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

# Ensure our modules are importable
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/opt/jarvis-utils/lib")

logger = logging.getLogger("swing_runner")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Total portfolio capital (reference, actual comes from Alpaca account)
TOTAL_CAPITAL_REFERENCE = 100_000

# Strategy allocations -- imported from combined_config (single source of truth)
# RSI2 = 1.00, all others = 0.00 (disabled). See combined_config.py.
from combined_config import STRATEGY_ALLOCATIONS

# Ticker universe -- imported from combined_config (Mar 12, CRITICAL-1 fix)
# Mar 16: CC_ELIGIBLE_SWING = TRADING_UNIVERSE (one flat list, no ETF/stock split)
from combined_config import CC_ELIGIBLE_SWING as TRADING_TICKERS

# Sector ETF map for relative strength strategy
# Updated Mar 12: removed stale tech/finance/energy/healthcare entries;
# added mining sector ETFs to match the silver-mining swing universe.
SECTOR_ETF_MAP = {
    "silver_mining": "SIL",
    "gold_mining": "GDX",
    "copper_mining": "COPX",
    "metals": "GDX",
    # Oil / Energy sectors (added Mar 18)
    "oil_etf": "XLE",
    "oil_producer": "XLE",
    "oil_services": "XLE",
    "oil_royalty": "XLE",
    "nat_gas": "XLE",
}

SECTOR_ETFS = list(set(SECTOR_ETF_MAP.values()))  # unique ETFs to fetch

# Sector mapping for trading tickers -- updated Mar 15 to match full mining universe.
# Imported from combined_config for single source of truth.
from combined_config import TICKER_SECTOR
from combined_config import MACRO_SIZE_MULT_ENABLED  # Edge 85 — gate

# All tickers we need data for (trading + sector ETFs, deduplicated)
ALL_TICKERS = sorted(set(TRADING_TICKERS + SECTOR_ETFS))

# Max positions per strategy — imported from combined_config (H1/H2 audit fix)
from combined_config import MAX_POSITIONS_PER_STRATEGY as MAX_POSITIONS
from combined_config import SWING_MAX_TOTAL_POSITIONS as MAX_TOTAL_POSITIONS

# --- Momentum Rotation parameters ---
MOM_REBALANCE_DAYS = 10         # Rebalance every 10 trading days
MOM_NUM_POSITIONS = 5
MOM_LOOKBACK_1M = 21            # 1-month momentum (weight 60%)
MOM_LOOKBACK_3M = 63            # 3-month momentum (weight 40%)
MOM_WEIGHT_1M = 0.6
MOM_WEIGHT_3M = 0.4

# --- VWAP Mean Reversion parameters ---
VWAP_PERIOD = 20
VWAP_ENTRY_Z = -2.0
VWAP_EXIT_Z = 0.0
VWAP_STOP_ATR_MULT = 2.5

# --- Sector Relative Strength parameters ---
SRS_RS_PERIOD = 20
SRS_ENTRY_THRESHOLD = 0.02      # Stock outperforms sector by >2%
SRS_EXIT_THRESHOLD = -0.02      # Stock underperforms by >2%
SRS_RS_MA_PERIOD = 5            # 5-day MA of relative strength
SRS_STOP_ATR_MULT = 2.0
SRS_MAX_HOLD_DAYS = 30

# --- Donchian Breakout parameters ---
DONCH_CHANNEL_PERIOD = 20
DONCH_VOLUME_MULT = 1.5         # Volume > 1.5x 20-day average
DONCH_TRAIL_ATR_MULT = 2.0

# --- RSI(2) Mean Reversion parameters (improved strategy, David approved 2026-03-09) ---
# RSI2_ENTRY_THRESHOLD: imported from combined_config (was 15 locally; Mar 12 sync)
from combined_config import RSI2_ENTRY_THRESHOLD, AVERAGING_DOWN_RSI2_THRESHOLD, MAX_LOTS_PER_TICKER
RSI2_MA_PERIOD = 200            # Price must be above 200-day SMA (uptrend filter)
RSI2_EXIT_SMA_PERIOD = 5        # Exit when price crosses above 5-day SMA
RSI2_MAX_HOLD_DAYS = 10         # Safety valve: force exit after 10 trading days
from combined_config import CC_ELIGIBLE_AFTER_DAYS as RSI2_CC_ELIGIBLE_AFTER_DAYS  # H3 audit fix: was hardcoded 5, now 0 per David
# NO stop loss -- backtest proved stops destroy value on these mean-reverting stocks

# --- Portfolio-level risk ---
MAX_SINGLE_POSITION_PCT = 0.15  # 15% of portfolio per position (100-share lots for CCs)
MAX_SECTOR_POSITIONS = 3        # Max 3 positions in same sector across ALL strategies
MAX_PORTFOLIO_EXPOSURE = 1.0    # 100% max (no leverage)
MIN_LOT_SIZE = 100              # Swing entries in 100-share lots (for covered calls)
DD_REDUCE_THRESHOLD = -0.15     # -15% DD: reduce all sizes by 50%
DD_CIRCUIT_BREAKER = -0.20      # -20% DD: go 100% cash for 5 trading days
CIRCUIT_BREAKER_DAYS = 5
SLIPPAGE_PCT = 0.001            # 0.1% slippage assumption

# --- Profit target (David's rule: $0.50/share target) ---
PROFIT_TARGET_PER_SHARE = 0.50  # Close when up $0.50/share ($50 on 100 shares)
# NO stop loss -- backtest proved stops destroy value on mean-reverting stocks

# Earnings protection
# Edge 25 (knowledge/edge_25_earnings_proximity.md, 1583 trades): pre-earnings
# RSI2 entries within 7 CALENDAR days have mean -0.07%, sharpe ≈ 0 vs baseline
# +0.97%. 5 trading days ≈ 7 calendar days. Bumped from 3 → 5 on 2026-04-16.
EARNINGS_BLOCK_DAYS = 5

# Paths
BASE_DIR = Path(__file__).parent.resolve()
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = BASE_DIR / "swing_state.json"
EARNINGS_FILE = BASE_DIR / "earnings_calendar.json"

# Alpaca
PAPER_BASE = "https://paper-api.alpaca.markets/v2"
DATA_BASE = "https://data.alpaca.markets/v2"
ET = pytz.timezone("US/Eastern")


# =============================================================================
# TECHNICAL INDICATOR CALCULATIONS
# =============================================================================

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all required technical indicators on OHLCV DataFrame."""
    df = df.copy()

    # Simple Moving Averages
    df["SMA5"] = df["Close"].rolling(5).mean()
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()
    df["SMA200"] = df["Close"].rolling(200).mean()

    # RSI(14)
    df["RSI14"] = compute_rsi(df["Close"], 14)

    # RSI(2)
    df["RSI2"] = compute_rsi(df["Close"], 2)

    # ATR(14)
    df["ATR14"] = compute_atr(df, 14)

    # VWAP + z-score (rolling 20-day)
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    tp_vol = typical_price * df["Volume"]
    vol_sum = df["Volume"].rolling(VWAP_PERIOD).sum().replace(0, np.nan)
    df["VWAP"] = tp_vol.rolling(VWAP_PERIOD).sum() / vol_sum
    vwap_diff = df["Close"] - df["VWAP"]
    vwap_std = vwap_diff.rolling(VWAP_PERIOD).std()
    df["VWAP_Z"] = vwap_diff / vwap_std.replace(0, np.nan)

    # Donchian channel (20-day)
    df["DONCH_HIGH"] = df["High"].rolling(DONCH_CHANNEL_PERIOD).max()
    df["DONCH_LOW"] = df["Low"].rolling(DONCH_CHANNEL_PERIOD).min()

    # Volume average (20-day)
    df["VOL_AVG20"] = df["Volume"].rolling(20).mean()

    # Returns for momentum scoring
    df["RET_21D"] = df["Close"].pct_change(MOM_LOOKBACK_1M)
    df["RET_63D"] = df["Close"].pct_change(MOM_LOOKBACK_3M)

    # Volatility for momentum scoring (annualized)
    df["VOL_20D"] = df["Close"].pct_change().rolling(20).std() * np.sqrt(252)
    df["VOL_63D"] = df["Close"].pct_change().rolling(63).std() * np.sqrt(252)

    return df


# =============================================================================
# DATA FETCHING
# =============================================================================

class DataFetcher:
    """Fetches daily bar data from Alpaca (primary) with yfinance fallback."""

    def __init__(self, api_key: str, secret_key: str):
        self._api_key = api_key
        self._secret_key = secret_key
        # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
        from alpaca_client import _AutoRefreshSession
        self._session = _AutoRefreshSession(self._refresh_session_credentials)
        self._session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        })

    def _refresh_session_credentials(self, session) -> None:
        """Edge 123 port (Apr 22 2026): re-pull creds on 401."""
        from jarvis_utils.secrets import get
        new_key = get("Alpaca", "api_key_id",
                      user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        new_secret = get("Alpaca", "secret_key",
                         user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        if not new_key or not new_secret:
            raise EnvironmentError("swing DataFetcher cred refresh: empty creds")
        self._api_key = new_key
        self._secret_key = new_secret
        session.headers["APCA-API-KEY-ID"] = new_key
        session.headers["APCA-API-SECRET-KEY"] = new_secret

    def fetch_daily_bars(self, symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
        """Fetch daily bars from Alpaca IEX feed.

        Returns dict of symbol -> DataFrame with columns: Open, High, Low, Close, Volume.
        Index is DatetimeIndex (tz-naive).
        """
        import requests

        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT23:59:59Z")

        result: Dict[str, List[Dict]] = {s: [] for s in symbols}
        page_token = None

        while True:
            params = {
                "symbols": ",".join(symbols),
                "timeframe": "1Day",
                "limit": 10000,
                "adjustment": "split",
                "feed": "iex",
                "start": start,
                "end": end,
            }
            if page_token:
                params["page_token"] = page_token

            try:
                r = self._session.get(f"{DATA_BASE}/stocks/bars", params=params)
                r.raise_for_status()
                data = r.json()

                for sym in symbols:
                    bars = data.get("bars", {}).get(sym, [])
                    result[sym].extend(bars)

                page_token = data.get("next_page_token")
                if not page_token:
                    break
                time.sleep(0.2)
            except Exception as e:
                logger.error("Alpaca bars fetch failed: %s", e)
                break

        dfs = {}
        for sym, bars in result.items():
            if not bars:
                logger.warning("No Alpaca bars for %s, trying yfinance fallback", sym)
                df = self._yfinance_fallback(sym, days)
                if df is not None and not df.empty:
                    dfs[sym] = df
                continue

            df = pd.DataFrame(bars)
            df.rename(columns={
                "t": "timestamp", "o": "Open", "h": "High",
                "l": "Low", "c": "Close", "v": "Volume",
            }, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.index = df.index.tz_localize(None)
            df.sort_index(inplace=True)
            for col in ["Open", "High", "Low", "Close"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            if "Volume" in df.columns:
                df["Volume"] = df["Volume"].astype(int)
            # Keep only OHLCV
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            dfs[sym] = df

        return dfs

    def _yfinance_fallback(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Fallback to yfinance if Alpaca data is unavailable."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            if df is not None and not df.empty:
                df.index = df.index.tz_localize(None)
                df = df[["Open", "High", "Low", "Close", "Volume"]]
                logger.info("yfinance fallback for %s: %d bars", symbol, len(df))
                return df
        except Exception as e:
            logger.error("yfinance fallback failed for %s: %s", symbol, e)
        return None


# =============================================================================
# ALPACA ORDER MANAGER
# =============================================================================

class AlpacaOrderManager:
    """Manages orders and positions on Alpaca paper trading."""

    def __init__(self, api_key: str, secret_key: str):
        self._api_key = api_key
        self._secret_key = secret_key
        # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
        from alpaca_client import _AutoRefreshSession
        self._session = _AutoRefreshSession(self._refresh_session_credentials)
        self._session.headers.update({
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type": "application/json",
        })

    def _refresh_session_credentials(self, session) -> None:
        """Edge 123 port (Apr 22 2026): re-pull creds on 401."""
        from jarvis_utils.secrets import get
        new_key = get("Alpaca", "api_key_id",
                      user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        new_secret = get("Alpaca", "secret_key",
                         user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        if not new_key or not new_secret:
            raise EnvironmentError("swing OrderManager cred refresh: empty creds")
        self._api_key = new_key
        self._secret_key = new_secret
        session.headers["APCA-API-KEY-ID"] = new_key
        session.headers["APCA-API-SECRET-KEY"] = new_secret

    def get_account(self) -> Dict[str, Any]:
        """Get account details."""
        r = self._session.get(f"{PAPER_BASE}/account")
        r.raise_for_status()
        return r.json()

    def get_equity(self) -> float:
        """Get current portfolio equity."""
        acct = self.get_account()
        return float(acct.get("equity", 0))

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        r = self._session.get(f"{PAPER_BASE}/positions")
        r.raise_for_status()
        return r.json()

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get position for a specific symbol."""
        r = self._session.get(f"{PAPER_BASE}/positions/{symbol}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific order by ID."""
        try:
            r = self._session.get(f"{PAPER_BASE}/orders/{order_id}")
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.error("Failed to get order %s: %s", order_id, e)
            return None

    def get_orders(self, status: str = "open") -> List[Dict[str, Any]]:
        """Get orders by status."""
        r = self._session.get(f"{PAPER_BASE}/orders", params={"status": status, "limit": 100})
        r.raise_for_status()
        return r.json()

    def place_market_order(
        self, symbol: str, qty: int, side: str, time_in_force: str = "day"
    ) -> Optional[Dict[str, Any]]:
        """Place a market order."""
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
        }
        return self._place_order(order_data)

    def place_limit_order(
        self, symbol: str, qty: int, side: str, limit_price: float,
        time_in_force: str = "day"
    ) -> Optional[Dict[str, Any]]:
        """Place a limit order."""
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "limit",
            "time_in_force": time_in_force,
            "limit_price": f"{limit_price:.2f}",
        }
        return self._place_order(order_data)

    def place_stop_order(
        self, symbol: str, qty: int, side: str, stop_price: float,
        time_in_force: str = "gtc"
    ) -> Optional[Dict[str, Any]]:
        """Place a stop order."""
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "stop",
            "time_in_force": time_in_force,
            "stop_price": f"{stop_price:.2f}",
        }
        return self._place_order(order_data)

    def place_oto_order(
        self, symbol: str, qty: int, side: str, stop_price: float,
        time_in_force: str = "day",
    ) -> Optional[Dict[str, Any]]:
        """Place a One-Triggers-Other (OTO) market order with an attached stop loss.

        This sends a single atomic order to Alpaca that combines a market
        entry with a dependent stop-loss leg, avoiding the "potential wash
        trade" rejection that occurs when a separate stop order is placed
        against an existing same-symbol market order.

        The child stop-loss leg defaults to GTC (good-till-cancelled) on
        Alpaca's side.

        Returns the parent order dict on success (the stop-loss child order
        ID is available under the 'legs' key), or None on failure.
        """
        order_data = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
            "order_class": "oto",
            "stop_loss": {
                "stop_price": f"{stop_price:.2f}",
            },
        }
        result = self._place_order(order_data)
        if result:
            legs = result.get("legs") or []
            stop_leg_id = legs[0].get("id", "") if legs else ""
            logger.info(
                "OTO order placed: %s %s %s qty=%s stop=$%.2f | "
                "parent_id=%s stop_leg_id=%s",
                side, symbol, "market", qty, stop_price,
                result.get("id", "?"), stop_leg_id,
            )
        return result

    def close_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Close a position at market."""
        r = self._session.delete(f"{PAPER_BASE}/positions/{symbol}")
        if r.status_code in (200, 204):
            logger.info("Position closed at market: %s", symbol)
            return r.json() if r.text else {}
        logger.warning("Close position failed for %s: %s %s", symbol, r.status_code, r.text)
        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a specific order."""
        r = self._session.delete(f"{PAPER_BASE}/orders/{order_id}")
        return r.status_code in (200, 204)

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        self._session.delete(f"{PAPER_BASE}/orders")

    def replace_stop_order(
        self, order_id: str, new_stop_price: float
    ) -> Optional[Dict[str, Any]]:
        """Replace (update) a stop order with a new stop price."""
        r = self._session.patch(
            f"{PAPER_BASE}/orders/{order_id}",
            json={"stop_price": f"{new_stop_price:.2f}"},
        )
        if r.status_code in (200, 201):
            return r.json()
        logger.warning("Replace stop order failed: %s %s", r.status_code, r.text)
        return None

    def _place_order(self, order_data: Dict[str, Any], retries: int = 3) -> Optional[Dict[str, Any]]:
        """Place an order with retry logic."""
        side = order_data.get("side", "?").upper()
        symbol = order_data.get("symbol", "?")
        qty = order_data.get("qty", "?")
        price_str = order_data.get("limit_price", order_data.get("stop_price", "market"))

        for attempt in range(retries):
            try:
                r = self._session.post(f"{PAPER_BASE}/orders", json=order_data)
                if r.status_code in (200, 201):
                    result = r.json()
                    order_id = result.get("id", "?")
                    logger.info(
                        "ORDER SUBMITTED: %s %s %s @ $%s | order_id=%s",
                        side, qty, symbol, price_str, order_id[:12] if len(str(order_id)) > 12 else order_id,
                    )
                    return result
                elif r.status_code in (422, 403):
                    logger.error(
                        "ORDER REJECTED: %s %s %s @ $%s | HTTP %d: %s",
                        side, qty, symbol, price_str, r.status_code, r.text,
                    )
                    self._alert_failed_order(side, qty, symbol, f"rejected (HTTP {r.status_code})")
                    return None
                else:
                    logger.warning(
                        "ORDER ATTEMPT %d/%d FAILED: %s %s %s | HTTP %d: %s",
                        attempt + 1, retries, side, qty, symbol, r.status_code, r.text,
                    )
            except Exception as e:
                logger.warning("ORDER ATTEMPT %d/%d ERROR: %s %s %s | %s", attempt + 1, retries, side, qty, symbol, e)
            if attempt < retries - 1:
                time.sleep(1.0 * (attempt + 1))

        logger.error("ORDER FAILED: %s %s %s @ $%s | exhausted %d retries", side, qty, symbol, price_str, retries)
        self._alert_failed_order(side, qty, symbol, f"failed after {retries} retries")
        return None

    @staticmethod
    def _alert_failed_order(side: str, qty, symbol: str, reason: str) -> None:
        """Send inbox alert for a failed order."""
        try:
            from jarvis_utils.inbox import send
            send(f"ORDER FAILED: {side} {qty} {symbol} -- {reason}", source="trading-bot")
        except Exception as e:
            logger.error("Failed to send order failure alert: %s", e)


# =============================================================================
# EARNINGS CALENDAR
# =============================================================================

class EarningsCalendar:
    """Manages earnings dates and trading restrictions."""

    def __init__(self, calendar_path: Path = EARNINGS_FILE):
        self._path = calendar_path
        self._calendar: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """Load earnings calendar from JSON file."""
        if self._path.exists():
            try:
                with open(self._path) as f:
                    data = json.load(f)
                self._calendar = {
                    k: v for k, v in data.items()
                    if not k.startswith("_") and isinstance(v, dict)
                }
                logger.info("Loaded earnings calendar: %d entries", len(self._calendar))
            except Exception as e:
                logger.error("Failed to load earnings calendar: %s", e)
        else:
            logger.warning("No earnings calendar at %s", self._path)

    def days_until_earnings(self, symbol: str, from_date: date = None) -> Optional[int]:
        """Return trading days until earnings, or None if no date known."""
        if from_date is None:
            from_date = datetime.now(ET).date()

        entry = self._calendar.get(symbol, {})
        earn_date_str = entry.get("date")
        if not earn_date_str:
            return None

        try:
            earn_date = date.fromisoformat(earn_date_str)
        except (ValueError, TypeError):
            return None

        # Count trading days (approximate: exclude weekends)
        days = 0
        current = from_date
        while current < earn_date:
            current += timedelta(days=1)
            if current.weekday() < 5:
                days += 1
        if current > earn_date:
            return -1 if earn_date < from_date else 0
        return days

    def is_blocked(self, symbol: str, from_date: date = None) -> Tuple[bool, str]:
        """Check if a symbol is blocked due to upcoming earnings."""
        days = self.days_until_earnings(symbol, from_date)
        if days is None:
            return (False, "")
        if days < 0:
            return (False, "earnings already passed")
        if days <= EARNINGS_BLOCK_DAYS:
            entry = self._calendar.get(symbol, {})
            return (True, f"{symbol} earnings in {days} trading days ({entry.get('date', '?')})")
        return (False, "")

    def summary(self) -> Dict[str, Any]:
        """Return calendar summary."""
        today = datetime.now(ET).date()
        blocked = []
        upcoming = []
        for sym, entry in self._calendar.items():
            days = self.days_until_earnings(sym, today)
            if days is not None and days >= 0:
                if days <= EARNINGS_BLOCK_DAYS:
                    blocked.append(f"{sym} ({entry.get('date', '?')}, {days}d)")
                elif days <= 10:
                    upcoming.append(f"{sym} ({entry.get('date', '?')}, {days}d)")
        return {
            "blocked": blocked,
            "upcoming": upcoming,
            "calendar_size": len(self._calendar),
        }


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

@dataclass
class SwingPosition:
    """Tracks an open swing position."""
    ticker: str
    strategy: str
    entry_date: str             # ISO date
    entry_price: float
    shares: int
    stop_price: float
    atr_at_entry: float
    days_held: int = 0
    highest_price: float = 0.0  # For trailing stop
    stop_order_id: Optional[str] = None
    entry_order_id: Optional[str] = None
    entry_order_status: str = "filled"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SwingPosition":
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class PendingOrder:
    """Tracks a pending entry order that has not yet filled."""
    ticker: str
    strategy: str
    entry_order_id: str
    entry_date: str
    entry_price: float
    shares: int
    stop_price: float
    atr_at_entry: float

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PendingOrder":
        valid_keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class BotState:
    """Persistent state for the multi-strategy swing bot."""
    # Positions and orders
    positions: List[SwingPosition] = field(default_factory=list)
    pending_orders: List[PendingOrder] = field(default_factory=list)

    # Per-strategy capital tracking (realized P&L adjustment)
    strategy_pnl: Dict[str, float] = field(default_factory=lambda: {
        "momentum_rotation": 0.0,
        "vwap_mean_reversion": 0.0,
        "sector_relative_strength": 0.0,
        "donchian_breakout": 0.0,
        "rsi2_mean_reversion": 0.0,
    })

    # Momentum rotation state
    momentum_last_rebalance_date: str = ""
    momentum_trading_days_since_rebal: int = 0

    # Drawdown tracking
    high_water_mark: float = 0.0
    circuit_breaker_active: bool = False
    circuit_breaker_days_remaining: int = 0
    drawdown_reduction_active: bool = False

    # Trade statistics
    total_trades: int = 0
    total_wins: int = 0
    total_losses: int = 0
    daily_pnl: float = 0.0
    last_run_date: str = ""
    last_pnl_reset_date: str = ""

    def save(self, path: Path = STATE_FILE):
        """Save state to JSON file using atomic temp-file-then-rename pattern."""
        def _json_default(obj):
            if hasattr(obj, 'item'):
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
            "pending_orders": [p.to_dict() for p in self.pending_orders],
            "strategy_pnl": {k: float(v) for k, v in self.strategy_pnl.items()},
            "momentum_last_rebalance_date": self.momentum_last_rebalance_date,
            "momentum_trading_days_since_rebal": int(self.momentum_trading_days_since_rebal),
            "high_water_mark": float(self.high_water_mark),
            "circuit_breaker_active": bool(self.circuit_breaker_active),
            "circuit_breaker_days_remaining": int(self.circuit_breaker_days_remaining),
            "drawdown_reduction_active": bool(self.drawdown_reduction_active),
            "total_trades": int(self.total_trades),
            "total_wins": int(self.total_wins),
            "total_losses": int(self.total_losses),
            "daily_pnl": float(self.daily_pnl),
            "last_run_date": self.last_run_date,
            "last_pnl_reset_date": self.last_pnl_reset_date,
        }
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=_json_default)
        tmp.rename(path)
        logger.info("State saved to %s", path)

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> "BotState":
        """Load state from JSON file."""
        if not path.exists():
            logger.info("No state file found, starting fresh")
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            state = cls()
            state.positions = [SwingPosition.from_dict(p) for p in data.get("positions", [])]
            state.pending_orders = [PendingOrder.from_dict(p) for p in data.get("pending_orders", [])]
            state.strategy_pnl = data.get("strategy_pnl", state.strategy_pnl)
            state.momentum_last_rebalance_date = data.get("momentum_last_rebalance_date", "")
            state.momentum_trading_days_since_rebal = data.get("momentum_trading_days_since_rebal", 0)
            state.high_water_mark = data.get("high_water_mark", 0.0)
            state.circuit_breaker_active = data.get("circuit_breaker_active", False)
            state.circuit_breaker_days_remaining = data.get("circuit_breaker_days_remaining", 0)
            state.drawdown_reduction_active = data.get("drawdown_reduction_active", False)
            state.total_trades = data.get("total_trades", 0)
            state.total_wins = data.get("total_wins", 0)
            state.total_losses = data.get("total_losses", 0)
            state.daily_pnl = data.get("daily_pnl", 0.0)
            state.last_run_date = data.get("last_run_date", "")
            state.last_pnl_reset_date = data.get("last_pnl_reset_date", "")
            logger.info(
                "State loaded: %d positions, %d pending, %d total trades, HWM=$%.2f",
                len(state.positions), len(state.pending_orders),
                state.total_trades, state.high_water_mark,
            )
            return state
        except Exception as e:
            logger.error("Failed to load state: %s", e)
            return cls()


# =============================================================================
# SIGNAL DATACLASS
# =============================================================================

@dataclass
class SwingSignal:
    """A detected swing trading signal."""
    ticker: str
    strategy: str
    direction: str = "buy"
    entry_price: float = 0.0
    stop_price: float = 0.0
    shares: int = 0
    atr: float = 0.0
    rationale: str = ""
    blocked: bool = False
    block_reason: str = ""
    priority: float = 0.0  # Higher = more desirable
    rsi2_value: float = 0.0  # RSI(2) at signal time (for averaging-down logic)


# =============================================================================
# STRATEGY ENGINE
# =============================================================================

class StrategyEngine:
    """Generates signals for all 5 strategies."""

    def __init__(
        self,
        equity: float,
        state: BotState,
        earnings: EarningsCalendar,
        data: Dict[str, pd.DataFrame],
        macro_output: Optional[Any] = None,
    ):
        self.equity = equity
        self.state = state
        self.earnings = earnings
        self.data = data
        self.macro_output = macro_output

        # Compute macro-based position size multiplier
        # Edge 85 (cycle 94) — gated on MACRO_SIZE_MULT_ENABLED (combined_config).
        # When False (default), pin to 1.0; multiplier values are statistically
        # net-negative on SPY (Edge 85) AND metals (Edge 97).
        self.macro_size_mult = 1.0
        if macro_output is not None and MACRO_SIZE_MULT_ENABLED:
            try:
                self.macro_size_mult = float(getattr(macro_output, 'position_size_multiplier', 1.0))
                self.macro_size_mult = max(0.5, min(self.macro_size_mult, 1.5))
            except Exception:
                self.macro_size_mult = 1.0

        # Drawdown reduction
        self.dd_size_mult = 0.5 if state.drawdown_reduction_active else 1.0

    # -----------------------------------------------------------------
    # Utility: position counts and capital
    # -----------------------------------------------------------------

    def _strategy_positions(self, strategy: str) -> List[SwingPosition]:
        """Get all positions for a specific strategy."""
        return [p for p in self.state.positions if p.strategy == strategy]

    def _strategy_pending(self, strategy: str) -> List[PendingOrder]:
        """Get all pending orders for a specific strategy."""
        return [p for p in self.state.pending_orders if p.strategy == strategy]

    def _strategy_position_count(self, strategy: str) -> int:
        """Count active + pending positions for a strategy."""
        active = len(self._strategy_positions(strategy))
        pending = len(self._strategy_pending(strategy))
        return active + pending

    def _strategy_capital(self, strategy: str) -> float:
        """Compute available capital for a strategy (allocation * equity + realized PnL)."""
        alloc = STRATEGY_ALLOCATIONS.get(strategy, 0.0)
        base_capital = self.equity * alloc
        # Add/subtract realized PnL for the strategy
        realized = self.state.strategy_pnl.get(strategy, 0.0)
        return max(base_capital + realized, 0.0)

    def _strategy_invested(self, strategy: str) -> float:
        """Compute capital currently invested in a strategy."""
        positions = self._strategy_positions(strategy)
        return sum(p.shares * p.entry_price for p in positions)

    def _strategy_available_capital(self, strategy: str) -> float:
        """Compute available (uninvested) capital for a strategy."""
        total = self._strategy_capital(strategy)
        invested = self._strategy_invested(strategy)
        return max(total - invested, 0.0)

    def _sector_position_count(self, sector: str) -> int:
        """Count positions in a given sector across ALL strategies."""
        count = 0
        for pos in self.state.positions:
            if TICKER_SECTOR.get(pos.ticker) == sector:
                count += 1
        for pend in self.state.pending_orders:
            if TICKER_SECTOR.get(pend.ticker) == sector:
                count += 1
        return count

    def _total_exposure(self) -> float:
        """Compute total portfolio exposure as fraction of equity."""
        if self.equity <= 0:
            return 0.0
        total_invested = sum(p.shares * p.entry_price for p in self.state.positions)
        total_pending = sum(p.shares * p.entry_price for p in self.state.pending_orders)
        return (total_invested + total_pending) / self.equity

    def _has_position(self, ticker: str) -> bool:
        """Check if we already hold or have pending order for a ticker in ANY strategy."""
        for p in self.state.positions:
            if p.ticker == ticker:
                return True
        for p in self.state.pending_orders:
            if p.ticker == ticker:
                return True
        return False

    def _has_strategy_position(self, ticker: str, strategy: str) -> bool:
        """Check if we hold or have pending for ticker in a specific strategy."""
        for p in self.state.positions:
            if p.ticker == ticker and p.strategy == strategy:
                return True
        for p in self.state.pending_orders:
            if p.ticker == ticker and p.strategy == strategy:
                return True
        return False

    def _compute_shares(
        self, strategy: str, price: float, atr: float, stop_atr_mult: float,
        max_positions: int
    ) -> int:
        """Compute position size in 100-share lots, capped by risk rules.

        All swing entries use MIN_LOT_SIZE (100) share blocks so that
        every position is eligible for covered-call overlays.  If even
        one lot (100 shares) exceeds the per-position or exposure cap,
        the ticker is skipped entirely.
        """
        if price <= 0 or atr <= 0:
            return 0

        # Available capital for this strategy
        available = self._strategy_available_capital(strategy)
        if available <= 0:
            return 0

        # Equal-weight: divide available capital by remaining open slots
        current_count = self._strategy_position_count(strategy)
        remaining_slots = max_positions - current_count
        if remaining_slots <= 0:
            return 0

        # For momentum, we want to be fully invested across positions
        per_position = available / remaining_slots

        # Apply macro size multiplier
        per_position *= self.macro_size_mult

        # Apply drawdown reduction
        per_position *= self.dd_size_mult

        # Cap at 5% of total portfolio
        max_by_portfolio = self.equity * MAX_SINGLE_POSITION_PCT
        per_position = min(per_position, max_by_portfolio)

        # Check total exposure cap
        current_exposure = self._total_exposure()
        remaining_exposure = max(MAX_PORTFOLIO_EXPOSURE - current_exposure, 0.0)
        max_by_exposure = remaining_exposure * self.equity
        per_position = min(per_position, max_by_exposure)

        # Round to 100-share lots for covered-call eligibility
        lot_size = MIN_LOT_SIZE  # 100
        min_lot_cost = lot_size * price

        # If even 1 lot exceeds the position cap, skip this ticker
        if min_lot_cost > per_position:
            logger.info(
                "LOT_SIZE_SKIP: %s -- 1 lot (%d shares) costs $%.0f, "
                "exceeds position cap $%.0f (%.1f%% of $%.0f equity)",
                strategy, lot_size, min_lot_cost, per_position,
                (min_lot_cost / self.equity) * 100, self.equity,
            )
            return 0

        # Compute number of lots, then total shares
        lots = max(1, round(per_position / (price * lot_size)))
        shares = lots * lot_size

        # Final cap: ensure we don't exceed per_position dollar amount
        while shares * price > per_position and shares > lot_size:
            shares -= lot_size

        # Safety: ensure we don't exceed max portfolio position pct
        max_shares_by_portfolio = int(max_by_portfolio / price)
        # Round down to nearest lot
        max_shares_by_portfolio = (max_shares_by_portfolio // lot_size) * lot_size
        if max_shares_by_portfolio < lot_size:
            logger.info(
                "LOT_SIZE_SKIP: %s -- max portfolio position $%.0f "
                "cannot fit 1 lot of %d shares @ $%.2f ($%.0f)",
                strategy, max_by_portfolio, lot_size, price, min_lot_cost,
            )
            return 0
        shares = min(shares, max_shares_by_portfolio)

        return max(shares, 0)

    def _can_open(self, ticker: str, strategy: str) -> Tuple[bool, str]:
        """Check if a new position can be opened."""
        # Max positions for this strategy
        max_pos = MAX_POSITIONS.get(strategy, 5)
        if self._strategy_position_count(strategy) >= max_pos:
            return (False, f"{strategy}: max positions ({max_pos}) reached")

        # Already have position in this ticker for this strategy
        if self._has_strategy_position(ticker, strategy):
            return (False, f"already have {ticker} in {strategy}")

        # Sector limit
        sector = TICKER_SECTOR.get(ticker, "unknown")
        if self._sector_position_count(sector) >= MAX_SECTOR_POSITIONS:
            return (False, f"max {MAX_SECTOR_POSITIONS} positions in {sector} sector reached")

        # Total exposure cap
        if self._total_exposure() >= MAX_PORTFOLIO_EXPOSURE:
            return (False, "total portfolio exposure at 100%")

        # Earnings block
        earn_blocked, earn_reason = self.earnings.is_blocked(ticker)
        if earn_blocked:
            return (False, f"EARNINGS: {earn_reason}")

        return (True, "")

    # -----------------------------------------------------------------
    # Strategy 1: Momentum Rotation
    # -----------------------------------------------------------------

    def generate_momentum_signals(self) -> List[SwingSignal]:
        """Momentum Rotation: rank by vol-adjusted momentum, hold top 5.

        This is an always-in rotational strategy. On rebalance day,
        we sell tickers no longer in top-5 and buy new top-5 entries.
        """
        strategy = "momentum_rotation"
        if MAX_POSITIONS.get(strategy, 0) == 0:
            return []

        signals = []

        # Check if it is rebalance day
        self.state.momentum_trading_days_since_rebal += 1
        needs_rebalance = self.state.momentum_trading_days_since_rebal >= MOM_REBALANCE_DAYS

        if not needs_rebalance:
            logger.info("Momentum: %d/%d days since last rebalance, not rebalancing",
                        self.state.momentum_trading_days_since_rebal, MOM_REBALANCE_DAYS)
            return signals

        logger.info("Momentum: REBALANCE DAY (day %d)", self.state.momentum_trading_days_since_rebal)

        # Score all tickers
        scores = {}
        for ticker in TRADING_TICKERS:
            df = self.data.get(ticker)
            if df is None or len(df) < MOM_LOOKBACK_3M + 5:
                continue

            row = df.iloc[-1]
            ret_21d = row.get("RET_21D")
            ret_63d = row.get("RET_63D")
            vol_20d = row.get("VOL_20D")
            vol_63d = row.get("VOL_63D")

            if any(pd.isna(v) for v in [ret_21d, ret_63d, vol_20d, vol_63d]):
                continue
            if vol_20d <= 0 or vol_63d <= 0:
                continue

            # Absolute momentum filter: skip tickers with negative 63-day return
            if ret_63d <= 0:
                logger.debug("Momentum: %s skipped, negative 63d return (%.2f%%)", ticker, ret_63d * 100)
                continue

            # Vol-adjusted momentum score
            score_1m = ret_21d / vol_20d
            score_3m = ret_63d / vol_63d
            composite = MOM_WEIGHT_1M * score_1m + MOM_WEIGHT_3M * score_3m
            scores[ticker] = composite

        if not scores:
            logger.warning("Momentum: no tickers passed absolute momentum filter")
            return signals

        # Rank and select top N
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        target_tickers = [t for t, _ in ranked[:MOM_NUM_POSITIONS]]

        logger.info("Momentum rankings: %s",
                     ", ".join(f"{t}={s:.3f}" for t, s in ranked[:10]))
        logger.info("Momentum target holdings: %s", target_tickers)

        # Determine sells (current holdings not in target)
        current_positions = self._strategy_positions(strategy)
        current_tickers = {p.ticker for p in current_positions}

        for pos in current_positions:
            if pos.ticker not in target_tickers:
                signals.append(SwingSignal(
                    ticker=pos.ticker,
                    strategy=strategy,
                    direction="sell",
                    entry_price=pos.entry_price,
                    shares=pos.shares,
                    rationale=f"Momentum rotation: {pos.ticker} dropped out of top-{MOM_NUM_POSITIONS}",
                    priority=10.0,  # exits first
                ))

        # Determine buys (target tickers not currently held)
        for ticker in target_tickers:
            if ticker in current_tickers:
                continue  # Already holding

            df = self.data.get(ticker)
            if df is None or len(df) < 5:
                continue

            row = df.iloc[-1]
            price = float(row["Close"])
            atr = float(row.get("ATR14", 0))
            if price <= 0 or atr <= 0:
                continue

            shares = self._compute_shares(strategy, price, atr, 0, MOM_NUM_POSITIONS)
            if shares <= 0:
                continue

            can_open, reason = self._can_open(ticker, strategy)

            signal = SwingSignal(
                ticker=ticker,
                strategy=strategy,
                direction="buy",
                entry_price=round(price, 2),
                stop_price=0.0,  # No stops for momentum rotation
                shares=shares,
                atr=atr,
                rationale=(
                    f"Momentum rotation: {ticker} in top-{MOM_NUM_POSITIONS} "
                    f"(score={scores.get(ticker, 0):.3f}). "
                    f"Shares={shares}, Value=${shares * price:,.0f}"
                ),
                priority=scores.get(ticker, 0),
            )

            if not can_open:
                signal.blocked = True
                signal.block_reason = reason

            signals.append(signal)

        # Reset rebalance counter
        self.state.momentum_trading_days_since_rebal = 0
        self.state.momentum_last_rebalance_date = datetime.now(ET).strftime("%Y-%m-%d")

        return signals

    # -----------------------------------------------------------------
    # Strategy 2: VWAP Mean Reversion
    # -----------------------------------------------------------------

    def generate_vwap_signals(self) -> List[SwingSignal]:
        """VWAP Mean Reversion: buy when z-score < -2.0, exit at z-score > 0."""
        strategy = "vwap_mean_reversion"
        if MAX_POSITIONS.get(strategy, 0) == 0:
            return []

        signals = []

        for ticker in TRADING_TICKERS:
            df = self.data.get(ticker)
            if df is None or len(df) < VWAP_PERIOD + 10:
                continue

            # Check for existing position - generate exit signals
            if self._has_strategy_position(ticker, strategy):
                continue

            row = df.iloc[-1]
            vwap_z = row.get("VWAP_Z")
            close = row.get("Close")
            atr = row.get("ATR14")

            if any(pd.isna(v) for v in [vwap_z, close, atr]) or atr <= 0:
                continue

            vwap_z = float(vwap_z)
            close = float(close)
            atr = float(atr)

            # Entry: z-score < -2.0
            if vwap_z >= VWAP_ENTRY_Z:
                continue

            shares = self._compute_shares(strategy, close, atr, VWAP_STOP_ATR_MULT,
                                          MAX_POSITIONS[strategy])
            if shares <= 0:
                continue

            # NO stop loss for VWAP entries -- 5-SMA exit only (David's design).
            # stop_price=0 tells combined_runner to skip the OTO stop order.
            stop_price = 0.0
            can_open, reason = self._can_open(ticker, strategy)

            signal = SwingSignal(
                ticker=ticker,
                strategy=strategy,
                direction="buy",
                entry_price=round(close, 2),
                stop_price=stop_price,
                shares=shares,
                atr=atr,
                rationale=(
                    f"VWAP MR: z-score={vwap_z:.2f} < {VWAP_ENTRY_Z}. "
                    f"Entry=${close:.2f}, no stop (5-SMA exit). "
                    f"Shares={shares}, Value=${shares * close:,.0f}"
                ),
                priority=abs(vwap_z),  # More oversold = higher priority
            )

            if not can_open:
                signal.blocked = True
                signal.block_reason = reason

            signals.append(signal)

        return signals

    # -----------------------------------------------------------------
    # Strategy 3: Sector Relative Strength
    # -----------------------------------------------------------------

    def generate_sector_rs_signals(self) -> List[SwingSignal]:
        """Sector RS: buy when stock outperforms sector ETF by >2%."""
        strategy = "sector_relative_strength"
        if MAX_POSITIONS.get(strategy, 0) == 0:
            return []

        signals = []

        for ticker in TRADING_TICKERS:
            if self._has_strategy_position(ticker, strategy):
                continue

            df = self.data.get(ticker)
            if df is None or len(df) < SRS_RS_PERIOD + SRS_RS_MA_PERIOD + 5:
                continue

            sector = TICKER_SECTOR.get(ticker, "unknown")
            sector_etf = SECTOR_ETF_MAP.get(sector)
            if not sector_etf:
                continue

            etf_df = self.data.get(sector_etf)
            if etf_df is None or len(etf_df) < SRS_RS_PERIOD + SRS_RS_MA_PERIOD + 5:
                continue

            # Align indices
            common_idx = df.index.intersection(etf_df.index)
            if len(common_idx) < SRS_RS_PERIOD + SRS_RS_MA_PERIOD:
                continue

            stock_close = df.loc[common_idx, "Close"]
            etf_close = etf_df.loc[common_idx, "Close"]

            # 20-day relative return
            stock_ret = stock_close.pct_change(SRS_RS_PERIOD)
            etf_ret = etf_close.pct_change(SRS_RS_PERIOD)
            relative_return = stock_ret - etf_ret

            # 5-day MA of relative strength
            rs_ma = relative_return.rolling(SRS_RS_MA_PERIOD).mean()

            if len(relative_return) < 2 or len(rs_ma) < 2:
                continue

            current_rs = relative_return.iloc[-1]
            current_rs_ma = rs_ma.iloc[-1]
            prev_rs_ma = rs_ma.iloc[-2]

            if pd.isna(current_rs) or pd.isna(current_rs_ma) or pd.isna(prev_rs_ma):
                continue

            # Entry: outperforms by >2% AND 5-day RS MA is rising
            if current_rs <= SRS_ENTRY_THRESHOLD:
                continue
            if current_rs_ma <= prev_rs_ma:
                continue

            row = df.iloc[-1]
            close = float(row["Close"])
            atr = float(row.get("ATR14", 0))
            if close <= 0 or atr <= 0:
                continue

            shares = self._compute_shares(strategy, close, atr, SRS_STOP_ATR_MULT,
                                          MAX_POSITIONS[strategy])
            if shares <= 0:
                continue

            stop_price = round(close - SRS_STOP_ATR_MULT * atr, 2)
            can_open, reason = self._can_open(ticker, strategy)

            signal = SwingSignal(
                ticker=ticker,
                strategy=strategy,
                direction="buy",
                entry_price=round(close, 2),
                stop_price=stop_price,
                shares=shares,
                atr=atr,
                rationale=(
                    f"Sector RS: {ticker} outperforms {sector_etf} by {current_rs * 100:.1f}% "
                    f"(>{SRS_ENTRY_THRESHOLD * 100:.0f}%), RS MA rising. "
                    f"Entry=${close:.2f}, Stop=${stop_price:.2f}. "
                    f"Shares={shares}"
                ),
                priority=current_rs,
            )

            if not can_open:
                signal.blocked = True
                signal.block_reason = reason

            signals.append(signal)

        return signals

    # -----------------------------------------------------------------
    # Strategy 4: Donchian Breakout
    # -----------------------------------------------------------------

    def generate_donchian_signals(self) -> List[SwingSignal]:
        """Donchian Breakout: buy on 20-day high + volume confirmation."""
        strategy = "donchian_breakout"
        if MAX_POSITIONS.get(strategy, 0) == 0:
            return []

        signals = []

        for ticker in TRADING_TICKERS:
            if self._has_strategy_position(ticker, strategy):
                continue

            df = self.data.get(ticker)
            if df is None or len(df) < DONCH_CHANNEL_PERIOD + 10:
                continue

            row = df.iloc[-1]
            close = row.get("Close")
            atr = row.get("ATR14")
            volume = row.get("Volume")
            vol_avg = row.get("VOL_AVG20")

            # Need previous day's Donchian high (not including today)
            if len(df) < DONCH_CHANNEL_PERIOD + 2:
                continue

            # Compute 20-day high excluding today
            prev_donch_high = df["High"].iloc[-(DONCH_CHANNEL_PERIOD + 1):-1].max()

            if any(pd.isna(v) for v in [close, atr, volume, vol_avg, prev_donch_high]):
                continue
            if atr <= 0 or vol_avg <= 0:
                continue

            close = float(close)
            atr = float(atr)
            volume = float(volume)
            vol_avg = float(vol_avg)
            prev_donch_high = float(prev_donch_high)

            # Entry: close > 20-day high AND volume > 1.5x average
            if close <= prev_donch_high:
                continue
            if volume <= DONCH_VOLUME_MULT * vol_avg:
                continue

            shares = self._compute_shares(strategy, close, atr, DONCH_TRAIL_ATR_MULT,
                                          MAX_POSITIONS[strategy])
            if shares <= 0:
                continue

            stop_price = round(close - DONCH_TRAIL_ATR_MULT * atr, 2)
            can_open, reason = self._can_open(ticker, strategy)

            vol_ratio = volume / vol_avg

            signal = SwingSignal(
                ticker=ticker,
                strategy=strategy,
                direction="buy",
                entry_price=round(close, 2),
                stop_price=stop_price,
                shares=shares,
                atr=atr,
                rationale=(
                    f"Donchian Breakout: {ticker} closed ${close:.2f} > 20d high ${prev_donch_high:.2f}, "
                    f"vol {vol_ratio:.1f}x avg. "
                    f"Stop=${stop_price:.2f} ({DONCH_TRAIL_ATR_MULT}x ATR). "
                    f"Shares={shares}"
                ),
                priority=vol_ratio,
            )

            if not can_open:
                signal.blocked = True
                signal.block_reason = reason

            signals.append(signal)

        return signals

    # -----------------------------------------------------------------
    # Strategy 5: RSI(2) Mean Reversion
    # -----------------------------------------------------------------

    def generate_rsi2_signals(self) -> List[SwingSignal]:
        """RSI(2) Mean Reversion: buy dips in uptrends with macro filters."""
        signals = []
        strategy = "rsi2_mean_reversion"

        # Fetch macro regime data (Apr 8 research)
        try:
            import yfinance as yf
            from combined_config import DOLLAR_FILTER_ENABLED, DOLLAR_ETF, DOLLAR_SMA_PERIOD
            from combined_config import VIX_FILTER_ENABLED
            if DOLLAR_FILTER_ENABLED:
                uup = yf.Ticker(DOLLAR_ETF).history(period=f"{DOLLAR_SMA_PERIOD + 10}d")
                if len(uup) >= DOLLAR_SMA_PERIOD:
                    uc = uup['Close'].values
                    self._dollar_strong = uc[-1] > uc[-DOLLAR_SMA_PERIOD:].mean()
                else:
                    self._dollar_strong = True  # default to allow if insufficient data
            if VIX_FILTER_ENABLED:
                vix = yf.Ticker("^VIX").history(period="5d")
                self._vix_level = float(vix['Close'].iloc[-1]) if len(vix) > 0 else 15.0
        except Exception as e:
            logger.warning(f"Macro filter fetch failed: {e}. Proceeding without filters.")
            self._dollar_strong = True
            self._vix_level = 15.0

        for ticker in TRADING_TICKERS:
            already_held = self._has_strategy_position(ticker, strategy)
            if already_held:
                # Still generate signal if RSI(2) is extreme (averaging-down path).
                # We read the data first to check RSI before deciding to skip.
                pass

            df = self.data.get(ticker)
            if df is None or len(df) < RSI2_MA_PERIOD + 10:
                continue

            row = df.iloc[-1]
            close = row.get("Close")
            rsi2 = row.get("RSI2")
            sma200 = row.get("SMA200")
            atr = row.get("ATR14")

            if any(pd.isna(v) for v in [close, rsi2, sma200, atr]):
                continue
            if atr <= 0:
                continue

            close = float(close)
            rsi2 = float(rsi2)
            sma200 = float(sma200)
            atr = float(atr)

            # Entry: RSI(2) < threshold AND price > 200-day MA
            if rsi2 >= RSI2_ENTRY_THRESHOLD:
                continue
            if close <= sma200:
                continue

            # Macro regime filters (Apr 8 backtest: Sharpe 3.0-3.4 with these)
            from combined_config import DOLLAR_FILTER_ENABLED, VIX_FILTER_ENABLED, VIX_MAX_ENTRY
            if DOLLAR_FILTER_ENABLED and hasattr(self, '_dollar_strong') and not self._dollar_strong:
                logger.debug(f"SKIP {ticker}: dollar weak (UUP below 200 SMA)")
                continue
            if VIX_FILTER_ENABLED and hasattr(self, '_vix_level') and self._vix_level > VIX_MAX_ENTRY:
                logger.debug(f"SKIP {ticker}: VIX {self._vix_level:.1f} > {VIX_MAX_ENTRY}")
                continue

            # If already held, only allow averaging-down signals (RSI2 < 5)
            if already_held and rsi2 >= AVERAGING_DOWN_RSI2_THRESHOLD:
                continue

            # Use a nominal ATR multiplier for position sizing only (no stop loss)
            shares = self._compute_shares(strategy, close, atr, 3.0,
                                          MAX_POSITIONS[strategy])
            if shares <= 0:
                continue

            # NO stop loss -- exit via 5-day SMA crossover or max hold
            sma5 = row.get("SMA5")
            sma5_val = float(sma5) if sma5 is not None and not pd.isna(sma5) else 0.0
            can_open, reason = self._can_open(ticker, strategy)

            signal = SwingSignal(
                ticker=ticker,
                strategy=strategy,
                direction="buy",
                entry_price=round(close, 2),
                stop_price=0.0,  # NO stop loss (backtest proved stops destroy value)
                shares=shares,
                atr=atr,
                rationale=(
                    f"RSI(2) MR: {ticker} RSI(2)={rsi2:.1f} < {RSI2_ENTRY_THRESHOLD}, "
                    f"price ${close:.2f} > SMA200 ${sma200:.2f} (uptrend). "
                    f"Exit: price > SMA5 ${sma5_val:.2f} or max {RSI2_MAX_HOLD_DAYS}d hold. "
                    f"NO stop loss. Shares={shares}"
                ),
                priority=RSI2_ENTRY_THRESHOLD - rsi2,  # Lower RSI = higher priority
                rsi2_value=rsi2,
            )

            if not can_open:
                # Don't block averaging-down candidates — let combined_runner decide
                if already_held and rsi2 < AVERAGING_DOWN_RSI2_THRESHOLD:
                    signal.block_reason = f"avg_down_candidate: {reason}"
                    # NOT blocked — will be evaluated by combined_runner
                else:
                    signal.blocked = True
                    signal.block_reason = reason

            signals.append(signal)

        return signals


# =============================================================================
# EXIT MANAGER
# =============================================================================

class ExitManager:
    """Determines exit signals for open positions across all strategies."""

    def check_exits(
        self,
        pos: SwingPosition,
        df: pd.DataFrame,
        current_price: float,
        data: Dict[str, pd.DataFrame],
    ) -> Optional[Tuple[str, float]]:
        """Route to strategy-specific exit logic.

        Returns (reason, exit_price) or None.
        """
        if pos.strategy == "momentum_rotation":
            return self._check_momentum_exits(pos, df, current_price)
        elif pos.strategy == "vwap_mean_reversion":
            return self._check_vwap_exits(pos, df, current_price)
        elif pos.strategy == "sector_relative_strength":
            return self._check_sector_rs_exits(pos, df, current_price, data)
        elif pos.strategy == "donchian_breakout":
            return self._check_donchian_exits(pos, df, current_price)
        elif pos.strategy == "rsi2_mean_reversion":
            return self._check_rsi2_exits(pos, df, current_price)
        return None

    def _check_momentum_exits(
        self, pos: SwingPosition, df: pd.DataFrame, current_price: float
    ) -> Optional[Tuple[str, float]]:
        """Momentum: no exit logic here -- exits handled by rotation signals."""
        # Momentum rotation exits are handled by the strategy engine generating
        # sell signals when a ticker drops out of the top-5 ranking.
        return None

    def _check_vwap_exits(
        self, pos: SwingPosition, df: pd.DataFrame, current_price: float
    ) -> Optional[Tuple[str, float]]:
        """VWAP MR exits: z-score > 0 (mean reverted) or stop hit."""
        if len(df) < 1:
            return None

        row = df.iloc[-1]

        # 1. Primary exit: z-score > 0 (reverted to VWAP)
        vwap_z = row.get("VWAP_Z")
        if vwap_z is not None and not pd.isna(vwap_z):
            if float(vwap_z) > VWAP_EXIT_Z:
                return ("vwap_mean_reverted", current_price)

        # 2. Stop loss DISABLED -- 5-SMA exit only (David's design, Mar 18).
        # Previously used VWAP_STOP_ATR_MULT * ATR; now stop_price=0 for all
        # VWAP entries.  Kept as dead code guard for old positions with stop > 0.
        if pos.stop_price > 0 and current_price <= pos.stop_price:
            return ("vwap_stop_loss", current_price)

        return None

    def _check_sector_rs_exits(
        self, pos: SwingPosition, df: pd.DataFrame, current_price: float,
        data: Dict[str, pd.DataFrame],
    ) -> Optional[Tuple[str, float]]:
        """Sector RS exits: underperformance reversal, time stop, or hard stop."""
        if len(df) < SRS_RS_PERIOD + 5:
            return None

        # 1. Stop loss
        if current_price <= pos.stop_price:
            return ("srs_stop_loss", current_price)

        # 2. Max hold days
        if pos.days_held >= SRS_MAX_HOLD_DAYS:
            return ("srs_time_stop", current_price)

        # 3. Relative underperformance > -2%
        sector = TICKER_SECTOR.get(pos.ticker, "unknown")
        sector_etf = SECTOR_ETF_MAP.get(sector)
        if sector_etf:
            etf_df = data.get(sector_etf)
            if etf_df is not None and len(etf_df) >= SRS_RS_PERIOD:
                common_idx = df.index.intersection(etf_df.index)
                if len(common_idx) >= SRS_RS_PERIOD:
                    stock_ret = df.loc[common_idx, "Close"].pct_change(SRS_RS_PERIOD).iloc[-1]
                    etf_ret = etf_df.loc[common_idx, "Close"].pct_change(SRS_RS_PERIOD).iloc[-1]
                    if not pd.isna(stock_ret) and not pd.isna(etf_ret):
                        relative = stock_ret - etf_ret
                        if relative < SRS_EXIT_THRESHOLD:
                            return ("srs_underperformance", current_price)

        return None

    def _check_donchian_exits(
        self, pos: SwingPosition, df: pd.DataFrame, current_price: float
    ) -> Optional[Tuple[str, float]]:
        """Donchian exits: trailing stop at 2x ATR from highest close since entry."""
        # Update highest price
        if current_price > pos.highest_price:
            pos.highest_price = current_price

        # Trailing stop: 2x ATR from highest close
        trailing_stop = pos.highest_price - DONCH_TRAIL_ATR_MULT * pos.atr_at_entry
        if trailing_stop > pos.stop_price:
            old_stop = pos.stop_price
            pos.stop_price = round(trailing_stop, 2)
            if pos.stop_price > old_stop:
                logger.info(
                    "%s (donchian): trailing stop raised to $%.2f (high=$%.2f, was $%.2f)",
                    pos.ticker, pos.stop_price, pos.highest_price, old_stop,
                )

        if current_price <= pos.stop_price:
            return ("donchian_trailing_stop", current_price)

        return None

    def _check_rsi2_exits(
        self, pos: SwingPosition, df: pd.DataFrame, current_price: float
    ) -> Optional[Tuple[str, float]]:
        """RSI(2) exits: per-ticker RSI2 threshold (Apr 9 research: beats SMA5 by 2-3x Sharpe).

        NO stop loss -- backtest proved stops destroy value on mean-reverting stocks.
        """
        if len(df) < 1:
            return None

        row = df.iloc[-1]

        # 1. Primary exit: RSI2 crosses above per-ticker threshold
        # Default 70, KGC uses 90 (Sharpe 3.93 vs 2.37)
        import json, os
        try:
            cfg_path = os.path.join(os.path.dirname(__file__), "knowledge", "ticker_configs.json")
            with open(cfg_path) as f:
                ticker_cfg = json.load(f)
            exit_threshold = ticker_cfg.get(pos.ticker, {}).get("exit_rsi2", 70)
        except Exception:
            exit_threshold = 70

        rsi2_val = row.get("RSI2")
        if rsi2_val is not None and not pd.isna(rsi2_val) and float(rsi2_val) > exit_threshold:
            return (f"rsi2_exit_{exit_threshold}", current_price)

        # 2. Max hold days (safety valve)
        if pos.days_held >= RSI2_MAX_HOLD_DAYS:
            return ("rsi2_max_hold", current_price)

        # NO stop loss -- intentionally omitted

        return None


# =============================================================================
# MAIN SWING BOT
# =============================================================================

class SwingBot:
    """Main multi-strategy swing trading bot orchestrator."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

        # Load credentials
        from jarvis_utils.secrets import get
        self.api_key = get("Alpaca", "api_key_id", user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        self.api_secret = get("Alpaca", "secret_key", user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        if not self.api_key or not self.api_secret:
            raise EnvironmentError("Alpaca API credentials not found in Settings Portal")

        # Initialize components
        self.fetcher = DataFetcher(self.api_key, self.api_secret)
        self.order_mgr = AlpacaOrderManager(self.api_key, self.api_secret)
        self.earnings = EarningsCalendar()
        self.state = BotState.load()
        self.exit_mgr = ExitManager()

        # Data storage
        self.data: Dict[str, pd.DataFrame] = {}
        self.equity = 0.0

        # Macro regime output
        self.macro_output = None

    def run(self) -> str:
        """Execute the full daily multi-strategy swing trading cycle.

        Returns a summary report string.
        """
        report_lines = []
        today_str = datetime.now(ET).strftime("%Y-%m-%d")

        logger.info("=" * 70)
        logger.info("MULTI-STRATEGY SWING BOT - %s %s", today_str, "(DRY RUN)" if self.dry_run else "")
        logger.info("=" * 70)

        # Reset daily P&L if this is a new day
        if self.state.last_pnl_reset_date != today_str:
            old_pnl = self.state.daily_pnl
            self.state.daily_pnl = 0.0
            self.state.last_pnl_reset_date = today_str
            if old_pnl != 0.0:
                logger.info("Daily P&L reset: was $%.2f, now $0.00 (new day)", old_pnl)

        try:
            # Step 1: Connect to Alpaca
            logger.info("[1/9] Connecting to Alpaca...")
            acct = self.order_mgr.get_account()
            self.equity = float(acct.get("equity", 0))
            buying_power = float(acct.get("buying_power", 0))
            report_lines.append(f"Account equity: ${self.equity:,.2f}")
            report_lines.append(f"Buying power: ${buying_power:,.2f}")
            report_lines.append(f"Account status: {acct.get('status', '?')}")
            logger.info("Account equity: $%.2f, status: %s", self.equity, acct.get("status"))

            # Update high water mark
            if self.equity > self.state.high_water_mark:
                self.state.high_water_mark = self.equity
                logger.info("New high water mark: $%.2f", self.state.high_water_mark)

            # Step 2: Check drawdown and circuit breaker
            logger.info("[2/9] Checking drawdown / circuit breaker...")
            dd_report = self._check_drawdown()
            report_lines.extend(dd_report)

            if self.state.circuit_breaker_active:
                logger.warning("CIRCUIT BREAKER ACTIVE: %d days remaining", self.state.circuit_breaker_days_remaining)
                report_lines.append(f"CIRCUIT BREAKER: {self.state.circuit_breaker_days_remaining} days remaining, all cash")
                self.state.circuit_breaker_days_remaining -= 1
                if self.state.circuit_breaker_days_remaining <= 0:
                    self.state.circuit_breaker_active = False
                    logger.info("Circuit breaker expired, resuming trading")
                    report_lines.append("Circuit breaker expired -- resuming next run")
                self.state.last_run_date = today_str
                self.state.save()
                report = self._build_report(report_lines)
                self._write_log(report, today_str)
                return report

            # Step 3: Check pending orders
            if not self.dry_run:
                logger.info("[3/9] Checking %d pending orders...", len(self.state.pending_orders))
                pending_report = self._check_pending_orders()
                report_lines.extend(pending_report)
                self.state.save()

            # Step 4: Reconcile positions with Alpaca
            if not self.dry_run:
                logger.info("[4/9] Reconciling positions with Alpaca...")
                recon_report = self._reconcile_positions()
                report_lines.extend(recon_report)
                self.state.save()

            # Step 5: Fetch market data
            logger.info("[5/9] Fetching daily bars for %d tickers...", len(ALL_TICKERS))
            self.data = self.fetcher.fetch_daily_bars(ALL_TICKERS, days=400)
            for sym, df in self.data.items():
                if len(df) > 0:
                    logger.info("  %s: %d bars (%s to %s)", sym, len(df),
                                df.index[0].strftime("%Y-%m-%d"),
                                df.index[-1].strftime("%Y-%m-%d"))
                else:
                    logger.warning("  %s: no bars returned", sym)

            # Step 6: Compute indicators
            logger.info("[6/9] Computing indicators...")
            for sym in list(self.data.keys()):
                if len(self.data[sym]) >= 50:
                    self.data[sym] = compute_indicators(self.data[sym])
                else:
                    logger.warning("  %s: insufficient data (%d bars), skipping indicators", sym, len(self.data[sym]))

            # Step 7: Run macro regime
            logger.info("[7/9] Running macro regime overlay...")
            self._run_macro_regime(report_lines)

            # Step 8: Check exits
            logger.info("[8/9] Checking exit signals for %d positions...", len(self.state.positions))
            exit_report = self._check_exits()
            report_lines.extend(exit_report)

            # Step 9: Generate entries
            logger.info("[9/9] Generating entry signals across all strategies...")
            entry_report = self._generate_entries()
            report_lines.extend(entry_report)

            # Save final state
            self.state.last_run_date = today_str
            self.state.save()

            # Earnings summary
            earn_summary = self.earnings.summary()
            if earn_summary["blocked"]:
                report_lines.append(f"\nEarnings blocked: {', '.join(earn_summary['blocked'])}")
            if earn_summary["upcoming"]:
                report_lines.append(f"Earnings upcoming: {', '.join(earn_summary['upcoming'])}")

        except Exception as e:
            logger.error("Bot run failed: %s\n%s", e, traceback.format_exc())
            report_lines.append(f"\nERROR: {e}")
            try:
                self.state.save()
                logger.info("State saved after error")
            except Exception as save_err:
                logger.error("Failed to save state after error: %s", save_err)

        report = self._build_report(report_lines)
        self._write_log(report, today_str)
        return report

    # -----------------------------------------------------------------
    # Drawdown Management
    # -----------------------------------------------------------------

    def _check_drawdown(self) -> List[str]:
        """Check portfolio drawdown from HWM and activate protections."""
        report = []

        if self.state.high_water_mark <= 0:
            self.state.high_water_mark = self.equity
            return report

        dd = (self.equity - self.state.high_water_mark) / self.state.high_water_mark
        report.append(f"Drawdown from HWM: {dd * 100:.1f}% (HWM=${self.state.high_water_mark:,.2f})")
        logger.info("Drawdown: %.1f%% (equity=$%.2f, HWM=$%.2f)",
                     dd * 100, self.equity, self.state.high_water_mark)

        if dd <= DD_CIRCUIT_BREAKER and not self.state.circuit_breaker_active:
            # -20% drawdown: go to cash
            self.state.circuit_breaker_active = True
            self.state.circuit_breaker_days_remaining = CIRCUIT_BREAKER_DAYS
            report.append(f"CIRCUIT BREAKER TRIGGERED: DD={dd * 100:.1f}%, going to cash for {CIRCUIT_BREAKER_DAYS} days")
            logger.warning("CIRCUIT BREAKER: DD=%.1f%%, closing all positions", dd * 100)

            # Close all positions
            if not self.dry_run:
                for pos in list(self.state.positions):
                    result = self.order_mgr.close_position(pos.ticker)
                    if result is not None:
                        report.append(f"  Closed {pos.ticker} (circuit breaker)")
                self.state.positions.clear()
                self.state.pending_orders.clear()
                self.order_mgr.cancel_all_orders()

        elif dd <= DD_REDUCE_THRESHOLD:
            # -15% drawdown: reduce all sizes by 50%
            if not self.state.drawdown_reduction_active:
                self.state.drawdown_reduction_active = True
                report.append(f"DD REDUCTION ACTIVE: DD={dd * 100:.1f}%, all sizes reduced 50%")
                logger.warning("DD reduction activated: DD=%.1f%%", dd * 100)
        else:
            # Recovery: deactivate DD reduction
            if self.state.drawdown_reduction_active:
                self.state.drawdown_reduction_active = False
                report.append("DD reduction deactivated (recovered above -15%)")
                logger.info("DD reduction deactivated")

        return report

    # -----------------------------------------------------------------
    # Macro Regime
    # -----------------------------------------------------------------

    def _run_macro_regime(self, report_lines: List[str]) -> None:
        """Run the macro regime overlay."""
        try:
            from macro_regime import MacroRegimeSystem
            macro = MacroRegimeSystem()
            self.macro_output = macro.run()
            logger.info("Macro regime: %s (confidence=%.1f%%, size_mult=%.2f)",
                        self.macro_output.regime,
                        self.macro_output.regime_confidence * 100,
                        self.macro_output.position_size_multiplier)
            report_lines.append(f"Macro regime: {self.macro_output.regime} "
                                f"(confidence={self.macro_output.regime_confidence * 100:.0f}%, "
                                f"size_mult={self.macro_output.position_size_multiplier:.2f})")
            if self.macro_output.overweight:
                report_lines.append(f"  Overweight: {', '.join(self.macro_output.overweight)}")
            if self.macro_output.underweight:
                report_lines.append(f"  Underweight: {', '.join(self.macro_output.underweight)}")
            if self.macro_output.opportunities:
                report_lines.append(f"  Opportunities: {len(self.macro_output.opportunities)}")
                for opp in self.macro_output.opportunities[:3]:
                    name = opp.get("name", "?")
                    conv = opp.get("conviction", 0)
                    report_lines.append(f"    - {name} (conviction={conv:.2f})")
        except Exception as e:
            logger.warning("Macro regime failed (non-blocking): %s", e)
            self.macro_output = None
            report_lines.append(f"Macro regime: unavailable ({e})")

    # -----------------------------------------------------------------
    # Pending Orders
    # -----------------------------------------------------------------

    def _check_pending_orders(self) -> List[str]:
        """Check pending entry orders and promote filled ones to positions."""
        report = []
        still_pending = []

        for pending in self.state.pending_orders:
            order = self.order_mgr.get_order(pending.entry_order_id)
            if order is None:
                logger.warning("Pending order %s for %s not found, removing",
                               pending.entry_order_id, pending.ticker)
                report.append(f"  PENDING REMOVED: {pending.ticker} order {pending.entry_order_id} not found")
                continue

            status = order.get("status", "")
            logger.info("Pending order %s (%s): status=%s", pending.ticker, pending.entry_order_id, status)

            if status == "filled":
                fill_price = float(order.get("filled_avg_price", pending.entry_price))
                fill_qty = int(order.get("filled_qty", pending.shares))

                pos = SwingPosition(
                    ticker=pending.ticker,
                    strategy=pending.strategy,
                    entry_date=pending.entry_date,
                    entry_price=fill_price,
                    shares=fill_qty,
                    stop_price=pending.stop_price,
                    atr_at_entry=pending.atr_at_entry,
                    highest_price=fill_price,
                    entry_order_id=pending.entry_order_id,
                    entry_order_status="filled",
                )
                self.state.positions.append(pos)
                report.append(
                    f"  FILLED: {pending.ticker} ({pending.strategy}) {fill_qty} shares @ ${fill_price:.2f}"
                )
                logger.info("Pending filled: %s %d shares @ $%.2f (%s)",
                            pending.ticker, fill_qty, fill_price, pending.strategy)

                # Extract stop leg ID from OTO order (the stop was submitted
                # as part of the parent OTO order, so Alpaca activates it
                # automatically on fill -- no separate stop order needed).
                legs = order.get("legs") or []
                if legs:
                    stop_leg_id = legs[0].get("id", "")
                    if stop_leg_id:
                        pos.stop_order_id = stop_leg_id
                        logger.info("Stop leg active for %s (id=%s) at $%.2f",
                                    pending.ticker, stop_leg_id, pending.stop_price)
                elif pending.stop_price > 0 and not self.dry_run:
                    # Fallback: if the order was not OTO (e.g. placed before
                    # this fix), place a standalone stop order.
                    stop_order = self.order_mgr.place_stop_order(
                        pending.ticker, fill_qty, "sell", pending.stop_price
                    )
                    if stop_order:
                        pos.stop_order_id = stop_order.get("id")
                        logger.info("Fallback stop order placed for %s at $%.2f",
                                    pending.ticker, pending.stop_price)

            elif status in ("canceled", "cancelled", "expired", "rejected"):
                report.append(
                    f"  CANCELLED/EXPIRED: {pending.ticker} ({pending.strategy}) {status}"
                )
            elif status == "partially_filled":
                still_pending.append(pending)
                filled_qty = order.get("filled_qty", "?")
                report.append(
                    f"  PARTIAL FILL: {pending.ticker} {filled_qty}/{pending.shares} shares"
                )
            else:
                still_pending.append(pending)
                report.append(
                    f"  PENDING: {pending.ticker} ({pending.strategy}) {pending.shares} shares @ ${pending.entry_price:.2f} ({status})"
                )

        self.state.pending_orders = still_pending
        return report

    # -----------------------------------------------------------------
    # Reconciliation
    # -----------------------------------------------------------------

    def _reconcile_positions(self) -> List[str]:
        """Reconcile local state with Alpaca positions."""
        report = []

        try:
            alpaca_positions = self.order_mgr.get_positions()
        except Exception as e:
            logger.error("Failed to get Alpaca positions for reconciliation: %s", e)
            report.append(f"  RECONCILIATION ERROR: {e}")
            return report

        alpaca_by_symbol = {}
        for ap in alpaca_positions:
            sym = ap.get("symbol", "?")
            alpaca_by_symbol[sym] = ap

        # Check each local position
        positions_to_remove = []
        for pos in self.state.positions:
            if pos.ticker not in alpaca_by_symbol:
                logger.info("RECONCILE: %s no longer on Alpaca", pos.ticker)

                exit_price = pos.entry_price
                exit_reason = "position_gone_from_alpaca"

                # Check if stop order filled
                if pos.stop_order_id:
                    stop_order = self.order_mgr.get_order(pos.stop_order_id)
                    if stop_order and stop_order.get("status") == "filled":
                        exit_price = float(stop_order.get("filled_avg_price", pos.stop_price))
                        exit_reason = "stop_loss_filled"

                pnl = (exit_price - pos.entry_price) * pos.shares
                pnl_pct = (exit_price / pos.entry_price - 1) * 100 if pos.entry_price > 0 else 0

                report.append(
                    f"  RECONCILE EXIT {pos.ticker} ({pos.strategy}): {exit_reason}, "
                    f"P&L=${pnl:.0f} ({pnl_pct:+.1f}%)"
                )

                # Update stats
                self.state.total_trades += 1
                self.state.daily_pnl += pnl
                self.state.strategy_pnl[pos.strategy] = self.state.strategy_pnl.get(pos.strategy, 0.0) + pnl
                if pnl > 0:
                    self.state.total_wins += 1
                else:
                    self.state.total_losses += 1

                positions_to_remove.append(pos)

        for pos in positions_to_remove:
            self.state.positions.remove(pos)

        # Flag Alpaca positions not tracked locally
        local_tickers = {p.ticker for p in self.state.positions}
        pending_tickers = {p.ticker for p in self.state.pending_orders}
        for sym, ap in alpaca_by_symbol.items():
            if sym not in local_tickers and sym not in pending_tickers:
                qty = ap.get("qty", "?")
                entry = float(ap.get("avg_entry_price", 0))
                pnl = float(ap.get("unrealized_pl", 0))
                report.append(
                    f"  UNTRACKED: {sym} qty={qty}, entry=${entry:.2f}, P&L=${pnl:.2f}"
                )
                logger.warning("UNTRACKED Alpaca position: %s qty=%s", sym, qty)

        if not positions_to_remove and not any("UNTRACKED" in r for r in report):
            report.append("  Reconciliation: all positions match")

        return report

    # -----------------------------------------------------------------
    # Exit Logic
    # -----------------------------------------------------------------

    def _check_exits(self) -> List[str]:
        """Check exit conditions for all open positions."""
        report = []
        positions_to_close = []

        for pos in self.state.positions:
            df = self.data.get(pos.ticker)
            if df is None or len(df) < 1:
                logger.warning("No data for position %s, cannot check exits", pos.ticker)
                continue

            # Verify position still on Alpaca
            if not self.dry_run:
                alpaca_pos = self.order_mgr.get_position(pos.ticker)
                if alpaca_pos is None:
                    logger.info("%s: no longer on Alpaca, skip exit checks", pos.ticker)
                    continue

            current_price = float(df.iloc[-1]["Close"])
            pos.days_held += 1

            # Check exit
            exit_result = self.exit_mgr.check_exits(pos, df, current_price, self.data)

            # Propagate trailing stop updates to Alpaca
            # (Donchian strategy updates pos.stop_price in-place)
            if pos.strategy == "donchian_breakout" and pos.stop_order_id and not self.dry_run:
                # Try to update the stop on Alpaca
                result = self.order_mgr.replace_stop_order(pos.stop_order_id, pos.stop_price)
                if result:
                    pos.stop_order_id = result.get("id", pos.stop_order_id)

            if exit_result:
                reason, exit_price = exit_result
                positions_to_close.append((pos, reason, exit_price, current_price))
            else:
                pnl = (current_price - pos.entry_price) * pos.shares
                pnl_pct = (current_price / pos.entry_price - 1) * 100
                report.append(
                    f"  HOLD {pos.ticker} ({pos.strategy}): {pos.shares} sh, "
                    f"entry=${pos.entry_price:.2f}, now=${current_price:.2f}, "
                    f"P&L=${pnl:.0f} ({pnl_pct:+.1f}%), stop=${pos.stop_price:.2f}, "
                    f"day {pos.days_held}"
                )

        # Execute exits
        for pos, reason, exit_price, current_price in positions_to_close:
            pnl = (exit_price - pos.entry_price) * pos.shares
            pnl_pct = (exit_price / pos.entry_price - 1) * 100

            report.append(
                f"  EXIT {pos.ticker} ({pos.strategy}): {reason}, "
                f"entry=${pos.entry_price:.2f}, exit=${exit_price:.2f}, "
                f"P&L=${pnl:.0f} ({pnl_pct:+.1f}%), held {pos.days_held}d"
            )
            logger.info(
                "EXIT %s (%s): %s, entry=$%.2f, exit=$%.2f, P&L=$%.0f",
                pos.ticker, pos.strategy, reason, pos.entry_price, exit_price, pnl,
            )

            if not self.dry_run:
                # Cancel any existing stop order first
                if pos.stop_order_id:
                    self.order_mgr.cancel_order(pos.stop_order_id)
                # Close position
                result = self.order_mgr.close_position(pos.ticker)
                if result is not None:
                    logger.info("Position closed on Alpaca: %s", pos.ticker)
                else:
                    logger.warning("Failed to close on Alpaca: %s", pos.ticker)

            # Update state
            self.state.positions.remove(pos)
            self.state.total_trades += 1
            self.state.daily_pnl += pnl
            self.state.strategy_pnl[pos.strategy] = self.state.strategy_pnl.get(pos.strategy, 0.0) + pnl
            if pnl > 0:
                self.state.total_wins += 1
            else:
                self.state.total_losses += 1

            self.state.save()

        return report

    # -----------------------------------------------------------------
    # Entry Generation
    # -----------------------------------------------------------------

    def _generate_entries(self) -> List[str]:
        """Generate and execute new entry signals across all 5 strategies."""
        report = []

        engine = StrategyEngine(
            equity=self.equity,
            state=self.state,
            earnings=self.earnings,
            data=self.data,
            macro_output=self.macro_output,
        )

        all_signals: List[SwingSignal] = []

        # Strategy 1: Momentum Rotation
        try:
            mom_signals = engine.generate_momentum_signals()
            all_signals.extend(mom_signals)
            logger.info("Momentum: %d signals", len(mom_signals))
        except Exception as e:
            logger.error("Momentum signal generation failed: %s\n%s", e, traceback.format_exc())
            report.append(f"  Momentum ERROR: {e}")

        # Strategy 2: VWAP Mean Reversion
        try:
            vwap_signals = engine.generate_vwap_signals()
            all_signals.extend(vwap_signals)
            logger.info("VWAP MR: %d signals", len(vwap_signals))
        except Exception as e:
            logger.error("VWAP signal generation failed: %s\n%s", e, traceback.format_exc())
            report.append(f"  VWAP MR ERROR: {e}")

        # Strategy 3: Sector Relative Strength
        try:
            srs_signals = engine.generate_sector_rs_signals()
            all_signals.extend(srs_signals)
            logger.info("Sector RS: %d signals", len(srs_signals))
        except Exception as e:
            logger.error("Sector RS signal generation failed: %s\n%s", e, traceback.format_exc())
            report.append(f"  Sector RS ERROR: {e}")

        # Strategy 4: Donchian Breakout
        try:
            donch_signals = engine.generate_donchian_signals()
            all_signals.extend(donch_signals)
            logger.info("Donchian: %d signals", len(donch_signals))
        except Exception as e:
            logger.error("Donchian signal generation failed: %s\n%s", e, traceback.format_exc())
            report.append(f"  Donchian ERROR: {e}")

        # Strategy 5: RSI(2) Mean Reversion
        try:
            rsi2_signals = engine.generate_rsi2_signals()
            all_signals.extend(rsi2_signals)
            logger.info("RSI(2): %d signals", len(rsi2_signals))
        except Exception as e:
            logger.error("RSI(2) signal generation failed: %s\n%s", e, traceback.format_exc())
            report.append(f"  RSI(2) ERROR: {e}")

        # Sort: sells first (momentum rotation), then buys by priority
        sells = [s for s in all_signals if s.direction == "sell"]
        buys = [s for s in all_signals if s.direction == "buy"]
        buys.sort(key=lambda s: s.priority, reverse=True)

        ordered_signals = sells + buys

        if not ordered_signals:
            report.append("\nNo new signals generated.")
            logger.info("No signals generated")
        else:
            report.append(f"\n{len(ordered_signals)} signal(s) generated:")

            # Per-strategy summary
            by_strat = {}
            for sig in ordered_signals:
                by_strat.setdefault(sig.strategy, []).append(sig)
            for strat, sigs in by_strat.items():
                actionable = sum(1 for s in sigs if not s.blocked)
                blocked = sum(1 for s in sigs if s.blocked)
                report.append(f"  {strat}: {actionable} actionable, {blocked} blocked")

            for sig in ordered_signals:
                status = "BLOCKED" if sig.blocked else ("SELL" if sig.direction == "sell" else "BUY")
                report.append(
                    f"  [{status}] {sig.strategy} {sig.ticker}: {sig.rationale}"
                )
                if sig.blocked:
                    report.append(f"    Block reason: {sig.block_reason}")

                # Execute actionable signals
                if not sig.blocked:
                    if sig.direction == "sell":
                        if self.dry_run:
                            report.append(f"    DRY RUN: Would sell {sig.shares} shares of {sig.ticker}")
                        else:
                            self._execute_exit(sig, report)
                    else:
                        # Fundamental quality gate (runs before any buy order)
                        try:
                            from fundamental_filter import should_trade as _fund_check
                            can_trade, fund_reason = _fund_check(sig.ticker)
                            if not can_trade:
                                logger.info("FUNDAMENTAL REJECT: %s -- %s", sig.ticker, fund_reason)
                                report.append(f"    FUNDAMENTAL REJECT: {sig.ticker} -- {fund_reason}")
                                continue
                        except ImportError:
                            pass  # fundamental_filter not available, skip check

                        if self.dry_run:
                            report.append(f"    DRY RUN: Would buy {sig.shares} shares of {sig.ticker} @ ${sig.entry_price:.2f}")
                        else:
                            self._execute_entry(sig, report)

        return report

    def _execute_entry(self, signal: SwingSignal, report: List[str]) -> None:
        """Execute a trade entry on Alpaca.

        Uses an OTO (One-Triggers-Other) order when a stop price is set so
        that the market buy and the GTC stop-loss sell are submitted as a
        single atomic order, avoiding Alpaca's "potential wash trade"
        rejection.
        """
        try:
            # Use OTO order when we have a stop price, plain market otherwise
            if signal.stop_price > 0:
                order = self.order_mgr.place_oto_order(
                    symbol=signal.ticker,
                    qty=signal.shares,
                    side="buy",
                    stop_price=signal.stop_price,
                    time_in_force="day",
                )
            else:
                order = self.order_mgr.place_market_order(
                    symbol=signal.ticker,
                    qty=signal.shares,
                    side="buy",
                    time_in_force="day",
                )

            if order:
                order_id = order.get("id", "?")
                order_status = order.get("status", "new")

                # Extract stop leg ID from OTO response
                legs = order.get("legs") or []
                stop_leg_id = legs[0].get("id", "") if legs else ""

                if order_status == "filled":
                    fill_price = float(order.get("filled_avg_price", signal.entry_price))
                    fill_qty = int(order.get("filled_qty", signal.shares))

                    pos = SwingPosition(
                        ticker=signal.ticker,
                        strategy=signal.strategy,
                        entry_date=datetime.now(ET).strftime("%Y-%m-%d"),
                        entry_price=fill_price,
                        shares=fill_qty,
                        stop_price=signal.stop_price,
                        atr_at_entry=signal.atr,
                        highest_price=fill_price,
                        entry_order_id=order_id,
                        entry_order_status="filled",
                    )
                    # Stop was submitted as part of the OTO order
                    if stop_leg_id:
                        pos.stop_order_id = stop_leg_id
                    self.state.positions.append(pos)

                    report.append(
                        f"    FILLED: {fill_qty} shares {signal.ticker} @ ${fill_price:.2f} "
                        f"(stop=${signal.stop_price:.2f})"
                    )
                    logger.info("Entry filled (OTO): %s %d @ $%.2f (%s) stop_leg=%s",
                                signal.ticker, fill_qty, fill_price, signal.strategy, stop_leg_id)
                else:
                    pending = PendingOrder(
                        ticker=signal.ticker,
                        strategy=signal.strategy,
                        entry_order_id=order_id,
                        entry_date=datetime.now(ET).strftime("%Y-%m-%d"),
                        entry_price=signal.entry_price,
                        shares=signal.shares,
                        stop_price=signal.stop_price,
                        atr_at_entry=signal.atr,
                    )
                    self.state.pending_orders.append(pending)
                    report.append(
                        f"    PENDING: {signal.shares} shares {signal.ticker} ({order_status})"
                    )

                self.state.save()
            else:
                report.append(f"    ORDER FAILED: {signal.ticker}")
                logger.error("Failed to place order for %s", signal.ticker)

        except Exception as e:
            report.append(f"    ORDER ERROR: {e}")
            logger.error("Entry execution error for %s: %s", signal.ticker, e)

    def _execute_exit(self, signal: SwingSignal, report: List[str]) -> None:
        """Execute a sell (momentum rotation exit)."""
        try:
            # Find and remove the position
            pos_to_remove = None
            for pos in self.state.positions:
                if pos.ticker == signal.ticker and pos.strategy == signal.strategy:
                    pos_to_remove = pos
                    break

            if pos_to_remove is None:
                report.append(f"    EXIT SKIP: no position found for {signal.ticker} in {signal.strategy}")
                return

            # Cancel stop order if exists
            if pos_to_remove.stop_order_id:
                self.order_mgr.cancel_order(pos_to_remove.stop_order_id)

            result = self.order_mgr.close_position(signal.ticker)
            if result is not None:
                # Get fill info from alpaca
                current_df = self.data.get(signal.ticker)
                exit_price = float(current_df.iloc[-1]["Close"]) if current_df is not None and len(current_df) > 0 else pos_to_remove.entry_price
                pnl = (exit_price - pos_to_remove.entry_price) * pos_to_remove.shares

                report.append(
                    f"    SOLD: {pos_to_remove.shares} shares {signal.ticker} "
                    f"(P&L=${pnl:.0f})"
                )

                self.state.positions.remove(pos_to_remove)
                self.state.total_trades += 1
                self.state.daily_pnl += pnl
                self.state.strategy_pnl[signal.strategy] = self.state.strategy_pnl.get(signal.strategy, 0.0) + pnl
                if pnl > 0:
                    self.state.total_wins += 1
                else:
                    self.state.total_losses += 1
                self.state.save()
            else:
                report.append(f"    SELL FAILED: {signal.ticker}")

        except Exception as e:
            report.append(f"    SELL ERROR: {e}")
            logger.error("Exit execution error for %s: %s", signal.ticker, e)

    # -----------------------------------------------------------------
    # Report Building
    # -----------------------------------------------------------------

    def _build_report(self, report_lines: List[str]) -> str:
        """Build the final summary report."""
        now = datetime.now(ET)
        win_rate = self.state.total_wins / max(self.state.total_trades, 1) * 100

        # Drawdown
        hwm = self.state.high_water_mark if self.state.high_water_mark > 0 else self.equity
        dd_pct = ((self.equity - hwm) / hwm * 100) if hwm > 0 else 0.0

        lines = [
            "=" * 80,
            f"MULTI-STRATEGY SWING BOT REPORT - {now.strftime('%Y-%m-%d %H:%M ET')}",
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE PAPER'}",
            "=" * 80,
            "",
            "--- PORTFOLIO SUMMARY ---",
            f"  Equity:       ${self.equity:,.2f}",
            f"  HWM:          ${hwm:,.2f}",
            f"  DD from HWM:  {dd_pct:.1f}%",
            f"  Positions:    {len(self.state.positions)}",
            f"  Pending:      {len(self.state.pending_orders)}",
            f"  Total trades: {self.state.total_trades} (W:{self.state.total_wins} L:{self.state.total_losses})",
            f"  Win rate:     {win_rate:.0f}%",
            f"  Daily P&L:    ${self.state.daily_pnl:,.2f}",
        ]

        # Macro regime
        if self.macro_output is not None:
            lines.append(f"  Macro regime: {self.macro_output.regime} "
                         f"(size_mult={self.macro_output.position_size_multiplier:.2f})")
        else:
            lines.append("  Macro regime: unavailable")

        if self.state.circuit_breaker_active:
            lines.append(f"  CIRCUIT BREAKER: ACTIVE ({self.state.circuit_breaker_days_remaining} days)")
        if self.state.drawdown_reduction_active:
            lines.append("  DD REDUCTION: ACTIVE (all sizes -50%)")

        # Per-strategy breakdown
        lines.append("")
        lines.append("--- STRATEGY BREAKDOWN ---")
        lines.append(f"  {'Strategy':<30} {'Alloc':>8} {'Pos':>4} {'P&L':>10}")
        lines.append("  " + "-" * 56)
        for strat, alloc in STRATEGY_ALLOCATIONS.items():
            pos_count = len([p for p in self.state.positions if p.strategy == strat])
            pnl = self.state.strategy_pnl.get(strat, 0.0)
            alloc_str = f"${self.equity * alloc:,.0f}"
            lines.append(f"  {strat:<30} {alloc_str:>8} {pos_count:>4} ${pnl:>9,.2f}")

        # All positions
        lines.append("")
        lines.append("--- OPEN POSITIONS ---")
        if self.state.positions:
            lines.append(f"  {'Ticker':<6} {'Strategy':<28} {'Shares':>6} {'Entry':>8} {'Stop':>8} {'Days':>5} {'P&L':>10}")
            lines.append("  " + "-" * 75)
            for pos in sorted(self.state.positions, key=lambda p: p.strategy):
                df = self.data.get(pos.ticker)
                if df is not None and len(df) > 0:
                    current = float(df.iloc[-1]["Close"])
                    pnl = (current - pos.entry_price) * pos.shares
                    pnl_pct = (current / pos.entry_price - 1) * 100
                    pnl_str = f"${pnl:,.0f} ({pnl_pct:+.1f}%)"
                else:
                    pnl_str = "N/A"
                stop_str = f"${pos.stop_price:.2f}" if pos.stop_price > 0 else "none"
                lines.append(
                    f"  {pos.ticker:<6} {pos.strategy:<28} {pos.shares:>6} "
                    f"${pos.entry_price:>7.2f} {stop_str:>8} {pos.days_held:>5} {pnl_str:>10}"
                )
        else:
            lines.append("  (no open positions)")

        # Detail lines from run
        lines.append("")
        lines.append("--- RUN DETAILS ---")
        lines.extend(report_lines)

        # Indicator snapshot (top 10 tickers by activity)
        lines.append("")
        lines.append("--- INDICATOR SNAPSHOT (trading tickers) ---")
        lines.append(
            f"  {'Sym':>6} | {'Close':>8} | {'RSI2':>6} | {'RSI14':>6} | "
            f"{'VWAP_Z':>7} | {'ATR14':>7} | {'Vol/Avg':>7} | >SMA200"
        )
        lines.append("  " + "-" * 75)
        for ticker in TRADING_TICKERS:
            df = self.data.get(ticker)
            if df is None or len(df) < 1:
                continue
            row = df.iloc[-1]
            close_val = row.get("Close", 0)
            rsi2_val = row.get("RSI2")
            rsi14_val = row.get("RSI14")
            vwap_z_val = row.get("VWAP_Z")
            atr_val = row.get("ATR14")
            vol_val = row.get("Volume", 0)
            vol_avg_val = row.get("VOL_AVG20", 1)
            sma200_val = row.get("SMA200")

            rsi2_str = f"{float(rsi2_val):>5.1f}" if rsi2_val is not None and not pd.isna(rsi2_val) else "  N/A"
            rsi14_str = f"{float(rsi14_val):>5.1f}" if rsi14_val is not None and not pd.isna(rsi14_val) else "  N/A"
            vwap_z_str = f"{float(vwap_z_val):>6.2f}" if vwap_z_val is not None and not pd.isna(vwap_z_val) else "   N/A"
            atr_str = f"${float(atr_val):>5.2f}" if atr_val is not None and not pd.isna(atr_val) else "   N/A"
            vol_ratio = float(vol_val) / float(vol_avg_val) if vol_avg_val and not pd.isna(vol_avg_val) and float(vol_avg_val) > 0 else 0
            vol_str = f"{vol_ratio:>5.1f}x"
            above_200 = "YES" if sma200_val is not None and not pd.isna(sma200_val) and float(close_val) > float(sma200_val) else "no"

            lines.append(
                f"  {ticker:>6} | ${float(close_val):>7.2f} | {rsi2_str} | {rsi14_str} | "
                f" {vwap_z_str} | {atr_str} | {vol_str} | {above_200}"
            )

        # Strategy allocation table
        lines.append("")
        lines.append(f"Strategies: {', '.join(STRATEGY_ALLOCATIONS.keys())}")
        lines.append(f"Universe: {len(TRADING_TICKERS)} tickers")

        lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)

    def _write_log(self, report: str, today_str: str) -> None:
        """Write report to log file."""
        log_path = LOG_DIR / f"swing_{today_str}.log"
        try:
            with open(log_path, "a") as f:
                f.write(report + "\n\n")
            logger.info("Report written to %s", log_path)
        except Exception as e:
            logger.error("Failed to write report: %s", e)

    # -----------------------------------------------------------------
    # Status Command
    # -----------------------------------------------------------------

    def status(self) -> str:
        """Generate a quick status report of current positions."""
        try:
            acct = self.order_mgr.get_account()
            equity = float(acct.get("equity", 0))
            alpaca_positions = self.order_mgr.get_positions()
        except Exception as e:
            return f"Failed to connect to Alpaca: {e}"

        hwm = self.state.high_water_mark if self.state.high_water_mark > 0 else equity
        dd = (equity - hwm) / hwm * 100 if hwm > 0 else 0.0
        win_rate = self.state.total_wins / max(self.state.total_trades, 1) * 100

        lines = [
            "=" * 70,
            f"SWING BOT STATUS - {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}",
            "=" * 70,
            f"Equity:       ${equity:,.2f}",
            f"Cash:         ${float(acct.get('cash', 0)):,.2f}",
            f"HWM:          ${hwm:,.2f}",
            f"DD from HWM:  {dd:.1f}%",
            f"Total trades: {self.state.total_trades} (W:{self.state.total_wins} L:{self.state.total_losses})",
            f"Win rate:     {win_rate:.0f}%",
            "",
        ]

        # Per-strategy summary
        lines.append("Strategy Positions:")
        for strat in STRATEGY_ALLOCATIONS:
            positions = [p for p in self.state.positions if p.strategy == strat]
            pnl = self.state.strategy_pnl.get(strat, 0.0)
            lines.append(f"  {strat}: {len(positions)} positions, realized P&L=${pnl:,.2f}")
            for pos in positions:
                lines.append(
                    f"    {pos.ticker}: {pos.shares} sh @ ${pos.entry_price:.2f}, "
                    f"stop=${pos.stop_price:.2f}, day {pos.days_held}"
                )

        # Alpaca positions
        if alpaca_positions:
            lines.append("")
            lines.append("Alpaca Positions:")
            for p in alpaca_positions:
                sym = p.get("symbol", "?")
                qty = int(p.get("qty", 0))
                entry = float(p.get("avg_entry_price", 0))
                current = float(p.get("current_price", 0))
                pnl = float(p.get("unrealized_pl", 0))
                pnl_pct = float(p.get("unrealized_plpc", 0)) * 100
                lines.append(
                    f"  {sym}: {qty} shares, entry=${entry:.2f}, now=${current:.2f}, "
                    f"P&L=${pnl:.2f} ({pnl_pct:+.1f}%)"
                )

        # Pending orders
        if self.state.pending_orders:
            lines.append("")
            lines.append(f"Pending Orders: {len(self.state.pending_orders)}")
            for p in self.state.pending_orders:
                lines.append(f"  {p.ticker} ({p.strategy}): {p.shares} sh @ ${p.entry_price:.2f}")

        if self.state.circuit_breaker_active:
            lines.append("")
            lines.append(f"CIRCUIT BREAKER: ACTIVE ({self.state.circuit_breaker_days_remaining} days remaining)")

        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(level: str = "INFO"):
    """Configure logging with console + rotating file."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file
    today_str = datetime.now(ET).strftime("%Y%m%d")
    log_file = LOG_DIR / f"swing_{today_str}.log"
    fh = logging.handlers.RotatingFileHandler(
        str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5,
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-strategy swing trading bot for Alpaca paper trading")
    parser.add_argument("--dry-run", action="store_true", help="Compute signals without placing orders")
    parser.add_argument("--status", action="store_true", help="Show current positions and status")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    bot = SwingBot(dry_run=args.dry_run)

    if args.status:
        print(bot.status())
    else:
        report = bot.run()
        print(report)


if __name__ == "__main__":
    main()
