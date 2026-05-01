#!/usr/bin/env python3
"""
options_overlay.py - Optional options premium selling overlay for the swing trading bot.

This is a STANDALONE, OPTIONAL add-on. The swing bot (swing_runner.py) works
perfectly without this module. This module can be:
  - Run standalone via CLI
  - Imported by swing_runner.py as an optional enhancement

Strategies:
  1. Covered Calls  - Sell OTM calls on positions with >= 100 shares
  2. Cash-Secured Puts (The Wheel) - Sell OTM puts on swing bot watchlist tickers
  3. Earnings Premium Harvesting - Sell iron condors before earnings on high-IV stocks

Architecture:
  - Reads positions directly from Alpaca (not from swing bot state file)
  - Never buys options -- only sells (profitable side of IV > RV gap)
  - Fails gracefully if option APIs are unavailable

Usage:
    python options_overlay.py                    # Full run (dry-run by default)
    python options_overlay.py --dry-run          # Show what it would do
    python options_overlay.py --live             # Actually place orders
    python options_overlay.py --status           # Current option positions
    python options_overlay.py --backtest         # Run historical backtest
    python options_overlay.py --conditions       # Show per-position on/off conditions

Author: Options Overlay System
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
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure our modules are importable
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/opt/jarvis-utils/lib")

logger = logging.getLogger("options_overlay")

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.resolve()
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = BASE_DIR / "options_overlay_state.json"
EARNINGS_FILE = BASE_DIR / "earnings_calendar.json"

PAPER_BASE = "https://paper-api.alpaca.markets/v2"
DATA_BASE = "https://data.alpaca.markets/v2"
OPTIONS_DATA_BASE = "https://data.alpaca.markets/v1beta1/options"

# The swing bot's ticker universe -- import from combined_config for single
# source of truth.  Only CC-eligible tickers should be used here.
# RSI2 mean-reversion mining universe: PAAS, AG, HL (updated Mar 12, David approved)
# CC overlay applies to positions held >= CC_ELIGIBLE_AFTER_DAYS (0 days, David Mar 15)
from combined_config import CC_ELIGIBLE_SWING, CC_ELIGIBLE_AFTER_DAYS
SWING_TICKERS = CC_ELIGIBLE_SWING

# --- Covered Call Parameters ---
CC_TARGET_DELTA = 0.30          # Target ~0.30 delta (roughly 1 std dev OTM)
CC_DELTA_MIN = 0.20             # Minimum acceptable delta
CC_DELTA_MAX = 0.40             # Maximum acceptable delta
CC_MIN_DTE = 14                 # Minimum days to expiry
CC_MAX_DTE = 28                 # Maximum days to expiry
CC_MIN_SHARES = 100             # Minimum shares for covered call (1 round lot)
CC_PROFIT_TARGET = 0.50         # Close at 50% profit
CC_ROLL_DTE = 7                 # Roll when 7 days to expiry remain

# --- Cash-Secured Put Parameters ---
CSP_TARGET_DELTA = 0.30         # Target ~0.30 delta
CSP_DELTA_MIN = 0.20
CSP_DELTA_MAX = 0.40
CSP_MIN_DTE = 14
CSP_MAX_DTE = 28
CSP_MAX_CONCURRENT = 3          # Max 3 concurrent puts
CSP_PROFIT_TARGET = 0.50        # Close at 50% profit
CSP_ROLL_DTE = 7
CSP_MAX_PORTFOLIO_PCT = 0.30    # Max 30% of portfolio in put obligations

# --- Earnings Iron Condor Parameters ---
IC_MAX_IV_RANK_THRESHOLD = 80   # Only trade when IV rank > 80%
IC_CALL_DELTA = 0.16            # Short call delta (~84% OTM)
IC_PUT_DELTA = 0.16             # Short put delta (~84% OTM)
IC_WING_WIDTH = 5.0             # $5 wide wings (caps max loss)
IC_MAX_CONTRACTS = 1            # 1 contract per side
IC_MAX_LOSS = 500               # Maximum loss per iron condor
IC_DAYS_BEFORE_EARNINGS = 1     # Enter 1 day before earnings

# --- General Risk Parameters ---
EARNINGS_BLOCK_DAYS = 5         # No CC/CSP within 5 days of earnings
MIN_OPEN_INTEREST = 50          # Minimum open interest for liquidity
MAX_BID_ASK_SPREAD_PCT = 0.10   # Max 10% bid-ask spread relative to mid

# Default assumed annualized volatility for delta estimation
DEFAULT_VOLATILITY = 0.30


# =============================================================================
# BLACK-SCHOLES PRICING & GREEKS
# =============================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price. S=spot, K=strike, T=years, r=rate, sigma=vol."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put delta (negative for puts, we return absolute value)."""
    return abs(bs_call_delta(S, K, T, r, sigma) - 1.0)


def bs_theta(S: float, K: float, T: float, r: float, sigma: float,
             option_type: str = "call") -> float:
    """Black-Scholes theta (per day). Negative for long options."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    first_term = -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
    if option_type == "call":
        theta_annual = first_term - r * K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        theta_annual = first_term + r * K * math.exp(-r * T) * _norm_cdf(-d2)
    return theta_annual / 365.0


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes vega (per 1% vol move)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return S * _norm_pdf(d1) * math.sqrt(T) / 100.0


def find_strike_for_delta(
    S: float, T: float, r: float, sigma: float,
    target_delta: float, option_type: str = "call",
    strike_step: float = 1.0,
) -> float:
    """Find the strike price closest to a target delta using bisection."""
    if option_type == "call":
        # For calls: higher strike = lower delta
        K_low = S * 0.80
        K_high = S * 1.40
    else:
        # For puts: lower strike = lower (absolute) delta
        K_low = S * 0.60
        K_high = S * 1.20

    delta_fn = bs_call_delta if option_type == "call" else bs_put_delta

    for _ in range(50):  # bisection iterations
        K_mid = (K_low + K_high) / 2.0
        d = delta_fn(S, K_mid, T, r, sigma)
        if option_type == "call":
            if d > target_delta:
                K_low = K_mid
            else:
                K_high = K_mid
        else:
            if d > target_delta:
                K_high = K_mid
            else:
                K_low = K_mid
        if abs(K_high - K_low) < strike_step * 0.1:
            break

    # Round to nearest strike step
    K_result = round((K_low + K_high) / 2.0 / strike_step) * strike_step
    return K_result


def compute_realized_vol(prices: pd.Series, window: int = 20) -> float:
    """Compute annualized realized volatility from a price series."""
    if prices is None or len(prices) < window + 1:
        return DEFAULT_VOLATILITY
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < window:
        return DEFAULT_VOLATILITY
    rv = log_returns.tail(window).std() * math.sqrt(252)
    return float(rv) if not math.isnan(rv) else DEFAULT_VOLATILITY


def compute_iv_rank(current_iv: float, iv_history: pd.Series) -> float:
    """Compute IV rank: where current IV sits relative to 52-week range."""
    if iv_history is None or len(iv_history) < 20:
        return 50.0  # default to middle
    iv_min = iv_history.min()
    iv_max = iv_history.max()
    if iv_max == iv_min:
        return 50.0
    return float((current_iv - iv_min) / (iv_max - iv_min) * 100.0)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OptionPosition:
    """Tracks a sold option position."""
    underlying: str
    option_symbol: str
    option_type: str            # "call", "put", "iron_condor"
    strike: float
    expiration: str             # YYYY-MM-DD
    contracts: int
    premium_received: float     # total premium (per-share price * contracts * 100)
    sell_price: float           # per-share option price we sold at
    sell_date: str              # ISO timestamp
    strategy: str               # "covered_call", "cash_secured_put", "earnings_ic"
    status: str = "open"        # open, closed, expired, assigned, rolled
    close_price: Optional[float] = None
    close_date: Optional[str] = None
    realized_pnl: Optional[float] = None
    # For iron condors, store all legs
    ic_legs: Optional[Dict[str, Any]] = None

    @property
    def days_to_expiration(self) -> int:
        try:
            exp_date = datetime.strptime(self.expiration, "%Y-%m-%d").date()
            return (exp_date - datetime.now(timezone.utc).date()).days
        except (ValueError, TypeError):
            return 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class OverlayState:
    """Persistent state for the options overlay."""
    positions: List[OptionPosition] = field(default_factory=list)
    total_premium_collected: float = 0.0
    total_realized_pnl: float = 0.0
    total_trades: int = 0
    last_run_date: str = ""

    def save(self, path: Path = STATE_FILE):
        data = {
            "positions": [p.to_dict() for p in self.positions],
            "total_premium_collected": self.total_premium_collected,
            "total_realized_pnl": self.total_realized_pnl,
            "total_trades": self.total_trades,
            "last_run_date": self.last_run_date,
        }
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp.rename(path)
        logger.debug("State saved to %s", path)

    @classmethod
    def load(cls, path: Path = STATE_FILE) -> "OverlayState":
        if not path.exists():
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            state = cls()
            state.total_premium_collected = data.get("total_premium_collected", 0.0)
            state.total_realized_pnl = data.get("total_realized_pnl", 0.0)
            state.total_trades = data.get("total_trades", 0)
            state.last_run_date = data.get("last_run_date", "")
            for p_data in data.get("positions", []):
                state.positions.append(OptionPosition(**{
                    k: v for k, v in p_data.items()
                    if k in OptionPosition.__dataclass_fields__
                }))
            return state
        except Exception as e:
            logger.error("Failed to load overlay state: %s", e)
            return cls()


# =============================================================================
# EARNINGS CALENDAR HELPER
# =============================================================================

class EarningsHelper:
    """Reads the shared earnings calendar to check earnings proximity."""

    def __init__(self, path: Path = EARNINGS_FILE):
        self._calendar: Dict[str, Dict] = {}
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                self._calendar = {
                    k: v for k, v in data.items()
                    if not k.startswith("_") and isinstance(v, dict)
                }
            except Exception as e:
                logger.warning("Could not load earnings calendar: %s", e)

    def days_until_earnings(self, symbol: str) -> Optional[int]:
        """Return calendar days until earnings, or None if unknown."""
        entry = self._calendar.get(symbol, {})
        earn_date_str = entry.get("date")
        if not earn_date_str:
            return None
        try:
            earn_date = datetime.strptime(earn_date_str, "%Y-%m-%d").date()
            today = datetime.now(timezone.utc).date()
            delta = (earn_date - today).days
            return delta
        except (ValueError, TypeError):
            return None

    def is_near_earnings(self, symbol: str, days: int = EARNINGS_BLOCK_DAYS) -> bool:
        """True if earnings are within `days` calendar days."""
        d = self.days_until_earnings(symbol)
        if d is None:
            return False  # Unknown earnings = allow trading
        return 0 <= d <= days

    def tickers_with_earnings_soon(self, tickers: List[str], days: int = 3) -> List[str]:
        """Return tickers with earnings within `days` calendar days."""
        result = []
        for t in tickers:
            d = self.days_until_earnings(t)
            if d is not None and 0 <= d <= days:
                result.append(t)
        return result


# =============================================================================
# ALPACA API CLIENT (OPTIONS-FOCUSED)
# =============================================================================

class AlpacaOptionsClient:
    """
    Thin Alpaca API wrapper focused on options operations.
    Reads credentials from jarvis-utils secrets.
    """

    def __init__(self):
        # Edge 123 port (Apr 22 2026): auto-refresh creds on 401.
        from alpaca_client import _AutoRefreshSession
        self._session = _AutoRefreshSession(self._refresh_session_credentials)
        self._api_key = None
        self._api_secret = None
        self._load_credentials()

    def _refresh_session_credentials(self, session) -> None:
        """Edge 123 port (Apr 22 2026): re-pull creds on 401."""
        from jarvis_utils.secrets import get
        new_key = get("Alpaca", "api_key_id",
                      user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        new_secret = get("Alpaca", "secret_key",
                         user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        if not new_key or not new_secret:
            raise EnvironmentError("options_overlay cred refresh: empty creds")
        self._api_key = new_key
        self._api_secret = new_secret
        session.headers["APCA-API-KEY-ID"] = new_key
        session.headers["APCA-API-SECRET-KEY"] = new_secret

    def _load_credentials(self):
        """Load Alpaca credentials from jarvis-utils secrets."""
        try:
            from jarvis_utils.secrets import get
            self._api_key = get("Alpaca", "api_key_id",
                                user="a4dc8459-608d-49f5-943e-e5e105ed5207")
            self._api_secret = get("Alpaca", "secret_key",
                                   user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        except Exception:
            pass

        if not self._api_key or not self._api_secret:
            # Fallback: try .env file
            try:
                from dotenv import load_dotenv
                load_dotenv(str(BASE_DIR / ".env"))
                self._api_key = os.environ.get("APCA_API_KEY_ID", "")
                self._api_secret = os.environ.get("APCA_API_SECRET_KEY", "")
            except Exception:
                pass

        if not self._api_key or not self._api_secret:
            raise EnvironmentError(
                "Alpaca API credentials not found. Set them in jarvis-utils secrets "
                "or in ~/trading_bot/.env (APCA_API_KEY_ID, APCA_API_SECRET_KEY)."
            )

        self._session.headers.update({
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
        })

    # -- Account & Positions --

    def get_account(self) -> Dict[str, Any]:
        r = self._session.get(f"{PAPER_BASE}/account")
        r.raise_for_status()
        return r.json()

    def get_equity(self) -> float:
        acct = self.get_account()
        return float(acct.get("equity", 0))

    def get_buying_power(self) -> float:
        acct = self.get_account()
        return float(acct.get("buying_power", 0))

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions (stocks and options)."""
        r = self._session.get(f"{PAPER_BASE}/positions")
        r.raise_for_status()
        return r.json()

    def get_stock_positions(self) -> List[Dict[str, Any]]:
        """Get only equity (stock) positions, excluding options."""
        positions = self.get_positions()
        return [p for p in positions if p.get("asset_class", "us_equity") == "us_equity"
                and "/" not in p.get("symbol", "")]

    def get_option_positions(self) -> List[Dict[str, Any]]:
        """Get only option positions."""
        positions = self.get_positions()
        return [p for p in positions
                if p.get("asset_class", "") == "us_option"
                or "/" in p.get("symbol", "")]

    # -- Market Data --

    def get_daily_bars(self, symbols: List[str], days: int = 400) -> Dict[str, pd.DataFrame]:
        """Fetch daily bars for historical vol calculation."""
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
            "%Y-%m-%dT00:00:00Z"
        )
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
                logger.error("Failed to fetch bars: %s", e)
                break

        dfs = {}
        for sym, bars in result.items():
            if not bars:
                continue
            df = pd.DataFrame(bars)
            df.rename(columns={
                "t": "timestamp", "o": "open", "h": "high",
                "l": "low", "c": "close", "v": "volume",
            }, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            for col in ["open", "high", "low", "close"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            dfs[sym] = df
        return dfs

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest trade price for a symbol."""
        try:
            r = self._session.get(
                f"{DATA_BASE}/stocks/trades/latest",
                params={"symbols": symbol, "feed": "iex"},
            )
            r.raise_for_status()
            trades = r.json().get("trades", {})
            trade = trades.get(symbol, {})
            return float(trade.get("p", 0)) or None
        except Exception:
            return None

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols."""
        prices = {}
        # Batch in groups of 50
        for i in range(0, len(symbols), 50):
            batch = symbols[i:i+50]
            try:
                r = self._session.get(
                    f"{DATA_BASE}/stocks/trades/latest",
                    params={"symbols": ",".join(batch), "feed": "iex"},
                )
                r.raise_for_status()
                trades = r.json().get("trades", {})
                for sym, trade in trades.items():
                    p = float(trade.get("p", 0))
                    if p > 0:
                        prices[sym] = p
            except Exception as e:
                logger.warning("Failed to get prices for batch: %s", e)
            time.sleep(0.1)
        return prices

    # -- Options API --

    def get_option_contracts(
        self,
        underlying: str,
        option_type: Optional[str] = None,
        expiration_gte: Optional[str] = None,
        expiration_lte: Optional[str] = None,
        strike_gte: Optional[float] = None,
        strike_lte: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Search for option contracts."""
        params: Dict[str, Any] = {
            "underlying_symbols": underlying,
            "limit": limit,
            "status": "active",
        }
        if option_type:
            params["type"] = option_type
        if expiration_gte:
            params["expiration_date_gte"] = expiration_gte
        if expiration_lte:
            params["expiration_date_lte"] = expiration_lte
        if strike_gte is not None:
            params["strike_price_gte"] = str(strike_gte)
        if strike_lte is not None:
            params["strike_price_lte"] = str(strike_lte)

        try:
            r = self._session.get(f"{PAPER_BASE}/options/contracts", params=params)
            r.raise_for_status()
            data = r.json()
            return data.get("option_contracts", data if isinstance(data, list) else [])
        except Exception as e:
            logger.error("Error fetching option contracts for %s: %s", underlying, e)
            return []

    def get_option_snapshot(self, option_symbol: str) -> Optional[Dict]:
        """Get snapshot (quote + greeks) for an option contract."""
        try:
            r = self._session.get(
                f"{OPTIONS_DATA_BASE}/snapshots/{option_symbol}",
                params={"feed": "indicative"},
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.debug("Option snapshot failed for %s: %s", option_symbol, e)
        return None

    def get_option_snapshots_bulk(self, option_symbols: List[str]) -> Dict[str, Dict]:
        """Get snapshots for multiple option symbols."""
        snapshots = {}
        # The bulk endpoint accepts comma-separated symbols
        for i in range(0, len(option_symbols), 20):
            batch = option_symbols[i:i+20]
            try:
                r = self._session.get(
                    f"{OPTIONS_DATA_BASE}/snapshots",
                    params={
                        "symbols": ",".join(batch),
                        "feed": "indicative",
                    },
                )
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, dict) and "snapshots" in data:
                        snapshots.update(data["snapshots"])
                    elif isinstance(data, dict):
                        snapshots.update(data)
            except Exception as e:
                logger.debug("Bulk option snapshot failed: %s", e)
            time.sleep(0.1)
        return snapshots

    # -- Order Placement --

    def place_option_order(
        self,
        option_symbol: str,
        qty: int,
        side: str,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
    ) -> Optional[Dict[str, Any]]:
        """Place an option order."""
        order_data: Dict[str, Any] = {
            "symbol": option_symbol,
            "qty": str(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if order_type == "limit" and limit_price is not None:
            order_data["limit_price"] = f"{limit_price:.2f}"

        try:
            r = self._session.post(f"{PAPER_BASE}/orders", json=order_data)
            if r.status_code in (200, 201):
                result = r.json()
                logger.info(
                    "Option order placed: %s %s %d @ %s (id=%s)",
                    side, option_symbol, qty,
                    f"${limit_price:.2f}" if limit_price else "market",
                    result.get("id", "?"),
                )
                return result
            else:
                logger.error(
                    "Option order failed: %s %s | %s %s",
                    side, option_symbol, r.status_code, r.text,
                )
        except Exception as e:
            logger.error("Error placing option order: %s", e)
        return None

    def wait_for_fill(self, order_id: str, timeout: int = 60) -> Optional[Dict]:
        """Wait for an order to fill."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = self._session.get(f"{PAPER_BASE}/orders/{order_id}")
                if r.status_code == 200:
                    order = r.json()
                    status = order.get("status", "")
                    if status == "filled":
                        return order
                    elif status in ("cancelled", "expired", "rejected"):
                        return None
            except Exception:
                pass
            time.sleep(2)
        # Timeout -- cancel the order
        try:
            self._session.delete(f"{PAPER_BASE}/orders/{order_id}")
        except Exception:
            pass
        return None


# =============================================================================
# MAIN OPTIONS OVERLAY CLASS
# =============================================================================

class OptionsOverlay:
    """
    Sells covered calls and cash-secured puts on swing bot positions.

    This is a standalone module that reads positions directly from Alpaca.
    The swing bot works fine without this module.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.client: Optional[AlpacaOptionsClient] = None
        self.earnings = EarningsHelper()
        self.state = OverlayState.load()

        # Market data caches
        self._prices: Dict[str, float] = {}
        self._vol_data: Dict[str, float] = {}  # realized vol per ticker
        self._bars: Dict[str, pd.DataFrame] = {}  # daily bars cache
        self._equity: float = 0.0
        self._buying_power: float = 0.0

        # Condition engine (determines which positions get overlay treatment)
        self.condition_engine: Optional[OverlayConditionEngine] = None
        self._portfolio_cond: Optional[PortfolioCondition] = None
        self._position_conds: Dict[str, PositionCondition] = {}

        # Try to connect to Alpaca
        try:
            self.client = AlpacaOptionsClient()
            logger.info("Alpaca client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Alpaca client: %s", e)
            logger.error("The overlay will run in analysis-only mode.")

    # ------------------------------------------------------------------
    # High-level run
    # ------------------------------------------------------------------

    def run(self) -> str:
        """
        Execute the full options overlay cycle.

        Returns a summary report string.
        """
        lines = []
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        mode = "DRY RUN" if self.dry_run else "LIVE"

        lines.append("=" * 70)
        lines.append(f"OPTIONS PREMIUM OVERLAY - {today_str} ({mode})")
        lines.append("=" * 70)
        lines.append("")

        if self.client is None:
            lines.append("[ERROR] No Alpaca connection. Cannot proceed.")
            return "\n".join(lines)

        # Step 0: Load market data
        try:
            self._load_market_data()
            lines.append(f"Portfolio equity: ${self._equity:,.2f}")
            lines.append(f"Buying power: ${self._buying_power:,.2f}")
            lines.append("")
        except Exception as e:
            lines.append(f"[ERROR] Failed to load market data: {e}")
            return "\n".join(lines)

        # Step 0.5: Condition engine summary
        if self._portfolio_cond is not None:
            lines.append("--- CONDITION ENGINE ---")
            lines.append(f"  {self._portfolio_cond.summary}")
            for r in self._portfolio_cond.reasons:
                lines.append(f"  * {r}")
            lines.append("")
        else:
            lines.append("--- CONDITION ENGINE ---")
            lines.append("  [WARNING] Condition engine unavailable, running without filters")
            lines.append("")

        # Step 1: Manage existing option positions
        lines.append("--- MANAGING EXISTING POSITIONS ---")
        mgmt_report = self._manage_existing_positions()
        lines.extend(mgmt_report)
        lines.append("")

        # Step 2: Sell covered calls
        lines.append("--- COVERED CALL OPPORTUNITIES ---")
        cc_report = self._run_covered_calls()
        lines.extend(cc_report)
        lines.append("")

        # Step 3: Cash-secured puts — chain-wide YTS scan (Apr 23 2026 rewrite).
        # The old "CSP DISABLED: CC seller only per David's direction" comment
        # turned out to be an agent-frozen attribution; David never made that
        # call. Replaced single-strike delta-targeting with yts_scanner.py
        # which combs the full chain across all watchlist tickers and ranks
        # globally by annualized yield-to-strike.
        lines.append("--- CASH-SECURED PUT OPPORTUNITIES (YTS scanner) ---")
        csp_report = self._run_cash_secured_puts_yts()
        lines.extend(csp_report)
        lines.append("")

        # Step 4: Earnings iron condors — chain-wide YTS pairing (Apr 23 2026).
        # Both short legs (put + call) come from the chain-wide YTS scan;
        # protective wings come from the same chain at ±IC_WING_WIDTH.
        # Ranked globally by credit-to-max-loss ratio.
        lines.append("--- EARNINGS PREMIUM HARVESTING (YTS-paired ICs) ---")
        ic_report = self._run_earnings_iron_condors_yts()
        lines.extend(ic_report)
        lines.append("")

        # Step 5: Summary
        lines.append("--- OVERLAY SUMMARY ---")
        lines.extend(self._format_summary())

        # Save state
        self.state.last_run_date = today_str
        self.state.save()

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Market data loading
    # ------------------------------------------------------------------

    def _load_market_data(self):
        """Load account info, positions, and historical vol data."""
        self._equity = self.client.get_equity()
        self._buying_power = self.client.get_buying_power()

        # Get latest prices for all swing tickers
        self._prices = self.client.get_latest_prices(SWING_TICKERS)

        # Load historical data for volatility calculation
        # Fetch 300 days for condition engine (needs 252 for IV rank)
        try:
            self._bars = self.client.get_daily_bars(SWING_TICKERS, days=300)
            for sym, df in self._bars.items():
                if not df.empty and "close" in df.columns:
                    self._vol_data[sym] = compute_realized_vol(df["close"], window=20)
        except Exception as e:
            logger.warning("Failed to load historical vol data: %s", e)

        # Initialize and load condition engine
        try:
            self.condition_engine = OverlayConditionEngine(
                client=self.client,
                earnings_helper=self.earnings,
            )
            self.condition_engine.load_data(
                tickers=SWING_TICKERS,
                prices=self._prices,
                bars=self._bars,
            )
            self._portfolio_cond, self._position_conds = (
                self.condition_engine.evaluate_all(SWING_TICKERS)
            )
            logger.info("Condition engine: %s", self._portfolio_cond.summary)
        except Exception as e:
            logger.warning("Condition engine failed to initialize: %s", e)
            logger.warning("Proceeding without condition filtering.")
            self.condition_engine = None
            self._portfolio_cond = None
            self._position_conds = {}

    def _get_vol(self, symbol: str) -> float:
        """Get realized vol for a symbol, with default fallback."""
        return self._vol_data.get(symbol, DEFAULT_VOLATILITY)

    # ------------------------------------------------------------------
    # Option selection
    # ------------------------------------------------------------------

    def select_option(
        self,
        ticker: str,
        option_type: str,
        target_delta: float = 0.30,
        min_dte: int = 14,
        max_dte: int = 28,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the best option to sell.

        Returns dict with keys: symbol, strike, expiration, dte, estimated_delta,
        estimated_premium, bid, ask, mid.
        Returns None if no suitable contract found.
        """
        price = self._prices.get(ticker)
        if not price:
            return None

        vol = self._get_vol(ticker)
        r = 0.045  # risk-free rate approximation

        # Compute the target strike using Black-Scholes
        T_mid = ((min_dte + max_dte) / 2.0) / 365.0
        ideal_strike = find_strike_for_delta(price, T_mid, r, vol, target_delta, option_type)

        # Search for contracts near the ideal strike
        exp_start = (datetime.now(timezone.utc) + timedelta(days=min_dte)).strftime("%Y-%m-%d")
        exp_end = (datetime.now(timezone.utc) + timedelta(days=max_dte)).strftime("%Y-%m-%d")

        strike_range = price * 0.15  # search +/- 15% around ideal
        if option_type == "call":
            strike_gte = ideal_strike * 0.90
            strike_lte = ideal_strike * 1.15
        else:
            strike_gte = ideal_strike * 0.85
            strike_lte = ideal_strike * 1.10

        contracts = self.client.get_option_contracts(
            underlying=ticker,
            option_type=option_type,
            expiration_gte=exp_start,
            expiration_lte=exp_end,
            strike_gte=strike_gte,
            strike_lte=strike_lte,
            limit=50,
        )

        if not contracts:
            logger.debug("No %s contracts found for %s", option_type, ticker)
            return None

        # Score each contract
        best = None
        best_score = -999

        for contract in contracts:
            strike = float(contract.get("strike_price", 0))
            expiration = contract.get("expiration_date", "")
            opt_sym = contract.get("symbol", "")

            if strike <= 0 or not expiration:
                continue

            try:
                exp_date = datetime.strptime(expiration, "%Y-%m-%d").date()
                dte = (exp_date - datetime.now(timezone.utc).date()).days
            except (ValueError, TypeError):
                continue

            if dte < min_dte or dte > max_dte:
                continue

            T = dte / 365.0
            if option_type == "call":
                delta = bs_call_delta(price, strike, T, r, vol)
                premium = bs_call_price(price, strike, T, r, vol)
            else:
                delta = bs_put_delta(price, strike, T, r, vol)
                premium = bs_put_price(price, strike, T, r, vol)

            # Try to get actual quote
            bid, ask, mid = 0.0, 0.0, premium
            snapshot = self.client.get_option_snapshot(opt_sym)
            if snapshot:
                quote = snapshot.get("latestQuote", {})
                bid = float(quote.get("bp", 0))
                ask = float(quote.get("ap", 0))
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                    # Liquidity check: bid-ask spread
                    spread_pct = (ask - bid) / mid if mid > 0 else 1.0
                    if spread_pct > MAX_BID_ASK_SPREAD_PCT:
                        logger.debug(
                            "Skipping %s: bid-ask spread %.1f%% too wide",
                            opt_sym, spread_pct * 100,
                        )
                        continue

                # Check greeks from snapshot if available
                greeks = snapshot.get("greeks", {})
                if greeks:
                    api_delta = abs(float(greeks.get("delta", 0)))
                    if api_delta > 0:
                        delta = api_delta

            # Score: closeness to target delta, premium, DTE sweetness
            delta_diff = abs(delta - target_delta)
            dte_score = 1.0 - abs(dte - 21) / 14.0  # 21 DTE is sweet spot
            premium_score = mid / price if price > 0 else 0  # premium yield

            score = -delta_diff * 10.0 + dte_score + premium_score * 100.0

            if score > best_score:
                best_score = score
                best = {
                    "symbol": opt_sym,
                    "strike": strike,
                    "expiration": expiration,
                    "dte": dte,
                    "estimated_delta": round(delta, 3),
                    "estimated_premium": round(mid, 2),
                    "bid": bid,
                    "ask": ask,
                    "mid": round(mid, 2),
                    "bs_premium": round(premium, 2),
                    "underlying_price": price,
                    "implied_vol": round(vol, 3),
                }

            time.sleep(0.05)  # Rate limiting

        return best

    # ------------------------------------------------------------------
    # Strategy 1: Covered Calls
    # ------------------------------------------------------------------

    def _select_cc_strike_via_yts(
        self,
        symbol: str,
        target_delta: float,
        min_dte: int,
        max_dte: int,
        spot: float,
    ) -> Optional[Dict[str, Any]]:
        """Pick a covered-call short strike via chain-wide YTS scan (Apr 23 2026).

        Combs the full call chain in the configured DTE band + a delta band
        centered on `target_delta` (±0.10), ranks by annualized yield-to-strike,
        and returns the top-ranked strike. Replaces the single-strike
        delta-targeting selector for CC short legs — same upgrade we did for
        CSP short puts and IC short legs. Returns None on empty result so the
        caller can fall back.

        Returns dict matching the shape of `select_option`'s return for
        drop-in compatibility with `_execute_covered_call`.
        """
        from yts_scanner import scan_ticker, ScanConfig

        # Centered delta band around target. CC default 0.30 → [0.20, 0.40].
        # Override (e.g. 0.20 from condition engine) → [0.10, 0.30].
        delta_min = max(0.05, target_delta - 0.10)
        delta_max = min(0.50, target_delta + 0.10)

        cfg = ScanConfig(
            min_dte=min_dte, max_dte=max_dte,
            delta_min=delta_min, delta_max=delta_max,
            min_open_interest=100, max_spread_pct=0.15,
            min_bid=0.05,
            # CCs run on stocks we already own — don't gate by yield floor;
            # any premium harvest beats holding the call we'd otherwise sell.
            min_yts_annualized=0.0,
        )
        try:
            cands = scan_ticker(
                self.client, symbol, "call",
                cfg=cfg, spot_price=spot,
            )
        except Exception as e:
            logger.warning("CC YTS scan failed for %s: %s", symbol, e)
            return None
        if not cands:
            return None

        top = cands[0]
        return {
            "symbol": top.option_symbol,
            "strike": top.strike,
            "expiration": top.expiration,
            "dte": top.dte,
            "estimated_delta": round(top.delta, 3),
            "estimated_premium": round(top.mid, 2),
            "bid": top.bid,
            "ask": top.ask,
            "mid": round(top.mid, 2),
            "yts_annualized": round(top.yts_annualized, 4),
            "implied_vol": top.implied_vol,
            "underlying_price": spot,
        }

    def _run_covered_calls(self) -> List[str]:
        """Find and execute covered call opportunities."""
        lines = []

        # Check portfolio-level CC gate
        if self._portfolio_cond is not None and not self._portfolio_cond.cc_enabled:
            lines.append("  Covered calls DISABLED at portfolio level:")
            for r in self._portfolio_cond.reasons:
                lines.append(f"    * {r}")
            return lines

        # ── VOL REGIME GATE (Apr 11 — GDX/GLD ratio check) ──
        try:
            import yfinance as yf
            from combined_config import VOL_RATIO_BUY_THRESHOLD
            gld_hist = yf.Ticker("GLD").history(period="30d")
            gdx_hist = yf.Ticker("GDX").history(period="30d")
            if len(gld_hist) >= 20 and len(gdx_hist) >= 20:
                import numpy as _np
                gld_r = _np.log(gld_hist['Close'] / gld_hist['Close'].shift(1)).dropna().values.flatten()
                gdx_r = _np.log(gdx_hist['Close'] / gdx_hist['Close'].shift(1)).dropna().values.flatten()
                gld_hv = _np.std(gld_r[-20:]) * _np.sqrt(252)
                gdx_hv = _np.std(gdx_r[-20:]) * _np.sqrt(252)
                vol_ratio = gdx_hv / gld_hv if gld_hv > 0 else 2.4
                lines.append(f"  Vol regime: GDX/GLD ratio={vol_ratio:.2f}x")
                if vol_ratio < VOL_RATIO_BUY_THRESHOLD:
                    lines.append(
                        f"  ⚠️ COMPLACENT REGIME (ratio {vol_ratio:.2f} < {VOL_RATIO_BUY_THRESHOLD}) "
                        f"— reducing CC aggression, consider buying puts"
                    )
                    # Don't block entirely, but log the warning
        except Exception:
            pass

        # Skip tickers with active PMCC spreads (short legs managed by PMCCManager)
        pmcc_tickers = set()
        try:
            from combined_config import PMCC_ENABLED
            if PMCC_ENABLED:
                # Check combined state for PMCC positions
                from combined_state import CombinedState
                state = CombinedState.load()
                from combined_state import TradeStage
                pmcc_tickers = {p.ticker for p in state.positions
                               if p.is_pmcc and p.stage != TradeStage.CLOSED.value}
        except Exception:
            pass

        stock_positions = self.client.get_stock_positions()
        if not stock_positions:
            lines.append("  No stock positions found.")
            return lines

        for pos in stock_positions:
            symbol = pos.get("symbol", "")
            qty = int(float(pos.get("qty", 0)))
            side = pos.get("side", "long")

            # Only long positions with round lots in our universe
            if side != "long" or qty < CC_MIN_SHARES or symbol not in SWING_TICKERS:
                continue

            # ── §5 CRASH-PAUSE GATE (Apr 21) ──
            # If the crash protocol recently flagged this ticker (>10% drop in 5d),
            # we've already closed any open short calls and we DO NOT want to re-sell
            # CCs until the pause window expires. State file is written by
            # crash_protocol.py and read here as the authoritative gate.
            try:
                import json as _json
                from datetime import datetime as _dt, timezone as _tz
                _pause_path = Path(__file__).parent / "crash_pause_state.json"
                if _pause_path.exists():
                    with open(_pause_path) as _pf:
                        _pause = _json.load(_pf)
                    _entry = _pause.get(symbol)
                    if _entry:
                        _until = _dt.fromisoformat(_entry["pause_until"])
                        if _dt.now(_tz.utc) < _until:
                            lines.append(
                                f"  {symbol}: SKIP CC — §5 crash pause until "
                                f"{_until.strftime('%Y-%m-%d %H:%M UTC')} ({_entry.get('reason','')})"
                            )
                            continue
            except Exception as _e:
                logger.warning("crash_pause_state check failed for %s: %s", symbol, _e)

            # Skip tickers with active PMCC spreads (short legs via PMCCManager)
            if symbol in pmcc_tickers:
                logger.info("Skipping CC for %s (PMCC position, short legs via PMCCManager)", symbol)
                continue

            # Check per-position condition from condition engine
            pos_cond = self._position_conds.get(symbol)
            if pos_cond is not None and not pos_cond.sell_cc:
                reason_str = "; ".join(pos_cond.reasons) if pos_cond.reasons else "condition engine"
                lines.append(f"  {symbol}: SKIP CC (condition engine: {reason_str})")
                continue

            # ── RSI14 momentum gate (Apr 15 audit fix) ──
            # Only sell CCs when RSI14 > 70 (sustained uptrend confirmed)
            # Knowledge base: "RSI14>70 alone is the best single filter (Sharpe 1.5, 94% WR)"
            try:
                import yfinance as _yf
                _hist = _yf.Ticker(symbol).history(period='30d')
                if len(_hist) >= 14:
                    _close = _hist['Close']
                    _delta = _close.diff()
                    _gain = _delta.clip(lower=0).ewm(span=14, adjust=False).mean()
                    _loss = (-_delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
                    _rsi14 = float(100 - (100 / (1 + _gain.iloc[-1] / _loss.iloc[-1])))
                    CC_RSI14_THRESHOLD = 70
                    if _rsi14 < CC_RSI14_THRESHOLD:
                        lines.append(f"  {symbol}: SKIP CC — RSI14={_rsi14:.1f} < {CC_RSI14_THRESHOLD} (waiting for uptrend)")
                        continue
                    lines.append(f"  {symbol}: RSI14={_rsi14:.1f} ≥ {CC_RSI14_THRESHOLD} ✓")
            except Exception as _e:
                logger.warning("RSI14 gate check failed for %s: %s", symbol, _e)

            # Skip if near earnings (unless intentional earnings play)
            # NOTE: also checked in condition engine, but kept as safety fallback
            if self.earnings.is_near_earnings(symbol, EARNINGS_BLOCK_DAYS):
                lines.append(f"  {symbol}: SKIP - earnings within {EARNINGS_BLOCK_DAYS} days")
                continue

            # ── TAIL RISK FILTER (Apr 11, Taleb/Spitznagel research) ──
            # Don't sell calls on tickers with α < CC_MIN_TAIL_INDEX (fattest tails)
            try:
                from combined_config import TAIL_INDEX, CC_MIN_TAIL_INDEX, CC_PREFERRED
                alpha = TAIL_INDEX.get(symbol)
                if alpha is not None and alpha < CC_MIN_TAIL_INDEX:
                    lines.append(
                        f"  {symbol}: SKIP CC - tail index α={alpha:.2f} < {CC_MIN_TAIL_INDEX} "
                        f"(fat tails, sell puts instead)"
                    )
                    continue
                # Log preference info
                if symbol in CC_PREFERRED:
                    lines.append(f"  {symbol}: ✓ CC-preferred (calls overpriced vs power law)")
            except ImportError:
                pass  # Config not updated yet, proceed without filter

            # Skip if we already have a covered call for this symbol
            existing = [p for p in self.state.positions
                        if p.underlying == symbol
                        and p.strategy == "covered_call"
                        and p.status == "open"]
            if existing:
                lines.append(f"  {symbol}: Already have CC open ({existing[0].option_symbol})")
                continue

            contracts_available = qty // 100
            price = self._prices.get(symbol)
            if not price:
                continue

            # Determine delta: use condition engine override or default
            target_delta = CC_TARGET_DELTA
            if pos_cond is not None and pos_cond.cc_delta_override is not None:
                target_delta = pos_cond.cc_delta_override
                lines.append(
                    f"  {symbol}: delta override {target_delta:.2f} "
                    f"(default {CC_TARGET_DELTA:.2f})"
                )

            # Find best call to sell — Apr 23 2026: chain-wide YTS scan
            # replaces single-strike delta-targeting. Falls back to the
            # legacy BS-derived selector if YTS returns nothing (sparse
            # chain, all snapshots stale, etc).
            option = self._select_cc_strike_via_yts(
                symbol,
                target_delta=target_delta,
                min_dte=CC_MIN_DTE,
                max_dte=CC_MAX_DTE,
                spot=price,
            )
            if not option:
                option = self.select_option(
                    symbol, "call",
                    target_delta=target_delta,
                    min_dte=CC_MIN_DTE,
                    max_dte=CC_MAX_DTE,
                )

            if not option:
                lines.append(f"  {symbol}: No suitable call contracts found (YTS+fallback)")
                continue

            premium_per_share = option["mid"]
            total_premium = premium_per_share * contracts_available * 100
            yts_str = (f" YTS={option['yts_annualized']*100:.1f}%/yr"
                       if "yts_annualized" in option else "")

            lines.append(
                f"  {symbol}: SELL {contracts_available} x {option['symbol']} "
                f"${option['strike']:.2f} call | "
                f"DTE={option['dte']} | delta={option['estimated_delta']:.2f} | "
                f"premium=${premium_per_share:.2f}/sh (${total_premium:.2f} total){yts_str}"
            )

            if not self.dry_run:
                self._execute_covered_call(symbol, option, contracts_available)
            else:
                lines.append(f"    [DRY RUN] Would sell {contracts_available} contract(s)")

        return lines

    def _execute_covered_call(self, symbol: str, option: Dict, contracts: int):
        """Execute a covered call sale."""
        sell_price = option["mid"]
        if option["bid"] > 0:
            # Sell slightly above mid to improve fill
            sell_price = round((option["mid"] + option["bid"]) / 2.0, 2)

        order = self.client.place_option_order(
            option_symbol=option["symbol"],
            qty=contracts,
            side="sell",
            order_type="limit",
            limit_price=sell_price,
        )

        if order:
            filled = self.client.wait_for_fill(order["id"], timeout=60)
            if filled:
                fill_price = float(filled.get("filled_avg_price", sell_price))
                total_premium = fill_price * contracts * 100

                pos = OptionPosition(
                    underlying=symbol,
                    option_symbol=option["symbol"],
                    option_type="call",
                    strike=option["strike"],
                    expiration=option["expiration"],
                    contracts=contracts,
                    premium_received=total_premium,
                    sell_price=fill_price,
                    sell_date=datetime.now(timezone.utc).isoformat(),
                    strategy="covered_call",
                )
                self.state.positions.append(pos)
                self.state.total_premium_collected += total_premium
                self.state.total_trades += 1
                logger.info(
                    "Covered call sold: %s %d @ $%.2f | premium=$%.2f",
                    option["symbol"], contracts, fill_price, total_premium,
                )

    # ------------------------------------------------------------------
    # Strategy 2: Cash-Secured Puts (YTS chain-wide scanner — Apr 23 2026)
    # ------------------------------------------------------------------

    def _run_cash_secured_puts_yts(self) -> List[str]:
        """Chain-wide YTS scan for CSP candidates.

        Comb the entire put chain across all eligible watchlist tickers,
        rank globally by annualized yield-to-strike, then sell the top N
        subject to portfolio caps + condition-engine + earnings gates.
        """
        from yts_scanner import scan_universe, ScanConfig

        lines: List[str] = []

        # Portfolio-level CSP gate (regime/condition engine)
        if self._portfolio_cond is not None and not self._portfolio_cond.csp_enabled:
            lines.append("  Cash-secured puts DISABLED at portfolio level:")
            for r in self._portfolio_cond.reasons:
                lines.append(f"    * {r}")
            return lines

        # Open-puts cap
        open_puts = [p for p in self.state.positions
                     if p.strategy == "cash_secured_put" and p.status == "open"]
        if len(open_puts) >= CSP_MAX_CONCURRENT:
            lines.append(f"  Max concurrent puts reached ({CSP_MAX_CONCURRENT}). Skipping.")
            return lines

        # Capital-obligation cap
        put_obligations = sum(p.strike * p.contracts * 100 for p in open_puts)
        max_obligation = self._equity * CSP_MAX_PORTFOLIO_PCT
        remaining_capacity = max_obligation - put_obligations
        if remaining_capacity <= 0:
            lines.append(
                f"  Put obligations ${put_obligations:,.0f} >= "
                f"max ${max_obligation:,.0f}. Skipping."
            )
            return lines

        # Build candidate ticker list (skip names we hold, names we already
        # have puts on, and names within earnings window).
        held_symbols = {pos.get("symbol", "")
                        for pos in self.client.get_stock_positions()}
        put_symbols = {p.underlying for p in open_puts}
        eligible: List[str] = []
        skipped_reasons: List[str] = []
        for t in SWING_TICKERS:
            if t in held_symbols or t in put_symbols:
                continue
            if self.earnings.is_near_earnings(t, EARNINGS_BLOCK_DAYS):
                skipped_reasons.append(f"  {t}: SKIP (within {EARNINGS_BLOCK_DAYS}d earnings)")
                continue
            pos_cond = self._position_conds.get(t)
            if pos_cond is not None and not pos_cond.sell_csp:
                reason = "; ".join(pos_cond.reasons) or "condition engine"
                skipped_reasons.append(f"  {t}: SKIP CSP ({reason})")
                continue
            eligible.append(t)

        lines.extend(skipped_reasons[:5])  # cap log noise
        if not eligible:
            lines.append("  No eligible candidates after earnings/holdings/condition gates.")
            return lines

        # Run the chain-wide scan
        cfg = ScanConfig(
            min_dte=CSP_MIN_DTE, max_dte=CSP_MAX_DTE,
            delta_min=CSP_DELTA_MIN, delta_max=CSP_DELTA_MAX,
            min_open_interest=100, max_spread_pct=0.15,
            min_bid=0.05, min_yts_annualized=0.18,
        )
        # Provide spot prices we already have cached so the scan bands strikes
        spot_prices = {t: self._prices.get(t) for t in eligible
                       if self._prices.get(t)}
        try:
            ranked = scan_universe(
                self.client, eligible, "put",
                cfg=cfg, spot_prices=spot_prices,
            )
        except Exception as e:
            lines.append(f"  YTS scan failed: {e}")
            logger.exception("CSP YTS scan crashed")
            return lines

        if not ranked:
            lines.append("  No contracts survived the YTS+liquidity filter "
                         f"(min YTS {cfg.min_yts_annualized*100:.0f}%/yr).")
            return lines

        lines.append(f"  Open puts: {len(open_puts)}/{CSP_MAX_CONCURRENT} | "
                     f"obligation room: ${remaining_capacity:,.0f}")
        lines.append(f"  Top YTS candidates ({len(ranked)} survived filters):")
        for i, c in enumerate(ranked[:10], 1):
            lines.append(
                f"    {i}. {c.underlying} {c.option_symbol} "
                f"K=${c.strike:.2f} {c.dte}d δ={c.delta:.2f} "
                f"IV={c.implied_vol or 0:.0%} mid=${c.mid:.2f} "
                f"YTS={c.yts_annualized*100:.1f}%/yr"
            )

        # Execute top-ranked candidates that fit the obligation budget
        slots_available = CSP_MAX_CONCURRENT - len(open_puts)
        executed = 0
        for cand in ranked:
            if executed >= slots_available:
                break
            obligation = cand.strike * 100
            if obligation > remaining_capacity:
                continue
            # One ticker, one CSP — don't stack (avoid concentration)
            if cand.underlying in put_symbols:
                continue

            lines.append(
                f"  EXECUTE: SELL 1 x {cand.option_symbol} "
                f"({cand.underlying} ${cand.strike:.2f} put, "
                f"{cand.dte}d, YTS {cand.yts_annualized*100:.1f}%/yr) | "
                f"premium=${cand.mid:.2f}/sh (${cand.mid*100:.2f} total) | "
                f"obligation=${obligation:,.0f}"
            )
            if not self.dry_run:
                option_dict = {
                    "symbol": cand.option_symbol,
                    "strike": cand.strike,
                    "expiration": cand.expiration,
                    "dte": cand.dte,
                    "estimated_delta": cand.delta,
                    "mid": cand.mid,
                    "bid": cand.bid,
                    "ask": cand.ask,
                }
                self._execute_cash_secured_put(cand.underlying, option_dict)
                remaining_capacity -= obligation
                put_symbols.add(cand.underlying)
            else:
                lines.append(f"    [DRY RUN] Would sell 1 contract")
            executed += 1

        if executed == 0:
            lines.append("  No candidates fit the obligation budget.")
        return lines

    def _run_cash_secured_puts(self) -> List[str]:
        """Find and execute cash-secured put opportunities (The Wheel).

        DEPRECATED — kept for backward compat. Production path is now
        `_run_cash_secured_puts_yts()`. Old delta-targeting + IV-ranked
        single-strike-per-ticker logic.
        """
        lines = []

        # Check portfolio-level CSP gate
        if self._portfolio_cond is not None and not self._portfolio_cond.csp_enabled:
            lines.append("  Cash-secured puts DISABLED at portfolio level:")
            for r in self._portfolio_cond.reasons:
                lines.append(f"    * {r}")
            return lines

        # Count current open puts
        open_puts = [p for p in self.state.positions
                     if p.strategy == "cash_secured_put" and p.status == "open"]
        if len(open_puts) >= CSP_MAX_CONCURRENT:
            lines.append(f"  Max concurrent puts reached ({CSP_MAX_CONCURRENT}). Skipping.")
            return lines

        # Check capital usage
        put_obligations = sum(
            p.strike * p.contracts * 100 for p in open_puts
        )
        max_obligation = self._equity * CSP_MAX_PORTFOLIO_PCT
        remaining_capacity = max_obligation - put_obligations

        if remaining_capacity <= 0:
            lines.append(
                f"  Put obligations ${put_obligations:,.0f} >= "
                f"max ${max_obligation:,.0f} (30% of equity). Skipping."
            )
            return lines

        lines.append(f"  Open puts: {len(open_puts)}/{CSP_MAX_CONCURRENT}")
        lines.append(f"  Put obligations: ${put_obligations:,.0f} / ${max_obligation:,.0f} max")
        lines.append("")

        # Find tickers we don't hold but are on the swing watchlist
        held_symbols = set()
        for pos in self.client.get_stock_positions():
            held_symbols.add(pos.get("symbol", ""))

        # Also exclude tickers we already have puts on
        put_symbols = set(p.underlying for p in open_puts)

        candidates = [t for t in SWING_TICKERS
                      if t not in held_symbols
                      and t not in put_symbols
                      and not self.earnings.is_near_earnings(t, EARNINGS_BLOCK_DAYS)]

        # Filter candidates through condition engine
        filtered_candidates = []
        for t in candidates:
            pos_cond = self._position_conds.get(t)
            if pos_cond is not None and not pos_cond.sell_csp:
                reason_str = "; ".join(pos_cond.reasons) if pos_cond.reasons else "condition engine"
                lines.append(f"  {t}: SKIP CSP (condition engine: {reason_str})")
                continue
            filtered_candidates.append(t)
        candidates = filtered_candidates

        if not candidates:
            lines.append("  No eligible put candidates (all held, near earnings, or filtered by conditions).")
            return lines

        # Score candidates by vol premium (higher RV = more premium)
        scored = []
        for ticker in candidates:
            price = self._prices.get(ticker)
            if not price:
                continue
            obligation = price * 100  # approximate (actual is strike * 100)
            if obligation > remaining_capacity:
                continue
            vol = self._get_vol(ticker)
            scored.append((ticker, vol, price))

        # Sort by vol (higher vol = more premium opportunity)
        scored.sort(key=lambda x: x[1], reverse=True)

        slots_available = CSP_MAX_CONCURRENT - len(open_puts)
        for ticker, vol, price in scored[:slots_available]:
            # Get per-position delta override if available
            pos_cond = self._position_conds.get(ticker)
            csp_delta = CSP_TARGET_DELTA
            if pos_cond is not None and pos_cond.csp_delta_override is not None:
                csp_delta = pos_cond.csp_delta_override
                lines.append(
                    f"  {ticker}: CSP delta override {csp_delta:.2f} "
                    f"(default {CSP_TARGET_DELTA:.2f})"
                )

            option = self.select_option(
                ticker, "put",
                target_delta=csp_delta,
                min_dte=CSP_MIN_DTE,
                max_dte=CSP_MAX_DTE,
            )

            if not option:
                lines.append(f"  {ticker}: No suitable put contracts found (price=${price:.2f})")
                continue

            obligation_amount = option["strike"] * 100
            if obligation_amount > remaining_capacity:
                lines.append(
                    f"  {ticker}: Put obligation ${obligation_amount:,.0f} "
                    f"exceeds remaining capacity ${remaining_capacity:,.0f}"
                )
                continue

            premium_per_share = option["mid"]
            total_premium = premium_per_share * 100  # 1 contract

            lines.append(
                f"  {ticker}: SELL 1 x {option['symbol']} "
                f"${option['strike']:.2f} put | "
                f"DTE={option['dte']} | delta={option['estimated_delta']:.2f} | "
                f"premium=${premium_per_share:.2f}/sh (${total_premium:.2f} total) | "
                f"obligation=${obligation_amount:,.0f}"
            )

            if not self.dry_run:
                self._execute_cash_secured_put(ticker, option)
                remaining_capacity -= obligation_amount
            else:
                lines.append(f"    [DRY RUN] Would sell 1 put contract")

        return lines

    def _execute_cash_secured_put(self, symbol: str, option: Dict):
        """Execute a cash-secured put sale."""
        sell_price = option["mid"]
        if option["bid"] > 0:
            sell_price = round((option["mid"] + option["bid"]) / 2.0, 2)

        order = self.client.place_option_order(
            option_symbol=option["symbol"],
            qty=1,
            side="sell",
            order_type="limit",
            limit_price=sell_price,
        )

        if order:
            filled = self.client.wait_for_fill(order["id"], timeout=60)
            if filled:
                fill_price = float(filled.get("filled_avg_price", sell_price))
                total_premium = fill_price * 100

                pos = OptionPosition(
                    underlying=symbol,
                    option_symbol=option["symbol"],
                    option_type="put",
                    strike=option["strike"],
                    expiration=option["expiration"],
                    contracts=1,
                    premium_received=total_premium,
                    sell_price=fill_price,
                    sell_date=datetime.now(timezone.utc).isoformat(),
                    strategy="cash_secured_put",
                )
                self.state.positions.append(pos)
                self.state.total_premium_collected += total_premium
                self.state.total_trades += 1
                logger.info(
                    "CSP sold: %s 1 @ $%.2f | premium=$%.2f",
                    option["symbol"], fill_price, total_premium,
                )

    # ------------------------------------------------------------------
    # Strategy 3: Earnings Premium Harvesting
    # ------------------------------------------------------------------

    def _run_earnings_iron_condors_yts(self) -> List[str]:
        """Chain-wide YTS-paired iron condors around earnings.

        Replaces the old delta-targeted single-strike build with a
        chain-wide scan: both short legs come from the YTS-ranked chain,
        wings are picked from the same chain at ±IC_WING_WIDTH. Pairs
        are then ranked globally by credit-to-max-loss ratio.

        Filters applied:
          - Universe = SWING_TICKERS within IC_DAYS_BEFORE_EARNINGS of earnings.
          - Skip names where we already hold an open earnings_ic position.
          - Per-IC: net_credit ≥ $0.30 + credit_to_max_loss ≥ 0.20.
          - Max loss per IC ≤ IC_MAX_LOSS dollars.
        """
        from yts_scanner import pair_iron_condors, ScanConfig, ICConfig

        lines: List[str] = []

        earners = self.earnings.tickers_with_earnings_soon(
            SWING_TICKERS, days=IC_DAYS_BEFORE_EARNINGS
        )
        if not earners:
            lines.append("  No earnings events within "
                         f"{IC_DAYS_BEFORE_EARNINGS}d.")
            return lines

        # Skip names with existing open ICs (don't double up)
        existing_ic_names = {
            p.underlying for p in self.state.positions
            if p.strategy == "earnings_ic" and p.status == "open"
        }
        eligible = [t for t in earners if t not in existing_ic_names]
        skipped = sorted(set(earners) - set(eligible))
        if skipped:
            lines.append(f"  SKIP (open IC already): {', '.join(skipped)}")
        if not eligible:
            lines.append("  All earnings names already have open ICs.")
            return lines

        # ICs around earnings target a tight DTE window — IV crush is the
        # whole edge, so we want short-dated and out the door fast. Delta
        # band is loosened (0.10-0.30) since both legs sit further OTM than
        # a CSP would.
        cfg = ScanConfig(
            min_dte=2, max_dte=14,
            delta_min=0.10, delta_max=0.30,
            min_open_interest=50,
            max_spread_pct=0.20,        # earnings chains widen — relax
            min_bid=0.10,
            min_yts_annualized=0.30,    # earnings premium = much fatter; raise floor
        )
        ic_cfg = ICConfig(
            wing_width=IC_WING_WIDTH,
            wing_tol=1.0,
            min_net_credit=0.30,
            min_credit_to_max_loss=0.20,
        )

        spot_prices = {t: self._prices.get(t)
                       for t in eligible if self._prices.get(t)}
        try:
            ranked = pair_iron_condors(
                self.client, eligible,
                cfg=cfg, ic_cfg=ic_cfg,
                spot_prices=spot_prices,
            )
        except Exception as e:
            lines.append(f"  IC pair scan failed: {e}")
            logger.exception("Earnings IC YTS scan crashed")
            return lines

        if not ranked:
            lines.append("  No IC pairs survived the YTS+wing+R:R filter.")
            return lines

        lines.append(f"  Top IC pairs ({len(ranked)} survived filters):")
        for i, ic in enumerate(ranked[:5], 1):
            max_loss_dollars = ic.max_loss * 100
            credit_dollars = ic.net_credit * 100
            lines.append(
                f"    {i}. {ic.underlying} exp={ic.expiration} ({ic.dte}d) | "
                f"PUT {ic.long_put_strike:.0f}/{ic.short_put.strike:.0f} | "
                f"CALL {ic.short_call.strike:.0f}/{ic.long_call_strike:.0f} | "
                f"credit=${ic.net_credit:.2f} (${credit_dollars:.0f}) | "
                f"max_loss=${ic.max_loss:.2f} (${max_loss_dollars:.0f}) | "
                f"R:R={ic.credit_to_max_loss:.2f} | "
                f"YTS_combined={ic.combined_short_yts*100:.0f}%/yr"
            )

        # Execute: top-ranked IC per ticker that fits the dollar-loss cap.
        # IC_MAX_CONTRACTS = 1 by default; one IC per ticker per cycle.
        executed = 0
        seen_tickers: set = set()
        for ic in ranked:
            if ic.underlying in seen_tickers:
                continue
            seen_tickers.add(ic.underlying)

            max_loss_dollars = ic.max_loss * 100
            if max_loss_dollars > IC_MAX_LOSS:
                lines.append(
                    f"  SKIP {ic.underlying}: max_loss ${max_loss_dollars:.0f} "
                    f"> ${IC_MAX_LOSS} cap."
                )
                continue

            lines.append(
                f"  EXECUTE: SELL IC on {ic.underlying} exp={ic.expiration} | "
                f"PUT {ic.long_put_strike:.0f}/{ic.short_put.strike:.0f} | "
                f"CALL {ic.short_call.strike:.0f}/{ic.long_call_strike:.0f} | "
                f"credit=${ic.net_credit*100:.0f} | "
                f"max_loss=${max_loss_dollars:.0f}"
            )
            if not self.dry_run:
                # Multi-leg execution stub — real wiring lives in
                # _execute_iron_condor (TBD; see Edge 134 in SHIP_LIST).
                lines.append(
                    "    [LIVE] Multi-leg IC execution not yet wired "
                    "(needs Alpaca multi-leg order support)."
                )
            else:
                lines.append("    [DRY RUN] Would sell 1 iron condor")
            executed += 1

        if executed == 0:
            lines.append("  No IC candidates passed the dollar-loss cap.")
        return lines

    def _run_earnings_iron_condors(self) -> List[str]:
        """Sell iron condors before earnings on high-IV stocks.

        DEPRECATED — kept for backward compat. Production path is now
        `_run_earnings_iron_condors_yts()` which scores both short legs
        chain-wide and ranks by credit-to-max-loss.
        """
        lines = []

        # Find tickers with earnings tomorrow (1 day out)
        earners = self.earnings.tickers_with_earnings_soon(
            SWING_TICKERS, days=IC_DAYS_BEFORE_EARNINGS
        )

        if not earners:
            lines.append("  No earnings events within 1 day.")
            return lines

        for ticker in earners:
            price = self._prices.get(ticker)
            if not price:
                continue

            vol = self._get_vol(ticker)

            # Estimate IV rank (use RV as proxy -- in production, use actual IV)
            # We approximate: if RV > 40%, IV rank is likely elevated
            iv_rank_estimate = min(100, vol / 0.50 * 100)  # rough proxy

            if iv_rank_estimate < IC_MAX_IV_RANK_THRESHOLD:
                lines.append(
                    f"  {ticker}: IV rank ~{iv_rank_estimate:.0f}% "
                    f"< {IC_MAX_IV_RANK_THRESHOLD}% threshold. Skipping."
                )
                continue

            # Check if we already have an IC on this ticker
            existing_ic = [p for p in self.state.positions
                           if p.underlying == ticker
                           and p.strategy == "earnings_ic"
                           and p.status == "open"]
            if existing_ic:
                lines.append(f"  {ticker}: Already have earnings IC open.")
                continue

            # Build iron condor legs
            r_rate = 0.045
            earn_days = self.earnings.days_until_earnings(ticker)
            if earn_days is None:
                continue

            # Find weekly expiry just after earnings (typically 2-5 DTE)
            target_dte = max(earn_days + 2, 3)

            T = target_dte / 365.0

            # Short call and put at ~16 delta
            short_call_strike = find_strike_for_delta(
                price, T, r_rate, vol, IC_CALL_DELTA, "call", strike_step=1.0
            )
            short_put_strike = find_strike_for_delta(
                price, T, r_rate, vol, IC_PUT_DELTA, "put", strike_step=1.0
            )

            # Long wings
            long_call_strike = short_call_strike + IC_WING_WIDTH
            long_put_strike = short_put_strike - IC_WING_WIDTH

            # Calculate estimated premiums
            short_call_prem = bs_call_price(price, short_call_strike, T, r_rate, vol)
            long_call_prem = bs_call_price(price, long_call_strike, T, r_rate, vol)
            short_put_prem = bs_put_price(price, short_put_strike, T, r_rate, vol)
            long_put_prem = bs_put_price(price, long_put_strike, T, r_rate, vol)

            net_credit = (short_call_prem - long_call_prem + short_put_prem - long_put_prem)
            max_loss = IC_WING_WIDTH - net_credit

            if max_loss * 100 > IC_MAX_LOSS:
                lines.append(
                    f"  {ticker}: Max loss ${max_loss * 100:.0f} > "
                    f"${IC_MAX_LOSS} limit. Skipping."
                )
                continue

            if net_credit < 0.10:
                lines.append(f"  {ticker}: Net credit ${net_credit:.2f} too low. Skipping.")
                continue

            lines.append(
                f"  {ticker} earnings IC (price=${price:.2f}, vol={vol:.0%}):"
            )
            lines.append(
                f"    SELL {short_put_strike:.0f}P / BUY {long_put_strike:.0f}P | "
                f"SELL {short_call_strike:.0f}C / BUY {long_call_strike:.0f}C"
            )
            lines.append(
                f"    Net credit: ${net_credit:.2f}/sh (${net_credit * 100:.0f} per IC) | "
                f"Max loss: ${max_loss * 100:.0f} | "
                f"DTE: ~{target_dte}"
            )

            if not self.dry_run:
                lines.append(f"    [LIVE] Iron condor execution not yet implemented "
                             f"(requires multi-leg order support)")
            else:
                lines.append(f"    [DRY RUN] Would sell 1 iron condor")

        return lines

    # ------------------------------------------------------------------
    # Position Management
    # ------------------------------------------------------------------

    def _manage_existing_positions(self) -> List[str]:
        """Manage existing option positions: profit targets, rolls, expiry."""
        lines = []

        open_positions = [p for p in self.state.positions if p.status == "open"]
        if not open_positions:
            lines.append("  No open option positions to manage.")
            return lines

        for pos in open_positions:
            dte = pos.days_to_expiration
            lines.append(
                f"  {pos.underlying} {pos.option_type.upper()} ${pos.strike:.2f} "
                f"exp {pos.expiration} (DTE={dte}) | "
                f"sold @ ${pos.sell_price:.2f} | strategy={pos.strategy}"
            )

            # Check if expired
            if dte <= 0:
                pos.status = "expired"
                pos.realized_pnl = pos.premium_received
                self.state.total_realized_pnl += pos.premium_received
                lines.append(f"    EXPIRED worthless -- kept ${pos.premium_received:.2f} premium")
                continue

            # Get current option price
            current_price = self._get_option_current_price(pos.option_symbol)
            if current_price is None:
                lines.append(f"    Could not get current price for {pos.option_symbol}")
                continue

            # Calculate profit percentage
            profit_pct = 1.0 - (current_price / pos.sell_price) if pos.sell_price > 0 else 0.0

            lines.append(
                f"    Current: ${current_price:.2f} | "
                f"Profit: {profit_pct:.0%} of premium"
            )

            # Check profit target (50%)
            if profit_pct >= CC_PROFIT_TARGET:
                lines.append(
                    f"    ** PROFIT TARGET HIT ({profit_pct:.0%}) -- "
                    f"{'closing' if not self.dry_run else 'would close'}"
                )
                if not self.dry_run:
                    self._close_position(pos, current_price, "profit_target")

            # Check roll trigger (7 DTE)
            elif dte <= CC_ROLL_DTE and profit_pct > 0:
                lines.append(
                    f"    ** NEAR EXPIRY (DTE={dte}) -- "
                    f"{'rolling' if not self.dry_run else 'would roll/close'}"
                )
                if not self.dry_run:
                    self._close_position(pos, current_price, "near_expiry_roll")

            # Check for underwater positions near expiry
            elif dte <= 3 and profit_pct < 0:
                lines.append(
                    f"    ** WARNING: underwater near expiry "
                    f"(loss={-profit_pct:.0%}, DTE={dte})"
                )

        return lines

    def _get_option_current_price(self, option_symbol: str) -> Optional[float]:
        """Get current mid price for an option."""
        snapshot = self.client.get_option_snapshot(option_symbol)
        if snapshot:
            quote = snapshot.get("latestQuote", {})
            bid = float(quote.get("bp", 0))
            ask = float(quote.get("ap", 0))
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            elif ask > 0:
                return ask
        return None

    def _close_position(self, pos: OptionPosition, current_price: float, reason: str):
        """Close an option position by buying it back."""
        buy_price = round(current_price * 1.02, 2)  # slight premium for fill

        order = self.client.place_option_order(
            option_symbol=pos.option_symbol,
            qty=pos.contracts,
            side="buy",
            order_type="limit",
            limit_price=buy_price,
        )

        if order:
            filled = self.client.wait_for_fill(order["id"], timeout=60)
            if filled:
                fill_price = float(filled.get("filled_avg_price", buy_price))
                cost = fill_price * pos.contracts * 100
                pnl = pos.premium_received - cost

                pos.status = "closed"
                pos.close_price = fill_price
                pos.close_date = datetime.now(timezone.utc).isoformat()
                pos.realized_pnl = pnl
                self.state.total_realized_pnl += pnl

                logger.info(
                    "Position closed (%s): %s | sold=$%.2f, bought=$%.2f | P&L=$%.2f",
                    reason, pos.option_symbol, pos.sell_price, fill_price, pnl,
                )

    # ------------------------------------------------------------------
    # Status report
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Generate a status report of all option positions."""
        lines = []
        lines.append("=" * 70)
        lines.append("OPTIONS OVERLAY STATUS")
        lines.append("=" * 70)
        lines.append("")

        lines.append(f"Total premium collected: ${self.state.total_premium_collected:,.2f}")
        lines.append(f"Total realized P&L: ${self.state.total_realized_pnl:,.2f}")
        lines.append(f"Total trades: {self.state.total_trades}")
        lines.append(f"Last run: {self.state.last_run_date or 'never'}")
        lines.append("")

        open_positions = [p for p in self.state.positions if p.status == "open"]
        closed_positions = [p for p in self.state.positions if p.status != "open"]

        if open_positions:
            lines.append("OPEN POSITIONS:")
            for p in open_positions:
                lines.append(
                    f"  {p.underlying} {p.option_type.upper()} ${p.strike:.2f} "
                    f"exp {p.expiration} (DTE={p.days_to_expiration}) | "
                    f"{p.contracts} contract(s) | sold @ ${p.sell_price:.2f} | "
                    f"premium=${p.premium_received:.2f} | {p.strategy}"
                )
        else:
            lines.append("OPEN POSITIONS: None")

        lines.append("")

        if closed_positions:
            lines.append(f"CLOSED POSITIONS (last 10):")
            for p in closed_positions[-10:]:
                pnl_str = f"${p.realized_pnl:.2f}" if p.realized_pnl is not None else "N/A"
                lines.append(
                    f"  {p.underlying} {p.option_type.upper()} ${p.strike:.2f} | "
                    f"status={p.status} | P&L={pnl_str} | {p.strategy}"
                )

        lines.append("=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Conditions report
    # ------------------------------------------------------------------

    def conditions_report(self) -> str:
        """
        Generate a conditions report showing why overlay is on/off for each position.

        This initializes the condition engine if needed (loads market data).
        """
        if self.client is None:
            return "[ERROR] No Alpaca connection. Cannot evaluate conditions."

        # Load data if not already loaded
        if self.condition_engine is None:
            try:
                self._load_market_data()
            except Exception as e:
                return f"[ERROR] Failed to load market data: {e}"

        if self.condition_engine is None:
            return "[ERROR] Condition engine could not be initialized."

        return self.condition_engine.format_conditions_report(SWING_TICKERS)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _format_summary(self) -> List[str]:
        """Format a summary of the overlay state."""
        lines = []
        open_cc = [p for p in self.state.positions
                   if p.strategy == "covered_call" and p.status == "open"]
        open_csp = [p for p in self.state.positions
                    if p.strategy == "cash_secured_put" and p.status == "open"]
        open_ic = [p for p in self.state.positions
                   if p.strategy == "earnings_ic" and p.status == "open"]

        lines.append(f"Open covered calls: {len(open_cc)}")
        lines.append(f"Open cash-secured puts: {len(open_csp)}/{CSP_MAX_CONCURRENT}")
        lines.append(f"Open earnings ICs: {len(open_ic)}")
        lines.append(f"Total premium collected (all time): ${self.state.total_premium_collected:,.2f}")
        lines.append(f"Total realized P&L (all time): ${self.state.total_realized_pnl:,.2f}")
        lines.append(f"Total trades: {self.state.total_trades}")

        if open_csp:
            put_obligations = sum(p.strike * p.contracts * 100 for p in open_csp)
            lines.append(f"Put obligations: ${put_obligations:,.0f} "
                         f"({put_obligations / self._equity * 100:.1f}% of equity)"
                         if self._equity > 0 else "")

        return lines


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class OptionsBacktest:
    """
    Backtest the covered call + CSP strategy using Black-Scholes estimation.

    Since historical option prices are not readily available, we estimate
    option premiums using B-S with realized volatility. This is an approximation
    and will UNDERSTATE actual premium (since IV > RV on average).
    """

    def __init__(self):
        self.client: Optional[AlpacaOptionsClient] = None
        try:
            self.client = AlpacaOptionsClient()
        except Exception:
            pass

    def run(self) -> str:
        """Run the full backtest and return a markdown report."""
        lines = []
        lines.append("# Options Overlay Backtest Results")
        lines.append("")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        lines.append(f"**Method:** Black-Scholes estimation with realized volatility")
        lines.append(f"**Period:** 1 year of daily data")
        lines.append(f"**Strategies:** Covered calls (0.30 delta) + Cash-secured puts (0.30 delta)")
        lines.append(f"**Frequency:** Sell every 2 weeks, close at 50% profit or expiry")
        lines.append("")
        lines.append("## Important Caveats")
        lines.append("")
        lines.append("1. **IV > RV gap not captured:** We use realized vol to estimate premiums, "
                      "but implied vol is typically 2-5% higher. Actual premiums would be larger.")
        lines.append("2. **No actual option price data:** These are B-S estimates, not real fills.")
        lines.append("3. **Assignment simplified:** We assume assignment at expiry if ITM, "
                      "no early assignment modeling.")
        lines.append("4. **Covered calls cap upside:** In strong bull runs, covered calls "
                      "underperform buy-and-hold because they cap gains at the strike.")
        lines.append("5. **Transaction costs:** We assume $0.65 per contract per leg.")
        lines.append("")

        # Select top 10 most liquid tickers
        test_tickers = ["NVDA", "AAPL", "MSFT", "AMZN", "META",
                        "GOOGL", "NFLX", "JPM", "GS", "WMT"]

        # Fetch 1+ year of data
        bars_data = {}
        if self.client:
            try:
                bars_data = self.client.get_daily_bars(test_tickers, days=400)
            except Exception as e:
                logger.error("Backtest data fetch failed: %s", e)

        if not bars_data:
            # Use synthetic data for demonstration
            lines.append("**Note:** Could not fetch live data. Using synthetic demonstration.")
            lines.append("")
            return self._run_synthetic_backtest(lines, test_tickers)

        return self._run_historical_backtest(lines, test_tickers, bars_data)

    def _run_historical_backtest(
        self,
        lines: List[str],
        tickers: List[str],
        bars_data: Dict[str, pd.DataFrame],
    ) -> str:
        """Run backtest with actual historical price data."""
        lines.append("## Per-Ticker Results")
        lines.append("")

        results = []
        r_rate = 0.045
        commission = 0.65  # per contract per leg

        for ticker in tickers:
            df = bars_data.get(ticker)
            if df is None or df.empty or len(df) < 260:
                lines.append(f"### {ticker}: Insufficient data (skipped)")
                lines.append("")
                continue

            # Use last 252 trading days
            df = df.tail(260).copy()
            prices = df["close"].values
            dates = df.index

            if len(prices) < 252:
                continue

            # --- Stock-only returns ---
            start_price = prices[0]
            end_price = prices[-1]
            stock_return = (end_price / start_price - 1.0) * 100
            stock_pnl = (end_price - start_price) * 100  # per 100 shares

            # --- Covered call simulation ---
            cc_premium_total = 0.0
            cc_assignments = 0
            cc_trades = 0
            cc_holding = True  # We hold 100 shares throughout
            cc_stock_pnl = stock_pnl  # start with same stock PnL

            # Every 14 trading days, sell a 0.30 delta call, 21 DTE
            i = 0
            while i < len(prices) - 21:
                if i % 14 != 0:
                    i += 1
                    continue

                current_price = prices[i]
                # Compute 20-day realized vol
                if i >= 20:
                    log_rets = np.diff(np.log(prices[max(0, i-20):i+1]))
                    vol = float(np.std(log_rets) * math.sqrt(252))
                else:
                    vol = 0.30

                vol = max(vol, 0.10)  # floor at 10%

                T = 21.0 / 365.0
                strike = find_strike_for_delta(
                    current_price, T, r_rate, vol, 0.30, "call", strike_step=1.0
                )
                premium = bs_call_price(current_price, strike, T, r_rate, vol)

                # Deduct commission
                net_premium = max(premium - commission / 100.0, 0)
                cc_premium_total += net_premium * 100  # per 100 shares
                cc_trades += 1

                # Check at "expiry" (21 days later)
                exp_idx = min(i + 21, len(prices) - 1)
                exp_price = prices[exp_idx]

                # 50% profit early close check
                closed_early = False
                for j in range(i + 1, exp_idx):
                    if j >= len(prices):
                        break
                    mid_price = prices[j]
                    mid_T = max((exp_idx - j), 1) / 365.0
                    mid_premium = bs_call_price(mid_price, strike, mid_T, r_rate, vol)
                    if mid_premium <= premium * 0.50:
                        # Close at 50% profit
                        buyback_cost = mid_premium * 100 + commission
                        # Net: received premium - buyback cost
                        # Already counted full premium, subtract the buyback
                        cc_premium_total -= mid_premium * 100 + commission
                        closed_early = True
                        break

                if not closed_early and exp_price > strike:
                    # Called away -- capped at strike
                    cap_loss = (exp_price - strike) * 100  # opportunity cost
                    cc_stock_pnl -= cap_loss  # reduce stock gains
                    cc_assignments += 1
                    # In wheel, we would sell puts to re-enter
                    # For backtest simplicity, assume we re-buy immediately
                    cc_premium_total -= commission  # re-entry commission

                i += 14

            # --- CSP simulation (simplified) ---
            csp_premium_total = 0.0
            csp_assignments = 0
            csp_trades = 0
            csp_assignment_pnl = 0.0

            # Every 14 trading days, sell a 0.30 delta put
            i = 0
            while i < len(prices) - 21:
                if i % 14 != 0:
                    i += 1
                    continue

                current_price = prices[i]
                if i >= 20:
                    log_rets = np.diff(np.log(prices[max(0, i-20):i+1]))
                    vol = float(np.std(log_rets) * math.sqrt(252))
                else:
                    vol = 0.30

                vol = max(vol, 0.10)

                T = 21.0 / 365.0
                strike = find_strike_for_delta(
                    current_price, T, r_rate, vol, 0.30, "put", strike_step=1.0
                )
                premium = bs_put_price(current_price, strike, T, r_rate, vol)

                net_premium = max(premium - commission / 100.0, 0)
                csp_premium_total += net_premium * 100
                csp_trades += 1

                exp_idx = min(i + 21, len(prices) - 1)
                exp_price = prices[exp_idx]

                # 50% profit early close
                closed_early = False
                for j in range(i + 1, exp_idx):
                    if j >= len(prices):
                        break
                    mid_price = prices[j]
                    mid_T = max((exp_idx - j), 1) / 365.0
                    mid_premium = bs_put_price(mid_price, strike, mid_T, r_rate, vol)
                    if mid_premium <= premium * 0.50:
                        cc_cost = mid_premium * 100 + commission
                        csp_premium_total -= mid_premium * 100 + commission
                        closed_early = True
                        break

                if not closed_early and exp_price < strike:
                    csp_assignments += 1
                    # Assigned: forced to buy at strike, stock is worth less
                    assignment_loss = (strike - exp_price) * 100
                    csp_assignment_pnl -= assignment_loss

                i += 14

            # Total returns
            cc_total_pnl = cc_stock_pnl + cc_premium_total
            csp_total_pnl = csp_premium_total + csp_assignment_pnl
            combined_pnl = cc_total_pnl + csp_total_pnl

            initial_investment = start_price * 100
            stock_return_pct = stock_pnl / initial_investment * 100
            cc_return_pct = cc_total_pnl / initial_investment * 100
            csp_return_pct = csp_total_pnl / initial_investment * 100 if initial_investment > 0 else 0

            result = {
                "ticker": ticker,
                "start_price": start_price,
                "end_price": end_price,
                "stock_pnl": stock_pnl,
                "stock_return_pct": stock_return_pct,
                "cc_premium": cc_premium_total,
                "cc_assignments": cc_assignments,
                "cc_trades": cc_trades,
                "cc_total_pnl": cc_total_pnl,
                "cc_return_pct": cc_return_pct,
                "csp_premium": csp_premium_total,
                "csp_assignments": csp_assignments,
                "csp_trades": csp_trades,
                "csp_total_pnl": csp_total_pnl,
                "csp_return_pct": csp_return_pct,
            }
            results.append(result)

            # Ticker report
            lines.append(f"### {ticker}")
            lines.append(f"- Price: ${start_price:.2f} -> ${end_price:.2f}")
            lines.append(f"- **Stock only:** ${stock_pnl:+,.0f} ({stock_return_pct:+.1f}%)")
            lines.append(f"- **Covered call premium:** ${cc_premium_total:,.0f} "
                         f"({cc_trades} trades, {cc_assignments} assignments)")
            lines.append(f"- **Stock + CC:** ${cc_total_pnl:+,.0f} ({cc_return_pct:+.1f}%)")
            cc_diff = cc_return_pct - stock_return_pct
            if cc_diff > 0:
                lines.append(f"  - CC added {cc_diff:+.1f}% to returns")
            else:
                lines.append(f"  - CC reduced returns by {cc_diff:.1f}% (upside capped)")
            lines.append(f"- **CSP premium:** ${csp_premium_total:,.0f} "
                         f"({csp_trades} trades, {csp_assignments} assignments, "
                         f"assignment P&L: ${csp_assignment_pnl:+,.0f})")
            lines.append(f"- **CSP net:** ${csp_total_pnl:+,.0f} ({csp_return_pct:+.1f}%)")
            lines.append("")

        # Aggregate summary
        if results:
            lines.append("## Aggregate Summary")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")

            avg_stock_ret = np.mean([r["stock_return_pct"] for r in results])
            avg_cc_ret = np.mean([r["cc_return_pct"] for r in results])
            total_cc_premium = sum(r["cc_premium"] for r in results)
            total_csp_premium = sum(r["csp_premium"] for r in results)
            total_cc_assignments = sum(r["cc_assignments"] for r in results)
            total_csp_assignments = sum(r["csp_assignments"] for r in results)
            total_cc_trades = sum(r["cc_trades"] for r in results)
            total_csp_trades = sum(r["csp_trades"] for r in results)
            avg_csp_ret = np.mean([r["csp_return_pct"] for r in results])

            lines.append(f"| Tickers tested | {len(results)} |")
            lines.append(f"| Avg stock-only return | {avg_stock_ret:+.1f}% |")
            lines.append(f"| Avg stock+CC return | {avg_cc_ret:+.1f}% |")
            lines.append(f"| CC premium boost/drag | {avg_cc_ret - avg_stock_ret:+.1f}% |")
            lines.append(f"| Total CC premium collected | ${total_cc_premium:,.0f} |")
            lines.append(f"| Total CC trades | {total_cc_trades} |")
            lines.append(f"| Total CC assignments | {total_cc_assignments} |")
            lines.append(f"| CC assignment rate | "
                         f"{total_cc_assignments / max(total_cc_trades, 1) * 100:.1f}% |")
            lines.append(f"| Total CSP premium collected | ${total_csp_premium:,.0f} |")
            lines.append(f"| Total CSP trades | {total_csp_trades} |")
            lines.append(f"| Total CSP assignments | {total_csp_assignments} |")
            lines.append(f"| CSP assignment rate | "
                         f"{total_csp_assignments / max(total_csp_trades, 1) * 100:.1f}% |")
            lines.append(f"| Avg CSP return (annualized on capital) | {avg_csp_ret:+.1f}% |")
            lines.append("")

            # Honest assessment
            lines.append("## Honest Assessment")
            lines.append("")

            bull_tickers = [r for r in results if r["stock_return_pct"] > 20]
            flat_tickers = [r for r in results if -10 <= r["stock_return_pct"] <= 20]
            bear_tickers = [r for r in results if r["stock_return_pct"] < -10]

            if bull_tickers:
                avg_bull_stock = np.mean([r["stock_return_pct"] for r in bull_tickers])
                avg_bull_cc = np.mean([r["cc_return_pct"] for r in bull_tickers])
                lines.append(f"### Strong bull tickers ({len(bull_tickers)} tickers, avg return > 20%)")
                lines.append(f"- Stock-only avg: {avg_bull_stock:+.1f}%")
                lines.append(f"- Stock+CC avg: {avg_bull_cc:+.1f}%")
                diff = avg_bull_cc - avg_bull_stock
                if diff < 0:
                    lines.append(f"- **Covered calls HURT returns by {abs(diff):.1f}%** in bull markets "
                                 f"because they cap upside at the strike price.")
                lines.append("")

            if flat_tickers:
                avg_flat_stock = np.mean([r["stock_return_pct"] for r in flat_tickers])
                avg_flat_cc = np.mean([r["cc_return_pct"] for r in flat_tickers])
                lines.append(f"### Sideways/moderate tickers ({len(flat_tickers)} tickers, -10% to +20%)")
                lines.append(f"- Stock-only avg: {avg_flat_stock:+.1f}%")
                lines.append(f"- Stock+CC avg: {avg_flat_cc:+.1f}%")
                diff = avg_flat_cc - avg_flat_stock
                if diff > 0:
                    lines.append(f"- **Covered calls ADDED {diff:.1f}%** -- "
                                 f"this is the sweet spot for the strategy.")
                lines.append("")

            if bear_tickers:
                avg_bear_stock = np.mean([r["stock_return_pct"] for r in bear_tickers])
                avg_bear_cc = np.mean([r["cc_return_pct"] for r in bear_tickers])
                lines.append(f"### Bear tickers ({len(bear_tickers)} tickers, return < -10%)")
                lines.append(f"- Stock-only avg: {avg_bear_stock:+.1f}%")
                lines.append(f"- Stock+CC avg: {avg_bear_cc:+.1f}%")
                diff = avg_bear_cc - avg_bear_stock
                lines.append(f"- Covered calls provided {diff:+.1f}% cushion, but did not prevent losses.")
                lines.append("")

            lines.append("### Key Takeaways")
            lines.append("")
            lines.append("1. **Covered calls are NOT free money.** They trade upside potential "
                         "for guaranteed premium income.")
            lines.append("2. **Best in sideways/slightly bullish markets.** The premium income "
                         "adds 3-8% annually when stocks move sideways.")
            lines.append("3. **Worst in strong bull markets.** If a stock rallies 50%, "
                         "your CC position caps you at the strike + premium.")
            lines.append("4. **CSPs generate income while waiting.** "
                         "If you are willing to own the stock at a lower price, "
                         "selling puts lets you get paid to wait.")
            lines.append("5. **The IV > RV edge is real but small.** "
                         "Expect 1-3% annualized edge from the volatility risk premium, "
                         "not a magic money printer.")
            lines.append("6. **Risk management matters.** The 50% profit target "
                         "and 7-DTE roll rule capture most of the theta decay "
                         "while avoiding gamma risk near expiry.")

        return "\n".join(lines)

    def _run_synthetic_backtest(self, lines: List[str], tickers: List[str]) -> str:
        """Run a simplified synthetic backtest when no data is available."""
        lines.append("## Synthetic Backtest (No Live Data)")
        lines.append("")
        lines.append("Using Monte Carlo simulation with typical parameters:")
        lines.append("- 252 trading days")
        lines.append("- 30% annualized volatility")
        lines.append("- 10% average annual stock return")
        lines.append("- Selling 0.30 delta calls/puts every 14 days")
        lines.append("")

        np.random.seed(42)
        num_sims = 1000
        days = 252

        # Simulate stock paths
        mu = 0.10 / 252  # daily drift
        sigma = 0.30 / math.sqrt(252)  # daily vol
        r_rate = 0.045

        stock_only_returns = []
        cc_returns = []
        csp_returns = []

        for _ in range(num_sims):
            prices = [100.0]
            for d in range(days):
                ret = mu + sigma * np.random.randn()
                prices.append(prices[-1] * math.exp(ret))

            # Stock only
            stock_ret = prices[-1] / prices[0] - 1.0

            # Covered call
            cc_premium = 0.0
            cc_capped = 0.0
            for start_day in range(0, days - 21, 14):
                S = prices[start_day]
                T = 21.0 / 365.0
                vol = 0.30
                strike = find_strike_for_delta(S, T, r_rate, vol, 0.30, "call")
                prem = bs_call_price(S, strike, T, r_rate, vol)
                cc_premium += prem
                exp_price = prices[min(start_day + 21, days)]
                if exp_price > strike:
                    cc_capped += (exp_price - strike)

            cc_total = (prices[-1] - prices[0] + cc_premium * 100 - cc_capped * 100) / prices[0] / 100

            # CSP
            csp_premium = 0.0
            csp_loss = 0.0
            for start_day in range(0, days - 21, 14):
                S = prices[start_day]
                T = 21.0 / 365.0
                vol = 0.30
                strike = find_strike_for_delta(S, T, r_rate, vol, 0.30, "put")
                prem = bs_put_price(S, strike, T, r_rate, vol)
                csp_premium += prem
                exp_price = prices[min(start_day + 21, days)]
                if exp_price < strike:
                    csp_loss += (strike - exp_price)

            csp_ret = (csp_premium - csp_loss) / prices[0]

            stock_only_returns.append(stock_ret * 100)
            cc_returns.append(stock_ret * 100 + (cc_premium - cc_capped) / prices[0] * 100)
            csp_returns.append(csp_ret * 100)

        lines.append("## Monte Carlo Results (1000 simulations)")
        lines.append("")
        lines.append("| Strategy | Mean Return | Median | Std Dev | Win Rate |")
        lines.append("|----------|------------|--------|---------|----------|")

        stock_arr = np.array(stock_only_returns)
        cc_arr = np.array(cc_returns)
        csp_arr = np.array(csp_returns)

        lines.append(f"| Stock Only | {np.mean(stock_arr):+.1f}% | "
                     f"{np.median(stock_arr):+.1f}% | {np.std(stock_arr):.1f}% | "
                     f"{np.mean(stock_arr > 0) * 100:.0f}% |")
        lines.append(f"| Stock + CC | {np.mean(cc_arr):+.1f}% | "
                     f"{np.median(cc_arr):+.1f}% | {np.std(cc_arr):.1f}% | "
                     f"{np.mean(cc_arr > 0) * 100:.0f}% |")
        lines.append(f"| CSP Only | {np.mean(csp_arr):+.1f}% | "
                     f"{np.median(csp_arr):+.1f}% | {np.std(csp_arr):.1f}% | "
                     f"{np.mean(csp_arr > 0) * 100:.0f}% |")
        lines.append("")

        cc_better = np.mean(cc_arr > stock_arr) * 100
        lines.append(f"Covered calls beat stock-only in {cc_better:.0f}% of simulations.")
        lines.append(f"CSPs are profitable in {np.mean(csp_arr > 0) * 100:.0f}% of simulations.")
        lines.append("")
        lines.append("**Note:** These results assume realized vol = implied vol. "
                     "In reality, IV typically exceeds RV by 2-5%, "
                     "which would improve option selling returns.")

        return "\n".join(lines)


# =============================================================================
# OVERLAY CONDITION ENGINE
# =============================================================================

class ConditionVerdict(str, Enum):
    """Result of the condition engine for a specific strategy on a position."""
    SELL = "SELL"
    SKIP = "SKIP"


@dataclass
class PositionCondition:
    """Per-position decision from the condition engine."""
    ticker: str
    sell_cc: bool
    sell_csp: bool
    cc_delta_override: Optional[float]    # None = use default, else override
    csp_delta_override: Optional[float]
    reasons: List[str]                    # Human-readable list of why each flag is set

    @property
    def summary(self) -> str:
        cc_str = "CC=ON" if self.sell_cc else "CC=OFF"
        csp_str = "CSP=ON" if self.sell_csp else "CSP=OFF"
        delta_str = ""
        if self.cc_delta_override is not None:
            delta_str += f" cc_delta={self.cc_delta_override:.2f}"
        if self.csp_delta_override is not None:
            delta_str += f" csp_delta={self.csp_delta_override:.2f}"
        return f"{self.ticker}: {cc_str} | {csp_str}{delta_str}"


@dataclass
class PortfolioCondition:
    """Portfolio-level on/off switch from the condition engine."""
    cc_enabled: bool
    csp_enabled: bool
    cc_delta_override: Optional[float]    # Global delta override from VIX regime
    csp_delta_override: Optional[float]
    reasons: List[str]
    regime: str
    vix_level: Optional[float]
    momentum_skip_pct: float = 0.80       # Percentile above which CC is skipped (default top 20%)

    @property
    def summary(self) -> str:
        cc_str = "CC=ENABLED" if self.cc_enabled else "CC=DISABLED"
        csp_str = "CSP=ENABLED" if self.csp_enabled else "CSP=DISABLED"
        skip_top = round((1.0 - self.momentum_skip_pct) * 100)
        return (f"Portfolio: {cc_str} | {csp_str} | regime={self.regime} "
                f"| VIX={self.vix_level or '?'} | momentum_skip=top {skip_top}%")


class OverlayConditionEngine:
    """
    Determines whether the options overlay should be active, per-position and
    portfolio-wide, based on macro regime, VIX, momentum, IV rank, earnings
    proximity, and trend strength.

    Backtest insight:
      - Covered calls HURT returns by ~4.7% on strong uptrending stocks (caps upside)
      - Covered calls HELPED by ~4.3% on sideways/weak stocks (income generation)

    So the core logic is: do NOT sell calls on stocks ripping higher; DO sell calls
    on stocks going sideways or down.
    """

    # --- VIX thresholds ---
    VIX_TOO_LOW = 15.0
    VIX_SWEET_LOW = 15.0
    VIX_SWEET_HIGH = 25.0
    VIX_ELEVATED_HIGH = 35.0

    # --- Momentum percentile thresholds ---
    MOMENTUM_TOP_PCT = 0.80     # top 20% = skip CC
    MOMENTUM_BOTTOM_PCT = 0.20  # bottom 20% = aggressive CC

    # --- IV Rank thresholds ---
    IV_RANK_LOW = 25.0
    IV_RANK_HIGH = 75.0

    # --- ADX thresholds ---
    ADX_STRONG_TREND = 30.0
    ADX_WEAK_TREND = 20.0

    def __init__(
        self,
        client: Optional["AlpacaOptionsClient"] = None,
        earnings_helper: Optional[EarningsHelper] = None,
    ):
        self.client = client
        self.earnings = earnings_helper or EarningsHelper()

        # Cached data populated by load_data()
        self._macro_regime: Optional[str] = None
        self._macro_confidence: float = 0.0
        self._vix: Optional[float] = None
        self._bars: Dict[str, pd.DataFrame] = {}
        self._prices: Dict[str, float] = {}
        self._momentum_scores: Dict[str, float] = {}      # 20-day ROC
        self._iv_ranks: Dict[str, float] = {}              # 0-100
        self._adx_values: Dict[str, float] = {}            # ADX reading
        self._adx_directions: Dict[str, str] = {}          # "positive" or "negative"

        self._cond_logger = logging.getLogger("options_overlay.conditions")

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(
        self,
        tickers: List[str],
        prices: Optional[Dict[str, float]] = None,
        bars: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> None:
        """
        Load all the data needed for condition evaluation.

        If prices/bars are already available (e.g., from OptionsOverlay._load_market_data),
        pass them in to avoid redundant API calls.
        """
        # 1. Load macro regime
        self._load_macro_regime()

        # 2. Load VIX
        self._load_vix()

        # 3. Use provided data or fetch
        if prices:
            self._prices = prices
        elif self.client:
            self._prices = self.client.get_latest_prices(tickers)

        if bars:
            self._bars = bars
        elif self.client:
            try:
                self._bars = self.client.get_daily_bars(tickers, days=300)
            except Exception as e:
                self._cond_logger.warning("Failed to fetch bars for conditions: %s", e)

        # 4. Compute per-stock indicators
        for ticker in tickers:
            df = self._bars.get(ticker)
            if df is None or df.empty or "close" not in df.columns:
                continue
            closes = df["close"]
            self._momentum_scores[ticker] = self._compute_momentum(closes)
            self._iv_ranks[ticker] = self._compute_iv_rank(closes)
            adx, direction = self._compute_adx(df)
            self._adx_values[ticker] = adx
            self._adx_directions[ticker] = direction

        self._cond_logger.info(
            "Condition engine loaded: regime=%s, VIX=%s, tickers=%d",
            self._macro_regime or "UNKNOWN",
            f"{self._vix:.1f}" if self._vix is not None else "N/A",
            len(tickers),
        )

    def _load_macro_regime(self) -> None:
        """Import and run the macro regime system."""
        try:
            from macro_regime import MacroRegimeSystem
            macro = MacroRegimeSystem()
            output = macro.run()
            self._macro_regime = output.regime
            self._macro_confidence = output.regime_confidence
            self._cond_logger.info(
                "Macro regime: %s (confidence=%.0f%%)",
                self._macro_regime, self._macro_confidence * 100,
            )
        except Exception as e:
            self._cond_logger.warning("Could not load macro regime: %s", e)
            self._macro_regime = None
            self._macro_confidence = 0.0

    def _load_vix(self) -> None:
        """Fetch current VIX level."""
        try:
            import yfinance as yf
            vix_data = yf.download("^VIX", period="5d", progress=False)
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix_data.columns = vix_data.columns.get_level_values(0)
            if not vix_data.empty:
                self._vix = float(vix_data["Close"].dropna().iloc[-1])
                self._cond_logger.info("VIX level: %.1f", self._vix)
        except Exception as e:
            self._cond_logger.warning("Could not fetch VIX: %s", e)
            # Fallback: try Alpaca
            if self.client:
                try:
                    price = self.client.get_latest_price("VIXY")
                    if price:
                        # VIXY is a proxy, not exact VIX. Log a warning.
                        self._cond_logger.info(
                            "Using VIXY price ($%.2f) as VIX proxy", price
                        )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Technical indicator computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_momentum(closes: pd.Series, period: int = 20) -> float:
        """
        20-day rate of change (ROC): (close - close_20d_ago) / close_20d_ago * 100.
        Returns percentage change.
        """
        if len(closes) < period + 1:
            return 0.0
        current = float(closes.iloc[-1])
        past = float(closes.iloc[-(period + 1)])
        if past == 0:
            return 0.0
        return (current - past) / past * 100.0

    @staticmethod
    def _compute_iv_rank(closes: pd.Series, rv_window: int = 20) -> float:
        """
        Compute IV rank proxy using realized vol percentile over 252 days.

        We use realized vol as a proxy for IV (since IV data is not always
        available). This understates the IV rank for names with elevated
        implied-to-realized vol gaps, but is a reasonable heuristic.
        """
        if len(closes) < 252:
            return 50.0  # default to middle if insufficient data

        log_returns = np.log(closes / closes.shift(1)).dropna()
        if len(log_returns) < 252:
            return 50.0

        # Rolling 20-day realized vol over the last year
        rolling_rv = log_returns.rolling(rv_window).std() * math.sqrt(252)
        rolling_rv = rolling_rv.dropna()

        if len(rolling_rv) < 20:
            return 50.0

        current_rv = float(rolling_rv.iloc[-1])
        rv_min = float(rolling_rv.min())
        rv_max = float(rolling_rv.max())

        if rv_max == rv_min:
            return 50.0

        return float((current_rv - rv_min) / (rv_max - rv_min) * 100.0)

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> Tuple[float, str]:
        """
        Compute the Average Directional Index (ADX) and determine trend direction.

        Uses the standard Wilder smoothing method:
        1. Compute TR, +DM, -DM for each bar
        2. Seed the first smoothed values as simple sums of the first `period` bars
        3. Subsequent smoothed values: prev - prev/period + current
        4. +DI = 100 * smoothed(+DM) / smoothed(TR)
        5. -DI = 100 * smoothed(-DM) / smoothed(TR)
        6. DX = 100 * |+DI - -DI| / (+DI + -DI)
        7. ADX = Wilder smooth of DX (seeded at 2*period)

        Returns (adx_value, direction) where direction is "positive" or "negative".
        """
        n = len(df)
        if n < period * 3:
            return 20.0, "neutral"  # default = no trend

        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        close = df["close"].values.astype(float)

        # Step 1: Compute raw TR, +DM, -DM (starting from index 1)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            h_l = high[i] - low[i]
            h_pc = abs(high[i] - close[i - 1])
            l_pc = abs(low[i] - close[i - 1])
            tr[i] = max(h_l, h_pc, l_pc)

            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Step 2: Wilder smoothed TR, +DM, -DM
        # Seed: sum of first `period` raw values (indices 1..period)
        smoothed_tr = np.zeros(n)
        smoothed_plus_dm = np.zeros(n)
        smoothed_minus_dm = np.zeros(n)

        smoothed_tr[period] = np.sum(tr[1:period + 1])
        smoothed_plus_dm[period] = np.sum(plus_dm[1:period + 1])
        smoothed_minus_dm[period] = np.sum(minus_dm[1:period + 1])

        for i in range(period + 1, n):
            smoothed_tr[i] = smoothed_tr[i - 1] - smoothed_tr[i - 1] / period + tr[i]
            smoothed_plus_dm[i] = smoothed_plus_dm[i - 1] - smoothed_plus_dm[i - 1] / period + plus_dm[i]
            smoothed_minus_dm[i] = smoothed_minus_dm[i - 1] - smoothed_minus_dm[i - 1] / period + minus_dm[i]

        # Step 3: +DI, -DI, DX
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)

        for i in range(period, n):
            if smoothed_tr[i] > 0:
                plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
                minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

        # Step 4: ADX = Wilder smooth of DX
        # Seed ADX at index 2*period as average of DX[period..2*period-1]
        adx_arr = np.zeros(n)
        adx_start = 2 * period
        if adx_start >= n:
            return 20.0, "neutral"

        adx_arr[adx_start] = np.mean(dx[period:adx_start + 1])
        for i in range(adx_start + 1, n):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period

        # Get the last ADX value
        adx_val = float(adx_arr[-1])
        adx_val = max(0.0, min(100.0, adx_val))

        # Direction from latest +DI vs -DI
        direction = "positive" if plus_di[-1] >= minus_di[-1] else "negative"

        return adx_val, direction

    # ------------------------------------------------------------------
    # Portfolio-level conditions (master on/off switch)
    # ------------------------------------------------------------------

    def evaluate_portfolio(self) -> PortfolioCondition:
        """
        Evaluate portfolio-level conditions.

        These act as a master on/off switch before any per-position checks.
        Based on: (1) macro regime, (2) VIX level.
        """
        cc_enabled = True
        csp_enabled = True
        cc_delta_override: Optional[float] = None
        csp_delta_override: Optional[float] = None
        reasons: List[str] = []

        # ---- Condition 1: Macro Regime ----
        regime = self._macro_regime or "UNKNOWN"

        # Soft-skip momentum threshold: in bullish regimes, widen the CC
        # skip window from the default top-20% to top-40% so more stocks
        # can still sell CCs while protecting the strongest movers.
        momentum_skip_pct = self.MOMENTUM_TOP_PCT  # default 0.80

        if regime == "RISK_ON_EXPANSION":
            cc_enabled = True
            csp_enabled = True
            momentum_skip_pct = 0.60  # top 40% skip instead of top 20%
            reasons.append(
                f"MACRO[{regime}]: CC ON (soft skip -- per-stock momentum decides, "
                f"top 40% skip window), CSP ON (accumulate on dips)"
            )
        elif regime == "LATE_CYCLE":
            cc_enabled = True
            csp_enabled = True
            reasons.append(
                f"MACRO[{regime}]: CC ON (income generation, gold-heavy portfolio "
                f"benefits from premium), CSP ON"
            )
        elif regime == "RISK_OFF_RECESSION":
            cc_enabled = True
            csp_enabled = False
            reasons.append(
                f"MACRO[{regime}]: CC ON (downside protection via premium), "
                f"CSP OFF (do not catch falling knives)"
            )
        elif regime == "RECOVERY":
            cc_enabled = True
            csp_enabled = True
            momentum_skip_pct = 0.60  # top 40% skip instead of top 20%
            reasons.append(
                f"MACRO[{regime}]: CC ON (soft skip -- per-stock momentum decides, "
                f"top 40% skip window), CSP ON (accumulate recovery names)"
            )
        else:
            # Unknown regime -- default to ON for both but note it
            reasons.append(
                f"MACRO[{regime}]: regime unknown, defaulting CC=ON, CSP=ON"
            )

        # ---- Condition 2: VIX Level ----
        if self._vix is not None:
            vix = self._vix
            if vix < self.VIX_TOO_LOW:
                cc_enabled = False
                csp_enabled = False
                reasons.append(
                    f"VIX[{vix:.1f}] < {self.VIX_TOO_LOW}: premiums too thin, "
                    f"ALL overlay OFF"
                )
            elif vix <= self.VIX_SWEET_HIGH:
                # Sweet spot -- no changes to regime decision
                reasons.append(
                    f"VIX[{vix:.1f}] in {self.VIX_SWEET_LOW}-{self.VIX_SWEET_HIGH} "
                    f"sweet spot: standard parameters"
                )
            elif vix <= self.VIX_ELEVATED_HIGH:
                # Elevated VIX: widen strikes
                cc_delta_override = 0.20  # wider = further OTM
                csp_delta_override = 0.20
                reasons.append(
                    f"VIX[{vix:.1f}] elevated ({self.VIX_SWEET_HIGH}-"
                    f"{self.VIX_ELEVATED_HIGH}): widening strikes to 20-delta"
                )
            else:
                # VIX > 35: too much gamma risk
                cc_enabled = False
                csp_enabled = False
                reasons.append(
                    f"VIX[{vix:.1f}] > {self.VIX_ELEVATED_HIGH}: extreme volatility, "
                    f"ALL overlay OFF (gamma risk, wide spreads)"
                )
        else:
            reasons.append("VIX: unavailable, no VIX-based filtering applied")

        # Log all reasons
        for r in reasons:
            self._cond_logger.info("PORTFOLIO CONDITION: %s", r)

        return PortfolioCondition(
            cc_enabled=cc_enabled,
            csp_enabled=csp_enabled,
            cc_delta_override=cc_delta_override,
            csp_delta_override=csp_delta_override,
            reasons=reasons,
            regime=regime,
            vix_level=self._vix,
            momentum_skip_pct=momentum_skip_pct,
        )

    # ------------------------------------------------------------------
    # Per-position conditions
    # ------------------------------------------------------------------

    def evaluate_position(
        self,
        ticker: str,
        portfolio_cond: PortfolioCondition,
        all_tickers: Optional[List[str]] = None,
    ) -> PositionCondition:
        """
        Evaluate per-position conditions for a given ticker.

        Checks: (1) momentum score, (2) IV rank, (3) earnings proximity,
        (4) trend strength (ADX).

        The portfolio_cond master switch is applied first -- if the portfolio
        says CC=OFF, the per-position check will not override that.
        """
        sell_cc = portfolio_cond.cc_enabled
        sell_csp = portfolio_cond.csp_enabled
        cc_delta: Optional[float] = portfolio_cond.cc_delta_override
        csp_delta: Optional[float] = portfolio_cond.csp_delta_override
        reasons: List[str] = []

        if not sell_cc and not sell_csp:
            reasons.append("Portfolio-level: both CC and CSP disabled")
            return PositionCondition(
                ticker=ticker,
                sell_cc=False,
                sell_csp=False,
                cc_delta_override=cc_delta,
                csp_delta_override=csp_delta,
                reasons=reasons,
            )

        # ---- Condition 3: Momentum (20-day ROC) ----
        momentum = self._momentum_scores.get(ticker)
        if momentum is not None and all_tickers:
            # Compute percentile rank among all tickers
            all_moms = [self._momentum_scores.get(t, 0.0) for t in all_tickers
                        if t in self._momentum_scores]
            if all_moms:
                rank = sum(1 for m in all_moms if m <= momentum) / len(all_moms)

                skip_pct = portfolio_cond.momentum_skip_pct
                skip_top_label = round((1.0 - skip_pct) * 100)

                if rank >= skip_pct:
                    # Above momentum skip threshold: let winners run
                    sell_cc = False
                    reasons.append(
                        f"MOMENTUM[{momentum:+.1f}%, rank={rank:.0%}]: "
                        f"top {skip_top_label}% -- SKIP CC (let winners run)"
                    )
                elif rank <= self.MOMENTUM_BOTTOM_PCT:
                    # Bottom 20%: sell CC aggressively at tighter strikes
                    if sell_cc:
                        cc_delta = 0.25  # closer strike = more premium
                        reasons.append(
                            f"MOMENTUM[{momentum:+.1f}%, rank={rank:.0%}]: "
                            f"bottom 20% -- CC at 25-delta (more premium to offset losses)"
                        )
                else:
                    # Middle 60%: standard
                    reasons.append(
                        f"MOMENTUM[{momentum:+.1f}%, rank={rank:.0%}]: "
                        f"middle 60% -- standard CC at 30-delta"
                    )
        elif momentum is not None:
            reasons.append(f"MOMENTUM[{momentum:+.1f}%]: no peer group for ranking")

        # ---- Condition 4: IV Rank ----
        iv_rank = self._iv_ranks.get(ticker)
        if iv_rank is not None:
            if iv_rank < self.IV_RANK_LOW:
                if sell_cc:
                    sell_cc = False
                    reasons.append(
                        f"IV_RANK[{iv_rank:.0f}] < {self.IV_RANK_LOW}: "
                        f"premiums not worth the cap -- SKIP CC"
                    )
                if sell_csp:
                    sell_csp = False
                    reasons.append(
                        f"IV_RANK[{iv_rank:.0f}] < {self.IV_RANK_LOW}: "
                        f"premiums not worth the obligation -- SKIP CSP"
                    )
            elif iv_rank <= self.IV_RANK_HIGH:
                reasons.append(
                    f"IV_RANK[{iv_rank:.0f}] in {self.IV_RANK_LOW}-{self.IV_RANK_HIGH}: "
                    f"standard parameters"
                )
            else:
                # IV Rank > 75: sell aggressively (richer premiums)
                if sell_cc and cc_delta is None:
                    cc_delta = 0.35  # slightly closer = more premium
                if sell_csp and csp_delta is None:
                    csp_delta = 0.35
                reasons.append(
                    f"IV_RANK[{iv_rank:.0f}] > {self.IV_RANK_HIGH}: "
                    f"elevated IV -- sell aggressively at 35-delta"
                )

        # ---- Condition 5: Earnings Proximity ----
        days_to_earn = self.earnings.days_until_earnings(ticker)
        if days_to_earn is not None and 0 <= days_to_earn <= EARNINGS_BLOCK_DAYS:
            sell_cc = False
            sell_csp = False
            reasons.append(
                f"EARNINGS[{days_to_earn}d away]: within {EARNINGS_BLOCK_DAYS}-day "
                f"block window -- SKIP ALL (IV crush + gap risk)"
            )

        # ---- Condition 6: Trend Strength (ADX) ----
        adx = self._adx_values.get(ticker)
        direction = self._adx_directions.get(ticker)
        if adx is not None and direction is not None:
            if adx > self.ADX_STRONG_TREND:
                if direction == "positive":
                    # Strong uptrend: do not cap upside
                    if sell_cc:
                        sell_cc = False
                        reasons.append(
                            f"ADX[{adx:.1f}]+DI=positive: strong uptrend -- "
                            f"SKIP CC (do not cap upside)"
                        )
                else:
                    # Strong downtrend: sell CC aggressively for protection
                    if sell_cc and cc_delta is None:
                        cc_delta = 0.25
                    reasons.append(
                        f"ADX[{adx:.1f}]+DI=negative: strong downtrend -- "
                        f"CC aggressively at 25-delta (premium as cushion)"
                    )
            elif adx < self.ADX_WEAK_TREND:
                # Choppy / range-bound: ideal for premium selling
                reasons.append(
                    f"ADX[{adx:.1f}] < {self.ADX_WEAK_TREND}: no trend / choppy -- "
                    f"ideal for premium selling"
                )
            else:
                reasons.append(
                    f"ADX[{adx:.1f}]: moderate trend -- standard parameters"
                )

        # Log per-position decision
        for r in reasons:
            self._cond_logger.info("POSITION CONDITION [%s]: %s", ticker, r)

        return PositionCondition(
            ticker=ticker,
            sell_cc=sell_cc,
            sell_csp=sell_csp,
            cc_delta_override=cc_delta,
            csp_delta_override=csp_delta,
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Evaluate all positions at once
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        tickers: List[str],
    ) -> Tuple[PortfolioCondition, Dict[str, PositionCondition]]:
        """
        Evaluate portfolio-level + per-position conditions for all tickers.

        Returns (portfolio_condition, {ticker: position_condition}).
        """
        portfolio_cond = self.evaluate_portfolio()
        position_conds: Dict[str, PositionCondition] = {}

        for ticker in tickers:
            position_conds[ticker] = self.evaluate_position(
                ticker, portfolio_cond, all_tickers=tickers,
            )

        return portfolio_cond, position_conds

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def format_conditions_report(
        self,
        tickers: List[str],
    ) -> str:
        """
        Generate a human-readable report of all conditions for the given tickers.

        This is the output of the --conditions CLI flag.
        """
        portfolio_cond, position_conds = self.evaluate_all(tickers)

        lines = []
        lines.append("=" * 74)
        lines.append("OPTIONS OVERLAY CONDITION ENGINE REPORT")
        lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("=" * 74)
        lines.append("")

        # Portfolio-level
        lines.append("--- PORTFOLIO-LEVEL CONDITIONS (MASTER SWITCH) ---")
        lines.append(f"  Macro Regime : {portfolio_cond.regime} "
                     f"(confidence={self._macro_confidence:.0%})")
        lines.append(f"  VIX Level    : {portfolio_cond.vix_level:.1f}"
                     if portfolio_cond.vix_level is not None else
                     "  VIX Level    : unavailable")
        lines.append(f"  Covered Calls: {'ENABLED' if portfolio_cond.cc_enabled else 'DISABLED'}")
        lines.append(f"  CSPs         : {'ENABLED' if portfolio_cond.csp_enabled else 'DISABLED'}")
        if portfolio_cond.momentum_skip_pct != 0.80:
            skip_top = round((1.0 - portfolio_cond.momentum_skip_pct) * 100)
            lines.append(f"  Momentum Skip: top {skip_top}% (widened from default 20%)")
        if portfolio_cond.cc_delta_override:
            lines.append(f"  CC Delta Override: {portfolio_cond.cc_delta_override:.2f}")
        if portfolio_cond.csp_delta_override:
            lines.append(f"  CSP Delta Override: {portfolio_cond.csp_delta_override:.2f}")
        lines.append("")
        for r in portfolio_cond.reasons:
            lines.append(f"  * {r}")
        lines.append("")

        # Per-position
        lines.append("--- PER-POSITION CONDITIONS ---")
        lines.append("")

        # Summary table
        lines.append(f"  {'Ticker':<8} {'CC':>5} {'CSP':>5} {'Momentum':>10} "
                     f"{'IV Rank':>8} {'ADX':>6} {'Earnings':>10} {'Delta':>7}")
        lines.append(f"  {'------':<8} {'---':>5} {'---':>5} {'--------':>10} "
                     f"{'-------':>8} {'---':>6} {'--------':>10} {'-----':>7}")

        for ticker in sorted(tickers):
            cond = position_conds.get(ticker)
            if cond is None:
                continue

            mom = self._momentum_scores.get(ticker)
            iv = self._iv_ranks.get(ticker)
            adx = self._adx_values.get(ticker)
            earn_days = self.earnings.days_until_earnings(ticker)

            cc_str = "ON" if cond.sell_cc else "OFF"
            csp_str = "ON" if cond.sell_csp else "OFF"
            mom_str = f"{mom:+.1f}%" if mom is not None else "N/A"
            iv_str = f"{iv:.0f}" if iv is not None else "N/A"
            adx_str = f"{adx:.0f}" if adx is not None else "N/A"
            earn_str = f"{earn_days}d" if earn_days is not None else "N/A"
            delta_str = (f"{cond.cc_delta_override:.2f}"
                         if cond.cc_delta_override is not None else "std")

            lines.append(
                f"  {ticker:<8} {cc_str:>5} {csp_str:>5} {mom_str:>10} "
                f"{iv_str:>8} {adx_str:>6} {earn_str:>10} {delta_str:>7}"
            )

        lines.append("")

        # Detailed reasons per position
        lines.append("--- DETAILED REASONS ---")
        lines.append("")
        for ticker in sorted(tickers):
            cond = position_conds.get(ticker)
            if cond is None:
                continue
            lines.append(f"  {cond.summary}")
            for r in cond.reasons:
                lines.append(f"    - {r}")
            lines.append("")

        # Summary counts
        cc_on = sum(1 for c in position_conds.values() if c.sell_cc)
        cc_off = sum(1 for c in position_conds.values() if not c.sell_cc)
        csp_on = sum(1 for c in position_conds.values() if c.sell_csp)
        csp_off = sum(1 for c in position_conds.values() if not c.sell_csp)

        lines.append("--- SUMMARY ---")
        lines.append(f"  Covered calls: {cc_on} ON / {cc_off} OFF out of {len(tickers)} tickers")
        lines.append(f"  CSPs: {csp_on} ON / {csp_off} OFF out of {len(tickers)} tickers")
        lines.append("=" * 74)

        return "\n".join(lines)


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(level: str = "INFO"):
    """Configure logging with console and rotating file."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    log_file = LOG_DIR / f"options_overlay_{today_str}.log"
    fh = logging.handlers.RotatingFileHandler(
        str(log_file), maxBytes=10 * 1024 * 1024, backupCount=5,
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Options premium selling overlay for the swing trading bot. "
                    "Sells covered calls, cash-secured puts, and earnings iron condors.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Show what the overlay would do without placing orders (default)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Actually place orders (override --dry-run)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show current option positions and overlay state",
    )
    parser.add_argument(
        "--backtest", action="store_true",
        help="Run historical backtest of the options overlay strategies",
    )
    parser.add_argument(
        "--conditions", action="store_true",
        help="Show current overlay conditions for all positions "
             "(macro regime, VIX, momentum, IV rank, ADX, earnings)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    if args.backtest:
        logger.info("Running options overlay backtest...")
        bt = OptionsBacktest()
        report = bt.run()
        print(report)

        # Save backtest report
        research_dir = BASE_DIR / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        report_path = research_dir / "options_overlay_backtest.md"
        with open(report_path, "w") as f:
            f.write(report)
        print(f"\nBacktest report saved to {report_path}")
        return

    dry_run = not args.live
    overlay = OptionsOverlay(dry_run=dry_run)

    if args.conditions:
        print(overlay.conditions_report())
    elif args.status:
        print(overlay.status())
    else:
        report = overlay.run()
        print(report)


if __name__ == "__main__":
    main()
