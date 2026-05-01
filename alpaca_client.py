"""
alpaca_client.py - Alpaca Markets API client for market data and paper trading.

Provides:
  - Historical bars (daily, intraday 1min/5min)
  - Latest quotes, trades, and snapshots
  - Account info and order management via paper trading endpoint
  - Options contract lookup (for covered call fallback strategy)

Authentication uses API key + secret stored in Settings Portal.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAPER_BASE = "https://paper-api.alpaca.markets/v2"
DATA_BASE = "https://data.alpaca.markets/v2"

# Data freshness thresholds (seconds)
STALE_WARN_SECONDS = 60    # warn if quote/trade is older than this
STALE_SKIP_SECONDS = 300   # skip the signal entirely if data is older than this

# IEX data gap detection thresholds
# IEX free feed can have missing bars.  If actual bar count falls below this
# fraction of the expected count we flag the data so callers can skip or warn.
IEX_GAP_WARN_RATIO = 0.70   # warn if < 70% of expected bars
IEX_GAP_SKIP_RATIO = 0.40   # treat as unreliable if < 40% of expected bars

# Expected bars per full trading day by timeframe (6.5 hour session)
_EXPECTED_BARS_PER_DAY = {
    "1Min":  390,   # 6.5h * 60
    "5Min":  78,    # 6.5h * 12
    "15Min": 26,    # 6.5h * 4
    "1Hour": 7,     # roughly 6.5 hours
    "1Day":  1,
}


# ---------------------------------------------------------------------------
# Data freshness helper
# ---------------------------------------------------------------------------

def is_data_fresh(
    timestamp_str: str,
    max_age_seconds: float = STALE_WARN_SECONDS,
) -> bool:
    """
    Check whether a quote/trade timestamp is fresh enough to act on.

    Args:
        timestamp_str: ISO-8601 timestamp from the Alpaca API (e.g.
            "2026-03-07T14:30:00.123Z" or with +00:00 offset).
        max_age_seconds: Consider data stale if older than this many seconds.

    Returns:
        True if the data is fresh (age <= max_age_seconds), False otherwise.
    """
    if not timestamp_str:
        return False
    try:
        trade_dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - trade_dt).total_seconds()
        if age > STALE_SKIP_SECONDS:
            logger.warning(
                "Data is %.0f seconds old (>%ds) -- TOO STALE, trade signal "
                "should be skipped",
                age, STALE_SKIP_SECONDS,
            )
            return False
        if age > STALE_WARN_SECONDS:
            logger.warning(
                "Data is %.0f seconds old (>%ds) -- stale warning",
                age, STALE_WARN_SECONDS,
            )
        return age <= max_age_seconds
    except (ValueError, TypeError) as exc:
        logger.warning("Could not parse timestamp %r: %s", timestamp_str, exc)
        return False


# ---------------------------------------------------------------------------
# IEX data gap detection  (IMPORTANT-12)
# ---------------------------------------------------------------------------

def check_iex_bar_gaps(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    trading_days_expected: int = 0,
) -> Dict[str, Any]:
    """
    Detect missing bars in IEX data by comparing actual bar count to expected.

    IEX (free feed) frequently has gaps -- missing 1-min or 5-min bars --
    because it only reports bars where there was IEX volume.  If a stock
    does not trade on IEX for a 5-min window, that bar is simply absent.

    This causes TA indicators (EMAs, RSI, VWAP) to produce incorrect values,
    which can generate false trade signals.

    Args:
        df: DataFrame of bars (index = timestamp).
        symbol: Ticker symbol (for logging).
        timeframe: One of "1Min", "5Min", "15Min", "1Hour", "1Day".
        trading_days_expected: Optional hint for how many trading days the
            request was meant to cover.  If 0, the function estimates from
            the DataFrame's date range (weekdays only).

    Returns:
        Dict with keys:
          - actual_bars: int
          - expected_bars: int
          - coverage_ratio: float (0.0-1.0)
          - gap_level: "OK" | "WARN" | "SKIP"
          - trading_days_actual: int
          - gap_days: list of date strings where bars were <50% of expected
          - message: human-readable summary
    """
    result: Dict[str, Any] = {
        "actual_bars": 0,
        "expected_bars": 0,
        "coverage_ratio": 1.0,
        "gap_level": "OK",
        "trading_days_actual": 0,
        "gap_days": [],
        "message": "",
    }

    if df is None or df.empty:
        result["gap_level"] = "SKIP"
        result["message"] = f"{symbol}: No bars returned from IEX"
        logger.warning(result["message"])
        return result

    actual_bars = len(df)
    result["actual_bars"] = actual_bars

    bars_per_day = _EXPECTED_BARS_PER_DAY.get(timeframe, 78)

    # Determine how many trading days are in the data
    if df.index.tz is not None:
        dates = df.index.tz_convert("US/Eastern").normalize().unique()
    else:
        dates = df.index.normalize().unique()
    trading_days_actual = len(dates)
    result["trading_days_actual"] = trading_days_actual

    # Use the hint if provided, otherwise use what we found
    td = trading_days_expected if trading_days_expected > 0 else trading_days_actual
    expected_bars = td * bars_per_day
    result["expected_bars"] = expected_bars

    if expected_bars <= 0:
        return result

    coverage = actual_bars / expected_bars
    result["coverage_ratio"] = round(coverage, 3)

    # Per-day gap analysis (only for intraday timeframes)
    if bars_per_day > 1:
        for d in dates:
            if df.index.tz is not None:
                day_mask = df.index.tz_convert("US/Eastern").normalize() == d
            else:
                day_mask = df.index.normalize() == d
            day_count = int(day_mask.sum())
            day_expected = bars_per_day
            if day_count < day_expected * 0.50:
                day_str = str(d.date()) if hasattr(d, "date") else str(d)[:10]
                result["gap_days"].append(day_str)

    # Classify
    if coverage < IEX_GAP_SKIP_RATIO:
        result["gap_level"] = "SKIP"
        result["message"] = (
            f"{symbol} [{timeframe}]: IEX data severely gapped -- "
            f"{actual_bars}/{expected_bars} bars ({coverage:.0%} coverage). "
            f"TA signals are UNRELIABLE. Skipping."
        )
        logger.warning(result["message"])
    elif coverage < IEX_GAP_WARN_RATIO:
        result["gap_level"] = "WARN"
        result["message"] = (
            f"{symbol} [{timeframe}]: IEX data has gaps -- "
            f"{actual_bars}/{expected_bars} bars ({coverage:.0%} coverage). "
            f"TA signals may be degraded."
        )
        logger.warning(result["message"])
    else:
        result["gap_level"] = "OK"
        result["message"] = (
            f"{symbol} [{timeframe}]: IEX data OK -- "
            f"{actual_bars}/{expected_bars} bars ({coverage:.0%} coverage)."
        )
        logger.debug(result["message"])

    if result["gap_days"]:
        logger.info(
            "%s: days with >50%% missing bars: %s",
            symbol, ", ".join(result["gap_days"]),
        )

    return result


# ---------------------------------------------------------------------------
# Alpaca client
# ---------------------------------------------------------------------------

# === Edge 123 root fix: session that auto-refreshes credentials on 401 ===
# Symptom (Apr 17 19:47 ET): paper-key was rotated server-side; cached
# session headers continued sending stale key for 48h until manual restart.
# Fix: subclass Session to detect 401, re-fetch credentials from the portal
# via the registered refresh callback, and retry the request once.
class _AutoRefreshSession(requests.Session):
    def __init__(self, refresh_callback: Callable[["_AutoRefreshSession"], None]) -> None:
        super().__init__()
        self._refresh_callback = refresh_callback
        self._refresh_lock = threading.Lock()
        self._last_refresh_ts = 0.0  # monotonic seconds

    def request(self, method, url, **kwargs):  # type: ignore[override]
        r = super().request(method, url, **kwargs)
        if r.status_code != 401:
            return r
        # 401 — try one credential refresh + retry. Lock so concurrent 401s
        # don't all refresh; debounce repeat refreshes within 30s to avoid
        # hammering the portal.
        with self._refresh_lock:
            now = time.monotonic()
            if now - self._last_refresh_ts >= 30.0:
                logger.warning("Alpaca 401 on %s %s — refreshing credentials", method, url)
                try:
                    self._refresh_callback(self)
                    self._last_refresh_ts = now
                except Exception as e:
                    logger.error("Credential refresh failed: %s", e)
                    return r
            else:
                logger.warning("Alpaca 401 on %s %s — refresh debounced (last refresh %.1fs ago)",
                               method, url, now - self._last_refresh_ts)
        # Retry once with refreshed headers
        return super().request(method, url, **kwargs)
# === end Edge 123 root fix ===


class AlpacaClient:
    """
    Thin wrapper around the Alpaca REST API.

    Designed for market data retrieval and paper trading operations.
    All methods return raw Python dicts or pandas DataFrames.
    """

    def __init__(self, api_key: str, secret_key: str) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        }
        self._session = _AutoRefreshSession(self._refresh_session_credentials)
        self._session.headers.update(self._headers)

    # === Edge 123 root fix: portal-fetch + header-rewrite on 401 ===
    def _refresh_session_credentials(self, session: requests.Session) -> None:
        """Re-pull Alpaca credentials from the portal and update session headers.

        Called by _AutoRefreshSession when a 401 is observed. Safe to call
        repeatedly; the session is mutated in place.
        """
        import sys
        sys.path.insert(0, "/opt/jarvis-utils/lib")
        from jarvis_utils.secrets import get
        new_key = get("Alpaca", "api_key_id", user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        new_secret = get("Alpaca", "secret_key", user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        if not new_key or not new_secret:
            raise EnvironmentError("Edge 123 refresh: portal returned empty Alpaca creds")
        self._api_key = new_key
        self._secret_key = new_secret
        self._headers["APCA-API-KEY-ID"] = new_key
        self._headers["APCA-API-SECRET-KEY"] = new_secret
        session.headers["APCA-API-KEY-ID"] = new_key
        session.headers["APCA-API-SECRET-KEY"] = new_secret
        logger.info("Edge 123: Alpaca credentials refreshed (key prefix %s...)", new_key[:6])
    # === end Edge 123 root fix ===

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> Dict[str, Any]:
        """Return paper trading account details."""
        r = self._session.get(f"{PAPER_BASE}/account")
        r.raise_for_status()
        return r.json()

    def get_positions(self) -> List[Dict[str, Any]]:
        """Return all open positions."""
        r = self._session.get(f"{PAPER_BASE}/positions")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Market data - Bars
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 10000,
        adjustment: str = "split",
        feed: str = "iex",
        check_gaps: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical bars for one or more symbols.

        When *check_gaps* is True (default) and feed is "iex", each symbol's
        returned DataFrame is validated against expected bar counts.  The gap
        analysis result is attached to the DataFrame as ``df.attrs["iex_gap"]``
        so callers can inspect it (e.g. skip symbols at "SKIP" level).

        Args:
            symbols: List of ticker symbols
            timeframe: "1Min", "5Min", "15Min", "1Hour", "1Day", "1Week", "1Month"
            start: ISO 8601 datetime string
            end: ISO 8601 datetime string
            limit: Maximum bars per request (up to 10000)
            adjustment: "raw", "split", "dividend", "all"
            feed: "iex" (free) or "sip" (requires Algo Trader Plus subscription)
            check_gaps: Run IEX bar gap detection (default True for iex feed)

        Returns:
            Dict mapping symbol -> DataFrame with columns:
            [timestamp, open, high, low, close, volume, vwap, trade_count]
        """
        result: Dict[str, List[Dict]] = {s: [] for s in symbols}
        page_token = None

        while True:
            params: Dict[str, Any] = {
                "symbols": ",".join(symbols),
                "timeframe": timeframe,
                "limit": limit,
                "adjustment": adjustment,
                "feed": feed,
            }
            if start:
                params["start"] = start
            if end:
                params["end"] = end
            if page_token:
                params["page_token"] = page_token

            r = self._session.get(f"{DATA_BASE}/stocks/bars", params=params)
            r.raise_for_status()
            data = r.json()

            for sym in symbols:
                bars = data.get("bars", {}).get(sym, [])
                result[sym].extend(bars)

            page_token = data.get("next_page_token")
            if not page_token:
                break

            # Rate limit protection
            time.sleep(0.2)

        # Convert to DataFrames
        dfs: Dict[str, pd.DataFrame] = {}
        for sym, bars in result.items():
            if not bars:
                dfs[sym] = pd.DataFrame()
                continue

            df = pd.DataFrame(bars)
            df.rename(columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "trade_count",
            }, inplace=True)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            for col in ["open", "high", "low", "close", "vwap"]:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            if "volume" in df.columns:
                df["volume"] = df["volume"].astype(int)
            dfs[sym] = df

        # IEX data gap detection (IMPORTANT-12)
        if check_gaps and feed == "iex" and timeframe in _EXPECTED_BARS_PER_DAY:
            for sym, df in dfs.items():
                gap_info = check_iex_bar_gaps(df, sym, timeframe)
                # Attach metadata to the DataFrame so callers can inspect it
                df.attrs["iex_gap"] = gap_info

        return dfs

    def get_daily_bars(
        self,
        symbols: List[str],
        days: int = 180,
    ) -> Dict[str, pd.DataFrame]:
        """Convenience: fetch N calendar days of daily bars."""
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT23:59:59Z")
        return self.get_bars(symbols, timeframe="1Day", start=start, end=end)

    def get_intraday_bars(
        self,
        symbols: List[str],
        timeframe: str = "1Min",
        days: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """Convenience: fetch N days of intraday bars."""
        start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00Z")
        end = datetime.now(timezone.utc).strftime("%Y-%m-%dT23:59:59Z")
        return self.get_bars(symbols, timeframe=timeframe, start=start, end=end)

    # ------------------------------------------------------------------
    # Market data - Snapshots / Latest
    # ------------------------------------------------------------------

    def get_snapshots(
        self,
        symbols: List[str],
        feed: str = "iex",
        check_freshness: bool = True,
    ) -> Dict[str, Dict]:
        """
        Get latest snapshot (daily bar, prev bar, minute bar, latest trade/quote).

        When *check_freshness* is True, each symbol's latest-trade timestamp
        is inspected.  Symbols whose data is older than STALE_SKIP_SECONDS are
        dropped from the returned dict so the caller never trades on stale data.
        """
        r = self._session.get(
            f"{DATA_BASE}/stocks/snapshots",
            params={"symbols": ",".join(symbols), "feed": feed},
        )
        r.raise_for_status()
        data = r.json()

        if check_freshness:
            fresh: Dict[str, Dict] = {}
            for sym, snap in data.items():
                trade_ts = snap.get("latestTrade", {}).get("t", "")
                if trade_ts and not is_data_fresh(trade_ts, STALE_SKIP_SECONDS):
                    logger.warning(
                        "Dropping snapshot for %s -- data too stale (ts=%s)",
                        sym, trade_ts,
                    )
                    continue
                # Still warn if moderately stale
                if trade_ts:
                    is_data_fresh(trade_ts, STALE_WARN_SECONDS)
                fresh[sym] = snap
            return fresh

        return data

    def get_latest_trades(self, symbols: List[str], feed: str = "iex") -> Dict[str, Dict]:
        """Get the most recent trade for each symbol.

        Trades older than STALE_SKIP_SECONDS are omitted and a warning is logged.
        """
        r = self._session.get(
            f"{DATA_BASE}/stocks/trades/latest",
            params={"symbols": ",".join(symbols), "feed": feed},
        )
        r.raise_for_status()
        trades = r.json().get("trades", {})
        fresh: Dict[str, Dict] = {}
        for sym, trade in trades.items():
            ts = trade.get("t", "")
            if ts and not is_data_fresh(ts, STALE_SKIP_SECONDS):
                logger.warning("Dropping latest trade for %s -- too stale (ts=%s)", sym, ts)
                continue
            if ts:
                is_data_fresh(ts, STALE_WARN_SECONDS)
            fresh[sym] = trade
        return fresh

    def get_latest_quotes(self, symbols: List[str], feed: str = "iex") -> Dict[str, Dict]:
        """Get the most recent quote for each symbol.

        Quotes older than STALE_SKIP_SECONDS are omitted and a warning is logged.
        """
        r = self._session.get(
            f"{DATA_BASE}/stocks/quotes/latest",
            params={"symbols": ",".join(symbols), "feed": feed},
        )
        r.raise_for_status()
        quotes = r.json().get("quotes", {})
        fresh: Dict[str, Dict] = {}
        for sym, quote in quotes.items():
            ts = quote.get("t", "")
            if ts and not is_data_fresh(ts, STALE_SKIP_SECONDS):
                logger.warning("Dropping latest quote for %s -- too stale (ts=%s)", sym, ts)
                continue
            if ts:
                is_data_fresh(ts, STALE_WARN_SECONDS)
            fresh[sym] = quote
        return fresh

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def get_options_contracts(
        self,
        underlying_symbol: str,
        expiration_gte: Optional[str] = None,
        expiration_lte: Optional[str] = None,
        option_type: Optional[str] = None,
        strike_price_gte: Optional[float] = None,
        strike_price_lte: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """
        Search for options contracts.

        Args:
            underlying_symbol: e.g. "WPM"
            expiration_gte/lte: Date bounds (YYYY-MM-DD)
            option_type: "call" or "put"
            strike_price_gte/lte: Strike price bounds
            limit: Max contracts to return

        Returns:
            List of option contract dicts.
        """
        params: Dict[str, Any] = {
            "underlying_symbols": underlying_symbol,
            "limit": limit,
        }
        if expiration_gte:
            params["expiration_date_gte"] = expiration_gte
        if expiration_lte:
            params["expiration_date_lte"] = expiration_lte
        if option_type:
            params["type"] = option_type
        if strike_price_gte is not None:
            params["strike_price_gte"] = str(strike_price_gte)
        if strike_price_lte is not None:
            params["strike_price_lte"] = str(strike_price_lte)

        r = self._session.get(f"{PAPER_BASE}/options/contracts", params=params)
        r.raise_for_status()
        data = r.json()
        return data.get("option_contracts", data if isinstance(data, list) else [])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_alpaca_client() -> AlpacaClient:
    """
    Create an AlpacaClient using credentials from Settings Portal.

    Edge 123 root fix (Apr 19): prefer ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``
    env vars when set. The combined_bot.sh wrapper now refreshes these per cycle
    via portal lookups, so a long-running supervisor never operates on stale
    creds. Portal fallback remains for non-wrapper invocations (preflight,
    one-off scripts, tests).
    """
    import os
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")

    if not api_key or not secret:
        import sys
        sys.path.insert(0, "/opt/jarvis-utils/lib")
        from jarvis_utils.secrets import get
        api_key = get("Alpaca", "api_key_id", user="a4dc8459-608d-49f5-943e-e5e105ed5207")
        secret = get("Alpaca", "secret_key", user="a4dc8459-608d-49f5-943e-e5e105ed5207")

    if not api_key or not secret:
        raise EnvironmentError("Alpaca API credentials not found in env or Settings Portal.")

    return AlpacaClient(api_key, secret)
