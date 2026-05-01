#!/usr/bin/env python3
"""
fundamental_filter.py - Fundamental quality gate for the trading bot.

Screens every ticker before any technical signal can trigger a trade.
Uses Yahoo Finance (yfinance) for fundamental data with a 24-hour JSON cache.

Hard filters (instant reject):
  - Chinese ADR / VIE structures (maintained blacklist)
  - Negative free cash flow (trailing 12 months) -- SKIPPED for financials
  - Debt/equity > 2.0 -- SKIPPED for financials
  - Market cap < $5 billion
  - No earnings in last 4 quarters

Soft score -- GENERAL (0-100, need >= 50):
  - Revenue growth YoY (0-20 pts)
  - EPS growth YoY (0-20 pts)
  - FCF yield (0-15 pts)
  - ROE (0-15 pts)
  - Interest coverage (0-15 pts)
  - Insider buying (0-15 pts)

Soft score -- FINANCIAL SECTOR (0-100, need >= 50):
  - Return on equity (0-25 pts)  -- banks live and die by ROE
  - Revenue growth YoY (0-20 pts)
  - EPS growth YoY (0-20 pts)
  - Price/Book ratio (0-15 pts)  -- standard bank valuation metric
  - Dividend yield (0-10 pts)    -- shareholder return quality
  - Net insider activity (0-10 pts)

ETFs and commodity products bypass screening entirely.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("fundamental_filter")

# =============================================================================
# CONFIGURABLE THRESHOLDS
# =============================================================================

# Hard filter thresholds
MIN_MARKET_CAP = 1_000_000_000          # $1 billion — David's universe includes junior miners
MAX_DEBT_EQUITY = 2.0                   # Debt/equity ceiling (skip for financials)
MIN_FCF = 0                             # Free cash flow must be non-negative
MIN_QUARTERLY_EARNINGS = 4              # Need earnings in last 4 quarters

# Soft score minimum to pass
MIN_SOFT_SCORE = 35                     # Lowered — hand-picked universe, don't over-filter

# Revenue growth scoring tiers (YoY %)
REVENUE_GROWTH_TIERS = [
    (0.20, 20),   # >20% = 20 pts
    (0.10, 15),   # >10% = 15 pts
    (0.05, 10),   # >5%  = 10 pts
    (0.00, 5),    # >0%  =  5 pts
]

# EPS growth scoring tiers (YoY %)
EPS_GROWTH_TIERS = [
    (0.20, 20),
    (0.10, 15),
    (0.05, 10),
    (0.00, 5),
]

# FCF yield scoring tiers
FCF_YIELD_TIERS = [
    (0.08, 15),   # >8%  = 15 pts
    (0.05, 12),   # >5%  = 12 pts
    (0.03, 8),    # >3%  =  8 pts
    (0.00, 4),    # >0%  =  4 pts
]

# ROE scoring tiers
ROE_TIERS = [
    (0.20, 15),   # >20% = 15 pts
    (0.15, 12),   # >15% = 12 pts
    (0.10, 8),    # >10% =  8 pts
    (0.05, 4),    # >5%  =  4 pts
]

# Interest coverage scoring tiers
INTEREST_COVERAGE_TIERS = [
    (10.0, 15),   # >10x = 15 pts
    (5.0,  12),   # >5x  = 12 pts
    (3.0,  8),    # >3x  =  8 pts
    (1.5,  4),    # >1.5x=  4 pts
]

# ---- Financial-sector scoring tiers ----
# ROE for financials (0-25 pts) -- higher weight because ROE is the core bank metric
FINANCIAL_ROE_TIERS = [
    (0.15, 25),   # >15% = 25 pts
    (0.12, 20),   # >12% = 20 pts
    (0.10, 15),   # >10% = 15 pts
    (0.08, 10),   # >8%  = 10 pts
    (0.05, 5),    # >5%  =  5 pts
]

# Price/Book for financials (0-15 pts)
FINANCIAL_PB_TIERS = [
    # Lower P/B is better for banks (scored in reverse)
    # Handled in _compute_financial_soft_score with custom logic
]

# Dividend yield for financials (0-10 pts)
FINANCIAL_DIVIDEND_TIERS = [
    (0.04, 10),   # >4% = 10 pts
    (0.03, 8),    # >3% =  8 pts
    (0.02, 6),    # >2% =  6 pts
    (0.01, 3),    # >1% =  3 pts
]

# Cache settings
CACHE_TTL_SECONDS = 24 * 60 * 60       # 24 hours
CACHE_FILE = Path(__file__).parent / "fundamental_cache.json"

# =============================================================================
# BLACKLISTS AND EXEMPTIONS
# =============================================================================

CHINESE_ADR_BLACKLIST = {
    "BABA", "PDD", "JD", "NIO", "XPEV", "LI", "BIDU", "TME", "BILI",
    "ZTO", "VNET", "IQ", "FUTU", "TAL", "EDU", "KC", "DIDI", "MNSO",
    "GDS", "WB", "TUYA", "YMM",
}

# ETF_TICKERS imported from combined_config (single source of truth, Mar 16)
try:
    from combined_config import ETF_TICKERS
except ImportError:
    # Fallback if running standalone
    ETF_TICKERS = {
        "GLD", "SLV", "GDX", "SPY", "QQQ", "IWM",
        "XLK", "XLE", "XLF", "XLV", "XLB", "XLI", "XLU", "XLP", "XLY",
        "TLT", "HYG", "EEM", "DIA", "VTI", "VOO",
    }

FINANCIAL_SECTORS = {"finance", "financial_services", "financial", "banks"}

# Sector map used to identify financial tickers (skip D/E check)
FINANCIAL_TICKERS = {"GS", "WFC", "JPM"}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FilterResult:
    """Result of running the fundamental filter on a single ticker."""
    ticker: str
    passed: bool
    hard_reject: bool = False
    hard_reject_reason: str = ""
    soft_score: int = 0
    score_breakdown: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    data_error: bool = False
    exempted: bool = False

    @property
    def summary(self) -> str:
        if self.exempted:
            return f"{self.ticker}: EXEMPT (ETF/commodity)"
        if self.data_error:
            return f"{self.ticker}: DATA ERROR (fail-open) -- {'; '.join(self.warnings)}"
        if self.hard_reject:
            return f"{self.ticker}: HARD REJECT -- {self.hard_reject_reason}"
        status = "PASS" if self.passed else "FAIL"
        return f"{self.ticker}: {status} (score={self.soft_score}/100) -- {self.score_breakdown}"


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

class FundamentalCache:
    """Simple JSON file cache with per-ticker TTL."""

    def __init__(self, cache_file: Path = CACHE_FILE, ttl: int = CACHE_TTL_SECONDS):
        self.cache_file = cache_file
        self.ttl = ttl
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Cache file corrupt or unreadable, starting fresh: %s", e)
                self._data = {}

    def _save(self) -> None:
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._data, f, indent=2)
        except IOError as e:
            logger.warning("Failed to save cache: %s", e)

    def get(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return cached data if present and not expired, else None."""
        entry = self._data.get(ticker)
        if entry is None:
            return None
        cached_at = entry.get("_cached_at", 0)
        if time.time() - cached_at > self.ttl:
            return None
        return entry

    def put(self, ticker: str, data: Dict[str, Any]) -> None:
        """Store data with a timestamp."""
        data["_cached_at"] = time.time()
        self._data[ticker] = data
        self._save()

    def put_batch(self, batch: Dict[str, Dict[str, Any]]) -> None:
        """Store multiple tickers at once (single write)."""
        now = time.time()
        for ticker, data in batch.items():
            data["_cached_at"] = now
            self._data[ticker] = data
        self._save()


# =============================================================================
# YAHOO FINANCE DATA FETCHER
# =============================================================================

def _fetch_fundamental_data(tickers: List[str], cache: FundamentalCache) -> Dict[str, Dict[str, Any]]:
    """
    Fetch fundamental data for a list of tickers from Yahoo Finance.
    Uses cache where available; fetches fresh data for the rest.
    Returns a dict of ticker -> fundamental data dict.
    """
    results: Dict[str, Dict[str, Any]] = {}
    to_fetch: List[str] = []

    # Check cache first
    for ticker in tickers:
        cached = cache.get(ticker)
        if cached is not None:
            results[ticker] = cached
            logger.debug("Cache hit for %s", ticker)
        else:
            to_fetch.append(ticker)

    if not to_fetch:
        return results

    logger.info("Fetching fundamental data for %d tickers from Yahoo Finance: %s",
                len(to_fetch), ", ".join(to_fetch))

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed. Run: pip install yfinance")
        for ticker in to_fetch:
            results[ticker] = {"_error": "yfinance not installed"}
        return results

    # Fetch all tickers using yfinance Tickers batch API for speed
    batch_data: Dict[str, Dict[str, Any]] = {}
    for ticker in to_fetch:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            financials = t.financials
            quarterly_financials = t.quarterly_financials
            cashflow = t.cashflow
            balance_sheet = t.balance_sheet

            data: Dict[str, Any] = {}

            # Market cap
            data["market_cap"] = info.get("marketCap", 0) or 0

            # Sector (for financial classification)
            data["sector"] = (info.get("sector", "") or "").lower().replace(" ", "_")

            # Revenue growth YoY
            data["revenue_growth"] = info.get("revenueGrowth")

            # EPS growth YoY
            data["earnings_growth"] = info.get("earningsGrowth")

            # Trailing EPS and forward EPS for fallback growth calc
            data["trailing_eps"] = info.get("trailingEps")
            data["forward_eps"] = info.get("forwardEps")

            # Free cash flow
            data["free_cash_flow"] = info.get("freeCashflow", 0) or 0

            # FCF yield = FCF / market_cap
            if data["market_cap"] and data["market_cap"] > 0 and data["free_cash_flow"]:
                data["fcf_yield"] = data["free_cash_flow"] / data["market_cap"]
            else:
                data["fcf_yield"] = None

            # Debt to equity
            data["debt_to_equity"] = info.get("debtToEquity")
            if data["debt_to_equity"] is not None:
                # yfinance returns D/E as a percentage (e.g., 150 = 1.5x)
                data["debt_to_equity"] = data["debt_to_equity"] / 100.0

            # ROE
            data["roe"] = info.get("returnOnEquity")

            # Price/Book ratio (used by financial-sector scoring)
            data["price_to_book"] = info.get("priceToBook")

            # Dividend yield (used by financial-sector scoring)
            data["dividend_yield"] = info.get("dividendYield")

            # Interest coverage (operating income / interest expense)
            data["interest_coverage"] = None
            if financials is not None and not financials.empty:
                try:
                    # Look for operating income and interest expense
                    op_income = None
                    interest_expense = None
                    for label in ["Operating Income", "EBIT", "Ebit"]:
                        if label in financials.index:
                            val = financials.loc[label].iloc[0]
                            if val is not None and not (hasattr(val, '__float__') and val != val):
                                op_income = float(val)
                                break
                    for label in ["Interest Expense", "Interest Expense Non Operating",
                                  "Net Interest Income"]:
                        if label in financials.index:
                            val = financials.loc[label].iloc[0]
                            if val is not None and not (hasattr(val, '__float__') and val != val):
                                interest_expense = abs(float(val))
                                break
                    if op_income is not None and interest_expense and interest_expense > 0:
                        data["interest_coverage"] = op_income / interest_expense
                except Exception as e:
                    logger.debug("Interest coverage calc failed for %s: %s", ticker, e)

            # Quarterly net income (check last 4 quarters have earnings)
            data["quarterly_net_income"] = []
            if quarterly_financials is not None and not quarterly_financials.empty:
                for label in ["Net Income", "Net Income Common Stockholders"]:
                    if label in quarterly_financials.index:
                        vals = quarterly_financials.loc[label].dropna().tolist()
                        data["quarterly_net_income"] = [float(v) for v in vals[:4]]
                        break

            # Insider transactions
            data["insider_net_buying"] = None
            try:
                insider_tx = t.insider_transactions
                if insider_tx is not None and not insider_tx.empty:
                    # Look at last 6 months of transactions
                    now = datetime.now(timezone.utc)
                    six_months_ago = now.timestamp() - (180 * 24 * 60 * 60)

                    net_shares = 0
                    for _, row in insider_tx.iterrows():
                        tx_text = str(row.get("Text", "")).lower()
                        shares = row.get("Shares", 0) or 0
                        if isinstance(shares, str):
                            shares = int(shares.replace(",", "").replace("+", ""))
                        if "purchase" in tx_text or "buy" in tx_text:
                            net_shares += abs(shares)
                        elif "sale" in tx_text or "sell" in tx_text:
                            net_shares -= abs(shares)

                    if net_shares > 0:
                        data["insider_net_buying"] = "buying"
                    elif net_shares < 0:
                        data["insider_net_buying"] = "selling"
                    else:
                        data["insider_net_buying"] = "flat"
            except Exception as e:
                logger.debug("Insider data fetch failed for %s: %s", ticker, e)

            batch_data[ticker] = data
            results[ticker] = data

        except Exception as e:
            logger.warning("Failed to fetch data for %s: %s", ticker, e)
            error_data = {"_error": str(e)}
            batch_data[ticker] = error_data
            results[ticker] = error_data

    # Save batch to cache
    if batch_data:
        cache.put_batch(batch_data)

    return results


# =============================================================================
# SCORING FUNCTIONS
# =============================================================================

def _score_tier(value: Optional[float], tiers: List[Tuple[float, int]]) -> int:
    """Score a value against descending threshold tiers. Returns points."""
    if value is None:
        return 0
    for threshold, points in tiers:
        if value > threshold:
            return points
    return 0


def _compute_soft_score(data: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    """
    Compute the soft quality score (0-100) from fundamental data.
    Returns (total_score, breakdown_dict).
    """
    breakdown: Dict[str, int] = {}

    # Revenue growth YoY (0-20 pts)
    rev_growth = data.get("revenue_growth")
    breakdown["revenue_growth"] = _score_tier(rev_growth, REVENUE_GROWTH_TIERS)

    # EPS growth YoY (0-20 pts)
    eps_growth = data.get("earnings_growth")
    breakdown["eps_growth"] = _score_tier(eps_growth, EPS_GROWTH_TIERS)

    # FCF yield (0-15 pts)
    fcf_yield = data.get("fcf_yield")
    if fcf_yield is not None and fcf_yield < 0:
        breakdown["fcf_yield"] = 0
    else:
        breakdown["fcf_yield"] = _score_tier(fcf_yield, FCF_YIELD_TIERS)

    # ROE (0-15 pts)
    roe = data.get("roe")
    breakdown["roe"] = _score_tier(roe, ROE_TIERS)

    # Interest coverage (0-15 pts)
    interest_cov = data.get("interest_coverage")
    breakdown["interest_coverage"] = _score_tier(interest_cov, INTEREST_COVERAGE_TIERS)

    # Insider buying (0-15 pts)
    insider = data.get("insider_net_buying")
    if insider == "buying":
        breakdown["insider_buying"] = 15
    elif insider == "flat":
        breakdown["insider_buying"] = 8
    else:  # selling or None
        breakdown["insider_buying"] = 0

    total = sum(breakdown.values())
    return total, breakdown


def _compute_financial_soft_score(data: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    """
    Compute the soft quality score (0-100) for financial-sector stocks.
    Banks have unusual financials (low FCF yield, no traditional interest
    coverage, high D/E by design) so they get their own rubric:

      - Return on equity (0-25 pts)
      - Revenue growth YoY (0-20 pts)  -- same tiers as general
      - EPS growth YoY (0-20 pts)      -- same tiers as general
      - Price/Book ratio (0-15 pts)
      - Dividend yield (0-10 pts)
      - Net insider activity (0-10 pts)

    Returns (total_score, breakdown_dict).
    """
    breakdown: Dict[str, int] = {}

    # ROE (0-25 pts) -- the single most important bank profitability metric
    roe = data.get("roe")
    breakdown["roe"] = _score_tier(roe, FINANCIAL_ROE_TIERS)

    # Revenue growth YoY (0-20 pts) -- same tiers as general scoring
    rev_growth = data.get("revenue_growth")
    breakdown["revenue_growth"] = _score_tier(rev_growth, REVENUE_GROWTH_TIERS)

    # EPS growth YoY (0-20 pts) -- same tiers as general scoring
    eps_growth = data.get("earnings_growth")
    breakdown["eps_growth"] = _score_tier(eps_growth, EPS_GROWTH_TIERS)

    # Price/Book ratio (0-15 pts) -- lower is better for banks
    pb = data.get("price_to_book")
    if pb is not None:
        if pb < 1.0:
            breakdown["price_to_book"] = 15
        elif pb < 1.5:
            breakdown["price_to_book"] = 12
        elif pb < 2.0:
            breakdown["price_to_book"] = 8
        elif pb < 3.0:
            breakdown["price_to_book"] = 4
        else:
            breakdown["price_to_book"] = 0
    else:
        breakdown["price_to_book"] = 0

    # Dividend yield (0-10 pts)
    div_yield = data.get("dividend_yield")
    breakdown["dividend_yield"] = _score_tier(div_yield, FINANCIAL_DIVIDEND_TIERS)

    # Net insider activity (0-10 pts) -- scaled version of general insider scoring
    insider = data.get("insider_net_buying")
    if insider == "buying":
        breakdown["insider_buying"] = 10
    elif insider == "flat":
        breakdown["insider_buying"] = 5
    else:  # selling or None
        breakdown["insider_buying"] = 0

    total = sum(breakdown.values())
    return total, breakdown


# =============================================================================
# MAIN FILTER FUNCTION
# =============================================================================

def check_fundamental(
    ticker: str,
    cache: Optional[FundamentalCache] = None,
    preloaded_data: Optional[Dict[str, Any]] = None,
) -> FilterResult:
    """
    Run the full fundamental quality filter on a single ticker.

    Args:
        ticker: Stock symbol.
        cache: Optional shared cache instance.
        preloaded_data: Optional pre-fetched data dict (skips Yahoo fetch).

    Returns:
        FilterResult with pass/fail, score, and reasons.
    """
    result = FilterResult(ticker=ticker, passed=True)

    # --- ETF/commodity exemption ---
    if ticker in ETF_TICKERS:
        result.exempted = True
        result.passed = True
        return result

    # --- Chinese ADR blacklist ---
    if ticker in CHINESE_ADR_BLACKLIST:
        result.hard_reject = True
        result.hard_reject_reason = "Chinese ADR / VIE structure (blacklisted)"
        result.passed = False
        return result

    # --- Fetch data ---
    if preloaded_data is not None:
        data = preloaded_data
    else:
        if cache is None:
            cache = FundamentalCache()
        fetched = _fetch_fundamental_data([ticker], cache)
        data = fetched.get(ticker, {})

    # --- Check for data errors (fail open) ---
    if "_error" in data:
        result.data_error = True
        result.passed = True  # Fail open
        result.warnings.append(f"Yahoo data unavailable: {data['_error']}")
        logger.warning("FUNDAMENTAL WARNING: %s -- data unavailable, allowing trade (fail-open)", ticker)
        return result

    # --- Hard filter: Market cap ---
    market_cap = data.get("market_cap", 0)
    if market_cap and market_cap < MIN_MARKET_CAP:
        result.hard_reject = True
        result.hard_reject_reason = f"Market cap ${market_cap / 1e9:.1f}B < ${MIN_MARKET_CAP / 1e9:.0f}B minimum"
        result.passed = False
        return result

    # --- Detect financial sector ---
    is_financial = (
        ticker in FINANCIAL_TICKERS
        or data.get("sector", "") in FINANCIAL_SECTORS
    )

    # --- Hard filter: Negative FCF (skip for financials) ---
    if not is_financial:
        fcf = data.get("free_cash_flow", 0)
        if fcf is not None and fcf < 0:
            result.hard_reject = True
            result.hard_reject_reason = f"Negative free cash flow (${fcf / 1e6:.0f}M trailing 12mo)"
            result.passed = False
            return result

    # --- Hard filter: Debt/equity (skip for financials) ---
    de_ratio = data.get("debt_to_equity")
    if de_ratio is not None and not is_financial:
        if de_ratio > MAX_DEBT_EQUITY:
            result.hard_reject = True
            result.hard_reject_reason = f"Debt/equity {de_ratio:.2f} > {MAX_DEBT_EQUITY:.1f} max"
            result.passed = False
            return result

    # --- Hard filter: No earnings in last 4 quarters ---
    quarterly_ni = data.get("quarterly_net_income", [])
    if len(quarterly_ni) >= MIN_QUARTERLY_EARNINGS:
        # Check if ALL of the last 4 quarters have positive net income
        all_positive = all(ni > 0 for ni in quarterly_ni[:MIN_QUARTERLY_EARNINGS])
        if not all_positive:
            # Check if at least some quarters have earnings (not all negative)
            any_positive = any(ni > 0 for ni in quarterly_ni[:MIN_QUARTERLY_EARNINGS])
            if not any_positive:
                result.hard_reject = True
                result.hard_reject_reason = "No positive earnings in last 4 quarters"
                result.passed = False
                return result
    elif len(quarterly_ni) == 0:
        # No quarterly data available -- warn but don't block
        result.warnings.append("No quarterly earnings data available")

    # --- Soft score (financial-sector stocks get their own rubric) ---
    if is_financial:
        total_score, breakdown = _compute_financial_soft_score(data)
    else:
        total_score, breakdown = _compute_soft_score(data)
    result.soft_score = total_score
    result.score_breakdown = breakdown

    if total_score < MIN_SOFT_SCORE:
        result.passed = False
        result.hard_reject = False  # Not a hard reject, just failed the score
    else:
        result.passed = True

    return result


def check_fundamentals_batch(
    tickers: List[str],
    cache: Optional[FundamentalCache] = None,
) -> Dict[str, FilterResult]:
    """
    Run fundamental filter on multiple tickers efficiently.
    Pre-fetches all data in one pass, then scores each ticker.

    Args:
        tickers: List of ticker symbols.
        cache: Optional shared cache instance.

    Returns:
        Dict of ticker -> FilterResult.
    """
    if cache is None:
        cache = FundamentalCache()

    results: Dict[str, FilterResult] = {}

    # Separate exempted and blacklisted tickers (no data fetch needed)
    need_data: List[str] = []
    for ticker in tickers:
        if ticker in ETF_TICKERS:
            results[ticker] = FilterResult(
                ticker=ticker, passed=True, exempted=True,
            )
        elif ticker in CHINESE_ADR_BLACKLIST:
            results[ticker] = FilterResult(
                ticker=ticker, passed=False, hard_reject=True,
                hard_reject_reason="Chinese ADR / VIE structure (blacklisted)",
            )
        else:
            need_data.append(ticker)

    # Fetch all fundamental data at once
    if need_data:
        all_data = _fetch_fundamental_data(need_data, cache)
        for ticker in need_data:
            data = all_data.get(ticker, {"_error": "No data returned"})
            results[ticker] = check_fundamental(ticker, cache=cache, preloaded_data=data)

    return results


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

# Module-level cache singleton (initialized on first use)
_global_cache: Optional[FundamentalCache] = None


def get_cache() -> FundamentalCache:
    """Get or create the module-level cache singleton."""
    global _global_cache
    if _global_cache is None:
        _global_cache = FundamentalCache()
    return _global_cache


def should_trade(ticker: str) -> Tuple[bool, str]:
    """
    Quick check for integration into signal pipelines.
    Returns (allowed: bool, reason: str).

    If fundamental_filter_enabled is False in combined_config, always returns True.
    """
    try:
        from combined_config import fundamental_filter_enabled
        if not fundamental_filter_enabled:
            return True, "Fundamental filter disabled"
    except (ImportError, AttributeError):
        pass

    cache = get_cache()
    result = check_fundamental(ticker, cache=cache)

    if result.exempted:
        return True, "ETF/commodity exempt"

    if result.data_error:
        logger.warning(
            "FUNDAMENTAL WARNING: %s -- data unavailable, allowing trade (fail-open)",
            ticker,
        )
        return True, f"Data unavailable (fail-open): {'; '.join(result.warnings)}"

    if not result.passed:
        if result.hard_reject:
            reason = f"FUNDAMENTAL REJECT: {ticker} -- {result.hard_reject_reason}"
        else:
            reason = (
                f"FUNDAMENTAL REJECT: {ticker} -- soft score {result.soft_score}/100 "
                f"< {MIN_SOFT_SCORE} minimum ({result.score_breakdown})"
            )
        logger.info(reason)
        return False, reason

    return True, f"Passed (score={result.soft_score}/100)"
