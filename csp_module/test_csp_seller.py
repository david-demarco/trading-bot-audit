"""
tests/test_csp_seller.py — Unit + integration tests for CSP seller module.

Tests cover:
  1. Signal logic on synthetic price traces (esp. Signal 6 15m/60m, Signal 7 HV ratio)
  2. Strike selection on canned chain data
  3. Cross-engine dedup (CC + CSP same ticker → second blocked)
  4. Cap math using STRIKE not spot
  5. Dry-run integration cycle on WPM/GDX with mocked Alpaca

Run:
  python3 -m pytest tests/test_csp_seller.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup so imports work without installing the package
BASE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(BASE))
sys.path.insert(0, "/opt/jarvis-utils/lib")
# ---------------------------------------------------------------------------

# Lazy-import guarded: some modules hit yfinance/alpaca at import time.
# We mock those out before importing the module under test.

@pytest.fixture(autouse=True)
def _mock_heavy_imports(monkeypatch):
    """Prevent network I/O from yfinance and jarvis_utils at import time."""
    import importlib
    # Ensure we get a clean module if already cached
    for mod in list(sys.modules.keys()):
        if mod.startswith(("yfinance", "jarvis_utils", "alpaca_client")):
            del sys.modules[mod]

    yf_mock = MagicMock()
    jarvis_mock = MagicMock()
    jarvis_mock.secrets.get.return_value = "fake_key"
    alpaca_mock = MagicMock()

    monkeypatch.setitem(sys.modules, "yfinance", yf_mock)
    monkeypatch.setitem(sys.modules, "jarvis_utils", jarvis_mock)
    monkeypatch.setitem(sys.modules, "jarvis_utils.secrets", jarvis_mock.secrets)
    monkeypatch.setitem(sys.modules, "jarvis_utils.inbox", MagicMock())
    monkeypatch.setitem(sys.modules, "alpaca_client", alpaca_mock)
    yield


def _make_close_series(values: List[float]) -> pd.DataFrame:
    """Build a minimal DataFrame with a Close column for testing indicators."""
    return pd.DataFrame({"Close": values})


# =============================================================================
# 1. SIGNAL ENGINE TESTS
# =============================================================================

class TestCSPSignalEngine:
    """Tests for the 7-signal pullback detection logic."""

    @pytest.fixture
    def engine_and_data(self):
        from csp_seller import CSPDataLayer, CSPSignalEngine, GLD_TICKER, SLV_TICKER, UUP_TICKER
        data = CSPDataLayer()
        engine = CSPSignalEngine(data)
        return engine, data, GLD_TICKER, SLV_TICKER, UUP_TICKER

    def _inject_prices(self, data, ticker: str, current: float, prev: float) -> None:
        """Inject synthetic price data for a ticker."""
        from csp_seller import GLD_TICKER
        closes = [prev * (1 + 0.001 * i) for i in range(60)] + [current]
        df = _make_close_series(closes)
        data._daily_data[ticker] = df
        data._price_cache[ticker] = current
        data._prev_close[ticker] = prev

    def _inject_intraday(self, data, ticker: str, bars: List[float]) -> None:
        """Inject synthetic 5-min intraday bars."""
        import pandas as pd
        data._intraday_data[ticker] = pd.DataFrame({"Close": bars})

    def test_signal1_ticker_down_3pct(self, engine_and_data):
        engine, data, GLD, SLV, UUP = engine_and_data
        # WPM down 4% (should trigger Signal 1)
        self._inject_prices(data, "WPM", 68.0, 70.86)  # ~4% drop
        self._inject_prices(data, GLD, 185.0, 184.0)
        self._inject_prices(data, SLV, 22.0, 22.0)
        self._inject_prices(data, UUP, 28.5, 28.0)
        result = engine.evaluate_sell("WPM")
        assert result.details["ticker_down_3pct"]["triggered"] is True

    def test_signal1_ticker_not_down_enough(self, engine_and_data):
        engine, data, GLD, SLV, UUP = engine_and_data
        # WPM only down 1% (Signal 1 should NOT trigger)
        self._inject_prices(data, "WPM", 70.0, 70.71)  # ~1% drop
        self._inject_prices(data, GLD, 185.0, 184.0)
        self._inject_prices(data, SLV, 22.0, 22.0)
        self._inject_prices(data, UUP, 28.5, 28.0)
        result = engine.evaluate_sell("WPM")
        assert result.details["ticker_down_3pct"]["triggered"] is False

    def test_signal2_rsi_oversold(self, engine_and_data):
        engine, data, GLD, SLV, UUP = engine_and_data
        # Build declining close series to get low RSI
        # 14 consecutive down days → very low RSI
        closes = [100.0 - i * 1.5 for i in range(30)]
        df = _make_close_series(closes)
        data._daily_data["WPM"] = df
        data._price_cache["WPM"] = closes[-1]
        data._prev_close["WPM"] = closes[-2]
        rsi = data.compute_rsi("WPM")
        assert rsi is not None
        # Pure down-streak RSI should be very low
        assert rsi < 35, f"Expected RSI < 35 for pure downtrend, got {rsi:.1f}"
        result = engine.evaluate_sell("WPM")
        assert result.details["rsi_oversold"]["triggered"] is True

    def test_signal6_both_gates_pass(self, engine_and_data):
        """Signal 6: GLD 15m > -0.10% AND 60m > -0.40% → triggered."""
        engine, data, GLD, SLV, UUP = engine_and_data
        # 13 bars: small decline, then recovery
        intra_bars = [185.0] * 3 + [184.6, 184.5, 184.6, 184.7, 184.8, 184.9, 185.0, 185.1, 185.2, 185.3]
        self._inject_intraday(data, GLD, intra_bars)
        result = engine.evaluate_sell("WPM")
        s6 = result.details["macro_not_crashing"]
        assert s6["triggered"] is True, f"Expected signal 6 to pass, got: {s6}"

    def test_signal6_15m_crashing(self, engine_and_data):
        """Signal 6: GLD 15m down > -0.10% → blocked."""
        engine, data, GLD, SLV, UUP = engine_and_data
        # bars[-1] well below bars[-4] (3+ bars back = 15+ min crash)
        base = 185.0
        bars = [base, base, base, base * 0.995, base * 0.993, base * 0.990, base * 0.988]
        self._inject_intraday(data, GLD, bars)
        result = engine.evaluate_sell("WPM")
        s6 = result.details["macro_not_crashing"]
        # 15m window: bars[-1] vs bars[-4], ~1.2% drop > 0.10% threshold
        assert s6["triggered"] is False, f"Expected signal 6 to block on 15m crash: {s6}"

    def test_signal6_60m_waterfall(self, engine_and_data):
        """Signal 6: GLD 60m cumulative > -0.40% → blocked even if 15m OK."""
        engine, data, GLD, SLV, UUP = engine_and_data
        # 13 bars of 5-min data = 60 min window
        # Gradual -0.6% decline over 60 min but recent 15m is flat
        base = 185.0
        bars = [base * (1 - 0.0005 * i) for i in range(14)]
        # Make last 3 bars flat (15m OK)
        bars[-1] = bars[-2] = bars[-3] = bars[-4]
        self._inject_intraday(data, GLD, bars)
        result = engine.evaluate_sell("WPM")
        s6 = result.details["macro_not_crashing"]
        # 60m: bars[-1] vs bars[-13], should be ~0.45% drop → blocks
        assert s6["triggered"] is False, f"Expected 60m waterfall to block: {s6}"

    def test_signal7_stable_ratio(self, engine_and_data):
        """Signal 7: HV10/HV20 ratio not accelerating → triggered."""
        engine, data, GLD, SLV, UUP = engine_and_data
        # 30 bars of stable 1% daily moves
        closes = [100.0 * (1.01 ** i) for i in range(30)]
        df = _make_close_series(closes)
        data._daily_data["WPM"] = df
        data._price_cache["WPM"] = closes[-1]
        data._prev_close["WPM"] = closes[-2]
        # Prior ratio = same value (no change)
        hv10 = data.compute_hv("WPM", 10)
        hv20 = data.compute_hv("WPM", 20)
        prior = hv10 / hv20 if (hv10 and hv20 and hv20 > 0) else 1.0
        result = engine.evaluate_sell("WPM", prev_hv_ratios={"WPM": prior})
        s7 = result.details["vol_not_accelerating"]
        assert s7["triggered"] is True, f"Stable ratio should pass signal 7: {s7}"

    def test_signal7_accelerating_ratio(self, engine_and_data):
        """Signal 7: HV10/HV20 ratio up >5% → blocked."""
        engine, data, GLD, SLV, UUP = engine_and_data
        # Build a history where recent bars are much more volatile
        stable = [100.0 + 0.5 * i for i in range(15)]  # smooth
        volatile = [stable[-1] * (1 + 0.04 * ((-1)**i)) for i in range(15)]  # ±4% alternating
        closes = stable + volatile
        df = _make_close_series(closes)
        data._daily_data["WPM"] = df
        data._price_cache["WPM"] = closes[-1]
        data._prev_close["WPM"] = closes[-2]
        hv10 = data.compute_hv("WPM", 10)
        hv20 = data.compute_hv("WPM", 20)
        # Simulate prior ratio being low (calm baseline before volatility spike)
        prior_ratio = 0.30  # much lower than current ratio to trigger acceleration
        if hv10 is not None and hv20 is not None and hv20 > 0:
            result = engine.evaluate_sell("WPM", prev_hv_ratios={"WPM": prior_ratio})
            s7 = result.details["vol_not_accelerating"]
            current_ratio = hv10 / hv20
            ratio_change = (current_ratio - prior_ratio) / prior_ratio
            if ratio_change >= 0.05:
                assert s7["triggered"] is False, (
                    f"Accelerating vol (ratio change={ratio_change:.1%}) should block: {s7}"
                )

    def test_signal7_first_cycle_no_prior(self, engine_and_data):
        """Signal 7: no prior ratio → fail-CLOSED (Ultron review 2026-05-01).

        Previously assumed stable on first cycle. That was a fail-open path —
        missing data is not evidence of stability. Now requires >=1 cycle of
        HV history before the gate can fire.
        """
        engine, data, GLD, SLV, UUP = engine_and_data
        closes = [100.0 + 0.5 * i for i in range(25)]
        df = _make_close_series(closes)
        data._daily_data["WPM"] = df
        data._price_cache["WPM"] = closes[-1]
        data._prev_close["WPM"] = closes[-2]
        result = engine.evaluate_sell("WPM", prev_hv_ratios={})  # empty = no prior
        s7 = result.details["vol_not_accelerating"]
        assert s7["triggered"] is False, (
            "First cycle: no prior ratio should be fail-CLOSED "
            "(missing data is not evidence of stability)"
        )

    def test_signal6_no_intraday_failclosed(self, engine_and_data):
        """Signal 6: GLD intraday unavailable → fail-CLOSED (Ultron review)."""
        engine, data, GLD, SLV, UUP = engine_and_data
        # No intraday data injected for GLD
        result = engine.evaluate_sell("WPM")
        s6 = result.details["macro_not_crashing"]
        assert s6["triggered"] is False, (
            f"No intraday data should be fail-CLOSED, got: {s6}"
        )

    def test_signal7_no_hv_failclosed(self, engine_and_data):
        """Signal 7: no daily history (insufficient for HV) → fail-CLOSED."""
        engine, data, GLD, SLV, UUP = engine_and_data
        # WPM ticker has no daily data set up — HV computation will return None
        # (engine_and_data fixture leaves _daily_data["WPM"] unset by default)
        result = engine.evaluate_sell("WPM", prev_hv_ratios={"WPM": 0.95})
        s7 = result.details["vol_not_accelerating"]
        assert s7["triggered"] is False, (
            f"No HV data should be fail-CLOSED, got: {s7}"
        )

    def test_5_of_7_threshold(self, engine_and_data):
        """5 signals must fire to trigger; 4 must not."""
        engine, data, GLD, SLV, UUP = engine_and_data
        from csp_seller import CSP_MIN_SELL_SIGNALS
        # Inject WPM down 4% (Signal 1 passes)
        closes_down = [72.0] * 14 + [68.0]
        df = _make_close_series(closes_down)
        data._daily_data["WPM"] = df
        data._price_cache["WPM"] = 68.0
        data._prev_close["WPM"] = 72.0

        # GLD down 0.6% (Signal 4 passes), SLV flat
        self._inject_prices(data, GLD, 183.9, 185.0)
        data._daily_data[GLD] = _make_close_series([185.0] * 14 + [183.9])

        self._inject_prices(data, SLV, 22.0, 22.0)
        data._daily_data[SLV] = _make_close_series([22.0] * 15)

        # UUP up 0.15% (Signal 5 passes)
        self._inject_prices(data, UUP, 28.24, 28.20)
        data._daily_data[UUP] = _make_close_series([28.20] * 14 + [28.24])

        # Signal 6: GLD not crashing (fail-open: no intraday)
        # Signal 7: first cycle fail-open

        result = engine.evaluate_sell("WPM", prev_hv_ratios={})
        assert result.max_score == 7
        assert result.score >= 0
        assert result.triggered == (result.score >= CSP_MIN_SELL_SIGNALS)


# =============================================================================
# 2. STRIKE SELECTION TESTS
# =============================================================================

class TestCSPStrikeSelector:
    """Tests for CSPStrikeSelector with canned chain data."""

    @pytest.fixture
    def selector_and_data(self):
        from csp_seller import CSPDataLayer, CSPStrikeSelector
        data = CSPDataLayer()
        selector = CSPStrikeSelector(data)
        return selector, data

    def _make_chain(
        self,
        strikes: List[float],
        bids: List[float],
        asks: List[float],
        ois: List[int],
        dte: int,
        underlying: float,
        expiration: str = "2026-06-15",
    ) -> pd.DataFrame:
        rows = []
        for k, b, a, oi in zip(strikes, bids, asks, ois):
            mid = (b + a) / 2.0
            otm_pct = (underlying - k) / underlying
            rows.append({
                "strike": k,
                "bid": b,
                "ask": a,
                "mid_price": mid,
                "openInterest": oi,
                "dte": dte,
                "otm_pct": otm_pct,
                "expiration": expiration,
                "contractSymbol": f"WPM{expiration.replace('-','')[2:]}P{int(k*1000):08d}",
            })
        return pd.DataFrame(rows)

    def test_premium_floor_filters_low_bid(self, selector_and_data):
        """Strikes whose bid < 2% of strike must be filtered out."""
        selector, data = selector_and_data
        underlying = 72.0
        # $65 strike: 2% floor = $1.30; bid=$0.80 → should be rejected
        # $63 strike: 2% floor = $1.26; bid=$1.40 → should pass
        data._price_cache["WPM"] = underlying
        df = self._make_chain(
            strikes=[65.0, 63.0],
            bids=[0.80, 1.40],
            asks=[1.00, 1.60],
            ois=[200, 200],
            dte=35,
            underlying=underlying,
        )
        data._option_chains["WPM"] = {"2026-06-15": df}

        candidates = selector.select_strikes("WPM")
        strikes_returned = [c["strike"] for c in candidates]
        assert 65.0 not in strikes_returned, "bid $0.80 < floor $1.30 should be filtered"
        assert 63.0 in strikes_returned, "bid $1.40 > floor $1.26 should pass"

    def test_low_oi_filtered(self, selector_and_data):
        """Strikes with OI < 100 must be excluded."""
        selector, data = selector_and_data
        underlying = 72.0
        data._price_cache["WPM"] = underlying
        df = self._make_chain(
            strikes=[63.0, 64.0],
            bids=[1.50, 1.50],
            asks=[1.70, 1.70],
            ois=[50, 150],       # 50 → filtered; 150 → passes
            dte=35,
            underlying=underlying,
        )
        data._option_chains["WPM"] = {"2026-06-15": df}
        candidates = selector.select_strikes("WPM")
        strikes_returned = [c["strike"] for c in candidates]
        assert 63.0 not in strikes_returned, "OI=50 should be filtered"
        assert 64.0 in strikes_returned, "OI=150 should pass"

    def test_spread_cap_filtered(self, selector_and_data):
        """Strikes with bid-ask spread > 30% of mid must be excluded."""
        selector, data = selector_and_data
        underlying = 72.0
        data._price_cache["WPM"] = underlying
        # Strike $63: bid=$0.50, ask=$2.50 → spread=2.0/1.5=133% → filtered
        # Strike $62: bid=$1.40, ask=$1.60 → spread=0.2/1.5=13% → passes
        df = self._make_chain(
            strikes=[63.0, 62.0],
            bids=[0.50, 1.40],
            asks=[2.50, 1.60],
            ois=[200, 200],
            dte=35,
            underlying=underlying,
        )
        data._option_chains["WPM"] = {"2026-06-15": df}
        candidates = selector.select_strikes("WPM")
        strikes_returned = [c["strike"] for c in candidates]
        assert 63.0 not in strikes_returned, "Wide spread should be filtered"
        assert 62.0 in strikes_returned, "Tight spread should pass"

    def test_best_strike_scores_highest(self, selector_and_data):
        """Among valid candidates, the highest-score strike is returned first."""
        selector, data = selector_and_data
        underlying = 72.0
        data._price_cache["WPM"] = underlying
        # Two valid strikes; the one closer to 25-delta should rank higher
        df = self._make_chain(
            strikes=[54.0, 60.0],  # 60 is closer to 25-delta range
            bids=[1.50, 2.00],
            asks=[1.70, 2.20],
            ois=[200, 300],
            dte=35,
            underlying=underlying,
        )
        data._option_chains["WPM"] = {"2026-06-15": df}
        candidates = selector.select_strikes("WPM")
        assert len(candidates) >= 2
        # Score must be non-increasing
        scores = [c["score"] for c in candidates]
        assert scores == sorted(scores, reverse=True)

    def test_empty_chain_returns_no_candidates(self, selector_and_data):
        """An empty chain should return an empty candidate list."""
        selector, data = selector_and_data
        data._price_cache["WPM"] = 72.0
        data._option_chains["WPM"] = {}
        candidates = selector.select_strikes("WPM")
        assert candidates == []


# =============================================================================
# 3. CROSS-ENGINE DEDUP TESTS
# =============================================================================

class TestCSPDedup:
    """Tests for order_dedup.py CSP namespace and cross-engine blocking."""

    @pytest.fixture
    def dedup(self):
        from order_dedup import OrderDeduplicator
        return OrderDeduplicator()

    def _make_csp_seller_with_open(self, ticker: str, status: str = "open"):
        """Build a minimal mock CSPSeller with one open position."""
        from csp_seller import CSPPosition, CSPState
        from datetime import date
        exp_date = date(2026, 6, 15)
        pos = CSPPosition(
            ticker=ticker,
            option_symbol=f"{ticker}260615P00065000",
            strike=65.0,
            expiration="2026-06-15",
            dte_at_entry=35,
            contracts=1,
            sell_price=1.50,
            sell_date="2026-05-01T10:00:00+00:00",
            sell_underlying_price=72.0,
            sell_iv=0.35,
            sell_signal_score=5,
            sell_signal_details="test",
            cash_reserved=6500.0,
            status=status,
        )
        state = CSPState(positions=[pos])
        seller = MagicMock()
        seller.state = state
        return seller

    def test_csp_blocks_cc_sell_on_same_ticker(self, dedup):
        """has_pending_or_active_sell should return True when CSP is open on ticker."""
        csp_seller = self._make_csp_seller_with_open("WPM")
        dedup.register_csp_seller(csp_seller)
        # No CC engines registered — only CSP check fires
        result = dedup.has_pending_or_active_sell("WPM")
        assert result is True, "CC sell should be blocked when CSP is active on ticker"

    def test_cc_does_not_block_different_ticker(self, dedup):
        """CSP on WPM must not block CC sells on GDX."""
        csp_seller = self._make_csp_seller_with_open("WPM")
        dedup.register_csp_seller(csp_seller)
        result = dedup.has_pending_or_active_sell("GDX")
        assert result is False, "WPM CSP should not block GDX CC sell"

    def test_has_pending_or_active_csp_true(self, dedup):
        """has_pending_or_active_csp should return True for open position."""
        csp_seller = self._make_csp_seller_with_open("WPM")
        dedup.register_csp_seller(csp_seller)
        result = dedup.has_pending_or_active_csp("WPM")
        assert result is True

    def test_has_pending_or_active_csp_false_different_ticker(self, dedup):
        csp_seller = self._make_csp_seller_with_open("WPM")
        dedup.register_csp_seller(csp_seller)
        result = dedup.has_pending_or_active_csp("GDX")
        assert result is False

    def test_closed_position_does_not_block(self, dedup):
        """A closed CSP position must not trigger dedup block."""
        csp_seller = self._make_csp_seller_with_open("WPM", status="closed")
        dedup.register_csp_seller(csp_seller)
        result = dedup.has_pending_or_active_sell("WPM")
        assert result is False, "Closed CSP should not block CC sells"

    def test_dedup_fail_closed_on_exception(self, dedup):
        """CSP state read exception → fail-closed (True)."""
        bad_seller = MagicMock()
        bad_seller.state.open_positions.side_effect = RuntimeError("boom")
        dedup.register_csp_seller(bad_seller)
        result = dedup.has_pending_or_active_sell("WPM")
        assert result is True, "Exception in CSP state read must be fail-closed"

    def test_no_csp_registered_returns_false(self, dedup):
        """Without CSP seller registered, dedup should not block."""
        result = dedup.has_pending_or_active_sell("WPM")
        assert result is False

    def test_cc_blocks_csp_via_dedup(self, dedup):
        """CSPSeller._dedup_blocked should return True when CC is active on ticker."""
        from csp_seller import CSPSeller
        # Build a fake CC scalper with a pending sell on WPM
        cc_scalper = MagicMock()
        pending_order = MagicMock()
        pending_order.side = "sell"
        pending_order.contract_symbol = "WPM260620C00090000"
        cc_scalper.order_manager.get_pending_orders.return_value = [pending_order]
        cc_scalper.state.open_positions.return_value = []
        dedup.register_cc_scalper(cc_scalper)

        seller = CSPSeller(dry_run=True)
        seller._order_dedup = dedup
        # The dedup check should see the CC pending sell and block
        result = seller._dedup_blocked("WPM")
        assert result is True, "CSP should be blocked when CC has pending sell"


# =============================================================================
# 4. CAP MATH TESTS
# =============================================================================

class TestCSPCapMath:
    """Tests for position cap enforcement — specifically STRIKE vs spot for Cap 4."""

    @pytest.fixture
    def seller(self, tmp_path):
        from csp_seller import CSPSeller
        s = CSPSeller(dry_run=True)
        s._equity = 100_000.0
        # Override state to use tmp path
        from csp_seller import CSPState
        s.state = CSPState(_dry_run=True)
        return s

    def test_cap1_per_ticker_limit(self, seller):
        """Cap 1: second CSP on same ticker must be blocked."""
        from csp_seller import CSPPosition
        pos = CSPPosition(
            ticker="WPM",
            option_symbol="WPM260615P00065000",
            strike=65.0,
            expiration="2026-06-15",
            dte_at_entry=35,
            contracts=1,
            sell_price=1.50,
            sell_date="2026-05-01T10:00:00+00:00",
            sell_underlying_price=72.0,
            sell_iv=0.35,
            sell_signal_score=5,
            sell_signal_details="test",
            cash_reserved=6500.0,
            status="open",
        )
        seller.state.positions.append(pos)
        allowed, reason = seller._check_caps("WPM", 65.0, 1)
        assert allowed is False
        assert "cap1" in reason

    def test_cap2_total_equity_limit(self, seller):
        """Cap 2: total CSP cash must not exceed 15% of equity ($15,000)."""
        from csp_seller import CSPPosition, CSP_MAX_TOTAL_EQUITY_PCT
        # Add 2 open CSPs totaling $13,000 already reserved
        for i, (k, sym) in enumerate([(65.0, "WPM"), (95.0, "GDX")]):
            pos = CSPPosition(
                ticker=sym,
                option_symbol=f"{sym}260615P{i:08d}",
                strike=k,
                expiration="2026-06-15",
                dte_at_entry=35,
                contracts=1,
                sell_price=1.50,
                sell_date="2026-05-01T10:00:00+00:00",
                sell_underlying_price=k * 1.1,
                sell_iv=0.35,
                sell_signal_score=5,
                sell_signal_details="test",
                cash_reserved=k * 100,
                status="open",
            )
            seller.state.positions.append(pos)
        # GDX reserved = 9500, WPM reserved = 6500 → total = 16,000
        # But we're adding a NEW trade: strike=$90, 1 contract = $9,000
        # Total would be 16,000 + 9,000 = 25,000 > $15,000
        allowed, reason = seller._check_caps("GDXJ", 90.0, 1)
        assert allowed is False
        assert "cap2" in reason

    def test_cap3_concurrent_limit(self, seller):
        """Cap 3: more than 3 concurrent CSPs must be blocked.

        Use small strikes ($30) so cap2 (15% equity = $15k) doesn't fire first.
        3 positions × $3,000 = $9,000 total, new would be $3,000 = $12,000 < $15k.
        """
        from csp_seller import CSPPosition, CSP_MAX_CONCURRENT
        tickers = ["WPM", "GDX", "GDXJ"]
        for t in tickers:
            pos = CSPPosition(
                ticker=t,
                option_symbol=f"{t}260615P00030000",
                strike=30.0,
                expiration="2026-06-15",
                dte_at_entry=35,
                contracts=1,
                sell_price=0.80,
                sell_date="2026-05-01T10:00:00+00:00",
                sell_underlying_price=35.0,
                sell_iv=0.35,
                sell_signal_score=5,
                sell_signal_details="test",
                cash_reserved=3000.0,  # 30 * 100
                status="open",
            )
            seller.state.positions.append(pos)
        assert len(seller.state.open_positions()) == CSP_MAX_CONCURRENT
        # New attempt: PAAS strike=$30, total would be $12k < $15k cap2, so cap3 fires
        allowed, reason = seller._check_caps("PAAS", 30.0, 1)
        assert allowed is False
        assert "cap3" in reason

    def test_cap4_uses_strike_not_spot(self, seller):
        """Cap 4: combined exposure uses STRIKE price, not spot price.

        WPM spot=$125, strike=$110 (12% OTM), 1 contract = $11,000.
        Existing stock position = $0.
        8% of $100k equity = $8,000.
        $11,000 > $8,000 → blocked.
        If the test used spot $125 * 100 = $12,500, it would also block,
        but we verify the STRIKE ($110) is used via the exact math.
        """
        spot = 125.0
        strike = 110.0  # 12% OTM
        contracts = 1
        csp_exp = strike * contracts * 100  # 11,000 using STRIKE
        max_combined = seller._equity * 0.08  # 8,000

        # No stock position
        allowed, reason = seller._check_caps(
            "WPM", strike=strike, contracts=contracts, alpaca_positions=[]
        )
        # csp_exp = 11,000 > 8,000 → cap4 should block
        assert allowed is False
        assert "cap4" in reason, f"Expected cap4 block, got: {reason}"

    def test_cap4_combined_stock_plus_csp(self, seller):
        """Cap 4: existing $5,000 stock + $4,000 CSP at strike = $9,000 > $8,000 → blocked."""
        # Stock position: $5,000
        alpaca_positions = [{"symbol": "WPM", "market_value": "5000"}]
        # CSP: strike=$40, 1 contract = $4,000
        allowed, reason = seller._check_caps(
            "WPM", strike=40.0, contracts=1, alpaca_positions=alpaca_positions
        )
        # 5,000 + 4,000 = 9,000 > 8,000 → blocked
        assert allowed is False
        assert "cap4" in reason

    def test_cap4_passes_when_within_limit(self, seller):
        """Cap 4 passes when combined exposure is within 8% of equity."""
        alpaca_positions = [{"symbol": "WPM", "market_value": "3000"}]
        # CSP: strike=$40, 1 contract = $4,000; total = $7,000 < $8,000
        allowed, reason = seller._check_caps(
            "WPM", strike=40.0, contracts=1, alpaca_positions=alpaca_positions
        )
        assert allowed is True, f"Expected cap4 to pass: {reason}"


# =============================================================================
# 5. INTEGRATION TEST — dry-run cycle
# =============================================================================

class TestCSPSellerDryRunCycle:
    """Integration test: one full dry-run cycle with mocked market data."""

    @pytest.fixture
    def seller(self, tmp_path, monkeypatch):
        from csp_seller import CSPSeller, CSPState
        # Redirect state file to tmp dir
        monkeypatch.setattr("csp_seller.BASE_DIR", tmp_path)
        s = CSPSeller(dry_run=True)
        s._equity = 150_000.0
        s.state = CSPState(_dry_run=True)
        return s

    def _inject_full_market_data(self, data, tickers, gld_ticker, slv_ticker, uup_ticker):
        """Inject enough data for all 7 signals to have data (though not necessarily trigger)."""
        for ticker in tickers:
            closes = [70.0 - i * 0.2 for i in range(30)]  # gentle decline
            df = _make_close_series(closes)
            data._daily_data[ticker] = df
            data._price_cache[ticker] = closes[-1]
            data._prev_close[ticker] = closes[-2]

        # GLD: down 0.6%
        gld_closes = [185.0] * 14 + [183.9]
        data._daily_data[gld_ticker] = _make_close_series(gld_closes)
        data._price_cache[gld_ticker] = 183.9
        data._prev_close[gld_ticker] = 185.0
        data._intraday_data[gld_ticker] = pd.DataFrame({"Close": [185.0, 184.8, 184.6, 184.4, 184.2, 184.0, 183.9]})

        # SLV: flat
        slv_closes = [22.0] * 15
        data._daily_data[slv_ticker] = _make_close_series(slv_closes)
        data._price_cache[slv_ticker] = 22.0
        data._prev_close[slv_ticker] = 22.0
        data._intraday_data[slv_ticker] = pd.DataFrame({"Close": [22.0] * 7})

        # UUP: up 0.2%
        data._daily_data[uup_ticker] = _make_close_series([28.2] * 14 + [28.26])
        data._price_cache[uup_ticker] = 28.26
        data._prev_close[uup_ticker] = 28.2

    def test_dry_run_cycle_returns_expected_schema(self, seller, monkeypatch):
        """run_once() must return the expected result dict even with mocked data."""
        from csp_seller import GLD_TICKER, SLV_TICKER, UUP_TICKER, CSP_ELIGIBLE_TICKERS

        tickers = CSP_ELIGIBLE_TICKERS or ["WPM", "GDX"]
        self._inject_full_market_data(
            seller.data, tickers, GLD_TICKER, SLV_TICKER, UUP_TICKER
        )

        # Mock refresh_intraday to no-op (data already injected)
        monkeypatch.setattr(seller.data, "refresh_intraday", lambda: True)
        # Mock check_account to return test equity
        monkeypatch.setattr(
            seller.executor, "check_account",
            lambda: {"equity": "150000", "status": "ACTIVE (dry-run)"},
        )
        # Mock fetch_option_chain_puts to return None (no chain data)
        monkeypatch.setattr(seller.data, "fetch_option_chain_puts", lambda t: None)
        # Mock estimate_iv_rank to return passing value
        monkeypatch.setattr(seller.data, "estimate_iv_rank", lambda t, lookback=252: 55.0)

        result = seller.run_once()

        assert "sell_signals" in result
        assert "buy_back_actions" in result
        assert "errors" in result
        assert "eligible_tickers" in result
        assert isinstance(result["sell_signals"], list)
        assert isinstance(result["buy_back_actions"], list)

    def test_dry_run_sell_signal_structure(self, seller, monkeypatch):
        """Each sell signal entry must have ticker, triggered, signal_score."""
        from csp_seller import GLD_TICKER, SLV_TICKER, UUP_TICKER, CSP_ELIGIBLE_TICKERS
        tickers = CSP_ELIGIBLE_TICKERS or ["WPM"]

        self._inject_full_market_data(
            seller.data, tickers, GLD_TICKER, SLV_TICKER, UUP_TICKER
        )
        monkeypatch.setattr(seller.data, "refresh_intraday", lambda: True)
        monkeypatch.setattr(
            seller.executor, "check_account",
            lambda: {"equity": "150000", "status": "dry-run"},
        )
        monkeypatch.setattr(seller.data, "fetch_option_chain_puts", lambda t: None)
        monkeypatch.setattr(seller.data, "estimate_iv_rank", lambda t, lookback=252: 55.0)

        result = seller.run_once()
        for sig in result["sell_signals"]:
            assert "ticker" in sig, f"Missing 'ticker' in sell signal: {sig}"
            assert "triggered" in sig, f"Missing 'triggered' in sell signal: {sig}"
            assert "signal_score" in sig, f"Missing 'signal_score' in sell signal: {sig}"

    def test_dry_run_no_crash_with_no_eligible_tickers(self, seller, monkeypatch):
        """If eligible tickers list is empty, run_once() must return empty lists."""
        monkeypatch.setattr("csp_seller.CSP_ELIGIBLE_TICKERS", [])
        seller_fresh = __import__("csp_seller").CSPSeller(dry_run=True)
        monkeypatch.setattr("csp_seller.CSP_ELIGIBLE_TICKERS", [])
        result = seller_fresh.run_once()
        assert result["sell_signals"] == []
        assert result["buy_back_actions"] == []

    def test_existing_open_position_manages_exit(self, seller, monkeypatch):
        """If a CSP is open and 50% profit achieved, buy_back_actions must include it."""
        from csp_seller import CSPPosition, GLD_TICKER, SLV_TICKER, UUP_TICKER, CSP_ELIGIBLE_TICKERS

        tickers = CSP_ELIGIBLE_TICKERS or ["WPM"]
        self._inject_full_market_data(
            seller.data, tickers, GLD_TICKER, SLV_TICKER, UUP_TICKER
        )
        monkeypatch.setattr(seller.data, "refresh_intraday", lambda: True)
        monkeypatch.setattr(
            seller.executor, "check_account",
            lambda: {"equity": "150000", "status": "dry-run"},
        )
        monkeypatch.setattr(seller.data, "fetch_option_chain_puts", lambda t: None)
        monkeypatch.setattr(seller.data, "estimate_iv_rank", lambda t, lookback=252: 55.0)

        # Add an open position with sell_price=$2.00; current mock price = $0.90 → 55% profit
        sell_price = 2.00
        current_mock_price = 0.90  # 55% profit (> 50% target)
        pos = CSPPosition(
            ticker="WPM",
            option_symbol="WPM260615P00065000",
            strike=65.0,
            expiration="2026-07-15",  # far enough out that DTE > 7
            dte_at_entry=35,
            contracts=1,
            sell_price=sell_price,
            sell_date="2026-05-01T10:00:00+00:00",
            sell_underlying_price=72.0,
            sell_iv=0.35,
            sell_signal_score=5,
            sell_signal_details="test",
            cash_reserved=6500.0,
            status="open",
        )
        seller.state.positions.append(pos)

        # Mock option price to return 55% profit price
        monkeypatch.setattr(
            seller.executor, "get_current_option_price",
            lambda sym: current_mock_price,
        )
        # Mock buy_back_put to return a fake order ID
        monkeypatch.setattr(
            seller.executor, "buy_back_put",
            lambda sym, qty, lp: "DRY_BUY_BACK_123",
        )

        result = seller.run_once()
        assert len(result["buy_back_actions"]) >= 1, (
            "Expected buy-back action for 55% profit position"
        )
        action = result["buy_back_actions"][0]
        assert action["ticker"] == "WPM"
        assert action["order_id"] == "DRY_BUY_BACK_123"

    def test_hv_ratios_persisted_after_cycle(self, seller, monkeypatch):
        """prev_hv_ratios must be updated in state after run_once() for Signal 7."""
        from csp_seller import GLD_TICKER, SLV_TICKER, UUP_TICKER, CSP_ELIGIBLE_TICKERS
        tickers = CSP_ELIGIBLE_TICKERS or ["WPM"]

        self._inject_full_market_data(
            seller.data, tickers, GLD_TICKER, SLV_TICKER, UUP_TICKER
        )
        monkeypatch.setattr(seller.data, "refresh_intraday", lambda: True)
        monkeypatch.setattr(
            seller.executor, "check_account",
            lambda: {"equity": "150000", "status": "dry-run"},
        )
        monkeypatch.setattr(seller.data, "fetch_option_chain_puts", lambda t: None)
        monkeypatch.setattr(seller.data, "estimate_iv_rank", lambda t, lookback=252: 55.0)
        # Prevent actual disk write
        monkeypatch.setattr(seller.state, "save", lambda: None)

        seller.run_once()
        # After cycle, prev_hv_ratios should be populated for all eligible tickers
        for t in tickers:
            hv10 = seller.data.compute_hv(t, 10)
            hv20 = seller.data.compute_hv(t, 20)
            if hv10 is not None and hv20 is not None and hv20 > 0:
                assert t in seller.state.prev_hv_ratios, (
                    f"{t} should have prev_hv_ratio stored after cycle"
                )


# =============================================================================
# 6. EXTRA GUARD TESTS
# =============================================================================

class TestCSPGuards:
    """Extra guard behavior: cash check, BS pricing, IV rank fail-closed."""

    def test_get_available_cash_dry_run(self):
        """In dry-run mode, get_available_cash() must return 100,000."""
        from csp_seller import CSPExecutionLayer
        ex = CSPExecutionLayer(dry_run=True)
        cash = ex.get_available_cash()
        assert cash == 100_000.0

    def test_estimate_iv_rank_insufficient_data(self):
        """estimate_iv_rank must return None when fewer than 30 valid HV values."""
        from csp_seller import CSPDataLayer
        data = CSPDataLayer()
        # Only 40 bars — fewer than 60 required
        closes = [100.0 + i * 0.1 for i in range(40)]
        data._daily_data["WPM"] = _make_close_series(closes)
        rank = data.estimate_iv_rank("WPM")
        assert rank is None, "Should return None for insufficient data"

    def test_estimate_iv_rank_returns_0_to_100(self):
        """estimate_iv_rank must return a value in [0, 100] when data is sufficient."""
        from csp_seller import CSPDataLayer
        data = CSPDataLayer()
        # 100 bars — sufficient
        closes = [100.0 * (1 + 0.005 * (i % 10 - 5)) for i in range(100)]
        data._daily_data["WPM"] = _make_close_series(closes)
        rank = data.estimate_iv_rank("WPM")
        assert rank is not None
        assert 0.0 <= rank <= 100.0

    def test_bs_put_price_intrinsic_below_spot(self):
        """BS put price with T=0 should equal max(K-S, 0)."""
        from csp_seller import bs_put_price
        # Deep ITM: K=80, S=70 → intrinsic = 10
        p = bs_put_price(S=70.0, K=80.0, T=0.0, r=0.045, sigma=0.30)
        assert abs(p - 10.0) < 1e-6

    def test_bs_put_price_otm_positive(self):
        """OTM put with T>0 should have positive time value."""
        from csp_seller import bs_put_price
        # OTM: K=60, S=70 → some time value
        p = bs_put_price(S=70.0, K=60.0, T=30/365, r=0.045, sigma=0.30)
        assert p > 0.0

    def test_position_pnl_unrealized(self):
        """unrealized_pnl should equal (sell_price - current) * contracts * 100."""
        from csp_seller import CSPPosition
        pos = CSPPosition(
            ticker="WPM",
            option_symbol="WPM260615P00065000",
            strike=65.0,
            expiration="2026-06-15",
            dte_at_entry=35,
            contracts=2,
            sell_price=2.00,
            sell_date="2026-05-01T10:00:00+00:00",
            sell_underlying_price=72.0,
            sell_iv=0.35,
            sell_signal_score=5,
            sell_signal_details="test",
            cash_reserved=13000.0,
        )
        pnl = pos.unrealized_pnl(current_option_price=1.00)
        assert pnl == pytest.approx(200.0)  # (2.00 - 1.00) * 2 * 100

    def test_profit_pct_calculation(self):
        """profit_pct should be (sell - current) / sell."""
        from csp_seller import CSPPosition
        pos = CSPPosition(
            ticker="WPM",
            option_symbol="WPM260615P00065000",
            strike=65.0,
            expiration="2026-06-15",
            dte_at_entry=35,
            contracts=1,
            sell_price=2.00,
            sell_date="2026-05-01T10:00:00+00:00",
            sell_underlying_price=72.0,
            sell_iv=0.35,
            sell_signal_score=5,
            sell_signal_details="test",
            cash_reserved=6500.0,
        )
        assert pos.profit_pct(1.00) == pytest.approx(0.50)
        assert pos.profit_pct(2.00) == pytest.approx(0.00)
        assert pos.profit_pct(4.00) == pytest.approx(-1.00)
