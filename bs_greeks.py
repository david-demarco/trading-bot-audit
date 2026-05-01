"""bs_greeks.py — Black-Scholes greeks fallback for when broker feed
returns delta=0 sentinel (no greeks attached to snapshot).

This is intentionally minimal: NO dividends, NO scipy dep. Uses math.erf
for the standard normal CDF. For short-dated equity options that's
adequate (dividend-yield correction would shift call delta by ~q*T,
≤0.005 over 30d at q=2%).

USAGE (in yts_scanner._extract_snapshot or score_contract):

    if delta == 0.0 and iv is not None and spot is not None:
        delta = compute_delta(
            spot=spot, strike=strike, iv=iv, dte_days=dte,
            option_type=option_type, risk_free=0.045,
        )

If `iv` is also missing the caller should keep the 0.0 sentinel and let
the existing "skip filter when delta unknown" path apply (better to admit
unknown than guess wildly).

Reference: Hull, Options Futures and Other Derivatives Ch.15.
"""
from __future__ import annotations

import math
from typing import Optional


# Standard 1-yr T-bill yield baseline (Apr 2026).  Caller can override.
DEFAULT_RISK_FREE = 0.045


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function via erf.

    N(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density (used for gamma/vega)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1(spot: float, strike: float, iv: float, t_years: float, r: float) -> float:
    return (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))


def compute_delta(
    spot: float,
    strike: float,
    iv: float,
    dte_days: float,
    option_type: str,
    risk_free: float = DEFAULT_RISK_FREE,
) -> Optional[float]:
    """Black-Scholes delta. Returns signed delta:
       call: +N(d1)            (0 → +1)
       put:  N(d1) - 1         (-1 → 0)

    Returns None on bad inputs (negative spot/strike/iv/dte, dte=0).
    """
    if spot is None or strike is None or iv is None or dte_days is None:
        return None
    if spot <= 0 or strike <= 0 or iv <= 0 or dte_days <= 0:
        return None
    t = dte_days / 365.0
    try:
        d1 = _d1(spot, strike, iv, t, risk_free)
    except (ValueError, ZeroDivisionError):
        return None
    nd1 = _norm_cdf(d1)
    ot = option_type.lower()
    if ot == "call":
        return nd1
    if ot == "put":
        return nd1 - 1.0
    return None


def compute_greeks(
    spot: float,
    strike: float,
    iv: float,
    dte_days: float,
    option_type: str,
    risk_free: float = DEFAULT_RISK_FREE,
) -> Optional[dict]:
    """Full {delta, gamma, vega, theta, rho} via Black-Scholes. Returns
    None on invalid inputs. theta is per-day (annualized / 365)."""
    if spot is None or strike is None or iv is None or dte_days is None:
        return None
    if spot <= 0 or strike <= 0 or iv <= 0 or dte_days <= 0:
        return None
    t = dte_days / 365.0
    try:
        d1 = _d1(spot, strike, iv, t, risk_free)
    except (ValueError, ZeroDivisionError):
        return None
    d2 = d1 - iv * math.sqrt(t)
    nd1 = _norm_cdf(d1)
    nd2 = _norm_cdf(d2)
    pd1 = _norm_pdf(d1)
    discount = math.exp(-risk_free * t)
    gamma = pd1 / (spot * iv * math.sqrt(t))
    vega = spot * pd1 * math.sqrt(t) / 100.0  # per 1 vol-pt change
    ot = option_type.lower()
    if ot == "call":
        delta = nd1
        theta = (
            -spot * pd1 * iv / (2.0 * math.sqrt(t))
            - risk_free * strike * discount * nd2
        ) / 365.0
        rho = strike * t * discount * nd2 / 100.0
    elif ot == "put":
        delta = nd1 - 1.0
        theta = (
            -spot * pd1 * iv / (2.0 * math.sqrt(t))
            + risk_free * strike * discount * (1.0 - nd2)
        ) / 365.0
        rho = -strike * t * discount * (1.0 - nd2) / 100.0
    else:
        return None
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho}
