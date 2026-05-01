"""
combined_config.py - Unified configuration for the combined trading system.

Defines ticker tiers, conversion parameters, capital allocation model,
scheduling constants, and all shared parameters consumed by the orchestrator,
conversion engine, and unified risk manager.

Reference: ~/trading_bot/research/combined_system_design.md
"""

from __future__ import annotations

# =============================================================================
# FUNDAMENTAL FILTER
# =============================================================================

fundamental_filter_enabled = True        # Toggle to False to disable the gate

# Chinese ADRs / VIE structures -- instant reject
# NIO, SOFI, EXK removed from active universe (2026-03-09, David approved)
BLACKLISTED_TICKERS = {
    "BABA", "PDD", "JD", "XPEV", "LI", "BIDU", "TME", "BILI",
    "ZTO", "VNET", "IQ", "FUTU", "TAL", "EDU", "KC", "DIDI", "MNSO",
    "GDS", "WB", "TUYA", "YMM",
    "NIO", "SOFI", "EXK",
}

# =============================================================================
# TRADING UNIVERSE — One flat list for the 3-layer system (Mar 16 flatten):
#   Layer 1: RSI2 entry (RSI(2) < 15, price > 200-day SMA)
#   Layer 2: Swing hold (5-SMA exit, no stop loss, 10-day max hold unless CC active)
#   Layer 3: Covered calls when 6-signal conditions fire (options-eligible tickers only)
#
# David approved Mar 15. ALL tickers participate in Layers 1-2.
# Layer 3 (CC overlay) only fires on tickers with Alpaca options.
# GOAU, GOEX removed Mar 12 (David: garbage spreads)
# Mar 16: Flattened — no more ETF/stock split. One universe.
# =============================================================================

TRADING_UNIVERSE = [
    # Sprott ETFs
    "SLVR",   # Sprott Silver Miners & Physical Silver (~$70)
    "SGDM",   # Sprott Gold Miners (~$85)
    "SGDJ",   # Sprott Junior Gold Miners (~$100)
    "GBUG",   # Sprott Active Gold & Silver Miners (~$51)
    "COPP",   # Sprott Copper Miners (~$38)
    "COPJ",   # Sprott Junior Copper Miners (~$43)
    # Non-Sprott ETFs
    "SIL",    # Global X Silver Miners (~$94)
    "SILJ",   # Amplify Junior Silver Miners (~$34)
    "GDX",    # VanEck Gold Miners (~$102)
    "GDXJ",   # VanEck Junior Gold Miners (~$137)
    "COPX",   # Global X Copper Miners (~$82)
    "SLVP",   # iShares Global Silver Miners (~$40)
    "RING",   # iShares Global Gold Miners (~$88)
    # Individual miners
    "PAAS",   # Pan American Silver (~$26)
    "BTG",    # B2Gold (~$4)
    "WPM",    # Wheaton Precious Metals (~$72)
    "AG",     # First Majestic Silver (~$8)
    # "HL",   # Hecla Mining — REMOVED Apr 8: negative RSI2 returns with any stop loss in 5yr backtest
    "FSM",    # Fortuna Mining (~$7)
    "SVM",    # Silvercorp Metals (~$5)
    "USAS",   # Americas Gold and Silver (~$1)
    "CDE",    # Coeur Mining (~$8)
    "EGO",    # Eldorado Gold (~$21)
    "KGC",    # Kinross Gold (~$13)
    # Oil / Energy (added Mar 18)
    "XLE",    # Energy Select ETF (~$59)
    "OXY",    # Occidental Petroleum (~$57)
    "HAL",    # Halliburton (~$35)
    "DVN",    # Devon Energy (~$47)
    "SLB",    # Schlumberger (~$45)
    "XOP",    # SPDR S&P Oil & Gas ETF (~$168)
    "COP",    # ConocoPhillips (~$115)
    "TPL",    # Texas Pacific Land (~$1400+, LEAP-preferred — too expensive for shares)
    "EQT",    # EQT Corp, natural gas (~$50)
    # Edge 65/66/68 ADDs (Apr 17, 2026) — large-cap gold majors
    "AEM",   # Agnico Eagle Mines (~$215) — Edge 65 ADD
    "NEM",   # Newmont (largest gold miner ~$113) — Edge 65 ADD
    "RGLD",  # Royal Gold (royalty ~$262) — Edge 65 ADD
    "AU",    # AngloGold Ashanti (~$104) — Edge 65 ADD
    "FNV",   # Franco-Nevada (royalty ~$258) — Edge 65 ADD
    "BVN",   # Buenaventura, Peru (~$36) — Edge 65 ADD
    "GOLD",  # Barrick Gold (~$47) — Edge 65 ADD
    "GLD",   # SPDR Gold Shares ETF (~$440) — Edge 65 ADD (top contributor)
]

# CC_ELIGIBLE_SWING = the full universe (alias kept for backward compat)
CC_ELIGIBLE_SWING = TRADING_UNIVERSE

# ETFs / commodity products — bypass fundamental screening entirely.
# Auto-derived from TRADING_UNIVERSE + common broad ETFs (Mar 16).
# Used by fundamental_filter.py to skip fundamental checks on ETFs.
_MINING_ETFS = {
    "SLVR", "SGDM", "SGDJ", "GBUG", "COPP", "COPJ",
    "SIL", "SILJ", "GDX", "GDXJ", "COPX", "SLVP", "RING", "GLD",
}
_OIL_ETFS = {"XLE", "XOP"}
ETF_TICKERS = _MINING_ETFS | _OIL_ETFS | {
    "GLD", "SLV", "GDX", "SPY", "QQQ", "IWM",
    "XLK", "XLE", "XLF", "XLV", "XLB", "XLI", "XLU", "XLP", "XLY",
    "TLT", "HYG", "EEM", "DIA", "VTI", "VOO",
    "USO",  # Oil proxy — used as macro indicator for energy tickers (not traded)
}

# Subset with confirmed Alpaca options chains (Layer 3 CC overlay only)
CC_OPTIONS_ELIGIBLE = [
    "GDX", "GDXJ", "COPX", "SILJ",
    "SIL", "SLVR", "SGDM", "RING",
    "PAAS", "AG",  # HL removed Apr 8
    "WPM", "BTG", "FSM", "SVM",
    # Oil / Energy (added Mar 18) — confirmed options chains exist
    "XLE", "OXY", "HAL", "DVN", "SLB", "COP", "EQT",
]

# =============================================================================
# TAIL RISK / POWER-LAW FILTER (Apr 11 — Taleb/Spitznagel research)
# =============================================================================
# Tail index α (Hill estimator, 2yr daily returns, measured Apr 11 2026).
# Lower α = fatter tails = Black-Scholes more wrong = selling calls riskier.
# α < 3: infinite variance — DO NOT sell calls (use puts instead)
# α 3-4: fat tails — sell calls cautiously, prefer tickers with overpriced calls
# α > 4: milder tails — CC selling is safer here

TAIL_INDEX = {
    "WPM": 2.92,  # FATTEST — do NOT sell calls, sell puts instead
    "GDX": 2.96,  # FATTEST — do NOT sell calls, sell puts instead
    "HL":  3.17,
    "EGO": 3.44,
    "PAAS": 3.69,
    "GLD": 3.69,
    "AG":  3.86,
    "SVM": 3.90,
    "SIL": 4.02,
    "FSM": 4.08,
    "KGC": 4.24,  # Mildest tails — safest for CC selling
}

# Tickers where calls are MORE overpriced than puts (power-law analysis Apr 11)
# These are the BEST candidates for selling covered calls
CC_PREFERRED = ["HL", "AG", "PAAS", "EGO", "SVM", "FSM", "KGC"]

# Tickers where puts are MORE overpriced than calls
# Better to sell cash-secured puts on these, not covered calls
CSP_PREFERRED = ["WPM", "GDX", "GDXJ"]

# Minimum tail index for selling covered calls
# α < 3 = infinite variance = don't sell calls
CC_MIN_TAIL_INDEX = 3.0

# Maximum IV/HV ratio to consider options "fairly priced" (avoid selling cheap)
CC_MIN_IV_HV_RATIO = 0.9  # Don't sell if IV < 90% of HV (underpriced)

# Tail hedge parameters
TAIL_HEDGE_ENABLED = False  # Enable when David approves
TAIL_HEDGE_TICKER = "GDX"   # Hedge vehicle
TAIL_HEDGE_OTM_PCT = 0.20   # 20% OTM puts
TAIL_HEDGE_BUDGET_PCT = 0.03  # 3% annual portfolio cost (Spitznagel target)

# =============================================================================
# VOL REGIME SIGNALS (Apr 11 — gold/miner vol ratio)
# =============================================================================
# GDX vol is typically 2.4x GLD vol. When the ratio deviates, it signals
# whether miner options are over/underpriced relative to the underlying metal.
#
# GDX/GLD vol ratio > 3.0: miners panicking → options overpriced → SELL premium
# GDX/GLD vol ratio 1.8-3.0: normal range → no strong signal
# GDX/GLD vol ratio < 1.8: miners complacent → options underpriced → BUY protection
#
# Combined with rolling α:
#   Low ratio + low α = DANGER ZONE (complacent + fat tails = crash setup)
#   High ratio + high α = SELL ZONE (panic + thin tails = premium overpriced)

VOL_RATIO_SELL_THRESHOLD = 3.0   # Sell premium when GDX/GLD ratio above this
VOL_RATIO_BUY_THRESHOLD = 1.8    # Buy protection when ratio below this
VOL_RATIO_REFERENCE_TICKER = "GLD"  # Gold proxy for metal vol

# Backward-compatible aliases (all point to TRADING_UNIVERSE)
MINING_ETF_TICKERS = [t for t in TRADING_UNIVERSE if t in _MINING_ETFS]
MINING_STOCK_TICKERS = [t for t in TRADING_UNIVERSE if t not in _MINING_ETFS]
TIER_1_TICKERS = []
TIER_1B_TICKERS = []
TIER_2_SCALP_ONLY = []
TIER_3_SWING_ONLY = []
SCALP_ELIGIBLE = TRADING_UNIVERSE
SWING_ELIGIBLE = TRADING_UNIVERSE
CONVERSION_ELIGIBLE = TRADING_UNIVERSE
ALL_TICKERS = sorted(set(TRADING_UNIVERSE))

# Sector mapping (used by unified risk manager for concentration checks)
TICKER_SECTOR = {
    # Mining ETFs — full universe (Mar 15)
    "SLVR": "silver_mining", "SIL": "silver_mining", "SILJ": "silver_mining",
    "SLVP": "silver_mining",
    "SGDM": "gold_mining", "SGDJ": "gold_mining", "GBUG": "gold_mining",
    "GDX": "gold_mining", "GDXJ": "gold_mining", "RING": "gold_mining",
    "COPP": "copper_mining", "COPJ": "copper_mining", "COPX": "copper_mining",
    # Physical metals
    "GLD": "metals", "SLV": "metals",
    # Individual miners (Mar 15 — full stock universe)
    "PAAS": "silver_mining", "AG": "silver_mining", "HL": "silver_mining",
    "FSM": "silver_mining", "SVM": "silver_mining", "USAS": "silver_mining",
    "CDE": "silver_mining",
    "BTG": "gold_mining", "WPM": "gold_mining",
    "EGO": "gold_mining", "KGC": "gold_mining",
    "CCJ": "uranium",
    # Oil / Energy (added Mar 18)
    "XLE": "oil_etf", "XOP": "oil_etf",
    "OXY": "oil_producer", "DVN": "oil_producer", "COP": "oil_producer",
    "HAL": "oil_services", "SLB": "oil_services",
    "TPL": "oil_royalty",
    "EQT": "nat_gas",
}

# All energy sector labels (used by signal engines to pick USO vs GLD as macro indicator)
ENERGY_SECTORS = {"oil_etf", "oil_producer", "oil_services", "oil_royalty", "nat_gas"}

# Aggregate energy sector position limit (H5 audit fix).
# Sub-sector limits (SWING_MAX_SECTOR_POSITIONS=3) still apply per sub-sector,
# but this umbrella limit caps total energy exposure across all sub-sectors.
ENERGY_MAX_TOTAL_POSITIONS = 5

# Aggregate metals sector position limit (Edge 90, Apr 17 2026).
# Sub-sector limits (SWING_MAX_SECTOR_POSITIONS=3) still apply per
# sub-sector, but this umbrella caps total metals exposure across
# gold/silver/copper. Edge 90: 5y avg pairwise rho=0.76 across the
# 14-name metals universe (COVID-regime rho=0.885). Effective N at
# 6 positions ~= 1.4. Mirrors energy umbrella.
METALS_SECTORS = {"gold_mining", "silver_mining", "copper_mining"}
METALS_MAX_TOTAL_POSITIONS = 5

# Macro indicator tickers per sector group
# Mining tickers use GLD; energy tickers use USO
MACRO_INDICATOR_TICKER = {
    "mining": "GLD",
    "energy": "USO",
}

def get_macro_indicator(ticker: str) -> str:
    """Return the appropriate macro indicator ticker (GLD or USO) for a given ticker."""
    sector = TICKER_SECTOR.get(ticker, "")
    if sector in ENERGY_SECTORS:
        return "USO"
    return "GLD"

# Edge 85 (cycle 94) — macro_size_mult kill switch.
# Edge 85 NFCI-proxy backtest 1993-2026 SPY: -0.93%/yr drag, bootstrap CI
# [-0.109%, -0.042%] per 20d EXCLUDES ZERO.
# Edge 97 NFCI-proxy on 15-ticker metals basket 2000-2026 (n=6,563 days):
# -0.121% per 20d ≈ -1.5%/yr drag, CI [-0.225%, -0.020%] EXCLUDES ZERO,
# AND Q5 high-stress bucket fwd return (+2.61%) > Q1 low-stress (+1.55%) —
# multiplier ordering is backwards on the bot's own universe.
# Hand-tuned multipliers (0.5-1.5x) have no empirical support on either SPY
# or metals. Pin to 1.0 until classifier is rebuilt with empirical
# calibration. Set True to re-engage after rebuild.
MACRO_SIZE_MULT_ENABLED = False


# =============================================================================
# SCALP PARAMETERS (match scalp_runner.py defaults)
# =============================================================================

SCALP_RISK_PCT = 0.005              # 0.5% risk per trade ($500 on $100K)
SCALP_MAX_POSITION_PCT = 0.10       # 10% of portfolio per position
SCALP_MAX_CONCURRENT = 3            # max concurrent scalp positions
SCALP_DAILY_LOSS_LIMIT = 0.01       # 1% daily loss limit ($1,000)
SCALP_FORCE_CLOSE_TIME = (15, 50)   # 3:50 PM ET
SCALP_POLL_SECONDS = 60             # check for signals every 60 seconds

# =============================================================================
# CONVERSION PARAMETERS (Section 7 of design doc)
# =============================================================================

CONVERSION_PRICE_TOLERANCE_ATR = 0.5    # Max adverse excursion in ATR units
CONVERSION_RESISTANCE_BUFFER = 0.003    # 0.3% buffer from daily high/low
CONVERSION_MIN_DAILY_SMA20_SLOPE = 0.0  # SMA20 must be non-negative for longs
CONVERSION_VOLUME_THRESHOLD = 0.8       # Today's vol must be > 80% of average
CONVERSION_MAX_LOSS_PCT = 0.003         # Max unrealized loss to still convert (0.3%)
CONVERSION_ALLOW_ADD = True             # Allow buying additional shares on conversion
CONVERSION_ALLOW_TRIM = True            # Allow selling excess shares on conversion
CONVERSION_ADD_MAX_LOSS = 0.0           # Only add if scalp portion is break-even or better

# =============================================================================
# SIGNAL PIPELINE
# =============================================================================

SCALP_WINDOW_CUTOFF_HOUR = 14  # 2:00 PM ET; after this, swing entries go direct

# =============================================================================
# SWING PARAMETERS (match swing_runner.py defaults)
# =============================================================================

SWING_CAPITAL_PCT = 0.90                # 90% of portfolio for swing strategies
SWING_MAX_SINGLE_POSITION_PCT = 0.15    # 15% of portfolio per position (100-share lots for CCs)
SWING_MAX_SECTOR_POSITIONS = 3          # max positions per sector across all strategies
SWING_MAX_PORTFOLIO_EXPOSURE = 1.0      # 100% max (no leverage)
SWING_MIN_LOT_SIZE = 100               # Swing entries in 100-share lots (for covered calls)
SWING_PROFIT_TARGET_PER_SHARE = 0.50   # $0.50/share profit target (David's rule)
SWING_STOP_LOSS_PCT = None             # NO stop loss -- backtest proved stops destroy value on mean-reverting stocks

# =============================================================================
# RSI2 MEAN-REVERSION STRATEGY PARAMETERS (David approved 2026-03-09)
# =============================================================================

RSI2_ENTRY_THRESHOLD = 10              # RSI(2) < 10 (tightened from 15 per Apr 8 backtest — better win rate)
SMA_TREND_PERIOD = 200                 # Price must be above 200-day SMA (uptrend filter)
SMA_EXIT_PERIOD = 5                    # Exit when price crosses above 5-day SMA
MAX_HOLD_DAYS = 10                     # Safety valve: force exit after 10 trading days
CC_ELIGIBLE_AFTER_DAYS = None          # REMOVED: age is irrelevant. CC timing now based on RSI2 > 50 bounce.

# =============================================================================
# MACRO REGIME FILTERS (added Apr 8 per backtest research)
# =============================================================================
# Dollar regime: UUP > 200 SMA = strong dollar = better RSI2 entries (Sharpe 3.0-3.4)
DOLLAR_FILTER_ENABLED = True
DOLLAR_ETF = "UUP"
DOLLAR_SMA_PERIOD = 200

# VIX filter: low VIX = reliable mean reversion, high VIX = dips keep dipping
VIX_FILTER_ENABLED = True
VIX_MAX_ENTRY = 25                    # Don't enter RSI2 trades when VIX > 25

# CC timing: sell covered call when RSI14 > 70 (sustained uptrend confirmed)
# Knowledge base: "RSI14>70 alone is the best single filter (Sharpe 1.5, 94% WR)"
# "RSI2 alone is too noisy for CC timing" — use RSI14 for multi-week trend.
CC_RSI14_SELL_THRESHOLD = 70          # Sell CC when RSI14 > 70
CC_RSI2_BOUNCE_THRESHOLD = 50        # DEPRECATED — kept for backward compat

# =============================================================================
# OPTIONS OVERLAY
# =============================================================================

OPTIONS_SEASONING_DAYS = 0              # No seasoning wait — sell on any spike (David, Mar 15)
OPTIONS_FAST_SEASONING_IV_RANK = 75     # Reduce to 1 day if IV rank > 75
OPTIONS_FAST_SEASONING_ADX_MAX = 20     # ... and ADX < 20

# =============================================================================
# RISK (UNIFIED)
# =============================================================================

SCALP_RESERVE_PCT = 0.10               # 10% of capital reserved for scalping
DAILY_COMBINED_LOSS_LIMIT = 0.02        # 2% combined daily loss limit
DD_REDUCE_THRESHOLD = -0.15             # -15% DD: reduce all sizes by 50%
DD_CIRCUIT_BREAKER = -0.20              # -20% DD: go 100% cash for 5 trading days
CIRCUIT_BREAKER_DAYS = 5

# Tiered DD size multipliers (SIZING_RULES.md §2.2, wired Apr 21 2026 22:15 EDT).
# Used by UnifiedRiskManager.dd_size_multiplier() — returns multiplier for new
# entry sizing based on current peak-to-trough drawdown. List ordered from
# most-severe to least-severe; first match wins. Each tuple = (max_dd_pct,
# multiplier, autonomous_blocked). HARD_HALT and 0.0 multiplier means no
# new entries at all.
DD_TIERS = [
    # (dd_threshold, size_multiplier, autonomous_blocked, label)
    (-0.25, 0.00, True,  "HARD_HALT"),       # ≤ -25% — flatten + pause autonomous
    (-0.20, 0.25, True,  "SEVERE_DD"),       # -20% to -25% — 0.25x + no new spec
    (-0.15, 0.50, False, "MODERATE_DD"),     # -15% to -20% — 0.50x (existing DD_REDUCE_THRESHOLD)
    (-0.10, 0.75, False, "MILD_DD"),         # -10% to -15% — 0.75x
    (float("-inf"), 1.0, False, "NORMAL"),   # > -10% — 1.0x (catch-all)
]
HARD_HALT_DD = -0.25                     # SIZING_RULES.md §2.2 line: "≤ -25% HARD HALT"

# =============================================================================
# SCHEDULING (ET times as (hour, minute) tuples)
# =============================================================================

PREMARKET_START = (6, 0)
MARKET_OPEN = (9, 30)
NO_TRADE_END = (9, 45)                 # First 15 minutes no-trade zone (David's rule)
GAP_FILL_WINDOW_END = (9, 40)
SWING_SIGNAL_TIME = (10, 0)
SCALP_CUTOFF = (14, 0)                 # After 2 PM, direct swing entries
OPTIONS_EVAL_TIME = (15, 30)
NO_NEW_ENTRIES_CUTOFF = (15, 50)       # Last 10 minutes: no new entries, exits OK
SCALP_FORCE_CLOSE = (15, 50)
FORCE_CLOSE_ALL_TIME = (15, 55)        # Force-close ALL positions 5 min before close
MARKET_CLOSE = (16, 0)
EOD_RECONCILIATION = (16, 5)

# =============================================================================
# IEX SPREAD CHECK
# =============================================================================

MAX_IEX_SPREAD_PCT = 0.005             # Max 0.5% bid-ask spread to enter a trade

# =============================================================================
# STRATEGY ALLOCATIONS (imported from swing_runner.py for reference)
# =============================================================================

STRATEGY_ALLOCATIONS = {
    "momentum_rotation":        0.00,  # DISABLED — RSI2 only (Mar 18)
    "vwap_mean_reversion":      0.00,  # DISABLED — RSI2 only (Mar 18)
    "sector_relative_strength": 0.00,  # DISABLED — RSI2 only (Mar 18)
    "donchian_breakout":        0.00,  # DISABLED — RSI2 only (Mar 18)
    "rsi2_mean_reversion":      1.00,  # Only active entry strategy
}

MAX_POSITIONS_PER_STRATEGY = {
    "momentum_rotation":        0,     # DISABLED
    "vwap_mean_reversion":      0,     # DISABLED
    "sector_relative_strength": 0,     # DISABLED
    "donchian_breakout":        0,     # DISABLED
    "rsi2_mean_reversion":      20,    # Primary and only entry strategy
}

# Total portfolio: LEAPs cost $500-$2500, not $5k-$15k per 100-share lot
SWING_MAX_TOTAL_POSITIONS = 20

# =============================================================================
# PMCC CONFIGURATION (Poor Man's Covered Calls)
# =============================================================================
# Buy deep ITM LEAPs instead of 100 shares for swing positions.
# Sell short-dated OTM calls against the LEAP for recurring premium.

PMCC_ENABLED = True                        # Master toggle: enable PMCC/LEAP subsystem

# =============================================================================
# POSITION RECONCILIATION (Alpaca as source of truth)
# =============================================================================
# When enabled, every intraday loop iteration pulls positions from Alpaca
# and reconciles local state. Alpaca IS the state -- local file is just cache.
ALPACA_RECONCILIATION_ENABLED = True       # Master toggle for position reconciliation
RECONCILIATION_INTERVAL_SECONDS = 120      # How often to reconcile (every 2 minutes)
# NOTE: PMCC_MODE removed (Mar 16). Replaced by intelligent per-signal routing
# in entry_router.py. The router decides LEAP vs shares per-ticker, per-signal.

# --- Long Leg (LEAP) Selection ---
PMCC_LONG_LEG_MIN_DTE = 180               # 6 months minimum
PMCC_LONG_LEG_MAX_DTE = 365               # 12 months maximum
PMCC_LONG_LEG_MIN_DELTA = 0.65            # Deep ITM (widened from 0.70 for illiquid chains)
PMCC_LONG_LEG_MAX_DELTA = 0.95            # Deep ITM LEAPs OK (David: up to .95)
PMCC_LEAP_MAX_SPREAD_PCT = 0.15           # Max 15% bid-ask spread (widened from 5% for mining ETFs)

# --- Short Leg Constraints ---
PMCC_SHORT_MAX_DELTA = 0.15               # Conservative short leg (aka PMCC_MAX_SHORT_DELTA)
PMCC_SHORT_MIN_DELTA = 0.05               # Floor — zero-delta strikes have no live market on Alpaca
                                          # and collect ~$0 premium. Prevents 422 retry storm
                                          # (Apr 30 2026: XLE260618C00105000 looped 273x in one day).
PMCC_SHORT_REQUIRE_LIVE_QUOTE = True      # Require bid > 0 AND ask > 0 (skip stale-mid-only chains)
PMCC_SHORT_DTE_MIN = 21
PMCC_SHORT_DTE_MAX = 60
PMCC_SHORT_DTE_OPTIMAL = 35

# --- Position Limits ---
PMCC_MAX_CONCURRENT_SPREADS = 5
PMCC_SEASONING_DAYS = 0                   # Sell short calls immediately (was 5)
PMCC_MAX_LEAP_COST_PCT = 0.05             # Max 5% of portfolio per LEAP contract
PMCC_TOTAL_ALLOCATION_PCT = 0.20          # Max 20% of portfolio in all LEAPs

# --- Risk Controls ---
PMCC_ASSIGNMENT_BUFFER_PCT = 0.10         # Extra OTM buffer vs regular CCs
PMCC_MAX_RISK_RATIO = 0.30               # Max short premium / LEAP cost
PMCC_MIN_NET_CREDIT = 0.10                # Min premium per short leg sale (per share)
PMCC_MIN_NET_CREDIT_TARGET = 0.10         # Alias used by pmcc_manager

# PMCC risk thresholds
PMCC_LEAP_MIN_DTE_WARN = 120              # Warn when LEAP DTE < 120
PMCC_LEAP_MIN_DTE_CLOSE = 60             # Close spread when LEAP DTE < 60
PMCC_LEAP_MIN_DELTA = 0.55               # Close if LEAP delta drops below this
PMCC_SHORT_DELTA_WARN = 0.40             # Assignment warning threshold
PMCC_SHORT_DELTA_DANGER = 0.50           # Assignment danger threshold

# --- Profit-Taking ---
PMCC_SHORT_PROFIT_TARGET_PCT = 0.50       # Buy back short at 50% profit
PMCC_AUTO_RESELL = False                  # After profit buy-back, wait for manual trigger

# --- Buy-Back Dual Trigger System (v2, Mar 19) ---
# Two independent triggers fire a buy-back:
#   1. 50% profit trigger: short call decayed to 50% of sell price -> buy back
#   2. 21 DTE checkpoint: if DTE <= 21, close regardless of profit level
#      (unless underwater -> evaluate roll)
#   3. Let-expire zone: DTE <= 5 AND delta < 0.10 AND value < $0.05 -> skip
PMCC_PROFIT_TARGET_EARLY = 0.50           # 50% profit trigger (first half of contract)
PMCC_PROFIT_TARGET_LATE = 0.50            # 50% profit trigger (second half, same for now)
PMCC_CLOSE_DTE = 21                       # Close at 21 DTE regardless of profit
PMCC_LET_EXPIRE_DTE = 5                   # Don't bother buying back below 5 DTE
PMCC_LET_EXPIRE_DELTA = 0.10             # ... if delta < 0.10
PMCC_LET_EXPIRE_VALUE = 0.05             # ... and value < $0.05

# --- Cheap BTC Stepping ---
# When a short call has decayed to near-zero (low delta, near-worthless),
# actively try to close it cheap so the LEAP is free to resell on the next spike.
# Places BTC limit orders starting at $0.01, stepping up $0.01 every 5 minutes.
# Uses DAY orders — unfilled orders expire at close, next morning starts fresh at $0.01.
# Cap: never pay more than PMCC_BTC_STEP_CAP_PCT of the original premium received.
PMCC_BTC_CHEAP_ENABLED = True             # Master toggle for cheap BTC stepping
PMCC_BTC_CHEAP_MAX_DELTA = 0.05           # Only attempt cheap BTC when delta <= this
PMCC_BTC_CHEAP_MAX_VALUE = 0.10           # Only attempt when BS theo value <= $0.10
PMCC_BTC_CHEAP_START_PRICE = 0.01         # Initial BTC limit price ($0.01)
PMCC_BTC_CHEAP_STEP = 0.01               # Step up by $0.01 per interval
PMCC_BTC_STEP_INTERVAL_SEC = 300          # Step up every 5 minutes (300 seconds)
PMCC_BTC_STEP_CAP_PCT = 0.50             # Max BTC price = 50% of original premium received

# --- DTE-Dependent Max Buyback Price ---
# First half of contract life: max buyback = 30% of sell price (more generous)
# Second half: max buyback = 20% of sell price (tighter as theta accelerates)
PMCC_BUYBACK_MAX_FIRST_HALF = 0.30        # 30% of sell price when DTE > original_DTE/2
PMCC_BUYBACK_MAX_SECOND_HALF = 0.20       # 20% of sell price when DTE <= original_DTE/2

# --- Crash Mode ---
# When breadth gate triggers OR VIX > threshold, close ALL short calls at
# market/ask price, ignoring max buyback caps. Don't resell for 2-3 days.
PMCC_CRASH_MODE_VIX = 30                  # VIX threshold for crash mode

# --- Price Ladder Schedule ---
# When profit target is hit, submit buy-back at theoretical value first,
# then escalate through mid and ask prices on a timed schedule.
# In crash mode: skip ladder, go straight to ask.
PMCC_BUYBACK_INITIAL_WAIT_MINUTES = 15    # Minutes before bumping from theo to mid
PMCC_BUYBACK_MID_WAIT_MINUTES = 30        # Minutes before bumping from mid to ask
PMCC_BUYBACK_MAX_PCT_OF_SOLD = 0.20       # Legacy alias (use FIRST_HALF/SECOND_HALF instead)

# Override swing entry: buy LEAP instead of 100 shares
SWING_MIN_LOT_SIZE_PMCC = 1              # 1 LEAP contract = 100 share equivalent

# =============================================================================
# ENTRY ROUTING — Intelligent LEAP vs Shares Decision (Mar 16)
# =============================================================================
# Per-signal routing replaces the old PMCC_MODE global boolean.
# For each RSI2 buy signal, the router checks:
#   1. Is ticker in CC_OPTIONS_ELIGIBLE?
#   2. Is a qualifying LEAP available?
#   3. Does the LEAP save >LEAP_CAPITAL_EFFICIENCY_THRESHOLD vs 100 shares?
# If yes to all three: buy LEAP.  Otherwise: buy 100 shares.
# Signals are NEVER silently dropped.

LEAP_CAPITAL_EFFICIENCY_THRESHOLD = 0.30  # LEAP must save >30% vs shares to prefer it

# =============================================================================
# AVERAGING DOWN (second lot on extreme oversold)
# =============================================================================
# If RSI(2) drops below AVERAGING_DOWN_RSI2_THRESHOLD while already holding a
# position that is underwater, allow a second lot at the same size.
# Capped at MAX_LOTS_PER_TICKER — never buy a third lot.

MAX_LOTS_PER_TICKER = 2                    # Maximum lots (entries) per ticker
AVERAGING_DOWN_RSI2_THRESHOLD = 5          # RSI(2) must be < 5 (extreme oversold)

# =============================================================================
# LAYER 4: CALL BUYING ON DIPS
# =============================================================================
# Mirror of Layer 3 CC sell system. Buys calls when miners are oversold and
# flips them on the bounce. Reference: ~/trading_bot/CALL_BUYING_SPEC.md

CALL_BUYER_ENABLED = True                  # Master toggle

# Entry signals (inverse of Layer 3 CC sell signals)
L4_TICKER_DOWN_PCT = -0.03                 # Ticker down 3%+ on the day
L4_RSI_THRESHOLD = 35                      # RSI(14) < 35
L4_RSI_PERIOD = 14                         # Same RSI period as Layer 3
L4_VOL_EXTREME = True                      # HV-10 > HV-20 OR HV-10 at 52-wk high
L4_GLD_PULLBACK_PCT = -0.003              # GLD down 0.3%+ on day
L4_USO_PULLBACK_PCT = -0.005              # USO down 0.5%+ on day (oil is more volatile)
L4_UUP_5D_UP = 0.002                      # UUP up 0.2%+ over 5 days
L4_UUP_INTRADAY_DOWN = -0.001             # UUP down intraday (dollar reversal)
L4_GLD_SHORT_TERM_DN_THRESHOLD = -0.003   # GLD NOT still dumping (wait for flush)
L4_USO_SHORT_TERM_DN_THRESHOLD = -0.005   # USO NOT still dumping (wait for flush)
L4_MIN_BUY_SIGNALS = 4                    # Minimum 4 of 6 signals

# RSI(2) coordination with Layer 1
L4_RSI2_SUPPLEMENTAL_RANGE = (15, 30)     # L4 fires independently in this range
L4_SKIP_IF_L1_ENTERING = True             # Skip if L1 entering same ticker

# Contract selection
L4_CALL_DELTA_MIN = 0.30
L4_CALL_DELTA_MAX = 0.50
L4_CALL_DELTA_TARGET = 0.40
L4_CALL_DTE_MIN = 30
L4_CALL_DTE_MAX = 60
L4_CALL_DTE_OPTIMAL_MIN = 35
L4_CALL_DTE_OPTIMAL_MAX = 50
L4_MAX_OTM_PCT = 0.05
L4_MAX_ITM_PCT = 0.03
L4_MIN_OPEN_INTEREST = 50
L4_MAX_BID_ASK_SPREAD_PCT = 0.15
L4_MIN_MID_PRICE = 0.15

# Exit rules
L4_PROFIT_TARGET_PCT = 0.30
L4_STRETCH_TARGET_PCT = 0.50
L4_TRAILING_ACTIVATE_PCT = 0.20
L4_TRAILING_STOP_PCT = 0.40
L4_THETA_EXIT_DTE = 21
L4_MAX_HOLD_DAYS = 7
L4_STOP_LOSS_PCT = 0.40
L4_EXIT_AT_5SMA = True

# Position sizing
L4_MAX_POSITION_PCT = 0.015
L4_MAX_POSITION_DOLLARS = 1500
L4_MIN_POSITION_DOLLARS = 200
L4_MAX_CONTRACTS_PER_TRADE = 5
L4_MAX_CONCURRENT_POSITIONS = 4
L4_MAX_PORTFOLIO_PCT = 0.05
L4_MAX_PER_SECTOR = 4

# Risk controls
L4_DAILY_LOSS_LIMIT = 0.005
L4_WEEKLY_LOSS_LIMIT = 0.015
L4_IV_RANK_MAX = 80
L4_IV_RANK_PREFERRED_MAX = 60
L4_IV_RANK_PENALTY = True
L4_EARNINGS_BLACKOUT_DAYS = 3
L4_CHECK_CC_CONFLICT = True

# Scheduling
L4_NO_ENTRY_BEFORE = (10, 0)
L4_NO_NEW_ENTRIES_AFTER = (15, 0)
L4_FORCE_CLOSE_TIME = (15, 30)

# Order execution
L4_BUY_OFFSET_FROM_MID = 0.03
L4_SELL_OFFSET_FROM_MID = -0.02
L4_ORDER_TIF = "day"
L4_ORDER_MAX_WAIT = 60

# Regime adjustments
L4_REGIME_ADJUSTMENTS = {
    'rip_likely':        {'signal_threshold_boost': -0.08},
    'correction_likely': {'signal_threshold_boost':  0.10},
    'chop_likely':       {'signal_threshold_boost':  0.00},
    'uncertain':         {'signal_threshold_boost':  0.00},
}

# --- Backward-compatible aliases for combined_runner.py ---
CALL_BUYER_MAX_POSITIONS = L4_MAX_CONCURRENT_POSITIONS
CALL_BUYER_MAX_ALLOCATION_PCT = L4_MAX_PORTFOLIO_PCT * 100
CALL_BUYER_PER_POSITION_PCT = L4_MAX_POSITION_PCT * 100
CALL_BUYER_PROFIT_TARGET = L4_PROFIT_TARGET_PCT
CALL_BUYER_STOP_LOSS = L4_STOP_LOSS_PCT
CALL_BUYER_THETA_CUTOFF_DTE = L4_THETA_EXIT_DTE
CALL_BUYER_MAX_HOLD_DAYS = L4_MAX_HOLD_DAYS
CALL_BUYER_IV_RANK_MAX = L4_IV_RANK_MAX
CALL_BUYER_MIN_DELTA = L4_CALL_DELTA_MIN
CALL_BUYER_MAX_DELTA = L4_CALL_DELTA_MAX
CALL_BUYER_MIN_DTE = L4_CALL_DTE_MIN
CALL_BUYER_MAX_DTE = L4_CALL_DTE_MAX

# =============================================================================
# TA OVERLAYS — Karim/Wadsworth-Inspired Regime Modifiers (Mar 18)
# =============================================================================
# Three signal overlays that modulate position sizing and gate entries.
# These are NOT entry signals — RSI2 entry logic is unchanged.
# Reference: ~/research/karim_wadsworth_ta_study.md

# --- Master toggles ---
TA_OVERLAY_GSR_ENABLED = True          # Overlay 1: Gold/Silver Ratio Regime
TA_OVERLAY_RED_ZONE_ENABLED = True     # Overlay 2: Distance from 60-day MA
TA_OVERLAY_GOLD_SPX_ENABLED = True     # Overlay 3: Gold/SPX Relative Strength

# --- Overlay 1: Gold/Silver Ratio Regime ---
# Track GLD/SLV ratio; falling = silver outperforming = bullish for silver miners
TA_GSR_MA_SHORT = 10                   # Short MA period for G/S ratio
TA_GSR_MA_LONG = 20                    # Long MA period for G/S ratio
TA_GSR_BULLISH_MULT = 1.2             # Size multiplier when ratio falling (silver outperforming)
TA_GSR_BEARISH_MULT = 0.7             # Size multiplier when ratio rising (gold outperforming)
# Applied only to silver-sector tickers (TICKER_SECTOR values in SILVER_SECTORS)
SILVER_SECTORS = {"silver_mining"}

# --- Overlay 2: Distance from 60-day MA ("Red Zone") ---
# Wadsworth: 20%+ above 60-day MA = "red zone" (overbought)
TA_RED_ZONE_PCT = 0.15                 # >15% above 60MA -> skip new RSI2 entries
TA_RED_ZONE_CC_FLAG_PCT = 0.20         # >20% above 60MA -> flag for Layer 3 CC selling

# --- Overlay 3: Gold/SPX Relative Strength ---
# GLD/SPY ratio trend: rising = gold outperforming = mining tailwind
TA_GOLD_SPX_MA_SHORT = 10             # Short MA period for GLD/SPY ratio
TA_GOLD_SPX_MA_LONG = 20              # Long MA period for GLD/SPY ratio
TA_GOLD_SPX_UNDERPERFORM_MULT = 0.8   # Size multiplier when gold underperforming stocks
# Applied to all mining tickers (not energy)
MINING_OVERLAY_SECTORS = {
    "silver_mining", "gold_mining", "copper_mining", "metals",
}

# Tickers required for TA overlay calculations (fetched alongside trading universe)
TA_OVERLAY_WATCH_TICKERS = ["GLD", "SLV", "SPY"]

# ── Falling Knife Guard ──────────────────────────────────────────────
OPENING_BUFFER_MINUTES = 15          # Skip first 15 min after open (9:30-9:45)
GAP_DOWN_THRESHOLD_PCT = 5.0         # If ticker gaps down >5%, extend delay
GAP_DOWN_EXTRA_DELAY_MINUTES = 15    # Extra 15 min delay on big gap-downs (total 30 min)
FALLING_KNIFE_GUARD_ENABLED = True

# ── Sector Breadth Gate (Falling Knife Guard #2) ────────────────────
# When >70% of the trading universe has RSI(2) < 15, delay ALL new entries
# by 1 trading day. Prevents buying into a sector-wide cascade.
BREADTH_GATE_ENABLED = True
BREADTH_GATE_THRESHOLD = 0.70        # 70% of universe must be oversold to trigger
BREADTH_GATE_RSI_THRESHOLD = 10      # RSI(2) level considered oversold (matched to entry threshold)
BREADTH_GATE_DELAY_DAYS = 1          # Trading days to delay (not calendar days)

# ── VIX-Based Position Sizing (Falling Knife Guard #3) ──────────────
# Scale position sizes based on VIX level. Higher VIX = smaller positions.
VIX_SIZING_ENABLED = True
VIX_TIERS = [(25, 1.0), (30, 0.75), (35, 0.50), (40, 0.25)]  # (threshold, multiplier)
VIX_PAUSE_THRESHOLD = 40             # VIX > 40: pause new entries entirely

# =============================================================================
# ITM PUT ENTRY (Apr 11 — replaces share buying for RSI2 entries)
# =============================================================================
# Instead of buying 100 shares on RSI2 signal, sell ITM put to enter at discount.
# 2-3% edge per trade from cost basis reduction (backtested Apr 11).
#
# Process:
# 1. RSI2 fires buy signal
# 2. Instead of market buy, sell 5% ITM put (~30 DTE)
# 3. If assigned: own shares at (strike - premium) = effective discount
# 4. If not assigned (stock ripped): keep premium, re-evaluate
#
ITM_PUT_ENTRY_ENABLED = True

# Edge 106 — keep ITM_PUT route disabled until itm_put_manager.enter()
# actually places broker orders. Setting True without implementing
# enter() will resume writing phantom state (Edge 106 root cause).
ITM_PUT_LIVE = False
ITM_PUT_STRIKE_PCT = 1.05        # 5% ITM — best time value per dollar of capital (3-4%)
ITM_PUT_MIN_DTE = 21             # Minimum 21 days to expiry
ITM_PUT_MAX_DTE = 45             # Maximum 45 days to expiry

# =============================================================================
# TAIL HEDGE — OTM Put Protection (Apr 11 — Spitznagel approach)
# =============================================================================
# Buy long-dated OTM puts on GDX as portfolio crash insurance.
# Buy 1-year, roll at 6 months remaining to minimize theta cost.
#
# Budget: ~3% of portfolio annually
# When to initiate: GDX HV20 < 30% AND/OR GDX/GLD ratio < 1.8
# When NOT to buy: if puts already expensive (HV20 > 50%)
#
TAIL_HEDGE_ENABLED = True        # Master toggle
TAIL_HEDGE_TICKER = "GDX"        # Hedge vehicle
TAIL_HEDGE_STRIKE_PCT = 0.80     # 20% OTM
TAIL_HEDGE_TARGET_DTE = 365      # Buy ~1 year out
TAIL_HEDGE_ROLL_DTE = 180        # Roll when 6 months remaining
TAIL_HEDGE_MAX_ANNUAL_COST_PCT = 0.04  # Max 4% of portfolio/year
TAIL_HEDGE_CONTRACTS = 9         # For ~$95K portfolio
TAIL_HEDGE_BUY_WHEN_HV_BELOW = 0.30   # Only buy when GDX HV20 < 30%

# =============================================================================
# MSA REGIME ROUTING (Apr 12 — Oliver-style momentum structural analysis)
# =============================================================================
# MSA regime determines WHICH instrument to use for entries:
#
# STRONG_BULL (P&F UP, momentum >20%): Buy ATM (0.50Δ) LEAPs — max leverage
# BULL (P&F UP, momentum 0-20%):       Buy 0.70Δ LEAPs — moderate leverage
# BULL_CORRECTION (P&F DOWN, mom >0%):  Sell ITM puts — income, discounted entry
# CHOP (P&F mixed, mom near 0%):        Sell ITM puts — wait for direction
# BEAR (P&F DOWN, momentum <0%):        Cash or OTM puts only — no longs
#
# Run msa_indicator.py weekly to update regime classification.

MSA_ROUTING_ENABLED = True

MSA_REGIME_INSTRUMENT = {
    'STRONG_BULL':     {'instrument': 'LEAP', 'delta': 0.80, 'reason': 'High conviction: more leverage justified, 19% theta, 3.4x'},
    'BULL':            {'instrument': 'LEAP', 'delta': 0.90, 'reason': 'Less certain: minimize theta (15%), max intrinsic protection'},
    'BULL_CORRECTION': {'instrument': 'ITM_PUT', 'delta': None, 'reason': 'Income, wait for structure test'},
    'CHOP':            {'instrument': 'ITM_PUT', 'delta': None, 'reason': 'No direction, collect premium'},
    'BEAR_BOUNCE':     {'instrument': 'NONE', 'delta': None, 'reason': 'Light longs only, defensive'},
    'STRONG_BEAR':     {'instrument': 'NONE', 'delta': None, 'reason': 'Cash/puts, no new longs'},
}

# MSA check frequency
MSA_CHECK_INTERVAL_DAYS = 7  # Re-run MSA weekly
MSA_INDICATOR_SCRIPT = "~/trading_bot/msa_indicator.py"
