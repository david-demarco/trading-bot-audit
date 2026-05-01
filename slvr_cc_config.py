"""
slvr_cc_config.py - Configuration constants for the deep OTM covered call
premium scalping strategy on metals mining ETFs.

All tunable parameters live here for easy adjustment without modifying the
core logic in slvr_cc_scalper.py.

Strategy summary:
  Hold shares of metals mining ETFs, sell deep OTM long-dated covered calls
  into IV/price spikes, buy them back on IV crush or pullbacks, repeat.
  This is a VEGA trade -- premium moves are driven by IV changes, not theta.

Reference: ~/trading_bot/research/deep_otm_cc_strategy.md
"""

from __future__ import annotations

# =============================================================================
# TRADEABLE UNIVERSE — one flat list (Mar 16: flattened ETF/stock split)
# =============================================================================

# Full mining universe — ETFs and individual stocks in one list.
# GOAU, GOEX removed Mar 12 (David: garbage spreads, 80-146% bid-ask)
TRADE_TICKERS = [
    # Sprott ETFs
    "SLVR",   # Sprott Silver Miners & Physical Silver ($70)
    "SGDM",   # Sprott Gold Miners ($85)
    "SGDJ",   # Sprott Junior Gold Miners ($100)
    "GBUG",   # Sprott Active Gold & Silver Miners ($51)
    "COPP",   # Sprott Copper Miners ($38)
    "COPJ",   # Sprott Junior Copper Miners ($43)
    # Non-Sprott ETFs
    "SIL",    # Global X Silver Miners (~$94)
    "SILJ",   # Amplify Junior Silver Miners ($34)
    "GDX",    # VanEck Gold Miners ($102)
    "GDXJ",   # VanEck Junior Gold Miners ($137)
    "COPX",   # Global X Copper Miners ($82)
    "SLVP",   # iShares Global Silver Miners ($40)
    "RING",   # iShares Global Gold Miners ($88)
    # Individual miners
    "PAAS",   # Pan American Silver ($26)
    "BTG",    # B2Gold ($4)
    "WPM",    # Wheaton Precious Metals ($72)
    "AG",     # First Majestic Silver ($8)
    "HL",     # Hecla Mining ($8)
    "FSM",    # Fortuna Mining ($7)
    "SVM",    # Silvercorp Metals ($5)
    "USAS",   # Americas Gold and Silver ($1)
    "CDE",    # Coeur Mining ($8)
    "EGO",    # Eldorado Gold ($21)
    "KGC",    # Kinross Gold ($13)
    # Oil / Energy (added Mar 18)
    "XLE",    # Energy Select ETF (~$59)
    "OXY",    # Occidental Petroleum (~$57)
    "HAL",    # Halliburton (~$35)
    "DVN",    # Devon Energy (~$47)
    "SLB",    # Schlumberger (~$45)
    "XOP",    # SPDR S&P Oil & Gas ETF (~$168)
    "COP",    # ConocoPhillips (~$115)
    "TPL",    # Texas Pacific Land (~$1400+, LEAP-preferred)
    "EQT",    # EQT Corp, natural gas (~$50)
]

# STOCK_TICKERS alias removed (M4 audit fix 2026-03-18) — no code imports it.

# 2x leveraged — on radar, enable after non-leveraged is dialed in
# Harder: leverage decay eats shares, getting stuck is 2x worse,
# premium must compensate for drag. Needs shorter share holding periods.
LEVERAGED_TICKERS = [
    "NUGT",   # 2x Gold Miners Bull ($242, 11 exps)
    "JNUG",   # 2x Junior Gold Miners Bull ($271, 5 exps)
]
# AGQ (2x Silver) and UGL (2x Gold) are leveraged METAL, not miners — excluded

# Set to True to include leveraged tickers in active trading
ENABLE_LEVERAGED = False

# =============================================================================
# SIGNAL TICKERS — not traded, used for signal generation
# =============================================================================

# Leading indicator -- GLD Granger-causes SLVR (p=0.0135)
GLD_TICKER = "GLD"

# Dollar index ETF -- inverse correlation with metals
UUP_TICKER = "UUP"

# Silver spot proxy -- used for additional confirmation
SLV_TICKER = "SLV"

# All tickers the data layer needs to track (trade + signal)
WATCH_TICKERS = (
    TRADE_TICKERS
    + (LEVERAGED_TICKERS if ENABLE_LEVERAGED else [])
    + [GLD_TICKER, UUP_TICKER, SLV_TICKER, "USO", "SPY"]
)

# =============================================================================
# STRIKE SELECTION
# =============================================================================

# OTM percentage range (as decimals: 0.35 = 35% OTM)
STRIKE_OTM_MIN = 0.35          # minimum 35% OTM
STRIKE_OTM_MAX = 0.50          # maximum 50% OTM
STRIKE_OTM_TARGET = 0.40       # ideal target: 40% OTM

# DTE range
DTE_MIN = 120                  # minimum days to expiration
DTE_MAX = 220                  # maximum days to expiration
DTE_OPTIMAL_MIN = 150          # optimal range lower bound
DTE_OPTIMAL_MAX = 180          # optimal range upper bound

# Premium floor (per-share price, NOT per-contract)
MIN_PREMIUM = 3.00             # minimum $3.00/share ($300/contract)

# Minimum open interest for liquidity
MIN_OPEN_INTEREST = 10         # SLVR options are thin, keep this realistic
MIN_VOLUME = 0                 # daily volume floor (0 = no requirement)

# Maximum bid-ask spread as fraction of mid-price
# Filters out illiquid options (e.g. GOAU/GOEX with 80-146% spreads)
MAX_BID_ASK_SPREAD_PCT = 0.50  # 50% spread = skip

# =============================================================================
# SELL SIGNALS (open new CC positions -- sell calls)
# =============================================================================

# SLVR price move threshold (sell into strength)
SELL_SLVR_UP_PCT = 0.03        # SLVR must be up 3%+ on the day

# RSI thresholds
SELL_RSI_THRESHOLD = 65        # RSI(14) must be > 65 (approaching overbought)
RSI_PERIOD = 14                # RSI lookback period

# Volatility regime
# HV-10 > HV-20 means vol is expanding (IV likely elevated)
HV_SHORT_WINDOW = 10           # short-term HV window
HV_LONG_WINDOW = 20            # long-term HV window

# GLD must be rallying (sector momentum)
GLD_RALLY_PCT = 0.005          # GLD up 0.5%+ on the day

# UUP must be weakening (dollar down = silver bullish)
UUP_WEAK_PCT = -0.001          # UUP down on the day (any amount)

# Silver direction: GLD must NOT be trending up in last 15-30 minutes
# If GLD is still ripping, don't sell calls -- wait for the spike to peak
GLD_SHORT_TERM_UP_THRESHOLD = 0.003   # if GLD up >0.3% in last 15min, skip

# --- Oil / Energy macro indicators (Mar 18) ---
# Energy tickers use USO (oil proxy) instead of GLD for signals 4 & 6.
USO_TICKER = "USO"
USO_RALLY_PCT = 0.008                 # USO up 0.8%+ on the day (oil is more volatile than gold)
USO_SHORT_TERM_UP_THRESHOLD = 0.005   # if USO up >0.5% in last 15min, skip

# Minimum number of sell signals that must align (out of 6 total)
MIN_SELL_SIGNALS = 4

# =============================================================================
# BUY BACK SIGNALS (close existing CC positions -- buy back calls)
# =============================================================================

# 50% profit target (buy back when premium drops to 50% of sell price)
BUYBACK_PROFIT_TARGET = 0.50

# SLVR pullback threshold (from the price on sell date)
BUYBACK_SLVR_DROP_PCT = 0.05   # 5% drop from sell-date price

# RSI oversold level
BUYBACK_RSI_THRESHOLD = 35     # RSI < 35 triggers buy-back

# IV crush threshold (absolute points)
BUYBACK_IV_CRUSH_POINTS = 15   # if IV has dropped 15+ points since sell

# DTE remaining -- force close/roll at this point
BUYBACK_DTE_REMAINING = 60     # close at 60 DTE remaining

# Silver direction: GLD turning up after pullback -> buy back before premium rises
GLD_TURNING_UP_THRESHOLD = 0.002  # GLD up 0.2%+ in last 15min after a down day
USO_TURNING_UP_THRESHOLD = 0.003  # USO up 0.3%+ in last 15min after a down day (oil more volatile)

# =============================================================================
# POSITION SIZING & RISK
# =============================================================================

# Hard daily contract limit per chain (broker compliance)
MAX_CONTRACTS_PER_CHAIN_PER_DAY = 50  # ABSOLUTE MAX -- no exceptions

# Volume participation limits
VOLUME_WARN_PCT = 0.20         # warn at 20% of chain's daily volume
VOLUME_HARD_STOP_PCT = 0.25    # hard stop at 25% of chain's daily volume

# Starting size
DEFAULT_CONTRACTS = 10         # start at 10 contracts per trade
MAX_CONTRACTS_PER_TRADE = 20   # scale up to 20 max per trade

# Maximum concurrent open positions (different strikes/expirations)
MAX_CONCURRENT_POSITIONS = 3

# =============================================================================
# ORDER EXECUTION
# =============================================================================

# Alpaca paper trading endpoint
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"

# Order type -- ALWAYS limit orders
# Sell above mid (into urgency), buy back below mid (into panic)
SELL_OFFSET_FROM_MID = 0.05    # sell $0.05 above mid-price
BUYBACK_OFFSET_FROM_MID = 0.05 # buy $0.05 below mid-price

# Order time in force
ORDER_TIF = "day"              # day orders for options

# --- Active Order Management ---
# Controls the cancel/replace loop that actively manages working orders.
# David manages orders in SECONDS, but Alpaca limits are 200 req/min global,
# 10 req/sec burst.  Each cancel + re-post = 2 requests.
ORDER_CHECK_INTERVAL = 3        # seconds between order-status checks
ORDER_MAX_WAIT = 30             # max seconds before first price adjustment
ORDER_MAX_ATTEMPTS = 10         # max cancel/replace cycles before giving up
ORDER_PRICE_STEP = 0.02         # $ per adjustment toward mid-price
ORDER_MIN_INTERVAL = 3          # minimum seconds between cancel/replace on same order
RATE_LIMIT_WINDOW = 60          # sliding window in seconds for rate tracking
RATE_LIMIT_MAX_REQUESTS = 180   # leave 20 req/min headroom (Alpaca max = 200)

# =============================================================================
# ALPACA CREDENTIALS
# =============================================================================

ALPACA_USER_ID = "a4dc8459-608d-49f5-943e-e5e105ed5207"

# =============================================================================
# STATE & LOGGING
# =============================================================================

STATE_FILE = "slvr_cc_state.json"
LOG_FILE = "logs/slvr_cc_scalper.log"
LOG_MAX_BYTES = 10 * 1024 * 1024   # 10 MB
LOG_BACKUP_COUNT = 5

# =============================================================================
# SCHEDULING
# =============================================================================

# Market hours (ET)
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (16, 0)

# Poll interval for signal checks (seconds)
POLL_INTERVAL_SECONDS = 120    # check every 2 minutes

# Minimum wait between trades on same chain (seconds)
MIN_TRADE_INTERVAL_SECONDS = 300  # 5 minutes between trades

# Risk-free rate for Black-Scholes (approximate)
RISK_FREE_RATE = 0.045         # 4.5% (current approximate short-term rate)

# =============================================================================
# DATA LAYER
# =============================================================================

# yfinance history periods for various calculations
YF_HISTORY_PERIOD = "6mo"      # for RSI, HV calculations
YF_INTRADAY_PERIOD = "5d"      # for intraday GLD direction monitoring
YF_INTRADAY_INTERVAL = "5m"    # 5-minute bars for intraday

# Alpaca feed (free tier)
ALPACA_FEED = "iex"

# =============================================================================
# PMCC / DIAGONAL SPREAD CONFIGURATION
# =============================================================================
# All PMCC params are defined in combined_config.py (single source of truth).
# Imported here for backward compatibility with pmcc_manager.py and
# slvr_cc_scalper.py which import from this module.
#
# "You don't have to have shares if you have LEAPS" -- David

from combined_config import (  # noqa: E402
    PMCC_ENABLED,
    PMCC_LONG_LEG_MIN_DTE,
    PMCC_LONG_LEG_MIN_DELTA,
    PMCC_LONG_LEG_MAX_DELTA,
    PMCC_SHORT_MAX_DELTA as PMCC_MAX_SHORT_DELTA,
    PMCC_SHORT_MIN_DELTA,
    PMCC_SHORT_REQUIRE_LIVE_QUOTE,
    PMCC_ASSIGNMENT_BUFFER_PCT,
    PMCC_MAX_RISK_RATIO,
    PMCC_MIN_NET_CREDIT_TARGET,
    PMCC_SHORT_DELTA_WARN,
    PMCC_SHORT_DELTA_DANGER,
    PMCC_SHORT_PROFIT_TARGET_PCT,
    PMCC_AUTO_RESELL,
    PMCC_BUYBACK_INITIAL_WAIT_MINUTES,
    PMCC_BUYBACK_MID_WAIT_MINUTES,
    PMCC_BUYBACK_MAX_PCT_OF_SOLD,
    PMCC_SHORT_DTE_MIN,
    PMCC_SHORT_DTE_MAX,
    PMCC_SHORT_DTE_OPTIMAL,
    PMCC_MAX_CONCURRENT_SPREADS,
    # v2 buyback params (Mar 19)
    PMCC_PROFIT_TARGET_EARLY,
    PMCC_PROFIT_TARGET_LATE,
    PMCC_CLOSE_DTE,
    PMCC_LET_EXPIRE_DTE,
    PMCC_LET_EXPIRE_DELTA,
    PMCC_LET_EXPIRE_VALUE,
    PMCC_BUYBACK_MAX_FIRST_HALF,
    PMCC_BUYBACK_MAX_SECOND_HALF,
    PMCC_CRASH_MODE_VIX,
    BREADTH_GATE_ENABLED,
    # Cheap BTC stepping params (Mar 25)
    PMCC_BTC_CHEAP_ENABLED,
    PMCC_BTC_CHEAP_MAX_DELTA,
    PMCC_BTC_CHEAP_MAX_VALUE,
    PMCC_BTC_CHEAP_START_PRICE,
    PMCC_BTC_CHEAP_STEP,
    PMCC_BTC_STEP_INTERVAL_SEC,
    PMCC_BTC_STEP_CAP_PCT,
)

# =============================================================================
# REGIME DETECTION
# =============================================================================
# Correlation-based regime prediction layer.  Predicts whether the metals
# market is about to RIP (rally), CHOP (consolidate), or CORRECT (sell off).
# The CC scalper uses this prediction to modulate how aggressively it sells
# premium.  See regime_detector.py for full implementation.

# Master toggle
REGIME_ENABLED = True

# How far back to look for correlation data (days)
REGIME_LOOKBACK_DAYS = 90

# Cache duration for regime data -- do not re-fetch every 90-second cycle
REGIME_CACHE_MINUTES = 60

# Below this confidence level, classify as UNCERTAIN and use default params
REGIME_CONFIDENCE_THRESHOLD = 0.60

# Signal weights (must sum to ~1.0)
# Backtest-validated ranking: Pd lead (#1), vol structure (#2), Cu/Au (#3)
# Cross-ticker divergence demoted — weakest signal in backtest
REGIME_WEIGHTS = {
    'gold_silver_ratio': 0.15,       # GLD/SLV ratio rate of change
    'palladium_lead': 0.25,          # PALL momentum vs SLV — best predictor
    'copper_gold': 0.15,             # CPER/GLD ratio trend
    'dollar_trend': 0.15,            # UUP 20-day EMA slope
    'yield_signal': 0.08,            # TLT momentum (yield proxy) — weak
    'cross_ticker_divergence': 0.07, # dispersion — weakest signal, demoted
    'volatility_structure': 0.15,    # HV-10 vs HV-30 — 2nd best predictor
}

# Parameter adjustments per regime
# BACKTEST-VALIDATED (Mar 10): Detector is ~35% accurate on directional calls.
# Narrowed adjustment range from 35-55% to 42-48% OTM — modest nudges, not
# aggressive swings.  Detector's main value is UNCERTAIN classification (keeps
# us at defaults 86% of the time) and gentle parameter modulation.
REGIME_ADJUSTMENTS = {
    'rip_likely': {
        'min_otm_pct': 0.48,             # wider OTM — rally risk
        'signal_threshold_boost': 0.08,   # higher bar to sell
        'max_positions_mult': 0.75,       # reduce size
    },
    'chop_likely': {
        'min_otm_pct': 0.38,             # tighter OTM — premium decays fast
        'signal_threshold_boost': -0.05,  # easier to trigger
        'max_positions_mult': 1.25,       # increase size
    },
    'correction_likely': {
        'min_otm_pct': 0.46,             # wider OTM — reversals are fast in super cycle
        'signal_threshold_boost': 0.05,   # slightly higher bar
        'max_positions_mult': 0.85,       # slight size reduction
    },
    'uncertain': {
        'min_otm_pct': None,             # use config default (86% of predictions)
        'signal_threshold_boost': 0.0,
        'max_positions_mult': 1.0,
    },
}
