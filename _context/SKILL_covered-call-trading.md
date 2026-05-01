---
name: covered-call-trading
description: Use this skill when managing the integrated RSI2 swing + covered call trading system -- one system, three layers, applied across the full mining universe (26 tickers, one flat TRADING_UNIVERSE list).
---

# Integrated RSI2 Swing + Covered Call System

## ⚠️ DEPRECATED SETTINGS — SEE rsi2-trading SKILL
**The rsi2-trading skill (Apr 9) is the canonical reference.** This file documents the older combined_runner.py system which has NOT been updated. Key differences flagged below with ⚠️.

## ONE System, FOUR Layers (updated Apr 9 2026)

### Layer 1: Put Sell (NEW Apr 9)
- RSI2<10 → sell 5% OTM cash-secured put (~30 day expiry)
- If options unavailable, falls back to buying 100 shares
- VIX>25 = sell put only (fat premiums); VIX<25 = put or stock

### ⚠️ OLD Layer 1: RSI2 Entry (combined_runner.py — NOT UPDATED)
- Buy 100 shares when RSI(2) < **15** ⚠️ DEPRECATED: backtested threshold is **10**
- $0.50/share bonus profit target if it bounces fast
- Applies to ALL tickers in the universe -- ETFs and stocks alike

### Layer 2: Swing Hold
- Holds positions waiting for mean-reversion bounce
- ⚠️ **Primary exit**: RSI2-based (per-ticker: >70 default, >90 for KGC). **NOT SMA5** — RSI2 exit beats SMA5 by 2-3x Sharpe.
- **-8% stop loss** on stock positions ⚠️ CHANGED: RSI2 backtest includes stops. Old "no stop loss" rule was for the SMA5-exit system.
- Safety valve: force close after 10 trading days -- **UNLESS a covered call is active** (see Exit Logic)
- Once held 5+ days without bouncing, becomes eligible for Layer 3

### Layer 3: Covered Calls (options-eligible tickers only)
- Sells covered calls against 100-share positions held 5+ days
- Premium income layer -- the real profit center for this strategy
- 0.30 delta, 14-28 DTE, 50% profit target, roll at 7 DTE
- Only fires on tickers with Alpaca options chains (CC_OPTIONS_ELIGIBLE subset)

**The flow:** RSI2 oversold entry -> hold for 5-SMA crossover -> exit on bounce OR CC overlay after 5 days

### PMCC as Alternative Long Exposure
PMCC/LEAP positions are part of the SAME system -- they're a capital-efficient way to get long exposure
(0.9 delta LEAPs instead of 100 shares). Same three layers apply; the LEAP replaces 100 shares as the
long leg, and short OTM calls are sold against it.

David scalps calls manually (human edge: reads momentum in real time).
The bots do the mechanical mean-reversion version.

## The Process
1. **Size before signal.** What can I hold 100 shares of? Filter to those. Then look for entries.
2. **100-share lots are non-negotiable.** Can't sell covered calls without them.
3. **Position sizing handles capital allocation.** No arbitrary price cap. 100 shares of GDX (~$10.2k) or GDXJ (~$13.7k) fits within the 15% position cap on a $100k account. Position sizing code skips any ticker where 100 shares exceeds the cap.
4. **Position cap: 15%** of portfolio ($15,000 on $100k).
5. **Max 5 positions** total across all strategies. Fewer, bigger, CC-ready.
6. **5-day seasoning** before selling calls (CC_ELIGIBLE_AFTER_DAYS=5). Position that hasn't bounced in 5 days = pivot from equity play to income play.

## Trading Universe (26 tickers -- one flat list, Mar 16)

All tickers participate in Layers 1-2 (RSI2 entry + swing hold). Layer 3 (CC overlay) only fires on tickers in CC_OPTIONS_ELIGIBLE. No ETF/stock split -- one universe, one list (`TRADING_UNIVERSE` in combined_config.py, `TRADE_TICKERS` in slvr_cc_config.py).

| Ticker | Sector | Description | Approx Price | CC-Eligible |
|--------|--------|-------------|-------------|-------------|
| SLVR | Silver | Sprott Silver Miners & Physical Silver | ~$70 | Yes |
| SGDM | Gold | Sprott Gold Miners | ~$85 | Yes |
| SGDJ | Gold | Sprott Junior Gold Miners | ~$100 | |
| GBUG | Gold | Sprott Active Gold & Silver Miners | ~$51 | |
| COPP | Copper | Sprott Copper Miners | ~$38 | |
| COPJ | Copper | Sprott Junior Copper Miners | ~$43 | |
| SIL | Silver | Global X Silver Miners | ~$94 | Yes |
| SILJ | Silver | Amplify Junior Silver Miners | ~$34 | Yes |
| GDX | Gold | VanEck Gold Miners | ~$102 | Yes |
| GDXJ | Gold | VanEck Junior Gold Miners | ~$137 | Yes |
| COPX | Copper | Global X Copper Miners | ~$82 | Yes |
| SLVP | Silver | iShares Global Silver Miners | ~$40 | |
| RING | Gold | iShares Global Gold Miners | ~$88 | Yes |
| PAAS | Silver | Pan American Silver | ~$26 | Yes |
| BTG | Gold | B2Gold | ~$4 | Yes |
| WPM | Gold | Wheaton Precious Metals | ~$72 | Yes |
| AYA | Gold | Aya Gold & Silver | ~$16 | |
| AG | Silver | First Majestic Silver | ~$8 | Yes |
| HL | Silver | Hecla Mining | ~$8 | Yes |
| FSM | Silver | Fortuna Mining | ~$7 | Yes |
| SVM | Silver | Silvercorp Metals | ~$5 | Yes |
| USAS | Silver | Americas Gold and Silver | ~$1 | |
| CDE | Silver | Coeur Mining | ~$8 | |
| MAG | Silver | MAG Silver | ~$20 | |
| EGO | Gold | Eldorado Gold | ~$21 | |
| KGC | Gold | Kinross Gold | ~$13 | |

**Blacklisted (Mar 9):** NIO, SOFI, EXK -- multi-year backtest proved unprofitable. Added to BLACKLISTED_TICKERS.

**Removed (Mar 12):** GOAU, GOEX -- garbage spreads (80-146% bid-ask).

## Entry Criteria (RSI2 Mean-Reversion)
- ⚠️ RSI(2) drops below **10** (backtested threshold — old threshold of 15 is deprecated)
- ⚠️ No 200-day SMA trend filter needed (strong dollar filter is better — backtested)
- Tight IEX spreads required (MAX_IEX_SPREAD_PCT = 0.5%)
- **Avoid last 10 minutes and first 15 minutes** (David's rule)
- "Don't force it, but don't give up" -- David's words

## Exit Logic (Apr 9 update)
- ⚠️ **Primary**: RSI2 crosses above per-ticker threshold (KGC: >90, others: >70). NOT SMA5.
- **Bonus**: $0.50/share profit target -> close (quick bounce)
- **Safety valve**: Held >= 10 days AND no active CC -> force close
- **CC pivot**: Held >= 5 days -> mark CC-eligible, carry for premium income
- **Friday**: Force-close ALL remaining (weekend risk)
- ⚠️ **-8% stop loss** on stock positions (RSI2 backtest includes stops)
- Old "no stop loss" rule applied to SMA5-exit system and is no longer valid

### HARD RULE: Never Create a Naked Short Call (Mar 12)
**Closing the stock while a short call is open = naked short call. This is an absolute rule -- no exceptions.**

All 4 stock-close paths in `combined_runner.py` check `_has_active_cc(ticker)` before executing:

| Exit trigger | Guard location |
|---|---|
| Intraday 10-day max hold | `_intraday_profit_check()` |
| EOD 10-day max hold | `_eod_profit_sweep()` |
| Intraday SMA5 crossover | `_intraday_profit_check()` |
| EOD SMA5 crossover | `_eod_profit_sweep()` |

When a CC is active: skip the close, log the skip reason, carry the position. The CC's own lifecycle (50% profit target, roll at 7 DTE, expiry) drives exit timing. Once the CC is closed, the next SMA5 or max-hold check will fire normally.

## Covered Call Selling (via options_overlay.py)
- Runs at 3:30 PM ET daily
- Only considers positions held 5+ days (CC_ELIGIBLE_AFTER_DAYS)
- Only on CC_OPTIONS_ELIGIBLE tickers (those with Alpaca options chains)
- Targets ~0.30 delta, 14-28 DTE
- Does NOT sell on top-20% momentum stocks (caps upside)
- DOES sell on sideways/weak holdings (premium income)
- Profit target: close at 50% of premium received
- Roll at 7 DTE remaining
- Black-Scholes model selects strike

## Key Config Files
- `~/trading_bot/combined_config.py` -- TRADING_UNIVERSE=[flat 26-ticker list], CC_ELIGIBLE_SWING=TRADING_UNIVERSE, CC_OPTIONS_ELIGIBLE=[15 tickers with Alpaca options], RSI2_ENTRY_THRESHOLD=**15**, SMA_TREND_PERIOD=200, SMA_EXIT_PERIOD=5, MAX_HOLD_DAYS=10, CC_ELIGIBLE_AFTER_DAYS=5, SWING_STOP_LOSS_PCT=None, SWING_PROFIT_TARGET_PER_SHARE=0.50, SWING_MAX_TOTAL_POSITIONS=5
- `~/trading_bot/swing_runner.py` -- TRADING_TICKERS=CC_ELIGIBLE_SWING (imported from combined_config), RSI2 entry + 5-SMA exit, no stop loss, MIN_LOT_SIZE=100
- `~/trading_bot/slvr_cc_config.py` -- TRADE_TICKERS=[flat 26-ticker list], CC scalper params, regime config, PMCC config
- `~/trading_bot/options_overlay.py` -- CC seller only (CSPs + iron condors disabled). CC_TARGET_DELTA=0.30, CC_MIN_DTE=14, CC_MAX_DTE=28, CC_PROFIT_TARGET=0.50. Uses CC_ELIGIBLE_AFTER_DAYS for seasoning.
- `~/trading_bot/combined_runner.py` -- orchestrator. `_has_active_cc(ticker)` checks open CCs. `_intraday_profit_check()` every 60s ($0.50 target + 5-SMA exit + max-hold skipped if CC active). `_check_sma5_exit()` helper. `_eod_profit_sweep()` at 3:55 PM (price > 5-SMA -> close, held >=10d AND no CC -> force, CC active -> carry, held >=5d -> carry CC-eligible, held <5d -> carry). Options overlay at 3:30 PM. Reconciliation on startup. Heartbeat every 30min + SIGTERM handler (added Mar 11 to diagnose recurring crash).

## API Notes
- **Alpaca paper account** PA3K7O4G7U36, $100k, Options Level 3
- **MUST use `feed=iex`** on all data calls (free tier)
- Credentials: `jarvis_utils.secrets.get('Alpaca', 'api_key_id'/'secret_key', user='a4dc8459-...')`

## David's Premium Scalping Strategy (Mar 9 -- PRIORITY)

**This is the real money strategy.** David makes $500-$3,000/session doing this manually on SLVR.

### The Play
1. **Hold shares** of high-IV silver ETF (SLVR -- Sprott Silver Miners, ~$70)
2. **Sell DEEP OTM, LONG-DATED covered calls** into IV/price spikes
   - Example: Oct 16 $115 calls (64% OTM, 220 DTE) at $5.70/contract
   - Example: Oct 16 $95 calls (35% OTM) at $9.45/contract
3. **Buy them back** when IV crushes or underlying pulls back
4. **Sell again** on next spike. Shares never move.
5. **Near-instant back and forth** -- not holding for days, scalping premium intraday/same session

### The Signal: SILVER SPOT LEADS
- **Silver going UP -> DON'T sell calls.** SLVR will follow higher, calls will get more expensive.
- **Silver going DOWN or stalling -> SELL calls.** Premium about to decay.
- **Silver turning UP again -> BUY BACK calls.** Before SLVR follows and calls get expensive.
- The metal is the signal for BOTH sell and buy-back timing.
- Key question (under research): how much lag between silver spot and SLVR? Even 5 seconds is tradeable.

### Wide Spreads Work FOR You
- Sell ABOVE mid into spike (buyers chasing, hitting the ask -- you sell inside spread above mid)
- Buy back BELOW mid on slam (sellers panicking, hitting the bid -- you buy inside spread below mid)
- You're providing liquidity on both sides, capturing spread as part of edge
- Market maker bots keep spreads wide -- you step in front when order flow is one-directional

### HARD RULE: Max 50 Contracts Per Chain Per Day
- **Absolute rule, no exceptions.** David learned this the hard way.
- Exceed 50 -> broker flags you for unusual activity, compliance review
- On low-liquidity options, 50 contracts = significant portion of daily volume
- Start at 10-20, stay under 20-25% of chain's daily volume
- As bot gets better, may need to LOWER this limit, not raise it

### Getting Stuck
- Sometimes you sell calls and stock keeps ripping -- calls get more expensive
- That's OK if they're deep OTM. You're not in trouble. Just wait for pullback.
- The loss isn't assignment -- it's opportunity cost above the strike
- Worst case: get assigned, sell shares at strike + premium, buy back on next dip

### Bot Code (BUILT Mar 10, MULTI-TICKER REFACTOR Mar 10)
- ~/trading_bot/slvr_cc_scalper.py (2,273 lines) -- CCScalper class (multi-ticker)
- ~/trading_bot/slvr_cc_config.py (358 lines) -- all tunable parameters + regime + PMCC config
- CLI: `python slvr_cc_scalper.py [--live] [--status] [--once] [--regime] [--pmcc-status] [--pmcc-register TICKER SYMBOL COST]`
- Black-Scholes engine, DataLayer (yfinance, per-ticker chains), SignalEngine (6-factor composite, per-ticker), StrikeSelector (per-ticker + spread filter), ExecutionLayer (Alpaca REST), DailyTradeCounter (hard 50 limit), ScalperState (JSON persistence)
- **26 tickers** (one flat TRADE_TICKERS list): SLVR, SGDM, SGDJ, GBUG, COPP, COPJ, SIL, SILJ, GDX, GDXJ, COPX, SLVP, RING, PAAS, BTG, WPM, AYA, AG, HL, FSM, SVM, USAS, CDE, MAG, EGO, KGC
- **2 leveraged tickers** (disabled): NUGT, JNUG (ENABLE_LEVERAGED=False)
- **Signal tickers** (not traded): GLD, UUP, SLV
- **Spread filter**: MAX_BID_ASK_SPREAD_PCT=0.50 -- auto-skips illiquid chains
- **Order manager**: ~/trading_bot/order_manager.py (846 lines) -- active order lifecycle (post -> monitor -> cancel/replace -> fill), sliding window rate limiter (180 req/60s), smart price adjustment ($0.02 steps toward mid, never crosses mid), 3s min between adjustments, max 10 attempts, dry-run simulator
- **PMCC manager**: ~/trading_bot/pmcc_manager.py (1,850 lines) -- DiagonalSpread dataclass, PMCCManager class. Handles diagonal spreads (long LEAP + short OTM call). Auto-detect LEAPs from Alpaca positions or manual registration. Assignment risk management (3-tier: warn at d>0.40, danger at d>0.50, urgent at underlying>strike). Auto-roll-up-and-out. PMCC_MAX_SHORT_DELTA=0.15, extra 10% OTM buffer vs regular CC. Phase 0.5 in run_once().
- **Regime detector**: ~/trading_bot/regime_detector.py -- 7 correlation signals predict RIP/CHOP/CORRECTION/UNCERTAIN. Signals: gold/silver ratio, palladium lead, copper/gold, dollar trend, yield, cross-ticker divergence, vol structure. Auto-adjusts OTM distance and signal thresholds. `--regime` CLI flag for standalone report. REGIME_CACHE_MINUTES=60.
- State: ~/trading_bot/slvr_cc_state.json
- Log: ~/trading_bot/logs/slvr_cc_scalper.log

### PMCC / Diagonal Spreads (BUILT Mar 10)
- **Capital efficiency**: LEAPS cost 65-70% less than 100 shares. ROC multiplier 2.9-3.3x.
- **Tom's SIL position**: Jan '27 $79 LEAP at $21.60 cost -> sell Jul $145 call at $3.90 = 18.1% ROC (4.7x vs shares)
- **SLVR**: No LEAPS (furthest Oct '26). Can run shorter diagonal (buy Oct ITM, sell Jul/Aug OTM) or use shares.
- **SIL/GDXJ/GDX**: Jan '27 and '28 LEAPS available -- best PMCC vehicles.
- **For $10,700**: ONE shares-based CC on SIL OR THREE PMCC spreads (SIL+GDXJ+GDX) at $1,249/cycle, 11.7% ROC.
- Analysis: ~/trading_bot/leaps_vs_shares_analysis.md

### Regime Detection (BUILT Mar 10, BACKTESTED Mar 10)
- David: "use correlations to figure out if ripping or chop coming" / "we were trying to do this with the silver roadmap -- it didn't work super well -- you need to get better at that too"
- Current reading (Mar 10): CORRECTION_LIKELY, 79% confidence
- **BACKTEST RESULTS (197 months, 2009-2025):**
  - Overall accuracy: 31.5% (vs 30.5% naive baseline -- barely better than guessing)
  - UNCERTAIN 86% of time (signals cancel each other out)
  - Actionable accuracy: 35.7% (only 28 directional calls in 16 years)
  - Inflection points: 2/15 correct (13%) -- fails when it matters most
  - Best signals: Palladium lead (#1), vol structure (#2). Weakest: cross-ticker divergence
- **Config updated based on backtest:**
  - Narrowed regime adjustments: RIP->48% OTM/0.75x, CHOP->38%/1.25x, CORR->46%/0.85x (was 55%/0.5x, 35%/1.5x, 55%/0.75x)
  - Reweighted: Pd lead 0.25 (was 0.15), vol structure 0.15 (was 0.10), cross-ticker 0.07 (was 0.15)
  - Detector now gently nudges params rather than swinging aggressively
- Results: ~/trading_bot/backtest_regime_detector_results.md

### Super-Cycle Backtest (Mar 10)
- Bull-choppy (current regime) = WORST for CC selling: 76.7% WR, $165/trade
- Bear = BEST: 97.5% WR, $1,112/trade
- Assignment risk in bull: 35% ITM at expiry (vs 0% in bear)
- 55% OTM cuts assignment risk 35%->26% with similar per-trade P&L
- Results: ~/trading_bot/backtest_supercycle_results.md

### Research (COMPLETE Mar 10)
- ~/trading_bot/research/options_scalping_research.md (1,498 lines, near-term scalp analysis)
- ~/trading_bot/research/deep_otm_cc_strategy.md (676 lines -- correlation analysis, lead/lag, backtest, Greeks, parameters)
- ~/trading_bot/research/correlation_analysis.py (1,604 lines -- full analysis script)
- ~/trading_bot/research/charts/ (10 PNGs -- correlation matrix, rolling correlations, lead/lag, vol, premium decay, Greeks surface, spike cycles, backtest, signals)
- ~/trading_bot/backtest_results_cc.md -- 152 trades, $101K P&L, 86% WR, Sharpe 1.36 (full period 2008-2025)
- ~/trading_bot/backtest_stress_tests.md -- 8 stress tests (regime, crisis, Monte Carlo)
- ~/trading_bot/leaps_vs_shares_analysis.md -- capital efficiency comparison

## Lessons Learned
1. **Stop losses destroy value on mean-reverting stocks.** Our backtest: EOD closes = +$12,540, but stops triggered 4,165 times destroying $12,527. Stocks mean-revert intraday; stops exit at noise bottom.
2. **Ticker selection > strategy design.** Dropping NIO/SOFI improved P&L by $4,738 -- more than any strategy refinement.
3. **RSI2 entry triples win rate** (19% -> 59%) and reduces drawdown 69%.
4. Options strategy for a bot != for a human. Bot sells premium. Human scalps directionally.
5. 100-share lots drive the entire ticker universe -- not the other way around.
6. Size before signal. Pick what you CAN hold 100 of, then filter for entries.
7. The scalp layer's primary value is building 100-share positions at oversold prices for CC income, not standalone scalp profit.
8. The speculative call-buying module (options_scalper.py) was WRONG for the bot. Disabled.
9. **The metal leads the miners.** Silver spot direction tells you when to sell and buy back SLVR calls. Don't sell calls when silver is going up.
10. **Wide spreads are a weapon when you're right, a trap when you're wrong.** Sell into urgency, buy back in panic.
11. **Stay invisible.** Max 50 contracts/chain/day. Stay under 20-25% of daily volume. Broker will flag you otherwise.
12. **No arbitrary price cap on RSI2 universe.** The old "$20 max" rule was for the pre-RSI2 scalping universe. RSI2 mean-reversion works on any price -- it's about statistical oversold bounces, not absolute price. Position sizing handles capital allocation (fewer shares on expensive ETFs, more on cheap stocks).

## What NOT to Do
- Don't use stop losses (mean-reverting stocks recover; stops destroy value)
- Don't buy speculative calls hoping for a pop (that's David's manual play)
- Don't enter positions in non-100-share lots (useless for CCs)
- Don't chase stocks that are already up big today
- Don't ignore spread width -- IEX can show stale/wide quotes
- Don't trade NIO, SOFI, or EXK (blacklisted -- backtest-proven losers)
- **Don't sell calls when silver spot is going UP** (the underlying will follow, you'll get stuck)
- **Don't exceed 50 contracts per chain per day** (HARD RULE -- broker compliance)
- Don't market-order options -- always use limits, work the spread
- Don't confuse "no options chain" with "not tradeable" -- tickers without Alpaca options still trade Layers 1-2 (RSI2 entry + swing hold)
