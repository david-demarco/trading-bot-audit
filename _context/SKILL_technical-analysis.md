---
name: technical-analysis
description: Load this skill before any trading decision, chart analysis, or market evaluation. Contains TA frameworks, decision rules, and lessons from real trades. This is not indicator math — it is trader thinking.
---

# Technical Analysis — Trader's Framework

## Core Principle: Context Before Signal

An indicator value is meaningless without context. Before acting on ANY signal:

1. **What is the TREND?** (Daily timeframe)
   - Price above 50-SMA on daily = uptrend. Only look for longs.
   - Price below 50-SMA on daily = downtrend. Only look for shorts (or stay flat if long-only).
   - Price chopping around 50-SMA = range. Be careful with any directional trade.

2. **What is the STRUCTURE?** (4H timeframe)
   - Where are the recent swing highs and lows?
   - Is price at support or resistance?
   - Has a level been tested multiple times? (More tests = weaker it gets)

3. **What is the ENTRY?** (1H timeframe)
   - Only now look at RSI, Bollinger Bands, etc.
   - The indicator confirms what structure already told you.

**Rule: Never trade an indicator in isolation. The indicator is the trigger, not the thesis.**

## Multi-Timeframe Analysis (MTA)

| Timeframe | Purpose | What to look for |
|-----------|---------|-----------------|
| Daily/Weekly | Trend direction | SMA50, SMA200, higher highs/lows vs lower highs/lows |
| 4-Hour | Market structure | Support/resistance zones, swing points, range boundaries |
| 1-Hour | Entry timing | RSI2 oversold/overbought, BB breakouts, volume surges |

**The higher timeframe always wins.** A 1H buy signal in a daily downtrend is a trap.

## RSI2 Mean Reversion — When It Works vs When It Kills

### Works (high probability):
- Price is in an UPTREND (above daily SMA50)
- RSI2 drops below 15 = temporary dip in a strong trend
- You're buying a pullback that the trend will carry back up

### Fails (catching knives):
- Price is in a DOWNTREND (below daily SMA50)
- RSI2 drops below 15 = breakdown accelerating
- You're buying into selling pressure with no floor
- **This is what happened with AVAX (-6.45%) and DOT (-3.58%) on Apr 7, 2026**

### Lesson from Apr 7:
- AVAX entered at $9.36 with RSI2=5.7, but was 4% BELOW SMA50
- The 1% SMA tolerance I added "for more trades" let it through
- More trades ≠ better trades. The filter exists for a reason.
- **Fix: Zero tolerance. Price must be strictly above SMA50.**

## Stop Losses — Live Price, Not Bar Close

**Critical bug discovered Apr 7:** Stop losses MUST use the live/current price, not the hourly bar close. Hourly candles can show a positive close while the live price has already breached the stop by 3%+.

- AVAX bar close was $9.38 (+0.23% from entry) while live price was $8.80 (-6.45%)
- The stop never fired because it only checked bar closes
- **Rule: Risk management uses real-time data. Signal generation can use bar data.**

## Momentum Breakout (Bollinger Band + Volume)

### Setup:
- Price breaks above upper BB(20,2)
- Volume > 1.5x 20-period average
- This confirms the breakout has conviction

### Exit:
- Price drops below middle BB (20-SMA)
- OR trailing stop 2% below peak
- OR hard stop -3%

### Backtest results (90 days, Apr 7):
- ETH: +$346 (43% WR) — works
- SOL: +$338 (44% WR) — works
- BTC: -$194 (29% WR) — doesn't work on BTC
- **Insight: Momentum works better on alts than BTC in this market**

## What Doesn't Work (Tested Apr 7)

| Strategy | Result | Why it failed |
|----------|--------|--------------|
| Grid trading | All configs negative | Downtrend eats grids alive |
| VWAP mean reversion | All configs negative | Same problem — dip buying in downtrends |
| RSI2 on 15-min | +$0.19-$0.81/trade | Edge too thin after slippage |
| Cross-pair ratio | 50% reversion rate | Coin flip, not tradeable |
| Fear & Greed as filter | 51% accuracy | Useless alone as predictor |

## Market Context Awareness

### Fear & Greed Index
- Currently 11 (extreme fear, 30 days running as of Apr 7)
- Not predictive by itself, but confirms environment
- Extreme fear = oversold bounces happen frequently = RSI2 mean reversion should work
- BUT only in uptrending assets. Fear + downtrend = capitulation, not opportunity.

### Iran War Context (Apr 7)
- Trump deadline: Tue Apr 7, 8 PM ET — bomb bridges/power plants if Hormuz stays closed
- Oil at $112+, gas at $4/gal
- Gold bullish on war premium, but could sell off hard on de-escalation
- **Always check: what is the macro doing to this asset class?**

## Decision Checklist (Before Every Trade)

1. [ ] What time is it? (Check clock, know the day, know what session)
2. [ ] What is the daily trend? (Above/below SMA50)
3. [ ] What is the 4H structure? (Support/resistance nearby?)
4. [ ] What is the 1H signal? (RSI, BB, volume)
5. [ ] Does the live price confirm? (Not just bar close)
6. [ ] What is the macro context? (War, Fed, earnings, sentiment)
7. [ ] What is my stop? (Defined BEFORE entry, using live price)
8. [ ] What is my target? (Risk:reward > 1:1 minimum)

## Platform Limitations

### Alpaca (current)
- Spot only, long only, no leverage on crypto
- Can only profit in uptrends
- 3 tradeable pairs in current market: ETH, SOL, LINK (above SMA50)

### Coinbase Advanced (next step)
- Perpetual-style futures, long AND short, up to 10x leverage
- Funding rate collection possible
- Bot scaffold built, waiting for David's API key
- **This unlocks shorting in downtrends — the other side of the market we're currently blind to**

## 5-Year Walk-Forward Results (BTC/USD 2021-2026)

Ran 100 RSI2 trades across every market regime. Key findings:

| Regime | Trades | Win Rate | Avg P&L |
|--------|--------|----------|---------|
| BULL (>SMA50 & >SMA200) | 34 | 79% | +1.65% |
| RECOVERY (>SMA50, <SMA200) | 15 | 80% | +1.39% |
| BEAR (<SMA50 & <SMA200) | 31 | 32% | -1.10% |
| CORRECTION (<SMA50, >SMA200) | 20 | 25% | -1.60% |

**Unfiltered return: +1.5% over 5 years. BTC buy-and-hold did +21%.**
**Regime-filtered (longs only in BULL/RECOVERY): +6.7% over 5 years. 39 trades, 74% WR.**
**Improvement: 4.5x better by simply not trading in BEAR/CORRECTION.**

### Critical Insight: REGIME DETECTION IS THE REAL EDGE

RSI2 is not the edge — knowing WHEN to use it is. The same strategy is +80% WR in bulls and 25% WR in corrections. The indicator doesn't change. The market context does.

**Rules derived from 5 years of data:**
1. Trade RSI2 LONG aggressively in BULL and RECOVERY regimes
2. Go FLAT in BEAR and CORRECTION — don't fight the trend
3. Shorts barely break even (46% WR, -0.01% avg) — not worth the risk
4. Regime = price vs SMA50 + SMA200. Both above = bull. Both below = bear.
5. Current regime (Apr 2026): BEAR/RECOVERY transition. Be cautious.

### Worst Trades (All Were Regime Mismatches)
- Long in BEAR: -8.44% (Apr 2022 — catching knife in crash)
- Short in CORRECTION: -8.64% (Mar 2025 — shorting a bounce)
- Long in BEAR: -5.66% (Jan 2026 — our current bear market)

Every catastrophic loss was a trade taken against the regime.

## Connors RSI2 — Original System (Larry Connors, 2008)

Source: "Short Term Trading Strategies That Work" by Larry Connors & Cesar Alvarez

### Original Rules (Long-Only, Stocks/ETFs):
1. **Trend filter: Price > 200-day SMA** (not 50-day — I was using 50)
2. **Entry: Cumulative RSI(2) < 5** (not single RSI < 15)
3. **Exit: Price closes above 5-day SMA**
4. **No stop loss** — Connors found fixed stops REDUCED performance because they triggered before the bounce

### What I Was Doing Wrong:
- Using SMA50 instead of SMA200 as trend filter — SMA200 is more reliable for regime
- Using single RSI2 < 15 instead of cumulative RSI2 < 5 — cumulative is stricter, fewer but better trades
- Adding stop losses — Connors showed stops hurt mean reversion. The whole thesis is "it will revert." If you stop out, you miss the reversion.
- BUT: no stops works for stocks. Crypto can gap 20% — need to adapt.

### Connors' Key Insight:
"The annualized return is limited due to infrequent trades — not suitable on its own in a single market. But in a basket of strategies across multiple instruments, it's a powerful component."

**This means**: RSI2 is ONE tool in a toolkit, not a standalone system. I need multiple uncorrelated strategies running simultaneously.

### Backtest Validation:
- Connors reported 88% accuracy on SPY from 1993-2008
- Strategy continues to work post-publication — suggests genuine edge, not overfitting

### RSI2 Across Asset Classes (3yr backtest, Apr 2026):
| Asset | Trades | Win Rate | Avg P&L | Notes |
|-------|--------|----------|---------|-------|
| KGC | 17 | 88% | +3.42% | BEST in universe |
| NEM | 13 | 85% | +1.33% | Strong |
| GLD | 18 | 83% | +0.81% | Reliable |
| SPY | 24 | 83% | +0.85% | Connors' original asset |
| SILJ | 18 | 78% | +1.27% | Good |
| HL | 20 | 75% | +1.28% | Good |
| FNV | 19 | 74% | +0.74% | Decent |
| AG | 16 | 69% | +0.55% | Marginal |
| PAAS | 16 | 69% | +1.09% | Decent |
| COPX | 12 | 67% | -0.16% | Doesn't work |
| WPM | 18 | 61% | +0.88% | Below threshold |
| SLV | 17 | 59% | -0.02% | Doesn't work on ETFs |
| GOLD | 11 | 36% | -3.06% | AVOID — Barrick is anti-RSI2 |
| BTC (crypto) | 39 | 74% | +1.11% | Only in BULL/RECOVERY regime |

**Key insight: RSI2 works best on individual miners (KGC, NEM, HL) and indices (SPY, GLD), NOT on commodity ETFs (SLV, COPX) or Barrick.**
**All underperform buy-and-hold — RSI2's value is as a TIMING LAYER on top of a trend-following position, not a standalone system.**

### The Three-Layer System (David's Approach)
Tested on KGC over 3 years ($4.98 → $31.36):

| Layer | Role | Return | Standalone? |
|-------|------|--------|-------------|
| Buy & Hold | Capture the trend | +530% | Yes — this IS the strategy |
| RSI2 Timing | Add on dips, trim on peaks | +8.7% alpha | No — weak alone |
| Covered Calls | Income while holding | +54% est. premium | No — needs the position |
| **Combined** | | **~592%** | All three together |

**The layers are multiplicative, not additive.** Each one works BECAUSE of the others:
- B&H without RSI2 = full position all the time, no tactical adds
- RSI2 without B&H = only +8.7% in 3 years (terrible)
- CCs without B&H = no position to sell calls against
- Together = trend capture + dip buying + income generation

**This is why crypto RSI2 alone was disappointing (+1.5% over 5yr on BTC).** There's no covered call layer and no strong underlying trend to ride. The edge in crypto must come from a different structure.

## Wyckoff Method — Market Structure Framework

Source: Richard Wyckoff, early 20th century. The institutional trader's roadmap.

### Three Laws:
1. **Supply & Demand** — price + volume together reveal who's in control
2. **Cause & Effect** — accumulation causes uptrends, distribution causes downtrends
3. **Effort vs Result** — volume (effort) should confirm price (result). Divergence = warning.

### Four Market Phases:
1. **Accumulation** — smart money buys after a downtrend. Sideways range with springs/shakeouts.
2. **Markup** — uptrend begins. Higher highs, higher lows.
3. **Distribution** — smart money sells at the top. Sideways range with upthrusts.
4. **Markdown** — downtrend begins. Lower highs, lower lows.

### How to Use This:
- **Before entering any trade, identify which Wyckoff phase we're in**
- Accumulation → look for spring (false breakdown) → buy
- Distribution → look for upthrust (false breakout) → sell/short
- Markup → buy pullbacks (RSI2 works well here)
- Markdown → stay flat or short (RSI2 longs fail here)

### Springs & Upthrusts (The Key Patterns):
- **Spring**: Price dips below support, immediately recovers. Trap for sellers. Smart money buying.
- **Upthrust**: Price breaks above resistance, immediately fails. Trap for buyers. Smart money selling.
- These are the highest-probability entries in the entire Wyckoff framework.

### Current Market (Apr 2026):
BTC appears to be in late Markdown / early Accumulation. Look for: selling climax (sharp drop on huge volume), automatic rally, secondary test. The spring hasn't happened yet.

## Volume Spread Analysis (VSA) — Tom Williams / Wyckoff

Source: Tom Williams, "Master the Markets." Built on Wyckoff's laws.

### Core Idea: Volume + Spread + Close = Intent
- **Wide spread + high volume + close near high** = genuine strength (smart money buying)
- **Wide spread + high volume + close near low** = genuine weakness (smart money selling)
- **Narrow spread + high volume** = ABSORPTION — the key signal. Someone is soaking up supply/demand without moving price.
- **Wide spread + low volume** = fake move, no conviction behind it

### Key Signals:
| Signal | What it looks like | What it means |
|--------|-------------------|---------------|
| Stopping Volume | High-vol down bar, close near top | Smart money absorbing selling — reversal coming |
| No Demand | Narrow up-bar, low volume | Nobody wants to buy here — weakness ahead |
| No Supply | Narrow down-bar, low volume | Sellers exhausted — strength ahead |
| Buying Climax | Ultra-wide up-bar, huge volume, close mid/low | Distribution — smart money dumping to retail |
| Selling Climax | Ultra-wide down-bar, huge volume, close mid/high | Accumulation — smart money buying the panic |

### How to Apply:
1. On EVERY trade entry, check the volume bar pattern — does volume confirm the move?
2. High volume + no price progress = absorption = reversal likely
3. Low volume pullback in an uptrend = healthy, no supply = buy
4. High volume rally with narrow spread = distribution = don't buy

## Support & Resistance — Where to Trade, Not Just When

### How to Find Levels:
1. Use 4H or daily bars, find swing highs/lows (local extremes with 5+ bars on each side)
2. Cluster nearby levels (within 1.5%) — multiple touches = stronger level
3. S/R flip: old resistance becomes new support after breakout (and vice versa)

### Rules:
- **Buy near support, not resistance.** RSI2 oversold at support = strong. RSI2 oversold at resistance = weak.
- **More touches = weaker level.** Each test chips away at support/resistance. The 4th test often breaks.
- **Volume at levels matters.** High volume at support = absorption = holds. Low volume at support = no buyers = breaks.

### Practical S/R Checklist Before Entry:
1. Where is the nearest support below? (This is your stop loss zone)
2. Where is the nearest resistance above? (This is your profit target)
3. Is risk:reward > 1:1? (Distance to target > distance to stop)
4. If buying near resistance — DON'T. Wait for breakout confirmation or pullback to support.

### Integration with RSI2:
- RSI2 < 15 is only meaningful if accompanied by a volume signal
- RSI2 < 15 + stopping volume (high vol, close near high) = STRONG buy
- RSI2 < 15 + wide spread down on high volume closing on lows = MORE SELLING COMING — don't buy!
- This is the layer I was missing. RSI2 tells me "oversold." VSA tells me "is someone buying this dip or not."

## Volume Profile / Market Profile — Where the Money Is

### Key Concepts:
- **Point of Control (POC)**: Price level with most volume traded. The "fair price" — acts as magnet in ranges, S/R in trends.
- **Value Area (VA)**: Price range containing 70% of volume. "Where the market agrees price should be."
- **High Volume Nodes (HVN)**: Price levels with heavy trading — act as support/resistance. Price stalls here.
- **Low Volume Nodes (LVN)**: Price levels with little trading — price moves FAST through these. Gaps in the profile.

### Trading Rules:
1. **Price above POC = bullish bias. Below = bearish.**
2. **Buy at HVN support (heavy volume below = floor). Sell at HVN resistance.**
3. **LVNs are acceleration zones** — once price enters, it moves quickly to the next HVN.
4. **Price opening outside Value Area → if it goes back in = fakeout. If it holds outside = trend continuation.**
5. **In a range: trade edges toward POC (mean reversion). In a trend: trade POC as support/resistance.**

### Stop/Target Placement:
- **Stop behind the heavy volume barrier** (HVN protects you)
- **Target before the next heavy volume barrier** (price will stall there)
- This gives clearer risk:reward than arbitrary percentage stops

### Limitation for Our Bots:
- Alpaca doesn't provide volume profile data directly
- Can approximate by building our own VPVR from OHLCV bars
- CoinGecko/TradingView have better volume profile tools for manual analysis

## Strategy Portfolio (Don't Just Use One)

Per Connors: RSI2 alone isn't enough. Need multiple uncorrelated strategies:

| Strategy | Market | Regime | Status |
|----------|--------|--------|--------|
| RSI2 Mean Reversion | Crypto (ETH) | BULL/RECOVERY only | LIVE on Alpaca |
| Momentum Breakout | Crypto (ETH/SOL) | All (rides trend) | LIVE on Alpaca |
| Paper Futures L/S | Crypto (BTC/ETH/SOL) | All (shorts in bear) | LIVE paper sim |
| Covered Calls | Equities (metals) | All | LIVE on Alpaca |
| Funding Rate Arb | Crypto perps | All (market neutral) | BLOCKED — need Coinbase API |

## Leverage Risk — Lessons from Paper Futures (Apr 7)

### The Math:
- 2% stop loss × 3x leverage = **6.8% capital loss per trade**
- 20% position size × 6.8% = **1.4% of total capital per loss**
- One SOL short stop-out (-$138) erased 10 winning trades

### Rules for Leveraged Trading:
1. **Reduce position size with leverage**: At 3x, use 10% per position max (not 20%)
2. **OR reduce leverage**: 2x instead of 3x keeps losses manageable
3. **Never let leverage multiply a regime mismatch**: BTC/SOL longs at 3x in BEAR = catastrophe
4. **The paper sim needs daily regime filtering, not just hourly SMA50** — it entered BTC long in daily BEAR (32% WR). Should be blocked.

### Win/Loss Asymmetry at 3x:
- Average winning short: +$28 (avg +0.45% × 3x × $2K = ~$28)
- One stop-out: -$138
- Need 5 winners per loser just to break even
- **This ratio is unsustainable without higher win rate or tighter stops**

## Evolving This Skill

This is a living document. After every trade:
- What worked? What failed? Why?
- Was the entry justified by the framework above?
- What would I do differently?
- Update this file with the lesson.

## Crypto-Specific Edge: What Replaces Covered Calls?

Equities have the three-layer system. Crypto doesn't have options markets on Alpaca. What fills the gap?

### Potential Crypto "Income Layers":
1. **Funding rate collection** (Coinbase perps) — short when funding positive, collect 8-hourly payments. ~10-20% annualized. THIS is the crypto equivalent of covered calls.
2. **Staking yield** — not available on Alpaca, but on native chains. 5-15% APY on SOL, ETH.
3. **DeFi yield farming** — liquidity provision on DEXs. Higher risk, higher return.
4. **Basis trade** — buy spot + short perps when futures trade at premium. Market-neutral ~10-30% APY.

### Crypto Three-Layer System (Proposed):
| Layer | Equities | Crypto Equivalent |
|-------|----------|-------------------|
| Trend capture | Buy & Hold miners | Buy & Hold ETH/SOL in BULL regime |
| Dip timing | RSI2 adds | RSI2 adds (regime-filtered) |
| Income | Covered calls | Funding rate collection (need Coinbase perps) |

### Current Limitation:
Without Coinbase perps, we only have Layers 1+2 on crypto. This is why crypto returns are thin.
The moment David opens a Coinbase account, we unlock Layer 3 and the full system.

### What To Do RIGHT NOW (No New Accounts Needed):
- Keep RSI2 bot on ETH (Layer 1+2, RECOVERY regime)
- Keep paper futures sim running to prove Layer 3 viability
- Scan for regime transitions — when BTC/SOL cross above SMA200, re-add them
- Focus the equities bot on the mining universe where ALL three layers are available
