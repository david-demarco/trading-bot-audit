---
name: rsi2-trading
description: Use this skill when managing the RSI mining stock trading system — scanning for signals, executing paper trades, checking exits, or reviewing the trading knowledge base.
---

# RSI Mining Stock Trading System

## Overview
Four-layer wheel strategy on 24 mining tickers using per-ticker asymmetric RSI with macro filters.
63 lessons documented through rigorous backtesting (Apr 8-10, 2026).

## The Four Layers
1. **Layer 1 — Put Sell**: RSI oversold → sell 5% OTM cash-secured put (~30 day). Collect premium.
2. **Layer 2 — Stock Hold**: If assigned (or options unavailable) → hold 100 shares, -8% stop loss.
3. **Layer 3 — Covered Call**: RSI5>70+RSI14>60 → sell 30% OTM covered call, 60-70 DTE.
4. **Layer 4 — Reset**: If called away → back to cash, wait for next entry signal.

## ASYMMETRIC RSI (Apr 10 breakthrough)
Each ticker uses a FAST RSI for entry and SLOW RSI for exit. The slow exit lets winners run.

| Ticker | Entry RSI | Entry < | Exit RSI | Exit > | Method | Sharpe |
|--------|-----------|---------|----------|--------|--------|--------|
| GDX | RSI(7) | 25 | RSI(14) | 65 | SMA | 5.9 |
| WPM | RSI(3) | 10 | RSI(7) | 70 | EWM | 4.9 |
| KGC | RSI(7) | 25 | RSI(14) | 65 | EWM | 7.6 |
| PAAS | RSI(2) | 10 | RSI(2) | 70 | EWM | 7.2 |
| EGO | RSI(7) | 25 | RSI(14) | 65 | EWM | 5.6 |
| RING | RSI(7) | 25 | RSI(14) | 65 | EWM | — |
| AG | RSI(2) | 10 | RSI(7) | 70 | EWM | — |

**DO NOT use RSI2 for all tickers.** Each ticker has its own optimal period.
**DO NOT use SMA5 for exits.** RSI exits beat SMA5 by 2-3x.

## Macro Filters
| Filter | Entry | CC Sell |
|--------|-------|--------|
| Dollar (UUP>200SMA) | Required (strong$, stable ROC) | Not required |
| G/S ratio (vs median) | G/S high = buy | G/S low = sell calls |
| VIX 15-20 (goldilocks) | Near-perfect entries | >25 for put/call selling |
| Cu/Au ratio (COPX/GLD>50SMA) | STRONGEST filter — KGC 91% WR, Sharpe 10.6 | — |
| TIP > 50SMA (real yields falling) | KGC Sharpe 4.2→6.5 | — |
| BB narrow (low volatility) | WPM Sharpe 5.0, KGC 4.5 | — |
| Near FOMC (±3 days) | GDX Sharpe 2.5→4.4 | — |
| RSI14 < 40 | Deep capitulation confirmation | — |

## Composite Score (10 factors)
Each factor adds +1. Higher score = higher confidence:
1. Per-ticker RSI oversold
2. Dollar strong (UUP>200SMA)
3. Dollar stable (ROC ±0.5%)
4. G/S high
5. BB narrow
6. Volume spike >1.5x
7. VIX 15-20
8. Near FOMC
9. TIP > 50SMA
10. Cu/Au > 50SMA

**Score 4 = minimum trade. Score 5 = good. Score 6+ = max position (90% WR, -10% max DD, Sharpe 5.4)**

## Signal Grading
- **A++**: Per-ticker RSI oversold + strong$ + G/S high + RSI14<40 + VIX<25
- **A+**: Per-ticker RSI oversold + strong$ + G/S high + VIX<25
- **A**: Per-ticker RSI oversold + strong$ + VIX<25
- **A(put)**: Per-ticker RSI oversold + strong$ + VIX>25 (sell put only)
- **B**: Per-ticker RSI oversold only
- **C**: Wrong regime — don't trade

## CC Sell Signal
Per-ticker RSI5 + RSI14 thresholds:
- WPM/KGC: RSI5>80 + RSI14>70
- All others: RSI5>70 + RSI14>60
- OTM: 30-35% (backtested optimal)
- DTE: 60-70d (30d has no real bids at 30% OTM)
- Close target: 25% of premium (GTC buy-to-close)
- Max 10-20 contracts/chain/day

## Tools
- **Scanner**: `python3 ~/trading_bot/rsi2_scanner.py` — per-ticker RSI, live intraday prices
- **CC Scanner**: `python3 ~/trading_bot/cc_scanner.py` — multi-RSI CC sell signals
- **Executor**: `python3 ~/trading_bot/rsi2_executor.py` — places orders on A+/A/A(put)
- **Exit checker**: `python3 ~/trading_bot/rsi2_executor.py --exits-only` — per-ticker RSI exits
- **Dashboard**: `python3 ~/trading_bot/daily_dashboard.py` — combined entry + CC view
- **Knowledge**: `~/trading_bot/knowledge/` — ticker_configs.json, lessons.json, trade_log.json

## CC Liquidity (real bids verified)
A-tier: WPM, GDX, SIL, AG (real bids at 35d+)
B-tier: EGO, GDXJ, COPX, KGC, HL, CDE (need 60-90d)
D-tier: PAAS (worst CC ticker — exclude from CC selling)

## Key Findings
- Asymmetric RSI doubles Sharpe on KGC/EGO (lesson #59)
- RSI period varies by ticker: KGC=7, PAAS=2, WPM=3, GDX=7 (lesson #57)
- SMA RSI beats EWM on GDX (Sharpe 3.4 vs 2.1) (lesson #58)
- G/S ratio is regime switch: high=buy, low=sell calls (lesson #45)
- Put selling: 79% WR, premium provides its own buffer (lesson #46)
- Kelly criterion confirms 15% position sizing (lesson #50)
- Out-of-sample: edge holds at 50-60% of backtest (lesson #48)
- 30-35% OTM optimal for CC, verified against real chain liquidity (lesson #41)
- Ticker correlations: GDX/RING/GDXJ 97-99% redundant, COPX best diversifier (lesson #52)

## HARD RULES
1. **Use per-ticker RSI periods** — not RSI2 for everything
2. **Asymmetric entry/exit** — fast RSI in, slow RSI out
3. **Never create a naked short call** — check for open CCs before selling stock
4. **100-share lots only** — for CC eligibility
5. **Max 10-20 contracts per chain per day**
6. **Verify RSI data is live** during market hours (scanner appends intraday price)

## Daily Routine
1. Pre-market: run v3 scanner (`python3 ~/trading_bot/rsi2_scanner_v3.py`)
2. If score ≥ 2: evaluate per vol regime (LEAP vs put sell)
3. During market: `python3 ~/trading_bot/rsi2_executor.py --exits-only`
4. Post-market: log results to knowledge base

## V3 SYSTEM UPDATE (Apr 12, 2026)

**Major findings from full data audit (191 trades, 11 tickers, real Alpaca options data):**

### What changed:
- **MSA regime indicator REMOVED** — 36mo lookback too slow, missed entire rally
- **Covered calls REMOVED** — net negative on volatile miners (5 CCs losing $677)
- **Deep ITM LEAPs REMOVED** — 0.90Δ gives only 1.5x leverage (worse than stock)
- **Tail hedging REMOVED** — 24-30% annual drag, negative EV on miners
- **Entry**: RSI2<10 (or per-ticker optimal) — unchanged
- **Exit**: RSI5>80 (was RSI2>70) — 5.3x improvement in avg return per trade
- **LEAP delta**: 0.60-0.70Δ optimal (3.2x theoretical, ~2x actual leverage)

### V3 Composite Scoring (replaces old 10-factor score):
| Factor | Points |
|--------|--------|
| VIX < 25 | +2 (strongest filter, +7.6% edge) |
| GDX RSI14 > 50 | +2 |
| GDX RSI14 < 40 | -3 KILL SWITCH (22% WR, avg -6.6%) |
| VIX ≥ 25 | -2 |
| Stock RSI14 > 50 | +1 |
| Stock RSI14 < 40 | -1 |
| Above 50 SMA | +1 |

**Score ≥ 4** → full position. **Score 2-3** → reduced. **Score < 2** → SKIP.
Negative scores are 100% losers in backtest.

### Vol-Adaptive Strategy:
- Vol expanding (RV20/RV60 > 1.2) → Buy 0.65Δ LEAP (options cheap)
- Vol normal (0.8-1.2) → Buy 0.65Δ LEAP at reduced size
- Vol contracting (< 0.8) → Sell ATM/ITM put (collect rich premium)

### V3 Scanner: `python3 ~/trading_bot/rsi2_scanner_v3.py`

### V3 Backtest Results ($95K, 5% sizing, 18 months):
- Return: +82.6% (vs +25.7% unfiltered)
- Win rate: 83% (vs 71%)
- Sharpe: 3.4→4.9
- March 2026 crash: scoring blocked ALL losing entries

### Taleb/Power-Law Finding:
- GDX tail index α=2.74 (matches market pricing — no mispricing)
- Miners have thinner tails than S&P → tail hedging even worse than on equities
- OTM puts on miners are 50-200x OVERPRICED vs power-law fair value
- Structural edge in SELLING puts on miners, not buying them

### Full framework: ~/trading_bot/knowledge/SYSTEM_V3_FRAMEWORK.md
