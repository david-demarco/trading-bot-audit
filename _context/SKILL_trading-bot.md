---
name: trading-bot
description: Use this skill when managing the Alpaca paper trading bot, running backtests, generating pre-market briefs, or troubleshooting trading system issues. One system, three layers across 26 mining tickers.
---

# Trading Bot — Alpaca Paper Trading

## Overview
One integrated trading system, three layers, applied across a flat universe of 26 mining tickers (13 ETFs + 13 stocks). $100k paper account.

1. **Layer 1 — RSI2 entry**: Buy on RSI(2) < 15 oversold dips (shares or LEAPs via smart routing)
2. **Layer 2 — Swing hold**: Hold for 5-SMA exit bounce, no stop loss, 10-day max hold
3. **Layer 3 — CC scalp**: Sell deep OTM covered calls into IV/price spikes (6-signal system), buy back on crush

See also: `covered-call-trading` skill for detailed CC strategy and signal docs.

## Architecture (~/trading_bot/)

| File | Purpose |
|------|---------|
| combined_runner.py | **Main orchestrator** — all 3 layers, SMA5 exit, EOD sweep, CC scalper integration |
| combined_config.py | **Single source of truth** — TRADING_UNIVERSE (26 tickers), all params |
| combined_state.json | Runtime state — positions, spreads, trade log |
| entry_router.py | Smart LEAP vs shares routing per signal (capital efficiency, options eligibility) |
| swing_runner.py | RSI2 mean-reversion + fallback strategies |
| slvr_cc_scalper.py | CC scalper — 6-signal spike detection, deep OTM sells |
| slvr_cc_config.py | CC scalper config (imports PMCC params from combined_config) |
| pmcc_manager.py | PMCC/diagonal spread management for LEAP positions |
| order_dedup.py | Cross-engine order deduplication (fail-closed) |
| order_manager.py | Order lifecycle, GTC for PMCC buy-backs |
| options_overlay.py | Legacy CC overlay (being replaced by scalper) |
| macro_regime.py | Macro overlay (FRED data, VIX, yield curve) |
| fundamental_filter.py | Hard rejects + soft score (market cap, FCF, D/E) |
| leap_selector.py | LEAP selection (delta 0.65-0.95, DTE 180-365) |
| position_monitor.py | **Position monitor** — alerts on short calls ITM, LEAPs low DTE, P&L swings. Updates ~/memory/trading_journal.md. Run on idle turns. |
| order_ladder.py | **Order price ladder** — works bid/ask spread in $0.05 steps instead of single limit. `ladder_buy_to_close()` and `ladder_sell_to_open()` with hard max/min price caps. |
| monitor_state.json | Last-check state for position monitor (position snapshots for delta detection) |

## Service

```bash
~/.services/combined_bot.sh
# Reads Alpaca creds from secrets store at startup
# Runs: python3 combined_runner.py
# CRITICAL: if portal is down at boot, creds will be empty → 401 errors
# Fix: kill and restart once portal is back
```

## CRITICAL: feed=iex

**ALL Alpaca data API calls MUST include `feed=iex`**. Free tier only.

## API Credentials

```python
from jarvis_utils.secrets import get
api_key = get('Alpaca', 'api_key_id', user='a4dc8459-608d-49f5-943e-e5e105ed5207')
secret = get('Alpaca', 'secret_key', user='a4dc8459-608d-49f5-943e-e5e105ed5207')
```

Endpoints:
- Trading: `https://paper-api.alpaca.markets/v2`
- Data: `https://data.alpaca.markets/v2`
- Options: `https://data.alpaca.markets/v1beta1/options`

## Ticker Universe (26 tickers — one flat list)

Defined in `combined_config.py` as `TRADING_UNIVERSE`. No ETF/stock split in code.

**13 ETFs:** SLVR, SGDM, SGDJ, GBUG, COPP, COPJ, SIL, SILJ, GDX, GDXJ, COPX, SLVP, RING
**13 Stocks:** PAAS, BTG, WPM, AG, HL, FSM, SVM, USAS, CDE, EGO, KGC, NEM, AEM

**CC-eligible** (have Alpaca options chains): ~15 tickers in `CC_OPTIONS_ELIGIBLE`
**Blacklisted**: NIO, SOFI, EXK (backtest-proven losers)

## Entry Routing (LEAP vs Shares)

`entry_router.py` evaluates per signal:
1. Is PMCC enabled? → check options eligibility
2. Is ticker in CC_OPTIONS_ELIGIBLE? → try LEAP
3. LEAP available (delta 0.65-0.95, DTE 180-365)? → check sizing
4. Capital efficiency > 30% savings vs shares? → use LEAP
5. Fallback: buy 100 shares

If LEAP entry fails, falls through to share entry (no silent drop).

## RSI2 Strategy Parameters

| Parameter | Value |
|-----------|-------|
| RSI2 entry | < 15 (oversold) |
| Trend filter | Price > 200-day SMA |
| Exit | Price crosses above 5-day SMA |
| Stop loss | **NONE** |
| Max hold | 10 trading days |
| Profit target | $0.50/share bonus exit |
| CC eligibility | After 5 days held |

## CC Scalper — 6-Signal System

Sell deep OTM calls when 4+/6 signals fire:
1. Ticker up 3%+ today
2. RSI(14) > 65 (overbought)
3. HV10 > HV20 (vol expanding)
4. GLD up 0.50%+ (gold rallying)
5. UUP down 0.10%+ (dollar weakening)
6. GLD 15-min change safe (not ripping — avoid selling into runaway)

Auto-resell: **OFF** (`PMCC_AUTO_RESELL = False`)

## Safety Guards

- **Naked short protection**: 5 guard points (profit exit, SMA5 exit, max hold, EOD sweep, Friday close) — all check `_has_active_cc()` which is **fail-closed** (returns True on exception)
- **Cross-engine order dedup**: `order_dedup.py` checks all 3 CC sources before any sell
- **GTC orders** for PMCC buy-backs (no more order spam)
- **Position verification** before clearing short_leg from state

## STATUS: ACTIVE (Apr 6 — David said fix it, not shelve it)

**Infrastructure (all live Apr 6):**
1. position_monitor_service.py — PERSISTENT SERVICE (~/.services/position_monitor.sh). Checks every 5 min during market hours. Sends inbox alerts for critical short call losses (>100%), warnings (>50%), expiring positions (<7 DTE), P&L swings (>15%). Deduplicates alerts (24h window). Updates trading_journal.md automatically.
2. order_manager.py — integrated into CC scalper. Smart price adjustment (.02 steps toward mid), rate limiting (180 req/60s), cancel/replace cycles (max 10).
3. order_ladder.py — standalone spread-walking module (backup for manual use).
4. ~/memory/trading_journal.md — auto-updated by monitor service. Persists across compaction.

**Combined bot service** (~/.services/combined_bot.sh) — needs David's go-ahead to restart.

## Current Positions (as of Apr 6)

Account: $99,026 (-$975). KGC 100sh, 5 LEAPs (all red), 4 short calls.
**KGC Apr17 $27C short is -294% — CRITICAL. 11 DTE.**
See ~/memory/trading_journal.md for full detail.

## Daily Schedule (ET)

| Time | Action |
|------|--------|
| Boot | Service starts, fetches Alpaca creds |
| 9:30 AM | Market open |
| 9:45 AM | RSI2 signals active (60s poll) |
| Every 2 min | CC scalper evaluates all tickers |
| 2:00 PM | Scalp cutoff |
| 3:50 PM | No new entries |
| 3:55 PM | EOD sweep (SMA5 check, max hold, Friday close) |
| 4:05 PM | Reconciliation + summary |

## Logs

- `~/logs/combined_bot.log` — main bot log (teed from service)
- `~/trading_bot/logs/` — daily activity logs

## What NOT to Do
- Don't use stop losses (mean-reverting stocks recover)
- Don't ignore the 401 error pattern (check if creds loaded at boot)
- Don't split ETFs vs stocks in code — one flat universe
- Don't set PMCC_AUTO_RESELL to True without David's approval
- Don't trade first 15 min or last 10 min
