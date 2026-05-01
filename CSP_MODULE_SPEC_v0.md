# Cash-Secured Put Module — Draft Spec v0

**Author:** Jarvis (Metals and AI)
**Date:** 2026-05-01
**Status:** DRAFT — for David/Ultron review before any code.

## Why this exists

`combined_config.TAIL_INDEX` flags tickers where the empirical Pareto α < 3.0
(fat tails, kurtosis high enough that short-call premium underprices crash
risk). The overlay's recommendation for these names is literally **"sell puts
instead"** — i.e. take the same directional view (long volatility / long the
underlying) but on the side where fat tails work for you, not against you.

Patched as of today (commit dd382c7 / e49bed7), the CC scalper and overlay
both **block** call-selling on these names. But there is currently **no
execution path** for the alternative. Every "sell put" recommendation falls
through to manual.

Today's eligible list: `WPM` (α=2.92), `GDX` (α=2.96). List grows as
TAIL_INDEX is updated.

## Scope

A new module `csp_seller.py` (mirror of `slvr_cc_scalper.py`) that:

1. Scans `[t for t in CC_OPTIONS_ELIGIBLE if TAIL_INDEX.get(t, 99) < CC_MIN_TAIL_INDEX]` each cycle.
2. Evaluates a 6-signal SELL-PUT score (inverse of the scalper's call score).
3. Selects strikes per the put-strike policy (below).
4. Submits via existing `alpaca_client.submit_option_order()` with
   `side="sell"` `option_type="put"`.
5. Tracks open positions in `csp_state.json` (mirror of scalper state).
6. Writes through `order_dedup.py` so PMCC / CC scalper / CSP can't collide.
7. Closes on profit-target / stop / DTE-trigger like the scalper.

## Signal logic (inverse of CC scalper)

The scalper sells calls when the macro is rallying (premium peaks at local
top). CSP sells puts when the macro is **selling off** (premium peaks at
local bottom — IV crush after = profit).

| # | Signal | Condition |
|---|--------|-----------|
| 1 | `ticker_down_3pct` | underlying down ≥ 3% on the day |
| 2 | `rsi_oversold` | RSI(14) < 35 |
| 3 | `vol_expanding` | HV10 > HV20 |
| 4 | `macro_selling` | GLD/SLV down ≥ 0.5% on the day |
| 5 | `uup_rallying` | UUP up ≥ 0.10% (dollar bounce often precedes silver/gold reversal) |
| 6 | `macro_not_crashing` | GLD 15-min change > -0.10% (don't sell into a free-fall) |

Trigger threshold: 4-of-6, same as scalper. Tunable.

## Strike selection

- 30–45 DTE
- ~25-delta (~12% OTM for WPM, ~10% for GDX)
- Bid ≥ 1.5% of strike (premium floor — same logic as scalper's premium
  filter, just on the put side)
- Open interest ≥ 100 (liquidity floor)

## Position sizing

- Cash-secured: contracts × strike × 100 ≤ available cash
- Max position size: same per-ticker dollar cap as the scalper
- Daily-trade-counter check (mirror of `DailyTradeCounter`)

## Exit logic

- Profit target: 50% of premium collected (close to lock in)
- Stop loss: option price 2× entry (i.e. -100% drawdown on premium)
- DTE trigger: close at 7 DTE regardless (avoid gamma)
- Assignment-on-expiration is acceptable (we WANT to own these names — see
  the silver thesis)

## Integration points (where the work actually lives)

1. **`combined_runner.py`** — add `CSPSeller` instance alongside `CCScalper`,
   add cycle hook in main loop. ~30 LOC.
2. **`order_dedup.py`** — extend `has_pending_or_active_sell()` to include
   put-side. Need new key namespace (`CSP:{ticker}` vs `CC:{ticker}`).
   ~20 LOC.
3. **`combined_state.py`** — add `csp_state` dict to combined state.
   ~10 LOC.
4. **`csp_seller.py`** (new file) — the bulk of the work. Mirror structure
   of `slvr_cc_scalper.py` (3071 LOC), but the put side is simpler because:
   - No "is short put covered" check (cash-secured, not stock-secured)
   - No PMCC interaction (PMCC is calls-only)
   - No tail-index gate needed (this IS the tail-index alternative)
   Estimate: ~1500 LOC for the seller, vs 3071 for the scalper.
5. **`csp_state.json`** — new persistence file.
6. **Position monitor** — extend to alert on CSP positions (currently
   monitors calls only). Modest change to `position_monitor.py`.
7. **EOD sweep** — CSPs do NOT block stock close (we don't own stock for
   CSPs unless assigned). EOD logic doesn't need new guards, just needs to
   not panic when it sees short puts in state.
8. **Reconciler** — Alpaca options query already returns puts; need to
   classify and route to `csp_state` instead of `pmcc_state`.

## Tests (must ship with the module)

- Unit: signal logic on synthetic price/IV traces
- Unit: strike selection on canned chain data
- Unit: dedup integration (CC + CSP same ticker → second blocked)
- Integration: dry-run cycle on WPM / GDX with mocked Alpaca
- Backtest: at least one full year on WPM / GDX with the signal set, so we
  see win-rate and expectancy before going live

## What I don't know yet (open questions)

1. **Premium target on the SHORT-PUT side** — for CCs the floor is 1.5% of
   the strike per cycle (~30 DTE). For CSPs at higher fat-tail IV, floor
   should probably be HIGHER (2–2.5%) to compensate for the pinned tail.
   Need backtest to calibrate.
2. **Concurrent-position cap** — how many CSPs across the fat-tail list?
   Today only 2 names eligible; capital concentration risk if both fire
   together.
3. **Interaction with manual stock long** — if you own WPM stock outright
   AND have a CSP open, assignment doubles position. Need explicit policy:
   skip CSP if existing stock position > X size? Or accept the doubling
   and use it as cost-basis improvement?
4. **Volatility filter** — should we require minimum IV percentile? CSPs
   into compressed IV are less attractive than at IV peaks.

## Time estimate

I withdrew the "~a day" claim. Honest scope:

- New `csp_seller.py` (1500 LOC + tests): **2-3 days** for first draft
- Integration into combined_runner / dedup / state: **1 day**
- Backtest harness for CSPs (mirror of scalper backtester): **1 day**
- Calibration runs + tuning: **1 day**
- Code review + smoke test cycle: **0.5 day**

**Total realistic: ~1 week of focused work** before paper-trading, plus
2-week paper run before live.

## Decision needed from David

- Greenlight to build, or punt as manual-trade-only?
- Premium-floor and concurrent-cap policy (open questions 1 + 2)
- Stock+CSP interaction policy (open question 3)
