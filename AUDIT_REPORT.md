# Trading Bot Audit — May 1 2026

**Auditor:** Jarvis (Metals and AI group)
**Scope:** combined_runner, swing_runner, slvr_cc_scalper, options_overlay, pmcc_manager, position_monitor, entry_router, order_manager, order_dedup, alpaca_client, leap_selector, macro_regime, combined_config (~26K LOC across 23 files)
**Method:** Read each file's safety-critical paths, traced the entry → exit → reconcile → EOD lifecycle, verified specific gates against the source. Cross-checked findings against an independent read by Ultron-the-Jarvis (chat:37bdbafc). Where we agree we agree; where I diverge it's flagged.

---

## TL;DR

Bot is **structurally sound**, with 2 real bugs in the tail-index gate enforcement layer (one of which Ultron found and I confirmed independently). Naked-short risk is well-defended (3 separate guard layers, all fail-closed). EOD sweep correctly preserves CC-covered stock. Position reconciler with Alpaca-as-source-of-truth disables trading on failure. Entry router never silently drops signals.

The communication-layer anti-pattern ("3 options instead of 1 recommendation") is real but it's not in the code — it's in how Jarvis formats alerts to David. That's a behavior fix, not a code fix.

---

## CONFIRMED REAL BUGS

### Bug #1 — CC scalper bypasses tail-index gate entirely
**File:** `slvr_cc_scalper.py`
**Verified:** `grep -E 'TAIL_INDEX|CC_MIN_TAIL_INDEX|tail_index' slvr_cc_scalper.py` → 0 matches.

**Impact:** WPM (α=2.92) and GDX (α=2.96) are flagged in `combined_config.TAIL_INDEX` as "do NOT sell calls — fattest tails." Both are in `CC_OPTIONS_ELIGIBLE`. The scalper has no awareness of this dict and will happily sell calls on them when its 6-signal system fires.

**Fix:** Mirror the `options_overlay.py:1217-1228` check inside `CCScalper.evaluate_covered_call()` (or wherever the scalper makes its call/no-call decision). Pseudocode:
```python
from combined_config import TAIL_INDEX, CC_MIN_TAIL_INDEX
alpha = TAIL_INDEX.get(ticker)
if alpha is not None and alpha < CC_MIN_TAIL_INDEX:
    logger.info("CC SKIP: %s α=%.2f < %.2f (tail risk)", ticker, alpha, CC_MIN_TAIL_INDEX)
    return None
```

**Severity:** Real money risk. The whole point of the tail-index research (Apr 11, Taleb/Spitznagel) was that fat-tailed names are precisely the ones where short-call premium is mispriced relative to crash risk. The overlay enforces it; the scalper doesn't.

---

### Bug #2 — `options_overlay.py` tail-index gate is fail-OPEN on import error
**File:** `options_overlay.py:1217-1230`
**Verified:** Read the lines directly.

```python
try:
    from combined_config import TAIL_INDEX, CC_MIN_TAIL_INDEX, CC_PREFERRED
    alpha = TAIL_INDEX.get(symbol)
    if alpha is not None and alpha < CC_MIN_TAIL_INDEX:
        lines.append(f"  {symbol}: SKIP CC - tail index α={alpha:.2f} < {CC_MIN_TAIL_INDEX} (fat tails, sell puts instead)")
        continue
    if symbol in CC_PREFERRED:
        lines.append(f"  {symbol}: ✓ CC-preferred (calls overpriced vs power law)")
except ImportError:
    pass  # Config not updated yet, proceed without filter
```

**Impact:** If `combined_config` ever fails to import the three names (rename, refactor, partial deploy, dependency cycle), the gate disappears silently. The overlay will recommend CCs on WPM/GDX as if the rule never existed.

**Fix:** Change `pass` to `continue` (skip this symbol) and log a WARNING. Fail-closed is the right posture for naked-short-adjacent decisions.

```python
except ImportError as e:
    logger.warning("TAIL_INDEX gate import failed for %s: %s — skipping CC (fail-closed)", symbol, e)
    continue
```

**Severity:** Latent. Will only fire if the config interface changes. But the pattern matters: every safety gate in this codebase should default to "no trade" on failure, and this one defaults to "trade anyway."

---

## SOLID DESIGN — VERIFIED

### Naked-short guard chain (3 independent layers)

**Layer 1 — `slvr_cc_scalper.py:1716-1747` (`sell_call`)**
- Calls `_is_short_call_covered()` BEFORE every live submit (Edge 129 guard)
- `_is_short_call_covered()` is fail-closed: every `except` branch returns `False`, default fallthrough returns `False`. Coverage requires explicit positive match against equity holdings or PMCC long leg.

**Layer 2 — `pmcc_manager.py:1142-1189`**
- Independent verification: queries Alpaca `/v2/positions` directly to confirm covering LEAP exists
- Fail-closed: line 1183-1189 returns None on ANY exception ("refusing to sell short call (fail-closed)")
- This is the post-PAAS-Mar-27 fix. Direct API call rather than trusting state.

**Layer 3 — `combined_runner.py:_has_active_cc()` (line 2532, fail-closed at 2539)**
- Used by EOD sweep + intraday checks to prevent closing the underlying when a CC is open
- Checks SIX sources: legacy overlay, CC scalper open positions, PMCC active short legs, CC scalper pending sell orders, unified dedup layer, AND live Alpaca options positions (Edge 96 — authoritative)
- Each source wrapped in `try/except: pass` (best-effort) but the function itself fails CLOSED — on outer exception assumes CC IS active

**Verdict:** Three independent paths must all fail in concert to leak a naked short. The `pmcc_manager.py` layer is the strongest because it bypasses internal state and queries Alpaca directly. ✓

### EOD sweep — CC-aware (`combined_runner.py:2693-2882`)

- Closes scalp positions first (always, regardless of CC status — scalps don't write CCs against themselves)
- For swing positions: BOTH SMA5 crossover exit (line 2744) AND max-hold exit (line 2799) explicitly skip when `_has_active_cc(ticker)` returns True
- Logs "carrying overnight to avoid naked short call" when skipping
- Records `eod_sma5_crossover` / `eod_max_hold` exit reasons for audit trail

**Verdict:** The 10-day max-hold rule does NOT trump the naked-short rule. ✓

### Position reconciler with trading-disable on failure (`combined_runner.py:574-892`)

Two paths:
- `_reconcile_positions_with_alpaca()` (line 574) — legacy equity-only reconcile that purges ghost positions
- `_run_alpaca_reconciliation()` (line 738) — full Alpaca-as-source-of-truth (options + equity), runs every `RECONCILIATION_INTERVAL_SECONDS`

Trading-allowed gate (`_trading_allowed`, line 894) checks the reconciler health flag. Edge 107 PR3-v2 fix (line 890): the healthy flag is reset every cycle and only set True after a reconcile that succeeded AND returned non-empty broker data. This prevents trading against stale state when Alpaca returns empty (transient outage looks like "0 positions").

**Verdict:** Reconcile failure → no new orders. ✓

### Entry router never silently drops signals (`entry_router.py`)

Read the entire file (210 lines). Six gates:
1. PMCC enabled globally?
2. Ticker in `CC_OPTIONS_ELIGIBLE`?
3. Spread count limit?
4. LEAP available?
5. Sizing OK?
6. Capital efficiency >30%?

Every failed gate falls back to `RoutingDecision(use_leap=False, ...)` — i.e., buy 100 shares. Signals are NEVER returned as `None` or skipped. Even on exceptions (line 117, 137) the fallback is shares.

**Verdict:** Binary, deterministic, fail-safe. The original docstring claim ("Signals are NEVER silently dropped") matches the code. ✓

### Order dedup + ladder

`order_dedup.py` (398 LOC) and `order_ladder.py` provide cross-engine dedup so the scalper, swing, PMCC, and overlay engines don't double-submit on the same ticker. The `_has_active_cc()` chain also reads from `_order_dedup.has_pending_or_active_sell()` as Source 5.

---

## MINOR (NOT CATASTROPHIC)

### Dry-run path skips coverage check (`slvr_cc_scalper.py:1732-1738`)
```python
if self.dry_run:
    order_id = f"DRY_SELL_{contract_symbol}_{int(time.time())}"
    logger.info("[DRY RUN] SELL %d x %s @ $%.2f ...", ...)
    return order_id
# Edge 129 guard runs HERE — after dry-run early return
if not self._is_short_call_covered(contract_symbol, contracts):
    ...
```

Paper trading won't surface coverage bugs that live trading would catch. If a coverage logic regression slipped in, dry-run wouldn't flag it. Low impact (paper mode is for testing, not safety) but worth knowing when reading dry-run logs.

**Fix:** Move the coverage check BEFORE the dry-run early return. Cost is trivial (one Alpaca positions fetch per dry-run sell) and you get real coverage validation on the paper side.

### `pmcc_manager` PMCC check uses bare `except: pass` (line 2575-2581 in _has_active_cc)
Source 3 of `_has_active_cc` swallows `pmcc_manager.get_active_spreads()` exceptions silently. Comment says "PMCC check is best-effort; Sources 1 & 2 are primary." Acceptable given the layered design, but worth a logger.warning instead of silent pass.

---

## COMMUNICATION-LAYER ANTI-PATTERN (NOT A CODE BUG)

David's "3 options" complaint (and my repeated surface-options reflex on the HL alert) is real but it's not in the trading bot code — it's in how Jarvis formats alerts and discusses positions.

The `--scalp-only` / `--swing-only` CLI flags (combined_runner.py main()) are labeled legacy, combined mode is default. The code structure is fine.

The anti-pattern is that when David asks "what should I do about position X" or position-monitor fires, my default is to enumerate scenarios ("could exit, could roll, could hold") rather than commit to ONE recommendation with rationale. That's a Jarvis behavior fix, not a bot fix.

**Self-fix (going forward):** When a position alert fires or David asks for guidance, lead with the recommendation in the first sentence. Then provide the rationale. If I'm uncertain, say "Recommend X, low confidence because Y" — never "here are 3 options."

---

## RECOMMENDED FIXES (PRIORITY ORDER)

1. **[NOW]** `slvr_cc_scalper.py` — add tail-index gate to scalper's CC eligibility check. Bug #1 above. WPM and GDX are actively in scope and the scalper can fire on them today.
2. **[NOW]** `options_overlay.py:1229` — change `pass` to `continue` and log warning. Bug #2 above.
3. **[SOON]** `slvr_cc_scalper.py:1732-1738` — move coverage check before dry-run early return. Minor above.
4. **[BACKLOG]** `combined_runner.py:_has_active_cc` Source 3 — change bare `except: pass` to `except: logger.warning(...)`. Minor above.
5. **[BEHAVIOR]** Jarvis: stop enumerating options. Lead with recommendation.

---

## CROSS-CHECK WITH ULTRON

Ultron's read (chat:37bdbafc), arrived 14:19:15 UTC:
> Confirmed real bugs (2):
> 1. slvr_cc_scalper.py — zero tail index awareness. WPM (α=2.92) and GDX (α=2.96) ... Fix: mirror the options_overlay.py check.
> 2. options_overlay.py line 1229 — `except ImportError: pass` is fail-open. Fix: change to `continue`.
>
> Looks solid:
> - Naked-short guard in sell_call() is fail-closed, runs before every live submit ✓
> - EOD sweep correctly skips stock close on both SMA5 trigger and max-hold when CC is active ✓
> - entry_router: binary LEAP vs shares, signals never silently dropped ✓
>
> Minor: sell_call() dry-run path returns before the coverage check (lines 1735-1738).

We agree on every finding. Independent reads converged. He gets credit for catching the dry-run bypass; I missed that on first pass and confirmed after reading lines 1732-1747.
