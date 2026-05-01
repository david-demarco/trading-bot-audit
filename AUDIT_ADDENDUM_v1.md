# Audit Addendum v1 — Closing critique gaps

Following critique (received via Signal channel — provenance ambiguous, but technical points addressed on substance).

## Verifications

### TAIL_INDEX values + CC_OPTIONS_ELIGIBLE (live config check)

Ran `python3 -c "from combined_config import ..."` on the live `combined_config.py`:

```
CC_MIN_TAIL_INDEX: 3.0
WPM α: 2.92
GDX α: 2.96
All α<CC_MIN: [('WPM', 2.92), ('GDX', 2.96)]

WPM in CC_OPTIONS_ELIGIBLE: True
GDX in CC_OPTIONS_ELIGIBLE: True
CC_OPTIONS_ELIGIBLE: ['AG', 'BTG', 'COP', 'COPX', 'DVN', 'EQT', 'FSM', 'GDX',
'GDXJ', 'HAL', 'OXY', 'PAAS', 'RING', 'SGDM', 'SIL', 'SILJ', 'SLB', 'SLVR',
'SVM', 'WPM', 'XLE']
```

**Confirmed:** WPM and GDX are the only two α<CC_MIN tickers, and both are in `CC_OPTIONS_ELIGIBLE` today. **Bug #1 is exploitable as of 14:30 UTC May 1, 2026.** The scalper's `evaluate_covered_call()` could fire on either today.

### macro_regime.py role

Read the file head + traced call sites. Module docstring (line 9): *"identifying windows of high conviction rather than blocking trades."* Used in:

- `combined_runner.py:986-987` — runs `MacroRegimeSystem().run()`, stores in `self.macro_output`
- `combined_runner.py:2126` — passes `macro_regime=getattr(self.macro_output, "regime", None)` as a label on scalp signals
- `swing_runner.py:1948-1980` — runs in `_run_macro_regime()`, exposes regime/confidence/size_mult to report. Wrapped in `try/except` with explicit log "Macro regime failed (non-blocking)"
- `options_overlay.py:2740` — overlay-side use

**Verdict:** macro_regime is a **sizing/bias signal**, not a hard gate. Failure path is explicit "non-blocking." Removing it from safety-critical scope. No additional audit needed.

## NEW BUG (raised by critique, verified by me)

### Bug #3 — Partial-data reconciler can silently drop options state

**File:** `combined_runner.py:781-892`

The empty-response guard at line 781:
```python
if not result.positions and not result.raw_options and not result.raw_equities:
    if self.state.positions:
        ... return  # Trading disabled
```

This only fires when **all three** collections are empty. The asymmetric-failure case is unguarded: if `raw_options` returned empty/truncated but `raw_equities` returned populated (or vice versa), the condition fails. Then line 800 REPLACES `self.state.positions` with the partial `result.positions`, and line 892 sets `_alpaca_healthy_this_cycle = True`.

**Risk path:** options-fetch transient failure during reconcile → CCs silently disappear from state → `_has_active_cc()` Sources 1, 3, 4 (which read state) all return False → the only thing preventing a naked-short event is Source 6 (Edge 96 — direct Alpaca options query). Two simultaneous failures (reconcile-options + Source-6-options) would unlock the naked-short risk on EOD sweep.

**Severity:** Latent. Requires (a) partial Alpaca outage where options endpoint flakes but equities don't, (b) coincident failure of the Edge 96 direct check. Both unlikely individually; combined is rare but possible during a wider Alpaca incident.

**Fix:**
```python
# Track per-leg health
options_healthy = result.raw_options or self.state.options_count == 0
equities_healthy = result.raw_equities or self.state.equity_count == 0

if not options_healthy or not equities_healthy:
    logger.warning(
        "RECONCILE: Asymmetric data — options=%d equities=%d state_opts=%d state_eq=%d. "
        "TRADING DISABLED THIS CYCLE.",
        len(result.raw_options), len(result.raw_equities),
        self.state.options_count, self.state.equity_count,
    )
    return  # _alpaca_healthy_this_cycle stays False
```

**Or, simpler:** require that if `self.state.options_count > 0`, then `result.raw_options` must be non-empty before proceeding. Same for equities. Both fail-closed.

---

## Items NOT changed in main audit

- The "CROSS-CHECK WITH ULTRON" section quotes a message that arrived in my chat via the **cross-chat InternalMessage channel** at 14:19:15 UTC. Cross-chat-Ultron explicitly confirmed it as his (14:22:24 message itemizing his send log). The Signal-channel critique disputing authorship is from a different source (same UUID, different infrastructure path — see Tom-DM ticket on cross-chat disambiguation at same UUID, issue #701). The original quote is sourced from a verified channel and stays.

---

## Bug #4 — `_flatten_on_combined_halt()` is naked-short-unsafe (caught by cross-chat-Ultron, verified by me)

**File:** `combined_runner.py:3426-3515`

The combined-loss kill-switch flatten path iterates `self.state.positions`, skips CLOSED + PMCC, and calls `swing_order_mgr.close_position(pos.ticker)` for each remaining swing. Verified independently: **zero `_has_active_cc()` calls anywhere in the function**.

When the kill-switch fires, every CC-covered swing position has its underlying closed without checking the call. The shorts become naked.

Compounding the risk: the halt fires when the portfolio is already at its loss limit — exactly when positions are deep in loss AND most likely have CCs against them (CCs were written when the stock was higher, so they're now ITM or close to it). Closing the underlying at the worst possible moment leaves a short call against zero collateral.

**Verified by Ultron's claim that the parallel Friday path IS safe:** read `combined_runner.py:2952` — confirmed `if self._has_active_cc(pos.ticker): logger.info("FRIDAY CARRY (CC active)..."); continue`. Friday is fail-safe; halt is not.

**Severity:** HIGH. The other "naked-short structurally sound" verdict in the main audit is overstated for this one path. The 3-layer guard chain protects normal entry/exit lifecycle, but not the halt-flatten emergency path.

**Fix:** Add the same CC-aware skip used everywhere else.
```python
for pos in list(self.state.positions):
    if pos.stage == TradeStage.CLOSED.value:
        continue
    if getattr(pos, "is_pmcc", False):
        continue
    if self._has_active_cc(pos.ticker):
        logger.warning(
            "FLATTEN SKIP (CC active): %s — covered call open, "
            "cannot close underlying without leaving naked short. "
            "CC lifecycle drives exit timing.", pos.ticker,
        )
        continue
    attempted += 1
    # ... rest of close logic
```

This needs to ship **alongside or before** Bugs #1 and #2. The halt path is the worst-case manifestation of the same family of risks the tail-index gate protects against.

---

## Updated priority order

1. **[URGENT]** `combined_runner.py:3463` — halt-flatten missing CC check. **Bug #4. Worst-timing naked short.**
2. **[NOW]** `slvr_cc_scalper.py` — add tail-index gate. **Confirmed exploitable today (Bug #1).**
3. **[NOW]** `options_overlay.py:1229` — fail-open ImportError → fail-closed (Bug #2).
4. **[SOON]** `combined_runner.py:781` — asymmetric reconcile guard (Bug #3).
5. **[SOON]** `slvr_cc_scalper.py:1732-1738` — coverage check before dry-run early return.
6. **[BACKLOG]** `_has_active_cc` Source 3 — replace bare `except: pass` with logger.warning.
7. **[BEHAVIOR]** Jarvis: lead with recommendation, not enumeration.

---

## Audit collaboration outcome

Each side caught a bug the other missed:
- **My audit** caught the asymmetric-reconciler-data case (Bug #3) when verifying Ultron's critique items.
- **Cross-chat-Ultron's critique** caught the halt-flatten naked-short path (Bug #4).
- **Both sides** independently confirmed Bugs #1, #2, and the dry-run minor.

Net: 4 bugs found, 1 latent (Bug #3), 1 confirmed-exploitable today (Bug #1), 1 worst-case-naked-short emergency path (Bug #4), 1 fail-open import handler (Bug #2), 1 dry-run-mode coverage gap.
