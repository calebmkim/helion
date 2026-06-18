# Stage 1 — liveness-aware chunk decision (NOTEBOOK / source of truth)

Base SHA: `0676dd32` (reduction-seed-heuristic = merged PR #2762). Branch `reduction-3stage-stack`,
worktree `/home/calebkim/helion-new-heuristics/helion-3stage`. GPU: 4×H100 (sm90), L2=50MB.

## Target (Task 1)
PRIMARY: `fused_linear_jsd` (examples/fused_linear_jsd.py, STANDARD track) narrow-V flips
persistent `[None]` → looped, AND the gate stack (D/R/H/A/F) passes. Standard track has NO liveness
term today → spills on heavy bodies.

## Established diagnosis (REPRODUCED on this box, grad-fair baseline, _lab/stage1_liveness/ab_flj.py)
flj persistent is a disaster; looping fixes it (G = fair_tc / seed, loss+grad baseline):
| shape | dtype | persist[None] | best looped (chunk) |
|---|---|---|---|
| (4096,32000) | bf16 | 0.577 | 1.324 (8192) |
| (4096,50257) | bf16 | 0.259 | 1.155 (2048) / 1.017 (8192) |
| (4096,32000) | fp32 | 0.906 | 2.358 (8192) |
| (4096,50257) | fp32 | 0.499 | 1.273 (4096) / 1.224 (8192) |
| (8192,32000) | bf16 | 0.564 | 1.341 (8192) |
chunk 16384 is weaker for wide-V (4096,50257 bf16 → 0.766); chunk 8192 → 1.017.

## The faithful signal: peak simultaneously-live rdim-shaped tiles (MAX_PEAK_LIVE)
`_lab/stage1_liveness/probe_live_tiles.py` — a real liveness sweep over the FX reduction graphs,
per reduction axis, max over graphs. **Shape-independent (graph structure)**, verified identical at
3 flj shapes. Counts: flj=7, rms_norm=3, layer_norm=3, softmax=2, sum=2, long_sum=2,
cross_entropy=2, welford=3, kl_div=6, jsd=12. Cleanly separates flj(7) from the standard-track light
kernels (≤3). Conservative over-count of "peak live" (counts all rdim-shaped values live at the
point; register rematerialization may reduce true pressure — declared conservative, errs to looped).

## The calibration crux (MEASURED, ab_persist_loop.py)
Naive `footprint_factor=peak_live` with the existing cap 245760 WRONGLY flips cross_entropy:
cross_entropy fp32 (8192,50257) persist G=1.092 (beats tc!), loop16384 G=0.802 — a real regression.
The existing cap was implicitly tuned for the curriculum's real liveness (~2). So the persist
budget must be recalibrated when liveness is made explicit.

## Budget window (budget_window.py)
persist iff `peak_live × m × sh × itemsize ≤ PERSIST_BUDGET`.
- keep ALL currently-persistent standard cells → PERSIST_BUDGET ≥ 458752 (binding: cross_entropy
  bf16 (4096,114688)).
- flip ALL flj narrow-V → PERSIST_BUDGET < 896000 (binding: flj(4096,32000)).
- no currently-LOOPED cell flips UP → PERSIST_BUDGET < 491522 (= 2×245760+).
=> safe window **[458752, 491520]**. Choose **PERSIST_BUDGET = 2 × ROW_PERSIST_MAX_BYTES = 491520**
   (top of window, principled = H100 per-program fast memory ≈ registers 256KB + SMEM 228KB ≈ 484KB;
   NOT a curriculum fence — it's a factor of 2). Curriculum stays byte-identical, flj flips.
The looped CHUNK keeps the register budget ROW_PERSIST_MAX_BYTES (hot-loop must stay register-
resident): chunk = min(LOOPED_CHUNK, prev_pow2(ROW_PERSIST_MAX_BYTES // (m×itemsize×peak_live))).
flj (peak=7, itemsize=4): 245760//28 = 8777 → 8192 (its best chunk). cross_entropy stays persistent.

## DESIGN (final)
- **Part A (refactor, zero-diff):** `_reduction_rblock(env, fact, m_block, footprint_factor=1,
  persist_scale=1) -> (r_block, persistent)`. Decide persist + chunk in ONE place; derive persistent
  from the result. At defaults (1,1) byte-identical (verified by config_recorder zero-diff).
- **Part B (liveness):**
  - Walker fact: peak-live per reduction axis, folded into the single collect pass
    (`_collect_memory_op_facts`) — consumer-agnostic per-axis (like `reductions_fed`).
  - Derived: `ReductionFact.body_live_tiles` reads its axis's slice (default 1).
  - Standard track: `footprint_factor = fact.body_live_tiles`, `persist_scale = LIVE_PERSIST_SCALE = 2`.
  - User-tiled track: KEEP ff=1 + `_bandb_r_block_cap` (sanctioned fallback; folding Band-B into
    footprint_factor needs a different byte budget 16KB vs 240KB — logged tradeoff). → byte-identical.
- New constant `LIVE_PERSIST_SCALE = 2` (register+SMEM story). Gate D: byte budget (itemsize a FACTOR,
  not a literal). Gate H: measured-crossover form, faithful key (peak_live × bytes ≤ HW budget).

## STATUS: design done; implementing Part A next.

═══════════════════════════════════════════════════════════════════════════
## POST-HOC #1 — carried-cap moved into `else`  (⚠️ SUPERSEDED — see POST-HOC #2 below)
## (made AFTER Stage 1 was gated/banked — recorded, NOT re-gated)
═══════════════════════════════════════════════════════════════════════════

Context: after Stage 1 was banked AND after the unified-v2 commits (`3c239777` welford m_block,
`96f14953` fold-carried-cap / derive-persist / rename-Band-B) were rebased into the Stage-1 range,
a further hand-edit was folded into the fold-cap commit via `git commit --amend` + `git rebase
--onto`. That commit's SHA therefore changed: **`96f14953` → `111f9ca7`** (message unchanged).

### What changed — helion/_compiler/autotuner_heuristics/triton.py, in `_reduction_rblock`
- **Substantive:** the loop-carried 2-D accumulator cap (`_carried_tile_r_block_cap`; kl_div/jsd)
  moved from being applied UNCONDITIONALLY (after the if/else) to INSIDE the `else` (looped) branch
  only — i.e. it is now SKIPPED when `extent_held` is True.
- Cosmetic: two long lines wrapped (the `num_carried_2d_tiles == 0` trailing-comment line; the
  `_m_block_cap` `return 1<<30` / `return max(1, prev_power_of_2(...))` lines).

### ⚠️ Reviewer caveat — this is a BEHAVIOR change, NOT a no-op, and is UNVERIFIED
- It only bites when `extent_held == True` for a carried-tile reduction (kl_div, jsd). In that case
  the OLD code emitted `R_BLOCK = min(rdim, carried_cap) =` the cap (jsd→2048, kl_div→4096; see
  `_lab/harness/run3_bandb_nro_ab.py`); the NEW code emits `R_BLOCK = rdim` (full width, e.g. 32768).
- `extent_held` needs `m·size_hint·itemsize ≤ ROW_PERSIST_MAX_BYTES (245760)`. With user-tiled
  `m=1`, even V=50257 fp32 (=201028) clears it → extent_held is plausibly True for kl_div/jsd, which
  would change their emitted config.
- **Invariant risk:** kl_div and jsd are 2 of the 9 standard-reduction curriculum kernels the hard
  invariant requires to stay byte-identical (0/739). This change MAY break that. It also reintroduces
  the full-width carried-tile residency the in-code comment (now self-contradictory — it still reads
  "Applied regardless of the persist verdict … a carried reduction is NOT given the persist path")
  warned causes catastrophic ptxas spills.
- **NOT re-verified:** config_recorder before/after was NOT re-run (conda `helion` python not on PATH
  at edit time). TODO before trusting it: re-run config_recorder over the full matrix and confirm
  kl_div/jsd stay byte-identical; if they change, revert this hunk (backup tag
  `backup/pre-fixup-164ee30e`) or rethink, and fix the now-stale comment.

### SHAs are STALE (downstream of the amend)
The amend + `rebase --onto` rewrote `96f14953` and EVERY descendant commit, so the SHAs cited in
`STACK_SUMMARY.md` and the per-stage `REPORT.md`s are now stale. PR-splitting is unaffected (it cuts
on the `[stageN-*]` commit-message prefixes, not SHAs). Post-amend code-commit SHAs:
  stage1 fold-cap   `96f14953` → `111f9ca7`
  stage2 A / B / C  `3c925d10` / `36836aac` / `9c384047`   (gate `09b1fa1d`)
  stage3            `25908e74` (composed) · `8f948735` (gate) · `cde27cb7` (fp32 dtype fix)
  tip (pre-this-note) `b9581c1b`
Refresh STACK_SUMMARY.md + the reports when convenient.

═══════════════════════════════════════════════════════════════════════════
## POST-HOC #2 — MEASURED the #1 regression, then FIXED it (extent_held guard + clamp)
═══════════════════════════════════════════════════════════════════════════

**Measured #1 with config_recorder** (739 cells, `b003d7ee`[before Stage 1] → `111f9ca7`[end of Stage 1]):
67 cells changed; the carried-cap-into-`else` edit (#1) was confirmed to REGRESS kl_div + jsd —
  - kl_div: 26 cells, `block_sizes (4096,1) → (32768,1)` / `(65536,1)`
  - jsd:    26 cells, `block_sizes (2048,1) → (32768,1)` / `(65536,1)`
  - (welford's 15 changed cells are the SEPARATE `3c239777` welford-m_block commit — INTENDED, not #1.)
R_BLOCK was uncapped from the carried-tile cap (jsd 2048 / kl_div 4096) up to the full padded extent —
exactly the full-width carried residency the original comment warned spills. So #1 DID break the
9-kernel byte-identical invariant: **kl_div + jsd = 52 cells.**

**Root cause — why `extent_held` was a no-op for carried tiles in the original.** Recorded facts for
kl_div/jsd: `full_width_output = False`, `itemsize = 4` (the fp32 ACCUMULATOR size — uniform across
bf16/fp32, which is why both dtypes capped identically), `num_carried_2d_tiles = 1 (kl_div) / 2 (jsd)`.
With `full_width_output=False` the FULL_WIDTH clause is vacuous, so `extent_held ≈ size_hint ≤
ROW_PERSIST_MAX_BYTES/itemsize = 61440`; every vocab-N (30522…50304) clears it → `extent_held=True`,
so in the original the ONLY thing keeping kl_div/jsd off the persist path was the UNCONDITIONAL carried
cap. `extent_held` models a ONE-SHOT held row; it does NOT model a `[M_BLOCK,R_BLOCK]` tile carried
across the WHOLE loop (a fundamentally heavier residency) — so it cannot answer "can a carried tile
hold?". (Confirmed a real sub-cap-N corner too: jsd/kl_div `size_hint=1024` → `rdim=1024 < carried_cap`,
so the original actually DOES persist them at R_BLOCK=1024 — "carried never persists" only held once N
pushed rdim above the cap.)

**The fix (folded into the fold-cap commit; now `440e8717`), in `_reduction_rblock`:**
  1. guard `extent_held` with `and fact.num_carried_2d_tiles == 0` — a carried 2-D accumulator can
     never hold the extent, so it always streams (the faithful encoding of #1's implicit behavior).
  2. clamp `r_block = max(1, min(r_block, rdim))` — for the sub-cap-N corner the carried cap can
     exceed rdim; the old held branch returned exactly rdim there, so the clamp matches it.
Byte-identical to the PRE-#1 original across the matrix (large-N carried → cap; sub-cap-N carried →
rdim via the clamp; non-carried untouched). Re-running config_recorder was SKIPPED per the human
(confident it's identical) — the equivalence is ANALYTIC, not re-measured.

**SHAs shifted again** (amend + `rebase --onto`):
  stage1 fold-cap  `96f14953` → `111f9ca7` → **`440e8717`**
  stage2 A / B / C `afac0b81` / `9bc8757c` / `c8429c85`   (gate `223bc3cd`)
  stage3           `a33a7831` (composed) · `0d1f2b74` (fp32 dtype fix) · gate `efd1d688`
Backups: `backup/pre-extentguard-7a5606cd` (pre-fix) · `backup/pre-fixup-164ee30e` (pre-#1).
STACK_SUMMARY.md + the per-stage reports remain stale; refresh when convenient.
