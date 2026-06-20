# Stage 2 — unify M-reduction into standard/user-tiled track (NOTEBOOK)

Base = Stage-1 tip `26427709` on `reduction-3stage-stack`. Goal: the 6 backward M-reduction
kernels (rms_norm_bwd, layer_norm_bwd, instance_norm, group_norm, bias_grad, dyt) each beat their
catastrophic default via the standard/user-tiled track, so m_reduction never needs to exist on main.
HARD INVARIANT (prove after EVERY edit): 9 standard reduction kernels byte-identical (config_recorder
739 cells) AND 8 Task-1 transfer kernels non-regressed.

## Landscape (LANDSCAPE.md) — confirmed
| kernel | facts today | track | mat-inner | seed/default | sub-problem |
|---|---|---|---|---|---|
| rms_norm_bwd | 0 | none | 1 | DEFAULT [32,32] | A (port) |
| layer_norm_bwd | 0 | none | 1 | DEFAULT [32,32] | A (port) |
| instance_norm | 0 | none | 1 | DEFAULT [32,32] | A (port) |
| group_norm | 0 | none | 2 | DEFAULT [32,32] | C (hill-climb) |
| bias_grad | 1 | user-tiled T2 | 0 | T2 seed [1,8192] BAD | B (hill-climb) |
| dyt | 1 | user-tiled T2 | 0 | T2 seed [1,8192] BAD | B (hill-climb) |

## Baseline perf (mr_bench seedtc fp32 train; G = tc/seed, >1 beats tc)
- bias_grad: current T2 seed [1,8192] G≈0.34-0.64 (loses to tc; worse than its default ~1.05). The
  M_BLOCK is floored to 1 → M-way finalize over M partials.
- dyt: current T2 seed [1,8192] G≈0.43-1.10 (mostly 0.66; loses). Same floored-M problem.
- rms/ln/instance/group_norm: no seed → generic default [32,32] (catastrophic, task table: 0.03-0.22).
TODO: measure DEFAULT arm explicitly (the bar) for all 6 + oracle for B/C answer keys.

## DESIGN — sub-problem A (PORT, mechanical; ws2 recognizer ported to merged classes)
Port the materialized-inner recognizer (ws2 `register_unrolled_reduction` materialized branch):
1. **device_ir.py `register_user_tiled_reductions`**: ADD a materialized-inner branch — if exactly
   ONE inner ReductionLowering axis is in NEITHER block_sizes NOR reduction_loops (materialized
   full-width feature/spatial reduction the roller declined), build a standard ReductionFact for it
   (non_reduction_loop_block_ids=()). Else keep the existing T2 branch (single inner in block_sizes).
2. **triton.py `_is_standard_reduction`**: change `fact.block_id in reduction_loops` →
   `fact.block_id NOT in block_sizes` — routes a materialized rdim (in neither) to STANDARD, while
   rolled (in rl, not in bs) and user-tiled (in bs) are unchanged ⇒ byte-identical for the 9+8.
3. **triton.py standard `get_seed_config`**: a materialized rdim has NO reduction_loops spec entry,
   so emit `reduction_loops=[]` (not [None], which would fail normalize) — vanilla T1, floored
   block_sizes. (ws2 proved [] is correct for a non-block_sizes rdim.)
Expected: rms/ln/instance fire T1 with ~[1,1], beating their default. PURELY ADDITIVE (the 9+8 have
no materialized inner reduction → don't match the new branch).

## sub-problem B (hill-climb 1) — bias_grad/dyt M-collapse → GOOD T2 seed
The T2 seed floors the grid CTA m_block (block_sizes[0]) to 1 → M partials + M-way finalize. Fix:
size the grid CTA for occupancy (m_block ≈ next_pow2(grid_rows/num_sm)). Must be gated to the
M-collapse case only (don't perturb softmax/welford/kl_div/jsd + transfer user-tiled). Faithful
M-collapse signature: reduction over a grid-correlated (rows) axis with a [feature] accumulator
output, full_width_output=False. Derive the lever via oracle + hill-climb. NOT a default-crutch.

## sub-problem C (hill-climb 2) — group_norm multi-materialized → real seed
2 materialized inner reductions → register_unrolled_reduction's "exactly one materialized" skips it.
Must pick the feature reduction driving the resident footprint + produce a good config. Hardest.

## STATUS: implementing sub-problem A next.

═══════════════════════════════════════════════════════════════════════════
## ADDENDUM (2026-06-19) — sub-problem D: recover m_reduction perf for the dual-axis backward norms
═══════════════════════════════════════════════════════════════════════════
Re-opened the task-deprioritized "vanilla-T1 perf < old m_reduction" item (rms/ln/instance/group
backward). Stage 2 routed these to the standard MATERIALIZED branch but FLOORED the grid M block to 1
(`_m_block_cap`'s residency model), so the per-feature grad accumulator (`grad_weight[N]`) finalized
over M partials — the M-way collapse the m_reduction heuristic existed to fix. The kernels already
have the two-level tiling (`m_block = hl.register_block_size`; inner `hl.tile(mb_cta.begin,
mb_cta.end)`), so this is a HEURISTIC-only gap.

**Discriminator (compile-only probe `probe_mcollapse.py`).** `per_feature_accumulator` (the
5f34e9d1 fact field) is the faithful signal: True iff a loop-carried accumulator's dims are ALL the
materialized feature axis (the grad-param buffer summed across the grid). Confirmed:
- rms/ln/instance/group bwd: PFA=True, is_materialized=True, full_width_output=True, a non-grid inner
  tile present → the lever fires.
- 9 standard (rolled): is_materialized=False. 8 transfer: 7 rolled (is_materialized=False), grpo
  user-tiled+PFA=False → GATE_FIRES=False for ALL → disjoint, byte-identical.
- bias_grad/dyt: PFA=True but full_width_output=False + no inner tile (pure user-tiled M-collapse,
  already seeded) → not touched.

**The lever (triton.py `TritonStandardReductionHeuristic.get_seed_config`, materialized branch).**
Gated on `is_materialized AND per_feature_accumulator AND full_width_output AND inner_tile_ids`:
1. grid M block → `next_pow2(grid_rows/num_sm)` (`_m_collapse_grid_block`) — DOMINANT lever, shrinks
   the grad_w finalize from M partials to ~num_sm.
2. inner re-tile → `next_pow2(32768/(footprint*itemsize))` (`_m_collapse_inner_tile`) against the TRUE
   PRODUCT footprint (N for 2-D → inner 2-8; C*S for 3-D → inner=1, the ws2 spill-trap dodge; note
   `fact.feature_extent` is the per-axis MAX, which under-counts a 3-D footprint).
3. num_warps: drop the narrow-w1 lever (it keys on the rdim extent alone → floored instance's S=128 to
   1 warp; the resident tile is `[inner, feature]`-wide). Use the plain extent ramp.

**Findings.**
- First cut (M_CTA only, inner=1) won big on rms/ln/group but REGRESSED instance_norm 0.67→0.38. Root
  cause: narrow-w1 (instance has 0 carried tiles → narrow-w1 fires; group has 5 → it never did). cfg
  sweep: instance [4,1] w1=0.38, w4=1.52, w8=2.24 → it was the WARPS, not the M-collapse. Fixed by the
  narrow-w1 bypass.
- inner byte-cap (inner=2) closed rms (8192,4096) 0.94→1.07.
- The `oracle` mode for instance returned [32,32] with seed_over_oracle=0.108 (seed 9× FASTER than
  "oracle") — a broken autotune (matches ws2's "autotuner hangs on these"); disregarded.

**Perf** (G=tc/seed, fp32, before→after): rms (4096,4096) 0.90→1.49, (8192,1024) 0.72→1.81,
(8192,4096) 0.64→1.07; ln (4096,4096) 0.68→1.49, (8192,4096) 0.49→1.13; instance (512,64,128)
0.67→1.65, (1024,32,256) 0.43→1.13; group (512,128,64,32) 2.45→2.82, (1024,64,128,32) 1.42→1.76.
Wide-N (N=8192) NEUTRAL (rms 0.93→0.93, ln 0.74→0.75 — per-row reduction dominates; no regression).
bf16 spot-check all >1, acc-pass. Matches/exceeds old m_reduction (rms 4096²=1.49 vs 1.48; 8192×1024
=1.81 vs 1.85; ln 4096²=1.49 vs 1.52).

**Invariant** (config_recorder, final code vs stashed baseline): 9 standard 739/739 byte-identical
(fp32/bf16/fp16 × train/val/robustness); 8 transfer gate-disjoint; bias_grad/dyt unchanged.

Residual: wide-N tail (m_reduction's HBM-bytes warp ramp 16/32 would close it); full curriculum sweep
before landing. See REPORT.md "ADDENDUM" + RESULTS.md.

---

## Sub-problem E (post-D) — M-collapse gate over-specified; rms/ln_bwd lost at non-pow2/wide N

Audit of the unified stack vs the W1 standalone `TritonMReductionHeuristic` found rms/ln_bwd fired the
floored `[1,1]` seed (G≈0.5–0.9) instead of the W1 `[32,2]` collapse (G≈1.2–1.6) on most shapes.
Bind-verified root cause (fp32+bf16): the D gate `is_materialized and per_feature_accumulator and
full_width_output and inner_tile_ids` is over-specified — `per_feature_accumulator` already implies
`is_materialized`, and `full_width_output` is a FALSE NEGATIVE at non-pow2 N (feature next_pow2 padding
spawns an extra block → the grad_x store axis no longer block-id-matches the rdim), so the collapse
fired only at small pow2 N.

Fix: gate = `per_feature_accumulator and inner_tile_ids` (provenance + the structural guard that the
grow-outer / byte-cap-inner decomposition exists). Verified by config binds: non-pow2 rms/ln_bwd now
`[1,1]→[m_cta,1]` (m_cta=next_pow2(M/num_sm)); pow2 unchanged; 0/18 forward cells changed (gate never
fires for PFA=False); bias_grad/dyt/group/instance unchanged; (2048,6144)/(2048,7168) correctly decline
(inner_tile_ids empty — kept safety gate is the deciding vote). Config-only; not timed in this commit.

---

## Post-review code-cleanup pass (2026-06-19) — readability/robustness, behavior-preserving

> NOTE: commit hashes cited anywhere in this log are APPROXIMATE — this stack is rebased repeatedly
> (e.g. to keep each cleanup commit in its own stage group), so SHAs drift on every rebase. Identify
> commits by their `[stageN-...]` prefix + description, not by the hash.

A review-driven cleanup of the Stage-2 recognizer + fact code. No behavior change for any tested kernel
(verified). Banked as: `[stage1-liveness]` test fixes, `[stage2-unify]` code cleanup, this `[stage2-lab]`
record.

**Stage 2 code cleanup** (`device_ir.py`, `config_spec.py`, `triton.py`):
- Dropped the dead `bid == red_block_id` clause in the floored-apply-loop warn loop — unreachable (the
  loop iterates `bs_ids` and `red_block_id ∉ bs_ids` by construction; `bid in red_block_ids` subsumes it).
- Added a `len(materialized_reduction_axes) > 1` warning. group_norm is the ONLY tested kernel that hits
  the multi-axis case (S=64 + Cg=4; only the dominant becomes the fact). Confirmed the emitted config is
  invariant to the pick, but `body_live_tiles`/`num_carried_2d_tiles` are axis-specific (tie-break risk
  if a future lever reads them).
- Unified `if materialized / elif user-tiled / else` into one priority-pool body
  (`pool = materialized_reduction_axes or inner_red`; dominant-by-extent + warn-if-many); dropped the
  `all_qualified` decline (floor+warn instead). This also **dropped the `has_matmul` gate**, which was
  introduced by Stage 3 (`0e991a86`) — strictly a Stage-3 concern leaking into Stage 2. Per human
  decision the leak is acceptable: the gate removal is behaviorally inert (pure matmul → empty-pool
  decline; matmul+materialized rides the same path; no tested kernel changes; `test_matmul_heuristics`
  passes).
- Removed the now-dead `_all_qualified` 2nd return of `_non_reduction_loop_candidates` (only reader was
  the deleted elif decline).
- Renamed `materialized_inner` → `materialized_reduction_axes` (the reduced-OVER set); added shared
  `_is_materialized_axis` helper whose docstring names the two provenances (reduced-OVER via
  `ReductionLowering` vs feature-footprint via `bs.reduction`); pinned `bs.reduction` = registration
  *role*, not "a reduction is lowered over it" (e.g. bias_grad's `[N]` output).
- Renamed `ReductionFact.feature_extent` → `max_feature_extent` (it is a MAX over the materialized
  feature axes).
- Replaced the `(None,)` `per_feature_accumulator` special-case with `_resolve_accumulator_dim_block_id`:
  `resolve_block_id`, else a **unique** `next_pow2(size_hint)` extent match (root cause: grad buffers are
  padded to next_pow2, so a logical [1000] axis is a [1024] buffer dim that matches neither the block
  size nor any origin → `resolve_block_id` returns None). Declines + warns on an ambiguous (≥2 same
  padded-extent) match rather than guessing — a wrong guess would corrupt the identity-keyed
  `num_carried_2d_tiles` (`dim_block_ids[-1] == red_block_id`). `per_feature_accumulator` simplifies to
  `all(d in materialized_feature_axes ...)`, now N-D-correct. Limit (documented): unhandled only for a
  genuine same-extent collision, which no extent scheme can resolve without real provenance.

**Stage 1 test fixes** (`test/test_autotuner_heuristics.py`, banked `[stage1-liveness]`) — two
pre-existing failures, base tests broken by Stage-1 commits:
- `test_kl_div_wide_seeds_band_b_r_block_cap`: `_bandb_r_block_cap` → `_carried_tile_r_block_cap` (renamed
  by `440e8717`).
- `test_persistent_seed_round_trips_through_config_generation`: thread `row_reread=True` through
  `_reduction_spec` so a wide reduction persists under the `row_reread` gate added by `53c88088`.

**Verification** (subagents, GPU0): 6 m-reduction seeds byte-identical at every step (rms 2/[64,2],
ln 1/[64,2], bias_grad 1/[128,128], dyt 2/[128,8], group 3/[4,1], instance 3/[4,1]);
`test_autotuner_heuristics + test_matmul_heuristics + test_reductions` = **52 passed / 0 failed** (was
50/2 before the test fixes); non-pow2 recovery confirmed (bias_grad N=1024 & N=1000 both resolve the
acc dim to `(2,)` and `per_feature_accumulator=True`).
