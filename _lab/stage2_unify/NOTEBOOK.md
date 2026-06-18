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
