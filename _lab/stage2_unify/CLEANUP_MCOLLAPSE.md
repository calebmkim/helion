# Post-landing M-collapse seed cleanup (behavior-preserving)

A second cleanup pass over the M-collapse seed path (after `8331cf03`), reconciling
the two tracks that now share the `per_feature_accumulator` signal:
- **standard** (`TritonStandardReductionHeuristic`): rms/ln/instance/group backward,
  materialized rdim, grid-grow + inner re-tile.
- **user-tiled** (`TritonUserTiledReductionHeuristic`): bias_grad/dyt, grid-grow + the
  reduction tile (`r_block`) as the slab.

## What changed (code commit `[stage2-unify] M-collapse seed cleanup`)
1. **Standard gate restructured.** `if per_feature_accumulator and inner_tile_ids:` →
   `if per_feature_accumulator:` with `inner_tile_ids` built inside and an inner
   `if inner_tile_ids:` skip. `per_feature_accumulator` already implies a device carry
   loop in `inner_tile_ids`, so the second term was a near-tautology as a *gate*; it
   stays as the cap-target list. Comment notes the real residual assumption is the
   **axis-match** (the inner tile re-tiles the grown `m_block`), not mere existence.
2. **User-tiled `grid_rows > 0` guard dropped.** The occupancy math self-floors to 1
   on an unbacked/AOT grid (`grid_rows == 0`), so `r_block == 1` there is a deliberate
   fall-through (a worse *seed* the autotuner refines, never a correctness issue) — same
   end state as the standard track's unguarded grid block. Documented in-place.
3. **Shared primitives extracted to `_TritonReductionSeedBase`.**
   - `_m_collapse_grid_block(env, fact, cap=None)` — occupancy block; `cap` distinguishes
     the tracks (user-tiled caps at `M_COLLAPSE_MAX_CTA` because its block doubles as the
     reduction slab; standard is uncapped, residency rides the inner tile).
   - `_m_collapse_inner_byte_cap(feat_bytes)` — the byte-cap formula, used by both.
4. **`max_feature_extent` → `feature_footprint`.** The fact now stores the **product** of
   the materialized feature axes (computed once in `_assemble_reduction_fact`) instead of
   the **max**. Both tracks read it; the standard track's `_m_collapse_inner_tile`
   `env.block_sizes` re-scan is deleted. `product == max` for the 2-D user-tiled kernels
   (so byte-identical), and the product is spill-correct for a hypothetical 3-D user-tiled
   collapse (max would under-count the footprint by the other axis — the ws2 spill).

## Verification (config_recorder + seed dumps, before/after via `git stash`)
- 9 standard reduction kernels, full matrix (fp32/bf16/fp16 × train/val/robustness, 739
  cells): **0 changed** (per_feature restructure pass).
- group_norm/instance_norm (standard) + bias_grad/dyt (user-tiled), varied shapes incl
  3-D wide-S, × fp32/bf16: **0 changed** across every cleanup step (210 / 36 / 18-cell
  samples). Lever firing confirmed (grid-grow `[4,1]`/`[8,1]`, inner-cap `[1,16/32/64]`,
  wide-S floor `[1,1]`), so the checks are not vacuous.
- `ruff check` clean; parse/import smoke clean; `max_feature_extent` fully retired (0 refs).

Only intentional behavior change: the `grid_rows == 0` (unbacked/AOT) `r_block == 1`
fall-through, which static-shape config recording does not exercise.
