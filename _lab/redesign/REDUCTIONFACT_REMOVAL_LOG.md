# ReductionFact-removal work log (Stage 2 cleanliness)

Task brief: `_lab/redesign/STAGE2_REDUCTIONFACT_REMOVAL_RECIPE.md`
Goal: make `_TritonReductionSeedBase` + the two reduction tracks read `pd`
(`ReductionDescriptor`) for ALL scalar reads, never a `ReductionFact`. Behavior-identical
cleanliness. `ReductionFact` stays built (matmul-epilogue + eligibility gate). Do NOT touch
`TritonMatmulReductionEpilogueHeuristic` (@~2048). No GPU.

## Step 0 — orientation + clean-start reconciliation (2026-06-29)
- HEAD = `5d02a00f` "docs: enrich ReductionFact-removal brief into a self-contained task".
- SURPRISE: `git status` was NOT clean — `triton.py` had an uncommitted diff (+26 -67) that had
  already renamed `_primary_fact` -> `_primary_descriptor_selected` (§4 selector) and rewritten
  `_is_standard_reduction(pd)` to the one-liner. BUT it was HALF-DONE / top-down: the call sites
  at lines 1668/1669/1703/1887 still called `_primary_fact(env)` and `_is_standard_reduction(spec, ...)`,
  so the tree did NOT compile. This is an interrupted prior attempt (a stale task list #1-4 from a
  prior session also present).
- DECISION: discard the uncommitted diff and restart from a clean HEAD using the brief's BOTTOM-UP
  slicing (§8), so every commit compiles + gates green. The discarded content is exactly the §4
  selector and is reproduced verbatim later — nothing is lost. Reasoning: the brief's hard rule is
  an empty `git status --short` at start, and bottom-up avoids the non-compiling intermediate the
  half-done diff left. (Memory rule: don't blindly self-revert — but here the diff is verifiably
  the §4 text and verifiably non-compiling, so discarding + redoing bottom-up is correct.)
- After discard: `git status --short` empty, HEAD `5d02a00f`. Baseline gate to be run before edits.

## Step 1 — flip all reduction-track readers to pd (IN PROGRESS — gate RED, investigating)
- Applied §4 selector (`_primary_descriptor_selected`) + one-line `_is_standard_reduction(pd)`;
  flipped all 16 base helpers' `fact: ReductionFact` -> `pd: ReductionDescriptor` + field map;
  dropped `fact` param + kf-None fallback from `_grid_axis_block_ids`/`_non_reduction_loop_ids`/
  `_secondary_red_values` (assert kf not None); removed now-dead `_primary_descriptor` +
  `_carried_2d_count` (callers use `pd.carried_2d_count` directly per §3/§5.2); threaded `pd`
  through both `get_seed_config` + both `is_eligible` heads; removed `ReductionFact` import (now
  doc-only). ruff format+check clean, module imports OK.
- GATE config recorder: RED — **15 movers, not 2**. The 2 known (layer/rms_norm_bwd) PLUS 13
  `curriculum/welford/*/fp32` cells, all `block_sizes[0]` (the grid M axis) 1 -> 2.
- HYPOTHESIS: the primary-descriptor SELECTION for welford differs from the legacy `_primary_fact`
  on a SCALAR the M-axis sizing reads (`_m_block_cap`/`_resident_tile_cap` use
  full_width_output / body_live_tiles / itemsize / size_hint). welford has multiple sized
  descriptors; the max-row-bytes pick may carry a different body_live_tiles/full_width_output
  than the legacy aggregate fact. INVESTIGATING before any commit (per hard rule: a NEW mover =
  STOP, do not commit). NOT a GPU question — pure selection/field divergence.

## Step 1 — ROOT CAUSE of the 13 welford movers + faithful fix
- validate_kernel_fact still 460/460, so descriptor scalars ARE 1:1 with the legacy fact under
  INV3. The divergence is NOT selection (both selectors pick welford bid=1) — it is a SINGLE
  scalar the brief's §3 field map over-simplified: **`full_width_output`**.
  - legacy `ReductionFact.full_width_output` keys on `extent_axes = {rdim} ∪ non_reduction_loops`
    (device_ir `_assemble_reduction_fact` ~L1933): a rank>=2 store over the rdim OR over a
    NORMALIZE-LOOP axis. welford's combine writes a scalar (no full-width store over rdim) but its
    normalize pass writes the full row over the loop axis -> legacy = True.
  - the descriptor's `full_width_output` (`_per_reduction_memory_fields` ~L1643) keys ONLY on the
    rdim, EXCLUDING the normalize loops -> welford `pd.full_width_output = False`.
  - INV3 KNOWS this (L150-187): it reconstructs the legacy scalar as
    `any(group desc fwo) OR loop_has_full_width_store(non_reduction_loops)`, which is why INV3 was
    green while a NAIVE `pd.full_width_output` read in `_m_block_cap` flips welford.
  - the only consumer of full_width_output is `_m_block_cap`: True -> apply M_BLOCK register cap
    (keeps welford M=1); False -> no cap -> M widens to 2. THAT is the block_sizes[0] 1->2 mover.
- FIX (more-faithful read, NOT a config change — brief explicitly allows re-keying onto a faithful
  read): add `_full_width_output(spec, pd) = pd.full_width_output OR loop_has_full_width_store(
  spec, _non_reduction_loop_ids(spec))`, and have `_m_block_cap` take `spec`+`pd` and read it.
  VERIFIED: this two-term reconstruction == legacy `fact.full_width_output` across ALL 443
  reduction cells (0 mismatch, /tmp probe). Matches the legacy `extent_axes` exactly.

## Step 1 — COMPLETE (gate GREEN), about to commit as ONE atomic commit
- The flip is necessarily ONE commit: the 16 helpers, the kf-None-fallback drops, the dead-helper
  removals, and both track heads are mutually call-coupled, so any partial slice would leave a
  non-compiling tree (violating the every-commit-compiles hard rule). Bottom-up "convert helper +
  callers together" collapses to a single atomic unit here.
- Changes (helion/_compiler/autotuner_heuristics/triton.py):
  - selector `_primary_descriptor_selected(env)` replaces `_primary_fact`; `_is_standard_reduction(pd)`
    one-liner (FULL_EXTENT_CATEGORIES).
  - 16 base helpers `fact: ReductionFact` -> `pd: ReductionDescriptor`, field map applied
    (size_hint/itemsize/input_load_itemsize/row_reread/reread_eviction_index/num_load/
    body_live_tiles/full_width_output -> pd.*; num_carried_2d_tiles -> pd.carried_2d_count;
    primary_reduction_block_id -> pd.block_id).
  - `_grid_axis_block_ids`/`_non_reduction_loop_ids`/`_secondary_red_values`: dropped `fact` param +
    kf-None fallback (now `assert kf is not None` — reachable only post-selector).
  - removed now-dead `_primary_descriptor` + `_carried_2d_count` (callers read pd.carried_2d_count).
  - NEW `_full_width_output(spec, pd)` helper (the more-faithful read fix); `_m_block_cap` takes
    `spec`+`pd` and reads it. This is the ONLY deviation from §3's bare field map (full_width_output
    is per-rdim on the descriptor but the legacy scalar OR'd in normalize-loop full-width stores).
  - removed the now-unused `ReductionFact` TYPE_CHECKING import; reworded 3 docstring `fact.X`
    legacy mentions so the reduction tracks (581-1980) contain ZERO `fact.` tokens.
  - both `get_seed_config` + both `is_eligible` heads thread `pd`; user-tiled head gains a
    `pd is None -> return None` guard (type-correct, mirrors standard track; unreachable post-gate).
- Tests ported (test/test_autotuner_heuristics.py):
  - `_reduction_spec` now also appends a `ReductionKernelFact` (FULL_SLICE desc bid=1, grid=(0,))
    mirroring the legacy fact; keeps the ReductionFact too (eligibility-gate + matmul-disqualify
    test). `_reduction_env` configures `env.block_sizes[bid].size` to a backed int for descriptor
    axes (selector needs it) and a MagicMock for the grid axis (so `_grid_rows` stays 0 = narrow-w1
    disabled, preserving prior mock behavior).
  - `test_dynamic_extent_normalize_tile_matches_reduction_tile`: builds a USER_TILE
    `ReductionDescriptor` (static_rnumel=None) + a minimal kernel fact (grid=(0,), nrl=(norm,)) so
    the directly-called `_build_block_sizes` resolves grid/loop axes; asserts unchanged.
  - `test_kl_div_wide_seeds_band_b_r_block_cap`: resolves the primary `pd` via the selector and
    passes it to `_carried_tile_r_block_cap`; asserts pd.carried_2d_count == fact.num_carried_2d_tiles.
- GATES: config diff = the 2 known movers ONLY; validate_kernel_fact 460/460; probes 13/13;
  test_reductions+test_autotuner_heuristics 52 passed / 22 skipped; matmul_layernorm 2 passed /
  2 skipped; ruff format+check clean.
