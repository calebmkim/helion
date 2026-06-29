# Stage-2 port plan: consume `ReductionKernelFact` directly, retire `ReductionFact`

> Goal (user-stated): the reduction heuristic should read `ReductionKernelFact` + its
> descriptors/groups **directly**, not a flattened `ReductionFact` derived from it. The kernel
> fact strictly *contains more* than the legacy fact, so the consuming code should get SIMPLER.
> End state = `ReductionFact` no longer built/consumed by the reduction path (matmul-epilogue
> §2.9 keeps whatever minimal shape it needs). Staged 3 → 1 → 2 for safety; every step is
> config-recorder zero-diff gated (the 2 known square-shape movers are the only allowed delta).

## The problem (measured)

Field-read census over the reduction heuristic (triton.py 549–1850):
- `fact.<legacy>` read **57×** across 14 fields — drives ALL sizing/caps/warps/persistence.
- `kf.<kernelfact>` read **5×**, only in 3 patch helpers (`_grad_collapse_group`,
  `_secondary_red_values`, `_coresident_with_other_sized`) — and even those re-anchor to
  `fact.primary_reduction_block_id`.

The kernel fact is built, validated reconstructible-into the legacy fact, then **ignored**; the
heuristic consumes the derived legacy fact. `_build_block_sizes` re-derives the taxonomy at
sizing time (the rolled/user-tiled/pinned branch = FULL_SLICE/USER_TILE/FULL_GRID) — exactly the
Stage-2-re-derives-structure anti-pattern §2.1/§6.1#2 set out to delete.

## Field → kernel-fact source map (the substitution table)

Every legacy field a consumer reads, and where it comes from on the kernel fact. "primary
descriptor" `pd` = the descriptor whose `block_id == _primary_fact`'s pick (see below).

| legacy `fact.X` | reads | kernel-fact source |
|---|---|---|
| `primary_reduction_block_id` | 10 | the primary descriptor's `block_id` (priority order; §6.2) |
| `m_block_ids` | 10 | `kf.grid_axis_block_ids` (parallel rows) — BUT see ⚠️ M1 below |
| `size_hint` | 7 | `pd.size_hint` |
| `num_carried_2d_tiles` | 7 | `sum(d.carried_2d for d in group)` (count over the co-resident group) |
| `itemsize` | 6 | `pd.itemsize` |
| `body_live_tiles` | 4 | `pd.body_live_tiles` (or max over group for a kernel scalar) |
| `secondary_reduction_block_ids` | 3 | sized descriptors minus primary — `_secondary_red_values` already does this from kf |
| `row_reread` | 2 | `pd.row_reread` |
| `per_feature_accumulator` | 2 | `_grad_collapse_group(kf)` already replaces the standard-track copy; user-tiled copy (bias_grad/dyt) → still needs a kf signal (see ⚠️ U1) |
| `non_reduction_loop_block_ids` | 2 | `kf.non_reduction_loop_block_ids` |
| `reread_eviction_index` | 1 | `pd.reread_eviction_index` |
| `num_load` | 1 | `pd.num_load` |
| `input_load_itemsize` | 1 | `pd.input_load_itemsize` |
| `full_width_output` | 1 | `pd.full_width_output` (OR over group + normalize loops for the kernel-scalar cap — see INV3 in validate_kernel_fact) |

Every field has a home. INV3 (validate_kernel_fact.py) already PROVED these match the legacy
fact on the primary across 443 cells — so the substitution is information-preserving.

## The two functions that get SIMPLER (the payoff)

1. **`_build_block_sizes`** `r_block_resident` 3-way branch (lines 1019-1024) — today re-derives
   rolled/user-tiled/pinned from `spec.reduction_loops` membership + a `red_values` dict. With
   the kernel fact: read `pd.category` (FULL_SLICE→persistent next_pow2; USER_TILE→its sized
   r_block; FULL_GRID/`pd.pinned`→1). **This is the consumer for the currently-unused `pinned`
   field.** `red_values` (a dict the caller pre-builds by re-scanning sized reductions) → readable
   straight off the sized descriptors.

2. **`_primary_fact`** ("max backed size_hint" over `reduction_facts`) → the priority-order
   primary over `kf.reductions` (§6.2: full-extent → user-tile → grid-tile, extent tiebreak).
   The §6.2.1 num_warps owner question is answered by the same ordering. NOTE: §2.5/worklog say
   the priority-order primary DIVERGES from max-size_hint on rms_norm_per_block (picks FULL_GRID
   over USER_TILE) — so this swap is NOT automatically byte-identical; must verify per-cell.

## ⚠️ Byte-identical risks (where a naive swap moves a config)

- **M1 — `m_block_ids` vs `grid_axis_block_ids`.** The legacy `m_block_ids` is `grid_ids` MINUS
  full-grid reduction axes (the p8 fix, device_ir ~1358). `kf.grid_axis_block_ids` =
  `grid_ids - sized_bids`. These should match (a FULL_GRID sized axis is excluded from both) but
  must be diffed per cell — a GRID_TILE axis is in m_block_ids (kept as a row) AND not sized, so
  in grid_axis too; a FULL_GRID is excluded from m_block_ids and IS sized so excluded from grid
  axis. Likely equal; VERIFY.
- **U1 — user-tiled `per_feature_accumulator`** (bias_grad/dyt). The standard-track copy is
  already on `_grad_collapse_group`. The user-tiled copy still reads `fact.per_feature_accumulator`
  directly. It needs an equivalent kf signal: "a loop-carried accumulator over ALL the
  materialized feature axes." This is an accumulator-shape property (AccumulatorFact +
  materialized_feature_axes), NOT currently on the descriptor — may need a new descriptor/group
  field or a kf-level helper. (This is the retained-proxy the audit flagged earlier; porting it
  is where we either make it faithful or confirm it must stay.)
- **P-primary — the priority-order primary** diverges from max-size_hint on rms_norm_per_block
  (above). Either keep the max-size_hint pick AS the priority order's extent-tiebreak result, or
  accept the divergence and re-verify the cell stays byte-identical.

## Staging (3 → 1 → 2)

- **Step 3 (proof of concept, reversible):** port `_build_block_sizes` + its inputs
  (`r_block_resident`, `red_values`, `m_block_ids`, `non_reduction_loop_ids`) to read the kernel
  fact. Add a `pd`/`group` accessor. Keep `ReductionFact` built and passed elsewhere. GATE:
  zero-diff. Resolves M1. Proves the `pinned`/category read replaces the 3-way branch.
- **Step 1 (full port, keep `ReductionFact`):** re-key the remaining consumers
  (`_reduction_rblock`, `_num_warps`, `_m_block_product`, both `get_seed_config` heads,
  `_primary_fact`→priority order) to the kernel fact + descriptors. Resolves U1, P-primary.
  `ReductionFact` still BUILT (matmul-epilogue + cross-check) but the reduction heuristic no
  longer reads it. GATE: zero-diff after each consumer.
- **Step 2 (retire `ReductionFact` from the reduction path):** matmul-epilogue (§2.9) reads only
  `reduction.size_hint` (→ n_extent) + the gate `len(reduction_facts)==1`. Give it a minimal
  composed value from the kernel fact (the single full-extent descriptor's size_hint) OR keep a
  tiny ReductionFact built ONLY for that composer. Drop the legacy builder
  (`_assemble_reduction_fact`, `register_unrolled_reductions`' fact emission) from the reduction
  path. Update the 2 unit tests that construct `ReductionFact` directly
  (test_autotuner_heuristics.py:719, 3646). GATE: zero-diff + full unit suite.

## Gating policy (user directive: faithfulness/cleanliness > strict config equivalence)
- The port is "same logic, different fact" — so it SHOULD stay byte-identical where the read is
  a faithful 1:1 substitution. But a config MOVE is ALLOWED when the kernel-fact read is the more
  faithful one (e.g. priority-order primary on rms_norm_per_block). For every mover: GPU-measure
  it stays within 10% of the prior config; if a faithful read regresses >=10%, STOP and escalate
  to the user with the situation. Do NOT contort the code to preserve a config the legacy proxy
  happened to produce.
- So the per-step gate is: config-diff vs baseline → for each NEW mover (beyond the 2 known
  square-shape ones), GPU head-to-head old-vs-new, accept if <10%, escalate if not.

## Invariants to hold every step
- validate_kernel_fact.py 460/460; probe_assertions.py 13/13.
- test_reductions.py + test_autotuner_heuristics.py green.
- matmul-epilogue: test_examples -k matmul_layernorm green (§2.9 must-not-break).

## Findings while porting (verified)
- **M1 RESOLVED**: `fact.m_block_ids == kf.grid_axis_block_ids` on every corpus cell, and BY
  CONSTRUCTION: among grid axes the SIZED ones are exactly the FULL_GRID ones (USER_TILE/FULL_SLICE
  are never grid axes), so `grid_ids − sized` ≡ `grid_ids − full_grid` (the legacy m_block_ids).
- **`_primary_fact` = max-ROW-BYTES, NOT tier-order.** The naive §6.2 priority-order primary
  (category tier first) picks the FULL_SLICE group axis (bid3, sh=128) over the dominant USER_TILE
  RMS sum (bid1, sh=4096) on rms_norm_per_block_quant → flips track + warps → regression. The
  §6.2.1 rule (max `size_hint * input_load_itemsize` over backed sized descriptors) is the faithful
  one for the SCALAR-LEVER primary and is ZERO-divergence from legacy on the whole corpus.
  Tier-order is for ALLOCATION BIDDING (a different question §6.2 conflated). Corpus has exactly 1
  reduction_fact per kernel (silu_mul_fp8=0, pointwise), so the multi-fact branch is exercised only
  by probe p7.
- **carried_2d bool → carried_2d_count int** (committed): the descriptor was lossier than the
  legacy fact (count is load-bearing in the carried cap denom); now a faithful superset.

## Field-equivalence nuances found while mapping (preserve or consciously change)
- `num_carried_2d_tiles` legacy = count of accumulators whose `dim_block_ids[-1] == PRIMARY
  rdim`. Descriptor `carried_2d` = same `[-1]==block_id` test but per-descriptor. The faithful
  port: `num_carried_2d_tiles` for a reduction = whether THAT reduction's own `carried_2d` (and
  for caps that need a group total, sum over the group). For the corpus (one carried reduction
  per kernel) these coincide; keep the per-reduction read so multi-carried kernels are faithful.
