# Budget-allocator rewrite — running log

Task: rewrite the reduction seed's `size_reduction_tiles` (helion/_compiler/autotuner_heuristics/
triton.py) into ONE per-co-residency-group budget allocator. DELETE the recognizers
(`_is_per_feature_accumulator`, `_grad_collapse_group`, the `inner_tile_ids` override, the
`_carried_*`/`_resident_tile_cap` footprint caps, the standard-vs-user sizing split). Floor-vs-
resident must fall out of budget depletion, not branches. Configs WILL change — record every
changed cell + perf, hill-climb the BUDGET constants only, keep correctness gates green.

Brief: `_lab/redesign/BUDGET_ALLOCATOR_DESIGN.md`. Companions: PROMPT.md §2.3/§2.4/§6.2.1,
CARRIED_AND_GREEDY_FINDINGS.md, GREEDY_ALLOCATOR_LOG.md (the REJECTED window-dressing).

Branch `reduction-redesign`. Interpreter `/home/dev/helion/.venv/bin/python`.
`HELION_AUTOTUNE_EFFORT=none` for recorder/validators/tests. GPU FOREGROUND only, never detached.

---

## Step 0 — start (HEAD 34ae072e, tree clean)

Read the brief + all companions + the full `_TritonReductionSeedBase` + both subclasses + the
Stage-1 fact structs. Key understanding:

- The CURRENT `size_reduction_tiles` (committed by the prior `GREEDY_ALLOCATOR_LOG.md` run) is the
  REJECTED window-dressing: it renamed the scattered passes into one function but kept byte-identical
  configs AND every recognizer (`_is_per_feature_accumulator`, `_grad_collapse_group`, the
  `inner_tile_ids` post-write override, the `if standard/else` sizing split, `red_values={} if
  standard`, `_carried_tile_r_block_cap`, `_carried_m_block_cap`, `_carried_grid_dims`,
  `_resident_tile_cap`, `_pinned_inner_resident_elems`). This rewrite DELETES all of that.
- The TARGET (§2): iterate co-residency groups in priority order; per group form ONE scalable budget
  (scaled by `num_live_tiles_in_group` + loop-carried accumulator count); seat axes
  first-crack-then-floor-by-remaining (full-extent → user-tile → grid-tile → grid-M remainder); hold
  fixed; next group; then `non_reduction_loop_block_ids` LAST. Floor-vs-resident = budget depletion
  outcome. Standard/user split = EMISSION ROUTING ONLY (reduction_loops knob vs block_sizes slot).
- `num_live_tiles_in_group` (pinned in §2): `max(d.body_live_tiles for d in group's SIZED reductions)`
  (defaults to 1). NOT a stored field; `AccumulatorFact` has no graph_id so per-group accumulator
  count isn't cleanly attributable — `body_live_tiles` is the group-attributable scaling property.
- Footprint (keep SIMPLE, §2): `group_footprint ≈ num_live_tiles_in_group × (∏ assigned tile sizes
  in the group) × itemsize`. Over-estimate is safe (errs toward flooring).
- KEEP intrinsic non-footprint caps: EXTENT, OCCUPANCY (grid ≥ num_sm·MIN_WAVES), PERSISTENCE
  (re-read row wants resident). num_warps stays a scalar lever OUTSIDE the budget, keyed on primary
  row bytes (`size_hint × input_load_itemsize`).

### Pre-edit baseline GATES (all GREEN at 34ae072e)
- config recorder /tmp/before_rewrite.json: 447 cells, 0 errored.
- diff vs frozen baseline_fc1dbaa0_configs.json = ONLY the 2 known movers
  (mreduction/{layer,rms}_norm_bwd/[4096,8192]/fp16, [1,1]→[32,1] — pre-existing starvation fix).
- validate_kernel_fact: 460/460. probe_assertions: 13/13.
- /tmp/before_rewrite.json = the CURRENT-HEAD config reference (my "before" for MY changes).
- baseline_fc1dbaa0_configs.json = the ORIGINAL heuristic perf reference (§6).

## Step 1 — DESIGN (validated numerically against the answer-key; ALLOCATOR_ANSWER_KEY.md)

Built the full corpus answer-key (every distinct structure → facts + current seed). 7 structures:
S1 full_slice rolled persistent (rms_norm/sum/long_sum/cross_entropy); S2 user_tile persistent
(softmax/dynamic_per_token); S3 carried-2D (kl_div/jsd); S4 full_grid resident (per_token_group);
S5 reduce-then-apply (welford); S6 multi-reduction sequential (rms_norm_per_block); S7 grad-param
M-collapse (bias_grad/dyt/*_bwd) ← the crux.

### THE allocator (one rule, no recognizers)
Per co-residency group, form ONE budget; seat axes first-crack-then-floor-by-remaining; hold fixed.
- **group budget** (capacity, in element-product terms): `num_live × ∏(seated tiles) × itemsize ≤
  ROW_PERSIST_MAX_BYTES`, where `num_live = max(d.body_live_tiles for SIZED d in group)` (§2 pin),
  `itemsize` = primary's fp32-promoted itemsize. num_live is the §3 legitimate budget SCALE
  (continuous, uniform, group-attributable).
- **seat reductions** in priority order full-extent→user-tile→grid-tile (extent tiebreak): each
  computes DESIRED size (persistence: full extent if row_reread & fits, else looped chunk min
  LOOPED_CHUNK), FLOORED by remaining budget. A grid-tile reduction (jsd) → desired ~1 (grid
  parallelizes it) → floors. Emission routing ONLY: rolled→reduction_loops, user-tiled→block_sizes.
- **grid-M axes take the remainder.** KEY UNIFICATION (the crux, no recognizer): a grid axis's
  treatment falls out of ONE faithful per-axis footprint property —
  `axis_reduced_away = accumulators_exist AND (axis NOT in any accumulator's dim_block_ids)`:
    * reduced_away (bias_grad CTA, norm-bwd grid-M): the axis is a SEQUENTIAL cross-grid reduction
      loop (its partial is finalized by .sum(0)), NOT resident → NOT in footprint ∏; its floor is
      RAISED to grid_rows/num_sm (collapse the finalize — the SAME §2 "reuse raises the floor"
      mechanism as full-slice persistence: reuse the cross-grid partial across more rows). Capped by
      extent. THIS is the collapse; flooring to 1 = the [1,1] catastrophe the 2 known movers fixed.
    * independent (rms_norm rows, softmax, per_token_group sibling, kl_div grid): resident row tile →
      IN footprint ∏; widens into remaining byte budget, capped by OCCUPANCY (grid ≥ num_sm·MIN_WAVES)
      + extent; FLOORS to 1 when budget full (wide rms_norm/cross_entropy/kl_div). Consumes budget.
  Membership (in-accumulator) decides BOTH in-footprint-ness AND occupancy direction — same physical
  fact (resident vs sequentially-reduced), applied uniformly to every grid axis. NOT a kernel/shape
  gate, NOT `_is_per_feature_accumulator` (a whole-kernel "all dims are features" recognizer picking a
  different FORMULA). Passes §3: a per-axis footprint property feeding the SAME clamp formula.
- **non_reduction_loops LAST**, own budget (welford normalize, rms_norm_per_block groups_per_row).

### Numerically validated (pre-implementation hand-trace, /tmp scripts):
- S1 rms_norm grid-M: 5/5 exact ([4],[1],[8],[4],[2] for the traced shapes).
- S7 collapse: bias_grad bid0 = grid_rows/num_sm → 16/64/32; rms_norm_bwd → 16/32 (incl. the known
  mover 4096×8192→32). 5/5.
- S4 per_token_group sibling: occupancy cap → 2 (the genuine first-bind; the [8192,…]=[2] recorder
  value is cache-propagated from the [128,…] first bind — VERIFIED, not a value I must reproduce
  independently). S3 jsd/kl_div grid: floors to 1 (budget full).
- num_warps stays the _num_warps lever keyed on primary row-bytes (§6.2.1), OUTSIDE the budget.
  m-collapse warp_override likely droppable (narrow-w1 can't fire: row_bytes ≫ 2048). Verify in recorder.

### Subtleties locked:
- m-collapse routes to STANDARD track because the inner register_block_size axis is UNBACKED →
  _primary_descriptor_selected (backed-only) picks the FULL_SLICE feature reduction. Keep as-is.
- The m-collapse INNER tile shrink (dyt bid1=8) falls out of the feature footprint × num_live in the
  ONE budget — EXPECT a config shift here (current uses a distinct M_COLLAPSE_TILE_BYTES=32768);
  bench + recover via budget per §4. NOT a separate cap.
- Configs WILL move on S7 (norm-bwd/bias_grad/dyt) — that is the point; record + bench + tune.

## Step 2 — formula refinement (numerical, pre-implementation)

First-cut formula (verify_formula.py) reproduced the HARD cases exactly: rms_norm grid-M (5/5),
per_token_group (occupancy=2), rms_norm_bwd/layer_norm_bwd m-collapse INCL the known mover [32,1],
welford, dynamic_per_token. DIFFs found + root-caused:

- **softmax [128,128] vs [16,128]**: my verifier used grid floor=1; the REAL floor is `_block_floor =
  max(1, min_size, autotuner_min)`. softmax (262144,128) has autotuner_min=16 (huge-M shape), so the
  grid-M floor IS 16 (not budget-widened). FIX: grid-M widen floors at `_block_floor`, not 1. (16384,512)
  has autotuner_min=1 → budget/occupancy sizes it (8, matched). VERIFIED via live cap trace.
- **per_token_group [2] is CACHE-PROPAGATED**: standalone bind gives occupancy=128→32; in recorder ORDER
  the (128,…) shape binds first (occupancy=2) and the cached kernel reuses it for (8192,…). The genuine
  computation is occupancy-capped (=2 at the small shape). My budget reproduces the genuine value.
- **kl_div/jsd carried-2D**: the carried `[M,R]` accumulator is resident the WHOLE loop → tighter than a
  streamed chunk. body_live_tiles already counts the simultaneously-live copies (kl=6, jsd=12). Footprint
  = num_live × R × itemsize → kl R=8192 (was 4096), jsd R=4096 (was 2048): a clean 2× shift (EXPECTED;
  §2.5 flagged this exact class). BENCH; if it spills, tighten budget via a faithful carried scale.
- **rms_norm_per_block [4096,1] vs [4096,32]**: bid2 (groups_per_row) is an INDEPENDENT tunable loop
  (not reduction/grid/nrl) → must size to its OWN extent capped by budget, not floor to 1. Need the 4th
  seating pass (independent loops), as the current _build_block_sizes else-branch does.
- **S7 m-collapse inner tile** (bias_grad bid1, dyt bid1, instance/group): the inner reduction tile is
  `[inner_rows, *feature]` resident — the materialized FEATURE footprint multiplies it. Footprint must
  include materialized feature axes (faithful derivation, NOT in the delete list). bias_grad bid1 → ~32
  (was 16), dyt → ~16 (was 8): 2× shifts. EXPECTED; bench + recover via budget.

### Resolved footprint model: the §2.3 group_footprint (faithful, accumulator-based)
The resident set of a group = its loop-carried ACCUMULATORS (accumulator_facts) + the materialized
feature/pinned axes. Sizing axis A: `A ≤ BUDGET / (num_live × itemsize × ∏(other co-resident dims))`,
where co-resident dims come from the accumulators containing A (∏-within-buffer, classified by
MEMBERSHIP — rdim→seated r_block, grid-M→its block, materialized→full extent) + materialized features.
This SUBSUMES _resident_tile_cap / _carried_* / _pinned_inner / M_COLLAPSE_TILE_BYTES into ONE
footprint. num_live = max(body_live_tiles) in the group is the §3 budget SCALE. ONE budget
ROW_PERSIST_MAX_BYTES; floor-vs-resident + collapse-vs-widen fall out of per-axis membership.

## Step 3 — grid-M formula resolved (the self-referential cap discovery)

Traced the LIVE caps. The current grid-M sizing is a min of TWO overlapping caps:
- `resident_tile = ROW_PERSIST / (m_block × r_block_resident × inner × itemsize)` — NO num_live;
  r_block_resident = np2(size_hint) for FULL_SLICE (full extent even when LOOPED), r_block for
  USER_TILE, 1 for FULL_GRID. **It divides by m_block (the already-floored value) — SELF-REFERENTIAL**
  (so on huge-M shapes where autotuner_min raised the floor, the cap collapses toward the floor and the
  floor wins). prev_power_of_2.
- `m_block_register = ROW_PERSIST / (size_hint × itemsize × body_live_tiles)` — WITH num_live, but
  GATED on `_full_width_output` (huge no-cap otherwise). prev_power_of_2.
- plus `occupancy = pp2(grid_rows / (num_sm·MIN_WAVES))`, `extent`, floor=`_block_floor`.

### DECISION (principled, per brief §3 — uniform, non-self-referential):
ONE grid-M footprint cap, NO self-reference, num_live applied UNIFORMLY (drops the full_width gate):
  `grid_M ≤ max(floor, min(occ, pp2(ROW_PERSIST/(num_live × r_resident × itemsize × other_resident)),
   extent))`
where r_resident = the seated resident reduction width co-held when this axis widens (np2(extent) for
FULL_SLICE, seated r_block for USER_TILE, 1 for FULL_GRID), other_resident = ∏ of OTHER resident grid
axes + materialized features in this axis's accumulators. This:
  - reproduces rms_norm widen on the non-floored shapes (byte/occ bind, 4/8/1…);
  - DROPS the self-referential m_block division → on huge-M shapes the budget may widen PAST the old
    autotuner_min floor (rms_norm(262144,256): old [16]→ budget [64]) — a SHIFT to bench (262144 rows
    ≫ occupancy even at 64; expected safe);
  - DROPS the full_width gate on num_live → sum/long_sum (blt=2, not full-width) shift (sum(8192,8192):
    [4]→[2]) — a SHIFT to bench (memory-bound; halving rows/program raises program count, neutral-ish).
These are the EXPECTED, principled config moves. Record + bench + recover via budget if a mover regresses.
Floor-vs-resident + collapse-vs-widen remain pure membership outcomes (no recognizer).

### Reduction-side already matches (verify_formula): rms_norm/sum/long_sum/cross_entropy/softmax/
per_token_group/dynamic_per_token/rms_norm_bwd[32,1]/layer_norm_bwd/welford all OK. The carried-2D
(kl/jsd rdim) shifts 2× (num_live footprint) — EXPECTED (§2.5 flagged). The S7 grad-param inner tiles
shift (feature footprint) — EXPECTED. The independent loop (rms_norm_per_block groups_per_row) needs
its own seating pass (size to own extent, budget-capped).

## Step 4 — footprint must be GROUP-TOTAL resident set (not per-accumulator)

model2 (per-accumulator footprint) found the real bug class: a persistent FULL_SLICE row (cross_entropy,
fused_add_rmsnorm) holds the full reduction row resident, but its ACCUMULATOR is a per-row scalar
[M,None] — the resident row width is NOT an accumulator dim. So a per-accumulator footprint under-counts
and grid-M over-widens ([1]→[4]). Likewise the m-collapse INNER re-tile (group/instance/layer-bwd
[1,1]→[1,8192]) co-holds the feature accumulator but the inner-tile dim and feature dims live in
DIFFERENT accumulators.

FIX: the §2.3 group_footprint is the group's TOTAL resident working set — every axis sized against
`BUDGET / (resident_set ÷ this_axis)`. resident_set = num_live × itemsize × ∏(all seated reduction
r_blocks in the group) × ∏(materialized feature/output extents) × ∏(resident grid-M blocks). A persistent
FULL_SLICE contributes its full extent; the m-collapse inner re-tile is sized against (features × ...),
so it shrinks automatically — the M_COLLAPSE_TILE_BYTES cap falls out. This is the "num_live × ∏ assigned
tile sizes × itemsize" of the brief, where ∏ ranges over the WHOLE group's seated tiles + pinned features.

## Step 5 — persistence is a TWO-TERM test (single-tile vs ROW_PERSIST, multi-tile vs LIVE_PERSIST)

model3 (313 OK) chunked softmax's persistent user-tile rdim ([1,32768]→[1,16384]) because I applied
num_live to the chunk cap directly. The faithful model (current _reduction_rblock, generalized to BOTH
tracks): persistence grant tests the SINGLE resident tile against ROW_PERSIST (num_live=1) AND the
num_live-tile live set against LIVE_PERSIST_BUDGET=3×ROW_PERSIST. softmax: 32768×4=131072≤245760 ✓,
2×…=262144≤737280 ✓ → STAYS persistent 32768. So num_live enters (a) persistence DENIAL for heavy
bodies (fused_linear_jsd) and (b) the LOOPED chunk shrink — NOT the single-tile persistence grant. This
is uniform across tracks (the current user-tiled ff=1 vs standard ff=blt split is the ad-hoc thing
removed; LIVE_PERSIST as 3×ROW_PERSIST makes num_live a no-op for light bodies, exactly as desired).

Apply the same two-term gate everywhere. Re-run model.

## Step 6 — model4 FINAL (317 OK, 126 DIFF) — the implementation spec

The numerical model (model4) is the allocator spec. ONE budget ROW_PERSIST_MAX_BYTES + LIVE_PERSIST=3×.
NO CARRIED_TILE_MAX_BYTES, NO M_COLLAPSE_TILE_BYTES, NO _resident_tile_cap/_carried_*/_pinned_inner —
all subsumed by the group-total footprint (num_live × itemsize × ∏ seated reduction tiles × ∏
materialized features × ∏ resident grid blocks) against ONE budget. 126 DIFFs, all explainable:

DIFF families (all to be RECORDED by the real recorder + BENCHED):
- kl_div(34)+jsd(35): carried-2D rdim 2× (num_live footprint replaces the 16384 carried cap). EXPECTED.
- sum(15)+welford(7)+fused_add_*(5)+gated_rmsnorm(3): grid-M halves — num_live (body_live_tiles)
  now applied UNIFORMLY (was full_width-gated). The principled uniform application (§3). BENCH.
- softmax(7)+scaled_masked_softmax(2)+rms_norm(1)+layer_norm(1) (262144,*): grid-M widens PAST the old
  autotuner_min floor — dropped the SELF-REFERENTIAL resident_tile cap (÷m_block). BENCH (huge-M, safe).
- bias_grad(3)+dyt(3)+group/instance_norm_bwd(2): m-collapse inner re-tile sized by feature footprint
  vs M_COLLAPSE_TILE_BYTES=32768. EXPECTED (the recognizer's inner cap, now a budget outcome).
- grpo(7): multi-grid carried accumulator. grid-M sizing shift.
- cross_entropy(1): (262144,4096) [8]→[4] (num_live).

Floor-vs-resident (rms_norm floor / per_token_group resident) and collapse-vs-widen (bias_grad collapse
vs rms_norm widen) are PURE per-axis MEMBERSHIP outcomes (in_acc / reduced_away) — NO recognizer.
NEXT: implement in helion exactly as model4, delete the recognizers, run the REAL recorder + gates + bench.

## Step 7 — IMPLEMENTED (recognizers DELETED, all correctness gates GREEN)

Rewrote `size_reduction_tiles` into the ONE budget allocator (helion/_compiler/autotuner_heuristics/
triton.py). DELETED (grep-confirmed gone from the base class): `_is_per_feature_accumulator`,
`_grad_collapse_group`, the `inner_tile_ids` post-write override, `_carried_tile_r_block_cap`,
`_carried_m_block_cap`, `_carried_grid_dims`, `_is_carried_reduction_acc`, `_resident_tile_cap`,
`_pinned_inner_resident_elems`, `_build_block_sizes`, `_reduction_rblock`, `_secondary_red_values`,
`_m_block_cap`, `_m_block_product`, `_m_axis_occupancy_cap`, `_full_width_output`,
`_m_collapse_grid_block`, `_m_collapse_inner_byte_cap`, `_m_collapse_resident_elems`. The `if
standard/else` SIZING split is GONE (emission-only); `red_values={} if standard` is GONE. Deleted
constants: CARRIED_TILE_MAX_BYTES, M_COLLAPSE_TILE_BYTES, M_COLLAPSE_MAX_CTA, PERSISTENT_REDUCTION_MAX,
FULL_WIDTH_PERSIST_MAX_ELEMS. KEPT: ROW_PERSIST_MAX_BYTES, LIVE_PERSIST_MAX_BYTES (renamed from
LIVE_PERSIST_BUDGET), LOOPED_CHUNK, MIN_WAVES, NARROW_W1_*, CORESIDENT_MAX_WARPS (num_warps levers,
outside the budget). _TileAllocation lost `m_block`/`warp_override` (no longer needed).

Two implementation bugs found + fixed during bring-up:
1. Persistence byte test must use the RAW extent (size_hint), not np2(size_hint) — a 30522-wide
   cross_entropy row was wrongly judged on 32768 and lost persistence. FIXED (raw_ext for the byte
   tests, np2 for the seated tile width).
2. `persistent` is the OUTCOME `r >= ext` (chunk reached full extent) — covers BOTH the reread floor
   AND a narrow `sum` whose looped chunk simply fits. Not just `ext_held`. FIXED.
Also made the allocator env-independent (reads stored hints; guards env-only reads) so the bare-spec
unit tests run outside the env ctx (matches the old helpers' try/except discipline).

GATES (all GREEN): config recorder 447/447, diff vs pre-edit = 131 changed cells (all explained,
to-be-benched); validate_kernel_fact 460/460; probe_assertions 13/13 (taxonomy routing intact —
floor-vs-resident + collapse fall out of the budget); matmul_layernorm 2p/2s; test_reductions +
test_autotuner_heuristics 41p (updated test_kl_div_wide + replaced the _build_block_sizes test with a
size_reduction_tiles test — new behavior, not contorted). ruff clean.
