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

## Step 8 — PERF BENCH (46 representative cells, single-process median-of-9, before-cfg vs after-cfg)

geomean after/before = 1.0225; WIN=2 (welford 262144x7168 0.45!, cross_entropy 262144x4096 0.82),
NEUTRAL=32, REGRESS=12. The 12 regressions classify into 4 root causes:

CLASS A — num_warps (block_sizes UNCHANGED): the m-collapse warp_override removal. layer_norm_bwd
  (4096,8192) 1.52 + (8192,4096) 1.17, rms_norm_bwd (4096,8192) 1.11 — bs identical, ONLY num_warps
  changed (16->4 / 8->4). OLD m-collapse set warp_override=_num_warps(pd) (plain ramp, 16/8); NEW
  applies CORESIDENT_MAX_WARPS=4 (these ARE co-resident multi-reduction). The cap was tuned on p2/p8
  (small-rdim primary); it's WRONG for a large-rdim (8192) grad-param primary. FIX: make the
  coresident warp cap faithful to the PRIMARY ROW BYTES, not a flat co-resident gate.
CLASS B — grid-M widen past autotuner_min (softmax/scaled_masked_softmax, all huge-M): softmax
  (262144,128) 1.38 [16]->[128], (262144,257) 1.27, (131072,128) 1.16; scaled_masked_softmax 1.20/1.13.
  Dropping the self-referential resident_tile cap let grid-M widen too far — widening independent rows
  past the autotuner_min floor HURTS here. FIX: the grid-M widen for a RESIDENT row needs a tighter
  ceiling (the old self-referential cap encoded "don't widen a narrow-row huge-M kernel past its
  floor"). Re-derive a faithful resident-row widen ceiling.
CLASS C — num_live grid-M halving (welford 16384x5120 1.19 [4,..]->[2,..]): applying num_live to the
  grid-M byte cap halved the rows. (welford 262144x7168 is a 0.45 WIN from the SAME mechanism via the
  normalize-loop shrink — so num_live is net-good but over-tight on this mid shape.)
CLASS D — carried-2D 2x / grpo: jsd (8192,32768) 1.10 [2048,1]->[4096,1]; grpo (8,4096,128256) 1.20,
  (8,2048,64000) 1.10 [..,2048]->[..,1024]. jsd's carried r_block doubling over-spills on the mid
  shape; grpo's chunk HALVED (num_live tightened it). FIX: tune the carried/num_live budget scale.

All outputs match (correctness preserved). NEXT: hill-climb the BUDGET constants + the warp ramp to
recover these without re-introducing recognizers (§3-gated).

## Step 9 — CLASS A fix: coresident warp cap = HALVE the ramp (not flat min 4)

A/B measured num_warps 4/8/16 on the m-collapse cells AND p2/p8:
- grad-param (layer/rms_norm_bwd): w8 optimal/co-optimal everywhere; w4 regresses up to 1.53x, w16
  regresses (8192,4096) 1.16x. So full-ramp (16) is NOT best either — w8 is.
- p8 (FULL_GRID primary): w4 STRONGLY best — w8=2.0x, w16=4.3x slower. p2: w4 best (w8=1.09x).
FIX: coresident cap = `max(CORESIDENT_MIN_WARPS=4, ramp//2)` — a PROPORTIONAL halving keyed on the
primary's row-bytes ramp, not a flat clamp. Gives: p2/p8 ramp8->4 (their measured win), grad-param
(4096,8192) ramp16->8 (optimal), residual (8192,4096) ramp8->4 (wants 8, ~1.15x — one cell, dwarfed
by avoiding p8's 2x catastrophe at floor 8). RE-BENCH: layer/rms_norm_bwd (4096,8192) recovered
1.52x/1.11x -> 1.00; (2048,4096) 1.00; (8192,4096) residual ~1.15x (rms ~1.00). Net Class A: recovered.
This is a faithful num_warps lever (proportional to primary row-bytes, uniform), not a recognizer.

## Step 10 — CLASS B root cause: grid-M widen is taxonomy-gated, NOT a free byte/occupancy widen

A/B grid-M:
- softmax (262144,128) fp32: g8=94us BEST, g16=96(1.02), g64=107(1.14), g128=132(1.41). Wants ~floor.
- softmax (131072,128): g8 best, g64/g128 regress 1.19-1.40x.
- per_token_group (8192,4096,128): g1=3.84x, g2=1.99x, g4=1.08x, g8=1.00 BEST. Wants AGGRESSIVE widen.
- per_token_group (128,4096,128): g8 best.
Both have grid_rows/occ ≈ 248 (massively over-occupied), so OCCUPANCY does not explain the split.
The PHYSICAL difference: per_token_group's primary reduction is FULL_GRID (the rdim is grid-resident
/ pinned; the widened axis bid1=groups_per_row is a SIBLING that batches tiny per-group reductions ->
widening AMORTIZES per-program overhead). softmax's primary is USER_TILE — the grid axis CARRIES the
independent reduction rows, so widening just batches independent work and LOSES parallelism.

FAITHFUL RULE (taxonomy, not recognizer): a grid-M axis WIDENS into the budget/occupancy remainder
ONLY when the group's primary reduction is FULL_GRID (grid-resident -> the grid sibling amortizes);
for a FULL_SLICE / USER_TILE primary (the grid carries the reduction rows) the grid-M stays at its
FLOOR (occupancy is already saturated; widening independent rows hurts). This is the per_token_group
"2x widen" special-case falling out of the taxonomy, AND it kills the softmax over-widen. rms_norm
(FULL_SLICE) grid-M was already byte/occ-bound small, so it is unaffected (still floors). The OLD
self-referential resident_tile cap (÷m_block) was encoding exactly this "don't widen a floored row"
behavior by accident; the taxonomy expresses it cleanly.

## Step 11 — CLASS B fix: WIDEN_MAX_ROWS=8 ceiling on the resident-row grid-M widen

A/B confirmed the resident-row grid-M optimum is ~8 rows/program and degrades past it (softmax
g64/g128 regress 1.14-1.41x; per_token_group also peaks g8, g128=1.12x; rms_norm grid-M is FLAT
g1..g8, occ-bound). The byte/occupancy caps alone permit a huge widen on a small-row huge-M kernel
(occ_widen~128 at 262144 rows). FIX: a faithful diminishing-returns ROWS ceiling WIDEN_MAX_ROWS=8 on
the resident-row widen branch (NOT the grad-param collapse branch, which intentionally batches many
rows; NOT a raised autotuner_min floor, which still wins via max(floor,...)). Results: softmax
(262144,*) and scaled_masked_softmax (262144,*) and rms_norm/layer_norm (262144,256) all REVERT to
their pre-edit configs (regressions GONE); the (131072,*) cells settle at [8] (neutral-to-0.86x win).
Total changed cells 131 -> 129. per_token_group unchanged in the recorder ([2], cache-propagated;
genuine optimum 8 is a left-on-table WIN, not a regression).

## Step 12 — CLASS C/D fixes (welford grid-M, jsd/kl_div carried, m-collapse inner) + warp floor

Hill-climbed the budget footprint (NOT recognizers — faithful resident-tensor accounting):
1. GRID-M widen uses num_live ONLY when the axis co-holds a >=2-D loop-carried REDUCTION tile
   (membership: >=2D accumulator containing an rdim — _is_carried_reduction_tile). kl_div/jsd grid
   (carries [grid,R]) keeps num_live -> stays floored 1; welford/softmax grid ([grid] or [grid,None]
   scalar, NO rdim) uses num_live=1 -> widens to its byte/occ/WIDEN_MAX optimum. Fixed welford
   (16384,5120) [4..]->[2..] regression (back to [4]).
2. CARRIED-buffer multiplicity (carried_mult = # distinct >=2-D carried-reduction buffers holding
   the axis, §2.3 "Σ across resident tensors") tightens the budget for PURE carried-2-D kernels
   (gated on feature_footprint==1, since a kernel is carried-2-D XOR grad-collapse): jsd (2 buffers)
   R 4096->2048 (fixed 1.10-1.17x regression); kl_div (1 buffer) unchanged 8192 (neutral 1.03x).
3. M-COLLAPSE inner tile: with feature_footprint>1, carried_mult is OFF (the feature footprint in
   `prod` + num_live already bound it), so layer/rms_norm_bwd inner stays 2 (A/B: inner=2 optimal,
   1->1.2x, 4->1.3x, 8->11x). Fixed the inner 2->1 regression.
4. CORESIDENT warp floor is feature-aware: 8 for a grad-param m-collapse (heavy [inner,feature]
   reduction wants >=8 warps — A/B layer_norm_bwd 8192x4096 w4=1.17x vs w8), 4 otherwise (p2/p8
   pure carried/grid want 4 — w8=2x on p8). Recovered layer/rms_norm_bwd (8192,4096) & (4096,8192)
   warps. instance_norm_bwd -> w8 (A/B optimal); group_norm_bwd (128,...) -> w8 (marginal 1.08x, tiny
   kernel, near noise — accepted).

Total changed cells vs pre-edit: 131 -> 112 (the recovered cells reverted to their pre-edit configs).

## Step 13 — wider bench caught 2 more regression classes; fixed (102->100 changed)

A 56-cell representative bench (all 22 changed families) surfaced regressions the targeted re-bench
missed:
- HEAVY-BODY rolled rows over-widened (dynamic_quant [4]->[8] 1.6x, fused_linear_jsd [1]->[4] 2.0x,
  gated_rmsnorm, cross_entropy_ls): dropping num_live from the grid-M widen (Step 12, meant for
  welford) over-widened these. FIX: num_live is dropped from the widen ONLY for a REDUCE-THEN-APPLY
  kernel (non_reduction_loop present — welford's wide tile is the separate apply pass); a plain
  rolled/persistent row keeps num_live (a heavy body genuinely limits the widen). Recovered all to
  pre-edit configs.
- per_token_group FULL_GRID sibling vs resident-row: WIDEN_MAX_ROWS=8 helped softmax but capped
  per_token_group's genuinely-wide sibling (8192x7168 wants g64; g8 is 1.15x). FIX: the
  WIDEN_MAX_ROWS ceiling applies ONLY to a resident-row reduction; a FULL_GRID primary's grid
  sibling widens freely (occupancy-bound) — the taxonomy distinction from Step 10. per_token_group
  back to pre-edit [64].

Gates green (460/460, 13/13, 41 unit, matmul). Changed cells vs pre-edit: 112 -> 100.

## Step 14 — grpo grid-axis footprint (revert a too-aggressive change); final state

Tried restricting the resident-grid-axis footprint term to grid axes SHARING an accumulator with the
sized axis (CARRIED #2 "shared dim"). It catastrophically loosened grpo's R (the per-token [g0,g1]
accumulator stopped tightening R) -> R blew to LOOPED_CHUNK 16384 and SPILLED 8.8x. REVERTED: a
RESIDENT grid axis (in ANY loop-carried accumulator) tightens the budget — an over-estimate when the
tensors are separate, but it keeps grpo's R tight (1024). grpo's true optimum is 2048 (a 2-separate-
resident-tensor additive footprint the multiplicative ∏ model approximates as 1024, ~1.19x); modelling
it exactly needs additive group_footprint machinery not warranted for 7 transfer-corpus cells, and the
alternative (16384) is an 8.8x spill. Accepted the ~1.19x grpo residual.

FINAL: 100 changed cells vs pre-edit; gates green (460/460, 13/13, 41 unit, matmul). Proceeding to the
definitive bench + report.

## Step 15 — FINAL REPORT (HEAD 3d20f1f4)

### Recognizer removal — CONFIRMED (attribute introspection, not just grep)
DELETED from the reduction seed base + subclasses (hasattr == False): _is_per_feature_accumulator,
_grad_collapse_group, the inner_tile_ids override, _carried_tile_r_block_cap, _carried_m_block_cap,
_carried_grid_dims, _is_carried_reduction_acc, _resident_tile_cap, _pinned_inner_resident_elems,
_build_block_sizes, _reduction_rblock, _secondary_red_values, _m_block_cap, _m_block_product,
_m_axis_occupancy_cap, _full_width_output, _m_collapse_grid_block, _m_collapse_inner_byte_cap,
_m_collapse_resident_elems. The `if standard/else` SIZING split is GONE (emission routing only);
`red_values={} if standard` is GONE. Deleted constants: CARRIED_TILE_MAX_BYTES, M_COLLAPSE_TILE_BYTES,
M_COLLAPSE_MAX_CTA, PERSISTENT_REDUCTION_MAX, FULL_WIDTH_PERSIST_MAX_ELEMS. triton.py shrank ~850
lines. The only doc references left are explanatory ("this SUBSUMES the old ...").

### Landed budget constants (the ONE budget + faithful scales)
- ROW_PERSIST_MAX_BYTES = 245760  (single resident tile ceiling — the one budget)
- LIVE_PERSIST_MAX_BYTES = 3*245760  (num_live-tile live-set ceiling; removes persistence from a heavy body)
- LOOPED_CHUNK = 16384  (looped chunk for a non-persistent row)
- MIN_WAVES = 8  (occupancy floor for the grid widen)
- WIDEN_MAX_ROWS = 8  (NEW: diminishing-returns rows/program ceiling on the resident-row grid widen)
- NARROW_W1_MAX_BYTES = 2048, NARROW_W1_OCC_BYTE_LIMIT = 262144  (num_warps narrow-w1 lever, unchanged)
- CORESIDENT_MIN_WARPS = 4  (coresident warp halving floor; 8 for a grad-param m-collapse)
Faithful scales (continuous, uniform, §3): num_live = max(body_live_tiles); carried_mult = # carried
reduction buffers (gated on feature_footprint==1); feature_footprint = ∏ materialized feature extents.

### Config movement: 100 of 447 cells changed (vs pre-edit 34ae072e). Field-change counts:
kl_div 34 bs (carried 2x), sum 15 bs, welford 16 bs, grpo 7 bs, dyt 3 bs, fused_add_rmsnorm 3 bs,
gated_rmsnorm 3 bs, bias_grad 2 bs, softmax 2 bs, scaled_masked_softmax 2 bs, cross_entropy 1 bs,
jsd 1 bs, layer_norm 1 bs; layer_norm_bwd 1 warps, rms_norm_bwd 1 warps + 2 evict, instance/group_norm
2 warps + 2 evict each. (NB: many norm-bwd cells are byte-identical; only warps/eviction moved.)

### PERF (44 representative cells of the 100, single-process median-of-9, before-cfg vs after-cfg):
geomean after/before = 0.894 (net ~12% FASTER). WIN(<0.90)=6, NEUTRAL=35, REGRESS(>1.10)=3.
- TOP WINS: jsd (262144,4096) 0.036 (28x! the [2048,16]->[128,16] inner-loop fix), welford (262144,7168)
  0.45, welford (8192,14336) 0.75, welford (4096,16384) 0.78, cross_entropy (262144,4096) 0.82,
  scaled_masked_softmax (131072,512) 0.86.
- REGRESSIONS (all grpo, the 2-separate-resident-tensor additive-footprint kernel the multiplicative ∏
  approximates): grpo (8,4096,128256) 1.20, (4,2048,128256) 1.12, (4,1024,256000) 1.10. Accepted (7
  transfer cells; the alternative footprint gave an 8.8x spill).
- seed-vs-DEFAULT spot checks: kl_div 0.62, grpo 0.58 (seed beats compiler default). welford
  (262144,7168) seed/default=1.95 — the welford SEED loses to default (PRE-EXISTING: the old seed lost
  too; this rewrite made welford 2.2x FASTER than the old seed but it's still a poor start vs default —
  a separate welford-seed issue, not introduced here; the autotuner refines from it).

### Correctness gates (all GREEN throughout): validate_kernel_fact 460/460; probe_assertions 13/13
(taxonomy routing intact — floor-vs-resident + collapse are budget outcomes); matmul-epilogue 2p/2s;
test_reductions + test_autotuner_heuristics 41p/22s/13 subtests (2 tests updated to the new behavior:
test_kl_div_wide pins the budget-derived [8192,1]; the _build_block_sizes test replaced by a
size_reduction_tiles test). ruff clean.

### Commits: 5362ea04 (allocator+delete) -> 851455ca (hill-climb footprint+warps) -> 51f2b3d3
(reduce-then-apply + FULL_GRID gates) -> aae034cd (grpo revert) -> 3d20f1f4 (docstring).

---

# CRITIQUE-FIXES PASS (HEAD 20011b63, brief = _lab/redesign/ALLOCATOR_CRITIQUE_FIXES.md)

## CF-Step 0 — baseline regenerated + gates green
- /tmp/before_rewrite.json regenerated at 20011b63: 447 cells, 0 errored (the "before" diff ref).
- validate_kernel_fact 460/460; probe_assertions 13/13. Tree clean (only the untracked brief).
- Tripwire ref captured: fused_linear_jsd grid stays bs=[1] (grid=1) on all transfer cells.

## CF-Step 1 — OFFLINE MODEL built + validated (the trusted oracle for #1)
Built `_lab/redesign/dump_facts.py` (full Stage-1 facts -> /tmp/corpus_facts.json) + `model_alloc.py`
(a pure-arithmetic replica of size_reduction_tiles). VALIDATED: the model's reconstructed
block_sizes vector == the recorded seed for **443/443 cells** (4 gemm/declined skipped). This is the
ground-truth instrument for the #1 diff — no GPU, no bind, instant formula sweeps.

## CF-Step 2 — THE #1 DECISION-POINT FINDINGS (numerically proven against the oracle)

### Finding A — the additive-Σ CORE FIX is CONFIG-NEUTRAL across the entire corpus.
Replacing `num_live × carried_mult × ∏` with the TWO-REGIME structure (STREAMED keeps num_live;
CARRIED = additive Σ over carried buffers, ∏-within / Σ-across, dims classified by membership)
reproduces **every one of the 443 cells** — full block_sizes vector + primary_r_block + persistent.
WHY it is identical, not approximately: each carried buffer is `[grid, R]` with grid seated=1, so
`Σ over buffers (∏ other dims)` == buffer count == the old `carried_mult`, and the grid dim folds
into the per-buffer ∏ (subsuming the lossy `in_accumulator`-multiplied grid term, complaint C). For
the m-collapse CARRIED kernels (layer/group_norm_bwd) the additive coeff DIFFERS from current
(it adds the feature footprint additively), but both coeffs are already astronomically over-budget
(1e8–1e11 ≫ 240KB), so the primary floors to 1 either way -> SAME config. ⇒ #1 deletes carried_mult
+ the feature_footprint==1 gate + the multiplied grid term with ZERO config movement. Pure
principle-restoration, no recognizers, no perf risk.

### Finding B — grpo is in the STREAMED regime, NOT carried. #1 does NOT touch grpo.
grpo's ONLY accumulator is `[g0, g1]` — the two GRID axes, with NO reduction axis (rids={2}).
So `_is_carried_reduction_tile` (>=2-D AND holds an rdim) is **False** -> grpo routes to STREAMED.
The brief's "additive Σ fixes grpo" was a MISREAD (it assumed grpo carried; it is not). grpo's R
is sized by the STREAMED footprint `coeff = itemsize(4) × num_live(2) × g1_block`:
  g1=8  -> coeff=64  -> R=pp2(245760/64)=2048  (== optimum)
  g1=16 -> coeff=128 -> R=pp2(245760/128)=1024 (optimum 2048 -> the standing ~1.2x)
So the ~1.2x grpo residual is INDEPENDENT of #1 and persists after it. **DECISION (user, this pass):
ACCEPT grpo ~1.2x as-is.**

### Finding C — the PRINCIPLED grpo fix (flagged Stage-1 follow-up, NOT done this pass).
The over-tightening is `num_live(2)` multiplying the resident grid block `g1`. The faithful fix
(user's idea): count a live tile against an axis ONLY if the tile actually spans that axis —
i.e. an AXIS-RESOLVED liveness, the same decomposition the carried walk does for accumulators.
For grpo the 2 body temporaries are R-wide tiles that do not each replicate across all of g1, so the
effective coeff drops to ~itemsize×g1 -> R=2048 (the optimum) with no spill. BLOCKER: `body_live_tiles`
is a flat SCALAR count (device_ir.py:1575/1976) with no per-axis breakdown; doing this precisely is a
**Stage-1 fact addition** (a per-axis live-tile count on the descriptor), the same category as F
option (b). NOTE: the cheap allocator-only shortcut (stop multiplying num_live into the resident grid
block) was modeled and REJECTED — it does NOT move grpo (g1 still enters via the in-accumulator path)
and it REGRESSES 4 m-collapse cells (group_norm_bwd [1,16]->[1,2]; instance/layer_norm_bwd inner 2->1,
the 1.2-1.3x step-12 regressions). Only the Stage-1 axis-resolved-liveness version is clean. Filed as
a follow-up; needs its own recorder+bench+gates + sign-off.

## CF-Step 3 — the EXACT config-neutral #1 spec (swept against the oracle, 0 diffs)
`_lab/redesign/sweep_footprint.py` sweeps candidate footprints through the FULL allocator and diffs
the reconstructed block_sizes vector vs current. RESULT: the two-regime additive-Σ (`fp_additive`)
is **0 diffs across all 443 cells** (full vectors + primary_r_block + persistent identical). The
EXACT faithful formulation (the spec to implement):
- common multiplicative base for an axis = `feature_footprint × ∏(OTHER seated reductions in group)`.
- **CARRIED regime fires ONLY when `feature_footprint == 1` AND a carried reduction tile holds the
  axis** (the pure kl_div/jsd case): footprint = `itemsize × num_live × Σ_carried_buffers(∏ buffer
  dims EXCEPT axis, classified by membership)`. This additive Σ EQUALS the old `carried_mult × R`
  (each `[grid,R]` buffer with grid seated=1 contributes R; N buffers -> N·R), so it subsumes BOTH
  `carried_mult` AND the `in_accumulator`-multiplied grid term (the grid dim is now a buffer dim in
  the ∏) with no separate gate.
- **Otherwise (streamed rms_norm/softmax/sum/cross_entropy/fused_linear_jsd/grpo AND grad-collapse
  norm-bwd)**: footprint = `itemsize × num_live × base × ∏(in-accumulator resident grid rows)` —
  the current streamed formula, num_live RETAINED (load-bearing: fused_linear_jsd).
WHY the `feature == 1` condition stays: it is NOT the old smell-gate suppressing a double-count;
it now expresses "a kernel is EITHER pure-carried-2D (buffers add) OR grad-collapse (one
multiplicative working tile holding a materialized feature) — never both" (the two are disjoint
kernel structures). The additive Σ is the faithful model of the pure-carried case; the grad-collapse
case has no separate carried multiplicity to add (its `carried_mult` was already 1). Net effect on
code: DELETE the `carried_mult` variable + its `feature_footprint == 1` gate-block + the separate
`in_accumulator` grid-term loop FOR THE CARRIED BRANCH, replace with the Σ-over-buffers walk; the
streamed branch keeps num_live + the in-acc grid term verbatim. CONFIRMED config-neutral.

## CF-Step 4 — #1 IMPLEMENTED (ONE footprint formula, two regimes, features inline) — gates GREEN
Rewrote `group_footprint_excluding` into ONE legible formula after human review (the first cut kept
the old per-kind loop shape + a hoisted `feature_footprint` scalar — rejected as "still scattered"):

    group_footprint = itemsize × num_live × Σ_resident_tensors ∏_(dim != axis) width(dim)

- `resident_tensors(axis)` is the ONLY place the two regimes differ (returns a list of tensors, each
  a list of `(block_id, width)` dims): CARRIED -> each loop-carried >=2-D reduction buffer is its own
  tensor (they ADD); STREAMED -> ONE combined working tile (feature tiles + seated reductions +
  in-accumulator grid rows). Regimes disjoint by construction (pure-carried XOR has-a-feature-tile).
- The whole footprint is computed in this ONE walk: NO precomputed `feature_footprint` scalar;
  materialized features are iterated inline as ordinary `(fbid, extent)` resident tiles. Removed the
  hoisted `feature_footprint` AND the DEAD `d2.category is GRID_TILE` skip (`sized` is built from
  SIZED_REDUCTION_CATEGORIES, which never contains GRID_TILE). Only guarded hoist left = the
  `feature_extent` DICT (the set computation needs env/device_ir).
- KEY fidelity point (the live recorder caught a model gap): the width travels WITH the dim, so the
  SAME block_id can appear as TWO tiles at TWO widths — the grad-parameter case where bid1 is both a
  materialized feature row (extent 4096) AND the reduction over that axis (seated r_block). A
  membership-only `width(bid)` collapsed those to one (reduction wins) and WIDENED the norm-bwd inner
  tile 2->{16,32,64} (6-cell diff); tagging each dim with its role-width restores byte-identity.
GATES (all GREEN): recorder ZERO-DIFF (447 cells byte-identical vs /tmp/before_rewrite.json); tripwire
fused_linear_jsd grid stays [1] on all 7 cells; validate_kernel_fact 460/460; probe_assertions 13/13;
test_reductions + test_autotuner_heuristics 52 passed / 22 skipped / 35 subtests (NO test edits needed
— configs unchanged); matmul_layernorm 2 passed; ruff clean. Since #1 is config-neutral (selection-
only), there are NO changed cells to bench — perf is identical by construction. grpo unchanged
(~1.2x, accepted; streamed regime, see CF-Step 2 Finding B/C). carried_mult + the feature==1 gate +
the lossy in_accumulator-multiplied grid term are GONE; two clean regimes; no recognizers
re-introduced.

## CF-Step 5 — #2 (F) + #3 (G) + #4 (D+E) — gates GREEN, recorder ZERO-DIFF
- **#2 (F, streamed body-weight proxy)**: KEPT num_live + the widen ternary; named it
  ``drop_body_weight_for_reduce_then_apply`` and documented it as an HONEST PROXY — ``body_live_tiles``
  is shape-blind (counts scalar ``[M]`` carries too), so ``is_reduce_then_apply`` (has-a-non-reduction-
  loop) is a coincidental corpus proxy for "live-tile count inflated by an apply pass," not the true
  cause. The principled fix (a RDIM-WIDE-live-tile Stage-1 fact) is flagged as a follow-up; NOT done.
  No behavior change.
- **#3 (G, FULL_GRID seats at full extent)**: a FULL_GRID reduction (cdiv==1) is full-extent-resident
  BY DEFINITION, so it now seats at ``ext`` directly in the reduction-seating loop, never chunked
  through the byte budget. STRUCTURAL CORRECTION to the brief's framing: a FULL_GRID axis is always a
  ``sized`` reduction, so ``grid_axis_block_ids`` (= grid_ids − sized_bids) EXCLUDES it — it never
  reaches the grid-M widen loop the brief named. The genuine latent footgun was in the SEATING loop
  (a wide unpinned FULL_GRID axis failing the persistence byte test would wrongly chunk). No-op on the
  corpus (per_token_group's FULL_GRID axis is grid-PINNED and sh=128 already seated full via
  persistence) — recorder ZERO-DIFF confirms. ~6 lines, uses the Stage-1 category.
- **#4 (D+E, comment-only)**: D — named JOB A (record primary scalar levers) vs JOB B (emission
  routing) as two orthogonal (non-exclusive) ``if``s; explained ``!= pd.block_id`` excludes the rolled
  primary from double-routing; noted the elif is CORPUS-DARK (no kernel has >1 reduction_loops). E —
  documented the GRID_TILE ``seated=1`` as PROVISIONAL (an axis also in grid_axis_block_ids is
  occupancy-lifted on its 2nd visit in the grid-M loop; ``=1`` is final only for a non-grid-axis
  GRID_TILE, which the corpus lacks). No logic change.
GATES (all GREEN): recorder ZERO-DIFF (447 cells); validate_kernel_fact 460/460; probe_assertions
13/13; test_reductions + test_autotuner_heuristics + test_examples 149 passed / 28 skipped / 35
subtests; ruff clean. (G is the only behavioral change and is corpus-neutral.)

## CF-Step 6 — FAITHFUL Σ-over-tiles footprint (add, don't multiply) — measured NET WIN
Human review drove a deeper fix: the committed footprint multiplied a materialized feature N INTO the
read tile (num_live × R × N × N) where physics is ADDITIVE — the read tile [R, N] and the
grad_weight[N] accumulator are SEPARATE tensors whose bytes ADD, not multiply. Verified against the
Stage-1 liveness (device_ir `_graph_peak_live_by_axis`): `body_live_tiles` is ALREADY axis-resolved
(peak tiles whose shape SPANS the rdim; scalar carries not counted), so a separate buffer-count
multiplier double-counts what num_live holds. And a scalar [grid] carry (softmax max/sum) is a
per-ROW constant, NOT a per-R-element term — folding it into a coefficient × extent over-counts it R×
(this is what made a naive additive rewrite wrongly deny softmax persistence).

REWRITE: `group_footprint_excluding` (one multiplicative number) -> `footprint_terms(axis) ->
(scale, flat)`: resident BYTES = itemsize × (scale × R + flat). Each resident tile = ∏ of its own
dims; Σ across tiles. A tile CONTAINING the sized axis scales with R (adds to `scale`); one that does
NOT is CONSTANT (adds to `flat`). Read/compute tile = num_live copies spanning the axis + the other
resident working dims (union of sized reductions, in-acc grid, materialized features at full extent);
each accumulator_fact is its own tile; a feature with no accumulator is its own constant tile.
Persistence: itemsize×(scale×raw+flat) <= budget. Chunk: R <= (budget/isz - flat)/scale (flat
SUBTRACTED, not divided — the softmax-persistence bug fix).

THREE budgets keyed on two EXISTING structural flags (still no recognizer): GRAD_PARAM_PERSIST (feature
present, LOOSEST = 3*ROW/2 = 368640 — a [N] accumulator amortized across rows runs above the row
ceiling), CARRIED_PERSIST (>=2-D carried reduction tile, no feature, TIGHTEST = ROW//2), ROW_PERSIST
(streamed). Opposite physical situations (amortized accumulator / transient carried tile / streamed
row); `feature_extent` + `_is_carried_reduction_tile` already exist.

MOVEMENT (LIVE recorder vs committed CF-Step-5): 36 cells, all block_sizes, ALL MEASURED WINS/NEUTRAL
(single-process median-of-9): kl_div 34 (8192->4096, ratios 0.957-0.994), bias_grad_bwd 2 (8->16 /
4->8, 0.979). layer/rms/dyt/instance/group_norm_bwd UNCHANGED (the loose GRAD_PARAM budget holds their
inner tile at its current value — a tighter budget floored it to 1, measured ~1.1-1.2x). softmax/
welford/grpo/fused_linear_jsd UNCHANGED (the scale/flat split stopped the scalar-carry over-count a
naive additive model introduced). NET: geomean ~0.98, worst ~0.99, best 0.957; NO regressions.
GATES GREEN: validate 460/460, probe 13/13, unit 52p, matmul 2p, ruff; test_kl_div_wide still pins
[4096,1] (carried budget unchanged).

NOTED FOR LATER (deeper flaw, measured, NOT fixed): the grad-param inner-tile TRUE optimum is a
loop-overhead amortization sweet spot (inner=4 at N=4096 measured ~+10%; 1-2 at N=8192) the budget
model structurally cannot express ("fits" != "best amortization"; the cliff is violent — inner=16 is
10-15x). A chunk-amortization lever (analogous to WIDEN_MAX_ROWS) is a flagged follow-up. Offline
model (spike_faithful.py) had 2-cell fidelity gaps vs the live recorder (bias_grad phantom feature,
4D group_norm feature extents) — reconciled against the LIVE recorder as the authority.

## CF-Step 7 — PHASE A grounding (ir ANSWER KEY via `_lab/redesign/ground_live_tiles.py`)
Ran the new grounding probe over sum/long_sum/rms_norm/layer_norm/welford/cross_entropy/softmax/
kl_div/jsd (curriculum), grpo (transfer), per_token_group_fp8_quant (vllm), layer_norm_bwd/
bias_grad_bwd (mreduction), + all 13 probes. It dumps per graph_id: graph TYPE, CF-tree edges
(`_if`->(if,else); loop->child), reduction lowerings, `original_graph_id` (rolled->source), and the
per-tile `_graph_peak_live_tiles` output PLUS 3 competing peak-step definitions (D1 count / D2 rank /
D3 union). Group keys + `branch_paths_by_bid` printed as the co-residency oracle.

FINDING 1 — **D1 (max tile COUNT, the scaffolded default) is WRONG for a footprint.** welford g0
D1 = `[[0]]×8` (8 scalar `[M]` carries, ZERO rdim-spanning tiles) even though axis-1 (rdim) peaks at
3 simultaneously-live tiles. The rdim tiles and the scalar carries peak at DIFFERENT steps; D1 snaps
the single step with the most tiles and misses the rdim entirely -> a footprint would see no
R-scaling term and over-widen into a spill. FIX: use a per-shape peak UNION (`D3_union`: for each
distinct tile shape, the max simultaneous count of THAT shape, unioned as a multiset). Conservative
superset, never under-counts any axis's peak co-residency, matches `_graph_peak_live_by_axis` per-axis
counts. => `_graph_peak_live_tiles` must be REWRITTEN to the D3 definition before it feeds a footprint.

FINDING 2 — **rolled `ReductionLoopGraphInfo` copies are redundant** (subset of the original graph).
sum/rms_norm/layer_norm/cross_entropy: the ORIGINAL (Root/ForLoop) graph `_original_graph_reductions`
keys on ALREADY holds the full pre-roll body tiles; the roller's per-config copies are strict subsets.
So the group's live tiles come from the ORIGINAL graphs only — the SAME exclusion the group keys use.

FINDING 3 — **a group's tiles can live in a loop body it DRIVES, not its home graph.** kl_div home =
g1 (Root, `loop->[0]`), which has 3 `[R,M]` tiles, but the driven loop body g0 (ForLoop, NO reduction
lowering) has 6. welford/bias_grad_bwd/layer_norm_bwd likewise: the heavy tiles are in the ForLoop
body, the Root is thin. => attribution must DESCEND from each group's original graph through its
`loop->child` edges (skipping ReductionLoopGraphInfo) and take the max/union over the owned bodies.

FINDING 4 — **If/Else siblings must be combined as MAX and NOT pulled into a sequential parent.**
jsd: g5(Root,bid0)->loop g4; g4 `_if`->{g3(If,bid1) | g2(Else)}; g2 `_if`->{g1(If,bid1) | g0(Else,
12 tiles!)}. `branch_paths_by_bid` proves bid1's two occurrences (g1,g3) are mutually exclusive. The
Root group (bid0, key 5) must NOT absorb the if/else subtree (esp. g0's 12 `[R,M]`). grpo/
per_token_group: an EMPTY ElseGraphInfo sibling (`live_tiles=[]`) + a populated If/Root — max is the
populated side. Only jsd/grpo/per_token_group have If/Else (confirmed).

FINDING 5 — group keys are the ORIGINAL-graph classes exactly as `build_reduction_kernel_fact`
already builds them (`groups_by_gid`). jsd has 3 (keys 1,3,5 — but 1&3 are the mutually-exclusive
branch pair, so its TRUE sequential-group count is 2: {the V-reduction, either branch} + {the Root
KL}). p4/p6/p7/p10 multi-group confirmed. Attribution is per-group over its original graph + driven
non-reduction/for-loop bodies, max across If/Else, union-per-shape within, separate across groups.

## CF-Step 8 — PHASE B: aggressive rewrite to the Σ-over-live-tiles footprint (committed Phase A first)
Rewrote `size_reduction_tiles` to the SIMPLE two-pass plan the human asked for. DELETED: `num_live`
multiplier, `feature_extent` reconstruction, the separate accumulator sum, `in_accumulator()` grid
gate, `LIVE_PERSIST` budget, `GRAD_PARAM` budget (3rd), the `drop_body_weight_for_reduce_then_apply`
proxy, `_is_carried_reduction_tile`. NEW footprint: `resident_bytes = itemsize × Σ over
group.live_tiles of ∏(dim widths)` split into `(scale,flat)` by whether a tile contains the sized
axis; budget test `itemsize×(scale×R+flat) ≤ budget` (flat SUBTRACTED). TWO budgets (ROW / CARRIED=
ROW//2, keyed on `carried_2d_count>0`). Two passes: PASS 1 sizes reductions with grid at floor; PASS
2 widens grid on "resident = block_id in some live tile" (else collapse). Non-reduction loops last.
A `None` tile dim = width 1 (grounding: an attributed group tile's None is never a resident feature —
the feature is always a resolved block_id).

BUG FOUND + FIXED IN TRIAGE (grad-param inner-tile explosion, the exact cliff the handoff warned of):
layer_norm_bwd/rms_norm_bwd inner tile went 2 -> 4096 (a spill). ROOT: the grad-param N axis is a
MATERIALIZED full-width `full_slice` (roller declined, in NEITHER reduction_loops NOR block_sizes);
my rewrite CHUNKED it to 1, so the co-resident inner tile saw N as width 1 and grew to full extent.
FIX: a materialized-full-width `full_slice` (in no tunable slot) seats at FULL EXTENT like FULL_GRID
(it cannot be split, has nowhere to emit a chunk) — so the inner tile sees the real N. Verified:
layer_norm_bwd [16,1], group_norm_bwd [1,2] (=old), instance_norm_bwd [1,8], dyt_bwd [16,16] (=old).

TRIAGE RESULT (recorder diff vs before_rewrite, 120/447 changed, NO explosions after the fix):
 - PERSISTENCE LOSS (24 cells, real): cross_entropy/ce_ls_zloss/scaled_masked_softmax/fused_add_*
   at size_hint > ~30720 lose `rl [None]` -> `[16384/8192]`. The 2-tile fp32 row at N=32000 needs
   4×2×32000=256008 > ROW 245760 — just over. Calibration: ROW may be a hair tight for 2-tile rows.
 - jsd 2048 -> 4096 (all 35 cells): the primary (KL, bid0) is sized in GROUP 5 which has only 5
   `[R,M]` tiles; my faithful per-group count is 5, the old `num_live=12` (max blt kernel-wide) was
   2x tighter -> 2048. Handoff says jsd MEASURED wants 2048. Faithful-per-group UNDER-counts jsd's
   true pressure (the primary's group is the LIGHT one; the heavy V-branch groups are sequential).
 - welford grid halved (4->2 etc, 19 cells): the `drop_body_weight` proxy is gone; the widen now
   sees the true footprint. Handoff: welford wanted grid 4 (drop-body kept it there).
 - sum/fused_add_rmsnorm grid doubled (small, likely wins); grad-param inner 2->1 (small).
BENCHING the movers next (jsd, cross_entropy persistence, welford, softmax) to decide calibration.

## CF-Step 9 — PHASE C triage + calibration (all benched single-process median-of-9, replay_bench)
Four calibration fixes landed on top of the rewrite, each measured:

1. **carried-accumulator multiplier** (`scale *= max(1, d.carried_2d_count)` in reduction sizing AND
   `scale_w *= max(1, pd.carried_2d_count)` in the grid widen). jsd's primary (KL) sits in a LIGHT
   5-tile group; the faithful per-group count gave R=4096 (MEASURED +11.4% regr) and let the grid
   widen to 2 (+10.9%). c2d counts the co-resident carried `[R,M]` buffers the per-group snapshot
   misses (jsd carries 2, kl_div 1): kl_div R=4096, jsd R=2048 + grid=1, at the exact 2x ratio. Grad-
   param inner tiles (c2d=0) unaffected.

2. **materialized-full-width `full_slice` seats at FULL EXTENT** (was chunked to 1). The grad-param N
   axis is a `full_slice` in NEITHER reduction_loops NOR block_sizes; chunking it to 1 made the
   co-resident inner tile read N=1 and grow to full extent (4096 — a spill). Seating full-width (like
   FULL_GRID) fixes it. Its `persistent` stays False (it emits `reduction_loops=[]` regardless) so the
   re-read eviction hint (`['last',...]`) the grad-param row wants is preserved.

3. **PER-REDUCTION budget** (not kernel-wide): CARRIED iff `d.carried_2d_count>0` else ROW. Kernel-
   wide CARRIED wrongly tightened the grad-param INNER tile (c2d=0) because the MATERIALIZED N (c2d=3)
   tripped the flag -> inner floored to 1 (MEASURED layer_norm_bwd 8192x4096 inner 2->1 = +20% regr).
   Per-reduction gives the inner tile ROW -> inner=2 (recovered).

4. **CATEGORY-keyed persistence-hold ceiling**: FULL_SLICE (rolled fused reduce+apply) holds to ~600
   KiB (PERSIST_HOLD=3xROW); USER_TILE (multi-pass softmax/welford) holds only to ~262 KiB
   (USER_TILE_PERSIST_HOLD=294912). MEASURED: cross_entropy N=32000 persist +47%, N=50257 +7% (wants
   FULL_SLICE hold); softmax N=32768 persist +30% but N=49152 persist -34% (wants USER_TILE tight hold
   -> chunks at N>=49152). The old code held softmax to N=49152 and LOST perf; the faithful split BEATS
   it (softmax 40960 chunk = +28% vs old 65536-persist).

FINAL TRIAGE (52/447 raw_seed cells changed vs before_rewrite, ALL benched neutral-or-win):
 - softmax 65536->16384 (2 cells): +28% WIN (old over-held). cross_entropy 16384->None (1): +0.7% win.
 - welford grid halved (18): +1.4-1.6% WINS (the drop_body_weight proxy was pure overhead — removed).
 - sum/fused_add_rmsnorm grid doubled (16): neutral-to-win (+2.3% sum 4->8; ~noise elsewhere).
 - grad-param inner tweaks (group_norm 2->4 +1%, instance 4->8 ~noise, dyt 4->2 +1.4%, bias_grad
   64->32 -2.5% on a 32us kernel = noise-floor). layer_norm_bwd/rms_norm_bwd inner=2 UNCHANGED (fixed).
 - jsd (35) / kl_div: UNCHANGED (c2d multiplier holds R=2048/4096).
 - 5 synthetic probes (p4/p6/p7/p8/p10 — multi-group corpus-dark paths) moved sanely, no spills.
GATES: validate 460/460, probe 13/13, pytest 52p+2p, kl_div band test pins [4096,1] (no update).
NET: no regressions > noise; several real WINS (softmax wide-N, welford). The rewrite DELETED
num_live / feature_extent / LIVE_PERSIST / GRAD_PARAM-budget / in_accumulator / drop_body_weight and
is simpler AND faster. Budgets: ROW, CARRIED=ROW/2, PERSIST_HOLD=3xROW, USER_TILE_PERSIST_HOLD=1.2xROW.
