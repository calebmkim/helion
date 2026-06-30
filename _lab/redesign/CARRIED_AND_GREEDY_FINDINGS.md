# Carried-tile + greedy-allocator findings (audit round 2)

Branch `reduction-redesign`. These are VERIFIED design flaws found auditing the post-port
heuristic. #1 is fixed (commit 96cd5c18). #2/#3 are the same root flaw; #4 is the big
re-architecture. None moves the current corpus (all stay at the 2 known square movers) -- they are
FAITHFULNESS/generality bugs latent behind the corpus's mostly-single-reduction shape.

## Evidence (every >=2D carried accumulator in the corpus; (kernel, rank, dim_block_ids, last_is_rdim, first_is_grid))
- 2-D [M,R] with rdim LAST: kl_div (1,0), jsd (1,0).
- 2-D [M,None] no rdim (per-row scalar accum): rms_norm/layer_norm/welford/... -- leading=grid, no rdim.
- grpo (0,1): BOTH dims grid, NO rdim.  <-- #2 witness
- rms_norm_per_block_quant (0,None,None): RANK 3.   <-- #3 rank witness
- group_norm_bwd: RANK 3 and RANK 4, e.g. (2,3,3,1); multiple rdim positions.  <-- #3 witness
- instance_norm_bwd (2,1,None) / (None,1,None): rdim NOT at [-1].  <-- #3 (a) witness
- rms_norm_bwd (1,2), layer_norm_bwd (2,1): rdim NOT necessarily leading/last.

## #1 DONE (commit 96cd5c18)
`_triton_reduction_eligible` kf-None branch dropped the `len(reduction_facts)==1` fallback ->
`return False`. Pure kernel-fact gate, no ReductionFact read. Tests green, config unchanged.

## #2 `_carried_leading_dims` is too narrow (leading-dim only)
CURRENT: returns `{a.dim_block_ids[0]}` for each carried >=2D accumulator -- only the LEADING dim
is treated as "co-holds the carried tile when widened".
FLAW: grpo's carried tile (0,1) has TWO grid dims; only bid0 is returned, so widening bid1 would
multiply the tile but isn't capped. The leading-dim rule is a PROXY reverse-engineered from the
`fullgrid_plus_carried2d` adversarial kernel (where the FULL_GRID axis G was NOT in the carried
tile at all, so "non-leading" happened to exclude it).
FAITHFUL RULE: a grid axis co-holds the carried tile iff it appears ANYWHERE in that accumulator's
`dim_block_ids` (it is a tiled dim of the resident tensor) -- NOT just at index 0. Strict superset
that still excludes the adversarial G (not in the tile) and now correctly includes grpo's bid1.
EXPECTED CONFIG IMPACT: none on the corpus (grpo's tile has no rdim so the carried cap doesn't fire
there) -> verify zero-diff and land as a faithfulness fix.

## #3 `_carried_m_block_cap` assumes a 2-D [leading_M, last_R] tile (multiple bugs)
CURRENT (lines ~684-737): iterates accumulator_facts; gates on `a.dim_block_ids[0]==m_axis_block_id`;
takes `rdim = a.dim_block_ids[-1]`; `total_r += max(1, r_block)`; budget = CARRIED_TILE_MAX_BYTES //
(total_r * itemsize).
FLAWS (all falsified by the evidence above):
 (a) `rdim = dim_block_ids[-1]` -- the rdim is NOT always last (instance_norm_bwd (2,1,None);
     group_norm_bwd (2,3,3,1)). FIX: classify each dim by MEMBERSHIP -- is `d` in the kernel fact's
     reduction block_ids? -- not by position.
 (b) gate `dim_block_ids[0]==m_axis_block_id` -- the M need not be leading (same as #2). FIX:
     `m_axis_block_id in a.dim_block_ids`.
 (c) `total_r += r_block` (ADD) is only right for SEPARATE 2-D [M,R_i] buffers stacked (footprint
     M*(R_1+..+R_N)). It is WRONG for a single >=3-D tile [M,A,B] whose footprint is the PRODUCT
     M*A*B (group_norm_bwd rank-4, rms_norm_per_block_quant rank-3). FAITHFUL: per accumulator,
     footprint contribution = PRODUCT of its non-M, non-(this-axis) dims; SUM across separate
     accumulator buffers. i.e. M_BLOCK * Σ_buffers(∏ other-dims) * itemsize <= budget.
 (d) ROOT: the function models the carried tile as exactly [M_BLOCK, R_BLOCK] -- assumes the only
     dims are one M and one R. Real carried tiles are rank 2-4 with arbitrary rdim/grid/other
     positions. It is fitted to the 2-D adversarial witnesses, not general (the "don't assume
     rank<=2" smell, PROMPT §2.1/p5).
FIX SHAPE: one helper that, given an accumulator and the axis being sized, computes
`∏ (extent of each OTHER tiled dim)` classifying dims by membership (rdim -> its sized r_block /
extent; grid M -> its block; other -> its extent), then sums across buffers carrying that axis.
EXPECTED CONFIG IMPACT: likely none on corpus (the multi-dim carriers are grad-collapse-overridden
or don't hit this cap) -- verify zero-diff; it's a generality fix.

## #4 (THE BIG ONE) `_reduction_rblock` + `_build_block_sizes` should be ONE greedy-budget allocator
USER'S MODEL (PROMPT §2.3): per CO-RESIDENCY GROUP: (1) form ONE budget from the floor/caps;
(2) greedily ASSIGN in priority order -- full-extent reductions, then user-tile reductions, then
grid-tile -- then the grid m_block takes the REMAINDER; (3) move to the next co-resident group,
seeing the already-assigned tiles as FIXED/pinned; (4) finally the non-reduction loops, also
against what's left. After each assignment more tiles are fixed -- that's expected; assign what you
can.

CURRENT REALITY (the ad-hoc/scattered version the user objects to):
 - `_reduction_rblock` sizes the reduction CHUNK in isolation (its own byte formula).
 - `_build_block_sizes` SEPARATELY sizes the M/grid axes, reading `r_block_resident` BACK to cap
   them, then sizes non-reduction loops in a third local pass.
 - Sizing is split across functions, each LOCALLY greedy, with r_block/m_block/loop values threaded
   between them as separate steps. There is NO single per-group budget; "the budget" and "the
   r_block decision" only collapse together because the corpus is mostly SINGLE-reduction
   (one group). It is structurally wrong for genuine multi-group / multi-co-resident kernels.
 - The user is explicit: "you really shouldn't have both the r_block and assign block sizes; they
   should be unified into one function."

TARGET: a single `size_reduction_tiles(kernel_fact, spec, env) -> {block_id: size}` (PROMPT §2.3
"ONE function assigns ALL block sizes", §6.2.1) that:
  - iterates co-residency groups in priority order (most/heaviest full-extent first -- the
    cross-group ordering, §2.3 step 3);
  - within a group, forms the budget (clamp(byte/occupancy ceiling, persistence/reuse floor)) and
    bids reductions in priority order, consuming the budget via the group_footprint rule
    (Σ distinct resident tensors, ∏ tiled dims; shared dim divides, sibling tensor adds -- §2.3);
  - holds earlier groups' assignments FIXED;
  - sizes non-reduction loops last, against the remaining headroom.
This SUBSUMES `_reduction_rblock`, `_build_block_sizes`, the M-axis cap loop, AND fixes #2/#3 for
free (the group_footprint product/sum rule is exactly the carried-tile footprint done right).

RISK/METHOD: large re-architecture of the byte-identical sizing path. MUST be done as a zero-diff
refactor first (reproduce the corpus + the 2 known movers via the unified allocator), gated by the
config recorder at each step, THEN the generality fixes (#2/#3) fall out. This is its own multi-
session task -- do NOT attempt as a single edit. The `Cap`/`size_axis` primitive is the right
building block and should be kept; what changes is that ONE allocator drives all axes of a group
from one budget, instead of r_block-then-m_block-then-loops in separate functions.

## Gates (verbatim) for any of the above
config recorder vs `_lab/unify/baseline_fc1dbaa0_configs.json` = ONLY the 2 known movers
(mreduction/{layer,rms}_norm_bwd/[4096,8192]/fp16). 460/460 validate_kernel_fact; 13/13
probe_assertions; 52p/22s test_reductions+test_autotuner_heuristics (FROM WORKTREE DIR);
test_examples -k matmul_layernorm green. HELION_AUTOTUNE_EFFORT=none, no GPU needed for the
config-equivalence work; GPU only if a faithful change MOVES a config (then measure <10%, escalate
if not -- user directive).
