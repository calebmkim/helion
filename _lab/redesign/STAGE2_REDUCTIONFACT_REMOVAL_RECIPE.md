# Making the reduction heuristic ReductionFact-FREE — execution recipe

## VERDICT (the question this answers): it is (b) refactoring, NOT (a) a structural blocker.
PROOF: `build_reduction_kernel_fact` runs UNCONDITIONALLY right after `build_reduction_facts` on
every live compile (`device_ir.py` lower_to_device_ir, ~lines 3825 then 3829) and ALWAYS assigns
`spec.reduction_kernel_fact` (even 0 reductions -> empty descriptors). So on the LIVE path the
kernel fact is never None when reduction_facts is populated. Every `kf is None` fallback in the
heuristic is TEST-ONLY (the bare-spec unit tests). Nothing prevents full independence.

Start from a CLEAN tree at the latest port commit (67602efd or later). Branch: reduction-redesign.

## The end state
The reduction heuristic (`_TritonReductionSeedBase` + `TritonStandardReductionHeuristic` +
`TritonUserTiledReductionHeuristic`, triton.py lines ~581-2047) reads a `ReductionDescriptor`
(the primary), NEVER a `ReductionFact`. `ReductionFact` survives ONLY for:
  - the matmul-epilogue heuristic (separate class @2048+; field `reduction: ReductionFact`,
    reads only `reduction.size_hint`; gate `len(spec.reduction_facts)==1`),
  - so it can stay BUILT in device_ir; just unread by the reduction tracks.
Full deletion of ReductionFact is a SEPARATE later step (needs the epilogue ported too) -- NOT
in scope here.

## Why the descriptor suffices (field map -- all verified equal by validate_kernel_fact INV3)
Every `fact.X` the helpers read has a descriptor equivalent:
  fact.size_hint            -> pd.size_hint
  fact.itemsize             -> pd.itemsize
  fact.input_load_itemsize  -> pd.input_load_itemsize
  fact.row_reread           -> pd.row_reread
  fact.reread_eviction_index-> pd.reread_eviction_index
  fact.num_load             -> pd.num_load
  fact.body_live_tiles      -> pd.body_live_tiles
  fact.full_width_output    -> pd.full_width_output
  fact.num_carried_2d_tiles -> pd.carried_2d_count          (count, already the field)
  fact.primary_reduction_block_id -> pd.block_id
  fact.m_block_ids          -> cls._grid_axis_block_ids(spec)  [kf.grid_axis_block_ids]
  fact.non_reduction_loop_block_ids -> cls._non_reduction_loop_ids(spec) [kf field]
  fact.secondary_reduction_block_ids -> the SIZED descriptors minus primary (already in
                                        _secondary_red_values via kf)

## Selector (already present, just adopt it)
`_primary_descriptor_selected(env) -> ReductionDescriptor | None` was drafted then reverted; re-add
it (max ROW-BYTES = size_hint*input_load_itemsize over BACKED sized descriptors; None if no sized
reduction or no kf). It REPLACES `_primary_fact`. `_is_standard_reduction(pd)` becomes
`return pd.category in FULL_EXTENT_CATEGORIES` (one line; no spec needed).

## Transformation (apply ONLY within lines ~581-2047; leave the @2048 epilogue class UNTOUCHED)
1. Every base-helper signature `fact: ReductionFact` -> `pd: ReductionDescriptor`. The 16 helpers:
   _carried_tile_r_block_cap, _carried_leading_dims(no fact), _carried_m_block_cap, _num_warps,
   _resident_tile_cap, _pinned_inner_resident_elems, _m_block_cap, _m_axis_occupancy_cap,
   _m_block_product, _build_block_sizes, _reduction_rblock, _secondary_red_values,
   _m_collapse_grid_block, _m_collapse_inner_byte_cap, _m_collapse_resident_elems, plus the two
   get_seed_config heads' local `fact = _primary_fact(env)` -> `pd = _primary_descriptor_selected(env)`.
2. Body reads: apply the field map above.
3. `_carried_2d_count(spec, rdim)` calls: pass `pd.block_id` (or the relevant rdim). It can read
   `pd.carried_2d_count` directly when the rdim IS the primary; keep the spec-based form for
   OTHER rdims (the carried_m cap iterates accumulator dims, not the primary).
4. Drop the `fact` param + `kf is None` fallback from `_grid_axis_block_ids`,
   `_non_reduction_loop_ids` (they become `(spec)` -> `kf.<field>`; kf is guaranteed present
   because the seed only runs when _primary_descriptor_selected returned non-None, which requires
   kf). KEEP a guard: if a caller can run with kf None (tests), assert kf is not None at the
   get_seed_config entry after the primary-descriptor None-check.
5. _secondary_red_values: drop the legacy `fact.secondary_reduction_block_ids` else-branch.

## Tests to port (test/test_autotuner_heuristics.py)
Two fixtures build a bare ConfigSpec + ReductionFact with NO kernel fact (:702 _reduction_spec,
:3636-area). Port them to ALSO build a ReductionKernelFact with one ReductionDescriptor matching
the fact's fields (category from the spec shape: a reduction_loops entry -> FULL_SLICE; a
block_sizes entry -> USER_TILE). 7 call sites of _reduction_spec / ReductionFact(. The descriptor
is a flat NamedTuple -- construction is no harder than the fact. After porting, the heuristic
under test reads the descriptor; the bare ReductionFact may be dropped from the fixture or kept
for the eligibility gate (`len(reduction_facts)`).

## GATE after EACH logical group (helpers, then heads, then tests, then fallback removal)
- config recorder vs _lab/unify/baseline_fc1dbaa0_configs.json = ONLY the 2 known square movers
  (mreduction/{layer,rms}_norm_bwd/[4096,8192]/fp16). Run from /tmp, PYTHONPATH=worktree.
- validate_kernel_fact.py 460/460; probe_assertions.py 13/13.
- pytest test/test_reductions.py test/test_autotuner_heuristics.py (RUN FROM WORKTREE DIR) -- expect
  52 passed / 22 skipped (may shift as fixtures change; keep green).
- matmul-epilogue MUST-NOT-BREAK: pytest test/test_examples.py -k matmul_layernorm.
- Per user directive: a config MOVE is OK if the read is more faithful AND GPU-measured <10%;
  escalate if a faithful read regresses >=10%. (Expect zero new movers -- these are scalar-identity.)

## Suggested commit slicing (each gated)
a) add _primary_descriptor_selected + _is_standard_reduction(pd); convert the two get_seed_config
   heads to resolve `pd` and pass it (helpers still take fact via pd-shim if needed) -- OR do it
   bottom-up: leaf helpers first. Bottom-up is safer (no half-typed intermediate).
b) convert each leaf helper's signature+body.
c) drop the fallbacks (_grid_axis_block_ids/_non_reduction_loop_ids/_secondary_red_values).
d) port the 2 unit-test fixtures.
e) remove `_primary_fact` (now unused) + the now-unused `ReductionFact` import if the reduction
   path no longer references the type (it will still be imported for the epilogue + device_ir).

## Open follow-ups (independent, noted by the audit)
- rollable / pinned descriptor fields still UNCONSUMED -> delete after this lands (9fb3d9d8 flags it).
- Full ReductionFact deletion (retire from device_ir + matmul-epilogue) -- only AFTER this + an
  epilogue port; tracked separately.
