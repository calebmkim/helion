# Codegen-Knob Seeding Map (extend the reduction seed beyond the 4-lever vocabulary)

> Grounded by code-investigator (2026-05-29). The v6 seed emits only block_sizes/reduction_loops/
> num_warps/num_stages. The small-N/grid-bound Product-A headroom is codegen-knob-bound. Oracle field-diffs:
> sum (8192,256) G=1.58 via pid=persistent_interleaved+num_sm_multiplier=2+maxnreg=128; rms_norm/softmax/
> cross_entropy small-N +15-19% via indexing/eviction/pid. Anchors in `helion/autotuner/config_spec.py`
> (active tree = wt-reduction). Line numbers approximate.

## Knobs
- **pid_type** (`VALID_PID_TYPES` ~L200): `('flat','xyz','persistent_blocked','persistent_interleaved')`, default `flat`.
  Codegen `tile_strategy.select_pid_strategy` -> ProgramIDs subclass. persistent_* launch grid=num_sms(*mult)
  and loop virtual PIDs -> amortize launch/tail + occupancy for GRID-BOUND (large M / small rnumel / many CTAs).
  GATES the other two (num_sm_multiplier/maxnreg valid ONLY with persistent_*; normalize RAISES for flat).
- **num_sm_multiplier** (~L206): pow2 in [1,128], default 1. grid = num_sms*mult for persistent. persistent-only.
- **maxnreg** (`VALID_MAXNREG=(None,32,64,128,256)` ~L215, default None): per-thread reg cap, CUDA-only,
  persistent-only, COUPLED to num_warps (cap = regfile/(warps*32)). DEFER (most coupled, least attributable).
- **indexing** (per-memory-op LIST, length = num_load+num_store): `('pointer','tensor_descriptor')` on H100
  (TMA), default all `pointer`. tensor_descriptor for WIDE CONTIGUOUS loads. `spec.indexing.length` available at
  seed time; `spec.store_indices` marks store slots. NOT length-validated by normalize -> build at EXACT length.
- **load_eviction_policies** (per-LOAD LIST, length = un-annotated loads): `('','first','last')` Triton, default
  `''`. `first`=evict_first for single-pass streaming (num_load==1); `last`=evict_last for reused operands.
  DIFFERENT/smaller index space than indexing (loads only). NOT length-validated -> build at exact length.

## normalize() behavior (failure modes)
- pid_type/num_sm_multiplier/maxnreg: validated; invalid -> InvalidConfig. num_sm_multiplier/maxnreg with a
  flat/xyz pid_type -> RAISE (or silently dropped under _fix_invalid). maxnreg capped vs num_warps budget.
  => a seed setting num_sm_multiplier/maxnreg MUST also set a persistent pid_type.
- indexing/load_eviction_policies: NOT length-validated. Wrong-length = silent (codegen reads positionally with
  bounds guards). Invalid option string -> caught only at flat-encode (ValueError), NOT at seed normalize.
  => build these lists at EXACTLY `spec.<field>.length` with only valid choices. Read length off the LIVE spec
  (cute.py:248 precedent: `["tensor_descriptor"]*spec.indexing.length`).

## Workload-keying + regime-conflict risk (the v3 lesson: a lever that helps small-N may HURT large-rnumel)
- pid=persistent_* : grid-bound (m_extent >> num_sms AND small rnumel). RISK: can hurt large-rnumel (already
  SM-saturated). MUST matched-A/B across small-N AND large-rnumel.
- num_sm_multiplier : rider on persistent when m_extent >> num_sms. pow2. couples to maxnreg/num_warps.
- indexing=tensor_descriptor : wide contiguous load slot only (not scalar store). per-op correctness (epilogue
  subtiling forces store tensor_descriptor; TMA needs alignment guards -> silent fallback if wrong).
- eviction=first : single-pass streamed load (num_load==1). per-slot for multi-load.

## PRIORITIZED PLAN
1. **STAGE 1 (biggest, cleanest): pid_type=persistent_interleaved + num_sm_multiplier** for grid-bound small-N.
   New branch in get_seed_config (persistent-only): if `fact.size_hint <= SMALL_RNUMEL AND m_extent >=
   GRID_BOUND_MIN*num_sms`: pid_type='persistent_interleaved', num_sm_multiplier=2. Set SMALL_RNUMEL/
   GRID_BOUND_MIN by a breakpoint sweep (mirror STREAM_WARPS32_MIN_ELEMS discipline). num_sms via
   helion.runtime.get_num_sm (H100=132) or a dimensionless m_extent/size_hint ratio. Emit pid_type +
   num_sm_multiplier TOGETHER (persistent-gated). HOLD maxnreg.
2. **STAGE 2 (broad): indexing + load_eviction_policies** (+15-19%). Build at live spec lengths; tensor_descriptor
   on wide-load slot(s); eviction=first for num_load==1 streamed load. Needs a small ReductionFact extension or
   spec.store_indices to target the big load vs the store.
3. **DEFER: maxnreg** (coupled to num_warps; only after Stage 1, with a maxnreg x num_warps matched grid).

## REQUIRED A/B / no-regression (matched-lever)
- pid_type lever isolation: flat vs persistent_interleaved(+mult), ALL other levers matched, across grid-bound
  small-N (sum (8192,256) + held-out small-N) AND large-rnumel (long_sum, wide rms_norm) -> prove the win AND
  no large-rnumel regression (the v3 failure mode).
- num_sm_multiplier sweep {1,2,4} at fixed persistent pid_type (confirm 2 generalizes, not a single-shape artifact).
- inert-proof: every in-sample shape NOT taking the new branch -> byte-identical seed.
- Validate vs the FULL verbatim oracle (lever-isolation guard), never an isolated re-paired lever.
