# Step 2 Code Map — ReductionFact + Triton reduction seed-heuristic

> Grounded map from the code-investigator (2026-05-28). Line numbers approximate — grep symbols.
> Worktree root: `/home/calebkim/helion-new-heuristics/wt-reduction`.

## The heuristic template to clone: `helion/_compiler/autotuner_heuristics/cute.py`
- **Eligibility gate** `_reduction_kernel_eligible(env, device_ir)` (~L28-38): requires
  `len(spec.block_sizes)==1 and len(spec.reduction_loops)==1`, `not spec.matmul_facts` (excludes GEMM),
  and `max(bs_spec.min_size, bs_spec.autotuner_min) <= 1` (M-axis accepts block_size=1).
  NOTE: `len(reduction_loops)==1` matches T1 (rollable rdim) ONLY — T2 manual-tile reductions are a
  `block_sizes` entry with `reduction=True`, NOT in reduction_loops, so this gate must be BROADENED for
  T2 kernels (softmax_two_pass, kl_div, jsd) later.
- `CuteReductionTileHeuristic` (~L83-124), `name="cute_reduction_tile"`, `backend="cute"`. In
  `get_seed_config`: `rl_spec = spec.reduction_loops[0]`; `size_hint = rl_spec.size_hint`;
  if `size_hint <= max_threads`: `reduction_loops=[None]` (persistent) else `[max_threads]`. Seed sets
  `block_sizes=[1]`, `num_threads=[1]` (CuTe-only — DROP), `reduction_loops`, optional
  `cute_vector_widths` (CuTe-only — DROP). Does NOT set num_warps/num_stages.
- `CuteReductionWideChunkHeuristic` (~L127-175): same but `reduction_loops=[chunk]`,
  `chunk=max(size_hint//2, max_threads)`, gated on `size_hint > 2*max_threads`.

## Registration: `helion/_compiler/autotuner_heuristics/__init__.py`
- `HEURISTICS_BY_BACKEND` (~L19-26): `"triton": (TritonSkinnyGemmHeuristic,)` — ONLY GEMM today.
- `compiler_seed_configs(env, device_ir)` (~L35-62): for each heuristic in `get_heuristics(backend)`,
  if `is_eligible`: `config = get_seed_config(...)`; append; record `name` in
  `config_spec.autotuner_heuristics`. Per-heuristic exceptions swallowed. Returns `dedupe_configs`.
- Invoked at `helion/runtime/kernel.py:~508`:
  `self.env.config_spec.compiler_seed_configs = compiler_seed_configs(env, host_function.device_ir)`
  — runs AFTER device-IR lowering (after register_rollable_reductions). Seeds consumed by autotuner at
  `autotuner/config_generation.py:~417` and `autotuner/base_search.py:~853`.
- TO ADD: import new class (~L10), add to the triton tuple (~L25):
  `"triton": (TritonSkinnyGemmHeuristic, TritonReductionHeuristic),`.

## Base interface: `helion/_compiler/autotuner_heuristics/registry.py` (~L12-29)
```python
class AutotunerHeuristic:
    name: ClassVar[str]; backend: ClassVar[str]
    @classmethod
    def is_eligible(cls, env, device_ir) -> bool: ...
    @classmethod
    def get_seed_config(cls, env, device_ir) -> Config | None: ...
```
Receives `env` (`env.config_spec`: matmul_facts, reduction_loops, block_sizes, max_reduction_threads;
`env.backend_name`) and `device_ir` (`device_ir.graphs`). Returns `helion.runtime.config.Config` or None.

## ReductionFact template: `MatmulFact` in `helion/autotuner/config_spec.py`
- `MatmulFact(NamedTuple)` (~L73-85): lhs_ndim, rhs_ndim, m/n/k_block_id, static_m/n/k, lhs/rhs_dtype.
- Stored: `self.matmul_facts: list[MatmulFact] = []` in `ConfigSpec.__init__` (~L327).
- Populated: `helion/language/matmul_ops.py:~336-349` during op tracing (static_* from
  `static_problem_extent()`).
- PRECEDENT for per-rdim metadata on spec: `self.cute_indexed_reduction_block_ids: set[int]` (~L265,
  set at device_ir.py:~880).
- DEFINE `ReductionFact(NamedTuple)` after MatmulFact (~L86); store
  `self.reduction_facts: list[ReductionFact] = []` next to matmul_facts (~L327).

## Populate point: `helion/_compiler/device_ir.py:register_rollable_reductions` (~L756-880)
- `rdims = [bs for bs in env.block_sizes if bs.reduction]` (~L766).
- Per rdim runs `ReductionRoller`; excludes matmul/stack-coupled rdims (`has_matmul_with_rdim`,
  `has_stack_tensor_with_rdim`). 2nd pass (~L819-843) appends `ReductionLoopSpec(block_id, size_hint)`.
- HOOK fact-population in the 2nd-pass loop (~L822-843), reading: `rdim.block_id`, `rdim.size_hint()`,
  non-reduction `block_sizes` ids (`[bs for bs in env.block_sizes if not bs.reduction]`), dtype from
  `graph_info.graph.nodes[*].meta["val"]` (node-walk pattern at ~L856-880), roller op/load counts
  (`roller.inner_count`/`outer_count` from `roll_reduction.py:~59-92`, `is_nontrivial` ~L466).

## Static T1/T2/out-of-scope signals (from env.config_spec)
- **T1** (rollable rdim): `block_id in spec.reduction_loops.valid_block_ids()`; lookup
  `spec.reduction_loops.block_id_lookup(block_id)` -> ReductionLoopSpec(.size_hint,.block_id). Per-config
  value: `spec.reduction_loops.config_get(config.reduction_loops, block_id, None)` (None=persistent).
- **T2** (user hl.tile reduction): a `block_sizes` entry; `bs_spec =
  spec.block_sizes.block_id_lookup(block_id)` (.size_hint,.min_size,.autotuner_min,.max_size). The axis
  has `reduction=True` but block_id NOT in reduction_loops.
- **out-of-scope**: `{bs.block_id for bs in env.block_sizes if bs.reduction} - set(spec.reduction_loops.valid_block_ids())`
  (roller bailed: matmul/stack-coupled). Skip per-axis seeding.
- `BlockSizeInfo.reduction` flag: `compile_environment.py:~1246`, set via `allocate_reduction_dimension`
  -> `allocate_block_size(reduction=True)` (~L630-632).

## reduction_loops knob semantics (Triton)
- One int per rdim. `ReductionLoopSpec._flat_config` (config_spec.py:~1746-1760): `value >= size_hint`
  -> `None` (persistent) else int chunk. Codegen `ReductionLoopBlockSizeSource.from_config`
  (compile_environment.py:~1382-1395): None -> `backend.static_rdim_size(size_hint)` (R_BLOCK = full
  pow2 extent); else int = looped R_BLOCK. Fragment range `[8, next_pow2(size_hint)]`, default
  `min(high, 4096)`.
- `max_reduction_threads()` is **None** on Triton (base Backend returns None; TritonBackend doesn't
  override). So DON'T rely on a thread cap; derive chunk from size_hint.
- `num_warps` = NumWarpsFragment(1,32) default 4 (`DEFAULT_NUM_WARPS`). `num_stages` =
  IntegerFragment(1,8) on CUDA, default 1 (`DEFAULT_NUM_STAGES`).
- OBSERVED default_config behavior: persistent (None) for size_hint<=4096; looped chunk 4096 for
  size_hint>4096 (= min(next_pow2, 4096)). This is the un-seeded baseline the heuristic must beat.

## Recommended Step-2 plan
1. Define `ReductionFact` NamedTuple in config_spec.py after MatmulFact; store `reduction_facts` list in
   ConfigSpec.__init__. Suggested fields: block_id, size_hint, m_block_ids, static_n, dtype, num_inner_ops,
   num_outer_ops (grow by co-design).
2. Populate in device_ir.py register_rollable_reductions 2nd-pass loop (before compiler_seed_configs).
3. Clone CuteReductionTileHeuristic -> triton.py:TritonReductionHeuristic (backend="triton"); drop
   CuTe-only knobs; read reduction_facts; set block_sizes=[1] + reduction_loops (+ optionally
   num_warps/num_stages); keep matmul-exclusion + single-rdim gate (broaden for T2 later).
4. Register in __init__.py HEURISTICS_BY_BACKEND['triton'].
