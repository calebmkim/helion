# T2 Support Map — user-tiled reductions (softmax_two_pass, kl_div, jsd)

> Grounded by code-investigator (2026-05-29). Worktree `/home/calebkim/helion-new-heuristics/wt-reduction`.
> File is the AUTHORITATIVE impl (`triton.py` gate is `_triton_reduction_eligible`, not the CuTe
> `_reduction_kernel_eligible`). Line numbers approximate.

## CRITICAL CORRECTION to the original assumption
T2 reduction axes are NOT `block_sizes` entries with `reduction=True`. Verified by binding all 3:
- softmax_two_pass / kl_div / jsd: BOTH block_sizes have `reduction=False` (`LoopSpecBlockSizeSource`);
  `reduction_loops=0`, `reduction_facts=0`, `matmul_facts=0`.
- A user `hl.tile(n)` over the reduction axis is an ordinary tile. The `.sum/.amax(dim=1)` reduces over it
  via `ReductionLowering`, which finds the EXISTING user block (`inductor_lowering.py:683-684`) and does
  NOT call `allocate_reduction_dimension` -> no reduction=True block, no ReductionLoopSpec.
- (Simple `softmax` whole-row IS T1: block_id reduction=True, ReductionLoopBlockSizeSource, gate fires.)

## How to FIND the T2 reduction axis (env + device_ir only)
```python
from helion._compiler.inductor_lowering import ReductionLowering
red_block_ids = {low.block_index for gi in device_ir.graphs for node in gi.graph.nodes
                 if isinstance((low := node.meta.get("lowering")), ReductionLowering)}
t2_red_block_ids = [b for b in red_block_ids if b not in set(spec.grid_block_ids)]  # filter grid axis
```
The `not in grid_block_ids` filter is LOAD-BEARING for jsd: its dead beta==0/1 branches do `amax(dim=0)`
over the M/grid tile, so red_block_ids={0,1}; grid_block_ids=[1] removes it -> real V reduction = 0.
- Probe: softmax_two_pass red block_id=1 (size_hint 2560); kl_div/jsd red block_id=0 (size_hint 65536).
- Spec->knob index: `idx = spec.block_sizes.block_id_to_index(red_block_id)` (block_id_sequence.py:119).

## ReductionFact field probe (T2)
- softmax_two_pass: num_load=2, num_store=1, num_reduction_ops=2, dtype fp32, size_hint 2560.
- kl_div: load=2, store=1, reductions=1.
- jsd: load=2, **store=2** (loss + dX), **reductions=4**  <- BAND B signature.

## T2 knob semantics (verified by codegen)
- Knob = `block_sizes[red_idx]` = R_BLOCK (the inner `hl.tile(n, block_size=...)` tile). The `for tile_n`
  loop is always in source; R_BLOCK sets trip count + tile width.
- **Persistent** = R_BLOCK >= next_pow2(N) -> inner loop runs once (whole row, no roffset sweep).
  **Looped** = R_BLOCK < N -> ceil(N/R_BLOCK) iters.
- Set R_BLOCK = next_pow2(size_hint) for persistent (analogous to T1 [None]); cap at
  `env.backend.max_tensor_numel` = 2^20 (same as T1). Fragment bounds reachable (probe: softmax R=4096,
  kl_div/jsd R=65536 all within max_size).
- BAND B numel constraint: kl_div/jsd carry `u0*u1 <= 2^20` (M_BLOCK x R_BLOCK) because the inner loop
  carries `[M_BLOCK,R_BLOCK]` live accumulators (loss_sum, intermediate_loss, jsd's intermediate_dX).
  Persistent R_BLOCK=full-N SURVIVES iff M_BLOCK at floor (1). So KEEP M_BLOCK at floor (the existing
  `_m_block_size` already does this). At M_BLOCK>1 a full-N R_BLOCK would be auto-shrunk.

## IMPLEMENTATION PLAN
(a) **Eligibility** (`triton.py` `_triton_reduction_eligible`): drop `len(block_sizes)==1` and
    `len(reduction_loops)==1`; gate on `len(spec.reduction_facts)==1 and not spec.matmul_facts`. Admits
    T2 once its fact exists; still excludes GEMM + multi-reduction.
(b) **T2 populate** (`device_ir.py`: new `register_user_tiled_reductions()` called right AFTER
    `register_rollable_reductions()` (~L2401) and before `raise_grid_block_minimums()`, GUARDED by
    `if not env.config_spec.reduction_facts:` so T1/T2 are mutually exclusive and fact count stays 1).
    Find red block_id via the ReductionLowering predicate above; build ReductionFact with
    `m_block_ids=tuple(grid_block_ids)`, `size_hint=env.block_sizes[block_id].size_hint()`, and
    dtype/itemsize/num_load/num_store/num_reduction_ops from the SAME counting loop as
    `_build_reduction_fact` (reuse its imports: _MEMORY_OPS, load/store, ReductionLowering).
(c) **get_seed_config T1-vs-T2 routing**: compute `extent` (persistent next_pow2(size_hint), else
    LOOPED_CHUNK) + num_warps exactly as today. If `fact.block_id` in reduction_loops block_ids (T1):
    emit `reduction_loops=[None|LOOPED_CHUNK]`, `block_sizes=[_m_block_size]`. Else (T2):
    `red_idx=spec.block_sizes.block_id_to_index(fact.block_id)`; set every block_size at its floor EXCEPT
    `block_sizes[red_idx]=extent`; emit NO reduction_loops.
(d) **Reuse** `_num_warps` (rnumel ramp 4/8/16/32) + the 2^20 persistent cap + LOOPED_CHUNK/LOOPED_NUM_WARPS
    verbatim — they key on `fact.size_hint`, not the knob mechanism. Only the knob WRITTEN changes.
(e) **Band B (jsd/kl_div)**: keep M_BLOCK at floor (numel constraint). If persistent/w32 REGRESSES vs the
    persistent/w32 baseline, the lever is a sub-cap R_BLOCK (e.g. <2^20) and/or lower warps, gated on
    `num_store>=2` or `num_reduction_ops>=2` (NEVER kernel identity) and VALIDATED with a matched A/B vs
    persistent/w32 (the methodology lesson: never A/B vs the default strawman).

## No-regression requirement
The gate change (reduction_loops==1 -> reduction_facts==1) + the guarded T2 populate must leave T1
(rms_norm/sum/long_sum/layer_norm) byte-identical. VERIFY: all existing T1 seeds unchanged (inert-proof).
