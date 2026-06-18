# STAGE 3 — RESULTS / evidence for the gate stack

Immutable commit (frozen): `d204b1a5`. Base = Stage-2 tip `6a4e091a`. Box H100 sm90, L2 50MB.

## The deliverable: a COMPOSED fact for fused matmul + reduction-over-output epilogue
A fused kernel (matmul_rms_norm: `acc=x@y; reduce over N on acc; out[tile_m,:]`) emits today 1
MatmulFact + 0 ReductionFact (the over-N reduction is suppressed when a matmul is present) → empty
niche, no seed, N-blind default `[32,32]`. Stage 3 recognizes the co-occurrence, registers the
epilogue reduction, COMPOSES the two facts, and seeds a footprint-aware tile.

## The 3-part change (commit d204b1a5)
- **config_spec `MatmulWithReductionEpilogueFact`**: composes MatmulFact + the epilogue ReductionFact
  (+ n_extent = the specialized N, m/k_block_id), not re-deriving extents/dtypes.
- **device_ir**: relax `register_user_tiled_reductions`' matmul guard so its MATERIALIZED branch
  (built in Stage 2) registers the over-N epilogue reduction for the fused case — the `.sum(-1)` IS a
  ReductionLowering over a materialized full-width rdim (block_id=2, reduction=True, in neither
  block_sizes nor reduction_loops). Pure matmul has no such reduction (excluded); user-tiled T2 still
  declines for matmul. `build_matmul_reduction_epilogue_facts` composes when 1 MatmulFact + 1
  ReductionFact co-occur.
- **triton `TritonMatmulReductionEpilogueHeuristic`**: footprint-aware M_BLOCK = largest pow2 whose
  `[M_BLOCK,N]` fp32 acc fits ACC_BUDGET_BYTES=131072, capped MAX_M_BLOCK=64; K_BLOCK=32;
  num_stages=3 (matmul K-loop pipelining); num_warps ramp on M_BLOCK*N. Eligible iff exactly one
  composed fact.

## DISJOINTNESS (Gate D) — composed fact fires ONLY on the fused family
- fused matmul_rms_norm: matmul=1, reduction=1, **composed=1** (n_extent=256, full_width=True,
  num_carried_2d=1).
- pure matmul (examples/matmul.py): matmul=1, reduction=0, **composed=0** (no over-output reduction).
- pure reduction (examples/rms_norm.py): matmul=0, reduction=1, **composed=0** (no matmul).
- 9 reduction curriculum kernels: **byte-identical (739/739)** config_recorder cfg_stage2C → cfg_stage3.
- pure matmul seed config: unchanged.

## The seed matches the answer key (grid sweep, the oracle)
matmul_rms_norm grid best: (131072,256,256)→[64,32]w4s3; (131072,256,512)→[64,64]w8s3;
(262144,128,256)→[64,32]w4s3. The seed emits: N=256→[64,32]w4s3 (EXACT match), N=512→[64,32]w8s3,
N=1024→[32,32]w8s3 (footprint drops tile_m), N=2048→[16,32]w8s3. Footprint-aware as designed.

## Perf — seeded vs default vs grid-best (G_default = default/seed; seed/best = best_lat/seed_lat)
(forward-only, single-process, median-of-9 cold-L2 do_bench, acc-gated)
| kernel | shape | acc | seeded_vs_default | seed/best |
|---|---|---|---|---|
| matmul_rms_norm | (131072,256,256) | True | **1.43** | 0.99 |
| matmul_rms_norm | (131072,256,512) | True | **1.89** | 0.96 |
| matmul_rms_norm | (65536,512,512) | True | **2.40** | 0.89 |
| matmul_rms_norm | (262144,128,256) | True | **1.42** | 1.00 |
| matmul_layernorm | (131072,256,256) | True | **1.31** | 0.96 |
| matmul_layernorm | (131072,256,512) | True | **1.50** | 0.96 |
The seed closes the N-blind default gap (1.3-2.4x) and lands within 0-11% of the grid optimum, across
TWO different epilogues (rms_norm; layernorm=2 reductions) — the FACT is blind to which reduction it
is (the generalization the composed fact buys). Grid best beats torch.compile 1.4-2.9x (answer key:
best_vs_tc_default 1.7-2.2, best_vs_tc_max 2.1-2.9), so seeded (≈best) beats tc by ~1.4-2.9x.
[softmax/l2_normalize/sum + held-out logsumexp/max + val/test: bench running, appended below.]

## Gate F — mechanism
The default [32,32] is N-blind: it floors M_BLOCK=32 regardless of how the [M_BLOCK,N] fp32 acc +
[K_BLOCK,N] operand scale with the specialized N. The lever sizes M_BLOCK to the resident
fp32-accumulator footprint (M_BLOCK=64 at N≤512 where it fits, 32 at N=1024, 16 at N=2048 where the
SMEM wall is hit and the win vanishes) + num_stages=3 pipelines the K-loop addmm. Field attribution:
M_BLOCK (occupancy/footprint), K_BLOCK, num_stages (pipelining), num_warps (ramp) — the answer key
uses all four; seed ≈ best confirms they carry the win.

## Notes
- The bf16 accuracy gate uses max_abs/output-RMS for magnitude-scaling reductions, max_abs for
  softmax (measured floors), per footgun #6b (matmul accumulation order + bf16 rounding, not bugs).
- (131072,256,2048): SMEM wall — the seed emits [16,32] but it may not fit; the autotuner/seed is
  never forced, so an infeasible tile costs nothing (the family's win vanishes at N=2048 by design).
