# STAGE 3 REPORT — composed-fact seed for matmul + reduction-epilogue

**Base:** Stage-2 tip `6a4e091a`. **Commit:** `d204b1a5` (branch `reduction-3stage-stack`). H100 sm90.

## The deliverable — the first COMPOSED fact
A fused matmul + reduction-over-output-axis epilogue (matmul_rms_norm: `acc=x@y` then a reduction
over the matmul's N axis on the register-resident `[M_BLOCK,N]` accumulator, in one kernel) emitted
1 MatmulFact + 0 ReductionFact (the over-N reduction is suppressed when a matmul is present) → an
empty niche, no seed, N-blind default `[32,32]` (1.4-2.7x off best). Stage 3 recognizes the
co-occurrence, registers the epilogue reduction, COMPOSES the two facts, and seeds a footprint-aware
tile.

## The change (commit d204b1a5)
- **`MatmulWithReductionEpilogueFact`** (config_spec.py) — composes the MatmulFact + the epilogue
  ReductionFact (+ n_extent=specialized N, m/k_block_id); holds them, does not re-derive.
- **device_ir** — relax `register_user_tiled_reductions`' matmul guard so its Stage-2 MATERIALIZED
  branch registers the over-N epilogue reduction (the `.sum(-1)` is a ReductionLowering over a
  materialized full-width rdim). `build_matmul_reduction_epilogue_facts` composes when 1 MatmulFact +
  1 ReductionFact co-occur. Pure matmul (no epilogue reduction) and pure reduction (no matmul) do not
  compose — disjoint.
- **`TritonMatmulReductionEpilogueHeuristic`** (triton.py) — footprint-aware M_BLOCK (largest pow2
  whose `[M_BLOCK,N]` fp32 accumulator fits ACC_BUDGET_BYTES=131072, capped MAX_M_BLOCK=64) +
  K_BLOCK=32 + num_stages=3 (K-loop addmm pipelining) + num_warps ramp on M_BLOCK*N. Keyed on the
  resident footprint (NOT an aspect ratio).

## Faithful + disjoint (Gate D PASS)
composed fact fires ONLY on the fused family: fused matmul_rms_norm → composed=1; pure matmul →
composed=0; pure reduction → composed=0. 9 reduction curriculum kernels byte-identical (739/739);
pure matmul keeps its skinny-gemm seed. Gate D proposed a one-line §2 doctrine clarification: a third
"COMPOSED fact" category (holds >=2 built facts, no graph walk, populator reads spec.<fact_list>).

## Perf — the seed closes the default gap and ~matches the grid optimum, across the family
| kernel | shape | acc | seeded_vs_default | seed/best | seeded_vs_tc |
|---|---|---|---|---|---|
| matmul_rms_norm | (131072,256,256) | T | 1.43 | 0.99 | **1.73** (driver repro) |
| matmul_rms_norm | (131072,256,512) | T | 1.88 | 0.96 | **1.53** (driver repro) |
| matmul_rms_norm | (65536,512,512) | T | 2.40 | 0.89 | — |
| matmul_rms_norm | (262144,128,256) | T | 1.42 | 1.00 | — |
| matmul_layernorm (2 reductions) | (131072,256,256/512) | T | 1.31 / 1.50 | 0.96 | — |
| matmul_softmax | (131072,256,512) | T | 1.44 | 0.95 | — |
| matmul_l2_normalize | (131072,256,256) | T | 1.45 | 0.999 | — |
| matmul_sum (scalar out) | (131072,256,256) | T | 1.73 | 0.988 | — |
| matmul_logsumexp (HELD OUT) | (131072,256,256) | T | 1.65 | 0.998 | — |
| matmul_logsumexp (HELD OUT, TEST shape) | (196608,256,768) | T | 1.91 | 0.965 | — |
| matmul_max (HELD OUT) | (131072,256,256) | T | 1.70 | 0.991 | — |
The seed emits the grid answer key (N=256 → [64,32]w4s3 EXACT), closes the N-blind default gap
(1.3-2.4x), lands within 0-11% of the grid optimum, and BEATS torch.compile 1.5-1.73x (driver-run
Gate-A repro). It generalizes across 7 epilogues (FIT + held-out) — the FACT is blind to which
reduction it is (the generalization the composed fact buys). matmul_softmax (256) is a borderline
bf16 tolerance row (0.098 vs 0.09 abs floor — config-independent rounding, all arms same), not a seed
issue.

## Gate verdicts (fresh-context adversarial agents; ledger.json + gate_*.json)
- **Gate D: PASS** — composed fact doctrine-clean (no graph walk; composes built facts), disjointness
  verified 3-way (+ a harder negative: a specialized-N matmul with a 2-D accumulator but no over-N
  reduction → composed=0), faithful population (n_extent = true specialized N).
- **Gate F: PASS** — mechanism = footprint/occupancy, verified in lowered Triton (M_BLOCK sets the
  grid 4096→2048 programs + the resident acc width); boundary verified at N=1024 ([64,1024]*4 overflows
  131072 → drops to [32,1024]*4 = exactly 131072) and N=2048 (SMEM wall, win vanishes); all 4 fields
  attributed (K_BLOCK+num_stages a coupling), none inert.
- **Gate H: KEEP** — faithful footprint key (resident fp32-acc bytes, not aspect ratio), constants are
  real SMEM/occupancy fit points + matmul pipelining depth (not curriculum fences — the monotone
  footprint law means no inter-N memorized fence), general FACT + narrow CLAIM with a Gate-F-proven
  reversal boundary (N=2048).
- **Gate A: refuted=false** — seed timed == seed emitted; fires via the faithful composed fact (a
  brand-new matmul_var kernel gets the identical seed → no kernel-name fence); arm-fair acc gate;
  driver-run independent repro confirms seed beats default 1.43-1.88x AND beats tc 1.53-1.73x.
- **Gate R: accept** — config_recorder cfg_stage2C → cfg_stage3 = 739/739 byte-identical (selection-
  only; the guard relaxation is a no-op for non-matmul kernels); 8 transfer kernels + pure matmul/
  reduction unchanged; the only changed configs are NEW seeds on the previously-unseeded fused family.
- **Gate E (freeze): PASS** — held-out kernels (logsumexp, max) + the held-out TEST shape
  (196608,256,768) all generalize (svd 1.65-1.91, seed/best 0.965-0.998); no overfit.

## Overtime / notes
- The CLAIM is scoped to the measured small-N skinny win (N≤512 decisive, N=1024 marginal, N=2048
  SMEM-wall). The FACT is general (whole family); widening the claim to large N needs a fresh autotune
  proving headroom (none expected — the win is structurally a small-N fusion).
