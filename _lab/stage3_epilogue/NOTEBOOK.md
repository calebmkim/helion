# Stage 3 — composed-fact seed for matmul + reduction-epilogue (NOTEBOOK)

Base = Stage-2 tip `6a4e091a` on `reduction-3stage-stack`. The GENUINE hill-climb (method §3 loop).

## The family
A fused matmul + reduction-over-N-epilogue kernel: `for tile_m: acc=zeros([tile_m,n]); for tile_k:
acc=addmm(...); <reduce over N on acc>; out[tile_m,:]`. N = hl.specialize'd (compile-time const,
never tiled) so the [tile_m,n] fp32 accumulator AND the [tile_k,n] y-operand both scale with N →
SMEM-bound. Today: 1 MatmulFact + 0 ReductionFact (the over-N reduction rides the matmul accumulator,
not an HBM row, and the reduction populators decline when matmul_facts present) → empty niche, no seed.

## Reliability (measured 2026-06-17, the task; to re-confirm on-box): Helion WINS small N
N<=512: 1.4-2.6x over Inductor (incl max-autotune, which picks unfused cuBLAS). Narrows at N=1024
(~1.0-1.3x), vanishes at N=2048 (no valid cfg, SMEM wall). h_default is 1.4-2.7x off h_best on every
shape — THAT gap is the seed's job. Bar: seeded >= 0.75x min(tc, helion-max-autotune); since Helion
oracle beats tc at small N, the interesting bar is seeded-vs-oracle.

## Plan (method §3 per-iteration loop)
1. Author the corpus from named refs: matmul_rms_norm (template), matmul_layernorm (in-tree),
   matmul_softmax, matmul_l2_normalize, matmul_sum (scalar) [FIT]; matmul_logsumexp, matmul_argmax
   [HELD OUT, read once at freeze]. Negative recognizers: pure matmul (examples/matmul.py), pure
   reduction (rms_norm) must NOT fire + stay byte-identical.
2. Write the harness (clone ab_three_arm_transfer): Helion seeded_vs_default ALWAYS + Inductor
   default & max-autotune on skinny rows. Forward-only, cold-L2 do_bench, acc-gate-first.
3. Build the composed MatmulWithReductionEpilogue fact (compose MatmulFact + a ReductionFact for the
   over-N reduction) + populator (register the currently-suppressed reduction) + a new heuristic.
4. Hill-climb the footprint-aware tile lever (M_BLOCK chooser keyed on resident bytes; eligibility =
   "does a productive tile fit", NOT aspect ratio). Spike the lever, don't pick a formula up front.
5. Gates D/F/H/A/R/E. Keep the FACT general (whole family), scope the CLAIM to the measured small-N
   win.

## HARD: pure-matmul + pure-reduction kernels' facts byte-identical (the composed fact fires ONLY on
fused matmul+reduction). The 9 reduction curriculum + 8 transfer must also stay byte-identical (config
recorder).

## STATUS: investigating fact mechanics + authoring corpus/harness.
