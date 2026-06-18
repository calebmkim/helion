# rms_norm vs layer_norm — shape-by-shape investigation

You flagged the "opposite-direction regressions" (rms_norm G_seed 0.83 ↓, layer_norm
1.12 ↑) as suspicious for two near-identical kernels. **They were both artifacts.**
Single-process, contention-guarded (foreign_mib=0), fp32, 15-repeat do_bench, both
kernels on the SAME 10 shapes. Medians:

| metric (median over 10 shapes) | rms_norm | layer_norm |
|---|---|---|
| helion seed µs (wrapper path = what run.py times) | 47.0 | 54.2 |
| helion default µs (wrapper) | 50.7 | 56.0 |
| helion seed µs (bare fwd-only) | 39.0 | 39.6 |
| **tc_default µs** | **48.3** | **66.2** |
| G_seed (wrapper) | 1.03 | 1.21 |
| G_seed (fwd-only) | 1.24 | 1.55 |
| **seed_lift (wrapper)** | **1.010** | **1.012** |
| **seed_lift (fwd-only)** | **1.022** | **1.010** |
| autograd-wrapper overhead µs | 9.1 | 13.6 |

## Finding 1 — the rms-vs-ln "divergence" is a torch.compile artifact, not the seed
Helion seed/default latencies are **nearly identical** between rms_norm and layer_norm at
every shape (the kernels are near-identical, as expected). The G_seed gap comes entirely
from **torch.compile being much slower at layer_norm**: e.g. (16384,896) tc_rms=54µs vs
**tc_ln=121µs** (2.2x); (8192,1280) 45µs vs 69µs. So layer_norm's higher G_seed means
"tc is a weak baseline for layer_norm," NOT "the seed helps layer_norm more." The seed's
own effect (`seed_lift`) is identical for both (~1.01).

## Finding 2 — the run.py-vs-direct discrepancy was autograd-wrapper overhead
My original run.py sweep gave rms G_seed ≈ 0.68–0.98; my first shape-by-shape gave
1.06–1.42 for the same shapes. Cause: run.py times the tritonbench wrapper `rms_norm()`
= `RMSNormFunction.apply` (an autograd Function that builds the backward graph +
save_for_backward), which adds a **~9–18µs fixed overhead** per call. My first script timed
the bare `rms_norm_fwd` kernel (no wrapper). The overhead is constant, so it dominates small
shapes (why small-N looked worst) and vanishes at the largest shape ((16384,1536): overhead
~0, both harnesses agree G≈1.15). The seed only controls the forward kernel config, so:
- **fwd-only G_seed** (1.24/1.55) = the seed-vs-tc on the work the seed actually governs.
- **wrapper G_seed** (1.03/1.21) = the user-facing fwd+autograd path run.py measures.
Both are valid; they answer different questions.

## Finding 3 — seed_lift (the real seed metric) is robust and modest
`seed_lift = unseeded_default / seeded` is ~identical in both the wrapper and fwd-only
paths (the autograd overhead is on both arms and cancels). For BOTH kernels it is **~1.0–1.28
per shape, median ~1.01** — the seed never regresses either kernel; it gives a small,
consistent win that grows with N (1.28 at 2048×10240). There is **no rms-vs-ln asymmetry**
in the seed's effect.

## Corrected takeaway for the headline
- rms_norm and layer_norm are NOT opposite-direction outcomes — both get the same modest
  seed_lift (~1.0–1.28). The earlier G_seed numbers (rms 0.83 / ln 1.12) reflected (a) the
  autograd wrapper overhead in run.py and (b) torch.compile's uneven baseline (slow ln),
  not the seed.
- For norms, the seed's value is a small forward-kernel improvement; the big seed wins
  remain on softmax/kl_div/jsd/long_sum/welford where the unseeded default genuinely spills.
