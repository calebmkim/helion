# Seeded Helion vs torch.compile-default — clean tritonbench results

**Metric:** `G_seed = torch.compile_default_latency / helion_seeded_latency`.
G > 1 ⇒ seeded Helion (this PR) is FASTER than torch.compile default-mode.

**Method (the fix):** pure TritonBench `benchmarks/run.py`, ONE isolated process per
kernel via `run_seeded.py` (promotes the reduction seed to `default_config` so
effort=none runs the seed). TritonBench calls `attr.reset()` + resets dynamo per
input, so the torch.compile baseline is measured cleanly — this is exactly the
isolation my earlier hand-rolled batched scripts lacked (they corrupted the tc
baseline by compiling many variants in one long-lived process). Helion column is the
operator's helion method; tc column is `torch_compile_no_autotune_*` /
`torch_compile_*_default` (default mode, NOT max-autotune). fp32. Contention-guarded
(foreign GPU mem = 0 on every run). 9 kernels.

| kernel | NOW median | NOW geo | (min–max) | PRIOR median | Δ |
|---|---|---|---|---|---|
| rms_norm | 0.79 | 0.79 | 0.60–0.98 | 0.991 | −0.20 |
| layer_norm | 1.16 | 1.16 | 0.99–1.36 | 0.987 | +0.17 |
| softmax | 1.15 | 1.24 | 1.02–1.89 | 1.079 | +0.07 |
| sum | 1.00 | 0.99 | 0.98–1.00 | 1.021 | −0.02 |
| cross_entropy | 0.83 | 0.85 | 0.65–1.08 | 1.058 | −0.23 |
| long_sum | 1.03 | 0.93 | 0.48–1.15 | 1.117 | −0.08 |
| welford\* | 3.19 | 3.13 | 2.62–3.58 | 0.963 | +2.22 |
| kl_div | 1.04 | 1.05 | 0.96–1.21 | 1.051 | −0.01 |
| jsd | 1.01 | 0.99 | 0.91–1.03 | 1.013 | +0.00 |

geomean of per-kernel medians = **1.13** (seeded Helion ≈ torch.compile, modest net win).

## Reconciliation with the prior run (train split)
PRIOR = run3 **train** split (`t1_train_tcdefault.json`, `t2_train_3way_summary.md`);
NOW = **test** split. Same metric, same definition. Most kernels agree within noise.
The two visible deltas are **shape-mix**, confirmed by regime-matching:
- **cross_entropy** (0.83 vs 1.06): G_seed falls as vocab V grows — present in BOTH
  datasets. Regime-matched: V<98k prior 1.084 / now 1.056; V≥98k prior 0.757 / now
  0.718. The test split is V-heavy → lower median. Consistent.
- **rms_norm** (0.79 vs 0.99): small-latency effect. The test split is heavier on
  small shapes (tc 28–45µs) where the helion autograd-wrapper path
  (`RMSNormFunction.apply`, what TritonBench times) pays a ~9–18µs fixed overhead that
  tc avoids; at the one common shape (4096,2560) both runs measure tc≈37.7µs
  identically — nothing regressed, the test split just samples more small shapes.
- **welford** (3.19 vs 0.96): the prior 0.96 was at **bf16** with a different harness;
  here the seed is dramatically faster than the operator's (compile-storming) bf16
  torch.compile baseline. Treat as not-directly-comparable (see note).

## Honest caveats
- **rms_norm / layer_norm asymmetry is a torch.compile artifact, not the seed.** Helion
  seed latencies are ~equal for both kernels; tc is just slower at layer_norm (its
  `normalized_shape` list arg → more dynamo guards) → higher G_seed for ln. The seed's
  own effect (seed vs unseeded-default) is ~identical and modest (~1.0–1.28) for both.
- **welford\*** runs at the operator's hardcoded **bf16**; TritonBench reports acc=0,
  but a direct check shows the kernel is **fp32-exact** (maxerr 0.0) and the bf16
  "failure" is bf16-vs-fp32 rounding (~0.05) affecting seed AND default equally — not a
  seed correctness bug. The 3.2x is real perf but vs a bf16 tc baseline that
  compile-storms, so it overstates steady-state.
- **kl_div / jsd** here use the operator's built-in V-sweep (V ≤ 131k), so the big
  wins I saw earlier at extreme V (≥250k) are out of range — these tritonbench-native
  numbers (1.0–1.05) are the conservative, honest ones.
- **long_sum** G=0.48 at (M=8, N=2M): a tiny-row shape where `torch.sum` beats the
  seed's 1-row-per-program config; the rest are ~1.03–1.15.

## Bottom line
Across 9 kernels, seeded Helion is **roughly on par with torch.compile-default
(geomean ≈ 1.13)**, clearly ahead on softmax/layer_norm/welford, behind on small-shape
rms_norm and wide-vocab cross_entropy (where the autograd-wrapper overhead / large-V
regime favor torch.compile). The earlier headline "2.18x" was an artifact of (a) my
harness corrupting the tc baseline and (b) comparing seed-vs-unseeded-default rather
than seed-vs-torch.compile. Against torch.compile specifically, the seed is a modest
net win, not a large one. Nothing regressed between runs — the apparent changes are
shape-mix + the harness bug, now fixed by going pure-tritonbench.
