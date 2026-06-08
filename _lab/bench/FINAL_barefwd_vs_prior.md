# Seeded Helion vs torch.compile-default — bare-forward (matches prior method)

**Metric:** `G_seed = torch.compile_default_lat / helion_seed_lat`. >1 ⇒ seed beats tc.

**Method:** the PRIOR run's method — time the BARE forward kernel
(`helion.kernel(fwd.fn, config=seed)`), not the autograd wrapper. Single process,
`torch._dynamo.reset()` per shape (clean tc baseline), median-of-15 do_bench, fp32,
test split, contention-guarded (foreign GPU = 0 on every run, no accuracy failures).
Seed extracted from the LIVE PR heuristic via `compiler_seed_configs`. Build fns + tc
references copied verbatim from `run3_task2_replay_bench.py`.

| kernel | bare-fwd NOW | min–max | PRIOR bare (test) | \|Δ\| | (TritonBench autograd) |
|---|---|---|---|---|---|
| rms_norm | 1.056 | 0.98–1.10 | 1.019 | 0.037 | 0.79 |
| layer_norm | 1.069 | 0.98–1.20 | 1.012 | 0.057 | 1.16 |
| softmax | 1.056 | 0.98–1.53 | 1.079 | 0.023 | 1.15 |
| sum | 1.002 | 0.98–1.01 | 1.004 | 0.002 | 1.00 |
| cross_entropy | 0.838 | 0.63–1.06 | 0.825 | 0.013 | 0.83 |
| long_sum | 1.034 | 0.47–1.15 | 1.014 | 0.020 | 1.03 |
| welford | 0.948 | 0.92–0.96 | 0.956 | 0.008 | 3.19 |
| kl_div | 1.039 | 1.00–1.31 | 1.038 | 0.001 | 1.04 |
| jsd | 1.016 | 1.00–1.12 | 1.014 | 0.002 | 1.01 |

**geomean of medians = 1.004** (prior bare geomean = 0.993). Match is near-exact
(max deviation 0.057, most ≤0.02).

## Conclusions
1. **The bare-forward re-run reproduces the prior tables.** Every kernel lands within
   ~0.02–0.06 of the prior numbers. The prior tables ARE in our records
   (`run2/_lab/logs/run3/task2_replay_bench.json` test split,
   `t1_train_tcdefault.json` / `t2_train_3way_summary.md` train split).

2. **The PR did NOT change seed-vs-torch.compile.** The numbers match the prior run;
   the seed behaves identically.

3. **The earlier "TritonBench autograd" column was the odd one out, for rms_norm /
   layer_norm only.** Those two kernels' TritonBench hooks call an autograd `Function`
   wrapper (`RMSNormFunction.apply` / `LayerNormFunction.apply`); the wrapper's host
   bookkeeping (autograd graph + save_for_backward) is a large *fraction* of latency at
   small shapes, depressing rms_norm to 0.79 and (because tc's layer_norm baseline is
   itself slower) lifting layer_norm to 1.16. The other 7 kernels match between methods
   (their TritonBench hooks call the kernel directly, no autograd wrapper). welford 3.19
   was a separate artifact: TritonBench benches it at bf16 vs a compile-storming tc
   baseline; the bare-fwd fp32 number (0.948) is the honest one.

## Honest read of the seed vs torch.compile-default
Across 9 kernels the seed is **≈ on par with torch.compile-default (geomean ~1.0)**:
- Small net wins: rms_norm, layer_norm, softmax, long_sum, kl_div, jsd, sum (1.00–1.07).
- Behind on **cross_entropy (0.84)** at large vocab and **welford (0.95)** — genuine,
  reproduced in both runs (the seed trails tc there).
- Per-shape spread is real: softmax up to 1.53, long_sum down to 0.47 at (M=8, N=2M)
  (tiny-row, torch.sum wins), cross_entropy down to 0.63 at the widest vocab.

The seed's BIG wins are vs the *unseeded Helion default* (geomeans 2–7× on
softmax/kl_div/welford/long_sum — see the `default` column in the prior tables), NOT vs
torch.compile. Against torch.compile specifically, the seed makes Helion competitive
(~parity), which is the correct and honest framing.
