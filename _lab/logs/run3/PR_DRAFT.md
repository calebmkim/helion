<!-- DRAFT PR message — refine freely. Perf numbers are from saved benches (not recomputed):
     t2_train_3way.json, perf_ab_merged.json, t1_train_tcdefault.json in _lab/logs/run3/. -->

# Reduction seed heuristics: add T2-style support + improve T1-style (esp. long reduction dims)

## What this does

Compile-time autotuner **seed heuristics** for forward inner-reduction kernels (H100 / sm_90, Triton, fp32). A heuristic reads pre-digested workload facts (`ReductionFact`) and emits one seed `Config` — used both to skip autotuning (the seed *is* the config) and to seed the autotuner's search.

Two changes:

1. **Add heuristic support for "T2-style" reductions.** T2 = user-tiled reductions where the reduction axis is an ordinary `block_sizes` entry (softmax, welford, kl_div, jsd). The existing reduction heuristic (`is_canonical_row_reduction`) only fires on T1 rollable-rdim kernels and leaves these to a generic default. A new `TritonReductionUserTileHeuristic` seeds them.
Example of T2-style reduction:
```
acc = hl.zeros([tile_m, block_size_n], dtype=x.dtype)
for tile_n in hl.tile(n, block_size=block_size_n):
  acc += x[tile_m, tile_n]
out[tile_m] = acc.sum(-1)
```

2. **Improve the heuristic for "T1-style" reductions on long reduction dims.**. Adds an rnumel-scaled `num_warps` ramp, a structural + byte-budget persistent-vs-looped decision, faithful per-slot load eviction, and a `persistent_interleaved` grid cluster for wide looped re-read rows. This matters most for kernels with **much wider reduction dims than rms_norm** — e.g. **cross_entropy**, whose vocab dim is routinely 30K–256K vs rms_norm's ≤16K. At those widths the previous fixed `num_warps=4` persistent seed register-spills and a performance cliff; the width-scaled seed avoids it.

*(Off sm_90 the deepened T1 path falls back to the original conservative seed, so no behavior changes on other hardware.)*

## Performance (train shapes)

Geomean **speedup of the new seed** vs each baseline (>1.0 = new seed faster). `default` = the unseeded Helion default config for T2, or the existing T1 heuristic for T1. `tc-default` = `torch.compile` default mode. All single-process, correctness-gated (rtol 1e-3 / atol 1e-4), median-of-7 `do_bench`, H100 fp32.

### T2-style (new heuristic vs *no* prior seed)

| Kernel | Num shapes | default (Helion default cfg) | new seed heuristic | tc-default |
|---|---|---|---|---|
| softmax | 15 | 3.82× | 1.00× (ref) | 1.08× |
| welford | 15 | 2.81× | 1.00× (ref) | 0.96× |
| kl_div | 13 | 4.79× | 1.00× (ref) | 1.08× |
| jsd | 13 | 2.43× | 1.00× (ref) | 1.01× |

*Read: on softmax the new seed is 3.82× faster than the unseeded Helion default and 1.08× faster than torch.compile-default. The new-seed column is the 1.00× reference; the other columns are how much faster the new seed is than that baseline.*

### T1-style (new heuristic vs the original T1 heuristic)

| Kernel | Num shapes | default (original T1 heuristic) | new seed heuristic | tc-default |
|---|---|---|---|---|
| rms_norm | 16 | 0.99× | 1.00× (ref) | 0.99× |
| layer_norm | 16 | 1.03× | 1.00× (ref) | 1.01× |
| sum | 14 | 1.08× | 1.00× (ref) | 1.03× |
| long_sum | 10 | 1.44× | 1.00× (ref) | 1.16× |
| cross_entropy | 14 | 6.68× | 1.00× (ref) | 0.93× |

*Read: cross_entropy is 6.68× faster than the original heuristic (the wide-vocab register-spill fix) mainly due to the fact that cross_entropy is generally run on much larger reduction dimensions; rms_norm/layer_norm are unchanged-to-slightly-better (their dims are narrow, so the new levers are inert — and correctly so).*

@ethche two high-level questions to discuss (before I get claude/you to review any code) 
1. There’s a lot of effort put into the eviction policy portion of the heuristic (including multiple fields on the ReductionFact) and it *does* lead to 1-1.3x speedups on some larger shapes (on cross entropy and softmax). So performance wise there’s no downside to these heuristics. But like 90% of the perf comes from selecting block size and num_warps well, which is quite simple.
    1. The code is kind of messy though, and tbh, the way it selects the eviction slots is brittle: there are kernels you can construct where it will “mistakenly” select the wrong index to “keep” in the eviction policy, although it works for all the kernels here. So that would be an argument for removing it: it’s too brittle and complicated. There’s an argument that this is just a heuristic so it’s fine to ship (and for my benchmarks it does improve perf), but there’s also an argument that maybe if we want to have eviction policies in the heuristic we should be more principled about it. 
2. welford make the heuristic more complicated (whenever you see is_structured_combine or apply_block_ids that’s for handling Welford). I’m gonna refactor that stuff into its own heuristic when I handle other reductions (e.g., backwards rms_norm), lmk if you’d rather me get rid of this complicated code in this PR, or it's fine to leave with a TODO.
3. The comments are probably too long-- I should tighten them up before merging.