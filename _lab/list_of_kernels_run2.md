# Reduction Kernel Worklist

This file tracks kernels to use for H100/fp32 reduction heuristic work.

Shape conventions:

- Row-wise reductions use `(M, N)`, where `N` is the reduced/feature dimension.
- Loss reductions use `(BT, V)`, where `BT = batch * sequence_length` and `V`
  is the vocabulary/class dimension.
- Welford uses `(S, D)`, matching the TritonBench naming for rows and hidden
  dimension.
- "In-sample" shapes are for heuristic development and hill-climbing.
- "Out-of-sample" shapes should be held back for validation.
- If an out-of-sample shape is too large for local iteration, first shrink
  `M`/`BT` while keeping the same `N`/`V`.
- To get more shape samples, feel free to use the TritonBench shapes as well (if you think that would be helpful).

## Already Implemented

### RMSNorm fwd/bwd

Implementation:

- `helion/examples/rms_norm.py`
- Functions: `rms_norm_fwd`, `rms_norm_bwd`, `rms_norm`,
  `rms_norm_tritonbench`

Reduction style coverage:

- Forward is written in compiler-managed row-reduction style. This is useful
  for testing persistent vs looped `reduction_loops` choices.
- Backward mixes per-row reductions over `N` with parameter-gradient reductions
  over `M`; use it after forward heuristics are stable.

Shape sources:

- TritonBench: `helion/benchmarks/tritonbench/tritonbench/operators/rms_norm/operator.py`
- Existing pretuned sweep: `helion/pretuned_kernels/rms_norm/rms_norm.py`

In-sample fwd shapes:

```text
(2048, 1024), (2048, 2048), (2048, 4096), (2048, 8192), (2048, 16384)
(4096, 1536), (4096, 3584), (4096, 5120), (4096, 7168)
(8192, 4096), (8192, 8192)
(32768, 256), (32768, 1024)
```

In-sample bwd subset:

```text
(2048, 1024), (2048, 4096), (4096, 1536), (4096, 4096)
(4096, 8192), (8192, 4096)
```

Out-of-sample shapes:

```text
(16, 4096), (128, 4096)
(2048, 1023), (2048, 2047), (2048, 3072), (2048, 6144)
(4096, 12288), (1024, 32768)
(262144, 256), (589824, 256)
```

### LayerNorm fwd/bwd

Implementation:

- `helion/examples/layer_norm.py`
- Functions: `layer_norm_fwd`, `layer_norm_bwd`, `layer_norm`,
  `layer_norm_tritonbench`

Reduction style coverage:

- Forward is a compiler-managed row reduction with two reductions over `N`
  (`sum(x)` and `sum((x - mean)^2)`).
- Backward has per-row reductions over `N` plus parameter-gradient reductions
  over `M`.

Shape sources:

- TritonBench: `helion/benchmarks/tritonbench/tritonbench/operators/layer_norm/operator.py`
- Existing pretuned sweep: `helion/pretuned_kernels/layer_norm/layer_norm.py`

In-sample fwd shapes:

```text
(4096, 1024), (4096, 2048), (4096, 4096), (4096, 8192)
(4096, 12288), (4096, 15872)
(2048, 3584), (2048, 8192)
(8192, 4096), (8192, 5120), (8192, 7168)
```

In-sample bwd subset:

```text
(2048, 1024), (2048, 4096), (4096, 2048), (4096, 4096)
(4096, 8192), (8192, 4096)
```

Out-of-sample shapes:

```text
(16, 4096), (128, 4096)
(2048, 1023), (2048, 1536), (2048, 2047)
(4096, 6144), (1024, 32768), (1024, 36864), (1152, 36864)
(262144, 256)
```

### Welford LayerNorm variant

Implementation:

- `helion/examples/welford.py`
- Function: `welford`

Reduction style coverage:

- This is a structured reduction over `(count, mean, M2)`.
- The combine operation is mathematically associative, but this implementation
  is manually looped over `D`, so it is better as out-of-sample validation than
  as the first target for persistent-vs-looped selection.

Shape source:

- TritonBench: `helion/benchmarks/tritonbench/tritonbench/operators/welford/operator.py`

In-sample shapes:

```text
(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096)
```

Out-of-sample shapes:

```text
(262144, 2560), (262144, 3072), (262144, 6144), (262144, 8192)
(65536, 16384)
```

### Softmax fwd/bwd

Implementation:

- `helion/examples/softmax.py`
- Functions: `softmax`, `softmax_decomposed`, `softmax_two_pass`,
  `softmax_bwd`, `softmax_fwd_bwd`, `softmax_tritonbench`

Reduction style coverage:

- `softmax` and `softmax_decomposed` are useful for compiler-managed row
  reductions.
- `softmax_two_pass` and large-`N` backward are manually looped/two-pass forms.
  Keep those separate when evaluating persistent-vs-looped defaults.

Shape sources:

- TritonBench: `helion/benchmarks/tritonbench/tritonbench/operators/softmax/operator.py`
- Existing pretuned sweep: `helion/pretuned_kernels/softmax/softmax.py`

In-sample shapes:

```text
(4096, 256), (4096, 512), (4096, 1024), (4096, 2048)
(4096, 4096), (4096, 8192), (4096, 12288), (4096, 16384)
(32768, 256), (32768, 1024)
```

Out-of-sample shapes:

```text
(16, 4096), (128, 4096)
(2048, 1023), (2048, 2047), (2048, 32768)
(512, 65536), (128, 131072), (262144, 256)
```

### Cross entropy fwd

Implementation:

- `helion/examples/cross_entropy.py`
- Function: `cross_entropy`

Reduction style coverage:

- Row-wise large-vocabulary `logsumexp` plus target gather.
- Source is compiler-managed over `V`, but realistic `V` values should strongly
  exercise looped reduction choices.

Shape sources:

- TritonBench: `helion/benchmarks/tritonbench/tritonbench/operators/cross_entropy/operator.py`
- Existing pretuned sweep: `helion/pretuned_kernels/cross_entropy/cross_entropy.py`

In-sample shapes:

```text
(4096, 4096), (4096, 16384)
(8192, 32768), (16384, 32768)
(8192, 65536), (16384, 65536)
(8192, 131072)
```

Out-of-sample shapes:

```text
(2048, 32000), (4096, 32000), (8192, 128000)
(2048, 128256), (4096, 129280), (2048, 151936)
(1024, 256000)
```

### Sum over last dim

Implementation:

- `helion/examples/sum.py`
- Functions: `sum_kernel`, `sum_tritonbench`

Reduction style coverage:

- Minimal compiler-managed row-wise sum over the last dimension.
- This is the cleanest sanity check for the default `reduction_loops`
  heuristic.

Shape source:

- TritonBench: `helion/benchmarks/tritonbench/tritonbench/operators/sum/operator.py`

In-sample shapes:

```text
(2048, 1024), (2048, 4096), (2048, 16384)
(4096, 1536), (4096, 5120)
(8192, 256), (8192, 4096)
(32768, 256), (32768, 1024)
```

Out-of-sample shapes:

```text
(1, 4096), (16, 4096)
(2048, 1023), (2048, 2047)
(4096, 6144), (1024, 32768), (512, 65536)
(262144, 256)
```

### Long dense sum

Implementation:

- `helion/examples/long_sum.py`
- Functions: `longsum`, `longsum_w_red_loop`, `longsum_manual`

Reduction style coverage:

- `longsum` uses `reduction_loops=[None]`.
- `longsum_w_red_loop` uses a configured looped reduction.
- `longsum_manual` is explicitly manually looped.
- This file is useful for comparing compiler-managed persistent, compiler-managed
  looped, and source-manual looped forms on long reductions.

Shape source:

- Local example uses `(4, 130000)`.
- Additional shapes below are chosen to stress long-`N` reduction behavior.

In-sample shapes:

```text
(1, 32768), (2, 65536), (4, 130000), (8, 131072), (16, 262144)
```

Out-of-sample shapes:

```text
(1, 100000), (1, 1048576), (4, 262143), (32, 65536)
```

### KL div loss

Implementation:

- `helion/examples/kl_div.py`
- Functions/classes: `kl_div_forward`, `HelionKLDivLoss`, `kl_div_tritonbench`

Reduction style coverage:

- Manually looped over `V`.
- Best used for explicit looped-reduction block-size heuristics, not for
  compiler-managed persistent-vs-looped source selection.

Shape source:

- TritonBench: `helion/benchmarks/tritonbench/tritonbench/operators/kl_div/operator.py`

In-sample shapes:

```text
(4096, 4096), (4096, 8192), (4096, 16384)
(4096, 32768), (4096, 65536), (4096, 131072)
```

Out-of-sample shapes:

```text
(4096, 32000), (8192, 65536)
(2048, 128256), (1024, 256000)
```

### JSD loss

Implementation:

- `helion/examples/jsd.py`
- Functions/classes: `jsd_forward`, `HelionJSD`, `jsd_tritonbench`

Reduction style coverage:

- Manually looped over `V`.
- Best used for explicit looped-reduction block-size heuristics and loss-style
  epilogue pressure.

Shape source:

- TritonBench: `helion/benchmarks/tritonbench/tritonbench/operators/jsd/operator.py`

In-sample shapes:

```text
(8192, 4096), (8192, 8192), (8192, 16384)
(8192, 32768), (8192, 65536), (8192, 131072)
```

Out-of-sample shapes:

```text
(4096, 128256), (4096, 129280)
(2048, 151936), (1024, 256000)
```

## Could Implement In Future

Highest-priority out-of-sample kernels:

- standalone `logsumexp`
- generic row-wise `sum` / `mean` / `max` / `min`
- cross entropy bwd
- log softmax

Useful after the core row-reduction heuristic is stable:

- argmax / argmin
- l1 / l2 / Frobenius norm
- mse loss
- sparsemax
- tvd and post-training loss reductions

Lower priority for this specific reduction heuristic pass because they add
extra layout/indexing behavior:

- batchnorm
- groupnorm
- maxpool / avgpool / globalavgpool

## In-sample-v2 (RUN 2, Goal 4) — real-AI-workload development shapes

Added 2026-05-31. These are a NEW in-sample split for hill-climbing (fold into Product A
under the >10% per-kernel referee-confirmed regression backstop). Firewall: for the 7
kernels whose TEST is NOT re-read (layer_norm, sum, long_sum, softmax, cross_entropy,
kl_div, jsd) every shape below is DISJOINT from that kernel's sealed TEST and VALIDATION
sets (verified vs `_lab/harness/TEST_readonce.py`). For rms_norm and welford the TEST set
is being re-constituted under run 2 (rms_norm TEST G regen + welford corrected-kernel TEST
re-read are pre-authorized), so a couple of their old TEST/validation shapes are promoted
here — the regenerated TEST will exclude them. The prime welford `(262144,1543)` stays a
CORRECTNESS canary, NOT an in-sample perf target.

```text
rms_norm     : (256,4096)  (512,8192)  (256,5120)  (1024,2560)
layer_norm   : (512,4096)  (512,8192)  (256,5120)  (1024,2560)
welford      : (262144,2560)  (262144,5120)  (8192,4096)  (16384,4096)
softmax      : (8192,3072)  (8192,32768)
cross_entropy: (8192,32000)  (8192,50257)  (4096,128256)  (4096,151936)
kl_div       : (8192,32000)  (4096,50257)  (4096,128256)  (2048,151936)
jsd          : (8192,32000)  (8192,50257)  (8192,128256)  (4096,151936)
sum          : (256,4096)  (512,8192)  (32,65536)  (256,262144)
long_sum     : (4,524288)  (8,393216)
```

Rationale: real vocabs (Llama-2 32000, GPT-2 50257, Llama-3 128256, Qwen 151936) x realistic
batch.seq for the loss kernels; small-M + Llama/Mistral hidden dims (5120, 2560) for the norms
(closes the tiny-M TEST drag); welford M-variation (in-sample was 100% M=262144) + non-pow2 N;
softmax non-pow2 + long-context; sum/long_sum intermediate-to-long pooling regimes.
