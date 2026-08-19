# B200 Pointwise-Heuristic Audit Plan

## Goal

Measure the current B200 pointwise seed heuristic against:

1. Helion's unseeded base default.
2. `torch.compile()` in default mode.
3. A checked-in pretuned config where one exists.

The primary metric is cold-L2 CUDA-graph device latency. The experiment is
limited to:

- `swiglu`, `geglu`, and forward `rope`.
- Every vLLM-ported Helion kernel governed by the pointwise heuristic.
- SGLang's Inkling `silu_and_mul_interleaved`.

No online autotuning is part of this experiment.

## Source and Hardware Pinning

- Run on physical GPU 1 only with `CUDA_VISIBLE_DEVICES=1`.
- Assert that exactly one logical GPU is visible and that it is a B200
  (`sm100`).
- Run from current `pytorch/helion:main`. Record the exact Helion, PyTorch,
  Triton, CUDA, and driver revisions in every result.
- Disable online autotuning and explicitly extract and replay each Helion
  config.
- Require `triton_pointwise` to fire for every kernel. A no-fire is a result,
  not permission to substitute another seed.
- Enforce a five-minute per-cell compile/run timeout. RoPE shapes that exceed
  it are recorded as timeouts and the resumable driver continues.

The execution worktree was created at Helion commit
`61f4058f3610e1b2bbabc82df8267ac450f591df` on 2026-08-19.

## Scope of "vLLM Pointwise"

Current Helion contains seven kernels ported from vLLM. Only
`silu_mul_fp8` has a pure pointwise fact. The other six contain an `amax`,
RMSNorm, or another reduction and are covered by the reduction audit:

- `dynamic_per_token_scaled_fp8_quant`
- `per_token_group_fp8_quant`
- `rms_norm_dynamic_per_token_quant`
- `rms_norm_per_block_quant`
- `silu_and_mul_per_block_quant`
- `fused_qk_norm_rope`

Therefore the vLLM cohort here contains only `silu_mul_fp8`.

## Comparison Arms

| Arm | Definition | Kernels |
|---|---|---|
| `default` | `config_spec._base_default_config()` | All |
| `seed` | Config produced by the current `triton_pointwise` heuristic | All |
| `torch_compile` | Bare `torch.compile(reference_fn)` in default mode | All |
| `aot` | Explicit replay of the selected pretuned config | RoPE, vLLM, SGLang |
| `default_null` | Independently compiled duplicate of `default` | All, timing-noise control only |

The default arm is the reporting baseline at `1.00x`. Relative performance is
`default_latency / arm_latency`, so values above one are faster than default.

The AOT label has architecture provenance:

- RoPE: Helion's checked-in SM90 table. B200 runtime fallback would select it.
- vLLM `silu_mul_fp8`: Helion's checked-in SM90 table. There is no SM100 table.
- SGLang: the checked-in SM100 JSON table from local SGLang commit
  `5f79cf35110d6a0be828f266160b75d83a2a6276`.

SM90 AOT results must not be described as B200-tuned results.

## Kernel-Body Rule

Every Helion config arm for a cell compiles the same body:

- Pinned copies of `examples/swiglu.py:_swiglu_fwd` and
  `examples/geglu.py:_geglu`
- `pretuned_kernels/rope/rope.py:rope_fwd`
- `pretuned_kernels/silu_mul_fp8/silu_mul_fp8.py:silu_mul_fp8`
- The vendored SGLang Helion `silu_and_mul_interleaved` body

The pretuned RoPE body is the examples body plus the AOT decorator and benchmark
plumbing. Using it permits explicit AOT replay without changing the computation.
The two flat example bodies are copied into this directory to avoid importing
the executable examples' unrelated `helion._testing`/`pytest` dependency.

## Dtype and Shape Curriculum

All primary inputs use BF16. `silu_mul_fp8` stores FP8 output, matching its
native contract.

### General Pointwise

The SwiGLU and GEGLU shapes sample the original PR2866 reproduction and held-out
sets. They span narrow through wide MLP dimensions and multiple token regimes.

| Kernel | Six shapes |
|---|---|
| `swiglu` | `(32768,1536)`, `(16384,2880)`, `(8192,11008)`, `(8192,14336)`, `(4096,28672)`, `(2048,24576)` |
| `geglu` | `(16384,2880)`, `(8192,6912)`, `(8192,14336)`, `(4096,21504)`, `(4096,36864)`, `(2048,24576)` |

RoPE uses six shapes from PR2866's reproduction/generalization corpus. Shape
syntax is `(batch, heads, sequence, head_dim)`; Q and K use the same head count,
matching that experiment.

| Kernel | Six shapes |
|---|---|
| `rope` | `(1,32,2048,256)`, `(1,32,8192,256)`, `(1,32,4096,256)`, `(2,32,2048,256)`, `(1,32,4096,128)`, `(4,8,4096,128)` |

### vLLM Pointwise

Nine exact keys from the 307-key `silu_mul_fp8` table sample tokens from 1 to
512 and all seven tuned intermediate widths:

`(1,2048)`, `(2,8192)`, `(8,4096)`, `(16,11008)`, `(64,2880)`,
`(128,2048)`, `(256,7688)`, `(384,8192)`, `(512,14336)`.

Shape syntax is `(num_tokens, intermediate)`; the input's last dimension is
`2 * intermediate`.

### SGLang Inkling

Nine exact SM100 table keys cover all three useful configs, with and without
top-k weights, and row counts from single-token decode through 16K-token routed
prefill:

`(6,4096,false)`, `(16,16384,true)`, `(48,512,false)`,
`(192,4096,true)`, `(768,3072,false)`, `(1024,12288,false)`,
`(3072,2048,false)`, `(12288,1536,true)`, `(98304,6144,false)`.

Shape syntax is `(rows, hidden, has_topk_weights)`. The output width is
`hidden / 2`.

## Correctness

Before timing each arm:

1. Run the eager torch reference.
2. Run the explicitly compiled Helion config or compiled torch callable.
3. Compare every returned tensor.
4. Use BF16 tolerances for general/SGLang/RoPE and FP8 tolerances for vLLM.
5. Exclude an inaccurate arm from timing while preserving the failure.

## Cold-L2 CUDA-Graph Timing

Pointwise kernels are small enough that a flush-record-replay loop can add a
graph-launch plateau to each sample. Use the calibrated method already developed
for the SGLang pointwise evaluation:

1. Capture a flush-only graph containing `batch` L2 clears.
2. Capture one graph per arm containing `batch` repetitions of
   `L2 clear -> operation`.
3. Time all graphs with CUDA events in an interleaved, rotating order.
4. Compute per-operation cold latency as
   `(flush_plus_operation_ms - flush_only_ms) / batch`.
5. Report the median of nine round medians, escalating to fifteen rounds when
   any arm's spread exceeds 5%.
6. Use `default_null` to expose the per-cell measurement floor.

The flush is inside each captured timed graph, so every operation sees cold L2
while common graph-launch and flush costs are subtracted.

## Reporting

Store one JSON record per shape containing:

- Environment and source revisions.
- Complete configs and AOT selector provenance.
- Fired heuristic names and pointwise fact.
- Accuracy details.
- Raw round medians, calibrated latency, spread, and null-arm delta.

Generate per-shape tables, per-kernel geomeans, cohort geomeans, and relative
performance charts. Keep general, vLLM, and SGLang cohort summaries separate.
