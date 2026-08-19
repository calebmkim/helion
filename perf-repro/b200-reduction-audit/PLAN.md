# B200 Reduction-Heuristic Audit Plan

## Goal

Measure the current B200 reduction seed heuristic against:

1. Helion's unseeded base default.
2. `torch.compile()` in default mode.
3. The checked-in B200 (`sm100`) AOT config when one exists.

The primary metric is cold-L2 CUDA-graph device latency. The experiment covers:

- Four general reduction kernels with checked-in B200 AOT heuristics.
- Six reduction kernels from the original reduction experiment without AOT
  heuristics.
- Six reduction-containing kernels ported from vLLM with checked-in B200 AOT
  configs.

No online autotuning is part of this experiment.

## Source and Hardware Pinning

- Run on one NVIDIA B200 only.
- Set `CUDA_VISIBLE_DEVICES=1`; all code must use logical `cuda:0`.
- Assert the device name and compute capability (`sm100`) before running.
- Start the execution branch from the latest `pytorch/helion:main` available
  when the run begins. Record the exact Helion commit, PyTorch commit/version,
  Triton version, CUDA driver/runtime versions, and GPU name in the results.
- Carry this directory onto that source revision. Do not run the historical
  `e3d5e2ce3` Helion implementation unchanged.
- Disable online autotuning. Extract and replay every Helion config explicitly.
- Assert that the B200 reduction heuristic actually produced the `seed` arm for
  every Helion cell. Record a no-fire as such rather than substituting another
  heuristic.

## Kernel-Body Rule

Every Helion arm for a cell must compile the same kernel body.

For the four general AOT kernels, use the body in `pretuned_kernels/`, not the
similarly named body in `examples/`. This makes the seed, default, and AOT arms
strictly comparable:

- Pretuned RMSNorm and LayerNorm are output-only, unlike the example forward
  kernels that also store statistics for backward.
- Pretuned Softmax matches `examples.softmax.softmax`, not the
  `softmax_two_pass` body used in the historical audit.
- Pretuned Cross Entropy is nearly identical to the example body.

Consequently, these four rows test the B200 reduction heuristic on the exact
AOT workload. They are not a direct rerun of the historical example-body rows.
The six non-AOT kernels below retain the original experiment's bodies.

To prevent `@helion.aot_kernel` dispatch from silently choosing the AOT arm,
bind the undecorated body once and explicitly compile/replay each Helion config.

## Comparison Arms

| Arm | Definition | Kernels |
|---|---|---|
| `seed` | First config returned by the current B200 reduction seed heuristic | All |
| `default` | `config_spec._base_default_config()` with heuristics disabled | All |
| `torch_compile` | `torch.compile(reference_fn)` in default mode, without max-autotune | All |
| `aot_sm100` | Config selected by the checked-in `_helion_aot_<kernel>_cuda_sm100.py` | Four general AOT kernels and six vLLM kernels |

Do not use `default_config()` for the `default` arm: a promoted seed can make it
return the heuristic config. Dump the complete effective config for every
Helion arm.

For `aot_sm100`, select the config using the checked-in selector, then replay
that config explicitly against the same body as `seed` and `default`. Record
the selector key/config index and whether the shape was part of the tuning
sweep. A compile failure or invalid config is a result; do not silently repair
or replace it.

## General AOT Cohort

Use the native dtype from each checked-in AOT sweep and six shapes from that
sweep. The shapes intentionally cover narrow, common-model, wide, non-power-of-
two, and materially different outer-dimension regimes.

| Kernel body | Dtype | Shape convention | Six shapes | Arms |
|---|---|---|---|---|
| `pretuned_kernels/rms_norm/rms_norm.py:rms_norm` | BF16 | `(M, N)` | `(2048, 48)`, `(2048, 1023)`, `(2048, 4096)`, `(4096, 7168)`, `(16384, 8192)`, `(589824, 256)` | seed/default/torch_compile/aot_sm100 |
| `pretuned_kernels/layer_norm/layer_norm.py:layer_norm` | FP16 | `(M, N)` | `(4096, 1024)`, `(4096, 3072)`, `(8192, 5120)`, `(4096, 12288)`, `(4096, 16384)`, `(1024, 36864)` | seed/default/torch_compile/aot_sm100 |
| `pretuned_kernels/softmax/softmax.py:softmax` | FP16 | `(M, N)` | `(4096, 256)`, `(4096, 384)`, `(4096, 768)`, `(4096, 4096)`, `(4096, 16384)`, `(2048, 32768)` | seed/default/torch_compile/aot_sm100 |
| `pretuned_kernels/cross_entropy/cross_entropy.py:cross_entropy` | BF16 | `(tokens, vocab)` | `(2048, 32000)`, `(1024, 256000)`, `(2048, 128256)`, `(8192, 128000)`, `(4096, 152064)`, `(2048, 256000)` | seed/default/torch_compile/aot_sm100 |

Use the torch-native reference adjacent to each pretuned kernel for both the
accuracy oracle and the `torch_compile` arm.

## Original-Kernel Cohort

Use BF16 for all six kernels. Use the original experiment's kernel bodies and
six shapes per kernel from its reproduction/generalization corpus.

| Kernel | Shape convention | Six shapes | Provenance |
|---|---|---|---|
| `kl_div` | `(M, N)` | `(8192, 32768)`, `(2048, 50257)`, `(4096, 114688)`, `(1024, 128256)`, `(4096, 151936)`, `(1024, 250000)` | Original `curriculum/test` |
| `jsd` | `(M, N)` | `(8192, 32768)`, `(2048, 50257)`, `(4096, 114688)`, `(2048, 128256)`, `(8192, 151936)`, `(1024, 250000)` | Original `curriculum/test` |
| `fused_linear_jsd` | `(M, vocab)` | `(8192, 32000)`, `(4096, 50257)`, `(8192, 128256)`, `(2048, 151936)`, `(2048, 256000)`, `(16384, 32000)` | Original `transfer/all` |
| `grpo` | `(batch, sequence, vocab)` | `(8, 1024, 32000)`, `(8, 2048, 64000)`, `(4, 2048, 128256)`, `(8, 4096, 128256)`, `(16, 1024, 50257)`, `(4, 1024, 256000)` | Original `transfer/all` |
| `rms_norm_bwd` | `(M, N)` | `(2048, 4096)`, `(8192, 4096)`, `(4096, 8192)`, `(16384, 4096)`, `(8192, 2048)`, `(2048, 11008)` | Three original `mreduction/all` plus three `mreduction_gen` |
| `layer_norm_bwd` | `(M, N)` | `(2048, 4096)`, `(8192, 4096)`, `(4096, 8192)`, `(16384, 4096)`, `(8192, 2048)`, `(2048, 11008)` | Three original `mreduction/all` plus three `mreduction_gen` |

These kernels have three arms: `seed`, `default`, and `torch_compile`.

Retain these exact historical body mappings:

- `kl_div` -> `examples/kl_div.py:kl_div_forward`
- `jsd` -> `examples/jsd.py:jsd_forward`
- `fused_linear_jsd` -> `examples/fused_linear_jsd.py:jsd_kernel`
- `grpo` -> `examples/grpo_loss.py:grpo_loss_forward`
- `rms_norm_bwd` -> `examples/rms_norm.py:rms_norm_bwd`
- `layer_norm_bwd` -> `examples/layer_norm.py:layer_norm_bwd`

In particular, do not substitute `fused_linear_jsd_fwd` for the historical
`fused_linear_jsd` label. The old harness also used loss-only torch references
for some multi-output kernels while timing extra Helion outputs. This run
intentionally changes that detail: the torch reference must produce every
observable output from the mapped Helion body. Record this as a methodology
change when comparing the B200 results with the historical report.

## vLLM Cohort

Benchmark all six reduction-containing vLLM kernels that ship an `sm100` AOT
table. Use each module's canonical argument setup, including optional argument
choices, mutation behavior, constants, and output dtypes.

All six use BF16 primary inputs. The five quantization kernels produce FP8
values plus FP32 scales; Fused QK Norm + RoPE mutates BF16 QKV in place.

Use nine exact tuned keys per kernel: the Cartesian product of three structural
sizes and token counts `{1, 128, 8192}`. This samples decode, an intermediate
batch, and large prefill while avoiding the AOT selector's fallback path.

| Kernel | Shape convention | Structural values | Nine-shape definition |
|---|---|---|---|
| `dynamic_per_token_scaled_fp8_quant` | `(tokens, hidden)` | hidden `{2048, 4096, 5120}` | `{1,128,8192} x {2048,4096,5120}` |
| `per_token_group_fp8_quant` | `(tokens, hidden, group)` | hidden `{2048, 4096, 5120}`, group `128` | `{1,128,8192} x {2048,4096,5120} x {128}` |
| `rms_norm_dynamic_per_token_quant` | `(tokens, hidden)` | hidden `{2048, 4096, 5120}` | `{1,128,8192} x {2048,4096,5120}` |
| `rms_norm_per_block_quant` | `(tokens, hidden, group)` | hidden `{2048, 4096, 5120}`, group `128` | `{1,128,8192} x {2048,4096,5120} x {128}` |
| `silu_and_mul_per_block_quant` | `(tokens, intermediate, group)` | intermediate `{6144, 12288, 25600}`, group `128` | `{1,128,8192} x {6144,12288,25600} x {128}` |
| `fused_qk_norm_rope` | `(tokens, q_heads, kv_heads)` | Q heads `{16, 32, 64}`, KV heads `8` | `{1,128,8192} x {16,32,64} x {8}` |

Each vLLM kernel has all four arms. This is a config comparison on the Helion
body, not a comparison with vLLM's native CUDA operator.

## `torch.compile` Requirements

- Compile the equivalent torch-native reference with bare
  `torch.compile(reference_fn)` in default mode.
- Compile and warm it before CUDA-graph capture; compilation time is excluded.
- Reset Dynamo between shapes so a previous dynamic-shape specialization does
  not alter a later cell.
- Verify with `torch._dynamo.explain` that the timed reference has no accidental
  graph breaks. A one-time `fullgraph=True` compile may be used as a validation
  gate, but the timed arm remains default `torch.compile()`.
- If a reference contains avoidable `.item()`, scatter, allocation, or mutation
  patterns that prevent fair fusion/capture, replace it with a mathematically
  equivalent torch-native reference and document the change.
- Preserve the kernel's observable output and mutation contract. Do not time a
  reduced-output reference against a multi-output Helion kernel.

## Correctness Gate

Run correctness before timing every `(kernel, shape)` cell:

1. Compute an eager torch reference, using FP32 accumulation where the operation
   requires it.
2. Run every Helion arm and the compiled torch arm from equivalent initial
   state.
3. Compare all outputs, including FP32 scales, saved statistics, and mutated
   tensors.
4. Use per-kernel tolerances appropriate to BF16, FP16, and FP8. Record maximum
   absolute/relative error and the exact tolerance.
5. Do not include a failing arm in performance ratios. Preserve the failure in
   the raw results.

For stateful kernels, give every arm separate working tensors. Restore inputs
that are read and mutated, such as QKV or residuals, from an immutable source
before each timed replay.

## Cold-L2 CUDA-Graph Timing

The headline number for every arm is CUDA-graph device time under cold L2:

1. Preallocate inputs, outputs, restore buffers, and a cache-clear buffer larger
   than B200 L2.
2. Warm each compiled callable, then capture one CUDA graph per arm.
3. Before each sample, restore read/write inputs outside the timed interval.
4. Clear L2 on the same stream after restoration and before the start event.
5. Record a CUDA start event, replay exactly one graph, then record the end
   event. The cache clear and state restoration are excluded from elapsed time.
6. Interleave arms round-robin within each cell and rotate the first arm across
   rounds to reduce clock/order bias.
7. Target roughly 100 ms wall time per round, with repetitions clamped to
   `[5, 1000]`.
8. Report the median of nine round medians. Escalate to 15 rounds when the
   relative spread exceeds 5%.
9. Synchronize only where required to read timing events. Record all per-round
   samples, not only the final median.

CUDA-graph capture must contain only the operation being compared. In
particular, do not capture input cloning, L2 clearing, correctness checks, or
output validation.

## Result Schema and Reporting

Write one raw JSON row per `(kernel, shape, dtype)` with:

- Source commit and complete software/hardware metadata.
- Kernel body identifier and input/output dtypes.
- Shape and canonical optional-argument settings.
- Full effective `seed`, `default`, and `aot_sm100` configs where applicable.
- AOT selector key/index and exact-key/tuning-sweep status.
- Whether the reduction heuristic fired.
- Accuracy status and error details for every arm.
- Median latency, all round medians, repetition count, and spread for every arm.

Define ratios so values above one mean the reduction seed is faster:

- `G_default = default_us / seed_us`
- `G_tc = torch_compile_us / seed_us`
- `G_aot = aot_sm100_us / seed_us`

Report:

- A per-shape latency/config table.
- Per-kernel geomeans and min/max ratios.
- Separate summaries for the general AOT, original-kernel, and vLLM cohorts.
- A list of correctness failures, compile/capture failures, heuristic no-fires,
  and cases with greater than 5% timing spread.

Do not combine the three cohorts into a single headline geomean without also
showing the separate cohort and per-kernel results.

## Execution Order

1. Pin the current Helion revision and environment metadata.
2. Implement explicit config extraction/replay and raw JSON output in this
   directory, reusing hardened utilities from the historical `perf-repro`
   harness where appropriate.
3. Run one smoke shape per kernel and require all expected arms to compile,
   capture, and pass correctness.
4. Run the 24 general-AOT cells, 36 original-kernel cells, and 54 vLLM cells on
   physical GPU 1 only.
5. Aggregate the 114 cells and regenerate all tables solely from raw JSON.
6. Investigate material losses using emitted Triton, register/spill metadata,
   and the recorded configs; do not modify the headline measurements after
   seeing results.
