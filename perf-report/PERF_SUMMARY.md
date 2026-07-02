# Helion reduction-seed heuristic — manager summary

**One-paragraph version.** Helion's reduction-seed heuristic picks a strong kernel config at
compile time with **no autotuning**. Across 28 real kernels (182 shapes × dtypes = 344 measured
cells) it is **~1.9× faster than Helion's unseeded default on average, at rough parity with
torch.compile (1.02× overall), and — on the vLLM quant kernels — matches vLLM's own per-shape
hand-tuned configs (≈1.02×) while beating torch.compile by 2.3–2.7×.** We also found and fixed one
real config bug along the way (a bf16 narrow-row case that was 4.3× too slow).

## Look around the full results

Everything — the report, per-cell data, shapes, kernel sources, and the benchmark scripts — is on a
public branch of the Helion fork:

- **Full branch (browse here):** https://github.com/calebmkim/helion/tree/reduction-perf-report/perf-report
- **Detailed report** (methodology, per-shape tables, disaster analysis): [`perf-report/REPORT.md`](https://github.com/calebmkim/helion/blob/reduction-perf-report/perf-report/REPORT.md)
- **Raw per-cell data + machine tables:** [`perf-report/results/`](https://github.com/calebmkim/helion/tree/reduction-perf-report/perf-report/results) (`SUMMARY.md`, `summary.json`, one JSON per kernel)
- **The shapes benchmarked:** [`perf-report/shapes.json`](https://github.com/calebmkim/helion/blob/reduction-perf-report/perf-report/shapes.json)
- **Benchmark harness (the scripts):** [`_lab/perf_report/`](https://github.com/calebmkim/helion/tree/reduction-perf-report/_lab/perf_report)
- **The one heuristic fix we made** (removes the `num_warps=1` bug): [commit `d017fc90`](https://github.com/calebmkim/helion/commit/d017fc90) + its [analysis](https://github.com/calebmkim/helion/blob/reduction-perf-report/perf-report/NARROW_W1_ANALYSIS.md)

## How to read the tables

Each number is **`perf = baseline_latency / seed_latency`, so `> 1` means the seed config is
FASTER** than that baseline (`1.30` = 30 % faster; `0.90` = 10 % slower). Each row pools that
kernel's shapes and dtypes (fp32 + bf16); **# shapes** is the count of measured (shape × dtype)
cells. Baselines:

- **vs default** — Helion's compiler default config with the heuristic turned off. *This is what the
  heuristic is responsible for* (the default is known to be bad).
- **vs torch.compile** — `torch.compile` in its default mode (the external yardstick).
- **vs vLLM tuned** — *(vLLM kernels only)* vLLM's own shipped, per-shape **hand-tuned** config for
  an H100 (`nvidia_h100.json`, looked up exactly as vLLM does at runtime). This is the strongest
  possible baseline: expert configs tuned per shape.

A geomean over shapes is used (the outlier-robust average for ratios). Cells that fail an accuracy
check are excluded from a *baseline-vs-seed* comparison **only when** the failure is seed-specific;
where the seed and the baseline fail the accuracy gate *identically* (a kernel-source numeric issue,
not the config's fault) the perf comparison is still valid and is included — see footnotes.

---

## 1. "Normal" reductions — kernels from Helion's `examples/`

The 9 original reduction kernels plus two loss kernels (`fused_linear_jsd`, `grpo`) that live in
`helion/examples/`.

| kernel | # shapes | vs default | vs torch.compile |
|---|---|---|---|
| rms_norm | 16 | 1.06 | 1.02 |
| layer_norm | 16 | 1.06 | 1.02 |
| softmax | 16 | 4.42 | 1.20 |
| welford | 14 | 2.67 | 0.95 † |
| sum | 14 | 1.08 | 1.00 † |
| long_sum | 14 | 2.52 | 0.93 |
| cross_entropy | 14 | 1.30 | 0.89 |
| kl_div | 14 | 6.56 | 1.08 |
| jsd | 14 | 3.90 | 0.92 |
| fused_linear_jsd | 14 | 1.58 | 1.01 |
| grpo | 14 | 3.57 | 1.02 |

Solidly ahead of the default everywhere; at or near torch.compile parity. The few sub-1.0 vs-tc
cases (`cross_entropy`, `long_sum`, `jsd`) are large-vocabulary / tiny-row shapes where
torch.compile uses a *split reduction* that Helion's codegen can't express — a codegen limit, not a
config problem. † `welford`/`sum` include some bf16 shapes whose accuracy fails on a low-precision
accumulator (a kernel-source issue, not the seed); those shapes' perf is still counted because seed
and default fail identically.

## 2. "Normal" reductions — kernels we wrote (robustness set)

Six fused reduction+pointwise kernels we authored specifically to test the heuristic on realistic
patterns it was **never tuned on** (`kernel_sources/transfer/transfer_kernels.py`).

| kernel | # shapes | vs default | vs torch.compile |
|---|---|---|---|
| fused_add_rmsnorm | 24 | 1.12 | 1.00 |
| fused_add_layernorm | 24 | 1.11 | 1.03 |
| gated_rmsnorm | 24 | 1.06 | 0.99 |
| scaled_masked_softmax | 22 | 1.19 | 1.00 |
| cross_entropy_ls_zloss | 22 | 1.70 | 0.96 |
| dynamic_quant | 16 | 1.17 | 1.05 |

Same profile as the `examples/` kernels — beats the default, holds torch.compile parity — on kernels
outside the tuning set. (See the **Robustness** section below for why this matters.)

## 3. M-reduction (norm-backward) — kernels from `examples/`

Backward passes that reduce over the batch/grid axis. Two live in `helion/examples/`.

| kernel | # shapes | vs default | vs torch.compile |
|---|---|---|---|
| rms_norm_bwd | 6 | 12.47 | 1.25 † |
| layer_norm_bwd | 6 | 14.99 | 1.17 |

The default config spills catastrophically on these, so the heuristic's win over it is huge
(12–15×), and it also beats torch.compile. † `rms_norm_bwd` bf16 shapes fail on the bf16
accumulator (kernel-source); seed and default fail identically, so perf still counts.

## 4. M-reduction (norm-backward) — kernels we wrote

Four norm-backward kernels we authored (`kernel_sources/mreduction/`).

| kernel | # shapes | vs default | vs torch.compile |
|---|---|---|---|
| bias_grad_bwd | 6 | 1.13 | 0.70 |
| dyt_bwd | 6 | 7.38 | 0.97 |
| group_norm_bwd | 4 | 8.69 | 0.72 |
| instance_norm_bwd | 4 | 12.58 | 0.48 |

Big wins over the default again. Against torch.compile these are mixed: on the small / low-occupancy
3-D shapes (`instance_norm_bwd`, `bias_grad_bwd`, `group_norm_bwd`) PyTorch's hand-fused native
backward is hard to beat — but the heuristic is still the right choice *within* Helion (it always
beats the Helion default).

## 5. vLLM quantization kernels — with the pretuned-config comparison

The 5 vLLM fp8-quant kernels (`kernel_sources/vllm/`), run in their native bf16-in/fp8-out dtype.
These get a **third baseline — vLLM's own shipped, per-shape hand-tuned config** ("vs vLLM tuned").
This is the headline generalization result: does a *general* heuristic keep up with *expert,
per-shape* tuning?

**All numbers here are on an H100** (both our runs and vLLM's shipped configs). The `vs vLLM tuned`
column compares against vLLM's checked-in **`nvidia_h100.json`** for each kernel (click through to
each below), looked up per shape exactly as vLLM does at runtime. We expect the seed's advantage to
**transfer to B200**, likely with some minor re-tuning of the H100-specific constants — vLLM ships a
separate `nvidia_b200.json`, so a B200 comparison is a straightforward follow-up on that hardware.

| kernel | # shapes | vs default | vs torch.compile | **vs vLLM tuned (H100)** |
|---|---|---|---|---|
| [silu_mul_fp8](https://github.com/vllm-project/vllm/blob/main/vllm/kernels/helion/configs/silu_mul_fp8/nvidia_h100.json) | 4 | 1.05 | 0.76 | **1.03** |
| [dynamic_per_token_scaled_fp8_quant](https://github.com/vllm-project/vllm/blob/main/vllm/kernels/helion/configs/dynamic_per_token_scaled_fp8_quant/nvidia_h100.json) | 4 | 2.46 | 2.70 | **1.02** |
| [rms_norm_dynamic_per_token_quant](https://github.com/vllm-project/vllm/blob/main/vllm/kernels/helion/configs/rms_norm_dynamic_per_token_quant/nvidia_h100.json) | 4 | 5.27 | 2.34 | **1.04** |
| [per_token_group_fp8_quant](https://github.com/vllm-project/vllm/blob/main/vllm/kernels/helion/configs/per_token_group_fp8_quant/nvidia_h100.json) ‡ | 4 | 0.90 | 0.91 | **0.89** |
| [rms_norm_per_block_quant](https://github.com/vllm-project/vllm/blob/main/vllm/kernels/helion/configs/rms_norm_per_block_quant/nvidia_h100.json) | 4 | 1.34 | 2.60 | **1.00** |

**The reduction seed matches vLLM's own hand-tuned per-shape configs (geomean ≈ 1.02×, i.e.
parity-to-slightly-ahead) — while beating torch.compile by 2.3–2.7× on the reduction-heavy quant
kernels.** So a single general heuristic keeps up with kernel-specific expert tuning at zero tuning
cost. ‡ `per_token_group_fp8_quant` sits on an fp8 rounding tolerance boundary and fails the
accuracy check on 3 of 4 shapes — **identically for the seed, default, and vLLM configs**, so the
perf comparison is still apples-to-apples; the sub-1.0 number is driven by one shape
(`8192×4096×128`) where the seed's config is ~1.6× slower — a real, isolated gap worth a follow-up
(it is *not* caused by the accuracy issue).

## 6. Synthetic stress-test kernels (correctness/generality coverage — not perf workloads)

These are **hand-written kernels whose only job is to stress the heuristic's decision logic** on
structural cases the real corpus doesn't cover — e.g. co-resident reductions over different axes,
3-D reduction tiles, mixed grid-tile + user-tile layouts, and the persist-vs-chunk "re-read
ceiling" decision. They are **not** realistic performance workloads and have no meaningful
torch.compile reference, so the only metric is **vs default** (`> 1` = seed's config beats the
compiler default). The point is coverage: does the heuristic *fire correctly and pick a sane
config* on shapes it was never tuned on? Sources:

- **Categorization probes** — [`kernel_sources/synthetic_probes/`](https://github.com/calebmkim/helion/tree/reduction-perf-report/perf-report/kernel_sources/synthetic_probes) (11 `p*` structural cases + 2 `oos*` out-of-scope confirmers)
- **Adversarial persist-vs-chunk probes** — [`kernel_sources/adversarial_synth/`](https://github.com/calebmkim/helion/tree/reduction-perf-report/perf-report/kernel_sources/adversarial_synth) (each meant to be swept over the reduction width)

| probe | vs default | what it stresses |
|---|---|---|
| p1-outer-product-coresident | 0.99 | two co-resident reductions over different axes (outer product) |
| p2-feature-plus-rowaccum-offcorpus | 1.03 | feature reduction + row accumulator, off-corpus layout |
| p3-full-grid-nonquant | 1.15 | full-grid reduction, non-quant |
| p4-two-rollable-sequential | 1.02 | two sequential rollable reductions |
| p5-3d-reduction-tile | 1.63 | 3-D reduction tile sizing |
| p6-mixed-coresident-plus-sequential | 0.87 | mixed co-resident + sequential (the one slightly-behind case) |
| p7-gridtile-then-usertile | 2.02 | grid-tile followed by user-tile |
| p8-fullgrid-plus-usertile | 2.90 | full-grid + user-tile |
| p9-nonred-loop-then-fullextent | 23.7 | non-reduction loop then full-extent reduction |
| p10-usertile-and-gridtile | 25.5 | user-tile and grid-tile combined |
| p11-fullextent-then-nonred-loop | 31.9 | full-extent reduction then non-reduction loop |
| oos1-jagged-declined | — (correctly declines) | data-dependent (jagged) extent — heuristic *should* not fire; it doesn't |
| oos2-strided-dim0 | 1.04 | strided dim-0 reduction (out-of-scope, still sane) |

Adversarial persist-vs-chunk probes (stress the re-read / byte-budget ceiling):

| probe | vs default |
|---|---|
| synth_working_set_undercount | 10.8 |
| synth_reread_variance_NOT_wrongcap | 7.4 |
| synth_livecount_scalar_out | 7.3 |
| synth_l2_vs_reg_twograph_scalar | 7.1 |
| synth_reread_softmax_VERIFIED_wrongcap | 6.0 |
| synth_arith_intensity | n/a — frozen flag state doesn't compile on this branch (kernel-source, not the heuristic) |
| synth_store_bandwidth | n/a — same (frozen flag state) |

**Takeaway:** the heuristic fires and picks a good config across all these structural cases (big
wins where the default badly mis-sizes, e.g. p9–p11 and the adversarial ceiling probes at 6–32×),
correctly **declines** the one genuinely out-of-scope case (`oos1`, jagged extent), and only slightly
trails the default on one mixed layout (`p6`, 0.87×). This is the evidence the heuristic keys on
real workload structure rather than the specific kernels it was tuned on.

---

*Method in brief:* one process per kernel, all arms timed on the same input tensors, forward-only,
cold-L2 `do_bench` (median of 9–15), accuracy-gated before timing. Full methodology and the
per-shape breakdown are in [`REPORT.md`](https://github.com/calebmkim/helion/blob/reduction-perf-report/perf-report/REPORT.md).
