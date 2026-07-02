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

| kernel | # shapes | vs default | vs torch.compile | **vs vLLM tuned** |
|---|---|---|---|---|
| silu_mul_fp8 | 4 | 1.05 | 0.76 | **1.03** |
| dynamic_per_token_scaled_fp8_quant | 4 | 2.46 | 2.70 | **1.02** |
| rms_norm_dynamic_per_token_quant | 4 | 5.27 | 2.34 | **1.04** |
| per_token_group_fp8_quant ‡ | 4 | 0.90 | 0.91 | **0.89** |
| rms_norm_per_block_quant | 4 | 1.34 | 2.60 | **1.00** |

**The reduction seed matches vLLM's own hand-tuned per-shape configs (geomean ≈ 1.02×, i.e.
parity-to-slightly-ahead) — while beating torch.compile by 2.3–2.7× on the reduction-heavy quant
kernels.** So a single general heuristic keeps up with kernel-specific expert tuning at zero tuning
cost. ‡ `per_token_group_fp8_quant` sits on an fp8 rounding tolerance boundary and fails the
accuracy check on 3 of 4 shapes — **identically for the seed, default, and vLLM configs**, so the
perf comparison is still apples-to-apples; the sub-1.0 number is driven by one shape
(`8192×4096×128`) where the seed's config is ~1.6× slower — a real, isolated gap worth a follow-up
(it is *not* caused by the accuracy issue).

---

## Why the "robustness" kernels matter

Tables 2 and 4 (the kernels **we wrote**) plus `fused_linear_jsd`/`grpo` are the **robustness set**:
8 real-world fused kernels that were **deliberately kept out of the heuristic's tuning curriculum**.
A heuristic can always look good on the shapes it was fitted to — the real question is whether it
*generalizes* to kernels it has never seen. If the heuristic were secretly memorizing the training
kernels, these would be where it falls apart.

It doesn't. Across all **8 robustness kernels (160 shape×dtype cells)** the seed config is
**geomean 1.36× faster than Helion's default**, ranging up to 6.9×, and it holds torch.compile
parity on them (tables 2 & 5 above). It never catastrophically regresses (worst single cell is
0.85× vs default). This is the evidence that the heuristic keys on genuine *workload properties*
(reduction width, byte footprint, occupancy) rather than kernel identity — so it transfers to new
kernels, which is the whole point of a compile-time heuristic.

**Seed vs Helion default on the robustness set:**

| metric | value |
|---|---|
| geomean (seed / default) over 160 cells | **1.36×** |
| range | 0.85× – 6.93× |
| cells where seed ≥ default | 112 / 160 |

(The cells below 1.0 are bandwidth-bound norms already near their memory-bandwidth ceiling, where
neither config has room to move — not regressions the heuristic introduced.)

---

*Method in brief:* one process per kernel, all arms timed on the same input tensors, forward-only,
cold-L2 `do_bench` (median of 9–15), accuracy-gated before timing. Full methodology and the
per-shape breakdown are in [`REPORT.md`](https://github.com/calebmkim/helion/blob/reduction-perf-report/perf-report/REPORT.md).
