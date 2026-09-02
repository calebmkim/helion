# PyTorch Blog: Heuristic Benchmark Results

This document indexes the benchmark results for the matmul, multi-matmul,
reduction, and pointwise heuristics. Unless stated otherwise, ratios above
`1.0x` mean the heuristic is faster than the named baseline.

The linear-attention and held-out matmul/multi-matmul inputs are listed in the
[benchmark shape inventory](PYTORCH_BLOG_BENCHMARK_SHAPES.md).
Per-cell latencies and ratios normalized to the selected default are in the
[raw data exports](PYTORCH_BLOG_RAW_DATA/README.md).

## Recommended Sources

| Area | Human-readable report | Raw data / figures |
|---|---|---|
| Linear attention, B200 | [Corrected V3 results](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/e2e_fla_v3/RESULTS.md) | [results.json](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/e2e_fla_v3/results.json), [figures directory](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/e2e_fla_v3) |
| Matmul and multi-matmul, B200 | [On-corpus pre-tuned comparison](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/on_corpus_pretuned/RESULTS.md) | [broader 244-cell results](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/RESULTS.md) |
| Expanded-seed search, B200 | [All-cell and significant-advantage aggregates](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_SIGNIFICANT_SEED_ADVANTAGE.md) | [three-way tables](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_TABLES.md), [tables.json](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_TABLES.json), [progress figures](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/autotune_progress) |
| SGLang KDA, B200 | [Four-arm operation benchmark](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/sglang_triton_head_to_head/four_arm_no_heuristics_v2_address_matched/RESULTS.md) | [head-to-head report](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/sglang_triton_head_to_head/REPORT.md) |
| Reduction, B200 | [Definitive 114-cell summary](/home/dev/local/wt-b200-reduction-audit-run/perf-repro/b200-reduction-audit/results/full/SUMMARY.md) | [summary.json](/home/dev/local/wt-b200-reduction-audit-run/perf-repro/b200-reduction-audit/results/full/summary.json), [charts](/home/dev/local/wt-b200-reduction-audit-run/perf-repro/b200-reduction-audit/results/charts) |
| Broader reduction/generalization | [Per-kernel blog table](/home/dev/local/wt-b200-perf-report-repro-plan/perf-repro/results/PERF_TABLES.md) | [full summary](/home/dev/local/wt-b200-perf-report-repro-plan/perf-repro/results/SUMMARY.md), [summary.json](/home/dev/local/wt-b200-perf-report-repro-plan/perf-repro/results/summary.json) |
| Pointwise, B200 | [Definitive 36-cell summary](/home/dev/local/wt-b200-pointwise-audit/perf-repro/b200-pointwise-audit/results/full/SUMMARY.md) | [summary.json](/home/dev/local/wt-b200-pointwise-audit/perf-repro/b200-pointwise-audit/results/full/summary.json), [charts](/home/dev/local/wt-b200-pointwise-audit/perf-repro/b200-pointwise-audit/results/charts) |

## 1. Linear Attention

### Benchmark

The corrected B200 V3 run contains all 96 planned operation, mode, and shape
cells:

- Seven dense variants, each measured in forward and forward-plus-backward
  modes over six production shapes.
- `kda_fused` and `kda_varlen`, each measured in forward mode over six shapes.
- BF16, chunk size 64, normal warmed eager dispatch, and cold-L2 timing.

The four arms are:

1. `Pre`: the pre-PR compiler heuristic.
2. `Post`: the new matmul and multi-matmul heuristic.
3. `AOT`: the shipped B200 configurations produced by full autotuning.
4. `FLA`: the handwritten Triton implementation from Flash Linear Attention.

`Pre` is the old compiler-selected default, not the raw unseeded
`_base_default_config()`.

### Overall Results

The following geometric means were calculated from the V3 raw JSON using the
report's prescribed per-variant aggregation. Values are normalized to `Pre`.

| Scope | Cells | Pre | FLA Triton | New heuristic | Full-autotuned AOT |
|---|---:|---:|---:|---:|---:|
| Forward | 54 | 1.0000x | 1.5217x | 1.9816x | 2.2599x |
| Forward + backward | 42 | 1.0000x | 1.2559x | 1.8108x | 2.1204x |
| All measured cells | 96 | 1.0000x | 1.3991x | 1.9050x | 2.1978x |

The 96-cell row is a convenience rollup. The benchmark plan treats forward
and forward-plus-backward as the primary, separate headline aggregates.

Direct comparisons:

| Scope | Heuristic / Pre | Heuristic / FLA | AOT / Pre | AOT / heuristic |
|---|---:|---:|---:|---:|
| Forward | 1.9816x | 1.3023x | 2.2599x | 1.1404x |
| Forward + backward | 1.8108x | 1.4418x | 2.1204x | 1.1710x |

### Per-Variant Highlights

| Variant | Mode | Heuristic / Pre | AOT / Pre | Heuristic / FLA |
|---|---|---:|---:|---:|
| Vanilla linear attention | Forward | 3.2689x | 3.5828x | 1.2728x |
| Vanilla linear attention | Forward + backward | 2.4520x | 2.8362x | 2.0281x |
| Simple GLA | Forward | 3.0176x | 3.3084x | 1.3967x |
| Simple GLA | Forward + backward | 2.3384x | 2.6389x | 1.6744x |
| Retention | Forward | 3.0006x | 3.2558x | 1.7975x |
| Retention | Forward + backward | 2.3447x | 2.6339x | 2.1690x |
| Full GLA | Forward | 1.9745x | 2.2174x | 1.5132x |
| Full GLA | Forward + backward | 1.9000x | 2.1040x | 2.0353x |
| Delta rule | Forward | 1.8613x | 2.1409x | 1.3070x |
| Delta rule | Forward + backward | 1.5272x | 1.7665x | 0.9624x |
| Gated delta rule | Forward | 1.7357x | 2.1537x | 1.0811x |
| Gated delta rule | Forward + backward | 1.3251x | 1.8664x | 0.8245x |
| KDA | Forward | 1.5158x | 1.7155x | 1.2367x |
| KDA | Forward + backward | 1.2348x | 1.4092x | 1.0888x |
| KDA fused | Forward | 1.4501x | 1.6492x | 1.2674x |
| KDA varlen | Forward | 1.1353x | 1.3771x | 1.0059x |

The full report contains every per-shape latency. The forward-plus-backward FLA
aggregate includes the reproducible D256 FLA backward cliff.

### Figures

- [FLA baseline, forward](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/e2e_fla_v3/linear_attention_e2e_fla_baseline_with_default_forward.png)
- [FLA baseline, forward + backward](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/e2e_fla_v3/linear_attention_e2e_fla_baseline_with_default_forward_backward.png)
- [Combined FLA-baseline figure](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/e2e_fla_v3/linear_attention_e2e_fla_baseline_with_default_combined.png)
- [Pre-change baseline, combined](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/e2e_fla_v3/linear_attention_e2e_prechange_baseline_combined.png)

Do not use the older `e2e_fla/` or `e2e_fla_v2/` numbers. The benchmark plan
explicitly invalidates both:
[LINEAR_ATTENTION_E2E_BENCHMARK_PLAN.md](/home/dev/local/wt-sm100-linattn/LINEAR_ATTENTION_E2E_BENCHMARK_PLAN.md).

## 2. SGLang KDA

The SGLang benchmark is a separate four-arm B200 operation benchmark. It
compares handwritten SGLang Triton, the raw unseeded compiler configuration,
the current heuristics, and SGLang's shipped pre-tuned configuration.

| Scope | Cells | Unseeded / Triton | Heuristic / Triton | Pre-tuned / Triton | Heuristic / unseeded |
|---|---:|---:|---:|---:|---:|
| Decode | 16 | 1.0829x | 0.8739x | 1.5061x | 0.8070x |
| Fixed/preactivated prefill | 10 | 0.6741x | 1.1556x | 1.5509x | 1.7143x |
| Packed/raw prefill | 10 | 0.6632x | 1.1110x | 1.1982x | 1.6752x |
| Overall | 36 | 0.8284x | 1.0095x | 1.4250x | 1.2186x |

The current heuristic improves both prefill paths but loses to handwritten
Triton on decode. The shipped configurations remain faster than the heuristic.
This run uses CUDA-graph operation timing and should not be merged numerically
with the eager-dispatch FLA results.

## 3. Matmul and Multi-Matmul

### On-Corpus Pre-Tuned Comparison

The main constituent-kernel report contains 149 B200 cells where either the
formula matmul or multi-matmul heuristic fired.

| Group | Cells | Heuristic / Pre | Full-autotuned / Pre | Full-autotuned / heuristic |
|---|---:|---:|---:|---:|
| All matmul cells | 149 | 1.5527x | 1.8704x | 1.2046x |
| Helion linear attention | 124 | 1.5358x | 1.8508x | 1.2051x |
| SGLang | 25 | 1.6391x | 1.9705x | 1.2022x |

By heuristic:

| Heuristic | Cells | Heuristic / Pre | Full-autotuned / heuristic |
|---|---:|---:|---:|
| Formula matmul | 20 | 1.9948x | 1.1257x |
| Multi-matmul | 129 | 1.4935x | 1.2173x |

The [per-kernel table](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/on_corpus_pretuned/RESULTS.md)
contains 30 kernel-family rows and every per-cell latency.

### Broader Corpus

The broader audit contains 244 cells, including off-corpus attention, matmul,
state-space, linear-attention, and SGLang kernels:

| Comparison | Cells | Geometric-mean speedup |
|---|---:|---:|
| Heuristic / pre-change | 244 | 1.4547x |
| Quick-autotune / pre-change | 178 | 1.8997x |
| Quick-autotune / heuristic | 178 | 1.3283x |

`Quick-autotune` is not the full-autotune ceiling. Use the AOT/pre-tuned arms
in the on-corpus and end-to-end reports when discussing max-autotune quality.

The separate 38-cell full-search study is under
[zero_seed_full_autotune](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune).
It studies search efficiency and final-winner quality with no seeds, old
seeds, and expanded seeds; it is not the main runtime comparison.

## 4. Expanded-Seed Search Efficiency

### Experiment and Metric

This separate B200 study runs full-effort LFBO autotuning on the same 38 frozen
kernel-and-shape cells with three search initializations:

1. No compiler seeds.
2. The old seed pool from clean commit `40151a23f`.
3. The expanded multi-seed pool.

For each cell, the shared raw-search best is the lowest latency found by any
arm. "Within X%" means latency at most `(1 + X) * shared_best`. The count is
the terminal attempt containing the first qualifying configuration; failed
attempts consume budget. If an arm never reaches a target, its full terminal
attempt count is substituted as a right-censored lower bound. The full-run row
is total benchmarked configurations, not another latency tolerance.

### All 38 Cells

Parenthetical ratios in the sum columns are relative to no seed. Geomean
ratios are calculated cell by cell rather than from the aggregate sums.

| Search target | No-seed sum | Old-seed sum | Expanded-seed sum | Old / no-seed geomean | Expanded / no-seed geomean |
|---|---:|---:|---:|---:|---:|
| Within 50% | 2,683 | 513 (0.1912x) | 78 (0.0291x) | 0.1006x | 0.0626x |
| Within 25% | 5,131 | 2,960 (0.5769x) | 2,097 (0.4087x) | 0.2184x | 0.1131x |
| Within 12.5% | 8,767 | 6,046 (0.6896x) | 4,867 (0.5551x) | 0.3413x | 0.2553x |
| Within 5% | 12,124 | 8,674 (0.7154x) | 8,419 (0.6944x) | 0.5044x | 0.4478x |
| Complete full search | 19,796 | 18,923 (0.9559x) | 16,936 (0.8555x) | 0.9643x | 0.8551x |

Using the report's aggregate sums (censored lower bounds for target rows),
expanded seeds reduce recorded effort versus no seeds by 97.1% to get within
50%, 59.1% within 25%, 44.5% within 12.5%, 30.6% within 5%, and 14.4% over
the complete search.

Target reach counts at 50%, 25%, 12.5%, and 5%, respectively:

- No seed: 38/38, 36/38, 32/38, and 28/38.
- Old seeds: 38/38, 37/38, 36/38, and 34/38.
- Expanded seeds: 38/38, 37/38, 35/38, and 28/38.

The censored 5% sums therefore need care: old seeds reached the shared 5%
target in more cells than either other arm, even though expanded seeds used
slightly fewer substituted aggregate attempts.

### Final-Winner Quality

Final winners were measured with same-GPU, rotated, interleaved replay rather
than raw search measurements. Relative to no seed, the old-seed winner-latency
geomean was `1.0005x` and expanded seeds were `1.0142x`, where lower is better.
All three replay winners were within 5% on 22/38 cells. Among the other 16,
the fastest replay winner was no seed on 5, old seed on 8, and expanded seed
on 3; the old-seed count includes two exact old/expanded ties.

This supports a search-efficiency claim, not a default-runtime claim:
expanded seeds usually find good configurations much earlier while the final
full-search winners remain close in aggregate.

### Significant Early-Advantage Cohort

The report also defines a filtered 26/38-cell cohort. At an equal early budget
equal to the expanded pool's normalized seed count, these are cells where the
no-seed incumbent is more than 1.5x slower than the better seeded incumbent.
Expanded seeds supply the better seeded incumbent in 12 cells, old seeds in 7,
and the two are tied in 7. The stronger values below must be labeled as this
selected cohort:

| Search target | No-seed sum | Old-seed sum | Expanded-seed sum | Expanded / no-seed sum | Expanded / no-seed geomean |
|---|---:|---:|---:|---:|---:|
| Within 50% | 2,646 | 494 | 53 | 0.0200x | 0.0201x |
| Within 25% | 4,631 | 2,520 | 1,951 | 0.4213x | 0.0707x |
| Within 12.5% | 7,599 | 4,800 | 3,929 | 0.5170x | 0.1896x |
| Within 5% | 9,751 | 6,605 | 6,720 | 0.6892x | 0.4429x |
| Complete full search | 15,186 | 13,909 | 11,745 | 0.7734x | 0.7493x |

Source locations:

- [Threshold aggregates and filtered cohort](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_SIGNIFICANT_SEED_ADVANTAGE.md)
- [Per-cell totals, 5% trajectories, wall time, and replay latency](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_TABLES.md)
- [Per-cell 25%/5% values grouped by replay winner](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_BY_WINNER.md)
- [Machine-readable merged records](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/THREE_WAY_AUTOTUNE_TABLES.json)
- [Aggregate and per-cell progress plots](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune/autotune_progress/README.md)
- [No-seed search and three-way replay artifacts](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/zero_seed_full_autotune)
- [Old-versus-expanded paired search artifacts](/home/dev/local/wt-sm100-linattn/matmul_heuristic_perf_results/multi_seed_full_autotune_ablation)

## 5. Reduction

### Definitive Four-Arm B200 Audit

The definitive audit recorded all 114 planned cells. Its arms are:

- New reduction heuristic.
- Raw unseeded compiler default.
- `torch.compile()` in default mode.
- Checked-in SM100 AOT configuration, where available.

Ratios below are baseline latency divided by heuristic latency.

| Cohort | Valid cells | Heuristic / default | Heuristic / torch.compile | Heuristic / SM100 AOT |
|---|---:|---:|---:|---:|
| General AOT | 24 | 1.029x | 1.124x | 0.917x |
| Original reduction kernels | 35 | 6.028x | 1.013x | n/a |
| vLLM kernels | 54 | 1.918x | 1.388x | 0.918x |

Selected per-kernel results:

| Kernel | Heuristic / default | Heuristic / torch.compile | Heuristic / AOT |
|---|---:|---:|---:|
| RMSNorm forward | 1.075x | 1.030x | 1.008x |
| LayerNorm forward | 0.978x | 0.961x | 0.945x |
| Softmax | 1.096x | 1.259x | 0.975x |
| Cross entropy | 0.974x | 1.282x | 0.762x |
| KL divergence | 9.509x | 1.183x | n/a |
| JSD | 5.237x | 0.936x | n/a |
| Fused linear JSD | 1.571x | 0.927x | n/a |
| GRPO | 3.654x | 0.898x | n/a |
| RMSNorm backward | 12.490x | 0.989x | n/a |
| LayerNorm backward | 15.173x | 1.181x | n/a |

One RMSNorm-backward shape failed the shared accuracy gate for both the
heuristic and unseeded configurations and is excluded from aggregates.

### Broader Reduction and Generalization Audit

The 455-cell audit covers reproduction and unseen-shape generalization across
35 kernel/dtype rows. The most convenient blog table is
[PERF_TABLES.md](/home/dev/local/wt-b200-perf-report-repro-plan/perf-repro/results/PERF_TABLES.md).

Overall:

| Scope | Heuristic / torch.compile | Heuristic / default | Heuristic / vLLM tuned |
|---|---:|---:|---:|
| Reproduction | 1.042x | 2.589x | 0.987x |
| Generalization | 1.102x | 2.899x | 0.955x |

This broader audit is useful for discussing unseen shapes and reductions such
as `sum`, `long_sum`, Welford, KL divergence, JSD, GRPO, RMSNorm backward, and
LayerNorm backward. Do not combine its aggregate directly with the definitive
114-cell audit: the kernel sets and vLLM timing scopes differ.

## 6. vLLM

There are two useful B200 views.

### Exact-Key SM100 AOT Audit

The latest reduction audit covers six reduction-containing vLLM kernels over
nine shapes each:

| Scope | Cells | Heuristic / default | Heuristic / torch.compile | Heuristic / SM100 AOT |
|---|---:|---:|---:|---:|
| vLLM cohort | 54 | 1.918x | 1.388x | 0.918x |

The kernels are dynamic per-token FP8 quantization, per-token group FP8
quantization, RMSNorm plus dynamic quantization, RMSNorm plus per-block
quantization, SiLU-and-multiply plus per-block quantization, and fused
QK-normalization plus RoPE.

### Broader Tuned-Grid Audit

The broader generalization audit covers the four main quantization kernels:

| Scope | Cells | Heuristic / torch.compile | Heuristic / default | Heuristic / vLLM tuned |
|---|---:|---:|---:|---:|
| Posted-number shapes | 16 | 1.249x | 3.816x | 0.987x |
| Full tuned-grid sweep | 96 | 1.192x | 3.523x | 0.956x |

In both audits, the vLLM/AOT arm runs the same Helion kernel body with vLLM's
shipped configuration. These are configuration comparisons, not comparisons
against vLLM's native CUDA operators.

An older H100 three-arm result also exists at
[vllm-bench/_bench_results/REPORT.md](/home/dev/local/prompts-lab/vllm-bench/_bench_results/REPORT.md).
After its documented fix, the seed/default latency ratio was `0.754` and the
seed/vLLM latency ratio was `1.013`. Keep this legacy H100 result separate from
the B200 figures.

## 7. Pointwise

The B200 pointwise audit recorded all 36 planned cells; 35 produced timings.
Every measured cell fired the pointwise heuristic.

All values in this section are performance relative to the raw unseeded
default:

| Cohort | Measured cells | Heuristic | torch.compile | AOT |
|---|---:|---:|---:|---:|
| General pointwise | 17 | 19.351x | 18.508x | 22.145x (5 RoPE cells) |
| vLLM `silu_mul_fp8` | 9 | 1.096x | 1.232x | 1.130x |
| SGLang interleaved SiLU | 9 | 1.257x | 1.420x | 1.036x |

Per kernel:

| Kernel | Heuristic / default | torch.compile / default | AOT / default |
|---|---:|---:|---:|
| SwiGLU | 18.741x | 16.562x | n/a |
| GEGLU | 14.711x | 14.539x | n/a |
| RoPE | 27.943x | 28.253x | 22.145x |
| vLLM `silu_mul_fp8` | 1.096x | 1.232x | 1.130x |
| SGLang `silu_and_mul_interleaved` | 1.257x | 1.420x | 1.036x |

The large general-pointwise gains primarily reflect how poor the tiny
unseeded base configurations are. The RoPE AOT arm is an SM90 table replayed
on B200; the SGLang AOT arm is SM100. One RoPE cell timed out after 300 seconds.

## Quoting Notes

- Keep eager linear-attention results separate from CUDA-graph SGLang KDA
  operation results.
- In linear attention, call `Pre` the "pre-PR compiler heuristic" or
  "pre-change default", not the raw unseeded default.
- `AOT` and `pre-tuned` are configurations produced by full autotuning and
  replayed without including tuning time.
- `torch.compile` means default mode, not `max-autotune`.
- vLLM comparisons use vLLM's shipped Helion configurations, not native vLLM
  CUDA kernels.
- Use all 38 cells for the expanded-seed headline. The 26-cell table is
  deliberately filtered for a greater-than-1.5x early seeded advantage.
- Expanded-seed target sums contain censored lower bounds. Retain target reach
  counts, especially the 5% counts, when interpreting them.
- Use geometric means for cross-shape summaries and retain the per-shape
  tables for outlier discussion.
