# Kernel inventory + test-shape counts (reduction-seed perf report)

Prepared for the perf report. This catalogs every kernel the reduction seed heuristic has been
tested on, where its source lives, how many test shapes it has, and which dtypes apply. The
benchmarking itself is a separate task — see `BENCH_PROMPT.md`. **The actual (M,N[,…]) shape values
are in `shapes.json` (machine-readable — the bench agent loads this) and `SHAPES.md` (human mirror).**
This file has counts + source/dtype/arm rules; the shape files have the values.

> **SCOPE:** for the **curriculum** the report benches the **`test` split ONLY** (66 shapes across the
> 9 kernels) — the train/val/robustness splits are recorded for reference but are NOT required.
> **transfer / m-reduction / vLLM have no split → bench ALL their shapes.** See `SHAPES.md`.

Heuristic under test: `_TritonReductionSeedBase` (+ the two subclasses) in
`helion/_compiler/autotuner_heuristics/triton.py`, branch `reduction-redesign` (HEAD at time of
writing: `4c201983`), worktree `/home/dev/local/helion-redesign`.

---

## Corpus summary

| corpus | # kernels | # shapes | dtype(s) that apply | source location |
|---|---|---|---|---|
| **curriculum** (the 9 original reduction kernels) | 9 | 331 | fp32 **and** bf16 | `helion/examples/` (in the helion repo) |
| **transfer** (the 8 "robustness" kernels) | 8 | 80 | fp32 **and** bf16 | `kernel_sources/transfer/transfer_kernels.py` (6) + `examples/` (2) |
| **m-reduction** (norm-backward family) | 6 | 16 | fp32 **and** bf16 (recorder used fp16 for 2) | `kernel_sources/mreduction/mreduction_styles_view_only.py` + `examples/` |
| **vLLM** (quant kernels) | 5 | 20 | **NATIVE dtype only** (bf16-in/fp8-out; no fp32/bf16 sweep) | `kernel_sources/vllm/kut/` |
| **synthetic probes** (categorization stress-tests) | 13 | 13 (1 fixed each) | native (fp32); **seed-vs-default only** | `kernel_sources/synthetic_probes/<name>/kernel.py` |
| **adversarial synth** (heuristic-generality tests) | 7 | 1 make_args each, meant to be SWEPT | native (fp32); **seed-vs-default only** | `kernel_sources/adversarial_synth/*.py` |

Shape counts are per-shape (each entry is one (M,N[,…]) point); a full run benches each shape at
each applicable dtype against each baseline (see `BENCH_PROMPT.md`).

**Benchmark arms:** the real corpora (curriculum/transfer/mreduction/vLLM) run 3 arms — seeded /
unseeded Helion default / torch.compile-default. The synthetic + adversarial kernels run **2 arms only
— seed vs unseeded default (no torch.compile)** — the metric there is `G_def` ("what the heuristic
buys" vs the unseeded default), since they have no meaningful torch reference.

---

## 1. Curriculum — 9 original reduction kernels (source: `helion/examples/`)

The recorder runs these at **fp32**; they have train/val/test/robustness splits. For this report we
want **both fp32 and bf16** (the builders are currently hardcoded fp32 — see BENCH_PROMPT §gap).

| kernel | total shapes | train | val | test | robustness | source (examples/) |
|---|---|---|---|---|---|---|
| rms_norm | 40 | 16 | 8 | 8 | 8 | `rms_norm.py` (`rms_norm_fwd`) |
| layer_norm | 40 | 16 | 8 | 8 | 8 | `layer_norm.py` (`layer_norm_fwd`) |
| softmax | 39 | 15 | 8 | 8 | 8 | `softmax.py` (`softmax_two_pass`) |
| welford | 37 | 15 | 7 | 7 | 8 | `welford.py` (`welford`) |
| sum | 36 | 14 | 7 | 7 | 8 | `sum.py` (`sum_kernel`) |
| cross_entropy | 36 | 14 | 7 | 7 | 8 | `cross_entropy.py` (`cross_entropy`) |
| kl_div | 35 | 13 | 7 | 7 | 8 | `kl_div.py` (`kl_div_forward`) |
| jsd | 35 | 13 | 7 | 7 | 8 | `jsd.py` (`jsd_forward`) |
| long_sum | 33 | 12 | 6 | 7 | 8 | `long_sum.py` (`longsum`) |
| **TOTAL** | **331** | | | | | |

Shapes are defined in `helion-redesign/_lab/prompts/shapes_v3_draft.py` (`SHAPES[kernel]` = a dict of
split → list of (M,N)). Kernel → source mapping is in `helion-redesign/_lab/harness/run2_measure_g.py`
(`KERNELS` dict). NOTE: `shapes_v3_draft.SHAPES` also contains 5 extra kernels (logsumexp, log_softmax,
groupnorm, l2_norm, argmax) that are NOT in the active 9-kernel curriculum — ignore them unless asked.

## 2. Transfer — 8 "robustness" kernels (source below)

Recorder runs these at **bf16**; the builders ARE dtype-parameterized (`build(shape, dt)`), so fp32
is a one-line change. Flat shape lists, no split. Shapes: `prompts-lab/transfer/shapes_transfer.py`.

| kernel | shapes | source |
|---|---|---|
| fused_add_rmsnorm | 12 | `kernel_sources/transfer/transfer_kernels.py` (`fused_add_rmsnorm_fwd`) |
| fused_add_layernorm | 12 | `transfer_kernels.py` (`fused_add_layernorm_fwd`) |
| gated_rmsnorm | 12 | `transfer_kernels.py` (`gated_rmsnorm_fwd`) |
| scaled_masked_softmax | 11 | `transfer_kernels.py` (`scaled_masked_softmax_fwd`) |
| cross_entropy_ls_zloss | 11 | `transfer_kernels.py` (`cross_entropy_ls_zloss_fwd`) |
| dynamic_quant | 8 | `transfer_kernels.py` (`dynamic_quant_fwd`) |
| fused_linear_jsd | 7 | `helion/examples/fused_linear_jsd.py` (`jsd_kernel`) |
| grpo | 7 | `helion/examples/grpo_loss.py` |
| **TOTAL** | **80** | |

## 3. M-reduction — 6 norm-backward kernels

Recorder dtypes: `rms_norm_bwd`/`layer_norm_bwd` = **fp16**, the other 4 = **fp32**. For the report we
want fp32+bf16 (builders hardcode dtype — see BENCH_PROMPT §gap). Shapes are inline in
`unified_config_recorder.py` (`_MRED_SHAPES`); kernel source is
`kernel_sources/mreduction/mreduction_styles_view_only.py` (`bias_grad_bwd`, `dyt_bwd`,
`group_norm_bwd`, `instance_norm_bwd`) + `helion/examples/rms_norm.py`, `layer_norm.py` (the two _bwd).

| kernel | shapes | shape form |
|---|---|---|
| bias_grad_bwd | 3 | (M,N) |
| dyt_bwd | 3 | (M,N) |
| rms_norm_bwd | 3 | (M,N) |
| layer_norm_bwd | 3 | (M,N) |
| group_norm_bwd | 2 | (N,C,S,G) 4-D |
| instance_norm_bwd | 2 | (N,C,S) 3-D |
| **TOTAL** | **16** | |

## 4. vLLM — 5 quant kernels (source: `kernel_sources/vllm/kut/`)

**Run in NATIVE dtype only** (bf16 input / fp8 output, some fp32 scales) — these are quantization
kernels; do NOT force fp32 and do NOT add a bf16 sweep, just run each as its builder authors it. 3
arms (seed / default / tc). 4 shapes each. Shapes inline in `unified_config_recorder.py`
(`_VLLM_SHAPES`), builders in `prompts-lab/vllm-bench/bench_arms.py` (`SPECS`), reference impls in
`kernel_sources/vllm/refs.py`.

| kernel | shapes | shape form |
|---|---|---|
| silu_mul_fp8 | 4 | (tok, inter, —) |
| dynamic_per_token_scaled_fp8_quant | 4 | (tok, hidden, —) |
| rms_norm_dynamic_per_token_quant | 4 | (tok, hidden, —) |
| per_token_group_fp8_quant | 4 | (tok, hidden, group=128) |
| rms_norm_per_block_quant | 4 | (tok, hidden, group) |
| **TOTAL** | **20** | |

## 5. Synthetic probes — 13 categorization stress-tests (source: `kernel_sources/synthetic_probes/`)

**CAVEAT: these are NOT realistic perf workloads.** They were authored to stress the Stage-1
FACT/categorization pass (the `probe_assertions.py` Tier-1 suite: 13 pass/fail categorization checks),
NOT to be fast. Each has ONE fixed shape in `make_args()`. **Bench them seed-vs-unseeded-default ONLY
(report `G_def`), no torch.compile** — the signal is "what the heuristic's config buys vs the base
default." Frame as generality/correctness coverage, not headline perf.

11 `p*` (structural coverage: coresident, full-grid, 3-D tile, sequential groups, etc.) + 2 `oos*`
(out-of-scope: jagged-declined, strided-dim0). Each dir has a `kernel.py` exposing a `@helion.kernel`
+ `make_args()` (oos1 uses `get_kernel()`).

## 6. Adversarial synth — 7 heuristic-generality tests (source: `kernel_sources/adversarial_synth/`)

Authored this effort to probe the persist-hold-ceiling heuristic (`_has_store_only_row_reread`). Each
is one `@helion.kernel` + `make_args()`, **meant to be swept over N** (persist-vs-chunk A/B), not run
at a single shape. One (`synth_reread_softmax_VERIFIED_wrongcap.py`) is a GPU-verified 2.24× wrong-cap.
**Bench seed-vs-unseeded-default ONLY (`G_def`), no torch.compile**; show the per-N `G_def` curve (the
sweep is the point). Diagnostic — see `helion-redesign/_lab/redesign/APPLY_REREAD_ADVERSARIAL_CANDIDATES.md`.

| file | what it probes |
|---|---|
| synth_reread_softmax_VERIFIED_wrongcap.py | VERIFIED wrong persist-cap (scalar-out 2-pass, 2.24× at N=49152) |
| synth_reread_variance_NOT_wrongcap.py | boundary: NOT a wrong-cap (row_reread=False, inert) |
| synth_working_set_undercount.py | full-width-output resident tile (2nd-order term) |
| synth_l2_vs_reg_twograph_scalar.py | L2 vs register cross-pass residency |
| synth_livecount_scalar_out.py | body_live_tiles multiplier |
| synth_store_bandwidth.py | output-store bandwidth |
| synth_arith_intensity.py | compute-bound vs memory-bound reduction |

---

## GRAND TOTALS

- **Real-workload kernels:** 9 + 8 + 6 + 5 = **28 kernels**.
- **All-shapes universe** (every split): 331 + 80 + 16 + 20 = **447 shapes** (= the config recorder's
  447 cells). This is the FULL curriculum; the report does NOT bench all of it — see scope.
- **REQUIRED bench set for this report:** curriculum **test split only = 66** + transfer 80 +
  m-reduction 16 + vLLM 20 = **182 shapes**.
- **Dtype-expanded REQUIRED bench cells (the actual run size):** curriculum-test 66×2 (fp32+bf16) +
  transfer 80×2 + m-reduction 16×2 + vLLM 20×1 (native) = **344 (kernel,shape,dtype) cells**, each
  run 3 arms (seed / unseeded-default / tc-default). (bf16 for curriculum/mreduction requires the
  builder change flagged in BENCH_PROMPT.)
- Plus 13 synthetic probes + 7 adversarial — diagnostics, **seed-vs-default (2 arms) only**, framed
  separately.
