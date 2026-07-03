# B200 handoff — reduction-seed benchmarking kit

Everything a B200 machine needs to stand up the reduction-seed benchmarking infrastructure and see
the **H100 baseline numbers** to target. This is an INDEX, not a plan — the hill-climb itself is
planned separately. The goal here: get the benchmark harness running on B200 and reproduce the
H100 result tables in B200 numbers so they can be compared side by side.

> **Two repos required (both assumed already checked out on the B200 as siblings under one parent,
> e.g. `~/local/`):**
> - **`helion`** on branch **`reduction-perf-report`** — the heuristic code, the curriculum kernels
>   (`helion/examples/`), the `perf-report/` package (numbers + kernel sources + shapes), and the
>   harness (`_lab/perf_report/`).
> - **`prompts-lab`** — supplies two harness adapter modules the bench imports
>   (`transfer/ab_three_arm_transfer.py`, `vllm-bench/bench_arms.py`) plus their kernel/ref helpers.
>
> The harness discovers `prompts-lab` as a sibling of the helion worktree's parent, so keep the same
> `<parent>/helion` + `<parent>/prompts-lab` layout.

---

## 1. The H100 numbers (the targets)

These were measured on one H100 (80GB, 132 SMs, 50MB L2), no autotune. They are what B200 should
reproduce/compare against.

- **`perf-report/PERF_SUMMARY.md`** — the digest: per-kernel `vs default` / `vs torch.compile` /
  `vs vLLM-tuned` tables, grouped by corpus. Read this first.
- **`perf-report/REPORT.md`** — full methodology + per-shape tables + the disaster/weird-shape
  analysis + the x/n·a (accuracy/compile) cells and why.
- **`perf-report/results/`** — raw data: `summary.json` + `SUMMARY.md` (machine/human aggregate) and
  one `<corpus>__<kernel>.json` per kernel with every per-(shape,dtype,arm) cell (latencies, ratios,
  the emitted seed vs default config, accuracy status).
- **Headline:** ~1.9× over the unseeded Helion default, ≈parity with torch.compile overall, ≈parity
  with vLLM's per-shape hand-tuned configs on the quant kernels. Full breakdown in the two docs above.

## 2. All the kernels (the benchmark corpus)

**182 shapes → 344 (shape×dtype) real cells + 20 synthetic/adversarial diagnostics.** The kernel
sources live in TWO places — the split is easy to miss:

- **curriculum (9)** — `rms_norm, layer_norm, softmax, welford, sum, long_sum, cross_entropy,
  kl_div, jsd` — are **NOT copied into `kernel_sources/`**; they are upstream Helion examples in
  **`helion/examples/`**. The kernel→fn mapping is in `_lab/harness/run2_measure_g.py` (`KERNELS`).
- **everything else** is under **`perf-report/kernel_sources/`**:
  - `transfer/` — 8 robustness kernels (`transfer_kernels.py` has 6; `fused_linear_jsd`/`grpo` are
    upstream `helion/examples/`). Shapes in `transfer/shapes_transfer.py`.
  - `mreduction/` — 6 norm-backward kernels (`mreduction_styles_view_only.py`; `rms_norm_bwd`/
    `layer_norm_bwd` are upstream examples).
  - `vllm/` — 5 fp8-quant kernels (`kut/`) + `refs.py` (pure-torch references).
  - `synthetic_probes/` — 13 categorization stress-tests (11 `p*` + 2 `oos*`), one dir each.
  - `adversarial_synth/` — 7 persist-vs-chunk probes.
- **shapes:** `perf-report/shapes.json` (machine-readable; iterate `corpora[*].kernels[*].shapes`).
  `perf-report/SHAPES.md` is the human mirror. Curriculum shape lists also in
  `_lab/prompts/shapes_v3_draft.py`.

## 3. The runnable harness

`_lab/perf_report/`:
- **`perf_report_bench.py`** — the 3-arm bench (seed / unseeded-default / torch.compile) for one
  `(corpus, kernel)`; iterates that kernel's shapes×dtypes, one process per kernel, cold-L2
  `do_bench`, accuracy-gated. Read its module docstring for the method (the footgun controls).
- **`run_sweep.sh <RESULTS_DIR> [corpus_filter]`** — serial driver: one fresh process per kernel,
  resume-safe (skips existing output JSON).
- **`aggregate_report.py <RESULTS_DIR>`** — rolls the per-cell JSON up into `SUMMARY.md` +
  `summary.json` (same tables as the H100 `results/`), so B200 output is directly comparable.

**It imports from `prompts-lab`** (present on B200): `ab_three_arm_transfer` (from
`prompts-lab/transfer/`) and `bench_arms` (from `prompts-lab/vllm-bench/`). `perf_report_bench.py`
adds these to `sys.path` by discovering `prompts-lab` as a sibling of the worktree parent — no action
needed if the layout matches.

### Run recipe (as used on H100)

```bash
# from the helion worktree; PYTHONPATH = the worktree root; run from /tmp
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
  PYTHONPATH=<helion-worktree> \
  <python> <helion-worktree>/_lab/perf_report/perf_report_bench.py \
  --corpus curriculum --kernel rms_norm --out-dir <RESULTS_DIR>
```

Or drive the whole sweep with `run_sweep.sh`. **Note:** `run_sweep.sh` currently hardcodes the H100
worktree path (`WT=/home/dev/local/helion-redesign`) and interpreter, and expects a
`/tmp/perf_worklist.json` (a JSON list of `[corpus, kernel]` pairs — build it by iterating
`shapes.json`). Adjust those paths for the B200 box.

## 4. The heuristic code under test (the base to hill-climb from)

`helion/_compiler/autotuner_heuristics/triton.py` — `_TritonReductionSeedBase` +
`TritonStandardReductionHeuristic` / `TritonUserTiledReductionHeuristic`; the Stage-1 fact is built in
`helion/_compiler/device_ir.py` (`build_reduction_kernel_fact`) and defined in
`helion/autotuner/config_spec.py`. (This is the same heuristic shipped in the reduction PR; the
`reduction-perf-report` branch additionally carries the `_lab/` + `perf-report/` material that the PR
branch drops.)

---

*Not covered here (intentionally): how to make the seed fire on B200, which constants to retune, and
the hill-climb worklist — those are for the B200 planning session.*
