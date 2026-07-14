# B200 matmul-family perf characterization (PR #3007 supporting data)

Supporting data + harness for the headline table in
[pytorch/helion#3007](https://github.com/pytorch/helion/pull/3007) — the B200 (sm100) general
budget-**formula** seed (`TritonB200FormulaMatmulHeuristic`) vs. (a) the old `[16,16,16]` compiler
default and (b) `torch.compile(max-autotune-no-cudagraphs)` (cuBLAS/cuBLASLt, except mamba=Triton),
across 48 shapes × 4 kernels. This is the sm100 analog of the H100 run in #3006.

## What's here

| file | what |
|---|---|
| `summary.md` | **the report** — device header, per-kernel + per-type geomeans, full per-shape tables (nothing hidden), M1-vs-M2 cross-check, honest weak spots, honesty caveats, and an adversarial self-audit |
| `results/results.jsonl` | machine-readable raw — one record per (kernel, shape, method), **full per-round `t_us` arrays** for every arm + status; all summaries are derived from these, never stored in place |
| `results/analysis.json` | derived aggregates (per-kernel / per-type / aligned-vs-adversarial / default_source split geomeans) |
| `results/verify_pr3007_table.json` | re-verification of PR #3007's own 18-row formula-vs-table table (geomean 1.20× reproduced) |
| `results/pr3007_table_full_coverage.md` | extends that to ALL 10 tuned buckets of the incumbent table (formula ≥ table on every bucket, no regression) |
| `results/tc_backend_probe.md` | CUDA-profiler verification of which backend `torch.compile` actually ran per shape (39/40 GEMM cells = genuine cuBLAS/cuBLASLt nvjet; only matmul M=1 → Triton GEMV) |
| `DECISIONS.md` | every non-obvious harness decision + the two silent-corruption bugs caught during bring-up |
| `STEP0_GATE.md` | the §5 Step-0 sanity gate result (accuracy / cold-L2 / M1≈M2 on the long kernel) |
| `mmperf/` | the harness (`common`, `kernels`, `compile_probe`, `time_cell`, `sweep`, `analyze`) |

## Method (one paragraph)

3 arms per cell, measured in one process on identical inputs: **seed** (the promoted formula config,
extracted live via the heuristic class — see `DECISIONS.md` D1 for why it's `compiler_default_config`,
not `compiler_seed_configs[0]`), **helion_default** (the incumbent — the `matmul_b200.json` table where
it fires, else the base `[16,16,16]`), and **tc_max_autotune**. Two cold-L2 timing methods, interleaved:
**M1** = CUDA-graph device time (canonical, matches #3006's headline) and **M2** = do_bench-style. L2
flush is **512 MiB** (4× the 126.5 MiB B200 L2 — triton's built-in 256 MiB does *not* evict on B200).
Ratios are the **geomean of per-shape ratios**: `vs default = default/seed`, `vs tc = tc/seed`.

## Reproduce

```bash
# from a Helion checkout on the PR #3007 branch, with a torch+triton+helion venv:
cd _lab/matmul-perf-char
CUDA_VISIBLE_DEVICES=<idle-gpu> PYTHONPATH=<helion-checkout>:$PWD \
  python -m mmperf.sweep --kernels matmul,fp8_gemm,bmm,mamba2_chunk_state \
  --out results/results.jsonl        # checkpoints per cell; --resume to continue
python -m mmperf.analyze             # -> results/analysis.json + console digest

# PR #3007 table re-verification (formula vs incumbent table, its own 18 rows):
python -m mmperf.verify_pr3007_table # -> results/verify_pr3007_table.json
```

Environment this was run on: NVIDIA B200 (sm100, cc 10.0, 148 SMs, 126.5 MiB L2), torch 2.12.0+cu132,
triton 3.7.0. Numbers are cuBLAS/driver/SKU-bound; the **ratios** are the portable claim.
