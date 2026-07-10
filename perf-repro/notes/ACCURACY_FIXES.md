# Accuracy fixes applied to the benchmark branch

An accuracy-only sweep (`tools/acc_sweep.py`) over all kernels found bf16 accuracy-gate
failures in 4 fixable classes + 1 inherent-fp8 class. All fixes are **on the
`perf-report-repro` benchmark branch only** — PR #2996 is intentionally NOT modified (per the
decision to keep the benchmarked heuristic byte-identical to the PR; these are separate,
upstream-worthy correctness improvements to the example kernels / harness, not heuristic changes).

## Fixes (all verified: seed AND default now PASS on the previously-failing cells)

| file | change | why |
|---|---|---|
| `examples/sum.py` | `.sum(-1)` -> `.to(fp32).sum(-1).to(dtype)` | bf16 row-sum lost precision (seed's wide r_block: maxabs 4.0 vs default 0.0). `torch.sum` upcasts bf16 internally — this matches it. |
| `examples/long_sum.py` (`longsum`) | same fp32-accumulate | same, on the ultra-wide reductions. |
| `examples/welford.py` | cast tile `chunk` to fp32 before the within-tile `sum`/`sum(x*x)` | accumulators were already fp32, but the intra-tile reduction summed bf16 (seed FAIL 0.0625 vs default PASS 0.0156). |
| `perf-repro/perf_report_bench.py` (`_rms_bwd_ref`) | ref now uses the SAME `rms_val` (rsqrt) tensor handed to the kernel, not a fp32 recompute | **harness bug**, not a kernel bug: the kernel got a bf16 rsqrt input but was graded against an fp32-rsqrt reference (both seed AND default "failed" maxabs=1.0). Now consistent; torch.compile of tc_ref gets the same rms_val too. |

## NOT fixed (inherent, documented)

- **`per_token_group_fp8_quant` (and siblings) fp8 output**: ~3% of output elements land one fp8
  quantization bucket off (97% bit-exact, scale 100% exact). This is fp8 rounding-tie behavior,
  **identical across seed / default / vLLM**, and cannot be fixed by fp32 accumulation (the OUTPUT
  is fp8 by definition). Perf comparison stays valid (all arms fail identically) — this is the `‡`
  footnote case, kept in the geomean with a note rather than silently loosening the fp8 tolerance.

## Perf caveat

The fp32 casts add a cast op; for these memory-bound reductions the HBM load dominates so the
perf impact is expected to be negligible, but it means the benchmarked example kernels are no
longer byte-identical to PR #2996's examples/. The reduction HEURISTIC (the thing under test) is
unchanged; only the example kernel bodies gained an fp32 accumulate. Worth a line in the report.
