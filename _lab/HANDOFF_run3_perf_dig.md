# HANDOFF — Run-3 perf-diff investigation (underperforming shapes)

> Task for the next agent: dig into WHY the seed underperforms on certain shapes
> (esp. **rms_norm, layer_norm, welford, softmax**) and decide, per shape, whether it's a
> **seedable gap** (fixable in the heuristic) or a **ceiling** (kernel-source /
> torch.compile limit). Read this first; it points at the data + the method.

## 0. Environment / wiring (re-prove before trusting anything)
- Interpreter: `/home/calebkim/.conda/envs/helion/bin/python` (conda env `helion`).
- Worktree: `/home/calebkim/helion-new-heuristics/wt-reduction-2` (branch `reduction-heuristics-run2`).
- Run scripts from a **non-checkout cwd** with `PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction-2`.
  Always `assert helion.__file__.startswith(".../wt-reduction-2/")`.
  **PREFIX TRAP:** `"wt-reduction-2".startswith("wt-reduction")` is True — NEVER
  `sys.path.insert(0, ".../wt-reduction")` (that silently runs the run-1 code). Rely on PYTHONPATH only.
- tritonbench operators resolve to the **ORIGINAL checkout** `/home/calebkim/helion-new-heuristics/local/helion/benchmarks/tritonbench/...` (hardcoded meta-path finder; PYTHONPATH can't shadow). Operator edits go there.
- GPUs: 4×H100. `nvidia-smi` first; GPU0 has a co-tenant; **pin `CUDA_VISIBLE_DEVICES` to an idle GPU; NEVER two timing runs on one GPU.** fp32 everywhere (softmax must be forced fp32).

## 1. Where the heuristic (the thing under test) lives
- **`helion/_compiler/autotuner_heuristics/triton.py`** — `TritonReductionHeuristic`. `get_seed_config` routes the branches; `_num_warps(fact)` rnumel ramp; byte-cap constants; `_eviction_policies`; `get_seed_configs` (opt-in portfolio). Every branch keyed on `ReductionFact`, never kernel identity.
- **`helion/autotuner/config_spec.py`** — `ReductionFact` NamedTuple (the workload facts the heuristic keys on).
- **`helion/_compiler/device_ir.py`** — fact population (`register_rollable_reductions` T1, `register_user_tiled_reductions` T2, `_count_reduction_workload`).
- **Kernels:** `examples/rms_norm.py`, `examples/layer_norm.py`, `examples/welford.py` (+ softmax/sum/long_sum/cross_entropy/kl_div/jsd).
- Deliverable docs: `_lab/HANDOFF_run2.md`, `_lab/FINAL_REPORT_run2.md`, `_lab/run2_notebook.md`, `ledger.json["run2"]`; run-1: `_lab/HANDOFF.md` (**§4 TRAPS — read it**), `_lab/FINAL_REPORT.md`.

## 2. Where the Run-3 scripts are (`_lab/harness/`)
- `run3_run1_matrix.py` — **Run 1** (no-autotune): per (kernel,TEST shape) measures C1 Helion-default / C4 Helion-seed / C7 torch.compile-default via `do_bench` median-of-7 + accuracy gate. Checkpoint `_lab/logs/run3/run1_<kernel>.json`.
- `run3_run2_matrix.py` — **Run 2** (autotune): per shape runs the 4 Helion autotune arms (C2 unseeded-quick / C5 seeded-quick / C3 unseeded-full / C6 seeded-full) by shelling out to `run2_productB_driver.py` (cold-cache autotune → per-gen CSV), then **fair-re-benches each winner** with `do_bench`, plus **C8** torch.compile max-autotune. Cell JSON `_lab/logs/run3/run2_<kernel>_<MxN>.json`. Global quick-pass then full-pass; per-shape checkpoint/skip.
- `run3_run2_launch.sh` — single autonomous launcher (2 GPUs, quick→full).
- `run3_report.py` — merges Run-1+Run-2 → `_lab/logs/run3/RUN3_REPORT.md` (per-shape 8-config table + 4-arm×generation tables).
- Reused plumbing (correct wt-reduction-2 wiring): `run2_measure_g.py` (`measure(kernel,M,N)`, `KERNELS`, `get_seed`, `median_do_bench`, `codegen_kind`), `run2_productB_driver.py` (one autotune→CSV), `run2_productB_analyze.py` (per-gen parse).

## 3. Where the results are (`_lab/logs/run3/`)
- `run1_<kernel>.json` — raw µs per TEST shape: `default_lat_us` (C1), `seed_lat_us` (C4), `tc_lat_us` (C7), `g_seed`, `g_default`, `seed_codegen`, `seed_cfg`.
- `run2_<kernel>_<MxN>.json` — `arms.{unseeded,seeded}_{quick,full}.rebench_lat_us_median` + `.reps[].config` (winner) + `.per_gen_best_ms`; `c8_tc_max_us`.
- `pb/<dkernel>_<MxN>_<mode>_<effort>_s0.csv` — per-generation autotune trace (timestamp_s,generation,status,perf_ms,config); `.out` has `[driver] DONE best_config={...}` (the oracle/winner config). NOTE kernel name: softmax→`softmax_two_pass` in `pb/`.
- `RUN3_REPORT.md` — the merged human report. `run2_gpu{2,3}.log` — run logs.

## 4. Headline results (context)
Matched-set geomean G vs tc-default (n=47 shapes, all 8 configs ok): C1 0.479 · **C4 seed 0.939** · C7 1.000 · C2 1.009 · **C5 seed-quick 1.062** · C3 unseeded-full 1.055 · C6 seed-full 1.044 · C8 tc-max 1.009. Seed≈2× the default; seeded-quick ≥ unseeded-full (budget reduction); Helion AT beats tc-max. Caveat: **reps=1** (small diffs are noise).

## 5. THE TARGETS — per-shape diagnostics (do_bench µs; `orcl/seed` = C3/C4)
`Gseed = c7/c4` (vs torch.compile-default). `orcl/seed < ~0.97` ⇒ the autotuner found a config the seed MISSED = **SEEDABLE GAP**. `orcl/seed ≈ 1` ⇒ oracle can't beat the seed either = **CEILING** (investigate the kernel source / why torch.compile wins).

### rms_norm
| shape | c1 dflt | c4 seed | c7 tc | c5 s-q | c3 oracle | c8 tc-max | Gseed | orcl/seed | bucket |
|---|---|---|---|---|---|---|---|---|---|
| (1,131072) | 67.2 | 21.2 | 10.8 | 19.5 | 20.1 | 11.2 | 0.508 | 0.95 | **CEILING** (single-row; tc 2× faster) |
| (256,4096) | 9.0 | 8.8 | 6.1 | 6.0 | 6.0 | 6.4 | 0.692 | **0.68** | **SEEDABLE** (small-M; ~noise-floor 6µs) |
| (2048,1025) | 12.9 | 13.5 | 9.6 | 8.8 | 8.7 | 8.3 | 0.713 | **0.64** | **SEEDABLE** (non-pow2 N) |

### layer_norm
| shape | c1 | c4 | c7 | c5 | c3 | c8 | Gseed | orcl/seed | bucket |
|---|---|---|---|---|---|---|---|---|---|
| (1,131072) | 92.1 | 26.6 | 16.3 | 25.2 | 25.3 | 16.9 | 0.612 | 0.95 | **CEILING** (single-row) |
| (256,4096) | 9.4 | 8.7 | 6.7 | 7.2 | 7.2 | 7.1 | 0.769 | 0.82 | seedable-ish (small-M) |
| (2048,1025) | 15.1 | 15.6 | 12.8 | 8.8 | 8.7 | 10.0 | 0.820 | **0.56** | **SEEDABLE** (non-pow2 N; oracle 1.8× the seed) |
| (2048,2560) | 24.9 | 24.6 | 23.9 | 18.3 | 22.6 | 23.7 | 0.973 | 0.92 | mild seedable |

### welford
| shape | c1 | c4 | c7 | c5 | c3 | c8 | Gseed | orcl/seed | bucket |
|---|---|---|---|---|---|---|---|---|---|
| (262144,7168) | 15616 | 9748 | 6749 | — | — (OOM) | 6742 | 0.692 | — | **CEILING?** wide-N looped apply; tc 1.45×; FULL AUTOTUNE OOMs |
| (262144,5120) | 11113 | 5983 | 4827 | 5944 | — (OOM) | 4830 | 0.807 | — | wide-N looped; quick≈seed (no recovery) |
| (262144,1543) | 4022 | 1604 | 1451 | 1478 | 1477 | 1450 | 0.904 | 0.92 | prime-N OK (correct+fast); slightly < tc |

### softmax
| shape | c1 | c4 | c7 | c5 | c3 | c8 | Gseed | orcl/seed | bucket |
|---|---|---|---|---|---|---|---|---|---|
| (1,131072) | 3143 | 32.7 | 16.0 | 26.2 | 26.6 | 16.5 | 0.490 | 0.81 | **CEILING** (single-row; default 3143µs catastrophic; tc ~2× the seed) |
| (4096,640) | 24.2 | 14.0 | 13.3 | 8.7 | 8.6 | 15.5 | 0.952 | **0.62** | **SEEDABLE** (small-N; oracle 1.6× the seed, also beats tc) |
| (4096,1025) | 57.0 | 20.9 | 24.9 | 17.2 | 16.4 | 20.9 | 1.195 | **0.79** | **SEEDABLE** (non-pow2 N; seed already > tc, but oracle 1.3× the seed) |
| (16384,512) | 56.1 | 36.7 | 33.6 | 32.0 | 31.7 | 36.3 | 0.916 | 0.87 | seedable (small-N) |
| (8192,8192) | 565.9 | 258.7 | 250.0 | 248.7 | 251.4 | 249.9 | 0.967 | 0.97 | ceiling |

(Other rms_norm/layer_norm/welford/softmax shapes are at ceiling, G≈0.97–1.06.)

**The pattern:** SEEDABLE gaps cluster on **small-M, small-N, and non-pow2-N** (the seed picks warps / M-block / blocking the oracle improves on — e.g. softmax (4096,640) the oracle is 1.6× the seed). CEILINGS cluster on **single-row (M=1, huge N — rms_norm / layer_norm / softmax are all ~2× off torch.compile there)** and **wide-N welford looped-apply** (Helion's strategy is structurally ~1.5–2× slower than torch.compile's).

## 7. FIREWALL (do not contaminate)
These are **TEST** shapes, already read once for the final eval. **Do NOT tune the heuristic on sealed TEST shapes.** Already promoted to `in-sample-v2` (tunable): rms_norm (256,4096); welford (262144,5120),(262144,2560). The rest in §5 are sealed TEST — reproduce the gap on `in-sample-v2` analogs or NEW dev shapes (small-M, non-pow2-N) and tune there; only re-read TEST to confirm a finished change.

## 8. Traps (from run-1 §4 / run-2 §7 — all in force)
- **do_bench vs autotuner perf_ms:** the `pb/*.csv` `perf_ms` is the autotuner's INTERNAL bench, NOT comparable to the `do_bench` median-of-7 in run1/rebench. Always fair-re-bench winners with do_bench before comparing.
- **Noise floor:** sub-25µs shapes swing ±25% on the same config — (256,4096)≈6µs, (2048,1025)≈9–13µs are borderline; lift M or median many reps.
- **reps=1** in Run-2 → don't over-read C3-vs-C6 deltas.
- Matched-lever A/B; oracle-is-a-bundle; fp32 assert; commit-don't-push.
