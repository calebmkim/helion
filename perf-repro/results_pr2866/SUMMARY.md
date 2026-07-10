# PR #2866 pointwise seed heuristic — perf reproduction SUMMARY

Independent reproduction of Helion PR #2866 (pointwise/partially-tiled seed heuristic) at the exact merge commit `89e986e9`. Arms per cell: **seed** (heuristic) / **default** (unseeded `_base_default_config`) / **tc** (`torch.compile`, default mode). Metric = **cold-L2 INTERLEAVED median-of-9 event timing** (round-robin across arms so clock/thermal drift is common-mode and cancels in the ratio) — this reuses `do_bench`'s cold-L2 256MB-flush + 100ms/25ms reps primitives but INTERLEAVES the arms (the PR used sequential `do_bench`; interleaving is a stricter ratio method). Forward-only, single-process, same tensors, no CUDA graphs in the headline (device-time cross-check below).

- `G_tc = tc_us / seed_us` (>1 ⇒ seed beats torch.compile)
- `G_def = default_us / seed_us` (>1 ⇒ seed beats the unseeded default — what the heuristic buys)
- `G_vllm = vllm_us / seed_us` (vLLM only; >1 ⇒ seed beats vLLM's own tuned config)
- All ratios RE-DERIVED here from the raw per-arm µs in the JSON (not from any stored table).


## Headline aggregate (geomean over cells with a VALID same-output comparison)

PR #2866 pointwise seed heuristic. THREE views: **reproduction** = in-sample, the exact PR-table shapes (`pointwise`); **generalization** = out-of-sample, shapes/kernels NEVER fitted — the held-out `test` split, the held-out `dyt` kernel, and expanded lever shapes (`pointwise_gen`); **combined** = both merged. Generalization tests whether the byte/stride/SFU seed interpolates or just fits the curriculum.

A cell joins ALL of a ratio's geomeans when the seed is CLOSE-ENOUGH: it passes accuracy, OR the DEFAULT config makes the IDENTICAL mistake (same acc_detail — a benign kernel/dtype fact: bf16-accumulator margin or fp8 ~1-ULP tie-rounding, shared by the whole family, not a seed-specific error). Close-enough cells are compared against every arm (G_tc, G_def, G_vllm) with a † asterisk. A seed that fails while the DEFAULT is CORRECT (the qk_norm_rope miscompile) is genuinely wrong and excluded from all ratios. See the 'Acc-fail cells' section for the exact per-cell inclusion.

| scope | geo G_tc | geo G_def | geo G_vllm | n(G_tc) | n(G_def) | n(G_vllm) |
|---|---|---|---|---|---|---|
| reproduction_overall | 0.990 | 4.639 | None | 87 | 87 | 0 |
| generalization_overall | 0.975 | 2.791 | None | 57 | 60 | 0 |
| combined_overall | 0.984 | 3.770 | None | 144 | 147 | 0 |
| reproduction_bf16 | 0.990 | 4.639 | None | 87 | 87 | 0 |
| generalization_bf16 | 0.975 | 2.791 | None | 57 | 60 | 0 |
| combined_bf16 | 0.984 | 3.770 | None | 144 | 147 | 0 |

### Device-time (cudagraph) co-equal view — same cells, launch amortized

Headline above is **eager** cold-L2 interleaved timing (the PR's method). Below is the SAME geomean recomputed from per-arm **cudagraph device time** (pure on-device, launch cost removed). Agreement with the eager headline is the proof the numbers are GPU-side truth, not CPU launch overhead. The two differ only where a cell is launch-bound (tiny/decode shapes) — there the device number is the fairer one.

| scope | geo G_tc | geo G_def | n(G_tc) | n(G_def) |
|---|---|---|---|---|
| reproduction_overall_device | 0.990 | 4.644 | 87 | 87 |
| generalization_overall_device | 0.975 | 2.795 | 57 | 60 |
| combined_overall_device | 0.984 | 3.775 | 144 | 147 |

(`geo G_vllm` = seed vs vLLM's own shipped tuned config — THE key comparison for a vLLM kernel; >1 ⇒ seed beats vLLM's hand-tuned config. Only vLLM-family kernels (vllm, vllm_gen, qk_norm_rope_gen) contribute; n(G_vllm) shows how many.)

### Per-corpus

| corpus | geo G_tc | geo G_def | geo G_vllm | n(G_tc) | n(G_def) | n(G_vllm) |
|---|---|---|---|---|---|---|
| pointwise | 0.990 | 4.639 | None | 87 | 87 | 0 |
| pointwise_gen | 0.975 | 2.791 | None | 57 | 60 | 0 |

## Per-kernel combined (in-sample + out-of-sample merged)

Each kernel's `seeded_vs_default` (= geo G_def) and `G_tc`/`min_G_tc` over ALL its cells across both corpora — the single per-kernel number comparable to the PR table.

| kernel | in-sample n | out-of-sample n | seeded_vs_default (geo G_def) | geo G_tc | min G_tc |
|---|---|---|---|---|---|
| bias_gelu | 15 | 5 | 1.131 | 0.957 | 0.859 |
| dyt | 0 | 18 | 1.154 | 0.939 | 0.805 |
| geglu | 17 | 5 | 9.271 | 1.000 | 0.999 |
| heavy_transcendental_1d | 1 | 5 | 2.849 | 1.007 | 0.989 |
| relu_squared | 12 | 4 | 13.665 | 0.999 | 0.997 |
| residual_add | 16 | 5 | 1.285 | 0.979 | 0.965 |
| rope_fwd | 2 | 6 | 16.800 | 1.085 | 1.005 |
| swiglu | 23 | 7 | 9.276 | 1.000 | 0.998 |
| transposed_out_add | 1 | 6 | 1.170 | 0.970 | 0.935 |

## Per-shape disasters (realistic shape with G_tc < 0.75)

_(none)_

## (A) Per-(kernel, dtype) geomeans


### pointwise

| kernel | dtype | geo G_tc | geo G_def | shapes | acc-pass | min G_tc |
|---|---|---|---|---|---|---|
| bias_gelu | bf16 | 0.953 | 1.129 | 15 | 15 | 0.859 |
| geglu | bf16 | 1.000 | 9.288 | 17 | 17 | 0.999 |
| heavy_transcendental_1d | bf16 | 1.003 | 3.976 | 1 | 1 | 1.003 |
| relu_squared | bf16 | 0.999 | 13.728 | 12 | 12 | 0.997 |
| residual_add | bf16 | 0.981 | 1.283 | 16 | 16 | 0.970 |
| rope_fwd | bf16 | 1.068 | 17.172 | 2 | 2 | 1.008 |
| swiglu | bf16 | 1.000 | 9.286 | 23 | 23 | 0.998 |
| transposed_out_add | bf16 | 1.030 | 1.049 | 1 | 1 | 1.030 |

### pointwise_gen

| kernel | dtype | geo G_tc | geo G_def | shapes | acc-pass | min G_tc |
|---|---|---|---|---|---|---|
| bias_gelu | bf16 | 0.970 | 1.136 | 5 | 5 | 0.904 |
| dyt | bf16 | 0.939 | 1.154 | 18 | 18 | 0.805 |
| geglu | bf16 | 1.000 | 9.216 | 5 | 5 | 1.000 |
| heavy_transcendental_1d | bf16 | 1.008 | 2.665 | 5 | 5 | 0.989 |
| relu_squared | bf16 | 0.999 | 13.479 | 4 | 4 | 0.999 |
| residual_add | bf16 | 0.974 | 1.289 | 5 | 5 | 0.965 |
| rope_fwd | bf16 | 1.102 | 16.653 | 6 | 6 | 1.005 |
| swiglu | bf16 | 1.000 | 9.242 | 7 | 7 | 1.000 |
| transposed_out_add | bf16 | 0.960 | 1.191 | 6 | 6 | 0.935 |

## Launch-overhead cross-check (seed arm: do_bench vs cold-L2 cudagraph)

For every cell we also time the seed arm under a cold-L2 CUDA-graph replay (pure GPU device time, launch amortized). If cold-L2 `do_bench` ≈ cudagraph, CPU launch overhead is hidden behind the 256MB L2 flush and the headline number is GPU-side truth. Cells listed below diverge by > 15% — the canary that launch cost is leaking into that measurement (see notes/LAUNCH_OVERHEAD_NOTE.md).


**Scoring decision (device-time override, gate = 5%).** A cell whose eager vs cudagraph divergence exceeds 5% on ANY arm is BELOW the eager timing floor — its do_bench-style ratio is dominated by per-arm-variable CPU launch tax, not compute. Those cells (below) are scored on **cudagraph DEVICE time** in the headline; all other cells stay on eager (the PR's method). 5% matches the harness round-to-round spread gate — 'if eager and device disagree by more than the timing noise floor, trust device.' Both ratios shown for full transparency; the device column is the one used in the headline geomeans.

| corpus | kernel | shape | dtype | worst arm div | eager G_def | **device G_def** | eager G_tc | **device G_tc** |
|---|---|---|---|---|---|---|---|---|
| pointwise | transposed_out_add | [2048, 512] | bf16 | 64% | 0.704 | **1.049** | 1.395 | **1.030** |
| pointwise_gen | heavy_transcendental_1d | [65536] | bf16 | 55% | 1.257 | **1.344** | 2.097 | **1.041** |
| pointwise_gen | heavy_transcendental_1d | [262144] | bf16 | 12% | 2.100 | **2.154** | 1.104 | **1.003** |
| pointwise_gen | dyt | [8192, 8192] | bf16 | 6% | 1.152 | **1.197** | 0.975 | **0.954** |

| corpus | kernel | shape | dtype | do_bench µs | cudagraph µs | launch_frac |
|---|---|---|---|---|---|---|
| pointwise | transposed_out_add | [2048, 512] | bf16 | 17.5 | 8.4 | 0.517 |

## Acc-fail cells — perf measured (0 cells)

A cell fails the strict accuracy gate but is treated as **CLOSE-ENOUGH** — and its speed IS compared against ALL arms (G_tc, G_def, G_vllm) — when the DEFAULT config makes the IDENTICAL mistake (same acc_detail). Default-matches-seed means the miss is a benign kernel/dtype FACT the whole family shares (bf16-accumulator margin on rms_norm_bwd; fp8 tie-rounding — ~1 ULP on ~3% of elements — on per_token_group), NOT a seed-specific wrong answer, so timing all arms is apples-to-apples (†). A seed that fails while the DEFAULT is CORRECT (the qk_norm_rope tile-size miscompile: seed maxabs~2, default fine) is genuinely wrong and is EXCLUDED from every ratio.

- **0** close-enough cells (default makes the same mistake) → included in all applicable ratios with a † asterisk.
- **0** seed-only-failure cells → excluded from all ratios (real wrong answer).
- **0** cells where the DEFAULT config could not compile (ptxas/Inductor timeout) — the seed compiled and ran, but no default ratio is possible (a point in the seed's favor, noted not counted).

_(none)_

## x / n/a cells (5 arm-level entries)

| corpus | kernel | shape | dtype | arm | reason |
|---|---|---|---|---|---|
| pointwise_gen | rope_fwd | [2, 32, 2048, 256] | bf16 | default | compile-fail:timeout |
| pointwise_gen | rope_fwd | [2, 32, 2048, 256] | bf16 | tc | compile-fail:InductorError |
| pointwise_gen | rope_fwd | [1, 32, 4096, 128] | bf16 | tc | compile-fail:InductorError |
| pointwise_gen | rope_fwd | [1, 16, 8192, 128] | bf16 | tc | compile-fail:InductorError |
| pointwise_gen | rope_fwd | [4, 8, 4096, 128] | bf16 | tc | compile-fail:InductorError |
