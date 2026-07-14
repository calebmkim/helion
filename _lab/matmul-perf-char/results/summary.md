# Helion matmul-family seed perf characterization — B200 (sm100)

**Seed under test:** `TritonB200FormulaMatmulHeuristic` (PR #3007) — the general budget-FORMULA
re-homed on Blackwell, **promoted as the sm100 compiler default**. It fires on **all 48** shapes ×
dtypes (`seed_fired=true` everywhere; `promoted_is_formula=true` everywhere — the promoted default
equals the formula seed on every cell).

**Baseline being beaten (`helion_default`):** the incumbent — the `matmul_b200.json` lookup **TABLE**
(`TritonB200MatmulHeuristic`, PR #2428) where it fires (in-bucket fp16/bf16 2-D matmul,
`default_source=table`), else the base `~[16,16,16]` compiler default (`default_source=base`). This is
"what ships today before PR #3007," **not** a separate arm.

This is a **measurement + reporting** run: no heuristic edits, no autotuning, no hill-climbing. Every
shape reported; nothing cherry-picked; the worst cells are in the tables.

> **NOTE — every number here is computed by `mmperf/analyze.py` from `results/results.jsonl` (raw
> per-round arrays); all medians/geomeans are derived from those arrays, never stored in their place
> (independently re-verified: 288/288 arm-medians + 192/192 ratio-medians match).** The prose is
> authored. Adversarial-verification synthesis is in "Trust & verification" at the end.

---

## Device header

| field | value |
|---|---|
| GPU | NVIDIA B200 |
| compute capability | sm100 (cc 10.0) |
| SM count | 148 |
| L2 cache | 126.5 MiB |
| **L2-flush buffer** | **512 MiB** (4× L2; triton's built-in do_bench uses only 256 MiB → would NOT evict on B200 — a real portability trap, avoided) |
| dense peak (used for %-peak) | bf16 = 2250 TFLOP/s, fp8-e4m3 = 4500 TFLOP/s |
| clocks | idle 120 MHz before/after; no thermal throttle observed during the run (32→30 °C) |
| arms per cell | `seed` (formula) · `helion_default` (incumbent table-or-base) · `tc_max_autotune` (torch.compile max-autotune-no-cudagraphs) |
| timing | **M1** = CUDA-graph device-time cold-L2 (canonical) · **M2** = do_bench-style cold-L2 · both interleaved, R=7 (15 for <25 µs), full per-round arrays retained |
| coverage | **48 / 48 cells, all 3 arms `ok`** — zero OOM / compile-fail / timeout / accuracy-fail |

---

## The B200 story in two numbers (M1, canonical)

The formula's win splits cleanly by `default_source`:

### (a) Head-to-head on the incumbent TABLE's own tuned turf — `default_source=table` (6 cells)
**Formula beats the incumbent table by geomean `xD_vs_default` = 1.29× (range 1.06–1.45×).**
These are the in-bucket fp16/bf16 matmul shapes where the tuned table actually fires — the hard test.

| shape | type | seed µs | table µs | **xD (table/seed)** | G_vs_tc |
|---|---|---|---|---|---|
| 1024³ | cube | 7.9 | 10.0 | **1.26×** | 0.61 |
| 2048³ | cube | 18.5 | 24.5 | **1.32×** | 0.78 |
| 4096³ | cube | 99.1 | 125.7 | **1.27×** | 0.86 |
| 4096,2048,2048 | rectangular | 29.0 | 30.8 | **1.06×** | 0.85 |
| 3072³ | non-pow2 | 41.5 | 60.2 | **1.45×** | 0.84 |
| 4000,4096,4096 | tile-tail | 98.6 | 137.9 | **1.40×** | 0.89 |

This reproduces PR #3007's central claim (formula subsumes the table, ~1.19–1.27× on cubes) and my
prior independent verification. The formula is never slower than the table on its own turf.

### (b) Coverage — shapes the incumbent DECLINED → `default_source=base` (42 cells)
**Formula rescues the catastrophic `~[16,16,16]` base default by geomean `xD_vs_default` = 18.7×
(range 3.5–50.3×).** These are fp8 (all 10), bmm (all 8), mamba (all 8), and every matmul with a dim
> 4096 or otherwise out-of-bucket (16 of 22 matmul cells). Before PR #3007 these shapes fell to the
base default (up to **135 ms** on the fp32-accum `[16,16,16]` for the vocab GEMM); the formula gives
a good tcgen05-shaped config instead.

**Coverage is the larger half of the B200 story: 42 of 48 cells had no tuned incumbent at all.**

---

## GEMM vs cuBLAS/cuBLASLt (`G_vs_tc` = tc/seed; >1 = seed faster) — M1 canonical

The external yardstick `tc_max_autotune` = `torch.compile(op, mode="max-autotune-no-cudagraphs")`.
**CUDA-profiler-verified** (`results/tc_backend_probe.md`): it selected genuine **cuBLAS/cuBLASLt**
(nvjet_sm100) on **39 of 40** GEMM cells — every fp8 cell (incl. M=1), every bmm cell (incl. tiny),
and every matmul except one. **The single exception is `matmul[1,4096,4096]` (M=1 decode), where
Inductor picked a Triton GEMV, not cuBLAS** — so that cell's `G_vs_tc = 1.40` is a win over Triton,
**not** over cuBLAS. Do NOT read it as "seed beats cuBLAS at decode." (My harness originally hardcoded
the winner label as cuBLAS; this is the corrected, profiled truth.)

| kernel | n | geomean G_vs_tc | min | max |
|---|---|---|---|---|
| matmul (bf16) | 22 | **0.740** | 0.171 (8191³ prime) | 1.396 (1×4096×4096 decode — **vs Triton**, see above) |
| fp8_gemm (e4m3) | 10 | **0.704** | 0.474 (1×4096×4096) | 0.873 (16384×8192×512) |
| bmm (bf16) | 8 | **0.533** | 0.342 | 0.886 |
| **all GEMM (mm+fp8+bmm)** | **40** | **0.684** | — | — |

*Robustness of the 0.684 headline to the decode-mislabel:* the M=1 matmul cell contributes a
*favorable* 1.40 to the matmul geomean via a Triton (not cuBLAS) comparison. Excluding it, all-GEMM
geomean moves 0.684 → **0.672** (matmul alone 0.740 → 0.718); excluding ALL decode cells → **0.672**.
So "~0.68× of cuBLAS on GEMMs" is robust — it does not rest on the mislabeled cell (the correction
moves it the "wrong" way by ~1 point, i.e. the true cuBLAS gap is marginally *larger*, not smaller).

**Reading:** the raw formula-seed config (no autotune) reaches ~68% of hand-tuned cuBLAS across the
GEMM family. It is strongest on large, aligned, compute-bound shapes (ffn/vocab/large-K/tall: 0.85–0.95,
60–69% of peak) and weakest on (i) unaligned/prime dims, (ii) occupancy-starved tiny-M·N deep-K, and
(iii) batched dots (bmm) — see "Honest weak spots" below.

### Aligned-friendly vs adversarial (GEMM, M1 G_vs_tc)
| bucket | n | geomean G_vs_tc |
|---|---|---|
| aligned-friendly | 23 | 0.707 |
| adversarial (non-pow2 / prime / deep-K / small-K / tall / wide / decode) | 17 | 0.655 |

The adversarial shapes cost only ~5 points of geomean vs aligned — the seed degrades gracefully, not
catastrophically, on the hard shapes (with two named exceptions below).

### Per-shape-type geomean (GEMM, M1)
| type | n | G_vs_tc | xD_vs_default |
|---|---|---|---|
| cube | 6 | 0.769 | 7.7 |
| ffn_largeK | 2 | 0.899 | 45.8 |
| ffn_largeN | 2 | 0.744 | 42.5 |
| ffn_large | 1 | 0.856 | 47.7 |
| vocab | 2 | 0.806 | 46.7 |
| attn / attn_qkv | 2 | 0.795 | 43.2 |
| tall_skinny (+extreme) | 3 | 0.885 | 40.9 |
| wide | 1 | 0.903 | 44.2 |
| decode | 4 | 0.811 | 7.8 |
| small_K | 2 | 0.526 | 22.2 |
| deep_K | 1 | 0.259 | 14.5 |
| non_pow2 | 3 | 0.704 | 12.7 |
| non_aligned (5000³) | 1 | 0.739 | 26.7 |
| prime (8191³) | 1 | 0.171 | 9.7 |
| tile_tail (4000·4096·4096) | 1 | 0.889 | 1.4 |
| few_large / many_medium / many_small / gqa / attn_scores / long_seq_attn (bmm) | — | 0.34–0.89 | 5–40 |

---

## mamba2_chunk_state — SEPARATE section (no cuBLAS analog)

mamba has **no library GEMM analog** (it's a fused GEMM+decay). Here `tc_max_autotune` is a **Triton**
kernel (torch.compile of the eager einsum reference), NOT cuBLAS — so this is reported apart from the
GEMM-vs-cuBLAS aggregate.

**The formula seed BEATS the torch.compile-Triton baseline: geomean G_vs_tc = 1.61× (range 1.32–1.90×),**
and rescues the base default by geomean xD = 6.5× (5.2–7.2×). %-of-(bf16-dense-)peak is low (6–9%) —
expected, as this kernel is bandwidth/decay-bound, not a dense GEMM; the FLOP count is the dominant dot
only.

| shape (b,seq,nh,chunk,hd,dstate) | type | G_vs_tc (vs Triton) | xD_vs_default | seed µs | TFLOP/s |
|---|---|---|---|---|---|
| 1,4096,64,256,64,128 | decode_grid | 1.58 | 5.9 | 27.2 | 158 |
| 2,4096,64,256,64,128 | model (1.3B) | 1.76 | 6.5 | 49.1 | 175 |
| 8,4096,80,256,64,128 | model (2.7B) | 1.85 | 7.0 | 222.4 | 193 |
| 8,8192,128,256,64,128 | long_seq (7B) | 1.90 | 7.2 | 692.3 | 199 |
| 2,4096,80,256,64,256 | big_dstate | 1.40 | 6.8 | 114.9 | 187 |
| 4,4096,64,128,64,128 | small_chunk | 1.39 | 5.2 | 127.1 | 135 |
| 8,2048,64,256,128,128 | big_headdim | 1.32 | 7.0 | 179.2 | 192 |
| 16,2048,64,256,64,128 | training_grid | 1.80 | 6.9 | 180.8 | 190 |

---

## Per-shape tables — ALL cells, nothing hidden (M1 canonical)

### matmul — per-shape (M,K,N)

| shape | type | src | seed µs | default µs | tc µs | G_vs_tc | xD_vs_default | TFLOP/s | %peak | acc rel |
|---|---|---|---|---|---|---|---|---|---|---|
| [1, 4096, 4096] | decode | base | 10.2 | 164.0 | 14.2 | 1.396 ᵗ | 16.06 | 3 | 0.1 | 0.00221 |
| [32, 8192, 8192] | decode | base | 31.9 | 335.4 | 27.7 | 0.851 | 10.53 | 135 | 6.0 | 0.00483 |
| [256, 16384, 256] | deep_K | base | 36.9 | 533.5 | 9.6 | 0.259 | 14.48 | 58 | 2.6 | 0.00342 |
| [512, 4096, 16384] | wide | base | 54.5 | 2395.9 | 48.1 | 0.903 | 44.18 | 1262 | 56.1 | 0.00299 |
| [1024, 1024, 1024] | cube | table | 7.9 | 10.0 | 4.9 | 0.614 | 1.26 | 270 | 12.0 | 0.00321 |
| [2048, 2048, 2048] | cube | table | 18.5 | 24.5 | 14.3 | 0.777 | 1.32 | 930 | 41.4 | 0.00426 |
| [3072, 3072, 3072] | non_pow2 | table | 41.5 | 60.2 | 34.9 | 0.841 | 1.45 | 1398 | 62.1 | 0.00345 |
| [4000, 4096, 4096] | tile_tail | table | 98.6 | 137.9 | 87.7 | 0.889 | 1.40 | 1361 | 60.5 | 0.00292 |
| [4096, 2048, 2048] | rectangular | table | 29.0 | 30.8 | 24.8 | 0.849 | 1.06 | 1184 | 52.6 | 0.00405 |
| [4096, 4096, 4096] | cube | table | 99.1 | 125.7 | 84.6 | 0.856 | 1.27 | 1387 | 61.7 | 0.00282 |
| [4096, 4096, 12288] | attn_qkv | base | 294.0 | 13006.2 | 239.1 | 0.814 | 44.26 | 1403 | 62.3 | 0.00538 |
| [4096, 4096, 14336] | ffn_largeN | base | 340.4 | 15188.2 | 280.8 | 0.824 | 44.63 | 1413 | 62.8 | 0.00565 |
| [4096, 4096, 128256] | vocab | base | 2842.2 | 135597.3 | 2457.2 | 0.865 | 47.71 | 1514 | 67.3 | 0.00513 |
| [4096, 14336, 4096] | ffn_largeK | base | 322.5 | 15412.5 | 302.4 | 0.937 | 47.84 | 1492 | 66.3 | 0.00303 |
| [5000, 5000, 5000] | non_aligned | base | 275.5 | 7357.6 | 202.0 | 0.739 | 26.71 | 908 | 40.3 | 0.00508 |
| [8191, 8191, 8191] | prime | base | 4308.0 | 41780.2 | 734.8 | **0.171** | 9.70 | 255 | 11.3 | 0.00342 |
| [8192, 512, 8192] | small_K | base | 84.4 | 2224.3 | 48.2 | 0.568 | 26.35 | 814 | 36.2 | 0.00388 |
| [8192, 5120, 5120] | attn | base | 318.1 | 13356.9 | 246.1 | 0.777 | 41.99 | 1350 | 60.0 | 0.00483 |
| [8192, 8192, 8192] | cube | base | 706.4 | 35554.6 | 632.0 | 0.894 | 50.33 | 1557 | 69.2 | 0.00388 |
| [8192, 8192, 28672] | ffn_large | base | 2609.6 | 124458.8 | 2233.7 | 0.856 | 47.69 | 1475 | 65.5 | 0.00370 |
| [16384, 8192, 512] | tall_skinny | base | 107.1 | 4565.5 | 101.0 | 0.945 | 42.63 | 1283 | 57.0 | 0.00392 |
| [32768, 4096, 256] | tall_skinny_extreme | base | 74.0 | 2327.2 | 62.2 | 0.838 | 31.45 | 928 | 41.2 | 0.00292 |

### fp8_gemm — per-shape (M,K,N)

| shape | type | src | seed µs | default µs | tc µs | G_vs_tc | xD_vs_default | TFLOP/s | %peak | acc rel |
|---|---|---|---|---|---|---|---|---|---|---|
| [1, 4096, 4096] | decode | base | 17.6 | 61.0 | 8.3 | 0.474 | 3.48 | 2 | 0.0 | 0.00000 |
| [32, 8192, 8192] | decode | base | 22.9 | 147.3 | 17.9 | 0.768 | 6.39 | 187 | 4.2 | 0.00030 |
| [3072, 3072, 3072] | non_pow2 | base | 26.1 | 985.5 | 20.9 | 0.786 | 37.65 | 2218 | 49.3 | 0.00084 |
| [4096, 4096, 4096] | cube | base | 65.2 | 2507.6 | 47.1 | 0.726 | 38.44 | 2107 | 46.8 | 0.00036 |
| [4096, 4096, 14336] | ffn_largeN | base | 215.5 | 8718.7 | 144.9 | 0.672 | 40.45 | 2232 | 49.6 | 0.00035 |
| [4096, 4096, 128256] | vocab | base | 1695.6 | 77623.3 | 1273.5 | 0.751 | 45.77 | 2538 | 56.4 | 0.00060 |
| [4096, 14336, 4096] | ffn_largeK | base | 181.3 | 7946.2 | 156.5 | 0.863 | 43.76 | 2653 | 58.9 | 0.00040 |
| [8192, 512, 8192] | small_K | base | 67.8 | 1265.2 | 32.9 | 0.486 | 18.65 | 1014 | 22.5 | 0.00050 |
| [8192, 8192, 8192] | cube | base | 404.2 | 20056.1 | 316.1 | 0.780 | 49.62 | 2720 | 60.5 | 0.00051 |
| [16384, 8192, 512] | tall_skinny | base | 62.9 | 3088.3 | 55.5 | 0.873 | 48.89 | 2184 | 48.5 | 0.00051 |

### bmm — per-shape (B,M,K,N)

| shape | type | src | seed µs | default µs | tc µs | G_vs_tc | xD_vs_default | TFLOP/s | %peak | acc rel |
|---|---|---|---|---|---|---|---|---|---|---|
| [4, 3072, 3072, 3072] | non_pow2 | base | 260.4 | 9830.3 | 136.8 | 0.527 | 37.67 | 890 | 39.6 | 0.00340 |
| [8, 4096, 4096, 4096] | few_large | base | 1195.8 | 48396.0 | 625.3 | 0.523 | 40.48 | 919 | 40.9 | 0.00552 |
| [16, 2048, 128, 128] | gqa | base | 8.3 | 44.4 | 6.1 | 0.742 | 5.37 | 130 | 5.8 | 0.00213 |
| [16, 4096, 128, 4096] | long_seq_attn | base | 296.6 | 2511.8 | 102.2 | 0.342 | 8.47 | 232 | 10.3 | 0.00350 |
| [32, 2048, 128, 2048] | attn_scores | base | 151.8 | 1263.3 | 52.1 | 0.342 | 8.32 | 226 | 10.1 | 0.00323 |
| [32, 2048, 2048, 128] | attn_context | base | 68.1 | 1413.8 | 60.4 | 0.886 | 20.74 | 505 | 22.4 | 0.00385 |
| [64, 512, 512, 512] | many_medium | base | 41.4 | 610.1 | 21.8 | 0.530 | 14.76 | 415 | 18.4 | 0.00405 |
| [128, 256, 256, 256] | many_small | base | 18.0 | 159.8 | 10.4 | 0.581 | 8.89 | 239 | 10.6 | 0.00281 |

ᵗ **matmul[1,4096,4096]** — the tc arm here is a **Triton GEMV**, not cuBLAS (profiler-verified); its
G_vs_tc=1.40 is a win over Triton, not cuBLAS. Every other GEMM cell's tc arm is genuine cuBLAS/cuBLASLt
(nvjet_sm100). See `results/tc_backend_probe.md`.

*Footnote — table arithmetic:* the `G_vs_tc`/`xD_vs_default` columns are **median-of-per-round-ratios**
(the robust estimator, computed from raw per-round arrays), NOT ratio-of-the-printed-median-columns, so
dividing the rounded µs columns won't exactly reproduce the ratio (e.g. fp8 [32,8192,8192]: 17.9/22.9 =
0.78 printed vs 0.768 reported). This is intended, not an error.

(mamba per-shape is in its own section above. M2 medians for every cell are in `results/results.jsonl`.)

---

## M1 vs M2 (methodology cross-check)

Median across all 48 cells of `|G_M2/G_M1 − 1|` = **2.8%** — the two methods agree closely overall.
Divergences concentrate exactly where theory predicts: **short kernels** (seed < ~25 µs), where M2's
`do_bench` includes host launch/guard overhead that M1's cudagraph replay strips.

| cell | seed µs | G_M1 | G_M2 | Δ | note |
|---|---|---|---|---|---|
| matmul 1024³ | 7.9 | 0.614 | 1.000 | +63% | tiny; tc arm's torch.compile guard overhead inflates M2 → flatters seed |
| matmul 256×16384×256 | 36.9 | 0.259 | 0.372 | +44% | deep-K, small kernel |
| fp8 1×4096×4096 | 17.6 | 0.474 | 0.618 | +30% | decode, tiny |
| bmm 128×256³ | 18.0 | 0.581 | 0.746 | +29% | many-small |

On **long** kernels M1≈M2 within a few % (4096³: 1.1% on the seed arm — the Step-0 gate criterion).
**M1 is canonical**; M2's tc arm is inflated on tiny/decode shapes by torch.compile host dispatch, so
`G_vs_tc` under M2 flatters the seed there — do not read the tiny-shape M2 ratios as the seed being
faster than cuBLAS.

---

## `helion_default` failures — surfaced

**Zero arm failures across all 48 cells** (no OOM, no ptxas hang, no compile-fail, no accuracy-fail on
any arm). Notably the catastrophic `~[16,16,16]` base default did **not crash** on the B200 (its 183 GB
HBM absorbs the huge intermediate register pressure) — it was merely **catastrophically slow**: e.g.
matmul vocab (4096×4096×128256) base default = **135.6 ms** vs formula seed 2.84 ms (47.7× / on M1);
fp8 vocab base = 77.6 ms vs 1.70 ms. The harness ran every arm under a 120 s isolated-subprocess
compile timeout + killpg regardless; none was needed. (On H100, or a smaller-HBM SKU, the same defaults
would be more likely to OOM/ptxas-hang — those would have been recorded as `oom`/`timeout` data points.)

---

## Honest weak spots (reported, not fixed — this is a characterization)

1. **matmul 8191³ (prime): G_vs_tc = 0.171** — the worst cell. Seed 4308 µs vs cuBLAS 735 µs (~6×
   slower). A fully-unaligned prime dim defeats the seed's 256×256×32 tile (heavy tail masking, poor
   wave quantization); cuBLAS's nvjet handles it far better. Accuracy still passes (rel 4e-3).
2. **matmul 256×16384×256 (deep-K, tiny M·N): G_vs_tc = 0.259** — occupancy-starved (only 256×256
   output → ~1 tile), the deep-K reduction can't fill 148 SMs. A known structural limit, not a bug.
3. **bmm overall (0.53 geomean)** — batched dots are the seed's weakest family; the per-batch
   independent tcgen05 accumulators don't reach cuBLAS batched efficiency. Worst: attn_scores /
   long_seq_attn (small head_dim K=128 → 0.34). This matches the prior-known bmm codegen ceiling
   (~0.66) and is a **lowering** limitation, not a seed-formula one.
4. **small-K matmul/fp8 (K=512): 0.49–0.57** — compute-light, launch/epilogue-bound.

These are first-class results: on the hardest shapes the raw seed is 2–6× off cuBLAS. The formula is a
*good general default*, not a cuBLAS replacement — see caveats.

---

## Honesty caveats (framing, not bugs) — §8

- **What ships:** with `promote_seed_to_default=True` (the state of PR #3007's branch, confirmed here),
  the formula seed **IS** the no-autotune compiler default on sm100 — so on this branch the "seed" arm
  is genuinely what a user gets with `effort=none`. (Contrast H100, where `promote=False` and the seed
  is only an autotuner *candidate*, not the default.)
- **`static_shapes=True`** (matches the shipped examples): dims are baked as constexpr, enabling
  tail-mask elision on aligned dims — a real specialization edge cuBLAS cannot use. Disclosed.
- **fp8** is vs `torch._scaled_mm(use_fast_accum=True)` (cuBLASLt, scale=1.0) — "~parity-ish" (0.70
  geomean), qualified; not dense-cuBLAS.
- **fp32 not measured** — the primaries here are bf16 (matmul/bmm/mamba) and fp8-e4m3 (fp8_gemm), per
  the curriculum. (My separate PR-#3007 verification found the fp32 seed is a real ~0.24× codegen
  ceiling vs TF32 cuBLAS — out of scope for this bf16/fp8 run.)
- **Numbers are cuBLAS/driver/SKU-bound** — torch 2.12.0+cu132, triton 3.7.0, this specific B200.
- **The incumbent TABLE is being retired** by PR #3007 (demoted `promote=False`); measuring it as the
  `helion_default` on its 6 in-bucket cells is measuring exactly what the formula replaces.

---

## Trust & verification

**Independent self-checks performed on the final dataset (all passed):**
1. **Median re-derivation:** all 288 arm-medians and 192 ratio-medians recompute EXACTLY from the raw
   per-round `t_us` arrays in `results.jsonl` (0 mismatches) — summaries are genuinely derived, not
   stored-in-place, so the data is re-analyzable.
2. **Cold-L2 realism (implied-bandwidth test):** for all 40 GEMM cells, implied HBM bandwidth
   (bytes-moved / seed µs) is **≤ 4.4 TB/s — zero cells exceed the ~8 TB/s B200 HBM peak**, and all
   are far below the >30 TB/s an L2-hot artifact would show. The smallest working set (1024³ = 6 MB,
   which *fits* in the 126.5 MB L2) reads only 0.79 TB/s → the 512 MB flush genuinely evicts it.
3. **cuBLAS-arm legitimacy (profiler-verified per cell):** `tc_max_autotune` ran genuine cuBLAS/cuBLASLt
   (nvjet_sm100) on **39 of 40** GEMM cells — including every fp8 (incl. M=1) and every bmm. The lone
   exception is matmul[1,4096,4096] (M=1 → Triton GEMV), corrected above. Plain eager `torch.matmul` ≈
   tc within 0.6% at 4096³ (both nvjet) — the yardstick is the native library, not a Triton fallback.
4. **Worst-cell FLOP sanity:** 8191³ tc = 1496 TFLOP/s (66% of peak, a plausible cuBLAS number),
   seed 255 TFLOP/s — the 0.171 ratio is a real seed weakness, not a tc measurement error. Decode M=1
   at 3.3 TFLOP/s is correctly latency-bound.
5. **seed ≠ default confirmed:** on the 6 table cells the seed config (formula) and default config
   (table) are distinct and give distinct times (e.g. 4096³ seed 99.1 µs vs table 125.7 µs) — the
   late-binding-closure bug (DECISIONS.md D9) is genuinely fixed.

**Adversarial verification (4 independent skeptic lenses + synthesis)** attacked timing methodology,
cold-L2 realism, ratio faithfulness, and cuBLAS-arm legitimacy. Per-lens verdicts:

| lens | verdict | confidence |
|---|---|---|
| timing methodology | **sound** | 0.82 |
| cold-L2 realism | **cold** | 0.88 |
| ratio faithfulness | **faithful** | 0.82 |
| cuBLAS-arm legitimacy | **mixed** (one real mislabel, below) | — |

**No lens flagged a headline-changing issue.** The three interpretive claims that matter survive because
they rest on arm-vs-arm M1 ratios (the methodology's strongest suit): **(b) 18.7× rescue of the default —
high confidence; (c) ~0.68× of cuBLAS on GEMMs — high, robust to the decode correction; (a) 1.29× over the
incumbent table — medium; (d) 1.6× over mamba's Triton baseline — medium.**

**Real issues surfaced → all now disclosed in this report:**
1. **[fixed above] The tc "winner" label was hardcoded, not per-cell-verified.** I re-profiled: exactly one
   cell — matmul[1,4096,4096] — is vs Triton, not cuBLAS. Corrected in the GEMM section + table.
2. **mamba's 1.6× is over a naive-einsum Triton reference**, not a production kernel — labeled as such (its
   own section; not in any cuBLAS aggregate).
3. **M2 has no discriminating power below ~15 µs** (floors on host+event overhead) — so the M1≈M2 cross-check
   only validates *long* kernels; short/decode-cell M1 numbers rest on M1 alone (already flagged
   `noise_floor`, R=15). Stated in the M1-vs-M2 section.
4. **fp8 tc uses `use_fast_accum=True` vs the seed's fp32 accumulate** — an asymmetry, but a *conservative*
   one (it makes cuBLASLt faster → shrinks G_vs_tc → makes the seed look worse, not better). The
   `fast_accum=False` secondary was not run this pass.
5. **Table ratios are median-of-per-round-ratios**, not ratio-of-printed-medians — footnoted.

**Skeptic overreach the synthesis dismissed** (recorded for completeness): the cold-L2 "streaming-store"
and "sub-15µs overhead-attribution" worries (empirically ruled out — no cell exceeds HBM peak); "rtol=5e-2
could pass a wrong kernel" (every arm's measured rel-err is 10–60× below threshold); the two worst cells
(8191³ prime, deep-K) being "tc cheating" (confirmed genuine seed weaknesses, cuBLAS near-peak there).

> **Synthesis bottom line (verbatim):** *"Trustworthy to show a manager… All four independent adversarial
> lenses returned non-fatal verdicts (timing sound, L2 genuinely cold with decisive BW evidence, ratios
> faithful, cuBLAS-arm mixed), and none flagged a headline-changing issue… its errors are conservative
> (fp8 fast_accum, the loose-but-unused accuracy tol) rather than self-flattering, and the worst cells are
> real seed weaknesses, not baseline cheating."* — the three framing edits it required are applied above.

---

## Reproduce

```
cd /home/dev/helion-matmul-b200/_lab/matmul-perf-char
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/dev/helion-matmul-b200:$PWD \
  /home/dev/.venvs/helion/bin/python -m mmperf.sweep \
  --kernels matmul,fp8_gemm,bmm,mamba2_chunk_state --out results/results.jsonl
/home/dev/.venvs/helion/bin/python -m mmperf.analyze   # -> results/analysis.json + digest
```
Raw data: `results/results.jsonl` (1 record/cell/method, full per-round `t_us` arrays + status).
Harness: `mmperf/{common,kernels,compile_probe,time_cell,sweep,analyze}.py`. Decisions & footguns:
`DECISIONS.md`. Step-0 gate: `STEP0_GATE.md`.
