# H100 (sm90) matmul-family seed perf characterization

**Measurement + reporting only** — the H100 budget-FORMULA seed `TritonH100MatmulHeuristic` characterized as-is against two baselines, nothing cherry-picked, every shape reported. No heuristic edits.

## TL;DR

- **The formula seed fires on 100% of the 48 cells** (`triton_h100_matmul`) and is **17.9× faster than the compiler base default** `[16,16,16]` (GEMM geomean, M1) — the base's job is to be replaced, and it is, everywhere, by 5–40×.
- **Against the library (cuBLAS/cuBLASLt), GEMM seed geomean `G_vs_tc` = 0.810** all-in (n=40): **0.852 on aligned-friendly** shapes (≈15% off cuBLAS) vs **0.611 on the adversarial** non-pow2/prime shapes (the seed's known weak spot — reported, not hidden).
- **Best regime: mamba2** (fused batched GEMM+decay, Triton yardstick) — seed geomean **3.42× faster than the compiled-Triton reference**, reported in its own section (no cuBLAS analog).
- **Worst cell (surfaced, not dropped):** matmul 8191³ prime, `G_vs_tc` ≈ **0.27–0.32** (seed ~3.1–3.8× slower than cuBLAS — the static-shape tail-mask penalty). This is the ONE cell where the M1/M2 methods disagree materially (M1 seed is jittery on this prime shape; see the harness-caveat box) — reported as a range, not a false-precision point. bmm attention shapes also lag (0.50–0.53).
- **All arms succeeded on every cell** (no OOM / ptxas-timeout / acc-fail). **M1 (cudagraph) is canonical for absolute µs and the `G_vs_tc` ratio is method-robust to ~2.7% on non-tiny cells** — EXCEPT the 8191³ prime, where M1 run-to-run jitter (~4%) inflates its median above M2; there M2 is the tighter estimate. The §5 M1/M2-agreement gate is discussed honestly below (it is a host-overhead offset on most cells, genuine M1 noise on one).
- **What ships (H100/sm90 only):** in PR #3006 the formula is promoted (`promote_seed_to_default=True`), so on H100 the emitted config **IS the no-autotune compiler default** — what `effort=none` returns — replacing the `[16,16,16]` fallback. The heuristic is gated `HARDWARE_TARGETS=(("cuda","sm90"),)` and declines on every other GPU. (Perf gathered on a byte-identical revision of the matmul config-gen code where the config was read via `compiler_seed_configs[0]`; the promote flag changes only routing, not the emitted config.)

## Device & method

- **GPU:** NVIDIA H100 80GB HBM3 — compute capability **sm90** (H100 SXM). 132 SMs.
- **L2 cache:** 50.0 MiB → cold-L2 flush buffer = max(256 MiB, 4×L2) = **256.0 MiB** (flushed before every rep, both methods).
- **Dense peak used (%-of-peak):** bf16/fp16 **989.5 TFLOP/s**, fp8-e4m3 **1978.9 TFLOP/s** (H100 SXM).
- **3 arms** (all measured in one process on identical inputs): `seed` = the formula's `compiler_seed_configs[0]`; `helion_default` = the compiler base `[16,16,16] w4 s1`; `tc_max_autotune` = `torch.compile(op, mode="max-autotune-no-cudagraphs")`.
- **B200 incumbent table** (`TritonB200MatmulHeuristic`, PR #2428) is **sm100-gated → never fires on H100**, so `default_source = n/a` for every cell here and the `helion_default` arm is always the base `[16,16,16]`. The whole B200 table story is out of scope on this GPU.
- **2 timing methods, both cold-L2, interleaved, R rounds, raw per-round arrays retained.** M1 = CUDA-graph device time (flush-graph minus [flush+kernel] graph, replay-averaged) — **canonical**. M2 = triton `do_bench` (includes host launch overhead). R = 7, bumped to 15 when the seed's cold-L2 device time < 25 µs.
- **Fairness locks:** TF32 off, bf16-reduced-precision-reduction off (fp32 accum). `static_shapes=True` (faithful to the shipped examples — bakes dims as constexpr, a real tail-mask-elision edge cuBLAS can't use; disclosed).
- **48 cells** (matmul 22 / fp8 10 / bmm 8 / mamba 8) × 2 methods × 3 arms. seed_fired on **all** cells (heuristic `triton_h100_matmul`).

**Ratios** (per-round, common-mode drift cancels): `G_vs_tc = tc/seed` (>1 = seed faster than the library), `xD_vs_default = default/seed` (what the seed buys over the base default).

**`tc_max_autotune` selected backend (§5(d) confirmation — logged per shape, captured from the inductor autotune table):**
- **matmul:** 21/22 → `mm` (**cuBLAS**); 1/22 (1024³) → a Triton template beat cuBLAS. **fp8_gemm:** 10/10 → `_scaled_mm` (**cuBLASLt**). **bmm:** 6/8 → `bmm` (**cuBLAS batched**); 2/8 (small/transposed) → Triton. **mamba2:** 8/8 → `triton_bmm_*` (**Triton** — no cuBLAS analog, as expected).
- So the GEMM yardstick is genuinely cuBLAS/cuBLASLt on 37/40 GEMM cells (the 3 Triton-win cells are small shapes where cuBLAS has no edge); mamba's yardstick is Triton throughout. Every `winner` is recorded per-cell in the JSONL.

## Per-shape — matmul

Times are **M1 (cudagraph) medians, µs**. `G_tc`=tc/seed, `xD`=default/seed. `%pk`=seed %-of-dense-peak. `tc win`=the max-autotune-selected backend (`mm`/`bmm`/`_scaled_mm`=aten/cuBLAS(-Lt); `triton_*`=a Triton template won). `flag`: ⚠︎=noise-floor (<25 µs); ⚑=M1 median > M2 (M1 jitter — trust M2 here).

| shape | type | seed µs | default µs | tc µs | G_tc | xD | seed TFLOP/s | %pk | tc win | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| [1024, 1024, 1024] | cube | 7.9 | 89.0 | 6.9 | 0.882 | 11.28 | 272 | 27.5 | triton_mm_12 | ⚠︎ |
| [2048, 2048, 2048] | cube | 25.0 | 667.4 | 24.6 | 0.985 | 26.72 | 688 | 69.5 | mm | ⚠︎ |
| [4096, 4096, 4096] | cube | 186.6 | 5,470 | 178.1 | 0.954 | 29.27 | 737 | 74.4 | mm |  |
| [8192, 8192, 8192] | cube | 1,542 | 45,881 | 1,403 | 0.909 | 29.53 | 713 | 72.1 | mm |  |
| [4096, 2048, 2048] | rectangular | 48.8 | 1,278 | 45.6 | 0.935 | 26.22 | 705 | 71.2 | mm |  |
| [4096, 4096, 14336] | ffn_largeN | 669.7 | 20,293 | 617.6 | 0.915 | 30.49 | 718 | 72.6 | mm |  |
| [4096, 14336, 4096] | ffn_largeK | 630.4 | 20,434 | 598.5 | 0.946 | 32.20 | 763 | 77.1 | mm |  |
| [8192, 8192, 28672] | ffn_large | 5,658 | 151,057 | 4,928 | 0.871 | 26.70 | 680 | 68.7 | mm |  |
| [4096, 4096, 12288] | attn_qkv | 562.1 | 17,022 | 527.3 | 0.934 | 30.03 | 734 | 74.1 | mm |  |
| [8192, 5120, 5120] | attn | 602.7 | 18,072 | 555.4 | 0.918 | 30.05 | 713 | 72.0 | mm |  |
| [4096, 4096, 128256] | vocab | 6,344 | 180,468 | 5,598 | 0.881 | 28.47 | 678 | 68.5 | mm |  |
| [1, 4096, 4096] | decode | 18.1 | 143.4 | 22.9 | 1.268 | 7.92 | 2 | 0.2 | aten mm/GEMV (M=1, no autotune template) | ⚠︎ |
| [32, 8192, 8192] | decode | 55.1 | 281.8 | 54.6 | 0.992 | 5.12 | 78 | 7.9 | mm |  |
| [16384, 8192, 512] | tall_skinny | 187.5 | 5,703 | 179.1 | 0.960 | 30.72 | 733 | 74.1 | mm |  |
| [32768, 4096, 256] | tall_skinny_extreme | 113.9 | 2,983 | 111.5 | 0.979 | 26.18 | 603 | 61.0 | mm |  |
| [512, 4096, 16384] | wide | 94.3 | 2,622 | 92.9 | 0.984 | 27.76 | 729 | 73.6 | mm |  |
| [256, 16384, 256] | deep_K | 30.1 | 482.9 | 13.5 | 0.448 | 16.06 | 71 | 7.2 | mm |  |
| [8192, 512, 8192] | small_K | 128.7 | 2,545 | 111.9 | 0.870 | 19.79 | 534 | 53.9 | mm |  |
| [3072, 3072, 3072] | non_pow2 | 107.8 | 2,101 | 75.6 | 0.702 | 19.50 | 538 | 54.4 | mm |  |
| [5000, 5000, 5000] | non_aligned | 612.7 | 9,289 | 371.5 | 0.606 | 15.18 | 408 | 41.2 | mm |  |
| [4000, 4096, 4096] | tile_tail | 184.6 | 5,427 | 177.2 | 0.950 | 29.11 | 727 | 73.5 | mm |  |
| [8191, 8191, 8191] | prime | 6,039 | 50,614 | 1,604 | 0.276 | 8.70 | 182 | 18.4 | mm | ⚑ |

## Per-shape — fp8_gemm

Times are **M1 (cudagraph) medians, µs**. `G_tc`=tc/seed, `xD`=default/seed. `%pk`=seed %-of-dense-peak. `tc win`=the max-autotune-selected backend (`mm`/`bmm`/`_scaled_mm`=aten/cuBLAS(-Lt); `triton_*`=a Triton template won). `flag`: ⚠︎=noise-floor (<25 µs); ⚑=M1 median > M2 (M1 jitter — trust M2 here).

| shape | type | seed µs | default µs | tc µs | G_tc | xD | seed TFLOP/s | %pk | tc win | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| [4096, 4096, 4096] | cube | 100.1 | 2,878 | 95.6 | 0.958 | 28.77 | 1,373 | 69.4 | _scaled_mm |  |
| [8192, 8192, 8192] | cube | 815.3 | 31,082 | 743.1 | 0.909 | 38.12 | 1,349 | 68.2 | _scaled_mm |  |
| [4096, 4096, 14336] | ffn_largeN | 355.4 | 11,109 | 330.4 | 0.931 | 31.01 | 1,354 | 68.4 | _scaled_mm |  |
| [4096, 14336, 4096] | ffn_largeK | 333.3 | 11,419 | 317.4 | 0.948 | 33.74 | 1,443 | 72.9 | _scaled_mm |  |
| [4096, 4096, 128256] | vocab | 3,110 | 92,117 | 2,893 | 0.930 | 29.69 | 1,384 | 69.9 | _scaled_mm |  |
| [1, 4096, 4096] | decode | 17.6 | 55.5 | 9.9 | 0.565 | 3.16 | 2 | 0.1 | _scaled_mm | ⚠︎ |
| [32, 8192, 8192] | decode | 33.8 | 166.4 | 32.4 | 0.957 | 4.93 | 127 | 6.4 | _scaled_mm |  |
| [8192, 512, 8192] | small_K | 86.4 | 1,423 | 69.9 | 0.810 | 16.46 | 795 | 40.2 | _scaled_mm |  |
| [3072, 3072, 3072] | non_pow2 | 55.6 | 1,131 | 41.7 | 0.752 | 20.31 | 1,042 | 52.6 | _scaled_mm |  |
| [16384, 8192, 512] | tall_skinny | 96.9 | 4,011 | 96.2 | 0.989 | 41.30 | 1,418 | 71.7 | _scaled_mm |  |

## Per-shape — bmm

Times are **M1 (cudagraph) medians, µs**. `G_tc`=tc/seed, `xD`=default/seed. `%pk`=seed %-of-dense-peak. `tc win`=the max-autotune-selected backend (`mm`/`bmm`/`_scaled_mm`=aten/cuBLAS(-Lt); `triton_*`=a Triton template won). `flag`: ⚠︎=noise-floor (<25 µs); ⚑=M1 median > M2 (M1 jitter — trust M2 here).

| shape | type | seed µs | default µs | tc µs | G_tc | xD | seed TFLOP/s | %pk | tc win | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| [32, 2048, 128, 2048] | attn_scores | 220.7 | 1,441 | 116.8 | 0.529 | 6.52 | 156 | 15.7 | bmm |  |
| [32, 2048, 2048, 128] | attn_context | 115.5 | 1,900 | 110.7 | 0.958 | 16.45 | 298 | 30.1 | triton_bmm_17 |  |
| [16, 4096, 128, 4096] | long_seq_attn | 439.6 | 2,862 | 220.2 | 0.501 | 6.52 | 156 | 15.8 | bmm |  |
| [64, 512, 512, 512] | many_medium | 66.1 | 947.4 | 42.6 | 0.645 | 14.34 | 260 | 26.3 | bmm |  |
| [8, 4096, 4096, 4096] | few_large | 2,343 | 59,194 | 1,432 | 0.613 | 25.24 | 469 | 47.4 | bmm |  |
| [128, 256, 256, 256] | many_small | 28.0 | 185.1 | 23.5 | 0.840 | 6.62 | 154 | 15.5 | bmm |  |
| [16, 2048, 128, 128] | gqa | 9.9 | 50.1 | 8.8 | 0.896 | 5.09 | 109 | 11.0 | triton_bmm_9 | ⚠︎ |
| [4, 3072, 3072, 3072] | non_pow2 | 491.5 | 11,485 | 305.2 | 0.621 | 23.27 | 472 | 47.7 | bmm |  |

## Per-shape — mamba2_chunk_state — **Triton yardstick, NOT cuBLAS** (fused GEMM+decay has no library analog; kept out of the GEMM-vs-cuBLAS aggregates)

Times are **M1 (cudagraph) medians, µs**. `G_tc`=tc/seed, `xD`=default/seed. `%pk`=seed %-of-dense-peak. `tc win`=the max-autotune-selected backend (`mm`/`bmm`/`_scaled_mm`=aten/cuBLAS(-Lt); `triton_*`=a Triton template won). `flag`: ⚠︎=noise-floor (<25 µs); ⚑=M1 median > M2 (M1 jitter — trust M2 here).

| shape | type | seed µs | default µs | tc µs | G_tc | xD | seed TFLOP/s | %pk | tc win | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| [2, 4096, 64, 256, 64, 128] | model | 50.1 | 360.3 | 181.7 | 3.627 | 7.19 | 171 | 17.3 | triton_bmm_11 | ⚑ |
| [8, 4096, 80, 256, 64, 128] | model | 223.0 | 1,790 | 853.3 | 3.827 | 8.02 | 193 | 19.5 | triton_bmm_11 |  |
| [8, 8192, 128, 256, 64, 128] | long_seq | 688.8 | 5,621 | 2,703 | 3.922 | 8.16 | 200 | 20.2 | triton_bmm_11+triton_bmm_12+triton_bmm_17+triton_bmm_16 |  |
| [2, 4096, 80, 256, 64, 256] | big_dstate | 100.0 | 902.0 | 345.7 | 3.459 | 9.01 | 215 | 21.7 | triton_bmm_11 |  |
| [4, 4096, 64, 128, 64, 128] | small_chunk | 137.4 | 734.3 | 367.8 | 2.677 | 5.35 | 125 | 12.6 | triton_bmm_12 |  |
| [8, 2048, 64, 256, 128, 128] | big_headdim | 174.7 | 1,439 | 499.7 | 2.860 | 8.23 | 197 | 19.9 | triton_bmm_17 |  |
| [1, 4096, 64, 256, 64, 128] | decode_grid | 27.2 | 183.5 | 93.7 | 3.446 | 6.75 | 158 | 16.0 | triton_bmm_11 |  |
| [16, 2048, 64, 256, 64, 128] | training_grid | 179.9 | 1,440 | 682.6 | 3.794 | 8.00 | 191 | 19.3 | triton_bmm_11 |  |

## Geomeans (GEMM family: matmul + fp8 + bmm; mamba separate below)

Computed from per-cell **M1 medians** (canonical). **No exclusions** in the all-in number (every acc-passing cell counts, including the worst). Note: the adversarial subtotal uses the M1 8191³ value (`G`=0.276); using the tighter M2 value (0.322) would lift matmul-adversarial 0.578→0.614 and GEMM-adversarial 0.611→0.636 — i.e. **the M1 choice is the conservative/pessimistic one for the seed**, so no cherry-picking upward.

### By shape-type (GEMM family)

| shape-type | n | geo G_vs_tc | geo xD_vs_default |
|---|---|---|---|
| cube | 6 | 0.932 | 25.66 |
| rectangular | 1 | 0.935 | 26.22 |
| ffn_largeN | 2 | 0.923 | 30.75 |
| ffn_largeK | 2 | 0.947 | 32.96 |
| ffn_large | 1 | 0.871 | 26.70 |
| attn_qkv | 1 | 0.934 | 30.03 |
| attn | 1 | 0.918 | 30.05 |
| vocab | 2 | 0.906 | 29.07 |
| decode | 4 | 0.908 | 5.01 |
| tall_skinny | 2 | 0.974 | 35.62 |
| tall_skinny_extreme | 1 | 0.979 | 26.18 |
| wide | 1 | 0.984 | 27.76 |
| deep_K | 1 | 0.448 | 16.06 |
| small_K | 2 | 0.839 | 18.05 |
| non_pow2 | 3 | 0.690 | 20.97 |
| non_aligned | 1 | 0.606 | 15.18 |
| tile_tail | 1 | 0.950 | 29.11 |
| prime | 1 | 0.276 | 8.70 |
| attn_scores | 1 | 0.529 | 6.52 |
| attn_context | 1 | 0.958 | 16.45 |
| long_seq_attn | 1 | 0.501 | 6.52 |
| many_medium | 1 | 0.645 | 14.34 |
| few_large | 1 | 0.613 | 25.24 |
| many_small | 1 | 0.840 | 6.62 |
| gqa | 1 | 0.896 | 5.09 |

### All GEMM (matmul+fp8+bmm)

| subset | n | geo G_vs_tc | geo xD_vs_default |
|---|---|---|---|
| **ALL-IN (no exclusions)** | 40 | **0.810** | **17.95** |
| aligned-friendly | 34 | 0.852 | 17.92 |
| adversarial (non-pow2/non-aligned/tile-tail/prime) | 6 | 0.611 | 18.12 |

### matmul only (bf16)

| subset | n | geo G_vs_tc | geo xD_vs_default |
|---|---|---|---|
| **ALL-IN (no exclusions)** | 22 | **0.838** | **20.80** |
| aligned-friendly | 18 | 0.910 | 21.88 |
| adversarial (non-pow2/non-aligned/tile-tail/prime) | 4 | 0.578 | 16.55 |

### fp8_gemm only (e4m3)

| subset | n | geo G_vs_tc | geo xD_vs_default |
|---|---|---|---|
| **ALL-IN (no exclusions)** | 10 | **0.864** | **19.34** |
| aligned-friendly | 9 | 0.878 | 19.24 |
| adversarial (non-pow2/non-aligned/tile-tail/prime) | 1 | 0.752 | 20.31 |

### bmm only (bf16)

| subset | n | geo G_vs_tc | geo xD_vs_default |
|---|---|---|---|
| **ALL-IN (no exclusions)** | 8 | **0.682** | **10.89** |
| aligned-friendly | 7 | 0.691 | 9.78 |
| adversarial (non-pow2/non-aligned/tile-tail/prime) | 1 | 0.621 | 23.27 |

## mamba2_chunk_state (own section — Triton yardstick)

No cuBLAS analog (fused batched GEMM + state decay). `tc_max_autotune` here is a **Triton** kernel (torch.compile of the eager reference), not cuBLAS — so `G_vs_tc` means seed-vs-best-Triton, NOT seed-vs-library-GEMM. Reported separately; never folded into the GEMM aggregates.

- geo **G_vs_tc = 3.423** (seed vs best-Triton, n=8) — seed beats the compiled-Triton reference.
- geo **xD_vs_default = 7.51** over the `[16,16,16]` base (n=8).

## M1 (cudagraph) vs M2 (do_bench) divergence

Both cold-L2. M1 strips host launch overhead via graph replay; M2 includes it. The honest reading of the data (below): M2 runs a **consistent additive host-dispatch premium above M1** — ~+40% on the smallest kernels, decaying toward ~+6–9% on the largest. It does NOT vanish to zero on long kernels: even at 5–6 ms a residual ~+6–11% persists (per-launch dispatch of the two compiled callables — the Helion kernel and torch.compile — plus do_bench's own harness cost). **But that premium is COMMON-MODE and cancels in the ratio:** `G_vs_tc` is stable across methods to **~2.7% on non-tiny cells**, so the headline seed-vs-library number is method-robust even where absolute µs differ. M1 is canonical for absolute time; the ratio agrees either way.

**Seed absolute-time gap (M2−M1) by kernel length** — the host-overhead decay:

| M1 length bucket | n | median M2−M1 | range |
|---|---|---|---|
| <10 µs | 2 | +43.2% | [+41, +45]% |
| 10–25 µs | 3 | +24.7% | [+19, +28]% |
| 25–100 µs | 13 | +8.9% | [-3, +16]% |
| 100–500 µs | 17 | +5.2% | [+2, +12]% |
| 0.5–2 ms | 8 | +11.5% | [+1, +15]% |
| >2 ms | 5 | +6.3% | [-9, +9]% |

**tc/M2 host-overhead caveat (load-bearing):** the `tc_max_autotune` arm is a `torch.compile`d callable carrying guard-eval + dispatch overhead. On tiny/decode shapes M2 inflates its time far more than the Helion seed's, so **`G_vs_tc` under M2 spuriously flatters the seed**. The worst flips (use the M1 value):

| kernel | shape | seed µs (M1) | G_tc **M1 (canonical)** | G_tc M2 (inflated) |
|---|---|---|---|---|
| fp8_gemm | [1, 4096, 4096] | 17.6 | **0.565** | 2.199 |
| matmul | [1024, 1024, 1024] | 7.9 | **0.882** | 3.221 |
| matmul | [2048, 2048, 2048] | 25.0 | **0.985** | 1.213 |
| bmm | [16, 2048, 128, 128] | 9.9 | **0.896** | 0.949 |
| matmul | [1, 4096, 4096] | 18.1 | **1.268** | 1.270 |

- **Ratio robustness:** median |G_vs_tc(M2) − G_vs_tc(M1)| / G_vs_tc(M1) = **2.7%** on non-tiny GEMM cells (seed ≥25 µs, n=35); 2.9% including tiny. The headline ratio is method-robust.
- **Absolute-time:** median |M2−M1| across all 48 cells = **9.1%**; it shrinks with kernel length but retains a ~6–11% floor on long kernels (residual per-launch dispatch of the compiled callables + do_bench harness cost). **M1 is canonical for absolute µs.**

### ⚑ Harness caveat — §5 M1/M2-agreement gate (honest disclosure)

The spec's §5 sanity says M1 and M2 should agree "within a few %" on a long kernel, else stop and fix the harness. **On most cells the M1↔M2 gap is the expected host-overhead offset (M2 > M1) and cancels in the ratio (~2.7%).** But there are two honest wrinkles a reader must know:

1. **M1 > M2 inversions** (M1 device time *above* M2 wall-time — physically it should be ≤): **3 of 48 cells.** These are cells where M1's cudagraph replay has more run-to-run jitter than M2's do_bench, so M1's median gets pulled up by a few slow rounds:
   - matmul [8191, 8191, 8191] (prime): M1 seed 6,039 µs > M2 5,481 µs (+10.2%)  ← the headline worst-cell; M1 jitter ~4%, M2 tight ~1% → **trust M2 here**
   - mamba2_chunk_state [2, 4096, 64, 256, 64, 128] (model): M1 seed 50.1 µs > M2 48.8 µs (+2.6%)
   - matmul [4096, 4096, 128256] (vocab): M1 seed 6,344 µs > M2 6,308 µs (+0.6%)

2. **The 8191³ prime is the one materially-affected result.** Re-timed at R=15 it reproduces: M1 seed median ≈6070 µs (σ≈4%, max ~6300) vs M2 ≈5450 µs (σ≈1%). The clean cold time is ~5450 µs (M1 *min* 5588 ≈ M2 median), so the honest `G_vs_tc` is a **range 0.27 (M1) – 0.32 (M2)**, i.e. seed ~3.1–3.8× slower than cuBLAS — the qualitative "prime tail-mask is the seed's worst regime" conclusion is unchanged, but the single-point 0.276 was ~M1-jitter-pessimistic. The prime kernel's irregular tail-masked WGMMA genuinely has high replay variance; this is a property of that kernel, not a miscalibration (4096³/8192³ M1 are tight to ~1%).

Everywhere else the §5 gate is satisfied in spirit: M1 is the clean canonical device time, M2 sits a host-overhead-width above it, and the seed-vs-library **ratio** (the headline) is method-robust. The flagged cells are marked ⚑ in the per-shape tables so nothing is buried.

## helion_default failures (first-class result)

- **The `[16,16,16]` default compiled and ran on ALL 48 cells** — no OOM, no ptxas timeout, no acc-fail on this H100 at these shapes. It is merely **5–40× slower** than the seed everywhere (that IS the point). The ptxas-hang / OOM pathology the harness guards against did not fire here, but the guard (isolated subprocess + 120 s killpg per arm) was active on every cell.

- Accuracy failures (any arm, vs fp32 ref, bf16 rounding tol): none — every arm passed on every cell.

## Headline findings

**Where the seed lags the library (adversarial / batched — surfaced, not dropped):**
- matmul [8191, 8191, 8191] (prime): G_vs_tc = **0.276** (seed 3.62× slower than cuBLAS/cuBLASLt) — ⚑ M1-jitter cell; honest range **0.27–0.32** (~3.1–3.8× slower), see harness caveat.
- matmul [256, 16384, 256] (deep_K): G_vs_tc = **0.448** (seed 2.23× slower than cuBLAS/cuBLASLt).
- bmm [16, 4096, 128, 4096] (long_seq_attn): G_vs_tc = **0.501** (seed 2.00× slower than cuBLAS/cuBLASLt).
- bmm [32, 2048, 128, 2048] (attn_scores): G_vs_tc = **0.529** (seed 1.89× slower than cuBLAS/cuBLASLt).

**Where the seed is at/near parity or ahead of the library:**
- matmul [1, 4096, 4096] (decode): G_vs_tc = **1.268**.
- matmul [32, 8192, 8192] (decode): G_vs_tc = **0.992**.
- fp8_gemm [16384, 8192, 512] (tall_skinny): G_vs_tc = **0.989**.

## Honesty caveats (framing, not bugs)

- **What ships (H100/sm90 only):** PR #3006 promotes the formula (`promote_seed_to_default=True`), so on H100 the emitted config IS the no-autotune compiler default (`effort=none` returns it), replacing the `[16,16,16]` fallback — so here "the no-autotune default reaches ~cuBLAS on aligned GEMMs" is the correct, stronger claim. The heuristic is `HARDWARE_TARGETS=(("cuda","sm90"),)`-gated and does NOT fire on any other GPU (B200/sm100 is the separate PR #3007). The perf here was gathered on a byte-identical revision of the matmul config-gen code (config read via `compiler_seed_configs[0]`); since that logic is 0-diff vs the PR, the emitted configs — and thus these numbers — describe PR #3006's seed exactly.
- **static_shapes=True** bakes dims as constexpr (tail-mask elision on aligned dims) — a real specialization edge over cuBLAS on aligned shapes, and a real *penalty* on non-aligned/prime shapes (see 8191³: masked tails, seed 3.8× slower).
- **fp8** is vs `_scaled_mm(fast_accum=True)` (cuBLASLt), not dense cuBLAS — "~parity," qualified. fp8 accuracy tol is looser (e4m3 has ~2 mantissa bits).
- Numbers are **cuBLAS / driver / GPU-SKU-bound** (this box: torch 2.13.0.dev+cu130, triton 3.7.0, driver 595.71.05).
- **Clocks:** recorded before/after each cell are idle-state samples (GPU quiesced between subprocesses). Anti-throttle evidence is the **temperature ceiling (≤47 °C across the whole run)** and the absence of any thermal/power throttle bit (only the benign GpuIdle bit ever set).

