# §5 Step-0 sanity gate — PASSED (2026-07-14)

Run on matmul 4096³ bf16 (the spec's compute-bound long-kernel gate shape), device B200
sm100. All five §5 criteria satisfied (post harness-fix; see DECISIONS.md D9):

| # | criterion | result | pass |
|---|---|---|---|
| a | accuracy: seed output ~matches ref within bf16 rounding | max_rel = 5.8e-3 (< 5e-2) | ✅ |
| b | seed_fired=true AND seed cfg differs from helion_default | seed `[256,256,32] w8 s6` (formula) ≠ default `[128,128,64] w4 s7` (table). `default_source=table` (in-bucket). `promoted_is_formula=True`. | ✅ |
| c | cold-L2 real (implied BW below HBM peak, not an L2 artifact) | seed 98.5 µs → implied BW ≈ 4.1 TB/s (3×4096²×2B / 98.5µs); B200 HBM ~8 TB/s, L2 would be >>30 TB/s → cold. Also 8192³ seed 707µs → 1556 TFLOP/s = 69% of dense peak, consistent w/ compute-bound cold run. | ✅ |
| d | tc_max_autotune winner is the expected backend (cuBLAS) | winner=cuBLAS/cuBLASLt; independent check: eager torch.matmul ≈ tc within 0.6% (both nvjet cuBLASLt) | ✅ |
| e | M1 (cudagraph) ≈ M2 (do_bench) within a few % on this LONG kernel | M1 seed 98.5 / M2 99.6 µs → 1.1% ; default M1 125.6 / M2 132.1 → 5.2% ; tc M1 84.4 / M2 91.1 → 7.9% (tc's larger gap = torch.compile guard host-overhead, expected & documented). Seed (the headline arm) agrees to 1.1%. | ✅ |

Headline at 4096³: **G_vs_tc = 0.857** (seed reaches 86% of cuBLAS), **xD_vs_default = 1.27**
(formula beats the incumbent TABLE by 1.27× on its own tuned turf — matches PR #3007's
1.19–1.27× claim and my prior independent verification). seed = 1395 TFLOP/s = 62% of the
2250 TFLOP/s dense bf16 peak.

Cross-kernel smoke (first cell each) also passed and matches the B200 story:
- fp8 4096³ (src=base): xD=39.5× (default 2509µs → seed 63.6µs), G_vs_tc=0.73, 2161 TFLOP/s
- bmm [8,4096,4096,4096] (src=base): xD=40.4× (default 48.5ms → seed 1.20ms), G_vs_tc=0.53
- mamba [2,4096,64,256,64,128] (src=base): G_vs_tc=1.75 (beats its Triton tc), xD=6.4×
- matmul 8192³ (src=base): xD=50.3× (default 35.6ms → seed 707µs), G_vs_tc=0.89, 69% peak

Proceeding to the full 48-cell sweep.
