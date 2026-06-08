# Perf A/B — my heuristic config vs main's, on the MERGED tree (main's codegen)

**Date:** 2026-06-05. **Tree:** /tmp/v2-wt (reduction-seed-heuristic-v2, = origin/main 8d5cc261 + the reduction PR). **GPU:** 1× H100 idx0, serial, single-process interleaved A/B (noise-robust ratio). **Method:** `configs=[cfg]` per arm, correctness-gated (rtol1e-3/atol1e-4, not loosened), median-of-7 do_bench. Configs pulled verbatim from `dual_merged.json` (the recorded merged-tree emissions). **ratio = main_lat / mine_lat** (>1 ⇒ MINE faster).

Raw data: `perf_ab_merged.json` (145 rows). This file = human summary.

## Headline
- **145 T1 differing-config overlap shapes benched. 141 both-arms-correct + ratio.** 0 shapes where mine regresses vs main (no ratio < 0.95).
- Overall **median ratio 1.056, geomean 1.598** (geomean pulled up by the cross_entropy spill-avoidance wins).
- Outcome split (mine perspective, 5% band): **mine wins >5%: 75 | tie: 66 | mine loses >5%: 0.**

## Per-kernel (ratio = main/mine)
| kernel | n | median | min | max | mine win >5% | tie | mine loss >5% |
|---|---|---|---|---|---|---|---|
| cross_entropy | 28 | **11.49** | 1.106 | 17.17 | 28 | 0 | 0 |
| long_sum | 21 | **1.382** | 0.999 | 2.561 | 19 | 2 | 0 |
| sum | 28 | **1.071** | 0.991 | 1.211 | 22 | 6 | 0 |
| layer_norm | 32 | 0.999 | 0.972 | 1.332 | 6 | 26 | 0 |
| rms_norm | 32 | 0.995 | 0.962 | 1.032 | 0 | 32 | 0 |

## Interpretation
- **cross_entropy — the big win (median 11.5×, up to 17×).** Main seeds persistent on wide-vocab rows and SPILLS (e.g. (4096,65536): main 6288µs vs mine 366µs). This is exactly the failure mode my 240 KiB persist-cap (EDIT#1) + looped fallback was built to prevent. Lowest CE ratios (~1.1×) are the narrow-V rows where both stay persistent — mine still ahead on num_warps.
- **sum / long_sum — solid wins from the num_warps ramp + streaming eviction** (sum median 1.07×, long_sum 1.38×, up to 2.56×). Main's fixed num_warps=4 + blanket ['last'] underserves wide streaming rows.
- **rms_norm / layer_norm — mostly TIES at ~1.0.** Where the ONLY config diff is eviction (['last'] vs ['first']/default), it's perf-inert on these shapes → tie. Where num_warps ALSO differs, layer_norm wins up to 1.33×. rms_norm is all-ties (its num_warps diffs land on shapes where the ramp value doesn't move the needle here). No regressions — min ratios 0.962/0.972 are within do_bench noise (both >0.95).

## The 4 not-both-correct long_sum shapes (NOT mine regressing)
- (256,131072) + (128,229376): **MAIN's config FAILS correctness, mine PASSES** → effectively a mine win (main mis-seeds these). mine: 50.8µs / 45.2µs, persistent, correct.
- (16,2097152) + (8,2097152): **BOTH arms hit a Triton CompilationError** (the 2M-element tail, rnumel>2^20 — a known source-limit candidate). Neither config compiles → genuinely unmeasurable, not a regression on either side.

## Bottom line
On main's own codegen, my heuristic is **≥ main's config on every measurable overlap shape** (0 regressions / 145), ties where the config delta is perf-inert (rms_norm, narrow layer_norm), and wins large where main mis-seeds — catastrophically so on wide-vocab cross_entropy (spill avoidance, ~11–17×) and solidly on wide streaming sum/long_sum (num_warps + eviction). Pending independent verification.
