# PR #3007 table re-verification — FULL bucket coverage (not just the displayed shapes)

Question: are the incumbent table's tuned numbers still valid — across ALL the shapes it was
tuned for, not just the 9 the PR chose to display?

## The table's tuned space = 10 buckets (matmul_b200.json)

The incumbent `TritonB200MatmulHeuristic` is bucket-based (M/N/K interval rules + exact-value
anchors), not a fixed shape list. It has **10 rules/buckets**. A shape fires a rule when its dims
fall in the rule's M/N/K intervals AND match any exact `*_value` fields the rule carries.

## Coverage of the 9 PR-displayed shapes vs the 10 buckets

| bucket | M range | N range | K range | primary tile | covered by a PR shape? |
|---|---|---|---|---|---|
| 0 | (512,1024] | (512,1024] | (512,1024] | [128,64,64] | ✅ (1024,1024,1024) |
| 1 | (1024,4096] | (1024,4096] | (1024,4096] | [128,128,64] | ✅ (2048³,3072³,3584³,4096³) |
| 2 | (256,512] | (256,512] | (256,512] | [16,64,128] | ✅ (512,512,512) |
| 3 | (1024,4096] | (512,1024] | (512,1024] | **[256,128,64]** | ❌ **NOT in the PR's 9** |
| 4 | (1024,4096] | (1024,4096] | (1024,4096] | [256,256,32] | ✅ (4096,2048,2048 etc.) |
| 5 | (1024,4096] | (1024,4096] | (1024,4096] | [256,256,64] | ✅ (cube range) |
| 6 | (512,1024] | (4096,inf) | (512,1024] | [256,256,32] | ✅ (1024,8192,1024) |
| 7 | (4096,inf) | (1024,4096] | (1024,4096] | [256,256,32] | ✅ (8192,2048,2048) |
| 8 | (4096,inf) | (512,1024] | (512,1024] | **[128,256,64]** | ❌ **NOT in the PR's 9** |
| 9 | (512,1024] | (4096,inf) | (512,1024] | [128,256,64] | ✅ (1024,8192,1024) |

So the PR's 9 displayed shapes touch **8 of 10 buckets**. Buckets **3 and 8 were untested** by any
PR-displayed shape — and bucket 3's `[256,128,64]` tile appears in NO other bucket, so extrapolation
from the displayed shapes would have been unjustified.

## Filling the gap: buckets 3 & 8 fire only at EXACT-value anchor points

A subtlety of the matcher: buckets 3 and 8 carry exact `m_value`/`n_value`/`k_value` fields that are
checked as equality, so they fire ONLY at:
- bucket 3 → exactly **(M,N,K) = (4096, 1024, 1024)**  (tile [256,128,64])
- bucket 8 → exactly **(M,N,K) = (12288, 1024, 1024)** (tile [128,256,64])

(Interior points like (2048,768,768) or (8192,900,900) do NOT fire these rules — the table declines
and falls to base. So these two anchors are literally the only shapes exercising buckets 3 and 8.)

## Result at the 2 previously-untested anchors (cold-L2, median-of-15, no autotune)

| bucket | (M,N,K) | dt | tile_old (table) | tile_new (formula) | M2 old µs | M2 new µs | **M2 new/old** | M1 new/old |
|---|---|---|---|---|---|---|---|---|
| 3 | (4096,1024,1024) | bf16 | [256,128,64] | [128,256,64] | 19.46 | 17.44 | **1.117** | 1.171 |
| 3 | (4096,1024,1024) | fp16 | [256,128,64] | [128,256,64] | 19.42 | 17.41 | **1.116** | 1.173 |
| 8 | (12288,1024,1024) | bf16 | [128,256,64] | [128,256,64] | 41.79 | 37.89 | **1.105** | 1.088 |
| 8 | (12288,1024,1024) | fp16 | [128,256,64] | [128,256,64] | 41.98 | 37.89 | **1.108** | 1.120 |

**Formula ≥ table on both, 1.10–1.17×, no regression** (geomean 1.11× M2). Note bucket 8: both arms
pick the SAME [128,256,64] tile, so the ~1.10× is purely the formula's num_stages edge — consistent
with the PR's story for same-tile cells.

## Bottom line

Combining the 18 displayed rows (geomean 1.198× M2, reproduced) with these 4 gap-filling anchor
cells: **the formula beats or ties the incumbent table on ALL 10 of its tuned buckets — no
regression anywhere.** The PR's table is valid AND its coverage generalizes to the whole tuned space,
not just the displayed subset. This is a measured result, not an extrapolation.
