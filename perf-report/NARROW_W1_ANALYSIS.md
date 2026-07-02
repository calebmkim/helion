# NARROW_W1 (num_warps=1) investigation — the fused_add_layernorm bf16 disaster

**Question from the manager:** the `fused_add_layernorm [16384,1024] bf16` disaster (`G_tc=0.23`,
seed picks `num_warps=1`) is probably the `num_warps=1` special-case. (a) What if we removed it —
which shapes regress and by how much? (b) Can we adjust the constant to avoid the disaster?

## The special-case

`_TritonReductionSeedBase._num_warps` (helion/_compiler/autotuner_heuristics/triton.py:751) has a
NARROW-row single-warp refinement:

```python
if have_enough_information:                       # num_sm>0, itemsize>0, grid_rows>0
    occ = grid_rows // num_sm
    if (pd.carried_2d_count == 0
        and row_bytes <= NARROW_W1_MAX_BYTES        # = 2048
        and occ * row_bytes <= NARROW_W1_OCC_BYTE_LIMIT):  # = 262144
        return 1
# else the ramp: rnumel<=1024->4, <=4096->8, <=16384->16, else 32
```

`row_bytes = pd.size_hint * input_load_itemsize` (the reduction extent × the HBM load width — a
faithful, dtype-agnostic key). `fused_add_layernorm [16384,1024] bf16`: `row_bytes = 1024·2 = 2048`
= exactly `NARROW_W1_MAX_BYTES`, so it fires w1. The fp32 version is `1024·4 = 4096 > 2048`, escapes
to w4, and is fine — which is why the disaster is bf16-only.

## Method

Matched-lever A/B (perf-report footgun-compliant): for every one of the **19 (kernel,shape,dtype)
cells in the report whose seed emits w1**, time the seed config AS-IS (w1) vs the SAME config with
`num_warps` bumped to the ramp value, **everything else held fixed**. Cold-L2 median-of-9/15
do_bench, single process, same tensors. Script: `helion-redesign/_lab/perf_report/narrow_w1_ab.py`;
raw data: `results/narrow_w1_ab.json`.

## (a) What removing NARROW_W1 does — measured on all 17 timeable w1 cells

`ramp/w1 > 1` ⇒ removing w1 is FASTER (w1 was hurting).

| effect | cells | detail |
|---|---|---|
| **improves** | **12** | 4.32× (the disaster), 1.76× (scaled_masked_softmax 16384×1024), 1.36× (dynamic_quant 16384×1024), 1.17× (welford), 1.16× (fal 8192×1024, layer_norm), 1.15× (gated), 1.11×, 1.07×, 1.04× ×3 |
| **regresses** | **1** | `softmax (131072,128) bf16`: **0.955×** (~4.5% slower). Re-measured: w1=29.3µs, w2=28.7µs, w4=30.2µs — a genuine but small win, at very narrow N=128 + very high occupancy |
| neutral (±3%) | 4 | p5-3d probe, fal(8192,768), gated(8192,768), per_token_group(128,4096,128) |

**Net: removing the special-case is a ~1.21× geomean speedup over the affected cells, improves 12,
regresses exactly 1 by ~4.5%.** The one regressor is a narrow-N (128), huge-M (131072) softmax — a
shape not in the report's required test split.

Removing it also **eliminates the disaster outright** (`fal 16384×1024 bf16`: 221µs → 51µs, back to
tc/default parity) and every other bf16-narrow-row regression in one stroke — no per-shape fence.

## (b) Can a constant fix it? — yes, but the REAL separator is occupancy, not row_bytes

Tightening `NARROW_W1_MAX_BYTES` (2048 → e.g. 512) was the obvious idea, but the data says the byte
threshold is **not** the right axis:

- The lab answer-key (`_lab/redesign/ALLOCATOR_ANSWER_KEY.md:24`) tuned 2048 for `softmax(16384,512)
  fp32` (row_bytes=2048). **Re-measured, that shape actually prefers the ramp too** (w1=33.2µs vs
  w8=28.0µs, 1.10× — the documented w1 choice there is itself a small over-fire, not an optimum).
- Fixing row_bytes=256 and sweeping M (→ occupancy) shows w1 flips purely on **occupancy**:

  | softmax bf16 N=128 (row_bytes=256) | occ≈15 | occ≈31 | occ≈62 | occ≈124 | occ≈124 (m_block=16) |
  |---|---|---|---|---|---|
  | w4/w1 (ramp vs w1) | 1.056 | 1.030 | 0.995 | **0.961** | 1.091 |
  | verdict | ramp | ramp | tie | **w1** | ramp |

  w1 wins **only** at very high occupancy (occ≈124) AND the very narrowest rows. The disaster
  `fal 16384×1024` is occ≈15 (only 2048 grid-rows ÷ 132 SMs) — **low occupancy, where w1 is exactly
  wrong** — yet the current `occ*row_bytes ≤ 262144` gate passes it (occ*rb = 30 720 ≪ 262 144: the
  occupancy gate is ~8× too loose).

The current gate conflates the two: `occ * row_bytes` is small both when occupancy is low (bad for
w1) and when rows are narrow (needed for w1). w1 actually needs occupancy **HIGH** (many independent
rows per SM already, so a warp spent on a cross-warp reduction tree is pure waste) — the opposite of
what a `≤` cap on the product selects.

## Recommendation

**Remove the NARROW_W1 refinement** (option a). It is the simplest faithful fix, kills the 4.32×
disaster plus 11 smaller bf16 regressions, and costs a single ~4.5% regression on one
out-of-test-split narrow-N softmax shape — a trade the manager already said they'd accept, and a good
one (12 wins ≥ 1 small loss). It removes a lever whose own tuning shape turned out not to want it.

If the softmax(131072,128) 4.5% is worth preserving, the **faithful** narrower fix is to re-key the
gate on **occupancy directly** — require `occ = grid_rows // num_sm` to be LARGE (empirically
≳ 100 here) in addition to (or instead of) the loose `occ*row_bytes` product — so w1 fires only in
the genuinely-machine-saturated regime it helps. But this keeps a fragile, marginal lever (the win is
~4% and evaporates at occ=124 once the widen bumps m_block, per the last column above); removal is
cleaner and the safer default. Either way, `NARROW_W1_MAX_BYTES = 2048` is not the knob to turn — the
byte cap is not the axis that separates w1-wins from w1-disasters.

*(This is a measurement/analysis writeup. No heuristic edit was applied — that is a separate
decision. All numbers are cold-L2 median do_bench on the dedicated H100, reproducible via
`narrow_w1_ab.py`.)*
