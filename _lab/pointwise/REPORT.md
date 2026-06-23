# Pointwise / Elementwise Seed Heuristic — Report

**Branch** `pointwise-seed-heuristic` (worktree `helion-pointwise`), off `reduction-4pr-stack` @ `9bf7b1f9`.
**Champion** `268edf3b6793e889e991c13693ddd997ff2d5870` (corpus byte-identical to the gated `b4a4a681`,
so the gated perf + Gate E freeze transfer; the tip adds a behavior-preserving fact-trim + a tall-skinny
BROADEN that only improves off-corpus short-inner shapes). Hardware: H100, bf16, cold-L2 do_bench
med-of-9, **cool GPUs** (thermal corruption footgun — see below).

## What it is
A NEW autotuner seed family for **pure elementwise/pointwise** kernels — a tiled map that reads its
inputs, computes, and writes its outputs with **no reduction, no matmul, no loop-carried accumulator**
(bandwidth-bound). The niche was empty: the compiler default tiles these at `block_size=32`
(`config_spec.BlockSizeSpec._fragment`: `total_ndim<=2 and reduction_numel<=128`), moving only ~10% of
HBM. The 4-file change (+~200 lines, purely additive):
- `PointwiseElementwiseFact` (a DERIVED fact) + `MemoryOpFact.accessed_numel` (a walker field) in
  `helion/autotuner/config_spec.py`.
- `DeviceIR.build_pointwise_facts` (Phase 5) + `accessed_numel` population in `helion/_compiler/device_ir.py`.
- `TritonPointwiseSeedHeuristic` in `helion/_compiler/autotuner_heuristics/triton.py`, registered in `__init__.py`.

## The fact (faithful & disjoint)
`build_pointwise_facts` builds the fact **only when** `reduction_facts`/`matmul_facts`/`accumulator_facts`
are all empty (the disjointness rule — a reducing kernel stays in the reduction family, never clawed back).
Fields are pure derivations over the walker `memory_op_facts` + block-size specs (no graph walk):
- `bytes_per_elem` = Σ over **full-extent** load+store ops of `dtype.itemsize`, where full-extent =
  `accessed_numel == total_numel`. `accessed_numel` = product of `size_hint(shape[i])` over **non-broadcast**
  dims (`stride != 0`) — the distinct-HBM-element count. This faithfully gives **traffic-3** for gated acts
  (swiglu/geglu = 6) vs **traffic-2** for unary ops (relu²/bias_gelu = 4), and **excludes every broadcast
  form** (rank-1 `bias[N]`, full-rank `[M,1]`/`[1,N]`, and stride-0 `.expand()`/`broadcast_tensors`),
  computed from itemsizes — never an op name or a dtype literal.
- `total_numel` (occupancy input), `n_block_dims`, `block_size_hints`, `n_load`, `n_store`.

## The lever
`get_seed_config`: a **bytes-aware** target tile `max(BLOCK_FLOOR=256, TILE_BYTES=8192 // bytes_per_elem)`
(traffic-3 → ~1024 elems, traffic-2 → ~2048), capped by **occupancy** `total_numel // (num_sm*MIN_WAVES=8)`
(size_hint-aware). The tile is distributed: **outer (strided) dims → 1**, the **innermost (contiguous) dim**
absorbs the budget (pow2, capped at `next_pow2(extent)` so a short row is covered in one masked tile). Emits
`Config(block_sizes=...)` only — `num_warps`/`num_stages`/`pid_type` stay at the compiler defaults (4/1/'flat'),
so `block_sizes` is the sole non-default field (no dead knobs). Mechanism (verified in lowered Triton):
`block_size=32` → 524288 tiny 64B-per-tensor programs (~0.32 TB/s); the seed → coalesced KB tiles (~2.2 TB/s).

## Perf (train+val, bf16, cool-GPU clean numbers @ a1b5402)
| kernel | seed | seeded_vs_default | G = tc/seed (geomean) | min_G | acc |
|---|---|---|---|---|---|
| swiglu | `[1024]` | 6.83× | 0.998 | 0.995 | 23/23 |
| geglu | `[1024]` | 6.84× | 0.996 | 0.993 | 17/17 |
| residual_add | `[1,1024]` | 1.35× | 0.986 | 0.964 | 16/16 |
| relu_squared | `[2048]` | 9.99× | 0.994 | 0.988 | 12/12 |
| bias_gelu | `[1,2048]` | 1.21× | 0.968 | 0.903 | 15/15 |

- Headline `seeded_vs_default` closes the 5–10× default gap on the 1-D kernels (1-D default `[32]`=32 elems);
  the N-D kernels (residual_add/bias_gelu) have a smaller gap because their N-D default `[32,32]`=1024 is
  already a decent tile.
- `seeded_vs_tc` ≈ **parity** on every kernel (0.968–0.998) — the oracle-retarget target met (oracle ≈ tc,
  fixed HBM ceiling; chasing >1× is chasing noise).
- **No realistic shape below the 0.75 floor** (min_G 0.90–0.995). The traffic-2 (relu²/bias_gelu) and
  traffic-3 (swiglu/geglu/residual_add) cases are captured by the SAME bytes-aware seed — no per-kernel knob.

## No-regression (the negative recognizers)
`rms_norm` (reduction), `matmul`, `softmax`, `layer_norm`, and the `per_token_fp8_quant` divergence kernel
(amax over N → a reduction) get **NO** pointwise fact and emit **byte-identical** facts + seed configs vs the
base `9bf7b1f9` (the change is purely additive; `accessed_numel` is consumed only by `build_pointwise_facts`).

## Held-out (Gate E firewall) — read once at freeze (champion b4a4a681, cool GPUs)
**TEST split** (held-out shapes, first read) — all 5 fit kernels near tc parity, all clear the floor:
swiglu G=0.997 (min 0.994), geglu 0.996 (0.995), residual_add 0.977 (0.972), relu_squared 0.998 (0.991),
bias_gelu 0.977 (0.952). **Held-out kernel `dyt`** (never fitted): G=0.967, min_G=0.876, beats default
1.18×, 18/18 acc, no below-floor. The activation-blind bytes/numel seed **generalizes to an unseen
kernel** — interpolation, not memorization. Overfit firewall: **PASS**.

## Gate verdicts (champion b4a4a681)
- **Gate A** (adversarial verify): PASS — 3/3 independent skeptics refuted=false; a separately-authored
  CUDA-Event repro (different method than do_bench) confirms the tc-parity claim (0.96–0.99).
- **Gate D** (fact faithfulness + population): PASS after **three** adversarial fixes, each a real
  divergence in the full-extent test: (1) full-rank `[M,1]`/`[1,N]` broadcasts (a bare-ndim proxy
  counted them) → `accessed_numel == total_numel`; (2) stride-0 `.expand()`/`broadcast_tensors` →
  stride-aware `accessed_numel`; (3) oversized operands (padded/sliced buffers wider than the tile) →
  `accessed_numel >= total_numel`. 4th-pass Gate D: refuted=false (population reads `dtype.itemsize`
  directly — mixed-dtype→8; threshold is a byte-budget divisor, no fence). One non-idiomatic corner
  (re-loading the same tile from HBM) is scoped-deferred (the corpus binds-once → correct bytes).
- **Gate F** (mechanism): PASS — verified in lowered Triton (`block_size` 32→KB is the sole non-default
  field carrying the win; no dead knobs).
- **Gate H** (generality): KEEP — faithful keys (bytes/numel/occupancy/num_sm), broad firing, byte-budget
  + occupancy-cap form. 6 BROADEN items queued (num_warps ramp, per-dtype budget, etc.).
- **Gate R** (regression-referee): accept — negatives byte-identical, no realistic shape below floor,
  5/5 cells up vs default; frozen anchor set.
- **Gate E** (overfit firewall): PASS — see above.

## Overtime validation (post-DoD, never-stop)
- **Refactor-critic**: heuristic is near-minimal; trimmed the fact 6→2 fields (the 4 dropped were read
  by no branch). Behavior-preserving.
- **Robustness/decode correctness**: all 6 kernels pass accuracy on every decode (M=1..256), odd/prime M,
  and non-pow2 N canary (partial tiles, tiny grids) — the seed is correct on edge shapes.
- **Disjointness in isolation**: a carried-state kernel (loop-carried accumulator, no reduction/matmul)
  is correctly excluded (accum=1, pointwise=0) — the 3rd negative-recognizer class, beyond reduction/matmul.
- **dtype-faithful**: fp16 (bytes=6→[1024]) and fp32 (bytes=12→[512], a *smaller* tile for the bigger
  per-element traffic) both fire + are correct — the bytes-aware budget reads `dtype.itemsize`, not bf16.
- **num_warps ramp**: measured-DEFER — w4 is optimal at the seed's tiles (w8 within-noise, w16 −10%),
  so no warp knob is added (would be dead weight).
- **Tall-skinny BROADEN** (completeness-critic 2nd pass): the corpus (min inner N=768) masked a starved
  tile for short-inner tensors (image RGBA N=4, per-head N=64) — the `outer→1` distribute emitted
  `[1,8]` (32× below the default, below floor). Fixed by **spilling leftover budget outward** (`[M,8]→
  [128,8]`, a coalesced row-major 1024-tile). Measured: tall-skinny now beats default 1.07–1.15× at tc
  parity; corpus byte-identical. The distribute now covers all aspect ratios (1-D, wide-N, tall-skinny).

## Method footguns hit (logged)
- **GPU thermal corruption**: sustained benching on one H100 (→65°C) corrupted timings up to 50%,
  asymmetrically across configs, flipping A/B signs (briefly faked a bias_gelu "seed<default" + a bogus
  broadcast-tiling theory). Fixed by re-measuring on cool GPUs (<40°C) + reproduce-twice. See the memory note.
- **tc max-autotune variance**: torch.compile's bias_gelu compile swung 160–304µs run-to-run; G ratios are
  tc-noisy, so the robust signals are min_G (≥0.90) and `seeded_vs_default` (default is stable).
- Do **not** measure through the swiglu/geglu *tritonbench operators* (full MLP); this harness times the
  standalone elementwise op on pre-projected `[M,N]` tensors.
