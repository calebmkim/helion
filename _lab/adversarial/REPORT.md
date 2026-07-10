# ADVERSARIAL POINTWISE HARDENING — REPORT

> **TWO SHIPPABLE VARIANTS (2026-07-02):**
> - `pointwise-seed-heuristic` @ **b890429d** — FULL 4-lever version (all 5 gap classes at tc-parity).
> - `pointwise-seed-L1L2` @ **d2ef1af2** — SIMPLIFIED: only L1 (contig) + L2 (SFU num_warps). Drops L3
>   (fan-in) + L4 (live-temp shrink), net −58 lines, 2 fact fields instead of 4. Deliberate trade
>   (simplicity): manydiff 1.02→0.75 (floor, still 1.12× default), fanin 0.96-0.98→0.82-0.87 (~default);
>   all still ≥ floor and ≥ default; transpose/compute/flat/rope identical to the full version. Recommended
>   if simplicity is the priority.


**Goal:** harden the pointwise autotuner seed heuristic against 5 adversarial gap classes where the seed
was >10% WORSE than the compiler default, climbing each to torch.compile PARITY (measured-reachable —
the oracle ties tc on every class, so none are codegen-limited).

**Base:** `pointwise-seed-heuristic` @ 792268d9 (prior pointwise champion 268edf3b + RoPE partial-tile fix
+ fan-in floor fix). **Champion:** @ b890429d (Levers 1-4 + overtime o2; Gate-E-frozen deliverable at
2e8d0b9c, o2 a strict-improvement overtime commit on 2 non-TEST small-conflict cells). Commit trail:
792268d9(base) → 583b93fc(L1) → 534edd13(L2) → 30a007a6(L4) → 2e8d0b9c(L3) → b890429d(o2). +333/-55 over
3 helion files (autotuner_heuristics/triton.py, _compiler/device_ir.py, autotuner/config_spec.py).
Worktree `helion-pointwise`. bf16 unless noted,
forward-only, cold-L2 do_bench median-of-9, vs `torch.compile(max-autotune-no-cudagraphs)` (a strict bar —
beating it implies beating tc-default).

## RESULT — per gap class, G = tc_latency / seed_latency (1.0 = tc parity)

| class | rep kernel(s) | seed BEFORE (G) | seed AFTER (G) | lever |
|---|---|---|---|---|
| **transposed / strided** | transposed_both_scale [4096,4096] fp32 | **0.19** ([1,1024]w4) | **0.99** ([1024,1]w4) | L1 contiguity |
| | transposed_in_relu2 [8192,8192] | 0.29 ([1,2048]w4) | 0.94 ([64,64]w4) | L1 (conflict→balanced) |
| | transposed_out_add [8192,8192] | <floor | 0.92 ([64,64]w4) | L1 |
| **compute-bound wide rows** | wide_row_trig_chain [2048,2048] | **0.63** ([1,2048]w4) | **0.99** ([1,2048]w16) | L2 num_warps |
| | chained_activation_2d [4096,4096] | ~1.0 vsdef | 1.2 vsdef ([1,1024]w16) | L2 |
| **broadcast + compute** | bcast_rowvec_transcend [8192,8192] | ~0.63 | **1.01** ([1,2048]w8) | L2 |
| **many live temporaries** | manydiff_chains [4096,4096] | **0.23** ([1,2048]w4) | **1.02** ([1,512]w8) | L2+L4 |
| **high fan-in** | fanin64_2d_add [4096,4096] fp32 | **0.87** ([1,128]w4) | **0.98** ([1,512]w4) | L3 |
| | fanin96_bf16_2d [4096,4096] | 0.82 ([1,128]) | 0.96 ([1,512]) | L3 |

Every class reaches the 0.75 floor with a large margin; 4/5 at full tc-parity, fan-in at 0.96-0.98.
The deepest disasters (0.19 / 0.23 / 0.29 / 0.63) are all fixed. rope (Tier B guard) and the flat
GLU/activation family (Tier A anchor) are byte-identical throughout.

## THE LEVERS (simplest-big-wins-first)

**L1 — stride/contiguity-aware tiling** (@583b93fc). The seed grew the LAST block dim, assuming it is
contiguous; for a transposed/strided VIEW the stride-1 axis is a DIFFERENT block dim, so the wide `[1,BIG]`
tile ran along the strided axis (stride M) → uncoalesced (up to 5.3× slower than default). Faithful
`MemoryOpFact.subscript_strides` (raw `.stride()` per subscript, walker field, no new walk) →
`PointwiseElementwiseFact.contig_block_ids` (derived, no walk). Distributor: single contiguous axis → put
the budget there (`[1024,1]`); a load-vs-store CONFLICT (>1 contiguous axis) → a balanced tile filling the
register/occupancy budget (`[64,64]`), since a wide single axis strides the other operand (the `[2048,1]`
disaster, 0.13×). Row-major → `contig={last}` → byte-identical. **Gate F mechanism confirmed in lowered
Triton** (the address strides are identical seed-vs-oracle; only the block WIDTHS change which axis carries
the coalesced run). Gates A/D/F/H all PASS; bound 5 unseen transpose patterns.

**L2 — num_warps ramp keyed on SFU (transcendental) op count** (@534edd13). The seed always left
`num_warps=4`; a transcendental-heavy tile is LATENCY-bound. The faithful signal is the SFU op count, NOT
total op count: `horner_poly24` (0 SFU, 77 cheap FMAs) wants w4 (w16 REGRESSES it), while `wide_row` (12
SFU) wants w16. `PointwiseElementwiseFact.sfu_ops` (hardware execution-unit count, in the existing walk);
`_warps_for(sfu, tile_numel)`: sfu<3→4, <9→8, else 16, capped by `tile_numel//64` (over-warping a small
tile regresses — `[1,256]`w16=0.5×). Emitted only when >4 → flat family byte-identical. Gates all PASS
(refuters tested novel kernels with SFU ops absent from the corpus → keyed on the count, not names).

**L4 — peak-live-SFU register model** (@30a007a6). `reg_bytes_per_elem` counted only I/O slabs, missing the
fp32 transcendental TEMPORARIES the body holds live. A multi-branch activation holding ~20 SFU results live
(computed then summed) spills at a bandwidth-sized tile; the num_warps ramp alone reached only the floor.
`peak_live_sfu` = peak concurrent LIVE SFU-result tiles (def..last-use liveness, folded into the single
existing walk), added to `reg_bytes_per_elem` → the existing reg_cap shrinks the tile ([1,2048]w16→[1,512]w8
= the oracle parity config). The faithful separator: a SEQUENTIAL SFU chain (~1-2 live) and a deep CHEAP-FMA
chain (Horner: 24 live MUL results, 0 SFU, cheaply rematerialized) are correctly NOT shrunk. A refuter
proved via causal disentanglement that the TILE SHRINK ([1,512]w16=0.99), not the warp drop
([1,2048]w8=0.49), carries the win. Gates all PASS.

**L3 — fan-in-aware register + bandwidth sizing** (@2e8d0b9c). A K-way sum (32-96 full-extent loads → one
output) was starved to `[1,128]` by two clamps that treat every operand as concurrently resident: (a)
`reg_bytes` summed all K load slabs → reg_cap=252 clamped the tile; fix caps the load contribution at a
resident-load window (`min(Σload, 4*max_load)`) — a pointwise combine streams loads, it does not hold all K
live (measured: fanin runs fine at [1,2048], no spill). (b) `budget = TILE_BYTES // bytes_per_elem` divided
by the full K-operand traffic; fix caps the divisor at `COALESCE_BYTES=16` (the tile-size target is a
per-operand coalescing run; 16 is above every low-fan-in traffic class incl fp32 traffic-3=12 → flat
byte-identical). Both are needed (a coupling: reg-only→[1,256], budget-only→[1,128]). Broadens faithfully by
high bytes/elem to fanout_flat_many (improved) and fp64 (neutral). Gates: [pending at time of writing].

## END-TO-END NO-REGRESSION (config-recorder diff baseline 792268d9 → champion b890429d, 264-cell matrix)
41 changed cells, ALL in the adversarial gap-class kernels (transpose 9, compute/warps 12, fanin 6, fanout
3, broadcast 2, temporaries/manydiff 2, wide_fanin 2, fp64 5). **Tier A flat family (swiglu/geglu/
residual_add/relu_squared/bias_gelu/dyt) + rope (Tier B) + negatives (softmax/rms_norm/matmul) = ZERO
changed (byte-identical end-to-end).** The levers fire only on the gap classes; the shipping family,
partial-tile rope, and the reduction-family disjointness are untouched.

## NO-REGRESSION (Gate R, config-recorder diff over the full 264-cell matrix per edit)
Each lever's config diff touched ONLY its target class; the flat GLU/activation family (Tier A anchor),
rope (Tier B), the other adversarial classes, and the negative recognizers (softmax/rms_norm/matmul —
`fires=False`, disjointness holds) stayed byte-identical except where a lever's faithful mechanism broadened
into them (L3 → fanout/fp64, measured non-regressing). Tool: `_lab/adversarial/cfg_recorder.py`
(record/diff, HELION_SRC-overridable for a baseline BEFORE).

## FILES
`helion/autotuner/config_spec.py` (MemoryOpFact.subscript_strides; PointwiseElementwiseFact.contig_block_ids
/ sfu_ops / peak_live_sfu), `helion/_compiler/device_ir.py` (populators, in the single walk),
`helion/_compiler/autotuner_heuristics/triton.py` (TritonPointwiseSeedHeuristic: contiguity distributor,
num_warps ramp, fan-in caps). Lab: `_lab/adversarial/{NOTEBOOK.md,ledger.json,cfg_recorder.py,
shapes_adversarial.py}`; harnesses `local/rope_probe/adv/{cfgbench.py,curr_bench.py,oracle_vs_tc.py}`.

## GATE E FREEZE (sole TEST read, once, @ champion 2e8d0b9c) — PASS
Held-out TEST split (never benched before this read), G = tc/seed, cold-L2 med-9:
| kernel | shape | G_seed | vs_default |
|---|---|---|---|
| transposed_in_relu2 | [8192,4096] / [3072,6144] | 0.95 / 0.941 | 1.03 / 1.04 |
| transposed_both_scale | [4096,8192] | 1.00 | 1.02 |
| wide_row_trig_chain | [3072,6144] / [256,16384] | 0.994 / 0.99 | 1.17 / 1.22 |
| wide_row_moderate_chain | [4096,2048] | 1.00 | 1.02 |
| skinny_m_wide_n_chain | [96,49152] | 0.98 | 1.19 |
| manydiff_chains | [8192,4096] / [3072,12288] | 1.019 / 1.016 | 1.52 / 1.52 |
| horner_poly24 | [8192,4096] | (no tc_ref) | 1.17 |
| fanin64_2d_add | [8192,4096] | 0.982 | 1.11 |
| fanin32_2d_silu | [8192,4096] | 0.977 | 1.11 |
| fanin96_bf16_2d | [8192,8192] | 0.956 | 1.69 |
| bcast_rowvec_transcend | [8192,4096] | 1.012 | 1.04 |
**BELOW FLOOR: none.** Every TEST shape clears the 0.75 floor; all at/near tc-parity (0.94-1.02) and beat
the compiler default (1.02-1.69×). TEST tracks TRAIN — the activation-blind bytes/stride/SFU/liveness seed
generalizes to held-out shapes (interpolation, not memorization). No overfit. **FREEZE VERDICT: PASS.**

## DEFERRED (human-review; Priority-2 overtime, logged with recipes)
- **register-model unification** (refactor-critic P1, BORDERLINE): merge the 3 register terms into one
  slab-weighted peak-live-bytes liveness. Elegance + retires LIVE_LOAD_WINDOW, but NO perf gain (at parity)
  and touches rope [1,1] + flat byte-identity. Gate D judged LIVE_LOAD_WINDOW faithful, weakening the motive.
- **iterated_map disjointness broaden** (completeness P0): a fixed-point map (`for _ in range(6)` Babylonian
  sqrt) makes an all-tiled AccumulatorFact fire → disjointness over-excludes it → default tile. Fix = admit
  pointwise when every accumulator is all-tiled (no collapsed/reduction dim). Deferred: CORE-invariant risk
  (a scan/SSM with an all-tiled carried state could be mis-classified; no scan kernel in this corpus to
  validate scan-safety). Realistic-but-niche.
