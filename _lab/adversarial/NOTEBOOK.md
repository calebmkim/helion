# ADVERSARIAL POINTWISE HARDENING — worker notebook (source of truth; trust over context, method §6.1)

Run: unattended hill-climb to harden the pointwise seed against 5 adversarial gap classes → tc-parity.
Base SHA (baseline committed): **792268d9** (branch `pointwise-seed-heuristic`, worktree
`/home/calebkim/helion-new-heuristics/helion-pointwise`; parent 268edf3b = prior pointwise champion).
Baseline = 268edf3b + RoPE partial-tile fix (reg_bytes_per_elem/reg_cap/floor) + fan-in floor fix.
Interpreter `/home/calebkim/.conda/envs/helion/bin/python`. GPU: 4×H100, pin CUDA_VISIBLE_DEVICES=0
(one at a time). L2=50MB/H100 (cold-L2). PYTHONPATH=worktree; scripts cwd=/tmp OR
/home/calebkim/helion-new-heuristics; assert helion.__file__ under worktree.
Assets (TOP-LEVEL, not worktree): `local/rope_probe/adv/{harness_one.py,oracle_vs_tc.py,shards/}`.
NEVER git push. NEVER launch a detached/background GPU job (harness-managed bg tasks that notify are OK).

Goal (lexical): FAITHFULNESS > GENERALITY > PERFORMANCE.
Bar: per-shape floor G≥0.75 vs tc; live target = tc-PARITY (oracle≈tc MEASURED-reachable → NOT weird
shapes, cannot retarget to 0.75×oracle). Tier A (flat family) = regression anchor, config-recorder every edit.

═══════════════════════════════════════════════════════════════════════
## STEP-0 — oracle-vs-tc reproduced on the CURRENT baseline (792268d9) — DONE, matches §3 exactly
═══════════════════════════════════════════════════════════════════════
`local/rope_probe/adv/oracle_vs_tc.py <kernel> M N` (targeted sweep vs tc max-autotune-no-cudagraphs,
cold-L2 med). All 5 §3 rows reproduced (~40s/kernel, GPU cool ~35°C):

| kernel | shape | seed cfg | G_seed(tc/seed) | oracle cfg | G_oracle | seed/default | lever revealed |
|---|---|---|---|---|---|---|---|
| transposed_both_scale | [4096,4096] fp32 | [1,1024] w4 | **0.188** | [64,64] w4 | 0.993 | 0.194 | tile the CONTIGUOUS axis (dim0) |
| transposed_in_relu2 | [8192,8192] bf16 | [1,2048] w4 | **0.294** | [32,128] w16 | 0.954 | 0.319 | contiguous tile **+ num_warps** (coupled) |
| manydiff_chains | [4096,4096] bf16 | [1,2048] w4 | **0.234** | [1,512] w8 | 1.012 | smaller tile **+ num_warps** |
| wide_row_trig_chain | [2048,2048] bf16 | [1,2048] w4 | **0.633** | [1,2048] w16 | 0.989 | **PURE num_warps** (same tile!) |
| fanin64_2d_add | [4096,4096] fp32 | [1,128] w4 | **0.869** | [1,2048] w4 | 0.993 | bigger tile (per-operand budget) |

Common thread (confirms §3): seed always `[1, big] + w4`. Oracles want num_warps 8-16 and/or a
stride/contiguity-aware tile shape. transposed_both_scale is PURE stride (w stays 4); wide_row is PURE
warps (tile stays [1,2048]); the rest are combinations.

Baseline seed configs (fast dump, no bench — all match §3):
transposed_both_scale[4096,4096]=[1,1024] bpe8 reg8; transposed_in_relu2[8192,8192]=[1,2048] bpe4 reg8;
transposed_out_add[8192,8192]=[1,1024] bpe6 reg12; manydiff[4096,4096]=[1,2048] bpe4 reg8;
wide_row[2048,2048]=[1,2048]; skinny_m_wide_n[64,65536]=[1,2048]; fanin64=[1,128] bpe260;
fanin32_silu=[1,256] bpe132; fanin96_bf16=[1,128] bpe194 reg388; bcast_rowvec_transcend=[1,2048] bpe4.
rope_fwd (Tier B guard): seq2048/hd256=[1,1] ; hd64=[1,2] (partial-tile fix live, was compile-fail [1,256]).

═══════════════════════════════════════════════════════════════════════
## CURRENT HEURISTIC STATE (baseline @ 792268d9)
═══════════════════════════════════════════════════════════════════════
- **Walker fact** `MemoryOpFact` (config_spec.py:178) — per load/store; already records `dtype`,
  `accessed_numel` (distinct HBM elems over stride!=0 dims), `subscript_block_ids` (block-id per
  subscript position), `indexed_block_ids`, `inner_extent`. Built by `_collect_memory_op_facts`
  (device_ir.py:2942) — the ONE graph walk; it ALREADY reads `fake.stride()` (line 3052). ← the
  contiguity provenance for Lever 1 hangs here (new field, no new walk).
- **Derived fact** `PointwiseElementwiseFact` (config_spec.py:237) — total_numel, bytes_per_elem,
  reg_bytes_per_elem. Built device_ir.py:1256 `build_pointwise_facts` (mostly derived + one compute-dtype
  walk). Disjointness: built only on absence of reduction/matmul/accumulator.
- **Heuristic** `TritonPointwiseSeedHeuristic` (triton.py:301). Constants TILE_BYTES=8192, MIN_WAVES=8,
  BLOCK_FLOOR=256, REGISTER_BYTES=65536. get_seed_config: budget_target=TILE_BYTES//bpe,
  reg_cap=REGISTER_BYTES//reg_bpe, occ_cap=total//(num_sm*8), target=min(3); inner_floor=min(256,
  pow2_floor(reg_cap)); `_seed_block_sizes` distributes INNERMOST-first spill-outward (last block dim
  gets budget). ONLY non-default field = block_sizes (num_warps=4/num_stages=1/pid=flat default).
- levers_since_refactor = 0 (fresh run). Fire refactor-critic every K≈4-5 levers / at freeze.

═══════════════════════════════════════════════════════════════════════
## DoD STATUS — all 5 gap classes CLEAR THE FLOOR (@ 534edd13, Levers 1+2) — MILESTONE MET
═══════════════════════════════════════════════════════════════════════
| class | rep kernel | baseline G | now G | status |
|---|---|---|---|---|
| transposed/strided | transposed_both_scale/in_relu2/out_add | 0.19/0.29/<floor | 0.79-1.0 | PARITY (min 0.785 small) |
| compute wide-rows | wide_row_trig/moderate/chained/skinny | 0.63 | 0.99-1.2 | PARITY |
| broadcast+compute | bcast_rowvec_transcend | 0.63(est) | 1.01/1.006 | PARITY (Lever 2 w8) |
| high fan-in | fanin64/32/96 | 0.87 | 0.82-0.95 | above floor, sub-parity (Lever 3) |
| many live temporaries | manydiff_chains | 0.23 | 0.750 | AT floor, sub-parity (Lever 4) |
DEEPEST DISASTERS (0.19/0.23/0.29/0.63) ALL FIXED. DoD (all classes >=0.75 floor, target parity) MET.
Sub-parity overtime: manydiff 0.75 (borderline floor -> Lever 4 peak-live-sfu), fanin96 0.822 / fanin64
0.869 / fanin32 0.954 (Lever 3 per-operand). transposed small [2048,512] 0.785 (o2).

═══════════════════════════════════════════════════════════════════════
## PER-SHAPE STATUS TABLE (G=tc/seed; floor 0.75; live target tc-parity)
═══════════════════════════════════════════════════════════════════════
Baseline @ 792268d9 (Step-0, the 5 disasters below floor except fanin64):
| kernel/shape | G_seed base | target | status |
|---|---|---|---|
| transposed_both_scale [4096,4096] | 0.188 | ~1.0 | BELOW FLOOR (disaster) — Lever 1 |
| transposed_in_relu2 [8192,8192] | 0.294 | ~0.95 | BELOW FLOOR — Lever 1+2 (coupled) |
| manydiff_chains [4096,4096] | 0.234 | ~1.0 | BELOW FLOOR — Lever 2 (+4?) |
| wide_row_trig_chain [2048,2048] | 0.633 | ~0.99 | BELOW FLOOR — Lever 2 (pure warps) |
| fanin64_2d_add [4096,4096] | 0.869 | ~0.99 | above floor, sub-parity — Lever 3 |

═══════════════════════════════════════════════════════════════════════
## PER-(KERNEL,DTYPE) GEOMEAN + FROZEN-CHAMPION ANCHOR
═══════════════════════════════════════════════════════════════════════
Tier A frozen anchor (INHERITED from prior pointwise champion 268edf3b, cool-GPU do_bench med-9):
swiglu,bf16 0.998 | geglu,bf16 0.996 | residual_add,bf16 0.986 | relu_squared,bf16 0.994 |
bias_gelu,bf16 0.968. (These must not collapse >10% — config-recorder scopes: Tier-A byte-identical is ideal.)
Adversarial anchors: SET at each class's first bank.

═══════════════════════════════════════════════════════════════════════
## LEVER 1 — stride/contiguity-aware tiling — COMMITTED @ 583b93fc (gates pending wf)
═══════════════════════════════════════════════════════════════════════
Design (Gate-F mechanism CONFIRMED via code-investigator, lowered Triton): address strides are
byte-identical seed-vs-oracle; the seed pinned the stride-1 (contiguous) axis to width 1 and grew the
strided axis (stride M) -> uncoalesced. num_warps is a PURE launch-param knob (Lever 2).
Impl: (1) MemoryOpFact.subscript_strides (walker field, raw .stride() per subscript, no new walk);
(2) PointwiseElementwiseFact.contig_block_ids (derived, no walk: block-ids stride-1 for a full-extent op);
(3) _seed_block_sizes: single-contig -> budget on that axis ([1024,1]); CONFLICT (>1 contig axis) ->
_balanced_block_sizes fills reg/occ budget ([64,64]) — a wide single axis strides the other operand
([2048,1]=0.13x disaster). Flat/row-major: contig={last} -> byte-identical (verified).
MEASURED (cfgbench cold-L2 med-9 vs tc):
| kernel | shape | before | after | cfg |
|---|---|---|---|---|
| transposed_both_scale | [4096,4096] | 0.188 | **0.991** | [1024,1] |
| transposed_both_scale | [8192,2048] | ~ | **1.003** | [1024,1] |
| transposed_both_scale | [1024,1024] | ~ | 0.939 | [512,1] |
| transposed_in_relu2 | [8192,8192] | 0.294 | **0.943** | [64,64] |
| transposed_in_relu2 | [4096,16384] | ~ | 0.946 | [64,64] |
| transposed_in_relu2 | [512,2048] | ~ | 0.919 | [16,16] |
| transposed_out_add | [8192,8192] | <floor | 0.923 | [64,64] |
| transposed_out_add | [16384,4096] | <floor | 0.923 | [64,64] |
| transposed_out_add | [2048,512] | <floor | 0.821 | [16,16] |
ALL 9 clear the 0.75 floor (min 0.821). Conflict-case ceiling ~0.96 (genuine transpose conflict).
GATE R (cfg_recorder diff baseline 792268d9 -> 583b93fc): 9 changed cells, ALL transpose; Tier A +
rope + all other adversarial + negatives BYTE-IDENTICAL. No new disaster, no collapse, all-up. ACCEPT.
GATES A/D/F/H — ALL PASS (wf_d9abfa63, ledger @ 583b93fc):
- A x3 refuted=False (reproduced via cfgbench; 2 NOVEL transposed kernels not-in-corpus get same treatment).
- D refuted=False, generality=general (subscript_strides walker field faithful raw .stride(); contig_block_ids
  pure derivation, no walk; stride==1 not a fence; authored divergence kernel confirmed stride-driven not name).
- F mechanism_found=True, inert_fields=[] (block_sizes only, no dead knobs). BETTER NEIGHBOR flagged: asymmetric
  [128,32] (same 4096 area) ~0.96 vs [64,64] ~0.94 on CONFLICT kernels -> OVERTIME item.
- H KEEP, GENERAL (bound 5 unseen patterns incl rank-3 transpose; not fenced to the 3 probes).
LEVER 1 FULLY BANKED @ 583b93fc.
FROZEN ANCHOR (transpose cells, Lever-1 champion): both_scale ~0.99/1.0/0.94; in_relu2 0.94/0.95/0.92;
out_add 0.92/0.92/0.785. (Small shapes [2048,512]/[512,2048] = 0.785-0.94, above floor but sub-default.)
OVERTIME QUEUE: (o1) asymmetric [128,32] for conflict kernels (~+2%, Gate F neighbor); (o2) small-transpose
shapes 0.785-0.94 (occ/reg-shrunk [16,16] tile too small — refine balance_cap floor).

═══════════════════════════════════════════════════════════════════════
## LEVER 2 — num_warps ramp (SFU-intensity keyed) — implemented, gates pending
═══════════════════════════════════════════════════════════════════════
KEY INSIGHT (measured): num_warps optimum is driven by SFU (transcendental) op count, NOT total op count.
horner_poly24 (sfu=0, 77 FMA) wants w4 (w16 REGRESSES 0.0377->0.0411); wide_row (sfu=12, 39 ops) wants
w16 (0.64->0.987). op-count would wrongly warp horner -> the DIVERGENCE that makes SFU the faithful key.
IMPL: PointwiseElementwiseFact.sfu_ops (counted in the EXISTING compute-dtype walk; base op-name in an
SFU set = hardware execution-unit property, not identity). _warps_for(sfu,tile_numel): sfu<3->w4,
<9->w8, else w16; capped by tile_numel//ELEMS_PER_WARP(64) -> a small tile can't feed many warps
(footgun: [1,256]w16=0.50). Emit num_warps only when >4 (flat stays block_sizes-only, byte-identical).
MEASURED warp landscape (cfgbench cold-L2 med-9):
| kernel | sfu | seed cfg | w4 | ramp | note |
|---|---|---|---|---|---|
| wide_row_trig [2048,2048] | 12 | [1,2048]w16 | 0.63 | **0.995** | tc |
| wide_row_trig [512,8192] | 12 | [1,2048]w16 | 0.65 | **0.992** | tc |
| manydiff [4096,4096] | 20 | [1,2048]w16 | 0.23 | **0.750** | tc (AT floor — Lever 4 for parity) |
| manydiff [8192,8192] | 20 | [1,2048]w16 | 0.24 | 0.749 | tc (hair below floor -> Lever 4) |
| wide_row_moderate [2048,2048] | 4 | [1,2048]w8 | 0.90vd | ~1.01vd | w4 was <default |
| chained_activation_2d [4096,4096] | 23 | [1,1024]w16 | 1.0vd | ~1.2vd | |
| skinny [64,65536] | 12 | [1,2048]w16 | 0.80vd | (w16) | w4 was <default |
| bcast_rowvec [8192,8192] | 4 | [1,2048]w8 | 0.90vd | (w8) | |
| horner [4096,4096] | 0 | [1,2048]w4 | — | (byte-identical, correctly NOT warped) |
GATE R (cfg_recorder diff 583b93fc -> Lever2): 18 changed cells, ALL compute-heavy (w4->w8/w16); Tier A
+ transpose + negatives + small-tile shapes (horner, manydiff[128,256]) BYTE-IDENTICAL. All improve
(net progress), no regression, no new disaster (manydiff IMPROVED 0.23->0.75, still ~floor -> Lever 4).
GATES A/D/F/H — ALL PASS (wf_780adc35, ledger @ 534edd13):
- A x3 refuted=False (reproduced; tested 4 NOVEL kernels incl SFU ops NOT in corpus atan/expm1/erfc ->
  correctly w16, fma-heavy 0-SFU -> w4: keys on sfu_ops COUNT, not names).
- D refuted=False generality=general (sfu_ops = hardware execution-unit count, counted in existing walk;
  abs/relu correctly NOT counted; divergence horner(0 sfu, 77 ops)->w4 vs wide_row(12)->w16 confirmed).
- F mechanism_found=True, inert_fields=[] (num_warps carries the win, w4->w16=0.63->0.99); block_sizes+
  num_warps a DOCUMENTED coupling (tile_numel caps warps); w16 is the ceiling (w32 unreachable, correct).
- H KEEP, BROAD unfenced (continuous sfu ramp + tile cap, no name/dtype/shape fence).
LEVER 2 FULLY BANKED @ 534edd13.

═══════════════════════════════════════════════════════════════════════
## BANKED WINS / TRIED-AND-REJECTED / BROADEN-QUEUE / HARD-PILE / HUMAN-REVIEW
═══════════════════════════════════════════════════════════════════════
BANKED: Lever 1 @ 583b93fc (transpose coalescing) — ALL gates PASS.
BANKED: Lever 2 @ 534edd13 (SFU num_warps ramp) — ALL gates PASS.
BANKED: Lever 4 @ 30a007a6 (peak-live-SFU register model) — ALL gates PASS (wf_a13d0249). manydiff 0.75->1.02.
BANKED: Lever 3 @ 2e8d0b9c (fan-in-aware register+bandwidth sizing) — ALL gates PASS (wf_f18a40c3; Gate D:
  LIVE_LOAD_WINDOW=4 FAITHFUL bounded-bound, COALESCE_BYTES=16 FAITHFUL breadth; a refuter's full 82-cell
  parity diff = exactly 14 changed cells, off-curriculum add4_fp32 bpe=20 also lifts -> generalizes).
  fanin 0.82-0.95 -> 0.957-0.979. (a) reg load-window cap min(Σload, 4*max_load) — a combine
  streams loads, doesn't hold all K live; (b) budget divisor cap min(bpe, COALESCE_BYTES=16). Coupling: both
  needed (reg-only->[1,256], budget-only->[1,128]). Broadened (by high bpe) to fanout_flat_many (improved)
  + fp64 (neutral). flat/transpose/compute/manydiff/rope BYTE-IDENTICAL.

STATE: ALL 5 GAP CLASSES AT/NEAR tc-PARITY:
  transpose 0.79-1.0 | compute 0.99-1.2 | broadcast 1.01 | temporaries(manydiff) 1.02 | fanin 0.957-0.979.
DoD (all classes >= floor, target parity) MET + EXCEEDED (4/5 full parity, fanin 0.96-0.98).
levers_since_refactor = 4 -> FIRE REFACTOR-CRITIC (cadence K~4-5). Then Gate E FREEZE + REPORT.

LEVER 3 (fanin) ANALYSIS: fanin's tile is too SMALL (seed [1,128]; oracle [1,2048]) -> over-saturates grid
(131072 tiny programs each doing 64 tiny loads, per-program 64-operand index overhead not amortized).
Root: budget_target = TILE_BYTES//total_bpe = 8192//260 = 31 (÷total starves); inner_floor rescues to 128.
Also reg_cap = 65536//260 = 252 (I/O OVERcount: 64 operands SEQUENTIALLY summed -> peak-live ~2-3, not 64).
The ÷total budget is LOAD-BEARING for the flat family (per-dtype/traffic tuning) so can't be replaced
globally. A reg-overcount fix (peak-live loads) -> reg_cap big -> floor gives [1,256] (still budget-capped,
not [1,2048]). To reach [1,2048] needs the budget fixed (risks flat) OR accept [1,256] near-parity. MEASURE
the fanin tile curve ([1,256]/[1,512]/[1,1024]/[1,2048]) to decide: if [1,256] near-parity -> faithful reg
fix + defer last stretch; else per-operand budget. fanin is ABOVE floor (mild polish, task's own words).
REG-MODEL INSIGHT (for Levers 3/4): reg_bytes_per_elem counts I/O SLABS, which mis-estimates peak-live
  registers: OVERcounts fanin (64 operands are SEQUENTIALLY summed -> ~2-3 live, not 64) and UNDERcounts
  manydiff (20 SFU results all live, counts only I/O=8). A faithful peak-live fix is HARD: horner has ~24
  live FMA values too but does NOT spill (cheap rematerialization) -> peak-live-COUNT would wrongly shrink
  horner. The true separator manydiff-vs-horner = live SFU-results (expensive) vs live FMA (cheap). Both
  fanin (0.87, above floor) and manydiff (0.749, ~floor) are OVERTIME polish, not floor disasters.
OVERTIME: o1 asymmetric [128,32] conflict tile (+2%); o2 small-transpose 0.785-0.94; o3 conflict-case
  more-warps (transposed_in_relu2 [64,64]w16=0.96 vs w4=0.94, +2%, strided-mem latency — separate from SFU).

═══════════════════════════════════════════════════════════════════════
## CRITIC FINDINGS (refactor-critic wf_c0da85de + completeness) — actions
═══════════════════════════════════════════════════════════════════════
REFACTOR-CRITIC:
- P1 COLLAPSE (register-model unification): merge the 3 reg terms (capped Σload + Σstore + peak_live_sfu)
  into ONE real peak-live-bytes liveness (loads+SFU-results, slab-weighted) — retires LIVE_LOAD_WINDOW,
  more principled. Verdict BORDERLINE (elegance-only, NO perf gain — already at parity; touches rope [1,1]
  + flat byte-identity, the two riskiest invariants). Gate D judged LIVE_LOAD_WINDOW FAITHFUL, so the
  "retire a fence" motive is weak. -> DEFER to human-review (recipe: extend the peak_live_sfu liveness to
  slab-weighted loads; validate = full Gate-R matrix + rope [1,1] bind-dump + fanin/manydiff re-bench).
- P3 DEAD-WEIGHT: unreachable default args inner_floor=BLOCK_FLOOR, balance_cap=1<<30 on _seed_block_sizes
  (always called with all args). Cosmetic. -> leave (removing risks a call-site subtlety for zero gain); logged.
- P4 fences: LIVE_LOAD_WINDOW/COALESCE_BYTES/SFU_W8/W16/ELEMS_PER_WARP all Gate-D/H-judged FAITHFUL
  (bounded models / measured crossovers), not curriculum fences. No action.
COMPLETENESS-CRITIC:
- P0 iterated_map_fp32 fires=False: 2 AccumulatorFacts dim_block_ids=(0,1) (BOTH tiled — the `for _ in
  range(6)` Babylonian-sqrt fixed-point `r`, a PER-ELEMENT carried value, NOT a reduction). Disjointness
  `or spec.accumulator_facts` over-excludes it -> default [32,32]. FIX (recipe): admit pointwise when every
  accumulator has ALL dim_block_ids tiled (none collapsed/reduction-axis). -> DEFER to human-review:
  touches the CORE disjointness invariant (risk: mis-classifying a scan/SSM whose carried state is
  all-tiled as pointwise); cannot validate scan-safety in this corpus (no scan kernels). Realistic-but-niche
  (iterative refinement). Recipe: refine `_accum_blocks_pointwise(a) = any(bid is None or bid not in tiled)`.
- P1 dtype coverage (int8/bool/fp16/fp64 not swept through levers): BENCH int8/bool (below).
- P1 fanout/multi-output class absent from shapes_adversarial: fanout_flat_many already lifted by L3
  (improved 1.29->1.32); add to curriculum notes. Low priority.
- P1 Gate E FREEZE pending: DO IT (below).
- P2 misc (LIVE_LOAD_WINDOW sensitivity, broadcast reg-blindness, variant reproduction): logged, low-pri.

═══════════════════════════════════════════════════════════════════════
## HUMAN-REVIEW QUEUE (append-only)
═══════════════════════════════════════════════════════════════════════
1. {register-model unification} BORDERLINE refactor: 3 reg terms -> 1 peak-live-bytes liveness. Elegance,
   no perf gain, rope [1,1]+flat risk. Recipe: slab-weighted load liveness in the existing walk. Reverse:
   device_ir.py build_pointwise_facts reg_bytes_per_elem.
2. {iterated_map disjointness broaden} P0-coverage but CORE-invariant risk: admit all-tiled-accumulator
   pointwise kernels (fixed-point maps). Risk: scan/SSM mis-classification (no scan test kernel here).
   Recipe: refine the `or spec.accumulator_facts` disjointness gate. Reverse: device_ir.py:1268.
3. {inherited} tall-skinny / oversized / stride-blind loop_orders curriculum shapes (from prior pointwise run).
TRIED-AND-REJECTED: (prior run: num_warps flat-family ramp DEFER'd — w8 within-noise, w16 regresses ~10%
  on the FLAT family's 1024-2048 tiles. NOTE: that was moderate-compute flat kernels; the adversarial
  finding is that HIGH-arith-intensity kernels DO want w8-16 → the ramp must key on intensity, not just numel.)
BROADEN-QUEUE: (stride-aware loop_orders was a prior human-review item — now Lever 1.)
HARD-PILE: (none)
HUMAN-REVIEW: (inherited prior items; adversarial to be appended)

═══════════════════════════════════════════════════════════════════════
## LEVER 2 PLAN — num_warps ramp (arith-intensity keyed). GPU-blocked on gate wf (uses GPU).
═══════════════════════════════════════════════════════════════════════
op-count profiler (opcount.py, bind-only) gives the INTENSITY signal per kernel:
| kernel | compute_ops | sfu | seed tile | class |
|---|---|---|---|---|
| swiglu | 9 | 1 | [1024] | flat (w4) |
| geglu | 16 | 1 | [1024] | flat |
| relu_squared | 9 | 0 | [2048] | flat |
| bias_gelu | 18 | 1 | [1,2048] | flat |
| wide_row_moderate | 17 | 4 | [1,2048] | MODERATE |
| bcast_rowvec_transcend | 32 | 4 | [1,2048] | mod-heavy |
| wide_row_trig | 39 | 12 | [1,2048] | HEAVY (w16) |
| skinny_m_wide_n | 39 | 12 | [1,2048] | HEAVY |
| chained_activation_2d | 64 | 23 | [1,1024] | HEAVY |
| horner_poly24 | 77 | 0 | [1,2048] | 0-SFU deep-FMA (op-count vs sfu DIVERGENCE probe) |
| manydiff | 88 | 20 | [1,2048] | HEAVIEST (w8 + smaller tile) |
| wide_fanin_combine | 77 | 16 | [1,512] | heavy+fanin |
SIGNAL: sfu count cleanly separates flat (0-1) from heavy (4-23). op-count captures horner's FMA chain.
ANSWER KEY (Step-0): wide_row [1,2048]w4→w16=0.99 (PURE warps); manydiff [1,2048]w4→[1,512]w8=1.01
  (smaller tile + warps); transposed_in_relu2 [64,64]w4→w16=0.96 (marginal). Footgun: manydiff [1,256]w16→0.50.
PLAN: (1) measure warp landscape w4/8/16/32 × {seed tile, smaller tile} for wide_row/moderate/horner/
  manydiff/chained + flat(swiglu/relu²/bias_gelu) — fit the simplest ramp; (2) add fact field
  compute_ops (+maybe sfu) via the EXISTING build_pointwise_facts compute-dtype walk (no new walk);
  (3) ramp num_warps keyed on intensity, tile-size-aware (small tile → fewer warps). Divergence kernel:
  a big-tile LOW-intensity kernel must stay w4 (ramp keys on intensity, not numel); horner (0 sfu, 77 ops)
  tests op-count-vs-sfu. Must not regress flat family (w4; prior run: w8 noise, w16 regresses flat).

═══════════════════════════════════════════════════════════════════════
## COMPLETENESS/OPEN
═══════════════════════════════════════════════════════════════════════
- iterated_map_fp32 fires=False (has `for _ in range(6)` refinement loop → trips a non-pointwise fact?).
  Out of scope for the pointwise seed IF disjointness is correct; INVESTIGATE (completeness-critic).
- Gate A refuter #1 noted a harmless narration mis-cite: oracle_vs_tc's transposed_both_scale oracle is
  [128,32]@1.003, not [64,64] (I wrote [64,64] from a later cfgbench run). Both prove parity reachable.
- curriculum shapes_adversarial.py: 11 kernels, 107 shapes, validate() OK. Real-anchor documented per kernel.
  Gate-E holdout: test split + variant kernels [wide_row_moderate_chain, fanin32_2d_silu].

═══════════════════════════════════════════════════════════════════════
## DELIVERABLE COMPLETE + VERIFIED @ b890429d (Gate-E-frozen at 2e8d0b9c) — 2026-07-02
═══════════════════════════════════════════════════════════════════════
END-TO-END VERIFIED: (1) config diff baseline 792268d9 -> champion b890429d = 41 changed cells, ALL
adversarial gap-class; Tier A (swiglu/geglu/residual_add/relu_squared/bias_gelu/dyt) + rope + negatives
(softmax/rms_norm/matmul) BYTE-IDENTICAL. (2) cross-family integration: pointwise/reduction/matmul/rope
all compile+correct; disjointness holds (reduction/matmul pointwise_fires=False). (3) Gate E FREEZE PASS
(held-out TEST, BELOW FLOOR none). All 25 gate verdicts PASS. Champion = b890429d, 3 helion files +333/-55.

## (earlier) DELIVERABLE BANKED @ 2e8d0b9c — Gate E FREEZE PASS (2026-07-02)
═══════════════════════════════════════════════════════════════════════
All 4 levers banked, all gates PASS (A/D/F/H/R ×4 + E freeze = 25 ledger entries). All 5 adversarial gap
classes at/near tc-PARITY on the HELD-OUT TEST split; BELOW FLOOR: none. Deepest disasters (0.19/0.23/
0.29/0.63) fixed. Flat family (Tier A) + rope (Tier B) + negatives byte-identical. dtype int8/bool beat
default 1.2-1.3×. REPORT.md finalized. Champion = 2e8d0b9c (4 helion files). NEVER PUSH.
Per never-stop §6.0: DoD is a milestone — KEEP CLIMBING (overtime below).

═══════════════════════════════════════════════════════════════════════
## NEXT ACTION (overtime — never-stop)
═══════════════════════════════════════════════════════════════════════
DoD banked. Overtime worklist status:
- o2 DONE @ b890429d: small-conflict balanced tile uses reg_cap not min(reg,occ) -> [16,16]->[64,64];
  weakest cell transposed_out_add [2048,512] 0.785->0.97, in_relu2 [512,2048] 0.919->1.106; large unchanged.
- o3 DEFER (measured): conflict-case warps MARGINAL+INCONSISTENT — in_relu2 [64,64]w16 +1.2% (0.94->0.954),
  out_add w16 NEUTRAL/worse, both_scale negligible; near 8us noise floor + conflict ceiling ~0.96. Not worth
  a new (non-SFU) warp signal.
- o1 DEFER (measured): asymmetric [64,128] conflict tile +2.4% on in_relu2 but operand-weighting-dependent
  and near noise; symmetric [64,64] is the clean rule. Log.
- DEFERRED-CORE (human-review): register unification (elegance-only, rope-risk, Gate-D-blessed LIVE_LOAD_WINDOW
  faithful so no fence to remove); iterated_map disjointness (scan-mis-classification risk, no scan test).
COMPLETENESS PASS 2 (agent a511e7e6) — DRY. Verified: (1) lever COMPOSITION sound — a 4-lever composite
(transposed-conflict + 10-SFU + 8-fan-in) bind-dumps reg_bpe=28 (capped-load16+store4+peak_live_sfu8),
balanced [32,32], w16 — all 4 levers compose to a sane tile; (2) negatives intact @ b890429d (reduction/
matmul fires=False); (3) ONE new candidate P2 (strided-not-transposed: stepped/dilated slice, no stride-1
axis) MEASURED and found a NON-ISSUE: stride-2/4 pointwise seed [1,2048] BEATS default 1.04-1.08x, G_seed
0.96-0.98 (the stride penalty is inherent/paid by all arms; Helion likely materializes strided-slice inputs,
unlike a preserved .t() view). No relative regression. -> logged, no fix needed.
RUN AT JUSTIFIED STEADY-STATE. All faithful/general/above-noise moves done; remaining = Priority-2
human-review defers (register unification, iterated_map disjointness — both bounded-risk core-invariant,
elegance/niche, logged with recipes) + o1/o3 (marginal near-noise). Gate B not triggered (no below-floor shape).

═══════════════════════════════════════════════════════════════════════
## (historical) NEXT ACTION
═══════════════════════════════════════════════════════════════════════
1. WAIT for lever1-gates workflow (wf_d9abfa63-d93: Gate A×3 / D / F / H). Read verdicts, write to ledger
   AS-RETURNED. Gate A#1 already refuted=false (win real+general). If all PASS → Lever 1 fully banked.
2. Lever 2 (num_warps ramp) — GPU-blocked on the gate wf (it benches). Once GPU frees: run warp landscape
   (cfgbench --cfgs, w4/8/16/32 × seed+smaller tile) for wide_row/moderate/horner/manydiff/chained + flat;
   fit ramp; add compute_ops fact field (existing walk); implement; A/B + gates. Coupled w/ Lever1 for
   transposed_in_relu2 (its [64,64]w16=0.96 vs w4=0.94).
3. Then Lever 3 (fan-in per-operand byte budget), Lever 4 (live-temp, low-pri if manydiff still <parity).
GPU RULE: gate wf uses GPU intermittently (cfgbench spot-checks). Do NOT run GPU jobs until it completes.
