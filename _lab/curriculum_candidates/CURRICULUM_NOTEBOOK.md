# WS2 CURRICULUM EXTENSION — 4 new backward kernels — WORKER NOTEBOOK

Continuation of the WS2 m-reduction run. The prior deliverable (rms/ln/softmax bwd) is COMPLETE
+ fully gated (see `_lab/ws2_notebook.md`, `_lab/WS2_REPORT.md` — treated as LIVE, not stale).
This notebook tracks the EXTENSION: add four new backward kernels to the m-reduction curriculum.
Ledger key: **ws2-curriculum** in `_lab/ws2_ledger.json`. Source of truth = this notebook + ledger.

---
## ENV CONTRACT (verified 2026-06-16)
- Worktree: `/home/calebkim/helion-new-heuristics/helion-ws2` branch `ws2-mreduction`.
- Interpreter: `/home/calebkim/.conda/envs/helion/bin/python`. EVERY run: `cwd=/tmp`,
  `PYTHONPATH=<worktree>`, `CUDA_VISIBLE_DEVICES=<idx>`. GPU 2/3 idle at boot; use ONE, foreground-serial.
- Kernels: `_lab/curriculum_candidates/mreduction_styles.py` (bias_grad/dyt/group_norm/instance_norm
  + torch refs + correctness gate). Shapes: `mreduction_shapes.py` SHAPES (train/val/test/robustness).
- Bench: `_lab/curriculum_candidates/mr_bench.py` (bare backward seed/cfg/oracle/tc, same-process,
  acc-gated, dynamo-reset, median-of-11). Factdump: `factdump_new.py`.

---
## GOAL (method §3, applied to the 4 new kernels × dtypes)
- Per-shape floor: no realistic shape below G=0.75 vs tc (or 0.75×oracle where codegen-bound).
- Per-(kernel,dtype) geomean ≥ 0.85.
- DO-NOT-REGRESS: the 9 forward kernels + rms/ln/softmax bwd champions (Gate R config-recorder).
- Beating tc / oracle parity = overtime. Never stop; never ask (§6.0).

---
## KERNEL TAXONOMY (factdump @ fp32, ground truth)
| kernel | class | fires | reduction over | grad_x |
|---|---|---|---|---|
| bias_grad | A (pure collapse) | **T1** (1 ReductionFact over inner M tile) | M→[N] | none |
| dyt | A (collapse + data) | **T1** (1 ReductionFact over inner M tile) | M→[N] | elementwise |
| group_norm | B (decoupled axis) | **m_reduction** | M(grid)→[C]; grad_x over (Cg,S) | over group |
| instance_norm | B (decoupled axis) | **m_reduction** | M(grid)→[C]; grad_x over S | over S |

- bias_grad/dyt fire T1 (the `.sum(dim=0)` over the inner M tile IS a single ReductionFact),
  NOT m_reduction. m_reduction declines (reduction_facts present) — disjoint, correct.
- group/instance fire m_reduction (0 reduction_facts; materialized C reduction + [C] gw accumulator).

---
## BANKED WIN #1 — lever fix (feature_extent = full resident footprint) @ f8a21c56
**Bug:** `MReductionFact.feature_extent` = gw accumulator width (C only); the 3D norms' resident
grad_x tile is `[inner, C, S]` so inner was sized S× too large → spills worse than generic default.
**Fix (device_ir.build_m_reduction_facts):** feature_extent = product of the input load's
MATERIALIZED feature dims (subscript_block_ids None), read from MemoryOpFact — NO graph walk. The
reshape-split (G,Cg) blocks never appear in the load's indexed_block_ids → no double count.
`feature_extent = max(load_footprint, accumulator_width)` (safety floor). 2D norms: N==N unchanged.
**Gate R:** forward 739 cells UNCHANGED 0-changed; rms/ln/softmax bwd seeds byte-IDENTICAL.
**Result (fp32):** group_norm (1024,64,128,32) G 0.22→**2.01**; instance_norm (1024,32,256) G 0.19→**1.05**.
Seed now [m_cta, inner=1] for C*S=8192 (was [m_cta, 64/128]).
- TODO gates on this lever: Gate D (feature_extent semantics change, reads MemoryOpFact subscript
  fields), Gate H (faithful key), Gate A (adversarial verify the win). [pending full sweep first]

---
## PER-SHAPE STATUS (G = tc/seed; floor 0.75; geomean target 0.85)
- fp32 train sweep: IN PROGRESS (/tmp/mr_train_fp32.json)
- KNOWN below-floor (pre-climb): bias_grad (16384,1024)=0.56, dyt (16384,1024)=0.62 [T1 seed quality].

---
## BANKED WIN #2 (pending gates) — re-route bias_grad/dyt T2→m_reduction
**Problem:** bias_grad/dyt fire T2 (the `.sum(dim=0)` over the inner M tile = 1 ReductionFact).
T2's persistent feature R_BLOCK seed `[1,8192] w16` floors the grid to 1 → ~16384 tiny partials →
expensive finalize. fp32 train: bias_grad ALL 14 below floor (G 0.33-0.64), dyt 12/14 below (0.43-0.69).
**Oracle (quick) confirms reachable, NOT codegen-bound:** bias_grad (16384,1024) oracle beats tc 1.27×;
dyt (8192,4096) oracle beats tc. BUT the real winner is the m_reduction byte-cap seed, not the quick
oracle's under-converged [4,4]/[128,128]: CONFIG BATTERY (8192,4096) found `[64,2]` (M_CTA occ + inner
byte-cap) gives bias_grad G=0.87, dyt G=1.52 — both clear floor, dyt beats tc.
**Fix (role-based selection, the task's intended rework):**
- `build_m_reduction_facts`: drop the blanket `if reduction_facts: return []`; instead decline only
  if a ReductionFact is over an axis OTHER than an inner row tile (a genuine feature reduction). A
  pure-collapse M-reduction registers its `.sum(dim=0)` as 1 ReductionFact over the inner tile → fire.
- `_triton_reduction_eligible` (T1/T2): add `and not m_reduction_facts` → T1/T2 yield to m_reduction.
- `m_reduction.is_eligible`: drop `not reduction_facts` (disjointness now lives in the builder).
**Gate R PASS:** forward 739 cells 0-changed; rms/ln/softmax bwd seeds byte-IDENTICAL. Only bias_grad/dyt
re-route (now fire m_reduction, seed [128,8] @ (16384,1024)). [pending: full sweep, Gate D/H/A]

## KEY INSIGHT — the m_reduction byte-cap seed generalizes to class A
The same [M_CTA=np2(M/num_sm), inner=byte_cap] seed works for ALL m-reductions:
bias_grad (pure collapse), dyt (collapse + elementwise grad_x), group/instance (decoupled grad_x),
rms/ln (same-axis grad_x). The M_CTA-occupancy lever (vs T2's floored grid=1) is what they all needed.

## bias_grad — CODEGEN-BOUND pure sum, scattered optima (autotuner territory)
fp32 train geo=0.834, 3 below-floor: (16384,1536)=0.62, (8192,2560)=0.70, (8192,3072)=0.67 (all
NON-pow2 N -> padded-load penalty). Real oracle: codegen-bound (oracle 0.82-1.27x tc, scattered
per-shape configs [16,16]w4 / [128,4]w32 / [64,64]w8ns3 — NO clean faithful rule). oracle GEOMEAN
~1.0 so it's not a hard ceiling, but matching the autotuner's per-shape optima needs Product B.
[16,16]w4 clears the 3 marginal cells BUT regresses (16384,1024) 1.14->0.79 (pow2-low-N wants large
m_cta; non-pow2-mid-N wants small) -> a REGRESSING TRADE, rejected. **DECISION (logged):** keep the
current m_reduction seed for bias_grad (geo 0.83, strong codegen-bound start); the 3 non-pow2 marginal
cells are STUCK (codegen padding artifact + scattered optima conflicting with pow2 cells) -> Gate B
cleared by the oracle; autotuner (Product B) handles per-shape. dyt (also re-routed) clears cleanly
(geo 1.37) because its dominant [M,N] elementwise grad_x is well-served by the byte-cap seed.

## GATE VERDICTS (workflow wf_1516e5dc-942 @ af969c5b; independent repro by driver)
- **Gate R PASS** (x2): forward 739 cells 0-changed; rms/ln/softmax bwd seeds byte-IDENTICAL (lever fix + re-route).
- **Gate D PASS 3/3**: build_m_reduction_facts is a pure derived fact (NO graph walk — grep-clean of
  device_ir.graphs/node.users/_classify_load_dataflow); feature_extent faithfully tracks the resident
  footprint (divergence tests FAILED to refute — reshape G/Cg blocks absent from the load's
  indexed_block_ids so no double-count); disjointness key is structural block-id set-membership (not a
  dtype/identity fence); dtype kept a FACTOR in the byte budget; (None,) non-pow2 fallback is sound
  documented padding-reproduction gated on len(mat_blocks)==1.
- **Gate H KEEP 2/2**: feature_extent=resident-footprint (faithful bytes/footprint key, catastrophe
  rescue) and the role-based re-route (faithful structural key) both belong in the core.
- **Gate A PASS** (2/3 skeptics PASS; the 1 REFUTE was procedural = missing the driver-run independent
  reproduction, now provided). INDEPENDENT REPRO (fresh-agent-authored /tmp/indep_repro.py, run serial,
  TWO timers do_bench+cuda.Event agree): group_norm (1024,64,128,32) G 2.07/2.26 acc✓; dyt (8192,4096)
  G 1.53/1.60 acc✓ — matches mr_bench (2.01, 1.53). Wins real, reproducible, correctly measured, faithful.

## GATE E bf16/fp16 TEST (firewall, half-precision) — NO OVERFIT (closes completeness gap #1)
| kernel | bf16 TEST tract | fp16 TEST tract | verdict |
|---|---|---|---|
| dyt          | 0.95 (full, 0 below) | 0.97 (full, 0 below) | tracks train ✓ |
| bias_grad    | 0.69 (5 below)       | 0.67 (5 below)      | consistent (codegen-bound) |
| group_norm   | 1.75 (0 below)       | 1.72 (0 below)      | tractable tracks train ✓ |
| instance_norm| 1.62 (0 below)       | 2.20 (0 below)      | tractable tracks train ✓ |
No half-precision overfit: held-out tractable cells track train; wide-S kernel-bound on TEST too.

## bias_grad fp32 geomean RECONCILED (completeness gap #3) ≈ 0.81
3 runs of (16384,1024) same config [128,8]: 0.757 / 1.136(outlier) / 0.754 -> true ~0.75 (tc
cross-process jitter inflated the AB2 run). Reconciled fp32 train geo = 0.806 (run3) ~ 0.83 (AB2);
report ~0.81. 3 consistently-below-floor cells: (16384,1536)=0.59, (8192,2560)=0.70, (8192,3072)=0.67.
Verdict (below the 0.85 bar, codegen-bound, HRQ #2) is ROBUST to the noise.

## GATE E (TEST firewall, SOLE held-out read @ 816569a4) — NO OVERFIT
| kernel | TEST full geo | TEST tractable geo | train tractable | verdict |
|---|---|---|---|---|
| dyt          | 1.24 (0 below) | 1.24 | 1.37 | tracks ✓ |
| bias_grad    | 0.79 (2 below) | 0.79 | — | consistent (codegen-bound) |
| group_norm   | 0.54 (4 below wide-S) | 1.62 (0 below) | 1.86 | tracks ✓ |
| instance_norm| 0.43 (4 below wide-S) | 1.44 (0 below) | 1.78 | tracks ✓ |
TEST tractable cells track train tractable -> no curriculum memorization; wide-S kernel-bound on TEST
too (structural, reproduced on held-out). Gate E PASS.

## FULL DTYPE GEOMEANS (train split, acc-gated; full-split incl. kernel-bound wide-S)
| kernel | fp32 | bf16 | fp16 | below-floor (per dtype) | note |
|---|---|---|---|---|---|
| dyt          | 1.37 | 1.00 | 1.00 | 0 / 1 / 1 | CLEARS BAR ✓ |
| bias_grad    | 0.83 | 0.66 | 0.67 | 3 / 10 / 10 | codegen-bound pure sum (oracle ~ties tc, scattered optima -> Product B) |
| group_norm   | ~0.64| 0.82 | 0.80 | 8 / 7 / 6 | tractable-F clears (G 1.1-2.9); WIDE-S kernel-bound |
| instance_norm| ~0.70| 0.71 | 0.86 | 6 / 6 / 7 | tractable-F clears (G 1.0-2.6); WIDE-S kernel-bound |

## OPEN WORKLIST
1. [DONE] bf16/fp16 sweeps all 4.  [DONE] Gates R/D/H/A.
2. Report (WS2_CURRICULUM_REPORT.md).  Gate E (TEST firewall) at freeze.
3. Overtime: bias_grad scattered-optima class-A lever (no clean rule found); wide-S re-author (HRQ).

## GATE F PASS (re-route mechanism, code-verified @ 1057e3b5)
Read generated Triton (to_triton_code, no bench) for bias_grad (8192,4096) under T2 [1,8192] vs
m_reduction [64,2]: T2 floors grid block to 1 -> 8192 partials -> gb_blocks [8192,4096]=128MiB written
then re-reduced over 8192 rows; m_reduction m_cta=64 -> 128 partials -> [128,4096]=2MiB, finalize over
128 rows (64x fewer partials). Byte-capped inner=2 keeps resident [inner,N]=32KiB. mechanism_found=True,
matches_story=True, inert_fields=[] (seed sets only block_sizes/num_warps/num_stages/pid_type — same as
rms/ln). Boundary: win vanishes (-> identical to T2) for M <= ~2*num_sm (~264); monotone above; degrades
gracefully (never regresses). PASS.

## COMPLETENESS-CRITIC GAPS (wf_0531817c-405) + dispositions
1. [CLOSED] bf16/fp16 TEST not swept (Gate E firewall only fp32) -> ran bf16/fp16 TEST (below).
2. [ALREADY DONE] bias_grad stuck-cell oracle -> /tmp/orc_bias3.json HAS (16384,1536)=[16,16]w4 0.82,
   (8192,2560)=[128,4]w32 1.00 (critic looked at wrong file). Both confirm codegen-bound (oracle ties/
   below tc) with scattered configs. HRQ #2 substantiated.
3. [RECONCILED] bias_grad (16384,1024) G 0.76 (AB) vs 1.14 (AB2) same config [128,8] = cross-process tc
   jitter (footgun #4/#13: seed in-process stable, tc baseline varies run-to-run). The "below-bar"
   VERDICT is robust to it (0.66-0.83 all < 0.85); re-bench below pins the number. bias_grad is a
   marginal codegen-bound kernel -> its geomean is inherently noise-sensitive (documented).
4. [SKIP+LOG] wide-S knob-sweep artifact not persisted -> the ceiling is independently corroborated by
   the kernel source (no S-tile knob; 2-pass c1/c2 forces full [C,S] resident); critic agrees HRQ #1
   correct. Minimal artifact: /tmp/cfggrid_gn.py output (logged in notebook).

## DELIVERABLE STATUS (fp32, pre bf16/fp16)
- **dyt**: geo 1.37, 0 below floor. CLEARS BAR. ✓
- **bias_grad**: geo 0.83, codegen-bound pure sum (3 non-pow2 STUCK cells). Near faithful seed ceiling.
- **group_norm/instance_norm**: CLEAR the bar on every kernel-tractable (small-mid C*S) shape
  (G 1.1-2.6); wide-S cells KERNEL-bound (no S-tiling) -> documented #1 follow-up (re-author to tile S).
- Core seed extension (lever fix + role-based re-route) BANKED + Gate-R-clean (fwd 739/0, champions byte-id).

## WIDE-S group/instance — CODEGEN/KERNEL-BOUND (knob sweep + structural)
The 3D norm kernels grid ONLY over N/B (batch) and materialize the full resident [inner,C,S] tile
for grad_x (x_hat reused after the (Cg,S)/(S) stats reduction). For WIDE C*S (vision GroupNorm,
S=H*W=thousands) two structural problems, NEITHER seed-tunable:
  1. Resident [1,C,S] fp32 spills to local mem (256KB @ C*S=65536 up to 4MB @ C*S=1M); inner=1 is
     already the floor (byte cap can't shrink further), S is materialized (no S-tile knob).
  2. Low-N shapes (N=8-16) launch only N CTAs -> massive under-occupancy; grid is over N only.
tc tiles S (2-pass) -> ~8x faster. KNOB SWEEP (16,512,1024,32): seed [1,1]w32=0.13 is the BEST over
all warps(8/16/32)/stages -> CONFIRMED codegen-bound for THIS kernel authoring. (Full autotune HANGS
on the spilling kernel - large-block trial configs won't compile/bench; the tractable config space
is block=1 + warps/stages, which the knob sweep covers.)
fp32 train below-floor (8/14 group_norm): (8,256,4096,32)=0.06 (16,512,1024,32)=0.13 (16,256,1024,32)=
0.29 (128,256,256,32)=0.43 (16,512,512,32)=0.45 (32,256,512,32)=0.49 (32,128,1024,32)=0.52 (64,256,256,32)=0.65.
**DECISION (logged, BORDERLINE -> HRQ):** the seed is OPTIMAL-given-the-kernel (beats generic default,
inner=1+w32) - this is a CURRICULUM-KERNEL authoring limitation, not a seed limitation. The fix is to
re-author group/instance_norm to TILE S (2-pass: stats over S-tiles, then apply) like a real
GroupNorm bwd / tc. That is a KERNEL change beyond the seed deliverable -> deferred to the human +
re-author recipe in HRQ. group/instance geomeans on the FULL train split are dragged below 0.85 by
wide-S; they clear the bar handily on every kernel-tractable (small-mid C*S) shape.

## TRIED-AND-REJECTED
- quick-oracle [4,4]/[128,128] for bias_grad/dyt: under-converged; the m_reduction byte-cap [64,2]
  beats them. (Quick autotune unreliable for a parity/win — confirm winners by direct cfg bench.)
- wide-S group/instance bigger m_cta/inner ([256,256] etc.): tanks (resident explodes); [1,1] is best.

## CORRECTION — robustness "no crashes" claim was WRONG (silent truncation, 2nd critic pass)
My earlier robustness sweep reported "0 crashes, all acc=True" — FALSE. The sweep's grep only saw cells
that emitted a ROW; the CRASHING cells emitted nothing, so I wrongly read "0 acc=False" as a pass
(method footgun: silent truncation reads as covered). VERIFIED: 4/6 group/instance robustness canaries
(the NON-pow2-C ones) CRASH AT COMPILE: group_norm (256,96,49,32) -> "shape '[32,3]' invalid for input
of size u1"; same for (16,320,256,32), instance_norm (256,96,49), (16,320,256). ROOT CAUSE: the kernels
do `weight[:].reshape(g,cg)` (group L127) / `weight[:].reshape(1,c,1)` (instance L202); for non-pow2 C
the [C] weight load is padded to next_pow2(C), so reshape to the TRUE [G,Cg]/[1,C,1] fails (config-free,
EFFORT=none, NO seed involved -> curriculum-KERNEL bug, not a seed bug). instance_norm is fixable
(weight[None,:,None], no reshape); group_norm's [G,Cg] reshape of padded non-pow2-C is harder. The
pow2-C robustness canaries (8,64,64,*) DO pass (acc✓, no crash) -- the bf16/fp16 perf canaries also ran.
-> HRQ #3 (non-pow2-C curriculum-kernel bug). The SEED is unaffected (it never runs; perf bars unchanged).
UPDATE: tried the weight fix (instance_norm `weight[None,:,None]`): it does NOT break pow2 (512,64,128
COMPILED acc✓) but does NOT fix non-pow2-C either — instance_norm (256,96,49) still HANGS at compile
(the next_pow2 padding for C=96 affects the whole kernel — x/dy reshapes, c1/c2 reductions — not just the
weight line). So the fix is deeper than one line -> REVERTED (kept the validated kernel). HRQ #3 stands
as a curriculum-kernel limitation (non-pow2 C unsupported), same bucket as HRQ #1 (kernel re-authoring).

## DEFERRED-HARD-PILE & BORDERLINE
- **[STUCK, Gate B = knob sweep + structural] group/instance_norm WIDE-S** — kernel-authoring-bound (no
  S-tiling). Seed is optimal-given-the-kernel ([1,1]w32 beats generic [32,32]). Re-author to tile S =
  the fix. Full autotune HANGS on the spilling kernel (un-compilable large-block trials); the tractable
  config space (block=1 + warps/stages) is swept and confirms ceiling. → HRQ #1.
- **[STUCK, Gate B = fp32 oracle] bias_grad non-pow2 marginal cells** — codegen-bound pure sum; oracle
  ~ties tc but per-shape optima scattered ([16,16]w4 / [128,4]w32 / [64,64]w8ns3), no clean faithful
  single-rule; [16,16]w4 clears marginals but regresses (16384,1024) 1.14→0.79 (rejected trade). →
  Product B (autotuner) territory; seed (geo 0.83) is a strong start. → HRQ #2.

## HUMAN-REVIEW QUEUE (append-only, ranked)
1. **Re-author group_norm/instance_norm to TILE the spatial dim S** (2-pass: stats over S-tiles, then
   apply) like a real GroupNorm/InstanceNorm bwd / torch.compile. The given single-pass kernels
   materialize the full [inner,C,S] resident tile (up to 4MB) and grid only over N → wide-S (vision
   GroupNorm S=H*W=thousands) is 8-16x slower than tc REGARDLESS of seed config (block sizes forced to
   1). This is a CURRICULUM-KERNEL limitation, not a seed limitation; the m_reduction seed is
   optimal-given-the-kernel. Re-authoring (+ a seed S-tile lever) would let group/instance clear the bar
   on wide-S. Deferred because it's a kernel-engineering change beyond the seed deliverable + risks the
   recognizer. Where to look: mreduction_styles.py group_norm_bwd/instance_norm_bwd; the seed would then
   need an S-tile reduction_loops lever. (Alternative: relax the device_ir roller to ROLL the grad_x
   (Cg,S)/S reduction so S gets a reduction_loops knob — but that's global + risks the rms/ln champions.)
3. **Non-pow2-C curriculum-kernel crash** (group/instance_norm): `weight[:].reshape(g,cg)` /
   `.reshape(1,c,1)` + the whole-kernel next_pow2 padding for non-pow2 C make group_norm (256,96,49,32),
   (16,320,256,32) and instance_norm (256,96,49), (16,320,256) CRASH/HANG at compile. Realistic
   GroupNorm has C=96/320 (non-pow2, divisible by G). Config-free (no seed) -> curriculum-kernel bug.
   The 1-line weight fix is insufficient (kernel hangs elsewhere). Where to look: handle non-pow2-C
   without reshaping the padded [C] load — restructure the [G,Cg]/[C] reshapes (same bucket as HRQ #1).
   The SEED is unaffected; robustness perf bars unchanged.

2. **bias_grad pure-collapse seed** is below the geomean bar (codegen-bound vs tc split-reduction;
   scattered per-shape optima). Product B (the autotuner) closes the gap; the Product-A seed is a strong
   start (geo 0.83 fp32, up from 0.39). A class-A (no-grad_x-store) lever could lift marginals but every
   variant tried is a regressing trade. Where to look: TritonMReductionHeuristic; the class-A signal =
   no store subscripted by an inner row-tile block.

## COMPLETENESS LOOP-UNTIL-DRY — 3rd pass DRY (run at faithful in-scope ceiling)
Pass1: 4 gaps (closed). Pass2: 1 genuine gap (non-pow2-C robustness crash falsely reported -> corrected
+ HRQ #3). Pass3: **DRY** — verified cell counts match curriculum (122 total, 0 acc-fails in measurable
splits); non-pow2-C crashes directly confirmed (4/6 canaries, both pow2-C pass); "0 below-floor on
kernel-tractable shapes" holds. Two refinements (both favor the deliverable):
- The AB-vs-AB2 bias_grad spread is NOT tc jitter for the non-pow2-N cells — it's stale config: the AB
  file predates the non-pow2 (None,) fallback (those cells ran the slow T2 [1,8192] there, m_reduction
  in AB2). The CURRENT seed is DETERMINISTIC (3 fresh processes return the fast m_reduction config for
  every cell); headline bias_grad ~0.81-0.83 / dyt 1.37 reproduces. (The pure-pow2 (16384,1024) residual
  spread is ordinary tc noise; run3=0.754 confirms the AB2 1.136 was the high outlier.)
- instance_norm (256,128,256) VAL fp32 read 0.653 (long-sweep under-read / phantom cliff); a clean short
  re-bench gives G=0.968 (ABOVE floor). So VAL instance tractable is actually 0 below-floor too.
No untried faithful in-scope seed lever remains: every below-floor REALISTIC shape is wide-S (HRQ #1,
kernel) or bias_grad codegen-bound (HRQ #2, Product B). The Product-A seed is MAXIMIZED for this
curriculum + these kernels.

## NEXT ACTION (resume anchor)
Core seed extension BANKED + fully gated (R/D/H/A) @ af969c5b. dyt clears the bar all dtypes;
group/instance clear on kernel-tractable shapes; bias_grad + wide-S are structural limits (HRQ #1/#2).
Remaining: write WS2_CURRICULUM_REPORT.md, optional Gate E TEST-firewall at freeze, then overtime on
the HRQ items only if a faithful in-scope lever appears. Do NOT git push.
