# Reduction-Unification Hillclimb — NOTEBOOK (resume-state, source of truth)

> **READ THIS FIRST on any re-invocation.** The 5 counters + checklist line-status at the
> top tell a fresh context "not done" WITHOUT re-deriving it. Trust this log over context
> (method §6.1). This is a NEVER-STOP run — DoD is a milestone to bank, then keep climbing.
> Task: `prompts-lab/tasks/reduction-unification-task.md`. Objective = FAITHFULNESS >
> GENERALITY > perf (perf demoted to a 10%-vs-fc1dbaa0-champion guardrail; NO torch.compile).

## ⏱ THE 5 SCALAR COUNTERS (the external gradient — log every N levers)

| # | counter | bootstrap | floor/target | CURRENT | owner |
|---|---------|-----------|--------------|---------|-------|
| 1 | SPECIAL-CASE COUNT | **6** | floor (irreducible) | **2** (Defects 1/2/3 re-keyed + bolt-on deleted + 2 dead consts + len==1 fence gone; residual = budget-constant cluster + max-extent tie-break, both Gate-D faithful/acknowledged) | refactor-critic / Gate D (data-flow) |
| 2 | DIVERGENCE-TEST DEBT | 0 | 0 | **0** (Gate-D r2: Defect-1+M_COLLAPSE_TILE_BYTES PASS; Gate-D r3: grid_collapse_block_ids+_resident_itemsize PASS, geo-split+dominant-pick acknowledged-heuristic) | fresh Gate-D agent |
| 3 | PERF-REGRESSION COUNT | 0 | 0 | **0** (all refactor commits byte-identical on 447 cells; SELECTION-ONLY proven — diff confined to 3 seed-config files, lowering untouched. The occ-override that regressed grpo 1.6× was REVERTED.) | Gate R |
| 4 | FAILED-FALSIFICATION TRIPLE | {open_RED:?, UNREACHED:?, clean:0} | {0,0,≥3 consec} | **{open_RED:0, UNREACHED:0, clean:0→pending Gate-T r5}** ⏳ re-opened @ Gate-T r4, both holes fixed @1650a3c6 | Gate T (INDEPENDENT) |
| 5 | DEFERRED-UNIFICATION DEBT | 0 | 0 | **0** | Gate B (faithfulness-DEFER variant) |

### ⏳ DoD TOTALITY conjunct RE-OPENED @ Gate-T r4 (072e7e08), both holes FIXED @1650a3c6, Gate-T r5 in-flight
The other 4 conjuncts still hold (STRUCTURE/FAITHFULNESS/PERF/COVERAGE — all unchanged, byte-identical).
Gate-T r4 (independent, 3 strategy-divergent adversaries) found **2 totality holes in the grid-collapse +
independent-loop FUSION regime** the r3 DRY sweep had not reached — both UNDER-SIZES the mechanical checker
misses (it only catches floor-to-1), found by A/B DE-FUSE:
 - **Finding 2** (fix 9f094896): a grid-collapse fused with a separate normalize/apply loop floored the
   apply tile to block_size=1 (`_pinned_inner_resident_elems` rolled-exclusion checked only
   `reduction_loops` membership = EMPTY on the user-tiled track → reduction-role axes leaked into the apply
   loop's co-residency denominator). FIX: also exclude any `info.reduction` axis. `[64,64,1]→[64,64,4096]`.
 - **Finding 1** (fix 55ab8110): a grid-collapse fused with an INDEPENDENT user-tiled loop whose BACKED
   reduction (extent coincidentally == the collapse's unbacked placeholder 8192) the dominant-fact picker
   selected as primary → the m-collapse primary-rdim override sized it to the 64-elem occupancy block (128×
   under). FIX: gate the override on the PRIMARY axis being UNBACKED (extent-provenance, scoped to primary);
   split the grid-CTA occupancy sizing onto the unscoped per-fact `grid_origin` (already loop-scoped to
   `grid_collapse_block_ids`). `[64,2048,64]→[64,2048,2048]`, grid 64 preserved.
BOTH **byte-identical on the 447-cell corpus** (zero-diff vs fc1dbaa0). Permanent RED→GREEN probe
`probes/gateT/grid_collapse_fused_indep_loop.py` (2/2 RED @ parent 8d722d13, 0/2 @ HEAD). Full probe corpus
(gen/ + gateT/ + Gate-D + coverage + corpus) re-swept GREEN @ HEAD. Ledger @ ea24dd1a: open_RED 2→0, clean
reset to 0. **Gate-T round 5 (wf_9caedffd-2db) running** to re-establish clean_divergent_sweeps≥3 + re-bank.
The earlier DRY was @479df334; the re-open is honest — this is a NEVER-STOP run, a found hole resets clean.

PRIOR MILESTONE (basis for the other 4 conjuncts): Gate-T DRY @479df334, 3 consec sweeps, model-well
measured 0-RED on 7 newly-fired cells; STRUCTURE L1/L2/L4 greppable, L3/L5 met-in-substance; counter2=0 all
keys Gate-D PASS/acknowledged; counter3=0 selection-only byte-identical; COVERAGE 6 RED→GREEN holes + P1-P5
guards. Completeness-critic's 2 framing blockers FIXED. See REPORT.md. HEAD has _lab commits on top of the
heuristic (helion/ tree = the r4-fixed seed).

**Banked lever 2 (SHA 2bf66355) + lever 3 (SHA 27506dea):** Defect 2 — PFA→GRO re-key (byte-identical,
Gate-D fixture PASS) + the bolt-on override FOLD into _build_block_sizes (byte-identical). Counter-delta:
special-case 5→4 (the bolt-on override smell removed; GRO is a faithful property not a recognizer).
GRO Gate-D fixture proves ≥2-distinct-families. NOT cosmetic. Satisfies STRUCTURE L1.
divergence-debt: +1 (M_COLLAPSE_TILE_BYTES=32768 is a named budget constant kept as the Move-2 inner
byte-cap — owes an EQUAL-FOOTPRINT divergence test proving it bites by occupancy-vs-residency, not by
the kernels measured; pre-registered, non-blocking per synthesis risk-2).

**Banked lever 1 (SHA 9f2d9bbf):** Defect-1 re-key. Counter-delta: special-case 6→5,
divergence-debt 0→1. Byte-identical (447/447). NOT cosmetic. Satisfies STRUCTURE L2.

**Gate-D verdict on lever 1 (ledger_gateD_defect1.json, refuted=true):** The materialized-primary
fix is REAL + byte-identical (rms_norm_bwd: OLD proxy=1 lie → NEW=4096=REAL). BUT the agent built a
misclassification kernel: for a STANDARD-track LOOPED rolled reduction the new key emits
`np2(size_hint)` (full extent) where the TRUE resident width is the rolled CHUNK (16384) — an
over-count. **CRUCIAL caveat the agent reported: this over-count is BYTE-IDENTICAL to OLD (the old
rolled branch was ALSO np2(size_hint)) — a PRE-EXISTING latent gap the edit faithfully preserves,
NOT a new regression; conservative-safe (errs toward smaller M-widen, never a spill).**

**Attempted the "true resident width" refinement (pass r_block as primary_resident_width) →
REJECTED by Gate R.** It changed 12 looped-wide-N cells (cross_entropy/sum/ce_ls_zloss at
N=65536-151936) from bs=[1]→[2], and benched cross_entropy 4096×65536: 527µs→711µs = **1.35×
REGRESSION** (>10% bar). Mechanism: widening M from 1→2 on a wide GRID-SATURATED looped reduction
halves the grid + doubles per-program work → worse occupancy/latency. The OLD over-count
ACCIDENTALLY produced the better config (M=1) for the WRONG stated reason. The `_m_axis_occupancy_cap`
permitted tile=2 (grid 4096/(132·8)≈3.9→pp2=2) but occupancy alone doesn't capture the
wide-reduction latency penalty. REVERTED (kept the byte-identical lever 1). Counter 3 stays 0.

**FINDING (logged for the refactor / human-review):** the Defect-1 ACCESS key honestly computes the
FULL-SLICE ACCESS width (not the true looped residency). Its divergence-debt is retired by NARROWING
the claim to "access width, conservative for looped" (the occupancy cap is the real M-widen guard) —
OR by a deeper fix to the M-widen path (footprint-admits-but-occupancy/latency-rejects). The latter is
a perf-coupled change that REGRESSES if done naively → it is a BROADEN/refactor-queue item, NOT part
of the byte-identical Defect-1 fix. Divergence-debt for lever 1 = retired-with-narrowed-claim (the key
IS faithful to "full-slice access width"; the looped-residency gap is a SEPARATE latent M-widen issue,
pre-existing, byte-identical, conservative-safe). Re-run a fresh Gate-D on the NARROWED claim to
confirm debt→0.

**Counter 1 — the 6 named special cases at bootstrap (fc1dbaa0), the de-bloat worklist:**
1. **Defect 1** — `r_block_resident` 3-way branch keyed on `reduction_loops`-membership /
   contingent rolling (`triton.py` `_build_block_sizes` ~849-854). CONTINGENT-STATE KEY.
2. **Defect 2** — `per_feature_accumulator` bolt-on OVERWRITES the generic sizer
   (standard `get_seed_config` ~1225-1256 `if fact.per_feature_accumulator:`; user-tiled
   ~1338 `is_m_collapse = fact.per_feature_accumulator`). BOLT-ON OVERRIDE + PER-KERNEL-SHAPED FIELD.
3. **Defect 3** — `ReductionRole` (TILED/FLOORED_ROW/DECLINED) stored-decision enum
   (`device_ir.py` ~99-130). STORED-DECISION ENUM (decisions-as-properties).
4. **Duplicated cap-stack** — the same `min`-of-residency-caps written ~3× (`_reduction_rblock`
   r_block, the `m_block_ids` branch, the `red_values`/catch-all loop in `_build_block_sizes`).
5. **Topology-proxy m_block_ids** — `grid_ids - tiled_reduction_axes` (device_ir) vs the
   `m_block_ids` branch's own footprint logic; "is it on the grid" proxies for origin.
6. **Latent hackiness (Issues 11/12/13)** — `ROW_PERSIST_MAX_BYTES`/`PERSISTENT_REDUCTION_MAX`
   reused-estimate constants, shape-conditional re-floor-to-1, max-extent dominant-reduction
   tie-break. (Counted as ONE smell-cluster of curriculum-tuned constants / fences.)

> NOTE the §3a.1 rule: a best-guess default that FIRES BROADLY is NOT a special case; a FENCE
> narrowing to the curriculum is. The distinction is the divergence test (§5d). Counter 1 is a
> DATA-FLOW verdict by a fresh owner (a rename/inline still counts), never a symbol-grep.

## PRE-REGISTERED STRUCTURE CHECKLIST (§8) — see `STRUCTURE_CHECKLIST.md` (pinned at bootstrap)

## DEFECT-2 PROGRESS (M-collapse / per_feature_accumulator bolt-on)
- **Design judge-panel (workflow w9weh8r9q, 3/3): park PREMATURE.** Faithful re-key =
  `grid_reduction_origin` (GRO): `bool(m_block_ids)` AND some reducing axis has an UNBACKED extent.
  Provenance-distinct from PFA, agrees on corpus, fires on non-norm `col_energy_collapse`, excludes
  near-misses. Gate-D fixture `probes/gro_divergence.py` PASS. Full synth: `design_mcollapse_workflow.json`.
- **Step 1 DONE (SHA 2bf66355, byte-identical):** PFA field+populator+2 read-sites → GRO. The
  `per_feature_accumulator` SYMBOL is gone from live code (only comparative comments remain). L1
  partially satisfied (symbol replaced by a faithful property w/ Gate-D fixture); the bolt-on
  OVERRIDE STRUCTURE still exists (gated on GRO) → Step 2 folds it.
- **Step 2 NEXT:** fold Move-1 (grid-block occupancy) into `_m_axis_occupancy_cap` (grid-origin arm),
  keep Move-2 (inner byte-cap `_m_collapse_inner_byte_cap`) as a named residency budget, DELETE the
  two post-hoc block_sizes-overwrite blocks (standard ~1242, user-tiled ~1357) + `_m_collapse_grid_block`.
  Verified-byte-identical arithmetic (grid blocks 16/16/1/1/16, inner caps 8/8/16/4/8). THIS drops
  counter 1 (removes the compute-then-overwrite bolt-on smell). M_COLLAPSE_TILE_BYTES owes an
  equal-footprint divergence test (counter-5 logged, not blocking).

## GENERATOR ROUNDS (steady-state engine)
- **Round 1 (8 cells):** 8 GREEN, 0 holes. multifact_3way (relaxed gate + dominant pick), grid_origin_nonnorm
  (GRO fires non-norm), carried_3d_highdim, unbacked_inner_nonbackward (GRO non-bwd), etc. Corpus in gen/.
- **Round 2 (8 cells, saturation+adversarial):** 8 compile-GREEN but **2 GENUINE FINDINGS**:
  - **adv_gro_falsify (RED→FIXED, SHA bfa012cc):** backed_col_sum_collapse (reduction OVER a tunable
    grid axis) had block 0 collected as a secondary reduction → sized full-extent → grid COLLAPSED to
    1 program. FIX: `_user_tiled_reduction_block_ids` now excludes PARTIAL_GRID axes (role classifier).
    Grid axis now floors; seed beats/ties default (0.612 SEED-WINS at (8192,2048)). Byte-identical.
  - **sat_bf16_reduction (MISSING-AXIS→FIXED, SHA 2e5926ab):** residency caps divided by input
    itemsize (bf16=2) but bound the fp32 accumulator → under-count 2x → wrong PERSIST at N=65536 bf16.
    FIX: `_resident_itemsize`=max(itemsize,4) on the fp32-acc caps. bf16 now LOOPS correctly.
    Byte-identical. Probe dtype_resident_probe.py.
  - Both findings: corpus byte-identical (the holes are off-corpus constructible kernels). open_RED→0.

## GATE-T ROUNDS (independent dry-verdict owner; binding)
- **Round 1 (SHA 4735ac7d): NOT-DRY.** 1 DRY sweep (boundary+saturation), 2 holes: (a) multi-rolled
  reduction_loops SLOT-misplacement (RED), (b) joint multi-tunable-grid-axis occupancy (missing-axis).
  Both FIXED byte-identically (SHA e9440ae4): reduction_loops by-slot + geometric occupancy split.
- **Round 2 (SHA d9eeca73): NOT-DRY.** 2 DRY sweeps (novelty + boundary/saturation), 1 hole: GRO
  CROSS-LOOP BLEED. FIXED byte-identically (SHA 479df334): grid_collapse_block_ids loop-scoping.
- **Round 3 (SHA 479df334): ✅ DRY** — 3 CONSECUTIVE strategy-divergent clean sweeps (30 fresh kernels),
  0 RED / 0 UNREACHED / 0 missing-axis. **DoD #1 TOTALITY certified. clean_divergent_sweeps=3.**
  **🏆 DoD CONJUNCTION MET at 479df334** (all 5 conjuncts hold). Milestone banked; run continues (§6.0).
- PATTERN: every Gate-T hole lives in the MULTI-LOOP/MULTI-FACT regime newly opened by the len==1
  relaxation (P3). Each is a genuine independent bug, all fixed BYTE-IDENTICAL on the corpus. The
  single-loop single-fact space (447-cell corpus + boundary/saturation/dtype) is DRY across all sweeps.

## ✅ P3 HOLE CLOSED (open_RED back to 0)
`two_tensor_two_loop` (2 independent reductions → 2 facts) no-fired at the `len(reduction_facts)==1`
fence → bad default. FIXED (SHA cf065557, Gate-H BROADEN): gate relaxed to `>=1`, seed the DOMINANT
(max-extent) fact via `_reduction_primary_fact`. Corpus BYTE-IDENTICAL (447/447); newly-fired kernel
acc-correct + clears the model-well floor (seed/default 0.876 SEED-WINS at (4096,2048)+(2048,4096),
1.016 tie at the larger). First generator-found open_RED, now closed. Permanent probes:
`len1_gate_probe.py`, `len1_reach_probe.py`.

## REFACTOR-CRITIC VERDICTS (workflow wdcjsdufu, saved refactor_critic_1.json)
- **P1 DONE** (SHA d9b5d95a): deleted 2 orphaned dead constants (byte-identical, smell #6).
- **P3 DONE** (SHA cf065557): the len==1 under-firing fence relaxed (above).
- **P2 DEFERRED (Gate-R-gated):** unify the m-collapse inner-rblock (`_m_collapse_inner_rblock`).
  User-tiled side byte-identical; STANDARD side NOT (would gain `min(m_collapse_block,..)` + the
  body_live<=1 slab) → perf-coupled, needs Gate R on rms/ln/instance/group bwd. L3 closes EITHER by
  the merge OR by documenting the legit per-track block_id difference. → BROADEN queue.
- **L3 ruling: MET IN SUBSTANCE.** `_resident_tile_cap` is ONE function with 3 arg-distinguished
  call sites (not duplicated arithmetic); `_carried_tile_budget` unified. The `size_reduction_tiles`
  RENAME alone = CHURN (REJECTED — R1). Pulling red_values into the builder = re-couples tracks = MORE
  complex (REJECTED — R2). The one real L3 gap is P2 (deferred).
- **L5 ruling: flat ReductionFact form is FAITHFUL** for every constructible+admitted kernel (the
  first-class per-reduction rewrite moves no counter = CHURN, REJECTED — R4). The co-residency-bit
  gap (R3) is unreachable + conservative-safe → BROADEN queue with a pre-registered fused-two-
  accumulator divergence test before bf16 work.

## CURRENT PHASE

## CURRENT PHASE
**BOOTSTRAP** (§5.0 phase 1): Step 0 (unified recorder) ✅ DONE + validated. Baseline ✅ captured
(447 cells, 0 err, 0 no-fire). RED probes ✅ written + run.

### 🔑 KEY BOOTSTRAP FINDING (shapes the whole refactor)
**All 5 output-hole probes (P1-P5) are GREEN at fc1dbaa0** (no crash / no floor-1 / no no-fire).
This CONFIRMS the task thesis: the Rounds 4-5 fixes (Issues 7/8, in fc1dbaa0) already closed the
floor-to-1 / no-fire holes, so **the gradient is STRUCTURAL hackiness (the Defects/smells), NOT
output holes**. The generator/Gate-T checker is therefore mainly a *totality-regression* guard
(don't OPEN a new hole during the refactor); the smell-critic + the 3 Defects drive the work.

**Defect 1 WITNESSED DIRECTLY** (`probes/defect1_divergence.py` + the rms_norm_bwd instrument):
`rms_norm_bwd` (8192,4096) has a MATERIALIZED primary (`in_reduction_loops=False`),
`per_feature_accumulator=True`, so the contingent branch assigns `r_block_resident=1` while the
REAL full-width resident is `next_pow2(4096)=4096`. **The PROXY (r_block_resident) DIVERGES from
the REAL property (resident reduction width) by 4096×.** This is Defect 1, and it is MASKED by
Defect 2: `per_feature_accumulator` overwrites the block_sizes vector afterward, so the lie never
reaches the emitted config FOR THESE kernels. Remove the bolt-on → the lie surfaces → must re-key
`r_block_resident` onto ACCESS, not just delete (the #2 misexecution warned about in §3b).

This is the divergence kernel the task asked me to author (self-divergence-test, [[feedback_self_divergence_test_before_predefense]]).
A full-slice reduction that is the SOLE reduction always ROLLS (so `in_rl=True`, truthful); the lie
appears only when it MATERIALIZES — which is exactly the per_feature_accumulator family + any
kernel where the roller's one-roll-slot is taken by another reduction.

Next: capture frozen-champion PERF baseline strategy (lazy per-changed-cell), then build
`size_reduction_tiles` re-keying Defects 1/2/3 (behavior-preserving sub-steps w/ ZERO-diff checkpoints).

## FROZEN-CHAMPION ANCHOR
- **Config anchor:** `_lab/unify/baseline_fc1dbaa0_configs.json` (447 cells; the BEFORE for every
  Gate-R diff). Recorder: `_lab/harness/unified_config_recorder.py` (`--diff BEFORE AFTER`).
- **Perf anchor — DECISION (logged for review):** the frozen-champion *latency* anchor is the
  BEFORE-config replayed IN-PROCESS against the AFTER-config, per changed cell, on the same
  tensors (median-of-9, cold-L2, forward-only, dtype-asserted). RATIONALE: (1) byte-identical
  cells are perf-invariant (deterministic codegen) so they need NO bench — the §3a.3/§5e skip;
  (2) a separately-pinned cross-process latency TABLE suffers ~5-10% cross-process do_bench
  jitter (footgun #4 / [[reduction_heuristic_bench_noise]]) that would inflate the 10% headroom,
  itself a cheat — the IN-PROCESS seed_AFTER/seed_BEFORE ratio (both timed identically, same
  process, same tensors) is the footgun-correct 10%-bar. So the anchor is the CONFIG set (pinned)
  + a per-changed-cell replay A/B (`_lab/unify/replay_bench.py`). The full perf sweep at the DoD
  Gate-R is the union of all changed cells, each replayed this way. This is the §7 comparison #1.
- Baseline firing matrix (all FIRE, 0 no-fire): curriculum {rms_norm,layer_norm,sum,long_sum,
  cross_entropy = T1_rolled standard; softmax,kl_div,jsd,welford = T2_usertiled}; vLLM
  {silu=pointwise; dynamic_per_token,rms_norm_dynamic,rms_norm_per_block = T2; per_token_group =
  materialized standard}; mreduction {bias_grad,dyt = T2; group/instance/rms/layer_norm_bwd =
  materialized standard}; transfer {add_rmsnorm,add_layernorm,gated_rmsnorm,scaled_softmax,
  ce_ls_zloss,dynamic_quant,fused_linear_jsd = T1; grpo = T2}.

## PER-CELL STATUS / CHANGED CELLS
(none yet — no edits)

## BANKED LEVERS (each tagged with which counter it moved; COSMETIC = moved none = anti-thrash)
(none yet)

## TRIED-AND-REJECTED
(none yet)

## BROADEN-AND-REFACTOR QUEUE
- **[P2 m-collapse inner-rblock unify]** Two divergent per-track formulas for the inner re-tile of a
  grid-origin collapse (user-tiled `min(m_collapse_block, inner_cap)` + body_live slab; standard
  `inner_byte_cap` alone). Unify into `_m_collapse_inner_rblock`. User-tiled byte-identical; standard
  perf-coupled (gains the slab + min) → Gate R on rms/ln/instance/group bwd. If regresses, document
  the legit per-track block_id difference instead (L3 closes either way). (refactor-critic P2.)
- **[feature_footprint multi-accumulator granularity]** Gate-D round2 found `feature_footprint`
  under-counts a body with MULTIPLE accumulators (kernel B with 2 accumulators gets the same inner
  byte-cap as 1-accumulator A at equal feat_bytes). NOT a fence (the cap is faithful to feat_bytes);
  a granularity gap in the `feature_footprint` populator (device_ir). Conservative direction unclear —
  assess if it under-caps (spill risk) before bf16 work. BROADEN queue.
- **[Gate-D divergence rounds for 4 post-round-2 keys]** (completeness-critic): grid_collapse_block_ids,
  _resident_itemsize(max(4,itemsize)), reduction_loops-by-slot, dominant-fact pick are Gate-T-TOTALITY-
  covered but NOT independently Gate-D divergence-tested. Counter-2 stays 0 (diff-anchored / conservative-
  direction), but author fresh Gate-D divergence kernels for each (the geometric occupancy split +
  NARROW_W1_OCC_BYTE_LIMIT owe equal-footprint tests; the dtype floor is a hardware fact, lower priority).
- **[equal-footprint tests for the 6 remaining budget constants]** (completeness-critic): only
  M_COLLAPSE_TILE_BYTES of 7 cleared Gate-D. ROW_PERSIST_MAX_BYTES (most load-bearing), LIVE_PERSIST_BUDGET,
  LOOPED_CHUNK, CARRIED_TILE_MAX_BYTES, M_COLLAPSE_MAX_CTA, MIN_WAVES owe equal-footprint divergence tests
  (faithful factor-in-budget vs curriculum fence). All validated by byte-identity + totality today.
- **[CARRIED-RESIDENT num_carried_2d_tiles>=2 x same-loop]** (completeness-critic): un-minted cross-product
  cell (every carried probe lands ==1). Conservative-direction (more tiles -> smaller chunk). Mint a probe.
- **[fp8/int8 reduction probe]** (completeness-critic): the _resident_itemsize 1->4 path has zero
  committed probes + zero anchor cells. Conservative-direction. Add a compile-time floor probe now,
  default-vs-seed when convenient.
- **[full default-vs-seed sweep over Gate-T corpus]**: model-well measured on 7 newly-fired cells (0 RED);
  a full sweep over every Gate-T-minted shape is a standing GPU follow-up.
- **[R3 co-residency bit]** ReductionFact assumes secondary reductions are sequential (not co-resident);
  a fused two-accumulator same-loop kernel could under-count footprint. Unreachable+conservative today.
  Pre-register a fused-loop divergence test before bf16/multi-dtype work. (refactor-critic R3.)
- **[M-widen occupancy/latency]** The `m_block_ids` widen path uses footprint + `_m_axis_occupancy_cap`
  but NOT a wide-reduction latency guard. On a grid-saturated wide LOOPED reduction the true resident
  width (chunk) admits M-widen=2, which REGRESSES 1.35× (grid halves, per-program work doubles). The
  current `np2(size_hint)` over-count masks this by keeping M=1. A faithful fix needs the M-widen to
  respect "don't collapse a grid that's already saturated by a wide reduction" — likely the occupancy
  cap should use the REDUCTION's effective program count, or widening should be gated on the reduction
  being small (per_token_group-like) vs wide. DEFER until the size_reduction_tiles rewrite; measure
  every firing shape (Gate R) before banking. (Found via Gate-D on lever 1.)

## DEFERRED-HARD-PILE / BORDERLINE
(none yet)

## HUMAN-REVIEW QUEUE (append-only)
2. **Asymmetric multi-tunable-grid joint occupancy floor** (Gate-D r3 KEY-3 + Gate-R + DEEPENED ANALYSIS):
   on a SYNTHETIC asymmetric 2-grid kernel the joint post-widen grid drops under the occupancy floor
   because a large-M axis's autotuner_min overrides the per-axis occupancy cap. A naive fix (occ overrides
   autotuner_min) REGRESSED the REAL 2-grid corpus kernel grpo 1.6× (Gate-R reject). **DESIGN DIRECTION
   IDENTIFIED (try-harder, past "stuck"):** the faithful distinguisher is PER-PROGRAM REDUCTION WORK. grpo
   `(B,L,V)` has R=V=64000 (HUGE reduction) → reduction-bound, so ~1024 programs is plenty (each does heavy
   work) and the wider L tile (16, via autotuner_min) is RIGHT. The synthetic dual_grid has R=8/16 (tiny) →
   occupancy-bound, so collapsing the grid hurts. FIX DIRECTION: the occupancy floor (MIN_WAVES) should
   RELAX for large-R reductions (a huge reduction hides latency via work, not wave count) — then dual_grid
   (tiny R) gets occ-capped while grpo (huge R) stays wide. This is PERF-COUPLED + needs a GPU A/B sweeping
   the flip axis (R extent × grid asymmetry, per [[feedback_ab_sweep_flip_axis]]), NOT a byte-identical
   edit. Deferred to a GPU-bearing session with this design. {where: _m_axis_occupancy_cap MIN_WAVES floor;
   make it a function of fact.size_hint (reduction extent)}
1. **M-widen on wide grid-saturated looped reductions** — the heuristic's M-axis widen path can
   over-widen a wide looped reduction (footprint-admits, occupancy-cap-admits, but latency rejects):
   true-resident-width refinement regressed cross_entropy 4096×65536 by 1.35×. Provisional decision:
   KEEP the np2(size_hint) over-count (byte-identical, conservative-safe, accidentally-optimal), log
   the latent gap. To reverse: fix the M-widen occupancy/latency model (BROADEN queue), re-bench every
   firing shape. {where: triton.py _build_block_sizes m_block_ids branch + _m_axis_occupancy_cap}

## POST-DoD GATE-D ROUND 3 (the 4 keys introduced after the last Gate-D round)
- grid_collapse_block_ids: **PASS** (proxy=all m_block_ids vs property=collapse-origin-only DISAGREE; no misclassify).
- _resident_itemsize=max(4,itemsize): **PASS** (fp32-float family; int8/16-sum int64 8B under-count is out-of-scope non-corpus edge — logged, no guard needed).
- geometric occupancy split: **acknowledged-heuristic** (faithful CAP, no fence; kept in counter-1). The
  asymmetric-grid joint-floor residual is DIABOLICAL-only — a fix regressed grpo 1.6× (Gate-R REJECT).
- dominant-fact pick (max-extent): **acknowledged-heuristic** (config_spec labels it so; NOT load-bearing
  for correctness — each reduction_loops slot sized by its own extent; only the shared num_warps + track).
- Net: divergence-debt stays 0 (2 PASS + 2 acknowledged-heuristic in counter-1, 0 REFUTE). DoD intact.

## BUDGET-CONSTANT GATE-D (the counter-1 residual cluster fully audited; ledger_gateD_constants.json)
Equal-footprint divergence test on the 6 remaining budget constants: **6/6 PASS, 0 FENCES.** All are
factor-in-budget (numerator/multiplier of a byte/occupancy budget, never a fact-vs-literal fence), bite
identically across dtype, hardware-grounded. FAITHFUL: ROW_PERSIST_MAX_BYTES (240KiB≈H100 SMEM/regfile;
killer test E=98304 loops all 3 dtypes where a raw-itemsize fence would split), MIN_WAVES (occupancy
wave-count). ACKNOWLEDGED-HEURISTIC (legible hardware estimates, faithful-enough): LIVE_PERSIST_BUDGET,
LOOPED_CHUNK, CARRIED_TILE_MAX_BYTES, M_COLLAPSE_MAX_CTA. **Counter-1's residual budget-constant cluster
is now confirmed NOT a fence.** The only remaining counter-1 item is the max-extent dominant-reduction
tie-break (Gate-D r3 acknowledged-heuristic, config_spec labels it so). Counter 1 = 2 is genuinely the floor.

## levers_since_refactor: 5  (FIRE refactor-critic NOW — at K≈4-5; Defects 1/2/3 + carried-tile done)
- Banked: lever1 Defect-1 (9f2d9bbf), lever2 Defect-2 rekey (2bf66355), lever3 Defect-2 fold (27506dea),
  lever4 Defect-3 rename (a3efb619), lever5 carried-tile budget collapse. All byte-identical, non-cosmetic.
- STRUCTURE: L1 ✅ L2 ✅ L4 ✅. L3 (size_reduction_tiles sole caller / cap-stacks collapsed): carried-tile
  done; `_resident_tile_cap` is already the single residency budget (3 call sites are one-function calls,
  not duplicated arithmetic). L5 (multi-reduction ReductionFact): pending — assess if needed.

## NEXT ACTION (post-DoD climb — never-stop; FRESH-CONTEXT RESUME POINT)
DoD banked @479df334 (heuristic SHA; HEAD is _lab commits on top, helion/ tree identical). The faithfulness
audit is now EXHAUSTIVE: every key (Gate-D r2+r3) AND every budget constant (Gate-D constants) tested,
0 fences. Completeness-critic's 2 blockers fixed (incl. a real model-well GPU sweep, 0 RED/7 cells).
Remaining Priority-2 work (NONE blocks the DoD — all conservative-direction or diabolical-only):
1. **int8/int16-sum int64-accumulator under-count** in _resident_itemsize: LOGGED do-not-chase (out-of-scope
   per dtype-perf memory; needs an accumulator-dtype fact field — speculative). [ledger gate=decision]
2. **M-widen wide-reduction latency** (human-review #1) + **asymmetric-grid occupancy** (human-review #2):
   both "investigated, naive fix REGRESSES a real kernel" (grpo 1.6×) — need a deeper per-axis occupancy/
   latency model that distinguishes grpo's wide-tile-is-better from the synthetic collapse. NOT byte-trivial.
3. periodically re-fire Gate T (open space keeps minting green = more failed-falsification evidence). Last
   Gate T DRY @479df334; re-running needs a fresh SHA-pinned independent sweep.
RECOMMENDED: a proactive context recycle here (this is a banked checkpoint, no work outstanding). A fresh
context reading this NOTEBOOK + REPORT.md + ledger.json resumes losslessly + un-biased. The honest state:
the unification is DONE and exhaustively gated; the remaining queue is genuine-but-hard perf-coupled
follow-ups that each need a deeper model (not a quick byte-identical edit), or are out-of-scope edges.
