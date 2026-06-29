# Reduction heuristic redesign — WORKLOG

> Gated log of work on the two-stage reduction-seed redesign. Every entry is a VERIFIED fact
> (measured / read from code / config-diffed), anchored on an immutable SHA where possible.
> Design lives in `/home/dev/local/prompts-lab/reduction-generality/PROMPT.md` (read §0 first).
> This file is the respawn-reconstruct artifact: baton + ledger + notebook in one.

## Fixed coordinates (verified)
- **Worktree:** `/home/dev/local/helion-redesign`, branch `reduction-redesign`.
- **Base SHA:** `de2c545f` = pristine `fc1dbaa0` `helion/` (`git diff fc1dbaa0 -- helion/` EMPTY, verified) + `_lab/` infra.
- **Interpreter:** `/home/dev/helion/.venv/bin/python`. Run scripts from `cwd=/tmp` with
  `PYTHONPATH=/home/dev/local/helion-redesign`; assert `helion.__file__` is under the worktree at top of every script.
- **Probe kernels:** `/home/dev/local/prompts-lab/reduction-generality/kernels/<slug>/kernel.py` (16 dirs: p1-p11, oos1, oos2, defect1, defect2, defect3). 13 stress = p1-p11 + oos1 + oos2.
- **Baseline configs:** `_lab/unify/baseline_fc1dbaa0_configs.json` (447 cells, frozen).
- **Config recorder:** `_lab/harness/unified_config_recorder.py` (the byte-identical / changed-cell oracle).
- **Heuristic code:** `helion/_compiler/autotuner_heuristics/triton.py` (reduction classes @ 1107–~1418).
  Stage-1 fact-building: `helion/_compiler/device_ir.py`. Fact structs: `helion/autotuner/config_spec.py:88+`.

## Hard rules (from PROMPT §0 + CLAUDE.md + memory)
- Commit after each green step (immutable SHA). config-diff EVERY edit. Never `pip install` / `git push` / `git commit` unless asked... NOTE: PROMPT §0 says "commit after each green step" — that IS the standing instruction for this task, so commits to `reduction-redesign` are authorized.
- GPU foreground-serial, NEVER detached/backgrounded (memory: detached GPU job → silent 13h stall + no notify).
- After each PHASE: run `test_reductions.py` + `test_autotuner_heuristics.py` + matmul-epilogue tests in `test_examples.py` green.
- Per probe kernel acceptance = BOTH (1) fired-right-path AND (2) perf ≥ default.
- Don't break `TritonMatmulReductionEpilogueHeuristic` (§2.9): keep its single-full-extent-reduction guard; don't leak the `>=1` relaxation into it.

## Phase status
- [x] P0 — runway verify + RED baseline of 13 probes (DONE @ 2c38a674)
- [x] P1 — Stage-1 categorizing fact-builder (DONE; GATE met: ZERO-DIFF + 460/460 fact-validation)
- [ ] P2 — Stage-2 cap-set + greedy allocator (GATE: within 10% of champion)
- [ ] P3 — delete special cases, ordered (Defect-1 re-key BEFORE deleting per_feature_accumulator)
- [ ] P4 — probes GREEN + two-check verify

---

## Stage-1 IR ground truth (answer key for the categorization; `_lab/redesign/ir_*.json`, introspector `ir_introspect.py`)

Category falls out of `(block_size_source type, in grid?, cdiv==1?)`:
- **FULL_SLICE** = `ReductionLoopBlockSizeSource` (rolled) OR materialized (in neither bs nor rl), NOT grid. (rms_norm bid1, cross_entropy, sum, long_sum, layer_norm, oos2; rms_norm_bwd/instance_norm_bwd feature bid)
- **FULL_GRID** = `FixedBlockSizeSource` with block==extent (cdiv==1), in grid. (per_token_group bid2 sh128; rms_norm_per_block bid3 sh128; p3 bid2; p8 bid2)
- **GRID_TILE (partial)** = in grid, cdiv>1 (`LoopSpec`/`Fixed` but block<extent). (jsd V-axis bid1 in g1/g3; p1 bid0/bid1; p10 bid0)
- **USER_TILE** = `LoopSpecBlockSizeSource`, in block_sizes, NOT grid. (welford bid1, softmax, kl_div, jsd bid0, dynamic_per_token, bias_grad/dyt bid, rms_norm_bwd inner-M bid1)

**`rollable` = sole-rdim-in-graph** (§3): `len(distinct rdims in this reduction's graph)==1`. Confirmed: rms_norm_bwd g0 has {bid1,bid2} → both roll=False; their sequential copies in g2 → roll=True. p1 g0 {bid0,bid1} → both False (the outer-product, blocked).

**CO-RESIDENCY = two distinct reductions share a graph_id.** Clean on probes (p7 `{g0,g2}` vs `{g1,g3}` = 2 seq groups; p2 feature+rowaccum share g0; rms_norm_bwd feature bid2 + inner-M bid1 share g0 = THE per_feature_accumulator group).

### ⚠️ DESIGN DECISION 1 — descriptor grain + rolling-invariant co-residency (the §2.2 crux, met early)
**Problem:** `device_ir.graphs` includes ROLLED SUBGRAPH copies (ids ≥ num_original_graphs), so raw `graph_id` over all graphs would (a) treat a rolled subgraph as a phantom separate graph and (b) the SAME block_id reduces in multiple graphs (rms_norm bid1 g0+g1; jsd V in g1+g3 = genuinely 2 sequential passes per §2.7).
**Resolved grain (matches §2.7 jsd "4 reduction ops across 3 graphs"):**
- A **reduction descriptor = one (graph_id, block_id) reduction OCCURRENCE** on the ORIGINAL (pre-roll) graphs — NOT one-per-block_id (loses jsd's 2 sequential V-passes) and NOT raw over all graphs (rolled-subgraph phantoms).
- **Co-residency group = one ORIGINAL graph_id.** Members = the distinct reduction axes occurring in that graph. Sizing reads the group; a block_id appearing in 2 groups (rms_norm reduce g0 + apply-reread g1) is sized once per group but to the same extent (consistent) — and the reread is already its own field.
- **Rolling-invariance:** co-residency computed on graphs with id < num_original_graphs (rolled subgraphs excluded), so it does NOT move when the autotuner flips a `reduction_loops` knob. `rollable` is the SEPARATE downstream field (§2.2). NEEDS: device_ir must expose num_original_graphs (currently a local in register_rollable_reductions) — add it as an attribute in P1.
- VERIFIED (graph-type dump): rolled subgraphs are EXACTLY `ReductionLoopGraphInfo`; originals are Root/ForLoop/If/Else GraphInfo. So co-residency = group reduction occurrences by graph_id over `[g for g in graphs if not isinstance(g, ReductionLoopGraphInfo)]`. No new `num_original_graphs` state needed.
  - p4: g0{1},g1{1} → 2 seq descriptors over same bid (the RED "1 fact" fix). p7: g0{1},g1{3} → 2 seq. p6: g0{0,1}co-resident + g1{1} seq. jsd: g1{1},g3{1} (2 seq V), g5{0} → 3 groups (matches §2.7). rms_norm_bwd: g0{1 USER_TILE,2 FULL_SLICE} co-resident + g2{2} seq → feature+innerM group = the per_feature_accumulator group.

### P1 IMPLEMENTATION APPROACH (chosen)
Build `ReductionKernelFact` (descriptors + graph_id groups) as the NEW Stage-1 product, then DERIVE the
legacy `ReductionFact` list from it and route the live heuristic through the derivation. **GATE: zero config
diff** proves the descriptors carry everything (the reproducibility map, end-to-end — exactly §5.4 P1's
"re-point without changing outputs"). De-risk by an offline field-equality validation (derived vs actual
legacy fact) across 447+13 BEFORE routing live. The relaxed `>=1` gate / multi-group sizing is BEHAVIOR
change → deferred to P2/P3; P1 preserves current behavior (incl. p7 still declining).

## Log

### 2026-06-29 — P1 DONE (Stage-1 categorizing fact-builder)
**What landed (helion/):**
- `config_spec.py`: `ReductionCategory` enum (FULL_SLICE/FULL_GRID/GRID_TILE/USER_TILE/DECLINED) +
  `SIZED_REDUCTION_CATEGORIES` + `FULL_EXTENT_CATEGORIES` constants; `ReductionDescriptor` (per-occurrence),
  `CoResidencyGroup`, `ReductionKernelFact` structs (§2.6); `spec.reduction_kernel_fact` slot.
- `device_ir.py`: `build_reduction_kernel_fact` (phase 3b, runs after build_reduction_facts) +
  `_original_graph_reductions` (excludes `ReductionLoopGraphInfo` rolled subgraphs → rolling-invariant
  graph_id), `_categorize_reduction`, `_per_reduction_memory_fields` (num_load graph-scoped),
  `_materialized_feature_footprint`. Built ALONGSIDE legacy reduction_facts.
- `triton.py`: re-pointed `_is_standard_reduction` to key on the new category (FULL_SLICE/FULL_GRID=standard,
  USER_TILE=user-tiled), with legacy-proxy fallback. Removes the `primary ∉ block_sizes` proxy.
**Validation (`_lab/redesign/validate_kernel_fact.py`):** 460/460 PASS (447 corpus + 13 probes) on 4 invariants:
INV1 category re-derivation, INV2 groups partition by graph_id, INV3 legacy ReductionFact reconstructible
(primary sized + size_hint/itemsize/ils/row_reread match + full_width reconstructs from group∪normalize-loops
+ secondaries represented), INV4 two-rollable-FULL_SLICE never co-resident.
**GATES MET:** config-recorder ZERO-DIFF 447/447 (after build, after re-point, after format). Track-discriminator
new-vs-legacy 452/452 match. Tests: test_autotuner_heuristics 24p/22s, test_reductions 28p, test_examples
-k matmul_layernorm 2p/2s. Ruff clean, format applied. Pyrefly: only 2 PRE-EXISTING errors (lines 125, 4094 —
not my code), 0 new. **Commit: P1 (next).**

**Two faithfulness wins surfaced (NOT bugs — record for P2):**
1. `full_width_output` is now PER-DESCRIPTOR (§2.6). Legacy was a kernel-scalar OR over {rdim}∪normalize-loops;
   welford's full-width store is on the NORMALIZE loop (bid2), not the combine reduction (bid1). New desc=False
   for bid1 is MORE faithful. **P2 allocator** must OR full_width over {group reductions + normalize loops} to
   reproduce the legacy kernel-scalar cap input.
2. Legacy `secondary_reduction_block_ids` for p2/p6 includes the GRID_TILE cross-row accum (bid0) — the messy
   "rediscover at sizing time" the redesign kills. New fact correctly types it GRID_TILE (a real reduction, not
   a sized one). The per_feature_accumulator group (p2: FULL_SLICE bid1 + GRID_TILE bid0 co-resident in g0) is
   exactly §6 Q4's "two co-resident categories, no recognizer."

**p1/p5 status:** still fire `triton_pointwise` (their reductions are GRID_TILE → not sized → no legacy fact →
pointwise). The new fact DOES capture them (p1: 2 co-resident GRID_TILE; p5: 2 GRID_TILE). Turning these into
sized reductions is a BEHAVIOR change = P2/P3 (allocator + relaxed gate), not P1.

### 2026-06-29 — P0 DONE
- Verified HEAD `de2c545f`; `git diff fc1dbaa0 -- helion/` empty (pristine base confirmed).
- Read PROMPT.md fully (§0–§7). Read all heuristic code: `triton.py` reduction classes (Pointwise@304,
  reduction base@454, Standard@1107, UserTiled@1284, MatmulEpilogue@1421), `config_spec.py` ReductionFact@88,
  `device_ir.py` fact-builders (ReductionRole@99, classify@1098, register_unrolled@1132, build_reduction_facts@1313,
  assemble@1542, phase orchestration@3459-3515). Registry: `autotuner_heuristics/__init__.py` (order: matmul-epi,
  splitjoin, standard, usertiled, pointwise).
- **GATE INSTRUMENT VALIDATED:** config recorder reproduces frozen baseline **ZERO-DIFF, 447/447 byte-identical**
  on pristine base. (`_lab/redesign/p0_repro.json` vs `baseline_fc1dbaa0_configs.json`.)
- **Probe corpus = 13 stress kernels** = p1-p11 + oos1 + oos2. (`defect1/2/3` dirs are TODO.md placeholders, NOT
  kernels — manifest §2.8 confirms 13 = p1-p11+oos1+oos2.)
- FIXED portability: oos1 hardcoded `/home/dev/local/helion-unify/examples` → now derives examples/ from active
  `helion.__file__`. (per portable-lab-state rule.)
- RED baseline recorded: `_lab/redesign/probe_red_baseline.json` (recorder `_lab/redesign/probe_recorder.py`,
  reusable as GREEN recorder later). All 13 bind without crashing. Diagnoses:

| probe | fired today | n_red | RED symptom (what the redesign must fix) |
|---|---|---|---|
| p1 outer-product-coresident | **triton_pointwise** | 0 | co-resident 2 FULL_SLICE over grid-tile axes → both FLOORED_ROW → no red fact → **mis-seen as pointwise** |
| p2 feature+rowaccum (Defect-2 witness) | triton_reduction_tile | 1 | fires standard via *secondary* path, `pfa=False`. graph_ids `{1:[0,1]}` (co-resident). bs=[2048] |
| p3 full-grid-nonquant | triton_reduction_tile | 1 | prim=2 (group axis) sh=128 FULL_GRID, m_block=[0,1]. bs=[32] — grid sibling widened (looks OK-ish) |
| p4 two-rollable-sequential | triton_reduction_tile | 1 | TWO rollable reductions but **only 1 fact** (graph_ids `{1:[0,1,2,3]}` — both rdims rolled to 1 fact?); bs=[8,8] |
| p5 3d-reduction-tile | triton_pointwise | 0 | reduced inner axes not seen as reduction → **pointwise** bs=[1,32,64] |
| p6 mixed-coresident+seq | triton_reduction_tile | 1 | only 1 fact for the mixed shape; bs=[2048,8] |
| **p7 gridtile-then-usertile** | **[] NONE** | **2** | **`==1` gate DECLINES a real 2-reduction kernel → falls to default.** graph_ids `{1:[0,2],3:[1,3]}` (2 seq groups). Headline witness for `>=1` relaxation. |
| p8 fullgrid+usertile | triton_reduction_tile | 1 | prim=3 (K) FULL_SLICE, group FULL_GRID dropped. bs=[1] rl=[128] |
| p9 nonred-loop-then-fullextent | triton_reduction_tile | 1 | bs=[1,4096] rl=[None] (nrl handling) |
| p10 usertile+gridtile | triton_reduction_tile | 1 | only 1 fact (one declined); bs=[4096] |
| p11 fullextent-then-nonred-loop | triton_reduction_tile | 1 | bs=[1,4096] rl=[None] |
| oos1 jagged | [] NONE | 0 | correctly DECLINED (✓ expected) |
| oos2 strided-dim0 | triton_reduction_tile | 1 | fires (the known cliff, left as-is); bs=[4] w=16 |

- **KEY RED themes:** (a) the `==1` gate (p7 declined, p4/p6/p10 collapse multi→1 fact); (b) co-resident reductions
  over grid-tile axes mis-classified FLOORED_ROW→dropped (p1) or kernel seen pointwise (p1/p5); (c) `pfa` recognizer
  absent off the 4 norm-bwds (p2). The redesign's first-class N-reduction fact + graph_id co-residency + positive
  taxonomy must turn these GREEN.
- **NOTE on p4:** two rollable reductions over different axes landed in ONE fact (graph_ids all 4 graphs). Per §2.7
  invariant "two rollable reductions are NEVER co-resident" → they should be 2 sequential facts. Investigate in P1
  (may be the `_rollable_reduction_records` stashing one rdim, or both rolled into one). Flag, don't block P0.
