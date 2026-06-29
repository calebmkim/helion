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
- [x] P1 — Stage-1 categorizing fact-builder (DONE @ 446e6e9d; GATE: ZERO-DIFF + 460/460 fact-validation)
- [x] P2 — cap fabric + kernel-fact-driven sizing (DONE @ 1877364e/6309d726/3060accf; all ZERO-DIFF).
- [x] P3 — delete special cases (DONE @ b4c92bbb/28491fc2/ff7927ed):
      (a) relaxed gate ==1→>=1 (p7 fires); (b) standard-track collapse keys on _grad_collapse_group taxonomy
      (Defect #1/#2/#6 standard copy gone); 2 square-shape movers re-benched 1.9x/2.7x FASTER; M_COLLAPSE
      constant KEPT (32768, faithful distinct cap); (c) classifier unified, ReductionRole→view (Defect #3).
      DEFECT STATUS: #1/#2/#6 standard ✓gone; #3 ✓unified; user-tiled pfa KEPT (faithful accumulator-shape
      property reading AccumulatorFact provenance, fires on exactly the 6 norm-bwds — NOT a recognizer, the
      Defect-2 complaint was the override+subtractive-filter, both deleted). CUMULATIVE vs frozen baseline:
      only 2 cells moved (both faster), 445 byte-identical.
- [ ] P4 — probes GREEN + two-check verify (Tier-1 fired-right-path + Tier-2 perf ≥ default). Includes the
      p2/p6 probe movers ([1]/[8,8]) perf-check + p7 GREEN + adversarial taxonomy sweep.
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

### ⚠️ DESIGN NUANCE — per_feature_accumulator is TWO shapes, not one (refines §6 Q4)
The §6 Q4 thesis ("pfa = FULL_SLICE co-resident with partial GRID_TILE") describes ONLY the norm-bwd
STANDARD-track family (rms/layer/instance/group_bwd): a `.mean(-1)`/`.sum(-1)` FULL_SLICE feature reduction
co-resident (same graph) with an inner-M grad-accum reduction. `_grad_collapse_group` catches these.
BUT bias_grad/dyt (USER-TILED track) are DIFFERENT: a SINGLE inner-M reduction (`sum(grad_out[mb,:],dim=0)`)
that accumulates into a per-feature `gb[n]` buffer — NO feature reduction co-resident. Source: outer grid
tile `mb_cta` (the collapse CTA) + inner re-tile `mb`; the reduction is over `mb`, result → per-feature accum.
**The unifying FAITHFUL property = "a reduction whose result accumulates into a buffer spanning the full
materialized feature axis" (the grad-param)** — which IS what `per_feature_accumulator` actually computes
(accumulator dim_block_ids == materialized feature axes), reading accumulator provenance, NOT kernel identity.
**DECISION:** `per_feature_accumulator` is ALREADY faithful (an accumulator-shape property, not a recognizer);
the Defect-2 complaint was the OVERRIDE BRANCH (recomputing block_sizes) + the subtractive `inner_tile_ids`
filter, not the signal. So: (1) STANDARD track norm-bwd → key the collapse on `_grad_collapse_group` (taxonomy,
co-residency) — DONE. (2) USER-TILED bias_grad/dyt → keep keying on `per_feature_accumulator` (the faithful
accumulator-shape property) but SOURCE the inner ids positively. Both deletions remove the SUBTRACTIVE filter
(Defect #2); the accumulator-shape signal stays. Re-bench the norm-bwd movers.

### ⚠️ p2 witness is STRUCTURALLY SIMPLER than rms_norm_bwd (refines the Defect-2 witness claim)
p2 (`grad_scale[:] += (...).sum(0)` over `tile_m` directly) has NO inner re-tile loop — the cross-row accum
reduces the GRID axis (bid0 GRID_TILE) directly. rms_norm_bwd/bias_grad have an INNER re-tile (`mb` within
`mb_cta`). So `_grad_collapse_group` (which looks for a non-grid inner re-tile co-resident with a full-extent
feature reduction) returns None for p2 — correctly, p2 isn't the same collapse shape. p2 = "FULL_SLICE feature
reduction (bid1) + a reduction OVER the grid-M (bid0)", co-resident. The faithful sizing: the GRID_TILE claims
~nothing tunable, the grid-M sibling takes the remainder (occupancy-widen) — the §2.3 greedy. Whether p2's
current `bs=[1]` is right is a P4 Tier-2 perf-check (it may need grid-M widening). DEFER p2 correctness to P4;
pfa fires on exactly the 6 real norm-bwds (no false positive on p2 — that's the recognizer's blindness the
taxonomy must cover, but p2's shape differs from the norm-bwd collapse so it's a separate sizing path).

### 2026-06-29 — P3b @28491fc2: standard grad-collapse keys on _grad_collapse_group (taxonomy)
Deleted the `fact.per_feature_accumulator` gate + subtractive `inner_tile_ids` filter on the STANDARD track;
keys on `_grad_collapse_group` (co-residency taxonomy), sources inner ids positively. 14/16 norm-bwd cells
byte-identical; 2 movers = square-shape (4096×8192) rms/layer_norm_bwd `[1,1]→[32,1]`. **Re-benched H100:
rms_norm_bwd 183.5→96.2µs (1.9x FASTER), layer_norm_bwd 290.4→106.7µs (2.7x FASTER), outputs match** — a
legacy STARVATION-BUG fix, far within the no-regression bar. Tests green.
REMAINING P3: (c) user-tiled bias_grad/dyt — keep faithful pfa accumulator-shape signal, drop subtractive
filter; (d) ReductionRole stored decisions; (e) probe perf-checks (p2/p6 movers) → P4.

### 2026-06-29 — P3a: relaxed gate (==1 → >=1 sized reductions), corpus ZERO-DIFF
- `_triton_reduction_eligible` now fires when the kernel fact has >=1 SIZED reduction + no matmul (was
  `len(reduction_facts)==1`). Corpus-safe: all 452 corpus cells have exactly 1 fact → ZERO-DIFF 447/447.
- `_primary_fact(env)` selects the dominant fact (max backed size_hint) for multi-fact kernels; = facts[0]
  for single-fact (byte-identical). Standard track `is_eligible`/`get_seed_config` route through it.
- Standard track emits ONE `reduction_loops` entry PER spec (multi rolled reduction): each sized against its
  own extent (sequential groups). Single-spec unchanged.
- **p7 RED→GREEN (fired-right-path):** was declined (`fired=[]`), now `triton_reduction_tile rl=[None,None]
  bs=[8,8]` — two sequential rolled FULL_SLICE reductions, each persistent. The headline relaxed-gate win.
- ⚠️ PROBE MOVERS to perf-check in P4 (NOT corpus, so zero-diff gate intact): p2 `[2048]→[1]`, p6
  `[2048,8]→[8,8]`. CAUSE = P2b's `_secondary_red_values` correctly EXCLUDES the GRID_TILE cross-row accum
  (bid0) from reduction-sizing (legacy wrongly sized it as a secondary). Faithful, but bid0 now floors → need
  perf check that [1]/[8,8] ≥ default for these off-corpus witnesses (P4 Tier-2). p4 unchanged [8,8].

### ⚠️ SQUARE-SHAPE legacy quirk (rms/layer_norm_bwd at M==N, e.g. 4096×8192) — P3 mover, re-bench
At rms_norm_bwd(4096,8192): inner-M (bid1 USER_TILE sh=8192) and feature (bid2 FULL_SLICE sh=8192) are TIED
at extent 8192. Legacy `_non_reduction_loop_candidates` then mis-claims bid1 as a NON-reduction apply loop
(its extent coincidentally == the reduction extent), so the pfa override's `inner_tile_ids` comes out EMPTY →
override no-ops → legacy emits `[1,1]` (a STARVED config: grid M=1, inner=1). The new taxonomy correctly keeps
bid1 USER_TILE (a reduction), so `_grad_collapse_group` finds inner=(1,) → the group-greedy would emit the
proper `[m_cta, inner]` (like the non-square `[64,2]`). **This is a legacy STARVATION bug the redesign FIXES**
— but it's a config CHANGE, so it's a P3 mover to re-bench (expect FASTER than `[1,1]`). Affects the 2 square
norm-bwd cells (rms_norm_bwd + layer_norm_bwd at 4096×8192). `_grad_collapse_group` vs legacy inner_tile_ids:
14/16 match, these 2 diverge (correctly).

### 2026-06-29 — P2 progress (P2a cap fabric @1877364e, P2b kernel-fact red_values @6309d726)
- P2a: `Cap`/`size_axis` primitive (§2.4); re-expressed `_reduction_rblock` (looped chunk) + `_build_block_sizes`
  M-axis widen onto it. ZERO-DIFF.
- P2b: `_secondary_red_values` reads sized descriptors from the kernel fact (GRID_TILE correctly NOT
  reduction-sized). ZERO-DIFF. Equivalence checks: primary = max-backed-size_hint (0/452 divergence);
  sized∩block_sizes = legacy red_values keys (only p2/p6 probes diverge, more-faithfully).

### ⚠️ USER REMINDER (pinned block sizes factor DOWN the budget) — record for the group budget (P3/P4)
Pinned axes (FixedBlockSizeSource, full-extent: per_token_group/rms_norm_per_block `group_size=128`) have a
FIXED size but MUST still factor down the resident budget. Status:
- ALREADY HANDLED per-axis: `_pinned_inner_resident_elems` returns the pinned product (128) and
  `_resident_tile_cap` divides the byte budget by it (verified: per_token_group bid1→128, rms_norm_per_block
  bid1/bid2→128). Pointwise seed does the analogous `target // pinned_elems`.
- TODO for the GROUP BUDGET (§2.3 group_footprint, when the greedy per-group allocation lands in P3/P4): a
  pinned/FULL_GRID axis resident in a co-residency group must appear as a MULTIPLIER in the group_footprint
  denominator (it consumes budget like any resident tile, just at a fixed extent the allocator can't tune).
  i.e. group_footprint = Σ distinct resident tensors (∏ tiled dims INCLUDING pinned dims' fixed extents). The
  FULL_GRID reduction "claims ~nothing tunable" but its pinned extent still costs footprint → don't drop it.

### 2026-06-29 — P2 START (cap-set + greedy allocator) — strategy: EQUIVALENCE-FIRST (user-chosen)
Build the cap-set + allocator over the new ReductionKernelFact to reproduce existing configs
BYTE-IDENTICALLY first (zero-diff gate); generality (relaxed >=1 gate, p1/p5/p7 firing, multi-group
sizing) lands in P3. The §2.5 M_COLLAPSE-constant shift ([16,2]→[16,1]) is the one anticipated mover —
decide + re-bench within-10% when reached.

**ANSWER KEY (recorded configs, the targets — from baseline_fc1dbaa0_configs.json):**
- T1 standard (rdim ROLLS, bs=[m_block], reduction_loops=[None]|[chunk]): rms_norm/layer_norm [4] rl[None] w4
  (sh768, persistent, m widened to 4 by occupancy); sum [8] rl[None] (sh1024, m widened 8); long_sum [1]
  rl[16384] w32 (sh65536, looped chunk); cross_entropy [1] rl[None] w32 (sh30522, persistent).
- T2 user-tiled (rdim IS block_sizes): softmax [16,128] (m16, r=full128); welford [8,1024,1024] nrl=[2]
  (m8, r=1024, apply-tile 1024); kl_div [4096,1] 2d=1 w32 (carried cap 4096); jsd [2048,1] 2d=2 (carried 2048);
  dynamic_per_token [4096,4096] nrl=[2]; rms_norm_per_block [4096,32] sec=[3] (RMS r=4096 + groups_per_row=32).
- per_feature_accumulator (norm-bwds, pfa=True): bias_grad [64,64] w16; dyt [64,2] w16; rms_norm_bwd [64,2]
  w8 (M_COLLAPSE_TILE_BYTES=32768 → inner 2); instance/group_norm [1,1] w4. ← the §2.5 M_COLLAPSE case.

**Approach:** new module `triton_reduction_alloc.py` (or methods on the base) — `size_reduction_tiles(kernel_fact,
spec, env)` returning {block_id: size} + reduction_loops + num_warps. Re-express current
`_reduction_rblock`+`_build_block_sizes`+`_m_collapse_*` onto the cap primitive. The two heuristic classes
delegate to it. Zero-diff each step.

**EQUIVALENCE FINDINGS (read-only checks, prerequisite for the zero-diff gate):**
- **Primary-selection for P2 = legacy rule = `max(size_hint over BACKED sized descriptors)`** — verified
  0/452 divergence from `lf.primary_reduction_block_id`. KEEP THIS for P2 (num_warps byte-identical).
- The §6.2 "priority-order primary" (category-tier first) DIVERGES on rms_norm_per_block (picks FULL_GRID
  sh128 over USER_TILE sh4096) — that's a sequential-group artifact. The §6.2.1 "num_warps = max ROW-BYTES
  owner" rule is the faithful resolution (user CONFIRMED: size across groups not category) and matches legacy
  EXCEPT the 8 norm-bwd kernels, where legacy num_warps is recomputed inside the per_feature_accumulator
  override (narrow-w1 dropped) AND the unbacked inner-M placeholder (8192) would mis-rank — excluding unbacked
  fixes it. **So: max-row-bytes-over-BACKED-sized = legacy primary, 0 divergence.** These refinements
  (priority-order bidding, row-bytes warps) are P3 generality swaps (re-benched), NOT P2.
- **P2/P3 split clarified:** P2 builds the cap-set fabric reproducing today's configs (legacy primary rule,
  the per_feature_accumulator branch still gated on `pfa` for now). P3 swaps proxies for faithful keys where
  they diverge (delete pfa override → greedy allocator over the co-resident group; row-bytes num_warps), each
  re-benched. This keeps P2 a pure refactor (zero-diff) and isolates every behavior change to P3.

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
