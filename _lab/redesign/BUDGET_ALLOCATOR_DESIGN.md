# Reduction seed: the ONE budget-based allocator — design + build brief

> Launch brief for a FRESH-CONTEXT agent. Self-contained. The mission is a REWRITE of the reduction
> seed's tile sizing into one principled per-co-residency-group BUDGET allocator, DELETING the
> special-cases the current code still hides. **Configs WILL change — that is expected and fine.**
> You record every changed config + its perf, hill-climb the BUDGET to recover/beat perf, and
> present the full before/after to the human at the end. Read this whole doc, then the two
> companion docs, before writing code.

## 0. ORIENTATION — what this project is
Helion's reduction SEED HEURISTIC picks a starting autotuner config for reduction kernels. A
two-stage redesign exists:
- **Stage 1** `helion/_compiler/device_ir.py` builds `ReductionKernelFact` — a list of per-occurrence
  `ReductionDescriptor`s + `CoResidencyGroup`s. Structs: `helion/autotuner/config_spec.py`.
  Taxonomy per reduction: FULL_SLICE / FULL_GRID / GRID_TILE / USER_TILE / DECLINED. Co-residency =
  same original `graph_id`. **Stage 1 is DONE and correct — do NOT change it** (unless you genuinely
  need a new faithful descriptor/group field; if so add it and justify).
- **Stage 2** `helion/_compiler/autotuner_heuristics/triton.py` is the seed heuristic you are
  rewriting: `_TritonReductionSeedBase` (~line 544) + `TritonStandardReductionHeuristic` (~1631) +
  `TritonUserTiledReductionHeuristic` (~1834). Both `get_seed_config` heads currently call
  `cls.size_reduction_tiles(env, spec, device_ir, pd)`.
- A SEPARATE class `TritonMatmulReductionEpilogueHeuristic` (~1984) is OUT OF SCOPE — DO NOT TOUCH.
  It legitimately uses the legacy `ReductionFact`. Likewise `ReductionFact` stays built (epilogue +
  eligibility gate use it); the reduction tracks already don't read it — keep it that way.

## 1. WHY THIS REWRITE (the human's repeated, emphatic feedback — internalize it)
The current `size_reduction_tiles` is "an impossibly complicated mess of ad-hoc solutions." It
reaches correct configs on the corpus via layers of `if`s + indirection that only *coincidentally*
align, NOT via budgeting. Specifically the human rejected, by name:
- the `if standard / else` split in `size_reduction_tiles` (two sizing paths);
- `footprint_factor=pd.body_live_tiles` and `pinned_resident_elems=pinned_co` applied to ONLY the
  standard track (resident-footprint terms belong to BOTH);
- `if not standard and cls._is_per_feature_accumulator(...)` — "EXACTLY what size_reduction_tiles is
  supposed to PREVENT" — a kernel-shape recognizer + the grad-collapse `inner_tile_ids` override;
- `red_values = {} if standard` — conflates sizing with emission (a rolled reduction still HAS a
  size; it just rides the `reduction_loops` knob);
- `_build_block_sizes` kept as a general-purpose multi-branch function.
ALL of that is the special-casing the whole redesign exists to delete. Do NOT port it, wrap it, or
re-smuggle it. The PRIOR attempt failed by preserving it under a "keep configs byte-identical"
constraint — **that constraint is DROPPED. It caused paralysis (window-dressing without changing
behavior).**

## 2. THE DESIGN — one budget-based allocator (this is the spec)
Replace the scattered passes (`_reduction_rblock` in isolation -> `_build_block_sizes` reading
r_block back -> loops -> grad-collapse override) with ONE allocator over the kernel fact:

**Iterate the CO-RESIDENCY GROUPS in priority order** (the group with the most/heaviest FULL-EXTENT
reductions first; PROMPT §2.3 cross-group ordering). For each group:
  1. **Form ONE budget for the whole group** — a register/byte capacity (a number). The budget MAY
     be SCALED by faithful group properties (see §3): more live tiles / more loop-carried
     accumulators -> tighter budget. This scaling is the LEGITIMATE home for `body_live_tiles` and
     the accumulator count — a continuous function of resident-footprint, applied UNIFORMLY to every
     group, NOT a per-track arg and NOT a shape gate.
  2. **Seat axes in priority order, each taking FIRST CRACK then floored by REMAINING budget**
     (the human's refinement — it need NOT be greedy-from-scratch):
       - full-extent reductions (FULL_SLICE + FULL_GRID) — a re-read full-slice raises its floor to
         the full extent (persistence);
       - then user-tile reductions;
       - then grid-tile reductions;
       - then the grid M / "row" axis takes the REMAINDER.
     Each axis computes its DESIRED size from its own (simplified) logic, then is FLOORED by what
     fits in the budget left after everything already seated. The rdim "gets first crack" but does
     NOT have to consume the whole budget (a 768-wide row leaves most of a 240KB budget for the grid
     sibling to widen into). When the budget is already spent, the next bidder FLOORS to 1.
  3. **Hold this group's assignments FIXED** as inputs to later groups (a shared grid-M tile sized
     by group A is fixed when group B sizes).
**Then `non_reduction_loop_block_ids` LAST** — a separate pass, sized against whatever budget
remains (maybe none). Welford's normalize loop; rms_norm_per_block's groups_per_row.

**THE FOOTPRINT (keep it SIMPLE — the human said per-tensor Σ∏ is overkill):**
  `group_footprint(assignment) ≈ num_live_tiles_in_group × (∏ of the assigned tile sizes in the
  group) × itemsize`  — a slight OVER-estimate is fine and safe (errs toward smaller tiles / flooring,
  never an unsafe spill). Use this; only reach for the exact per-distinct-tensor sum if this proves
  too coarse on a real kernel (and say why).

  **`num_live_tiles_in_group` — USE THIS DEFINITION** (a scout dry-run flagged it as the one
  underspecified input; resolved here): it is NOT a stored fact field, and you CANNOT derive a
  per-group accumulator count cleanly because `AccumulatorFact` carries no `graph_id` (so
  accumulators are not attributable to a co-residency group). What IS group-attributable is the
  per-descriptor `body_live_tiles` (the walker-measured peak count of simultaneously-live
  rdim-shaped tiles in that reduction's body). So define, for the group's descriptors `D`:
  `num_live_tiles_in_group = max(d.body_live_tiles for d in D` that are SIZED reductions`)`
  (max, not sum — a conservative over-estimate that bounds the heaviest body; defaults to 1).
  This is the faithful, uniform, group-attributable scaling property. If a kernel ever needs a
  per-group accumulator count, that is a Stage-1 change (add `graph_id` to `AccumulatorFact`) —
  out of scope; note it if you hit it. The `∏ of assigned tile sizes` = the product of every tile
  this group seats (each reduction's r_block + the grid M block + any grid widening); `itemsize` =
  the primary descriptor's (fp32-promoted) itemsize.

**THE FLOOR-vs-RESIDENT INSIGHT (the whole reason budgeting is the only correct approach):**
A grid-axis reduction sometimes must FLOOR (jsd — the grid parallelizes it across programs, it
can't sit resident) and sometimes must be held as a RESIDENT r_block (a vLLM kernel's specialized
`group_size`). This is NOT a category to branch on (`cdiv==1`, recognizers). It is simply *whether
its resident tile FITS the group budget after the co-resident reductions took their share*. Try to
seat it; fits -> resident; doesn't -> floors to 1 and the grid parallelizes it. BUDGETING DERIVES
THIS. The current code gets both right via a pile of `if`s that coincidentally align — that is the
thing to delete.

**BUDGET SUBSUMES the footprint caps; KEEP the intrinsic caps (avoid double-counting):**
  - DELETE / fold into the budget: `_resident_tile_cap`, `_carried_tile_r_block_cap`,
    `_carried_m_block_cap`, `_carried_grid_dims`, `_pinned_inner_resident_elems`,
    `M_COLLAPSE_TILE_BYTES` inner cap — these ARE footprint approximations; the one group budget +
    depletion replaces them all.
  - KEEP as per-axis "desired size" (intrinsic, non-footprint) limits: EXTENT (never exceed the
    axis's own size), OCCUPANCY (post-tile grid >= num_sm * MIN_WAVES — a grid-COUNT currency, not
    bytes, so separate), and the PERSISTENCE preference (does a re-read row want to be held vs
    chunked — the r_block "desire").
  So `_reduction_rblock` simplifies to "desired r_block from persistence + extent" (its internal
  byte-cap becomes the budget); the M-axis logic simplifies to "desired widen from occupancy +
  extent"; the allocator floors each by remaining budget.

**EMISSION (the ONLY legitimate standard-vs-user difference):** a reduction's computed size is
WRITTEN to `reduction_loops` if it is a rolled/standard reduction, or to a `block_sizes` slot if it
is user-tiled. That is codegen routing of WHERE the number lands — NOT a different way to COMPUTE
it. The `if standard/else` collapses to this emission routing and nothing else. Every reduction
(rolled included) gets a size from the budget; rolled ones happen to ride the reduction_loops knob.

**num_warps** stays a scalar lever OUTSIDE the budget loop, keyed on the primary's resident ROW
BYTES (`size_hint * input_load_itemsize`, PROMPT §6.2.1) — the existing `_num_warps` + selection via
`_primary_descriptor_selected` (max row-bytes). Do not fold it into the budget.

**RESOLVED DETAILS (scout dry-run):**
- **Priority order (used BOTH within a group and to order groups):** full-extent (FULL_SLICE +
  FULL_GRID) → user-tile (USER_TILE) → grid-tile (GRID_TILE), with extent (size_hint) as the
  in-tier tiebreaker (bigger first). Order the GROUPS by their heaviest member under this same key
  (the group with the most/heaviest full-extent reductions sizes the shared grid-M tile first).
- **When a seated tile is a ≥3-D / multi-buffer carried accumulator, its footprint contribution is
  ∏ of its tiled dims within a buffer, Σ across separate buffers** (the #2/#3 finding — see
  CARRIED_AND_GREEDY_FINDINGS.md). Classify each dim by MEMBERSHIP (is it an rdim? a grid axis?),
  never by position. The `num_live_tiles × ∏ tile_sizes` over-estimate already captures this at the
  group level; if you compute per-tensor footprint instead, use product-within / sum-across.

**DELETE these recognizers entirely (do not port):** `_is_per_feature_accumulator` + the
grad-collapse `inner_tile_ids` override + `_grad_collapse_group` (the norm-bwd M-collapse must FALL
OUT of counting the resident grad-param `[inner, *features]` tensor in the group footprint — a
bigger resident tensor tightens the budget and shrinks the inner tile automatically); the
`if standard/else` SIZING split (keep only emission); `_carried_*` (subsumed by footprint).

## 3. BUDGETS MAY BE SCALED — the discipline (so you don't re-smuggle special-casing)
A budget is a CAPACITY number. It may be a function of faithful, CONTINUOUS workload properties of
the group (num_live_tiles, num loop-carried accumulators). "A heavy body gets a tighter budget" =
`budget = f(num_live_tiles, num_accumulators, ...)`, applied to EVERY group identically. It is OK to
have a few budget tiers scaled this way; you may need to hill-climb them.
THE LINE: a "budget" keyed on KERNEL IDENTITY or a CATEGORY GATE (`if per_feature_accumulator`,
`if standard`) is NOT a budget — it is a recognizer in disguise. Test: *is it a continuous function
of a resident-footprint property, applied uniformly?* Yes -> budget input. A branch picking a
different formula for a recognized shape -> the special-casing being deleted. If you find yourself
wanting to hill-climb something that ISN'T a budget capacity/scale, STOP, gate it, and explain in
your log WHY it cannot be expressed as a budget before touching it.

## 4. METHOD (configs WILL change; no byte-identical gate)
1. Build the allocator per §2. First pass: use honest first-principles budget constants (you can
   seed them from today's: ROW_PERSIST_MAX_BYTES=245760, CARRIED_TILE_MAX_BYTES=16384,
   LOOPED_CHUNK=16384, MIN_WAVES=8, M_COLLAPSE_TILE_BYTES=32768 — but they are now ONE unified
   budget + scales, not scattered caps).
2. Run the config recorder (§6). RECORD every changed cell (before -> after). Expect MANY changes —
   that is the point.
3. For the changed cells, measure perf (before-config vs after-config, §6 replay_bench / probe_perf)
   and RECORD seed_after/seed_before per cell.
4. HILL-CLIMB the BUDGET CONSTANTS / SCALES ONLY to recover or beat perf on regressed cells without
   re-introducing special-casing. (Budget first. Anything else -> gate + explain per §3.) Iterate:
   adjust budget -> recorder -> re-bench the movers. GPU rules in §5.
5. Keep correctness gates GREEN throughout (validate_kernel_fact 460/460; probe_assertions Tier-1
   13/13 — these test Stage-1 facts + fired-right-path, NOT exact configs, so they should stay green;
   if a probe's fired-path assertion breaks because the taxonomy routing changed, that is a real
   signal — investigate, do not just edit the assertion).
6. Unit tests: test_reductions + test_autotuner_heuristics WILL likely need updating (they pin
   specific seed configs / construct facts). Update them to the new principled behavior; do NOT
   contort the allocator to satisfy a stale pinned config. matmul-epilogue test
   (test_examples -k matmul_layernorm) MUST stay green (you didn't touch that class).
7. At the END, present the human: (a) the full table of changed configs (cell -> before/after);
   (b) per-changed-cell perf (after/before, and after/default); (c) net perf summary (geomean,
   worst regression, best win); (d) which budget constants you landed on + why; (e) confirmation the
   recognizers are GONE (grep shows no _is_per_feature_accumulator / inner_tile_ids / if-standard
   sizing split). The human decides if the perf trade is acceptable.

## 5. ENVIRONMENT + GPU rules (verbatim — gotchas matter)
- Worktree `/home/dev/local/helion-redesign`, branch `reduction-redesign`, start at HEAD `b2df25a9`
  (or later), tree clean. Interpreter `/home/dev/helion/.venv/bin/python`. NEVER pip install / git
  push. `PYTHONPATH=/home/dev/local/helion-redesign`. `HELION_AUTOTUNE_EFFORT=none` for recorder /
  validators / unit tests (no autotune); perf benching does its OWN do_bench (see §6).
- GPU: this is an H100 box. Perf benching NEEDS the GPU. HARD RULES (from project memory):
  * NEVER background/detach a GPU job (`run_in_background:true` on a GPU job dies silently -> 13h
    stall). Run every GPU bench FOREGROUND, one kernel/shape at a time.
  * do_bench cross-process jitter is ~5-10%; use SINGLE-PROCESS head-to-head (before vs after on the
    same tensors in one process) — replay_bench.py / probe_perf.py already do this. A <10% delta is
    noise, not signal.
- COMMIT after each meaningful green step; end messages with:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Keep a running log at `_lab/redesign/BUDGET_ALLOCATOR_LOG.md` (append throughout: design choices,
  config diffs, perf numbers, budget-constant iterations, decisions/surprises). This is your
  resumable trail AND the source of the final human report.

## 6. THE TOOLS (verbatim commands; cwd MATTERS)
```
# CONFIG RECORDER — record all emitted configs; diff vs the frozen baseline to SEE what moved.
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
  PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
  /home/dev/local/helion-redesign/_lab/harness/unified_config_recorder.py --out /tmp/after.json
cd /tmp && PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
  /home/dev/local/helion-redesign/_lab/harness/unified_config_recorder.py \
  --diff /home/dev/local/helion-redesign/_lab/unify/baseline_fc1dbaa0_configs.json /tmp/after.json
# (baseline_fc1dbaa0_configs.json = the ORIGINAL heuristic's configs = your perf reference point.
#  The 2 "known movers" mreduction/{layer,rms}_norm_bwd/[4096,8192]/fp16 are a pre-existing
#  starvation FIX, not a regression — expect them plus MANY more now.)

# PER-CELL PERF (before-config vs after-config, single-process, median-of-9, accuracy-gated):
#   _lab/unify/replay_bench.py — full corpus, one fresh process per kernel (read its header for args).
#   _lab/redesign/probe_perf.py --all — the 13 probe kernels seed-vs-default.
# Also: --default mode compares seed vs the compiler DEFAULT (the "must beat default" floor).

# FACT FAITHFULNESS + Tier-1 (should stay green; config-independent):
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=/home/dev/local/helion-redesign \
  /home/dev/helion/.venv/bin/python /home/dev/local/helion-redesign/_lab/redesign/validate_kernel_fact.py  # 460/460
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=/home/dev/local/helion-redesign \
  /home/dev/helion/.venv/bin/python /home/dev/local/helion-redesign/_lab/redesign/probe_assertions.py     # 13/13

# UNIT TESTS — FROM THE WORKTREE DIR (from /tmp pytest reports "no tests ran"):
cd /home/dev/local/helion-redesign && PYTHONPATH=$PWD HELION_AUTOTUNE_EFFORT=none \
  /home/dev/helion/.venv/bin/python -m pytest test/test_reductions.py test/test_autotuner_heuristics.py \
  test/test_examples.py -k "matmul_layernorm or reduction" -q -p no:cacheprovider

# IR INTROSPECTION (see a kernel's descriptors/groups/accumulators while designing):
cd /tmp && ... python _lab/redesign/ir_introspect.py --probes   # or --kernel rms_norm
# Fact dump per corpus kernel: _lab/redesign/fact_dump.py
```

## 7. THE KERNEL CORPUS (what the recorder/bench sweep — for tracing a mover)
- 9 reduction: `examples/{rms_norm,layer_norm,softmax,sum,cross_entropy,welford,kl_div,jsd,long_sum}.py`
- 8 transfer: `/home/dev/local/prompts-lab/transfer/transfer_kernels.py` + `examples/grpo_loss.py` +
  `examples/fused_linear_jsd.py`; shapes `/home/dev/local/prompts-lab/transfer/shapes_transfer.py`
- 6 m-reduction (norm-bwd): `/home/dev/local/prompts-lab/vllm-bench/mreduction_styles_view_only.py`
  (bias_grad/dyt/group_norm/instance_norm) + rms_norm_bwd/layer_norm_bwd in examples/. THESE are the
  grad-param M-collapse kernels — the recognizer you're deleting; they MUST now fall out of the
  budget (resident grad-param tensor tightens the budget). EXPECT their configs to move; bench them.
- 5 vLLM: `/home/dev/local/prompts-lab/vllm-bench/kut/*.py` — incl. the `group_size` grid-axis
  reduction that must be seated RESIDENT (the floor-vs-resident contrast with jsd).
- 13 stress probes: `/home/dev/local/prompts-lab/reduction-generality/kernels/<slug>/kernel.py`.

## 8. COMPANION DOCS (read for the WHY)
- `/home/dev/local/prompts-lab/reduction-generality/PROMPT.md` — the LOCKED design. §2.3 (budget +
  greedy/priority allocation, group_footprint, cross-group ordering), §2.4 (Cap/size_axis — the
  primitive; KEEP it as the per-axis min-of-caps mechanism), §6.2/§6.2.1 (primary + num_warps owner).
- `_lab/redesign/CARRIED_AND_GREEDY_FINDINGS.md` — the audit findings #2/#3/#4 that motivated this
  (carried-tile footprint = ∏ within a buffer / Σ across buffers, membership not position).
- `_lab/redesign/WORKLOG.md` — the full redesign narrative.
- `_lab/redesign/GREEDY_ALLOCATOR_LOG.md` — the PRIOR (rejected) attempt's log; read to see what the
  human rejected as "window dressing" so you don't repeat it.

## 9. DEFINITION OF DONE
- ONE budget allocator: per co-residency group, form one (scalable) budget, seat axes
  first-crack-then-floor-by-remaining-budget, hold fixed, next group, then non-reduction loops.
- Floor-vs-resident is a budget OUTCOME (no cdiv branch, no recognizer). The standard/user split is
  EMISSION-ONLY. `_is_per_feature_accumulator`, `inner_tile_ids`, `_grad_collapse_group`, the
  `_carried_*` footprint caps, `_resident_tile_cap`, `_pinned_inner_resident_elems` are GONE or
  folded into the budget. `grep` confirms no recognizer remains.
- Correctness gates green (validate_kernel_fact 460/460; probes 13/13 fired-right-path;
  matmul-epilogue green; unit tests updated to the new behavior, not contorted-around).
- Final human report (§4.7): full config-change table + per-cell perf + net summary + landed budget
  constants + recognizer-removal confirmation. Human decides if the perf trade is acceptable.

## 10. NON-GOALS
- Do NOT touch `TritonMatmulReductionEpilogueHeuristic` or delete `ReductionFact` (epilogue + gate
  use it). Do NOT change Stage-1 device_ir fact-building unless you need a new faithful field.
- Do NOT remove `rollable`/`pinned` descriptor fields (separate cleanup).
- Do NOT chase byte-identical configs. Build the principled allocator; let configs move; recover
  perf via the BUDGET.
```
