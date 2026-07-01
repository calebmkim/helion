# Handoff: faithful live-tile footprint + simplified budget

> A phased handoff for the next agent. The budget-allocator footprint in
> `_TritonReductionSeedBase.size_reduction_tiles` (helion/_compiler/autotuner_heuristics/triton.py)
> works and is committed, but its footprint is a *reconstruction* of the resident set from
> (num_live count + reduction/accumulator/feature axes). Through a long review we established that
> **no faithful footprint is reconstructable from those** — it needs each live tile's actual
> dim-set. A Stage-1 fact for that is scaffolded (uncommitted-then-committed WIP). This doc is the
> plan to finish it: build the fact properly, simplify the footprint to one budget over the real
> tiles, triage regressions, report, then tune.

---

## 0. STATE / WHERE THINGS STAND

Branch `reduction-redesign`. Interpreter `/home/dev/helion/.venv/bin/python`. PYTHONPATH = the
worktree `/home/dev/local/helion-redesign`. `HELION_AUTOTUNE_EFFORT=none` for recorders/validators/
tests. GPU FOREGROUND only, single-process, one kernel/shape at a time (project hard rule — NEVER
detach). NEVER pip install / git push / git commit unless asked.

### Commits already landed this effort (all gates green, all config-neutral OR measured net-win)
- `8e71c601` — #1: unified the footprint into ONE two-regime Σ-over-resident-tensors formula
  (deleted `carried_mult`, the `feature_footprint==1` gate, the lossy in_accumulator-multiplied grid
  term). CONFIG-NEUTRAL (447/447 byte-identical).
- `4fa5be1c` — #2 (F: named the `drop_body_weight_for_reduce_then_apply` proxy) + #3 (G: FULL_GRID
  seats at full extent, corpus no-op) + #4 (D+E comment-only). CONFIG-NEUTRAL.
- `e8c7c477` — the FAITHFUL(er) `(scale, flat)` footprint: resident bytes = `itemsize × (scale·R +
  flat)`, accumulators ADD (not multiply), + THREE budgets (GRAD_PARAM 3·ROW/2 / CARRIED ROW/2 /
  ROW). MEASURED NET WIN: 36 cells move, all wins/neutral (kl_div 8192→4096 +2%, bias_grad +2%),
  geomean ~0.98, no regressions. **This is the current HEAD and a good fallback if the work below
  doesn't pan out.**

### WIP scaffold (committed on top as a separate WIP commit — see §7 to commit it)
- `helion/_compiler/device_ir.py`: added `_graph_peak_live_tiles(graph, env)` — the liveness sweep
  that returns each live tile's `dim_block_ids` at the peak-live step (NOT just the per-axis count
  that `_graph_peak_live_by_axis` returns). **Currently has ZERO callers — inert, behavior
  unchanged.** This is the seed of the new Stage-1 fact.
- `_lab/redesign/dump_facts.py`: dumps `live_tiles_by_graph` per cell so offline models can read it.

---

## 1. THE CORE FINDING (why we're doing this)

The committed footprint reconstructs the resident set from `read_dims = sized_reductions ∪
in_acc_grid ∪ feature_extent` plus a separate accumulator sum. Through review (see
BUDGET_ALLOCATOR_LOG.md CF-Step 6 and the transcript) we proved this is NOT faithful:

1. **`feature_extent` is a confusing proxy** — a "materialized feature" is SOMETIMES a `full_slice`
   reduction (layer/rms/group/instance-bwd) and SOMETIMES not-a-reduction (bias_grad, dyt). Using it
   to shape the read tile is redundant for the former (already in `sized`) and load-bearing for the
   latter. It is `⊆ sized ∪ accumulator_dims` on the corpus, so it adds no information — it's a
   reconstruction artifact.
2. **`build_accumulator_facts` over-counts** — it emits one `AccumulatorFact` per loop-carried
   tensor PER LOOP NESTING LEVEL. layer_norm_bwd has 2 logical accumulators (grad_w[N], grad_b[N])
   but 5 facts (`[None,1],[1],[1],[2,1],[2,1]`) — same buffers at different loop levels. Summing all
   5 over-counts the footprint (~2.5×).
3. **`read_dims` can't know a tile's real shape** — it assumes every live tile spans every read_dim.
   For `sum` (`load [m,n]`, reduce n → `[m]`) it can't tell the `[m,n]` read from the `[m]` result.
   THE ROOT: faithfulness requires **each live tile's actual dim-set**, which none of the current
   facts carry — only `body_live_tiles` (a scalar count).

The fix is the live-tile-shapes fact (§2) + a footprint that sums the ACTUAL tiles (§3).

### Verified properties of `_graph_peak_live_tiles` (the new fact)
- Liveness sweep matches the intuitive model: at `c = a + b`, all of a, b, c are counted live
  (3), then a,b die if last-used there. CONSERVATIVE over-count of register pressure (no
  reuse/remat modeled) — errs toward smaller tiles, never an unsafe spill. STATED, not random.
- **Captures accumulators inline, once each, at real shape** (verified: kl_div's `[1,0]` carried
  buffer is among its 6 live `[1,0]` tiles; softmax's `[0]` carries present). So the footprint sums
  live tiles and NEVER needs a separate accumulator sum — the `build_accumulator_facts` over-count
  (§1.2) simply doesn't touch the footprint anymore. (accumulator_facts is still needed for its
  OTHER consumers: `carried_2d_count`, the budget-regime flag.)
- **Distinguishes resident vs reduced-away grid axes faithfully** — a grid axis is resident iff it
  appears in a live tile. This is STRICTLY MORE FAITHFUL than the old `in_accumulator(g)` gate,
  which got `sum` wrong (sum's m IS a resident read-tile dim but is in no accumulator). See §4.

---

## 2. PHASE A — build the fact PROPERLY (the hard, delicate part)

`_graph_peak_live_tiles` extracts tiles PER GRAPH, but the footprint needs them attributed to
CO-RESIDENCY GROUPS, and there are TWO control-flow subtleties that make a naive per-kernel union
WRONG (both over-count in the damaging direction):

1. **If/Else branches are MUTUALLY EXCLUSIVE** — a `torch.where`/mask compiles to `IfGraphInfo` +
   `ElseGraphInfo`; only ONE executes per element. Their tiles must combine as **max, not sum**.
   ONLY 3 corpus kernels have If/Else reduction graphs: **jsd, grpo, per_token_group_fp8_quant**
   (verified — survey in transcript). Everything else is ForLoop/ReductionLoop/Root only.
2. **Sequential co-residency groups are NOT simultaneously resident** — jsd has 3 groups (3 V-reductions
   run in sequence); summing them over-counts ~3×. 42 corpus kernels are multi-group (all jsd-family
   + the multi-`ReductionLoopGraphInfo` norm kernels). Each group's tiles must be summed SEPARATELY,
   then the group budget applies per-group.

### The graph-namespace gotcha (why a quick join fails)
- Co-residency groups are keyed on ORIGINAL graph_ids (from `_original_graph_reductions`, which
  EXCLUDES `ReductionLoopGraphInfo` — the roller's per-config copies).
- A ROLLED reduction's body tiles live in a `ReductionLoopGraphInfo` (e.g. layer_norm's reduction is
  in graphs 2,3; the group is keyed 0). A USER-TILED reduction's body is in a `ForLoopGraphInfo`.
- jsd's group graph_ids (1,3,5) are `IfGraphInfo`/`RootGraphInfo`, and its reduction-body tiles are
  scattered across Else(0,2)/If(1,3)/ForLoop(4)/Root(5). No clean 1:1 group↔graph.

### What Phase A must produce
A Stage-1 fact giving, PER CO-RESIDENCY GROUP, the resident tile set (each tile = `dim_block_ids`),
combining: **max across If/Else siblings, sum within a group, separate across sequential groups.**
Reuse the existing control-flow walker (`device_ir.py` ~line 888, the `walk(graph_id, path)` over
if/else) and `_original_graph_reductions` (~1400) to map body graphs → groups. Attribute a live
tile to a group by the reduction/loop nesting that encloses its graph.

RECOMMENDED SHAPE: store `live_tiles: tuple[tuple[int|None, ...], ...]` on each `CoResidencyGroup`
(or a parallel per-group list on `ReductionKernelFact`). Build it in `build_reduction_kernel_fact`
(device_ir.py ~1482) where the group structure + graph walk are already in scope.

VERIFY the fact on: single-group easy cases (sum, softmax, kl_div, layer_norm_bwd, bias_grad) AND
the 3 If/Else kernels (jsd, grpo, per_token_group) — confirm branches are max'd and groups separated.
Cross-check tile counts against `body_live_tiles` for sanity (should be >= per-axis count).

---

## 3. PHASE B — the simplified footprint (likely ONE budget)

Replace `footprint_terms` + `tile_width` + `read_dims` + the 3-budget selector with:

```
resident_bytes(axis A, block b_A) = itemsize × Σ over the group's resident live tiles of
                                     ∏(dim widths of that tile)        # A at variable b_A
```
using the SCALE/FLAT split already in the committed code (a tile CONTAINING A scales with b_A → adds
to `scale`; a tile WITHOUT A is constant → adds to `flat`). Persistence: `itemsize×(scale·raw+flat)
≤ budget`. Chunk/widen: `b_A ≤ (budget/itemsize − flat)/scale` (flat SUBTRACTED not divided — the
softmax-persistence bug; see CF-Step 6).

**Delete:** `read_dims` reconstruction, `feature_extent` from the footprint (keep it only for
`loop_budget` + the regime flag if still needed), the separate accumulator sum, the `tile_width`
feature-vs-reduction dispatch, and the `in_accumulator(g)` grid gate (replace with "g in a live
tile", §4).

**Single budget hypothesis:** the 3 budgets (GRAD_PARAM/CARRIED/ROW) existed to COMPENSATE for the
unfaithful footprint (correction factors for over/under-count). Summing the REAL tiles once each
should let them collapse to ONE budget — BUT the faithful `scale` now counts ALL live tiles (2–7×
where the old counted ~1), so the single budget's calibrated VALUE will differ from 245760. Expect
to re-tune the one constant. TRY one budget first; only split if a regime genuinely can't be
expressed by tile-counting (report which + why).

### The two-pass sizing structure (the human's plan, endorsed)
1. **Pass 1 — size reductions with grid axes at their FLOOR** (`_m_axis_block_size` = `_block_floor`,
   NOT 1 — they coincide except on large-M shapes where autotuner_min raises the floor; use floor).
   Grid dims then contribute their floored width to the footprint.
2. **Pass 2 — size/widen each grid axis** with reductions now seated. The widen must RE-CHECK the
   byte budget at the tile size that exists AT THAT POINT (`scale·b_m + flat ≤ budget`), SEPARATE
   from the occupancy cap. This is the piece the old code half-missed.

---

## 4. PHASE B detail — the `.sum(0)` / reduced-away grid axis (KEEP as a distinct branch)

A grid axis is one of two physically-different things, and the live-tile fact tells them apart:
- **Resident** (in some live tile — softmax/sum/kl_div/grpo): widening it holds `b_m ×` its tile →
  BYTE-budget-limited. `sum`'s m IS resident (read tile `[m,n]`) — the OLD `in_accumulator` gate got
  this WRONG (said not-resident); "in a live tile" gets it right.
- **Reduced-away** (in NO live tile — bias_grad/norm-bwd): the grad-parameter idiom has TWO m-axes —
  an OUTER `hl.tile` grid block (the program-count / loop-trip axis, holds NO bytes) and an INNER
  reduction loop. The outer axis is finalized by a separate `.sum(0)` over per-program `[N]` partials.
  Widening it costs NO bytes (just fewer/fatter programs), so the byte budget can't size it — it's
  purely an OCCUPANCY / finalize-cost lever (collapse floor = `grid_rows/num_sm`). KEEP this branch;
  trigger it on "grid axis in NO live tile" (faithful) instead of "not in_accumulator".

---

## 5. PHASE C — REGRESSION TRIAGE (do this BEFORE fine work; report back)

Expect MANY configs to change (the footprint is materially different). Do NOT try to make it
config-neutral. Instead:
1. Recorder diff vs `/tmp/before_rewrite.json` (regen at HEAD if stale — §6) — get the changed-cell
   list + field.
2. **Flag the DISASTROUS swings first** — it's usually obvious (a reduction tile going to full extent
   8192 when it was 2, or a grid collapsing 64→1). Grep the diff for order-of-magnitude block_size
   jumps. These are spills/starvation — fix the footprint/budget, don't bench them all.
3. Once no obvious disasters, single-process bench the movers (§6 replay_bench) and REPORT BACK with
   the swing table before tuning. The human wants a checkpoint here.

Known sensitivities to watch (measured this effort):
- grad-param inner tile (layer/rms/dyt/group/instance-bwd): TRUE optimum is inner=4 at N=4096 (~+10%)
  but the CLIFF is violent — inner=8 is 1.5x, inner=16 is 10-15x. Stay on the safe side; the budget
  must not let the inner tile grow past ~4.
- softmax/welford: persistence-sensitive; the scalar `[grid]` carries must be FLAT (constant in R),
  never multiplied through the extent (that denied softmax persistence in a naive attempt).
- kl_div wants R=4096 (measured +2% vs 8192); jsd wants R=2048; grpo ~1.2x is ACCEPTED (streamed
  regime, needs a Stage-1 axis-resolved-liveness fact to fix — separate follow-up).

---

## 6. VERIFICATION + BENCH MECHANISMS (critical — the next agent needs these)

All under `_lab/`. Run recorders/validators with `HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=
$(mktemp -d) PYTHONPATH=/home/dev/local/helion-redesign` and cwd `/tmp`.

### Config recorder (the primary instrument — 447 cells across 4 corpora)
```
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
  PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
  /home/dev/local/helion-redesign/_lab/harness/unified_config_recorder.py --out /tmp/after.json
# diff (exit 1 if any cell changed; prints changed cells):
  ... unified_config_recorder.py --diff /tmp/before_rewrite.json /tmp/after.json
# REGEN the before-baseline at current HEAD first (the /tmp one is wiped on reboot):
#   record to /tmp/before_rewrite.json at HEAD e8c7c477 BEFORE editing.
```

### Correctness gates (must stay green)
```
_lab/redesign/validate_kernel_fact.py     # expect: 460 pass, 0 fail
_lab/redesign/probe_assertions.py         # expect: Tier-1 13 pass, 0 fail
pytest test/test_reductions.py test/test_autotuner_heuristics.py -q   # 52 passed / 22 skipped
pytest test/test_examples.py -k matmul_layernorm -q                   # 2 passed
# test_autotuner_heuristics.py::...test_kl_div_wide_seeds_band_b_r_block_cap pins kl_div [4096,1];
# if the budget changes kl_div's R, UPDATE that test to the new value (don't contort the allocator).
ruff:  /home/dev/helion/.venv/bin/ruff format <file> ; /home/dev/helion/.venv/bin/ruff check <file>
```

### Single-process perf bench (foreground, one cell per process — footgun #11)
```
cd /tmp && HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=/home/dev/local/helion-redesign \
  /home/dev/helion/.venv/bin/python /home/dev/local/helion-redesign/_lab/unify/replay_bench.py \
  --corpus curriculum --kernel kl_div --shape 8192,32000 --dtype fp32 \
  --before '{"block_sizes":[8192,1],"num_warps":32,"num_stages":1,"pid_type":"flat"}' \
  --after  '{"block_sizes":[4096,1],"num_warps":32,"num_stages":1,"pid_type":"flat"}'
# prints ratio_after_over_before (>1.10 = regression); median-of-9; --default compares vs compiler default.
# NB: replay_bench uses plain do_bench; for bandwidth-bound seed-vs-tc use CUDA-graph/device-time
#     (memory note [[project_rms_norm_tc_gap_is_host_overhead]]). For A/B of two SEEDS it is fine.
# NB: a GPU OOM warning appeared when binding many kernels in ONE process — keep runs lean / per-cell.
```

### Offline fact dump + models (fast iteration WITHOUT GPU binds)
```
_lab/redesign/dump_facts.py --out /tmp/corpus_facts.json   # dumps ALL Stage-1 facts + NOW live_tiles_by_graph
_lab/redesign/model_alloc.py       # replicates the CURRENT allocator (validated 443/443 vs recorded seeds)
_lab/redesign/spike_faithful.py    # the (scale,flat) model spike (has fidelity gaps vs live — see below)
_lab/redesign/sweep_footprint.py   # sweeps footprint-formula variants vs the oracle
```
IMPORTANT: the offline models had a 2-cell fidelity gap vs the LIVE recorder (bias_grad phantom
feature, 4D group_norm feature extents). **The LIVE recorder is the authority** — always confirm a
model prediction against a recorder run before trusting it. Once the live-tile fact is on the
descriptor, `model_alloc.py` should read it (not re-derive read_dims) so the model regains fidelity.

### fact_dump for eyeballing one kernel's facts
```
_lab/redesign/fact_dump.py --corpus curriculum --kernels jsd,kl_div   # prints reductions/accumulators/groups
```

---

## 7. IMMEDIATE FIRST STEPS FOR THE NEXT AGENT

1. Confirm clean at HEAD `e8c7c477` (`git status`; the WIP scaffold commit may sit on top — see below).
2. Commit the WIP scaffold if not already: `_graph_peak_live_tiles` (device_ir.py, inert) +
   dump_facts.py live-tiles. (Message: "wip: scaffold live-tile-shapes fact (no callers)".)
3. Regen `/tmp/before_rewrite.json` at HEAD (recorder, §6). Confirm gates green (460/460, 13/13).
4. **Phase A**: build the per-group, If/Else-max, sequential-separate live-tile fact in device_ir
   (§2). Verify it on sum/softmax/kl_div/layer_norm_bwd/bias_grad + jsd/grpo/per_token_group.
5. **Phase B**: rewrite `footprint_terms` to sum the group's real live tiles (§3), try ONE budget,
   two-pass sizing (§3), reduced-away branch on "in no live tile" (§4).
6. **Phase C**: recorder diff, flag disasters, REPORT BACK before benching everything (§5).
7. Then tune the single budget constant, bench the movers, gates green, commit.

## 8. THE DEFENSIBLE FALLBACK
If the faithful build proves too costly or regresses net-negative: HEAD `e8c7c477` is already a
measured net-win over the original and is fully green. It's a fine place to stop. The faithful
live-tile footprint is the RIGHT direction (removes every reconstruction hack) but is not required
for a shippable result — frame it as an improvement, not a prerequisite.
