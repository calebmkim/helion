# Stage-2 Landscape: the 6 backward M-reduction kernels (compile-time map)

Worktree `helion-3stage` @ branch `reduction-3stage-stack` (Stage-1 committed).
Compile-time only (bind + factdump + roller trace) — NO GPU timing.
Probe: `_lab/stage2_unify/probe_stage2.py` (factdump + ReductionLowering-axis census);
roller-decline trace: `/tmp/probe_roll3.py` (monkeypatches `ReductionRoller.process`).
Verified identical at **fp32 and bf16** (Helion fp32-promotes the norm family).

## Goal recap
Port the "materialized M-reduction recognizer" so the 6 bwd kernels are served by the
STANDARD (T1) or USER-TILED (T2) reduction track. On `main`/this branch there is no
`m_reduction` heuristic; the kernels with a **materialized** inner reduction register
**0 reduction facts** and fall to the catastrophic generic `[32,32]` default.

---

## TASK 1 — Factdump table (representative TRAIN shape, fp32; bf16 identical)

| kernel | shape | n_red_facts | track | n_mm_facts | seed | mat-inner count | mat-inner axes (blk,#lowerings) |
|---|---|---|---|---|---|---|---|
| `rms_norm_bwd`      | M=8192,N=4096 | **0** | none | 0 | **no seed → default [32,32]** | **1** | blk2 (×2) |
| `layer_norm_bwd`    | M=8192,N=4096 | **0** | none | 0 | **no seed → default [32,32]** | **1** | blk1 (×4) |
| `bias_grad_bwd`     | M=16384,N=1024 | 1 | **user-tiled T2** | 0 | **YES** (block_sizes=[1,1], num_warps=16) | **0** | — |
| `dyt_bwd`           | M=16384,N=1024 | 1 | **user-tiled T2** | 0 | **YES** (block_sizes=[1,1], evict=[last,first,first], nw=16) | **0** | — |
| `group_norm_bwd`    | N=512,C=128,S=64,G=32 (F=8192) | **0** | none | 0 | **no seed → default [32,32]** | **2** | blk3 (×8), blk5 (×4) |
| `instance_norm_bwd` | B=512,C=64,S=128 (F=8192) | **0** | none | 0 | **no seed → default [32,32]** | **1** | blk3 (×8) |

ReductionFact detail for the 2 kernels that DO register one (user-tiled T2):
- `bias_grad_bwd`: block_id=1, size_hint=8192, full_width_output=False, num_carried_2d_tiles=0,
  body_live_tiles=1, non_reduction_loop_block_ids=[], m_block_ids=[0], itemsize=4, num_load=1.
- `dyt_bwd`: block_id=2, size_hint=8192, full_width_output=False, num_carried_2d_tiles=0,
  body_live_tiles=3, non_reduction_loop_block_ids=[], m_block_ids=[0], itemsize=4, num_load=3.

### ReductionLowering-axis census (block_index of every `ReductionLowering`, classified)
"MATERIALIZED" = axis is in NEITHER `block_sizes` nor `reduction_loops` (i.e. an inner
reduction that is neither rolled-T1 nor user-tiled-T2); grid axes excluded from the count.

| kernel | in_block_sizes (user-tiled) | in_reduction_loops (rolled T1) | MATERIALIZED (neither) | grid |
|---|---|---|---|---|
| `rms_norm_bwd`      | blk1 (×1, normalize/apply N tile) | — | **blk2 (×2)** — feature-N sum, size 4096 | blk0 (M) |
| `layer_norm_bwd`    | blk2 (×2, normalize/apply N tile) | — | **blk1 (×4)** — feature-N sums, size 4096 | blk0 (M) |
| `bias_grad_bwd`     | blk1 (×1) — the inner `hl.tile` over M | — | none | blk0 (M) |
| `dyt_bwd`           | blk2 (×2) — the inner `hl.tile` over M | — | none | blk0 (M) |
| `group_norm_bwd`    | blk2 (×2) — the per-CTA M tile | — | **blk3 (×8) S=64 + blk5 (×4) Cg=4** — two materialized | blk0 (N) |
| `instance_norm_bwd` | blk2 (×2) — the per-CTA B tile | — | **blk3 (×8) S=128** — spatial reduce | blk0 (B) |

### Structural reading (confirms the hypothesis exactly)
- **1 materialized inner reduction** → needs the STANDARD-track port:
  `rms_norm_bwd`, `layer_norm_bwd`, `instance_norm_bwd`.
  These have a single materialized feature/spatial reduction axis (rms/ln: feature-N; instance:
  spatial-S over a DIFFERENT axis than the channel-C param accumulator). The grad_w/grad_b
  accumulator collapses M via the per-CTA `hl.tile` (a `block_sizes` axis, NOT a ReductionLowering).
- **0 materialized** → pure M-collapse, already served by USER-TILED T2:
  `bias_grad_bwd`, `dyt_bwd`. The only reduction is `sum_M grad_out` over the inner `hl.tile`
  loop, which is a `block_sizes` axis ⇒ a user-tiled ReductionFact + seed already fire today.
- **several materialized** → multi-materialized, currently skipped:
  `group_norm_bwd` (TWO: spatial-S blk3 + intra-group-channel Cg blk5).

---

## TASK 2 — Exact decline-point map (`main`/this branch)

### A. Why rms/ln/instance register 0 `reduction_loops` (the ROOT decline)
`helion/_compiler/roll_reduction.py` — `ReductionRoller.should_go_in_inner_graph`,
the `is_for_loop_target` branch (**roll_reduction.py:107–124**, raise at **:123**):

```
if used_infos:
    if not all(info.can_be_rolled_by_caller for info in used_infos):
        raise NotImplementedError("for loop with mixed reduction dim usage")   # :123
```

Traced order (identical shape for all 3; from `/tmp/probe_roll3.py`):
1. The roller processes the **inner** per-CTA `hl.tile` body graph first → succeeds, but with
   `outer_count != 0` (rms 20 / ln 28 / instance 18 nontrivial non-reduction nodes left outside
   the rolled reduction) and/or `graphs_added != 1` ⇒
   `RolledReductionInfo.can_be_rolled_by_caller = (outer_count==0 and len(graphs_added)==1)` = **False**
   (set in `register_rollable_reductions`, **device_ir.py:955–956**).
2. The roller then processes the **outer** graph holding the `_for_loop` node that references that
   inner info; `should_go_in_inner_graph` sees `used_rdim=True` but `can_be_rolled_by_caller=False`
   ⇒ raises `NotImplementedError("for loop with mixed reduction dim usage")`.
3. In `register_rollable_reductions` the `except NotImplementedError` (**device_ir.py:948–950**)
   sets `all_graphs_processed=False` ⇒ `allow_loop=False` (**:963–964**) ⇒ the second pass
   (**:970–991**) registers NO `ReductionLoopSpec` ⇒ `spec.reduction_loops` empty ⇒
   `_rollable_reduction_records` empty ⇒ `build_reduction_facts` builds **0 standard facts**.

Root cause in plain terms: the feature/spatial reduction lives INSIDE the user's hand-written
per-CTA `for mb in hl.tile(mb_cta.begin, mb_cta.end)` loop, mixed with non-reduction work (grad_x
apply + grad_w accumulate). The materialized inner reduction is therefore neither auto-rollable
(T1) nor a `block_sizes` user tile (T2) — it is `reduction=True` but in NEITHER spec.
(NB: the `has_matmul/has_stack/has_unrollable` can_roll gate at device_ir.py:925–938 PASSES for
all 3 — the decline is purely the mixed-for-loop NotImplementedError, not those predicates.)

### B. `register_user_tiled_reductions` (device_ir.py:1060–1142) — no materialized branch (confirmed)
- Caller-guarded `if not spec.reduction_loops:` (**device_ir.py:1186**) — only runs when no standard.
- Collects every `ReductionLowering.block_index`, drops grid axes → `inner_red`
  (**:1088–1097**). Then the gate: `if len(inner_red) != 1: return` (**:1098**) and
  `if red_block_id not in spec.block_sizes.valid_block_ids(): return` (**:1101–1103**).
- ⇒ It REQUIRES the single inner reduction axis to be a **`block_sizes`** entry (a user `hl.tile`).
  A **materialized** axis (in NEITHER spec) fails the `:1102` block_sizes membership check ⇒ return.
  group_norm additionally fails the `:1098` `len(inner_red)!=1` check (it has 2).
  **There is no materialized-inner branch on this branch — confirmed.** This is what Stage 2 adds.

### C. Eligibility + track discriminator (`triton.py`)
- `_triton_reduction_eligible` (**triton.py:301–305**): `len(spec.reduction_facts) == 1 and not
  spec.matmul_facts`. With 0 facts (rms/ln/group/instance) the gate fails ⇒ neither reduction
  heuristic fires ⇒ no seed ⇒ generic default.
- `_is_standard_reduction` (**triton.py:308–313**): `return fact.block_id in
  spec.reduction_loops.valid_block_ids()`. Standard iff the rdim is a rollable `reduction_loops`
  entry; else user-tiled. A materialized rdim is in NEITHER spec, so even if a fact existed it
  would be classed "user-tiled", but `TritonUserTiledReductionHeuristic` ultimately relies on the
  rdim being a `block_sizes` knob (`_build_block_sizes(... fact.block_id, r_block ...)`), which a
  materialized axis is not. ⇒ today no track can serve a materialized inner reduction.

### D. Does `TritonStandardReductionHeuristic.get_seed_config` handle a materialized rdim?
Read **triton.py:674–742**. The standard branch ALREADY emits `reduction_loops: [None]` (persistent)
or `[r_block]` (looped) and `red_block_id=None` into `_build_block_sizes` — i.e. it does NOT need
the rdim to be a `block_sizes` entry; it sizes the block_sizes from the grid/normalize axes and
rides persistent-vs-looped on the `reduction_loops` knob (**triton.py:712–725**). So the seed-EMIT
machinery is materialized-ready. The MISSING piece is upstream: a materialized rdim has **no
`reduction_loops` spec entry**, so (a) no fact is built and (b) `_is_standard_reduction` returns
False. The Stage-2 port must (1) build a ReductionFact for the materialized axis and (2) register
a `reduction_loops` (or equivalent) spec entry for it so the standard track recognizes + seeds it.
(WS2 already proved `reduction_loops: []`/`[None]` is the right emission for a non-block_sizes rdim.)

---

## TASK 3 — Per-kernel verdict

| kernel | facts today | track today | mat-inner | seed-or-default | Stage-2 action |
|---|---|---|---|---|---|
| `rms_norm_bwd`      | 0 | none | **1** | DEFAULT [32,32] (catastrophic) | **standard-track port** |
| `layer_norm_bwd`    | 0 | none | **1** | DEFAULT [32,32] (catastrophic) | **standard-track port** |
| `instance_norm_bwd` | 0 | none | **1** | DEFAULT [32,32] (catastrophic) | **standard-track port** |
| `bias_grad_bwd`     | 1 | user-tiled T2 | 0 | **SEEDED** (already served) | none (control) |
| `dyt_bwd`           | 1 | user-tiled T2 | 0 | **SEEDED** (already served) | none (control) |
| `group_norm_bwd`    | 0 | none | **2** | DEFAULT [32,32] (catastrophic) | multi-materialized (later) |

**Hypothesis CONFIRMED** (fp32 + bf16):
- rms_norm_bwd / layer_norm_bwd / instance_norm = **1 materialized inner reduction** → standard-track port.
- bias_grad / dyt = **0 materialized** (pure M-collapse) → already user-tiled, already seeded.
- group_norm = **several (2) materialized** → multi-materialized, skipped (out of scope for the first port).

---

## Decline points to edit for the recognizer port (file:line)

1. **`helion/_compiler/roll_reduction.py:123`** — `raise NotImplementedError("for loop with mixed
   reduction dim usage")` in `should_go_in_inner_graph`'s `is_for_loop_target` branch
   (:107–124). The root decline: a materialized inner reduction inside a user `hl.tile` loop with
   mixed (non-reduction) body. Either relax here, or recognize this axis BEFORE the roller bails.
2. **`helion/_compiler/device_ir.py:1098 & :1101–1103`** — in `register_user_tiled_reductions`,
   the `len(inner_red) != 1` and `red_block_id not in block_sizes` gates. Add the materialized-inner
   branch here (the single materialized rdim that is `reduction=True` but in neither spec → build a
   ReductionFact + register the standard `reduction_loops` spec entry for it).
3. **`helion/_compiler/device_ir.py:970–991`** (second pass of `register_rollable_reductions`) /
   **:1186** (`build_reduction_facts` user-tiled guard) — where a `reduction_loops` ReductionLoopSpec
   would be registered + the standard fact stashed. The port must register the materialized axis as a
   standard `reduction_loops` entry so `_is_standard_reduction` (triton.py:313) classifies it T1.
4. **`helion/_compiler/autotuner_heuristics/triton.py:313`** (`_is_standard_reduction`) /
   **:301–305** (`_triton_reduction_eligible`) — the track discriminator + eligibility. Once a
   materialized fact + reduction_loops entry exist (edits 2–3), the standard heuristic's emit path
   (triton.py:712–725, already materialized-ready, emits `reduction_loops:[None]/[r_block]`) fires
   with no change. Verify the discriminator routes the materialized rdim to T1, not T2.
