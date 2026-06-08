# Welford / `is_structured_combine`: findings for the synthesizing agent

> Scope of this doc: **the welford kernel and whether `ReductionFact.is_structured_combine`
> is necessary.** Assumes you know the overall project (seed heuristic for forward
> inner-reductions, H100/sm90/fp32/Triton; per-shape `seed ≈ oracle` bar; the
> T1/T2/Band-B/Band-C structure). This is one input among several conversations you are
> synthesizing — treat the *findings* as load-bearing and the *implementation plan* at the
> end as a non-binding proposal.

## TL;DR

- **`is_structured_combine` is NOT necessary. Delete it.** Under the premise below it goes
  *dead on its own* (never fires) once the welford kernel shares one tile.
- The cliff it was implicitly protecting against is a **generic working-tile register spill**
  driven by **`M_BLOCK × tile_N × itemsize`** — NOT by the reduce-then-apply structure, and
  NOT by accumulator size. Cliff-protection belongs on the whole T2 path, keyed on footprint.
- **Keep** the `apply_block_ids` field (rename it; e.g. `coupled_tile_block_ids`), and for now
  **assign those axes the same block size as the reduction tile.** This costs ~0 on the
  curriculum and leaves the door open to a measured 5–7% gain later.

> **Priority for this work: clean, simple code > squeezing out perf.** The goal here is
> *simplification* — fewer facts, fewer branches, fewer magic constants. The ~5–7% split gain
> is explicitly being left on the table in favor of a simpler heuristic. Don't re-introduce
> complexity to chase small perf; prefer the cleaner structure even at a few % on
> out-of-curriculum shapes. The hard floor is "no catastrophic (cliff) regression" — that one
> is non-negotiable, the rest is a clarity-vs-perf trade you should resolve toward clarity.
>
> **Verify every perf claim empirically with a SEPARATE subagent on the GPU** — do not trust
> this doc's numbers (or your own reasoning) for the accept/reject decision. The findings below
> were measured, but the budget value and any new config you emit must be re-measured by an
> independent context (a results-referee-style subagent: its own command, ≥3 launches, fixed
> seed, accuracy on, pinned idle GPU, median clearly beats noise). See "Verification owed".

---

## ⚠️ Corrected mechanism (overturns both the logs and an intuitive read)

A natural (and previously-stated) explanation was: *"welford has 2-D tile accumulators while
softmax has single values, so the perf cliff differs."* **This is false — verified against
source + GPU:**

- welford accumulators (`examples/welford.py:46-48`): `acc_cnt/acc_mean/acc_m2 =
  torch.zeros_like(x[tile_m, 0])` → shape **`[M_BLOCK]`**, scalar-per-row.
- softmax accumulators (`examples/softmax.py:81-82`): `mi/di` → shape **`[M_BLOCK]`**,
  scalar-per-row. **Identical dimensionality.** welford carries 3 such scalars, softmax 2.
- `ReductionFact.num_carried_accumulators` (the count of 2-D `[M_BLOCK,R_BLOCK]` carried
  tiles) is **0 for welford** — that field detects jsd(=2)/kl_div(=1) **Band-B**, NOT welford.
  Keying welford cliff-protection on it would NOT fire → cliff re-opens. Do not do that.

**The real driver (proven by an M_BLOCK control):** the same softmax tile `[16384,16384]`
costs **25733µs at M_BLOCK=16 (a 4.34× cliff)** but **6354µs at M_BLOCK=1 (the *best* config)**.
If accumulator size or `tile_N` alone drove the cliff, M_BLOCK=1 would cliff too — it doesn't.
The cliff is the **loaded data working-tile `M_BLOCK × tile_N × itemsize` spilling out of
registers/SMEM.** It is kernel-structure-agnostic.

**This cliff is independently corroborated by the run's own history (ledger
`run3.gate_verdicts[18-19]`, the EDIT#4 episode).** EDIT#4 raised the welford apply cap
`STRUCTURED_APPLY_LOOP_CHUNK_BYTES` 8192→16384 and was **results-referee REJECTED**: a
*single-constant* apply-tile change caused a **catastrophic 4–7× regression on welford
`(262144,5120)`** — `gate_verdicts[19].CRITICAL` records `apply 2048→4096 there: 4612us→33571us
(~7.3× slower)`. That is the same footprint cliff, found by a different route a day earlier. Two
takeaways the implementing agent should carry: (1) the cliff is real, large, and lives exactly
where the footprint argument predicts (high-M, large tile); (2) **a one-constant cap change is
NOT automatically "Band-C-safe" — it moves the tile on EVERY structured shape, including the
curriculum-extreme high-M ones a 3-shape A/B misses.** That lesson is *why* the footprint sweep
below must cover the canaries, not just train.

---

## (A) Settled findings (proven this conversation)

1. **The "widen the apply axis off floor=1" justification is a kernel BUG, not a real need.**
   The structured branch widened a second tile because plain-T2 floors non-reduction axes to
   1 and flooring welford's apply axis to 1 was catastrophic. Per the user: that catastrophe
   was a kernel bug and **should be completely ignored.** So this is no longer a reason to keep
   `is_structured_combine`.

2. **`is_structured_combine` goes dead automatically once welford shares one tile.** The gate
   (`device_ir.py` ~L1003) is `len(non_grid_tiles) > 1 and len(apply_tiles) >= 1`. It is True
   *only* because welford registers **two independent** `hl.tile(n)` (welford.py:50 & 76).
   softmax_two_pass registers **one** shared `register_block_size` for both loops →
   `non_grid_tiles == 1` → gate False. Welford is the **sole** `is_structured_combine=True`
   kernel in the 259-shape facts; all others already False. So sharing one tile makes welford
   fall through to plain-T2 with no special-casing.

3. **Separate tile sizes CAN avoid a spill cliff — but a smaller *shared* tile recovers it
   equally well.** On all high-M (cliffed) softmax shapes tested, the best shared tile was
   within ≤0.07% of the best split. Sharing-then-shrinking escapes the cliff; the split is not
   required for cliff-avoidance.

4. **A hypothetical split-softmax does NOT get a welford-like benefit from the split — proving
   the effect is not `is_structured_combine`.** We hand-wrote `softmax_split` (softmax_two_pass
   with two independent tiles) and ran the full combine×apply grid:
   - At N≤4096: 0/7 split-wins; the tile never gets big enough to matter.
   - At high-M **wide-N**: softmax **cliffs too** (4.34×, 5.2×) — same generic spill — and a
     smaller shared tile recovers it. So softmax both *cliffs* and *recovers-by-sharing* just
     like welford; the reduce-then-apply structure is irrelevant to the cliff.

5. **Separate tile sizes DO buy ~5–7% on certain OUT-OF-CURRICULUM shapes (a real but small,
   non-cliff effect).** welford `(262144,4096)`: best split `[16,4096,2048]`=3801µs strictly
   beats *every* shared tile (best shared `[16,2048,2048]`=4111µs) by **7.5%** — combine wants
   the wide 4096 tile, apply needs 2048. split-softmax `(131072,16384)` M_BLOCK=1: split beats
   best shared by **4.9%**. Same direction, smaller magnitude. **This is a separate phenomenon
   from the cliff** (it appears in the *no-cliff* regime). It is OK to forego it for now.

6. **On the welford CURRICULUM, sharing one tile loses nothing.** A matched A/B over 11 welford
   shapes (`wf_shared_vs_split_ab.py`): 0/11 SPLIT-NEEDED; a single shared tile is within 3% of
   the heuristic's split everywhere, and *beats* it 26–29% at wide-N (the existing `apply=2048`
   clamp is over-conservative for low-M trained shapes). The split-wins regime
   (`(262144,4096)`) is not a curriculum shape.

> Cross-check against the run's prior conclusions: the notebook/ledger had concluded (from
> *apply-only* sweeps + the combine×apply grid, e.g. `run3.results[6]`, the EDIT#4 work) that
> "the split is necessary." That earlier conclusion is **narrower than it sounds** — it was
> measured by sweeping one tile while holding the other, never by collapsing both to a single
> shared tile. The shared-vs-split A/B (finding 6) and the M_BLOCK control (corrected mechanism)
> are NEW measurements this conversation; they refine the prior conclusion to: the split is only
> necessary for a small (~5–7%) off-curriculum gain, NOT for cliff-avoidance (a shared tile +
> footprint cap handles the cliff). If you re-read the older logs, weight these newer
> controls over the older "split necessary" framing.

---

## (B) Design implication

`is_structured_combine` was conflating **two independent jobs**. Separate them:

- **Job 1 — "widen a separate apply axis off floor=1":** moot (the kernel bug above). Dead.
- **Job 2 — "avoid the spill cliff":** real, but it's a **generic footprint** concern
  (`M_BLOCK × tile × itemsize`), belongs on the **whole T2 path**, and must NOT be keyed on
  the reduce-then-apply structure. The existing plain-T2 persist cap
  (`MULTILOAD_PERSIST_MAX_BYTES`, keyed on **row bytes** = `size_hint × itemsize`) is **NOT
  sufficient** — it is not M_BLOCK-aware (at `(262144,5120)` row-bytes=20KB ≪ 240KiB, wouldn't
  fire, yet `16×8192×4`=512KB cliffs). A new footprint-keyed cap is genuinely needed.

Nothing is left that `is_structured_combine` uniquely earns → it's deletable.

### ⚠️ The one critical ordering constraint for whatever implementation you choose

"Assign the coupled (apply) tiles the same block size as the reduction tile" is correct —
**but only if the reduction tile is footprint-capped FIRST, then the coupled tiles set equal to
that capped value.** Otherwise `apply := combine = 8192` at `(262144,5120)` M_BLOCK=16 *is* the
6.6× cliff config `[16,8192,8192]` (35124µs). Cap combine→2048 by footprint first, then share →
`[16,2048,2048]` (5226µs, ties the current split). The footprint cap is the thing that makes
"share" safe.

---

## Proposed implementation (NON-BINDING — you may choose differently)

Data structure (`helion/autotuner/config_spec.py`, `ReductionFact`):
- **Delete** `is_structured_combine: bool`.
- **Rename** `apply_block_ids` → `coupled_tile_block_ids` (or a better name encoding "tile axes
  spanning the reduction extent, currently sized = the reduction tile; decouple later for the
  5–7%"). Keep it computed exactly as today (non-grid tiles carrying no `ReductionLowering`).
- `static_rnumel`: its ONLY consumer is the structured branch (`triton.py:479`, `n_valid`).
  Once that branch goes it is **consumerless/dead** → recommend dropping it too (fact-integrity
  would flag it). Flagged as a separate decision, not bundled.

Algorithm:
- `device_ir.py register_user_tiled_reductions`: drop the `is_structured_combine` routing+bool;
  keep computing + storing `coupled_tile_block_ids`; relax the `len(non_grid_tiles) != 1`
  early-return so a fact registers with 1-or-more non-grid tiles (extras → coupled). Keep the
  "each coupled tile spans the reduction extent with a resolvable static size" guard (now
  load-bearing: we assign those axes a reduction-extent-sized block). Update both
  `ReductionFact(...)` build sites (T1 builder + T2 builder).
- `triton.py get_seed_config`: **delete the `if fact.is_structured_combine:` branch** (~40 lines)
  and the 3 constants `STRUCTURED_COMBINE_CAP_BYTES` / `STRUCTURED_APPLY_PERSIST_MAX_BYTES` /
  `STRUCTURED_APPLY_LOOP_CHUNK_BYTES`. On the unified T2 path: (1) add a footprint cap
  `r_block = min(r_block, np2(FOOTPRINT_BUDGET_BYTES // (itemsize × M_BLOCK)))` where
  `M_BLOCK = prod of _block_floor over fact.m_block_ids`; (2) call the EXISTING
  `_build_block_sizes(..., apply_ids=set(fact.coupled_tile_block_ids), apply_value=r_block)` —
  the plumbing already exists; just pass the capped `r_block` as the coupled value.
- Net: **−1 fact (`is_structured_combine`), −2 constants** (3 structured → 1 footprint),
  **−~40-line branch**, possibly **−1 more field (`static_rnumel`)**. The surviving footprint
  cap is strictly more general (guards softmax/kl_div/jsd at high-M for free).

Tests: update as needed — you'll find what references the dropped/renamed fields (the
heuristic + fact tests live in `test/test_autotuner_heuristics.py`; there are existing
`is_structured_combine` asserts that will need to change once the field is gone). Use your
judgment on coverage; not over-specifying here.

PR branch (`reduction-seed-heuristic-run2`, the `_TritonReductionSeedBase` split): mirror via
the AGENT_HANDOFF procedure; not edited directly.

---

## Verification still OWED (do not skip — this is config-CHANGING, not config-preserving)

**All perf checks below MUST be run by a SEPARATE subagent on a pinned idle GPU** (its own
command, ≥3 launches, fixed seed, accuracy gate on, median clearly beats noise) — not by the
context that wrote the heuristic. This is the run's load-bearing discipline (acceptance gated by
a context separate from the producer). Do not accept a budget value or a "no regression" claim
on reasoning alone.

1. **`FOOTPRINT_BUDGET_BYTES` is the one unverified unknown — pin it with a GPU sweep
   (separate subagent).** Data brackets it (128KB fine / 256KB borderline / 512KB cliff at
   M_BLOCK=16), but it must ALSO not over-constrain the M_BLOCK=1 welford train shapes (which
   want large tiles). Sweep it: emit across the welford train curriculum + the `(262144,5120)`
   & `(262144,7168)` canaries; require (a) no shape cliffs and (b) no curriculum regression vs
   the current structured heuristic. This sweep IS the proof that "one cap replaces three."
   **Not yet run.** (The EDIT#4 reject — ledger `run3.gate_verdicts[19]` — is the standing
   reminder that this sweep must include the high-M canaries, not just the 3-shape train A/B
   that missed the cliff the first time.)
2. **259-shape config diff** (`_lab/AGENT_HANDOFF_verify_and_push.md`): confirm ONLY welford
   rows change; softmax/kl_div/jsd/T1 byte-identical (proves the reroute didn't perturb them).
3. **Eviction shift to check:** the structured branch ALWAYS emitted reread eviction; plain-T2
   emits it only `if row_reread and not persistent`. Welford shapes that go persistent under
   the footprint cap would DROP an eviction policy the structured path set. Measure whether
   that costs anything (likely noise; verify).
4. **Gates** (welford is out-of-scope for *this run's parity* but IS in the PR deliverable, so
   it matters there): fact-integrity (renamed + dropped fields), results-referee (welford
   no-regression incl. canaries), fresh-oracle re-validation.

---

## Artifacts (raw data + repro)

- `_lab/harness/wf_shared_vs_split_ab.py` — welford best-shared vs heuristic-split (0/11 split-needed).
- `_lab/harness/wf_split_grid_ab.py` — full combine×apply grid; `(262144,4096)` split-wins 7.5%.
- `_lab/harness/wf_force_share_vs_heuristic.py` — apply:=combine: +16% train geomean but 4–7× canary cliff (shows naive share is wrong; footprint cap needed).
- `_lab/harness/softmax_split_tile_ab.py` — hand-written split-softmax; 0/7 at N≤4096.
- `_lab/harness/softmax_cliff_mechanism.py` — the M_BLOCK control proving footprint (not accumulator) drives the cliff; raw at `/tmp/softmax_cliff.out`.
