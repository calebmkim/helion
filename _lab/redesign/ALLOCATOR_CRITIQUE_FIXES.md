# Budget-allocator critique + fix plan (post-autonomous-run review)

> Recorded after the human reviewed the committed budget allocator and raised a series of
> well-founded critiques. This is a HANDOFF doc: it states each problem, whether it stands, how to
> reproduce/see it, and the proposed fix — so a fresh-context agent (or an in-context fix pass) can
> execute. The autonomous run is at HEAD `20011b63` on branch `reduction-redesign`; the function
> under review is `_TritonReductionSeedBase.size_reduction_tiles` in
> `helion/_compiler/autotuner_heuristics/triton.py`.

## Context / where things stand
- The reduction seed was rewritten into ONE per-co-residency-group budget allocator
  (`size_reduction_tiles`, triton.py ~line 838). All recognizers (`_is_per_feature_accumulator`,
  `_grad_collapse_group`, `_carried_*`, `_resident_tile_cap`, etc.) are deleted. Gates green
  (validate_kernel_fact 460/460, probe_assertions 13/13, matmul-epilogue, unit tests 41p). Final
  bench: geomean 0.894 (net faster), 6 wins, 3 regressions (all grpo, ≤1.20x).
- The human's review found the allocator is **principled in intent but patched with structural
  gates** that make it read as unprincipled AND introduce a real double-count. The core insight
  (below) is that several complaints are ONE bug.
- Interpreter `/home/dev/helion/.venv/bin/python`; `HELION_AUTOTUNE_EFFORT=none`; `PYTHONPATH=`
  the worktree. Recorder/validators need no GPU; perf benching is single-process foreground only
  (NEVER detached — project hard rule). Tools: `_lab/harness/unified_config_recorder.py`,
  `_lab/unify/replay_bench.py`, `_lab/redesign/{fact_dump,validate_kernel_fact,probe_assertions}.py`.
- A pre-edit config snapshot is at `/tmp/before_rewrite.json` (regenerate if stale: recorder at the
  pre-rewrite HEAD `34ae072e`). The frozen original-heuristic reference is
  `_lab/unify/baseline_fc1dbaa0_configs.json`.

---

## THE CORE FINDING (complaints A, B, C are one bug; F is RELATED but SEPARATE — see its entry)

The allocator approximates the resident set as `num_live × ∏(one tile shape) × itemsize` and then
patches that approximation with structural gates (`carried_mult` gated on `feature_footprint==1`;
the `in_accumulator(gbid)` grid term; the `widen_live` ternary). Each gate is keyed on a *faithful*
property (none is a banned kernel-identity recognizer), but threading them as multiplier-patches
inside one formula is what reads as unprincipled — and the `num_live × carried_mult` overlap is a
genuine DOUBLE-COUNT (jsd's `body_live_tiles=12` already includes its 2 carried buffers; multiplying
by `carried_mult=2` double-counts; R=2048 lands by accident and the `feature_footprint==1` gate
exists only to stop the double-count from also firing on the m-collapse case).

**THE UNIFYING FIX (applies to the CARRIED regime only — see regime note below):** stop summarizing
residency with `num_live × one-shape`. Instead walk `spec.accumulator_facts` and sum the ACTUAL
loop-carried tiles by their ACTUAL `dim_block_ids`:
`group_footprint = Σ over distinct resident carried buffers (∏ of that buffer's tiled dims) × itemsize`
(ADDITIVE across separate buffers, §2.3 / CARRIED_AND_GREEDY #3). The accumulator facts already
carry per-tile `dim_block_ids` + `itemsize` — the true shapes. This single change:
- deletes `carried_mult` and the `feature_footprint==1` gate (A);
- folds the materialized-feature term (B) into the same walk;
- replaces the lossy `in_accumulator`-multiplied grid term with the real Σ (C);
- gives the softmax-vs-grpo distinction for free (rank of the carry: scalar `[m]` vs 2-D `[g0,g1]`);
- FIXES grpo (its `[g0,g1]` carry + `[grid,R]` working tile are SEPARATE tensors that ADD — the
  multiplicative ∏ can't express that, the additive Σ can).

**DOES NOT delete `num_live` or the `widen_live` ternary — that was an error in an earlier draft.**
F (the `num_live`/`widen_live` issue) is a SEPARATE bug in the STREAMED regime; `num_live` is a real,
load-bearing footprint term there and is NOT recoverable from `accumulator_facts`. See F's entry.
VERIFIED: fused_linear_jsd has `blt=7` but ALL 40 of its accumulators are 1-D `[0,None]` (ZERO wide
carried tiles) — its 7 live tiles are TRANSIENT compute intermediates (the JSD math), invisible to an
accumulator walk. Drop `num_live` and it widens its grid and SPILLS ~2x (this WAS the step-13
regression). So the streamed regime KEEPS `num_live`.

**REGIME NOTE — the Σ-walk is NOT always needed; it is the CARRIED-regime footprint.** Two regimes,
selected by "is there a >=2-D loop-carried accumulator holding a reduction axis"
(`_is_carried_reduction_tile`: `len(dims)>=2 and any(d in reduction_ids)`):
- **NO >=2-D carried reduction tile → STREAMED / PERSISTENT regime.** No carried-tile walk needed; the
  resident set is just the single `[M_BLOCK, R]` row (+ scalar accumulators), sized by the
  persistence / looped-chunk logic. **softmax falls HERE** even though it HAS loop-carried
  accumulators — they are 1-D scalars (`[m_tile]`, the running max/sum, ACC dims `[0]`), NOT a wide
  `[M,R]` tile, and its reduced dim (512) is in NO accumulator (streamed, re-read across the 2
  passes). So a 1-D carry is treated like a persistent reduction's scalar result: the Σ-walk over its
  accumulators contributes only the scalar (cheap) term and never binds. Do NOT give softmax a third
  regime — the rank test (>=2-D) routes it to streamed automatically, and it's size-independent (keys
  on RANK, not on "m is small"). VERIFIED: softmax (16384,512) accumulators are all `[0]` (1-D), the
  reduction bid1 is in NO accumulator.
- **HAS a >=2-D carried reduction tile → CARRIED regime.** kl_div `[grid,R]`, jsd `[grid,R]`×2,
  grpo `[g0,g1]`. Use the additive Σ-over-buffers footprint above.
The streamed regime is the common case (rms_norm/sum/softmax/cross_entropy/most transfer); the
carried regime is kl_div/jsd/grpo. The human explicitly endorsed this TWO-PATH structure.

Implementation note: a kernel is in the "carried / looped" regime iff it has a >=2-D loop-carried
accumulator holding a reduction axis (`_is_carried_reduction_tile`: `len(dims)>=2 and any(d in
reduction_ids)`). The human explicitly endorsed a TWO-PATH structure keyed on this (looped-carried
vs streamed) — it is a faithful structural property, NOT a recognizer, and §2.5/§3 of the design
doc bless "two named byte-caps with a faithful reason." Prefer the explicit two-regime form over
one formula threaded with gates.

### WHY the regime carries PERSISTENCE-eligibility (the second job of the regime tag, beyond footprint)
The persistence gate (`ext_held`, triton.py ~1028) requires `carried_2d_count == 0`. PHYSICAL reason:
a persistent reduction holds the FULL row `[M_BLOCK, R]` in ONE tile (reduced to a scalar result); a
carried-2-D reduction holds `[M_BLOCK, R_chunk] × num_carried` LIVE every iteration, so making R full
would blow that to `full_R × num_carried` resident — a spill. So a >=2-D carried tile STRUCTURALLY
forbids persistence; only the streamed regime runs the persistence gate. THIS is the general reason
the regime matters beyond the footprint number.
CAVEAT (don't overstate): for softmax SPECIFICALLY this is numerically MOOT — its row is 512 (tiny)
and its carry is 1-D scalar, so even a carried budget would admit the full 512. softmax reaches
full-extent r_block=512 regardless of regime. The persistence-eligibility distinction BITES on a
kernel with a scalar carry AND a wide row that must be held persistent (the wide-rms_norm family),
not on softmax. So: regime decides persistence-eligibility in GENERAL (carried ⟹ never persistent),
and the footprint number is regime-invariant for a 1-D carry — both true; softmax just doesn't
exercise the persistence half. (NB the user-tiled track has no `reduction_loops=[None]` knob;
"persistent" there means rdim block_size == full extent, which softmax achieves at 512.)

### ORDERING IS LOAD-BEARING — do NOT collapse the two passes (reduction-then-grid)
The allocator sizes REDUCTIONS FIRST (against the grid axis at its provisional FLOOR, usually 1),
THEN widens the grid into the REMAINING budget (§2.3 "earlier bids held fixed as inputs to later
sizing"). This is INTENTIONAL and the budget is enforced at the WIDEN step: `byte_widen =
ROW_PERSIST / (r_block × itemsize × ...)` divides by the now-FIXED r_block and takes only what's
left, so the post-widen product `grid_M × r_block × itemsize` is bounded BY CONSTRUCTION. VERIFIED on
rms_norm: (8192,768) r=1024,grid=4 → 16384; (2048,16384) r=16384,grid=1 → 65536; (32768,2048)
r=2048,grid=8 → 65536 — all <= ROW_PERSIST=245760. The two passes are consistent: a LARGE r_block ⇒
`byte_widen` floors the grid to 1 (no widen) ⇒ the "grid=1" assumption the reduction made was correct;
a SMALL r_block ⇒ the grid widens, and the widen's own byte cap keeps the product in budget. So the
reduction sizing against grid=floor is NOT "a calculation done with a too-small grid" — it is the
sequential greedy allocation, and floor-vs-widen FALLS OUT of it. A fixer must NOT "fix" this into a
single simultaneous pass — that would break the floor-vs-widen fall-out. (The footprint RE-derivation
inside the widen step is where the lossy `num_live × one-shape` lives; the CORE FIX makes the
reduction pass AND the widen pass compute the footprint the same correct way, so they stay consistent
without the gates.)

---

## Per-complaint table

### A. `num_live` vs `carried_mult` — double-count.  **STANDS.**
- WHERE: triton.py ~989-999, `carried_mult = 1; if feature_footprint == 1: carried_mult = sum(...)`,
  then `return itemsize * _num_live * carried_mult * prod`.
- WHY IT STANDS: `num_live = max(body_live_tiles)` already counts the carried buffers; `carried_mult`
  re-counts them. The `feature_footprint==1` gate is a smell — it exists only to suppress the
  double-count on the m-collapse path.
- SEE IT: `fact_dump.py --corpus curriculum --kernels jsd,kl_div` — jsd has 2 `[1,0]` carried tiles
  (carried_2d_count=2, blt=12) → current lands R=2048 via the double-count; kl_div 1 tile (blt=6) →
  8192. Both happen to be ~right, but for the wrong reason.
- FIX: the Σ-over-accumulators footprint (CORE FINDING). jsd → `16384/(2 buffers×4)=2048`, kl_div →
  `16384/(1×4)=4096`-class — both fall out of additive Σ with NO num_live multiplier and NO gate.

### B. `feature_footprint` computed separately.  **STANDS (cleanup).**
- WHERE: triton.py ~897-903 (hoisted try/except block) + used at ~960 (`prod = feature_footprint`).
- WHY: materialized feature axes are just MORE resident tiles in the group's working set; no
  principled reason they're a separate hoisted term rather than another factor in the resident-tile
  walk. (Only reason it's hoisted: `_materialized_feature_axes` needs env/device_ir, wrapped in
  try/except for the no-env unit tests.)
- FIX: iterate materialized-feature axes INSIDE the footprint walk (subsumed by the CORE FINDING).
  Keep the no-env guard.

### C. Why only count a grid axis `if in_accumulator(gbid)`?  **GATE CORRECT; model lossy → subsumed by A.**
- WHERE: triton.py ~968-979.
- VERDICT: the gate IS faithful — "appears in a loop-carried accumulator" is the right proxy for "is
  this grid axis a RESIDENT row" (rms_norm `[grid_M,None]`, grpo `[g0,g1]` → resident → count; a pure
  parallel fan-out axis in no accumulator → don't). Do NOT change it to "shares the SAME accumulator
  as the sized axis" — that was TRIED and blew grpo up 8.8x (grpo's R is reduced AGAINST the grid
  rows but isn't in the same buffer).
- THE REAL ISSUE: multiplying is an over-estimate when the tensors are SEPARATE (grpo's `[g0,g1]`
  accumulator vs the transient `[grid,R]` working tile) → the residual grpo 1.2x. The additive Σ
  footprint (CORE FINDING) models this correctly (two tensors ADD). So C is fixed by A, not by
  touching the gate.

### D. The `if`/`if`/`elif` emission block + `!= pd.block_id`.  **NOT A BUG; legibility STANDS.**
- WHERE: triton.py ~1042-1060.
- IT IS CORRECT and NOT collapsible to if/elif/elif: there are TWO orthogonal jobs.
  - JOB A (`if d.block_id == pd.block_id`, ~1042): record the PRIMARY's scalar levers
    (`primary_r_block`/`persistent`) for num_warps + the standard track's reduction_loops emission.
  - JOB B (emission routing, the `if d.block_id in valid` / `elif ... reduction_loops.valid`):
    WHERE the size lands — block_sizes slot (user-tiled) vs reduction_loops knob (rolled non-primary).
  - A user-tiled primary hits BOTH (records lever AND routes to red_values) → not mutually exclusive.
  - `!= pd.block_id` in the elif excludes the ROLLED PRIMARY (which is `not in valid` because rolled,
    so it would otherwise fall into the elif and be double-routed; its size is emitted via
    primary_r_block instead). `not in valid` is already implied by being in the elif, so `!= pd` is
    the necessary+sufficient extra guard.
- CAVEAT: the entire `elif` (rolled_loop_sizes) is CORPUS-DARK — zero corpus kernels have
  `len(reduction_loops) > 1`. Verify: bind every corpus kernel, none has >1 reduction_loops.
- FIX: comment naming JOB A vs JOB B; note the elif is corpus-dark (for the relaxed >=1 multi-rolled
  gate only). No logic change.

### E. `GRID_TILE → seated=1; continue`.  **NOT a behavioral bug; the `=1` is a PROVISIONAL seat that gets occupancy-lifted. Legibility STANDS.**
- WHERE: triton.py ~1014-1017 (reduction-seating loop) AND ~1062-1125 (grid-M widen loop).
- KEY FACT (verified): jsd's GRID_TILE bid1 IS ALSO a grid axis (`grid_axis_block_ids=(1,)`,
  `in_grid_axis=True`) and co-holds the carried 2-D tile (`ACC dims=[1,0]`). So bid1 is visited TWICE:
  (1) the reduction loop sets `seated[bid1]=1` PROVISIONALLY, then (2) the grid-M widen loop sizes it
  via `max(floor, min(byte, occ, ...))` — i.e. it IS occupancy-lifted, exactly like a non-reduction
  grid axis. The `=1` is overwritten, not final.
- So the earlier framing "asserted not derived / skips the budget" was WRONG for any GRID_TILE that is
  also a grid axis (all corpus cases — jsd). The `=1` is the final value ONLY for a GRID_TILE whose
  block_id is NOT in `grid_axis_block_ids` (a pure grid-parallelized reduction with no separate
  tunable grid sibling) — which the corpus does not contain.
- WHY NOT occupancy-lift the GRID_TILE branch directly: it already happens, in the second visit. The
  reduction-loop `=1` just needs to be a sane provisional (it is). Sizing it via remaining-budget in
  the reduction loop instead would be redundant with the grid-M widen.
- FIX: NONE behavioral. LEGIBILITY: comment that `seated[bid]=1` here is provisional and that an axis
  which is also a grid axis is sized (occupancy-lifted) in the grid-M loop below; only a non-grid-axis
  GRID_TILE keeps `=1`. SEE IT: `fact_dump.py --corpus curriculum --kernels jsd` → bid1 is `cat=
  grid_tile` AND in `grid_axis_block_ids`.

### F. `widen_live = 1 if is_reduce_then_apply else num_live` / "num_live over-counts".  **STANDS — but SEPARATE from A, NOT subsumed. `num_live` STAYS; do NOT delete it.**
- WHERE: triton.py ~1092-1093 (the grid-WIDEN footprint; `num_live` also enters the reduction-chunk
  byte budget and the LIVE_PERSIST persistence gate).
- WHAT `num_live` IS AND WHY IT'S LOAD-BEARING (the correction to the earlier "subsumed by A" claim):
  `num_live = body_live_tiles` = peak count of simultaneously-live rdim-shaped tiles in the body. It
  is NOT recoverable from `accumulator_facts` — a heavy body's live tiles are TRANSIENT COMPUTE
  intermediates, not accumulators. VERIFIED: fused_linear_jsd (8192,32000) has blt=7 but ALL 40
  accumulators are 1-D `[0,None]` (ZERO wide carried tiles); its 7 live tiles are the JSD-math
  intermediates. It is in the STREAMED regime (no wide carried tile), and it NEEDS num_live in the
  grid-widen footprint: dropping num_live widens its grid and SPILLS ~2x (the step-13 regression).
  dynamic_quant / gated_rmsnorm / cross_entropy are the same shape. So the Σ-over-accumulators walk
  CANNOT replace num_live — they measure different things (carried accumulators vs transient live
  tiles). The streamed regime KEEPS num_live.
- THE ACTUAL F-BUG (subtle): `body_live_tiles` is shape-BLIND — it counts ALL live tiles, not just
  the WIDE `[M,R]`-shaped ones. welford's blt=3 = one wide `[M,R]` row + two SCALAR `[M]` carries; I
  multiply the WIDE footprint by 3, over-counting ~3x → grid wrongly 4→2 (~1.2x). fused_linear_jsd's
  blt=7 ARE genuinely wide → ×7 is correct. So the bug is "num_live conflates wide-tile count with
  total-live-tile count," and `is_reduce_then_apply` (has a non-reduction loop) is a COINCIDENTAL
  PROXY for "this kernel's live-tile count is inflated by a scalar/apply pass" — works on the corpus,
  not the true cause.
- SEE IT: `fact_dump.py --corpus curriculum --kernels welford` → accumulators `[0],[0],[0]` (scalars)
  + `[0,None]`; blt=3. vs fused_linear_jsd blt=7, accumulators all `[0,None]`, NO normalize loop.
- FIX OPTIONS (none free; pick one, do NOT just delete num_live):
  (a) SIMPLEST / what works today: KEEP `num_live` in the streamed budget, KEEP the `widen_live`
      ternary, but RENAME/comment it as an explicit, named heuristic — "drop the body-weight multiplier
      for a reduce-then-apply kernel, whose wide tile lives in the separate apply pass" — honest about
      being a proxy. No behavior change; just stop pretending it's principled.
  (b) PRINCIPLED but a Stage-1 change (out of allocator scope; the brief discouraged device_ir edits
      unless necessary): add a fact "peak count of RDIM-WIDE live tiles" distinct from the
      shape-blind `body_live_tiles`, and use THAT as the multiplier. Then welford→1, fused_linear_jsd
      →7 fall out with no `is_reduce_then_apply` gate. Flag it; don't do it without sign-off.
- RECOMMENDATION: ship (a) (named heuristic) in this pass; note (b) as a follow-up. Do NOT conflate
  with A — A is the carried-regime footprint; F is the streamed-regime body-weight term.

### G. Unpinned FULL_GRID grid axis gets a bad config.  **STANDS (latent, corpus-dark).**
- WHY: per_token_group's 128 axis is FULL_GRID but PINNED (`pinned=True`, FixedBlockSizeSource, no
  tunable slot) so the allocator skips it (`if mbid not in valid: continue`). If that axis were
  UNPINNED (tunable), it is still categorized FULL_GRID but would go through the resident-row WIDEN
  path (occupancy/byte/WIDEN_MAX) instead of being held at its full extent — a potential bad seed.
- SEE IT: `fact_dump.py --corpus vllm --kernels per_token_group_fp8_quant` → `cat=full_grid
  pinned=True`. No corpus kernel has an UNPINNED full_grid axis, so this is latent (would need a
  synthetic kernel to trigger).
- FIX: in the grid-M loop, a FULL_GRID grid axis should seat at its FULL EXTENT (full-extent-resident
  by definition, like FULL_SLICE), not go through the widen heuristic. ~1 line, uses the Stage-1
  category already computed. No-op on the corpus; removes a latent footgun.

### H. Grid-widen MAGNITUDE for per_token_group's groups_per_row (the FULL_GRID grid SIBLING).
###    **STANDS, but it is OCCUPANCY-tunable — NOT incapable, and NOT a regression.**
- WHERE: the resident-row widen path, triton.py ~1080-1123 (`byte_widen`, `occ_widen`,
  `rows_ceiling`). The binding cap for this axis is `occ_widen = pp2(grid_rows/(num_sm*MIN_WAVES))`.
- IMPORTANT CORRECTION (measured this session): earlier I claimed "no formula captures it, leave it to
  the autotuner." That was TOO PESSIMISTIC. The cost is a U-shape with a FLAT VALLEY (~10% over a
  4-8x span of the widen); you don't need to nail it, just land in the valley. Occupancy DOES bind;
  `MIN_WAVES=8` is just set too LOW so it over-collapses on one shape.
- MEASURED (single-process median-of-9, /tmp/ptg_sweep.py):
    (8192,4096) groups=32 BEST=g8   (g2=2.00, g4=1.08, g8=1.00, g16=1.02, g32=1.05)  waves@opt≈248
    (8192,7168) groups=56 BEST=g64  (g8=1.15, g32=1.08, g64=1.00)                    waves@opt≈54
    (2048,8192) groups=64 BEST=g8   (g4=1.04, g8=1.00, g32=1.09, g64=1.51)           waves@opt≈124
    (128, 4096) groups=32 BEST=g2   (g2=1.00, g4=1.00, g8=1.07, g16=1.29)            waves@opt≈15
  Current MIN_WAVES=8 mis-fires ONLY on (2048,8192): occ=64→widen 64 (1.51x; over-collapses to ~15
  waves; optimum ~124). Raising the grid-widen floor to ~MIN_WAVES=32 gives that cell occ=16 (~1.02x)
  while keeping the others in-valley. (Risk: the tiny (128,4096) shape — only ~31 waves at floor — may
  under-widen to 1; optimum is g2, ~1.0-1.1x, minor.)
- NOT A REGRESSION: VERIFIED byte-identical to the original heuristic fc1dbaa0 — fc1dbaa0's
  `_m_axis_occupancy_cap` is the SAME `prev_pow2(grid_rows/(num_sm*MIN_WAVES))` and produces the SAME
  mis-fire (recorded configs identical: [2],[2],[64],[64]). Both versions inherit it.
- FIX: a MINOR HILL-CLIMB on ONE constant (§3/§4) — not a redesign. A/B the grid-widen occupancy
  floor `MIN_WAVES ∈ {8,16,32,64}` over the cells it moves: the 4 per_token_group shapes, plus a
  sanity check that softmax/rms_norm (which widen on the same constant but are largely shielded by
  WIDEN_MAX_ROWS) do NOT move. Pick the floor that lands the most cells in their flat valley
  (`MIN_WAVES=32` is the hand-trace candidate: fixes (2048,8192) 1.51→~1.02, keeps others in-valley,
  mild under-widen risk on the tiny (128,4096)). NO single floor is exact (optima span ~15-248 waves)
  — the goal is "most cells in-valley," accepting this is good-budget not perfect-formula. The widen
  MAGNITUDE has no closed form and the byte/residency budget cannot bind here (each widen step adds
  only group_size=128 elems, far under any spill), so occupancy-floor is the only lever.
  CONSIDER a SEPARATE constant for the grid-WIDEN floor vs the general occupancy floor: `MIN_WAVES`
  is reused as the base occupancy floor elsewhere, so bumping it globally may perturb other kernels —
  prefer a distinct `WIDEN_MIN_WAVES` (or similar) so the tune is scoped to the widen and can't leak.
  After landing: one full recorder re-run + bench to confirm no collateral movement.
- NB: E does NOT fix H — different axis (groups_per_row has NO reduction over it; its only reduction
  is the pinned FULL_GRID group_size), different code path (grid-M widen, not the GRID_TILE branch),
  and budget-DEPLETION would over-widen it (byte budget admits k<=480). H is an OCCUPANCY-floor tune.

### (resolved) softmax-2pass vs grpo differentiation.  **NOT STANDING.**
- The faithful differentiator already exists: the RANK of the loop-carried accumulator. softmax's
  carry is a 1-D scalar `[m]` (cheap, streamed regime); grpo/kl_div carry a 2-D `[..]` tile (carried
  regime). `_is_carried_reduction_tile` (`len(dims)>=2 and holds an rdim`) is exactly this test, and
  it is size-independent (keys on rank, not on "m is small"). The Σ-over-accumulators footprint uses
  it directly.

---

## Suggested execution order (for the fixer)
1. **CORE FIX (A+B+C):** replace the `num_live × carried_mult × prod` footprint with the TWO-REGIME
   structure: STREAMED/persistent (no >=2-D carried reduction tile — rms_norm, softmax, sum,
   cross_entropy, fused_linear_jsd, most transfer) vs CARRIED (kl_div/jsd/grpo). In the carried regime
   use the additive Σ-over-`accumulator_facts` group_footprint (∏ within a buffer, Σ across buffers;
   classify dims by membership; fold in materialized features). softmax STAYS in the streamed regime
   (its carry is 1-D) — the rank test routes it automatically. Deletes `carried_mult` and the
   `feature_footprint==1` gate, and replaces the lossy `in_accumulator`-multiplied grid term with the
   real Σ. **KEEPS `num_live` in the STREAMED regime (it is load-bearing — fused_linear_jsd; see F).**
   EXPECT: jsd 2048 / kl_div ~ unchanged / welford grid 4 / grpo R correctly tight (the grpo 1.2x
   residual should improve once additive). Re-run recorder + diff vs /tmp/before_rewrite.json; gates
   (460/460, 13/13, matmul, unit); bench the movers single-process.
2. **F (streamed-regime body-weight term):** KEEP `num_live` + the `widen_live` ternary; RENAME/comment
   it as an explicit named heuristic (drop the body-weight multiplier for a reduce-then-apply kernel,
   whose wide tile is a separate apply pass). NO behavior change — just stop dressing the proxy as
   principled. (Option (b), a Stage-1 "wide-live-tile count" fact, is a flagged follow-up needing
   sign-off — do NOT do it in this pass.) Do this alongside #1 since both touch the streamed footprint.
3. **G (FULL_GRID → seat at extent):** ~1 line; latent footgun; no corpus change.
4. **D + E (legibility only, NO behavior change):** comment the emission block (JOB A vs JOB B,
   `!= pd` excludes the rolled primary, elif corpus-dark); comment that the GRID_TILE `seated=1` is a
   PROVISIONAL seat that the grid-M loop occupancy-lifts for any axis that is also a grid axis (all
   corpus cases). Do NOT change GRID_TILE logic — it's already occupancy-lifted on the 2nd visit.
5. **H (grid-widen occupancy floor):** a minor hill-climb on ONE constant — A/B a SEPARATE
   `WIDEN_MIN_WAVES ∈ {8,16,32,64}` for the widen only (not the shared `MIN_WAVES`), land the floor
   that puts the most per_token_group cells in-valley, confirm softmax/rms_norm don't move. Separate
   from #1-4; the only one needing GPU A/B beyond the standard mover re-bench.

### WHY this order
- **#1+#2 together first (both touch the streamed/carried footprint)** — #1 is the big,
  behavior-changing, principle-restoring refactor (deletes carried_mult + the feature gate, makes the
  grid term additive); #2 keeps num_live but renames the proxy. Everything else should be measured
  AGAINST this post-refactor code, not the current code (doing the small ones first would just be
  redone). #1 is the one most likely to shift configs (and ideally fix grpo), so its recorder-diff +
  bench is the main gate. CRITICAL: do NOT delete num_live (the earlier draft's mistake) — verify
  fused_linear_jsd's grid stays 1 in the post-#1 recorder diff (if it widened, num_live got dropped).
- **#3 (G) next, while in the grid-M loop** — a 1-line addition in the same loop, no corpus change,
  cheap to verify (no-op on the recorder diff).
- **#4 (D+E) are comment-only** — do them last among the no-GPU changes; they can't break a gate, so
  they don't gate anything and shouldn't interleave with behavioral work.
- **#5 (H) LAST and SEPARATE** — it's the only GPU-A/B hill-climb, it's orthogonal to the footprint
  rewrite (it tunes the grid-widen occupancy floor, a different lever), and it should be tuned on the
  FINAL post-#1-#4 code so the A/B reflects shipping behavior. Keep it in its own commit so the
  perf-tune is isolated from the structural cleanup.
- DECISION POINT after #1: if the additive footprint fixes grpo (the only standing perf regression),
  H's urgency drops to "nice-to-have." If grpo persists, H is still independent of it (grpo is a
  reduction-r_block/footprint issue; H is a grid-widen-magnitude issue) — don't conflate them.

## Gates that must stay green throughout
- config recorder (447 cells) + diff vs /tmp/before_rewrite.json — record + bench every changed cell.
- validate_kernel_fact 460/460; probe_assertions 13/13 (taxonomy routing intact); matmul-epilogue;
  test_reductions + test_autotuner_heuristics (update tests to new behavior, do NOT contort the
  allocator — the kl_div test pins a budget-derived value, the independent-loop test exercises
  size_reduction_tiles).
- ruff clean. Commit after each meaningful green step. NEVER detach a GPU job.
