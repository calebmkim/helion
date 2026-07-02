# Reduction seed — change log + suggested changes

## DONE (uncommitted, this session) — unify `persistent = (r >= ext)`

`helion/_compiler/autotuner_heuristics/triton.py`, forced-full-width branch (FULL_GRID + materialized
full-slice). Replaced `persistent = d.category is ReductionCategory.FULL_GRID` with `persistent = True`
(the axis is seated at full extent there, `r == ext`, so it's persistent under the honest uniform rule
`persistent = (r >= ext)` the normal sizing path already uses).

WHY the old `is FULL_GRID` special-case was wrong (all MEASURED this session):
- For a FULL_GRID axis `persistent` is INERT: `reduction_loops=[]` via `is_materialized`, and eviction
  short-circuits on `num_load==1` (per_token_group emits `rl=[], evict=['first']` either way).
- For a materialized grad-param full-slice it flipped the outcome only through the eviction gate
  (`pd.row_reread and not persistent`) — i.e. it was silently controlling an L2 eviction hint, not a
  persistence fact. Overloading `persistent` for that is the smell.

IMPACT (recorder diff vs committed fe91137e): exactly 10 cells, ALL grad-param (layer_norm_bwd,
rms_norm_bwd, group_norm_bwd, instance_norm_bwd), and the ONLY change is the re-read eviction hint
`['last',...] -> None`. block_sizes / reduction_loops unchanged everywhere. Measured cost of losing
that hint: layer_norm_bwd/rms_norm_bwd ~1-2.5%, group_norm/instance_norm ~neutral. Accepted as the
price of the honest rule; recoverable by S1 below. Gates green (validate 460, probe 13, pytest).

---

## PRIORITY / ORDER for the implementing agent

Do **S1, S2, S4 first**; treat **S3 as optional** (see below). Recommended order and why:

1. **S2 first** (persist-hold ceiling: category -> apply-reread). It reshapes the PERSISTENCE
   decisions (softmax wide-N, cross_entropy), which are the largest measured swings, and it builds the
   "apply-reread / single-load-vs-multi-pass" detector. Settle persistence FIRST so the later gates
   reason against final verdicts, not ones that shift underneath them.
2. **S1 second** (re-read eviction gate: `not persistent` -> pinned-tile-bytes). It DEPENDS on the
   persistence outcome (`not persistent` is exactly what it replaces), so do it after S2 when
   persistence is stable. It also recovers the grad-param ~2% the DONE `persistent=(r>=ext)` change
   gave up, closing that loop. NOTE: S1 and S2 key on the SAME underlying signal (which cache tier the
   reuse lives in / single-load-vs-reread) — keep them adjacent and SHARE the detection helper.
3. **S4 last** (both `scale *= carried_2d_count` -> 2a cross-group-max + 2b occupancy guard). Most
   LOCALIZED (jsd-only) and most INDEPENDENT (touches footprint/widen, not persistence/eviction), and
   it carries the heaviest validation burden (the ncu register-occupancy confirmation). Doing it last
   banks S1/S2 first and keeps any ncu rat-hole from blocking them.

After S1/S2/S4 land + gates green + a single recorder/probe diff pass: **only then** consider **S3**,
and only if it shows a CLEAR win. S3 is a NICE-TO-HAVE (see its section) — do NOT spend much effort:
one quick loop-width sweep on welford / rms_norm_per_block; if a tighter cap is neutral-or-better, add
it; otherwise SKIP S3 entirely. Do not let S3 delay shipping S1/S2/S4.

VALIDATION for the whole batch (run ONCE after all changes, not per-change): recorder diff vs
`/tmp/before_rewrite.json`, probe diff, `validate_kernel_fact.py` (460), `probe_assertions.py` (13),
`pytest test/test_reductions.py test/test_autotuner_heuristics.py -q`, `pytest test/test_examples.py
-k matmul_layernorm -q`, ruff. Then single-process bench the movers (jsd, softmax wide-N, cross_entropy,
grad-param) per `_lab/unify/replay_bench.py`.

---

## S1 (suggested) — faithful eviction gate: pin the re-read row iff its tile is L2-friendly

Today the standard-track re-read eviction hint is gated `elif pd.row_reread and not persistent:`.
`not persistent` is a **proxy for "the pinned tile isn't too big to keep in L2"** — state it as such.
It is CONSERVATIVE: chunked ⟹ tile ≤ `LOOPED_CHUNK × itemsize` = 64 KiB (always L2-friendly), so the
proxy withholds the hint on EVERY persistent row — including persistent-but-small/medium ones where
it would help.

MEASURED evidence the real variable is TILE BYTES, not persistence (both rows below are persistent):
- cross_entropy V=50257 → pinned tile ~196 KiB → hint ~neutral/+1.4% (1 run).
- cross_entropy V=65536 → pinned tile ~256 KiB → hint **-7.6%** (3 runs, stable). Pinning a 256 KiB
  row as `'last'` oversubscribes the shared L2 (concurrent CTAs × per-tile bytes) and evicts the
  store/intermediate lines; on a persistent row the row's reuse is already register/SMEM-resident, so
  the pin buys nothing and only costs. This is the crossover the `not persistent` proxy blunt-force
  avoids by never pinning any persistent row.
- rms_norm N=768 → 3 KiB → hint ~neutral (small tile, no harm either way).

FAITHFUL RULE: `apply reread hint iff pd.row_reread AND pinned_tile_bytes <= L2_FRIENDLY_BYTES`.
Compute `pinned_tile_bytes` from the ACTUAL re-read load — the `MemoryOpFact` at
`pd.reread_eviction_index` — NOT a reconstruction:
    reread_load = the memory_op_fact whose eviction_index == pd.reread_eviction_index
    pinned_tile_bytes = m_block × reread_load.inner_extent × reread_load.dtype.itemsize
`inner_extent` is that load's reduced-axis width and `dtype.itemsize` its real element size (verified
present on the fact: cross_entropy reread load = `logits`, inner_extent=30522, fp32; rms_norm = `x`,
768, fp32). This is strictly better than `m_block × primary_r_block × itemsize`, which (a) assumes the
re-read load IS the primary reduction's tile and (b) uses the primary's dtype — both can diverge (a
bf16 kernel, or a persistent full-slice whose padded `primary_r_block` != the true extent). Constant
lands between 196 and 256 KiB. This drops the `persistent` coupling entirely: it pins small persistent
rows (the value the proxy leaves on the floor — cross_entropy V=50257 +1.4%, and recovers the
grad-param ~2% since those tiles are small) and withholds on the genuinely-huge ones (V=65536).

NOTE on problem size: the gate is on the TILE (`m_block × r_block × itemsize`), not the problem N.
Concurrent L2 pressure is occupancy-bound (concurrent CTAs × per-tile bytes), and concurrent CTAs is
capped by occupancy not grid size, so two shapes with the same tile bytes behave the same regardless
of M/N. (My earlier two cross_entropy points covaried M; the design does not depend on that — same
tile bytes ⟹ same behavior.)

VALIDATE when batched: recorder diff + gates; confirm the grad-param hint returns (small tile) and
wide cross_entropy stays unpinned. Calibrate the one constant with a tile-bytes sweep on
cross_entropy (a single M is enough).

---

## S2 (suggested) — persist-hold ceiling: key on APPLY-REREAD (separate re-reading pass), not category

### The two ceilings are RIGHT; the KEY is a proxy
`size_reduction_tiles` currently picks the persistence-hold ceiling by reduction CATEGORY:
```
hold_ceiling = USER_TILE_PERSIST_HOLD_MAX_BYTES (294912)  if d.category is USER_TILE
               else PERSIST_HOLD_MAX_BYTES (737280)
```
Two ceilings ~3x apart is CORRECT and each end is measured (see below), but `d.category is USER_TILE`
is the wrong discriminator. The real physical variable is **how many PHYSICAL passes re-read the
reduction row, i.e. which cache tier the reuse lives in**:
- SINGLE fused pass (cross_entropy rolled): the row is loaded ONCE and reused from REGISTERS/SRAM
  across amax->sub->exp->sum. No cross-pass HBM/L2 dependency -> persistence pays out to the SRAM
  limit -> the BIG ceiling.
- A SEPARATE apply/normalize loop that RE-READS the row (softmax_two_pass: pass-1 online-softmax then
  pass-2 `out=exp(x-mi)/di` re-loads x): the reuse is served from L2 (pass-1 warms L2, pass-2 hits it)
  -> persistence pays ONLY while the row fits the L2 working set -> the SMALL ceiling; past it,
  streaming wins.

Category is only a COINCIDENTAL proxy: on the corpus most USER_TILE happen to be multi-pass and most
FULL_SLICE single-pass — but a SINGLE-PASS user-tiled reduction (no apply reread) should get the BIG
ceiling, which the category rule wrongly denies. This is the same root cause as S1's eviction gate
(single-load-vs-reread / which cache tier) — S1 and S2 are two faces of one signal.

### The measured crossovers the two ceilings must bracket (all this session, replay_bench, single-proc)
- softmax_two_pass (multi-pass) N=32768 -> persist_bytes 262144 -> persist BEATS chunk-16384 by +30%.
- softmax_two_pass N=49152 -> 393216 -> persist LOSES by -34% (must chunk). => small ceiling in
  (262144, 393216); 294912 sits there.
- cross_entropy (single fused) N=50257 -> ~402056 -> persist wins +7%.
- cross_entropy_ls_zloss (single fused, 3-tile bf16) N=50257 -> ~603084 -> chunking LOSES +20%
  (persist needed). => big ceiling must be >= ~603 KiB; 737280 provides it.
So: the ~3x gap is the L2-working-set budget (multi-pass) vs the SRAM/register budget (single fused).
NOTE: the small ceiling (294912) rests on ONE softmax crossover (2-tile fp32); firm it up with a
tile-bytes sweep (softmax N in {24576,32768,40960,49152} + a bf16 point) when batched.

### The faithful signal (verified computable from existing facts)
"apply-reread" = there exists a LOAD of a tensor that ALSO feeds the primary reduction, but THIS load
feeds a STORE and NO reduction (a separate pass re-reading the reduction row to write output):
```
facts = _collect_memory_op_facts(dev)[0]
red_tensors = {f.tensor_name for f in facts
               if f.kind=='load' and any(ax==pd.block_id for ax,_ in f.reductions_fed)}
apply_reread = any(f.kind=='load' and f.tensor_name in red_tensors
                   and f.stores_fed and not f.reductions_fed
                   for f in facts)
hold_ceiling = SMALL if apply_reread else BIG
```
Measured classification across the corpus (this session):
  apply_reread=True  (SMALL): softmax, welford, rms_norm, layer_norm, gated_rmsnorm, fused_add_rmsnorm,
                              fused_add_layernorm, scaled_masked_softmax, dynamic_quant
  apply_reread=False (BIG):   cross_entropy, sum, long_sum, kl_div, jsd
Matches the intent: a single-pass user-tiled reduction (kl_div/jsd) gets BIG; a multi-pass one
(softmax/welford) gets SMALL. `non_reduction_loop_block_ids` is NOT the signal — softmax's 2nd loop
reduces over the SAME axis so it's () there; only welford (normalize over a different axis) sets it.

### The ONE behavior change + the risk to verify
vs today's category rule, `apply_reread` flips ONLY **rms_norm / layer_norm forward** from BIG->SMALL
(both are FULL_SLICE that DO re-read x in a normalize pass -> physically should be SMALL). This is
CONFIG-NEUTRAL on the current corpus: rms_norm/layer_norm shapes are N<=16384 (persist under both
ceilings) then jump to N=131072 (~2 MiB, chunks under both). The small ceiling only changes a
HYPOTHETICAL N in (16384, ~49152), which the corpus does not contain -> replay_bench has no such cell
("cell not found: curriculum/rms_norm/(8192,32768)"). So the reclassification is physically motivated
(rms/layer_norm re-read x like softmax) but UNMEASURED in the gap. TO VERIFY when batched: add a
wide-N rms_norm/layer_norm shape (e.g. (8192,32768)) to the recorder/bench and confirm it wants chunk
(the signal's prediction) not persist. If it wants persist, the signal is too aggressive for the
fused-normalize case and needs refinement (e.g. also require the reread be a genuinely separate
grid/tile loop, not a rolled reduction_loop copy).

### Relation to S1
Same underlying quantity (which cache tier the reuse lives in / single-load-vs-reread). Ideal end
state: ONE derived signal — passes x tile-bytes vs the relevant cache budget — driving BOTH the
persist-hold ceiling (S2) and the eviction hint (S1), replacing both the `persistent`/category proxies.

---

## S3 (NICE-TO-HAVE, low priority — not worth much time) — the non-reduction `loop_budget` is generous and thinly validated

The final pass sizes non-reduction loops (welford's normalize, rms_norm_per_block's groups_per_row)
and any standalone tiled loop (`bid not in seated`) via:
```
loop_budget = _pp2(max(1, cls.ROW_PERSIST_MAX_BYTES // itemsize))   # fp32: 245760//4 -> pp2 -> 16384
sizes[bid]  = max(1, min(extent_of(bid), loop_budget))
```
Rationale (correct as far as it goes): such a loop is a SEPARATE sequential pass co-resident with
nothing in a reduction group, so it gets a FRESH ROW budget rather than being floored (flooring to 1
would serialize the pass). => it can take up to a 16384-element tile (fp32) / 32768 (bf16).

WHY IT'S UNDER-TESTED / MIGHT NEED A CEILING:
- Generous: 16384 elems is large; no measurement backs that specific width — it was never swept.
  A hard cap (analogous to WIDEN_MAX_ROWS / LOOPED_CHUNK) may be warranted if a wide apply loop
  over-tiles and spills.
- RANK-BLIND: `ROW_PERSIST // itemsize` models a single `[extent]` vector. If the loop's actual
  resident tile is >=2-D (e.g. `[m_block, loop_extent]`), the true footprint is `m_block×` larger
  than this per-element budget accounts for — so it can over-size a multi-dim apply tile. (Contrast
  the reduction footprint, which sums ACTUAL live-tile dims; this fallback does not.)
- THINLY EXERCISED: only welford + rms_norm_per_block hit the non_reduction_loop path on the corpus;
  the `bid not in seated` standalone-loop branch is largely corpus-dark. So the 16384 generosity has
  almost no measured backing either way.
SUGGESTED when batched: sweep the welford normalize / rms_norm_per_block loop widths; if a smaller
cap is neutral-or-better, add a `LOOP_BUDGET_MAX` (or fold this loop into the same Σ-over-live-tiles
footprint the reductions use, so it becomes rank-aware instead of a per-element proxy).

---

## S4 (suggested) — replace BOTH `scale *= carried_2d_count` fudges: reduction site -> cross-group-max (residency); widen site -> occupancy guard

### The two fudges
`size_reduction_tiles` has two `*= max(1, carried_2d_count)` lines:
- (A) REDUCTION sizing: `scale, flat = footprint_terms(group.live_tiles, axis); scale *= c2d`
- (B) grid WIDEN:      `scale_w, flat_w = footprint_terms(tiles, mbid); scale_w *= pd.carried_2d_count`
Both are ACTIVE only for jsd on the corpus (kl_div c2d=1 -> ×1 no-op; grad-param c2d>1 reductions are
MATERIALIZED full-slice -> seated at full extent, never reach either line). Dropping (A) => jsd R
2048->4096, MEASURED +11.6%. Dropping (B) => jsd grid 1->2, MEASURED +10.8%. So neither is removable
as-is; both need a faithful replacement.

### Root mechanism (established this session, partly reg-probe / partly reasoned)
The byte budget (CARRIED = 122880 ≈ HALF the H100 256KB register file) is really a REGISTER-OCCUPANCY
proxy: "the per-CTA footprint that still leaves room for 2 CTAs/SM." jsd's two +11% regressions are
register-occupancy cliffs, NOT spills (measured n_spills=0 both; grid=1 -> ~30 regs/thread -> 2 CTAs/SM;
grid=2 -> ~42 regs -> 1 CTA/SM). Carried `[grid, R]` accumulators are PINNED in registers across the
whole inner loop, so they dominate n_regs; widening the grid multiplies exactly that pinned state.
(Caveat: the 30/42 reg counts are from ONE successful probe; the GC-based reg hook was flaky on
re-runs -> treat exact numbers as reasoned-not-reconfirmed. Lock with ncu `sm__warps_active` before
implementing.)

WHY occupancy bites the WIDEN but not the REDUCTION site (the asymmetry): the budget is a TIGHT
occupancy proxy when you size an axis to FILL it (Pass 1 / R: R=2048 is chosen precisely because R=4096
would exceed the half-register-file budget -> 1 CTA/SM; occupancy is baked into the budget value). It
is a LOOSE proxy when you size against its RESIDUAL (Pass 2 / grid: R is already seated ~big, and the
leftover-byte arithmetic permits grid=2 at 2.14x even though the pinned carried state's register cost
pushes past the CTA cliff). Same physics, but Pass 1's budget-as-occupancy works and Pass 2's does not.
NOTE the existing `occ_widen` cap does NOT catch this: it is PROGRAM-COUNT occupancy (keep >= num_sm ×
MIN_WAVES programs); jsd grid=2 leaves 4096 >> 1056 programs, so occ_widen permits it. Register
occupancy is a SEPARATE, unmodeled notion.

### The fix — three parts

(1) DO NOT touch the live-tile fact. Per-group snapshots are FAITHFUL (verified: no under-count within
    any group — jsd gid5=5, If-branches=7, else-branch=12). jsd's issue is ATTRIBUTION, not the fact.

(2a) REDUCTION site -> CROSS-GROUP-MAX (a residency fix; DELETE the (A) multiplier). Size a SHARED
     reduction axis (appears in >1 group's live_tiles) against `max(scale over the groups that span
     it)`, taking `flat` from that same max-group (NOT an independent max — mixing terms across groups
     breaks the chunk arithmetic). VERIFIED (grid at floor=1, CARRIED budget): jsd bid0 group scales
     {gid1:9, gid3:9, gid5:5} -> MAX=9 -> R = pp2(122880/4/9) = 2048 (target, for the RIGHT reason —
     the heaviest group spanning the shared axis). kl_div: one group spans bid0 (scale 6) -> max==6 ->
     4096 (unchanged). The budget already encodes occupancy, so no extra occupancy term is needed once
     the correct (heaviest) group is used.

(2b) WIDEN site -> OCCUPANCY GUARD (DELETE the (B) multiplier). The residual-byte proxy is too loose;
     `n_regs` is unknowable at seed time, so use the seed-visible proxy: a kernel with a carried 2-D
     accumulator (`carried_2d_count >= 1` on any reduction in the group, or on pd) KEEPS its resident
     grid at FLOOR — it does not widen. Rationale: carried accumulators are pinned in registers across
     the loop; widening multiplies that pinned register state, tripping the CTA-occupancy cliff the
     leftover-byte budget can't see. This is a SECOND occupancy rule beside `occ_widen` (program-count):
     one guards starvation, one guards register-residency. VERIFIED: jsd grid stays 1; kl_div's grid is
     already at floor 1; they are the only carried kernels on the corpus.

### Scope / risk / validation
- CORPUS IMPACT: jsd only (only kernel with a shared-across-groups reduction axis AND the only carried
  kernel whose grid could widen). Expected to land on jsd's CURRENT configs (R=2048, grid=1) but derived
  faithfully -> likely config-NEUTRAL vs today, with both multipliers deleted.
- VERIFY when batched: (i) recorder diff shows jsd unchanged, nothing else moves; (ii) re-bench jsd
  (R stays 2048, grid stays 1); (iii) the multi-`ReductionLoopGraphInfo` norm kernels (multiple groups
  over the SAME rolled axis) are undisturbed by cross-group-max — their groups have equal scale, so
  max == any; (iv) LOCK the register-occupancy mechanism with ncu (`sm__warps_active` / CTAs-per-SM at
  jsd grid=1 vs grid=2, and ideally a synthetic carried-2D-no-branches kernel) so 2b rests on measured
  occupancy, not one flaky reg probe.
- OPEN DESIGN POINT for 2b: "keep grid at floor for any carried kernel" is the simplest guard. If a
  future carried kernel genuinely WANTS a widened grid (register-light despite a carried tile), this is
  too blunt — the truly-faithful form is a register-occupancy cap, which needs a compile-time reg
  estimate the seed doesn't have. Flagged; the blunt guard is corpus-safe today (jsd/kl_div only).

---

## IMPLEMENTED (vllm-climb3-latest session) — S2, S1, S4 landed; results below

All three implemented on top of the audit-session uncommitted baseline. NOTE: that baseline was NOT
green as handed over — `test_seed_is_persistent_one_row` failed (num_warps 8!=4) because the audit's
`_has_reduced_away_grid` residency change had no matching fixture update (the mock `CoResidencyGroup`
had empty `live_tiles`, so the grid axis read as reduced-away -> the >=8 grad-param warp floor fired).
Fixed the FIXTURE (populated `live_tiles=((0,1),(0,))` — a real resident-grid one-row reduction), not
the code. All 52 pass after.

S2 (persist-hold ceiling -> apply-reread): DONE as written. Built `_apply_reread(spec, pd)` (a load
of a primary-reduction tensor that feeds a store and NO reduction). Replaced the `d.category is
USER_TILE` key. Verified config-neutral on corpus; the rms_norm/layer_norm forward BIG->SMALL
reclassification VERIFIED CORRECT off-corpus (new `_lab/redesign/microbench_offcorpus.py`): rms_norm
N=49152 chunk BEATS persist +32% (SMALL's verdict), N=32768 persist beats chunk +8% (SMALL persists
there) — the SMALL crossover is right for the fused-normalize kernels too, and BEATS the old category
rule which would have held them persistent and lost 32%.

S1 (eviction gate): the doc's TILE-BYTES hypothesis is REFUTED. Measured the 'last' hint effect:
grad-param 128 KiB HELPS (+2-3%), cross_entropy 122 KiB HURTS (+2.2%), 196 KiB helps (~1%), 256 KiB
HURTS (+6.9%) — NON-MONOTONIC in tile bytes, so no single threshold fits. `_apply_reread` ALSO fails
(both grad-param and cross_entropy are False — grad-param's reread x load feeds BOTH a reduction and
a store). The faithful discriminator is `_has_reduced_away_grid`: the grad-param M-COLLAPSE loop
reloads x per-row across the collapse (reread genuinely hits L2 -> pin helps), cross_entropy is one
fused persistent row (no L2 reload -> pin only oversubscribes). Gate is now
`pd.row_reread and _has_reduced_away_grid(spec)`. Recovers grad-param (evict `['last',...]`), keeps
cross_entropy unpinned. This also closes the loop the DONE `persistent=(r>=ext)` change opened.

S4 (both carried fudges): DONE. 2a = `_max_group_footprint` (size a shared reduction axis against the
MAX-scale group spanning it): jsd bid0 spans groups {gid1:9, gid3:9, gid5:5} -> max 9 -> R 2048;
kl_div one group scale 6 -> 4096. 2b = `carried_kernel` guard (any `carried_2d_count>0` -> resident
grid stays at FLOOR, no widen). Both `scale *= carried_2d_count` lines deleted.

ncu LOCK of 2b (the flaky-reg-probe concern) — jsd `_helion_jsd_forward`, R=2048, grid=1 vs grid=2:
  grid=1: regs/thread=32, occupancy_limit_registers=2 CTAs, warps_active=98.7%, ctas_active=6.18%
  grid=2: regs/thread=42, occupancy_limit_registers=1 CTA,  warps_active=49.9%, ctas_active=3.13%
Widening the resident grid pushes regs 32->42, halving the register-limited CTAs/SM (2->1) and
warps_active (98.7->49.9%). The +11% grid=2 regression IS this register-occupancy cliff — measured,
not reasoned. 2b is justified.

---

## _apply_reread WRONG-CAP: a VERIFIED synthetic false-negative (adversarial workflow, 2026-07-02)

Adversarial workflow (4 agents) generated 11 synthetic kernels targeting _apply_reread's ceiling
pick; verified serially on H100 via _lab/redesign/verify_synth_kernel.py.

CONFIRMED WRONG CAP (/tmp/synth_reread_softmax.py — "reread entropy"): a two-physical-pass kernel
byte-isomorphic to softmax_two_pass, EXCEPT pass-2 REDUCES the normalized probs into an entropy
scalar instead of STORING them. Pass-2's load is therefore (R,-) not (-,S), so _apply_reread finds
no store-only load -> returns False -> picks BIG (737280). row_reread survives via pass-1's amax+sum
fork (cnt=2), so the persistence hold IS armed. At N=49152 fp32 the seed emits block_sizes=[1,65536]
(PERSIST the full row). MEASURED: persist 708us vs chunk 316us -> CHUNK is 2.24x FASTER. So BIG is
the wrong cap by 2.24x; the kernel wants SMALL (chunk), identical to the softmax refetch cliff it
mimics. The ONLY delta from the measured-SMALL corpus kernel (softmax) is pass-2's sink kind — which
is exactly the bit _apply_reread keys on, and is causally irrelevant to row residency.

NOT a wrong cap (/tmp/synth_reread_variance.py): same 2-pass shape but pass-2 = cheap (x-mean)^2 sum.
row_reread came out FALSE (the ones-count reduction constant-folded), AND measured true preference is
PERSIST (+31%) anyway. So this one is correctly-ish handled / inert — confirms only the
softmax-isomorphic heavy-pass-2 case (#2) is the real false-negative, not every reduce-then-reread.

CONCLUSION: _apply_reread's `not reductions_fed` clause IS evadable by a genuine second pass that
reduces-before-storing (the false-negative class flagged earlier, now concretely reproduced at 2.24x).
It does NOT occur on the current corpus (all corpus two-pass kernels store in pass 2), so committed
HEAD d6d4ac21 is safe, but the heuristic is NOT causally airtight — a normalize-then-reduce second
pass fools it. The faithful signal would be "a 2nd physical load of the primary row exists" regardless
of that load's sink, but that over-fires on rolled config-copies (established earlier) — no clean
static predicate captures it; the true axis is per-config pass count, unknown at seed time.
