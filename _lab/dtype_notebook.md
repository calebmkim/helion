# DTYPE CLIMB — worker notebook (bf16/fp16 reduction seed heuristic)

Fresh dtype climb on the **inherited run3 fp32 champion** (branch `reduction-pr-with-lab`,
baseline SHA `e9251140`). This is NOT a run3 resume — run3 logs are reference only. The
authoritative manual is `_lab/prompts/{hillclimb-method,local-setup,gate-prompts,dtype-task}.md`.

Source of truth (method §6.1): THIS notebook + `_lab/ledger.json` key `dtype`. Trust the log
over context. Wins AND rejections recorded as one-liners with evidence pointers.

## ⟹⟹⟹ FRESH-CONTEXT RESUME (NEWEST, 2026-06-08, D4 BUILD session) — READ FIRST
**Human asked to BUILD D4 (deferred narrow-N occupancy win) — DONE, GATE-CONFIRMED, BANKED.**
Branch `reduction-pr-with-lab` tip ~`e70e0a79`. D4 = SHIPPED WIN #3. Commits:
`3408c4f5` D4 code → `7b9f476d` notebook → `f1869309` gate verdicts → `d666933e` report →
`fa63c50d` fp32-cap decision → `e70e0a79` dynamic-grid hardening. NO git push.

**GATES ALL PASS** (ledger `D4_narrowN_w1_gate_suite` + `D4_no_regression_config_proof`):
Gate A 3/3 NOT-refuted (softmax bf16 +20-26%, welford +33%, held-out (8192,768) +61%; danger cells
excluded; noise≪signal). Gate D faithful (both new facts; input_load_itemsize diverges from fact.itemsize
exactly where needed). Gate F mechanism (cross-warp reduction-tree overhead; boundary verified
occ124 w1+52%→occ497 w8+2%→occ993 w8+29%). Gate E no overfit. NO-REGRESSION config-proof: 777-cell
BEFORE(ab9dcb98)/AFTER, 39 diffs ALL num_warps→1 in the predicted narrow zone, ZERO non-warp diffs, fp32
invariant except softmax (16384,512) (−0.3% = NOISE; logged keep-cap decision, net +6.5% on 22 fp32 cells).
Dynamic-grid hardening (`e70e0a79`): require grid_rows>0 so jagged/dynamic-M (unknown occ) declines.

**REFRAME (human, mid-run): oracle≤tc does NOT excuse a seed<oracle gap. Re-aim exempt losers at oracle
parity (~5%), not tc.** See [[feedback-oracle-parity-not-tc-exemption]]. D4 already rescues 3 prior-"exempt"
shapes (welford (16384,896), rms/ln narrow). IN FLIGHT (task #6): fresh full-oracle re-audit of welford
mid-N (5120/7168/14336) bf16 + jsd bf16 → /tmp/oracle_reaudit.json.

### OPEN LEAD — welford mid-N seed↔oracle gap (NEW, from re-audit; UNCONFIRMED)
Oracle re-audit agent (harness flaky, partial signal): welford bf16 (16384,5120) seed=w16 ~135us vs a
quick-oracle candidate w2 blocks=[1,2048,1024] ~128us → seed loses ~5-9%. These mid-N welford shapes are
ABOVE the D4 byte cap (N=5120 → 10240 B > 2048) so D4 does NOT fire — they stay ramp-w16. Hypothesis:
welford's SERIAL recurrence (count/mean/M2 combine) is reduction-tree-bound across a WIDER byte range than
the generic narrow-N effect → wants fewer warps even at mid-N. CAUTION: oracle ALSO shrank block_sizes
(8192→2048/1024), so the w16→w2 gain may be COUPLED to smaller blocks — matched-lever A/B (perturb DOWN
from the full candidate, warps-only vs warps+blocks) is running: /tmp/welford_midN.py → welford_midN_result.json.
If confirmed + warps-carried, the faithful key is welford's Band-C structure (non_reduction_loop_block_ids,
already a fact), NOT a byte cap. DO NOT bank until the A/B separates warps from blocks + a fresh full oracle.

**A/B RESULT (CUDA-graph, /tmp/welford_midN_result.json) — gap is REAL + warps-ALONE, optimum is w4:**
  - welford bf16 (16384,5120): seed w16=133.5us → **w4=116.7us = +14.4%** (warps-only, seed blocks). Sweep
    w1=164,w2=119,w4=117,w8=121,w16=134,w32=199. COUPLED smaller-block cands are WORSE (w4+2048x1024=117.8,
    w2+2048x1024=135) → the partial-oracle's "smaller blocks" was a red herring; lever is PURE num_warps.
  - (16384,7168): seed w16=170 → w4=166 = +2.5% (marginal). (8192,14336): seed w16 already optimal (+0%).
  - So: welford wants **w4 not w16** at mid-N, gain FADES with N (+14.4%@5120 → +2.5%@7168 → 0@14336).
    The ramp gives w16 (rnumel 5120-16384 → w16). DISAMBIGUATION RUNNING (/tmp/midN_crosskernel.py): do
    rms/ln/softmax/sum ALSO want w4 at (16384,5120)? If yes → general ramp mid-N fix; if welford-only →
    welford serial-recurrence structure (Band-C fact). Decision pending that result.

### ★ CROSS-KERNEL RESULT — mid-N over-warping is GENERAL (re-read kernels), NOT welford-specific ★
(16384,5120) bf16, ramp gives w16, but BEST num_warps per kernel (CUDA-graph /tmp/midN_crosskernel.json):
  - **softmax w16→w4 = +44.7%** (sweep w2=124,w4=117,w8=121,w16=169,w32=218) — HUGE
  - layer_norm w16→w4 = +11.5% ; welford w16→w4 = +14.4% (from prior A/B)
  - rms_norm: FLAT, w16 already best (+0%) — resident-reuse, warp-insensitive
  - sum: FLAT, +0.8% — single-load, warp-insensitive
SO: the shared ramp is OVER-WARPED at MID-N (rnumel 4097-16384) for RE-READ kernels (softmax/ln/welford)
at moderate occupancy — same cross-warp-tree mechanism as D4 narrow-N, the NEXT ramp step up. This is a
2nd, BIGGER occupancy-gated regime (softmax +44.7%). rms/sum flat → safe to lower their warps too.
NEW TASK #7. Need M-axis sweep at N=5120/8192 (does w4 invert at high occ? — /tmp/midN_occ_sweep.json
running) + Gate A/D/F before banking. RISK: shared ramp touches all; bound by occ + faithful facts.
The faithful key may be the SAME (occ + input_load_itemsize), just extending the byte cap from 2048 up to
~the mid-N band — OR a re-read-gated rule (row_reread distinguishes the warp-sensitive softmax/ln/welford
from flat rms/sum... but rms_norm IS row_reread too yet flat — so the separator is subtler, TBD from data).

### D4 IMPLEMENTATION (HEAD 3408c4f5) — what was built
Two faithful `ReductionFact` fields (device_ir builders T1+T2; config_spec):
  - `grid_rows` = product of static M-axis extents (occupancy numerator); `_grid_rows()`.
  - `input_load_itemsize` = element size of the HBM **input row load** feeding the reduction
    (`_input_load_itemsize()`, via `_accessed_tensor_fake` on reduction-fed loads) = **2 bf16 / 4 fp32**,
    DISTINCT from `fact.itemsize` (=4 fp32-accumulator at BOTH dtypes for softmax/rms/ln — the exact
    reason Path A's cap couldn't discriminate). kl_div/jsd → 0 (no single reduction-fed row load).
`_num_warps(fact, num_sm)` NEW narrow branch (priority over the ramp):
  **w1 IF `input_load_itemsize>0` AND `rnumel*ils <= 2048` AND `grid_rows//num_sm <= 512//ils`.**
  (bf16: rnumel≤1024 AND occ≤256; fp32: rnumel≤512 AND occ≤128.) num_sm via `get_num_sm(env.device)`
  threaded through `_persistent_looped`. Constants `NARROW_W1_MAX_BYTES=2048`, `NARROW_W1_OCC_BYTES=512`.

### EVIDENCE (fresh CUDA-graph, /tmp/occ_results.json + /tmp/numwarps_gaps_results.json, num_sm=132)
Threshold FIT across softmax/rms/ln/sum/welford/CE × bf16/fp32 × M-grid + an rnumel sweep (N 512..3072):
  - **BYTE_CAP=2048 is the clean winner**: 9+ >3% wins (up to **+62%** softmax bf16), **worst regression
    ~2.3%**, ZERO bad cells. 3072 → a −4.9% cell; 4096 → −9% cells. So 2048.
  - The byte cap UNIFIES both axes: bf16 N=1536→3072B and fp32 N=768→3072B break at the same bytes;
    2048B sits safely below (bf16 rnumel≤1024, fp32 rnumel≤512).
  - occ cap 512//ils (bf16 256 / fp32 128) sits BELOW every measured crossover cliff (softmax bf16 ~496,
    fp32 ~248).
  - **Path-A poison cell softmax fp32 (32768,512) occ248 → w4 (NOT w1)** ✓ — fp32 occ cap=128 < 248.
    THIS is the structural fix Path A lacked.
  - **kl_div danger (bf16 V≥2048 = +27-46% w1 regression) EXCLUDED** by ils=0 guard ✓.
  - **CE narrow-V BONUS win** (V≤1024 → w1 +13-35%) gained for free; CE wide-V w8 shipped win INTACT
    (V=32000 bf16 → w8, fp32 → w32) ✓.
Tests: 20 autotuner-heuristic pass, 22 reduction example tests pass. ruff+pyrefly clean (2 pyrefly errors
pre-exist at L102/L3451, unrelated).

## ⟹⟹ PRIOR FRESH-CONTEXT RESUME (2026-06-08, branch session 5fec6ded)
**Run is in a banked, healthy state (HEAD on this branch ~a890a5c4+report edits). Two shipped wins
intact (jsd correctness + CE bf16 wide-V w8 via full_width_output, gate-confirmed 3/3). All commits are
`git log` on branch `reduction-pr-with-lab`. NO git push done.**

## D4 — narrow-N fewer-warps occupancy win (DEFERRED; status as of HEAD 04155998)
**The loss:** at a narrow reduction extent, fewer warps (w1) beat the ramp's pick by 10-20%
(ncu: the cross-warp shared-mem reduction tree is pure overhead when each warp's slice is tiny).
Applies fp32 AND bf16 (hardware/occupancy effect, not dtype-specific); run3 found+deferred it too.

**What a correct rule needs:** w1 gated on `grid_rows//num_sm` (occupancy) — w1 wins at low/moderate
occ, then INVERTS as a **bimodal CLIFF** (softmax bf16 N=512: −24% at occ≤248 → **w32 +6× at occ≈1985**).
Ceiling is kernel-structure-dependent (re-read kernels softmax ~occ496/welford ~992 flip ~8× earlier
than resident-reuse rms/ln >1985) AND dtype-shifted ~4× for re-read kernels (softmax N=512: fp32 flips
~occ124 vs bf16 ~occ496 — two-pass HBM re-read amplifies the 2× input-byte ratio to ~4×). Full mechanism
(why fewer warps, the cliff, the bimodality, timing-fairness) is in DTYPE_REPORT.md "Mechanistic discoveries".

**Why it's hard (it's validation+modeling cost, NOT code):** the *code* is ~half a day (occupancy
machinery already exists — `grid_rows`/`num_sm` computed in T1 `get_seed_config` ~L644; `_persistent_
looped` already has `env`; add an input-itemsize field like `full_width_output`). The cost is: (1) this
lever has a ~75% gate-kill rate here — 3 prior warps rules looked obviously-correct yet each hid a >10%
regression (byte-ramp ×2, num_load), so a full kernel×dtype×M CUDA-graph sweep + Gate A/D is mandatory;
(2) the ~4× crossover scaling must be empirically fit per memory-traffic class from sparse data without
overfitting — the part Gate D distrusts. Reward is small (~5-10% on ~6 niche curriculum shapes).

**Two documented paths, BOTH refuted** (gated assessment, ledger `D4_landing_feasibility_assessment`):
  • **Path A** `fact.itemsize*size_hint<=2048 AND occ<=250 → w1`: REFUTED — admits softmax fp32
    (32768,512) at **+19.6%** (Gate-A hard fail). `fact.itemsize` is the fp32 *accumulator* width (=4 for
    softmax/rms/ln at BOTH dtypes) so it can't discriminate fp32 from bf16.
  • **Path B** new `second_hbm_pass` fact: NOT BUILDABLE — premise false at device-IR (all 4 kernels
    re-read in ≥2 loop graphs, layer_norm most; reduction-rolling dissolves the resident-vs-reread split
    → redundant with `row_reread`, Gate-D fail).

**Sketch of the real path (if pursued):** thread `grid_rows//num_sm` into the warp decision; key the
occ threshold on the **INPUT-LOAD itemsize** (`MemoryOpFact.dtype`, =2 bf16/4 fp32 — faithful, dtype-
AGNOSTIC, NOT a dtype-kind branch and NOT `fact.itemsize`) with an empirically-fit scaling (~4× for the
two-pass class); start with the safe **resident-class-only** subset (`full_width_output AND input
itemsize≥4 AND size_hint≤512 AND occ≤~200` → rms/ln bf16, low regression risk) and only extend to
softmax once the input-itemsize-scaled threshold is fit + Gate-A'd across the high-occ shapes.
**RECOMMENDATION: keep deferred** — effort/reward unfavorable vs the shipped CE win; build only if the
occupancy win is specifically wanted. Artifacts: harness `_lab/harness/occ_sweep.py`, /tmp/occ_results.json,
/tmp/xover_check.py (the fp32-vs-bf16 crossover measurement).

---

## ⟹ CURRENT STATE (for resume) — HEAD ~0a861118 — TWO SHIPPED WINS
1. **jsd bf16/fp16 correctness fix** (examples/jsd.py:102 intermediate_dX→fp32; true fp32 no-op).
2. **EDIT#2-v2: CE bf16 wide-V num_warps=8 (+49% on V=32k/50k, GPT2/Llama2 vocabs)** via a new FAITHFUL
   fact `full_width_output` (device_ir `_has_full_width_output_store`: does a store write the result
   back over the reduction extent [M,N] vs a per-row scalar [M]). Gate: `rnumel>16384 AND row_reread
   AND NOT full_width_output AND size_hint*itemsize<=102400 → w8`. **SURVIVED Gate A+D 3/3** (b11e57bd):
   no-regression (0% worst, zero non-CE flips, fp32 config-invariant), win (+30-38% latency, M-stable),
   fact-integrity (faithful on synthetic + all 9 kernels). Scope `NOT full_width` PROVEN necessary —
   broadening to softmax/ln is a +30% non-monotonic trap (softmax 24576→w8 −23% but 32768→w8 +33%).
   geo CE bf16 1.19→1.36. ncu mechanism: re-read scalar-output reduction is reduction-tree-bound, w32's
   cross-warp shared-mem tree throttles reads.

Production diff vs e9251140: examples/jsd.py +6/-1; full_width_output fact (device_ir +43, config_spec
+7); w8 gate + docstrings (triton.py +32). fp32 PROVEN config-invariant. 20 autotuner-heuristic + 23
example tests pass, lint clean.

**PATH TO EDIT#2-v2 (gate-killed attempts taught the faithful property):** EDIT#1 v1/v2 byte-ramps
(occupancy-flip + fp32 under-warp, Gate-A-killed); EDIT#2-v1 num_load≥3 (false proxy — CE loads its row
ONCE; num_load conflated CE's scalar-output with layer_norm's full-width, regressed ln +32%, Gate-D-
killed). The win landed once `full_width_output × row_reread` faithfully separated scalar-output CE
(clean+monotonic w8 win) from chaotic full_width kernels. Gates worked: 4 attempts → 1 faithful win.

**STILL DEFERRED (D4):** narrow-N fewer-warps (10-20%) is OCCUPANCY-gated (needs grid_rows//num_sm fact).
A fork's two landing paths were both found NOT-FEASIBLE by a gated assessment (HEAD 39fe2034) — see the
★★ GATED RESOLUTION ★★ at the top of this notebook + ledger `D4_landing_feasibility_assessment`. Keep deferred.

**ESTABLISHED:** bf16/fp16 transfer for free from fp32 champion (seed beats tc on most shapes:
geo G_cg softmax 1.31, CE 1.19, ln 1.09, kl 1.08, sum 1.07, rms 1.06). CUDA-graph is the headline
metric (do_bench host-overhead artifact). fp16≈bf16 confirmed.
**GATES:** Gate F PASS (warps mechanism). Gate A KILLED warps v1 AND v2 (real >10% fp32/bf16
regressions — the win is occupancy-gated, not bytes-keyable). Reverted warps; kept jsd.
**CONFIRMED-EXEMPT losers (oracle≤tc, codegen-bound):** welford narrow/mid-N, long_sum few-row,
rms_norm(2048,4096).
**OPEN / NEXT ACTIONS:**
  - Confirm remaining small bf16 losers exempt-or-reachable via oracle: layer_norm(2048,6144-10240)
    & (16384,896), rms_norm(16384,896), CE/kl(4096,151936) — all ≤7% from tc (oracle batch was running).
  - DEFERRED big win (D4): occupancy-gated narrow-N w1 (bytes≤2048 AND grid_rows≤~150*num_sm → w1),
    fully characterized (64/91 bucket shapes +5-25%), needs grid_rows//num_sm fact threaded into
    _num_warps + its own Gate A. Worth it only if a broader shape set than the curriculum's 6 matters.
  - Kernel-authoring (out of heuristic scope, documented): sum/welford bf16 input-width accumulator
    accuracy; kl_div/jsd fp16 wide-V NaN; jsd beta=0/1 fp16 NaN.
**KEY LESSON:** validate any warps/config-flip across the FULL M (grid/occupancy) axis at mid+extreme,
not just N/bytes — the Gate A kills were M-axis regressions my N-only sweep missed.

---

## INHERITED STATE (run3 fp32, do-not-regress baseline)
- Heuristic `helion/_compiler/autotuner_heuristics/triton.py`: base `_TritonReductionSeedBase`
  + T1 `TritonReductionTileHeuristic` (rollable rdim) + T2 `TritonReductionUserTileHeuristic`.
- Caps that key on `fact.itemsize` (the dtype-sensitive surface):
  - `ROW_PERSIST_MAX_BYTES=245760` → `can_persist` via `size_hint*itemsize` (per-row resident).
  - `BANDB_R_BLOCK_BYTES=16384` → `cap = 16384 // (itemsize*num_carried_2d_tiles)` (kl_div/jsd).
  - `STRUCTURED_COMBINE_CAP_BYTES=32768` → welford combine `cap = 32768//itemsize`.
  - `_build_block_sizes`: `persist_max_bytes=12288`, `loop_chunk_bytes=8192`, key `n_valid*itemsize`.
  - `_num_warps`: keys on `rnumel` (ELEMENT count → dtype-invariant ✓, not a concern).
- fp32 per-shape open work (reference, from run3 digest): softmax narrow-N (131072,256),
  rms_norm narrow-N (8192,768), cross_entropy persist-boundary (8192,57344). Exempt/codegen-bound:
  long_sum 2M tail, CE wide-V ≥98k.

## STEP-0 GROUND TRUTH (probed e9251140, 9 kernels × 3 dtypes) — THE DTYPE SPINE
`fact.itemsize` reflects the **reduction-input** width (post-upcast), which is what actually
drives every byte-cap. Measured directly (more reliable than source reading):

| kernel | bf16 itemsize | reduction input | seed changes vs fp32? | dtype-sensitive cap |
|---|---|---|---|---|
| sum | **2** | input-width, num_load==1 persistent | no (config-invariant) | ROW_PERSIST (neutral, single-load) |
| long_sum | **2** | input-width, already looped >2^20 | no | none effective |
| rms_norm | 4 (stays) | fp32-upcast (`x.to(fp32)`) | no | none |
| layer_norm | 4 (stays) | fp32-upcast | no | none |
| softmax | 4 (stays) | fp32 accums | no | none |
| cross_entropy | **2** | input-width load | no (50257 persists both) | **ROW_PERSIST crossover moves ~2x wider** |
| kl_div | 4 (stays) | fp32 accum | no | BANDB keyed on 4 → unaffected |
| jsd | 4 (stays) | fp32 accum | **CORRECTNESS FAIL bf16/fp16** | n/a (broken) |
| welford | **2** | input-width load | yes (caps double) | **COMBINE_CAP + normalize tile DOUBLE** |

Empirical fact (Explore + reduction_strategy.py): Helion ALWAYS promotes the *accumulator* to
fp32 via `get_computation_dtype` (input-width caps still under-budget the resident fp32 working
set). itemsize comes from the reduction INPUT (`in_val.element_size()`, device_ir.py:1195).

**Thesis to test (not assume):** byte-caps modeling *resident* state were tuned at
input==accum==4B. The 4 kernels whose itemsize DROPS to 2 (sum, long_sum, cross_entropy, welford)
are where a cap may now be mis-sized. The 4 fp32-upcast kernels are config-invariant — their bf16
bar is purely "did tc's bf16 path shift relative to ours" → still must MEASURE.

## IMMEDIATE FINDINGS (Step 0, pre-climb)
- **jsd bf16/fp16: `ControlFlowTensorMismatch` on `intermediate_dX`** (dtype fp32 != bf16). This is
  a KERNEL-AUTHORING issue in examples/jsd.py, not a heuristic failure (task §traps anticipates
  this). Characterize + decide whether to fix the example or scope jsd to fp32. LOGGED.

---

## STEP 1 — bf16 test-split yardstick (bare-fwd, seed/tc/default, single-proc N=15) @ d6c9da9e
G_seed = tc/seed (>1.03 seed wins; <0.97 seed loses). Full rows in /tmp/bf16_chunk1.json + chunk2.

| kernel | geo G | min..max | losers vs tc (G<0.97) | notes |
|---|---|---|---|---|
| sum | 1.05 | 0.995–1.16 | none | acc bf16-accum floor (kernel, see CORRECTNESS) |
| long_sum | 0.93 | 0.56–1.43 | (8,2097152)=0.56, (48,786432)=0.70, (64,294912)=0.92 | huge-N looped tail; likely codegen-bound (fp32-exempt match) |
| rms_norm | 0.96 | 0.83–1.06 | (2048,4096)=0.83, (2048,6144)=0.87, (16384,896)=0.95, (2048,7168)=0.96 | CONFIG-INVARIANT (fp32-upcast itemsize=4); inherited gap |
| layer_norm | 0.96 | 0.87–1.06 | (16384,896)=.96,(4096,2048)=.95,(2048,4096)=.87,(2048,6144)=.94,(2048,7168)=.94,(2048,10240)=.93 | CONFIG-INVARIANT; tol-trap acc (tiny max_abs) |
| softmax | **1.32** | 0.92–1.81 | (8192,896)=0.92, (131072,128)=0.95 | CRUSHES tc except tiny-N attention |
| cross_entropy | 1.14 | 0.94–1.44 | (4096,151936)=0.94, (1024,250000)=0.955 | wide-V persistent_interleaved (fp32-exempt-ish); narrow beats tc |
| kl_div | 1.08 | 0.93–1.51 | (8192,32768)=0.93, (4096,151936)=0.95 | BANDB keyed on itemsize=4 → unaffected |
| **welford** | 0.99 | 0.80–1.25 | (16384,896)=**0.80**, (16384,5120)=0.89, (16384,7168)=0.92, (8192,14336)=0.95 | **DTYPE-SENSITIVE: caps ÷itemsize → 2x tiles at bf16** |
| jsd | n/a | — | ALL (broken) | ControlFlowTensorMismatch on intermediate_dX |

**WORKLIST priority (round-one bar = beat tc per-shape):**
1. **welford** — only kernel whose SEED CONFIG changes with dtype. Caps `STRUCTURED_COMBINE_CAP_BYTES//itemsize`
   + normalize tile `loop_chunk_bytes//itemsize` give **2x-wider tiles at bf16** (itemsize=2). The combine
   accumulator is fp32 (var_mean→get_computation_dtype), so resident bytes DOUBLE vs fp32 — risks the exact
   spill the run3 fp32 note warned about ("raising normalize cap 2048→4096 regressed welford(262144,5120) 7.3x").
   HYPOTHESIS: cap should reflect the fp32 ACCUMULATOR footprint, not input itemsize=2. Prime climb target.
2. **long_sum** huge-N tail (0.56–0.92) — validate codegen-bound via fresh oracle (anti-giving-up gate), exempt.
3. **config-invariant inherited gaps** (rms_norm/layer_norm/softmax narrow-N) — these lose the SAME at bf16 as
   fp32 (config didn't change). bf16 halves mem traffic but not compute → optimal warps/tiling may differ.
   Needs a NEW dtype-sensitive property (e.g. mem-bound-ness) to move; or oracle≤tc → exempt. Lower priority.
4. **cross_entropy/kl_div** wide-V losers — small gaps, check oracle.
5. **jsd correctness** — kernel-authoring fix or scope decision.

## FP16 OVERFLOW CHARACTERIZATION (Agent-verified, all 9 kernels) — task deliverable #1
fp16 max≈65504, 5-bit exponent. Accumulator promoted to fp32 (safe), but PRE-reduction elementwise
ops execute at input width. Per-kernel risk for fp16 input:
- **NONE** (fp32-cast-before-square or exp-of-≤0-by-max-shift): rms_norm, layer_norm, softmax,
  cross_entropy, kl_div, jsd — safe by construction.
- **OUTPUT-ONLY** (sum, long_sum): row-sum could exceed 65504 only for adversarial all-positive
  input; randn curriculum ~±√N → safe. Kernel output-cast property, not heuristic.
- **INTERMEDIATE** (welford L61 `chunk*chunk` at fp16): overflows if |x|>~256; curriculum uses
  rand[0,1) → safe. Same line that degrades welford bf16 accuracy. Kernel-authoring, not heuristic.
ALL 9 fp16-safe for the realistic curriculum. welford L61 is the one theoretical hazard (would be
fixed by `chunk.to(fp32)` before squaring, like rms/ln — a kernel fix, parallel to the jsd fix).

## CORRECTNESS CHARACTERIZATION (bf16) — Triton-backend accumulator dtypes (Agent-verified @ backend.py:1259)
Helion's Triton `reduction_expr` emits `tl.sum/max/min` with **NO fp32 promotion** (only var_mean/Welford
promotes via inductor_lowering_extra.py:108). So the N-axis reduction accumulates in:
- **sum: bf16** (operand never upcast) → mean_abs 0.265 vs torch 0.057 (~5x coarser). KERNEL-AUTHORING floor,
  NOT seed (seed≈unseeded-default mean_abs). This is the "input-width accumulator kernel" task §traps names.
- **cross_entropy: bf16** (max+sum) — but acc PASSES (output is scalar mean loss; exp computed fp32).
- **rms_norm/layer_norm/softmax/kl_div/welford: fp32** (explicit upcast or fp32 accums) → correctness ok;
  "acc fails" on these are TOLERANCE TRAPS (near-zero normalized outputs, tiny max_abs, huge max_rel).
- **jsd: broken at bf16/fp16** (ControlFlowTensorMismatch), kernel-authoring.

DECISION D1: accuracy gate must compare helion-bf16 vs **torch's-own-bf16** path (both bf16) to isolate
"is the seed/kernel as good as torch at bf16" from "bf16's inherent floor". Will refine the harness gate.

## ★ ITER2 — PIVOTAL: low-M bf16 "tc losses" are a do_bench HOST-OVERHEAD artifact ★
The rms_norm/layer_norm bf16 "losses" (ITER1 reframe) are largely a **measurement bug**, not kernel
gaps. Triple-confirmed (perf-investigator ncu + CUDA-graph; INDEPENDENTLY re-confirmed by my own
CUDA-graph script @ c89bd628):

| rms_norm bf16 | do_bench G | **CUDA-graph G** | verdict |
|---|---|---|---|
| (2048,4096) | 0.758 | **1.061** (seed 8.1us vs tc 8.6us) | artifact — Helion FASTER on-device |
| (2048,6144) | 0.945 | **1.075** | artifact — Helion faster |
| (16384,896) | 0.963 | 0.963 | REAL small gap (high-M amortizes host cost) |
| (8192,2048) | 1.090 | 1.151 | Helion wins both |
| (2048,16384) | 1.046 | 1.061 | Helion wins both |

MECHANISM (ncu, full oracle 361s converged): on-device Helion seed = 12.8us, tc = 13.2us, IDENTICAL
DRAM traffic (18.35MB), both grid=2048 — Helion is AT roofline and slightly faster. The do_bench gap
is Python host-enqueue: Helion 69.7us/call vs tc 47.4us/call; at low-M bandwidth-bound shapes the CPU
can't feed the GPU fast enough so the ~22us host delta bleeds into wall-clock. ~6.8us of it is
Helion's extra `inv_rms=torch.empty([m])`+reshape per call (host-side, NOT autotuner-addressable).
The full oracle "lost" (oracle/tc=0.664) because it optimized a non-existent problem — block_sizes=[2]
/reduction_loops=[2048]/num_stages=4 all make the DEVICE kernel slower and can't touch host overhead.

**DECISION D2 (harness fix): the headline metric for bandwidth-bound seed-vs-tc is CUDA-GRAPH
per-call device time** (captures both arms identically → fair, removes host overhead from both). Plain
do_bench stays as a recorded secondary. This is a method §4 footgun specialization: do_bench host-
overhead, not just cross-process jitter, swamps low-M kernels. (tritonbench's own do_bench has the same
exposure; CUDA-graph is the trustworthy anchor here.)

**Only real device lever found for rms_norm bf16: num_warps 8→16** (~1.3us / ~5% on the persistent
seed at wide rows; ncu+graph confirmed real, not host). Candidate seed improvement — but seed already
uses the rnumel ramp (4096→w8, the boundary). Investigate whether the warps ramp wants to shift at
bf16. ANTI-GIVING-UP: rms_norm (2048,4096) is NOT codegen-bound — it's already ≥ tc on-device; the
do_bench "ceiling" was false. Re-anchor ALL bf16 numbers under CUDA-graph before any exempt claim.

## ★ CORRECTED bf16 yardstick (CUDA-graph G_cg, test split) @ 789e89a3 — THE REAL HEADLINE ★
Re-anchored under CUDA-graph (D2). Most do_bench "losses" were host-overhead artifacts and vanished.
geo G_cg by kernel: rms_norm **1.056**, layer_norm **1.087**, softmax **1.31**, sum **1.07**,
cross_entropy **1.19**, kl_div **1.11**, welford **0.98**, long_sum **0.80** (jsd broken).

**GENUINE device-level CG-losers (the round-one worklist), by cluster:**
| cluster | shapes (G_cg) | hypothesis |
|---|---|---|
| **long_sum few-row huge-N** | (8,2097152)=**0.35**, (48,786432)=0.60, (64,294912)=0.73, (96,196608)=0.89 | 8–96 programs on 132 SMs → massive UNDER-OCCUPANCY; tc split-K's the reduction. BIGGEST gap, top priority. |
| **welford narrow/mid-N high-M** | (16384,896)=0.77, (16384,5120)=0.88, (16384,7168)=0.92 | NOT cap-driven (ITER1). Needs own answer key. (896 also acc-fail.) |
| **softmax narrow-N** | (8192,896)=0.65 | fp32 G=0.97 → dtype-AMPLIFIED. 1 row/4 warps under-utilizes; tc tiles multiple rows. |
| **layer_norm narrow+mid low-M** | (16384,896)=0.93, (2048,6144)=0.96, (2048,7168)=0.97, (2048,10240)=0.94 | small real gaps; check vs oracle (may be host residual / codegen-bound). |
| **CE/kl_div widest-V** | both (4096,151936)=0.95-0.96 | small; likely persistent_interleaved boundary. |

**BEAT TC (CG-confirmed ✓):** all of sum, softmax-wide (up to 1.9!), CE-narrow/mid (up to 1.49),
kl_div-narrow/mid (up to 1.59), rms_norm-mid, long_sum-mid (128,524288)=1.51.

## ITER3 — jsd correctness FIXED + long_sum few-row huge-N is CODEGEN-BOUND
**jsd bf16/fp16 correctness FIXED** (examples/jsd.py:102): `intermediate_dX` was `dtype=_input.dtype`
(bf16) but the loop body's exp/log math promotes to fp32 → ControlFlowTensorMismatch. Changed to
fp32 (matches sibling intermediate_loss + the fp32 dX output). VERIFIED: fp32 acc max_abs=0.0 AND
Triton codegen BYTE-IDENTICAL at fp32 (only src-line comments renumbered — proven by stripped diff)
= true no-op; bf16 acc PASS 0.0001; fp16 acc PASS 0.00002. Kernel-authoring fix (task anticipates it),
NOT a heuristic change. → all 9 kernels now COMPILE+CORRECT at bf16/fp16.

**long_sum (8,2097152) bf16 = CODEGEN-BOUND EXEMPT**: full oracle (287s, converged) returned EXACTLY
the seed config (block_sizes=[1], reduction_loops=[16384], w32, flat) — zero field-diff — and
oracle/tc=0.535 (oracle ALSO loses tc ~2x). seed/oracle=0.999. Mechanism: 8 rows → 8 programs on 132
SMs = under-occupancy; Helion can't split one row's reduction across SMs (no split-K), tc can. The
seed is OPTIMAL among expressible configs. Per Step-2a invariant seed≈oracle<tc → EXEMPT, not a seed
failure. (48,786432)+(64,294912) oracle pending to confirm the whole few-row cluster is the same.

## ★ ITER4 — softmax narrow-N wants FEWER WARPS (clean, large, CUDA-graph-confirmed) ★
Oracle answer key for softmax(8192,896) bf16: oracle picked **num_warps=1** (seed=4). CUDA-graph
confirmed (NOT do_bench — the oracle's do_bench parity was host-contaminated; I re-timed on graph):
two_pass w1=10.1us vs seed w4=13.1us = **23% faster**, vs tc 8.55us. The remaining w1→tc gap is the
TWO-PASS algorithm (reads HBM twice; seed sits on the 2-pass roofline 13.1, tc on 1-pass 8.76;
w1=10.1 is between) — that's codegen/kernel-bound (softmax_decomposed one-pass ties tc), NOT seed.

**WARPS SWEEP across rnumel (CUDA-graph, two_pass, M=8192 narrow / lower-M wide):**
| rnumel | seed ramp | BEST w | seed_us | w1_us | seed over-sub |
|---|---|---|---|---|---|
| 512 | w4 | **w1** | 8.29 | 6.56 | 21% |
| 896 | w4 | **w1** | 13.10 | 10.15 | 23% |
| 1024 | w4 | **w1** | 12.60 | 10.39 | 18% |
| 1280 | w8 | **w1** | 25.63 | 13.53 | **47%!** |
| 2048 | w8 | w4 | 26.92 | 24.67 (w4) | 8% |
| 3072 | w8 | w4 | 43.91 | 35.54 (w4) | 19% |
| 4096 | w8 | w8 | 24.92 | — | 0% (seed correct) |

THE FINDING: the `_num_warps` ramp (rnumel≤1024→4, ≤4096→8, else…) is **over-subscribed at narrow-N
for softmax two_pass** — the optimal is FEWER warps until rnumel~2048, where the ramp finally becomes
right. Huge at (8192,1280) where the ramp jumps to w8 but w1 wins by 47%.
- This is the run3 fp32 hard-pile "narrow-N wants fewer warps" — now CG-confirmed at bf16, clean+large.
- COUNTER-INTUITIVE (fewer warps faster) → Gate F mechanism REQUIRED before banking as a rule.
- NO-REGRESSION: a fewer-warps rule MUST be bounded to narrow-N (w1 regresses at rnumel≥2048).
- Is it dtype-specific? warps-vs-row-width is dtype-agnostic in principle; MUST check fp32/fp16 +
  the OTHER T2/T1 kernels (rms/ln/sum) don't regress — the warps ramp is SHARED across all reductions.

NEXT: (1) Gate-F mechanism (why fewer warps win at narrow row — occupancy/launch? reduction-tree?),
(2) sweep fp32 to confirm not bf16-only, (3) check the SHARED ramp's effect on rms/ln/sum/CE narrow-N
before editing (the ramp is in _TritonReductionSeedBase._num_warps — touches ALL kernels).

## ★ ITER5 — the warps lever should key on BYTES (rnumel×itemsize), not rnumel. GATE F PASSED. ★
GATE F PASS (ncu-confirmed): fewer-warps-at-narrow-N mechanism = the cross-warp reduction's
shared-mem tree + __syncthreads is pure overhead when per-warp work is tiny. w1: 0 shared ld/st, 0
reduction barriers (pure intra-warp shuffle); w8: 262144 shared ld/st, +74% instructions, HIGHER
occupancy yet SLOWER (occupancy hypothesis refuted). Faithful property = per-program reduction work.

DTYPE-GENERAL KEY = **bytes (rnumel × itemsize)**, proven across 3 dtypes (softmax two_pass, CG us):
| bytes | fp32 best | bf16 best | fp16 best |
|---|---|---|---|
| 1536 | — | w1 | w1 |
| 3072 | w1 (fp32 768e) | w1 | w1-w2 |
| 6144 | w8?(fp32 1536e=w8) | w4 | w4 |
| 12288 | w16 (fp32 3072e) | w8 | w8 |
| 24576 | w8 (fp32 6144e) | — | — |
- **bf16 ≈ fp16 everywhere** (itemsize=2 identical) → task's transfer hypothesis CONFIRMED.
- The current ramp keys on rnumel (ELEMENTS): rnumel≤1024→4, ≤4096→8, ≤16384→16, else 32. This
  mis-sizes at bf16: rnumel=4096 is 8192 bytes (wants ~w8) but fp32 rnumel=4096 is 16384 bytes
  (wants ~w16). Keying on BYTES makes it dtype-correct AND is the task's prime-directive lever
  (itemsize/bytes, never dtype-kind).
- Mechanism boundary: warps scale with per-lane work = bytes/(warp_bytes). w1 optimal while the row
  fits one warp's registers cheaply (≲2-3KB); double warps each ~2x bytes thereafter.

PROPOSED byte-keyed ramp (rough, from data — MUST validate across ALL kernels before edit):
  bytes <= 2048 -> 1 ; <= 4096 -> 2 ; <= 8192 -> 4 ; <= 32768 -> 8 ; <= 65536 -> 16 ; else 32
  (fp32: rnumel<=512->1,<=1024->2,<=2048->4,<=8192->8,<=16384->16,else32;
   bf16: rnumel<=1024->1,<=2048->2,<=4096->4,<=16384->8,<=32768->16,else32)
RISK: SHARED ramp touches sum/long_sum/rms/ln/CE/kl/welford too. softmax (2-pass, BW-bound) shows it
biggest; MUST sweep all kernels at the new ramp before banking (no-regression backstop). Note this
also RE-TUNES fp32 (the ramp was do_bench-tuned in run3 → may have been wrong at fp32 bottom too;
must prove fp32 net-positive, not just no-regress).

## ★★ GATE A KILLED EDIT#1-v1 (3/3 refute) → retreated to EDIT#1-v2 (narrow-band only) ★★
The full byte ramp (v1) was REFUTED by all 3 skeptics — the gate did its job, caught what I missed:
1. **fp32 >10% regression**: layer_norm fp32 (512,24576) +15.4% (HARD FAIL). v1's `<=98304→16` bucket
   under-warped wide fp32 (rnumel 16384-24576 wants w32, got w16). MY VALIDATION NEVER TESTED rnumel
   in (8192,24576] — the classic "endpoints not mid-range" backstop failure the method warns about.
2. **Mechanism mischaracterized**: softmax/rms/ln have itemsize=4 at BOTH dtypes (fp32 accumulator),
   so the byte ramp gave them identical bf16/fp32 warps — the "bf16=half bytes" story DIDN'T FIRE on
   the headline kernels; the bf16 gains there came from the finer LOW-END buckets (help fp32 equally).
3. **Overfit**: held-out softmax bf16 (768/1280/1536) regret 1.10-1.65x vs 1.04 in-sample; thresholds
   fenced the curriculum byte-lattice; non-pow2 6144 worst; welford fp32 (16384,1536) LOST to old ramp.

**EDIT#1-v2 (committed 660c90f3) = the defensible core only:** keep the original element ramp for
rnumel>1024 (wide fp32 byte-identical to champion → refutation #1 GONE), refine ONLY rnumel<=1024 by
bytes (<=2048→1, else→2) — exactly the band where bf16's 2KiB vs fp32's 4KiB genuinely differ AND the
ncu narrow-N win lives. Validated: bf16 2.4% faster, fp32 neutral, ZERO >10% regressions, every shape
>= old ramp except +4.2% nick (softmax fp32 1024). 9 reduction tests pass. Gate A re-firing on v2.
LESSON: validate the flip-axis at MID + EXTREME (rnumel up to the band edges), not just my sweep's
8192-element ceiling. The bigger middle/wide-band bf16 wins (softmax w1 to 7KB) CONFLICT with fp32
two-reduction kernels at the same bytes → need a kernel-structure-aware signal, deferred.

## (v1, superseded by v2 above) byte-keyed warps ramp (Candidate C2) — KILLED by Gate A
`_num_warps` now keys on BYTES (size_hint*itemsize) not elements:
  <=2048->1, <=4096->2, <=6144->4, <=16384->8, <=98304->16, else 32.
VALIDATION (CUDA-graph, vs current ramp baseline, all 7 measured kernels × fp32/bf16/fp16):
- bf16: 4.3% faster overall (geomean C_vs_cur=0.957), fp32 neutral (0.9931), zero >10% regressions.
- Worst single-shape regression: softmax fp32 (8192,1024) +4.2% (w2 vs w4) — within 10% bound, net-pos.
- Captures CE/kl_div bf16 wide-V (w32→w16, was ~50% regret) AND softmax/norm narrow-N (w1/w2, up to 21%).
- Gate F PASS (mechanism: cross-warp reduction shared-mem overhead at tiny per-warp work, ncu-confirmed).
- bf16==fp16 (softmax itemsize=4 both; sum/CE/welford itemsize=2 both) — transfer hypothesis holds.

CORRECTNESS after edit (gate, vs at-dtype eager): rms/ln/softmax/CE/kl/jsd/long_sum all PASS at all 3
dtypes. The 5 "fails" (sum bf16/fp16, welford bf16/fp16, kl_div fp16) are PRE-EXISTING kernel-authoring
floors, NOT caused by the edit:
- sum bf16 max_abs=1.5 at BOTH old-w4 and new-w2 (warps-independent kernel floor; sum reduces at input
  width — ITER1). welford bf16 ~0.08, kl_div/welford fp16 NaN at ALL warps (pre-existing fp16 issue).
- Confirmed pre-edit: sum & welford bf16 already acc-failed at the original ramp (bf16_cg_chunk1/2.json).
- The warps change only jiggles rounding within already-failing kernels (sum fp16 0.19→0.25). NOT a new
  failure. ROOT CAUSE = input-width accumulator (sum L45, welford L61 chunk*chunk); fix = upcast before
  reduce (like rms_norm), parallel to the jsd fix. DECISION D3: characterize as kernel-authoring; the
  warps perf lever must not be held hostage to an already-broken kernel's rounding.

## PER-SHAPE STATUS (bf16) — live
- softmax/norm narrow-N: FIXED by EDIT#1 (byte ramp, w1/w2). CG re-bench confirming.
- CE/kl_div bf16 wide-V over-warp: PARTIALLY fixed by EDIT#1 (w32→w16 where bytes≤96KiB); the >96KiB
  shapes (50257+ at bf16) still w32, deferred (need re-read-aware top, or wider w16 band risks fp32).
- long_sum few-row huge-N: (8,2097152) CODEGEN-BOUND EXEMPT ✓; (48/64/96,...) oracle pending
- welford narrow/mid-N: OPEN (not cap-driven; needs answer key)
- softmax (8192,896): OPEN (dtype-amplified narrow-N; oracle next)
- layer_norm low-M N=6k-10k: OPEN (small)
- CE/kl_div (4096,151936): OPEN (small)
- rms_norm (16384,896): OPEN (small, ~0.95)
- jsd: BROKEN (correctness)
- everything else: BEAT TC ✓ (CG-confirmed)

## PER-SHAPE STATUS (fp16)
(pending)

## TRIED & REJECTED (dead ideas — do not repeat)
- **[ITER1 REJECTED] "welford bf16 caps mis-sized 2x → force fp32-equiv narrower tiles"** — the
  headline dtype thesis, FALSIFIED by matched-lever A/B (welford_cap_ab.py @ 9cd94ba7). Forcing the
  fp32-equivalent tiles `[1,8192,2048]` at bf16 is SLOWER not faster: f32eq_vs_cur median **0.957x**
  (0.88–1.01), i.e. the CURRENT wider bf16 tiles win/tie on all 7 divergent shapes. WHY: the resident
  reduction tile is the bf16 LOADED CHUNK (2B/elem), not the fp32 accumulator (the fp32 count/mean/M2
  are tiny per-row SCALARS, not tiles). So `cap // itemsize=2` correctly sizes a wider tile and the
  wider tile is good. The cap is faithful as-is. Evidence: /tmp/welford_cap_ab.json. → welford's
  tc-losses are NOT cap-driven; the bf16 byte caps are CORRECT.
- run3 rejections still valid: ÷num_reduction_ops divisor FAILED fact-integrity; welford apply-cap
  8192→16384 REJECTED; "narrow-N one warp lever" FALSIFIED.

## REFRAME (after ITER1) — where the REAL bf16 work is
The dtype-task premise "byte caps mis-sized 2x at bf16" is **falsified for the resident-tile caps**
(welford), because those tiles are at input width and a wider bf16 tile is beneficial. The genuine
bf16 gaps are in the **config-INVARIANT** norm kernels (rms_norm/layer_norm), proven dtype-specific:
| shape | rms_norm fp32 G | rms_norm bf16 G |
|---|---|---|
| (2048,4096) | 0.996 (tie) | **0.83** |
| (2048,6144) | 0.987 (tie) | 0.87 |
| (2048,7168) | 0.987 | 0.96 |
| (16384,896) | 0.982 | 0.95 |
Same byte-identical config, but bf16 opens a real gap fp32 doesn't have. MECHANISM HYPOTHESIS: bf16
halves memory traffic → these mem-bound norms become relatively more compute/launch-bound, so the
fp32-optimal warps/tiling is no longer bf16-optimal. The seed keys warps on `rnumel` (element count,
dtype-invariant) — but the right lever may need a dtype-aware (bytes-based) signal. Oracle answer-key
pending (rms_norm 2048,4096 + 2048,6144 bf16 full).

## DEFERRED HARD-PILE
- long_sum huge-N tail bf16 (0.56–0.92) — verify codegen-bound via fresh oracle (anti-giving-up).
- welford bf16 narrow-N tc-losses (896/5120/7168) — NOT cap-driven (ITER1); needs its own answer key
  (quick oracle unreliable — picks tiny blocks losing to tc; needs full or a different approach).
- welford bf16 ACCURACY ~0.07 max_abs (vs torch 0.008) — kernel sums chunk at bf16 (sum_x/sum_x2
  input-width); kernel-authoring, not heuristic. Characterize for the report.

## DECISIONS LOG (autonomous choices for later human review)
- D0: Treating this as fresh dtype climb on inherited baseline per dtype-task.md. New notebook
  `_lab/dtype_notebook.md` + ledger key `dtype`; run3 lineage preserved untouched.
