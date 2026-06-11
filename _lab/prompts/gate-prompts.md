# GATE PROMPTS — verbatim adversarial frames for the high-stakes gates

These are **copy-paste prompts** for the gates whose value is *independence + adversarial framing*
(method §5), plus the key ephemeral-helper frames (try-harder, completeness-critic) at the end. For
these, do **NOT** improvise the wording: how you phrase the ask is exactly how you bias the verdict.
The rule:

> **Fill the `{SLOTS}`. Do not edit the prose around them. Never paste your own conclusion** — give
> the gate the *claim* and the *receipts to reproduce it*, never "I found a 1.3× win, confirm it."
> Spawn each gate **fresh** (no rapport), brief it **neutrally on an immutable commit SHA**, and
> require it to **write its verdict object to the ledger AS-RETURNED before you read it** — you
> cannot launder a FAIL by re-narrating it.

**Fan-out / serial-GPU (method §6 invariant):** every gate's *analysis* fans out by default (token
cost is not a constraint); but **GPU re-measurement is foreground-serial** — the driver runs ONE
authoritative re-bench and hands the SAME numbers to the N analytical skeptics. Never fan out N
concurrent benchers (it corrupts the medians the gates exist to trust). A gate with an analysis part
AND a re-measure part splits along that line.

The lower-stakes checks (correctness gate, config-actually-ran, the behavior recorder) are
mechanical and don't need a scripted frame — run them as plain steps.

---

## GATE A — Adversarial verify (anti-OVER-claim). Fire on EVERY claimed win, before banking.
Spawn **N≥3 independent** copies of this, each a fresh context. Kill the claim on **majority
refute**. GPU re-measurement is **shared + serial** (see the requirements below) — not one bench per
skeptic. (Absorbs the former results-referee: independent own-script reproduction is now an axis here.)

```
You are an independent skeptic. Your ONLY job is to REFUTE the claim below. You do not know who
made it and you owe it no benefit of the doubt. Default to refuted=true: if you cannot, with your
OWN measurements/analysis, positively confirm the claim is real, reproducible, correctly measured,
and general, you must return refuted=true.

CLAIM (verbatim, do not assume it is true): {CLAIM — e.g. "seed config X makes kl_div(2048,50257)
bf16 beat torch.compile by 1.31x"}
IMMUTABLE COMMIT: {SHA}
THE AUTHORITATIVE RE-BENCH (the driver's single serial measurement — when Gate R ran on this edit, reuse
the focal cell from its step-4b sweep rather than commissioning a duplicate): {per-shape medians + spread
+ the normalized config that ran}
HOW TO REPRODUCE (the receipts, not my conclusion): {EXACT COMMAND(S) + the curriculum shapes +
dtype + the normalized config that is supposed to have run}

Attack it on every axis and report what you find:
1. NOISE / FABRICATION — scrutinize the authoritative re-bench: fixed seed, accuracy ON, GPU idle,
   median clearly beating run-to-run spread? Is the delta inside noise (lift M / use the in-process
   ratio if near the ~25µs floor)? Report the numbers.
2. MEASURING THE WRONG THING — was a grad graph built (autograd-wrapper overhead)? dynamo reset per
   shape? was the SEED config actually the one timed (check the normalized running config), or a
   fabricated off-frontier config (e.g. warps varied while block_sizes pinned to [1])? same dtype on
   both arms?
3. INDEPENDENT REPRODUCTION (the absorbed results-referee axis) — the delta MUST be reproduced from an
   INDEPENDENTLY-AUTHORED script: a fresh-context agent writes it from the harness primitives (NOT the
   worker's script). The **driver then runs that script foreground-serial** on the GPU; the parallel
   skeptics never touch the GPU (that would fan out measurement, §6). The independence that matters is
   *who wrote the script*, not who pressed run. Does the own-script number match the authoritative one?
4. OVERFIT — re-run on a HELD-OUT shape (val) AND an OFF-FOCUS kernel the edit shouldn't touch. Did
   anything regress?
5. KERNEL-IDENTITY SMUGGLING — does the rule fire via a faithful WORKLOAD PROPERTY, or is a constant
   secretly fencing exactly one kernel's shapes? Construct a hypothetical other kernel in the same
   regime: would it correctly get the same treatment?
6. METRIC GAMING — was a tolerance loosened, a shape dropped, or an aggregate (geomean) used to hide
   a per-shape loss?

GPU RULE: the analytical skeptics fan out **GPU-free** over the driver's authoritative numbers and
launch NO benches. The one independent own-script reproduction (axis 3) is authored by a fresh agent
and **run by the driver, foreground-serial** — never inside the parallel fan-out. N skeptics each
benching would fan out the GPU and corrupt the medians. (Re-measurement beyond that one reproduction is
reserved for a high-stakes claim — a fact changed / counter-intuitive direction / claimed beat-tc — and
even then the driver runs each serially.) The MID+EXTREME flip-axis sweep is NOT your job — it is the
Regression-referee's (Gate R).

Record your verdict object to the ledger AS-RETURNED before reporting: {refuted: bool, axis_that_
failed, your_raw_numbers, repro_command, reasoning}. PASS only means: real, reproducible, measured
correctly, generalizes, no identity branching, no gaming.
```

---

## GATE B — Anti-giving-up (anti-UNDER-claim). Fire on ANY stop/ceiling/noise/done/stuck claim — AND on the *action* of hard-piling a shape as STUCK or moving off a still-below-floor shape.
This is the gate that keeps the run honest about *quitting too early*. A never-stop worker won't
*narrate* a stop, so the trigger is the ACTION: you may not hard-pile a shape as STUCK (unmeasurable /
codegen-ceiling / no-faithful-re-key), nor move off a still-below-floor shape, until this gate clears
the claim as non-premature. Give GPU access (serial).

```
Someone has claimed the work on a shape/kernel is finished or impossible — or is about to hard-pile it
as STUCK. Your job is to prove that claim PREMATURE. Default to "there is a next move": a stop/park is
only valid if you cannot find one.

THE STOP/PARK CLAIM (verbatim): {CLAIM — e.g. "rms_norm bf16 wide-N is at a ceiling, seed≈oracle,
can't do better" / "this gap is just noise" / "no clean rule exists" / "hard-pile as stuck: no
faithful re-key found"}
IMMUTABLE COMMIT: {SHA}
RELEVANT SHAPES + CURRENT NUMBERS: {per-shape seed / oracle / tc latencies + the configs}

Do all of the following before accepting any stop/park:
1. RUN A FRESH FULL ORACLE on the shape(s) in question (not a cached one, not a quick one). Read the
   answer key: does the oracle's winning config differ from the seed? Those differing fields are
   the next experiments — hand them back.
2. CHECK THE CLAIM AGAINST THE FLOOR (§3). A stop/park is NEVER valid on a **below-floor** shape
   (`G < 0.75` vs tc, or `seed < 0.75 × oracle` if codegen-bound) UNLESS one of the bounded "stuck"
   reasons genuinely holds (unmeasurable even after lifting M; no faithful re-key after try-harder's K
   attempts; or a true codegen ceiling). A **"codegen ceiling"** (the claim tc is unreachable) only
   *retargets* the floor from tc to `0.75 × oracle`; it NEVER exempts the shape, which stays on the
   worklist until `seed ≥ 0.75 × oracle`. It is valid ONLY if the FRESH oracle **can't beat tc** AND
   the oracle run is verified real (converged, not truncated/OOM/mid-search). An **overtime** "nothing
   more reachable" claim is valid only if `seed ≈` the fresh converged oracle. If the oracle is faster
   than the seed and the seed is below its bar, it is NOT a ceiling — perf is on the table.
3. A "just noise" dismissal needs a noise-ROBUST disproof: lift M to get above the noise floor, or
   use the in-process seed/oracle ratio (both timed identically). If the gap survives, it's real.
4. A "no clean rule / contradictory / a fix also regresses peer kernel X" claim: check X's ACTUAL
   code branch — it may be STRUCTURALLY excluded (a different track/band) and unable to regress.
   And measure EVERY firing shape, not the hand-picked ones. The "no rule" is usually an unmeasured
   shape or a structurally-impossible objection.
5. Try attacking from a DIFFERENT workload property than the one already tried (and confirm try-harder
   was actually run if the claim is "no faithful re-key").

Record your verdict AS-RETURNED: {park_is_valid: bool, fresh_oracle_result, the_next_experiment_if_
any, reasoning}. Only return park_is_valid=true if 1–5 all came up empty. On a valid park, the shape
goes to the hard-pile tagged with its reason and one line to the human-review queue.
```

---

## GATE D — Fact-gate (doctrine + faithfulness). Fire on any NEW/CHANGED fact field a heuristic reads, a new fact, or any threshold/constant a branch compares a fact against.
Analysis only (no GPU); fresh + independent; read committed source, do NOT author kernels. Answer BOTH
parts in one spawn (ingest the fact code once). Default to refuted. (Replaces the former Gate D
fact-integrity and Gate G fact-doctrine — one gate, both checks. The doctrine it enforces is method §2.)

```
A heuristic reads a fact field that was newly added or changed (or a branch compares a fact against a
threshold/constant). Your ONLY job is to REFUTE it — on EITHER part. Default to refuted=true.

THE FIELD/FACT: {name, which fact it lives on, the FULL population code}
THE FACT'S KIND as claimed: {walker = per-op/per-entity, may walk the graph | derived = per-kernel-class, must NOT walk}
THE BRANCH that reads it: {the branch + exactly which SLICE it consumes + the threshold/constant}
IMMUTABLE COMMIT: {SHA}

=== PART 1 — DOCTRINE (walker/derived placement + walker-field generality) ===
1. NO WALK IN A DERIVED FACT (bright line). If the field lives on a derived fact (ReductionFact-like,
   kernel-specific), does ANYTHING in its population touch the graph — iterate `device_ir.graphs`,
   trace `node.users`/`node.args`, call `_classify_load_dataflow`, re-walk a rolled subgraph, inspect
   IR? ANY graph access ⇒ REFUTE. A derived field must be a pure derivation over walker facts +
   trivial structural reads (`block_id`, `size_hint`, `block_sizes`, `static_rnumel`).
2. THE WALK BELONGS ON A WALKER FACT, ONCE — AND PREFER AN EXISTING ONE. If graph info is needed, it
   must come from a walker fact (`MemoryOpFact`, `AccumulatorFact`, or a new walker fact) whose walk is
   folded into the SINGLE collector pass. Two failures ⇒ REFUTE: (a) a second/bespoke graph traversal
   (a fresh ad-hoc walk, a per-config re-walk) instead of the one pass; (b) a NEW walker fact created
   when the field would fit an existing one (proliferation is a cost — argue the entity is unmodeled).
3. WALKER-FIELD GENERALITY (the different-consumer test). Name a concrete, plausible OTHER consumer (a
   different reduction axis, a different kernel class/band, a register-pressure heuristic) that would
   read a DIFFERENT slice of this same field. If none can exist, it is under-general. Does the field
   bake in the consumer's identity/axis (e.g. `is_reread_for_my_rdim: bool`) instead of exposing the
   raw property and letting the DERIVED fact specialize? Specialization on the walker field ⇒
   under-general. Calibrate: `reductions_fed` (per-axis, not `feeds_my_reduction`);
   `indexed_block_ids`/`inner_extent` (raw shape provenance); `AccumulatorFact.dim_block_ids` +
   `itemsize` (reduction-agnostic; last-dim==rdim match in the reader). Verdict: general |
   justified-specific (no general form yet AND that is evidenced + logged) | under-general (REFUTE).
4. SOUND PROVENANCE. Genuine compiler provenance (`resolve_block_id`, real block-ids), not a guess;
   the field is SOUND. Do NOT bless NEW accepted-unsoundness (a guess dressed as provenance, or a
   config-dependent value stored as config-free ⇒ REFUTE). The ONLY tolerated pre-existing unsoundness
   is the config-free eviction-index slot (`MemoryOpFact.eviction_index` → `reread_eviction_index`) —
   a known TODO to FIX, not a precedent. (A documented fallback that faithfully reproduces the real
   property where provenance does not resolve — e.g. a size-match — is fine; an unflagged shortcut is not.)

=== PART 2 — FAITHFULNESS (divergence test on the fact AND the threshold) ===
5. FACT DIVERGENCE. Construct (in the abstract, reusing existing probe kernels — do NOT author new DSL
   kernels) a case where the LAZY PROXY and the REAL property DISAGREE. If the fact tracks the proxy
   rather than the real property it FAILS (this is how `num_load` and `num_reduction_ops` were
   falsified — observationally identical on the curriculum, wrong on a divergence kernel). Also: is the
   fact style-independent (not fooled by how the kernel is written)? Does a branch actually READ it (a
   fact no branch reads must be cut)?
6. THRESHOLD DIVERGENCE. A faithful fact can be read UNFAITHFULLY: a continuous quantity becomes a
   disguised dtype/identity fence the moment it is COMPARED TO A CONSTANT that splits its real
   value-set along dtype/kernel lines (e.g. an itemsize-like field whose only values are {2,4}, gated
   `<= 2`, is just `if dtype is half` wearing a faithful name). Construct two workloads with the SAME
   value of the real property but a DIFFERENT dtype/kernel; the branch must decide identically. If it
   diverges at equal real-property, the threshold keys on the incidental thing — FAIL. Faithful use
   keeps a dtype/identity-correlated quantity a FACTOR inside a hardware-unit budget (bytes = elems ×
   itemsize, occ = grid_rows // num_sm), never the operand of a literal comparison; any rationale that
   restates as "fires only for / excludes the dtype (or kernel) D" IS the fence.

OUT OF SCOPE (route elsewhere): whether the resulting LEVER overfits the curriculum / belongs in the
core → GATE H; whether an edit regressed a shape below floor → GATE R. This gate is ONLY about where
the walk lives, how general the field is, and whether the fact + its threshold are faithful.

Record AS-RETURNED: {doctrine: pass | refuted(rule, fix), generality_verdict:
general|justified-specific|under-general, faithfulness: faithful_property | scoped_deferral |
disguised_fence, the_fact_divergence_case, the_threshold_divergence_case, what_it_tracks, reasoning}.
PASS only if BOTH parts pass. A `scoped_deferral` (excluding a dtype/kernel because NO faithful rule
exists — e.g. a non-monotonic optimum left to the autotuner) is acceptable ONLY if that "no rule"
claim is itself evidenced and logged AS a deferral, never blessed as faithful. Do NOT over-correct
into demanding a general dataflow framework where one specific faithful fact suffices.
```

---

## GATE E — Overfit guard + TEST-firewall keeper. The PORTFOLIO gate (a must, alongside A, B, H, and R).
Distinct from A, which checks ONE win: this guards the heuristic-as-a-whole against silently
memorizing the curriculum — the failure that *looks done* (run 2 declared victory on a good geomean
while individual shapes lagged the oracle). Two duties:
- **Periodic (during the climb):** audit the accumulated heuristic for curriculum-overfit. Analysis
  only (no GPU); reuse the recorded configs.
- **Firewall (once, at freeze):** this gate is the **SOLE reader of the TEST split, read EXACTLY
  ONCE**, on the frozen champion. The worker/orchestrator must never have benched TEST. Give GPU for
  the freeze read. **This single read must NOT be fanned out** — a single reader, a single TEST bench;
  fanning it out means multiple TEST benches, breaking the firewall.

```
You guard against the heuristic OVERFITTING the curriculum — the failure where every individual
edit passed its gate, the aggregate looks great, and the heuristic has quietly memorized the
training shapes. Default to "overfit until shown otherwise."

MODE: {PERIODIC during-climb audit  |  FREEZE firewall read}
IMMUTABLE COMMIT (frozen champion for FREEZE mode): {SHA}
THE HEURISTIC'S CONSTANTS + FACTS + which curriculum shapes each branch fires on: {the recorded
per-shape configs + the constant list}
TRAIN vs VAL/TEST split definition: {the splits — and for FREEZE mode, confirm TEST has NEVER been
benched before this read}

Check:
1. CURRICULUM FENCES — does any constant/threshold sit exactly at a curriculum shape's boundary
   (a value that fires for precisely the training shapes and nothing structurally adjacent)? A
   threshold tuned to land between two specific N values is a memorized fence, not a property.
   List every constant and the shape(s) it gates; flag any that track shape-identity not workload.
2. TRAIN↔HELD-OUT GAP — measure (or, periodic mode, reason from recorded configs) the per-shape
   floor (`G ≥ 0.75`) and the per-(kernel, dtype) geomean (`≥ 0.85`) on VAL and an OFF-FOCUS kernel.
   Report the gap between train performance and held-out performance as a FIRST-CLASS number, never
   buried in a single matrix-wide geomean.
3. FREEZE-MODE ONLY — read TEST exactly once: report per-shape seed/oracle/tc, the train↔test gap,
   and whether any TEST shape falls below the 0.75 floor (or drags its kernel's geomean below 0.85)
   where train shapes clear it (the overfit signal). State plainly that this was the single sanctioned
   TEST read.
4. INTERPOLATION CHECK — is every VAL/TEST shape's N-regime actually covered by a TRAIN shape, so
   held-out is interpolation (fair) not extrapolation into an untrained regime (which would mask
   overfit as "expected")?

Record AS-RETURNED: {overfit: bool, train_vs_heldout_gap, fences_found:[...], test_read_results (freeze
only), reasoning}. PASS only if held-out tracks train within ε AND no constant is a shape-identity
fence. A passing per-(kernel, dtype) geomean with a held-out shape below the 0.75 floor is a FAIL.
```

---

## GATE F — Mechanism gate ("explain the win"). Fire ONLY on a SURPRISING / counter-intuitive win.
NOT every edit — only wins that contradict the expected direction (e.g. a NARROW reduction wanting
FEWER warps, a cap that helps where theory says it shouldn't). A number can be real + reproducible
(passed A) yet MIS-UNDERSTOOD, so the rule you generalize from it misfires off-curriculum. Cheap
gate, narrow trigger. Analysis/profiling; give GPU if ncu/IR inspection needs it (serial).

```
A performance win has been measured and reproduced, but its DIRECTION is counter-intuitive. Before
it can be banked into a generalizable rule, you must explain the MECHANISM — WHY this config is
faster here. "It just measured faster" is NOT acceptable: an unexplained win keys the fact wrong.

THE SURPRISING WIN (verbatim): {CLAIM + why it is counter-intuitive — e.g. "rms_norm narrow-N is
faster with num_warps=4 than 8, opposite the usual ramp"}
IMMUTABLE COMMIT: {SHA}
THE TWO CONFIGS + their reproduced latencies: {configs + numbers}

Find the mechanism using ncu / generated Triton / IR — not speculation:
1. What HARDWARE resource explains it (occupancy, register/SMEM pressure, the reduction-tree cost,
   L2 residency, wave quantization, grid saturation)? Show the metric that moves with the win.
2. Does the explanation predict a BOUNDARY — the workload property at which the win should reverse?
   Name it. A real mechanism generalizes to a rule keyed on that property; a coincidence does not.
3. Sanity-check the boundary: at a shape on the OTHER side of it, does the win correctly disappear/
   reverse? If you cannot find a boundary the rule would overfit — say so.

Record AS-RETURNED: {mechanism_found: bool, the_resource_and_metric, the_predicted_boundary, the_
property_to_key_the_fact_on, reasoning}. PASS only if there is a mechanistic explanation that yields
a workload-property the fact can key on. If no mechanism: do NOT bank as a general rule (at most a
logged, shape-scoped observation flagged for revisit).
```

---

## GATE H — Generality gate (KEEP / DEFER / REJECT a lever for the generalizable core). A MUST — fire on EVERY proposed or existing lever / branch / constant before it enters the core PR.
Analysis only (no GPU) — reuse the recorded per-shape numbers + the lever's gate/constant. It is NOT a
re-measurement (that's A), NOT a premature-stop check (B), NOT the whole-heuristic overfit audit (E),
and NOT the regression check (R): it adjudicates a SINGLE lever's place — KEEP in the core, DEFER to
the removed-heuristics-log (re-add retained), REJECT-and-re-key, or BORDERLINE (record the tradeoff
for the human's later judgement) — by weighing **key-faithfulness × magnitude × realism × downside ×
complexity**. The bar is the maintainer's generalizability standard **encoded entirely in rules 1–7
below** — judge by those rules; you are NOT given the maintainer's own keep/defer calls to copy: a
small principled core; magnitude buys some overfit tolerance; **an unfaithful KEY is never bought**;
and genuine conflicts land at BORDERLINE, not reflexive DEFER. This gate will not be 100% — gray area
is expected, and on it the gate RECORDS the tradeoff for the human's later review rather than faking
certainty.

```
You decide whether the lever below belongs in the generalizable-core heuristic. Return exactly one of
KEEP / DEFER / REJECT / BORDERLINE. Default to the FAITHFUL bar — do not KEEP unless the lever earns it
on every axis — but do NOT over-DEFER a genuine, faithful, bounded win either.

THE LEVER (verbatim): {what it does — the branch/constant + the config it emits}
THE GATE/KEY it fires on: {the exact workload quantity + threshold it branches on}
MEASURED BENEFIT (per-shape, never a geomean): {which shapes, how much, which dtypes; is it above the
   ~5% noise floor? is there an articulable reason/mechanism — non-ncu "makes sense logically" counts?}
MEASURED / EXPECTED REGRESSION: {which shapes, how much, on-curriculum AND off-curriculum}
SHAPE REALISM (the Regression-referee's verdict, NOT your own assertion): {for the shapes it HELPS and
   the shapes it HURTS — realistic real-workload shapes, or synthetic/diabolical (e.g. 2 rows of 16M
   elems, a box around one curriculum shape)?}
ADDS/READS A FACT?: {none | the fact + how/where it is computed}
CURRENT CORE: {how many levers already; is each easy to reason about?}
IMMUTABLE COMMIT: {SHA}

Adjudicate in order; the first HARD STOP wins, otherwise weigh the pulls (rule 7):

1. KEY FAITHFULNESS — the one hard line (deep test = the Fact-gate, GATE D). FAITHFUL keys:
   bytes/footprint budgets, occupancy, load/iteration/trip/accumulator COUNTS, extent, and raw SHAPE
   dims (M, N). UNFAITHFUL keys: kernel identity (kernel_name); a BARE dtype literal (`itemsize == 2`,
   or any dtype switch with no faithful reason — "if you exclude fp32 there MUST be a faithful
   reason"); or an OP-PATTERN used as a kernel-class proxy (logsumexp/softmax detection) when the win
   is only modest. An unfaithful key ⇒ it cannot enter the core IN THIS FORM. **Magnitude NEVER buys
   an unfaithful key.** If the win is high-value, return REJECT and invoke TRY-HARDER (re-key mode) to
   re-key it on a workload property (bytes/occupancy/count/shape); accept the unfaithful form ONLY on a
   solid, evidenced proof that no workload gate is possible (rarely satisfiable — distrust the excuse).
   NOTE: a raw SHAPE box (`M<=8 and N>=1e6`) is a FAITHFUL key — shape is a real property, not an
   identity fence — so it is NOT rejected; but if the box wraps ~exactly one curriculum shape, nudge
   toward a more general gate (e.g. occupancy `grid_rows // num_sm`) rather than blessing the tight box.

2. MAGNITUDE × NOISE × REASON. Within-noise (≲5%) AND no articulable reason ⇒ DEFER (faithful +
   never-negative is NOT sufficient; there must be a reason AND a win above noise). A real win (≳5%)
   with a reason that "makes sense logically" ⇒ KEEP-eligible (ncu NOT required). A large win
   (≳15–40%+) is a strong KEEP-pull even behind a soft op-pattern gate ⇒ KEEP-but-FLAG-as-specific.
   EXCEPTION: merely tuning the knobs of an EXISTING faithful lever (e.g. moving a ramp breakpoint) ⇒
   KEEP even unexplained, if above noise with no regression; RESTRUCTURING it (an unexplained
   non-monotonic carve-out) ⇒ DEFER. (These %s are rough, not bright lines — see rule 7.)

3. CATASTROPHE PRIORITY. Rescuing a below-floor shape — `G < 0.75` vs tc, or `< 0.75 × oracle` if
   codegen-bound (the §3 disaster line; the deeper the cliff, the higher the priority) — up to its
   FLOOR is the single highest-value justification and OUTRANKS overtime win-chasing (pushing an
   already-above-floor shape toward beating tc / oracle parity). It pulls even a narrow shape-boundary
   case to KEEP (then nudge toward a general gate). It does NOT override rule 1 — an unfaithful key
   still needs try-harder, not a pass.

4. DOWNSIDE × REALISM (master axis, BOTH directions; realism = the Regression-referee's verdict).
   ACCEPTABLE: a regression confined to synthetic/diabolical shapes (a real regression is far worse
   than a diabolical one); OR a **bounded, principled, net-positive regression on REALISTIC shapes**,
   judged by the **§3 backstop's net-positive-trade rules** — the benefit outweighs the cost over the
   FULL affected set, it's ≤~10% (comfortably ~5%), it stays **above the shape's floor**, and **no
   faithful separator exists** (fire anti-giving-up first — a separator that gets BOTH always beats a
   trade). DEFER when: a separator is achievable-but-unbuilt; the downside is unbounded/unknown
   off-curriculum with no separating gate; or it leans on a deliberately UNSOUND / "shady" tactic
   (e.g. tolerating spill for a win) — a cleaner/gated version may return (DEFER, not REJECT). A
   regression that pushes a REALISTIC shape **below its floor** is a disaster → reject (§3), never a
   trade. A benefit ONLY on diabolical shapes ⇒ not worth the complexity (DEFER/BORDERLINE).

5. FACT HYGIENE (if it adds/reads a fact; walk-LOCATION/generality = the Fact-gate, GATE D). New
   FAITHFUL facts are WELCOME — compute MORE of them, future levers may read them. BUT: a fact needing
   RUNTIME information ⇒ REJECT (configs carry no runtime info; it must be computable from source / the
   input fx-graph at compile time). A fact computed by a brittle SECOND graph re-walk (drift risk) ⇒
   DEFER (fix = fold it into the canonical single fact-build pass, then it KEEPs).

6. COMPLEXITY (portfolio). Judge REASON-ABILITY, not COUNT — many easy-to-reason levers are fine; flag
   only levers that are individually complicated, interact confusingly, or flip a near-universal default
   / bundle many knobs at once (each deviation from a strong default — e.g. `pid='flat'` — needs its own
   justification; sprinkling in an extra unusual knob "because it helps" is a yellow flag). Mild
   redundancy with an existing lever is not, by itself, fatal.

7. WEIGHT & GRAY ZONE (the calibration — read this LAST and let it temper 1–6). When axes CONFLICT,
   default to BORDERLINE: RECORD the specific tradeoff to the ledger + the human-review queue for the
   human's later judgement, and provisionally DEFER (the reversible default — logged to the
   removed-heuristics-log, re-add retained) so the run keeps moving. NEVER ask, block, or wait on the
   human — this is an unattended run (method §6.0). Let a strong positive (catastrophe-rescue, large
   REALISTIC win, a clean faithful fact, an easy-to-reason lever) tilt to KEEP. Do NOT reflexively
   DEFER on conflict — most genuine conflicts are BORDERLINE, not DEFERs. There is always gray area;
   this gate is not expected to be right every time. When genuinely split, return BORDERLINE and RECORD
   the tradeoff for review — never manufacture false certainty.

DEFER vs REJECT: REJECT = an unfaithful KEY that cannot be fixed without changing what it branches on
(identity / bare dtype literal) — pair with TRY-HARDER (re-key mode) when the win is high-value. DEFER =
faithful-or-fixable but not ready (within-noise, restructures a lever, realistic regression with no
gate, brittle re-walk) — log it to the removed-heuristics-log WITH a re-add recipe; nothing is lost.

Record AS-RETURNED before reporting: {verdict: KEEP | DEFER | REJECT | BORDERLINE, key_faithfulness:
faithful | identity | dtype-literal | op-pattern, magnitude_class: noise | real | large |
catastrophe-rescue, reason_present: bool, downside: bounded | realistic-regression | unbounded |
diabolical-only, fact_hygiene: n/a | clean-faithful | runtime-impossible | brittle-rewalk,
complexity_note, conflicting_axes:[...], action: keep-in-core | defer-to-removed-log |
try-harder(re-key on {property}) | fold-into-fact-build | record-borderline-for-human-review (provisionally defer, keep climbing), reasoning}.
KEEP only if: the key is faithful, the win clears noise WITH a reason (or is a catastrophe-rescue / an
existing-lever tuning), and the downside is bounded or realistic-only-on-diabolical. Otherwise DEFER,
REJECT (+ try-harder), or BORDERLINE per the rules above.

OUT OF SCOPE (route elsewhere): is the win real/reproduced → GATE A; is a STOP/park premature → GATE B;
is the fact/threshold faithful + the walk in the right place → the Fact-gate (GATE D); does the
heuristic-AS-A-WHOLE overfit the curriculum + TEST firewall → GATE E; explain a counter-intuitive win
→ GATE F; did an edit push a realistic shape below its floor → GATE R (regression-referee).
```

---

## GATE R — Regression-referee (disaster-avoidance). A MUST. Fire on EVERY edit that changed ≥1 emitted config.
Independent, fresh context. **Owns the realistic↔diabolical verdict** (so the realism call is not the
self-interested worker's). GPU re-measurement is foreground-serial. This is the disaster-avoidance bar
(§1 goal) as an adversarial gate — it replaces the worker's config-recorder *self-check* (which is "not
a pass/fail gate"). Its changed-cell sweep subsumes the former Gate C flip-axis sweep; and its per-cell
medians are the authoritative numbers for this edit — Gate A reads the focal cell from this sweep instead
of running a duplicate bench (§6.2 dedups the serial GPU).

```
You are an independent referee. Prove the edit below did NOT push any REALISTIC shape below its floor,
and adjudicate any regression. Default to "a regression is hiding until proven otherwise."

THE EDIT: {heuristic / fact / constant change}
BEFORE / AFTER: {two SHAs, or the two versions}
THE FULL ACTIVE MATRIX: {every kernel × shape × dtype × split incl. robustness}
THE FLOORS: {G ≥ 0.75 vs tc; or seed ≥ 0.75 × oracle for codegen-bound shapes}
IMMUTABLE COMMIT: {SHA}

1. SCOPE — run config_recorder over the FULL active matrix BEFORE and AFTER; enumerate the cells whose
   normalized config CHANGED. Byte-identical cells are perf-invariant (skip). A source/fact/normalize/
   lowering edit needs the `--triton` diff, not config-only. FULL MATRIX or it's a false all-clear.
2. RE-BENCH every changed cell to its floor — foreground-serial GPU, one at a time. changed ≠ win (a
   changed cell still earns measurement, never an assumed improvement). The MID+EXTREME sweep of any
   axis a config-cap flips on is included here (the cells along it that changed).
3. REALISM — for any shape that ends BELOW its floor, return a realistic↔diabolical verdict.
   DEFAULT-TO-REALISTIC: a real workload that occurs in actual models is realistic REGARDLESS of
   curriculum membership; only a genuinely synthetic/diabolical shape (e.g. 2 rows of 16M elems, a
   tight box around one shape) may sit below floor. NEVER launder a genuine regression as
   "unrealistic." This verdict is recorded per-shape and is the one Gate H and the §3 backstop read.
4. ADJUDICATE — a REALISTIC shape below its floor ⇒ REJECT (disaster). A regression that stays ABOVE
   floor is allowed only under the §3 net-positive-trade rules (bounded ≤~10%, net-positive over the
   FULL affected set, a workload-property reason, anti-giving-up fired for a separator first).

Record AS-RETURNED: {accept: bool, changed_cells:[...], below_floor:[{shape, G, realism}],
realism_verdicts:{shape: realistic|diabolical}, trade_assessment, reasoning}. Accept only if NO
realistic shape sits below its floor and every trade passes the §3 rules.
```

---

## HELPER — Try-harder (the general "escalate when stuck" agent). Fire on a MEASURABLE stuck signal, not a self-declared mood.
One escalation pattern with several MODES, by trigger: fan out N diverse attempts from the SAME
evidence (non-GPU → free under ultracode), then SELECT/synthesize the best (the N raw attempts never
land in the driver). BOUNDED in every mode: after K rounds with nothing new, the item is hard-piled as
"stuck" (Gate B clears that park) and written to the human-review queue — NOT re-triggered.

Modes:
- **hypothesis** — fire when M consecutive attempts on a shape were refuted/empty (stuck reasoning).
- **re-key** — fire on a Gate H REJECT (unfaithful key, win worth saving).
- **approach** — fire on a hard-pile / counter-intuitive item that resists the obvious approaches.

```
You are one of N independent attempt-generators (fresh context). The driver is STUCK and is escalating.
Produce the BEST single attempt you can; you will be pooled with the others and the best selected.

MODE: {hypothesis | re-key | approach}
THE STUCK ITEM: {the shape + its oracle/tc diff | the REJECTED lever + its unfaithful key | the
hard-pile item}
ALREADY TRIED-AND-REJECTED (do NOT repeat): {the rejected hypotheses/configs from the notebook}
FAITHFUL-PROPERTY MENU (for re-key): bytes/footprint budgets, occupancy (grid_rows // num_sm),
load/trip/accumulator counts, extent, raw shape dims (M, N).
ATTEMPT BUDGET: K (e.g. 3); rounds already spent: {n}.
IMMUTABLE COMMIT: {SHA}

Return ONE candidate:
- hypothesis mode: {candidate_workload_property, predicted_mechanism, config_implication}.
- re-key mode: a re-key of the SAME win onto a FAITHFUL property (state property + new gate/threshold),
  OR an evidenced proof that NO faithful gate is possible (rare — distrust the excuse; show the two
  workloads that force the unfaithful split). NEVER re-key onto an unfaithful property just to have an
  answer.
- approach mode: {a different attack — a different workload property, a different lever, a different
  field-diff reading}.

It is NOT self-certifying: a selected candidate re-enters the loop fresh (re-key ⇒ re-run the Fact-gate
+ Gate H; a hypothesis/approach ⇒ loop step 1). 

SELECTION (the driver, or a judge — never the N raw dumps): pick the 1–2 best; log untried candidates
as "candidate, not yet tried" (distinct from tried-and-rejected). If THIS was round K and nothing
beats what's tried, return empty → the item is hard-piled as "stuck" (Gate B clears it) + one line to
the human-review queue. Do NOT loop further.

Record AS-RETURNED: {mode, candidate | empty, property_keyed_on (re-key), must_rerun:[...],
rounds_spent, reasoning}.
```

---

## HELPER — Completeness-critic (loop-until-dry). Fire on a cadence — every K levers, whenever the below-floor worklist empties, and before any dead-end is accepted (NOT "near the end", which is undefined for a never-stopping run).
Analysis only (no GPU).

```
Find what the run has MISSED. Default to "something is uncovered."

THE STATE: {per-shape status table, the ledger index, the hard-pile, the human-review queue}
IMMUTABLE COMMIT: {SHA}

Hunt every gap: a dtype not swept; a claim banked but never independently reproduced; a cap/constant
never audited by the Fact-gate or Gate H; a shape under the noise floor never lifted; a hard-pile or
BORDERLINE item never revisited; a changed cell the Regression-referee never re-benched.

Record AS-RETURNED: {gaps:[{kind, specific_cell_or_claim, ledger_pointer}]}. Each gap is appended to
the worklist and either cleared or explicitly logged-and-skipped — never silently dropped. Re-run on
the cadence until K consecutive passes find nothing new (loop-until-dry).
```

---

### Note for the orchestrator
A **non-verdict** (a watchdog stall or API error after the analysis but before the verdict was
recorded) is **never a verdict** — re-fire a fresh completion, never bank or fail on it. And these
gates are *gates*, not advisors: on PASS you read the ledger entry's `{verdict, ledger-ref}` and
continue (do NOT re-read the full object into your context); only a FAIL — or a BORDERLINE/blocked-park,
which routes to the human-review queue and a provisional reversible default — changes what you do next.
Routing every PASS receipt back through your own context is a dominant context sink — let the ledger
hold them (§6.1).
