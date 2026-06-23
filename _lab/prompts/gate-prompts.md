# GATE PROMPTS — verbatim adversarial frames for the high-stakes gates

These are **copy-paste prompts** for the gates whose value is *independence + adversarial framing*
(method §5), plus the key ephemeral-helper frames (try-harder, refactor-critic, completeness-critic) at
the end. For these, do **NOT** improvise the wording: how you phrase the ask is exactly how you bias the
verdict. The priority the gates enforce is lexical (method §3): **faithfulness > generality >
performance** — Gate D (incl. its population/authoring part) and Gate A's identity axis are the F-gates;
Gate H (incl. BROADEN), Gate E, and the refactor-critic are the G-gates; all of them REJECT/return-work on
an edit regardless of how good its perf or aggregate number is. The rule:

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
GATE R'S FULL-MATRIX DIFF + SPLIT MAP (for axis 4): {Gate R's step-1 changed-cell list over the full
matrix, tagged by split (train/val/robustness — NOT test, which is firewalled to Gate E) and by
focal-vs-off-focus kernel}
HOW TO REPRODUCE (the receipts, not my conclusion): {EXACT COMMAND(S) + the curriculum shapes +
dtype + the normalized config that is supposed to have run}

Attack it on every axis and report what you find:
1. NOISE / FABRICATION — scrutinize the authoritative re-bench: fixed seed, accuracy ON, GPU idle,
   median clearly beating run-to-run spread? Is the delta inside noise (lift M / use the in-process
   ratio if near the ~25µs floor)? Is the working set ≤ L2 AND a plain-cudagraph device metric used? —
   if so the ratio may be L2-distorted (method §4 #9); demand a cold-L2 re-bench before accepting it.
   Report the numbers.
2. MEASURING THE WRONG THING — was a grad graph built (autograd-wrapper overhead)? dynamo reset per
   shape? was the SEED config actually the one timed (check the normalized running config), or a
   fabricated off-frontier config (e.g. warps varied while block_sizes pinned to [1])? same dtype on
   both arms?
3. INDEPENDENT REPRODUCTION (the absorbed results-referee axis) — the delta MUST be reproduced from an
   INDEPENDENTLY-AUTHORED script: a fresh-context agent writes it from the harness primitives (NOT the
   worker's script). The **driver then runs that script foreground-serial** on the GPU; the parallel
   skeptics never touch the GPU (that would fan out measurement, §6). The independence that matters is
   *who wrote the script*, not who pressed run. Does the own-script number match the authoritative one?
4. OVERFIT (analysis-only first — you do NOT bench; the fan-out launches no benches). From Gate R's
   full-matrix config-recorder diff (handed in below), inspect the HELD-OUT (val) cells + an OFF-FOCUS
   kernel the edit shouldn't touch: a byte-identical cell is provably perf-invariant (deterministic
   codegen) ⇒ no regression; a CHANGED one was already re-benched by Gate R ⇒ read its number. If analysis
   raises a perf CONCERN it cannot settle — a held-out/off-focus cell you suspect regressed, or a hunch
   the rule overfits the focal regime and would tank a specific held-out shape — do NOT refute on the
   hunch; FLAG it for GPU verification: a fresh agent authors a targeted bench of that exact cell/shape and
   the DRIVER runs it foreground-serial (never the fan-out — same mechanism as axis 3). The MEASURED result
   decides: confirmed regression ⇒ refute; not confirmed ⇒ drop the hunch. Record the hunch + its
   verification result.
5. KERNEL-IDENTITY SMUGGLING — does the rule fire via a faithful WORKLOAD PROPERTY, or is a constant
   secretly fencing exactly one kernel's shapes? Construct a hypothetical other kernel in the same
   regime: would it correctly get the same treatment?
6. METRIC GAMING (sharpened for the §3 aggregate-acceptance model) — was a tolerance loosened, a shape
   dropped, or an aggregate used to hide a per-shape loss? Was an acc-fail or NaN row folded INTO a
   geomean (was the aggregate gated on accuracy first, §4 #6)? AND the new aggregate-specific vectors:
   (a) **noise-floor majority manufacturing** — is the "majority of (kernel,dtype) cells up" built from sub-25µs
   within-noise wobbles counted as wins? Every claimed up/down must clear the ~5% noise band (median-of-N),
   else it is NEUTRAL. (b) **anchor drift** — is the "no >10% drop" measured against the rolling per-edit
   champion instead of the FROZEN champion (permitting ratchet-collapse)? (c) **generality-license
   laundering** — is this edit labeled "generality" to spend the 3–5% drop WITHOUT a named, measured
   off-train shape it actually lifts? No measured breadth ⇒ the license does not apply ⇒ the drop is a
   plain regression.

GPU RULE: the analytical skeptics fan out **GPU-free** over the driver's authoritative numbers and
launch NO benches. The one independent own-script reproduction (axis 3) is authored by a fresh agent
and **run by the driver, foreground-serial** — never inside the parallel fan-out. N skeptics each
benching would fan out the GPU and corrupt the medians. (Re-measurement beyond that one reproduction is
reserved for a high-stakes claim — a fact changed / counter-intuitive direction / claimed beat-tc — OR an
axis-4 overfit/regression hunch flagged for verification — and even then the driver runs each serially,
never the fan-out.) The MID+EXTREME flip-axis sweep is NOT your job — it is the
Regression-referee's (Gate R).

Record your verdict object to the ledger AS-RETURNED before reporting: {refuted: bool, axis_that_
failed, your_raw_numbers, repro_command, reasoning}. PASS only means: real, reproducible, measured
correctly, generalizes, no identity branching, no gaming.
```

---

## GATE B — Anti-giving-up (anti-UNDER-claim). Fire on ANY stop/ceiling/noise/done/stuck claim — AND on the *action* of hard-piling a shape as STUCK or moving off a still-below-floor shape.
Keeps the run honest about *quitting too early*. A never-stop worker won't *narrate* a stop, so the
trigger is the ACTION: hard-piling a shape as STUCK, or moving off a still-below-floor shape, until this
gate clears it as non-premature. Give GPU access (serial).

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
   (`G < 0.75` vs tc, or `seed < 0.75 × max-autotune` if it's a weird shape) UNLESS one of the bounded
   "stuck" reasons genuinely holds (unmeasurable even after lifting M; no faithful re-key after
   try-harder's K attempts; or a true codegen ceiling). A **"weird shape" claim** (tc is unreachable —
   even helion max-autotune can't beat it) only *lowers* the floor from `0.75 × tc` to `0.75 ×
   max-autotune`; it NEVER exempts the shape, which stays on the worklist until `seed ≥ 0.75 ×
   max-autotune`. It is valid ONLY if a FRESH **converged max-autotune can't beat tc** AND the run is
   verified real (converged, not truncated/OOM/mid-search). (A *quick*-autotune verdict is NOT enough
   here — a quick parity/loss vs tc is suspect and can flip at full; the weird-shape verdict always uses
   the converged max-autotune.) An **overtime** "nothing more reachable" claim is valid only if `seed ≈`
   the fresh converged oracle. If the oracle is faster than the
   seed and the seed is below its bar, it is NOT a ceiling — perf is on the table.
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

## GATE D — Fact-gate (doctrine + faithfulness + POPULATION). Fire on any NEW/CHANGED fact field a heuristic reads, a new fact, a threshold/constant a branch compares a fact against, OR a CHANGE TO HOW A FIELD IS POPULATED (a `device_ir.py` builder, or an inline derivation a branch keys on — e.g. a `bs.reduction`-style filter).
Fresh + independent; read committed source. Answer all THREE parts in one spawn (ingest the fact +
populator code once). Default to refuted. Parts 1–2 are analysis-only; **Part 3 (population) MAY author a
compile-only divergence kernel** (never GPU — see its BOUNDS). Doctrine = method §2; Part 3 is the fix for
the lucky-proxy populator class (the `bs.reduction` miss) the old field/threshold-only trigger missed.
**Lens (full FRAMING inside the prompt):** this gate guards the FAITHFULNESS of what the populator RECORDS
— not how broadly it fires (broad firing is GOOD; under-firing is a Gate-H BROADEN), nor whether the
heuristic's config is tuned (the heuristic's guess, Gate H). It kills a LUCKY PROXY; broad faithful firing
and an honestly-declared fallback pass, while the heuristic's may-be-wrong best-guess CONFIG is out of
scope (Gate H).

```
A heuristic reads a fact field (newly added/changed), a branch compares a fact against a
threshold/constant, OR the way a field is POPULATED changed (a builder, or an inline derivation a branch
keys on). Your ONLY job is to REFUTE it — on ANY part. Default to refuted=true.

THE FIELD/FACT: {name, which fact it lives on, the FULL population code — the builder AND any inline derivation a branch keys on}
THE FACT'S KIND as claimed: {walker = per-op/per-entity, may walk the graph | derived = per-kernel-class, must NOT walk}
THE BRANCH that reads it: {the branch + exactly which SLICE it consumes + the threshold/constant}
THE REAL PROPERTY the populator CLAIMS to compute, vs the PROXY it may actually compute: {e.g. "a 2-D tile reduced over dim-0 leaving [n]" vs "any block_size with bs.reduction set, which the reduction-dim allocator ALSO sets for hl.zeros accumulators"}
IMMUTABLE COMMIT: {SHA}

FRAMING (read FIRST — what this gate is and is NOT). The populator RECORDS a workload property; the
heuristic chooses a config. This gate polices the FAITHFULNESS of what is RECORDED — never how BROADLY the
populator fires, never whether the config it leads to is well-tuned. Three consequences:
- BROAD FIRING IS GOOD, not suspect. Building the fact for an unforeseen / off-curriculum workload is the
  DESIRED behavior (method §2 'prefer to fire'). Narrowing the builder so the "is-the-property-clean?"
  question never arises is UNDER-FIRING → a Gate-H BROADEN, NEVER how a populator passes here. Do not
  reward a narrow populator, and never refute one merely for firing where the property is unusual.
- A BEST-GUESS DEFAULT is out of scope. A sensible default config for a fact-regime the heuristic has no
  tuned branch for is the HEURISTIC's choice (Gate H) — not a value the populator records, so this gate
  never sees it. Keep the guess on the heuristic side.
- The ONE thing this gate KILLS is a LUCKY PROXY: a populator that records a correlate IN PLACE OF the real
  property it COULD have computed, presents it AS exact so a branch trusts it, and diverges SILENTLY
  off-curriculum. The populator's honest counterpart is a FAITHFUL computation — including a DECLARED
  fallback that still REPRODUCES the property where the primary provenance does not resolve (e.g. a
  size-match, flagged as a fallback). (The may-be-wrong best-guess for an unforeseen regime is a DIFFERENT
  object — it lives in the heuristic's CONFIG, bullet above, NEVER in a recorded fact value.) A proxy is
  MINIMUM generality (lucky on the curriculum, silently wrong off it); broad faithful firing is MAXIMUM
  generality (extends to the unforeseen on purpose) — opposite failures, and ONLY the proxy REFUTEs here.
THE DISCRIMINATOR (apply in every part below): could the populator have computed the real property HERE and
used a correlate instead (PROXY → REFUTE), or is the primary provenance genuinely unresolved and the
recorded value a DECLARED fallback that still REPRODUCES the property (FALLBACK → fine)? Firing the fact on
an unforeseen workload — or the heuristic later defaulting its config — is never grounds to refute;
substituting a correlate for a COMPUTABLE property is.

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
   a known TODO to FIX, not a precedent. (Kept distinct, per the FRAMING discriminator: a DECLARED fallback
   that still faithfully REPRODUCES the real property where the primary provenance does not resolve — e.g. a
   size-match, recorded AS a fallback — is sound and fine; the heuristic's may-be-wrong best-guess CONFIG for
   an unforeseen regime is a Gate-H matter, NOT a walker-fact value. FORBIDDEN: a guess dressed as exact
   provenance, or a SILENT shortcut that substitutes a correlate for provenance that WAS available — a proxy
   ⇒ REFUTE.)

=== PART 2 — FAITHFULNESS (divergence test on the fact AND the threshold) ===
5. FACT DIVERGENCE. Construct (FIRST in the abstract, reusing existing probe kernels) a case where the
   LAZY PROXY and the REAL property DISAGREE. If the fact tracks the proxy rather than the real property
   it FAILS (this is how `num_load` and `num_reduction_ops` were falsified — observationally identical on
   the curriculum, wrong on a divergence kernel). Also: is the fact style-independent (not fooled by how
   the kernel is written)? Does a branch actually READ it (a fact no branch reads must be cut)? (If the
   abstract case is inconclusive and a POPULATION change is in scope, escalate to Part 3 and *author*
   the divergence kernel.)
6. THRESHOLD DIVERGENCE. A faithful fact can be read UNFAITHFULLY: a continuous quantity becomes a
   disguised dtype/identity fence the moment it is COMPARED TO A CONSTANT that splits its real
   value-set along dtype/kernel lines (e.g. an itemsize-like field whose only values are {2,4}, gated
   `<= 2`, is just `if dtype is half` wearing a faithful name). Construct two workloads with the SAME
   value of the real property but a DIFFERENT dtype/kernel; the branch must decide identically. If it
   diverges at equal real-property, the threshold keys on the incidental thing — FAIL. Faithful use
   keeps a dtype/identity-correlated quantity a FACTOR inside a hardware-unit budget (bytes = elems ×
   itemsize, occ = grid_rows // num_sm), never the operand of a literal comparison; any rationale that
   restates as "fires only for / excludes the dtype (or kernel) D" IS the fence.

=== PART 3 — POPULATION FAITHFULNESS (does the populator faithfully COMPUTE-or-DECLARE the property, or piggyback a lucky proxy?) ===
Fire this part whenever the POPULATION changed (a builder, or an inline derivation a branch keys on), or
whenever Part 2's abstract divergence case was inconclusive. This is the part that catches the
`bs.reduction` class — a populator that reaches the right answer on every in-curriculum kernel by reading
a flag that merely CORRELATES with the real property. Apply the FRAMING DISCRIMINATOR at every step: a
correlate used where the real property WAS computable is a proxy (REFUTE); a value DECLARED as a fallback
where it genuinely was not — or simply firing the fact on an unforeseen workload — is fine. Never treat
"the property is not trivially clean here" as a reason to want the fact UNBUILT (that is Gate-H breadth,
not a Part-3 refute).
7. STATED-PROPERTY vs CODE. Read the populator against THE REAL PROPERTY IT CLAIMS (slot above). Does the
   code compute that property DIRECTLY from faithful provenance (the reduction-axis / store-extent /
   block-id provenance the compiler already builds), or does it select/branch on an incidental flag
   (`bs.reduction`, a membership test, an `isinstance`) that only lines up on the curriculum? A populator
   whose CODE does not match its STATED PROPERTY ⇒ REFUTE, and name the faithful primitive it should
   derive from instead.
8. AUTHOR THE DIVERGENCE KERNEL (compile-only; NO GPU). Mutate a VETTED TEMPLATE so the proxy and the
   real property DISAGREE, compile it, and dump the registered facts (`compiler_seed_configs` + the
   relevant `ReductionFact`/walker facts). Calibrated mutant for the canonical case: add an
   `hl.zeros([m,n])` accumulator dim — `allocate_reduction_dimension` sets `bs.reduction=True` though
   the dim is NEVER reduced, so a populator reading `bs.reduction` now diverges from is-actually-reduced.
   Other axes: change ONLY itemsize (same elem-count/structure) to expose an itemsize-multiplied key;
   rewrite the reduction style (loop vs single-pass) at fixed real-property to expose a style-dependent
   proxy. Report: the proxy field's value, the real property's value, and whether they AGREE on the
   mutant. They disagree ⇒ the populator is an unfaithful proxy ⇒ REFUTE.
   BOUNDS: TEMPLATE LIBRARY ONLY (no free-hand DSL authoring from scratch — that historically stalled);
   K≤3 compile/dump attempts; on K failures, FALL BACK to the Part-2 abstract argument, LOG the authoring
   failure (which template/mutation) to the human-review queue, and return the abstract verdict. NEVER run
   the mutant on the GPU — compile + fact-dump only (a mis-compile is caught by the compile step, so a
   buggy mutant cannot fabricate a number).
   WHY THIS IS A GENERALITY DEFECT: an unfaithful populator decides WHICH KERNELS THE HEURISTIC FIRES ON,
   so a lucky proxy silently mis-fires on the first real kernel where proxy≠property. It therefore REFUTEs
   even with a perfect perf/aggregate number (method §3: faithfulness > performance).

OUT OF SCOPE (route elsewhere): whether the resulting LEVER overfits the curriculum / belongs in the
core / is gated too NARROWLY, AND whether a populator UNDER-FIRES (declines to build the fact on an
unforeseen workload) → GATE H (BROADEN); whether the heuristic's default/best-guess CONFIG for an untuned
regime is any good → GATE H; whether an edit regressed a shape below floor / passes the §3 acceptance
model → GATE R. This gate is ONLY about where the walk lives, how general the field is, whether the fact +
its threshold are faithful, and whether the POPULATION faithfully computes-or-declares the claimed property
(a silent proxy fails; broad firing and a declared fallback do not).

Record AS-RETURNED: {doctrine: pass | refuted(rule, fix), generality_verdict:
general|justified-specific|under-general, faithfulness: faithful_property | scoped_deferral |
disguised_fence, population: faithful_derivation | declared_fallback(where_provenance_unresolved) | lucky_proxy(the_faithful_primitive_to_use) | authoring_failed_fell_back,
the_fact_divergence_case, the_threshold_divergence_case, the_authored_mutant_and_dumped_facts, what_it_tracks, reasoning}.
PASS only if ALL THREE parts pass. `faithful_derivation` and `declared_fallback` BOTH pass — broad firing
on unforeseen workloads is desired, never a refute (a too-narrow / under-firing populator is a Gate-H
BROADEN, not a Gate-D fail). A `scoped_deferral` (excluding a dtype/kernel because NO faithful rule
exists — e.g. a non-monotonic optimum left to the autotuner) is acceptable ONLY if that "no rule"
claim is itself evidenced and logged AS a deferral, never blessed as faithful. Do NOT over-correct
into demanding a general dataflow framework where one specific faithful fact suffices, and do NOT refute a
populator merely for firing where the property is unusual — only for SUBSTITUTING a correlate for a
property it could have COMPUTED.
```

---

## GATE E — Overfit guard + TEST-firewall keeper. The PORTFOLIO gate (a must, alongside A, B, H, and R).
Distinct from A (which checks ONE win), this guards the **heuristic-as-a-whole** against silently
memorizing the curriculum — the failure that *looks done* (run 2's good-geomean false victory). It carries
MORE weight under the §3 aggregate model: the aggregate alone cannot catch curriculum-memorization, and
VAL/TEST are interpolation *inside* the train envelope (frame check 4), so held-out-tracks-train is
**necessary-not-sufficient** — the real signal is FENCES + key/population faithfulness. **Distinct from the
refactor-critic:** Gate E *DETECTS* overfit/fences and is the sole-reader-at-freeze of TEST (a verdict
gate, can FAIL); the refactor-critic only *PROPOSES* the faithful re-key for the fences E flags (a helper,
no reject power, never reads a split). Two duties:
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
2. TRAIN↔HELD-OUT GAP (necessary, not sufficient) — measure (or, periodic mode, reason from recorded
   configs) the per-shape floor (`G ≥ 0.75`) and the per-(kernel, dtype) geomean (a diagnostic now, not a
   hard bar) on VAL and an OFF-FOCUS kernel. Report the train↔held-out gap as a FIRST-CLASS number. But
   remember VAL is interpolation inside the train envelope: VAL-tracks-train is REQUIRED but does NOT
   clear overfit on its own — a regime-memorizing edit lifts VAL too. So weight checks 1 (fences) and the
   D/H faithfulness verdicts at least as heavily as this gap.
3. FREEZE-MODE ONLY — read TEST exactly once: report per-shape seed/tc (and max-autotune for any weird
   shape), the train↔test gap, and whether any TEST shape falls below its 0.75 floor where train shapes
   clear it (the overfit signal). State plainly that this was the single sanctioned TEST read.
4. INTERPOLATION CHECK — is every VAL/TEST shape's N-regime actually covered by a TRAIN shape, so
   held-out is interpolation (fair) not extrapolation into an untrained regime (which would mask
   overfit as "expected")? (This is WHY check 2 is necessary-not-sufficient — by construction held-out
   cannot catch regime-extrapolation overfit; only faithful keys/populations can.)

Record AS-RETURNED: {overfit: bool, train_vs_heldout_gap, fences_found:[...], faithfulness_reliance_note,
test_read_results (freeze only), reasoning}. PASS only if NO constant is a shape-identity fence AND the
key/population faithfulness holds (D) AND held-out at least tracks train (necessary). A held-out shape
below its 0.75 floor where train clears it is a FAIL. Do NOT PASS on a good held-out geomean alone — that
is the interpolation trap.
```

---

## GATE F — Mechanism gate ("explain the win" + "justify every field"). Fire on a SURPRISING win, OR any STRUCTURAL/MULTI-FIELD lever.
Fires on ANY of: (a) a win whose DIRECTION is counter-intuitive (e.g. a NARROW reduction wanting
FEWER warps, a cap that helps where theory says it shouldn't); (b) a lever that FLIPS a
structural / near-universal default (`pid_type`, persistent↔looped, indexing/TMA mode, `num_stages`
off its hardcoded value, eviction policy); or (c) a lever that sets MORE THAN ONE non-default field.
A number can be real + reproducible (passed A) yet MIS-UNDERSTOOD — a plausible-but-wrong story keys the
rule wrong AND banks dead fields (the `persistent_blocked` miss, frame check 1). The fix: READ THE
GENERATED CODE. Analysis/profiling; the field-attribution reverts (check 4) use the GPU, foreground-serial.

```
A performance win has been measured and reproduced. Before it can be banked into a generalizable
rule, you must (i) explain the MECHANISM — WHY this config is faster, verified IN THE GENERATED
TRITON/IR, not from what the knob "ought" to do — and (ii) prove EVERY non-default field it sets
actually earns its place. "It just measured faster" is NOT acceptable: an unverified story keys the
fact wrong and an inert field is dead weight.

THE WIN (verbatim): {CLAIM + WHY this gate fired — counter-intuitive direction / structural-default
flip / multi-field lever}
IMMUTABLE COMMIT: {SHA}
THE CONFIG(S) + their reproduced latencies: {the lever's emitted config vs the base seed + numbers;
the full set of non-default fields the lever sets}

Find the mechanism using ncu / generated Triton / IR — not speculation:
1. What HARDWARE resource explains it (occupancy, register/SMEM pressure, the reduction-tree cost,
   L2 residency, wave quantization, grid saturation, launch/scheduling)? Show the metric that moves
   with the win. CONFIRM the worker's stated mechanism against the LOWERED CODE: if the story names
   a behavior ("fills the machine", "splits the row", "better coalescing", "strides rows"), point to
   the generated Triton that shows it actually happens — or REFUTE the story if the code shows it
   does not (the win may be real for a DIFFERENT reason; name the real one).
2. Does the explanation predict a BOUNDARY — the workload property at which the win should reverse?
   Name it. A real mechanism generalizes to a rule keyed on that property; a coincidence does not.
   If the verified mechanism differs from the gate the lever currently uses, say so — the KEY may
   need to change (route to Gate H / try-harder re-key).
3. Sanity-check the boundary: at a shape on the OTHER side of it, does the win correctly disappear/
   reverse? If you cannot find a boundary the rule would overfit — say so.
4. FIELD ATTRIBUTION (the dead-knob / better-neighbor check). For EACH non-default field the lever
   sets: (i) read the generated Triton to confirm the field actually changes the emitted code on the
   firing shapes; a field that is provably inert given the lever's OWN gate precondition (e.g. a
   `num_sm_multiplier` that cannot change `block_size` because the gate guarantees `grid_rows <
   num_sm`) is INERT — flag it to be dropped, no bench needed. (ii) Otherwise revert that field
   ALONE to its default (lever otherwise intact) and measure the marginal delta, foreground-serial.
   A field whose solo-revert is within noise is INERT → drop it. **COUPLINGS ARE EXEMPT:** if
   reverting field X alone looks free BUT reverting it together with its partner Y tanks the win
   (a redundant-substitute / mutually-dependent pair), the fields are coupled — KEEP the bundle and
   record it as a coupling, do NOT drop either. Only a field that is inert WITH its partners present
   is dead. (iii) While the codegen is open, check the immediate NEIGHBORS of any structural knob
   (`persistent_blocked` vs `persistent_interleaved`, ±one warp/stage) — if a neighbor is clearly
   better, surface it as an overtime candidate.

GPU RULE: the analysis (codegen reading, inert-by-precondition detection) is GPU-free and fans out;
the marginal-revert benches in check 4 are foreground-serial, one at a time (never N concurrent).

Record AS-RETURNED: {mechanism_found: bool, mechanism_matches_stated_story: bool, the_resource_and_
metric, the_predicted_boundary, the_property_to_key_the_fact_on, inert_fields:[...], coupled_field_
sets:[...], better_neighbor_found: null | {config, delta}, reasoning}. PASS only if (a) there is a
mechanistic explanation VERIFIED IN CODE that yields a workload-property the fact can key on, AND
(b) every non-default field is either attributed (moves the number) or part of a recorded coupling.
If no mechanism: do NOT bank as a general rule (at most a logged, shape-scoped observation flagged
for revisit). If inert fields found: the lever must drop them before banking.
```

---

## GATE H — Generality gate (KEEP / BROADEN / DEFER / REJECT / BORDERLINE a lever for the generalizable core). A MUST — fire on EVERY proposed or existing lever / branch / constant before it enters the core PR.
Analysis only (no GPU) — reuse the recorded per-shape numbers + the lever's gate/constant. It adjudicates
a SINGLE lever's place AND its BREADTH (KEEP / BROADEN / DEFER / REJECT / BORDERLINE), per the rules in
the frame. Scope deltas: NOT a re-measurement (A), NOT a premature-stop check (B), NOT the whole-heuristic
overfit audit (E), NOT the regression check (R), NOT the cross-lever architecture critic (refactor-critic).
Generality is Priority 2 (method §3) — **narrowness needs a reason, not breadth**: an unjustified narrow
fence BROADENs, but a narrow scope with a documented, tested mechanism reason KEEPs.

```
You decide whether the lever below belongs in the generalizable-core heuristic AND whether it is gated
WIDELY ENOUGH. Return exactly one of KEEP / BROADEN / DEFER / REJECT / BORDERLINE. Default to the FAITHFUL
bar (don't KEEP unless it earns it on every axis), and default to BROADEN on a narrow fence with NO
justified, tested reason for its narrowness — but KEEP a narrow scope whose narrowness IS justified +
tested, and do NOT over-DEFER a genuine, faithful, bounded win either.

THE LEVER (verbatim): {what it does — the branch/constant + the config it emits}
THE GATE/KEY it fires on: {the exact workload quantity + threshold it branches on}
THE STATED MECHANISM: {WHY the author says this win happens — the hardware resource it names}
THE PREDICTED BOUNDARY (from Gate F, if run): {the workload property at which the win should reverse}
MEASURED BENEFIT (per-shape, never a geomean): {which shapes, how much, which dtypes; is it above the
   ~5% noise floor? is there an articulable reason/mechanism — non-ncu "makes sense logically" counts?}
MEASURED / EXPECTED REGRESSION: {which shapes, how much, on-curriculum AND off-curriculum}
SHAPE REALISM (the Regression-referee's verdict, NOT your own assertion): {for the shapes it HELPS and
   the shapes it HURTS — realistic real-workload shapes, or synthetic/diabolical (e.g. 2 rows of 16M
   elems, a box around one curriculum shape)?}
ADDS/READS A FACT?: {none | the fact + how/where it is computed}
THE FORM of the branch: {a bare `if x > K` magic threshold | a ramp | a named bool | a single measured crossover}
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

1b. BREADTH — the generality-EXPANSION line (the dual of rule 1). **A narrow scope is FINE *with a
   reason*; what's banned is a narrow scope *without* one.** A faithful, real win that is gated narrowly
   has exactly two acceptable states — KEEP it narrow IF the narrowness is JUSTIFIED, else BROADEN:
   - **JUSTIFIED narrow ⇒ KEEP** — there is a *well-documented, mechanism-grounded, and TESTED* reason
     the lever applies only in this scope: a **proven reversal boundary** (a shape just outside the gate
     where the win demonstrably reverses/vanishes, shown in the lowered code or Gate F's predicted
     boundary), OR an equivalent documented + measured argument that the mechanism genuinely does not
     hold wider. The reason must be recorded with the lever (a one-line "scoped to X because <mechanism>;
     reverses at <shape>, tested") so it is auditable, not folklore. This KEEPs — generality does NOT
     force you to widen a lever whose narrowness you have justified and tested.
   - **UNJUSTIFIED narrow ⇒ BROADEN** — the mechanism (a hardware resource: occupancy, bytes,
     register/SMEM pressure, reduction-tree cost) would logically apply WIDER than the gate allows, and
     the author has NOT exhibited a reversal boundary or a tested reason. Default to BROADEN: name the
     wider form, re-enter the loop to re-measure it (accepted under the §3 generality license — may dip
     the majority / cost ≤~3–5% on a (kernel,dtype) cell, as long as no disaster and no >10%-vs-frozen-anchor
     collapse). Tells of an unjustified fence: it multiplies in a dtype/identity-correlated quantity the
     mechanism does NOT need (the BANDB_W8 tell — a dtype-independent register mechanism keyed on
     `input_load_itemsize`); or it is a two-sided band reverse-engineered to bracket exactly the
     curriculum (the B200_LOOP_HBM tell). BROADEN is Priority-2 standing work, NOT a
     disaster — it never preempts a below-floor rescue, but it is never "deferred until perf is done"
   either.
   - **UNDER-FIRING is the same defect, one level up** — a gate so narrow the lever simply does NOT FIRE
     on an unforeseen (usually off-curriculum) workload: a populator that refuses to BUILD the fact, or a
     `get_seed_config`/`is_eligible` returning `None`/ineligible, silently routing that kernel back to the
     really-bad compiler default. A seed is never forced (method §2), so firing a best-guess config is
     cheap and declining is expensive ⇒ **BROADEN: build the fact / return a best-guess config and fire**
     (sub-optimal on the unforeseen kernel is FINE — record it). This is the OPPOSITE of a lucky proxy: a
     best-guess is MAXIMUM generality (you fire on purpose for the unforeseen, accepting it may be
     sub-optimal); a lucky proxy is MINIMUM generality (it only works because it was fitted to the
     curriculum). The verdict here keys on the gate being too RESTRICTIVE, independent of whether the
     emitted config is yet well-tuned.

1c. FORM — ramp / named-bool / measured-crossover, never a bare magic threshold. Look at THE FORM of the
   branch. The ONLY disfavored form is a **bare unexplained `if x > K`**. Require the form to match the
   mechanism, choosing exactly one of FOUR:
   (i) **continuous** effect → a monotone **RAMP** keyed on the faithful quantity (cf. the
       warps-vs-reduction-bytes ramp);
   (ii) a genuine **paradigm split** (e.g. reduction-bound vs memory-bound) → a **NAMED BOOL** derived
       from a faithful fact (`is_memory_bound = bytes <= K`) — the bool's threshold STILL passes Gate D's
       threshold-divergence test (a named bool can smuggle a dtype/identity fence — the
       `bytes<=K`-is-really-`dtype==half` trap);
   (iii) a **single measured hardware crossover** → an `if`/threshold **IS correct** (cf.
       `FULL_WIDTH_PERSIST_MAX_ELEMS` — a real fp32-tile-vs-SMEM spill point); do **NOT** smear a real
       cliff into a fake ramp;
   (iv) **NO faithful rule exists** (two fact-identical kernels want OPPOSITE configs — the
       softmax-vs-rms_norm small-N landmine; or a non-monotonic / multi-variable optimum no monotone 1-D
       surface captures) → **leave the split to the autotuner** and log it as a Gate-D `scoped_deferral`.
   ESCAPE HATCH: a genuine, EXPLAINED win whose faithful form fits none of (i)–(iv) cleanly is allowed —
   state the faithful key(s) and the mechanism, and it still passes Gate D. The form is free; the KEY is
   not. Only a bare `if x > K` with no stated mechanism is rejected outright (→ DEFER pending a form +
   reason). Do NOT force a ramp where (iii)/(iv) is correct, and do NOT re-classify a lever that is merely
   having an existing ramp's breakpoint tuned (that stays fast-path, rule 2's exception).

2. MAGNITUDE × NOISE × REASON. Within-noise (≲5%) AND no articulable reason ⇒ DEFER (faithful +
   never-negative is NOT sufficient; there must be a reason AND a win above noise). A real win (≳5%)
   with a reason that "makes sense logically" ⇒ KEEP-eligible (ncu NOT required). A large win
   (≳15–40%+) is a strong KEEP-pull even behind a soft op-pattern gate ⇒ KEEP-but-FLAG-as-specific.
   EXCEPTION: merely tuning the knobs of an EXISTING faithful lever (e.g. moving a ramp breakpoint) ⇒
   KEEP even unexplained, if above noise with no regression; RESTRUCTURING it (an unexplained
   non-monotonic carve-out) ⇒ DEFER. (These %s are rough, not bright lines — see rule 7.)

3. CATASTROPHE PRIORITY. Rescuing a below-floor shape — `G < 0.75` vs tc, or `< 0.75 × max-autotune`
   if it's a weird shape (the §3 disaster line; the deeper the cliff, the higher the priority) — up to
   its FLOOR is the single highest-value justification and OUTRANKS breadth/win-chasing. It pulls even a
   narrow shape-boundary case to KEEP for now (then flag it BROADEN — rule 1b — so the rescue gets a
   general gate rather than staying a curriculum fence). It does NOT override rule 1 — an unfaithful key
   still needs try-harder, not a pass.

4. DOWNSIDE × REALISM (master axis, BOTH directions; realism = the Regression-referee's verdict).
   ACCEPTABLE under the **§3 acceptance model**: a regression confined to synthetic/diabolical shapes (a
   real regression is far worse than a diabolical one); OR a regression that satisfies the acceptance
   model — no realistic shape below its floor, no (kernel,dtype) cell >10% below the **frozen-champion anchor**, and
   either net-progress holds OR the **§3 generality license** applies (a *measured*-breadth edit may take
   ≤~3–5% on a (kernel,dtype) cell, once-per-(kernel,dtype)-cell against the frozen anchor). Before accepting any realistic-shape
   regression, **fire anti-giving-up for a faithful SEPARATOR** — a separator that gets BOTH always beats
   a trade. DEFER when: a separator is achievable-but-unbuilt; the downside is unbounded/unknown
   off-curriculum with no separating gate; or it leans on a deliberately UNSOUND / "shady" tactic (e.g.
   tolerating spill for a win) — a cleaner/gated version may return (DEFER, not REJECT). A regression that
   pushes a REALISTIC shape **below its floor** is a disaster → reject (§3), never a trade. A "generality"
   drop with NO measured off-train shape it lifts is NOT a licensed trade — it is a plain regression
   (don't let the label buy the drop). A benefit ONLY on diabolical shapes ⇒ not worth the complexity
   (DEFER/BORDERLINE).

5. FACT HYGIENE (if it adds/reads a fact; walk-LOCATION/generality = the Fact-gate, GATE D). New
   FAITHFUL facts are WELCOME — compute MORE of them, future levers may read them. BUT: a fact needing
   RUNTIME information ⇒ REJECT (configs carry no runtime info; it must be computable from source / the
   input fx-graph at compile time). A fact computed by a brittle SECOND graph re-walk (drift risk) ⇒
   DEFER (fix = fold it into the canonical single fact-build pass, then it KEEPs).

6. COMPLEXITY (portfolio). Judge REASON-ABILITY, not COUNT — many easy-to-reason levers are fine; flag
   only levers that are individually complicated, interact confusingly, or flip a near-universal default
   / bundle many knobs at once (each deviation from a strong default — e.g. `pid='flat'` — needs its own
   justification; sprinkling in an extra unusual knob "because it helps" is a yellow flag). Mild
   redundancy with an existing lever is not, by itself, fatal. **DEAD-KNOB RULE: every non-default
   field the lever emits must earn its place with BOTH an articulable reason AND a measured marginal
   effect (Gate F check 4 supplies the attribution). A field that is provably inert — it cannot change
   the emitted code given the lever's own gate precondition, or its solo-revert is within noise — must
   be DROPPED; "it doesn't hurt" is not a reason to set a field. EXCEPTION — couplings: a field that
   looks free to revert ALONE but whose removal TOGETHER WITH its partner tanks the win (a
   mutually-dependent / redundant-substitute pair) is NOT dead — keep the bundle and record it as a
   coupling. Only fields inert with their partners present are dropped. The dead-knob rule trims
   single-field cruft; it never forces a coupled pair apart.**

7. WEIGHT & GRAY ZONE (the calibration — read this LAST and let it temper 1–6). When axes CONFLICT,
   default to BORDERLINE: RECORD the specific tradeoff to the ledger + the human-review queue for the
   human's later judgement, and provisionally DEFER (the reversible default — logged to the
   removed-heuristics-log, re-add retained) so the run keeps moving. NEVER ask, block, or wait on the
   human — this is an unattended run (method §6.0). Let a strong positive (catastrophe-rescue, large
   REALISTIC win, a clean faithful fact, an easy-to-reason lever) tilt to KEEP. Do NOT reflexively
   DEFER on conflict — most genuine conflicts are BORDERLINE, not DEFERs. There is always gray area;
   this gate is not expected to be right every time. When genuinely split, return BORDERLINE and RECORD
   the tradeoff for review — never manufacture false certainty.

BROADEN vs DEFER vs REJECT: **BROADEN** = faithful + real + a win, but gated narrower than its mechanism
warrants and with NO proven reversal boundary — emit the wider form and re-enter the loop to re-measure
(the win is real, it just deserves a wider gate). **REJECT** = an unfaithful KEY that cannot be fixed
without changing what it branches on (identity / bare dtype literal) — pair with TRY-HARDER (re-key mode)
when the win is high-value. **DEFER** = faithful-or-fixable but not ready (within-noise, restructures a
lever, a regression with no gate the acceptance model won't take, brittle re-walk, a bare magic threshold
with no form/reason) — log to the removed-heuristics-log WITH a re-add recipe; nothing is lost.

Record AS-RETURNED before reporting: {verdict: KEEP | BROADEN | DEFER | REJECT | BORDERLINE,
key_faithfulness: faithful | identity | dtype-literal | op-pattern, breadth: mechanism-matched |
too-narrow(wider_form, the_boundary_that_would_justify_narrow) | has-proven-boundary, form: ramp |
named-bool | measured-crossover | explained-other | bare-magic-threshold, magnitude_class: noise | real |
large | catastrophe-rescue, reason_present: bool, downside: bounded | realistic-regression | unbounded |
diabolical-only, fact_hygiene: n/a | clean-faithful | runtime-impossible | brittle-rewalk,
complexity_note, inert_fields_to_drop:[...] (from Gate F check 4; [] if none), conflicting_axes:[...],
action: keep-in-core | broaden-and-remeasure(wider_gate) | keep-after-dropping-inert-fields |
defer-to-removed-log | try-harder(re-key on {property}) | fold-into-fact-build |
record-borderline-for-human-review (provisionally defer, keep climbing), reasoning}.
KEEP only if: the key is faithful, the win clears noise WITH a reason (or is a catastrophe-rescue / an
existing-lever tuning), the downside is acceptable per the §3 model, the FORM matches the mechanism (rule
1c), the gate is as WIDE as the mechanism allows OR has a proven reversal boundary (rule 1b), AND no field
is inert (dead-knob rule 6 — couplings exempt). If the only issue is breadth, return BROADEN. If the only
issue is inert fields, return keep-after-dropping-inert-fields. Otherwise DEFER, REJECT (+ try-harder), or
BORDERLINE per the rules above.

OUT OF SCOPE (route elsewhere): is the win real/reproduced → GATE A; is a STOP/park premature → GATE B;
is the fact/threshold faithful + the walk in the right place → the Fact-gate (GATE D); does the
heuristic-AS-A-WHOLE overfit the curriculum + TEST firewall → GATE E; explain the win against the
LOWERED CODE + attribute/trim every field → GATE F (fires on counter-intuitive OR structural/multi-field
levers, not just surprises); did an edit push a realistic shape below its floor → GATE R (regression-referee).
```

---

## GATE R — Regression-referee + acceptance model (disaster-avoidance). A MUST. Fire on EVERY edit that changed ≥1 emitted config.
Independent, fresh context. **Owns the realistic↔diabolical verdict + the frozen-champion-anchor
bookkeeping** (so neither call is the self-interested worker's). GPU re-measurement is foreground-serial.
Runs the **§3 acceptance model** (frame steps 3/3b/4/5 = the disaster floor + widening-coverage, the collapse-vs-anchor bound,
the net-progress aggregate). Its changed-cell sweep subsumes the former flip-axis sweep; its per-cell
medians are authoritative — Gate A reads the focal cell from this sweep rather than re-benching (§6.2).

```
You are an independent referee. Adjudicate the edit below against the §3 ACCEPTANCE MODEL. Default to
"a regression is hiding until proven otherwise." Faithfulness/generality (Gates D/H) are decided
elsewhere — your job is the PERF acceptance: no disaster, no collapse-vs-anchor, net progress.

THE EDIT: {heuristic / fact / constant change}  +  IS IT A GENERALITY EDIT?: {no | yes — and the NAMED off-train realistic/TRANSFER shape it is supposed to lift (required to claim the license)}
BEFORE / AFTER: {two SHAs, or the two versions}
THE FULL ACTIVE MATRIX: {every kernel × shape × dtype × split incl. robustness}
THE FLOORS: {G ≥ 0.75 vs tc; or seed ≥ 0.75 × max-autotune for confirmed weird shapes}
THE FROZEN-CHAMPION ANCHOR: {each (kernel,dtype) cell's geomean at the last banked freeze — the up-only reference the 10% bound and the 3–5% license are measured against; NOT the rolling per-edit champion}
IMMUTABLE COMMIT: {SHA}

1. SCOPE — run config_recorder over the FULL active matrix BEFORE and AFTER; enumerate the cells whose
   normalized config CHANGED. Byte-identical cells are perf-invariant (skip). A source/fact/normalize/
   lowering edit needs the `--triton` diff, not config-only. FULL MATRIX or it's a false all-clear.
2. RE-BENCH every changed cell — foreground-serial GPU, one at a time, median-of-N. changed ≠ win (a
   changed cell still earns measurement, never an assumed improvement). A move counts as UP/DOWN only if
   it clears the ~5% noise band; within-noise is NEUTRAL (this kills noise-floor majority manufacturing).
3. DISASTER (per-shape, the one hard bar) — for any shape that ends BELOW its floor, return a
   realistic↔diabolical verdict. DEFAULT-TO-REALISTIC: a real workload that occurs in actual models is
   realistic REGARDLESS of curriculum membership; only a genuinely synthetic/diabolical shape (2 rows of
   16M elems, a tight box around one shape) may sit below floor. NEVER launder a genuine regression as
   "unrealistic." A REALISTIC shape below its floor ⇒ REJECT (disaster), full stop.
3b. WIDENING COVERAGE (fire ONLY if the edit WIDENS a lever's firing region — a BROADEN, refactor, or
   re-key whose after-predicate fires on shapes/kernels the before-predicate did NOT; skip a pure
   narrowing / in-region tune). Steps 1–2 already catch the widen's effect on every regime the curriculum
   SAMPLES (those cells change config ⇒ they get benched). The only blind spot is a realistic regime the
   curriculum does NOT sample — which rule 1's floor still binds (realistic shapes bind INDEPENDENT of
   curriculum membership). You MAY close it, but ONLY with a REAL workload: if the newly-covered region
   contains a shape that occurs in an ACTUAL named model/op, ADD that shape to the active matrix (train or
   robustness split — NEVER TEST, which would break Gate E's firewall), bench it as a normal cell (steps
   2–3), and NOTE it (ledger + one line to the human-review queue: the shape, the real workload it came
   from, the prompting edit). It is then a permanent, always-swept cell; on this edit it has no
   before-value and no frozen anchor, so it binds only rule 1 (disaster) and gets its anchor at the next
   freeze. HARD GUARD — never FABRICATE a shape to manufacture a regression: if you cannot NAME a real
   workload in the newly-covered region, add NOTHING and reject NOTHING on a hypothetical. A below-floor
   shape is a disaster ONLY when it is a real, named workload — the floor guards against real harm, not
   invented edge cases (generality is Priority 2; do not kill a general edit on a fabricated corner). If
   the task file pins the workload as FIXED, do NOT add — flag the missing real shape to the human-review
   queue instead.
4. COLLAPSE (per-(kernel,dtype)-cell, vs the FROZEN anchor) — no (kernel,dtype) cell's geomean may drop
   more than ~10% below the FROZEN-CHAMPION ANCHOR (not the rolling champion — measuring vs a sliding
   champion permits unbounded ratchet-collapse; the anchor is per cell, never dtype-blended). >10% below
   anchor ⇒ REJECT.
5. NET PROGRESS (aggregate) — a MAJORITY of (kernel,dtype) cells' geomeans must improve beyond the ~5% noise band
   (neutral moves don't count either way). The denominator is the cells you re-benched in step 2 (the
   changed set from step 1); byte-identical cells are perf-invariant and **excluded by construction** (not
   "neutral"). So a subset-firing edit is judged on its moved cells — 1-of-1-moved-up PASSES — but a moved
   up/down split that is not majority-up ⇒ REJECT. If the majority does NOT improve ⇒ REJECT, UNLESS the
   generality exception applies: IS IT A GENERALITY EDIT with a NAMED off-train shape that the re-bench
   confirms it actually LIFTS? Then the majority may dip and one (kernel,dtype) cell may drop ≤~3–5% below anchor —
   but ONLY once-per-(kernel,dtype)-cell against the frozen anchor (check the spent-license ledger; a cell already
   at its license vs this anchor cannot spend again until a re-freeze). A "generality" claim with no
   measured lift is NOT licensed — treat its drop as a plain regression.

Record AS-RETURNED: {accept: bool, changed_cells:[...], below_floor:[{shape, G, realism}],
realism_verdicts:{shape: realistic|diabolical}, cells_vs_anchor:[{kernel, dtype, delta_vs_frozen}],
majority_up: bool, generality_license: {claimed: bool, measured_lift_shape, confirmed: bool, within_once_per_cell_budget: bool}, curriculum_additions:[{shape, real_workload_cited, prompting_edit}], reasoning}.
Accept only if: NO realistic shape below its floor (3), NO (kernel,dtype) cell >10% below the frozen anchor (4), AND
either the majority improves beyond noise OR a confirmed-measured generality license covers the dip (5).
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

## HELPER — Refactor-critic (the whole-heuristic simplification dual of Gate H's BROADEN). Fire on a CADENCE — every K levers / at milestones / when a whole kernel-family lags (worst-kernel triage). Priority-2 standing work.
Analysis only (no GPU) to PROPOSE; a proposed refactor then re-enters the loop like any edit (Gate R
acceptance + the F/G gates). This is the de-bloat engine: it attacks the heuristic's ARCHITECTURE, not a
single lever (that's Gate H). **A refactor is EXPECTED to change emitted configs / behavior — it need NOT
be behavior-preserving.** Collapsing levers, removing a gate it judges redundant, splitting one gate into
two, re-keying — all change configs, and that is FINE: generality is Priority 2, above perf, so a
simpler/more-principled heuristic that shifts (and may slightly cost) perf is a *good* trade — **lean hard into this: do not let perf-timidity preserve complexity; Gate R is the perf backstop, not this critic's job.** It ships if
it is genuinely simpler AND clears the §3 acceptance model (no new disaster, no >10%-vs-anchor collapse;
a perf dip rides the generality license). The one hard limit: it may not knowingly break a **known-valid**
invariant (remove a gate already established necessary, or drop a fact a live lever depends on); removing
something it only *suspects* is redundant is fine — the loop re-proves it.

```
You look for ways to make the heuristic SIMPLER and MORE PRINCIPLED as a whole — collapsing narrow
levers into general rules, not adding new ones. Default to "this may be able to be simplified; every lever / constant / gate / fact could be unnecessary, and should have a concrete, shown-relevant reason to exist."

THE HEURISTIC: {the current lever/constant/fact inventory + which curriculum cells each fires on — the recorded configs}
THE PROVENANCE LOG (why each lever/constant/fact exists): {the ledger's banked-win + gate-verdict entries (AS-RETURNED, keyed by {SHA, gate, claim}) that JUSTIFIED each one — the mechanism it was banked on + the Gate F/H verdict; the removed-heuristics-log; and the notebook's TRIED-AND-REJECTED (so you don't re-propose a removal already refused)}
THE LAGGING KERNEL/FAMILY (if triggered by worst-kernel triage): {kernel + why it lags}
IMMUTABLE COMMIT: {SHA}

Hunt, in priority order:
1. COLLAPSE — do N narrow levers share an underlying mechanism that ONE faithful rule (a ramp / named
   bool / single budget keyed on a workload property) would express? (e.g. several per-band caps that are
   really one bytes-budget; a proxy + a fence that are really one occupancy rule.) Name the unifying
   property and the single rule that replaces them.
2. RE-PRINCIPLE — is a lagging family lagging because the ARCHITECTURE is wrong (a missing fact, a
   mis-factored band structure), not because it needs another narrow lever? Propose the structural change.
3. DEAD-WEIGHT — levers/fields/facts no branch reads, or that the recorder shows never change an emitted
   config; flag for removal. CASCADE: removal is transitive — when you remove X, trace its dependents
   forward (a value only X feeds; a gate only that value feeds) and bundle the whole chain as ONE ORDERED
   proposal; the known-valid-invariant check is evaluated against the POST-refactor state (dropping a fact
   a lever needs IS allowed when the same ordered cascade removes that lever first).
4. PROXY/FENCE DEBT — proxies or curriculum-fences flagged by Gate D/E/H that a cleaner faithful form
   would replace; propose the re-key. (Gate E/D/H DETECT and flag these fences; this critic only PROPOSES
   their faithful replacement — it does not itself audit for overfit and never reads VAL/TEST.)
5. OBSOLETE PROVENANCE (archaeology — retire a check whose reason is gone). For each lever/gate/constant/
   fact, read THE PROVENANCE LOG for WHY it was banked, then ask whether that justification is STILL LIVE.
   Propose removal whenever the original reason is moot — the condition it addressed no longer exists (a
   workaround for a codegen bug since fixed; a guard for a kernel/dtype/machine out of scope; a constant
   sized for an old SM count / accumulator width; a gate orphaned by the cascade in hunt 3). Be AGGRESSIVE
   — you do NOT need certainty: the full loop (correctness gate + Gate R + Gate D/H) re-proves every removal
   and the re-add recipe makes a wrong call cheap to reverse, and that backstop is what licenses boldness,
   so do NOT preserve a check merely because you are unsure why it exists. Label each removal's confidence
   so the loop knows how hard to scrutinize the re-proof: `reason-moot` (provenance shows it addressed X and
   X is provably gone) | `speculative` (no live reason reconstructable; betting the gate stack clears it) — both license a direct removal (action `remove-with-readd-recipe`).
   A THIRD case is different: `reason-disputed` — the banked justification is PRESENT but you judge its
   MECHANISM STORY false (fishy, not stale). This does NOT license a direct delete: you are GPU-free and
   would be overriding a gate-verified win on a hunch, and a wrong story often coexists with a real effect
   (a lever can win for a DIFFERENT reason than the one banked — the `persistent_blocked` "fills the machine"
   miss). So a disputed justification REOPENS the banked win — action `reopen-via-GateF-rebench`: route it to
   Gate F (re-explain the mechanism against the lowered code) + a re-bench. Outcomes: story wrong AND no real
   independent effect → remove (now justified by verified no-effect, not a hunch); helps for a real reason →
   correct the record (re-key / re-explain) and KEEP; challenge fails → KEEP. (A fishy KEY — an unfaithful
   fence/proxy — is hunt 4 instead; this is a fishy mechanism behind a possibly-faithful lever.)
   The ONLY thing off-limits is a gate/fact the provenance log shows is still live AND established-necessary.

For each proposal: state the SIMPLER replacement and the property it keys on. It is EXPECTED to change
configs — re-bench under the §3 acceptance model (don't require a 0-diff). The only limits: no NEW
disaster, no >10%-vs-anchor collapse (a perf dip otherwise is fine — generality > perf), and don't break
a KNOWN-VALID invariant (an established-necessary gate, a fact a live lever needs). A proposed refactor is
NOT self-certifying — it re-enters the loop (Gate R + D/H) like any edit, which re-proves anything it
removed on suspicion. Every removal WRITES A RE-ADD RECIPE to the removed-heuristics-log (boldness is safe
because it is reversible); if Gate R later rejects a removal because the lever still wins, that is the
signal to re-key/re-justify it on a faithful property (try-harder / Gate F), not to silently restore it
unchanged. (For a cascade, "a fact a live lever needs" is judged on the POST-refactor state — hunt 3.)

Record AS-RETURNED: {proposals:[{what_collapses, unifying_property, the_single_rule, validation_path,
expected_simplification}], dead_weight:[...], obsolete:[{lever, banked_at_SHA, recorded_reason (quoted), confidence: reason-moot|speculative|reason-disputed, why_moot_or_disputed, action: remove-with-readd-recipe|reopen-via-GateF-rebench, readd_recipe}], reasoning}. Each proposal goes to the BROADEN-and-refactor
queue (method §6); pursued as Priority-2 standing work, never ahead of a live disaster.
```

---

## HELPER — Completeness-critic (loop-until-dry). Fire on a cadence — every K levers, whenever the disaster worklist empties, and before any dead-end is accepted (NOT "near the end", which is undefined for a never-stopping run).
Analysis only (no GPU).

```
Find what the run has MISSED. Default to "something is uncovered."

THE STATE: {per-shape status table, per-(kernel,dtype) geomean table, the ledger index, the hard-pile, the BROADEN-and-refactor queue, the human-review queue}
IMMUTABLE COMMIT: {SHA}

Hunt every gap: a dtype not swept; a claim banked but never independently reproduced; a cap/constant
never audited by the Fact-gate or Gate H; **a fact-POPULATION never audited by Gate D Part 3**; a shape
under the noise floor never lifted; a hard-pile or BORDERLINE item never revisited; a changed cell the
Regression-referee never re-benched; **a narrow fence never sent to Gate H BROADEN; a BROADEN/refactor
whose newly-covered realistic regime was never checked for a real named workload to add to the curriculum
(Gate R step 3b); a lagging family the refactor-critic never revisited; a spent generality-license never
re-checked at the next freeze**.

Record AS-RETURNED: {gaps:[{kind, specific_cell_or_claim, ledger_pointer}]}. Each gap is appended to
the worklist and either cleared or explicitly logged-and-skipped — never silently dropped. Re-run on
the cadence until K consecutive passes find nothing new (loop-until-dry).
```

---

### Note for the orchestrator
A **non-verdict** (a stall/API error after the analysis but before the verdict was recorded) is **never
a verdict** — re-fire fresh, never bank or fail on it. On PASS, read only the ledger entry's `{verdict,
ledger-ref}` and continue (re-reading full PASS objects into context is a dominant context sink, §6.1);
only a FAIL — or a BORDERLINE/blocked-park (→ human-review queue + a provisional reversible default) —
changes what you do next.
