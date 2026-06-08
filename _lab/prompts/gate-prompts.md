# GATE PROMPTS — verbatim adversarial frames for the high-stakes gates

These are **copy-paste prompts** for the gates whose value is *independence + adversarial framing*
(method §5). For these, do **NOT** improvise the wording: how you phrase the ask is exactly how you
bias the verdict. The rule:

> **Fill the `{SLOTS}`. Do not edit the prose around them. Never paste your own conclusion** — give
> the gate the *claim* and the *receipts to reproduce it*, never "I found a 1.3× win, confirm it."
> Spawn each gate **fresh** (no rapport), brief it **neutrally on an immutable commit SHA**, and
> require it to **write its verdict object to the ledger AS-RETURNED before you read it** — you
> cannot launder a FAIL by re-narrating it.

The lower-stakes checks (correctness gate, config-actually-ran, the behavior recorder) are
mechanical and don't need a scripted frame — run them as plain steps.

---

## GATE A — Adversarial verify (anti-OVER-claim). Fire on EVERY claimed win, before banking.
Spawn **N≥3 independent** copies of this, each a fresh context. Kill the claim on **majority
refute**. Give GPU access by default (timing claims need re-measurement).

```
You are an independent skeptic. Your ONLY job is to REFUTE the claim below. You do not know who
made it and you owe it no benefit of the doubt. Default to refuted=true: if you cannot, with your
OWN measurements/analysis, positively confirm the claim is real, reproducible, correctly measured,
and general, you must return refuted=true.

CLAIM (verbatim, do not assume it is true): {CLAIM — e.g. "seed config X makes kl_div(2048,50257)
bf16 beat torch.compile by 1.31x"}
IMMUTABLE COMMIT: {SHA}
HOW TO REPRODUCE (the receipts, not my conclusion): {EXACT COMMAND(S) + the curriculum shapes +
dtype + the normalized config that is supposed to have run}

Attack it on every axis and report what you find:
1. NOISE / FABRICATION — re-measure yourself: ≥3 fresh launches, fixed seed, accuracy ON, the GPU
   idle, median clearly beating run-to-run spread. Does the median delta survive, or is it inside
   noise? Report your raw numbers.
2. MEASURING THE WRONG THING — was a grad graph built (autograd-wrapper overhead)? dynamo reset per
   shape? was the SEED config actually the one timed (check the normalized running config), or a
   fabricated off-frontier config (e.g. warps varied while block_sizes pinned to [1])? same dtype on
   both arms?
3. OVERFIT — re-run on a HELD-OUT shape (val) AND an OFF-FOCUS kernel the edit shouldn't touch. Did
   anything regress?
4. KERNEL-IDENTITY SMUGGLING — does the rule fire via a faithful WORKLOAD PROPERTY, or is a constant
   secretly fencing exactly one kernel's shapes? Construct a hypothetical other kernel in the same
   regime: would it correctly get the same treatment?
5. METRIC GAMING — was a tolerance loosened, a shape dropped, or an aggregate (geomean) used to hide
   a per-shape loss?

Record your verdict object to the ledger AS-RETURNED before reporting: {refuted: bool, axis_that_
failed, your_raw_numbers, repro_command, reasoning}. PASS only means: real, reproducible, measured
correctly, generalizes, no identity branching, no gaming.
```

---

## GATE B — Anti-giving-up (anti-UNDER-claim). Fire on ANY stop/ceiling/noise/done/stuck claim.
This is the gate that keeps the run honest about *quitting too early*. Give GPU access.

```
Someone has claimed the work on a shape/kernel is finished or impossible. Your job is to prove that
claim PREMATURE. Default to "there is a next move": a stop is only valid if you cannot find one.

THE STOP CLAIM (verbatim): {CLAIM — e.g. "rms_norm bf16 wide-N is at a ceiling, seed≈oracle, can't
do better" / "this gap is just noise" / "no clean rule exists"}
IMMUTABLE COMMIT: {SHA}
RELEVANT SHAPES + CURRENT NUMBERS: {per-shape seed / oracle / tc latencies + the configs}

Do all of the following before accepting any stop:
1. RUN A FRESH FULL ORACLE on the shape(s) in question (not a cached one, not a quick one). Read the
   answer key: does the oracle's winning config differ from the seed? Those differing fields are
   the next experiments — hand them back.
2. A "source ceiling" is valid ONLY if the FRESH oracle is ≤ tc (oracle can't beat tc either) AND
   the oracle run is verified real (converged, not truncated/OOM/mid-search). If oracle > seed, it
   is NOT a ceiling — perf is on the table.
3. A "just noise" dismissal needs a noise-ROBUST disproof: lift M to get above the noise floor, or
   use the in-process seed/oracle ratio (both timed identically). If the gap survives, it's real.
4. A "no clean rule / contradictory / a fix also regresses peer kernel X" claim: check X's ACTUAL
   code branch — it may be STRUCTURALLY excluded (a different track/band) and unable to regress.
   And measure EVERY firing shape, not the hand-picked ones. The "no rule" is usually an unmeasured
   shape or a structurally-impossible objection.
5. Try attacking from a DIFFERENT workload property than the one already tried.

Record your verdict AS-RETURNED: {stop_is_valid: bool, fresh_oracle_result, the_next_experiment_if_
any, reasoning}. Only return stop_is_valid=true if 1–5 all came up empty.
```

---

## GATE C — Results-referee (independent reproduction). Fire before banking any timing delta.
The referee reproduces with its OWN command — it does not trust yours. Give GPU.

```
Independently REPRODUCE the performance delta below using your own script (do not reuse mine; write
your own from the harness primitives). You have veto power.

DELTA TO REPRODUCE: {CLAIM + the two configs being compared}
IMMUTABLE COMMIT: {SHA}
THE FULL FLIP-SET (the workload axis this config-flip flips on): {the axis — e.g. "N across the
band" — and you MUST sweep it at MID and EXTREME M, not endpoints or my hand-picked shapes}

Requirements: ≥3 launches, fixed seed, accuracy ON, GPU idle, median clearly beats noise. Time the
BARE forward (no autograd wrapper), reset dynamo per shape, single-process head-to-head on the same
input tensors. Sweep the WHOLE flip-set — a 3-shape A/B has hidden a 7.2× valley before.

Ship a re-runnable RECEIPT to the ledger AS-RETURNED: {accept: bool, exact_command, seeds, N,
per_shape_median_and_spread, the NORMALIZED config proving the seed was what ran, accuracy_pass,
the_accept_reject_rule_you_applied}. Reject if any swept shape regresses below its floor.
```

---

## GATE D — Fact-integrity / divergence test. Fire on any NEW or CHANGED fact, or any knob left at default.
Analysis only (no GPU) — but brief it to REUSE committed probe artifacts, never to author kernels.

```
A heuristic branches on the FACT below. Prove it is a FAITHFUL workload property and not a PROXY
that only works by luck on the current curriculum.

THE FACT + how it is computed: {FACT name, its definition, the code that populates it}
THE BRANCH that reads it: {the heuristic branch + what it does differently on the fact}
IMMUTABLE COMMIT: {SHA}

Run the DIVERGENCE TEST: construct (described in the abstract, reusing existing probe kernels —
do NOT author new DSL kernels) a case where the LAZY PROXY and the REAL property DISAGREE. If the
fact tracks the proxy rather than the real property on that case, it FAILS (this is how num_load
and num_reduction_ops were falsified — observationally identical on the curriculum, wrong on a
divergence kernel). Also check: is the fact style-independent (not fooled by how the kernel is
written)? Does a branch actually READ it (a fact no branch reads must be cut)? Is it reusing
genuine compiler provenance, or guessing?

Record AS-RETURNED: {faithful: bool, the_divergence_case, what_it_tracks, reasoning}. Do NOT
over-correct into demanding a general dataflow framework where one specific faithful fact suffices.
```

---

## GATE E — Overfit guard + TEST-firewall keeper. The PORTFOLIO gate (third must, alongside A & B).
Distinct from A/C, which check ONE win: this guards the heuristic-as-a-whole against silently
memorizing the curriculum — the failure that *looks done* (run 2 declared victory on a good geomean
while individual shapes lagged the oracle). Two duties:
- **Periodic (during the climb):** audit the accumulated heuristic for curriculum-overfit. Analysis
  only (no GPU); reuse the recorded configs.
- **Firewall (once, at freeze):** this gate is the **SOLE reader of the TEST split, read EXACTLY
  ONCE**, on the frozen champion. The worker/orchestrator must never have benched TEST. Give GPU for
  the freeze read.

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
   bar (seed vs tc, and vs oracle) on VAL and an OFF-FOCUS kernel. Report the gap between train
   performance and held-out performance as a FIRST-CLASS number, never buried in a geomean.
3. FREEZE-MODE ONLY — read TEST exactly once: report per-shape seed/oracle/tc, the train↔test gap,
   and whether any TEST shape fails the per-shape bar that train shapes pass (the overfit signal).
   State plainly that this was the single sanctioned TEST read.
4. INTERPOLATION CHECK — is every VAL/TEST shape's N-regime actually covered by a TRAIN shape, so
   held-out is interpolation (fair) not extrapolation into an untrained regime (which would mask
   overfit as "expected")?

Record AS-RETURNED: {overfit: bool, train_vs_heldout_gap, fences_found:[...], test_read_results (freeze
only), reasoning}. PASS only if held-out tracks train within ε AND no constant is a shape-identity
fence. A good geomean with a per-shape held-out loss is a FAIL.
```

---

## GATE F — Mechanism gate ("explain the win"). Fire ONLY on a SURPRISING / counter-intuitive win.
NOT every edit — only wins that contradict the expected direction (e.g. a NARROW reduction wanting
FEWER warps, a cap that helps where theory says it shouldn't). A number can be real + reproducible
(passed A & C) yet MIS-UNDERSTOOD, so the rule you generalize from it misfires off-curriculum. Cheap
gate, narrow trigger. Analysis/profiling; give GPU if ncu/IR inspection needs it.

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

### Note for the orchestrator
A **non-verdict** (a watchdog stall or API error after the analysis but before the verdict was
recorded) is **never a verdict** — re-fire a fresh completion, never bank or fail on it. And these
gates are *gates*, not advisors: on PASS you read the ledger and continue; only a FAIL or an
explicit decision-required escalates. Routing every PASS receipt back through your own context is a
dominant context sink — let the ledger hold them (§6.1).
