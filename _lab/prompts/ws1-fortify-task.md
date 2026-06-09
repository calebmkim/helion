# TASK (WS1): fortify the reduction seed heuristic — stress its overfit levers with sibling kernels

**Read in order (same dir): `hillclimb-method.md` (the durable method, gates A–F, footguns,
orchestration) → `local-setup.md` (this machine: paths, GPU, scripts, env) → `gate-prompts.md`
(verbatim adversarial-gate frames) → this file (the *what* for this run).** These four are
authoritative; where any other doc/notebook/comment disagrees, they win — and so does the live code.

**Resume context — fresh climb.** There is no prior log for *this* workstream. Per the method's
START-HERE "starting from scratch" path: create a fresh notebook `_lab/ws1_notebook.md` + a fresh
ledger key `ws1`; do NOT overwrite the run3/dtype lineage (reference only). Run on your **own
branch/worktree**.

**The inherited reduction heuristic (the dtype-climb champion, serving 9 forward kernels at
fp32/bf16/fp16) is your DO-NOT-REGRESS baseline.** This task hardens the EXISTING heuristic.
Adding a small faithful fact to narrow an overfit lever IS in scope; building a new reduction-fact family is not.

---

## Background — what "fortify" means

The reduction seed heuristic has several levers tuned against a **single** kernel — "families of
one." We can't tell whether such a lever encodes a *faithful workload property* or a
*curriculum-lucky fit*, because only one kernel ever exercises it. Your job: **add sibling kernels
that land in the same regime as each overfit lever, then prove (under the gates) whether the lever
GENERALIZES (helps the sibling too) or is OVERFIT (only helps the original)** — and keep / faithfully
narrow / generalize the lever accordingly, never regressing the existing 9.

This is part authoring small forward kernels + running the **matched-lever A/Bs** (method §3 —
perturb DOWN from the good config) + hardening levers under Gates A/D/F and the no-regression
backstop — and part **adversarially hunting for further non-robust levers/fences beyond the named
suspects** (next section).

## Overfit hunting — known suspects to START with, then hunt BROADER

The levers below are the *currently-known* family-of-one suspects (three, to start) — a **starting point, NOT the whole
worklist.** Your real charge is to **adversarially hunt for ANY non-robust lever or overfit fence in
the heuristic, including ones not named here**: periodically spawn a skeptic (Gate-E style) to poke
holes in the whole heuristic + the kernels — does any constant fence exactly the curriculum's shapes?
does a lever lean on a curriculum-lucky proxy? would some other sibling kernel expose a different
brittle path? Treat the table as the *seed* of the hunt, then push past it, if you feel there are other suspects.

| lever (in `triton.py`) | family-of-one | sibling probe | the experiment |
|---|---|---|---|
| **`REREAD_W8_MAX_BYTES` + the w8 branch** in `_num_warps` (~357-402) | **cross_entropy alone** | **logsumexp** (+ an fp32 control) | logsumexp bf16-wide **fires the w8 branch** (the gate keys on `row_reread AND not full_width_output`, NOT num_load — confirmed: logsumexp's one load feeds amax+sum → row_reread=True, scalar out → full_width_output=False). Does w8 *beat* w32 on logsumexp at the bf16 vocabs (**generalizes**) or only on CE (**overfit**)? Also check the fp32 byte-cap boundary (V∈[16384,25600] fp32 should fire+benefit; V≥30522 fp32 stays w32). |
| **`STRUCTURED_COMBINE_CAP_BYTES`** + normalize-tile `persist_max_bytes`/`loop_chunk_bytes` (Band C, ~711-721, ~441-444) | **welford alone** | **groupnorm fwd** (the clean welford-bandmate) | Does the combine cap + normalize-tile sizing generalize to groupnorm's different combine arithmetic + M/N regime, or is it welford-spill-specific? (The code itself flags these caps as an M-unaware PROXY that regressed welford(262144,5120) ~7.3× when loosened — so this is the highest-risk cap.) **batchnorm is NOT a bandmate** — it reduces over a different axis than it applies → secretly multi-axis → belongs in WS2; do not use it here. |
| **`persistent_interleaved` + `maxnreg=64`** (T1 tail, ~633-653) | **cross_entropy** (+ softmax at very wide N; rms/ln never reach it — they stay persistent below the 240 KiB byte cap) | **log_softmax** wide-N | persistent_interleaved fires only on wide **LOOPED re-read** rows (bytes > `ROW_PERSIST_MAX_BYTES`=245760). CE (scalar output) is the tuning target. log_softmax is the **untested FULL-WIDTH** case. Does interleaved + `maxnreg=64` help a full-width store, or is it CE-reread-specific (maxnreg a no-op/regression on log_softmax)? |

**Examples of cheap coverage probes** (non-exhaustive; low overfit risk, still worth landing): **l2_norm** (the
`ROW_PERSIST_MAX_BYTES` byte-boundary on a streamed single-load reduction — currently only long_sum
lives there); **row_max / row_mean** (the `_num_warps` element ramp with no override branch).
**argmax** is "free new-fact coverage" but needs an integer-output accuracy path (see traps).

## Per-sibling protocol — transfer-first, then climb

For each sibling, the FIRST measurement is its **untuned transfer perf**: run the *existing*
heuristic's seed on the sibling (all shapes, all 3 dtypes) **before any tuning**. That initial number
IS the overfit-generalization signal:
- **Good initial perf → the implicated lever generalizes** (bank the verdict).
- **Poor initial perf → FLAG it with a note** (first-class evidence that the lever is overfit / the
  heuristic doesn't transfer to this kernel).

Then **branch on the initial result** (the done-bar is defined in Goal):
- **Initial geomean already clears the bar** (beats tc, or within-margin-of-error of the oracle) →
  **DONE, do NOT hill-climb** — the lever generalized; record the number and move to the next
  sibling/lever. (The hoped-for common case.)
- **Initial geomean fails the bar** → the lever did not generalize: **hill-climb on the
  train/val/test split** (train = develop the fix, val = mid-climb overfit check, test = Gate-E
  firewall once) to harden/narrow the implicated lever until the geomean clears the bar. Do NOT
  exhaustively climb to per-shape beat-tc / oracle parity.

Either way, **record the initial transfer perf** (the probe / overfit signal); record the post-climb
perf too if you climbed. A poor transfer number is never a stop — flag it, then climb (§6.0).

## Goal (apply the method's goal hierarchy, §3)

**The fortify climb is BOUNDED and often SKIPPED entirely — this is the KEY difference.**
These siblings are *probes* for the overfit hunt, not perf-championship targets, and the existing
heuristic should make most of them good **out of the gate**. The **done-bar is per-kernel GEOMEAN**
(across all 3 dtypes, tuned together — no inherited fp32 champion to transfer from): a sibling is DONE,
**move on**, when its geomean either
- **beats/matches torch.compile** (geomean `seed ≤ tc`), OR
- **loses to tc but is within margin-of-error of the oracle** (geomean `seed ≤ oracle × ~1.05`, i.e.
  within do_bench noise of max-autotune — Helion can't beat tc on this kernel, but the seed is already
  at the reachable ceiling).

**Crucially: measure the INITIAL untuned transfer perf FIRST — if it already clears the bar, do NO
hill-climbing** (the lever generalized; record the number and move on — this is the hoped-for common
case). You **only hill-climb when the initial transfer perf FAILS the bar** (the lever did not
generalize). Don't be strict per-shape — the geomean is the gate; you *may* revisit an individual
worse shape after everything else is cleared, but it is NOT required. This is a **deliberate,
authorized RELAXATION of the method's strict per-shape done-bar and its geomean-trap caution (§3),
scoped to these probe siblings** (the task file wins on scope). It does NOT relax §6.0: "move on" means
advance to the next sibling / lever / the broader overfit hunt — the run never halts. Two guards still
hold: the no-regression backstop on the existing 9, and a **catastrophic single-shape outlier on a
sibling is itself an overfit signal — flag it** even if the geomean is fine.

AND for **each known overfit lever — plus any further non-robust lever or fence your hunt surfaces** —
a **gate-verified verdict**: *generalizes* (bank confidence; keep, possibly widen) or *overfit* (faithfully **narrow** its firing via a workload
property — never a kernel-identity fence — or **generalize** it properly). **Do-not-regress the
existing 9 kernels at all 3 dtypes** — the no-regression backstop (§3) spans the whole banked matrix,
and the overfit/TEST-firewall gate (E) reads TEST once at freeze.

## The code & prime directive

Heuristic: `helion/_compiler/autotuner_heuristics/triton.py`. Facts:
`helion/autotuner/config_spec.py` (`ReductionFact`), populated in `device_ir.py`. All yours to change
aggressively (method §2) — subject only to the gates.

- **Faithful workload properties only** — never a `dtype`-kind branch or a kernel-identity fence
  (Gate D, the fact-integrity divergence test, hunts exactly this).
- A "generalizes" verdict needs **Gate A** (adversarial-verify) + **Gate D** (if you touched a fact)
  + **Gate F** (only if the win is counter-intuitive). An "overfit → narrow it" change is governed by
  the **no-regression backstop** (don't regress CE/welford/the 9).

## Where everything lives

- **The 9 kernels:** `examples/`. **The siblings you AUTHOR** (their curriculum shapes are
  pre-specified — see Curriculum below): `logsumexp`, `log_softmax`, `groupnorm`, `l2_norm`, `argmax`
  (+ optional stretch siblings `row_max`/`row_mean` — author them only AFTER the core siblings + all lever verdicts are banked; never block on them, and never ask the human). Templates: `examples/sum.py` (scalar-output T1),
  `cross_entropy.py:70-74` (the stable logsumexp body), `softmax.py` (full-width log_softmax),
  `welford.py` (the Band-C groupnorm). **CRITICAL: accumulate in fp32**
  (`x.to(torch.float32) … .to(out.dtype)`, like `sum.py:44-50`) or the bf16/fp16 accuracy gate fails
  — the #1 footgun for sum-family siblings.
- **groupnorm must be written WELFORD-IDIOMATICALLY** (a combine loop + an apply loop over the SAME
  within-group extent) or it will NOT hit Band C (the `_non_reduction_loop_candidates` gate at
  `device_ir.py:1267-1269` requires the apply loop's extent == the reduction extent). **Assert the
  resulting `ReductionFact` has `non_reduction_loop_block_ids` non-empty before trusting the seed.**
- **Harness — tritonbench is the method's PREFERRED headline; a hand-rolled harness is a CROSS-CHECK,
  not automatically the headline** (§4 footgun #3: tritonbench handles the footguns for free — attr +
  dynamo reset per input, accuracy, the tc baseline, and device-time modes `--cudagraph` /
  `--latency-measure-mode gpu_events`). The new siblings have **no tritonbench operator**, so two paths:
  **(a)** author a tritonbench operator per sibling (most method-canonical; more setup; the fp32-only
  `_lab/bench/seed_vs_tc.py` wrapper would need a dtype axis threaded through to drive it); or **(b)**
  extend the already-dtype-aware `_lab/bench/bare_fwd_dtype.py` (add a build-fn + `KERNELS` entry +
  torch baseline, e.g. `torch.logsumexp`) — the lighter path, but it has **NOT been shown better than
  tritonbench**; it's a dtype-aware convenience, so you MUST pass the §0 setup gate (its number agrees
  with tritonbench / a single-process do_bench within 3%) and keep tritonbench as the anchor. Either
  way honor §4's footguns (forward-only, dynamo reset per shape, single-process, fp32-accumulate
  accuracy gate) AND the tritonbench guardrails below.
- **TRITONBENCH RELIABILITY GUARDRAILS** — the prior climb logged tritonbench producing FALSE results;
  "reliable" means "used correctly." Heed every one:
  1. **Device-time mode, NOT the `do_bench` default** (`--latency-measure-mode gpu_events` or
     `--cudagraph`): the default `triton_do_bench` charges ~20µs Python host-enqueue to the kernel at
     low-M / bandwidth-bound shapes → **phantom losses** that vanish on device time (rms_norm bf16
     (2048,4096): do_bench 0.76 → device 1.06).
  2. **NEVER cudagraph a `reduce-overhead` tc baseline** (it self-graphs → an outer graph garbles it,
     8µs→30µs). Compare to **tc_default** (what tritonbench uses); device-time modes sidestep it.
  3. **Gate ACCURACY first — tritonbench reports perf on accuracy-FAILING kernels and emits false
     `acc=0`.** Verify accuracy yourself; compare helion-at-dtype vs torch at the SAME dtype (isolates
     the kernel's dtype floor from the seed); near-zero-output "fails" are tolerance traps (use
     max_abs, not max_rel); EXCLUDE NaN rows (fp16 wide-V) from geomeans.
  4. **One isolated process per kernel; one timing run per GPU** — batching compiles in a long-lived
     process corrupts the tc baseline (dynamo recompiles; this fabricated a bogus 2.18× "win").
  5. **Apples-to-apples: bare forward, both arms no-grad, dtype FORCED** — an operator's
     autograd-wrapper / grad-mode / hardcoded-dtype mismatch fabricated artifacts (rms_norm 0.79,
     welford 3.19); some operators default to fp16 (softmax) — set + assert the dtype.
  6. **tritonbench's baseline is `tc_default`** (not max-autotune, not reduce-overhead); **seed-vs-
     unseeded-Helion-default is a DIFFERENT, inflated axis** — report seed-vs-tc.
  7. **The timer is NOT itself biased** — if a number looks anomalous, suspect your *analysis* (a
     field-diff that re-benched a fabricated config produced a bogus 33× "artifact"); re-bench the
     FULL verbatim config; treat <25µs as noise floor and re-run any >5% do_bench spread.
- **Curriculum — shapes are PRE-SPECIFIED; do NOT choose your own.** The WS1 sibling shapes live in
  `_lab/prompts/shapes_v3_draft.py` with a **train / val / test** split per kernel (logsumexp,
  log_softmax, groupnorm, l2_norm, argmax). **Use them exactly as given** — do not add, drop, or
  re-split shapes. Discipline (mirrors the main curriculum): **train** = develop a lever fix here;
  **val** = re-measure during the climb to catch a fix overfitting train (iterate if it regresses);
  **test** = read ONCE at freeze by the overfit/TEST-firewall gate (E), never benched during the
  climb. Run the CPU `validate()` to confirm invariants before relying on a split. This split discipline
  applies to **every** authored sibling — including `row_max`/`row_mean` and any new sibling the hunt
  motivates — not only the five listed; **test is NEVER benched during the climb (only Gate E reads
  it, once, at freeze).** (These are WS1 *tuned* siblings: `l2_norm`/`argmax` move OUT of the old flat
  `TRANSFER` list into this train/val/test structure, the **test split serving as the held-out
  generalization check** — superseding their prior "untuned transfer probe" classification.)

## Correctness traps (method §4)

- **fp32-accumulate-in-kernel** or the half-precision accuracy gate fails silently → kernel excluded
  as `acc_fail` (the #1 footgun).
- **argmax:** int64 index output — the float `allclose` gate is WRONG for indices. Use this contract
  (pick the most defensible per §6.0; do NOT ask the human): compare indices by **exact equality**;
  for tie rows where helion and the reference pick different equal-max positions, fall back to
  comparing the **gathered max VALUE** via `allclose`. Wire an argmax branch into the accuracy gate in
  `bare_fwd_dtype.py`; log the choice.
- **log_softmax / logsumexp over very wide N at fp16** may NaN (5-bit exponent), like the loss
  kernels — add to the fp16 skip-list if so (kernel-authoring, not heuristic).
- **full-width siblings (log_softmax)** route to the softmax-class warp branch, not sum-class — don't
  assume scalar-output behavior.

## Deliverable (DoD — a milestone to BANK, then keep climbing, §6.0)

1. The sibling kernels authored + correct at all 3 dtypes (measured/justified tolerances).
2. For **each known overfit lever — and any further non-robust lever/fence the adversarial hunt
   surfaces** — a gate-verified **generalizes-or-overfit** verdict, the lever kept / faithfully-
   narrowed / generalized accordingly, and the existing 9 kernels **not regressed** at any dtype
   (no-regression backstop + Gate E at freeze).
3. Each sibling either **already clears the done-bar out of the gate** (geomean beats tc, or is within
   margin-of-error of the oracle — recorded, NO climb needed) **or is climbed just until it does**;
   NOT exhaustively tuned (that's WS2). The primary deliverable is the lever verdicts (item 2) + the
   initial transfer numbers, not per-sibling perf maximization.
4. A short report: **the initial untuned transfer perf per sibling (flagged where poor — the overfit
   signal)** AND the post-climb G vs tc/oracle at 3 dtypes; per-lever verdict + the faithful property
   if you narrowed one; what the overfit audit found.

**Scope boundary:** the deliverable lives as a validated lab result in the worktree (changes +
proofs + report) — bank it and keep climbing (§6.0). Do **not** `git push` / touch the PR.
