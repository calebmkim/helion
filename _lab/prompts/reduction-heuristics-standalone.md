# Reduction Seed-Heuristic — Hill-Climbing Work Order (standalone)

> Authoritative for this work: **this file** + **`_lab/prompts/shapes_v3_draft.py`** (the kernel list +
> the train/val/test/robustness/transfer splits — the single source of truth for *which* shapes you tune,
> validate, and report on). All file:line references anywhere are **approximate — grep the symbol, don't
> trust the number.** Paths/GPU indices/interpreters differ per machine — **discover them, never hardcode
> them** (see Setup).

## Goal

**You are the orchestrator for this run.** You own acceptance and spawn the helper agents (worker,
investigators, referee, auditors — see "Orchestration"); you do **not** do the heuristic work in your own
context. The reason is structural: acceptance must be gated by a context *separate* from the one that
produced the result, and you are that separate context. Keep the run going (see "Never stop"); assume the
human is unavailable.

The objective: write **general** compile-time autotuner **seed heuristics** for **forward inner-reduction**
kernels on **H100 / sm_90, fp32, Triton backend**. A heuristic reads pre-digested **workload facts** about a
kernel (`ReductionFact`) and emits **one seed `Config`**. It serves two products:

- **Product A — skip autotune:** the seed *is* the config (`configs=[seed]`, no search). The bar is
  per-shape and two-tier (see "Objective & acceptance"): at minimum match-or-beat `torch.compile` default
  (within noise); the real target is to **match the autotuner's own best config** (the *oracle*).
- **Product B — seed the autotuner:** the seed is injected into the autotuner's initial population so a
  search reaches a good config in **less wall-clock** (shrink the budget).

The work is **hill-climbing**: read the answer key, form a workload hypothesis, change the fact/heuristic,
measure, gate, commit, repeat. Commit early and often (dozens/hundreds of commits is fine). Never `git push`.

## What exists, and what is freely rewritable

An existing seed heuristic already covers the forward inner-reduction kernels:

- the heuristic: `helion/_compiler/autotuner_heuristics/triton.py` (`TritonReductionHeuristic`,
  `get_seed_config`, registered under `HEURISTICS_BY_BACKEND['triton']`);
- the fact it consumes: `ReductionFact` in `helion/autotuner/config_spec.py`;
- where the fact is populated during compile: `helion/_compiler/device_ir.py`
  (`register_rollable_reductions` for T1, `register_user_tiled_reductions` for T2,
  `_count_reduction_workload`).

**Do not care how that heuristic was arrived at, and do not trust it.** Treat it as a *starting point*, not
a finished artifact. **Both the heuristic's branch structure AND the `ReductionFact` vocabulary are yours
to rewrite aggressively.** Do not assume any existing fact is correctly computed, or that any existing
branch keys on the right property, or that any constant is right. The only thing you may not weaken is what
an independent gate has confirmed (correctness, a reproduced win) — see "Orchestration." Aggressive
rewriting is *safe precisely because acceptance is gated on the oracle* (below), which catches a regression
that an aggregate-speedup number would hide.

---

## Objective & acceptance — the two-tier bar (read this twice)

For each shape, with `seed_latency` = the bare seed (`configs=[seed]`, correctness-gated) and
`tc_default_latency` = `torch.compile` default:

- **`G = tc_default_latency / seed_latency`** (`> 1` ⇒ seed beats tc-default).
- **The FLOOR (baseline):** `seed` within ε of `tc_default` or better (`G ≥ 1 − ε`). Being within
  `do_bench` noise of tc-default **passes** the floor — you do not have to strictly beat it. Necessary, but
  **not victory.**
- **The ORACLE (answer key):** the best config the Helion autotuner finds for that shape, plus its latency.
  This is the *achievable* on this kernel/shape, and — critically — it is a **readable config** you can
  field-diff your seed against.
- **VICTORY (the real bar):** `seed_latency ≤ oracle_latency × (1 + ε)`, i.e. `seed/oracle ≤ 1 + ε`, with
  **ε ≈ 3–5%** (treat ≤3% as a tie; ≤5% generously a tie). The seed *matches the oracle*.

**Done = VICTORY on EVERY measurable shape. No exceptions and no escape clause.** In particular:

- **A kernel-source limit is never an excuse for a seed↔oracle gap.** A source ceiling (e.g. a kernel that
  re-reads the row where a fused kernel wouldn't, or a single-program-per-row layout that can't saturate the
  GPU) caps the **oracle too** — both `seed` and `oracle` lose to `tc_default` together. So a source limit
  can only ever explain a `seed`-vs-**tc-default** gap; it can **never** explain a `seed`-vs-**oracle** gap.
  If `seed < oracle`, the heuristic is leaving achievable performance on the table — period.
- **`seed ≈ oracle < tc_default` means the heuristic has done its job.** The residual is a *kernel-source*
  signal, not a heuristic failure. It is an **optional, separate** Product-A-via-source-rewrite opportunity
  (correctness-gated; historically the highest-leverage wins come from here), but it does **not** block
  victory and is not part of the seed deliverable.
- **The aggregate geomean `O = geomean_k(geomean_shapes(G))` is a SUMMARY, not the bar.** Reporting `O` is
  fine; *declaring done because `O` looks good* is a failure mode (it averages the tail away). The bar is
  **per-shape seed≈oracle**, not an aggregate vs tc-default.

Why two bars and not one: the floor and the victory bar **disagree on the shapes that look finished.** A
shape can beat tc-default outright (`G > 1`) and still be far from the oracle (e.g. the oracle is 1.3× the
seed). Measuring only against tc-default makes such a shape read as "won" when a third of the achievable
speedup is unclaimed. Always compute `seed/oracle`, not just `G`.

### Oracle discipline (the answer key is the whole method)

- **Measure the oracle ONCE per `(kernel, shape, kernel-source-hash)` and cache it** in the ledger
  (`{winning_config, latency_distribution}`). Thereafter, comparing the seed against the **cached** oracle
  is *free* every iteration — re-bench only the seed. This is what makes oracle-gating affordable.
- **Use it as an answer key, not just a number.** Field-diff the seed against the oracle's winning config
  (`reduction_loops`/`block_sizes`, `num_warps`, `num_stages`, `load_eviction_policies`, `indexing`,
  `pid_type`, …); the differing fields *are* the worklist. A/B the differing fields one at a time
  (matched-lever; see below).
- **Cheap-first.** Quick-autotune is the iterating proxy; full-autotune confirms a shape is closed. Because
  you have **one GPU** and a full oracle can take minutes/shape, oracle **one representative shape per
  `(kernel, N-band)`** while iterating, and run the fuller per-shape pass when confirming victory. (Bands
  are in `shapes_v3_draft.py`.)
- **Staleness is fatal — the oracle is the sole victory signal.** Invalidate the cached oracle for a kernel
  on **any** edit to its source (`examples/<kernel>.py`) or to codegen that changes its generated Triton. A
  stale oracle silently moves the goalposts. Key the cache by a source-hash and re-validate the final
  champion against a fresh oracle.
- **Unavailable / OOM fallback chain.** If a full Helion oracle cannot be produced for a shape:
  full-Helion-autotune → **quick-Helion-autotune best** (still a readable Helion config) → **`torch.compile`
  max-autotune** (latency-only; you lose the readable answer key, so this is the last resort). Note: a
  single config that OOMs during a search does **not** end the search — config trials are subprocess-isolated,
  scored `inf`, and skipped, and the search returns the best *non*-OOM config. So **"the oracle OOMed" is
  almost never true unavailability** — verify it (it usually means the *harness* OOMed holding inputs, or a
  codegen-knob subset OOMed while a valid winner existed). Treat "no oracle available" as a claim to be
  falsified, not accepted.

---

## Operating rules

- **Generalize on workload, never on kernel identity.** Branching on reduction **shape/workload
  properties is required** (`if rnumel < X`). Branching on **kernel identity is banned** (`if kernel ==
  rms_norm`). When two kernels want different treatment, find the *workload property* that distinguishes
  them, expose it as a `ReductionFact` field, and branch on that. A magic threshold that happens to fence
  off exactly one kernel's shapes is kernel-identity smuggling — also banned.
- **Facts must be principled, not hacky** (this is as important as "no kernel identity"; see the failure
  catalog §11–13). A fact is **hacky** when it uses one easily-available thing to *stand in* for the
  property it is really trying to capture — e.g. testing `shape[-1] == size_hint` ("this dim *equals* the
  reduction extent") as a proxy for "this dim *is* the reduction axis" (a non-reduction dim that merely
  happens to equal the extent is then mis-classified), or counting `load` ops as a proxy for "live bytes
  resident across the reduction" (two reads of one tensor may be fused into one op, so the count tracks
  coding style, not the workload). The cure for a hacky fact is to compute the **real** thing — and the
  real thing is almost always already sitting in **compiler provenance** (block-id / host-buffer identity /
  loop-region dataflow), so prefer reusing that. If, and only if, an **adversarial agent confirms** the
  property is genuinely *not* recoverable from any existing provenance, **write your own analysis to compute
  it** — that is strictly preferred to leaving the fact hacky or abandoning the property. Every fact must
  also be **used by a branch** (a fact no branch reads is dead weight — cut it).
- **Justify every branch with an empirical/physical *why*** in a comment, grounded in the answer-key diff
  or a perf dig. "It made kernel X faster" is not a why; "wide multi-load rows re-stream per load pass and a
  resident whole row spills → loop above N bytes" is.
- **Correctness is the first filter, every iteration.** `configs=[seed]` **bypasses the autotuner's own
  accuracy check**, so run your own against an eager/reference baseline at the operator's tolerance before
  comparing latency. fp32 reduction-order drift can legitimately exceed tight tolerances — any tolerance
  change must be **logged, never silent**.
- **Precision is fixed at fp32** for this whole run; **assert it on every benchmark call** (some operators
  default to fp16, e.g. softmax — override). The heuristic must read **`dtype`/`itemsize` as fact fields**,
  never hardcode fp32, so it stays dtype-general for a future bf16/fp16 expansion.
- **One GPU.** Pin to it, verify it is idle (`nvidia-smi`) before trusting a number, and **time serially**
  (never two timing runs concurrently — it corrupts `do_bench`). A single dedicated idle GPU gives *cleaner*
  `do_bench` than a shared one; take **median-of-N with spread**, and re-run any shape with large spread.
- **Measure via the established benchmark harness** (TritonBench `do_bench`) as the headline; hand-rolled
  timers are sanity checks, not headline numbers.
- **Commit early/often on a dedicated worktree + branch; never `git push`** (a human handles that). Keep all
  lab state (ledger, notebook, harness, logs) **machine-portable** — no hardcoded paths/SHAs/GPU indices.

---

## Curriculum & the firewall (`_lab/prompts/shapes_v3_draft.py`)

`shapes_v3_draft.py` is the **single source of truth** for the kernel set and the per-kernel splits. Run its
validator (`python _lab/prompts/shapes_v3_draft.py`) and respect its invariants (measurable splits clear the
noise floor; train covers every N-band that val/test probe; splits disjoint). The splits and how you may use
each:

- **`train`** — the only shapes you **hill-climb on**. Realistic, model-anchored, measurable.
- **`val`** — read **only at adversarial checkpoints** to detect overfit; **never** drives an edit.
- **`test`** — **read exactly once, at the very end**, on the frozen champion. The headline generalization
  metric is the train↔test gap; a large gap is overfit — **report it, don't bury it.**
- **`robustness`** — **correctness-only canaries** (tiny / prime / non-pow2 / extreme / grid-bound). **No
  perf/G claim** is made on these. They must be *correct and not catastrophically slow*, but they are not
  perf targets — the perf curriculum deliberately optimizes the shapes real workloads actually run.
- **`TRANSFER`** — kernels the heuristic is **not** tuned on. Reported as a **separate** "generality to an
  unseen kernel" number; **never folded** into a per-kernel headline.

Promoting any held-out shape to a tunable split is a deliberate, documented change — log it; never tune on
val/test silently.

---

## Failure modes — the ways an agent fools itself, in BOTH directions

These are not accusations and not exhaustive — they are the catalog of mistakes that are **easy to make
without noticing**, which is exactly why **acceptance is gated by a context separate from the one that
produced the result.** The first group is over-claiming (showing a win that isn't there); the second is
**under-claiming / giving up** (declaring a real gain unreachable); the third is a foundation defect that
*no perf gate can see*. All three are equally disqualifying.

**A. Over-claiming**
1. **Fabricated / noise.** A delta within run-to-run noise reported as a win.
2. **Measuring the wrong thing.** The benchmarked kernel didn't actually use the seed (silently dropped /
   normalized away); a mismatched baseline, precision, or set of flags across the two arms.
3. **Overfitting.** Tuning on val/test; a threshold/window that fits the in-sample shapes and falls apart
   on held-out ones.
4. **Kernel-identity smuggling.** A branch that is "if this is kernel X" in disguise — a constant/window
   fencing exactly one kernel's shapes, or a threshold with no workload justification.
5. **Metric gaming.** Loosened tolerance, dropped shapes from the geomean, or autotune budget quietly blown
   to buy latency.

**B. Under-claiming / giving up (equally banned)**
6. **Noise-floor dismissal.** Declaring a gap "just noise" without the **noise-robust proof**: either
   re-measure with **M lifted** so the same N-regime clears the noise floor, **or** use the **`seed/oracle`
   ratio**, which is noise-robust because both arms are timed identically in the same process. "It's
   sub-25µs" is a reason to measure more carefully, **not** a reason to stop.
7. **False ceiling.** Declaring a shape a "kernel-source limit / can't be beaten" without proving the
   **oracle also cannot beat tc-default** on it (`oracle ≥ tc_default`) **and** that the oracle run was
   **real** (not truncated / OOM-aborted / a mid-search snapshot). A source ceiling is a statement about
   `seed`-vs-tc with `seed ≈ oracle`; it is never a license for `seed < oracle`.
8. **"No clean rule" / "the property is contradictory."** This is **never** a fact about the world — only
   about your **current fact vocabulary**. Two settings that look contradictory at the kernel level
   (operand A wants policy X, operand B wants policy Y) are a **missing or too-coarse fact**, not the
   absence of a rule. The fix is **always** to find the finer, principled, provenance-traceable property
   (enrich `ReductionFact`, key at the right granularity), **never** to leave the knob at default. The only
   admissible "no rule" is a proof that the distinguishing value is knowable **only at runtime** — and that
   claim itself must be proven from the source, not asserted.
9. **Premature convergence.** Declaring "done / at ceiling / converged" on the **tc-default aggregate**
   instead of **per-shape `seed ≈ oracle`**. Convergence is reached when every measurable shape meets the
   victory bar against a *fresh* oracle — not when an aggregate looks good.
10. **Unfalsified limitation claims.** Any "X is not available / not cheap / not possible / not exposed"
    (a missing API, an inaccessible provenance, an unbuildable config) must be **killed by a code/source
    search before it is accepted.** A fluent, well-written caveat is *more* dangerous than a silent gap
    because it reads like diligence — verify the premise.

**C. Hacky facts (the substrate failure no perf gate can catch)**
11. **Proxy population (a fact that stands in for what it really means).** Computing a fact from an
    easily-available **stand-in** — a fake-tensor `shape[-1] == size_hint` value/int-equality test, or a
    syntactic `load`-op count — instead of the property it is meant to capture ("is this the reduction
    axis?", "how many live bytes are resident across the reduction?"). A proxy that is *observationally
    identical to the true fact on the current curriculum* will pass every perf gate and still be **wrong**
    on an unseen structure (an outer reduction; a non-reduction dim that merely equals the extent; the same
    reduction with reads fused vs not). "Currently equivalent" ≠ "correct." The cure is to compute the real
    property — first by **reusing existing compiler provenance**, and (only if an adversarial agent confirms
    it is genuinely absent from provenance) by **writing a dedicated analysis**. Writing an analysis is
    strictly preferred to keeping a proxy.
12. **Style-dependent facts.** A fact that changes when the kernel source is refactored **without changing
    the computation** (counting `load` ops, when two reads of one tensor may be fused into one). Two kernels
    computing the same reduction, written differently, must get the **same** fact — the fact tracks the
    workload, not the coding convention.
13. **Consumerless / unfalsifiable facts.** A fact no branch reads (dead weight — cut it), or a fact whose
    intended property is too vaguely defined to state what its value *should* be on a given kernel (so no
    test could ever confirm it tracks that property — demand a precise definition). *Note this is NOT the
    same as a fact for which no proxy/real-property-disagreeing kernel exists — that just means the simple
    computation is already exact, which is fine.*

The license, restated: **rewrite the heuristic AND `ReductionFact` as aggressively as the evidence
demands** (if the answer-key diff says `num_warps` must key on `m_block` and tile bytes, not `rnumel` alone,
then change the fact and the branch). The oracle gate is the guardrail that makes aggression safe. But every
fact you add or keep must be **non-hacky** (it computes the real property, not a stand-in — reusing
provenance where it exists, writing an analysis where it doesn't), **used by a branch**, and **proxy-checked**
(on a kernel that separates the lazy proxy from the real property, the fact tracks the real property, not the
proxy).

---

## Orchestration

- **A single hub** spawns the worker and the gate agents and **owns acceptance**; it keeps a **persistent
  worker** (continued across iterations so it keeps its accumulated intuition). The hub does **not** do the
  heuristic work in its own context. The load-bearing invariant is: **acceptance is gated by a context
  *separate* from the one that produced the result** — that separation is the source of supervisory power,
  because the tired context that wants to declare victory is the worst-placed one to judge whether victory
  is real.
- **The hub is a switchboard, not a thinker — and its context is a scarce resource it actively protects.**
  A hub that routes *everything* through its own context — running its own tools, narrating each step,
  absorbing every gate verdict and worker reply — fills its window and **dies mid-run before the work is
  done.** Context exhaustion is a silent run-killer, and the orchestrator is the worst place to spend context
  because it is the one context that must live longest. So the hub holds only *state and routing*, and
  offloads *work, judging, and thinking* to other contexts. Two hard rules fall out:
  - **Zero heavy in-context (IC) work.** The hub runs **no** builds, greps, multi-KB file reads, edits, or
    benchmarks in its own context — tool output is one of the largest context sinks, and it accumulates fastest
    in the orchestrator. Setup (Step 0/1) goes to a one-shot setup agent; any file/code fact goes to the worker
    or an investigator and comes back as one line. The *only* sanctioned IC file touch is reading the worker's
    one-line **status beacon** (below) — a ~30-token read that buys a heartbeat without a round-trip. "No heavy
    IC" is the rule; the beacon is the deliberate cheap exception.
  - **Spawn the reasoning, don't hold it.** A hard orchestration call reasoned through *in the hub's context*
    costs the hub permanently; the same call spawned to a one-shot **orchestration-advisor** costs the hub only
    the one-line recommendation it returns. So the hub follows the decision table below mechanically, and for
    anything off-table it spawns an advisor and executes the answer — it does not deliberate in-context.
- **Pass the model explicitly on EVERY spawn.** The `opus` alias is **not one global alias** — it resolves
  *differently by spawn path*, and on top of that an omitted model does not reliably inherit. Two distinct
  downgrade traps follow, both verified live 2026-06-03 (independently re-verified against on-disk
  `config.json` + runtime API-call records):
  - **Agent-tool team-member spawn, `model` omitted → silently `claude-opus-4-7`** (NOT inheritance — the lead
    was 4-8 — and NOT 4-8). `model:"opus"` resolves correctly to opus-4-8. So **every** team spawn (worker,
    investigators, gates) passes `model:"opus"`. (Caveat for intuition only: a *plain* sub-agent with no
    `team_name` *does* inherit the parent — but your agents are all team members, so the 4-7 trap is the one
    that applies. Note `config.json` records the omitted member's resolved id as `us.anthropic.claude-opus-4-7`
    but stores the explicit case as the literal alias string `"opus"`; the 4-8 resolution shows up in the
    member's runtime transcript, not the config file.)
  - **CLI `claude -p --model opus` → `claude-opus-4-6`** (a *different*, worse downgrade than the team path's
    4-7 — same token, different resolution). The respawn launch (below) is a CLI invocation, so it must
    **hardcode the full literal model string `us.anthropic.claude-opus-4-8[1m]`** — the `opus` alias is a 4-6
    trap on the CLI path. A mis-resolved successor in a cramped non-1M window is exactly the failure respawn
    exists to prevent, so the successor also **self-asserts its model+context at boot and refuses to run on
    mismatch** (Step-0-style wiring check).
- **Three lifetimes, by role** (lifetime *and* topology are per-role, not global):
  - **Worker — persistent.** Its accumulated context (intuition, tried-and-rejected list, open hypotheses)
    *is* the asset; carry it across iterations. Realize persistence at the highest rung your harness offers
    (see Step 0's discovery ladder). Persistence ≠ immortality: **respawn the worker fresh from the notebook
    at ~50% context or when stuck** — the notebook, not the worker's context, is the source of truth (this
    makes "continue-in-place" and "fresh-worker-per-iteration" the *same* safety model at different
    cadences). Any agent can read its **own** live context occupancy (verified 2026-06-03) — there is no need
    to guess: its session id is in `$CLAUDE_CODE_SESSION_ID`, its transcript is
    `~/.claude/projects/<project-dir>/$CLAUDE_CODE_SESSION_ID.jsonl`, and the last `usage` object's
    `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` is the occupancy (÷ 1e6 = fraction
    of the 1M window; it climbs monotonically, so a plain `≥ threshold` check never flaps). The worker checks
    this itself and flags the hub when it crosses ~50%.
  - **Investigators (code / perf) — reusable peers.** The worker reaches them **directly**, not relayed
    through the hub — they are tools the worker uses, the hub adds nothing to "why is A slower than B," and
    a direct peer call avoids a hub round-trip + hub-context bloat. Mild persistence is fine (they return
    root-causes, not accept/reject verdicts, so the contamination that bans persistent gates does not
    apply). The hub **must stand them up as peers up front** — on this harness a spawned agent cannot spawn
    (verified; see Step 0's primitives block), so the worker *messages an existing teammate* and could not
    create one itself even if it tried. On one GPU the worker must **await** an investigator's timing run,
    never time concurrently with it (see scheduling below); peer replies are async, so "my turn ended with
    no reply" ≠ "no answer."
  - **Gates (results-referee, adversarial-auditor, anti-giving-up, fact-integrity, harness-integrity) —
    ephemeral, fresh per claim.** Independence is the asset; accumulated context is *contamination* — a gate
    that marinates in the worker's narrative develops rapport and stops re-examining what it once blessed. A
    gate verifies one claim and is discarded; the next invocation is a brand-new context. It reads the
    **ledger** for run history (data, not framing), so fresh ≠ amnesiac. **The worker controls none of:
    whether a gate runs, what it sees, or where the verdict goes** — the worker is the *subject* of the gate,
    so it must not spawn the gate, write its prompt, or receive its verdict. Cleanest enforcement: a worker
    *commit* deterministically fires the gate pipeline and verdicts land in the ledger **as-returned**, so
    neither the worker nor the hub can skip a gate or re-narrate a FAIL.
    - **Gates self-record; the hub is pinged only on FAIL / decision-required.** Each gate **appends its verdict
      object to the ledger directly** (any agent can write a file). On a **PASS**, that is the whole
      transaction: the worker reads the ledger and continues, and the verdict **never enters the hub's
      context.** Only a **FAIL** (or an explicit decision-required) escalates to the hub. This is load-bearing,
      not cosmetic: the full pipeline fires on every commit, and at "dozens/hundreds of commits" routing every
      PASS verdict (each with its receipt) into the hub is a dominant context sink. Self-record + FAIL-only
      escalation drops the hub's per-commit gate intake to ~zero — and because the cost is paid to the ledger,
      not the hub, it stays ~zero no matter how many gates fire, so each claim is gated as it arrives.
- **One GPU ⇒ timing is a contended resource the hub schedules.** Classify every agent by what it needs:
  - **GPU / timing-bound** — worker measurements, results-referee re-runs, harness-integrity, oracle runs,
    perf-investigator benches. The hub **serializes these into one timing queue; never two at once**
    (concurrent `do_bench` corrupts both arms).
  - **Analysis-only** — code-investigator, kernel-classifier, fact-integrity (reading code),
    adversarial-auditor (reading configs / Triton). **Run these concurrently**; they never touch the GPU.
  - Use any deterministic fan-out / workflow primitive for **sequencing the gate pipeline** (propose →
    referee → auditor as a fixed chain) and **parallel read-only analysis** — *not* for parallel timing.
- **Never stop.** "Stuck / out of ideas / converged / hit a wall / can't get a clean rule / looks like a
  ceiling" is **never an exit** — it is a prompt for the next move (a different workload property; an oracle
  run to read the answer key; a fresh worker context; a perf dig into *why* the seed loses). The only things
  surfaced to the human are **non-blocking** signals (a soft "looks converged" flag; passive ledger
  breadcrumbs). Assume the human is unavailable and that there is no token limit.
- **The hub runs on a decision table, not deliberation.** Answering questions and deliberating in the hub's
  own context is a primary way the orchestrator's window fills; the table keeps it mechanical. Match the
  event, take the action; reason in-context for nothing on this list:

  | Event | Mechanical hub action |
  |---|---|
  | Worker: claim ready to gate | fire the gate pipeline; verdicts self-record (above) |
  | Gate PASS | nothing — worker reads the ledger and continues |
  | Gate FAIL / decision-required | DM the worker the receipt + "address & resubmit" |
  | Worker: stuck / ceiling / noise / "no rule" / done | spawn the **anti-giving-up agent**; relay its experiment |
  | Worker: **technical** question ("why is A slower?", "where does X live?") | tell the worker to **DM the investigator** — the hub does **not** answer it |
  | Worker: **orchestration** question (sequencing, which split, when to freeze) | one-line answer |
  | Worker: fact change/defense | spawn the **fact-integrity agent** |
  | Hub context ≥ threshold | **respawn** (see § Respawn lifecycle) |
  | Parity / frozen / overtime reached | log the boundary, switch mode |
  | **None of the above** | spawn a one-shot **orchestration-advisor**, execute its recommendation |

  **The table is never an exit — every row ends in an action that *continues the run*, and "ask the human / wait
  / stop" is not on it and never gets added.** The table is a tool for moving fast on the common cases, **not**
  the set of situations in which the hub is allowed to keep going — so an event you don't see listed is **not**
  permission to halt or escalate to the human; it is precisely what the **"None of the above → orchestration-advisor"**
  row is for. There is **always** a next move (a different workload property, an oracle run, a fresh worker
  context, an advisor's recommendation), the human is unavailable by assumption, and there is no token limit.
  If you ever find yourself wanting to stop, ask the human, or wait for input, that wanting **is** the trigger
  for the last row: spawn the advisor and execute what it returns. The run does not pause for the hub's
  uncertainty.

  Routing technical questions to investigators is not just load-shedding — it is the **peer topology working as
  designed** (investigators exist precisely so the worker gets technical answers without a hub round-trip). The
  hub adds nothing to "why is A slower than B"; inserting itself there both bloats its context and adds a relay
  hop.
- **Comms are pull-based; SendMessage is for pushes only.** The harness **echoes each SendMessage body back
  inside the ack**, so every message the hub sends is stored *twice* in its context — the send, then the ack
  re-quoting it — and a chatty push-based loop burns hub context on traffic it learns nothing new from. So:
  - The worker keeps a **one-line status beacon** (a file: current shape, last action, alive-tick). The hub
    `tail`s it to confirm progress *without a round-trip* — this is the heartbeat that satisfies "never stop"
    cheaply. Routine progress the worker already writes to the notebook/ledger; the hub is **not** CC'd on it.
  - Routine acks ("proceed", "noted") go via **`TaskUpdate`** on the shared task list (no echo envelope), not
    SendMessage.
  - **SendMessage is reserved for genuine pushes** — a FAIL, a stop, a phase flip, an answer to an
    orchestration question — and kept **terse** (terseness pays double: the ack echoes whatever you send).
- **Acceptance is independently gated.** Nothing enters the champion unless an independent **results-referee**
  reproduces the delta (its own command, ≥3 launches, fixed seed, accuracy on, pinned GPU, median clearly
  beats noise) **and** the relevant adversarial gates pass. Reject — never silently accept — a cheated or
  unreproducible result; reject — never loosen — a correctness failure. Then keep going. This is what makes
  endless churning *safe*: at worst it spends time, it can't corrupt the result.
- **Structured verdicts, recorded as-returned; neutral briefing.** Each gate returns a fixed verdict object
  (PASS/FAIL + the receipt fields in the roster below), recorded to the ledger **as returned, before the hub
  decides** — the hub cannot launder a FAIL into an accept by re-narrating it. **Brief gates neutrally**:
  hand them the artifact and the claim *flatly* (exact command, the normalized `Config`, both latency
  distributions), not "the worker found a win — confirm it" (that primes confirmation). The hub itself is
  trusted only with *progress*, not with the verdict — which is why verdicts are recorded mechanically, not
  paraphrased.
- A non-regression backstop is in force across the **whole** curriculum: no shape's referee-confirmed result
  may regress beyond a small bound vs the champion while chasing a gain elsewhere.

### Agent roster

Keep the standard roster (each tagged by resource — `[timing]` = needs the GPU, hub-serialized into the one
timing queue; `[analysis]` = read-only, runs concurrently): **worker** `[timing]` (drives the hill-climb,
owns the fact/heuristic code and the lab notebook), **code-investigator** `[analysis]` ("where/how does X
work?"), **perf-investigator** `[timing]` ("why is A faster than B?" via benchmarking / `ncu` / IR /
codegen), **kernel-classifier** `[analysis]` (static T1/T2/out-of-scope per kernel), **adversarial-auditor**
`[analysis]` while reading configs / Triton, `[timing]` for its own re-runs (anti-cheating). Three of them
carry responsibilities specific to this run and are spelled out:

- **measurement-harness-verifier + results-referee** `[timing]` — the whole method hinges on `seed/oracle`
  and `seed`-vs-tc being *trustworthy numbers*, so these two own that trust. The harness-verifier owns the
  bare-seed measurement mechanism (`configs=[seed]` actually used the seed, no autotune ran, correctness
  passes) and the established `do_bench` harness. The results-referee independently **reproduces every
  accepted delta with its own command** and has veto power. **Every trust verdict ships a compact,
  re-runnable receipt** — exact command, pinned random seed(s), N launches, raw median + spread, the
  resolved/**normalized** `Config` (proving the seed was the one measured), accuracy pass/fail, and the
  accept/reject rule applied. A bare "verified" with no receipt is not acceptable — an unauditable verdict
  is how a confound sneaks back in one layer down.
- **ledger-keeper** — a **discipline, not a write-broker daemon.** Durable state is a *file*; any agent
  appends to it directly — do **not** serialize every state write behind one more agent. The ledger-keeper
  role exists to enforce exactly two things that genuinely need a guardian: (1) the **oracle cache keyed by
  kernel-source-hash** — `{winning_config, latency_distribution}` per `(kernel, shape, source-hash)`,
  **invalidated on any edit to a kernel's source or to codegen that changes its generated Triton** (a stale
  oracle silently moves the victory bar — see § Oracle discipline); and (2) the **train/val/test firewall**,
  as the **only reader of the TEST split, exactly once, at freeze.**

**Add two adversarial agents that mirror the auditor in the opposite direction** — the roster has
historically been asymmetric (many guards against over-claiming, none against giving up or against a hacky
foundation). Both are **ephemeral, fresh per claim** like the other gates. Invoke them at exactly the
moments those failures happen: the **anti-giving-up agent** `[timing]` (it runs a fresh oracle to read the
answer key) on **any** ceiling / noise / "no rule" / done claim; the **fact-integrity agent** `[analysis]`
on **any** change to or defense of the `ReductionFact` vocabulary.

---

### Adversarial-auditor prompt (anti-over-claim)

> You are the **adversarial auditor**. You are spawned **fresh for this single claim** with no prior context
> shared with the worker; read the ledger for run history, but extend no rapport — your separation from the
> producing context is the entire point. The worker just claimed a heuristic improvement. Your job is
> **not** to demand more performance — a seed that honestly matches the oracle is done. Your job is to determine
> whether the worker is **cheating, fooling itself, or fooling me**. Assume it's under pressure to show
> progress and will take the easy way out; find it. Prefer experiments over theory — re-run, diff configs,
> read generated Triton; use the perf-investigator. Hunt each cheat: **(1) Fabricated/noise** — independently
> re-run (≥3 launches, fixed seed, your own command); if the median delta doesn't clearly beat noise, it's
> noise. **(2) Measuring the wrong thing** — confirm the benchmarked kernel actually used the seed (via the
> normalized config / generated code), with identical baseline/precision/flags on both arms. **(3)
> Overfitting** — run on held-out val and on a kernel outside this iteration's focus; report the gap with
> per-shape numbers. **(4) Kernel-identity smuggling** — read the code; flag any branch that is "if kernel X"
> in disguise. **(5) Metric gaming** — tolerance not loosened, no shapes dropped, autotune budget not blown.
> **(6) Anything else** — the list isn't exhaustive. **Verdict:** PASS only if the gain is real, reproducible,
> measured correctly, and generalizes without kernel-identity branching. Else FAIL with the specific cheat +
> evidence. When uncertain, lean FAIL and name the experiment that would resolve it.

### Anti-giving-up agent prompt (anti-under-claim / missed-gain)

> You are the **anti-giving-up agent**. You are spawned **fresh for this single claim** with no prior context
> shared with the worker; read the ledger for run history, but extend no rapport — your separation from the
> producing context is the entire point. The worker has claimed that some work is **done**, that a shape is
> **at ceiling**, that a gap is **noise**, that a knob has **no clean rule**, or that something is a
> **kernel-source limit**. Your job is **not** to demand the impossible — if the seed genuinely matches the
> oracle on a shape, that shape is done, and you say so. Your job is to determine whether the worker is
> **giving up early, fooling itself that an achievable gain is unreachable, or settling for an impoverished
> fact vocabulary.** Assume it is under pressure to declare victory and will rationalize stopping; find it.
> Prefer experiments over theory — **run the oracle and read the answer key.** Hunt each give-up:
> 1. **Noise dismissal.** A gap called "noise" must be disproven with a **noise-robust** measurement —
>    re-measure with **M lifted** off the floor, or use the **`seed/oracle` ratio** (both arms timed
>    identically). "Sub-25µs" is not proof of noise.
> 2. **False ceiling.** A "kernel-source limit" claim is only valid if the **oracle also fails to beat
>    tc-default** on that shape (`oracle ≥ tc_default`) **and** the oracle run is verified real (not
>    truncated / OOM-aborted / mid-search). If `seed < oracle`, it is not a ceiling — it is a missed gain.
> 3. **"No clean rule."** Reject any knob left at default justified by "contradictory / no separating
>    property." That is always a statement about the current facts, never the world. Demand the finer,
>    provenance-traceable property — or a proof the distinguishing value is knowable only at runtime.
> 4. **Premature convergence.** "Done / converged / at ceiling" requires **per-shape `seed ≈ oracle` (within
>    ε) against a FRESH oracle**, not a good aggregate vs tc-default. Spot-check the shapes the worker is
>    quietest about.
> 5. **Unfalsified limitation.** Any "X is not available / not possible / not exposed" must be killed by a
>    code search before you accept it. A fluent caveat is a red flag, not a clearance.
> 6. **Anything else** — the list isn't exhaustive.
> **Verdict:** for each shape/claim in question, PASS only if `seed` is within ε of a fresh valid oracle (or
> the worker's own perf claim is in fact *under*-stated and the gain is larger). Else FAIL with the specific
> give-up + the experiment that would close it (which oracle to run, which M to lift to, which fact to add).
> When uncertain, lean FAIL and name that experiment. (A `seed ≈ oracle < tc_default` shape PASSES — the seed
> is done — but separately verify any "this is a source ceiling" claim with a real `oracle ≥ tc_default`
> measurement before it is recorded as such.)

### Fact-integrity agent prompt (substrate / provenance)

> You are the **fact-integrity agent**. You are spawned **fresh for this single claim** with no prior context
> shared with the worker; read the ledger for run history, but extend no rapport — your separation from the
> producing context is the entire point. The worker has added or changed a `ReductionFact` field, or has left
> a knob at default claiming no workload property distinguishes its settings. A fact that *happens to be
> correct on the current kernels* but is computed by a **hacky proxy** (an easily-available stand-in: a
> fake-tensor `shape[-1] == size_hint` value test, a syntactic `load`-op count) is a **latent bug the perf
> gate cannot see** — because the proxy and the true fact are observationally identical on the curriculum,
> every speedup measurement will agree with the proxy. Your job is to ensure every fact computes the **real
> property it is meant to capture**, not a stand-in, and is used by a branch. Hunt:
> 1. **Proxy vs real property.** For each fact, state the property it is *meant* to capture, then check
>    whether the code computes that or a stand-in. The fix order is: (a) **reuse existing provenance** —
>    name the analysis it should source from (the compiler already computes buffer-level read/write
>    provenance and loop-carried state for reduction rolling; prefer reusing it); (b) if the worker claims
>    no such provenance exists, **falsify that with a code search**; (c) **only if it is genuinely absent**,
>    the worker should **write a dedicated analysis** to compute the real property. Writing an analysis is
>    correct and expected — **leaving the fact hacky, or dropping the property, is the failure.** Do not
>    accept "it's not in provenance" as a reason to keep a proxy until you have verified it yourself.
> 2. **The divergence test (a proxy-catcher).** Posit the lazy proxy the fact might secretly be (a
>    shape-equality, an op-count) and construct a kernel where **that proxy and the *real* property
>    disagree** (a non-reduction last dim equal to the reduction extent; the same reduction with reads fused
>    vs not; an in-iteration scratch buffer vs a genuinely loop-carried accumulator; an outer reduction). On
>    that kernel, does the fact track the **real property** (✓ pass) or the **proxy** (✗ hacky — reject)?
>    Be precise about what *passes*: a fact that tracks the real property on every such kernel — i.e. its
>    computed value never deviates from the property we want — is **success**, not a reason to reject. And
>    if the proxy and the real property *cannot be made to disagree by any kernel*, the proxy is exact and
>    the fact is fine. The genuinely **unfalsifiable** case (reject) is different: it is a fact whose
>    intended property is too vaguely defined to even state what the output *should* be on a given kernel —
>    there is nothing to test against. Demand a precise definition before trusting it.
> 3. **Style-dependence.** Does the fact change when the source is refactored without changing the
>    computation? If yes, it tracks coding convention, not the workload — reject it.
> 4. **"No clean rule."** Reject any knob-left-at-default justified by "contradictory / no separating
>    property." Demand the finer traceable property (or a runtime-dependence proof).
> 5. **Consumer check.** Every fact must name the branch that uses it. A fact with no consumer is dead
>    weight — cut it.
> 6. **Anything else.**
> **Verdict:** PASS only if every fact computes its real property (reusing provenance, or via a written
> analysis where provenance is genuinely absent), **tracks that property — not the lazy proxy — on a kernel
> that separates the two**, is style-independent, and has a consumer. Else FAIL with the specific fact + the
> proxy-separating kernel on which it tracks the proxy (or the provenance source / analysis it should use
> instead). **Caution — do not overcorrect:** do **not** demand a
> general-purpose dataflow framework where a single specific fact suffices. The mandate is "compute the
> exact property the branch needs" (reusing provenance, or a small targeted analysis), not "build a maximal
> analysis." Over-engineering (a consumerless general framework) is its own failure, symmetric to the proxy
> hack.

---

## Plan

NOTE: for Step 0 and Step 1, please use a separate, one-time subagent to do this work, and report back to you when this is succesful.
I don't want you context window poluted with setup builds and benchmark sanity checks.

**Step 0 — Setup; prove your wiring; prove the measurement mechanism (mandatory; nothing downstream is
trusted until green).** Set up a dedicated git worktree + branch. **Discover** your environment (the Python
env with the dev deps, an idle GPU) — do not hardcode it. **Prove your worktree's code is what runs:** assert
`helion.__file__` resolves inside *your* worktree, make a throwaway edit, and confirm it changes the
generated Triton (`to_triton_code` / `HELION_PRINT_OUTPUT_CODE=1`). If the benchmark operators live in a
*separate* editable install (a meta-path finder that `PYTHONPATH` can't shadow), prove an operator edit takes
effect too. Then prove the **bare-seed mechanism** on one shape: build a seed `Config`, launch with
`configs=[seed]` (no search), and confirm (a) no autotune ran, (b) the kernel used the seed (inspect the
**normalized** `bound_kernel._config` — `normalize()` may mutate the seed), (c) correctness passes, (d)
latency is stable across N runs.

**Also discover your orchestration primitives** (don't assume a topology — "adapt to your harness" is exactly
the instruction that, left vague, collapses to the worst option). Probe for: (a) a **persistent-agent /
message-continuation** primitive (can you continue one agent in place, keeping its context, or only spawn
fresh?); (b) a **team / shared-task** primitive (named peers that message each other directly + shared
state); (c) a **deterministic fan-out / workflow** primitive. Then realize the **persistent worker** at the
**highest available rung of this ladder, and record which rung in the ledger:**
1. **Continue-in-place** — the worker is one agent continued across iterations with its context intact
   *(best; e.g. a named, backgrounded agent reached over a message channel)*.
2. **Fresh-worker-per-iteration, re-briefed from the notebook** — lossy of tacit intuition and pays a
   re-brief tax every iteration, but still keeps hub≠worker *(acceptable degradation)*.
3. **Hub drives the hill-climb directly** — **last resort only**, after 1–2 are proven unavailable; it
   merges the producing and gating contexts and bloats the hub, so log it as a known degradation.

Topology is a **separate, orthogonal** choice (see § Orchestration): independently of the rung above, if a
team/peer primitive exists, give helpers **peer** topology so the worker reaches investigators directly
(no hub relay) — gates stay hub-owned regardless. *On a harness offering both continue-in-place and teams,
the target is rung 1 realized as a team: a persistent worker peer + standing investigator peers + fresh
per-claim gates.*

> **VERIFIED on the Claude Code Agent/Team harness (probed live 2026-06-03 — if you are on a *different*
> harness, re-run the discovery above; if you are on this one, these are facts, skip the probing):**
> - **Continue-in-place ✅ (rung 1 is available).** `SendMessage` to a background agent that has already
>   "completed" *resumes the SAME context from its transcript* ("had no active task; resumed from transcript")
>   — verified by having it recall a token set before the boundary. **This is the exact primitive run 1/2
>   lacked; the hub-drives-directly fallback is moot — do NOT use it.** Realize the persistent worker as a
>   named, backgrounded agent continued via `SendMessage`.
> - **Nesting ❌ (impossible — even inside a team).** A spawned agent has **no `Agent` tool** (confirmed for
>   both a plain sub-agent and a team member). A teammate *does* get `SendMessage` + `Task*` +
>   `Bash/Read/Edit/Write`, but it **cannot spawn**. ⇒ **The worker cannot create its own investigators**, so
>   the peer model is the *only* way to give it direct helper access: **the hub stands up the investigators as
>   peers up front**, and the worker *messages* them. (This also means only the hub can spawn gates — which
>   enforces "the worker can't spawn its own gate" *for free*.)
> - **Direct peer messaging ✅ + shared task list ✅.** A teammate `SendMessage`s another teammate directly
>   (not routed through the lead), and all members read/write one shared task list (`TaskCreate/Update/List`).
>   ⇒ realize the whole thing as **one team**: hub = lead (only spawner, owns acceptance + gate spawning +
>   `Workflow`); worker = persistent peer; investigators = standing peers the worker DMs; gates = hub spawns
>   fresh per claim.
> - **Peer replies are ASYNC.** A reply can arrive *after* the requester's turn ends (observed: a peer
>   reported "no reply yet," the reply landed the next turn). An agent awaiting a peer must not read
>   "turn ended, nothing yet" as "no answer." (Harmless here: the one-GPU rule already forces the worker to
>   *block* on an investigator's timing result — see § Orchestration — so it waits anyway.)
> - **Spawned agents do NOT inherit the lead's model — pass it explicitly.** With **no** `model` arg, teammates
>   came up on **opus-4-7** while the lead was opus-4-8[1m] — a silent downgrade. Passing `model:"opus"`
>   (the tier enum is `opus`/`sonnet`/`haiku`; you cannot name a version) resolved to the full
>   **`us.anthropic.claude-opus-4-8[1m]`**, 1M context included. **Always pass `model:"opus"` on every spawn**
>   (worker, investigators, gates) so they match the orchestrator; omitting it is a downgrade, not inheritance.
> - **No per-spawn effort/thinking knob exists.** The `Agent` tool has no `effort`/thinking-budget param
>   (`mode` is *permission* mode, unrelated). A spawned agent runs at its model's default effort. ⇒ you cannot
>   dial a gate to "max effort" at spawn — **encode the required rigor in the prompt instead** (the gate
>   prompts already do: re-run ≥3×, read the IR, construct a separating kernel). Max-effort for the whole run
>   is a *session/harness* setting, not an orchestrator lever — a human-level knob, flag it if it matters.
> - **`Workflow` is hub-only** (not in any teammate's toolset) — deterministic fan-outs / gate-pipeline
>   sequencing are the hub's job, as intended.

**Step 1 — Benchmark sanity check.** On a medium compute-bound shape, compare Helion-default, Helion-quick,
Helion-max, tc-default, tc-max; confirm the rough ordering (Helion-default ≲ tc-default ≲ tc-max ≈
Helion-max). HALT only if something is *really* wrong; otherwise log surprises and continue. Certify the
harness isn't biased (an independent hand-rolled cross-check that reconciles with the established harness)
before trusting any hill-climb number.

**The hill-climb runs in THREE phases. The split is deliberate and cost-driven:** the oracle is the
expensive resource (a full autotune can take minutes/shape; on one GPU an eager oracle pass over the whole
curriculum is hours before the first edit), so we do not pay for it until the cheap floor work is done and
has told us where the oracle budget is best spent.

### Phase 1 — Floor sweep (cheap; NO autotune)
Drive every `train` shape to the **floor** (`seed ≥ tc_default − ε`). Pure `configs=[seed]` + tc-default
`do_bench` — no oracle runs at all. This is fast, it is where the catastrophic losses live, and the result
*also triages Phase 2*: a shape that limps against tc-default is where the answer key will pay off most.
- **Read the generated Triton — it is your answer key here.** Unlike Phase 2, the tc-default baseline gives
  you NO Helion `Config` to field-diff against. So the way to see *what you are climbing toward* is to read
  the **generated Triton of both sides** — your seed's (`to_triton_code(seed)` / `HELION_PRINT_OUTPUT_CODE=1`)
  and torch.compile's (e.g. `TORCH_LOGS=output_code`). Diff the *strategy*: is tc persistent where your seed
  loops? Splitting the row across programs where your seed runs one program per row? Fusing loads your seed
  re-streams? Using a different warp/tile shape? That strategy diff is the workload-property hypothesis for
  the iteration — the same role the oracle's config field-diff plays in Phase 2.
- Per iteration: read both sides' Triton + perf-dig → workload hypothesis → change fact/heuristic
  (correctness-gate first; `configs=[seed]` skips the autotuner's check) → matched-lever A/B vs the best
  *simple* alternative (never the catastrophic default) → check `seed` vs tc-default → gate (auditor on a
  win; fact-integrity on any fact change; results-referee reproduces) → commit + update ledger/notebook.
- **"Floor passed" is NOT a victory state — it is the entry condition to Phase 2.** Do not declare any shape
  "done" here; the anti-giving-up agent does not yet engage (nothing is being declared finished). You will
  revisit every shape in Phase 2.

### Phase 2 — Oracle ascent (expensive; the real bar)
Now build the oracle cache and close `seed → oracle` on every shape.
- **2a — Build / refresh the oracle cache (cheap-first, targeted).** For each `(kernel, N-band)`
  representative shape, run the autotuner once and cache `{winning_config, latency_distribution}` keyed by
  source-hash (quick for iterating, full to confirm). Spend first on the shapes Phase 1 flagged as
  furthest from the floor. Keep it fresh (§ Oracle discipline) — invalidate on any source/codegen change.
- **2b — Per iteration:** pick the shape(s) furthest from the oracle → **field-diff the seed against the
  oracle's winning config**; the differing fields are the worklist → form a **workload-property hypothesis**
  per field ("oracle picks eviction='first' on the streamed reduction input → key on a per-load *role* fact,
  not a load count") → if it needs a fact that is missing or hacky, **fix the fact** (reuse provenance, or
  write the analysis — never a proxy; fact-integrity gate) → change the heuristic (iterate, don't rewrite
  unless the structure is wrong) → **correctness-gate** → **matched-lever A/B**: vary only the candidate
  field, hold all else equal; to split "seedable" from "oracle-only," bench the oracle's block sizes at
  *default* codegen knobs then add only the candidate knob; **re-bench the oracle's FULL VERBATIM config as
  the baseline — never isolate one of its levers and re-pair it** (that fabricates an unmeasured config).
  **The seedable ladder (peak vs plateau) — run this whenever an oracle is a complicated bundle.** A *clear*
  oracle config is not always a *reachable-by-one-lever* config, so decompose it as a LADDER: oracle
  block_sizes alone → + oracle `num_warps` → + oracle `num_stages` (all three SEEDABLE) vs the **full verbatim
  oracle**, and read one of three verdicts. **(a) a seedable rung already matches the full oracle (within ε)**
  ⇒ a *plateau* — seed that rung, the gap is closeable now. **(b) only the full bundle matches** ⇒ a *peak*
  whose residual lives in the per-range codegen knobs (`range_unroll_factors` / `range_num_stages` /
  `range_multi_buffers`) — but these **ARE seedable** (the same list-valued, block-id-keyed knob class as
  `load_eviction_policies`, which the heuristic already emits), so a peak is *harder-but-possible*, **NOT** an
  automatic Product-B punt; build the per-range seed (verifying the round-trip preserves list knobs) before
  declaring any part oracle-only. **(c) within noise** ⇒ not a gap. (A *quick* oracle can hand a FALSE target;
  full-confirm before climbing a quick gap — see § Oracle discipline.)
- **2c — Evaluate + persist.** Compute `seed/oracle` (victory) AND re-confirm `seed ≥ tc_default − ε`
  (floor). **Persist the raw A/B numbers** (both arms' distributions, N, median, spread) for **every**
  decision, accepted or rejected, keyed by `(kernel, field, shape)`.
- **2d — Gate.** Accepted win → adversarial-auditor. Any "ceiling / noise / no-rule / done" claim →
  anti-giving-up agent (gated on `seed ≈ fresh oracle`). Any fact change/defense → fact-integrity agent.
  Any suspicious result → harness-integrity re-cert. Results-referee reproduces every accepted delta.
- **2e — No-regression backstop (load-bearing in this phase).** Chasing the oracle on one shape must not
  push another shape back **below the floor**. A designated checker (the results-referee on every accepted
  change, with the adversarial-auditor on the regression backstop) must re-confirm that **no shape has
  regressed below `tc_default − ε`** as a result of a Phase-2 edit — a Phase-2 oracle gain that silently
  re-opens a Phase-1 floor loss is rejected, not accepted. The floor, once reached, is held for the whole
  curriculum.
- **2f — Commit + update** ledger/notebook (decision + empirical *why* + every tried-and-rejected idea +
  open hypotheses + champion, so a fresh context can continue losslessly).
- **Phase 2 is done only when every measurable shape meets the victory bar (`seed ≈ oracle`, within ε)
  against a FRESH oracle** — not when an aggregate vs tc-default looks good.

### Phase 3 — Product B, freeze, TEST
- **Product B (on the frozen seed).** Measure the convergence curve seeded vs unseeded (cold cache,
  `HELION_FORCE_AUTOTUNE`, per-generation log), best-perf vs **generation index** AND vs **wall-clock**
  *separately*. Headline = **budget reduction**: does seeded-quick reach the unseeded-full optimum? (the
  budget gap is the savings). Or even just looking at the generation-by-generation results, does the seeded
  config converge to the optimum quicker than the unseeded config, both in the quick autotuner and the
  full autotuner?
- **Not part of Phase 3: "beat max-effort autotune."** Product B *here* is **budget-reduction only** (the
  seed gets a search to a good config in less wall-clock). Do **not** let beating an unbounded max-effort
  search bleed into Phase 3 and delay the freeze — it is compute-heavy, and within Phase 3 the never-stop
  rule must not be read as license to pursue it. If the budget-reduction result is solid, Phase 3 is done.
  **(Beating max-effort is not abandoned — it is deferred to Phase 4 *overtime*, entered only AFTER the
  deliverable is frozen + TEST-validated below; see the hub-kickoff prompt's Phase-4 section. The ordering
  is the point: bank the validated deliverable first, THEN chase the open-ended beat.)**
- **Freeze, then read TEST once** (ledger-keeper). Re-validate the champion against a **fresh** oracle.
  Report the train↔test gap (overfit signal) and the transfer-kernel generality number, separately. A large
  gap is reported, not hidden. **This freeze + TEST read is the `=== DELIVERABLE FROZEN + VALIDATED ===`
  milestone; only after it may Phase-4 beat-oracle overtime begin.**

---

## Technical appendix (verify symbols; line numbers drift — grep)

**Where things live.** Heuristics: `helion/_compiler/autotuner_heuristics/` (`triton.py` = the reduction
heuristic + the GEMM template; `__init__.py` = `HEURISTICS_BY_BACKEND` + `compiler_seed_configs`;
`registry.py` = `AutotunerHeuristic` base). Facts + search engine: `helion/autotuner/config_spec.py`
(`ReductionFact`, `ReductionLoopSpec`, `block_sizes`, `default_config`, `normalize`). Fact population:
`helion/_compiler/device_ir.py` (`register_rollable_reductions`, `register_user_tiled_reductions`,
`_count_reduction_workload`).

**Seed flow.** `BoundKernel.__init__` sets `config_spec.compiler_seed_configs`; during a real search the
autotuner injects those into the initial population. Each heuristic emits one config via `get_seed_config`;
to inject **several** structurally-distinct seeds (a portfolio), the base class exposes a
`get_seed_configs() -> list[Config] | None` hook used by `compiler_seed_configs` when present.

**Running a bare seed (the Product-A / Step-0 mechanism).** `helion.kernel(fn, config=seed)` (or
`configs=[seed]`) hits the `len(configs)==1` short-circuit → returns the seed, never autotunes. Caveats:
(1) it **bypasses the autotuner's accuracy check** → run your own; (2) **`normalize()` may mutate the seed**
(e.g. forces persistent when a value ≥ size_hint, caps by size hint) — benchmark/inspect the **normalized**
`bound_kernel._config`, not the raw seed.

**Silent seed-drop (make it audible).** In a real search, a structurally-invalid seed can be caught and
**dropped** (the search proceeds as if unseeded), logged only at INFO. Don't rely on it: validate the seed
eagerly (`config_spec.normalize(seed)` + `to_triton_code(seed)`) and treat any exception as a hard failure,
or run `configs=[seed]` so a bad seed raises instead of being swallowed. List-valued knobs
(`load_eviction_policies`, `indexing`) are **not length-validated by `normalize`** — build them at exactly
the live spec's length with only valid choices, and verify they survive the autotuner flatten/unflatten
round-trip (so Product B's seeded arm actually carries them).

**Persistent vs looped (Triton).** Encoded by one integer per rdim (`reduction_loops` for T1; the reduction
axis's `block_sizes` entry for T2): `value ≥ size_hint` ⇒ persistent (whole row in one pass), else looped
with that chunk. On Triton there is no finite hardware reduction-thread cap forcing a loop; the only hard
ceiling is the backend's per-tile element cap (`max_tensor_numel`, 2²⁰ elems) above which a whole-row
`tl.arange` can't compile. `num_warps` and `num_stages` are global scalar knobs.

**Static T1/T2 classification (no GPU run; kernel must be bound first).** **T1** = the rdim's `block_id` is
in `config_spec.reduction_loops` (`None` = persistent, int = rolled). **T2** = the reduction axis is a user
`hl.tile` → a normal `block_sizes` entry; knob = that `block_sizes` index. **Out-of-scope** = an rdim exists
but its `block_id` is not in `reduction_loops` (roller bailed) → no per-axis knob; skip. Exclude GEMM via
non-empty `matmul_facts`.

**Provenance for facts — reuse what the compiler already computes.** The reduction roller already resolves
each load/store/atomic's tensor back to **host-buffer names** and distinguishes them from device-internal
temporaries (grep for the reduction-loop read/write inference, e.g. `_reduction_fx_inter_loop_rw_names`), and
**loop-carried state is explicit** as a rolled subgraph's carried inputs/outputs. Source facts (distinct
operands, re-reads, atomics-vs-stores, loop-carried accumulators, the dtype of the *resident* reduction
input) from **that provenance** — not from fake-tensor `shape[-1] == size_hint` value tests. (atomics are the
signature of a split-K / cross-CTA-combine structure — a distinct future regime; keep them a *distinct*
signal, don't merge them into a generic store count.)

**Autotune budget & convergence traces (Product B).** Quick vs full effort profiles differ in
initial_population / copies / max_generations; `HELION_AUTOTUNE_MAX_GENERATIONS=N` shrinks the budget;
`HELION_FORCE_AUTOTUNE=1` forces a real search (a cache hit writes no trace); `HELION_AUTOTUNE_LOG=<path>`
writes a per-config CSV (`timestamp_s, generation, status, perf_ms, config`). Parsing traps: two rows per
config (a `started` row with empty `perf_ms`, then `ok`/`error`); missing fields are **empty strings**, not
NaN; **only analyze COMPLETE runs** — a mid-run cumulative-min is not the converged best (a classic
false-positive source). The autotuner's internal `perf_ms` is **not** comparable to a `do_bench` median —
**fair-re-bench every winner** with `do_bench` before comparing. For a fair seeded-vs-unseeded comparison,
pin the random seed, hold the budget knobs identical, and vary only the injected seed config.
