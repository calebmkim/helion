# Reduction Autotuner Heuristics — Work Orchestration

> Authoritative docs for this work: **this file** + **`list_of_kernels.md`** (benchmark kernels +
> in-sample/out-of-sample shapes). Everything in `old/` is superseded — do **not** open, cite, or
> follow it (`old/reduction_heuristics_plan.md` in particular is a disowned bad plan; the overview
> and recipe docs were folded into the Technical Appendix below). All file:line references anywhere
> are **approximate** — grep the symbol, don't trust the number.

## Goal

Write **general** compile-time autotuner **seed heuristics** for **reduction** kernels on
**H100 / sm_90, fp32, Triton backend**. A heuristic reads pre-digested facts about a kernel and emits
**one seed `Config`**. Two products (see "The two products"):

- **Product A — skip autotune:** the seed *is* the config. Bar: beat `torch.compile` default; approach
  Helion / `torch.compile` max-autotune. Losing a little to tc-default is acceptable.
- **Product B — seed the autotuner:** the seed makes quick-autotune reach a good config in **less wall
  clock**. Time is the priority here, as long as perf doesn't go wildly wrong.

The work is hill-climbing: write heuristics, measure, improve, repeat — with periodic adversarial and
autotune-time checks. Make commits early and often (dozens/hundreds is fine).

---

## Agents

> **Execution model (this harness does NOT allow nested sub-agents — verified empirically).** A
> sub-agent cannot spawn its own sub-agent. So the **main session is the manager and the only spawner —
> the "hub."** It holds one **persistent worker** (continued in place via follow-up messages, not
> respawned each round) and spawns every helper agent itself, relaying results back. Wherever this doc
> says an agent "spawns", "uses", or "delegates to" another agent (the worker spawning investigators, the
> auditor using the perf-investigator, harness-integrity delegating to perf-investigator, etc.), read it
> as: that agent **requests** the helper from the hub, which spawns it and relays the result. Spawn a
> *fresh* worker only when the worker is stuck or its context is filling up — hand over the lab notebook
> so nothing is lost.

### Manager — the entry point (= the main session / hub)
You are the outermost agent and the **only one that can spawn sub-agents** (see Execution model), so you
are the **hub**: you hold one persistent worker, spawn every helper yourself on the worker's request, and
relay results back. You are a *separate context from the worker on purpose* — that is the source of your
supervisory power: an agent left to manage itself rationalizes quitting, because the same tired context
that wants to stop is the one deciding whether to stop. **Launch the worker** with the rest of this file
(Worker section onward) as its brief, then supervise it.

**Never let the worker stop.** The worker does not get to quit — "stuck / out of ideas / converged / hit
a wall" is never an exit, only a prompt for your next move. When the worker flags it's stuck, in order:
(1) tell it to attack from a genuinely different angle (a new workload property; a perf-investigator dig
into *why* the current config loses to the oracle); (2) **spawn a fresh worker** seeded with the **lab
notebook** (champion config + the reasoning/why + the tried-and-rejected list) but a clean context —
do this when the worker is **stuck** or its **context exceeds ~50%** (err early — switch at a clean
iteration boundary with the notebook current; treat the worker's context as disposable and the notebook as
the source of truth, so early handoff loses nothing and you never drift toward auto-compaction); otherwise
keep the *same* persistent worker going (don't respawn every iteration — that throws away its intuition).
A fresh context is the single most effective way to get unstuck, and only you (the hub) can do it. Then
keep going. If it asks you for permission to run a command, you *always* let it run. Furthermore, you should
*never* ask me to run a command. You will be in yolo mode, that's for a reason. You should never ask me for
permission to run a command. Just run it! I will not be there for you.

This is only safe because **acceptance is independently gated**: nothing enters the champion unless the
results-referee reproduces it and the regression/correctness gates pass. So endless churning can never
corrupt the result — at worst it spends time. Keep the worker churning.

**Soft convergence flag (a signal, not a stop).** When K consecutive iterations (K≥5), each a genuinely
**different** idea, are all referee-rejected as noise *and* the adversarial-auditor agrees no untried
angle remains, **emit** "looks converged — you may want to pull the plug" to the human — then **keep
going anyway** (another angle / a fresh worker). You inform; you never stop on your own.

**Never give up — not even on broken infrastructure.** You never stop or escalate-as-an-exit for *any*
reason; everything you emit is a non-blocking signal (the convergence flag above; passive ledger
breadcrumbs), never a stop. Two things you must never let slide — but you handle both by *grinding on*,
never by stopping:
- **Broken infra** (environment won't go green, Step 0 won't pass, the harness can't be made fair): keep
  trying to fix it — every approach, fresh workers, for as long as it takes. Spend 10 hours if that's
  what it takes; it almost certainly won't get there, but try anyway. Until it's green, don't *trust*
  results — but don't stop either.
- **A cheated / false result** (silently loosened tolerance, dropped regressing shapes, swapped baseline,
  kernel-identity smuggling, a biased harness): never *accept* it — reject it and send the worker back to
  do it honestly, then keep going. The acceptance gate enforces this automatically; your job is to keep
  the gate honest, not to halt. (Routine rejections are likewise not stops: a candidate that fails
  correctness, or a claimed win the referee can't reproduce, is simply rejected — keep churning.)

For long grinds (e.g. infra still red after hours), drop a **non-blocking** status note in the ledger so
the human sees it on check-in — a breadcrumb, not a stop and not a request for input.

Answer the worker's questions yourself with best judgement (the human is not available).

### Worker
The driver of the hill-climb. **Persistent** — continued in place across iterations (not respawned each
round), so it keeps its accumulated intuition. Owns the `ReductionFact` design, the heuristic code, and
the iteration loop. It **cannot spawn** the sub-agents below — it *requests* them from the hub and gets
results relayed back. Continuously maintains the **lab notebook** (every heuristic decision + its
empirical *why*; every tried-and-rejected idea + why it failed; open hypotheses; the champion) so a fresh
worker can take over losslessly.

### Sub-agents
- **code-investigator** — answers "where/how does X work?" by scouring `helion/` and `pytorch/`. Returns
  a concrete, grounded answer (artifact only; intermediate search is disposable).
- **perf-investigator** — answers "why is config A faster/slower than B?" with benchmarking, `ncu`,
  IR/codegen inspection, etc. Returns the root-cause. **Supplies the empirical "why" behind heuristic
  choices.** Knows this is a shared machine — other users' jobs may run concurrently; rule out
  co-tenants before trusting a delta.
- **measurement-harness-verifier** *(owns Step 0)* — the single source of truth for how a bare seed is
  measured. Proves no search ran, that the seed was actually used (not silently dropped), and runs
  correctness. Emits the fixed evidence block every other agent consumes.
- **results-referee** *(independent, veto power)* — re-runs any worker-claimed delta with its own command
  (≥3 launches, fixed seed, accuracy on, pinned GPU). Admits a result only if the median delta clearly
  exceeds run-to-run noise **and** correctness passes. Do not trust the worker's logged numbers —
  regenerate them.
- **ledger-keeper** *(state owner)* — durable JSON: per-kernel best heuristic version, in-sample +
  out-of-sample scores, the **oracle cache** (see below), iteration history, and the
  in-sample/out-of-sample **firewall**. The only agent that reads the held-out TEST shapes (once, at the
  end). Also persists raw benchmark/autotune logs + Product-B convergence traces under `logs/` (kept for
  audit/bookkeeping; not read inline — trust-receipts point at them). **Holds the lab notebook** (the
  worker's reasoning trace: decisions + why, tried-and-rejected + why) — the durable knowledge a fresh
  worker reads to continue losslessly.
- **kernel-classifier** *(one-shot per kernel, static)* — inspects `config_spec` (no GPU run) and assigns
  each kernel a track + reason: **T1** (compiler-managed rollable rdim → seed via `reduction_loops`),
  **T2** (user-tiled reduction → seed via `block_sizes[v]`), or **out-of-scope** (roller bailed →
  persistent-only, no per-axis knob; skip). See Technical Appendix for the exact signals.
- **adversarial-auditor** — **anti-cheating**, not a perf hardass (prompt below).
- **harness-integrity agent** — guards against *systematic* benchmark **bias** — the one thing the
  referee's re-runs can't catch (re-running a biased harness just confirms the bias; N repeats fight
  variance, not bias). Owns the footgun checklist, the Step-1 sanity check, and an independent
  **cross-check**: extract the generated Triton + shipped configs from **both** Helion and
  `torch.compile`, hand-write a standalone harness that times each with identical overhead, and reconcile
  against TritonBench. Agreement → trust the harness; disagreement → *localize* the discrepancy (then
  decide which path matches reality via ncu / kernel-counts — the hand-rolled number is **not** an
  automatic oracle). Caveats it must respect: replicate the **full set of launched kernels** per side
  (not just the headline one — host-side reductions / multi-kernel splits bite here); identical
  inputs / layout / dtype / strides / grid; verify the hand harness matches the real path via
  ncu/kernel-count. **Kernel-only timing is the diagnostic; end-to-end TritonBench stays the optimization
  objective** (you ship a Helion config — users feel the wrapper too). Runs **periodically** (calibrate)
  + **on-demand** (suspicious result), **not per-iteration**; delegates the extract+hand-bench to the
  perf-investigator. Distinct from the results-referee (variance, per-claim) and the Step-0
  measurement-harness-verifier (the seed-run mechanism). Bonus: its extracted both-sides Triton is also
  the highest-fidelity **answer-key diff** against the oracle.

### Context-saving convention (trust levels)
- **Non-trust agents** (code-investigator, perf-investigator, env setup): return the **artifact only** —
  swallow the noisy build/benchmark/search logs. This is the legitimate context win.
- **Trust-critical agents** (results-referee, adversarial-auditor, measurement-harness-verifier): return
  the verdict **plus a compact, re-runnable receipt** — exact command, pinned seed(s), N runs, raw
  median + spread, the resolved/normalized `Config`, whether the seed was actually used, accuracy
  pass/fail, and the accept/reject rule applied. **Never swallow the trace of an agent whose output is a
  trust verdict** — an unauditable "verified" is how cheating sneaks back in one layer down.

---

## Operating rules

- **Generalize on workload, never on kernel identity.** Branching on reduction **shape/workload
  properties is encouraged** (`if rnumel < 50: ...`). Branching on **kernel identity is banned**
  (`if kernel == rms_norm`). When two kernels need different treatment, find the *workload property* that
  distinguishes them, expose it as a `ReductionFact` field, and branch on that. A magic threshold that
  happens to fence off exactly one kernel's shapes is kernel-identity smuggling — also banned.
- **Justify every decision.** Each heuristic branch carries a comment with the *why*, grounded in
  empirical evidence (perf-investigator supplies it). "It made rms_norm faster" is not a why; "contiguous
  inner reductions with rnumel ≤ X fit in registers → persistent wins" is.
- **Correctness is the first filter** every iteration — reject a config before comparing latency if it
  fails accuracy against the eager/reference baseline at the operator's tolerance. (`configs=[seed]`
  bypasses the autotuner's own accuracy check, so the worker must run its own.) fp32 reduction-order
  drift can legitimately exceed tight tolerances — any tolerance loosening must be logged, never silent.
- **Commits:** commit locally on every substantial heuristic change / perf-improving iteration. Do
  **not** `git push` (the human handles that). Work on a dedicated worktree + branch (not main).
- **Precision is fixed at fp32** for the whole hill-climb; assert it on every benchmark call (override
  per-operator defaults — e.g. softmax defaults to fp16). The heuristic must read **`dtype`/itemsize as a
  `ReductionFact` field**, not hardcode fp32, so it generalizes to bf16/fp16 later.
- **Shared machine:** pin `CUDA_VISIBLE_DEVICES` to one idle device (verify with `nvidia-smi`), pin
  `HELION_AUTOTUNE_RANDOM_SEED` to a small fixed set, take median-of-N with spread, and re-run any
  suspicious delta before believing it.

---

## The two products & the objective

**Product A — skip autotune (what we hill-climb every iteration; cheap):**
- Run the **bare seed** (`configs=[seed]`, no search — see Appendix), correctness-gate it, then measure.
- Maximize `O = geomean over active kernels of G_k`, where `G_k = geomean over that kernel's in-sample
  shapes of (tc_default_latency / seed_latency)`. Geomean across kernels — the right way to combine
  speedup *ratios* ("2× on A, ½× on B" → 1.0, not a false 1.25) — and it naturally resists cratering any
  single kernel.
- **Accept iff `O` improves.** Hard gates (a candidate that raises `O` is still rejected if it trips
  one): correctness; the seed was actually used (not silently dropped); and as a backstop against robbing
  one kernel to pay another, **no active kernel's referee-confirmed `G_k` regresses > 10%** vs the
  champion. A small local regression is fine if the aggregate improves (3% proved too strict).
- **Slow-drift watch (auditor, soft — not a hard gate):** the adversarial-auditor flags any kernel that
  has drifted well below its best-ever (e.g. > 15%) across many accepts while still passing the per-step
  backstop.

**Product B — seed the autotuner (time-first; run on the every-5 cadence):**
Success is two slices of one **convergence curve** (best-config-perf vs budget): a good seed shifts it
**up and to the left**. Measure the curve **once** and read both slices off it — don't run separate
experiments.
- **Measure:** run quick-autotune **seeded vs unseeded, N times each** (fixed seed-set; it's stochastic);
  log **best-perf-so-far** against both **generation index** (the fair search-quality axis) and
  **cumulative wall-clock** (the real time axis). Report the median trace + spread.
- **Slice 1 — same-budget perf** (the common case — most users keep the default budget): seeded vs
  unseeded best-perf at a fixed budget. Sharpest at a **small** budget (gen 1–2), where the seed earns
  its keep; at the full budget it often washes out (both converge), so use full-budget perf as a
  **no-regression guardrail** (seeded ≥ unseeded − ε), not the headline.
- **Slice 2 — time-to-target** (*headline*, per time-first): wall-clock for seeded to first reach X% of
  unseeded-full-budget perf (or of the oracle), vs unseeded. The time win comes from a good seed reaching
  target sooner / letting you shrink the budget.
- **A is the lever for B.** A better Product-A seed lifts the whole curve; the limit (budget → 0) is
  Product A itself. Optimize A; harvest B. (Exact knobs to shrink the budget and read the per-generation
  trace: see "Autotune budget & convergence traces" in the Appendix.)
- **Persist** the raw convergence traces + autotune logs to `logs/` (referenced from the ledger); agents
  needn't read them inline, but keep them for audit/bookkeeping.

**Oracle cache (avoid re-autotuning every iteration):** when a kernel first enters the curriculum, run
Helion-max and tc-max **once** per in-sample shape (a few repeats, keep best); store
`{winning_config, latency_distribution}` in the ledger. Each iteration compares the seed against the
**cached** oracle latency — no autotuner runs. The cached `winning_config` is the **answer key**:
field-diff seed vs oracle (`reduction_loops` persistent-vs-looped, R/x blocks, `num_warps`,
`num_stages`), then A/B the differing fields. Re-run an oracle only if the example/IR or shapes change;
re-validate the final champion against a fresh oracle.

---

## Scope & curriculum

- **Forward kernels only, for now.** All 9 forward reductions are **inner** (reduce the contiguous last
  dim) — so we ignore inner/outer entirely and tune inner reductions only. **Defer backward** kernels:
  the only outer-shaped reductions live in `rms_norm_bwd`/`layer_norm_bwd` weight/bias gradients (sum
  over M), a separate workstream (Band D) we'll pick up later.
- **Start on rms_norm**, then widen to the other forward kernels. Re-measure all already-active kernels
  each iteration (the > 10% per-kernel regression backstop is cross-kernel).
- **In-sample / out-of-sample discipline** (shapes per kernel are in `list_of_kernels.md`):
  hill-climb on **in-sample only**. Split out-of-sample into **VALIDATION** (read at adversarial
  checkpoints, never drives edits) and **TEST** (looked at exactly once, at the very end, by the
  ledger-keeper). Headline generalization metric = in-sample vs TEST geomean gap; a large gap = overfit,
  report it rather than declare success.
- **Recipe bands** (keyed on workload properties → `ReductionFact` fields, NOT kernel names):
  - **Band A — scalar-accumulator inner:** sum, rms_norm-fwd, layer_norm-fwd, softmax, long_sum. One
    shared `(R_BLOCK, num_warps, num_stages)` recipe; only the **knob name** differs by track
    (`reduction_loops=[R_BLOCK]` for T1 rolled, `block_sizes[v]=R_BLOCK` for T2 manual). `num_warps`/
    `num_stages` are global scalar knobs shared across both.
  - **Band B — heavy-epilogue inner:** kl_div, jsd. Same loop skeleton but high arithmetic intensity +
    extra live accumulators / output (jsd has 2 accumulators + a `dX` store) → **smaller R_BLOCK**,
    different warps/stages. Distinguish via fact fields (num_load, num_reduction_ops, num_accumulators).
  - **Band C — structured combine:** welford. Not rollable; 3-scalar combine (count/mean/M2) + a second
    pass → its own treatment; don't inherit the scalar-sum recipe.

---

## Plan

**Step 0 — Set up your worktree, prove your edits run, then prove the measurement harness (mandatory; nothing downstream is trusted until green).**
*First:* create your dedicated git worktree + branch off the helion checkout and **prove your worktree's
code is what actually runs** — this is the easy way to silently test nothing (see "Machine-level
reference" in the Appendix for the import gotcha). Assert `helion.__file__` points inside *your* worktree,
then make a throwaway edit and confirm it changes `to_triton_code`/the generated Triton; do the same for a
*tritonbench* operator edit (it's a separate editable that `PYTHONPATH` may not shadow). Do not assume the
smoke-tested setup carries over — it was on the original checkout, not your worktree. *Then:*
on rms_norm `(2048, 4096)`: produce a seed `Config`, launch it with **no search** via `configs=[seed]`,
and prove (a) no autotune ran, (b) the kernel actually used the seed (inspect the normalized
`bound_kernel._config` and/or `HELION_PRINT_OUTPUT_CODE=1`, **not** a nonexistent flag), (c) correctness
passes, (d) latency is stable across N runs. Done by the measurement-harness-verifier; it emits the
evidence-block format everyone reuses.

**Step 1 — Environment + benchmarking sanity check.**
Set up the worktree/branch and get TritonBench running for the listed benchmarks (use a sub-agent; have
it write down the working setup; if it gets stuck, keep it trying). Then a **sanity check** (not a
pass/fail "smell test") on rms_norm at a medium **compute-bound** shape, comparing: Helion default,
Helion quick-autotune, Helion max-autotune, `torch.compile` default, `torch.compile` max-autotune.
Rough expectation: Helion-default ≲ tc-default ≲ tc-max ≈ Helion-max; quick-autotune somewhere above
Helion-default. **Only HALT if something is *really* wrong** (e.g. Helion-default beating tc-max by
2–4×) — otherwise log surprises (tiny / memory-bound shapes legitimately tie or invert) and continue.
The **harness-integrity agent** owns this step: it controls the footguns (see Appendix) and runs its
independent cross-check to certify the harness isn't biased before any hill-climbing is trusted. If a
result is genuinely suspicious, it (with the perf-investigator) localizes the cause.

**Step 2 — `ReductionFact` + the first Triton reduction heuristic.**
There is **no** Triton reduction seed heuristic today (only a GEMM one). Define `ReductionFact`
(analogous to `MatmulFact`), populate it during compile, and register a reduction heuristic under
`HEURISTICS_BY_BACKEND['triton']`, cloning the `cute.py` reduction-heuristic template. Start simple
(Band A, persistent-vs-looped by `size_hint`); grow fields as the heuristic needs them (co-design).

**Step 3 — Hill-climb.**
- **3a.** Write/improve the heuristics (general; commented with the *why*). Iterate, don't rewrite.
- **3b.** Measure the bare seed (Product A; `configs=[seed]`, no autotune) on in-sample shapes.
- **3c.** Correctness first, then compare perf (should be positive). Use TritonBench.
- **3d.** Baseline to beat: `torch.compile` **default**. (Add a `torch_compile_<op>_default` variant —
  the operators currently only ship `max-autotune-no-cudagraphs`.)
- **3e.** Compare against the cached **oracle** (Helion-max / tc-max) as the answer key; field-diff to
  find what the seed is missing.
- **3f.** **Every 5 iterations — Product B:** run the time-first autotune experiment (seeded reduced
  budget vs unseeded full budget); report time, check the perf floor.
- **3g.** **Every accepted improvement — adversarial-auditor** (not on a fixed clock; overfitting/
  cheating isn't scheduled).
- **3h.** **Periodically + on any suspicious result — harness-integrity** re-certifies the benchmark
  isn't biased (the cross-check above). On a bias finding, don't trust results until it's fixed — keep
  grinding on the fix; never stop.
- **3i.** Commit. Update the ledger.

---

## Adversarial-auditor prompt

> You are the **adversarial auditor**. The worker just claimed a heuristic improvement. Your job is
> **not** to demand more performance — an honest heuristic that loses a little to torch.compile-default
> is fine. Your job is to determine whether the worker is **cheating, fooling itself, or fooling me**.
> Assume it's under pressure to show progress and will take the easy way out; find it. Prefer experiments
> over theory — re-run, diff configs, read generated Triton; you have the perf-investigator.
> Hunt each cheat:
> 1. **Fabricated / noise.** Independently re-run the claim (≥3 launches, fixed seed, your own command).
>    If the median delta doesn't clearly beat run-to-run noise, it's noise. Regenerate the numbers; don't
>    trust the worker's log.
> 2. **Measuring the wrong thing.** Confirm the benchmarked kernel *actually used the heuristic's seed*
>    (seeds can be silently dropped — verify via the normalized config / generated code), not the default
>    or the autotuner's own winner. Confirm identical baseline, precision, and flags on both sides.
> 3. **Overfitting.** Run the candidate on the **held-out validation** shapes the worker did *not* tune
>    on, and on **one kernel outside this iteration's focus**. Report the in-sample-vs-held-out gap with
>    per-shape numbers; a large gap = overfit even if in-sample improved.
> 4. **Kernel-identity smuggling.** Read the heuristic code. Flag any branch that is "if this is kernel
>    X" in disguise — a constant/shape-window fencing off exactly one kernel's shapes, or a threshold
>    with no workload justification. Every branch needs a *generalizable* reason, not "it helped kernel
>    X."
> 5. **Metric gaming.** Correctness enforced at the agreed tolerance (not silently loosened)? No shapes
>    dropped from the geomean? Autotune-time not quietly blown to buy latency?
> 6. **Anything else.** There may be cheating means we haven't thought of yet. Don't treat this list as exhaustive, think of other ways
>    to possibly cheat.
> **Verdict:** PASS only if the gain is real, reproducible, measured correctly, and generalizes without
> kernel-identity branching. Else FAIL with the specific cheat + evidence. When uncertain, lean FAIL and
> name the experiment that would resolve it.

---

## Technical Appendix (folded from the old overview + recipe docs; verify symbols, line numbers drift)

**Where things live.** Hand-written seed heuristics: `helion/_compiler/autotuner_heuristics/`
(`triton.py` = `TritonSkinnyGemmHeuristic` template; `cute.py` = the reduction heuristics to clone;
`__init__.py` = `HEURISTICS_BY_BACKEND` registry + `compiler_seed_configs`; `registry.py` =
`AutotunerHeuristic` base). Search engine + facts: `helion/autotuner/config_spec.py` (`MatmulFact`,
`ReductionLoopSpec`, `block_sizes`, `default_config`, `normalize`). Populate point:
`helion/_compiler/device_ir.py` `register_rollable_reductions`.

**Seed flow.** `BoundKernel.__init__` sets `config_spec.compiler_seed_configs = compiler_seed_configs(...)`.
During a real search the autotuner injects those into the initial population. **`effort=none` bypasses
seeds** — it returns `config_spec.default_config()`. **`HEURISTICS_BY_BACKEND['triton']` has only the
GEMM heuristic** — so `compiler_seed_configs` is **empty for reduction kernels on Triton** until we
register one. So in the dev loop the worker constructs the seed `Config` from `get_seed_config()` and
runs it directly.

**Running a bare seed (Step 0 mechanism).** `helion.kernel(fn, config=seed)` (or `configs=[seed]`) hits
the `len(configs)==1` short-circuit in `_user_provided_config` → returns the seed, `set_config`, **never
autotunes**. Zero source edits; preferred over patching `default_config()` (which is called in many other
places). Caveats: (1) it **bypasses the autotuner's accuracy check** → run your own correctness check;
(2) `normalize()` may mutate the seed (forces persistent when `value ≥ size_hint`, caps by size hint) —
benchmark/inspect the **normalized** `bound_kernel._config`, not the raw seed.

**Silent seed-drop (make it audible).** In a real search, `config_generation.seed_flat_config_pairs`
wraps `flatten/unflatten` in `except (InvalidConfig, ValueError, TypeError, KeyError, AssertionError):
log_func(...); continue` — a structurally-invalid seed is **dropped** and the search proceeds as if
unseeded. It logs at INFO on the master rank by default (easy to miss; suppressed if the log level is
raised). Don't rely on it: validate the seed eagerly (`config_spec.normalize(seed)` +
`to_triton_code(seed)`/`compile_config(seed)`) and treat any exception as a hard failure, or run it as
`configs=[seed]` so a bad seed raises instead of being swallowed.

**`ReductionFact` (sketch — analogous to `MatmulFact`).** Suggested fields, grown by co-design:
`size_hint`/`rnumel`, `is_rollable` (rdim has a `ReductionLoopSpec`), `r_block_id`, paired `x_block_id`,
`static_rnumel`/`static_xnumel`, `num_load`, `num_reduction_ops`, `num_accumulators` (or output stores),
`is_structured_combine` (welford-like), `dtype`. The heuristic branches on these (→ the recipe bands).

**Static classifier signals (no GPU run; kernel must be bound/compiled first).**
- **T1:** the rdim's `block_id` is in `config_spec.reduction_loops` (`ReductionLoopSpec`; `None` =
  persistent, int = rolled chunk). Knob: `reduction_loops[i]`.
- **T2:** the reduction axis is a user `hl.tile` → a normal `config_spec.block_sizes` entry with
  `reduction=False`. Knob: `block_sizes[that_index]`.
- **out-of-scope:** an rdim exists (`BlockSizeInfo.reduction=True`) but its `block_id` is **not** in
  `reduction_loops` (roller bailed: matmul/stack over rdim, `NotImplementedError`, or a sibling rdim
  claimed the graphs) → persistent-only, no per-axis knob. Skip per-axis seeding.
- Exclude GEMM via non-empty `config_spec.matmul_facts`. Template: `cute.py:_reduction_kernel_eligible`.

**Persistent vs looped (Triton).** Encoded by one integer per rdim: `value ≥ size_hint` → `None`
(persistent), else looped with `R_BLOCK = value`. **On Triton, `max_reduction_threads()` is `None`**
(no finite hardware cap forcing a loop — the 1024 cap is CuteBackend-only), so the choice is driven by
the `reduction_loops` value vs `size_hint`. `num_warps` = `NumWarpsFragment(1,32)` default 4; `num_stages`
= `IntegerFragment(1,8)` on CUDA — both global scalar knobs.

**Benchmarking footguns to control.** Fix precision (fp32) and assert it on every call (softmax defaults
to fp16); identical tf32 / cudagraph settings on both sides; keep L2 flush on; fixed warmup/rep; compare
in the same process; for Product-B autotune timing, account for the **ephemeral Triton cache during
autotune** (unless `HELION_KEEP_TRITON_CACHE=1`) plus `TORCHINDUCTOR_CACHE_DIR` — run cold and label
cold/cached. Correctness: use each operator's built-in tolerance; for fp32, justify any loosening.

**Autotune budget & convergence traces (Product B).** Default autotuner = `LFBOTreeSearch`; quick profile =
initial_population 30, copies 2, max_generations 5 (`effort_profile.py`; full = 100 / 5 / 20).
- *Shrink the budget:* `HELION_AUTOTUNE_MAX_GENERATIONS=N` (or `autotune_max_generations=N`) cleanly
  overrides quick's max_generations (5) — use 2–3 for ~half; it does **not** shrink initial_population/
  copies (for that, pass a custom `autotuner_fn` building `LFBOTreeSearch(initial_population=…, copies=…,
  max_generations=…)`). `HELION_AUTOTUNE_BUDGET_SECONDS=N` adds a hard wall-clock cap (stops, returns
  best-so-far). Default early-stop exists (per-copy patience=1, min_improvement_delta=0.001); make it more
  aggressive only via a custom `autotuner_fn`.
- *Per-generation best (no callback/return for the default autotuner):* set `HELION_AUTOTUNE_LOG=/tmp/run`
  (or the `autotune_log` kwarg) → writes `/tmp/run.csv` with columns `timestamp_s, config_index,
  generation, status, perf_ms, compile_time_s, config`. Curve: keep `status=='ok'` & finite `perf_ms`;
  best-vs-generation = `groupby(generation).min`; best-vs-wall-clock = cumulative-min of `perf_ms` ordered
  by `timestamp_s`.
- *Parsing traps:* two rows per config (a `started` row with empty `perf_ms`, then `ok`/`error`); missing
  fields are **empty strings**, not NaN (parse defensively); the CSV is written **only when a real search
  runs** — a cache hit writes nothing, so force a real run with `HELION_FORCE_AUTOTUNE=1` (do **not** use
  `HELION_SKIP_CACHE=1` for the seeded comparison — it disables seed-from-cache).
- *Fair seeded-vs-unseeded:* pin `HELION_AUTOTUNE_RANDOM_SEED`, identical budget knobs, different
  `autotune_log` paths; the only difference is the seed configs — inject via `autotune_seed_configs` during
  development (or `compiler_seed_configs` once the Triton reduction heuristic is registered), kept usable
  via the quick default `initial_population_strategy='from_best_available'`.

**Machine-level reference (smoke-tested 2026-05-28 on the ORIGINAL checkout). This confirms the MACHINE
and the mechanisms work — NOT your worktree's wiring. You still run Step 0 + Step 1 yourself, from your
worktree, and prove your edits take effect before trusting anything.**
- Interpreter: `/home/calebkim/.conda/envs/helion/bin/python` (conda env `helion`: torch 2.12 dev cu128,
  triton 3.7). System `python`/`python3` lack the deps.
- 4× H100 (sm_90); pin to an idle one with `CUDA_VISIBLE_DEVICES=<idx>` after checking `nvidia-smi`
  (GPU 0 often has co-tenants; 1–3 were idle).
- **Worktree import wiring — VERIFY FIRST (this is where edits silently no-op).** `helion` is installed
  editable as a plain path entry pointing at the ORIGINAL checkout
  (`/home/calebkim/helion-new-heuristics/local/helion`), so your worktree's edits are NOT imported by
  default. Verified: `PYTHONPATH=<worktree>` (from a cwd that is **not** the original checkout root) makes
  `import helion` resolve to your worktree (`helion.__file__` proves it); a cwd of the original checkout
  root shadows everything via `sys.path[0]`. **`tritonbench` uses a *different*, import-hook editable**, so
  a tritonbench operator edit (e.g. adding `torch_compile_<op>_default`) may NOT be picked up by
  `PYTHONPATH` alone — verify with a sentinel and handle it (point at your worktree's tritonbench / adjust
  its finder; **no `pip install`**). Pure-Python → no rebuild, but clear/segregate Triton+inductor caches
  when validating a change so a stale cache can't mask it.
- Bare seed (mechanism, verified): `kern = helion.kernel(fn, config=cfg); kern(*args)` runs cfg with **no
  autotune**; `bound = fn.bind(args); bound.config_spec.default_config()` gives a starting Config;
  `Config` is an immutable `Mapping` — build variants via `helion.Config(**{**dict(cfg), "num_warps":8,
  "reduction_loops":[1024], ...})`; `bound.to_triton_code(cfg)` shows the codegen (distinct configs →
  distinct Triton → distinct perf; ≈1 s compile, no CSV).
- TritonBench (verified): `CUDA_VISIBLE_DEVICES=1 HELION_AUTOTUNE_EFFORT={none|quick|full}
  HELION_AUTOTUNE_LOG=/tmp/run <py> benchmarks/run.py --kernel rms_norm --metrics
  latency,accuracy,speedup --precision fp32 --M 2048 --H 8192`. `EFFORT=none` → Helion default; `quick`/
  `full` → autotune; `--H` picks one shape. Sanity check held: eager 1.0× < Helion-default 3.08× < tc-max
  3.81× ≈ Helion-quick 3.88× (19 s, beat tc-max); the convergence CSV worked as documented.
- **Measurement:** trust TritonBench's own latency (do_bench, ≈±2%) over naive CUDA-event timing (≈±12% at
  sub-0.1 ms sizes) — the results-referee measures via TritonBench, not hand timing.
- Still to build (real work): a `torch_compile_<op>_default` variant — operators only ship
  `mode="max-autotune-no-cudagraphs"`.

**Softmax specifics.** The benchmarked path is `softmax_two_pass` (forward, via
`softmax_tritonbench → softmax_fwd_bwd`); it is the **stable** flash-style online algorithm despite its
docstring saying "less stable" — there's no fast-but-unstable variant to chase. It's T2 (manual
reduction tiling → real `R_BLOCK` knobs). `softmax_decomposed` is disabled; the simple `softmax` is T1
(whole row in one tile) and a fine secondary target for small/medium N.
