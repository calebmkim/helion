# Helion seed-heuristic hill-climb — METHOD (durable, task-agnostic)

This is the reusable *how*. It does not change run-to-run. A per-task file (e.g.
`dtype-task.md`) supplies the *what*: which heuristic, which goal, which knobs, which shapes,
the deliverable. **Read this file, then `local-setup.md`, then the task file.** Where they
conflict, the task file wins on scope; this file wins on method and discipline.

## START HERE — are you resuming or starting from scratch?
**If you are picking up an in-progress run:** the gated log is your source of truth (§6.1) — read
the existing **notebook + ledger** (paths in the task file / `local-setup.md`)
to recover the current heuristic, what's banked, the deferred hard-pile, and the exact next action,
then resume from there. **Trust the log over any assumption.**
**If you are starting from scratch (no prior log):** there is no heuristic to inherit — your first
job is to establish a **baseline seed** (the simplest defensible config, or the current
`compiler_seed_configs()` output if one exists) and a **fresh notebook + ledger**, then hill-climb
*from* that baseline using the method below. Don't wait for a heuristic to be handed to you; create
the baseline and start climbing.

The job is always the same shape: a Helion **autotuner seed heuristic** emits a strong starting
config for a class of kernels; you make the seed produce **good, robust performance** on a
curriculum of shapes, proven rigorously, without regressing anything. The task file says which
heuristic and what "good" means this time.

---

## 1. Environment & hard rules (never violate)

**First read `local-setup.md` (same dir)** for the concrete machine facts: worktree path,
interpreter, git remotes, GPU status for *this* run, and the key script locations. The durable
principles below don't change; the specifics live there so this file stays reusable.

- **Interpreter:** the shared venv named in `local-setup.md` (has every dep). **Never `pip
  install`** or any networked/system install.
- **Run scripts from `cwd=/tmp` with `PYTHONPATH=<worktree>`** so `import helion` resolves to the
  worktree (not the editable install), and **assert `helion.__file__` is under the worktree at the
  top of every script** — the silent wrong-helion footgun has burned multiple runs.
- **GPU:** `local-setup.md` says whether the GPU is shared for this run. **If it is shared,**
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` first and only time when idle. **If
  it is dedicated** (the common case here — the human will tell you if another agent needs it), you
  may run back-to-back without idle-gating; a quick `nvidia-smi` before a headline timing is still
  cheap insurance. **Regardless of sharing, NEVER launch a detached/background GPU job** — they get
  SIGKILL'd silently and never notify (a detached oracle once stalled a run 13h). Long oracles run
  **foreground, one shape at a time**, JSON-checkpointed after every shape so a kill loses nothing.
  Per-config 60s compile / 30s bench timeouts are normal **skip-and-continue**, not failure.
- **Git — commit frequently (use your judgement; err toward more).** Banking working state on
  immutable SHAs is how a fresh context resumes cleanly and how gates pin what they verified — so
  commit after each green step (a passing A/B, a behavior-oracle 0-diff, a parity milestone). You
  do **not** need to ask before committing to your own branch. Do **not** `git push` unless asked —
  the human updates the PR. Never `git checkout <file>` while you have pending edits on it (it
  discards them — looks like env reversion, is self-inflicted). Verify tree state with `git status
  --porcelain helion/` + `git rev-parse HEAD`, never `git diff <remembered-sha>` (a committed stack
  reads as spurious divergence). `origin` = the fork, `upstream` = pytorch/helion — never push
  `upstream`.
- **Model:** run on `us.anthropic.claude-opus-4-8[1m]`. The CLI `opus` alias resolves to an OLDER
  model; a team-spawn with `model` omitted silently downgrades. Hardcode the literal string on any
  spawn/respawn and self-assert at boot.
- **Portability:** no hardcoded paths/SHAs/GPU indices/`startswith`-literal asserts in committed
  state. Derive the worktree root (`git rev-parse --show-toplevel`).
- **No `del`/`_ =` on unused args; no defensive `hasattr`/`getattr`/`except` noise; no `print()`
  inside kernels.** Match surrounding style; run `./lint.sh fix` before any push the human asks for.

---

## 2. What a seed heuristic is (the substrate)

A Helion `AutotunerHeuristic` (in `helion/_compiler/autotuner_heuristics/`) has
`is_eligible(env, device_ir)` and `get_seed_config(env, device_ir) -> Config | None`.
`compiler_seed_configs()` collects every eligible heuristic's config as a **seed** — a strong
point planted in the autotuner's search space, and (if `promote_seed_to_default`) the compiler
default. **A seed is never forced**: a bad seed only costs autotuning time, never correctness.

It branches on **facts** — faithful *workload properties* recorded at compile time (e.g.
`ReductionFact`, `MatmulFact`), **never kernel identity**. This is the whole philosophy: a fact is
something the kernel structurally *is* (extent, itemsize, a loop-carried accumulator, a re-read
row), so the seed generalizes to shapes/kernels never seen. If you ever want to branch on "which
kernel is this," you've mis-modeled — find the underlying property.

**The heuristic and its facts are YOURS to rewrite — be aggressive.** The facts (`ReductionFact`
etc.), the heuristic classes, their constants, levers, and branches are not fixed scaffolding: add
fields, add methods, add new facts, restructure or split classes, introduce new levers — whatever
the perf demands. The only hard constraints are the gates (§5): a new fact must be a faithful
workload property (not a kernel-identity fence, not a curriculum-lucky proxy — it must survive the
divergence test), and every change is measured and verified. Within that, change the substrate
liberally; don't contort a workaround to avoid touching the fact definition. (Adding a *fact* means
also populating it in `device_ir`'s fact builders — that's expected, not off-limits.)

**Two kinds of facts — and only walker facts may touch the graph.** Facts split by *how they are
populated*, and the split is a hard rule, not a preference:
- **Walker facts** (`MemoryOpFact`, `AccumulatorFact`) — one per structural entity (a load/store op;
  a loop-carried accumulator). They are the **only** facts allowed to walk the graph, and they walk
  it **once**, in the compiler's collect pass (`_collect_memory_op_facts` + the accumulator builder).
  Each field records **raw compiler provenance about the entity itself**, kept **consumer-agnostic** —
  no notion of "the reduction," "the band," or any one heuristic. The **standing goal is maximum
  generality**: shape a field as the entity's raw property so a *different* future fact/heuristic can
  read a *different slice*. Calibration for "general enough": `reductions_fed` is per-axis
  `(axis, count)`, **not** `feeds_my_reduction: bool`; `indexed_block_ids`/`inner_extent` are raw
  shape provenance; `AccumulatorFact` carries `dim_block_ids` + `itemsize` with **no** reduction-axis
  notion. Generality is the goal, not an absolute — a field with genuinely no general form is
  acceptable *only if* you have shown so and logged it.
- **Derived facts** (`ReductionFact`, and the per-kernel-class facts you add) — what the heuristic
  branches on. They **NEVER walk the graph**: no `device_ir.graphs` iteration, no
  `node.users`/`node.args` dataflow, no `_classify_load_dataflow`, no IR inspection. Every field is a
  **pure derivation** over walker facts + trivial structural reads (`block_id`, `size_hint`,
  `block_sizes`, `static_rnumel`). The **kernel-specific interpretation lives here, in the derived
  fact** — pick *this* reduction's axis out of `reductions_fed`, match `dim_block_ids[-1] ==
  red.block_id`, and so on.

The bright line: **if a value needs the graph, the walk goes on a walker fact (one general field,
computed once) and the derived fact reads a slice of it. Never walk in a derived fact; never bake a
consumer's identity into a walker field.** Two more rules:
- **Prefer a field on an existing walker fact.** Adding a *new* walker fact is allowed, but
  proliferation is a cost — first try to hang the field on `MemoryOpFact`/`AccumulatorFact` (whichever
  models the entity); create a new walker fact only when the property is about an entity none of the
  existing ones model.
- **Soundness is required.** A walker field must use genuine compiler provenance, never a guess, and
  must be sound — do **not** add new accepted-unsoundness. The single pre-existing exception is the
  config-free eviction-index slot (`MemoryOpFact.eviction_index` → `ReductionFact.reread_eviction_index`):
  it is **unsound today and a known TODO to fix**, *not* a precedent to copy.

The fact-gate (§5; Gate D in `gate-prompts.md`, Part 1) enforces all of the above whenever a
heuristic reads a new or changed fact field.

Two products the seed serves:
- **Product A** — *skip autotuning*: `configs=[seed]`, the seed IS the config. This **bypasses the
  autotuner's own accuracy check**, so you must correctness-gate it yourself.
- **Product B** — *seed the search*: the seed is the autotuner's initial population, so a full
  search converges in less wall-clock. **Secondary for now** — focusing on A almost certainly helps B.

---

## 3. THE METHOD — the goal hierarchy (this is "how to direct it")

> **THE WHOLE JOB IS THE LOOP.** After the one-time Step 0 gate, ~99% of your time is the
> per-iteration loop below, run over and over — banking the task's definition-of-done as a milestone
> when you reach it and then **climbing past it** (§6.0); you never stop on your own. The goal
> hierarchy, footguns (§4), and gates (§5) all exist to make each turn correct. Unsure what to do
> next? Do the next turn of the loop.

### Step 0 — One-time setup + a sanity check (do this ONCE, then never again)
Before touching the heuristic, prove the machine works on **a single shape** end-to-end — a
miscalibrated harness produces *plausible-but-wrong* numbers you'd chase for hours (see §4 footguns).
Cheap insurance:
1. **Env wired:** `import helion` resolves to the worktree (assert `helion.__file__`); the venv runs;
   GPU visible (`nvidia-smi`).
2. **Bench harness sane on ONE shape** (throwaway calibration — Step 1 re-measures fresh): on one
   representative curriculum shape, measure seed / unseeded-default / tc, forward-only, dynamo-reset,
   single-process (§4). Sanity-check (right order of magnitude, accuracy passes, the *seed* config is
   the one that ran). If the harness disagrees with a hand-rolled **single-process** `do_bench`
   cross-check by >3%, **stop and fix the harness** before climbing (the ~5–10% figure in §4 is
   *cross*-process jitter — it doesn't apply to a same-process cross-check).
3. **Seed mechanism proven:** `configs=[seed]` runs *no* autotune and uses the normalized config.
4. **Notebook + ledger created** (or the prior ones loaded — see START-HERE).

That's it. **Setup is a gate, not a phase.** Once it's green, you are in the loop for the rest of
the run.

---

**The goal hierarchy (Steps 1–4 = how you decide what to feed the loop, and when to stop).** The
hard part is knowing *what to chase and when to stop*. **The primary objective is DISASTER AVOIDANCE
+ pretty-good performance against torch.compile-default** — the portable, reproducible yardstick.
Concretely, two bars, both measured as `G = tc_latency / seed_latency`:
1. a **per-shape floor (disaster avoidance):** **no shape below `G = 0.75`** (seed never worse than
   ~1.33× tc-default's latency). The hard, non-negotiable constraint — a shape under it is a
   *disaster* and is top-priority worklist.
2. a **per-kernel quality target:** **every kernel's geomean(`G`) ≥ 0.85** — the geomean over that
   *one kernel's* shapes, computed **separately for each kernel and each active dtype** (every
   (kernel, dtype) pair must clear 0.85 on its own). "Pretty good on average" per kernel — there is
   **no** single matrix-wide geomean, so a strong kernel or dtype can never carry a weak one.

**Both are MINIMUMS, not targets to settle at.** Clearing them makes the deliverable bankable; it is
never a reason to stop — never-stop (§6.0) keeps you climbing past them (beat tc, then chase the
oracle — §Step 4). And the per-shape floor is precisely what stops a healthy *per-kernel* geomean
from *masking* a disaster shape inside that kernel: you cannot average your way over a shape that's under 0.75.

**The oracle (autotune) is a *tool you reach for*, not the lead — and you may reach for it fairly freely.**
Property-reasoning (Step 2a) is the default first move: it's cheap and usually clears a 0.75 floor on its
own. But you do **not** have to suspect a fundamental limit to run the oracle — reach for it whenever you're
**having trouble reaching torch.compile-default by reasoning alone and want an "answer key"** (the
autotuner's winning config is *information* that makes the hill-climb faster), **or** when you suspect a
**fundamental (codegen/source) limitation**. The only caution is cost: a full oracle is slow, so don't
*reflexively* autotune every gap before you've reasoned about it — but a shape you're stuck on, or one
where an answer key would clearly speed the climb, is a perfectly good reason to run it. **Bar reminder:**
even when the oracle *beats* tc, your goal for the shape is still only **the floor, then tc-parity**
(`G ≥ 0.75`, then `G ≥ 1` in overtime) — not matching the oracle; chasing the oracle past tc is overtime
bonus (§Step 4), good if you have the time. It serves two roles:
1. a **reachability detector** — can *any* achievable config clear the floor here, or is tc itself
   codegen-unreachable? and
2. an **answer key** — a concrete config to hill-climb the seed *toward*, and (potentially) the basis for
   **retargeting** a codegen-bound shape's floor to `0.75 × oracle` (Step 2).

### Step 1 — Establish the torch.compile yardstick (cheap, always first)
Measure **seed vs torch.compile-default** across the curriculum (and, for a multi-target run, across
every active dtype). This is the headline: `G = tc_latency / seed_latency` (G≥1 = seed beats tc;
G=0.75 = seed is 1.33× tc's latency). Cheap (no autotune); it triages where to spend effort against
the two bars (above):
- **any shape with `G < 0.75`** → a **disaster**, top-priority worklist. Go to Step 2.
- **any kernel whose geomean(`G`) < 0.85** (at any active dtype — no single disaster, but that
  kernel lags on average) → lift *that kernel's* lowest-`G` shapes until its geomean clears. Go to
  Step 2 on them.
- **every shape ≥ 0.75 AND every (kernel, dtype) geomean ≥ 0.85** → both bars met; the deliverable
  is **bankable**. Everything beyond is **overtime** (Step 4) — and never-stop means you keep going
  (beat tc, then the oracle).

Apply normal measurement-noise tolerance at the boundary — don't thrash a shape oscillating around
0.75; it's the clear sub-0.75 shapes that are disasters.

### Step 2 — Lift a below-floor shape (reason first; reach for the oracle when stuck or on suspicion of a hard limit)
A shape is below floor (`G < 0.75`) or dragging the geomean. **First try to lift it WITHOUT the
oracle** — that is the default path now, and it's usually enough to clear a 0.75 floor:

**(a) Understand *why* it's slow, then form a workload-property hypothesis.** Read the generated
Triton on both arms and reason about the gap (block/tile sizes, num_warps, stages, eviction,
`pid_type`, persistent-vs-looped, loop chunk…):
- Helion: `Kernel.to_triton_code(config)` (or `HELION_PRINT_OUTPUT_CODE=1`).
- torch.compile: `TORCH_LOGS=output_code python ...`.
Many below-floor shapes are a recognizable, *general* mis-seed (wrong warps for the rnumel, a
persist/loop flip, a missing byte cap) you can fix from the property alone — no autotune needed. Climb it
(Step 3) and re-measure against the 0.75 floor. But if reasoning stalls, or an answer key would simply be
faster than reasoning blind, don't hesitate — pull the oracle (2b).

**(b) Reach for the oracle when reasoning is struggling — or when you suspect a hard limit.** You do
**not** need to suspect a fundamental limitation; either trigger suffices: (i) you **can't lift the shape
to its floor by reasoning** (you're stuck, or a concrete answer-key config to field-diff against would
clearly speed the climb), or (ii) you suspect tc's structure may be **codegen-unreachable** (no Helion
primitive for it, e.g. split-reduction). Getting an answer key is a legitimate reason on its own — don't
wait to "earn" the oracle by first proving a fundamental limit (just don't reflexively autotune *every*
gap before reasoning — it's slower). Run the autotune oracle, which does two jobs at once:
- **oracle beats (or ties) tc** → the floor is reachable and tc was *not* a fundamental limit. Your
  **answer key** is the oracle's winning config — or, if that's complicated, the simplest
  config that still clears the floor (Step 3). Climb toward it (and, in overtime, to beat
  tc).
- **oracle can't beat tc** → Helion's codegen genuinely can't express tc's optimum here; tc is
  unreachable. **Retarget this shape's floor to `0.75 × oracle`** (the best Helion can structurally
  do) and climb to it. The shape is **acceptable only once `seed ≥ 0.75 × oracle`**; while it's below
  that there is reachable perf on the table, so it **stays on the worklist even though tc is out of
  reach** (real miss: jsd — the oracle was ~18% faster than the seed, yet it was waved off as
  "exempt" because *both* lose to tc). "Helion can't beat tc here" is a real, document-worthy
  finding, but it only changes *which yardstick the floor is measured against* — it never removes the
  floor. A codegen-ceiling claim is valid only on a **converged** oracle (search-finished, not
  truncated/OOM) genuinely losing to tc — route it through anti-giving-up (§5), never self-certify.

**Quick-autotune is the cheap escalation** when you do go to the oracle: much faster than a full
oracle and, unlike tc, it directly hands you a Helion **config you can hill-climb toward**. Use it to
get a target fast, then confirm with a full oracle before banking a reachability/ceiling claim — a
quick *gap* is real (only widens at full), but a quick *parity/win* is **suspect** and must be
full-confirmed (a real run saw `softmax(1024,65536)` flip 1.000→1.338 between quick and full).

### Step 3 — Hill-climb the seed up to its floor (and, in overtime, beyond)
This is the same edit→A/B loop whether your target came from **property-reasoning** (Step 2a — the
common path) or an **oracle answer key** (Step 2b). The answer-key mechanics below apply when you
have an oracle target; the bar you climb to is the **floor** (§Step 2), not "beat tc."
- **Cache the oracle** `{winning_config, latency}` keyed by `(kernel, shape, dtype, source-hash)`;
  invalidate on any source/codegen change. Re-comparing the cached oracle is free; only re-bench
  the seed.
- **The oracle hands you ONE config — a *clue*, not a target to reproduce.** A full autotune returns a
  single winning config (its finishing phase already drives fields toward defaults; there is **no ranked
  leaderboard with latencies to scan** — don't look for one). That winner is the *peak*, but your bar
  here is only the **floor** (`G ≥ 0.75`, or `0.75 × oracle` if codegen-bound) and the kernel geomean
  ≥ 0.85. Use the winner for its *information* — what it reveals about the mechanism — not as a config
  to copy.
- **Target the SIMPLEST config that clears the floor, not the peak.** Field-diff the winner vs your seed
  (block/tile sizes, reduction loops, `num_warps`, `num_stages`, eviction, `pid_type`, `maxnreg`) —
  **the differing fields are your worklist.** Hypothesize, from the diff *as a whole*, which field(s)
  carry the win, aim the heuristic at *those*, and **confirm by re-benching the candidate**
  (`configs=[cfg]`, the seed's bare-forward harness) that it clears the floor — a couple of confirmatory
  re-benches, not a leaderboard scan. A few % under the winner is plenty: pretty good, not the peak.
  **Simplicity breaks only genuine ties:** among configs of *comparable* complexity, take the one with
  **better perf** — never give up real perf when the simpler-looking option isn't clearly simpler.
  **Couplings are real and common — be ready to move two fields *together*:** sometimes neither field
  alone gains anything (one alone may even *regress*) and only the *pair* helps (e.g. raising
  `num_warps` pays off only once the block size moves with it). So don't force a one-field split — when
  the diff and your hypothesis point at a coupled pair, change *both* and emit the bundle as a unit. The
  matched-lever A/B (loop step 4) — which perturbs *down* from the good config so the partner stays
  present — is what proves which field(s) actually carry the win, and is the only way to see a coupling
  that a build-up-from-seed test would hide.
- **"Suspect" = an *unfaithful key*, not a complicated-looking config.** A field is suspect when its
  *value has no faithful workload-property mapping* (a hand-tuned `maxnreg`, an exotic `pid_type` you
  can't key on a real property) — coupled or not. Don't bake a suspect field in to chase perf the floor
  doesn't need; if the win genuinely *requires* one (the A/B shows it and no faithful property reaches
  it), it's a **hard-pile item (below)**. This is a pre-screen — the real generality verdict is Gate H
  (§5).
- **HARD PILE — the "set it aside so we don't churn" pile.** A shape goes here whenever it is hard,
  for either reason: **(a) HANDLED** — a complicated/suspect config DOES clear its floor but you want a
  simpler general rule (cache it, move on); or **(b) STUCK** — you canNOT clear its floor right now: the
  win is real but under the noise floor (unmeasurable), there's no faithful re-key (try-harder came back
  empty), or seed ≈ a fresh converged oracle (true codegen ceiling). A **STUCK** entry requires **Gate B
  to clear it first** (anti-giving-up: fresh oracle, a different workload property tried, every firing
  shape measured) — hard-piling a stuck shape IS a stop-claim. **Tag every entry with its reason.** Then
  move on (never re-pick a hard-piled shape every turn — that churn is exactly what this prevents).
  Revisit the pile as a **batch**: HANDLED entries for a **common thread** (a rule from the batch
  generalizes where a one-shape fit overfits — work it by that shared workload-property, not one
  contorted config at a time); STUCK entries are also written to the human-review queue. (A HANDLED entry
  needs no gate — its bar is met. A confirmed codegen ceiling retargets the floor to `0.75 × oracle`,
  Step 2b — that's a STUCK reason, not an exemption.)
- If a full oracle dies *post-convergence* (search done, cache-write crashes), extract the converged
  winner from the `.out` log and fair-re-bench it (don't discard). To re-bench an arbitrary extracted
  config, run it as `configs=[that_config]` through the **same** bare-forward harness as the seed arm
  — never hand-roll (you'd reintroduce the §4 footguns).

**The bar you climb to** (per the two bars above): a shape's floor is **`G ≥ 0.75` vs tc** by
default, or **`seed ≥ 0.75 × oracle`** if it's codegen-bound (Step 2b) — either way it's **live
worklist, not exempt** until cleared, and the kernel's geomean must reach **0.85**. Past the floor,
**keep climbing in overtime** (§Step 4) — the floor is the *bankable* bar, not a stop sign.

**Invariant that ties it together:** the oracle searches the *same codegen* the seed emits, so a
source/codegen ceiling caps the oracle *too*. Therefore:
- `seed < oracle` ⇒ achievable perf is on the table; if you're in overtime on this shape, **keep
  climbing** (never "noise"/"stuck" until a fresh oracle says so).
- `seed ≈ oracle` ⇒ **nothing more is reachable for this shape** — you're at the codegen ceiling (the
  per-shape overtime endpoint). If that ceiling can't beat tc, the kernel source can't beat tc;
  confirm with a fresh converged oracle (Step 2b), never self-certify.

### Step 4 — Overtime: once the bars are met, keep climbing (beat tc → oracle parity → beat the oracle)
The bars (every shape ≥ floor, every (kernel, dtype) geomean ≥ 0.85) are **bankable, not a stop.** Once they hold, the
climb continues in priority order: push the cleared shapes from floor → **beating tc** (`G ≥ 1`) →
**oracle parity** (`seed ≤ oracle × 1.03`, per-shape) → and beyond, hunt configs that *beat* the
oracle (couplings its bounded stochastic search under-samples). This is the "if you get good perf,
more power to you" zone — pure upside on top of a banked deliverable. A win must come from **theory +
the answer-key diff**, never from cherry-picking an observed search winner (p-hacking). "No clean
rule / noise / stuck / done / ceiling" is **not an exit** — it's the trigger to run a fresh oracle
and read the answer key.

**Priority discipline (overtime waits its turn):** never spend effort pushing an already-above-floor
shape toward beating tc / oracle parity while *another shape is still below floor* or *any kernel's
geomean (at any dtype) is still under 0.85*. Disasters first, then per-kernel geomeans, then overtime
— see the priority guard in the loop.

**The DoD is a milestone to BANK, not a finish line.** When the task file's definition-of-done is
met and verified — every shape ≥ its floor and every (kernel, dtype) geomean ≥ 0.85 — **freeze and bank it** — commit the
champion, write the report — so a fully-valid deliverable is locked in and can never be lost. Then
**keep climbing** (overtime above): push the floor-clear shapes toward beating tc, then `seed ≈
oracle`, and past parity hunt configs that *beat* the oracle. There is no "done" that ends the run;
the DoD just guarantees there's always a banked win behind you while you keep going (any later
improvement re-banks a fresh champion). The run ends only when the human stops it or a hard external
block hits (§6.0) — never because "the deliverable is met."

### ★ THE PER-ITERATION LOOP — this is the engine; you live here ★
Steps 0–4 above just decide *what to feed in* (which shape, which answer key, which bar). **This
loop is the actual work** — run it for every single change, no exceptions, hundreds of times, until
the deliverable is met. One turn = pick the worst un-finished shape → through these steps → bank
→ immediately start the next turn (never pause to narrate; §6.0). When in doubt, **take another
turn.**

> **Priority guard (read before every turn):** the bar is the **floor** (`G ≥ 0.75`), not beat tc and
> not oracle parity. Pick the **worst below-floor shape first** (lowest `G`; any shape under 0.75 is a
> disaster and outranks everything); with no disasters left, pick the lowest-`G` shapes in any
> kernel whose geomean (at any dtype) is under 0.85 until it clears. The oracle is your *reachability test +
> answer key* (Step 2) — reach for it freely when reasoning is struggling or you want an answer key, just
> not reflexively on every gap before you've reasoned. If a shape is **already above floor**, do NOT keep tuning it
> toward beating tc / oracle parity while another shape is still below floor — that's Step-4 overtime,
> and it waits. Spending a turn on overtime while a disaster shape remains is the #1 misexecution.
> Once NO shape is below floor and every (kernel, dtype) geomean ≥ 0.85 (the regime the run lives in
> most), "pick the worst below-floor shape" no longer applies — in overtime pick the shape with the
> **largest gap to its next rung** (largest tc-gap first toward `G ≥ 1`, then largest oracle-gap toward
> parity), refreshing the oracle per Step 4. A reappearing disaster preempts everything.

1. Read the generated Triton on both arms (**and the oracle answer-key diff if you have one**) → form a **workload-property hypothesis** (never kernel-identity, never a
   dtype/identity special-case in disguise). Keep this step IN the worker — it's cheap and uses your
   intuition. But **offload the context-heavy investigation that feeds it** (a Triton dump, ncu
   output, a full oracle log, a 9-kernel field-diff) to an ephemeral code/perf-investigator (§6.2)
   that returns only the distilled finding — the raw dump must never land in your context. (STUCK on
   a hard-pile/counter-intuitive item? fan out N idea-generators from the same evidence for
   *diversity* — a stuck-state escalation, not a per-turn step.)
2. Edit the fact/heuristic.
3. **Correctness-gate** vs an eager/reference baseline (you bypass the autotuner's accuracy check).
4. **Matched-lever A/B (attribution, not target-selection).** From the config you're shipping (the
   simplest confirmed-faster floor-clearer — a simplification of the oracle's winner, or your hypothesis
   candidate in the no-oracle path), revert **one** field to the seed value and measure the delta;
   **perturb DOWN, never UP from the seed.** Only ever bench a *valid* config the search could emit
   (never a seed-bits + chosen-bits Frankenstein — the "1174µs w32" mirage: `[block=1, w32]` is a config
   the search never visited, so its latency tells you nothing). Couplings are real: when neither field
   alone gains but the *pair* does, move both and validate the bundle as a unit; catch **redundant
   substitutes** (reverting either alone looks free, reverting both tanks it) by re-benching with the
   'free-looking' fields dropped *together*. In the no-oracle path you changed only a field or two, so
   this A/B is light. (Full rationale: §5 + the §3 no-regression backstop.)
4b. **Regression-referee (disaster-avoidance MUST, Gate R).** config-recorder over the full active
   matrix → re-bench every changed cell to its floor → reject the edit if any *realistic* shape fell
   below floor (the referee owns the realistic↔diabolical verdict; it subsumes the flip-axis sweep).
   Rules: the §3 no-regression backstop.
5. **Adversarially verify the win is real** (§5 — Gate A; plus the Fact-gate (Gate D) if the edit
   touched a fact, Gate F if the win is counter-intuitive). Kill it on majority-refute. Gate A reuses
   the focal cell from step 4b's Regression-referee sweep as its authoritative re-bench (no duplicate
   bench); the skeptics analyze those numbers + one independently-authored own-script reproduction
   (a fresh agent writes it, the driver runs it serially).
6. **Generality gate — does this lever belong in the core?** (§5 Gate H, a MUST on every lever.) Even
   a real, reproduced win (it passed step 5) can be **DEFERred or REJECTed** here: adjudicate **KEEP /
   DEFER / REJECT / BORDERLINE** by *key-faithfulness × magnitude × realism × downside × complexity*.
   **KEEP** → bank it (commit + record, steps 7–8); **DEFER** (faithful but not ready) → log it to the
   removed-heuristics-log with a re-add recipe, don't ship it; **REJECT** (unfaithful key —
   identity / bare-dtype / op-pattern proxy) → invoke **try-harder** (re-key mode) to re-key it on a
   workload property; **BORDERLINE** → record the tradeoff for the human's later judgement and provisionally DEFER
   (keep climbing — never ask the human, §6.0). This is the overfit firewall *per lever*: a lever that only
   wins by fencing the curriculum dies here even though step 5 said the number was real.
7. **Commit** the green change to your branch (the immutable SHA the ledger entry will cite).
8. **Record + update state.** Notebook + ledger — **wins AND rejections** (a rejected fix/hypothesis is
   first-class data: one compact line — *what you tried, why it failed, the evidence pointer*: "raising
   welford's normalize cap 2048→4096 → regressed ~7.3× at (262144,5120)", not "bigger cap bad"). Record
   gate FAILs/REJECTs as-returned (never laundered) — this is what stops a re-invoked context from
   re-deriving a dead idea (§6.1), and the rejected pile is where the hard-pile's *common thread*
   surfaces (§3). **Update the per-shape status table** (new G for the edited shape + every cell the
   Regression-referee re-benched) and write the **exact next action** so START-HERE resumes without
   re-deriving it. Record the compact lesson, not the full A/B transcript.

### The no-regression backstop (the part that makes multi-target safe)
A gain on one shape that **silently** drops a *realistic* shape below its floor is **rejected**. A config-flipping
cap's A/B **must sweep the workload axis it flips on, at MID and EXTREME values** — never endpoints
or hand-picked shapes (a 3-shape A/B once passed an auditor but a broad re-run caught a ~7.3× valley
at an untested in-curriculum shape). **For a multi-target run (e.g. several dtypes), "no regression"
spans the entire active matrix:** every edit must hold the floor on *every realistic* dtype/shape
already in play, not just the one you're currently chasing. **The floor binds *realistic* shapes only — and *realistic* means a *real workload* (a shape that
occurs in actual models), which is INDEPENDENT of curriculum membership.** Curriculum vs. real-workload
are two different axes: in-curriculum shapes are realistic real workloads, **and a realistic shape
*outside* the curriculum still binds the floor** (the heuristic must generalize to it — that's the
whole point). *Unrealistic* = genuinely synthetic/diabolical (not a real workload — e.g. 2 rows of 16M
elems, or a tight box around a single shape), **wherever it appears (in- or out-of-curriculum)** —
only these may regress, even below floor (the realistic↔diabolical line is the **Regression-referee's** to
call — Gate R, §5; Gate H reads that verdict). *When in
doubt, treat a shape as realistic* — the misses below were all realistic shapes; never launder a
genuine regression as "unrealistic." (So wherever this method says "every shape ≥ floor," read it as
every *realistic* shape.) **During the climb you discharge this by
MEASURING** — re-bench the flip-axis sweep and the previously-banked shapes the edit could plausibly
touch, and **DERIVE that affected set mechanically rather than guess it:** run the config-recorder
(`_lab/harness/config_recorder.py`, §5) over the FULL active matrix (every kernel × shape × dtype ×
split incl. robustness) BEFORE and AFTER the edit and diff the normalized configs — the cells that
*changed* ARE the no-regression sweep; byte-identical cells are provably perf-invariant (codegen is
deterministic in config+source) and need no re-bench. Every backstop miss here was a *changed cell at
a shape/dtype the worker never thought to sweep* (the ~7.3× valley above; CE fp32 +24% — a dtype not
swept; the D4 occ corner) — exactly what a full-matrix diff surfaces up front. THREE rules keep the
skip sound: **changed ≠ win** (a flagged cell still earns the full A/B + gates — the D4 corner was a
flagged cell wrongly assumed a win); **full matrix or it's a false all-clear** (valid only if the
diff spans every active dtype + robustness — an fp32-only diff green-lit the bf16/robustness valleys);
**selection-only** (config-identity ⇒ perf-identity ONLY when the edit changed *which* config is
emitted — an edit touching kernel source / a fact builder / normalize / a lowering needs the
generated-Triton diff instead: `--triton`). A climbing edit *should* change some configs — that is not
a failure, it is the worklist.

**A small, principled net-positive trade IS allowed** (not every edit must be Pareto-clean). Accept
a regression on some shape(s) only when ALL of these hold — otherwise reject:
1. **Bounded — and never below the disaster floor:** no single shape regresses more than **~10%**,
   **and no *realistic* shape may end below its floor** (`G = 0.75` vs tc, or `0.75 × oracle` if codegen-bound)
   regardless of how small the regression is. A "net win" average can hide a catastrophic valley
   (the EDIT#4 lesson); the absolute floor is the backstop against that. The relaxation is for ~5%
   nicks, not gambles.
2. **Net positive over the FULL affected set:** more shapes improve than regress, by more than they
   regress, measured across the *complete* flip-axis sweep + every banked shape the edit could touch
   — never a sample of winners. You cannot claim "net win" on shapes you didn't measure.
3. **Principled, not arithmetic:** there is a *workload-property reason* the trade exists (a cap that
   genuinely helps the common regime forces a few edge shapes to share it). "The numbers happen to
   net out" is **not** a reason — that's just p-hacking the curriculum (the geomean trap that caused
   run 2's false victory). **Having a per-kernel geomean *target* (0.85) does not license arithmetic trades** —
   the target is a floor to clear, never a budget to spend by sacrificing shapes; every trade still
   needs a property reason *and* must respect the per-shape floor (rule 1).
4. **No separator first:** before accepting the trade, fire **anti-giving-up** — is there a faithful
   fact that *separates* the conflicting shapes so you could get BOTH? A trade is the answer only
   when no such separator exists (and the only alternative would be a kernel-identity fence). "They
   just conflict, take the net win" is a stop-claim that must survive the gate.

This relaxes **edit-acceptance only — NOT the bankable bar.** The bar is **every shape ≥ its floor
(`0.75 × tc`, or `0.75 × oracle` if codegen-bound) AND every (kernel, dtype) geomean ≥ 0.85** — the
per-shape floor and the per-kernel geomean together, never an aggregate over kernels. The floor is
precisely what stops a kernel's geomean from masking a lagging shape inside it, so "this kernel's
geomean is fine" is never a pass while any of its shapes sits below floor. Log every accepted trade
with its per-shape deltas + the principle, so the loss is visible, not buried.

---

## 4. Benchmarking footguns (these dominate the numbers)

1. **FORWARD ONLY — never build a grad graph for either arm.** Build inputs `requires_grad=False`;
   no `.backward()`. The biggest artifact seen was timing the **autograd wrapper**
   (`*.apply`/`save_for_backward`), which adds ~9–18 µs fixed host overhead and flipped one kernel
   1.056→0.79. Time the **bare forward** (`helion.kernel(fwd.fn, config=seed)`) for *both* Helion
   and the tc reference.
2. **Reset dynamo per shape:** `torch._dynamo.reset()` before each compile, else tc caches multiple
   shapes and recompiles into slower **dynamic-shapes** mode (unfair to tc).
3. **Prefer tritonbench** — it already handles the footguns (it calls `attr.reset()` + dynamo reset
   per input for free). Drive the seeded arm by promoting the seed to `default_config`
   (`promote_seed_to_default=True` via an env flag) at `HELION_AUTOTUNE_EFFORT=none`, and run it
   through tritonbench's own `do_bench`/accuracy/tc-baseline. Three arms: **(a)** helion-seeded (the
   work), **(b)** helion-default (heuristics disabled — the unseeded control), **(c)**
   torch.compile-default (NOT max-autotune). Hand-rolled scripts are *cross-checks*, not the
   headline.
4. **Single-process head-to-head.** Cross-process `do_bench` jitter is ~5–10% on small kernels and
   swamps the seed effect. Time all arms **in one process on the same input tensors**, median-of-N
   (`sorted(do_bench(fn, return_mode='median') for _ in range(N))[N//2]`, N≈9–15). If arms must run
   in separate processes, report the within-process tc-anchored **lift**
   `(s_tc/s_hl)/(d_tc/d_hl)`, never raw cross-process latency.
5. **Contention guard (only if the GPU is shared this run — see `local-setup.md`).** When shared,
   parse `nvidia-smi` compute-apps before/around each timing; if foreign GPU mem > ~300 MiB, mark
   contaminated and retry / wait-idle (headline runs need foreign mem ≈ 0).
6. **Accuracy gate before timing**, vs the eager reference *built at the same dtype*, upcasting both
   to fp32 before `allclose`. Tolerance is task-specific (see the task file) but the rule is
   universal: **any tolerance change is logged with measured justification, never silent.**
7. **Verify the config actually ran:** record the *normalized* running config after
   `bound.ensure_config_exists(args)` to prove `configs=[cfg]` (no autotune) executed the config you
   intended.
8. **Memory hygiene** between multi-GB shapes (`del ...; torch.cuda.empty_cache()`), and
   **incremental JSON checkpointing** so a foreground kill never loses completed rows.

---

## 5. Gate disciplines (keep these even solo)

Independence is the asset — a gate that develops rapport stops re-examining. Under ultracode these
run as **fresh-context fan-out agents by default** (a solo self-check keeps none of the independence —
reserve it only for the explicit fast-path on a trivial within-noise tune); keep the discipline either
way. **For the high-stakes
adversarial gates (adversarial-verify [+absorbed independent reproduction], anti-giving-up, the
fact-gate [doctrine+faithfulness], generality, overfit/TEST-firewall, regression-referee, mechanism),
use the VERBATIM prompts in `gate-prompts.md` (gates A, B, D, E, F, H, R + the try-harder /
completeness-critic helper frames) — do not improvise
their wording** (how you phrase an adversarial ask is how you bias its verdict; the scripted frames
bake in default-to-refuted, record-verdict-first, pin-the-SHA, and never-hand-over-your-conclusion).
Fill the slots, don't edit the frame. **The five MUSTS are anti-over-claim (Gate A), anti-under-claim
(Gate B), the portfolio guard (overfit/TEST-firewall, Gate E), the per-lever generality gate (Gate H),
and disaster-avoidance (the regression-referee, Gate R);** the rest fire per their trigger.

**Fan-out disposition (ultracode; GPU re-measurement stays serial per §6):** N-refuter panels — Gate A,
the Fact-gate (D). Perspective-diverse (one verifier per axis, then synthesize) — Gate H and Gate E's
periodic audit. Judge-panel — Gate F, the hard-pile common-thread, the beat-the-oracle hunt. Solo/serial
— the GPU re-measurement inside Gate A and Gate R (fan out only the analysis around it). **Sole reader,
never fanned out — Gate E's FREEZE TEST read** (one reader, one TEST bench; fanning it out breaks the
firewall).
- **Correctness-first**, every iteration (you bypass the autotuner's accuracy check).
- **Matched-lever A/B** against the config you *chose to target* (§3's simplest floor-clearer — **not** the oracle's peak), perturbing **down** from it, never **up** from the seed (never pin block=1).
- **Regression-referee (Gate R, a MUST)** — on every edit that changed ≥1 cell, an independent referee
  runs the config-recorder over the FULL active matrix and re-benches every changed cell to its floor
  (this subsumes the old mid+extreme flip-axis sweep), defaulting to "a regression is hiding"; it
  **owns the realistic↔diabolical verdict**. Disaster-avoidance as a gate, not a self-check.
- **Adversarial verify** each claimed win: N independent skeptics prompted to *refute* it (default
  to "refuted" under uncertainty); kill on majority-refute. Hunt: noise/fabrication, measuring the
  wrong thing, overfit (check held-out val + an off-focus kernel), **kernel-identity smuggling** (a
  constant fencing exactly one kernel's shapes), metric gaming (loosened tolerance, dropped shapes).
  The independent own-script reproduction (the absorbed results-referee) is **authored by a fresh agent
  and run by the driver serially**; all N analytical skeptics share ONE authoritative re-bench — the
  focal cell from Gate R's step-4b sweep, not a fresh duplicate — never N concurrent benches (§6 invariant).
- **Fact-gate (doctrine + faithfulness, Gate D):** on any new/changed fact field a heuristic reads
  (or a new fact, or a threshold a branch compares a fact against), ONE spawn checks BOTH parts.
  *Part 1 — doctrine:* **derived facts never walk the graph** (no `device_ir.graphs`/`node.users`/
  `_classify_load_dataflow`); the walk lives on a **walker fact** (`MemoryOpFact`/`AccumulatorFact`,
  computed once — prefer a field on an existing one over a new fact); the walker field is
  **consumer-agnostic** (the different-consumer test); sound provenance (the lone tolerated unsoundness
  is the pre-existing eviction-index TODO — to fix, not copy). *Part 2 — faithfulness:* the
  **divergence test** on the fact (construct a kernel where the lazy proxy and real property *disagree*
  — this falsified `num_load` and `num_reduction_ops`; a fact no branch reads is cut) AND on the
  **threshold** (a `fact <= K` that splits the value-set along dtype/kernel lines is a disguised fence;
  keep a dtype/identity-correlated quantity a FACTOR inside a byte/occupancy budget, never the operand
  of a literal). Doctrine in §2.
- **Generality (per-lever KEEP / DEFER / REJECT / BORDERLINE — a MUST):** fires on **every** proposed or
  existing lever/branch/constant before it enters the core, adjudicating the lever's *place* by
  **key-faithfulness × magnitude × realism × downside × complexity**, judged purely by the gate's own
  embedded rules (the maintainer's generalizability standard encoded in the gate — NOT the maintainer's
  individual keep/defer calls). Hard line: an **unfaithful key** (kernel
  identity, a bare dtype literal like `itemsize == 2`, an op-pattern used as a kernel-class proxy) is
  **never bought by magnitude** — REJECT + spawn a try-harder agent to re-key on a workload property.
  A below-floor rescue outranks overtime win-chasing; faithful-but-not-ready ⇒ DEFER (→
  removed-heuristics-log, re-add retained); a genuine conflict ⇒ BORDERLINE — record the tradeoff for
  the human's later judgement + provisionally DEFER and keep climbing, never reflexive DEFER and never
  ask the human mid-run (§6.0). Distinct from the portfolio overfit guard below (whole-heuristic) — this is one lever at a
  time; walk-location/field-generality is the Fact-gate (Gate D) above.
- **Anti-giving-up:** any ceiling/noise/stuck/done claim must survive a *fresh* oracle — **and it fires
  on the ACTION of hard-piling a shape as STUCK or moving off a still-below-floor shape** (a never-stop
  worker won't narrate a stop, so the action is the trigger; B must clear the park as non-premature). A
  "source ceiling" is valid only if the oracle **can't beat tc** AND the oracle run is verified real (not
  truncated/OOM). Before declining because "a gate also regresses peer X," check X's actual code branch
  (it may be *structurally* excluded) and measure **every** firing shape.
- **Overfit guard + TEST-firewall (a MUST — the *portfolio* gate):** the per-claim gates miss the
  failure that *looks done* — the heuristic silently memorizing the curriculum (run 2's false
  geomean victory). Periodically audit the whole heuristic for constants that fence exactly the
  curriculum's shapes, and report the train↔held-out gap as a first-class number. This gate is the
  **sole reader of TEST, exactly once, at freeze** — nothing else may bench it.
- **Mechanism (surprise-triggered only):** a win can be real + reproduced yet *mis-understood*, so
  the generalized rule misfires off-curriculum. On a **counter-intuitive** win only (e.g. narrow-N
  wanting fewer warps), require a hardware-level mechanism (ncu/IR) that predicts the boundary where
  it reverses — else don't bank it as a rule, only a shape-scoped observation. Not fired per-edit.
- **A non-verdict is never a verdict:** a watchdog stall or API error after the analysis but before
  the verdict is recorded → re-fire fresh; never bank or fail on it.

(The compile-time config-recorder "behavior oracle" is a TOOL — not itself a gate — used in two places.
**(1) Inside Gate R** (the Regression-referee, §5 — the pass/fail MUST that replaced the old worker
self-check), running the no-regression backstop (§3): BEFORE/AFTER an edit, diff the
normalized configs over the FULL active matrix to *derive* which cells changed — bench only those;
byte-identical cells are perf-invariant and skipped (sound only over all dtypes + robustness;
changed ≠ win; a codegen/source edit needs the `--triton` generated-Triton diff). A climbing edit
*should* change some configs — that's the worklist, not a failure. **(2) A plain post-climb self-check** (still not a gate) that a
*cosmetic/refactor* edit left every emitted config byte-identical, so frozen perf verdicts transfer.
Default bar = ZERO config diffs; but if the human pre-approves specific expected config changes, honor
that — treat those as allowed and verify only the rest are byte-identical. The generalized
dtype/robustness/sibling recorder is `_lab/harness/config_recorder.py` (`record` +
`diff`, `--triton` for non-selection-only edits); the fp32-only `run3_task1_seed_configs.py` stays the
deliverable config-oracle replayed to MEASURE the win. The task file names the recorder script.)

---

## 6. Orchestration, persistence, and NEVER STOPPING

> **THE FAN-OUT / SERIAL-GPU INVARIANT (read first).** *Cognition fans out; GPU measurement never
> does.* Reading, analysis, field-diffs, idea-generation, and the analytical/refutation part of any
> gate are fan-out workflows **by default** (token cost is not a constraint). Any phase that **runs the
> GPU** — an A/B bench, the oracle / quick-autotune, any gate step that re-times latency — is
> **foreground, one job at a time**, never fanned out / backgrounded / detached. A gate with both an
> analysis part and a re-measure part **splits**. A workflow that launches two GPU jobs at once is
> **malformed — serialize it.**

### 6.0 NEVER STOP — this is the cardinal rule
**You are expected to run continuously, including unattended overnight, for as long as the human
leaves you running. Do NOT stop, pause, wind down, or hand back to the human — not at a checkpoint,
not when the task's definition-of-done is met (that's a milestone to bank, then climb past), not
ever short of a hard external block.** Read this literally:

- **There is no such thing as a good place to stop early.** "I've made good progress / this is a
  natural checkpoint / I'll await your confirmation before continuing / let me know how you'd like
  to proceed" are all **forbidden** mid-task. The human is asleep. Stopping to ask wastes the entire
  unattended window. If you find yourself composing a "should I continue?" message — **the answer is
  always yes; don't send it, just continue.**
- **Every claimed dead-end is the trigger for the next action, never an exit.** "No clean rule /
  noisy / stuck / at a ceiling / done / deliverable met / nothing left to try / blocked" → fire the
  anti-giving-up gate (§5), run a fresh oracle, attack from a different workload property, work the
  deferred hard-pile, push a floor-clear shape toward beating tc / oracle parity, or run the completeness critic.
  There is *always* a next move; the gates exist to find it. (The **only** real end is the human
  stopping the run or a hard external block like the GPU physically unavailable with no
  non-GPU-work left — and even then you keep doing the non-GPU work.)
- **Keep yourself alive mechanically.** Long oracles run foreground under the turn (§1). Never end
  your turn voluntarily with work outstanding; after every banked step, immediately pick up the next
  worklist item — don't return to narrate. (The one sanctioned voluntary handoff is a **proactive
  context-recycle at a banked checkpoint** — §6.1 — where no work is outstanding and a fresh
  re-invocation resumes immediately from the log; that is a clean handoff, not a stop.)
- **NEVER ask the human a question.** The human is asleep / not watching — a question doesn't pause
  the clock, it *ends* the unattended window with zero progress until they return. There is no
  clarifying question, no "which would you prefer?", no "is this OK?", no approval request. Resolve
  every choice yourself from this doc + the task file + the gated log; when genuinely ambiguous, pick
  the most defensible option, **log the decision and your reasoning** so the human can review it
  later, and keep going. Even a destructive/irreversible action you're unsure about is **skipped and
  logged for human review**, never turned into a question — route around it and keep climbing on
  everything else. The *only* thing that legitimately surfaces to the human is a hard external block
  with no remaining work (§-bullet above), and that's a final state, not a question.

**Meeting the task's definition-of-done is NOT a stop.** It's a milestone: freeze and bank it
(commit the champion + write the report) so a valid deliverable is locked in, then **keep climbing**
into beat-the-oracle overtime (§-Step 4). The DoD guarantees a banked win behind you; it never ends
the run. The run ends only when the human stops it or a hard external block hits. "Never stop"
therefore means exactly that — past the DoD too; the rule's whole job is to keep you climbing rather
than quitting on a good geomean, a premature ceiling, an "I'll check in," *or* a met deliverable.

### 6.1 The gated LOG is the source of truth — trust it OVER your own context
As a run goes long, your live context fills with stale intermediate reasoning, abandoned
hypotheses, and noise. **Do not trust your own memory of "what we found" over the written record.**
The notebook + ledger are not a backup of your context — they are a **higher-quality, more
trustworthy artifact than your context**, and here is *why*:

- Every result in them has **passed the adversarial gates** (§5) — independently reproduced + attacked
  by the adversarial-verify skeptics (Gate A), fact-checked for proxies (the Fact-gate), regression-swept
  (Gate R). Your live context contains the
  *un-gated* version (including the wins that got *refuted* and should never be acted on again).
- They are **compactified**: the distilled decision + the empirical *why* + the per-shape status,
  not the rambling path that produced it. They hold *more useful information per token* than your
  context does.
- They are **stable**: a banked verdict on an immutable SHA doesn't drift; your recollection does.

**Operating rule: before acting, re-read the relevant ledger/notebook entry and treat it as
authoritative. If your context disagrees with the gated log, your context is wrong — defer to the
log.** Write each banked result *as the gate returned it* (you cannot launder a FAIL by
re-narrating). A fresh re-invocation reading the log resumes **better-informed than your bloated
self would be** — which is why context limits are not a threat.

**Recycle proactively — don't wait for the limit.** Because re-invocation is lossless (the log is the
real state), treat a fresh context as routine hygiene, not just crash-recovery: at a **banked checkpoint
with no work outstanding** (every ~N banked levers, or whenever your context crosses a comfortable
threshold), write the resume-state (step 8's per-shape status table + the exact-next-action pointer),
then hand off to a fresh re-invocation that resumes from the log (START-HERE). This is the **only** thing
that bounds steady context growth *and* clears the subtle in-context bias that "defer to the log" does
not catch — the log holds the *facts*, but your per-turn *reasoning* (hypothesis-forming, reading the
diff, the A/B) still accretes in context and can quietly skew it. A recycle at a banked point is **not a
stop** (§6.0): no work is outstanding and the run continues in the fresh context — a clean handoff, the
proactive form of "context limits are not a threat." Prefer a deliberate recycle over drifting until
forced (or until auto-compaction silently drops nuance).

### 6.2 Machinery — fan out by default (ultracode)
GPU timing is foreground-serial on one GPU (the §6 invariant); everything else is fan-out-able. Under
ultracode the **default** is: the driver AUTHORS AND RUNS a fan-out Workflow for every substantial
NON-GPU phase — the seed-vs-oracle field-diff across all kernels, idea-generation, adversarial
verification, the analytical part of every gate, completeness / hard-pile sweeps — token cost is not a
constraint; optimize for the most exhaustive correct result. Multi-phase work = several workflows in
sequence with the driver staying in the loop between them; **dispatching a workflow is NEVER a stop**,
and the driver stays the single thread of accountability that picks the next worklist item.

**Three guardrails (safety, not cost):**
1. **GPU measurement stays foreground-serial.** A gate's N skeptics fan out **GPU-free** over ONE
   authoritative re-bench (for a per-edit gate, the focal cell from Gate R's step-4b sweep — not a
   duplicate); the one independent own-script reproduction is **authored by a fresh agent and run by the
   driver serially**. The parallel skeptics never bench — that would fan out the GPU. Reserve extra
   per-claim re-measurement (still driver-run, serial) for high-stakes claims (a fact changed, a
   counter-intuitive direction, a claimed beat-tc).
2. **No detached/background GPU PROCESS** — the silently-dying class (§1's 13h stall). This is DISTINCT
   from in-turn Workflow fan-out, which is foreground sub-orchestration whose results return to the
   driver; that is encouraged, not the dying class.
3. **Anti-laundering:** every subagent writes its verdict to the ledger AS-RETURNED; the driver
   integrates from the WRITTEN record and never relays a sub-verdict in its own words. On a PASS the
   driver reads only `{verdict, ledger-ref}` (not the full object — context hygiene).

**Stakes-based GPU triage (the serial GPU is the scarce resource — spend it deliberately; token cost is
free, GPU wall-clock is not):**
- **Fast path** — a within-noise tune of an existing faithful lever (no fact change, no direction
  surprise): the matched-lever A/B (step 4) + the Regression-referee (Gate R), whose sweep IS the
  authoritative re-bench; skip the heavy stack (Gate A/D/F/H).
- **Full stack** — a new/changed fact, a counter-intuitive win, a claimed beat-tc, or a borderline
  lever: fire the relevant gates (Fact-gate, F, H, …) with their fan-out analysis.

**The driver + ephemeral helpers** (all non-GPU fan-outs; the dump never lands in the driver — only the
distilled finding does): **code-investigator** ("where/how does X work?" — reads source/IR),
**perf-investigator** ("why is A faster than B?" via ncu / generated-Triton — returns the mechanism,
not the dump), **try-harder** (the general "escalate when stuck" agent — hypothesis / re-key / approach
modes; gate-prompts.md), and the **completeness-critic** (loop-until-dry). They are ephemeral and
fresh-context: no message-relaying, no acceptance-laundering, no surviving past their one task.

**The notebook + ledger are the standing artifacts** (never an agent), and the resume-after-death
source of truth (§6.1). Required notebook sections: CURRENT HEURISTIC STATE / PER-SHAPE STATUS TABLE /
BANKED WINS / TRIED-AND-REJECTED / DEFERRED-HARD-PILE-AND-BORDERLINE / NEXT ACTION / **HUMAN-REVIEW
QUEUE** (append-only, ranked, deduped — every BORDERLINE, every STUCK hard-pile entry, and every
skipped irreversible action writes ONE line: {what, why-blocked / the tradeoff, the provisional
decision, where to look to reverse it} — the orchestration counterpart to "never ask mid-run"). The
ledger is append-only gate verdict objects AS-RETURNED keyed by {SHA, gate, claim}.

The completeness-critic runs on a **cadence** (every K levers, whenever the below-floor worklist
empties, and before any dead-end is accepted — NOT just "near the end", which is undefined for a
never-stopping run) and **loop-until-dry** (re-run until K consecutive passes find nothing new):
*what's missing — a dtype not swept, a claim unverified, a cap not audited, a shape under the noise
floor, a deferred hard-pile item never revisited?* Each gap is appended to the worklist and cleared or
explicitly logged-and-skipped. **Never silently cap coverage** (top-N, no-retry, sampling) without
logging what was dropped — silent truncation reads as "covered everything" when it didn't. Frame:
gate-prompts.md.
