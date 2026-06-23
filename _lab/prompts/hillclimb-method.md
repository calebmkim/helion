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
curriculum of shapes via a **faithful, general** heuristic, proven rigorously. The priority is lexical
(§3): **faithfulness > generality > performance**. The task file says which heuristic and what "good"
means this time.

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

**Faithfulness has THREE surfaces, and the populator is the one most often missed.** A fact is faithful
only if all three are: (1) the field's **meaning** is a real workload property; (2) the **branch key**
that reads it is faithful (not an identity/dtype fence); and (3) — the easy one to miss — **how the field
is POPULATED** (the `device_ir.py` builder, or any inline derivation a branch keys on) actually computes
that property, rather than a *lucky proxy* that merely correlates with it on the curriculum. The real
example: a derivation that filters `block_sizes` on `bs.reduction` to mean "a tile reduced over dim-0" —
but `bs.reduction` actually means "allocated via the reduction-dim allocator," which is *also* True for an
`hl.zeros([m,n])` accumulator that is never reduced. It gets the right config on every in-curriculum
kernel by luck. **An unfaithful populator is a GENERALITY bug, not merely a wrong number:** the population
decides *which kernels the heuristic fires on*, so a lucky proxy silently mis-fires (or stays silent) on
the first kernel where the proxy and the real property diverge. Derive each field *directly* from the
faithful provenance the compiler already builds (e.g. the reduction-axis / store-extent provenance), not
from whatever existing flag happens to line up. The Fact-gate (Gate D, §5) now adversarially verifies the
populator — by **authoring a divergence kernel** (mutate a template so proxy and property disagree,
compile, dump the facts), not only the field's meaning.

**A lucky proxy and a best-guess default sit at OPPOSITE ends of the generality axis — don't conflate
them.** A lucky proxy is *minimum* generality: a correlate that works only because it got (accidentally)
fitted to the curriculum and silently breaks the moment you leave it. A best-guess default is *maximum*
generality: a value you assign *on purpose* for an out-of-curriculum workload you still want to fire on,
knowing it may be sub-optimal there. The divergence test separates them — the proxy *secretly* diverges
from the property it claims to compute (you didn't know, and a branch trusts it as exact); the best-guess
*honestly* diverges from optimal off-curriculum (you knew, and you recorded it). Banning the proxy is NOT
a license to refuse to generalize — see 'Prefer to fire' below.

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

The fact-gate (§5; Gate D in `gate-prompts.md`) enforces all of the above whenever a heuristic reads a
new or changed fact field or changes how a field is populated (Part 3 — the populator must compute
the property directly, not via a curriculum-lucky proxy).

Two products the seed serves:
- **Product A** — *skip autotuning*: `configs=[seed]`, the seed IS the config. This **bypasses the
  autotuner's own accuracy check**, so you must correctness-gate it yourself.
- **Product B** — *seed the search*: the seed is the autotuner's initial population, so a full
  search converges in less wall-clock. **Secondary for now** — focusing on A almost certainly helps B.

**Prefer to FIRE — the bad default is the enemy.** The compiler's existing defaults are *really* bad, and
a seed is never forced (above): for Product B a wrong seed only costs autotuning time, and for Product A
you correctness-gate it yourself — so a fired best-guess is cheap while a *declined* heuristic is
expensive, because it drops the kernel back to that terrible default. So **over-fire rather than
under-fire.** Two narrow gates cause under-firing, and BOTH should open: (1) the **populator** that guards
fact *creation* so tightly an unforeseen workload yields no fact at all — build/record the fact anyway,
even for inputs outside the tested regime (the populator only *records the property*; it never invents a
config); and (2) the **heuristic** (`get_seed_config`/`is_eligible`) returning `None` when the facts land
in a regime it has no tuned branch for — return a **best-guess config** (a sensible default block size,
etc.) and fire. It may do badly on that unforeseen kernel — that is FINE; **record it** (ledger /
human-review queue) so a later curriculum extension or review can catch the guess if it turns out actively
harmful. This loosens NONE of the rules: the **key** stays a faithful workload property (never kernel
identity), and walker facts still record genuine provenance, never a guess (above) — the *guess lives in
the heuristic's config choice*, not in the recorded fact. Generalizing your best guess onto a kernel the
curriculum never showed you, and firing, beats the really-bad default — this is the firing-side corollary
of the GENERALITY priority (§3) and is enforced by Gate H's BROADEN verdict (§5).

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
   *cross*-process jitter — it doesn't apply to a same-process cross-check). **Also plausibility-check
   that the device metric is COLD-L2:** for a ≤L2 shape the implied bandwidth must be well under HBM
   peak, not a 3–5 TB/s L2-hot artifact — a cudagraph harness and a cudagraph cross-check will AGREE on
   the same fake number, so >3%-agreement alone is NOT sufficient (§4 #9).
3. **Seed mechanism proven:** `configs=[seed]` runs *no* autotune and uses the normalized config.
4. **Notebook + ledger created** (or the prior ones loaded — see START-HERE).

That's it. **Setup is a gate, not a phase.** Once it's green, you are in the loop for the rest of
the run.

---

**The priority order — what you optimize, lexically (this is the whole reframe).** The hard part is
knowing *what to chase and what you may never trade away*. The ordering is **lexical, not a weighted
sum** — a lower priority can never buy its way past a higher one:
1. **FAITHFULNESS — never traded.** Every fact field's *meaning*, every fact field's *population*
   (how `device_ir.py`'s builders fill it), and every *branch key* must be a faithful workload
   property — not a curriculum-lucky proxy, not a kernel/dtype-identity fence. **A faithful win you
   chased for hours beats an unfaithful win you got in five minutes** — the unfaithful one is *worse
   than nothing*, because it silently fails to generalize and you will trust it anyway. (Enforced hard
   by the F-gates: the Fact-gate D — field + **population** + threshold; Gate A's identity-smuggling axis.)
2. **GENERALITY — never traded.** A lever applies as widely as its mechanism allows; **a narrow scope
   needs a reason, not breadth.** This is NOT "narrow is bad" — a lever scoped to one kernel or one band
   is perfectly fine *when there is a well-documented, justified, and tested reason it applies only
   there* (a mechanism that genuinely reverses outside that scope). What's banned is a narrow scope with
   *no such reason* — a fence that exists only because that's where you happened to measure. Justified
   narrowness KEEPs; unjustified narrowness BROADENs. **Under-firing is the same failure, one level up:** a
   populator gate so narrow it won't even build the fact for an unforeseen workload — or a heuristic that
   returns `None` instead of a best-guess config — silently routes that kernel back to the (really bad)
   compiler default; that is an unjustified narrow scope, so it BROADENs too (fire with a recorded best
   guess rather than decline into the default — §2 'Prefer to fire'). (Enforced hard by the G-gates: Gate H
   per-lever incl. its BROADEN verdict, Gate E whole-heuristic, the refactor-critic.)
3. **PERFORMANCE — pursued relentlessly, but only through 1 + 2.** You never stop pushing perf (§6.0);
   you may **never** buy a perf number by sacrificing faithfulness or generality. Perf is the *goal*;
   F + G are the *constraints*, and they win every conflict. When stuck, the answer is to chug harder
   for a faithful/general path (Gate B), never to drop to a proxy or a fence.

**Definition — a "WEIRD" (or "odd") shape (used throughout):** a shape where **helion max-autotune (the
full, converged autotuner — `HELION_AUTOTUNE_EFFORT=full`) loses to torch.compile-default.** I.e.
helion's *own best achievable config* can't beat tc there, so tc is structurally out of reach for helion
on that shape. This is the one crisp test — not occupancy/aspect heuristics, not a guess. A shape is
"normal" until a converged max-autotune actually loses to tc (Step 2b confirms it via Gate B).

**The two perf measures, both `G = tc_latency / seed_latency`:**
1. a **per-shape disaster floor (the ONE hard perf bar):** no *realistic* shape below `G = 0.75` (seed
   never worse than ~1.33× tc) — **UNLESS the shape is "weird"** (above: even helion max-autotune can't
   beat tc), in which case the floor **retargets to `0.75 × helion-max-autotune`** (0.75× the best helion
   can structurally do) — lowered for that shape, *never* exempted. It is per-*shape* on purpose: it is
   the only thing that catches a single-shape cliff a kernel-geomean would average away (the
   non-monotonic softmax small-N landmine — softmax cliffs while rms_norm is flat at *identical* facts).
   This is the floor as an **acceptance bar** (you may never *introduce* or *leave* a realistic shape
   below it — Gate R), distinct from the worklist *ordering* (Step 1: worst **kernel** first, not worst
   shape).
2. a **per-(kernel, dtype) geomean — a DIAGNOSTIC, not a gate.** `geomean(G)` over one kernel's
   accuracy-passing rows (per active dtype; §4 #6) tells you *which kernel to investigate first*
   (Step 1). **The (kernel, dtype) cell is THE unit of account for the acceptance model below**
   (collapse-vs-anchor, net-progress, the generality license, the frozen anchor): every "kernel geomean"
   is computed and counted **per (kernel, dtype) cell — never blended across dtypes**, so a multi-dtype
   kernel contributes one cell per active dtype. A high one is good and you keep pushing it — but it is **not** a hard bar you may stop at,
   and **not** a budget you may spend by sacrificing shapes or generality. ~0.85 is a healthy number to
   aim every kernel at; it is a target you relentlessly pursue **through faithful/general means**, never
   a line that ends the work and never a reason to write a narrow rescue-if.

**Never settle, never cheat.** Clearing the floor and reaching a healthy geomean makes the deliverable
bankable; it is never a reason to stop (§6.0), and reaching it *unfaithfully* is a regression, not a win.

**The oracle (autotune) is a *tool you reach for*, not the lead.** Property-reasoning (Step 2a) is the
default first move; it usually clears a 0.75 floor on its own. Reach for the oracle when you're **stuck by
reasoning and want an answer key**, OR when you **suspect tc is codegen-unreachable** — either trigger
suffices (an answer key alone is a legit reason; just don't reflexively autotune *every* gap). Its two
roles (operationalized in Step 2b): a **reachability detector** and an **answer key**. **Bar reminder:**
even when the oracle beats tc, your goal is still only the floor then tc-parity — not matching the oracle
(chasing it past tc is Step-4 overtime). **Cost:** max-autotune is slow — **never sweep it**; reach for it
only on a shape **stuck below floor AND suspected codegen-unreachable**. The weird-shape *verdict* uses
**full max-autotune, not quick** (a quick parity/loss vs tc is suspect — softmax(1024,65536) flipped
1.000→1.338 quick→full), so one converged verdict beats a cheap one you'd re-confirm anyway.

### Step 1 — Establish the torch.compile yardstick + triage by worst KERNEL (cheap, always first)
Measure **seed vs torch.compile-default** across the curriculum (and, for a multi-target run, across
every active dtype). This is the headline: `G = tc_latency / seed_latency` (G≥1 = seed beats tc;
G=0.75 = seed is 1.33× tc's latency). Cheap (no autotune). Triage in this order:
- **Worklist ordering: the worst KERNEL first.** Rank (kernel, dtype) by geomean
  and attack the lowest one — **investigate *why* the whole kernel/family lags** and fix it *generally*
  (Step 2 → Step 3). A lagging *kernel* is the signal of a *systematic* problem worth a general fix; a
  lagging single *shape* tempts a narrow rescue-if AND tempts chasing near-unreachable weird shapes —
  both failure modes we are designing out. **The worst kernel almost always contains the most
  below-floor shapes anyway**, so worst-kernel-first naturally sweeps up the disasters without making
  any one shape the thing you drop everything to chase.
- **Low-hanging fruit inside that kernel is fair game.** While investigating a lagging kernel, if its
  worst/below-floor shapes can be lifted *faithfully and generally* with little effort, take the easy
  wins — the cheapest way to lift a geomean. (Easy ≠ narrow: an easy *faithful/general* fix is great; an
  easy *narrow fence* is the thing you must not write.)
- **The per-shape floor is an ACCEPTANCE bar, not the worklist driver.** `G < 0.75` (or `< 0.75 ×
  max-autotune` for a confirmed-weird shape, Step 2b) on a *realistic* shape is a **disaster** — but its
  force is at *acceptance*: Gate R will not let you bank an edit that **introduces or leaves** a realistic
  shape below floor (§3 acceptance model). It does **not** mean "drop everything and chase that shape" —
  that ordering is exactly what bred narrow rescue-ifs and weird-shape-chasing. A pre-existing disaster is
  a tracked row in the per-shape status table; it gets fixed because it drags its kernel into worst-kernel
  position (above), and the completeness-critic surfaces any disaster *masked* inside an otherwise-healthy
  kernel so it is never lost. Fix disasters via the worst-kernel mechanism, generally — not by fencing the
  one shape.
- **Geomean is a DIAGNOSTIC, not a finish line.** A healthy geomean everywhere means the deliverable is
  bankable, but never-stop (§6.0) means you keep pushing it higher — through faithful/general means.

Apply normal measurement-noise tolerance at the boundary — don't thrash a shape oscillating around
0.75; it's the clear sub-0.75 shapes that are disasters.

### Step 2 — Lift a below-floor shape (reason first; reach for the oracle when stuck or on suspicion of a hard limit)
A shape is below floor (`G < 0.75`), or it's one of the easy/worst shapes in the lagging kernel you're
working (Step 1). **First try to lift it WITHOUT the oracle** — that is the default path now, and it's
usually enough to clear a 0.75 floor:

**(a) Understand *why* it's slow, then form a workload-property hypothesis.** Read the generated
Triton on both arms and reason about the gap (block/tile sizes, num_warps, stages, eviction,
`pid_type`, persistent-vs-looped, loop chunk…):
- Helion: `Kernel.to_triton_code(config)` (or `HELION_PRINT_OUTPUT_CODE=1`).
- torch.compile: `TORCH_LOGS=output_code python ...`.
Many below-floor shapes are a recognizable, *general* mis-seed (wrong warps for the rnumel, a
persist/loop flip, a missing byte cap) you can fix from the property alone — no autotune needed. Climb it
(Step 3) and re-measure against the 0.75 floor. But if reasoning stalls, or an answer key would simply be
faster than reasoning blind, don't hesitate — pull the oracle (2b).

**(b) Reach for the oracle — triggers: stuck-by-reasoning OR suspected codegen-unreachable** (e.g. no
Helion primitive for tc's structure, like split-reduction; full rationale above). Run the autotune
oracle, which does two jobs at once:
- **oracle beats (or ties) tc** → the floor is reachable and tc was *not* a fundamental limit. Your
  **answer key** is the oracle's winning config — or, if that's complicated, the simplest
  config that still clears the floor (Step 3). Climb toward it (and, in overtime, to beat
  tc).
- **max-autotune can't beat tc → this is a "WEIRD" shape; retarget its floor.** Helion's codegen
  genuinely can't express tc's optimum here; tc is unreachable. This is a **weird shape** (def: §3).
  **Retarget its floor to `0.75 × helion-max-autotune`** (0.75× the best Helion can structurally do) and
  climb to it — the retarget only **lowers** the yardstick, it never exempts the shape, which stays on the
  worklist until `seed ≥ 0.75 × max-autotune` (real miss: jsd — the oracle was ~18% faster than the seed,
  yet waved off as "exempt" because *both* lose to tc). Valid only on a **converged** max-autotune
  genuinely losing to tc — route through anti-giving-up (Gate B, §5), never self-certify.

**Quick-autotune is a cheap *answer-key* escalation — but NOT the weird-shape verdict.** When you only
want a config to hill-climb toward (faster than full, and unlike tc it hands you a Helion config),
quick-autotune is fine: a quick *gap* is real (it only widens at full). But a quick *parity/win vs tc* is
**suspect** (softmax(1024,65536) flipped 1.000→1.338 between quick and full), so the **weird-shape floor
retarget always uses a converged max-autotune** (Step 2b) — never a quick verdict. Whatever autotune you
run, its **config is often WACKY** (an over-specific knob bundle). Do **not** emit the wacky config —
field-diff it against the seed, find the **1–2 knobs that carry most of the payoff**, and emit the
**simplest faithful config** that clears the bar (same "simplest floor-clearer" discipline as Step 3). A
wacky config baked into the heuristic is itself an unfaithful key.

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
  here is only the **floor** (`G ≥ 0.75`, or `0.75 × max-autotune` if it's a weird shape). Use the
  winner for its *information* — what it reveals about the mechanism — not as a config to copy. Lifting
  the kernel's geomean past the floor is a goal you keep pursuing (§6.0), not a line you stop at.
- **Target the SIMPLEST config that clears the floor, not the peak — where "simplest" is about the
  CONFIG, never the GATE.** *Simplest config* = fewest non-default fields. It does **NOT** mean *most
  narrowly gated*: a simple config riding a **broad** key (the widest the mechanism allows) is the goal;
  a simple config fenced to 2–3 curriculum cells is the failure mode. (Generality of the *gate* is
  Priority 2 and Gate H's BROADEN verdict — never sacrificed to make a config look simple.) Field-diff
  the winner vs your seed (block/tile sizes, reduction loops, `num_warps`, `num_stages`, eviction,
  `pid_type`, `maxnreg`) — **the differing fields are your worklist.** Hypothesize, from the diff *as a
  whole*, which field(s) carry the win, aim the heuristic at *those*, and **confirm by re-benching the
  candidate** (`configs=[cfg]`, the seed's bare-forward harness) that it clears the floor — a couple of
  confirmatory re-benches, not a leaderboard scan. A few % under the winner is plenty: pretty good, not
  the peak. **Simplicity breaks only genuine ties:** among configs of *comparable* complexity, take the
  one with **better perf** — never give up real perf when the simpler-looking option isn't clearly simpler.
  **Couplings are real — when neither field alone gains but the *pair* does, move both and emit the
  bundle as a unit** (don't force a one-field split). The matched-lever A/B (loop step 4) is what proves
  which field(s) carry the win and is the only way to see a coupling a build-up-from-seed test would hide.
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
  needs no gate — its bar is met. A confirmed weird shape retargets the floor to `0.75 × max-autotune`,
  Step 2b — that's a STUCK reason, not an exemption.)
- If a full oracle dies *post-convergence* (search done, cache-write crashes), extract the converged
  winner from the `.out` log and fair-re-bench it (don't discard). To re-bench an arbitrary extracted
  config, run it as `configs=[that_config]` through the **same** bare-forward harness as the seed arm
  — never hand-roll (you'd reintroduce the §4 footguns).

**The bar you climb to:** a shape's floor is **`G ≥ 0.75` vs tc** by default, or **`seed ≥ 0.75 ×
max-autotune`** if it's a weird shape (Step 2b) — either way it's **live worklist, not exempt** until
cleared. The kernel geomean is a diagnostic you keep lifting, not a hard line. Past the floor, **keep
climbing** (§Step 4) — the floor is the *disaster* bar, not a stop sign; and you keep climbing only
through faithful/general means (Priority 1+2), never a narrow fence.

**Invariant that ties it together:** the oracle searches the *same codegen* the seed emits, so a
source/codegen ceiling caps the oracle *too*. Therefore:
- `seed < oracle` ⇒ achievable perf is on the table; if you're in overtime on this shape, **keep
  climbing** (never "noise"/"stuck" until a fresh oracle says so).
- `seed ≈ oracle` ⇒ **nothing more is reachable for this shape** — you're at the codegen ceiling (the
  per-shape overtime endpoint). If that ceiling can't beat tc, the kernel source can't beat tc;
  confirm with a fresh converged oracle (Step 2b), never self-certify.

### Step 4 — Keep climbing (lift the worst kernel → beat tc → generalize the heuristic)
No floor outstanding does **not** mean stop. The climb continues, in this priority order: **(1)** lift
the lowest-geomean **kernel** higher (Step 1 triage) through faithful/general means; **(2)** push
cleared shapes from floor → **beating tc** (`G ≥ 1`) → **oracle parity** (`seed ≤ oracle × 1.03`); and
**(3)** — the standing Priority-2 work — **broaden and simplify the heuristic itself**: every narrow
fence is a BROADEN candidate (Gate H), and the refactor-critic periodically asks whether N narrow levers
collapse into one principled rule. This is pure upside on top of a banked deliverable. A perf win must
come from **theory + the answer-key diff**, never from cherry-picking an observed search winner
(p-hacking). "No clean rule / noise / stuck / done / ceiling" is **not an exit** — it's the trigger to
run a fresh oracle, attack from a different property, or work the BROADEN/refactor queue.

**Priority discipline:** worst-kernel-first + floor-at-acceptance are covered in §1 / §3. The one point
to land here: **generality work (Priority 2) is co-equal standing work, never "deferred until perf is
done"** — a BROADEN or a refactor is done whenever the evidence favors it, because an unfaithful/narrow
heuristic is a defect *now*. Lift the worst kernel, broaden, and simplify in whatever order the evidence
favors; the one guardrail is that you don't bank an edit that creates a *new* disaster.

**The DoD is a milestone to BANK, not a finish line.** When it's met, freeze and bank the champion
(commit + report), then **keep climbing** — do NOT stop and show the human (record it in the ledger and
continue; the human is truly not in the loop, §6.0). The DoD is never a stop; the run ends only when the
human stops it or a hard external block hits (§6.0).

### ★ THE PER-ITERATION LOOP — this is the engine; you live here ★
Steps 0–4 above just decide *what to feed in* (which kernel, which answer key, which bar). **This
loop is the actual work** — run it for every single change, no exceptions, hundreds of times. One
turn = pick the work item → through these steps → bank → immediately start the next turn (never pause
to narrate; §6.0). When in doubt, **take another turn.**

> **Priority guard (read before every turn):**
> 1. **Worst KERNEL, not worst shape** (§1) — fix the lagging kernel/family *generally*; this sweeps up
>    its disasters without per-shape fences. **The #1 misexecution is a narrow per-shape fence / chasing
>    one weird shape**, not "spending a turn on overtime."
> 2. **The floor binds at ACCEPTANCE, not as the worklist driver** (§3) — Gate R rejects any edit that
>    *introduces* a disaster; you don't "drop everything" to chase a below-floor shape (the
>    completeness-critic surfaces any disaster masked inside a healthy kernel).
> 3. **Generality work is co-equal, not deferred** — a BROADEN or a refactor is Priority-2 standing work
>    (Step 4), done whenever the evidence points there.
> 4. **The oracle = reachability/answer-key/weird-shape signal** (Step 2) — reach for it when stuck,
>    **never sweep autotune across every shape**.
> Whatever you pick, the edit must clear the acceptance check (4b) + the F/G gates (5–6); perf never buys
> its way past faithfulness or generality.

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
   this A/B is light. (Full rationale: §5 + the §3 acceptance model.)
4b. **Acceptance check (disaster-avoidance MUST, Gate R).** config-recorder over the full active matrix
   → re-bench every changed cell. The edit is **accepted** iff ALL hold (full rules: §3 acceptance model):
   (a) **no NEW disaster** — no realistic at/above-floor shape ends below it, and no already-below shape
   drops further (`0.75` vs tc, or `0.75 × max-autotune` for a weird shape; a pre-existing disaster left
   unchanged is *not* a rejection — see §3); (b) **no (kernel,dtype) cell collapse** — no (kernel,dtype)
   cell's geomean drops >10% below the *frozen-champion* anchor (not the rolling champion); (c) **net
   progress** — a *majority* of (kernel,dtype) cells improve beyond the ~5% noise band (within-noise =
   neutral). **Generality exception:** a breadth
   edit (Gate H BROADEN / refactor) MAY waive (c) **only** if its breadth is *measured* (§3 license) and
   (a)+(b) still hold; perf-only edits get no waiver.
5. **Adversarially verify the win is real** (§5 — Gate A; plus the Fact-gate (Gate D) if the edit
   touched a fact field's *meaning*, its **population** (a `device_ir.py` builder or an inline derivation a
   branch keys on — the lucky-proxy class, §2), OR a threshold a branch compares it against; Gate F if the
   win is
   counter-intuitive OR the lever flips a structural default / sets more than one non-default field).
   Kill it on majority-refute. Gate A reuses
   the focal cell from step 4b's Regression-referee sweep as its authoritative re-bench (no duplicate
   bench); the skeptics analyze those numbers + one independently-authored own-script reproduction
   (a fresh agent writes it, the driver runs it serially). **When your hypothesis names a hardware
   mechanism ("fills the machine", "splits the row", "better coalescing", "strides rows"), CONFIRM IT
   IN THE GENERATED TRITON/IR before banking it as a rule — a plausible story you did not read the
   code to verify is a shape-scoped observation, not a banked mechanism (the `persistent_blocked`
   "fills the machine" miss: the stated reason was wrong AND it carried an inert `num_sm_multiplier`
   field). Reading the lowered code is what (a) verifies the story, (b) tells you whether every field
   you set is doing work — drop the inert ones (Gate F check 4 / Gate H dead-knob rule; couplings
   exempt) — and (c) surfaces whether a neighboring config is strictly better. This is Gate F's job;
   it now fires on structural/multi-field levers, not only surprises.**
6. **Generality gate — does this lever belong in the core, and is it WIDE enough?** (§5 Gate H, a MUST
   on every lever.) Even a real, reproduced win (it passed step 5) can be sent back here: adjudicate
   **KEEP / BROADEN / DEFER / REJECT / BORDERLINE** by *key-faithfulness × magnitude × realism × downside
   × complexity × breadth*. **KEEP** → bank it (commit + record, steps 7–8); **BROADEN** (faithful, real,
   but gated narrower than its mechanism implies) → the default verdict on any narrow fence: emit the
   wider form and re-enter the loop to re-measure it (Priority-2 standing work, accepted under the step-4b
   generality exception); **DEFER** (faithful but not ready) → log it to the removed-heuristics-log with a
   re-add recipe, don't ship it; **REJECT** (unfaithful key — identity / bare-dtype / op-pattern proxy) →
   invoke **try-harder** (re-key mode) to re-key it on a workload property; **BORDERLINE** → record the
   tradeoff for the human's later judgement and provisionally DEFER (keep climbing — never ask the human,
   §6.0). This is the generality firewall *per lever*, in BOTH directions: a lever that wins by fencing
   the curriculum dies here, **and** a lever narrower than its mechanism warrants is pushed wider —
   narrowness must be *justified* (a proven reversal boundary), not assumed.
7. **Commit** the green change to your branch (the immutable SHA the ledger entry will cite).
8. **Record + update state.** Notebook + ledger — **wins AND rejections** (a rejected fix/hypothesis is
   first-class data: one compact line — *what you tried, why it failed, the evidence pointer*: "raising
   welford's normalize cap 2048→4096 → regressed ~7.3× at (262144,5120)", not "bigger cap bad"). Record
   gate FAILs/REJECTs as-returned (never laundered) — this is what stops a re-invoked context from
   re-deriving a dead idea (§6.1), and the rejected pile is where the hard-pile's *common thread*
   surfaces (§3). **Update the per-shape status table** (new G for the edited shape + every cell the
   Regression-referee re-benched) and write the **exact next action** so START-HERE resumes without
   re-deriving it. Record the compact lesson, not the full A/B transcript.

### The acceptance model (what makes an edit bankable)
The per-(kernel,dtype) ≥0.85 hard bar is **gone** (it bred rescue-ifs and 0.85-gaming); the model accepts
**aggregate progress** under hard guardrails. An edit is bankable iff:

1. **No NEW disaster (the one hard perf bar, per-SHAPE).** The edit must not push a *realistic* shape
   that was **at/above its floor below it**, nor drop an already-below shape further — floor = `G = 0.75`
   vs tc, or `0.75 × max-autotune` for a weird shape (Step 2b). (A *pre-existing* below-floor shape that
   the edit leaves unchanged is **not** a per-edit rejection — it's a tracked disaster the edit simply
   didn't fix, addressed via worst-kernel triage. The bar here is "don't make it worse"; since Gate R
   only re-benches *changed* cells, an untouched disaster never trips it.) It stays **per-shape** because
   it is the only thing that catches a single-shape cliff a kernel-geomean averages away (the
   non-monotonic softmax small-N landmine). *Realistic* = a real-model workload, INDEPENDENT of curriculum
   membership: an in-curriculum shape is realistic, **and a realistic shape *outside* the curriculum still
   binds the floor**. *Unrealistic* = genuinely synthetic/diabolical (2 rows of 16M elems; a tight box
   around one shape) — only these may sit below floor. *When in doubt, realistic.* The
   realistic↔diabolical call is the **Regression-referee's** (Gate R), never the self-interested
   worker's; never launder a genuine regression as "unrealistic."

2. **No (kernel,dtype) cell collapse, anchored to the FROZEN champion.** No (kernel,dtype) cell's geomean
   drops more than **~10% below the frozen-champion anchor** (the anchor is **per cell** — never a
   dtype-blended per-kernel number, which would let one dtype tank ~30% behind another's ~25% lift). The
   anchor is the **best banked champion so far**, and it
   **only ratchets up** (re-freezes when you bank net progress); it is **never the rolling per-edit
   champion** — measuring "don't drop >10%" against a champion that slides down with every edit permits
   unbounded ratchet-collapse (100→90→81→…, each step "legal"). Anchoring to the frozen point caps total
   drift no matter how many edits. tc itself is the ultimate floor under this (rule 1), since tc never moves.

3. **Net progress (aggregate, noise-gated).** A **majority of (kernel,dtype) cells' geomeans improve
   beyond the ~5% noise band** (median-of-N; a within-noise move is **neutral**, never counted as a "win" — otherwise
   many sub-25µs noise wobbles manufacture a fake majority). The **denominator is the cells the edit
   actually moved** (the changed/re-benched set — see "Deriving the affected set"): byte-identical cells are
   **perf-invariant** (deterministic codegen guarantees they're unchanged) and **excluded by construction**,
   not counted as neutral; a moved cell whose change is within-noise is **neutral** and dropped from the
   tally. A faithful rule firing on only a subset is therefore judged on that subset (not diluted by the
   untouched matrix) — but an edit that moves several cells still needs a **majority of *those*** up (an
   up/down split that is not a majority-up ⇒ fails net-progress). This single majority-beyond-noise test is the
   positive criterion; there is **no** separate "mean-of-geomeans up" (the raw mean is skewed by one
   volatile small-N kernel — majority-beyond-noise is the outlier-robust version).

**The generality exception (Priority-2 license).** A breadth edit (a Gate H BROADEN, a refactor-critic
simplification) **may waive rule 3** — it may dip the majority and take up to a **~3–5% per-(kernel,dtype)-cell drop**
— but ONLY if: (i) rules 1 and 2 still hold (no disaster, no >10% collapse vs the frozen anchor); (ii) the
breadth is **MEASURED, not asserted** — a named off-train *realistic* / TRANSFER shape the edit
**demonstrably lifts**, recorded per-edit (no measured breadth ⇒ it is not a generality edit ⇒ no license);
and (iii) the license is a **once-per-(kernel,dtype)-cell budget against the frozen anchor**, not renewable every edit
(else rotating 4% nicks across cells erode the whole matrix while each edit "looks fine"). Gate H / Gate R
own this verdict — the worker cannot self-label an edit "generality" to buy the drop. Log every spent
license with its per-shape deltas + the breadth shape it bought, so the cost is visible, not buried.

**Overfit is blocked structurally, not by the aggregate.** Aggregate acceptance is exactly the
"geomean trap" the old method feared — *on its own* it lets a curriculum-memorizing edit pass. What
stops that here is **not** a held-out perf clause (the curriculum's val/test are interpolation *inside*
the train envelope by construction — they rise when train overfits, so VAL-up is *necessary but not
sufficient*). It is the **F/G gates as hard constraints**: a memorizing edit keys on an unfaithful proxy
or a curriculum fence, so **Gate D (faithful field + population + threshold), Gate H (faithful key +
breadth), and Gate E (curriculum-fence audit + the sole-reader TEST firewall) reject it regardless of how
good the aggregate looks.** Priority 1+2 dominating Priority 3 *is* the overfit defense.

**Deriving the affected set mechanically (the config-recorder skip).** Re-bench the cells the edit could
touch, and **DERIVE that set rather than guess it:** run the config-recorder
(`_lab/harness/config_recorder.py`, §5) over the FULL active matrix (every kernel × shape × dtype × split
incl. robustness) BEFORE and AFTER the edit and diff the normalized configs — the cells that *changed* are
the ones to re-bench; byte-identical cells are provably perf-invariant (codegen is deterministic in
config+source) and need no re-bench. Every historical backstop miss was a *changed cell at a shape/dtype
the worker never thought to sweep* (a ~7.3× valley at an untested in-curriculum shape; CE fp32 +24% — a
dtype not swept; the D4 occ corner). THREE rules keep the skip sound: **changed ≠ win** (a flagged cell
still earns the full A/B + gates); **full matrix or it's a false all-clear** (valid only if the diff spans
every active dtype + robustness); **selection-only** (config-identity ⇒ perf-identity ONLY when the edit
changed *which* config is emitted — an edit touching kernel source / a fact builder / normalize / a
lowering needs the generated-Triton diff: `--triton`). A climbing edit *should* change some configs — that
is the worklist, not a failure.

**The skip's one blind spot — closed by a representative curriculum, not by invented shapes.** The
changed-cell skip catches a widen's effect on every regime the curriculum SAMPLES (those in-matrix cells
change config ⇒ get re-benched); it cannot see a realistic regime the curriculum does not sample, which
rule 1 still binds (realistic shapes bind INDEPENDENT of curriculum membership). The defense is the
curriculum's **representativeness** — it should sample every realistic regime a lever can fire in. When a
BROADEN/refactor extends firing into a realistic regime the curriculum misses, the Regression-referee
(Gate R step 3b) may **extend the curriculum — but ONLY with a REAL, named workload** (an actual model/op,
cited), added to the active matrix (never TEST) and noted; it becomes a permanent, always-swept cell. The
referee may **never fabricate** a shape to manufacture a regression: a below-floor shape is a disaster
only when it is a real named workload — the floor guards against real harm, not invented edge cases
(unless the task file pins the workload as FIXED — then flag the needed real shape to the human-review
queue rather than adding it).

---

## 4. Benchmarking footguns (these dominate the numbers)

1. **FORWARD ONLY — never build a grad graph for either arm.** Build inputs `requires_grad=False`;
   no `.backward()`. The biggest artifact seen was timing the **autograd wrapper**
   (`*.apply`/`save_for_backward`), which adds ~9–18 µs fixed host overhead and flipped one kernel
   1.056→0.79. Time the **bare forward** (`helion.kernel(fwd.fn, config=seed)`) for *both* Helion
   and the tc reference.
2. **Reset dynamo per shape:** `torch._dynamo.reset()` before each compile, else tc caches multiple
   shapes and recompiles into slower **dynamic-shapes** mode (unfair to tc).
3. **Prefer tritonbench** — it already handles the footguns it CAN (it calls `attr.reset()` + dynamo
   reset per input for free) — but its DEFAULT timing mode is itself a footgun: default `do_bench`
   mis-times low-M (#9) and it does not save you if you batch kernels (#11). Drive the seeded arm by promoting the seed to `default_config`
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
   contaminated and retry / wait-idle (headline runs need foreign mem ≈ 0). Also confirm the **GPU
   clock is stable** before a headline timing (pinned/boost steady, no thermal/power throttle —
   `nvidia-smi` clock + throttle-reasons): throttle/boost drift can flip a near-floor verdict, and on
   a shared host a foreign job can throttle you even when GPU memory looks clean.
6. **Accuracy gate before timing**, vs the eager reference *built at the same dtype*, upcasting both
   to fp32 before `allclose`. Tolerance is task-specific (see the task file) but the rule is
   universal: **any tolerance change is logged with measured justification, never silent.**
   - **The gate also binds the AGGREGATE + the tolerance metric.** (a) A per-(kernel, dtype) geomean
     is computed ONLY over rows that PASSED accuracy — exclude every acc-fail and NaN/inf row and
     surface the excluded set; never fold a wrong/NaN-output latency into the headline (the
     sum/welford/fp16 wide-V false-win class). (b) For outputs that can be near zero, gate on
     **max_abs, NOT max_rel/rtol** — a tiny absolute error is a huge relative one → FALSE acc=0
     (welford's "acc=0" was a false alarm: fp32-exact, bf16 rounding hit BOTH arms). (c)
     **Arm-equivalence:** both timed arms must compute the SAME work — an extra Helion output the
     reference omits biases G by a fixed factor invisible to the noise/accuracy checks (jsd timed an
     extra dX → loss inflated ~11–18%; fair loss-vs-loss ≈ 0.94–0.97). (d) **Force + assert the
     dtype** — some operators silently default (softmax → fp16); set it AND assert the tensors took
     it. (e) tritonbench reports perf even on accuracy-FAILING kernels — gate yourself.
7. **Verify the config actually ran:** record the *normalized* running config after
   `bound.ensure_config_exists(args)` to prove `configs=[cfg]` (no autotune) executed the config you
   intended.
8. **Memory hygiene** between multi-GB shapes (`del ...; torch.cuda.empty_cache()`), and
   **incremental JSON checkpointing** so a foreground kill never loses completed rows.
9. **Device-time must be COLD-L2 — and `do_bench`'s DEFAULT mis-times low-M.** Two linked traps.
   (i) `do_bench` DEFAULT charges ~20µs Python host-enqueue to the kernel → **phantom losses** on
   low-M / bandwidth-bound shapes. (ii) The tempting escape — plain CUDA-graph replay (`g.replay()`
   ×N over the SAME buffers) for device time — is WORSE for any working set ≤ the GPU's L2: the data
   stays HOT in L2, the kernel never re-reads HBM → physically-impossible ~3–5 TB/s, and the two arms
   benefit by DIFFERENT L2 factors so the **ratio G is distorted, not just the absolute** (a real seed
   LOSS read as a fake WIN: rms_norm bf16 (2048,4096) plain-cudagraph G=1.27 vs cold-L2 truth ~0.91).
   FIX: a **cold-L2 device metric** — `do_bench` (this Triton build flushes L2 between reps — a *build*
   property; re-confirm if the Triton version changes) OR profiler `self_device_time` with an explicit
   ~128MB flush before every call. Plain cudagraph is trustworthy ONLY when the working set ≫ L2
   (wide-N / large-vocab). The L2 SIZE is machine-specific (see `local-setup.md`); sub-L2 is where this
   bites. **Why host-overhead is the SAFE bias and L2-residency the DANGEROUS one:** equal host
   overhead H makes `(T+H)/(S+H)` move toward 1.0 — it can shrink a margin but NEVER flip a true win
   into a loss; L2-residency is asymmetric across the arms and CAN flip the sign.
10. **Measure-mode meta-rule: default to the baseline harness's OWN mode; deviate only on MEASURED
   evidence.** tritonbench's default is `triton_do_bench`. Reach for a bespoke device-time / cold-L2
   mode ONLY when you have a MEASURED artifact proving the default biases YOUR regime — and even then
   deviate to a **cold-L2** device metric (the forward climb's host-overhead artifact was real and
   triple-confirmed, but the plain-cudagraph metric it was *originally* executed with was itself
   L2-contaminated — see #9). Do NOT carry a deviation tuned for one regime (low-M, bandwidth-bound)
   into another (heavy / compute-bound, e.g. backward) without re-confirming the artifact exists there
   — and **NEVER cudagraph-wrap a `torch.compile(mode='reduce-overhead')` baseline** (it self-graphs;
   the outer graph turns a true ~8µs into a measured ~30µs → fabricated-slow baseline → false G). The
   default baseline here is `tc_default` (footgun #3), not reduce-overhead; if you ever cudagraph, wrap
   only the non-self-graphing arms (seed + tc_default).
11. **ONE FRESH PROCESS PER KERNEL — distinct from footgun #4.** Footgun #4 puts the ARMS of ONE
   comparison in one process (to kill cross-process jitter); this is the orthogonal axis: do NOT batch
   many KERNELS or many config VARIANTS through one long-lived process. Compiling variant after variant
   accumulates dynamo guards/recompiles that silently corrupt the tc baseline — a real run fabricated a
   bogus **2.18× "win"** this way. tritonbench's per-input `attr.reset()` + dynamo-reset only helps if
   you do NOT batch. So: arms together in one process, but a **fresh process per kernel**, one timing
   run per GPU. (Do not mis-read #4's "one process" as license to batch everything — that IS the
   footgun.)
12. **An accuracy FAIL usually means a low-precision ACCUMULATOR, not a bad seed.** A naked Helion
   reduction (`tl.sum` / `.sum()` / `.mean()` / `.prod()`) accumulates at the INPUT dtype — it is NOT
   auto-promoted to fp32 (only `var_mean` promotes). So at bf16/fp16 a reduction can be **2–4.5× less
   accurate** than torch (which upcasts) and fp16 wide reductions can overflow/NaN. Before blaming the
   seed: re-run the SAME kernel with an fp32 input — if it then passes, it's the accumulator (per-kernel
   `.to(fp32)` before the reduction is the fix; the compiler-level fix is unbuilt). A "win" measured on
   numerically-wrong output is NOT a win.
13. **Noise floor — the measurement numbers belong here, not only in the gates.** Sub-~25µs shapes
   swing up to **±25% at the SAME config**. Re-run any measurement whose `do_bench` spread exceeds
   **~5%** (median-of-N, take the median of medians). Where possible lift M to push the shape above the
   floor before trusting a G; near the floor prefer the in-process seed/oracle ratio (both timed
   identically) over a raw latency. (The cached oracle LATENCY is itself a `do_bench` number with the
   same low-M / host-overhead bias — the cached winning *config* is free to reuse, but any
   reachability/ceiling/exempt call hinging on oracle-vs-tc near the floor must be re-measured COLD,
   not read from cache.)
14. **The timer is NOT biased — distrust your analysis, not the harness.** The timer was proven <1% vs
   a hand-rolled single-process `do_bench`, three ways — don't over-distrust it and re-engineer timing
   when a number surprises you. A shocking "anomaly" is almost always an ANALYSIS bug: the bogus "33×
   artifact" came from a field-diff that RECONSTRUCTED a config the search never emitted (`{block=1,
   w32}`) and re-benched that fabrication — same family as the "1174µs w32 mirage" (§3 loop step 4).
   The oracle config is a BUNDLE: re-bench it FULL and VERBATIM (`configs=[cfg]`, the seed's
   bare-forward harness); never isolate one lever and re-pair it with seed bits.
15. **tritonbench resolves ELSEWHERE — an operator/baseline edit in your worktree may silently
   no-op.** tritonbench operators load from the ORIGINAL tritonbench checkout via a hardcoded meta-path
   finder, NOT your worktree — so an operator or baseline edit (e.g. adding `torch_compile_<op>_default`)
   must be made in THAT checkout to take effect. The `assert helion.__file__` worktree check (§1) does
   NOT catch this — it's the tritonbench package, not helion. Verify the operator file you edited is the
   one actually imported. (The concrete resolution path is machine-specific — see `local-setup.md`.)

---

## 5. Gate disciplines (keep these even solo)

Independence is the asset — a gate that develops rapport stops re-examining. Under ultracode these
run as **fresh-context fan-out agents by default** (a solo self-check keeps none of the independence —
reserve it only for the explicit fast-path on a trivial within-noise tune); keep the discipline either
way. **For the high-stakes
adversarial gates (adversarial-verify [+absorbed independent reproduction], anti-giving-up, the
fact-gate [doctrine+faithfulness], generality, overfit/TEST-firewall, regression-referee, mechanism),
use the VERBATIM prompts in `gate-prompts.md` (gates A, B, D, E, F, H, R + the try-harder /
refactor-critic / completeness-critic helper frames) — do not improvise
their wording** (how you phrase an adversarial ask is how you bias its verdict; the scripted frames
bake in default-to-refuted, record-verdict-first, pin-the-SHA, and never-hand-over-your-conclusion).
Fill the slots, don't edit the frame. **The five MUSTS are anti-over-claim (Gate A), anti-under-claim
(Gate B), the portfolio guard (overfit/TEST-firewall, Gate E), the per-lever generality gate (Gate H —
which now both REJECTs overfit keys AND BROADENs under-narrow ones), and disaster-avoidance (the
regression-referee, Gate R, running the §3 acceptance model);** the rest fire per their trigger.
**The priority the gates enforce is lexical (§3): faithfulness > generality > perf** — the F-gates (D,
plus A's identity axis) and the G-gates (H, E, the refactor-critic) can REJECT an edit no matter how good
its perf/aggregate number is. Perf never buys its way past them.

**Fan-out disposition (ultracode; GPU re-measurement stays serial per §6):** N-refuter panels — Gate A,
the Fact-gate (D). Perspective-diverse (one verifier per axis, then synthesize) — Gate H and Gate E's
periodic audit. Judge-panel — Gate F, the hard-pile common-thread, the beat-the-oracle hunt. Solo/serial
— the GPU re-measurement inside Gate A and Gate R (fan out only the analysis around it). **Sole reader,
never fanned out — Gate E's FREEZE TEST read** (one reader, one TEST bench; fanning it out breaks the
firewall).
- **Correctness-first**, every iteration (you bypass the autotuner's accuracy check).
- **Matched-lever A/B** against the config you *chose to target* (§3's simplest floor-clearer — **not** the oracle's peak), perturbing **down** from it, never **up** from the seed (never pin block=1).
- **Regression-referee (Gate R, a MUST)** — on every edit that changed ≥1 cell, an independent referee
  runs the config-recorder over the FULL active matrix, re-benches every changed cell, and adjudicates the
  **§3 acceptance model**: no realistic shape below its floor (disaster), no (kernel,dtype) cell >10% below the
  **frozen-champion** anchor, and a majority of (kernel,dtype) cells up beyond noise (waivable only for a *measured*
  generality edit). Defaults to "a regression is hiding"; **owns the realistic↔diabolical verdict** and
  the frozen-anchor bookkeeping. Disaster-avoidance + acceptance as a gate, not a self-check.
- **Adversarial verify** each claimed win: N independent skeptics prompted to *refute* it (default
  to "refuted" under uncertainty); kill on majority-refute. Hunt: noise/fabrication, measuring the
  wrong thing, overfit (check held-out val + an off-focus kernel), **kernel-identity smuggling** (a
  constant fencing exactly one kernel's shapes), metric gaming (loosened tolerance, dropped shapes).
  The independent own-script reproduction (the absorbed results-referee) is **authored by a fresh agent
  and run by the driver serially**; all N analytical skeptics share ONE authoritative re-bench — the
  focal cell from Gate R's step-4b sweep, not a fresh duplicate — never N concurrent benches (§6 invariant).
- **Fact-gate (doctrine + faithfulness + POPULATION, Gate D):** fires on a new/changed fact field a
  heuristic reads, a new fact, a threshold a branch compares a fact against, **OR a change to how a field
  is POPULATED** (the `device_ir.py` builder, or an inline derivation a branch keys on — e.g. the
  `bs.reduction` lucky-proxy filter). ONE spawn checks THREE parts. *Part 1 — doctrine:* **derived facts
  never walk the graph**; the walk lives on a **walker fact** (`MemoryOpFact`/`AccumulatorFact`, computed
  once — prefer a field on an existing one over a new fact); the walker field is **consumer-agnostic** (the
  different-consumer test); sound provenance (the lone tolerated unsoundness is the pre-existing
  eviction-index TODO — to fix, not copy). *Part 2 — faithfulness (meaning + threshold):* the
  **divergence test** on the fact (construct a kernel where the lazy proxy and real property *disagree* —
  this falsified `num_load` and `num_reduction_ops`; a fact no branch reads is cut) AND on the
  **threshold** (a `fact <= K` that splits the value-set along dtype/kernel lines is a disguised fence;
  keep a dtype/identity-correlated quantity a FACTOR inside a byte/occupancy budget, never the operand of a
  literal). *Part 3 — POPULATION faithfulness:* does the populator **compute the claimed property
  directly** from faithful provenance, or piggyback on a flag that merely correlates on the curriculum?
  Gate D **authors a divergence kernel** to settle it — mutate a vetted template so the proxy and the real
  property diverge (add an `hl.zeros` accumulator dim → `bs.reduction` goes True on a non-reduced axis),
  **compile it, dump the registered facts** (NO GPU run), and check they disagree. An unfaithful populator
  is a *generality* defect (it decides which kernels fire), so it REJECTs even with a perfect perf number.
  But Gate D polices the faithfulness of what is RECORDED, not how *broadly* it fires: broad firing on an
  unforeseen workload and an honestly-*declared* fallback both PASS — only a *silent* lucky proxy fails, and
  an under-firing populator (declines to build the fact) is a Gate-H BROADEN, never a Gate-D pass. Doctrine
  in §2 ('Prefer to fire' + the best-guess↔lucky-proxy contrast).
- **Generality (per-lever KEEP / BROADEN / DEFER / REJECT / BORDERLINE — a MUST):** fires on **every**
  proposed or existing lever/branch/constant before it enters the core, adjudicating the lever's *place AND
  its breadth* by **key-faithfulness × magnitude × realism × downside × complexity × breadth**, judged
  purely by the gate's own embedded rules (NOT the maintainer's individual calls). Two hard lines, both
  directions: **(over-narrow)** the **default verdict on any fence narrower than its mechanism implies is
  BROADEN** — emit the wider form (the widest the mechanism supports), re-enter the loop to re-measure it
  under the §3 generality license; narrowness is kept only with a *proven reversal boundary* (a shape just
  outside where the win provably reverses, shown in the lowered code), never assumed. **Over-narrow includes
  UNDER-FIRING:** a fact-creation/eligibility gate so tight it drops an unforeseen workload into the
  really-bad default — a populator that won't build the fact, or a `get_seed_config`/`is_eligible` returning
  `None` — is itself an unjustified narrow scope ⇒ BROADEN (build the fact / return a best-guess config and
  fire; sub-optimal on that kernel is fine, and recorded), never a fence to leave. **(unfaithful)** an
  **unfaithful key** (kernel identity, a bare dtype literal like `itemsize == 2`, an op-pattern used as a
  kernel-class proxy) is **never bought by magnitude** — REJECT + spawn a try-harder agent to re-key on a
  workload property. A disaster rescue outranks breadth-chasing; faithful-but-not-ready ⇒ DEFER (→
  removed-heuristics-log, re-add retained); a genuine conflict ⇒ BORDERLINE — record the tradeoff for the
  human's later judgement + provisionally DEFER and keep climbing, never reflexive DEFER and never ask the
  human mid-run (§6.0). Distinct from the portfolio overfit guard below (whole-heuristic) and the
  refactor-critic (cross-lever architecture) — this is one lever at a time; walk-location/field/population
  generality is the Fact-gate (Gate D) above.
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
  **sole reader of TEST, exactly once, at freeze** — nothing else may bench it. (Under the §3 acceptance
  model — aggregate, no per-shape 0.85 bar — this gate carries *more* weight: it is a primary overfit
  defense, since the aggregate alone cannot catch curriculum-memorization. Held-out val is
  *necessary-not-sufficient* — it's interpolation inside the train envelope — so this gate leans on the
  *faithfulness* of keys/populations, not just the held-out number.)
- **Refactor-critic (Priority-2 standing work, on a cadence — a helper frame, not a per-edit MUST):** the
  whole-heuristic *simplification* dual of Gate H's per-lever BROADEN. Periodically (every K levers / at
  milestones) it asks: **can N narrow levers collapse into one principled rule? can the architecture be
  reshaped simpler / more faithful? — and, reading the PROVENANCE LOG, is any still-firing check there for
  a reason that has gone moot (a fixed codegen bug, an out-of-scope kernel/dtype/machine, an orphaned
  cascade gate)?** This is the standing de-bloat engine and the natural partner to
  worst-kernel-first (a lagging *family* often signals an architectural problem, not a missing narrow
  lever). **It is allowed to CHANGE the emitted configs / the heuristic — it does not have to be
  behavior-preserving.** Collapsing levers, removing a gate it judges unnecessary, splitting one gate into
  two, re-keying — all change behavior, and **that's fine**: generality is Priority 2, above perf, so a
  simpler/more-principled heuristic that shifts some configs is a *good* trade even if it costs a little
  perf (the §3 acceptance model still binds — no new disaster, no >10% collapse, and a perf dip is covered
  by the generality license). The ONE thing it may not do is knowingly violate a **known-valid**
  invariant — e.g. remove a gate already *established* to be necessary, or drop a fact a live lever
  depends on. Removing a gate it merely *suspects* is redundant is fair game (it re-enters the loop and
  the gates re-prove themselves). Committed **only if genuinely simpler**; its proposals re-enter the loop
  through Gate R + D/H like any edit. Frame in `gate-prompts.md`.
- **Mechanism + field-attribution (Gate F — broadened trigger):** a win can be real + reproduced yet
  *mis-understood*, so the generalized rule misfires off-curriculum AND carries dead fields. Fires on
  (a) a **counter-intuitive** win (e.g. narrow-N wanting fewer warps), (b) a lever that **flips a
  structural / near-universal default** (`pid_type`, persistent↔looped, indexing/TMA, `num_stages`,
  eviction), or (c) a **multi-field** lever. Two duties: require a hardware-level mechanism (ncu/IR)
  **verified against the lowered code, not from what the knob "ought" to do** (it predicts the
  boundary where the win reverses — else don't bank it as a rule, only a shape-scoped observation);
  and **attribute every non-default field** the lever sets — a field provably inert under the lever's
  own gate precondition, or whose solo-revert is within noise, is dropped (the dead-knob rule, Gate H
  rule 6), **except a coupled pair that only works together**. The analysis is GPU-free; the
  per-field marginal reverts are foreground-serial. (The `persistent_blocked` miss — wrong stated
  mechanism + an inert multiplier that passed A and H — is exactly what the broadened trigger now
  catches.)
- **A non-verdict is never a verdict:** a watchdog stall or API error after the analysis but before
  the verdict is recorded → re-fire fresh; never bank or fail on it.

(The compile-time config-recorder "behavior oracle" is a TOOL — not itself a gate — used in two places.
**(1) Inside Gate R** (the Regression-referee, §5 — the pass/fail MUST that replaced the old worker
self-check), running the §3 acceptance model: BEFORE/AFTER an edit, diff the
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
  deferred hard-pile, lift the worst kernel, push a floor-clear shape toward beating tc, **work the
  BROADEN-and-refactor queue (Priority-2 standing work — there is essentially always a fence to widen or
  an architecture to simplify)**, or run the completeness critic. There is *always* a next move; the
  gates exist to find it. (The **only** real end is the human
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
- **Fast path** — a within-noise tune of an existing faithful lever (no fact-field/**population**
  change, no direction surprise, no structural-default flip, no broadening of the gate, ≤1 non-default
  field changed): the matched-lever A/B (step 4) + the Regression-referee (Gate R), whose sweep IS the
  authoritative re-bench; skip the heavy stack (Gate A/D/F/H).
- **Full stack** — a new/changed fact, **a change to a fact's POPULATION (a builder or an inline
  derivation a branch keys on — the Gate-D population trigger)**, a counter-intuitive win, a claimed
  beat-tc, a borderline lever, **a lever that flips a structural default
  (`pid_type`/persistent↔looped/indexing/`num_stages`/eviction), sets more than one non-default field, or
  a BROADEN of an existing lever's gate**: fire the relevant gates (Fact-gate incl. its population/
  authoring part, F, H incl. BROADEN, …) with their fan-out analysis. (F owns
  mechanism-verification-against-code + field-attribution, so a structural/multi-field lever cannot take
  the fast path; a population change cannot, because that is where the lucky-proxy class hides.)

**The driver + ephemeral helpers** (all non-GPU fan-outs; the dump never lands in the driver — only the
distilled finding does): **code-investigator** ("where/how does X work?" — reads source/IR),
**perf-investigator** ("why is A faster than B?" via ncu / generated-Triton — returns the mechanism,
not the dump), **try-harder** (the general "escalate when stuck" agent — hypothesis / re-key / approach
modes; gate-prompts.md), the **refactor-critic** (the cadence-fired whole-heuristic simplification dual
of Gate H's BROADEN — Priority-2 standing work; gate-prompts.md), and the **completeness-critic**
(loop-until-dry). They are ephemeral and fresh-context: no message-relaying, no acceptance-laundering, no
surviving past their one task.

**The notebook + ledger are the standing artifacts** (never an agent), and the resume-after-death
source of truth (§6.1). Required notebook sections: CURRENT HEURISTIC STATE / PER-SHAPE STATUS TABLE
(carries each shape's G + its frozen-anchor reference, so the disaster floor and the >10%-vs-anchor check
are auditable) / PER-(KERNEL,DTYPE) GEOMEAN TABLE (the worst-kernel-first triage signal) / FROZEN-CHAMPION ANCHOR
(the current per-(kernel,dtype)-cell anchors + when each last re-froze) / BANKED WINS / TRIED-AND-REJECTED /
BROADEN-AND-REFACTOR QUEUE (standing Priority-2 work: each narrow fence flagged for BROADEN + each
candidate architectural simplification, with the measured-breadth shape it needs) /
DEFERRED-HARD-PILE-AND-BORDERLINE / NEXT ACTION / **HUMAN-REVIEW QUEUE** (append-only, ranked, deduped —
every BORDERLINE, every STUCK hard-pile entry, every spent generality-license, and every skipped
irreversible action writes ONE line: {what, why-blocked / the tradeoff, the provisional decision, where to
look to reverse it} — the orchestration counterpart to "never ask mid-run"). The ledger is append-only
gate verdict objects AS-RETURNED keyed by {SHA, gate, claim}.

The completeness-critic runs on a **cadence** (every K levers, whenever the disaster worklist empties,
and before any dead-end is accepted — NOT just "near the end", which is undefined for a never-stopping
run) and **loop-until-dry** (re-run until K consecutive passes find nothing new): *what's missing — a
dtype not swept, a claim unverified, a cap or **fact-population** not audited, a shape under the noise
floor, a deferred hard-pile item never revisited, a narrow fence never sent to BROADEN, a BROADEN/refactor
whose newly-covered realistic regime was never checked for a real named shape to add (Gate R step 3b), a
stale architecture the refactor-critic never revisited?* Each gap is appended to the worklist and cleared or
explicitly logged-and-skipped. **Never silently cap coverage** (top-N, no-retry, sampling) without
logging what was dropped — silent truncation reads as "covered everything" when it didn't. Frame:
gate-prompts.md.

**The refactor-critic fires on a HARD cadence — track it like a gate, not a someday.** Generality is
Priority 2, so simplification is not optional background work; it must actually run. Enforce it
mechanically: keep a **`levers_since_refactor` counter** in the notebook (HEURISTIC STATE) and **fire the
refactor-critic every K banked levers** (K≈4–5) **AND at every freeze/DoD milestone AND whenever
worst-kernel triage lands on a family that already has ≥2 narrow levers** (a smell that the architecture,
not another lever, is the problem). Resetting the counter is the ONLY thing that clears the obligation;
if the counter is over K, firing the refactor-critic is the next action — it outranks overtime
win-chasing (it is Priority-2 standing work, ahead of Priority-3 perf). The completeness-critic's
"stale architecture the refactor-critic never revisited?" check is the backstop that catches a missed
cadence. (Its proposals are not self-certifying — each re-enters the loop through Gate R + D/H like any
edit; see its frame in gate-prompts.md.)
