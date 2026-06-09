# Helion seed-heuristic hill-climb — METHOD (durable, task-agnostic)

This is the reusable *how*. It does not change run-to-run. A per-task file (e.g.
`dtype-task.md`) supplies the *what*: which heuristic, which goal, which knobs, which shapes,
the deliverable. **Read this file, then `local-setup.md`, then the task file.** Where they
conflict, the task file wins on scope; this file wins on method and discipline.

## START HERE — are you resuming or starting from scratch?
**If you are picking up an in-progress run:** the gated log is your source of truth (§6.1) — read
the existing **hub log + worker notebook + ledger** (paths in the task file / `local-setup.md`)
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
> next? Do the next turn of the loop — don't drift into open-ended analysis, re-reading, or re-planning.

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
hard part is knowing *what to chase and when to stop*. **The primary objective is BEATING
torch.compile-default** — the portable, reproducible yardstick. The oracle (autotune) plays *two*
supporting roles in round one, never the lead:
1. a **reachability detector** — does an achievable config that beats tc even exist? and
2. an **answer key** — a concrete config to hill-climb the seed *toward*.

### Step 1 — Establish the torch.compile yardstick (cheap, always first)
Measure **seed vs torch.compile-default** across the curriculum (and, for a multi-target run, across
every active dtype). This is the headline: `G = tc_latency / seed_latency` (G≥1 = seed beats tc).
Cheap (no autotune); it triages where to spend effort. **The bar uses ε: the literal test is
`seed_latency ≤ tc_latency × 1.03`** (≤3% slower counts as a tie; reserve the looser 5% only for a
deliberately-accepted trade, §-backstop).
- `seed > tc × 1.03` on a shape → a real gap, **round-one work**. Go to Step 2.
- `seed ≤ tc × 1.03` everywhere → at/above the portable reference; round one's bar is met. **Only
  now** does chasing the oracle (Step 4) make sense.

### Step 2 — On a tc-gap, get an answer key (the oracle confirms reachability AND hands you a target)
When seed loses to tc, you need two things: *is the win reachable*, and *what config gets it*. Two
complementary tools — use both:

**(a) Run the autotune oracle and compare it to tc.** This is the cleanest reachability test:
- **oracle BEATS tc** → a beating config exists and is reachable; its winning config is your
  **answer key**. Go climb toward it (Step 3).
- **oracle CANNOT beat tc** → Helion's codegen can't express the optimal Triton here, so the *tc*
  bar is unreachable — **but this is NOT permission to stop tuning the seed.** The goal just
  **switches target**: chase the **Helion oracle** instead of tc, i.e. drive `seed ≤ oracle × 1.05`
  for this shape (Step 3 mechanics, measured against the oracle's config/latency instead of tc's).
  The shape is **done/exempt only once `seed ≈ oracle`** (within ~5%); while `seed < oracle` there
  is still reachable perf on the table, so the shape **stays on the worklist even though tc is out
  of reach** (real miss: jsd — the oracle was ~18% faster than the seed, yet it was waved off as
  "exempt" because *both* lose to tc). "Helion can't beat tc here" is a real, document-worthy
  kernel/codegen finding, but it bounds the **tc bar only** — never the **seed-vs-oracle** bar. This
  ceiling-claim is valid only on a **converged** oracle (search-finished, not truncated/OOM)
  genuinely ≤ tc — route it through anti-giving-up (§5), never self-certify.

**(b) Optionally read the generated Triton** to *understand the mechanism* of the gap:
- Helion: `Kernel.to_triton_code(config)` (or `HELION_PRINT_OUTPUT_CODE=1`).
- torch.compile: `TORCH_LOGS=output_code python ...`.
This explains *why* (block/tile sizes, num_warps, stages, eviction, `pid_type`, persistent-vs-looped,
loop chunk…) and, if the oracle came up short, confirms whether tc's structure is codegen-unreachable.

**Quick-autotune is the cheap middle option** and is often the right first move: it's much faster
than a full oracle and, unlike tc, it directly hands you a Helion **config you can hill-climb
toward** (tc only gives you a config if you reverse-engineer its Triton). Use quick-autotune to get
a target fast, then confirm with a full oracle before banking — a quick *gap* is real (only widens
at full), but a quick *parity/win* is **suspect** and must be full-confirmed (a real run saw
`softmax(1024,65536)` flip 1.000→1.338 between quick and full).

### Step 3 — Hill-climb the seed toward the answer key
- **Cache the oracle** `{winning_config, latency}` keyed by `(kernel, shape, dtype, source-hash)`;
  invalidate on any source/codegen change. Re-comparing the cached oracle is free; only re-bench
  the seed.
- **Field-diff** seed vs the answer-key config (block/tile sizes, reduction loops, num_warps,
  num_stages, eviction, `pid_type`, maxnreg). **The differing fields are the worklist.** Form a
  workload-property hypothesis per field, edit, A/B (the per-iteration loop below).
- **EASY CONFIGS FIRST — defer the "weird" oracle winners.** A *close* answer-key (a clean field
  diff or two: a warp bump, a persistent↔looped flip, an eviction tweak) → climb now (fast,
  high-confidence). A *weird* one (odd warp count, `pid_type=persistent_interleaved`, tuned
  `maxnreg`, a coupled-knob bundle with no obvious property) → queue in a "hard pile". Why: bank easy
  wins fast instead of stalling on the hardest shape, and the hard ones are best attacked *together*
  — a batch of weird winners often shares a **common thread** (a rule from the batch generalizes
  where a one-shape fit overfits). Drain the easy pile, then work the hard pile by its shared
  property, not knob-by-knob.
- If a full oracle dies *post-convergence* (search done, cache-write crashes), extract the converged
  winner from the `.out` log and fair-re-bench it (don't discard). To re-bench an arbitrary extracted
  config, run it as `configs=[that_config]` through the **same** bare-forward harness as the seed arm
  — never hand-roll (you'd reintroduce the §4 footguns).

**The round-one bar — per shape, hit the BEST REACHABLE target** (per-shape, **never** an aggregate
geomean — declaring done on a good geomean while individual shapes lag is a classic failure):
- **tc reachable** (oracle ≥ tc) → the bar is **beat tc**: `seed ≤ tc × 1.03`.
- **tc unreachable** (oracle < tc, codegen-bound per Step 2a) → the bar **drops to the oracle, it
  does NOT disappear**: `seed ≤ oracle × 1.05`. Document the "can't beat tc" finding, then still
  climb the seed to the oracle. A shape is **exempt (stop tuning) ONLY when `seed ≈ oracle`**; a
  codegen-bound shape with `seed < oracle` is **live worklist, not exempt**. Never conflate "Helion
  can't beat tc" with "the seed is optimal" — they are independent (jsd: oracle ~18% faster than the
  seed, yet both lose tc).

**Invariant that ties it together:** the oracle searches the *same codegen* the seed emits, so a
source/codegen ceiling caps the oracle *too*. Therefore:
- `seed < oracle` ⇒ achievable perf is on the table; **keep climbing** (never "noise"/"stuck" until
  a fresh oracle says so).
- `seed ≈ oracle < tc` ⇒ **done for this shape** — the kernel source can't beat tc, confirmed by
  Step 2a (oracle ≤ tc). Never self-certify this ceiling without that fresh-oracle confirmation.

### Step 4 — After consistently beating tc: chase the oracle, then beat it (overtime)
Only once round one holds (seed beats tc on every reachable shape, **and** is within ~5% of the
oracle on the tc-unreachable ones) does the broader *oracle-parity* bar engage: push `seed ≤ oracle
× 1.03` (per-shape) everywhere — including the shapes that already beat tc. Beyond parity, hunt configs that *beat*
the oracle (couplings its bounded stochastic search under-samples). A win must come from **theory +
the answer-key diff**, never from cherry-picking an observed search winner (p-hacking). "No clean
rule / noise / stuck / done / ceiling" is **not an exit** — it's the trigger to run a fresh oracle
and read the answer key.

**The DoD is a milestone to BANK, not a finish line.** When the task file's definition-of-done is
met and verified, **freeze and bank it** — commit the champion, write the report — so a fully-valid
deliverable is locked in and can never be lost. Then **keep climbing**: push every shape from
`seed ≥ tc` toward `seed ≈ oracle`, and past parity hunt configs that *beat* the oracle. There is no
"done" that ends the run; the DoD just guarantees there's always a banked win behind you while you
keep going (any later improvement re-banks a fresh champion). The run ends only when the human stops
it or a hard external block hits (§6.0) — never because "the deliverable is met."

### ★ THE PER-ITERATION LOOP — this is the engine; you live here ★
Steps 0–4 above just decide *what to feed in* (which shape, which answer key, which bar). **This
loop is the actual work** — run it for every single change, no exceptions, hundreds of times, until
the deliverable is met. One turn = pick the worst un-finished shape → through these six steps → bank
→ immediately start the next turn (never pause to narrate; §6.0). When in doubt, **take another
turn.**

> **Round-one guard (read before every turn):** the bar is **beat tc**, not match the oracle. Pick a
> shape that still **loses to tc** (`seed > tc × 1.03`). The oracle is only your *reachability test +
> answer key* (Step 2). If a shape already beats tc, do NOT keep tuning it toward oracle parity —
> that's Step-4 overtime, and it waits until *every* reachable shape beats tc. Chasing oracle parity
> on an already-winning shape while another still loses to tc is the #1 misexecution.

1. Read the oracle/tc diff → form a **workload-property hypothesis** (never kernel-identity, never a
   dtype/identity special-case in disguise). Keep this step IN the worker — it's cheap and uses your
   intuition. But **offload the context-heavy investigation that feeds it** (a Triton dump, ncu
   output, a full oracle log, a 9-kernel field-diff) to an ephemeral code/perf-investigator (§6.2)
   that returns only the distilled finding — the raw dump must never land in your context. (STUCK on
   a hard-pile/counter-intuitive item? fan out N idea-generators from the same evidence for
   *diversity* — a stuck-state escalation, not a per-turn step.)
2. Edit the fact/heuristic.
3. **Correctness-gate** vs an eager/reference baseline (you bypass the autotuner's accuracy check).
4. **Matched-lever A/B — perturb DOWN from the good config, never UP from the seed.** To attribute
   *which* lever carries a win, start from the oracle's **full verbatim** config (every lever at its
   proven-good value) and revert **one** field to the seed value; measure the delta. **Direction is
   everything:** reverting down keeps every *other* lever at its good value, so a lever whose payoff
   is *unlocked by a partner* still shows it (the partner is present). Building *up* from the seed
   (e.g. pin `block_sizes=[1]`, then add `num_warps=32`) strips the partner away and makes the lever
   look useless or harmful — that's the "1174µs w32" mirage: `[block=1, w32]` is a config the search
   **never visited** (warps are physically coupled to block size; on large-M, block=1 is below the
   `raise_grid_block_minimums` floor and outright invalid), so its latency tells you *nothing* about
   whether w32 helps where it actually appears. **Never read a hand-assembled seed-bits+oracle-bits
   Frankenstein as one lever's marginal effect.**

   **This is NOT a ban on coupled configs.** Wins that require two levers moving *together* are real
   and common (e.g. `persistent_interleaved` + `maxnreg=64` + `sm_mult` for wide-reread CE — pid
   alone regresses, maxnreg carries it, they only win jointly). You capture couplings three ways,
   all legitimate: (a) the **autotuner hands you the bundle** on-frontier — oracle/quick-autotune
   already explored couplings, so you don't hand-invent them; (b) you **test a coupled candidate as
   a UNIT** (the whole bundle vs the oracle), derived from theory + the answer-key diff (never a
   cherry-picked search winner); (c) the **seedable ladder** — add levers cumulatively (oracle block
   → +warps → +stages vs full verbatim), classifying plateau-vs-peak, which also exposes *redundant*
   levers (two substitutes where reverting either alone looks free but reverting both tanks it).
5. **Adversarially verify** the win (§5).
6. Record to the notebook + ledger — **wins AND rejections.** A rejected fix/hypothesis is
   first-class data: log it as one compact line — *what you tried, why it failed, the evidence
   pointer* ("raising welford's normalize cap 2048→4096 → regressed ~7.3× at (262144,5120)", not
   "bigger cap bad").
   Record gate FAILs/REJECTs as-returned (never laundered). This is what stops a re-invoked context
   from re-deriving a dead idea (§6.1), and the rejected pile is where the hard-pile's *common
   thread* surfaces (§3). Record the compact lesson, not the full A/B transcript.

### The no-regression backstop (the part that makes multi-target safe)
A gain on one shape that **silently** drops another below its floor is **rejected**. A config-flipping
cap's A/B **must sweep the workload axis it flips on, at MID and EXTREME values** — never endpoints
or hand-picked shapes (a 3-shape A/B once passed an auditor but a broad re-run caught a ~7.3× valley
at an untested in-curriculum shape). **For a multi-target run (e.g. several dtypes), "no regression"
spans the entire active matrix:** every edit must hold the floor on *every* dtype/shape already in
play, not just the one you're currently chasing. **During the climb you discharge this by
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
1. **Bounded:** no single shape regresses more than **~10%** (a hard ceiling — a "net win" average
   can hide a catastrophic valley; that's the EDIT#4 lesson). The relaxation is for ~5% nicks, not
   gambles.
2. **Net positive over the FULL affected set:** more shapes improve than regress, by more than they
   regress, measured across the *complete* flip-axis sweep + every banked shape the edit could touch
   — never a sample of winners. You cannot claim "net win" on shapes you didn't measure.
3. **Principled, not arithmetic:** there is a *workload-property reason* the trade exists (a cap that
   genuinely helps the common regime forces a few edge shapes to share it). "The numbers happen to
   net out" is **not** a reason — that's just p-hacking the curriculum (the geomean trap that caused
   run 2's false victory).
4. **No separator first:** before accepting the trade, fire **anti-giving-up** — is there a faithful
   fact that *separates* the conflicting shapes so you could get BOTH? A trade is the answer only
   when no such separator exists (and the only alternative would be a kernel-identity fence). "They
   just conflict, take the net win" is a stop-claim that must survive the gate.

This relaxes **edit-acceptance only — NOT the done-bar.** "Are we done?" stays strictly per-shape
(seed ≥ tc where reachable; then oracle parity); never declare the deliverable finished on a good
aggregate while individual shapes lag. Log every accepted trade with its per-shape deltas + the
principle, so the loss is visible, not buried.

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

Independence is the asset — a gate that develops rapport stops re-examining. Whether you run these
as separate agents or as explicit self-checks, keep the discipline. **For the high-stakes
adversarial gates (adversarial-verify, anti-giving-up, results-referee, fact-integrity, overfit/TEST-
firewall, mechanism), use the VERBATIM prompts in `gate-prompts.md` (gates A–F) — do not improvise
their wording** (how you phrase an adversarial ask is how you bias its verdict; the scripted frames
bake in default-to-refuted, record-verdict-first, pin-the-SHA, and never-hand-over-your-conclusion).
Fill the slots, don't edit the frame. **The three MUSTS are anti-over-claim (adversarial-verify),
anti-under-claim (anti-giving-up), and the portfolio guard (overfit/TEST-firewall);** the rest fire
per their trigger.
- **Correctness-first**, every iteration (you bypass the autotuner's accuracy check).
- **Matched-lever A/B** with the full-verbatim oracle baseline (§3 loop, never pin block=1).
- **Flip-axis sweep** at mid + extreme of the axis a config-flipping cap flips on.
- **Adversarial verify** each claimed win: N independent skeptics prompted to *refute* it (default
  to "refuted" under uncertainty); kill on majority-refute. Hunt: noise/fabrication, measuring the
  wrong thing, overfit (check held-out val + an off-focus kernel), **kernel-identity smuggling** (a
  constant fencing exactly one kernel's shapes), metric gaming (loosened tolerance, dropped shapes).
- **Fact-integrity / divergence test:** for any new or changed fact, construct a kernel where the
  lazy proxy and the real property *disagree*; a fact that only works by curriculum luck fails
  (real runs falsified `num_load` and `num_reduction_ops` this way). A fact no branch reads is cut.
  **Test the THRESHOLD, not just the fact** — a faithful fact can be read unfaithfully: a quantity
  compared straight to a constant (`fact <= K`) is a disguised dtype/identity fence whenever that
  cut splits its value-set along dtype/kernel lines (the divergence case: two workloads equal in
  the real property but a different dtype/kernel must decide the same). Keep a dtype- or
  identity-correlated quantity a FACTOR inside a byte/occupancy budget, never the operand of a literal.
- **Anti-giving-up:** any ceiling/noise/stuck/done claim must survive a *fresh* oracle; a "source
  ceiling" is valid only if `oracle ≥ tc` AND the oracle run is verified real (not truncated/OOM).
  Before declining because "a gate also regresses peer X," check X's actual code branch (it may be
  *structurally* excluded) and measure **every** firing shape.
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

(The compile-time config-recorder "behavior oracle" has **two** roles, neither a pass/fail gate.
**(1) During-climb SCOPING** (the no-regression backstop, §3): BEFORE/AFTER an edit, diff the
normalized configs over the FULL active matrix to *derive* which cells changed — bench only those;
byte-identical cells are perf-invariant and skipped (sound only over all dtypes + robustness;
changed ≠ win; a codegen/source edit needs the `--triton` generated-Triton diff). A climbing edit
*should* change some configs — that's the worklist, not a failure. **(2) Post-climb PROOF** that a
*cosmetic/refactor* edit left every emitted config byte-identical, so frozen perf verdicts transfer.
Default bar = ZERO config diffs; but if the human pre-approves specific expected config changes, honor
that — treat those as allowed and verify only the rest are byte-identical. The generalized
dtype/robustness/sibling recorder is `_lab/harness/config_recorder.py` (`record` +
`diff`, `--triton` for non-selection-only edits); the fp32-only `run3_task1_seed_configs.py` stays the
deliverable config-oracle replayed to MEASURE the win. The task file names the recorder script.)

---

## 6. Orchestration, persistence, and NEVER STOPPING

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
  deferred hard-pile, push a beaten-tc shape toward oracle parity, or run the completeness critic.
  There is *always* a next move; the gates exist to find it. (The **only** real end is the human
  stopping the run or a hard external block like the GPU physically unavailable with no
  non-GPU-work left — and even then you keep doing the non-GPU work.)
- **Keep yourself alive mechanically.** Long oracles run foreground under the turn (§1). Never end
  your turn voluntarily with work outstanding; after every banked step, immediately pick up the next
  worklist item — don't return to narrate.
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

- Every result in them has **passed the adversarial gates** (§5) — reproduced by the results-referee,
  attacked by the adversarial-auditor, fact-checked for proxies. Your live context contains the
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

### 6.2 Machinery — default light
Match the machinery to the scope. GPU timing is foreground-serial on one GPU, so there is nothing
to parallelize *there*; the parallelizable work is the **non-GPU analysis** (field-diffs,
adversarial verification, completeness sweeps, reads).

**Default — single driving agent + a notebook + a ledger** (the gated source of truth, §6.1). Keep
a `*_notebook.md` reasoning trace (decisions + empirical *why* + tried-and-rejected list +
per-shape status table + the deferred hard-pile from §3) and append gate verdicts to a ledger. If
you approach your context limit, a fresh re-invocation reading the log resumes cleanly — no respawn
ceremony.

**One worker, no standing manager — but delegate freely to *ephemeral* helpers.** The worker stays
the driver and single thread of accountability; it spawns short-lived helpers that eat a
context-heavy task, return the distilled finding, and die. The two recurring archetypes:
- **code-investigator** — answers *"where/how does X work?"* by reading source/IR (e.g. "which
  branch emits this eviction policy?", "field-diff the oracle config across all 9 kernels"). No GPU.
- **perf-investigator** — answers *"why is A faster than B?"* via ncu / generated-Triton / a probe
  bench (e.g. "which hardware resource moves between these two configs?"). Returns the *mechanism*,
  not the dump.

The rule is *the dump never lands in the worker* (§-loop step 1). These are helpers, not managers:
no message-relaying, no acceptance-laundering, no surviving past their one task. Routine
hypothesis-forming stays in the worker (cheap, needs its intuition); only the heavy investigation —
and, when stuck, a divergent brainstorm — is offloaded.

**Use `Workflow` sub-orchestrations for the fan-out-able non-GPU work:** the seed-vs-oracle
field-diff across all kernels, adversarial verification of each win (the N-skeptic pattern),
completeness sweeps ("which cap did I not audit? which shape did I not measure on every dtype?").
Implement the §5 gates as workflow phases rather than a standing team.

**Escalate to the heavyweight hub/worker/respawn/baton machinery only for a genuinely multi-day,
open-ended run** (one agent surviving many context deaths). Otherwise it adds cost without safety:
if persistence (§6.1) is solid a plain re-invocation already survives a context limit; respawn
processes are exactly the silently-dying background class (§1's 13h stall); and a relay-hub burns
tokens and gives verdicts a place to get laundered. When in doubt, stay light and lean on §6.1.

The completeness critic is worth running near the end regardless: *what's missing — a dtype not
swept, a claim unverified, a cap not audited, a shape under the noise floor, a deferred hard-pile
item never revisited?* What it finds is the next round of work. **Never silently cap coverage**
(top-N, no-retry, sampling) without logging what was dropped — silent truncation reads as "covered
everything" when it didn't.
