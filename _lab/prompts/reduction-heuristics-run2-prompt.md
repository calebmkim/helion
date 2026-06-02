# Reduction-Heuristics — Run 2 (next-agent prompt)

> This is the prompt to hand the next autonomous agent. It REUSES the run-1 orchestration verbatim
> (manager-hub → persistent worker → specialized sub-agents, the hill-climb loop, the Step-0/Step-1
> setup, the lab notebook + ledger, the adversarial-auditor, never-stop, yolo). **Only the goals change.**
> `reduction-heuristics-wip.md` remains authoritative for the orchestration model.

---

You are the HUB / MANAGER for an autonomous performance-engineering run. Work from
`/home/calebkim/helion-new-heuristics/local`.

**Your orchestration model is unchanged from run 1, and `reduction-heuristics-wip.md` is its single source
of truth — READ IT IN FULL FIRST.** It defines, and you must follow EXACTLY:

- the **Execution model** (this harness does NOT allow nested sub-agents → the main session is the only
  spawner = the hub; one persistent worker continued in place; every helper spawned by you and relayed),
- the **Agents** roster (worker; code-investigator; perf-investigator; measurement-harness-verifier;
  results-referee; ledger-keeper; kernel-classifier; adversarial-auditor; harness-integrity agent) and the
  context-saving trust levels,
- the **Operating rules** (generalize-never-pattern-match; justify every branch; correctness-first;
  commit-don't-push; fixed precision; shared-machine GPU pinning),
- the **two products** and the **objective** (the measurement methodology for Product A and Product B,
  the oracle cache, the in-sample/VALIDATION/TEST firewall),
- the **Adversarial-auditor prompt** (anti-cheating, not a perf hardass), and
- the **Technical Appendix** (seed flow, bare-seed mechanism, silent-seed-drop, the ReductionFact sketch,
  the static classifier signals, persistent-vs-looped mechanics, benchmarking footguns, autotune budget /
  convergence traces, the machine-level reference + worktree import gotcha).

Also read `list_of_kernels.md`. IGNORE everything in `old/`. Remember the wip's rule: **all file:line
references anywhere are approximate — grep the symbol, don't trust the number.**

**WHAT THIS PROMPT SUPERSEDES / EXTENDS IN THE WIP:** it supersedes only the wip's **"Plan"** (run 1 built
the heuristic from scratch; you are NOT starting from scratch), it **extends** the wip's **"Scope &
curriculum"** (new shapes + new kernels below), and it **extends the wip's Adversarial-auditor** for Goal
3b's statistical claims (an anti-over-claim mandate — see Goal 3b; this is an orchestration extension, not a
replacement). The Execution model, Agents, Operating rules, the two products' methodology, and the
Technical Appendix are all still in force, unchanged.

## YOUR ROLE — do not deviate (identical to run 1)
- You are the ONLY agent that can spawn sub-agents, so you are the hub. DO NOT do the heuristic work
  yourself. Spawn ONE persistent worker sub-agent (give it the wip's "Worker" section onward as its brief,
  plus the run-2 goals below) and keep it alive across iterations via follow-up messages. Spawn a FRESH
  worker only when it is stuck or its context exceeds ~50% — at a clean iteration boundary, handing over
  the lab notebook (the notebook, not the worker's context, is the source of truth). Do not respawn every
  iteration.
- The worker CANNOT spawn its own sub-agents. Whenever it needs a code-investigator, perf-investigator,
  results-referee, adversarial-auditor, harness-integrity check, or measurement-harness-verifier, YOU
  spawn that helper and relay the result back.
- NEVER let the worker stop — "stuck / converged / a wall / can't get the env green" is never an exit,
  only a prompt for your next move (different angle, fresh worker, keep grinding for hours). The only
  things you surface to me are the non-blocking "looks converged" flag and passive ledger breadcrumbs;
  you never stop or wait for input. I am NOT available — answer the worker's questions yourself.
- The importance of the above CANNOT be stressed enough. You must *not* let the worker stop, and you
  yourself must not stop either. There is *NO* token limit. Please keep on trying.
- NEVER ask me for permission to run a command. I will not be available; you are in yolo mode for a reason.
- Safe ONLY because acceptance is independently gated: nothing enters the champion unless the
  results-referee reproduces it and the correctness/regression gates pass. Reject (don't accept) cheated
  or unreproducible results; reject (don't loosen) correctness failures — then keep going.

## OPERATIONAL — dedicated git worktree + branch, prove your wiring FIRST
Run 1's deliverable is the **v8 `TritonReductionHeuristic`**, complete for all 9 forward inner-reduction
kernels, in worktree `/home/calebkim/helion-new-heuristics/wt-reduction`, branch
`reduction-heuristics-autotuner`. You build ON TOP of v8. (**IGNORE `wt-clean-pr`** — it is a PR-only clean
export that does NOT contain `_lab/` or `logs/`, i.e. none of the intermediate state you need; never read
the deliverable or any intermediate state from it.)

- **Create a NEW git worktree + branch off the tip of `reduction-heuristics-autotuner`** (the v8
  deliverable; confirm HEAD with `git log` — do not trust any hard-coded SHA). This brings the v8 heuristic
  code AND all of `_lab/` (the ledger, notebook, HANDOFF, FINAL_REPORT, harness scripts) into your
  worktree. Leave `wt-reduction` untouched as your read-only reference for the v8 code + `_lab/` + `logs/`
  (do NOT use `wt-clean-pr` for any of this — see above). Seed your ledger's champion to the v8 state so v8
  is your starting floor.
- **Re-prove the worktree wiring before trusting any edit** (this silently bit run 1): assert
  `helion.__file__` is inside YOUR worktree (`helion` is an editable plain-path install pointing at the
  ORIGINAL checkout, so use `PYTHONPATH=<your-worktree>` from a non-original cwd), change code and confirm
  the generated Triton changes, and confirm a tritonbench operator edit takes effect (separate import-hook
  editable that `PYTHONPATH` may NOT shadow → operator-level edits go in the original checkout). Then re-run
  the rest of Step 0 (bare-seed mechanism) and a Step-1 sanity check. Nothing downstream is trusted until
  green. The MACHINE is known-good (conda interpreter, 4× H100 — pin `CUDA_VISIBLE_DEVICES` to an idle GPU
  after `nvidia-smi`), but that does not prove YOUR worktree's wiring.
- Measure via TritonBench (`do_bench`), not hand timing. Commit early and often; NEVER `git push`.

## REQUIRED READING before you touch anything — run 1's hard-won knowledge
Run 1's adversarial-auditor caught real cheats/confounds SEVEN times. Do not re-make those mistakes. From
your new worktree's `_lab/`, read **in this order**, and brief the worker on them:
1. `HANDOFF.md` — especially **§4 (TRAPS)**. Non-negotiable, and they apply to EVERY new A/B you run below:
   **matched-lever A/B** (hold all other levers equal; A/B vs the best SIMPLE alternative, never the
   catastrophic default), **the oracle is a BUNDLE** (re-bench the FULL VERBATIM config; never
   isolate-a-lever-and-re-pair-it — that fabricates an unmeasured config), the **do_bench noise floor**
   (tiny-M / sub-25µs shapes), the **persistent-seed round-trip fix**, and **welford-divides-N** (which
   Goal 1 makes obsolete).
2. `FINAL_REPORT.md` — the heuristic structure + per-branch why, the per-kernel G table, the verified
   ceilings/residuals (§6), and the rejected ideas (§7).
3. `ledger.json`, `notebook.md`, `HUB_LOG.md` — durable state, the worker's reasoning trace, the run arc.
4. Code maps: `step2_code_map.md`, `t2_code_map.md`, `codegen_knob_map.md` (treat the last as a DRAFT plan,
   not a completed result — see Goal 2).

**Keep v8 as the floor.** Do not regress the 8 working kernels while chasing new gains (the wip's per-kernel
">10% referee-confirmed regression" backstop is in force across the whole curriculum, including every new
shape and new kernel you add).

## Cadence, dependencies & the TEST firewall (read before planning the goals)
- **Hill-climb cadence is unchanged** (wip §3): Product A measured every iteration; Product B on the
  every-5 cadence (relevant from Goal 3 onward); adversarial-auditor on every accepted improvement;
  harness-integrity periodically + on any suspicious result. New shapes (Goal 4) and new kernels (Goal 5)
  fold INTO Product A and are subject to the same >10% per-kernel regression backstop.
- **Hard dependency:** Goal 3b ("beat max-effort") is gated on Goal 2 landing. On the forward kernels the
  v8 seed is already at the block_sizes ceiling (seed/oracle ≈ 1.007); the unseeded max-effort oracle
  additionally searches the codegen knobs, so a block_sizes-only seed CANNOT out-perform a full max-effort
  search by construction. Do not start 3b until Goal 2 gives the seed codegen-knob bundles to inject.
- **TEST firewall** (extends the wip's read-once discipline). The TEST set stays sealed except for **two
  pre-authorized re-reads, done together in one pass, then re-locked:** (1) **welford TEST** — its kernel
  source was wrong (Goal 1), so the v7 welford TEST numbers are invalid; (2) **rms_norm TEST G (≈0.828)** —
  it has no raw-log backing (run 1's biggest evidence gap), so regenerate it. No other kernel's TEST column
  is re-read. **New development shapes go into a clearly-labeled NEW in-sample split (`in-sample-v2`), never
  into TEST.** Goal-1's prime-N welford check `(…,1543)` is a one-time correctness canary, NOT an in-sample
  perf target.

---

# RUN-2 GOALS — TWO PHASES

The six goals split into two phases. **Phase I builds the best, *general* Product-A seed; Phase II measures
Product B on that finished seed.** Settle Phase I before starting Phase II, so the expensive Product-B
experiments run ONCE — on a curriculum and a heuristic you are NOT about to change. (Rationale: if you ran
Goal 3 first and then added shapes/kernels, you'd have invalidated it and would re-run the most
compute-heavy goal; and Goal 5's generality reworks change the seed itself — i.e. Product B's seeded arm —
so measuring convergence/beat before the heuristic is settled measures an artifact you're about to modify.)

**Phase I — build the general seed (Goals 1, 2, 4, 5).** These shape the SAME artifact (the heuristic + the
curriculum it is judged on) and reinforce each other. Order within Phase I:
- **Goal 1 first** — the welford correctness fix is foundational and bounded.
- **Then Goals 2, 4, 5 interleaved:** start Goal 4 (new `in-sample-v2` shapes) right after Goal 1 so Goal 2
  is tuned against realistic shapes; bring in Goal 5 (new kernels) alongside Goal 2 as the heuristic
  stabilizes, so each new kernel immediately tests structural generality — and may send you back into Goal 2
  if a band or a codegen-knob rule doesn't generalize (this is expected; 2/4/5 converge together).
- Phase I is "done" when the seed and curriculum stop changing — NOT when Goal 5 is "perfect." Never stall:
  if one band resists generalization, keep grinding (the never-stop rule), but do NOT hold Phase II hostage
  to an open-ended generality chase — begin Phase II once the heuristic is structurally settled.

**Phase II — measure Product B (Goal 3): 3a then 3b.** Run on the FINAL Phase-I seed + curriculum. 3b ("beat
max-effort") is the compute-heavy stretch and additionally needs Goal 2's codegen-knob bundles (see the
Cadence note) — don't rabbit-hole on it at the expense of a solid Phase I. Run 1's 1.94× quick-budget
Product-B result is the floor if Phase II is only partially reached.

**Goal 6** is small hygiene — fit it in opportunistically (any phase).

## Goal 1 (Phase I) — Fix the welford kernel source, then re-derive its heuristic on the CORRECT kernel
`examples/welford.py` has a **correctness bug**. It uses `Tn = chunk.size(-1)`, which returns the constexpr
**tile width**, not the count of VALID columns — so for the last tile when the block size does not evenly
divide `n`, the per-chunk mean / count / M2 are computed wrong (`FINAL_REPORT §6` calls this the
"Tn-padding bug"; it is why prime-N welford collapses to G≈0.08).

- **The fix is one line.** Helion masks out-of-bounds loads with `other=0`, so `sum_x`/`sum_x2` are already
  correct over the valid columns — **only `Tn` is wrong** (the divide-by-padded-width corrupts the mean/M2
  and the count). Replace it with the masked valid count: `Tn = (tile_n.index < n).sum()` (note `.index`,
  singular — the canonical Helion mask idiom; grep for the `tile.index < bound` pattern in
  `helion/language/loops.py` and `examples/segment_reduction.py` to confirm, line numbers drift). Verify on
  this kernel that OOB lanes are indeed zero-filled, then confirm correctness on well-factored, odd, AND
  **prime** N (e.g. 1543 — the prime-N canary is a deliverable requirement; prime N must now be both
  correct AND fast). Make sure to commit after fixing this.
- **This was a real cascade, not a cosmetic bug.** Because the masked combine tile had to be a power-of-2
  **divisor** of N to keep the buggy `Tn` correct (it degenerated to width 1 at prime N → catastrophically
  slow), run 1 introduced the `largest_pow2_div(N)` constraint and the whole `is_structured_combine` /
  `apply_block_ids` Band-C machinery. **With the source fixed, the divisor constraint and the prime-N cliff
  are gone** — the combine tile could perhaps become a normal byte-capped `min(N, cap)` like any other, depending
  on what works during the hillclimbing.
- **Re-derive the welford heuristic on the corrected kernel.** Welford genuinely has TWO `hl.tile(n)` loops
  over the same axis — a reduction (combine) pass and an elementwise (apply/normalize) pass — and seeding
  the two block sizes differently is legitimate. So you will still need *some* mechanism to seed the apply
  pass WIDE (the general T2 path floors non-reduction blocks to 1, which is catastrophic for the apply
  loop). But **simplify and generalize** the Band-C treatment: strip the bug-artifact divisor logic, keep
  only what a *correctly-written* two-pass kernel needs, and decouple the combine/apply caps from the
  now-deleted correctness constraint. If `is_structured_combine` ends up fit to exactly one kernel, that is
  a generality flag for Goal 5 — fix it there (find the generalizable two-pass workload property), don't
  leave it as a one-kernel special case.
- **Re-measure everything welford-touched, and re-validate the oracle.** The v8 welford oracle cache was
  built under the buggy kernel and its accuracy gate **presumably** rejected non-divisor combine configs
  (the buggy kernel produces wrong outputs → fails allclose), so the oracle was **likely** artificially
  constrained. Confirm this rather than assume it: re-run a fresh quick-autotune oracle on a welford
  in-sample shape (e.g. `(262144,2048)`) under the CORRECTED kernel and compare to the v8 oracle latency for
  the same shape — if they diverge (>~3% beyond do_bench noise), the oracle was confounded and must be
  fully re-run. Then re-measure welford in-sample G and re-read the welford TEST column (a pre-authorized
  re-read; see the firewall note). Expect welford to go from the worst kernel (TEST G=0.396) to a normal
  one, lifting the overall O.
- The welford fix is a genuine correctness fix and **ships with the deliverable** (it is independent of the
  seed; it also fixes the un-seeded default, which is wrong at prime N today).

## Goal 2 (Phase I) — Seed the codegen knobs (lift the bare-seed Product-A perf to parity-or-better with torch.compile)
Internalize this framing: **a seed `Config` flows through the exact same `normalize()` + flat-encode path
as any autotuner config** (`Config(**seed)` → gen-0 population). There is NO API barrier — the heuristic
confining itself to `block_sizes`/`reduction_loops`/`num_warps`/`num_stages` was a **design choice**, so
"autotuner-only" is a misnomer. Your job: for each codegen knob, find the **workload property** that
separates the configs that want it ON from those that want it OFF, encode it as a `ReductionFact` field,
and key on it. "No single global setting wins" ≠ "no separating fact exists." This is primarily a
**Product-A** task — it makes the bare seed itself match/beat torch.compile — and the codegen-knob bundles
it produces are also what makes Goal 3b winnable.

- **`pid_type` — LOCKED to `flat` for all forward kernels; do not re-litigate.** Run 1 rejected
  persistent/interleaved with a gold-standard matched-lever A/B (flat dominates 1.5–4×; the oracle's pid
  pick is a confounded passenger); the numbers are persisted in the ledger. Just make the constant explicit:
  emit `pid_type='flat'` with the justifying comment + the run-1 ledger reference (a principled constant is
  a valid degenerate heuristic). This lock covers any NEW forward kernel you add in Goal 5 too. Only a
  future Band-D / backward / grid-bound kernel could reopen it — and that would be a separate branch/fact,
  not a re-tune of the forward set. However, if later on, you find that pid_type needs a genuine heuristic
  for new kernels/shapes, of course you can create a knob for it and adjust it, just like any of the other knobs.
- **`num_sm_multiplier` / `maxnreg` — NOT applicable to forward kernels.** They require a persistent
  `pid_type`, which is locked to flat, so they are silently dropped here. Skip them (they are Band-D /
  future territory). Again, if you find for perf, they are needed for the kernels we are testing, then feelf ree
  to introduce them as knobs. This is just the current plan.
- **`load_eviction_policies` / `indexing` (tensor_descriptor) / `num_stages` / `range_*` (and any other
  axis you find in the LIVE spec) — GENUINELY OPEN; the prior "no seedable win" is unproven hand-waving.**
  Unlike pid, these were NEVER run through a completed matched-lever A/B — `codegen_knob_map.md` is a DRAFT
  plan that was never executed, and the welford "0.968 seedable" was a coupled-bundle overstatement
  (corrected in §6). The eviction headroom is real (run-1 notes +7–23% on small-N) and was never seeded.
  Treat the rejection as **provisional** and pursue these knobs hard:
  - Extend `ReductionFact` with the separating fact. Eviction likely keys on **reuse / cache-residency**,
    not a global rule: a single-pass streamed load wants `first`, a reused operand wants `last`; rms_norm
    picks `last` at small N but `first` at huge N at the *same* `num_load`, implying an rnumel/byte-residency
    fact. `num_stages` is a global scalar the seed already sets (=1); make it workload-keyed where evidence
    supports (run 1's notebook flagged an untried `num_stages=3` for long_sum's largest rows). Co-design
    fact ↔ branch.
  - **Mechanics/traps:** `indexing` and `load_eviction_policies` are per-op LISTS that `normalize()` does
    NOT length-validate — build them at EXACTLY `spec.indexing.length` / `spec.load_eviction_policies.length`
    with only valid choices (read the length off the LIVE spec).
  - **Discipline (MANDATORY — this is where run 1 fell short):** (1) matched-lever A/B — vary ONLY the
    candidate knob, hold all else equal; (2) re-bench the oracle's FULL VERBATIM winning config as the
    baseline, and to split seedable-from-oracle-only also measure the oracle's block_sizes at DEFAULT
    codegen knobs then add only the candidate knob — NEVER isolate a knob from the oracle bundle and re-pair
    it; (3) **PERSIST the raw A/B numbers (both arms' distributions, N, median, spread) to the ledger for
    EVERY knob decision, accepted OR rejected**, keyed by (kernel, knob, shape) — this is the receipt the
    auditor will demand; (4) **do NOT fit to the oracle's picks on your tuning shapes** — form the
    workload-property hypothesis first, then validate that it predicts the knob choice on a DISJOINT set of
    shapes; (5) if a knob's best value is constant across all workloads, that is allowed ONLY with a
    principled physical reason (like pid=flat), not "it matched the median in-sample."
- You can touch ALL the knobs, and you SHOULD pursue parity-or-better with torch.compile on every axis.
  **"No excuse" means no excuse to skip the systematic experiment — it does NOT mean fabricating a win that
  the data doesn't support.** A matched-lever A/B that finds no win is an honest, valuable null result:
  record it and move on.

## Goal 3 (Phase II) — Product B: budget reduction, then beat max-effort autotune (run after Phase I is settled)
See the Cadence note + the two-phase rationale above: Goal 2 (and a settled, general heuristic) is the
precondition — measure Product B once, on the final seed.

### 3a — Budget reduction at max effort (reliable). Quantify the budget seeding saves at equal final quality.
Two cheap experiments on a handful of representative shapes per kernel (NOT the full sweep). Budgets,
operationally (wip Appendix): **quick** = LFBOTreeSearch initial_population 30 / copies 2 / max_generations
5; **full** = 100 / 5 / 20 (drive via `HELION_AUTOTUNE_MAX_GENERATIONS`, and `HELION_FORCE_AUTOTUNE=1` for a
real search; log per-generation best with `HELION_AUTOTUNE_LOG`).
- **(a) seeded-QUICK vs unseeded-FULL** — does the seed at the quick budget already match the unseeded
  full-budget optimum? The budget gap is the savings.
- **(b) Convergence curves at full effort** — on the SAME instrumented full-budget runs (one experiment,
  two axes), plot best-perf-so-far vs **generation index** AND vs **wall-clock**, SEPARATELY. (A seeded
  curve can reach target in fewer *generations* but more *wall-clock* — LFBO explores around the extra good
  config — so separate axes are mandatory to see the real trade-off; this is run 1's honest caveat.) Report
  generations/time to reach 95–99% of the final optimum; the seeded curve should start higher AND converge
  sooner.
- Fair-re-bench final configs; pin GPUs. **Pilot seeded-FULL vs unseeded-FULL on 1–2 shapes first** — run
  1's Product B was quick-only (time-to-95% median 1.94×); do not assume that number transfers to full
  budget without checking.

### 3b — Beat max-effort autotune (aggressive stretch; compute-heavy; depends on Goal 2).
The max-effort autotuner is a stochastic, bounded search (LFBOTreeSearch) and can under-sample a globally
best coupling (e.g. block_sizes × eviction × num_stages) run after run — though not necessarily on every
kernel. A designed seed encodes structure the search reaches only by luck.

- **Strategy — aggressive multi-seed portfolio, designed BEFORE you run the experiment.** Inject several
  structurally-distinct seeds, EACH embodying ONE falsifiable hypothesis (write the hypothesis + the seed +
  the outcome that would prove/disprove it, before running). Examples: persistent-aggressive (full R_BLOCK,
  w32); looped-conservative (small chunk, w4); high-warps-streaming; the **codegen-knob bundles from Goal
  2** (e.g. eviction=`first` vs `last` paired with their block-size variant); the v8 seed as baseline.
  Derive these from Goal 2, run-1's rejected-ideas/ledger, and principled theory — **NOT** from observing
  which configs the unseeded search finds in your 3b runs (fitting the seed to the outcome you're trying to
  beat is p-hacking and is banned). Post-hoc you may analyze *why* a seed won/lost, but you may not re-tune
  the portfolio and re-run.
- **API:** VERIFY that `compiler_seed_configs` already injects MULTIPLE configs into gen-0
  (`config_generation.py` iterates all of `config_spec.compiler_seed_configs`) and that user
  `autotune_seed_configs` accepts a Sequence. The ONLY limitation is that each heuristic emits ONE config
  via `get_seed_config`. Minimal change: add `get_seed_configs() -> list[Config] | None` to the heuristic
  base class (default returns None); in `compiler_seed_configs`, use it when present else fall back to
  `[get_seed_config()]`; verify `dedupe_configs` preserves structurally-distinct seeds. If the plumbing
  turns out NOT to accept multiple, patch it minimally so it does.
- **Measurement protocol (critical — the autotuner is nondeterministic; you cannot claim a beat from one
  run). Three phases, no goalpost-shifting:**
  1. **Pilot N.** Run N ∈ {3, 5, 10, 20} on 1–2 representative shapes (cover both an at-ceiling Band-A
     kernel and a harder one — Band-B or welford), measure how much best-of-N and its rank move with N, and
     pick the smallest N whose best-of-N is stable. **COMMIT to that single N for ALL subsequent 3b
     measurements on every kernel/shape** — N is part of the stated confidence threshold and must not vary
     per kernel (N=3 is too small to characterize a stochastic best-of-N).
  2. **Pre-register shapes, then run the stochastic comparison.** Fix the representative shape list (≥1 per
     kernel, keep it small to avoid breadth-dilution) BEFORE running; report the FULL pre-registered sweep,
     not a post-hoc cherry-picked subset. Run seeded best-of-N and unseeded best-of-N (independent random
     seeds / GPUs). Define "beat" PROBABILISTICALLY and upfront: `P(seeded best-of-N ≥ X) > P(unseeded
     best-of-N ≥ X)` at a stated confidence, X e.g. 99% of the unseeded-full oracle — NOT a single
     best-of-3 point estimate. Report best-of-N, median-of-N, and spread for both arms.
  3. **Fair re-bench every winner:** fresh process, FULL verbatim config, median-of-7, pinned idle GPU;
     never re-pair an isolated lever.
- **Exclude noise-floor shapes** (sub-25µs / tiny-M — lift M or drop them; any "beat" there is
  undefendable). **Matched knob sets:** compare like with like — the seed must carry the codegen knobs too
  (Goal 2), or both arms are handicapped to the same knob space.
- **Calibration:** the winnable beats are the hard-to-sample codegen-knob couplings (post-Goal-2), Band-B
  (kl_div/jsd), welford (post-fix), capped-budget regimes, and new kernel families — NOT generic forward
  Band-A, which is at ceiling.
- **Auditor (expanded mandate for 3b — see the supersede note):** beyond the standard anti-cheating checks,
  the auditor is anti-lucky-run / anti-over-claim here and will FAIL the claim (not minor-note it) unless:
  N independent runs with the pre-registered N and shape list; best-of-N + median-of-N + spread reported;
  probabilistic beat definition fixed upfront; noise-floor shapes excluded; matched knob sets; fresh-process
  full-verbatim median-of-7 re-bench of every winner.

## Goal 4 (Phase I) — Add representative real-AI-workload shapes
Start with the additions below (these are a starting point — refine and extend as you learn; if you find
other high-priority regimes, e.g. unusual M-ranges or batch effects, add them with a one-line rationale).
Add them to a CLEARLY-LABELED new **`in-sample-v2`** split in `list_of_kernels.md`; this unblocks
development against real regimes WITHOUT touching the read-once TEST set (see the firewall note — promoting
any currently-held-out shape to in-sample is a deliberate, documented change).
- **rms_norm / layer_norm:** small-M `(256,4096)`, `(512,8192)`; Llama/Mistral hidden dims `(256,5120)`,
  `(1024,2560)` (closes the tiny-M TEST drag).
- **welford (most under-sampled — in-sample is 100% M=262144, all well-factored N):** add M-variation
  `(512,4096)`, `(8192,4096)`; non-pow2/odd N `(262144,2560)`, `(262144,5120)`. (The prime `(262144,1543)`
  is Goal-1's correctness canary, NOT an in-sample perf target — keep it separate.)
- **softmax:** non-pow2 `(4096,3072)`, long-context `(8192,32768)`.
- **cross_entropy / kl_div / jsd:** real vocabs — Llama-2 `32000`, GPT-2 `50257`, Llama-3 `128256`, Qwen
  `151936` (× realistic batch·seq). These currently live only in the held-out set; bringing them in-sample
  unblocks development against real regimes.
- **sum / long_sum:** intermediate pooling regimes `(256,4096)`, `(512,8192)`, `(32,65536)`, `(256,262144)`.

## Goal 5 (Phase I) — Add new kernels as adversarial GENERALITY probes (you write + validate them)
Adding shapes (Goal 4) tests *shape* generality; it does NOT test *structural* generality. Run 1's three
byte-cap bands (multi-load, Band-B accumulator, structured-combine) are each currently fit to exactly ONE
kernel — the auditor's standing concern (HANDOFF §1). Use new kernels as DISCOVERY probes for whether each
band generalizes beyond its originating exemplar.

- **What to add (1–2 new kernels exercising each band with a DIFFERENT, structurally-distinct identity):**
  multi-load band — **log_softmax** and/or the online **logsumexp** from Goal 1's cross_entropy fix exposed
  as a standalone kernel; Band-B accumulator — a second heavy-epilogue loss kernel; structured-combine — a
  second reduce-then-apply kernel. They must be STRUCTURALLY distinct in the facts the heuristic gates on
  (different `(num_load, num_tiled_accumulators, num_reduction_ops, is_structured_combine)` profile, or a
  genuinely different code path) — a cosmetic wrapper (e.g. log_softmax that is just softmax + a final log)
  that exercises the identical path is NOT a valid probe. Propose each kernel's `ReductionFact` profile to
  the auditor and justify why it tests the band differently before implementing.
- **Implementation site:** add the kernel to `examples/` (parallel to `examples/welford.py`), wire it into
  the benchmark path (`benchmarks/run.py` KERNEL_MAPPINGS + a tritonbench operator with a
  `torch_compile_<op>_default` baseline — operator-level edits go in the ORIGINAL checkout per the wiring
  note), add a `_lab/harness/measure_g_<kernel>.py`, and put its shapes in `in-sample-v2`. Correctness-gate
  against an eager reference before any perf claim.
- **The rule, with teeth (the wip's "generalize, never pattern-match" applied):** if a band fails to
  generalize on the new kernel — its best seed is >10% off the band recipe on the new kernel's in-sample
  shapes — it is YOUR responsibility to **rework the heuristic to be genuinely general** (find the
  distinguishing workload property, extend `ReductionFact`, re-key the branch). You may NOT carve a
  kernel-identity branch or a shape-window that fences exactly one kernel; you may NOT accept an unjustified
  exception. The ONLY acceptable "can't generalize" outcome is a PROVABLE kernel-structure/source limit
  (documented as out-of-scope, not as a band-generality claim). The auditor has veto power: it must REJECT
  any "band X is general" claim the new kernel contradicts, and reject any kernel-identity smuggling.

## Goal 6 (any phase) — `device_ir` robustness + cleanups (small hygiene)
- **`_count_reduction_workload` uses general rules, not airtight ones** (the user's point #5 — NOT a
  correctness bug; at worst the seed gets wrong info → a bad seed). `last == size_hint` is used at two
  sites: in `_last_dim_is_reduction` (feeds `num_tiled_accumulators` → Band-B gating; a "lucky"
  non-reduction last-dim equal to the reduction extent would over-count accumulators and MIS-ROUTE the seed
  Band-A↔Band-B — the higher-severity site, and if it ever bites it is the seed's bug to fix with a more
  precise heuristic), and in the dtype/itemsize detection (lower severity — it has a fallback search). A
  principled `block_id`-match is NOT cheaply available (tensors don't carry block_id provenance; only the
  `ReductionLowering` does). So do the user's minimum: **add an acknowledging comment** explaining the
  int-equality assumption and its mis-routing failure mode, and extend the existing symbolic-fallback
  (already used for dynamic dims) to the static dtype site for symmetry.
- **Atomics-as-stores is ALREADY documented** (the scope-assumption comment is already in `device_ir.py` at
  the `_MEMORY_OPS[2:]` counting site) — do not redo it.
- **Regenerate the one unbacked number:** rms_norm's TEST G (≈0.828) has no raw log (run 1's biggest
  evidence gap) — regenerate it in the same pass as the welford TEST re-read (see the firewall note), then
  re-lock TEST.
- **Anything else you notice** — there may be cheats/gaps not listed here; don't treat this list as
  exhaustive.

---

## Scope notes / decisions already made (do not re-open without strong evidence)
- **Source rewrites — allowed for welford and the cross_entropy online-logsumexp variant ONLY.** welford:
  the Goal-1 fix. cross_entropy: ADD an online-logsumexp implementation alongside the existing one
  (softmax-style — keep BOTH; the new variant removes the wide-row re-read that caps it at G≈0.54, a direct
  Product-A win like Goal 1). **Do NOT rewrite split-K / the >2²⁰ long_sum looped tail** — it is a
  structurally different reduction (cross-CTA split + atomic/second-kernel combine) needing its own
  fact/branch, a separate future workstream. You may LIVE with that gap, **but you must provably attribute
  it:** measure the best CORRECT single-kernel looped config and show it still loses to torch.compile's
  multi-stage split there (i.e. prove it is a source/structure ceiling, not a config the seed missed).
- **Precision stays fp32** for this whole run (assert it on every benchmark call; the heuristic reads
  `dtype`/`itemsize` so it stays dtype-general). bf16/fp16 expansion is deferred — expected to be
  straightforward once the technique holds.
- **Backward / outer reductions (Band D) remain DEFERRED** (not in this run's scope).

Begin: read the wip and `list_of_kernels.md`, read run-1's `_lab/` (HANDOFF §4 traps first), set up your
new worktree off the v8 tip, prove your wiring (Step 0/1), seed your ledger champion to v8, then start on
Goal 1.
