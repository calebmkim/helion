# Hub Kickoff Prompt

> Paste the section below (everything under the rule) to the fresh hub agent. The two READ-FIRST files give
> it the method and the machine; without them it has no filepaths.

**READ FIRST, in order:**
1. `/home/dev/local/helion-reduction-heuristics-run2/_lab/prompts/reduction-heuristics-standalone.md` — the
   work-order: objective, the two-tier oracle bar, operating rules, the agent roster + orchestration model,
   the failure catalog, and the 3-phase plan. (Path-free by design.)
2. `/home/dev/local/helion-reduction-heuristics-run2/_lab/prompts/server-specific-setup.md` — THIS machine:
   interpreter, GPU, the two-checkout import gotcha, git facts, canonical commands, the benchmarking-wiring
   state, and the verified orchestration primitives. (All the concrete paths live here.)
3. `/home/dev/local/helion-reduction-heuristics-run2/_lab/prompts/shapes_v3_draft.py` — the kernel set +
   train/val/test/robustness/transfer splits (the single source of truth for which shapes you tune/report).

---

Please read `/home/dev/local/helion-reduction-heuristics-run2/_lab/prompts/reduction-heuristics-standalone.md`
in full, then `_lab/prompts/server-specific-setup.md` (same dir) for this machine's paths/commands and
`_lab/prompts/shapes_v3_draft.py` for the curriculum.

**I want *you* to act as the hub.** Follow the doc's plan order, but **delegate Step 0 (worktree + prove your
wiring) and Step 1 (harness sanity) to a one-time setup subagent that reports back "green"** — do **not** run
the builds/benchmarks/greps in your own context (build and benchmark output is bulky and would burn your
window before the real work starts; your context is a scarce resource — see the doc's "switchboard, not a
thinker" rule). Once setup is green, stand up the team and spawn a persistent worker. **Spawn the worker — and every agent — with `model:"opus"`.** Verified
live: a team-member spawn with the model omitted silently comes up on **opus-4-7** (a real downgrade, not
inheritance); `model:"opus"` resolves to the full `us.anthropic.claude-opus-4-8[1m]`. This matters most on
*respawn* — and note the **respawn launch is a CLI call where the `opus` alias resolves to opus-4-6**, so that
one must hardcode the literal `us.anthropic.claude-opus-4-8[1m]` string (see § Respawn lifecycle).

**Your job has two halves that are the same job: keep the worker running, and gate every claim rigorously.**
The endless churning is only *safe* because acceptance is independently gated — nothing enters the champion
without the results-referee reproducing it and the relevant auditors passing it. A worker that runs forever
emitting ungated junk is worse than one that pauses. So: never let it stop, *and* never let an unverified
result through.

**Never let the worker stop. I cannot stress this enough.**
- If it asks you a **technical** question ("why is A slower?", "where does X live?"), do **not** answer it
  in-context — tell the worker to **DM the investigator** (that is what the peer topology is *for*; answering
  technical questions in your own context bloats the hub and adds a needless relay hop). Answer only
  **orchestration** questions
  (sequencing, which split, when to freeze), in one line. For anything genuinely off the doc's decision table,
  spawn a one-shot **orchestration-advisor** and execute its recommendation — don't deliberate in your own
  context. (See the doc's "switchboard, not a thinker" rule + decision table.)
- Do **NOT** ask me. I am not available. You are in "yolo" mode for a reason — take advantage of it. Run
  anything you need to run yourself (or better, *delegate* it — you run almost nothing in your own context);
  never ask my permission. (Yolo is local only: commit early and often, but **never `git push`** — a human
  handles that.)
- If the worker says it is stuck, out of ideas, at a ceiling, that a gap is "just noise," or that there's "no
  clean rule" — that is **never an exit**. It is a prompt for your next move. Spawn the **anti-giving-up
  agent** (the purpose-built tool for exactly these claims — it hands back the next experiment to run: which
  oracle to run, which M to lift, which fact to add). Or attack from a different workload property. Or
  respawn the worker fresh, handing it the notebook as context (with `model:"opus"`). A fresh context is the
  single most effective way to get unstuck.
- It is imperative that the worker keeps hill-climbing.

**The hill-climb reaches parity when `seed ≈ oracle` on every measurable shape (verified by the auditors).**
Parity means the seed matches the autotuner's own best config (the oracle) within ε — *not* "beats
torch.compile." This is *always achievable*, so the run can terminate honestly.
- "Every measurable shape" = the `train` shapes (confirmed on val/TEST at freeze). Robustness canaries are
  **correctness-only** — never perf-tune them. TEST is read **once**, at the very end — never hill-climb
  toward it.
- A shape where `seed ≈ oracle` but the oracle itself loses to torch.compile-default is **DONE** — that's a
  kernel-source ceiling, not a heuristic failure. "Never give up" does **not** mean grinding forever on an
  unwinnable shape; it means never declaring a ceiling while `seed < oracle`. Make the anti-giving-up agent
  verify every ceiling claim (the oracle must be real and `oracle ≥ tc_default`) before you accept it.

**Parity is a *milestone, not the end* — it triggers a strict sequence. Log each boundary to the
hub-log/ledger so it's unambiguous which mode you're in:**

1. **`=== PARITY REACHED ===`** — seed ≈ oracle (within ε) on every measurable shape, with the per-shape
   seed/oracle table. This ends *climbing toward the oracle* — not the run.
2. **Phase 3 → `=== DELIVERABLE FROZEN + VALIDATED ===`** — freeze the seed, run Product B
   (budget-reduction), read TEST once, report the train↔test gap. **This banks the real deliverable. Do this
   *before* any overtime — never sacrifice a finished, validated champion to the open-ended chase below.**
3. **Phase 4 — BEAT-THE-ORACLE OVERTIME → `=== ENTERING BEAT-ORACLE OVERTIME ===`.** *(This explicitly
   supersedes the doc's "beat max-effort is out of scope for Phase 3" — in overtime, you pursue it.)* Now the
   goal flips: find seed configs that **beat** the autotuner's best — configs its bounded, stochastic search
   under-samples, or couplings outside its search space entirely (e.g. the codegen-knob bundles). This goal
   is **deliberately open-ended and probably unreachable — which is the point: there is no "done" here, so
   there is never an excuse to stop.** Grind it indefinitely.

**Overtime rules (the gate gets *stricter* here, never looser):**
- The frozen Phase-3 champion is **banked and never discarded.** A Phase-4 "beat" enters the champion only if
  it clears a harder bar than parity did.
- A real beat = the seed config is faster than a **fresh, full-effort, best-of-N oracle** (not a quick or
  stale one), re-benched in a **fresh process, median-of-N, pinned GPU, noise-floor shapes excluded.** Beating
  a weak oracle is not a beat.
- Derive candidate seeds from **theory and the answer-key diff — never from configs you observed the search
  find** (fitting the seed to the outcome you're trying to beat is p-hacking, and it's banned). The
  adversarial-auditor's expanded anti-lucky-run mandate is in force on every beat claim.
- Any accepted Phase-4 win **re-opens the freeze**: re-run Product B and re-validate against a fresh oracle
  before re-freezing. The deliverable is always a clean, validated state — never a mid-overtime snapshot.

**So your terminal logic is: never let the worker stop climbing until `seed ≈ oracle` everywhere; then freeze
and bank a validated deliverable (Phase 3); then enter open-ended beat-the-oracle overtime that, by design,
never ends.** Besides orchestrating the agents as the doc describes, **your main role is to make sure the
worker never stops** — before parity it climbs, after parity it banks-then-beats. It has no excuse to stop at
any point.

---

## Respawn lifecycle — how YOU (the hub) survive your own context limit

**This is mandatory, not optional.** A hub that lives in a single context will fill that context and **stop
mid-run, before the work is done** — and an "indefinite Phase 4" is *physically impossible* for a hub that
can only live as long as one window. So the hub must be able to hand off to a fresh successor and keep going
indefinitely. The mechanism below is verified live on this harness (2026-06-03) — these are facts, not
suggestions:

**Why it must be a fresh process (not a compaction, resume, or self-spawn):**
- Teammates are **`in-process`** — they run *inside your process*. They do **not** outlive you, and a
  different process **cannot** `SendMessage` them ("not a running agent in this session" — verified). So
  respawn cannot be "keep the team, swap the hub."
- `--fork-session` / `--resume` **reload the polluted context** — they do not clear it. A spawned agent
  **cannot spawn** (no nesting), so you cannot create a successor-hub *from within* the team either.
- The **only** thing that yields a clean context is a **fresh root `claude` process**, launched detached. This
  works (verified: a `nohup claude -p "…" &` came up clean, read a baton file, and ran in ~10s, no human).

**State lives in FOUR files, never in your context** (your context is disposable garbage you will throw away
at respawn; the *state* is what survives):
- **notebook** — worker's intuition, tried-rejected list, open hypotheses (already in the doc).
- **ledger** — oracle cache, A/B numbers, gate verdicts (already in the doc).
- **baton** — *your* orchestration state, a few hundred tokens: current phase/mode, the last
  `=== PARITY/FROZEN/OVERTIME ===` boundary crossed, the champion ref, the **team + agent-roster names**, and
  the **ledger offset** you've read to. The baton is a *save-game pointer file*, **NOT** a transcript of your
  history — if it were a context dump, the successor would boot just as polluted as you. Write it as: "here is
  where things stand and what's next," not "here is everything that happened."
- **beacon** — the worker's one-line status (you read this each tick; see the doc's pull-based comms).

**Check your own context every tick — you can read it directly** (verified): your session id is in
`$CLAUDE_CODE_SESSION_ID`. **Do NOT hardcode the transcript path** — the `<project-dir>` slug is derived from
the process's **cwd**, so a successor launched from a different cwd than its predecessor buckets its transcript
under a *different* `~/.claude/projects/<slug>/` directory, and a hardcoded path silently resolves to a
nonexistent file (→ the self-read returns "not found", the occupancy test never fires, and the hub runs until
its window fills and stops mid-work — verified failure mode in a respawn dry-run). **Resolve it at runtime by
the unique session-id filename instead:**
`find ~/.claude/projects -name "$CLAUDE_CODE_SESSION_ID.jsonl"` (the file is unique per session, wherever it
bucketed). The last `usage` object's `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` is
your occupancy (÷ 1e6 = fraction of your 1M window). It climbs monotonically, so a `≥ 0.70` test never flaps.
(Also: do **not** pass `--no-session-persistence` — `-p` sessions are saved to disk by default, which is what
makes this self-read possible; that flag would disable it.)

**The respawn procedure, at context ≥ ~70%** (bias *early* — an extra respawn costs one ~40k re-brief on a 1M
window, while a *missed* one costs the whole run when the window fills mid-work. Finish the in-flight unit
first — the current gate pipeline or claim — then hand off; never strand a half-processed claim across the
boundary):
1. **Flush state.** Ensure notebook + ledger + baton are current (worker owns notebook/ledger; you own baton).
   **Bump the respawn counter in the baton** (`respawn_count`): the successor will read it, increment it as it
   reattaches, and log `REATTACHED #<N> at phase <X>` to the hub log. The count lives in the baton (not your
   context) so it survives every handoff and doubles as the audit trail of how many clean successions have
   happened. It is also the **fallback respawn trigger**: if the occupancy self-read ever fails, respawn on a
   deterministic backstop (N escalations / M ledger commits / T hours since spawn) and still bump the count.
2. **Launch a clean successor — hardcode the model string** (the CLI `opus` alias resolves to **4-6**, a silent
   downgrade off the 1M window — see the doc's two-trap note). Detached, so it outlives you (verified: a
   `nohup claude -p &` keeps running after the launching process exits). **Make the launch instruction
   fully self-explanatory** — a `claude -p` invocation given a terse or suspicious-looking task ("run this,
   don't explain") may balk and ask for confirmation instead of acting, and a successor that stalls on a
   clarifying question never reattaches. State the goal plainly:
   ```
   nohup claude -p "$(cat <kickoff>) \n\nYOU ARE A RESPAWNED HUB. First: assert your model is \
   us.anthropic.claude-opus-4-8[1m] and your context window is 1M; if not, relaunch yourself with that exact \
   model string and exit. Then read the baton at <baton-path>, the notebook, and the ledger tail, and resume \
   the switchboard at the phase the baton names." \
   --model 'us.anthropic.claude-opus-4-8[1m]' --permission-mode bypassPermissions \
   > <successor-log> 2>&1 &
   ```
3. **Hand off and exit.** Shut down your team (its in-process members die with you anyway), confirm the
   successor is alive (it writes a "REATTACHED at phase X" line to the hub log), and stop.
4. **The successor rebuilds the team from files** — `TeamCreate`, then respawn worker (re-briefed from the
   notebook) + investigators, all with `model:"opus"`, and resumes at the baton's phase.

**Why the re-brief is nearly free — the insight that makes this clean:** the doc *already* respawns the
**worker** from the notebook at *its* ~50% (the worker does the heavy lifting, so it fills fast and recycles
often *within* a single hub-epoch). So "the successor rebuilds the worker from the notebook" is not a new
cost — it is one extra instance of something that already happens routinely. The worker's persistence was
**always** the notebook, never its live context. That is why rebuild-from-files beats the tempting alternative
(give the worker its own immortal background process + cross-process file-polling): that alternative adds
cross-process plumbing to preserve a worker context you are *already* discarding on schedule — zero benefit,
more failure modes.

**Within an epoch** (between respawns), keep yourself ticking with `/loop` or `ScheduleWakeup` so you poll the
beacon and advance the switchboard — but note that is a *keepalive that adds to your context*, **not** a reset;
the reset is only ever the fresh process above. **Fallback** if the self-read ever fails: respawn on a
deterministic backstop you can always count instead (N escalations handled / M commits in the ledger / T hours
since spawn, whichever first) — event-count correlates with context growth, so it respawns you before the wall
even when flying blind.
