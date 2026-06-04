# HANDOFF — RUN 3 hub (read this FIRST, then the kickoff's 3 files)

> You are the **hub** continuing run 3. The kickoff (`_lab/prompts/hub-kickoff.md`) + its 3 READ-FIRST files
> (`reduction-heuristics-standalone.md` = method/bar/gates, `server-specific-setup.md` = this machine's
> paths/commands, `shapes_v3_draft.py` = curriculum) define the job. THIS file is the live state: what's
> banked, what's in flight, the exact next action, and the hard-won protocols. Setup (Step 0/1) is long done —
> do NOT redo it. **All file:line refs drift — grep the symbol.**

## 0. THE ONE-LINE STATE (2026-06-03)
Deep in **Phase 2 (oracle ascent)**, converging on PARITY but NOT there yet. **5 champion advances banked**
(all separated-context gated, on immutable SHAs). **EDIT#5 (jsd) is ACK'd-to-build** — that's your immediate
next action to gate. Real multi-property work remains (welford, narrow-N). Nothing unverified is in the
champion. **Worktree branch `reduction-heuristics-run2`, HEAD `102c9977` (advances as the worker commits
notebook/spot-checks; `helion/` source is clean — apply-cap=8192 confirms EDIT#4 reverted). NEVER `git push`.**

> FIRST CHECK on pickup: `git -C /home/dev/local/helion-reduction-heuristics-run2 status --porcelain helion/`
> (expect clean) + grep `STRUCTURED_APPLY_LOOP_CHUNK_BYTES` (must be **8192** — if 16384, the EDIT#4 regression
> is somehow back, revert it). Then read `ledger.run3.champion_advances` (5) + `gate_verdicts` (20) to confirm
> this handoff's state. Then DM the `worker` to resync: ask for its current HEAD sha + whether EDIT#5 is
> committed yet (it was ACK'd-to-build with the full-jsd-flip-set referee condition — §5).

## 1. THE REFRAME (why run 3 exists — internalize this)
Run 2 declared COMPLETE on the **aggregate geomean** `O`=0.998 vs tc-default. The work-order resets the bar to
**per-shape `seed ≈ oracle` within ε≈3-5% on EVERY measurable train shape** — and names "done because the
aggregate looks good" as failure mode #9. So run 3 = re-open the inherited champion against the per-shape bar,
surface the gaps the geomean buried, climb each to parity (or verify it's a true source-ceiling / proven
runtime-only), then freeze+bank (Phase 3), then beat-oracle overtime (Phase 4). **The oracle (Helion
autotuner's own best config) is the bar, NOT tc-default.** A shape at `seed≈oracle<tc` is DONE (source
ceiling) — but you must VERIFY that with a fresh FULL oracle (anti-giving-up), never self-certify it.

## 2. ORCHESTRATION (this Claude Code Agent/Team harness — all verified)
- Team `reduction-heuristics`. **Hub = you (team-lead)**; you own acceptance + spawn ALL gates + hold the GPU token.
- **Persistent worker** (name `worker`) drives the climb, owns the heuristic code + `_lab/run3_notebook.md`.
- Standing peers: **code-investigator** `[analysis]`, **perf-investigator** `[timing]` — the worker DMs them directly.
- **Gates = fresh ephemeral background `Agent` calls per claim** (results-referee, adversarial-auditor,
  anti-giving-up, fact-integrity). Brief them NEUTRALLY on the immutable commit SHA; their final message IS the
  verdict — record it to `ledger.run3.gate_verdicts` AS-RETURNED, before you decide.
- **`model:"opus"` on EVERY spawn** (worker, investigators, gates). Omitting silently downgrades to opus-4-7.
- **Worker briefs already written:** `_lab/RUN3_WORKER_BRIEF.md`, `_lab/RUN3_INVESTIGATOR_BRIEFS.md`. If you
  respawn the worker (do it at ~50% context or if stuck — fresh context is the best unstick), point it at the
  brief + `_lab/run3_notebook.md` (its source of truth) with `model:"opus"`.

## 3. WHAT'S BANKED (5 champion advances — `ledger.run3.champion_advances`)
The heuristic = `helion/_compiler/autotuner_heuristics/triton.py :: TritonReductionHeuristic`. Three disjoint
branches in `get_seed_config`: **T1** (`is_t1`, rollable: rms/ln/sum/long_sum/cross_entropy), **Band-C**
(`is_structured_combine`: welford), **T2-plain** (softmax/kl_div/jsd). This branch structure is load-bearing
(it's how edits scope without kernel-identity fences).

1. **Faithful `row_reread` fact** (+ consumer-trace refinement, commits 91dfd8ef→fa11264a) — replaced run-2's
   hacky `num_load` op-count proxy after THREE rejected framings (num_load over-counts gathers/broadcasts;
   num_reduction_ops under-counts the apply re-read; "≥2 read-regions" missed in-register reuse). The faithful
   computation: reduction-input host buffer consumed across the reduction boundary (loaded in ≥2 loop graphs /
   reduction+apply). fact-integrity PASS. **byte-identical seeds** to the gated bool.
2. **EDIT#1** (dab1eea8) — CE persist-cap MULTILOAD_PERSIST_MAX_BYTES 128→240KiB → 3 CE-boundary shapes
   (49152/50257/50304) at oracle parity. Overturned run-2's "CE wide-vocab = source ceiling" claim.
3. **EDIT#3** (a62e26da) — reread load-eviction (T1 CE + Band-C welford), keyed on `reread_buffer_slots`
   (the re-read reduction-input buffer's load slots → 'last' on first, 'first' on rest). De-hacks run-2's
   POSITIONAL slot-0 rule. CE 1.31/1.19/1.08×, welford byte-identical-to-champion. (rms_norm "1.07×" was
   HEDGED to a within-noise canary per the auditor — NOT a win.)
4. **EDIT-PID** (94851da9) — **the signature recovery.** persistent_interleaved pid + physics-derived
   `num_sm_multiplier=clamp(np2(ceil(grid_rows/get_num_sm)),1,32)` + maxnreg=64, in the **T1 branch ONLY**,
   gated `fact.row_reread and not persistent`. Recovered a 38%/19.5% wide-CE seed<oracle gap the worker (and
   I) twice nearly declined as Product-B — anti-giving-up FAILED the decline (the "no clean rule" was a #8
   trap: welford is Band-C = structurally unreachable by a T1 gate; the worker had skipped the layer_norm
   measurement). sm_mult is NOT oracle-fit (ships derived-16 at 256000 where the oracle picked 4). 3 gates PASS.
5. **EDIT#6** (8577b675) — extended reread-eviction to T2 softmax (1.31/1.08× tc-beating). **The eviction
   family is now complete across all 3 tracks (T1/Band-C/T2) off the ONE faithful `reread_buffer_slots` fact** —
   the "general heuristic not kernel-fitted" payoff.

**Per-shape VICTORY confirmed:** 3 CE-boundary, softmax-wide (256K/512K), CE-wide-V (eviction+pid ~1.05). The
at-floor sweep (commit 25bd7d6b) confirmed **most of the 6 at-floor kernels (rms/ln/sum/long_sum-persistent/
kl_div) ARE at oracle parity** — the geomean wasn't hiding gaps everywhere.

## 4. THE EDIT#4 REJECT — the over-claim guard's headline catch (SETTLED, do not reopen)
**EDIT#4 (welford apply-cap 8192→16384, c3d90e8d) was results-referee REJECTED and REVERTED (d09e08b7).**
The single constant flipped welford's apply tile 2048→4096 on EVERY structured-combine shape above the persist
threshold — including **welford(262144,5120) [in-curriculum], where apply=4096 is a 7.2× pathological valley**
(33.5ms vs 4.6ms, reproduced 5×/3 codepaths/2 seeds). The worker's 3-shape A/B and even the auditor MISSED it;
only the referee's independent BROAD re-run (testing shapes between narrow and wide) caught it. **This is
settled: EDIT#4 is dead-as-a-standalone-constant; cap is back at 8192.** The welford wide-N gain (real:
1.05-1.07× at (4096,16384)/(32768,8192)) returns ONLY via the harder **EDIT#4b** (task #16): a coupled,
**(M,N)-grid-keyed** welford-tile rule (combine + M_block + apply together, gated so apply≤2048 at large-M),
because welford tile-sizing is M-dependent and a bare per-N cap is too coarse.

> ⚠️ If the worker's messages still say "build EDIT#4" or ask you to gate it — those are STALE crosses. EDIT#4
> is rejected+reverted. Confirm via `ledger.run3.gate_verdicts` (EDIT#4 results-referee = REJECT) +
> `_lab/logs/run3/REFEREE_edit4.json`. Do not rebuild it.

## 5. YOUR IMMEDIATE NEXT ACTION: gate EDIT#5 (jsd narrow-V)
I just **ACK'd the worker to BUILD EDIT#5** with a mandatory condition. EDIT#5 = a 1-line Band-B chunk cap:
divisor `max(1,itemsize)` → `max(1,itemsize) * max(1,num_reduction_ops)` (triton.py T2 Band-B block ~785).
- jsd (num_reduction_ops=2) → R_BLOCK 2048; kl_div (nro=1) → 4096 unchanged. Closes jsd narrow-V 1.21→tie +
  wide jsd 1.03→tie (full-oracle confirmed World A: V-independent). kl_div byte-identical.
- **The nro reuse is LEGITIMATE** (I verified the argument): nro was rejected as a proxy for *row_reread*
  (it under-counts apply passes); here it's the **DIRECT faithful count of reduction accumulators** sharing
  the R_BLOCK budget — `budget/(itemsize×nro)` = per-accumulator share = resident-footprint physics. A fact
  can be a rejected proxy for property X yet the faithful direct measure of property Y. fact-integrity will
  scrutinize this hard (it FAILed nro once) — the worker must defend exactly this distinction.
- **MANDATORY gate condition I attached (the EDIT#4 lesson):** the referee's no-regression MUST sweep the
  **FULL jsd train split** (every shape the cap flips 4096→2048), ESPECIALLY large-M jsd **(16384,32000)** and
  the wider-V rows (4096,128256)/(4096,151936)/(8192,128256) — NOT just the 3 shapes the worker A/B'd. EDIT#4
  died on exactly an untested large-M shape. + kl_div byte-identity across its M-range + sum/non-Band-B untouched.
- **When the worker DMs the EDIT#5 commit sha:** fire **fact-integrity** (nro-direct-count-not-proxy + not a
  jsd-fence; tv_distance transfer nro=1→stays-4096 generalizes) + **adversarial-auditor** (no fence) +
  **results-referee** (FULL jsd flip-set, the condition above). If all PASS → bank as advance #6.

## 6. REMAINING WORKLIST TO PARITY (honest — real work, not "almost done")
- **EDIT#5 jsd** — building/gating now (clean, expected PASS with the full-flip-set check).
- **EDIT#4b welford full-close** (task #16) — the harder coupled (M,N)-keyed redesign. welford(4096,16384)=1.163
  wants combine 8192→16384 + apply→16384 + M_block 1→2 *together* (apply-alone HURTS 0.844; the levers are
  coupled). Two principled-property Qs: why does wide-N want M_block=2 (grid/occupancy, not shape-fit)? combine
  16384 exceeds STRUCTURED_COMBINE_CAP_BYTES and run-1 noted >16KiB spills at N≥32768 → the cap raise must be
  **N-bounded**. Full-curriculum A/B incl (262144,5120) + all wide-M welford (the EDIT#4 lesson).
- **Narrow-N cluster (task #15) — HARD, parked low-priority.** rms(8192,768) wants warps 16→8, softmax(131072,256)
  wants 4→8 — OPPOSITE directions. The occupancy hypothesis was **FALSIFIED** by code-investigator (progs/SM=124
  identical for both → occupancy can't separate them, same shape as the CE-pid falsification). The warp separator
  is NEITHER rnumel (both narrow) NOR occupancy. Likely a finer-static-fact-not-yet-found OR the admissible
  "knowable only at runtime" case → possibly honest Product-B. Do NOT pre-conclude; when the worker claims a
  disposition, fire anti-giving-up (it'll demand the fact-hunt). Doesn't block the other gaps.
- **long_sum(16,2097152) 2M tail** — source-limit candidate (N>2²⁰ structural). Worker FLAGS it (doesn't
  self-certify); you fire anti-giving-up with a FULL oracle (must show seed≈oracle AND oracle<tc).
- **Quick-VICTORY full-confirms** — the at-floor sweep's at-parity shapes were QUICK oracles; full-confirm the
  extreme bands before they count toward PARITY (worker running sum/kl spot-confirms now: sum(16384,2048)=0.989 VICTORY).

When the full per-shape train table is all `seed≈oracle` OR verified-source-ceiling/proven-runtime-only →
**log `=== PARITY REACHED ===`** to HUB_LOG + ledger with the table → **Phase 3** (freeze; Product B
budget-reduction via `_lab/harness/run2_productB_*` patterns; ledger-keeper reads TEST ONCE; report train↔test
gap + transfer generality) → `=== DELIVERABLE FROZEN + VALIDATED ===` → **Phase 4** beat-oracle overtime.

## 7. HARD-WON PROTOCOLS (these were paid for in this session — keep them)
- **GPU TOKEN, strict serial.** ONE H100. Worker DMs "REQ-GPU" → you "GPU-GRANTED" only when clear (check
  `nvidia-smi --query-compute-apps=pid,used_memory` — authoritative; `ps|grep` count is misleading). NEVER fire
  a timing gate (referee) while the worker holds GPU permission — reclaim the token first. Two concurrent
  `do_bench` corrupt both. (I slipped once — caught it, no corruption.)
- **Gates pin IMMUTABLE SHAs.** A gate that times a baseline uses its OWN `git archive <sha>`/clean checkout +
  `rm __pycache__` — NEVER `git checkout` in the worker's shared worktree (that caused a multi-hour "tree drift"
  saga — a gate's baseline-checkout clobbered the worker's live tree). Brief every referee with this.
- **A config-flipping cap/constant's no-regression A/B MUST sweep the axis it flips on, at MID + EXTREME M** —
  not endpoints/hand-picked shapes. This is the EDIT#4 lesson; apply it to EDIT#5 (full jsd split) and EDIT#4b.
- **Verify tree state with `git status --porcelain helion/` + `git rev-parse HEAD`** — NOT `git diff <remembered-sha>`
  (a committed stack on top reads as a spurious "uncommitted divergence"; I misread this twice).
- **Watchdog stalls:** background gate agents sometimes get killed by a 600s stall watchdog AFTER finishing the
  substantive analysis. Don't treat a stall as a verdict — capture the substantive result + spawn a focused
  completion agent to render the verdict (happened twice, both salvaged cleanly).
- **Leapfrog / async crossing:** the worker is fast; your replies lag its per-turn cadence, so it re-asks
  settled things. Don't re-send status fragments — wait for the next ARTIFACT (a commit sha), and when you must
  reconcile, give ONE authoritative "here is the settled state, verify it in the ledger yourself" message. The
  worker has the **don't-execute-through-a-stop** rule (flag conflict + one-word ack rather than executing a
  stale directive) — it applies it well.

## 8. DURABLE STATE (where to look)
- `_lab/HUB_LOG.md` — dated hub arc, newest run-3 entries at TOP (under "## 2026-06-03 (RUN 3 begins...)").
  The full narrative of every edit/gate/decision.
- `_lab/ledger.json` key **`run3`**: `gate_verdicts` (20 recorded), `champion_advances` (5), `results`,
  `PARITY_gap_list`, oracle cache. The structured source of truth.
- `_lab/run3_notebook.md` — the worker's reasoning trace (decisions + empirical why + tried/rejected).
- `_lab/harness/run3_*` + `run2_measure_g.py` (the 9-kernel fn/arg/fp32-ref plumbing all scripts import).
- `_lab/logs/run3/` — oracle caches, A/B JSONs, REFEREE_edit4.json, etc.

## 9. CANONICAL COMMANDS (verified, this machine)
- Interpreter `/home/dev/helion/.venv/bin/python` (venv; NEVER pip install). 1× H100 idx 0, pin `CUDA_VISIBLE_DEVICES=0`.
- **Run from `cwd=/tmp`** (NOT a checkout root) with `PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2`
  so `helion.__file__` resolves to the worktree (NO sys.path.insert). `tritonbench` resolves to the ORIGINAL
  checkout `/home/dev/local/helion` (operator edits go there; everything else in the worktree).
- Bare seed: `cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none PYTHONPATH=<worktree> <py> <script>`.
- Oracle: `HELION_FORCE_AUTOTUNE=1 HELION_AUTOTUNE_EFFORT={quick,full}`. **Full oracle ≈ 5-18 min/shape** — the
  expensive resource; cache it, cheap-first (quick to triage, full to confirm). A quick oracle can UNDERSHOOT
  and fake a parity → a quick GAP is real, a quick PARITY is SUSPECT (full-confirm before counting). This was
  proven twice (softmax-wide null reversed 1.00→1.36; CE "ceiling" reversed at full effort).
- long_sum needs `--reduce-dim 1` + `--shapes`. welford defaults bf16, softmax fp16 — assert fp32 took.

## 10. THE TERMINAL LOGIC (never lose this)
Never let the worker stop. Before parity it climbs (every "ceiling/noise/no-rule/done" → fire anti-giving-up,
which hands back the next experiment). Gate every champion-advancing claim from a separated context on an
immutable SHA — nothing banks without results-referee reproduction + the relevant adversarial gate. At parity:
freeze + bank the validated deliverable (Phase 3) BEFORE any overtime. Then open-ended beat-oracle overtime
(Phase 4) — derive beat-candidates from theory + the answer-key diff, NEVER from observed search winners
(p-hacking, banned); a real beat clears a harder bar than parity (fresh full-effort best-of-N oracle, fresh
process, median-of-N, noise-floor excluded). The gate works in BOTH directions — it caught a 38% under-claim
(EDIT-PID) AND a 7× over-claim regression (EDIT#4). Trust it; keep it strict; keep the worker climbing.
