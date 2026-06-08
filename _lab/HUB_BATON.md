# RUN 3 HUB BATON — orchestration save-game pointer (NOT a history dump)
# Successor: read THIS + notebook tail + ledger.run3 tail, rebuild team from files, resume at PHASE below.
# This is "where things stand + what's next," a few hundred tokens — not a transcript of what happened.

respawn_count: 0   # bump on each clean succession; successor logs "REATTACHED #<N> at phase <X>" to HUB_LOG.md

## ⚠️ NO AUTO-HEARTBEAT THIS SESSION — STALL RISK (caused a 13h stall, 2026-06-04 02:47→16:09)
ScheduleWakeup / /loop dynamic gate is REFUSED this session → hub has NO self-timer. Liveness depends on (a) worker DM
auto-waking the hub, or (b) human prompt. (a) is what FAILED: worker launched a DETACHED bg oracle that DIED silently,
ended its turn waiting on it → never re-invoked → worker+hub dormant 13h until the human asked. MITIGATIONS IN FORCE:
(1) HARD RULE to worker — NO detached/bg GPU jobs; long oracles run FOREGROUND-under-token, one shape/file at a time.
(2) Hub VERIFIES (beacon mtime + nvidia-smi + _lab/logs write-times), NEVER infers progress from idle notifications.
(3) Foreground jobs keep the worker's turn ALIVE → it re-engages the hub when the tool returns. Successor via claude -p
may also lack the timer — same discipline. See memory hub_no_detached_gpu_and_verify_beacon.

## LIVE CLIMB STATE (2026-06-04 ~16:43, post-stall-recovery) — narrow-N warps cluster
Worker climbing narrow-N after EDIT#5-v2 banked (#6). Oracle results so far:
- softmax(131072,256): FULL oracle DONE, cached, seed/oracle=1.170. Oracle bundle = num_warps 4→16 + M_BLOCK
  8→16 + num_stages 5 + eviction (NOT pure occupancy-warps — it's a BUNDLE).
- rms_norm(8192,768): FULL oracle FLAKY — search ran to convergence (259s "Copy 0 finish no improvement") then
  proc DIED before result-write/cache (no traceback, not the timeout). Cache holds OLD quick 1.136. PARTIAL
  mid-search signal: rms explored persistent_interleaved + num_warps=2 (FEWER) + maxnreg + ns8 — OPPOSITE of
  softmax's →16. ⇒ narrow-N is NOT one unified lever (softmax small-N wants MORE warps, rms narrow-N wants FEWER).
  This is the live arbiter of the old "opposite directions" tension — trust the oracle, not the stale note.
- Worker switched to QUICK-effort triage (full flaky on narrow shapes) for rms+ln(8192,768). QUICK = iterating
  only; PARITY-confirm needs FULL (or log-extracted winner + fair re-bench — asked worker if dying full at least
  LOGS its winning config; if only cache-write dies post-search, recoverable).

NARROW-N RESOLVED = 3 STORIES, NOT 1 RULE (occupancy-contour on grid_rows//num_sm = EDIT-PID quantity, NO new fact):
  1. softmax(131072,256) HIGH-OCC(993): FULL 1.170, oracle wants MORE warps 4→16 + M_BLOCK 8→16 + ns5 + evict
     (BUNDLE). = CLEANEST/highest-value target. Worker now lever-decomposing (seedable ladder: block_sizes →
     +warps → +ns vs full verbatim oracle; classify plateau-vs-peak; check EDIT#6 evict already partly emits).
  2. rms_norm(8192,768) LOW-OCC(62): quick 1.136, oracle wants FEWER warps 4→1 + LOOPED row (rl[None]→[256]).
     OPPOSITE warp dir from softmax (consistent w/ OCC contour: low OCC → don't add warps). LOWER PRIORITY: 22µs
     noise-floor risk + counter-intuitive levers (w1 + loop-a-fitting-row). When revisited: use seed/oracle RATIO
     not raw µs, lift M off floor, perf-investigator must explain the mechanism before banking. anti-giving-up will
     demand noise-robust re-measure.
  3. layer_norm(8192,768) LOW-OCC(62): quick 0.939 = seed BEATS oracle = NON-GAP, DROPPED (status-table 1.056 was
     quick-noise).
  Worker holds GPU token for the softmax decomp. Next artifact: ladder verdict (plateau/peak) or gate-ready softmax.

## ⏸ RUN STOPPED BY HUMAN 2026-06-04 16:56 — RESUME POINT = build+gate EDIT#7 (NOT yet built/committed/gated)
Team shut down (worker+code-inv+perf-inv), GPU released (0 MiB), helion/ clean @ HEAD 12f341ef, champion 6 adv/24 verd.
Worker's FINAL report (crossed the shutdown) = a GATE-READY candidate EDIT#7 for high-OCC small-N softmax:
- DECOMP softmax(131072,256): carrier = COUPLED {M_BLOCK 8→16, num_warps 4→16} = 1.107 (= oracle 1.110). num_stages=5
  INERT, eviction passenger. Each lever alone only ~1.04-1.05; TOGETHER super-additive (doubling M_BLOCK gives 16
  warps enough work to pay). Full oracle on (131072,256) already = 1.170.
- GENERALITY A/B (run3_smalln_occ_contour_ab.py, the overfit guard): all 3 hi-OCC small-N want SAME bump M_BLOCK×2 +
  warps16: (131072,256)OCC993=1.108, (131072,512)OCC993=1.265, (262144,128)OCC1986=1.034. M×4 over-bumps (worse).
  LOW-OCC CONTROLS confirm the gate: (16384,512)OCC124 bump HURTS (w16=0.785), (4096,256)OCC31 w16=0.765. CLEAN
  threshold between OCC=124 (don't) and OCC=993 (do). Keyed on grid_rows//num_sm = EDIT-PID quantity, NO new fact.
  PHYSICAL: hi OCC = SMs full → each program affords 2 rows + 16 warps; lo OCC = few programs → bump over-subscribes.
- EDIT#7 RULE (T2): when grid_rows//num_sm >= THRESH (between 124 & 993) AND small persistent rnumel → M_BLOCK=2×floor
  + num_warps=16. Worker's proposed NEXT (NOT done — this is the resume worklist):
  (a) build rule in triton.py T2 path (OCC threshold + small-N co-condition + coupled M_block/warps);
  (b) FULL 4-split flip-set — ✅ DONE (non-GPU). RESULT: gate (OCC=grid_rows//num_sm>=256 AND rnumel<=1024) fires on
      softmax (262144,128)train+(131072,256)train+(262144,256)val+(131072,128)test+(262144,257)robust = intended +
      held-out softmax. ONLY non-softmax firers = rms_norm/layer_norm/sum at (262144,256) ROBUSTNESS — and those are
      **T1** (rollable); softmax is **T2**. ⇒ SCOPE EDIT#7 TO THE T2 PATH (like EDIT-PID is T1-scoped) → rms/ln/sum
      STRUCTURALLY EXCLUDED (never reach T2 branch) = zero regression risk. Within T2, every firer is softmax (kl/jsd
      are T2 but N>>1024 → excluded by rnumel). "Only softmax fires" = curriculum INCIDENCE (softmax is the only T2
      kernel in the high-OCC small-N regime), NOT an identity fence — same shape as jsd-for-EDIT#5. Auditor defense
      ready. ⇒ plan-step-(b) identity/overfit guard RESOLVED.
  (c) pin THRESH + "small-N" co-key. Worker's PROPOSAL (NOT yet adjudicated/built — run stopped): THRESH >= 2*num_sm
      (=264 on H100, "fills grid twice"; sits in the proven gap OCC124-doesn't-fire / OCC993-does), rnumel<=1024
      (proven small-N ceiling; contour had no high-OCC mid-N shape to test higher). Hub light-steer for resume (NOT a
      build authorization): 2*num_sm is the principled choice over a round 256; the rnumel<=1024 co-key is data-justified
      but flag it as "untested above 1024 at high OCC" — a future high-OCC mid-N shape may want the ceiling raised, so
      comment it as a conservative bound, not a law.
  then GPU: full-confirm >=1 more hi-OCC shape (131072,512 quick→full) before it counts. Gate path = fact-integrity
  (if any new fact / threshold defense) + auditor (no-fence, no-overfit) + results-referee (full flip-set no-regress).
- UNRESOLVED INFRA BLOCKER for banking: full oracles DIE post-convergence on narrow shapes (search finishes, result-
  write crashes) — twice. Quick iterates; banking needs a completing full oracle OR log-extracted-winner re-bench.
  CHECK whether the dying full run LOGS its winning config (recoverable) before treating narrow-N full-confirm as blocked.
LOWER-PRIORITY / SEPARATE: rms_norm(8192,768) low-OCC narrow (FEWER warps + loop, 22µs NOISY — seed/oracle ratio not
µs, perf-investigator must explain mechanism before banking); layer_norm(8192,768) = NON-GAP (dropped). long_sum
(16,2097152) 2M tail = source-limit CANDIDATE — fire anti-giving-up w/ full oracle (must show seed≈oracle AND oracle<tc).
OPEN FULL-ORACLE-COMPLETION ISSUE: full oracles die post-convergence on narrow shapes (twice now). Likely
teardown/serialization/OOM-at-write, not a search failure. RESOLVE before banking any narrow-N parity claim.

PHASE: Phase 2 (oracle ascent) — converging on PARITY, NOT there yet.
LAST_BOUNDARY: none (no `=== PARITY REACHED ===` logged yet; still climbing).
MODE: climb (pre-parity: keep worker climbing seed→oracle on the 8 in-scope kernels; never let it stop).

## SCOPE THIS RUN: IGNORE WELFORD ENTIRELY
Another agent owns welford (the is_structured_combine / Band-C kernel). Do NOT climb/oracle/gate/report it.
Its branch in get_seed_config is disjoint, so leaving it untouched costs nothing. PARITY this run = parity on
the OTHER 8 kernels: rms_norm, layer_norm, sum, long_sum, cross_entropy, softmax, kl_div, jsd.
Carry NO welford numbers in tables or PARITY accounting. (EDIT#4b welford full-close = OUT OF SCOPE.)

## TREE
Worktree /home/dev/local/helion-reduction-heuristics-run2, branch reduction-heuristics-run2.
HEAD 264ba037 (notebook-only commits past handoff's 102c9977). helion/ source CLEAN.
STRUCTURED_APPLY_LOOP_CHUNK_BYTES == 8192 (EDIT#4 stays REVERTED — settled; do NOT rebuild). NEVER git push.

## BANKED (trustworthy): 5 champion_advances + 20 gate_verdicts in ledger.run3
1 row_reread faithful fact; 2 EDIT#1 CE persist-cap→240KiB; 3 EDIT#3 reread-eviction (reread_buffer_slots);
4 EDIT-PID persistent_interleaved+sm_mult+maxnreg (T1 wide-CE); 5 EDIT#6 softmax T2 reread-eviction.
EDIT#4 (welford apply-cap 8192→16384) = results-referee REJECT + reverted (do not reopen).

## ✅ EDIT#5-v2 BANKED = champion advance #6 (SHA 6bcfeed1). champion_advances now 6, gate_verdicts 24.
3/3 gates: fact-integrity PASS #24 (native, 6bcfeed1) + auditor #20 + referee #21 (TRANSFERRED via byte-identity
num_carried==nro 70/70). Faithful fact num_carried_accumulators (inner-ForLoop carry-set [M,R] tiles) replaced the
÷nro proxy. Oracle-cache key fixed deterministic (12f341ef) + re-stamped (uncommitted, welford-drift not ours).
GPU-GRANTED to worker for task#2 (live seed/oracle status table) at ~02:42 (confirmed idle). MODE: back to CLIMB.
NEXT: worker rebuilds live per-shape picture → climbs shapes furthest from oracle (8 in-scope). Remaining worklist:
narrow-N warps (rms(8192,768)+softmax small-N; occupancy still-open per code-investigator), long_sum(16,2097152) 2M
tail (source-limit CANDIDATE — worker FLAGS, hub fires anti-giving-up w/ full oracle), kl_div full-confirm owed.
GATE-INFRA LESSON (banked): brief [analysis] gates to REUSE committed probe artifacts, NEVER re-author kernels in
DSL (cost 2 fact-integrity fatalities). Record-verdict-FIRST in every gate brief.
======================================================================================================
## (history below) EDIT#5 (÷nro) — fact-integrity FAIL (case B, proxy). Reworked. [resolved above]
EDIT#5 ÷nro gate result: auditor PASS + referee PASS (measurement, curriculum-correct) but **fact-integrity FAIL**
(ledger entry 23) — substrate veto, so champion_advances STAYS 5, #6 NOT banked. WHY: num_reduction_ops is a
LUCKY-ON-CURRICULUM PROXY (failure mode #11). It counts ReductionLowering nodes = resident-accumulator count
ONLY under jsd/kl_div's coincidental 1:1 structure. Gate built 2 real Band-B divergence kernels where ÷nro
mis-sizes: DIV-A (1 carried [M,R] accum, 2 reductions → nro=2 → UNDER-sizes 2048-when-4096); DIV-B (2 carried
accums, 1 reduced+1 stored → nro=1 → OVER-sizes 4096-when-2048 = the exact spill the cap prevents). The SEED
VALUES are right on curriculum (jsd carries 2 loop-carried accums, kl_div 1 — kl_loss is in-loop scratch); the
FACT deriving them is hacky.
FIX (gate handed it back): re-key the divisor onto a NEW faithful fact = count of [M,R] 2D accumulators CARRIED
ACROSS the inner hl.tile loop (loop-carried 2D outputs of the inner reduction ForLoop; = num_tiled_accumulators
restricted to root-created carried buffers, EXCLUDING in-loop scratch like kl_loss). Yields jsd:2/kl_div:1
(curriculum) AND DIV-A:1/DIV-B:2 (divergence). num_reduction_ops is wrong; num_tiled_accumulators-as-shipped is
ALSO wrong (=2 both — over-counts kl_div's scratch). Needs the finer fact.
CARRY-OVER: new fact == nro on curriculum → seeds BYTE-IDENTICAL → auditor+referee PASSes TRANSFER if byte-id holds.

## EDIT#5-v2 COMMITTED 6bcfeed1 (source-only: config_spec+device_ir+triton.py; lab=d58e6ee3 on top = HEAD).
Verified clean source-only commit, on-branch, divisor now `// (itemsize * num_carried_accumulators)` (triton.py:807).
NEW FACT: num_carried_accumulators = inner reduction ForLoop carry-IN node_args filtered to 2D [M,R] tiles
(device_ir._count_carried_tiled_accumulators). Worker values (code-investigator corroborated): jsd=2, kl_div=1
(kl_loss is in-loop SCRATCH, excluded), DIV-A=1, DIV-B=2 (tracks real property where nro diverged), distinct from
num_tiled_accumulators (=2 both, over-counts scratch). Worker proved num_carried==nro on all 70 jsd+kl shapes → seeds
byte-id to 46e58c73 (the ÷nro version auditor+referee PASSed).
GATE (2026-06-04): **fact-integrity ONLY** [analysis, no GPU], DUAL mandate: (1) faithfulness via DIV-A=1/DIV-B=2
+ reading kernels, (2) INDEPENDENTLY re-verify byte-identity to ÷nro across 4 splits (the auditor+referee transfer
condition — gate confirms, doesn't trust worker's proof). PASS both → bank #6 on 6bcfeed1 (auditor+referee CARRIED
OVER). If byte-id fails → referee must re-fire too.
GATE INFRA NOTE: 1st fire a325d720 DIED on transient API error before recording (ledger stayed 23, NO v2 entry —
verified by dumping last 6 entries; champion stays 5). RE-FIRED as a3431d1001b773b3a (same mandate + hardened: RECORD
VERDICT FIRST then return). This is the 3rd gate-infra fatality salvaged this session (fact-integrity v1 watchdog
stall→completion gate a8f7737f; this API error→re-fire). Protocol: a non-verdict (stall/API-error) is NEVER a verdict
— always salvage via fresh re-fire/completion, never bank or fail on it.
UPDATE: a3431d1 ALSO STALLED (watchdog) — its trace showed it wasted ~10min trying to AUTHOR DIV-A/DIV-B kernels in
Helion DSL (the trap that killed both v2 instances). Ledger still 23, no v2 entry. RE-FIRED 3rd time as **a3431d1001…
→ new id below**, RESCOPED: forbid re-authoring kernels (DIV-A/DIV-B already established by ledger entry 22); just RUN
the worker's committed probe run3_carried_accum_probe.py (faithfulness) + fact-emit num_carried==nro per shape
(byte-id); RECORD-FIRST; ~15-call budget. CURRENT LIVE fact-integrity = **a (3rd rescoped re-fire — see latest Agent
spawn)**. If THIS one also dies on infra: the substantive answer is essentially known (probe shows jsd=2/kl_div=1;
entry 22 established DIV-A=1/DIV-B=2 by the carry-set definition; byte-id is worker-proven 70/70) — a 4th re-fire or
a hub-rendered verdict citing the probe output + entry 22 is defensible. Do NOT let gate-infra flakiness block #6
indefinitely; the gate's JOB is nearly done, only its RENDERING keeps dying.
Then GPU-GRANT worker for task#2 (status-table live seed/oracle re-bench, welford-idle-gated).

## (history) EDIT#5 COMMITTED 46e58c73 (triton.py-only; orphaned d2ff878a ≡ it); gates fired ~01:30
EDIT#5 committed as **d2ff878a** (triton.py-only; HEAD dc8755e3 is _lab-only on top → d2ff878a==live seed,
verified `git diff d2ff878a..HEAD -- helion/` empty). 3 gates firing on d2ff878a, all model:opus, background:
  fact-integrity   a24ad820 [analysis] — nro-footprint-vs-row_reread-proxy distinction
  adversarial-auditor af9b3c12 [analysis] — over-claim/fence/no-regression
  results-referee  a144dd29 [timing, HOLDS GPU TOKEN] — FULL jsd flip-set no-regression (EDIT#4 lesson)
Gates self-record to ledger.run3.gate_verdicts; ping hub only on FAIL/done. Referee signals "GPU RELEASED" when
done → THEN grant worker the token for task#2 (status-table live seed/oracle re-bench). All PASS → bank advance #6.
GATE STATUS: adversarial-auditor af9b3c12 = **PASS** (recorded, ledger=21). Strong: gain real (oracle picks 2048
narrow+mid+wide), not kernel-id (jsd sole Band-B∧nro≥2; kl_div nro=1 control), and structurally the INVERSE of
EDIT#4 (EDIT#5 LOWERS R_BLOCK → footprint shrinks monotonically → hidden valley physically implausible).
Auditor asked referee to also time extreme jsd(262144,4096)+(8192,151936) (thoroughness, not blocker) — forwarded.
results-referee a144dd29 = **PASS** (ledger entry 22). Independently reproduced: 34 flips, seed emits R_BLOCK=2048,
ZERO regressions across full flip-set (worst (2048,256000)@1.021 still 2048-faster); EDIT#4 valley-analog
jsd(262144,4096)=1.047x WIN (no valley — cap LOWERS R_BLOCK so footprint shrinks monotonically). GPU RELEASED.
NOISE NOTE (important): a transient external ~62GB proc (THE WELFORD-OWNER) appeared mid-run → referee re-ran
idle-gated, spreads collapsed 1-2%, stable. CONFIRMS welford-owner intermittently grabs GPU → keep strict-serial
+ nvidia-smi-before-EVERY-bench discipline.
STILL OUT (the only blocker to banking): fact-integrity. Original a24ad820 STALLED on 600s watchdog AFTER the
substantive analysis, BEFORE the verdict (recorded NOTHING — ledger still 22; the known watchdog-stall failure
mode, salvageable). Spawned COMPLETION gate **a8f7737f** [analysis, no GPU] resuming the exact crux:
  CRUX: the Band-B cap bounds the [M_BLOCK,R_BLOCK] 2D-accumulator footprint, which is what num_tiled_accumulators
  counts (=2 for BOTH jsd & kl_div per worker) — but the edit divides by num_reduction_ops (jsd=2, kl_div=1).
  Is ÷nro faithful (case A: kl_div's SIMULTANEOUSLY-resident [M,R] accum count is really 1, n_tiled over-counts)
  or a lucky-on-jsd PROXY (case B: kl_div truly holds 2× but oracle wants 4096 for an unmodeled reason)? Decided
  by READING examples/jsd.py + examples/kl_div.py inner loops. PASS=A, FAIL=B(+divergence kernel).
2/3 PASS (auditor+referee); bank only on 3/3. fact-integrity has substrate VETO — do not bank on a stall.
CANONICAL BANK SHA = 46e58c73 (helion/ ≡ orphaned d2ff878a the gates read; verified blob d3bb8476).
Worker flip-set receipt (codegen): 34 jsd flips 4096→2048 all 4 splits incl large-M (16384,32000) + wide-V
(4096,128256)/(4096,151936)/(8192,128256); kl_div 0 flips byte-id; sum/softmax excluded; 0 INCONSISTENT.
Correctness: 5 jsd@2048 maxerr≤1.49e-8 + 2 kl_div@4096, rtol1e-3/atol1e-4 not loosened.

## (orig EDIT#5 spec, for reference) the edit (Band-B block ~line 785): divisor
  `BANDB_R_BLOCK_BYTES // max(1, itemsize)` → `// (max(1, itemsize) * max(1, num_reduction_ops))`.
jsd nro=2 → R_BLOCK 4096→2048 (closes narrow-V 1.21→tie + wide 1.03→tie); kl_div nro=1 → byte-identical.
GATE CONDITION (mandatory — the EDIT#4 lesson): results-referee no-regression MUST sweep the FULL jsd train
split (every shape the cap flips 4096→2048), ESP large-M (16384,32000) + wide-V (4096,128256)/(4096,151936)/
(8192,128256) — NOT just the 3 A/B'd shapes. + kl_div byte-identity across its M-range + sum/non-Band-B
untouched. fact-integrity: nro here = DIRECT faithful live-accumulator count (budget/(itemsize*nro) =
per-accumulator R_BLOCK share = footprint physics), NOT a row_reread proxy — worker defends exactly this
distinction (fact-integrity FAILed nro once as a re-read proxy; this use is different). adversarial-auditor:
no jsd-fence (jsd is sole firer by curriculum incidence, not by identity; tv_distance transfer nro=1→stays
4096). All PASS → bank as advance #6.

## TEAM: reduction-heuristics (lead = me, the hub)
worker [timing] — persistent climb driver; owns triton.py heuristic + _lab/run3_notebook.md (its source of truth).
code-investigator [analysis], perf-investigator [timing] — standing peers; worker DMs them directly (no hub relay).
Gates = fresh ephemeral background Agent per claim (results-referee, adversarial-auditor, anti-giving-up,
fact-integrity). model:"opus" on EVERY spawn (omit → silent opus-4-7 downgrade; "opus" → opus-4-8[1m]).
Gates self-record verdicts to ledger.run3.gate_verdicts AS-RETURNED; hub pinged only on FAIL/decision-required.

## LEDGER OFFSET: successor reads ledger.run3.gate_verdicts[20:] + champion_advances[5:] for post-baton state.

## LEDGER-KEEPER RULING (2026-06-04 ~02:15) — oracle-cache key staleness, disposition (2)
Worker found: adding ReductionFact.num_carried_accumulators (and earlier row_reread/reread_buffer_slots) moved the
oracle-cache source_hash for ALL 8 in-scope kernels. ROOT CAUSE: run3_oracle.py source_hash component (c) iterates
dir(config_spec) and sweeps in `reduction_facts` (the ReductionFact repr) → every SEED-ONLY fact-add invalidates the
whole cache. Contradicts the recipe's own intent (component (c) should hash the SEARCH SPACE, not seed-fact values).
DEEPER FINDING (worker, 2nd flag): source_hash was PROCESS-NON-DETERMINISTIC all along — component (c) iterates
dir(config_spec) and includes BlockIdSequence attrs (reduction_loops, num_threads, range_*, loop_orders, l2_groupings,
flatten_loops, static_ranges, cute_vector_widths, range_warp_specialize) whose repr is the address form
`<...object at 0xADDR>` → differs per process (jsd:8192x30522 = 34747730 vs 0d13c6c7 in 2 procs). CONSEQUENCE: the
hash-match staleness guard has ALWAYS read "stale" cross-process → was VACUOUS → cache always used as by-shape
latencies, never hash-gated → NO past result silently corrupted (reassuring), but the guard must be fixed to mean
anything. Dropping reduction_facts alone would NOT fix it (addresses remain).
RULING (re-ruled, scope grew): **option (ii)** — minimize (c) to STABLE content-repr search-space knobs only.
Component (b)=DEFAULT-config generated Triton already captures per-shape search-space STRUCTURE, so (c) only adds
scalar/ListOf knobs (b) misses. (c) KEEPS scalar configs (target_device_capability, max_num_sm_multiplier…) +
ListOf knobs w/ content reprs (indexing, load_eviction_policies). (c) DROPS via STRUCTURAL filter (not a by-name
list): skip any value whose repr matches `<.* object at 0x[0-9a-f]+>` + skip `*_facts` (future-proof: auto-excludes
new BlockIdSequence knobs). AUDIT CONDITIONS: (1) commit shows before/after of (c)'s dump; (2) DETERMINISM PROOF —
same-shape hash MATCHES across 2 processes (old 34747730≠0d13c6c7 → must become equal, include both); (3) cache diff
hash-strings-ONLY, NO latency change. Cached oracle LATENCIES valid (num_carried + the dropped knobs are seed/codegen-
captured, not search-space-altering). GUARDRAIL REAFFIRMED (non-negotiable): re-stamp does NOT let a cached oracle
count toward PARITY — every banked/parity/ceiling shape needs a FRESH FULL oracle. Cache = cheap iterating picture
(task#2 status-table by-shape join) ONLY. Worker doing (ii) NOW as non-GPU work (parallel-safe w/ the v2 gate).
DONE: recipe-fix committed **12f341ef** (_lab: run3_oracle.py + run3_restamp_oracle_hashes.py). All 3 receipts
AUDITED PASS: (1) structural filter drops *_facts + compiler_seed_configs[=the seed!, worker-caught] + `at 0x`
addr-reprs, keeps 27 content knobs; (2) determinism proof hash=6e8256298c53a1cd in BOTH procs (was 34747730≠
0d13c6c7); (3) re-stamp 29 in-scope (3 welford SKIPPED, out of scope), per-entry assert proves latencies untouched,
idempotent. CACHE DISPOSITION = **(y) LEAVE UNCOMMITTED** (committing would entangle the welford-owner's pre-existing
latency drift into this branch = scope crossing; re-stamp recoverable via committed idempotent script; working-tree
state survives respawn → task#2 has key-fresh cache). oracle_cache.json stays uncommitted guardian artifact BY DESIGN.
A respawn: do NOT "git checkout" or commit oracle_cache.json (welford drift is not ours); re-run the restamp script
if the working-tree copy is ever lost.

## GPU: 1× H100 idx 0, idle at handoff. Strict-serial timing token.
worker DMs "REQ-GPU <what/dur>" → hub grants only when nvidia-smi clear → worker DMs "GPU-RELEASED".
NEVER fire a timing gate (referee/anti-giving-up bench) while worker holds GPU — reclaim token first.
CAVEAT: the welford-owner is a SEPARATE process the hub cannot serialize against → worker MUST nvidia-smi-check
idle before EVERY bench (don't trust a number taken while another proc holds the GPU).

## NEXT MOVE
1. Worker commits EDIT#5 → DMs SHA → hub fires gate pipeline (fact-integrity+auditor [analysis] concurrent;
   results-referee [timing] on GPU with the FULL jsd flip-set condition). PASS → advance #6.
2. Worker rebuilds the LIVE seed/oracle picture (_lab/harness/run3_status_table.py: cached ORACLE valid — key
   excludes heuristic; re-bench LIVE seed, oracle_cache.json seed_us is stale palimpsest) and climbs shapes
   furthest from oracle. Worker discovers targets itself; hub keeps it climbing + gates each claim.

## REMAINING TO PARITY (8 kernels — honest, real work):
- EDIT#5 jsd (gating now).
- narrow-N warps cluster (HARD): rms(8192,768) + softmax small-N. Occupancy hypothesis FALSIFIED by
  code-investigator (progs/SM identical). Separator is neither rnumel nor occupancy → finer static fact OR
  honest "runtime-only" Product-B. Do NOT pre-conclude; on any disposition claim fire anti-giving-up.
- long_sum(16,2097152) 2M tail — source-limit candidate (N>2^20 structural). Worker FLAGS, doesn't self-certify;
  hub fires anti-giving-up with a FULL oracle (must show seed≈oracle AND oracle<tc).
- quick-VICTORY full-confirms — at-floor sweep parities were QUICK oracles; full-confirm extreme bands before
  they count toward PARITY (sum(16384,2048)=0.989 confirmed; kl_div full-confirm still owed).

## TERTIARY: after PARITY → log boundary → Phase 3 (freeze, Product B budget-reduction, read TEST once via
ledger-keeper, report train↔test gap) → `=== DELIVERABLE FROZEN + VALIDATED ===` → Phase 4 beat-oracle overtime.
