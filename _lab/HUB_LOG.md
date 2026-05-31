# Hub Log — non-blocking breadcrumbs for the human

> Passive status notes the human sees on check-in. Convergence flags. Never a stop, never a request
> for input. Newest at top.

## 2026-05-28
- **Setup.** Created git worktree `reduction-heuristics-autotuner` at `/home/calebkim/helion-new-heuristics/wt-reduction`
  off `reduction-heuristic-plan` HEAD `1ec3193d`. GPUs 1/2/3 idle; pinned `CUDA_VISIBLE_DEVICES=2`.
- **Step 0a DONE (hub-verified).** Import wiring proven (`helion.__file__` -> worktree via PYTHONPATH);
  codegen edit flows into generated Triton (sentinel test, sha 9f3e2398 -> 18ed86ed); worktree clean.
  Canonical setup written to `_lab/SETUP.md`.
- **Architecture note.** `SendMessage` is unavailable in this harness, so the "persistent worker" is
  implemented as fresh worker invocations driven off the durable `_lab/notebook.md` + `_lab/ledger.json`
  (the wip blesses this as lossless). Hub stays in the loop, spawns all helpers, runs independent gates.
- **Step 0b DONE (measurement-harness-verifier, certified).** Bare-seed mechanism sound. rms_norm_fwd
  (2048,4096) fp32: persistent (reduction_loops=[None]), num_warps=4, median 0.03498ms, correctness PASS
  (max_abs 1.9e-6). configs=[seed] -> no autotune (no CSV); invalid seed RAISES; distinct config->distinct
  Triton (looped [512]/warps8 -> 0.0378ms). Canonical scripts `_lab/harness/{bare_seed_run,evidence_block}.py`.
  Commit 38bca573.
  - GOTCHA 1: tritonbench operators resolve to the ORIGINAL checkout (hardcoded meta-path finder); the
    Helion kernel-under-test runs from worktree. `torch_compile_<op>_default` baseline must be added in the
    original checkout (or a worktree meta-path overlay). See SETUP.md "tritonbench edit wiring".
  - GOTCHA 2: normalize() does NOT collapse reduction_loops>=size_hint to None; persistent-vs-looped is a
    CODEGEN fact. Always inspect generated Triton for the loop, not just the normalized dict.
- **Spawn-mechanism decision (per human latitude):** hybrid. Direct Agent calls for persistent worker +
  trust gates; Workflow scripts for parallel non-timing fan-outs (classification/investigation/verification,
  GPU-partitioned 1/2/3); measurement sweeps = serial scripts on one pinned GPU (parallel GPU timing corrupts
  do_bench).
- **Step 1 DONE (harness-integrity, CERTIFIED unbiased).** Hand-rolled standalone harness reconciles with
  TritonBench to <1% both sides at (4096,8192) fp32 (Helion-default +0.52%, tc-default -0.05%); 1 CUDA
  kernel/call each (no hidden host-side split). 5-way sanity (x vs eager): Helion-default 2.89/2.78,
  quick 3.76/3.71, max 3.74/3.72, tc-default 3.75/3.73, tc-max 3.77/3.75 at (4096,8192)/(8192,8192).
  Ordering as expected; no HALT. KEY: Helion default_config picks a LOOPED reduction and loses ~23-25% to
  tc-default -> that's the real seed-quality gap (oracle G_rms_norm~=1.0, no-seed baseline G~=0.77).
  `torch_compile_rms_norm_default` baseline added (orig checkout; patch in _lab/harness/patches/).
  Report `_lab/harness/step1_harness_integrity.md`; scripts sanity_5way.py, crosscheck_bias.py. Commit bbd89997.
- **Step 2 map DONE (code-investigator).** Saved to `_lab/step2_code_map.md` (ReductionFact site, populate
  point, heuristic clone target, registration). reduction_loops: value>=size_hint -> persistent at codegen;
  Triton max_reduction_threads=None; default_config = persistent for N<=4096, looped chunk 4096 for N>4096.
- **Worker invocation 1 DONE -> v1 triton_reduction_tile ACCEPTED as champion.** Implemented ReductionFact
  + populate + TritonReductionHeuristic + registration. Bare-seed G_rms_norm referee-confirmed **0.979**
  (vs no-seed default 0.908, +7.8%). Commits b20b42ea/2c1163b5/7a31a28f.
- **Gates (parallel, GPUs 2/1/3):** results-referee ACCEPT (fresh subprocess/shape; worst -6.3% within
  backstop). adversarial-auditor PASS (overfit gap ~2.1%; flagged 2 defects: PERSIST_MAX is a FENCE at
  in-sample max -> persistent wins to ~256KiB/65536 fp32 elems, costs 1.38x on held-out; (2048,2048)
  narrative wrong). harness-integrity: NO autotuner bug -- the w32 'anomaly' was a bug in our
  oracle_field_diff.py (coupled warps x block; re-benched a fabricated block=1 config). Oracle trustworthy.
  Gate verdicts recorded in ledger.gate_verdicts.
- **Self-fooling caught + corrected** (honest, not cheating): (2048,2048) is a real ~1.4% regression (not a
  tie/artifact); oracle_field_diff.py has a lever-isolation bug. Both corrected in ledger; worker to fix
  notebook + the script next.
- **Worker invocation 2 -> v2 PROPOSED, then MECHANISM REJECTED (split gate, auditor wins).** v2:
  corrections + PERSIST_MAX 64->256KiB + looped warps32 + grid-occupancy branch; widened to sum (wash,
  root-caused to num_load=1) + long_sum (claimed 3.3x win). Commits 700e2bdc/47ae968f/f5a91837/91acb0b2.
- **Gates (parallel, GPUs 2/1):** results-referee ACCEPT (long_sum G~1.03 reproduces, no regressions,
  correctness honest) BUT adversarial-auditor **FAIL on mechanism**: the long_sum win is ENTIRELY from
  num_warps=32, not the looped/grid-occupancy branches. Controlled A/B (warps held EQUAL) shows
  **persistent/w32 beats the shipped looped/w32 on 8/9 long_sum shapes** (up to 2.65x on held-out). The
  grid-occupancy branch premise was a CONFOUND (worker A/B'd persistent/w16 vs looped/w32) and effectively
  fences long_sum. Both worker+referee only compared vs the catastrophic DEFAULT strawman.
- **Decision: REJECT v2 mechanism** (the safety gate working as designed). Champion stays v1-mechanism.
  SALVAGE for v3: corrections, PERSIST_MAX raise direction, num_warps=32 (move to persistent path), kernel
  widening. KEY METHODOLOGY LESSON for worker: A/B every branch vs the best SIMPLE alternative
  (persistent/w32), never vs the default strawman.
- **Worker invocation 3 -> v3 (honest fix). Gates SPLIT again (auditor wins).** v3 deleted the v2
  grid-occ + byte-fence branches; persistent workhorse + rnumel-based w32 ramp + structural looped tail
  (only above the 2^20 compile cap). long_sum 1.018->1.10 (+8%), O=1.005 (>1.0!), no regression, and
  v3==persistent/w32 (proving v2's branches didn't earn their place). Commit 2d087e78.
- **Gates (parallel GPUs 2/1):** referee ACCEPT. auditor FAIL on a NEW subtler fence: the `num_load==1`
  condition on the w32 gate is INERT in-sample (0/27 change), FALSE physics (matched-pair A/B: w32 keyed
  on rnumel for BOTH num_load), and HARMFUL out-of-sample (rms_norm/layer_norm large-rnumel want w32 but
  the gate gives w16, -30-40%). A curriculum-split fence dressed as physics.
- **Decision: REJECT the num_load gate; ACCEPT the rest of v3.** Surgical v4 fix: gate w32 on
  rnumel>16384 ALONE (byte-identical in-sample so O~1.005 holds; recovers 30-40% on held-out large-rnumel
  multi-load). LESSON for worker: gate on the DIRECTLY-measured property (rnumel for warps) and TEST the
  gate where it actually fires (synthetic/OOS large-rnumel multi-load), not just in-sample.
- Next: WORKER invocation 4 (surgical v4) -> then a FOCUSED auditor re-check (fence gone + no regression
  + OOS recovered). If PASS, v4 becomes champion (O~1.005 over 3 kernels). Then iter 5 = Product B.
- **Worker invocation 4 -> v4 DONE (surgical, one line).** Deleted the `num_load` condition; w32 step now
  gates on `rnumel > 16384` ALONE in `_num_warps`. Verified: (1) **in-sample byte-identical** —
  AUDITOR_gate_inert_proof.py = **0/27** mismatch, so O~1.005 + per-kernel G unchanged + correctness PASS
  (seed spot-checks rms_norm(2048,16384)=w16, long_sum(8,131072)=w32, both persistent). (2) **OOS recovery**
  — live v4 emits w32 for all held-out large-rnumel rms_norm(nl=2)/layer_norm(nl=3); measured w32/w16:
  rms_norm (16,131072)=0.617 / (16,262144)=0.616; layer_norm (16,131072)=0.641 / (16,262144)=0.540
  (recovers 30-46%). (3) **matched-pair physics** re-confirmed (AUDITOR_numload_warps_ab.py): num_load=2
  ALSO wants w32 at rnumel=131072 (w32/w16=0.57) -> w32 is rnumel-driven, num_load-agnostic. num_load/
  num_store kept in ReductionFact as DATA, just not gated. notebook+ledger updated, committed. NEVER pushed.
- **v4 auditor re-check: PASS -> v4 is the ACCEPTED CHAMPION** (O~1.005 over {rms_norm,sum,long_sum};
  generalizes to multi-load large-rnumel). Gate now rnumel-only; fence gone; no new cliff.
- **Product B DONE (iter 5, GPUs 2+3).** Seeded vs unseeded quick-autotune, N=3 seeds, cold cache, 4 shapes.
  HEADLINE Slice-2 time-to-95%: seeded **1.5-1.8x** less wall-clock on 3/4 shapes (rms_norm (2048,16384)
  1.78x, (8192,8192) 1.70x, sum (2048,16384) 1.52x; long_sum ~tie at noise floor but big early-budget win).
  Slice-1 gen0/gen1 seeded clearly ahead; full-budget guardrail passes (no regression). Curve shifts
  up-and-left. Commit 057f70bd; traces in logs/productB/.
- **CRITICAL BUG found by Product B: persistent seed silently DEGRADED on autotuner injection.**
  reduction_loops=[None] flat-encodes to looped-4096 (ReductionLoopSpec._encode_flat_value maps
  None->min(next_pow2(rnumel),4096); round-trips to None only if >=size_hint -> false for rnumel>4096). So
  the injected seed loses its DOMINANT lever (persistent), keeping only num_warps. Product-A bare-seed path
  (configs=[seed]) UNAFFECTED. => Product-B 1.5-1.8x is a LOWER BOUND; fixing the round-trip should enlarge it.
- Next: code-investigator (read-only) scopes the round-trip fix blast radius BEFORE touching core autotuner
  code. Then worker fixes _encode_flat_value + re-runs Product B. Then widen Product A (layer_norm-fwd next:
  num_load>=2, benefits from v4 w32 fix; then softmax = T2).

## 2026-05-29 (cont.)
- **layer_norm-fwd ACCEPTED** (referee+auditor PASS). 4-kernel active set {rms_norm,sum,long_sum,layer_norm};
  v4 heuristic byte-identical (0 lines changed); G_layer_norm=0.99 (+12% vs default); O_4kernel=0.9997.
- **Round-trip fix ACCEPTED** (auditor PASS). config_spec `_encode_flat_value(None)` .default()->.high so a
  persistent seed round-trips into the autotuner. Tests green (87/107/24); Product A byte-identical; gen0 seed
  now persistent; autotuner not degraded. Product B GREW: Slice-2 time-to-95% rms_norm (2048,16384) 1.78->1.89x,
  (8192,8192) 1.70->1.93x; long_sum dip at M=8 is noise-floor (flips to 2.17x at M=256). Commit 664a9524.
- **T2 support fully mapped** -> `_lab/t2_code_map.md`. KEY: T2 reduction axis is NOT a reduction=True
  block_size; found via ReductionLowering.block_index filtered against grid_block_ids (load-bearing for jsd).
  Plan: new register_user_tiled_reductions() populate (guarded if not reduction_facts), gate on
  reduction_facts==1, knob=block_sizes[red_idx], reuse rnumel/warps logic, Band-B watch for jsd (num_store=2).
- Next: WORKER widens to T2 (softmax_two_pass, kl_div, jsd) per t2_code_map. Then gate (auditor esp. on the
  gate-broadening no-regression + Band B). Then Band C (welford). All gates green so far; loop healthy.

## 2026-05-29 (endgame)
- **v5 ACCEPTED** (T1+T2, Bands A+B, 7 kernels, O=0.9982); **layer_norm earlier**; **v6 ACCEPTED**
  (cross_entropy + multi-load persist cap + welford decline; 8 seeded kernels, O=0.9874; welford out-of-scope).
  All gates PASS. Forward curriculum (8 seedable) complete.
- **Product B FULL (8 kernels): median time-to-95% = 1.94x** (cross_entropy 6x, jsd 2.9x, softmax 2.4x);
  referee ADMITTED. **VALIDATION sweep: generalizes, no overfit** (4/8 kernels beat in-sample OOS; the 2
  negative gaps = documented source ceilings).
- **Codegen-knob workstream EXHAUSTED with a verified negative** (consolidation auditor): pid_type=confound
  (flat wins everywhere); indexing=no matched win; eviction=real +11-23% but AUTOTUNER-ONLY (mutually
  contradictory per-slot patterns, no seedable rule); M-block=regime-conflict fence. The oracle Gs are REAL
  (sum oracle 1.75) but the residual is NOT seedable -> it's Product-B territory. **v6 at its true
  deterministic-seed Product-A ceiling.**
- **SOFT-CONVERGENCE FLAG (non-blocking, for the human):** the 8-kernel forward Product-A seed has reached
  its seedable ceiling (O~0.99, generalizes; remaining headroom is autotuner-only). You may consider this
  milestone "done" for the forward reduction-seed. Still grinding: welford (Band C, its own treatment) is the
  last untried in-curriculum angle; then the terminal TEST read + generalization report. (Backward Band D is
  explicitly deferred / out of scope for this run.)
- Next: welford Band-C attempt (keyed on is_structured_combine, correctness-first; keep out-of-scope if not
  generalizably+correctly seedable). Then FREEZE + terminal TEST read on the final champion.

## 2026-05-29 (FORWARD CURRICULUM COMPLETE — milestone)
- **v7 ACCEPTED (both gates PASS): welford Band-C seeded** via is_structured_combine (proven-generalizable
  structural signal; built a different structured-combine -> gate fires). welford 0.526->0.894 (+70%), CORRECT
  at non-pow2 incl PRIME N. All 9 forward kernels now seeded. O_in-sample=0.9765. Commit 53ed8762.
- **TERMINAL TEST read DONE (ledger-keeper, once).** In-sample->TEST geomean gap -0.114 (O_TEST 0.8628), NOT
  broad overfit: 8-kernel gap only -0.036; cross_entropy/kl_div/jsd BEAT in-sample on TEST. Dominant driver =
  welford -0.498 at prime/poorly-factored N (correctness forces combine=largest_pow2_div(N)->1 at prime N;
  fast masked tile is numerically WRONG -> a welford-KERNEL-STRUCTURE limit, not a seedable miss).
  rms_norm/layer_norm -0.13/-0.15 = tiny-M sub-25us noise-floor edges (at-ceiling on grid-occupied shapes).
- **Fresh-oracle re-validation: seed/oracle = 1.007 geomean** (within 0-1.6% everywhere) -> oracle hasn't
  drifted, champion holds, seed at the DETERMINISTIC-SEED CEILING. `_lab/FINAL_REPORT.md` written (commit 1de57007).
- **DELIVERABLES COMPLETE:** Product A (9 forward kernels, O~0.98, at ceiling, generalizes) + Product B
  (median time-to-95% 1.94x). Adversarial gates caught+rejected 5 cheats/confounds (v2 looped/grid-occ, v3
  num_load fence, pid confound, indexing no-win, M-block regime-conflict) -- the safety mechanism worked.
- **STATUS for human (non-blocking):** the defined forward reduction-seed task is COMPLETE & at ceiling.
  Genuinely-remaining angles are out-of-scope (backward Band D deferred) or not-seedable (codegen eviction =
  Product-B/autotuner; welford prime-N = kernel-structure). Continuing per never-stop with a GENERALITY
  STRESS-TEST: does frozen v7 seed NEW (non-curriculum) forward reductions well out-of-the-box?
- **GENERALITY STRESS-TEST PASSED (strong).** Frozen v7 seeds BRAND-NEW kernels well out-of-the-box: simple
  softmax (T1 whole-row, a different code path) G=0.881 (+34%), softmax_decomposed 0.865 (+34%), both correct;
  jagged_mean/jagged_softmax correctly DECLINE (dynamic rdim). Only residual = disclosed tiny-M ceiling. NO new
  gap. helion/ unchanged. Commit 5eb23859. -> Strong evidence the heuristic GENERALIZES to kernels it wasn't built on.
- **HARNESS-INTEGRITY RE-CERT: still unbiased** (hand-rolled vs TritonBench <1% on rms_norm + cross_entropy;
  headline G's match ledger; GPU2 had a co-tenant mid-run -> GPU pinning vindicated). Whole result body trustworthy.
  Commit be8210b3.
- **CODE REVIEW: APPROVE w/ minor fixes** (design/evidence/generalization/correctness PR-ready; 3 mechanical
  blockers + 2 cleanups). **PR-READINESS FIXES applied + verified byte-identical (9/9 seeds, sha256 identical):**
  return-annotation, ruff format, PIE804, dead _m_extent removed, boolean simplified, + a new unit test
  (TestTritonReductionHeuristic). ruff/pyrefly clean; tests 24/107/46/10 green. Commit d63faba7.
- **DELIVERABLE COMPLETE + MERGEABLE.** Product A (9 forward kernels, O~0.98, at deterministic-seed ceiling,
  generalizes to TEST + brand-new kernels) + Product B (1.94x time-to-target). Harness re-certified. Code PR-ready.
  Final report `_lab/FINAL_REPORT.md`. Adversarial gates caught+rejected 5 cheats/confounds across the run.
- Next: final comprehensive adversarial audit of the COMPLETE v7 (whole-curriculum overfit/cheat/fence sweep +
  FINAL_REPORT claim check). Then the forward reduction-seed task is at its honest end (Band D backward = wip-deferred,
  out of scope; codegen-eviction = autotuner-only; welford prime-N = kernel-structure). Soft-convergence: FORWARD COMPLETE.
- **CAPSTONE AUDIT of complete v7: PASS** (honest, general, coherent routing, correct; multi-load cap generalizes to
  wide softmax = real physics). Surfaced a missed seedable lever: welford apply-tile should be capped/looped at wide N.
- **v8 ACCEPTED (both gates PASS) — FINAL CHAMPION.** welford apply-tile cap + coupled combine cap: (262144,4096)
  0.706->0.757 (block_sizes ceiling closed), welford 0.894->0.9105, O 0.9765->**0.9786**. 8 kernels + welford small-N
  byte-identical; correct at non-pow2 + PRIME N. The capstone's "27% seedable" was itself an overstatement (apply-cap
  alone +2.5%; the 0.968 oracle is autotuner-only codegen knobs, proven by knob-isolation) -> CORRECTED in report+ledger.
  Commit daca67c4 (+ verdicts ca9d76d2). FINAL_REPORT finalized to v8 (TEST carried from v7 per read-once; v8's
  welford-large-N change doesn't touch the dominant prime-N TEST driver).
- **=== RUN COMPLETE (forward reduction-seed task at its honest ceiling) ===**
  Product A: v8, 9 forward kernels, in-sample O=0.9786, generalizes (TEST -0.036 ex-welford; +34% on BRAND-NEW kernels;
  fresh-oracle seed/oracle=1.007). Product B: median time-to-95% 1.94x (~half the autotune budget). Harness re-certified
  unbiased. Code PR-ready (lint/type/tests green + unit test; byte-identical-verified fixes). Adversarial chain
  caught+corrected ~7 cheats/confounds/overstatements. All residuals are kernel-source (cross_entropy online-softmax;
  welford prime-N count-logic) or autotuner-only codegen knobs (eviction/indexing/pid) = Product-B territory, NOT
  deterministic-seedable. NO in-scope seedable Product-A angle remains.
- **SOFT-CONVERGENCE FLAG (strong, non-blocking, for human):** the defined forward reduction-seed task is COMPLETE +
  at ceiling + exhaustively verified. The next workstream (backward kernels, Band D) is EXPLICITLY wip-deferred =
  a scope decision for the human. Durable state (ledger/notebook/FINAL_REPORT/harness) is ready to resume Band-D or any
  scope expansion losslessly. Not stopping on my own judgement; surfacing the converged signal per the manager protocol.
- **GENERALITY (NEW REDUCTION OP) PASSED — final/strongest dimension.** Frozen v8 seeds max/min (a different
  accumulator) out-of-the-box: G_seed~1.0, EXACT correctness, op-agnostic levers confirmed (keys on
  size_hint/num_load/bytes, not the op). helion/ byte-identical. Commit 46ba7059. => Heuristic generalizes across
  ALL THREE dimensions: shapes (TEST -0.036 ex-welford), structures (softmax T1<->T2 +34%), reduction ops (max/min).
  No gap. This conclusively validates "general heuristics."
- **FORWARD TASK EXHAUSTIVELY COMPLETE + AT CEILING + GENERALITY-PROVEN (3 dims).** No in-scope seedable Product-A
  angle remains; all generality dimensions validated. The only further work is wip-DEFERRED (backward Band D) =
  a human scope decision. Durable state ready to resume losslessly. Holding at the converged forward milestone.

## 2026-05-30 (RUN 2 begins — new goals, same orchestration)
> Run 2 works in a SEPARATE worktree `wt-reduction-2` (branch `reduction-heuristics-run2`) off the v8 tip
> `25561778`. `_lab/run2_notebook.md` is the live run-2 reasoning trace; run-1's notebook.md/FINAL_REPORT/
> HANDOFF are historical reference. `wt-reduction` is read-only reference — do NOT edit it.
- **Setup DONE.** Worktree `wt-reduction-2` off v8 tip `25561778`; `_lab/` (165 tracked files) inherited.
  GPUs 0(1GB co-tenant)/1/2/3 — 1/2/3 idle; pin per-job, never 2 timing runs on one GPU.
- **Step 0 wiring RE-PROVEN on wt-reduction-2.** (1) helion+examples resolve to wt-reduction-2 — note
  `wt-reduction-2`.startswith(`wt-reduction`)=True, so run-1 scripts that hardcode the old path +
  `sys.path.insert(0, OLD)` would SILENTLY run old code; run-2 scripts pin `wt-reduction-2` exactly and rely on
  PYTHONPATH (no sys.path.insert). (2) codegen flows. (b) heuristic edits flow: sentinel in `_num_warps`
  changed live seed w8→w16, reverted. (c) tritonbench rms_norm operator resolves to ORIGINAL checkout w/
  `torch_compile_rms_norm_default` present. (3) bare-seed rms_norm(2048,4096) v8 persistent/w8, no-autotune,
  used, correct(1.9e-6), 35.46us/1.2% spread.
- **Step 1 sanity GREEN.** rms_norm(8192,8192) fp32 GPU1: default 336.3 / seed(v8) 252.6 / tc-default 250.3 /
  tc-max 248.4 us → G_seed=0.991, G_default=0.744, seed/tc_max=1.017. Reproduces run-1 exactly; no HALT.
- **Ledger seeded to v8 floor** (champion O=0.9785; `run2` marker added). >10% per-kernel referee-confirmed
  regression backstop in force across ALL kernels+shapes (incl in-sample-v2 + new kernels).
- Next: GOAL 1 (welford source fix + re-derive Band-C), then interleave Goal 4 (in-sample-v2) → Goal 2
  (codegen knobs) → Goal 5 (new-kernel generality). Phase II (Goal 3) after Phase I settles.
- **GOAL 1 DONE + BOTH GATES PASS (2026-05-31).** welford source bug fixed + Band-C re-derived. Commits
  ddf8fc34 (source `Tn=(tile_n.index<n).sum()`), 43492809 (band-C: independent byte caps, deleted
  divisor+coupling). welford in-sample G(orig4) 0.911→0.926; prime-N 1543 0.082(WRONG)→0.958(CORRECT+FAST);
  1536 +6.7%, 2560 +12.2%; 8 non-welford kernels BYTE-IDENTICAL. Referee ACCEPT, auditor PASS (fix real,
  gate structural+generalizes to 2 synthetic kernels; FLAG: byte-cap values curriculum-fit→Goal 5).
  Source fix ships w/ deliverable (also fixes un-seeded default at prime N). Confirmed v8 welford oracle was
  divisor-confounded. Big Goal-2 welford codegen residual: N=4096 seed 0.76 vs oracle 0.961 (TD+eviction).
- Next: GOAL 4 (in-sample-v2 shapes) → GOAL 2 (codegen knobs, big welford+small-N headroom) → GOAL 5.
- **GOAL 4 DONE.** in-sample-v2 split added to list_of_kernels.md (firewall-validated; snapshot _lab/
  list_of_kernels_run2.md) + baselined (overall G_seed 0.890). Commit 5db6d42b.
- **GOAL 2 (eviction) DONE + BOTH GATES PASS (2026-05-31).** Commit 136d187a + 5b4133dc. Found the
  separating workload property run-1 missed: per-load cache RESIDENCY. num_load==1 stream (sum/long_sum) →
  'first' (sum +29%, long_sum +16% geomean, 0 regressions); is_structured_combine re-read (welford) →
  ['last']+['first']*(n-1) (welford 4096 0.760→0.951, +6.68% geomean, small-N neutral). REJECTED (recorded):
  softmax re-read (−10% at 4096,2048), rms/ln x-only (noisy), TD-on-welford (OOM). Referee PASS (full-curric
  matched A/B + noise rechecks), auditor PASS (workload-keyed, 8 byte-identical, really-used). FLAG:
  is_structured_combine eviction welford-only → Goal 5.
- **GOAL 6 (device_ir) partial DONE.** Commit c07d9751: _count_reduction_workload int-equality caveat
  comment + symbolic-fallback symmetry at dtype site (behavior-neutral). REMAINING Goal-6: rms_norm TEST G
  regen + welford TEST re-read (consolidated TEST pass at Phase-I end).
- IN FLIGHT: in-sample O re-measurement (Goals 1+2 lift, GPU1/2); codegen-knob explorer (num_stages/TD/
  range_* on rms/ln weak shapes, GPU3). Then: finish Goal 2 remaining knobs + pid_type-explicit, Goal 5
  (new kernels), then Phase II Goal 3.
- **PRODUCT-A MILESTONE: in-sample O 0.9786 → 0.9980 (+2.0%)** after Goal1+Goal2-eviction. Seed now ≈
  torch.compile-default parity on the geomean. Drivers: sum 0.937→1.019, welford 0.911→0.975, long_sum
  1.099→1.138 (all eviction / Goal-1). rms/ln/softmax/kl/jsd/ce ~unchanged (byte-identical; ±noise).
  Residuals: CE (8192,131072) 0.539 = SOURCE ceiling (Goal-5 online-logsumexp); rms_norm (2048,2048) 0.871
  + small-N (codegen-knob explorer running). CHAMPION advanced (O improves, both gates passed, no >10% regr).
- **GOAL 2 COMPLETE (2026-05-31).** WINS: load_eviction_policies (sum/long_sum 'first', welford re-read [last,first]
  → in-sample O 0.9786→0.9980) + pid_type='flat' explicit (principled constant, behavior-neutral). NULL (honest,
  matched-lever A/B recorded w/ raw numbers in ledger.run2.codegen_knobs_other + _lab/logs/run2/knob_explore.json):
  num_stages (noisy, regresses ln 256,5120 -5%), tensor_descriptor (doesn't engage/OOMs), range_* (no win).
  KEY: rms/ln in-sample-v2 "weak" shapes (256-1024 rows) are NOISE-FLOOR (fresh default G ~1.0-1.13, not 0.75-0.88)
  — seed is at tc-default parity on reliably-measurable shapes. Genuine residuals are kernel-source (CE wide-vocab
  → Goal-5) / out-of-scope (long_sum split-K) / irreducible (welford wide-N codegen-OOM). pid lock covers new
  forward kernels (Goal 5) too.
- Next: GOAL 5 (new-kernel generality probes — structured-combine worker running on GPU2; also validates the
  is_structured_combine + eviction generality flags) → then Phase II GOAL 3 (Product B) once Phase I settles.
  Remaining Goal-6: consolidated TEST re-read (welford + rms_norm G).
- **GOAL 5 + GOAL 6 DONE → PHASE I COMPLETE (2026-05-31).** Goal 5: structured-combine generality VALIDATED
  (standardize within 1.2% of best everywhere — resolves the welford-only flag); multi-load + Band-B shown
  already-multi-kernel; cross_entropy_online (single-pass, verified) closes the wide-vocab SOURCE ceiling
  ((8192,131072) 0.539→0.956, CE in-sample 0.917→0.975, regime-best 0.995). Goal 6: device_ir robustness +
  consolidated TEST re-read DONE & RE-LOCKED — **welford TEST 0.396→0.892 (PRIME 0.082-wrong→0.905-correct+
  fast); rms_norm TEST 0.828→0.841 (0.992 excl noise-floor); TEST O 0.863→0.946.**
- **PHASE-I HEADLINE: in-sample O 0.9786→0.998 (regime-best ~1.005); TEST O 0.863→0.946; in-sample↔TEST gap
  0.115→~0.05.** Heuristic (triton.py) FROZEN for Phase I.
- **PHASE-II PREREQ VERIFIED:** the run-2 seed (load_eviction_policies + pid + reduction_loops/block_sizes)
  survives the autotuner flat-encode round-trip PRESERVED (welford/sum/long_sum) — Product-B seeded arm carries
  the eviction win (no round-trip bug; run-1's persistent fix in place).
- IN FLIGHT: capstone adversarial auditor (whole run-2 deliverable; gates Phase II). NEXT: Phase II Goal 3a
  (budget reduction) + 3b (beat max-effort, multi-seed portfolio) once capstone PASSES.
- **GOAL 3 (Product B) DONE + GATED → RUN 2 COMPLETE (2026-05-31).** 3a BUDGET REDUCTION validated on 3
  kernels (welford/softmax/cross_entropy): seeded-QUICK matches unseeded-FULL optimum within 0.1-0.9% → drop
  full→quick budget (welford ~30x wall-clock reduction to the same optimum). 3b BEAT-MAX-EFFORT = HONEST NULL
  (welford 4096 hard-coupling + sum Band-A control, N=5/arm: both arms reach optimum 5/5 at full; seed within
  1.3% = at ceiling). A preliminary incomplete-data "beat" was CAUGHT + corrected (anti-lucky-run discipline).
  Product-B auditor PASS (seed-injection genuine, 3a real, 3b null honest, no over-claim, no cherry-picking).
- **=== RUN 2 COMPLETE — all 6 goals delivered + independently gated ===** in-sample O 0.9786→0.998, TEST O
  0.863→0.946, prime-N welford 0.082(WRONG)→0.905(correct+fast). Welford fix + simplified Band-C; eviction
  (overturns run-1's "autotuner-only"); pid owned; codegen-knobs honest null; in-sample-v2; generality
  (standardize + cross_entropy_online closing the wide-vocab source ceiling); device_ir robustness; TEST
  re-locked; multi-seed plumbing. Product B: 3a budget reduction (the win) + 3b honest null. FINAL_REPORT_run2.md.
  ruff/pyrefly clean, tests pass, 8 non-touched kernels byte-identical, capstone+Product-B auditors PASS. NEVER pushed.
