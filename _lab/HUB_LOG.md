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
