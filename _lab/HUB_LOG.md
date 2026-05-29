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
- Next: WORKER invocation 2 -- (a) corrections, (b) raise PERSIST_MAX by synthetic-crossover evidence +
  strengthen looped branch (rms_norm G unchanged), (c) WIDEN to sum + long_sum (T1 Band A; long_sum
  exercises the looped branch) -> recompute O over active kernels. All GPUs idle (bg autotune finished).
