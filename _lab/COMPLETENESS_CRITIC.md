# COMPLETENESS-CRITIC — 3-stage reduction-heuristic stack

Branch `reduction-3stage-stack` (worktree `helion-3stage`), base `0676dd32`. Analysis-only (no GPU).
Default posture: something is uncovered. Each gap is {kind, claim/cell, where, severity, cheap?}.

## What I independently re-verified (held)
- **HARD INVARIANT is AIRTIGHT.** I re-diffed the recorded `normalized_cfg` per cell across the WHOLE
  chain myself (not trusting the banked claim): base→partA→partB→stage2C→stage3→stage3_dtypefix AND
  base→stage3_dtypefix end-to-end = **0/739 changed, 0 keyset mismatch** every hop. The 739 matrix
  genuinely covers fp32(265)/bf16(265)/fp16(209) × train(358)/val(181)/robustness(200) for the 9
  standard kernels. The 9 reduction kernels are byte-identical end-to-end across all 3 dtypes. SOUND.
- The selection-only skip is legitimate: the device_ir relaxation is `not has_matmul`-gated (no-op for
  all 9+8 non-matmul kernels) and `build_matmul_reduction_epilogue_facts` returns early at
  `len(matmul_facts)!=1`. Config-identical ⇒ Triton-identical for the protected set. SOUND.
- `LIVE_PERSIST_SCALE` is genuinely GONE (belonged to the abandoned naive scaled-cap design); the
  shipped `LIVE_PERSIST_BUDGET=3×ROW_PERSIST_MAX_BYTES` is additive and provably can only REMOVE
  persistence, so no looped curriculum cell can flip up. Window argument is sound.
- The three changed source files py_compile clean; worktree is committed (no uncommitted drift).
- The M_BLOCK ladder math reproduces the answer key (bf16 64@N≤512/32@N1024/16@N2048; fp32 32 cap).

---

## RANKED GAPS (severity × cheapness)

### 1. [REAL — should-fix, CHEAP] The loss corner (131072,512,1024) was NEVER benched — Stage 3
- **Claim:** task required "the seed must DECLINE or at least not regress" at this shape. STACK_SUMMARY
  scopes the claim "N=1024 marginal."
- **Where:** it IS in the test split (`mm_epilogue_bench.py:61`) but NO `ROW` for it exists in ANY
  `s3_*.out` or the ledger. I grepped all outputs — zero rows for 512×1024.
- **Why it bites:** at N=1024 the seed emits `[32,32]` (= default block_sizes) BUT with num_stages=3 +
  num_warps=8, whereas default is num_stages=1, num_warps=4. So the seed is NOT identical to default at
  the loss corner — it changes 2 knobs and is unmeasured. "Does not regress" is asserted-by-ladder, not
  measured. We saw exactly this category of surprise blow up once already (fp32 N=256 regressed to
  0.53× before the dtype fix). At M=131072,N=1024 a wrong num_warps/stages could regress materially.
- **Cheap-to-close?** YES — one bench row (it's already in the corpus, just run it).

### 2. [REAL — should-fix, CHEAP] fp32 footprint fix validated on ONLY 1 of 7 epilogues — Stage 3
- **Claim:** commit `c265d421` "fp32 footprint fix"; STACK says the fp32 gap is closed.
- **Where:** `s3_fp32fix.out` + `s3_dtype.out` cover ONLY `matmul_rms_norm` at fp32/fp16. softmax,
  sum, l2_normalize, max, logsumexp were benched **bf16-only** (`s3_softmax/sum/l2/max/lse.out`).
- **Why it bites:** the pre-fix run (`s3_dtype.out`) caught a 2× regression (svd 0.53, seed/best 0.34)
  that ONLY showed up at fp32. The fix `max_m = MAX_M_BLOCK*2//itemsize` is dtype-faithful and SHOULD
  generalize (it's an itemsize factor, kernel-blind), but the only dtype where a regression was ever
  observed is the one dtype NOT swept on 6/7 epilogues. The composed fact is "blind to which reduction
  it is," so the seed IS identical across epilogues at a given N — but that is an argument, not a
  measurement, and softmax/logsumexp have an extra max/exp pass that changes the register pressure.
- **Cheap-to-close?** YES — re-run the existing 6 epilogue benches at `--dtype fp32` (corpus exists).

### 3. [REAL — should-fix, MEDIUM] "Beats torch.compile" claim is bf16-and-N≤512 only; the held-out
       TEST shape actually LOSES to tc — Stage 3
- **Claim:** REPORT/STACK headline "beats torch.compile 1.5-1.73×"; STACK line 60 cites the held-out
  TEST shape (196608,256,768) as a clean generalization win.
- **Where:** `s3_lse.out` for (196608,256,768): `best_vs_tc_default=0.7247`, `best_vs_tc_max=0.9652`
  — the seed (≈best) LOSES to tc by ~38% / ~4% there. The "beats tc 1.5-1.73×" numbers are all
  N=256/512 bf16; the driver's independent repro (gate_A_repro_driver_run) only covers rms_norm
  256/512 bf16.
- **Why it bites:** the tc-win is real but narrowly scoped, and the one HELD-OUT TEST shape is a tc
  loss that the summary presents (via seed/best=0.965 vs the grid) as a generalization success without
  noting it's below tc. Gate E "no overfit" verdict is about seed-vs-grid, which is fine, but the
  user-facing "beats tc" headline does not hold at the test shape. This is a scoping/over-claim, not a
  regression (the kernel is faithful; it's just not a tc win at N=768).
- **Cheap-to-close?** YES — it's a one-line scope correction in REPORT/STACK ("beats tc at N≤512;
  at N≥768 it trails tc — the small-N fusion regime"). No new bench needed (data already exists).

### 4. [REAL — should-fix, CHEAP] group_norm (the newest recognizer, sub-problem C) was NOT in the
       Stage-2 freeze/TEST read — Stage 2
- **Claim:** Gate E (`gate_E_freeze_test` in ledger) attests generalization on the freeze split.
- **Where:** the freeze read covers **bias_grad, dyt, instance_norm only**. group_norm — the kernel
  the multi-materialized recognizer in commit `faecdbfc` was BUILT for, and the riskiest of the three
  changes (a ">=1, pick dominant" generalization of the recognizer) — is absent from the freeze read.
- **Why it bites:** group_norm's perf is the most bimodal in the stack (large-N 1.16-1.89 but wide-S
  0.035-0.27 and "kernel-authoring-bound"). The freeze split is exactly where a recognizer that picks
  the wrong dominant axis would surface. Untested on the held-out split.
- **Cheap-to-close?** YES — `probe_stage2.py group_norm` on the test shapes.

### 5. [REAL — nice-to-have] No helion test-suite / golden-codegen run anywhere in the stack — all 3
- **Claim:** correctness rests entirely on the config_recorder byte-identical diff.
- **Where:** grep of `_lab/` for pytest/test_examples/lint/pyrefly = empty. No `test_*.py` run logged.
- **Why it bites:** the config diff proves SELECTION is unchanged for the 9+8 curriculum kernels, but
  the source edits are in the compiler (Part-A `_reduction_rblock` refactor, the new `body_live_tiles`
  walker fact in the single collect pass, the device_ir guard relaxation, the new composed-fact
  builder). A config-only diff does not exercise golden `.expected` codegen, kernels outside the
  curriculum, or the new walker fact's interaction with odd reduction structures. The new
  `_graph_peak_live_by_axis` liveness sweep is entirely new code with no unit test.
- **Cheap-to-close?** MEDIUM — `pytest test/test_reductions.py test/test_matmul.py test/test_examples.py
  -x` (CUDA box, minutes). Highest-value single action: it's the one correctness backstop never run.

### 6. [REAL — nice-to-have] matmul_softmax (131072,256,256) has NO valid config + acc FAIL — Stage 3
- **Claim:** REPORT calls it "a borderline bf16 tolerance row (0.098 vs 0.09 floor)…not a seed issue."
- **Where:** `s3_softmax.out`: default AND seeded both `FAIL:max_abs=0.09766`, best=`no-valid-cfg`,
  `grid_valid_cfgs:0`. So at this shape NOTHING in the grid produces a valid config — it's not merely a
  tolerance miss, the grid is EMPTY. The arm-fairness holds (all arms fail identically), so it's not a
  seed regression, but "borderline tolerance" undersells that there is no admissible config at all.
- **Why it bites:** mostly honesty — the row is correctly excluded from the geomean, and softmax N=512
  is a clean 1.44× win, so the family claim stands. But if matmul_softmax small-N is a target, N=256 is
  effectively unsupported, not "borderline."
- **Cheap-to-close?** Characterize-only (one line): "matmul_softmax N=256 has no admissible config
  (grid empty); the supported softmax regime is N≥512."

### 7. [COSMETIC] Stage-1 flj fp16 perf never measured; Stage-2 M-reduction perf is fp32-only.
- Stage 1 flj win is bf16+fp32 only (no fp16 perf row). The byte-cap is itemsize-faithful so fp16
  routes like bf16 — low risk. Stage 2 perf is "fp32 train"; LANDSCAPE asserts seed config is
  "identical at fp32 and bf16" (norm family fp32-promotes), so config is dtype-invariant — but
  bias_grad's M_COLLAPSE_MAX_CTA=256 spill crossover was measured at fp32 (worst case), so bf16 is
  strictly safer. Acceptable; note that the bf16/fp16 perf is inferred-by-structure, not measured.

### 8. [COSMETIC] Stage-3 heuristic is HARDWARE_TARGETS=sm90-only; constants are answer-key-derived.
- `TritonMatmulReductionEpilogueHeuristic.HARDWARE_TARGETS=(("cuda","sm90"),)` — the composed fact
  silently does nothing off-H100. ACC_BUDGET_BYTES=131072 is explicitly "derived from the grid answer
  key" so M_BLOCK lands on 64/32/16; Gate F's N=1024/N=2048 boundary tests give it legitimacy and the
  footprint law is monotone, but the constant is a fitted H100 value (acknowledged). Not a blocker for
  an sm90 PR; flag the hardware scope in the PR description.

### 9. [COSMETIC] Known doc-drift, already flagged by Gate H: triton.py line-797 comment still says
       'persist_scale=1' though the param is now `live_budget`. One-line fix.

---

## HIGHEST-PRIORITY RECOMMENDATION
Bench the **Stage-3 loss corner (131072,512,1024)** AND **re-run the 6 non-rms epilogues at fp32**
(gaps #1+#2) — both are the same cheap action (the corpus + harness already exist; ~one bench pass),
and together they close the exact regression class that already bit once (the fp32 N=256 2× regression
the dtype-fix caught). The seed at the loss corner is NOT identical to default (it flips
num_warps/num_stages), so "does not regress" is currently asserted-by-ladder, not measured — that is
the single unverified possible-regression in the stack. If a GPU pass is run at all, fold in
`pytest test/test_reductions.py test/test_matmul.py` (#5) as the correctness backstop no stage ran.
