# FINAL REPORT — RUN 2 (H100/fp32 Triton Reduction Seed-Heuristic)

**Built on:** v8 (run-1 deliverable, in-sample O=0.9786). **Worktree:** `wt-reduction-2`, branch
`reduction-heuristics-run2`, off v8 tip `25561778`. **Never pushed.** Read `HANDOFF.md` (run-1 traps)
+ this + `run2_notebook.md` + `ledger.json["run2"]`.

## 0. HEADLINE (every change independently results-referee + adversarial-auditor gated; capstone PASS)

| metric | v8 (run-1) | **run-2** | driver |
|---|---|---|---|
| **in-sample O** (geomean Gₖ, 9 fwd kernels, original shapes) | 0.9786 | **0.998** | welford fix + eviction |
| **TEST O** | 0.8628 | **0.946** (0.964 excl noise-floor) | welford prime-N fix |
| **welford TEST G** | 0.396 | **0.892** | source fix + Band-C re-derive |
| **welford PRIME-N (262144,1543)** | 0.082 (NUMERICALLY WRONG) | **0.905 (CORRECT + FAST)** | the Tn bug fix |
| in-sample↔TEST gap | −0.115 | **−0.05** | — |

Per-kernel in-sample G (run-2): rms_norm 0.979, sum **1.019** (was 0.937), long_sum **1.138** (1.099),
layer_norm 0.985, welford **0.975** (0.911), softmax 0.960, kl_div 1.028, jsd 0.997, cross_entropy 0.916
(→ **0.975 with the online variant**; regime-best 0.995).

## 1. What changed (the run-2 deliverable, on top of v8)

1. **welford correctness fix (Goal 1, ships independently).** `Tn = chunk.size(-1)` (constexpr tile width)
   → `Tn = (tile_n.index < n).sum()` (masked valid count). Fixes the mean/var/count at non-divisor N.
   **Prime-N 1543: 0.082 (wrong) → 0.905 (correct + fast).** Also fixes the un-seeded default at prime N.
2. **Band-C re-derivation (Goal 1).** With the source correct, the pow2-divisor combine constraint + the
   apply↔combine coupling (both bug-artifacts) are DELETED. Band-C is now two INDEPENDENT byte caps:
   combine = min(np2(N), 32 KiB/itemsize) [persistent — looping it regresses via the serial recurrence];
   apply = persistent np2(N) if per-row valid bytes ≤ 12 KiB else looped 8 KiB chunk. Wins at non-pow2
   (2560 0.857→0.973, 1536 0.924→0.986). 8 non-welford kernels byte-identical.
3. **load_eviction_policies seeding (Goal 2 — overturns run-1's "eviction is autotuner-only").** The
   separating workload property run-1 missed = **per-load cache RESIDENCY**: num_load==1 streamed reduction
   (sum/long_sum) → 'first' (**sum +29%, long_sum +16%** geomean); is_structured_combine RE-READ (welford
   x combine→apply) → ['last']+['first']*(n-1) (**welford 4096 0.76→0.95**). rms_norm/layer_norm (fused x +
   reused operands) left DEFAULT — blanket policy regresses (the genuinely-contradictory case run-1 saw).
   REJECTED with receipts: softmax re-read (−10% mid-N), rms/ln x-only (noisy), TD-on-welford (OOM).
4. **pid_type='flat' explicit (Goal 2).** Principled constant (run-1 matched-lever A/B; behavior-neutral).
5. **codegen knobs num_stages / tensor_descriptor / range_* (Goal 2): honest NULL.** Matched-lever A/B w/
   noise-floor characterization: no clean workload-keyed win. KEY: the rms/ln "weak" in-sample-v2 shapes
   were sub-25µs NOISE-FLOOR (±25% same-config swing); the seed is at parity on reliably-measurable shapes.
   tensor_descriptor is inert unless ≥2 rows in the leading tile (seed tiles rows at 1) and OOMs where it
   engages (welford). Recorded as null per the brief.
6. **in-sample-v2 curriculum (Goal 4).** Real-AI-workload shapes (Llama/GPT vocabs, hidden dims, small-M,
   non-pow2), firewall-validated disjoint from sealed TEST for the 7 non-re-read kernels.
7. **generality (Goal 5).** `examples/standardize.py` (plain two-moment LayerNorm — a DISTINCT combine path)
   fires is_structured_combine + the Band-C recipe transfers within 1.2% → resolves the auditor's
   "is_structured_combine welford-only" flag. multi-load (CE+rms+ln) and Band-B (kl_div+jsd) already
   multi-kernel. **`cross_entropy_online`** (single-pass online logsumexp) closes the wide-vocab SOURCE
   ceiling: (8192,131072) **0.539→0.956**; CE in-sample geomean 0.917→0.975 (−7% at V=65536, disclosed).
8. **device_ir robustness (Goal 6).** int-equality-(`last==size_hint`) caveat + symbolic-fallback symmetry
   (behavior-neutral). Consolidated TEST re-read done + re-locked.
9. **multi-seed plumbing (Goal 3b).** `get_seed_configs()` hook (opt-in; Product-A/3a unaffected).

## 2. Product B (Goal 3)

- **3a — budget reduction (the seeded WIN), validated on 5 kernels across ALL BANDS.** seeded-QUICK reaches
  the unseeded-FULL optimum within **0.1–0.9%** (welford Band-C, kl_div Band-B, rms_norm T1-persistent,
  cross_entropy T1-looped-multiload, softmax T2-persistent) → the autotune budget can drop full→quick. The
  convergence/time-to-target advantage is DRAMATIC where the optimal config is hard for the blind search
  (welford at-optimum gen-0 vs unseeded gen-6.5/380s; kl_div gen-0 vs gen-2/271s; rms_norm gen-0 vs
  gen-1/116s) and modest where easy (cross_entropy 1.2–1.36×, softmax 1.35×). Exceeds run-1's quick-only
  1.94× with a full-effort, all-bands characterization (the floor corroborated and surpassed).
- **3b — beat max-effort: HONEST NULL.** Pre-registered pilot (welford 262144,4096 hard eviction coupling +
  sum 2048,16384 Band-A control), N=5/arm, full budget. Both seeded-portfolio AND unseeded reach the optimum
  reliably (5/5); the seed is within ~1.3% of it (at ceiling). At max effort the LFBO search is thorough
  enough to find the optimum. A preliminary incomplete-data "beat" was caught + corrected by the complete-N
  + fair-re-bench protocol (anti-lucky-run discipline). NOT claiming a beat.

## 3. Residuals (honestly attributed, not hidden)

- cross_entropy wide-vocab: kernel-SOURCE ceiling → CLOSED by the online variant (ships).
- long_sum few-row (4,524288)=0.66: grid-starved (4-8 programs) → split-K, wip-DEFERRED out-of-scope.
- welford wide-N (5120/8192) + small-N noise-floor: codegen-OOM / measurement-noise ceilings.
- num_stages/TD/range_*: no seedable win (recorded null).

## 4. Code changed (helion/, shippable)
`_compiler/autotuner_heuristics/triton.py` (Band-C re-derive, eviction, pid, get_seed_configs),
`autotuner/config_spec.py` (ReductionFact docstring), `_compiler/device_ir.py` (robustness comments +
symbolic-fallback symmetry), `_compiler/autotuner_heuristics/registry.py` + `__init__.py` (multi-seed hook),
`examples/welford.py` (Tn fix), `examples/standardize.py` (new probe), `examples/cross_entropy.py`
(+cross_entropy_online). ruff/pyrefly clean; 56 heuristic + best_available tests pass.
