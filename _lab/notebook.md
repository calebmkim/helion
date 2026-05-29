# Lab Notebook — Reduction Autotuner Heuristics

> The DURABLE source of truth for the hill-climb. A fresh worker reads this to continue losslessly.
> Maintained by the worker (decisions + empirical why; tried-and-rejected + why; open hypotheses;
> champion). The hub appends gate verdicts. Keep it current at every clean iteration boundary.

## Champion (current best heuristic)
- **v6 `triton_reduction_tile` (WORKER-PROPOSED 2026-05-29) — cross_entropy widening + welford out-of-scope.**
  Adds **cross_entropy** (T1, num_load=3, num_reduction_ops=2, wide-V logsumexp+gather) to the active set.
  ONE new general branch (the **multi-load persist byte-cap**): for `num_load >= 2` AND
  `rnumel*itemsize > MULTILOAD_PERSIST_MAX_BYTES (128 KiB)` the reduction goes LOOPED instead of persistent.
  Validated by a matched-WARPS persistent-vs-looped A/B *across num_load* (`persist_crossover_by_numload.py`,
  `ce_crossover_tight.py`): num_load=1 (sum) ties persistent==looped at EVERY rnumel up to 1 MiB (NO change,
  the structural-cap policy stays correct for it); num_load>=2 (rms_norm/layer_norm/cross_entropy) persistent
  WINS to ~128 KiB then LOSES 1.6-4x (sharp crossover at 256->288 KiB). **welford: classified T2 but
  OUT-OF-SCOPE** for the single-axis seed (it has a SECOND user tile over the reduction extent — the normalize
  pass — that the seed floors to width 1, ~10-20x slower AND numerically wrong for non-pow2 N). Heuristic now
  DECLINES welford (a general "single non-grid tile" guard in `register_user_tiled_reductions`) → falls back to
  the correct un-seeded default. T1+T2 no-regression BYTE-IDENTICAL (27/27 inert proof; test_reductions 24;
  test_examples 19). **G_cross_entropy = 0.915** (+52% vs default 0.600). O_7kernel UNCHANGED 0.9982;
  **O_8kernel = 0.9874** (cross_entropy mechanically dilutes the geomean — per-kernel no regression). See the
  "v6 — cross_entropy + welford" section below. HELPER_REQUEST queued (referee + auditor). Awaiting commit hash.
- **v5 `triton_reduction_tile` (ACCEPTED 2026-05-29; commit 6430b37e/63142bde) — T2 SUPPORT.** Adds user-tiled/
  manually-looped reductions (softmax_two_pass T2 Band A; kl_div, jsd T2 Band B) on top of the v4 T1 champion.
  T1 seeds are BYTE-IDENTICAL (38/38 unchanged). ONE new branch (the Band-B R_BLOCK footprint cap, gated on
  `num_tiled_accumulators>=1`, validated by matched A/B vs persistent). Per-kernel G: softmax 0.967 (+120% vs
  default), kl_div 1.029 (+163%), jsd 0.996 (+50%); 4 T1 kernels unchanged. O_7kernel=0.9982 (referee+auditor
  both PASS). See the "v5 — T2 SUPPORT" section below for full evidence.
- **v1 `triton_reduction_tile`** (ACCEPTED 2026-05-28). Referee-confirmed **G_rms_norm = 0.979** vs
  un-seeded `default_config` baseline 0.908 (+7.8%). Auditor PASS. Code:
  `helion/_compiler/autotuner_heuristics/triton.py` `TritonReductionHeuristic`.
- **v2 `triton_reduction_tile` — MECHANISM-REJECTED 2026-05-29 by the adversarial auditor.** The long_sum
  win was REAL but came ENTIRELY from num_warps=32, NOT the looped/grid-occupancy branches. Controlled
  A/B holding warps EQUAL: persistent/w32 BEATS the v2 shipped looped/w32 seed on 8/9 long_sum shapes. The
  grid-occupancy branch's premise was a CONFOUND (compared persistent/w16 vs looped/w32). v2's branches
  were net-harmful + effectively fenced long_sum's shapes. SUPERSEDED by v3.

- **v3 `triton_reduction_tile` (2026-05-29) — the HONEST FIX; SUPERSEDED by v4 (auditor FAILED its
  `num_load` gate).** Referee ACCEPT but the adversarial auditor rejected the `num_load==1` w32-gate
  condition (see v4 below for the resolving one-line fix). v3's other parts (structural persistent/looped
  split, branch deletions, rnumel w32 breakpoint) all STAND in v4 byte-identical. WORKER-measured (do_bench, fp32;
  rms_norm/sum on GPU2/3 median-of-7, long_sum fresh-process-per-shape median-of-9; awaiting referee).
  Per-kernel G_seed vs un-seeded default baseline:
  - rms_norm_fwd: **0.9815** (champion 0.979 — UNCHANGED; byte-for-byte identical codegen/warps to v1/v2).
  - sum:          **0.9449** — still a WASH (no regression; default baseline 0.9365). Byte-for-byte
    identical to v2 (sum's max in-sample rnumel=16384 stays at w16; the w32 step is gated STRICTLY ABOVE
    16384 — see `STREAM_WARPS32_MIN_ELEMS`).
  - long_sum:     **1.099** — the v3 fix: all 5 in-sample shapes now land on **persistent/w32** (was
    looped/w32 in v2). Recovers the 1.04–1.16× the v2 looped branch was LOSING in-sample. default 0.310.
  **3-kernel geomean O = 1.0064** (v2 was 0.9763). No kernel regresses >10% vs champion (none regress at
  all; rms_norm +0.3%). long_sum geomean ROSE from 1.018→1.099 (+8%) exactly as predicted — recovering
  the looped branch's in-sample loss.
  v3 changes vs v2 (the auditor's fix list, all applied):
  - **DELETED** the byte-fence (`PERSIST_MAX_BYTES`) AND the grid-occupancy branch (its premise was a
    confound). The ONLY persistent-vs-looped lever now is the **structural** one: persistent up to the
    backend's per-tile element cap `env.backend.max_tensor_numel` (Triton = 2**20 elems); looped only
    ABOVE it, where a single `tl.arange` over the row literally cannot compile.
  - **Moved num_warps=32 into the PERSISTENT path**, gated on `num_load==1` (the real workload lever) AND
    rnumel > 16384 — NOT a generic huge-rnumel ramp. rms_norm (num_load=2) keeps the conservative v1 ramp
    (it NEVER wants w32 — w32 is catastrophic at large-M/tiny-N: (32768,256) w16=574us, w32=1182us).
  - The looped branch is now a **synthetic/structural generalization tail with NO in-sample coverage** —
    disclosed (see "looped tail disclosure" below). No silent caps.

- **v4 `triton_reduction_tile` (WORKER-PROPOSED champion candidate, 2026-05-29) — the SURGICAL auditor fix.**
  v3 was gate-split: referee ACCEPT, but the **adversarial auditor FAILED the `num_load==1` condition** on
  the w32 ramp. v4 is the one-line resolving fix: **delete the `num_load` condition**, gate the w32 step on
  **`rnumel > 16384` ALONE**. Everything else is byte-identical to v3 (persistent workhorse @ structural
  cap; warps ramp <=1024→4 / <=4096→8 / then 16 / then 32 above 16384; structural looped tail above
  `max_tensor_numel`; all v3 deletions). `num_load`/`num_store` stay in `ReductionFact` as DATA — just not
  gated on. Before/after:
  ```
  - if fact.num_load <= 1 and rnumel > cls.STREAM_WARPS32_MIN_ELEMS:   # v3
  + if rnumel > cls.STREAM_WARPS32_MIN_ELEMS:                          # v4
  ```
  - **In-sample byte-identical → O unchanged.** `AUDITOR_gate_inert_proof.py`: **0/27 mismatches** between
    v4 and the rnumel-only gate (= v3 in-sample), so O_3kernel stays ~1.005 and every per-kernel G_seed is
    UNCHANGED (rms_norm 0.9815, sum 0.9449, long_sum 1.099). Correctness PASS (seed spot-checks:
    rms_norm(2048,16384)=w16 persistent maxabs 1.9e-6; sum(2048,16384)=w16 persistent maxabs 9.9e-5;
    long_sum(8,131072)=w32 persistent maxabs 1.3e-4).
  - **OOS recovery (the auditor's harm, now fixed).** Live v4 emits w32 for ALL held-out large-rnumel
    multi-load shapes; w32 is measured FASTER (recovers 30-40%):
    - rms_norm (num_load=2): w32/w16 (1,131072)=**0.734**, (16,131072)=**0.617**, (16,262144)=**0.616**.
    - layer_norm (num_load=3): w32/w16 (1,131072)=**0.760**, (16,131072)=**0.641**, (16,262144)=**0.540**.
  - **WHY num_load-agnostic (matched-pair A/B physics).** `AUDITOR_numload_warps_ab.py` (num_load=1 vs
    num_load=2, IDENTICAL structure) shows the w32 benefit is driven by **rnumel, NOT num_load** —
    num_load=2 ALSO wants w32 at large rnumel (rnumel=131072: w32/w16=**0.57** for num_load=2). The v3
    num_load fence was a curriculum-split fence dressed as physics: inert in-sample (the condition never
    fires — no in-sample num_load>=2 kernel has rnumel>16384), false on the matched pair, harmful OOS.
  - The tiny-rnumel w32 catastrophe ((32768,256): w16=570us→w32=1174us) is at **rnumel=256, already
    excluded by `>16384`** — an rnumel guard, NOT a num_load one. This is the only thing the num_load fence
    ever "protected against," and the rnumel breakpoint already covers it.
  - Evidence: `_lab/harness/AUDITOR_gate_inert_proof.py`, `AUDITOR_rmsnorm_largeN_warps.py`,
    `AUDITOR_numload_warps_ab.py`, `AUDITOR_v4_oos_recovery.py` (live-seed + layer_norm A/B).
  - **v4 ACCEPTED champion (commit c2845bdd).** WIDENED to **layer_norm_fwd 2026-05-29 with NO heuristic
    change** (byte-identical source; 27/27 existing-kernel seeds unchanged). Active set now 4 kernels.
    Per-kernel G (v4): rms_norm 0.980, sum 0.937, long_sum 1.10, layer_norm **0.989** (+11.7% vs default
    0.886). O_4kernel = **0.9997** (O_3kernel was 1.003; adding a high-but-sub-1 kernel mechanically nudges
    the geomean — same effect as adding sum; the honest claim is per-kernel: no kernel regresses, layer_norm
    is a clean +11.7% win). No per-kernel referee-confirmed G regresses >10% (none regress at all).

## Objective
- Product A: maximize `O = geomean_k G_k`, `G_k = geomean over kernel k's in-sample shapes of
  (tc_default_latency / seed_latency)`. Accept iff O improves AND gates pass (correctness; seed used;
  no active kernel's referee-confirmed G_k regresses >10% vs champion).
- Product B (every 5 iters): seeded vs unseeded quick-autotune convergence curve.

## Persistent-seed round-trip FIX (2026-05-29) — RESOLVES the Product-B injection trap
The Product-B injection TRAP (below) is now FIXED. One-line change in
`ReductionLoopSpec._encode_flat_value` (`helion/autotuner/config_spec.py` ~L1799):
```
- if value is None: return self._flat_fragment(base).default()   # capped at 4096 -> looped for rnumel>4096
+ if value is None: return self._flat_fragment(base).high        # == next_power_of_2(size_hint) -> decodes to None for ALL size_hints
```
WHY: `_flat_config` decodes a flat int back to `None` (persistent) only when `value >= size_hint`. The
fragment **default** is capped at `max_reduction_loop` (4096) by `_flat_fragment`, so for rnumel>4096 it is
`< size_hint` and decodes to a LOOPED chunk — silently degrading the injected persistent seed. The fragment
**`.high`** is `next_power_of_2(size_hint) >= size_hint` (and a power of two, so in-bounds for the
PowerOfTwoFragment), the ONE flat value that round-trips to `None` for every size_hint.

- **Scope:** changed ONLY `_encode_flat_value`. `_flat_config`/`_flat_fragment` untouched. The autotuner
  DEFAULT path is unaffected (it calls `fragment.default()`, still capped → looped for rnumel>4096, which is
  the correct default). CuTe/cache/search-diversity unaffected (this method only changes how an EXPLICIT
  `None` Config value is encoded for flatten).
- **Round-trip proof** (`_lab/harness/roundtrip_probe.py`, real bound kernels): `unflatten(flatten(
  reduction_loops=[None]))` →
  | size_hint | 8192 | 65536 | 262144 | 256 (guard) | 4096 (guard) |
  |---|---|---|---|---|---|
  | PRE-FIX  | [4096] | [4096] | [4096] | [None] | [None] |
  | POST-FIX | [None] | [None] | [None] | [None] | [None] |
- **New repo test:** `test/test_best_available.py::test_flatten_persistent_reduction_loop_roundtrip_large_rnumel`
  — asserts persistent round-trips for rnumel ∈ {8192,65536,262144} + the ≤4096 guard. FAILS pre-fix
  ([4096]!=[None]), PASSES post-fix. No breakage: test_best_available+test_config_api 87 passed;
  test_autotuner 107 passed/7 skip/11 subtests; test_reductions 24 passed.
- **Product A unaffected:** bare-seed path (`configs=[seed]`) does NOT flatten — byte-identical seeds
  (rms_norm(2048,16384)=[None]/w16/bs[1]; long_sum(8,131072)=[None]/w32/bs[1]), codegen still PERSISTENT
  (no `for roffset`). So G is byte-identical.
- **Gen0 seed now persistent (proven in-pipeline):** every seeded Product-B run's `.driver.log` SEED-FLAT-
  ENCODE line now reads `[None]->[None] PRESERVED` (was DEGRADED(persistent->looped) pre-fix). 9/9 seeded
  runs PRESERVED.

## Product B — COMPREHENSIVE across the full 8-kernel curriculum (2026-05-29, post round-trip-fix) — HEADLINE
The time-first deliverable, run over the WHOLE active set (the 3-shape POSTFIX run below is now subsumed).
8 active kernels, 1 representative in-sample shape each where the Product-A seed gives signal + latency is
above the do_bench noise floor (long_sum M lifted 8→256 per the brief). Protocol = identical to POSTFIX:
quick LFBOTreeSearch (init_pop 30, copies 2, max_gen 5), HELION_FORCE_AUTOTUNE=1, cold cache per run,
seeds {0,1,2}, NOT SKIP_CACHE; seeded = registered `triton_reduction_tile`, unseeded =
`HELION_DISABLE_AUTOTUNER_HEURISTICS=1`. GPU2 (Group A: rms_norm×2, layer_norm, long_sum, sum) + GPU3
(Group B: cross_entropy, softmax_two_pass, kl_div, jsd) in parallel, one autotune run per GPU. 54 runs,
all OK (no FAILED). Logs: `logs/productB_full/` (54 CSVs + .driver.log + analysis_t95/t98.txt +
summary.json). Driver extended to all 8 kernels: `productB_driver.py`; runner `productB_full_run.sh`.

- **Seed-injection PROVEN per kernel:** 27/27 seeded runs SEED-FLAT-ENCODE = PRESERVED (0 DEGRADED) — the
  T1 persistent `reduction_loops=[None]`, T1 looped `[16384]` (cross_entropy), and T2 `block_sizes` levers
  all round-trip intact post-fix. Independent gen0-CSV cross-check: 27/27 seeded runs have the expected
  seed config present in gen0 (lever-matched). 27/27 unseeded: n_seeds=0, heuristics=[].

- **gen0 seed-quality advantage (uns gen0 / seed gen0, median of 3) — GEOMEAN = 1.614x:**
  cross_entropy 1.88x · jsd 1.50x · kl_div 2.80x · layer_norm 1.45x · long_sum 1.11x · rms_norm(2048,16384)
  1.41x · rms_norm(8192,8192) 1.32x · softmax_two_pass 3.13x · sum 1.01x (wash). The seed lands a markedly
  better gen0 config on the heavy/wide-row kernels.

- **Slice 1 — same-budget (seeded-advantage = unseeded/seeded), median of 3 · [gen1 / gen2 / gen5-guardrail]:**
  | shape | gen1 | gen2 | gen5 (guardrail) |
  |---|---|---|---|
  | cross_entropy(8192,65536) | 1.126 | 1.000 | 1.000 |
  | jsd(8192,65536) | 1.149 | 1.093 | 1.014 |
  | kl_div(4096,65536) | 1.407 | 1.078 | 1.002 |
  | layer_norm(4096,15872) | 1.005 | 1.002 | 1.004 |
  | long_sum(256,131072) | 1.010 | 1.000 | 1.009 |
  | rms_norm(2048,16384) | 1.118 | 1.031 | 0.997 |
  | rms_norm(8192,8192) | 1.004 | 1.002 | 0.998 |
  | softmax_two_pass(4096,16384) | **2.011** | 1.505 | **1.454** |
  | sum(2048,16384) | 1.076 | 1.027 | 1.013 |
  Slice-1 gen1 GEOMEAN = 1.182x (median 1.118x). **Guardrail PASSES all 9** (gen5 ≥ 1−eps, eps=0.03; worst
  rms_norm 0.997/0.998 is within fp32 do_bench ~2% noise — both modes converge to the same optimum at full
  budget). softmax_two_pass uniquely keeps a 1.45x edge even at full budget: the unseeded LFBO rarely finds
  the persistent config within 5 generations.

- **Slice 2 — time-to-95%-of-unseeded-full-budget (speedup = uns_t / seed_t), median[min,max] of 3 — HEADLINE:**
  | shape | seeded s | unseeded s | speedup@95% | (@98%) |
  |---|---|---|---|---|
  | cross_entropy(8192,65536) | 8.12[7.99,8.35] | 48.71[44.48,62.40] | **5.999x** | 0.771x |
  | jsd(8192,65536) | 11.51[11.41,11.79] | 32.88[30.82,65.32] | **2.857x** | 2.857x |
  | kl_div(4096,65536) | 7.83[7.68,7.93] | 12.70[12.45,14.15] | 1.622x | 1.631x |
  | layer_norm(4096,15872) | 4.96[4.91,5.04] | 8.84[7.67,46.32] | 1.782x | 1.782x |
  | long_sum(256,131072) | 31.16[7.74,35.61] | 9.56[8.74,10.37] | **0.307x** (artifact) | 0.353x |
  | rms_norm(2048,16384) | 4.60[4.60,4.63] | 9.18[7.84,30.25] | 1.996x | 5.074x |
  | rms_norm(8192,8192) | 4.70[4.69,4.79] | 9.13[8.52,11.32] | 1.943x | 1.943x |
  | softmax_two_pass(4096,16384) | 4.70[4.66,4.80] | 11.36[9.33,21.84] | 2.417x | 2.657x |
  | sum(2048,16384) | 5.87[5.61,6.12] | 8.03[7.00,9.05] | 1.368x | 1.368x |

- **AGGREGATE CURRICULUM HEADLINE:** median Slice-2 time-to-95% speedup = **1.943x across all 9 shapes**
  (1.970x median / 2.241x geomean excluding the long_sum noise-floor artifact). **Seeding shifts the
  convergence curve UP-AND-LEFT across the whole curriculum: the seeded search reaches 95% of the
  full-budget optimum in ~HALF the wall-clock — equivalently the autotune budget could be ~halved on most
  shapes.** Largest wins on the heavy/wide-row T2 + cross_entropy kernels (cross_entropy 6.0x,
  jsd 2.9x, softmax_two_pass 2.4x) where the seed injects a non-obvious persistent/capped-loop config the
  unseeded LFBO is slow to discover.

- **Honest non-wins / caveats:**
  - **long_sum(256,131072) Slice-2 = 0.307x (REGRESSED on the metric, NOT a perf loss).** Noise-floor +
    longer-seeded-search artifact: 95% target = 74.3us sits in the ~70-85us small-M (M=256) noise band;
    seeded gen0 = 76.4us vs unseeded 84.7us (the perf lever IS +11% better), but seeded explores far more
    candidates (n_ok 78-107 vs 24-49 → total wall-clock 39-68s vs 13-36s), so 2/3 unseeded runs trip 74us
    via a random LFBO dip sooner. Lifting M 8→256 did NOT clear the noise floor (was 0.479x at M=8). Honest
    long_sum claim = the gen0/Slice-1 PERF win, not Slice-2.
  - **sum(2048,16384):** Product-A wash (full-budget perf ties). Still a modest 1.37x time-to-target from a
    better gen0 — reported honestly as a small/no-win shape as the brief predicted.
  - **98% target is fragile:** cross_entropy flips to 0.771x at 98% because the seeded full search runs
    longer (95s vs 73.5s total) and both converge to ~0.949ms, so the last 1% needs the slower full sweep.
    The robust, defensible headline is the **95% target**. (The "seeded full search is LONGER" caveat from
    prior runs persists curriculum-wide — the Product-B value is early-budget perf + time-to-target, NOT a
    cheaper full search.)

## Product B — POST-FIX re-run (2026-05-29) — the ENLARGED win  [SUBSUMED by the comprehensive run above]
Same protocol/budget as the pre-fix run (quick effort, force-autotune, cold cache, seeds {0,1,2}, no
SKIP_CACHE), 3 shapes where the persistent lever matters. Logs: `logs/productB_postfix/` (18 CSVs +
.driver.log + analysis_t95/t98.txt). Driver/runner: `productB_driver.py` + `productB_postfix_run.sh`.

- **The seed-quality lever GREW on all 3 shapes (gen0 seeded-advantage = unseeded/seeded, median of 3):**
  | shape | pre-fix gen0 | post-fix gen0 |
  |---|---|---|
  | rms_norm(2048,16384) | 1.241x | **1.408x** |
  | rms_norm(8192,8192)  | 1.306x | **1.325x** |
  | long_sum(8,131072)   | 3.829x | **4.023x** |
  The seed now reaches gen0 carrying the PERSISTENT lever (not just num_warps) → a better gen0 config.

- **Slice 1 — same-budget (seeded-advantage=unseeded/seeded), median of 3:**
  | shape | gen1 | gen2 | gen5 (guardrail) |
  |---|---|---|---|
  | rms_norm(2048,16384) | 1.123 | 1.035 | 1.020 |
  | rms_norm(8192,8192)  | 1.005 | 1.002 | 1.000 |
  | long_sum(8,131072)   | 1.452 | 1.202 | 1.008 |
  Guardrail PASSES all 3 (seeded ≥ unseeded at full budget).

- **Slice 2 — time-to-95% (speedup=uns_t/seed_t), median[min,max]:**
  | shape | post-fix seeded s | post-fix unseeded s | post-fix speedup | pre-fix speedup |
  |---|---|---|---|---|
  | rms_norm(2048,16384) | 4.53[4.50,4.68] | 8.55[7.54,30.15] | **1.89x** (@98%: 2.00x) | 1.78x |
  | rms_norm(8192,8192)  | 4.74[4.72,4.83] | 9.15[8.53,11.38] | **1.93x** (@98%: 1.93x) | 1.70x |
  | long_sum(8,131072)   | 29.19[12.56,35.55] | 13.98[12.24,33.59] | 0.479x | 1.04x |

- **rms_norm wins GREW as predicted** (1.78→1.89x, 1.70→1.93x): the persistent lever now survives injection.
- **long_sum Slice-2 dropped to 0.479x — an HONEST noise-floor + total-wallclock artifact, NOT a real loss.**
  Its perf lever GREW (gen0 4.02x; Slice-1 gen1 1.45x). But the 95% target = 0.0081ms (8.1us) sits in the
  ~7-10us tiny-M noise floor (M=8, 5 rows), where BOTH modes hit target by random LFBO dips and the
  wallclock-to-target is dominated by total search time. Post-fix the persistent seed makes LFBO compile
  MANY more neighbors (seeded total wall-clock 17.6-47s vs unseeded 14.6-40s — the same documented "seeded
  full search is LONGER" caveat), so a lucky unseeded run can trip the razor-thin 8.1us threshold sooner.
  This was already a near-tie pre-fix (1.04x); the metric is unreliable at this absolute latency. The honest
  long_sum claim is the gen0/Slice-1 PERF win (4.0x gen0, 1.45x gen1), which the fix INCREASED.

## Product B — seed the autotuner (RUN 2026-05-29, v4 seed) — TIME WIN CONFIRMED + an injection TRAP (PRE-FIX, superseded above)
Ran quick-autotune SEEDED vs UNSEEDED, N=3 random seeds {0,1,2} each, cold cache per run, full
max_generations=5, on rms_norm (2048,16384) & (8192,8192), long_sum (8,131072), sum (2048,16384).
GPU2 (rms_norm) + GPU3 (long_sum,sum) in parallel, one autotune run per GPU. Default LFBOTreeSearch,
quick profile. Harness: `_lab/harness/productB_{driver.py,run.sh,analyze.py}`; raw + summary in
`logs/productB/` (results.json, analysis_t95/t98.txt, 24 CSVs, per-run .driver.log).

- **Method + seed-injection VERIFICATION (per run):** SEEDED = default (compiler_seed_configs n=1,
  autotuner_heuristics=['triton_reduction_tile']; seed enters gen0). UNSEEDED =
  `HELION_DISABLE_AUTOTUNER_HEURISTICS=1` (n=0, heuristics=[]; gen0=default only). Both proven in each
  `.driver.log` and cross-checked against gen0 of the CSV (seeded gen0 = {default w4, seed w16/w32};
  unseeded gen0 = {default w4}). Only difference = the one compiler seed config in gen0.

- **TRAP (headline finding): the persistent seed is DEGRADED on injection.** On ALL 4 shapes (rnumel>4096)
  the seed's `reduction_loops=[None]` (PERSISTENT — the dominant Product-A lever) is silently flat-encoded
  to a LOOPED chunk of 4096 when the autotuner injects the compiler seed. `num_warps`+`block_sizes`
  survive; the persistent choice does NOT. Root cause: `ReductionLoopSpec._encode_flat_value`
  (config_spec.py ~1799) maps `None -> _flat_fragment.default() = min(next_pow2(rnumel), 4096)`; unflatten
  restores `None` only if the flat int `>= size_hint` (false for rnumel>4096). Proof: `[None]` codegen has
  no `for roffset` loop (1 tl.arange), `[4096]` has it (2 tl.arange) — different kernels. The Product-A
  bare-seed path (`configs=[seed]`) is UNAFFECTED (keeps `[None]`); ONLY the autotuner-injection path
  degrades. So the Product-B wins below are a LOWER BOUND — the seed reaches gen0 carrying only its
  num_warps advantage. OPEN LEVER: make `ReductionLoopSpec` round-trip `None` (a sentinel that decodes back
  to persistent) before re-running Product B — should widen the early-budget gap.

- **Convergence curve (best-perf-so-far vs gen, median ms) — shifts UP-and-LEFT:**
  | shape | mode | g0 | g1 | g2 | g5 |
  |---|---|---|---|---|---|
  | rms_norm(2048,16384) | seeded/unseeded | 0.147/0.183 | 0.129/0.144 | 0.129/0.133 | 0.128/0.129 |
  | rms_norm(8192,8192)  | seeded/unseeded | 0.257/0.335 | 0.252/0.253 | 0.252/0.252 | 0.249/0.249 |
  | long_sum(8,131072)   | seeded/unseeded | 0.0101/0.0387 | 0.0100/0.0141 | 0.0081/0.0106 | 0.0077/0.0089 |
  | sum(2048,16384)      | seeded/unseeded | 0.0756/0.0762 | 0.0698/0.0757 | 0.0692/0.0707 | 0.0678/0.0694 |

- **Slice 1 — same-budget perf (seeded-advantage = unseeded/seeded, >1=seeded faster):**
  | shape | gen1 | gen2 | gen5 (guardrail) |
  |---|---|---|---|
  | rms_norm(2048,16384) | 1.116 | 1.034 | 1.009 |
  | rms_norm(8192,8192)  | 1.004 | 1.001 | 1.000 |
  | long_sum(8,131072)   | 1.409 | 1.306 | 1.145 |
  | sum(2048,16384)      | 1.085 | 1.020 | 1.024 |
  Sharpest at small budget. Full-budget guardrail PASSES all 4 (seeded >= unseeded; no regression).

- **Slice 2 — time-to-target (HEADLINE; wall-clock to 95% of unseeded-full-budget, speedup=uns_t/seed_t):**
  | shape | seeded s | unseeded s | speedup | @98% |
  |---|---|---|---|---|
  | rms_norm(2048,16384) | 8.86 | 15.76 | **1.78x** | 2.59x |
  | rms_norm(8192,8192)  | 5.25 | 8.94  | **1.70x** | 1.17x |
  | long_sum(8,131072)   | 19.38| 20.18 | 1.04x (uns 2/3 reached) | 1.17x |
  | sum(2048,16384)      | 9.30 | 14.17 | **1.52x** | 1.52x |

- **HONEST caveat:** the seed makes the FULL 5-gen search take LONGER (seeded total wall-clock 24-62s vs
  unseeded 22-36s — LFBO explores around the extra good config, compiling more neighbors). Product-B value
  is NOT a cheaper full search; it is reaching a good config SOONER (Slice 2) and better early-budget perf
  (Slice 1). The practical lever = with a seed you can SHRINK the budget (stop at gen1-2) and still land
  near the full-budget optimum.

- **Verdict:** seeding shifts the curve up-and-left on all 4 shapes; headline time-to-95% win 1.5-1.8x on
  3/4 shapes; long_sum a time-to-target ~tie but a 3.8x gen0 win (its ~10us latencies are in the noise
  floor). No full-budget regression. All achieved with the persistent lever degraded on injection.
  WORKER-PROPOSED; HELPER_REQUEST queued for results-referee spot-repro of Slice-2 on (2048,16384).

## Active kernels (curriculum)
- Active (SEEDED): **rms_norm_fwd, sum, long_sum, layer_norm_fwd** (T1, Band A) + **cross_entropy** (T1, Band A,
  num_load=3, wide-V) + **softmax_two_pass** (T2 Band A) + **kl_div, jsd** (T2 Band B) as of 2026-05-29 (v6) =
  **8 seeded kernels**. **welford: classified T2 but OUT-OF-SCOPE** (multi-tiled-pass; heuristic DECLINES → not
  seeded, falls back to default). Forward curriculum COMPLETE. Defer backward (Band D).

## v5 — T2 SUPPORT (user-tiled / manually-looped reductions): softmax_two_pass + kl_div + jsd (2026-05-29)
**The heuristic now seeds BOTH tracks; T1 is byte-identical (no-regression PROVEN).**

### T2 plumbing (3 files; T1-transparent)
- **device_ir.py `register_user_tiled_reductions()`** (new; called right AFTER `register_rollable_reductions()`,
  GUARDED by `if not config_spec.reduction_facts:` so T1/T2 are mutually exclusive and the fact count stays 1).
  Finds the T2 reduction axis via the ReductionLowering predicate FILTERED against `grid_block_ids`
  (load-bearing for jsd: its dead beta==0/1 `amax(dim=0)` over the M tile makes raw red_block_ids={0,1};
  removing the grid axis (1) leaves the real V reduction (0)). Builds ONE ReductionFact. **Declines (no fact)
  if** the inner reduction axis isn't unique (len!=1), isn't a block_sizes entry, or has a **dynamic/None size**
  (jagged_softmax: `size=None` AutoSize — guard added after that test REGRESSED; the persistent-vs-looped lever
  is undefined without a static extent). Counting (loads/stores/reductions/dtype/2D-accumulators) is factored
  into a SHARED `_count_reduction_workload()` used by BOTH the T1 and T2 fact builders → identical digestion.
- **triton.py `_triton_reduction_eligible`**: dropped the T1-only shape signature (`len(block_sizes)==1 and
  len(reduction_loops)==1`); now gates on `len(reduction_facts)==1 and not matmul_facts` ALONE. Generalizes on
  the WORKLOAD (one inner reduction) not the rolling mechanism; admits T2 (2 block_sizes / 0 reduction_loops)
  while still excluding GEMM + multi-axis manual reductions (which leave reduction_facts at 0).
- **triton.py `get_seed_config` T1-vs-T2 routing**: the persistent-vs-looped lever (extent = next_pow2(size_hint)
  capped at max_tensor_numel; num_warps = the rnumel ramp) is SHARED. `is_t1 = fact.block_id in
  reduction_loops.valid_block_ids()`. T1 → `reduction_loops=[None|LOOPED_CHUNK]`, `block_sizes=[m_floor]` (as
  before). T2 → `block_sizes[red_idx]=R_BLOCK`, every other block_size at its floor (keeps M_BLOCK=1 — the
  Band-B u0*u1<=2^20 numel constraint), NO reduction_loops. Persistent T2 == R_BLOCK>=next_pow2(N) so the user
  `for offset in tl.range(0, N, R_BLOCK)` runs ONCE (verified in codegen: `_BLOCK_SIZE_<red> = tl.constexpr(R)`
  with R>=N → single masked pass).

### T1 NO-REGRESSION (the sacred gate) — BYTE-IDENTICAL
`layer_norm_no_regression_proof.py` = **27/27 OK** (rms_norm 13 + sum 9 + long_sum 5) AND a separate
layer_norm check = **11/11 OK** → 38/38 v4 champion seeds UNCHANGED. The gate broadening + guarded T2 populate +
the new `num_tiled_accumulators` fact field (default 0) are all T1-transparent. `test_reductions.py` 24 passed.

### softmax_two_pass (T2 Band A) — clean WIN, NO branch needed
- num_tiled_accumulators=0 (carries only `[M_BLOCK]` row state mi/di) → stays PERSISTENT to the structural cap
  exactly like the T1 kernels. Seed = block_sizes=[m_floor, next_pow2(N)] + rnumel warps ramp.
- **G_softmax = 0.967** vs un-seeded default **0.440** (+120%; default goes looped chunk-4096 and loses 2-3x at
  wide rows). do_bench median-of-7, fp32, GPU2. Correctness PASS all shapes (maxabs ~1e-8 vs F.softmax fp32).
- **DECISIVE A/B vs persistent/w32: seed/p32 = 1.50x geomean (seed FASTER).** The seed's rnumel warps ramp (w4
  at N<=1024) AVOIDS the tiny-N/large-warps catastrophe that sinks a fixed-w32 baseline (seed/p32 = 3.1x/2.4x/
  1.7x/3.0x at (4096,256)/(4096,512)/(4096,1024)/(32768,256); ties ~1.0 at wide rows). Same coupling catastrophe
  documented for T1 (32768,256). So the rnumel ramp is the right lever — NOT a fixed high warp count.

### kl_div + jsd (T2 Band B) — the Band-B lever ADDED (validated by matched A/B vs persistent)
- num_tiled_accumulators=2 for BOTH (kl_div: kl_loss+loss_sum; jsd: intermediate_loss+intermediate_dX) — each a
  `[M_BLOCK, R_BLOCK]` buffer CARRIED across the inner loop, so the persistent live-state footprint SCALES with
  R_BLOCK. A full-N persistent R_BLOCK over-allocates and SPILLS: matched A/B (M_BLOCK floor, persistent full-N
  vs small looped chunk) shows the persistent seed is **1.2-9.7x SLOWER** at the widest rows
  (jsd (8192,65536) persistent=19.8ms vs loop4096=2.0ms; (8192,131072) persistent unmeasurably slow).
- **Band-B branch (the ONLY new branch):** when `fact.num_tiled_accumulators >= 1`, cap R_BLOCK at
  `BANDB_R_BLOCK_BYTES // itemsize` (16 KiB → 4096 fp32 elems; byte-based so it generalizes across dtypes).
  Gated on the WORKLOAD property (live-state footprint), NOT kernel identity. Scalar-/row-accumulator T2
  (softmax, num_tiled=0) and ALL T1 are UNAFFECTED (stay persistent to the structural cap).
- **VALIDATION (the methodology lesson — A/B vs the best SIMPLE alternative, not the default strawman):**
  the R=4096 cap is best-or-tied at EVERY in-sample Band-B row. NARROW rows (`t2_bandb_narrow_check.py`):
  persist(full-N)/cap = 0.996-1.017 → NO regression. WIDE rows (`t2_bandb_chunk_sweep.py`): R=4096 recovers the
  spill (R>=32768 catastrophic on jsd). The chunk R=4096 is optimal for BOTH kl_div and jsd.
- **G_kl_div = 1.029** vs default 0.392 (the seed BEATS tc on 4/6 shapes; +163% vs default). seed s/AB = 0.99-1.41
  (ties/beats the best simple looped alt). **G_jsd = 0.996** vs default 0.665 (+50%; essentially ties tc all 6
  shapes; was a 0.10/unmeasurable disaster WITHOUT the branch). Correctness PASS: kl_div rel-err <=1.2e-7 vs
  torch.nn.KLDivLoss(batchmean); jsd loss rel-err 0.0 vs TorchJSDBaseline, dX matches default ~1e-12. The benign
  TensorOperationInWrapper warning on jsd's host-side `torch.sum(loss)` is documented/unrelated.

### O over the 7-kernel active set
Per-kernel G (v5): rms_norm 0.980, sum 0.937, long_sum 1.099, layer_norm 0.989 (all UNCHANGED, byte-identical) +
softmax_two_pass 0.967, kl_div 1.029, jsd 0.996 (new). **O_7kernel = 0.9985** (O_4kernel was 0.9995 — adding
high-but-sub-1 kernels mechanically nudges the geomean down, same effect as adding sum/layer_norm). Honest claim
= PER-KERNEL: no kernel regresses (T1 byte-identical), 3 new clean wins (softmax +120%, kl_div +163%, jsd +50%
all vs un-seeded default). Harness: `_lab/harness/{t2_classify,t2_seed_probe,t2_codegen_probe,measure_g_softmax,
measure_g_lossk,measure_g_jsd,t2_bandb_chunk_sweep,t2_bandb_narrow_check,t2_lossk_correct}.py`.

## v6 — cross_entropy widening (general multi-load persist cap) + welford OUT-OF-SCOPE (2026-05-29)

### cross_entropy — T1, FIRES, but needed a GENERAL heuristic change (the multi-load persist cap)
- **Classification: T1** (rollable rdim). `len(block_sizes)==1` (the M/row tile), `len(reduction_loops)==1`,
  `len(reduction_facts)==1`, no matmul. The `logits[tile_n, :]` whole-row `amax`+`sum` over V is a rollable
  reduction (block_id=1 = the V axis); 1 M-block (block_id=0). RF: **num_load=3, num_store=1,
  num_reduction_ops=2** (amax + exp-sum), num_tiled_accumulators=0. Plus a label gather (`hl.load`). Heuristic
  FIRES 1 seed (`triton_reduction_tile`). NOTE: at the degenerate (4096,4096) shape (N==V) the RF dtype scan
  grabs the int64 labels (itemsize=8) — INERT (num_tiled_accumulators=0 so the Band-B cap never reads it; the
  seed is persistent/rnumel-warps regardless). At V!=N it correctly reads fp32/4.
- **Seed USED + CORRECT:** bare-seed codegen is persistent (no `for roffset`) below the cap / looped above;
  num_warps matches the rnumel ramp. Correct vs `F.cross_entropy` fp32: maxabs 0–1.9e-6, rel-err ~1e-7
  (bit-exact at narrower V). Benign `TensorOperationInWrapper` on the host-side `losses.mean()` (documented).
- **THE FINDING (overturns a champion assumption): the "persistent to the 2^20 structural cap" policy was
  validated ONLY on num_load=1 (sum, v3_crossover_sweep). It is WRONG for num_load>=2 at large rnumel.** A
  matched-WARPS persistent-vs-looped A/B *across num_load* (`persist_crossover_by_numload.py` +
  `ce_crossover_tight.py`, H100/fp32):
  | rnumel(KiB) | sum nl=1 P/L | rms_norm nl=2 P/L | cross_entropy nl=3 P/L |
  |---|---|---|---|
  | 64 (16K)  | 1.01 | 0.99 | 0.98 |
  | 128 (32K) | 1.00 | 0.96 | 0.98 |
  | 256 (64K) | 1.00 | 0.99 | 1.04 (~tie) |
  | 288 (72K) | —    | **2.75** | 1.09 |
  | 384 (96K) | 1.01 | **3.00** | **1.61** |
  | 512(128K) | 1.00 | **2.91** | **3.97** |
  num_load=1 (sum) ties persistent==looped at EVERY rnumel up to 1 MiB at BOTH M=8 (starved) and M=1024
  (occupied) → the structural-cap policy stays correct for it (NO regression). num_load>=2 persistent wins/ties
  to ~256 KiB then LOSES decisively (a multi-load reduction re-streams the wide row per load pass; a persistent
  kernel holding the whole row resident spills, looped streams). Crossover SHARP at 256->288 KiB, holds across
  M. This is the SAME finding for rms_norm/layer_norm/cross_entropy — NOT cross_entropy-specific. It has NO
  in-sample coverage in the prior 7 (rms_norm/layer_norm max rnumel 16384/15872; sum/long_sum are num_load=1).
  Distinct from the v3/v4 `num_load` debate: that was about the **num_warps=32 ramp** (rnumel-driven for all
  num_load — correct). THIS is the **persistent-vs-looped** lever, which IS num_load-dependent.
- **THE BRANCH (the only new one): `MULTILOAD_PERSIST_MAX_BYTES = 131072 (128 KiB)`.** For `num_load >= 2` AND
  `rnumel*itemsize > 128 KiB` → go looped (T1: `reduction_loops=[LOOPED_CHUNK=16384]`; T2: cap R_BLOCK). Cap at
  128 KiB (not 256): at 65536 (256 KiB) persistent is a tie BUT the w32 ramp is suboptimal there
  (CE(8192,65536) persist/w32 1195us vs looped ~990us = tc) so looped is the better+robust choice. Keyed on the
  WORKLOAD property `num_load` (already a ReductionFact field) + a byte threshold (generalizes via itemsize),
  NOT kernel identity. **In-sample INERT for all 7** (every num_load>=2 kernel <=64 KiB; ce_inert_proof_7.py =
  27/27 byte-identical). num_load=1 (sum/long_sum: long_sum 262144/w32 persistent UNCHANGED) and Band-B
  (kl/jsd, tighter 16 KiB cap dominates) unaffected.
- **Per-shape G_cross_entropy (v6 seed, do_bench median-of-7, fp32, GPU3; two runs agree 0.912/0.915):**
  | shape | codegen | warps | G_seed | G_default | s/p32 |
  |---|---|---|---|---|---|
  | (4096,4096) | persist | 8 | 0.92 | 0.93 | 1.13 |
  | (4096,16384) | persist | 16 | 1.00 | 0.73 | 1.01 |
  | (8192,32768) | persist | 32 | 1.05 | 0.54 | 1.00 |
  | (16384,32768) | persist | 32 | 1.05 | 0.53 | 1.00 |
  | (8192,65536) | **looped** | 32 | 1.00 | 0.53 | 1.21 |
  | (16384,65536) | **looped** | 32 | 0.99 | 0.52 | 1.21 |
  | (8192,131072) | **looped** | 32 | 0.54 | 0.52 | 2.25 |
  | **GEOMEAN** | | | **0.915** | **0.600** | **1.21** |
  G_cross_entropy = **0.915** (+52% vs default). The ONLY sub-0.9 shape is (8192,131072) G=0.54: a kernel-source
  limit, NOT config headroom — the best CORRECT Helion looped config (chunk 32768/w32 = 3485us; my seed
  16384/w32 = 3575us, within 2.5%) still loses to tc's 1928us because Helion CE re-reads the row for the
  exp-sum pass while tc fuses an online softmax (num_stages doesn't help; sweep confirmed). Disclosed gap,
  analogous to the long_sum looped-tail-vs-tc story. seed/p32=1.21 → the seed BEATS fixed-persistent/w32
  (correctly loops the wide rows).

### welford — T2, FIRES, but the seed is BROKEN; classified OUT-OF-SCOPE → heuristic now DECLINES
- **Classification: T2 (user-tiled), num_tiled_accumulators=0 (Band-A-like), but STRUCTURALLY DISTINCT.** 3
  block_sizes (block 0 = M grid, autotuner_min=16 at M=262144; block 1 = Welford-combine `tile_n`; block 2 =
  normalize `tile_n` — BOTH over the SAME N), 0 reduction_loops, 1 reduction_fact. The detector finds block 1
  as the reduction axis (it has the ReductionLowering); block 2 is NOT a reduction → treated as a row/grid axis
  and FLOORED to width 1. RF: num_load=4, num_store=1, num_reduction_ops=2, num_tiled_accumulators=0 (acc_cnt/
  acc_mean/acc_m2 are `[M_BLOCK]` row scalars, not 2D).
- **The seed FIRES but is CATASTROPHIC + INCORRECT.** Persistent seed `block_sizes=[16, next_pow2(N), 1]`:
  - **G_seed = 0.051** (10-20x slower than tc AND than the un-seeded default G=0.526) — because the SECOND
    `tile_n` loop (normalize pass) is floored to width 1 → N iterations of width 1. (262144,1024): seed
    11769us vs tc 979us vs default 1821us. The seed actively HARMS welford.
  - **NUMERICALLY WRONG for non-pow2 N** (in-sample 1536): R1=next_pow2(1536)=2048 with masking → err=0.696
    (FAILS allclose). ROOT CAUSE = a kernel bug: the Welford combine uses `Tn = chunk.size(-1)` for the
    per-chunk count; a masked persistent tile makes `Tn`=tile-width (2048) not valid-width (1536), so
    `sum_x/Tn` divides by the wrong count. A CORRECT persistent welford needs R1 to DIVIDE N (looped divisor
    chunk, e.g. 512 for 1536) — factorization-dependent, NOT next_pow2. (`welford_probe.py`,
    `welford_ceiling.py`.)
- **What welford WOULD need (the HELPER_REQUEST):** (1) the seed must control BOTH N-axis tiles (widen block 2,
  the normalize pass, to persistent — the ceiling sweep shows M=1, R1=R2=N, w8-16 ties tc G~0.99 at pow2 N);
  (2) a non-pow2 correctness guard that picks a looped DIVISOR chunk for the Welford-combine tile (R1) instead
  of next_pow2 (masked R1 breaks the count). Both are welford-shaped and validated on ONE kernel only (no
  held-out coverage) → forcing them now would be a kernel-identity hack. Per the brief's out-of-scope track:
  DECLINE per-axis seeding, document why.
- **THE DECLINE (general, structural):** `register_user_tiled_reductions` now declines when there is **>1
  non-grid user tile** (`len(non_grid_tiles) != 1`). The working T2 kernels (softmax/kl_div/jsd) have EXACTLY 1
  non-grid tile (the reduction axis itself); welford has 2 (Welford pass + normalize pass). The single-axis
  seed is undefined for a multi-pass / multi-tiled-axis kernel. Keys on WORKLOAD structure (# of non-grid user
  tiles), NOT kernel identity. welford now emits 0 seeds → falls back to the CORRECT un-seeded default (G=0.53)
  instead of the broken G=0.05 seed. Same decline pattern as jagged_softmax (dynamic-size) / longsum_manual.
- **No-regression:** softmax/kl_div/jsd still fire with byte-identical seeds (the decline guard only triggers on
  >1 non-grid tile, which they don't have). test_examples (welford/jagged/softmax/ce/kl/jsd/ln/rms) 19 passed;
  test_reductions 24 passed.

### O over the active set (v6: 8 seeded kernels; welford declined/out-of-scope)
Per-kernel G (v6): rms_norm 0.980, sum 0.937, long_sum 1.099, layer_norm 0.989, softmax_two_pass 0.967,
kl_div 1.026, jsd 0.997 (all 7 BYTE-IDENTICAL, UNCHANGED) + **cross_entropy 0.915** (new). **O_8kernel =
0.9874** (O_7kernel UNCHANGED at 0.9982; cross_entropy at 0.915 mechanically nudges the geomean down — same
effect as adding sum/layer_norm; the (8192,131072) Helion-source gap holds it sub-1). Honest PER-KERNEL: NO
kernel regresses (7 byte-identical; cross_entropy +52% vs un-seeded default). welford is NOT seeded (declined)
so it is not in the active G set — falls back to its correct default. Harness:
`_lab/harness/{classify_ce_welford,ce_seed_used,measure_g_ce,ce_persist_vs_loop,persist_crossover_by_numload,
ce_crossover_tight,wide_rnumel_persist_vs_loop,ce_inert_proof_7,welford_probe,welford_ceiling}.py`.

## layer_norm_fwd — WIDENED 2026-05-29 (v4 SUFFICES UNCHANGED; byte-identical heuristic)
The cleanest possible outcome: **adding layer_norm_fwd to the active set required ZERO heuristic change**
(git shows 0 lines changed under `helion/`; the 3 existing kernels emit byte-identical v4 champion seeds —
`layer_norm_no_regression_proof.py` = 27/27 OK). layer_norm benefits from the persistent workhorse + the
rnumel warps ramp DIRECTLY, exactly like rms_norm.
- **G_layer_norm (v4 seed, do_bench median-of-7, fp32, GPU2; two runs agree 0.9888/0.9891) = 0.989** vs
  un-seeded default baseline **0.886** (+11.7%). With bias (tritonbench default). Reference =
  `torch.nn.functional.layer_norm` fp32; correctness PASS all shapes (maxabs ~2-3e-6 « tol).
- Same mechanism as rms_norm: at wide rows (rnumel>=8192) the un-seeded default goes looped chunk-4096 and
  LOSES big (G_default 0.68 @15872, 0.79 @12288, 0.80 @8192-7168); the persistent seed recovers to ~0.99.
- **num_reduction_ops=2 does NOT want a different config.** The oracle field-diff (quick-autotune, fair
  re-bench of the FULL verbatim winner, 6 shapes) is G_seed=1.003 vs G_oracle=1.009 (~0.6%, noise): the
  oracle KEEPS persistent (`reduction_loops=[None]`) at the widest rows and ties the seed on warps/stages.
  Where the oracle picks w32 or a looped chunk it is perf-NEUTRAL vs the seed. So the extra live
  accumulator (mean+var) does NOT shift the warps/stages optimum — no num_reduction_ops-keyed branch is
  warranted. The only real headroom is (4096,1024) small-N (oracle w2 + looped-256, ~4%) — the same
  tiny-N/large-warp regime already noted for rms_norm (32768,256)/sum, an unseeded indexing lever.
- harness: `_lab/harness/{classify_layer_norm.py, measure_g_layer_norm.py, oracle_layer_norm.py,
  layer_norm_no_regression_proof.py}`.

## Per-shape G_layer_norm (v4 seed, do_bench median-of-7, fp32, GPU2; with bias) — two runs agree
| shape | codegen | warps | G_seed | G_default(baseline) |
|---|---|---|---|---|
| (4096,1024) | persist | 4 | 0.99 | 0.99 |
| (4096,2048) | persist | 8 | 0.99 | 0.99 |
| (4096,4096) | persist | 8 | 0.98 | 0.98 |
| (4096,8192) | persist | 16 | 0.99 | 0.80 |
| (4096,12288) | persist | 16 | 0.99 | 0.79 |
| (4096,15872) | persist | 16 | 0.99 | 0.68 |
| (2048,3584) | persist | 8 | 0.99 | 1.00 |
| (2048,8192) | persist | 16 | 0.99 | 0.85 |
| (8192,4096) | persist | 8 | 1.00 | 0.99 |
| (8192,5120) | persist | 16 | 0.99 | 0.92 |
| (8192,7168) | persist | 16 | 0.99 | 0.81 |
| **GEOMEAN** | | | **0.989** | **0.886** |
NOTE: all in-sample rnumel <= 15872 < the w32 breakpoint (16384), so every shape gets the ramp's w4/w8/w16
(no shape crosses w32) AND every shape is well under the 2^20 structural cap (persistent). The w32 step is
inert IN-SAMPLE for layer_norm (would only fire for held-out rnumel>16384, where v4's matched-pair physics
says w32 is correct for num_load>=2 too — see v4 OOS recovery: layer_norm (1,131072) w32/w16=0.760).

## Track classification (T1 rolled / T2 manual / out-of-scope) — per kernel
- **rms_norm_fwd: T1** (rollable rdim; `reduction_loops` has 1 entry; `reduction_facts` has 1 entry).
  Single block_size (M-axis) + single reduction_loop, no matmul_facts. RF: num_load=2, num_store=2.
- **sum (`sum_kernel`): T1** confirmed (classify_kernels.py). 1 block_size, 1 reduction_loop, 1 RF, no
  matmul. RF: **num_load=1**, num_store=1, num_reduction_ops=1 (differs from rms_norm's num_load=2 — sum
  reads x ONCE). Heuristic fires 1 seed. M-block autotuner_min=1 even at 32768 rows (rms_norm got 2 —
  rms_norm's two-pass loads make its grid-min logic trip differently; not a problem).
- **long_sum (`longsum` naive): T1** confirmed — the rollable target. Identical structure to sum
  (`for tile_m: out[tile_m]=x[tile_m,:].sum(-1)`); RF num_load=1/num_store=1/num_reduction_ops=1. The
  shipped `@helion.kernel(config=...)` is irrelevant to us — we bare-seed via `helion.kernel(fn.fn,
  configs=[our_seed])`. `longsum_w_red_loop` is the SAME rollable T1 kernel (just ships a looped config).
- **long_sum (`longsum_manual`): OUT-OF-SCOPE.** Uses an explicit `hl.tile(n)` inner reduction loop →
  2 block_sizes entries, 0 reduction_loops, 0 reduction_facts (manual T2, not rollable). Heuristic
  correctly emits 0 seeds. Not a target.
- **cross_entropy: T1** confirmed (classify_ce_welford.py). 1 block_size (M/row), 1 reduction_loop, 1 RF, no
  matmul → eligibility gate PASSES. The whole-row `amax`+`sum` over V (block_id=1) is rollable; 1 M-block
  (block_id=0). RF: **num_load=3, num_store=1, num_reduction_ops=2** (amax + exp-sum), num_tiled_accumulators=0,
  static_rnumel=V. Plus a label gather (`hl.load`, not counted as a row reduction). Heuristic FIRES 1 seed,
  seed used + correct vs F.cross_entropy. Needed the v6 multi-load persist cap (num_load>=2, >128 KiB → looped).
- **welford: T2 (user-tiled) but OUT-OF-SCOPE** (classify_ce_welford.py). 3 block_sizes (M grid + TWO `tile_n`
  loops over the SAME N: the Welford-combine pass = the detected reduction axis block 1, and the normalize pass
  = block 2, NOT a reduction), 0 reduction_loops, 1 RF (block 1). RF: num_load=4, num_store=1,
  num_reduction_ops=2, num_tiled_accumulators=0. The single-axis seed FIRES but is BROKEN: it floors the second
  tile (normalize, block 2) to width 1 → G=0.05 (10-20x worse than default) AND is numerically wrong for
  non-pow2 N (masked Welford count). The heuristic now DECLINES it (general guard: >1 non-grid tile → no fact),
  so welford falls back to its correct un-seeded default. See "v6 — welford OUT-OF-SCOPE" + welford_probe.py.
- **layer_norm_fwd: T1** confirmed (classify_layer_norm.py). 1 block_size, 1 reduction_loop, 1 RF, no
  matmul → eligibility gate (`len(reduction_loops)==1`) PASSES. The TWO reductions over N (mean=`sum(x)`,
  var=`sum(centered^2)`) reduce over the SAME N rdim → ONE rollable rdim → 1 reduction_loop (the gate's
  single-rdim assumption holds with 2 reductions). RF: **num_reduction_ops=2** (two ReductionLowerings
  over the rdim), **num_load=3** with bias (x + weight + bias) / **num_load=2** without bias, num_store=3
  (out, mean, rstd), dtype=fp32. Heuristic FIRES 1 seed (`triton_reduction_tile`), seed used + correct.
  M-block autotuner_min=1 even at 8192 rows. (A benign `TensorOperationInWrapper` warning fires on the
  `if bias is not None` host-side branch — unrelated to the reduction seed; bind/seed/codegen all fine.)

## ReductionFact design (config_spec.py, after MatmulFact)
NamedTuple, one per registered ReductionLoopSpec (T1 rollable rdim). Populated in
`device_ir.register_rollable_reductions._build_reduction_fact` (2nd-pass loop, reading the ORIGINAL
graphs that USE the rdim). Fields (grown by co-design):
- `block_id`, `size_hint` (rnumel — the persistent-vs-looped lever)
- `m_block_ids` (non-reduction kept-tile block_ids)
- `static_rnumel` (rnumel iff compile-time constant, else None)
- `dtype`, `itemsize` (read as a FACT so the heuristic generalizes to bf16/fp16 — never hardcode fp32;
  the persist threshold is expressed in BYTES via itemsize)
- `num_load`, `num_store` (memory-op counts in the rolling graphs — arith-intensity / live-state proxy
  → Band A vs Band B distinction later)
- `num_reduction_ops` (count of ReductionLowerings over this rdim → #accumulators; welford-like)
Observed for rms_norm (all shapes): num_load=2, num_store=2, num_reduction_ops=1, dtype=fp32, itemsize=4.

## Heuristic decisions (with empirical why)
- **Persistent-vs-looped by rnumel-in-BYTES (the first & dominant lever).**
  Branch: `rnumel*itemsize <= PERSIST_MAX_BYTES` → `reduction_loops=[None]` (persistent); else looped
  chunk `LOOPED_CHUNK`. Threshold in BYTES (via itemsize) so it generalizes across dtypes.
  WHY persistent: the un-seeded Triton default goes LOOPED (chunk=min(next_pow2,4096)) once rnumel>4096,
  which LOSES big at wide shapes ((2048,16384) G_default=0.70, etc.) because tc/Helion-max keep the row
  PERSISTENT. Seeding persistent recovers all to G≈0.99.
- **[v6] Multi-load persist byte-cap (`MULTILOAD_PERSIST_MAX_BYTES = 128 KiB`).** REFINES the v3 "structural
  cap only" claim, which was correct ONLY for num_load=1. For `num_load >= 2` AND `rnumel*itemsize > 128 KiB`
  → LOOPED (T1: `reduction_loops=[16384]`; T2: cap R_BLOCK). WHY (the cross_entropy widening): a matched-WARPS
  persistent-vs-looped A/B across num_load (persist_crossover_by_numload.py, ce_crossover_tight.py) shows the
  crossover is num_load-DEPENDENT — num_load=1 (sum) ties persistent==looped to 1 MiB (no change), num_load>=2
  (rms_norm/layer_norm/cross_entropy) persistent LOSES 1.6-4x above ~256 KiB (a multi-load reduction re-streams
  the wide row per load pass; a persistent kernel holding it resident spills). Cap at 128 KiB (sends 65536/
  256 KiB looped — at 256 KiB persistent ties but the w32 ramp is suboptimal there, looped is the robust
  choice). INERT in-sample for all 7 (every num_load>=2 kernel <=64 KiB; ce_inert_proof_7.py = 27/27 byte-
  identical). Distinct lever from the v3/v4 num_warps debate (that was the w32 ramp = rnumel-driven for all
  num_load; THIS is persistent-vs-looped = num_load-driven). Keyed on the WORKLOAD (num_load + bytes), NOT
  kernel identity.
- **[v3] Persistent-vs-looped lever (for num_load=1) = the STRUCTURAL cap ONLY.** Persistent
  (`reduction_loops=[None]`) for every num_load=1 row up to `env.backend.max_tensor_numel` (Triton's
  `TRITON_MAX_TENSOR_NUMEL` = 2**20 elems); looped chunk ONLY above it. WHY: above the cap a single `tl.arange`
  over the row is REJECTED at codegen (`numel exceeds triton maximum tensor numel`) — so looped is structurally
  REQUIRED, not a perf choice. Below the cap, persistent wins/ties for num_load=1 (next bullet), so there is NO
  perf-based byte fence for num_load=1. (num_load>=2 gets the v6 cap above.)
  - This DELETES v1's 64-KiB fence and v2's 256-KiB `PERSIST_MAX_BYTES` fence. The auditor proved v2's
    fence was net-harmful: it sent every in-sample long_sum row (128 KiB–1 MiB) LOOPED, but persistent/w32
    beats looped/w32 on all of them. The fences were a CONFOUND with num_warps (the real lever).
- **[v3] Step-A crossover sweep (warps HELD EQUAL).** `_lab/harness/v3_crossover_sweep.py`: sum_kernel
  (num_load=1, same memory class as long_sum), persistent vs looped both at warps∈{16,32}, fp32/H100,
  median-of-9. Metric bestP/bestL over warps (>1 ⇒ looped wins):
    rnumel | KiB  | bestP/bestL across M∈{1,4,16,64,256}   | verdict
    131072 |  512 | 0.92–1.00                              | PERSISTENT
    262144 | 1024 | 0.87–1.01                              | PERSISTENT
    393216 | 1536 | 0.52–0.97                              | PERSISTENT
    524288 | 2048 | 1.20–1.22 @M≤16, ~1.0 @M≥64            | (looped@small-M*)
    786432 | 3072 | 0.54–1.08                              | PERSISTENT
   1048576 | 4096 | 1.00–1.10 (noisy ~tie)                 | PERSISTENT (= cap)
   >1048576|      | persistent FAILS to compile            | LOOPED ONLY
  i.e. persistent wins or ties at EVERY feasible byte size; the only clean looped win (524288/small-M) is
  non-monotone (persistent wins at 393216 AND 786432 around it) so it is NOT keyed. This corrects the v2
  crossover_sweep, which conflated num_warps with the loop flip (it compared *best*-persist vs *best*-loop
  over different warps).
- **[v3] num_warps ramp, gated on `num_load` (the generalizable lever, NOT kernel identity).**
  `_num_warps`: for `num_load==1` (single-stream: sum, long_sum) AND rnumel > 16384 → **32**; else the v1
  ramp (`<=1024→4`, `<=4096→8`, else `16`). EVIDENCE (_lab/harness/v3_persist_warps_ramp.py, sum_kernel
  PERSISTENT path): w32 dominates from rnumel 32768 up (rnumel 262144/M=1: w4=47.9us → **w32=11.1us**, a
  4.3× speedup). The w32 step sits STRICTLY ABOVE 16384 so sum's max in-sample row (16384) is unchanged.
  - **WHY gated on num_load, not just rnumel:** rms_norm (num_load=2, re-reads x) NEVER wants w32.
    EVIDENCE (_lab/harness/v3_rmsnorm_warps_ab.py): rms_norm best warp is w4–w16 everywhere; at large-M
    tiny-N (32768,256) w16=574us and w32=**1182us** (catastrophic — high warps couple badly with the
    small M-block, consistent with the harness-integrity coupled-warps×block finding). So the w32 win is a
    num_load=1 property (streaming, bandwidth-bound), not a generic huge-rnumel one. This is the
    generalize-don't-pattern-match distinction the auditor demanded.
- **[v3] DELETED the grid-occupancy branch.** Its premise ("looped wins at small M") was a CONFOUND: the
  worker's grid_occupancy_probe compared persistent/**w16** vs looped/**w32**, attributing a pure
  num_warps win to the loop flip. At equal warps persistent wins. `_m_extent` is kept as a DIAGNOSTIC-only
  helper (no branch keys on it) for trace/audit scripts.
- **LOOPED_CHUNK = 16384, LOOPED_NUM_WARPS = 32** (unchanged from v2): only reached above the structural
  cap; re-confirmed adequate for the >1 MiB rows in v3_crossover_sweep.py.
- **num_stages=1.** Both paths run a single (rolled) reduction pass; default is 1.

## sum is a WASH, why (auditor's finding, root-caused — a generalizable property, not kernel identity)
- G_seed(sum)≈G_default(sum)≈0.93 while rms_norm WON big. The seed DOES change codegen for sum (at
  rnumel 8192/16384 the default goes looped chunk-4096, the seed goes persistent — verified). Yet
  persistent≈looped in PERF for sum, but persistent≫looped for rms_norm. ROOT CAUSE = **num_load**:
  rms_norm has **num_load=2** (re-reads x for the normalize pass) → looped re-streams x from DRAM twice
  with poor reuse, so persistent (x resident) wins; sum has **num_load=1** → looped vs persistent both
  stream x exactly once and are equally bandwidth-bound, so it's a wash. This is the workload property
  (num_load, already a ReductionFact field) that distinguishes the two — NOT kernel identity. The seed's
  remaining gap to tc on sum (~0.84–0.93 at mid shapes) is tc being a better pure-sum codegen, not a
  default-vs-seed gap; no seed change recovers it without an indexing/codegen lever we don't yet expose.
- **M-block = autotuner floor, not forced 1.** The CuTe template gated on M-axis `floor<=1` and seeded
  `block_sizes=[1]`. Triton's `raise_grid_block_minimums` raises `autotuner_min` to 2+ for LARGE-M shapes
  (e.g. 32768 rows → autotuner_min=2) purely to keep the autotuner from exploring tiny-block huge-grid
  configs — it is NOT a correctness limit on block=1. So we BROADENED the gate to accept any floor and
  seed `block_sizes=[max(1,min_size,autotuner_min)]`. This is what lets (32768,256)/(32768,1024) get a
  seed (they were silently skipped with the `<=1` gate). They land on G≈0.94/0.99.

## Bare-seed verification (Step-2 seed-USED proof)
- `compiler_seed_configs(env, device_ir)` now returns exactly 1 reduction seed for rms_norm_fwd, with
  `config_spec.autotuner_heuristics == ['triton_reduction_tile']`. Verified via
  `_lab/harness/verify_step2.py`. For each of 4 representative shapes the EXACT seed was run BARE
  (`configs=[seed]`, len==1 short-circuit, no autotune): `seed_used=True` (codegen persistent-vs-looped +
  num_warps in launcher match the normalized config), correctness PASS (max_abs ~1.9e-6 « rtol=1e-3),
  stable latency.

## Per-shape G_rms_norm (v1 seed, kernel-only do_bench, fp32, GPU2; two runs agree)
| shape | codegen | warps | G_seed | G_default(baseline) |
|---|---|---|---|---|
| (2048,1024) | persist | 4 | 1.03 | 1.00 |
| (2048,2048) | persist | 8 | 0.87 | 0.88 |
| (2048,4096) | persist | 8 | 1.00 | 0.99 |
| (2048,8192) | persist | 16 | 0.99 | 0.81 |
| (2048,16384) | persist | 16 | 0.99 | 0.70 |
| (4096,1536) | persist | 8 | 0.98 | 1.02 |
| (4096,3584) | persist | 8 | 1.02 | 1.00 |
| (4096,5120) | persist | 16 | 0.98 | 0.91 |
| (4096,7168) | persist | 16 | 0.99 | 0.83 |
| (8192,4096) | persist | 8 | 0.99 | 1.00 |
| (8192,8192) | persist | 16 | 0.99 | 0.75 |
| (32768,256) | persist | 4 | 0.94 | 0.98 |
| (32768,1024) | persist | 4 | 0.99 | 0.99 |
| **GEOMEAN** | | | **0.983** | **0.908** |

(2048,2048) — REFEREE CORRECTION (was wrongly called a measurement artifact): this is a REAL G≈0.87,
  NOT an artifact. In a fresh-subprocess-per-shape re-measure it is stable at G_seed≈0.87 — torch.compile
  is genuinely faster at this medium shape (tc≈17.6us vs both Helion variants ≈20us). The prior worker's
  "isolated probe reads 13.5us → G=1.32" claim does NOT reproduce (worker error). Worse, the SEED
  (num_warps=8 here) mildly REGRESSES ≈1.4% vs the un-seeded default (num_warps=4): so at this one shape
  the heuristic's warps=8 breakpoint is slightly wrong vs warps=4. It is well within the −10% backstop
  (a real ~1.4% local loss, not a tie/win), but it is a genuine small loss, not an artifact. Open lever:
  revisit the num_warps breakpoints with a COUPLED warps×block A/B (warps and block are coupled — never
  A/B warps with block pinned).

NOTE: v2 (raised thresholds) is a NO-OP on rms_norm (all rows ≤64 KiB stay persistent with identical
warps); re-measured G_rms_norm=0.982, codegen/warps identical to v1. So the v1 table above still stands.

## Per-shape G_sum (v3 seed, do_bench median-of-7, fp32, GPU3) — a WASH (UNCHANGED from v2)
| shape | codegen | warps | G_seed | G_default | maxrel |
|---|---|---|---|---|---|
| (2048,1024) | persist | 4 | 1.003 | 1.006 | 7e-4 |
| (2048,4096) | persist | 8 | 0.899 | 0.895 | 2e-3 |
| (2048,16384) | persist | 16 | 0.930 | 0.925 | 3e-4 |
| (4096,1536) | persist | 8 | 0.936 | 0.914 | 2e-3 |
| (4096,5120) | persist | 16 | 0.844 | 0.842 | 3e-3 |
| (8192,256) | persist | 4 | 1.003 | 0.970 | 9e-4 |
| (8192,4096) | persist | 8 | 0.886 | 0.885 | 3e-2* |
| (32768,256) | persist | 4 | 1.018 | 1.006 | 1e-2* |
| (32768,1024) | persist | 4 | 1.001 | 1.001 | 9e-3* |
| **GEOMEAN** | | | **0.9449** | **0.9365** | |
\* high maxrel is on near-zero row sums (random normals sum ≈0 → tiny denominator blows up RELATIVE
error); absolute error «atol=1e-3, allclose PASSES. Standard fp32-sum-near-zero. NOTE: warps are IDENTICAL
to v2 — the w32 step is gated rnumel>16384, and sum's max in-sample row is exactly 16384 → stays w16.

## Per-shape G_long_sum (v3 seed, fresh-process-per-shape do_bench median-of-9, fp32, GPU2) — the FIX
v3 seed = PERSISTENT/w32 for ALL in-sample shapes (was LOOPED/w32 in v2). G_p32 column = the decisive A/B
(v3 seed should TIE persistent/w32 since it IS that config). seed_used=True, correctness PASS all shapes.
| shape | codegen | warps | G_seed | G_default | G_p32 | seed/p32 | note |
|---|---|---|---|---|---|---|---|
| (1,32768) | persist | 32 | 1.328 | 0.636 | 1.440 | 1.084† | tiny-M, 5us, noise floor |
| (2,65536) | persist | 32 | 1.135 | 0.367 | 1.214 | 1.070† | tiny-M, 7us, noise floor |
| (4,130000) | persist | 32 | 1.057 | 0.192 | 1.057 | 1.000 | exact tie |
| (8,131072) | persist | 32 | 1.123 | 0.277 | 1.123 | 1.000 | exact tie |
| (16,262144) | persist | 32 | 0.897 | 0.230 | 0.897 | 1.000 | exact tie |
| **GEOMEAN** | | | **1.099** | **0.310** | | | v2 geomean was 1.018 |
† seed/p32 = 1.08/1.07 at the two tiny-M shapes is NOISE on BYTE-IDENTICAL configs (5–7us latencies, near
the do_bench floor). Earlier 3-run measures of the same configs gave 1.000/1.009. The seed IS persistent/
w32 (branch trace + codegen confirm rl=[None],w32). On the 3 larger shapes seed/p32 = 1.000 exactly.

### High-M held-out long_sum (lift above the noise floor) — v3 persistent/w32, seed/p32 = 1.000 exactly
| shape | G_seed | G_default | seed/p32 |
|---|---|---|---|
| (256,131072) | 1.038 | 0.935 | 1.000 |
| (256,262144) | 1.094 | 0.992 | 1.000 |
| (128,131072) | 0.920 | 0.649 | 1.000 |
These confirm the persistent/w32 win generalizes across M (not just tiny-M) and is real above the noise
floor; default still loses (0.65–0.99). v3 seed == persistent/w32 exactly at these higher latencies.

## Looped tail DISCLOSURE (no in-sample coverage; synthetic/structural evidence ONLY)
The looped branch fires ONLY for rnumel > the backend element cap (Triton 2**20 = 1048576 elems). **NO
in-sample shape — and no held-out shape below ~1 MiB rows — reaches it.** Proof it is structurally REQUIRED
(not a perf fence): for (1,2097152) the persistent/w32 config FAILS to compile ("numel 2097152 exceeds
triton maximum tensor numel 1048576"); the v3 seed correctly goes LOOPED, is correct (maxabs 3.7e-4), and
beats the un-seeded Helion default 10.8× (42.6us vs 462us). It still LOSES to torch.compile there
(G_seed=0.303 — tc uses a multi-stage/atomic split reduction for enormous rows that a single looped Helion
kernel doesn't match) — but this is a disclosed generalization tail with no in-sample coverage, NOT tuned.
See `_lab/harness/v3_looped_tail_check.py`. No silent caps.

## Oracle field-diff — sum + long_sum (next levers)
- **sum** (quick-autotune, fair re-bench of FULL verbatim winner): oracle ≈ seed within ~1–3% at most
  shapes (e.g. (2048,4096) seed/oracle both persistent; (32768,256) oracle goes block≥2 + small looped
  chunk like rms_norm's tiny-N winner). Headroom for sum is vs tc, not vs default — exposed via levers we
  don't yet seed (indexing/eviction). Low priority (sum is a wash).
- **long_sum** (quick-autotune, fair re-bench): field-diff (seed → oracle):
  | shape | seed redloop/warps | oracle redloop/warps/stages | G_seed | G_oracle |
  |(1,32768) | [16384]/32 | [16384]/32/1 | ~1.3 | ~1.4 (some runs) |
  |(2,65536) | [16384]/32 | [16384]/32 | ~1.1–1.26 | match |
  |(4,130000)| [16384]/32 | [None]/.../3 stages | 0.88 | 0.98 ← oracle persistent+stages3 |
  |(8,131072)| [16384]/32 | [8192]/32/1 | 1.00 | match |
  |(16,262144)| [16384]/32 | [16384]/32/1 | 0.87 | 1.14 |
  TWO residual levers: (a) **num_stages>1** for the very largest rows ((4,130000),(16,262144)) — oracle
  picks 3; (b) the chunk is near-optimal (8192–16384 both fine). These are the next A/B targets for
  long_sum. NOTE: the tiny-absolute latencies (5–17 us) make G noisy run-to-run; the referee should pin
  per-shape fresh-process timing.

## Tried and rejected (with why it failed)
- _Gate `M-floor <= 1` (CuTe template's): REJECTED — silently dropped (32768,*) shapes whose autotuner_min
  is 2. Replaced with "accept any floor, seed block at the floor"._
- **[v2, REJECTED in v3] Byte-fence `PERSIST_MAX_BYTES` + grid-occupancy branch.** Both were a CONFOUND
  with num_warps. The v2 crossover/grid-occupancy probes compared persistent/**w16** vs looped/**w32** and
  attributed the warp win to the loop flip. At EQUAL warps (Step-A sweep) persistent wins/ties at every
  feasible byte size; the fences sent in-sample long_sum rows looped and LOST 1.04–1.16×. DELETED in v3;
  the warps=32 lever moved to the persistent path (gated on num_load). The METHODOLOGY LESSON: A/B every
  branch against the best SIMPLE alternative (persistent/w32), holding all OTHER levers equal — never vs
  the catastrophic default strawman, never conflating warps with loop-flip.

## Open hypotheses
- **(2048,2048)** resolved-as-real; rms_norm num_warps=8 vs 4 there is a small open lever (coupled
  warps×block A/B). v3 unchanged here.
- **[v3 RESOLVED] persistent-vs-looped crossover (warps held equal)** — persistent wins to the structural
  cap (2**20 elems); looped only above it. The byte fence + grid-occupancy branch are DELETED. Done.
- **[v4 RESOLVED, supersedes v3] num_warps=32 lever** — it is an **rnumel** property, NOT a num_load one.
  The w32 step is gated on `rnumel > 16384` ALONE in the persistent path. The v3 `num_load==1` gate was
  REJECTED (inert in-sample 0/27, false on the matched-pair A/B where num_load=2 ALSO wants w32 at large
  rnumel — rnumel=131072 w32/w16=0.57, harmful OOS where it denied rms_norm/layer_norm a 30-40% w32 win).
  rms_norm (num_load=2) and layer_norm (num_load=3) now correctly get w32 at large rnumel. The earlier
  "rms_norm never wants w32" claim was an ARTIFACT of only testing in-sample rnumel<=16384 (where nobody
  wants w32) plus the (32768,256) catastrophe — which is an rnumel=256 effect, already excluded by >16384.
  Done.
- **num_stages>1 for the very largest rows** — long_sum oracle field-diff picked stages=3 on (4,130000)/
  (16,262144). Now those rows are PERSISTENT (not looped); re-A/B stages 2–3 in the persistent path for
  huge num_load=1 rows (gated on rnumel, generalizable). Open; small residual headroom. (16,262144) is the
  one in-sample long_sum shape still <1.0 (G 0.897) — stages may be the lever; revisit.
- **sum vs tc gap** — sum is a wash vs default; the ~5–15% gap to tc is a codegen/indexing lever we don't
  expose. Low priority.
- **Looped tail vs tc** — above the structural cap (>2**20 elems) the single looped Helion kernel loses to
  tc's multi-stage split reduction (G~0.30 at 2M elems). A split-K / atomic-accumulate looped recipe could
  close it, but NO in-sample shape reaches the cap — pure generalization-tail headroom, deferred.
- **[v6] cross_entropy (8192,131072) G=0.54 — kernel-source gap, not config headroom.** The best CORRECT
  Helion looped config (chunk 32768/w32=3485us; my seed 16384/w32=3575us, within 2.5%) loses to tc's 1928us
  because Helion CE re-reads the wide row for the exp-sum pass while tc fuses an online (single-pass) softmax.
  num_stages doesn't help. Closing it needs a kernel-source rewrite (online logsumexp), out of scope for a
  config seed. Bumping LOOPED_CHUNK 16384->32768 would recover ~2.5% (tiny; 16384 is the validated chunk).
- **[v6 OPEN — welford future work, the HELPER_REQUEST]** welford is OUT-OF-SCOPE for the current single-axis
  T2 seed (declined). A correct+fast welford seed needs TWO things, both welford-shaped (validated on ONE
  kernel → no held-out coverage; do NOT force as a kernel-identity hack now): (1) **widen the SECOND N-axis
  tile** (the normalize pass, block 2) to persistent too — the ceiling sweep (welford_ceiling.py) shows
  M=1/R1=R2=N/w8-16 ties tc (G~0.99) at pow2 N; (2) a **non-pow2 correctness guard** that seeds the
  Welford-combine tile (R1) as a looped DIVISOR chunk (e.g. 512 | 1536) instead of next_pow2 — a masked
  persistent R1 makes `Tn=chunk.size(-1)` count the padding and breaks the mean/variance (err=0.696 at
  N=1536). This is a general "multi-pass user-tiled reduction" treatment; needs >=2 such kernels to validate
  the secondary-tile-widening rule and the non-pow2 guard before shipping. Until then: DECLINE (correct
  default fallback) is the honest choice.

## Oracle field-diff (answer key) — CORRECTED per harness-integrity
- **(32768,256) full-autotune VERBATIM winner** (re-parsed from /tmp/autotune_log_32768_256.csv, the
  real full-autotune CSV): `block_sizes=[4]`, **`reduction_loops=[128]` (LOOPED, not persistent!)**,
  `num_warps=1`, `num_stages=5`, persistent_interleaved pid, some tensor_descriptor indexing + eviction.
  Autotuner perf 30.6us. The top-8 are all `block_sizes=[2 or 4]`, `reduction_loops=[128]`, warps 1-2.
- **CORRECTION (harness-integrity):** the earlier "oracle num_warps=32 is an artifact" story was itself a
  FIELD-DIFF BUG in our oracle_field_diff.py — it flattened a coupled multi-field winner and re-benched a
  FABRICATED block=1 config. warps×block are COUPLED; block=1/w32 (1174us) is a config the autotuner NEVER
  tested (raise_grid_block_minimums floors the M-block at 2 for 32768 rows). The autotuner do_bench is NOT
  biased (3 timing methods agree <1%). So the FIX: always re-bench the FULL verbatim oracle config (all
  levers together), never a single isolated lever. (Done — see oracle_field_diff.py guard.)
- **Field-diff verdict for small-N/large-M (32768,256):** the oracle reaches ~30.6us with
  block=4 + LOOPED chunk 128 + warps=1; my seed is persistent + block=2(floor) + warps=4 at ~36us. So the
  oracle exposes a real ~15% headroom here via (a) a larger M-block (more rows/program → better grid
  occupancy at tiny N) and (b) a small looped chunk + warps=1 (at N=256 a single warp suffices; 1 row
  doesn't saturate even 4 warps). This is a tiny-N / large-M regime the persistent-warps=4 seed
  under-serves — a candidate lever once sum/long_sum widen (long_sum is the extreme of this regime).
- (32768,1024): fair A/B warps: warps=4 best (G=1.00-1.005); high warps hurt (w32=G0.70). Seed warps=4 is
  fair-optimal there.
- Medium/large shapes: my seed already hits G≈0.98-0.99 (≈ the G_oracle_ceiling of ~1.0 from step1), so
  the field-diff headroom there is small; deferred a clean quick-effort oracle sweep for the full 13.

## Oracle cache pointers
- See `_lab/ledger.json` `oracle_cache`. Field-diff script: `_lab/harness/oracle_field_diff.py`
  (Helion effort=full). NOTE: full-effort autotune over 3 shapes is SLOW (>10min/shape) and its internal
  timing is biased for tiny-N (see above) — prefer quick-effort + a FAIR re-bench of the winner.
