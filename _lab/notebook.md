# Lab Notebook — Reduction Autotuner Heuristics

> The DURABLE source of truth for the hill-climb. A fresh worker reads this to continue losslessly.
> Maintained by the worker (decisions + empirical why; tried-and-rejected + why; open hypotheses;
> champion). The hub appends gate verdicts. Keep it current at every clean iteration boundary.

## Champion (current best heuristic)
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

## Product B — seed the autotuner (RUN 2026-05-29, v4 seed) — TIME WIN CONFIRMED + an injection TRAP
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
- Active: **rms_norm_fwd, sum, long_sum, layer_norm_fwd** (all T1, Band A). Widen next to: softmax
  (Band A); kl_div, jsd (Band B); welford (Band C). Forward only for now; defer backward (Band D).

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
- **[v3] Persistent-vs-looped lever = the STRUCTURAL cap ONLY.** Persistent (`reduction_loops=[None]`)
  for every row up to `env.backend.max_tensor_numel` (Triton's `TRITON_MAX_TENSOR_NUMEL` = 2**20 elems);
  looped chunk ONLY above it. WHY: above the cap a single `tl.arange` over the row is REJECTED at codegen
  (`numel exceeds triton maximum tensor numel`) — so looped is structurally REQUIRED, not a perf choice.
  Below the cap, persistent always wins/ties (next bullet), so there is NO perf-based byte fence.
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
