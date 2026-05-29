# Lab Notebook — Reduction Autotuner Heuristics

> The DURABLE source of truth for the hill-climb. A fresh worker reads this to continue losslessly.
> Maintained by the worker (decisions + empirical why; tried-and-rejected + why; open hypotheses;
> champion). The hub appends gate verdicts. Keep it current at every clean iteration boundary.

## Champion (current best heuristic)
- **v1 `triton_reduction_tile`** (ACCEPTED 2026-05-28). Referee-confirmed **G_rms_norm = 0.979** vs
  un-seeded `default_config` baseline 0.908 (+7.8%). Auditor PASS. Code:
  `helion/_compiler/autotuner_heuristics/triton.py` `TritonReductionHeuristic`.
- **v2 `triton_reduction_tile` (PROPOSED, 2026-05-29) — WIDENED to 3 kernels + raised thresholds.**
  WORKER-measured (kernel-only do_bench, fp32, GPU2; awaiting referee). Per-kernel G_seed vs the
  un-seeded default baseline:
  - rms_norm_fwd: **0.982** (champion 0.979 — UNCHANGED; all in-sample rows still persistent, threshold
    raise is a no-op here). default baseline 0.914.
  - sum:          **0.931** — a WASH (default baseline 0.933; the seed neither helps nor hurts — see
    "sum is a wash, why" below). No regression; tc is just genuinely a bit faster for a 1-load sum.
  - long_sum:     **1.018** — HUGE generalization win: default baseline **0.311** (the un-seeded default
    loops chunk-4096/warps-4 → 3.3× too slow on these huge rows). The seed reaches/slightly beats tc.
  v2 changes vs v1: PERSIST_MAX_BYTES 65536→**262144** (64→256 KiB), LOOPED_CHUNK 4096→**16384**, looped
  num_warps→**32**, and a NEW **grid-occupancy** branch (M-extent<64 ∧ rnumel≥128KiB → looped/warps32).
  rms_norm + sum are byte-for-byte unaffected; the wins are entirely on long_sum (the looped + grid-
  occupancy branches, previously UNTESTED, now validated). NOT yet referee-confirmed (HELPER_REQUEST).
  NOTE on O: prior O=0.979 was rms_norm-ONLY; new 3-kernel geomean O=**0.976** is NOT directly comparable
  (adding a 0.93 wash mechanically lowers a geomean even though the heuristic strictly improved). The
  honest claim is per-kernel: every active kernel beats/matches its default, no kernel regresses, and
  long_sum is a 3.3× generalization win.

## Objective
- Product A: maximize `O = geomean_k G_k`, `G_k = geomean over kernel k's in-sample shapes of
  (tc_default_latency / seed_latency)`. Accept iff O improves AND gates pass (correctness; seed used;
  no active kernel's referee-confirmed G_k regresses >10% vs champion).
- Product B (every 5 iters): seeded vs unseeded quick-autotune convergence curve.

## Active kernels (curriculum)
- Active: **rms_norm_fwd, sum, long_sum** (all T1, Band A). Widen next to: layer_norm-fwd, softmax
  (Band A); kl_div, jsd (Band B); welford (Band C). Forward only for now; defer backward (Band D).

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
- **[v2] PERSIST_MAX_BYTES = 262144 (256 KiB = 65536 fp32 elems)** — was 65536 BYTES (16384 elems) in v1,
  which the auditor correctly flagged as a FENCE set at the in-sample rms_norm max row (~16× too low).
  EVIDENCE (synthetic crossover sweep, _lab/harness/crossover_sweep.py, best-persist vs best-loop over
  warps/stages): persistent KEEPS WINNING up to ~256 KiB for grid-occupied shapes —
  rnumel 16384/32768/49152/65536 all persist-win at M≥1024 (pers/loop 0.98/0.80/0.74/0.86); looped only
  wins from 98304 (384 KiB: 2.9–3.1×). For grid-starved M=8 the crossover is a touch lower (~192 KiB).
  We pick the OCCUPIED crossover 256 KiB; tiny-M loses only ~6% exactly at 65536 elems (and the
  grid-occupancy branch below catches that case anyway). rms_norm is unaffected (all rows ≤64 KiB).
- **[v2] LOOPED_CHUNK = 16384** (was 4096). EVIDENCE (_lab/harness/looped_chunk_probe.py): in the
  looped-winning region a LARGER R_BLOCK wins — at M=8/rnumel=131072 best-us by chunk: 2048→36.4,
  4096→25.1, 8192→23.5, **16384→22.0**; same ordering at M=4/16/1024. ~15–25% over old 4096.
- **num_warps:** persistent branch scales with rnumel (`<=1024→4`, `<=4096→8`, else `16`). Looped branch
  uses **LOOPED_NUM_WARPS=32** [v2] — EVIDENCE: in the looped/grid-starved regime best (w,s) was (32,1)
  at essentially every probe case; few programs ⇒ each must extract max ILP over the long row.
- **[v2] Grid-occupancy lever (SECOND branch, on M-extent = #rows = grid size).** When `m_extent < 64`
  AND `rnumel >= 128 KiB` → force the LOOPED+warps32 recipe even UNDER the byte ceiling.
  WHY (generalizable, occupancy): a persistent reduction runs ONE pass per program; with a grid far below
  the H100's ~132 SMs the GPU is under-filled and a single pass can't hide memory latency. EVIDENCE
  (_lab/harness/grid_occupancy_probe.py — sum_kernel, persistent/16 vs looped(16384)/32, sweeping the
  grid M at rnumel 32768 & 65536, both UNDER 256 KiB):
    M(grid):  1    2    4    8    16   32  | 64   128  256  1024
    pers/loop 1.17 1.20 1.19 1.14 1.05 1.10| 0.98 1.00 1.00 1.01
  i.e. for M≲32 looped+warps32 wins 5–21%; at M≳64 they wash. Threshold 64 ≈ SMs/2. The `rnumel≥128 KiB`
  guard (smallest row where the grid-starved win was observed) prevents looping a tiny row just because M
  is small. This branch fires ONLY for long_sum's tiny-M shapes ((1,32768),(2,65536)); rms_norm/sum have
  M≥2048 so it never touches them. Lifted long_sum (1,32768) G 1.24→1.32, (2,65536) 0.94→1.09.
- **num_stages=1.** Persistent + looped both run a single (rolled) reduction pass; default is 1. The
  long_sum oracle field-diff hinted at stages=3 on a couple of huge shapes (4,130000)/(16,262144) — small
  residual headroom, OPEN.

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

## Per-shape G_sum (v2 seed, kernel-only do_bench, fp32, GPU2) — a WASH
| shape | codegen | warps | G_seed | G_default | maxrel |
|---|---|---|---|---|---|
| (2048,1024) | persist | 4 | 0.96 | 0.96 | 8e-5 |
| (2048,4096) | persist | 8 | 0.89 | 0.90 | 3e-4 |
| (2048,16384) | persist | 16 | 0.93 | 0.93 | 2e-4 |
| (4096,1536) | persist | 8 | 0.89 | 0.91 | 9e-5 |
| (4096,5120) | persist | 16 | 0.85 | 0.84 | 2e-3 |
| (8192,256) | persist | 4 | 1.00 | 0.97 | 4e-4 |
| (8192,4096) | persist | 8 | 0.89 | 0.89 | 3e-4 |
| (32768,256) | persist | 4 | 0.99 | 1.00 | 1e-2* |
| (32768,1024) | persist | 4 | 1.00 | 1.00 | 2e-3 |
| **GEOMEAN** | | | **0.931** | **0.933** | |
\* maxrel 1e-2 at (32768,256) is on near-zero row sums (256 random normals sum ≈0 → tiny denominator
blows up relative error); absolute error «atol=1e-3, allclose PASSES. Standard fp32-sum-near-zero.

## Per-shape G_long_sum (v2 seed, kernel-only do_bench, fp32, GPU2) — BIG win over default
| shape | codegen | warps | G_seed | G_default | note |
|---|---|---|---|---|---|
| (1,32768) | LOOPED | 32 | 1.32 | 0.67 | grid-occupancy branch (M=1<64, 128KiB≥128KiB) |
| (2,65536) | LOOPED | 32 | 1.09 | 0.37 | grid-occupancy branch (M=2<64, 256KiB≥128KiB) |
| (4,130000) | LOOPED | 32 | 0.88 | 0.19 | byte-ceiling branch (508KiB>256KiB) |
| (8,131072) | LOOPED | 32 | 1.00 | 0.27 | byte-ceiling branch (512KiB>256KiB) |
| (16,262144) | LOOPED | 32 | 0.87 | 0.23 | byte-ceiling branch (1MiB>256KiB) |
| **GEOMEAN** | | | **1.018** | **0.311** | seed = 3.3× the un-seeded default |

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

## Open hypotheses
- **(2048,2048)** resolved-as-real (see above); num_warps=8 vs 4 there is a small open lever (coupled
  warps×block A/B).
- **[v2 RESOLVED] PERSIST_MAX_BYTES + LOOPED_CHUNK + looped warps** — set by the crossover + chunk sweeps
  (256 KiB / 16384 / 32). Looped branch now TESTED (long_sum). Done.
- **[v2 RESOLVED] grid-occupancy** — tiny-M wants looped/warps32 even under the byte ceiling; added the
  branch (M-extent<64 ∧ rnumel≥128KiB). Done.
- **num_stages>1 for the very largest looped rows** — long_sum oracle field-diff picks stages=3 on
  (4,130000) (G 0.88→0.98 persistent+stages3) and (16,262144) (G 0.87→1.14). NEXT lever: try stages=2–3
  in the looped branch for huge rnumel (gated on rnumel, generalizable). Must re-check it doesn't hurt
  the smaller looped shapes. (perf-investigator: why does stages help only the largest rows?)
- **sum vs tc gap** — sum is a wash vs default; the ~10–15% gap to tc is a codegen/indexing lever we don't
  expose. Low priority.
- **(4,130000) oracle prefers PERSISTENT** while our byte-ceiling sends it looped (508KiB>256KiB), 5% gap.
  The byte ceiling is right for occupied shapes; for tiny-M the persistent-vs-looped crossover at huge
  rnumel is subtler (oracle: persistent+stages3). Possible refinement: tiny-M may want persistent+stages
  even above 256KiB up to some larger bound. Needs its own sweep; small headroom.

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
