# RUN 3 — Lab Notebook (persistent worker; source of truth)

> The hill-climb for the Helion forward-inner-reduction Triton **seed** heuristic (H100/sm_90, fp32).
> This notebook — NOT my context — is the source of truth. A fresh worker must be able to continue
> losslessly from here. Every iteration: decision + empirical WHY + tried-and-rejected + open hypotheses
> + current champion.

## The bar (run 3 is STRICTLY per-shape; the aggregate is NOT the bar)

- `G = tc_default_lat / seed_lat`. **FLOOR:** `G >= 1-eps` (eps≈0.05). Necessary, NOT victory.
- **ORACLE (the real bar):** best config the Helion autotuner finds for that shape + its latency.
  **VICTORY = `seed/oracle <= 1+eps`** (eps 3-5%). The seed MATCHES the oracle, per shape, no exceptions.
- A source ceiling caps the oracle too -> can only ever explain a seed-vs-**tc_default** gap, NEVER a
  seed-vs-**oracle** gap. If `seed < oracle`, performance is on the table.
- Run 2 declared COMPLETE on the AGGREGATE geomean O (in-sample 0.998, TEST 0.946) = failure mode #9.
  The per-shape seed/oracle table was essentially UNMEASURED. Run 3 re-opens the champion against the
  per-shape bar.

## Environment (THIS machine — verified 2026-06-03 at spawn)

- Worktree/cwd: `/home/dev/local/helion-reduction-heuristics-run2` (branch `reduction-heuristics-run2`).
- Interpreter: `/home/dev/helion/.venv/bin/python`. NEVER `pip install`.
- Wiring (re-verified): `helion.__file__` + `examples.*` resolve INTO the worktree under PYTHONPATH;
  `tritonbench` resolves to the ORIGINAL checkout `/home/dev/local/helion` (operator edits go there).
- GPU: 1x H100 index 0. Pin `CUDA_VISIBLE_DEVICES=0`; confirm idle via nvidia-smi before trusting a number.
- Canonical bare-seed invocation:
  `cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none PYTHONPATH=<worktree> <py> <script>`
- One GPU => time SERIALLY. I am `[timing]`; await perf-investigator timing runs, never concurrent do_bench.

## Standing peers (DM directly, not via hub)
- `code-investigator` [analysis] — "where/how does X work in the compiler?" (provenance, fact population).
- `perf-investigator` [timing] — "WHY is config A faster than B?" (re-bench + Triton/IR + ncu). AWAIT it.
- Gates are hub-spawned fresh per claim; I am the SUBJECT, never the operator. A commit fires the pipeline.

## The inherited champion (run-2 `TritonReductionHeuristic`) — what I'm re-opening

`helion/_compiler/autotuner_heuristics/triton.py`. Gate `_triton_reduction_eligible` =
`len(reduction_facts)==1 and not matmul_facts` (admits T1 rollable + T2 user-tiled). `get_seed_config`
branches (all workload-keyed, NEVER kernel identity):
1. **persistent-vs-looped**: persistent up to Triton structural cap `max_tensor_numel` (2^20 elems);
   multi-load extra cap `MULTILOAD_PERSIST_MAX_BYTES=131072` for `num_load>=2` (fires for wide cross_entropy).
2. **`_num_warps` rnumel ramp**: <=1024->4, <=4096->8, <=16384->16, else 32 (`STREAM_WARPS32_MIN_ELEMS=16384`).
3. **Band-B R_BLOCK cap** `BANDB_R_BLOCK_BYTES=16384` (T2 `num_tiled_accumulators>=1`: kl_div/jsd).
4. **Band-C structured combine** (welford/standardize): combine `min(np2(N), 32768/itemsize)` [persistent];
   apply persistent if `n_valid*itemsize<=12288` else looped chunk `8192/itemsize`.
5. **`_eviction_policies`**: stream (T1 num_load==1: sum/long_sum) -> `['first']*n`; reread
   (is_structured_combine: welford) -> `['last']+['first']*(n-1)`. Others (rms/ln/softmax/kl/jsd/CE) -> default.
6. **pid='flat'** everywhere; **M-block** at the autotuner floor.
7. `get_seed_configs()` = opt-in portfolio (env `HELION_REDUCTION_SEED_PORTFOLIO`).

Facts: `helion/autotuner/config_spec.py` (`ReductionFact`). Population: `helion/_compiler/device_ir.py`
(`register_rollable_reductions` T1, `register_user_tiled_reductions` T2, `_count_reduction_workload`).
**Everything is rewritable** — branch structure AND ReductionFact vocabulary. The oracle gate makes
aggression safe. Don't trust any inherited constant.

## Prior diagnostic context (HANDOFF_run3_perf_dig.md — reps=1, TEST shapes, treat as HINTS not facts)

A pre-run perf dig (reps=1 caveat) found a pattern worth re-litigating rigorously:
- **SEEDABLE gaps** cluster on **small-M, small-N, non-pow2-N** (oracle finds a config the seed misses):
  softmax (4096,640) orcl/seed 0.62; layer_norm (2048,1025) orcl/seed 0.56; rms_norm (256,4096) 0.68;
  softmax (4096,1025) 0.79; rms_norm (2048,1025) 0.64. -> warps / M-block / blocking the oracle improves on.
- **CEILINGS** cluster on **single-row (M=1, huge N)** (rms/ln/softmax ~2x off torch.compile) and **wide-N
  welford looped-apply** (Helion strategy structurally ~1.5-2x slower than tc; FULL autotune OOMed there).
- Run-2 ledger "give-up" residuals to re-litigate against a FRESH oracle (NONE settled): rms_norm
  (2048,2048)=0.871, (1,131072)=0.512 "noise floor"; cross_entropy (8192,131072)=0.539 "source ceiling";
  welford (262144,7168)=0.69 TEST; long_sum (4,524288)=0.66 "split-K deferred".

These are TEST shapes (firewalled) — I reproduce on `train` analogs / dev shapes and tune there; TEST is
read once at freeze by the ledger-keeper.

## Method (per the brief + work-order)
- Phase 1 FLOOR sweep (cheap, NO autotune): drive every `train` shape to `G >= 1-eps`. Read generated
  Triton of seed vs tc (answer key) to diff STRATEGY. Triages where the oracle budget pays off.
- Phase 2 ORACLE ascent: fresh oracle cache (cheap-first, per (kernel,N-band)), field-diff seed vs oracle,
  fix hacky/missing facts, change heuristic, correctness-gate, matched-lever A/B (re-bench oracle's FULL
  VERBATIM config), compute seed/oracle + re-confirm floor, gate, no-regression backstop, commit.
- Oracle discipline: measure ONCE per (kernel, shape, source-hash), CACHE in ledger; staleness fatal
  (invalidate on any kernel-source/codegen edit). Cheap-first: quick-autotune iterates, full confirms.

---

# ITERATION LOG

## 2026-06-03 — Spawn + Phase-1 floor sweep (DIAGNOSTIC, no heuristic edits)

**Setup:** Read the three core files + HANDOFF_run2 + the current heuristic + the perf-dig handoff. Wiring
re-verified GREEN on this machine (helion+examples->worktree, tritonbench->original checkout, H100 idle).
Made `run2_measure_g.py` machine-portable (derived `_WT_ROOT` from `__file__` instead of the old hardcoded
`/home/calebkim/...` path — per portable-lab-state discipline). Wrote `_lab/harness/run3_floor_sweep.py`
(reuses run2_measure_g's verbatim 9-kernel plumbing; iterates the `train` split; median-of-7 + spread;
re-runs high-spread shapes; ranks G ascending; per-kernel checkpoints + merged JSON to `_lab/logs/run3/`).

Smoke test (2 shapes): sum(16384,2048) G=0.996, rms_norm(8192,4096) G=0.996 — both at floor, persistent,
correct. Mechanism confirmed.

**Floor sweep result (median-of-7, spreads ~0.00-0.01, 0 correctness fails, 0 OOMs):**
Merged JSON `_lab/logs/run3/floor_sweep_merged.json`; per-kernel `_lab/logs/run3/floor_<kernel>.json`.

PER-KERNEL GEOMEAN G (train), OVERALL 0.977:
- AT/ABOVE FLOOR: long_sum 1.086, kl_div 1.078, softmax 1.053, sum 1.011, rms_norm 0.995, layer_norm 0.993
- BELOW FLOOR: **cross_entropy 0.744** (dominant), welford 0.946, jsd 0.936

27 FLOOR LOSSES (G < 0.95). The structure (NOT a flat tail — sharply clustered):

**(A) cross_entropy — the catastrophe (8 of the 10 worst; ALL looped).** The persistent->looped crossover
at V~50257 collapses the floor. Boundary is `MULTILOAD_PERSIST_MAX_BYTES=131072`: np2(V)*4 > 128KiB -> looped.
  - PERSISTENT CE (V<=32064 + the V=49152/65536 byte-edge ones): G 1.05-1.14 -- seed BEATS tc-default.
  - LOOPED CE (`reduction_loops=[16384], w=32, evict=None`): G 0.52-0.91 -- ~2x SLOWER than tc-default.
    Worst: (8192,128256)=0.520, (4096,128256)=0.531, (4096,128000)=0.532, (2048,151936)=0.541,
    (2048,256000)=0.545, (4096,98304)=0.592, (8192,50257)=0.649, (4096,50304)=0.689.
  - Crossover sharp: (8192,32064) persist G=1.085 -> (8192,50257) loop G=0.649. NOTE non-monotone in the
    looped band: (8192,49152) loop G=1.052 and (4096,65536) loop G=0.906 are NOT catastrophic -- so it is
    not "all looped CE is bad", it is "looped CE at the WIDE vocabs is ~2x off". WORTH A CLOSE LOOK.
  - This DIRECTLY contradicts run-2's "wide-CE source ceiling CLOSED by cross_entropy_online": the standard
    `cross_entropy` kernel (what train measures) has tc-default ~2x FASTER for the SAME kernel -> a 2x-faster
    strategy provably exists; this is a SEEDABLE gap, not a source ceiling. TOP PHASE-2 ORACLE TARGET.

**(B) long_sum(16,2097152) G=0.734 looped (`rl=[16384] w=32`).** The ONLY long_sum loss; every persistent
long_sum row is G 1.0-1.5 (strong). This is the run-2 ">2^20 structural looped tail, synthetic-only"
disclosure -- but it IS a train shape, looped, 27% below floor. Re-litigate (split-K? bigger chunk?).

**(C) welford graded loss (geomean 0.946).** Two regimes:
  - narrow-N persistent: (16384,768) w4 G=0.908, (16384,1024) w4 G=0.942 -- small (~40-50us) but spread 0.00.
  - wide-N looped-apply (`bs=[1,8192,2048]`, apply tile capped 2048): (4096,16384) G=0.862, (32768,8192)
    G=0.888, (65536,4096) G=0.916, (8192,5120) G=0.947, (8192,12288) G=0.951.

**(D) jsd narrow-vocab loss (geomean 0.936; ALL `bs=[4096,1] w=32` looped Band-B).** Loss at the NARROW end
(opposite to CE): (8192,30522) G=0.847, (16384,32000) 0.896, (8192,32064) 0.900, (8192,50257) 0.908,
(8192,32000) 0.910, (8192,50304) 0.915 -- recovering to ~1.0 at wide V (256000 G=0.996).

**(E) softmax small-N + one wide-N (geomean 1.053 overall; huge mid-range wins).** Losses:
(131072,256) persist w4 G=0.796, (262144,128) w4 G=0.919, (16384,512) w4 G=0.943, (4096,16384) w16 G=0.908.
Big WINS mid-range: (8192,2560) G=1.429, (2048,24576) 1.280, (128,262144) long_sum-like. The small-N losses
have tiny block_sizes ([8,256],[16,128]) -- a grid-occupancy / warps question. (NOTE (8192,2560) G=1.43 vs
tc-default tells me NOTHING about the oracle -- that shape may still be far from its oracle; need the oracle.)

INTERPRETATION: most of the curriculum (6/9 kernels) is at floor with the inherited seed (confirms run-2's
"floor is mostly reached"). The oracle budget should go FIRST at: (1) CE wide-vocab looped (G 0.52-0.69),
(2) long_sum >2^20 tail, (3) welford wide-N looped-apply + narrow-N warps, (4) jsd narrow-vocab,
(5) softmax small-N. The FLOOR pass already gives a strong answer-key hint for CE: tc-default is 2x faster
on the SAME kernel, so the looped seed strategy is wrong there -- the oracle will confirm what to do.

## 2026-06-03 — Phase-2 oracle harness + FIRST oracle (cross_entropy answer key)

Wrote `_lab/harness/run3_oracle.py` (machine-portable): fresh autotune (force=True, ephemeral triton cache),
fair-re-bench the winner with do_bench median-of-7, measure the live seed in the SAME process (noise-robust
seed/oracle), correctness-gate BOTH, field-diff seed vs oracle (the worklist), cache keyed by source-hash.
Source-hash = sha256 over examples/<kernel>.py + triton.py (heuristic) + __init__/registry + config_spec.py
(ReductionFact) + device_ir.py (fact population) — any edit invalidates. Cache: `_lab/logs/run3/oracle_cache.json`.

**FIRST ORACLE — cross_entropy(4096,50304), quick effort (autotune 107s):**
- **seed/oracle = 1.576** (58% gap — seed FAR from oracle). G_floor=0.682 (matches floor sweep 0.689).
- **oracle/tc = 1.074** — the oracle BEATS tc-default by 7.4%. => UNAMBIGUOUSLY a SEEDABLE gap, NOT a source
  ceiling (a source ceiling would have oracle<=tc). The run-2 "wide-CE source ceiling" framing is wrong for
  the standard cross_entropy kernel.
- **Mechanism:** seed=LOOPED (reduction_loops=[16384], w32), oracle=PERSISTENT (reduction_loops=[None]).
  seed 445.7us -> oracle 282.8us. The oracle ALSO picked num_stages 1->4 and mixed indexing (tensor_descriptor
  on some loads), but kept pid_type='flat'. The DOMINANT lever is persistent-vs-looped.
- **This indicts `MULTILOAD_PERSIST_MAX_BYTES=131072`.** V=50304 -> np2 65536 -> 256KiB, exactly run-2's
  claimed crossover where it said persistent LOSES 1.6-4x (ce_crossover_tight.py). The FRESH oracle says
  persistent WINS here (1.58x faster than the looped seed). TENSION to resolve: was run-2's A/B confounded
  (e.g. by num_warps, or by measuring persistent/w32 vs looped/w32 when the win needs persistent/w16+stages)?
  Running the CE vocab-range batch next to see if persistent wins at ALL wide vocabs or just near the boundary.

**ORACLE BATCH 1 (quick effort) — cross_entropy persistent-vs-looped answer key:**
Cache `_lab/logs/run3/oracle_cache.json`; batch out `_lab/logs/run3/oracle_batch1.out`.

| shape | seed/oracle | G_floor | oracle/tc | seed_rl | oracle_rl | oracle codegen | note |
|---|---|---|---|---|---|---|---|
| (8192,32000)  | 0.993 | 1.082 | 1.074 | [None]  | [None]   | persistent | VICTORY (persistent seed=oracle) |
| (8192,49152)  | 1.025 | 1.032 | 1.057 | [16384] | [None]   | persistent | tie-ish; oracle PERSISTENT, seed looped |
| (4096,50304)  | 1.576 | 0.682 | 1.074 | [16384] | [None]   | persistent | BIG GAP; oracle persistent beats tc |
| (8192,50257)  | 1.066 | 0.651 | **0.694** | [16384] | [4096]   | looped | quick oracle looped & LOSES to tc -- SUSPECT |
| (4096,98304)  | 1.578 | 0.582 | **0.919** | [16384] | [32768]  | looped | quick oracle looped & ~tc -- SUSPECT |
| (8192,128256) | 1.247 | 0.514 | **0.641** | [16384] | (looped) | looped | quick oracle looped & LOSES to tc -- SUSPECT |

**READ:** The oracle goes PERSISTENT and beats tc-default at V up to ~50304 (4096,50304: persistent, 1.58x
faster than the looped seed; oracle/tc=1.074). The seed's `MULTILOAD_PERSIST_MAX_BYTES=131072` cap forces
looped there -> the cap is WRONG at the boundary. BUT at the WIDEST vocabs (50257@M8192, 98304, 128256) the
QUICK oracle stayed LOOPED and LOSES to tc-default (oracle/tc 0.64-0.92). Per the brief, an oracle that loses
to tc is a claim to FALSIFY (quick-effort under-exploration), NOT an accepted ceiling -- ESPECIALLY when the
ADJACENT V=49152/50304 prove persistent is feasible+faster. The persistent footprint is M_BLOCK=1 * np2(V) *
4B = 256KiB-1MiB per program regardless of M (M only scales the grid), so persistent should be feasible at
M=8192 too. NEXT: re-run the 3 SUSPECT shapes at FULL effort + an explicit forced-persistent A/B to settle
whether persistent wins all the way up (=> delete the CE cap) or there's a real looped regime at the widest V.

cross_entropy.py structure (read): standard `cross_entropy` is a T1 rollable reduction over V. The row
`logits[tile_n,:]` is loaded into `logits_rows` ONCE in source, then consumed by amax-pass + exp-sum-pass.
So num_load>=2 likely reflects the compiler re-reading the row for the 2nd pass (asked code-investigator to
confirm). If so, the run-2 reasoning "wide multi-load rows re-stream and a persistent kernel spills, so loop"
is BACKWARDS for CE: persistent keeps the whole row resident and AVOIDS the re-stream -- which is exactly what
the oracle's persistent win shows. The run-2 "wide-CE source ceiling closed by cross_entropy_online" was a
misdiagnosis: tc-default + the fresh persistent oracle both beat the looped seed on the STANDARD kernel.

**COMPLETE ORACLE BATCH 1 (13 entries, quick effort).** source_hash consistent within each kernel (per-kernel
source file included). Full table sorted by kernel,N:

| shape | seed/orcl | G_floor | orcl/tc | category |
|---|---|---|---|---|
| cross_entropy(8192,32000)  | 0.993 | 1.082 | 1.074 | VICTORY (persistent) |
| cross_entropy(8192,49152)  | 1.025 | 1.032 | 1.057 | tie; oracle PERSISTENT |
| cross_entropy(8192,50257)  | 1.066 | 0.651 | 0.694 | SUSPECT (quick orcl looped<tc) |
| cross_entropy(4096,50304)  | 1.576 | 0.682 | 1.074 | SEEDABLE; oracle PERSISTENT beats tc |
| cross_entropy(4096,98304)  | 1.578 | 0.582 | 0.919 | SUSPECT (quick orcl looped<tc) |
| cross_entropy(8192,128256) | 1.247 | 0.514 | 0.641 | SUSPECT (quick orcl looped<tc) |
| cross_entropy(2048,256000) | 1.132 | 0.544 | 0.616 | SUSPECT (quick orcl looped<tc) |
| jsd(8192,30522)            | 1.196 | 0.844 | 1.009 | SEEDABLE; oracle beats tc (Band-B) |
| long_sum(16,2097152)       | 0.998 | 0.737 | 0.735 | SOURCE-LIMIT cand (seed=oracle<tc; N>2^20) |
| softmax(131072,256)        | 1.147 | 0.871 | 0.998 | SEEDABLE-ish (orcl~tc) small-N |
| welford(16384,768)         | 1.032 | 0.908 | 0.937 | near-tie; orcl wants w1+small apply |
| welford(32768,8192)        | 1.089 | 0.909 | 0.990 | SEEDABLE; orcl wants bigger apply+w32 |
| welford(4096,16384)        | 1.146 | 0.875 | 1.002 | SEEDABLE; orcl wants bigger apply tile |

THREE CATEGORIES (treat differently, per the work-order's seed<oracle vs seed≈oracle<tc distinction):

**(1) SEEDABLE, oracle proven > tc — the real wins to claim:** CE persist-boundary (4096,50304 orcl 1.58x
seed, beats tc), jsd narrow-V (1.20x, beats tc), welford wide-N apply (4096,16384 1.15x, orcl≈tc). These have
oracle/tc>=~1.0 so a real config beats both seed and tc -> heuristic is leaving perf on the table. FIX HERE.

**(2) SUSPECT — quick oracle LOOPED and LOSES to tc; FALSIFY at full effort.** The 4 widest CE shapes +
welford(32768,8192) + softmax(131072,256). An oracle that loses to tc is a claim to falsify (brief: quick
under-exploration), ESPECIALLY when adjacent shapes prove a better strategy exists. For CE the adjacent
V=49152/50304 PROVE persistent is feasible+faster -> the quick oracle just didn't explore persistent at the
widest V. Resolving via the CE persist A/B (run3_ce_persist_ab.py, no autotuner) FIRST, then full-effort
oracle on any that the A/B can't settle.

**(3) SOURCE-LIMIT candidate — seed≈oracle<tc (seed has DONE ITS JOB; residual is a kernel-source signal).**
long_sum(16,2097152): seed/oracle=0.998, BOTH looped because N=2097152 > 2^20 structural cap (persistent
can't compile), oracle/tc=0.735. Per the bar this is VICTORY for the SEED (it matches the oracle); the gap to
tc is a Product-A-via-source-rewrite opportunity (split-K / cross-CTA on a grid-starved 16-row 2M kernel),
NOT a seed-heuristic failure. welford(16384,768) similar (seed/oracle=1.032 near-tie, orcl/tc=0.937 -- but
the small orcl gain (w1) is worth checking). MUST still verify these oracles are REAL at full effort before
recording "source limit" (anti-giving-up discipline). NOTE: these are correctness-only `robustness`-adjacent
extremes in spirit, but (16,2097152) IS a train shape.

WORKLOAD-PROPERTY HYPOTHESES forming (to test in Phase 2):
- H-CE-persist: the `MULTILOAD_PERSIST_MAX_BYTES=131072` cap is WRONG for the 2-pass CE re-read pattern.
  Persistent keeps the row resident & avoids re-streaming it twice; looped re-streams. Likely fix: raise/delete
  the cap for CE, OR re-key it on the real property (re-read vs distinct-streamed-operands; awaiting
  code-investigator on whether num_load counts re-reads). The run-2 ce_crossover_tight A/B that justified the
  cap is suspect -- it may have compared persistent/w32 vs looped/w32 and missed that persistent needs w16+ns.
- H-welford-apply: `STRUCTURED_APPLY_LOOP_CHUNK_BYTES=8192` (2048 fp32) over-caps the apply tile at wide N;
  oracle wants 4096 fp32 apply (16KiB). And `_num_warps` ramp is wrong at welford extremes (w1 at N=768, w32
  at N=8192 wide-M) -- welford's structured-combine warps may need a different ramp than the streamed ramp.
- H-jsd-bandb: jsd narrow-V (Band-B) seed/oracle=1.20, oracle beats tc -- the BANDB_R_BLOCK_BYTES=16384 cap or
  the w32 may be wrong at narrow V. Need the field-diff (next: look at jsd oracle cfg).
- H-softmax-smallN: softmax small-N (131072,256) seed/oracle=1.15 -- warps/M-block/grid-occupancy at small N.

## 2026-06-03 — CE persistent-vs-looped A/B (matched-lever, NO autotuner) — CROSSOVER PINNED

`run3_ce_persist_ab.py` benches seed_looped vs forced-persist (rl=[None], +w16/+ns4 variants) vs tc-default,
median-of-7, correctness-gated, one process. Spreads ~0.00. THIS REVERSES my "cap is just wrong" hypothesis
for the WIDEST V and reveals a REAL crossover:

| V (M)        | tc_us | seed(loop16K) | best_persist | seed/bestP | bestP/tc | WINNER |
|---|---|---|---|---|---|---|
| 49152 (8192) | 563 | 539 | 527  | 1.02 | 1.07 | persist (marginal) |
| 50304 (4096) | 303 | 442 | **283**  | **1.56** | 1.07 | PERSIST (beats tc) |
| 50257 (8192) | 678 | 1042| **620**  | **1.68** | 1.10 | PERSIST (beats tc) |
| 98304 (4096) | 566 | **955** | 1837 | 0.68 | 0.40 | LOOPED (but seed 1.7x > tc) |
| 128256(8192) | 1403| **2711**| 5794 | 0.47 | 0.24 | LOOPED (but seed 1.9x > tc) |
| 256000(2048) | 743 | **1364**| 5011 | 0.27 | 0.15 | LOOPED (but seed 1.8x > tc) |

THE CE STORY SPLITS INTO TWO SEPARATE PROBLEMS (both real, opposite directions):

**(P1) V boundary ~50K: the cap fires TOO EARLY.** Persistent WINS and beats tc up to V~50304 (256KiB row),
but `MULTILOAD_PERSIST_MAX_BYTES=131072` (128KiB) forces looped from np2(V)*4 > 128KiB, i.e. from V>32768
(np2 65536 = 256KiB). So V=49152/50257/50304 are wrongly looped. Raising the cap to ~256KiB recovers
(4096,50304) 1.56x, (8192,50257) 1.68x, no-regression at (8192,49152). The TRUE crossover is between
V=50257 (persist wins) and V=98304 (looped wins) -- a row-bytes threshold near 256-320KiB.
  -> resolves the SUSPECT verdict on (8192,50257): NOT a ceiling; persist beats tc (1.095). The quick oracle
     just failed to find persist there. CONFIRMED seedable.

**(P2) V wide >=98304: looped is the RIGHT family but seed's looped PARAMS are wrong.** At V>=98304 looped
beats persist (persist spills 2-4x -- the run-2 cap reasoning is CORRECT here). BUT the seed's looped config
(chunk 16384, w32) is ~2x SLOWER than tc-default (98304: 955 vs 566; 128256: 2711 vs 1403; 256000: 1364 vs
743). So looped IS correct but the chunk/warps are wrong. The quick oracle also stayed ~2x off tc (under-
explored). OPEN: what looped config matches tc at wide V? Candidates: different chunk (bigger/smaller than
16384), different warps, eviction, num_stages, or tc uses a 2-pass/online strategy the standard kernel can't.
  -> the SUSPECT verdict at V>=98304 is NOT resolved by persist (persist is worse). Need: (a) a looped-chunk
     /warps A/B sweep, and (b) a FULL-effort oracle to see if ANY Helion config matches tc, or if it's a
     source ceiling on the 2-pass kernel (would need cross_entropy_online, run-2's variant -- but that's a
     SOURCE change, separate from the seed). Read tc's generated Triton (TORCH_LOGS=output_code) for strategy.

NET CE PLAN: (P1) is a clean seedable win -- raise/replace the persist cap so the ~50K boundary stays
persistent. (P2) is a looped-param tuning problem + possible source ceiling -- needs more digging before any
claim. DO NOT just delete the cap (that regresses P2's wide V by 2-4x).

## 2026-06-03 — CE crossover PINNED + M-invariant (finer sweep) -> the P1 edit

Fine V-sweep at M∈{2048,4096,8192} (run3_ce_persist_ab.py, no autotuner, median-of-7, spreads ~0.00-0.03;
logs ce_crossover_fine.out + ce_crossover_M.out). persist(rl=[None],w32) / seed_looped(rl=[16384],w32):

| V | actual row KiB | M=2048 | M=4096 | M=8192 | verdict |
|---|---|---|---|---|---|
| 50304 | 196 |  -    | 1.56 |  -   | PERSIST |
| 57344 | 224 | 1.11 | 1.09 | 1.08 | PERSIST (all M) |
| 65536 | 256 | 0.92 | 0.89 | 0.88 | LOOPED (all M) |
| 73728 | 288 |  -   | 0.98 | 0.98 | ~tie (looped marginal) |
| 81920 | 320 |  -   | 0.70 |  -   | LOOPED |

CROSSOVER is between 224KiB (persist wins ~9-11%) and 256KiB (looped wins ~9-12%), IDENTICAL at M=2048/4096/
8192 -> a PER-PROGRAM working-set property (one row per program; M only scales the grid), NOT M-dependent.
Physical WHY: persistent holds the whole valid row resident across the 2-pass (amax then exp-sum) re-read;
it wins until the resident row + working set spills (~256KiB), then a looped chunk that streams wins. The
current cap `MULTILOAD_PERSIST_MAX_BYTES=131072` (128KiB, keyed on fact.size_hint*itemsize = ACTUAL row
bytes, triton.py:463-465) is ~1.75x too LOW -> it loops V>=32769 (incl. 49152/50304/57344 where persist wins).

THE P1 EDIT (planned): raise MULTILOAD_PERSIST_MAX_BYTES 131072 -> 229376 (224KiB, the last confirmed
persist-win row size; conservative -- below the 256KiB looped-win so no wide-V regression). Keeps persist for
actual row <=224KiB, loops above. Recovers CE (4096,50304) 1.56x, (8192,50257) 1.68x, (8192,57344) 1.08x,
(2048,57344) 1.11x; no-regression at V>=65536 (stays looped) and at rms_norm/layer_norm (rnumel<=64KiB <<
224KiB, byte-identical). The warps are ALREADY correct (seed picks w32 at rnumel>16384; persist w32 beat w16
at 50304: 283 vs 310). So the ONLY lever is the cap value -- a single principled constant, matched-lever clean.
OPEN before commit: (a) code-investigator on whether num_load>=2 is a hacky re-read-count proxy (may re-key
the cap on a re-read/distinct-operand fact instead -- fact-integrity); (b) the threshold's generality is
CE-only so far -- the held-out CE shapes + transfer kernels will test it; (c) P2 (wide-V looped params) is
SEPARATE and unresolved -- this edit does NOT touch it.

### Tried / rejected
- **REJECTED: "delete/raise the MULTILOAD cap so all CE goes persistent."** A/B shows forced-persist at
  V>=98304 is 2-4x SLOWER than the looped seed (98304 persist 1837us vs loop 955us; 256000 persist 5011 vs
  loop 1364). The cap's looped path is CORRECT at wide V; only the boundary (~50K) is mis-capped. (Matched-
  lever A/B, median-of-7, run3_ce_persist_ab.json.)
- **REJECTED (for wide V): "the cap is the whole CE problem."** Two independent problems: cap-too-early at
  ~50K (seedable, persist wins) AND looped-params-wrong at >=98304 (looped right, but seed 2x off tc).

### Open hypotheses (to test against the FRESH oracle in Phase 2)
- H-small-MN: small-M and small-N shapes have a seedable warps/M-block gap (perf-dig hint, reps=1 — confirm).
- H-nonpow2: non-pow2-N shapes have a seedable blocking gap (perf-dig hint — confirm).
- H-singlerow: M=1 huge-N may be a real source ceiling (oracle≈seed<tc) OR a seedable gap — oracle decides.
  (NOTE: M=1 huge-N lives in `robustness` = correctness-only; not a train perf target. The train analog is
  tiny-M-large-M variation rows — check what train actually covers.)
- H-welford-wideN: welford wide-N looped-apply ceiling claim — re-litigate vs fresh oracle (was OOM, suspect).

## 2026-06-03 — EDIT #1: CE persist cap MULTILOAD_PERSIST_MAX_BYTES 131072 -> 245760 (240KiB)

CHANGE: `helion/_compiler/autotuner_heuristics/triton.py` constant only (NOT a ReductionFact change; the
`num_load>=2` gate is unchanged). Rewrote the justifying comment with the run-3 A/B grid.

VERIFIED (HELION_AUTOTUNE_EFFORT=none, median-of-7, correctness-gated):
- CE seed codegen now: V in {32000,49152,50257,50304,57344} -> PERSISTENT; V in {65536,98304,128256} -> LOOPED.
- Floor recoveries (G=tc/seed): (4096,50304) 0.682->1.075; (8192,50257) 0.651->1.092; (8192,49152) 1.032->
  1.074; (8192,57344) (was looped ~0.89-equiv) -> persistent G=0.962 (PASSES floor >=0.95, net improvement).
- Wide V UNCHANGED: (8192,65536) G=0.907, (4096,98304) 0.593, (8192,128256) 0.518 (P2 untouched, as intended).
- Correctness: maxerr <= 9.5e-7 on every checked shape.
- NO-REGRESSION (structural): ALL 8 non-CE kernels (rms/ln/softmax/welford/sum/long_sum/kl/jsd) byte-IDENTICAL
  codegen across their full train splits (verified by re-emitting every seed and diffing codegen vs the
  committed floor_sweep_merged.json -> zero changes). The cap only fires for num_load>=2 AND row>240KiB; the
  only num_load>=2 kernel with rows in the 128-240KiB flip-zone is cross_entropy.

STALENESS: this edits triton.py -> source_hash changes for ALL kernels -> the quick oracle cache (batch1) is
now STALE. For the 8 non-CE kernels the SEED is byte-identical so their oracle is effectively unchanged, but
the hash moved; for CE the seed changed materially. Re-measuring seed/oracle on the changed CE shapes next
(fresh source_hash) to confirm VICTORY (new persistent seed ≈ oracle). Flagging the hash move to the hub
(ledger-keeper guards staleness).

## 2026-06-03 — EDIT#1 seed/oracle re-measure (fresh source_hash) — 3/4 at oracle PARITY

Re-ran the oracle (quick) on the 4 changed CE shapes with the NEW seed + NEW source_hash (cache updated):

| shape | OLD seed/orcl | NEW seed/orcl | G_floor | orcl/tc | oracle codegen | verdict |
|---|---|---|---|---|---|---|
| (8192,49152) | 1.025 | **0.996** | 1.054 | 1.050 | persistent | VICTORY |
| (4096,50304) | 1.576 | **1.000** | 1.057 | 1.056 | persistent | VICTORY (closed 58%->0) |
| (8192,50257) | 1.066 | **1.000** | 1.085 | 1.086 | persistent | VICTORY |
| (8192,57344) | -     | **1.105** | 0.946 | 1.046 | LOOPED [32768] | residual (P2 bleed) |

EDIT#1 is a clean incremental win: 3/4 CE boundary shapes now at EXACT oracle parity (0.996-1.000), oracle
beats tc on all 4 (1.05-1.09) confirming SEEDABLE not source-ceiling. Headline (4096,50304) closed 1.576->1.000.

REFINED UNDERSTANDING at V=57344 (the 224KiB crossover edge): the oracle there is NOT persistent -- it's
LOOPED with chunk=32768 (612us) beating my persist seed (676us) AND the old loop-16384 (729us) AND tc (640).
So the residual at 57344 is the SAME issue as P2: the LOOPED CHUNK (fixed 16384) is too small for wide rows;
the oracle scales it up (32768 at V=57344). My persist-cap edit still improved 57344 (676 persist < 729
old-loop16384) and is byte-identical elsewhere, but 57344's TRUE best is a bigger looped chunk. Honest: 57344
is NOT done (seed/oracle=1.105, G_floor 0.946 just under floor -- net up but not at oracle).

=> P2 sharpened: the looped chunk LOOPED_CHUNK=16384 is a fixed constant; the oracle wants it to SCALE with
the row (32768 at V=57344, and likely bigger / different at 98304+). NEXT LEVER: scale the looped chunk (and
re-check warps/eviction/num_stages on the looped path) -- this should close 57344 AND the wide-V P2 shapes
(98304/128256/256000) together. Will A/B looped-chunk {16384,32768,65536,131072} x warps on the wide CE shapes
+ read tc's Triton (TORCH_LOGS) to see if tc uses a fundamentally different (2-pass/split) strategy.

### Current champion
- Run-2 `TritonReductionHeuristic` + EDIT#1 (CE persist cap 131072->245760, dab1eea8). 3/4 CE boundary shapes
  at oracle parity, 8 non-CE kernels byte-identical (no-regression), correctness clean. Pending gate pipeline
  (auditor + results-referee; fact-integrity N/A; anti-giving-up re-litigates run-2 "wide-CE source ceiling").
  Open: CE(8192,57344) residual + wide-V P2 (looped-chunk scaling) -- next lever.
