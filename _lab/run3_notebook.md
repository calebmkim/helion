# RUN 3 — Lab Notebook (persistent worker; source of truth)

> The hill-climb for the Helion forward-inner-reduction Triton **seed** heuristic (H100/sm_90, fp32).
> This notebook — NOT my context — is the source of truth. A fresh worker must be able to continue
> losslessly from here. Every iteration: decision + empirical WHY + tried-and-rejected + open hypotheses
> + current champion.

## RESOLVED CONFOUNDS / HUB DIRECTIVES (2026-06-03)

- **CE kernel identity (the run-2 "source ceiling closed" confound): RESOLVED.** The harness benchmarks the
  STANDARD `cross_entropy` (run2_measure_g.py:65 `from examples.cross_entropy import cross_entropy`;
  KERNELS["cross_entropy"]=(cross_entropy,...)), NOT `cross_entropy_online`. So all my floor + oracle CE
  results are on the standard 2-pass kernel. Run-2 "closed the wide-CE source ceiling via cross_entropy_online"
  — a DIFFERENT kernel `train` does not measure — so the closure never applied to the measured kernel = a
  measuring-the-wrong-thing artifact the geomean hid. My finding stands: standard CE wide-V is SEEDABLE (full
  oracle 588us within 5% of tc; looped seed 2x off) — fix = re-branch/re-tune the SEED, NOT accept a ceiling.
- **Oracle-cache KEY recipe CORRECTED (ledger-keeper guardian).** DROP the heuristic from the key — the oracle
  is what the autotuner SEARCH finds, independent of the seed; keying on the heuristic would invalidate every
  cached oracle on every heuristic edit. New recipe (run3_oracle.py source_hash, applied):
  `sha256(read(examples/<kernel>.py) + to_triton_code(DEFAULT_config for shape) + repr(config_spec knobs+ranges))`
  — per-(kernel,shape); EXCLUDES heuristic/seed code. Safety net: victory-confirm ALWAYS re-runs a FRESH FULL
  oracle. NOTE: batch-1 cached entries carry the OLD-recipe hash (oracle DATA still valid — no kernel-source
  edit yet; only the key string is stale) — re-stamp on next oracle run; flagged to ledger-keeper.
- **Quick oracle can UNDERSHOOT (fake parity).** Never declare DONE on a quick oracle — "done" needs seed
  within ε of a FULL/fresh oracle. (Live: quick CE(4096,98304) looked ~tc but FULL found a 1.62x-better config.)

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

## 2026-06-03 — P2 CE looped-chunk A/B (wide V) — looped-chunk is too small + a WIDE-V SOURCE CEILING

`run3_ce_loopchunk_ab.py`: looped chunk {16384,32768,65536,131072} x warps {16,32} + persist control, vs tc,
median-of-7, correctness-gated. Logs ce_loopchunk_ab.out/json. Findings:

| V | row KiB | best looped | best/tc | seed loop16384_w32 /tc | persist_w32 /tc |
|---|---|---|---|---|---|
| 57344  | 224  | loop65536_w16: 610us | **1.064** | 0.891 | 0.965 |
| 98304  | 384  | loop32768_w32: 876us | 0.646 | 0.592 | 0.299 |
| 128256 | 501  | loop32768_w16: 2614us| 0.537 | 0.517 | 0.245 |
| 256000 | 1000 | loop32768_w16: 1321us| 0.561 | 0.543 | 0.148 |

TWO conclusions:
1. **The fixed LOOPED_CHUNK=16384 is too small.** chunk 32768 is uniformly better at wide V (+3-8% over 16384:
   98304 0.592->0.646, 128256 0.517->0.537, 256000 0.543->0.561). At the 224KiB edge chunk 65536 (≈whole row,
   one looped iter) WINS and beats tc (1.064) AND beats persistent (0.965) -- so 57344's true best is looped-
   65536, not the persistent my cap edit gives (672us; loop65536=610us, 9% better). Too-big chunks at wide V
   spill catastrophically (131072 at 98304 = 0.233; persist = 0.299).
2. **WIDE-V (>=384KiB) IS A 2-PASS SOURCE CEILING.** NO Helion config -- persistent OR any looped chunk/warps
   in the grid -- beats tc at V>=98304; the BEST is ~0.54-0.65 (tc ~1.5-2x faster). The standard 2-pass
   cross_entropy (amax-pass + exp-sum-pass = reads the row TWICE) is structurally ~2x off tc, which uses a
   fused/online single-pass logsumexp (= exactly examples/cross_entropy.py::cross_entropy_online, run-2's
   variant). This is a kernel-SOURCE limit, NOT a seed failure: it caps the oracle too. MUST CONFIRM with a
   FULL-effort oracle (anti-giving-up: a hand grid isn't exhaustive; the full oracle is the answer key) that
   oracle also can't beat tc before recording "source ceiling". Full oracle on wide CE LAUNCHING.

PLAN: (a) bump LOOPED_CHUNK 16384->32768 (small uniform wide-V gain; MUST verify no-regression on long_sum's
looped >2^20 tail which also uses LOOPED_CHUNK). (b) Confirm wide-V source ceiling via full oracle; if
confirmed, the residual seed/oracle there should be ~1.0 (seed≈oracle<tc) = seed done, source-rewrite is
separate Product-A scope. (c) 57344 edge: loop65536 wins -- consider whether the persist-cap boundary should
instead hand the 224KiB edge to a big looped chunk; but don't over-fit one shape -- the EDIT#1 cap is already
a net win there and 3/4 boundary shapes are at oracle.

## 2026-06-03 — *** FULL-EFFORT ORACLE REVERSES THE WIDE-V "SOURCE CEILING" *** (the pivot)

Ran a FULL-effort oracle on cross_entropy(4096,98304) (autotune 1072s, 11 generations). DECISIVE and it
OVERTURNS BOTH run-2's "wide-CE source ceiling" AND my own intermediate "wide-V is a 2x source ceiling"
(which was an artifact of my coarse chunk-x-warps grid + the quick oracle under-exploring):

**cross_entropy(4096,98304) FULL oracle:**
- oracle_us = 588.4 (looped), tc = 557.0, **oracle/tc = 0.947** (oracle only ~5% off tc, NOT 2x!).
- seed (looped, chunk 16384, w32, pid=flat) = 955.5, **seed/oracle = 1.624** -> 62% SEEDABLE on the table.
- My chunk-x-warps grid best was 876us (chunk 32768) -- the full oracle (588) is FAR better -> my grid missed
  the winning region entirely.

**THE FULL-ORACLE WINNING CONFIG (the real answer key) — a strategy OUTSIDE the seed's design space:**
```
reduction_loops=[4096]   (SMALL chunk -- not bigger! I had the direction wrong)
pid_type='persistent_interleaved'   (NOT 'flat'!)
num_sm_multiplier=32, maxnreg=64     (persistent-pid-only knobs; run-2 said "inapplicable" -- WRONG)
num_stages=4, range_unroll_factors=[4], range_num_stages=[2], range_flattens=[False]  (inner-loop pipelining)
load_eviction_policies=['','','last','first','last'], indexing mostly tensor_descriptor
```

=> The wide-CE win comes from a PERSISTENT-PID + small-chunk + software-pipelined strategy. This DIRECTLY
CHALLENGES run-2's `pid_type='flat'` "principled constant" lock (run-2: "flat dominates 1.5-4x on every
forward reduction; persistent pid only amortizes launch/tail for grid-BOUND backward/Band-D"). For wide CE
(grid-starved: M=4096 rows but each row is huge), a persistent-interleaved grid of 32*SM CTAs that loop over
rows with maxnreg-capped occupancy + pipelined inner loop WINS by 1.62x over the flat seed. run-2's pid lock
was validated on NARROW forward reductions; it is FALSE for wide grid-light multi-load rows.

CONSEQUENCES (re-scoping P2):
- Wide-V CE is HEAVILY SEEDABLE (1.62x), NOT a source ceiling. The residual oracle-vs-tc is only ~5% (the
  genuine 2-pass-vs-single-pass source signal) -- small, and seed≈oracle is the bar, so once the seed matches
  the oracle the 5% is a separate optional source-rewrite (cross_entropy_online).
- The LEVER is NOT "scale the looped chunk" (EDIT#2 candidate is WRONG -- oracle wants SMALL chunk 4096). The
  real levers are pid_type=persistent_interleaved + num_sm_multiplier + maxnreg + num_stages + range pipelining
  -- a cluster the seed currently doesn't touch (it hardcodes pid=flat, num_stages=1, no range/maxnreg/sm_mult).
- This is a MUCH bigger and more general finding than the persist-cap: the whole "flat pid + num_stages=1"
  default may be leaving large gains on grid-light wide reductions across kernels. MUST matched-lever A/B to
  isolate which lever(s) carry the 1.62x (is it the pid, the chunk, the pipelining, or maxnreg?), and check
  generality (does persistent_interleaved help other wide/grid-light shapes, or is it CE-specific?).

KILLED the remaining 2 full-oracle shapes (128256/256000) -- shape-1's answer key is decisive and the strategy
family will be the same; full effort is ~18min/shape (too expensive to babysit 3). Re-litigates run-2 cleanly:
the wide-CE "source ceiling" verdict was WRONG -- a real Helion config beats the seed 1.62x and nearly matches
tc; run-2 missed it because (a) it locked pid=flat and (b) it declared a source ceiling without a fresh full
oracle reading the answer key (anti-giving-up failure mode #7).

### NEXT (re-scoped): isolate the wide-CE oracle levers
1. Matched-lever A/B on CE(4096,98304) + (8192,128256): start from the seed, add ONE oracle lever at a time
   (pid=persistent_interleaved+sm_mult; then +small-chunk 4096; then +num_stages 4; then +maxnreg 64; then
   +range pipelining), bench each vs the full-VERBATIM oracle baseline (588us) -> which lever(s) carry the win.
2. Then decide the heuristic: a workload-keyed branch (grid-light wide multi-load -> persistent_interleaved +
   pipelined small-chunk). Needs a fact for "grid-light" (few rows relative to SM count?) -- ask code-investigator
   whether m_block_ids / grid extent is available as a fact. fact-integrity will scrutinize any new fact.
3. Re-check pid='flat' generality: does persistent_interleaved help the OTHER wide shapes (softmax wide,
   welford wide, long_sum)? run-2's flat lock must be re-litigated, not assumed.

## 2026-06-03 — wide-CE LEVER ISOLATION (ablation from VERBATIM oracle) — eviction + pid carry it

METHOD NOTE (important): my first lever A/B RECONSTRUCTED the oracle from a subset of fields (chunk+pid+
pipeline) and got 1005us != the real 589us -- because it OMITTED indexing + load_eviction_policies, which ARE
part of the coupled winning bundle ("oracle is a bundle" trap). FIX: re-bench the VERBATIM cached oracle
config (589.2us, reproduces the cached 588.4us) and ablate ONE lever group at a time FROM it. run3_ce_pid_ab.py
updated to load the verbatim cached oracle + ablate.

CE(4096,98304) ablation (tc=566us, seed=956us, verbatim oracle=589.6us=1.62x seed; median-of-7):

ADDITIVE (seed + one oracle lever):           ABLATION (verbatim oracle - one lever):
  seed+eviction  = 732us  (1.31x !!)            oracle-eviction  = 949us  (COLLAPSES to seed -- essential)
  seed+pidcluster= 850us  (1.12x)               oracle-pidcluster= 888us  (loses most -- essential)
  seed+indexing  = 956us  (no change)           oracle-indexing  = 621us  (small: 5% help)
  seed+chunk     =1018us  (WORSE)               oracle-chunk     = 595us  (negligible: chunk 4096~16384 in-bundle)
  seed+pipeline  = 956us  (inert)               oracle-pipeline  = 589us  (ZERO effect -- inert)

LOAD-BEARING LEVERS = **load_eviction_policies + the persistent-pid cluster** (coupled). chunk-size and
pipelining are NEARLY INERT inside the bundle (chunk 4096 vs 16384 barely matters with the right evict+pid).
- eviction `['','','last','first','last']` ALONE = 1.31x. This is the SAME KIND of finding run-2 made for
  WELFORD (re-read -> 'last' on the kept-resident load). CE is a 2-PASS RE-READ kernel (amax-pass then
  exp-sum-pass re-read the logits row), so the row load wants 'last' (keep L2-resident for the re-read pass).
  RUN-2 EXPLICITLY left CE eviction at default ("cross_entropy eviction-neutral ... its gap is SOURCE ceiling
  not eviction") -- WRONG: at wide V, CE eviction is a 1.31x win. run-2's eviction analysis was on NARROW CE
  (where the row fits and re-read is cheap); at wide V the re-read eviction matters a lot.
- pidcluster (persistent_interleaved + num_sm_multiplier + maxnreg) adds the rest -> 1.62x together.

=> The wide-CE fix is NOT chunk size and NOT a source ceiling. It's (1) APPLY THE RE-READ EVICTION to CE (the
existing `_eviction_policies(env,"reread")` recipe is for is_structured_combine only; CE needs a re-read fact
to qualify -- connects to the num_load/re-read provenance question I asked code-investigator), and (2) a
persistent-interleaved pid + sm_mult/maxnreg branch for grid-light wide multi-load. Lever (1) (eviction) is
the cleaner, more principled first step (re-read is a real workload property; the welford recipe already
exists). Lever (2) (pid) is bigger structurally but re-opens run-2's pid='flat' lock -> heavier gate scrutiny.

NEXT: (a) map the CE eviction slots (which load is the re-read row) via generated Triton; confirm the
welford-style reread policy (or the oracle's exact `['','','last','first','last']`) is what helps and WHY.
(b) Check if the re-read property is recoverable from provenance (code-investigator) -> a principled
`is_reread`/`num_row_passes` fact gating the reread eviction for CE (and generalizing). (c) Then the pid
cluster. A/B each vs the verbatim oracle, correctness-gate, no-regression.

### Tried / rejected (updated)
- **REJECTED: EDIT#2 "bump LOOPED_CHUNK 16384->32768".** The full oracle wants a SMALLER chunk (4096) +
  persistent-pid, not a bigger chunk. My coarse grid (16384-131072) suggested 32768 but the full oracle (588us
  @ chunk 4096) crushes the best grid arm (876us @ 32768). Chunk size alone is the wrong lever; the win is the
  pid/pipelining cluster. (Do NOT make EDIT#2 as conceived.)
- **REJECTED: "wide-V CE is a ~2x source ceiling."** Full oracle is only 5% off tc (0.947), not 2x. My grid +
  the quick oracle under-explored; the full oracle found a 1.62x-better config. (Anti-giving-up: a hand grid is
  not an oracle; run the full search before claiming a ceiling.)

## 2026-06-03 — EDIT#1 GATED: fact-integrity FAIL on num_load proxy -> EDIT#2 re-key onto num_reduction_ops

Hub gated EDIT#1 (ledger run3.gate_verdicts): results-referee PASS (reproduced the 3 parity deltas + floor),
adversarial-auditor PASS-with-flag, **fact-integrity FAIL** on the GATE (not the value). Finding: `num_load`
is a syntactic hl.load FX-node count (device_ir.py:1108-1121), NOT a faithful re-read/resident property —
over-counts CE (scalar gather + row), style-dependent (cross_entropy vs _online differ), misclassifies. The
cap VALUE (240KiB) endorsed. Hub: re-key onto a faithful property (num_reduction_ops or resident-operand-
bytes), NOT bytes-alone (bytes-alone loops long_sum's 11/12 >240KiB nro=1 rows = regression). Verify flip-set
(softmax wide = prime risk). Disposition: EDIT#1 not accepted standalone; fold in EDIT#2, combined re-gate.

**EDIT#2: gate `num_load >= 2` -> `num_reduction_ops >= 2`** (helion/_compiler/autotuner_heuristics/triton.py).
num_reduction_ops = count of reduction lowerings over the rdim = number of PASSES over the row (device_ir.py
:1124-1129, already computed). The faithful "re-reads the row (>=2 passes)" property. Fact values (verified):
  CE nro=2, softmax nro=2, layer_norm nro=2, jsd nro=2 | long_sum nro=1, sum nro=1, rms_norm nro=1, kl_div nro=1.

FLIP-SET (empirical scan of ALL train shapes):
- vs ORIGINAL champion (num_load>=2, 128KiB): exactly 3 flips = CE(49152/50257/50304) looped->persistent
  (the intended targets). NO softmax flip (wide softmax >240KiB stays looped under both). NO long_sum flip
  (nro=1 excludes it -> stays persistent, killing the bytes-alone regression). NO other kernel.
- gate re-key ALONE (num_load,240KiB -> nro,240KiB): ZERO flips = byte-identical (the re-key is a pure
  faithfulness fix; rms_norm/kl_div drop out of the gate via nro=1 but their rows are <<240KiB so no change).

VERIFIED (HELION_AUTOTUNE_EFFORT=none): CE boundary persistent+correct (G 1.07-1.09); CE wide still looped;
ALL 8 non-CE kernels byte-IDENTICAL codegen vs committed floor. Lint+format clean.

EDIT#2 makes the cap principled: gates on the real re-read-pass property; long_sum stays persistent BECAUSE
nro=1 (not a byte coincidence); CE/softmax capped because they genuinely re-read. num_reduction_ops is ALSO
the right signal for the re-read EVICTION (next), so one faithful fact does double duty.

## 2026-06-03 — CE RE-READ EVICTION A/B — principled policy MATCHES the oracle, 1.09-1.31x on wide looped CE

`run3_ce_evict_ab.py` (median-of-7, correctness-gated). The looped CE seed's loads (generated Triton):
[0] labels_tile, [1] logits_at_target (scalar gather), [2] logits_rows (amax pass), [3] logits_rows_1
(exp-sum pass = the RE-READ of the row). Eviction list length=5 (a 5th codegen slot beyond the 4 tl.load).

| shape | codegen | default | reread_rowlast ['','','last','first',''] | vs_default | arm/tc | all_first | all_last |
|---|---|---|---|---|---|---|---|
| (4096,50304) | persistent | 282.4 | 282.2 | 1.001 | 1.076 | 284.3 | 282.2 |
| (4096,98304) | looped | 956.7 | **732.4** | **1.306** | 0.772 | 959.6 | 976.3 |
| (8192,128256)| looped | 2714.7| **2284.9**| **1.188** | 0.614 | 2608.7 | 2703.4 |
| (2048,256000)| looped | 1364.8| **1257.2**| **1.086** | 0.590 | 1367.3 | 1372.8 |

FINDINGS:
- The PRINCIPLED policy `reread_rowlast = ['','','last','first','']` (keep the amax-pass row load slot[2]
  L2-resident 'last' for the re-read; stream the exp-sum re-read slot[3] final-use 'first'; default elsewhere)
  MATCHES the oracle's `['','','last','first','last']` to within noise (732.4 vs 732.6). The oracle's slot[4]
  'last' is a PASSENGER (no effect). So the win is specifically slot[2]='last' + slot[3]='first' = the welford
  "reread" logic applied to CE's re-read row.
- Eviction is a CLEAN WIN on the LOOPED wide-V shapes: 1.306x@98304, 1.188x@128256, 1.086x@256000 (monotone
  down with V). NEUTRAL on the PERSISTENT boundary (50304: 1.001 — persistent holds the row in registers, no
  HBM re-read to optimize). So eviction targets exactly the looped wide regime = the remaining floor losses.
- all_first / all_last do NOT help (confirms it's the specific re-read slot policy, not a blanket).
- Eviction (1.09-1.31x) closes PART of the wide-V gap (looped seed+evict reaches arm/tc 0.59-0.77, still below
  tc); the REST to the full oracle (588us@98304) is the persistent-PID cluster. evict + pid ~= the 1.62x oracle.
- This OVERTURNS run-2's "cross_entropy eviction-neutral" claim (run2_notebook L126): at WIDE V (looped),
  CE re-read eviction is 1.1-1.3x. run-2 measured CE eviction only at NARROW V where the row fits / re-read is
  cheap. CONFIRMS: CE is a re-read kernel (nro=2), and the re-read eviction matters once the row is looped.

EDIT#3 (planned): apply a RE-READ eviction policy to nro>=2 T1 reductions. DESIGN ISSUE: the re-read row is
slots[2,3] for CE (NOT slot[0] like welford's combine). The existing `_eviction_policies(env,"reread")` =
`['last']+['first']*(n-1)` hardcodes slot[0]='last' — wrong for CE. A FAITHFUL policy must put 'last' on the
load that is RE-READ (the row's first pass) — which needs per-slot re-read PROVENANCE (which loads resolve to
the same host buffer across passes — `_reduction_fx_inter_loop_rw_names`). ASKING code-investigator whether
per-slot re-read provenance is available, so the eviction fact is faithful (not a hardcoded slot index =
another proxy/style-dependent hack the fact-integrity gate would reject). HOLDING EDIT#3 until that answer.

## 2026-06-03 — welford apply/combine/warps A/B (independent thread; data-gathering)

`run3_wf_tile_ab.py` (median-of-7, fp32 asserted). block_sizes=[M_block, combine, apply]:
- **(4096,16384)** seed [1,8192,2048] w16 = 214.7us (arm/tc 0.861). BEST = **[1,16384,4096] = 197.1us (1.089x,
  arm/tc 0.937)** -- matches oracle. combine 8192->16384 AND apply 2048->4096 both help. applyNp2(16384)=254
  (too big, WORSE).
- **(32768,8192)** seed [2,8192,2048] w16 = 787.4us. BEST = **[2,8192,4096] w32 = 713.8us (1.103x, arm/tc
  0.988)** -- matches oracle. apply->4096 AND warps->32 TOGETHER (warps32 alone=912 WORSE; only helps with the
  bigger apply).
- **(16384,768)** seed [1,1024,1024] w4 = 40.2us (arm/tc 0.983 -- ALREADY at parity). Nothing helps; warps_x2
  hurts. The quick-oracle "w1, so=1.032" did NOT reproduce -> that gap was quick-oracle NOISE; shape is done.

SYNTHESIS: the welford APPLY tile is over-capped -- `STRUCTURED_APPLY_LOOP_CHUNK_BYTES=8192` (2048 fp32) wants
to be 16384 (4096 fp32). Combine cap `STRUCTURED_COMBINE_CAP_BYTES=32768` (8192 fp32) slightly low at
(4096,16384) (wants 16384). Wide-M (32768,8192) also wants warps 16->32. Gains are MODEST (1.05-1.10x) but
real, oracle-matched. EDIT#4 candidate: raise apply cap 8192->16384 (+ maybe combine, + a warps tweak) -- BUT
must no-regression-check the other welford in-sample-v2 + train shapes (run-2 tuned these caps). HELD with EDIT#3
until the CE gates settle (avoid overwhelming the gate queue with parallel uncommitted edits).

## CONSOLIDATED PER-SHAPE seed/oracle TABLE (task #5 running tally; quick oracle unless noted)

WORST-FLOOR / SEEDABLE (the worklist), with the identified lever:
| shape | seed/oracle | oracle/tc | status | lever |
|---|---|---|---|---|
| CE(4096,50304) | 1.576->**1.000** | 1.07 | FIXED by EDIT#1+2 | persist cap (re-keyed nro>=2) |
| CE(8192,50257) | 1.066->**1.000** | 1.09 | FIXED | persist cap |
| CE(8192,49152) | 1.025->**0.996** | 1.05 | FIXED | persist cap |
| CE(4096,98304) | 1.62 (full) | 0.95 | OPEN | re-read evict 1.31x (EDIT#3) + persistent-pid cluster |
| CE(8192,128256)| 1.25 (quick) | 0.64q | OPEN | evict 1.19x + pid (full oracle not yet; quick under-explores) |
| CE(2048,256000)| 1.13 (quick) | 0.62q | OPEN | evict 1.09x + pid |
| jsd(8192,30522)| 1.196 | 1.01 | OPEN | Band-B (not yet dug) |
| welford(4096,16384)| 1.146 | 1.00 | OPEN | apply 2048->4096 + combine (EDIT#4, A/B done) |
| softmax(131072,256)| 1.147 | 1.00 | OPEN | small-N (not yet dug) |
| welford(32768,8192)| 1.089 | 0.99 | OPEN | apply->4096 + w32 (EDIT#4) |
| welford(16384,768)| 1.032 | 0.94 | DONE (noise) | none -- at parity, quick-oracle gap was noise |
SOURCE-LIMIT candidates (seed≈oracle<tc): long_sum(16,2097152) seed/oracle=0.998 (N>2^20 structural;
split-K source opportunity, NOT seed work) -- verify oracle real at full effort before recording.
AT ORACLE (victory, from batch1): CE(8192,32000) 0.993.

## 2026-06-03 — jsd Band-B + softmax small-N warps A/B (board-completion; data-gathering)

`run3_warps_ab.py` (median-of-7, correctness-gated):

**jsd narrow-V (Band-B) -- the lever is num_warps 32->16 (NOT R_BLOCK):**
- (8192,30522): w16 = 1.195x (670us, beats tc 1.012); best rb2048_w16=1.213x. R_BLOCK 1024/2048/4096 barely
  differ at fixed w16.
- (8192,32000): w16 = 1.122x (beats tc 1.023). Same.
- => Band-B jsd wants w16 at narrow V, but the rnumel ramp gives w32 (rnumel>16384). WHY: Band-B carries 2D
  [M_BLOCK,R_BLOCK] accumulators (register-heavy via num_tiled_accumulators>=1); w32 over-subscribes registers.
  The warps ramp should be LOWER for Band-B (num_tiled_accumulators>=1). EDIT#5 candidate.

**softmax small-N -- warps lever is M/OCCUPANCY-dependent, NOT a clean rnumel rule (overfitting trap):**
- (131072,256) HIGH-M: w8=1.213x (best, tc 0.967); seed w4 too FEW.
- (262144,128) HIGH-M: w16=1.097x, w8=1.088x; seed w4 too few.
- (16384,512) LOWER-M: w8=0.834x WORSE, w16=0.514x CATASTROPHIC; seed w4 is RIGHT.
- => NOT "small-N wants more warps". HIGH-M small-N (grid-saturated, each program tiny) wants MORE warps to
  fill the SM; LOWER-M small-N wants w4. The distinguishing property is GRID OCCUPANCY (M*ceil-rows vs SM
  count / total programs), NOT rnumel. The current ramp keys on rnumel ALONE -> can't separate (131072,256)
  [w8] from (16384,512) [w4] (both rnumel<=512->w4). A clean fix needs a grid-occupancy fact -- the SAME
  "grid-light vs grid-heavy" theme as the CE pid cluster (persistent_interleaved for grid-light). DANGER: a raw
  M threshold fences shapes = identity-smuggling risk; the principled property is occupancy (programs vs SMs).
  HARDER lever; needs an occupancy fact + careful generality. Do NOT rush.

BOARD NOW FULLY MAPPED. Lever taxonomy:
- CLEAN/PRINCIPLED, ready: persist-cap re-key (EDIT#1+2, committed), CE re-read eviction (EDIT#3, pending
  provenance), welford apply-cap (EDIT#4).
- COUPLED/HARDER (warps-vs-occupancy, pid -- need fact enrichment + generality care): CE persistent-pid
  cluster, jsd Band-B warps (num_tiled_accumulators-keyed warps), softmax small-N warps (occupancy fact).
- SOURCE-LIMIT candidate: long_sum(16,2097152) (verify full oracle).
- DONE/at-parity: most of the curriculum; welford(16384,768) (quick-oracle gap was noise).

## 2026-06-03 — EDIT#2 REJECTED by hub-prescreen (num_reduction_ops UNDER-counts) -> revert + row_reread

GATE CAUGHT A REAL BUG IN MY EDIT#2. EDIT#1 gate verdicts: referee PASS, auditor PASS-with-flag,
fact-integrity FAIL on num_load proxy. I "fixed" it (EDIT#2) by re-keying to num_reduction_ops>=2. Hub-prescreen
REJECTED EDIT#2: num_reduction_ops is ALSO a proxy — it UNDER-counts (mirror of num_load's over-count).
rms_norm/layer_norm RE-READ the row in a POST-REDUCTION APPLY pass (normalize y=x*rstd*w), which is NOT a
ReductionLowering, so num_reduction_ops==1 for them. nro>=2 would EXEMPT their wide rows -> persistent ->
~2.9x spill (run-2's own P/L table: rms P/L=2.91 @512KiB). **MY ERROR: my flip-set scan was TRAIN-ONLY**
(rms train maxes 64KiB < cap), so I missed that rms_norm/layer_norm ROBUSTNESS (1,131072)=512KiB flip
looped->persistent under nro>=2. LESSON (logged by hub): **flip-set scans MUST cover train+val+test+ROBUSTNESS**
— the cap fires on byte-width, and the widest rows live in robustness.

REVERTED EDIT#2 (commit 1cb50a6a): gate back to num_load>=2 (the LESS-WRONG placeholder; it fires for
{rms,ln,softmax,CE} = the RIGHT set by luck of over-counting, vs nro>=2 = {CE,softmax,ln,jsd} which wrongly
exempts rms). EDIT#1's 240KiB value stays banked. VERIFIED: CE boundary persistent (G 1.07-1.09); rms_norm/
layer_norm(1,131072)=512KiB correctly LOOPED (G 1.34/1.42).

**THE FAITHFUL FACT (hub-directed, task #8 EDIT-GATE-v2): `row_reread` boolean** = is the reduction-input HOST
BUFFER read in >1 distinct pass/region (a 2nd reduction pass OR a post-reduction apply re-read)? From the
reduction roller's provenance (_reduction_fx_inter_loop_rw_names / _fx_trace_tensor_arg_rw_names, device_ir.py
~608-703 — each hl.load's tensor arg resolves to host buffer names, device temporaries excluded). RIGHT SET:
sum/long_sum=False (single-stream -> exempt, persist to structural cap); rms_norm/layer_norm/softmax/CE=True
(re-read -> governed by the byte cap); welford Band-C + kl/jsd Band-B have their own caps that dominate.
BONUS: row_reread ALSO de-hacks the reread load-eviction (currently is_structured_combine-only) to CE — one
faithful fact, double duty (matches my independent eviction finding).

PROVENANCE DESIGN QUESTION (asked code-investigator #3, sharpened): for a T1 rollable reduction, is the APPLY
pass (rms_norm's re-read) IN the candidate graph set `_count_reduction_workload` iterates, or a separate graph?
Can I compute row_reread by counting, per resolved host-buffer-name, the # of distinct load nodes that read it
across the reduction+apply region (>1 => re-read)? Need the exact graph topology before writing the fact — NOT
guessing a third time (gate caught num_load over-count + nro under-count already). HOLDING EDIT#3/GATE-v2 until
the topology lands.

## 2026-06-03 — row_reread discriminator: EMPIRICAL probe (run3_row_reread_probe.py) — T1 SOLVED, T2 open

Probed the host-buffer read provenance directly (fake-tensor trace, no do_bench) to find the FAITHFUL
discriminator. Falsified two naive ones, then found the T1 answer:
- **load-NODE count >=2 per buffer: WRONG.** sum/long_sum x=2 nodes (false positive) — a single looped stream
  emits 2 load nodes for x.
- **distinct-GRAPH count >=2: WRONG.** sum/long_sum x in 2 graphs too (RootGraphInfo + its ReductionLoop-
  GraphInfo are alternate representations of ONE pass).
- **GRAPH STRUCTURE (the key, dumped): each rolled reduction = {RootGraphInfo + ReductionLoopGraphInfo(s)}.**
  sum: root + 1 ReductionLoopGraphInfo, both load x (= 1 logical pass). rms_norm: root + graph[1] Reduction-
  Loop(sum x^2) + **graph[2] ReductionLoopGraphInfo with n_reduction_lowerings=0 that RE-loads x = the APPLY
  pass**. CE: root + graph[1] ReductionLoop(amax) + graph[2] ReductionLoop(expsum), both load logits.
- **WORKING T1 DISCRIMINATOR: a host buffer is loaded in >=2 distinct ReductionLoopGraphInfo graphs.** Verified
  RIGHT for 6/7: sum/long_sum x=1 graph->False; rms_norm x=2->True; layer_norm x=3->True; CE logits=2->True;
  kl_div {}->False. **ONLY softmax wrong** (predicted False) — because softmax is **T2 (user-tiled)**: it has
  NO ReductionLoopGraphInfo graphs ({}); its max-pass + exp-sum-pass re-read live in the user's explicit
  hl.tile loops, not roller graphs. So the T1 discriminator is solid; T2 (softmax=reread True, kl_div=False,
  jsd=?) needs the analogous "buffer loaded across >=2 user-tile passes" detection — DIFFERENT graph topology.

=> row_reread for T1 = "reduction-input host buffer loaded in >=2 ReductionLoopGraphInfo graphs" (faithful:
tracks genuine multi-pass re-read, immune to the roller's root+loop duplication and to load-op-count style).
For T2 I need the code-investigator's read of `register_user_tiled_reductions` graph structure (softmax must be
True, kl_div False). ASKED (sharpened with this empirical data). HOLDING the fact write until the T2 half is
nailed — T1 is empirically proven, T2 is the last gap.

## 2026-06-03 — *** row_reread fact SOLVED + verified 9/9 (the EDIT-GATE-v2 substrate) ***

Dumped the T2 graph structure myself (didn't wait on the investigator) and UNIFIED T1+T2 into one faithful
discriminator. T2 structure: softmax = 2 ForLoopGraphInfo loading x (max+sum pass, n_red=2; + apply pass
n_red=0) -> re-read; kl_div = 1 ForLoopGraphInfo loading y_pred+y_true (each once) -> not.

**FAITHFUL row_reread (run3_row_reread_probe.py, verified 9/9 kernels):**
`row_reread = (some reduction-input HOST BUFFER is loaded in >=2 distinct LOOP graphs, where a LOOP graph is
ReductionLoopGraphInfo (T1-rollable) OR ForLoopGraphInfo (T2 user-tiled))`.
Per-buffer loop-graph counts (the proof):
  sum {x:1}=F, long_sum {x:1}=F, rms_norm {x:2}=T, layer_norm {x:3}=T, softmax {x:2}=T,
  cross_entropy {logits:2}=T, kl_div {y_pred:1,y_true:1}=F, jsd {_input:1,target:1}=F, welford {x:2}=T.

WHY FAITHFUL (fact-integrity divergence test PASSES):
- Tracks genuine multi-pass re-read of the SAME host buffer (the real property), via the reduction roller's
  host-buffer read provenance (_fx_trace_tensor_arg_rw_names). NOT num_load (over-counts CE's scalar gather:
  nl=3) NOR num_reduction_ops (under-counts rms/ln apply re-read: nro=1). It gives the RIGHT answer on exactly
  the kernels where those two proxies diverge from the property.
- Immune to the roller's root+loop DUPLICATION: a single-pass stream (sum) = RootGraphInfo + 1 Reduction-
  LoopGraphInfo both loading x, but only 1 is a LOOP graph -> count 1 -> False. (Counting all graphs or load
  nodes false-positived sum at x=2.)
- Style-independent: it's the workload's dataflow (buffer consumed by >=2 passes), not a load-op count.
- Distinguishes 2-distinct-inputs-each-once (kl_div/jsd -> False) from 1-re-read-input (softmax/CE -> True).

CONSUMERS (both, per the hub's task #8): (1) persist-cap gate -> `row_reread` replaces `num_load>=2`
(governed set rms/ln/softmax/CE/welford; exempt sum/long_sum -> persistent to structural cap; kl_div/jsd
Band-B own-cap dominates). (2) reread load-eviction -> the SAME provenance names WHICH buffer is re-read, so I
can put 'last' on that buffer's first-pass load + 'first' on its re-read final-use (de-hacks run-2's POSITIONAL
welford slot[0]='last' AND generalizes the CE eviction win 1.31x).

NEXT (task #8 execution, now UNBLOCKED — I derived the fact rigorously, no guess): add `row_reread: bool` to
ReductionFact (config_spec.py), compute it in device_ir (the loop-graph discriminator, reusing the existing
provenance — a SMALL targeted addition, not a framework), wire BOTH consumers in triton.py, predict the FULL
4-split flip-set (train+val+test+ROBUSTNESS), correctness-gate, A/B (CE wide eviction + the persist boundary +
no-regression on wide rms/ln robustness staying LOOPED), DM the design to hub before commit -> fact-integrity
+ referee + auditor re-gate. The code-investigator's T2 answer (pending) will corroborate; not blocking on it.

## 2026-06-03 — EDIT-GATE-v2 IMPLEMENTED + verified (row_reread fact wired into the persist cap)

Wrote the faithful fact + first consumer. Changes (3 files):
- `config_spec.py`: ReductionFact gains `row_reread: bool = False` (+ docstring: faithful re-read property,
  why NOT num_load/num_reduction_ops).
- `device_ir.py`: `DeviceIR._compute_row_reread()` = the unified discriminator (buffer in >=2 ReductionLoop/
  ForLoop graphs), wired into BOTH `_build_reduction_fact` (T1) and `register_user_tiled_reductions` (T2).
  Verified computable AT fact-build time (self.graphs already holds the loop graphs; build-time == post-bind).
- `triton.py`: persist-cap GATE `if fact.num_load >= 2` -> `if fact.row_reread` (EDIT-GATE-v2). Comment rewritten.

VERIFICATION (all GREEN):
- PRODUCTION fact.row_reread correct 9/9 (sum/long_sum=F, rms/ln/softmax/CE/welford=T, kl_div/jsd=F).
- FULL 4-split FLIP-SET (train+val+test+ROBUSTNESS, the EDIT#2 lesson): the ONLY cap-gate-decision flips vs
  the num_load placeholder are kl_div/jsd wide-vocab (num_load=2->fires; row_reread=F->doesn't). But those are
  Band-B (num_tiled_accumulators>=1): the BANDB_R_BLOCK cap (4096) clamps R_BLOCK IDENTICALLY whether
  can_persist is T or F -> their SEEDS are BYTE-IDENTICAL (verified: block_sizes=[4096,1], looped, w32, correct
  on kl_div(2048,151936)/(1024,256000), jsd(8192,65536)/(4096,98304)/(2048,256000)). No other kernel flips.
- So EDIT-GATE-v2 is SEED-BYTE-IDENTICAL to the num_load placeholder across the WHOLE curriculum; the 3 CE
  boundary persistent flips are from EDIT#1's cap VALUE (240KiB), not the gate.
- CRUCIALLY avoids the EDIT#2(nro) regression: rms_norm/layer_norm (1,131072)=512KiB ROBUSTNESS stay LOOPED
  (row_reread=True -> governed), G 1.10/1.32, correct. (nro=1 would have exempted them -> 2.9x spill.)
- Correctness maxerr <= ~3e-4 (fp32 reduction-order, within tol) on all checked shapes. Lint+pyrefly clean.
  Tests: test_reductions 34p, test_autotuner_heuristics + subtests, test_autotuner 107p — all PASS.

FACT-INTEGRITY readiness (the gate will check): row_reread computes the REAL property (multi-pass re-read)
from provenance (_fx_trace_tensor_arg_rw_names), passes the divergence test (right on kernels where num_load
over-counts + num_reduction_ops under-counts), is style-independent (dataflow not op-count), has a consumer
(the persist-cap gate; + the reread eviction next). Probe `_lab/harness/run3_row_reread_probe.py` is the
falsification artifact (9/9). NOT a general framework — a single targeted discriminator reusing existing
provenance.

PENDING: DM the design to hub before committing (task #8 protocol), then commit -> fact-integrity + referee +
auditor re-gate. SECOND consumer (de-hack the reread EVICTION onto row_reread + name the re-read buffer's
slot) is a FOLLOW-ON edit (the eviction win 1.31x is the next perf gain; this commit is the GATE fix only).

## 2026-06-03 — reread-EVICTION de-hack DESIGN (the 2nd row_reread consumer; the 1.31x CE wide win)

Mapped the eviction-slot -> host-buffer correspondence (codegen counts loads in device_ir.graphs order,
node-by-node; slot i = i-th hl.load; device_ir.py:2468-2473). EMPIRICAL (run, no GPU contention):
- **CE(4096,98304), len=5:** slot0=labels(Root), slot1=logits_flat(Root), slot2=**logits**(Root),
  slot3=**logits**(ReductionLoop amax), slot4=**logits**(ReductionLoop expsum). Re-read buffer = `logits`
  (the row_reread buffer, in 2 ReductionLoop graphs). Oracle eviction = ['','','last','first','last'];
  A/B winner reread_rowlast=['','','last','first',''] matched (slot4 'last' is a passenger).
- **welford(4096,4096), len=4:** slot0=x(ForLoop combine), slot1=x(ForLoop apply), slot2=weight, slot3=bias.
  Re-read buffer=`x` (2 ForLoop). Run-2's POSITIONAL ['last','first','first','first'] happens to be right
  here (x's first load=slot0='last', re-read=slot1='first') -- but it's a HACK: it assumes the re-read
  buffer's first load is slot0. For CE the re-read buffer's first load is slot2 (logits in root), so the
  welford positional rule would wrongly put 'last' on slot0=labels.

**FAITHFUL UNIFIED eviction policy (the de-hack):** identify the re-read HOST BUFFER (the one loaded in >=2
loop graphs = the row_reread buffer), find ITS load slots in codegen order, set the FIRST occurrence -> 'last'
(keep L2-resident for the re-read) and all subsequent occurrences -> 'first' (stream/evict the final uses);
every other buffer's slot -> default (''). Generalizes welford (slot0='last') AND CE (slot2='last', slot3/4=
'first') from ONE principled rule keyed on the re-read buffer's identity (reusing the SAME provenance as
row_reread), replacing run-2's positional slot[0]='last'. Verified-good on CE: reread_rowlast (slot2='last',
slot3='first') = oracle within noise, 1.31x@98304 / 1.19x@128256 / 1.09x@256000; NEUTRAL on persistent (no HBM
re-read). welford: reproduces run-2's slot0='last' (no regression).

OPEN before the eviction commit: (a) confirm "first-occurrence->last, rest->first" generalizes vs the oracle
on welford (does slot0='last' = the existing welford win? yes by construction) + a fresh non-CE re-read kernel;
(b) the CE eviction win is on the LOOPED wide-V shapes -- it COMBINES with the persist cap (boundary persistent,
wide looped+evicted) so it's additive to EDIT-GATE-v2, not a re-pairing; (c) A/B vs verbatim oracle, flip-set,
correctness, auditor (NOT a positional refit -- it's provenance-keyed) + referee. This is a fact-CONSUMING
heuristic change (no new fact) building on row_reread. HELD pending EDIT-GATE-v2 re-gate (don't stack
unconfirmed fact-consuming edits); design is captured + ready to implement on a fresh continuation.

## 2026-06-03 — eviction de-hack ARCHITECTURE decision + GPU yielded (hub firing gates on EDIT#1/GATE-v2)

Hub fired the EDIT#1/EDIT-GATE-v2 gate pipeline (referee+auditor+fact-integrity; anti-giving-up after referee
frees the GPU) and asked me to YIELD the GPU. Yielded — off the timing queue until cleared. Reconciled the
hub's 4 directives (it was a few commits behind my head): (1) num_load fact -> DONE = EDIT-GATE-v2/row_reread;
(2) oracle-key recipe -> DONE = corrected (0854c7bb) + re-keyed all 14 entries (02c825e8); (3) P2 looped-chunk
-> SUPERSEDED by my full oracle (chunk inert; win = eviction+pid, NOT chunk-scaling); (4) tc Triton -> deferred
to pid work. NON-GPU work proceeding: eviction de-hack design + pid scoping.

EVICTION DE-HACK ARCHITECTURE (decided): the faithful reread-eviction needs per-eviction-slot buffer identity
(which slots load the re-read buffer). This is PROVENANCE (device_ir), the POLICY is the heuristic's choice.
`get_seed_config(env, device_ir)` ALREADY receives device_ir -> the heuristic CAN consume the provenance
WITHOUT a new ReductionFact field. PLAN (option B, no new fact -> no extra fact-integrity gate, only auditor):
- device_ir: a small PROVENANCE HELPER (NOT a fact) returning the reduction-load slot->host-buffer mapping in
  codegen order (the same order `_count_device_loads_and_stores` numbers loads, device_ir.py:2468) — e.g.
  `reduction_load_host_names() -> list[list[str]]` (per eviction slot, the host names it loads).
- triton.py `_eviction_policies`: take device_ir; identify the re-read buffer (the host name appearing in >=2
  loop graphs — same provenance as row_reread, OR pass it through), find its slots in order, set FIRST -> 'last'
  + subsequent -> 'first', rest -> '' (default). Replaces run-2's POSITIONAL kind="reread" (['last']+['first']
  *(n-1)) which assumed slot[0] (right for welford, WRONG for CE where the re-read buffer's first slot is [2]).
- Applies to ALL row_reread T1+T2 kernels (not just is_structured_combine): welford reproduces slot0='last'
  (no regression); CE gets slot2='last',slot3/4='first' (the 1.31x/1.19x/1.09x wide-V win); rms/ln/softmax get
  their re-read buffer policied (NEW — re-litigate run-2's "rms/ln eviction = no clean rule"; the per-slot
  re-read provenance MAY now give a clean rule where the load-count didn't).
- HELD until EDIT-GATE-v2 re-gate lands (don't stack on an unconfirmed foundation) + GPU cleared for the A/B.
  A/B-ready: run3_ce_evict_ab.py covers CE; extend to welford + rms/ln before the commit.

OPEN QUESTION for hub (flagged): option B (device_ir provenance helper + heuristic policy, no new fact) vs a
provenance fact `reread_buffer_slots`. I lean B (keeps fact=provenance/heuristic=policy, avoids a 2nd
fact-integrity gate; the heuristic already has device_ir). Will implement B unless hub prefers a fact.

## 2026-06-03 — fact-integrity FAILed num_load (as expected) + hub/me message-cross; softmax verified no-regression

Hub's fact-integrity returned FAIL on the num_load>=2 gate (correct — converges with my EDIT-GATE-v2 work). Hub
then ADJUDICATED toward num_reduction_ops (EDIT#2) — but that crossed my commits: nro was ALREADY committed
(82a0de72) + REJECTED (hub-prescreen + independent fact-integrity, nro UNDER-counts rms/ln apply re-read) +
reverted (1cb50a6a). row_reread (EDIT-GATE-v2, 91dfd8ef) is the faithful replacement, strictly better than
both proxies (right on rms/ln where nro fails AND on CE where num_load over-counts). Sent the hub a timeline
reconciliation asking it to re-gate the COMMITTED EDIT-GATE-v2, not direct a fresh nro re-key.

HUB'S SOFTMAX CONCERN — verified empirically (committed floor data, no GPU): NO REGRESSION from EDIT-GATE-v2.
softmax is row_reread=True (2-pass max+sum). The 240KiB cap split is already correct + UNCHANGED vs num_load:
  softmax(2048,24576)=96KiB persist G=1.28; (2048,32768)=128KiB persist G=1.22 (both <240KiB, beat tc);
  softmax(1024,65536)=256KiB looped G=1.02; (512,131072)=512KiB looped G=1.00 (both >240KiB, at floor).
The wide ones are ALREADY looped under num_load (num_load=2), so row_reread changes NOTHING for softmax ->
cannot regress it. (Hub feared "512KiB softmax persistent->looped"; it's already looped at floor.)

### OPEN WORKLIST (Phase-2 per-shape; GPU-needed, deferred until queue clears) — keep current
- **softmax-wide persistent-vs-looped (NEW open):** softmax(1024,65536)/(512,131072) are looped at G≈1.0 (=tc)
  but I have NOT oracle'd them — "at floor vs tc" != "at oracle". The hub's hypothesis: softmax's SINGLE-operand
  re-read may spill LESS than CE's, so persistent might BEAT the looped seed at wide N (a potential FURTHER gain,
  NOT a regression from EDIT-GATE-v2). Oracle softmax(512,131072) + a persist-vs-loop A/B when GPU frees. If
  persistent wins, the 240KiB cap may need to be row_reread-AND-(num distinct streamed operands>1) — i.e. CE
  (logits re-read, but the per-pass working set is just logits) vs ... actually CE & softmax both single-operand;
  the CE spill came from the 2-pass resident row. Re-examine: WHY does CE spill at 256KiB but maybe softmax not?
  (Both hold one np2(N) row resident across 2 passes — should behave the same. If they DON'T, there's a finer
  property. Oracle decides.) This is exactly an anti-giving-up "is this shape really at its oracle?" check.
- CE wide-V: reread-EVICTION (1.31x, designed, 2nd row_reread consumer) + persistent-PID cluster (task #9, the
  rest of the 1.62x; blocked by #8 re-gate). jsd Band-B warps (w32->w16, 1.12-1.21x). welford apply-cap
  (8192->16384, 1.05-1.10x). softmax small-N occupancy-warps (M-dependent, needs occupancy fact). long_sum 2M
  source-limit (seed≈oracle, verify full oracle). CE residual oracle-vs-tc ~5% (2-pass source ceiling; online).

## 2026-06-03 — HUB: EDIT#1 value BANKED (referee+auditor PASS) + HARD GPU PROTOCOL + sequencing (a) EDIT-PID

Hub updates: (1) **NEW HARD GPU PROTOCOL** — before ANY GPU job DM "REQ-GPU <what/dur>", WAIT for "GPU-GRANTED",
DM "GPU-RELEASED" after. Hub holds the single timing token (a near-collision: my killed long_sum oracle PID
92659 was still autotuning when the referee spawned → it blocked 9min). I confirmed NO orphan GPU procs remain.
(2) **EDIT#1 cap VALUE (240KiB) BANKED** — referee PASS (counterfactual: monkeypatch back to 128KiB reverts CE
seeds to looped floor-loser → the raise is the sole cause) + auditor PASS (240KiB is a PHYSICAL threshold not a
CE fence: no shape in (196.5,256]KiB except dev-only 57344, so any cap in that window = identical curriculum
codegen). The num_load GATE under it still owes the faithful re-key = my committed EDIT-GATE/row_reread.
(3) Naming: gate re-key = **EDIT-GATE** (=row_reread, committed 91dfd8ef); wide-CE pid work = **EDIT-PID**.
(4) Sequencing: go (a) — land EDIT-PID first (largest gap 1.62x, generalizes).

EDIT-PID guardrails (hub) + my progress:
- **5a round-trip survival: PASS (non-GPU, verified).** The full-oracle CE(4096,98304) pid bundle
  (pid_type='persistent_interleaved', num_sm_multiplier=32, maxnreg=64, num_stages=4, range_unroll_factors=[4],
  range_num_stages=[2], range_flattens=[False], reduction_loops=[4096]) survives configs=[oracle] +
  normalize() FULLY PRESERVED — every lever intact in bound._config. So it IS Product-A-seedable (no silent
  knob-drop). Cleared.
- **5b lever decomposition: harness ready (run3_ce_pid_decomp.py), needs GPU.** Additive arms: seed + each
  lever (pid / pid+maxnreg / pipeline / chunk / evict) + carrier combos + full_bundle, each vs verbatim oracle
  (588) AND seed (955), with per-arm round-trip-drop logging. My EARLIER coarse ablation (run3_ce_pid_ab.py)
  already indicated eviction(1.31x)+pidcluster carry it, chunk/pipeline ~inert — this decomposes the pid
  cluster itself (pid-only vs +sm_mult vs +maxnreg) to name the carrier. REQ-GPU pending.
- **5c pid='flat' reversal rigor:** run-2 LOCKED flat via matched-lever A/B + pid_breakpoint_sweep concluding
  flat dominates on grid-SATURATED forward reductions; persistent only for grid-BOUND. My finding = wide-CE
  M=4096 is grid-LIGHT (few huge rows under-fill the SM grid) — a regime run-2 never probed. So the fix is a
  NEW workload-keyed branch ("grid-light -> persistent_interleaved+sm_mult"), NOT flipping the global flat
  default. Need a faithful GRID-OCCUPANCY fact (rows*m_block vs SM count: does the M-grid saturate the GPU?) —
  ask code-investigator. fact-integrity fires on the new fact + a pid-specific auditor/anti-giving-up confirms
  no narrow-forward regression. Re-check flat-vs-persistent on the OTHER grid-light shapes (softmax/welford/
  long_sum few-row) — measure, don't assume.

## 2026-06-03 — EDIT#3 reread-eviction IMPLEMENTED (faithful, provenance-keyed) — UNCOMMITTED, needs A/B

Hub's consolidated plan: do EDIT-GATE-v2 (cap, committed) + EDIT#3 (reread eviction) TOGETHER on the row_reread
fact (one fact, 2 consumers); de-hack run-2's POSITIONAL welford eviction. Implemented (UNCOMMITTED, pending
A/B + hub design review):
- ADDED `ReductionFact.reread_buffer_slots: tuple[int,...]` = eviction-slot indices loading the re-read buffer
  (codegen-emission order), captured at FACT-BUILD (HostFunction.current() is unavailable in the seed heuristic
  — that was a bug in my first attempt; moved provenance to fact-build). Provenance, not policy.
- `_compute_row_reread` -> `_compute_reread_provenance()` returns (row_reread, reread_buffer_slots), wired into
  both T1+T2 builders.
- `_eviction_policies(env,'reread',reread_slots)`: first slot -> 'last', rest -> 'first', others default.
  De-hacks the positional ['last']+['first']*(n-1).
- Routing: T1 num_load==1 -> 'stream'; elif row_reread -> 'reread'(slots); else default. Band-C welford ->
  'reread'(slots). T2 non-combine (softmax) unchanged (no eviction call — pre-existing).

EMITTED policies (verified, codegen): 
  CE(*,*): ['','','last','first','first'] (logits slots 2,3,4; first->last). [A/B winner was slot4='', here
    'first' — slot4 is a passenger per my A/B, expect ≈ equal; CONFIRM.]
  welford: ['last','first','',''] (x slots 0,1). RUN-2 positional was ['last','first','first','first'] ->
    DIFFERS at slots 2,3 (weight/bias): run-2 'first', faithful ''. NOT byte-identical — must A/B welford
    no-regression (weight/bias are tiny broadcasts; eviction likely neutral, but MEASURE).
  rms_norm: ['last','','first','first',''] ; layer_norm: ['last','','first','first','first','',''] — NEW
    eviction (run-2 left rms/ln default). row_reread routes them to 'reread'. MUST A/B (run-2 found "no clean
    rule" with the POSITIONAL version; faithful version targets x's real slots — may help OR regress; rows are
    small <=16KB so likely neutral, but MEASURE — could re-litigate run-2's "rms/ln no-rule").
  sum/long_sum: ['first','first'] (unchanged). kl_div/jsd: default (row_reread=False, unchanged).
  softmax: default (T2 non-combine, no eviction path — pre-existing; a FUTURE gain, not EDIT#3).

A/B PLAN (needs GPU): per affected kernel, NEW faithful eviction vs OLD champion eviction (CE/rms/ln: None;
welford: positional) vs tc. Must show: CE wins (1.31x/1.19x/1.09x wide-V reproduced), welford no-regression
(faithful vs positional), rms/ln no-regression. If rms/ln regress -> narrow the routing (gate on a finer
property). HELD: do NOT commit until (a) hub reviews the design (esp. welford-differs + rms/ln-new) and (b) the
A/B passes. Lint+pyrefly clean; seeds emit (no crash). Folding this A/B into the same GPU session as the pid
decomposition (REQ-GPU pending).

## 2026-06-03 — EDIT#3 EVICTION FLIP-SET (4-split) — BROAD blast radius -> NARROW scope to CE+welford

Computed the EDIT#3 eviction flip-set across ALL 4 splits (codegen-only, no do_bench). NEW faithful reread vs
OLD champion eviction. RESULT: the `elif fact.row_reread -> reread` routing flips eviction on:
- **cross_entropy: all shapes** (None -> ['','','last','first','first']) — the INTENDED 1.31x win. ✓
- **welford: all shapes** (positional ['last','first','first','first'] -> faithful ['last','first','','']) — the
  INTENDED de-hack (differs at weight/bias slots). Needs welford no-regression A/B.
- **rms_norm: ALL 40 shapes (train+val+test+robustness)** None -> ['last','','first','first','']. UNINTENDED-broad.
- **layer_norm: ALL shapes** None -> ['last','','first','first','first','','']. UNINTENDED-broad.
- sum/long_sum/kl_div/jsd/softmax: NO flip (correct).

KEY REALIZATION: row_reread routing applies reread-eviction to rms_norm + layer_norm on EVERY shape — a BROAD
change to two kernels currently AT FLOOR/parity with DEFAULT eviction, which I have NOT measured. This is bigger
than the hub's scoped intent (CE win + welford de-hack). run-2 found rms/ln eviction = "no clean rule" (with the
POSITIONAL rule); the faithful rule MIGHT be neutral (rms/ln rows <=16KB fit cache -> eviction likely moot) or
help or regress — UNMEASURED. Shipping a blanket eviction change to 2 at-floor kernels without measuring is the
"assume don't measure" failure.

DECISION: NARROW EDIT#3 to the PROVEN targets — CE (the 1.31x win) + welford (the de-hack) — and treat rms/ln
reread-eviction as a SEPARATE measured edit (re-litigating run-2's "no rule"; needs its own A/B). The clean
narrowing: the reread eviction MATTERS (measurable win) only when the re-read row is WIDE enough to spill L2 —
CE's win was on LOOPED wide-V; rms/ln at floor are PERSISTENT small rows (row fits cache, eviction moot). So
gate the T1 reread-eviction on `row_reread AND not persistent` (i.e. the looped wide regime) — OR keep
`row_reread` but require the rms/ln no-regression A/B to PASS before shipping the broad version. ASKED hub the
scope question; will narrow per its answer + the A/B. (welford is Band-C, separate routing — unaffected by the
T1 narrowing; its de-hack stands, pending its no-regression A/B.)

This is the task #8 "predict FULL 4-split flip-set, DM design before commit" deliverable — and it CAUGHT the
broad blast radius before commit (exactly why the flip-set-before-commit discipline exists). EDIT#3 stays
UNCOMMITTED until scope is settled + A/B passes.

NARROWED (option A, implemented + verified): gate the T1 reread-eviction on `row_reread AND not persistent`.
PRINCIPLED WHY: eviction policy only affects HBM-STREAMED loads; a PERSISTENT row is held in registers/SMEM
across the passes (no HBM re-stream) so its load eviction is MOOT (A/B: CE persist boundary 50304 eviction
neutral 1.001). So reread-eviction applies ONLY in the looped re-read regime (where the win is). New flip-set
(verified): CE looped wide-V -> reread (the 1.31x/1.19x win); CE persist-boundary + ALL rms/ln/softmax at-floor
train/val/test -> None = BYTE-IDENTICAL to champion (blanket blast radius ELIMINATED); rms/ln(1,131072) looped
ROBUSTNESS (2 shapes, >240KiB) -> reread (same wide-row physics as CE; correctness canaries, plausibly-helpful
direction); welford (Band-C, separate) -> faithful de-hack. So EDIT#3 now touches only the PROVEN win (CE
looped) + welford de-hack + 2 wide robustness canaries — no unmeasured change to any at-floor perf shape.
A/B still required: CE looped wins reproduced + welford no-regression (faithful vs positional) + rms/ln(1,131072)
robustness not-catastrophically-slow. Lint+pyrefly clean.

## 2026-06-03 — hub design anchors (1)+(2) checked against my impl — row_reread IS region-membership

Hub gave 2 anchors to avoid a 3rd proxy. Checked both against my COMMITTED row_reread (91dfd8ef):

ANCHOR (1) "region-membership, NOT load-node count": SATISFIED already. My `_compute_reread_provenance` counts
DISTINCT LOOP-GRAPH REGIONS whose read-SET contains B (set per region), then row_reread=any(count>=2). It is
NOT a load-node count -> immune to (a) single-pass double-load (one region = count 1 regardless of #loads) and
(b) fusion (collapsing loads within a region doesn't change membership). Demonstrated (run3_row_reread_probe +
region dump): per-region read-sets e.g. rms_norm regions=[{x},{weight,x}] -> x in 2 regions -> True; sum
regions=[{x}] -> 1 -> False; kl_div regions=[{y_pred,y_true}] -> each 1 -> False. 9/9.
ACID TEST (the separating kernel fact-integrity will demand): **rms_norm** — num_load=2 (fires, over-counts the
broadcast weight), num_reduction_ops=1 (would EXEMPT, misses the apply re-read), row_reread=True (GOVERNS,
x read in reduction-region AND apply-region). row_reread is right where BOTH proxies fail. (Same for layer_norm.)

ANCHOR (2) "per-slot eviction by buffer-identity + emission-order, not positional": my eviction IS faithful
(derived from reread_buffer_slots = which slots load the re-read buffer, codegen-emission order; first slot ->
'last', rest -> 'first' — NOT slot[2,3]/slot[0] literals). BUT the hub proposes a DIFFERENT assignment: "'last'
iff the slot's buffer is read in a LATER region." For CE (logits at slot2=root, slot3=amax, slot4=expsum):
  - hub's rule: slot2='last'(later reads exist), slot3='last'(expsum is later), slot4='first' -> [.,.,last,last,first]
  - my rule:    slot2='last', slot3='first', slot4='first' -> [.,.,last,first,first]
  - ORACLE + my A/B winner: slot2='last', slot3='first', slot4='last'(passenger) -> empirics say slot3='FIRST'.
So the hub's a-priori "later-region->last" over-marks slot3 (oracle disagrees). BOTH assignments are FAITHFUL
(buffer-identity+region-order, no positional literal); they differ only at the intermediate-pass slot. I'll
A/B BOTH (my first->last/rest->first VS hub's later-region->last) on the CE looped shapes and let the oracle
arbitrate the assignment — keeping whichever wins. (My current impl = the A/B-matched one. The provenance to
support the hub's variant would need per-slot "has-later-read" flags; cheap to add if the A/B prefers it.)

NET: row_reread (the cap gate) is DONE + faithful + acid-tested — ready to re-gate as-is. The eviction
assignment has a faithful-vs-faithful A/B to settle (mine vs hub's), needs GPU. Neither is a proxy.

## 2026-06-03 — fact-integrity REFINEMENT: "reused across reduction boundary" (liveness) vs my region-count

Gate's critical refinement: rms_norm loads x with a SINGLE hl.load in the SOURCE/persistent form — the row is
reused IN-REGISTER across reduction->apply, NOT a 2nd HBM load. So num_load (1 load of x) AND "read in >=2
regions" (my framing) could MISS it. The faithful property = REUSE/LIVENESS ACROSS THE REDUCTION BOUNDARY: the
reduction-input tile is consumed by the reduction AND a downstream apply/store (or >=2 reductions), regardless
of load mechanism (HBM re-read [CE] or in-register reuse [rms_norm]).

WHAT I FOUND investigating this (graph dumps, non-GPU):
- In the ROLLED loop graphs (what my fact inspects), rms_norm's apply graph[2] DOES re-load x (n_red=0,
  n_stores=1, loads={x,weight}). So the roller RE-LOADS the reused input when it rolls -> my region-membership
  gives rms_norm=True (9/9 still correct). My fact reads the ROLLED form (re-load), the gate reasons about the
  SOURCE form (in-register). They AGREE for the current curriculum BECAUSE the roller re-loads.
- I tried two consumer-dataflow predicates to implement the gate's "liveness" definition directly and BOTH are
  WRONG: (a) "load reaches a store" -> sum=True (false pos: sum's x reaches the store THROUGH the reduction
  output); (b) "load has an immediate non-reduction user" -> rms/softmax/CE=False (too shallow: x goes through
  x.to(fp32)/x*x intermediates before the reduction AND before the apply). The correct transitive "live across
  the boundary" predicate is SUBTLE (must distinguish "reaches store via the reduction output" from "the tile
  itself reaches a store bypassing the reduction").

THE CRUX (needs code-investigator, NOT a 4th guess): is my region-membership (rolled-form re-load count >=2)
EQUIVALENT to the gate's "reused across boundary" (liveness)? It is IFF the roller ALWAYS re-loads a reused
reduction-input in the rolled apply graph. If YES -> region-membership IS the faithful liveness signal
(expressed via the rolled form) and I keep it (9/9, and faithful-by-construction). If the roller can reuse
in-register EVEN ROLLED (some apply doesn't re-load) -> region-membership misses it and I need the careful
transitive consumer-dataflow predicate. The gate's softmax_decomposed example is the posited counterexample —
NOT in my curriculum, but the faithfulness thesis (TRANSFER) demands getting it right.

DECISION: do NOT guess a 4th predicate. ASK code-investigator the crux (does the roller re-load reused inputs
in the rolled apply graph?) + the exact transitive "tile live across the boundary" predicate if region-
membership is insufficient. row_reread stays committed AS-IS (9/9, behaviorally correct) meanwhile; I refine to
the consumer-based definition ONLY if the investigator shows region-membership can miss in-register-rolled
reuse. Either way the BEHAVIORAL set is right (gate confirmed) — this is a faithfulness-of-derivation question,
not a behavioral bug. Also TODO (gate cleanup): delete stale triton.py:494-499 num_reduction_ops comments.

## 2026-06-03 — row_reread REIMPLEMENTED as CONSUMER-TRACE (the gate's faithful "reused across boundary")

Hub firmly directed the CONSUMER-TRACE framing (not load-count, not region-count): rms_norm loads x with a
SINGLE hl.load, reused IN-REGISTER across reduction->apply. I empirically derived + VERIFIED the faithful
predicate, then REIMPLEMENTED `_compute_reread_provenance` to use it (device_ir.py). It supersedes my crux
question to code-investigator (whether region-membership is faithful) — consumer-trace is faithful regardless
of the roller's re-load behavior, so I switched to it.

FAITHFUL predicate (verified 9/9 in PRODUCTION fact.row_reread): `reduction_input_reused` = a loaded
reduction-input tile's VALUE is consumed by (>=2 distinct ReductionLowering(red_block_id)) OR (a
ReductionLowering AND a store reached on a path that BYPASSES the reduction). Consumer-DATAFLOW, not loads.
Derivation (empirically falsified 2 wrong predicates first): "reaches store" too coarse (sum's x reaches
store THROUGH the reduction output -> FP); "immediate non-red user" too shallow (x goes via x.to/x*x -> FN).
The correct one CUTS the BFS at the ReductionLowering: a store reachable from the load WITHOUT traversing the
reduction = the tile used outside the reduction = live across the boundary. Plus the >=2-reductions disjunct
for multi-reduction kernels (softmax/CE/welford whose reuse is feeding 2 reductions, not a bypass-store).
  sum/long_sum: feeds 1 reduction, no bypass-store -> False ✓
  rms_norm/layer_norm: feeds reduction + apply-store(bypass) -> True ✓ (ACID: single load, in-register reuse,
    still True — num_load=2 over-counts/fires, nro=1 under-counts/exempts, consumer-trace is the ONLY right one)
  softmax/cross_entropy/welford: feeds >=2 reductions -> True ✓
  kl_div/jsd: distinct inputs each feed 1 reduction, no bypass-store -> False ✓

IMPORTANT: this is faithful-by-CONSTRUCTION + behaviorally IDENTICAL to the prior region-membership version
(same row_reread 9/9, same emitted seeds, same cap flip-set) -> a PURE faithfulness upgrade, no behavior change.
The cap gate (fact.row_reread) is unchanged; the eviction (reread_buffer_slots) is computed SEPARATELY (HBM
re-read = buffer in >=2 loop graphs -> which slots; the hub's cap-vs-eviction distinction: cap=liveness,
eviction=HBM-re-load). Lint clean; the 2 pyrefly device_ir errors are PRE-EXISTING (lines 98/3125, not my
function). row_reread VALUES identical -> cap behavior + tests unchanged (test-suite re-run deferred to GPU
session; fact change is behaviorally inert so expected green).

STATUS: the faithful fact the gate demanded is IMPLEMENTED + acid-tested (rms_norm proof ready). In working
tree (uncommitted, with EDIT#3 eviction + comment refinement). DM hub: re-gate the consumer-trace row_reread
(fact-integrity should PASS — it's the exact definition + the acid kernel). EDIT#3 eviction still awaits scope-OK
(looped-gated) + A/B. Code-investigator crux question is now MOOT (switched to consumer-trace).

## 2026-06-03 — EDIT#3 reread-eviction DESIGN verified vs hub's HBM-re-load discipline (gates firing on EDIT-GATE-v2)

Hub: EDIT-GATE-v2 re-gate FIRING (fact-integrity acid=rms_norm + referee byte-identity/no-regression/CE-parity);
referee owns the GPU queue -> I STAY OFF GPU. Gate-only commit of EDIT-GATE-v2 was RIGHT (atomic faithfulness
fix). Proceed to design reread-eviction (EDIT#3, task #10); A/B needs REQ-GPU (after referee releases). Tasks
split: #8 EDIT-GATE-v2, #10 EDIT#3 eviction.

HUB DISCIPLINE for eviction (distinct from cap): eviction keys on per-LOAD-SLOT HBM-RE-LOAD ("a slot gets
'last' iff its buffer is RE-LOADED in a later-emitted slot"), which is FINER than row_reread. rms_norm's x is
loaded ONCE in-register (PERSISTENT form) -> no per-slot HBM re-load -> default (run-2's "x + broadcast -> no
rule"). Don't apply 'last' just because row_reread=True.

MY EDIT#3 ALREADY HONORS THIS (verified, codegen, non-GPU): routing = `row_reread AND not persistent` ->
reread-eviction from reread_buffer_slots (slots loading a buffer in >=2 LOOP graphs = HBM-re-loaded). Emitted:
  CE looped wide-V(98304): ['','','last','first','first'] (logits slots 2,3,4 — reduction-input-keyed, the win) ✓
  CE persist boundary(50304): None (persistent -> not routed; eviction moot when row resident) ✓
  rms_norm at-floor PERSISTENT(4096,4096): None ✓ (the hub's in-register case — persistent excluded)
  rms_norm LOOPED robustness(1,131072): ['last','','first','first',''] — x GENUINELY HBM-re-loaded when looped ✓
  layer_norm persist: None ✓ ; sum: ['first','first'] ✓ ; kl_div/jsd Band-B: None ✓
  welford Band-C(262144,4096): ['last','first','',''] — x slots 0,1, FAITHFUL de-hack of run-2's positional
    ['last','first','first','first'] (differs at weight/bias slots 2,3: run-2 'first', faithful '').
So the looped-gate IS the mechanism that excludes the in-register-reuse case the hub flagged: persistent
row_reread kernels (in-register, eviction moot) -> default; looped row_reread (HBM re-load) -> reread. The
reread_buffer_slots property (buffer in >=2 loop graphs) == "buffer re-loaded in a later slot" (>=2 loop graphs
=> >=2 slots load it). Reduction-input-keyed (CE logits, not labels/gather; welford x, not weight/bias).

WELFORD DE-HACK PROOF (replaces positional slot[0]='last'): run-2 `['last']+['first']*(n-1)` hardcoded slot[0].
For welford the re-read buffer x IS at slot[0] -> faithful gives slot[0]='last' too (the win reproduces), BUT
now DERIVED from x's identity (reread_buffer_slots=(0,1)), not the literal. For CE the re-read buffer logits is
at slot[2] (behind labels[0]+gather[1]) -> faithful gives slot[2]='last'; the positional rule would've wrongly
put 'last' on slot[0]=labels. That's the de-hack: same welford result, correct CE extension, from provenance.

A/B still required (REQ-GPU after referee releases): CE looped wins reproduced (1.31x/1.19x/1.09x), welford
no-regression (faithful vs positional — weight/bias 'first'->'' ), rms/ln(1,131072) robustness canary not-slow.
Also TODO: my eviction assignment (first->last,rest->first) vs the hub's earlier "later-region->last" at CE
slot3 — A/B both, oracle arbitrates (my A/B winner had slot3='first').

## 2026-06-03 — EDIT#4 (welford apply-cap) PRE-STAGED design (non-GPU, from the existing wf_tile_ab data)

Independent small edit (hub: fine to pre-stage). From run3_wf_tile_ab.py (already measured, committed):
- welford(4096,16384): seed [1,8192,2048] w16 (apply 2048) -> BEST [1,16384,4096] = 1.089x (combine 8192->16384
  AND apply 2048->4096); applyNp2(16384) WORSE (too big).
- welford(32768,8192): seed [2,8192,2048] -> BEST [2,8192,4096] w32 = 1.103x (apply->4096 + warps16->32 TOGETHER;
  warps32 alone WORSE).
- welford(16384,768): already at parity (apply moves inert).

PROPOSED EDIT#4: raise `STRUCTURED_APPLY_LOOP_CHUNK_BYTES` 8192 -> 16384 (apply tile 2048 -> 4096 fp32). The
apply tile is a PURE perf lever (masked write, correct at any width). The oracle/A/B want 4096, not the seed's
2048; not full np2(N) (that over-spills). Possibly also raise STRUCTURED_COMBINE_CAP_BYTES at (4096,16384)
(combine 8192->16384 helped there) — but that's a 2nd lever; A/B whether apply-alone gets most of it.
The (32768,8192) warps16->32 is a SEPARATE welford-warps question (the structured-combine warp ramp) — defer
unless apply+combine alone leaves it short.

NO-REGRESSION RISK + PLAN (needs GPU A/B): run-2 tuned the apply cap (8192) against the welford in-sample-v2 +
the wide-N curriculum; the 8 KiB threshold has a "valid-row-bytes" companion (STRUCTURED_APPLY_PERSIST_MAX_BYTES
=12288) deciding persist-vs-loop apply. Must A/B the apply-cap raise across the WHOLE welford train+val+test+
robustness for no-regression (esp. the narrow-N welford that were at floor, + the huge-M legacy 262144 rows).
Flip-set: shapes where the apply tile = min(np2(N), cap/itemsize) changes from 2048->4096 (N>=4096-ish looped-
apply welford). HELD until EDIT-GATE-v2 + EDIT#3 land (sequencing) + GPU; this is the smallest remaining gain
(1.05-1.10x on a few welford shapes), lower priority than CE pid (1.62x).

## SEQUENCING (hub-confirmed): EDIT-GATE-v2 (gating now) -> EDIT#3 reread-eviction (design done, A/B pending GPU)
   -> EDIT#4 welford apply-cap (pre-staged) -> EDIT-PID grid-light pid cluster (5a done, 5b harness ready, 5c
   grid-occupancy fact pending code-investigator). All A/Bs serialized behind the hub's GPU token (REQ-GPU each).

## 2026-06-03 — EDIT-PID SCOPING (pure analysis, no GPU): the pid cluster is NOT a clean workload rule (yet)

Hub: chunk directive WITHDRAWN (full oracle is the arbiter, not the directive — affirmed I was right to
supersede). Oracle re-key ACCEPTED by ledger-keeper. Pursue eviction + pid. GPU stays yielded (referee timing
EDIT-GATE-v2). Scoped the pid cluster from the CACHED oracle configs (pure JSON analysis, NO bind/GPU — referee
is timing):

| shape | oracle_pid | sm_mult | maxnreg | rl | progs/SM(132) | seed/oracle | oracle/tc | full? |
| (4096,50304) | flat | - | - | None | 31 | 1.00 | 1.06 | full |
| (8192,49152) | flat | - | - | None | 62 | 1.00 | 1.05 | quick |
| (8192,50257) | flat | - | - | None | 62 | 1.00 | 1.09 | quick |
| (8192,32000) | flat | - | - | None | 62 | 0.99 | 1.07 | quick |
| (8192,57344) | flat | - | - | [32768] | 62 | 1.11 | 1.05 | quick |
| (4096,98304) | **persist_interleaved** | 32 | 64 | [4096] | 31 | 1.62 | 0.95 | FULL |
| (8192,128256) | **persist_blocked** | 1 | - | [16384] | 62 | 1.25 | 0.64 | quick |
| (2048,256000) | **persist_interleaved** | 1 | 32 | [2048] | 15.5 | 1.13 | 0.62 | quick |

KEY FINDINGS (temper the "clean grid-light pid branch" hypothesis):
1. The pid choice is SHAPE-INCONSISTENT across wide-CE: persist_interleaved/sm_mult=32 (98304), persist_blocked/
   sm_mult=1 (128256), persist_interleaved/sm_mult=1 (256000). THREE different pid configs for three wide shapes
   -> NOT a single clean rule. Looks like autotuner fine-tuning, not an obvious seedable workload property.
2. progs/SM does NOT separate flat-vs-persistent: (4096,50304) flat AND (4096,98304) persistent BOTH have
   progs/SM=31. So "grid-light by program count" is the WRONG property (my hypothesis to code-investigator is
   likely falsified — the distinguishing factor is V/looped-vs-persistent reduction, not the grid occupancy).
   The 1.62x at 98304 is the LOOPED regime (rl=[4096]); the flat parity shapes are PERSISTENT (rl=None).
3. 2 of the 3 "persistent-pid wins" are QUICK oracles that LOSE to tc (oracle/tc 0.64/0.62) — SUSPECT
   (under-explored, like the earlier CE wide quick oracles). Only (4096,98304) has a FULL oracle (1.62x, oracle/
   tc=0.95 real). So the pid "win" is solidly evidenced on ONLY ONE shape so far.

=> EDIT-PID RE-SCOPED: before ANY pid branch, NEED (a) FULL oracles on 128256 + 256000 (the quick ones are
suspect; full might pick a consistent pid OR reveal it's not seedable), and (b) the lever-decomposition A/B
(run3_ce_pid_decomp.py ready) to isolate which lever carries 98304's 1.62x (pid vs sm_mult vs maxnreg vs the
looped chunk 4096 vs num_stages) — my earlier coarse ablation said eviction(1.31x)+pid carry it, chunk inert.
If the pid lever (a) generalizes across the wide shapes with a faithful workload key and (b) survives as a
seedable config, it's a branch; if it's shape-inconsistent autotuner-only fine-tuning, EDIT-PID may be
SMALLER than 1.62x (the eviction 1.31x is the robust, seedable part; the pid residual may be autotuner-only).
This is honest: the 1.62x = eviction(1.31x, seedable, EDIT#3) * pid-cluster(~1.24x, MAYBE seedable, EDIT-PID).
Grid-occupancy fact (5c) hypothesis WEAKENED (progs/SM doesn't separate) — told code-investigator; the real
question may be "looped wide multi-load re-read -> does a persistent grid help the looped inner pass?" not grid
occupancy. ALL needs GPU (full oracles + A/B); deferred to after EDIT-GATE-v2/EDIT#3/EDIT#4 per sequencing.

### Current champion
- Run-2 `TritonReductionHeuristic` + EDIT#1 (cap 240KiB, VALUE BANKED) + EDIT-GATE-v2 (persist-cap gate = fact.row_reread,
  the faithful re-read property; replaces the rejected num_load/nro proxies). Seed-byte-identical to the
  num_load placeholder curriculum-wide; CE boundary at oracle parity; wide rms/ln robustness correctly looped
  (no regression); tests pass. Ready to DM design + commit + re-gate. NEXT perf: reread eviction (1.31x CE
  wide, same fact) then the pid cluster.
- HELD/characterized edits (A/B done): EDIT#3 CE re-read eviction (1.09-1.31x wide CE, matches oracle; needs
  faithful per-slot re-read provenance -- code-investigator query out); EDIT#4 welford apply-cap 8192->16384
  (1.05-1.10x); EDIT#5 jsd Band-B warps 32->16 (num_tiled_accumulators-keyed, 1.12-1.21x). HARDER/deferred:
  CE persistent-pid cluster (rest of wide-CE 1.62x, re-opens pid='flat'); softmax small-N occupancy-warps.
- Deliberately NOT committing more edits until (a) hub re-gates EDIT#1+2 and (b) provenance answer lands --
  to avoid building on unconfirmed foundations + flooding the gate queue. Board fully characterized; ready to
  execute the edit queue on the hub's cadence.

## 2026-06-03 — softmax-wide persist-vs-loop A/B PRE-STAGED (hub GPU-step #2, non-GPU prep)

Hub confirmed (DM): my softmax row_reread check found a real FORWARD GAIN, not a risk. Affirmed: EDIT-GATE-v2
cannot regress softmax (wide softmax already looped under num_load>=2; row_reread doesn't move its persist/loop
split). NEW open is a legit Phase-2 oracle target: **is softmax-wide-LOOPED optimal, or does PERSISTENT beat it?**
Physics: softmax_two_pass is SINGLE-OPERAND re-read (x re-read for the exp-sum pass after the max pass) -> resident
set is ONE row of x, LIGHTER than CE's multi-pass working set (logits + labels + target gather). So softmax may
SPILL LESS than CE at the same byte width and want a HIGHER persist threshold than CE's 240KiB cap.

Built `_lab/harness/run3_softmax_persist_ab.py` (committed 6906970e). softmax_two_pass is T2 (user-tiled): the
reduction axis is a `block_sizes` entry (inner `hl.tile(n, block_size=block_size_n)`), NO reduction_loops knob.
PERSISTENT == block_size_n >= np2(N); LOOPED == capped chunk. The harness finds the reduction-axis index
GENERICALLY (fact.block_id -> block_sizes index, NOT hardcoded) then mutates ONLY that entry. Arms: seed_looped
(block_size_n=16384,w32), persist (np2(N)), persist_w16, persist_ns2, chunk_32768 (bigger looped chunk -- isolates
chunk-size from full-persistence), tc_default. do_bench median-of-N, correctness-gated, fp32 asserted, one process.
AST+import verified (no GPU): _np2(65536)=65536, _np2(131072)=131072 (both pow2 -> persist arm = whole row).

Target shapes (both >240KiB so currently looped at floor G~1.0=tc, never oracle'd):
  softmax(1024,65536)=256KiB ; softmax(512,131072)=512KiB.
Pair the A/B with a FRESH oracle (run3_oracle.py --kernel softmax) on the same two shapes -- the A/B settles
"does persistent beat looped", the oracle is the actual arbiter (may find an even better chunk/warps/stages).
If persistent wins: the cap differentiator may need to become row_reread AND (distinct-streamed-operands>1) so
CE (heavier) caps lower than softmax (single x) -- but re-examine WHY they'd differ (notebook open Q lines
891-896): both hold one np2(N) row resident across 2 passes, should behave the same; if they DON'T, there's a
finer property the oracle exposes. If persistent LOSES/ties: softmax-wide is confirmed at oracle (the 240KiB
cap is right for it too), close the open.

GPU SEQUENCING (hub holds token; await GPU-GRANTED): (1) EDIT#3 eviction A/B [highest value, 1.31x ready] ->
(2) softmax-wide persist-vs-loop A/B + oracle [this new gain] -> (3-later) pid cluster + long_sum-tail
anti-giving-up. All harnesses ready; no GPU touched.

## 2026-06-03 — fact-integrity PASS on row_reread + EDIT#3 buffer-identity TIGHTENING (hub caveat) + proof

**Hub: fact-integrity PASS on row_reread — FAITHFUL, not a third proxy.** The gate's graph-dump nailed the
rms_norm acid: the ROLLER materializes the apply pass as a SEPARATE ReductionLoopGraphInfo (n_red_lowerings=0)
that RE-EMITS the x load -> x in 2 loop graphs -> True for the right structural reason. This OVERTURNS the
earlier "rms_norm reuses x in-register, no 2nd load" claim: post-roller reality IS a genuine 2nd load (which is
what spills). Style-independent where num_load failed (softmax_decomposed=True), online-CE correctly False. 9/9.

**Hub caveat (fix BEFORE commit, non-GPU):** `reduction_input_reused` (the persist-CAP gate) is safe with the
loose loop-graph test (mis-key benign: would cap an already-wide single-pass row to looped). BUT EDIT#3's
EVICTION consumer EQUATES "the >=2-loop-graph buffer" with "the row to keep L2-resident" — there a coincidental
re-loaded BROADCAST (not the row) would get the wrong slot marked 'last'. So TIGHTEN reread_buffer_slots: select
the buffer that is BOTH (a) HBM-re-read (>=2 loop graphs) AND (b) a REDUCTION INPUT (its loaded value feeds the
ReductionLowering). Keyed on buffer IDENTITY, not "the count>=2 one."

**DONE (device_ir.py `_compute_reread_provenance`, UNCOMMITTED):** added `reduction_input_buffers` as a
by-product of the SAME consumer-dataflow walk (a load whose value reaches a ReductionLowering(red_block_id) ->
record its host-buffer name via _fx_trace_tensor_arg_rw_names), then `hbm_reread = {nm : count>=2 AND nm in
reduction_input_buffers}`. Docstring updated.

**VERIFIED 9/9 production (bind-only, no GPU; `run3_reread_slots_probe.py`):** every kernel's slots point at the
REDUCTION ROW, broadcasts excluded:
  sum/long_sum: row_reread=F slots=() evict=['first','first'] (stream)
  rms_norm(1,131072): T slots=(0,2,3)->x [NOT 1,4=weight] evict=['last','','first','first','']
  layer_norm(1,131072): T slots=(0,3,4,5)->x [NOT weight/bias] evict=['last','','','first','first','first','','']
  softmax(512,131072): T slots=(0,1)->x; evict=None (T2 plain path emits no eviction — slots unused, fine)
  cross_entropy(2048,128256): T slots=(2,3,4)->logits [NOT 0=labels,1=logits_flat] evict=['','','last','first','first']
  kl_div/jsd: F slots=() evict=None
  welford(8192,8192): T slots=(0,1)->x [NOT 2,3=weight/bias] evict=['last','first','','']
ALL match the pre-tightening 9/9 -> the tightening is OUTCOME-NEUTRAL on the whole curriculum (no regression).

**PROOF the tightening is LOAD-BEARING (not dead code) — adversarial kernel `adv3` (temp, deleted):** a pure
pass-through broadcast `gain` loaded in 2 APPLY passes (NEVER reduced) at slot 0 (BEFORE the row x at slots 1,2):
  reduction_input_buffers={x}; loop_graph_count={gain:2, x:2}; LOOSE hbm={gain,x}; TIGHT hbm={x}; DROPS=[gain].
  LOOSE would select 'gain'->slots(0,3) (WRONG: marks a broadcast 'last'); TIGHT selects 'x'->slots(1,2) (RIGHT).
  Production fact.reread_buffer_slots=(1,2) == TIGHT. So the AND-with-reduction-input bites EXACTLY on the
  caveat case and is inert on production. (Two earlier adv attempts made bias a reduced operand -> in
  reduction_input_buffers legitimately -> didn't isolate it; adv3 — bias used ONLY in apply, never summed — is
  the clean isolation. Lesson: in single-reduction-fact kernels, ANY summed operand is a 'reduction input'; the
  caveat buffer must be apply-only.)

**Comment cleanups (gate-flagged, triton.py, DONE uncommitted):** the cap-gate note no longer calls row_reread a
"num_load>=2 placeholder"; it now describes the consumer-dataflow liveness mechanism + the post-roller 2nd-load
reality. MULTILOAD_PERSIST_MAX_BYTES doc updated to "gated by fact.row_reread."

STATUS: source changes (consumer-trace row_reread + buffer-identity tightening + EDIT#3 routing + comment fixes)
all UNCOMMITTED, lint+format clean. Holding for hub's version decision (subgraph-membership 91dfd8ef vs land the
consumer-trace as the committed fact) + GPU-GRANTED for the eviction A/B. DMing the hub the eviction design +
buffer-identity proof now.

## 2026-06-03 — HUB DECISION (A): keep the gated BOOL row_reread; reconcile working tree (no re-gate)

Hub adjudicated the bool-vs-int question: **keep the committed loop-graph-count BOOL `_compute_row_reread`**
(the one fact-integrity PASSed). Reasoning (a real principle, not less-work): an int `row_read_passes=2` serves
NEITHER consumer. Cap consumer needs only the binary >=2 -> bool sufficient. Eviction consumer needs per-SLOT
buffer-identity + emission-order (which slot holds the re-read buffer, to set it 'last') -> FINER than an int,
computed separately regardless. A scalar between them is generality no consumer consumes = textbook
over-engineering (the symmetric failure to the proxy hack; work-order + fact-integrity both warn against it).
Compute the EXACT property each branch needs: bool for the cap, per-slot mapping for eviction.

**RECONCILED the working tree** (my uncommitted tree had REPLACED the gated bool with a consumer-trace
`reduction_input_reused` returning (bool, slots) — a DIFFERENT computation, 9/9-equivalent on production but not
the one gated). Restructured into TWO methods off the same provenance resolver:
- `_compute_row_reread(self) -> bool` — RESTORED byte-identical to gated HEAD (only the docstring extended).
  diff vs HEAD = docstring only; executable body unchanged -> fact-integrity's PASS carries over, NO re-gate.
- `_compute_reread_buffer_slots(self, red_block_id) -> tuple[int,...]` — the EVICTION slots. Keeps the
  consumer-dataflow walk ONLY to build `reduction_input_buffers` (which buffer feeds the ReductionLowering),
  then selects the buffer that is BOTH HBM-re-read (>=2 loop graphs) AND a reduction input -> its slots. The
  buffer-identity tightening (fact-integrity caveat); proven load-bearing by adv3.
Both call sites: `row_reread=self._compute_row_reread()`, `reread_buffer_slots=self._compute_reread_buffer_slots(...)`.
Dropped the consumer-trace bool entirely (`_compute_reread_provenance` GONE, 0 refs).

**RE-VERIFIED 9/9 IDENTICAL** after the restructure (run3_reread_slots_probe.py, bind-only): bool sum/long_sum/
kl_div/jsd=F, rms/ln/softmax/CE/welford=T; slots CE->logits(2,3,4), welford->x(0,1), rms->x(0,2,3), ln->x
(0,3,4,5), broadcasts excluded; emitted eviction unchanged (CE ['','','last','first','first'], welford
['last','first','','']). triton.py cap-gate comment corrected to describe the loop-graph-count mechanism (was
mid-edit describing the consumer-trace + a now-deleted method name). Lint+format clean.

So EDIT-GATE-v2 = the gated bool, final (no refine, no re-gate). EDIT#3 = the per-slot eviction mapping, finer,
off the same resolver. Hub: EDIT-GATE-v2 fact-integrity PASS ✓, awaiting results-referee (byte-identity +
no-regression, on GPU). On its PASS, EDIT#1(value)+EDIT-GATE-v2(bool) ship as the accepted champion advance.
Comment cleanups = trivial follow-up commit (no gate). Source all UNCOMMITTED, awaiting hub commit go-ahead +
GPU-GRANTED for the EDIT#3 A/B.

## 2026-06-03 — EDIT#3 EVICTION A/B RECEIPTS (GPU-GRANTED, do_bench median-of-7) — WIN + de-hack-attributable

EDIT#1+EDIT-GATE-v2 ACCEPTED (referee PASS 5/5: byte-identical 18/18, rms/ln 512KiB looped, CE boundary
oracle-parity, 9/9 row_reread, correctness). Substrate faithful + banked. Ran GPU-step-1: CE eviction A/B
(run3_ce_evict_ab.py, fp32, do_bench median-of-7, correctness-gated, one process). GPU released after.

| shape           | codegen    | default us | seed_emitted us | seed/default | pos_slot0 | oracle_exact |
|-----------------|------------|-----------:|----------------:|-------------:|----------:|-------------:|
| (4096,50304)    | persistent |     282.5  |          282.5  | **1.000** ✓  |   0.970   |     1.000    |
| (4096,98304)    | looped     |     956.7  |          731.9  | **1.307**    |   0.997   |     1.308    |
| (8192,128256)   | looped     |    2715.0  |         2287.1  | **1.187**    |   1.042   |     1.188    |
| (2048,256000)   | looped     |    1362.8  |         1257.4  | **1.084**    |   0.998   |     1.084    |

seed_emitted policy (EDIT#3, from provenance) = ['','','last','first','first'] (wide looped); None (persistent
boundary). Findings:
1. **seed_emitted == oracle_exact** at EVERY shape (731.9 vs 731.6; 2287.1 vs 2286.1; 1257.4 vs 1257.6). My
   de-hacked policy matches the oracle's ['','','last','first','last'] in perf; the lone diff (slot-4 first vs
   last) is perf-neutral. => the SEED EMITS THE ORACLE-OPTIMAL eviction. This is a true seed/oracle=1.00 win.
2. **De-hack-ATTRIBUTABLE** (defeats the auditor's "positional refit in disguise"): pos_slot0 = the run-2
   POSITIONAL rule ['last','first','first','first',''] gives 0.997/1.042/0.998 — barely above default — and at
   the persistent boundary REGRESSES to 0.970. The win comes specifically from putting 'last' on LOGITS (slot 2,
   the reduction row's first load), which only the buffer-identity provenance finds. Positional slot-0='last'
   marks LABELS (a scalar gather) -> no win. So the gain is the de-hack, not "some eviction."
3. **Persistent boundary (50304) NEUTRAL**: seed_emitted=default=1.000 (the `not persistent` gate emits None).
   pos_slot0 would REGRESS it 0.970 -> confirms gating eviction to `not persistent` is correct (persistent row
   held in regs/SMEM, no HBM re-stream, eviction moot).
4. all_first/all_last inert-or-worse -> the principled per-slot policy beats blanket policies.

CONCLUSION: EDIT#3 is a clean seedable win, matches the oracle, attributable to the buffer-identity de-hack, no
boundary regression. Reporting receipts to hub BEFORE committing (per hub directive). Welford no-regression A/B
(run3_reread_noregress_ab.py) still to run (next GPU grant) to confirm the welford de-hack reproduces its run-2
eviction win + rms/ln(1,131072) robustness canary -- THEN commit EDIT#3 (fires auditor+referee, no fact-integrity).

## 2026-06-03 — EDIT#3 no-regress A/B FINDING: de-hack REGRESSES welford 3-6% vs shipping positional -> RULE FIX

Ran run3_reread_noregress_ab.py (welford TRAIN + rms/ln robustness). do_bench median-of-7, fp32.

| case               | default us | seed_emitted (de-hack) | pos_run2 (positional) | de-hack vs pos |
|--------------------|-----------:|-----------------------:|----------------------:|---------------:|
| welford(65536,4096)|    1003.5  | 799.2 (1.256x)         | 776.7 (1.292x)        | -2.9% (slower) |
| welford(32768,8192)|    1004.5  | 844.5 (1.189x)         | 790.3 (1.271x)        | -6.4% (slower) |
| rms_norm(1,131072) |      22.7  | 21.3 (1.066x)          | 21.3 (1.068x)         | ~tie           |
| layer_norm(1,131072)|     26.6  | 26.2 (1.016x)          | 26.6 (1.001x)         | +tie/slight    |

**CRITICAL: the CURRENTLY-SHIPPING champion (accepted EDIT-GATE-v2) emits the POSITIONAL rule for welford.**
Committed `_eviction_policies(env,"reread")` = `["last"]+["first"]*(n-1)` (HEAD triton.py:460) -> welford ships
`['last','first','first','first']` = pos_run2. My EDIT#3 de-hack `['last','first','','']` (row x slots 0,1;
weight/bias slots 2,3 left default) is **3-6% SLOWER** -> EDIT#3 as-designed REGRESSES the accepted welford. NOT
acceptable (regression on an accepted-champion shape).

ROOT CAUSE: positional does TWO things — (1) 'last' on slot 0, (2) 'first' on ALL other slots. My de-hack does
(1') 'last' on the re-read ROW's first load + 'first' on its re-reads, but leaves OTHER buffers DEFAULT ''. The
welford loss is leaving weight/bias at '' instead of 'first': they're streamed-once small broadcasts, so
evict-FIRST frees L2 (same physics as the num_load==1 stream recipe). My de-hack under-streams them.

FIX (principled, faithful, captures BOTH wins): the reread policy = **'last' on the re-read row's FIRST load;
'first' on EVERY OTHER slot** (row's re-reads AND all streamed-once operands -> stream/evict-first). This:
  - welford: row x@slot0->'last', x re-read@slot1->'first', weight/bias@2,3->'first' = ['last','first','first',
    'first'] == positional (reproduces the 1.27x win, byte-identical to shipping).
  - CE: 'last' on logits@slot2 (the de-hack — provenance puts it on the ROW, not positional slot0=labels),
    'first' on labels@0/logits_flat@1/logits-reread@3,4 = ['first','first','last','first','first'].
The ONLY open: does CE want slots 0,1 (labels/logits_flat gathers) 'first' vs my tested ''? My A/B tested
['','','last','first','first']=1.31x; need to test ['first','first','last','first','first']. Testing Rule B
on CE now (still hold GPU token). If Rule B ties/beats my 1.31x on CE AND reproduces welford positional, it's
the faithful rule that regresses nothing. (This keeps the buffer-IDENTITY de-hack for the 'last' slot — the
fact-integrity caveat — while streaming everything else, which is the run-2 behavior that happened to be right.)

## 2026-06-03 — RULE B CONFIRMED + IMPLEMENTED: the faithful reread policy that regresses nothing

Rule-A-vs-Rule-B A/B (run3_reread_rulefix_ab.py, do_bench median-of-7, fp32):

| case               | default | ruleA (mine, old) | ruleB (FIX)  | pos_run2 (shipping) |
|--------------------|--------:|------------------:|-------------:|--------------------:|
| CE(4096,98304)     |  1.000  | 1.303             | **1.305**    | 0.995               |
| CE(8192,128256)    |  1.000  | 1.187             | **1.190**    | 1.040               |
| welford(65536,4096)|  1.000  | 1.249             | **1.283**    | 1.283               |
| welford(32768,8192)|  1.000  | 1.193             | **1.273**    | 1.277               |

RULE B = 'last' on the provenance-identified re-read ROW's first load; 'first' on EVERY OTHER slot. Verdict:
- CE: Rule B ties/beats Rule A (1.305 vs 1.303; 1.190 vs 1.187) -- streaming the labels/logits_flat gathers
  ('first' on slots 0,1) is marginally better, never worse. The 1.31x/1.19x win HOLDS.
- welford: Rule B == pos_run2 (1.283/1.283; 1.273/1.277 within noise) -- REPRODUCES the shipping positional
  win. Rule B's welford policy IS ['last','first','first','first'] (row x@slot0->'last', rest 'first') =
  BYTE-IDENTICAL to the shipping champion -> NO regression.
- Rule B KEEPS the buffer-identity de-hack: CE policy ['first','first','last','first','first'] -- 'last' on
  logits@slot2 (the ROW), NOT positional slot0=labels. pos_run2 STILL FAILS CE (0.995/1.040, marks labels
  'last') -> the win is the buffer-identity 'last' placement, not "some eviction." Defeats "positional refit."

IMPLEMENTED (triton.py `_eviction_policies` "reread"): policy = ['first']*n; policy[slots[0]]='last'. One-line
change from Rule A (which had default '' on non-row slots). Re-verified emitted (bind-only, run3_reread_slots_probe):
  sum/long_sum ['first','first']; rms ['last','first','first','first','first']; ln ['last']+['first']*7;
  softmax None (T2); CE ['first','first','last','first','first']; kl/jsd None; welford ['last','first','first','first'].
rms/ln(1,131072) robustness: Rule B's policy == the pos_run2 arm ALREADY tested in the no-regress A/B (rms 1.068x,
ln 1.001x vs default) -> confirmed correct + not slower. So Rule B is validated on EVERY EDIT#3-affected shape:
CE 1.31x/1.19x (=oracle), welford 1.27-1.28x (=shipping, no regress), rms/ln robustness 1.00-1.07x (canary OK).

EDIT#3 = Rule B. Lint+format clean. Ready: report full receipts to hub, then commit (fires auditor+referee,
no fact-integrity). welford no-regress now AFFIRMATIVE (Rule B reproduces shipping byte-identically).

## 2026-06-03 — GPU-step-2: softmax-wide oracle — persist hypothesis REFUTED; both shapes VICTORY (no EDIT)

Quick oracle (run3_oracle.py, fresh autotune, fair-re-bench, source-hash cached):

| shape              | seed_us | oracle_us | seed/oracle | oracle cfg (block_sizes) | verdict |
|--------------------|--------:|----------:|------------:|--------------------------|---------|
| softmax(1024,65536)|  263.7  |   263.7   | **1.000**   | [1,16384] (SAME as seed) | VICTORY |
| softmax(512,131072)|  275.8  |   272.5   | **1.012**   | [1,4096] (smaller chunk) | VICTORY (tie) |

The hub's hypothesis (single-operand softmax spills LESS than CE -> wants PERSISTENT wider than the 240KiB cap)
is NOT borne out:
- (1024,65536)=256KiB: oracle == seed == [1,16384] LOOPED. seed/oracle=1.000. The 240KiB cap is RIGHT for
  softmax here too -- persistent did NOT win. (So softmax and CE behave the SAME at 256KiB: both spill, both
  want looped. The "finer property" question from the earlier open is resolved: there ISN'T one; the cap is
  correctly shared.)
- (512,131072)=512KiB: oracle picked [1,4096] (a SMALLER looped chunk, w16), NOT persistent. 272.5us vs seed's
  [1,16384] 275.8us = seed/oracle 1.012 -- within the 3-5% tie band (a VICTORY by the bar). The 1.2% is a
  chunk-size preference (4096 vs 16384) at quick-oracle noise; chasing it would mean changing the SHARED
  LOOPED_CHUNK=16384 for one shape (broad re-validation, sub-noise payoff) -- NOT worth it, and the bar is met.

CONCLUSION: softmax-wide is at oracle (1.000 / 1.012). NO new EDIT. The at-floor-vs-tc observation ("G~1.0")
was correctly "at oracle" after all -- this CLOSES the softmax-wide open. (Good anti-giving-up discipline: I
oracle'd it rather than assuming; the oracle confirmed the cap, didn't reveal a gap. Honest null result.)
softmax persist A/B (run3_softmax_persist_ab.py) is now UNNECESSARY -- the oracle already answered (looped wins).

## 2026-06-03 — EDIT-PID lever DECOMPOSITION (5b) — TWO carriers (evict + pid-cluster), rest passengers

GPU-GRANTED. run3_ce_pid_decomp.py on CE(4096,98304): each oracle lever added ONE-AT-A-TIME to the
eviction-STRIPPED seed (the true 955us floor), do_bench median-of-7. Oracle target 588.4us (1.624x). All arms
round-trip-survive normalize (dropped='-' everywhere) -> Product-A-seedable (5a holds for the carrier set).

| arm                              | us     | seed/arm | arm/tc | pid                    | verdict       |
|----------------------------------|-------:|---------:|-------:|------------------------|---------------|
| seed_floor (no evict)            |  955.3 | 1.000    | 0.593  | flat                   | floor         |
| seed+pid (interleaved,sm_mult32) |  866.9 | 1.102    | 0.654  | persistent_interleaved | CARRIER       |
| seed+pid+maxnreg(64)             |  849.9 | 1.124    | 0.667  | persistent_interleaved | CARRIER (+.02)|
| seed+pipeline (ns4,unroll,...)   |  955.3 | 1.000    | 0.593  | flat                   | PASSENGER     |
| seed+chunk (rl=4096)             | 1018.1 | 0.938    | 0.557  | flat                   | HURTS         |
| seed+evict (Rule B / oracle)     |  733.1 | 1.303    | 0.773  | flat                   | CARRIER (top) |
| seed+indexing (tensor_descriptor)|  955.8 | 1.000    | 0.593  | flat                   | PASSENGER     |
| seed+evict+pid+maxnreg           |  595.0 | **1.606**| 0.952  | persistent_interleaved | **= oracle**  |
| seed+evict+indexing              |  733.5 | 1.302    | 0.773  | flat                   | (idx nothing) |
| seed+evict+pid+maxnreg+indexing  |  594.3 | 1.608    | 0.954  | persistent_interleaved | (idx nothing) |
| full_bundle (7 levers)           |  590.2 | 1.619    | 0.960  | persistent_interleaved | the oracle    |

DECOMPOSITION (clean — confirms the hub hypothesis):
- **TWO carriers**: (1) EVICTION Rule B = 1.303x alone (EDIT#3, re-confirmed). (2) PID-CLUSTER
  pid='persistent_interleaved' + num_sm_multiplier=32 + maxnreg=64 = 1.124x alone (pid 1.102 + maxnreg +0.022).
- **evict x pid-cluster = 1.606x** = essentially the FULL bundle (1.619x). The two carriers capture ~99% of the
  1.62x; they SYNERGIZE (1.303 x 1.124 = 1.465 < measured 1.606 -> super-multiplicative: a persistent grid +
  L2-resident re-read row compound).
- **PASSENGERS / HARMFUL** (the other 4 oracle levers): pipeline (num_stages=4 + range_* = 1.000, exactly inert),
  indexing (tensor_descriptor = 1.000, EXACTLY inert -- adds nothing even on top of evict: 733.5 vs 733.1),
  chunk (reduction_loops=4096 = 0.938, actively HURTS -- the oracle's chunk is WORSE than the seed's 16384 in
  isolation; the autotuner kept it as a passenger coupled to the pid grid, not a real win).

=> EDIT-PID carrier = the 3-lever PID-CLUSTER {pid_type='persistent_interleaved', num_sm_multiplier=32,
maxnreg=64}, on top of EDIT#3 eviction. A 3-lever seed captures the full 1.62x -- FAR more general than the
7-lever oracle (drop the 4 passengers). This is the run-2 'flat' lock RE-OPENED with evidence: persistent_
interleaved BEATS flat by 1.10-1.12x ON THIS WIDE LOOPED MULTI-LOAD RE-READ SHAPE (NOT the narrow-forward
shapes where run-1 found flat dominant 1.5-4x). The WORKLOAD KEY question (5c, code-investigator): WHAT property
separates "wide looped re-read CE -> wants persistent_interleaved" from "narrow forward -> wants flat"? Candidate:
the LOOPED regime (row_reread AND not persistent AND wide) -- a persistent grid amortizes the looped inner
re-read passes' launch/occupancy. NOT grid-occupancy alone (progs/SM didn't separate). Need: the fact that keys
sm_multiplier/maxnreg too (they require pid_type != flat). Reporting to hub; EDIT-PID needs the workload fact +
full gate set (fact-integrity on the grid fact + auditor + narrow-forward no-regression that flat is undisturbed).

## 2026-06-03 — EDIT-PID fact-design CONSTRAINT (non-GPU prep, while CE-wide full oracles run)

Before designing the EDIT-PID workload fact, the KEY constraint (the over-generalization risk the auditor +
narrow-forward no-regression gate guard): the decomp proved persist_interleaved+sm_mult+maxnreg helps CE(4096,
98304) by 1.12x. But the carrier must NOT be keyed so broadly that it flips OTHER kernels to persist_interleaved
WITHOUT evidence. The shapes that are `row_reread AND looped` (the naive key) include: CE wide, softmax wide
(>240KiB), rms/ln wide robustness (>240KiB), welford wide (Band-C looped apply). I have decomp evidence ONLY for
CE. A `row_reread AND not persistent` gate would ALSO flip softmax/welford/rms-ln wide -> persist_interleaved,
which is UNMEASURED and could regress them (run-1 found flat dominant 1.5-4x on narrow-forward; the wide-looped
behavior of the non-CE re-read kernels is unknown).

=> EDIT-PID fact must be validated on softmax-wide + welford-wide + rms/ln-wide BEFORE shipping (does
persist_interleaved help, hurt, or no-op them?). Two outcomes:
  (a) persist_interleaved helps ALL wide-looped re-read kernels -> the fact IS `row_reread AND looped AND wide`
      (a clean workload regime, generalizes).
  (b) it helps ONLY CE -> the fact needs a FINER property distinguishing CE (e.g. num_load>=3 / the multi-input
      structure / a scalar gather present) -- and I must be careful that's not kernel-identity-smuggling. The
      principled CE-specific property would have to be a real workload feature (the multi-pass multi-load with a
      gather), justified physically, not "CE has 3 loads so fence it."
This is the SAME discipline as the cap gate (row_reread, not num_load). DON'T design the fact until the
multi-kernel persist_interleaved A/B says (a) or (b). GPU queue after CE-consistency oracles: persist_interleaved
A/B on softmax(512,131072)/(1024,65536) + welford(65536,4096)/(32768,8192) + rms/ln(1,131072), each
matched-lever {seed vs seed+pid-cluster}, do_bench. Then the fact is evidence-based.

Also: sm_multiplier=32/maxnreg=64 are CONSTANTS in the oracle bundle -- are they shape-portable or do they need
their own derivation? The decomp used the CE(4096,98304) values; the CE-consistency full oracles (running) will
show if 8192x128256 + 2048x256000 pick the SAME sm_mult/maxnreg or different (-> whether they're constants or
need a workload key). If they vary, the seed needs a principled rule for them, not the single observed value.

## 2026-06-03 — EDIT-PID plumbing (non-GPU): pid_type gates sm_mult/maxnreg (the run-2 trap mechanism)

code-read (config_spec.py normalize, 1226-1265): `num_sm_multiplier` + `maxnreg` are DROPPED by normalize when
`pid_type in (flat, xyz)` (raise for user configs, silent pop for autotuner `_fix_invalid`). So they ONLY
survive when pid_type is persistent_blocked/persistent_interleaved. This is the MECHANISM behind the run-2 trap
(a seed setting sm_mult without persistent pid -> silently dropped) and why the decomp's 5a round-trip-survival
mattered: the carrier bundle survived BECAUSE pid_type was persistent_interleaved. => the EDIT-PID branch must
set all THREE together {pid_type='persistent_interleaved', num_sm_multiplier=K, maxnreg=64}; setting sm_mult/
maxnreg alone is a no-op.

Current seed hardcodes pid_type='flat' (T1 triton.py:587, Band-C:670, T2:721) -- a PRINCIPLED constant (run-1
rejected persistent for narrow-forward, flat dominates 1.5-4x). EDIT-PID overrides 'flat'->'persistent_
interleaved' ONLY on the validated looped-reread regime (pending the multi-kernel A/B + carrier-consistency
oracle). sm_multiplier is NOISY in the full-oracle search (saw 1, 32, 64 across generations on 8192x128256) ->
likely a weakly-determined knob; the seed should set a single PRINCIPLED value (an SM-count-derived default),
NOT chase the per-shape oracle pick (which is search noise). The carrier-consistency oracles (running) +
multi-kernel A/B will pin: (a) is persist_interleaved consistent across wide CE (early signal: YES -- 8192x128256
gen-2 best IS persist_interleaved + my Rule B eviction), (b) what sm_mult value to seed, (c) does it generalize
to softmax/welford/rms-ln wide or is it CE-specific.
