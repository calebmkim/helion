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

### Tried / rejected
- (none yet — diagnostic pass)

### Open hypotheses (to test against the FRESH oracle in Phase 2)
- H-small-MN: small-M and small-N shapes have a seedable warps/M-block gap (perf-dig hint, reps=1 — confirm).
- H-nonpow2: non-pow2-N shapes have a seedable blocking gap (perf-dig hint — confirm).
- H-singlerow: M=1 huge-N may be a real source ceiling (oracle≈seed<tc) OR a seedable gap — oracle decides.
  (NOTE: M=1 huge-N lives in `robustness` = correctness-only; not a train perf target. The train analog is
  tiny-M-large-M variation rows — check what train actually covers.)
- H-welford-wideN: welford wide-N looped-apply ceiling claim — re-litigate vs fresh oracle (was OOM, suspect).

### Current champion
- Inherited run-2 `TritonReductionHeuristic` (unmodified). No run-3 heuristic edits yet.
