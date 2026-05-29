# Lab Notebook — Reduction Autotuner Heuristics

> The DURABLE source of truth for the hill-climb. A fresh worker reads this to continue losslessly.
> Maintained by the worker (decisions + empirical why; tried-and-rejected + why; open hypotheses;
> champion). The hub appends gate verdicts. Keep it current at every clean iteration boundary.

## Champion (current best heuristic)
- **v1 `triton_reduction_tile`** (Step 2 first heuristic). Bare-seed **G_rms_norm = 0.983**
  (kernel-only do_bench, fp32, GPU2), vs un-seeded `default_config` baseline **G_default = 0.908**.
  Proposed champion: beats the no-seed baseline by ~8% aggregate, no per-shape regression vs default
  (worst shape (2048,2048) TIES default). Code: `helion/_compiler/autotuner_heuristics/triton.py`
  `TritonReductionHeuristic`. NOT yet referee-confirmed / auditor-passed (HELPER_REQUEST pending).

## Objective
- Product A: maximize `O = geomean_k G_k`, `G_k = geomean over kernel k's in-sample shapes of
  (tc_default_latency / seed_latency)`. Accept iff O improves AND gates pass (correctness; seed used;
  no active kernel's referee-confirmed G_k regresses >10% vs champion).
- Product B (every 5 iters): seeded vs unseeded quick-autotune convergence curve.

## Active kernels (curriculum)
- Active: rms_norm (fwd) — T1, Band A. Widen to: sum, layer_norm-fwd, softmax, long_sum (Band A);
  kl_div, jsd (Band B); welford (Band C). Forward only for now; defer backward (Band D).

## Track classification (T1 rolled / T2 manual / out-of-scope) — per kernel
- **rms_norm_fwd: T1** (rollable rdim; `reduction_loops` has 1 entry; `reduction_facts` has 1 entry).
  Confirmed: single block_size (M-axis) + single reduction_loop, no matmul_facts.

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
  Branch: `rnumel*itemsize <= PERSIST_MAX_BYTES (=65536, i.e. 16384 fp32 elems)` → `reduction_loops=[None]`
  (persistent, single-pass, no `for roffset` loop); else looped chunk `LOOPED_CHUNK=4096`.
  WHY: the un-seeded Triton `default_config` goes LOOPED (chunk=min(next_pow2,4096)) once
  `rnumel > reduction_loop_force_threshold` (None on Triton → effectively persistent up to 4096, looped
  4096 above). Empirically that looped default LOSES big at the wide shapes:
  (2048,16384) G_default=0.70, (8192,8192) 0.75, (2048,8192) 0.81, (4096,7168) 0.83 — while tc/Helion-max
  keep the reduction PERSISTENT (whole contiguous fp32 row in regs/SMEM). Seeding persistent recovers
  all of these to G≈0.99. Threshold in BYTES (via itemsize) so it generalizes across dtypes.
- **num_warps scales with rnumel:** `<=1024 → 4`, `<=4096 → 8`, else `16`. WHY: wider rows give each warp
  more independent lane work + more memory traffic to overlap; too few warps under-occupy the SM on the
  bandwidth-bound persistent sweep. Power-of-2 (NumWarpsFragment). To be A/B'd vs oracle.
- **num_stages=1.** Persistent reduction has a single inner pass; default is 1; no pipelined loop to
  multi-buffer. (Looped fallback may want >1 later — open.)
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
| (2048,2048) | persist | 8 | 0.87* | 0.88* |
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

\* (2048,2048) ANOMALY: in the full 13-shape sweep BOTH seed (~19.8us) and default (~19.6us) read slow
  → G≈0.87 for both. But an ISOLATED probe of the exact same seed config reads 13.5us → **G=1.32**. So
  the 0.87 is an in-process MEASUREMENT-CONTEXT artifact (not a property of the config, and not a
  regression my seed introduces — seed TIES default there). Likely the seed+default+tc kernels for that
  one shape contend / a dynamo-state interaction. Flagged for the results-referee; true G_rms_norm is
  likely HIGHER than 0.983 once this is resolved (re-measuring each shape in a fresh process).

## Tried and rejected (with why it failed)
- _Gate `M-floor <= 1` (CuTe template's): REJECTED — silently dropped (32768,*) shapes whose autotuner_min
  is 2. Replaced with "accept any floor, seed block at the floor"._

## Open hypotheses
- **(2048,2048) measurement artifact** — re-measure each shape in a fresh subprocess (results-referee).
  If the artifact clears, G_rms_norm rises.
- **num_warps tuning** — isolated (2048,2048) probe: warps=8 persist best (13.47us) vs warps=2 (13.66)
  /4 (13.82)/16 (14.08). Small spread; current breakpoints look reasonable but the oracle field-diff
  will say what max-autotune picks per shape.
- **LOOPED_CHUNK / num_stages for >PERSIST_MAX rows** — no rms_norm in-sample shape currently exceeds
  16384 elems (all persistent), so the looped branch is UNTESTED on rms_norm. Will matter for long_sum
  (Band A, huge rnumel) and as we widen. Need a shape that triggers it.
- **PERSIST_MAX_BYTES=65536** — is 16384 fp32 elems really the persistent ceiling on H100? All in-sample
  rms_norm rows are <=16384 so we never cross it here. Probe a synthetic wide row (e.g. 32768/65536) to
  find where persistent stops winning, to set this threshold by evidence rather than by the in-sample max.

## Oracle cache pointers
- See `_lab/ledger.json` `oracle_cache`. Field-diff script: `_lab/harness/oracle_field_diff.py`
  (Helion effort=full winning config vs seed for 3 representative shapes). [oracle run in progress]
