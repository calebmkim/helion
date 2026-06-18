# Autotuner Timing "Bias" for tiny-N/large-M rms_norm — VERDICT: NO BUG

**Agent:** harness-integrity (autotuner-timing localization). **Date:** 2026-05-28.
**GPU:** H100 #3 (probes) and #2 (parallel grid), both verified idle (≤7 MiB, 0%) before/between
trusted timings; re-checked across the run. GPU 1 picked up a co-tenant mid-run — avoided.
**Interpreter:** `/home/calebkim/.conda/envs/helion/bin/python`. helion asserted = worktree in every script.

---

## VERDICT

**The 1183us number is real. The 73.6us number is ALSO real. They are timing DIFFERENT configs.**
There is **no autotuner measurement bug** and the autotuner's do_bench is **not biased**. The anomaly
was a **mis-attribution in the oracle field-diff**, not a timing error.

- `{persistent, block_sizes=[1], num_warps=32}` genuinely runs at **~1174us** — every timing method
  agrees (fair triton do_bench, autotuner `do_bench`, autotuner `interleaved_bench`), all <1% apart.
- The autotuner's recorded **73.6us "w32" config was NOT block=1** — it was `num_warps=32` paired with
  a **large M-block** (`block_sizes=[32]` or `[128]`, confirmed in the autotune CSV). w32 is only
  catastrophic at block=1; with block ≥ 8 it is fine (~36-41us).
- The oracle field-diff reconstructed the winner as `{block=1, w32}` and then "fair re-benched" that
  **wrong pairing**, got 1174us, and wrongly concluded "the autotuner ranked w32 best but it's a 33x
  artifact." The autotuner never tested block=1 (see below) and never ranked block=1/w32 as best.

## ROOT CAUSE (physical)

block=1 ⇒ each program normalizes ONE row of N=256 fp32 (1 KiB). With num_warps=32 that's 1024 threads
for 256 elements (≤4 elems/thread) across a 32768-program persistent grid — extreme under-utilization +
occupancy/scheduling thrash. Larger M-blocks pack many rows per program so the 32 warps have real work.
**warps and block_size are coupled**: high warps REQUIRE a large block. Judging warps with block held at
1 (as the field-diff did) is the trap.

## EVIDENCE 1 — fair vs autotuner timing agree (the timing primitive is NOT biased)

`(32768,256)` fp32, persistent, **block_sizes=[1]**, configs=[seed] (seed used verbatim):

| warps | persist | correct max_abs | kernels/call | FAIR do_bench (us) | AUTOTUNER do_bench (us) | interleaved_bench (us) | G_fair |
|---|---|---|---|---|---|---|---|
| 4  | yes | 4.768e-07 | 1.0 |   36.5 |   36.5 |   36.4–37.0 | 0.935 |
| 8  | yes | 4.768e-07 | 1.0 |   42.8 |   43.2 |   42.9–43.7 | 0.797 |
| 16 | yes | 4.768e-07 | 1.0 |  569.8 |  570.1 |  569.9–570.5 | 0.060 |
| 32 | yes | 4.768e-07 | 1.0 | 1174.2 | 1173.9 | 1173.4–1174.5 | 0.029 |

tc-default fair median = 34.1us (G basis). All three timing methods within <1%. **At block=1, w32 IS
1174us** — and the autotuner would measure exactly that if asked. So the autotuner do_bench is honest.
(autotuner do_bench = warmup=1/rep=50/median, CUDA events, L2 flush; interleaved_bench tested at
repeat=50/200/1000 — all agree.)

## EVIDENCE 2 — the smoking gun: warps × block_size grid (fair do_bench, us)

`(32768,256)` fp32 persistent:

| block \ warps | w4 | w16 | w32 |
|---|---|---|---|
| **1**   |  36.1 |  570.2 | **1174.4** |
| 2   |  36.9 |   46.4 |   86.6 |
| 8   |  34.7 |   36.1 |   40.6 |
| 32  |  34.8 |   36.3 | **36.7** |
| 128 | 213.9 |   36.4 | **35.9** |

w32 is catastrophic ONLY at block=1; at block≥8 it's ~36-41us. The **73.6us** the oracle attributed to
"w32" is exactly the regime of w32 + large block (CSV: 0.073568ms at `num_warps=32, block_sizes=[128]`).
Both the 73.6us and the 1174us are correct measurements of DIFFERENT configs. (All cells correct to
max_abs=5e-07; block_sizes used verbatim, no normalization override.)

## EVIDENCE 3 — the real full-effort autotune (LFBOTreeSearch, fresh run, GPU3)

From the live autotune CSV (`HELION_AUTOTUNE_LOG`, the autotuner's own recorded do_bench perf):

- **Fastest config per num_warps (recorded `ok` perf, ms):**
  w1=0.03098, w2=0.03109, w4=0.03270, w8=0.03475, w16=0.03488, **w32=0.03430**.
  → The autotuner's OWN numbers rank **low warps (w1/w2) best (~31us)**, NOT w32. The premise "the
  autotuner ranked w32 best" was itself a misreading of the field-diff, not of the autotuner.
- **`block_sizes=[1]` appears 0 times in the entire search.** Helion's `raise_grid_block_minimums`
  raises the M-block floor to 2 for 32768-row shapes, so block=1 is OUTSIDE the search space. The
  autotuner literally cannot have selected block=1/w32. The fastest w32 entries are block=[32]/[128],
  looped reductions — all FAIR-rebench at ~36-74us, matching the grid.
- Search converges toward `block_sizes=[2-4], num_warps=1-2, reduction_loops=[32-128]` (LOOPED), ~31us,
  `pid_type=persistent_interleaved`, mixed tensor_descriptor indexing.

## CORRECTNESS of the w32 config

`{persistent, block=1, num_warps=32}` is **CORRECT, not degenerate** — max_abs = 4.768e-07 vs the fp32
PyTorch reference (bit-for-bit identical to w4/w8/w16). It is genuinely slow, not silently no-op'ing.
The 1174us is real compute that is just badly under-occupied. (Kernel count = 1/call, profiler-confirmed;
the 0.95/call at high warps is a profiler cycle-boundary artifact, not a missing launch.)

## IMPLICATIONS

**(a) Is the oracle cache a trustworthy answer key for tiny-N/large-M? — PARTIALLY; the recorded
`(32768,256)` "winner" is contaminated by a reconstruction error, not by the autotuner.** The autotuner
itself is fine (its do_bench is unbiased and it correctly avoids block=1 and doesn't rank w32 best). What
is wrong is the **field-diff's flattening** of a multi-field winner into `{persistent, block=1, w32}` and
its "fair re-bench" of that fabricated config. Fix: the field-diff must re-bench the FULL winning config
(all fields: block_sizes, reduction_loops, num_warps, num_stages, pid_type, indexing) verbatim — never
substitute block=1 or hold one lever fixed while varying another. The true fair optimum at (32768,256) is
~31us (G≈1.10 vs tc 34us) at low warps + small-but-≥2 block + looped reduction.

**(b) Does this bias Product B (seeding)? — NO, not via the autotuner timing.** A good low-warp seed will
be timed correctly by the autotuner (same do_bench, no bias). The real Product-B risk is the OPPOSITE of
the worry: **warps and block must be seeded as a COUPLED pair.** A seed of `num_warps=32, block=1` would
be (correctly) timed as terrible; a seed of high-warps must come with a large block. The current heuristic
seeds LOW warps for tiny-N (good, robust) — safe. Do NOT add a "match the oracle's w32" rule; that would
only help if also paired with a large block, and low-warps already wins.

**(c) Genuine Helion autotuner bug worth flagging? — NO timing bug.** do_bench / interleaved_bench /
benchmark_provider are all sound (event-based, L2 flush, adaptive repeat that does NOT hide slow configs:
estimate_ms scales n_repeat correctly even at 1174us). The only "bug" is in OUR `oracle_field_diff.py`
(lever-isolated reconstruction). Minor upstream observation (not a bug): the search wastes effort on
block=1-equivalent under-occupied high-warp configs, but `raise_grid_block_minimums` already prunes the
worst (block=1) case for large-M, which is the correct guard.

## NEXT EXPERIMENT (if deeper digging wanted)

Let the in-flight full LFBO run finish (GPU3, `/tmp/autotune_log_32768_256.csv`) and record the FINAL
winner + its verbatim fair re-bench to replace the contaminated `oracle_cache.(32768,256)` entry. Expect
low-warp/small-block/looped ~31us, G≈1.05-1.10. Then audit (32768,1024) the same way (its ledger
`fair_ab_warps` was generated with the same block=1-implicit method and may share the artifact).

## RECEIPT

- GPU 3 (probes/full run) + GPU 2 (grid), both idle pre-measurement (nvidia-smi). fp32 throughout.
- Env: `cd /home/calebkim/helion-new-heuristics/wt-reduction && CUDA_VISIBLE_DEVICES=<2|3>
  PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction /home/calebkim/.conda/envs/helion/bin/python <script>`
- Scripts (in `_lab/harness/`):
  - `autotuner_timing_probe.py` — fair vs autotuner do_bench, correctness, kernel-count (Evidence 1).
  - `autotuner_interleaved_probe.py` — interleaved_bench (population/rebenchmark path) at repeat 50/200/1000.
  - `warp_block_probe.py` — the warps×block grid (Evidence 2).
  - `autotuner_full_repro.py` — real `effort=full` autotune + CSV scan (Evidence 3); `HELION_AUTOTUNE_LOG=/tmp/autotune_log_32768_256.csv`.
- Config under test: `helion.Config(reduction_loops=[None], block_sizes=[b], num_warps=w, num_stages=1)`
  via `helion.kernel(rms_norm_fwd.fn, configs=[seed])` (len==1 short-circuit → seed used verbatim).
- Raw: see tables above. CSV facts: block=1 count = 0; fastest w32 ok = 0.034304ms; 0.073568ms entry =
  w32/block=[128]; overall fastest = 0.030976ms (w1, block=[4], reduction_loops=[128]).
