# Matmul-family perf characterization — decisions log (B200 / sm100)

Run date: 2026-07-14. Autonomous overnight run. Worktree `/home/dev/helion-matmul-b200`
(branch `pr-stack-v2`, heuristic file byte-identical to PR #3007 `origin/calebmkim/stack/28`).
`helion.__file__` asserted under the worktree in every script.

This file records every judgment call I made without asking, per the user's instruction
("use best judgement, write it down, I'll review later").

---

## D1 (CRITICAL — deviation from the letter of the spec). Seed arm ≠ `compiler_seed_configs[0]`.

The task text (§1 arms table, §4) says: `seed = bind(...).config_spec.compiler_seed_configs[0]`.
**On B200 that index-0 entry is the TABLE's config, NOT the formula's** — so taking it
literally would benchmark the wrong arm and silently invert the whole headline.

Why: the loader (`autotuner_heuristics/__init__.py::compiler_seed_configs`) walks heuristics in
registration order and `.extend()`s each eligible heuristic's ranked configs into one flat list.
`TritonB200MatmulHeuristic` (the TABLE) is registered *before* `TritonB200FormulaMatmulHeuristic`
(the FORMULA). On in-bucket fp16/bf16 matmul BOTH are eligible, so the flat list is
`[table_cfg, formula_cfg, ...]` → `compiler_seed_configs[0]` = the TABLE.

The FORMULA is planted as the **promoted default**: `env.config_spec.compiler_default_config`
(the loader sets it from the *last* `promote_seed_to_default=True` heuristic's rank-0). Measured
directly across regimes:

| shape | compiler_seed_configs[0] | compiler_default_config (promoted) | FML.get_seed_config |
|---|---|---|---|
| matmul 4096³ (in-bucket) | `[128,128,64]` = TABLE | `[256,256,32]` = FORMULA | `[256,256,32]` |
| matmul 8192³ (table declines) | `[256,256,32]` = FORMULA | `[256,256,32]` | `[256,256,32]` |
| matmul 1024³ (table point) | `[128,64,64]` = TABLE | `[128,64,128]` = FORMULA | `[128,64,128]` |
| matmul 4096,2048,2048 | `[256,256,32] s7` = TABLE | `[256,256,32] s6` = FORMULA | `[256,256,32] s6` |

**Resolution — faithful extraction used by the harness:**
- `seed` arm  = `TritonB200FormulaMatmulHeuristic.get_seed_config(env, dir)` — the formula, verified
  == `compiler_default_config` on every cell (the PR promotes it, so the promoted default IS the
  formula seed; `cd==fml` was True for all 4 kernels probed). `seed_fired` = (cfg is not None); the
  formula fired on 100% of probed cells (matmul/fp8/bmm/mamba).
- `helion_default` arm (the incumbent being beaten) = `TritonB200MatmulHeuristic.get_seed_config`
  when it fires (`default_source=table`), ELSE `config_spec._base_default_config()` (the
  `~[16,16,16]` base, `default_source=base`). This matches the spec's §1 definition of the B200
  helion_default exactly — it just is NOT the same object as `compiler_seed_configs[0]`.
- `tc_max_autotune` arm = `torch.compile(op, mode="max-autotune-no-cudagraphs")`.

This is faithful to the spec's *intent* (§1/§8 both define the seed as the formula and the
helion_default as the incumbent table-or-base) — I am fixing the extraction expression, not
changing which arms are measured. Flagging loudly because it is a literal-vs-intent deviation.

## D2. L2 flush = 512 MiB (4× the 126.5 MiB B200 L2), for BOTH methods.
triton's built-in `do_bench` hardcodes a 256 MiB flush (`get_empty_cache_for_benchmark`) — only
~2× B200 L2, the exact "256 MB may not evict on B200" trap §0/§6 warns about. So I do NOT use
triton's do_bench; I implement M2 as a manual flush(512MiB)+launch+CUDA-event loop, and M1 as a
graph-diff where BOTH the [flush+kernel] and [flush] graphs contain the 512 MiB zero_(). Cold-L2
confirmed real in step-0 (implied BW 1.34 TB/s on 2048³ ≪ any L2 bandwidth).

## D3. Two-phase per cell to survive the `helion_default` ptxas-hang / OOM.
Phase A: compile each arm's config in an ISOLATED subprocess (setsid, 120 s timeout, killpg on
expiry) — records status {ok,compile_fail,oom,timeout}. This is where the `[16,16,16]`-on-a-huge-
GEMM ptxas hang or OOM is caught as a data point without wedging the run. Triton/inductor disk
caches are shared (verified: 2nd process compile 0.69s→0.36s), so Phase B (interleaved M1+M2
timing, one warm process per (kernel,shape)) re-uses the cached cubin and never re-invokes ptxas →
cannot hang. Only arms that survived Phase A are timed in Phase B.

## D4. R rounds. R=7 default; R=15 for shapes whose seed median < 25 µs (noise floor). Decided
per-shape after a quick timing probe in the timing process (the JSON `_meta` sets the threshold).

## D5. Accuracy gate before timing, fp32 reference (`x.float()@y.float()`), rel-tol 5e-2 ("within
bf16 rounding"), near-zero-safe. Acc-fail arms excluded from geomeans and surfaced. fp8 reference
is the fp32 recompute of the dequantized operands (scale=1.0). mamba reference = `ref_chunk_state`.

## D6. Fairness locks every run: allow_tf32=False, allow_bf16_reduced_precision_reduction=False,
fresh randn per shape identical across arms, fwd-only, dynamo.reset() per shape, one fresh process
per kernel. fp8 tc = `_scaled_mm(use_fast_accum=True)` scale=1.0 (record fast_accum=False too).

## D7. static_shapes=True (matches the shipped examples; matmul/bmm/fp8/mamba decorators already
set it or default to it). Disclosed as a specialization edge in the writeup (§8).

## D8. Peaks for %-of-peak: B200 dense bf16 = 2250 TFLOP/s, fp8 = 4500 TFLOP/s (per SKU spec).

## D9. Harness-validation findings during bring-up (all fixed before the real sweep)
- **[CRITICAL BUG, fixed] Late-binding closure in time_cell arm construction.** The seed
  and helion_default lambdas both closed over the same loop variable `c`; after `c` was
  reassigned to the default's compiled kernel, the SEED lambda also invoked the DEFAULT
  kernel. Symptom: seed measured 128 us == default 125 us in the sweep, but 98.5 us when
  timed alone. This would have made every multi-arm cell report seed≈default (G_vs_tc and
  xD_vs_default both wrong). Fixed via `make_call(compiled, args)` default-arg capture.
  After fix: matmul 4096³ seed=98.5us, G_vs_tc=0.857, xD_vs_default=1.27 (=formula beats
  the incumbent table 1.27× on its turf — consistent with PR #3007's 1.19-1.27× claim and
  my prior independent verification's ~0.87 seed-vs-cuBLAS). THIS is why bring-up matters.
- **[method calibration] M2 must use do_bench run-ahead semantics, not inner-sync.** An
  early M2 synced after every launch, inflating host overhead 2-4×. Fixed to enqueue the
  whole round then one sync (faithful to triton.testing.do_bench). After: on the long
  4096³ kernel M1≈M2 within 3%; on short 2048³ M2 exceeds M1 ~20-36% (largest on the tc
  arm) — the EXPECTED short-kernel host-overhead divergence, not a bug.
- **[method calibration] M1 batches `inner=10` graph replays per event window** ("replay-
  averaged") so a ~15 us kernel riding on a ~64 us flush is measurable; single-replay
  events were too noisy (gave a spurious 2× M1/M2 gap on short shapes).
- Independent calibration confirmed the tc arm IS cuBLAS: eager torch.matmul and
  torch.compile-max-autotune agree within 0.6% at 4096³ (both nvjet cuBLASLt).

## D10. Run completed 48/48. All four kernels validated end-to-end (first cell each) BEFORE
the full sweep; all 3 arms `ok` on all 48 cells; 0 OOM / 0 ptxas-hang / 0 compile-fail /
0 accuracy-fail. Independent recompute confirmed all 288 arm-medians + 192 ratio-medians match
the raw per-round `t_us` arrays exactly (summaries derived, not stored-in-place). Headlines:
table head-to-head geo xD=1.29× (6 cells), base coverage geo xD=18.7× (42 cells), GEMM-vs-cuBLAS
geo G_vs_tc=0.68, mamba-vs-Triton geo 1.61×. See summary.md + STEP0_GATE.md.

## D11. NOTE for reviewer — a possibly-surprising observation worth a human eye:
On B200 the catastrophic `[16,16,16]` base default did NOT crash on ANY of the 42 base cells
(even 135 ms vocab GEMMs) — the 183 GB HBM absorbs the register/occupancy pathology that would
OOM or hang ptxas on an H100 or smaller SKU. So the spec's expected `oom`/`timeout` data points
did not materialize HERE; they remain a real risk on other hardware. I kept the full
subprocess-timeout+killpg machinery anyway (it's the correct portable design), it just never
fired. This is a hardware-delta finding, not a harness gap.

## D12. Adversarial verification (4 skeptic lenses + synthesis) — COMPLETE, verdict "trustworthy
with 3 framing edits." timing=sound(0.82), coldL2=cold(0.88), ratios=faithful(0.82),
cuBLAS-arm=mixed. No lens flagged a headline-changing issue. Synthesis + per-lens verdicts folded
into summary.md "Trust & verification". The ONE real must-fix it found:

- **tc "winner" label was hardcoded cuBLAS, not per-cell verified.** I re-profiled every GEMM
  family with the CUDA profiler (`results/tc_backend_probe.md`): 39/40 GEMM cells genuinely run
  cuBLAS/cuBLASLt (nvjet_sm100) — incl. ALL fp8 (even M=1) and ALL bmm. EXACTLY ONE cell,
  matmul[1,4096,4096] (M=1 decode), runs a Triton GEMV instead. So its G_vs_tc=1.40 is a win over
  Triton, NOT cuBLAS — corrected + footnoted in summary.md. (The skeptic OVER-claimed M=32 and
  small fp8/bmm cells were also Triton; profiling disproved that — they're cuBLAS.) Headline
  robustness: dropping that cell moves all-GEMM geomean 0.684→0.672 (the "wrong" way, i.e. true
  cuBLAS gap marginally larger), so "~0.68× of cuBLAS" holds. Harness winner label also softened
  to "expected; profiler-verified except matmul M=1 decode".
- Other disclosed-not-fatal items: mamba 1.6× is vs a naive-einsum Triton ref (labeled); M2 blind
  <15µs so short-cell M1 has no cross-check (flagged noise_floor); fp8 fast_accum=True asymmetry is
  CONSERVATIVE against the seed; table ratios are median-of-per-round-ratios (footnoted).

## Open items / notes for review
- fp32 dtype deliberately NOT measured (curriculum primaries are bf16 + fp8-e4m3). The fp32 seed
  is a known ~0.24× codegen ceiling vs TF32 cuBLAS (from [[helion-matmul-b200-pr3007-verification]]).
- fp8 fast_accum=False secondary arm (promised as optional in D6/spec §1) NOT run — a weaker-bar
  variant; the fast_accum=True numbers already reported are the stronger, production bar.
</content>
