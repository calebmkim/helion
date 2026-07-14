# Adversarial audit — findings & dispositions

After the sweep, a 4-dimension adversarial audit (harness-methodology, stats-derivation,
honesty-framing, coverage-completeness) was run, each finding independently
verified by a skeptic agent (default-refute). 3 issues were CONFIRMED real (2 are the
same root cause seen from two dimensions). All have been ADDRESSED. Stats-derivation and
honesty-framing raised no confirmed blocker/major issues (medians/geomeans/ratios/TFLOP-s
recomputed-from-raw and matched; framing judged faithful).

## Confirmed #1 (major) — tc_max_autotune backend `winner` was not recorded

**Finding:** §1/§4/§5(d)/§7 all require logging the max-autotune-selected backend per
shape; the original harness built the `torch.compile` callables but never captured which
kernel won, yet `summary.md` asserted "cuBLAS/cuBLASLt" as fact.

**Disposition: FIXED.** Added `mmperf.py backend` subcommand (captures the inductor
`AlgorithmSelectorCache` autotune table from stderr under a fresh inductor cache +
`autotune_in_subproc=False`; the `100.0%` line is the winner) and `collect_backends.py`
(runs it over all 48 cells, merges `winner`/`winner_kind` into every record's tc arm).
Result — the §5(d) claim is now data-backed:
- matmul 21/22 → `mm` (cuBLAS); 1024³ → a Triton template won.
- fp8_gemm 10/10 → `_scaled_mm` (cuBLASLt).
- bmm 6/8 → `bmm` (cuBLAS batched); 2 small/transposed → Triton.
- mamba2 8/8 → `triton_bmm_*` (Triton — no cuBLAS analog, as expected).
- matmul `[1,4096,4096]` (M=1) → aten GEMV, no autotune template (labeled as such).

## Confirmed #2 (major) — M1 median inflated/noisy on the longest kernels; §5 gate breach undisclosed

**Finding (as refined by the verifier):** M1 (cudagraph) is supposed to be ≤ M2 (it
strips host overhead), but **3 of 48 cells show M1 > M2**, worst = matmul 8191³ prime
(M1 6039 µs vs M2 5481 µs, +10.2%) — which is the report's headline "worst cell"
(`G_vs_tc`=0.276). On that prime shape M1 is the *noisier* arm (σ≈4% vs M2 σ≈1%) and its
median is pulled up by a few slow replay rounds, so the shipped point was ~M1-jitter-
pessimistic. Separately, the §5 "M1/M2 agree within a few %" gate is breached on several
long kernels (4096³ 9%, 8192³ 10%) and the report did not disclose it. The verifier
corrected the *mechanism*: it is NOT flush-subtraction noise (the 256 MiB flush is only
~1–3% of a multi-ms kernel) — it is genuine run-to-run jitter of the irregular tail-
masked prime WGMMA kernel plus median-of-few-rounds sensitivity.

**Disposition: ADDRESSED (measure + disclose, no silent change).**
- Re-timed 8191³ at R=15 — the M1 jitter **reproduces** (M1 median ≈6070 σ≈4%, M2 ≈5450
  σ≈1%; M1 *min* 5588 ≈ M2 median). Confirmed it's a property of that kernel, not a
  miscalibration (4096³/8192³ M1 are tight to ~1%).
- `summary.md` now (a) reports the 8191³ result as a **range 0.27–0.32** (M1–M2), not a
  false-precision 0.276; (b) flags all 3 M1>M2 cells with ⚑ in the per-shape tables;
  (c) has a dedicated **"Harness caveat — §5 gate" box** honestly disclosing the breach,
  splitting it into the benign host-overhead offset (most cells) vs the genuine M1 noise
  (the one prime cell); (d) notes the adversarial geomean uses the *conservative* M1
  value (using M2 would help the seed), so there is no upward cherry-picking.
- The qualitative conclusion (prime tail-mask = the seed's worst regime, ~3.1–3.8×
  slower than cuBLAS) is unchanged.

## Confirmed #3 (minor) — same as #2's winner-logging, from the coverage dimension

Duplicate of #1; resolved by the same fix.

## Non-issues (raised, verified not-a-defect)

- Extra JSONL provenance fields beyond the §7 minimal schema (`heuristics_fired`,
  `matmul_facts`, `default_cfg`, `seed_est_us`, `prov`, `gpu`) — additive, useful, all
  §7-required fields present. Kept.
- R-bump (≤~30 µs guard on seed_est) vs noise_floor flag (<25 µs) using slightly
  different thresholds — a labeling nit; both recorded per cell, no data wrong.
