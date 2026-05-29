# results-referee VERDICT: v2 TritonReductionHeuristic

**VERDICT: ACCEPT** (independent re-measurement, 2026-05-29, GPU 2/H100, fp32)

GPU: CUDA_VISIBLE_DEVICES=2 (verified idle 4MiB/0% before, during, after; GPU 0
had an unrelated co-tenant on GPU 0 only — CVD=2 isolates). Fresh subprocess PER
shape. torch.manual_seed=0 (+stability re-run seed=123). do_bench median, N=9
(sum/rms), N=15 (long_sum tiny-lat), N=11 (long_sum bigM). G = tc_default_lat/seed_lat.
helion confirmed worktree: .../wt-reduction/helion/__init__.py. heuristics_fired=
["triton_reduction_tile"] every shape. autotune_ran=False every shape (no CSV).
seed_used=True every shape (codegen persistent-vs-looped + num_warps match normalized).

## Exact commands
    cd /home/calebkim/helion-new-heuristics/wt-reduction && \
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction \
    /home/calebkim/.conda/envs/helion/bin/python \
    _lab/harness/referee_verify.py --kernel <K> --m <M> --n <N> --n-runs <N> --seed 0
    # orchestrated per-shape by _lab/harness/referee_run.sh {long_sum,sum,rms_norm,long_sum_bigM}

Seed obtained via compiler_seed_configs(bound.env, bound.host_function.device_ir)
(the REGISTERED heuristic path) — asserted exactly 1 seed; run bare via
helion.kernel(fn.fn, configs=[seed]) (len==1 short-circuit, no autotune; eager
normalize so a bad seed RAISES).

## long_sum (CLAIM: G=1.018 BIG WIN vs default 0.311) — CONFIRMED
shape        codegen     seed_us(spread)  dflt_us  tc_us  G_seed  G_dflt
(1,32768)    looped/w32  5.82 (5.44-6.05) 11.52   7.84   1.346   0.681
(2,65536)    looped/w32  6.91 (6.88-7.42) 20.93   7.90   1.144   0.378
(4,130000)   looped/w32 10.30 (10.3-10.8) 49.15   8.96   0.870   0.182
(8,131072)   looped/w32 10.27 (10.2-10.8) 38.40  10.40   1.012   0.271
(16,262144)  looped/w32 19.71 (19.3-19.8) 74.85  16.90   0.857   0.226
GEOMEAN G_seed = 1.030   G_default = 0.310   seed-over-default = 3.32x
seed always BEATS default per-shape (worst seed/default = 1.98x). Worker's 1.018 is
within noise of my 1.030 (tiny abs lat). Stability re-run (seed=123): (1,32768)
G=1.309, (16,262144) G=0.853 — matches.
ROBUSTNESS (lift M to raise latency off the noise floor, same N):
(256,131072) 76.5us G=1.042 ; (512,131072) 133us G=1.025 ; (256,262144) 133us
G=1.103. Win persists at non-noisy latencies => NOT a tiny-latency artifact.

## sum (CLAIM: WASH G~0.931, no regression) — CONFIRMED
GEOMEAN G_seed = 0.937   G_default = 0.929 (worker 0.931/0.933 within noise).
Worst per-shape seed-vs-default = 0.9996 (i.e. NO regression anywhere).
Note: at (2048,16384) & (4096,5120) seed=PERSISTENT/w16 while default=LOOPED/4096
(real behavioral difference, correctly distinguished) yet still a wash.

## rms_norm (CLAIM: no regression, G~0.982) — CONFIRMED
4-shape spot-check GEOMEAN G_seed = 0.977 (worker's 0.982 / champion 0.979 within
noise). (2048,16384) 0.992, (4096,5120) 0.984, (8192,4096) 1.004, (32768,256) 0.930.
All <=64KiB rows => seed PERSISTENT, UNCHANGED by the threshold bump (confirmed:
seed_codegen=persistent at every rms shape; only default differs at 16384 where
default loops). (32768,256) seed mildly regresses ~6.5% vs default (block=2/w4 vs
default block=16/w4) — pre-existing tiny-N/large-M under-serve, within -10% backstop,
NOT introduced by v2.

## Correctness — PASS all shapes/kernels
allclose(rtol=1e-3, atol=1e-3) PASSES every shape, every kernel (seed, default, tc).
Worker's atol=1e-3 for sum/long_sum is HONEST, not error-hiding:
- max_rel is tiny everywhere (sum/long_sum <=1e-2 in worst near-zero row case;
  rms_norm ~3e-7). max_abs scales with magnitude (long_sum offset rows: ref~3e5,
  abs_err~0.06-0.25 => rel ~1e-7).
- NON-DEGENERATE check (input +10 offset so row sums are FAR from zero, ref_abs
  1e3-2.6e6): rel error stays ~1e-7 and the TIGHTER tol (rtol=1e-4,atol=1e-5)
  PASSES. So atol=1e-3 is needed only for the near-zero randn row-sums (genuine
  fp32 cancellation), exactly as claimed — it is not masking real divergence.
- The few atol=1e-5 FAILs on standard randn input are tc-vs-Helion BOTH drifting
  from torch.sum equally on cancelling rows (a reduction-ORDER artifact of the
  near-zero reference, present for tc too), not a Helion correctness defect.

## Admit rule applied
long_sum win real beyond noise (3.32x over default; persists at 76-147us bigM);
no active kernel regresses >10% (sum wash; rms_norm worst -6.5% pre-existing);
correctness PASS all; seed used all. => ACCEPT.
