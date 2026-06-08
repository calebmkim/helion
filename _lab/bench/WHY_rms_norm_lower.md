# Why seeded Helion rms_norm shows G_seed < 1 vs torch.compile

G_seed = tc_default_lat / helion_seeded_lat. rms_norm test-split median ≈ 0.79.
Decomposition (single-process, fresh dynamo, 15-rep do_bench, fp32):

| shape | tc | seed wrapper (TritonBench times this) | seed fwd-only | wrapper OH | G_seed(TB) | G_seed(fwd) | seed_vs_def(fwd) |
|---|---|---|---|---|---|---|---|
| (2048,4096) | 28.5 | 37.4 | 25.9 | +11.5 | 0.76 | 1.10 | 1.05 |
| (8192,1280) | 33.7 | 38.0 | 31.7 | +6.4 | 0.89 | 1.06 | 0.98 |
| (16384,1536) | 72.1 | 73.6 | 73.6 | ~0 | 0.98 | 0.98 | 0.99 |
| (2048,10240) | 61.9 | 62.5 | 62.6 | ~0 | 0.99 | 0.99 | 1.28 |

(µs.)

## Cause: apples-to-oranges baseline, NOT the seed
TritonBench times `rms_norm()` = `RMSNormFunction.apply` (autograd path): it calls
`rms_norm_fwd`, then does `ctx.save_for_backward(x, weight)` + stores `ctx.rms` to set
up the backward pass. torch.compile's reference (`LlamaRMSNorm`) is forward-only with no
autograd graph. That autograd bookkeeping is a small, roughly constant-MAGNITUDE host
cost (~6–12µs).

NOTE (correction): this cost is NOT a clean "fixed per call" overhead. The measured
wrap−fwd deltas are +11.5µs (2048,4096), +6.4µs (8192,1280), ~0µs (16384,1536),
~0µs (2048,10240) — and (2048,4096) vs (2048,10240) share M=2048 yet differ by 11.5µs,
so it does not scale with M and is not literally constant; at large shapes it is within
do_bench noise of zero. The robust statement is a RATIO effect: a small absolute
host/dispatch cost is a large FRACTION of latency at sub-30µs shapes and negligible at
60µs+. (The kernel's second output `inv_rms` is written in BOTH the bare and wrapper
paths, so it cancels in the bare-vs-wrapper delta — it is not part of the wrapper cost.)

At tiny shapes the real GPU reduction is only ~25–30µs, so the autograd cost is a large
fraction → G_seed(TB) ≈ 0.76–0.89. As the shape grows and compute dominates, its share
shrinks → G_seed → ~1.0 (16384,1536 and 2048,10240 both ≈0.98–0.99).

## Decisive cross-check (prior run, SAME test shapes, bare forward kernel)
The prior run's bare-`rms_norm_fwd` measurement on the IDENTICAL test split
(`run2/_lab/logs/run3/task2_replay_bench.json`) gives rms_norm G_seed median **1.019**
(0.982–1.078, n=8) — vs the PR's autograd-path 0.79 on the same shapes. Same shapes,
same seed, same split ⇒ the 0.79 vs ~1.0 gap is ENTIRELY bare-fwd vs autograd-wrapper,
not split-mix, dtype, or tc reference. 7 of 9 kernels match the prior bare-fwd numbers
within noise (cross_entropy 0.825 vs 0.83, kl_div 1.038 vs 1.04, jsd 1.014 vs 1.01,
sum 1.004 vs 1.00); ONLY rms_norm and layer_norm diverge — the two whose TritonBench
hook routes through an autograd `Function`.

It is NOT the seed:
- The bare forward kernel (`G_seed(fwd)`) is 1.06–1.10× FASTER than tc at the small
  shapes — the seed's num_warps=8 helps.
- `seed_vs_default(fwd)` shows the seed beats the unseeded default (1.04–1.28, biggest
  at wide 2048×10240 where it bumps to 16 warps). The seed does its job.

## Why layer_norm looks BETTER (corrected — it is NOT a mirror of rms_norm)
EARLIER CLAIM (wrong): "same wrapper overhead, but tc's layer_norm is just a slower
kernel, so the ratio tips up." A symmetric wrapper overhead can only push G DOWN for
both kernels; it cannot lift one above 1. The real cause is a torch.compile MEASUREMENT
ARTIFACT in TritonBench's layer_norm baseline, proven by decomposition:

Single-process, dynamo-reset, ISOLATED (clean tc baseline):
  rms (2048,4096): fwd=26.0 wrap=40.3 (OH 14.4) tc=28.5 → G_wrap=0.707
  ln  (2048,4096): fwd=26.6 wrap=40.3 (OH 13.7) tc=29.1 → G_wrap=0.722
  ⇒ rms and ln Helion kernels are ~EQUAL (fwd 26 vs 27, wrap 40 vs 40, OH ~14 both),
    AND isolated tc_rms≈tc_ln (28.5 vs 29.1). Clean G is ~0.71 for BOTH — no asymmetry.

TritonBench tc-default latency vs isolated, same shape (2048,4096): TB tc_rms=28.7
(≈isolated 28.5) but TB tc_ln=70.1 (vs my isolated 29.1) → 2.4× higher.

ROOT CAUSE (proven, NOT a session/recompile artifact — that hypothesis was tested and
refuted: TB single-shape tc_ln=62µs ≈ TB multi-shape tc_ln=64µs):
The two TritonBench operators set requires_grad DIFFERENTLY:
  - rms_norm operator: `requires_grad = self.mode in (BWD, FWD_BWD)` → forward-only =
    requires_grad=False → tc_rms ~29µs (no autograd graph).
  - layer_norm operator: `x.requires_grad_()` + weight/bias `requires_grad=True`
    UNCONDITIONALLY → even forward-only builds the autograd graph → tc_ln ~62µs.
Confirmed directly: torch.compile(F.layer_norm) at (2048,4096) is 28.9µs with
requires_grad=False and 61.4µs with requires_grad=True (matches TB's 62µs exactly);
tc_rms similarly 28.4→58.7µs when requires_grad is forced on. So the ~2× is autograd-
graph construction in the forward, deterministic (not noise), and asymmetric between
the two operators by their own input-iter code.

So the "opposite directions" were TWO DIFFERENT things, not one mechanism:
  - rms_norm 0.79 = REAL: symmetric autograd-wrapper overhead vs a clean forward-only
    tc baseline. (Pull G below 1 at small shapes.)
  - layer_norm 1.16 = ARTIFACT: TritonBench's OWN layer_norm tc baseline is ~2× too slow
    in-session, inflating G = tc_ln/helion_ln. Not a real layer_norm advantage.

Both numbers are TritonBench-measurement noise around the truth. The trustworthy
apples-to-apples values are the bare-forward, isolated-tc numbers: rms_norm ≈1.06,
layer_norm ≈1.07 — the two near-identical kernels land in the same place, as they
should. The seed's true effect (seed vs unseeded default) is the same modest ~1.0–1.28
for both.
