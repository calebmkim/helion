# Audit: is the rms_norm/layer_norm measuring mistake present in other kernels?

The bug class: the Helion side and the torch.compile baseline measured under MISMATCHED
autograd conditions (autograd-wrapper on Helion, and/or `requires_grad=True` inputs
making the tc baseline build a grad graph in default FWD mode — TritonBench's default
`Mode.FWD` enables grad; only `fwd_no_grad` wraps in `torch.no_grad()`).

## Two failure-mode dimensions, per kernel
(a) Helion tritonbench hook routes through an autograd `Function` / grad-tracking module.
(b) Operator sets `requires_grad=True` on inputs → tc baseline pays autograd in FWD mode.

| kernel | (a) helion autograd | (b) tc requires_grad | static verdict |
|---|---|---|---|
| rms_norm | YES (RMSNormFunction.apply) | NO (gated on mode) | MISMATCH → G too low |
| layer_norm | YES (LayerNormFunction.apply) | YES (unconditional) | both grad |
| softmax | YES (SoftmaxFunction.apply) | YES (not FWD_NO_GRAD) | both grad |
| sum | no (bare sum_kernel) | NO | clean |
| long_sum | no (bare longsum) | NO | clean |
| welford | no (bare welford) | NO | clean (but TB runs bf16) |
| cross_entropy | no (bare kernel) | YES (unconditional) | MISMATCH → G too high |
| kl_div | nn.Module (grad if inputs rg) | YES | matched-ish |
| jsd | nn.Module (grad if inputs rg) | YES | matched-ish |

## Empirical resolution: TritonBench G vs clean bare-forward G (both sides no-grad)
The GAP is the artifact magnitude. Large gap (>0.1) = a real measuring mismatch.

| kernel | TB G | bare-fwd G | gap | note |
|---|---|---|---|---|
| rms_norm | 0.79 | 1.056 | −0.27 | **ARTIFACT** (autograd wrapper vs grad-free tc) |
| welford | 3.19 | 0.948 | +2.24 | **ARTIFACT** (bf16 in TB vs fp32 + compile-storm baseline) |
| layer_norm | 1.16 | 1.069 | +0.09 | within noise; both-grad, consistent w/ softmax |
| softmax | 1.15 | 1.056 | +0.09 | within noise |
| sum | 1.00 | 1.002 | ~0 | clean |
| long_sum | 1.03 | 1.034 | ~0 | clean |
| cross_entropy | 0.83 | 0.838 | −0.01 | grad-mismatch is IMMATERIAL (large-V kernel dominates) |
| kl_div | 1.04 | 1.039 | ~0 | immaterial |
| jsd | 1.01 | 1.016 | ~0 | immaterial |

## Why the static "mismatches" (cross_entropy, kl_div, jsd) DON'T bite
The autograd overhead is a roughly fixed ~tens-of-µs host cost. It is a large FRACTION of
a ~25µs norm kernel (rms_norm: ~14µs ≈ 50%) but negligible for a ~300µs+ large-vocab loss
kernel. Confirmed directly: tc kl_div at (2048,50257) is 334µs (requires_grad=False) vs
336µs (requires_grad=True) — a <1% effect. So cross_entropy/kl_div/jsd's grad mismatch
exists in principle but does not move G measurably.

## Verdict
The rms_norm/layer_norm measuring mistake is **NOT silently present elsewhere**. Only TWO
kernels diverge materially from the clean apples-to-apples (bare-forward) measurement:
- **rms_norm** (autograd-wrapper overhead vs a grad-free tc baseline) — documented.
- **welford** (bf16 inputs + a different/compile-storming tc baseline) — documented.
The other 7 kernels' TritonBench G matches the clean bare-forward G within noise. The
trustworthy apples-to-apples table (bare-forward, both sides forward-only, no autograd):
rms 1.06, layer_norm 1.07, softmax 1.06, sum 1.00, cross_entropy 0.84, long_sum 1.03,
welford 0.95, kl_div 1.04, jsd 1.02 — geomean ~1.0 vs torch.compile-default.
