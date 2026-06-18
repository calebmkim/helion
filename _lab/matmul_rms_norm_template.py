"""TEMPLATE kernel for the matmul + reduction-epilogue task: matmul + RMSNorm.

This is the ONE kernel handed to you verified-working (measured on H100, bf16, accuracy True,
~1.4-2.6x over Inductor at skinny M>>K,N small-N). Pattern-match it to author the rest of the
corpus (layernorm, softmax, l2_normalize, sum, logsumexp, argmax): the matmul skeleton is
SHARED; only the EPILOGUE block (the over-N reduction) changes.

The fused structure the heuristic targets:
  - one grid loop over M (rows); N is hl.specialize'd (compile-time constant) and full-width,
    so the matmul accumulator [tile_m, n] is register-resident;
  - an inner K-loop matmul (addmm) into that accumulator;
  - a reduction over the N (output) axis on the register-resident accumulator (the EPILOGUE),
    then the write-back.
This is why no ReductionFact is registered today (the reduction rides the matmul accumulator,
not an HBM row) and why the win regime is small-N (the [tile_m, n] fp32 acc + [tile_k, n]
operand both scale with N -> SMEM-bound).

Real-world instantiation: QK-Norm = this with N=head_dim (64-128) on the per-head
[B*S*H, head_dim] view; MoE-router softmax = this with N=n_experts; an embedding head's
L2-normalize = this with N=embedding_dim.

NOTE: author every OTHER corpus member yourself, each from its own named torch/library
reference (the Gate-E circularity firewall) -- do not invent structure.
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


@helion.kernel(static_shapes=True)  # static_shapes=True gives a perf boost for matmuls
def matmul_rms_norm(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """out = rms_norm(x @ y, weight), reducing over the N (output) axis.

    x: [M, K], y: [K, N], weight: [N] -> out: [M, N].
    """
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))  # N full-width, never tiled -> accumulator is [tile_m, n]
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        # ---- EPILOGUE: reduction over N on the register-resident accumulator ----
        # Swap THIS block to author other members; the matmul skeleton above is shared.
        eps = 1e-6
        ms = (acc * acc).sum(dim=-1, keepdim=True) / n  # the over-N reduction
        normed = acc * torch.rsqrt(ms + eps)
        out[tile_m, :] = normed * weight[:].to(torch.float32)
        # -----------------------------------------------------------------------
    return out


def matmul_rms_norm_ref(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Pure-torch reference (the accuracy oracle). bf16 in -> compute in fp32 -> cast back."""
    mm = torch.matmul(x, y).to(torch.float32)
    ms = mm.pow(2).mean(dim=-1, keepdim=True)
    normed = mm * torch.rsqrt(ms + 1e-6)
    return (normed * weight.to(torch.float32)).to(
        torch.promote_types(x.dtype, y.dtype)
    )
