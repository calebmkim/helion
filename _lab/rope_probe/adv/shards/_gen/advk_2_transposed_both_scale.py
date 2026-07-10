from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def transposed_both_scale(xT, s):
    M, N = xT.size()
    out = torch.empty(N, M, device=xT.device, dtype=xT.dtype).t()
    for tm, tn in hl.tile([M, N]):
        out[tm, tn] = xT[tm, tn] * s
    return out
def make_inputs(shape):
    M, N = shape
    base = torch.randn(N, M, device='cuda', dtype=torch.float32)
    return (base.t(), 1.5)
