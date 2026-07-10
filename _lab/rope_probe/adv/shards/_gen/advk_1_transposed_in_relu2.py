from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def transposed_in_relu2(xT):
    M, N = xT.size()
    out = torch.empty(M, N, device=xT.device, dtype=xT.dtype)
    for tm, tn in hl.tile([M, N]):
        v = xT[tm, tn].to(torch.float32)
        out[tm, tn] = (v * v).to(out.dtype)
    return out
def make_inputs(shape):
    M, N = shape
    base = torch.randn(N, M, device='cuda', dtype=torch.bfloat16)
    return (base.t(),)
