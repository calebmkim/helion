from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def wide_row_moderate_chain(x):
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn].to(torch.float32)
        v = torch.sin(v) + torch.cos(v * 1.1)
        v = torch.tanh(v * 0.9) + torch.exp(-torch.abs(v) * 0.3)
        out[tm, tn] = v.to(x.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    return (torch.randn(M, N, device='cuda', dtype=torch.bfloat16),)
