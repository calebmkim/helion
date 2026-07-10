from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def skinny_m_wide_n_chain(x):
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn].to(torch.float32)
        v = torch.sin(v * 1.3) + torch.cos(v)
        v = torch.exp(-torch.abs(v) * 0.35) + torch.tanh(v * 0.8)
        v = torch.sin(v) + torch.cos(v * 1.15)
        v = torch.tanh(v * 1.1) + torch.exp(-v * v * 0.2)
        v = torch.log(1.0 + torch.abs(v)) * torch.sin(v)
        v = torch.sqrt(torch.abs(v) + 1e-3) + torch.cos(v * 0.9)
        out[tm, tn] = v.to(x.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    return (torch.randn(M, N, device='cuda', dtype=torch.bfloat16),)
