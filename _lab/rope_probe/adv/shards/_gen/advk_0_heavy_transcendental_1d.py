from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def heavy_transcendental_1d(x):
    out = torch.empty_like(x)
    for t in hl.tile(x.size(0)):
        v = x[t].to(torch.float32)
        v = torch.sin(v) + torch.cos(v * 1.3)
        v = torch.tanh(v) - torch.tanh(v * 0.7)
        v = torch.exp(-v * v)
        v = torch.log(v + 2.0) / (torch.exp(v) + 1.0)
        v = torch.erf(v) + torch.sin(v * 2.1)
        v = torch.tanh(v) * torch.cos(v)
        v = torch.exp(v) / (torch.exp(-v) + 3.0)
        v = torch.log1p(torch.abs(v)) + torch.sin(v)
        v = v / (torch.cos(v) * torch.cos(v) + 1.5)
        v = torch.tanh(torch.exp(-torch.abs(v)))
        v = torch.erf(v * 0.9) - torch.sin(v)
        v = torch.exp(v) / (torch.log(torch.abs(v) + 2.0) + 1.0)
        out[t] = v.to(out.dtype)
    return out
def make_inputs(shape):
    (N,) = shape
    return (torch.randn(N, device='cuda', dtype=torch.bfloat16),)
