from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def iterated_map_fp32(x):
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn].to(torch.float32)
        r = v * 0.5 + 0.5
        for _ in range(6):
            r = 0.5 * (r + (torch.abs(v) + 1e-3) / (torch.abs(r) + 1e-3))
        w = torch.exp(-torch.abs(v)) * torch.sin(v) + torch.cos(v * 1.3)
        w = torch.tanh(w) - torch.erf(w * 0.8)
        w = torch.log(torch.abs(w) + 2.0) / (torch.exp(w) + 1.5)
        w = torch.pow(torch.abs(w) + 1e-3, 1.5) - torch.sin(w * 2.0)
        w = torch.exp(w) / (torch.exp(-w) + 3.0)
        w = torch.tanh(torch.pow(torch.abs(w) + 1e-3, 0.75))
        v = r * torch.cos(w) + w * torch.sin(r)
        v = torch.erf(v) + torch.log1p(torch.abs(v)) - torch.tanh(v)
        out[tm, tn] = v.to(out.dtype)
    return out
def make_inputs(shape):
    M, N = shape
    return (torch.randn(M, N, device='cuda', dtype=torch.float32),)
