from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def chained_activation_2d(x, y):
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        a = x[tm, tn].to(torch.float32)
        b = y[tm, tn].to(torch.float32)
        v = torch.sin(a) * torch.cos(b) + torch.tanh(a - b)
        v = torch.exp(-v * v) + torch.erf(v * 0.5)
        v = torch.log(torch.abs(v) + 1.5) / (torch.exp(v) + 2.0)
        v = torch.tanh(v) - torch.sin(v * 1.7) * torch.cos(v * 0.9)
        v = torch.exp(v) / (torch.exp(-v) + 4.0)
        v = torch.erf(v) + torch.log1p(torch.abs(v))
        v = v / (torch.cos(v) * torch.cos(v) + 1.25)
        v = torch.tanh(torch.exp(-torch.abs(v))) + torch.sin(v)
        v = torch.exp(-torch.abs(v)) * torch.erf(v * 1.1)
        v = torch.log(torch.abs(v) + 2.0) - torch.tanh(v)
        out[tm, tn] = v.to(out.dtype)
    return out
def make_inputs(shape):
    M, N = shape
    return (torch.randn(M, N, device='cuda', dtype=torch.bfloat16),
            torch.randn(M, N, device='cuda', dtype=torch.bfloat16))
