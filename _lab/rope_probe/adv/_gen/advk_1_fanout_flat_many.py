from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def fanout_flat_many(x, y):
    o0 = torch.empty_like(x)
    o1 = torch.empty_like(x)
    o2 = torch.empty_like(x)
    o3 = torch.empty_like(x)
    o4 = torch.empty_like(x)
    o5 = torch.empty_like(x)
    o6 = torch.empty_like(x)
    o7 = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        a = x[tm, tn].to(torch.float32)
        b = y[tm, tn].to(torch.float32)
        o0[tm, tn] = (a + b).to(o0.dtype)
        o1[tm, tn] = (a - b).to(o1.dtype)
        o2[tm, tn] = (a * b).to(o2.dtype)
        o3[tm, tn] = (a * a - b * b).to(o3.dtype)
        o4[tm, tn] = torch.tanh(a).to(o4.dtype)
        o5[tm, tn] = torch.tanh(b).to(o5.dtype)
        o6[tm, tn] = (a * torch.sigmoid(b)).to(o6.dtype)
        o7[tm, tn] = (b * torch.sigmoid(a)).to(o7.dtype)
    return o0, o1, o2, o3, o4, o5, o6, o7
def make_inputs(shape):
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    y = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    return (x, y)
