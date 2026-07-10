from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def fanout_partial_moderate(x):
    rows, chan = x.size()
    half = chan // 2
    o0 = torch.empty_like(x)
    o1 = torch.empty_like(x)
    o2 = torch.empty_like(x)
    o3 = torch.empty_like(x)
    o4 = torch.empty_like(x)
    o5 = torch.empty_like(x)
    for tr in hl.tile(rows):
        v = x[tr, :].to(torch.float32).reshape([tr, 2, half]).permute(0, 2, 1)
        a, b = hl.split(v)
        o0[tr, :] = hl.join(a + b, a - b).permute(0, 2, 1).reshape([tr, chan]).to(o0.dtype)
        o1[tr, :] = hl.join(a * b, a * a).permute(0, 2, 1).reshape([tr, chan]).to(o1.dtype)
        o2[tr, :] = hl.join(torch.tanh(a), torch.tanh(b)).permute(0, 2, 1).reshape([tr, chan]).to(o2.dtype)
        o3[tr, :] = hl.join(a * torch.sigmoid(b), b * torch.sigmoid(a)).permute(0, 2, 1).reshape([tr, chan]).to(o3.dtype)
        o4[tr, :] = hl.join(torch.exp(a) - a, torch.exp(b) - b).permute(0, 2, 1).reshape([tr, chan]).to(o4.dtype)
        o5[tr, :] = hl.join(a - b, b - a).permute(0, 2, 1).reshape([tr, chan]).to(o5.dtype)
    return o0, o1, o2, o3, o4, o5
def make_inputs(shape):
    rows, chan = shape
    x = torch.randn(rows, chan, device='cuda', dtype=torch.bfloat16)
    return (x,)
