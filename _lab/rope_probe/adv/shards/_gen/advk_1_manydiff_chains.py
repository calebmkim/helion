from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def manydiff_chains(x):
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn].to(torch.float32)
        c1 = torch.sin(v * 1.1)
        c2 = torch.cos(v * 1.2)
        c3 = torch.tanh(v * 1.3)
        c4 = torch.sigmoid(v * 1.4)
        c5 = torch.exp(v * 0.15)
        c6 = torch.log(torch.abs(v) * 1.6 + 1.0)
        c7 = torch.rsqrt(v * v + 1.7)
        c8 = torch.sin(v * 1.8 + c1)
        c9 = torch.cos(v * 1.9 + c2)
        c10 = torch.tanh(v * 2.0 + c3)
        c11 = torch.sigmoid(v * 2.1 + c4)
        c12 = torch.exp(v * 0.22 + c5 * 0.1)
        c13 = torch.log(torch.abs(v * 2.3 + c6) + 1.0)
        c14 = torch.rsqrt(c7 * c7 + 2.4)
        c15 = torch.sin(c8 * 2.5 + c9)
        c16 = torch.cos(c10 * 2.6 + c11)
        c17 = torch.tanh(c12 * 0.27 + c13)
        c18 = torch.sigmoid(c14 * 2.8 + c15)
        c19 = torch.exp(c16 * 0.29 + c17 * 0.1)
        c20 = torch.log(torch.abs(c18 * 3.0 + c19) + 1.0)
        acc = (
            c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + c9 + c10
            + c11 + c12 + c13 + c14 + c15 + c16 + c17 + c18 + c19 + c20
        )
        out[tm, tn] = (0.05 * acc).to(x.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    return (torch.randn(M, N, device='cuda', dtype=torch.bfloat16) * 0.5,)
