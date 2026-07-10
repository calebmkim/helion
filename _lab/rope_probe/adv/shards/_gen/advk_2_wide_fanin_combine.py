from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def wide_fanin_combine(a, b, c, d):
    out = torch.empty_like(a)
    for tm, tn in hl.tile(a.size()):
        av = a[tm, tn].to(torch.float32)
        bv = b[tm, tn].to(torch.float32)
        cv = c[tm, tn].to(torch.float32)
        dv = d[tm, tn].to(torch.float32)
        a1 = torch.sin(av * 1.1); a2 = torch.tanh(a1 * 1.2 + av); a3 = torch.exp(a2 * 0.13); a4 = torch.log(torch.abs(a3) + 1.0)
        b1 = torch.cos(bv * 1.4); b2 = torch.sigmoid(b1 * 1.5 + bv); b3 = torch.rsqrt(b2 * b2 + 1.6); b4 = torch.sin(b3 * 1.7)
        c1 = torch.tanh(cv * 1.8); c2 = torch.exp(c1 * 0.19 + cv * 0.1); c3 = torch.log(torch.abs(c2) + 1.0); c4 = torch.cos(c3 * 2.1)
        d1 = torch.sigmoid(dv * 2.2); d2 = torch.sin(d1 * 2.3 + dv); d3 = torch.tanh(d2 * 2.4); d4 = torch.rsqrt(d3 * d3 + 2.5)
        acc = (
            a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4
            + c1 + c2 + c3 + c4 + d1 + d2 + d3 + d4
            + a4 * b4 + c4 * d4 + a2 * c2 + b2 * d2
        )
        out[tm, tn] = (0.05 * acc).to(a.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    mk = lambda: torch.randn(M, N, device='cuda', dtype=torch.bfloat16) * 0.5
    return (mk(), mk(), mk(), mk())
