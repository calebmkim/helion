from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def int8_bitwise_heavy(a, b, c, d):
    out = torch.empty_like(a)
    for tm, tn in hl.tile(a.size()):
        av = a[tm, tn].to(torch.float32)
        bv = b[tm, tn].to(torch.float32)
        cv = c[tm, tn].to(torch.float32)
        dv = d[tm, tn].to(torch.float32)
        t0 = av * bv + cv
        t1 = (av - dv) * (bv + cv)
        t2 = t0 * t0 - t1 * dv
        t3 = (t1 + t2) * (av - cv)
        t4 = t2 * bv - t3 * av + t0 * dv
        r = (t0 + t1 + t2 + t3 + t4) * 0.03125
        out[tm, tn] = r.to(out.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    g = lambda: torch.randint(-8, 8, (M, N), device='cuda', dtype=torch.int8)
    return (g(), g(), g(), g())
