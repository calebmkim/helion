from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def int8_polyeval(a, b, c):
    out = torch.empty_like(a)
    for tm, tn in hl.tile(a.size()):
        x = a[tm, tn].to(torch.float32)
        y = b[tm, tn].to(torch.float32)
        z = c[tm, tn].to(torch.float32)
        p0 = x * x * x + y * y - z
        p1 = (x + y) * (y + z) * (z + x)
        p2 = x * y * z + p0 - p1
        p3 = (p0 * p1 - p2) * (x - y + z)
        p4 = p3 * p2 + p1 * p0 - x * y
        p5 = (p4 - p3 + p2 - p1 + p0) * (x + 1.0)
        p6 = p5 * p4 - p3 * p2 + p1 * p0
        r = (p0 + p1 + p2 + p3 + p4 + p5 + p6) * 0.015625
        out[tm, tn] = r.to(out.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    g = lambda: torch.randint(-4, 4, (M, N), device='cuda', dtype=torch.int8)
    return (g(), g(), g())
