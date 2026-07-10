from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def wide_f64_add_relu2(x, y):
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        a = x[tm, tn].to(torch.float32)
        b = y[tm, tn].to(torch.float32)
        s = a + b
        r = torch.where(s > 0, s, torch.zeros_like(s))
        out[tm, tn] = (r * r).to(out.dtype)
    return out


def make_inputs(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float64)
    y = torch.randn(shape, device="cuda", dtype=torch.float64)
    return (x, y)
