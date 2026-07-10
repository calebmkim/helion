from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def wide_f64_3d_fma(x, y, z):
    out = torch.empty_like(x)
    for ti, tj, tk in hl.tile(x.size()):
        a = x[ti, tj, tk].to(torch.float32)
        b = y[ti, tj, tk].to(torch.float32)
        c = z[ti, tj, tk].to(torch.float32)
        out[ti, tj, tk] = (a * b + c).to(out.dtype)
    return out


def make_inputs(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.float64)
    y = torch.randn(shape, device="cuda", dtype=torch.float64)
    z = torch.randn(shape, device="cuda", dtype=torch.float64)
    return (x, y, z)
