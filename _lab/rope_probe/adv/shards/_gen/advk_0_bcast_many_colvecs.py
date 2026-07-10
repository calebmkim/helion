from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def bcast_many_colvecs(x, a, b, c, d, e, f, g, h):
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        xv = x[tm, tn].to(torch.float32)
        acc = (
            xv * a[tn][None, :].to(torch.float32)
            + b[tn][None, :].to(torch.float32)
            + xv * c[tn][None, :].to(torch.float32)
            + d[tn][None, :].to(torch.float32)
            + xv * e[tn][None, :].to(torch.float32)
            + f[tn][None, :].to(torch.float32)
            + xv * g[tn][None, :].to(torch.float32)
            + h[tn][None, :].to(torch.float32)
        )
        out[tm, tn] = acc.to(x.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    dev = 'cuda'
    dt = torch.bfloat16
    return (
        torch.randn(M, N, device=dev, dtype=dt),
        *[torch.randn(N, device=dev, dtype=dt) for _ in range(8)],
    )
