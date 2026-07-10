from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def bcast_dual_axis_grid(x, cs0, cs1, cs2, rs0, rs1, rs2):
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn].to(torch.float32)
        c0 = cs0[tn][None, :].to(torch.float32)
        c1 = cs1[tn][None, :].to(torch.float32)
        c2 = cs2[tn][None, :].to(torch.float32)
        r0 = rs0[tm][:, None].to(torch.float32)
        r1 = rs1[tm][:, None].to(torch.float32)
        r2 = rs2[tm][:, None].to(torch.float32)
        v = (v * c0 + r0) * c1 + r1
        v = v * c2 + r2
        out[tm, tn] = v.to(x.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    dev = 'cuda'
    dt = torch.bfloat16
    return (
        torch.randn(M, N, device=dev, dtype=dt),
        *[torch.randn(N, device=dev, dtype=dt) for _ in range(3)],
        *[torch.randn(M, device=dev, dtype=dt) for _ in range(3)],
    )
