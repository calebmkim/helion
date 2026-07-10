from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def bcast_rowvec_transcend(x, r0, r1, r2, r3):
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn].to(torch.float32)
        a0 = r0[tm][:, None].to(torch.float32)
        a1 = r1[tm][:, None].to(torch.float32)
        a2 = r2[tm][:, None].to(torch.float32)
        a3 = r3[tm][:, None].to(torch.float32)
        v = torch.sin(v * a0) + torch.cos(v * a1) + torch.exp(-(v * a2) * (v * a2)) + torch.tanh(v * a3)
        out[tm, tn] = v.to(x.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    dev = 'cuda'
    dt = torch.bfloat16
    return (
        torch.randn(M, N, device=dev, dtype=dt),
        *[torch.randn(M, device=dev, dtype=dt) for _ in range(4)],
    )
