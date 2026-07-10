from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def bool_where_fp32(m, a, b):
    out = torch.empty_like(a)
    for tm, tn in hl.tile(a.size()):
        mv = m[tm, tn]
        av = a[tm, tn].to(torch.float32)
        bv = b[tm, tn].to(torch.float32)
        r = torch.where(mv, av * 2.0 + 1.0, bv * 3.0 - 1.0)
        out[tm, tn] = r.to(out.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    return (
        torch.randint(0, 2, (M, N), device='cuda', dtype=torch.bool),
        torch.randint(-8, 8, (M, N), device='cuda', dtype=torch.int8),
        torch.randint(-8, 8, (M, N), device='cuda', dtype=torch.int8),
    )
