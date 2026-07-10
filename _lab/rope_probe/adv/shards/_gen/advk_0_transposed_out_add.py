from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def transposed_out_add(x, y):
    M, N = x.size()
    out = torch.empty(N, M, device=x.device, dtype=x.dtype).t()
    for tm, tn in hl.tile([M, N]):
        out[tm, tn] = (x[tm, tn].to(torch.float32) + y[tm, tn].to(torch.float32)).to(out.dtype)
    return out
def make_inputs(shape):
    M, N = shape
    return (torch.randn(M, N, device='cuda', dtype=torch.bfloat16),
            torch.randn(M, N, device='cuda', dtype=torch.bfloat16))
