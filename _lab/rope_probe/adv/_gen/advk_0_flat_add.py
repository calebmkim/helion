from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def flat_add(x,y):
    out=torch.empty_like(x)
    for tm,tn in hl.tile(x.size()):
        out[tm,tn]=x[tm,tn]+y[tm,tn]
    return out
def make_inputs(shape):
    M,N=shape
    return (torch.randn(M,N,device='cuda',dtype=torch.bfloat16),torch.randn(M,N,device='cuda',dtype=torch.bfloat16))
