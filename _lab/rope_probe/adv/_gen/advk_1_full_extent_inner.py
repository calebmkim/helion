from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def full_extent_inner(x,s):
    out=torch.empty_like(x)
    for tb,tt in hl.tile([x.size(0),x.size(1)]):
        out[tb,tt,:]=(x[tb,tt,:].to(torch.float32)*s[tb,tt,None].to(torch.float32)).to(x.dtype)
    return out
def make_inputs(shape):
    B,T,D=shape
    return (torch.randn(B,T,D,device='cuda',dtype=torch.bfloat16),torch.randn(B,T,device='cuda',dtype=torch.bfloat16))
