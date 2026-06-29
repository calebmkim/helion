from __future__ import annotations
import torch, helion
import helion.language as hl
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def carried2d_plus_fullslice(x: torch.Tensor, y: torch.Tensor):
    BT, V = x.shape
    bn = hl.register_block_size(V); bm = hl.register_block_size(BT)
    o_loss = torch.zeros([BT], dtype=torch.float32, device=x.device)
    o_max = torch.zeros([BT], dtype=torch.float32, device=x.device)
    for tb in hl.tile(BT, block_size=bm):
        acc = hl.zeros([tb, bn], dtype=torch.float32)
        for tv in hl.tile(V, block_size=bn):
            acc += x[tb, tv].to(torch.float32) * y[tb, tv].to(torch.float32)
        o_loss[tb] = acc.sum(-1)
        o_max[tb] = torch.amax(x[tb, :].to(torch.float32), -1)
    return o_loss, o_max
def make_args(BT=8192, V=4096, dtype=torch.bfloat16, device="cuda"):
    return (torch.randn(BT, V, device=device, dtype=dtype), torch.randn(BT, V, device=device, dtype=dtype))
