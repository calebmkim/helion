from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def fullgrid_plus_carried2d(x: torch.Tensor, z: torch.Tensor):
    T, G, GS = x.shape
    _, R = z.shape
    hl.specialize(GS)
    hl.specialize(G)
    bn = hl.register_block_size(R)
    o_grid = torch.empty([T, G], dtype=torch.float32, device=x.device)
    o_user = torch.zeros([T], dtype=torch.float32, device=x.device)
    for tile_t, tile_g, tile_c in hl.tile([T, G, GS], block_size=[1, None, GS]):
        o_grid[tile_t, tile_g] = torch.amax(x[tile_t, tile_g, tile_c].to(torch.float32), -1)
        acc = hl.zeros([tile_t, bn], dtype=torch.float32)
        for tile_r in hl.tile(R, block_size=bn):
            zr = z[tile_t, tile_r].to(torch.float32)
            acc += zr * zr
        o_user[tile_t] = acc.sum(-1)
    return o_grid, o_user


def make_args(T=8192, G=32, GS=128, R=8192, dtype=torch.bfloat16, device="cuda"):
    return (torch.randn(T, G, GS, device=device, dtype=dtype),
            torch.randn(T, R, device=device, dtype=dtype))
