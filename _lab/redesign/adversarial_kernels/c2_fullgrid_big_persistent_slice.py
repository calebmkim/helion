from __future__ import annotations
import torch, helion
import helion.language as hl
@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def c2_fullgrid_plus_bigslice(x: torch.Tensor, w: torch.Tensor):
    T, G, GS = x.shape
    _, N = w.shape
    hl.specialize(GS); hl.specialize(G)
    o1 = torch.empty([T, G], dtype=torch.float32, device=x.device)
    o2 = torch.empty([T, N], dtype=torch.float32, device=x.device)
    for tile_t, tile_g, tile_c in hl.tile([T, G, GS], block_size=[1, None, GS]):
        o1[tile_t, tile_g] = torch.amax(x[tile_t, tile_g, tile_c].to(torch.float32), -1)
        row = w[tile_t, :].to(torch.float32)
        s = row.sum(-1)
        o2[tile_t, :] = (row / (s[:, None] + 1.0)).to(torch.float32)
    return o1, o2
def make_args(T=8192, G=256, GS=2048, N=61440, dtype=torch.bfloat16, device="cuda"):
    return (torch.randn(T, G, GS, device=device, dtype=dtype),
            torch.randn(T, N, device=device, dtype=dtype))
