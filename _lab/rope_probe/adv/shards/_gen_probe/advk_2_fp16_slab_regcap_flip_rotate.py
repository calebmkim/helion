from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def fp16_slab_regcap_flip_rotate(x, scale):
    batch, seq, d = x.size()
    half = d // 2
    out = torch.empty_like(x)
    for tb, tt in hl.tile([batch, seq]):
        sc = scale[tb, tt, :].to(torch.float32).reshape([tb, tt, 2, half]).permute(0, 1, 3, 2)
        sc_a, sc_b = hl.split(sc)
        xp = x[tb, tt, :].to(torch.float32).reshape([tb, tt, 2, half]).permute(0, 1, 3, 2)
        x_a, x_b = hl.split(xp)
        y_a = x_a * sc_a - x_b * sc_b
        y_b = x_b * sc_a + x_a * sc_b
        out[tb, tt, :] = hl.join(y_a, y_b).permute(0, 1, 3, 2).reshape([tb, tt, d]).to(out.dtype)
    return out
def make_inputs(shape):
    batch, seq, d = shape
    x = torch.randn(batch, seq, d, device='cuda', dtype=torch.float16)
    scale = torch.randn(batch, seq, d, device='cuda', dtype=torch.float16)
    return (x, scale)
