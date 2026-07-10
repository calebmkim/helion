from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def regcap_flip_rotate(x, cos, sin):
    batch, heads, seq, hd = x.size()
    half = hd // 2
    out = torch.empty_like(x)
    for tb, tt in hl.tile([batch, seq]):
        cos_pair = cos[tb, tt, :].to(torch.float32).reshape([tb, tt, 2, half]).permute(0, 1, 3, 2)
        sin_pair = sin[tb, tt, :].to(torch.float32).reshape([tb, tt, 2, half]).permute(0, 1, 3, 2)
        cos_a, cos_b = hl.split(cos_pair)
        sin_a, sin_b = hl.split(sin_pair)
        xp = x[tb, :, tt, :].to(torch.float32).reshape([tb, heads, tt, 2, half]).permute(0, 1, 2, 4, 3)
        x_a, x_b = hl.split(xp)
        y_a = x_a * cos_a[:, None, :, :] - x_b * sin_a[:, None, :, :]
        y_b = x_b * cos_b[:, None, :, :] + x_a * sin_b[:, None, :, :]
        out[tb, :, tt, :] = hl.join(y_a, y_b).permute(0, 1, 2, 4, 3).reshape([tb, heads, tt, hd]).to(out.dtype)
    return out
def make_inputs(shape):
    batch, heads, seq, hd = shape
    x = torch.randn(batch, heads, seq, hd, device='cuda', dtype=torch.bfloat16)
    ang = torch.randn(batch, seq, hd, device='cuda', dtype=torch.bfloat16)
    return (x, torch.cos(ang), torch.sin(ang))
