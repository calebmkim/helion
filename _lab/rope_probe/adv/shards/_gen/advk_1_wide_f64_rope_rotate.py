from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def wide_f64_rope_rotate(x, cos, sin):
    batch, heads, seq_len, head_dim = x.size()
    half = head_dim // 2
    out = torch.empty_like(x)
    for tb, tt in hl.tile([batch, seq_len]):
        cos_pair = (
            cos[tb, tt, :]
            .to(torch.float32)
            .reshape([tb, tt, 2, half])
            .permute(0, 1, 3, 2)
        )
        sin_pair = (
            sin[tb, tt, :]
            .to(torch.float32)
            .reshape([tb, tt, 2, half])
            .permute(0, 1, 3, 2)
        )
        cos_first, cos_second = hl.split(cos_pair)
        sin_first, sin_second = hl.split(sin_pair)
        x_pair = (
            x[tb, :, tt, :]
            .to(torch.float32)
            .reshape([tb, heads, tt, 2, half])
            .permute(0, 1, 2, 4, 3)
        )
        x_first, x_second = hl.split(x_pair)
        x_first_out = x_first * cos_first[:, None, :, :] - x_second * sin_first[:, None, :, :]
        x_second_out = x_second * cos_second[:, None, :, :] + x_first * sin_second[:, None, :, :]
        out[tb, :, tt, :] = (
            hl.join(x_first_out, x_second_out)
            .permute(0, 1, 2, 4, 3)
            .reshape([tb, heads, tt, head_dim])
            .to(out.dtype)
        )
    return out


def make_inputs(shape):
    batch, heads, seq_len, head_dim = shape
    x = torch.randn(shape, device="cuda", dtype=torch.float64)
    angles = torch.randn([batch, seq_len, head_dim], device="cuda", dtype=torch.float64)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return (x, cos, sin)
