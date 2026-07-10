from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def fanin32_2d_silu(
    x0: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    x3: torch.Tensor,
    x4: torch.Tensor,
    x5: torch.Tensor,
    x6: torch.Tensor,
    x7: torch.Tensor,
    x8: torch.Tensor,
    x9: torch.Tensor,
    x10: torch.Tensor,
    x11: torch.Tensor,
    x12: torch.Tensor,
    x13: torch.Tensor,
    x14: torch.Tensor,
    x15: torch.Tensor,
    x16: torch.Tensor,
    x17: torch.Tensor,
    x18: torch.Tensor,
    x19: torch.Tensor,
    x20: torch.Tensor,
    x21: torch.Tensor,
    x22: torch.Tensor,
    x23: torch.Tensor,
    x24: torch.Tensor,
    x25: torch.Tensor,
    x26: torch.Tensor,
    x27: torch.Tensor,
    x28: torch.Tensor,
    x29: torch.Tensor,
    x30: torch.Tensor,
    x31: torch.Tensor,
) -> torch.Tensor:
    out = torch.empty_like(x0)
    for tm, tn in hl.tile(x0.size()):
        s = (x0[tm, tn].to(torch.float32) * 0.5000 + x1[tm, tn].to(torch.float32) * 0.5100 + x2[tm, tn].to(torch.float32) * 0.5200 + x3[tm, tn].to(torch.float32) * 0.5300 + x4[tm, tn].to(torch.float32) * 0.5400 + x5[tm, tn].to(torch.float32) * 0.5500 + x6[tm, tn].to(torch.float32) * 0.5600 + x7[tm, tn].to(torch.float32) * 0.5700 + x8[tm, tn].to(torch.float32) * 0.5800 + x9[tm, tn].to(torch.float32) * 0.5900 + x10[tm, tn].to(torch.float32) * 0.6000 + x11[tm, tn].to(torch.float32) * 0.6100 + x12[tm, tn].to(torch.float32) * 0.6200 + x13[tm, tn].to(torch.float32) * 0.6300 + x14[tm, tn].to(torch.float32) * 0.6400 + x15[tm, tn].to(torch.float32) * 0.6500 + x16[tm, tn].to(torch.float32) * 0.6600 + x17[tm, tn].to(torch.float32) * 0.6700 + x18[tm, tn].to(torch.float32) * 0.6800 + x19[tm, tn].to(torch.float32) * 0.6900 + x20[tm, tn].to(torch.float32) * 0.7000 + x21[tm, tn].to(torch.float32) * 0.7100 + x22[tm, tn].to(torch.float32) * 0.7200 + x23[tm, tn].to(torch.float32) * 0.7300 + x24[tm, tn].to(torch.float32) * 0.7400 + x25[tm, tn].to(torch.float32) * 0.7500 + x26[tm, tn].to(torch.float32) * 0.7600 + x27[tm, tn].to(torch.float32) * 0.7700 + x28[tm, tn].to(torch.float32) * 0.7800 + x29[tm, tn].to(torch.float32) * 0.7900 + x30[tm, tn].to(torch.float32) * 0.8000 + x31[tm, tn].to(torch.float32) * 0.8100) * 0.031250
        acc = s * torch.sigmoid(s)
        out[tm, tn] = acc.to(out.dtype)
    return out


def make_inputs(shape):
    return tuple(torch.randn(shape, device='cuda', dtype=torch.float32) for _ in range(32))