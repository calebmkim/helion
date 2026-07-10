from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def horner_poly24(x):
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn].to(torch.float32)
        p2 = v * v
        p3 = p2 * v
        p4 = p3 * v
        p5 = p4 * v
        p6 = p5 * v
        p7 = p6 * v
        p8 = p7 * v
        p9 = p8 * v
        p10 = p9 * v
        p11 = p10 * v
        p12 = p11 * v
        p13 = p12 * v
        p14 = p13 * v
        p15 = p14 * v
        p16 = p15 * v
        p17 = p16 * v
        p18 = p17 * v
        p19 = p18 * v
        p20 = p19 * v
        p21 = p20 * v
        p22 = p21 * v
        p23 = p22 * v
        p24 = p23 * v
        acc = (
            0.101 * v + 0.203 * p2 + 0.307 * p3 + 0.409 * p4
            + 0.511 * p5 + 0.613 * p6 + 0.717 * p7 + 0.819 * p8
            + 0.921 * p9 + 1.023 * p10 + 1.127 * p11 + 1.229 * p12
            + 1.331 * p13 + 1.433 * p14 + 1.537 * p15 + 1.639 * p16
            + 1.741 * p17 + 1.843 * p18 + 1.947 * p19 + 2.049 * p20
            + 2.151 * p21 + 2.253 * p22 + 2.357 * p23 + 2.459 * p24
        )
        out[tm, tn] = (0.001 * acc).to(x.dtype)
    return out

def make_inputs(shape):
    M, N = shape
    return (torch.randn(M, N, device='cuda', dtype=torch.bfloat16) * 0.25,)
