from __future__ import annotations
import torch
import helion
import helion.language as hl

@helion.kernel()
def fanout_rope6(q, k, cos, sin):
    batch, q_heads, seq, head_dim = q.size()
    _, k_heads, _, _ = k.size()
    half = head_dim // 2
    o0 = torch.empty_like(q)
    o1 = torch.empty_like(q)
    o2 = torch.empty_like(q)
    o3 = torch.empty_like(k)
    o4 = torch.empty_like(k)
    o5 = torch.empty_like(k)
    for tb, tt in hl.tile([batch, seq]):
        cp = cos[tb, tt, :].to(torch.float32).reshape([tb, tt, 2, half]).permute(0, 1, 3, 2)
        sp = sin[tb, tt, :].to(torch.float32).reshape([tb, tt, 2, half]).permute(0, 1, 3, 2)
        c0, c1 = hl.split(cp)
        s0, s1 = hl.split(sp)
        qp = q[tb, :, tt, :].to(torch.float32).reshape([tb, q_heads, tt, 2, half]).permute(0, 1, 2, 4, 3)
        q0, q1 = hl.split(qp)
        o0[tb, :, tt, :] = hl.join(q0 * c0[:, None, :, :] - q1 * s0[:, None, :, :], q1 * c1[:, None, :, :] + q0 * s1[:, None, :, :]).permute(0, 1, 2, 4, 3).reshape([tb, q_heads, tt, head_dim]).to(o0.dtype)
        o1[tb, :, tt, :] = hl.join(q0 * c1[:, None, :, :], q1 * s1[:, None, :, :]).permute(0, 1, 2, 4, 3).reshape([tb, q_heads, tt, head_dim]).to(o1.dtype)
        o2[tb, :, tt, :] = hl.join(q0 + q1 * c0[:, None, :, :], q1 - q0 * s0[:, None, :, :]).permute(0, 1, 2, 4, 3).reshape([tb, q_heads, tt, head_dim]).to(o2.dtype)
        kp = k[tb, :, tt, :].to(torch.float32).reshape([tb, k_heads, tt, 2, half]).permute(0, 1, 2, 4, 3)
        k0, k1 = hl.split(kp)
        o3[tb, :, tt, :] = hl.join(k0 * c0[:, None, :, :] - k1 * s0[:, None, :, :], k1 * c1[:, None, :, :] + k0 * s1[:, None, :, :]).permute(0, 1, 2, 4, 3).reshape([tb, k_heads, tt, head_dim]).to(o3.dtype)
        o4[tb, :, tt, :] = hl.join(k0 * c1[:, None, :, :], k1 * s1[:, None, :, :]).permute(0, 1, 2, 4, 3).reshape([tb, k_heads, tt, head_dim]).to(o4.dtype)
        o5[tb, :, tt, :] = hl.join(k0 + k1 * c0[:, None, :, :], k1 - k0 * s0[:, None, :, :]).permute(0, 1, 2, 4, 3).reshape([tb, k_heads, tt, head_dim]).to(o5.dtype)
    return o0, o1, o2, o3, o4, o5
def make_inputs(shape):
    batch, heads, seq, head_dim = shape
    q = torch.randn(batch, heads, seq, head_dim, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(batch, heads, seq, head_dim, device='cuda', dtype=torch.bfloat16)
    cos = torch.randn(batch, seq, head_dim, device='cuda', dtype=torch.bfloat16)
    sin = torch.randn(batch, seq, head_dim, device='cuda', dtype=torch.bfloat16)
    return (q, k, cos, sin)
