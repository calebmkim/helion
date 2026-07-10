"""
WS2 m-reduction curriculum candidates — DRAFT kernels for the recognizer-broadening effort.

These exercise "M-reduction" = a reduction over the GRID axis (the per-CTA tile-accumulation
half), authored grad_w-style (grid over M, per-CTA partial buffer, separate finalize) so they
stay clear of the split-K *combine* track. All are backward kernels.

Structural classes (relative to the rms/ln baseline = "class C: feature reduction over the SAME
axis as the param accumulator"):
  A. pure collapse        — param-grad accumulates over M, NO feature reduction   -> bias_grad, dyt
  B. decoupled axis       — feature reduction exists, over a DIFFERENT axis than param accumulator
                            -> group_norm, instance_norm

Purpose: a separate agent will broaden build_m_reduction_facts to fire on these. This file only
provides correct, compilable drafts + a correctness gate. Run with:
  PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES=2 HELION_AUTOTUNE_EFFORT=none python this_file.py
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# ---------------------------------------------------------------------------
# Class A.1 — bias gradient: grad_bias[n] = sum_M grad_out[m, n]  (pure collapse, no feature reduce)
# ---------------------------------------------------------------------------
@helion.kernel
def bias_grad_bwd(grad_out: torch.Tensor) -> torch.Tensor:
    m, n = grad_out.size()
    n = hl.specialize(n)
    m_block = hl.register_block_size(m)
    num_blocks = (m + m_block - 1) // m_block
    gb_blocks = grad_out.new_empty([num_blocks, n], dtype=torch.float32)
    for mb_cta in hl.tile(m, block_size=m_block):
        gb = grad_out.new_zeros(n, dtype=torch.float32)
        for mb in hl.tile(mb_cta.begin, mb_cta.end):
            gb += torch.sum(grad_out[mb, :].to(torch.float32), dim=0)
        gb_blocks[mb_cta.id, :] = gb
    return gb_blocks.sum(0).to(grad_out.dtype)


def bias_grad_ref(grad_out: torch.Tensor) -> torch.Tensor:
    return grad_out.to(torch.float32).sum(0).to(grad_out.dtype)


# ---------------------------------------------------------------------------
# Class A.2 — DyT (Dynamic Tanh) backward: y = weight * tanh(alpha * x) + bias
#   grad_weight[n] = sum_M(grad_out * tanh(alpha*x))   (pure collapse, no feature reduce)
#   grad_bias[n]   = sum_M(grad_out)
#   grad_x[m,n]    = grad_out * weight * alpha * (1 - tanh^2)   (elementwise, no reduction)
# ---------------------------------------------------------------------------
@helion.kernel
def dyt_bwd(
    grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.size()
    n = hl.specialize(n)
    m_block = hl.register_block_size(m)
    grad_x = torch.empty_like(x)
    num_blocks = (m + m_block - 1) // m_block
    gw_blocks = x.new_empty([num_blocks, n], dtype=torch.float32)
    gb_blocks = x.new_empty([num_blocks, n], dtype=torch.float32)
    for mb_cta in hl.tile(m, block_size=m_block):
        gw = weight.new_zeros(n, dtype=torch.float32)
        gb = weight.new_zeros(n, dtype=torch.float32)
        w = weight[None, :].to(torch.float32)
        for mb in hl.tile(mb_cta.begin, mb_cta.end):
            x_mb = x[mb, :].to(torch.float32)
            dy = grad_out[mb, :].to(torch.float32)
            t = torch.tanh(alpha * x_mb)
            gw += torch.sum(dy * t, dim=0)
            gb += torch.sum(dy, dim=0)
            dx = dy * w * alpha * (1.0 - t * t)
            grad_x[mb, :] = dx.to(x.dtype)
        gw_blocks[mb_cta.id, :] = gw
        gb_blocks[mb_cta.id, :] = gb
    return (
        grad_x,
        gw_blocks.sum(0).to(weight.dtype),
        gb_blocks.sum(0).to(weight.dtype),
    )


def dyt_ref(
    grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, alpha: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xf = x.detach().to(torch.float32).requires_grad_(True)
    wf = weight.detach().to(torch.float32).requires_grad_(True)
    bf = bias.detach().to(torch.float32).requires_grad_(True)
    y = wf[None, :] * torch.tanh(alpha * xf) + bf[None, :]
    y.backward(grad_out.to(torch.float32))
    return xf.grad, wf.grad, bf.grad


# ---------------------------------------------------------------------------
# Class B.1 — group_norm backward (CANONICAL 3D layout [N, C, S] with spatial).
#   Normalize per (n, group) over (Cg channels x spatial S); weight/bias per channel [C].
#   grad_weight[c] = sum over (N, S) of (x_hat * grad_out)  -> accumulator [C], reduces over N (grid) + S
#   grad_x[n,c,s]  = group-norm bwd, feature-reduce over (Cg, S) per (n, group)
#   The resident [inner_N, C, S] fp32 tile is the spill the m-reduction byte-cap targets.
# ---------------------------------------------------------------------------
@helion.kernel
def group_norm_bwd(
    grad_out: torch.Tensor,  # [N, C, S]
    x: torch.Tensor,  # [N, C, S]
    mean: torch.Tensor,  # [N, G]
    rstd: torch.Tensor,  # [N, G]
    weight: torch.Tensor,  # [C]
    num_groups: hl.constexpr,  # type: ignore[valid-type]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_size, c_size, s = x.size()
    c_size = hl.specialize(c_size)
    s = hl.specialize(s)
    g = hl.specialize(num_groups)
    cg = hl.specialize(c_size // num_groups)
    cnt = cg * s  # group element count (Cg x S)
    m_block = hl.register_block_size(n_size)
    grad_x = torch.empty_like(x)
    num_blocks = (n_size + m_block - 1) // m_block
    gw_blocks = x.new_empty([num_blocks, c_size], dtype=torch.float32)
    gb_blocks = x.new_empty([num_blocks, c_size], dtype=torch.float32)
    for n_cta in hl.tile(n_size, block_size=m_block):
        gw = weight.new_zeros(c_size, dtype=torch.float32)
        gb = weight.new_zeros(c_size, dtype=torch.float32)
        wg = weight[:].to(torch.float32).reshape(g, cg)  # [G, Cg]
        for nn in hl.tile(n_cta.begin, n_cta.end):
            tn = nn.block_size
            x_n = x[nn, :, :].to(torch.float32).reshape(tn, g, cg, s)  # [tn, G, Cg, S]
            dy_n = grad_out[nn, :, :].to(torch.float32).reshape(tn, g, cg, s)
            mean_n = mean[nn, :].to(torch.float32)  # [tn, G]
            rstd_n = rstd[nn, :].to(torch.float32)  # [tn, G]
            x_hat = (x_n - mean_n[:, :, None, None]) * rstd_n[:, :, None, None]
            # grad_weight/grad_bias: reduce over N (dim 0) and S (dim 3) -> [G, Cg] -> [C]
            gw += torch.sum(torch.sum(dy_n * x_hat, dim=3), dim=0).reshape(c_size)
            gb += torch.sum(torch.sum(dy_n, dim=3), dim=0).reshape(c_size)
            wdy = dy_n * wg[None, :, :, None]  # [tn, G, Cg, S]
            # grad_x stats per (n, group): reduce over (Cg, S) = dims (2, 3)
            c1 = torch.sum(torch.sum(x_hat * wdy, dim=3), dim=2) / cnt  # [tn, G]
            c2 = torch.sum(torch.sum(wdy, dim=3), dim=2) / cnt  # [tn, G]
            dx = (
                wdy - (x_hat * c1[:, :, None, None] + c2[:, :, None, None])
            ) * rstd_n[:, :, None, None]
            grad_x[nn, :, :] = dx.reshape(tn, c_size, s).to(x.dtype)
        gw_blocks[n_cta.id, :] = gw
        gb_blocks[n_cta.id, :] = gb
    return (
        grad_x,
        gw_blocks.sum(0).to(weight.dtype),
        gb_blocks.sum(0).to(weight.dtype),
    )


def group_norm_ref(
    grad_out: torch.Tensor,  # [N, C, S]
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    num_groups: int,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xf = x.detach().to(torch.float32).requires_grad_(True)
    wf = weight.detach().to(torch.float32).requires_grad_(True)
    bf = bias.detach().to(torch.float32).requires_grad_(True)
    y = torch.nn.functional.group_norm(xf, num_groups, wf, bf, eps)
    y.backward(grad_out.to(torch.float32))
    n_size, c_size, s = x.size()
    cg = c_size // num_groups
    xg = x.detach().to(torch.float32).reshape(n_size, num_groups, cg * s)
    mean = xg.mean(-1)  # [N, G]
    var = xg.var(-1, unbiased=False)  # [N, G]
    rstd = torch.rsqrt(var + eps)  # [N, G]
    return xf.grad, wf.grad, bf.grad, mean, rstd


# ---------------------------------------------------------------------------
# Class B.2 — instance_norm backward (3D layout [B, C, S]).
#   Normalize each (b, c) over spatial S; weight/bias per channel [C].
#   grad_weight[c] = sum over (B, S) of (x_hat * grad_out)  -> accumulator [C], reduces over B (grid) + S
#   grad_x[b,c,s]  = layer-norm bwd over S (feature axis), a DIFFERENT axis than the channel param C
# ---------------------------------------------------------------------------
@helion.kernel
def instance_norm_bwd(
    grad_out: torch.Tensor,  # [B, C, S]
    x: torch.Tensor,  # [B, C, S]
    mean: torch.Tensor,  # [B, C]
    rstd: torch.Tensor,  # [B, C]
    weight: torch.Tensor,  # [C]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b_size, c_size, s = x.size()
    c_size = hl.specialize(c_size)
    s = hl.specialize(s)
    m_block = hl.register_block_size(b_size)
    grad_x = torch.empty_like(x)
    num_blocks = (b_size + m_block - 1) // m_block
    gw_blocks = x.new_empty([num_blocks, c_size], dtype=torch.float32)
    gb_blocks = x.new_empty([num_blocks, c_size], dtype=torch.float32)
    for b_cta in hl.tile(b_size, block_size=m_block):
        gw = weight.new_zeros(c_size, dtype=torch.float32)
        gb = weight.new_zeros(c_size, dtype=torch.float32)
        w = weight[:].to(torch.float32).reshape(1, c_size, 1)
        for bb in hl.tile(b_cta.begin, b_cta.end):
            x_b = x[bb, :, :].to(torch.float32)  # [tb, C, S]
            dy_b = grad_out[bb, :, :].to(torch.float32)
            mean_b = mean[bb, :].to(torch.float32)  # [tb, C]
            rstd_b = rstd[bb, :].to(torch.float32)
            x_hat = (x_b - mean_b[:, :, None]) * rstd_b[:, :, None]  # [tb, C, S]
            gw += torch.sum(torch.sum(dy_b * x_hat, dim=-1), dim=0)  # [C]
            gb += torch.sum(torch.sum(dy_b, dim=-1), dim=0)  # [C]
            wdy = w * dy_b  # [tb, C, S]
            c1 = torch.sum(x_hat * wdy, dim=-1) / s  # [tb, C]
            c2 = torch.sum(wdy, dim=-1) / s  # [tb, C]
            dx = (wdy - (x_hat * c1[:, :, None] + c2[:, :, None])) * rstd_b[:, :, None]
            grad_x[bb, :, :] = dx.to(x.dtype)
        gw_blocks[b_cta.id, :] = gw
        gb_blocks[b_cta.id, :] = gb
    return (
        grad_x,
        gw_blocks.sum(0).to(weight.dtype),
        gb_blocks.sum(0).to(weight.dtype),
    )


def instance_norm_ref(
    grad_out: torch.Tensor,  # [B, C, S]
    x: torch.Tensor,
    weight: torch.Tensor,  # [C]
    bias: torch.Tensor,  # [C]
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xf = x.detach().to(torch.float32).requires_grad_(True)
    wf = weight.detach().to(torch.float32).requires_grad_(True)
    bf = bias.detach().to(torch.float32).requires_grad_(True)
    mean = xf.mean(-1, keepdim=True)  # [B, C, 1]
    var = xf.var(-1, unbiased=False, keepdim=True)
    rstd_full = torch.rsqrt(var + eps)
    x_hat = (xf - mean) * rstd_full
    y = x_hat * wf[None, :, None] + bf[None, :, None]
    y.backward(grad_out.to(torch.float32))
    return (
        xf.grad,
        wf.grad,
        bf.grad,
        mean.squeeze(-1).detach(),  # [B, C]
        rstd_full.squeeze(-1).detach(),
    )


# ---------------------------------------------------------------------------
# Correctness gate (sequential — one GPU)
# ---------------------------------------------------------------------------
def _chk(name: str, got: torch.Tensor, ref: torch.Tensor, atol: float = 2e-2, rtol: float = 2e-2) -> bool:
    got = got.detach().to(torch.float32)
    ref = ref.detach().to(torch.float32)
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    err = (got - ref).abs().max().item()
    denom = ref.abs().max().item() + 1e-12
    print(f"   {name:14s} max_abs={err:.3e}  rel={err/denom:.3e}  {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    dev = "cuda"
    print("helion:", helion.__file__)
    torch.manual_seed(0)

    for dtype in (torch.float32, torch.bfloat16):
        print(f"\n================= dtype={dtype} =================")
        M, N = 2048, 1024

        # --- bias_grad (A.1) ---
        print("[A.1] bias_grad")
        go = torch.randn(M, N, device=dev, dtype=dtype)
        _chk("grad_bias", bias_grad_bwd(go), bias_grad_ref(go))

        # --- dyt (A.2) ---
        print("[A.2] dyt_bwd")
        x = torch.randn(M, N, device=dev, dtype=dtype)
        w = torch.randn(N, device=dev, dtype=dtype)
        bvec = torch.randn(N, device=dev, dtype=dtype)
        go = torch.randn(M, N, device=dev, dtype=dtype)
        alpha = 0.7
        gx, gw, gb = dyt_bwd(go, x, w, alpha)
        rgx, rgw, rgb = dyt_ref(go, x, w, bvec, alpha)
        _chk("grad_x", gx, rgx)
        _chk("grad_weight", gw, rgw)
        _chk("grad_bias", gb, rgb)

        # --- group_norm (B.1) — canonical 3D [N, C, S] with spatial ---
        # Kept small: declines/fires aside, the generic [32,32] default spills the 3D
        # [inner, C, S] resident tile hard at large shapes (the byte-cap target).
        print("[B.1] group_norm_bwd  (N=128, C=64, G=8, S=64)")
        Nn, C, G, S = 128, 64, 8, 64
        x = torch.randn(Nn, C, S, device=dev, dtype=dtype)
        w = torch.randn(C, device=dev, dtype=dtype)
        bvec = torch.randn(C, device=dev, dtype=dtype)
        go = torch.randn(Nn, C, S, device=dev, dtype=dtype)
        rgx, rgw, rgb, mean, rstd = group_norm_ref(go, x, w, bvec, G)
        gx, gw, gb = group_norm_bwd(go, x, mean, rstd, w, G)
        _chk("grad_x", gx, rgx)
        _chk("grad_weight", gw, rgw)
        _chk("grad_bias", gb, rgb)

        # --- instance_norm (B.2) ---
        # NB: kept small — the kernel currently DECLINES the m-reduction seed (class B,
        # decoupled axis) and falls to the generic [32,32] config, whose 3D resident tile
        # spills hard at large shapes (the unseeded catastrophe this curriculum targets).
        print("[B.2] instance_norm_bwd  (B=64, C=16, S=128)")
        Bb, C, S = 64, 16, 128
        x = torch.randn(Bb, C, S, device=dev, dtype=dtype)
        w = torch.randn(C, device=dev, dtype=dtype)
        bvec = torch.randn(C, device=dev, dtype=dtype)
        go = torch.randn(Bb, C, S, device=dev, dtype=dtype)
        rgx, rgw, rgb, mean, rstd = instance_norm_ref(go, x, w, bvec)
        gx, gw, gb = instance_norm_bwd(go, x, mean, rstd, w)
        _chk("grad_x", gx, rgx)
        _chk("grad_weight", gw, rgw)
        _chk("grad_bias", gb, rgb)


if __name__ == "__main__":
    main()
