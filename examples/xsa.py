"""
Exclusive Self-Attention (XSA) Example
======================================

This example demonstrates a Helion kernel that fuses the exclusive
self-attention (XSA) forward pass on top of standard non-causal
self-attention. After computing the attention output ``y``, XSA subtracts the
projection of ``y`` onto the L2-normalized value vector for the same token::

    y = softmax(Q @ K ^ T / sqrt(d)) @ V
    vn = normalize(
        V, dim=-1, eps=eps
    )  # F.normalize semantics: divide by max(||v||, eps)
    z = y - (y * vn).sum(dim=-1, keepdim=True) * vn

The win over an unfused implementation is that ``y`` is never materialized to
HBM and read back by a separate epilogue kernel. The kernel still reads ``V``
once in the inner attention loop and a second time in the per-tile epilogue to
project ``y`` onto ``vn`` for the current rows.

Scope of this first pass:

- Non-causal attention.
- Forward only.
- Pre-projected ``Q/K/V`` of shape ``(B, H, T, D)``.
- Self-attention only (``Q``, ``K``, ``V`` share sequence length ``T``).
- No output projection ``W_o``.
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import helion
from helion._testing import DEVICE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
# XSA Kernel
# ----------


# %%
@helion.kernel(
    # Static shapes provides a speedup for attention.
    static_shapes=True,
)
def xsa_kernel(
    q_in: torch.Tensor,
    k_in: torch.Tensor,
    v_in: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused exclusive self-attention (XSA) forward kernel.

    Args:
        q_in: Query tensor of shape ``[..., T, D]``.
        k_in: Key tensor of shape ``[..., T, D]``.
        v_in: Value tensor of shape ``[..., T, D]``.
        eps: Lower bound on the per-token L2 norm of ``V`` used to match
            ``F.normalize`` semantics (``vn = v / max(||v||, eps)``).

    Returns:
        Output tensor with the same shape and dtype as ``q_in``.
    """
    m_dim = q_in.size(-2)
    n_dim = k_in.size(-2)
    assert n_dim == v_in.size(-2)
    # Self-attention only in this first pass.
    assert n_dim == m_dim, (
        "xsa_kernel is self-attention only: Q, K, V must share sequence length"
    )
    head_dim = hl.specialize(q_in.size(-1))
    assert head_dim == k_in.size(-1) == v_in.size(-1)
    q_view = q_in.reshape([-1, m_dim, head_dim])
    v_view = v_in.reshape([-1, n_dim, head_dim])
    k_view = k_in.reshape([-1, n_dim, head_dim])
    out = torch.empty_like(q_view)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # 1/log(2)
    for tile_b, tile_m in hl.tile([q_view.size(0), m_dim]):
        m_i = hl.full([tile_b, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_b, tile_m, head_dim], dtype=torch.float32)
        q = q_view[tile_b, tile_m, :]
        for tile_n in hl.tile(v_view.size(1)):
            # Online softmax, mirroring examples/attention.py.
            q_scaled = q * qk_scale
            k = k_view[tile_b, tile_n, :]
            qk = torch.bmm(q_scaled, k.transpose(1, 2), torch.float32)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1))
            qk = qk - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]
            v = v_view[tile_b, tile_n, :]
            p = p.to(v.dtype)
            acc = torch.baddbmm(acc, p, v)
            m_i = m_ij
        acc = acc / l_i[:, :, None]
        # XSA epilogue. Match ``F.normalize(v.float(), dim=-1, eps=eps)``:
        # divide by max(||v||, eps), not by sqrt(||v||^2 + eps).
        v_self = v_view[tile_b, tile_m, :].to(torch.float32)
        v_sq_sum = torch.sum(v_self * v_self, dim=-1, keepdim=True)
        v_norm = torch.sqrt(v_sq_sum)
        v_denom = torch.clamp(v_norm, min=eps)
        vn = v_self / v_denom
        proj = torch.sum(acc * vn, dim=-1, keepdim=True)
        z = acc - proj * vn
        out[tile_b, tile_m, :] = z.to(out.dtype)
    return out.view(q_in.size())


# %%
# Pure-torch Reference
# --------------------


# %%
def ref_xsa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Pure-torch reference for the fused XSA forward."""
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    p = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    y = torch.matmul(p, v)
    vn = torch.nn.functional.normalize(v.float(), dim=-1, eps=eps)
    z = y.float() - (y.float() * vn).sum(dim=-1, keepdim=True) * vn
    return z.to(y.dtype)


# %%
# Tritonbench Wrapper
# -------------------


# %%
def xsa_tritonbench(
    tb_op: object,
    *args: torch.Tensor,
) -> Callable[[], list[torch.Tensor]]:
    """TritonBench wrapper used by ``benchmarks/run.py --kernel xsa``.

    The XSA TritonBench operator wraps each ``(q, k, v)`` triple through a
    ``multi_input_wrapper``, so a single benchmark input may contain multiple
    Q/K/V triples (e.g. with ``--gen-cache-size-inputs``). Mirror that here by
    returning a list of outputs, one per triple.
    """
    assert len(args) % 3 == 0, (
        f"xsa_tritonbench expects (q, k, v) triples, got {len(args)} tensors"
    )
    # pyrefly: ignore [missing-attribute]
    eps = getattr(tb_op, "eps", 1e-6)

    def run() -> list[torch.Tensor]:
        outputs = []
        for i in range(0, len(args), 3):
            q, k, v = args[i : i + 3]
            outputs.append(xsa_kernel(q, k, v, eps))
        return outputs

    return run


# %%
# Verification Functions
# ----------------------


# %%
def check(
    z: int,
    h: int,
    t: int,
    d: int,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Check ``xsa_kernel`` against the pure-torch reference for one shape."""
    eps = 1e-6
    q = torch.randn((z, h, t, d), dtype=dtype, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    run_example(
        lambda q, k, v: xsa_kernel(q, k, v, eps),
        lambda q, k, v: ref_xsa(q, k, v, eps),
        (q, k, v),
        kernel_name="helion_xsa",
        baseline_name="torch_eager",
    )


def check_near_zero_v(
    z: int,
    h: int,
    t: int,
    d: int,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Check ``F.normalize(..., eps=eps)`` semantics on a near-zero ``V_i`` row.

    The output must remain finite even when ``||V_i|| < eps``; the denominator
    is clamped at ``eps`` rather than letting division by zero blow up.
    """
    eps = 1e-6
    q = torch.randn((z, h, t, d), dtype=dtype, device=DEVICE)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    # Force a row of V to be exactly zero so ||V_i|| = 0 < eps.
    v[..., 0, :] = 0.0

    out = xsa_kernel(q, k, v, eps)
    assert torch.isfinite(out).all(), (
        "xsa_kernel produced non-finite values on a zero V row; "
        "check F.normalize(..., eps=eps) semantics."
    )
    expected = ref_xsa(q, k, v, eps)
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)


# %%
# Main Function
# -------------


# %%
def main() -> None:
    """Run the XSA kernel correctness check against the pure-torch reference."""
    check(2, 4, 128, 64)
    check(2, 4, 512, 64)
    check_near_zero_v(2, 4, 128, 64)


if __name__ == "__main__":
    main()
