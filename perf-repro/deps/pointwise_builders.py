"""PR #2866 pointwise builders. One builder per kernel, returning the harness's
(kfn, args, ref_out, acc_fn, tc_ref) contract.

Kernels:
  flat family (examples/):     swiglu (_swiglu_fwd), geglu (_geglu), residual_add (add)
  flat family (vendored):      relu_squared, bias_gelu, dyt      (deps/pointwise_kernels.py)
  levers:                      rope_fwd (examples/rope.py),
                               transposed_out_add, heavy_transcendental_1d (vendored)

All bf16. Shapes:
  2-D (M, N):        swiglu, geglu, residual_add, relu_squared, bias_gelu, dyt,
                     transposed_out_add
  4-D (b, h, s, d):  rope_fwd
  1-D (numel,):      heavy_transcendental_1d

The accuracy fn / tc_ref match the harness convention used by _cur_build:
  acc = _acc_close(rtol, atol)  (or _acc_tuple for multi-output rope)
  tc_ref = a zero-arg callable running the standalone torch reference (fusion-friendly, no
           .item()/scatter/graph-break).
"""

from __future__ import annotations

import torch

DEV = "cuda"

# tolerances: bf16 elementwise, fp32-internal compute. Matches _lab ptw_bench (0.05/0.005)
# but the harness curriculum bf16 default is (2e-2, 2e-2); we use the looser lab value the
# kernels were validated at to avoid false fp32-vs-bf16 rounding failures.
_RTOL, _ATOL = 2e-2, 2e-2


def _rn(*shape, dt=torch.bfloat16):
    return torch.randn(*shape, device=DEV, dtype=dt)


# --------------------------------------------------------------------------- #
#  Flat family — examples/
# --------------------------------------------------------------------------- #
def _build_swiglu(shape, dt):
    import examples.swiglu as SW

    m, n = shape
    a, b = _rn(m, n, dt=dt), _rn(m, n, dt=dt)
    args = (a, b)

    def ref_fn():
        return (torch.nn.functional.silu(a.float()) * b.float()).to(a.dtype)

    return SW._swiglu_fwd, args, ref_fn(), _RTOL, _ATOL, ref_fn, "close"


def _build_geglu(shape, dt):
    import examples.geglu as GE

    m, n = shape
    a, b = _rn(m, n, dt=dt), _rn(m, n, dt=dt)
    args = (a, b)
    _c = 0.7978845608028654

    def ref_fn():
        af = a.float()
        gelu = 0.5 * af * (1.0 + torch.tanh(_c * (af + 0.044715 * af * af * af)))
        return (gelu * b.float()).to(a.dtype)

    return GE._geglu, args, ref_fn(), _RTOL, _ATOL, ref_fn, "close"


def _build_residual_add(shape, dt):
    import examples.add as ADD

    m, n = shape
    x, y = _rn(m, n, dt=dt), _rn(m, n, dt=dt)
    args = (x, y)

    def ref_fn():
        return x + y

    return ADD.add, args, ref_fn(), _RTOL, _ATOL, ref_fn, "close"


# --------------------------------------------------------------------------- #
#  Flat family — vendored (relu_squared, bias_gelu, dyt)
# --------------------------------------------------------------------------- #
def _build_relu_squared(shape, dt):
    import pointwise_kernels as PK

    m, n = shape
    x = _rn(m, n, dt=dt)
    args = (x,)

    def ref_fn():
        return PK.ref_relu_squared(x)

    return PK.relu_squared, args, ref_fn(), _RTOL, _ATOL, ref_fn, "close"


def _build_bias_gelu(shape, dt):
    import pointwise_kernels as PK

    m, n = shape
    x, bias = _rn(m, n, dt=dt), _rn(n, dt=dt)
    args = (x, bias)

    def ref_fn():
        return PK.ref_bias_gelu(x, bias)

    return PK.bias_gelu, args, ref_fn(), _RTOL, _ATOL, ref_fn, "close"


def _build_dyt(shape, dt):
    import pointwise_kernels as PK

    m, n = shape
    x, gamma, beta = _rn(m, n, dt=dt), _rn(n, dt=dt), _rn(n, dt=dt)
    alpha = 0.5
    args = (x, gamma, beta, alpha)

    def ref_fn():
        return PK.ref_dyt(x, gamma, beta, alpha)

    return PK.dyt, args, ref_fn(), _RTOL, _ATOL, ref_fn, "close"


# --------------------------------------------------------------------------- #
#  Levers
# --------------------------------------------------------------------------- #
def _build_transposed_out_add(shape, dt):
    import pointwise_kernels as PK

    m, n = shape
    x, y = _rn(m, n, dt=dt), _rn(m, n, dt=dt)
    args = (x, y)

    def ref_fn():
        return PK.ref_transposed_out_add(x, y)

    return PK.transposed_out_add, args, ref_fn(), _RTOL, _ATOL, ref_fn, "close"


def _build_heavy_transcendental_1d(shape, dt):
    import pointwise_kernels as PK

    (numel,) = shape
    x = _rn(numel, dt=dt)
    args = (x,)

    def ref_fn():
        return PK.ref_heavy_transcendental_1d(x)

    return PK.heavy_transcendental_1d, args, ref_fn(), _RTOL, _ATOL, ref_fn, "close"


def _build_rope_fwd(shape, dt):
    """rope_fwd(q, k, cos, sin) -> (q_out, k_out). shape = (batch, heads, seq, head_dim).
    k uses the same head count as q (self-attn). cos/sin are [batch, seq, head_dim].
    Reference: examples.rope.rope_pytorch (handles the dim()==3 unsqueeze)."""
    import examples.rope as RP

    batch, heads, seq, head_dim = shape
    q = _rn(batch, heads, seq, head_dim, dt=dt)
    k = _rn(batch, heads, seq, head_dim, dt=dt)
    angles = _rn(batch, seq, head_dim, dt=dt)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    args = (q, k, cos, sin)

    def ref_fn():
        return RP.rope_pytorch(q, k, cos, sin)

    # multi-output -> tuple accuracy
    return RP.rope_fwd, args, ref_fn(), _RTOL, _ATOL, ref_fn, "tuple"


_BUILDERS = {
    "swiglu": _build_swiglu,
    "geglu": _build_geglu,
    "residual_add": _build_residual_add,
    "relu_squared": _build_relu_squared,
    "bias_gelu": _build_bias_gelu,
    "dyt": _build_dyt,
    "transposed_out_add": _build_transposed_out_add,
    "heavy_transcendental_1d": _build_heavy_transcendental_1d,
    "rope_fwd": _build_rope_fwd,
}


def build(kernel, shape, dt):
    """Return (kfn, args, ref_out, rtol, atol, tc_ref, acc_kind)."""
    if kernel not in _BUILDERS:
        raise KeyError(f"unknown pointwise kernel: {kernel}")
    return _BUILDERS[kernel](tuple(shape), dt)
