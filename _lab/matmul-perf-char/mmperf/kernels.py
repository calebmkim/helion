"""Per-kernel operand construction, references, and arm builders.

Each kernel exposes:
  make_inputs(spec)  -> (args tuple, ref tensor, meta dict)   [fresh randn]
  bind(args)         -> bound helion kernel  (for config extraction / compile)
  tc_callable(args)  -> a torch.compile(op, max-autotune-no-cudagraphs) callable
  flops(spec)        -> total FLOPs for TFLOP/s
The 'seed'/'helion_default' arm callables are just bound.compile_config(cfg).
Shape convention for GEMMs: (M, N, K) -- x=[M,K], y=[K,N]. bmm: [B,M,K,N]. mamba: see below.
"""

from __future__ import annotations

from typing import Any

import torch
import torch._dynamo as dyn

import helion  # noqa: F401  (import guard lives in common)
from helion._testing import HALF_DTYPE

from examples.matmul import matmul as _matmul_kernel
from examples.bmm import bmm as _bmm_kernel
from examples.fp8_gemm import fp8_gemm as _fp8_kernel
from examples.mamba2_chunk_state import (
    helion_mamba2_chunk_state_kernel as _mamba_kernel,
)
from examples.mamba2_chunk_state import ref_chunk_state as _mamba_ref


# ---------------------------------------------------------------------------
# matmul (bf16) — (M, N, K)
# ---------------------------------------------------------------------------
def matmul_make_inputs(spec: dict[str, Any]):
    # JSON key "mkn" is literally (M, K, N) order -- x=[M,K], y=[K,N].
    m, k, n = spec["mkn"]
    dt = torch.bfloat16
    x = torch.randn(m, k, device="cuda", dtype=dt)
    y = torch.randn(k, n, device="cuda", dtype=dt)
    ref = (x.float() @ y.float()).to(dt)
    return (x, y), ref, {"m": m, "n": n, "k": k}


def matmul_bind(args):
    return _matmul_kernel.bind(args)


def matmul_tc(args):
    dyn.reset()
    x, y = args
    f = torch.compile(torch.matmul, mode="max-autotune-no-cudagraphs")
    f(x, y)  # trigger compile+autotune for THIS shape
    torch.cuda.synchronize()
    return lambda: f(x, y)


def matmul_flops(spec):
    m, k, n = spec["mkn"]
    return 2 * m * n * k


# ---------------------------------------------------------------------------
# fp8_gemm (e4m3) — (M, N, K).  out is half-precision (HALF_DTYPE).
# ---------------------------------------------------------------------------
def fp8_make_inputs(spec: dict[str, Any]):
    # "mkn" is (M, K, N) order.
    m, k, n = spec["mkn"]
    xf = torch.randn(m, k, device="cuda", dtype=torch.float32)
    yf = torch.randn(k, n, device="cuda", dtype=torch.float32)
    x_fp8 = xf.to(torch.float8_e4m3fn)
    # y must be column-major for _scaled_mm (example does .T.contiguous().T)
    y_fp8 = yf.to(torch.float8_e4m3fn).T.contiguous().T
    # reference: fp32 recompute of the dequantized fp8 operands (scale=1.0)
    ref = (x_fp8.float() @ y_fp8.float()).to(HALF_DTYPE)
    return (x_fp8, y_fp8), ref, {"m": m, "n": n, "k": k}


def fp8_bind(args):
    return _fp8_kernel.bind(args)


def fp8_tc(args, fast_accum: bool = True):
    dyn.reset()
    x_fp8, y_fp8 = args
    scale_a = torch.tensor(1.0, device="cuda")
    scale_b = torch.tensor(1.0, device="cuda")

    def op(a, b):
        return torch._scaled_mm(
            a, b, scale_a, scale_b, use_fast_accum=fast_accum, out_dtype=HALF_DTYPE
        )

    f = torch.compile(op, mode="max-autotune-no-cudagraphs")
    f(x_fp8, y_fp8)
    torch.cuda.synchronize()
    return lambda: f(x_fp8, y_fp8)


def fp8_flops(spec):
    m, k, n = spec["mkn"]
    return 2 * m * n * k


# ---------------------------------------------------------------------------
# bmm (bf16) — [B, M, K, N]
# ---------------------------------------------------------------------------
def bmm_make_inputs(spec: dict[str, Any]):
    b, m, k, n = spec["bmkn"]
    dt = torch.bfloat16
    A = torch.randn(b, m, k, device="cuda", dtype=dt)
    B = torch.randn(b, k, n, device="cuda", dtype=dt)
    ref = torch.bmm(A.float(), B.float()).to(dt)
    return (A, B), ref, {"b": b, "m": m, "k": k, "n": n}


def bmm_bind(args):
    return _bmm_kernel.bind(args)


def bmm_tc(args):
    dyn.reset()
    A, B = args
    f = torch.compile(torch.bmm, mode="max-autotune-no-cudagraphs")
    f(A, B)
    torch.cuda.synchronize()
    return lambda: f(A, B)


def bmm_flops(spec):
    b, m, k, n = spec["bmkn"]
    return 2 * b * m * n * k


# ---------------------------------------------------------------------------
# mamba2_chunk_state (bf16) — [batch, seqlen, nheads, chunk, headdim, dstate]
#   B: (batch, seqlen, ngroups, dstate)   ngroups=1
#   x: (batch, seqlen, nheads, headdim)
#   dt, dA_cumsum: (batch, nheads, nchunks, chunk)
# tc arm here wraps the eager ref_chunk_state (NO cuBLAS analog -> Triton).
# ---------------------------------------------------------------------------
def mamba_make_inputs(spec: dict[str, Any]):
    batch, seqlen, nheads, chunk, headdim, dstate = spec["s"]
    ngroups = 1
    nchunks = (seqlen + chunk - 1) // chunk
    dt_ = HALF_DTYPE
    # 'uuuu' init in the example (uniform [0,1)) — keeps exp/decay well-conditioned.
    B = torch.rand(batch, seqlen, ngroups, dstate, device="cuda", dtype=dt_)
    x = torch.rand(batch, seqlen, nheads, headdim, device="cuda", dtype=dt_)
    dt = torch.rand(batch, nheads, nchunks, chunk, device="cuda", dtype=dt_)
    dA = torch.rand(batch, nheads, nchunks, chunk, device="cuda", dtype=dt_)
    ref = _mamba_ref(B, x, dt, dA)
    return (
        (B, x, dt, dA),
        ref,
        {
            "batch": batch,
            "seqlen": seqlen,
            "nheads": nheads,
            "chunk": chunk,
            "headdim": headdim,
            "dstate": dstate,
            "nchunks": nchunks,
        },
    )


def mamba_bind(args):
    return _mamba_kernel.bind(args)


def mamba_tc(args):
    dyn.reset()
    B, x, dt, dA = args
    f = torch.compile(_mamba_ref, mode="max-autotune-no-cudagraphs")
    f(B, x, dt, dA)
    torch.cuda.synchronize()
    return lambda: f(B, x, dt, dA)


def mamba_flops(spec):
    # dominant dot: for each (batch, chunk_block, nheads) a [headdim x chunk] @
    # [chunk x dstate]. Total ~ 2 * batch * nheads * nchunks * chunk * headdim * dstate.
    batch, seqlen, nheads, chunk, headdim, dstate = spec["s"]
    nchunks = (seqlen + chunk - 1) // chunk
    return 2 * batch * nheads * nchunks * chunk * headdim * dstate


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
KERNELS: dict[str, dict[str, Any]] = {
    "matmul": {
        "make_inputs": matmul_make_inputs,
        "bind": matmul_bind,
        "tc": matmul_tc,
        "flops": matmul_flops,
        "dtype": "bf16",
        "shape_key": "mkn",
        "tc_is_cublas": True,
    },
    "fp8_gemm": {
        "make_inputs": fp8_make_inputs,
        "bind": fp8_bind,
        "tc": fp8_tc,
        "flops": fp8_flops,
        "dtype": "fp8_e4m3",
        "shape_key": "mkn",
        "tc_is_cublas": True,
    },
    "bmm": {
        "make_inputs": bmm_make_inputs,
        "bind": bmm_bind,
        "tc": bmm_tc,
        "flops": bmm_flops,
        "dtype": "bf16",
        "shape_key": "bmkn",
        "tc_is_cublas": True,
    },
    "mamba2_chunk_state": {
        "make_inputs": mamba_make_inputs,
        "bind": mamba_bind,
        "tc": mamba_tc,
        "flops": mamba_flops,
        "dtype": "bf16",
        "shape_key": "s",
        "tc_is_cublas": False,
    },
}


def shape_tuple(kernel: str, spec: dict[str, Any]):
    return tuple(spec[KERNELS[kernel]["shape_key"]])
