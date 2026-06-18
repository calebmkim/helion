"""Stage-3 corpus: fused matmul + reduction-over-N-epilogue kernels.

Every kernel shares ONE matmul skeleton (copied verbatim from
`_lab/matmul_rms_norm_template.py`):

    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        <EPILOGUE: a reduction over the N (output) axis on the register-resident acc>
        out[tile_m, :] = ...

Only the EPILOGUE block changes between members. N is `hl.specialize`d (compile-time
constant, never tiled) so the [tile_m, n] fp32 accumulator stays register-resident --
this is the structural invariant the Stage-3 heuristic targets.

Each kernel was authored from its OWN named torch/library reference (the Gate-E
circularity firewall), and each `*_ref` computes the matmul in fp32, applies the
epilogue in fp32, and casts back to the promoted in-dtype (the accuracy oracle).

FIT set    : matmul_rms_norm, matmul_layernorm, matmul_softmax, matmul_l2_normalize,
             matmul_sum (scalar [M,1] output DOF).
HELD-OUT   : matmul_logsumexp ([M,1]), matmul_max ([M,1]).

Run `python matmul_epilogue_kernels.py` for a tiny correctness check of every FIT
kernel at M=131072, K=256, N=256, bf16.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import helion
import helion.language as hl


# =============================================================================
# FIT set
# =============================================================================
@helion.kernel(static_shapes=True)
def matmul_rms_norm(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """out = rms_norm(x @ y, weight) over the N axis.  Copied from the template.

    x: [M, K], y: [K, N], weight: [N] -> out: [M, N].
    """
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        # ---- EPILOGUE ----
        eps = 1e-6
        ms = (acc * acc).sum(dim=-1, keepdim=True) / n
        normed = acc * torch.rsqrt(ms + eps)
        out[tile_m, :] = normed * weight[:].to(torch.float32)
        # ------------------
    return out


def matmul_rms_norm_ref(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    mm = torch.matmul(x, y).to(torch.float32)
    ms = mm.pow(2).mean(dim=-1, keepdim=True)
    normed = mm * torch.rsqrt(ms + 1e-6)
    return (normed * weight.to(torch.float32)).to(
        torch.promote_types(x.dtype, y.dtype)
    )


@helion.kernel(static_shapes=True)
def matmul_layernorm(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """out = layer_norm(x @ y, weight, bias) over N.  Two reductions (mean + var).

    Epilogue lifted from examples/matmul_layernorm.py (the in-tree carrier).
    x: [M, K], y: [K, N], weight: [N], bias: [N] -> out: [M, N].
    """
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))
    assert k == k2, f"size mismatch {k} != {k2}"
    assert weight.size(0) == n, f"weight size mismatch {weight.size(0)} != {n}"
    assert bias.size(0) == n, f"bias size mismatch {bias.size(0)} != {n}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        # ---- EPILOGUE ----
        eps = 1e-5
        mean = acc.sum(dim=-1, keepdim=True) / n
        centered = acc - mean
        var = (centered * centered).sum(dim=-1, keepdim=True) / n
        normalized = centered * torch.rsqrt(var + eps)
        out[tile_m, :] = normalized * weight[:].to(torch.float32) + bias[:].to(
            torch.float32
        )
        # ------------------
    return out


def matmul_layernorm_ref(
    x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    mm = torch.matmul(x, y).to(torch.float32)
    ln = F.layer_norm(
        mm,
        normalized_shape=[mm.shape[-1]],
        weight=weight.to(torch.float32),
        bias=bias.to(torch.float32),
        eps=1e-5,
    )
    return ln.to(torch.promote_types(x.dtype, y.dtype))


@helion.kernel(static_shapes=True)
def matmul_softmax(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """out = softmax(x @ y, dim=-1) over N.  max + sum -> exp-normalize, full-width.

    Reference: torch.softmax over the last axis.
    x: [M, K], y: [K, N] -> out: [M, N].
    """
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        # ---- EPILOGUE ----
        amax = acc.amax(dim=-1, keepdim=True)
        e = torch.exp(acc - amax)
        denom = e.sum(dim=-1, keepdim=True)
        out[tile_m, :] = e / denom
        # ------------------
    return out


def matmul_softmax_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mm = torch.matmul(x, y).to(torch.float32)
    sm = torch.softmax(mm, dim=-1)
    return sm.to(torch.promote_types(x.dtype, y.dtype))


@helion.kernel(static_shapes=True)
def matmul_l2_normalize(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """out = (x @ y) / ||x @ y||_2 over N.  sum-sq -> rsqrt scale, full-width.

    Reference: torch.nn.functional.normalize(., p=2, dim=-1).
    x: [M, K], y: [K, N] -> out: [M, N].
    """
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, n], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        # ---- EPILOGUE ----
        eps = 1e-12
        sumsq = (acc * acc).sum(dim=-1, keepdim=True)
        out[tile_m, :] = acc * torch.rsqrt(torch.clamp(sumsq, min=eps))
        # ------------------
    return out


def matmul_l2_normalize_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mm = torch.matmul(x, y).to(torch.float32)
    out = F.normalize(mm, p=2.0, dim=-1, eps=1e-12)
    return out.to(torch.promote_types(x.dtype, y.dtype))


@helion.kernel(static_shapes=True)
def matmul_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """out = (x @ y).sum(-1) over N -> scalar [M, 1] (the scalar-output DOF).

    Reference: (x @ y).sum(dim=-1, keepdim=True).
    x: [M, K], y: [K, N] -> out: [M, 1].
    """
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, 1], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        # ---- EPILOGUE ----
        out[tile_m, :] = acc.sum(dim=-1, keepdim=True)
        # ------------------
    return out


def matmul_sum_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mm = torch.matmul(x, y).to(torch.float32)
    s = mm.sum(dim=-1, keepdim=True)
    return s.to(torch.promote_types(x.dtype, y.dtype))


# =============================================================================
# HELD-OUT set (read once at freeze; not used to tune)
# =============================================================================
@helion.kernel(static_shapes=True)
def matmul_logsumexp(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """out = logsumexp(x @ y, dim=-1) over N -> scalar [M, 1].

    Reference: torch.logsumexp(., dim=-1, keepdim=True).
    x: [M, K], y: [K, N] -> out: [M, 1].
    """
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, 1], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        # ---- EPILOGUE ----
        amax = acc.amax(dim=-1, keepdim=True)
        lse = amax + torch.log(torch.exp(acc - amax).sum(dim=-1, keepdim=True))
        out[tile_m, :] = lse
        # ------------------
    return out


def matmul_logsumexp_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mm = torch.matmul(x, y).to(torch.float32)
    lse = torch.logsumexp(mm, dim=-1, keepdim=True)
    return lse.to(torch.promote_types(x.dtype, y.dtype))


@helion.kernel(static_shapes=True)
def matmul_max(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """out = (x @ y).amax(-1) over N -> scalar [M, 1] (max, not argmax, for a float output).

    Reference: (x @ y).amax(dim=-1, keepdim=True).
    x: [M, K], y: [K, N] -> out: [M, 1].
    """
    m, k = x.size()
    k2 = y.size(0)
    n = hl.specialize(y.size(1))
    assert k == k2, f"size mismatch {k} != {k2}"
    out = torch.empty(
        [m, 1], dtype=torch.promote_types(x.dtype, y.dtype), device=x.device
    )
    for tile_m in hl.tile(m):
        acc = hl.zeros([tile_m, n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, :])
        # ---- EPILOGUE ----
        out[tile_m, :] = acc.amax(dim=-1, keepdim=True)
        # ------------------
    return out


def matmul_max_ref(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mm = torch.matmul(x, y).to(torch.float32)
    mx = mm.amax(dim=-1, keepdim=True)
    return mx.to(torch.promote_types(x.dtype, y.dtype))


# =============================================================================
# Registry + correctness main()
# =============================================================================
# name -> (kernel, ref, extra-arg-builder over (m, k, n, dtype, device))
def _w(n, dtype, device):
    return (torch.randn(n, device=device, dtype=dtype),)


def _wb(n, dtype, device):
    return (
        torch.randn(n, device=device, dtype=dtype),
        torch.randn(n, device=device, dtype=dtype),
    )


def _none(n, dtype, device):
    return ()


FIT = {
    "matmul_rms_norm": (matmul_rms_norm, matmul_rms_norm_ref, _w),
    "matmul_layernorm": (matmul_layernorm, matmul_layernorm_ref, _wb),
    "matmul_softmax": (matmul_softmax, matmul_softmax_ref, _none),
    "matmul_l2_normalize": (matmul_l2_normalize, matmul_l2_normalize_ref, _none),
    "matmul_sum": (matmul_sum, matmul_sum_ref, _none),
}

HELD_OUT = {
    "matmul_logsumexp": (matmul_logsumexp, matmul_logsumexp_ref, _none),
    "matmul_max": (matmul_max, matmul_max_ref, _none),
}

KERNELS = {**FIT, **HELD_OUT}


# Measured accuracy gate (see CORPUS.md "Accuracy floors").  The matmul accumulation
# order differs from cuBLAS, so over a wide-N reduction the per-element abs error is large
# in absolute terms but tiny relative to the OUTPUT SCALE.  We therefore gate on
# max_abs / output-RMS for every kernel whose output magnitude varies with N
# (norm/l2/sum/logsumexp/max), and on plain max_abs for softmax (a bounded [0,1] output
# whose RMS is dominated by tiny non-peak probs -> rel-to-RMS would be a false alarm).
# Thresholds are the measured bf16 floor + headroom; fp16/fp32 are tighter.
_REL_RMS_TOL = {"bf16": 0.07, "fp16": 0.03, "fp32": 0.02}
_SOFTMAX_ABS_TOL = {"bf16": 0.09, "fp16": 0.03, "fp32": 0.02}


def acc_ok(name, out, ref_out, dt_name):
    """fp32-upcast accuracy gate (footgun #6).  Returns (ok, detail-string)."""
    o = out.float()
    r = ref_out.float()
    if not torch.isfinite(o).all():
        return False, "non-finite"
    d = (o - r).abs()
    if name == "matmul_softmax":
        max_abs = d.max().item()
        return max_abs <= _SOFTMAX_ABS_TOL[dt_name], f"max_abs={max_abs:.4g}"
    rms = r.pow(2).mean().sqrt().item()
    rel = (d.max().item() / rms) if rms > 1e-9 else d.max().item()
    return rel <= _REL_RMS_TOL[dt_name], f"rel_to_rms={rel:.4g}"


def _check_one(name, m, k, n, dtype, device):
    kernel, ref, extra = KERNELS[name]
    x = torch.randn(m, k, device=device, dtype=dtype)
    y = torch.randn(k, n, device=device, dtype=dtype)
    extra_args = extra(n, dtype, device)
    out = kernel(x, y, *extra_args)
    ref_out = ref(x, y, *extra_args)
    dt_name = {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[
        dtype
    ]
    ok, detail = acc_ok(name, out, ref_out, dt_name)
    print(
        f"  {'PASS' if ok else 'FAIL':4} {name:22} "
        f"M={m} K={k} N={n} {str(dtype).split('.')[-1]:8} {detail}"
    )
    return ok


def main() -> None:
    device = "cuda"
    m, k, n, dtype = 131072, 256, 256, torch.bfloat16
    print(f"Stage-3 corpus correctness check @ M={m} K={k} N={n} bf16")
    all_ok = True
    for name in KERNELS:
        all_ok &= _check_one(name, m, k, n, dtype, device)
    print("ALL PASS" if all_ok else "SOME FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
