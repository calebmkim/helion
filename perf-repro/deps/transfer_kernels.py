"""Transfer-test kernels for the merged reduction seed heuristic (PR #2762).

These are reduction + pointwise fused kernels (and one quantization kernel) that the
heuristic was NOT tuned on (none is in the merged curriculum), used to test whether the
seed's perf generalizes to real-world fused patterns. Every kernel here:
  * has exactly ONE reduction axis and NO matmul in the @helion.kernel body, so it FIRES
    the seed heuristic (_triton_reduction_eligible: 1 ReductionFact, no matmul_facts);
  * is forward-only (the lab harness is fwd-only);
  * promotes the reduced row to fp32 (matches the curriculum norm/loss family).

The two other transfer kernels (GRPO, fused_linear_jsd) already exist in examples/ and are
wired in the harness directly, not re-implemented here.

Each kernel ships with a pure-torch reference (used both for the accuracy gate and as the
torch.compile baseline arm).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

# =========================================================================== #
#  Bucket 1 — reduction + pointwise fused
# =========================================================================== #


@helion.kernel
def fused_add_rmsnorm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """h = x + residual ; out = rms_norm(h) * w ; returns (out, h).

    The #1 real-world fused reduction (vLLM/TRT-LLM `fused_add_rms_norm`): the residual
    add is the pointwise prologue, mean(h^2) is the single reduction, the scale is the
    epilogue, and the updated residual `h` is returned for the next block. T1 (rollable
    rdim), full-width output, one extra input load (residual).
    """
    m, n = x.size()
    out = torch.empty_like(x)
    res_out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        h = x[tile_m, :].to(torch.float32) + residual[tile_m, :].to(torch.float32)
        var = torch.mean(h * h, dim=-1)
        inv = torch.rsqrt(var + eps)
        normed = h * inv[:, None]
        out[tile_m, :] = (normed * weight[:].to(torch.float32)).to(out.dtype)
        res_out[tile_m, :] = h.to(res_out.dtype)
    return out, res_out


def ref_fused_add_rmsnorm(x, residual, weight, eps=1e-6):
    h = x.float() + residual.float()
    var = h.pow(2).mean(-1, keepdim=True)
    out = (h * torch.rsqrt(var + eps)) * weight.float()
    return out.to(x.dtype), h.to(x.dtype)


@helion.kernel
def fused_add_layernorm_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """h = x + residual ; out = layer_norm(h) * w + b ; returns (out, h).

    LN-family residual block. mean + variance are two reductions over the SAME rdim, so
    Helion rolls one reduction loop -> one ReductionFact -> fires T1 (the chained-same-axis
    case). Full-width output, one extra input load (residual).
    """
    m, n = x.size()
    out = torch.empty_like(x)
    res_out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        h = x[tile_m, :].to(torch.float32) + residual[tile_m, :].to(torch.float32)
        mean = torch.sum(h, dim=-1) / n
        centered = h - mean[:, None]
        var = torch.sum(centered * centered, dim=-1) / n
        inv = torch.rsqrt(var + eps)
        normed = centered * inv[:, None]
        affine = normed * weight[:].to(torch.float32) + bias[:].to(torch.float32)
        out[tile_m, :] = affine.to(out.dtype)
        res_out[tile_m, :] = h.to(res_out.dtype)
    return out, res_out


def ref_fused_add_layernorm(x, residual, weight, bias, eps=1e-5):
    h = x.float() + residual.float()
    out = torch.nn.functional.layer_norm(h, [h.size(-1)], weight.float(), bias.float(), eps)
    return out.to(x.dtype), h.to(x.dtype)


@helion.kernel
def scaled_masked_softmax_fwd(
    x: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """softmax(x * scale + mask, dim=-1). Attention's reduction+pointwise (scale + additive
    mask prologue, then softmax). amax + sum over the SAME rdim -> one ReductionFact -> T1.
    Full-width output.
    """
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        s = x[tile_m, :].to(torch.float32) * scale + mask[tile_m, :].to(torch.float32)
        amax = torch.amax(s, dim=-1, keepdim=True)
        e = torch.exp(s - amax)
        denom = torch.sum(e, dim=-1, keepdim=True)
        out[tile_m, :] = (e / denom).to(out.dtype)
    return out


def ref_scaled_masked_softmax(x, mask, scale):
    s = x.float() * scale + mask.float()
    return torch.softmax(s, dim=-1).to(x.dtype)


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def cross_entropy_ls_zloss_fwd(
    logits: torch.Tensor,  # [N, V]
    labels: torch.Tensor,  # [N]
    smoothing: float = 0.1,
    lambda_z: float = 1e-4,
) -> torch.Tensor:
    """Cross-entropy + label-smoothing + z-loss (LM pretraining loss).

    per-row: lse = logsumexp(row); nll = lse - row[target]; smooth = lse - mean(row);
             loss_i = (1-eps)*nll + eps*smooth + lambda_z * lse^2
    Reduction over V (amax + sum + sum-of-logits, all same rdim) + a target gather. Scalar
    per-row output. T1.
    """
    n, v = logits.shape
    losses = torch.zeros([n], dtype=torch.float32, device=logits.device)
    logits_flat = logits.view(-1)
    for tile_n in hl.tile(n):
        labels_tile = labels[tile_n]
        flat_indices = tile_n.index * v + labels_tile
        logit_target = hl.load(logits_flat, [flat_indices]).to(torch.float32)

        row = logits[tile_n, :].to(torch.float32)
        amax = torch.amax(row, dim=-1)
        shifted = row - amax[:, None]
        sumexp = torch.sum(torch.exp(shifted), dim=-1)
        lse = amax + torch.log(sumexp)
        mean_logits = torch.sum(row, dim=-1) / v

        nll = lse - logit_target
        smooth = lse - mean_logits
        loss_i = (1.0 - smoothing) * nll + smoothing * smooth + lambda_z * lse * lse
        losses[tile_n] = loss_i
    return losses.mean()


def ref_cross_entropy_ls_zloss(logits, labels, smoothing=0.1, lambda_z=1e-4):
    logits = logits.float()
    lse = torch.logsumexp(logits, dim=-1)
    logit_target = logits.gather(1, labels[:, None]).squeeze(1)
    nll = lse - logit_target
    smooth = lse - logits.mean(-1)
    loss = (1.0 - smoothing) * nll + smoothing * smooth + lambda_z * lse * lse
    return loss.mean()


@helion.kernel
def gated_rmsnorm_fwd(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """out = rms_norm(x) * w * silu(gate). Gated norm (Mamba2 RMSNormGated / gated attn).
    mean(x^2) is the single reduction; the silu-gate epilogue is compute-heavier than a
    plain scale and adds one input load. T1, full-width.
    """
    m, n = x.size()
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        xf = x[tile_m, :].to(torch.float32)
        var = torch.mean(xf * xf, dim=-1)
        inv = torch.rsqrt(var + eps)
        normed = xf * inv[:, None] * weight[:].to(torch.float32)
        g = gate[tile_m, :].to(torch.float32)
        silu_g = g * torch.sigmoid(g)
        out[tile_m, :] = (normed * silu_g).to(out.dtype)
    return out


def ref_gated_rmsnorm(x, gate, weight, eps=1e-6):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    normed = xf * torch.rsqrt(var + eps) * weight.float()
    silu_g = torch.nn.functional.silu(gate.float())
    return (normed * silu_g).to(x.dtype)


# =========================================================================== #
#  Bucket 2 — new kernel: dynamic per-token quantization scale
# =========================================================================== #


@helion.kernel
def dynamic_quant_fwd(
    x: torch.Tensor,
    qmax: float = 127.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (row) dynamic int8 quantization.

    scale = amax(|x|, dim=-1) / qmax ; q = round(clamp(x/scale, -qmax, qmax)).
    The REDUCTION is the amax (absolute-max) over the row; the quantize (divide, round,
    clamp, cast to int8) is the pointwise epilogue with a 1-byte full-width output.
    Ubiquitous in fp8/int8 training+inference. T1.
    """
    m, n = x.size()
    q = torch.empty([m, n], dtype=torch.int8, device=x.device)
    scale = torch.empty([m], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        amax = torch.amax(torch.abs(row), dim=-1)
        s = torch.clamp(amax / qmax, min=1e-12)
        scale[tile_m] = s
        qrow = torch.round(row / s[:, None])
        qrow = torch.clamp(qrow, -qmax, qmax)
        q[tile_m, :] = qrow.to(torch.int8)
    return q, scale


def ref_dynamic_quant(x, qmax=127.0):
    xf = x.float()
    amax = xf.abs().amax(dim=-1)
    s = (amax / qmax).clamp(min=1e-12)
    qrow = torch.round(xf / s[:, None]).clamp(-qmax, qmax)
    return qrow.to(torch.int8), s
