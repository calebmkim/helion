"""Pure-torch references mirroring each vLLM Helion kernel's exact arithmetic.

(The kernels' own baseline() functions call torch.ops._C.*, vLLM's compiled
extension, which we don't have standalone — so these reproduce the same math.)
Each ref takes the SAME positional args as the kernel and writes the same
in-place outputs (or returns, for silu).
"""

import torch

F = torch.nn.functional


def _fp8_min_max():
    fi = torch.finfo(torch.float8_e4m3fn)
    return fi.min, fi.max


def ref_silu_mul_fp8(input, scale):
    d = input.shape[-1] // 2
    x = input.view(-1, input.shape[-1])
    a, b = x[:, :d], x[:, d:]
    prod = (F.silu(a) * b).to(torch.float32)
    out = (prod * (1.0 / scale)).to(torch.float8_e4m3fn)
    return out.view(input.shape[:-1] + (d,))


def ref_dynamic_per_token(result, input, scale, scale_ub=None):
    fp8_min, fp8_max = _fp8_min_max()
    min_sf = 1.0 / (fp8_max * 512.0)
    x = input.to(torch.float32)
    s = x.abs().amax(dim=-1)
    if scale_ub is not None:
        s = s.clamp(max=scale_ub.item())
    s = (s * (1.0 / fp8_max)).clamp(min=min_sf)
    scale[:, 0] = s
    y = x * (1.0 / s[:, None])
    result.copy_(y.clamp(fp8_min, fp8_max).to(result.dtype))


def _qtraits(result_dtype):
    if result_dtype == torch.int8:
        return -128.0, 127.0, torch.finfo(torch.float32).eps, True
    fi = torch.finfo(torch.float8_e4m3fn)
    return fi.min, fi.max, 1.0 / (fi.max * 512.0), False


def ref_rms_norm_dynamic(result, input, weight, scale, epsilon,
                         scale_ub=None, residual=None):
    qmin, qmax, min_sf, is_int8 = _qtraits(result.dtype)
    x = input.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    xn = (x * rms).to(input.dtype) * weight           # bf16 * bf16 (matches kernel)
    xn_f = xn.to(torch.float32)
    s = xn_f.abs().amax(dim=-1)
    if scale_ub is not None:
        s = s.clamp(max=scale_ub.item())
    s = (s * (1.0 / qmax)).clamp(min=min_sf)
    scale[:, 0] = s
    if residual is not None:
        residual.copy_(x.to(residual.dtype))
    y = xn_f / s[:, None]
    if is_int8:
        y = y.round()
    result.copy_(y.clamp(qmin, qmax).to(result.dtype))


def ref_rms_norm_per_block(result, input, weight, scale, epsilon, scale_ub,
                           residual, group_size, is_scale_transposed):
    qmin, qmax, min_sf, is_int8 = _qtraits(result.dtype)
    T, H = input.shape
    G = H // group_size
    x = input.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)   # full row
    xn = (x * rms).to(input.dtype) * weight           # bf16
    xn_f = xn.to(torch.float32).view(T, G, group_size)
    s = xn_f.abs().amax(dim=-1)                        # [T, G]
    if scale_ub is not None:
        s = s.clamp(max=scale_ub.item())
    s = (s * (1.0 / qmax)).clamp(min=min_sf)
    scale.copy_(s)
    if residual is not None:
        residual.copy_(x.to(residual.dtype))
    y = xn_f / s[:, :, None]
    if is_int8:
        y = y.round()
    result.copy_(y.clamp(qmin, qmax).reshape(T, H).to(result.dtype))


def ref_per_token_group(input, output_q, output_s, group_size, eps, fp8_min,
                        fp8_max, scale_ue8m0, *dummies):
    T, H = input.shape
    G = H // group_size
    x = input.view(T, G, group_size).to(torch.float32)
    s = x.abs().amax(dim=-1).clamp(min=eps) / fp8_max
    if scale_ue8m0:
        s = torch.exp2(torch.ceil(torch.log2(s)))
    output_s.copy_(s)
    q = (x / s[:, :, None]).clamp(fp8_min, fp8_max).to(output_q.dtype)
    output_q.view(T, G, group_size).copy_(q)
