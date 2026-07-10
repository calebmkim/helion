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
        # tensor-broadcast clamp (0-d scale_ub), NOT scale_ub.item(): .item() forces a GPU->CPU
        # sync that graph-breaks torch.compile mid-kernel. This form is bit-identical and fuses.
        s = torch.clamp(s, max=scale_ub)
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
        s = torch.clamp(s, max=scale_ub)  # tensor broadcast, not .item() (see ref_dynamic_per_token)
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
        s = torch.clamp(s, max=scale_ub)  # tensor broadcast, not .item() (see ref_dynamic_per_token)
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


def ref_fused_qk_norm_rope(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
                           q_weight, k_weight, cos_sin_cache, is_neox, position_ids,
                           forced_token_heads_per_warp=-1):
    """Mirror the kernel's math (mutates qkv in place): per-head RMS-norm (fp32 accumulate) on the
    q and k heads with q_weight/k_weight, then RoPE on the rotary_dim, leaving v untouched.
    Matches examples/kut/fused_qk_norm_rope.py exactly (weight applied in input dtype after cast-back).

    Written in a FUSION-FRIENDLY form for torch.compile: the neox path (rotary_dim == head_dim,
    which is every benchmarked shape) uses the HF `rotate_half` identity with full-width cos/sin and
    a SINGLE contiguous slice-write back into qkv — no advanced-index `index_put`/scatter. That lets
    Inductor lower it to 2 kernels (RMS reduction + one pointwise epilogue) instead of the 6 kernels
    the earlier advanced-indexing form produced (index_put blocks reduction↔pointwise fusion). This
    is the same "make our torch reference fuse well so the tc baseline is fair" principle applied to
    the vLLM `.item()` graph-break fix. Verified bit-identical to the advanced-index form across all
    benchmarked (q_heads, kv_heads, num_tokens) shapes. The general (non-neox / partial-rotary) path
    keeps the exact advanced-index form for correctness."""
    T = qkv.shape[0]
    qk_heads = num_heads_q + num_heads_k
    rotary_dim = cos_sin_cache.shape[1]
    embed = rotary_dim // 2
    dt = qkv.dtype
    x = qkv.view(T, -1, head_dim)  # view into qkv storage (in-place)

    # --- per-head RMS-norm on q|k heads (fp32), weight in input dtype (matches kernel) ---
    xf = x[:, :qk_heads, :].to(torch.float32)
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    w = torch.where(
        (torch.arange(qk_heads, device=qkv.device) < num_heads_q)[None, :, None],
        q_weight[None, None, :], k_weight[None, None, :],
    )
    xn = (xf * rms).to(dt) * w  # [T, qk_heads, head_dim]; cast back to dt BEFORE weight, as the kernel does

    cos = cos_sin_cache[position_ids, :embed]              # [T, embed]
    sin = cos_sin_cache[position_ids, embed:2 * embed]
    if is_neox and rotary_dim == head_dim:
        # neox rotate_half: o = xn*cos_full + rotate_half(xn)*sin_full, one fused slice-write.
        cos_f = torch.cat((cos, cos), dim=-1)[:, None, :]  # [T, 1, head_dim]
        sin_f = torch.cat((sin, sin), dim=-1)[:, None, :]
        xr1, xr2 = xn[..., :embed], xn[..., embed:]
        rot_half = torch.cat((-xr2, xr1), dim=-1)
        x[:, :qk_heads, :] = xn * cos_f + rot_half * sin_f
    else:
        # general path (partial rotary or GPT-J interleave): advanced-index form.
        x[:, :qk_heads, :] = xn
        if is_neox:
            i1 = torch.arange(embed, device=qkv.device)
            i2 = i1 + embed
        else:
            i1 = torch.arange(embed, device=qkv.device) * 2
            i2 = i1 + 1
        blk = x[:, :qk_heads, :]
        x1 = blk[:, :, i1]
        x2 = blk[:, :, i2]
        o1 = x1 * cos[:, None, :] - x2 * sin[:, None, :]
        o2 = x2 * cos[:, None, :] + x1 * sin[:, None, :]
        blk[:, :, i1] = o1
        blk[:, :, i2] = o2
        x[:, :qk_heads, :] = blk
