from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_compat import libdevice
from helion.runtime import default_launcher as _default_launcher

@triton.jit
def _helion_rms_norm(x, weight, out, out_stride_0, out_stride_1, weight_stride_0, x_stride_0, x_stride_1, n, eps, _RDIM_SIZE_1: tl.constexpr):
    # src[rms_norm.py:16]: for tile_m in hl.tile(m):
    pid_0 = tl.program_id(0)
    offset_0 = pid_0
    indices_0 = offset_0 + tl.zeros([1], tl.int32)
    indices_1 = tl.arange(0, _RDIM_SIZE_1).to(tl.int32)
    mask_1 = indices_1 < n
    # src[rms_norm.py:17]: acc = x[tile_m, :].to(torch.float32)
    load = tl.load(x + (indices_0[:, None] * x_stride_0 + (0 + indices_1)[None, :] * x_stride_1), mask_1[None, :], other=0)
    v_0 = tl.cast(load, tl.float32)
    # src[rms_norm.py:18]: variance = torch.mean(acc * acc, dim=-1)
    v_1 = v_0 * v_0
    variance_extra = tl.cast(tl.sum(v_1, 1), tl.float32)
    v_2 = tl.cast(n, tl.float32)
    v_3 = variance_extra / v_2
    # src[rms_norm.py:19]: inv_rms = torch.rsqrt(variance + eps)
    v_4 = v_3 + eps
    v_5 = libdevice.rsqrt(v_4)
    # src[rms_norm.py:20]: out[tile_m, :] = (acc * inv_rms[:, None] * weight[:].to(torch.float32)).to(
    subscript = v_5[:, None]
    v_6 = v_0 * subscript
    load_1 = tl.load(weight + (0 + indices_1) * weight_stride_0, mask_1, other=0)
    v_7 = tl.cast(load_1, tl.float32)
    v_8 = v_7[None, :]
    v_9 = v_6 * v_8
    # src[rms_norm.py:20]: out[tile_m, :] = (acc * inv_rms[:, None] * weight[:].to(torch.float32)).to(
    # src[rms_norm.py:21]:     x.dtype
    # src[rms_norm.py:22]: )
    v_10 = tl.cast(v_9, tl.bfloat16)
    tl.store(out + (indices_0[:, None] * out_stride_0 + (0 + indices_1)[None, :] * out_stride_1), v_10, mask_1[None, :])

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float=1e-05, *, _launcher=_default_launcher):
    # src[rms_norm.py:14]: m, n = x.size()
    m, n = x.size()
    # src[rms_norm.py:15]: out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    # src[rms_norm.py:16]: for tile_m in hl.tile(m):
    _RDIM_SIZE_1 = triton.next_power_of_2(n)
    # src[rms_norm.py:16]: for tile_m in hl.tile(m):
    # src[rms_norm.py:17]:     acc = x[tile_m, :].to(torch.float32)
    # src[rms_norm.py:18]:     variance = torch.mean(acc * acc, dim=-1)
    # src[rms_norm.py:16-22]: ...
    _launcher(_helion_rms_norm, (m,), x, weight, out, out.stride(0), out.stride(1), weight.stride(0), x.stride(0), x.stride(1), n, eps, _RDIM_SIZE_1, num_warps=4, num_stages=1)
    # src[rms_norm.py:23]: return out
    return out