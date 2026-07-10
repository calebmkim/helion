import torch

from vllm.platforms import current_platform


def get_fp8_min_max() -> tuple[float, float]:
    """Min/max for FP8 quantization (CUDA e4m3 path)."""
    if current_platform.is_fp8_fnuz():
        return -224.0, 224.0
    finfo = torch.finfo(current_platform.fp8_dtype())
    return finfo.min, finfo.max
