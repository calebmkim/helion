import torch

from vllm.platforms import current_platform


def get_fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()


def get_int8_min_max() -> tuple[int, int]:
    qtype_traits = torch.iinfo(torch.int8)
    return qtype_traits.min, qtype_traits.max


def get_int8_min_scaling_factor() -> float:
    return torch.finfo(torch.float32).eps
