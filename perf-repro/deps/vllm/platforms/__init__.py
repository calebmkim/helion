import torch


class _CurrentPlatform:
    """Minimal stand-in for vllm.platforms.current_platform (CUDA, non-fnuz)."""

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        return torch.float8_e4m3fn

    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        return False

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)


current_platform = _CurrentPlatform()
