"""Shim for vllm.kernels.helion.register.register_kernel.

Faithfully reproduces vLLM's create_helion_decorated_kernel: it forces
static_shapes=False and applies any helion_settings, but DROPS the
config-picker / preset-config / custom-op machinery so the decorated symbol is
just a plain runnable helion.kernel (default/seed config under
HELION_AUTOTUNE_EFFORT=none). Extra kwargs (config_picker, input_generator,
fake_impl, mutates_args, op_name, ...) are accepted and ignored.
"""

from __future__ import annotations

from typing import Any

import helion


def register_kernel(*_args: Any, helion_settings: Any = None, **_kwargs: Any):
    def deco(raw_kernel_func):
        kernel_kwargs: dict[str, Any] = {}
        if helion_settings is not None:
            try:
                kernel_kwargs.update(helion_settings.to_dict())
            except Exception:
                pass
        # vLLM forces dynamic shapes for variable batch/seq lengths.
        kernel_kwargs["static_shapes"] = False
        return helion.kernel(**kernel_kwargs)(raw_kernel_func)

    return deco
