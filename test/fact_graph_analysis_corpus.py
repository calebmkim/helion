"""Focused bind-only corpus for graph-derived matmul and reduction facts."""

from __future__ import annotations

from dataclasses import dataclass
import enum
import importlib.util
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING
from typing import Any

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "test" / "data" / "fact_graph_analysis_corpus_sm100.json"


@dataclass(frozen=True)
class FactCorpusCase:
    name: str
    family: str
    source: str
    shape: str
    expects_reduction: bool
    build: Callable[[str], tuple[object, tuple[object, ...]]]


def _load_module(relative_path: str) -> Any:
    path = REPO_ROOT / relative_path
    module_name = f"_helion_fact_corpus_{path.parent.name}_{path.stem}"
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _rms_norm_fwd(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("examples/rms_norm.py")
    x = torch.randn(128, 256, device=device, dtype=torch.bfloat16)
    weight = torch.randn(256, device=device, dtype=torch.bfloat16)
    return module.rms_norm_fwd, (x, weight, 1e-5)


def _layer_norm_fwd(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("examples/layer_norm.py")
    x = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
    weight = torch.randn(64, device=device, dtype=torch.bfloat16)
    bias = torch.randn(64, device=device, dtype=torch.bfloat16)
    return module.layer_norm_fwd, (x, [64], weight, bias, 1e-5)


def _cross_entropy(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("examples/cross_entropy.py")
    logits = torch.randn(128, 1000, device=device, dtype=torch.float32)
    labels = torch.randint(0, 1000, (128,), device=device, dtype=torch.int64)
    return module.cross_entropy, (logits, labels)


def _jsd(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("examples/jsd.py")
    student = torch.randn(32, 256, device=device).log_softmax(dim=-1)
    teacher = torch.randn(32, 256, device=device).log_softmax(dim=-1)
    return module.jsd_forward, (student, teacher, None, 0.5, -100)


def _kl_div(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("examples/kl_div.py")
    prediction = torch.randn(32, 256, device=device).log_softmax(dim=-1)
    target = torch.randn(32, 256, device=device).softmax(dim=-1)
    return module.kl_div_forward, (
        prediction,
        target,
        False,
        "batchmean",
        1e-10,
    )


def _rms_norm_bwd(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("examples/rms_norm.py")
    grad_out = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
    x = torch.randn_like(grad_out)
    weight = torch.randn(64, device=device, dtype=torch.bfloat16)
    rsqrt = torch.rand(32, 1, device=device, dtype=torch.float32)
    return module.rms_norm_bwd, (grad_out, x, weight, rsqrt)


def _layer_norm_bwd(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("examples/layer_norm.py")
    grad_out = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
    x = torch.randn_like(grad_out)
    mean = torch.randn(32, device=device, dtype=torch.float32)
    rstd = torch.rand(32, device=device, dtype=torch.float32)
    weight = torch.randn(64, device=device, dtype=torch.bfloat16)
    return module.layer_norm_bwd, (grad_out, x, mean, rstd, weight, True)


def _sum(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("examples/sum.py")
    x = torch.randn(512, 512, device=device, dtype=torch.float32)
    return module.sum_kernel, (x,)


def _grpo_loss(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("examples/grpo_loss.py")
    batch, sequence, vocab = 2, 16, 128
    logits = torch.randn(
        batch,
        sequence + 1,
        vocab,
        device=device,
        dtype=torch.bfloat16,
    )
    selected_logits = torch.randn(batch, sequence, device=device)
    old_logp = torch.randn(batch, sequence, device=device)
    ref_logp = torch.randn(batch, sequence, device=device)
    advantages = torch.randn(batch, device=device)
    completion_mask = torch.ones(batch, sequence, device=device)
    return module.grpo_loss_forward, (
        logits,
        selected_logits,
        old_logp,
        ref_logp,
        advantages,
        completion_mask,
        0.9,
        0.04,
        0.2,
        0.4,
    )


def _vllm_silu_mul_fp8(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("pretuned_kernels/silu_mul_fp8/silu_mul_fp8.py")
    x = torch.randn(8, 4096, device=device, dtype=torch.bfloat16)
    scale = torch.ones(1, device=device, dtype=torch.float32)
    return module.silu_mul_fp8, (x, scale)


def _vllm_dynamic_quant(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module(
        "pretuned_kernels/dynamic_per_token_scaled_fp8_quant/"
        "dynamic_per_token_scaled_fp8_quant.py"
    )
    x = torch.randn(8, 2048, device=device, dtype=torch.bfloat16)
    result = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty(8, 1, device=device, dtype=torch.float32)
    return module.dynamic_per_token_scaled_fp8_quant, (result, x, scale, None)


def _vllm_rms_norm_per_block(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module(
        "pretuned_kernels/rms_norm_per_block_quant/rms_norm_per_block_quant.py"
    )
    assert device == "cuda"
    return module.rms_norm_per_block_quant, module._make_inputs(8, 2048, 128)


def _vllm_fused_qk_norm_rope(device: str) -> tuple[object, tuple[object, ...]]:
    module = _load_module("pretuned_kernels/fused_qk_norm_rope/fused_qk_norm_rope.py")
    assert device == "cuda"
    return module.fused_qk_norm_rope, module._make_inputs(8, 16, 8)


CASES = (
    FactCorpusCase(
        "rms_norm_fwd",
        "rms_norm",
        "examples/rms_norm.py",
        "128x256",
        True,
        _rms_norm_fwd,
    ),
    FactCorpusCase(
        "layer_norm_fwd",
        "layer_norm",
        "examples/layer_norm.py",
        "32x64",
        True,
        _layer_norm_fwd,
    ),
    FactCorpusCase(
        "cross_entropy",
        "cross_entropy",
        "examples/cross_entropy.py",
        "128x1000",
        True,
        _cross_entropy,
    ),
    FactCorpusCase(
        "jsd_forward",
        "jsd",
        "examples/jsd.py",
        "32x256",
        True,
        _jsd,
    ),
    FactCorpusCase(
        "kl_div_forward",
        "kl_div",
        "examples/kl_div.py",
        "32x256",
        True,
        _kl_div,
    ),
    FactCorpusCase(
        "rms_norm_bwd",
        "rms_norm_bwd",
        "examples/rms_norm.py",
        "32x64",
        True,
        _rms_norm_bwd,
    ),
    FactCorpusCase(
        "layer_norm_bwd",
        "layer_norm_bwd",
        "examples/layer_norm.py",
        "32x64",
        True,
        _layer_norm_bwd,
    ),
    FactCorpusCase(
        "sum",
        "sum",
        "examples/sum.py",
        "512x512",
        True,
        _sum,
    ),
    FactCorpusCase(
        "grpo_loss_forward",
        "grpo_loss",
        "examples/grpo_loss.py",
        "2x16x128",
        True,
        _grpo_loss,
    ),
    FactCorpusCase(
        "vllm_silu_mul_fp8",
        "vllm",
        "pretuned_kernels/silu_mul_fp8/silu_mul_fp8.py",
        "8x2048",
        False,
        _vllm_silu_mul_fp8,
    ),
    FactCorpusCase(
        "vllm_dynamic_per_token_scaled_fp8_quant",
        "vllm",
        "pretuned_kernels/dynamic_per_token_scaled_fp8_quant/"
        "dynamic_per_token_scaled_fp8_quant.py",
        "8x2048",
        True,
        _vllm_dynamic_quant,
    ),
    FactCorpusCase(
        "vllm_rms_norm_per_block_quant",
        "vllm",
        "pretuned_kernels/rms_norm_per_block_quant/rms_norm_per_block_quant.py",
        "8x2048, group=128",
        True,
        _vllm_rms_norm_per_block,
    ),
    FactCorpusCase(
        "vllm_fused_qk_norm_rope",
        "vllm",
        "pretuned_kernels/fused_qk_norm_rope/fused_qk_norm_rope.py",
        "tokens=8, q_heads=16, kv_heads=8",
        True,
        _vllm_fused_qk_norm_rope,
    ),
)


def normalize(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return {
            "__enum__": f"{type(value).__name__}.{value.name}",
            "value": normalize(value.value),
        }
    fields = getattr(value, "_fields", None)
    if isinstance(fields, tuple):
        return {
            "__type__": type(value).__name__,
            **{name: normalize(getattr(value, name)) for name in fields},
        }
    if isinstance(value, dict):
        return {
            str(key): normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return {"__type__": type(value).__name__, "__repr__": repr(value)}


def canonicalize_unordered_live_sets(
    value: object,
    parent: str = "",
) -> object:
    """Sort only liveness collections whose source is a mathematical set."""
    if isinstance(value, dict):
        return {
            key: canonicalize_unordered_live_sets(item, key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        items = [canonicalize_unordered_live_sets(item, parent) for item in value]

        def key(item: object) -> str:
            return json.dumps(item, sort_keys=True, separators=(",", ":"))

        if parent in {"live_tiles", "live_dot_outputs"}:
            return sorted(items, key=key)
        if parent == "live_tile_steps":
            return [sorted(step, key=key) for step in items]
        return items
    return value


def _matmul_projection(spec: object) -> object:
    fact = getattr(spec, "kernel_matmul_fact", None)
    if fact is not None:
        resolved = fact.matmuls
        return canonicalize_unordered_live_sets(
            {
                "matmuls": normalize(tuple(item.fact for item in resolved)),
                "axes": normalize(tuple(item.axes for item in resolved)),
                "sites": normalize(tuple(item.site for item in resolved)),
                **{
                    field: normalize(getattr(fact, field))
                    for field in fact._fields
                    if field != "matmuls"
                },
            }
        )
    fact = getattr(spec, "multi_matmul_fact", None)
    if fact is None:
        return None
    return canonicalize_unordered_live_sets(
        {field: normalize(getattr(fact, field)) for field in fact._fields}
    )


def _heuristic_eligibility(bound: object) -> list[dict[str, object]]:
    from helion._compiler.autotuner_heuristics import get_heuristics

    result: list[dict[str, object]] = []
    env = bound.env
    device_ir = bound.host_function.device_ir
    with env:
        for heuristic in get_heuristics(env.backend_name):
            record: dict[str, object] = {"name": heuristic.name}
            try:
                record["eligible"] = bool(heuristic.is_eligible(env, device_ir))
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            result.append(record)
    return result


def capture_corpus(device: str = "cuda") -> dict[str, object]:
    records: dict[str, object] = {}
    for case in CASES:
        torch.manual_seed(0)
        kernel, args = case.build(device)
        reset = getattr(kernel, "reset", None)
        if callable(reset):
            reset()
        bound = kernel.bind(args)
        spec = bound.env.config_spec
        reduction_fact = spec.reduction_kernel_fact
        contributing = list(spec.autotuner_heuristics)
        reduction_contributors = [
            name for name in contributing if name.startswith("triton_reduction_")
        ]
        reduction_fact_snapshot = canonicalize_unordered_live_sets(
            normalize(reduction_fact)
        )
        matmul_fact_snapshot = _matmul_projection(spec)
        local_matmul_facts = normalize(spec.matmul_facts)
        compiler_seeds = [
            normalize(dict(config.config)) for config in spec.compiler_seed_configs
        ]
        promoted_default = (
            normalize(dict(spec.compiler_default_config.config))
            if spec.compiler_default_config is not None
            else None
        )
        default = normalize(dict(spec.default_config().config))
        heuristic_eligibility = _heuristic_eligibility(bound)
        records[case.name] = {
            "family": case.family,
            "source": case.source,
            "shape": case.shape,
            "expects_reduction": case.expects_reduction,
            "has_reduction": bool(
                reduction_fact is not None and reduction_fact.reductions
            ),
            "reduction_heuristics": reduction_contributors,
            "contributing_heuristics": contributing,
            "heuristic_eligibility": heuristic_eligibility,
            "reduction_fact": reduction_fact_snapshot,
            "matmul_fact": matmul_fact_snapshot,
            "local_matmul_facts": local_matmul_facts,
            "compiler_seeds": compiler_seeds,
            "promoted_default": promoted_default,
            "default": default,
        }
        del args, bound
        torch.cuda.empty_cache()
    return records
