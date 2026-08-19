from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any
from typing import Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
DEVICE = "cuda:0"
EPS = 1e-5
FP8 = torch.float8_e4m3fn


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    exact: bool = False


@dataclass
class WorkloadState:
    args: tuple[Any, ...]
    restore_sources: dict[int, torch.Tensor]

    def restore(self) -> None:
        for index, source in self.restore_sources.items():
            self.args[index].copy_(source)


@dataclass
class Workload:
    cohort: str
    kernel: str
    body: str
    shape: tuple[int, ...]
    dtype: str
    kernel_fn: Any
    reference_fn: Callable[..., Any]
    base_args: tuple[Any, ...]
    mutable_indices: tuple[int, ...]
    restore_indices: tuple[int, ...]
    observe: Callable[[Any, tuple[Any, ...]], dict[str, torch.Tensor]]
    tolerances: dict[str, Tolerance]
    optional_args: dict[str, Any]
    aot_key: tuple[int, ...] | None = None

    def make_state(self) -> WorkloadState:
        args: list[Any] = []
        mutable = set(self.mutable_indices)
        for index, arg in enumerate(self.base_args):
            if index in mutable and torch.is_tensor(arg):
                args.append(arg.clone())
            else:
                args.append(arg)
        restore_sources = {
            index: self.base_args[index]
            for index in self.restore_indices
            if torch.is_tensor(self.base_args[index])
        }
        return WorkloadState(tuple(args), restore_sources)


@dataclass(frozen=True)
class KernelSpec:
    name: str
    cohort: str
    shapes: tuple[tuple[int, ...], ...]
    build: Callable[[tuple[int, ...]], Workload]
    has_aot: bool


_MODULES: dict[str, Any] = {}
_MISSING = object()


def _testing_stub() -> ModuleType:
    module = ModuleType("helion._testing")
    module.DEVICE = torch.device(DEVICE)
    module.HALF_DTYPE = torch.float16

    def run_example(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("run_example is unavailable in the audit import shim")

    module.run_example = run_example
    return module


def _load_module(name: str, relative_path: str) -> Any:
    cached = _MODULES.get(name)
    if cached is not None:
        return cached
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"_h100_audit_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    previous_testing: Any = _MISSING
    if relative_path.startswith("examples/"):
        previous_testing = sys.modules.get("helion._testing", _MISSING)
        sys.modules["helion._testing"] = _testing_stub()
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_testing is _MISSING:
            sys.modules.pop("helion._testing", None)
        else:
            sys.modules["helion._testing"] = previous_testing
    _MODULES[name] = module
    return module


def _returned(names: tuple[str, ...]) -> Callable:
    def observe(output: Any, args: tuple[Any, ...]) -> dict[str, torch.Tensor]:
        values = output if isinstance(output, (tuple, list)) else (output,)
        return dict(zip(names, values, strict=True))

    return observe


def _mutated(mapping: dict[str, int]) -> Callable:
    def observe(output: Any, args: tuple[Any, ...]) -> dict[str, torch.Tensor]:
        return {name: args[index] for name, index in mapping.items()}

    return observe


def _tensor_dtypes(args: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "arg": index,
            "dtype": str(arg.dtype).removeprefix("torch."),
            "shape": list(arg.shape),
        }
        for index, arg in enumerate(args)
        if torch.is_tensor(arg)
    ]


def describe_inputs(workload: Workload) -> list[dict[str, Any]]:
    return _tensor_dtypes(workload.base_args)


def _build_rms_norm(shape: tuple[int, ...]) -> Workload:
    module = _load_module("rms_norm", "pretuned_kernels/rms_norm/rms_norm.py")
    m, n = shape
    x = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
    return Workload(
        "general_aot",
        "rms_norm",
        "pretuned_kernels/rms_norm/rms_norm.py:rms_norm",
        shape,
        "bf16",
        module.rms_norm,
        module._rms_norm_torch,
        (x, weight),
        (),
        (),
        _returned(("output",)),
        {"output": Tolerance(1e-2, 1e-2)},
        {"eps": 1e-5},
    )


def _build_layer_norm(shape: tuple[int, ...]) -> Workload:
    module = _load_module(
        "layer_norm", "pretuned_kernels/layer_norm/layer_norm.py"
    )
    m, n = shape
    x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
    weight = torch.randn(n, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(n, device=DEVICE, dtype=torch.float16)
    return Workload(
        "general_aot",
        "layer_norm",
        "pretuned_kernels/layer_norm/layer_norm.py:layer_norm",
        shape,
        "fp16",
        module.layer_norm,
        module._layer_norm_torch,
        (x, weight, bias),
        (),
        (),
        _returned(("output",)),
        {"output": Tolerance(1e-2, 1e-2)},
        {"eps": 1e-5},
    )


def _build_softmax(shape: tuple[int, ...]) -> Workload:
    module = _load_module("softmax", "pretuned_kernels/softmax/softmax.py")
    m, n = shape
    x = torch.randn(m, n, device=DEVICE, dtype=torch.float16)
    return Workload(
        "general_aot",
        "softmax",
        "pretuned_kernels/softmax/softmax.py:softmax",
        shape,
        "fp16",
        module.softmax,
        module._softmax_torch,
        (x,),
        (),
        (),
        _returned(("output",)),
        {"output": Tolerance(1e-3, 1e-3)},
        {},
    )


def _build_cross_entropy(shape: tuple[int, ...]) -> Workload:
    module = _load_module(
        "cross_entropy",
        "pretuned_kernels/cross_entropy/cross_entropy.py",
    )
    tokens, vocab = shape
    logits = torch.randn(tokens, vocab, device=DEVICE, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab, (tokens,), device=DEVICE, dtype=torch.int64)
    return Workload(
        "general_aot",
        "cross_entropy",
        "pretuned_kernels/cross_entropy/cross_entropy.py:cross_entropy",
        shape,
        "bf16",
        module.cross_entropy,
        module._cross_entropy_torch,
        (logits, labels),
        (),
        (),
        _returned(("loss",)),
        {"loss": Tolerance(1e-2, 1e-2)},
        {},
    )


def _kl_reference(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    log_target: bool,
    reduction: str,
    eps: float,
) -> torch.Tensor:
    return F.kl_div(y_pred, y_true, reduction=reduction, log_target=log_target)


def _build_kl_div(shape: tuple[int, ...]) -> Workload:
    module = _load_module("kl_div", "examples/kl_div.py")
    m, n = shape
    y_pred = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16).log_softmax(-1)
    y_true = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16).softmax(-1)
    return Workload(
        "original",
        "kl_div",
        "examples/kl_div.py:kl_div_forward",
        shape,
        "bf16",
        module.kl_div_forward,
        _kl_reference,
        (y_pred, y_true, False, "batchmean", 1e-10),
        (),
        (),
        _returned(("loss",)),
        {"loss": Tolerance(2e-2, 2e-2)},
        {"log_target": False, "reduction": "batchmean", "eps": 1e-10},
    )


def _jsd_reference(
    log_q: torch.Tensor,
    log_p: torch.Tensor,
    shift_labels: torch.Tensor | None,
    beta: float,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.exp(log_q.to(torch.float32))
    p = torch.exp(log_p.to(torch.float32))
    mixture = beta * p + (1.0 - beta) * q
    log_mixture = torch.log(mixture)
    per_value_loss = beta * p * (log_p - log_mixture) + (1.0 - beta) * q * (
        log_q - log_mixture
    )
    per_value_dx = (1.0 - beta) * q * (log_q - log_mixture)
    scale = 1.0 / log_q.shape[0]
    per_row_loss = torch.sum(per_value_loss * scale, dim=-1)
    per_row_dx = torch.sum(per_value_dx * scale, dim=-1)
    return torch.sum(per_row_loss), per_row_dx


def _build_jsd(shape: tuple[int, ...]) -> Workload:
    module = _load_module("jsd", "examples/jsd.py")
    m, n = shape
    log_q = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16).log_softmax(-1)
    log_p = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16).log_softmax(-1)
    return Workload(
        "original",
        "jsd",
        "examples/jsd.py:jsd_forward",
        shape,
        "bf16",
        module.jsd_forward,
        _jsd_reference,
        (log_q, log_p, None, 0.5, -100),
        (),
        (),
        _returned(("loss", "dX")),
        {
            "loss": Tolerance(3e-2, 3e-2),
            "dX": Tolerance(3e-2, 3e-2),
        },
        {"shift_labels": None, "beta": 0.5, "ignore_index": -100},
    )


def _fused_linear_jsd_reference(
    beta: float,
    ignore_index: int,
    temperature: float,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    student_scaled = student_logits.to(torch.float32) / temperature
    teacher_scaled = teacher_logits.to(torch.float32) / temperature
    student_prob = torch.softmax(student_scaled, dim=-1)
    teacher_prob = torch.softmax(teacher_scaled, dim=-1)
    student_log_prob = torch.log_softmax(student_scaled, dim=-1)
    teacher_log_prob = torch.log_softmax(teacher_scaled, dim=-1)
    mixture = (1.0 - beta) * student_prob + beta * teacher_prob
    log_mixture = torch.log(mixture)
    student_kl = torch.sum(student_prob * (student_log_prob - log_mixture), dim=-1)
    teacher_kl = torch.sum(teacher_prob * (teacher_log_prob - log_mixture), dim=-1)
    loss = (1.0 - beta) * student_kl + beta * teacher_kl
    grad = ((1.0 - beta) / temperature) * (student_prob - mixture)
    return loss, grad


def _build_fused_linear_jsd(shape: tuple[int, ...]) -> Workload:
    module = _load_module("fused_linear_jsd", "examples/fused_linear_jsd.py")
    m, vocab = shape
    student = torch.randn(m, vocab, device=DEVICE, dtype=torch.bfloat16)
    teacher = torch.randn(m, vocab, device=DEVICE, dtype=torch.bfloat16)
    return Workload(
        "original",
        "fused_linear_jsd",
        "examples/fused_linear_jsd.py:jsd_kernel",
        shape,
        "bf16",
        module.jsd_kernel,
        _fused_linear_jsd_reference,
        (0.5, -100, 1.0, student, teacher),
        (),
        (),
        _returned(("loss", "grad_student_logits")),
        {
            "loss": Tolerance(3e-2, 3e-2),
            "grad_student_logits": Tolerance(3e-2, 3e-2),
        },
        {"beta": 0.5, "ignore_index": -100, "temperature": 1.0},
    )


def _grpo_reference(
    logits: torch.Tensor,
    selected_logits: torch.Tensor,
    old_logp: torch.Tensor | None,
    ref_logp: torch.Tensor | None,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor | None,
    temperature: float,
    beta: float,
    eps_low: float,
    eps_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scaled = logits[:, :-1, :].to(torch.float32) / temperature
    lse = torch.logsumexp(scaled, dim=-1)
    logp = selected_logits - lse
    old = logp if old_logp is None else old_logp
    coef_1 = torch.exp(logp - old)
    coef_2 = torch.clamp(coef_1, 1.0 - eps_low, 1.0 + eps_high)
    loss_1 = coef_1 * advantages[:, None]
    loss_2 = coef_2 * advantages[:, None]
    loss = -torch.minimum(loss_1, loss_2)
    if completion_mask is not None:
        loss = loss * completion_mask
    kl = torch.zeros_like(loss)
    if beta != 0.0 and ref_logp is not None:
        kl = torch.exp(ref_logp - logp) - (ref_logp - logp) - 1.0
        if completion_mask is not None:
            kl = kl * completion_mask
        loss = loss + beta * kl
    is_clipped = (loss_1 < loss_2).to(torch.float32)
    return loss, kl, is_clipped, lse


def _build_grpo(shape: tuple[int, ...]) -> Workload:
    module = _load_module("grpo_loss", "examples/grpo_loss.py")
    batch, sequence, vocab = shape
    temperature, beta, eps_low, eps_high = 0.9, 0.2, 0.2, 0.4
    logits = torch.randn(
        batch,
        sequence + 1,
        vocab,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    completion_ids = torch.randint(
        0,
        vocab,
        (batch, sequence),
        device=DEVICE,
        dtype=torch.int64,
    )
    selected = module.extract_selected_logits_pytorch(
        logits[:, :-1, :], completion_ids, temperature
    )
    old_logp = torch.randn(
        batch, sequence, device=DEVICE, dtype=torch.float32
    )
    ref_logp = torch.randn(
        batch, sequence, device=DEVICE, dtype=torch.float32
    )
    advantages = torch.randn(batch, device=DEVICE, dtype=torch.float32)
    completion_mask = torch.ones(
        batch, sequence, device=DEVICE, dtype=torch.float32
    )
    args = (
        logits,
        selected,
        old_logp,
        ref_logp,
        advantages,
        completion_mask,
        temperature,
        beta,
        eps_low,
        eps_high,
    )
    return Workload(
        "original",
        "grpo",
        "examples/grpo_loss.py:grpo_loss_forward",
        shape,
        "bf16",
        module.grpo_loss_forward,
        _grpo_reference,
        args,
        (),
        (),
        _returned(("loss", "kl_loss", "is_clipped", "lse")),
        {
            "loss": Tolerance(3e-2, 3e-2),
            "kl_loss": Tolerance(3e-2, 3e-2),
            "is_clipped": Tolerance(0.0, 0.0, exact=True),
            "lse": Tolerance(3e-2, 3e-2),
        },
        {
            "temperature": temperature,
            "beta": beta,
            "eps_low": eps_low,
            "eps_high": eps_high,
            "old_logp": True,
            "ref_logp": True,
            "completion_mask": True,
        },
    )


def _rms_norm_bwd_reference(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rsqrt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_f = x.to(torch.float32)
    grad_f = grad_out.to(torch.float32)
    rsqrt_f = rsqrt.to(torch.float32)
    weight_f = weight.to(torch.float32)[None, :]
    grad_weight = torch.sum(x_f * grad_f * rsqrt_f, dim=0)
    grad_x = weight_f * grad_f * rsqrt_f - x_f * rsqrt_f**3 * torch.mean(
        weight_f * grad_f * x_f, dim=-1, keepdim=True
    )
    return grad_x.to(x.dtype), grad_weight.to(weight.dtype)


def _build_rms_norm_bwd(shape: tuple[int, ...]) -> Workload:
    module = _load_module("example_rms_norm", "examples/rms_norm.py")
    m, n = shape
    x = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
    grad_out = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
    rsqrt = torch.rsqrt(
        torch.mean(x.to(torch.float32) ** 2, dim=-1, keepdim=True) + EPS
    ).to(torch.bfloat16)
    return Workload(
        "original",
        "rms_norm_bwd",
        "examples/rms_norm.py:rms_norm_bwd",
        shape,
        "bf16",
        module.rms_norm_bwd,
        _rms_norm_bwd_reference,
        (grad_out, x, weight, rsqrt),
        (),
        (),
        _returned(("grad_x", "grad_weight")),
        {
            "grad_x": Tolerance(3e-2, 3e-2),
            "grad_weight": Tolerance(3e-2, 3e-2),
        },
        {"eps": EPS, "rsqrt_dtype": "bf16"},
    )


def _layer_norm_bwd_reference(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_f = x.to(torch.float32)
    grad_f = grad_out.to(torch.float32)
    weight_f = weight.to(torch.float32)[None, :]
    x_hat = (x_f - mean.to(torch.float32)[:, None]) * rstd.to(torch.float32)[
        :, None
    ]
    grad_weight = torch.sum(grad_f * x_hat, dim=0)
    grad_bias = torch.sum(grad_f, dim=0)
    weighted_grad = weight_f * grad_f
    c1 = torch.mean(x_hat * weighted_grad, dim=-1, keepdim=True)
    c2 = torch.mean(weighted_grad, dim=-1, keepdim=True)
    grad_x = (weighted_grad - (x_hat * c1 + c2)) * rstd.to(torch.float32)[
        :, None
    ]
    return (
        grad_x.to(x.dtype),
        grad_weight.to(weight.dtype),
        grad_bias.to(weight.dtype),
    )


def _build_layer_norm_bwd(shape: tuple[int, ...]) -> Workload:
    module = _load_module("example_layer_norm", "examples/layer_norm.py")
    m, n = shape
    x = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
    grad_out = torch.randn(m, n, device=DEVICE, dtype=torch.bfloat16)
    mean = torch.mean(x.to(torch.float32), dim=-1)
    rstd = torch.rsqrt(
        torch.var(x.to(torch.float32), dim=-1, unbiased=False) + EPS
    )
    return Workload(
        "original",
        "layer_norm_bwd",
        "examples/layer_norm.py:layer_norm_bwd",
        shape,
        "bf16",
        module.layer_norm_bwd,
        _layer_norm_bwd_reference,
        (grad_out, x, mean, rstd, weight),
        (),
        (),
        _returned(("grad_x", "grad_weight", "grad_bias")),
        {
            "grad_x": Tolerance(3e-2, 3e-2),
            "grad_weight": Tolerance(3e-2, 3e-2),
            "grad_bias": Tolerance(3e-2, 3e-2),
        },
        {"compute_bias_grad": True, "eps": EPS},
    )


def _build_dynamic_per_token(shape: tuple[int, ...]) -> Workload:
    module = _load_module(
        "dynamic_per_token_scaled_fp8_quant",
        "pretuned_kernels/dynamic_per_token_scaled_fp8_quant/"
        "dynamic_per_token_scaled_fp8_quant.py",
    )
    tokens, hidden = shape
    x = torch.randn(tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
    result = torch.empty_like(x, dtype=FP8)
    scale = torch.empty(tokens, 1, device=DEVICE, dtype=torch.float32)
    return Workload(
        "vllm",
        "dynamic_per_token_scaled_fp8_quant",
        "pretuned_kernels/dynamic_per_token_scaled_fp8_quant/"
        "dynamic_per_token_scaled_fp8_quant.py:dynamic_per_token_scaled_fp8_quant",
        shape,
        "bf16",
        module.dynamic_per_token_scaled_fp8_quant,
        module._dynamic_per_token_scaled_fp8_quant_torch,
        (result, x, scale),
        (0, 2),
        (),
        _mutated({"result": 0, "scale": 2}),
        {
            "result": Tolerance(0.2, 0.2),
            "scale": Tolerance(1e-2, 1e-4),
        },
        {"scale_ub": None, "output_dtype": "float8_e4m3fn"},
        (hidden, tokens),
    )


def _build_per_token_group(shape: tuple[int, ...]) -> Workload:
    module = _load_module(
        "per_token_group_fp8_quant",
        "pretuned_kernels/per_token_group_fp8_quant/"
        "per_token_group_fp8_quant.py",
    )
    tokens, hidden, group = shape
    x = torch.randn(tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
    output_q = torch.empty_like(x, dtype=FP8)
    output_s = torch.empty(
        tokens, hidden // group, device=DEVICE, dtype=torch.float32
    )
    args = (x, output_q, output_s, group, 1e-10, -448.0, 448.0, False)
    return Workload(
        "vllm",
        "per_token_group_fp8_quant",
        "pretuned_kernels/per_token_group_fp8_quant/"
        "per_token_group_fp8_quant.py:per_token_group_fp8_quant",
        shape,
        "bf16",
        module.per_token_group_fp8_quant,
        module._per_token_group_fp8_quant_torch,
        args,
        (1, 2),
        (),
        _mutated({"output_q": 1, "output_s": 2}),
        {
            "output_q": Tolerance(0.2, 0.2),
            "output_s": Tolerance(1e-2, 1e-6),
        },
        {
            "group_size": group,
            "eps": 1e-10,
            "fp8_min": -448.0,
            "fp8_max": 448.0,
            "scale_ue8m0": False,
            "output_dtype": "float8_e4m3fn",
        },
        (hidden, group, tokens),
    )


def _build_rms_norm_dynamic(shape: tuple[int, ...]) -> Workload:
    module = _load_module(
        "rms_norm_dynamic_per_token_quant",
        "pretuned_kernels/rms_norm_dynamic_per_token_quant/"
        "rms_norm_dynamic_per_token_quant.py",
    )
    tokens, hidden = shape
    x = torch.randn(tokens, hidden, device=DEVICE, dtype=torch.bfloat16)
    weight = torch.normal(
        1.0, 1.0, (hidden,), device=DEVICE, dtype=torch.bfloat16
    )
    result = torch.empty_like(x, dtype=FP8)
    scale = torch.empty(tokens, 1, device=DEVICE, dtype=torch.float32)
    args = (result, x, weight, scale, 1e-6)
    return Workload(
        "vllm",
        "rms_norm_dynamic_per_token_quant",
        "pretuned_kernels/rms_norm_dynamic_per_token_quant/"
        "rms_norm_dynamic_per_token_quant.py:rms_norm_dynamic_per_token_quant",
        shape,
        "bf16",
        module.rms_norm_dynamic_per_token_quant,
        module._rms_norm_dynamic_per_token_quant_torch,
        args,
        (0, 3),
        (),
        _mutated({"result": 0, "scale": 3}),
        {
            "result": Tolerance(0.2, 0.2),
            "scale": Tolerance(2e-2, 1e-4),
        },
        {
            "epsilon": 1e-6,
            "scale_ub": None,
            "residual": None,
            "output_dtype": "float8_e4m3fn",
        },
        (hidden, tokens),
    )


def _build_rms_norm_per_block(shape: tuple[int, ...]) -> Workload:
    module = _load_module(
        "rms_norm_per_block_quant",
        "pretuned_kernels/rms_norm_per_block_quant/"
        "rms_norm_per_block_quant.py",
    )
    tokens, hidden, group = shape
    args = module._make_inputs(tokens, hidden, group)
    return Workload(
        "vllm",
        "rms_norm_per_block_quant",
        "pretuned_kernels/rms_norm_per_block_quant/"
        "rms_norm_per_block_quant.py:rms_norm_per_block_quant",
        shape,
        "bf16",
        module.rms_norm_per_block_quant,
        module._rms_norm_per_block_quant_torch,
        args,
        (0, 3, 6),
        (6,),
        _mutated({"result": 0, "scale": 3, "residual": 6}),
        {
            "result": Tolerance(0.2, 0.2),
            "scale": Tolerance(2e-2, 1e-4),
            "residual": Tolerance(2e-2, 2e-2),
        },
        {
            "epsilon": 1e-6,
            "scale_ub": "scalar_tensor",
            "residual": True,
            "group_size": group,
            "is_scale_transposed": False,
            "output_dtype": "float8_e4m3fn",
        },
        (hidden, group, tokens),
    )


def _build_silu_per_block(shape: tuple[int, ...]) -> Workload:
    module = _load_module(
        "silu_and_mul_per_block_quant",
        "pretuned_kernels/silu_and_mul_per_block_quant/"
        "silu_and_mul_per_block_quant.py",
    )
    tokens, intermediate, group = shape
    x = torch.randn(
        tokens,
        2 * intermediate,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    out = torch.empty(tokens, intermediate, device=DEVICE, dtype=FP8)
    scales = torch.empty(
        tokens,
        intermediate // group,
        device=DEVICE,
        dtype=torch.float32,
    )
    args = (out, x, scales, group)
    return Workload(
        "vllm",
        "silu_and_mul_per_block_quant",
        "pretuned_kernels/silu_and_mul_per_block_quant/"
        "silu_and_mul_per_block_quant.py:silu_and_mul_per_block_quant",
        shape,
        "bf16",
        module.silu_and_mul_per_block_quant,
        module._silu_and_mul_per_block_quant_torch,
        args,
        (0, 2),
        (),
        _mutated({"out": 0, "scales": 2}),
        {
            "out": Tolerance(0.2, 0.2),
            "scales": Tolerance(1e-2, 1e-4),
        },
        {
            "group_size": group,
            "scale_ub": None,
            "is_scale_transposed": False,
            "output_dtype": "float8_e4m3fn",
        },
        (intermediate, group, tokens),
    )


def _fused_qk_norm_rope_reference(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor,
    forced_token_heads_per_warp: int = -1,
) -> None:
    num_tokens = qkv.shape[0]
    qk_heads = num_heads_q + num_heads_k
    embed_dim = cos_sin_cache.shape[1] // 2
    qkv_view = qkv.view(num_tokens, -1, head_dim)
    x = qkv_view[:, :qk_heads, :].to(torch.float32)
    rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    head_idx = torch.arange(qk_heads, device=qkv.device)
    weight = torch.where(
        (head_idx < num_heads_q)[None, :, None],
        q_weight[None, None, :],
        k_weight[None, None, :],
    )
    normalized = (x * rms).to(qkv.dtype) * weight
    cos = cos_sin_cache[position_ids, :embed_dim]
    sin = cos_sin_cache[position_ids, embed_dim : 2 * embed_dim]
    if is_neox and 2 * embed_dim == head_dim:
        cos_full = torch.cat((cos, cos), dim=-1)[:, None, :]
        sin_full = torch.cat((sin, sin), dim=-1)[:, None, :]
        first, second = normalized[..., :embed_dim], normalized[..., embed_dim:]
        rotated = torch.cat((-second, first), dim=-1)
        qkv_view[:, :qk_heads, :] = normalized * cos_full + rotated * sin_full
    else:
        qkv_view[:, :qk_heads, :] = normalized
        if is_neox:
            first_idx = torch.arange(embed_dim, device=qkv.device)
        else:
            first_idx = torch.arange(embed_dim, device=qkv.device) * 2
        second_idx = first_idx + (embed_dim if is_neox else 1)
        heads = qkv_view[:, :qk_heads, :]
        first = heads[:, :, first_idx]
        second = heads[:, :, second_idx]
        heads[:, :, first_idx] = first * cos[:, None, :] - second * sin[:, None, :]
        heads[:, :, second_idx] = second * cos[:, None, :] + first * sin[:, None, :]
        qkv_view[:, :qk_heads, :] = heads


def _build_fused_qk_norm_rope(shape: tuple[int, ...]) -> Workload:
    module = _load_module(
        "fused_qk_norm_rope",
        "pretuned_kernels/fused_qk_norm_rope/fused_qk_norm_rope.py",
    )
    tokens, q_heads, kv_heads = shape
    args = module._make_inputs(tokens, q_heads, kv_heads)
    return Workload(
        "vllm",
        "fused_qk_norm_rope",
        "pretuned_kernels/fused_qk_norm_rope/"
        "fused_qk_norm_rope.py:fused_qk_norm_rope",
        shape,
        "bf16",
        module.fused_qk_norm_rope,
        _fused_qk_norm_rope_reference,
        args,
        (0,),
        (0,),
        _mutated({"qkv": 0}),
        {"qkv": Tolerance(2e-2, 2e-2)},
        {
            "head_dim": 128,
            "eps": 1e-6,
            "is_neox": True,
            "rotary_dim": 128,
            "forced_token_heads_per_warp": -1,
            "mutation": "qkv",
        },
        (q_heads, kv_heads, tokens),
    )


GENERAL_SHAPES = {
    "rms_norm": (
        (2048, 48),
        (2048, 1023),
        (2048, 4096),
        (4096, 7168),
        (16384, 8192),
        (589824, 256),
    ),
    "layer_norm": (
        (4096, 1024),
        (4096, 3072),
        (8192, 5120),
        (4096, 12288),
        (4096, 16384),
        (1024, 36864),
    ),
    "softmax": (
        (4096, 256),
        (4096, 384),
        (4096, 768),
        (4096, 4096),
        (4096, 16384),
        (2048, 32768),
    ),
    "cross_entropy": (
        (2048, 32000),
        (1024, 256000),
        (2048, 128256),
        (8192, 128000),
        (4096, 152064),
        (2048, 256000),
    ),
}

ORIGINAL_SHAPES = {
    "kl_div": (
        (8192, 32768),
        (2048, 50257),
        (4096, 114688),
        (1024, 128256),
        (4096, 151936),
        (1024, 250000),
    ),
    "jsd": (
        (8192, 32768),
        (2048, 50257),
        (4096, 114688),
        (2048, 128256),
        (8192, 151936),
        (1024, 250000),
    ),
    "fused_linear_jsd": (
        (8192, 32000),
        (4096, 50257),
        (8192, 128256),
        (2048, 151936),
        (2048, 256000),
        (16384, 32000),
    ),
    "grpo": (
        (8, 1024, 32000),
        (8, 2048, 64000),
        (4, 2048, 128256),
        (8, 4096, 128256),
        (16, 1024, 50257),
        (4, 1024, 256000),
    ),
    "rms_norm_bwd": (
        (2048, 4096),
        (8192, 4096),
        (4096, 8192),
        (16384, 4096),
        (8192, 2048),
        (2048, 11008),
    ),
    "layer_norm_bwd": (
        (2048, 4096),
        (8192, 4096),
        (4096, 8192),
        (16384, 4096),
        (8192, 2048),
        (2048, 11008),
    ),
}


def _vllm_shapes(
    structural: tuple[int, ...], with_group: bool = False
) -> tuple[tuple[int, ...], ...]:
    if with_group:
        return tuple(
            (tokens, value, 128)
            for tokens in (1, 128, 8192)
            for value in structural
        )
    return tuple(
        (tokens, value)
        for tokens in (1, 128, 8192)
        for value in structural
    )


VLLM_SHAPES = {
    "dynamic_per_token_scaled_fp8_quant": _vllm_shapes((2048, 4096, 5120)),
    "per_token_group_fp8_quant": _vllm_shapes((2048, 4096, 5120), True),
    "rms_norm_dynamic_per_token_quant": _vllm_shapes((2048, 4096, 5120)),
    "rms_norm_per_block_quant": _vllm_shapes((2048, 4096, 5120), True),
    "silu_and_mul_per_block_quant": _vllm_shapes((6144, 12288, 25600), True),
    "fused_qk_norm_rope": tuple(
        (tokens, q_heads, 8)
        for tokens in (1, 128, 8192)
        for q_heads in (16, 32, 64)
    ),
}


SPECS: dict[str, KernelSpec] = {
    "rms_norm": KernelSpec(
        "rms_norm", "general_aot", GENERAL_SHAPES["rms_norm"], _build_rms_norm, True
    ),
    "layer_norm": KernelSpec(
        "layer_norm",
        "general_aot",
        GENERAL_SHAPES["layer_norm"],
        _build_layer_norm,
        True,
    ),
    "softmax": KernelSpec(
        "softmax", "general_aot", GENERAL_SHAPES["softmax"], _build_softmax, True
    ),
    "cross_entropy": KernelSpec(
        "cross_entropy",
        "general_aot",
        GENERAL_SHAPES["cross_entropy"],
        _build_cross_entropy,
        True,
    ),
    "kl_div": KernelSpec(
        "kl_div", "original", ORIGINAL_SHAPES["kl_div"], _build_kl_div, False
    ),
    "jsd": KernelSpec(
        "jsd", "original", ORIGINAL_SHAPES["jsd"], _build_jsd, False
    ),
    "fused_linear_jsd": KernelSpec(
        "fused_linear_jsd",
        "original",
        ORIGINAL_SHAPES["fused_linear_jsd"],
        _build_fused_linear_jsd,
        False,
    ),
    "grpo": KernelSpec(
        "grpo", "original", ORIGINAL_SHAPES["grpo"], _build_grpo, False
    ),
    "rms_norm_bwd": KernelSpec(
        "rms_norm_bwd",
        "original",
        ORIGINAL_SHAPES["rms_norm_bwd"],
        _build_rms_norm_bwd,
        False,
    ),
    "layer_norm_bwd": KernelSpec(
        "layer_norm_bwd",
        "original",
        ORIGINAL_SHAPES["layer_norm_bwd"],
        _build_layer_norm_bwd,
        False,
    ),
    "dynamic_per_token_scaled_fp8_quant": KernelSpec(
        "dynamic_per_token_scaled_fp8_quant",
        "vllm",
        VLLM_SHAPES["dynamic_per_token_scaled_fp8_quant"],
        _build_dynamic_per_token,
        True,
    ),
    "per_token_group_fp8_quant": KernelSpec(
        "per_token_group_fp8_quant",
        "vllm",
        VLLM_SHAPES["per_token_group_fp8_quant"],
        _build_per_token_group,
        True,
    ),
    "rms_norm_dynamic_per_token_quant": KernelSpec(
        "rms_norm_dynamic_per_token_quant",
        "vllm",
        VLLM_SHAPES["rms_norm_dynamic_per_token_quant"],
        _build_rms_norm_dynamic,
        True,
    ),
    "rms_norm_per_block_quant": KernelSpec(
        "rms_norm_per_block_quant",
        "vllm",
        VLLM_SHAPES["rms_norm_per_block_quant"],
        _build_rms_norm_per_block,
        True,
    ),
    "silu_and_mul_per_block_quant": KernelSpec(
        "silu_and_mul_per_block_quant",
        "vllm",
        VLLM_SHAPES["silu_and_mul_per_block_quant"],
        _build_silu_per_block,
        True,
    ),
    "fused_qk_norm_rope": KernelSpec(
        "fused_qk_norm_rope",
        "vllm",
        VLLM_SHAPES["fused_qk_norm_rope"],
        _build_fused_qk_norm_rope,
        True,
    ),
}


def build_workload(kernel: str, shape: tuple[int, ...]) -> Workload:
    return SPECS[kernel].build(shape)


def all_kernel_names() -> list[str]:
    return list(SPECS)
