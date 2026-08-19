"""Run one cell of the B200 reduction-heuristic audit."""

from __future__ import annotations

# The harness necessarily handles dynamically loaded kernel/config objects.
# ruff: noqa: ANN401
import argparse
from dataclasses import dataclass
import gc
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

AUDIT_DIR = Path(__file__).resolve().parent
ROOT = AUDIT_DIR.parents[1]
for path in (ROOT, ROOT / "examples", AUDIT_DIR):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    raise RuntimeError("This audit must run with CUDA_VISIBLE_DEVICES=1")

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

from matrix import SPEC_BY_NAME  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import triton  # noqa: E402
from triton import runtime as triton_runtime  # noqa: E402

import helion  # noqa: E402


@dataclass
class Case:
    kernel: str
    kfn: Any
    args: tuple[Any, ...]
    ref_fn: Callable[..., Any]
    observe: Callable[[tuple[Any, ...], Any], tuple[torch.Tensor, ...]]
    inplace_indices: tuple[int, ...] = ()
    restore_indices: tuple[int, ...] = ()
    rtol: float = 0.03
    atol: float = 0.03
    metadata: dict[str, Any] | None = None


@dataclass
class CapturedArm:
    name: str
    graph: torch.cuda.CUDAGraph


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _load_file(module_name: str, path: Path) -> Any:
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _pretuned_module(name: str) -> Any:
    path = ROOT / "pretuned_kernels" / name / f"{name}.py"
    return _load_file(f"_b200_audit_kernel_{name}", path)


def _aot_module(name: str) -> Any:
    path = ROOT / "pretuned_kernels" / name / f"_helion_aot_{name}_cuda_sm100.py"
    return _load_file(f"_b200_audit_aot_{name}", path)


def _flatten_output(value: Any) -> tuple[torch.Tensor, ...]:
    if torch.is_tensor(value):
        return (value,)
    if isinstance(value, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in value:
            result.extend(_flatten_output(item))
        return tuple(result)
    if value is None:
        return ()
    raise TypeError(f"unsupported output type: {type(value).__name__}")


def _observe_return(_args: tuple[Any, ...], value: Any) -> tuple[torch.Tensor, ...]:
    return _flatten_output(value)


def _observe_indices(
    indices: tuple[int, ...],
) -> Callable[[tuple[Any, ...], Any], tuple[torch.Tensor, ...]]:
    def observe(args: tuple[Any, ...], _value: Any) -> tuple[torch.Tensor, ...]:
        return tuple(args[index] for index in indices if torch.is_tensor(args[index]))

    return observe


def _clone_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(value.clone() if torch.is_tensor(value) else value for value in args)


def _ref_kl_div(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    log_target: bool = False,
    reduction: str = "batchmean",
    eps: float = 1e-10,
) -> torch.Tensor:
    del eps
    return F.kl_div(
        y_pred,
        y_true,
        reduction=reduction,
        log_target=log_target,
    )


def _ref_jsd(
    input_log: torch.Tensor,
    target_log: torch.Tensor,
    shift_labels: torch.Tensor | None = None,
    beta: float = 0.5,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    if shift_labels is not None:
        raise NotImplementedError("the audit uses shift_labels=None")
    del ignore_index
    x = input_log.float()
    y = target_log.float()
    q = torch.exp(x)
    p = torch.exp(y)
    mix = beta * p + (1.0 - beta) * q
    log_mix = torch.log(mix)
    row_loss = (beta * p * (y - log_mix) + (1.0 - beta) * q * (x - log_mix)).sum(-1)
    row_dx = ((1.0 - beta) * q * (x - log_mix)).sum(-1)
    scale = 1.0 / input_log.shape[0]
    return (row_loss * scale).sum(), row_dx * scale


def _ref_fused_linear_jsd(
    beta: float,
    ignore_index: int,
    temperature: float,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del ignore_index
    student = student_logits.float() / temperature
    teacher = teacher_logits.float() / temperature
    student_prob = torch.softmax(student, dim=-1)
    teacher_prob = torch.softmax(teacher, dim=-1)
    student_log_prob = torch.log_softmax(student, dim=-1)
    teacher_log_prob = torch.log_softmax(teacher, dim=-1)
    mix = (1.0 - beta) * student_prob + beta * teacher_prob
    log_mix = torch.log(mix)
    loss = (1.0 - beta) * (student_prob * (student_log_prob - log_mix)).sum(
        -1
    ) + beta * (teacher_prob * (teacher_log_prob - log_mix)).sum(-1)
    grad = ((1.0 - beta) / temperature) * (student_prob - mix)
    return loss, grad


def _ref_grpo(
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
    logits_f = logits[:, :-1, :].float() / temperature
    lse = torch.logsumexp(logits_f, dim=-1)
    logp = selected_logits.float() - lse
    old = logp if old_logp is None else old_logp
    coef_1 = torch.exp(logp - old)
    coef_2 = torch.clamp(coef_1, 1.0 - eps_low, 1.0 + eps_high)
    advantage = advantages[:, None]
    loss_1 = coef_1 * advantage
    loss_2 = coef_2 * advantage
    loss = -torch.minimum(loss_1, loss_2)
    if completion_mask is not None:
        loss = loss * completion_mask
    kl = torch.zeros_like(loss)
    if beta != 0.0 and ref_logp is not None:
        delta = ref_logp - logp
        kl = torch.exp(delta) - delta - 1.0
        if completion_mask is not None:
            kl = kl * completion_mask
        loss = loss + beta * kl
    clipped = (loss_1 < loss_2).float()
    return loss, kl, clipped, lse


def _ref_rms_norm_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    rsqrt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xf = x.float()
    dy = grad_out.float()
    inv = rsqrt.float()
    wf = weight.float()[None, :]
    grad_weight = (xf * dy * inv).sum(0)
    grad_x = wf * dy * inv - xf * inv**3 * (wf * dy * xf).mean(-1, keepdim=True)
    return grad_x.to(x.dtype), grad_weight.to(weight.dtype)


def _ref_layer_norm_bwd(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    compute_bias_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    xf = x.float()
    dy = grad_out.float()
    wf = weight.float()[None, :]
    x_hat = (xf - mean.float()[:, None]) * rstd.float()[:, None]
    grad_weight = (dy * x_hat).sum(0)
    grad_bias = dy.sum(0)
    wdy = wf * dy
    c1 = (x_hat * wdy).mean(-1, keepdim=True)
    c2 = wdy.mean(-1, keepdim=True)
    grad_x = (wdy - (x_hat * c1 + c2)) * rstd.float()[:, None]
    if compute_bias_grad:
        return (
            grad_x.to(x.dtype),
            grad_weight.to(weight.dtype),
            grad_bias.to(weight.dtype),
        )
    return grad_x.to(x.dtype), grad_weight.to(weight.dtype), None


def _build_general_aot(kernel: str, shape: tuple[int, ...]) -> Case:
    module = _pretuned_module(kernel)
    m, n = shape
    if kernel == "rms_norm":
        x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        return Case(
            kernel,
            module.rms_norm,
            (x, weight),
            module._rms_norm_torch,
            _observe_return,
            rtol=0.02,
            atol=0.02,
            metadata={"body": "pretuned_kernels/rms_norm/rms_norm.py:rms_norm"},
        )
    if kernel == "layer_norm":
        x = torch.randn(m, n, device="cuda", dtype=torch.float16)
        weight = torch.randn(n, device="cuda", dtype=torch.float16)
        bias = torch.randn(n, device="cuda", dtype=torch.float16)
        return Case(
            kernel,
            module.layer_norm,
            (x, weight, bias),
            module._layer_norm_torch,
            _observe_return,
            rtol=0.02,
            atol=0.02,
            metadata={"body": "pretuned_kernels/layer_norm/layer_norm.py:layer_norm"},
        )
    if kernel == "softmax":
        x = torch.randn(m, n, device="cuda", dtype=torch.float16)
        return Case(
            kernel,
            module.softmax,
            (x,),
            module._softmax_torch,
            _observe_return,
            rtol=0.02,
            atol=0.02,
            metadata={"body": "pretuned_kernels/softmax/softmax.py:softmax"},
        )
    if kernel == "cross_entropy":
        logits = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, n, (m,), device="cuda", dtype=torch.int64)
        return Case(
            kernel,
            module.cross_entropy,
            (logits, labels),
            module._cross_entropy_torch,
            _observe_return,
            rtol=0.02,
            atol=0.02,
            metadata={
                "body": "pretuned_kernels/cross_entropy/cross_entropy.py:cross_entropy"
            },
        )
    raise KeyError(kernel)


def _build_original(kernel: str, shape: tuple[int, ...]) -> Case:
    if kernel == "kl_div":
        module = importlib.import_module("examples.kl_div")
        m, n = shape
        y_pred = torch.randn(m, n, device="cuda", dtype=torch.bfloat16).log_softmax(-1)
        y_true = torch.randn(m, n, device="cuda", dtype=torch.bfloat16).softmax(-1)
        return Case(
            kernel,
            module.kl_div_forward,
            (y_pred, y_true, False, "batchmean", 1e-10),
            _ref_kl_div,
            _observe_return,
            metadata={"body": "examples/kl_div.py:kl_div_forward"},
        )
    if kernel == "jsd":
        module = importlib.import_module("examples.jsd")
        m, n = shape
        input_log = torch.randn(m, n, device="cuda", dtype=torch.bfloat16).log_softmax(
            -1
        )
        target_log = torch.randn(m, n, device="cuda", dtype=torch.bfloat16).log_softmax(
            -1
        )
        return Case(
            kernel,
            module.jsd_forward,
            (input_log, target_log, None, 0.5, -100),
            _ref_jsd,
            _observe_return,
            metadata={"body": "examples/jsd.py:jsd_forward"},
        )
    if kernel == "fused_linear_jsd":
        module = importlib.import_module("examples.fused_linear_jsd")
        m, n = shape
        student = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        teacher = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        return Case(
            kernel,
            module.jsd_kernel,
            (0.5, -100, 1.0, student, teacher),
            _ref_fused_linear_jsd,
            _observe_return,
            metadata={"body": "examples/fused_linear_jsd.py:jsd_kernel"},
        )
    if kernel == "grpo":
        module = importlib.import_module("examples.grpo_loss")
        batch, sequence, vocab = shape
        logits = torch.randn(
            batch,
            sequence + 1,
            vocab,
            device="cuda",
            dtype=torch.bfloat16,
        )
        completion_ids = torch.randint(
            0,
            vocab - 1,
            (batch, sequence),
            device="cuda",
            dtype=torch.int64,
        )
        temperature, beta, eps_low, eps_high = 0.9, 0.2, 0.2, 0.4
        selected = module.extract_selected_logits_pytorch(
            logits[:, :-1, :], completion_ids, temperature
        )
        old_logp = torch.randn(batch, sequence, device="cuda", dtype=torch.float32)
        ref_logp = torch.randn(batch, sequence, device="cuda", dtype=torch.float32)
        advantages = torch.randn(batch, device="cuda", dtype=torch.float32)
        mask = torch.ones(batch, sequence, device="cuda", dtype=torch.float32)
        return Case(
            kernel,
            module.grpo_loss_forward,
            (
                logits,
                selected,
                old_logp,
                ref_logp,
                advantages,
                mask,
                temperature,
                beta,
                eps_low,
                eps_high,
            ),
            _ref_grpo,
            _observe_return,
            metadata={"body": "examples/grpo_loss.py:grpo_loss_forward"},
        )
    if kernel == "rms_norm_bwd":
        module = importlib.import_module("examples.rms_norm")
        m, n = shape
        x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        grad_out = torch.randn_like(x)
        rsqrt = torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5).to(
            torch.bfloat16
        )
        return Case(
            kernel,
            module.rms_norm_bwd,
            (grad_out, x, weight, rsqrt),
            _ref_rms_norm_bwd,
            _observe_return,
            metadata={"body": "examples/rms_norm.py:rms_norm_bwd"},
        )
    if kernel == "layer_norm_bwd":
        module = importlib.import_module("examples.layer_norm")
        m, n = shape
        x = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        grad_out = torch.randn_like(x)
        mean = x.float().mean(-1)
        rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + 1e-5)
        return Case(
            kernel,
            module.layer_norm_bwd,
            (grad_out, x, mean, rstd, weight, True),
            _ref_layer_norm_bwd,
            _observe_return,
            metadata={"body": "examples/layer_norm.py:layer_norm_bwd"},
        )
    raise KeyError(kernel)


def _build_vllm(kernel: str, shape: tuple[int, ...]) -> Case:
    module = _pretuned_module(kernel)
    if kernel == "dynamic_per_token_scaled_fp8_quant":
        tokens, hidden = shape
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        result = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty(tokens, 1, device="cuda", dtype=torch.float32)
        args = (result, x, scale)
        observe = _observe_indices((0, 2))
        inplace = (0, 2)
    elif kernel == "per_token_group_fp8_quant":
        tokens, hidden, group = shape
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        output_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        output_s = torch.empty(
            tokens, hidden // group, device="cuda", dtype=torch.float32
        )
        args = (x, output_q, output_s, group, 1e-10, -448.0, 448.0, False)
        observe = _observe_indices((1, 2))
        inplace = (1, 2)
    elif kernel == "rms_norm_dynamic_per_token_quant":
        tokens, hidden = shape
        x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        weight = torch.normal(
            mean=1.0,
            std=1.0,
            size=(hidden,),
            device="cuda",
            dtype=torch.bfloat16,
        )
        result = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty(tokens, 1, device="cuda", dtype=torch.float32)
        args = (result, x, weight, scale, 1e-6)
        observe = _observe_indices((0, 3))
        inplace = (0, 3)
    elif kernel == "rms_norm_per_block_quant":
        tokens, hidden, group = shape
        args = module._make_inputs(tokens, hidden, group)
        observe = _observe_indices((0, 3, 6))
        inplace = (0, 3, 6)
    elif kernel == "silu_and_mul_per_block_quant":
        tokens, intermediate, group = shape
        x = torch.randn(
            tokens,
            2 * intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        )
        output = torch.empty(
            tokens,
            intermediate,
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        scales = torch.empty(
            tokens,
            intermediate // group,
            device="cuda",
            dtype=torch.float32,
        )
        args = (output, x, scales, group)
        observe = _observe_indices((0, 2))
        inplace = (0, 2)
    elif kernel == "fused_qk_norm_rope":
        tokens, q_heads, kv_heads = shape
        args = module._make_inputs(tokens, q_heads, kv_heads)
        observe = _observe_indices((0,))
        inplace = (0,)
    else:
        raise KeyError(kernel)

    ref_name = f"_{kernel}_torch"
    return Case(
        kernel,
        getattr(module, kernel),
        tuple(args),
        getattr(module, ref_name),
        observe,
        inplace_indices=inplace,
        restore_indices=(
            (6,)
            if kernel == "rms_norm_per_block_quant"
            else (0,)
            if kernel == "fused_qk_norm_rope"
            else ()
        ),
        rtol=0.05,
        atol=0.05,
        metadata={
            "body": f"pretuned_kernels/{kernel}/{kernel}.py:{kernel}",
            "primary_input_dtype": "bf16",
        },
    )


def build_case(cohort: str, kernel: str, shape: tuple[int, ...]) -> Case:
    torch.manual_seed(0)
    if cohort == "general_aot":
        return _build_general_aot(kernel, shape)
    if cohort == "original":
        return _build_original(kernel, shape)
    if cohort == "vllm":
        return _build_vllm(kernel, shape)
    raise KeyError(cohort)


def _authored_kernel(kfn: Any, config: Any | None = None) -> Any:
    settings = kfn.settings
    return helion.kernel(
        kfn.fn,
        config=config,
        static_shapes=settings.static_shapes,
        ignore_warnings=list(settings.ignore_warnings or []),
    )


def _extract_configs(kfn: Any, args: tuple[Any, ...]) -> tuple[Any, Any, list[str]]:
    plain = _authored_kernel(kfn)
    plain.reset()
    bound = plain.bind(args)
    config_spec = bound.env.config_spec
    seeds = list(config_spec.compiler_seed_configs)
    fired = list(config_spec.autotuner_heuristics)
    with bound.env:
        default = config_spec._base_default_config()
    return (seeds[0] if seeds else None), default, fired


def _config_dict(config: Any | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return dict(config.config)


def _aot_key(kernel: str, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if kernel in {
        "dynamic_per_token_scaled_fp8_quant",
        "rms_norm_dynamic_per_token_quant",
    }:
        tokens, hidden = shape
        return hidden, tokens
    if kernel in {"per_token_group_fp8_quant", "rms_norm_per_block_quant"}:
        tokens, hidden, group = shape
        return hidden, group, tokens
    if kernel == "silu_and_mul_per_block_quant":
        tokens, intermediate, group = shape
        return intermediate, group, tokens
    if kernel == "fused_qk_norm_rope":
        tokens, q_heads, kv_heads = shape
        return q_heads, kv_heads, tokens
    return None


def _select_aot(
    kernel: str, shape: tuple[int, ...], args: tuple[Any, ...]
) -> tuple[Any, int, bool]:
    module = _aot_module(kernel)
    index = getattr(module, f"key_{kernel}")(*args)
    body = getattr(module, f"autotune_{kernel}")(*args)
    exact_key = _aot_key(kernel, shape)
    exact = (
        True if exact_key is None else exact_key in getattr(module, "_INDEX_BY_KEY", {})
    )
    return helion.Config(**body), int(index), exact


def _tensor_tolerance(
    actual: torch.Tensor, expected: torch.Tensor, case: Case
) -> tuple[float, float]:
    float8_dtypes = {torch.float8_e4m3fn}
    if hasattr(torch, "float8_e4m3fnuz"):
        float8_dtypes.add(torch.float8_e4m3fnuz)
    if actual.dtype in float8_dtypes or expected.dtype in float8_dtypes:
        return 0.2, 1.0
    return case.rtol, case.atol


def _compare(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    case: Case,
) -> tuple[bool, str, list[dict[str, Any]]]:
    if len(actual) != len(expected):
        return (
            False,
            f"output-count {len(actual)} != {len(expected)}",
            [],
        )
    details = []
    outputs = []
    all_ok = True
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        output: dict[str, Any] = {
            "index": index,
            "actual_dtype": str(got.dtype).removeprefix("torch."),
            "expected_dtype": str(want.dtype).removeprefix("torch."),
            "actual_shape": list(got.shape),
            "expected_shape": list(want.shape),
        }
        if got.shape != want.shape:
            all_ok = False
            details.append(f"{index}:shape {tuple(got.shape)}!={tuple(want.shape)}")
            output["ok"] = False
            outputs.append(output)
            continue
        got_f = got.detach().float()
        want_f = want.detach().float()
        rtol, atol = _tensor_tolerance(got, want, case)
        ok = bool(torch.allclose(got_f, want_f, rtol=rtol, atol=atol))
        all_ok = all_ok and ok
        if got.numel():
            abs_error = (got_f - want_f).abs()
            max_abs = float(abs_error.max())
            significant = want_f.abs() > atol
            max_rel = (
                float((abs_error[significant] / want_f.abs()[significant]).max())
                if bool(significant.any())
                else 0.0
            )
            tolerance = atol + rtol * want_f.abs()
            max_tolerance_ratio = float((abs_error / tolerance).max())
        else:
            max_abs = 0.0
            max_rel = 0.0
            max_tolerance_ratio = 0.0
        output.update(
            {
                "ok": ok,
                "rtol": rtol,
                "atol": atol,
                "max_abs_error": max_abs,
                "max_relative_error": max_rel,
                "max_tolerance_ratio": max_tolerance_ratio,
            }
        )
        outputs.append(output)
        details.append(
            f"{index}:ok={ok},maxabs={max_abs:.4g},maxrel={max_rel:.4g},"
            f"rtol={rtol:.4g},atol={atol:.4g}"
        )
    return all_ok, ";".join(details), outputs


def _accuracy_check(
    fn: Callable[..., Any],
    case: Case,
) -> tuple[bool, str, list[dict[str, Any]]]:
    if case.inplace_indices:
        kernel_args = _clone_args(case.args)
        ref_args = _clone_args(case.args)
    else:
        kernel_args = case.args
        ref_args = case.args
    actual_ret = fn(*kernel_args)
    expected_ret = case.ref_fn(*ref_args)
    torch.cuda.synchronize()
    actual = case.observe(kernel_args, actual_ret)
    expected = case.observe(ref_args, expected_ret)
    result = _compare(actual, expected, case)
    del actual_ret, expected_ret, actual, expected
    if case.inplace_indices:
        del kernel_args, ref_args
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _dynamo_explain(case: Case) -> dict[str, Any]:
    if os.environ.get("AUDIT_DYNAMO_EXPLAIN") != "1":
        return {"enabled": False}
    args = _clone_args(case.args) if case.inplace_indices else case.args
    try:
        explanation = torch._dynamo.explain(case.ref_fn)(*args)
        return {
            "enabled": True,
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "break_reasons": [str(reason) for reason in explanation.break_reasons],
        }
    except Exception as exc:
        return {
            "enabled": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if case.inplace_indices:
            del args
            gc.collect()
            torch.cuda.empty_cache()


def _capture_graph(
    fn: Callable[[], Any],
    restore: Callable[[], None],
) -> torch.cuda.CUDAGraph:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            restore()
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    restore()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    restore()
    torch.cuda.synchronize()
    return graph


def _spread(values: list[float]) -> float:
    median = statistics.median(values)
    if median <= 0:
        return 0.0
    return (max(values) - min(values)) / median


def _time_graphs(
    graphs: dict[str, torch.cuda.CUDAGraph],
    restore: Callable[[], None],
) -> dict[str, dict[str, Any]]:
    rounds = int(os.environ.get("AUDIT_ROUNDS", "9"))
    rounds_high = int(os.environ.get("AUDIT_ROUNDS_HIGH", "15"))
    spread_gate = float(os.environ.get("AUDIT_SPREAD_GATE", "0.05"))
    budget_ms = float(os.environ.get("AUDIT_ROUND_BUDGET_MS", "100"))
    reps_fixed = os.environ.get("AUDIT_REPS_PER_ROUND")
    reps_floor = int(os.environ.get("AUDIT_REPS_FLOOR", "5"))
    reps_cap = int(os.environ.get("AUDIT_REPS_CAP", "1000"))
    clear_buffer = triton_runtime.driver.active.get_empty_cache_for_benchmark()

    def clear_l2() -> None:
        triton_runtime.driver.active.clear_cache(clear_buffer)

    for graph in graphs.values():
        for _ in range(3):
            restore()
            clear_l2()
            graph.replay()
    torch.cuda.synchronize()

    if reps_fixed is not None:
        reps = int(reps_fixed)
    else:
        probe = 3
        start = time.perf_counter()
        for _ in range(probe):
            for graph in graphs.values():
                restore()
                clear_l2()
                graph.replay()
        torch.cuda.synchronize()
        per_rep_ms = (time.perf_counter() - start) * 1000.0 / probe
        reps = (
            reps_cap
            if per_rep_ms <= 0
            else max(reps_floor, min(reps_cap, int(budget_ms / per_rep_ms)))
        )

    names = list(graphs)
    medians = {name: [] for name in names}

    def collect(start_round: int, end_round: int) -> None:
        for round_index in range(start_round, end_round):
            offset = round_index % len(names)
            order = names[offset:] + names[:offset]
            events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
                name: [] for name in names
            }
            for _ in range(reps):
                for name in order:
                    restore()
                    clear_l2()
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    graphs[name].replay()
                    end.record()
                    events[name].append((begin, end))
            torch.cuda.synchronize()
            for name in names:
                samples = [
                    begin.elapsed_time(end) * 1000.0 for begin, end in events[name]
                ]
                medians[name].append(statistics.median(samples))

    collect(0, rounds)
    if any(_spread(values) > spread_gate for values in medians.values()):
        collect(rounds, rounds_high)

    return {
        name: {
            "cold_l2_graph_us": round(statistics.median(values), 3),
            "spread": round(_spread(values), 5),
            "round_medians_us": [round(value, 3) for value in values],
            "reps_per_round": reps,
        }
        for name, values in medians.items()
    }


def _environment_metadata() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    try:
        driver_version = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "1",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        driver_version = "unknown"
    return {
        "helion_commit": _git("rev-parse", "HEAD"),
        "helion_branch": _git("branch", "--show-current"),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "triton_version": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_index": 1,
        "logical_device": 0,
        "gpu_name": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": props.total_memory,
    }


def run_cell(kernel: str, shape_index: int) -> dict[str, Any]:
    spec = SPEC_BY_NAME[kernel]
    shape = spec.shapes[shape_index]
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one visible GPU, found {torch.cuda.device_count()}"
        )
    if torch.cuda.get_device_capability(0) != (10, 0):
        raise RuntimeError(
            f"expected B200 sm100, got {torch.cuda.get_device_capability(0)}"
        )

    case = build_case(spec.cohort, kernel, shape)
    seed_config, default_config, fired = _extract_configs(case.kfn, case.args)
    row: dict[str, Any] = {
        "cohort": spec.cohort,
        "kernel": kernel,
        "shape_index": shape_index,
        "shape": list(shape),
        "dtype": spec.dtype,
        "has_aot": spec.has_aot,
        "metadata": case.metadata or {},
        "tensor_arguments": [
            {
                "index": index,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
                "inplace_output": index in case.inplace_indices,
                "restored_before_replay": index in case.restore_indices,
            }
            for index, value in enumerate(case.args)
            if torch.is_tensor(value)
        ],
        "scalar_arguments": [
            {"index": index, "value": value}
            for index, value in enumerate(case.args)
            if not torch.is_tensor(value)
        ],
        "environment": _environment_metadata(),
        "fired_heuristics": fired,
        "reduction_seed_present": seed_config is not None,
        "dynamo_explain": _dynamo_explain(case),
        "arms": {},
    }

    configs: list[tuple[str, Any | None]] = [
        ("seed", seed_config),
        ("default", default_config),
    ]
    row["seed_config"] = _config_dict(seed_config)
    row["default_config"] = _config_dict(default_config)
    if spec.has_aot:
        aot_config, aot_index, aot_exact = _select_aot(kernel, shape, case.args)
        configs.append(("aot_sm100", aot_config))
        row["aot_config"] = _config_dict(aot_config)
        row["aot_index"] = aot_index
        row["aot_exact_key_or_sweep_shape"] = aot_exact

    callables: dict[str, Callable[[], Any]] = {}
    for name, config in configs:
        arm = {"config": _config_dict(config)}
        if config is None:
            arm["status"] = "no-config"
            row["arms"][name] = arm
            continue
        try:
            compiled = _authored_kernel(case.kfn, config)
            accurate, detail, outputs = _accuracy_check(compiled, case)
            arm["accuracy"] = accurate
            arm["accuracy_detail"] = detail
            arm["accuracy_outputs"] = outputs
            arm["status"] = "ok" if accurate else "accuracy-fail"
            callables[name] = lambda compiled=compiled: compiled(*case.args)
        except Exception as exc:
            arm["status"] = f"compile-or-run-fail:{type(exc).__name__}"
            arm["error"] = f"{type(exc).__name__}: {exc}"
            arm["traceback"] = traceback.format_exc()
        row["arms"][name] = arm

    try:
        torch._dynamo.reset()
        torch_compiled = torch.compile(case.ref_fn)
        accurate, detail, outputs = _accuracy_check(torch_compiled, case)
        row["arms"]["torch_compile"] = {
            "status": "ok" if accurate else "accuracy-fail",
            "accuracy": accurate,
            "accuracy_detail": detail,
            "accuracy_outputs": outputs,
        }
        callables["torch_compile"] = lambda: torch_compiled(*case.args)
    except Exception as exc:
        row["arms"]["torch_compile"] = {
            "status": f"compile-or-run-fail:{type(exc).__name__}",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    restore_sources = {
        index: case.args[index].clone() for index in case.restore_indices
    }

    def restore() -> None:
        for index, source in restore_sources.items():
            case.args[index].copy_(source)

    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for name, fn in callables.items():
        try:
            graphs[name] = _capture_graph(fn, restore)
        except Exception as exc:
            row["arms"][name]["status"] = f"capture-fail:{type(exc).__name__}"
            row["arms"][name]["capture_error"] = f"{type(exc).__name__}: {exc}"
            row["arms"][name]["capture_traceback"] = traceback.format_exc()

    if graphs:
        timing = _time_graphs(graphs, restore)
        for name, measurements in timing.items():
            row["arms"][name].update(measurements)

    seed_us = row["arms"].get("seed", {}).get("cold_l2_graph_us")
    ratios = {}
    if seed_us:
        for name in ("default", "torch_compile", "aot_sm100"):
            arm = row["arms"].get(name, {})
            value = arm.get("cold_l2_graph_us")
            if value and arm.get("accuracy") is not False:
                ratios[name] = round(value / seed_us, 5)
    row["ratios_vs_seed"] = ratios
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, choices=sorted(SPEC_BY_NAME))
    parser.add_argument("--shape-index", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        row = run_cell(args.kernel, args.shape_index)
    except Exception as exc:
        spec = SPEC_BY_NAME[args.kernel]
        row = {
            "cohort": spec.cohort,
            "kernel": args.kernel,
            "shape_index": args.shape_index,
            "shape": list(spec.shapes[args.shape_index]),
            "dtype": spec.dtype,
            "fatal_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    output.write_text(json.dumps(row, indent=2, default=str) + "\n")
    print(json.dumps(row, default=str), flush=True)
    if "fatal_error" in row:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
