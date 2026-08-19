"""Run one cell of the B200 pointwise-heuristic audit."""

from __future__ import annotations

# The harness necessarily handles dynamically loaded kernel/config objects.
# ruff: noqa: ANN401
import argparse
from dataclasses import dataclass
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
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

from general_kernels import geglu  # noqa: E402
from general_kernels import swiglu_fwd  # noqa: E402
from matrix import SPEC_BY_NAME  # noqa: E402
from sglang_kernel import select_sm100_config  # noqa: E402
from sglang_kernel import silu_and_mul_interleaved  # noqa: E402
from sglang_kernel import silu_and_mul_interleaved_torch  # noqa: E402
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
    rtol: float = 0.03
    atol: float = 0.03
    metadata: dict[str, Any] | None = None


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stderr=subprocess.DEVNULL,
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
    return _load_file(f"_b200_pointwise_kernel_{name}", path)


def _aot_module(name: str, arch: str) -> Any:
    path = ROOT / "pretuned_kernels" / name / f"_helion_aot_{name}_cuda_{arch}.py"
    return _load_file(f"_b200_pointwise_aot_{name}_{arch}", path)


def _ref_swiglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (F.silu(a.float()) * b.float()).to(a.dtype)


def _ref_geglu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.float()
    gelu = (
        0.5
        * af
        * (1.0 + torch.tanh(0.7978845608028654 * (af + 0.044715 * af * af * af)))
    )
    return (gelu * b.float()).to(a.dtype)


def _build_case(kernel: str, shape: tuple[Any, ...]) -> Case:
    torch.manual_seed(0)
    if kernel == "swiglu":
        m, n = shape
        a = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        return Case(
            kernel,
            swiglu_fwd,
            (a, b),
            _ref_swiglu,
            metadata={
                "body": (
                    "perf-repro/b200-pointwise-audit/general_kernels.py:swiglu_fwd"
                ),
                "copied_from": "examples/swiglu.py:_swiglu_fwd",
            },
        )
    if kernel == "geglu":
        m, n = shape
        a = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        return Case(
            kernel,
            geglu,
            (a, b),
            _ref_geglu,
            metadata={
                "body": ("perf-repro/b200-pointwise-audit/general_kernels.py:geglu"),
                "copied_from": "examples/geglu.py:_geglu",
            },
        )
    if kernel == "rope":
        module = _pretuned_module("rope")
        batch, heads, sequence, head_dim = shape
        q = torch.randn(
            batch,
            heads,
            sequence,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        k = torch.randn_like(q)
        angles = torch.randn(
            batch,
            sequence,
            head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return Case(
            kernel,
            module.rope_fwd,
            (q, k, cos, sin),
            module.rope_pytorch,
            metadata={
                "body": "pretuned_kernels/rope/rope.py:rope_fwd",
                "shape_key": [heads * head_dim, sequence],
                "q_heads": heads,
                "k_heads": heads,
            },
        )
    if kernel == "silu_mul_fp8":
        module = _pretuned_module("silu_mul_fp8")
        tokens, intermediate = shape
        x = torch.randn(
            tokens,
            2 * intermediate,
            device="cuda",
            dtype=torch.bfloat16,
        )
        scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
        return Case(
            kernel,
            module.silu_mul_fp8,
            (x, scale),
            module._silu_mul_fp8_torch,
            rtol=0.2,
            atol=1.0,
            metadata={
                "body": ("pretuned_kernels/silu_mul_fp8/silu_mul_fp8.py:silu_mul_fp8"),
                "primary_input_dtype": "bf16",
                "output_dtype": "float8_e4m3fn",
            },
        )
    if kernel == "silu_and_mul_interleaved":
        rows, hidden, has_topk_weights = shape
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        weights = (
            torch.randn(rows, device="cuda", dtype=torch.bfloat16)
            if has_topk_weights
            else None
        )
        return Case(
            kernel,
            silu_and_mul_interleaved,
            (x, weights, None),
            silu_and_mul_interleaved_torch,
            metadata={
                "body": (
                    "perf-repro/b200-pointwise-audit/sglang_kernel.py:"
                    "silu_and_mul_interleaved"
                ),
                "sglang_source_commit": ("5f79cf35110d6a0be828f266160b75d83a2a6276"),
                "has_topk_weights": has_topk_weights,
            },
        )
    raise KeyError(kernel)


def _authored_kernel(kfn: Any, config: Any | None = None) -> Any:
    return helion.kernel(kfn.fn, config=config, settings=kfn.settings)


def _extract_configs(
    kfn: Any,
    args: tuple[Any, ...],
) -> tuple[Any | None, Any, list[str], dict[str, Any] | None]:
    plain = _authored_kernel(kfn)
    plain.reset()
    bound = plain.bind(args)
    config_spec = bound.env.config_spec
    seeds = list(config_spec.compiler_seed_configs)
    fired = list(config_spec.autotuner_heuristics)
    with bound.env:
        default = config_spec._base_default_config()
        facts = config_spec.pointwise_facts
        fact = facts[0]._asdict() if facts else None
    return (seeds[0] if seeds else None), default, fired, fact


def _config_dict(config: Any | None) -> dict[str, Any] | None:
    if config is None:
        return None
    return dict(config.config)


def _select_aot(
    kernel: str,
    shape: tuple[Any, ...],
    case: Case,
) -> tuple[Any, dict[str, Any]]:
    if kernel == "rope":
        batch, heads, sequence, head_dim = shape
        del batch
        hidden = heads * head_dim
        module = _aot_module("rope", "sm90")
        key = (hidden, sequence)
        config = helion.Config(**module.autotune_rope_fwd(*key))
        return config, {
            "architecture": "sm90",
            "selector_key": list(key),
            "exact_key": key in module._FWD_CONFIGS_BY_SHAPE,
            "fallback_key": None
            if key in module._FWD_CONFIGS_BY_SHAPE
            else list(module._DEFAULT_SHAPE),
            "source": ("pretuned_kernels/rope/_helion_aot_rope_cuda_sm90.py"),
        }
    if kernel == "silu_mul_fp8":
        tokens, intermediate = shape
        module = _aot_module("silu_mul_fp8", "sm90")
        index = int(module.key_silu_mul_fp8(*case.args))
        key = (intermediate, tokens)
        config = helion.Config(**module.autotune_silu_mul_fp8(*case.args))
        return config, {
            "architecture": "sm90",
            "selector_key": list(key),
            "config_index": index,
            "exact_key": key in module._INDEX_BY_KEY,
            "source": (
                "pretuned_kernels/silu_mul_fp8/_helion_aot_silu_mul_fp8_cuda_sm90.py"
            ),
        }
    if kernel == "silu_and_mul_interleaved":
        rows, hidden, has_topk_weights = shape
        del rows
        config, index, key = select_sm100_config(hidden, has_topk_weights)
        return config, {
            "architecture": "sm100",
            "selector_key": [
                key[0],
                list(key[1]),
                list(key[2]),
            ],
            "config_index": index,
            "exact_key": True,
            "source": (
                "sglang/srt/layers/moe/moe_runner/triton_utils/configs/"
                "silu_and_mul_interleaved_sm_100.json"
            ),
            "source_commit": "5f79cf35110d6a0be828f266160b75d83a2a6276",
        }
    raise KeyError(kernel)


def _flatten_output(value: Any) -> tuple[torch.Tensor, ...]:
    if torch.is_tensor(value):
        return (value,)
    if isinstance(value, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in value:
            result.extend(_flatten_output(item))
        return tuple(result)
    raise TypeError(f"unsupported output type: {type(value).__name__}")


def _compare(
    actual: Any,
    expected: Any,
    case: Case,
) -> tuple[bool, str, list[dict[str, Any]]]:
    actual_tensors = _flatten_output(actual)
    expected_tensors = _flatten_output(expected)
    if len(actual_tensors) != len(expected_tensors):
        return (
            False,
            f"output-count {len(actual_tensors)} != {len(expected_tensors)}",
            [],
        )
    details = []
    outputs = []
    all_ok = True
    for index, (got, want) in enumerate(
        zip(actual_tensors, expected_tensors, strict=True)
    ):
        output: dict[str, Any] = {
            "index": index,
            "actual_dtype": str(got.dtype).removeprefix("torch."),
            "expected_dtype": str(want.dtype).removeprefix("torch."),
            "actual_shape": list(got.shape),
            "expected_shape": list(want.shape),
        }
        if got.shape != want.shape:
            all_ok = False
            output["ok"] = False
            outputs.append(output)
            details.append(f"{index}:shape {tuple(got.shape)}!={tuple(want.shape)}")
            continue
        got_f = got.detach().float()
        want_f = want.detach().float()
        ok = bool(
            torch.allclose(
                got_f,
                want_f,
                rtol=case.rtol,
                atol=case.atol,
            )
        )
        all_ok = all_ok and ok
        if got.numel():
            abs_error = (got_f - want_f).abs()
            max_abs = float(abs_error.max())
            significant = want_f.abs() > case.atol
            max_rel = (
                float((abs_error[significant] / want_f.abs()[significant]).max())
                if bool(significant.any())
                else 0.0
            )
            tolerance = case.atol + case.rtol * want_f.abs()
            max_tolerance_ratio = float((abs_error / tolerance).max())
        else:
            max_abs = 0.0
            max_rel = 0.0
            max_tolerance_ratio = 0.0
        output.update(
            {
                "ok": ok,
                "rtol": case.rtol,
                "atol": case.atol,
                "max_abs_error": max_abs,
                "max_relative_error": max_rel,
                "max_tolerance_ratio": max_tolerance_ratio,
            }
        )
        outputs.append(output)
        details.append(
            f"{index}:ok={ok},maxabs={max_abs:.4g},maxrel={max_rel:.4g},"
            f"rtol={case.rtol:.4g},atol={case.atol:.4g}"
        )
    return all_ok, ";".join(details), outputs


def _accuracy_check(
    fn: Callable[..., Any],
    case: Case,
) -> tuple[bool, str, list[dict[str, Any]]]:
    actual = fn(*case.args)
    expected = case.ref_fn(*case.args)
    torch.cuda.synchronize()
    result = _compare(actual, expected, case)
    del actual, expected
    return result


def _dynamo_explain(case: Case) -> dict[str, Any]:
    if os.environ.get("AUDIT_DYNAMO_EXPLAIN") != "1":
        return {"enabled": False}
    try:
        explanation = torch._dynamo.explain(case.ref_fn)(*case.args)
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


def _capture_repeat(
    body: Callable[[], Any],
    *,
    repeats: int,
) -> tuple[torch.cuda.CUDAGraph, Any]:
    for _ in range(3):
        body()
    torch.cuda.synchronize()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            body()
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    output = None
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            output = body()
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph, output


def _spread(values: list[float]) -> float:
    median = statistics.median(values)
    if median <= 0:
        return 0.0
    return (max(values) - min(values)) / median


def _collect_graph_round(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    round_index: int,
    iterations: int,
) -> dict[str, float]:
    names = list(graphs)
    samples = {name: [] for name in names}
    events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
    for iteration in range(iterations):
        offset = (round_index + iteration) % len(names)
        order = names[offset:] + names[:offset]
        for name in order:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name].replay()
            end.record()
            events.append((name, begin, end))
    torch.cuda.synchronize()
    for name, begin, end in events:
        samples[name].append(begin.elapsed_time(end))
    return {name: statistics.median(values) for name, values in samples.items()}


def _time_calibrated_cold_graphs(
    callables: dict[str, Callable[[], Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    rounds = int(os.environ.get("AUDIT_ROUNDS", "9"))
    rounds_high = int(os.environ.get("AUDIT_ROUNDS_HIGH", "15"))
    spread_gate = float(os.environ.get("AUDIT_SPREAD_GATE", "0.05"))
    batch = int(os.environ.get("AUDIT_COLD_BATCH", "16"))
    iterations = int(os.environ.get("AUDIT_COLD_ITERS", "20"))

    active = triton_runtime.driver.active
    clear_buffer = active.get_empty_cache_for_benchmark()

    def clear_l2() -> None:
        active.clear_cache(clear_buffer)

    flush_graph, flush_output = _capture_repeat(clear_l2, repeats=batch)
    del flush_output
    graphs = {"__flush_only__": flush_graph}
    kept_outputs = {}
    for name, fn in callables.items():

        def cold_body(fn: Callable[[], Any] = fn) -> Any:
            clear_l2()
            return fn()

        graph, output = _capture_repeat(cold_body, repeats=batch)
        graphs[name] = graph
        kept_outputs[name] = output

    for graph in graphs.values():
        graph.replay()
    torch.cuda.synchronize()

    calibrated = {name: [] for name in callables}
    raw_batched = {name: [] for name in callables}
    flush_rounds = []

    def collect(start: int, end: int) -> None:
        for round_index in range(start, end):
            medians = _collect_graph_round(
                graphs,
                round_index=round_index,
                iterations=iterations,
            )
            flush_ms = medians["__flush_only__"]
            flush_rounds.append(flush_ms)
            for name in callables:
                both_ms = medians[name]
                raw_batched[name].append(both_ms)
                calibrated[name].append((both_ms - flush_ms) * 1000.0 / batch)

    collect(0, rounds)
    if any(
        _spread(values) > spread_gate
        for values in calibrated.values()
        if all(value > 0 for value in values)
    ):
        collect(rounds, rounds_high)

    result = {}
    for name, values in calibrated.items():
        positive = all(value > 0 and math.isfinite(value) for value in values)
        result[name] = {
            "status": "ok" if positive else "invalid-calibrated-timing",
            "cold_l2_graph_us": (
                round(statistics.median(values), 4) if positive else None
            ),
            "spread": round(_spread(values), 5) if positive else None,
            "round_medians_us": [round(value, 4) for value in values],
            "raw_flush_plus_op_ms": [round(value, 6) for value in raw_batched[name]],
            "batch": batch,
            "iterations_per_round": iterations,
        }
    method = {
        "name": "calibrated cold-L2 CUDA graph",
        "formula": ("(flush_plus_operation_graph_ms - flush_only_graph_ms) / batch"),
        "flush_inside_graph": True,
        "interleaved": True,
        "rounds_requested": rounds,
        "rounds_high_spread": rounds_high,
        "spread_gate": spread_gate,
        "batch": batch,
        "iterations_per_round": iterations,
        "flush_only_round_medians_ms": [round(value, 6) for value in flush_rounds],
    }
    # Keep graph outputs alive until all timing has completed.
    del kept_outputs
    return result, method


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

    case = _build_case(kernel, shape)
    seed_config, default_config, fired, pointwise_fact = _extract_configs(
        case.kfn,
        case.args,
    )
    row: dict[str, Any] = {
        "cohort": spec.cohort,
        "kernel": kernel,
        "shape_index": shape_index,
        "shape": list(shape),
        "dtype": spec.dtype,
        "has_aot": spec.has_aot,
        "aot_arch": spec.aot_arch,
        "metadata": case.metadata or {},
        "tensor_arguments": [
            {
                "index": index,
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
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
        "pointwise_seed_present": seed_config is not None,
        "pointwise_fact": pointwise_fact,
        "dynamo_explain": _dynamo_explain(case),
        "arms": {},
    }

    configs: list[tuple[str, Any | None]] = [
        ("default", default_config),
        ("default_null", default_config),
        ("seed", seed_config),
    ]
    row["default_config"] = _config_dict(default_config)
    row["seed_config"] = _config_dict(seed_config)
    if spec.has_aot:
        aot_config, aot_metadata = _select_aot(kernel, shape, case)
        configs.append(("aot", aot_config))
        row["aot_config"] = _config_dict(aot_config)
        row["aot_selector"] = aot_metadata

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
            if accurate:
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
        if accurate:
            callables["torch_compile"] = lambda: torch_compiled(*case.args)
    except Exception as exc:
        row["arms"]["torch_compile"] = {
            "status": f"compile-or-run-fail:{type(exc).__name__}",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }

    if callables:
        timing, method = _time_calibrated_cold_graphs(callables)
        row["timing_method"] = method
        for name, measurements in timing.items():
            row["arms"][name].update(measurements)

    default_us = row["arms"].get("default", {}).get("cold_l2_graph_us")
    ratios = {}
    if default_us:
        for name in ("seed", "torch_compile", "aot"):
            arm = row["arms"].get(name, {})
            value = arm.get("cold_l2_graph_us")
            if value and arm.get("accuracy") is not False:
                ratios[name] = round(default_us / value, 6)
    row["relative_performance_vs_default"] = ratios

    null_us = row["arms"].get("default_null", {}).get("cold_l2_graph_us")
    row["default_null_delta"] = (
        round(abs(default_us - null_us) / default_us, 6)
        if default_us and null_us
        else None
    )
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
    gc.collect()
    torch.cuda.empty_cache()
    if "fatal_error" in row:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
