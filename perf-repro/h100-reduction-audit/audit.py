#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
import gc
import importlib.util
import json
import os
from pathlib import Path
import platform
import signal
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any
from typing import Callable


os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402
from triton import runtime as triton_runtime  # noqa: E402
import triton  # noqa: E402

import helion  # noqa: E402

from workloads import all_kernel_names  # noqa: E402
from workloads import build_workload  # noqa: E402
from workloads import describe_inputs  # noqa: E402
from workloads import ROOT  # noqa: E402
from workloads import SPECS  # noqa: E402
from workloads import Tolerance  # noqa: E402
from workloads import Workload  # noqa: E402
from workloads import WorkloadState  # noqa: E402


AUDIT_DIR = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
DEFAULT_ROUNDS = int(os.environ.get("H100_AUDIT_ROUNDS", "9"))
NOISY_ROUNDS = int(os.environ.get("H100_AUDIT_NOISY_ROUNDS", "15"))
SPREAD_THRESHOLD = float(os.environ.get("H100_AUDIT_SPREAD_THRESHOLD", "0.05"))
ROUND_TARGET_MS = float(os.environ.get("H100_AUDIT_ROUND_TARGET_MS", "100"))
MIN_REPS = int(os.environ.get("H100_AUDIT_MIN_REPS", "5"))
MAX_REPS = int(os.environ.get("H100_AUDIT_MAX_REPS", "1000"))
FIXED_REPS = os.environ.get("H100_AUDIT_REPS")
COMPILE_TIMEOUT_SECONDS = int(
    os.environ.get("H100_AUDIT_COMPILE_TIMEOUT_SECONDS", "300")
)
REDUCTION_HEURISTICS = {
    "triton_reduction_tile",
    "triton_reduction_user_tile",
}


class CompileTimeout(RuntimeError):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise CompileTimeout(
        f"compile or first execution exceeded {COMPILE_TIMEOUT_SECONDS}s"
    )


@contextmanager
def compile_timeout():
    previous = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(COMPILE_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _reap_compile_children() -> None:
    try:
        parent = os.getpid()
        output = subprocess.run(
            ["ps", "-eo", "pid,ppid,cmd"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except Exception:
        return
    for line in output.splitlines():
        fields = line.split(None, 2)
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        pid, ppid, command = int(fields[0]), int(fields[1]), fields[2]
        if ppid != parent:
            continue
        if "ptxas" not in command and "compile_worker" not in command:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_text(command: list[str], cwd: Path | None = None) -> str:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except Exception as exc:
        return f"unavailable:{type(exc).__name__}"


def _hardware_assertions() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "0":
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must be exactly '0', found {visible!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if "H100" not in name or capability != (9, 0):
        raise RuntimeError(
            f"H100 sm90 required, found {name!r} with capability {capability}"
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "exactly one logical CUDA device must be visible; "
            f"found {torch.cuda.device_count()}"
        )


def collect_metadata(cache_bytes: int | None = None) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(0)
    status = _run_text(["git", "status", "--porcelain"], ROOT)
    driver = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
            "--id=0",
        ]
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "helion_commit": _run_text(["git", "rev-parse", "HEAD"], ROOT),
        "helion_commit_time": _run_text(
            ["git", "show", "-s", "--format=%cI", "HEAD"], ROOT
        ),
        "helion_dirty": bool(status),
        "helion_path": str(Path(helion.__file__).resolve()),
        "python": sys.version,
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "triton_version": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": driver.splitlines()[0] if driver else "unavailable",
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": "sm90",
        "gpu_total_memory_bytes": props.total_memory,
        "gpu_multiprocessor_count": props.multi_processor_count,
        "gpu_l2_cache_bytes": getattr(props, "L2_cache_size", None),
        "cache_clear_buffer_bytes": cache_bytes,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "helion_autotune_effort": os.environ.get("HELION_AUTOTUNE_EFFORT"),
        "timing": {
            "rounds": DEFAULT_ROUNDS,
            "noisy_rounds": NOISY_ROUNDS,
            "spread_threshold": SPREAD_THRESHOLD,
            "round_target_ms": ROUND_TARGET_MS,
            "min_reps": MIN_REPS,
            "max_reps": MAX_REPS,
            "fixed_reps": int(FIXED_REPS) if FIXED_REPS is not None else None,
        },
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return _jsonable(value.value)
    return repr(value)


def config_dict(config: Any) -> dict[str, Any] | None:
    if config is None:
        return None
    return _jsonable(dict(config.config))


def _extract_configs(workload: Workload) -> tuple[Any, Any, list[str], bool]:
    kernel_fn = workload.kernel_fn
    kernel_fn.reset()
    bound = kernel_fn.bind(workload.base_args)
    config_spec = bound.env.config_spec
    seed_configs = list(config_spec.compiler_seed_configs)
    fired = list(config_spec.autotuner_heuristics)
    with bound.env:
        default_config = config_spec._base_default_config()
    reduction_fired = any(name in REDUCTION_HEURISTICS for name in fired)
    seed_config = seed_configs[0] if seed_configs else None
    return seed_config, default_config, fired, reduction_fired


def _load_aot_module(kernel: str) -> Any:
    path = (
        ROOT
        / "pretuned_kernels"
        / kernel
        / f"_helion_aot_{kernel}_cuda_sm90.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"_h100_audit_aot_{kernel}", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import AOT selector {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_aot(workload: Workload) -> tuple[Any, dict[str, Any]]:
    module = _load_aot_module(workload.kernel)
    key_fn = getattr(module, f"key_{workload.kernel}")
    config_fn = getattr(module, f"autotune_{workload.kernel}")
    index = int(key_fn(*workload.base_args))
    selected_config = config_fn(*workload.base_args)
    keys = list(getattr(module, "_KEYS", []))
    selected_key = keys[index] if keys else None
    if workload.aot_key is None:
        exact = True
    else:
        exact = tuple(selected_key) == tuple(workload.aot_key)
    details = {
        "selector_index": index,
        "requested_key": list(workload.aot_key)
        if workload.aot_key is not None
        else list(workload.shape),
        "selected_tuned_key": list(selected_key)
        if selected_key is not None
        else None,
        "exact_tuned_key": exact,
        "tuning_sweep_member": exact,
        "selector_file": (
            f"pretuned_kernels/{workload.kernel}/"
            f"_helion_aot_{workload.kernel}_cuda_sm90.py"
        ),
    }
    return helion.Config(**selected_config), details


def _replay_kernel(kernel_fn: Any, config: Any) -> Any:
    settings = kernel_fn.settings
    return helion.kernel(
        kernel_fn.fn,
        config=config,
        static_shapes=settings.static_shapes,
        ignore_warnings=list(settings.ignore_warnings or []),
    )


def _accuracy(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    tolerances: dict[str, Tolerance],
) -> dict[str, Any]:
    if set(actual) != set(expected):
        return {
            "pass": False,
            "error": (
                f"output names differ: actual={sorted(actual)}, "
                f"expected={sorted(expected)}"
            ),
        }
    outputs: dict[str, Any] = {}
    passed = True
    for name in expected:
        tolerance = tolerances[name]
        actual_value = actual[name]
        expected_value = expected[name]
        actual_float = actual_value.detach().to(torch.float32)
        expected_float = expected_value.detach().to(torch.float32)
        difference = torch.abs(actual_float - expected_float)
        denominator = torch.clamp(torch.abs(expected_float), min=1e-12)
        if difference.numel():
            max_abs = float(torch.max(difference).item())
            max_rel = float(torch.max(difference / denominator).item())
            exact_fraction = float(
                torch.mean((actual_float == expected_float).to(torch.float32)).item()
            )
        else:
            max_abs = 0.0
            max_rel = 0.0
            exact_fraction = 1.0
        if tolerance.exact:
            output_passed = bool(torch.equal(actual_value, expected_value))
        else:
            output_passed = bool(
                torch.allclose(
                    actual_float,
                    expected_float,
                    rtol=tolerance.rtol,
                    atol=tolerance.atol,
                    equal_nan=False,
                )
            )
        outputs[name] = {
            "pass": output_passed,
            "actual_dtype": str(actual_value.dtype).removeprefix("torch."),
            "expected_dtype": str(expected_value.dtype).removeprefix("torch."),
            "shape": list(actual_value.shape),
            "rtol": tolerance.rtol,
            "atol": tolerance.atol,
            "exact_required": tolerance.exact,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "exact_fraction": exact_fraction,
        }
        passed = passed and output_passed
    return {"pass": passed, "outputs": outputs}


def _eager_reference(workload: Workload) -> tuple[dict[str, torch.Tensor], dict]:
    state = workload.make_state()
    state.restore()
    output = workload.reference_fn(*state.args)
    observed = workload.observe(output, state.args)
    description = {
        name: {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
        }
        for name, value in observed.items()
    }
    return observed, description


def _dynamo_explain(workload: Workload) -> dict[str, Any]:
    state = workload.make_state()
    state.restore()
    torch._dynamo.reset()
    try:
        with compile_timeout():
            explanation = torch._dynamo.explain(workload.reference_fn)(*state.args)
        return {
            "status": "ok",
            "graph_count": int(explanation.graph_count),
            "graph_break_count": int(explanation.graph_break_count),
            "op_count": int(explanation.op_count),
            "break_reasons": [str(reason) for reason in explanation.break_reasons],
        }
    except CompileTimeout as exc:
        _reap_compile_children()
        return {"status": "timeout", "error": str(exc)}
    except Exception as exc:
        return {
            "status": f"error:{type(exc).__name__}",
            "error": str(exc),
            "trace": traceback.format_exc(),
        }
    finally:
        torch._dynamo.reset()


@dataclass
class PreparedArm:
    name: str
    state: WorkloadState
    call: Callable[[], Any]
    accuracy: dict[str, Any]
    graph: torch.cuda.CUDAGraph | None = None
    graph_output: Any = None


def _prepare_helion_arm(
    name: str,
    workload: Workload,
    config: Any,
    expected: dict[str, torch.Tensor],
) -> tuple[PreparedArm | None, dict[str, Any]]:
    result: dict[str, Any] = {"config_present": config is not None}
    if config is None:
        result["status"] = "no-config"
        return None, result
    state = workload.make_state()
    try:
        fixed_kernel = _replay_kernel(workload.kernel_fn, config)
        state.restore()
        with compile_timeout():
            output = fixed_kernel(*state.args)
        observed = workload.observe(output, state.args)
        accuracy = _accuracy(observed, expected, workload.tolerances)
        result["accuracy"] = accuracy
        result["status"] = "compiled"
        prepared = PreparedArm(
            name,
            state,
            lambda kernel=fixed_kernel, arm_state=state: kernel(*arm_state.args),
            accuracy,
        )
        return prepared, result
    except CompileTimeout as exc:
        _reap_compile_children()
        result.update(status="compile-fail:timeout", error=str(exc))
    except Exception as exc:
        result.update(
            status=f"compile-fail:{type(exc).__name__}",
            error=str(exc),
            trace=traceback.format_exc(),
        )
    return None, result


def _prepare_torch_compile_arm(
    workload: Workload,
    expected: dict[str, torch.Tensor],
) -> tuple[PreparedArm | None, dict[str, Any], dict[str, Any]]:
    validation = _dynamo_explain(workload)
    result: dict[str, Any] = {}
    state = workload.make_state()
    try:
        torch._dynamo.reset()
        compiled = torch.compile(workload.reference_fn)
        state.restore()
        with compile_timeout():
            output = compiled(*state.args)
        observed = workload.observe(output, state.args)
        accuracy = _accuracy(observed, expected, workload.tolerances)
        result["accuracy"] = accuracy
        result["status"] = "compiled"
        prepared = PreparedArm(
            "torch_compile",
            state,
            lambda compiled_fn=compiled, arm_state=state: compiled_fn(*arm_state.args),
            accuracy,
        )
        return prepared, result, validation
    except CompileTimeout as exc:
        _reap_compile_children()
        result.update(status="compile-fail:timeout", error=str(exc))
    except Exception as exc:
        result.update(
            status=f"compile-fail:{type(exc).__name__}",
            error=str(exc),
            trace=traceback.format_exc(),
        )
    return None, result, validation


def _capture_arm(arm: PreparedArm, result: dict[str, Any]) -> bool:
    try:
        for _ in range(3):
            arm.state.restore()
            arm.call()
        torch.cuda.synchronize()
        arm.state.restore()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = arm.call()
        arm.graph = graph
        arm.graph_output = graph_output
        result["status"] = "ok"
        return True
    except Exception as exc:
        result.update(
            status=f"capture-fail:{type(exc).__name__}",
            error=str(exc),
            trace=traceback.format_exc(),
        )
        return False


def _spread(values: list[float]) -> float:
    median = statistics.median(values)
    if median <= 0:
        return 0.0
    return (max(values) - min(values)) / median


def _estimate_repetitions(
    arms: list[PreparedArm],
    clear_cache: Callable[[], None],
) -> int:
    if FIXED_REPS is not None:
        return max(MIN_REPS, min(MAX_REPS, int(FIXED_REPS)))
    probe_repetitions = 3
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(probe_repetitions):
        for arm in arms:
            arm.state.restore()
            clear_cache()
            assert arm.graph is not None
            arm.graph.replay()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1e3
    per_repetition_ms = elapsed_ms / probe_repetitions
    if per_repetition_ms <= 0:
        return MAX_REPS
    repetitions = int(ROUND_TARGET_MS / per_repetition_ms)
    return max(MIN_REPS, min(MAX_REPS, repetitions))


def _collect_round(
    arms: list[PreparedArm],
    order: list[int],
    repetitions: int,
    clear_cache: Callable[[], None],
) -> dict[str, float]:
    starts = {
        arm.name: [
            torch.cuda.Event(enable_timing=True) for _ in range(repetitions)
        ]
        for arm in arms
    }
    ends = {
        arm.name: [
            torch.cuda.Event(enable_timing=True) for _ in range(repetitions)
        ]
        for arm in arms
    }
    for repetition in range(repetitions):
        for index in order:
            arm = arms[index]
            arm.state.restore()
            clear_cache()
            starts[arm.name][repetition].record()
            assert arm.graph is not None
            arm.graph.replay()
            ends[arm.name][repetition].record()
    torch.cuda.synchronize()
    return {
        arm.name: statistics.median(
            starts[arm.name][index].elapsed_time(ends[arm.name][index]) * 1e3
            for index in range(repetitions)
        )
        for arm in arms
    }


def _time_captured_arms(
    arms: list[PreparedArm],
    arm_results: dict[str, dict[str, Any]],
) -> int:
    cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()

    def clear_cache() -> None:
        triton_runtime.driver.active.clear_cache(cache)

    for arm in arms:
        for _ in range(3):
            arm.state.restore()
            assert arm.graph is not None
            arm.graph.replay()
    torch.cuda.synchronize()
    repetitions = _estimate_repetitions(arms, clear_cache)
    samples = {arm.name: [] for arm in arms}
    round_index = 0
    while round_index < DEFAULT_ROUNDS:
        offset = round_index % len(arms)
        order = list(range(offset, len(arms))) + list(range(offset))
        medians = _collect_round(arms, order, repetitions, clear_cache)
        for name, value in medians.items():
            samples[name].append(value)
        round_index += 1
    if any(_spread(values) > SPREAD_THRESHOLD for values in samples.values()):
        while round_index < NOISY_ROUNDS:
            offset = round_index % len(arms)
            order = list(range(offset, len(arms))) + list(range(offset))
            medians = _collect_round(arms, order, repetitions, clear_cache)
            for name, value in medians.items():
                samples[name].append(value)
            round_index += 1
    for arm in arms:
        values = samples[arm.name]
        arm_results[arm.name]["timing"] = {
            "median_us": statistics.median(values),
            "round_medians_us": values,
            "repetitions_per_round": repetitions,
            "rounds": len(values),
            "relative_spread": _spread(values),
            "cold_l2": True,
            "cuda_graph": True,
        }
    return cache.numel() * cache.element_size()


def _valid_latency(arm: dict[str, Any]) -> float | None:
    if arm.get("status") != "ok":
        return None
    if not arm.get("accuracy", {}).get("pass"):
        return None
    return arm.get("timing", {}).get("median_us")


def _ratios(arms: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    seed = _valid_latency(arms.get("seed", {}))
    if not seed:
        return {"G_default": None, "G_tc": None, "G_aot": None}

    def ratio(name: str) -> float | None:
        latency = _valid_latency(arms.get(name, {}))
        return latency / seed if latency else None

    return {
        "G_default": ratio("default"),
        "G_tc": ratio("torch_compile"),
        "G_aot": ratio("aot_sm90"),
    }


def run_cell(
    kernel: str,
    shape: tuple[int, ...],
    base_metadata: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    workload = build_workload(kernel, shape)
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metadata": dict(base_metadata),
        "cohort": workload.cohort,
        "kernel": workload.kernel,
        "kernel_body": workload.body,
        "shape": list(workload.shape),
        "dtype": workload.dtype,
        "input_tensors": describe_inputs(workload),
        "optional_args": workload.optional_args,
        "tolerances": {
            name: {
                "rtol": tolerance.rtol,
                "atol": tolerance.atol,
                "exact": tolerance.exact,
            }
            for name, tolerance in workload.tolerances.items()
        },
        "methodology_changes": (
            [
                "torch reference emits every observable output from the Helion body; "
                "the historical report used loss-only references for some "
                "multi-output kernels"
            ]
            if workload.cohort == "original"
            and workload.kernel in {"jsd", "fused_linear_jsd", "grpo"}
            else []
        ),
    }
    expected, output_description = _eager_reference(workload)
    row["output_tensors"] = output_description

    seed_config, default_config, fired, reduction_fired = _extract_configs(workload)
    row["heuristic"] = {
        "fired_names": fired,
        "reduction_heuristic_fired": reduction_fired,
        "seed_config_present": seed_config is not None,
        "expected_h100_heuristics": sorted(REDUCTION_HEURISTICS),
    }
    configs: dict[str, Any] = {
        "seed": config_dict(seed_config),
        "default": config_dict(default_config),
    }
    aot_config = None
    if SPECS[kernel].has_aot:
        aot_config, aot_details = _select_aot(workload)
        row["aot_sm90"] = aot_details
        configs["aot_sm90"] = config_dict(aot_config)
    row["configs"] = configs

    arm_results: dict[str, dict[str, Any]] = {}
    prepared: list[PreparedArm] = []
    for name, config in (("seed", seed_config), ("default", default_config)):
        arm, result = _prepare_helion_arm(name, workload, config, expected)
        arm_results[name] = result
        if arm is not None:
            prepared.append(arm)

    torch_arm, torch_result, dynamo_validation = _prepare_torch_compile_arm(
        workload, expected
    )
    row["torch_compile_validation"] = dynamo_validation
    arm_results["torch_compile"] = torch_result
    if torch_arm is not None:
        prepared.append(torch_arm)

    if aot_config is not None:
        aot_arm, aot_result = _prepare_helion_arm(
            "aot_sm90", workload, aot_config, expected
        )
        arm_results["aot_sm90"] = aot_result
        if aot_arm is not None:
            prepared.append(aot_arm)

    captured = [
        arm for arm in prepared if _capture_arm(arm, arm_results[arm.name])
    ]
    cache_bytes = 0
    if captured:
        cache_bytes = _time_captured_arms(captured, arm_results)
    row["metadata"]["cache_clear_buffer_bytes"] = cache_bytes
    row["arms"] = arm_results
    row["ratios"] = _ratios(arm_results)
    row["wall_seconds"] = time.perf_counter() - started
    return row


def _shape_key(row: dict[str, Any]) -> tuple[int, ...]:
    return tuple(row["shape"])


def _existing_shapes(path: Path) -> set[tuple[int, ...]]:
    if not path.exists():
        return set()
    shapes = set()
    with path.open() as handle:
        for line in handle:
            if line.strip():
                shapes.add(_shape_key(json.loads(line)))
    return shapes


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_jsonable(row), sort_keys=True)
    with path.open("a") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _log_row(row: dict[str, Any]) -> None:
    if "error" in row:
        print(
            f"[error] {row['kernel']} {row['shape']}: {row['error']}",
            flush=True,
        )
        return
    arms = row["arms"]

    def latency(name: str) -> str:
        arm = arms.get(name, {})
        value = arm.get("timing", {}).get("median_us")
        return f"{value:.3f}" if value is not None else arm.get("status", "n/a")

    print(
        f"[cell] {row['kernel']} {row['shape']} "
        f"seed={latency('seed')}us default={latency('default')}us "
        f"tc={latency('torch_compile')}us aot={latency('aot_sm90')}us "
        f"ratios={row['ratios']} wall={row['wall_seconds']:.1f}s",
        flush=True,
    )


def _parse_shape(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.lower().replace(",", "x").split("x"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", required=True, choices=all_kernel_names())
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shape", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    _hardware_assertions()
    if not str(Path(helion.__file__).resolve()).startswith(str(ROOT) + os.sep):
        raise RuntimeError(
            f"imported Helion from {helion.__file__}, expected worktree {ROOT}"
        )

    spec = SPECS[args.kernel]
    if args.shape:
        requested = _parse_shape(args.shape)
        if requested not in spec.shapes:
            raise SystemExit(
                f"shape {requested} is not in the {args.kernel} audit matrix"
            )
        shapes = (requested,)
    elif args.smoke:
        shapes = (spec.shapes[0],)
    else:
        shapes = spec.shapes

    output_path = Path(args.out_dir) / f"{args.kernel}.jsonl"
    if not args.resume and output_path.exists():
        output_path.unlink()
    done = _existing_shapes(output_path) if args.resume else set()
    metadata = collect_metadata()
    print(
        f"kernel={args.kernel} cohort={spec.cohort} shapes={len(shapes)} "
        f"output={output_path} resume={args.resume}",
        flush=True,
    )

    for shape in shapes:
        if shape in done:
            print(f"[skip] {args.kernel} {shape}", flush=True)
            continue
        try:
            row = run_cell(args.kernel, shape, metadata)
        except torch.cuda.OutOfMemoryError as exc:
            row = {
                "schema_version": SCHEMA_VERSION,
                "metadata": metadata,
                "cohort": spec.cohort,
                "kernel": args.kernel,
                "shape": list(shape),
                "error": f"OutOfMemoryError: {exc}",
                "trace": traceback.format_exc(),
            }
        except Exception as exc:
            row = {
                "schema_version": SCHEMA_VERSION,
                "metadata": metadata,
                "cohort": spec.cohort,
                "kernel": args.kernel,
                "shape": list(shape),
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
            }
        _append_row(output_path, row)
        _log_row(row)
        gc.collect()
        torch.cuda.empty_cache()
        torch._dynamo.reset()


if __name__ == "__main__":
    main()
