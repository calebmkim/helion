"""Aggregate raw B200 pointwise-audit rows into JSON and Markdown."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import json
import math
from operator import itemgetter
from pathlib import Path
from typing import Any

from matrix import SPECS

AUDIT_DIR = Path(__file__).resolve().parent
ARM_KEYS = ("seed", "torch_compile", "aot")
COHORT_KEYS = ("general", "vllm", "sglang")


def _geomean(values: list[float]) -> float | None:
    valid = [value for value in values if value > 0 and math.isfinite(value)]
    if not valid:
        return None
    return math.exp(sum(math.log(value) for value in valid) / len(valid))


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _ratio_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "geomean": _geomean(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "n": len(values),
    }


def _fmt_ratio_stats(values: dict[str, Any]) -> str:
    if values["geomean"] is None:
        return "n/a (n=0)"
    return (
        f"{_fmt(values['geomean'])} "
        f"[{_fmt(values['min'])}, {_fmt(values['max'])}] "
        f"(n={values['n']})"
    )


def _arm_us(row: dict[str, Any], arm: str) -> float | None:
    data = row.get("arms", {}).get(arm, {})
    if data.get("status") != "ok" or data.get("accuracy") is False:
        return None
    value = data.get("cold_l2_graph_us")
    return float(value) if value is not None else None


def _relative_performance(row: dict[str, Any], arm: str) -> float | None:
    default = _arm_us(row, "default")
    other = _arm_us(row, arm)
    if default is None or other is None or other <= 0:
        return None
    return default / other


def _load_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((results_dir / "raw").glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            row = {"fatal_error": str(exc)}
        row["source_file"] = str(path.relative_to(results_dir))
        rows.append(row)
    return rows


def _rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        arm: _ratio_stats(
            [
                value
                for row in rows
                if (value := _relative_performance(row, arm)) is not None
            ]
        )
        for arm in ARM_KEYS
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = sum(len(spec.shapes) for spec in SPECS)
    measured_rows = [row for row in rows if not row.get("fatal_error")]
    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "kernel" in row:
            by_kernel[row["kernel"]].append(row)

    kernels = {}
    for spec in SPECS:
        kernel_rows = by_kernel.get(spec.kernel, [])
        kernels[spec.kernel] = {
            "cohort": spec.cohort,
            "dtype": spec.dtype,
            "has_aot": spec.has_aot,
            "aot_arch": spec.aot_arch,
            "expected_cells": len(spec.shapes),
            "recorded_cells": len(kernel_rows),
            "relative_performance": _rollup(kernel_rows),
        }

    cohorts = {}
    for cohort in COHORT_KEYS:
        cohort_rows = [row for row in rows if row.get("cohort") == cohort]
        cohorts[cohort] = {
            "recorded_cells": len(cohort_rows),
            "relative_performance": _rollup(cohort_rows),
        }

    failures = []
    for row in rows:
        if row.get("fatal_error"):
            failures.append(
                {
                    "kernel": row.get("kernel"),
                    "shape": row.get("shape"),
                    "arm": "cell",
                    "failure": row["fatal_error"],
                }
            )
        for arm, data in row.get("arms", {}).items():
            status = data.get("status", "")
            if status != "ok":
                failures.append(
                    {
                        "kernel": row.get("kernel"),
                        "shape": row.get("shape"),
                        "arm": arm,
                        "failure": status,
                        "detail": data.get("error") or data.get("accuracy_detail"),
                    }
                )

    base_environment = rows[0].get("environment", {}) if rows else {}
    environment_consistent = all(
        row.get("environment", {}) == base_environment
        for row in rows
        if row.get("environment")
    )
    heuristic_counts = Counter(
        heuristic for row in rows for heuristic in row.get("fired_heuristics", [])
    )
    no_seed = [
        {
            "kernel": row.get("kernel"),
            "shape": row.get("shape"),
            "fired_heuristics": row.get("fired_heuristics", []),
        }
        for row in measured_rows
        if not row.get("pointwise_seed_present")
    ]
    high_spread = []
    for row in rows:
        for arm, data in row.get("arms", {}).items():
            spread = data.get("spread")
            if spread is not None and float(spread) > 0.05:
                high_spread.append(
                    {
                        "kernel": row.get("kernel"),
                        "shape": row.get("shape"),
                        "arm": arm,
                        "latency_us": data.get("cold_l2_graph_us"),
                        "spread": float(spread),
                        "rounds": len(data.get("round_medians_us", [])),
                    }
                )
    high_spread.sort(key=itemgetter("spread"), reverse=True)
    high_null_delta = [
        {
            "kernel": row.get("kernel"),
            "shape": row.get("shape"),
            "delta": row["default_null_delta"],
        }
        for row in rows
        if row.get("default_null_delta") is not None
        and float(row["default_null_delta"]) > 0.03
    ]
    high_null_delta.sort(key=itemgetter("delta"), reverse=True)

    return {
        "expected_cells": expected,
        "recorded_cells": len(rows),
        "measured_cells": len(measured_rows),
        "environment": base_environment,
        "environment_consistent": environment_consistent,
        "heuristic_counts": dict(sorted(heuristic_counts.items())),
        "heuristic_no_seed": no_seed,
        "kernels": kernels,
        "cohorts": cohorts,
        "overall": _rollup(rows),
        "failures": failures,
        "high_spread": high_spread,
        "high_default_null_delta": high_null_delta,
        "rows": rows,
    }


def _compact_config(row: dict[str, Any], arm: str) -> str:
    config = row.get("arms", {}).get(arm, {}).get("config")
    if not config:
        return "n/a"
    return (
        f"bs={config.get('block_sizes')};"
        f"w={config.get('num_warps')};"
        f"s={config.get('num_stages')}"
    )


def _shape_text(row: dict[str, Any]) -> str:
    return "(" + ",".join(str(value) for value in row.get("shape", [])) + ")"


def _markdown(summary: dict[str, Any]) -> str:
    environment = summary["environment"]
    lines = [
        "# B200 Pointwise-Heuristic Audit Results",
        "",
        (
            f"Recorded {summary['recorded_cells']} of "
            f"{summary['expected_cells']} planned cells; "
            f"{summary['measured_cells']} produced timings. The default arm is "
            "`1.00x`; relative performance above one is faster."
        ),
        "",
        "## Environment",
        "",
        (
            f"- Helion: `{environment.get('helion_commit', 'unknown')}` "
            f"(`{environment.get('helion_branch', 'unknown')}`)"
        ),
        (
            f"- PyTorch: `{environment.get('torch_version', 'unknown')}` "
            f"(`{environment.get('torch_git_version', 'unknown')}`)"
        ),
        f"- Triton: `{environment.get('triton_version', 'unknown')}`",
        (
            "- CUDA runtime / driver: "
            f"`{environment.get('cuda_runtime', 'unknown')}` / "
            f"`{environment.get('cuda_driver', 'unknown')}`"
        ),
        (
            f"- GPU: physical `{environment.get('physical_gpu_index', 'unknown')}`, "
            f"`{environment.get('gpu_name', 'unknown')}`; "
            f"`CUDA_VISIBLE_DEVICES={environment.get('cuda_visible_devices', 'unknown')}`"
        ),
        (
            "- Calibrated cold-L2 CUDA graphs, flush inside the timed graph, "
            "nine round medians with fifteen-round high-spread escalation."
        ),
        "",
        "## AOT Provenance",
        "",
        "- RoPE and vLLM `silu_mul_fp8`: checked-in SM90 tables selected on B200.",
        "- SGLang `silu_and_mul_interleaved`: checked-in SM100 table.",
        "- SwiGLU and GEGLU have no AOT arm.",
        "",
        "## Findings",
        "",
        (
            "- General pointwise: the seed is "
            f"`{_fmt(summary['cohorts']['general']['relative_performance']['seed']['geomean'])}x` "
            "default and `torch.compile` is "
            f"`{_fmt(summary['cohorts']['general']['relative_performance']['torch_compile']['geomean'])}x`. "
            "The large gains come from replacing the tiny unseeded base configs. "
            "The RoPE-only SM90 AOT result is "
            f"`{_fmt(summary['cohorts']['general']['relative_performance']['aot']['geomean'])}x`."
        ),
        (
            "- vLLM `silu_mul_fp8`: seed "
            f"`{_fmt(summary['cohorts']['vllm']['relative_performance']['seed']['geomean'])}x`, "
            "`torch.compile` "
            f"`{_fmt(summary['cohorts']['vllm']['relative_performance']['torch_compile']['geomean'])}x`, "
            "and SM90 AOT "
            f"`{_fmt(summary['cohorts']['vllm']['relative_performance']['aot']['geomean'])}x` "
            "default."
        ),
        (
            "- SGLang `silu_and_mul_interleaved`: seed "
            f"`{_fmt(summary['cohorts']['sglang']['relative_performance']['seed']['geomean'])}x`, "
            "`torch.compile` "
            f"`{_fmt(summary['cohorts']['sglang']['relative_performance']['torch_compile']['geomean'])}x`, "
            "and SM100 AOT "
            f"`{_fmt(summary['cohorts']['sglang']['relative_performance']['aot']['geomean'])}x` "
            "default."
        ),
        (
            f"- {summary['measured_cells']} cells completed; the remaining planned "
            "RoPE cell reached the 300-second timeout."
        ),
        "",
        "## Coverage",
        "",
    ]
    for name, count in summary["heuristic_counts"].items():
        lines.append(f"- `{name}`: {count} cells")
    lines.append(
        f"- Pointwise seed present: "
        f"{summary['measured_cells'] - len(summary['heuristic_no_seed'])}/"
        f"{summary['measured_cells']} measured cells"
    )

    lines.extend(
        [
            "",
            "## Cohorts",
            "",
            "Entries are geomean [min, max] (count), relative to default.",
            "",
            "| Cohort | Seed | torch.compile | AOT |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, cohort in summary["cohorts"].items():
        values = cohort["relative_performance"]
        lines.append(
            f"| `{name}` | {_fmt_ratio_stats(values['seed'])} | "
            f"{_fmt_ratio_stats(values['torch_compile'])} | "
            f"{_fmt_ratio_stats(values['aot'])} |"
        )

    lines.extend(
        [
            "",
            "## Kernels",
            "",
            "| Kernel | Dtype | Cells | Seed | torch.compile | AOT |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, kernel in summary["kernels"].items():
        values = kernel["relative_performance"]
        aot_label = (
            f"{_fmt_ratio_stats(values['aot'])} ({kernel['aot_arch']})"
            if kernel["has_aot"]
            else "n/a"
        )
        lines.append(
            f"| `{name}` | {kernel['dtype']} | "
            f"{kernel['recorded_cells']}/{kernel['expected_cells']} | "
            f"{_fmt_ratio_stats(values['seed'])} | "
            f"{_fmt_ratio_stats(values['torch_compile'])} | {aot_label} |"
        )

    lines.extend(
        [
            "",
            "## Per Shape",
            "",
            (
                "Latency is calibrated cold-L2 device time. Configs show "
                "`block_sizes`, `num_warps`, and `num_stages`."
            ),
            "",
            (
                "| Kernel | Shape | Default config | Seed config | AOT config | "
                "Default us | Seed x | torch.compile x | AOT x | Null delta |"
            ),
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    ordered_rows = sorted(
        summary["rows"],
        key=lambda row: (
            row.get("cohort", ""),
            row.get("kernel", ""),
            row.get("shape_index", -1),
        ),
    )
    for row in ordered_rows:
        null_delta = row.get("default_null_delta")
        lines.append(
            f"| `{row.get('kernel', '?')}` | `{_shape_text(row)}` | "
            f"`{_compact_config(row, 'default')}` | "
            f"`{_compact_config(row, 'seed')}` | "
            f"`{_compact_config(row, 'aot')}` | "
            f"{_fmt(_arm_us(row, 'default'))} | "
            f"{_fmt(_relative_performance(row, 'seed'))} | "
            f"{_fmt(_relative_performance(row, 'torch_compile'))} | "
            f"{_fmt(_relative_performance(row, 'aot'))} | "
            f"{'n/a' if null_delta is None else f'{null_delta:.2%}'} |"
        )

    lines.extend(["", "## Measurement Flags", ""])
    if not summary["high_spread"] and not summary["high_default_null_delta"]:
        lines.append("No arm spread above 5% and no default/null delta above 3%.")
    else:
        lines.extend(
            [
                f"- Arms above 5% round spread: {len(summary['high_spread'])}",
                (
                    "- Cells above 3% default/null delta: "
                    f"{len(summary['high_default_null_delta'])}"
                ),
            ]
        )
        for item in summary["high_default_null_delta"]:
            lines.append(
                f"- `{item['kernel']}` `{item['shape']}` default/null "
                f"delta: {item['delta']:.2%}"
            )

    lines.extend(["", "## Failures", ""])
    if not summary["failures"]:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Kernel | Shape | Arm | Failure |",
                "|---|---|---|---|",
            ]
        )
        for failure in summary["failures"]:
            detail = failure.get("detail") or failure["failure"]
            detail = str(detail).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{failure.get('kernel')}` | `{failure.get('shape')}` | "
                f"`{failure['arm']}` | {detail[:300]} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(AUDIT_DIR / "results" / "full"),
    )
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    rows = _load_rows(results_dir)
    summary = _summary(rows)
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    (results_dir / "SUMMARY.md").write_text(_markdown(summary))
    print(f"wrote {results_dir / 'SUMMARY.md'} and {results_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
