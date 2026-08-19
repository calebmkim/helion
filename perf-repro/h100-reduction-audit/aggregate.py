#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from datetime import timezone
import json
import math
from pathlib import Path
from typing import Any

from workloads import all_kernel_names
from workloads import SPECS


COHORT_ORDER = {"general_aot": 0, "original": 1, "vllm": 2}


def _load_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{line_number}: {exc}") from exc
                row["_source"] = f"{path.name}:{line_number}"
                rows.append(row)
    kernel_index = {name: index for index, name in enumerate(all_kernel_names())}
    shape_index = {
        (name, shape): index
        for name, spec in SPECS.items()
        for index, shape in enumerate(spec.shapes)
    }
    rows.sort(
        key=lambda row: (
            COHORT_ORDER.get(row.get("cohort"), 99),
            kernel_index.get(row.get("kernel"), 999),
            shape_index.get(
                (row.get("kernel"), tuple(row.get("shape", []))), 999
            ),
        )
    )
    return rows


def _latency(row: dict[str, Any], arm_name: str) -> float | None:
    arm = row.get("arms", {}).get(arm_name, {})
    if arm.get("status") != "ok":
        return None
    if not arm.get("accuracy", {}).get("pass"):
        return None
    return arm.get("timing", {}).get("median_us")


def _ratio(row: dict[str, Any], arm_name: str) -> float | None:
    seed = _latency(row, "seed")
    comparison = _latency(row, arm_name)
    if not seed or not comparison:
        return None
    return comparison / seed


def _geomean(values: list[float]) -> float | None:
    usable = [value for value in values if value > 0 and math.isfinite(value)]
    if not usable:
        return None
    return math.exp(sum(math.log(value) for value in usable) / len(usable))


def _format_number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _format_latency(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _compact_config(config: dict[str, Any] | None) -> str:
    if not config:
        return "-"
    block = config.get("block_sizes")
    reduction = config.get("reduction_loops")
    warps = config.get("num_warps")
    pid = config.get("pid_type")
    return f"b={block};r={reduction};w={warps};pid={pid}"


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = {
        "G_default": [
            value
            for row in rows
            if (value := _ratio(row, "default")) is not None
        ],
        "G_tc": [
            value
            for row in rows
            if (value := _ratio(row, "torch_compile")) is not None
        ],
        "G_aot": [
            value
            for row in rows
            if (value := _ratio(row, "aot_sm90")) is not None
        ],
    }
    return {
        "cells": len(rows),
        "ratios": {
            name: {
                "valid_cells": len(values),
                "geomean": _geomean(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "seed_wins": sum(value > 1.0 for value in values),
            }
            for name, values in ratios.items()
        },
    }


def _failures(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        identity = {
            "kernel": row.get("kernel"),
            "shape": row.get("shape"),
            "source": row.get("_source"),
        }
        if row.get("error"):
            output["cell_errors"].append({**identity, "error": row["error"]})
            continue
        heuristic = row.get("heuristic", {})
        if not heuristic.get("reduction_heuristic_fired"):
            output["heuristic_no_fires"].append(
                {**identity, "fired_names": heuristic.get("fired_names")}
            )
        validation = row.get("torch_compile_validation", {})
        if validation.get("status") != "ok" or validation.get(
            "graph_break_count", 0
        ):
            output["torch_compile_graph_breaks"].append(
                {**identity, "validation": validation}
            )
        for arm_name, arm in row.get("arms", {}).items():
            if arm.get("status") != "ok":
                output["arm_failures"].append(
                    {
                        **identity,
                        "arm": arm_name,
                        "status": arm.get("status"),
                        "error": arm.get("error"),
                    }
                )
            elif not arm.get("accuracy", {}).get("pass"):
                output["accuracy_failures"].append(
                    {
                        **identity,
                        "arm": arm_name,
                        "accuracy": arm.get("accuracy"),
                    }
                )
            timing = arm.get("timing", {})
            if timing.get("relative_spread", 0.0) > 0.05:
                output["high_spread"].append(
                    {
                        **identity,
                        "arm": arm_name,
                        "relative_spread": timing["relative_spread"],
                    }
                )
    return dict(output)


def _report(
    rows: list[dict[str, Any]],
    cohort_summary: dict[str, Any],
    kernel_summary: dict[str, Any],
    failures: dict[str, list[dict[str, Any]]],
) -> str:
    lines = [
        "# H100 Reduction-Heuristic Audit",
        "",
        f"Generated solely from raw JSONL on {datetime.now(timezone.utc).isoformat()}.",
        "",
    ]
    metadata = next((row.get("metadata") for row in rows if row.get("metadata")), {})
    lines.extend(
        [
            "## Environment",
            "",
            f"- Helion: `{metadata.get('helion_commit', 'unknown')}`",
            f"- PyTorch: `{metadata.get('torch_version', 'unknown')}` "
            f"(`{metadata.get('torch_git_version', 'unknown')}`)",
            f"- Triton: `{metadata.get('triton_version', 'unknown')}`",
            f"- CUDA runtime/driver: `{metadata.get('cuda_runtime', 'unknown')}` / "
            f"`{metadata.get('cuda_driver', 'unknown')}`",
            f"- GPU: `{metadata.get('gpu_name', 'unknown')}` "
            f"(`{metadata.get('compute_capability', 'unknown')}`)",
            f"- Cells: `{len(rows)}`",
            "",
            "Ratios above one mean the H100 reduction seed is faster.",
            "",
            "## Cohort Summary",
            "",
            "| Cohort | Cells | G_default geo [min,max] | G_tc geo [min,max] | "
            "G_aot geo [min,max] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cohort in ("general_aot", "original", "vllm"):
        summary = cohort_summary.get(cohort, {"cells": 0, "ratios": {}})

        def ratio_cell(name: str) -> str:
            ratio = summary["ratios"].get(name, {})
            return (
                f"{_format_number(ratio.get('geomean'))} "
                f"[{_format_number(ratio.get('min'))},"
                f"{_format_number(ratio.get('max'))}] "
                f"({ratio.get('valid_cells', 0)} valid)"
            )

        lines.append(
            f"| {cohort} | {summary['cells']} | {ratio_cell('G_default')} | "
            f"{ratio_cell('G_tc')} | {ratio_cell('G_aot')} |"
        )

    lines.extend(
        [
            "",
            "## Per-Kernel Summary",
            "",
            "| Cohort | Kernel | Cells | G_default geo [min,max] | "
            "G_tc geo [min,max] | G_aot geo [min,max] |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for kernel in all_kernel_names():
        if kernel not in kernel_summary:
            continue
        summary = kernel_summary[kernel]

        def ratio_cell(name: str) -> str:
            ratio = summary["ratios"][name]
            return (
                f"{_format_number(ratio['geomean'])} "
                f"[{_format_number(ratio['min'])},{_format_number(ratio['max'])}]"
            )

        lines.append(
            f"| {SPECS[kernel].cohort} | {kernel} | {summary['cells']} | "
            f"{ratio_cell('G_default')} | {ratio_cell('G_tc')} | "
            f"{ratio_cell('G_aot')} |"
        )

    lines.extend(
        [
            "",
            "## Per-Shape Results",
            "",
            "| Cohort | Kernel | Shape | seed us | default us | tc us | aot_sm90 us | "
            "G_default | G_tc | G_aot | Seed config | AOT config |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        configs = row.get("configs", {})
        lines.append(
            f"| {row.get('cohort')} | {row.get('kernel')} | "
            f"`{tuple(row.get('shape', []))}` | "
            f"{_format_latency(_latency(row, 'seed'))} | "
            f"{_format_latency(_latency(row, 'default'))} | "
            f"{_format_latency(_latency(row, 'torch_compile'))} | "
            f"{_format_latency(_latency(row, 'aot_sm90'))} | "
            f"{_format_number(_ratio(row, 'default'))} | "
            f"{_format_number(_ratio(row, 'torch_compile'))} | "
            f"{_format_number(_ratio(row, 'aot_sm90'))} | "
            f"`{_compact_config(configs.get('seed'))}` | "
            f"`{_compact_config(configs.get('aot_sm90'))}` |"
        )

    lines.extend(["", "## Failures And Noise", ""])
    categories = (
        "cell_errors",
        "heuristic_no_fires",
        "torch_compile_graph_breaks",
        "arm_failures",
        "accuracy_failures",
        "high_spread",
    )
    for category in categories:
        entries = failures.get(category, [])
        lines.append(f"### {category.replace('_', ' ').title()} ({len(entries)})")
        lines.append("")
        if not entries:
            lines.append("None.")
        else:
            for entry in entries:
                lines.append(f"- `{json.dumps(entry, sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    rows = _load_rows(raw_dir)
    if not rows:
        raise SystemExit(f"no JSONL rows found in {raw_dir}")

    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cohort[row.get("cohort", "unknown")].append(row)
        by_kernel[row.get("kernel", "unknown")].append(row)
    cohort_summary = {
        cohort: _group_summary(group_rows)
        for cohort, group_rows in by_cohort.items()
    }
    kernel_summary = {
        kernel: _group_summary(group_rows)
        for kernel, group_rows in by_kernel.items()
    }
    failures = _failures(rows)
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "cells": len(rows),
        "cohorts": cohort_summary,
        "kernels": kernel_summary,
        "failures": failures,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "REPORT.md").write_text(
        _report(rows, cohort_summary, kernel_summary, failures)
    )
    print(
        f"aggregated {len(rows)} cells into {out_dir / 'summary.json'} "
        f"and {out_dir / 'REPORT.md'}"
    )


if __name__ == "__main__":
    main()
