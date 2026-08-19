"""Aggregate raw B200 reduction-audit rows into JSON and Markdown."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

from matrix import SPECS

AUDIT_DIR = Path(__file__).resolve().parent


def _geomean(values: list[float]) -> float | None:
    valid = [value for value in values if value > 0 and math.isfinite(value)]
    if not valid:
        return None
    return math.exp(sum(math.log(value) for value in valid) / len(valid))


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _arm_us(row: dict[str, Any], arm: str) -> float | None:
    data = row.get("arms", {}).get(arm, {})
    if data.get("accuracy") is False:
        return None
    value = data.get("cold_l2_graph_us")
    return float(value) if value is not None else None


def _ratio(row: dict[str, Any], arm: str) -> float | None:
    seed = _arm_us(row, "seed")
    other = _arm_us(row, arm)
    if seed is None or other is None or seed <= 0:
        return None
    return other / seed


def _load_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((results_dir / "raw").glob("*.json")):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            row = {"source_file": str(path), "fatal_error": str(exc)}
        row["source_file"] = str(path.relative_to(results_dir))
        rows.append(row)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected = sum(len(spec.shapes) for spec in SPECS)
    by_kernel: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if "kernel" in row:
            by_kernel[row["kernel"]].append(row)

    kernels = {}
    for spec in SPECS:
        kernel_rows = by_kernel.get(spec.kernel, [])
        ratios = {
            arm: [
                value for row in kernel_rows if (value := _ratio(row, arm)) is not None
            ]
            for arm in ("default", "torch_compile", "aot_sm100")
        }
        kernels[spec.kernel] = {
            "cohort": spec.cohort,
            "dtype": spec.dtype,
            "expected_cells": len(spec.shapes),
            "recorded_cells": len(kernel_rows),
            "G_default": _geomean(ratios["default"]),
            "G_torch_compile": _geomean(ratios["torch_compile"]),
            "G_aot_sm100": _geomean(ratios["aot_sm100"]),
            "n_default": len(ratios["default"]),
            "n_torch_compile": len(ratios["torch_compile"]),
            "n_aot_sm100": len(ratios["aot_sm100"]),
        }

    cohorts = {}
    for cohort in ("general_aot", "original", "vllm"):
        cohort_rows = [row for row in rows if row.get("cohort") == cohort]
        cohorts[cohort] = {}
        for arm in ("default", "torch_compile", "aot_sm100"):
            values = [
                value for row in cohort_rows if (value := _ratio(row, arm)) is not None
            ]
            cohorts[cohort][f"G_{arm}"] = _geomean(values)
            cohorts[cohort][f"n_{arm}"] = len(values)

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
                        "detail": data.get("error")
                        or data.get("capture_error")
                        or data.get("accuracy_detail"),
                    }
                )

    return {
        "expected_cells": expected,
        "recorded_cells": len(rows),
        "kernels": kernels,
        "cohorts": cohorts,
        "failures": failures,
        "rows": rows,
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# B200 Reduction-Heuristic Audit Results",
        "",
        (
            f"Recorded {summary['recorded_cells']} of "
            f"{summary['expected_cells']} planned cells. Ratios above 1 mean the "
            "reduction seed is faster."
        ),
        "",
        "## Cohorts",
        "",
        "| Cohort | vs default | vs torch.compile | vs SM100 AOT |",
        "|---|---:|---:|---:|",
    ]
    for name, values in summary["cohorts"].items():
        lines.append(
            f"| `{name}` | {_fmt(values['G_default'])} "
            f"(n={values['n_default']}) | "
            f"{_fmt(values['G_torch_compile'])} "
            f"(n={values['n_torch_compile']}) | "
            f"{_fmt(values['G_aot_sm100'])} "
            f"(n={values['n_aot_sm100']}) |"
        )

    lines.extend(
        [
            "",
            "## Kernels",
            "",
            "| Kernel | Dtype | Cells | vs default | vs torch.compile | vs SM100 AOT |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, values in summary["kernels"].items():
        lines.append(
            f"| `{name}` | {values['dtype']} | "
            f"{values['recorded_cells']}/{values['expected_cells']} | "
            f"{_fmt(values['G_default'])} (n={values['n_default']}) | "
            f"{_fmt(values['G_torch_compile'])} "
            f"(n={values['n_torch_compile']}) | "
            f"{_fmt(values['G_aot_sm100'])} "
            f"(n={values['n_aot_sm100']}) |"
        )

    lines.extend(
        [
            "",
            "## Per Shape",
            "",
            "| Kernel | Shape | seed us | default/seed | tc/seed | AOT/seed |",
            "|---|---|---:|---:|---:|---:|",
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
        lines.append(
            f"| `{row.get('kernel', '?')}` | `{row.get('shape')}` | "
            f"{_fmt(_arm_us(row, 'seed'))} | "
            f"{_fmt(_ratio(row, 'default'))} | "
            f"{_fmt(_ratio(row, 'torch_compile'))} | "
            f"{_fmt(_ratio(row, 'aot_sm100'))} |"
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
