"""Aggregate raw B200 reduction-audit rows into JSON and Markdown."""

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
        ratio_stats = {arm: _ratio_stats(values) for arm, values in ratios.items()}
        kernels[spec.kernel] = {
            "cohort": spec.cohort,
            "dtype": spec.dtype,
            "expected_cells": len(spec.shapes),
            "recorded_cells": len(kernel_rows),
            "ratios": ratio_stats,
            "G_default": ratio_stats["default"]["geomean"],
            "G_torch_compile": ratio_stats["torch_compile"]["geomean"],
            "G_aot_sm100": ratio_stats["aot_sm100"]["geomean"],
            "n_default": ratio_stats["default"]["n"],
            "n_torch_compile": ratio_stats["torch_compile"]["n"],
            "n_aot_sm100": ratio_stats["aot_sm100"]["n"],
        }

    cohorts = {}
    for cohort in ("general_aot", "original", "vllm"):
        cohort_rows = [row for row in rows if row.get("cohort") == cohort]
        cohorts[cohort] = {}
        for arm in ("default", "torch_compile", "aot_sm100"):
            values = [
                value for row in cohort_rows if (value := _ratio(row, arm)) is not None
            ]
            stats = _ratio_stats(values)
            cohorts[cohort][arm] = stats
            cohorts[cohort][f"G_{arm}"] = stats["geomean"]
            cohorts[cohort][f"n_{arm}"] = stats["n"]

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

    base_environment = rows[0].get("environment", {}) if rows else {}
    environment_consistent = all(
        row.get("environment", {}) == base_environment for row in rows
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
        for row in rows
        if not row.get("reduction_seed_present")
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

    return {
        "expected_cells": expected,
        "recorded_cells": len(rows),
        "environment": base_environment,
        "environment_consistent": environment_consistent,
        "heuristic_counts": dict(sorted(heuristic_counts.items())),
        "heuristic_no_seed": no_seed,
        "kernels": kernels,
        "cohorts": cohorts,
        "failures": failures,
        "high_spread": high_spread,
        "rows": rows,
    }


def _compact_config(row: dict[str, Any], arm: str) -> str:
    config = row.get("arms", {}).get(arm, {}).get("config")
    if not config:
        return "n/a"
    block_sizes = config.get("block_sizes")
    num_warps = config.get("num_warps")
    num_stages = config.get("num_stages")
    return f"bs={block_sizes};w={num_warps};s={num_stages}"


def _markdown(summary: dict[str, Any]) -> str:
    environment = summary["environment"]
    lines = [
        "# B200 Reduction-Heuristic Audit Results",
        "",
        (
            f"Recorded {summary['recorded_cells']} of "
            f"{summary['expected_cells']} planned cells. Ratios above 1 mean the "
            "reduction seed is faster."
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
            f"- CUDA runtime / driver: "
            f"`{environment.get('cuda_runtime', 'unknown')}` / "
            f"`{environment.get('cuda_driver', 'unknown')}`"
        ),
        (
            f"- GPU: physical "
            f"`{environment.get('physical_gpu_index', 'unknown')}`, "
            f"logical `{environment.get('logical_device', 'unknown')}`, "
            f"`{environment.get('gpu_name', 'unknown')}` "
            f"(SM{''.join(str(value) for value in environment.get('compute_capability', []))})"
        ),
        (
            f"- `CUDA_VISIBLE_DEVICES="
            f"{environment.get('cuda_visible_devices', 'unknown')}`; "
            f"metadata identical across cells: "
            f"`{summary['environment_consistent']}`"
        ),
        "",
        "## Heuristic Coverage",
        "",
    ]
    for name, count in summary["heuristic_counts"].items():
        lines.append(f"- `{name}`: {count} cells")
    if summary["heuristic_no_seed"]:
        lines.append(
            f"- Missing reduction seed: {len(summary['heuristic_no_seed'])} cells"
        )
    else:
        lines.append(
            f"- Reduction seed present: {summary['recorded_cells']}/"
            f"{summary['recorded_cells']} cells"
        )

    lines.extend(
        [
            "",
            "## Cohorts",
            "",
            "Entries are geomean [min, max] (count).",
            "",
            "| Cohort | vs default | vs torch.compile | vs SM100 AOT |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in summary["cohorts"].items():
        lines.append(
            f"| `{name}` | {_fmt_ratio_stats(values['default'])} | "
            f"{_fmt_ratio_stats(values['torch_compile'])} | "
            f"{_fmt_ratio_stats(values['aot_sm100'])} |"
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
            f"{_fmt_ratio_stats(values['ratios']['default'])} | "
            f"{_fmt_ratio_stats(values['ratios']['torch_compile'])} | "
            f"{_fmt_ratio_stats(values['ratios']['aot_sm100'])} |"
        )

    lines.extend(
        [
            "",
            "## Per Shape",
            "",
            (
                "Configs show `block_sizes`, `num_warps`, and `num_stages`; "
                "raw JSON contains each complete config."
            ),
            "",
            (
                "| Kernel | Shape | seed config | default config | AOT config | "
                "seed us | default/seed | tc/seed | AOT/seed |"
            ),
            "|---|---|---|---|---|---:|---:|---:|---:|",
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
            f"`{_compact_config(row, 'seed')}` | "
            f"`{_compact_config(row, 'default')}` | "
            f"`{_compact_config(row, 'aot_sm100')}` | "
            f"{_fmt(_arm_us(row, 'seed'))} | "
            f"{_fmt(_ratio(row, 'default'))} | "
            f"{_fmt(_ratio(row, 'torch_compile'))} | "
            f"{_fmt(_ratio(row, 'aot_sm100'))} |"
        )

    lines.extend(["", "## Timing Spread Above 5%", ""])
    if not summary["high_spread"]:
        lines.append("None.")
    else:
        lines.extend(
            [
                (
                    "These cells already used the 15-round escalation. Very "
                    "short kernels are especially sensitive to event-timing "
                    "granularity."
                ),
                "",
                "| Kernel | Shape | Arm | Latency us | Spread | Rounds |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for item in summary["high_spread"]:
            lines.append(
                f"| `{item['kernel']}` | `{item['shape']}` | `{item['arm']}` | "
                f"{_fmt(item['latency_us'])} | {item['spread']:.1%} | "
                f"{item['rounds']} |"
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
