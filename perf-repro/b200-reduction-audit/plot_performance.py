"""Plot B200 reduction-audit performance relative to the base default."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import matplotlib

matplotlib.use("Agg")

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


AUDIT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    color: str
    edge: str


@dataclass(frozen=True)
class Cohort:
    key: str
    title: str
    kernels: tuple[tuple[str, str], ...]


ARMS = (
    Arm("seed", "Reduction seed", "#0072B2", "#004E7A"),
    Arm("torch_compile", "torch.compile", "#E69F00", "#9B6900"),
    Arm("aot_sm100", "SM100 AOT", "#009E73", "#006B4E"),
)

COHORTS = (
    Cohort(
        "general_aot",
        "General AOT",
        (
            ("rms_norm", "RMSNorm"),
            ("layer_norm", "LayerNorm"),
            ("softmax", "Softmax"),
            ("cross_entropy", "Cross\nEntropy"),
        ),
    ),
    Cohort(
        "original",
        "Original Kernels",
        (
            ("kl_div", "KL Div"),
            ("jsd", "JSD"),
            ("fused_linear_jsd", "Fused Linear\nJSD"),
            ("grpo", "GRPO"),
            ("rms_norm_bwd", "RMSNorm\nBackward"),
            ("layer_norm_bwd", "LayerNorm\nBackward"),
        ),
    ),
    Cohort(
        "vllm",
        "vLLM",
        (
            (
                "dynamic_per_token_scaled_fp8_quant",
                "Dynamic per-token\nFP8 quant",
            ),
            (
                "per_token_group_fp8_quant",
                "Per-token group\nFP8 quant",
            ),
            (
                "rms_norm_dynamic_per_token_quant",
                "RMSNorm + dynamic\nper-token quant",
            ),
            (
                "rms_norm_per_block_quant",
                "RMSNorm +\nper-block quant",
            ),
            (
                "silu_and_mul_per_block_quant",
                "SiLU + multiply\nper-block quant",
            ),
            (
                "fused_qk_norm_rope",
                "Fused QK norm\n+ RoPE",
            ),
        ),
    ),
)


def _geomean(values: list[float]) -> float | None:
    valid = [value for value in values if value > 0 and math.isfinite(value)]
    if not valid:
        return None
    return math.exp(sum(math.log(value) for value in valid) / len(valid))


def _arm_us(row: dict[str, Any], arm: str) -> float | None:
    data = row.get("arms", {}).get(arm, {})
    if data.get("status") != "ok" or data.get("accuracy") is False:
        return None
    value = data.get("cold_l2_graph_us")
    return float(value) if value is not None else None


def _relative_performance(row: dict[str, Any], arm: str) -> float | None:
    default_us = _arm_us(row, "default")
    arm_us = _arm_us(row, arm)
    if default_us is None or arm_us is None or arm_us <= 0:
        return None
    return default_us / arm_us


def _load_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows = [json.loads(path.read_text()) for path in sorted(raw_dir.glob("*.json"))]
    if not rows:
        raise RuntimeError(f"no raw JSON rows found in {raw_dir}")
    return rows


def _cohort_values(
    rows: list[dict[str, Any]],
    cohort: Cohort,
) -> tuple[list[str], dict[str, list[float | None]]]:
    cohort_rows = [row for row in rows if row.get("cohort") == cohort.key]
    labels = [label for _, label in cohort.kernels] + ["Overall"]
    values: dict[str, list[float | None]] = {}
    for arm in ARMS:
        arm_values: list[float | None] = []
        for kernel, _label in cohort.kernels:
            samples = [
                value
                for row in cohort_rows
                if row.get("kernel") == kernel
                if (value := _relative_performance(row, arm.key)) is not None
            ]
            arm_values.append(_geomean(samples))
        overall_samples = [
            value
            for row in cohort_rows
            if (value := _relative_performance(row, arm.key)) is not None
        ]
        arm_values.append(_geomean(overall_samples))
        values[arm.key] = arm_values
    return labels, values


def _active_arms(values: dict[str, list[float | None]]) -> list[Arm]:
    return [arm for arm in ARMS if any(value is not None for value in values[arm.key])]


def _legend_handles(arms: list[Arm]) -> list[Patch | Line2D]:
    handles: list[Patch | Line2D] = [
        Patch(
            facecolor=arm.color,
            edgecolor=arm.edge,
            linewidth=0.8,
            label=arm.label,
        )
        for arm in arms
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="#343A40",
            linewidth=1.7,
            linestyle=(0, (5, 3)),
            label="Default (1.00x)",
        )
    )
    return handles


def _value_label(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}x"
    return f"{value:.2f}x"


def _plot_cohort(
    ax: Axes,
    rows: list[dict[str, Any]],
    cohort: Cohort,
    *,
    title_size: float,
    tick_size: float,
) -> list[Arm]:
    labels, values = _cohort_values(rows, cohort)
    arms = _active_arms(values)
    x = np.arange(len(labels), dtype=float)
    group_width = 0.74
    bar_width = group_width / len(arms)
    offsets = (np.arange(len(arms), dtype=float) - (len(arms) - 1) / 2.0) * bar_width

    all_values = [
        value for arm in arms for value in values[arm.key] if value is not None
    ]
    max_value = max(all_values, default=1.0)
    upper = max(1.3, max_value * 1.16)

    ax.axvspan(
        x[-1] - 0.58,
        x[-1] + 0.58,
        color="#F0F2F4",
        zorder=0,
    )
    ax.axvline(
        x[-1] - 0.67,
        color="#C8CDD2",
        linewidth=0.9,
        zorder=1,
    )

    for arm_index, arm in enumerate(arms):
        arm_values = values[arm.key]
        numeric = [np.nan if value is None else value for value in arm_values]
        bars = ax.bar(
            x + offsets[arm_index],
            numeric,
            width=bar_width * 0.88,
            color=arm.color,
            edgecolor=arm.edge,
            linewidth=0.8,
            zorder=3,
        )
        for index, (bar, value) in enumerate(zip(bars, arm_values, strict=True)):
            if value is None:
                continue
            if index == len(labels) - 1:
                bar.set_hatch("//")
            ax.annotate(
                _value_label(value),
                xy=(bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=max(7.5, tick_size - 1.5),
                fontweight="semibold",
                color="#212529",
                clip_on=False,
                zorder=5,
            )

    ax.axhline(
        1.0,
        color="#343A40",
        linewidth=1.7,
        linestyle=(0, (5, 3)),
        zorder=4,
    )

    ax.set_title(
        cohort.title,
        loc="left",
        fontsize=title_size,
        fontweight="bold",
        color="#17202A",
        pad=14,
    )
    ax.set_ylabel(
        "Performance relative to default",
        fontsize=tick_size,
        color="#343A40",
    )
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", labelsize=tick_size, length=0, pad=9)
    ax.tick_params(axis="y", labelsize=tick_size - 1, colors="#495057")
    ax.set_xlim(-0.62, x[-1] + 0.62)
    ax.set_ylim(0, upper)
    ax.yaxis.grid(True, color="#DDE1E5", linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#AEB4BA")
        ax.spines[spine].set_linewidth(0.8)

    tick_labels = ax.get_xticklabels()
    tick_labels[-1].set_fontweight("bold")
    tick_labels[-1].set_color("#17202A")
    return arms


def _save_separate(
    rows: list[dict[str, Any]],
    cohort: Cohort,
    output: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.3), facecolor="white")
    arms = _plot_cohort(
        ax,
        rows,
        cohort,
        title_size=20,
        tick_size=10.5,
    )
    fig.text(
        0.075,
        0.94,
        "Kernel bars are geometric means across shapes; Overall is the "
        "cell-weighted cohort geomean. Higher is faster.",
        ha="left",
        va="top",
        fontsize=10.5,
        color="#495057",
    )
    fig.legend(
        handles=_legend_handles(arms),
        loc="upper left",
        bbox_to_anchor=(0.068, 0.895),
        ncol=4,
        frameon=False,
        fontsize=10.5,
        handlelength=1.8,
        columnspacing=1.7,
    )
    fig.text(
        0.99,
        0.015,
        "Cold-L2 CUDA-graph latency | NVIDIA B200 | default latency / arm latency",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#6C757D",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.79, bottom=0.19)
    fig.savefig(
        output,
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
        metadata={"Title": f"{cohort.title} performance relative to default"},
    )
    plt.close(fig)


def _save_combined(
    rows: list[dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        len(COHORTS),
        1,
        figsize=(15.5, 20.5),
        facecolor="white",
    )
    fig.suptitle(
        "B200 Reduction Audit: Performance Relative to Default",
        x=0.06,
        y=0.988,
        ha="left",
        va="top",
        fontsize=23,
        fontweight="bold",
        color="#17202A",
    )
    fig.text(
        0.06,
        0.965,
        "Kernel bars are geometric means across shapes; Overall is the "
        "cell-weighted cohort geomean. Higher is faster.",
        ha="left",
        va="top",
        fontsize=11,
        color="#495057",
    )
    fig.legend(
        handles=_legend_handles(list(ARMS)),
        loc="upper left",
        bbox_to_anchor=(0.054, 0.948),
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=1.8,
        columnspacing=1.8,
    )

    for ax, cohort in zip(axes, COHORTS, strict=True):
        _plot_cohort(
            ax,
            rows,
            cohort,
            title_size=18,
            tick_size=10,
        )

    fig.text(
        0.99,
        0.01,
        "Cold-L2 CUDA-graph latency | NVIDIA B200 | default latency / arm latency",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#6C757D",
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.91,
        bottom=0.05,
        hspace=0.43,
    )
    fig.savefig(
        output,
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
        metadata={"Title": "B200 reduction audit performance relative to default"},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=AUDIT_DIR / "results" / "full" / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AUDIT_DIR / "results" / "charts",
    )
    args = parser.parse_args()

    rows = _load_rows(args.raw_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for cohort in COHORTS:
        output = args.output_dir / f"{cohort.key}_relative_performance.png"
        _save_separate(rows, cohort, output)
        outputs.append(output)

    combined = args.output_dir / "all_cohorts_relative_performance.png"
    _save_combined(rows, combined)
    outputs.append(combined)

    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
