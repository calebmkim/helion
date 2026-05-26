"""Signal-(a) driver for the reduction heuristic.

Builds two `@helion.kernel(config=...)` instances of the same kernel — one
hardcoded to `default_config()`, one hardcoded to the heuristic's seed —
and benches them head-to-head with `triton.testing.do_bench` (quantiles).

Single-process is load-bearing for low-noise comparisons of small-tensor
reductions, where wall-clock differences are O(1us) and cross-process
re-init/state shifts dwarf the signal.

    PYTHONPATH=$PWD python scripts/bench_reduction_heuristic.py --kernel sum
    PYTHONPATH=$PWD python scripts/bench_reduction_heuristic.py --kernel sum --validation
"""

from __future__ import annotations

import argparse
import math
import statistics
from typing import Callable

import torch
import triton.testing

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
import helion.language as hl

# Train/validation shapes from the plan §10.2. Tiny-M shapes (M < 16) are
# bucketed separately per §7.3 and not part of the headline geomean.
SHAPES: dict[str, list[tuple[int, int]]] = {
    "sum_train": [
        (2048, 1024),
        (2048, 4096),
        (2048, 16384),
        (4096, 1536),
        (4096, 5120),
        (8192, 256),
        (8192, 4096),
        (32768, 256),
        (32768, 1024),
    ],
    "sum_validation": [
        (16, 4096),  # tiny-M, launch-bound
        (2048, 1023),
        (4096, 6144),
        (512, 65536),
    ],
}
# Shapes excluded from headline geomean per §7.3 (launch-bound).
LAUNCH_BOUND = {(16, 4096)}


KernelFn = Callable[[torch.Tensor], torch.Tensor]


def _build_sum_kernels(
    default_cfg: helion.Config, seed_cfg: helion.Config
) -> tuple[KernelFn, KernelFn]:
    @helion.kernel(config=default_cfg, static_shapes=True)
    def k_default(x: torch.Tensor) -> torch.Tensor:
        m, _ = x.shape
        out = torch.empty([m], dtype=x.dtype, device=x.device)
        for tile_m in hl.tile(m):
            out[tile_m] = x[tile_m, :].sum(-1)
        return out

    @helion.kernel(config=seed_cfg, static_shapes=True)
    def k_seed(x: torch.Tensor) -> torch.Tensor:
        m, _ = x.shape
        out = torch.empty([m], dtype=x.dtype, device=x.device)
        for tile_m in hl.tile(m):
            out[tile_m] = x[tile_m, :].sum(-1)
        return out

    return k_default, k_seed


def _resolve_configs(
    kernel_name: str, shape: tuple[int, int]
) -> tuple[helion.Config, helion.Config, str]:
    """Bind a probe kernel for this shape, then read default + heuristic seed."""
    x = torch.empty(*shape, device="cuda", dtype=torch.float32)
    if kernel_name == "sum":

        @helion.kernel(static_shapes=True)
        def probe(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.shape
            out = torch.empty([m], dtype=x.dtype, device=x.device)
            for tile_m in hl.tile(m):
                out[tile_m] = x[tile_m, :].sum(-1)
            return out
    else:
        raise SystemExit(f"unknown kernel: {kernel_name}")

    bound = probe.bind((x,))
    default_cfg = bound.config_spec.default_config()
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    if not seeds:
        return default_cfg, default_cfg, "no-seed"
    return default_cfg, seeds[0], bound.config_spec.autotuner_heuristics[0]


def _bench_kernel(fn: Callable[[], torch.Tensor]) -> tuple[float, float, float]:
    fn()
    torch.cuda.synchronize()
    qs = triton.testing.do_bench(fn, warmup=200, rep=2000, quantiles=[0.1, 0.5, 0.9])
    return float(qs[0]), float(qs[1]), float(qs[2])


def _run_shape(
    shape: tuple[int, int],
    kernel_name: str,
    with_torch_compile: bool,
) -> dict:
    default_cfg, seed_cfg, hname = _resolve_configs(kernel_name, shape)
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    k_default, k_seed = _build_sum_kernels(default_cfg, seed_cfg)
    k_default(x)
    k_seed(x)
    torch.cuda.synchronize()

    _, d_p50, _ = _bench_kernel(lambda: k_default(x))
    _, s_p50, _ = _bench_kernel(lambda: k_seed(x))
    speedup = d_p50 / s_p50

    if with_torch_compile:
        tc = torch.compile(lambda t: t.sum(-1), fullgraph=True)
        tc(x)
        torch.cuda.synchronize()
        _, tc_p50, _ = _bench_kernel(lambda: tc(x))
        ratio = tc_p50 / s_p50
    else:
        tc_p50 = float("nan")
        ratio = float("nan")

    return {
        "shape": shape,
        "default_p50": d_p50,
        "heuristic_p50": s_p50,
        "tc_p50": tc_p50,
        "speedup": speedup,
        "tc_ratio": ratio,
        "launch_bound": shape in LAUNCH_BOUND,
        "heuristic": hname,
    }


def _run(kernel_name: str, mode: str, with_torch_compile: bool) -> None:
    shapes = SHAPES[f"{kernel_name}_{mode}"]
    print(f"# kernel={kernel_name} mode={mode}")
    print(
        f"# {'shape':<20} {'default p50':>12} {'heuristic p50':>14} "
        f"{'speedup':>9} {'tc p50':>9} {'tc_ratio':>9} heuristic"
    )
    headline_speedups: list[float] = []
    tc_ratios: list[float] = []

    for shape in shapes:
        row = _run_shape(shape, kernel_name, with_torch_compile)
        if not row["launch_bound"]:
            headline_speedups.append(row["speedup"])
            if not math.isnan(row["tc_ratio"]):
                tc_ratios.append(row["tc_ratio"])

        suffix = " [tiny-M]" if row["launch_bound"] else ""
        print(
            f"  {row['shape']!s:<18} {row['default_p50'] * 1000:>10.3f}us "
            f"{row['heuristic_p50'] * 1000:>12.3f}us "
            f"{row['speedup']:>9.3f} {row['tc_p50'] * 1000:>7.3f}us "
            f"{row['tc_ratio']:>9.3f} {row['heuristic']}{suffix}"
        )

    if headline_speedups:
        gs = math.exp(statistics.fmean(math.log(s) for s in headline_speedups))
        print(f"\ngeomean(default_speedup) [headline] = {gs:.3f}x")
        worst = min(headline_speedups)
        print(f"worst_shape_speedup_vs_default [headline] = {worst:.3f}x")
    if tc_ratios:
        gtc = math.exp(statistics.fmean(math.log(r) for r in tc_ratios))
        print(f"geomean(torch_compile_ratio) [headline] = {gtc:.3f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="sum", choices=["sum"])
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--with-torch-compile", action="store_true")
    args = parser.parse_args()
    mode = "validation" if args.validation else "train"
    _run(args.kernel, mode, args.with_torch_compile)


if __name__ == "__main__":
    main()
