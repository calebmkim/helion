"""Signal-(a) driver for the reduction heuristic.

Builds two `@helion.kernel(config=...)` instances of the same kernel — one
hardcoded to `default_config()`, one hardcoded to the heuristic's seed —
and benches them head-to-head with `triton.testing.do_bench` (quantiles).

Single-process is load-bearing for low-noise comparisons of small-tensor
reductions, where wall-clock differences are O(1us) and cross-process
re-init/state shifts dwarf the signal.

    PYTHONPATH=$PWD python scripts/bench_reduction_heuristic.py --kernel sum
    PYTHONPATH=$PWD python scripts/bench_reduction_heuristic.py --kernel rms_norm --validation
    PYTHONPATH=$PWD python scripts/bench_reduction_heuristic.py --kernel all
"""

from __future__ import annotations

import argparse
import math
import statistics
from typing import Callable

import torch
import torch.nn.functional as F
import triton.testing

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs
import helion.language as hl

# Train/validation shapes from the plan §10.2.
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
    "rms_norm_train": [
        (2048, 1024),
        (2048, 2048),
        (2048, 4096),
        (2048, 8192),
        (2048, 16384),
        (4096, 1536),
        (4096, 3584),
        (4096, 5120),
        (4096, 7168),
        (8192, 4096),
        (8192, 8192),
        (32768, 256),
        (32768, 1024),
    ],
    "rms_norm_validation": [
        (16, 4096),
        (2048, 1023),
        (2048, 3072),
        (4096, 12288),
        (262144, 256),
    ],
    "layer_norm_train": [
        (4096, 1024),
        (4096, 2048),
        (4096, 4096),
        (4096, 8192),
        (4096, 12288),
        (4096, 15872),
        (2048, 3584),
        (2048, 8192),
        (8192, 4096),
        (8192, 5120),
        (8192, 7168),
    ],
    "layer_norm_validation": [
        (16, 4096),
        (2048, 1023),
        (2048, 1536),
        (4096, 6144),
        (1024, 32768),
    ],
    "softmax_train": [
        (4096, 256),
        (4096, 512),
        (4096, 1024),
        (4096, 2048),
        (4096, 4096),
        (4096, 8192),
        (4096, 12288),
        (4096, 16384),
        (32768, 256),
        (32768, 1024),
    ],
    "softmax_validation": [
        (16, 4096),
        (2048, 1023),
        (2048, 32768),
        (128, 131072),
    ],
    "softmax_decomposed_train": [
        (4096, 256),
        (4096, 512),
        (4096, 1024),
        (4096, 2048),
        (4096, 4096),
        (4096, 8192),
        (4096, 12288),
        (4096, 16384),
        (32768, 256),
        (32768, 1024),
    ],
    "softmax_decomposed_validation": [
        (16, 4096),
        (2048, 1023),
        (2048, 32768),
        (128, 131072),
    ],
    "cross_entropy_train": [
        (4096, 4096),
        (4096, 16384),
        (8192, 32768),
        (16384, 32768),
        (8192, 65536),
        (16384, 65536),
        (8192, 131072),
    ],
    "cross_entropy_validation": [
        (2048, 32000),
        (8192, 128000),
        (4096, 129280),
        (1024, 256000),
    ],
    "longsum_train": [
        (1, 32768),
        (2, 65536),
        (4, 130000),
        (8, 131072),
        (16, 262144),
    ],
    "longsum_validation": [
        (1, 100000),
        (4, 262143),
    ],
}

# Tiny-M shapes excluded from headline geomean per §7.3 (launch-bound).
LAUNCH_BOUND: set[tuple[int, int]] = {(16, 4096)}


KernelFn = Callable[..., object]
KernelPair = tuple[KernelFn, KernelFn]


# ---------------- per-kernel definitions ----------------


def _build_sum(default_cfg: helion.Config, seed_cfg: helion.Config) -> KernelPair:
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


def _probe_sum() -> KernelFn:
    @helion.kernel(static_shapes=True)
    def probe(x: torch.Tensor) -> torch.Tensor:
        m, _ = x.shape
        out = torch.empty([m], dtype=x.dtype, device=x.device)
        for tile_m in hl.tile(m):
            out[tile_m] = x[tile_m, :].sum(-1)
        return out

    return probe


def _build_rms_norm(
    default_cfg: helion.Config, seed_cfg: helion.Config
) -> KernelPair:
    @helion.kernel(config=default_cfg, static_shapes=True)
    def k_default(
        x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, n = x.size()
        out = torch.empty_like(x)
        inv_rms = torch.empty([m], dtype=torch.float32, device=x.device)
        for tile_m in hl.tile(m):
            x_tile = x[tile_m, :].to(torch.float32)
            x_squared = x_tile * x_tile
            mean_x_squared = torch.mean(x_squared, dim=-1)
            inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
            normalized = x_tile * inv_rms_tile[:, None]
            out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)
            inv_rms[tile_m] = inv_rms_tile
        return out, inv_rms.reshape(-1, 1)

    @helion.kernel(config=seed_cfg, static_shapes=True)
    def k_seed(
        x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, n = x.size()
        out = torch.empty_like(x)
        inv_rms = torch.empty([m], dtype=torch.float32, device=x.device)
        for tile_m in hl.tile(m):
            x_tile = x[tile_m, :].to(torch.float32)
            x_squared = x_tile * x_tile
            mean_x_squared = torch.mean(x_squared, dim=-1)
            inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
            normalized = x_tile * inv_rms_tile[:, None]
            out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)
            inv_rms[tile_m] = inv_rms_tile
        return out, inv_rms.reshape(-1, 1)

    return k_default, k_seed


def _probe_rms_norm() -> KernelFn:
    @helion.kernel(static_shapes=True)
    def probe(
        x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m, n = x.size()
        out = torch.empty_like(x)
        inv_rms = torch.empty([m], dtype=torch.float32, device=x.device)
        for tile_m in hl.tile(m):
            x_tile = x[tile_m, :].to(torch.float32)
            x_squared = x_tile * x_tile
            mean_x_squared = torch.mean(x_squared, dim=-1)
            inv_rms_tile = torch.rsqrt(mean_x_squared + eps)
            normalized = x_tile * inv_rms_tile[:, None]
            out[tile_m, :] = (normalized * weight[:].to(torch.float32)).to(out.dtype)
            inv_rms[tile_m] = inv_rms_tile
        return out, inv_rms.reshape(-1, 1)

    return probe


def _build_layer_norm(
    default_cfg: helion.Config, seed_cfg: helion.Config
) -> KernelPair:
    @helion.kernel(config=default_cfg, static_shapes=True)
    def k_default(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        mean = torch.empty([m], dtype=torch.float32, device=x.device)
        rstd = torch.empty([m], dtype=torch.float32, device=x.device)
        for tile_m in hl.tile(m):
            acc = x[tile_m, :].to(torch.float32)
            mean_val = torch.sum(acc, dim=-1) / n
            centered = acc - mean_val[:, None]
            var_val = torch.sum(centered * centered, dim=-1) / n
            rstd_val = torch.rsqrt(var_val + eps)
            normalized = centered * rstd_val[:, None]
            acc = normalized * (weight[:].to(torch.float32)) + (
                bias[:].to(torch.float32)
            )
            out[tile_m, :] = acc.to(x.dtype)
            mean[tile_m] = mean_val
            rstd[tile_m] = rstd_val
        return out, mean, rstd

    @helion.kernel(config=seed_cfg, static_shapes=True)
    def k_seed(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        mean = torch.empty([m], dtype=torch.float32, device=x.device)
        rstd = torch.empty([m], dtype=torch.float32, device=x.device)
        for tile_m in hl.tile(m):
            acc = x[tile_m, :].to(torch.float32)
            mean_val = torch.sum(acc, dim=-1) / n
            centered = acc - mean_val[:, None]
            var_val = torch.sum(centered * centered, dim=-1) / n
            rstd_val = torch.rsqrt(var_val + eps)
            normalized = centered * rstd_val[:, None]
            acc = normalized * (weight[:].to(torch.float32)) + (
                bias[:].to(torch.float32)
            )
            out[tile_m, :] = acc.to(x.dtype)
            mean[tile_m] = mean_val
            rstd[tile_m] = rstd_val
        return out, mean, rstd

    return k_default, k_seed


def _probe_layer_norm() -> KernelFn:
    @helion.kernel(static_shapes=True)
    def probe(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m, n = x.size()
        out = torch.empty([m, n], dtype=x.dtype, device=x.device)
        mean = torch.empty([m], dtype=torch.float32, device=x.device)
        rstd = torch.empty([m], dtype=torch.float32, device=x.device)
        for tile_m in hl.tile(m):
            acc = x[tile_m, :].to(torch.float32)
            mean_val = torch.sum(acc, dim=-1) / n
            centered = acc - mean_val[:, None]
            var_val = torch.sum(centered * centered, dim=-1) / n
            rstd_val = torch.rsqrt(var_val + eps)
            normalized = centered * rstd_val[:, None]
            acc = normalized * (weight[:].to(torch.float32)) + (
                bias[:].to(torch.float32)
            )
            out[tile_m, :] = acc.to(x.dtype)
            mean[tile_m] = mean_val
            rstd[tile_m] = rstd_val
        return out, mean, rstd

    return probe


def _build_softmax(
    default_cfg: helion.Config, seed_cfg: helion.Config
) -> KernelPair:
    @helion.kernel(config=default_cfg, static_shapes=True)
    def k_default(x: torch.Tensor) -> torch.Tensor:
        n, _m = x.size()
        out = torch.empty_like(x)
        for tile_n in hl.tile(n):
            out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
        return out

    @helion.kernel(config=seed_cfg, static_shapes=True)
    def k_seed(x: torch.Tensor) -> torch.Tensor:
        n, _m = x.size()
        out = torch.empty_like(x)
        for tile_n in hl.tile(n):
            out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
        return out

    return k_default, k_seed


def _probe_softmax() -> KernelFn:
    @helion.kernel(static_shapes=True)
    def probe(x: torch.Tensor) -> torch.Tensor:
        n, _m = x.size()
        out = torch.empty_like(x)
        for tile_n in hl.tile(n):
            out[tile_n, :] = torch.nn.functional.softmax(x[tile_n, :], dim=1)
        return out

    return probe


def _build_softmax_decomposed(
    default_cfg: helion.Config, seed_cfg: helion.Config
) -> KernelPair:
    @helion.kernel(config=default_cfg, static_shapes=True)
    def k_default(x: torch.Tensor) -> torch.Tensor:
        n, _m = x.size()
        out = torch.empty_like(x)
        for tile_n in hl.tile(n):
            values = x[tile_n, :]
            amax = torch.amax(values, dim=1, keepdim=True)
            exp = torch.exp(values - amax)
            sum_exp = torch.sum(exp, dim=1, keepdim=True)
            out[tile_n, :] = exp / sum_exp
        return out

    @helion.kernel(config=seed_cfg, static_shapes=True)
    def k_seed(x: torch.Tensor) -> torch.Tensor:
        n, _m = x.size()
        out = torch.empty_like(x)
        for tile_n in hl.tile(n):
            values = x[tile_n, :]
            amax = torch.amax(values, dim=1, keepdim=True)
            exp = torch.exp(values - amax)
            sum_exp = torch.sum(exp, dim=1, keepdim=True)
            out[tile_n, :] = exp / sum_exp
        return out

    return k_default, k_seed


def _probe_softmax_decomposed() -> KernelFn:
    @helion.kernel(static_shapes=True)
    def probe(x: torch.Tensor) -> torch.Tensor:
        n, _m = x.size()
        out = torch.empty_like(x)
        for tile_n in hl.tile(n):
            values = x[tile_n, :]
            amax = torch.amax(values, dim=1, keepdim=True)
            exp = torch.exp(values - amax)
            sum_exp = torch.sum(exp, dim=1, keepdim=True)
            out[tile_n, :] = exp / sum_exp
        return out

    return probe


def _build_cross_entropy(
    default_cfg: helion.Config, seed_cfg: helion.Config
) -> KernelPair:
    @helion.kernel(
        config=default_cfg,
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    def k_default(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n, v = logits.shape
        losses = torch.zeros([n], dtype=logits.dtype, device=logits.device)
        logits_flat = logits.view(-1)
        for tile_n in hl.tile(n):
            labels_tile = labels[tile_n]
            base_indices_tile = tile_n.index * v
            flat_indices = base_indices_tile + labels_tile
            logits_at_target = hl.load(logits_flat, [flat_indices])
            logits_rows = logits[tile_n, :]
            max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
            shifted = logits_rows - max_logits
            exp_shifted = torch.exp(shifted)
            sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
            log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
            losses[tile_n] = log_sum_exp - logits_at_target
        return losses.mean()

    @helion.kernel(
        config=seed_cfg,
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    def k_seed(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n, v = logits.shape
        losses = torch.zeros([n], dtype=logits.dtype, device=logits.device)
        logits_flat = logits.view(-1)
        for tile_n in hl.tile(n):
            labels_tile = labels[tile_n]
            base_indices_tile = tile_n.index * v
            flat_indices = base_indices_tile + labels_tile
            logits_at_target = hl.load(logits_flat, [flat_indices])
            logits_rows = logits[tile_n, :]
            max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
            shifted = logits_rows - max_logits
            exp_shifted = torch.exp(shifted)
            sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
            log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
            losses[tile_n] = log_sum_exp - logits_at_target
        return losses.mean()

    return k_default, k_seed


def _probe_cross_entropy() -> KernelFn:
    @helion.kernel(
        static_shapes=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    )
    def probe(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        n, v = logits.shape
        losses = torch.zeros([n], dtype=logits.dtype, device=logits.device)
        logits_flat = logits.view(-1)
        for tile_n in hl.tile(n):
            labels_tile = labels[tile_n]
            base_indices_tile = tile_n.index * v
            flat_indices = base_indices_tile + labels_tile
            logits_at_target = hl.load(logits_flat, [flat_indices])
            logits_rows = logits[tile_n, :]
            max_logits = torch.amax(logits_rows, dim=-1, keepdim=True)
            shifted = logits_rows - max_logits
            exp_shifted = torch.exp(shifted)
            sum_exp = torch.sum(exp_shifted, dim=-1, keepdim=True)
            log_sum_exp = max_logits.squeeze(-1) + torch.log(sum_exp.squeeze(-1))
            losses[tile_n] = log_sum_exp - logits_at_target
        return losses.mean()

    return probe


# ---- longsum: same shape as sum_kernel, separated for shape lists. ----
_build_longsum = _build_sum
_probe_longsum = _probe_sum


# ---------------- kernel registry ----------------


def _make_args(kernel_name: str, shape: tuple[int, int]) -> tuple:
    m, n = shape
    if kernel_name in ("sum", "longsum"):
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        return (x,)
    if kernel_name == "rms_norm":
        x = torch.randn(m, n, device="cuda", dtype=torch.float16)
        weight = torch.randn(n, device="cuda", dtype=torch.float16)
        return (x, weight)
    if kernel_name == "layer_norm":
        x = torch.randn(m, n, device="cuda", dtype=torch.float16)
        weight = torch.randn(n, device="cuda", dtype=torch.float16)
        bias = torch.randn(n, device="cuda", dtype=torch.float16)
        return (x, weight, bias)
    if kernel_name in ("softmax", "softmax_decomposed"):
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        return (x,)
    if kernel_name == "cross_entropy":
        logits = torch.randn(m, n, device="cuda", dtype=torch.float32)
        labels = torch.randint(0, n, (m,), device="cuda", dtype=torch.long)
        return (logits, labels)
    raise SystemExit(f"unknown kernel: {kernel_name}")


def _torch_compile_baseline(kernel_name: str) -> Callable[..., torch.Tensor]:
    """Eager equivalent of the kernel for torch.compile timing."""
    if kernel_name in ("sum", "longsum"):
        return torch.compile(lambda x: x.sum(-1), fullgraph=True)
    if kernel_name == "rms_norm":

        def _rms(
            x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5
        ) -> torch.Tensor:
            xf = x.to(torch.float32)
            inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
            return (xf * inv * weight.to(torch.float32)).to(x.dtype)

        return torch.compile(_rms, fullgraph=True)
    if kernel_name == "layer_norm":

        def _ln(
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: torch.Tensor,
            eps: float = 1e-5,
        ) -> torch.Tensor:
            return F.layer_norm(x, [x.size(-1)], weight, bias, eps)

        return torch.compile(_ln, fullgraph=True)
    if kernel_name == "softmax":
        return torch.compile(lambda x: F.softmax(x, dim=-1), fullgraph=True)
    if kernel_name == "softmax_decomposed":
        return torch.compile(lambda x: F.softmax(x, dim=-1), fullgraph=True)
    if kernel_name == "cross_entropy":
        return torch.compile(
            lambda logits, labels: F.cross_entropy(logits, labels), fullgraph=True
        )
    raise SystemExit(f"unknown kernel for torch.compile: {kernel_name}")


def _builders(
    kernel_name: str,
) -> tuple[Callable[..., KernelPair], Callable[[], KernelFn]]:
    return {
        "sum": (_build_sum, _probe_sum),
        "rms_norm": (_build_rms_norm, _probe_rms_norm),
        "layer_norm": (_build_layer_norm, _probe_layer_norm),
        "softmax": (_build_softmax, _probe_softmax),
        "softmax_decomposed": (_build_softmax_decomposed, _probe_softmax_decomposed),
        "cross_entropy": (_build_cross_entropy, _probe_cross_entropy),
        "longsum": (_build_longsum, _probe_longsum),
    }[kernel_name]


# ---------------- bench plumbing ----------------


def _resolve_configs(
    kernel_name: str, shape: tuple[int, int]
) -> tuple[helion.Config, helion.Config, str]:
    args = _make_args(kernel_name, shape)
    _, probe_factory = _builders(kernel_name)
    probe = probe_factory()
    bound = probe.bind(args)
    default_cfg = bound.config_spec.default_config()
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    if not seeds:
        return default_cfg, default_cfg, "no-seed"
    name = bound.config_spec.autotuner_heuristics[0]
    return default_cfg, seeds[0], name


def _bench(fn: Callable[[], object]) -> float:
    fn()
    torch.cuda.synchronize()
    qs = triton.testing.do_bench(fn, warmup=200, rep=2000, quantiles=[0.1, 0.5, 0.9])
    return float(qs[1])


def _run_shape(
    shape: tuple[int, int],
    kernel_name: str,
    with_torch_compile: bool,
) -> dict:
    default_cfg, seed_cfg, hname = _resolve_configs(kernel_name, shape)
    args = _make_args(kernel_name, shape)
    build_factory, _ = _builders(kernel_name)
    k_default, k_seed = build_factory(default_cfg, seed_cfg)
    k_default(*args)
    k_seed(*args)
    torch.cuda.synchronize()

    d_p50 = _bench(lambda: k_default(*args))
    s_p50 = _bench(lambda: k_seed(*args))
    speedup = d_p50 / s_p50

    if with_torch_compile:
        tc = _torch_compile_baseline(kernel_name)
        tc(*args)
        torch.cuda.synchronize()
        tc_p50 = _bench(lambda: tc(*args))
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
        try:
            row = _run_shape(shape, kernel_name, with_torch_compile)
        except Exception as e:
            print(f"  {shape!s:<18}  ERROR: {type(e).__name__}: {e}")
            continue
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
        worst_tc = min(tc_ratios)
        print(f"geomean(torch_compile_ratio) [headline] = {gtc:.3f}x")
        print(f"worst_shape_tc_ratio [headline] = {worst_tc:.3f}x")


KERNELS = (
    "sum",
    "rms_norm",
    "layer_norm",
    "softmax",
    "softmax_decomposed",
    "cross_entropy",
    "longsum",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kernel", default="sum", choices=(*KERNELS, "all")
    )
    parser.add_argument("--validation", action="store_true")
    parser.add_argument("--with-torch-compile", action="store_true")
    args = parser.parse_args()
    mode = "validation" if args.validation else "train"
    if args.kernel == "all":
        for k in KERNELS:
            _run(k, mode, args.with_torch_compile)
            print()
    else:
        _run(args.kernel, mode, args.with_torch_compile)


if __name__ == "__main__":
    main()
