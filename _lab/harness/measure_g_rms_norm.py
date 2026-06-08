"""Measure Product-A G_rms_norm for the Triton reduction seed heuristic.

For each in-sample rms_norm shape, measure (all fp32, same do_bench, same inputs):
  - seed_lat       : the heuristic's bare seed (configs=[seed], no autotune)
  - default_lat    : the UN-seeded default_config (the pre-Step-2 baseline)
  - tc_default_lat : torch.compile default-mode of the fp32 reference

Per-shape G_seed    = tc_default_lat / seed_lat   (the objective)
Per-shape G_default = tc_default_lat / default_lat (baseline to beat: ~0.77)

Reports per-shape table + geomean G_rms_norm for both seed and default.

Run with the canonical invocation (see SETUP.md).
"""

from __future__ import annotations

import math
import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
N_RUNS = 7

IN_SAMPLE = [
    (2048, 1024), (2048, 2048), (2048, 4096), (2048, 8192), (2048, 16384),
    (4096, 1536), (4096, 3584), (4096, 5120), (4096, 7168),
    (8192, 4096), (8192, 8192),
    (32768, 256), (32768, 1024),
]


def build_args(shape):
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, EPS)


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


def get_seed(args):
    bound = rms_norm_fwd.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    return seeds[0]


def check_correct(out, ref):
    o = out.to(torch.float32)
    r = ref.to(torch.float32)
    return bool(torch.allclose(o, r, rtol=1e-3, atol=1e-4)), float((o - r).abs().max())


def measure_shape(shape):
    args = build_args(shape)
    x, w, eps = args
    ref = rms_norm_pytorch(x, w, eps)

    # --- seed (heuristic) ---
    seed = get_seed(args)
    seeded = helion.kernel(rms_norm_fwd.fn, configs=[helion.Config(**dict(seed))])
    bound_s = seeded.bind(args)
    bound_s.ensure_config_exists(args)
    out_s = bound_s(*args)
    out_s = out_s[0] if isinstance(out_s, tuple) else out_s
    ok_s, err_s = check_correct(out_s, ref)
    assert ok_s, f"seed correctness FAIL {shape} err={err_s}"
    seed_lat = median_do_bench(lambda: bound_s(*args))

    # --- default_config (un-seeded baseline) ---
    default_k = helion.kernel(rms_norm_fwd.fn)
    bound_d = default_k.bind(args)
    cfg_d = bound_d.config_spec.default_config()
    default_k2 = helion.kernel(rms_norm_fwd.fn, configs=[cfg_d])
    bound_d2 = default_k2.bind(args)
    bound_d2.ensure_config_exists(args)
    out_d = bound_d2(*args)
    out_d = out_d[0] if isinstance(out_d, tuple) else out_d
    ok_d, err_d = check_correct(out_d, ref)
    assert ok_d, f"default correctness FAIL {shape} err={err_d}"
    default_lat = median_do_bench(lambda: bound_d2(*args))

    # --- torch.compile default mode of the fp32 reference ---
    torch._dynamo.reset()
    tc = torch.compile(rms_norm_pytorch)  # default mode
    out_tc = tc(x, w, eps)
    ok_tc, err_tc = check_correct(out_tc, ref)
    assert ok_tc, f"tc correctness FAIL {shape} err={err_tc}"
    tc_lat = median_do_bench(lambda: tc(x, w, eps))

    g_seed = tc_lat / seed_lat
    g_default = tc_lat / default_lat
    return {
        "shape": shape, "seed": dict(seed),
        "seed_lat_us": seed_lat * 1000, "default_lat_us": default_lat * 1000,
        "tc_lat_us": tc_lat * 1000, "g_seed": g_seed, "g_default": g_default,
        "seed_codegen": "looped" if dict(bound_s._config).get("reduction_loops", [None])[0] else "persistent",
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}\n")
    header = (
        f"{'shape':>14} {'codegen':>10} {'warps':>5} "
        f"{'seed_us':>9} {'dflt_us':>9} {'tc_us':>9} "
        f"{'G_seed':>7} {'G_dflt':>7}"
    )
    print(header)
    print("-" * len(header))
    g_seeds, g_defaults = [], []
    rows = []
    for shape in IN_SAMPLE:
        r = measure_shape(shape)
        g_seeds.append(r["g_seed"])
        g_defaults.append(r["g_default"])
        rows.append(r)
        print(
            f"{str(r['shape']):>14} {r['seed_codegen']:>10} "
            f"{r['seed']['num_warps']:>5} "
            f"{r['seed_lat_us']:>9.1f} {r['default_lat_us']:>9.1f} "
            f"{r['tc_lat_us']:>9.1f} {r['g_seed']:>7.3f} {r['g_default']:>7.3f}"
        )

    def geomean(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs))

    print("-" * len(header))
    print(f"GEOMEAN  G_seed = {geomean(g_seeds):.4f}   "
          f"G_default(baseline) = {geomean(g_defaults):.4f}")


if __name__ == "__main__":
    main()
