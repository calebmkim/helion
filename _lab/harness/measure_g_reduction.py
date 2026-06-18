"""Measure Product-A G for a row-reduction kernel (sum / long_sum) — generic.

For each in-sample shape, measure (all fp32, same do_bench, same inputs):
  - seed_lat       : the heuristic's bare seed (configs=[seed], no autotune)
  - default_lat    : the UN-seeded default_config (pre-heuristic baseline)
  - tc_default_lat : torch.compile default-mode of the fp32 reference (x.sum(-1))

Per-shape G_seed    = tc_default_lat / seed_lat   (the objective)
Per-shape G_default = tc_default_lat / default_lat (baseline to beat)

Reports per-shape table + geomean G_kernel for both seed and default. Also
confirms the seed was USED (persistent-vs-looped + num_warps reflected in codegen)
and correctness vs the fp32 reference.

Kernel selected by --kernel {sum,long_sum}. Single-tensor input, reference
torch.sum(x, dim=-1).

Run with the canonical invocation (see SETUP.md).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.long_sum import longsum  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7

KERNELS = {
    "sum": {
        "fn": sum_kernel,
        "shapes": [(2048, 1024), (2048, 4096), (2048, 16384), (4096, 1536),
                   (4096, 5120), (8192, 256), (8192, 4096), (32768, 256),
                   (32768, 1024)],
    },
    "long_sum": {
        "fn": longsum,
        "shapes": [(1, 32768), (2, 65536), (4, 130000), (8, 131072),
                   (16, 262144)],
    },
}


def build_args(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def reference(x):
    return torch.sum(x, dim=-1)


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


def check_correct(out, ref):
    o = out.to(torch.float32)
    r = ref.to(torch.float32)
    # fp32 sum reduction-order drift grows with rnumel; use a relative tol on the
    # magnitude. rtol=1e-3 is the standard Helion reduction tol; for very long
    # rows we also report max_rel so any loosening is explicit.
    ok = bool(torch.allclose(o, r, rtol=1e-3, atol=1e-3))
    max_abs = float((o - r).abs().max())
    max_rel = float(((o - r).abs() / (r.abs() + 1e-12)).max())
    return ok, max_abs, max_rel


def get_seed(fn, args):
    bound = fn.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    return seeds[0]


def looped_signature(triton_code):
    return "for roffset" in triton_code


def measure_shape(fn, shape):
    args = build_args(shape)
    x = args[0]
    ref = reference(x)

    # --- seed (heuristic) ---
    seed = get_seed(fn, args)
    seeded = helion.kernel(fn.fn, configs=[helion.Config(**dict(seed))])
    bound_s = seeded.bind(args)
    bound_s.ensure_config_exists(args)
    # seed-used proof: codegen persistent-vs-looped matches the seed
    tcode = bound_s.to_triton_code(helion.Config(**dict(bound_s._config)))
    want_looped = bool(dict(bound_s._config).get("reduction_loops", [None])[0])
    got_looped = looped_signature(tcode)
    seed_used = want_looped == got_looped
    out_s = bound_s(*args)
    out_s = out_s[0] if isinstance(out_s, tuple) else out_s
    ok_s, abs_s, rel_s = check_correct(out_s, ref)
    assert ok_s, f"seed correctness FAIL {shape} max_abs={abs_s} max_rel={rel_s}"
    assert seed_used, f"seed NOT used {shape}: want_looped={want_looped} got={got_looped}"
    seed_lat = median_do_bench(lambda: bound_s(*args))

    # --- default_config (un-seeded baseline) ---
    default_k = helion.kernel(fn.fn)
    bound_d = default_k.bind(args)
    cfg_d = bound_d.config_spec.default_config()
    default_k2 = helion.kernel(fn.fn, configs=[cfg_d])
    bound_d2 = default_k2.bind(args)
    bound_d2.ensure_config_exists(args)
    out_d = bound_d2(*args)
    out_d = out_d[0] if isinstance(out_d, tuple) else out_d
    ok_d, abs_d, _ = check_correct(out_d, ref)
    assert ok_d, f"default correctness FAIL {shape} max_abs={abs_d}"
    default_lat = median_do_bench(lambda: bound_d2(*args))

    # --- torch.compile default mode of the fp32 reference ---
    torch._dynamo.reset()
    tc = torch.compile(reference)  # default mode
    out_tc = tc(x)
    ok_tc, _, _ = check_correct(out_tc, ref)
    assert ok_tc, f"tc correctness FAIL {shape}"
    tc_lat = median_do_bench(lambda: tc(x))

    return {
        "shape": shape, "seed": dict(seed),
        "seed_lat_us": seed_lat * 1000, "default_lat_us": default_lat * 1000,
        "tc_lat_us": tc_lat * 1000,
        "g_seed": tc_lat / seed_lat, "g_default": tc_lat / default_lat,
        "codegen": "looped" if want_looped else "persistent",
        "max_rel": rel_s, "seed_used": seed_used,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=list(KERNELS))
    a = ap.parse_args()
    spec = KERNELS[a.kernel]
    fn = spec["fn"]

    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  kernel={a.kernel}\n")
    header = (f"{'shape':>14} {'codegen':>10} {'warps':>5} "
              f"{'seed_us':>9} {'dflt_us':>9} {'tc_us':>9} "
              f"{'G_seed':>7} {'G_dflt':>7} {'maxrel':>8}")
    print(header)
    print("-" * len(header))
    g_seeds, g_defaults = [], []
    for shape in spec["shapes"]:
        r = measure_shape(fn, shape)
        g_seeds.append(r["g_seed"])
        g_defaults.append(r["g_default"])
        print(f"{str(r['shape']):>14} {r['codegen']:>10} "
              f"{r['seed']['num_warps']:>5} "
              f"{r['seed_lat_us']:>9.1f} {r['default_lat_us']:>9.1f} "
              f"{r['tc_lat_us']:>9.1f} {r['g_seed']:>7.3f} {r['g_default']:>7.3f} "
              f"{r['max_rel']:>8.1e}")

    def geomean(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs))

    print("-" * len(header))
    print(f"GEOMEAN  G_seed = {geomean(g_seeds):.4f}   "
          f"G_default(baseline) = {geomean(g_defaults):.4f}")


if __name__ == "__main__":
    main()
