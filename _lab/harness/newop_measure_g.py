"""GENERALITY (NEW REDUCTION OP): Product-A G for the max/min fixtures (a NEW
accumulator vs the sum/mean curriculum). fp32, H100, fresh process per op.

For each shape (all fp32, same do_bench, same inputs):
  - seed_lat       : v8 bare seed (configs=[seed], no autotune)
  - default_lat    : the UN-seeded default_config (pre-heuristic baseline)
  - p32_lat        : decisive A/B = same seed block_sizes but num_warps=32 forced
                     (the "best simple alternative" the method demands)
  - tc_default_lat : torch.compile(default) of torch.amax/amin(x, dim=-1)

G_seed = tc/seed ; G_default = tc/default ; G_p32 = tc/p32 ; seed/p32 = p32/seed.

seed-USED check: persistent iff codegen has NO `for roffset` inner reduction loop;
assert codegen matches the seed's persistent-vs-looped choice.

Usage: ... newop_measure_g.py [max|min]
Run with the canonical invocation (SETUP.md).
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

from _lab.harness.fixture_maxmin import max_kernel  # noqa: E402
from _lab.harness.fixture_maxmin import min_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7

# Representative shapes spanning regimes (from the task):
#   small-N, wide, grid-bound tiny-N/large-M, medium-large grid, very-wide tiny-M.
SHAPES = [
    (2048, 1024),    # small-N
    (2048, 16384),   # wide (w16)
    (8192, 256),     # grid-bound tiny-N / large-M
    (8192, 8192),    # medium-large grid
    (32768, 256),    # grid-bound tiny-N / very-large-M (M-floor 2)
    (256, 131072),   # tiny-M huge-N (grid-starved, w32)
]


def build_args(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def reference_fn(which):
    if which == "max":
        return lambda x: torch.amax(x, dim=-1)
    return lambda x: torch.amin(x, dim=-1)


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


def check_exact(out, ref):
    return bool(torch.equal(out, ref)), float((out.to(torch.float32) - ref).abs().max())


def codegen_persistent(triton_code):
    return "for roffset" not in triton_code


def get_seed(kern, args):
    bound = kern.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    return seeds[0]


def build_kernel(kern, cfg_dict, args):
    k = helion.kernel(kern.fn, configs=[helion.Config(**cfg_dict)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b


def measure_shape(which, kern, shape):
    ref_fn = reference_fn(which)
    args = build_args(shape)
    ref = ref_fn(*args)

    # --- seed (heuristic v8) ---
    seed = get_seed(kern, args)
    sd = dict(seed)
    bound_s = build_kernel(kern, sd, args)
    cfg_s = dict(bound_s._config)
    tcode = bound_s.to_triton_code(helion.Config(**cfg_s))
    cg_persist = codegen_persistent(tcode)
    seed_persist = cfg_s.get("reduction_loops", [None])[0] is None
    assert cg_persist == seed_persist, (
        f"seed-used MISMATCH {shape}: cg_persist={cg_persist} seed_persist={seed_persist}")
    out_s = bound_s(*args)
    ok_s, err_s = check_exact(out_s, ref)
    assert ok_s, f"seed correctness FAIL {shape} err={err_s}"
    seed_lat = median_do_bench(lambda: bound_s(*args))

    # --- default_config (un-seeded baseline) ---
    bound_d0 = kern.bind(args)
    cfg_d = dict(bound_d0.config_spec.default_config())
    bound_d = build_kernel(kern, cfg_d, args)
    out_d = bound_d(*args)
    ok_d, _ = check_exact(out_d, ref)
    assert ok_d, f"default correctness FAIL {shape}"
    default_lat = median_do_bench(lambda: bound_d(*args))

    # --- persistent/w32 (decisive A/B): same block_sizes/reduction_loops, w32 ---
    p32 = dict(sd)
    p32["num_warps"] = 32
    bound_p = build_kernel(kern, p32, args)
    out_p = bound_p(*args)
    ok_p, _ = check_exact(out_p, ref)
    assert ok_p, f"p32 correctness FAIL {shape}"
    p32_lat = median_do_bench(lambda: bound_p(*args))

    # --- torch.compile default of amax/amin ---
    torch._dynamo.reset()
    tc = torch.compile(ref_fn)
    out_tc = tc(*args)
    ok_tc, _ = check_exact(out_tc, ref)
    assert ok_tc, f"tc correctness FAIL {shape}"
    tc_lat = median_do_bench(lambda: tc(*args))

    return {
        "shape": shape, "warps": sd["num_warps"],
        "codegen": "persist" if cg_persist else "looped",
        "seed_lat_us": seed_lat * 1000, "default_lat_us": default_lat * 1000,
        "p32_lat_us": p32_lat * 1000, "tc_lat_us": tc_lat * 1000,
        "g_seed": tc_lat / seed_lat, "g_default": tc_lat / default_lat,
        "g_p32": tc_lat / p32_lat, "seed_over_p32": p32_lat / seed_lat,
        "err": err_s,
    }


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "max"
    kern = {"max": max_kernel, "min": min_kernel}[which]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  op={which} (NEW reduction OP, fp32)\n")
    header = (f"{'shape':>14} {'cg':>7} {'w':>3} {'seed_us':>9} {'dflt_us':>9} "
              f"{'p32_us':>9} {'tc_us':>9} {'G_seed':>7} {'G_dflt':>7} "
              f"{'G_p32':>7} {'s/p32':>6} {'maxabs':>8}")
    print(header)
    print("-" * len(header))
    g_seeds, g_defaults, g_p32s = [], [], []
    for shape in SHAPES:
        r = measure_shape(which, kern, shape)
        g_seeds.append(r["g_seed"]); g_defaults.append(r["g_default"])
        g_p32s.append(r["g_p32"])
        print(f"{str(r['shape']):>14} {r['codegen']:>7} {r['warps']:>3} "
              f"{r['seed_lat_us']:>9.1f} {r['default_lat_us']:>9.1f} "
              f"{r['p32_lat_us']:>9.1f} {r['tc_lat_us']:>9.1f} "
              f"{r['g_seed']:>7.3f} {r['g_default']:>7.3f} {r['g_p32']:>7.3f} "
              f"{r['seed_over_p32']:>6.3f} {r['err']:>8.1e}")

    def geomean(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs))

    print("-" * len(header))
    print(f"GEOMEAN  G_seed={geomean(g_seeds):.4f}  "
          f"G_default={geomean(g_defaults):.4f}  G_p32={geomean(g_p32s):.4f}")


if __name__ == "__main__":
    main()
