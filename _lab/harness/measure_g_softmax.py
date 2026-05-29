"""Measure Product-A G_softmax for softmax_two_pass (T2, Band A, fp32).

For each in-sample shape (all fp32, same do_bench, same inputs):
  - seed_lat       : the heuristic's bare seed (configs=[seed], no autotune)
  - default_lat    : the UN-seeded default_config (pre-heuristic baseline)
  - tc_default_lat : torch.compile default-mode of F.softmax(x, dim=1)
  - p32_lat        : the DECISIVE A/B baseline = persistent (R_BLOCK=next_pow2(N))
                     forced to num_warps=32, M_BLOCK at floor. Same persistent
                     codegen as the seed but with a fixed w32 (the "best simple
                     alternative" the methodology demands we beat, NOT the default).

G_seed   = tc_lat / seed_lat ; G_default = tc_lat / default_lat ; G_p32 = tc_lat/p32_lat
seed/p32 = p32_lat / seed_lat (>1 => seed faster than persistent/w32).

T2 seed-used check: the reduction-axis block_size R_BLOCK >= N so the inner
`for offset in tl.range(0, N, R_BLOCK)` runs EXACTLY ONCE (persistent). We assert
the codegen `_BLOCK_SIZE_<red> = tl.constexpr(R_BLOCK)` with R_BLOCK>=N.

Run with the canonical invocation (SETUP.md).
"""

from __future__ import annotations

import math
import os
import re
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.softmax import softmax_two_pass  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7

IN_SAMPLE = [
    (4096, 256), (4096, 512), (4096, 1024), (4096, 2048), (4096, 4096),
    (4096, 8192), (4096, 12288), (4096, 16384), (32768, 256), (32768, 1024),
]


def build_args(shape):
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    return (x,)


def reference(x):
    return torch.nn.functional.softmax(x, dim=1)


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


def get_seed(args):
    bound = softmax_two_pass.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    return seeds[0]


def check_correct(out, ref):
    o = out.to(torch.float32)
    r = ref.to(torch.float32)
    ok = bool(torch.allclose(o, r, rtol=1e-3, atol=1e-4))
    return ok, float((o - r).abs().max())


def t2_persistent_used(triton_code, n):
    """For T2: find the largest _BLOCK_SIZE constant used as a tl.range step over
    the reduction axis; persistent iff that step >= n (loop runs once)."""
    steps = []
    consts = {m.group(1): int(m.group(2))
              for m in re.finditer(r"(_BLOCK_SIZE_\d+)\s*=\s*tl\.constexpr\((\d+)\)",
                                   triton_code)}
    for m in re.finditer(r"tl\.range\(0,\s*(\d+),\s*(_BLOCK_SIZE_\d+)\)", triton_code):
        extent = int(m.group(1))
        step = consts.get(m.group(2))
        if step is not None and extent == n:
            steps.append(step)
    if not steps:
        return None  # could not locate
    return max(steps) >= n


def build_kernel(seed_dict, args):
    k = helion.kernel(softmax_two_pass.fn, configs=[helion.Config(**seed_dict)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    return b


def measure_shape(shape):
    m, n = shape
    args = build_args(shape)
    ref = reference(*args)

    # --- seed (heuristic) ---
    seed = get_seed(args)
    sd = dict(seed)
    bound_s = build_kernel(sd, args)
    tcode = bound_s.to_triton_code(helion.Config(**dict(bound_s._config)))
    used = t2_persistent_used(tcode, n)
    out_s = bound_s(*args)
    ok_s, err_s = check_correct(out_s, ref)
    assert ok_s, f"seed correctness FAIL {shape} err={err_s}"
    assert used is True, f"seed NOT persistent T2 {shape}: used={used}"
    seed_lat = median_do_bench(lambda: bound_s(*args))

    # --- default_config (un-seeded baseline) ---
    bound_d0 = softmax_two_pass.bind(args)
    cfg_d = dict(bound_d0.config_spec.default_config())
    bound_d = build_kernel(cfg_d, args)
    out_d = bound_d(*args)
    ok_d, err_d = check_correct(out_d, ref)
    assert ok_d, f"default correctness FAIL {shape} err={err_d}"
    default_lat = median_do_bench(lambda: bound_d(*args))

    # --- persistent/w32 (decisive A/B) : same block_sizes as seed, num_warps=32 ---
    p32 = dict(sd)
    p32["num_warps"] = 32
    bound_p = build_kernel(p32, args)
    out_p = bound_p(*args)
    ok_p, _ = check_correct(out_p, ref)
    assert ok_p, f"p32 correctness FAIL {shape}"
    p32_lat = median_do_bench(lambda: bound_p(*args))

    # --- torch.compile default mode of F.softmax ---
    torch._dynamo.reset()
    tc = torch.compile(reference)
    out_tc = tc(*args)
    ok_tc, _ = check_correct(out_tc, ref)
    assert ok_tc, f"tc correctness FAIL {shape}"
    tc_lat = median_do_bench(lambda: tc(*args))

    return {
        "shape": shape, "seed": sd,
        "seed_lat_us": seed_lat * 1000, "default_lat_us": default_lat * 1000,
        "p32_lat_us": p32_lat * 1000, "tc_lat_us": tc_lat * 1000,
        "g_seed": tc_lat / seed_lat, "g_default": tc_lat / default_lat,
        "g_p32": tc_lat / p32_lat, "seed_over_p32": p32_lat / seed_lat,
        "warps": sd["num_warps"], "err": err_s,
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  kernel=softmax_two_pass (T2 fp32)\n")
    header = (f"{'shape':>14} {'warps':>5} {'seed_us':>9} {'dflt_us':>9} "
              f"{'p32_us':>9} {'tc_us':>9} {'G_seed':>7} {'G_dflt':>7} "
              f"{'G_p32':>7} {'s/p32':>6} {'maxabs':>8}")
    print(header)
    print("-" * len(header))
    g_seeds, g_defaults, g_p32s = [], [], []
    for shape in IN_SAMPLE:
        r = measure_shape(shape)
        g_seeds.append(r["g_seed"]); g_defaults.append(r["g_default"])
        g_p32s.append(r["g_p32"])
        print(f"{str(r['shape']):>14} {r['warps']:>5} "
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
