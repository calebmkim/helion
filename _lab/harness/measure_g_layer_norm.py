"""Measure Product-A G_layer_norm for the Triton reduction seed heuristic.

For each in-sample layer_norm_fwd shape, measure (all fp32, same do_bench, same
inputs):
  - seed_lat       : the heuristic's bare seed (configs=[seed], no autotune)
  - default_lat    : the UN-seeded default_config (pre-heuristic baseline)
  - tc_default_lat : torch.compile default-mode of the fp32 reference
                     (torch.nn.functional.layer_norm)

Per-shape G_seed    = tc_default_lat / seed_lat   (the objective)
Per-shape G_default = tc_default_lat / default_lat (baseline to beat)

Reports per-shape table + geomean G_layer_norm for seed and default. Confirms
seed USED (persistent-vs-looped + num_warps in codegen) and correctness vs the
fp32 F.layer_norm reference (only the normalized OUTPUT, which is what tc returns).

WITH bias (tritonbench default). Run with the canonical invocation (SETUP.md).
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

from examples.layer_norm import layer_norm_fwd  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
N_RUNS = 7

IN_SAMPLE = [
    (4096, 1024), (4096, 2048), (4096, 4096), (4096, 8192),
    (4096, 12288), (4096, 15872),
    (2048, 3584), (2048, 8192),
    (8192, 4096), (8192, 5120), (8192, 7168),
]


def build_args(shape):
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, [n], w, b, EPS)


def reference(x, normalized_shape, w, b, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, w, b, eps)


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


def get_seed(args):
    bound = layer_norm_fwd.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    return seeds[0]


def check_correct(out, ref):
    o = out.to(torch.float32)
    r = ref.to(torch.float32)
    # fp32 layer_norm: rtol=1e-3 standard Helion tol; atol=1e-3 covers the small
    # absolute drift in the centered/variance pass. allclose must pass.
    ok = bool(torch.allclose(o, r, rtol=1e-3, atol=1e-3))
    return ok, float((o - r).abs().max())


def looped_signature(triton_code):
    return "for roffset" in triton_code


def measure_shape(shape):
    args = build_args(shape)
    x, ns, w, b, eps = args
    ref = reference(*args)

    # --- seed (heuristic) ---
    seed = get_seed(args)
    seeded = helion.kernel(layer_norm_fwd.fn, configs=[helion.Config(**dict(seed))])
    bound_s = seeded.bind(args)
    bound_s.ensure_config_exists(args)
    tcode = bound_s.to_triton_code(helion.Config(**dict(bound_s._config)))
    want_looped = bool(dict(bound_s._config).get("reduction_loops", [None])[0])
    got_looped = looped_signature(tcode)
    seed_used = want_looped == got_looped
    out_s = bound_s(*args)
    out_s = out_s[0] if isinstance(out_s, tuple) else out_s
    ok_s, err_s = check_correct(out_s, ref)
    assert ok_s, f"seed correctness FAIL {shape} err={err_s}"
    assert seed_used, f"seed NOT used {shape}: want_looped={want_looped} got={got_looped}"
    seed_lat = median_do_bench(lambda: bound_s(*args))

    # --- default_config (un-seeded baseline) ---
    default_k = helion.kernel(layer_norm_fwd.fn)
    bound_d = default_k.bind(args)
    cfg_d = bound_d.config_spec.default_config()
    default_k2 = helion.kernel(layer_norm_fwd.fn, configs=[cfg_d])
    bound_d2 = default_k2.bind(args)
    bound_d2.ensure_config_exists(args)
    out_d = bound_d2(*args)
    out_d = out_d[0] if isinstance(out_d, tuple) else out_d
    ok_d, err_d = check_correct(out_d, ref)
    assert ok_d, f"default correctness FAIL {shape} err={err_d}"
    default_lat = median_do_bench(lambda: bound_d2(*args))

    # --- torch.compile default mode of the fp32 reference ---
    torch._dynamo.reset()
    tc = torch.compile(reference)  # default mode
    out_tc = tc(*args)
    ok_tc, err_tc = check_correct(out_tc, ref)
    assert ok_tc, f"tc correctness FAIL {shape} err={err_tc}"
    tc_lat = median_do_bench(lambda: tc(*args))

    return {
        "shape": shape, "seed": dict(seed),
        "seed_lat_us": seed_lat * 1000, "default_lat_us": default_lat * 1000,
        "tc_lat_us": tc_lat * 1000,
        "g_seed": tc_lat / seed_lat, "g_default": tc_lat / default_lat,
        "codegen": "looped" if want_looped else "persistent",
        "seed_used": seed_used, "err": err_s,
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  kernel=layer_norm_fwd (with bias)\n")
    header = (f"{'shape':>14} {'codegen':>10} {'warps':>5} "
              f"{'seed_us':>9} {'dflt_us':>9} {'tc_us':>9} "
              f"{'G_seed':>7} {'G_dflt':>7} {'maxabs':>8}")
    print(header)
    print("-" * len(header))
    g_seeds, g_defaults = [], []
    for shape in IN_SAMPLE:
        r = measure_shape(shape)
        g_seeds.append(r["g_seed"])
        g_defaults.append(r["g_default"])
        print(f"{str(r['shape']):>14} {r['codegen']:>10} "
              f"{r['seed']['num_warps']:>5} "
              f"{r['seed_lat_us']:>9.1f} {r['default_lat_us']:>9.1f} "
              f"{r['tc_lat_us']:>9.1f} {r['g_seed']:>7.3f} {r['g_default']:>7.3f} "
              f"{r['err']:>8.1e}")

    def geomean(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs))

    print("-" * len(header))
    print(f"GEOMEAN  G_seed = {geomean(g_seeds):.4f}   "
          f"G_default(baseline) = {geomean(g_defaults):.4f}")


if __name__ == "__main__":
    main()
