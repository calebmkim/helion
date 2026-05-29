"""Measure Product-A G_cross_entropy for the Triton reduction seed heuristic.

For each in-sample cross_entropy shape (all fp32, same do_bench, same inputs):
  - seed_lat       : the heuristic's bare seed (configs=[seed], no autotune)
  - default_lat    : the UN-seeded default_config (pre-heuristic baseline)
  - tc_default_lat : torch.compile default-mode of F.cross_entropy (fp32 ref)
  - p32_lat        : DECISIVE A/B -- persistent (reduction_loops=[None]) at a
                     FIXED num_warps=32 (the best SIMPLE alternative; holds the
                     persistent lever EQUAL so we isolate the rnumel warps ramp).

Per-shape G_seed    = tc_default_lat / seed_lat
Per-shape G_default = tc_default_lat / default_lat
seed/p32            = p32_lat / seed_lat (>1 => seed faster than fixed-w32)

Confirms seed USED (codegen persistent + num_warps) and correct vs F.cross_entropy.
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

from examples.cross_entropy import cross_entropy  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

LONG = torch.int64
N_RUNS = 7

IN_SAMPLE = [
    (4096, 4096), (4096, 16384), (8192, 32768), (16384, 32768),
    (8192, 65536), (16384, 65536), (8192, 131072),
]


def build_args(shape):
    n, v = shape
    logits = torch.randn(n, v, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device="cuda", dtype=LONG)
    return (logits, labels)


def reference(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels)


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


def get_seed(args):
    bound = cross_entropy.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    return seeds[0]


def check_correct(out, ref):
    o = out.to(torch.float32)
    r = ref.to(torch.float32)
    ok = bool(torch.allclose(o, r, rtol=1e-3, atol=1e-3))
    return ok, float((o - r).abs().max())


def run_cfg(args, cfg):
    k = helion.kernel(cross_entropy.fn, configs=[helion.Config(**dict(cfg))])
    b = k.bind(args)
    b.ensure_config_exists(args)
    tcode = b.to_triton_code(helion.Config(**dict(b._config)))
    out = b(*args)
    out = out[0] if isinstance(out, tuple) else out
    return b, out, tcode


def measure_shape(shape):
    args = build_args(shape)
    ref = reference(*args)

    # --- seed (heuristic) ---
    seed = get_seed(args)
    bound_s, out_s, tcode = run_cfg(args, seed)
    want_looped = bool(dict(bound_s._config).get("reduction_loops", [None])[0])
    got_looped = "for roffset" in tcode
    seed_used = want_looped == got_looped
    ok_s, err_s = check_correct(out_s, ref)
    assert ok_s, f"seed correctness FAIL {shape} err={err_s}"
    assert seed_used, f"seed NOT used {shape}"
    seed_lat = median_do_bench(lambda: bound_s(*args))

    # --- default_config (un-seeded baseline) ---
    bound_d0 = helion.kernel(cross_entropy.fn).bind(args)
    cfg_d = bound_d0.config_spec.default_config()
    bound_d, out_d, _ = run_cfg(args, cfg_d)
    ok_d, err_d = check_correct(out_d, ref)
    assert ok_d, f"default correctness FAIL {shape} err={err_d}"
    default_lat = median_do_bench(lambda: bound_d(*args))

    # --- persistent / fixed-w32 A/B alternative ---
    p32_cfg = {"block_sizes": dict(seed)["block_sizes"],
               "reduction_loops": [None], "num_warps": 32, "num_stages": 1}
    bound_p, out_p, _ = run_cfg(args, p32_cfg)
    ok_p, _ = check_correct(out_p, ref)
    assert ok_p, f"p32 correctness FAIL {shape}"
    p32_lat = median_do_bench(lambda: bound_p(*args))

    # --- torch.compile default mode of the fp32 reference ---
    torch._dynamo.reset()
    tc = torch.compile(reference)
    out_tc = tc(*args)
    ok_tc, _ = check_correct(out_tc, ref)
    assert ok_tc, f"tc correctness FAIL {shape}"
    tc_lat = median_do_bench(lambda: tc(*args))

    return {
        "shape": shape, "seed": dict(seed),
        "seed_lat_us": seed_lat * 1000, "default_lat_us": default_lat * 1000,
        "p32_lat_us": p32_lat * 1000, "tc_lat_us": tc_lat * 1000,
        "g_seed": tc_lat / seed_lat, "g_default": tc_lat / default_lat,
        "seed_over_p32": p32_lat / seed_lat,
        "codegen": "looped" if want_looped else "persistent",
        "err": err_s,
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  kernel=cross_entropy\n")
    header = (f"{'shape':>15} {'codegen':>10} {'warps':>5} "
              f"{'seed_us':>9} {'dflt_us':>9} {'p32_us':>9} {'tc_us':>9} "
              f"{'G_seed':>7} {'G_dflt':>7} {'s/p32':>6} {'maxabs':>8}")
    print(header)
    print("-" * len(header))
    g_seeds, g_defaults, sp32 = [], [], []
    for shape in IN_SAMPLE:
        r = measure_shape(shape)
        g_seeds.append(r["g_seed"])
        g_defaults.append(r["g_default"])
        sp32.append(r["seed_over_p32"])
        print(f"{str(r['shape']):>15} {r['codegen']:>10} "
              f"{r['seed']['num_warps']:>5} "
              f"{r['seed_lat_us']:>9.1f} {r['default_lat_us']:>9.1f} "
              f"{r['p32_lat_us']:>9.1f} {r['tc_lat_us']:>9.1f} "
              f"{r['g_seed']:>7.3f} {r['g_default']:>7.3f} {r['seed_over_p32']:>6.3f} "
              f"{r['err']:>8.1e}")

    def geomean(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs))

    print("-" * len(header))
    print(f"GEOMEAN  G_seed = {geomean(g_seeds):.4f}   "
          f"G_default = {geomean(g_defaults):.4f}   "
          f"seed/p32 = {geomean(sp32):.4f}")


if __name__ == "__main__":
    main()
