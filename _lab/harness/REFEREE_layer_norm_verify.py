"""INDEPENDENT results-referee verification for layer_norm_fwd addition.

Written from scratch by the referee (does NOT call the worker's measure script).
Verifies the three claims:
  1. G_layer_norm (geomean tc_default/seed) > G_default (geomean tc_default/un-seeded-default)
     on the 11 in-sample shapes, beyond noise.
  2. Heuristic FIRES (num_reduction_ops==2): exactly 1 seed for layer_norm_fwd,
     seed USED (codegen persistent + matching num_warps), NOT a default fallback.
  3. Correctness PASS vs F.layer_norm (fp32) on all 11.

Fresh inputs per shape; median-of-N do_bench; spread reported.
Run with the canonical invocation (CUDA_VISIBLE_DEVICES=2, worktree PYTHONPATH).
"""

from __future__ import annotations

import math
import os
import sys
from statistics import median
from statistics import pstdev

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.layer_norm import layer_norm_fwd  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
N_RUNS = 9  # odd; referee uses more samples than worker (7)
RTOL = 1e-3
ATOL = 1e-3

IN_SAMPLE = [
    (4096, 1024), (4096, 2048), (4096, 4096), (4096, 8192),
    (4096, 12288), (4096, 15872),
    (2048, 3584), (2048, 8192),
    (8192, 4096), (8192, 5120), (8192, 7168),
]


def build_args(shape, seed_val):
    m, n = shape
    g = torch.Generator(device="cuda").manual_seed(seed_val)
    x = torch.randn(m, n, device="cuda", dtype=torch.float32, generator=g)
    w = torch.randn(n, device="cuda", dtype=torch.float32, generator=g)
    b = torch.randn(n, device="cuda", dtype=torch.float32, generator=g)
    return (x, [n], w, b, EPS)


def reference(x, normalized_shape, w, b, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, w, b, eps)


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    samples.sort()
    return median(samples), min(samples), max(samples), pstdev(samples)


def get_seeds(args):
    bound = layer_norm_fwd.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    # also pull num_reduction_ops to confirm =2
    facts = bound.env.config_spec.reduction_facts
    nrops = None
    if facts:
        f = facts[0]
        nrops = getattr(f, "num_reduction_ops", None)
    return seeds, bound, nrops


def check_correct(out, ref):
    o = out.to(torch.float32)
    r = ref.to(torch.float32)
    ok = bool(torch.allclose(o, r, rtol=RTOL, atol=ATOL))
    return ok, float((o - r).abs().max())


def looped_signature(code):
    return "for roffset" in code


def num_warps_in_code(code):
    import re
    m = re.search(r"num_warps=(\d+)", code)
    return int(m.group(1)) if m else None


def measure_shape(shape, seed_val):
    args = build_args(shape, seed_val)
    ref = reference(*args)

    seeds, _, nrops = get_seeds(args)
    assert len(seeds) == 1, f"{shape}: expected EXACTLY 1 seed, got {len(seeds)}: {seeds}"
    seed = seeds[0]
    seed_d = dict(seed)

    # --- bare seed: configs=[seed] => len==1 short-circuit, NO autotune ---
    seeded = helion.kernel(layer_norm_fwd.fn, configs=[helion.Config(**seed_d)])
    bound_s = seeded.bind(args)
    bound_s.ensure_config_exists(args)
    norm = dict(bound_s._config)
    code = bound_s.to_triton_code(helion.Config(**norm))

    want_looped = bool(norm.get("reduction_loops", [None])[0])
    got_looped = looped_signature(code)
    want_warps = norm.get("num_warps")
    got_warps = num_warps_in_code(code)
    seed_used = (want_looped == got_looped) and (want_warps == got_warps)

    out_s = bound_s(*args)
    out_s = out_s[0] if isinstance(out_s, tuple) else out_s
    ok_s, err_s = check_correct(out_s, ref)

    # Compare seed vs the UN-seeded default to confirm it's NOT a default fallback:
    default_k = helion.kernel(layer_norm_fwd.fn)
    bound_dprobe = default_k.bind(args)
    cfg_d = dict(bound_dprobe.config_spec.default_config())
    seed_differs_from_default = (
        seed_d.get("reduction_loops") != cfg_d.get("reduction_loops")
        or seed_d.get("num_warps") != cfg_d.get("num_warps")
        or seed_d.get("block_sizes") != cfg_d.get("block_sizes")
    )

    seed_lat, smin, smax, sstd = median_do_bench(lambda: bound_s(*args))

    # --- un-seeded default config (pre-heuristic baseline) ---
    default_k2 = helion.kernel(layer_norm_fwd.fn, configs=[bound_dprobe.config_spec.default_config()])
    bound_d2 = default_k2.bind(args)
    bound_d2.ensure_config_exists(args)
    code_d = bound_d2.to_triton_code(helion.Config(**dict(bound_d2._config)))
    out_d = bound_d2(*args)
    out_d = out_d[0] if isinstance(out_d, tuple) else out_d
    ok_d, err_d = check_correct(out_d, ref)
    default_lat, dmin, dmax, dstd = median_do_bench(lambda: bound_d2(*args))

    # --- torch.compile default of the fp32 reference ---
    torch._dynamo.reset()
    tc = torch.compile(reference)
    out_tc = tc(*args)
    ok_tc, err_tc = check_correct(out_tc, ref)
    tc_lat, tcmin, tcmax, tcstd = median_do_bench(lambda: tc(*args))

    return {
        "shape": shape,
        "nrops": nrops,
        "seed": seed_d,
        "norm_redloops": norm.get("reduction_loops"),
        "codegen": "looped" if got_looped else "persistent",
        "default_codegen": "looped" if looped_signature(code_d) else "persistent",
        "seed_warps": want_warps,
        "default_warps": cfg_d.get("num_warps"),
        "seed_used": seed_used,
        "seed_differs": seed_differs_from_default,
        "ok_s": ok_s, "err_s": err_s, "ok_d": ok_d, "ok_tc": ok_tc,
        "seed_lat": seed_lat * 1000, "seed_spread": (smax - smin) / seed_lat * 100,
        "default_lat": default_lat * 1000, "default_spread": (dmax - dmin) / default_lat * 100,
        "tc_lat": tc_lat * 1000, "tc_spread": (tcmax - tcmin) / tc_lat * 100,
        "g_seed": tc_lat / seed_lat, "g_default": tc_lat / default_lat,
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"REFEREE GPU={gpu} helion={helion.__file__}")
    print(f"N_RUNS={N_RUNS} rtol={RTOL} atol={ATOL} dtype=fp32 WITH bias\n")
    hdr = (f"{'shape':>13} {'nr':>2} {'cg':>10} {'dcg':>10} {'sw':>3} {'dw':>3} "
           f"{'used':>4} {'diff':>4} {'ok':>2} "
           f"{'seed_us':>8}({'sp%':>4}) {'dflt_us':>8}({'sp%':>4}) {'tc_us':>8}({'sp%':>4}) "
           f"{'Gseed':>6} {'Gdflt':>6} {'err':>8}")
    print(hdr)
    print("-" * len(hdr))
    gs, gd = [], []
    all_used, all_correct, all_one_seed, all_persistent = True, True, True, True
    for shape in IN_SAMPLE:
        r = measure_shape(shape, seed_val=1234)
        gs.append(r["g_seed"]); gd.append(r["g_default"])
        all_used &= r["seed_used"]
        all_correct &= (r["ok_s"] and r["ok_d"] and r["ok_tc"])
        all_persistent &= (r["codegen"] == "persistent")
        if r["nrops"] != 2:
            print(f"  WARN: {shape} num_reduction_ops={r['nrops']} (expected 2)")
        okstr = "P" if (r["ok_s"] and r["ok_d"] and r["ok_tc"]) else "F"
        print(f"{str(r['shape']):>13} {str(r['nrops']):>2} {r['codegen']:>10} "
              f"{r['default_codegen']:>10} {r['seed_warps']:>3} {r['default_warps']:>3} "
              f"{('Y' if r['seed_used'] else 'N'):>4} {('Y' if r['seed_differs'] else 'N'):>4} "
              f"{okstr:>2} "
              f"{r['seed_lat']:>8.1f}({r['seed_spread']:>4.1f}) "
              f"{r['default_lat']:>8.1f}({r['default_spread']:>4.1f}) "
              f"{r['tc_lat']:>8.1f}({r['tc_spread']:>4.1f}) "
              f"{r['g_seed']:>6.3f} {r['g_default']:>6.3f} {r['err_s']:>8.1e}")

    def geomean(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs))

    print("-" * len(hdr))
    print(f"GEOMEAN  G_seed={geomean(gs):.4f}  G_default={geomean(gd):.4f}  "
          f"ratio={geomean(gs)/geomean(gd):.4f}")
    print(f"\nSUMMARY: all_one_seed={all_one_seed} all_seed_used={all_used} "
          f"all_persistent={all_persistent} all_correct={all_correct}")


if __name__ == "__main__":
    main()
