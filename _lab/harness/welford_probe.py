"""welford: seed-used (codegen) + correctness + G + A/B vs alternatives.

welford = T2 (3 block_sizes, 0 reduction_loops, 1 reduction_fact,
num_tiled_accumulators=0). The heuristic seeds:
  block_sizes=[M_BLOCK(floor), R_BLOCK_welford=next_pow2(N), normalize_block(floor=1)]
The CONCERN: the SECOND tile_n loop (normalize pass, block_id 2) is NOT the
detected reduction axis, so it gets seeded at floor=1 -> N iterations of width 1.
This probe measures whether that wrecks perf, and A/Bs the seed against:
  - persistent-both : R_welford=N AND R_normalize=N (both passes whole-row)
  - default_config  : the un-seeded baseline
  - tc              : torch.compile(F.layer_norm)

In-sample shapes: huge M (262144), modest D.
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

from examples.welford import welford, eager_layer_norm  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 1e-5
N_RUNS = 5

IN_SAMPLE = [(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096)]


def build_args(m, n):
    w = torch.rand(n, device="cuda", dtype=torch.float32)
    b = torch.rand(n, device="cuda", dtype=torch.float32)
    x = torch.rand(m, n, device="cuda", dtype=torch.float32)
    return (w, b, x, EPS)


def reference(w, b, x, eps):
    return eager_layer_norm(w, b, x, eps)


def median_do_bench(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def get_seed(args):
    bound = welford.bind(args)
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    assert len(seeds) == 1, f"expected 1 seed got {len(seeds)}"
    return dict(seeds[0])


def run_cfg(args, cfg, ref):
    try:
        k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        out = b(*args)
        out = out[0] if isinstance(out, tuple) else out
        ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
        err = float((out.float() - ref.float()).abs().max())
        lat = median_do_bench(lambda: b(*args)) * 1000
        return lat, ok, err
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:50]}", None


def measure(m, n):
    args = build_args(m, n)
    ref = reference(*args)
    seed = get_seed(args)
    # codegen: count for-loops to see if normalize pass is persistent or looped
    k = helion.kernel(welford.fn, configs=[helion.Config(**seed)])
    bnd = k.bind(args)
    bnd.ensure_config_exists(args)
    code = bnd.to_triton_code(helion.Config(**dict(bnd._config)))
    nloops = code.count("for roffset") + code.count("for tile") + code.count("tl.range")

    seed_lat, ok_s, err_s = run_cfg(args, seed, ref)
    # default
    bd = helion.kernel(welford.fn).bind(args)
    cfg_d = bd.config_spec.default_config()
    dflt_lat, ok_d, _ = run_cfg(args, dict(cfg_d), ref)
    # persistent-both (R_normalize = N too)
    bs = list(seed["block_sizes"])
    bs_pboth = list(bs)
    bs_pboth[2] = np2(n)  # normalize block = full N
    pboth = dict(seed); pboth["block_sizes"] = bs_pboth
    pboth_lat, ok_p, _ = run_cfg(args, pboth, ref)
    # tc
    torch._dynamo.reset()
    tc = torch.compile(reference)
    tc(*args)
    tc_lat = median_do_bench(lambda: tc(*args)) * 1000

    return {
        "shape": (m, n), "seed": seed, "nloops": nloops,
        "seed_lat": seed_lat, "ok_s": ok_s, "err_s": err_s,
        "dflt_lat": dflt_lat, "ok_d": ok_d,
        "pboth_lat": pboth_lat, "pboth_bs": bs_pboth, "ok_p": ok_p,
        "tc_lat": tc_lat,
        "g_seed": (tc_lat / seed_lat) if seed_lat else None,
        "g_dflt": (tc_lat / dflt_lat) if dflt_lat else None,
        "g_pboth": (tc_lat / pboth_lat) if pboth_lat else None,
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  kernel=welford\n")
    gs, gd, gp = [], [], []
    for (m, n) in IN_SAMPLE:
        r = measure(m, n)
        print(f"=== {r['shape']} ===")
        print(f"  seed block_sizes={r['seed']['block_sizes']} "
              f"num_warps={r['seed']['num_warps']} (rl={r['seed'].get('reduction_loops')})")
        err_str = (f"{r['err_s']:.2e}" if isinstance(r['err_s'], float)
                   else str(r['err_s']))
        print(f"  seed   : {r['seed_lat'] and round(r['seed_lat'],1)} us  ok={r['ok_s']} "
              f"err={err_str} G={r['g_seed'] and round(r['g_seed'],3)}")
        print(f"  default: {r['dflt_lat'] and round(r['dflt_lat'],1)} us  ok={r['ok_d']} "
              f"G={r['g_dflt'] and round(r['g_dflt'],3)}")
        print(f"  p-both : {r['pboth_lat'] and round(r['pboth_lat'],1)} us  "
              f"(bs={r['pboth_bs']}) ok={r['ok_p']} G={r['g_pboth'] and round(r['g_pboth'],3)}")
        print(f"  tc     : {round(r['tc_lat'],1)} us")
        if r["g_seed"]:
            gs.append(r["g_seed"])
        if r["g_dflt"]:
            gd.append(r["g_dflt"])
        if r["g_pboth"]:
            gp.append(r["g_pboth"])
        print()

    def gm(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else None
    print(f"GEOMEAN  G_seed={gm(gs) and round(gm(gs),4)}  "
          f"G_default={gm(gd) and round(gm(gd),4)}  "
          f"G_pboth={gm(gp) and round(gm(gp),4)}")


if __name__ == "__main__":
    main()
