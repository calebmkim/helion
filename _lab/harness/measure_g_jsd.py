"""Measure G_jsd (Band-B T2) with the Band-B branch active. The seed is now the
CAPPED R_BLOCK config. Decisive A/B: seed vs a sweep of safe looped R_BLOCK
{2048,4096,8192} at the seed's warps (NOT the full-N persistent, which SPILLS
catastrophically for jsd -- that IS the regression the Band-B branch fixes, shown
separately in t2_bandb_chunk_sweep.py). Confirms the seed is at/near the best
simple looped alternative. fp32, M_BLOCK floor.
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

from examples.jsd import jsd_forward, TorchJSDBaseline  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7
SHAPES = [(8192, 4096), (8192, 8192), (8192, 16384), (8192, 32768),
          (8192, 65536), (8192, 131072)]


def median_do_bench(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def inputs(BT, V):
    return (torch.randn(BT, V, device="cuda").log_softmax(-1),
            torch.randn(BT, V, device="cuda").log_softmax(-1), None, 0.5, -100)


def relerr(a, b):
    return abs(float(a) - float(b)) / (abs(float(b)) + 1e-12)


def build(seed_dict, args):
    k = helion.kernel(jsd_forward.fn, configs=[helion.Config(**seed_dict)])
    b = k.bind(args); b.ensure_config_exists(args)
    return b


def main():
    baseline = TorchJSDBaseline(beta=0.5, ignore_index=-100)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} kernel=jsd (Band-B T2 fp32)\n")
    header = (f"{'shape':>14} {'sw':>3} {'seedR':>6} {'seed_us':>9} {'dflt_us':>9} "
              f"{'tc_us':>9} {'G_seed':>7} {'G_dflt':>7} {'bestAB':>14} "
              f"{'bestAB_us':>9} {'s/AB':>6} {'relerr':>8}")
    print(header); print("-" * len(header))
    gss, gds = [], []
    for (BT, V) in SHAPES:
        args = inputs(BT, V)
        ref = baseline(args[0], args[1])
        bound0 = jsd_forward.bind(args)
        seed = dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])
        bs = list(seed["block_sizes"]); red_idx = max(range(len(bs)), key=lambda i: bs[i])
        seedR = bs[red_idx]; sw = seed["num_warps"]
        bsd = build(seed, args)
        loss = bsd(*args)[0]
        assert relerr(loss, ref) < 1e-3, f"seed FAIL {(BT,V)}"
        seed_lat = median_do_bench(lambda: bsd(*args))
        cfgd = dict(bound0.config_spec.default_config())
        bdd = build(cfgd, args)
        default_lat = median_do_bench(lambda: bdd(*args))
        torch._dynamo.reset()
        tc = torch.compile(lambda *a: baseline(a[0], a[1]))
        tc_lat = median_do_bench(lambda: tc(*args))
        # safe looped AB sweep (NEVER full-N persistent for jsd wide rows)
        ab = {}
        for R in (2048, 4096, 8192):
            if R > V:
                continue
            cfg = dict(seed); cfg["block_sizes"] = list(bs); cfg["block_sizes"][red_idx] = R
            try:
                b = build(cfg, args)
                if relerr(b(*args)[0], ref) < 1e-3:
                    ab[f"R{R}/w{sw}"] = median_do_bench(lambda b=b: b(*args))
            except Exception:
                pass
        best = min(ab, key=ab.get); best_us = ab[best] * 1000
        g_seed = tc_lat / seed_lat; g_def = tc_lat / default_lat
        gss.append(g_seed); gds.append(g_def)
        print(f"{str((BT,V)):>14} {sw:>3} {seedR:>6} {seed_lat*1000:>9.1f} "
              f"{default_lat*1000:>9.1f} {tc_lat*1000:>9.1f} {g_seed:>7.3f} "
              f"{g_def:>7.3f} {best:>14} {best_us:>9.1f} "
              f"{(ab[best]/seed_lat):>6.3f} {relerr(loss,ref):>8.1e}")

    def geomean(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs))
    print("-" * len(header))
    print(f"GEOMEAN  G_seed={geomean(gss):.4f}  G_default={geomean(gds):.4f}")


if __name__ == "__main__":
    main()
