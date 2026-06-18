"""Measure Product-A G for kl_div / jsd (Band-B T2 loss kernels, fp32) + the
Band-B A/B: is the seed (persistent / rnumel-ramp-warps) the best SIMPLE config,
or does a sub-cap R_BLOCK and/or a different warp count beat it?

For each in-sample shape (all fp32, same do_bench, same inputs):
  - seed_lat   : heuristic bare seed (persistent R_BLOCK>=V, warps per rnumel ramp)
  - default_lat: un-seeded default_config baseline
  - tc_lat     : torch.compile default mode of the torch reference (loss kernels)
  - AB grid    : persistent (R_BLOCK>=V) at warps in {8,16,32} AND a sub-cap looped
                 R_BLOCK (V//4, capped pow2) at the seed's warps -- the candidate
                 Band-B levers (lower warps / sub-cap R_BLOCK) we must BEAT or
                 declare unnecessary. M_BLOCK pinned at floor (numel constraint).

Reports G_seed/G_default/tc and the best-AB config so we can decide if a Band-B
branch is warranted (only if a simple alternative beats the seed materially).

--kernel kl_div | jsd ; run with the canonical invocation (SETUP.md).
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

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion._utils import next_power_of_2  # noqa: E402

N_RUNS = 7

KL_SHAPES = [(4096, 4096), (4096, 8192), (4096, 16384), (4096, 32768),
             (4096, 65536), (4096, 131072)]
JSD_SHAPES = [(8192, 4096), (8192, 8192), (8192, 16384), (8192, 32768),
              (8192, 65536), (8192, 131072)]


def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


def setup(kernel):
    if kernel == "kl_div":
        from examples.kl_div import kl_div_forward as fn

        def inputs(BT, V):
            yp = torch.randn(BT, V, device="cuda", dtype=torch.float32).log_softmax(-1)
            yt = torch.randn(BT, V, device="cuda", dtype=torch.float32).softmax(-1)
            return (yp, yt, False, "batchmean", 1e-10)

        def ref(args):
            yp, yt = args[0], args[1]
            return torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to(
                "cuda")(yp, yt)

        def out_loss(o):
            return o
        return fn, inputs, ref, out_loss, KL_SHAPES
    else:
        from examples.jsd import jsd_forward as fn
        from examples.jsd import TorchJSDBaseline

        def inputs(BT, V):
            lq = torch.randn(BT, V, device="cuda", dtype=torch.float32).log_softmax(-1)
            lp = torch.randn(BT, V, device="cuda", dtype=torch.float32).log_softmax(-1)
            return (lq, lp, None, 0.5, -100)

        baseline = TorchJSDBaseline(beta=0.5, ignore_index=-100)

        def ref(args):
            return baseline(args[0], args[1])

        def out_loss(o):
            return o[0] if isinstance(o, tuple) else o
        return fn, inputs, ref, out_loss, JSD_SHAPES


def build(fn, seed_dict, args):
    k = helion.kernel(fn.fn, configs=[helion.Config(**seed_dict)])
    b = k.bind(args); b.ensure_config_exists(args)
    return b


def relerr(a, b):
    return abs(float(a) - float(b)) / (abs(float(b)) + 1e-12)


def measure(kernel):
    fn, inputs, ref, out_loss, shapes = setup(kernel)
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} kernel={kernel} (Band-B T2 fp32)\n")
    header = (f"{'shape':>14} {'sw':>3} {'seed_us':>9} {'dflt_us':>9} {'tc_us':>9} "
              f"{'G_seed':>7} {'G_dflt':>7} {'bestAB':>16} {'bestAB_us':>9} "
              f"{'s/AB':>6} {'relerr':>8}")
    print(header); print("-" * len(header))
    gss, gds = [], []
    for (BT, V) in shapes:
        args = inputs(BT, V)
        r = ref(args)
        bound0 = fn.bind(args)
        seed = dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])
        bs = list(seed["block_sizes"])
        # find the reduction-axis index (the one set to >= V)
        red_idx = max(range(len(bs)), key=lambda i: bs[i])
        sw = seed["num_warps"]
        # seed
        bsd = build(fn, seed, args)
        ok = relerr(out_loss(bsd(*args)), r) < 1e-3
        assert ok, f"seed correctness FAIL {kernel} {(BT,V)}"
        seed_lat = median_do_bench(lambda: bsd(*args))
        # default
        cfgd = dict(bound0.config_spec.default_config())
        bdd = build(fn, cfgd, args)
        default_lat = median_do_bench(lambda: bdd(*args))
        # tc
        torch._dynamo.reset()
        tc = torch.compile(lambda *a: ref(a))
        tc_lat = median_do_bench(lambda: tc(*args))
        # AB grid: persistent at warps {8,16,32}; plus a sub-cap looped R_BLOCK
        ab = {}
        full_R = next_power_of_2(V)
        for w in (8, 16, 32):
            cfg = dict(seed); cfg["num_warps"] = w
            cfg["block_sizes"] = list(bs); cfg["block_sizes"][red_idx] = full_R
            try:
                b = build(fn, cfg, args)
                if relerr(out_loss(b(*args)), r) < 1e-3:
                    ab[f"persist/w{w}"] = median_do_bench(lambda b=b: b(*args))
            except Exception:
                pass
        for frac in (4,):  # sub-cap R_BLOCK = full_R/frac (looped)
            sub_R = max(1, next_power_of_2(full_R // frac))
            cfg = dict(seed); cfg["num_warps"] = sw
            cfg["block_sizes"] = list(bs); cfg["block_sizes"][red_idx] = sub_R
            try:
                b = build(fn, cfg, args)
                if relerr(out_loss(b(*args)), r) < 1e-3:
                    ab[f"loop{sub_R}/w{sw}"] = median_do_bench(lambda b=b: b(*args))
            except Exception:
                pass
        best_name = min(ab, key=ab.get)
        best_us = ab[best_name] * 1000
        g_seed = tc_lat / seed_lat; g_def = tc_lat / default_lat
        gss.append(g_seed); gds.append(g_def)
        print(f"{str((BT,V)):>14} {sw:>3} {seed_lat*1000:>9.1f} "
              f"{default_lat*1000:>9.1f} {tc_lat*1000:>9.1f} {g_seed:>7.3f} "
              f"{g_def:>7.3f} {best_name:>16} {best_us:>9.1f} "
              f"{(ab[best_name]/seed_lat):>6.3f} {relerr(out_loss(bsd(*args)),r):>8.1e}")

    def geomean(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs))
    print("-" * len(header))
    print(f"GEOMEAN  G_seed={geomean(gss):.4f}  G_default={geomean(gds):.4f}")
    print("(s/AB <1 => seed faster than the best simple alternative; "
          ">1 => an AB config beats the seed -> candidate Band-B lever)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=["kl_div", "jsd"], required=True)
    a = ap.parse_args()
    measure(a.kernel)


if __name__ == "__main__":
    main()
