"""Oracle + field-diff for a row-reduction kernel (sum / long_sum).

For each in-sample shape: run Helion quick-autotune to get a winning config
(the oracle), then re-bench the FULL VERBATIM oracle config fairly (all levers
together — NEVER an isolated lever; see oracle_field_diff.py guard) and compare to
the heuristic seed. Reports tc_default_lat, seed_lat, oracle_lat, G_seed,
G_oracle, and the lever field-diff (block_sizes, reduction_loops, num_warps,
num_stages).

Quick effort (not full) per harness-integrity guidance: quick + a FAIR re-bench of
the FULL verbatim winner is the trustworthy path (full is slow and unnecessary
here). The winner that runs is bound._config — we re-bench exactly that.
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
LEVERS = ["block_sizes", "reduction_loops", "num_warps", "num_stages"]

KERNELS = {
    "sum": {"fn": sum_kernel,
            "shapes": [(2048, 4096), (4096, 5120), (8192, 4096), (32768, 256)]},
    "long_sum": {"fn": longsum,
                 "shapes": [(1, 32768), (2, 65536), (4, 130000), (8, 131072),
                            (16, 262144)]},
}


def build_args(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True, choices=list(KERNELS))
    a = ap.parse_args()
    spec = KERNELS[a.kernel]
    fn = spec["fn"]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  kernel={a.kernel}\n")

    g_seeds, g_oracles = [], []
    for shape in spec["shapes"]:
        args = build_args(shape)
        x = args[0]
        ref = torch.sum(x, dim=-1)

        # seed
        bound = fn.bind(args)
        seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
        seeded = helion.kernel(fn.fn, configs=[helion.Config(**seed)])
        bs = seeded.bind(args)
        bs.ensure_config_exists(args)
        seed_lat = med(lambda: bs(*args))

        # tc default
        torch._dynamo.reset()
        tc = torch.compile(lambda t: torch.sum(t, dim=-1))
        tc(x)
        tc_lat = med(lambda: tc(x))

        # oracle: quick autotune; re-bench the FULL verbatim winner
        os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"
        os.environ["HELION_FORCE_AUTOTUNE"] = "1"
        k = helion.kernel(fn.fn)
        bound_o = k.bind(args)
        bound_o.autotune(args)
        oracle_cfg = dict(bound_o._config)
        # GUARD: bench ONLY the full verbatim config that actually resolved.
        assert dict(bound_o._config) == oracle_cfg
        out_o = bound_o(*args)
        out_o = out_o[0] if isinstance(out_o, tuple) else out_o
        ok = bool(torch.allclose(out_o.float(), ref.float(), rtol=1e-3, atol=1e-3))
        oracle_lat = med(lambda: bound_o(*args))

        g_seed = tc_lat / seed_lat
        g_oracle = tc_lat / oracle_lat
        g_seeds.append(g_seed)
        g_oracles.append(g_oracle)

        print(f"=== {shape}  (oracle correct={ok}) ===")
        print(f"  tc={tc_lat*1000:.1f}us  seed={seed_lat*1000:.1f}us (G={g_seed:.3f})  "
              f"oracle={oracle_lat*1000:.1f}us (G={g_oracle:.3f})")
        print(f"  {'lever':>16} {'seed':>14} {'oracle':>22}")
        for lever in LEVERS:
            sv, ov = seed.get(lever), oracle_cfg.get(lever)
            mark = "" if sv == ov else "  <-- DIFF"
            print(f"  {lever:>16} {str(sv):>14} {str(ov):>22}{mark}")
        print()

    def geomean(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs))

    print(f"GEOMEAN  G_seed={geomean(g_seeds):.4f}  G_oracle={geomean(g_oracles):.4f}")


if __name__ == "__main__":
    main()
