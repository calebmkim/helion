"""Oracle + field-diff for layer_norm_fwd.

For a subset of in-sample shapes: run Helion quick-autotune to get a winning
config (the oracle), re-bench the FULL VERBATIM oracle config fairly (all levers
together), and compare to the heuristic seed. Reports tc_default_lat, seed_lat,
oracle_lat, G_seed, G_oracle, and the lever field-diff.

Quick effort + a FAIR re-bench of the full verbatim winner (harness-integrity
guidance). Reference = torch.nn.functional.layer_norm fp32, output only.

Run with the canonical invocation (see SETUP.md).
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
LEVERS = ["block_sizes", "reduction_loops", "num_warps", "num_stages"]

# Representative subset: the widest in-sample rows (where the persistent lever
# matters most), plus a couple mid rows and a small one.
SHAPES = [(4096, 1024), (4096, 4096), (4096, 8192), (4096, 15872),
          (2048, 8192), (8192, 7168)]


def build_args(shape):
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, [n], w, b, EPS)


def reference(x, ns, w, b, eps):
    return torch.nn.functional.layer_norm(x, ns, w, b, eps)


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  kernel=layer_norm_fwd (with bias)\n")

    g_seeds, g_oracles = [], []
    for shape in SHAPES:
        args = build_args(shape)
        ref = reference(*args)

        # seed
        bound = layer_norm_fwd.bind(args)
        seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
        seeded = helion.kernel(layer_norm_fwd.fn, configs=[helion.Config(**seed)])
        bs = seeded.bind(args)
        bs.ensure_config_exists(args)
        seed_lat = med(lambda: bs(*args))

        # tc default
        torch._dynamo.reset()
        tc = torch.compile(reference)
        tc(*args)
        tc_lat = med(lambda: tc(*args))

        # oracle: quick autotune; re-bench the FULL verbatim winner
        os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"
        os.environ["HELION_FORCE_AUTOTUNE"] = "1"
        k = helion.kernel(layer_norm_fwd.fn)
        bound_o = k.bind(args)
        bound_o.autotune(args)
        oracle_cfg = dict(bound_o._config)
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

    print(f"GEOMEAN(subset)  G_seed={geomean(g_seeds):.4f}  G_oracle={geomean(g_oracles):.4f}")


if __name__ == "__main__":
    main()
