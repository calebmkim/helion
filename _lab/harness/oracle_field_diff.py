"""Oracle field-diff: Helion full-autotune winning config vs the heuristic seed.

For representative shapes, run Helion effort=full to get the MAX winning config
(bound._config after autotune), then diff against the heuristic's seed on the
levers: reduction_loops (persistent vs looped), block_sizes (R/M), num_warps,
num_stages. Reports the oracle latency and G_oracle = tc_default_lat / oracle_lat.

Run with the canonical invocation (set CUDA_VISIBLE_DEVICES to a free GPU).
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
N_RUNS = 7
SHAPES = [(32768, 256), (4096, 5120), (2048, 16384)]
LEVERS = ["block_sizes", "reduction_loops", "num_warps", "num_stages"]


def build_args(shape):
    m, n = shape
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, EPS)


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}\n")
    for shape in SHAPES:
        args = build_args(shape)
        x, w, e = args
        ref = rms_norm_pytorch(x, w, e)

        # seed
        bound = rms_norm_fwd.bind(args)
        seed = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]

        # tc default
        torch._dynamo.reset()
        tc = torch.compile(rms_norm_pytorch)
        tc(x, w, e)
        tc_lat = med(lambda: tc(x, w, e))

        # Helion full autotune (the oracle)
        os.environ["HELION_AUTOTUNE_EFFORT"] = "full"
        k = helion.kernel(rms_norm_fwd.fn)
        bound_o = k.bind(args)
        bound_o.autotune(args)  # runs the full search
        oracle_cfg = dict(bound_o._config)
        out_o = bound_o(*args)
        out_o = out_o[0] if isinstance(out_o, tuple) else out_o
        ok = torch.allclose(out_o.float(), ref.float(), rtol=1e-3, atol=1e-4)
        oracle_lat = med(lambda: bound_o(*args))

        print(f"=== shape {shape}  (correct={ok}) ===")
        print(f"  tc_default_lat = {tc_lat*1000:.1f}us")
        print(f"  oracle_lat     = {oracle_lat*1000:.1f}us  G_oracle={tc_lat/oracle_lat:.3f}")
        print(f"  {'lever':>16} {'seed':>16} {'oracle':>16}")
        sd = dict(seed)
        for lever in LEVERS:
            print(f"  {lever:>16} {str(sd.get(lever)):>16} {str(oracle_cfg.get(lever)):>16}")
        print(f"  full oracle config: {oracle_cfg}\n")


if __name__ == "__main__":
    main()
