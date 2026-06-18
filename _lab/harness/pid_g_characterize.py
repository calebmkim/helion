"""Where does the grid-bound small-N headroom actually come from? (pid is NOT it.)

For the grid-bound small-N shapes the brief flags (sum/rms_norm 8192x256, 32768x256),
measure G = tc_default_lat / lat for:
  - v6           : the current champion seed (persistent, flat pid)
  - oracle_flat  : the oracle lever bundle (block=4/looped-128/w1/s5) with FLAT pid
  - oracle_pi2   : the SAME bundle with pid=persistent_interleaved + num_sm_multiplier=2
  - v6_pi2       : the v6 seed + pid=persistent_interleaved + num_sm_multiplier=2 (the
                   brief's proposed Stage-1 branch, matched-lever)

If oracle_flat >= oracle_pi2 and v6 >= v6_pi2, then pid_type contributes nothing
positive; the headroom (if any) is the other levers (M-block/looped-chunk/warps).
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

from examples.sum import sum_kernel  # noqa: E402
from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
import helion.runtime as rt  # noqa: E402

N_RUNS = 7
NUM_SM = rt.get_num_sm(torch.device("cuda"))


def median_do_bench(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def sum_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def sum_ref(args):
    return torch.sum(args[0], dim=-1)


def rms_args(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, 1e-5)


def rms_ref(args):
    return rms_norm_pytorch(args[0], args[1], args[2])


def check(out, ref):
    out = out[0] if isinstance(out, tuple) else out
    return bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))


def bench_cfg(fn, args, cfg, ref):
    k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args); b.ensure_config_exists(args)
    out = b(*args)
    assert check(out, ref), f"INCORRECT {cfg}"
    return median_do_bench(lambda: b(*args)) * 1000


def v6_seed(fn, args):
    bound = fn.bind(args)
    return dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])


CASES = [
    ("sum", sum_kernel, sum_args, sum_ref, [(8192, 256), (32768, 256)]),
    ("rms_norm", rms_norm_fwd, rms_args, rms_ref, [(8192, 256), (32768, 256)]),
]
ORACLE_BUNDLE = {"block_sizes": [4], "reduction_loops": [128], "num_warps": 1, "num_stages": 5}


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} NUM_SM={NUM_SM}\n")
    hdr = f"{'kernel/shape':>20} {'tc_us':>8} {'v6_us':>8} {'G_v6':>6} {'v6pi2':>7} {'G':>6} {'orFlat':>7} {'G':>6} {'orPI2':>7} {'G':>6}"
    print(hdr); print("-" * len(hdr))
    for name, fn, af, rf, shapes in CASES:
        for (m, n) in shapes:
            args = af(m, n); ref = rf(args)
            torch._dynamo.reset()
            tc = torch.compile(lambda *a: rf(a))
            _ = tc(*args)
            tc_lat = median_do_bench(lambda: tc(*args)) * 1000
            v6 = v6_seed(fn, args)
            l_v6 = bench_cfg(fn, args, v6, ref)
            l_v6pi = bench_cfg(fn, args, {**v6, "pid_type": "persistent_interleaved", "num_sm_multiplier": 2}, ref)
            l_orf = bench_cfg(fn, args, ORACLE_BUNDLE, ref)
            l_orpi = bench_cfg(fn, args, {**ORACLE_BUNDLE, "pid_type": "persistent_interleaved", "num_sm_multiplier": 2}, ref)
            print(f"{f'{name} {m}x{n}':>20} {tc_lat:>8.1f} {l_v6:>8.1f} {tc_lat/l_v6:>6.2f} "
                  f"{l_v6pi:>7.1f} {tc_lat/l_v6pi:>6.2f} {l_orf:>7.1f} {tc_lat/l_orf:>6.2f} "
                  f"{l_orpi:>7.1f} {tc_lat/l_orpi:>6.2f}")


if __name__ == "__main__":
    main()
