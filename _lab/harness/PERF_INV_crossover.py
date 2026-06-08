"""Perf-investigator: find the rnumel crossover where the M-block lever flips.

For a fixed kernel (rms_norm, num_load=2) at LARGE M (32768 rows, so the grid is
big and per-row overhead matters), sweep rnumel and for each measure the best
M-block (warps held at the seed's rnumel-ramp value). Goal: pin the WORKLOAD
boundary -- below which "pack more rows/program" wins, above which it regresses
(register pressure: mblock*rnumel live elems spill).

Reports, per rnumel: seed(mblock=floor) G, and the best mblock + its G, and the
mblock*rnumel live-numel at the win. This tells us whether the discriminator is
"small rnumel" or "small mblock*rnumel (live tile)".
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
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 5
MBLOCKS = [1, 2, 4, 8, 16, 32, 64]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def build_rms(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), 1e-5)


def build_x(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


REG = {
    "rms_norm": (rms_norm_fwd, build_rms, lambda a: rms_norm_pytorch(*a),
                 lambda: torch.compile(rms_norm_pytorch)),
    "sum": (sum_kernel, build_x, lambda a: torch.sum(a[0], -1),
            lambda: torch.compile(lambda x: torch.sum(x, -1))),
}

# Hold M=32768 (big grid), sweep rnumel across the crossover region.
RNUMELS = [256, 512, 1024, 2048, 4096, 8192]
M_FIXED = 32768


def correct(out, ref):
    o = out[0] if isinstance(out, tuple) else out
    o = o.float(); r = ref.float()
    return bool(o.shape == r.shape and torch.allclose(o, r, rtol=1e-3, atol=1e-3))


def main():
    kernel = os.environ.get("INV_KERNEL", "rms_norm")
    fn, build, reffn, tcfn = REG[kernel]
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  kernel={kernel}  M_FIXED={M_FIXED}  helion={helion.__file__}\n")
    print(f"  {'rnumel':>7} {'seedmb':>6} {'seedG':>6} {'seedW':>5} | "
          f"{'bestmb':>6} {'bestG':>6} {'gain%':>6} {'liveN':>9}")
    for rn in RNUMELS:
        shape = (M_FIXED, rn)
        args = build(shape)
        ref = reffn(args)
        torch._dynamo.reset()
        tc = tcfn(); _ = tc(*args)
        tc_lat = med(lambda: tc(*args))
        bound0 = fn.bind(args)
        seed = dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])
        floor = seed["block_sizes"][0]
        seedW = seed["num_warps"]
        # seed G
        ks = helion.kernel(fn.fn, configs=[helion.Config(**seed)])
        bs = ks.bind(args); bs.ensure_config_exists(args); _ = bs(*args)
        seed_lat = med(lambda: bs(*args))
        seedG = tc_lat / seed_lat
        # best mblock (warps held at seed)
        best = (floor, seedG)
        for mb in MBLOCKS:
            if mb <= floor:
                continue
            cfg = dict(seed)
            cfg["block_sizes"] = [mb] + list(seed["block_sizes"][1:])
            try:
                k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
                b = k.bind(args); b.ensure_config_exists(args)
                out = b(*args)
                if not correct(out, ref):
                    continue
                lat = med(lambda: b(*args))
                g = tc_lat / lat
                if g > best[1]:
                    best = (mb, g)
            except Exception:
                continue
        gain = (best[1] / seedG - 1) * 100
        print(f"  {rn:>7} {floor:>6} {seedG:>6.3f} {seedW:>5} | "
              f"{best[0]:>6} {best[1]:>6.3f} {gain:>6.1f} {best[0]*rn:>9}")


if __name__ == "__main__":
    main()
