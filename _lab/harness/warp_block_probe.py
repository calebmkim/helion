"""The smoking gun: for rms_norm (32768,256) fp32, num_warps=32 is catastrophic
ONLY at block_sizes=[1]. Larger M-blocks give each program many rows, so 32
warps have enough work and the config is FAST (~37-74us) -- which is the 73.6us
the oracle recorded. Probe the warps x block_size grid (persistent), fair do_bench.
"""

from __future__ import annotations

import functools
import os
import sys

import torch

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
sys.path.insert(0, WT)

import helion  # noqa: E402

assert helion.__file__.startswith(WT)

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402
from triton.testing import do_bench  # noqa: E402

EPS = 1e-6
M, N = 32768, 256
BLOCKS = [1, 2, 8, 32, 128]
WARPS = [4, 16, 32]


def fair_med(fn, reps=5):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(reps))[reps // 2]


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32, device="cuda")
    weight = torch.ones(N, dtype=torch.float32, device="cuda")
    ref = rms_norm_pytorch(x, weight, EPS)

    print(f"=== warps x block_size grid ({M},{N}) fp32 persistent GPU={gpu} ===")
    print("    (fair triton do_bench median, us; corr=max_abs vs ref)\n")

    print(f"{'block':>6} | " + " | ".join(f"w{w:>2}" for w in WARPS))
    grid = {}
    for b in BLOCKS:
        cells = []
        for w in WARPS:
            seed = helion.Config(
                reduction_loops=[None], block_sizes=[b], num_warps=w, num_stages=1
            )
            try:
                kern = helion.kernel(rms_norm_fwd.fn, configs=[seed])
                bound = kern.bind((x, weight, EPS))
                cfg = bound._config if getattr(bound, "_config", None) is not None else seed
                compiled = bound.compile_config(cfg, allow_print=False)
                call = functools.partial(compiled, x, weight, EPS)
                out = compiled(x, weight, EPS)
                out0 = out[0] if isinstance(out, tuple) else out
                ma = float((out0.float() - ref.float()).abs().max())
                us = fair_med(call) * 1000
                # report normalized block actually used
                nb = dict(cfg).get("block_sizes")
                cells.append(f"{us:7.1f}({ma:.0e})")
                grid[(b, w)] = (us, ma, nb)
            except Exception as e:
                cells.append(f" ERR:{type(e).__name__}")
                grid[(b, w)] = (None, None, str(e)[:40])
        print(f"{b:>6} | " + " | ".join(cells))

    print("\nNormalized block_sizes actually used (after config_spec.normalize):")
    for b in BLOCKS:
        nbs = {str(grid[(b, w)][2]) for w in WARPS}
        print(f"  requested block={b} -> {nbs}")


if __name__ == "__main__":
    main()
