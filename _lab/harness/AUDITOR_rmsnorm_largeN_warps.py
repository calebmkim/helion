"""AUDITOR: does the REAL rms_norm (num_load==2) want w32 at LARGE rnumel?

The worker's claim: num_load>=2 reductions do NOT want w32 ("high warps hurt
them"), so the gate excludes them. The worker's rms_norm A/B only tested
in-sample shapes (rnumel<=16384). My synthetic red2 showed num_load==2 DOES
want w32 at large rnumel (131072+). This tests the REAL rms_norm at large rnumel
(held-out, beyond in-sample) to see whether the gate would deny rms_norm a real
w32 win there.

If rms_norm at large rnumel wants w32 (w32/w16 < 1), the worker's mechanism
claim is FALSE in general and the num_load gate is leaving perf on the table
(over-conservative), even if it happens to be right for rms_norm's in-sample.
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

EPS = 1e-5
N_RUNS = 9
WARPS = [16, 32]
# large-rnumel rms_norm, held out (in-sample max rnumel was 16384). Keep M small
# so the row is the dominant cost (matches the long_sum regime where w32 wins).
SHAPES = [(1, 32768), (1, 65536), (1, 131072), (1, 262144),
          (16, 131072), (16, 262144)]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def time_cfg(args, ref, warps):
    x, w, eps = args
    cfg = helion.Config(block_sizes=[1], reduction_loops=[None],
                        num_warps=warps, num_stages=1)
    k = helion.kernel(rms_norm_fwd.fn, configs=[cfg])
    b = k.bind(args)
    b.ensure_config_exists(args)
    tcode = b.to_triton_code(helion.Config(**dict(b._config)))
    assert "for roffset" not in tcode, "expected persistent codegen"
    out = b(*args)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-4), "correctness"
    return med(lambda: b(*args)) * 1000


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    bound = rms_norm_fwd.bind((torch.randn(1, 4096, device="cuda", dtype=torch.float32),
                               torch.randn(4096, device="cuda", dtype=torch.float32), EPS))
    nl = bound.env.config_spec.reduction_facts[0].num_load
    print(f"GPU={gpu}  helion={helion.__file__}  N_RUNS={N_RUNS}  rms_norm num_load={nl} (held-out large rnumel)\n")
    print(f"{'shape':>14} {'rnumel':>8} | {'w16':>8} {'w32':>8} | best | w32/w16  gate_gives")
    for shape in SHAPES:
        m, n = shape
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        w = torch.randn(n, device="cuda", dtype=torch.float32)
        args = (x, w, EPS)
        ref = rms_norm_pytorch(x, w, EPS)
        t = {wp: time_cfg(args, ref, wp) for wp in WARPS}
        best_w = min(t, key=t.get)
        ratio = t[32] / t[16]
        # gate: num_load<=1 AND rnumel>16384 -> w32, else ramp (>=4096 -> w16)
        gate_w = 32 if (nl <= 1 and n > 16384) else 16
        print(f"{str(shape):>14} {n:>8} | {t[16]:>8.2f} {t[32]:>8.2f} | w{best_w:>2} | {ratio:.3f}   w{gate_w}")


if __name__ == "__main__":
    main()
