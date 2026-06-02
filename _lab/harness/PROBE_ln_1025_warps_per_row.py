"""Is the (2048,1025) M-block win really about WARPS-PER-ROW (cross-warp combine)?

Hypothesis: latency tracks how many warps must cooperate on ONE row's reduction.
- M=1 forces all `num_warps` onto the single row -> expensive cross-warp combine.
- M>=num_warps lets each row map to <=1 warp -> cheap intra-warp shuffle.

So sweep (M_BLOCK, num_warps) points chosen to vary warps-per-row independently:
  pure intra-warp (w small / M large), vs forced cross-warp (M=1, w large).
If the theory holds: M1/w1 (no cross-warp) should be ~as fast as M4/w4, and
adding warps at fixed M (more warps/row) should monotonically hurt.

Control: (2048,1024) dense-narrow and (2048,2048) dense-wide should NOT show it.
persistent / stages=1 / flat. fp32. do_bench median-of-9. PYTHONPATH-only.
"""

from __future__ import annotations

import os

import torch

import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), helion.__file__

from triton.testing import do_bench  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402

helion.exc  # noqa
N_RUNS = 9
EPS = 1e-5

# (M_BLOCK, num_warps) probes. "wpr" annotation = rough warps-per-row intuition.
PROBES = [
    (1, 1),   # 1 row, 1 warp  -> intra-warp, low occupancy
    (1, 4),   # 1 row, 4 warps -> cross-warp x4
    (1, 8),   # 1 row, 8 warps -> cross-warp x8 (seed)
    (2, 2),
    (4, 1),   # 4 rows, 1 warp total
    (4, 2),
    (4, 4),   # the winner
    (4, 8),
    (4, 16),  # 4 rows, 16 warps -> >1 warp/row again
    (8, 8),
]
SHAPES = [(2048, 1025), (2048, 1024), (2048, 2048)]


def med(fn):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[N_RUNS // 2]


def build(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, [n], w, b, EPS), torch.nn.functional.layer_norm(x, [n], w, b, EPS)


def run(args, ref, mb, w):
    cfg = dict(block_sizes=[mb], reduction_loops=[None], num_warps=w,
               num_stages=1, pid_type="flat")
    try:
        k = helion.kernel(layer_norm_fwd.fn, configs=[helion.Config(**cfg)],
                          ignore_warnings=[helion.exc.TensorOperationInWrapper])
        bound = k.bind(args)
        bound.ensure_config_exists(args)
        out = bound(*args)
        o = (out[0] if isinstance(out, tuple) else out).float()
        if not torch.allclose(o, ref, rtol=1e-3, atol=1e-4):
            return "BAD"
        return f"{med(lambda: bound(*args)) * 1000:.1f}"
    except Exception as e:  # noqa: BLE001
        return f"ERR:{type(e).__name__}"


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}\n", flush=True)
    for (m, n) in SHAPES:
        args, ref = build(m, n)
        print(f"==== layer_norm ({m},{n}) ====", flush=True)
        print(f"  {'(M_BLK, warps)':>16}   us", flush=True)
        for (mb, w) in PROBES:
            lat = run(args, ref, mb, w)
            print(f"  {f'M={mb}, w={w}':>16}   {lat}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
