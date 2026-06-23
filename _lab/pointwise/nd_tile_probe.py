"""Probe the right N-D tile for residual_add (the seed's [32,inner] lags tc, G~0.88-0.91).
Force candidate configs on the worst shapes vs default vs tc(max-autotune). cold-L2 med-of-9."""

from __future__ import annotations

import itertools
import sys

import torch

import helion

WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
assert helion.__file__.startswith(WT)
from examples.add import add  # noqa: E402

DEV = "cuda"
N_RUNS = 9
DT = torch.bfloat16


def med(fn):
    from triton.testing import do_bench
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2]


def force(cfg, a, b):
    k = helion.kernel(add.fn, config=cfg, static_shapes=True)
    return lambda: k(a, b)


SHAPES = [(16384, 5120), (8192, 8192), (32768, 768), (8192, 4096)]
CANDS = [[32, 32], [32, 64], [1, 1024], [1, 2048], [1, 4096], [1, 8192],
         [2, 2048], [4, 1024], [8, 512], [16, 256], [64, 64], [2, 4096]]

for (m, n) in SHAPES:
    torch._dynamo.reset()
    a = torch.randn(m, n, device=DEV, dtype=DT)
    b = torch.randn(m, n, device=DEV, dtype=DT)
    ref = a + b
    tc = torch.compile(lambda x, y: x + y, mode="max-autotune-no-cudagraphs")
    tc(a, b)
    tt = med(lambda: tc(a, b))
    gbps = 3 * m * n * DT.itemsize / 1e12
    rows = []
    for cfg in CANDS:
        if cfg[1] > n and cfg[0] == 1:
            pass  # inner > N still valid (masked)
        try:
            call = force(helion.Config(block_sizes=cfg), a, b)
            out = call()
            if (out.float() - ref.float()).abs().max().item() > 0.01:
                continue
            t = med(call)
            rows.append((cfg, t))
        except Exception as e:  # noqa: BLE001
            rows.append((cfg, None))
    rows_ok = sorted([r for r in rows if r[1]], key=lambda r: r[1])
    print(f"\n=== ({m},{n})  tc={tt*1e3:.1f}us ({gbps/(tt*1e-3):.2f}TB/s) ===")
    for cfg, t in rows_ok[:6]:
        print(f"   {str(cfg):14} {t*1e3:8.1f}us  {gbps/(t*1e-3):5.2f}TB/s  G(tc/this)={tt/t:.3f}")
    del a, b, ref, tc
    torch.cuda.empty_cache()
