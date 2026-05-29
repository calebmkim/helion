"""Reproduce the BRIEF's exact claim verbatim at (262144,4096):
  v7 seed     = [16, 2048, 4096] (combine=2048, apply=4096 persist), claimed G=0.708
  oracle win  = [16, 2048, 2048] (combine=2048, apply=2048 looped),  claimed G=0.968 (~27%)

Also sweep the FULL combine x apply grid to find where (if anywhere) G~0.96 lives, so we
can correctly ATTRIBUTE any large win (apply tile? combine tile? warps?). All correctness
checked. Tries num_warps {8,16} too (the brief's oracle may carry a different warp count).
"""
from __future__ import annotations

import math
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.welford import eager_layer_norm  # noqa: E402
from examples.welford import welford  # noqa: E402

EPS = 1e-5
M, N = 262144, 4096


def args(m, n):
    return (
        torch.rand(n, device="cuda"),
        torch.rand(n, device="cuda"),
        torch.rand(m, n, device="cuda"),
        EPS,
    )


def med(fn, reps=5):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def run(a, bs, w, ref):
    k = helion.kernel(
        welford.fn,
        configs=[helion.Config(block_sizes=bs, num_warps=w, num_stages=1)],
    )
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
    maxabs = float((out.float() - ref.float()).abs().max())
    lat = med(lambda: b(*a)) * 1000
    return lat, ok, maxabs


def main():
    print(f"helion={helion.__file__}  shape=({M},{N})\n", flush=True)
    a = args(M, N)
    ref = eager_layer_norm(*a)
    torch._dynamo.reset()
    tc = torch.compile(eager_layer_norm)
    tc(*a)
    tclat = med(lambda: tc(*a)) * 1000
    print(f"tc = {round(tclat,1)}us\n", flush=True)

    print("== BRIEF verbatim configs (w8) ==", flush=True)
    for bs in ([16, 2048, 4096], [16, 2048, 2048]):
        lat, ok, mx = run(a, bs, 8, ref)
        print(f"  {bs} w8: {round(lat,1)}us ok={ok} G={round(tclat/lat,3)} maxabs={mx:.1e}", flush=True)

    print("\n== combine x apply GRID (w8), find best G ==", flush=True)
    combines = [128, 256, 512, 1024, 2048, 4096]
    applies = [512, 1024, 2048, 4096]
    best = None
    for cb in combines:
        row = []
        for ap in applies:
            lat, ok, mx = run(a, [16, cb, ap], 8, ref)
            g = tclat / lat
            row.append(f"ap{ap}:{round(g,2)}{'' if ok else '!'}")
            if ok and (best is None or g > best[0]):
                best = (g, cb, ap, 8, lat)
        print(f"  combine={cb:5d}: " + "  ".join(row), flush=True)

    print("\n== best-combine, sweep warps {8,16,32} ==", flush=True)
    _, cb, ap, _, _ = best
    for w in (8, 16, 32):
        lat, ok, mx = run(a, [16, cb, ap], w, ref)
        g = tclat / lat
        print(f"  combine={cb} apply={ap} w{w}: {round(lat,1)}us ok={ok} G={round(g,3)}", flush=True)
        if ok and g > best[0]:
            best = (g, cb, ap, w, lat)

    print(f"\nBEST: G={round(best[0],3)} combine={best[1]} apply={best[2]} w{best[3]} ({round(best[4],1)}us)", flush=True)


if __name__ == "__main__":
    main()
