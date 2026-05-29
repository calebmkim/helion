"""welford config ceiling: sweep BOTH tile_n blocks (welford-pass R1 and
normalize-pass R2) + M_BLOCK + warps to find the best achievable welford config,
vs the heuristic seed and tc. Tells us what welford NEEDS (is there a config that
ties tc? what block pattern? does it generalize across the 4 in-sample shapes?).

block_sizes = [M_BLOCK, R1(welford pass), R2(normalize pass)].
Run with the canonical invocation (SETUP.md).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.welford import welford, eager_layer_norm  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 1e-5
N_RUNS = 5


def build_args(m, n):
    return (torch.rand(n, device="cuda", dtype=torch.float32),
            torch.rand(n, device="cuda", dtype=torch.float32),
            torch.rand(m, n, device="cuda", dtype=torch.float32), EPS)


def median_do_bench(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def t(args, cfg, ref):
    try:
        k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        out = b(*args)
        ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
        return round(median_do_bench(lambda: b(*args)) * 1000, 1), ok
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}", False


def main():
    print(f"helion={helion.__file__}\n")
    for (m, n) in [(262144, 1024), (262144, 2048), (262144, 4096)]:
        args = build_args(m, n)
        ref = eager_layer_norm(*args)
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm); tc(*args)
        tclat = round(median_do_bench(lambda: tc(*args)) * 1000, 1)
        print(f"=== ({m},{n}) tc={tclat}us ===")
        best = None
        for mb in (1, 16, 64):
            for r1 in (np2(n), 1024, 2048):
                for r2 in (np2(n), 1024, 2048):
                    for w in (4, 8, 16):
                        cfg = {"block_sizes": [mb, r1, r2], "num_warps": w,
                               "num_stages": 1}
                        lat, ok = t(args, cfg, ref)
                        if ok and isinstance(lat, float):
                            if best is None or lat < best[0]:
                                best = (lat, mb, r1, r2, w)
        if best:
            lat, mb, r1, r2, w = best
            print(f"  BEST: {lat}us  M={mb} R1={r1} R2={r2} w={w}  "
                  f"G_best={round(tclat/lat,3)}")
        print()


if __name__ == "__main__":
    main()
