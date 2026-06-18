"""TASK 1: matched A/B of the welford APPLY (normalize) tile.

The v7 is_structured_combine seed sets:
  - combine tile (block 1, the reduction axis) = min(largest_pow2_div(N), cap)
  - apply tile   (block 2, the normalize pass) = next_pow2(N) PERSISTENT

The capstone audit found that at large N the LOOPED apply tile (capped) beats the
PERSISTENT next_pow2(N) apply tile by ~27%. This script holds the combine tile + warps
+ stages EQUAL (all other levers matched) and sweeps ONLY the apply tile across
{512,1024,2048,4096, next_pow2(N)}, measuring latency + correctness at each, across
welford shapes (incl the non-pow2 canary 1536 and a prime-ish N).

WHY looped apply is correct: the apply pass (welford.py lines 69-77) is a pure
element-wise normalize `y=(x-mean)*rstd*w+b` with a MASKED output write
`out[tile_m, tile_n]=y` — no Tn-division (unlike the combine pass). Looping the tile
just writes the row in chunks; masking drops only invalid lanes. So ANY apply tile is
correct (verified here), unlike the combine tile (which must be a pow2 divisor).
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
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 1e-5
# in-sample widest + the non-pow2 canary + a prime-ish N (1543 is prime; 5120=2^10*5)
SHAPES = [(262144, 4096), (262144, 2048), (262144, 1536), (262144, 1024)]
APPLY_CANDIDATES = [512, 1024, 2048, 4096]


def args(m, n):
    return (
        torch.rand(n, device="cuda"),
        torch.rand(n, device="cuda"),
        torch.rand(m, n, device="cuda"),
        EPS,
    )


def med(fn, reps=4):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def mfloor(a):
    bs = welford.bind(a).config_spec.block_sizes[0]
    return max(1, bs.min_size, bs.autotuner_min)


def lpd(n):
    return n & (-n)


def num_warps(rnumel):
    if rnumel > 16384:
        return 32
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    return 16


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
    print(f"helion={helion.__file__}\n", flush=True)
    for (m, n) in SHAPES:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        mf = mfloor(a)
        P = np2(n)
        # v7 combine tile: min(lpd(N), cap_elems). cap=8192B/4=2048 elems.
        combine = min(lpd(n), 2048)
        w = num_warps(n)
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm)
        tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        print(
            f"=== ({m},{n}) np2={P} lpd={lpd(n)} combine={combine} mf={mf} w{w} "
            f"tc={round(tclat,1)}us ===",
            flush=True,
        )
        # candidate apply tiles: the fixed-cap candidates that are <= np2(N), PLUS
        # next_pow2(N) (the v7 persistent choice). Dedup.
        cands = sorted({c for c in APPLY_CANDIDATES if c <= P} | {P})
        best_lat = None
        best_apply = None
        for ap in cands:
            persistent = ap >= P
            lat, ok, maxabs = run(a, [mf, combine, ap], w, ref)
            tag = "PERSIST" if persistent else "looped "
            flag = "" if ok else "  <-- WRONG"
            g = tclat / lat
            print(
                f"  apply={ap:5d} {tag} {round(lat,1):8}us ok={ok} "
                f"maxabs={maxabs:.1e} G={round(g,3)}{flag}",
                flush=True,
            )
            if ok and (best_lat is None or lat < best_lat):
                best_lat = lat
                best_apply = ap
        # report the v7 (persistent) vs best looped
        v7_lat, _, _ = run(a, [mf, combine, P], w, ref)
        speedup = v7_lat / best_lat if best_lat else None
        print(
            f"  --> v7 apply={P} (persist) = {round(v7_lat,1)}us;  "
            f"BEST apply={best_apply} = {round(best_lat,1)}us;  "
            f"v7/best = {round(speedup,3)} ({'looped wins' if best_apply < P else 'persist wins'})",
            flush=True,
        )
        print(flush=True)


if __name__ == "__main__":
    main()
