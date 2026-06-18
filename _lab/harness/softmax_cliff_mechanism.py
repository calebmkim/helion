"""Is the welford apply 'cliff' a GENERIC working-tile register spill (so softmax cliffs
too if given a big enough tile), or is it welford-specific?

Mechanism under test: the 6.6x welford blowup at (262144,5120) apply=8192,M_BLOCK=16 was
attributed (wrongly, first pass) to 'fat accumulators'. But welford's carried state is
[M_BLOCK] scalars (acc_cnt/mean/m2) — IDENTICAL in size to softmax's mi/di. So the cliff
should be a generic register spill driven by the LOADED DATA TILE [M_BLOCK x tile_N x
itemsize], independent of the carried-accumulator size.

PREDICTION if generic-spill is correct:
  (1) softmax ALSO cliffs when M_BLOCK x tile_N gets large enough (it just wasn't given a
      big tile in the np2(4096)-capped test).
  (2) For softmax, a SMALLER SHARED tile fully recovers the cliff (no split needed),
      because softmax's combine (running max/sum) is cheap+memory-bound and does NOT need
      a larger tile than apply. => softmax: best_shared ~= best_split (SHARED-OK).
  This is the difference from welford(262144,4096), where combine WANTED 4096 (persistent)
  but apply needed 2048 -> the split recovered 7.5% a shared tile could not.

This script: softmax_split at HIGH M_BLOCK + WIDE N (so the tile can be pushed large
enough to spill), full combine x apply grid. Per shape report:
  - cliff_ratio = worst_ok / best_ok over the whole grid (how bad does a big tile get?)
  - whether the LARGE shared tile (np2(N),np2(N)) is the cliff
  - best_shared vs best_split (does a smaller SHARED tile recover, or is split needed?)
  - a M_BLOCK=1 control (no cliff expected: tile footprint stays small)
Correctness: allclose vs torch.softmax fp32, wrong configs discarded.
"""
from __future__ import annotations

import math
import os
import sys

import torch

import helion
import helion.language as hl

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(helion.__file__)))
assert helion.__file__.startswith(WORKTREE), helion.__file__
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 0.03


def softmax_split(x: torch.Tensor) -> torch.Tensor:
    m, n = x.size()
    out = torch.empty_like(x)
    block_size_m = hl.register_block_size(m)
    block_size_n_combine = hl.register_block_size(n)
    block_size_n_apply = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=block_size_m):
        mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        di = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=block_size_n_combine):
            values = x[tile_m, tile_n]
            local_amax = torch.amax(values, dim=1)
            mi_next = torch.maximum(mi, local_amax)
            di = di * torch.exp(mi - mi_next) + torch.exp(
                values - mi_next[:, None]
            ).sum(dim=1)
            mi = mi_next
        for tile_n in hl.tile(n, block_size=block_size_n_apply):
            values = x[tile_m, tile_n]
            out[tile_m, tile_n] = torch.exp(values - mi[:, None]) / di[:, None]
    return out


# High-M wide-N (push the tile big enough to spill at high M_BLOCK) + M_BLOCK=1 controls.
CASES = [
    ((131072, 16384), 16),   # M16: [16 x tile] can reach [16,16384]=1MB -> expect cliff
    ((131072, 16384), 1),    # control: [1 x tile] stays small -> no cliff
    ((65536, 16384), 16),
    ((65536, 8192), 16),
]
TILES = [1024, 2048, 4096, 8192, 16384]


def warps_for(n):
    if n > 16384:
        return 32
    if n <= 1024:
        return 4
    if n <= 4096:
        return 8
    return 16


def med(fn, reps=4):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def run(x, bs, w, ref):
    k = helion.kernel(
        softmax_split, configs=[helion.Config(block_sizes=bs, num_warps=w, num_stages=1)]
    )
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-4, atol=1e-4))
    lat = med(lambda: b(x)) * 1000
    return lat, ok


def main():
    print(f"helion={helion.__file__}\nEPS={EPS}\n", flush=True)
    summary = []
    for (m, n), mblk in CASES:
        x = torch.rand(m, n, device="cuda", dtype=torch.float32)
        ref = torch.softmax(x, dim=1)
        w = warps_for(n)
        P = np2(n)
        tiles = sorted({t for t in TILES if t <= P} | {P})
        print(f"=== ({m},{n}) M_BLOCK={mblk} np2={P} w{w} ===", flush=True)
        results = {}
        for combine in tiles:
            for apply_ in tiles:
                lat, ok = run(x, [mblk, combine, apply_], w, ref)
                results[(combine, apply_)] = (lat, ok)
                eq = " (shared)" if combine == apply_ else ""
                flag = "" if ok else "  <-- WRONG"
                print(f"  c={combine:6d} a={apply_:6d} {round(lat,1):10}us ok={ok}{eq}{flag}",
                      flush=True)
        ok_res = {k: v[0] for k, v in results.items() if v[1]}
        shared = {k: v for k, v in ok_res.items() if k[0] == k[1]}
        split = {k: v for k, v in ok_res.items() if k[0] != k[1]}
        worst = max(ok_res.values())
        best = min(ok_res.values())
        cliff_ratio = worst / best
        big_shared = shared.get((P, P))
        bsh_k = min(shared, key=shared.get)
        bsp_k = min(split, key=split.get) if split else None
        bsh, bsp = shared[bsh_k], (split[bsp_k] if bsp_k else None)
        split_ratio = (bsp / bsh) if bsp else None
        split_verdict = ("SPLIT-WINS" if split_ratio and split_ratio < 1 - EPS
                         else "SHARED-OK")
        big_is_cliff = (big_shared is not None and big_shared / best > 1.5)
        print(f"  --> cliff_ratio(worst/best)={round(cliff_ratio,2)}x  "
              f"big_shared({P},{P})={round(big_shared,1) if big_shared else None}us "
              f"{'= CLIFF' if big_is_cliff else 'ok'}", flush=True)
        print(f"      best_shared={bsh_k}@{round(bsh,1)}us  "
              f"best_split={bsp_k}@{round(bsp,1) if bsp else None}us  "
              f"split/shared={round(split_ratio,3) if split_ratio else None}  "
              f"{split_verdict}\n", flush=True)
        summary.append(((m, n, mblk), round(cliff_ratio, 2), big_is_cliff, bsh_k,
                        round(bsh, 1), bsp_k, round(bsp, 1) if bsp else None,
                        round(split_ratio, 3) if split_ratio else None, split_verdict))

    print("================ SUMMARY ================", flush=True)
    print(f"{'shape(M)':>20} {'cliff':>7} {'bigShrCliff':>11} {'best_shared':>14} "
          f"{'us':>9} {'best_split':>14} {'split/shr':>9}  verdict", flush=True)
    for (m, n, mb), cr, bic, ksh, lsh, ksp, lsp, sr, v in summary:
        print(f"{f'({m},{n})[{mb}]':>20} {f'{cr}x':>7} {str(bic):>11} {str(ksh):>14} "
              f"{lsh:>9} {str(ksp):>14} {str(sr):>9}  {v}", flush=True)
    nwin = sum(1 for *_, v in summary if v == "SPLIT-WINS")
    print(f"\nSPLIT-WINS on {nwin}/{len(summary)}.", flush=True)
    print("INTERPRETATION KEYS:", flush=True)
    print("  - If high-M shapes show big cliff_ratio AND big_shared is the CLIFF AND "
          "SHARED-OK (a smaller shared tile recovers) => cliff is GENERIC working-tile "
          "spill, split NOT needed for softmax (smaller shared tile suffices).", flush=True)
    print("  - M_BLOCK=1 control should show small cliff_ratio (tile footprint stays "
          "small even at large tile) => confirms M_BLOCK x tile_N is the driver.",
          flush=True)


if __name__ == "__main__":
    main()
