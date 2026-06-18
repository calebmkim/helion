"""Does softmax gain perf from INDEPENDENT combine/apply tile sizes (the welford question,
ported to softmax)?

softmax_two_pass is structurally a reduce-then-apply kernel:
  loop 1 (combine): running max + running sum-exp  -> REDUCTIONS over the tile
  loop 2 (apply):   out = exp(x - m)/d             -> pure masked elementwise write, NO reduction
But it is written with ONE shared block_size_n, so both loops use the same tile -> the
seed heuristic sees ONE non-grid tile (is_structured_combine=False). This script defines
a SPLIT variant with two independent block sizes (combine, apply) and runs the full
combine x apply grid, asking — exactly as the welford grid A/B did — whether any UNEQUAL
pair strictly beats EVERY equal (shared) tile.

Compares, per shape:
  best_shared = min latency over the diagonal combine==apply==b
  best_split  = min latency over combine != apply
  SPLIT-WINS iff best_split < best_shared * (1 - EPS)

Regimes (where the welford lesson predicts a split could matter):
  - wide-N (the apply pass re-streams the row; combine is an online recurrence that may
    prefer a wider/persistent tile) — both curriculum looped shapes and force-loopable ones
  - high-M mid-N (the welford footprint-cliff analog: welford(262144,4096) split-won 7.5%)

Correctness: online-softmax combine is correct for ANY tile (running rescale); the apply
write is masked, correct for any tile. Still allclose-gated vs torch.softmax fp32; wrong
configs discarded. num_warps/num_stages held at the heuristic's ladder value per shape so
only the tile split varies.

NOTE on block_sizes order: the kernel registers block_size_m, then combine, then apply, so
block_sizes == [M_BLOCK, combine, apply]. The script ASSERTS this by dumping the bound
spec's block layout; if the order differs it prints a warning (the SPLIT-WINS verdict is
order-invariant, but the combine/apply LABELS would swap).
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

EPS = 0.03  # split must beat best-shared by > 3% to count as SPLIT-WINS


def softmax_split(x: torch.Tensor) -> torch.Tensor:
    """softmax_two_pass with INDEPENDENT combine/apply tiles over the reduction axis."""
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


# (shape, role, M_BLOCK). M_BLOCK pinned so the footprint regime is explicit. The high-M
# shapes use a bumped M_BLOCK (the welford cliff appeared at M_BLOCK>=16); softmax floor is
# typically 1, so we test BOTH the floor and a bump on the high-M analogs.
CASES = [
    # curriculum wide-N (low-M): does the apply pass want a different tile than combine?
    ((4096, 16384), "train-wideN", 1),
    ((2048, 32768), "train-wideN", 1),
    ((1024, 65536), "train-loopedN", 1),   # 256KiB row -> heuristic loops this today
    ((512, 131072), "train-loopedN", 1),   # 512KiB row -> heuristic loops this today
    # welford-footprint analog: high-M mid-N (off-curriculum, the regime welford split-won)
    ((131072, 4096), "highM-analog", 1),
    ((131072, 4096), "highM-analog", 16),
    ((262144, 4096), "highM-analog", 16),
]
TILES = [1024, 2048, 4096, 8192, 16384, 32768]


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


def show_layout():
    a = torch.rand(4096, 16384, device="cuda")
    b = helion.kernel(softmax_split).bind((a,))
    bsz = b.env.config_spec.block_sizes
    hints = [int(bsz[i].size_hint) for i in range(len(bsz))]
    print(f"split-softmax block_sizes layout: count={len(bsz)} size_hints={hints} "
          f"(expect [M=4096, combine=16384, apply=16384])", flush=True)
    if not (len(bsz) == 3 and hints[0] == 4096 and hints[1] == 16384 and hints[2] == 16384):
        print("  !! WARNING: block_sizes layout differs from assumed [M,combine,apply]; "
              "combine/apply labels may be swapped (SPLIT-WINS verdict still valid).",
              flush=True)


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
    print(f"helion={helion.__file__}\nEPS(split must beat shared by >)={EPS}\n", flush=True)
    show_layout()
    print(flush=True)
    summary = []
    for (m, n), role, mblk in CASES:
        x = torch.rand(m, n, device="cuda", dtype=torch.float32)
        ref = torch.softmax(x, dim=1)
        w = warps_for(n)
        P = np2(n)
        tiles = [t for t in TILES if t <= P] + ([P] if P not in TILES else [])
        tiles = sorted(set(tiles))
        torch._dynamo.reset()
        tc = torch.compile(lambda t: torch.softmax(t, dim=1))
        tc(x)
        tclat = med(lambda: tc(x)) * 1000
        print(f"=== ({m},{n}) [{role}] M_BLOCK={mblk} np2={P} w{w} tc={round(tclat,1)}us ===",
              flush=True)
        results = {}  # (combine, apply) -> (lat, ok)
        for combine in tiles:
            for apply_ in tiles:
                lat, ok = run(x, [mblk, combine, apply_], w, ref)
                results[(combine, apply_)] = (lat, ok)
                eq = " (shared)" if combine == apply_ else ""
                flag = "" if ok else "  <-- WRONG (discarded)"
                print(f"  combine={combine:6d} apply={apply_:6d} {round(lat,1):9}us "
                      f"ok={ok}{eq}{flag}", flush=True)
        ok_res = {k: v[0] for k, v in results.items() if v[1]}
        shared = {k: v for k, v in ok_res.items() if k[0] == k[1]}
        split = {k: v for k, v in ok_res.items() if k[0] != k[1]}
        bsh_k = min(shared, key=shared.get) if shared else None
        bsp_k = min(split, key=split.get) if split else None
        bsh = shared.get(bsh_k) if bsh_k else None
        bsp = split.get(bsp_k) if bsp_k else None
        if bsh is None or bsp is None:
            verdict, ratio = "INCONCLUSIVE", None
        else:
            ratio = bsp / bsh
            verdict = "SPLIT-WINS" if ratio < 1 - EPS else "SHARED-OK"
        print(f"  --> best_shared={bsh_k}@{round(bsh,1) if bsh else None}us "
              f"best_split={bsp_k}@{round(bsp,1) if bsp else None}us "
              f"split/shared={round(ratio,3) if ratio else None}  {verdict}\n", flush=True)
        summary.append(((m, n, mblk), role, bsh_k, bsh, bsp_k, bsp,
                        round(ratio, 3) if ratio else None, verdict))

    print("================ SUMMARY ================", flush=True)
    print(f"{'shape(M_BLOCK)':>22} {'role':>14} {'best_shared':>14} {'us':>9} "
          f"{'best_split':>14} {'us':>9} {'split/shared':>12}  verdict", flush=True)
    for (m, n, mb), role, ksh, lsh, ksp, lsp, r, v in summary:
        print(f"{f'({m},{n})[{mb}]':>22} {role:>14} {str(ksh):>14} "
              f"{round(lsh,1) if lsh else None:>9} {str(ksp):>14} "
              f"{round(lsp,1) if lsp else None:>9} {str(r):>12}  {v}", flush=True)
    nwin = sum(1 for *_, v in summary if v == "SPLIT-WINS")
    print(f"\nSPLIT-WINS on {nwin}/{len(summary)} cases.", flush=True)
    if nwin:
        print("=> softmax DOES gain from independent combine/apply tiles on some shapes "
              "(the welford lesson transfers).", flush=True)
    else:
        print("=> softmax does NOT benefit from splitting the tiles (shared is within EPS "
              "everywhere) — differs from welford.", flush=True)


if __name__ == "__main__":
    main()
