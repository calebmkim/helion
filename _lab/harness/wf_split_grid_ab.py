"""Is there a welford shape where UNEQUAL tiles (combine != apply) beat EVERY equal tile?

The prior A/B (wf_shared_vs_split_ab.py) compared the HEURISTIC's split (apply pinned
to 2048) vs best-shared, and found 0/11 SPLIT-NEEDED. But that is not a fair test of
whether the SPLIT CAPABILITY is ever necessary — the heuristic's apply=2048 is a known-
conservative clamp. This script runs the FULL combine x apply grid and asks the real
question:

  best_split  = min latency over pairs with combine != apply
  best_shared = min latency over the diagonal (combine == apply == b)
  SPLIT-WINS iff best_split < best_shared * (1 - EPS)   (an unequal pair strictly beats
                                                          every equal tile by > EPS)

Lead from wf_combine_spill_probe: at (262144,4096), combine=4096+apply=2048 was "the
BEST", while combine=4096+apply=4096 was catastrophic -> a split-wins candidate.

Grid is TRIANGULAR (apply <= combine), matching the heuristic's combine>=apply regime
and the physics (combine wants persistence; apply wants small footprint). M_BLOCK is held
at the heuristic's emitted value so the footprint regime matches what ships. num_warps /
num_stages held at the emit. Every config allclose-gated; wrong results discarded.
"""
from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(helion.__file__)))
assert helion.__file__.startswith(WORKTREE), helion.__file__
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.welford import eager_layer_norm  # noqa: E402
from examples.welford import welford  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 0.03  # split must beat best-shared by > 3% to count as a real SPLIT-WINS

# Focus on the high-M / footprint-pressured shapes where combine(persist) vs
# apply(footprint) tension is sharpest — that is where a split could be necessary.
SHAPES = [
    (262144, 4096),   # the combine-spill-probe lead
    (262144, 5120),   # EDIT#4 canary (shared@8192 was a 6.6x blowup)
    (262144, 2048),   # very high-M, narrow
    (131072, 4096),   # high-M, mid (clean test point)
    (65536, 4096),    # M-variation curriculum shape
    (32768, 8192),    # M-variation curriculum shape
]
ALL_TILES = [1024, 2048, 4096, 8192, 16384]


def args(m, n):
    return (
        torch.rand(n, device="cuda"),
        torch.rand(n, device="cuda"),
        torch.rand(m, n, device="cuda"),
        1e-5,
    )


def med(fn, reps=4):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def emitted_seed(a):
    k = helion.kernel(welford.fn)
    b = k.bind(a)
    seeds = b.env.config_spec.compiler_seed_configs
    assert seeds, "no compiler seed config emitted"
    cfg = seeds[0]
    return list(cfg.block_sizes), int(cfg.num_warps), int(cfg.num_stages)


def run(a, bs, w, ns, ref):
    k = helion.kernel(
        welford.fn,
        configs=[helion.Config(block_sizes=bs, num_warps=w, num_stages=ns)],
    )
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
    lat = med(lambda: b(*a)) * 1000
    return lat, ok


def main():
    print(f"helion={helion.__file__}\nEPS(split must beat shared by >)={EPS}\n", flush=True)
    summary = []
    for (m, n) in SHAPES:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        bs, w, ns = emitted_seed(a)
        M = bs[0]
        P = np2(n)
        tiles = [t for t in ALL_TILES if t <= P]
        print(f"=== ({m},{n}) np2={P} M_BLOCK={M} w{w} ns{ns}  heuristic_emit={bs} ===",
              flush=True)
        # triangular grid: apply <= combine
        results = {}  # (combine, apply) -> (lat, ok)
        for combine in tiles:
            for apply_ in tiles:
                if apply_ > combine:
                    continue
                lat, ok = run(a, [M, combine, apply_], w, ns, ref)
                results[(combine, apply_)] = (lat, ok)
                eq = "  (shared)" if combine == apply_ else ""
                flag = "" if ok else "  <-- WRONG (discarded)"
                print(f"  combine={combine:5d} apply={apply_:5d} "
                      f"{round(lat,1):9}us ok={ok}{eq}{flag}", flush=True)

        ok_res = {k: v[0] for k, v in results.items() if v[1]}
        shared = {k: lat for k, lat in ok_res.items() if k[0] == k[1]}
        split = {k: lat for k, lat in ok_res.items() if k[0] != k[1]}
        best_shared_k = min(shared, key=shared.get) if shared else None
        best_split_k = min(split, key=split.get) if split else None
        bsh = shared.get(best_shared_k) if best_shared_k else None
        bsp = split.get(best_split_k) if best_split_k else None
        if bsh is None or bsp is None:
            verdict, ratio = "INCONCLUSIVE", None
        else:
            ratio = bsp / bsh
            verdict = "SPLIT-WINS" if ratio < 1 - EPS else "SHARED-OK"
        print(f"  --> best_shared={best_shared_k}@{round(bsh,1) if bsh else None}us "
              f"best_split={best_split_k}@{round(bsp,1) if bsp else None}us "
              f"split/shared={round(ratio,3) if ratio else None}  {verdict}\n", flush=True)
        summary.append(((m, n), best_shared_k, bsh, best_split_k, bsp,
                        round(ratio, 3) if ratio else None, verdict))

    print("================ SUMMARY ================", flush=True)
    print(f"{'shape':>15} {'best_shared':>14} {'us':>8} {'best_split':>14} {'us':>8} "
          f"{'split/shared':>12}  verdict", flush=True)
    for sh, ksh, lsh, ksp, lsp, r, v in summary:
        print(f"{str(sh):>15} {str(ksh):>14} {round(lsh,1) if lsh else None:>8} "
              f"{str(ksp):>14} {round(lsp,1) if lsp else None:>8} {str(r):>12}  {v}",
              flush=True)
    nwin = sum(1 for *_, v in summary if v == "SPLIT-WINS")
    print(f"\nSPLIT-WINS on {nwin}/{len(summary)} shapes (unequal tiles strictly beat "
          f"every equal tile by >{int(EPS*100)}%).", flush=True)


if __name__ == "__main__":
    main()
