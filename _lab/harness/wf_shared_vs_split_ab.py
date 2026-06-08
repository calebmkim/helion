"""Does welford NEED different block sizes for its two inner loops (combine vs apply)?

The is_structured_combine seed emits block_sizes = [M_BLOCK, COMBINE, APPLY], sizing
the reduction (combine) pass and the normalize (apply) pass INDEPENDENTLY. At large N
they diverge (e.g. [1, 8192, 2048]); at small N they already coincide ([1,1024,1024]).

This A/B asks whether the independence is load-bearing for PERF. For each welford shape:
  - BASELINE  = the heuristic's actual emitted [M, combine*, apply*]  (split, current)
  - SHARE@combine = [M, combine*, combine*]   (force apply UP to combine)
  - SHARE@apply   = [M, apply*,   apply*]     (force combine DOWN to apply)
  - SHARE@<b>     = [M, b, b] for b in a small candidate set (best single shared tile)
All other levers (num_warps, num_stages) held EQUAL to the heuristic's emit.

VERDICT per shape: is there ANY single shared block size within EPS of the split
baseline? If yes on every shape -> independent sizing is NOT necessary (the structured
path could collapse to one tile). If some shape's split baseline beats every shared
choice by > EPS -> the two-tile split is genuinely necessary there.

Correctness: welford.py uses the masked-count idiom `Tn=(tile_n.index<n).sum()`, so the
per-chunk mean/M2 are correct for ANY combine tile (not just pow2 divisors); the apply
pass is a masked elementwise write, correct for any apply tile. We still allclose-gate
every config and DISCARD any that fails (never compare latency on a wrong result).
"""
from __future__ import annotations

import os
import sys

import torch

import helion

# Resolve the worktree from the live import (set via PYTHONPATH); do not hardcode.
WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(helion.__file__)))
assert helion.__file__.startswith(WORKTREE), helion.__file__
if WORKTREE not in sys.path:
    sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.welford import eager_layer_norm  # noqa: E402
from examples.welford import welford  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 0.05  # <=5% slower than baseline counts as a tie (no perf lost by sharing)
EPS_TIE = 0.03

# welford train shapes (shapes_v3) + the EDIT#4 catastrophe canary (262144,5120 robust)
SHAPES = [
    (16384, 768), (16384, 2048),                 # small: combine==apply already
    (16384, 4096), (8192, 5120), (8192, 7168),   # the split regime (combine>apply)
    (8192, 8192), (8192, 12288), (4096, 16384),
    (65536, 4096), (32768, 8192),                # M-variation, split regime
    (262144, 5120),                              # EDIT#4 7x-regression canary (robustness)
]
# candidate single shared tiles to search for a best-shared
SHARED_CANDS = [1024, 2048, 4096, 8192]


def args(m, n):
    return (
        torch.rand(n, device="cuda"),
        torch.rand(n, device="cuda"),
        torch.rand(m, n, device="cuda"),
        1e-5,
    )


def med(fn, reps=5):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def emitted_seed(a):
    """The heuristic's actual emitted [M,combine,apply] + num_warps for this shape.

    Read the seed straight off the compiler-seed-heuristic output
    (config_spec.compiler_seed_configs), populated at bind() time. We must NOT
    rely on ensure_config_exists()/b._config: under HELION_AUTOTUNE_EFFORT=none
    that path returns the DEFAULT config ([16,16,16]), not the heuristic seed.
    """
    k = helion.kernel(welford.fn)  # no config -> seed heuristic fires at bind()
    b = k.bind(a)
    seeds = b.env.config_spec.compiler_seed_configs
    assert seeds, f"no compiler seed config emitted (heuristics={b.env.config_spec.autotuner_heuristics})"
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
    print(f"helion={helion.__file__}", flush=True)
    print(f"EPS(tie<=)={EPS_TIE}  EPS(share-ok<=){EPS}\n", flush=True)
    verdicts = []
    for (m, n) in SHAPES:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        bs, w, ns = emitted_seed(a)
        M, combine, apply_ = bs[0], bs[1], bs[2]
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm)
        tc(*a)
        tclat = med(lambda: tc(*a)) * 1000

        base_lat, base_ok = run(a, [M, combine, apply_], w, ns, ref)
        print(
            f"=== ({m},{n}) np2={np2(n)} SPLIT baseline=[{M},{combine},{apply_}] "
            f"w{w} ns{ns} : {round(base_lat,1)}us ok={base_ok} "
            f"G(tc/seed)={round(tclat/base_lat,3)} ===",
            flush=True,
        )
        assert base_ok, "baseline wrong — abort"

        # the two forced-shared extremes + the candidate sweep
        cand_tiles = sorted({combine, apply_} | {c for c in SHARED_CANDS if c <= np2(n)})
        best_shared_lat, best_shared_b = None, None
        for b_tile in cand_tiles:
            lat, ok = run(a, [M, b_tile, b_tile], w, ns, ref)
            ratio = lat / base_lat
            tags = []
            if b_tile == combine:
                tags.append("=combine")
            if b_tile == apply_:
                tags.append("=apply")
            tag = ",".join(tags) or "shared"
            flag = "" if ok else "  <-- WRONG (discarded)"
            print(
                f"  share={b_tile:5d} [{M},{b_tile},{b_tile}] {round(lat,1):8}us "
                f"ok={ok} ratio_to_split={round(ratio,3)} ({tag}){flag}",
                flush=True,
            )
            if ok and (best_shared_lat is None or lat < best_shared_lat):
                best_shared_lat, best_shared_b = lat, b_tile

        ratio = best_shared_lat / base_lat if best_shared_lat else float("inf")
        if ratio <= 1 + EPS_TIE:
            v = "SHARE-OK (tie)"
        elif ratio <= 1 + EPS:
            v = "SHARE-OK (<=5%)"
        else:
            v = "SPLIT-NEEDED"
        print(
            f"  --> best shared={best_shared_b} @ {round(best_shared_lat,1)}us; "
            f"best_shared/split={round(ratio,3)}  VERDICT: {v}\n",
            flush=True,
        )
        verdicts.append(((m, n), combine, apply_, best_shared_b, round(ratio, 3), v))

    print("================ SUMMARY ================", flush=True)
    print(f"{'shape':>16} {'combine':>8} {'apply':>6} {'best_share':>10} "
          f"{'share/split':>11}  verdict", flush=True)
    for sh, c, ap, bsh, r, v in verdicts:
        print(f"{str(sh):>16} {c:>8} {ap:>6} {str(bsh):>10} {r:>11}  {v}", flush=True)
    n_need = sum(1 for *_, v in verdicts if v == "SPLIT-NEEDED")
    print(f"\nSPLIT-NEEDED on {n_need}/{len(verdicts)} shapes.", flush=True)
    if n_need == 0:
        print("=> independent combine/apply sizing NOT necessary for perf (collapsible).",
              flush=True)
    else:
        print("=> independent combine/apply sizing IS necessary (split wins on those).",
              flush=True)


if __name__ == "__main__":
    main()
