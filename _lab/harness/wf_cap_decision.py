"""Decide the welford combine-chunk CAP. For each candidate rule, emit a CORRECT
config per shape and compute the geomean G. Rule = combine tile is the largest
pow2 divisor of N, capped at CAP (always a pow2 divisor, hence correct);
normalize tile = next_pow2(N) persistent; warps per a ramp.

Compare CAP in {1024, 2048} and warps {8, 16} held / ramped.
"""
from __future__ import annotations

import math
import os
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
IN_SAMPLE = [(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096)]


def args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def med(fn, reps=3):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def mfloor(a):
    bs = welford.bind(a).config_spec.block_sizes[0]
    return max(1, bs.min_size, bs.autotuner_min)


def lpd(n):
    return n & (-n)


def run_seed(a, bs, w, ref):
    k = helion.kernel(welford.fn, configs=[helion.Config(
        block_sizes=bs, num_warps=w, num_stages=1)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
    maxabs = float((out.float() - ref.float()).abs().max())
    lat = med(lambda: b(*a)) * 1000
    return lat, ok, maxabs


def warps_for(rnumel):
    # reuse the existing rnumel ramp idea
    if rnumel <= 1024:
        return 8
    if rnumel <= 4096:
        return 8
    return 16


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    # rules: (name, cap_combine, warps_fixed_or_None)
    rules = [
        ("cap1024/w8", 1024, 8),
        ("cap2048/w8", 2048, 8),
        ("cap1024/w16", 1024, 16),
    ]
    tcs = {}
    per_rule = {r[0]: [] for r in rules}
    for (m, n) in IN_SAMPLE:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        mf = mfloor(a)
        P = np2(n)
        L = lpd(n)
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm); tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        tcs[(m, n)] = tclat
        print(f"=== ({m},{n}) np2={P} lpd={L} mf={mf} tc={round(tclat,1)}us ===", flush=True)
        for (name, cap, w) in rules:
            combine = min(L, cap)  # pow2 divisor of N (correct)
            lat, ok, maxabs = run_seed(a, [mf, combine, P], w, ref)
            g = tclat / lat
            flag = "" if ok else "  <-- WRONG"
            print(f"  {name:14s} combine={combine:5d} norm={P:5d} w{w}: "
                  f"{round(lat,1):8}us ok={ok} maxabs={maxabs:.1e} G={round(g,3)}{flag}",
                  flush=True)
            if ok:
                per_rule[name].append(g)
        print(flush=True)

    def gm(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else None
    print("=== RULE GEOMEANS (over 4 in-sample shapes) ===", flush=True)
    for name, gs in per_rule.items():
        print(f"  {name:14s}: geomean G = {gm(gs) and round(gm(gs),4)}  "
              f"(n={len(gs)})", flush=True)


if __name__ == "__main__":
    main()
