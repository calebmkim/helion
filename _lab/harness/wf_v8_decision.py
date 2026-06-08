"""TASK 1+2 decision sweep for v8 welford apply-tile cap.

FINDINGS so far (matched A/B, GPU2):
  (262144,4096): combine=2048/apply=2048 = G0.722; combine=4096/apply=2048 = G0.756;
                 combine=4096/apply=4096 = G0.128 (SPILL); v7 combine=2048/apply=4096 = G0.704.
  -> the ~7% oracle win needs combine=4096 (FULL) AND apply=2048 (LOOPED) TOGETHER.
  -> full combine is only spill-safe when the apply is looped.

This script decides the v8 rule. Candidate rules (all keep the combine = pow2-DIVISOR of N
for correctness, but with DIFFERENT caps; apply tile capped at a byte threshold = looped):

  R0  v7 baseline:   combine=min(lpd, 2048),  apply=np2(N) persist
  R1  apply-cap-only: combine=min(lpd, 2048), apply=min(np2, 2048 cap)  [looped apply]
  R2  apply+combine:  combine=min(lpd, BIG),  apply=min(np2, 2048 cap)  [full combine + looped apply]

We sweep the combine cap and the apply cap to set BOTH by evidence, across all 4 in-sample
shapes incl the non-pow2 canary 1536, with correctness at each.
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
IN_SAMPLE = [(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096)]


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
    mx = float((out.float() - ref.float()).abs().max())
    lat = med(lambda: b(*a)) * 1000
    return lat, ok, mx


def rule_config(n, combine_cap, apply_cap):
    """combine = largest pow2 div of N, capped at combine_cap (always a pow2 div ->
    correct). apply = min(np2(N), apply_cap) -> looped when np2(N) > apply_cap."""
    combine = min(lpd(n), np2(combine_cap))
    apply = min(np2(n), np2(apply_cap))
    return combine, apply


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    BIG = 1 << 20  # effectively uncapped combine
    rules = [
        ("R0_v7", 2048, BIG),            # v7: combine cap 2048, apply uncapped (persist)
        ("R1_applycap", 2048, 2048),     # apply cap 2048, combine cap 2048
        ("R2_full_combine", BIG, 2048),  # combine uncapped (=lpd), apply cap 2048
        ("R3_combine4096_applycap", 4096, 2048),  # combine cap 4096, apply cap 2048
        ("R4_applycap4096", 2048, 4096), # apply cap 4096, combine cap 2048
    ]
    per_rule = {r[0]: [] for r in rules}
    detail = {r[0]: {} for r in rules}
    for (m, n) in IN_SAMPLE:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        mf = mfloor(a)
        w = num_warps(n)
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm)
        tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        print(f"=== ({m},{n}) lpd={lpd(n)} np2={np2(n)} w{w} tc={round(tclat,1)}us ===", flush=True)
        for (name, ccap, acap) in rules:
            cb, ap = rule_config(n, ccap, acap)
            lat, ok, mx = run(a, [mf, cb, ap], w, ref)
            g = tclat / lat
            flag = "" if ok else "  <-- WRONG"
            tag = "persist" if ap >= np2(n) else "looped "
            print(
                f"  {name:24s} combine={cb:5d} apply={ap:5d}({tag}) "
                f"{round(lat,1):8}us ok={ok} maxabs={mx:.1e} G={round(g,3)}{flag}",
                flush=True,
            )
            detail[name][(m, n)] = (g, ok)
            if ok:
                per_rule[name].append(g)
        print(flush=True)

    def gm(xs):
        return math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else None

    print("=== RULE GEOMEANS (over 4 in-sample, CORRECT shapes only) ===", flush=True)
    for name, gs in per_rule.items():
        allok = all(detail[name][s][1] for s in IN_SAMPLE)
        print(
            f"  {name:24s}: geomean G = {gm(gs) and round(gm(gs),4)}  "
            f"(n_correct={len(gs)}/4, all_correct={allok})",
            flush=True,
        )


if __name__ == "__main__":
    main()
