"""Confirm the FINAL welford seed = the exact config the heuristic will emit, using
the EXISTING _num_warps rnumel ramp (1024->4, 1536/2048/4096->8) and combine cap.
Measure G per shape + geomean, all CORRECT. This is the seed A/B'd vs default + the
per-shape oracle ceilings already measured.
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
from examples.welford import welford, eager_layer_norm  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 1e-5
IN_SAMPLE = [(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096)]
COMBINE_CAP = 2048  # bytes-equivalent: cap the structured-combine chunk


def args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def med(fn, reps=4):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def mfloor(a):
    bs = welford.bind(a).config_spec.block_sizes[0]
    return max(1, bs.min_size, bs.autotuner_min)


def num_warps(rnumel):
    if rnumel > 16384:
        return 32
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    return 16


def run(a, bs, w, ref):
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


def main():
    print(f"helion={helion.__file__}  COMBINE_CAP={COMBINE_CAP}\n", flush=True)
    gs = []
    for (m, n) in IN_SAMPLE:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        mf = mfloor(a); P = np2(n); L = n & (-n)
        combine = min(L, COMBINE_CAP)
        w = num_warps(n)
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm); tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        lat, ok, maxabs = run(a, [mf, combine, P], w, ref)
        g = tclat / lat
        flag = "" if ok else "  <-- WRONG"
        print(f"({m},{n}) seed=[{mf},{combine},{P}] w{w}: {round(lat,1)}us "
              f"tc={round(tclat,1)}us ok={ok} maxabs={maxabs:.1e} G={round(g,3)}{flag}",
              flush=True)
        if ok:
            gs.append(g)
    gm = math.exp(sum(math.log(v) for v in gs) / len(gs)) if gs else None
    print(f"\nG_welford GEOMEAN = {round(gm,4)}  (over {len(gs)} shapes, all CORRECT)", flush=True)


if __name__ == "__main__":
    main()
