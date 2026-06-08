"""Disentangle the welford spill: the heuristic comment claims combine=4096 (full-N)
SPILLS to G~0.13 at (262144,4096). The grid in wf_brief_repro shows combine=4096 is
catastrophic ONLY when apply is ALSO 4096; with apply=2048 (looped) combine=4096 is the
BEST (G=0.76). Pin this down: sweep combine {2048,4096} x apply {2048,4096} at (4096)
AND check the same at (2048) and the non-pow2 (1536) for correctness/spill.

This decides whether the ~7% residual at 4096 is reachable by capping the APPLY tile
alone (combine stays 2048) or REQUIRES raising the combine tile to full-N (4096) too —
which the existing STRUCTURED_COMBINE_CAP_BYTES forbids for spill reasons.
"""
from __future__ import annotations

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


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    for (m, n) in [(262144, 4096), (262144, 2048), (262144, 1536)]:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm)
        tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        print(f"=== ({m},{n}) tc={round(tclat,1)}us ===", flush=True)
        # combine x apply at the high end + warp sweep on the winner
        for cb in (2048, 4096):
            for ap in (2048, 4096):
                lat, ok, mx = run(a, [16, cb, ap], 8, ref)
                print(
                    f"  combine={cb} apply={ap} w8: {round(lat,1):8}us "
                    f"ok={ok} maxabs={mx:.1e} G={round(tclat/lat,3)}",
                    flush=True,
                )
        print(flush=True)


if __name__ == "__main__":
    main()
