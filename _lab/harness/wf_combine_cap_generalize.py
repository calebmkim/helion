"""Set the v8 combine cap SAFELY for the apply-looped regime. With the apply tile looped
(capped at 2048), how large can the combine tile go before IT spills? Sweep combine across
{1024,2048,4096,8192} at N in {4096, 8192} (8192 is OUT-OF-SAMPLE -- a generalization /
spill-safety check, NOT tuned to). apply fixed looped at 2048.

This decides whether to leave the combine UNCAPPED (=lpd) or cap it at e.g. 4096 to be
spill-safe at larger N than the in-sample set.
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
    # M reduced for 8192 to keep numel sane (welford u0*u1 constraint not an issue; just memory)
    for (m, n) in [(262144, 4096), (131072, 8192)]:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        w = 8 if n <= 4096 else 16
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm)
        tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        print(f"=== ({m},{n}) w{w} tc={round(tclat,1)}us; apply looped @2048 ===", flush=True)
        combines = [c for c in (1024, 2048, 4096, 8192) if c <= n]
        for cb in combines:
            lat, ok, mx = run(a, [16, cb, 2048], w, ref)
            print(
                f"  combine={cb:5d} apply=2048: {round(lat,1):8}us ok={ok} "
                f"maxabs={mx:.1e} G={round(tclat/lat,3)}",
                flush=True,
            )
        print(flush=True)


if __name__ == "__main__":
    main()
