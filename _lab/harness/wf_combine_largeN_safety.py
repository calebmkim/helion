"""Bound the v8 combine cap (apply-looped regime) at LARGE N. The win at N=4096 needs
combine up to 4096. But an UNBOUNDED combine (=lpd(N)) could spill at huge N even with a
looped apply. Probe N=16384 and N=32768 (OUT-OF-SAMPLE generalization/safety, not tuned):
sweep combine {2048,4096,8192,16384,...} with apply looped at 2048, find where combine
stops helping / starts spilling. -> set a spill-safe combine cap.
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


def med(fn, reps=3):
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
    for (m, n) in [(65536, 16384), (32768, 32768)]:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        w = 32
        torch._dynamo.reset()
        tc = torch.compile(eager_layer_norm)
        tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        print(f"=== ({m},{n}) w{w} tc={round(tclat,1)}us; apply looped @2048 ===", flush=True)
        for cb in (2048, 4096, 8192, 16384):
            if cb > n:
                continue
            lat, ok, mx = run(a, [16, cb, 2048], w, ref)
            print(
                f"  combine={cb:6d} apply=2048: {round(lat,1):9}us ok={ok} "
                f"maxabs={mx:.1e} G={round(tclat/lat,3)}",
                flush=True,
            )
        print(flush=True)


if __name__ == "__main__":
    main()
