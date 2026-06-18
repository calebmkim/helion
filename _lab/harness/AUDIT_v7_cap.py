"""AUDITOR cap-honesty probe at (262144,4096) and a wide non-pow2.

The oracle at (262144,4096) found bs=[16,4096,2048] w16 G=0.95, but the SEED caps
combine at 2048 (8KiB) and uses w8 -> G=0.71. Is the cap leaving ~34% on the
table? Sweep combine_block x num_warps (apply fixed at the seed's value) to
isolate which lever costs the gap. Also probe a few wider N to see if the cap is
fitted to the in-sample set or genuinely structural.
"""
from __future__ import annotations

import sys
import torch
import torch.nn.functional as F
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402
from examples.welford import welford  # noqa: E402

EPS = 1e-5


def eager_ln(weight, bias, x, eps):
    return F.layer_norm(x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=eps)


def args(m, n):
    g = torch.Generator(device="cuda").manual_seed(0)
    return (torch.rand(n, device="cuda", generator=g),
            torch.rand(n, device="cuda", generator=g),
            torch.rand(m, n, device="cuda", generator=g), EPS)


def med(fn, reps=5):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def lat_for(a, bs, w, ref):
    cfg = {"block_sizes": bs, "num_warps": w, "num_stages": 1}
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = (out[0] if isinstance(out, tuple) else out).float()
    ok = bool(torch.allclose(out, ref.float(), rtol=1e-2, atol=1e-1))
    return med(lambda: b(*a)) * 1000, ok


def main():
    print(f"helion={helion.__file__}", flush=True)
    print(f"dev={torch.cuda.get_device_name(0)}\n", flush=True)
    # (M, N, mfloor, [combine candidates], apply=np2(N))
    SWEEPS = [
        (262144, 4096, 16, [512, 1024, 2048, 4096], 4096),
        (262144, 2048, 16, [512, 1024, 2048], 2048),
        (262144, 8192, 16, [1024, 2048, 4096, 8192], 8192),
    ]
    for (m, n, mf, combines, ap) in SWEEPS:
        a = args(m, n)
        ref = eager_ln(*a)
        torch._dynamo.reset()
        tc = torch.compile(eager_ln); tc(*a)
        tclat = med(lambda: tc(*a)) * 1000
        print(f"=== ({m},{n}) tc={tclat:.1f}us  apply={ap} ===", flush=True)
        for cb in combines:
            for w in (8, 16):
                lat, ok = lat_for(a, [mf, cb, ap], w, ref)
                tag = "  <-SEED" if (cb == min(2048, n) and w == 8) else ""
                print(f"   combine={cb:>5} w={w:>2}: {lat:>8.1f}us ok={ok} "
                      f"G={tclat/lat:.3f}{tag}", flush=True)
        print(flush=True)
        del a, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
