"""Reconcile the 4096 oracle: pin the oracle winner [16,4096,2048] and several
nearby points (full combine + LOOPED apply) to confirm G~0.95 reproduces and is
CORRECT, vs the seed's [16,2048,4096] (capped combine + PERSISTENT apply) G~0.71.
This isolates whether the seed's 4096 'floor' is a structure ceiling (worker's
claim) or a config the seed simply does not explore.
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
M, N = 262144, 4096


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
    maxabs = float((out - ref.float()).abs().max())
    return med(lambda: b(*a)) * 1000, ok, maxabs


def main():
    a = args(M, N)
    ref = eager_ln(*a)
    torch._dynamo.reset()
    tc = torch.compile(eager_ln); tc(*a)
    tclat = med(lambda: tc(*a)) * 1000
    print(f"helion={helion.__file__}  ({M},{N}) tc={tclat:.1f}us\n", flush=True)
    # roles: bs = [M_floor, combine(block_id1), apply(block_id2)]
    CFGS = [
        ([16, 2048, 4096], 8, "SEED (cap combine=2048, PERSIST apply=4096)"),
        ([16, 4096, 2048], 16, "ORACLE-winner (FULL combine=4096, LOOPED apply=2048)"),
        ([16, 4096, 2048], 8, "full combine + looped apply, w8"),
        ([16, 4096, 4096], 16, "full combine + persist apply (both 4096)"),
        ([16, 2048, 2048], 16, "cap combine + looped apply, w16"),
    ]
    for bs, w, label in CFGS:
        lat, ok, mx = lat_for(a, bs, w, ref)
        print(f"  bs={bs} w={w:>2}: {lat:>8.1f}us ok={ok} maxabs={mx:.1e} "
              f"G={tclat/lat:.3f}   {label}", flush=True)


if __name__ == "__main__":
    main()
