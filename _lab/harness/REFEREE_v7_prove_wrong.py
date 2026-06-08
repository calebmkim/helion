"""Adversarial: PROVE the divisor-chunk fix is load-bearing at non-pow2 N.
Run welford at (262144,1536) with:
  (a) the emitted v7 seed (combine=512, a DIVISOR of 1536) -> must be CORRECT,
  (b) a MASKED next_pow2 combine (combine=2048, the OLD broken approach) -> must
      be WRONG (Tn=chunk.size(-1) counts padding lanes).
If (b) is also correct, the divisor fix is NOT actually the reason -> investigate.
"""
from __future__ import annotations

import sys
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.welford import welford  # noqa: E402

EPS = 1e-5


def ref_layer_norm(weight, bias, x):
    return torch.nn.functional.layer_norm(
        x, normalized_shape=[x.shape[-1]], weight=weight, bias=bias, eps=EPS
    )


def args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def run_cfg(a, bs, w):
    k = helion.kernel(welford.fn, configs=[helion.Config(
        block_sizes=bs, num_warps=w, num_stages=1)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = (out[0] if isinstance(out, tuple) else out).float()
    return out


def main():
    m, n = 262144, 1536
    a = args(m, n)
    ref = ref_layer_norm(a[0], a[1], a[2]).float()
    print(f"helion={helion.__file__}  shape=({m},{n}) N=1536 (NON-POW2)\n")

    for label, combine in [("v7 DIVISOR combine=512", 512),
                           ("OLD masked next_pow2 combine=2048", 2048),
                           ("naive divisor-violation combine=1024", 1024)]:
        out = run_cfg(a, [16, combine, 2048], 8)
        max_abs = float((out - ref).abs().max())
        ok = bool(torch.allclose(out, ref, rtol=1e-3, atol=1e-3))
        div = "DIVIDES 1536" if 1536 % combine == 0 else "does NOT divide 1536"
        print(f"  combine={combine:>5} ({div}): ok={ok} max_abs={max_abs:.3e}  [{label}]")


if __name__ == "__main__":
    main()
