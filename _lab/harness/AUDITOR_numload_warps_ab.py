"""AUDITOR INDEPENDENT A/B: does num_warps=32 help num_load==1 and hurt
num_load==2 at MATCHED rnumel and equal other levers?

This is the auditor's OWN test, not the worker's sum-vs-rms_norm pair (which
could differ by kernel identity, not just num_load). We build two SYNTHETIC
Helion reduction kernels with the SAME reduction structure (block_sizes=[1],
persistent reduction over the last dim) that differ ONLY in num_load:

  red1  (num_load==1): out[m] = sum(x[m,:])            -- one load of x
  red2  (num_load==2): s=sum(x[m,:]); out[m,:]=x*(1/s) -- TWO loads of x
                       (re-read pattern, like rms_norm's normalize pass)

For each, at matched rnumel, persistent path, num_stages=1, we A/B num_warps in
{16,32} (the decisive pair the gate keys on) and also 4/8 for context.

Prints, per (kernel,rnumel,M): per-warp median-of-9 us, best warp, and the
w32/w16 ratio (>1 => w32 HURTS, <1 => w32 HELPS). The gate generalizes iff
red1 wants w32 (ratio<1) at large rnumel AND red2 does not (ratio>=1).
"""

from __future__ import annotations

import os
import sys

import torch

import helion
import helion.language as hl

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 9
WARPS = [4, 8, 16, 32]
RNUMELS = [16384, 32768, 65536, 131072, 262144]
MS = [1, 16]


@helion.kernel()
def red1(x: torch.Tensor) -> torch.Tensor:
    """num_load==1: single load + reduce."""
    m, _ = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        out[tile_m] = x[tile_m, :].sum(-1)
    return out


@helion.kernel()
def red2(x: torch.Tensor) -> torch.Tensor:
    """num_load==2: load+reduce to scalar, then load x AGAIN to normalize."""
    m, n = x.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        s = x[tile_m, :].sum(-1)  # load #1
        out[tile_m, :] = x[tile_m, :] * (1.0 / (s[:, None] + 1.0))  # load #2
    return out


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def time_cfg(kfn, x, warps):
    cfg = helion.Config(block_sizes=[1], reduction_loops=[None],
                        num_warps=warps, num_stages=1)
    k = helion.kernel(kfn.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    tcode = b.to_triton_code(helion.Config(**dict(b._config)))
    assert "for roffset" not in tcode, "expected persistent codegen"
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    return med(lambda: b(x)) * 1000


def report_numload(kfn, x):
    bound = kfn.bind((x,))
    fact = bound.env.config_spec.reduction_facts[0]
    return fact.num_load


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  N_RUNS={N_RUNS}  persistent path\n")
    # report num_load for each synthetic kernel
    xprobe = torch.randn(1, 4096, device="cuda", dtype=torch.float32)
    nl1 = report_numload(red1, xprobe)
    nl2 = report_numload(red2, xprobe)
    print(f"red1 num_load={nl1}   red2 num_load={nl2}\n")
    assert nl1 == 1, f"red1 should be num_load==1, got {nl1}"
    assert nl2 == 2, f"red2 should be num_load==2, got {nl2}"

    for kfn, label, nl in [(red1, "red1", nl1), (red2, "red2", nl2)]:
        print(f"=== {label} (num_load={nl}) ===")
        print(f"{'rnumel':>8} {'M':>4} | " + " ".join(f"{'w'+str(w):>8}" for w in WARPS)
              + " | best | w32/w16")
        for n in RNUMELS:
            for m in MS:
                x = torch.randn(m, n, device="cuda", dtype=torch.float32)
                t = {w: time_cfg(kfn, x, w) for w in WARPS}
                best_w = min(t, key=t.get)
                ratio = t[32] / t[16]
                print(f"{n:>8} {m:>4} | " + " ".join(f"{t[w]:>8.2f}" for w in WARPS)
                      + f" | w{best_w:>2} | {ratio:.3f}")
        print()


if __name__ == "__main__":
    main()
