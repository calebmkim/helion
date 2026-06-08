"""Is the persistent->looped crossover at huge rnumel a GENERAL multi-load
wide-row property, or cross_entropy-specific?

A/B persistent vs best-looped at huge rnumel (131072, 262144) for:
  - rms_norm_fwd (num_load=2, 1 reduction pass effectively but re-reads x)
  - layer_norm_fwd (num_load=3)
  - cross_entropy (num_load=3, amax + exp-sum = 2 passes over the row)

This tells us whether "persistent to the structural cap" is wrong in general for
wide multi-load rows, or whether cross_entropy is special (the gather + double
pass). Holds warps EQUAL per comparison; reports best looped vs persistent.

Run with the canonical invocation (SETUP.md).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402

LONG = torch.int64
N_RUNS = 5
EPS = 1e-5


def median_do_bench(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def time_cfg(fn, args, cfg):
    try:
        k = helion.kernel(fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        b(*args)
        return median_do_bench(lambda: b(*args)) * 1000
    except Exception as e:  # noqa: BLE001
        import traceback
        tb = traceback.format_exc().strip().splitlines()[-3:]
        return f"{type(e).__name__}: {' | '.join(tb)}"


def ce_args(n, v):
    return (torch.randn(n, v, device="cuda", dtype=torch.float32),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG))


def rms_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), EPS)


def ln_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32), [n],
            torch.randn(n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), EPS)


def probe(label, fn, args, v, mblock=1):
    best_p = None
    p_err = None
    for w in (8, 16, 32):
        x = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [None],
                                "num_warps": w, "num_stages": 1})
        if isinstance(x, float):
            best_p = x if best_p is None else min(best_p, x)
        else:
            p_err = x
    best_l = None
    l_err = None
    for ch in (8192, 16384, 32768):
        if ch >= v:
            continue
        for w in (16, 32):
            x = time_cfg(fn, args, {"block_sizes": [mblock], "reduction_loops": [ch],
                                    "num_warps": w, "num_stages": 1})
            if isinstance(x, float):
                best_l = x if best_l is None else min(best_l, x)
            else:
                l_err = x
    ratio = (best_p / best_l) if (best_p and best_l) else None
    print(f"  {label:<32} bestPersist={best_p and round(best_p,1)}us  "
          f"bestLoop={best_l and round(best_l,1)}us  "
          f"P/L={ratio and round(ratio,2)} (>1 => looped wins)"
          + (f"  [perr={p_err}]" if best_p is None else "")
          + (f"  [lerr={l_err}]" if best_l is None else ""))


def main():
    print(f"helion={helion.__file__}\n")
    for v in (131072, 262144):
        print(f"### rnumel = {v} ###")
        probe(f"rms_norm (8,{v}) nl=2", rms_norm_fwd.fn, rms_args(8, v), v)
        probe(f"rms_norm (1024,{v}) nl=2", rms_norm_fwd.fn, rms_args(1024, v), v)
        probe(f"layer_norm (8,{v}) nl=3", layer_norm_fwd.fn, ln_args(8, v), v)
        probe(f"layer_norm (1024,{v}) nl=3", layer_norm_fwd.fn, ln_args(1024, v), v)
        probe(f"cross_entropy (1024,{v}) nl=3", cross_entropy.fn, ce_args(1024, v), v)
        probe(f"cross_entropy (8192,{v}) nl=3", cross_entropy.fn, ce_args(8192, v), v)
        print()


if __name__ == "__main__":
    main()
