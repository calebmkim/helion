"""CRITICAL: the fresh oracle at (262144,4096) reports [16,2048,2048] w8 = G0.968, but the
SAME block_sizes via configs=[seed] measures G0.722. Isolate WHY: is the win from codegen
knobs (indexing / eviction / range_*) the bare seed does NOT set, or from block_sizes?

Re-bench:
  A) oracle FULL verbatim winner (all knobs)            -> the 0.968 number
  B) oracle block_sizes ONLY, default knobs             -> isolates the block_sizes effect
  C) v8 seed [16,4096,2048] default knobs               -> our seed
  D) [16,2048,2048] + oracle's indexing/eviction knobs  -> add knobs onto the brief config
This decides whether the 0.968 is SEEDABLE (block_sizes) or autotuner-only (knobs).
"""
from __future__ import annotations

import os
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
M, N = 262144, 4096


def args(m, n):
    return (
        torch.rand(n, device="cuda"),
        torch.rand(n, device="cuda"),
        torch.rand(m, n, device="cuda"),
        EPS,
    )


def med(fn, reps=5):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def rebench(a, cfg, ref, label):
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg)])
    b = k.bind(a)
    b.ensure_config_exists(a)
    out = b(*a)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
    mx = float((out.float() - ref.float()).abs().max())
    lat = med(lambda: b(*a)) * 1000
    print(f"  {label:42s}: {round(lat,1):8}us ok={ok} maxabs={mx:.1e}", flush=True)
    return lat


def main():
    print(f"helion={helion.__file__}  ({M},{N})\n", flush=True)
    a = args(M, N)
    ref = eager_layer_norm(*a)
    torch._dynamo.reset()
    tc = torch.compile(eager_layer_norm)
    tc(*a)
    tclat = med(lambda: tc(*a)) * 1000
    print(f"tc = {round(tclat,1)}us\n", flush=True)

    # get a fresh oracle winner (full verbatim, all knobs)
    os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"
    k = helion.kernel(welford.fn)
    b = k.bind(a)
    b.autotune(a)
    win = dict(b._config)
    print(f"ORACLE winner full config keys: {sorted(win.keys())}", flush=True)
    print(f"  block_sizes={win.get('block_sizes')} indexing={win.get('indexing')} "
          f"load_eviction_policies={win.get('load_eviction_policies')} "
          f"num_warps={win.get('num_warps')} num_stages={win.get('num_stages')}\n", flush=True)

    base = {"block_sizes": [16, 2048, 2048], "num_warps": 8, "num_stages": 1}
    knob_keys = ["indexing", "load_eviction_policies", "range_flattens",
                 "range_multi_buffers", "range_num_stages", "range_unroll_factors",
                 "range_warp_specializes", "pid_type", "atomic_indexing"]
    knobs = {kk: win[kk] for kk in knob_keys if kk in win}

    la = rebench(a, win, ref, "A) oracle FULL verbatim (all knobs)")
    lb = rebench(a, base, ref, "B) [16,2048,2048] DEFAULT knobs")
    lc = rebench(a, {"block_sizes": [16, 4096, 2048], "num_warps": 8, "num_stages": 1},
                 ref, "C) v8 seed [16,4096,2048] DEFAULT knobs")
    ld = rebench(a, {**base, **knobs}, ref, "D) [16,2048,2048] + oracle KNOBS")
    print(f"\nG: A_oracle={round(tclat/la,3)}  B_bare={round(tclat/lb,3)}  "
          f"C_v8seed={round(tclat/lc,3)}  D_brief+knobs={round(tclat/ld,3)}", flush=True)


if __name__ == "__main__":
    main()
