"""cross_entropy persistent-vs-looped A/B at MATCHED warps, wide V.

The G_cross_entropy collapse at V=131072 (seed persistent 8067us vs tc 1927us,
default looped 3694us) suggests persistent is WRONG for huge multi-load rows.
A/B: for each (N,V), measure persistent vs several looped chunks, all at the SAME
num_warps (the heuristic's ramp value), holding the persistent lever as the only
difference. Report best looped / persistent so we can see the crossover.

Also sweep warps for the looped path to find the real optimum.
Run with the canonical invocation (SETUP.md).
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

from examples.cross_entropy import cross_entropy  # noqa: E402

LONG = torch.int64
N_RUNS = 7


def build_args(n, v):
    logits = torch.randn(n, v, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device="cuda", dtype=LONG)
    return (logits, labels)


def reference(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels)


def median_do_bench(fn):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(s)[len(s) // 2]


def time_cfg(args, cfg, ref):
    try:
        k = helion.kernel(cross_entropy.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        out = b(*args)
        out = out[0] if isinstance(out, tuple) else out
        ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
        lat = median_do_bench(lambda: b(*args)) * 1000
        return lat, ok
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {str(e)[:60]}"


def main():
    print(f"helion={helion.__file__}\n")
    shapes = [(8192, 32768), (8192, 65536), (8192, 131072), (16384, 65536)]
    chunks = [2048, 4096, 8192, 16384, 32768]
    warps_list = [8, 16, 32]
    for (n, v) in shapes:
        args = build_args(n, v)
        ref = reference(*args)
        print(f"=== ({n},{v}) rnumel={v} ===")
        # persistent at warps 8/16/32
        for w in warps_list:
            lat, ok = time_cfg(args, {"block_sizes": [1], "reduction_loops": [None],
                                      "num_warps": w, "num_stages": 1}, ref)
            print(f"  PERSIST w={w:>2}: {lat if lat is None else round(lat,1)} us  ok={ok}")
        # looped chunks at warps 8/16/32
        for ch in chunks:
            if ch >= v:
                continue
            for w in warps_list:
                lat, ok = time_cfg(args, {"block_sizes": [1], "reduction_loops": [ch],
                                          "num_warps": w, "num_stages": 1}, ref)
                tag = f"  LOOP {ch:>6}/w{w:>2}"
                print(f"{tag}: {lat if lat is None else round(lat,1)} us  ok={ok}")
        # tc baseline
        torch._dynamo.reset()
        tc = torch.compile(reference)
        tc(*args)
        tclat = median_do_bench(lambda: tc(*args)) * 1000
        print(f"  tc: {round(tclat,1)} us")
        print()


if __name__ == "__main__":
    main()
