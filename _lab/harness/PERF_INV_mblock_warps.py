"""Perf-investigator: joint M-block x num_warps grid for the headroom shapes.

The v3 lesson: warps x block are COUPLED. A bigger M-block packs more rows/program
-> more independent work -> may want more warps. So the M-block win could be an
artifact of holding warps at the seed floor. A/B the FULL grid (M-block x warps),
re-bench each point verbatim, so we know whether the win is a real generalizable
M-block lever or just a warps-coupling.

Also reports the grid size (M / mblock) so we can see the per-row-overhead story.
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

from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def build_rms(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), 1e-5)


def build_x(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


REG = {
    "rms_norm": (rms_norm_fwd, build_rms, lambda a: rms_norm_pytorch(*a),
                 lambda: torch.compile(rms_norm_pytorch)),
    "sum": (sum_kernel, build_x, lambda a: torch.sum(a[0], -1),
            lambda: torch.compile(lambda x: torch.sum(x, -1))),
    "softmax_two_pass": (softmax_two_pass, build_x,
                         lambda a: torch.nn.functional.softmax(a[0], 1),
                         lambda: torch.compile(lambda x: torch.nn.functional.softmax(x, 1))),
}

CASES = [
    ("rms_norm", (32768, 256)),
    ("softmax_two_pass", (32768, 256)),
    # large-rnumel control: confirm bigger M-block + more warps STILL regresses
    ("rms_norm", (2048, 16384)),
]
MBLOCKS = [1, 2, 4, 8, 16, 32]
WARPS = [4, 8, 16, 32]


def correct(out, ref):
    o = out[0] if isinstance(out, tuple) else out
    o = o.float(); r = ref.float()
    return bool(o.shape == r.shape and torch.allclose(o, r, rtol=1e-3, atol=1e-3))


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}\n")
    for name, shape in CASES:
        fn, build, reffn, tcfn = REG[name]
        args = build(shape)
        ref = reffn(args)
        torch._dynamo.reset()
        tc = tcfn()
        _ = tc(*args)
        tc_lat = med(lambda: tc(*args))
        bound0 = fn.bind(args)
        seed = dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])
        floor = seed["block_sizes"][0]
        print(f"=== {name} {shape} (rnumel={shape[1]}) tc={tc_lat*1000:.2f}us "
              f"floor_mblock={floor} seed_warps={seed['num_warps']} ===")
        print(f"  G = tc/lat. grid = M/mblock = {shape[0]}/mblock")
        hdr = "  mblock\\warps " + " ".join(f"{w:>8}" for w in WARPS)
        print(hdr)
        best = (None, None, 0.0)
        for mb in MBLOCKS:
            if mb < floor:
                continue
            row = f"  {mb:>11} "
            for w in WARPS:
                cfg = dict(seed)
                cfg["block_sizes"] = list(seed["block_sizes"])
                cfg["block_sizes"][0] = mb
                cfg["num_warps"] = w
                try:
                    k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
                    b = k.bind(args)
                    b.ensure_config_exists(args)
                    out = b(*args)
                    ok = correct(out, ref)
                    lat = med(lambda: b(*args))
                    g = tc_lat / lat
                    if ok and g > best[2]:
                        best = (mb, w, g)
                    row += f"{g:>8.3f}" if ok else f"{'BAD':>8}"
                except Exception as exc:
                    row += f"{'ERR':>8}"
            print(row)
        print(f"  BEST: mblock={best[0]} warps={best[1]} G={best[2]:.3f}  "
              f"(seed G at floor/seed_warps printed above)\n")


if __name__ == "__main__":
    main()
