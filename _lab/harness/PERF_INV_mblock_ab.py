"""Perf-investigator M-block A/B for small-rnumel/large-M reductions.

Question: do the small-N / tiny-rnumel shapes sit below ceiling because the seed
pins the M-block at the autotuner FLOOR (1-2 rows/program)? Does packing more rows
per program (larger M-block) amortize per-row launch/setup overhead and recover the
gap -- WITHOUT regressing large-rnumel shapes (where M-block=floor is correct)?

For each (kernel, shape) we A/B M-block in {floor, 2, 4, 8, 16}, holding EVERY
other lever at the SEED value (reduction_loops, num_warps, num_stages, the R-block
for T2). All fp32. Reports seed-baseline G and each variant's latency + G.

NO heuristic edits. Pure measurement on the bare-seed path (configs=[cfg]).
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_pytorch  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 7
LONG = torch.int64
MBLOCKS = ["floor", 2, 4, 8, 16]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


# --- kernel registry: builder + reference + tc-baseline ---
def build_rms(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),
            torch.randn(n, device="cuda", dtype=torch.float32), 1e-5)


def ref_rms(args):
    x, w, e = args
    return rms_norm_pytorch(x, w, e)


def tc_rms():
    return torch.compile(rms_norm_pytorch)


def build_x(shape):
    m, n = shape
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def ref_sum(args):
    return torch.sum(args[0], dim=-1)


def tc_sum():
    return torch.compile(lambda x: torch.sum(x, dim=-1))


def ref_softmax(args):
    return torch.nn.functional.softmax(args[0], dim=1)


def tc_softmax():
    return torch.compile(lambda x: torch.nn.functional.softmax(x, dim=1))


def build_ce(shape):
    n, v = shape
    return (torch.randn(n, v, device="cuda", dtype=torch.float32),
            torch.randint(0, v, (n,), device="cuda", dtype=LONG))


def ref_ce(args):
    logits, labels = args
    return torch.nn.functional.cross_entropy(logits, labels)


def tc_ce():
    return torch.compile(torch.nn.functional.cross_entropy)


REG = {
    "rms_norm": (rms_norm_fwd, build_rms, ref_rms, tc_rms, 1),
    "sum": (sum_kernel, build_x, ref_sum, tc_sum, 1),
    "long_sum": (longsum, build_x, ref_sum, tc_sum, 1),
    "softmax_two_pass": (softmax_two_pass, build_softmax := build_x, ref_softmax, tc_softmax, 1),
    "cross_entropy": (cross_entropy, build_ce, ref_ce, tc_ce, 2),
}

CASES = [
    # small-N / tiny-rnumel headroom shapes
    ("rms_norm", (32768, 256)),
    ("sum", (8192, 256)),
    ("sum", (32768, 256)),
    ("softmax_two_pass", (32768, 256)),
    ("cross_entropy", (4096, 4096)),
    # large-rnumel no-regression controls
    ("rms_norm", (2048, 16384)),
    ("long_sum", (256, 131072)),
]


def correct(out, ref, nargs):
    o = out[0] if isinstance(out, tuple) else out
    o = o.float()
    r = ref.float()
    if o.shape != r.shape:
        return False, float("nan")
    ok = bool(torch.allclose(o, r, rtol=1e-3, atol=1e-3))
    ma = float((o - r).abs().max())
    return ok, ma


def run_cfg(fn, args, cfg, ref, nargs):
    k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
    bound = k.bind(args)
    bound.ensure_config_exists(args)
    resolved = dict(bound._config)
    out = bound(*args)
    ok, ma = correct(out, ref, nargs)
    lat = med(lambda: bound(*args))
    return lat, resolved, ok, ma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="restrict to kernel name")
    a = ap.parse_args()
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  N_RUNS={N_RUNS}\n")

    for name, shape in CASES:
        if a.only and name != a.only:
            continue
        fn, build, reffn, tcfn, nargs = REG[name]
        args = build(shape)
        ref = reffn(args)

        # tc-default baseline
        torch._dynamo.reset()
        tc = tcfn()
        _ = tc(*args)
        tc_lat = med(lambda: tc(*args))

        # seed
        bound0 = fn.bind(args)
        seed = dict(compiler_seed_configs(bound0.env, bound0.host_function.device_ir)[0])
        floor_mblock = seed["block_sizes"][0]

        print(f"=== {name} {shape}  (rnumel={shape[1]})  tc={tc_lat*1000:.2f}us  "
              f"floor_mblock={floor_mblock}  seed={seed} ===")
        seed_lat, _, ok0, ma0 = run_cfg(fn, args, seed, ref, nargs)
        print(f"  {'mblock':>8} {'lat_us':>9} {'G':>7} {'vs_seed':>8} {'ok':>4} {'maxabs':>9}  cfg")
        print(f"  {'SEED('+str(floor_mblock)+')':>8} {seed_lat*1000:>9.2f} "
              f"{tc_lat/seed_lat:>7.3f} {1.0:>8.3f} {str(ok0):>4} {ma0:>9.1e}")

        for mb in MBLOCKS:
            mblock = floor_mblock if mb == "floor" else mb
            if mb == "floor":
                continue  # already printed as SEED
            if mblock < floor_mblock:
                continue  # below floor is illegal; skip
            cfg = dict(seed)
            cfg["block_sizes"] = list(seed["block_sizes"])
            cfg["block_sizes"][0] = mblock
            try:
                lat, resolved, ok, ma = run_cfg(fn, args, cfg, ref, nargs)
            except Exception as exc:
                print(f"  {mb:>8} {'ERR':>9}  {type(exc).__name__}: {str(exc)[:60]}")
                continue
            # confirm the M-block actually resolved to what we asked (normalize can clamp)
            got_mb = resolved.get("block_sizes", [None])[0]
            note = "" if got_mb == mblock else f" (resolved mblock={got_mb})"
            print(f"  {mb:>8} {lat*1000:>9.2f} {tc_lat/lat:>7.3f} "
                  f"{seed_lat/lat:>8.3f} {str(ok):>4} {ma:>9.1e}{note}")
        print()


if __name__ == "__main__":
    main()
