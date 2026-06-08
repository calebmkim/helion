"""Cross-kernel / cross-dtype num_warps sweep (CUDA-graph device time) to validate the
byte-keyed warps ramp (ITER5). For each (kernel, dtype, shape) it holds the seed config
fixed and sweeps ONLY num_warps, reporting: the optimum, the current-ramp pick, the
proposed-byte-ramp pick, and the regret of each vs optimum. The no-regression backstop for
the shared _num_warps ramp is discharged by THIS measurement.

Reuses bare_fwd_dtype build fns. CUDA-graph (host-overhead-free). Foreground, JSON-checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

_WT = "/home/dev/local/helion-pr-with-lab"
sys.path.insert(0, os.path.join(_WT, "_lab", "bench"))
import bare_fwd_dtype as BF  # noqa: E402


def cg_us(fn, warmup=5, iters=50, reps=11):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / iters * 1000.0)
    return sorted(ts)[len(ts) // 2]


def current_ramp(rnumel: int) -> int:
    if rnumel > 16384:
        return 32
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    return 16


def proposed_byte_ramp(rnumel: int, itemsize: int) -> int:
    b = rnumel * max(1, itemsize)
    if b <= 2048:
        return 1
    if b <= 4096:
        return 2
    if b <= 8192:
        return 4
    if b <= 32768:
        return 8
    if b <= 65536:
        return 16
    return 32


WARPS = [1, 2, 4, 8, 16, 32]


def sweep_one(kn, dt_name, m, n):
    fn, build, _ = BF.KERNELS[kn]
    dt = BF.DTYPES[dt_name]
    args, ref, extract = build(m, n, dt)
    tol = BF._tol(kn, dt_name)
    torch._dynamo.reset()
    b0 = fn.bind(args)
    fact = b0.env.config_spec.reduction_facts[0]
    seed = compiler_seed_configs(b0.env, b0.host_function.device_ir)[0]
    sd = dict(seed)
    seedw = sd.get("num_warps")
    rnumel = fact.size_hint
    itemsize = fact.itemsize
    times = {}
    for w in WARPS:
        cfgd = dict(sd); cfgd["num_warps"] = w
        try:
            k = helion.kernel(fn.fn, config=helion.Config(**cfgd), static_shapes=True)
            out = extract(k(*args))
            ok = bool(torch.allclose(out.float(), ref.float(), rtol=tol, atol=tol))
            times[w] = round(cg_us(lambda: k(*args)), 3) if ok else None
        except Exception:  # noqa: BLE001
            times[w] = None
    valid = {w: t for w, t in times.items() if t is not None}
    if not valid:
        return None
    best_w = min(valid, key=lambda w: valid[w])
    cur_w = current_ramp(rnumel)
    prop_w = proposed_byte_ramp(rnumel, itemsize)
    best_t = valid[best_w]

    def regret(w):
        return round(valid[w] / best_t, 4) if w in valid else None
    return {"kernel": kn, "dtype": dt_name, "shape": [m, n], "rnumel": rnumel,
            "itemsize": itemsize, "bytes": rnumel * itemsize,
            "times": times, "best_w": best_w, "best_us": best_t,
            "seed_w": seedw, "cur_ramp_w": cur_w, "prop_ramp_w": prop_w,
            "regret_cur": regret(cur_w), "regret_prop": regret(prop_w),
            "regret_seed": regret(seedw)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="bf16,fp32,fp16")
    ap.add_argument("--out", default="/tmp/warps_sweep.json")
    ap.add_argument("--shapes", default=None, help="kernel:M,N;kernel:M,N override")
    ap.add_argument("kernels", nargs="*")
    a = ap.parse_args()
    assert os.path.realpath(helion.__file__).startswith(_WT), helion.__file__
    dtypes = a.dtypes.split(",")
    # default shape set: narrow→wide per kernel family
    DEFAULT = {
        "softmax": [(8192, 512), (8192, 1024), (8192, 2048), (4096, 4096), (2048, 8192)],
        "rms_norm": [(8192, 512), (8192, 1024), (8192, 2048), (4096, 4096), (2048, 8192)],
        "layer_norm": [(8192, 512), (8192, 1024), (8192, 2048), (4096, 4096), (2048, 8192)],
        "sum": [(16384, 512), (16384, 1024), (16384, 2048), (8192, 4096), (8192, 8192)],
        "cross_entropy": [(8192, 32768), (4096, 50257), (2048, 128256)],
        "kl_div": [(8192, 32768), (4096, 50257), (2048, 128256)],
        "welford": [(16384, 512), (16384, 1024), (16384, 2048), (8192, 4096)],
        "long_sum": [(256, 65536), (128, 131072), (64, 262144)],
    }
    kernels = a.kernels or list(DEFAULT)
    out = []
    for kn in kernels:
        shapes = DEFAULT[kn]
        for dt in dtypes:
            for (m, n) in shapes:
                try:
                    r = sweep_one(kn, dt, m, n)
                except Exception as e:  # noqa: BLE001
                    r = {"kernel": kn, "dtype": dt, "shape": [m, n], "error": str(e)[:160]}
                if r:
                    out.append(r)
                    rc = r.get("regret_cur"); rp = r.get("regret_prop")
                    print(f"  {kn:>11} {dt} ({m},{n}) rnumel={r.get('rnumel')} bytes={r.get('bytes')} "
                          f"| best=w{r.get('best_w')} cur=w{r.get('cur_ramp_w')}(reg {rc}) "
                          f"prop=w{r.get('prop_ramp_w')}(reg {rp})", file=sys.stderr)
                    json.dump(out, open(a.out, "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
