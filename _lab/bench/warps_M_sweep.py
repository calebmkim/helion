"""Independent re-measurement of the Gate-A-v2 contested shapes: does the narrow-band
warps change (rnumel<=1024 -> w1/w2) regress at LOW M (the axis my original sweep fixed)?

Sweeps M for the contested (kernel, rnumel) pairs, comparing the new seed warps vs the old
ramp's w4, CUDA-graph device time, accuracy-gated, single-process (run ALONE, GPU idle).
This is the M-axis the no-regression backstop requires and my warps_sweep.py missed.
"""
from __future__ import annotations

import json
import os
import sys

import torch

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

_WT = "/home/dev/local/helion-pr-with-lab"
sys.path.insert(0, os.path.join(_WT, "_lab", "bench"))
import bare_fwd_dtype as BF  # noqa: E402


def cg_us(fn, warmup=8, iters=80, reps=15):
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


def old_ramp(rnumel):
    if rnumel > 16384:
        return 32
    if rnumel <= 1024:
        return 4
    if rnumel <= 4096:
        return 8
    return 16


# contested (kernel, dtype, N) — sweep M. N chosen so rnumel<=1024 (the changed band).
CASES = [
    ("softmax", "fp32", 1024),
    ("softmax", "bf16", 1024),
    ("rms_norm", "bf16", 768),
    ("rms_norm", "bf16", 896),
    ("rms_norm", "fp32", 768),
    ("layer_norm", "bf16", 768),
    ("sum", "bf16", 768),
]
MS = [2048, 4096, 8192, 16384, 32768]


def main():
    assert os.path.realpath(helion.__file__).startswith(_WT), helion.__file__
    rows = []
    for (kn, dtn, n) in CASES:
        fn, build, _ = BF.KERNELS[kn]
        dt = BF.DTYPES[dtn]
        for m in MS:
            torch._dynamo.reset()
            args, ref, extract = build(m, n, dt)
            b = fn.bind(args)
            fact = b.env.config_spec.reduction_facts[0]
            seed = compiler_seed_configs(b.env, b.host_function.device_ir)[0]
            new_w = dict(seed).get("num_warps")
            old_w = old_ramp(fact.size_hint)
            k_new = helion.kernel(fn.fn, config=seed, static_shapes=True)
            cfg_old = dict(seed); cfg_old["num_warps"] = old_w
            k_old = helion.kernel(fn.fn, config=helion.Config(**cfg_old), static_shapes=True)
            o = extract(k_new(*args))
            tol = BF._tol(kn, dtn)
            acc = bool(torch.allclose(o.float(), ref.float(), rtol=tol, atol=tol))
            t_new = cg_us(lambda: k_new(*args))
            t_old = cg_us(lambda: k_old(*args))
            ratio = t_new / t_old  # >1 = new (changed) slower than old = REGRESSION
            row = {"kernel": kn, "dtype": dtn, "shape": [m, n], "rnumel": fact.size_hint,
                   "itemsize": fact.itemsize, "new_w": new_w, "old_w": old_w,
                   "new_us": round(t_new, 3), "old_us": round(t_old, 3),
                   "new_vs_old": round(ratio, 4), "acc": acc,
                   "regress_pct": round((ratio - 1) * 100, 1)}
            rows.append(row)
            flag = " <-- REGRESS>10%" if ratio > 1.10 else (" <-- regress" if ratio > 1.02 else "")
            print(f"  {kn:>11} {dtn} ({m:>5},{n}) rnumel={fact.size_hint} new=w{new_w} old=w{old_w} | "
                  f"new={t_new:.2f} old={t_old:.2f} ratio={ratio:.3f} ({(ratio-1)*100:+.1f}%) acc={acc}{flag}",
                  file=sys.stderr)
            json.dump(rows, open("/tmp/warps_M_sweep.json", "w"), indent=2)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
