"""Standalone elementwise 3-arm bench for the pointwise seed heuristic.

DO NOT measure through the swiglu/geglu tritonbench operators (they bench the full
matmul-heavy MLP; task §6). This harness times the STANDALONE elementwise op on
pre-projected [M,N] tensors, cold-L2 med-of-9 do_bench, acc-gate BEFORE timing,
dynamo-reset per shape, single-GPU pin.

Arms (all on the SAME pre-projected tensors):
  default : config_spec.default_config()  (the broken block_size=32 baseline)
  seeded  : compiler_seed_configs(...)     (the pointwise seed; configs=[seed], no autotune)
  tc      : torch.compile(ref, mode='max-autotune-no-cudagraphs')  (parity target)

Reports per shape:  seeded_vs_default = default/seed ,  G = seeded_vs_tc = tc/seed
(G >= 1 = seed beats tc;  floor = 0.75).  Per-(kernel,dtype) geomean over acc-passing rows.

Run (cwd=/tmp):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-pointwise \
    /home/calebkim/.conda/envs/helion/bin/python \
    /home/calebkim/helion-new-heuristics/helion-pointwise/_lab/pointwise/ptw_bench.py \
    --kernels swiglu,geglu --splits train,val
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys

import torch

import helion

WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
assert helion.__file__.startswith(WT), f"WRONG helion: {helion.__file__}"

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

sys.path.insert(0, f"{WT}/_lab/prompts")
sys.path.insert(0, f"{WT}/_lab/pointwise")
import shapes_pointwise_draft as SH  # noqa: E402

DEV = "cuda"
N_RUNS = 9
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
RTOL, ATOL = 0.05, 0.005
FLOOR = 0.75


# ---- standalone elementwise references (match the kernel's fp32-internal compute) ----
def ref_swiglu(a, b):
    return torch.nn.functional.silu(a.float()).to(b.dtype) * b


def ref_geglu(a, b):
    af = a.float()
    g = 0.5 * af * (1.0 + torch.tanh(0.7978845608028654 * (af + 0.044715 * af * af * af)))
    return g.to(b.dtype) * b


def ref_residual_add(x, r):
    return x + r


def _ab(m, n, dt):
    return (torch.randn(m, n, device=DEV, dtype=dt), torch.randn(m, n, device=DEV, dtype=dt))


# kernel registry: name -> (helion kernel fn, ref fn, build_args(m,n,dt)->tuple)
def _kernels():
    from examples.add import add as _add
    from examples.geglu import _geglu
    from examples.swiglu import _swiglu_fwd

    import ptw_kernels as PK

    return {
        "swiglu": (_swiglu_fwd, ref_swiglu, _ab),
        "geglu": (_geglu, ref_geglu, _ab),
        "residual_add": (_add, ref_residual_add, _ab),
        "relu_squared": (PK.relu_squared, PK.ref_relu_squared,
                         lambda m, n, dt: (torch.randn(m, n, device=DEV, dtype=dt),)),
        "bias_gelu": (PK.bias_gelu, PK.ref_bias_gelu,
                      lambda m, n, dt: (torch.randn(m, n, device=DEV, dtype=dt),
                                        torch.randn(n, device=DEV, dtype=dt))),
        "dyt": (PK.dyt, PK.ref_dyt,
                lambda m, n, dt: (torch.randn(m, n, device=DEV, dtype=dt),
                                  torch.randn(n, device=DEV, dtype=dt),
                                  torch.randn(n, device=DEV, dtype=dt), 0.5)),
    }


def med(fn):
    from triton.testing import do_bench

    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2]


def force(kfn, args, cfg):
    if cfg is None:
        return None
    k = helion.kernel(kfn.fn, config=cfg, static_shapes=True)
    return lambda: k(*args)


def acc(call, ref_out):
    if call is None:
        return "no-config"
    try:
        out = call()
        torch.testing.assert_close(out.float(), ref_out.float(), rtol=RTOL, atol=ATOL)
        return True
    except Exception as e:  # noqa: BLE001
        return f"FAIL:{str(e)[:70]}"


def bench_kernel(name, kfn, ref, build_args, shapes, dt, traffic):
    rows = []
    for (m, n) in shapes:
        torch._dynamo.reset()
        args = build_args(m, n, dt)
        ref_out = ref(*args)

        bound = kfn.bind(args)
        default_cfg = bound.config_spec.default_config()
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        seed_cfg = seeds[0] if seeds else None

        d_call = force(kfn, args, default_cfg)
        s_call = force(kfn, args, seed_cfg)
        d_acc, s_acc = acc(d_call, ref_out), acc(s_call, ref_out)

        tc = torch.compile(ref, mode="max-autotune-no-cudagraphs")
        tc(*args)  # warmup/autotune
        t_acc = acc(lambda: tc(*args), ref_out)

        td = med(d_call) if d_acc is True else float("nan")
        ts = med(s_call) if s_acc is True else float("nan")
        tt = med(lambda: tc(*args)) if t_acc is True else float("nan")
        gbps = traffic * m * n * dt.itemsize / 1e12

        row = {
            "kernel": name, "shape": [m, n], "dtype": dt_name(dt),
            "us": {"default": rd(td * 1e3), "seeded": rd(ts * 1e3), "tc": rd(tt * 1e3)},
            "seeded_vs_default": rd(td / ts) if ts == ts and ts else None,
            "G_seeded_vs_tc": rd(tt / ts) if ts == ts and ts else None,
            "seed_gbps": rd(gbps / (ts * 1e-3)) if ts == ts and ts else None,
            "seed_cfg": cfg_brief(seed_cfg),
            "acc": {"default": d_acc, "seeded": s_acc, "tc": t_acc},
            "noisy_sub25us": (min([x for x in (td, ts, tt) if x == x] or [9]) * 1e3 < 25.0),
        }
        rows.append(row)
        print("ROW " + json.dumps(row), file=sys.stderr)
        del args, ref_out, d_call, s_call, tc
        torch.cuda.empty_cache()
    return rows


def dt_name(dt):
    return {torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32"}[dt]


def rd(x):
    return None if (x != x) else round(float(x), 4)


def cfg_brief(c):
    if c is None:
        return None
    d = dict(c.config)
    return {"block_sizes": d.get("block_sizes"), "num_warps": d.get("num_warps"),
            "num_stages": d.get("num_stages"), "pid": d.get("pid_type")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="swiglu,geglu")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--dtypes", default="bf16")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    reg = _kernels()
    kernels = a.kernels.split(",")
    splits = a.splits.split(",")
    dtypes = [_DTYPES[x] for x in a.dtypes.split(",")]
    print(f"helion: {helion.__file__}")
    all_rows = []
    summary = []
    for kname in kernels:
        kfn, ref, build_args = reg[kname]
        traffic = SH.TRAFFIC[kname]
        for dt in dtypes:
            shapes = []
            for sp in splits:
                shapes += SH.SHAPES[kname][sp]
            rows = bench_kernel(kname, kfn, ref, build_args, shapes, dt, traffic)
            all_rows += rows
            ok = [r for r in rows if r["acc"]["seeded"] is True]
            svd = [r["seeded_vs_default"] for r in ok if r["seeded_vs_default"]]
            g = [r["G_seeded_vs_tc"] for r in ok if r["G_seeded_vs_tc"]]
            below = [(r["shape"], r["G_seeded_vs_tc"]) for r in ok
                     if r["G_seeded_vs_tc"] and r["G_seeded_vs_tc"] < FLOOR
                     and not r["noisy_sub25us"]]
            cell = {
                "kernel": kname, "dtype": dt_name(dt),
                "n_acc_pass": len(ok), "n_total": len(rows),
                "geomean_seeded_vs_default": rd(statistics.geometric_mean(svd)) if svd else None,
                "geomean_G_vs_tc": rd(statistics.geometric_mean(g)) if g else None,
                "min_G": rd(min(g)) if g else None,
                "below_floor_realistic": below,
            }
            summary.append(cell)
            print("CELL " + json.dumps(cell))
    out = {"rows": all_rows, "summary": summary}
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2)
    print("\n===== SUMMARY =====")
    for c in summary:
        print(f"  {c['kernel']:14} {c['dtype']:5} "
              f"vs_default={c['geomean_seeded_vs_default']}x  G_vs_tc={c['geomean_G_vs_tc']}  "
              f"min_G={c['min_G']}  acc={c['n_acc_pass']}/{c['n_total']}  "
              f"below_floor={c['below_floor_realistic']}")


if __name__ == "__main__":
    main()
