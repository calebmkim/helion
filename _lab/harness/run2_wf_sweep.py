"""Goal-1 welford re-derivation sweep (CORRECTED kernel, no divisor constraint).

For one (M,N): measure tc_default, the live v8 seed, a fresh quick-autotune oracle,
and a (combine x apply x warps) grid at DEFAULT codegen knobs (correctness-gated).
Prints G_seed / G_oracle / best-grid config, so we can design the new Band-C and
check whether the v8 oracle was confounded by the (now-fixed) divisor accuracy gate.

Usage: ... python run2_wf_sweep.py M N
"""
from __future__ import annotations
import json
import math
import os
import sys
from statistics import median

import torch
import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2"
assert helion.__file__.startswith(WT + "/"), helion.__file__
from examples.welford import welford, eager_layer_norm
from helion._compiler.autotuner_heuristics import compiler_seed_configs
from helion._utils import next_power_of_2 as np2
from triton.testing import do_bench

EPS = 1e-5
N_RUNS = 7


def args_for(m, n):
    w = torch.rand(n, device="cuda", dtype=torch.float32)
    b = torch.rand(n, device="cuda", dtype=torch.float32)
    x = torch.rand(m, n, device="cuda", dtype=torch.float32)
    return (w, b, x, EPS)


def med(fn):
    torch.cuda.synchronize()
    return median([float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)])


def roles(bound):
    spec = bound.env.config_spec
    fact = spec.reduction_facts[0]
    ci = spec.block_sizes.block_id_to_index(fact.block_id)
    ai = [spec.block_sizes.block_id_to_index(b) for b in fact.apply_block_ids]
    return ci, ai, len(spec.block_sizes), fact


def run_cfg(a, cfg_dict, ref):
    k = helion.kernel(welford.fn, configs=[helion.Config(**cfg_dict)])
    bk = k.bind(a)
    bk.ensure_config_exists(a)
    out = bk(*a)
    out = out[0] if isinstance(out, tuple) else out
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-4))
    if not ok:
        return None, dict(bk._config)
    return med(lambda: bk(*a)), dict(bk._config)


def main():
    m, n = int(sys.argv[1]), int(sys.argv[2])
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    a = args_for(m, n)
    w, b, x, eps = a
    ref = eager_layer_norm(w, b, x, eps)

    # tc_default
    torch._dynamo.reset()
    tc = torch.compile(eager_layer_norm)
    _ = tc(w, b, x, eps)
    tc_lat = med(lambda: tc(w, b, x, eps))

    # live v8 seed
    bound = welford.bind(a)
    ci, ai, nbs, fact = roles(bound)
    seed = dict(compiler_seed_configs(bound.env, bound.host_function.device_ir)[0])
    seed_lat, seed_norm = run_cfg(a, seed, ref)

    # grid sweep (combine x apply x warps) at default codegen knobs
    np2n = np2(n)
    combines = sorted({c for c in (256, 512, 1024, 2048, 4096, 8192, np2n) if c <= np2n})
    applies = sorted({c for c in (512, 1024, 2048, 4096, np2n) if c <= np2n})
    warpset = (8, 16, 32)
    best = None
    grid = []
    base = dict(seed)
    for cw in warpset:
        for cb in combines:
            for ap in applies:
                bs = list(base["block_sizes"])
                bs[ci] = cb
                for j in ai:
                    bs[j] = ap
                cfg = {**base, "block_sizes": bs, "num_warps": cw}
                lat, norm = run_cfg(a, cfg, ref)
                if lat is None:
                    continue
                rec = {"combine": cb, "apply": ap, "warps": cw, "lat_us": lat * 1e3,
                       "G": tc_lat / lat}
                grid.append(rec)
                if best is None or lat < best["lat_us"] / 1e3:
                    best = rec

    # fresh quick-autotune oracle (corrected kernel) — canonical pattern
    try:
        os.environ["HELION_AUTOTUNE_EFFORT"] = "quick"
        ko = helion.kernel(welford.fn)
        bo = ko.bind(a)
        bo.autotune(a)
        oracle_cfg = dict(bo._config)
        # re-bench the FULL VERBATIM winning config in a fresh kernel
        olat, onorm = run_cfg(a, oracle_cfg, ref)
    except Exception as e:  # noqa: BLE001
        oracle_cfg = f"ERR {type(e).__name__}: {e}"
        olat = None

    out = {
        "shape": [m, n], "gpu": gpu, "np2n": np2n,
        "tc_default_us": tc_lat * 1e3,
        "v8_seed": {"cfg": seed_norm, "lat_us": (seed_lat * 1e3) if seed_lat else None,
                    "G": (tc_lat / seed_lat) if seed_lat else None},
        "best_grid": best,
        "oracle": {"cfg": oracle_cfg, "lat_us": (olat * 1e3) if olat else None,
                   "G": (tc_lat / olat) if olat else None},
        "fact": {"num_load": fact.num_load, "num_store": fact.num_store,
                 "num_reduction_ops": fact.num_reduction_ops,
                 "num_tiled_accumulators": fact.num_tiled_accumulators,
                 "is_structured_combine": fact.is_structured_combine,
                 "static_rnumel": fact.static_rnumel},
        "grid": sorted(grid, key=lambda r: r["lat_us"]),
    }
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
