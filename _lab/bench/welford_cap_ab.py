"""Matched-lever A/B for the welford combine/normalize cap divergence at bf16.

The bf16 seed diverges from fp32 only because the caps divide by itemsize:
  combine cap = STRUCTURED_COMBINE_CAP_BYTES // itemsize  -> bf16 2x wider (16384 vs 8192)
  normalize   = loop_chunk_bytes // itemsize               -> bf16 2x wider (4096 vs 2048)
The welford accumulators are fp32; resident CHUNK bytes are held constant, but the
reduction-tree ELEMENT width doubles at bf16. This A/B tests whether forcing the
fp32-equivalent (narrower, element-matched) tiles at bf16 input beats the current bf16 seed.

Arms per shape: current-bf16-seed, fp32-equiv-tiles (forced), tc-default. Single-process,
N=15 median, dynamo-reset, accuracy each (vs torch eager bf16). Prints G vs tc for each arm.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

import torch
from triton.testing import do_bench

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

_WT = "/home/dev/local/helion-pr-with-lab"
sys.path.insert(0, os.path.join(_WT, "_lab", "prompts"))
from examples.welford import welford, eager_layer_norm  # noqa: E402

N_RUNS = 15


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2] * 1000.0


def acc(out, ref, tol):
    return bool(torch.allclose(out.float(), ref.float(), rtol=tol, atol=tol)), \
        float((out.float() - ref.float()).abs().max())


# (M, N): the divergent welford shapes + the bf16-seed blocks + fp32-equiv blocks (from the
# divergence table). 'cur' = current bf16 seed; 'f32eq' = the tile widths the fp32 path emits.
SHAPES = [
    (16384, 4096, [1, 4096, 4096], [1, 4096, 2048]),
    (16384, 5120, [1, 8192, 8192], [1, 8192, 2048]),
    (8192, 7168, [1, 8192, 4096], [1, 8192, 2048]),
    (8192, 8192, [1, 8192, 4096], [1, 8192, 2048]),
    (8192, 12288, [1, 16384, 4096], [1, 8192, 2048]),
    (8192, 14336, [1, 16384, 4096], [1, 8192, 2048]),
    (8192, 16384, [1, 16384, 4096], [1, 8192, 2048]),
]
TOL = 4e-2


def build(m, n):
    w = torch.rand(n, device="cuda", dtype=torch.bfloat16)
    b = torch.rand(n, device="cuda", dtype=torch.bfloat16)
    x = torch.rand(m, n, device="cuda", dtype=torch.bfloat16)
    return (w, b, x, 1e-5), eager_layer_norm(w, b, x, 1e-5)


def cfg_from_seed_with_blocks(args, blocks):
    """Take the live seed config but override block_sizes (so all other levers — warps,
    eviction, stages — match the seed; only the cap-driven tile widths change)."""
    b0 = welford.bind(args)
    seed = compiler_seed_configs(b0.env, b0.host_function.device_ir)[0]
    d = dict(seed)
    d["block_sizes"] = list(blocks)
    return helion.Config(**d)


def main():
    assert os.path.realpath(helion.__file__).startswith(_WT), helion.__file__
    rows = []
    for (m, n, cur_blocks, f32_blocks) in SHAPES:
        torch._dynamo.reset()
        args, ref = build(m, n)
        # current bf16 seed (verify its blocks match cur_blocks)
        b0 = welford.bind(args)
        seed = compiler_seed_configs(b0.env, b0.host_function.device_ir)[0]
        seed_blocks = dict(seed).get("block_sizes")
        k_cur = helion.kernel(welford.fn, config=seed, static_shapes=True)
        out_cur = k_cur(*args); a_cur, e_cur = acc(out_cur, ref, TOL)
        # fp32-equivalent tiles
        cfg_f = cfg_from_seed_with_blocks(args, f32_blocks)
        k_f = helion.kernel(welford.fn, config=cfg_f, static_shapes=True)
        out_f = k_f(*args); a_f, e_f = acc(out_f, ref, TOL)
        # tc
        tcfn = torch.compile(lambda: eager_layer_norm(*args)); tcfn()
        t_cur = med(lambda: k_cur(*args))
        t_f = med(lambda: k_f(*args))
        t_tc = med(tcfn)
        row = {"shape": [m, n], "seed_blocks": seed_blocks, "f32_blocks": f32_blocks,
               "cur_us": round(t_cur, 2), "f32eq_us": round(t_f, 2), "tc_us": round(t_tc, 2),
               "G_cur": round(t_tc / t_cur, 4), "G_f32eq": round(t_tc / t_f, 4),
               "f32eq_vs_cur": round(t_cur / t_f, 4),
               "acc_cur": a_cur, "err_cur": round(e_cur, 5), "acc_f32eq": a_f, "err_f32eq": round(e_f, 5)}
        rows.append(row)
        print("ROW " + json.dumps(row), file=sys.stderr)
    print(json.dumps(rows, indent=2))
    json.dump(rows, open("/tmp/welford_cap_ab.json", "w"), indent=2)
    # summary
    speedups = [r["f32eq_vs_cur"] for r in rows]
    print(f"\nf32eq_vs_cur (>1 = fp32-equiv tiles faster): median={st.median(speedups):.4f} "
          f"min={min(speedups):.4f} max={max(speedups):.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
