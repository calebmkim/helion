"""Stage-2 factdump + ReductionLowering-axis classifier for the 6 backward M-reduction kernels.

For each kernel at a representative train shape, report:
  - n_reduction_facts, classification (standard T1 / user-tiled T2 / none),
  - n_matmul_facts, emitted seed config from compiler_seed_configs (or "no seed"),
  - if a ReductionFact exists: block_id, size_hint, full_width_output,
    num_carried_2d_tiles, body_live_tiles, non_reduction_loop_block_ids,
  - the ReductionLowering axes (block_index) classified as:
      in block_sizes (user-tiled) / in reduction_loops (rolled T1) / MATERIALIZED (neither).

Compile-time only (bind + factdump). NO GPU timing. CUDA must be present.

Run: cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
  PYTHONPATH=<worktree> python <this> --dtype fp32
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

_WT = "/home/calebkim/helion-new-heuristics/helion-3stage"
assert os.path.realpath(helion.__file__).startswith(_WT + os.sep), helion.__file__
sys.path.insert(0, _WT)
sys.path.insert(0, os.path.join(_WT, "_lab", "curriculum_candidates"))
sys.path.insert(0, os.path.join(_WT, "_lab", "bench"))

from helion._compiler.compile_environment import CompileEnvironment  # noqa: E402
from helion._compiler.inductor_lowering import ReductionLowering  # noqa: E402

import mreduction_styles as MS  # noqa: E402

from examples.layer_norm import layer_norm_bwd  # noqa: E402
from examples.rms_norm import rms_norm_bwd  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
EPS = 1e-5


# ---- builders: representative TRAIN shapes (from mreduction_shapes / curriculum) ----
def build_rms_norm(dt):
    m, n = 8192, 4096
    x = torch.randn(m, n, device="cuda", dtype=dt)
    weight = torch.randn(n, device="cuda", dtype=dt)
    grad_out = torch.randn(m, n, device="cuda", dtype=dt)
    inv_rms = torch.rsqrt(x.float().pow(2).mean(-1) + EPS).reshape(m, 1)
    return rms_norm_bwd, (grad_out, x, weight, inv_rms), f"(M={m},N={n})"


def build_layer_norm(dt):
    m, n = 8192, 4096
    x = torch.randn(m, n, device="cuda", dtype=dt)
    weight = torch.randn(n, device="cuda", dtype=dt)
    grad_out = torch.randn(m, n, device="cuda", dtype=dt)
    mean = x.float().mean(-1)
    rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + EPS)
    return layer_norm_bwd, (grad_out, x, mean, rstd, weight), f"(M={m},N={n})"


def build_bias_grad(dt):
    m, n = 16384, 1024
    go = torch.randn(m, n, device="cuda", dtype=dt)
    return MS.bias_grad_bwd, (go,), f"(M={m},N={n})"


def build_dyt(dt):
    m, n = 16384, 1024
    go = torch.randn(m, n, device="cuda", dtype=dt)
    x = torch.randn(m, n, device="cuda", dtype=dt)
    w = torch.randn(n, device="cuda", dtype=dt)
    return MS.dyt_bwd, (go, x, w, 0.7), f"(M={m},N={n})"


def build_group_norm(dt):
    nn, c, s, g = 512, 128, 64, 32
    x = torch.randn(nn, c, s, device="cuda", dtype=dt)
    go = torch.randn(nn, c, s, device="cuda", dtype=dt)
    mean = torch.randn(nn, g, device="cuda", dtype=torch.float32)
    rstd = torch.rand(nn, g, device="cuda", dtype=torch.float32) + 0.5
    w = torch.randn(c, device="cuda", dtype=dt)
    return MS.group_norm_bwd, (go, x, mean, rstd, w, g), f"(N={nn},C={c},S={s},G={g}) F=C*S={c*s}"


def build_instance_norm(dt):
    bb, c, s = 512, 64, 128
    x = torch.randn(bb, c, s, device="cuda", dtype=dt)
    go = torch.randn(bb, c, s, device="cuda", dtype=dt)
    mean = torch.randn(bb, c, device="cuda", dtype=torch.float32)
    rstd = torch.rand(bb, c, device="cuda", dtype=torch.float32) + 0.5
    w = torch.randn(c, device="cuda", dtype=dt)
    return MS.instance_norm_bwd, (go, x, mean, rstd, w), f"(B={bb},C={c},S={s}) F=C*S={c*s}"


BUILDERS = {
    "rms_norm_bwd": build_rms_norm,
    "layer_norm_bwd": build_layer_norm,
    "bias_grad_bwd": build_bias_grad,
    "dyt_bwd": build_dyt,
    "group_norm_bwd": build_group_norm,
    "instance_norm_bwd": build_instance_norm,
}


def analyze(name, dt_name):
    dt = DTYPES[dt_name]
    fn, args, desc = BUILDERS[name](dt)
    bound = fn.bind(args)
    env = bound.env
    device_ir = bound.host_function.device_ir
    spec = env.config_spec

    bs_ids = set(spec.block_sizes.valid_block_ids())
    rl_ids = set(spec.reduction_loops.valid_block_ids())
    grid_ids = {b for bids in device_ir.grid_block_ids for b in bids}

    rfacts = list(getattr(spec, "reduction_facts", []))
    mfacts = list(getattr(spec, "matmul_facts", []))
    afacts = list(getattr(spec, "accumulator_facts", []))

    # classification
    if not rfacts:
        track = "none"
    else:
        f0 = rfacts[0]
        track = "standard-T1" if f0.block_id in rl_ids else "user-tiled-T2"

    # ReductionLowering axis census across all device graphs
    red_axes = []  # (block_id, count)
    axis_count: dict[int, int] = {}
    with env, bound.host_function:
        for gi in device_ir.graphs:
            for node in gi.graph.nodes:
                low = node.meta.get("lowering")
                if isinstance(low, ReductionLowering):
                    bid = getattr(low, "block_index", None)
                    if bid is not None:
                        axis_count[bid] = axis_count.get(bid, 0) + 1
    axes_cls = {"in_block_sizes": [], "in_reduction_loops": [], "MATERIALIZED": []}
    for bid, cnt in sorted(axis_count.items()):
        if bid in rl_ids:
            axes_cls["in_reduction_loops"].append((bid, cnt))
        elif bid in bs_ids:
            axes_cls["in_block_sizes"].append((bid, cnt))
        else:
            axes_cls["MATERIALIZED"].append((bid, cnt))

    # materialized-inner reductions (exclude grid axes): the key Stage-2 signal
    materialized_nongrid = [
        (bid, cnt) for (bid, cnt) in axes_cls["MATERIALIZED"] if bid not in grid_ids
    ]

    seeds = compiler_seed_configs(env, device_ir)
    seed_str = "no seed"
    if seeds:
        norm = dict(seeds[0])
        spec.normalize(norm)
        seed_str = json.dumps(norm)
    default = dict(spec.default_config())

    # block structure
    block_struct = []
    for bs in env.block_sizes:
        try:
            sz = env.size_hint(bs.size)
        except Exception:
            sz = "?"
        block_struct.append(
            {
                "blk": bs.block_id,
                "reduction": bool(bs.reduction),
                "size": sz,
                "in_bs": bs.block_id in bs_ids,
                "in_rl": bs.block_id in rl_ids,
                "grid": bs.block_id in grid_ids,
            }
        )

    out = {
        "kernel": name,
        "shape": desc,
        "dtype": dt_name,
        "n_reduction_facts": len(rfacts),
        "track": track,
        "n_matmul_facts": len(mfacts),
        "n_accumulator_facts": len(afacts),
        "grid_block_ids": sorted(grid_ids),
        "seed": seed_str,
        "default_config": default,
        "block_struct": block_struct,
        "red_axes_classified": {
            k: [list(t) for t in v] for k, v in axes_cls.items()
        },
        "materialized_inner_count": len(materialized_nongrid),
        "materialized_inner_axes": [list(t) for t in materialized_nongrid],
    }
    if rfacts:
        f0 = rfacts[0]
        out["reduction_fact"] = {
            "block_id": f0.block_id,
            "size_hint": f0.size_hint,
            "full_width_output": f0.full_width_output,
            "num_carried_2d_tiles": f0.num_carried_2d_tiles,
            "body_live_tiles": f0.body_live_tiles,
            "non_reduction_loop_block_ids": list(f0.non_reduction_loop_block_ids),
            "m_block_ids": list(f0.m_block_ids),
            "itemsize": f0.itemsize,
            "num_load": f0.num_load,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("kernels", nargs="*", default=list(BUILDERS))
    a = ap.parse_args()
    print(f"helion={helion.__file__}", flush=True)
    results = []
    for name in (a.kernels or list(BUILDERS)):
        torch._dynamo.reset()
        r = analyze(name, a.dtype)
        results.append(r)
        print("ROW " + json.dumps(r), flush=True)
    json.dump(results, open(f"/tmp/stage2_probe_{a.dtype}.json", "w"), indent=2)
    print(f"\nWROTE /tmp/stage2_probe_{a.dtype}.json", flush=True)


if __name__ == "__main__":
    main()
