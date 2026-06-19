"""Compile-only probe: confirm the per_feature_accumulator discriminator + the
non-grid inner tile, for the 6 backward M-reduction kernels. NO GPU timing."""
from __future__ import annotations

import json
import os
import sys

import torch

import helion

_WT = "/home/calebkim/helion-new-heuristics/helion-3stage"
assert os.path.realpath(helion.__file__).startswith(_WT + os.sep), helion.__file__
sys.path.insert(0, _WT)
sys.path.insert(0, os.path.join(_WT, "_lab", "curriculum_candidates"))

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
import mreduction_styles as MS  # noqa: E402
from examples.layer_norm import layer_norm_bwd  # noqa: E402
from examples.rms_norm import rms_norm_bwd  # noqa: E402

EPS = 1e-5


def build_rms(dt):
    m, n = 8192, 4096
    x = torch.randn(m, n, device="cuda", dtype=dt)
    w = torch.randn(n, device="cuda", dtype=dt)
    go = torch.randn(m, n, device="cuda", dtype=dt)
    inv = torch.rsqrt(x.float().pow(2).mean(-1) + EPS).reshape(m, 1)
    return rms_norm_bwd, (go, x, w, inv), "(8192,4096)"


def build_ln(dt):
    m, n = 8192, 4096
    x = torch.randn(m, n, device="cuda", dtype=dt)
    w = torch.randn(n, device="cuda", dtype=dt)
    go = torch.randn(m, n, device="cuda", dtype=dt)
    mean = x.float().mean(-1)
    rstd = torch.rsqrt(x.float().var(-1, unbiased=False) + EPS)
    return layer_norm_bwd, (go, x, mean, rstd, w), "(8192,4096)"


def build_bias(dt):
    m, n = 16384, 1024
    return MS.bias_grad_bwd, (torch.randn(m, n, device="cuda", dtype=dt),), "(16384,1024)"


def build_dyt(dt):
    m, n = 16384, 1024
    go = torch.randn(m, n, device="cuda", dtype=dt)
    x = torch.randn(m, n, device="cuda", dtype=dt)
    w = torch.randn(n, device="cuda", dtype=dt)
    return MS.dyt_bwd, (go, x, w, 0.7), "(16384,1024)"


def build_group(dt):
    nn, c, s, g = 512, 128, 64, 32
    x = torch.randn(nn, c, s, device="cuda", dtype=dt)
    go = torch.randn(nn, c, s, device="cuda", dtype=dt)
    mean = torch.randn(nn, g, device="cuda", dtype=torch.float32)
    rstd = torch.rand(nn, g, device="cuda", dtype=torch.float32) + 0.5
    w = torch.randn(c, device="cuda", dtype=dt)
    return MS.group_norm_bwd, (go, x, mean, rstd, w, g), "(512,128,64,32)F=8192"


def build_instance(dt):
    bb, c, s = 512, 64, 128
    x = torch.randn(bb, c, s, device="cuda", dtype=dt)
    go = torch.randn(bb, c, s, device="cuda", dtype=dt)
    mean = torch.randn(bb, c, device="cuda", dtype=torch.float32)
    rstd = torch.rand(bb, c, device="cuda", dtype=torch.float32) + 0.5
    w = torch.randn(c, device="cuda", dtype=dt)
    return MS.instance_norm_bwd, (go, x, mean, rstd, w), "(512,64,128)F=8192"


BUILDERS = {
    "rms_norm_bwd": build_rms,
    "layer_norm_bwd": build_ln,
    "instance_norm": build_instance,
    "group_norm": build_group,
    "bias_grad": build_bias,
    "dyt": build_dyt,
}


def analyze(name, dt):
    fn, args, desc = BUILDERS[name](dt)
    bound = fn.bind(args)
    env = bound.env
    spec = env.config_spec
    bs_ids = set(spec.block_sizes.valid_block_ids())
    rl_ids = set(spec.reduction_loops.valid_block_ids())
    grid_ids = {b for bids in bound.host_function.device_ir.grid_block_ids for b in bids}
    rf = spec.reduction_facts[0] if spec.reduction_facts else None
    seeds = compiler_seed_configs(env, bound.host_function.device_ir)
    seed = "no seed"
    if seeds:
        d = dict(seeds[0])
        spec.normalize(d)
        seed = {k: d[k] for k in ("block_sizes",) if k in d}
        seed["full"] = json.dumps(dict(seeds[0]))
    out = {"kernel": name, "shape": desc, "seed_block_sizes": seed}
    if rf is not None:
        # non-grid inner tile = block_sizes not grid, not the rdim, not a normalize loop
        inner = sorted(
            b for b in bs_ids
            if b not in grid_ids
            and b != rf.block_id
            and b not in set(rf.non_reduction_loop_block_ids)
        )
        out["reduction_fact"] = {
            "block_id": rf.block_id,
            "size_hint": rf.size_hint,
            "full_width_output": rf.full_width_output,
            "m_block_ids(grid)": list(rf.m_block_ids),
            "non_grid_inner_tile_ids": inner,
            "feature_extent": rf.feature_extent,
            "PER_FEATURE_ACCUMULATOR": rf.per_feature_accumulator,
            "num_carried_2d_tiles": rf.num_carried_2d_tiles,
            "body_live_tiles": rf.body_live_tiles,
            "num_load": rf.num_load,
        }
    return out


def main():
    print(f"helion={helion.__file__}", flush=True)
    for name in BUILDERS:
        torch._dynamo.reset()
        print("ROW " + json.dumps(analyze(name, torch.float32)), flush=True)


if __name__ == "__main__":
    main()
