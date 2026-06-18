"""Stage-1 liveness probe: for each reduction kernel, count the rdim-shaped INTERMEDIATE
tensor nodes in its device graphs (the candidate "live full-width tiles" signal for Part B).

Faithful property under test: # of tensor-valued nodes whose meta['val'].shape resolves to
include the reduction block_id (i.e. spans the full reduction width). This is the conservative
over-count of "peak simultaneously-live full-width intermediates" that drives the register/SMEM
spill (fused_linear_jsd holds ~8; rms_norm ~1-2). Compile-time only, NO GPU compute.

Run: cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
  PYTHONPATH=<worktree> python <this>
"""

from __future__ import annotations

import os
import sys

import torch

import helion  # noqa: E402

_WT = "/home/calebkim/helion-new-heuristics/helion-3stage"
assert os.path.realpath(helion.__file__).startswith(_WT), helion.__file__
sys.path.insert(0, os.path.join(_WT, "_lab", "bench"))
sys.path.insert(0, os.path.join(_WT, "_lab", "prompts"))
sys.path.insert(0, os.path.join(_WT, "_lab", "transfer"))

from helion._compiler.compile_environment import CompileEnvironment  # noqa: E402
from helion._compiler.inductor_lowering import ReductionLowering  # noqa: E402

import bare_fwd_dtype as BF  # noqa: E402


def _build_flj(m, v, dt):
    from examples.fused_linear_jsd import jsd_kernel

    sl = torch.randn(m, v, device="cuda", dtype=dt)
    tl = torch.randn(m, v, device="cuda", dtype=dt)
    return jsd_kernel, (0.5, -100, 1.0, sl, tl)


def analyze(name, fn, args, dt_name):
    bound = fn.bind(args)
    env = bound.env
    device_ir = bound.host_function.device_ir
    spec = env.config_spec
    rfacts = spec.reduction_facts
    if not rfacts:
        print(f"{name:20} {dt_name:4}: NO reduction_facts (matmul={bool(spec.matmul_facts)})")
        return
    fact = rfacts[0]
    rdim = fact.block_id
    host = device_ir.host_function

    def is_rdim_shaped(node):
        val = node.meta.get("val")
        if not isinstance(val, torch.Tensor):
            return False
        try:
            return rdim in {env.resolve_block_id(s) for s in val.shape}
        except Exception:
            return False

    # Per-graph PEAK simultaneously-live rdim-shaped values (proper liveness sweep over the
    # topologically-ordered FX nodes): a value is live from its def to its last use within the
    # graph; count live rdim-shaped values at each step, take the peak.
    with env, host:
        per_graph = []
        max_peak = 0
        sum_peak = 0
        for gi in device_ir.graphs:
            g = gi.graph
            nodes = list(g.nodes)
            # last-use index of each node within this graph
            last_use = {}
            for i, node in enumerate(nodes):
                for inp in node.all_input_nodes:
                    last_use[inp] = i
            live = set()
            peak = 0
            n_red = 0
            for i, node in enumerate(nodes):
                if isinstance(node.meta.get("lowering"), ReductionLowering):
                    n_red += 1
                if node.op == "call_function" and is_rdim_shaped(node):
                    live.add(node)
                # retire values whose last use was this node
                cur = sum(1 for v in live if is_rdim_shaped(v))
                peak = max(peak, cur)
                dead = {v for v in live if last_use.get(v, -1) <= i}
                live -= dead
            per_graph.append((type(gi).__name__, gi.graph_id, peak, n_red, len(nodes)))
            max_peak = max(max_peak, peak)
            sum_peak += peak
    print(
        f"{name:20} {dt_name:4}: rdim_id={rdim} sh={fact.size_hint} fw={fact.full_width_output} "
        f"itemsz={fact.itemsize} ntiles2d={fact.num_carried_2d_tiles} nload={fact.num_load} "
        f"=> MAX_PEAK_LIVE={max_peak}  SUM_PEAK_LIVE={sum_peak}"
    )
    for tn, gid, pk, nred, ntot in per_graph:
        if pk or nred:
            print(f"        graph[{gid}] {tn:28} peak_live={pk} red_lowerings={nred} (of {ntot})")


def main():
    dt = torch.bfloat16
    # fused_linear_jsd narrow-V (the target) + wide-V
    for (m, v) in [(4096, 32000), (4096, 50257), (2048, 128256)]:
        fn, args = _build_flj(m, v, dt)
        analyze(f"flj({m},{v})", fn, args, "bf16")
    # the 9 standard/user-tiled curriculum kernels at a representative train shape
    reps = {
        "rms_norm": (8192, 4096), "layer_norm": (8192, 4096), "softmax": (8192, 4096),
        "sum": (8192, 8192), "long_sum": (1024, 131072), "cross_entropy": (8192, 50257),
        "welford": (8192, 4096), "kl_div": (8192, 32000), "jsd": (8192, 32000),
    }
    for kn, (m, n) in reps.items():
        fn, build, _ = BF.KERNELS[kn]
        args, _ref, _ex = build(m, n, dt)
        analyze(f"{kn}({m},{n})", fn, args, "bf16")


if __name__ == "__main__":
    main()
