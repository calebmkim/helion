"""Classify softmax_two_pass / kl_div / jsd: probe block_sizes, reduction_loops,
reduction_facts, matmul_facts, grid_block_ids, and FIND the T2 reduction block_id
via the ReductionLowering predicate (filtered against grid_block_ids).

PRE-IMPLEMENTATION probe (no source change yet) to ground the t2_code_map.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from helion._compiler.inductor_lowering import ReductionLowering  # noqa: E402

from examples.softmax import softmax_two_pass  # noqa: E402
from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward  # noqa: E402

DEV = "cuda"


def probe(name, fn, args):
    print(f"\n===== {name} =====")
    bound = fn.bind(args)
    spec = bound.env.config_spec
    device_ir = bound.host_function.device_ir
    # reduction flag lives on env.block_sizes (BlockSizeInfo), keyed by block_id
    red_flag = {bi.block_id: bi.reduction for bi in bound.env.block_sizes}
    print("  len(block_sizes):", len(spec.block_sizes))
    for i, bs in enumerate(spec.block_sizes):
        print(
            f"    block_sizes[{i}]: block_id={bs.block_id} "
            f"size_hint={bs.size_hint} reduction={red_flag.get(bs.block_id)} "
            f"min_size={bs.min_size} autotuner_min={bs.autotuner_min} "
            f"src={type(bs).__name__}"
        )
    print("  len(reduction_loops):", len(spec.reduction_loops))
    for rl in spec.reduction_loops:
        print(f"    reduction_loop: block_id={rl.block_id} size_hint={rl.size_hint}")
    print("  len(reduction_facts):", len(spec.reduction_facts))
    print("  len(matmul_facts):", len(spec.matmul_facts))
    print("  grid_block_ids:", device_ir.grid_block_ids)

    # FIND the T2 reduction axis via ReductionLowering predicate.
    red_block_ids = set()
    for gi in device_ir.graphs:
        for node in gi.graph.nodes:
            low = node.meta.get("lowering")
            if isinstance(low, ReductionLowering):
                red_block_ids.add(low.block_index)
    grid_ids = {b for bids in device_ir.grid_block_ids for b in bids}
    t2_red = [b for b in red_block_ids if b not in grid_ids]
    print("  ReductionLowering block_indices:", sorted(red_block_ids))
    print("  grid_block_ids (flat):", sorted(grid_ids))
    print("  -> T2 reduction block_ids (filtered):", sorted(t2_red))
    for b in sorted(t2_red):
        try:
            idx = spec.block_sizes.block_id_to_index(b)
            bs = spec.block_sizes.block_id_lookup(b)
            print(f"     block_id={b} -> spec index={idx} size_hint={bs.size_hint}")
        except KeyError:
            print(f"     block_id={b} NOT in block_sizes spec")


def main():
    print(f"helion={helion.__file__}\n")
    torch.manual_seed(0)
    M, N = 4096, 2560

    x = torch.randn(M, N, device=DEV, dtype=torch.float32)
    probe("softmax_two_pass", softmax_two_pass, (x,))

    BT, V = 4096, 65536
    y_pred = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(dim=-1)
    y_true = torch.randn(BT, V, device=DEV, dtype=torch.float32).softmax(dim=-1)
    probe("kl_div_forward", kl_div_forward, (y_pred, y_true, False, "batchmean", 1e-10))

    log_q = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(dim=-1)
    log_p = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(dim=-1)
    probe("jsd_forward", jsd_forward, (log_q, log_p, None, 0.5, -100))


if __name__ == "__main__":
    main()
