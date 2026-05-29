"""GENERALITY STRESS-TEST: static classification + v7-fires probe for NEW
forward reduction kernels NOT in the 9-kernel curriculum.

Candidates:
  - softmax            (examples/softmax.py): SIMPLE whole-row T1 softmax
                        (torch.nn.functional.softmax over a hl.tile(n) row).
                        DIFFERENT code path from softmax_two_pass (T2) we tuned.
  - softmax_decomposed (examples/softmax.py): explicit amax/exp/sum T1 softmax.
  - jagged_mean        (examples/jagged_mean.py): jagged reduction (expect declined
                        by the v5 dynamic-size guard -> 0 seeds, falls back to default).
  - jagged_softmax     (examples/jagged_softmax.py): jagged 2-pass (expect declined).

Reuses the classify_ce_welford probe shape. Run with the canonical invocation.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.softmax import softmax  # noqa: E402
from examples.softmax import softmax_decomposed  # noqa: E402
from examples.jagged_mean import jagged_mean_kernel  # noqa: E402
from examples.jagged_softmax import jagged_softmax_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion._compiler.inductor_lowering import ReductionLowering  # noqa: E402


def _dump_red_lowerings(device_ir, grid_block_ids):
    grid_ids = {b for bids in grid_block_ids for b in bids}
    red = {}
    for gi in device_ir.graphs:
        for node in gi.graph.nodes:
            low = node.meta.get("lowering")
            if isinstance(low, ReductionLowering):
                bid = getattr(low, "block_index", None)
                red.setdefault(bid, 0)
                red[bid] += 1
    inner = [b for b in red if b not in grid_ids]
    print(f"  ReductionLowering block_indices (count): {red}")
    print(f"  grid_block_ids: {grid_block_ids} -> grid_ids={grid_ids}")
    print(f"  inner (non-grid) reduction axes: {inner}")


def probe(name, kern, args):
    print(f"=== {name} ===")
    try:
        bound = kern.bind(args)
    except Exception as e:  # noqa: BLE001
        print(f"  bind FAILED: {type(e).__name__}: {e}\n")
        return
    spec = bound.env.config_spec
    dev = bound.host_function.device_ir
    print(f"  block_sizes spec entries: {len(spec.block_sizes)}")
    print(f"  reduction_loops entries:  {len(spec.reduction_loops)}")
    print(f"  matmul_facts:             {len(spec.matmul_facts)}")
    rf = getattr(spec, "reduction_facts", [])
    print(f"  reduction_facts:          {len(rf)}")
    for f in rf:
        print(f"    ReductionFact: block_id={f.block_id} size_hint={f.size_hint} "
              f"itemsize={f.itemsize} dtype={f.dtype} num_load={f.num_load} "
              f"num_store={f.num_store} num_reduction_ops={f.num_reduction_ops} "
              f"num_tiled_accumulators={f.num_tiled_accumulators} "
              f"static_rnumel={f.static_rnumel} m_block_ids={f.m_block_ids} "
              f"is_structured_combine={getattr(f, 'is_structured_combine', '?')}")
    try:
        _dump_red_lowerings(dev, dev.grid_block_ids)
    except Exception as e:  # noqa: BLE001
        print(f"  red-lowering dump FAILED: {type(e).__name__}: {e}")
    for i in range(len(spec.block_sizes)):
        bs = spec.block_sizes[i]
        print(f"  block_sizes[{i}]: block_id={bs.block_id} min_size={bs.min_size} "
              f"autotuner_min={bs.autotuner_min} size_hint={bs.size_hint}")
    try:
        seeds = compiler_seed_configs(bound.env, dev)
        names = list(spec.autotuner_heuristics)
        print(f"  compiler_seed_configs -> {len(seeds)} seed(s); heuristics_used={names}")
        for s in seeds:
            sd = dict(s)
            print(f"    seed: block_sizes={sd.get('block_sizes')} "
                  f"reduction_loops={sd.get('reduction_loops')} "
                  f"num_warps={sd.get('num_warps')} num_stages={sd.get('num_stages')}")
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"  compiler_seed_configs FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
    print()


def sm_args(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    return (x,)


def jagged_args(num_rows, max_cols, M):
    lengths = torch.randint(1, max_cols + 1, (num_rows,), device="cuda")
    x_offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long, device="cuda"), torch.cumsum(lengths, dim=0)]
    )
    nnz = int(x_offsets[-1])
    x_data = torch.randn(nnz, M, dtype=torch.float32, device="cuda")
    return x_data, x_offsets


def main():
    print(f"helion={helion.__file__}\n")
    print("##### SOFTMAX (simple, T1 whole-row) #####")
    for (m, n) in [(4096, 256), (4096, 1024), (4096, 4096), (4096, 16384), (32768, 256)]:
        probe(f"softmax ({m},{n})", softmax, sm_args(m, n))
    print("##### SOFTMAX_DECOMPOSED (T1 amax/exp/sum) #####")
    for (m, n) in [(4096, 1024), (4096, 4096), (4096, 16384)]:
        probe(f"softmax_decomposed ({m},{n})", softmax_decomposed, sm_args(m, n))
    print("##### JAGGED_MEAN (jagged reduction; expect declined) #####")
    xd, xo = jagged_args(512, 64, 8)
    fc = torch.full((512,), 8, dtype=torch.int32, device="cuda")
    probe("jagged_mean", jagged_mean_kernel, (xd, xo, fc, 8))
    print("##### JAGGED_SOFTMAX (jagged 2-pass; expect declined) #####")
    xd2, xo2 = jagged_args(512, 64, 128)
    probe("jagged_softmax", jagged_softmax_kernel, (xd2, xo2))


if __name__ == "__main__":
    main()
