"""Static classification + heuristic-fires probe for cross_entropy + welford.

Bind each at representative in-sample shapes; inspect config_spec (T1 rollable
rdim? T2 user-tiled? matmul? how many block_sizes / reduction_loops /
reduction_facts), print the ReductionFact (if any), and report whether
compiler_seed_configs emits a reduction seed (and the seed).

Run with the canonical invocation (see SETUP.md).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.welford import welford  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402
from helion._compiler.inductor_lowering import ReductionLowering  # noqa: E402

LONG = torch.int64


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
              f"static_rnumel={f.static_rnumel} m_block_ids={f.m_block_ids}")
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


def ce_args(n, v):
    logits = torch.randn(n, v, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, v, (n,), device="cuda", dtype=LONG)
    return (logits, labels)


def wf_args(m, n):
    w = torch.rand(n, device="cuda", dtype=torch.float32)
    b = torch.rand(n, device="cuda", dtype=torch.float32)
    x = torch.rand(m, n, device="cuda", dtype=torch.float32)
    return (w, b, x, 1e-5)


def main():
    print(f"helion={helion.__file__}\n")
    print("##### CROSS_ENTROPY #####")
    for (n, v) in [(4096, 4096), (4096, 16384), (8192, 32768), (8192, 131072)]:
        probe(f"cross_entropy ({n},{v})", cross_entropy, ce_args(n, v))
    print("##### WELFORD #####")
    for (m, n) in [(262144, 1024), (262144, 2048), (262144, 4096)]:
        probe(f"welford ({m},{n})", welford, wf_args(m, n))


if __name__ == "__main__":
    main()
