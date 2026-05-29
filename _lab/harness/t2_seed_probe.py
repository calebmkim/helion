"""T2 seed-emission probe: for each T2 kernel print the ReductionFact and the
seed emitted by compiler_seed_configs. Confirm: heuristic FIRES (1 seed), seed
routes T2 (block_sizes has R_BLOCK at the reduction index, NO reduction_loops),
R_BLOCK = next_pow2(N) (persistent), num_warps per the rnumel ramp.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

from examples.softmax import softmax_two_pass  # noqa: E402
from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward  # noqa: E402

DEV = "cuda"


def probe(name, fn, args):
    print(f"\n===== {name} =====")
    bound = fn.bind(args)
    spec = bound.env.config_spec
    rf = spec.reduction_facts
    print(f"  reduction_facts={len(rf)} matmul_facts={len(spec.matmul_facts)} "
          f"reduction_loops={len(spec.reduction_loops)} block_sizes={len(spec.block_sizes)}")
    for f in rf:
        print(f"  RF: block_id={f.block_id} size_hint={f.size_hint} "
              f"m_block_ids={f.m_block_ids} dtype={f.dtype} itemsize={f.itemsize} "
              f"num_load={f.num_load} num_store={f.num_store} "
              f"num_reduction_ops={f.num_reduction_ops} static_rnumel={f.static_rnumel}")
    seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    print(f"  compiler_seed_configs -> {len(seeds)} seed(s); "
          f"heuristics_used={list(spec.autotuner_heuristics)}")
    for s in seeds:
        sd = dict(s)
        print(f"    seed: block_sizes={sd.get('block_sizes')} "
              f"reduction_loops={sd.get('reduction_loops')} "
              f"num_warps={sd.get('num_warps')} num_stages={sd.get('num_stages')}")


def main():
    print(f"helion={helion.__file__}")
    torch.manual_seed(0)
    # softmax: vary N to confirm R_BLOCK = next_pow2(N) and warps ramp
    for (M, N) in [(4096, 256), (4096, 2560), (4096, 8192), (4096, 16384), (32768, 1024)]:
        x = torch.randn(M, N, device=DEV, dtype=torch.float32)
        probe(f"softmax_two_pass ({M},{N})", softmax_two_pass, (x,))

    for (BT, V) in [(4096, 4096), (4096, 65536), (4096, 131072)]:
        y_pred = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(dim=-1)
        y_true = torch.randn(BT, V, device=DEV, dtype=torch.float32).softmax(dim=-1)
        probe(f"kl_div ({BT},{V})", kl_div_forward,
              (y_pred, y_true, False, "batchmean", 1e-10))

    for (BT, V) in [(8192, 4096), (8192, 65536), (8192, 131072)]:
        log_q = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(dim=-1)
        log_p = torch.randn(BT, V, device=DEV, dtype=torch.float32).log_softmax(dim=-1)
        probe(f"jsd ({BT},{V})", jsd_forward, (log_q, log_p, None, 0.5, -100))


if __name__ == "__main__":
    main()
