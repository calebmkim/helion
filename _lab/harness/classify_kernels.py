"""Static classification + heuristic-fires probe for sum + long_sum (no GPU timing).

For each kernel variant, bind at a representative in-sample shape, inspect the
config_spec (T1 rollable rdim? matmul? single block?), print the ReductionFact,
and confirm compiler_seed_configs emits exactly one reduction seed (or report why
not). Also classify each long_sum variant (naive / w_red_loop / manual).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.long_sum import longsum  # noqa: E402
from examples.long_sum import longsum_manual  # noqa: E402
from examples.long_sum import longsum_w_red_loop  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402


def x_args(m, n):
    return (torch.randn(m, n, device="cuda", dtype=torch.float32),)


def probe(name, fn, args):
    print(f"=== {name}  shape={tuple(args[0].shape)} ===")
    try:
        bound = fn.bind(args)
    except Exception as e:  # noqa: BLE001
        print(f"  bind FAILED: {type(e).__name__}: {e}\n")
        return
    spec = bound.env.config_spec
    print(f"  block_sizes spec entries: {len(spec.block_sizes)}")
    print(f"  reduction_loops entries:  {len(spec.reduction_loops)}")
    print(f"  matmul_facts:             {len(spec.matmul_facts)}")
    rf = getattr(spec, "reduction_facts", [])
    print(f"  reduction_facts:          {len(rf)}")
    for f in rf:
        print(f"    ReductionFact: size_hint={f.size_hint} itemsize={f.itemsize} "
              f"dtype={f.dtype} num_load={f.num_load} num_store={f.num_store} "
              f"num_reduction_ops={f.num_reduction_ops} static_rnumel={f.static_rnumel} "
              f"m_block_ids={f.m_block_ids}")
    # M-block floor
    if spec.block_sizes:
        bs0 = spec.block_sizes[0]
        print(f"  M-block: min_size={bs0.min_size} autotuner_min={bs0.autotuner_min} "
              f"size_hint={bs0.size_hint}")
    try:
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        names = list(spec.autotuner_heuristics)
        print(f"  compiler_seed_configs -> {len(seeds)} seed(s); heuristics_used={names}")
        for s in seeds:
            sd = dict(s)
            print(f"    seed: block_sizes={sd.get('block_sizes')} "
                  f"reduction_loops={sd.get('reduction_loops')} "
                  f"num_warps={sd.get('num_warps')} num_stages={sd.get('num_stages')}")
    except Exception as e:  # noqa: BLE001
        print(f"  compiler_seed_configs FAILED: {type(e).__name__}: {e}")
    print()


def main():
    print(f"helion={helion.__file__}\n")
    # sum: representative in-sample shapes (compiler-managed row sum)
    probe("sum_kernel (2048,16384)", sum_kernel, x_args(2048, 16384))
    probe("sum_kernel (8192,256)", sum_kernel, x_args(8192, 256))
    probe("sum_kernel (32768,1024)", sum_kernel, x_args(32768, 1024))
    # long_sum variants: tiny M, huge rnumel
    probe("longsum [naive] (8,131072)", longsum, x_args(8, 131072))
    probe("longsum [naive] (1,32768)", longsum, x_args(1, 32768))
    probe("longsum_w_red_loop (8,131072)", longsum_w_red_loop, x_args(8, 131072))
    probe("longsum_manual (8,131072)", longsum_manual, x_args(8, 131072))


if __name__ == "__main__":
    main()
