"""AUDITOR independent probe: print full ReductionFact (incl num_tiled_accumulators)
and the emitted seed for every relevant kernel. Verifies the Band-B gate keys on a
REAL IR property (num_tiled_accumulators) not kernel identity.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402


def f32(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.float32)


def logsm(m, n):
    return torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(dim=-1)


def sm(m, n):
    return torch.randn(m, n, device="cuda", dtype=torch.float32).softmax(dim=-1)


def probe(name, fn, args):
    print(f"=== {name}  shapes={[tuple(a.shape) if hasattr(a,'shape') else a for a in args]} ===")
    try:
        bound = fn.bind(tuple(args))
    except Exception as e:  # noqa: BLE001
        print(f"  bind FAILED: {type(e).__name__}: {e}\n")
        return
    spec = bound.env.config_spec
    print(f"  block_sizes={len(spec.block_sizes)} reduction_loops={len(spec.reduction_loops)} "
          f"matmul_facts={len(spec.matmul_facts)} reduction_facts={len(spec.reduction_facts)}")
    for f in spec.reduction_facts:
        print(f"    FACT block_id={f.block_id} size_hint={f.size_hint} itemsize={f.itemsize} "
              f"dtype={f.dtype}")
        print(f"         num_load={f.num_load} num_store={f.num_store} "
              f"num_reduction_ops={f.num_reduction_ops} "
              f"num_tiled_accumulators={f.num_tiled_accumulators}  <-- BAND-B KEY")
    try:
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        print(f"  heuristics_used={list(spec.autotuner_heuristics)}; {len(seeds)} seed(s)")
        for s in seeds:
            sd = dict(s)
            print(f"    SEED block_sizes={sd.get('block_sizes')} "
                  f"reduction_loops={sd.get('reduction_loops')} "
                  f"num_warps={sd.get('num_warps')} num_stages={sd.get('num_stages')}")
    except Exception as e:  # noqa: BLE001
        print(f"  compiler_seed_configs FAILED: {type(e).__name__}: {e}")
    print()


def main():
    print(f"helion={helion.__file__}\n")
    from examples.kl_div import kl_div_forward
    from examples.jsd import jsd_forward
    from examples.softmax import softmax_two_pass
    from examples.sum import sum_kernel
    from examples.rms_norm import rms_norm_fwd
    from examples.long_sum import longsum
    from examples.layer_norm import layer_norm_fwd

    # T2 Band-B candidates
    probe("kl_div (1024,65536)", kl_div_forward, (logsm(1024, 65536), sm(1024, 65536)))
    probe("kl_div (4096,32000)", kl_div_forward, (logsm(4096, 32000), sm(4096, 32000)))
    probe("jsd (1024,65536)", jsd_forward, (logsm(1024, 65536), logsm(1024, 65536)))
    probe("jsd (4096,32000)", jsd_forward, (logsm(4096, 32000), logsm(4096, 32000)))

    # T2 Band-A (should be num_tiled_accumulators==0)
    probe("softmax_two_pass (2048,2560)", softmax_two_pass, (f32(2048, 2560),))
    probe("softmax_two_pass (8192,32768)", softmax_two_pass, (f32(8192, 32768),))

    # T1 (should be num_tiled_accumulators==0)
    probe("sum (2048,16384)", sum_kernel, (f32(2048, 16384),))
    probe("long_sum (8,131072)", longsum, (f32(8, 131072),))
    probe("rms_norm (4096,8192)", rms_norm_fwd, (f32(4096, 8192), f32(8192)))
    probe("layer_norm (4096,8192)", layer_norm_fwd,
          (f32(4096, 8192), [8192], f32(8192), f32(8192)))


if __name__ == "__main__":
    main()
