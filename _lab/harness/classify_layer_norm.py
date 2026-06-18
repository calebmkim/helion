"""Static classification + heuristic-fires probe for layer_norm_fwd (no GPU timing).

Bind layer_norm_fwd at representative in-sample shapes, inspect config_spec
(T1 rollable rdim? matmul? single block? single reduction_loop?), print the
ReductionFact (expect num_reduction_ops=2, num_load>=2), and confirm
compiler_seed_configs emits exactly one reduction seed (or report why not).

layer_norm-fwd does TWO reductions over N (mean = sum(x); var = sum(centered^2)).
Both reduce over the SAME N axis -> should share ONE rdim -> 1 reduction_loop ->
the eligibility gate (len(reduction_loops)==1) should pass.

Run with the canonical invocation (see SETUP.md).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.layer_norm import layer_norm_fwd  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5


def build_args(m, n, with_bias=True):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32) if with_bias else None
    return (x, [n], w, b, EPS)


def probe(name, args):
    print(f"=== {name} ===")
    try:
        bound = layer_norm_fwd.bind(args)
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
    # in-sample fwd shapes
    for (m, n) in [(4096, 1024), (4096, 4096), (4096, 8192), (4096, 12288),
                   (4096, 15872), (2048, 3584), (8192, 7168)]:
        probe(f"layer_norm_fwd ({m},{n}) with bias", build_args(m, n, with_bias=True))
    # also check without bias (tritonbench runs both)
    probe("layer_norm_fwd (4096,4096) NO bias", build_args(4096, 4096, with_bias=False))


if __name__ == "__main__":
    main()
