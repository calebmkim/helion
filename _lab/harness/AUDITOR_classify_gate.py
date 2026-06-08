"""AUDITOR: for every in-scope kernel + welford, bind and report whether the
seed gate fires (reduction_facts==1 and is_eligible). welford MUST decline;
all 7 + cross_entropy MUST fire. Also prints the ReductionFact + the seed the
heuristic would emit (so I can see what fires and what gets capped to looped).
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    TritonReductionHeuristic,
    _triton_reduction_eligible,
)

LONG = torch.int64
EPS = 1e-5


def classify(name, kfn, args):
    try:
        b = kfn.bind(args)
    except Exception as e:  # noqa: BLE001
        print(f"{name:<22} BIND-ERROR: {type(e).__name__}: {e}")
        return
    env = b.env
    dev_ir = b.host_function.device_ir
    spec = env.config_spec
    nf = len(spec.reduction_facts)
    # eligibility + seed
    try:
        elig = _triton_reduction_eligible(env, dev_ir)
    except Exception as e:  # noqa: BLE001
        elig = f"ERR:{e}"
    info = f"reduction_facts={nf} matmul_facts={len(spec.matmul_facts)} eligible={elig}"
    if nf == 1:
        f = spec.reduction_facts[0]
        info += (f"\n      FACT: rnumel={f.size_hint} itemsize={f.itemsize} "
                 f"num_load={f.num_load} num_tiled_accum={f.num_tiled_accumulators} "
                 f"row_bytes={f.size_hint*max(1,f.itemsize)}")
        try:
            with env:
                cfg = TritonReductionHeuristic.get_seed_config(env, dev_ir)
            info += f"\n      SEED: {dict(cfg) if cfg else None}"
        except Exception as e:  # noqa: BLE001
            info += f"\n      SEED-ERR: {type(e).__name__}: {e}"
    fires = (nf == 1 and elig is True)
    print(f"{name:<22} FIRES={fires} | {info}")


def main():
    print(f"helion={helion.__file__}\n")
    from examples.rms_norm import rms_norm_fwd
    from examples.layer_norm import layer_norm_fwd
    from examples.softmax import softmax
    from examples.sum import sum_kernel
    from examples.long_sum import longsum as long_sum
    from examples.cross_entropy import cross_entropy
    from examples.welford import welford

    def rms_args(m, n):
        return (torch.randn(m, n, device="cuda"), torch.randn(n, device="cuda"), EPS)

    def ln_args(m, n):
        return (torch.randn(m, n, device="cuda"), [n], torch.randn(n, device="cuda"),
                torch.randn(n, device="cuda"), EPS)

    def sm_args(m, n):
        return (torch.randn(m, n, device="cuda"),)

    def ce_args(n, v):
        return (torch.randn(n, v, device="cuda"),
                torch.randint(0, v, (n,), device="cuda", dtype=LONG))

    def wf_args(m, n):
        return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
                torch.rand(m, n, device="cuda"), EPS)

    print("=== MUST FIRE (in-scope) ===")
    classify("rms_norm (2048,4096)", rms_norm_fwd, rms_args(2048, 4096))
    classify("layer_norm (2048,4096)", layer_norm_fwd, ln_args(2048, 4096))
    classify("softmax (2048,4096)", softmax, sm_args(2048, 4096))
    classify("sum (2048,16384)", sum_kernel, sm_args(2048, 16384))
    classify("long_sum (8,131072)", long_sum, sm_args(8, 131072))
    classify("cross_entropy (2048,65536)", cross_entropy, ce_args(2048, 65536))
    classify("cross_entropy (2048,131072)", cross_entropy, ce_args(2048, 131072))
    print("\n=== MUST DECLINE (out-of-scope) ===")
    classify("welford (4096,1024)", welford, wf_args(4096, 1024))
    classify("welford (4096,2048)", welford, wf_args(4096, 2048))


if __name__ == "__main__":
    main()
