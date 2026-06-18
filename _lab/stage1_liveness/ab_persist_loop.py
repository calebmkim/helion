"""A/B: persistent [None] vs looped reduction_loops for the flip-risk cells, single-process
median-of-9 do_bench (working sets >> 50MB L2 here so do_bench is cold enough; sanity-check BW).
Decides whether ff=peak_live flipping cross_entropy/long_sum wide-V to looped REGRESSES or helps.
"""
from __future__ import annotations

import os
import sys

import torch
from triton.testing import do_bench

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

_WT = "/home/calebkim/helion-new-heuristics/helion-3stage"
assert os.path.realpath(helion.__file__).startswith(_WT), helion.__file__
sys.path.insert(0, os.path.join(_WT, "_lab", "bench"))
import bare_fwd_dtype as BF  # noqa: E402

N = 9


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N))[N // 2] * 1e3


def ab(kn, m, n, dt, rls):
    fn, build, _ = BF.KERNELS[kn]
    args, ref, ex = build(m, n, dt)
    bound = fn.bind(args)
    seed = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]
    base = dict(seed.config)
    # tc baseline (default inductor)
    torch._dynamo.reset()
    tcfn = torch.compile(lambda: BF.KERNELS[kn][2](args))
    tcfn()
    t_tc = med(tcfn)
    print(f"\n{kn} {dt} ({m},{n})  seed_rl={base.get('reduction_loops')}  tc={t_tc:.1f}us")
    for rl in rls:
        cfg = dict(base)
        cfg["reduction_loops"] = rl
        k = helion.kernel(fn.fn, config=helion.Config(**cfg), static_shapes=True)
        out = ex(k(*args))
        acc = bool(torch.allclose(out.float(), ref.float(), rtol=2e-2, atol=2e-2))
        t = med(lambda: k(*args))
        print(f"    rl={str(rl):9} acc={acc} lat={t:8.1f}us  G_vs_tc={t_tc/t:.3f}")


def main():
    dtn = {"fp32": torch.float32, "bf16": torch.bfloat16}
    # cross_entropy fp32 wide-V: the 19-flip risk
    ab("cross_entropy", 8192, 50257, dtn["fp32"], [[None], [16384], [8192]])
    ab("cross_entropy", 4096, 50304, dtn["fp32"], [[None], [16384]])
    ab("cross_entropy", 8192, 32000, dtn["fp32"], [[None], [16384]])
    # long_sum bf16 wide
    ab("long_sum", 256, 65536, dtn["bf16"], [[None], [16384]])


if __name__ == "__main__":
    main()
