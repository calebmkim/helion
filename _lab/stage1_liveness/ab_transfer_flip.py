"""A/B the transfer-kernel cells Part B flips: persistent [None] (base) vs the Part-B looped
chunk, vs tc-default. Decides whether the flip regresses (Gate R, transfer invariant).
Single-process median-of-9 do_bench; wide-V logits are multi-GB (>>50MB L2) so cold enough.
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
sys.path.insert(0, os.path.join(_WT, "_lab", "transfer"))
import ab_three_arm_transfer as T  # noqa: E402

N = 9


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N))[N // 2] * 1e3


def ab(kn, shape, dtn, rls):
    build = T._make(kn)
    dt = T._DTYPES[dtn]
    torch._dynamo.reset()
    kfn, args, ref, chk = build(shape, dt)
    bound = kfn.bind(args)
    seed = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]
    base = dict(seed.config)
    ref_out = ref()
    tcf = torch.compile(ref)
    tcf()
    t_tc = med(tcf)
    print(f"\n{kn} {dtn} {shape}  seed_rl={base.get('reduction_loops')}  tc={t_tc:.1f}us")
    for rl in rls:
        cfg = dict(base)
        cfg["reduction_loops"] = rl
        k = helion.kernel(kfn.fn, config=helion.Config(**cfg), static_shapes=True)
        out = k(*args)
        try:
            chk(out, ref_out)
            acc = True
        except Exception:
            acc = False
        t = med(lambda: k(*args))
        print(f"    rl={str(rl):9} acc={acc} lat={t:8.1f}us  G_vs_tc={t_tc/t:.3f}")
    del args, bound
    torch.cuda.empty_cache()


def main():
    for shape in [(8192, 50257), (4096, 50304), (8192, 49152)]:
        for dtn in ("bf16", "fp32"):
            ab("cross_entropy_ls_zloss", shape, dtn, [[None], [16384]])


if __name__ == "__main__":
    main()
