"""PRIMARY target A/B: fused_linear_jsd (standard track) persistent [None] vs looped chunks,
arm-fair (footgun #6c): the tc baseline computes BOTH loss AND grad (the kernel returns both).
Single-process median-of-9 do_bench; working set multi-GB (>>50MB L2) so do_bench is cold.
Tells us (a) flj narrow-V flips persistent->looped and beats fair-tc, (b) the best looped chunk.
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
from examples.fused_linear_jsd import jsd_kernel  # noqa: E402

N = 9


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N))[N // 2] * 1e3


def fair_tc(beta, temp, sl, tl):
    # Computes loss [M] AND grad [M,V] fp32 — matches jsd_kernel's two outputs (arm-fair).
    def f():
        ss, ts = sl.float() / temp, tl.float() / temp
        sp = torch.softmax(ss, -1)
        tp = torch.softmax(ts, -1)
        slp = torch.log_softmax(ss, -1)
        tlp = torch.log_softmax(ts, -1)
        m = (1 - beta) * sp + beta * tp
        logm = torch.log(m)
        skl = (sp * (slp - logm)).sum(-1)
        tkl = (tp * (tlp - logm)).sum(-1)
        loss = (1 - beta) * skl + beta * tkl
        grad = ((1 - beta) / temp) * (sp - m)
        return loss, grad
    return f


def ab(m, v, dt, rls):
    beta, temp = 0.5, 1.0
    sl = torch.randn(m, v, device="cuda", dtype=dt)
    tl = torch.randn(m, v, device="cuda", dtype=dt)
    args = (beta, -100, temp, sl, tl)
    bound = jsd_kernel.bind(args)
    seed = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]
    base = dict(seed.config)
    ref_loss, ref_grad = fair_tc(beta, temp, sl, tl)()
    torch._dynamo.reset()
    tcf = torch.compile(fair_tc(beta, temp, sl, tl))
    tcf()
    t_tc = med(tcf)
    print(f"\nflj {dt} ({m},{v}) seed_rl={base.get('reduction_loops')} fair_tc(loss+grad)={t_tc:.1f}us")
    for rl in rls:
        cfg = dict(base)
        cfg["reduction_loops"] = rl
        k = helion.kernel(jsd_kernel.fn, config=helion.Config(**cfg), static_shapes=True)
        loss, grad = k(*args)
        acc = bool(torch.allclose(loss.float(), ref_loss.float(), rtol=3e-2, atol=3e-2))
        t = med(lambda: k(*args))
        print(f"    rl={str(rl):9} acc={acc} lat={t:8.1f}us  G_vs_fairtc={t_tc/t:.3f}")
    del sl, tl
    torch.cuda.empty_cache()


def main():
    dtn = {"fp32": torch.float32, "bf16": torch.bfloat16}
    for dt in ("bf16", "fp32"):
        ab(4096, 32000, dtn[dt], [[None], [16384], [8192], [4096], [2048]])
        ab(4096, 50257, dtn[dt], [[None], [16384], [8192], [4096], [2048]])
    ab(8192, 32000, dtn["bf16"], [[None], [16384], [8192], [4096]])
    # wide-V (already looped at [16384]) — confirm no change needed
    ab(2048, 128256, dtn["bf16"], [[None], [16384], [8192]])


if __name__ == "__main__":
    main()
