"""BROADEN #1: does num_warps>4 help at the seed's tiles? If w4 is best/tied -> DEFER (dead-knob).
Vary num_warps at the seed block_sizes on large shapes. cool GPU, med-of-9, vs tc."""
from __future__ import annotations
import sys
import torch
import helion

WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
assert helion.__file__.startswith(WT)
sys.path.insert(0, f"{WT}/_lab/pointwise")
from examples.swiglu import _swiglu_fwd  # noqa: E402
from examples.add import add  # noqa: E402
import ptw_kernels as PK  # noqa: E402

DEV = "cuda"; N = 9; DT = torch.bfloat16


def med(fn):
    from triton.testing import do_bench
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N))[N // 2]


def run(name, kfn, bs, shapes, mkargs):
    print(f"--- {name} bs={bs} ---")
    for (m, n) in shapes:
        torch._dynamo.reset()
        a = mkargs(m, n)
        out = []
        for w in (4, 8, 16):
            k = helion.kernel(kfn.fn, config=helion.Config(block_sizes=bs, num_warps=w), static_shapes=True)
            out.append((w, med(lambda: k(*a))))
        base = [t for w, t in out if w == 4][0]
        print(f"  ({m},{n}): " + "  ".join(f"w{w}={t*1e3:.1f}us(x{base/t:.3f})" for w, t in out))
        del a; torch.cuda.empty_cache()


run("swiglu", _swiglu_fwd, [1024], [(16384, 11008), (8192, 14336), (4096, 28672)],
    lambda m, n: (torch.randn(m, n, device=DEV, dtype=DT), torch.randn(m, n, device=DEV, dtype=DT)))
run("residual_add", add, [1, 1024], [(16384, 5120), (8192, 8192)],
    lambda m, n: (torch.randn(m, n, device=DEV, dtype=DT), torch.randn(m, n, device=DEV, dtype=DT)))
run("bias_gelu", PK.bias_gelu, [1, 2048], [(8192, 18176), (16384, 6400)],
    lambda m, n: (torch.randn(m, n, device=DEV, dtype=DT), torch.randn(n, device=DEV, dtype=DT)))
