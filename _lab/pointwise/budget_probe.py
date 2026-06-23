"""Lock the tile budget: does 1-D swiglu (fp32-promote, traffic-3) stay at parity as the inner
tile grows to 8192? And confirm N-D residual_add wants the bigger inner. cold-L2 med-of-9."""
from __future__ import annotations
import sys
import torch
import helion

WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
assert helion.__file__.startswith(WT)
from examples.swiglu import _swiglu_fwd  # noqa: E402
from examples.add import add  # noqa: E402

DEV = "cuda"; N_RUNS = 9; DT = torch.bfloat16


def med(fn):
    from triton.testing import do_bench
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2]


def ref_swiglu(a, b):
    return torch.nn.functional.silu(a.float()).to(b.dtype) * b


print("===== swiglu 1-D (fp32 promote, traffic-3): inner bs sweep =====")
for (m, n) in [(16384, 11008), (4096, 4096), (8192, 14336), (4096, 28672), (2048, 29568)]:
    torch._dynamo.reset()
    a = torch.randn(m, n, device=DEV, dtype=DT); b = torch.randn(m, n, device=DEV, dtype=DT)
    tc = torch.compile(ref_swiglu, mode="max-autotune-no-cudagraphs"); tc(a, b)
    tt = med(lambda: tc(a, b))
    out = []
    for bs in [2048, 4096, 8192, 16384]:
        k = helion.kernel(_swiglu_fwd.fn, config=helion.Config(block_sizes=[bs]), static_shapes=True)
        t = med(lambda: k(a, b))
        out.append(f"bs{bs}:G={tt/t:.3f}")
    print(f"  ({m},{n}) tc={tt*1e3:.1f}us  " + "  ".join(out))
    del a, b, tc; torch.cuda.empty_cache()

print("===== residual_add N-D (bf16, traffic-3): [1,inner] sweep =====")
for (m, n) in [(16384, 5120), (8192, 8192), (4096, 8192), (16384, 2048)]:
    torch._dynamo.reset()
    a = torch.randn(m, n, device=DEV, dtype=DT); b = torch.randn(m, n, device=DEV, dtype=DT)
    tc = torch.compile(lambda x, y: x + y, mode="max-autotune-no-cudagraphs"); tc(a, b)
    tt = med(lambda: tc(a, b))
    out = []
    for inner in [2048, 4096, 8192]:
        k = helion.kernel(add.fn, config=helion.Config(block_sizes=[1, inner]), static_shapes=True)
        t = med(lambda: k(a, b))
        out.append(f"[1,{inner}]:G={tt/t:.3f}")
    print(f"  ({m},{n}) tc={tt*1e3:.1f}us  " + "  ".join(out))
    del a, b, tc; torch.cuda.empty_cache()
