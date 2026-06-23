"""Decide TILE_BYTES: does 8192 (t3->1024, t2->2048) beat 16384 (t3->2048, t2->4096)?
Confirm swiglu[1024]/relu_squared[2048] stay parity; bias_gelu[1,2048]>[1,4096]; residual_add neutral."""
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

DEV = "cuda"; N_RUNS = 9; DT = torch.bfloat16


def med(fn):
    from triton.testing import do_bench
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def run(tag, kfn, ref, shapes, cfgs, mkargs):
    print(f"--- {tag} ---")
    for (m, n) in shapes:
        torch._dynamo.reset()
        args = mkargs(m, n)
        r = ref(*args)
        tc = torch.compile(ref, mode="max-autotune-no-cudagraphs"); tc(*args)
        tt = med(lambda: tc(*args))
        out = []
        for cfg in cfgs:
            k = helion.kernel(kfn.fn, config=helion.Config(block_sizes=cfg), static_shapes=True)
            if (k(*args).float() - r.float()).abs().max().item() > 0.05:
                out.append(f"{cfg}:ACCFAIL"); continue
            out.append(f"{cfg}:G={tt/med(lambda: k(*args)):.3f}")
        print(f"  ({m},{n}) tc={tt*1e3:.0f}us  " + "  ".join(out))
        del args, r, tc; torch.cuda.empty_cache()


def ref_sw(a, b):
    return torch.nn.functional.silu(a.float()).to(b.dtype) * b


run("swiglu 1024 vs 2048", _swiglu_fwd, ref_sw,
    [(16384, 11008), (4096, 4096), (2048, 29568), (8192, 14336)], [[1024], [2048]],
    lambda m, n: (torch.randn(m, n, device=DEV, dtype=DT), torch.randn(m, n, device=DEV, dtype=DT)))
run("relu_squared 2048 vs 4096", PK.relu_squared, PK.ref_relu_squared,
    [(32768, 3072), (4096, 16384), (8192, 11008)], [[2048], [4096]],
    lambda m, n: (torch.randn(m, n, device=DEV, dtype=DT),))
run("residual_add [1,1024] vs [1,2048]", add, lambda x, y: x + y,
    [(16384, 5120), (8192, 8192), (16384, 2048)], [[1, 1024], [1, 2048]],
    lambda m, n: (torch.randn(m, n, device=DEV, dtype=DT), torch.randn(m, n, device=DEV, dtype=DT)))
run("bias_gelu [1,2048] vs [1,4096]", PK.bias_gelu, PK.ref_bias_gelu,
    [(16384, 5120), (8192, 10240), (8192, 8192), (16384, 4096)], [[1, 2048], [1, 4096]],
    lambda m, n: (torch.randn(m, n, device=DEV, dtype=DT), torch.randn(n, device=DEV, dtype=DT)))
