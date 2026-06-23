"""Verify the spill fix: corpus configs byte-identical + tall-skinny tensors get a budget-sized
[R,N] tile (not a starved [1,N]) + the tall-skinny seed now beats the default and clears the floor."""
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
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

DEV = "cuda"; N = 9; DT = torch.bfloat16


def seed_cfg(kfn, args):
    bound = kfn.bind(args)
    s = compiler_seed_configs(bound.env, bound.host_function.device_ir)
    return dict(s[0].config).get("block_sizes") if s else None


def med(fn):
    from triton.testing import do_bench
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N))[N // 2]


print("=== corpus (must be byte-identical) ===")
print("  swiglu(4096,4096):", seed_cfg(_swiglu_fwd, (torch.randn(4096, 4096, device=DEV, dtype=DT), torch.randn(4096, 4096, device=DEV, dtype=DT))))
print("  residual_add(16384,5120):", seed_cfg(add, (torch.randn(16384, 5120, device=DEV, dtype=DT), torch.randn(16384, 5120, device=DEV, dtype=DT))))
print("  residual_add(32768,768):", seed_cfg(add, (torch.randn(32768, 768, device=DEV, dtype=DT), torch.randn(32768, 768, device=DEV, dtype=DT))))

print("=== tall-skinny (spill to outer; was [1,N]) ===")
for (m, n) in [(1048576, 8), (524288, 16), (262144, 64), (131072, 256), (1048576, 4)]:
    a = torch.randn(m, n, device=DEV, dtype=DT); b = torch.randn(m, n, device=DEV, dtype=DT)
    print(f"  residual_add({m},{n}): seed={seed_cfg(add, (a, b))}")
    del a, b; torch.cuda.empty_cache()

print("=== perf: tall-skinny seed vs default vs tc (residual_add) ===")
for (m, n) in [(1048576, 8), (262144, 64), (131072, 256)]:
    torch._dynamo.reset()
    a = torch.randn(m, n, device=DEV, dtype=DT); b = torch.randn(m, n, device=DEV, dtype=DT)
    bound = add.bind((a, b))
    dcfg = bound.config_spec.default_config()
    scfg = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]
    kd = helion.kernel(add.fn, config=dcfg, static_shapes=True)
    ks = helion.kernel(add.fn, config=scfg, static_shapes=True)
    tc = torch.compile(lambda x, y: x + y, mode="max-autotune-no-cudagraphs"); tc(a, b)
    td, ts, tt = med(lambda: kd(a, b)), med(lambda: ks(a, b)), med(lambda: tc(a, b))
    print(f"  ({m},{n}): default={td*1e3:.1f}us seed={ts*1e3:.1f}us({dict(scfg.config)['block_sizes']}) tc={tt*1e3:.1f}us "
          f"| seed_vs_default={td/ts:.2f}x G_vs_tc={tt/ts:.3f}")
    del a, b, tc; torch.cuda.empty_cache()
