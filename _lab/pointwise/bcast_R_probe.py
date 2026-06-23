"""Broadcast amortization: a bias[N] broadcast over rows costs full M*N traffic at R=1 rows.
Find the robust outer-row tile R for bias_gelu. Metric = speedup vs default[32,32] and vs [1,2048]
(helion-vs-helion, tc-invariant => stable despite tc's unstable bias_gelu compile). med-of-15."""
from __future__ import annotations
import sys
import torch
import helion

WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
assert helion.__file__.startswith(WT)
sys.path.insert(0, f"{WT}/_lab/pointwise")
import ptw_kernels as PK  # noqa: E402

DEV = "cuda"; N = 15; DT = torch.bfloat16


def med(fn):
    from triton.testing import do_bench
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N))[N // 2]


# budget ~2048 elems (traffic-2 @ TILE_BYTES=8192): vary the outer R, inner = 2048//R
CANDS = [[1, 2048], [4, 512], [8, 256], [16, 128], [32, 64], [32, 32], [32, 128], [16, 256], [64, 64]]

for (m, n) in [(16384, 5120), (8192, 8192), (8192, 10240), (4096, 32768), (16384, 4096), (8192, 18176)]:
    torch._dynamo.reset()
    x = torch.randn(m, n, device=DEV, dtype=DT)
    bias = torch.randn(n, device=DEV, dtype=DT)
    ref = PK.ref_bias_gelu(x, bias)
    times = {}
    for cfg in CANDS:
        k = helion.kernel(PK.bias_gelu.fn, config=helion.Config(block_sizes=cfg), static_shapes=True)
        if (k(x, bias).float() - ref.float()).abs().max().item() > 0.05:
            continue
        times[tuple(cfg)] = med(lambda: k(x, bias))
    base = times.get((32, 32))
    thin = times.get((1, 2048))
    ranked = sorted(times.items(), key=lambda kv: kv[1])
    gbps = 2 * m * n * DT.itemsize / 1e12
    print(f"({m},{n})  best->worst (us, TB/s, vs[32,32], vs[1,2048]):")
    for cfg, t in ranked[:5]:
        print(f"   {str(list(cfg)):10} {t*1e3:7.1f}us {gbps/(t*1e-3):4.2f}TB/s  "
              f"x[32,32]={base/t:.2f}  x[1,2048]={thin/t:.2f}")
    del x, bias, ref; torch.cuda.empty_cache()
