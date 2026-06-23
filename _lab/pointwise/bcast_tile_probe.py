"""Does a broadcast input (bias[N]) want a MULTI-ROW tile to amortize the bias reload?
Probe bias_gelu (N-D broadcast) vs residual_add (N-D, no broadcast) tile shapes vs tc/default."""
from __future__ import annotations
import sys
import torch
import helion

WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
assert helion.__file__.startswith(WT)
sys.path.insert(0, f"{WT}/_lab/pointwise")
import ptw_kernels as PK  # noqa: E402

DEV = "cuda"; N_RUNS = 9; DT = torch.bfloat16


def med(fn):
    from triton.testing import do_bench
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2]


CANDS = [[1, 4096], [1, 2048], [2, 2048], [4, 1024], [8, 512], [8, 1024],
         [16, 256], [16, 512], [32, 128], [32, 256], [32, 32], [64, 64]]

print("===== bias_gelu (N-D, BROADCAST bias[N]) =====")
for (m, n) in [(16384, 5120), (8192, 10240), (8192, 8192), (16384, 4096)]:
    torch._dynamo.reset()
    x = torch.randn(m, n, device=DEV, dtype=DT)
    bias = torch.randn(n, device=DEV, dtype=DT)
    ref = PK.ref_bias_gelu(x, bias)
    tc = torch.compile(PK.ref_bias_gelu, mode="max-autotune-no-cudagraphs"); tc(x, bias)
    tt = med(lambda: tc(x, bias))
    gbps = 2 * m * n * DT.itemsize / 1e12
    rows = []
    for cfg in CANDS:
        try:
            k = helion.kernel(PK.bias_gelu.fn, config=helion.Config(block_sizes=cfg), static_shapes=True)
            out = k(x, bias)
            if (out.float() - ref.float()).abs().max().item() > 0.05:
                continue
            rows.append((cfg, med(lambda: k(x, bias))))
        except Exception:  # noqa: BLE001
            pass
    rows.sort(key=lambda r: r[1])
    print(f"  ({m},{n}) tc={tt*1e3:.1f}us:  " + "  ".join(f"{c}:G={tt/t:.3f}" for c, t in rows[:5]))
    del x, bias, ref, tc; torch.cuda.empty_cache()
