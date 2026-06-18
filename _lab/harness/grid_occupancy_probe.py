"""Grid-occupancy probe: when does tiny-M (small grid) prefer looped+warps32
over persistent+warps16 at moderate rnumel?

The long_sum oracle picks LOOPED reduction_loops + num_warps=32 even at rnumel
32768/65536 (which my 256KiB persistent threshold keeps persistent) when M is
tiny (1,2) — few programs ⇒ a single persistent pass per program can't saturate
the GPU. We sweep M (= grid size, since m_block=1) at a few moderate rnumel and
compare persistent/16 (my current seed) vs looped(16384)/32 to locate the
grid-occupancy crossover. We use sum_kernel (num_load=1, the long_sum workload).
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402

from examples.sum import sum_kernel  # noqa: E402

N_RUNS = 5
MS = [1, 2, 4, 8, 16, 32, 64, 132, 256, 1024]
RNUMELS = [32768, 65536]  # both <= 256KiB persistent threshold


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def run(m, n, reduction_loops, warps):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    ref = x.sum(-1)
    cfg = helion.Config(block_sizes=[1], reduction_loops=reduction_loops,
                        num_warps=warps, num_stages=1)
    k = helion.kernel(sum_kernel.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3)
    return med(lambda: b(x)) * 1000


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')}  helion={helion.__file__}\n")
    for n in RNUMELS:
        print(f"=== rnumel={n} ({n*4//1024}KiB) ===")
        print(f"  {'M(grid)':>8} {'pers/16':>9} {'loop16k/32':>11} {'pers/loop':>9} {'winner':>7}")
        for m in MS:
            p = run(m, n, [None], 16)
            l = run(m, n, [16384], 32)
            ratio = p / l
            print(f"  {m:>8} {p:>9.1f} {l:>11.1f} {ratio:>9.3f} {'loop' if ratio>1 else 'pers':>7}")
        print()


if __name__ == "__main__":
    main()
