"""AUDITOR: is the rnumel>16384 w32 breakpoint evidence-based, or fenced at
sum's max in-sample row (16384)?

The gate is `num_load<=1 AND rnumel>16384 -> w32`. sum's MAX in-sample rnumel is
EXACTLY 16384 (shape (2048,16384)), which the `>16384` (strict) excludes. If
w32 already wins at rnumel=16384 for num_load==1, then the breakpoint was set to
spare sum's in-sample row, NOT where physics flips.

Test real sum_kernel (num_load==1), persistent path, w16 vs w32, across rnumel
straddling the breakpoint, at sum's in-sample M values AND M=1 (long_sum regime).
Includes sum's exact in-sample shape (2048,16384). w32/w16 < 1 => w32 wins.
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

N_RUNS = 9
WARPS = [16, 32]
# rnumel straddling 16384, at sum-in-sample M (2048) and long_sum M (1)
CASES = [
    (2048, 8192), (2048, 16384),  # (2048,16384) IS sum's max in-sample shape
    (2048, 16385), (2048, 24576), (2048, 32768),
    (1, 8192), (1, 16384), (1, 24576), (1, 32768), (1, 65536),
]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def time_cfg(x, ref, warps):
    cfg = helion.Config(block_sizes=[1], reduction_loops=[None],
                        num_warps=warps, num_stages=1)
    k = helion.kernel(sum_kernel.fn, configs=[cfg])
    b = k.bind((x,))
    b.ensure_config_exists((x,))
    tcode = b.to_triton_code(helion.Config(**dict(b._config)))
    assert "for roffset" not in tcode, "expected persistent codegen"
    out = b(x)
    out = out[0] if isinstance(out, tuple) else out
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3), "correctness"
    return med(lambda: b(x)) * 1000


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  N_RUNS={N_RUNS}  sum_kernel num_load=1 persistent\n")
    print(f"{'shape':>14} {'rnumel':>8} | {'w16':>8} {'w32':>8} | best | w32/w16  gate_gives  note")
    for m, n in CASES:
        x = torch.randn(m, n, device="cuda", dtype=torch.float32)
        ref = x.sum(-1)
        t = {wp: time_cfg(x, ref, wp) for wp in WARPS}
        best_w = min(t, key=t.get)
        ratio = t[32] / t[16]
        gate_w = 32 if n > 16384 else 16
        note = "<-- sum max in-sample" if (m, n) == (2048, 16384) else ""
        print(f"{str((m,n)):>14} {n:>8} | {t[16]:>8.2f} {t[32]:>8.2f} | w{best_w:>2} | "
              f"{ratio:.3f}   w{gate_w}   {note}")


if __name__ == "__main__":
    main()
