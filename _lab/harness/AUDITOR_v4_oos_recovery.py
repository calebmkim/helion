"""AUDITOR (v4): confirm the rnumel-only gate now RECOVERS the large-rnumel
multi-load w32 win that the deleted num_load fence denied.

Two parts:
  (A) LIVE v4 seed: for held-out large-rnumel rms_norm (num_load=2) and
      layer_norm (num_load>=2/3), assert the LIVE heuristic now emits
      num_warps=32 (the fence is gone -> gate is rnumel>16384 alone).
  (B) A/B: time w16 vs w32 for layer_norm at large rnumel to confirm w32 is
      actually faster (recovers 30-40%). rms_norm A/B is in the sibling
      AUDITOR_rmsnorm_largeN_warps.py.
"""

from __future__ import annotations

import os
import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from examples.layer_norm import layer_norm_fwd  # noqa: E402
from examples.rms_norm import rms_norm_fwd  # noqa: E402
from triton.testing import do_bench  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

EPS = 1e-5
N_RUNS = 9
WARPS = [16, 32]
SHAPES = [(1, 32768), (1, 131072), (1, 262144), (16, 131072), (16, 262144)]


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def build(kernel, m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    if kernel == "rms_norm":
        return rms_norm_fwd, (x, w, EPS)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    return layer_norm_fwd, (x, [n], w, b, EPS)


def live_seed(fn, args):
    b = fn.bind(args)
    f = b.env.config_spec.reduction_facts[0]
    s = dict(compiler_seed_configs(b.env, b.host_function.device_ir)[0])
    return f, s


def time_layer_norm(args, warps):
    x, ns, w, bias, eps = args
    cfg = helion.Config(block_sizes=[1], reduction_loops=[None],
                        num_warps=warps, num_stages=1)
    k = helion.kernel(layer_norm_fwd.fn, configs=[cfg])
    bnd = k.bind(args)
    bnd.ensure_config_exists(args)
    tcode = bnd.to_triton_code(helion.Config(**dict(bnd._config)))
    assert "for roffset" not in tcode, "expected persistent codegen"
    out = bnd(*args)
    out = out[0] if isinstance(out, tuple) else out
    ref = torch.nn.functional.layer_norm(x, ns, w, bias, eps)
    assert torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3), "correctness"
    return med(lambda: bnd(*args)) * 1000


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu}  helion={helion.__file__}  N_RUNS={N_RUNS}\n")

    print("=== (A) LIVE v4 seed num_warps for held-out large-rnumel multi-load ===")
    for kernel in ("rms_norm", "layer_norm"):
        for m, n in SHAPES:
            fn, args = build(kernel, m, n)
            f, s = live_seed(fn, args)
            w = s["num_warps"]
            ok = "OK(w32)" if (n > 16384 and w == 32) else ("OK(w<=16)" if n <= 16384 else f"FAIL got w{w}")
            print(f"  {kernel:>10}{(m, n)} nl={f.num_load} rn={f.size_hint} "
                  f"-> seed num_warps=w{w}  loops={s['reduction_loops']}  [{ok}]")

    print("\n=== (B) layer_norm A/B: w16 vs w32 at large rnumel ===")
    print(f"{'shape':>14} {'rnumel':>8} | {'w16':>8} {'w32':>8} | best | w32/w16")
    for m, n in SHAPES:
        _, args = build("layer_norm", m, n)
        t = {wp: time_layer_norm(args, wp) for wp in WARPS}
        best_w = min(t, key=t.get)
        print(f"{(m, n)!s:>14} {n:>8} | {t[16]:>8.2f} {t[32]:>8.2f} | w{best_w:>2} | {t[32] / t[16]:.3f}")


if __name__ == "__main__":
    main()
