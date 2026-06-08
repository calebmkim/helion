"""AUDITOR: looped tail is STRUCTURAL-ONLY (persistent literally fails above 2^20).

Verifies:
  1. persist_cap == env.backend.max_tensor_numel == 2**20 (1048576).
  2. At (1, 2097152) > cap: forcing persistent (reduction_loops=[None]) FAILS to
     compile (the structural reason the looped branch exists).
  3. The v3 seed at that shape is the looped branch and is CORRECT vs torch.sum.
  4. The branch has no in-sample coverage (all in-sample <= cap) -- disclosed.
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

from examples.long_sum import longsum  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

N_RUNS = 5


def med(fn):
    torch.cuda.synchronize()
    return sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))[N_RUNS // 2]


def try_cfg(x, cfg):
    """Return (compiled_ok, codegen_looped, correct, maxabs, t_us) or error str."""
    try:
        k = helion.kernel(longsum.fn, configs=[cfg])
        b = k.bind((x,))
        b.ensure_config_exists((x,))
        tcode = b.to_triton_code(helion.Config(**dict(b._config)))
        looped = "for roffset" in tcode
        out = b(x)
        out = out[0] if isinstance(out, tuple) else out
        ref = x.sum(-1)
        ok = torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3)
        maxabs = float((out.float() - ref.float()).abs().max())
        t = med(lambda: b(x)) * 1000
        return (True, looped, ok, maxabs, t)
    except Exception as e:  # noqa: BLE001
        return ("ERROR: " + type(e).__name__ + ": " + str(e)[:160])


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    bound = longsum.bind((torch.randn(1, 4096, device="cuda", dtype=torch.float32),))
    cap = bound.env.backend.max_tensor_numel
    print(f"GPU={gpu}  helion={helion.__file__}")
    print(f"persist_cap = env.backend.max_tensor_numel = {cap}  (2**20={2**20})\n")

    m, n = 1, 2097152  # > cap
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    seed = dict(compiler_seed_configs(longsum.bind((x,)).env,
                                      longsum.bind((x,)).host_function.device_ir)[0])
    print(f"shape=({m},{n})  rnumel={n} > cap={cap}")
    print(f"  v3 seed = rl={seed['reduction_loops']},w{seed['num_warps']},bs={seed['block_sizes']}")

    # 1) Force PERSISTENT at this oversized shape -> expect compile failure.
    print("\n  Forcing PERSISTENT (reduction_loops=[None]) at oversized shape:")
    r_p = try_cfg(x, helion.Config(block_sizes=[1], reduction_loops=[None],
                                   num_warps=32, num_stages=1))
    if isinstance(r_p, str):
        print(f"    persistent -> {r_p}")
        print("    => CONFIRMED: persistent cannot compile above the structural cap.")
    else:
        ok2, looped2, corr2, ma2, t2 = r_p
        print(f"    persistent COMPILED unexpectedly: looped_codegen={looped2} ok={corr2} t={t2:.1f}us")
        print("    => NOTE: persistent did NOT fail -- structural justification weak.")

    # 2) The v3 seed (looped) at this shape -> expect correct.
    print("\n  v3 SEED (looped) at oversized shape:")
    r_s = try_cfg(x, helion.Config(**seed))
    if isinstance(r_s, str):
        print(f"    seed -> {r_s}")
    else:
        ok2, looped2, corr2, ma2, t2 = r_s
        print(f"    looped_codegen={looped2}  correct={corr2}  maxabs={ma2:.2e}  t={t2:.1f}us")


if __name__ == "__main__":
    main()
