"""Dump T1 seeds (rms_norm/sum/long_sum/layer_norm) across a comprehensive shape
grid as a stable, sorted text blob. Run at HEAD and at the v4 champion (37b27f67)
and diff the two blobs to PROVE byte-identical T1 seeds.
"""

from __future__ import annotations

import sys

import torch

import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402


def f32(*s):
    return torch.randn(*s, device="cuda", dtype=torch.float32)


def emit(label, fn, args):
    try:
        bound = fn.bind(tuple(args))
        seeds = compiler_seed_configs(bound.env, bound.host_function.device_ir)
        names = sorted(bound.env.config_spec.autotuner_heuristics)
        out = []
        for s in seeds:
            sd = dict(s)
            out.append({k: sd.get(k) for k in
                        ("block_sizes", "reduction_loops", "num_warps", "num_stages")})
        print(f"{label} :: heuristics={names} :: seeds={out}")
    except Exception as e:  # noqa: BLE001
        print(f"{label} :: ERROR {type(e).__name__}: {str(e)[:100]}")


def main():
    from examples.sum import sum_kernel
    from examples.rms_norm import rms_norm_fwd
    from examples.long_sum import longsum, longsum_w_red_loop, longsum_manual
    from examples.layer_norm import layer_norm_fwd

    # sum: vary M and N (rnumel = N)
    for (m, n) in [(2048, 256), (2048, 1024), (2048, 4096), (2048, 16384),
                   (8192, 256), (8192, 1024), (32768, 256), (32768, 1024),
                   (1024, 8192), (4096, 2048)]:
        emit(f"sum({m},{n})", sum_kernel, (f32(m, n),))

    # rms_norm: x + weight
    for (m, n) in [(2048, 1024), (2048, 4096), (2048, 16384), (8192, 8192),
                   (4096, 1024), (1, 131072), (8, 65536)]:
        emit(f"rms_norm({m},{n})", rms_norm_fwd, (f32(m, n), f32(n), 1e-5))

    # long_sum variants: tiny M, huge rnumel
    for (m, n) in [(1, 32768), (8, 131072), (16, 262144), (4, 524288), (2, 1048576)]:
        emit(f"longsum({m},{n})", longsum, (f32(m, n),))
    emit("longsum_wred(8,131072)", longsum_w_red_loop, (f32(8, 131072),))
    emit("longsum_man(8,131072)", longsum_manual, (f32(8, 131072),))

    # layer_norm: x, normalized_shape, weight, bias
    for (m, n) in [(2048, 1024), (2048, 4096), (4096, 8192), (8192, 16384)]:
        emit(f"layer_norm({m},{n})", layer_norm_fwd, (f32(m, n), [n], f32(n), f32(n)))


if __name__ == "__main__":
    main()
