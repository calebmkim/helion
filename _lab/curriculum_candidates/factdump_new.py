"""Factdump for the four NEW m-reduction curriculum kernels (bias_grad, dyt,
group_norm, instance_norm). Establishes ground truth for:
  - which facts fire (reduction_facts / accumulator_facts / m_reduction_facts / matmul_facts)
  - the emitted seed config vs the generic default
  - the BLOCK STRUCTURE (each block: reduction flag, size, in block_sizes? in reduction_loops?)
    so the lever bug (feature_extent reads accumulator width, misses spatial S) is visible.

Usage (cwd=/tmp, PYTHONPATH=<worktree>):
  CUDA_VISIBLE_DEVICES=2 HELION_AUTOTUNE_EFFORT=none python factdump_new.py --dtype fp32
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _WT_ROOT not in sys.path:
    sys.path.insert(0, _WT_ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch  # noqa: E402

import helion  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

import mreduction_styles as MS  # noqa: E402

assert os.path.realpath(helion.__file__).startswith(os.path.realpath(_WT_ROOT) + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT_ROOT}); set PYTHONPATH=<worktree>."
)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
EPS = 1e-5


def build_bias_grad(dt):
    M, N = 16384, 1024
    go = torch.randn(M, N, device="cuda", dtype=dt)
    return MS.bias_grad_bwd, (go,), f"(M={M},N={N})"


def build_dyt(dt):
    M, N = 16384, 1024
    go = torch.randn(M, N, device="cuda", dtype=dt)
    x = torch.randn(M, N, device="cuda", dtype=dt)
    w = torch.randn(N, device="cuda", dtype=dt)
    return MS.dyt_bwd, (go, x, w, 0.7), f"(M={M},N={N})"


def build_group_norm(dt):
    Nn, C, S, G = 512, 128, 64, 32
    x = torch.randn(Nn, C, S, device="cuda", dtype=dt)
    go = torch.randn(Nn, C, S, device="cuda", dtype=dt)
    mean = torch.randn(Nn, G, device="cuda", dtype=torch.float32)
    rstd = torch.rand(Nn, G, device="cuda", dtype=torch.float32) + 0.5
    w = torch.randn(C, device="cuda", dtype=dt)
    return MS.group_norm_bwd, (go, x, mean, rstd, w, G), f"(N={Nn},C={C},S={S},G={G}) C*S={C*S}"


def build_instance_norm(dt):
    Bb, C, S = 512, 64, 128
    x = torch.randn(Bb, C, S, device="cuda", dtype=dt)
    go = torch.randn(Bb, C, S, device="cuda", dtype=dt)
    mean = torch.randn(Bb, C, device="cuda", dtype=torch.float32)
    rstd = torch.rand(Bb, C, device="cuda", dtype=torch.float32) + 0.5
    w = torch.randn(C, device="cuda", dtype=dt)
    return MS.instance_norm_bwd, (go, x, mean, rstd, w), f"(B={Bb},C={C},S={S}) C*S={C*S}"


BUILDERS = {
    "bias_grad": build_bias_grad,
    "dyt": build_dyt,
    "group_norm": build_group_norm,
    "instance_norm": build_instance_norm,
}


def _facts_dump(facts) -> list:
    out = []
    for f in facts:
        out.append(dict(f._asdict()) if hasattr(f, "_asdict") else repr(f))
    return out


def dump(name: str, dt_name: str) -> None:
    dt = DTYPES[dt_name]
    fn, args, desc = BUILDERS[name](dt)
    bound = fn.bind(args)
    env = bound.env
    device_ir = bound.host_function.device_ir
    spec = env.config_spec

    bs_ids = set(spec.block_sizes.valid_block_ids())
    rl_ids = set(spec.reduction_loops.valid_block_ids())

    print(f"\n========== {name}  {desc}  {dt_name} ==========")
    print("  BLOCK STRUCTURE (block_id: reduction? size in_block_sizes? in_reduction_loops?):")
    for bs in env.block_sizes:
        try:
            sz = env.size_hint(bs.size)
        except Exception:
            sz = "?"
        print(f"     blk{bs.block_id}: reduction={bs.reduction} size={sz} "
              f"in_bs={bs.block_id in bs_ids} in_rl={bs.block_id in rl_ids}")
    print(f"  grid_block_ids: {device_ir.grid_block_ids}")

    rfacts = list(getattr(spec, "reduction_facts", []))
    afacts = list(getattr(spec, "accumulator_facts", []))
    mfacts = list(getattr(spec, "matmul_facts", []))
    mrfacts = list(getattr(spec, "m_reduction_facts", []))
    print(f"  reduction_facts: n={len(rfacts)}")
    for i, f in enumerate(_facts_dump(rfacts)):
        print(f"     [{i}] {f}")
    print(f"  accumulator_facts: n={len(afacts)}")
    for i, f in enumerate(_facts_dump(afacts)):
        print(f"     [{i}] {f}")
    print(f"  m_reduction_facts: n={len(mrfacts)}")
    for i, f in enumerate(_facts_dump(mrfacts)):
        print(f"     [{i}] {f}")
    print(f"  matmul_facts: n={len(mfacts)}")

    seeds = compiler_seed_configs(env, device_ir)
    print(f"  SEED (compiler_seed_configs): n={len(seeds)}")
    for i, s in enumerate(seeds):
        norm = dict(s)
        spec.normalize(norm)
        print(f"     seed[{i}] normalized: {norm}")
    print(f"  GENERIC default_config: {dict(spec.default_config())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("kernels", nargs="*", default=list(BUILDERS))
    a = ap.parse_args()
    print(f"helion={helion.__file__}")
    for name in (a.kernels or list(BUILDERS)):
        dump(name, a.dtype)


if __name__ == "__main__":
    main()
