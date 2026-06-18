"""WS2 Step-0: re-confirm the backward fact-structure baseline (EFFORT=none, no GPU bench).

For each backward kernel, bind concrete inputs, run the live seed heuristic, and dump:
  - reduction_facts (count + fields)
  - accumulator_facts (count + fields)   <- the walker fact grad_w is derived from
  - matmul_facts (count)
  - the emitted seed config (compiler_seed_configs) and the generic default_config

Expected at BASE (d6ad1156), per m-reduction.md KEY MECHANISM:
  rms_norm_bwd / layer_norm_bwd -> 0 reduction_facts -> generic _base_default_config()
  softmax_bwd                   -> 1 reduction_fact   -> T2 seed (like softmax_two_pass fwd)

Usage (cwd=/tmp, PYTHONPATH=<worktree>):
  python ws2_step0_factdump.py            # default shape, fp32
  python ws2_step0_factdump.py --M 8192 --N 4096 --dtype bf16
"""

from __future__ import annotations

import argparse
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
if _WT_ROOT not in sys.path:
    sys.path.insert(0, _WT_ROOT)

import torch  # noqa: E402

import helion  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

from examples.layer_norm import layer_norm_bwd  # noqa: E402
from examples.rms_norm import rms_norm_bwd  # noqa: E402
from examples.softmax import softmax_bwd  # noqa: E402

assert os.path.realpath(helion.__file__).startswith(os.path.realpath(_WT_ROOT) + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT_ROOT}); set PYTHONPATH=<worktree>."
)

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
EPS = 1e-5


def build_rms_bwd(m: int, n: int, dt: torch.dtype):
    x = torch.randn(m, n, device="cuda", dtype=dt)
    grad_out = torch.randn(m, n, device="cuda", dtype=dt)
    weight = torch.randn(n, device="cuda", dtype=dt)
    rsqrt = torch.rand(m, 1, device="cuda", dtype=torch.float32) + 0.5
    return rms_norm_bwd, (grad_out, x, weight, rsqrt)


def build_ln_bwd(m: int, n: int, dt: torch.dtype):
    x = torch.randn(m, n, device="cuda", dtype=dt)
    grad_out = torch.randn(m, n, device="cuda", dtype=dt)
    mean = torch.randn(m, device="cuda", dtype=torch.float32)
    rstd = torch.rand(m, device="cuda", dtype=torch.float32) + 0.5
    weight = torch.randn(n, device="cuda", dtype=dt)
    return layer_norm_bwd, (grad_out, x, mean, rstd, weight)


def build_softmax_bwd(m: int, n: int, dt: torch.dtype):
    so = torch.randn(m, n, device="cuda", dtype=dt).softmax(dim=1)
    go = torch.randn(m, n, device="cuda", dtype=dt)
    return softmax_bwd, (go, so)


BUILDERS = {
    "rms_norm_bwd": build_rms_bwd,
    "layer_norm_bwd": build_ln_bwd,
    "softmax_bwd": build_softmax_bwd,
}


def _facts_dump(facts) -> list:
    out = []
    for f in facts:
        if hasattr(f, "_asdict"):
            out.append(dict(f._asdict()))
        else:
            out.append(repr(f))
    return out


def dump(name: str, m: int, n: int, dt_name: str) -> None:
    dt = DTYPES[dt_name]
    fn, args = BUILDERS[name](m, n, dt)
    bound = fn.bind(args)
    env = bound.env
    device_ir = bound.host_function.device_ir
    spec = env.config_spec

    rfacts = list(getattr(spec, "reduction_facts", []))
    afacts = list(getattr(spec, "accumulator_facts", []))
    mfacts = list(getattr(spec, "matmul_facts", []))
    seeds = compiler_seed_configs(env, device_ir)
    default_cfg = spec.default_config()

    print(f"\n========== {name}  ({m},{n})  {dt_name} ==========")
    print(f"  reduction_facts: n={len(rfacts)}")
    for i, f in enumerate(_facts_dump(rfacts)):
        print(f"     [{i}] {f}")
    print(f"  accumulator_facts: n={len(afacts)}")
    for i, f in enumerate(_facts_dump(afacts)):
        print(f"     [{i}] {f}")
    print(f"  matmul_facts: n={len(mfacts)}")
    print(f"  SEED (compiler_seed_configs): n={len(seeds)}")
    for i, s in enumerate(seeds):
        norm = dict(s)
        spec.normalize(norm)
        print(f"     seed[{i}] normalized: {norm}")
    print(f"  GENERIC default_config: {dict(default_cfg)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES))
    ap.add_argument("kernels", nargs="*", default=list(BUILDERS))
    a = ap.parse_args()
    print(f"helion={helion.__file__}")
    for name in (a.kernels or list(BUILDERS)):
        dump(name, a.M, a.N, a.dtype)


if __name__ == "__main__":
    main()
