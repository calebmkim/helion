"""MODEL-WELL PERF TRIPWIRE (§5b half-i, the checker's one GPU step).

The mechanical checker (checker.py) is compile-only and flags CRASH/NO_FIRE/FLOOR1. §3a.4's open_RED
ALSO includes "the compiler DEFAULT beats the SEED" (the model-well floor: the seed must at least beat
doing nothing). This tripwire measures seed-vs-default on a kernel, foreground-serial (the ONLY GPU
step), and flags RED if default beats seed by >5% (the noise band). It closes Blocker B: the Gate-T
corpus was certified dry on the compile-time branches only; this measures the model-well branch on the
durable probe kernels (the newly-fired multi-fact / multi-loop / grid-collapse regions).

Usage (from /tmp, foreground-serial, fresh process per kernel):
  HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=<worktree> \
    python perf_tripwire.py <module_path> <fn_name> <arg_builder_expr>
The arg builder is supplied per kernel by the caller harness; this module exposes tripwire(fn, args).
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
from triton.testing import do_bench  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep)

N_RUNS = 9
NOISE = 1.05  # default-beats-seed only counts as RED past the ~5% noise band


def _med(fn) -> float:
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return s[len(s) // 2] * 1000.0  # us


def tripwire(fn, args, label: str = "") -> dict:
    """Bench the live seed config vs the compiler default config on (fn, args). RED iff
    default beats seed past the noise band (seed/default > NOISE)."""
    bound = fn.bind(args)
    spec = bound.env.config_spec
    seeds = list(spec.compiler_seed_configs)
    with bound.env:
        default_cfg = dict(spec.default_config())
    if not seeds:
        return {"label": label, "fired": False, "note": "no seed (declined)"}
    seed_cfg = dict(seeds[0])
    k_seed = helion.kernel(fn.fn, config=helion.Config(**seed_cfg), static_shapes=False)
    k_def = helion.kernel(fn.fn, config=helion.Config(**default_cfg), static_shapes=False)
    # warmup / sanity: both run + agree
    out_s, out_d = k_seed(*args), k_def(*args)

    def _first(o):
        return o[0] if isinstance(o, (tuple, list)) else o
    os_, od_ = _first(out_s), _first(out_d)
    match = None
    if torch.is_tensor(os_) and torch.is_tensor(od_):
        try:
            match = bool(torch.allclose(os_.float(), od_.float(), rtol=2e-2, atol=2e-2))
        except Exception:  # noqa: BLE001
            match = None
    t_seed = _med(lambda: k_seed(*args))
    t_def = _med(lambda: k_def(*args))
    ratio = t_seed / t_def if t_def else float("nan")
    red = ratio > NOISE  # default beats seed past noise => model-well RED
    return {
        "label": label, "fired": True, "seed_us": round(t_seed, 2),
        "default_us": round(t_def, 2), "seed_over_default": round(ratio, 4),
        "outputs_match": match, "RED_default_beats_seed": red,
        "seed_block_sizes": seed_cfg.get("block_sizes"),
    }
