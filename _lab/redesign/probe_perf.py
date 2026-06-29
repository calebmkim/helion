"""P4 Tier-2 perf check for the probe kernels: seed config vs the compiler DEFAULT.

For each probe that FIRES a reduction seed, replay the seed config vs the compiler default
config IN ONE PROCESS on the same tensors, median-of-9 cold do_bench, accuracy-gated against
the kernel's own default output. Reports seed/default; the §5 "perf >= default, expect better"
bar = ratio <= ~1.05 (seed at least as fast as default). A probe that fires the right path but
is SLOWER than default is a failed Tier-2.

One fresh process per kernel (do_bench jitter / no cross-kernel contamination). GPU foreground.

Usage (from /tmp):
  HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=<worktree> python probe_perf.py --probe p7-gridtile-then-usertile
  # or --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HARNESS = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS, "..", ".."))
if _HARNESS not in sys.path:
    sys.path.insert(0, _HARNESS)

import torch  # noqa: E402
from triton.testing import do_bench  # noqa: E402

import helion  # noqa: E402

import ir_introspect as II  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep)

N_RUNS = 9


def _med(fn) -> float:
    torch.cuda.synchronize()
    samples = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return samples[len(samples) // 2] * 1000.0  # us


def _first(o):
    return o[0] if isinstance(o, (tuple, list)) else o


def check(slug: str) -> dict:
    fn, args = II._probe_fn_args(slug)
    bound = fn.bind(args)
    spec = bound.env.config_spec
    with bound.env:
        default_cfg = dict(spec.default_config())
        seeds = list(spec.compiler_seed_configs)
    rec = {"probe": slug, "fired": list(spec.autotuner_heuristics)}
    if not seeds:
        rec["note"] = "no seed (declined) — Tier-2 N/A"
        return rec
    seed_cfg = dict(seeds[0])
    rec["seed_cfg_bs"] = seed_cfg.get("block_sizes")

    k_seed = helion.kernel(fn.fn, config=helion.Config(**seed_cfg), static_shapes=False)
    k_def = helion.kernel(fn.fn, config=helion.Config(**default_cfg), static_shapes=False)
    out_seed = _first(k_seed(*args))
    out_def = _first(k_def(*args))
    same = None
    if torch.is_tensor(out_seed) and torch.is_tensor(out_def):
        try:
            same = bool(
                torch.allclose(out_seed.float(), out_def.float(), rtol=2e-2, atol=2e-2)
            )
        except Exception:  # noqa: BLE001
            same = None
    t_seed = _med(lambda: k_seed(*args))
    t_def = _med(lambda: k_def(*args))
    ratio = t_seed / t_def if t_def else float("nan")
    rec.update(
        seed_us=round(t_seed, 3),
        default_us=round(t_def, 3),
        seed_over_default=round(ratio, 4),
        outputs_match=same,
        seed_at_least_as_fast=bool(ratio <= 1.05),
        seed_faster=bool(ratio < 0.95),
    )
    del bound
    torch.cuda.empty_cache()
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probe", default="")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    print(f"helion={helion.__file__}", flush=True)
    nv = os.popen(
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
    ).read().strip()
    print(f"GPU mem.used={nv}", flush=True)
    slugs = II._PROBES if args.all else [args.probe]
    for slug in slugs:
        try:
            rec = check(slug)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            rec = {"probe": slug, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
