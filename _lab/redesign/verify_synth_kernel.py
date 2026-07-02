"""Verify a synthetic adversarial kernel against the _apply_reread heuristic.

Given a module path exposing `kernel` (a @helion.kernel fn) and `make_args()`, this:
  1. binds it, prints the primary descriptor + _apply_reread verdict + which ceiling it picks
     + the seed reduction_loops (persist vs chunk) the heuristic emits;
  2. A/B benches persist ([None]/full-extent) vs chunk ([16384]-style) at the given shape
     (median-of-9 do_bench) to see the TRUE preference.

So we can check: does the heuristic's picked ceiling match the measured winner?

Usage (FOREGROUND, serial, one kernel/process):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/redesign/verify_synth_kernel.py --mod /tmp/synth_xxx.py \
    --persist '{"reduction_loops":[null],...}' --chunk '{"reduction_loops":[16384],...}'
The --persist/--chunk configs are OPTIONAL; if omitted only the heuristic verdict is printed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402
from triton.testing import do_bench  # noqa: E402

import helion  # noqa: E402


def _load(mod_path):
    spec = importlib.util.spec_from_file_location("synth_mod", mod_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _med(fn, n=9):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    return s[len(s) // 2] * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mod", required=True)
    p.add_argument("--persist", default="")
    p.add_argument("--chunk", default="")
    p.add_argument(
        "--n",
        type=int,
        default=0,
        help="override the reduction extent N: rebuilds a single [M,N] fp32 input "
        "(only valid for synth kernels whose make_args is one [M,N] tensor)",
    )
    args = p.parse_args()
    m = _load(args.mod)
    fn = m.kernel
    kargs = m.make_args()
    if args.n:
        # rebuild the sole [M,N] input at the requested N (keeps M, dtype, device).
        t0 = kargs[0]
        kargs = (
            torch.randn(t0.shape[0], args.n, device=t0.device, dtype=t0.dtype),
            *kargs[1:],
        )

    from helion._compiler.autotuner_heuristics.triton import (
        _primary_descriptor_selected,
    )
    from helion._compiler.autotuner_heuristics.triton import (
        TritonStandardReductionHeuristic as S,
    )
    from helion._compiler.autotuner_heuristics.triton import (
        TritonUserTiledReductionHeuristic as U,
    )

    b = fn.bind(kargs)
    env = b.env
    dev = b.host_function.device_ir
    env.__enter__()
    b.host_function.__enter__()
    try:
        pd = _primary_descriptor_selected(env)
        if pd is None:
            print("pd=None (reduction seed declines this kernel)")
        else:
            ar = S._apply_reread(env.config_spec, pd)
            ceiling = "SMALL(294912)" if ar else "BIG(737280)"
            # what the heuristic emits:
            cfg = None
            for H in (S, U):
                if H.is_eligible(env, dev):
                    cfg = H.get_seed_config(env, dev).config
                    track = H.name
                    break
            # footprint bytes at full extent
            print(
                json.dumps(
                    {
                        "primary_bid": pd.block_id,
                        "category": pd.category.value,
                        "size_hint": pd.size_hint,
                        "itemsize": pd.itemsize,
                        "row_reread": pd.row_reread,
                        "carried_2d_count": pd.carried_2d_count,
                        "apply_reread": ar,
                        "ceiling_picked": ceiling,
                        "seed_track": track if cfg else None,
                        "seed_block_sizes": cfg.get("block_sizes") if cfg else None,
                        "seed_reduction_loops": cfg.get("reduction_loops")
                        if cfg
                        else None,
                    },
                    indent=1,
                )
            )
            # per-load R/S dump (the apply_reread evidence)
            facts = env.config_spec.memory_op_facts
            red_t = {
                f.tensor_name
                for f in facts
                if f.kind == "load"
                and f.tensor_name
                and any(ax == pd.block_id for ax, _ in f.reductions_fed)
            }
            print("  primary-red tensors:", sorted(t for t in red_t if t))
            for f in facts:
                if f.kind == "load" and f.tensor_name in red_t:
                    print(
                        f"    load {f.tensor_name}@g{f.graph_id} R={bool(f.reductions_fed)} S={bool(f.stores_fed)}"
                    )
    finally:
        b.host_function.__exit__(None, None, None)
        env.__exit__(None, None, None)

    if args.persist and args.chunk:

        def run(cfg_json):
            cfg = helion.Config(**json.loads(cfg_json))
            k = helion.kernel(fn.fn, config=cfg, static_shapes=False)
            return lambda: k(*kargs)

        fp, fc = run(args.persist), run(args.chunk)
        op, oc = fp(), fc()
        tp, tc = _med(fp), _med(fc)
        print(
            json.dumps(
                {
                    "persist_us": round(tp, 3),
                    "chunk_us": round(tc, 3),
                    "chunk_over_persist": round(tc / tp, 4),
                    "true_preference": "PERSIST" if tp < tc else "CHUNK",
                },
                indent=1,
            )
        )


if __name__ == "__main__":
    main()
