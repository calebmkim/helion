"""Dump the full ReductionKernelFact + accumulator_facts for chosen corpus kernels.

Reuses unified_config_recorder's corpus iterators so the kernels/shapes/dtypes match the
gate. For each (kernel, shape) prints every descriptor (category, block_id, graph_id,
size_hint, itemsize, input_load_itemsize, carried_2d_count, full_width_output,
body_live_tiles, row_reread), the co-residency groups, grid_axis_block_ids,
non_reduction_loop_block_ids, the accumulator_facts (dim_block_ids + itemsize), and the
spec block-size membership (valid block_sizes / reduction_loops). This is the answer key for
the membership-based group_footprint (#2/#3).

Usage (from /tmp):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/redesign/fact_dump.py \
    --corpus mreduction --kernels group_norm_bwd,instance_norm_bwd
"""

from __future__ import annotations

import argparse
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
sys.path.insert(0, os.path.abspath(os.path.join(_HARNESS_DIR, "..", "harness")))

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT_ROOT})."
)

import unified_config_recorder as REC  # noqa: E402


def dump(fn: object, args: tuple) -> None:
    bound = fn.bind(args)
    env = bound.env
    spec = env.config_spec
    kf = spec.reduction_kernel_fact
    with env:
        print(f"  bs_valid={sorted(spec.block_sizes.valid_block_ids())} "
              f"rl_valid={sorted(spec.reduction_loops.valid_block_ids())}")
        if kf is None:
            print("  reduction_kernel_fact = None")
        else:
            print(f"  grid_axis_block_ids={kf.grid_axis_block_ids} "
                  f"non_reduction_loop_block_ids={kf.non_reduction_loop_block_ids}")
            for d in kf.reductions:
                print(f"  RED cat={d.category.value:10s} bid={d.block_id} g={d.graph_id} "
                      f"sh={d.size_hint} isz={d.itemsize} ilsz={d.input_load_itemsize} "
                      f"c2d={d.carried_2d_count} fwo={d.full_width_output} "
                      f"blt={d.body_live_tiles} reread={d.row_reread} "
                      f"rollable={d.rollable} pinned={d.pinned}")
            for g in kf.coresidency_groups:
                print(f"  GROUP graph_id={g.graph_id} idx={g.descriptor_indices}")
        for a in spec.accumulator_facts:
            sizes = []
            for d in a.dim_block_ids:
                if d is None:
                    sizes.append(None)
                else:
                    try:
                        sizes.append(env.block_sizes[d].size_hint())
                    except Exception:  # noqa: BLE001
                        sizes.append("?")
            print(f"  ACC dim_block_ids={list(a.dim_block_ids)} sizes={sizes} "
                  f"itemsize={a.itemsize}")
        seeds = list(spec.compiler_seed_configs)
        print(f"  fired={list(spec.autotuner_heuristics)} "
              f"seed={dict(seeds[0]) if seeds else None}")
    del bound
    torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="")
    p.add_argument("--kernels", default="")
    a = p.parse_args()
    print(f"helion={helion.__file__}\n", flush=True)
    corpora = a.corpus.split(",") if a.corpus else list(REC._CORPORA)
    kfilter = set(a.kernels.split(",")) if a.kernels else None
    for corpus in corpora:
        for (cps, kname, shape, dtype, fn, kargs, split) in REC._CORPORA[corpus](kfilter):
            print(f"=== {cps}/{kname}/{shape}/{dtype} ===", flush=True)
            try:
                dump(fn, kargs)
            except Exception as e:  # noqa: BLE001
                import traceback
                print(f"  ERROR {type(e).__name__}: {e}")
                traceback.print_exc()
            print(flush=True)


if __name__ == "__main__":
    main()
