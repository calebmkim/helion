"""Dump the Stage-1 facts needed to model size_reduction_tiles OFFLINE, for every corpus
cell. One bind per cell; serialize reductions + accumulators + grid seats + feature axes +
groups to JSON so candidate footprint formulas can be swept instantly without re-binding.

Usage (from /tmp):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/redesign/dump_facts.py --out /tmp/corpus_facts.json
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "harness")))
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import json  # noqa: E402

import torch  # noqa: E402

import helion  # noqa: E402

import unified_config_recorder as REC  # noqa: E402
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    _TritonReductionSeedBase as B,
)
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    _primary_descriptor_selected,
)

OUT: list = []


def dump(corpus, kernel, shape, dtype, fn, args):
    bound = fn.bind(args)
    env = bound.env
    spec = env.config_spec
    kf = spec.reduction_kernel_fact
    rec = {"corpus": corpus, "kernel": kernel, "shape": list(shape), "dtype": dtype}
    if kf is None:
        rec["kf"] = None
        OUT.append(rec)
        del bound
        torch.cuda.empty_cache()
        return
    with env:
        pd = _primary_descriptor_selected(env)
        rec["pd_block_id"] = pd.block_id if pd else None
        rec["pd_category"] = pd.category.value if pd else None
        rec["pd_itemsize"] = pd.itemsize if pd else None
        rec["grid_ids"] = sorted(kf.grid_axis_block_ids)
        rec["nrl_ids"] = sorted(kf.non_reduction_loop_block_ids)
        rec["reduction_ids"] = sorted({d.block_id for d in kf.reductions})
        rec["bs_valid"] = sorted(spec.block_sizes.valid_block_ids())
        rec["rl_valid"] = sorted(spec.reduction_loops.valid_block_ids())
        reds = []
        for d in kf.reductions:
            reds.append(
                {
                    "block_id": d.block_id,
                    "graph_id": d.graph_id,
                    "category": d.category.value,
                    "size_hint": d.size_hint,
                    "itemsize": d.itemsize,
                    "input_load_itemsize": d.input_load_itemsize,
                    "carried_2d_count": d.carried_2d_count,
                    "full_width_output": d.full_width_output,
                    "body_live_tiles": d.body_live_tiles,
                    "row_reread": d.row_reread,
                }
            )
        rec["reductions"] = reds
        rec["groups"] = [list(g.descriptor_indices) for g in kf.coresidency_groups]
        accs = []
        for a in spec.accumulator_facts:
            dim_sizes = []
            for d in a.dim_block_ids:
                if d is None:
                    dim_sizes.append(None)
                else:
                    try:
                        dim_sizes.append(int(env.block_sizes[d].size_hint()))
                    except Exception:  # noqa: BLE001
                        dim_sizes.append(None)
            accs.append(
                {
                    "dims": list(a.dim_block_ids),
                    "dim_sizes": dim_sizes,
                    "itemsize": a.itemsize,
                }
            )
        rec["accumulators"] = accs
        seats = {}
        for gbid in sorted(kf.grid_axis_block_ids):
            seats[str(gbid)] = B._m_axis_block_size(spec, gbid)
        rec["grid_seats"] = seats
        ext = {}
        floor = {}
        bs_index_to_bid = []  # spec block_sizes index order -> block_id (for seed reconstruction)
        for i in range(len(spec.block_sizes)):
            bsp = spec.block_sizes[i]
            ext[str(bsp.block_id)] = int(bsp.size_hint)
            floor[str(bsp.block_id)] = B._block_floor(bsp)
            bs_index_to_bid.append(bsp.block_id)
        rec["bs_index_to_bid"] = bs_index_to_bid
        for d in kf.reductions:
            ext.setdefault(str(d.block_id), d.size_hint)
        rec["ext"] = ext
        rec["floor"] = floor
        try:
            mat = B._materialized_feature_axes(env, spec, bound.host_function.device_ir)
        except Exception:  # noqa: BLE001
            mat = set()
        feat = {}
        for fb in mat:
            try:
                feat[str(fb)] = int(env.block_sizes[fb].size_hint())
            except Exception:  # noqa: BLE001
                feat[str(fb)] = None
        rec["feature_axes"] = feat
        gr = 1
        try:
            for gbid in kf.grid_axis_block_ids:
                size = env.block_sizes[gbid].size
                if isinstance(size, (int, torch.SymInt)):
                    gr *= env.size_hint(size)
                else:
                    gr = 0
                    break
        except Exception:  # noqa: BLE001
            gr = 0
        rec["grid_rows"] = gr
        from helion.runtime import get_num_sm

        rec["num_sm"] = max(1, get_num_sm(env.device))
        rec["element_cap"] = env.backend.max_tensor_numel
        seeds = list(spec.compiler_seed_configs)
        rec["seed"] = dict(seeds[0]) if seeds else None
    OUT.append(rec)
    del bound
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/corpus_facts.json")
    a = p.parse_args()
    for corpus in list(REC._CORPORA):
        for (cps, kname, shape, dtype, fn, kargs, split) in REC._CORPORA[corpus](None):
            try:
                dump(cps, kname, shape, dtype, fn, kargs)
            except Exception as e:  # noqa: BLE001
                OUT.append(
                    {
                        "corpus": cps,
                        "kernel": kname,
                        "shape": list(shape),
                        "dtype": dtype,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                torch.cuda.empty_cache()
    json.dump(OUT, open(a.out, "w"), indent=1)
    print(f"dumped {len(OUT)} cells to {a.out}")


if __name__ == "__main__":
    main()
