"""Stage-1 IR ground-truth dumper — the ANSWER KEY for the categorization pass (P1).

For each kernel, enumerate every ReductionLowering occurrence with its FAITHFUL structural
signature, so the new categorization pass (§2.1) can be written against real IR, not guesses:
  - graph_id (the co-residency key §2.2)
  - block_index (reduction axis)
  - is it a grid axis? fully-resident grid (cdiv==1, FULL_GRID) vs partial (cdiv>1, GRID_TILE)?
  - block_size_source type (Fixed / ReductionLoop / LoopSpec ...)
  - in spec.block_sizes? in spec.reduction_loops? (the legacy ACCESS discriminators)
  - rollable = sole-rdim-in-graph (§3): does its graph contain >1 DISTINCT rdim?
  - the per-graph distinct-rdim set (proves the rolling-blocked invariant)

Also dumps accumulator dim_block_ids (carried_2d signal) and grid_block_ids.

Usage (from /tmp):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-redesign \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/redesign/ir_introspect.py --probes
    # or --kernel rms_norm  (a curriculum kernel)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
_LOCAL_ROOT = os.path.abspath(os.path.join(_WT_ROOT, ".."))
_KERNELS_DIR = os.path.join(
    _LOCAL_ROOT, "prompts-lab", "reduction-generality", "kernels"
)
for _d in (
    os.path.join(_HARNESS_DIR, "..", "harness"),
    os.path.join(_HARNESS_DIR, "..", "prompts"),
    os.path.join(_WT_ROOT, "examples"),
):
    _d = os.path.abspath(_d)
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT_ROOT})."
)

_PROBES = [
    "p1-outer-product-coresident",
    "p2-feature-plus-rowaccum-offcorpus",
    "p3-full-grid-nonquant",
    "p4-two-rollable-sequential",
    "p5-3d-reduction-tile",
    "p6-mixed-coresident-plus-sequential",
    "p7-gridtile-then-usertile",
    "p8-fullgrid-plus-usertile",
    "p9-nonred-loop-then-fullextent",
    "p10-usertile-and-gridtile",
    "p11-fullextent-then-nonred-loop",
    "oos1-jagged-declined",
    "oos2-strided-dim0",
]


def _load_probe(slug: str):
    path = os.path.join(_KERNELS_DIR, slug, "kernel.py")
    spec = importlib.util.spec_from_file_location(
        f"probe_{slug.replace('-', '_')}", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _probe_fn_args(slug: str):
    mod = _load_probe(slug)
    from helion.runtime.kernel import Kernel

    if hasattr(mod, "get_kernel"):
        fn = mod.get_kernel()
    else:
        fn = next(
            getattr(mod, n)
            for n in dir(mod)
            if isinstance(getattr(mod, n), Kernel)
        )
    return fn, mod.make_args()


def introspect(fn, args) -> dict:
    from helion._compiler.compile_environment import FixedBlockSizeSource
    from helion._compiler.compile_environment import (
        ReductionLoopBlockSizeSource,
    )
    from helion._compiler.inductor_lowering import ReductionLowering

    bound = fn.bind(args)
    env = bound.env
    dev = bound.host_function.device_ir
    spec = env.config_spec
    # size_hint() reads the live env; run the whole introspection inside it.
    _ctx = env
    _ctx.__enter__()
    grid_ids = {b for bids in dev.grid_block_ids for b in bids}
    bs_valid = set(spec.block_sizes.valid_block_ids())
    rl_valid = set(spec.reduction_loops.valid_block_ids())

    # Per-graph: distinct rdims (block_index of each ReductionLowering).
    per_graph_rdims: dict[int, set[int]] = {}
    occurrences = []
    for gi in dev.graphs:
        for node in gi.graph.nodes:
            low = node.meta.get("lowering")
            if isinstance(low, ReductionLowering):
                bid = getattr(low, "block_index", None)
                if bid is None:
                    continue
                per_graph_rdims.setdefault(gi.graph_id, set()).add(bid)
                occurrences.append((gi.graph_id, bid))

    def _src_type(bid: int) -> str:
        try:
            src = env.block_sizes[bid].block_size_source
        except Exception as e:  # noqa: BLE001
            return f"<{type(e).__name__}>"
        return type(src).__name__

    def _cdiv1(bid: int) -> bool:
        """FULL_GRID test: FixedBlockSizeSource with block == extent (cdiv==1)."""
        try:
            info = env.block_sizes[bid]
            src = info.block_size_source
            if not isinstance(src, FixedBlockSizeSource):
                return False
            if not isinstance(info.size, (int, torch.SymInt)):
                return False
            return bool(env.known_equal(src.value, info.size))
        except Exception:  # noqa: BLE001
            return False

    red_axes = []
    for (gid, bid) in sorted(set(occurrences)):
        rdims_in_graph = per_graph_rdims.get(gid, set())
        try:
            sh = env.block_sizes[bid].size_hint()
        except Exception:  # noqa: BLE001
            sh = None
        red_axes.append({
            "graph_id": gid,
            "block_id": bid,
            "size_hint": sh,
            "src": _src_type(bid),
            "in_grid": bid in grid_ids,
            "full_grid_cdiv1": _cdiv1(bid),
            "in_block_sizes": bid in bs_valid,
            "in_reduction_loops": bid in rl_valid,
            "rollable_sole_rdim_in_graph": len(rdims_in_graph) == 1,
            "distinct_rdims_in_graph": sorted(rdims_in_graph),
        })

    accs = [
        {"dim_block_ids": list(a.dim_block_ids), "itemsize": a.itemsize}
        for a in spec.accumulator_facts
    ]
    out = {
        "grid_block_ids": [list(b) for b in dev.grid_block_ids],
        "bs_valid": sorted(bs_valid),
        "rl_valid": sorted(rl_valid),
        "n_graphs": len(dev.graphs),
        "per_graph_rdims": {str(g): sorted(r) for g, r in per_graph_rdims.items()},
        "reduction_axes": red_axes,
        "accumulators": accs,
        "n_reduction_facts": len(spec.reduction_facts),
        "fired": list(spec.autotuner_heuristics),
    }
    _ctx.__exit__(None, None, None)
    del bound
    torch.cuda.empty_cache()
    return out


def _curriculum_fn_args(kname: str):
    from run2_measure_g import KERNELS  # noqa: E402
    from shapes_v3_draft import SHAPES  # noqa: E402

    fn, builder, _ref = KERNELS[kname]
    # one representative shape (the first val/train shape)
    shapes = next(iter(SHAPES[kname].values()))
    m, n = shapes[0]
    return fn, builder(m, n)[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probes", action="store_true")
    p.add_argument("--kernel", default="", help="a curriculum kernel name")
    p.add_argument("--out", default="")
    args = p.parse_args()
    print(f"helion={helion.__file__}\n", flush=True)
    results = {}
    targets = []
    if args.probes:
        targets = [("probe", s) for s in _PROBES]
    if args.kernel:
        targets += [("curriculum", k) for k in args.kernel.split(",")]
    for kind, name in targets:
        try:
            fn, kargs = (
                _probe_fn_args(name)
                if kind == "probe"
                else _curriculum_fn_args(name)
            )
            res = introspect(fn, kargs)
        except Exception as e:  # noqa: BLE001
            import traceback

            res = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}
        results[name] = res
        ra = res.get("reduction_axes", [])
        print(f"=== {name} (n_graphs={res.get('n_graphs')}, "
              f"n_red_facts={res.get('n_reduction_facts')}, fired={res.get('fired')}) ===")
        print(f"    grid={res.get('grid_block_ids')} bs={res.get('bs_valid')} "
              f"rl={res.get('rl_valid')} per_graph_rdims={res.get('per_graph_rdims')}")
        for a in ra:
            cat = (
                "FULL_GRID" if a["full_grid_cdiv1"]
                else "GRID_TILE(partial)" if a["in_grid"]
                else "USER_TILE" if a["in_block_sizes"]
                else "ROLLED/MATERIALIZED(FULL_SLICE)"
            )
            print(f"    g{a['graph_id']} bid={a['block_id']} sh={a['size_hint']} "
                  f"src={a['src']} -> {cat} "
                  f"rollable={a['rollable_sole_rdim_in_graph']} "
                  f"(rdims_in_g={a['distinct_rdims_in_graph']})")
        if res.get("accumulators"):
            print(f"    acc={res['accumulators']}")
        print()
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
