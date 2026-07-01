"""Phase-A grounding probe — the ANSWER KEY for the per-group live-tile attribution walk.

For each kernel dumps, per graph_id:
  - graph TYPE (Root / ForLoop / ReductionLoop / If / Else / WhileLoop / WhileCond)
  - the CF-tree EDGES out of it: ``_if`` -> (if_gid, else_gid) with the shared _if node key,
    and for-loop / reduction-loop nesting edges (child gid).
  - every ReductionLowering block_index in the graph.
  - ``_graph_peak_live_tiles(graph, env)`` — the per-tile dim_block_ids of the peak-live set.
  - ``_graph_peak_live_by_axis`` — the per-axis count (for cross-check).

Also dumps the original-graph co-residency occurrences (``_original_graph_reductions``) + the
group keys, so the graph->group attribution can be designed against real IR: which body graph
(rolled ReductionLoop / user ForLoop / If / Else / Root) feeds which original-graph group key.

Usage (from /tmp):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-redesign \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/redesign/ground_live_tiles.py --probes
    # or --kernel jsd,kl_div,softmax  (curriculum kernels)
    # or --probe-slug p6-mixed-coresident-plus-sequential
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
            getattr(mod, n) for n in dir(mod) if isinstance(getattr(mod, n), Kernel)
        )
    return fn, mod.make_args()


def _curriculum_fn_args(kname: str):
    from run2_measure_g import KERNELS
    from shapes_v3_draft import SHAPES

    fn, builder, _ref = KERNELS[kname]
    shapes = next(iter(SHAPES[kname].values()))
    m, n = shapes[0]
    return fn, builder(m, n)[0]


def _recorder_targets(corpus: str, kernels: set[str] | None):
    """Reuse the unified_config_recorder corpus iterators to reach transfer/vllm/mreduction
    kernels (grpo, per_token_group_fp8_quant, layer_norm_bwd, ...). Yields (label, fn, args),
    ONE representative shape per (corpus, kernel) — the first the iterator emits."""
    import unified_config_recorder as UCR

    seen: set[tuple[str, str]] = set()
    for cps, kname, shape, dtype, fn, kargs, _split in UCR._CORPORA[corpus](kernels):
        if (cps, kname) in seen:
            continue
        seen.add((cps, kname))
        yield (f"{cps}/{kname}/{shape}/{dtype}", fn, kargs)


def _peak_candidates(graph, env) -> dict:
    """Return competing 'peak-live tile set' definitions over ONE graph, for the fidelity
    exploration in Phase A. Each is a list of dim_block_ids tuples.

      D1_count  — the live set at the step with the MOST tiles (the scaffolded default).
      D2_rank   — the live set at the step with the max Σ(#block dims) [register-pressure
                  proxy assuming equal per-dim width].
      D3_union  — for EACH distinct tile shape, the live set at the step that MAXIMIZES that
                  shape's simultaneous count; unioned as a multiset (max count per shape). A
                  superset that never under-counts any shape's peak co-residency.
    """
    nodes = list(graph.nodes)
    last_use = {}
    for i, node in enumerate(nodes):
        for inp in node.all_input_nodes:
            last_use[inp] = i
    shape_of = {}
    for node in nodes:
        val = node.meta.get("val")
        import torch as _t

        if isinstance(val, _t.Tensor) and val.shape:
            dims = tuple(env.resolve_block_id(s) for s in val.shape)
            if any(d is not None for d in dims):
                shape_of[node] = dims

    # sweep, recording the live shape-multiset at every step.
    steps = []  # list[list[dim_tuple]]
    live = set()
    for i, node in enumerate(nodes):
        if node in shape_of:
            live.add(node)
        steps.append([shape_of[v] for v in live])
        live = {v for v in live if last_use.get(v, -1) > i}

    def _rank(dt):
        return sum(1 for d in dt if d is not None)

    # D1: max count.
    d1 = max(steps, key=len, default=[])
    # D2: max Σ rank.
    d2 = max(steps, key=lambda s: sum(_rank(t) for t in s), default=[])
    # D3: per-shape max simultaneous count, unioned.
    from collections import Counter

    best_per_shape: dict[tuple, int] = {}
    for s in steps:
        c = Counter(s)
        for shape, cnt in c.items():
            if cnt > best_per_shape.get(shape, 0):
                best_per_shape[shape] = cnt
    d3 = []
    for shape, cnt in best_per_shape.items():
        d3.extend([list(shape)] * cnt)
    return {
        "D1_count": [list(t) for t in d1],
        "D2_rank": [list(t) for t in d2],
        "D3_union": d3,
    }


def introspect(fn, args) -> dict:
    bound = fn.bind(args)
    env = bound.env
    hostfn = bound.host_function
    dev = hostfn.device_ir
    env.__enter__()
    hostfn.__enter__()
    try:
        return _introspect_body(env, dev)
    finally:
        try:
            hostfn.__exit__(None, None, None)
        except Exception:
            pass
        env.__exit__(None, None, None)
        del bound
        torch.cuda.empty_cache()


def _introspect_body(env, dev) -> dict:
    from helion._compiler import device_ir as DIR
    from helion._compiler.device_ir import ElseGraphInfo
    from helion._compiler.device_ir import ForLoopGraphInfo
    from helion._compiler.device_ir import IfGraphInfo
    from helion._compiler.device_ir import ReductionLoopGraphInfo
    from helion._compiler.inductor_lowering import ReductionLowering
    from helion.language import _tracing_ops

    grid_ids = sorted({b for bids in dev.grid_block_ids for b in bids})

    # Per-graph structural dump.
    graphs_out = []
    for gi in dev.graphs:
        tname = type(gi).__name__
        # CF edges out of this graph.
        if_edges = []
        loop_edges = []
        red_bids = []
        for node in gi.graph.nodes:
            low = node.meta.get("lowering")
            if isinstance(low, ReductionLowering) and isinstance(
                getattr(low, "block_index", None), int
            ):
                red_bids.append(low.block_index)
            if node.op != "call_function":
                continue
            if node.target is _tracing_ops._if and len(node.args) >= 3:
                _, if_gid, else_gid, *_rest = node.args
                if_edges.append(
                    {"if_node_key": id(node), "if_gid": if_gid, "else_gid": else_gid}
                )
            elif (
                _tracing_ops.is_for_loop_target(node.target)
                and node.args
                and isinstance(node.args[0], int)
            ):
                loop_edges.append(node.args[0])
        # original_graph_id for rolled subgraphs (maps rolled body -> its source graph).
        orig_gid = getattr(gi, "original_graph_id", None)
        loop_block_ids = list(getattr(gi, "block_ids", []) or [])
        live_tiles = [list(t) for t in DIR._graph_peak_live_tiles(gi.graph, env)]
        by_axis = {
            str(k): v for k, v in DIR._graph_peak_live_by_axis(gi.graph, env).items()
        }
        # Competing "peak step" definitions (fidelity exploration): D1 = max tile COUNT
        # (the scaffolded default), D2 = max Σ(tile ranks) [register-pressure proxy at unit
        # block], D3 = UNION over every step of the distinct tile shapes ever co-live at the
        # step that maximizes that shape's own count (a superset never under-counting any axis).
        peaks = _peak_candidates(gi.graph, env)
        graphs_out.append(
            {
                "graph_id": gi.graph_id,
                "type": tname,
                "is_reduction_loop": isinstance(gi, ReductionLoopGraphInfo),
                "is_if": isinstance(gi, IfGraphInfo),
                "is_else": isinstance(gi, ElseGraphInfo),
                "is_for_loop": isinstance(gi, ForLoopGraphInfo)
                and not isinstance(gi, ReductionLoopGraphInfo),
                "original_graph_id": orig_gid,
                "loop_block_ids": loop_block_ids,
                "reduction_block_ids": sorted(set(red_bids)),
                "if_edges": if_edges,
                "loop_edges": loop_edges,
                "live_tiles": live_tiles,
                "live_by_axis": by_axis,
                "peaks": peaks,
            }
        )

    # co-residency ground truth.
    occurrences = dev._original_graph_reductions()
    groups_by_gid: dict[int, list[int]] = {}
    for gid, bid in occurrences:
        groups_by_gid.setdefault(gid, []).append(bid)

    # The STORED per-group live-tile fact (the actual wired CoResidencyGroup.live_tiles).
    kf = env.config_spec.reduction_kernel_fact
    stored_groups = []
    if kf is not None:
        for g in kf.coresidency_groups:
            stored_groups.append(
                {
                    "graph_id": g.graph_id,
                    "descriptor_bids": [
                        kf.reductions[i].block_id for i in g.descriptor_indices
                    ],
                    "live_tiles": [list(t) for t in g.live_tiles],
                }
            )

    # branch paths per reduction block_id (mutual-exclusivity oracle).
    branch_paths = {str(k): v for k, v in dev.reduction_block_id_branch_paths().items()}

    # root ids.
    root_ids = list(dev.root_ids)

    accs = [
        {"dim_block_ids": list(a.dim_block_ids), "itemsize": a.itemsize}
        for a in env.config_spec.accumulator_facts
    ]

    out = {
        "n_graphs": len(dev.graphs),
        "root_ids": root_ids,
        "grid_block_ids": grid_ids,
        "original_graph_reductions": [list(x) for x in occurrences],
        "coresidency_group_keys": {
            str(g): sorted(set(b)) for g, b in sorted(groups_by_gid.items())
        },
        "stored_groups": stored_groups,
        "branch_paths_by_bid": branch_paths,
        "accumulators": accs,
        "graphs": graphs_out,
        "fired": list(env.config_spec.autotuner_heuristics),
    }
    return out


def _print(name: str, res: dict) -> None:
    if "error" in res:
        print(f"=== {name}: ERROR {res['error']} ===")
        return
    print(
        f"=== {name}  n_graphs={res['n_graphs']} roots={res['root_ids']} "
        f"grid={res['grid_block_ids']} fired={res['fired']} ==="
    )
    print(f"    original_graph_reductions={res['original_graph_reductions']}")
    print(f"    group_keys={res['coresidency_group_keys']}")
    for sg in res.get("stored_groups", []):
        print(
            f"    >> GROUP gid={sg['graph_id']} bids={sg['descriptor_bids']} "
            f"live_tiles={sg['live_tiles']}"
        )
    if res["branch_paths_by_bid"]:
        print(f"    branch_paths_by_bid={res['branch_paths_by_bid']}")
    if res["accumulators"]:
        print(f"    acc={res['accumulators']}")
    for g in res["graphs"]:
        tag = g["type"]
        edges = ""
        if g["if_edges"]:
            edges += " if=" + ",".join(
                f"(if{e['if_gid']}/else{e['else_gid']})" for e in g["if_edges"]
            )
        if g["loop_edges"]:
            edges += f" loop->{g['loop_edges']}"
        if g["loop_block_ids"]:
            edges += f" block_ids={g['loop_block_ids']}"
        orig = (
            f" orig_gid={g['original_graph_id']}"
            if g["original_graph_id"] is not None
            else ""
        )
        print(
            f"    g{g['graph_id']:<2} {tag:<22}{orig} rbids={g['reduction_block_ids']}"
            f"{edges}"
        )
        print(f"        live_tiles(D1)={g['live_tiles']}  by_axis={g['live_by_axis']}")
        pk = g.get("peaks", {})
        if pk.get("D2_rank") != pk.get("D1_count"):
            print(f"        D2_rank ={pk.get('D2_rank')}")
        if pk.get("D3_union") != pk.get("D1_count"):
            print(f"        D3_union={pk.get('D3_union')}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--probes", action="store_true")
    p.add_argument("--probe-slug", default="")
    p.add_argument("--kernel", default="", help="curriculum kernel name(s), comma list")
    p.add_argument(
        "--rec",
        default="",
        help="recorder corpus:kernels, e.g. transfer:grpo or vllm:per_token_group_fp8_quant",
    )
    p.add_argument("--out", default="")
    args = p.parse_args()
    print(f"helion={helion.__file__}\n", flush=True)

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

    targets = []
    if args.probes:
        targets += [("probe", s) for s in _PROBES]
    if args.probe_slug:
        targets += [("probe", s) for s in args.probe_slug.split(",")]
    if args.kernel:
        targets += [("curriculum", k) for k in args.kernel.split(",")]

    results = {}
    for kind, name in targets:
        try:
            fn, kargs = (
                _probe_fn_args(name) if kind == "probe" else _curriculum_fn_args(name)
            )
            res = introspect(fn, kargs)
        except Exception as e:
            import traceback

            res = {"error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}
        results[name] = res
        _print(name, res)

    if args.rec:
        corpus, _, kstr = args.rec.partition(":")
        kernels = set(kstr.split(",")) if kstr else None
        for label, fn, kargs in _recorder_targets(corpus, kernels):
            try:
                res = introspect(fn, kargs)
            except Exception as e:
                import traceback

                res = {
                    "error": f"{type(e).__name__}: {e}",
                    "tb": traceback.format_exc(),
                }
            results[label] = res
            _print(label, res)
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
