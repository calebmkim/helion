"""Faithfulness check for `_apply_reread`'s `not f.reductions_fed` clause.

Compares the CURRENT gate against a DIRECTLY-FAITHFUL "two sibling passes over the row" detector:
the reduction tensor is LOADED in >=2 distinct loop-body graphs that are SIBLINGS (neither is an
ancestor of the other in the loop-nest tree) -> the row is physically re-read in a separate pass ->
L2 round-trip. Dumps, per kernel:
  - CURRENT: any load of a reduction-tensor with (stores_fed and not reductions_fed)
  - SIBLING: reduction tensor loaded in >=2 non-nested graphs (+ which graphs, + nesting)
  - the loop-nest tree (loop->child edges) and per-load graph/feeds classification

Usage (from /tmp, FOREGROUND, serial):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/redesign/probe_sibling_reread.py [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
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

import ground_live_tiles as G  # noqa: E402


def _loop_child_edges(dev):
    """child[gid] = set of graph_ids gid drives via a for/reduction loop (parent->child)."""
    from helion.language import _tracing_ops

    child: dict[int, set[int]] = {}
    for gi in dev.graphs:
        edges: set[int] = set()
        for node in gi.graph.nodes:
            if node.op != "call_function":
                continue
            if (
                _tracing_ops.is_for_loop_target(node.target)
                and node.args
                and isinstance(node.args[0], int)
            ):
                edges.add(node.args[0])
        if edges:
            child[gi.graph_id] = edges
    return child


def _ancestors(child):
    """anc[gid] = set of all transitive ancestors of gid (who drives it, transitively)."""
    parent: dict[int, set[int]] = {}
    for p, kids in child.items():
        for k in kids:
            parent.setdefault(k, set()).add(p)
    anc: dict[int, set[int]] = {}

    def walk(g, seen):
        for p in parent.get(g, ()):
            if p not in seen:
                seen.add(p)
                walk(p, seen)
        return seen

    all_g = set(parent) | {k for ks in child.values() for k in ks} | set(child)
    for g in all_g:
        anc[g] = walk(g, set())
    return anc


def analyze(fn, args, label):
    from helion._compiler.autotuner_heuristics.triton import (
        _primary_descriptor_selected,
    )

    bound = fn.bind(args)
    env = bound.env
    dev = bound.host_function.device_ir
    env.__enter__()
    bound.host_function.__enter__()
    try:
        spec = env.config_spec
        pd = _primary_descriptor_selected(env)
        if pd is None:
            return {"label": label, "skip": "pd=None"}
        facts = spec.memory_op_facts
        red_tensors = {
            f.tensor_name
            for f in facts
            if f.kind == "load"
            and f.tensor_name is not None
            and any(ax == pd.block_id for ax, _ in f.reductions_fed)
        }
        # CURRENT gate
        current = any(
            f.kind == "load"
            and f.tensor_name in red_tensors
            and f.stores_fed
            and not f.reductions_fed
            for f in facts
        )
        # SIBLING gate: for each reduction tensor, the set of graphs it is LOADED in; is there a
        # pair of those graphs that are NOT in an ancestor/descendant relation (siblings)?
        child = _loop_child_edges(dev)
        anc = _ancestors(child)

        def nested(a, b):
            return b in anc.get(a, ()) or a in anc.get(b, ())

        per_tensor_graphs: dict[str, set[int]] = {}
        for f in facts:
            if f.kind == "load" and f.tensor_name in red_tensors:
                per_tensor_graphs.setdefault(f.tensor_name, set()).add(f.graph_id)
        sibling = False
        sibling_detail = {}
        for t, gids in per_tensor_graphs.items():
            gl = sorted(gids)
            has_sib = any(
                not nested(gl[i], gl[j])
                for i in range(len(gl))
                for j in range(i + 1, len(gl))
            )
            sibling_detail[t] = {"graphs": gl, "has_sibling_pair": has_sib}
            if has_sib:
                sibling = True
        # per-load classification for the record
        loads = [
            {
                "tensor": f.tensor_name,
                "graph": f.graph_id,
                "reductions_fed": list(f.reductions_fed),
                "stores_fed": bool(f.stores_fed),
            }
            for f in facts
            if f.kind == "load" and f.tensor_name in red_tensors
        ]
        return {
            "label": label,
            "primary_bid": pd.block_id,
            "red_tensors": sorted(t for t in red_tensors if t),
            "CURRENT_apply_reread": current,
            "SIBLING_two_passes": sibling,
            "AGREE": current == sibling,
            "loop_child_edges": {str(k): sorted(v) for k, v in child.items()},
            "per_tensor": sibling_detail,
            "loads": loads,
        }
    finally:
        bound.host_function.__exit__(None, None, None)
        env.__exit__(None, None, None)
        del bound
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="")
    args = p.parse_args()
    print(f"helion={helion.__file__}\n", flush=True)
    out = {}
    curr = [
        "sum",
        "long_sum",
        "softmax",
        "kl_div",
        "jsd",
        "rms_norm",
        "layer_norm",
        "welford",
        "cross_entropy",
    ]
    recs = [
        "transfer:cross_entropy_ls_zloss",
        "transfer:gated_rmsnorm",
        "transfer:fused_add_rmsnorm",
        "transfer:fused_add_layernorm",
        "transfer:scaled_masked_softmax",
        "transfer:dynamic_quant",
        "transfer:fused_linear_jsd",
        "mreduction:layer_norm_bwd",
        "mreduction:rms_norm_bwd",
        "mreduction:bias_grad_bwd",
        "mreduction:group_norm_bwd",
        "mreduction:instance_norm_bwd",
        "mreduction:dyt_bwd",
        "vllm:per_token_group_fp8_quant",
        "vllm:rms_norm_per_block_quant",
    ]
    for k in curr:
        try:
            r = analyze(*G._curriculum_fn_args(k), f"curriculum/{k}")
        except Exception as e:  # noqa: BLE001
            r = {"label": f"curriculum/{k}", "error": f"{type(e).__name__}: {e}"}
        out[r["label"]] = r
    for rec in recs:
        c, _, kn = rec.partition(":")
        try:
            for lbl, fn, a in list(G._recorder_targets(c, {kn}))[:1]:
                r = analyze(fn, a, lbl.rsplit("/", 2)[0])
                out[r["label"]] = r
        except Exception as e:  # noqa: BLE001
            out[rec] = {"label": rec, "error": f"{type(e).__name__}: {e}"}
    for lbl, r in out.items():
        if "error" in r or "skip" in r:
            print(f"{lbl:42} {r.get('error') or r.get('skip')}")
            continue
        flag = "" if r["AGREE"] else "  <<<< DISAGREE"
        print(
            f"{lbl:42} CURRENT={int(r['CURRENT_apply_reread'])} "
            f"SIBLING={int(r['SIBLING_two_passes'])}{flag}"
        )
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
