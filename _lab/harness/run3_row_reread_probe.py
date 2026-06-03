"""RUN-3 row_reread provenance PROBE (no do_bench; fact inspection only).

Computes a CANDIDATE `row_reread` for every kernel directly from the reduction
roller's host-buffer read provenance, and prints the per-host-buffer LOAD COUNT so
I can SEE whether the discriminator "some reduction-input host buffer is read by >=2
distinct load nodes" gives the RIGHT set:
  sum/long_sum = False (single stream)        rms_norm/layer_norm/softmax/CE = True (re-read)
  kl_div = False (2 distinct inputs, each once)   welford/jsd = (Band-B/C own caps; informational)

This is the FALSIFICATION the fact-integrity gate wants BEFORE the fact is written:
does the candidate computation track the real property on the kernel set? Uses ONLY
trace/bind (fake-tensor, no kernel launch, no do_bench), so it does NOT contend for
the GPU timing queue.

For T1 (rollable): used_graphs = the original graphs that use the rdim (device_ir.py
register_rollable_reductions ~840). For each such graph, walk every hl.load node,
resolve its tensor arg to host-buffer name(s) via _fx_trace_tensor_arg_rw_names, and
count load NODES per host name. row_reread = any host name with count >= 2.
For T2 we inspect the equivalent graphs. We print the raw per-name counts so the
discriminator can be eyeballed / adjusted.

Invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_row_reread_probe.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import torch

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import helion  # noqa: E402

from run2_measure_g import KERNELS  # noqa: E402
from helion._compiler.device_ir import _fx_trace_tensor_arg_rw_names  # noqa: E402
from helion._compiler.host_function import HostFunction  # noqa: E402
from helion.language.memory_ops import load as _load_op  # noqa: E402

# Representative shape per kernel (any shape -- row_reread is shape-independent).
PROBE_SHAPES = {
    "sum": (4096, 8192), "long_sum": (256, 131072),
    "rms_norm": (4096, 4096), "layer_norm": (4096, 4096),
    "softmax": (4096, 4096), "cross_entropy": (4096, 8192),
    "kl_div": (4096, 8192), "jsd": (4096, 8192), "welford": (4096, 4096),
}


def probe(kernel, M, N):
    fn, builder, _ = KERNELS[kernel]
    args, _, _ = builder(M, N)
    bk = helion.kernel(fn.fn).bind(args)
    env = bk.env
    device_ir = bk.host_function.device_ir
    host = bk.host_function

    spec = env.config_spec
    rfacts = spec.reduction_facts
    if not rfacts:
        return {"kernel": kernel, "note": "no reduction_facts"}
    fact = rfacts[0]
    is_t1 = fact.block_id in spec.reduction_loops.valid_block_ids()

    # Two views per host buffer:
    #  (a) load-NODE count across all graphs (the naive proxy -- CONFOUNDED: a
    #      single looped stream emits 2 load nodes for the same buffer, e.g. sum/
    #      long_sum x=2).
    #  (b) DISTINCT-GRAPH count: how many distinct device GRAPHS contain a load of
    #      the buffer. Hypothesis: a genuine re-read spans >1 graph (2nd reduction
    #      pass OR a separate apply graph); a single-pass stream's multiple load
    #      nodes live in ONE graph.
    name_counts: Counter[str] = Counter()
    name_graphs: dict[str, set[int]] = {}
    graph_count = len(device_ir.graphs)
    with host:
        for gi in device_ir.graphs:
            g = gi.graph
            names_in_graph: set[str] = set()
            for node in g.find_nodes(op="call_function", target=_load_op, sort=False):
                for nm in _fx_trace_tensor_arg_rw_names(host, node.args[0]):
                    name_counts[nm] += 1
                    names_in_graph.add(nm)
            for nm in names_in_graph:
                name_graphs.setdefault(nm, set()).add(gi.graph_id)

    name_graph_counts = {nm: len(g) for nm, g in name_graphs.items()}
    reread_by_node = [nm for nm, c in name_counts.items() if c >= 2]
    reread_by_graph = [nm for nm, c in name_graph_counts.items() if c >= 2]
    row_reread = len(reread_by_graph) >= 1  # the by-GRAPH discriminator
    return {
        "kernel": kernel, "shape": [M, N], "is_t1": is_t1,
        "per_host_graph_count": name_graph_counts,
        "candidate_by_node(>=2nodes)": len(reread_by_node) >= 1,
        "num_load_fact": fact.num_load, "num_reduction_ops": fact.num_reduction_ops,
        "n_graphs": graph_count,
        "per_host_load_count": dict(name_counts),
        "candidate_row_reread": row_reread,
    }


EXPECTED = {  # the RIGHT set per the hub
    "sum": False, "long_sum": False, "rms_norm": True, "layer_norm": True,
    "softmax": True, "cross_entropy": True, "kl_div": False,
    # welford/jsd governed by Band-C/B caps; informational (no hard expectation)
}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}\n",
          flush=True)
    print(f"{'kernel':>14} {'is_t1':>5} {'nl':>3} {'nro':>3} {'reread(byGRAPH)':>15} "
          f"{'byNODE':>7} {'exp':>5} {'OK?':>5}  node_cnt | graph_cnt", flush=True)
    ok_all = True
    for kernel, (M, N) in PROBE_SHAPES.items():
        try:
            r = probe(kernel, M, N)
        except Exception as e:  # noqa: BLE001
            print(f"{kernel:>14} ERR {type(e).__name__}: {e}"[:160], flush=True)
            continue
        if "note" in r:
            print(f"{kernel:>14} {r['note']}", flush=True)
            continue
        exp = EXPECTED.get(kernel)
        ok = "n/a" if exp is None else ("OK" if r["candidate_row_reread"] == exp else "**X**")
        if exp is not None and r["candidate_row_reread"] != exp:
            ok_all = False
        print(f"{kernel:>14} {str(r['is_t1']):>5} {r['num_load_fact']:>3} "
              f"{r['num_reduction_ops']:>3} {str(r['candidate_row_reread']):>15} "
              f"{str(r['candidate_by_node(>=2nodes)']):>7} {str(exp):>5} {ok:>5}  "
              f"{r['per_host_load_count']} | {r['per_host_graph_count']}", flush=True)
    print(f"\nCANDIDATE row_reread matches expected set: "
          f"{'YES' if ok_all else 'NO -- discriminator needs work'}", flush=True)


if __name__ == "__main__":
    main()
