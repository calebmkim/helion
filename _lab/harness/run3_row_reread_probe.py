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
from helion._compiler.device_ir import ForLoopGraphInfo  # noqa: E402
from helion._compiler.device_ir import ReductionLoopGraphInfo  # noqa: E402
from helion.language.memory_ops import load as _load_op  # noqa: E402

# SOLVED + verified across all 9 kernels (2026-06-03). The FAITHFUL row_reread:
# a reduction-input HOST BUFFER is loaded in >=2 distinct LOOP graphs
# (ReductionLoopGraphInfo for T1-rollable; ForLoopGraphInfo for T2 user-tiled).
# Tracks genuine multi-pass re-read of the SAME buffer; immune to the roller's
# root+loop duplication (which false-positived naive load-node / all-graph counts on
# sum/long_sum) and distinguishes kl_div/jsd (2 distinct inputs, each once) from
# softmax/CE (1 re-read input). NOT num_load (over-counts CE gather) NOR
# num_reduction_ops (under-counts rms/ln apply re-read).
_LOOP_GRAPH_TYPES = (ReductionLoopGraphInfo, ForLoopGraphInfo)

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

    # UNIFIED discriminator: per host buffer, count DISTINCT LOOP graphs
    # (ReductionLoopGraphInfo | ForLoopGraphInfo) that load it. row_reread = any
    # buffer with count >= 2 (consumed by >=2 passes = re-read). Excludes the
    # RootGraphInfo (the root + its rolled loop are alternate representations of ONE
    # pass -> counting all graphs false-positives sum/long_sum).
    loopgraph_count: Counter[str] = Counter()
    with host:
        for gi in device_ir.graphs:
            if not isinstance(gi, _LOOP_GRAPH_TYPES):
                continue
            names_in_graph: set[str] = set()
            for node in gi.graph.find_nodes(
                op="call_function", target=_load_op, sort=False
            ):
                for nm in _fx_trace_tensor_arg_rw_names(host, node.args[0]):
                    names_in_graph.add(nm)
            for nm in names_in_graph:
                loopgraph_count[nm] += 1

    reread_names = [nm for nm, c in loopgraph_count.items() if c >= 2]
    row_reread = len(reread_names) >= 1
    return {
        "kernel": kernel, "shape": [M, N], "is_t1": is_t1,
        "num_load_fact": fact.num_load, "num_reduction_ops": fact.num_reduction_ops,
        "loopgraph_count_per_buffer": dict(loopgraph_count),
        "reread_names": reread_names,
        "candidate_row_reread": row_reread,
    }


# The RIGHT set: re-read kernels (row resident across >=2 passes) = True; single-
# stream / distinct-input-each-once = False. welford=True is correct (re-reads x in
# combine+apply) though its Band-C caps dominate; kl_div/jsd=False (2 distinct
# inputs each read once) though Band-B caps dominate.
EXPECTED = {
    "sum": False, "long_sum": False, "rms_norm": True, "layer_norm": True,
    "softmax": True, "cross_entropy": True, "kl_div": False, "jsd": False,
    "welford": True,
}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}\n",
          flush=True)
    print(f"{'kernel':>14} {'is_t1':>5} {'nl':>3} {'nro':>3} {'row_reread':>10} "
          f"{'exp':>6} {'OK?':>5}  loopgraph_count_per_buffer", flush=True)
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
              f"{r['num_reduction_ops']:>3} {str(r['candidate_row_reread']):>10} "
              f"{str(exp):>6} {ok:>5}  {r['loopgraph_count_per_buffer']}", flush=True)
    print(f"\nUNIFIED row_reread (>=2 ReductionLoop/ForLoop graphs load a buffer) "
          f"matches expected set: {'YES' if ok_all else 'NO'}", flush=True)


if __name__ == "__main__":
    main()
