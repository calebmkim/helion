"""RUN-3 EDIT#5-v2 — CARRIED-[M,R]-ACCUMULATOR probe (compile-only, NO autotune, NO do_bench).

fact-integrity FAILed the Band-B divisor for keying on num_reduction_ops (a lucky-on-curriculum
proxy: nro == carried-accum count ONLY under jsd/kl_div's 1:1 reduction<->accum coincidence).
The FAITHFUL fact the gate handed back = count of [M_BLOCK, R_BLOCK] 2D accumulators CARRIED
ACROSS the inner hl.tile loop (loop-carried 2D outputs of the inner reduction ForLoop), EXCLUDING
in-loop scratch (kl_div's kl_loss dies each iter; jsd's _phi/_phi_1 are carried).

This probe DUMPS, for jsd + kl_div (+ softmax/welford contrast), every [M,R] zeros/full
accumulator-create op and WHICH graph it lives in (root vs ForLoopGraphInfo body), plus each
ForLoopGraphInfo's carry structure (node_args / placeholders), to find the structural discriminator:
  carried accumulator = created in ROOT graph, threaded into the ForLoop as a node_arg (placeholder)
  in-loop scratch     = created INSIDE the ForLoop body graph
Target faithful values: jsd=2, kl_div=1.

Pure bind() + graph walk — NO timing, does NOT enter the GPU timing queue.

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_carried_accum_probe.py
"""

from __future__ import annotations

import os
import sys

import torch

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import helion  # noqa: E402

from run2_measure_g import KERNELS  # noqa: E402

CASES = [
    ("jsd", 8192, 30522),       # target carried=2
    ("kl_div", 8192, 30522),    # target carried=1
    ("softmax", 1024, 65536),   # contrast: T2 scalar/row-acc, n_tiled=0
]


def _accum_ops_and_loops(bound, red_block_id, size_hint):
    from helion._compiler.device_ir import ForLoopGraphInfo, ReductionLoopGraphInfo
    from helion.language.creation_ops import full as _full_op
    from helion.language.creation_ops import zeros as _zeros_op

    device_ir = bound.host_function.device_ir
    accum_targets = (_full_op, _zeros_op)

    def last_is_red(val):
        if not (isinstance(val, torch.Tensor) and val.ndim >= 2):
            return False
        last = val.shape[-1]
        return (isinstance(last, int) and last == size_hint) or (
            not isinstance(last, int)
        )

    print(f"    --- graphs ({len(device_ir.graphs)}) ---", flush=True)
    accum_sites = []  # (graph_id, graph_kind, node_name, shape)
    forloops = []  # (graph_id, kind, block_ids, n_node_args, n_placeholders)
    for gid, gi in enumerate(device_ir.graphs):
        kind = type(gi).__name__
        is_forloop = isinstance(gi, (ForLoopGraphInfo, ReductionLoopGraphInfo))
        if is_forloop:
            nargs = len(getattr(gi, "node_args", []) or [])
            nph = len(list(gi.graph.find_nodes(op="placeholder")))
            bids = getattr(gi, "block_ids", None)
            forloops.append((gid, kind, bids, nargs, nph))
        for node in gi.graph.nodes:
            if node.op == "call_function" and node.target in accum_targets:
                val = node.meta.get("val")
                if last_is_red(val):
                    shp = tuple(val.shape) if isinstance(val, torch.Tensor) else None
                    accum_sites.append((gid, kind, node.name, shp))
    for gid, gi in enumerate(device_ir.graphs):
        kind = type(gi).__name__
        print(f"      graph[{gid}] {kind}", flush=True)
    print(f"    --- [M,R] accumulator-create ops (zeros/full, last==reduction) ---",
          flush=True)
    for gid, kind, name, shp in accum_sites:
        print(f"      graph[{gid}]({kind})  {name}  shape={shp}", flush=True)
    print(f"    --- ForLoop/ReductionLoop graphs (carry structure) ---", flush=True)
    for gid, kind, bids, nargs, nph in forloops:
        print(f"      graph[{gid}]({kind}) block_ids={bids} "
              f"node_args={nargs} placeholders={nph}", flush=True)
    return accum_sites, forloops


def probe(kernel, M, N):
    from helion._compiler.device_ir import ForLoopGraphInfo, ReductionLoopGraphInfo

    fn, builder, _ = KERNELS[kernel]
    args, _, _ = builder(M, N)
    bound = fn.bind(args)
    spec = bound.env.config_spec
    facts = spec.reduction_facts
    if not facts:
        print(f"\n=== {kernel}({M},{N}) ===  NO reduction_facts", flush=True)
        return
    f = facts[0]
    print(f"\n=== {kernel}({M},{N}) ===  "
          f"n_tiled_accum(shipped)={f.num_tiled_accumulators} "
          f"nro={f.num_reduction_ops} size_hint={f.size_hint} block_id={f.block_id}",
          flush=True)
    accum_sites, forloops = _accum_ops_and_loops(bound, f.block_id, f.size_hint)

    # HYPOTHESIS: carried accumulators = [M,R] accum-create ops in the ROOT graph
    # (graph[0] / non-ForLoop), threaded into the inner ForLoop; in-loop scratch =
    # those created INSIDE a ForLoop body graph. Count each way to find the rule.
    device_ir = bound.host_function.device_ir
    forloop_gids = {
        gid for gid, gi in enumerate(device_ir.graphs)
        if isinstance(gi, (ForLoopGraphInfo, ReductionLoopGraphInfo))
    }
    in_root = [s for s in accum_sites if s[0] not in forloop_gids]
    in_loop = [s for s in accum_sites if s[0] in forloop_gids]
    print(f"    => total [M,R] accum-create ops = {len(accum_sites)} "
          f"(shipped num_tiled_accumulators)", flush=True)
    print(f"       in ROOT/non-forloop graph = {len(in_root)} "
          f"{[s[2] for s in in_root]}", flush=True)
    print(f"       in FORLOOP-body graph     = {len(in_loop)} "
          f"{[s[2] for s in in_loop]}", flush=True)


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__} "
          f"(bind-only, no do_bench)", flush=True)
    for kernel, M, N in CASES:
        try:
            probe(kernel, M, N)
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"[ERR] {kernel}({M},{N}): {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
