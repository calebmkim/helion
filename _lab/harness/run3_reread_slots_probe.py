"""RUN-3 reread_buffer_slots TIGHTENING verification (no do_bench; fact-only).

Reads the ACTUAL ReductionFact built by `_compute_reread_provenance` for every
kernel (bind only -- fake-tensor, no kernel launch, no autotune, so it does NOT
contend for the GPU timing queue), and prints:
  - fact.row_reread            (the persist-cap gate property)
  - fact.reread_buffer_slots   (the EVICTION provenance, hub-tightened)
  - the host-buffer name(s) loaded at each reread slot (to CONFIRM the slot points
    at the REDUCTION-INPUT ROW, not a coincidental >=2-loop-graph broadcast operand)
  - the seed's emitted load_eviction_policies (what the heuristic actually sets)

EXPECT (production, pre-tightening, verified 9/9):
  sum/long_sum=False slots=()        kl_div/jsd=False slots=()
  rms_norm/layer_norm/softmax=True  (persistent at these shapes -> eviction NOT
                                     emitted by T1 'not persistent' gate; slots may
                                     be () because the persistent row isn't HBM-re-read)
  cross_entropy=True  slots -> logits  (the wide looped shape: slots=(2,...) the
                                     logits loads, NOT the labels/target gather)
  welford=True        slots -> x      (Band-C: x re-read combine+apply)

The TIGHTENING must NOT change any production slot set (CE's re-read buffer IS logits
= the reduction input; welford's IS x). It only rejects an adversarial reused-broadcast
operand. So: every kernel's (row_reread, slots) here must match the pre-tightening 9/9.

Invocation (run from /tmp; NO GPU token needed -- bind-only, no do_bench):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_reread_slots_probe.py
"""

from __future__ import annotations

import os
import sys

import torch

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import helion  # noqa: E402

from run2_measure_g import KERNELS, get_seed  # noqa: E402
from helion._compiler.device_ir import _fx_trace_tensor_arg_rw_names  # noqa: E402
from helion.language.memory_ops import load as _load_op  # noqa: E402

# (kernel, M, N): pick a shape per kernel that exercises the WIDE/looped path where
# reread_buffer_slots actually MATTERS (CE/softmax wide so they're looped; welford
# wide so the apply loops; sum/long_sum/kl_div/jsd any). Keep small M for cheap bind.
SHAPES = {
    "sum": (256, 8192),
    "long_sum": (16, 2097152),
    "rms_norm": (1, 131072),       # wide -> looped robustness regime
    "layer_norm": (1, 131072),
    "softmax": (512, 131072),      # 512KiB -> looped
    "cross_entropy": (2048, 128256),  # wide V -> looped (where eviction wins)
    "kl_div": (1024, 4096),
    "jsd": (1024, 4096),
    "welford": (8192, 8192),       # wide -> looped apply
}


def _ordered_load_host_names(bound) -> list[list[str]]:
    """The host-buffer name(s) for each hl.load, in codegen-emission order — the
    SAME order reread_buffer_slots indexes (slot i = i-th hl.load)."""
    host = bound.host_function
    device_ir = host.device_ir
    out: list[list[str]] = []
    for gi in device_ir.graphs:
        for node in gi.graph.find_nodes(
            op="call_function", target=_load_op, sort=False
        ):
            out.append(list(_fx_trace_tensor_arg_rw_names(host, node.args[0])))
    return out


def probe(kernel: str, M: int, N: int):
    fn, builder, _ = KERNELS[kernel]
    args, _, _ = builder(M, N)
    bound = fn.bind(args)
    spec = bound.env.config_spec
    fact = spec.reduction_facts[0]
    seed_cfg, _ = get_seed(fn, args)
    evict = seed_cfg.get("load_eviction_policies")
    load_names = _ordered_load_host_names(bound)
    slots = fact.reread_buffer_slots
    slot_names = {
        i: (load_names[i] if i < len(load_names) else "??") for i in slots
    }
    print(
        f"\n=== {kernel}({M},{N}) ===  row_reread={fact.row_reread}  "
        f"num_load={fact.num_load}  reread_buffer_slots={slots}",
        flush=True,
    )
    print(f"  slot->host-buffer (the re-read ROW): {slot_names}", flush=True)
    print(f"  all load slots -> names: "
          f"{dict(enumerate(load_names))}", flush=True)
    print(f"  seed load_eviction_policies = {evict}", flush=True)
    return {
        "kernel": kernel,
        "row_reread": fact.row_reread,
        "reread_buffer_slots": list(slots),
        "slot_names": {str(k): v for k, v in slot_names.items()},
        "evict": evict,
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} helion={helion.__file__} (bind-only, no do_bench)", flush=True)
    for k, (M, N) in SHAPES.items():
        try:
            probe(k, M, N)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {k}: {type(e).__name__}: {e}"[:300], flush=True)


if __name__ == "__main__":
    main()
