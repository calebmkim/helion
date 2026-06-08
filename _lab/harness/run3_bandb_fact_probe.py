"""RUN-3 EDIT#5 — Band-B FACT PROBE (compile-only, NO autotune, NO do_bench).

Resolves the EDIT#5 gate-key crux: the heuristic's Band-B cap (triton.py ~787) is
``bandb_cap = BANDB_R_BLOCK_BYTES // itemsize`` — it does NOT multiply by the number
of live carried accumulators. The fact docstring says jsd carries 2 ([M,R]) accumulators
(intermediate_loss + intermediate_dX) and kl_div carries 1 (loss_sum). If
``num_tiled_accumulators`` is genuinely 2 (jsd) vs 1 (kl_div), the PRINCIPLED fix is to
divide the cap by num_tiled_accumulators (true live-state footprint = n_acc*R_BLOCK*
itemsize) — NOT a jsd fence, and it predicts jsd-wants-2048 / kl_div-wants-4096 from a
SINGLE byte budget. A prior decomp claimed BOTH were 2 (which would mean the fact
mis-counts). This probe dumps the EXACT fact for several Band-B shapes to settle it.

Pure bind() + fact read — no timing, does NOT enter the GPU timing queue.

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_bandb_fact_probe.py
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

# Band-B shapes (narrow + wide V) for jsd + kl_div; add softmax (T2 num_tiled_accum==0)
# and welford (Band-C) as CONTRAST so the dump shows the fact discriminates correctly.
CASES = [
    ("jsd", 8192, 30522),
    ("jsd", 8192, 32000),
    ("jsd", 2048, 256000),
    ("kl_div", 8192, 30522),
    ("kl_div", 8192, 32000),
    ("kl_div", 1024, 256000),
    ("softmax", 1024, 65536),
    ("welford", 2048, 7168),
]

FIELDS = [
    "size_hint", "static_rnumel", "itemsize", "num_load", "num_store",
    "num_reduction_ops", "num_tiled_accumulators", "is_structured_combine",
    "row_reread",
]


def probe(kernel, M, N):
    fn, builder, _ = KERNELS[kernel]
    args, _, _ = builder(M, N)
    k = helion.kernel(fn.fn)
    b = k.bind(args)
    spec = b.env.config_spec
    facts = spec.reduction_facts
    if not facts:
        print(f"{kernel}({M},{N}): NO reduction_facts (matmul/ineligible?)", flush=True)
        return
    f = facts[0]
    vals = {name: getattr(f, name) for name in FIELDS}
    n_acc = vals["num_tiled_accumulators"]
    itemsize = vals["itemsize"]
    # The current cap vs a num_tiled_accumulators-aware cap (the proposed EDIT#5 fix).
    BANDB = 16384
    cur_cap = max(1, BANDB // max(1, itemsize))
    aware_cap = max(1, BANDB // (max(1, itemsize) * max(1, n_acc)))
    print(f"\n=== {kernel}({M},{N}) ===  n_facts={len(facts)}", flush=True)
    for name in FIELDS:
        print(f"    {name:>24} = {vals[name]}", flush=True)
    print(f"    {'cur_bandb_cap(elems)':>24} = {cur_cap}   "
          f"naccaware_cap(elems) = {aware_cap}", flush=True)


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    for kernel, M, N in CASES:
        try:
            probe(kernel, M, N)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {kernel}({M},{N}): {type(e).__name__}: {e}"[:200], flush=True)


if __name__ == "__main__":
    main()
