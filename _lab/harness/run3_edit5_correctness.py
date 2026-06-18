"""RUN-3 EDIT#5 — CORRECTNESS GATE on the LIVE seed (configs=[seed], own fp32 ref).

configs=[seed] bypasses the autotuner's accuracy check, so we run our OWN check vs the
operator's fp32 reference at its tolerance (rtol=1e-3, atol=1e-4 — NOT loosened). Confirms
the post-EDIT#5 jsd seed (R_BLOCK=2048) is NUMERICALLY correct on representative shapes:
narrow-V, wide-V, large-M (the new shapes the cap flips), + a kl_div sanity (byte-identical,
nro=1 -> R_BLOCK stays 4096). Executes the kernel on the GPU (NO do_bench / no timing) — needs
the GPU token but does not enter the timing queue as an A/B.

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_edit5_correctness.py
"""

from __future__ import annotations

import os
import sys

import torch

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import helion  # noqa: E402

from run2_measure_g import KERNELS, check_correct, get_seed  # noqa: E402

# narrow-V + wide-V + large-M jsd (the cap flips R_BLOCK 4096->2048 on all) + kl_div sanity.
CASES = [
    ("jsd", 8192, 30522),    # narrow-V (the headline 1.21 gap)
    ("jsd", 8192, 128256),   # wide-V (the hub's named probe)
    ("jsd", 4096, 151936),   # wide-V (hub named)
    ("jsd", 16384, 32000),   # large-M (hub named)
    ("jsd", 2048, 256000),   # widest-V
    ("kl_div", 8192, 30522), # nro=1 -> byte-identical (R_BLOCK 4096); sanity
    ("kl_div", 1024, 256000),
]


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} helion={helion.__file__}\n", flush=True)
    all_ok = True
    for kernel, M, N in CASES:
        fn, builder, _ = KERNELS[kernel]
        args, ref, out_extract = builder(M, N)
        # fp32 assertion (the run-wide invariant).
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_floating_point():
                assert a.dtype == torch.float32, f"{kernel}{(M, N)} non-fp32 {a.dtype}"
        seed_cfg, bound = get_seed(fn, args)
        red = None
        try:
            spec = bound.env.config_spec
            ridx = spec.block_sizes.block_id_to_index(spec.reduction_facts[0].block_id)
            red = seed_cfg.get("block_sizes", [None])[ridx]
        except Exception:  # noqa: BLE001
            red = None
        k = helion.kernel(fn.fn, configs=[helion.Config(**seed_cfg)])
        out = out_extract(k(*args))
        ok, err = check_correct(out, ref)
        all_ok = all_ok and ok
        print(
            f"{kernel:>9}({M:>6},{N:>7})  R_BLOCK={red}  "
            f"maxerr={err:.2e}  -> {'OK' if ok else '*** BAD ***'}",
            flush=True,
        )
    print(f"\n{'ALL CORRECT' if all_ok else '*** CORRECTNESS FAILURE ***'}", flush=True)
    assert all_ok, "EDIT#5 correctness gate FAILED"


if __name__ == "__main__":
    main()
