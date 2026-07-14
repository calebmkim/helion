"""Phase-A compile probe: compile ONE arm's config for ONE (kernel, shape) in an
ISOLATED process, so a ptxas hang or OOM on the ~[16,16,16] helion_default is caught
by the parent's timeout+killpg instead of wedging the whole sweep (DECISIONS.md D3).

Usage (invoked by sweep.py via setsid):
    python -m mmperf.compile_probe <kernel> <arm> <shape_json> [--fast-accum 0|1]

Prints one JSON line to stdout: {"status": "ok"|"oom"|"compile_fail", "cfg": {...},
"error": "..."}. The parent treats non-zero exit / killed as status="timeout".
The compiled cubin is written to the SHARED triton/inductor disk cache, so Phase B
(the timing process) gets a cache hit and never re-invokes ptxas.
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from mmperf import common
from mmperf import kernels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel")
    ap.add_argument("arm", choices=["seed", "helion_default", "tc_max_autotune"])
    ap.add_argument("shape_json")
    ap.add_argument("--fast-accum", type=int, default=1)
    args = ap.parse_args()

    common.set_fairness_locks()
    spec = json.loads(args.shape_json)
    kdef = kernels.KERNELS[args.kernel]

    result: dict = {"status": "ok", "cfg": None, "error": None}
    try:
        arg_tuple, _ref, _meta = kdef["make_inputs"](spec)

        if args.arm == "tc_max_autotune":
            # compile + autotune the torch.compile op for this shape (warms cache)
            if args.kernel == "fp8_gemm":
                fn = kdef["tc"](arg_tuple, fast_accum=bool(args.fast_accum))
            else:
                fn = kdef["tc"](arg_tuple)
            fn()
            torch.cuda.synchronize()
            result["status"] = "ok"
        else:
            bound = kdef["bind"](arg_tuple)
            cfgs = common.extract_configs(bound)
            if args.arm == "seed":
                cfg = cfgs["seed_cfg"]
            else:
                cfg = cfgs["default_cfg"]
            if cfg is None:
                result["status"] = "compile_fail"
                result["error"] = f"{args.arm} config is None"
            else:
                result["cfg"] = common.cfg_summary(cfg)
                result["default_source"] = cfgs.get("default_source")
                compiled = bound.compile_config(cfg)
                out = compiled(*arg_tuple)
                torch.cuda.synchronize()
                # cheap finiteness check (full acc gate happens in timing process)
                if not torch.isfinite(out.float()).all():
                    result["status"] = "acc_fail"
                    result["error"] = "non-finite output"
    except torch.cuda.OutOfMemoryError as e:
        result["status"] = "oom"
        result["error"] = str(e)[:300]
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        result["status"] = "oom" if "out of resource" in msg or "out of memory" in msg else "compile_fail"
        result["error"] = f"{type(e).__name__}: {msg[:300]}"

    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
