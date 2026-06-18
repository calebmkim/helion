"""TASK 1 — record the config the reduction heuristic CHOOSES for every shape.

Compile-time only: bind the kernel for each (kernel, M, N) across train/val/test
(all 9 kernels, welford included), run the live seed heuristic, and record:

  - raw_seed        : compiler_seed_configs(env, device_ir)[0] as a dict (the
                      heuristic's literal output, before normalize()).
  - normalized_cfg  : the SAME config after config_spec.normalize() — i.e. the
                      config that would ACTUALLY run with configs=[seed]
                      (normalize forces persistent when value>=size_hint, caps
                      by size hint, expands scalar reduction_loop to a list, …).
  - reduction_fact  : the ReductionFact[0] fields the heuristic keyed on (the
                      "why" behind the choice).
  - classification  : T1 (rdim block_id in reduction_loops) / T2 (reduction axis
                      is a block_sizes entry) / out_of_scope.
  - heuristics_fired: env.config_spec.autotuner_heuristics (which heuristic(s)
                      emitted the seed).

NO Triton codegen, NO autotune, NO do_bench, NO torch.compile — that is task 2.
Binding only allocates the input tensors (freed between shapes). Single serial
foreground process (one GPU; allocation-only, but we keep the serial discipline).

Run from /tmp:
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_task1_seed_configs.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "prompts"))
for _d in (_HARNESS_DIR, _PROMPTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch  # noqa: E402

import helion  # noqa: E402

# Reuse the established, proven per-kernel arg builders + the kernel objects.
from run2_measure_g import KERNELS  # noqa: E402

# The single source of truth for the curriculum splits.
from shapes_v3_draft import SHAPES  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

# Machine-portable wiring assert (same spirit as run2_measure_g): helion must
# resolve to THIS worktree, not the original editable install.
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), (
    f"helion ({helion.__file__}) not under harness worktree ({_WT_ROOT}); "
    "set PYTHONPATH to this worktree."
)

LOG_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "logs", "run3"))
os.makedirs(LOG_DIR, exist_ok=True)
OUT = os.path.join(LOG_DIR, "task1_seed_configs.json")

SPLITS = ("train", "val", "test")
# All 9 curriculum kernels (welford INCLUDED for this task, per the request).
KERNEL_ORDER = [
    "rms_norm", "layer_norm", "softmax", "welford", "sum", "long_sum",
    "cross_entropy", "kl_div", "jsd",
]


def _jsonify(v):
    """Best-effort JSON-safe rendering of arbitrary config / fact values."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonify(x) for k, x in v.items()}
    return repr(v)


def _classify(spec, fact) -> str:
    """T1 (rdim block_id in reduction_loops) / T2 (reduction axis is a
    block_sizes entry) / out_of_scope — per the work-order's static rule."""
    try:
        if getattr(spec, "matmul_facts", None):
            return "gemm"
        rl_ids = set(spec.reduction_loops.valid_block_ids())
        if fact.block_id in rl_ids:
            return "T1"
        bs_ids = set(spec.block_sizes.valid_block_ids())
        if fact.block_id in bs_ids:
            return "T2"
    except Exception as e:  # noqa: BLE001
        return f"unknown ({type(e).__name__})"
    return "out_of_scope"


def record_one(kernel_name, M, N) -> dict:
    fn, builder, _tc_ref = KERNELS[kernel_name]
    args, _ref, _out = builder(M, N)

    bound = fn.bind(args)
    env = bound.env
    device_ir = bound.host_function.device_ir
    spec = env.config_spec

    seeds = compiler_seed_configs(env, device_ir)
    fired = list(getattr(spec, "autotuner_heuristics", []))

    rec: dict = {
        "kernel": kernel_name,
        "M": M,
        "N": N,
        "heuristics_fired": fired,
        "n_reduction_facts": len(getattr(spec, "reduction_facts", [])),
        "n_seeds": len(seeds),
    }

    if not seeds:
        rec["raw_seed"] = None
        rec["normalized_cfg"] = None
        rec["note"] = "heuristic emitted no seed (not eligible / no rdim)"
    else:
        raw = dict(seeds[0])
        rec["raw_seed"] = _jsonify(raw)
        norm = dict(raw)
        try:
            spec.normalize(norm)
            rec["normalized_cfg"] = _jsonify(norm)
        except Exception as e:  # noqa: BLE001
            rec["normalized_cfg"] = None
            rec["normalize_error"] = f"{type(e).__name__}: {e}"

    facts = getattr(spec, "reduction_facts", [])
    if facts:
        fact = facts[0]
        rec["reduction_fact"] = _jsonify(fact._asdict())
        rec["classification"] = _classify(spec, fact)
    else:
        rec["reduction_fact"] = None
        rec["classification"] = "no_reduction_fact"

    # free GPU memory before the next (possibly multi-GB) shape
    del args, bound, env, device_ir, spec, seeds
    torch.cuda.empty_cache()
    return rec


def main():
    print(f"helion={helion.__file__}", flush=True)
    print(f"out={OUT}\n", flush=True)

    rows = []
    n_ok = n_err = 0
    for kernel_name in KERNEL_ORDER:
        splits = SHAPES[kernel_name]
        for split in SPLITS:
            for (M, N) in splits[split]:
                tag = f"{kernel_name:14s} {split:5s} ({M},{N})"
                try:
                    rec = record_one(kernel_name, M, N)
                except Exception as e:  # noqa: BLE001
                    n_err += 1
                    print(f"[ERR ] {tag}: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    rows.append({"kernel": kernel_name, "split": split,
                                 "M": M, "N": N,
                                 "error": f"{type(e).__name__}: {e}"})
                    torch.cuda.empty_cache()
                    continue
                rec["split"] = split
                rows.append(rec)
                n_ok += 1
                cls = rec.get("classification", "?")
                norm = rec.get("normalized_cfg") or {}
                rl = norm.get("reduction_loops")
                bs = norm.get("block_sizes")
                nw = norm.get("num_warps")
                ns = norm.get("num_stages")
                pid = norm.get("pid_type")
                print(
                    f"[ OK ] {tag}  cls={cls:11s} "
                    f"bs={bs} rl={rl} warps={nw} stages={ns} pid={pid}",
                    flush=True,
                )
                # checkpoint after every kernel so a crash leaves a partial file
                json.dump({"rows": rows}, open(OUT, "w"), indent=1)

    json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print(f"\n=== DONE: {n_ok} recorded, {n_err} errored ===", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
