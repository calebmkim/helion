"""TASK D — DUAL-config recorder over all curriculum shapes (train/val/test).

For each (kernel, split, M, N) in the v3 curriculum, bind the kernel ONCE
(reusing run3_task1_seed_configs.record_one's binding path: run2's KERNELS
arg-builders + fn.bind) and record, on the SAME env/device_ir:

  MINE  : compiler_seed_configs(env, device_ir)  -> the live deep heuristic
          (TritonReductionHeuristic, name "triton_reduction_tile") seed.
          raw + normalized (via spec.normalize on a copy). mine_fires = (len>0).

  MAIN  : the PROVEN-FAITHFUL port of main's narrow TritonReductionTileHeuristic
          (main_reduction_heuristic_port, SHA 8d5cc261 / commit ea35dfdd):
          main_is_eligible then main_get_seed_config. raw + normalized.
          main_fires = (config is not None).

          TASK C VERDICT (struct_mine.json vs struct_main.json): for the overlap
          kernels the STRUCTURAL inputs main's heuristic reads (block_sizes,
          reduction_loops, matmul_facts, load_eviction_policies) are IDENTICAL
          across trees. Therefore computing main's config on MY tree's binding is
          faithful to what main emits natively -> clean single-tree recording.

  configs_identical = (mine_norm == main_norm) when BOTH fire.
  classification    = run3_task1_seed_configs._classify (T1/T2/out_of_scope/gemm).

Compile-time ONLY: binding allocates inputs (freed per shape); NO autotune,
NO do_bench, NO torch.compile. Foreground serial single process.

Run from /tmp:
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_dual_heuristic_configs.py
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

# Reuse the established binding path + arg builders + curriculum + classifier.
from run2_measure_g import KERNELS  # noqa: E402
from shapes_v3_draft import SHAPES  # noqa: E402
from run3_task1_seed_configs import _classify, _jsonify  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

# The proven-faithful port of MAIN's narrow heuristic (validated in Task B).
from main_reduction_heuristic_port import (  # noqa: E402
    main_is_eligible,
    main_get_seed_config,
)

# Machine-portable wiring assert: helion must resolve to THIS worktree.
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), (
    f"helion ({helion.__file__}) not under harness worktree ({_WT_ROOT}); "
    "set PYTHONPATH to this worktree."
)

LOG_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "logs", "run3"))
os.makedirs(LOG_DIR, exist_ok=True)
OUT = os.path.join(LOG_DIR, "dual_heuristic_configs.json")

SPLITS = ("train", "val", "test")
KERNEL_ORDER = [
    "rms_norm", "layer_norm", "softmax", "welford", "sum", "long_sum",
    "cross_entropy", "kl_div", "jsd",
]

# Knobs to surface in the per-shape diff (Task E worklist columns).
DIFF_KNOBS = [
    "block_sizes", "reduction_loops", "num_warps", "num_stages", "pid_type",
    "num_sm_multiplier", "maxnreg", "load_eviction_policies",
]


def _norm_copy(spec, raw):
    """Return a normalized copy of a raw config dict (or None on failure)."""
    norm = dict(raw)
    spec.normalize(norm)
    return norm


def record_one(kernel_name, M, N) -> dict:
    fn, builder, _tc_ref = KERNELS[kernel_name]
    args, _ref, _out = builder(M, N)

    bound = fn.bind(args)
    env = bound.env
    device_ir = bound.host_function.device_ir
    spec = env.config_spec

    rec: dict = {"kernel": kernel_name, "M": M, "N": N}

    # ---- MINE: the live deep heuristic ----
    mine_seeds = compiler_seed_configs(env, device_ir)
    mine_fired = list(getattr(spec, "autotuner_heuristics", []))
    rec["mine_heuristics_fired"] = mine_fired
    rec["n_reduction_facts"] = len(getattr(spec, "reduction_facts", []))
    mine_fires = len(mine_seeds) > 0
    rec["mine_fires"] = mine_fires
    mine_raw = mine_norm = None
    if mine_fires:
        mine_raw = dict(mine_seeds[0])
        rec["mine_raw"] = _jsonify(mine_raw)
        try:
            mine_norm = _norm_copy(spec, mine_raw)
            rec["mine_norm"] = _jsonify(mine_norm)
        except Exception as e:  # noqa: BLE001
            rec["mine_norm"] = None
            rec["mine_norm_error"] = f"{type(e).__name__}: {e}"
    else:
        rec["mine_raw"] = None
        rec["mine_norm"] = None

    # ---- MAIN: the proven-faithful port on the SAME env/device_ir ----
    # NOTE: compiler_seed_configs above reset spec.autotuner_heuristics; the port
    # does not touch it, so mine_heuristics_fired stays MINE-only. We re-run the
    # port directly (it reads only spec fields main reads; Task C proved the
    # structural inputs are identical across trees -> faithful single-tree).
    main_eligible = main_is_eligible(env, device_ir)
    main_cfg = main_get_seed_config(env, device_ir)
    main_fires = main_cfg is not None
    rec["main_eligible"] = main_eligible
    rec["main_fires"] = main_fires
    main_raw = main_norm = None
    if main_fires:
        main_raw = dict(main_cfg)
        rec["main_raw"] = _jsonify(main_raw)
        try:
            main_norm = _norm_copy(spec, main_raw)
            rec["main_norm"] = _jsonify(main_norm)
        except Exception as e:  # noqa: BLE001
            rec["main_norm"] = None
            rec["main_norm_error"] = f"{type(e).__name__}: {e}"
    else:
        rec["main_raw"] = None
        rec["main_norm"] = None

    # ---- comparison ----
    both_fire = mine_fires and main_fires
    rec["both_fire"] = both_fire
    configs_identical = None
    knob_diffs = None
    if both_fire and mine_norm is not None and main_norm is not None:
        configs_identical = (mine_norm == main_norm)
        if not configs_identical:
            knob_diffs = {}
            for knob in DIFF_KNOBS:
                mv = mine_norm.get(knob)
                xv = main_norm.get(knob)
                if mv != xv:
                    knob_diffs[knob] = {"mine": _jsonify(mv), "main": _jsonify(xv)}
            # also surface any knob present in one normalized config but not other
            for knob in set(mine_norm) | set(main_norm):
                if knob in DIFF_KNOBS:
                    continue
                mv = mine_norm.get(knob)
                xv = main_norm.get(knob)
                if mv != xv:
                    knob_diffs[knob] = {"mine": _jsonify(mv), "main": _jsonify(xv)}
    rec["configs_identical"] = configs_identical
    rec["knob_diffs"] = knob_diffs

    # ---- classification (MINE's reduction fact) ----
    facts = getattr(spec, "reduction_facts", [])
    if facts:
        fact = facts[0]
        rec["classification"] = _classify(spec, fact)
    else:
        rec["classification"] = "no_reduction_fact"

    del args, bound, env, device_ir, spec, mine_seeds, main_cfg
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
                ident = rec.get("configs_identical")
                ident_s = ("=" if ident is True else
                           "DIFF" if ident is False else "-")
                print(
                    f"[ OK ] {tag}  mine={rec['mine_fires']!s:5s} "
                    f"main={rec['main_fires']!s:5s} ident={ident_s:4s} "
                    f"cls={rec.get('classification')}",
                    flush=True,
                )
            # checkpoint after each (kernel, split)
            json.dump({"rows": rows}, open(OUT, "w"), indent=1)
        # checkpoint after each kernel too
        json.dump({"rows": rows}, open(OUT, "w"), indent=1)

    json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print(f"\n=== DONE: {n_ok} recorded, {n_err} errored ===", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
