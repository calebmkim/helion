"""Autotune ONE (kernel, shape, dtype) with a real Helion search (no LLM — offline box),
print the winning Config as JSON. Bounded by a wall-clock timeout imposed by the CALLER
(subprocess + killpg) plus HELION_AUTOTUNE_MAX_GENERATIONS. Foreground, one GPU job.

Usage:
  CUDA_VISIBLE_DEVICES=0 MMPERF_WORKTREE=<wt> python autotune_one.py \
      --kernel bmm --shape 16,4096,128,4096 --dtype bf16 --out /tmp/at.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_WT = os.path.dirname(os.path.dirname(_HERE))
WORKTREE = os.environ.get("MMPERF_WORKTREE", _DEFAULT_WT)
for p in (WORKTREE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402
import helion  # noqa: E402

assert os.path.realpath(helion.__file__).startswith(os.path.realpath(WORKTREE) + os.sep), \
    f"WRONG HELION: {helion.__file__}"

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# reuse the exact input builders from the harness (identical inputs to the benchmark)
import mmperf  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--dtype", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shape = [int(s) for s in a.shape.split(",")]
    ss = mmperf.STATIC_SHAPES[a.kernel]
    fn = mmperf._kernel_fn(a.kernel)
    args, ref, tc_fn, meta = mmperf.make_inputs(a.kernel, shape, a.dtype)

    info = {"kernel": a.kernel, "shape": shape, "dtype": a.dtype,
            "helion_file": helion.__file__}
    t0 = time.time()
    # Build the kernel WITHOUT a fixed config so autotune searches the full space.
    kernel = helion.kernel(fn, static_shapes=ss)
    bound = kernel.bind(kernel.normalize_args(*args))

    # Explicit Differential Evolution search — a real GPU search, NO LLM (offline box).
    from helion.autotuner.differential_evolution import DifferentialEvolutionSearch
    # bound population/generations so it fits the caller's wall-clock cap.
    pop = int(os.environ.get("AT_POP", "40"))
    gens = int(os.environ.get("AT_GENS", "10"))
    search = DifferentialEvolutionSearch(
        bound, args, population_size=pop, max_generations=gens)
    best = search.autotune()
    info["best_config"] = dict(best)
    info["elapsed_s"] = round(time.time() - t0, 1)
    info["seed_config"] = dict(bound.env.config_spec.compiler_seed_configs[0])
    info["ok"] = True
    with open(a.out, "w") as fh:
        json.dump(info, fh, default=str)
    print("AUTOTUNE_BEST " + json.dumps({"best": info["best_config"],
                                         "seed": info["seed_config"],
                                         "elapsed_s": info["elapsed_s"]}, default=str))


if __name__ == "__main__":
    main()
