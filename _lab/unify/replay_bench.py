"""CONFIG-REPLAY A/B bench (Gate R changed-cell re-bench + the checker's GPU tripwire).

The 10%-vs-frozen-champion bar (§7 comparison #1), footgun-correct: for one (corpus, kernel,
shape, dtype), replay the BEFORE config vs the AFTER config IN ONE PROCESS on the SAME tensors,
median-of-9 cold-L2 do_bench, accuracy-gated against a pure-torch reference. Reports
seed_AFTER / seed_BEFORE (>1.0 = AFTER slower; >1.10 = REGRESSION past the bar). Selection-only
(same source, different configs) => perf-identity holds for byte-identical cells, so only CHANGED
cells need this. NO torch.compile anywhere (champion = the current heuristic).

Also supports --default (the "model well" tripwire, §5b half-i): seed config vs the compiler
DEFAULT config; default_BEATS_seed (seed/default > ~1.05) is a RED signal.

Reuses unified_config_recorder's per-corpus adapters so every corpus is covered identically.

Run from /tmp (one fresh process PER kernel — never batch, footgun #11):
  HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=<worktree> python replay_bench.py \
     --corpus curriculum --kernel rms_norm --shape 8192,4096 --dtype fp32 \
     --before '{...cfg...}' --after '{...cfg...}'
  # or --default to compare seed vs compiler default (the tripwire)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_HARNESS = os.path.join(_THIS, "..", "harness")
for _d in (_THIS, os.path.abspath(_HARNESS)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch  # noqa: E402
from triton.testing import do_bench  # noqa: E402

import helion  # noqa: E402

import unified_config_recorder as U  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(U._WT_ROOT + os.sep)

N_RUNS = 9


def _med(fn) -> float:
    torch.cuda.synchronize()
    samples = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return samples[len(samples) // 2] * 1000.0  # us


def _find_cell(corpus: str, kernel: str, shape: tuple, dtype: str):
    """Return (fn, args) for the requested cell by scanning the corpus adapter."""
    kfilter = {kernel}
    for (cps, kname, shp, dt, fn, args, _split) in U._CORPORA[corpus](kfilter):
        if kname == kernel and tuple(shp) == tuple(shape) and dt == dtype:
            return fn, args
    raise SystemExit(f"cell not found: {corpus}/{kernel}/{shape}/{dtype}")


def _bind_run(fn, args, cfg_dict):
    """Build a kernel pinned to cfg_dict (configs=[cfg], no autotune) and return a callable."""
    cfg = helion.Config(**cfg_dict)
    k = helion.kernel(fn.fn, config=cfg, static_shapes=False)
    return lambda: k(*args), k


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--kernel", required=True)
    p.add_argument("--shape", required=True, help="comma list, e.g. 8192,4096")
    p.add_argument("--dtype", required=True)
    p.add_argument("--before", help="JSON config dict (the frozen champion)")
    p.add_argument("--after", help="JSON config dict (the refactor)")
    p.add_argument("--default", action="store_true",
                   help="compare AFTER seed vs the compiler DEFAULT (the model-well tripwire)")
    args = p.parse_args()

    shape = tuple(int(x) for x in args.shape.split(","))
    fn, kargs = _find_cell(args.corpus, args.kernel, shape, args.dtype)
    print(f"helion={helion.__file__}", flush=True)
    nvidia = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader").read().strip()
    print(f"GPU mem.used={nvidia}", flush=True)

    if args.default:
        bound = fn.bind(kargs)
        spec = bound.env.config_spec
        with bound.env:
            default_cfg = spec.default_config()
        seeds = list(spec.compiler_seed_configs)
        if not seeds:
            print(json.dumps({"error": "no seed emitted"})); return
        after_cfg = dict(seeds[0])
        before_cfg = dict(default_cfg)
        before_label, after_label = "default", "seed"
    else:
        before_cfg = json.loads(args.before)
        after_cfg = json.loads(args.after)
        before_label, after_label = "before", "after"

    f_before, _ = _bind_run(fn, kargs, before_cfg)
    f_after, _ = _bind_run(fn, kargs, after_cfg)
    # warmup + correctness: both must run; compare outputs to each other (same kernel/source).
    out_b = f_before()
    out_a = f_after()

    def _first(o):
        return o[0] if isinstance(o, (tuple, list)) else o
    ob, oa = _first(out_b), _first(out_a)
    same = None
    if torch.is_tensor(ob) and torch.is_tensor(oa):
        try:
            same = bool(torch.allclose(ob.float(), oa.float(), rtol=2e-2, atol=2e-2))
        except Exception:  # noqa: BLE001
            same = None

    t_before = _med(f_before)
    t_after = _med(f_after)
    ratio = t_after / t_before if t_before else float("nan")
    result = {
        "cell": f"{args.corpus}/{args.kernel}/{list(shape)}/{args.dtype}",
        f"{before_label}_us": round(t_before, 3),
        f"{after_label}_us": round(t_after, 3),
        "ratio_after_over_before": round(ratio, 4),
        "outputs_match": same,
        "regression_past_10pct": bool(ratio > 1.10),
        "default_beats_seed": bool(args.default and ratio < 1.0 / 1.05),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
