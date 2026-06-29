"""P0 RED-baseline recorder for the 13 stress-test probe kernels.

Binds each probe (compile-time only, HELION_AUTOTUNE_EFFORT=none) against the CURRENTLY
ACTIVE heuristic (the helion on PYTHONPATH) and records the "before picture":
  - which AutotunerHeuristic(s) fired (or none -> default)
  - the fact counts + each ReductionFact's full field set
  - the per-reduction graph_id (co-residency signal) + classification
  - the emitted seed config (raw + normalized)
  - any compile error (recorded, never silently dropped)

This is corpus-AGNOSTIC about the heuristic version, so it doubles as the GREEN recorder
after Stage 1/2 land — diff RED vs GREEN to see how each probe's path/seed moved.

Usage (from /tmp):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-redesign \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/redesign/probe_recorder.py --out OUT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
_LOCAL_ROOT = os.path.abspath(os.path.join(_WT_ROOT, ".."))
_KERNELS_DIR = os.path.join(
    _LOCAL_ROOT, "prompts-lab", "reduction-generality", "kernels"
)

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT_ROOT}); set PYTHONPATH."
)

# The 13 stress-test probes (slug -> the kernel-getter + arg-builder). p1-p11 + oos1/oos2.
# Most expose a single @helion.kernel fn; oos1 hides it behind get_kernel().
_PROBES = [
    "p1-outer-product-coresident",
    "p2-feature-plus-rowaccum-offcorpus",
    "p3-full-grid-nonquant",
    "p4-two-rollable-sequential",
    "p5-3d-reduction-tile",
    "p6-mixed-coresident-plus-sequential",
    "p7-gridtile-then-usertile",
    "p8-fullgrid-plus-usertile",
    "p9-nonred-loop-then-fullextent",
    "p10-usertile-and-gridtile",
    "p11-fullextent-then-nonred-loop",
    "oos1-jagged-declined",
    "oos2-strided-dim0",
]


def _load_probe(slug: str):
    """Import a probe kernel module by file path (dir name has hyphens)."""
    path = os.path.join(_KERNELS_DIR, slug, "kernel.py")
    spec = importlib.util.spec_from_file_location(
        f"probe_{slug.replace('-', '_')}", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_fn(mod):
    """The kernel callable: get_kernel() if present (oos1), else the single @helion.kernel."""
    if hasattr(mod, "get_kernel"):
        return mod.get_kernel()
    # The one Kernel object defined in the module.
    from helion.runtime.kernel import Kernel

    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, Kernel):
            return obj
    raise RuntimeError(f"no Kernel found in {mod.__name__}")


def _jsonify(v: object) -> object:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonify(x) for k, x in v.items()}
    return repr(v)


def _reduction_graph_ids(spec) -> dict:
    """Per-reduction primary block_id -> the graph_ids of its reduction-fed loads (the
    co-residency signal §2.2). Read off memory_op_facts. Best-effort (RED baseline: the
    current fact has no graph grouping, so this surfaces the raw signal)."""
    out = {}
    mofs = list(getattr(spec, "memory_op_facts", []))
    for rf in spec.reduction_facts:
        bid = rf.primary_reduction_block_id
        gids = sorted(
            {
                f.graph_id
                for f in mofs
                if f.kind == "load"
                and any(ax == bid for ax, _ in f.reductions_fed)
            }
        )
        out[str(bid)] = gids
    return out


def record_probe(slug: str) -> dict:
    mod = _load_probe(slug)
    fn = _get_fn(mod)
    args = mod.make_args()
    rec: dict = {"probe": slug}
    try:
        bound = fn.bind(args)
    except Exception as e:  # noqa: BLE001
        import traceback

        rec["bind_error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        return rec
    spec = bound.env.config_spec
    seeds = list(spec.compiler_seed_configs)
    rec["heuristics_fired"] = list(spec.autotuner_heuristics)
    rec["fact_counts"] = {
        "reduction": len(spec.reduction_facts),
        "matmul": len(spec.matmul_facts),
        "pointwise": len(spec.pointwise_facts),
        "accumulator": len(spec.accumulator_facts),
        "matmul_redux_epi": len(spec.matmul_reduction_epilogue_facts),
    }
    rec["n_seeds"] = len(seeds)
    rec["reduction_facts"] = [_jsonify(rf._asdict()) for rf in spec.reduction_facts]
    rec["reduction_graph_ids"] = _reduction_graph_ids(spec)
    rec["block_sizes_valid"] = list(spec.block_sizes.valid_block_ids())
    rec["reduction_loops_valid"] = list(spec.reduction_loops.valid_block_ids())
    if seeds:
        raw = dict(seeds[0])
        rec["raw_seed"] = _jsonify(raw)
        norm = dict(raw)
        try:
            with bound.env:
                spec.normalize(norm)
            rec["normalized_cfg"] = _jsonify(norm)
        except Exception as e:  # noqa: BLE001
            rec["normalized_cfg"] = None
            rec["normalize_error"] = f"{type(e).__name__}: {e}"
    else:
        rec["raw_seed"] = None
        rec["normalized_cfg"] = None
        rec["note"] = "NO SEED (declined / not eligible / fell to default)"
    del bound, spec, seeds
    torch.cuda.empty_cache()
    return rec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_HARNESS_DIR, "probe_red_baseline.json"))
    p.add_argument("--only", default="", help="comma list of probe slugs")
    args = p.parse_args()
    only = set(args.only.split(",")) if args.only else None
    print(f"helion={helion.__file__}\nout={args.out}\n", flush=True)
    rows = []
    for slug in _PROBES:
        if only and slug not in only:
            continue
        rec = record_probe(slug)
        rows.append(rec)
        fc = rec.get("fact_counts", {})
        fired = rec.get("heuristics_fired", rec.get("bind_error", "?"))
        norm = rec.get("normalized_cfg") or {}
        print(
            f"[{slug:38s}] nred={fc.get('reduction','-')} "
            f"fired={fired} bs={norm.get('block_sizes')} "
            f"rl={norm.get('reduction_loops')} w={norm.get('num_warps')}",
            flush=True,
        )
        json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out} ({len(rows)} probes)", flush=True)


if __name__ == "__main__":
    main()
