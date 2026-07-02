"""Standalone SEED-CONFIG MANIFEST + differ for the reduction-seed heuristic.

Records, for EVERY (corpus, kernel, dtype, split, shape) over the FULL shape matrix
(all splits — not just the perf-report's test subset), the config the seed heuristic
actually emits. Front-end only (bind -> compiler_seed_configs -> normalize): NO codegen,
NO ptxas, NO timing, so it is fast (~minutes for the whole matrix) and immune to the
ptxas-hang that bites the timing sweep on the big 3-D/4-D shapes.

Per cell it records:
  fired_heuristics   which AutotunerHeuristic(s) emitted the seed
  n_reduction_facts  eligibility witness
  raw_seed           the heuristic's literal Config dict (what it DECIDED)
  normalized_seed    that config after spec.normalize() (what actually RUNS)
  base_default       the unseeded compiler base (_base_default_config), normalized
  configs_differ     normalized_seed != base_default
  classification     T1_rolled / T2_usertiled / materialized / gemm / no_reduction_fact
  reduction_fact     the ReductionFact[0] fields the heuristic keyed on (the "why")

Reuses the exact builder wiring from perf_report_bench.py (dtype-parameterized), so the
manifest is over the same kernels/shapes the perf report benched, expanded to all splits.

Usage (from /tmp):
  # RECORD a manifest
  HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-redesign \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/perf_report/config_manifest.py \
    --out /home/dev/local/prompts-lab/perf-report/results/config_manifest.json

  # DIFF two manifests (exit 1 if any cell's config changed)
  ... config_manifest.py --diff BEFORE.json AFTER.json

  # subset while iterating on a heuristic change:
  ... config_manifest.py --out X.json --corpus curriculum,vllm --kernels rms_norm,softmax
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback

# Reuse the harness's path wiring + builders (it sets sys.path + asserts worktree helion).
_THIS = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(_THIS))

import perf_report_bench as H  # noqa: E402  (does the path + helion-worktree assert)
import torch  # noqa: E402
import helion  # noqa: E402


# --------------------------------------------------------------------------- #
#  One cell: bind, extract raw + normalized seed, base default, classification.
# --------------------------------------------------------------------------- #
def _classify(spec, fact) -> str:
    try:
        if getattr(spec, "matmul_facts", None):
            return "gemm"
        bid = getattr(fact, "primary_reduction_block_id")
        rl = set(spec.reduction_loops.valid_block_ids())
        bs = set(spec.block_sizes.valid_block_ids())
        if bid in rl:
            return "T1_rolled"
        if bid in bs:
            return "T2_usertiled"
        return "materialized"
    except Exception as e:  # noqa: BLE001
        return f"unknown({type(e).__name__})"


def _normalized(bound, cfg) -> dict | None:
    if cfg is None:
        return None
    norm = dict(cfg.config) if hasattr(cfg, "config") else dict(cfg)
    try:
        with bound.env:
            bound.env.config_spec.normalize(norm)
        return H._jsonify(norm) if hasattr(H, "_jsonify") else norm
    except Exception:  # noqa: BLE001
        return {"_normalize_error": True, **{k: str(v) for k, v in norm.items()}}


def _jsonify(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonify(x) for k, x in v.items()}
    return repr(v)


def record_cell(corpus, kernel, split, shape, dtype, kfn, args) -> dict:
    rec = {"corpus": corpus, "kernel": kernel, "split": split,
           "shape": list(shape) if isinstance(shape, (list, tuple)) else shape,
           "dtype": dtype}
    bound = H.with_compile_timeout(lambda: kfn.bind(args), seconds=90)
    spec = bound.env.config_spec
    seeds = list(spec.compiler_seed_configs)
    fired = list(spec.autotuner_heuristics)
    with bound.env:
        base = spec._base_default_config()
    rec["fired_heuristics"] = fired
    rec["fact_counts"] = {
        "reduction": len(spec.reduction_facts),
        "matmul": len(spec.matmul_facts),
        "pointwise": len(spec.pointwise_facts),
        "accumulator": len(spec.accumulator_facts),
    }
    seed = seeds[0] if seeds else None
    rec["raw_seed"] = _jsonify(dict(seed.config)) if seed is not None else None
    rec["normalized_seed"] = _normalized(bound, seed)
    rec["base_default"] = _normalized(bound, base)
    rec["configs_differ"] = rec["normalized_seed"] != rec["base_default"]
    rec["n_seeds"] = len(seeds)
    rfacts = spec.reduction_facts
    if rfacts:
        rec["reduction_fact"] = _jsonify(rfacts[0]._asdict())
        rec["classification"] = _classify(spec, rfacts[0])
    else:
        rec["reduction_fact"] = None
        rec["classification"] = "no_reduction_fact"
    del bound, spec, seeds
    for a in (args if isinstance(args, (list, tuple)) else ()):
        if torch.is_tensor(a):
            del a
    torch.cuda.empty_cache()
    return rec


# --------------------------------------------------------------------------- #
#  Full-matrix iteration (all splits) reusing perf_report_bench builders.
# --------------------------------------------------------------------------- #
def _build_real(corpus, kernel, shape, dtype):
    """Return (kfn, args) for a real-corpus cell (config extraction only)."""
    dt = H._DT.get(dtype)
    if corpus == "curriculum":
        kfn, args, *_ = H._cur_build(kernel, shape[0], shape[1], dt)
        return kfn, args
    if corpus == "transfer":
        kfn, args, *_ = H._transfer_build(kernel, shape, dt)
        return kfn, args
    if corpus == "mreduction":
        kfn, args, *_ = H._mred_build(kernel, tuple(shape), dt)
        return kfn, args
    if corpus == "vllm":
        import bench_arms as B
        mod_name, kern_attr, builder, _sub, _key = B.SPECS[kernel]
        mod = importlib.import_module(mod_name)
        kfn = getattr(mod, kern_attr)
        tok, hidden = shape[0], shape[1]
        group = shape[2] if len(shape) > 2 else None
        args = builder(tok, hidden, group)[0]
        return kfn, args
    raise KeyError(corpus)


def iter_real(SH, corpora, kfilter):
    """Yield (corpus, kernel, split, shape, dtype) over the FULL matrix (all splits)."""
    for corpus in corpora:
        if corpus not in ("curriculum", "transfer", "mreduction", "vllm"):
            continue
        c = SH["corpora"][corpus]
        for kernel, kdef in c["kernels"].items():
            if kfilter and kernel not in kfilter:
                continue
            for split, shapes in kdef["shapes"].items():
                for shape in shapes:
                    for dtype in c["dtypes"]:
                        yield corpus, kernel, split, tuple(shape), dtype


def iter_synth(SH, corpora, kfilter):
    for corpus in corpora:
        if corpus not in ("synthetic_probes", "adversarial_synth"):
            continue
        for kernel in SH["corpora"][corpus]["kernels"]:
            if kfilter and kernel not in kfilter:
                continue
            yield corpus, kernel


# --------------------------------------------------------------------------- #
#  Record + diff drivers
# --------------------------------------------------------------------------- #
def _key(rec):
    return f"{rec['corpus']}/{rec['kernel']}/{rec['dtype']}/{tuple(rec['shape']) if isinstance(rec.get('shape'), list) else rec.get('shape')}"


def cmd_record(args):
    SH = H._load_shapes()
    corpora = args.corpus.split(",") if args.corpus else \
        ["curriculum", "transfer", "mreduction", "vllm",
         "synthetic_probes", "adversarial_synth"]
    kfilter = set(args.kernels.split(",")) if args.kernels else None
    print(f"helion={helion.__file__}\nout={args.out}\n", flush=True)
    rows = []
    n_ok = n_err = 0

    for (corpus, kernel, split, shape, dtype) in iter_real(SH, corpora, kfilter):
        tag = f"{corpus:11s} {kernel:26s} {str(list(shape)):20s} {dtype:5s} {split}"
        try:
            kfn, kargs = _build_real(corpus, kernel, shape, dtype)
            rec = record_cell(corpus, kernel, split, shape, dtype, kfn, kargs)
            rows.append(rec)
            n_ok += 1
            ns = rec.get("normalized_seed") or {}
            print(f"[OK ] {tag} cls={rec['classification']:13s} "
                  f"bs={ns.get('block_sizes')} rl={ns.get('reduction_loops')} "
                  f"w={ns.get('num_warps')} differ={rec['configs_differ']}", flush=True)
        except H._CompileTimeout:
            H._reap_compile_children()
            n_err += 1
            rows.append({"corpus": corpus, "kernel": kernel, "split": split,
                         "shape": list(shape), "dtype": dtype,
                         "error": "bind-timeout>90s"})
            print(f"[ERR] {tag}: bind-timeout", flush=True)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            rows.append({"corpus": corpus, "kernel": kernel, "split": split,
                         "shape": list(shape), "dtype": dtype,
                         "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"[ERR] {tag}: {type(e).__name__}: {str(e)[:100]}", flush=True)
            if args.verbose:
                traceback.print_exc()
        json.dump({"rows": rows}, open(args.out, "w"), indent=1)  # checkpoint

    for (corpus, kernel) in iter_synth(SH, corpora, kfilter):
        tag = f"{corpus:17s} {kernel:40s}"
        try:
            kfn, kargs, shape = H._load_synth(
                "synthetic_probes" if corpus == "synthetic_probes" else "adversarial_synth",
                kernel)
            rec = record_cell(corpus, kernel, "native", shape or "n/a", "native", kfn, kargs)
            rows.append(rec)
            n_ok += 1
            ns = rec.get("normalized_seed") or {}
            print(f"[OK ] {tag} cls={rec['classification']:13s} "
                  f"n_seeds={rec['n_seeds']} differ={rec['configs_differ']}", flush=True)
        except Exception as e:  # noqa: BLE001
            n_err += 1
            rows.append({"corpus": corpus, "kernel": kernel, "split": "native",
                         "dtype": "native", "shape": None,
                         "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"[ERR] {tag}: {type(e).__name__}: {str(e)[:100]}", flush=True)
        json.dump({"rows": rows}, open(args.out, "w"), indent=1)

    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print(f"\n=== DONE: {n_ok} recorded, {n_err} errored -> {args.out} ===", flush=True)


_DIFF_FIELDS = ("normalized_seed", "raw_seed", "base_default", "fired_heuristics",
                "classification", "n_seeds", "configs_differ")


def cmd_diff(before_path, after_path):
    before = {_key(r): r for r in json.load(open(before_path))["rows"] if "error" not in r}
    after = {_key(r): r for r in json.load(open(after_path))["rows"] if "error" not in r}
    changed = []
    for k in sorted(set(before) | set(after)):
        b, a = before.get(k), after.get(k)
        if b is None:
            changed.append((k, "ADDED", None, None)); continue
        if a is None:
            changed.append((k, "REMOVED", None, None)); continue
        for f in _DIFF_FIELDS:
            if b.get(f) != a.get(f):
                changed.append((k, f, b.get(f), a.get(f)))
    if not changed:
        print(f"ZERO-DIFF: {len(after)} cells byte-identical (BEFORE vs AFTER). "
              "No config, heuristic, or classification changed.")
        return
    ncells = len({c[0] for c in changed})
    print(f"CHANGED: {len(changed)} field(s) across {ncells} cell(s):\n")
    for k, field, bv, av in changed:
        if field in ("ADDED", "REMOVED"):
            print(f"  {field:8s} {k}")
        else:
            print(f"  {k}\n    {field}: {bv}  ->  {av}")
    sys.exit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/home/dev/local/prompts-lab/perf-report/results/config_manifest.json")
    p.add_argument("--corpus", default="", help="comma list; default = all 6")
    p.add_argument("--kernels", default="", help="comma list of kernel names")
    p.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    if a.diff:
        cmd_diff(a.diff[0], a.diff[1])
        return
    a.out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cmd_record(a)


if __name__ == "__main__":
    main()
