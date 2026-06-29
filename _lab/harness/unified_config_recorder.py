"""UNIFIED all-kernel config-recorder (Step 0 of the reduction-unification hillclimb).

The single instrument every gate depends on: a BEFORE/AFTER normalized-config diff
across the FULL active matrix (9 curriculum + 8 transfer + 6 m-reduction + 5 vLLM),
in ONE JSON schema. Compile-time only (HELION_AUTOTUNE_EFFORT=none); binds each
kernel x shape, reads the seed the live heuristic PERSISTED on the spec during bind
(spec.compiler_seed_configs -- NOT a re-call of compiler_seed_configs, which raises
NoCurrentEnvironment outside the env ctx), and records:

  - heuristics_fired   : which AutotunerHeuristic(s) emitted the seed
  - n_reduction_facts  : eligibility-gate witness (the len==1 gate)
  - raw_seed           : the heuristic's literal Config dict (pre-normalize)
  - normalized_cfg     : the SAME config after spec.normalize() -- what configs=[seed]
                         actually runs (forces persistent when value>=size_hint, etc.)
  - reduction_fact     : the ReductionFact[0] fields the heuristic keyed on (the "why")
  - fact_counts        : {reduction, matmul, pointwise, accumulator, matmul_redux_epi}

Each row keyed by (corpus, kernel, shape-tuple, dtype). A row that errors records the
error string (never silently dropped). Serial foreground (one GPU; allocation-only but
we keep the discipline + free between shapes).

Usage (from /tmp):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-unify \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-unify/_lab/harness/unified_config_recorder.py --out /path/to/out.json

  # subset: --corpus curriculum,vllm   --kernels rms_norm,per_token_group_fp8
  # diff:   --diff BEFORE.json AFTER.json   (prints the changed cells; exit 1 if any)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
_PROMPTS_DIR = os.path.join(_HARNESS_DIR, "..", "prompts")
# vLLM + m-reduction infra lives in prompts-lab/vllm-bench (NOT in the worktree); the
# transfer infra in prompts-lab/transfer. Discover prompts-lab as a sibling of the
# worktree's parent ("/home/dev/local") so paths stay portable.
_LOCAL_ROOT = os.path.abspath(os.path.join(_WT_ROOT, ".."))
_VLLM_DIR = os.path.join(_LOCAL_ROOT, "prompts-lab", "vllm-bench")
_TRANSFER_DIR = os.path.join(_LOCAL_ROOT, "prompts-lab", "transfer")
for _d in (_HARNESS_DIR, _PROMPTS_DIR, _VLLM_DIR, _TRANSFER_DIR):
    _d = os.path.abspath(_d)
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT_ROOT}); set PYTHONPATH."
)


# --------------------------------------------------------------------------- #
#  JSON helpers
# --------------------------------------------------------------------------- #
def _jsonify(v: object) -> object:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonify(x) for k, x in v.items()}
    return repr(v)


def _classify(spec: object, fact: object) -> str:
    """T1 (rdim in reduction_loops) / T2 (rdim is a block_sizes entry) /
    materialized (rdim in neither -- the standard materialized case) / gemm / oos."""
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
        return f"unknown ({type(e).__name__})"


# --------------------------------------------------------------------------- #
#  The unified bind+record core (corpus-agnostic).
# --------------------------------------------------------------------------- #
def record_bound(corpus: str, kernel: str, shape: object, dtype: str, fn: object,
                 args: tuple) -> dict:
    """Bind one kernel and snapshot the persisted seed + facts. The seed the live
    heuristic produced is persisted on spec during bind() (inside its env ctx)."""
    bound = fn.bind(args)
    spec = bound.env.config_spec
    seeds = list(spec.compiler_seed_configs)
    fired = list(spec.autotuner_heuristics)
    rec: dict = {
        "corpus": corpus,
        "kernel": kernel,
        "shape": list(shape) if isinstance(shape, (list, tuple)) else shape,
        "dtype": dtype,
        "heuristics_fired": fired,
        "fact_counts": {
            "reduction": len(spec.reduction_facts),
            "matmul": len(spec.matmul_facts),
            "pointwise": len(spec.pointwise_facts),
            "accumulator": len(spec.accumulator_facts),
            "matmul_redux_epi": len(spec.matmul_reduction_epilogue_facts),
        },
        "n_seeds": len(seeds),
    }
    if not seeds:
        rec["raw_seed"] = None
        rec["normalized_cfg"] = None
        rec["note"] = "no seed (declined / not eligible)"
    else:
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
    rfacts = spec.reduction_facts
    if rfacts:
        rec["reduction_fact"] = _jsonify(rfacts[0]._asdict())
        rec["classification"] = _classify(spec, rfacts[0])
    else:
        rec["reduction_fact"] = None
        rec["classification"] = "no_reduction_fact"
    # free GPU memory before next (possibly multi-GB) shape
    del bound, spec, seeds
    torch.cuda.empty_cache()
    return rec


# --------------------------------------------------------------------------- #
#  CORPUS 1 — the 9 curriculum kernels (fp32), via run2_measure_g builders.
# --------------------------------------------------------------------------- #
def _iter_curriculum(kernels_filter: set[str] | None):
    from run2_measure_g import KERNELS  # noqa: E402
    from shapes_v3_draft import SHAPES  # noqa: E402

    order = ["rms_norm", "layer_norm", "softmax", "welford", "sum", "long_sum",
             "cross_entropy", "kl_div", "jsd"]
    for kname in order:
        if kernels_filter and kname not in kernels_filter:
            continue
        fn, builder, _ref = KERNELS[kname]
        for split, shapes in SHAPES[kname].items():
            for (m, n) in shapes:
                args = builder(m, n)[0]
                yield ("curriculum", kname, (m, n), "fp32", fn, args, split)


# --------------------------------------------------------------------------- #
#  CORPUS 2 — the 8 transfer kernels (bf16), via ab_three_arm_transfer adapters.
# --------------------------------------------------------------------------- #
def _iter_transfer(kernels_filter: set[str] | None):
    import ab_three_arm_transfer as AB  # noqa: E402
    import shapes_transfer as SH  # noqa: E402

    for kname in SH.SHAPES:
        if kernels_filter and kname not in kernels_filter:
            continue
        build = AB._make(kname)
        for shape in SH.SHAPES[kname]:
            kfn, args, _ref, _chk = build(shape, torch.bfloat16)
            yield ("transfer", kname, tuple(shape), "bf16", kfn, args, "transfer")


# --------------------------------------------------------------------------- #
#  CORPUS 3 — the 5 vLLM kernels (bf16), via bench_arms builders.
# --------------------------------------------------------------------------- #
# Decode + prefill token counts spanning the firing shapes that matter.
_VLLM_SHAPES = {
    "silu_mul_fp8": [(32, 14336, None), (128, 14336, None), (2048, 8192, None),
                     (8192, 4096, None)],
    "dynamic_per_token_scaled_fp8_quant": [(128, 8192, None), (8192, 4096, None),
                                           (8192, 8192, None), (2048, 16384, None)],
    "rms_norm_dynamic_per_token_quant": [(128, 8192, None), (8192, 4096, None),
                                         (8192, 8192, None), (2048, 16384, None)],
    "per_token_group_fp8_quant": [(128, 4096, 128), (8192, 4096, 128),
                                  (8192, 7168, 128), (2048, 8192, 128)],
    "rms_norm_per_block_quant": [(128, 4096, 128), (8192, 4096, 128),
                                 (8192, 7168, 128), (2048, 8192, 64)],
}


def _iter_vllm(kernels_filter: set[str] | None):
    import bench_arms as B  # noqa: E402

    for kname, shapes in _VLLM_SHAPES.items():
        if kernels_filter and kname not in kernels_filter:
            continue
        mod_name, kern_attr, builder, _sub, _key = B.SPECS[kname]
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, kern_attr)
        for (tok, hidden, group) in shapes:
            args = builder(tok, hidden, group)[0]
            yield ("vllm", kname, (tok, hidden, group), "bf16", fn, args, "vllm")


# --------------------------------------------------------------------------- #
#  CORPUS 4 — the 6 m-reduction (norm-backward) kernels (fp16/fp32).
# --------------------------------------------------------------------------- #
def _build_mreduction(kname: str, shape: tuple):
    """Return (fn, args) for a m-reduction kernel at a given shape."""
    import mreduction_styles_view_only as MR  # noqa: E402

    DEV = "cuda"
    if kname == "bias_grad_bwd":
        (M, N) = shape
        go = torch.randn(M, N, device=DEV, dtype=torch.float32)
        return MR.bias_grad_bwd, (go,)
    if kname == "dyt_bwd":
        (M, N) = shape
        x = torch.randn(M, N, device=DEV, dtype=torch.float32)
        w = torch.randn(N, device=DEV, dtype=torch.float32)
        go = torch.randn(M, N, device=DEV, dtype=torch.float32)
        return MR.dyt_bwd, (go, x, w, 0.7)
    if kname == "group_norm_bwd":
        (Nn, C, S, G) = shape
        xg = torch.randn(Nn, C, S, device=DEV, dtype=torch.float32)
        wg = torch.randn(C, device=DEV, dtype=torch.float32)
        bg = torch.randn(C, device=DEV, dtype=torch.float32)
        gog = torch.randn(Nn, C, S, device=DEV, dtype=torch.float32)
        _, _, _, mean_g, rstd_g = MR.group_norm_ref(gog, xg, wg, bg, G)
        return MR.group_norm_bwd, (gog, xg, mean_g, rstd_g, wg, G)
    if kname == "instance_norm_bwd":
        (Bb, Ci, Si) = shape
        xi = torch.randn(Bb, Ci, Si, device=DEV, dtype=torch.float32)
        wi = torch.randn(Ci, device=DEV, dtype=torch.float32)
        bi = torch.randn(Ci, device=DEV, dtype=torch.float32)
        goi = torch.randn(Bb, Ci, Si, device=DEV, dtype=torch.float32)
        _, _, _, mean_i, rstd_i = MR.instance_norm_ref(goi, xi, wi, bi)
        return MR.instance_norm_bwd, (goi, xi, mean_i, rstd_i, wi)
    if kname in ("rms_norm_bwd", "layer_norm_bwd"):
        (M, N) = shape
        sys.path.insert(0, os.path.join(_WT_ROOT, "examples"))
        if kname == "rms_norm_bwd":
            rms_ex = importlib.import_module("rms_norm")
            xr = torch.randn(M, N, device=DEV, dtype=torch.float16)
            wr = torch.randn(N, device=DEV, dtype=torch.float16)
            gor = torch.randn(M, N, device=DEV, dtype=torch.float16)
            rms_val = torch.rsqrt(
                (xr.float() ** 2).mean(-1, keepdim=True) + 1e-5
            ).to(torch.float16)
            return rms_ex.rms_norm_bwd, (gor, xr, wr, rms_val)
        ln_ex = importlib.import_module("layer_norm")
        xl = torch.randn(M, N, device=DEV, dtype=torch.float16)
        wl = torch.randn(N, device=DEV, dtype=torch.float16)
        gol = torch.randn(M, N, device=DEV, dtype=torch.float16)
        mean_l = xl.float().mean(-1)
        rstd_l = torch.rsqrt(xl.float().var(-1, unbiased=False) + 1e-5)
        return ln_ex.layer_norm_bwd, (gol, xl, mean_l, rstd_l, wl)
    raise KeyError(kname)


# (kname, [shapes]). 2-D norms: (M,N). group/instance: (N,C,S[,G]).
_MRED_SHAPES = {
    "bias_grad_bwd": [(2048, 1024), (8192, 4096), (4096, 8192)],
    "dyt_bwd": [(2048, 1024), (8192, 4096), (4096, 8192)],
    "group_norm_bwd": [(128, 64, 64, 8), (256, 128, 128, 16)],
    "instance_norm_bwd": [(64, 16, 128), (128, 32, 256)],
    "rms_norm_bwd": [(2048, 4096), (8192, 4096), (4096, 8192)],
    "layer_norm_bwd": [(2048, 4096), (8192, 4096), (4096, 8192)],
}


def _iter_mreduction(kernels_filter: set[str] | None):
    for kname, shapes in _MRED_SHAPES.items():
        if kernels_filter and kname not in kernels_filter:
            continue
        for shape in shapes:
            fn, args = _build_mreduction(kname, shape)
            dt = "fp16" if kname in ("rms_norm_bwd", "layer_norm_bwd") else "fp32"
            yield ("mreduction", kname, shape, dt, fn, args, "mreduction")


_CORPORA = {
    "curriculum": _iter_curriculum,
    "transfer": _iter_transfer,
    "vllm": _iter_vllm,
    "mreduction": _iter_mreduction,
}


# --------------------------------------------------------------------------- #
#  Record + diff drivers
# --------------------------------------------------------------------------- #
def _row_key(rec: dict) -> str:
    return f"{rec['corpus']}/{rec['kernel']}/{rec['shape']}/{rec['dtype']}"


def cmd_record(args: argparse.Namespace) -> None:
    corpora = args.corpus.split(",") if args.corpus else list(_CORPORA)
    kfilter = set(args.kernels.split(",")) if args.kernels else None
    print(f"helion={helion.__file__}\nout={args.out}\n", flush=True)
    rows: list[dict] = []
    n_ok = n_err = 0
    for corpus in corpora:
        for (cps, kname, shape, dtype, fn, kargs, split) in _CORPORA[corpus](kfilter):
            tag = f"{cps:11s} {kname:30s} {str(shape):20s} {dtype}"
            try:
                rec = record_bound(cps, kname, shape, dtype, fn, kargs)
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"[ERR ] {tag}: {type(e).__name__}: {e}", flush=True)
                if args.verbose:
                    traceback.print_exc()
                rows.append({"corpus": cps, "kernel": kname,
                             "shape": list(shape) if isinstance(shape, tuple) else shape,
                             "dtype": dtype, "split": split,
                             "error": f"{type(e).__name__}: {e}"})
                torch.cuda.empty_cache()
                continue
            rec["split"] = split
            rows.append(rec)
            n_ok += 1
            norm = rec.get("normalized_cfg") or {}
            print(f"[ OK ] {tag} cls={rec.get('classification','?'):13s} "
                  f"fired={rec.get('heuristics_fired')} "
                  f"bs={norm.get('block_sizes')} rl={norm.get('reduction_loops')} "
                  f"w={norm.get('num_warps')}", flush=True)
            json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    json.dump({"rows": rows}, open(args.out, "w"), indent=1)
    print(f"\n=== DONE: {n_ok} recorded, {n_err} errored ===\nwrote {args.out}",
          flush=True)


# Fields whose change is a behavior change worth flagging.
_DIFF_FIELDS = ("normalized_cfg", "heuristics_fired", "n_seeds", "classification")


def cmd_diff(args: argparse.Namespace) -> None:
    before = {_row_key(r): r for r in json.load(open(args.diff[0]))["rows"]
              if "error" not in r}
    after = {_row_key(r): r for r in json.load(open(args.diff[1]))["rows"]
             if "error" not in r}
    changed = []
    for key in sorted(set(before) | set(after)):
        b, a = before.get(key), after.get(key)
        if b is None:
            changed.append((key, "ADDED", None, None)); continue
        if a is None:
            changed.append((key, "REMOVED", None, None)); continue
        for f in _DIFF_FIELDS:
            if b.get(f) != a.get(f):
                changed.append((key, f, b.get(f), a.get(f)))
    if not changed:
        print(f"ZERO-DIFF: {len(after)} cells byte-identical (BEFORE vs AFTER).")
        return
    print(f"CHANGED {len(changed)} field(s) across "
          f"{len({c[0] for c in changed})} cell(s):\n")
    for key, field, bv, av in changed:
        if field in ("ADDED", "REMOVED"):
            print(f"  {field:8s} {key}")
        else:
            print(f"  {key}\n    {field}: {bv}  ->  {av}")
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(_HARNESS_DIR, "..", "logs",
                                                 "unified", "configs.json"))
    p.add_argument("--corpus", default="", help="comma list: curriculum,transfer,vllm,mreduction")
    p.add_argument("--kernels", default="", help="comma list of kernel names to include")
    p.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                   help="diff two recorded JSONs; exit 1 if any cell changed")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.diff:
        cmd_diff(args)
        return
    args.out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cmd_record(args)


if __name__ == "__main__":
    main()
