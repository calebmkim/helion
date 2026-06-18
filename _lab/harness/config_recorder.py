"""Config recorder + diff — the generalized "behavior oracle", for REGRESSION SCOPING.

Compile-time only: binds each cell to allocate inputs + run the live seed heuristic
(NO codegen by default, NO autotune, NO do_bench). Generalizes ``run3_task1_seed_configs.py``
with a DTYPE axis, the ROBUSTNESS split, the WS1 sibling kernels, and an optional
generated-Triton diff — so a BEFORE/AFTER config-diff can SCOPE the no-regression sweep:

  a cell whose NORMALIZED config is byte-identical before/after a heuristic edit is provably
  perf-invariant (codegen is deterministic in config+source) and needs NO perf re-benchmark;
  you only benchmark the cells that CHANGED.

THREE soundness rules baked in (see hillclimb-method.md §3 "config-diff scoping"):
  1. CHANGED != WIN — the diff scopes WHAT to bench, never proves an improvement. Every
     changed cell still needs the full correctness + matched-lever A/B + adversarial gates
     (the D4 flat-occ corner was a *flagged* cell wrongly assumed a win).
  2. FULL MATRIX or it is a false all-clear — the skip is sound only over every active cell:
     all dtypes (fp32/bf16/fp16) AND the robustness split (the extreme rows where valleys
     live). The fp32-only train/val/test recorder gave false all-clears for the bf16/fp16
     corners + robustness valleys behind real regressions (welford 7.3x valley; CE fp32 +24%).
  3. SELECTION-ONLY — config-identity => perf-identity only when the edit changed WHICH config
     is emitted, not the codegen of a fixed config. For an edit touching kernel source / a
     device_ir fact builder / normalize / a lowering, run with ``--triton``: it diffs the
     generated Triton (origin-line-stripped) and FAILS LOUD on any cell whose config is
     identical but whose Triton changed (the jsd-class edit).

TEST is EXCLUDED by default (the overfit/TEST-firewall gate is the sole TEST reader, at
freeze). Binding is allocation-only / no-perf, but the routine scoping loop stays off TEST;
pass ``--include-test`` only with intent.

Reuses the dtype-aware ``_lab/bench/bare_fwd_dtype.py`` KERNELS + build fns (the 9 + the WS1
siblings) and ``shapes_v3_draft.SHAPES``. ``run3_task1_seed_configs.py`` stays the fp32
deliverable "config oracle" (replayed in task2 to MEASURE the win); THIS script is the
during-climb scoper.

Usage (cwd=/tmp, PYTHONPATH=<worktree>):
  # snapshot BEFORE an edit (all dtypes, all non-test splits incl robustness):
  python config_recorder.py record --out /tmp/cfg_before.json
  #   ... make the heuristic edit ...
  python config_recorder.py record --out /tmp/cfg_after.json
  python config_recorder.py diff --before /tmp/cfg_before.json --after /tmp/cfg_after.json
  # scope to the kernels an edit plausibly touches (faster):
  python config_recorder.py record --out /tmp/cfg_before.json --dtypes bf16,fp16 welford groupnorm
  # codegen/source edit (jsd-class): add --triton to BOTH records; diff flags any
  # config-identical-but-Triton-changed cell as an unsound skip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
# Wiring: resolve helion to THIS worktree (not the editable install), and let
# bare_fwd_dtype's `from examples...` imports resolve. PYTHONPATH=<worktree> also works.
for _d in (_WT_ROOT, os.path.join(_WT_ROOT, "_lab", "bench"),
           os.path.join(_WT_ROOT, "_lab", "prompts")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch  # noqa: E402

import helion  # noqa: E402
from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

import bare_fwd_dtype as BF  # noqa: E402  (dtype-aware KERNELS + build fns + siblings)
import shapes_v3_draft as SH  # noqa: E402

assert os.path.realpath(helion.__file__).startswith(os.path.realpath(_WT_ROOT) + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT_ROOT}); set PYTHONPATH=<worktree>."
)

# Non-test splits the routine scoping loop covers (robustness INCLUDED — the extreme rows
# are where the un-scoped valleys lived). TEST is the overfit gate's read-once.
DEFAULT_SPLITS = ("train", "val", "robustness")


def _jsonify(v: object) -> object:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonify(x) for k, x in v.items()}
    return repr(v)


def _classify(spec: object, fact: object) -> str:
    """T1 (rdim block_id in reduction_loops) / T2 (reduction axis is a block_sizes entry)."""
    if getattr(spec, "matmul_facts", None):
        return "gemm"
    rl_ids = set(spec.reduction_loops.valid_block_ids())
    if fact.block_id in rl_ids:
        return "T1"
    bs_ids = set(spec.block_sizes.valid_block_ids())
    if fact.block_id in bs_ids:
        return "T2"
    return "out_of_scope"


def record_one(kernel_name: str, m: int, n: int, dt_name: str, with_triton: bool) -> dict:
    """Bind one (kernel, M, N, dtype) cell and record the emitted seed config + facts.

    Records the NORMALIZED config (what would actually run with configs=[seed]) — that is the
    field ``diff`` compares; raw_seed and the ReductionFact are kept as the "why".
    """
    fn, build, _tc = BF.KERNELS[kernel_name]
    dt = BF.DTYPES[dt_name]
    args, _ref, _extract = build(m, n, dt)

    bound = fn.bind(args)
    env = bound.env
    device_ir = bound.host_function.device_ir
    spec = env.config_spec
    seeds = compiler_seed_configs(env, device_ir)

    rec: dict = {
        "kernel": kernel_name,
        "dtype": dt_name,
        "M": m,
        "N": n,
        "heuristics_fired": list(getattr(spec, "autotuner_heuristics", [])),
    }
    if not seeds:
        rec["normalized_cfg"] = None
        rec["note"] = "no seed emitted (not eligible / no rdim)"
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
        if with_triton:
            # origin-line-stripped so a comment-line renumber is not a false codegen diff
            # (the jsd fp32-no-op proof: byte-identical apart from src line-number comments).
            try:
                src = bound.to_triton_code(seeds[0], output_origin_lines=False)
                rec["triton_sha"] = hashlib.sha256(src.encode()).hexdigest()[:16]
            except Exception as e:  # noqa: BLE001
                rec["triton_err"] = f"{type(e).__name__}: {str(e)[:120]}"

    facts = getattr(spec, "reduction_facts", [])
    if facts:
        rec["reduction_fact"] = _jsonify(facts[0]._asdict())
        rec["classification"] = _classify(spec, facts[0])
    else:
        rec["classification"] = "no_reduction_fact"

    del args, bound, env, device_ir, spec, seeds
    torch.cuda.empty_cache()
    return rec


def _cell_key(r: dict) -> str:
    return f"{r['kernel']}|{r['dtype']}|{r['M']}|{r['N']}"


def cmd_record(a: argparse.Namespace) -> None:
    kernels = a.kernels or list(BF.KERNELS)
    dtypes = a.dtypes.split(",")
    splits = a.splits.split(",") if a.splits else (
        list(DEFAULT_SPLITS) + (["test"] if a.include_test else [])
    )
    print(
        f"helion={helion.__file__}\nout={a.out}\nkernels={kernels}\n"
        f"dtypes={dtypes}  splits={splits}  triton={a.triton}\n",
        flush=True,
    )
    rows: list[dict] = []
    n_ok = n_err = 0
    for kn in kernels:
        for dt_name in dtypes:
            if dt_name == "fp16" and kn in BF.FP16_UNSUPPORTED:
                continue
            for split in splits:
                for (m, n) in SH.SHAPES.get(kn, {}).get(split, []):
                    tag = f"{kn:13s} {dt_name:4s} {split:10s} ({m},{n})"
                    try:
                        rec = record_one(kn, m, n, dt_name, a.triton)
                    except Exception as e:  # noqa: BLE001
                        n_err += 1
                        print(f"[ERR ] {tag}: {type(e).__name__}: {e}", flush=True)
                        rows.append({"kernel": kn, "dtype": dt_name, "M": m, "N": n,
                                     "split": split, "error": f"{type(e).__name__}: {e}"})
                        torch.cuda.empty_cache()
                        continue
                    rec["split"] = split
                    rows.append(rec)
                    n_ok += 1
                    # checkpoint after every cell so a kill loses nothing
                    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
    json.dump({"rows": rows}, open(a.out, "w"), indent=1)
    print(f"\n=== recorded {n_ok}, errored {n_err} -> {a.out} ===", flush=True)


def cmd_diff(a: argparse.Namespace) -> None:
    before = {_cell_key(r): r for r in json.load(open(a.before))["rows"] if "error" not in r}
    after = {_cell_key(r): r for r in json.load(open(a.after))["rows"] if "error" not in r}
    keys = sorted(set(before) | set(after))
    changed: list[tuple[str, object, object]] = []
    unchanged: list[str] = []
    only_before: list[str] = []
    only_after: list[str] = []
    triton_alarm: list[str] = []
    for k in keys:
        b, af = before.get(k), after.get(k)
        if b is None:
            only_after.append(k)
            continue
        if af is None:
            only_before.append(k)
            continue
        if b.get("normalized_cfg") != af.get("normalized_cfg"):
            changed.append((k, b.get("normalized_cfg"), af.get("normalized_cfg")))
        else:
            unchanged.append(k)
            # selection-only soundness check: same config but different Triton => the edit
            # touched codegen/source, so the "skip because config-identical" is UNSOUND here.
            if (b.get("triton_sha") and af.get("triton_sha")
                    and b["triton_sha"] != af["triton_sha"]):
                triton_alarm.append(k)

    print(f"=== CONFIG DIFF: {a.before}  ->  {a.after} ===")
    print(f"UNCHANGED (provably perf-invariant, SKIP benching): {len(unchanged)}")
    print(f"CHANGED  (MUST perf-test each — changed != win):    {len(changed)}")
    if only_before or only_after:
        print(f"matrix mismatch: only_before={len(only_before)} only_after={len(only_after)} "
              "(re-record both with the same --dtypes/--splits/kernels)")
    if changed:
        print("\n--- CHANGED cells = the no-regression sweep scope ---")
        for k, bc, ac in changed:
            print(f"  {k}\n      before: {bc}\n      after : {ac}")
    if triton_alarm:
        print("\n!!! SOUNDNESS ALARM: config IDENTICAL but generated Triton CHANGED on "
              f"{len(triton_alarm)} cell(s).")
        print("    The edit is NOT selection-only (it touched codegen/source/normalize), so a")
        print("    config-diff skip is UNSOUND — perf-test these too (the jsd-class edit):")
        for k in triton_alarm:
            print(f"      {k}")
    elif any("triton_sha" in r for r in list(before.values()) + list(after.values())):
        print("\n(Triton-hash check ran: every config-identical cell is also Triton-identical "
              "-> selection-only edit confirmed; the skip is sound.)")
    print(f"\nSCOPE: benchmark the {len(changed)} CHANGED cells; the {len(unchanged)} unchanged "
          "need no re-bench (sound only because this matrix spans the active dtypes/splits).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="config recorder + diff (behavior oracle as a regression scoper)"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="snapshot emitted seed configs across the matrix")
    rec.add_argument("--out", default="/tmp/cfg_snapshot.json")
    rec.add_argument("--dtypes", default="fp32,bf16,fp16")
    rec.add_argument("--splits", default=None,
                     help="override; default train,val,robustness (+test if --include-test)")
    rec.add_argument("--include-test", action="store_true",
                     help="include TEST (off by default — it is the firewall's read-once)")
    rec.add_argument("--triton", action="store_true",
                     help="also hash generated Triton (slow; needed for non-selection-only edits)")
    rec.add_argument("kernels", nargs="*")
    rec.set_defaults(func=cmd_record)

    df = sub.add_parser("diff", help="diff two snapshots -> the CHANGED cells to benchmark")
    df.add_argument("--before", required=True)
    df.add_argument("--after", required=True)
    df.set_defaults(func=cmd_diff)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
