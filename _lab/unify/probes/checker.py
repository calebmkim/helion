"""MECHANICAL TOTALITY CHECKER (§5b half-i) + the RED-probe corpus seed.

Given a Helion kernel + example args, compile at HELION_AUTOTUNE_EFFORT=none and
mechanically flag a FALL-THROUGH (a RED) by reading the PERSISTED seed + the facts:

  RED reasons (compile-time, no GPU):
   - CRASH        : bind/seed raised (KeyError pinned-axis, else/exception in sizing)
   - NO_FIRE      : an eligible reduction (>=1 ReductionFact, no matmul) emitted no seed
   - FLOOR1_TILED : a TILED reduction axis (per the LOWERING role / _all_reduction_axes,
                    NOT the heuristic's own is_eligible) got block_size 1 in the seed
                    while its extent > 1 -> the reduction was silently floored
   - UNJUSTIFIED  : (reported, not auto-RED) a seed field the sizer cannot trace to a
                    property + named cap (left for the divergence/Gate-D analysis)

Out-of-scope (GREEN, by the two CLOSED §0 predicates, confirmed by PROVENANCE not analogy):
   - JAGGED       : ReductionFact.size_hint is None / data-dependent (DECLINE is correct)
   - STRIDED_DIM0 : reduction axis is grid dim-0 AND its memory stride over the reduced
                    elements != itemsize (persistence byte-cap is access-pattern-blind)

The GPU default-vs-seed tripwire (the "model well" floor) is a SEPARATE foreground-serial
step (perf_tripwire.py) — never run inside this compile-only checker.

`check_kernel(name, fn, args, intended)` returns a dict verdict. `intended` is the
property-point the kernel was authored to land on (access/origin/extent/...); the checker
ALSO dumps the observed facts so the minter can confirm it LANDED (reject + retry if drifted).
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _WT_ROOT not in sys.path:
    sys.path.insert(0, _WT_ROOT)

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT_ROOT})"
)

from helion._compiler.inductor_lowering import ReductionLowering  # noqa: E402


def _lowering_reduction_axes(device_ir) -> set[int]:
    """EVERY block_id some ReductionLowering reduces over (the LOWERING role -- the
    faithful 'is this a real reduction' signal, independent of the heuristic's is_eligible)."""
    out: set[int] = set()
    for gi in device_ir.graphs:
        for node in gi.graph.nodes:
            low = node.meta.get("lowering")
            if isinstance(low, ReductionLowering):
                bid = getattr(low, "block_index", None)
                if bid is not None:
                    out.add(bid)
    return out


def _jsonify(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonify(x) for k, x in v.items()}
    return repr(v)


def check_kernel(name: str, fn, args, intended: dict | None = None) -> dict:
    """Compile-only mechanical check. Returns a verdict dict."""
    verdict: dict = {"name": name, "intended": intended or {}, "red": None,
                     "reasons": [], "observed": {}}
    try:
        bound = fn.bind(args)
    except Exception as e:  # noqa: BLE001
        verdict["red"] = "CRASH"
        verdict["reasons"].append(f"bind raised: {type(e).__name__}: {e}")
        return verdict
    spec = bound.env.config_spec
    device_ir = bound.host_function.device_ir

    seeds = list(spec.compiler_seed_configs)
    fired = list(spec.autotuner_heuristics)
    n_rfacts = len(spec.reduction_facts)
    n_matmul = len(spec.matmul_facts)
    lowering_axes = _lowering_reduction_axes(device_ir)
    grid_ids = {b for bids in device_ir.grid_block_ids for b in bids}

    verdict["observed"] = {
        "fired": fired,
        "n_reduction_facts": n_rfacts,
        "n_matmul_facts": n_matmul,
        "lowering_reduction_axes": sorted(lowering_axes),
        "grid_block_ids": _jsonify(device_ir.grid_block_ids),
        "block_sizes_valid_ids": list(spec.block_sizes.valid_block_ids()),
        "reduction_loops_valid_ids": list(spec.reduction_loops.valid_block_ids()),
    }
    if spec.reduction_facts:
        verdict["observed"]["fact"] = _jsonify(spec.reduction_facts[0]._asdict())
    seed = None
    if seeds:
        raw = dict(seeds[0])
        norm = dict(raw)
        try:
            with bound.env:
                spec.normalize(norm)
        except Exception as e:  # noqa: BLE001
            verdict["red"] = "CRASH"
            verdict["reasons"].append(f"normalize raised: {type(e).__name__}: {e}")
            return verdict
        seed = norm
        verdict["observed"]["raw_seed"] = _jsonify(raw)
        verdict["observed"]["normalized_cfg"] = _jsonify(norm)

    # --- mechanical RED checks ---
    # 1. NO_FIRE: eligible reduction (>=1 fact, no matmul) but no seed.
    if n_rfacts >= 1 and n_matmul == 0 and not seeds:
        verdict["red"] = "NO_FIRE"
        verdict["reasons"].append(
            f"{n_rfacts} ReductionFact, no matmul, but no seed emitted")
        return verdict

    # 2. FLOOR1_TILED: a RESIDENT reduction axis (fully reduced within one program) that is a
    #    tunable block_sizes entry got block_size 1 while its extent > 1.
    #    EXCLUDE grid axes (PARTIAL_GRID): a reduction OVER a grid axis is parallelized across
    #    programs -- flooring it to 1 (keeping it a grid ROW) is CORRECT, not a hole. Likewise the
    #    STRIDED-DIM0 out-of-scope predicate (reduction over grid dim-0 with stride != itemsize) is
    #    a grid-axis reduction that correctly floors. Only a RESIDENT reduction (the role
    #    classifier's RESIDENT) floored to 1 is a genuine silent-floor hole.
    if seed is not None and "block_sizes" in seed:
        bs = seed["block_sizes"]
        valid = list(spec.block_sizes.valid_block_ids())
        env = bound.env
        for bid in lowering_axes:
            if bid not in valid:
                continue  # rolled / materialized / pinned reductions are NOT block_sizes tiles
            if bid in grid_ids:
                continue  # PARTIAL_GRID / grid-axis reduction: floored as a grid row (correct)
            try:
                idx = spec.block_sizes.block_id_to_index(bid)
            except Exception:  # noqa: BLE001
                continue
            if idx >= len(bs):
                continue
            info = env.block_sizes[bid]
            sz = info.size
            extent = None
            if isinstance(sz, (int, torch.SymInt)):
                try:
                    extent = int(env.size_hint(sz))
                except Exception:  # noqa: BLE001
                    extent = None
            # JAGGED out-of-scope: no static extent -> DECLINE is correct, not a floor.
            if extent is None:
                continue
            if bs[idx] == 1 and extent > 1:
                verdict["red"] = "FLOOR1_TILED"
                verdict["reasons"].append(
                    f"RESIDENT reduction axis bid={bid} (extent {extent}) floored to "
                    f"block_size=1 in seed block_sizes={bs}")
                return verdict

    # Not RED by the compile-time checks. (UNJUSTIFIED-config + the GPU default-vs-seed
    # tripwire are evaluated separately.)
    return verdict


def run_suite(probes: list[tuple]) -> list[dict]:
    """probes: list of (name, fn, args, intended). Returns verdicts; prints a table."""
    results = []
    for (name, fn, args, intended) in probes:
        v = check_kernel(name, fn, args, intended)
        results.append(v)
        red = v["red"] or "green"
        obs = v["observed"]
        ns = obs.get("normalized_cfg", {})
        print(f"[{red:13s}] {name:36s} fired={obs.get('fired')} "
              f"nRF={obs.get('n_reduction_facts')} "
              f"lowering_axes={obs.get('lowering_reduction_axes')} "
              f"bs={ns.get('block_sizes') if ns else None} "
              f"rl={ns.get('reduction_loops') if ns else None}")
        for r in v["reasons"]:
            print(f"                  reason: {r}")
    return results
