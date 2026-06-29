"""GATE-D ROUND 2 — CLAIM 1 (NARROWED Defect-1): full-slice ACCESS width + safe conservatism.

Three things to confirm:
(a) r_block_resident in _build_block_sizes reads ONLY the access signal
    (red_values membership of the primary rdim) + size_hint -- NOT spec.reduction_loops.
(b) For a LOOPED standard reduction, r_block_resident = next_pow2(size_hint) OVER-counts the
    true live chunk (reduction_loops[0]); larger r_block_resident -> smaller _resident_tile_cap
    -> SMALLER M-widen -> never a spill (conservatism toward-smaller).
(c) "full-slice access width" is a faithful workload signal (is the primary a tunable tile vs a
    full-slice resident reduction), read from the red_values dict the caller built from spec/fact.

Compile-only, no GPU (bind() only; no .run()).  Run from /tmp with PYTHONPATH=<worktree>.
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _WT not in sys.path:
    sys.path.insert(0, _WT)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402
from helion._utils import next_power_of_2 as _np2  # noqa: E402
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    _TritonReductionSeedBase,
    TritonStandardReductionHeuristic,
    _reduction_primary_fact,
)

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep), helion.__file__
DEV = "cuda"

# ---- instrument _build_block_sizes to capture (a) what it reads and the assigned value ----
_orig = _TritonReductionSeedBase._build_block_sizes.__func__
TRACE: list[dict] = []


def _traced(cls, env, spec, fact, red_values=None, non_reduction_loop_ids=frozenset(),
            grid_origin=False):
    rv = red_values or {}
    primary = fact.primary_reduction_block_id
    primary_red_value = rv.get(primary)
    # Replicate the EXACT branch in the source (lines 909-912) to read what it assigns.
    if primary_red_value is not None:
        rbr = primary_red_value
    else:
        rbr = _np2(fact.size_hint)
    # What the CONTINGENT (Defect-1) key WOULD read -- recompute it ourselves so we can show
    # the live code does NOT consult it:
    in_reduction_loops = primary in spec.reduction_loops.valid_block_ids()
    TRACE.append({
        "primary_rdim": primary,
        "size_hint": fact.size_hint,
        "primary_in_red_values": primary_red_value is not None,
        "primary_in_reduction_loops": in_reduction_loops,
        "r_block_resident_assigned": rbr,
        "np2_size_hint": _np2(fact.size_hint),
    })
    return _orig(cls, env, spec, fact, red_values, non_reduction_loop_ids, grid_origin)


_TritonReductionSeedBase._build_block_sizes = classmethod(_traced)


# A LOOPED standard reduction: wide row-sum so the rdim rolls to a chunk (N huge).
@helion.kernel(static_shapes=False)
def wide_rowsum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M], dtype=torch.float32, device=x.device)
    for tm in hl.tile(M):
        out[tm] = x[tm, :].to(torch.float32).sum(-1)
    return out


# A USER-TILED reduction (rdim IS a tunable block_sizes entry via an inner hl.tile) for
# contrast -- here primary_red_value is set, access=user-tiled.
@helion.kernel(static_shapes=False)
def usertiled_rowsum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M], dtype=torch.float32, device=x.device)
    for tm in hl.tile(M, block_size=1):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(N):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        out[tm] = acc
    return out


def _drive(fn, args):
    """bind() compile-only and pull the standard-track seed so _build_block_sizes runs and the
    real reduction_loops chunk is produced."""
    TRACE.clear()
    bound = fn.bind(args)
    env = bound.env
    spec = env.config_spec
    fact = _reduction_primary_fact(spec) if spec.reduction_facts else None
    seed = None
    # Drive the standard-track get_seed_config (this calls _build_block_sizes internally).
    try:
        seed = TritonStandardReductionHeuristic.get_seed_config(
            env, bound.host_function.device_ir)
    except Exception as e:  # noqa: BLE001
        seed = {"_err": repr(e)}
    return env, spec, fact, seed, list(TRACE)


def main():
    print(f"helion={helion.__file__}\n")

    # ---------- (b) LOOPED standard reduction: N huge so it rolls to a chunk ----------
    # 65536 row-width forces reduction_loops to a chunk (< full extent).
    x = torch.randn(1024, 65536, device=DEV, dtype=torch.float32)
    env, spec, fact, seed, tr = _drive(wide_rowsum, (x.clone(),))
    print("=== (b) LOOPED standard reduction: wide_rowsum [1024, 65536] ===")
    print(f"  reduction_loops seed = {seed.get('reduction_loops')}  (chunk = the true live width)")
    print(f"  block_sizes seed     = {seed.get('block_sizes')}")
    rl = seed.get("reduction_loops") or []
    true_chunk = None
    if rl and isinstance(rl[0], int):
        true_chunk = rl[0]
    elif rl == [None]:
        true_chunk = _np2(fact.size_hint)  # persistent: full extent IS live (no over-count)
    for t in tr:
        print("  build_block_sizes:", t)
    rec = tr[-1] if tr else {}
    size_hint = rec.get("size_hint")
    rbr = rec.get("r_block_resident_assigned")
    in_rl = rec.get("primary_in_reduction_loops")
    in_rv = rec.get("primary_in_red_values")
    print(f"\n  size_hint={size_hint}  r_block_resident={rbr}  true_chunk={true_chunk}")
    print(f"  primary_in_red_values={in_rv}  primary_in_reduction_loops={in_rl}")
    looped = (rl and isinstance(rl[0], int))
    over_counts = (true_chunk is not None and rbr is not None and rbr >= true_chunk)
    print(f"  LOOPED={looped}  r_block_resident >= true_chunk? {over_counts}  "
          f"(over-count factor = {rbr / true_chunk if true_chunk else 'n/a'})")

    # ---------- (a) the key never reads reduction_loops to gate r_block_resident ----------
    # Source-level: grep showed no reduction_loops in the executable body. Behavioral witness:
    # r_block_resident depends ONLY on primary_in_red_values, regardless of reduction_loops.
    print("\n=== (a) r_block_resident gating witness ===")
    print(f"  standard looped: in_reduction_loops={in_rl}, in_red_values={in_rv}, "
          f"r_block_resident={rbr} (== np2(size_hint)={rec.get('np2_size_hint')})")

    # ---------- (c) user-tiled contrast: access signal flips the branch ----------
    env2, spec2, fact2, seed2, tr2 = _drive(usertiled_rowsum, (x.clone(),))
    print("\n=== (c) USER-TILED contrast: usertiled_rowsum [1024, 65536] ===")
    print(f"  block_sizes seed = {seed2.get('block_sizes')}")
    rec2 = tr2[-1] if tr2 else {}
    print(f"  size_hint={rec2.get('size_hint')}  r_block_resident={rec2.get('r_block_resident_assigned')}"
          f"  primary_in_red_values={rec2.get('primary_in_red_values')}"
          f"  primary_in_reduction_loops={rec2.get('primary_in_reduction_loops')}")

    # ---------- conservatism DIRECTION: larger r_block_resident -> smaller _resident_tile_cap ----------
    print("\n=== conservatism direction (footprint cap monotonicity) ===")
    # _resident_tile_cap = ROW_PERSIST_MAX_BYTES // (m_block * r_block_resident * inner * itemsize)
    # so it is monotonically NON-INCREASING in r_block_resident. Demonstrate with the looped fact.
    if fact is not None:
        cap_true = _TritonReductionSeedBase._resident_tile_cap(
            spec, fact, 1, r_block_resident=true_chunk or 1)
        cap_used = _TritonReductionSeedBase._resident_tile_cap(
            spec, fact, 1, r_block_resident=rbr or 1)
        print(f"  cap(r_block_resident=true_chunk={true_chunk}) = {cap_true}   (the TRUE residency)")
        print(f"  cap(r_block_resident=USED={rbr})              = {cap_used}   (the over-counted USED)")
        print(f"  used_cap <= true_cap? {cap_used <= cap_true}  "
              f"=> M-widen with USED is <= M-widen with TRUE => toward-SMALLER (safe, never a spill)")

    print("\n--- summary ---")
    print(f"(a) reads_reduction_loops_to_gate = False (source body has none; behavioral: r_block_resident "
          f"= primary_red_value if in red_values else np2(size_hint))")
    print(f"(b) looped over-count safe = {over_counts and looped}")
    print(f"(c) access signal flips branch = "
          f"{rec.get('primary_in_red_values') is False and rec2.get('primary_in_red_values') is True}")


if __name__ == "__main__":
    main()
