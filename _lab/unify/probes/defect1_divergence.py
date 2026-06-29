"""DEFECT-1 DIVERGENCE PROBE — witness the contingent-rolling lie directly.

Defect 1: `_build_block_sizes` chooses r_block_resident by
  if primary in reduction_loops:        next_pow2(size_hint)   # rolled -> full-width
  elif primary_red_value is not None:   primary_red_value      # user-tiled
  else:                                 1                       # MATERIALIZED full-slice -> "costs nothing" (a LIE)

The key (`primary in reduction_loops`) is CONTINGENT: the roller rolls at most ONE
reduction per loop. So a full-slice reduction's r_block_resident depends on whether
ANOTHER reduction also exists in the loop. This probe INSTRUMENTS _build_block_sizes to
dump (primary_rdim, in_reduction_loops, r_block_resident, real_full_width) and builds two
kernels that differ ONLY in whether a SECOND full-slice reduction shares the loop:

  K_alone   : one full-slice sum over N        -> N rolled  -> r_block_resident = pow2(N)  (truthful)
  K_pair    : two full-slice sums over N and P -> one rolls, one MATERIALIZES -> the
              materialized axis hits the `else` -> r_block_resident = 1  (the LIE: it is
              full-width P resident)

PROXY value = r_block_resident the contingent branch assigns.
REAL property = the resident reduction width = next_pow2(extent) of that materialized axis.
If they DISAGREE on K_pair's materialized axis, Defect 1 is confirmed (the proxy diverges).

Compile-only, no GPU. Run from /tmp with PYTHONPATH=<worktree>.
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
)

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep)

DEV = "cuda"
DUMP: list[dict] = []

# Instrument _build_block_sizes to capture the contingent r_block_resident decision.
_orig = _TritonReductionSeedBase._build_block_sizes.__func__


def _traced(cls, env, spec, fact, red_values=None, non_reduction_loop_ids=frozenset()):
    rv = red_values or {}
    primary = fact.primary_reduction_block_id
    in_rl = primary in spec.reduction_loops.valid_block_ids()
    primary_red_value = rv.get(primary)
    if in_rl:
        rbr = _np2(fact.size_hint)
        route = "rolled->pow2(size_hint)"
    elif primary_red_value is not None:
        rbr = primary_red_value
        route = "usertiled->primary_red_value"
    else:
        rbr = 1
        route = "MATERIALIZED->else->1 (THE LIE)"
    real_full_width = _np2(fact.size_hint)
    DUMP.append({
        "primary_rdim": primary,
        "in_reduction_loops": in_rl,
        "size_hint": fact.size_hint,
        "r_block_resident_PROXY": rbr,
        "real_full_width_resident": real_full_width,
        "route": route,
        "diverges": (rbr != real_full_width) and not in_rl and primary_red_value is None,
    })
    return _orig(cls, env, spec, fact, red_values, non_reduction_loop_ids)


_TritonReductionSeedBase._build_block_sizes = classmethod(_traced)


@helion.kernel(static_shapes=False)
def k_alone(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        a = (x[tile_m, :].to(torch.float32) ** 2).sum(-1)   # full-slice over N
        out[tile_m, 0] = a
    return out


@helion.kernel(static_shapes=False)
def k_pair(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Two full-slice reductions over DIFFERENT axes (N and P) in one loop. The roller
    rolls at most one; the other MATERIALIZES -> Defect-1 else -> r_block_resident=1."""
    M, N = x.shape
    _, P = w.shape
    out = torch.empty([M, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        a = (x[tile_m, :].to(torch.float32) ** 2).sum(-1)   # full-slice over N (extent 4096)
        b = (w[tile_m, :].to(torch.float32)).sum(-1)        # full-slice over P (extent 2048)
        out[tile_m, 0] = a + b
    return out


def main():
    print(f"helion={helion.__file__}\n")
    x = torch.randn(4096, 4096, device=DEV, dtype=torch.float32)
    w = torch.randn(4096, 2048, device=DEV, dtype=torch.float32)

    DUMP.clear()
    k_alone(x.clone())
    print("=== K_alone (one full-slice sum over N=4096) ===")
    for d in DUMP:
        print("  ", d)

    DUMP.clear()
    k_pair(x.clone(), w.clone())
    print("\n=== K_pair (two full-slice sums over N=4096 and P=2048) ===")
    for d in DUMP:
        print("  ", d)
    diverged = [d for d in DUMP if d["diverges"]]
    print(f"\nDEFECT-1 DIVERGENCE: {'CONFIRMED' if diverged else 'not observed'} "
          f"({len(diverged)} materialized axis told r_block_resident=1 while real "
          f"full-width > 1)")


if __name__ == "__main__":
    main()
