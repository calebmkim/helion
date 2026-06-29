"""CELL: sat_argmax_op — SATURATION on reduction-OP identity (argmax / argmin).

The §1 axes model ACCESS / ORIGIN / EXTENT / CARRIED-RESIDENT / CO-RESIDENCY / REUSE /
NON-REDUCTION-LOOP / DIMS / PINNED-GRID. They do NOT model the *reduction-OP identity*:
sum / amax / prod all reduce a single fp32 accumulator (combine arity 1, value-only), but
``torch.argmax`` / ``torch.argmin`` are INDEX reductions — the reduction tree carries TWO
loop-resident values per lane (the running extremum VALUE and its running INDEX), the combine
is the 2-ary argmax-combine, and the materialized output is int64 (8 bytes), NOT fp32.

This cell varies that NOT-modeled property: a plain per-row ``argmax`` over a moderately wide
feature axis (extent 4096), grid over the row axis, the rdim rolled (a STANDARD reduction --
an inner ``reduction=True`` axis the roller rolls into a ``reduction_loops`` loop). Everything
about the kernel SHAPE is identical to a per-row ``sum``/``amax`` row reduction (same ACCESS=
standard, ORIGIN=inner, EXTENT=static-4096, single-load, scalar-ish output, no carried-2D-tile,
no co-residency). The ONLY thing that differs is the OP: argmax.

SATURATION question (§3a.4): does the heuristic FIRE + emit a JUSTIFIED config for the index
reduction, where every emitted field traces to a recorded property + a named cap? Or does the
op identity (different combine arity + int64 output) make some field UNJUSTIFIED -> a MISSING
AXIS (the op-identity / combine-arity axis to add)?

Property point: ACCESS=standard(rolled) x ORIGIN=inner x EXTENT=static-4096 x
CARRIED-RESIDENT=no x CO-RESIDENCY=none x REUSE=none x NON-REDUCTION-LOOP=none x DIMS=2D x
PINNED-GRID=row.  NOT-modeled axis varied: reduction-OP identity = argmax (index, int64,
combine-arity-2).
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROBES = os.path.abspath(os.path.join(_THIS, ".."))
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import check_kernel  # noqa: E402

torch.manual_seed(0)
DEV = "cuda"
F32 = torch.float32


@helion.kernel(static_shapes=False)
def row_argmax(x: torch.Tensor) -> torch.Tensor:
    """Per-row argmax over the feature axis N. Output is int64 [M] (index-carrying).

    The rdim N is an inner ``reduction=True`` axis -> the roller rolls it into a
    ``reduction_loops`` loop = a STANDARD reduction. The combine is the 2-ary
    argmax-combine (running value + running index), distinct from the value-only
    sum/amax combine the §1 corpus exercised.
    """
    M, N = x.shape
    out = torch.empty([M], dtype=torch.int64, device=x.device)
    for tile_m in hl.tile(M):
        out[tile_m] = torch.argmax(x[tile_m, :], dim=-1)
    return out


def main() -> None:
    print(f"helion={helion.__file__}\n")
    M = 4096
    N = 4096  # moderately wide static feature axis
    x = torch.randn(M, N, device=DEV, dtype=F32)
    intended = {
        "cell": "sat_argmax_op",
        "access": "standard(rolled)",
        "origin": "inner",
        "extent": "static-4096",
        "carried_resident": "no",
        "co_residency": "none",
        "reuse": "none",
        "non_reduction_loop": "none",
        "dims": "2D",
        "pinned_grid": "row",
        "not_modeled_axis_varied": "reduction-OP identity = argmax (index, int64, combine-arity-2)",
    }
    v = check_kernel("sat_argmax_op__row_argmax", row_argmax, (x,), intended)
    import json

    print(json.dumps({
        "red": v["red"],
        "reasons": v["reasons"],
        "observed": v["observed"],
    }, indent=2, default=repr))


if __name__ == "__main__":
    main()
