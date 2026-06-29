"""SATURATION CELL: sat_bf16_reduction — bf16 (itemsize=2) full-slice row reduction.

ROUND-2 SATURATION CHECK (task §3a.4): vary a property NOT in the §1 axis list — here
DTYPE / ITEMSIZE — and ask whether the heuristic emits a JUSTIFIED config (every field
traces to a recorded property + a named cap), or an UNJUSTIFIED one (a MISSING AXIS).

The §1 axes (ACCESS/ORIGIN/EXTENT/CARRIED-RESIDENT/CO-RESIDENCY/REUSE/NON-REDUCTION-LOOP/
DIMS/PINNED-GRID) are all held at their COMMON corpus point (standard rolled row-sum,
inner origin, static extent, no carry, single loop, full-width-output apply). The ONLY
thing varied off the corpus is the reduced tensor's DTYPE: bfloat16 (itemsize=2) instead
of the fp32-promoted itemsize=4 every existing reduction probe uses.

CRITICAL IDIOM: the corpus always writes ``x[tile_m, :].to(torch.float32).sum(-1)`` — that
``.to(float32)`` makes the ReductionLowering INPUT fp32, so ``_reduction_input_itemsize``
records 4 at BOTH dtypes (per its own docstring: "Helion fp32-promotes the norm/softmax
family, so this is 4 at both bf16 and fp32"). To actually exercise itemsize=2 the reduction
op must consume bf16 DIRECTLY — so here we reduce straight on the bf16 slice (no ``.to``
before ``.sum``); the accumulator Helion materializes is still fp32. This is the exact
dtype-readiness concern: the residency byte caps (ROW_PERSIST_MAX_BYTES in
``_reduction_rblock`` / ``_resident_tile_cap`` / ``_m_block_cap``) divide the budget by
``fact.itemsize`` — if that is the bf16 INPUT width (2) while the resident accumulator tile
is fp32 (4), the persistent/looped sizing models HALF the true footprint.

WIDE N so the persist-vs-loop boundary is reached: N=131072 (2^17). At itemsize=2 the
single resident row is 131072*2 = 256 KiB > ROW_PERSIST_MAX_BYTES (240 KiB) -> should LOOP;
but if the cap "should" use the fp32 accumulator width (4) the row is 512 KiB and the loop
chunk is half. We dump fact.itemsize + input_load_itemsize + the seed and check whether the
emitted reduction_loops chunk / num_warps trace to a NAMED cap keyed on the FAITHFUL width.

Property point: ACCESS=standard(rolled) x ORIGIN=inner x EXTENT=static-wide(2^17) x
CARRIED-RESIDENT=no x CO-RESIDENCY=single-loop x REUSE=row_reread(full-width apply) x
NON-REDUCTION-LOOP=none x DIMS=2 x PINNED-GRID=none x DTYPE=bf16(itemsize=2).
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
BF16 = torch.bfloat16


@helion.kernel(static_shapes=False)
def bf16_rowsum_normalize(x: torch.Tensor) -> torch.Tensor:
    """x: [M, N] bfloat16. Standard rolled row reduction OVER N with a full-width apply
    (row_reread = True, the persist-prize signal). The reduction consumes bf16 DIRECTLY
    (NO ``.to(float32)`` before ``.sum``) so the ReductionLowering input is bf16 and
    ``fact.itemsize`` records 2 — the saturation lever this cell turns. The accumulator is
    still fp32 (sum upcasts), and the normalize write-back (``x / s``) re-reads the row,
    so this is a persistent-eligible full-width-output standard reduction at itemsize=2."""
    M, N = x.shape
    out = torch.empty([M, N], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M):
        row = x[tile_m, :]                       # [M_BLOCK, N] bf16
        s = row.sum(-1)                          # reduce N — INPUT is bf16 (itemsize=2)
        out[tile_m, :] = row.to(torch.float32) / (s[:, None] + 1.0)  # full-width re-read apply
    return out


def main() -> None:
    print(f"helion={helion.__file__}\n")
    M = 8192
    N = 131072  # 2^17 — wide; single bf16 row = 256 KiB > ROW_PERSIST_MAX_BYTES (240 KiB)
    x = torch.randn(M, N, device=DEV, dtype=BF16)
    intended = {
        "cell": "sat_bf16_reduction",
        "saturation_axis": "dtype/itemsize=2 (bf16)",
        "access": "standard(rolled)",
        "origin": "inner",
        "extent": f"static-wide(N={N}=2^17)",
        "carried_resident": False,
        "co_residency": "single-loop",
        "reuse": "row_reread(full-width apply)",
        "dims": 2,
        "pinned_grid": None,
        "dtype": "bfloat16",
    }
    v = check_kernel("bf16_rowsum_normalize", bf16_rowsum_normalize, (x,), intended)
    import json
    print(json.dumps(v, indent=2, default=repr))


if __name__ == "__main__":
    main()
