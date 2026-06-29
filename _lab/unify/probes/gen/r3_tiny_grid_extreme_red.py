"""CELL r3_tiny_grid_extreme_red: a TINY grid (M < num_sm) crossed with an EXTREME
reduction (N >= 1M). Stresses OCCUPANCY (grid < num_sm) against the LOOPED extreme
reduction -- does the seed floor/loop sanely, or does it UNDER-OCCUPY / collapse?

BOUNDARY stressed (occupancy floor vs the looped extreme rdim):
  - M = 64 is a TUNABLE grid axis (``hl.tile(M)`` UNPINNED -> a ``block_sizes`` entry).
    64 << num_sm (132 on H100) and FAR below the occupancy floor ``num_sm * MIN_WAVES``
    (132*8 = 1056), so ``_m_axis_occupancy_cap`` = ``grid // (num_sm*MIN_WAVES)`` = 64//1056
    = 0 -> ``max(1, prev_power_of_2(max(1,0)))`` = 1. The M-axis widen
    ``max(floor=1, min(own=64, fits, occ=1, _m_block_cap))`` therefore FLOORS to 1: the
    seed must NOT widen/collapse the already-tiny grid (widening 64 rows into one program
    would leave a 1-program launch -- catastrophic under-occupancy). The 64-program grid is
    inherent to the WORKLOAD (M=64), not something the heuristic should "fix" by collapse.
  - The reduction over N=2^20=1,048,576 is a STANDARD (Helion-rolled) reduction riding
    ``reduction_loops``. The fp32-accumulator resident row is 1*1048576*4 = 4 MiB, FAR over
    ROW_PERSIST_MAX_BYTES (240 KiB) -> NOT extent_held -> it must LOOP:
    ``reduction_loops=[r_block]`` with ``r_block = min(LOOPED_CHUNK=16384,
    prev_power_of_2(240KiB/(m*itemsize*ff)))`` -- a sane bounded chunk, never [None]
    (persistent would be a 4 MiB resident spill) and never floored to 1 (extreme N).

The question this cell answers: with a sub-num_sm grid AND a heavy looped extreme
reduction colliding, does the seed (a) CRASH, (b) FLOOR1_TILED a real reduction (N rolls
into reduction_loops, NOT a block_sizes tile, so FLOOR1_TILED cannot apply to it; M is the
only block_sizes tile and is a grid axis, not a reduction axis), (c) NO_FIRE, or (d) emit
an UNJUSTIFIED config -- specifically GRID COLLAPSE (M widened to its full extent 64,
leaving 1 program). The expected JUSTIFIED outcome: M floored to 1 (occupancy cap = 1,
so block_sizes=[1]) keeping the 64-program grid, reduction_loops=[r_block<=16384] looped,
num_warps=32 (rnumel>16384). Under-occupancy of 64 programs on 132 SMs is the WORKLOAD,
not a heuristic hole -- the hole would be the heuristic COLLAPSING the grid further.

Property point (all MODELED axes; TOTALITY/boundary cell, not saturation):
  ACCESS         = standard (Helion-rolled rdim -> reduction_loops, not a block_sizes tile)
  ORIGIN         = inner (the reduction axis N is innermost, NOT the grid axis)
  EXTENT         = static{ M = 64 TINY grid (< num_sm), N = 2^20 EXTREME reduction (>= 1M) }
  CARRIED-RESIDENT = no carried [M_BLOCK, R_BLOCK] accumulator (rolled streaming sum)
  CO-RESIDENCY   = single reduction
  REUSE          = row re-read (reduce then normalize -> the row is loaded, reduced, re-read)
  NON-REDUCTION-LOOP = the normalize/apply loop over N (full-width store)
  DIMS           = 2
  PINNED-GRID    = none (M is the UNPINNED tunable grid axis -- the occupancy lever's subject)

NOT strided-dim0 / NOT jagged (out-of-scope, GREEN by provenance): the reduction axis N is
the INNERMOST contiguous axis (stride == itemsize), static extent (not data-dependent /
size None). The grid axis is M (dim-0) but it is NOT the reduction axis -- the reduction is
over dim-1, contiguous.
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


# TINY grid (tunable M=64 << num_sm), EXTREME standard reduction over N=2^20, full-width
# normalize store (so row_reread is True and the persist/loop boundary is reached). The grid
# is sub-num_sm: the occupancy cap must FLOOR the M widen to 1 (no collapse), and the 4 MiB
# fp32 row must LOOP (reduction_loops=[r_block], not [None]).
@helion.kernel(static_shapes=False)
def tiny_grid_extreme_reduce(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x, dtype=torch.float32)
    for tile_m in hl.tile(M):  # UNPINNED tunable grid axis; M=64 << num_sm -> occupancy floor
        row = x[tile_m, :].to(torch.float32)
        denom = row.sum(-1, keepdim=True) + 1.0  # STANDARD rolled reduction over EXTREME N
        out[tile_m, :] = row / denom  # FULL-WIDTH normalize store (row_reread = True)
    return out


def main() -> None:
    print(f"helion={helion.__file__}\n")
    M = 64  # TINY grid: < num_sm (132 on H100), far below num_sm*MIN_WAVES (1056)
    N = 1024 * 1024  # 1,048,576 = 2^20, EXTREME reduction (>= 1M)
    x = torch.randn(M, N, device=DEV, dtype=F32)
    intended = {
        "cell": "r3_tiny_grid_extreme_red",
        "access": "standard (rolled rdim -> reduction_loops)",
        "origin": "inner (reduction axis N innermost)",
        "extent": "static{M=64 tiny grid (<num_sm), N=2^20 extreme reduction (>=1M)}",
        "carried_resident": "no",
        "co_residency": "single reduction",
        "reuse": "row re-read (reduce then normalize)",
        "non_reduction_loop": "normalize loop over N (full-width store)",
        "dims": 2,
        "pinned_grid": "none (M is the UNPINNED tunable grid axis)",
        "expect": "JUSTIFIED: M floored to 1 (occupancy cap=1, NO grid collapse, "
        "block_sizes=[1] keeps the 64-program grid); reduction_loops=[r_block<=16384] "
        "(4 MiB fp32 row > ROW_PERSIST_MAX_BYTES -> LOOP, never [None], never floored to 1); "
        "num_warps=32 (rnumel>16384). Under-occupancy of 64 progs is the WORKLOAD not a hole "
        "-- the hole would be a grid widen/collapse to full extent (M=64 -> 1 program).",
    }
    v = check_kernel("r3_tiny_grid_extreme_red__tiny_grid_extreme_reduce",
                     tiny_grid_extreme_reduce, (x,), intended)
    import json

    obs = v["observed"]
    print(f"RED={v['red']}")
    print(f"reasons={v['reasons']}")
    print(f"fired={obs.get('fired')}")
    print(f"n_reduction_facts={obs.get('n_reduction_facts')}")
    print(f"n_matmul_facts={obs.get('n_matmul_facts')}")
    print(f"lowering_reduction_axes={obs.get('lowering_reduction_axes')}")
    print(f"grid_block_ids={obs.get('grid_block_ids')}")
    print(f"block_sizes_valid_ids={obs.get('block_sizes_valid_ids')}")
    print(f"reduction_loops_valid_ids={obs.get('reduction_loops_valid_ids')}")
    print(f"fact={json.dumps(obs.get('fact'), indent=2, default=repr)}")
    print(f"raw_seed={obs.get('raw_seed')}")
    print(f"normalized_cfg={obs.get('normalized_cfg')}")


if __name__ == "__main__":
    main()
