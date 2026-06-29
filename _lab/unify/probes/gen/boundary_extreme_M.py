"""CELL boundary_extreme_M: EXTREME grid (tunable M-axis, M = 2^21 >> 1<<20) crossed
with a NARROW reduction, stressing the ``_m_block_cap`` / raised-``autotuner_min`` floor
interplay in the STANDARD reduction track.

The boundary stressed (§ _m_block_cap vs raise_grid_block_minimums):
  - M is a TUNABLE grid axis (``hl.tile(M)`` unpinned), so it IS a ``block_sizes`` entry and
    ``raise_grid_block_minimums`` RAISES its ``autotuner_min`` (a huge-M grid is clamped to
    <= n_cus*64 blocks, so the per-program row count floor is bumped well above 1).
  - The reduction is a STANDARD (Helion-rolled) reduction over a NARROW inner axis N -- the
    rdim rides ``reduction_loops``, NOT ``block_sizes``, so M is the only tunable block_sizes
    tile and is the axis whose ``autotuner_min`` gets raised.
  - The store ``out[tile_m, :] = x[tile_m, :] / denom`` is FULL-WIDTH over the reduction
    extent axis (the normalize loop), so ``fact.full_width_output`` is True and
    ``_m_block_cap`` ENGAGES: it bounds M_BLOCK to keep the resident ``[M_BLOCK, rdim]`` live
    set inside ``ROW_PERSIST_MAX_BYTES``.

The question this cell answers: when the huge-M ``autotuner_min`` floor and the
full-width-output ``_m_block_cap`` collide on the SAME tunable grid axis, does the seed
(a) CRASH in sizing / normalize, or (b) FLOOR1_TILED a real reduction (a tunable reduction
block_sizes axis driven to block_size=1 with extent>1), or (c) land a JUSTIFIED config
(the M-axis widen = ``max(floor, min(own, fits, occ, _m_block_cap))`` traces to the
autotuner_min floor + the named residency/occupancy caps; the rolled rdim rides
reduction_loops, not a tunable tile, so FLOOR1_TILED cannot apply to it).

Property point (§1, all MODELED axes -- this is a TOTALITY/boundary cell, NOT a saturation
cell, so no not-modeled property is varied):
  ACCESS         = standard (Helion-rolled rdim -> reduction_loops, not a block_sizes tile)
  ORIGIN         = inner (the reduction axis N is the innermost, NOT the grid axis)
  EXTENT         = static{ M = 2^21 EXTREME grid, N = 8 NARROW reduction }
  CARRIED-RESIDENT = no carried [M_BLOCK, R_BLOCK] accumulator (persistent narrow row)
  CO-RESIDENCY   = single reduction
  REUSE          = row re-read (reduce then normalize -> the row is loaded, reduced, re-read)
  NON-REDUCTION-LOOP = the normalize/apply loop over N (full-width store)
  DIMS           = 2
  PINNED-GRID    = none (M is the UNPINNED tunable grid axis -- the whole point: its
                   autotuner_min is the lever raise_grid_block_minimums pulls)

NOT strided-dim0 (out-of-scope): the reduction axis N is the INNERMOST contiguous axis
(stride == itemsize), not a grid dim-0 with a strided access pattern over reduced elems.
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


# EXTREME grid (tunable M, M = 2^21), NARROW standard reduction over N, full-width normalize
# store (so full_width_output is True and _m_block_cap engages against the raised autotuner_min).
@helion.kernel(static_shapes=False)
def extreme_m_narrow_reduce(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x, dtype=torch.float32)
    for tile_m in hl.tile(M):  # UNPINNED tunable grid axis -- autotuner_min raised for huge M
        row = x[tile_m, :].to(torch.float32)
        denom = row.sum(-1, keepdim=True) + 1.0  # STANDARD rolled reduction over NARROW N
        out[tile_m, :] = row / denom  # FULL-WIDTH store over the reduction-extent axis
    return out


def main() -> None:
    print(f"helion={helion.__file__}\n")
    M = 2 * 1024 * 1024  # 2,097,152 = 2^21, EXTREME (>> 1<<20)
    N = 8  # NARROW reduction extent
    x = torch.randn(M, N, device=DEV, dtype=F32)
    intended = {
        "cell": "boundary_extreme_M",
        "access": "standard (rolled rdim -> reduction_loops)",
        "origin": "inner (reduction axis N innermost)",
        "extent": "static{M=2^21 extreme grid, N=8 narrow reduction}",
        "carried_resident": "no",
        "co_residency": "single reduction",
        "reuse": "row re-read (reduce then normalize)",
        "non_reduction_loop": "normalize loop over N (full-width store)",
        "dims": 2,
        "pinned_grid": "none (M is the UNPINNED tunable grid axis)",
        "expect": "JUSTIFIED: M-axis widen = max(autotuner_min floor, min(own, fits, occ, "
        "_m_block_cap)); narrow N rolls into reduction_loops (not a tunable tile, so "
        "FLOOR1_TILED cannot apply); no crash/floor of a real reduction",
    }
    v = check_kernel("boundary_extreme_M__extreme_m_narrow_reduce",
                     extreme_m_narrow_reduce, (x,), intended)
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
