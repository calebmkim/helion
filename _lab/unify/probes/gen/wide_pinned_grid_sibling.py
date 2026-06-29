"""CELL: wide_pinned_grid_sibling

A grid-pinned-M kernel (token axis pinned ``block_size=1``, the per_token_group idiom)
with a WIDE TUNABLE fan-out sibling (the groups axis ``G``, ``hl.tile(G)`` unpinned) beside
a SMALL PINNED reduction (the per-group element axis ``C``, pinned full-extent ``block_size=C``,
reduced via ``amax(-1)``).

Property point (§1):
  ACCESS         = full-slice (pinned full-extent reduction over C, x[m, g, :])
  ORIGIN         = inner-axis pinned reduction (C is the reduced axis, NOT a grid axis)
  EXTENT         = static (C small)
  CARRIED-RESIDENT = no carried [M_BLOCK, R_BLOCK] accumulator (per-group scalar output)
  CO-RESIDENCY   = single reduction
  DIMS           = 3
  PINNED-GRID    = M pinned block_size=1; G is the WIDE TUNABLE grid fan-out sibling

The probe target: G is a tunable grid (m_block_ids) axis that hits the
``elif bs_spec.block_id in fact.m_block_ids:`` FOOTPRINT-WIDEN branch in ``_build_block_sizes``.
Because the reduction R_BLOCK (=C) is SMALL, ``_resident_tile_cap`` returns large, so the
footprint budget alone would permit a very wide G tile. The question this cell exercises:
does the m_block_ids footprint-widen OVER-WIDEN G, or does the occupancy guard
(``_m_axis_occupancy_cap``: never collapse the post-widen grid below ``num_sm * MIN_WAVES``)
floor it correctly? Distinct from P2 (where the sibling G is itself REDUCED -- a
tunable-grid-reduction); here G is a pure FAN-OUT (the output keeps the G dimension).

Unlike STRIDED-DIM0 (out-of-scope): the reduction axis here is the INNERMOST axis C
(contiguous, stride == itemsize), NOT a grid dim-0 with a strided access pattern.
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
F32 = torch.float32


# M pinned (block_size=1, grid-pinned token axis), G WIDE tunable grid fan-out sibling,
# C SMALL pinned full-extent reduction (the per_token_group per-group amax quant idiom).
@helion.kernel(static_shapes=False)
def wide_pinned_grid_sibling(x: torch.Tensor) -> torch.Tensor:
    M, G, C = x.shape
    C = hl.specialize(C)
    out = torch.empty([M, G], dtype=torch.float32, device=x.device)
    for tile_m, tile_g, tile_c in hl.tile([M, G, C], block_size=[1, None, C]):
        # SMALL pinned reduction over C -> per-group scalar; G is a WIDE non-reduced fan-out
        # sibling kept in the output (the per_token_group quant scale pattern).
        per_g = torch.amax(x[tile_m, tile_g, tile_c].to(torch.float32).abs(), dim=-1)
        out[tile_m, tile_g] = per_g
    return out


def main():
    print(f"helion={helion.__file__}\n")
    # M tokens grid-pinned; G WIDE sibling (8192); C SMALL pinned reduction (128).
    x = torch.randn(4096, 8192, 128, device=DEV, dtype=BF16)
    intended = {
        "access": "full-slice (pinned full-extent C)",
        "origin": "inner-axis pinned reduction (C)",
        "pinned_grid": "M block_size=1",
        "sibling": "WIDE tunable grid fan-out G (non-reduced)",
        "dims": 3,
    }
    v = check_kernel("wide_pinned_grid_sibling", wide_pinned_grid_sibling, (x,), intended)
    import json

    print(json.dumps(v, indent=2, default=repr))


if __name__ == "__main__":
    main()
