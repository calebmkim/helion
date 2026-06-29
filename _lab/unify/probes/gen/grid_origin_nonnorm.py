"""CELL grid_origin_nonnorm: a NON-norm-backward grid-origin collapse.

A per-feature MAX-over-batch collapse (col-wise max across the M/grid axis into a per-column
[N] accumulator), via the two-level grid decomposition
    for cta in hl.tile(M, m_block):          # GRID axis
        acc = <per-feature accumulator over N>
        for mb in hl.tile(cta.begin, cta.end):  # UNBACKED inner re-tile (data-dep extent)
            acc = combine(acc, reduce(x[mb, :], dim=0))
        blocks[cta.id, :] = acc
    return blocks.<finalize>(0)

Structurally an ORIGIN=grid M-collapse (reduce ACROSS programs into a per-feature accumulator,
finalize cross-CTA) but it is NOT rms / layer / group / instance / bias-grad / dyt -- it is a
plain column-max (and, in the variance variant, a column-variance). The question this probes:
does ``grid_reduction_origin`` FIRE (the inner re-tile's range cta.begin..cta.end is unbacked,
the in-graph signature of grid-parallelized reduction) and does the seed size the grid CTA for
occupancy (vs flooring it to 1)?

This is provenance-NOVEL vs gro_divergence.col_energy_collapse (sum-of-squares): max-over-batch
exercises a torch.maximum combine into an hl.full(-inf) accumulator rather than a += sum.
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

DEV = "cuda"


# NON-NORM grid-origin collapse: per-feature MAX over the batch/grid (M) axis.
@helion.kernel(static_shapes=False)
def col_max_collapse(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    m_block = hl.register_block_size(M)
    nb = (M + m_block - 1) // m_block
    blocks = x.new_full([nb, N], float("-inf"), dtype=torch.float32)
    for cta in hl.tile(M, block_size=m_block):
        acc = x.new_full([N], float("-inf"), dtype=torch.float32)
        for mb in hl.tile(cta.begin, cta.end):
            acc = torch.maximum(acc, torch.amax(x[mb, :].to(torch.float32), dim=0))
        blocks[cta.id, :] = acc
    return torch.amax(blocks, dim=0)


def main() -> None:
    print(f"helion={helion.__file__}\n")
    x = torch.randn(4096, 1024, device=DEV, dtype=torch.float32)
    intended = {
        "cell": "grid_origin_nonnorm",
        "access": "user-tiled",
        "origin": "grid",
        "extent": "unbacked-inner-retile",
        "family": "column-max (non-norm)",
        "expect": "grid_reduction_origin True; seed sizes grid CTA for occupancy",
    }
    v = check_kernel("grid_origin_nonnorm__col_max_collapse", col_max_collapse,
                     (x.clone(),), intended)
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
    print(f"fact={obs.get('fact')}")
    print(f"normalized_cfg={obs.get('normalized_cfg')}")


if __name__ == "__main__":
    main()
