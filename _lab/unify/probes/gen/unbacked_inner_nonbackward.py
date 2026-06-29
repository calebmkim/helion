"""GEN cell: unbacked_inner_nonbackward.

A kernel whose INNER reduction extent is UNBACKED (a data-dependent range via
``hl.tile(cta.begin, cta.end)``) but is NOT a grad-collapse: a chunked segment-sum.

The outer grid loop tiles a values vector into fixed CTAs; the inner loop re-tiles
each CTA's ``[cta.begin, cta.end)`` slab (an UNBACKED SymInt range) and sums the 1-D
values over that data-dependent extent into a per-CTA scalar. The reduction axis IS
the unbacked inner re-tile (a ``block_sizes`` tunable entry), reducing over the SAME
grid dim-0 the CTA tiled -- structurally an M-collapse-shaped re-tile, but it produces
a per-CTA SCALAR (chunk sum), not a per-feature [N] accumulator finalized cross-CTA,
and there is no backward/grad provenance.

Property point (§1):
  ACCESS = user-tiled (hl.tile over the reduction axis)
  ORIGIN = grid (the inner re-tile rides the outer grid dim-0; bool(m_block_ids) True)
  EXTENT = unbacked (data-dependent hl.tile(begin,end); size is a SymInt, NOT None ->
           NOT the JAGGED out-of-scope predicate -- it has a SymInt extent, classified
           RESIDENT, and the unbacked size_hint() returns the 8192 placeholder)
  CARRIED-RESIDENT = no (scalar accumulator)
  NON-BACKWARD = yes (plain segment-sum, no grad-collapse)

The cell question: does grid_reduction_origin mis-fire / correctly classify, and does
the 8192 unbacked size_hint placeholder mis-size (e.g. floor the tunable reduction tile
to block_size=1, or NO_FIRE)?

NOTE the OUT-OF-SCOPE guards we deliberately avoid:
  * JAGGED -- the reduction axis size is None (hl.jagged_tile). We use hl.tile(begin,end),
    whose extent is an UNBACKED SymInt (not None), so it is IN scope.
  * STRIDED-DIM0 -- a grid dim-0 reduction whose memory stride != itemsize. Here the
    reduced tensor is a contiguous 1-D ``values`` vector indexed by the dim-0 tile, so
    its stride over the reduced elements == itemsize (NOT the strided-dim0 predicate).
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROBES = os.path.abspath(os.path.join(_THIS, ".."))
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import check_kernel  # noqa: E402

_WT = os.path.abspath(os.path.join(_THIS, "..", "..", "..", ".."))
assert os.path.abspath(helion.__file__).startswith(_WT + os.sep), helion.__file__

DEV = "cuda"


# UNBACKED-INNER NON-BACKWARD: chunked segment-sum of a 1-D values vector. The outer
# grid loop tiles E into fixed CTAs; the inner hl.tile(cta.begin, cta.end) re-tiles each
# CTA's data-dependent slab (unbacked extent) and sums the 1-D values into a per-CTA
# scalar. No feature axis, no grad, no [N] accumulator -- a plain non-backward reduction
# whose reduction axis has an UNBACKED extent.
@helion.kernel(static_shapes=False)
def chunked_segment_sum(values: torch.Tensor) -> torch.Tensor:
    (E,) = values.shape
    cta_block = hl.register_block_size(E)
    n_cta = (E + cta_block - 1) // cta_block
    out = values.new_empty([n_cta, 1], dtype=torch.float32)
    for cta in hl.tile(E, block_size=cta_block):
        acc = hl.zeros([1], dtype=torch.float32)
        for inner in hl.tile(cta.begin, cta.end):
            acc += values[inner].to(torch.float32).sum(dim=0)
        out[cta.id, :] = acc
    return out


def main() -> None:
    print(f"helion={helion.__file__}\n")
    values = torch.randn(1 << 16, device=DEV, dtype=torch.float32)
    intended = {
        "cell": "unbacked_inner_nonbackward",
        "access": "user-tiled",
        "origin": "grid",
        "extent": "unbacked",
        "carried_resident": False,
        "non_backward": True,
    }
    v = check_kernel("chunked_segment_sum", chunked_segment_sum, (values,), intended)
    obs = v["observed"]
    print(f"red       = {v['red']}")
    print(f"reasons   = {v['reasons']}")
    print(f"fired     = {obs.get('fired')}")
    print(f"n_rfacts  = {obs.get('n_reduction_facts')}")
    print(f"lowering_reduction_axes = {obs.get('lowering_reduction_axes')}")
    print(f"grid_block_ids          = {obs.get('grid_block_ids')}")
    print(f"block_sizes_valid_ids   = {obs.get('block_sizes_valid_ids')}")
    print(f"reduction_loops_valid   = {obs.get('reduction_loops_valid_ids')}")
    print(f"normalized_cfg          = {obs.get('normalized_cfg')}")
    print(f"fact                    = {obs.get('fact')}")


if __name__ == "__main__":
    main()
