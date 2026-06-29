"""CELL: sat_prod_op — SATURATION CHECK on the reduction-OP identity axis.

The §1 axes the reduction seed models: ACCESS / ORIGIN / EXTENT / CARRIED-RESIDENT /
CO-RESIDENCY / REUSE / NON-REDUCTION-LOOP / DIMS / PINNED-GRID. The OP IDENTITY of the
reduction (sum vs amax vs argmax vs PROD vs welford-mean/var) is NOT in that list.

This cell varies the op to ``torch.prod`` — a multiplicative running-product reduction over
a contiguous row — holding every modeled property fixed at the canonical single-reduction
point (ACCESS=standard/rolled-rdim, ORIGIN=inner/grid-pinned, EXTENT=static, no carried 2-D
tile, single loop, single streamed load, no row-reread, full-width-output=False, 2-D, M
grid-pinned). A ``sum`` over the SAME row would land on the IDENTICAL ReductionFact (the fact
records only structural/access properties — size_hint, itemsize, num_load, row_reread,
body_live_tiles, full_width_output, ... — and NEVER the op identity; see
``_assemble_reduction_fact`` in device_ir.py). So the seed it emits for ``prod`` is the same
config it would emit for ``sum``.

SATURATION QUESTION (§3a.4): does ``prod`` FIRE and emit a JUSTIFIED config — every emitted
field tracing to a recorded property + a named cap — or is the OP IDENTITY a MISSING AXIS
(prod needs a config the heuristic cannot give because it does not model the op)?

Property point: ACCESS=standard(rolled) x ORIGIN=inner x EXTENT=static(4096) x
CARRIED-RESIDENT=none x CO-RESIDENCY=single x REUSE=stream-once x NON-RED-LOOP=none x DIMS=2 x
PINNED-GRID=M(block_size=1) x OP=torch.prod (the NOT-modeled axis under test).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/dev/local/helion-unify/_lab/unify/probes")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import check_kernel  # noqa: E402

_WT = "/home/dev/local/helion-unify"
assert os.path.abspath(helion.__file__).startswith(_WT + os.sep), (
    f"helion ({helion.__file__}) not under worktree ({_WT})"
)

torch.manual_seed(0)
DEV = "cuda"
F32 = torch.float32


# A single PRODUCT reduction over a contiguous row. Grid-pinned over M (block_size=1) so M is
# the only grid axis and the reduction over N is the dominant (and only) reduction fact. The
# rolled rdim is a standard-track reduction (NOT a block_sizes tile).
@helion.kernel(static_shapes=False)
def row_prod(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        # torch.prod over the row: the multiplicative analogue of .sum(-1). One streamed
        # load, scalar [M_BLOCK] carry, full row reduced into one product.
        out[tile_m] = torch.prod(x[tile_m, :].to(torch.float32), dim=-1)
    return out


def main():
    print(f"helion={helion.__file__}\n")
    M = 8192
    N = 4096
    x = torch.randn(M, N, device=DEV, dtype=F32)
    intended = {
        "cell": "sat_prod_op",
        "access": "standard(rolled-rdim)",
        "origin": "inner/grid-pinned",
        "extent": "static(4096)",
        "carried_resident": "none",
        "co_residency": "single",
        "reuse": "stream-once",
        "non_red_loop": "none",
        "dims": 2,
        "pinned_grid": "M(block_size=1)",
        "op": "torch.prod  (NOT-modeled axis under test)",
    }
    v = check_kernel("row_prod", row_prod, (x,), intended)
    obs = v["observed"]
    ns = obs.get("normalized_cfg", {})
    red = v["red"] or "green"
    print(f"[{red:13s}] sat_prod_op / row_prod")
    print(f"  fired                   = {obs.get('fired')}")
    print(f"  n_reduction_facts       = {obs.get('n_reduction_facts')}")
    print(f"  n_matmul_facts          = {obs.get('n_matmul_facts')}")
    print(f"  lowering_reduction_axes = {obs.get('lowering_reduction_axes')}")
    print(f"  grid_block_ids          = {obs.get('grid_block_ids')}")
    print(f"  block_sizes_valid_ids   = {obs.get('block_sizes_valid_ids')}")
    print(f"  reduction_loops_valids  = {obs.get('reduction_loops_valid_ids')}")
    print(f"  fact[0]                 = {obs.get('fact')}")
    print(f"  raw_seed                = {obs.get('raw_seed')}")
    print(f"  normalized_cfg          = {ns}")
    for r in v["reasons"]:
        print(f"  reason: {r}")
    return v


if __name__ == "__main__":
    main()
