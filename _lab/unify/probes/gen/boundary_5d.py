"""GEN CELL: boundary_5d — BOUNDARY/high-dim rank stress on _build_block_sizes.

Stresses the RANK assumption of the reduction sizer. A 2-D grid over ``[M, P]`` nests a
SINGLE user-tiled inner loop over THREE reduction axes ``hl.tile([A, B, C])``, accumulating
a RANK-5 resident tile ``[M_BLOCK, P_BLOCK, A_BLOCK, B_BLOCK, C_BLOCK]`` that is then folded
over all three reduction dims. So:

  - tile rank = 5 (two grid CTA dims + three reduction-tile dims) -> rank > 4.
  - THREE high-dim, statically-extented, user-tiled RESIDENT reduction axes in ONE loop ->
    ``build_reduction_facts`` picks the widest (A) as ``primary_reduction_block_id`` and
    records B, C in ``secondary_reduction_block_ids``; ``_build_block_sizes`` must size ALL
    THREE via ``red_values`` (primary r_block + per-axis secondary r_block) rather than
    flooring the two non-dominant reduction tiles to 1.

The §1 axes this lands on: ACCESS=user-tiled, ORIGIN=inner (loop nested in a [M,P] grid),
EXTENT=static, CO-RESIDENCY=same-loop (all three reductions share ONE inner loop, co-resident
in the rank-5 accumulator), DIMS=5, N-FACTS multi-axis (3 reducing axes on one fact).

QUESTION (the boundary probe): does ``_build_block_sizes`` handle rank>4 with MULTIPLE
high-dim reductions without (a) CRASH (a rank assumption / index error in the per-axis
sizer), (b) NO_FIRE (the multi-axis fact declines), or (c) FLOOR1_TILED (a non-dominant
reduction tile B or C floored to block_size=1 while extent>1)?

AVOIDS the corpus: carried_3d_highdim is rank-3 with a CARRIED resident tile across a
user-tiled K loop; four_d_inner_reduce (corpus C5) is rank-4 with the reduction collapsed
each pass; multifact_3way is 3 reductions in 3 SEPARATE loops over 3 tensors. HERE all three
reductions are co-resident in ONE rank-5 tile in ONE loop.

Run (compile-only):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-unify /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-unify/_lab/unify/probes/gen/boundary_5d.py
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


@helion.kernel(static_shapes=False)
def boundary_5d(x: torch.Tensor) -> torch.Tensor:
    """x: [M, P, A, B, C]. Grid over [M, P]; ONE inner user-tiled loop over the THREE
    reduction axes [A, B, C], accumulating a RANK-5 resident partial
    [M_BLOCK, P_BLOCK, A_BLOCK, B_BLOCK, C_BLOCK], then folding all three reduction dims.

    The widest reduction extent (A) is the primary rdim; B and C are recorded as secondary
    reducing axes. This is the rank>4 + multi-high-dim-reduction boundary for the sizer."""
    M, P, A, B, C = x.shape
    out = torch.empty([M, P], dtype=torch.float32, device=x.device)
    for tile_m, tile_p in hl.tile([M, P]):
        # rank-2 grid accumulator (the result lives per [M_BLOCK, P_BLOCK]).
        acc = hl.zeros([tile_m, tile_p], dtype=torch.float32)
        # ONE inner loop over THREE reduction axes -> rank-5 co-resident tile
        # [M_BLOCK, P_BLOCK, A_BLOCK, B_BLOCK, C_BLOCK]. Fold the three reduction dims with
        # SEPARATE single-dim .sum(-1) ops (the lowering rejects a single multi-dim reduction
        # `.sum([-1,-2,-3])` with NotImplementedError), so each tile axis is its own
        # ReductionLowering -- three co-resident tunable reduction tiles in ONE loop.
        for tile_a, tile_b, tile_c in hl.tile([A, B, C]):
            blk = x[tile_m, tile_p, tile_a, tile_b, tile_c].to(torch.float32)
            acc = acc + blk.sum(-1).sum(-1).sum(-1)
        out[tile_m, tile_p] = acc
    return out


def main() -> None:
    print(f"helion={helion.__file__}\n")
    # [M, P, A, B, C]: grid [M,P]; three reduction extents A > B > C (all static, all > 1,
    # all genuinely tunable user-tiled reduction tiles). A is the dominant/primary rdim.
    M, P = 256, 8
    A, B, C = 512, 64, 32
    x = torch.randn(M, P, A, B, C, device=DEV, dtype=F32)
    intended = {
        "cell": "boundary_5d",
        "access": "user-tiled",
        "origin": "inner",
        "extent": "static",
        "co_residency": "same-loop",
        "dims": 5,
        "n_reducing_axes": 3,
        "reduction_extents": {"A_primary": A, "B": B, "C": C},
    }
    v = check_kernel("boundary_5d", boundary_5d, (x,), intended)
    obs = v["observed"]
    ns = obs.get("normalized_cfg", {})
    red = v["red"] or "green"
    print(f"[{red}] boundary_5d")
    print(f"  fired                   = {obs.get('fired')}")
    print(f"  n_reduction_facts       = {obs.get('n_reduction_facts')}")
    print(f"  n_matmul_facts          = {obs.get('n_matmul_facts')}")
    print(f"  lowering_reduction_axes = {obs.get('lowering_reduction_axes')}")
    print(f"  grid_block_ids          = {obs.get('grid_block_ids')}")
    print(f"  block_sizes_valid_ids   = {obs.get('block_sizes_valid_ids')}")
    print(f"  reduction_loops_valids  = {obs.get('reduction_loops_valid_ids')}")
    print(f"  fact[0]                 = {obs.get('fact')}")
    print(f"  raw_seed                = {obs.get('raw_seed')}")
    print(f"  normalized block_sizes  = {ns.get('block_sizes') if ns else None}")
    print(f"  normalized reduction_loops = {ns.get('reduction_loops') if ns else None}")
    print(f"  reasons                 = {v['reasons']}")
    import json
    print(json.dumps(v, indent=2, default=repr))
    return v


if __name__ == "__main__":
    main()
