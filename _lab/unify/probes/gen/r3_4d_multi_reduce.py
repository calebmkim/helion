"""GEN CELL: r3_4d_multi_reduce — rank-4 RESIDENT x PARTIAL_GRID classification interplay.

A 4-D kernel ``x: [B, M, R, N]`` that reduces over TWO of the four axes in a SINGLE
per-CTA partial, then finalizes cross-CTA:

  - The GRID M axis (block 1) is REDUCED ACROSS (a backed column-collapse
    ``x[b, cta, :, :].sum(dim=1)`` folds the static M grid-tile rows into the per-CTA
    partial). M is a TUNABLE grid tile (``hl.tile([B, M])``) -> its whole-axis extent is a
    per-grid COUNT, not a per-program reduction size. The just-landed grid-tile-reduction fix
    must classify M as ``PARTIAL_GRID`` and FLOOR it (keep it a grid row), NOT size it full
    extent -- else the M grid collapses to 1 program (the backed_col_sum_collapse hole, now
    at rank 4).
  - The INNER R axis is a genuine RESIDENT reduction (a non-grid inner ``hl.tile(R)`` loop
    folded with ``.sum(-1)`` into the per-CTA partial). It MUST be sized as a reduction
    (tiled, NOT floored to block_size=1) -- the FLOOR1_TILED guard.
  - N is a pure non-reduced fan-out (kept in the output / the cross-CTA finalize).

So at rank 4 the kernel forces the sizer to make BOTH calls in ONE fact: FLOOR the
PARTIAL_GRID grid-M axis AND TILE the RESIDENT inner-R axis. A regression in the fix would
show as EITHER (a) the grid-M axis widened to full M extent (grid collapse = UNJUSTIFIED), OR
(b) the inner-R reduction floored to block_size=1 (FLOOR1_TILED). The adversarial question:
does excluding PARTIAL_GRID from the user-tiled secondaries (bfa012cc) still admit the
co-resident RESIDENT inner reduction at rank 4, or did the fix over-exclude and open a new
NO_FIRE / FLOOR1_TILED hole?

§1 axes: ACCESS=user-tiled (manual inner R loop) + grid-collapse over M; ORIGIN=MIXED (one
grid-origin reduced axis M + one inner-origin reduced axis R in the same partial);
EXTENT=static (all four extents backed); CO-RESIDENCY=same-partial (M-fold and R-fold both
accumulate into the per-CTA [B_BLOCK, N] slab); DIMS=4; N-REDUCING-AXES=2 (one PARTIAL_GRID,
one RESIDENT).

AVOIDS the corpus: backed_col_sum_collapse (round-2) is rank-2 with a SINGLE grid-M reduction
(no co-resident inner RESIDENT reduction); boundary_5d reduces THREE RESIDENT inner axes (no
grid-reduced axis); carried_3d_highdim carries a resident tile across a user-tiled K loop (no
grid collapse); grid_origin_nonnorm / gro_divergence are 2-D unbacked-inner-retile collapses.
HERE: rank-4, ONE PARTIAL_GRID grid reduction + ONE RESIDENT inner reduction, co-resident.

Run (compile-only):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-unify /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-unify/_lab/unify/probes/gen/r3_4d_multi_reduce.py
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
def r3_4d_multi_reduce(x: torch.Tensor) -> torch.Tensor:
    """x: [B, M, R, N]. Grid over [B, M] (M is a TUNABLE grid tile = block 1).

    Per-CTA partial folds TWO reduction axes into a [B_BLOCK, N] slab:
      - the GRID-M rows (``.sum(dim=1)`` over the static M grid-tile) -> PARTIAL_GRID,
      - the INNER R axis (a nested ``hl.tile(R)`` loop, ``.sum(-1)``) -> RESIDENT.
    The N fan-out is kept; partials are written per CTA and finalized cross-CTA via
    ``blocks.sum(0)`` (so reducing over the partial M grid axis is sound)."""
    B, M, R, N = x.shape
    m_block = hl.register_block_size(M)
    nb = (M + m_block - 1) // m_block
    # cross-CTA partial buffer: one [B, N] slab per M grid-block.
    blocks = torch.zeros([nb, B, N], dtype=torch.float32, device=x.device)
    for tile_b, tile_m in hl.tile([B, M], block_size=[None, m_block]):
        # per-CTA accumulator over the (kept) N fan-out, for this B-tile.
        acc = hl.zeros([tile_b, N], dtype=torch.float32)
        # INNER RESIDENT reduction over R (manual hl.tile -> ordinary block_sizes entry):
        # fold the R-tile lanes AND the M grid-tile rows into the [B_BLOCK, N] partial.
        for tile_r in hl.tile(R):
            blk = x[tile_b, tile_m, tile_r, :].to(torch.float32)  # [B_BLK, M_BLK, R_BLK, N]
            # .sum(1): fold the M grid-tile rows (PARTIAL_GRID, finalized cross-CTA);
            # .sum(1) again: fold the inner R-tile lanes (RESIDENT). -> [B_BLK, N].
            acc = acc + blk.sum(1).sum(1)
        blocks[tile_m.id, tile_b, :] = acc
    return blocks.sum(0)  # cross-CTA finalize over the M grid-blocks -> [B, N]


def main() -> None:
    print(f"helion={helion.__file__}\n")
    # [B, M, R, N]: grid over [B, M]; M (4096) is the partial grid axis reduced cross-CTA;
    # R (256) is the inner RESIDENT reduction; N (512) is the non-reduced fan-out.
    B, M, R, N = 8, 4096, 256, 512
    x = torch.randn(B, M, R, N, device=DEV, dtype=F32)
    intended = {
        "cell": "r3_4d_multi_reduce",
        "access": "user-tiled + grid-collapse",
        "origin": "mixed (grid-M PARTIAL_GRID + inner-R RESIDENT)",
        "extent": "static",
        "co_residency": "same-partial",
        "dims": 4,
        "n_reducing_axes": 2,
        "reduced": {"M_grid_partial": M, "R_inner_resident": R},
        "fan_out": {"N": N},
        "expect": "grid-M FLOORED (PARTIAL_GRID), inner-R TILED (RESIDENT); no grid collapse",
    }
    v = check_kernel("r3_4d_multi_reduce", r3_4d_multi_reduce, (x,), intended)
    obs = v["observed"]
    ns = obs.get("normalized_cfg", {})
    red = v["red"] or "green"
    print(f"[{red}] r3_4d_multi_reduce")
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
