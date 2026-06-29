"""CELL r3_fp16_mcollapse: an fp16 GRID-ORIGIN M-collapse WITH per-row work.

ADVERSARIAL re-probe of the two recent fixes, on a provenance-NOVEL cell:

  (1) grid-tile-reduction exclusion (PARTIAL_GRID) -- we must NOT trip it: the
      collapse here is over an UNBACKED inner re-tile (``hl.tile(cta.begin, cta.end)``),
      the same in-graph signature ``grid_reduction_origin`` keys on, so GRO FIRES and the
      grid M axis is occupancy-sized (``_m_collapse_grid_block``), NOT footprint-widened to
      the full extent (which would collapse the grid).
  (2) dtype-faithful residency caps -- the INPUT is fp16 (``fact.itemsize == 2``) but the
      reduction tree accumulates in fp32, so the resident ``[inner, feature]`` tile the
      M-collapse inner byte-cap protects is fp32-WIDE. The cap must divide the budget by
      ``_resident_itemsize(fact) == max(itemsize, 4) == 4``, NOT by the raw fp16 itemsize 2.
      If it keyed on 2 it would size the inner tile 2x too wide (admitting a spilling tile at
      half precision). This cell exercises THAT exact arithmetic at fp16.

This is a grad-PARAMETER M-collapse (dyt-backward shape): collapse the grid/batch (M) axis
into a per-feature ``grad_weight[N]`` accumulator finalized cross-CTA, but WITH PER-ROW WORK
(a full-width ``grad_x`` store + a tanh-derivative intermediate). The per-row work makes
``body_live_tiles > 1`` (multiple full-width tiles live at the inner-loop peak), which routes
the user-tiled M-collapse seed into the ELSE branch (collapse WITH per-row work):

    feat_bytes = max(1, fact.feature_footprint) * cls._resident_itemsize(fact)   # N * max(2,4)
    inner_cap  = cls._m_collapse_inner_byte_cap(feat_bytes)                       # 32768 // feat_bytes
    r_block    = max(1, min(m_collapse_block, inner_cap))

So the inner reduction tile is byte-capped by the fp32-acc feature footprint -- the EXACT
dtype-faithful cap to scrutinize at fp16.

Property point (§1):
  ACCESS         = user-tiled (hand-written hl.tile over the reducing/grid axis)
  ORIGIN         = grid (the inner re-tile rides the outer grid dim-0; reduce ACROSS programs
                   into a per-feature [N] accumulator finalized by a cross-CTA sum(0))
  EXTENT         = unbacked inner re-tile (hl.tile(cta.begin, cta.end), data-dependent SymInt
                   extent -> grid_reduction_origin True; NOT None, so NOT the JAGGED predicate)
  CARRIED-RESIDENT = a per-feature [N] grad_weight accumulator (fp32)
  CO-RESIDENCY   = single primary reduction (the grad_weight collapse)
  PER-ROW WORK   = YES (full-width grad_x store + tanh' intermediate) -> body_live_tiles > 1
  DIMS           = 2
  PINNED-GRID    = none (M is a plain tunable hl.tile(M, block_size=m_block) grid axis)
  DTYPE          = fp16 input (fact.itemsize == 2; fp32 accumulator) -- the NOT-modeled axis

EXPECTED (the JUSTIFIED config, every field tracing to a property + named cap):
  - GRO fires (m_block_ids non-empty AND the inner re-tile axis carries an unbacked symbol).
  - The grid M axis is OCCUPANCY-sized via _m_collapse_grid_block = next_pow2(grid_rows//num_sm)
    capped at M_COLLAPSE_MAX_CTA=256 -- NOT widened to the full M extent (no grid collapse).
  - The inner reduction r_block = min(m_collapse_block, _m_collapse_inner_byte_cap(N*4)). At
    N=2048: feat_bytes = 2048*4 = 8192, inner_cap = 32768 // 8192 = 4 (pow2). The cap used 4,
    NOT 2: had it keyed fp16 itemsize 2 -> feat_bytes 4096 -> inner_cap 8 (2x too wide, a
    spilling resident [8, 2048] fp32 = 64 KiB tile vs the 32 KiB M_COLLAPSE_TILE_BYTES budget).
  - num_warps from the plain extent ramp (grid-origin drops narrow-w1).

UNJUSTIFIED watch: scrutinize normalized_cfg for the grid M axis widened toward the full M
extent (a grid collapse the occupancy sizer would have prevented) or an inner r_block that only
makes sense at itemsize=2 (the fp16 under-count). We ALSO build the fp32 sibling in the same
run: the JUSTIFIED prediction is the two emit the SAME grid block + SAME inner r_block (the cap
is fp32-acc-floored, so dtype-invariant here) -- a divergence would be the dtype hole.

OUT-OF-SCOPE guards deliberately avoided:
  * JAGGED -- the inner re-tile extent is an UNBACKED SymInt (hl.tile(cta.begin, cta.end)),
    NOT None (hl.jagged_tile). It HAS a size_hint placeholder -> in scope.
  * STRIDED-DIM0 -- x is a contiguous [M, N] tensor; the reduced ELEMENTS x[mb, :] load as full
    contiguous rows (stride over the reduced feature elements == itemsize). The ordinary
    batch-collapse access pattern, NOT a 1-D dim-0 reduction with stride != itemsize.
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
F16 = torch.float16
F32 = torch.float32


# fp16 GRID-ORIGIN M-collapse WITH per-row work (dyt-backward shape). The outer grid loop tiles
# the batch (M) axis into CTAs; the inner hl.tile(cta.begin, cta.end) re-tiles each CTA's
# data-dependent slab (UNBACKED extent -> grid_reduction_origin). Per row we compute the tanh
# derivative (per-row intermediate) AND store a full-width grad_x[mb, :] (per-row work, so
# body_live_tiles > 1), while ACCUMULATING the per-feature grad_weight[N] across the grid rows --
# finalized cross-CTA via sum(0). The reduction's ORIGIN is the grid M axis.
@helion.kernel(static_shapes=False)
def dyt_bwd_collapse_fp16(
    grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    m_block = hl.register_block_size(M)
    nb = (M + m_block - 1) // m_block
    grad_x = torch.empty_like(x)
    # per-CTA partial grad_weight[N], finalized cross-CTA.
    gw_blocks = x.new_zeros([nb, N], dtype=torch.float32)
    for cta in hl.tile(M, block_size=m_block):
        gw_acc = x.new_zeros([N], dtype=torch.float32)
        for mb in hl.tile(cta.begin, cta.end):
            go = grad_out[mb, :].to(torch.float32)
            xi = x[mb, :].to(torch.float32)
            w = weight[:].to(torch.float32)
            # per-row work: tanh derivative intermediate + full-width grad_x store (multiple
            # full-width tiles live at the inner-loop peak -> body_live_tiles > 1).
            t = torch.tanh(xi)
            dx = go * w * (1.0 - t * t)
            grad_x[mb, :] = dx.to(grad_x.dtype)
            # per-feature grad_weight collapse ACROSS the grid rows.
            gw_acc = gw_acc + torch.sum(go * t, dim=0)
        gw_blocks[cta.id, :] = gw_acc
    grad_weight = torch.sum(gw_blocks, dim=0)
    return grad_x, grad_weight


# fp32 SIBLING: identical structure, fp32 input (itemsize=4) -- the dtype-invariance baseline.
@helion.kernel(static_shapes=False)
def dyt_bwd_collapse_fp32(
    grad_out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    m_block = hl.register_block_size(M)
    nb = (M + m_block - 1) // m_block
    grad_x = torch.empty_like(x)
    gw_blocks = x.new_zeros([nb, N], dtype=torch.float32)
    for cta in hl.tile(M, block_size=m_block):
        gw_acc = x.new_zeros([N], dtype=torch.float32)
        for mb in hl.tile(cta.begin, cta.end):
            go = grad_out[mb, :].to(torch.float32)
            xi = x[mb, :].to(torch.float32)
            w = weight[:].to(torch.float32)
            t = torch.tanh(xi)
            dx = go * w * (1.0 - t * t)
            grad_x[mb, :] = dx.to(grad_x.dtype)
            gw_acc = gw_acc + torch.sum(go * t, dim=0)
        gw_blocks[cta.id, :] = gw_acc
    grad_weight = torch.sum(gw_blocks, dim=0)
    return grad_x, grad_weight


def _inner_byte_cap_pred(feature_footprint: int, resident_itemsize: int) -> int:
    """Mirror of _m_collapse_inner_byte_cap for the post-hoc justification print."""
    from helion._utils import next_power_of_2 as _np2

    feat_bytes = max(1, feature_footprint) * resident_itemsize
    return max(1, _np2(max(1, 32768 // max(1, feat_bytes))))


def main() -> None:
    print(f"helion={helion.__file__}\n")
    M = 8192
    N = 2048
    go16 = torch.randn(M, N, device=DEV, dtype=F16)
    x16 = torch.randn(M, N, device=DEV, dtype=F16)
    w16 = torch.randn(N, device=DEV, dtype=F16)
    go32 = torch.randn(M, N, device=DEV, dtype=F32)
    x32 = torch.randn(M, N, device=DEV, dtype=F32)
    w32 = torch.randn(N, device=DEV, dtype=F32)

    intended = {
        "cell": "r3_fp16_mcollapse",
        "access": "user-tiled",
        "origin": "grid (per-feature grad_weight collapse across the grid/batch axis)",
        "extent": "unbacked inner re-tile (hl.tile(cta.begin, cta.end))",
        "carried_resident": "per-feature [N] fp32 grad_weight accumulator",
        "co_residency": "single primary reduction",
        "per_row_work": "yes (full-width grad_x store + tanh' intermediate) -> body_live_tiles>1",
        "dims": 2,
        "pinned_grid": "none (M tunable hl.tile)",
        "dtype": "fp16 input (itemsize 2), fp32 accumulator",
        "expect": "GRO True; grid M occupancy-sized (not full-extent); inner r_block byte-capped "
                  "by feature_footprint * max(itemsize,4)=4 (fp32-acc floor), dtype-invariant vs fp32",
    }

    v16 = check_kernel("r3_fp16_mcollapse__fp16", dyt_bwd_collapse_fp16,
                       (go16, x16, w16), intended)
    v32 = check_kernel("r3_fp16_mcollapse__fp32", dyt_bwd_collapse_fp32,
                       (go32, x32, w32),
                       {**intended, "dtype": "fp32 baseline (itemsize 4)"})

    import json

    for tag, v in (("fp16", v16), ("fp32", v32)):
        obs = v["observed"]
        fact = obs.get("fact") or {}
        cfg = obs.get("normalized_cfg") or {}
        isz = fact.get("itemsize")
        ff = fact.get("feature_footprint")
        resident_isz = max(4, max(1, isz)) if isz else None
        print(f"\n=== {tag} ===")
        print(f"red                     = {v['red']}")
        print(f"reasons                 = {v['reasons']}")
        print(f"fired                   = {obs.get('fired')}")
        print(f"n_reduction_facts       = {obs.get('n_reduction_facts')}")
        print(f"n_matmul_facts          = {obs.get('n_matmul_facts')}")
        print(f"lowering_reduction_axes = {obs.get('lowering_reduction_axes')}")
        print(f"grid_block_ids          = {obs.get('grid_block_ids')}")
        print(f"block_sizes_valid_ids   = {obs.get('block_sizes_valid_ids')}")
        print(f"reduction_loops_valid   = {obs.get('reduction_loops_valid_ids')}")
        print(f"itemsize                = {isz}")
        print(f"feature_footprint       = {ff}")
        print(f"grid_reduction_origin   = {fact.get('grid_reduction_origin')}")
        print(f"body_live_tiles         = {fact.get('body_live_tiles')}")
        print(f"normalized_cfg          = {cfg}")
        if ff is not None and resident_isz is not None:
            cap4 = _inner_byte_cap_pred(ff, 4)
            cap_isz = _inner_byte_cap_pred(ff, isz)
            print(f"inner_byte_cap @itemsize4(fp32-acc floor) = {cap4}")
            print(f"inner_byte_cap @raw_itemsize({isz})        = {cap_isz}")

    print("\n=== FULL fp16 VERDICT ===")
    print(json.dumps(v16, indent=2, default=repr))


if __name__ == "__main__":
    main()
