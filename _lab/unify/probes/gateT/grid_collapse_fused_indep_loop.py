"""Gate-T round-4 PERMANENT regression probes (findings 1 + 2): a two-level grid-collapse loop
FUSED in one kernel with an INDEPENDENT non-collapse loop, in the regime where the collapse's
UNBACKED inner-re-tile placeholder size_hint (create_unbacked_symint == 8192) coincidentally
equals the independent loop's BACKED reduction/loop extent.

This collision is the trap: ``grid_reduction_origin`` is a per-FACT bool (True iff SOME reducing
axis is the unbacked collapse re-tile), and the dominant-fact picker may select the INDEPENDENT
loop's backed reduction as the primary. Two distinct under-sizes resulted, both fixed
byte-identically on the 447-cell corpus (zero-diff vs fc1dbaa0):

  FINDING 2 (user-tiled track, ``_pinned_inner_resident_elems``): a grid-collapse fused with a
    separate NORMALIZE/apply loop -- the collapse's reduction-role axes leaked into the apply
    loop's co-residency denominator (the rolled-exclusion only checked ``reduction_loops``
    membership, which is EMPTY on the user-tiled track), flooring the apply tile to block_size=1.
    FIX: ``_pinned_inner_resident_elems`` also excludes any ``info.reduction`` axis.
    GUARD: ``collapse_plus_normalize`` -- the normalize loop must NOT floor to 1.

  FINDING 1 (user-tiled track, the primary-rdim m-collapse override): the override sized the
    PRIMARY rdim to the 64-elem collapse occupancy block, but the dominant-fact picker selected
    the INDEPENDENT loop-2's backed Q reduction (extent 8192) as primary -- a 128x+ under-size of
    an ordinary per-program reduction. FIX: gate the primary-rdim override additionally on the
    PRIMARY reduction axis being UNBACKED (the same extent-provenance test, scoped to the primary);
    the grid-CTA occupancy sizing stays keyed on the unscoped per-fact bool (loop-scoped to
    ``grid_collapse_block_ids`` inside ``_build_block_sizes``).
    GUARD: ``collapse_plus_indep_usertiled`` -- the collapse grid axis stays occupancy-sized (64)
    AND the independent loop-2 primary keeps a real reduction width (NOT floored to the 64 block).

Both RED at parent 8d722d13 (collapse_plus_normalize -> apply floored to 1;
collapse_plus_indep_usertiled -> loop-2 primary sized 64), GREEN at the fix SHAs.
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROBES = os.path.abspath(os.path.join(_THIS, ".."))
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

from checker import check_kernel  # noqa: E402
import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

_WT = os.path.abspath(os.path.join(_THIS, "..", "..", "..", ".."))
assert os.path.abspath(helion.__file__).startswith(_WT + os.sep), helion.__file__

torch.manual_seed(0)
DEV = "cuda"


@helion.kernel(static_shapes=False)
def collapse_plus_indep_usertiled(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """FINDING 1 reproducer. Loop 1: two-level grid-collapse over M (unbacked inner re-tile ->
    grid_reduction_origin=True). Loop 2: an INDEPENDENT grid over P, each row a user-tiled inner
    reduction over Q. With Q == the collapse's unbacked placeholder (8192), the dominant-fact
    picker selects loop-2's Q reduction as primary; the m-collapse override must NOT size it to
    the 64-elem collapse occupancy block."""
    m, n = x.shape  # collapse over M into per-feature [N]
    p, q = z.shape  # independent grid over P, user-tiled inner reduce over Q
    m_block = hl.register_block_size(m)
    nb = (m + m_block - 1) // m_block
    blocks = torch.zeros([nb, n], dtype=torch.float32, device=x.device)
    rowsum = torch.empty([p], dtype=torch.float32, device=x.device)
    for cta in hl.tile(m, block_size=m_block):
        acc = torch.zeros([n], dtype=torch.float32, device=x.device)
        for mb in hl.tile(cta.begin, cta.end):
            acc = acc + torch.sum(x[mb, :].to(torch.float32), dim=0)
        blocks[cta.id, :] = acc
    for tile_p in hl.tile(p, block_size=1):
        s = hl.zeros([tile_p], dtype=torch.float32)
        for tile_q in hl.tile(q):
            s = s + z[tile_p, tile_q].to(torch.float32).sum(-1)
        rowsum[tile_p] = s
    feat = torch.sum(blocks, dim=0)
    return feat.sum() + rowsum


@helion.kernel(static_shapes=False)
def collapse_plus_normalize(x: torch.Tensor, scale: torch.Tensor):
    """FINDING 2 reproducer (the Gate-T r4 sweep-2 kernel): grid-collapse over M into grad_w[N]
    PLUS a normalize/apply loop over N nested in the same CTA loop. The apply loop's reduction-role
    co-residents (the collapse re-tile) must be EXCLUDED from its co-residency denominator, else
    the apply tile floors to block_size=1 (serializing the apply pass). The normalize loop must be
    widened, not floored."""
    m, n = x.shape
    grad_w = torch.zeros([n], dtype=torch.float32, device=x.device)
    out = torch.empty([m, n], dtype=torch.float32, device=x.device)
    for cta in hl.tile(m):
        acc = hl.zeros([n], dtype=torch.float32)
        for mb in hl.tile(cta.begin, cta.end):
            acc += x[mb, :].to(torch.float32).sum(0)
        hl.atomic_add(grad_w, [slice(None)], acc)
        for tn in hl.tile(n):
            out[cta.begin : cta.end, tn] = (
                x[cta.begin : cta.end, tn].to(torch.float32) * scale[tn]
            )
    return grad_w, out


def _bs(name, kernel, args):
    v = check_kernel(name, kernel, args, {"cell": name})
    o = v.get("observed", {})
    raw = (o.get("raw_seed") or {}).get("block_sizes")
    fact = o.get("fact", {})
    return v.get("red"), raw, o.get("block_sizes_valid_ids"), fact


def main() -> None:
    print(f"helion={helion.__file__}\n")
    red = 0

    # FINDING 1: collapse grid axis (bid 0) occupancy-sized to 64; loop-2 primary (Q=8192) a real
    # reduction width, NOT the 64 occupancy block.
    x = torch.randn(8192, 1024, device=DEV, dtype=torch.float32)
    z = torch.randn(4096, 8192, device=DEV, dtype=torch.float32)
    r1, bs1, ids1, f1 = _bs(
        "collapse_plus_indep_usertiled", collapse_plus_indep_usertiled, (x, z)
    )
    prim = f1.get("primary_reduction_block_id")
    prim_val = bs1[ids1.index(prim)] if (bs1 and prim in (ids1 or [])) else None
    f1_bad = (r1 is not None) or (prim_val is not None and prim_val <= 64)
    print(
        f"[F1] bs={bs1} valid={ids1} primary_bid={prim} primary_block={prim_val} "
        f"red={r1} -> {'RED' if f1_bad else 'green'}"
    )
    red += int(bool(f1_bad))

    # FINDING 2: normalize/apply loop must NOT floor to 1.
    x2 = torch.randn(8192, 4096, device=DEV, dtype=torch.float32)
    scale = torch.randn(4096, device=DEV, dtype=torch.float32)
    r2, bs2, ids2, f2 = _bs(
        "collapse_plus_normalize", collapse_plus_normalize, (x2, scale)
    )
    nrl = set(f2.get("non_reduction_loop_block_ids") or [])
    # The apply/normalize tile over N (the WIDE feature axis) must be widened, not floored. The
    # floor manifested as a block_size==1 on a tunable axis that is NOT the collapse grid CTA
    # (bid 0, legitimately occupancy-sized small) and NOT a real reduction. Guard: the WIDEST
    # tunable block must reflect the feature width (the apply loop), i.e. > 1 -- a floor-to-1 of
    # the apply loop drops the max block to the collapse CTA/re-tile sizes.
    prim2 = f2.get("primary_reduction_block_id")
    apply_candidates = [
        bs2[ids2.index(b)]
        for b in (ids2 or [])
        if bs2 and b != prim2 and b not in (f2.get("grid_collapse_block_ids") or [])
    ]
    f2_bad = (r2 is not None) or (
        bool(apply_candidates) and max(bs2) <= 1
    ) or (bool(bs2) and max(bs2) < 4096)
    print(
        f"[F2] bs={bs2} valid={ids2} apply_loop_ids={sorted(nrl)} max_block={max(bs2) if bs2 else None} "
        f"red={r2} -> {'RED' if f2_bad else 'green'}"
    )
    red += int(bool(f2_bad))

    print(f"\n=== {red}/2 RED at this SHA ===")
    if red:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
