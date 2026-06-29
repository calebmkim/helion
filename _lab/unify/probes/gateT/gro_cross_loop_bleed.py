"""GATE-T NOVEL probe k5 — grid_reduction_origin UNBACKED-placeholder dominance vs a real
narrow reduction in the SAME kernel (attack axes b + d).

Attack: grid_reduction_origin fires when a reducing axis has an UNBACKED extent (size_hint =
create_unbacked_symint placeholder, 8192). _reduction_primary_fact picks the DOMINANT fact by
size_hint. In register_unrolled_reductions the RANKING excludes unbacked axes (good), but across
MULTIPLE facts (build_reduction_facts, one per rolled record) the dominant pick is over WHOLE facts
by fact.size_hint. If a grid-collapse fact's primary is an unbacked axis whose size_hint is the
8192 placeholder, it can SPURIOUSLY OUTRANK or UNDERRANK a real reduction fact -> the collapse seed
mis-fires (sizes the wrong fact) or the occupancy CTA block is computed off a placeholder grid.

Structure: a two-level grid-collapse (unbacked inner re-tile -> grid_reduction_origin) over one
tensor, PLUS a separate plain rolled reduction over another tensor of a DIFFERENT real extent. Two
ReductionFacts; check the dominant pick and whether the collapse CTA block / occupancy sizing is
justified from the (placeholder vs real) extents.

Intended: ORIGIN=grid (collapse) + a second inner-loop fact; multi-fact; one unbacked extent.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import check_kernel  # noqa: E402

_WT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
assert os.path.abspath(helion.__file__).startswith(_WT + os.sep)

torch.manual_seed(0)
DEV = "cuda"
F32 = torch.float32


@helion.kernel(static_shapes=False, ignore_warnings=[helion.exc.TensorOperationInWrapper])
def gro_plus_rolled(x: torch.Tensor, y: torch.Tensor) -> tuple:
    # x: [M, F] grid-collapse into a per-feature accumulator (two-level grid loop with an
    # UNBACKED inner re-tile -> grid_reduction_origin). y: [My, Ny] a plain rolled reduction.
    M, F = x.shape
    My, Ny = y.shape
    m_block = hl.register_block_size(M)
    nb = (M + m_block - 1) // m_block
    blocks = x.new_zeros([nb, F], dtype=torch.float32)
    oy = torch.empty([My], dtype=torch.float32, device=y.device)
    # Grid-collapse: outer grid over M, inner re-tile over the cta's row slab (UNBACKED range).
    for cta in hl.tile(M, block_size=m_block):
        acc = x.new_zeros([F], dtype=torch.float32)
        for mb in hl.tile(cta.begin, cta.end):
            acc = acc + torch.sum(x[mb, :].to(torch.float32), dim=0)
        blocks[cta.id, :] = acc
    # Plain rolled reduction over y (a real, backed extent Ny) -- second top-level loop.
    for tn in hl.tile(My):
        oy[tn] = y[tn, :].to(torch.float32).sum(-1)
    gw = torch.sum(blocks, dim=0)
    return gw, oy


def main():
    print(f"helion={helion.__file__}\n")
    x = torch.randn(8192, 1024, device=DEV, dtype=F32)   # F=1024 feature (collapse)
    y = torch.randn(256, 2048, device=DEV, dtype=F32)     # Ny=2048 real rolled reduction
    intended = {"cell": "gro_plus_rolled", "origin": "grid+inner",
                "extent": "unbacked+static", "n_facts": ">=1"}
    v = check_kernel("gro_plus_rolled", gro_plus_rolled, (x, y), intended)
    obs = v["observed"]
    ns = obs.get("normalized_cfg", {})
    red = v["red"] or "green"
    print(f"[{red}] gro_plus_rolled")
    for k in ("fired", "n_reduction_facts", "n_matmul_facts",
              "lowering_reduction_axes", "grid_block_ids",
              "block_sizes_valid_ids", "reduction_loops_valid_ids", "fact"):
        print(f"  {k:24s}= {obs.get(k)}")
    print(f"  normalized block_sizes  = {ns.get('block_sizes') if ns else None}")
    print(f"  normalized reduction_loops = {ns.get('reduction_loops') if ns else None}")
    print(f"  reasons                 = {v['reasons']}")
    return v


if __name__ == "__main__":
    main()
