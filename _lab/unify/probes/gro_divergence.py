"""GATE-D divergence fixture for the grid_reduction_origin (GRO) key (Defect-2 re-key).

GRO = bool(m_block_ids) AND any reducing axis has an UNBACKED extent
      (free_unbacked_symbols(env.block_sizes[b].size)).
It claims to be the faithful (ORIGIN=grid) property the per_feature_accumulator (PFA) bolt-on
recognized -- but provenance-DISTINCT (PFA reads accumulator SHAPE; GRO reads reduction
EXTENT-PROVENANCE). The Gate-D bar: GRO must (a) AGREE with PFA on the corpus, (b) FIRE on a
NON-norm-bwd grid-collapse family (proving it is a property, not a norm-bwd recognizer -- the
≥2-structurally-distinct-families test), and (c) NOT fire on near-miss kernels (a plain
user-tiled rowsum; a per-row feature reduce). Permanent fixture.
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _WT not in sys.path:
    sys.path.insert(0, _WT)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep)
DEV = "cuda"


def gro_pfa(fn, args):
    bound = fn.bind(args)
    spec = bound.env.config_spec
    env = bound.env
    if not spec.reduction_facts:
        return ("no-fact", None, None)
    rf = spec.reduction_facts[0]
    reducing = {rf.primary_reduction_block_id, *rf.secondary_reduction_block_ids}
    gro_recomputed = bool(rf.m_block_ids) and any(
        free_unbacked_symbols(env.block_sizes[b].size) for b in reducing
    )
    # Read the populated field (post-rename it is grid_reduction_origin; the recomputed value must
    # match the populator). Fall back to the old name so the fixture works pre/post rename.
    populated = getattr(rf, "grid_reduction_origin",
                        getattr(rf, "per_feature_accumulator", None))
    return (None, populated, gro_recomputed)


# NON-NORM grid-collapse: column-energy accumulation (sum of x^2 over the batch/grid axis into
# a per-column [N] accumulator, two-level grid tile + host finalize). Structurally an M-collapse
# but NOT a norm backward -- if GRO fires here it is a PROPERTY, not a norm-bwd recognizer.
@helion.kernel(static_shapes=False)
def col_energy_collapse(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    m_block = hl.register_block_size(M)
    nb = (M + m_block - 1) // m_block
    blocks = x.new_empty([nb, N], dtype=torch.float32)
    for cta in hl.tile(M, block_size=m_block):
        acc = x.new_zeros(N, dtype=torch.float32)
        for mb in hl.tile(cta.begin, cta.end):
            acc += torch.sum(x[mb, :].to(torch.float32) ** 2, dim=0)
        blocks[cta.id, :] = acc
    return blocks.sum(0)


# NEAR-MISS 1: plain user-tiled rowsum (reduces over the INNER feature axis, NOT the grid; no
# unbacked inner re-tile) -> GRO must be False.
@helion.kernel(static_shapes=False)
def usertiled_rowsum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M], dtype=torch.float32, device=x.device)
    for tm in hl.tile(M, block_size=1):
        acc = hl.zeros([tm], dtype=torch.float32)
        for tn in hl.tile(N):
            acc = acc + x[tm, tn].to(torch.float32).sum(-1)
        out[tm] = acc
    return out


# NEAR-MISS 2: per-row feature reduce with a [M] accumulator (rms-like fwd shape) -> GRO False.
@helion.kernel(static_shapes=False)
def per_row_feature_reduce(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M, N], dtype=x.dtype, device=x.device)
    for tm in hl.tile(M, block_size=1):
        s = (x[tm, :].to(torch.float32) ** 2).sum(-1)
        out[tm, :] = (x[tm, :].to(torch.float32) * (s[:, None] + 1.0)).to(x.dtype)
    return out


def main():
    print(f"helion={helion.__file__}\n")
    x = torch.randn(2048, 1024, device=DEV, dtype=torch.float32)
    cases = [
        ("col_energy_collapse (NON-NORM grid-collapse)", col_energy_collapse, True),
        ("usertiled_rowsum (near-miss: inner reduce)", usertiled_rowsum, False),
        ("per_row_feature_reduce (near-miss: per-row)", per_row_feature_reduce, False),
    ]
    ok = True
    for nm, fn, expect_gro in cases:
        err, populated, gro = gro_pfa(fn, (x.clone(),))
        # PASS iff: no error, recomputed GRO matches expectation, AND the populated field agrees
        # with the recomputed value (the populator computes the same property).
        verdict = "OK" if (err is None and gro == expect_gro and populated == gro) else "MISMATCH"
        if verdict != "OK":
            ok = False
        print(f"  [{verdict:8s}] {nm:46s} populated_field={populated} GRO_recomputed={gro} "
              f"(expect GRO={expect_gro})" + (f" ERR={err}" if err else ""))
    print(f"\n=== GRO Gate-D fixture: {'PASS' if ok else 'FAIL'} ===")
    print("(GRO must fire on the non-norm grid-collapse AND agree with PFA, but NOT fire on the "
          "near-misses -- proving it is the ORIGIN=grid property, not a norm-bwd recognizer.)")


if __name__ == "__main__":
    main()
