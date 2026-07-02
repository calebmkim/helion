"""STRESS-TEST KERNEL — P6: MIXED — a co-resident pair AND a separate sequential reduction.

TAXONOMY POINT: one loop with TWO co-resident reductions (rms_norm_bwd-style: a feature
reduction + a cross-row accumulation over the same tile), followed by a SEPARATE loop with a
third reduction over the SAME extent (sequential, different graph_id). So the kernel has both
a co-residency group of size 2 AND an independent sequential group — in one kernel.

WHY IT MATTERS: the human's suggestion. The corpus has this MIXED shape only in jsd, which is
tangled (4 reduction ops, 3 graphs, fused_linear epilogue — PROMPT.md §2.7). This is the
CLEAN version: it isolates "co-resident group + sequential group coexist" so the allocator's
group partitioning is testable without jsd's noise. The "same extent" twist checks that the
sequential reduction gets its OWN budget (NOT shared with the co-resident group) even when the
extents coincide — the faithful fix to the Issue-8 ÷4096 over-count.

IMPLEMENTER'S ASSERTIONS (add when Stage 1/2 exist):
  Tier 1: coresidency_groups has TWO groups — {feature, row-accum} together (same graph_id),
          {the third reduction} alone (different graph_id) — even though all reduce extent N.
  Tier 2: the co-resident pair shares ONE budget; the sequential reduction gets its OWN full
          budget (not divided by the co-resident group's footprint).
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def mixed_coresident_sequential(x: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Loop 1: a co-resident pair (per-row feature sum over N  +  cross-row accum over M).
       Loop 2 (sequential): a third reduction over the SAME N extent.
    All three reduce a width-N axis; loops 1 and 2 are separate graphs."""
    M, N = x.shape
    row_feat = torch.empty([M], dtype=torch.float32, device=x.device)
    col_accum = torch.empty([N], dtype=torch.float32, device=x.device)
    row_other = torch.empty([M], dtype=torch.float32, device=x.device)
    # Loop 1: two co-resident reductions over the shared [m, N] tile.
    for tile_m in hl.tile(M):
        xt = x[tile_m, :].to(torch.float32)
        gt = g[tile_m, :].to(torch.float32)
        row_feat[tile_m] = xt.sum(-1)              # FULL_SLICE feature reduction
        col_accum[:] += (xt * gt).sum(0)           # cross-row accumulation (co-resident)
    # Loop 2 (sequential pass, separate graph): a third reduction over the SAME N extent.
    for tile_m in hl.tile(M):
        row_other[tile_m] = (x[tile_m, :].to(torch.float32) ** 2).sum(-1)
    return row_feat, col_accum, row_other


def make_args(M: int = 2048, N: int = 4096, dtype=torch.float32, device="cuda"):
    return (
        torch.randn(M, N, device=device, dtype=dtype),
        torch.randn(M, N, device=device, dtype=dtype),
    )


def main() -> None:
    mixed_coresident_sequential(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
