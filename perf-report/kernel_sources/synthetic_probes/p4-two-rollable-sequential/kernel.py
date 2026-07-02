"""STRESS-TEST KERNEL — P4: two ROLLABLE reductions, SEQUENTIAL (different graphs).

TAXONOMY POINT: two FULL_SLICE reductions, each the sole rdim in its own graph (so each
actually ROLLS), in separate sequential loops -> different graph_id -> NOT co-resident. Two
first-class ReductionFacts (or descriptors), each sized against its own extent + own budget.

WHY IT MATTERS: (1) the corpus's multi-reduction kernels all pair a reduction with a
PINNED/materialized secondary; none has two genuinely ROLLABLE reductions. (2) it is the
clean witness for the INVARIANT (PROMPT.md §2.7): two rollable reductions can NEVER be
co-resident (co-resident = same graph = >1 rdim = rolling blocked), so they are always a
SEQUENTIAL pair. (3) it stresses the relaxed `len(reduction_facts) >= 1` gate — the old
`==1` gate would no-fire this entirely.

VERIFIED possible at fc1dbaa0 (_lab/coresidency_probe2.py `sibling_diff_rdim`): n_graphs=4,
rolled=[1,3] — BOTH rolled, separate graphs.

IMPLEMENTER'S ASSERTIONS (add when Stage 1/2 exist):
  Tier 1: two FULL_SLICE descriptors, each rollable=True, in DIFFERENT coresidency_groups
          (different graph_id). The kernel FIRES (not gated out by a `==1` rule).
  Tier 2: each reduction sized against its OWN extent + own budget (sequential), not sharing.
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def two_sequential_reductions(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Two independent full-slice reductions over the same N, in SEPARATE loops (sequential
    passes -> separate graphs -> each rolls independently)."""
    M, N = x.shape
    out_sq = torch.empty([M], dtype=torch.float32, device=x.device)
    out_abs = torch.empty([M], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M):
        out_sq[tile_m] = (x[tile_m, :].to(torch.float32) ** 2).sum(-1)   # loop 1: rolls
    for tile_m in hl.tile(M):
        out_abs[tile_m] = x[tile_m, :].to(torch.float32).abs().sum(-1)   # loop 2 (sibling): rolls
    return out_sq, out_abs


def make_args(M: int = 2048, N: int = 4096, dtype=torch.float32, device="cuda"):
    return (torch.randn(M, N, device=device, dtype=dtype),)


def main() -> None:
    two_sequential_reductions(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
