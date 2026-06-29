"""CELL: multifact_mixed_access (minted probe).

TWO genuinely-independent ReductionFacts in SEPARATE loops, of DIFFERENT tracks:
  - loop 1: a ROLLED full-slice reduction  ``x[m, :].sum(-1)`` over N   (standard track --
            the roller rolls the rdim into a ``reduction_loops`` entry, so its axis is NOT a
            ``block_sizes`` tile -> ``_is_standard_reduction`` True for that fact)
  - loop 2: a USER-TILED reduction         ``for tk in hl.tile(K): acc += y[m,tk].sum(-1)``
            (user-tiled track -- the rdim IS a ``block_sizes`` tile)

The relaxed gate (``len(reduction_facts) >= 1``) FIRES on the 2-fact kernel and
``_reduction_primary_fact`` routes the SEED to the DOMINANT (max-``size_hint``) fact's track.
The probe question (target cell): which track does it route to, and does the OTHER loop's tile
get sized correctly -- specifically, does the user-tiled K axis (a real TILED reduction in the
LOWERING role) get FLOORED to block_size=1 when the seed lands on the standard-track rolled fact?

We exercise BOTH dominance orderings (N>K -> standard dominant; K>N -> user-tiled dominant) as
two kernels so the checker sees both routings. Same access mix, different which-is-dominant.
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_PROBES = os.path.abspath(os.path.join(_THIS, ".."))
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import check_kernel  # noqa: E402

torch.manual_seed(0)
DEV = "cuda"
BF16 = torch.bfloat16


# ROLLED full-slice (over N) in grid-loop1 + USER-TILED (over K) in grid-loop2.
# TWO SEPARATE top-level grid loops -> two genuinely-independent ReductionFacts of
# DIFFERENT tracks (standard rolled, user-tiled). The relaxed gate fires on the 2-fact
# kernel; _reduction_primary_fact routes the seed to the dominant (max size_hint) fact.
@helion.kernel(static_shapes=False)
def rolled_then_usertiled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    _, K = y.shape
    out_a = torch.empty([M], dtype=torch.float32, device=x.device)
    out_b = torch.empty([M], dtype=torch.float32, device=x.device)
    # GRID LOOP 1: ROLLED full-slice reduction over N (standard track)
    for tile_m in hl.tile(M, block_size=1):
        out_a[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
    # GRID LOOP 2 (separate): USER-TILED reduction over K (user-tiled track)
    for tile_m2 in hl.tile(M, block_size=1):
        acc = hl.zeros([tile_m2], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acc = acc + y[tile_m2, tile_k].to(torch.float32).sum(-1)
        out_b[tile_m2] = acc
    return out_a + out_b


# PROBE for whether TWO ReductionFacts is reachable at all: two ROLLED full-slice
# reductions over DIFFERENT axes in SEPARATE grid loops (the gate docstring's own example).
@helion.kernel(static_shapes=False)
def two_rolled_separate(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    _, K = y.shape
    out_a = torch.empty([M], dtype=torch.float32, device=x.device)
    out_b = torch.empty([M], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        out_a[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
    for tile_m2 in hl.tile(M, block_size=1):
        out_b[tile_m2] = y[tile_m2, :].to(torch.float32).sum(-1)
    return out_a + out_b


def main():
    print(f"helion={helion.__file__}\n")
    # ORDERING A: N (rolled, standard) DOMINATES K (user-tiled). Seed should route to standard.
    xN = torch.randn(8192, 8192, device=DEV, dtype=BF16)   # rolled over N=8192
    yK = torch.randn(8192, 2048, device=DEV, dtype=BF16)   # user-tiled over K=2048
    vA = check_kernel(
        "multifact_mixed_access__standard_dominant",
        rolled_then_usertiled, (xN.clone(), yK.clone()),
        {"cell": "multifact_mixed_access", "access": "rolled-fullslice + user-tiled",
         "co_residency": "different-loop", "dominant": "standard(rolled,N=8192)",
         "tracks": ["standard", "user-tiled"]},
    )

    # ORDERING B: K (user-tiled) DOMINATES N (rolled). Seed should route to user-tiled.
    xN2 = torch.randn(8192, 2048, device=DEV, dtype=BF16)  # rolled over N=2048
    yK2 = torch.randn(8192, 8192, device=DEV, dtype=BF16)  # user-tiled over K=8192
    vB = check_kernel(
        "multifact_mixed_access__usertiled_dominant",
        rolled_then_usertiled, (xN2.clone(), yK2.clone()),
        {"cell": "multifact_mixed_access", "access": "rolled-fullslice + user-tiled",
         "co_residency": "different-loop", "dominant": "user-tiled(K=8192)",
         "tracks": ["standard", "user-tiled"]},
    )

    # ARM C: two ROLLED full-slice reductions, separate loops -- is 2-fact reachable?
    xC = torch.randn(8192, 8192, device=DEV, dtype=BF16)
    yC = torch.randn(8192, 2048, device=DEV, dtype=BF16)
    vC = check_kernel(
        "multifact_mixed_access__two_rolled",
        two_rolled_separate, (xC.clone(), yC.clone()),
        {"cell": "multifact_mixed_access", "access": "two rolled-fullslice",
         "co_residency": "different-loop", "tracks": ["standard", "standard"]},
    )

    for tag, v in (("A standard-dominant", vA), ("B usertiled-dominant", vB),
                   ("C two-rolled", vC)):
        obs = v["observed"]
        ns = obs.get("normalized_cfg") or {}
        print(f"\n--- {tag}: {v['name']} ---")
        print(f"  red                   = {v['red']}")
        print(f"  reasons               = {v['reasons']}")
        print(f"  fired                 = {obs.get('fired')}")
        print(f"  n_reduction_facts     = {obs.get('n_reduction_facts')}")
        print(f"  n_matmul_facts        = {obs.get('n_matmul_facts')}")
        print(f"  lowering_reduction_axes = {obs.get('lowering_reduction_axes')}")
        print(f"  grid_block_ids        = {obs.get('grid_block_ids')}")
        print(f"  block_sizes_valid_ids = {obs.get('block_sizes_valid_ids')}")
        print(f"  reduction_loops_valid_ids = {obs.get('reduction_loops_valid_ids')}")
        print(f"  block_sizes (norm)    = {ns.get('block_sizes')}")
        print(f"  reduction_loops (norm)= {ns.get('reduction_loops')}")
        print(f"  fact[0]               = {obs.get('fact')}")
    return vA, vB, vC


if __name__ == "__main__":
    main()
