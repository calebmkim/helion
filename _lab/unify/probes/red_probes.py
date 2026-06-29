"""COVERAGE / totality-regression probes (§5a probe-first) — written BEFORE the refactor for the
property-space points the prior effort's docs flagged as historically mis-handled.

IMPORTANT (corrected framing, completeness-critic Blocker A): when RUN at fc1dbaa0 these P1-P5 came
back ALL GREEN (no crash / no floor-1 / no no-fire) — the Rounds 4-5 fixes (Issues 7/8, already in
fc1dbaa0) had ALREADY closed these output-hole gaps. So P1-P5 are NOT a RED->GREEN demonstration;
they are permanent TOTALITY-REGRESSION GUARDS that the refactor must keep GREEN (and did,
byte-identically). The genuine RED->GREEN demonstrations of this run are the SIX generator/Gate-T-found
holes (the len==1 under-firing fence, the grid-tile-reduction collapse, the bf16 dtype residency, the
multi-rolled reduction_loops slot-misplacement, the joint multi-grid occupancy, and the
grid_reduction_origin cross-loop bleed) — see probes/gen/ and probes/gateT/, each RED at its
pre-fix SHA and GREEN after.

The property-space points (all GREEN at fc1dbaa0; kept as regression guards):
  P1  rollable_secondary       : rolled primary + user-tiled secondary reduction (Issue 7/8 area)
  P2  pinned_grid_secondary    : pinned-full-extent grid amax + a tunable grid sum (Issue 10b area)
  P3  materialized_full_slice  : TWO full-slice reductions in one loop (the Defect-1 invariance
                                 kernel: the same x[m,:].sum reports ∈reduction_loops alone,
                                 ∉ when a second full-slice reduction shares the loop)
  P4  rolled_plus_two_tiled    : rolled primary + TWO user-tiled secondaries (combination)
  P5  twopass_usertiled        : two user-tiled sequential reductions, different loops

Each probe's INTENDED property-point is recorded so the minter/Gate-T can confirm it LANDED.
Run: HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=<worktree> python red_probes.py
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import run_suite  # noqa: E402

torch.manual_seed(0)
DEV = "cuda"
BF16 = torch.bfloat16


# --- P1: rolled primary (N) + user-tiled secondary (K) ----------------------
@helion.kernel(static_shapes=False)
def rolled_plus_tiled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    _, K = y.shape
    out = torch.empty([M, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        s = (x[tile_m, :].to(torch.float32) ** 2).sum(-1)        # ROLLED over N
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(K):                                 # USER-TILED over K
            acc = torch.maximum(acc, torch.amax(y[tile_m, tile_k].to(torch.float32).abs(), dim=-1))
        out[tile_m, 0] = s + acc
    return out


# --- P2: pinned full-extent grid amax (C) + tunable grid sum (G) -------------
@helion.kernel(static_shapes=False)
def pinned_grid_secondary(x: torch.Tensor) -> torch.Tensor:
    M, G, C = x.shape
    C = hl.specialize(C)
    out = torch.empty([M, 1], dtype=torch.float32, device=x.device)
    for tile_m, tile_g, tile_c in hl.tile([M, G, C], block_size=[1, None, C]):
        per_g = torch.amax(x[tile_m, tile_g, tile_c].to(torch.float32).abs(), dim=-1)  # reduce C (pinned grid)
        s = per_g.sum(dim=-1)                                                          # reduce G (tunable grid)
        out[tile_m, 0] = s
    return out


# --- P3: TWO full-slice (materialized) reductions in ONE loop (Defect-1 invariance) ---
@helion.kernel(static_shapes=False)
def materialized_two_full_slice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two full-slice reductions x[m,:].sum and y[m,:].sum in the SAME loop. The roller
    rolls AT MOST ONE per loop (graphs_with_rolled_rdim), so at least one full-slice
    reduction MATERIALIZES (∉ reduction_loops) -> Defect 1 tells it r_block_resident=1
    (a lie: it is full-width N resident)."""
    M, N = x.shape
    out = torch.empty([M, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        a = (x[tile_m, :].to(torch.float32) ** 2).sum(-1)
        b = (y[tile_m, :].to(torch.float32)).sum(-1)
        out[tile_m, 0] = a + b
    return out


# --- P4: rolled primary (N) + TWO user-tiled secondaries (K, L) --------------
@helion.kernel(static_shapes=False)
def rolled_plus_two_tiled(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    _, K = y.shape
    _, L = z.shape
    out = torch.empty([M, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        s = (x[tile_m, :].to(torch.float32) ** 2).sum(-1)
        acck = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acck = torch.maximum(acck, torch.amax(y[tile_m, tile_k].to(torch.float32).abs(), dim=-1))
        accl = hl.zeros([tile_m], dtype=torch.float32)
        for tile_l in hl.tile(L):
            accl = accl + z[tile_m, tile_l].to(torch.float32).sum(-1)
        out[tile_m, 0] = s + acck + accl
    return out


# --- P5: two user-tiled sequential reductions, DIFFERENT loops ---------------
@helion.kernel(static_shapes=False)
def twopass_usertiled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Two user-tiled reductions in SEPARATE loops (sequential passes, each own budget).
    Co-residency separator test: neither should crush the other's tile (Issue 8 area)."""
    M, N = x.shape
    _, K = y.shape
    out = torch.empty([M, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        accn = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(N):
            accn = accn + x[tile_m, tile_n].to(torch.float32).sum(-1)
        acck = hl.zeros([tile_m], dtype=torch.float32)
        for tile_k in hl.tile(K):
            acck = torch.maximum(acck, torch.amax(y[tile_m, tile_k].to(torch.float32).abs(), dim=-1))
        out[tile_m, 0] = accn + acck
    return out


def main():
    print(f"helion={helion.__file__}\n")
    xN = torch.randn(8192, 4096, device=DEV, dtype=BF16)
    yK = torch.randn(8192, 2048, device=DEV, dtype=BF16)
    zL = torch.randn(8192, 1024, device=DEV, dtype=BF16)
    x3 = torch.randn(2048, 40, 128, device=DEV, dtype=BF16)
    probes = [
        ("P1_rolled_plus_tiled", rolled_plus_tiled, (xN.clone(), yK.clone()),
         {"access": "rolled+usertiled", "co_residency": "same-loop", "secondary": "tiled"}),
        ("P2_pinned_grid_secondary", pinned_grid_secondary, (x3.clone(),),
         {"access": "pinned-grid", "secondary": "tunable-grid-reduction"}),
        ("P3_materialized_two_full_slice", materialized_two_full_slice, (xN.clone(), xN.clone()),
         {"access": "materialized-full-slice x2", "defect": "1-invariance"}),
        ("P4_rolled_plus_two_tiled", rolled_plus_two_tiled,
         (xN.clone(), yK.clone(), zL.clone()),
         {"access": "rolled+2 usertiled", "secondary": "2 tiled"}),
        ("P5_twopass_usertiled", twopass_usertiled, (xN.clone(), yK.clone()),
         {"access": "2 usertiled", "co_residency": "different-loop"}),
    ]
    results = run_suite(probes)
    n_red = sum(1 for r in results if r["red"])
    print(f"\n=== {n_red}/{len(results)} RED at this SHA ===")


if __name__ == "__main__":
    main()
