"""P3 probe — does the len(reduction_facts)==1 eligibility gate UNDER-FIRE on a kernel with
TWO rolled reductions over different axes (different graphs -> 2 ReductionFacts)?

The refactor-critic flagged the `len(reduction_facts)==1` gate (triton.py:_triton_reduction_eligible)
as a possible under-firing FENCE: two rolled reductions in different graphs mint 2 facts -> the gate
REJECTS -> the kernel falls back to the (bad) upstream default. Asymmetric vs the user-tiled track,
which folds multi-axis into ONE fact + secondaries. This probe constructs `out1 = x.sum(-1);
out2 = x.sum(-2)` (two reducible axes) and checks: how many ReductionFacts, does the seed fire?

A no-fire on an ELIGIBLE multi-reduction = a totality hole (Gate-H BROADEN). Compile-only.
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

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep)
DEV = "cuda"


# Two rolled reductions over DIFFERENT axes. out1 reduces the row (dim 1), out2 reduces the
# column (dim 0). Each is a full-slice .sum the roller can roll -> potentially 2 reduction_loops
# specs -> 2 ReductionFacts.
@helion.kernel(static_shapes=False)
def two_axis_sum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    out_row = torch.empty([M], dtype=torch.float32, device=x.device)
    out_col = torch.empty([N], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M):
        out_row[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
    for tile_n in hl.tile(N):
        out_col[tile_n] = x[:, tile_n].to(torch.float32).sum(0)
    return out_row, out_col


def main():
    print(f"helion={helion.__file__}\n")
    x = torch.randn(4096, 4096, device=DEV, dtype=torch.float32)
    bound = two_axis_sum.bind((x,))
    spec = bound.env.config_spec
    seeds = list(spec.compiler_seed_configs)
    fired = list(spec.autotuner_heuristics)
    nrf = len(spec.reduction_facts)
    print(f"n_reduction_facts={nrf}")
    print(f"heuristics_fired={fired}")
    print(f"n_seeds={len(seeds)}")
    if seeds:
        print(f"seed={dict(seeds[0])}")
    # The hole: >=2 reduction facts (or 2 reduction_loops) + NO seed fired = under-fire.
    n_rl = len(spec.reduction_loops)
    print(f"n_reduction_loops={n_rl}")
    under_fire = (nrf >= 2 or n_rl >= 2) and not seeds
    print(f"\nUNDER-FIRE (multi-reduction eligible but no seed): {under_fire}")
    if nrf == 1 and seeds:
        print("(folded into ONE fact + fired -- no hole)")
    elif nrf >= 2 and not seeds:
        print("(CONFIRMED P3 hole: len==1 gate rejects a 2-fact kernel -> bad default)")


if __name__ == "__main__":
    main()
