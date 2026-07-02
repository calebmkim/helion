"""OUT-OF-SCOPE CONFIRMER — OOS2: strided / non-contiguous dim-0 reduction (the known cliff).

TAXONOMY POINT: a standard rolled reduction over the NON-contiguous outer (dim-0) axis — a
per-column / per-vocab statistic over a token minibatch. Per PROMPT.md §0 this is OUT OF SCOPE
(predicate 2: reduction axis is grid dim-0 AND its stride over the reduced elements != itemsize).
It is the known accepted weakness (ATTACK_REPORT.md: per-vocab amax 1.98x, per-channel 1.52x
slower than default) — the persistence byte-cap is blind to access pattern.

WHY IT'S HERE (not to fix): contiguity is the SOLE property OK to leave incompletely modeled
(PROMPT.md §3a-counter-4 / §0). This kernel confirms the heuristic RECOGNIZES it as the
out-of-scope predicate (so the generator/saturation-check does not flag it as a totality hole),
NOT that the heuristic handles it well. "Left as-is" must be distinguishable from "declined".

IMPLEMENTER'S ASSERTION (add when Stage 1 exists):
  the reduced axis matches the strided-dim-0 OOS predicate (grid dim-0, stride != itemsize, read
  off the recorded stride provenance) — so it is logged OUT-OF-SCOPE-by-predicate-2, never
  counted as a fall-through and never "fixed".
"""
from __future__ import annotations
import torch
import helion
import helion.language as hl


@helion.kernel(ignore_warnings=[helion.exc.TensorOperationInWrapper])
def per_column_amax(x: torch.Tensor) -> torch.Tensor:
    """out[v] = amax over the token axis (dim 0) of |x[:, v]|. Reduces the STRIDED outer axis
    (stride = V), grids the contiguous column axis — the non-contiguous-reduction cliff."""
    M, V = x.shape
    out = torch.empty([V], dtype=torch.float32, device=x.device)
    for tile_v in hl.tile(V):
        out[tile_v] = torch.amax(x[:, tile_v].to(torch.float32).abs(), dim=0)  # reduce dim-0 (strided)
    return out


def make_args(M: int = 8192, V: int = 32000, dtype=torch.float32, device="cuda"):
    return (torch.randn(M, V, device=device, dtype=dtype),)


def main() -> None:
    per_column_amax(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
