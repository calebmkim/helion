"""GATE-T regression probe: JOINT multi-tunable-grid-axis occupancy.

A kernel with TWO TUNABLE grid axes (hl.tile([M, P])) + an inner reduction. Before the fix,
_m_axis_occupancy_cap computed grid//floor PER AXIS, so each widened independently to 128 and the
JOINT post-widen grid collapsed to 16 programs -- 66x under the occupancy floor (num_sm*MIN_WAVES).
After the fix (distribute the collapse budget geometrically across the n tunable m-axes): each axis
caps at the n-th root so the PRODUCT stays within budget (post-grid >= floor). Compile-only.
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(_THIS, "..", "..", "..", ".."))
if _WT not in sys.path:
    sys.path.insert(0, _WT)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402
from helion.runtime import get_num_sm  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep)
DEV = "cuda"
MIN_WAVES = 8


@helion.kernel(static_shapes=False)
def dual_grid(x: torch.Tensor) -> torch.Tensor:
    m, p, _r = x.shape
    out = torch.empty([m, p], dtype=torch.float32, device=x.device)
    for tm, tp in hl.tile([m, p]):
        out[tm, tp] = x[tm, tp, :].to(torch.float32).sum(-1)
    return out


def main():
    print(f"helion={helion.__file__}\n")
    gm, gp = 512, 512
    x = torch.randn(gm, gp, 16, device=DEV, dtype=torch.bfloat16)
    bound = dual_grid.bind((x,))
    spec = bound.env.config_spec
    seed = dict(list(spec.compiler_seed_configs)[0])
    bs = seed.get("block_sizes")
    num_sm = get_num_sm(bound.env.device)
    floor = num_sm * MIN_WAVES
    # block_sizes order = spec.block_sizes.valid_block_ids(); the two grid M tiles are positions 0,1.
    post = (gm // max(1, bs[0])) * (gp // max(1, bs[1])) if bs and len(bs) >= 2 else None
    under = post is not None and post < floor
    print(f"block_sizes={bs} num_sm={num_sm} occ_floor={floor}")
    print(f"post-widen grid = ({gm}/{bs[0]})*({gp}/{bs[1]}) = {post} programs")
    print(f"\n=== joint grid occupancy: {'RED (grid under floor)' if under else 'GREEN'} ===")


if __name__ == "__main__":
    main()
