"""COVERAGE follow-up probes (completeness-critic Priority-2): un-minted cross-product cells.
Compile-only. All should be GREEN (no crash / no-fire / floor1-of-RESIDENT) and JUSTIFIED.

F1  fp8 reduction          : a reduction over an fp8 (itemsize=1) input -> _resident_itemsize floors
                             to 4 (fp32 accumulator); the dtype-residency path (lever 9) at itemsize=1.
F2  int8 reduction         : itemsize=1 int path.
F3  two_carried_2d_tiles   : num_carried_2d_tiles>=2 (two simultaneously-live [M_BLOCK,R_BLOCK]
                             accumulators in one loop) -> _carried_tile_budget divides by 2 (a distinct
                             arithmetic path the corpus [all ==1] never exercises). Conservative (more
                             tiles -> smaller chunk -> never spill).
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(_THIS, "..", "..", ".."))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

from checker import run_suite  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep)
DEV = "cuda"


@helion.kernel(static_shapes=False)
def fp8_rowsum(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.shape
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m):
        out[tm] = x[tm, :].to(torch.float32).sum(-1)
    return out


@helion.kernel(static_shapes=False)
def int8_rowsum(x: torch.Tensor) -> torch.Tensor:
    m, _n = x.shape
    out = torch.empty([m], dtype=torch.int32, device=x.device)
    for tm in hl.tile(m):
        out[tm] = x[tm, :].to(torch.int32).sum(-1)
    return out


@helion.kernel(static_shapes=False)
def two_carried_2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """TWO simultaneously-live [M_BLOCK, R_BLOCK] carried accumulators whose LAST DIM is the reduction
    dim (kl_div/jsd-style: carry the 2-D tile across the loop, reduce at the END) -> the
    num_carried_2d_tiles=2 path the corpus (all ==1) never exercises. The accumulators are NOT reduced
    inside the loop (that would make them [M_BLOCK] scalars, num_carried_2d_tiles=0)."""
    m, n = x.shape
    rb = hl.register_block_size(n)
    out = torch.empty([m, 2], dtype=torch.float32, device=x.device)
    for tm in hl.tile(m, block_size=1):
        a = hl.zeros([tm, rb], dtype=torch.float32)
        b = hl.zeros([tm, rb], dtype=torch.float32)
        for tn in hl.tile(n, block_size=rb):
            a = a + x[tm, tn].to(torch.float32) * y[tm, tn].to(torch.float32)
            b = b + x[tm, tn].to(torch.float32) * x[tm, tn].to(torch.float32)
        out[tm, 0] = a.sum(-1)
        out[tm, 1] = b.sum(-1)
    return out


def main():
    print(f"helion={helion.__file__}\n")
    probes = []
    try:
        xf8 = torch.randn(4096, 4096, device=DEV).to(torch.float8_e4m3fn)
        probes.append(("F1_fp8_rowsum", fp8_rowsum, (xf8,), {"dtype": "fp8", "itemsize": 1}))
    except Exception as e:  # noqa: BLE001
        print(f"  (fp8 unavailable: {type(e).__name__}: {e})")
    xi8 = torch.randint(-8, 8, (4096, 4096), device=DEV, dtype=torch.int8)
    probes.append(("F2_int8_rowsum", int8_rowsum, (xi8,), {"dtype": "int8", "itemsize": 1}))
    x = torch.randn(8192, 4096, device=DEV, dtype=torch.bfloat16)
    y = torch.randn(8192, 4096, device=DEV, dtype=torch.bfloat16)
    probes.append(("F3_two_carried_2d", two_carried_2d, (x, y), {"carried_2d_tiles": 2}))
    results = run_suite(probes)
    n_red = sum(1 for r in results if r["red"])
    print(f"\n=== {n_red}/{len(results)} RED at this SHA ===")


if __name__ == "__main__":
    main()
