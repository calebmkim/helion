"""GEN CELL: r3_bf16_carried_2d

ADVERSARIAL re-probe of the dtype-faithful residency cap at the CARRIED-2D-tile band (Band B).

A kl_div/jsd-style carried [M_BLOCK, R_BLOCK] accumulator reduction, run DIRECTLY in bf16
(itemsize=2) so the fact's ``itemsize`` records 2 (the ReductionLowering input over the rdim
is bf16, NOT a fp32-promoted .to(float32)). The carried-tile R_BLOCK ceiling is sized by

    _carried_tile_r_block_cap -> _carried_tile_budget(co_resident=1)
        = CARRIED_TILE_MAX_BYTES(16384) / (_resident_itemsize(fact) * num_carried_2d_tiles)

The accumulator Helion carries is fp32-wide REGARDLESS of input dtype, so the cap must divide
by max(fact.itemsize, 4) = 4, NOT the raw bf16 itemsize 2.

DIVERGENCE the cell falsifies:
  - CORRECT (fp32 floor): budget = 16384/(4*1) = 4096  -> R_BLOCK cap = 4096.
  - BUG  (raw bf16=2)   : budget = 16384/(2*1) = 8192  -> R_BLOCK cap = 8192  (2x over-size).

N is chosen WIDE (32768) so rdim = next_pow2(N) >= 8192: the carried-tile cap (4096) BINDS
below both LOOPED_CHUNK (16384) and the extent (32768). If the seed R_BLOCK is 8192 the carried
bf16 tile mis-sized (UNJUSTIFIED -- no fp32-floored cap explains 8192); 4096 = the fix holds.

Property point: ACCESS=user-tiled x ORIGIN=inner x EXTENT=static-wide(N=32768) x
CARRIED-RESIDENT=yes (last acc dim == reduction-tile dim, Band B) x CO-RESIDENCY=single carried
tile x REUSE=re-read (kl_div re-reads regardless) x DIMS=2 x PINNED-GRID=none x
DTYPE=bf16(itemsize=2 -- the saturation lever this cell turns at the CARRIED cap).
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


@helion.kernel(static_shapes=False)
def bf16_carried_kl(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """x: [M, N] bf16. kl_div-style carried-2D-tile reduction run DIRECTLY in bf16.

    Carry a [M_BLOCK, R_BLOCK] partial-sum tile (bf16) across the user-tiled N loop -- its
    LAST dim is the reduction-tile axis, so this is num_carried_2d_tiles == 1 (Band B). Each
    iter accumulates a bf16 per-element kl term; the final ``loss_sum.sum(dim=-1)`` folds the
    carried R-tile. The carried accumulator AND the fold input are bf16 (itemsize=2) so
    fact.itemsize == 2 -- the saturation lever for the carried-tile residency cap. The genuine
    resident tile is fp32-wide, so the carried-tile cap must floor itemsize at 4 (R_BLOCK=4096),
    not size to the raw bf16 width (R_BLOCK=8192)."""
    M, N = y_pred.shape
    block_size_n = hl.register_block_size(N)
    block_size_m = hl.register_block_size(M)
    loss = torch.zeros((M,), dtype=torch.float32, device=y_pred.device)
    for tile_m in hl.tile(M, block_size=block_size_m):
        # carried [M_BLOCK, R_BLOCK] bf16 accumulator: last dim is the reduction-tile axis.
        loss_sum = hl.zeros([tile_m, block_size_n], dtype=BF16)
        for tile_n in hl.tile(N, block_size=block_size_n):
            yp = y_pred[tile_m, tile_n]                 # [M_BLOCK, R_BLOCK] bf16
            yt = y_true[tile_m, tile_n]                 # [M_BLOCK, R_BLOCK] bf16
            loss_sum += yt * (yt - yp)                  # bf16 carried partial
        loss[tile_m] = loss_sum.sum(dim=-1).to(torch.float32)  # fold carried bf16 R-tile
    return loss


def main() -> None:
    print(f"helion={helion.__file__}\n")
    M = 4096
    N = 32768  # 2^15 -- rdim=next_pow2=32768 >> carried cap 4096, so the cap BINDS.
    y_pred = torch.randn(M, N, device=DEV, dtype=BF16)
    y_true = torch.rand(M, N, device=DEV, dtype=BF16)
    intended = {
        "cell": "r3_bf16_carried_2d",
        "saturation_axis": "dtype/itemsize=2 (bf16) at the CARRIED-2D-tile cap",
        "access": "user-tiled",
        "origin": "inner",
        "extent": f"static-wide(N={N}=2^15)",
        "carried_resident": True,
        "carried_band": "B (num_carried_2d_tiles==1)",
        "co_residency": "single carried tile",
        "reuse": "re-read",
        "dims": 2,
        "pinned_grid": None,
        "dtype": "bfloat16",
        "expected_r_block_fp32_floor": 4096,
        "buggy_r_block_raw_bf16": 8192,
    }
    v = check_kernel("bf16_carried_kl", bf16_carried_kl, (y_pred, y_true), intended)
    import json
    print(json.dumps(v, indent=2, default=repr))


if __name__ == "__main__":
    main()
