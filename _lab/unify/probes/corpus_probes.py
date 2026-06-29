"""EXTENDED PROBE CORPUS — adversarial property-cross-product points beyond the 5 RED probes.

Seeds the generator's corpus (§5b) and Gate-T's avoid-list. Each kernel targets a CELL of the
property cross-product (ACCESS x ORIGIN x EXTENT x CARRIED-RESIDENT x CO-RESIDENCY x DIMS x
PINNED-GRID) the base 5 probes (red_probes.py) and the 27 corpus kernels miss. All run through the
mechanical checker (compile-only); a fall-through = a totality hole = a work-item.

These are hand-authored seeds; the steady-state generator MINTS more via workflow. Kept permanent.
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
F32 = torch.float32


# C1: 3-D user-tiled reduction over the innermost axis (DIMS=3, ACCESS=user-tiled, ORIGIN=inner)
@helion.kernel(static_shapes=False)
def three_d_inner_reduce(x: torch.Tensor) -> torch.Tensor:
    M, P, N = x.shape
    out = torch.empty([M, P], dtype=torch.float32, device=x.device)
    for tile_m, tile_p in hl.tile([M, P]):
        acc = hl.zeros([tile_m, tile_p], dtype=torch.float32)
        for tile_n in hl.tile(N):
            acc = acc + x[tile_m, tile_p, tile_n].to(torch.float32).sum(-1)
        out[tile_m, tile_p] = acc
    return out


# C2: full-slice reduce + full-width apply (reduce-then-apply, ACCESS=full-slice, NON-RED-LOOP)
@helion.kernel(static_shapes=False)
def reduce_then_apply_fullslice(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M, N], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        s = x[tile_m, :].to(torch.float32).sum(-1)         # full-slice reduce over N
        out[tile_m, :] = (x[tile_m, :].to(torch.float32) / (s[:, None] + 1.0)).to(x.dtype)  # apply
    return out


# C3: carried 2-D accumulator [M_BLOCK, R_BLOCK] over a user-tiled axis (CARRIED-RESIDENT, like kl_div)
@helion.kernel(static_shapes=False)
def carried_2d_usertiled(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(N):
            # a per-element product carried as a 2-D [M_BLOCK, R_BLOCK] tile then reduced
            prod = x[tile_m, tile_n].to(torch.float32) * y[tile_m, tile_n].to(torch.float32)
            acc = acc + prod.sum(-1)
        out[tile_m] = acc
    return out


# C4: two reductions over the SAME loop, different extents (CO-RESIDENCY=same-loop), user-tiled
@helion.kernel(static_shapes=False)
def same_loop_two_reduce(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M, 2], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        smax = hl.full([tile_m], float("-inf"), dtype=torch.float32)
        ssum = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(N):
            v = x[tile_m, tile_n].to(torch.float32)
            smax = torch.maximum(smax, torch.amax(v, dim=-1))
            ssum = ssum + v.sum(-1)
        out[tile_m, 0] = smax
        out[tile_m, 1] = ssum
    return out


# C5: 4-D reduction (DIMS=4), reduce innermost, grid over outer two
@helion.kernel(static_shapes=False)
def four_d_inner_reduce(x: torch.Tensor) -> torch.Tensor:
    A, B, C, N = x.shape
    out = torch.empty([A, B, C], dtype=torch.float32, device=x.device)
    for tile_a, tile_b, tile_c in hl.tile([A, B, C]):
        acc = hl.zeros([tile_a, tile_b, tile_c], dtype=torch.float32)
        for tile_n in hl.tile(N):
            acc = acc + x[tile_a, tile_b, tile_c, tile_n].to(torch.float32).sum(-1)
        out[tile_a, tile_b, tile_c] = acc
    return out


# C6: grid-pinned-M full-slice reduce (the vLLM idiom, NO secondary) -- standard materialized
@helion.kernel(static_shapes=False)
def gridpin_fullslice(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M, 1], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):
        out[tile_m, 0] = x[tile_m, :].to(torch.float32).abs().amax(-1)
    return out


def main():
    print(f"helion={helion.__file__}\n")
    x3 = torch.randn(1024, 64, 4096, device=DEV, dtype=BF16)
    x2 = torch.randn(8192, 4096, device=DEV, dtype=BF16)
    y2 = torch.randn(8192, 4096, device=DEV, dtype=BF16)
    x4 = torch.randn(64, 32, 16, 2048, device=DEV, dtype=BF16)
    probes = [
        ("C1_three_d_inner_reduce", three_d_inner_reduce, (x3.clone(),),
         {"dims": 3, "access": "user-tiled", "origin": "inner"}),
        ("C2_reduce_then_apply_fullslice", reduce_then_apply_fullslice, (x2.clone(),),
         {"access": "full-slice", "non_red_loop": True}),
        ("C3_carried_2d_usertiled", carried_2d_usertiled, (x2.clone(), y2.clone()),
         {"carried_resident": True, "access": "user-tiled"}),
        ("C4_same_loop_two_reduce", same_loop_two_reduce, (x2.clone(),),
         {"co_residency": "same-loop", "n_reduce": 2}),
        ("C5_four_d_inner_reduce", four_d_inner_reduce, (x4.clone(),),
         {"dims": 4, "access": "user-tiled"}),
        ("C6_gridpin_fullslice", gridpin_fullslice, (x2.clone(),),
         {"access": "full-slice", "pinned_grid": True}),
    ]
    results = run_suite(probes)
    n_red = sum(1 for r in results if r["red"])
    print(f"\n=== {n_red}/{len(results)} RED at this SHA ===")


if __name__ == "__main__":
    main()
