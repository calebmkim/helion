"""FRESH Gate D probe round 2: the two sharpest divergence candidates.

(B-deep) double-read: is x counted TWICE in bytes_per_elem? Is that faithful traffic?
   Real HBM: a coalesced kernel reading x twice issues 2 loads but they hit L1/L2;
   effective HBM traffic ~ 1x x (cache), not 2x. So bytes_per_elem may OVER-count.
   BUT: is this a fact-faithfulness bug or an inherent cost-model limit? The fact CLAIMS
   "per-element HBM byte traffic (read+write), SUM over full-extent ops". By its own
   definition it sums PER-OP, so x+x = 2 ops = double. Question: does the DOC over-promise
   ("HBM byte traffic") vs what it computes (per-op itemsize sum)?

(G) broadcast whose accessed_numel ACCIDENTALLY equals total_numel.
   total_numel = product(block_size_hints). If a broadcast operand's non-stride-0 dims
   happen to multiply to total_numel, it would be COUNTED as full-extent though it is a
   broadcast. Construct: x[M,N] op v[M,N-as-2-dims]? Hard. Try a [K]-shaped broadcast where
   K == M*N. Or a square problem M==N with a [M,1]->? No, [M,1] gives M != M*N.
   The risk only arises if a real broadcast's surviving (stride!=0) dims product == M*N.

(F) partial-extent FULL-RANK op: out[tile_m, tile_n] writing only part. Or an op on a
   tensor whose real shape < block dims (a [M, N//2] companion read broadcast/indexed).
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl


def info(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    pf = spec.pointwise_facts
    mof = spec.memory_op_facts
    if not pf:
        print(f"{name:34} NO pointwise fact "
              f"(red={len(spec.reduction_facts)} mm={len(spec.matmul_facts)} "
              f"acc={len(spec.accumulator_facts)})")
        return
    f = pf[0]
    print(f"{name:34} total={f.total_numel} bytes/elem={f.bytes_per_elem} "
          f"nL={f.n_load} nS={f.n_store} hints={f.block_size_hints}")
    for m in mof:
        flag = "FULL" if m.accessed_numel == f.total_numel else "amort"
        print(f"     {m.kind:5} {str(m.dtype):14} acc={m.accessed_numel:>12} [{flag}] {m.tensor_name}")


@helion.kernel()
def triple_read(x: torch.Tensor) -> torch.Tensor:
    """x read 3 times -> 3 load facts? bytes/elem = 3*2 + 2 = 8 for ONE distinct buffer."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        v = x[tm, tn]
        out[tm, tn] = v + v * v
    return out


@helion.kernel()
def square_bcast_col(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """SQUARE M==N problem, c is [1, N] broadcast over rows. accessed_numel(c) = N (one
    stride!=0 dim). total_numel = M*N = N*N. N != N*N, so NOT counted. Confirms a square
    broadcast is still excluded (no accidental equality at this shape)."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + c[0, tn]
    return out


@helion.kernel()
def partial_store(x: torch.Tensor) -> torch.Tensor:
    """Write only the first half of columns (out[:, :N//2]) from a full read. The store op's
    accessed tensor is the [M, N//2] out -> accessed_numel = M*N//2 < total_numel? Depends on
    block dims: the tile is over x.size() = [M,N]. Tests partial-extent store handling."""
    m, n = x.size()
    out = torch.zeros_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] * 2.0
    return out


def main():
    dev, dt = "cuda", torch.bfloat16
    print("=== (B-deep) triple read of one buffer ===")
    x = torch.randn(4096, 4096, device=dev, dtype=dt)
    info("triple_read(x) [x x x]", triple_read, (x,))

    print("\n=== (G) square M==N broadcast col [1,N] ===")
    c = torch.randn(1, 4096, device=dev, dtype=dt)
    info("square_bcast_col M=N=4096", square_bcast_col, (x, c))

    print("\n=== (G2) square with [N] rank-1 broadcast, N==M ===")
    # add x[M,N] + bias[N], M==N==4096; bias accessed_numel = 4096, total = 16.7M -> amort
    from examples.add import add
    bias = torch.randn(4096, device=dev, dtype=dt)
    info("add(x, bias[N]) M=N", add, (x, bias))

    print("\n=== mixed dtype: fp32 x + bf16 y ? (different itemsizes) ===")
    xf = torch.randn(4096, 4096, device=dev, dtype=torch.float32)
    info("triple_read fp32", triple_read, (xf,))


if __name__ == "__main__":
    main()
