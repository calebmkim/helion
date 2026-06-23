"""FRESH Gate D probe round 3: total_numel (block-dim hints) vs real accessed extents.

The crux: total_numel = product(block_size_hints) = the ITERATION-SPACE numel. accessed_numel
= product of the TENSOR's stride!=0 dims. The full-extent test accessed_numel==total_numel
assumes the tiled iteration space == the data tensor's element count. Divergences:

(H) iteration space SMALLER than the buffer: tile over a sub-shape; a full read of a LARGER
    operand would have accessed_numel > total_numel (never ==, so EXCLUDED -> under-count).
(I) iteration space LARGER per-op than one operand whose real shape < tile (operand reused
    across the tile via indexing): accessed_numel < total_numel -> excluded (correct if bcast).
(J) a broadcast whose surviving dims happen to PRODUCT to total_numel (false FULL).
(K) 3-D pointwise (n_block_dims=3): does total_numel = M*N*K and full ops match?
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
        print(f"{name:36} NO pointwise fact "
              f"(red={len(spec.reduction_facts)} mm={len(spec.matmul_facts)} "
              f"acc={len(spec.accumulator_facts)})")
        for m in mof:
            print(f"     {m.kind:5} {str(m.dtype):14} acc={m.accessed_numel:>12} {m.tensor_name}")
        return
    f = pf[0]
    print(f"{name:36} total={f.total_numel} bytes/elem={f.bytes_per_elem} "
          f"nL={f.n_load} nS={f.n_store} hints={f.block_size_hints}")
    for m in mof:
        flag = "FULL" if m.accessed_numel == f.total_numel else "amort"
        print(f"     {m.kind:5} {str(m.dtype):14} acc={m.accessed_numel:>12} [{flag}] {m.tensor_name}")


@helion.kernel()
def add_3d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """3-D pointwise: tile over [A,B,C]. n_block_dims should be 3."""
    out = torch.empty_like(x)
    for ta, tb, tc in hl.tile(x.size()):
        out[ta, tb, tc] = x[ta, tb, tc] + y[ta, tb, tc]
    return out


@helion.kernel()
def add_larger_operand(x: torch.Tensor, big: torch.Tensor) -> torch.Tensor:
    """x is [M,N]; big is [M, 2N]; we read big[:, :N] sliced. The full big buffer is larger,
    but we only access the [M,N] view -> accessed_numel(view) = M*N = total. Tests that a
    SLICE of a larger buffer is measured by the accessed VIEW, not the storage."""
    out = torch.empty_like(x)
    m, n = x.size()
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + big[tm, tn]
    return out


@helion.kernel()
def sub_tile(big: torch.Tensor) -> torch.Tensor:
    """Tile over a sub-shape [M, N] of a [M, 2N] buffer; read the FULL big buffer? No -- we
    index big[tm, tn] over the [M,N] iteration space, so accessed = M*N. To get accessed >
    total we'd need a read whose own shape exceeds the tile -- not expressible pointwise."""
    m, n2 = big.size()
    out = torch.empty((m, n2 // 2), device=big.device, dtype=big.dtype)
    for tm, tn in hl.tile(out.size()):
        out[tm, tn] = big[tm, tn] * 2.0
    return out


def main():
    dev, dt = "cuda", torch.bfloat16

    print("=== (K) 3-D pointwise add ===")
    x3 = torch.randn(256, 256, 256, device=dev, dtype=dt)
    y3 = torch.randn(256, 256, 256, device=dev, dtype=dt)
    info("add_3d [256,256,256]", add_3d, (x3, y3))

    print("\n=== (H) slice of a LARGER operand (big[:, :N] accessed) ===")
    x = torch.randn(4096, 4096, device=dev, dtype=dt)
    big = torch.randn(4096, 8192, device=dev, dtype=dt)
    info("add x + big[:, :N]", add_larger_operand, (x, big[:, :4096]))

    print("\n=== (H2) sub_tile: iter space [M,N] over a [M,2N] storage ===")
    big2 = torch.randn(4096, 8192, device=dev, dtype=dt)
    info("sub_tile big[4096,8192]", sub_tile, (big2,))


if __name__ == "__main__":
    main()
