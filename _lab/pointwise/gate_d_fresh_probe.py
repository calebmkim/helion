"""FRESH Gate D divergence probe @ a1b5402 (stride-aware accessed_numel).

Tries to REFUTE the faithfulness of:
  accessed_numel = product(size_hint(shape[i]) for stride[i] != 0)
as a "distinct HBM elements" signal, and total_numel = product(block_size_hints).

NEW divergence attempts the prior probes did NOT cover:
  (A) transposed / non-contiguous-but-FULL tensor (all strides != 0 -> counted fully)
  (B) a tensor read by 2 ops (double count? per-op fact, so 2 facts)
  (C) dynamic/symbolic shape (size_hint vs real)
  (D) total_numel (product of block-dim hints) vs the real M*N of the accessed tensor
  (E) 1-D flatten kernel (.view(-1)) -- block dims vs tensor shape rank mismatch
  (F) a partial-extent op (slice) that is full-rank but touches < total_numel
"""

from __future__ import annotations

import torch

import helion
import helion.language as hl

from helion._compiler.autotuner_heuristics import compiler_seed_configs


def info(name, kfn, args):
    bound = kfn.bind(args)
    spec = bound.config_spec
    pf = spec.pointwise_facts
    mof = spec.memory_op_facts
    if not pf:
        # report why: which family fact fired
        why = []
        if spec.reduction_facts:
            why.append(f"reduction={len(spec.reduction_facts)}")
        if spec.matmul_facts:
            why.append(f"matmul={len(spec.matmul_facts)}")
        if spec.accumulator_facts:
            why.append(f"accum={len(spec.accumulator_facts)}")
        print(f"{name:30} NO pointwise fact ({','.join(why) or 'no memfacts/blocks'})")
        for m in mof:
            print(f"     memop kind={m.kind} dtype={m.dtype} ndim={m.ndim} "
                  f"accessed_numel={m.accessed_numel} name={m.tensor_name}")
        return
    f = pf[0]
    print(f"{name:30} total_numel={f.total_numel} n_block_dims={f.n_block_dims} "
          f"block_hints={f.block_size_hints} bytes/elem={f.bytes_per_elem} "
          f"n_load={f.n_load} n_store={f.n_store}")
    for m in mof:
        flag = "FULL" if m.accessed_numel == f.total_numel else "amort"
        print(f"     memop kind={m.kind:5} dtype={str(m.dtype):16} ndim={m.ndim} "
              f"accessed_numel={m.accessed_numel:>12} [{flag}] name={m.tensor_name}")


# ---- kernels ----

@helion.kernel()
def add2d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        out[tm, tn] = x[tm, tn] + y[tm, tn]
    return out


@helion.kernel()
def double_read(x: torch.Tensor) -> torch.Tensor:
    """Reads x twice (x + x) -- is x double-counted in bytes_per_elem? (2 load facts)."""
    out = torch.empty_like(x)
    for tm, tn in hl.tile(x.size()):
        a = x[tm, tn]
        b = x[tm, tn]
        out[tm, tn] = a * a + b
    return out


@helion.kernel()
def flatten_1d(x: torch.Tensor) -> torch.Tensor:
    """1-D flatten kernel: hl.tile over a flattened view. 1 block dim vs 2-D tensor."""
    out = torch.empty_like(x)
    n = x.size(0)
    for t in hl.tile(n):
        out[t] = x[t] * 2.0
    return out


def main():
    dev = "cuda"
    dt = torch.bfloat16

    print("=== (control) contiguous add2d ===")
    x = torch.randn(4096, 4096, device=dev, dtype=dt)
    y = torch.randn(4096, 4096, device=dev, dtype=dt)
    info("add2d contig", add2d, (x, y))

    print("\n=== (A) TRANSPOSED full tensor: y = y0.t() -- all strides != 0, full extent ===")
    # y0 is [4096,4096] contiguous; y0.t() is a full [4096,4096] view, strides (1,4096),
    # both != 0. accessed_numel should = 4096*4096 = total_numel -> counted FULL. Is that
    # FAITHFUL? The op DOES touch all 16M distinct HBM elements (just strided), so yes it
    # should count. But the *cost model* (a strided/uncoalesced read moves more effective
    # bytes) is NOT captured -- accessed_numel is distinct-elements, not effective traffic.
    y0 = torch.randn(4096, 4096, device=dev, dtype=dt)
    info("add2d x + y.t() (transposed)", add2d, (x, y0.t()))

    print("\n=== (A2) NON-CONTIGUOUS strided slice that is FULL extent (step view) ===")
    # x2 = base[::1, ::1] is contiguous; use base[:, :] of a larger buffer sliced to full.
    base = torch.randn(4096, 8192, device=dev, dtype=dt)
    xs = base[:, :4096]  # [4096,4096], strides (8192,1), both !=0, but only 4096 cols of 8192
    info("add2d x + base[:, :4096]", add2d, (x, xs))

    print("\n=== (B) tensor read by 2 ops (x+x) -- double count? ===")
    info("double_read(x)", double_read, (x,))

    print("\n=== (C) DYNAMIC / symbolic shape ===")
    # Force dynamic by constructing with a non-specialized size. Helion specializes by default;
    # try a shape that triggers size_hint vs real divergence.
    xd = torch.randn(4099, 4099, device=dev, dtype=dt)  # non-pow2, odd
    yd = torch.randn(4099, 4099, device=dev, dtype=dt)
    info("add2d 4099x4099 (non-pow2)", add2d, (xd, yd))

    print("\n=== (E) 1-D flatten kernel ===")
    xf = torch.randn(16, 1048576, device=dev, dtype=dt)  # 2-D tensor, 1-D tile over dim0? no
    x1 = torch.randn(16777216, device=dev, dtype=dt)  # true 1-D
    info("flatten_1d(1-D 16M)", flatten_1d, (x1,))


if __name__ == "__main__":
    main()
