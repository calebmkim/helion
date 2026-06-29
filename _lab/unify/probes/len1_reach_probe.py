"""P3 reachability — try several kernel shapes to find one that mints >=2 ReductionFacts (the
precondition for the len==1 gate to under-fire). If NONE is constructible, the fence is UNREACHED
(logged, not a silent pass); if one IS, it is a confirmed under-firing hole (Gate-H BROADEN)."""

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


# V1: two independent grid loops, each reducing a DIFFERENT tensor over its own axis.
@helion.kernel(static_shapes=False)
def two_tensor_two_loop(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    P, Q = y.shape
    ox = torch.empty([M], dtype=torch.float32, device=x.device)
    oy = torch.empty([P], dtype=torch.float32, device=y.device)
    for tm in hl.tile(M):
        ox[tm] = x[tm, :].to(torch.float32).sum(-1)
    for tp in hl.tile(P):
        oy[tp] = y[tp, :].to(torch.float32).sum(-1)
    return ox, oy


# V2: same grid loop, two reductions of different tensors over the same row.
@helion.kernel(static_shapes=False)
def two_reduce_same_loop(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    ox = torch.empty([M], dtype=torch.float32, device=x.device)
    oy = torch.empty([M], dtype=torch.float32, device=x.device)
    for tm in hl.tile(M):
        ox[tm] = x[tm, :].to(torch.float32).sum(-1)
        oy[tm] = y[tm, :].to(torch.float32).amax(-1)
    return ox, oy


def probe(name, fn, args):
    bound = fn.bind(args)
    spec = bound.env.config_spec
    nrf = len(spec.reduction_facts)
    nrl = len(spec.reduction_loops)
    seeds = list(spec.compiler_seed_configs)
    fired = list(spec.autotuner_heuristics)
    print(f"  {name:28s} n_reduction_facts={nrf} n_reduction_loops={nrl} "
          f"fired={fired} n_seeds={len(seeds)}")
    return nrf, len(seeds)


def main():
    print(f"helion={helion.__file__}\n")
    x = torch.randn(4096, 2048, device=DEV, dtype=torch.float32)
    y = torch.randn(2048, 4096, device=DEV, dtype=torch.float32)
    ys = torch.randn(4096, 2048, device=DEV, dtype=torch.float32)
    results = []
    for nm, fn, args in [
        ("two_tensor_two_loop", two_tensor_two_loop, (x.clone(), y.clone())),
        ("two_reduce_same_loop", two_reduce_same_loop, (x.clone(), ys.clone())),
    ]:
        try:
            results.append(probe(nm, fn, args))
        except Exception as e:  # noqa: BLE001
            print(f"  {nm:28s} COMPILE-FAIL: {type(e).__name__}: {e}")
            results.append((None, None))
    multi = [r for r in results if r[0] and r[0] >= 2]
    print(f"\nKernels minting >=2 ReductionFacts: {len(multi)}")
    under = [r for r in results if r[0] and r[0] >= 2 and r[1] == 0]
    print(f"  of those, NO-FIRE (under-fire hole): {len(under)}")


if __name__ == "__main__":
    main()
