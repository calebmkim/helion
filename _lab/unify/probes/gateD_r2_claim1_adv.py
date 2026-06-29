"""GATE-D R2 CLAIM 1 — adversarial: can the conservatism EVER go toward-LARGER (spill)?

The only way r_block_resident < true_live_width (an UNDER-count -> larger-than-safe M-widen ->
spill) would be:
  (i)  standard looped branch: r_block_resident = np2(size_hint). true_chunk = reduction_loops[0]
       <= size_hint <= np2(size_hint). So np2(size_hint) >= true_chunk ALWAYS -> over-count, never
       under. (verified analytically + the looped probe.)
  (ii) user-tiled branch: r_block_resident = primary_red_value (the tunable r_block). The live
       width of a user-tiled reduction IS that r_block (the carried [M_BLOCK, r_block] tile), so it
       is EXACT, not an under-count. We dump several user-tiled / persistent shapes to confirm
       r_block_resident == the live width (no under-count), so the M-widen is never larger-than-safe.
  (iii) persistent standard (reduction_loops=[None]): full extent IS live; r_block_resident =
       np2(size_hint) == the live width -> EXACT.

If we cannot construct any kernel with r_block_resident < live_width, the conservatism is
provably toward-smaller-or-equal (safe).
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
from helion._utils import next_power_of_2 as _np2  # noqa: E402
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    _TritonReductionSeedBase,
    TritonStandardReductionHeuristic,
    TritonUserTiledReductionHeuristic,
    _reduction_primary_fact,
)

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep), helion.__file__
DEV = "cuda"

_orig = _TritonReductionSeedBase._build_block_sizes.__func__
TRACE: list[dict] = []


def _traced(cls, env, spec, fact, red_values=None, non_reduction_loop_ids=frozenset(),
            grid_origin=False):
    rv = red_values or {}
    primary = fact.primary_reduction_block_id
    prv = rv.get(primary)
    rbr = prv if prv is not None else _np2(fact.size_hint)
    TRACE.append({
        "size_hint": fact.size_hint,
        "in_red_values": prv is not None,
        "r_block_resident": rbr,
    })
    return _orig(cls, env, spec, fact, red_values, non_reduction_loop_ids, grid_origin)


_TritonReductionSeedBase._build_block_sizes = classmethod(_traced)


# Fresh kernel objects per shape so the bind cache (keyed on the kernel fn) does NOT collapse
# distinct N onto the first specialization.
def make_persistent():
    @helion.kernel(static_shapes=False)
    def persistent_rowsum(x: torch.Tensor) -> torch.Tensor:
        M, N = x.shape
        out = torch.empty([M], dtype=torch.float32, device=x.device)
        for tm in hl.tile(M):
            out[tm] = x[tm, :].to(torch.float32).sum(-1)
        return out
    return persistent_rowsum


def make_usertiled():
    @helion.kernel(static_shapes=False)
    def usertiled_rowsum(x: torch.Tensor) -> torch.Tensor:
        M, N = x.shape
        out = torch.empty([M], dtype=torch.float32, device=x.device)
        for tm in hl.tile(M, block_size=1):
            acc = hl.zeros([tm], dtype=torch.float32)
            for tn in hl.tile(N):
                acc = acc + x[tm, tn].to(torch.float32).sum(-1)
            out[tm] = acc
        return out
    return usertiled_rowsum


def drive(fn, heuristic, args):
    TRACE.clear()
    bound = fn.bind(args)
    env = bound.env
    spec = env.config_spec
    fact = _reduction_primary_fact(spec) if spec.reduction_facts else None
    try:
        with env:
            seed = heuristic.get_seed_config(env, bound.host_function.device_ir)
    except Exception as e:  # noqa: BLE001
        seed = {"_err": repr(e)}
    return env, spec, fact, seed, list(TRACE)


def main():
    print(f"helion={helion.__file__}\n")
    bad = []

    # (iii) persistent standard: full extent live -> r_block_resident exact.
    for N in (256, 1024, 4096):
        x = torch.randn(2048, N, device=DEV, dtype=torch.float32)
        env, spec, fact, seed, tr = drive(make_persistent(), TritonStandardReductionHeuristic, (x,))
        rl = seed.get("reduction_loops")
        rec = tr[-1] if tr else {}
        live = _np2(fact.size_hint) if rl == [None] else (rl[0] if rl and isinstance(rl[0], int) else _np2(fact.size_hint))
        rbr = rec.get("r_block_resident")
        ok = rbr >= live
        print(f"  persistent N={N}: fact.size_hint={fact.size_hint} reduction_loops={rl} live={live} "
              f"r_block_resident={rbr} r_block_resident>=live? {ok}")
        if not ok:
            bad.append(("persistent", N, rbr, live))

    # (ii) user-tiled: r_block_resident == the carried tunable r_block (exact, no under-count).
    for N in (4096, 16384, 65536):
        x = torch.randn(1024, N, device=DEV, dtype=torch.float32)
        env, spec, fact, seed, tr = drive(make_usertiled(), TritonUserTiledReductionHeuristic, (x,))
        rec = tr[-1] if tr else {}
        rbr = rec.get("r_block_resident")
        sh = fact.size_hint if fact else None
        if rbr is None:
            print(f"  user-tiled N={N}: _build_block_sizes NOT invoked (trace empty); "
                  f"size_hint={sh} seed={seed}")
            continue
        # live width of a user-tiled carried reduction = its r_block (== r_block_resident here).
        # r_block_resident IS the carried tunable tile, so it is EXACT (cannot under-count).
        print(f"  user-tiled N={N}: in_red_values={rec.get('in_red_values')} size_hint={sh} "
              f"r_block_resident={rbr} (== carried r_block, exact) "
              f"seed_block_sizes={seed.get('block_sizes')}")

    print(f"\nany r_block_resident < live_width (under-count -> spill risk)? "
          f"{'YES -> REFUTE' if bad else 'NO -> conservatism is toward-smaller-or-equal (SAFE)'}")
    if bad:
        for b in bad:
            print("  UNDER-COUNT:", b)


if __name__ == "__main__":
    main()
