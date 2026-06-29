"""CELL: sat_fp16_widereduce — SATURATION CHECK on the dtype/itemsize axis.

A WIDE (N >= 65536) inner reduction whose reduction-axis input is fp16 (itemsize=2,
NOT fp32-promoted), reduced via a plain ``.sum(-1)`` so the row is NOT re-read
(``row_reread=False``) -> the reduction cannot be held persistent and the standard
track ROLLS it into a ``reduction_loops`` LOOPED chunk. This is the fp16 wide-loop
saturation point: dtype/itemsize is NOT one of the §1 modeled axes, so the question
is whether the emitted looped chunk + num_warps TRACE to a property + a named cap, or
whether fp16 produces an UNJUSTIFIED config (a missing axis).

The §1 property point this lands on (all modeled):
  ACCESS         = standard (Helion-rolled .sum(-1), rdim rides reduction_loops)
  ORIGIN         = inner axis (N is the inner reduced dim, not a grid axis)
  EXTENT         = static, WIDE (N = 131072 = 2^17, >> 65536)
  CARRIED-RESIDENT = no carried 2-D accumulator (plain scalar-per-row sum -> [M_BLOCK])
  CO-RESIDENCY   = single reduction
  REUSE          = streamed (num_load==1, row_reread=False -> looped, NOT persistent)
  NON-REDUCTION-LOOP = none
  DIMS           = 2
  PINNED-GRID    = none (M is a plain tunable hl.tile(M) grid axis)

The NOT-modeled property being VARIED: dtype/itemsize. The reduction input is fp16
(``fact.itemsize`` should be 2, vs 4 for the fp32 sibling). The saturation question:
  - The LOOPED chunk r_block = min(LOOPED_CHUNK=16384, prev_pow2(ROW_PERSIST_MAX_BYTES
    // (M_BLOCK * itemsize * ff))). At a wide extent the byte budget at itemsize=2 is
    122880 elems (-> pp2 65536), at itemsize=4 is 61440 (-> pp2 32768); BOTH exceed
    LOOPED_CHUNK=16384, so BOTH clamp to 16384. The chunk is the occupancy-optimal
    LOOPED_CHUNK, not the byte cap -> JUSTIFIED + dtype-INVARIANT here.
  - Implied RESIDENT bytes of the chunk: fp16 16384*2 = 32 KiB; fp32 16384*4 = 64 KiB.
    Both << ROW_PERSIST_MAX_BYTES (240 KiB) -> no spill, the chunk is justified.
  - num_warps keys on ``fact.size_hint`` (element extent), not bytes: N=131072 > 16384
    -> 32 warps in BOTH dtypes. NARROW_W1 keys on input_load_itemsize but the row is
    far wider than NARROW_W1_MAX_BYTES (2048 B), so it never fires -> 32 warps JUSTIFIED.

So the PREDICTION is: fp16 produces the SAME (16384, 32-warp) looped config as fp32,
every field tracing to size_hint + LOOPED_CHUNK + ROW_PERSIST_MAX_BYTES (the looped
chunk and warps are keyed on element extent + a dtype-faithful byte cap, NOT on a dtype
literal). If so -> JUSTIFIED, not a missing axis. We confirm by also building the fp32
sibling fact in the same run and comparing the emitted seed + the implied resident bytes.
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
F16 = torch.float16
F32 = torch.float32


# WIDE fp16 reduction: plain .sum(-1) on the fp16 row (NO .to(fp32) on the reduction
# input, so fact.itemsize stays 2). row_reread=False (single streamed load, no apply
# pass) -> the standard track LOOPS the rdim. M is a plain tunable grid axis.
@helion.kernel(static_shapes=False)
def fp16_wide_sum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M):
        # reduce the fp16 row directly; accumulate the result (fp32 out) per row.
        out[tile_m] = x[tile_m, :].sum(-1).to(torch.float32)
    return out


# fp32 SIBLING: identical structure, fp32 input (itemsize=4) -- the comparison baseline
# for the resident-byte / chunk justification.
@helion.kernel(static_shapes=False)
def fp32_wide_sum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M):
        out[tile_m] = x[tile_m, :].sum(-1).to(torch.float32)
    return out


def _resident_bytes(seed: dict, itemsize: int) -> object:
    """Implied resident reduction-chunk bytes = chunk * itemsize (M_BLOCK=1 seed)."""
    if not seed:
        return None
    rl = seed.get("reduction_loops")
    if rl and rl[0] is not None:
        return rl[0] * itemsize
    if rl == [None]:
        return "persistent(full-extent)"
    return None


def main():
    print(f"helion={helion.__file__}\n")
    M = 1024
    N = 131072  # 2^17, WIDE (>> 65536)
    x16 = torch.randn(M, N, device=DEV, dtype=F16)
    x32 = torch.randn(M, N, device=DEV, dtype=F32)

    intended = {
        "cell": "sat_fp16_widereduce",
        "access": "standard (rolled .sum(-1), rdim in reduction_loops)",
        "origin": "inner",
        "extent": f"static WIDE N={N} (2^17)",
        "carried_resident": "none (scalar-per-row [M_BLOCK])",
        "co_residency": "single",
        "reuse": "streamed (num_load==1, row_reread=False -> looped)",
        "dims": 2,
        "pinned_grid": "none (M tunable hl.tile)",
        "varied_not_modeled_property": "dtype/itemsize = fp16 (itemsize 2, NOT fp32-promoted)",
    }

    v16 = check_kernel("fp16_wide_sum", fp16_wide_sum, (x16,), intended)
    v32 = check_kernel("fp32_wide_sum", fp32_wide_sum, (x32,),
                       {**intended, "varied_not_modeled_property": "dtype=fp32 baseline (itemsize 4)"})

    import json

    obs16 = v16["observed"]
    obs32 = v32["observed"]
    cfg16 = obs16.get("normalized_cfg") or {}
    cfg32 = obs32.get("normalized_cfg") or {}
    fact16 = obs16.get("fact") or {}
    fact32 = obs32.get("fact") or {}

    summary = {
        "fp16": {
            "red": v16["red"],
            "reasons": v16["reasons"],
            "fired": obs16.get("fired"),
            "n_reduction_facts": obs16.get("n_reduction_facts"),
            "itemsize": fact16.get("itemsize"),
            "input_load_itemsize": fact16.get("input_load_itemsize"),
            "size_hint": fact16.get("size_hint"),
            "row_reread": fact16.get("row_reread"),
            "num_carried_2d_tiles": fact16.get("num_carried_2d_tiles"),
            "block_sizes": cfg16.get("block_sizes"),
            "reduction_loops": cfg16.get("reduction_loops"),
            "num_warps": cfg16.get("num_warps"),
            "implied_resident_bytes": _resident_bytes(cfg16, fact16.get("itemsize") or 2),
        },
        "fp32": {
            "red": v32["red"],
            "reasons": v32["reasons"],
            "fired": obs32.get("fired"),
            "itemsize": fact32.get("itemsize"),
            "input_load_itemsize": fact32.get("input_load_itemsize"),
            "size_hint": fact32.get("size_hint"),
            "row_reread": fact32.get("row_reread"),
            "block_sizes": cfg32.get("block_sizes"),
            "reduction_loops": cfg32.get("reduction_loops"),
            "num_warps": cfg32.get("num_warps"),
            "implied_resident_bytes": _resident_bytes(cfg32, fact32.get("itemsize") or 4),
        },
    }
    print(json.dumps(summary, indent=2, default=repr))

    # Justification verdict (printed for the minter to read).
    print("\n=== SATURATION JUSTIFICATION ===")
    print(json.dumps(v16, indent=2, default=repr))


if __name__ == "__main__":
    main()
