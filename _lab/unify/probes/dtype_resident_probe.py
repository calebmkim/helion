"""DTYPE saturation-axis regression probe (the bf16 finding from generator round 2).

A DIRECT bf16/fp16 reduction (no .to(fp32) before .sum) accumulates in fp32 REGARDLESS of input
dtype, so the resident [M_BLOCK, R_BLOCK] reduction tile is fp32-wide. Before the _resident_itemsize
fix, the residency byte-caps divided by fact.itemsize (=2 for bf16) -> UNDER-counted the resident
footprint 2x -> a wide bf16 row in the band N=65536..122880 wrongly chose PERSISTENT (reduction_loops
[None]) where the fp32 accumulator (N*4 bytes) exceeds ROW_PERSIST_MAX_BYTES=245760 and would spill.

After the fix (max(itemsize,4) on the loop-carried-fp32-accumulator caps): N=65536 bf16 correctly
LOOPS (the fp32-acc footprint 262144 > 245760). This was RED (wrong persist) at the pre-fix HEAD;
GREEN after. Permanent regression fixture. Compile-only.
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

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep)
DEV = "cuda"
ROW_PERSIST_MAX_BYTES = 245760


@helion.kernel(static_shapes=False)
def bf16_rowsum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty([M], dtype=torch.float32, device=x.device)
    for tm in hl.tile(M):
        out[tm] = x[tm, :].sum(-1)  # DIRECT bf16 reduce (fp32-promoted accumulator)
    return out


def main():
    print(f"helion={helion.__file__}\n")
    ok = True
    # Band where itemsize=2 says persist but the fp32 accumulator (N*4) exceeds the cap.
    for N in [65536, 98304, 122880]:
        x = torch.randn(1024, N, device=DEV, dtype=torch.bfloat16)
        bound = bf16_rowsum.bind((x,))
        spec = bound.env.config_spec
        seed = dict(list(spec.compiler_seed_configs)[0])
        rl = seed.get("reduction_loops")
        fp32_bytes = N * 4
        exceeds = fp32_bytes > ROW_PERSIST_MAX_BYTES
        looped = rl != [None]
        # FAITHFUL: if the fp32-accumulator footprint exceeds the cap, the seed must LOOP.
        verdict = "OK" if (looped == exceeds) else "RED"
        if verdict != "OK":
            ok = False
        print(f"  [{verdict:3s}] N={N} bf16: reduction_loops={rl} "
              f"({'LOOPED' if looped else 'PERSISTENT'}); fp32_acc={fp32_bytes}B "
              f"{'>cap (must loop)' if exceeds else '<=cap (may persist)'}")
    print(f"\n=== dtype resident probe: {'PASS' if ok else 'FAIL (wrong persist/loop at bf16)'} ===")


if __name__ == "__main__":
    main()
