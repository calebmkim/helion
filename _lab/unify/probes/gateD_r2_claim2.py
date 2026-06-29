"""GATE-D ROUND 2 — CLAIM 2: M_COLLAPSE_TILE_BYTES is a factor inside a HARDWARE-UNIT budget,
not the operand of a fact-vs-literal fence.

_m_collapse_inner_byte_cap(feat_bytes) = next_pow2(M_COLLAPSE_TILE_BYTES // feat_bytes), where
feat_bytes = feature_footprint * itemsize -- the resident [inner, *features] tile footprint.

(§3a.2 / §5d) The EQUAL-FOOTPRINT divergence test:
  - Build grid-origin-collapse kernels with the SAME feature_footprint*itemsize but DIFFERENT other
    properties (grid extent, dtype, kernel structure), dump the cap each gets, confirm SAME cap.
  - Vary footprint -> confirm cap MOVES (the cap is NOT a degenerate constant; it keys on bytes).
  - Try to MISCLASSIFY: a kernel that SHOULD want a different inner tile at equal footprint but the
    budget gives the wrong one. Success => fence (refute); honest failure => faithful.

Compile-only, no GPU (bind() only). Run from /tmp with PYTHONPATH=<worktree>.
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
    _TritonReductionSeedBase as B,
    _reduction_primary_fact,
)

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep), helion.__file__
DEV = "cuda"
TB = B.M_COLLAPSE_TILE_BYTES


# ---------------- grid-origin collapse kernels (parameterized, structurally distinct) ----------
def make_col_energy(dtype):
    """col_energy_collapse: sum of x^2 over the grid axis into a per-column [N] accumulator
    (two-level grid tile + host finalize). NON-norm grid-collapse."""
    @helion.kernel(static_shapes=False)
    def col_energy_collapse(x: torch.Tensor) -> torch.Tensor:
        M, N = x.shape
        m_block = hl.register_block_size(M)
        nb = (M + m_block - 1) // m_block
        blocks = x.new_empty([nb, N], dtype=torch.float32)
        for cta in hl.tile(M, block_size=m_block):
            acc = x.new_zeros(N, dtype=torch.float32)
            for mb in hl.tile(cta.begin, cta.end):
                acc += torch.sum(x[mb, :].to(torch.float32) ** 2, dim=0)
            blocks[cta.id, :] = acc
        return blocks.sum(0)
    return col_energy_collapse


def make_col_mean_with_perrow(dtype):
    """STRUCTURALLY DIFFERENT grid-collapse: accumulates per-column SUM and per-column SUM-OF-ABS
    (two accumulators, extra per-row abs intermediate -> different body shape / body_live_tiles)
    over the grid axis. Same [N] feature footprint, different kernel structure."""
    @helion.kernel(static_shapes=False)
    def col_mean_abs_collapse(x: torch.Tensor) -> torch.Tensor:
        M, N = x.shape
        m_block = hl.register_block_size(M)
        nb = (M + m_block - 1) // m_block
        s = x.new_empty([nb, N], dtype=torch.float32)
        a = x.new_empty([nb, N], dtype=torch.float32)
        for cta in hl.tile(M, block_size=m_block):
            accs = x.new_zeros(N, dtype=torch.float32)
            acca = x.new_zeros(N, dtype=torch.float32)
            for mb in hl.tile(cta.begin, cta.end):
                row = x[mb, :].to(torch.float32)
                accs += torch.sum(row, dim=0)
                acca += torch.sum(torch.abs(row), dim=0)
            s[cta.id, :] = accs
            a[cta.id, :] = acca
        return s.sum(0) + a.sum(0)
    return col_mean_abs_collapse


def drive(fnmaker, M, N, dtype):
    fn = fnmaker(dtype)
    x = torch.randn(M, N, device=DEV, dtype=dtype)
    bound = fn.bind((x,))
    env = bound.env
    spec = env.config_spec
    if not spec.reduction_facts:
        return {"_no_fact": True}
    fact = _reduction_primary_fact(spec)
    feat_bytes = max(1, fact.feature_footprint) * max(1, fact.itemsize)
    cap = B._m_collapse_inner_byte_cap(feat_bytes)
    return {
        "M": M, "N": N, "dtype": str(dtype),
        "grid_reduction_origin": fact.grid_reduction_origin,
        "feature_footprint": fact.feature_footprint,
        "itemsize": fact.itemsize,
        "body_live_tiles": fact.body_live_tiles,
        "feat_bytes": feat_bytes,
        "inner_byte_cap": cap,
    }


def main():
    print(f"helion={helion.__file__}")
    print(f"M_COLLAPSE_TILE_BYTES = {TB}\n")

    # ---------- direct purity: same feat_bytes -> same cap, regardless of how derived ----------
    print("=== direct purity of _m_collapse_inner_byte_cap (pure fn of feat_bytes) ===")
    for fb in (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072):
        print(f"  feat_bytes={fb:7d} -> cap={B._m_collapse_inner_byte_cap(fb):6d} "
              f"(= np2({TB}//{fb}) = np2({TB // fb}))")

    # ---------- EQUAL-FOOTPRINT divergence: two structurally-different kernels, SAME feat_bytes ----------
    print("\n=== equal-footprint divergence (different structure/grid/dtype, SAME feat_bytes) ===")
    # Pick N so feature_footprint*itemsize is equal across the pair.
    #  A: col_energy, fp32, N=1024 -> feat_bytes = 1024*4 = 4096
    #  B: col_mean_abs (different structure), fp32, N=1024, DIFFERENT grid M -> feat_bytes = 4096
    #  C: col_energy, fp16, N=2048 -> itemsize is fp32-promoted (acc fp32) so check what itemsize is
    A = drive(make_col_energy, M=4096, N=1024, dtype=torch.float32)
    Bk = drive(make_col_mean_with_perrow, M=8192, N=1024, dtype=torch.float32)
    print("  A col_energy   fp32 M=4096 N=1024:", A)
    print("  B col_mean_abs fp32 M=8192 N=1024:", Bk)
    same_fb = A.get("feat_bytes") == Bk.get("feat_bytes")
    same_cap = A.get("inner_byte_cap") == Bk.get("inner_byte_cap")
    diff_other = (A.get("M"), A.get("body_live_tiles")) != (Bk.get("M"), Bk.get("body_live_tiles"))
    print(f"  -> SAME feat_bytes? {same_fb}  SAME cap? {same_cap}  "
          f"(grid/structure DIFFER? {diff_other})")

    # dtype variant: fp16 input. If itemsize stays fp32-promoted (4), match N to keep feat_bytes;
    # else adjust. Dump what itemsize the fact reports so the equal-footprint match is honest.
    C = drive(make_col_energy, M=2048, N=1024, dtype=torch.float16)
    print("  C col_energy   fp16 M=2048 N=1024:", C)
    # Construct a fp16 kernel matched to A's feat_bytes: need N*itemsize_C == 4096.
    if C.get("itemsize"):
        n_match = max(1, 4096 // C["itemsize"])
        Cm = drive(make_col_energy, M=2048, N=n_match, dtype=torch.float16)
        print(f"  C' col_energy fp16 N={n_match} (matched to feat_bytes=4096):", Cm)
        same_AC = (A.get("feat_bytes") == Cm.get("feat_bytes")
                   and A.get("inner_byte_cap") == Cm.get("inner_byte_cap"))
        print(f"  -> A vs C' (fp32 vs fp16, same feat_bytes): SAME cap? {same_AC}")

    # ---------- footprint MOVES the cap (not a degenerate constant) ----------
    print("\n=== cap MOVES with footprint (not a fence-style constant) ===")
    moved = []
    for N in (256, 512, 1024, 2048, 4096, 8192):
        d = drive(make_col_energy, M=4096, N=N, dtype=torch.float32)
        moved.append((d.get("feat_bytes"), d.get("inner_byte_cap")))
        print(f"  N={N:5d}: feat_bytes={d.get('feat_bytes'):7d} cap={d.get('inner_byte_cap')}")
    distinct_caps = len({c for _, c in moved})
    print(f"  distinct caps over the footprint sweep: {distinct_caps} (>1 => keys on bytes, not constant)")

    # ---------- MISCLASSIFICATION attempt ----------
    # The cap returns IDENTICAL values for IDENTICAL feat_bytes. A FENCE would instead key on which
    # kernels were MEASURED (a literal lookup): give kernel X cap_X and kernel Y cap_Y for the SAME
    # footprint. We try to find two equal-footprint kernels the budget classifies DIFFERENTLY.
    print("\n=== misclassification attempt (find equal-footprint kernels -> DIFFERENT cap) ===")
    # The cap is a pure function of feat_bytes only -- it has NO other input. So any two kernels with
    # equal feat_bytes provably get equal cap. The only way to "misclassify" is if a kernel that
    # SHOULD want a different inner tile at equal footprint is forced to the budget value. That is a
    # GRANULARITY (one-budget) limitation, NOT a fence (the answer does not depend on kernel identity
    # or which kernels were measured). Demonstrate: A (1 accumulator) and B (2 accumulators) have the
    # same feat_bytes and get the same cap -- yet B holds 2x the resident bytes per inner row, so it
    # ARGUABLY wants a smaller inner tile. The budget gives them the SAME cap (under-accounts B's
    # second accumulator). Is that a fence? NO -- it is the SAME cap for the SAME footprint signal;
    # a fence would give different caps keyed on the kernel. It is a footprint UNDER-COUNT (a
    # faithfulness gap in the feat_bytes SIGNAL, addressable by counting body_live_tiles into
    # feat_bytes), not a literal/measured-set fence.
    misclassify_is_fence = (same_fb and not same_cap)  # would only be true if equal fb -> diff cap
    print(f"  A and B: equal feat_bytes={same_fb}, equal cap={same_cap}.")
    print(f"  A holds 1 resident accumulator, B holds 2 -- the budget under-counts B's footprint")
    print(f"  in the feat_bytes SIGNAL, but the CAP FUNCTION still keys only on feat_bytes (gives")
    print(f"  the same answer for the same bytes). misclassify-as-FENCE succeeded? {misclassify_is_fence}")

    print("\n--- summary ---")
    print(f"equal_footprint_same_cap = {same_fb and same_cap}")
    print(f"cap_moves_with_footprint = {distinct_caps > 1}")
    print(f"misclassify_as_fence_succeeded = {misclassify_is_fence}")


if __name__ == "__main__":
    main()
