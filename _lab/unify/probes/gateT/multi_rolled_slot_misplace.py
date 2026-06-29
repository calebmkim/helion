"""GATE-T regression probe: multi-rolled-reduction reduction_loops SLOT placement.

A kernel with TWO rolled standard reductions in separate grid loops, the DOMINANT (wider) one
SECOND so its rolled axis is reduction_loops slot 1, not slot 0. Before the fix, the standard track
emitted a LENGTH-1 reduction_loops=[r_block] sized for the dominant fact; BlockIdSequence._normalize
filled POSITIONALLY -> the dominant's LOOP chunk landed on slot-0 (the non-dominant 512 axis) and the
dominant 262144 axis got _fill_missing()->None=PERSISTENT (a 1 MB resident tile held persistent vs the
heuristic's own LOOP decision). RED. After the fix (build reduction_loops BY SLOT, sizing each rolled
reduction against its own extent): the dominant axis correctly gets its chunk. Compile-only.
"""

from __future__ import annotations

import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_WT = os.path.abspath(os.path.join(_THIS, "..", "..", "..", ".."))
_PROBES = os.path.abspath(os.path.join(_THIS, ".."))
if _PROBES not in sys.path:
    sys.path.insert(0, _PROBES)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
import helion.language as hl  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT + os.sep)
DEV = "cuda"
ROW_PERSIST_MAX_BYTES = 245760


@helion.kernel(static_shapes=False)
def two_rolled_dominant_second(x: torch.Tensor, y: torch.Tensor) -> tuple:
    """out1 reduces x over a NARROW N1 (rolled, slot 0); out2 reduces y over a WIDE N2 (rolled,
    slot 1 = the DOMINANT, max-extent). The dominant rolled reduction is NOT in slot 0."""
    m1, _n1 = x.shape
    m2, _n2 = y.shape
    o1 = torch.empty([m1], dtype=torch.float32, device=x.device)
    o2 = torch.empty([m2], dtype=torch.float32, device=y.device)
    for tm in hl.tile(m1):
        o1[tm] = x[tm, :].to(torch.float32).sum(-1)
    for tn in hl.tile(m2):
        o2[tn] = y[tn, :].to(torch.float32).sum(-1)
    return o1, o2


def main():
    print(f"helion={helion.__file__}\n")
    # N1=512 narrow, N2=262144 wide (dominant). Dominant fp32 tile = 262144*4 = 1MB >> cap -> LOOP.
    x = torch.randn(256, 512, device=DEV, dtype=torch.float32)
    y = torch.randn(256, 262144, device=DEV, dtype=torch.float32)
    bound = two_rolled_dominant_second.bind((x, y))
    spec = bound.env.config_spec
    rl_ids = spec.reduction_loops.valid_block_ids()
    seed = dict(list(spec.compiler_seed_configs)[0])
    norm = dict(seed)
    env = bound.env
    with env:
        spec.normalize(norm)
        dom_bid = max(rl_ids, key=lambda b: env.block_sizes[b].size_hint())
        dom_extent = env.block_sizes[dom_bid].size_hint()
    rl = norm.get("reduction_loops")
    dom_slot = rl_ids.index(dom_bid)
    dom_val = rl[dom_slot] if rl and dom_slot < len(rl) else None
    dom_bytes = dom_extent * 4
    must_loop = dom_bytes > ROW_PERSIST_MAX_BYTES
    red = must_loop and (dom_val is None)  # held persistent despite exceeding the cap
    print(f"reduction_loops valid ids (slots): {rl_ids}")
    print(f"normalized reduction_loops: {rl}")
    print(f"dominant rdim bid={dom_bid} extent={dom_extent} slot={dom_slot} value={dom_val}")
    print(f"dominant fp32 bytes={dom_bytes} cap={ROW_PERSIST_MAX_BYTES} must_loop={must_loop}")
    print(f"\n=== multi-rolled slot placement: "
          f"{'RED (dominant forced persistent)' if red else 'GREEN'} ===")


if __name__ == "__main__":
    main()
