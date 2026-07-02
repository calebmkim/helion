"""OUT-OF-SCOPE CONFIRMER — OOS1: jagged / data-dependent reduction extent.

TAXONOMY POINT: a reduction whose extent is data-dependent (a jagged `hl.jagged_tile` range,
size_hint = None). Per PROMPT.md §0 this is OUT OF SCOPE — the correct behavior is DECLINED
(the extent-keyed seed is undefined). NOT a defect to fix; this confirms the heuristic cleanly
DECLINES rather than crashing or emitting a garbage seed.

WHY IT USES THE EXISTING EXAMPLE: the jagged idiom is `hl.jagged_tile(seqlens)` (see
examples/jagged_softmax.py). Rather than author a fragile new jagged kernel, this confirmer
reuses the REAL corpus jagged kernel — a faithful instance of the predicate is what matters,
not a novel kernel. The implementer can point Stage-1 at `jagged_softmax_kernel` directly.

IMPLEMENTER'S ASSERTION (add when Stage 1 exists):
  the jagged reduction axis classifies DECLINED (or the kernel does not fire a reduction seed);
  NO crash, NO floored-to-1 garbage. "Handled" here == "correctly declined".
"""
from __future__ import annotations
import sys
import torch

# The faithful jagged exemplar lives in the helion examples (hl.jagged_tile). Import it rather
# than re-author the idiom. Resolve examples/ from the ACTIVE helion package (the one on
# PYTHONPATH), so this travels with whatever worktree is in use — never a hardcoded path.
import os
import helion as _helion

_EXAMPLES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(_helion.__file__))), "examples"
)
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)


def get_kernel():
    """Return the real corpus jagged-softmax kernel (the OOS exemplar)."""
    import jagged_softmax  # noqa: E402

    return jagged_softmax.jagged_softmax_kernel


def make_args(num_rows: int = 256, max_M: int = 128, avg: int = 300, device="cuda"):
    lens = torch.randint(1, 2 * avg, (num_rows,), device=device)
    offsets = torch.zeros(num_rows + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(lens, 0)
    total = int(offsets[-1].item())
    x_data = torch.randn(total, max_M, device=device, dtype=torch.float32)
    return (x_data, offsets)


def main() -> None:
    get_kernel()(*make_args())
    print("compiled + ran")


if __name__ == "__main__":
    main()
