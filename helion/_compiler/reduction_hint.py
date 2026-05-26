from __future__ import annotations

from enum import Enum


class ReductionHint(Enum):
    """Classifies a compiler-managed reduction by load-stride coalescing.

    Mirrors ``torch._inductor.runtime.hints.ReductionHint`` but is independent
    so we can evolve semantics if needed.
    """

    INNER = 0
    OUTER = 1
    DEFAULT = 2
