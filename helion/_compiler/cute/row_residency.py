from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import ast


@dataclasses.dataclass(frozen=True)
class RowOwnership:
    """A thread's stable ordering of the row elements it visits."""

    slots_per_thread: int
    slot_expr: str

    def smem_slot_expr(self, linear_thread_expr: str) -> str:
        return f"({linear_thread_expr}) * {self.slots_per_thread} + ({self.slot_expr})"


@dataclasses.dataclass(frozen=True)
class SegmentedRowCacheBinding:
    """A resident value's first-load and later-replay lowering operations."""

    producer_stmt: ast.AST
    value_var: str
    publish_stmt: ast.AST | None = None
    replay_stmt: ast.AST | None = None
