"""Shared AST helpers for the CuTe source-to-source rewrite passes.

These small utilities are used by several of the post-codegen AST passes
(``hoist_loop_invariant_recip``, ``hoist_warp_reduce``,
``merge_sibling_v_loops``, ``pipeline_inner_loads``).  They previously
lived as byte-identical copies in each pass module.
"""

from __future__ import annotations

import ast

from ..ast_extension import ExtendedAST


class _NameRefCollector(ast.NodeVisitor):
    """Collect all ``ast.Name`` ids that appear as Load contexts in ``node``."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.names.add(node.id)


def _names_read(node: ast.AST) -> set[str]:
    collector = _NameRefCollector()
    collector.visit(node)
    return collector.names


def _assignment_lhs_name(stmt: ast.stmt) -> str | None:
    """If ``stmt`` is ``LHS = RHS`` with LHS a single ``ast.Name``, return LHS.id."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target = stmt.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def ext_deepcopy(node: object) -> object:
    """Deepcopy an AST subtree, preserving ``ExtendedAST`` mixin attributes.

    ⚠ ``copy.deepcopy`` does NOT work on these nodes: ``ExtendedAST.__init__`` requires a
    keyword-only ``_location`` and the default deepcopy reconstructor passes positional
    args, so it raises ``TypeError``.  Walk the tree and recreate each node through its own
    ``copy()`` helper instead, which carries ``_location`` / ``_type_info`` /
    ``_loop_type`` / ``_root_id`` across.

    Lifted verbatim from ``online_to_3pass._ext_copy``, which is now a thin alias -- two
    passes need it and a third copy would be the fourth byte-identical clone this module
    exists to prevent.
    """
    if isinstance(node, list):
        return [ext_deepcopy(x) for x in node]
    if not isinstance(node, ast.AST):
        return node
    if isinstance(node, ExtendedAST):
        return node.copy(
            **{field: ext_deepcopy(getattr(node, field)) for field in node._fields}
        )
    cls = type(node)
    new_node = cls(
        **{
            field: ext_deepcopy(getattr(node, field))
            for field in node._fields
            if hasattr(node, field)
        }
    )
    for attr in getattr(node, "_attributes", ()):
        if hasattr(node, attr):
            setattr(new_node, attr, getattr(node, attr))
    return new_node
