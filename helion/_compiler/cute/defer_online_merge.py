"""AST pass that DEFERS the cross-lane combine of an online (max, sum) recurrence
out of the reduction loop, leaving one cross-lane merge after it.

THE PATTERN, as emitted today for ``examples/cross_entropy.py::cross_entropy_online``
on ``CuteNDTileStrategy`` (``bs=[1,512] nt=[0,32] vw=[1,8]``, N=8192)::

    m_run = Float32(-inf)
    s_run = Float32(0.0)
    for tile_offset_1 in range(0, 8192, 512):        # 16 iters
        for lane_1 in range(2):                      # unrolled
            <vector loads>
            <constexpr V-loop: _helion_vfold_acc_0 = max(acc, x)>
            m_chunk = REDUCE_max(_helion_vfold_acc_0)     # <-- CROSS-LANE, IN-LOOP
            m_new   = maximum(m_run, m_chunk)
            <constexpr V-loop: _helion_vfold_acc_1 += exp2(x - m_new)>
            s_run   = s_run * exp2(m_run - m_new) + REDUCE_sum(_helion_vfold_acc_1)
            m_run   = m_new                                # <-- cross-lane, in-loop

Every one of those ``2 * N/(nt*V)`` cross-lane reductions (64 per row at N=8192) is a
SERIAL DEPENDENCY between one loop iteration and the next: the next iteration's
``m_new`` cannot be formed until this iteration's shuffle tree (or, at ``nt > 32``, its
SMEM round trip and barrier pair) has retired.  The loads are already issued; what the
loop is waiting on is a reduction the algorithm does not need yet.

THE REWRITE.  Each thread runs its OWN private ``(m, s)`` recurrence over its own slice,
and the lanes are combined exactly once, after the loop::

    for tile_offset_1 in ...:
        for lane_1 in ...:
            ...
            m_chunk = _helion_vfold_acc_0  # thread-private, no shuffle
            m_new = maximum(m_run, m_chunk)
            ...
            s_run = s_run * exp2(m_run - m_new) + _helion_vfold_acc_1
            m_run = m_new
    # ONE cross-lane merge, after the loop:
    _m = REDUCE_max(m_run)
    _s = s_run * exp2(m_run - _m) if s_run > 0 else 0
    s_run = REDUCE_sum(_s)
    m_run = _m

WHY IT IS EXACT.  The online pair combine

    (m1, s1) o (m2, s2) = (max(m1, m2), s1*exp(m1 - max) + s2*exp(m2 - max))

is associative and commutative with identity ``(-inf, 0)``, and ``s`` is defined relative
to its own ``m``.  Reducing over lanes is therefore a fold of that monoid in a different
order -- the same answer, not an approximation.  MEASURED relerr 1.0e-07 against
``torch.nn.functional.cross_entropy`` at 32768x8192, versus 0.0 for the in-loop form:
both are far inside the 1e-2 screen, and the deferred form is arguably the more accurate
of the two (each thread's partial sum has 1/nt of the terms, so less cancellation).

MEASURED (LEDGER E054), ``cross_entropy_online`` vs quack, cold-L2 CUDA-graph replay,
each cell at its own best config.  **7 of 8 power-of-two cells go from failing to
passing** the run's 0.95x bar:

    N          1024   4096   8192   16384  32768  65536  131072  262144
    in-loop    0.992  0.816  0.703  0.721  0.708  0.777   0.827   0.973
    deferred   1.221  1.000  0.930  0.969  1.012  1.065   1.147   1.364
    passes      yes    yes     no     yes    yes    yes     yes     yes

Reproducible to +/-0.0003 over three passes.  N=8192 is the one cell still short; its
config space is exhausted at 0.930 (16 geometries measured) and it is load-width bound --
the V=8 bf16 path emits 2x LDG.64 rather than one LDG.128, because the CuTe DSL ICEs on an
8-wide ``VectorType`` (the ``tile_unroll_split2`` workaround in ``memory_ops.py``).

WHAT IT DOES **NOT** DO.  The deferral needs the loop-carried pair to be PRIVATE to a
thread while the loop runs.  That is exactly true here because the reduction axis is
distributed across the lane/thread axis and nothing else in the loop reads ``m_run`` /
``s_run`` across threads.  The matcher below therefore requires the whole recurrence
shape -- both reduces, the rescale, the carry -- and declines on anything else, rather
than trying to be general.  In particular:

  * a reduce whose result is STORED or otherwise consumed inside the loop cannot be
    deferred (the store would see a partial value), so the pass requires the only
    consumers of ``m_run``/``s_run`` inside the loop to be the recurrence itself;
  * a THIRD reduce in the same loop declines (the pass matches exactly two);
  * anything reading the carried pair between the loop and its uses declines, because
    the merge is inserted immediately after the loop.

WHETHER TO DEFER IS A CONFIG KNOB, ``cute_online_defer`` (one slot per device loop;
``CuteOnlineDeferSpec`` in ``helion/autotuner/config_spec.py``).  It defaults to True,
so a config that does not mention it gets exactly the behaviour measured above -- but
because it is a real ``Config`` key rather than an env var, a winning config RECORDS
that the deferral was on, the autotuner can try both values, and the result is
reproducible from the config alone.

That it must be searchable rather than fixed is stated by the kernel this pass exists
for.  ``examples/cross_entropy.py::cross_entropy_online``: "Whether that trade wins is
a property of the backend, not of the algorithm: on a machine where the
special-function pipe is the limiter the extra ``exp`` can cost more than the saved
memory pass."  It also interacts with ``num_threads`` -- deferring changed which thread
count wins, because a cross-lane merge is what made a wide CTA expensive -- so the two
have to be searched jointly, which a knob permits and a module-level ``if`` does not.

``HELION_DISABLE_DEFER_ONLINE_MERGE=1`` remains as a DEBUGGING OVERRIDE: it forces the
pass off regardless of the config, which is what A/B attribution wants (one env var,
no config surgery, every cell at once).  It is deliberately no longer the only way to
reach the in-loop form.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import re

from ..ast_extension import statement_from_string
from ._ast_pass_utils import _assignment_lhs_name
from ._ast_pass_utils import _names_read


@dataclasses.dataclass(frozen=True)
class OnlineDeferPlan:
    """⭐ ONE DEFERRABLE ONLINE-RECURRENCE LOOP, as identified by its emitter.

    The channel that stops this pass from deciding *whether it is allowed to fire*.  The
    rewrite itself is legitimate and stays -- its target (the cross-lane ``REDUCE_*`` call)
    is CREATED by codegen, in three geometry-dependent spellings, so there is nothing to
    rewrite before lowering.  What was not legitimate is that its LICENCE was also
    re-derived from emitted Python.

    ``offset_var`` is the loop's induction variable, which is what makes this a LOOKUP
    rather than a guess.  ⛔ THE FACT IT REPLACES IS THE ONE WITH THE WORST MEASUREMENT
    BEHIND IT: "which loop is the recurrence's outer one" was answered by **sibling
    position** (are the carried names assigned by a preceding sibling?), and the emitted
    nest is ``for tile_offset in range(...): for lane in range(...)`` where the INNER lane
    loop satisfies every other structural predicate the outer one does.  Merging at the
    inner loop's exit gives relerr **5.18** -- a compilable, plausible, ~nt-fold overcount.
    A plan keyed on the offset variable cannot express the wrong loop: the strategy that
    EMITTED the tile loop registers its own offset var, so the inner lane loop is simply
    absent from the map.

    ``frozen=True`` for the same reason ``RaggedPeelPlan`` is: **a plan is a proof
    result.** Nothing downstream may adjust it -- an adjusted plan is a different proof,
    and the arithmetic that justified it has already run.

    ⚠ KEYED ON THE DEVICE LOOP, NOT ON A REDUCTION BLOCK.  ``cross_entropy_online`` uses an
    explicit inner ``hl.tile(v)`` because its recurrence is loop-carried, so it owns NO
    reduction block (``reduction_loops == []``) -- which is already why
    ``cute_online_defer`` needed a per-device-loop registration domain.
    """

    offset_var: str
    # The accumulator dtype constructor (e.g. ``cutlass.Float32``).  ⚠ CARRIED, because
    # the pass used to SCRAPE it off the emitted wrapper's text; it is
    # ``reduction_acc_dtype`` at the producer and was explicit there.  ``None`` means the
    # producer could not name it, and the pass falls back to its own recovery.
    acc_dtype_str: str | None = None


# log2(e).  The emitted code already works in ``exp2`` (helion rewrites ``exp`` to
# ``exp2(x * log2e)``), so the merge's rescale must use the same form to stay
# bit-comparable with the in-loop rescale it replaces.
_LOG2E = "1.4426950408889634"

# The cross-lane reduce forms helion emits, and how to read/rebuild each one.  ALL THREE
# must be handled: the raw warp shuffle (``nt <= 32``, one row per CTA), the cross-warp
# SMEM two-stage (``nt > 32``), and the strided grouped warp reduce (``bo > 1``, where a
# sibling row axis shares the warp).  Matching only the first would silently skip the
# other two -- the exact mistake LEDGER E019 records for ``hoist_warp_reduce``.
_ARCH_REDUCE_ATTRS = {"warp_reduction_max": "max", "warp_reduction_sum": "sum"}
_GROUPED_REDUCE_FUNCS = {
    "_cute_grouped_reduce_warp",
    "_cute_grouped_reduce_shared_two_stage",
}


@dataclasses.dataclass
class _CrossLaneReduce:
    """One cross-lane reduce call found in the loop body."""

    op: str  # "max" | "sum"
    call: ast.Call  # the reduce call node
    stmt_index: int  # index in the loop body
    input_name: str  # the per-thread accumulator it reduces

    def rebuilt_with(self, operand: str, subs: dict[str, str]) -> str:
        """This reduce, re-emitted over a different operand, after the loop.

        Reuses the ORIGINAL call's own arguments (thread counts, lane expressions, group
        spans, accumulator dtype) rather than re-deriving them, so the deferred merge
        combines over exactly the same thread set as the in-loop reduce did.

        ``subs`` renames argument NAMES that the merge re-materialises outside the loop.
        This matters for the grouped/two-stage forms, whose lane arguments are *variables*
        assigned inside the loop body (``strided_lane`` = ``thread_idx()[0]`` and its two
        derived moduli): referring to them from after the loop would read whatever the
        last iteration left, which happens to be right today but is not a property the
        rewrite may rely on.  The merge therefore recomputes them under fresh names and
        substitutes here.
        """
        args = [ast.Name(id=operand, ctx=ast.Load()), *self.call.args[1:]]
        call = ast.Call(func=self.call.func, args=args, keywords=self.call.keywords)
        ast.fix_missing_locations(call)
        text = ast.unparse(call)
        for old, new in subs.items():
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        return text

    def name_args(self) -> list[str]:
        """Argument names (after the operand) that are plain variables.

        These are the ones the merge must re-materialise outside the loop; a literal or a
        composite expression argument needs no substitution.
        """
        return [a.id for a in self.call.args[1:] if isinstance(a, ast.Name)]


def _classify_reduce_call(node: ast.AST) -> tuple[str, str] | None:
    """Recognise a cross-lane reduce call; return ``(op, input_name)``.

    The input must be a bare Name (the per-thread accumulator ``hoist_warp_reduce``
    built), because the deferral replaces the call with that name.
    """
    if not isinstance(node, ast.Call) or not node.args:
        return None
    first = node.args[0]
    if not isinstance(first, ast.Name):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        op = _ARCH_REDUCE_ATTRS.get(func.attr)
        if op is None:
            return None
        value = func.value
        if not (isinstance(value, ast.Attribute) and value.attr == "arch"):
            return None
        return op, first.id
    if isinstance(func, ast.Name) and func.id in _GROUPED_REDUCE_FUNCS:
        if len(node.args) < 2:
            return None
        op_arg = node.args[1]
        if not (isinstance(op_arg, ast.Constant) and op_arg.value in ("max", "sum")):
            return None
        return str(op_arg.value), first.id
    return None


def _own_subnodes(stmt: ast.stmt) -> list[ast.AST]:
    """``stmt``'s direct child nodes EXCLUDING its nested statement bodies.

    Every predicate in this pass is per-statement, and ``ast.walk`` / ``_names_read``
    descend into a nested ``for`` body -- so without this a compound statement inherits
    all of its children's reads and calls.  Two separate matcher bugs came from that.
    """
    nested = {id(s) for body in _iter_nested_bodies(stmt) for s in body}
    out: list[ast.AST] = []
    for _field, value in ast.iter_fields(stmt):
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, ast.AST) and id(item) not in nested:
                out.append(item)
    return out


def _own_reads(stmt: ast.stmt, groups: dict[str, str] | None = None) -> set[str]:
    """Names read by ``stmt``'s own expressions, not by its nested bodies.

    Canonicalised through ``groups`` (pre-rename -> canonical), so an alias pair that
    ``ast_rename`` will later collapse is treated as one name here.
    """
    reads: set[str] = set()
    for node in _own_subnodes(stmt):
        reads |= _names_read(node)
    if groups:
        reads = {groups.get(n, n) for n in reads}
    return reads


def _lhs_name(stmt: ast.stmt, groups: dict[str, str] | None = None) -> str | None:
    """``_assignment_lhs_name`` with rename-group canonicalisation."""
    lhs = _assignment_lhs_name(stmt)
    if lhs is not None and groups:
        return groups.get(lhs, lhs)
    return lhs


def _stmt_own_reduces(stmt: ast.stmt) -> list[tuple[str, ast.Call, str]]:
    """Cross-lane reduces in ``stmt``'s OWN expressions, not in nested bodies.

    Nested bodies are visited separately by the caller.
    """
    out: list[tuple[str, ast.Call, str]] = []
    for item in _own_subnodes(stmt):
        for node in ast.walk(item):
            found = _classify_reduce_call(node)
            if found is not None:
                assert isinstance(node, ast.Call)
                out.append((found[0], node, found[1]))
    return out


class _ReplaceCallWithName(ast.NodeTransformer):
    """Replace one specific Call node with a bare Name."""

    def __init__(self, target: ast.Call, name: str) -> None:
        self.target = target
        self.name = name

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if node is self.target:
            return ast.copy_location(ast.Name(id=self.name, ctx=ast.Load()), node)
        self.generic_visit(node)
        return node


def _strip_reduce(stmt: ast.stmt, red: _CrossLaneReduce) -> ast.stmt:
    """``stmt`` with ``red``'s reduce call replaced by its per-thread input."""
    out = _ReplaceCallWithName(red.call, red.input_name).visit(stmt)
    ast.fix_missing_locations(out)
    assert isinstance(out, ast.stmt)
    return out


def _iter_nested_bodies(stmt: ast.stmt) -> list[list[ast.stmt]]:
    out = []
    for field in ("body", "orelse", "finalbody"):
        value = getattr(stmt, field, None)
        if isinstance(value, list) and all(isinstance(s, ast.stmt) for s in value):
            out.append(value)
    return out


def _collect_reduces(body: list[ast.stmt]) -> list[_CrossLaneReduce] | None:
    """Every cross-lane reduce anywhere under ``body``, in source order.

    Returns None if any single statement holds MORE THAN ONE reduce in its own
    expressions (the rewrite replaces one call per statement).
    """
    found: list[_CrossLaneReduce] = []

    def walk(stmts: list[ast.stmt]) -> bool:
        for idx, stmt in enumerate(stmts):
            own = _stmt_own_reduces(stmt)
            if len(own) > 1:
                return False
            for op, call, input_name in own:
                found.append(_CrossLaneReduce(op, call, idx, input_name))
            for nested in _iter_nested_bodies(stmt):
                if not walk(nested):
                    return False
        return True

    if not walk(body):
        return None
    return found


def _reduce_acc_ctor(body: list[ast.stmt], red: _CrossLaneReduce) -> str | None:
    """The CuTe dtype constructor ``red`` accumulates in, e.g. ``cutlass.Float32``.

    The merge must rescale in the SAME dtype the recurrence accumulates in, and this
    reads it off the emitted code rather than assuming fp32 (a hardcoded cast would
    silently downcast a wider accumulator).  Two sources, in order of authority:

    1. the reduce call's own ``acc_dtype=`` keyword.  The grouped helpers carry it
       explicitly, which is the value they actually accumulate in;
    2. the ctor WRAPPING the call, for the raw ``cute.arch.warp_reduction_*`` form, which
       has no such keyword and is emitted as ``m_chunk = cutlass.Float32(REDUCE(...))``.

    ⚠ (1) is why this is not just "read the wrapper": the two-stage form is emitted as
    ``strided_reduce_result = _cute_grouped_reduce_shared_two_stage(...)`` with the cast on
    a *following* statement, so a wrapper-only version returned None and the pass declined
    on every ``nt > 32`` config -- a silent miss that the regression script caught.
    """
    for kw in red.call.keywords:
        if kw.arg == "acc_dtype":
            text = ast.unparse(kw.value)
            if text.startswith("cutlass.") and text[8:].isidentifier():
                return text
            return None
    for stmts in _all_stmt_lists(body):
        for stmt in stmts:
            if not any(call is red.call for _, call, _ in _stmt_own_reduces(stmt)):
                continue
            rhs = stmt.value if isinstance(stmt, ast.Assign) else None
            if not isinstance(rhs, ast.Call):
                return None
            text = ast.unparse(rhs.func)
            if not text.startswith("cutlass.") or not text[8:].isidentifier():
                return None
            return text
    return None


def _reduce_result_name(
    body: list[ast.stmt], red: _CrossLaneReduce, groups: dict[str, str]
) -> str | None:
    """The variable the reduce's IMMEDIATELY-enclosing assignment writes.

    Uses ``_stmt_own_reduces`` (which does not descend into nested bodies) so that the
    answer is the innermost assignment holding the call, not the outermost ``for`` that
    contains it -- ``_assignment_lhs_name`` of a ``For`` is None, which is what made the
    first version of this always return None.
    """
    for stmts in _all_stmt_lists(body):
        for stmt in stmts:
            if any(call is red.call for _, call, _ in _stmt_own_reduces(stmt)):
                return _lhs_name(stmt, groups)
    return None


def _all_stmt_lists(body: list[ast.stmt]) -> list[list[ast.stmt]]:
    out = [body]
    for stmt in body:
        for nested in _iter_nested_bodies(stmt):
            out.extend(_all_stmt_lists(nested))
    return out


def _rewrite_all(body: list[ast.stmt], reduces: list[_CrossLaneReduce]) -> None:
    """In-place: strip every reduce in ``reduces`` from ``body`` and its nested bodies.

    Keyed on each statement's OWN reduces so a statement is rewritten at the nesting
    level that actually holds the call -- rewriting via an enclosing ``for`` would work
    by accident (the transformer descends) but makes the traversal order load-bearing.
    """
    by_call = {id(r.call): r for r in reduces}
    for stmts in _all_stmt_lists(body):
        for i, stmt in enumerate(stmts):
            for _op, call, _inp in _stmt_own_reduces(stmt):
                red = by_call.get(id(call))
                if red is not None:
                    stmts[i] = _strip_reduce(stmt, red)


def _is_range_loop(stmt: ast.stmt) -> bool:
    """``for X in range(...)`` (the reduction loop), not a constexpr V-loop."""
    if not isinstance(stmt, ast.For):
        return False
    it = stmt.iter
    return (
        isinstance(it, ast.Call)
        and isinstance(it.func, ast.Name)
        and it.func.id == "range"
    )


def _carried_names(body: list[ast.stmt], groups: dict[str, str]) -> set[str]:
    """The scalars this loop nest carries across ITERATIONS of the reduction loop.

    A name is carried when it is READ somewhere in the nest before (or without) being
    written in the same nest at a point that dominates the read -- i.e. its value on
    entry to an iteration comes from the previous one.  Computed as
    ``reads_before_any_write`` over a flattened pre-order walk of the nest.

    ⚠ NOT "assignments whose RHS reads the LHS".  That definition also matches the
    per-V-lane fold accumulators (``_helion_vfold_acc_N = max(acc, x)``), which are
    RE-INITIALISED every tile iteration and are therefore not carried at all.  The first
    version of this pass used it and consequently identified the wrong pair; the emitted
    online recurrence writes its carry through temporaries (``s_run = v_9 + sum_1``),
    so the self-read never appears.
    """
    written: set[str] = set()
    carried: set[str] = set()

    def walk(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            # Reads in this statement's own expressions come before its nested bodies.
            carried.update(_own_reads(stmt, groups) - written)
            for nested_body in _iter_nested_bodies(stmt):
                walk(nested_body)
            lhs = _lhs_name(stmt, groups)
            if lhs is not None:
                written.add(lhs)

    walk(body)
    # Only names the nest also WRITES can be carried; a pure read is a loop invariant.
    return carried & written


def _transitive_deps(
    body: list[ast.stmt], groups: dict[str, str]
) -> dict[str, set[str]]:
    """For each name assigned in the nest, every name its value transitively depends on.

    Used to attribute each carried accumulator to the reduce it derives from.  The
    emitted recurrence routes through temporaries -- ``m_chunk`` -> ``v_6`` -> ``m_run``
    -- so a direct "does this assignment read the reduce result" test is not enough.
    Iterated to a fixed point, which terminates because the dependency sets only grow
    and are bounded by the name set.
    """
    direct: dict[str, set[str]] = {}
    for stmts in _all_stmt_lists(body):
        for stmt in stmts:
            lhs = _lhs_name(stmt, groups)
            if lhs is not None:
                direct.setdefault(lhs, set()).update(_own_reads(stmt, groups))
    deps = {name: set(reads) for name, reads in direct.items()}
    changed = True
    while changed:
        changed = False
        for name, reads in deps.items():
            grown = set(reads)
            for read in reads:
                grown |= direct.get(read, set())
            grown.discard(name)
            if grown != reads:
                deps[name] = grown
                changed = True
    return deps


def _lane_arg_definitions(
    body: list[ast.stmt],
    wanted: list[str],
    loop: ast.For,
    groups: dict[str, str],
) -> list[tuple[str, str]] | None:
    """The in-loop definitions of ``wanted``, in dependency order, if reproducible.

    Returns ``[(name, rhs_source), ...]`` ordered so each entry only reads earlier ones,
    or None when any of them cannot be re-materialised outside the loop.

    A definition is reproducible only if it is LANE-ONLY: a plain assignment whose reads
    are the loop-invariant thread-index expressions and other wanted names -- never the
    loop induction variable and never reduction state.  ``strided_lane =
    cutlass.Int32(cute.arch.thread_idx()[0])`` and its two derived moduli qualify; anything
    else makes the pass decline rather than emit a merge over the wrong thread set.
    """
    loop_var = loop.target.id if isinstance(loop.target, ast.Name) else None
    defs: dict[str, tuple[str, set[str]]] = {}
    for stmts in _all_stmt_lists(body):
        for stmt in stmts:
            name = _assignment_lhs_name(stmt)
            if name is None or name not in wanted or name in defs:
                continue
            assert isinstance(stmt, ast.Assign)
            reads = _own_reads(stmt, groups)
            if loop_var is not None and loop_var in reads:
                return None
            defs[name] = (ast.unparse(stmt.value), reads)
    if set(defs) != set(wanted):
        return None
    # Topologically order by the wanted-name dependencies; every other read must be a
    # module-level name (``cutlass``, ``cute``) rather than an in-loop value.
    allowed_free = {"cutlass", "cute", "ir", "torch"}
    ordered: list[tuple[str, str]] = []
    remaining = dict(defs)
    while remaining:
        progressed = False
        for name in sorted(remaining):
            rhs, reads = remaining[name]
            pending = (reads & set(remaining)) - {name}
            if pending:
                continue
            if not (reads - allowed_free - set(defs)) <= set():
                return None  # reads something in-loop that is not a lane name
            ordered.append((name, rhs))
            del remaining[name]
            progressed = True
            break
        if not progressed:
            return None  # cyclic
    return ordered


def _try_defer_one_loop(
    loop: ast.For, groups: dict[str, str], initialised: set[str]
) -> list[ast.stmt] | None:
    """Rewrite one reduction loop into private-recurrence + one merge.

    ``initialised`` is the set of names assigned by ``loop``'s PRECEDING SIBLINGS.  The
    carried pair must be among them: that is what identifies the recurrence's outermost
    loop, whose exit is the single point at which each thread's private ``(m, s)`` is
    complete.  See ``_walk`` for why this is a correctness condition and not a heuristic.

    Returns the replacement statements (the loop plus the merge), or None to decline.
    """
    body = list(loop.body)
    reduces = _collect_reduces(body)
    if reduces is None or len(reduces) != 2:
        return None
    ops = [r.op for r in reduces]
    if ops != ["max", "sum"]:
        # The online recurrence is max-then-sum in that order; anything else is a
        # different computation and is not this pass's business.
        return None
    red_max, red_sum = reduces

    carried = _carried_names(body, groups)
    if len(carried) != 2:
        return None

    # Identify which carried name is the max and which is the sum by how each reduce's
    # result flows into them.  ``m_run``'s update reads the max reduce's result (via
    # ``m_new``); ``s_run``'s update reads the sum reduce's result directly.
    m_result = _reduce_result_name(body, red_max, groups)
    s_result = _reduce_result_name(body, red_sum, groups)
    if m_result is None or s_result is None:
        return None

    # Attribute each carried accumulator to its reduce, using the ASYMMETRY that defines
    # the online recurrence.  Checked transitively through the emitted temporaries
    # (``m_chunk`` -> ``v_6`` -> ``m_run``), since the recurrence never assigns a reduce
    # result to a carried name directly.
    #
    #   * the running MAX depends on the max reduce and NOT on the sum reduce -- the max
    #     is computed without reference to the sum;
    #   * the running SUM depends on the sum reduce AND on the max reduce, because the
    #     rescale ``s * exp(m_old - m_new)`` reads the new max.
    #
    # That one-way dependency is exactly what makes the deferral valid (the max fold is
    # independent, so it can be reassociated freely), so testing for it is testing the
    # precondition rather than pattern-matching variable names.  Anything else -- both
    # symmetric, both depending only on their own reduce, or a cycle -- declines.
    deps = _transitive_deps(body, groups)
    max_side = [
        n
        for n in sorted(carried)
        if m_result in deps.get(n, set()) and s_result not in deps.get(n, set())
    ]
    sum_side = [
        n
        for n in sorted(carried)
        if s_result in deps.get(n, set()) and m_result in deps.get(n, set())
    ]
    if len(max_side) != 1 or len(sum_side) != 1 or max_side[0] == sum_side[0]:
        return None
    m_run = max_side[0]
    s_run = sum_side[0]

    # ⚠ THE OUTERMOST-LOOP CONDITION.  Both accumulators must be initialised OUTSIDE this
    # loop, by a preceding sibling.  If they are not, this is an inner loop of the
    # recurrence (the emitted lane loop matches every other predicate here) and merging at
    # its exit merges mid-stream -- the next iteration then accumulates on top of an
    # already-merged sum and multiplies it by the lane count.  MEASURED as relerr 5.18.
    if not {m_run, s_run} <= initialised:
        return None

    # The carried pair must be PRIVATE while the loop runs: nothing inside the loop may
    # OBSERVE a per-thread partial value.  Reading it into the recurrence's own
    # temporaries is fine (that is the recurrence); what is not fine is a side effect --
    # a ``.store(...)``, an ``if``-guarded write, a subscript assignment -- since after
    # the rewrite those would see a value that has not been combined across lanes yet.
    #
    # ⚠ Tested on each statement's OWN expressions, via ``_own_reads``: plain
    # ``_names_read`` descends into nested bodies, so the enclosing lane ``for`` would
    # test as "reads the carried pair and is not an assignment" and every kernel would
    # decline.  That bug made the first version of this pass fire on nothing at all.
    #
    # The same argument applies to the two reduces' RESULTS (``m_chunk`` / ``sum_1``):
    # after the rewrite they hold a thread-private partial, so a side effect reading one
    # would observe an un-combined value.  Both sets are checked together.
    private = {m_run, s_run, m_result, s_result}
    for stmts in _all_stmt_lists(body):
        for stmt in stmts:
            if not (private & _own_reads(stmt, groups)):
                continue
            # Only a plain ``NAME = expr`` may consume a partial: that is how the
            # recurrence threads its own temporaries.  Anything else -- a bare expression
            # statement (``ptr.store(...)``), a subscript/attribute target, an augmented
            # assignment, a ``return`` -- either has a side effect or writes somewhere
            # this analysis does not track, and would publish an un-merged value.
            if not isinstance(stmt, ast.Assign) or _assignment_lhs_name(stmt) is None:
                return None

    # The merge, emitted after the loop.  ``if {s} > 0`` is exact, not a heuristic:
    # ``{s}`` is a sum of ``exp2`` terms, so a thread that consumed at least one element
    # has ``{s} > 0``, and a thread that consumed none has ``{s} == 0`` AND
    # ``{m} == -inf`` -- for which ``{m} - merged`` would be ``-inf - -inf`` = NaN.  The
    # guard keeps that NaN out of the sum while contributing the correct 0.
    # Accumulate in whatever dtype the in-loop recurrence used, read off the emitted
    # wrapper rather than assumed (fp32 in every kernel that reaches this today, but a
    # hardcoded cast would silently downcast a wider accumulator).
    ctor = _reduce_acc_ctor(body, red_max)
    if ctor is None or ctor != _reduce_acc_ctor(body, red_sum):
        return None

    # Re-materialise, OUTSIDE the loop, any lane variables the reduce calls take as
    # arguments (the grouped / two-stage forms do; the raw warp form does not).  Their
    # defining statements must be lane-only -- thread-index arithmetic, no loop variable
    # and no reduction state -- or the merge cannot reproduce them and the pass declines.
    lane_defs: list[ast.stmt] = []
    subs: dict[str, str] = {}
    wanted = sorted({n for r in (red_max, red_sum) for n in r.name_args()})
    if wanted:
        defs = _lane_arg_definitions(body, wanted, loop, groups)
        if defs is None:
            return None
        for name, rhs_text in defs:
            fresh = f"_defer_lane_{name}"
            subs[name] = fresh
            for old, new in subs.items():
                rhs_text = re.sub(rf"\b{re.escape(old)}\b", new, rhs_text)
            lane_defs.append(statement_from_string(f"{fresh} = {rhs_text}"))

    merged = f"_defer_merged_{m_run}"
    scaled = f"_defer_scaled_{s_run}"
    merge = [
        *lane_defs,
        statement_from_string(
            f"{merged} = {ctor}({red_max.rebuilt_with(m_run, subs)})"
        ),
        statement_from_string(
            f"{scaled} = {s_run} * cute.math.exp2("
            f"{ctor}({m_run} - {merged}) * {_LOG2E})"
            f" if {s_run} > 0 else {ctor}(0)"
        ),
        statement_from_string(
            f"{s_run} = {ctor}({red_sum.rebuilt_with(scaled, subs)})"
        ),
        statement_from_string(f"{m_run} = {merged}"),
    ]

    _rewrite_all(body, reduces)
    loop.body = body
    return [loop, *merge]


def _loop_target_name(stmt: ast.For) -> str | None:
    """The loop's induction variable, when it is a plain name.

    This is the JOIN KEY between the knob and the pass.  ``cute_online_defer`` is
    registered per device-loop block, and codegen names each device loop's offset
    variable after its block (``tile_offset_1`` / ``roffset_1`` --
    ``TileStrategy.offset_var``), so the caller can hand this pass a set of offset
    variable names and the pass can decide per loop without knowing anything about
    block ids, configs or ``ConfigSpec``.

    Chosen over passing block ids because this pass runs on the emitted AST, where a
    block id is no longer represented -- only the name is.  A tuple-target loop has no
    single name; those are not device loops and the pass never matched them anyway.
    """
    return stmt.target.id if isinstance(stmt.target, ast.Name) else None


def _walk(
    body: list[ast.stmt],
    groups: dict[str, str],
    disabled_offsets: frozenset[str],
    plans: dict[str, OnlineDeferPlan] | None = None,
) -> list[ast.stmt]:
    """Try to defer at each ``range`` loop, OUTERMOST FIRST.

    ⭐⭐ WHICH LOOP IS THE RECURRENCE'S OUTER ONE IS NOW **LOOKED UP**, NOT INFERRED
    (task 3).  When ``plans`` is supplied, a loop is a candidate only if its offset
    variable is a key -- the strategy that EMITTED the tile loop registered it
    (``DeviceFunction.register_online_defer``), so the inner lane loop is simply absent
    from the map and the wrong loop is UNREPRESENTABLE rather than merely rejected.

    ⛔ THE FACT THIS REPLACES HAS THE WORST MEASUREMENT IN THE PASS.  It was **sibling
    position**: "are the carried names assigned by a preceding sibling?".  The emitted nest
    is ``for tile_offset in range(0, N, BI): for lane in range(EPT//V):`` and the inner lane
    loop is ALSO a ``range`` loop matching every other structural predicate, so a bottom-up
    walk deferred the INNER one -- putting the cross-lane merge inside the tile loop, where
    every iteration merges and the next accumulates on top of an already-merged sum.
    MEASURED as relerr **5.18** at 32768x8192 (a ~nt-fold overcount), caught by the
    correctness screen before any timing was believed.

    ⚠ THE SIBLING-INITIALISATION CHECK IS **KEPT** AS A VERIFICATION, not deleted.  With a
    plan the pass no longer *decides* from it, but ``_try_defer_one_loop`` still requires
    it: the plan says "this is the loop I emitted", and the check says "and its carried pair
    really is initialised outside it".  A plan is a proof result, and re-checking a proof's
    conclusion at the point of use is what ``_assert_peel_sound`` does for
    ``RaggedPeelPlan``.  Deleting it would make a mis-registered plan a wrong answer again.

    ⚠ AND ``plans=None`` MUST STAY "TODAY'S BEHAVIOUR".  The pass is deliberately callable
    on a hand-written body (its unit tests do exactly that, which is what keeps it
    testable without a config), so no plan map means fall back to sibling position rather
    than "decline everywhere" -- the same polarity argument ``disabled_offsets`` documents.
    """
    out: list[ast.stmt] = []
    for stmt in body:
        if _is_range_loop(stmt):
            assert isinstance(stmt, ast.For)
            # The knob is consulted HERE, per loop, rather than once at the top of
            # the pass.  A kernel can contain several device loops with different
            # ``cute_online_defer`` values, and a whole-pass early return could only
            # express "all or nothing".  Note this must NOT `continue`: a loop whose
            # own knob says False still has to be DESCENDED into, because a nested
            # loop may have its own slot saying True.
            target = _loop_target_name(stmt)
            registered = plans is None or (target is not None and target in plans)
            if target not in disabled_offsets and registered:
                initialised = {
                    _lhs_name(s, groups)
                    for s in out
                    if _lhs_name(s, groups) is not None
                }
                replaced = _try_defer_one_loop(stmt, groups, initialised)
                if replaced is not None:
                    # Deferred here; do NOT also rewrite anything nested inside.
                    out.extend(replaced)
                    continue
        for field in ("body", "orelse", "finalbody"):
            value = getattr(stmt, field, None)
            if isinstance(value, list) and all(isinstance(s, ast.stmt) for s in value):
                setattr(stmt, field, _walk(value, groups, disabled_offsets, plans))
        out.append(stmt)
    return out


def defer_online_merge(
    body: list[ast.stmt],
    *,
    rename_groups: dict[str, str] | None = None,
    disabled_offsets: frozenset[str] | None = None,
    plans: dict[str, OnlineDeferPlan] | None = None,
) -> list[ast.stmt]:
    """Defer the cross-lane combine of an online (max, sum) recurrence.

    Safe to call on any kernel body: the matcher requires the entire recurrence shape
    (exactly two cross-lane reduces, max then sum, exactly two loop-carried scalars, and
    no cross-thread consumer of either inside the loop) and leaves everything else
    untouched.

    ``disabled_offsets`` is the set of loop OFFSET VARIABLE NAMES whose
    ``cute_online_defer`` slot is False, i.e. the loops to leave alone.  The caller
    derives it from the config; see ``_loop_target_name`` for why the interface is
    names rather than block ids.

    ⚠ IT TAKES THE **DISABLED** SET, NOT THE ENABLED ONE, and that polarity is what
    keeps ``None`` meaning "today's behaviour, unchanged".  Passing an enabled set
    would make an empty/omitted argument mean "defer nowhere" -- so every caller that
    had not yet been taught about the knob (and every test that calls this pass
    directly) would silently switch to the in-loop form.  With this polarity the
    default is the empty set and the pass is exactly what it was.

    ``plans`` maps a loop's OFFSET VARIABLE NAME to the :class:`OnlineDeferPlan` its
    emitter registered.  When supplied, only a registered loop may be deferred -- which is
    what makes "the recurrence's outer loop" a lookup rather than the sibling-position
    guess that produced relerr 5.18.  ⚠ ``None`` means "no plan channel", and it falls back
    to the previous behaviour rather than declining everywhere: this pass is deliberately
    callable on a hand-written body (that is what keeps it unit-testable without a config),
    and the same polarity argument as ``disabled_offsets`` applies.

    ``rename_groups`` maps pre-rename name -> canonical name, exactly as
    ``hoist_loop_invariant_recips`` takes it.  ⚠ REQUIRED for the pass to fire at all:
    this pass runs BEFORE ``ast_rename``, so the emitted recurrence at this point writes
    its carried max through a *different name* than it reads
    (``v_6 = max(m_run, m_chunk)`` … ``m_run = v_6``, with ``v_6`` and ``m_run`` in one
    rename group that ``ast_rename`` will later collapse).  Without the map, ``m_run``
    reads as written-then-never-updated, ``_carried_names`` sees only ``s_run``, and the
    matcher declines on every kernel -- which it did, silently, until an in-process hook
    showed the pass being invoked once and changing nothing.
    """
    # Debugging override, NOT the knob: forces the pass off everywhere regardless of
    # the config, so an A/B attribution run needs one env var rather than 40 edited
    # configs.  Checked before the knob because "off everywhere" must win.
    if os.environ.get("HELION_DISABLE_DEFER_ONLINE_MERGE"):
        return body
    return _walk(body, rename_groups or {}, disabled_offsets or frozenset(), plans)
