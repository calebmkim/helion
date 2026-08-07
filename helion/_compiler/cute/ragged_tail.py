"""Ragged-tail predication for the CuTe TV reduction path.

THE PROBLEM THIS FILE OWNS.  ``reduction_loops`` is forced to a power of two, so
at a reduction extent like ``N = 12000`` or ``N = 100000`` (both ``2^5 · odd``)
**no** legal chunk divides ``N``.  Before this module, every place that needed
``numel % chunk == 0`` answered "no" and the whole TV plan was declined, taking
four optimisations down with it: the vector width, the ``_tv_sweep_cache`` rmem
reload, ``cute_reduction_reload="smem"``, and the cluster.  The kernel fell back
to scalar ``.load()`` and lost 38–200%.

THE FIX, WHICH IS QUACK'S.  quack's geometry **never sees N**
(``reduction_base.py:47``)::

    num_blocks_N = cute.ceil_div(self.N // vecsize, threads_per_row * cluster_n)

``ceil_div``, so the *tile* is ``>= N``; the tail is then **predicated**
(``rmsnorm.py:225-230`` ``predicate_k``) and out-of-bounds lanes are filled with
the reduction op's **identity** (``utils.fill_oob``).  quack never pads memory --
round-up is purely a tile concept.

So: **round the TILE up, predicate the TAIL, and keep every lane group a power of
two.**  That last clause matters: ``LEDGER`` E031 measured that admitting a
non-pow2 ``threads_per_row`` returns relerr 0.15-2.6, because the cross-lane
combine is a butterfly (xor-shuffle) and a butterfly only totals correctly over a
power-of-two group.  Nothing here changes any lane group; ``threads_per_row``,
``vec`` and ``cluster_n`` stay exactly the powers of two they were.


⭐ THE INVARIANT, STATED FOR EVERY CALLER
=========================================
Let ``chunk``, ``vec``, ``tpr`` and ``cluster_n`` be the TV plan's geometry and
``N`` the true reduction extent.  Write ``G = chunk * cluster_n`` (the *tile
granularity*, :func:`tile_granularity`) and ``N' = ceil(N / G) * G``
(:func:`rounded_extent`).  Then:

  **(I1) COVERAGE.**  The emitted loop walks ``[0, N')`` in steps of ``chunk``, so
  every column in ``[0, N)`` is visited exactly once by exactly one
  ``(chunk, lane, thread, vec-slot)`` tuple -- the same bijection the divisible
  case has, extended over a strictly larger interval.

  **(I2) FRAGMENT ALIGNMENT.**  A thread's column base is
  ``base = chunk_idx*chunk + lane*(tpr*vec) + tid*vec``, so ``vec | base``.
  :func:`assert_vec_divides_extent` additionally requires ``vec | N``.  Together,
  for every ``0 <= vi < vec``::

      base <  N  =>  base <= N - vec  =>  base + vi <= N - 1   (all vec IN bounds)
      base >= N                       =>  base + vi >= N       (all vec OOB)

  i.e. **a fragment is wholly in bounds or wholly out of bounds; it can never
  straddle the row end.**  This is why one scalar compare per lane suffices where
  quack needs a Boolean ``predicate_k`` tensor: ``cute.copy``'s ``pred``
  granularity is one whole vector block anyway, so quack's
  ``vecsize = gcd(N, 128 // width)`` is not an optimisation but *the
  precondition*.  :func:`assert_vec_divides_extent` is that precondition made
  explicit, and :func:`legal_tail_vec` is what establishes it.

  **(I3) NO OUT-OF-BOUNDS MEMORY EFFECT.**  Every ``cute.copy`` on a chunk whose
  tile can exceed ``N`` is wrapped in ``if base < N:``.  By (I2) the guard is
  exact, so:
    * a **load** never reads an address outside the tensor (memory safety), and
    * a **store** never writes an address outside its row (correctness -- an
      unguarded flush would write into row ``m+1``).
  The load and the store take the *same* predicate because they are
  ``partition_S``/``partition_D`` of the *same* slice, so they address the same
  elements by construction.

  **(I4) IDENTITY AT THE COMBINE.**  Nothing here supplies a per-op identity,
  because helion already does, at the right place: ``node_masking`` inserts a
  ``_mask_to(x, <identity>)`` between a masked value and its reduction, and the
  cute ``_mask_to`` codegen emits ``x if mask else <identity>`` with the identity
  chosen **per op** -- ``0`` for a sum, ``-inf`` for a max, both for the online
  ``(max, sum-of-exp)`` monoid.  Because that gate sits at the *combine* and not
  at the *load*, ``layer_norm``'s second moment is handled for free: its emitted
  form is ``sum_2_acc += (Float32(centered*centered) if mask else 0)``, which is
  exactly quack's ``fill_oob(x_centered, tXpX, 0)`` **after** centering
  (``rmsnorm.py:289-295``).  A fill placed at the load would be wrong there --
  an OOB lane zero-filled before centering contributes ``mean^2`` to the variance.
  So the ``:1692`` comment's premise, that "the per-element mask would have to
  gate a whole fragment", was **false**: on the TV path the per-element index
  ``rindex = base + vi`` *is* the element that fragment slot ``vi`` holds, so the
  existing per-element mask is already per-element and already correct.

  🔴 **I4 HAS A PRECONDITION, AND IT IS NOT AUTOMATIC.**  ``_mask_to`` only fires
  when the strategy actually created a reduction-axis ``mask_var``, and that
  decision is made from ``numel % <granularity>``.  Historically the granularity
  was ``chunk``; with a rounded tile it must be ``chunk * cluster_n``, because a
  ``numel`` that divides the *chunk* but not the *tile* still gets rounded up and
  therefore still has phantom columns.  MEASURED: at ``N = 12288``,
  ``chunk = 4096``, ``cluster_n = 2`` the extent is an exact multiple of the chunk,
  so no mask existed, yet the tile granularity is 8192 and the loop rounded to
  16384 -- 4096 phantom columns reaching the accumulator ungated, 18 of 1440
  configs wrong (relerr 0.14 rms_norm / 6.4 layer_norm).  The copies were
  correctly guarded throughout; what leaked was stale fragment content reaching
  the *combine*.  Caught by the ``n_wrong`` A/B, not by any suite.  The guard now
  lives in ``LoopedReductionStrategy.__init__`` and is asserted in
  ``cute_tv_rounded_extent``.

  **(I5) DEFERRED FOLDS STAY SOUND.**  ``defer_online_merge`` folds lanes *after*
  the loop, so an OOB lane's accumulator survives to the end.  By (I4) an OOB
  lane's contribution is the op identity, and by (I2) an OOB lane is OOB for
  *every* element of *every* fragment it owns -- so its accumulator is the
  identity element and folding it in is a no-op.  The same argument covers
  ``hoist_warp_reduce``, ``merge_sibling_v_loops`` and the sweep-2 reload path:
  each reorders or relocates a fold over values that are already identity-filled.

  **(I6) THE SWEEP CACHE IS INDEX-FOR-INDEX.**  ``fuse_tv_copy_sweeps`` caches a
  producer sweep's fragment and rewrites consumer sweeps to read it.  The tile
  round-up is *identical* in every sweep (same ``chunk``, same trip count, same
  ``N'``), so an OOB slot is OOB in producer and consumer alike; the cache holds
  whatever the guarded copy left in the fragment for those slots, and (I4)
  discards it at the combine.  The pass must, however, still *see* the copy --
  see :func:`tail_guard_stmt`, which is why the guard wraps the copy in place
  rather than restructuring the lane body.

WHAT IS **NOT** COVERED, AND IS THEREFORE DECLINED
--------------------------------------------------
Guarding by *declining* rather than degrading (``01_HOWTO.md`` step 4, "fail
closed"):

* ``vec`` must divide ``N``.  :func:`legal_tail_vec` narrows ``vec`` until it
  does; if it cannot reach a legal width the plan is declined, never emitted at a
  width that could straddle.
* the round-up must not exceed :data:`MAX_TILE_OVERSHOOT` of ``N``.  The tail's
  work is real (a wholly-OOB lane still issues its loop iterations), and MEASURED
  a round-up past ~1/3 of ``N`` is a losing trade.
* a **dynamic** extent is declined: ``N'`` is a compile-time constant here, so a
  symbolic ``numel`` has no round-up to compute.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sympy

    from ..compile_environment import CompileEnvironment

# The tail is *executed*: a lane whose whole fragment is out of bounds still runs
# its loop iteration, evaluates its (identity-gated) arithmetic and issues no
# copy.  So the round-up buys tile coverage at the price of that work, and past
# some point the price wins.  MEASURED on the ceiling probe
# (``_redfix2/repro/r3_b3_ragged_probe.py``) at rms_norm 8192x100000: +0.35%,
# +2.4%, +6.5% and +14.7% overshoot all beat the declined (scalar) path by
# 0.71-1.01x vs 0.42x, and the ranking is dominated by GEOMETRY not by overshoot
# (+6.5% at tpr512/cluster2 scored 1.013 while +2.4% at tpr256/cluster2 scored
# 0.914).  So this bound is deliberately generous -- it exists to stop a
# pathological case (a tiny ``N`` with a huge chunk, where ``N'`` could be many
# multiples of ``N``), not to express a tuning preference.  1/3 is the point at
# which the tail costs more iterations than the body can amortise.
MAX_TILE_OVERSHOOT = 1 / 3


def tile_granularity(chunk: int, cluster_n: int) -> int:
    """The extent the rounded-up tile must be a multiple of.

    ``chunk * cluster_n``, and the ``cluster_n`` factor is **not** optional: the
    cluster splits the row across CTAs and each CTA's share must itself be a whole
    number of chunks, which is a *second* divisibility requirement
    (``LoopedReductionStrategy._cute_cluster_n_config``).  Rounding only to
    ``chunk`` leaves the cluster declining for its own reasons -- MEASURED, that
    is worth 0.92 vs 1.01 at rms_norm 8192x100000.

    This is quack ``reduction_base.py:47`` with ``vec`` already folded into
    ``chunk``: ``ceil_div(N // vecsize, threads_per_row * cluster_n)`` scaled by
    ``vecsize * threads_per_row``.
    """
    return max(1, chunk) * max(1, cluster_n)


def rounded_extent(numel: int, chunk: int, cluster_n: int = 1) -> int:
    """``N'``: ``numel`` rounded UP to a whole number of tiles (quack's ``ceil_div``)."""
    gran = tile_granularity(chunk, cluster_n)
    return -(-max(numel, 0) // gran) * gran


def overshoot_fraction(numel: int, chunk: int, cluster_n: int = 1) -> float:
    """``(N' - N) / N``: the fraction of the tile that is pure tail."""
    if numel <= 0:
        return 0.0
    return (rounded_extent(numel, chunk, cluster_n) - numel) / numel


def legal_tail_vec(vec: int, numel: int) -> int:
    """The widest ``vec <= vec`` that may be predicated per-atom at extent ``numel``.

    ``cute.copy``'s ``pred`` granularity is one whole vector block, so a vector
    that straddles the row end cannot be partially masked -- it would either read
    past the row (loading another row's data into the reduction) or drop live
    elements.  quack's ``vecsize = math.gcd(N, 128 // width)``
    (``rmsnorm.py:120``, ``cross_entropy.py:91``) is exactly this constraint, and
    it is a *precondition* rather than a tuning choice.

    ⚠ ``ChunkTVPlan`` computes its width from ``chunk`` alone
    (``tv_layout.chunk_plan`` calls ``legal_vec(chunk, ...)``), which is correct
    for the divisible case -- there ``chunk | N`` so ``vec | chunk | N`` holds
    transitively.  Once the tile may exceed ``N`` that implication is gone and
    the constraint has to be imposed directly.  At ``N = 12000`` and
    ``N = 100000`` it happens to be satisfied at ``vec == 8``
    (``gcd(12000, 8) == gcd(100000, 8) == 8``), which is precisely what makes
    omitting it a *latent* fault rather than an immediate one: it would first
    misfire at an odd ``N``.

    Halving rather than taking the gcd outright keeps the result a power of two,
    which every downstream lane/index computation assumes (E031).
    """
    vec = max(1, vec)
    while vec > 1 and numel % vec:
        vec //= 2
    return vec


def assert_vec_divides_extent(vec: int, numel: int) -> None:
    """Invariant (I2), as an assertion at the point of emission.

    Not a decline: by the time codegen runs, :func:`legal_tail_vec` has already
    narrowed the width, so a violation here means the width was chosen somewhere
    that did not go through it -- a compiler bug, not a config we should silently
    degrade.
    """
    assert vec >= 1 and numel >= 0, (vec, numel)
    assert vec == 1 or numel % vec == 0, (
        f"ragged-tail predication needs vec | N so that no vector block straddles "
        f"the row end (cute.copy's pred granularity is one whole block), but "
        f"vec={vec} does not divide numel={numel}.  See ragged_tail.legal_tail_vec."
    )


def tail_is_ragged(numel: int, chunk: int, cluster_n: int = 1) -> bool:
    """True when the last tile is partial and therefore needs predication."""
    return rounded_extent(numel, chunk, cluster_n) != numel


def ragged_tile_admissible(
    env: CompileEnvironment,
    numel: sympy.Expr,
    chunk: int,
    *,
    cluster_n: int = 1,
    vec: int = 1,
) -> bool:
    """Whether a ragged extent may be covered by a rounded-up, predicated tile.

    This is the single predicate that replaces ``env.known_multiple(numel, chunk)``
    at every gate on the TV path, so the plan, the staging sizing and the cluster
    cannot drift apart in their answer.  It returns True for the *divisible* case
    too (``N' == N``, no tail), which is what keeps the divisible path unchanged.

    Declines, all of them "fail closed":

    * a non-static extent -- ``N'`` is a compile-time constant, so there is
      nothing to round;
    * a non-positive chunk or extent;
    * ``vec`` not dividing ``N`` -- see :func:`legal_tail_vec`.  A caller that
      wants the wider width must narrow ``vec`` first and ask again;
    * overshoot above :data:`MAX_TILE_OVERSHOOT`.
    """
    if chunk <= 0:
        return False
    if env.known_multiple(numel, tile_granularity(chunk, cluster_n)):
        return True
    hint = _static_extent(env, numel)
    if hint is None or hint <= 0:
        return False
    if vec > 1 and hint % vec:
        return False
    return overshoot_fraction(hint, chunk, cluster_n) <= MAX_TILE_OVERSHOOT


def _static_extent(env: CompileEnvironment, numel: sympy.Expr) -> int | None:
    """``numel`` as a compile-time int, or None when it is genuinely dynamic.

    ``shape_env_size_hint`` returns a *hint* for a dynamic shape, which would let
    a symbolic extent through with a guessed round-up -- so the static case is
    tested first and a hint alone is not accepted.
    """
    import sympy as _sympy

    if isinstance(numel, (int, _sympy.Integer)):
        return int(numel)
    return None


def gcd_all(*values: int) -> int:
    """``math.gcd`` folded over ``values`` (readability at the call sites)."""
    return math.gcd(*values) if values else 1
