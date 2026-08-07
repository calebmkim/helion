"""The single owner of the CuTe reduction TV layout.

This module is the in-tree instantiation of quack's reduction copy idiom.  It
exists so that the *access width* and the *addresses* of a row reduction are
derived from one number in one place, instead of being decided independently by
the reduction strategy (which builds an integer index) and the memory-op
emitter (which picks a vector width).

The idiom being instantiated, verbatim from quack:

* ``quack/copy_utils.py:150-166`` ``tiled_copy_2d`` -- one
  ``make_tiled_copy_tv(atom, thr_layout, val_layout)`` where
  ``thr_layout = make_ordered_layout((num_threads // tpr, tpr), order=(1, 0))``
  and ``val_layout = make_layout((1, vec))``.
* ``quack/reduction_base.py:42-50`` ``_get_tiled_copy`` -- the tiler arithmetic.
* ``quack/reduction_base.py:28-40`` ``_cap_cluster_n`` -- the cluster cap that
  stops peer CTAs from re-reducing the same columns.
* ``quack/reduction_base.py:52-62`` ``_get_reduction_buffer_layout`` -- the SMEM
  combine buffer shape, derived *from* the TV layout.
* ``quack/rmsnorm.py:184-217`` -- partition every tensor through **one**
  ``get_slice(tidx)``, including the output via ``partition_D``.

Three sites already emit this idiom as ad-hoc source strings
(``program_id.py:4344-4356``, ``language/memory_ops.py:2075-2101``,
``cute/cute_flash.py:6725-6738``).  They are deliberately *not* refactored onto
this helper: they are on the matmul/flash critical paths and a fourth textual
copy is what this module exists to prevent going forward.

Nothing here emits code by itself; the reduction path is the intended first
user.  Every method is pure so the ladder can be unit-checked on the host.

The INVARIANT this module exists to enforce (see ``PORT_SPEC_layout.md`` §7):

    If any condition would prevent a width-``vec`` copy, the *only* legal
    response is to rebuild the plan with a smaller ``vec`` -- which re-derives
    the tiler, the partitions and the atom together -- and re-emit.  It is
    never legal to emit a copy whose atom width differs from ``val_layout``'s
    ``vec``, and never legal to keep a plan's addresses while changing its
    width.

``ChunkTVPlan`` is a frozen dataclass and every width/address string is a
method on it, so "decline" has no way to express itself except as
``plan.with_vec(smaller)``.
"""

from __future__ import annotations

import dataclasses
import math

# ``ragged_tail`` imports nothing from this module (it is pure int arithmetic over
# ``(numel, chunk, cluster_n)``), so this is a one-way edge and not a cycle.  The
# tail's width bound belongs to the ragged-tail argument, not to the layout, which
# is why it is imported rather than restated here.
from .ragged_tail import legal_tail_vec
from .thread_budget import MAX_THREADS_PER_BLOCK

# ``cute.arch.WARP_SIZE``; kept as a plain int so this module stays importable
# without a live MLIR context (the ladder is checked on the host).
WARP_SIZE = 32

# quack ``reduction_base.py:22-23`` / ``rmsnorm_config.py:50``.
#
# ⚠ THESE DRIVE :func:`quack_num_threads_for` AND NOTHING ELSE IN CODEGEN.  helion's live
# CTA-size knob is ``num_threads`` (``NumThreadsSpec``), and the reduction thread axis
# is ``LoopedReductionStrategy._thread_count``; neither reads this ladder.  It exists
# so a host-side probe can reproduce quack's verified layout table
# (``_redfix2/adversary/r2/t9_host_invariants.py``), which is what makes "our layout
# differs from quack's here" a checkable statement rather than an opinion.  Promoting
# them to knobs would create a second, dead CTA-size dimension.
# ⚠ The whole-row ``plan_for_shape`` / ``TVLayoutPlan`` pair that used to consume these
# was DELETED (A5) as unreachable from codegen; these ladders survive because
# ``quack_num_threads_for`` and the host invariant probe still read them.
_NUM_THREADS_SMALL = 128
_NUM_THREADS_LARGE = 256
_NUM_THREADS_THRESHOLD = 16 * 1024

# quack ``rmsnorm_config.py:52-56`` == ``softmax.py:36-42``.  The ladder is a
# list of ``(N upper bound, threads cooperating on one row)`` pairs; above the
# last bound the answer is :data:`_DEFAULT_MAX_THREADS_PER_ROW`.
#
# ⚠ ``_DEFAULT_`` IS IN THE NAME ON PURPOSE (renamed from ``_THREADS_PER_ROW_LADDER``).
# It reads as *the* value of ``threads_per_row``, and it is not: it only picks which element
# of ``THREADS_PER_ROW_CHOICES`` the fragment offers FIRST, i.e. the seed.  MEASURED on the
# frozen table 2026-07-31 -- of the 24 cells that name ``cute_threads_per_row``, **13
# DISAGREE with this ladder**, every one of them upward:
#     rms_norm/32768x4096   N=4096   frozen 256  ladder 64    (4x)
#     layer_norm/32768x12000 N=12000 frozen 512  ladder 128   (4x)
#     rms_norm/32768x1024   N=1024   frozen  64  ladder 32
#     ... 10 more, all frozen > ladder
# So the searched winner is systematically WIDER than the seed, which is exactly why the
# name mattered: a reader who takes this for the value would conclude the table is
# mis-tuned.  A constant that only seeds a searchable knob says DEFAULT.
_DEFAULT_THREADS_PER_ROW_LADDER: tuple[tuple[int, int], ...] = (
    (64, 8),
    (128, 16),
    (3072, 32),
    (6144, 64),
    (16384, 128),
)

# The ladder's smallest rung.  quack ``rmsnorm_config.py:52``; below one N per byte
# there is nothing left to cooperate on, so the menu has no reason to go under it.
_THREADS_PER_ROW_MIN = _DEFAULT_THREADS_PER_ROW_LADDER[0][1]


# The ladder's saturation value, DERIVED rather than written down.
#
# ⭐ THIS USED TO BE ``_THREADS_PER_ROW_MAX = 256``, and the 256 was not an
# independent number: it is exactly ``_NUM_THREADS_LARGE``.  ``quack_rows_per_cta_for``
# is ``max(1, quack_num_threads_for(n) // threads_per_row_for(n))`` and a CTA must own at
# least one row, so ``threads_per_row <= num_threads`` is a *requirement* of the
# ladder, not a preference -- which is why quack's own menu stops at 256: quack's
# CTA never exceeds 256 threads.  Writing the number down instead of deriving it is
# what produced the seed/search asymmetry this removes (see
# :data:`THREADS_PER_ROW_CHOICES`).
#
# ⚠ IT WAS A FUNCTION, ``max_threads_per_row_for_ladder()``, AND THE FUNCTION FORM WAS
# THE WRONG QUANTIFIER.  ``max(_NUM_THREADS_SMALL, _NUM_THREADS_LARGE)`` takes no input
# and has no reachable second arm (128 < 256 unconditionally), so it read as a max over
# the CODOMAIN of ``quack_num_threads_for`` -- while the invariant it cites is PER ``n``:
# ``threads_per_row_for(n) <= quack_num_threads_for(n)``.  Those two only coincide because the
# ladder's last rung (16384) and ``_NUM_THREADS_THRESHOLD`` (16*1024) happen to be the
# same number, i.e. the ladder saturates exactly where ``quack_num_threads_for`` steps up.  A
# module constant states the value without implying a quantifier it does not have; the
# per-``n`` invariant is asserted where it belongs, in
# ``test_cute_layout_constants.py::test_ladder_cap_is_the_cta_size_not_a_literal``.
_DEFAULT_MAX_THREADS_PER_ROW = max(_NUM_THREADS_SMALL, _NUM_THREADS_LARGE)


def _pow2_menu(lo: int, hi: int) -> tuple[int, ...]:
    """Powers of two in ``[lo, hi]``, ascending.

    ⚠ POWERS OF TWO, and that is the load-bearing property rather than the
    magnitude: ``threads_per_row`` becomes the lane group of a BUTTERFLY shuffle
    (``reduce_helpers.py:33``), which only totals correctly for a power-of-two
    group.  LEDGER E031 measured what happens otherwise -- a divisor of a ragged
    extent emits the wanted 128-bit kernel and returns relerr 0.15-2.6.  So this
    menu may only ever be built out of powers of two, whatever the bounds.
    """
    out: list[int] = []
    v = 1
    while v <= hi:
        if v >= lo:
            out.append(v)
        v *= 2
    return tuple(out)


# The ``cute_threads_per_row`` SEARCH menu, DERIVED from the two bounds that actually
# constrain it rather than written out.
#
# ⭐ WHY THIS IS A DERIVATION AND NOT A LIST.  run 2 widened this list to 512/1024
# (commit b39ef515b) because the ladder's own answer was too narrow at N=32768, but it
# left the ladder's cap at a literal 256 -- so **the seed could not reach values the
# search could**, and that asymmetry is invisible unless both lines are read together.
# The two numbers have different *principles*, which is exactly why they differ:
#
#   * the LADDER saturates at ``num_threads`` (:data:`_DEFAULT_MAX_THREADS_PER_ROW`),
#     because a CTA must own >= 1 row -- quack's constraint, and quack's CTA is 256;
#   * the SEARCH saturates at the hardware CTA limit
#     (``thread_budget.MAX_THREADS_PER_BLOCK`` == 1024), because helion's reduction
#     thread axis may be the whole block.  ``LoopedReductionStrategy`` only ever
#     *lowers* ``thread_count`` toward this value, so a rung above 1024 could never be
#     taken and one at 1024 is exactly ``next_power_of_2(min(rl, 1024))``, i.e. the
#     value the pinned-derived case already produces.
#
# The measured reason the wide rungs matter (run 2, at N=32768): they are the ones that
# keep ELEMENTS-PER-THREAD near 64 -- ``02_PERF.md`` §6's "single number that most
# distinguishes quack's shape from helion's" -- while still admitting a wide copy.
#
#     rl     tpr   vec  lanes   EPT   copy bits
#     1024   512     2      1    64      32
#     2048   512     4      1    64      64
#     4096   512     8      1    64     128   <- full width AT the EPT invariant
#     4096   256     8      2   128     128   <- what the 256 cap alone could reach
THREADS_PER_ROW_CHOICES: tuple[int, ...] = _pow2_menu(
    _THREADS_PER_ROW_MIN, MAX_THREADS_PER_BLOCK
)

# quack ``rmsnorm_config.py:21-23`` ``RmsNormFwdConfig.reload_from``.  ``"gmem"``
# is deliberately omitted: quack's forward ladder never selects it
# (``rmsnorm_config.py:89``).
REDUCTION_RELOAD_CHOICES: tuple[str | None, ...] = (None, "smem")

# One 128-bit copy is the widest the copy atom can issue.
MAX_COPY_BITS = 128

# helion promises a blanket 16-byte ``assumed_align`` on every tensor pointer it hands
# the DSL (``runtime/__init__.py:3981``).  That promise is the SOURCE OF TRUTH for the
# alignment half of :func:`legal_vec`'s rule, so it is named once here and defaulted
# from rather than repeated at each signature -- four copies of a promise made in one
# place is three chances to drift from it.
ASSUMED_ALIGN_BYTES = 16

# ``PORT_SPEC_layout.md`` §0 F2 / §5b(3): MEASURED on B200, ``reload_from="smem"``
# beats quack's own Hopper-tuned ``reload_threshold = 8*1024``
# (``rmsnorm_config.py:84``) at N=4096 (1.09x vs 0.89x) and N=8192 (1.09x vs
# 0.88x).  This is a deliberate, measured deviation from quack's ladder.
# ⚠ ``_DEFAULT_`` IS IN THE NAME ON PURPOSE (renamed from ``_RELOAD_FROM_MIN_N``), for the
# same reason as the threads-per-row ladder above and with a MEASURED margin that is even
# larger.  This threshold seeds ``cute_row_residency``'s ladder; it does not decide it.
# MEASURED on the frozen table 2026-07-31, after task 1 migrated all 25 legacy cells onto the
# residency axis, comparing each cell's NAMED residency against what this threshold implies:
#     29 cells name cute_row_residency;  3 agree with the ladder;  **26 DISAGREE**
# and the disagreement is one-directional -- 24 cells sit at ``registers``/``gmem`` where this
# threshold asks for ``smem``.  ⇒ the composed ladder is wrong for almost the whole table, and
# a name that reads as the decision hides that.  (It is still the right DEFAULT: it reproduces
# the pre-axis behaviour, which is what makes the axis a reachability change.)
_DEFAULT_RELOAD_FROM_MIN_N = 4096

# ⛔⛔ ``cute_stage_smem_kb`` IS DELETED (run 2, task 1 steps 2-3), AND WITH IT
# ``STAGE_SMEM_KB_CHOICES`` / ``stage_smem_kb_for`` / ``_DEFAULT_STAGE_SMEM_KB``.
#
# It was the whole-kernel SMEM budget, in KiB, that ``smem`` staging tiles could occupy,
# defaulting to 64.  ⇒ WHY IT WENT: ``cute_row_residency`` names WHERE the second read of
# a reduction row comes from, and a *performance* budget that can overrule it means a
# config can name one memory and emit another.  MEASURED on the frozen table, 13 of 40
# cells recorded a residency the kernel never used.  A named residency may be refused only
# by CAPACITY (the tile does not fit the device) or GEOMETRY (there is no row axis / no TV
# plan) -- both hardware or structural facts, neither a policy preference.
#
# ⚠ WHAT WAS LOST, RECORDED RATHER THAN QUIETLY DROPPED.  The 64 KiB default encoded a
# measured OCCUPANCY table -- 32/64 KiB win by 2-8%, 128 KiB loses by 24-28%
# (``launch__occupancy_limit_shared_mem``: 6 blocks/SM unstaged, 3 at 64 KiB, 1 at 128
# KiB).  Occupancy is NOT expressible as a size limit: a tile can fit the device and still
# cost 93% (measured, ``rms_norm`` 8192x100000 -- 112 KiB against a 227 KiB device).  So a
# wide-but-device-legal row will now STAGE where it previously declined, and may be slower.
# That is the deliberate trade: the residency the config names is the one it gets, and a
# shape that does not want staging says ``gmem``.  ⭐ The frozen table pays nothing for it --
# that cell was re-frozen to the ``gmem`` it was already emitting.
#
# The device-capacity half survives, as
# ``TileStrategy._cute_stage_smem_capacity_bytes``: a B200 reports
# ``shared_memory_per_block_optin = 232448 B = 227 KiB``, and a 240 KiB request ICEs in the
# CuTe DSL while 224 KiB runs -- so the limit is real, it is just not a knob.

# One kibibyte, so ``* 1024`` is not written at each of the several sites that convert
# the knob's units.  The knob is in KiB rather than bytes because every value in the
# measured table is a whole number of KiB and a byte-granular search menu would offer
# thousands of indistinguishable neighbours.
BYTES_PER_KIB = 1024


def threads_per_row_for(n: int) -> int:
    """quack ``rmsnorm_config.py:52-56``: threads cooperating on one row."""
    for limit, threads in _DEFAULT_THREADS_PER_ROW_LADDER:
        if n <= limit:
            return threads
    return _DEFAULT_MAX_THREADS_PER_ROW


def quack_num_threads_for(n: int) -> int:
    """quack ``reduction_base.py:22-23``: the CTA size.  ⚠ REFERENCE ONLY, NOT LIVE.

    ⭐ THE ``quack_`` PREFIX IS THE POINT (renamed from ``num_threads_for``).  This reaches
    codegen through NOTHING: its only consumer is :func:`quack_rows_per_cta_for` (the other,
    ``plan_for_shape`` -> ``TVLayoutPlan``, was itself unreachable and deleted in A5).  helion's
    real CTA-size default is ``NumThreadsSpec``'s own, and the reduction thread axis is
    ``LoopedReductionStrategy._thread_count``.  MEASURED 2026-07-31: outside this module and
    ``test_cute_layout_constants.py`` there are ZERO references in ``helion/``.

    ⚠ IT MUST NOT BE CALLED ``default_*`` -- there is no knob it seeds.  ``*_for(n)``
    elsewhere in this file names a per-``n`` LADDER that feeds a spec's ``_default()``; this
    one feeds a host-side reproduction of quack's table, kept so "our layout differs from
    quack's here" is a checkable statement rather than an opinion.  Two different jobs, and
    the prefix says which at the definition site instead of in a ⚠ block 250 lines up.
    """
    return _NUM_THREADS_SMALL if n <= _NUM_THREADS_THRESHOLD else _NUM_THREADS_LARGE


def quack_rows_per_cta_for(n: int) -> int:
    """``tiler_mn[0]``: rows one CTA owns.

    This is the value ``block_sizes[M]`` must take for the CTA size to come out
    at quack's ``num_threads`` (``PORT_SPEC_layout.md`` §5b(4a) -- the CTA size
    is *derived*, not a separate knob).
    """
    return max(1, quack_num_threads_for(n) // threads_per_row_for(n))


def reload_from_for(n: int) -> str | None:
    """The measured B200 ``reload_from`` ladder (see ``_DEFAULT_RELOAD_FROM_MIN_N``)."""
    return "smem" if n >= _DEFAULT_RELOAD_FROM_MIN_N else None


# ---------------------------------------------------------------------------
# ``cute_row_residency``: WHERE THE SECOND READ OF A REDUCTION ROW COMES FROM.
#
# A two-pass reduction walks its row twice (``rms_norm``: ``sum(x**2)`` then the
# rescale; ``layer_norm``: three times; ``cross_entropy``: ``amax`` then
# ``sum(exp)``).  *Where the row lives between the sweeps* is a SINGLE THREE-WAY
# CHOICE, and quack encodes it as exactly that -- ``rmsnorm_config.py``'s
# ``reload_from_vals = (None, "smem", "gmem")``.
#
# ⭐ WHY THIS AXIS EXISTS AT ALL, given that ``cute_reduction_reload`` and
# ``cute_tv_sweep_cache`` already reach all three kernels between them.  Because
# they reach them as a CONJUNCTION over two knobs of different types on different
# block-id domains, and one of the three has no name:
#
#     registers  <- ``cute_tv_sweep_cache = k > 0``   (an int slot BUDGET, per DEVICE LOOP)
#     smem       <- ``cute_reduction_reload = "smem"`` (an enum, per REDUCTION BLOCK)
#     gmem       <- no encoding: reachable ONLY as ``reload=None AND cache=0``
#
# So (a) ``cute_reduction_reload=None`` means *registers* when the budget is
# positive and *gmem* when it is 0 -- one enum value, two kernels; (b) "exactly one
# mechanism is in effect" is not representable, so nothing enforces it; and (c)
# MEASURED on this tree, ``reload="smem"`` with ``cache=128`` emits the SMEM
# signature and the register cache never fires -- the 2x2 grid has only THREE
# reachable kernels, which is the argument for one three-valued axis.
#
# ⚠ THE ORDER IS LOAD-BEARING TWICE OVER.
#   * ``EnumFragment.default() == choices[0]``, so whatever is first here would
#     become the DEFAULT residency -- and 2b is a REACHABILITY change, not a
#     behaviour change.  Nothing may be first, which is why the fragment does NOT
#     use this tuple directly: ``CuteRowResidencySpec._fragment`` rotates the
#     per-shape ladder's answer to the front (the same shape the four ladders above
#     use).  This tuple is the DOMAIN, the legal set, and the ERROR MESSAGE's
#     vocabulary -- never the default.
#   * It is the order a decline walks DOWNWARD in.  ``"registers"`` and ``"smem"``
#     are both optimisations over ``"gmem"``, and both can be refused by geometry
#     or budget; ``"gmem"`` cannot be refused by anything, because re-reading the
#     row from global is what the kernel does when no mechanism fires.  So gmem is
#     LAST here and is the terminal fallback in ``reduction_strategy``.
ROW_RESIDENCY_CHOICES: tuple[str, ...] = ("registers", "smem", "gmem")

# The residency each mechanism's own knob spells today, so the translation between
# the old two-knob encoding and this axis is written ONCE.  ``memory_ops`` and
# ``reduction_strategy`` both name residencies; a bare string literal in each is
# three chances to drift from this domain.
ROW_RESIDENCY_REGISTERS = "registers"
ROW_RESIDENCY_SMEM = "smem"
ROW_RESIDENCY_GMEM = "gmem"


def row_residency_for(n: int) -> str:
    """The ``cute_row_residency`` ladder: TODAY'S EFFECTIVE DEFAULT, re-derived.

    ⭐ THIS FUNCTION IS WHY 2b IS A REACHABILITY CHANGE.  It must return, for every
    ``n``, the residency the two old knobs' own ladders jointly produce -- so that a
    config which names no residency compiles to the byte-identical kernel it did
    before this axis existed.  Both ladders are unconditional functions of ``n``:

        reload_from_for(n)     -> "smem" for n >= 4096, else None
        tv_sweep_cache_for(n)  -> 128, always (> 0, so "registers" is requested)

    and ``smem`` WINS the overlap wherever both fire.  That is not a policy choice
    made here, it is a MEASURED property of the emission: with ``reload="smem"`` and
    ``cache=128`` the emitted kernel carries ``_tv_spart``/``alloc_smem`` and ZERO
    ``_tv_sweep_cache_*`` declarations, because staging consumes the second read
    before ``fuse_tv_copy_sweeps`` can see two sibling sweeps to fuse.  So the
    composition of the two ladders is exactly the step function below.

    ⚠ It is a REQUEST, not a promise, and that distinction is the whole reason
    ``_fill_missing`` may return it.  At ``n < 4096`` this asks for ``"registers"``
    and the register cache may still decline on its budget; at ``n >= 4096`` it asks
    for ``"smem"`` and staging may still decline on geometry or the SMEM budget --
    in which case the effective residency drops to ``"gmem"`` and the emitted kernel
    SAYS SO (``memory_ops._cute_tv_record_residency``).  A ladder that tried to
    predict the decline would have to know ``plan.lane_extent``, ``thread_block_dims()``
    and the running SMEM charge, none of which exist where ``_fill_missing`` runs.
    """
    return (
        ROW_RESIDENCY_SMEM if reload_from_for(n) == "smem" else ROW_RESIDENCY_REGISTERS
    )


def row_residency_from_legacy(reload_from: str | None, cache_slots: int | None) -> str:
    """The residency the OLD two-knob encoding spells.  ⭐ ONE translation, one place.

    ``(cute_reduction_reload, cute_tv_sweep_cache)`` -> ``cute_row_residency``, so that a
    config written before this axis existed keeps compiling to the kernel it named.  Both
    the normalizer (which fills the new key from the old ones when a caller supplied only
    the old ones) and the strategy (which falls back to this when no slot is registered)
    call THIS function -- two copies of the mapping would be two chances to disagree
    about what an existing config means, and the frozen table is 40 such configs.

    MEASURED on this tree, through the two-knob side door, on ``rms_norm``-shaped
    M=2048 N=8192 bf16 (``_tv_spart`` / rmem-cache-decl / ``cute.copy(`` counts):

        reload=None , cache=128  ->  0 / 1 / 3   == registers
        reload="smem", cache=0   ->  4 / 0 / 3   == smem
        reload=None , cache=0    ->  0 / 0 / 4   == gmem      (the row read TWICE)
        reload="smem", cache=128 ->  4 / 0 / 3   == smem      (staging WINS the overlap)

    The fourth row is why this is not a symmetric 2x2: staging consumes the second read
    before ``fuse_tv_copy_sweeps`` can find two sibling sweeps to fuse, so the register
    cache never fires alongside it.  Hence ``"smem"`` is tested FIRST below -- the 2x2
    grid has only three reachable kernels, which is the whole argument for one axis.

    ``cache_slots is None`` means "this block owns no budget slot at all", which is not
    the same as a budget of ``0``: the pass falls back to its own positive default there
    (``fuse_tv_copy_sweeps``'s ``_DEFAULT_MAX_CACHE_SLOTS``), so the residency it spells
    is ``registers``.  Conflating the two would silently read every slotless kernel as
    ``gmem``.
    """
    if reload_from == ROW_RESIDENCY_SMEM:
        return ROW_RESIDENCY_SMEM
    if cache_slots is None or cache_slots > 0:
        return ROW_RESIDENCY_REGISTERS
    return ROW_RESIDENCY_GMEM


# ---------------------------------------------------------------------------
# The two AST-pass knobs (``cute_online_defer`` / ``cute_tv_sweep_cache``).
#
# These sit here rather than in the passes themselves for the same reason the
# three layout ladders above do: a knob's DOMAIN and its DEFAULT are facts about
# the layout, and ``config_spec`` must be able to read them without importing an
# AST pass (which would import ``ast_extension`` -> ``helion.exc`` from inside
# the autotuner's own import, i.e. a cycle).
# ---------------------------------------------------------------------------

# ``cute_online_defer``: True places the online recurrence's cross-lane combine
# AFTER the reduction loop (one merge per row); False leaves it inside the loop
# (``2 * N/(nt*V)`` merges per row, each a serial dependency between iterations).
#
# ⚠ ORDER IS LOAD-BEARING and it is why this is an ``EnumFragment`` rather than a
# ``BooleanFragment``.  ``BooleanFragment.default()`` is a hardcoded ``False``
# (``config_fragment.py:327``), so a boolean fragment cannot express "searchable,
# defaults True" -- it would flip the default and move the emitted kernel on every
# cell the pass fires on.  ``EnumFragment.default()`` is ``choices[0]``, so putting
# ``True`` first is the whole mechanism by which today's behaviour is preserved.
ONLINE_DEFER_CHOICES: tuple[bool, ...] = (True, False)

# ⚠ SAME ORDER ARGUMENT AS ABOVE, OPPOSITE DEFAULT.  ``cute_ndtile_tv`` promotes the
# ``HELION_CUTE_NDTILE_TV`` env gate to a config knob, and the env gate's default is
# OFF -- so ``False`` must come FIRST here, or every existing config silently acquires
# the TV emission.  MEASURED, that is not a hypothetical: enabling the TV path is a
# REPLACEMENT of an established emission (2x ``cute.arch.load`` of vector<4 x Uint16>
# -> 1x 128-bit ``cute.copy``), it moves 8 of the 40 frozen perf cells, and it
# displaces a form pinned by 3 tests in ``test_cute_tile_loop_vec_hoist.py``.
NDTILE_TV_CHOICES: tuple[bool, ...] = (False, True)


def ndtile_tv_for(n: int) -> bool:
    """The ``cute_ndtile_tv`` ladder: today's default, which is OFF.

    ⭐ RETURNS FALSE UNCONDITIONALLY, AND THAT IS A DELIBERATE NON-DECISION rather
    than a measured threshold.  The promotion from env gate to config knob is a pure
    REACHABILITY change: it must leave every existing config compiling to the byte it
    compiles to today, and the env gate's default is off.  Encoding a shape-dependent
    ladder here would make the promotion a behaviour change as well, which is exactly
    what the byte-identical-hash gate (level 1) exists to catch.

    ⚠ AND A LADDER WOULD BE WRONG EVEN AS A FOLLOW-UP, on the evidence.  MEASURED on
    all 8 ``cross_entropy_online`` cells the gate moves, one gate arm per process,
    position-balanced, judged on the mean (the event timer is quantized to ~2.04us on
    this box so a median cannot resolve sub-quantum deltas):

      * the sign tracks LANE EXTENT (= chunk / (num_threads * vec)), not N -- but ⛔
        THE "+81% TO +110%" READING OF IT IS REFUTED, and this comment used to assert
        it (corrected 2026-08-01, run 2 T0).  Those magnitudes are the gate-**OFF**
        LOSSES from raising ``chunk``, not wins for the TV arm:
            32768x8192   tv_OFF chunk2048 ext8   +80.084%   loss
            32768x12000  tv_OFF chunk4096 ext16  +83.238%   loss
        i.e. the number says how much the TV arm RESCUES a wide chunk, not how much it
        beats the incumbent.  With the arm ON, extent 8-16 is worth only -0.5% to
        -6.9%, and **extent 4 BEATS extent 16** on 3 of the 4 cells measured in both
        arms (32768x8192: ext4 -4.99% vs ext16 -1.34%; 32768x16384: -4.54% vs -1.35%).
        The extent-2 claim is refuted too: ``tv_on`` at the incumbent chunk (extent 2)
        WINS on 4 cells, up to -4.46%, rather than losing ~2%.
        ⇒ the rule is "extent 4 on the narrow-chunk cells, 8-16 only on the already-wide
        ones", NOT "bigger extent is better".  The mechanism claim survives: the legacy
        path hoists exactly TWO ``arch.load``s regardless of extent, so it cliffs ~1.8x
        between extent 4 and 8 while the TV path stays flat -- that cliff IS the
        gate-OFF loss above, which is how the sign got inverted in the first place.
      * so three of the eight cells prefer OFF at their CURRENT config and ON after
        their chunk/threads are re-tuned.  A single default cannot express that; a
        per-loop knob can, which is the whole reason for this promotion.  (Run 2 of the
        overtime climb then re-froze 8 of them onto ``tv=True`` with re-tuned chunk, at
        -0.5% to -5.4% confirmed at 7 rounds x 60 reps.)

    ⇒ the ladder stays inert and the SEARCH (or a hand-pinned config) decides.  A
    future ladder should be a function of lane extent, and lane extent is not knowable
    here -- it depends on ``block_sizes`` and ``num_threads``, which are themselves
    being searched.  That is the second reason this is a knob rather than a heuristic.
    """
    return False


def online_defer_for(n: int) -> bool:
    """The ``cute_online_defer`` ladder: today's default, which is ON.

    ⚠ RESTORED (run 2): the block that deleted ``cute_tv_sweep_cache``'s constants took
    this with it, and it is a LIVE ladder for a knob that is being KEPT --
    ``CuteOnlineDeferSpec._default`` and ``_redfix2/bench_reductions.py`` both call it.
    Caught immediately by an ImportError rather than by a wrong answer, but it is the
    second time a scripted range-delete in this cleanup overreached; prefer a targeted
    edit over a computed span.

    Returns True unconditionally, and that is a deliberate non-decision rather than a
    measured threshold: promoting the deferral from an env gate to a config knob must
    leave every existing config compiling to the byte it compiles to today, and the
    deferral ran unconditionally before the knob existed.  See
    ``cute/defer_online_merge.py`` for the measured 7-of-8 win it reproduces.
    """
    del n
    return True


# ⛔⛔ ``cute_tv_sweep_cache`` IS DELETED (run 2, task 1 steps 2-3), AND WITH IT
# ``TV_SWEEP_CACHE_CHOICES`` / ``tv_sweep_cache_for`` / ``_DEFAULT_TV_SWEEP_CACHE_SLOTS``
# (128) and ``fuse_tv_copy_sweeps``'s ``_DEFAULT_MAX_CACHE_SLOTS`` alias.
#
# It was the per-thread register budget, IN SLOTS, that ``fuse_tv_copy_sweeps`` could spend
# caching a row's fragment lanes across sweeps, and ``cache_slots > max_slots -> continue``
# was a LIVE decline -- i.e. a *performance* ceiling could turn a config's ``registers``
# into an emitted ``gmem``.  Same reason as the SMEM budget above: a residency the config
# NAMES may be refused only by capacity or geometry, never by policy.
#
# ⚠ WHAT REPLACES THE DECLINE: nothing.  A row whose footprint exceeds what a thread can
# hold now SPILLS under the residency that was asked for, rather than being silently demoted
# to a different kernel.  A visible cost beats an invisible substitution -- and the 128 was
# never a hardware fact (the rmem tensor is the loaded dtype, so 128 slots is 64 registers
# at worst), it was a spill-avoidance guess.
#
# ⚠ AND THE ``gmem`` VETO IS NOT A BUDGET.  ``codegen_function_def`` used to express
# "residency is explicitly gmem" as ``budget=0``, reusing the pass's own spelling for
# decline.  That conflation let ``HELION_TV_SWEEP_FUSE_SLOTS`` silently UN-VETO three
# explicit-``gmem`` cells (measured), collapsing two residencies into one kernel.  The veto
# is now its own channel -- ``fuse_tv_copy_sweeps(vetoed_offsets=...)``, a SET -- which no
# numeric override can raise.


def cap_cluster_n(cluster_n: int, n: int, vec: int, threads_per_row: int) -> int:
    """quack ``reduction_base.py:28-40`` ``_cap_cluster_n``.

    Without this the cluster double-counts: when
    ``threads_per_row * cluster_n > n // vec`` one CTA tile already spans the
    whole row, ``local_tile`` collapses every peer onto tile 0, and every peer
    re-reduces the same columns.
    """
    return max(1, min(cluster_n, max(1, (n // vec) // threads_per_row)))


def legal_vec(
    n: int,
    dtype_bits: int,
    *,
    row_stride_elems: int | None = None,
    assumed_align_bytes: int = ASSUMED_ALIGN_BYTES,
) -> int:
    """The widest legal ``vec`` for this access, in elements.

    quack computes ``gcd(N, 128 // widest_participating_dtype_width)``
    (``rmsnorm.py:117-120``).  Two additions, both required in helion
    (``PORT_SPEC_layout.md`` §6c, MEASURED in ``_redfix/sanity/s3d_align_rule.py``):

    * the **row stride**, not just ``N``.  quack ``_ensure_contiguous``-es on the
      host (``rmsnorm.py:55-61``); helion passes the real ``arg.stride(d)``
      through (``runtime/__init__.py:3974-3982``), so a non-contiguous view can
      have ``row_stride % vec != 0``.  That is an IR-verification failure at
      compile time, for **both** ``CopyUniversalOp`` and ``cpasync.CopyG2SOp``
      -- the in-tree comment at ``program_id.py:4330-4341`` claiming universal
      copy dodges the check is FALSE.  Clamping here degrades a legitimate
      non-contiguous input to a narrower copy instead of failing to compile.
    * the pointer's ``assumed_align`` (helion promises a blanket 16 at
      ``runtime/__init__.py:3981``), because the rule is
      ``gcd(assumed_align, byte row stride) >= num_bits_per_copy / 8``.
    """
    if dtype_bits <= 0:
        return 1
    vec = max(1, MAX_COPY_BITS // dtype_bits)
    vec = math.gcd(vec, max(n, 1))
    if row_stride_elems is not None:
        vec = math.gcd(vec, max(abs(row_stride_elems), 1))
    # Pointer-alignment half of the rule: a ``vec``-wide copy moves
    # ``vec * dtype_bits / 8`` bytes and needs that many bytes of alignment.
    while vec > 1 and (vec * dtype_bits) // 8 > assumed_align_bytes:
        vec //= 2
    return max(1, vec)


def widest_dtype_bits(dtype_bits: tuple[int, ...] | list[int]) -> int:
    """The widest dtype partitioned *through the layout*.

    quack ``rmsnorm.py:117-120`` takes the max over every participating tensor
    (``mX, mRes, mW, mB, mO, mResO``), because one ``vec`` serves them all.  A
    rank-1 side output such as ``inv_rms`` is not partitioned through the row
    layout and must not be included.
    """
    return max((bits for bits in dtype_bits if bits > 0), default=16)


# ``cute.copy`` through a 128-bit ``CopyUniversalOp`` atom vs
# ``cute.arch.load`` with an explicit ``ir.VectorType``: MEASURED
# (``_redfix/repro/probe_load_widths.py``, ``probe_target_body.py``) these have
# DIFFERENT width limits, and the difference is what makes V=8 reachable.
#
#   cute.arch.load(ptr, VectorType([8], Uint16))  -> ICE at V=8 (16-bit elems)
#   cute.copy(atom(128 bits), tXgX[...], frag)    -> OK at V=8, all of
#                                                   bf16/fp16/fp32
#
# That is why the legacy hoist path caps at V=4 (``memory_ops.py:1980-1983``)
# and needs ``tile_unroll_split2``'s two half-loads to reach 16 bytes, while the
# TV path needs no such workaround.  ``TV_COPY_MAX_VEC_BITS`` records the real
# ceiling for the ``cute.copy`` route.
#
# ⭐ AND THAT CEILING IS ``MAX_COPY_BITS``, not a second number.  The whole content of
# the paragraph above is that ``cute.copy`` REACHES the copy atom's own limit while
# ``cute.arch.load`` falls short of it -- so the two names describe one hardware fact
# (the widest a copy atom can issue) plus one DSL shortfall, and only the shortfall is a
# separate value (the ``vec_width > 4`` gate in ``cute/memory_ops.py``).  Written as a
# literal these could drift apart, and the drift would be silent: a plan built at
# ``MAX_COPY_BITS`` would pass ``ChunkTVPlan.__post_init__``'s check against a smaller
# ``TV_COPY_MAX_VEC_BITS`` only by accident of them being equal today.
TV_COPY_MAX_VEC_BITS = MAX_COPY_BITS


@dataclasses.dataclass(frozen=True)
class ChunkTVPlan:
    """The TV layout for ONE of helion's chunk iterations.

    (Formerly described as "a ``TVLayoutPlan`` scoped to one chunk"; that whole-row class
    had no production consumer and was deleted in A5, so this is now the only plan type.)

    This is the object codegen routes through.  It answers exactly three
    questions, all from the same ``vec``:

    * how wide is one copy         -> :attr:`copy_bits` (the atom)
    * which elements does a thread own -> ``val_layout = (1, vec)``
    * how many copies per chunk    -> :attr:`lane_extent`

    Because ``lane_extent`` is *derived* here rather than divided down
    in-place by whoever happens to pick a width, there is no way to emit a
    chunk whose loop trip count assumes one ``vec`` while its copies use
    another.  That is the whole of class 1 (``PORT_SPEC_layout.md`` §7):
    "index and access width disagree" stops being expressible.

    Attributes:
        chunk: the reduction-axis extent one outer iteration covers
            (helion's ``reduction_loops[i]``, i.e. ``_REDUCTION_BLOCK_n``).
        threads_per_row: threads cooperating on the chunk == helion's
            ``_thread_count``.  ``thr_layout`` is ``(1, threads_per_row)``.
        vec: the ONE width.  Drives ``copy_bits``, ``val_layout`` and
            ``lane_extent`` together.
        dtype_str: CuTe dtype expression for the copy atom's element type.
        dtype_bits: that dtype's width in bits.
    """

    chunk: int
    threads_per_row: int
    vec: int
    dtype_str: str
    dtype_bits: int

    def __post_init__(self) -> None:
        if self.threads_per_row <= 0:
            raise ValueError(f"threads_per_row must be positive: {self}")
        if self.vec <= 0:
            raise ValueError(f"vec must be positive: {self}")
        if self.chunk <= 0:
            raise ValueError(f"chunk must be positive: {self}")
        # The bijection: every element of the chunk is owned by exactly one
        # (thread, value) pair.  Without this the partition is not a partition
        # and coverage would be an arithmetic coincidence rather than an
        # identity -- which is precisely how class 1 happened.
        if self.chunk % (self.threads_per_row * self.vec):
            raise ValueError(
                f"chunk={self.chunk} is not a multiple of "
                f"threads_per_row*vec={self.threads_per_row * self.vec}: {self}"
            )
        if self.copy_bits > TV_COPY_MAX_VEC_BITS:
            raise ValueError(
                f"copy width {self.copy_bits} bits exceeds "
                f"{TV_COPY_MAX_VEC_BITS}: {self}"
            )

    @property
    def copy_bits(self) -> int:
        """``num_bits_per_copy``: the single source of the access width."""
        return self.vec * self.dtype_bits

    @property
    def lane_extent(self) -> int:
        """Copies one thread issues per chunk == the lane loop's trip count.

        Derived from ``vec``, never divided down in place.  ``chunk`` is
        covered iff ``lane_extent * threads_per_row * vec == chunk``, which
        ``__post_init__`` guarantees.
        """
        return self.chunk // (self.threads_per_row * self.vec)

    def covers_chunk(self) -> bool:
        """The coverage identity, as an assertable predicate."""
        return self.lane_extent * self.threads_per_row * self.vec == self.chunk

    @property
    def lane_stride(self) -> int:
        """Element distance between consecutive lane iterations: ``tpr * vec``.

        One lane iteration advances by a WHOLE tile, because
        :meth:`emit_tiled_copy` lays a tile out as ``thr_layout=(1, tpr)``
        ``order=(1, 0)`` × ``val_layout=(1, vec)`` -- thread ``t`` owns columns
        ``[t*vec, t*vec + vec)`` and the ``tpr`` threads together cover
        ``tpr*vec`` consecutive columns.  So the *thread* stride is ``vec`` and
        the *lane* stride is ``tpr*vec``, NOT the other way round.
        """
        return self.threads_per_row * self.vec

    def emit_lane_base(
        self,
        offset_expr: str | None,
        lane_var: str,
        thread_expr: str,
        *,
        extra_terms: str = "",
    ) -> str:
        """⭐ THE ONE lane-base column expression.  Every TV consumer asks here.

        Returns the element index of the START of the ``vec``-wide run that
        ``thread_expr`` owns on lane iteration ``lane_var``::

            column(lane, tid) = offset + tid*vec + lane*(tpr*vec)

        and the caller adds the constexpr ``vi`` (0 <= vi < ``vec``) to reach an
        element.  Derived from :attr:`vec` and :attr:`lane_stride`, i.e. from the
        same two numbers :meth:`emit_tiled_copy` builds the layout from, so the
        expression and the layout cannot disagree.

        ⛔⛔ WHY THIS METHOD EXISTS -- IT IS A FIXED WRONG-ANSWER BUG, NOT TIDYING.
        This formula used to be hand-written at three call sites.  Two agreed;
        the third (``CuteNDTileStrategy.codegen_device_loop``) had the two
        strides TRANSPOSED -- ``offset + tid*EPT + lane*vec`` -- which addresses
        a *blocked* layout while the copy it feeds addresses this *interleaved*
        one.  On a divisible extent that merely permutes the columns (a
        reduction cannot see it); on a RAGGED extent it walks off the end of the
        row, and two frozen ``cross_entropy_online`` cells returned ``nan`` and
        read out of bounds.  ⇒ do not re-derive this anywhere; call this.

        ⚠ The transposition is INVISIBLE at ``lane_extent == 1``, where the
        ``lane*`` term is zero and both spellings coincide.  That is why the bug
        survived: most configs cannot express it.

        Args:
            offset_expr: the chunk's base column (already in ELEMENT units), or
                ``None`` when the chunk IS the row and there is no outer offset
                (the persistent path) -- omitted rather than passed as ``"0"`` so
                no dead ``0 +`` term reaches the emitted source.
            lane_var: the outer lane-loop induction variable.
            thread_expr: this thread's index on the row axis, **already spelled
                exactly as the caller wants it to appear** (parenthesise it if it
                is compound).  A string rather than a plan field because how a
                strategy names and wraps its thread index is the strategy's
                business (``cute.arch.thread_idx()[ax]``, a cached var, an
                ``_index_init_expr`` call, ...).  ⭐ The division of labour is the
                point: the plan owns the STRIDES and the TERM ORDER -- the part
                that was got wrong -- and the caller owns the spelling.
            extra_terms: appended verbatim, for a caller that must add a term in
                element units (the cluster column offset).  Kept explicit so an
                extra term is visible at the call site instead of hidden in a
                bespoke local formula.
        """
        return _lane_base_expr(
            vec=self.vec,
            lane_stride=self.lane_stride,
            offset_expr=offset_expr,
            lane_var=lane_var,
            thread_expr=thread_expr,
            extra_terms=extra_terms,
        )

    def rmem_slots(self, num_chunks: int) -> int:
        """Register slots the sweep cache needs: ``num_chunks * lane_extent * vec``."""
        return num_chunks * self.lane_extent * self.vec

    def emit_rmem_sweep_base(self, chunk_expr: str, lane_var: str) -> str:
        """The sweep-cache index -- ⚠ a DIFFERENT formula, deliberately, and here so
        that the difference is stated by the plan instead of rediscovered.

        ``make_rmem_tensor`` is **per-thread** storage, so thread identity is
        implicit in *which registers you own* and there is NO ``tid`` term::

            slot(chunk, lane) = chunk*(lane_extent*vec) + lane*vec

        Compare :meth:`emit_lane_base`, the *global column* index, which does
        carry ``tid`` and strides the lane by ``tpr*vec``.  Mixing the two up
        would be the same class of bug as the transposition that method
        documents, so both live here rather than at their call sites.

        ⚠ ``per_chunk = lane_extent * vec`` is the per-thread footprint of one
        chunk (not ``chunk`` itself, which counts the whole row's elements
        across all ``tpr`` threads).  The index is in range iff
        ``chunk_expr < num_chunks`` and ``lane_var < lane_extent``, which the
        caller's own loop bounds give.
        """
        return f"({chunk_expr}) * {self.lane_extent * self.vec} + ({lane_var}) * {self.vec}"

    def with_vec(self, vec: int) -> ChunkTVPlan:
        """The ONLY legal way to narrow a copy (``PORT_SPEC_layout.md`` §7).

        Re-derives ``copy_bits`` and ``lane_extent`` together.  A caller that
        cannot do a width-``vec`` copy must come back through here; it may
        never keep these addresses and change the width.
        """
        return dataclasses.replace(self, vec=vec)

    def for_dtype(self, dtype_str: str, dtype_bits: int) -> ChunkTVPlan:
        """⭐ THE SAME LAYOUT, RETYPED FOR ONE PARTICIPANT (A7c: mixed dtypes).

        A mixed-dtype access group needs ONE copy atom per distinct dtype -- the atom's
        element type has to match the tensor being copied -- but it must keep ONE shared
        geometry, or the legs address different elements.  This returns the plan with only
        ``dtype_str``/``dtype_bits`` replaced, so ``chunk``/``threads_per_row``/``vec`` (and
        therefore ``lane_extent``) are carried over unchanged and ``copy_bits`` re-derives.

        ⭐ WHY THAT IS SOUND, and it is a property of the emitters rather than a convention:
        :meth:`emit_tiled_copy` builds ``thr_layout=(1, tpr) order=(1, 0)`` x
        ``val_layout=(1, vec)`` and **neither mentions the dtype**.  So two atoms that differ
        only in element type tile the chunk IDENTICALLY -- thread ``t`` owns columns
        ``[t*vec, t*vec+vec)`` in both -- and the per-dtype ``partition_S``/``partition_D`` of
        the same coordinates therefore name the same elements.  MEASURED at
        ``chunk=512 tpr=32 vec_cap=8``: fp8 / bf16 / fp32 all give ``vec=4 lane_extent=4
        tpr=32``, differing only in ``copy_bits`` (32 / 64 / 128).

        ⚠ IT DOES **NOT** RE-ASK FOR A WIDTH, and must not.  ``vec`` was already bounded by
        the WIDEST participant (``build_tv_plan`` clamps with ``widest_dtype_bits``), which is
        what makes one count legal for every member.  Re-deriving a width per dtype here would
        let a narrow participant widen past the group's bound -- two answers to "how wide is
        one copy", which is bug class 1 by definition.
        """
        return dataclasses.replace(self, dtype_str=dtype_str, dtype_bits=dtype_bits)

    def widest_legal_vec(
        self,
        *,
        row_stride_elems: int | None = None,
        assumed_align_bytes: int = ASSUMED_ALIGN_BYTES,
    ) -> int:
        """The widest ``vec`` this chunk could legally use.

        Clamped by :func:`legal_vec` (dtype width, ``N``, the real row stride
        and the pointer alignment) and additionally by the chunk-coverage
        requirement, so the result is always constructible.
        """
        vec = legal_vec(
            self.chunk,
            self.dtype_bits,
            row_stride_elems=row_stride_elems,
            assumed_align_bytes=assumed_align_bytes,
        )
        while vec > 1 and self.chunk % (self.threads_per_row * vec):
            vec //= 2
        return max(1, vec)

    # -- emission ------------------------------------------------------------

    def emit_copy_atom(self, var: str) -> str:
        """The store leg needs ``CopyUniversalOp`` (cp.async is gmem->smem
        only), and helion's load leg goes gmem->rmem directly, so both legs use
        the universal op and can therefore share ONE atom."""
        return (
            f"{var} = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), "
            f"{self.dtype_str}, num_bits_per_copy={self.copy_bits})"
        )

    def emit_tiled_copy(self, var: str, atom_var: str) -> str:
        """``thr_layout = (1, tpr)`` because the row is helion's, not the
        layout's (see the note above ``ChunkTVPlan``).  ``order=(1, 0)`` keeps
        ``threads_per_row`` fastest-varying so consecutive threads take
        consecutive column blocks; ``val_layout = (1, vec)`` puts a thread's
        ``vec`` elements contiguous along N."""
        return (
            f"{var} = cute.make_tiled_copy_tv({atom_var}, "
            f"cute.make_ordered_layout((1, {self.threads_per_row}), order=(1, 0)), "
            f"cute.make_layout((1, {self.vec})))"
        )

    def emit_get_slice(self, var: str, tiled_copy_var: str, tidx: str) -> str:
        """Exactly ONE of these per (dtype, chunk) group.  More than one slice
        means the load and store legs can disagree, whatever the store-sector
        counter says (LEDGER E001)."""
        return f"{var} = {tiled_copy_var}.get_slice({tidx})"

    def emit_row_broadcast_view(
        self, var: str, tensor: str, n_expr: str, stride_expr: str
    ) -> str:
        """A rank-1 tensor (e.g. ``weight``) viewed as ``(1, N)`` with row
        stride 0, so it partitions through the SAME slice as the row tensors.
        quack does this host-side (``rmsnorm.py:124-127``)."""
        return (
            f"{var} = cute.make_tensor({tensor}.iterator, "
            f"cute.make_layout((1, {n_expr}), stride=(0, {stride_expr})))"
        )

    def emit_local_tile(
        self, var: str, tensor: str, row_coord: str, col_coord: str
    ) -> str:
        """One chunk of one row.  ``row_coord`` is helion's CLAMPED row index
        -- passed as a tile coordinate, never folded into the iterator (that
        loses the DSL's alignment proof; MEASURED)."""
        return (
            f"{var} = cute.local_tile({tensor}, (1, {self.chunk}), "
            f"({row_coord}, {col_coord}))"
        )

    def emit_partition_source(self, var: str, thr_var: str, tile: str) -> str:
        return f"{var} = {thr_var}.partition_S({tile})"

    def emit_partition_dest(self, var: str, thr_var: str, tile: str) -> str:
        """``partition_D`` on the OUTPUT from the same slice -- this is what
        makes the store coalesced for free (MEASURED 8,388,608 store sectors =
        1.00x floor at rms 32768x4096, ``s7_store_sectors.py``)."""
        return f"{var} = {thr_var}.partition_D({tile})"

    def emit_fragment_like(self, var: str, partition_var: str) -> str:
        """One lane iteration's register fragment.  ``[None, 0, 0]`` selects
        mode 0 (the ``vec`` values) at row 0, lane 0; every lane iteration has
        the same shape, so one ``_like`` serves them all."""
        return f"{var} = cute.make_rmem_tensor_like({partition_var}[None, 0, 0])"

    def emit_lane_slice(self, partition_var: str, lane_var: str) -> str:
        """The lane's slice of a partition: ``tXgX[None, 0, <lane>]``.

        helion's existing ``for lane`` loop IS this partition's rest mode --
        MEASURED (``probe_dsl_caps.py`` probe B): a chunk partition has shape
        ``((vec,1), 1, lane_extent)``.
        """
        return f"{partition_var}[None, 0, {lane_var}]"

    def emit_copy(self, src: str, dst: str, atom_var: str) -> str:
        """``cute.copy`` with the ATOM, not the ``TiledCopy``: MEASURED
        (``s3b_cpasync_alignment.py``) passing the ``TiledCopy`` on a
        ``from_dlpack`` tensor fails with "dynamic layout is not supported".

        Also note ``cute.copy`` is what makes V=8 reachable at all: the same
        16 bytes via ``cute.arch.load(..., VectorType([8], Uint16))`` ICEs.
        See :data:`TV_COPY_MAX_VEC_BITS`.
        """
        return f"cute.copy({atom_var}, {src}, {dst})"

    # -- ``reload_from="smem"``: stage the row, then re-read it ---------------
    #
    # quack ``rmsnorm.py:166-168`` allocates ``sX`` with the SAME tiler and the
    # SAME ``order=(1, 0)`` as the gmem tiler, and ``:277-288`` / ``:332-343``
    # re-read it with ``cute.autovec_copy`` after the reduction barrier.
    # ``PORT_SPEC_layout.md`` §5 is explicit that diverging on tiler or order is
    # where bank conflicts come from, so both emitters below take their shape
    # from ``self`` and are not parameterised.
    #
    # ⚠ THE SIZING TRAP, and it is the whole story of step 1c.  quack's ``sX`` is
    # ``(rows_per_cta, tiler_n)`` where ``tiler_n * cluster_n == N``, and quack has
    # NO chunk loop -- one CTA tile spans the row, so ``sX`` is written once on
    # the way in and read back once after the barrier.  helion keeps its
    # ``reduction_loops`` chunk loop (``ChunkTVPlan``'s note 2) AND runs the two
    # sweeps as two SEPARATE ``for roffset`` loops.  So sweep 2's chunk ``c`` needs
    # the data sweep 1 wrote at chunk ``c`` -- which means the buffer must hold
    # EVERY chunk, i.e. the whole row.  A chunk-sized buffer is 1/num_chunks the
    # size and MEASURED 1.033x at rms 32768x16384, but it is WRONG: relerr 261.6,
    # because every chunk of sweep 2 reads back the LAST chunk sweep 1 wrote.
    #
    # Consequence: the footprint is ``rows_per_cta * N``, and it GROWS WITH N,
    # where quack's is flat at 32 KB because ``cluster_n`` sits in ``tiler_n``'s
    # denominator.  That is why staging pays at N=4096/8192 and is declined above:
    # ``LoopedReductionStrategy.cute_stage_feasible`` caps it at the measured
    # occupancy cliff.  Step 2's cluster is what would make it flat again.

    def stage_smem_elems(self, rows_per_cta: int, num_chunks: int) -> int:
        """Elements in the staged tile: every chunk of the row, for every CTA row.

        ``num_chunks * chunk == N``, so this is ``rows_per_cta * N``.  See the
        sizing trap above for why it cannot be ``rows_per_cta * chunk``.
        """
        return max(1, rows_per_cta) * max(1, num_chunks) * self.chunk

    def stage_smem_bytes(self, rows_per_cta: int, num_chunks: int) -> int:
        return self.stage_smem_elems(rows_per_cta, num_chunks) * self.dtype_bits // 8

    def emit_stage_smem_alloc(
        self, ptr_var: str, rows_per_cta: int, num_chunks: int
    ) -> str:
        """The staged tile's alignment is :data:`ASSUMED_ALIGN_BYTES`, matching quack's
        ``byte_alignment=16``.

        ⭐ IT IS THE SAME NUMBER AS THE GMEM LEG'S AND FOR THE SAME REASON, so it is
        derived from the same constant rather than written again: the copy atom moves up
        to ``MAX_COPY_BITS`` bits == 16 bytes per thread, and a narrower alignment fails
        the DSL's alignment proof here exactly as it does on gmem.  Written as a literal
        it could drift from the promise :func:`legal_vec` clamps against, and the drift
        would surface as a compile-time IR-verification failure on the SMEM leg only --
        the hardest half to attribute.
        """
        return (
            f"{ptr_var} = cute.arch.alloc_smem({self.dtype_str}, "
            f"{self.stage_smem_elems(rows_per_cta, num_chunks)}, "
            f"alignment={ASSUMED_ALIGN_BYTES})"
        )

    def emit_stage_smem_tensor(
        self, var: str, ptr_var: str, rows_per_cta: int, num_chunks: int
    ) -> str:
        """quack ``rmsnorm.py:166-168``: same ``order=(1, 0)`` as the gmem tiler,
        so ``partition_S``/``partition_D`` off the SAME slice land on the same
        elements and a thread reads back exactly what it wrote.  ``PORT_SPEC``
        §5 is explicit that diverging on the order is where bank conflicts come
        from, which is why the order is not a parameter here."""
        return (
            f"{var} = cute.make_tensor({ptr_var}, "
            f"cute.make_ordered_layout(({max(1, rows_per_cta)}, "
            f"{max(1, num_chunks) * self.chunk}), order=(1, 0)))"
        )

    def emit_stage_local_tile(
        self, var: str, smem_var: str, row_coord: str, chunk_coord: str
    ) -> str:
        """The staged tile for ONE chunk of ONE CTA row.

        Same ``(1, chunk)`` tiler and same chunk coordinate as the gmem
        ``local_tile``, so the staged copy is index-for-index the gmem copy with
        a different address space.  ``row_coord`` is the CTA-LOCAL row
        (``thread_idx()[1]``), never the global row -- the buffer is per-CTA.
        """
        return (
            f"{var} = cute.local_tile({smem_var}, (1, {self.chunk}), "
            f"({row_coord}, {chunk_coord}))"
        )

    def emit_stage_copy(self, src: str, dst: str) -> str:
        """quack ``rmsnorm.py:278`` / ``:334`` use ``cute.autovec_copy`` for the
        SMEM legs rather than the gmem copy atom.

        MEASURED equivalent in throughput here (0.4837 vs 0.4779 ms at
        rms 32768x16384 on a hand-edited kernel), so this follows quack.
        ``autovec_copy`` also needs no atom, which keeps the atom count at one
        and therefore keeps ``one_layout_drives_load_and_store`` meaningful.
        """
        return f"cute.autovec_copy({src}, {dst})"

    def describe(self) -> str:
        return (
            f"chunk={self.chunk} tpr={self.threads_per_row} vec={self.vec} "
            f"copy_bits={self.copy_bits} lane_extent={self.lane_extent} "
            f"dtype={self.dtype_str}"
        )


@dataclasses.dataclass(frozen=True)
class TVParticipants:
    """The tensors a TV layout partitions, as the plan builder needs them.

    ⭐ THE POINT OF THE TYPE IS THAT IT REPLACES A BARE ``(dtypes, strides)``
    TUPLE THAT TWO WALKERS RETURNED AND ONE FUNCTION HAD TO INTERPRET.  There
    were two such walkers -- ``ReductionStrategy._cute_layout_participants``
    (which recognises its axis as the bare ``slice(None)``) and
    ``CuteNDTileStrategy._cute_ndtile_layout_participants`` (which recognises it
    by ``block_id`` identity) -- and the *interpretation* of what they returned
    was duplicated in each of the two plan builders. Now the walkers keep their
    different notions of "my axis" (they must: one is syntax, one is identity)
    and hand the answer here in one shape.

    ``dtype_bits`` is derived rather than stored, so "the widest participant"
    cannot be computed one way in one caller and another way in the other.
    """

    dtypes: tuple[str, ...]
    """CuTe dtype *expressions*, one per participating access, in walk order."""

    dtype_bits: tuple[int, ...]
    """Each participant's element width in bits, parallel to :attr:`dtypes`."""

    row_strides: tuple[int, ...]
    """Each participant's stride-1-most non-contiguous stride (its row stride)."""

    torch_dtypes: tuple[object, ...] = ()
    """The ``torch.dtype`` behind each entry of :attr:`dtypes`, parallel to it.

    ⚠ CARRIED SO A CALLER'S OWN GATE DOES NOT NEED A SECOND WALK.  A gate that
    belongs to a *consumer* rather than to the layout -- ``CuteNDTileStrategy``
    refusing fp8 because its ``tile_unroll`` pipeline shift-extracts a packed
    integer -- has to ask a torch-level question (``_cute_is_byte_packed``) about
    the same tensors this walk already visited.  Re-walking to answer it is how
    two notions of "the participants" appear, which is the defect this type
    exists to remove.

    Typed ``object`` rather than ``torch.dtype`` deliberately: this module is pure
    host arithmetic over ints and strings and is unit-checked without a live MLIR
    context, so it does not import torch.  A caller that reads this knows what it
    put in.
    """

    @property
    def is_empty(self) -> bool:
        return not self.dtypes

    @property
    def distinct_dtypes(self) -> tuple[str, ...]:
        """The dtype expressions, de-duplicated, in first-seen order.

        ⭐ ORDER-STABLE, and that is load-bearing rather than tidy: the atom for
        ``distinct_dtypes[0]`` is the one a single-dtype layout emits today, so a
        set would make the emitted atom depend on hash order.
        """
        seen: dict[str, None] = {}
        for d in self.dtypes:
            seen.setdefault(d, None)
        return tuple(seen)

    @classmethod
    def empty(cls) -> TVParticipants:
        """No participants -- the honest "this walk found nothing" answer.

        A caller reads it as a DECLINE (:func:`build_tv_plan` returns ``None`` on
        it), which is the same thing an empty ``(dtypes, strides)`` tuple meant
        before this type existed.
        """
        return cls(dtypes=(), dtype_bits=(), row_strides=(), torch_dtypes=())

    @classmethod
    def from_accesses(
        cls,
        accesses: list[tuple[str, int, tuple[int, ...], object]],
    ) -> TVParticipants:
        """Build from ``[(dtype_str, dtype_bits, full stride tuple, torch_dtype), ...]``.

        ⭐ THE ROW-STRIDE REDUCTION HAPPENS HERE, ONCE, so the two walkers cannot
        pick different strides off the same tensor.  Both used to spell it
        in-line as ``min(s for s in val.stride() if isinstance(s, int) and s > 1)``
        -- the same expression, written twice, with nothing making them agree.

        ⚠ ``min`` over ONE tensor's own strides, then ``gcd`` ACROSS tensors, and
        the two folds are not interchangeable.  Within one tensor the smallest
        stride above 1 is the row stride (for a 2-D row-major tensor that is
        ``N``; for ``base[:, :N]`` it is the base's row length, which is what
        actually constrains the copy width).  Across tensors the constraint is a
        conjunction, so the fold is ``gcd`` -- see :attr:`row_stride_gcd`.

        ``s > 1`` drops the stride-1 lane dim (never a constraint) and a 0-stride
        broadcast dim from ``.expand()``.  A tensor with no such stride
        contributes no constraint at all, which is why the row-stride tuple can be
        shorter than the dtype tuple.
        """
        dtypes: list[str] = []
        dtype_bits: list[int] = []
        row_strides: list[int] = []
        torch_dtypes: list[object] = []
        for dtype_str, bits, strides, torch_dtype in accesses:
            dtypes.append(dtype_str)
            dtype_bits.append(bits)
            torch_dtypes.append(torch_dtype)
            cand = [int(s) for s in strides if isinstance(s, int) and s > 1]
            if cand:
                row_strides.append(min(cand))
        return cls(
            dtypes=tuple(dtypes),
            dtype_bits=tuple(dtype_bits),
            row_strides=tuple(row_strides),
            torch_dtypes=tuple(torch_dtypes),
        )

    @property
    def row_stride_gcd(self) -> int | None:
        """``gcd`` of the participants' row strides, or ``None`` when there are none.

        ⚠ ``gcd``, NOT ``min``.  ``row_stride_elems`` feeds :func:`legal_vec`'s
        alignment clamp and that clamp must hold for EVERY participant -- it is a
        conjunction, and the conjunction of "stride is a multiple of v" over a set
        of strides is exactly ``gcd(strides) % v == 0``. ``min`` is not a sound
        fold for it.

        MEASURED (run 3 E040, found by an adversary): with a sliced input
        ``base[:, :4096]`` of a ``(128, 4097)`` base, the participants are
        ``[4097, 4096, 4097, 4097, 4096]`` -- the 4096s being the CONTIGUOUS
        output of ``torch.empty_like(x)``. ``min`` returned 4096, so the clamp
        was computed against a stride the INPUT does not have, a 128-bit copy was
        emitted on a row whose stride is odd, and the CuTe DSL raised ``ICE IR
        Verification Failed`` at compile time. ``gcd`` returns
        ``gcd(4097, 4096) == 1`` and the width clamps to 1 -- correct, and it
        declines rather than ICEs.
        """
        if not self.row_strides:
            return None
        return math.gcd(*(abs(s) for s in self.row_strides))


def build_tv_plan(
    *,
    chunk: int,
    threads_per_row: int,
    participants: TVParticipants,
    vec_cap: int | None,
    numel: int | None = None,
    allow_mixed_dtypes: bool = False,
    require_exact_vec_cap: bool = False,
    tail_predicated: bool = False,
    assumed_align_bytes: int = ASSUMED_ALIGN_BYTES,
) -> ChunkTVPlan | None:
    """⭐⭐ THE ONE place a TV access width is decided, for EVERY strategy.

    Every CuTe strategy that wants a TV copy -- looped reduction, loop-free
    (persistent) reduction, N-D tile, and a pure pointwise tile -- calls exactly
    this function. Each supplies the three inputs (``chunk``,
    ``threads_per_row``, ``participants``) and states its policy; nothing else
    about a strategy reaches the width decision.

    ⛔ WHY THIS EXISTS.  There used to be TWO plan constructors --
    ``ReductionStrategy._build_cute_tv_plan`` and
    ``CuteNDTileStrategy._build_cute_tv_plan_for_block`` -- which asked the same
    questions of the same helpers and ended in the same :func:`chunk_plan` call,
    but reached them by different routes and carried gates that differed **by
    accident of coverage rather than by design**: the reduction path rounded a
    ragged extent up and predicated the tail while the tile path declined at
    ``numel % chunk``; the tile path refused any narrowing while the reduction
    path accepted one; the tile path declined fp8 while the reduction path did
    not. None of those divergences was a statement about TV mechanics.

    ⇒ the divergences that remain are **parameters, named and defaulted to the
    conservative answer**, so a caller that does not think about one gets the
    behaviour that cannot be wrong:

    ``allow_mixed_dtypes``
        Admit a participant set with more than one dtype. Off by default: one
        atom per distinct dtype has to be emitted, and a caller whose emission
        site mints exactly one atom must not be handed such a plan.

    ``require_exact_vec_cap``
        Return ``None`` unless the resulting ``vec`` is exactly the ``vec_cap`` **the
        caller passed** -- not the cap after ``tail_predicated`` narrowed it.
        Needed by any caller that has ALREADY fixed its loop trip count from
        ``vec_cap`` before asking (``CuteNDTileStrategy`` builds its outer lane
        loop at ``EPT // vec_cap`` and cannot reshape it afterwards), because a
        narrower copy would then visit fewer elements than the loop assumes --
        bug class 1 exactly. A caller that reads ``lane_extent`` back OFF the
        returned plan does not need it and should not pass it.

    ``tail_predicated``
        The caller will cover a ROUNDED-UP extent and predicate the tail
        (:mod:`ragged_tail`). Then ``vec`` must additionally divide ``numel``,
        because ``cute.copy``'s ``pred`` granularity is one whole vector block
        and a block straddling the row end cannot be partially masked. ⚠ Without
        this the CAP is not narrowed and the plan is built wide, which is class 1
        (the trip count would assume the wider width) -- so it narrows the cap
        here rather than patching later. quack imposes the same thing as
        ``vecsize = gcd(N, 128 // width)``.

    ``numel`` is the axis's true extent. It is only read when
    ``tail_predicated`` is set, and it is a REQUIRED input in that case -- a
    caller that predicates a tail without saying how long the row is cannot be
    served, so that combination raises rather than guessing.

    Returns ``None`` for every decline. A ``None`` and a plan with ``vec == 1``
    are deliberately equivalent in effect (a length-1 copy is the scalar path),
    and neither can produce a strided index without a matching width.
    """
    if chunk <= 0 or threads_per_row <= 0:
        return None
    if participants.is_empty:
        return None
    distinct = participants.distinct_dtypes
    if len(distinct) != 1 and not allow_mixed_dtypes:
        return None
    # ⚠ ``dtype_bits`` FOR THE ATOM IS THE FIRST PARTICIPANT'S, while the WIDTH
    # bound is the widest participant's.  Those are two different questions and
    # conflating them is E010 trap 3: the atom's element type has to match the
    # tensor being copied, but ONE ``vec`` (an element COUNT) serves the whole
    # layout, so it must be legal for the widest element in it.
    atom_dtype_str = distinct[0]
    atom_dtype_bits = participants.dtype_bits[0]
    # ⭐ THE CALLER'S CAP IS REMEMBERED BEFORE THE TAIL NARROWS IT, and the two must
    # not be confused.  ``require_exact_vec_cap`` asks "is this the width my ALREADY
    # EMITTED loop was built for?", which is a question about the caller's own number.
    # The tail bound is a further clamp the LAYOUT imposes.
    #
    # ⛔ MEASURED, when this compared against the narrowed cap instead: at N=1543 the
    # tail bound is ``legal_tail_vec(8, 1543) == 1`` (1543 is prime-ish: 8 ∤ 1543), so
    # ``vec_cap`` fell 8 -> 1, ``plan.vec == 1 == vec_cap`` passed the exactness test,
    # and a ``vec=1`` plan reached a lane loop built at ``EPT // 8``:
    #     AssertionError: TV plan lane_extent=16 disagrees with the emitted outer lane
    #     extent 2 for block 1: chunk=512 tpr=32 vec=1
    # i.e. bug class 1 surfacing as a compile crash.  ⇒ a tail narrowing must make such
    # a caller DECLINE, not silently agree at the narrower width.
    requested_cap = vec_cap
    if tail_predicated:
        if numel is None:
            raise ValueError(
                "build_tv_plan(tail_predicated=True) needs numel: the tail "
                "predicate's width bound is gcd(numel, widest legal vec), and "
                "there is no safe default for a row length."
            )
        widest = max(1, MAX_COPY_BITS // widest_dtype_bits(participants.dtype_bits))
        tail_cap = legal_tail_vec(widest, int(numel))
        vec_cap = tail_cap if vec_cap is None else min(vec_cap, tail_cap)
    try:
        plan = chunk_plan(
            chunk,
            threads_per_row,
            atom_dtype_str,
            atom_dtype_bits,
            vec_cap=vec_cap,
            participating_dtype_bits=list(participants.dtype_bits),
            row_stride_elems=participants.row_stride_gcd,
            assumed_align_bytes=assumed_align_bytes,
        )
    except ValueError:
        # An unconstructible plan is a DECLINE, never a silent width skew: the
        # caller keeps vec == 1 and today's scalar path.
        return None
    if (
        require_exact_vec_cap
        and requested_cap is not None
        and plan.vec != requested_cap
    ):
        return None
    return plan


def emit_lane_base_for(
    *,
    threads_per_row: int,
    vec: int,
    offset_expr: str | None,
    lane_var: str,
    thread_expr: str,
    extra_terms: str = "",
) -> str:
    """:meth:`ChunkTVPlan.emit_lane_base` for a caller that has no plan object.

    ⭐ WHY THIS EXISTS.  The lane-base column formula is a function of exactly
    ``(threads_per_row, vec)``, and the *legacy* per-element ``unroll`` mode needs
    the very same expression as a TV copy -- it reads the elements of the same
    interleaved partition one at a time instead of fetching them in one copy.  So
    that caller must be able to ask for the formula without owning a plan, or it
    keeps a private second spelling -- and a private second spelling is exactly
    what produced the transposed-stride wrong answer that
    :meth:`ChunkTVPlan.emit_lane_base` documents.

    ⚠ A free function, NOT a synthetic ``ChunkTVPlan``: a plan whose ``chunk`` and
    dtype were invented to satisfy the constructor would be a real plan object
    carrying meaningless fields (its ``lane_extent`` and ``copy_bits`` would be
    fiction), and ``ChunkTVPlan`` has exactly ONE construction site on purpose.
    The formula body is not duplicated -- this delegates.
    """
    return _lane_base_expr(
        vec=vec,
        lane_stride=threads_per_row * vec,
        offset_expr=offset_expr,
        lane_var=lane_var,
        thread_expr=thread_expr,
        extra_terms=extra_terms,
    )


def _lane_base_expr(
    *,
    vec: int,
    lane_stride: int,
    offset_expr: str | None,
    lane_var: str,
    thread_expr: str,
    extra_terms: str,
) -> str:
    """⭐ THE one and only spelling of the lane-base column expression.

    Both :meth:`ChunkTVPlan.emit_lane_base` and :func:`emit_lane_base_for` are
    thin wrappers over this, so there is a single place the strides and the term
    order are written down.  See ``emit_lane_base`` for the formula, why the term
    order matters, and the wrong answer that a second copy caused.
    """
    prefix = f"{offset_expr} + " if offset_expr is not None else ""
    return (
        f"{prefix}{thread_expr} * {vec}"
        f" + cutlass.Int32({lane_var}) * {lane_stride}"
        f"{extra_terms}"
    )


def chunk_plan(
    chunk: int,
    threads_per_row: int,
    dtype_str: str,
    dtype_bits: int,
    *,
    vec_cap: int | None = None,
    participating_dtype_bits: tuple[int, ...] | list[int] | None = None,
    row_stride_elems: int | None = None,
    assumed_align_bytes: int = ASSUMED_ALIGN_BYTES,
) -> ChunkTVPlan:
    """Build a :class:`ChunkTVPlan` at the widest legal ``vec``, then cap.

    ``vec_cap`` is ``PORT_SPEC_layout.md`` §5c's inversion:
    ``cute_vector_widths`` is a **cap**, not a request.  The layout computes
    what is legal and the config may only lower it -- never raise it past what
    the layout can do.  That is what makes class 1 unrepresentable: there is no
    path by which a config value becomes a width the addresses do not match.
    """
    widest_bits = (
        widest_dtype_bits(participating_dtype_bits)
        if participating_dtype_bits
        else dtype_bits
    )
    vec = legal_vec(
        chunk,
        widest_bits,
        row_stride_elems=row_stride_elems,
        assumed_align_bytes=assumed_align_bytes,
    )
    if vec_cap is not None:
        vec = max(1, min(vec, vec_cap))
    # Coverage: a vec that does not divide the chunk evenly across the threads
    # cannot be a partition, so narrow until it is.
    while vec > 1 and chunk % (threads_per_row * vec):
        vec //= 2
    return ChunkTVPlan(
        chunk=chunk,
        threads_per_row=threads_per_row,
        vec=max(1, vec),
        dtype_str=dtype_str,
        dtype_bits=dtype_bits,
    )
