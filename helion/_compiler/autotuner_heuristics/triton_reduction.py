from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...runtime.config import Config
from ..reduction_hint import ReductionHint
from .common import matches_hardware
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_spec import ReductionFact
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


def _next_pow2_clamped(value: int, lo: int, hi: int) -> int:
    """Match Inductor's ``next_power_of_2(min(max(v, lo), hi))`` semantics.

    Rounds UP to next power of two after clamping into [lo, hi]. Matches the
    rounding direction in Inductor's ``_num_warps`` at
    ``triton_heuristics.py:3087-3096``: rounding down would systematically
    under-warp INNER reductions.
    """
    value = max(value, lo)
    value = min(value, hi)
    if value <= 0:
        return lo
    if value & (value - 1) == 0:
        return value
    return 1 << value.bit_length()


def _prev_pow2(value: int) -> int:
    """Largest power of two <= value."""
    if value <= 0:
        return 1
    return 1 << (value.bit_length() - 1)


@dataclass(frozen=True)
class _Recipe:
    xblock: int
    reduction_loops: list[int | None]
    num_warps: int


def _compute_limits(fact: ReductionFact) -> tuple[int, int, int, bool]:
    """Returns (MAX_R0_BLOCK, NW_MIN, NW_MAX, register_intensive).

    Mirrors ``triton_heuristics.py:3834-3856``. On H100 (sm90),
    MAX_R0_BLOCK starts at 2048 and drops to 1024 under register pressure.
    """
    max_r0_block = 4096
    nw_max = 16 if (fact.static_rnumel or 0) <= 8192 else 32
    nw_min = 2

    register_intensive = (fact.static_xnumel or 0) >= 1024 and (
        fact.num_load + fact.num_reduction
    ) >= 10
    if register_intensive:
        max_r0_block = 2048
        nw_max = max(nw_min, nw_max // 2)

    return max_r0_block, nw_min, nw_max, register_intensive


def _kernel_class(fact: ReductionFact) -> str:
    """Classify the kernel for recipe selection.

    The boundary between persistent and looped depends more on the reduction
    *structure* than on raw load count:

      "sum-class" (num_reduction == 1, num_load <= 2):
        sum (1L+1R), rms_norm (2L+1R). Memory-bound 1-pass reductions where
        looped + high warps wins from R>=2048 because the inner loop can
        keep more loads in flight than persistent's xb=1 single-program.

      "norm-class" (num_reduction >= 2, num_load <= 1):
        softmax / softmax_decomposed (1L+2R). Two-pass via online or
        explicit amax/sum, but only one input tensor. Persistent through
        R=16384 is the Pareto front -- the second reduction reuses
        registers without re-loading.

      "multi-load" (num_reduction >= 2, num_load >= 2):
        layer_norm (3L+2R), cross_entropy (3L+2R), and similar multi-input
        normalizers. Persistent through R=32768 wins big on cross_entropy
        even though Inductor would pick looped, because Helion's persistent
        codegen keeps the row in registers across both reductions.
    """
    if fact.num_reduction <= 1:
        return "sum"
    if fact.num_load <= 1:
        return "norm"
    return "multi-load"


def _persistent_r_max(fact: ReductionFact) -> int:
    """Largest R for which we keep the reduction persistent (xblock=1).

    Calibrated from per-shape sweeps on H100. See ``_kernel_class``.

      sum-class:    persistent up to R=4096. Above that the looped recipe's
                    in-flight pipeline depth dominates the persistent xb=1
                    single-program throughput on memory-bound 1-pass sums.
      norm-class:   persistent up to R=16384.
      multi-load:   persistent up to R=32768 if X<=16384, else R=8192
                    (X-large kernels like cross_entropy(8192,131072) fit
                    less per program before register pressure).
    """
    cls = _kernel_class(fact)
    if cls == "sum":
        return 4096
    if cls == "norm":
        return 16384
    return 32768 if (fact.static_xnumel or 0) <= 16384 else 8192


def _xblock_for_persistent(fact: ReductionFact, cls: str) -> int:
    """Power-of-two xblock for the persistent recipe, by kernel class.

    xblock>1 only helps when R is small enough that a single row's compute
    can't keep the SM busy. For R>=512 the per-row tile already saturates,
    and packing additional rows mostly adds register pressure without
    speeding up the rate-limiting step.

    The cap formula tile_target/R lets the cap shrink as R grows. The
    tile_target depends on the per-program working-set budget: 2048 for
    sum-class (single load amortizes less per row), 4096 for norm-class
    (re-uses x in registers across two reductions). Multi-load sticks at
    a fixed 4 because its 3+ load streams already fight for the L1.
    """
    r = fact.static_rnumel
    x = fact.static_xnumel
    assert r is not None and x is not None
    if cls == "sum":
        if r <= 256:
            cap = min(16, max(1, 2048 // max(r, 1)))
        else:
            cap = 1
    elif cls == "norm":
        if r <= 256:
            cap = min(16, max(1, 4096 // max(r, 1)))
        elif r <= 512:
            cap = 4
        else:
            cap = 1
    else:  # multi-load
        if r <= 512:
            cap = 4
        else:
            cap = 1
    xblock = 1
    while xblock * 2 <= cap and xblock * 2 <= x:
        xblock *= 2
    return xblock


def _num_warps_persistent(r: int, tile: int, cls: str, nw_max: int) -> int:
    """Warp count for persistent recipe.

    sum-class:  R<=256 -> nw=1 (small-R fast path: a single warp covers
                the row even when xblock packs multiple rows).
                tile<=2048 -> nw=4 (single load + reduction fits a small
                warp pool, and h2h sweeps put nw=4 within 1-2% of nw=8 at
                tile=2048). tile>2048 -> nw=8: extra warps amortize the
                second load in rms_norm-style kernels at larger tiles.
    norm/multi: ramp tile/1024 clamped [4, 16]. tile/1024 lands on nw=4
                through tile=4096, nw=8 at tile=5120-8192, nw=16 at >=12288.
                This matches per-shape sweeps within ~1% across the
                persistent regime; tile/512 over-warps small tiles and
                tile/2048 under-warps large tiles by similar margins.
    """
    if cls == "sum":
        if r <= 256:
            return 1
        if tile <= 2048:
            return 4
        return 8
    cap = min(nw_max, 16)
    return _next_pow2_clamped(max(4, tile // 1024), 4, cap)


def _recipe_inner_persistent(fact: ReductionFact) -> _Recipe:
    """Persistent INNER recipe.

    See ``_persistent_r_max`` for the eligibility gate. Recipe components:
      xblock = ``_xblock_for_persistent`` (kernel-class dependent)
      reduction_loops = [None] (persistent)
      num_warps = ``_num_warps_persistent`` (tile + class dependent)
    """
    _, _, nw_max, _ = _compute_limits(fact)
    r = fact.static_rnumel
    x = fact.static_xnumel
    assert r is not None and x is not None

    cls = _kernel_class(fact)
    xblock = _xblock_for_persistent(fact, cls)
    tile = xblock * r
    num_warps = _num_warps_persistent(r, tile, cls, nw_max)

    return _Recipe(
        xblock=max(xblock, 1),
        reduction_loops=[None],
        num_warps=num_warps,
    )


def _recipe_inner_looped(fact: ReductionFact) -> _Recipe:
    """Looped INNER recipe.

    r0_block target: largest pow2 <= min(R, MAX_R0_BLOCK).
    Helion invariant: r0_block must be < r_size_hint or the flat-config
    autotune path silently re-decodes the value as persistent.

    num_warps: ``r0_block / 128`` rounded up to power of two, clamped to
    [4, NW_MAX]. The 4-warp floor matches the autotuner's preference for
    the looped regime where math + load pipeline depth both want >=4
    in-flight warps.
    """
    max_r0_block, _, nw_max, _ = _compute_limits(fact)
    r = fact.static_rnumel
    assert r is not None

    pow2_cap = min(_prev_pow2(r), max_r0_block)
    if pow2_cap >= fact.r_size_hint:
        pow2_cap //= 2
    r0_block = max(pow2_cap, 1)

    xblock = 2 if r <= 2048 else 1

    num_warps = _next_pow2_clamped(r0_block // 128, 4, nw_max)

    return _Recipe(
        xblock=xblock,
        reduction_loops=[r0_block],
        num_warps=num_warps,
    )


def _select_recipe(fact: ReductionFact) -> _Recipe:
    """Persistent if ``R <= _persistent_r_max(fact)``, looped otherwise."""
    r = fact.static_rnumel
    assert r is not None
    if r <= _persistent_r_max(fact):
        return _recipe_inner_persistent(fact)
    return _recipe_inner_looped(fact)


# Back-compat aliases used by tests that pin recipe-specific function names.
_recipe_a_inner_persistent = _recipe_inner_persistent
_recipe_b_inner_looped = _recipe_inner_looped


def _emit_config(
    env: CompileEnvironment, fact: ReductionFact, recipe: _Recipe
) -> Config:
    """Build a complete Config: full block_sizes list + reduction_loops + num_warps.

    block_sizes must be emitted as a complete list in env.config_spec.block_sizes
    order. Partial lists fail normalize via _PowerOfTwoBlockIdItem._fill_missing
    -> NotImplementedError -> InvalidConfig (block_id_sequence.py:222-232), and
    the seed gets dropped. For axes the recipe doesn't target, fall back to the
    fragment's default value.
    """
    block_sizes: list[int] = []
    for spec in env.config_spec.block_sizes:
        block_id = spec.block_ids[0]
        if block_id == fact.x_block_id:
            block_sizes.append(_clamp_pow2_to_spec(spec, recipe.xblock))
        else:
            fragment = spec._fragment(env.config_spec)
            block_sizes.append(int(fragment.default()))

    reduction_loops = list(recipe.reduction_loops)

    return Config(
        block_sizes=block_sizes,
        reduction_loops=reduction_loops,
        num_warps=recipe.num_warps,
        num_stages=1,
    )


def _clamp_pow2_to_spec(spec: object, value: int) -> int:
    """Clamp ``value`` to ``[spec.min_size, spec.max_size]`` and round down
    to a power of two. The block-size spec normalizer rejects non-pow2 and
    out-of-range values (BlockSizeSpec._normalize), so the seed must satisfy
    both constraints up front.
    """
    min_size = max(int(getattr(spec, "min_size", 1)), 1)
    max_size = int(getattr(spec, "max_size", value))
    candidate = max(value, min_size)
    candidate = min(candidate, max_size)
    candidate = 1 << (candidate.bit_length() - 1)
    return max(candidate, min_size)


class TritonReductionHeuristic(AutotunerHeuristic):
    """Heuristic seed configs for Triton-backend reductions on H100/sm90.

    v1 scope: INNER-classified compiler-managed reductions with statically
    known shapes. OUTER and DEFAULT hints fall through to default_config(),
    matching pre-PR behavior.

    The recipe constants (MAX_R0_BLOCK = 2048 default / 1024 register-intensive,
    132 SMs, 16-byte TMA floor) are H100-specific. sm100 is deliberately NOT
    in HARDWARE_TARGETS because Hopper vs Blackwell differ in SM count,
    register file, and likely warp-count sweet spots, so the constants would
    need re-tuning before they can be trusted on sm100.
    """

    name = "triton_reduction"
    backend = "triton"
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return False
        facts = env.config_spec.reduction_facts
        if len(facts) != 1:
            return False
        fact = facts[0]
        if fact.hint is not ReductionHint.INNER:
            return False
        return (
            fact.is_rollable
            and fact.static_rnumel is not None
            and fact.x_block_id is not None
            and fact.static_xnumel is not None
        )

    @classmethod
    def get_seed_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config:
        fact = env.config_spec.reduction_facts[0]
        recipe = _select_recipe(fact)
        return _emit_config(env, fact, recipe)
