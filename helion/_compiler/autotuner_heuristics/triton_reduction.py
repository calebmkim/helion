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
    Register pressure also halves NW_MAX (see ``_num_warps`` at
    ``triton_heuristics.py:3087-3096``).
    """
    max_r0_block = 2048
    nw_max = 16 if (fact.static_rnumel or 0) <= 8192 else 32
    nw_min = 2

    register_intensive = (fact.static_xnumel or 0) >= 1024 and (
        fact.num_load + fact.num_reduction
    ) >= 10
    if register_intensive:
        max_r0_block = 1024
        nw_max = max(nw_min, nw_max // 2)

    return max_r0_block, nw_min, nw_max, register_intensive


def _recipe_a_inner_persistent(fact: ReductionFact) -> _Recipe:
    """Recipe A: INNER persistent (R <= 1024).

    Mirrors ``_persistent_reduction_configs`` and INNER pruning at
    ``triton_heuristics.py:4347-4477``.
    """
    _, nw_min, nw_max, _ = _compute_limits(fact)
    r = fact.static_rnumel
    x = fact.static_xnumel
    assert r is not None and x is not None

    if 256 <= r <= 1024 and x // 8 >= 128:
        xblock = min(1024 // max(r, 1), 8)
        num_warps = 1
    else:
        xblock = 1
        for cand in (128, 32, 8, 1):
            if cand <= x and r * cand <= 4096:
                xblock = cand
                break
        tile = xblock * r
        num_warps = _next_pow2_clamped(tile // 128, nw_min, nw_max)

    return _Recipe(
        xblock=max(xblock, 1),
        reduction_loops=[None],
        num_warps=num_warps,
    )


def _recipe_b_inner_looped(fact: ReductionFact) -> _Recipe:
    """Recipe B: INNER looped (R > 1024).

    Mirrors ``_reduction_configs`` / ``contiguous_config`` at
    ``triton_heuristics.py:3956-3960`` and the ``num_warps = r // 128``
    rule on the capped r at ``:3320``.

    The looped r0_block satisfies two Helion-specific invariants:
      1. power of two (``_PowerOfTwoBlockIdItem._normalize`` requires it)
      2. strictly less than ``r_size_hint`` -- otherwise the flat-config
         autotune path silently re-decodes the value as persistent, and
         the seed we benchmark differs from the recipe we wrote.
    """
    max_r0_block, nw_min, nw_max, _ = _compute_limits(fact)
    r = fact.static_rnumel
    assert r is not None

    pow2_cap = min(_prev_pow2(r), max_r0_block)
    if pow2_cap >= fact.r_size_hint:
        pow2_cap //= 2
    r0_block = max(pow2_cap, 1)

    xblock = 2 if r <= 2048 else 1
    num_warps = _next_pow2_clamped(r0_block // 128, nw_min, nw_max)

    return _Recipe(
        xblock=xblock,
        reduction_loops=[r0_block],
        num_warps=num_warps,
    )


def _select_recipe(fact: ReductionFact) -> _Recipe:
    """Pick Recipe A or B. v1 only handles INNER (eligibility filters
    OUTER and DEFAULT)."""
    r = fact.static_rnumel
    assert r is not None
    if r <= 1024:
        return _recipe_a_inner_persistent(fact)
    return _recipe_b_inner_looped(fact)


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
