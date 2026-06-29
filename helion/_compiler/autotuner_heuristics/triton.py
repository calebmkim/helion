from __future__ import annotations

import functools
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast

import torch

from ...autotuner.config_fragment import EnumFragment
from ...autotuner.config_spec import FULL_EXTENT_CATEGORIES
from ...autotuner.config_spec import SIZED_REDUCTION_CATEGORIES
from ...runtime.config import Config
from .common import REDUCTION_TARGET_NAMES
from .common import clamp_block_size_targets
from .common import matches_hardware
from .common import op_name_parts
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_spec import BlockSizeSpec
    from ...autotuner.config_spec import ConfigSpec
    from ...autotuner.config_spec import MatmulFact
    from ...autotuner.config_spec import ReductionFact
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


log = logging.getLogger(__name__)

_B200_MATMUL_HEURISTICS_PATH = Path(__file__).resolve().parent / "matmul_b200.json"


# Heuristic was originally contributed by @umechand-amd
# in https://github.com/pytorch/helion/pull/2357.
class TritonSkinnyGemmHeuristic(AutotunerHeuristic):
    name = "triton_skinny_gemm"
    backend = "triton"
    MIN_ASPECT_RATIO = 8
    BLOCK_TARGETS = (64, 64, 256)
    HARDWARE_TARGETS = (("cuda", "sm90"), ("rocm", "gfx950"))

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return False
        facts = env.config_spec.matmul_facts
        if len(facts) != 1:
            return False
        fact = facts[0]
        if fact.lhs_ndim != 2 or fact.rhs_ndim != 2:
            return False
        if (
            fact.static_m is None
            or fact.static_n is None
            or fact.static_k is None
            or fact.m_block_id is None
            or fact.n_block_id is None
            or fact.k_block_id is None
        ):
            return False
        if max(fact.static_m, fact.static_n) < cls.MIN_ASPECT_RATIO * min(
            fact.static_m, fact.static_n
        ):
            return False
        return (
            clamp_block_size_targets(
                env,
                [
                    (fact.m_block_id, fact.static_m, cls.BLOCK_TARGETS[0]),
                    (fact.n_block_id, fact.static_n, cls.BLOCK_TARGETS[1]),
                    (fact.k_block_id, fact.static_k, cls.BLOCK_TARGETS[2]),
                ],
            )
            is not None
        )

    @classmethod
    def get_seed_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config:
        assert len(env.config_spec.matmul_facts) == 1
        fact = env.config_spec.matmul_facts[0]
        assert fact.static_m is not None
        assert fact.static_n is not None
        assert fact.static_k is not None
        assert fact.m_block_id is not None
        assert fact.n_block_id is not None
        assert fact.k_block_id is not None
        block_sizes = clamp_block_size_targets(
            env,
            [
                (fact.m_block_id, fact.static_m, cls.BLOCK_TARGETS[0]),
                (fact.n_block_id, fact.static_n, cls.BLOCK_TARGETS[1]),
                (fact.k_block_id, fact.static_k, cls.BLOCK_TARGETS[2]),
            ],
        )
        assert block_sizes is not None
        return Config(block_sizes=block_sizes)


def _dtype_family_from_dtype(dtype: object) -> str:
    dtype = str(dtype)
    if "float16" in dtype or "bfloat16" in dtype:
        return "fp16_bf16"
    if "float32" in dtype:
        return "fp32"
    return "other"


def _single_2d_static_matmul_fact(config_spec: ConfigSpec) -> MatmulFact | None:
    facts = config_spec.matmul_facts
    if len(facts) != 1 or len(config_spec.block_sizes) != 3:
        return None
    fact = facts[0]
    if fact.lhs_ndim != 2 or fact.rhs_ndim != 2:
        return None
    if fact.static_m is None or fact.static_n is None or fact.static_k is None:
        return None
    if (fact.m_block_id, fact.n_block_id, fact.k_block_id) != (0, 1, 2):
        return None
    return fact


def _shape_bucket_from_fact(fact: MatmulFact) -> dict[str, object]:
    assert fact.static_m is not None
    assert fact.static_n is not None
    assert fact.static_k is not None
    return {
        "dtype": _dtype_family_from_dtype(fact.lhs_dtype),
        "m_value": fact.static_m,
        "n_value": fact.static_n,
        "k_value": fact.static_k,
    }


@functools.cache
def _heuristic_rules() -> tuple[dict[str, object], ...]:
    with _B200_MATMUL_HEURISTICS_PATH.open(encoding="utf-8") as handle:
        data = cast("dict[str, list[dict[str, object]]]", json.load(handle))
    return tuple(data["rules"])


def _interval_contains(interval: str, value: int) -> bool:
    lower_text, upper_text = interval[1:-1].split(",", maxsplit=1)
    lower = float(lower_text)
    upper = float("inf") if upper_text == "inf" else float(upper_text)

    lower_ok = value >= lower if interval[0] == "[" else value > lower
    upper_ok = value <= upper if interval[-1] == "]" else value < upper
    return lower_ok and upper_ok


def _shape_bucket_matches(
    rule_bucket: dict[str, object],
    query_bucket: dict[str, object],
) -> bool:
    for key, value in rule_bucket.items():
        if key in {"k_bucket", "m_bucket", "n_bucket"}:
            intervals = value if isinstance(value, list) else [value]
            dim_value = cast("int", query_bucket[f"{key[0]}_value"])
            if not any(
                _interval_contains(cast("str", interval), dim_value)
                for interval in intervals
            ):
                return False
            continue
        query_value = query_bucket.get(key)
        values = value if isinstance(value, list) else [value]
        if query_value not in values:
            return False
    return True


def _rules_for_bucket(
    shape_bucket: dict[str, object],
) -> list[dict[str, object]]:
    matches = [
        rule
        for rule in _heuristic_rules()
        if _shape_bucket_matches(
            cast("dict[str, object]", rule["shape_bucket"]),
            shape_bucket,
        )
    ]
    matches.sort(
        key=lambda rule: len(cast("dict[str, object]", rule["shape_bucket"])),
        reverse=True,
    )
    return matches


def _materialize_config(
    raw: dict[str, object],
    *,
    config_spec: ConfigSpec,
) -> Config:
    flat_fields = config_spec._flat_fields()
    supported = {key: value for key, value in raw.items() if key in flat_fields}
    allowed_pid_types = config_spec.allowed_pid_types
    if (
        "pid_type" in supported
        and allowed_pid_types
        and supported["pid_type"] not in allowed_pid_types
    ):
        supported.pop("pid_type")
    config_spec.normalize(supported, _fix_invalid=True)
    config = Config(**cast("dict[str, Any]", supported))
    config_spec._shrink_for_numel_constraints(config)
    return config


def _seed_config_for_bucket(
    shape_bucket: dict[str, object],
    *,
    config_spec: ConfigSpec,
) -> Config | None:
    rules = _rules_for_bucket(shape_bucket)
    if not rules:
        return None

    for rule in rules:
        for template in cast("list[dict[str, object]]", rule["templates"]):
            return _materialize_config(template, config_spec=config_spec)
    return None


def _seed_config_for_config_spec(config_spec: ConfigSpec) -> Config | None:
    fact = _single_2d_static_matmul_fact(config_spec)
    if fact is None:
        return None
    return _seed_config_for_bucket(
        _shape_bucket_from_fact(fact),
        config_spec=config_spec,
    )


class TritonB200MatmulHeuristic(AutotunerHeuristic):
    name = "triton_b200_matmul"
    backend = "triton"
    promote_seed_to_default = True
    HARDWARE_TARGETS = (("cuda", "sm100"),)

    @classmethod
    def is_eligible(
        cls,
        env: CompileEnvironment,
        device_ir: DeviceIR,
    ) -> bool:
        return matches_hardware(env, cls.HARDWARE_TARGETS)

    @classmethod
    def get_seed_config(
        cls,
        env: CompileEnvironment,
        device_ir: DeviceIR,
    ) -> Config | None:
        return _seed_config_for_config_spec(env.config_spec)


class TritonSplitJoinRotateHeuristic(AutotunerHeuristic):
    """Seed all-ones ``block_sizes`` for split/join rotate kernels (rope).

    These kernels load a large untiled inner slab per program, so tiling any
    outer dim past 1 only wastes work and overflows Triton's block-numel cap.
    Detected by ``hl.split`` + ``hl.join`` with no matmul and no reduction op.
    """

    name = "triton_split_join_rotate"
    backend = "triton"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # A GEMM (even fused) is not a rope-style rotate.
        if env.config_spec.matmul_facts:
            return False
        if not env.config_spec.block_sizes:
            return False
        # Local import avoids a circular import at module load
        # (runtime.kernel -> autotuner_heuristics -> helion.language).
        from ...language import join as hl_join
        from ...language import split as hl_split

        saw_split = False
        saw_join = False
        for graph_info in device_ir.graphs:
            for node in graph_info.graph.nodes:
                if node.op != "call_function":
                    continue
                target = node.target
                if target is hl_split:
                    saw_split = True
                elif target is hl_join:
                    saw_join = True
                elif op_name_parts(target) & REDUCTION_TARGET_NAMES:
                    # Fused reduction → not a pure rotate; keep its own tiling.
                    return False
        return saw_split and saw_join

    @classmethod
    def get_seed_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config:
        return Config(block_sizes=[1] * len(env.config_spec.block_sizes))


class TritonPointwiseSeedHeuristic(AutotunerHeuristic):
    """Seed a bandwidth-saturating tile for PURE elementwise/pointwise kernels.

    A pointwise kernel reads its inputs, computes, and writes its outputs with no
    reduction / matmul / loop-carried accumulator — it is BANDWIDTH-bound. The compiler
    default tiles it at ``block_size=32`` (``BlockSizeSpec._fragment``: ``total_ndim <= 2
    and reduction_numel <= 128``), moving only ~10% of HBM (~5-7x slower than a saturating
    tile; measured headroom in ``_lab/pointwise``). This seed sizes the tile to (a) a byte
    budget that saturates HBM and (b) the grid occupancy (size_hint-aware), keyed on the
    derived ``PointwiseElementwiseFact`` (bytes/elem + total numel) — NEVER on the
    activation identity or a dtype literal. Fires on the presence of that fact, which is
    built only on the ABSENCE of the reduction/matmul/accumulator facts (disjointness rule),
    so it never claws a reducing kernel into the pointwise track.
    """

    name = "triton_pointwise"
    backend = "triton"

    # Hill-climbed constants (see _lab/pointwise/NOTEBOOK.md). TILE_BYTES=8192 gives a ~1024-elem
    # tile for traffic-3 (swiglu/geglu/residual_add) and ~2048 for traffic-2 (relu²/bias_gelu) —
    # the robust zone: 1-D kernels are flat across 1-8K, but N-D/traffic-2 kernels (bias_gelu) lose
    # ~5-25% at 4096 vs 2048, so the smaller budget lifts the worst kernel with no 1-D regression.
    TILE_BYTES = 8192  # target HBM bytes moved per tile
    MIN_WAVES = 8  # grid >= num_sm * MIN_WAVES (size_hint-aware grid floor)
    BLOCK_FLOOR = 256  # never regress toward the bs=32 default

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return bool(env.config_spec.pointwise_facts)

    @classmethod
    def get_seed_config(cls, env: CompileEnvironment, device_ir: DeviceIR) -> Config:
        from ...runtime import get_num_sm
        from ...runtime.config import Config as _Config

        spec = env.config_spec
        fact = spec.pointwise_facts[0]
        num_sm = max(1, get_num_sm(env.device))
        bytes_per_elem = max(1, fact.bytes_per_elem)
        # PINNED per-program elements: a grid-pinned (block_size=1) or specialized full-extent
        # axis contributes elements to EVERY program but has no tunable tile, so the distributor
        # (which only sees spec.block_sizes) cannot account for them. Their product is the work
        # each program already does; the tunable dims only need to fill the REMAINING budget.
        # (A grid-pinned block_size=1 axis contributes 1 -> no effect; a specialized group_size
        # contributes its full width.) Read the fixed block size from env.block_sizes (keyed by
        # ALL ids) via an empty Config, NOT size_hint (which is the whole-axis extent, not the
        # per-program tile).
        tunable = set(spec.block_sizes.valid_block_ids())
        pinned_elems = 1
        for info in env.block_sizes:
            if info.block_id in tunable:
                continue
            value = info.from_config(_Config(block_sizes=[]))
            if isinstance(value, (int, torch.SymInt)):
                pinned_elems *= max(1, int(value))
        # Bytes-aware target tile (in elements), DISCOUNTED by the pinned per-program work: a
        # traffic-3 kernel moves 50% more bytes per element than traffic-2, so for the same byte
        # budget it gets a SMALLER element tile.
        target = max(1, (cls.TILE_BYTES // bytes_per_elem) // pinned_elems)
        # size_hint-aware: cap the tile so the grid keeps the SMs busy on small problems (the
        # occupancy cap already uses the pinned-inclusive total_numel, so divide it too).
        max_tile = max(1, fact.total_numel // (num_sm * cls.MIN_WAVES) // pinned_elems)
        target = max(1, min(target, max_tile))
        # Floor (the never-regress-to-bs=32 guard) likewise discounts the pinned work: a program
        # already covering >= BLOCK_FLOOR elements via pinned axes needs no tunable floor.
        floor = max(1, cls.BLOCK_FLOOR // pinned_elems)
        target = max(floor, target)
        # block_sizes is the only non-default field; num_warps/num_stages/pid_type stay at the
        # compiler defaults (4 / 1 / 'flat'), so emitting them would be inert (Gate F). A measured
        # num_warps ramp for large tiles is a BROADEN-queue item.
        return Config(block_sizes=cls._seed_block_sizes(spec, target, floor))

    @staticmethod
    def _pow2_floor(value: int) -> int:
        return 1 << (value.bit_length() - 1) if value >= 1 else 1

    @classmethod
    def _clamp_dim(cls, target: int, bs_spec: BlockSizeSpec, floor: int) -> int:
        # Round DOWN to a pow2 within [floor (and the spec's correctness min), max_size]. max_size
        # is next_pow2(extent), so capping there (rather than pre-capping at the raw size_hint) lets
        # the inner tile COVER a short row in one masked tile (extent 768 -> 1024, not floored to
        # 512). autotuner_min is the AUTOTUNER's search floor, NOT a seed constraint (the reduction
        # and split-join seeds emit block=1 below it), so it is intentionally not applied here.
        cand = cls._pow2_floor(max(1, target))
        cand = max(cand, floor, bs_spec.min_size)
        cand = min(cand, bs_spec.max_size)
        return max(1, cand)

    @classmethod
    def _seed_block_sizes(
        cls, spec: ConfigSpec, target: int, floor: int | None = None
    ) -> list[int]:
        """Distribute the target tile (in elements) across the block dims, INNERMOST first,
        SPILLING the leftover budget outward. The innermost (contiguous, last) dim takes as much
        of the budget as its extent allows (capped at max_size = next_pow2(extent), so a short row
        is covered in one masked tile); whatever budget it cannot absorb spills to the next-outer
        dim, and so on. Because a row-major tensor stores rows contiguously, an ``[R, N]`` tile of
        a short-N tensor is still a coalesced ``R*N`` contiguous block — so a tall-skinny tensor
        (small inner extent, e.g. image RGBA N=4 or a per-head N=64) gets a budget-sized tile (e.g.
        ``[128, 8]``) instead of a starved ``[1, 8]`` that regresses below the bs=32 default. For a
        WIDE row (extent >= budget) the inner absorbs the whole budget and the outer dims stay 1
        (``[1, 1024]``) — the 1-D-flatten-equivalent coalesced run; for a 1-D kernel this is just
        ``[clamp(target)]``."""
        inner_floor = cls.BLOCK_FLOOR if floor is None else floor
        n = len(spec.block_sizes)
        specs = [cast("BlockSizeSpec", spec.block_sizes[i]) for i in range(n)]
        block = [1] * n
        remaining = max(1, target)
        for i in reversed(
            range(n)
        ):  # innermost (contiguous) dim first, spill the rest outward
            dim_floor = inner_floor if i == n - 1 else 1
            block[i] = cls._clamp_dim(remaining, specs[i], dim_floor)
            remaining = max(1, remaining // block[i])
            if remaining <= 1:
                break
        return block


def _triton_reduction_eligible(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    """Gate: the kernel has >= 1 SIZED reduction (PROMPT §6 / §2.6 — the relaxed gate, so a
    MULTI-reduction kernel fires instead of falling to the default) and no ``matmul_facts``
    (GEMMs route to the matmul seeds). Admits both tracks (standard rollable, user-tiled).

    The relaxation from the legacy ``len(reduction_facts) == 1`` is corpus-SAFE: every corpus
    kernel has exactly one reduction fact (verified 452/452), so this only newly admits the
    genuinely multi-reduction kernels (e.g. two sequential rolled reductions). Falls back to the
    legacy ``== 1`` rule if the kernel fact is absent (defensive). A reduction with NO sized
    member (only GRID_TILE / DECLINED) still declines, as today.
    """
    spec = env.config_spec
    if spec.matmul_facts:
        return False
    kf = spec.reduction_kernel_fact
    if kf is None:
        return len(spec.reduction_facts) == 1
    return any(d.category in SIZED_REDUCTION_CATEGORIES for d in kf.reductions)


def _primary_fact(env: CompileEnvironment) -> ReductionFact | None:
    """The PRIMARY ``ReductionFact`` the scalar levers + track discriminator key on, selected by
    the priority order (PROMPT §6.2): the max-extent SIZED reduction over BACKED axes. For a
    single-reduction kernel this is ``reduction_facts[0]`` (byte-identical to the legacy code);
    for a multi-reduction kernel (the relaxed gate) it picks the dominant fact rather than
    blindly taking index 0. Returns None when there is no reduction fact.
    """
    spec = env.config_spec
    facts = spec.reduction_facts
    if not facts:
        return None
    if len(facts) == 1:
        return facts[0]
    from torch._inductor.utils import free_unbacked_symbols

    backed = [
        f
        for f in facts
        if not free_unbacked_symbols(env.block_sizes[f.primary_reduction_block_id].size)
    ]
    pool = backed or list(facts)
    return max(pool, key=lambda f: f.size_hint)


def _is_standard_reduction(spec: ConfigSpec, fact: ReductionFact) -> bool:
    """standard vs user-tiled discriminator, keyed on the Stage-1 TAXONOMY (PROMPT §2.1/§4
    ACCESS): standard iff the primary reduction's category is FULL_SLICE (a rolled rdim OR a
    materialized full-width rdim the roller declined) or FULL_GRID; user-tiled is the USER_TILE
    (rdim-is-a-block_sizes-entry) case. This replaces the legacy ``primary ∉ block_sizes``
    proxy with the positive category — proven equivalent across the 447-cell corpus + 13 probes
    (``_lab/redesign/validate_kernel_fact.py``). Falls back to the legacy proxy only if the
    kernel fact or the primary's descriptor is somehow absent (defensive; never hit on the
    corpus).
    """
    kf = spec.reduction_kernel_fact
    if kf is not None:
        prim = next(
            (
                d
                for d in kf.reductions
                if d.block_id == fact.primary_reduction_block_id
                and d.category in SIZED_REDUCTION_CATEGORIES
            ),
            None,
        )
        if prim is not None:
            return prim.category in FULL_EXTENT_CATEGORIES
    return fact.primary_reduction_block_id not in spec.block_sizes.valid_block_ids()


def _grid_rows(env: CompileEnvironment, m_block_ids: tuple[int, ...]) -> int:
    """Product of the static M-axis (non-reduction grid) extents — the program count the
    reduction launches, the numerator of the occupancy ``grid_rows // num_sm``. Returns 0
    if any extent is not a statically-resolvable size (a dynamic grid has no compile-time
    occupancy, so the occupancy-gated narrow-w1 lever declines).
    """
    grid_rows = 1
    for mbid in m_block_ids:
        size = env.block_sizes[mbid].size
        if not isinstance(size, (int, torch.SymInt)):
            return 0
        grid_rows *= env.size_hint(size)
    return grid_rows


class Cap(NamedTuple):
    """The cap primitive (PROMPT §2.4): a NAMED, faithful, composable budget on a tile size.

    ``applies`` decides whether the cap is in force for the axis being sized (a function of a
    workload property, never a kernel-identity gate); ``value`` is the budget it imposes (a
    function of a workload property + a hardware constant). Sizing any axis is then literally
    ``max(floor, min(c.value() for c in caps if c.applies()))`` — see :func:`size_axis`.

    Properties (§2.4): COMPOSABLE (a new consideration = a new Cap, never a new branch in the
    sizing body), FAITHFUL BY CONSTRUCTION (``value`` reads a property + a constant), and
    INSPECTABLE (``size_axis`` can report which cap bound the result — the per-axis explanation a
    reviewer/witness checks, and the death of the "decision-as-stored-label" anti-pattern).
    """

    name: str
    applies: bool
    value: int


def size_axis(floor: int, caps: list[Cap]) -> tuple[int, str]:
    """Size one axis = ``max(floor, min(applicable caps))`` (PROMPT §2.3/§2.4) — the
    generalization of the pointwise seed's ``max(1, min(target, max_tile))``. Returns
    ``(size, binding_cap_name)`` so the binding cap (the DECISION, computed not stored) is
    inspectable. With no applicable cap the floor binds.
    """
    binding = "floor"
    best = None
    for c in caps:
        if not c.applies:
            continue
        if best is None or c.value < best:
            best = c.value
            binding = c.name
    if best is None or best <= floor:
        return max(1, floor), "floor" if (best is None or best < floor) else binding
    return max(1, best), binding


class _TritonReductionSeedBase(AutotunerHeuristic):
    """Shared base for the two Triton inner-reduction seed heuristics. Both share the
    workload facts (``ReductionFact``), the M_BLOCK-aware reduction-block lever
    (``_reduction_rblock``, from which each track derives ``persistent``), the ``num_warps``
    ramp, eviction provenance, and the block-size builders; the subclasses differ only in
    mapping that decision onto knobs:

    - **standard** (:class:`TritonStandardReductionHeuristic`): Helion rolls the rdim into
      a ``reduction_loops`` loop.
    - **user-tiled** (:class:`TritonUserTiledReductionHeuristic`): the user hand-writes the
      ``hl.tile`` loop, so the reduction axis is a ``block_sizes`` entry (plain user-tiled
      softmax, carried-2-D-tile kl_div/jsd, reduce-then-apply welford).

    Not registered; only the subclasses are.
    """

    backend = "triton"
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    # Occupancy floor for the M-axis widen: keep the post-widen grid >= num_sm * MIN_WAVES so
    # collapsing a fan-out sibling never under-occupies the machine (mirrors the pointwise seed's
    # MIN_WAVES grid floor).
    MIN_WAVES = 8
    # Looped-fallback reduction chunk (pow2) for a row that does not fit the persistent
    # residency budget. ``_reduction_rblock`` shrinks it when a raised M_BLOCK divides the
    # footprint budget below it; at M_BLOCK==1 it is used as-is.
    LOOPED_CHUNK = 16384
    # Per-program byte budget for LOOP-CARRIED 2-D accumulator tiles ([M_BLOCK, R_BLOCK], e.g.
    # kl_div/jsd): caps R_BLOCK via num_carried_2d_tiles. Tightest budget -- resident the whole loop.
    CARRIED_TILE_MAX_BYTES = 16384
    # Per-program persistent byte ceiling. The resident reduction tile is [M_BLOCK, R_BLOCK]
    # in BOTH tracks (the persistent load and the looped accumulator both carry the M_BLOCK
    # dim), so the per-program footprint is ``m_block * r_block * itemsize`` and every cap
    # below divides the budget by M_BLOCK. Above it a wide resident tile spills register/SMEM,
    # so the reduction loops a chunk instead. ~240 KiB, just over H100 SMEM.
    ROW_PERSIST_MAX_BYTES = 245760
    # A FORCED-PERSISTENT reduction (fully-resident grid axis, no rollable r_block) is only
    # cheap to fan a wide parallel sibling out beside when the reduction itself is SMALL; a large
    # one is already a heavy resident tile. Above this rdim extent the m-axis sibling floors
    # rather than widens. The _resident_tile_cap also shrinks the sibling as the rdim grows, so
    # this is primarily an explicit, legible bound (and a backstop against a loose byte budget),
    # not the sole magnitude control. per_token_group's rdim is group_size (64/128) << this.
    PERSISTENT_REDUCTION_MAX = 2048
    # Per-program ELEMENT ceiling for a FULL-WIDTH-output row (stores the whole [M, N] row
    # back): its resident tile is fp32-promoted, so it spills at a WIDTH independent of input
    # dtype, which the byte cap above (input bytes) undercounts 2x for a half-precision row.
    # M_BLOCK-aware; gates only full_width_output rows, steering half-precision full-width
    # standard rows onto the looped path.
    FULL_WIDTH_PERSIST_MAX_ELEMS = 81920
    # Max resident bytes a PERSISTENT reduction body (body_live_tiles full-width tiles) may hold
    # before catastrophic register spill. Only REMOVES persistence from a heavy body, never grows it.
    LIVE_PERSIST_BUDGET = 3 * 245760
    # M-COLLAPSE (grad-parameter reduction, e.g. bias_grad): max rows one CTA reduces in a single
    # in-register inner tile, capped so the reduction tree + [rows, feature] tile don't spill.
    M_COLLAPSE_MAX_CTA = 256
    # M-COLLAPSE inner reduction tile byte budget: a grad-parameter collapse is memory-bound, so
    # the inner [rows, feature] tile wants the SMALLEST footprint (~2-8 rows) for CTA occupancy.
    M_COLLAPSE_TILE_BYTES = 32768
    # No welford "structured-combine floor": welford is memory-bound (profiler-confirmed),
    # so a wide combine tile only spills — register-residency via the reduction footprint cap
    # is what matters. The apply/normalize tile gets the SAME M_BLOCK-aware footprint cap as
    # the reduction tile, NOT a flat per-row cap (which needlessly narrowed the memory-bound
    # apply pass); applied inline in ``_build_block_sizes``.

    # NARROW-row single-warp (occupancy-gated): a narrow reduction extent wants ONE warp (the
    # cross-warp reduction tree is pure overhead; w1 reduces in-register via shuffle). The win
    # inverts past an occupancy ceiling (the SMs saturate), so it is gated on a row-byte cap
    # AND an occupancy cap, both keyed on input_load_itemsize (the HBM-load element width —
    # faithful and dtype-agnostic, unlike the fp32-promoted accumulator itemsize which is 4 at
    # both dtypes):
    #   - row cap: rnumel * input_load_itemsize <= NARROW_W1_MAX_BYTES.
    #   - occ cap: occ * row_bytes <= NARROW_W1_OCC_BYTE_LIMIT (a wider row saturates at lower
    #     occupancy, so the ceiling is on the product, not a flat occ).
    NARROW_W1_MAX_BYTES = 2048
    NARROW_W1_OCC_BYTE_LIMIT = 262144

    @classmethod
    def _carried_tile_r_block_cap(cls, fact: ReductionFact) -> int:
        """Pow2 R_BLOCK ceiling for a reduction carrying loop-resident 2-D accumulator tiles
        (kl_div, jsd): the per-program byte budget ``CARRIED_TILE_MAX_BYTES`` split across the
        accumulator itemsize and the carried-tile count. ``max(1, ..)`` guards a zero itemsize
        or tile count.
        """
        from ..._utils import next_power_of_2 as _np2

        cap = cls.CARRIED_TILE_MAX_BYTES // (
            max(1, fact.itemsize) * max(1, fact.num_carried_2d_tiles)
        )
        return _np2(max(1, cap))

    @classmethod
    def _carried_leading_dims(cls, spec: ConfigSpec) -> set[int]:
        """The block_ids that are the LEADING (M_BLOCK) dim of a loop-carried 2-D accumulator
        ``[M_BLOCK, R_BLOCK]`` -- the grid axis whose widening genuinely MULTIPLIES the carried
        tile, so the carried byte cap applies to it.

        Widening a DIFFERENT (parallel) grid axis does NOT scale the carried tile: a FULL_GRID
        group axis (G) co-resident with a per-token carried sum (the ``fullgrid_plus_carried2d``
        adversarial case) must NOT be capped by the carried footprint, or it floors to 1 and the
        seed regresses ~11x below the default. Read off ``AccumulatorFact.dim_block_ids[0]`` (the
        accumulator's outer/M dim); kl_div/jsd (a single grid axis that IS the carried M dim) are
        unchanged. Empty if no carried 2-D accumulator (the cap then applies to no M axis, a no-op
        guarded by ``num_carried_2d_tiles >= 1`` at the call site too).
        """
        out: set[int] = set()
        for a in spec.accumulator_facts:
            if len(a.dim_block_ids) >= 2 and a.dim_block_ids[0] is not None:
                out.add(a.dim_block_ids[0])
        return out

    @classmethod
    def _carried_m_block_cap(
        cls,
        spec: ConfigSpec,
        fact: ReductionFact,
        m_axis_block_id: int,
        red_values: dict[int, int],
    ) -> int:
        """Pow2 ceiling on an M (grid) axis that is the LEADING dim of one or more loop-carried
        2-D accumulators ``[M_BLOCK, R_BLOCK]`` -- so the carried resident set
        ``M_BLOCK * Σ R_BLOCK * itemsize`` fits ``CARRIED_TILE_MAX_BYTES``.

        Reads the carried tiles from ``accumulator_facts`` and their R_BLOCK from the SIZED
        co-resident reduction (``red_values[reduction_axis]``, else its full extent). This is the
        FAITHFUL footprint even when the kernel's PRIMARY reduction is NOT the carrier (the
        ``carried2d_plus_fullslice`` adversarial case: a full-slice amax primary co-resident with
        a carried sum -- ``fact.num_carried_2d_tiles`` reads 0 off the primary, so the legacy
        cap was DARK and the leading M axis over-widened ~1.36x past default). Returns a huge cap
        (no constraint) when this axis carries no 2-D accumulator -- byte-identical for kernels
        whose carrier IS the primary (kl_div/jsd: handled by the existing r_block_resident path).
        """
        from ..._utils import next_power_of_2 as _np2
        from ..._utils import prev_power_of_2
        from ..compile_environment import CompileEnvironment
        from ..compile_environment import NoCurrentEnvironment

        total_r = 0
        itemsize = 1
        for a in spec.accumulator_facts:
            if (
                len(a.dim_block_ids) >= 2
                and a.dim_block_ids[0] == m_axis_block_id
                and a.dim_block_ids[-1] is not None
            ):
                rdim = a.dim_block_ids[-1]
                # The carried tile's resident R width: the SIZED r_block of its reduction axis,
                # else the padded full extent.
                if rdim in red_values:
                    r_block = red_values[rdim]
                else:
                    try:
                        r_block = _np2(
                            CompileEnvironment.current().block_sizes[rdim].size_hint()
                        )
                    except (NoCurrentEnvironment, IndexError, KeyError):
                        r_block = 1
                total_r += max(1, r_block)
                itemsize = max(itemsize, a.itemsize)
        if total_r <= 0:
            return 1 << 30  # this axis carries no 2-D accumulator -> no carried constraint
        budget = cls.CARRIED_TILE_MAX_BYTES // max(1, total_r * max(1, itemsize))
        return max(1, prev_power_of_2(max(1, budget)))

    @classmethod
    def _num_warps(
        cls, fact: ReductionFact, num_sm: int = 0, grid_rows: int = 0
    ) -> int:
        """Scale num_warps with the reduction extent (pow2, per NumWarpsFragment):
        rnumel <= 1024 -> 4, <= 4096 -> 8, <= 16384 -> 16, > 16384 -> 32. Too few
        under-occupies the SM, too many wastes the reduction tree.

        NARROW-row single-warp refinement at the LOW end (the occupancy-gated lever): a
        narrow row at low/moderate occupancy wants ONE warp (the cross-warp reduction tree
        is pure overhead — see ``NARROW_W1_MAX_BYTES``). Fires only when the row-byte cap AND
        the resident-pressure cap (``occ * row_bytes <= NARROW_W1_OCC_BYTE_LIMIT``) hold; both
        key on ``input_load_itemsize`` (faithful, no dtype-kind branch) and the occ ceiling
        scales DOWN as the row grows (a wider row cliffs at lower occupancy). Needs ``num_sm``
        (0 disables it, e.g. an off-device caller). Disjoint from the wide-row branch below
        (``NARROW_W1_MAX_BYTES`` << the rnumel>16384 region), so the two never interact.
        """
        rnumel = fact.size_hint
        ils = fact.input_load_itemsize
        row_bytes = rnumel * ils
        # NARROW-row single-warp (see NARROW_W1_MAX_BYTES); needs a known device + static grid.
        have_enough_information = num_sm > 0 and ils > 0 and grid_rows > 0
        if have_enough_information:
            occ = grid_rows // num_sm
            if (
                fact.num_carried_2d_tiles
                == 0  # not a carried-2-D-tile reduction (kl_div/jsd)
                and row_bytes <= cls.NARROW_W1_MAX_BYTES
                and occ * row_bytes <= cls.NARROW_W1_OCC_BYTE_LIMIT
            ):
                return 1
        # >16384 (not >=) so a 16384-wide row stays w16, excluding the w32 regression there.
        warps32_min_elems = 16384
        if rnumel > warps32_min_elems:
            return 32
        if rnumel <= 1024:
            return 4
        if rnumel <= 4096:
            return 8
        return 16

    @classmethod
    def _block_floor(cls, bs_spec: BlockSizeSpec) -> int:
        """The smallest valid block size for an entry, used for every non-reduction axis
        the seed does not widen. Prefers one row/program but honors a raised
        ``autotuner_min`` (large-M shapes) rather than emitting an invalid ``block_size=1``.
        """
        return max(1, bs_spec.min_size, bs_spec.autotuner_min)

    @classmethod
    def _resident_tile_cap(
        cls,
        spec: ConfigSpec,
        fact: ReductionFact,
        inner_resident_elems: int = 1,
        r_block_resident: int = 1,
    ) -> int:
        """Max pow2 tile (along one axis) that keeps the resident live set inside the
        per-program register budget: ``tile <= ROW_PERSIST_MAX_BYTES /
        (M_BLOCK * r_block_resident * inner_resident_elems * itemsize)``.

        ``r_block_resident`` is the resident width of the reduction tile co-held with the axis
        being capped (default 1). For the M-AXIS widen branch this is the rdim's resident extent
        (``next_pow2(size_hint)`` when persistent, else the rolled chunk): widening an m-axis by
        ``k`` holds a ``[k, R_BLOCK]`` reduction tile, so the footprint is ``k * R_BLOCK *
        itemsize`` and the cap must divide by ``R_BLOCK`` -- WITHOUT it, a wide-reduction kernel
        (rms_norm, R_BLOCK=16384) is told it can widen M to thousands of rows (the cap returns
        ~32768) and spills catastrophically (~10x), while a tiny-reduction kernel
        (per_token_group, R_BLOCK=128) correctly widens its parallel sibling. Dividing by the
        resident R_BLOCK is what makes the floor-vs-widen decision fall out of the footprint
        arithmetic alone -- no row-vs-sibling axis classification. Default 1 keeps the
        apply/normalize-loop caller (a flat ``[M_BLOCK, tile]`` set) byte-identical.

        ``inner_resident_elems`` is the PRODUCT of the extents of any OTHER axes that are
        co-resident in the same tile alongside the one being capped (default 1). The
        apply/normalize-loop caller holds a flat ``[M_BLOCK, tile]`` set, so it passes 1 (the
        cap is byte-identical to the prior formula). The independent-loop caller's tile is
        ``[M_BLOCK, tile, *pinned_inner]`` (rms_norm_per_block: a ``[1, groups, group_size]``
        per-group quant tile), so it passes the pinned inner product (``group_size``) -- without
        it the cap under-counts the footprint by that factor and would admit a spilling tile.

        CAVEAT (best-estimate, subject to change): ``ROW_PERSIST_MAX_BYTES`` is borrowed from
        the persistent-reduction path -- it was tuned as the resident-row budget for a reduction
        tree, NOT for an independent map/quant loop, whose register pressure profile differs.
        The footprint ACCOUNTING above is faithful (it now counts every co-resident axis), but
        the BUDGET CONSTANT itself is a reused estimate, validated only against rms_norm_per_block
        (whose real shapes stay well under it, so the cap never actually bites there -- it only
        guards a hypothetical wide independent loop). When a real kernel exercises a large
        branch-4 loop that this clamps, re-tune the ceiling for the map case rather than assume
        the reduction budget transfers; a dedicated constant may be warranted then.
        """
        from ..._utils import prev_power_of_2

        m_block = cls._m_block_product(spec, fact)
        denom = (
            m_block
            * max(1, r_block_resident)
            * max(1, inner_resident_elems)
            * max(1, fact.itemsize)
        )
        budget = cls.ROW_PERSIST_MAX_BYTES // max(1, denom)
        return prev_power_of_2(max(1, budget))

    @classmethod
    def _pinned_inner_resident_elems(
        cls, spec: ConfigSpec, fact: ReductionFact, loop_block_id: int
    ) -> int:
        """Product of the extents of the PINNED inner axes co-resident with an independent
        (branch-4) loop tile -- the factor ``_resident_tile_cap`` must divide the budget by so
        the ``[M_BLOCK, loop_tile, *pinned_inner]`` footprint is bounded faithfully.

        A pinned inner axis is one that is NOT tunable (absent from
        ``block_sizes.valid_block_ids()`` -- a ``FixedBlockSizeSource`` materialized at full
        extent), NOT a grid/M axis (not in ``fact.m_block_ids``), NOT a ROLLED reduction (absent
        from ``reduction_loops`` -- see below), and NOT the loop being capped itself. For
        ``rms_norm_per_block`` that is block 3 (``group_size``), the pinned per-group ``amax``
        axis resident inside the per-group quant tile; for a kernel with no such axis the product
        is 1 (the cap reduces to the flat ``[M_BLOCK, tile]`` form).

        ROLLED-REDUCTION EXCLUSION: a rolled reduction axis lives in ``reduction_loops`` (its own
        subgraph, a SEPARATE sequential pass), NOT ``block_sizes``, so it too is non-tunable and
        passed the old filters -- wrongly counting it as co-resident with this loop's tile. A
        kernel that rolls a primary reduction over N and hand-writes an independent loop over K
        would then divide K's residency budget by N's full extent (e.g. 4096), crushing K's tile
        though N is not live during the K pass. Exclude ``reduction_loops`` axes so only genuinely
        co-resident pinned axes (full-extent ``FixedBlockSizeSource`` tiles, e.g. ``group_size``)
        bound the footprint. Byte-identical for the corpus: ``group_size`` is a fixed full-extent
        axis (not rolled, ``reduction_loops`` empty for the user-tiled multi-reduction kernels),
        so it is still counted.

        The pinned extents live in ``env.block_sizes`` (keyed by ALL block_ids incl. non-tunable
        ones); with no active ``CompileEnvironment`` (a unit test exercising the block-size math
        on a bare ``spec``) there is nothing to enumerate, so the product defaults to 1.
        """
        from ..compile_environment import CompileEnvironment
        from ..compile_environment import NoCurrentEnvironment

        try:
            env = CompileEnvironment.current()
        except NoCurrentEnvironment:
            return 1
        tunable = set(spec.block_sizes.valid_block_ids())
        grid = set(fact.m_block_ids)
        rolled = set(spec.reduction_loops.valid_block_ids())
        product = 1
        for info in env.block_sizes:
            bid = info.block_id
            if bid == loop_block_id or bid in tunable or bid in grid or bid in rolled:
                continue
            size = info.size
            if isinstance(size, (int, torch.SymInt)):
                product *= max(1, env.size_hint(size))
        return product

    @classmethod
    def _m_block_cap(cls, fact: ReductionFact) -> int:
        """Upper bound on M_BLOCK (rows/program) for a FULL-WIDTH-output reduction, so a huge-M
        grid-size ``autotuner_min`` raise cannot force an occupancy-starving M_BLOCK on a
        memory-bound held-row reduction. A seed below ``autotuner_min`` (raised only to cap the
        grid) is still valid -- it survives ``normalize``.

        The cap keeps the resident ``[M_BLOCK, rdim]`` live set inside the per-program register
        budget: ``M_BLOCK <= ROW_PERSIST_MAX_BYTES / (rdim * itemsize * body_live_tiles)``.
        Applied only via ``min`` with ``_block_floor`` (only ever LOWERS an over-raised floor)
        and only for full-width output -- streamed/scalar reductions ride occupancy on the
        chunk, not M_BLOCK, so they are uncapped.
        """
        from ..._utils import prev_power_of_2

        if not fact.full_width_output:
            return (
                1 << 30
            )  # no cap: scalar/streamed occupancy rides the reduction chunk
        live = max(1, fact.body_live_tiles)
        isz = max(1, fact.itemsize)
        sh = max(1, fact.size_hint)
        return max(
            1, prev_power_of_2(max(1, cls.ROW_PERSIST_MAX_BYTES // (sh * isz * live)))
        )

    @classmethod
    def _m_axis_occupancy_cap(
        cls, env: CompileEnvironment, fact: ReductionFact, mbid: int
    ) -> int:
        """Largest pow2 tile for M axis ``mbid`` that keeps the POST-widen grid at or above the
        occupancy floor ``num_sm * MIN_WAVES``. Widening collapses ``tile`` programs into one, so
        the grid is ``total_m_program_count / tile``; cap ``tile`` so that stays >= the floor.

        Guards the over-collapse failure mode: a fan-out sibling whose footprint budget would
        permit a very wide tile must not be widened so far the GPU under-occupies. Returns 1 (NO
        widen) when the grid extents are not statically known: a dynamic grid has no compile-time
        occupancy to protect, so the conservative choice is to floor rather than risk an
        under-occupying widen we cannot verify.
        """
        from ..._utils import prev_power_of_2
        from ...runtime import get_num_sm

        grid = _grid_rows(
            env, fact.m_block_ids
        )  # product of static m extents, 0 if dynamic
        if grid <= 0:
            return 1
        num_sm = max(1, get_num_sm(env.device))
        max_tile = grid // (num_sm * cls.MIN_WAVES)
        return max(1, prev_power_of_2(max(1, max_tile)))

    @classmethod
    def _m_axis_block_size(cls, spec: ConfigSpec, mbid: int) -> int:
        """Seed block size (rows/program) for one M-axis (grid) block_id, whether or not it is
        a tunable ``block_sizes`` entry.

        ``fact.m_block_ids`` is the grid axes (``tuple(sorted(grid_ids))``). A grid axis is a
        tunable ``block_sizes`` entry ONLY when its tile is unpinned (``hl.tile(M)``); when the
        user pins it (``hl.tile(M, block_size=1)``) the axis lives solely on the program grid
        and is absent from ``ConfigSpec.block_sizes`` (which carries only tunable tiles, indexed
        positionally to lay out the autotuner ``Config``). Reading the pinned axis from the
        tunable spec raised ``KeyError`` -- the grid-pinned-M idiom every vLLM quant kernel uses.

        Tunable axis: its floored block size (the seed's one-row/program start, honoring a
        raised ``autotuner_min``). Pinned axis: its FIXED block size, read from
        ``env.block_sizes`` (keyed by ALL block_ids incl. grid) -- the real pinned value (1 for
        ``block_size=1``), never a hardcoded fallback.
        """
        if mbid in spec.block_sizes.valid_block_ids():
            m_idx = spec.block_sizes.block_id_to_index(mbid)
            return cls._block_floor(cast("BlockSizeSpec", spec.block_sizes[m_idx]))
        from ...runtime.config import Config as _Config
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        # A grid-pinned axis carries a FixedBlockSizeSource; from_config returns its pinned value
        # independent of the config, so an empty Config resolves it.
        value = env.block_sizes[mbid].from_config(_Config(block_sizes=[]))
        if isinstance(value, (int, torch.SymInt)):
            return max(1, int(value))
        # A non-grid-pinned axis always resolves to a static int/SymInt here (a pinned axis
        # carries a FixedBlockSizeSource with a concrete value), so this is a should-not-happen
        # fallback. Defaulting to 1 silently would reintroduce the floor-to-1 starvation the
        # grid-pinned handling exists to prevent -- warn loudly rather than hide it.
        log.warning(
            "reduction seed: M-axis block_id=%s resolved to a non-static block size %r; "
            "falling back to block_size=1 (this should not happen for a pinned grid axis)",
            mbid,
            value,
        )
        return 1

    @classmethod
    def _m_block_product(cls, spec: ConfigSpec, fact: ReductionFact) -> int:
        """Product of the seed's M-axis (grid) block sizes -- the number of rows each program
        processes (1 unless a huge-M shape raised ``autotuner_min``, capped by ``_m_block_cap``
        for full-width reductions). Shared by the apply-loop stream cap (``_build_block_sizes``)
        and the Band-C combine cap so they read the same M_BLOCK.
        """
        m_block = 1
        cap = cls._m_block_cap(fact)
        for mbid in fact.m_block_ids:
            m_block *= min(cls._m_axis_block_size(spec, mbid), cap)
        return m_block

    @classmethod
    def _build_block_sizes(
        cls,
        env: CompileEnvironment | None,
        spec: ConfigSpec,
        fact: ReductionFact,
        red_values: dict[int, int] | None = None,
        non_reduction_loop_ids: frozenset[int] | set[int] = frozenset(),
    ) -> list[int]:
        """Build the ``block_sizes`` list: each reducing axis present in ``red_values`` (a
        ``block_id -> r_block`` map) gets its sized chunk, each non-reduction loop tile
        (``non_reduction_loop_ids``, disjoint from the reduction axes) gets ``loop_block``,
        every other axis its ``_block_floor``.

        ``red_values`` is empty/``None`` for the STANDARD track (the reduction rides
        ``reduction_loops``, not a ``block_sizes`` entry) -- every axis floors. The USER-TILED
        track passes ``{primary_rdim: r_block}`` for a single-reduction kernel (byte-identical
        to the prior scalar form) and ``{axis: r_block_per_axis, ...}`` for a multi-reduction
        kernel (rms_norm_per_block), sizing EVERY tunable reducing axis instead of flooring the
        non-dominant ones to 1. A PINNED reducing axis (no tunable ``block_sizes`` slot, already
        full-extent resident) is simply absent from ``red_values`` -- the caller only inserts
        axes in ``spec.block_sizes.valid_block_ids()`` (the Error-1 guard).

        The non-reduction loop tile matches the PRIMARY reduction tile — ``red_values[block_id]``
        (user-tiled) or ``next_pow2(size_hint)`` (standard, where ``red_values`` is empty). The
        normalize pass carries no accumulator, so this tile is a pure seed (a sane non-size-1
        start, never a correctness constraint); the autotuner refines it from there.
        """
        from ..._utils import next_power_of_2 as _np2
        from ..._utils import prev_power_of_2 as _pp2

        red_values = red_values or {}
        # The primary (dominant) rdim value drives the normalize-loop tile; absent on the
        # standard track (red_values empty) where it falls back to next_pow2(size_hint).
        primary_red_value = red_values.get(fact.primary_reduction_block_id)

        # Resident width of the reduction tile co-held when an M axis widens, used by the
        # footprint cap. The reduction's resident R_BLOCK reaches the cap by one of three routes,
        # and we must count it EXACTLY once:
        #   - ROLLED (rdim in ``reduction_loops``): persistent at ``next_pow2(size_hint)``.
        #   - USER-TILED (rdim in ``red_values``): resident at its sized r_block (already capped,
        #     e.g. kl_div's carried-tile-capped 4096) -- use that value, NOT the full size_hint.
        #   - PINNED (full-extent ``FixedBlockSizeSource``, e.g. per_token_group's ``group_size``):
        #     ALREADY counted by ``_pinned_inner_resident_elems`` in the cap -> count as 1 here to
        #     avoid double-counting and over-shrinking the sibling.
        if fact.primary_reduction_block_id in spec.reduction_loops.valid_block_ids():
            r_block_resident = _np2(fact.size_hint)
        elif primary_red_value is not None:
            r_block_resident = primary_red_value
        else:
            r_block_resident = 1

        loop_block: int | None = None
        if non_reduction_loop_ids:
            # The apply/normalize tile starts at the reduction tile — primary red_value
            # (user-tiled) or next_pow2(size_hint) (standard, where red_values is empty)...
            loop_block = (
                primary_red_value
                if primary_red_value is not None
                else _np2(fact.size_hint)
            )
            # ...then is clamped to the same M_BLOCK-aware footprint cap as the reduction
            # tile (the apply tile is [M_BLOCK, loop_block] resident, so a wide one spills).
            # Only the Band-C reduce-then-apply kernels (welford, groupnorm) have a normalize
            # loop, so everything else is byte-identical (no non_reduction_loop_ids). The cap
            # clamps an otherwise-spilling apply pass back to register residency; the pass is
            # memory-bound so this is a net win. A flat per-row cap always narrowed it.
            loop_block = min(loop_block, cls._resident_tile_cap(spec, fact))

        out: list[int] = []
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            if bs_spec.block_id in red_values:
                # ANY reducing axis with a tunable slot (the primary rdim, or a secondary
                # reducing axis in a multi-reduction kernel) -- size it to its own r_block.
                out.append(red_values[bs_spec.block_id])
            elif bs_spec.block_id in non_reduction_loop_ids and loop_block is not None:
                out.append(loop_block)
            elif bs_spec.block_id in fact.m_block_ids:
                # Size this tunable grid (M) axis by the FOOTPRINT it would produce, not by any
                # row-vs-sibling classification. Widening this axis by ``k`` holds a
                # ``[k, R_BLOCK]`` resident reduction tile, so ``_resident_tile_cap`` (now keyed on
                # the resident R_BLOCK) is the largest ``k`` that fits the register budget. The
                # floor-vs-widen outcome falls out of the arithmetic:
                #   - a LARGE-R_BLOCK reduction (rms_norm, R_BLOCK=16384) -> cap ~= 1-2 (widening
                #     would spill), so the axis effectively FLOORS, capped further by _m_block_cap;
                #   - a TINY-R_BLOCK reduction (per_token_group / a full-slice per-group amax,
                #     R_BLOCK=128) -> cap is large, so a fan-out sibling WIDENS (the per_token_group
                #     2x win, and the full-slice variant fix) -- regardless of how the kernel is
                #     written (grid-pinned, specialized, or rolled-full-extent).
                # Occupancy guard: never widen so far the POST-widen grid drops below
                # ``num_sm * MIN_WAVES`` (collapsing too many programs would under-occupy).
                # Size the M (grid) axis = max(floor, min(applicable caps)) (PROMPT §2.4 fabric).
                # Each cap is a named function of a faithful property; the binding cap IS the
                # widen-vs-floor decision (computed, not a stored label):
                #   - RESIDENT-TILE (byte): widening M by k holds a [k, R_BLOCK] reduction tile,
                #     so the register-budget cap divides by the resident R_BLOCK -- a large-R_BLOCK
                #     reduction (rms_norm) caps to ~1 (FLOORS), a tiny-R_BLOCK one (per_token_group)
                #     caps large (the parallel sibling WIDENS). No row-vs-sibling classification.
                #   - CARRIED-2D (byte): a kl_div/jsd [M_BLOCK, R_BLOCK] accumulator is live the
                #     whole loop, so widening multiplies it against the tight carried budget.
                #   - OCCUPANCY: keep the post-widen grid >= num_sm * MIN_WAVES (no device -> 1).
                #   - EXTENT: never exceed the axis extent.
                #   - M_BLOCK register cap (full-width-output rows).
                inner = cls._pinned_inner_resident_elems(spec, fact, bs_spec.block_id)
                # Does widening THIS M axis co-hold the reduction tile (so its resident R_BLOCK
                # multiplies the footprint)? For a CARRIED-2D reduction the carried
                # ``[M_BLOCK, R_BLOCK]`` tile is pinned to its leading (M) axis ONLY -- a parallel
                # grid axis (a co-resident FULL_GRID group G, not the carried token axis) does
                # NOT co-hold it, so it must be sized with ``r_block_resident=1`` (else the
                # ``_resident_tile_cap`` divides by R_BLOCK=4096 and floors G to 1: the
                # fullgrid_plus_carried2d ~11x regression). A NON-carried reduction (rms_norm,
                # per_token_group) holds a ``[k, R]`` tile on EVERY grid row, so every M axis
                # co-holds it -- keep the full ``r_block_resident`` (byte-identical for the corpus).
                is_carried = fact.num_carried_2d_tiles >= 1
                axis_co_holds = (not is_carried) or (
                    bs_spec.block_id in cls._carried_leading_dims(spec)
                )
                axis_r_block_resident = r_block_resident if axis_co_holds else 1
                carried_cap = max(
                    1,
                    cls.CARRIED_TILE_MAX_BYTES
                    // (
                        max(1, r_block_resident)
                        * max(1, fact.itemsize)
                        * max(1, fact.num_carried_2d_tiles)
                    ),
                )
                # The carried-2D byte cap likewise applies to this M axis ONLY if widening it
                # genuinely MULTIPLIES the carried tile -- i.e. this axis co-holds it (the leading
                # dim). kl_div/jsd (single grid axis that IS the carried M dim) are unchanged.
                carried_applies = is_carried and axis_co_holds
                m_caps = [
                    Cap(
                        "resident_tile",
                        True,
                        cls._resident_tile_cap(
                            spec, fact, inner, r_block_resident=axis_r_block_resident
                        ),
                    ),
                    Cap("carried_2d", carried_applies, _pp2(carried_cap)),
                    # CARRIED-M cap: when THIS axis is the leading dim of a co-resident carried
                    # 2-D accumulator, bound it so M_BLOCK * Σ(carried R_BLOCK) * itemsize fits the
                    # carried byte budget -- faithful EVEN when the primary reduction is not the
                    # carrier (carried2d_plus_fullslice: a full-slice amax primary, num_carried_2d
                    # reads 0 off it, so the cap above is dark and T over-widens ~1.36x past
                    # default). Returns a huge value (no constraint) for a non-carrying axis.
                    Cap(
                        "carried_m",
                        True,
                        cls._carried_m_block_cap(
                            spec, fact, bs_spec.block_id, red_values
                        ),
                    ),
                    Cap(
                        "occupancy",
                        True,
                        cls._m_axis_occupancy_cap(env, fact, bs_spec.block_id)
                        if env is not None
                        else 1,
                    ),
                    Cap("extent", True, _np2(bs_spec.size_hint)),
                    Cap("m_block_register", True, cls._m_block_cap(fact)),
                ]
                widened, _binding = size_axis(cls._block_floor(bs_spec), m_caps)
                out.append(widened)
            else:
                # An INDEPENDENT non-grid tunable loop: by elimination, not a reducing axis
                # (not in ``red_values``), not an extent-matching apply/normalize loop (not in
                # ``non_reduction_loop_ids``), and not the grid-M axis (not in ``m_block_ids``,
                # which has its own branch above). The ONLY remaining shape is a standalone
                # tiled loop at its own extent -- ``rms_norm_per_block``'s ``groups_per_row``
                # pass is the canonical case. Flooring it to 1 serialized that pass (the
                # [..., 1] catastrophe); widen to its own extent (read from the passed-in
                # ``bs_spec``, never the global env), capped by the SAME resident-byte budget as
                # the apply/normalize tile so a huge loop cannot spill. Flooring was only ever
                # the right default to catch the grid axis, which is now handled above -- so
                # widening, not ``_block_floor``, is the principled catch-all. NO existing
                # reduction kernel reaches this branch (their only non-reducing/non-apply axis
                # IS the grid-M axis), so it is a no-op for the 23-kernel corpus and changes
                # only the new block-quant kernels. The cap divides the budget by the PINNED
                # inner axes co-resident in this loop's tile (rms_norm_per_block: ``group_size``),
                # so the ``[M_BLOCK, loop_tile, *pinned_inner]`` footprint is bounded faithfully
                # -- a flat ``[M_BLOCK, tile]`` cap would under-count and admit a spilling tile.
                inner = cls._pinned_inner_resident_elems(spec, fact, bs_spec.block_id)
                own = _np2(bs_spec.size_hint)
                out.append(max(1, min(own, cls._resident_tile_cap(spec, fact, inner))))
        return out

    @classmethod
    def _eviction_policies(
        cls,
        env: CompileEnvironment,
        kind: str,
        reread_slot: int | None = None,
    ) -> list[str] | None:
        """``load_eviction_policies`` list (spec length), keyed on per-load residency;
        None leaves the autotuner default.

        - ``"stream"`` — single streamed input (``num_load == 1``: sum, long_sum), read
          once: every load -> ``'first'`` (frees L2).
        - ``"reread"`` — the row is re-read across passes: its first load -> ``'last'``
          (L2-resident), rest -> ``'first'``. ``reread_slot`` is that load's actual slot,
          read directly from ``ReductionFact.reread_eviction_index`` (the re-read load's
          ``MemoryOpFact.eviction_index``), not guessed or re-walked per config.

        Other kinds leave the default until a per-slot win is confirmed.
        """
        n = env.config_spec.load_eviction_policies.length
        if n <= 0:
            return None
        if kind == "stream":
            return ["first"] * n
        if kind == "reread":
            if reread_slot is None or not 0 <= reread_slot < n:
                return None
            policy = ["first"] * n
            policy[reread_slot] = "last"
            return policy
        return None

    @classmethod
    def _reduction_rblock(
        cls,
        env: CompileEnvironment,
        fact: ReductionFact,
        m_block: int,
        footprint_factor: int = 1,
        rnumel: int | None = None,
    ) -> tuple[int, bool]:
        """The reduction-axis chunk (pow2) AND the persistent verdict, decided together in one
        budgeted formula and shared by both tracks. Returns ``(r_block, persistent)``.

        ``rnumel`` is the reduction-axis extent to size against; it defaults to
        ``fact.size_hint`` (the PRIMARY rdim) so single-reduction callers are unchanged. A
        multi-reduction kernel (``rms_norm_per_block``-class) calls this once per TUNABLE
        secondary reducing axis in ``fact.secondary_reduction_block_ids``, passing that axis's own
        ``env.block_sizes[b].size_hint()`` so each axis is chunked against its OWN extent (the
        passes are sequential / not co-resident, so a secondary axis is sized independently --
        see the per-axis ``red_values`` construction in each ``get_seed_config``).

        ``footprint_factor`` = how many resident rdim-shaped tiles one program holds live at
        the peak (1 = a single result tile). It bounds the decision two ways:
        - PERSISTENT additionally requires (besides ``row_reread`` and no carried 2-D tile) the
          single resident tile to fit ``ROW_PERSIST_MAX_BYTES`` AND the full
          ``footprint_factor``-tile resident set to fit ``LIVE_PERSIST_BUDGET`` (the multi-tile
          spill ceiling). This liveness term only ever REMOVES persistence from a heavy body.
        - the LOOPED chunk is shrunk by ``footprint_factor`` so a heavy body gets a smaller
          chunk, keeping the looped resident set inside the register budget.

        The standard track passes ``footprint_factor=body_live_tiles``; the user-tiled track
        keeps the default ``1`` (where ``LIVE_PERSIST_BUDGET`` collapses to the base byte cap,
        a no-op). The carried-2-D-tile cap (kl_div/jsd) is folded into the chunk decision.

        No welford "structured-combine floor": register-residency via the footprint cap is what
        matters; welford is memory-bound and a wide combine tile only spills.
        """
        from ..._utils import next_power_of_2 as _np2
        from ..._utils import prev_power_of_2

        # extent: the axis being sized. Defaults to the primary rdim (fact.size_hint) so every
        # single-reduction caller is byte-identical; a multi-reduction caller passes a secondary
        # axis's own extent.
        extent = fact.size_hint if rnumel is None else rnumel
        rdim = _np2(extent)
        itemsize = max(1, fact.itemsize)
        m = max(1, m_block)
        ff = max(1, footprint_factor)
        # PERSISTENCE (the floor that RAISES the chunk to the full extent, PROMPT §2.3 #4): hold
        # the full extent one-shot iff looping would RE-READ the row (``row_reread`` -- a
        # full-width apply or a second reduction needing the first's result; cross_entropy's
        # logsumexp which full_width_output misses) AND no carried 2-D tile (kl_div/jsd are too
        # heavy to hold) AND the resident set clears every per-program ceiling: the element
        # compile limit, the single-tile byte cap, and the ff-tile live-set cap (the liveness
        # ceiling only ever REMOVES persistence from a heavy body, never grants it).
        element_cap = env.backend.max_tensor_numel
        extent_held = (
            fact.row_reread
            and fact.num_carried_2d_tiles == 0
            and (element_cap is None or extent <= element_cap)
            and (m * extent * itemsize <= cls.ROW_PERSIST_MAX_BYTES)
            and (ff * m * extent * itemsize <= cls.LIVE_PERSIST_BUDGET)
        )
        if extent_held:
            return rdim, True
        # Not held -> stream at the chunk = max(1, min(applicable caps)) (PROMPT §2.4 fabric):
        #   - LOOPED chunk ceiling (occupancy-optimal; a capacity-sized chunk would re-read AND
        #     lower occupancy);
        #   - the M_BLOCK/liveness-shrunk BYTE budget (a heavy body / wide grid gets a smaller
        #     chunk so the looped resident set fits the register budget);
        #   - the CARRIED-2D byte cap (kl_div/jsd hold an [M_BLOCK, R_BLOCK] accumulator the whole
        #     loop -- applies only when num_carried_2d_tiles >= 1);
        #   - the EXTENT cap (never size past the padded extent).
        byte_budget = cls.ROW_PERSIST_MAX_BYTES // (m * itemsize * ff)
        caps = [
            Cap("looped_chunk", True, cls.LOOPED_CHUNK),
            Cap("byte_budget", True, prev_power_of_2(byte_budget)),
            Cap(
                "carried_2d",
                fact.num_carried_2d_tiles >= 1,
                cls._carried_tile_r_block_cap(fact),
            ),
            Cap("extent", True, rdim),
        ]
        r_block, _binding = size_axis(1, caps)
        # `persistent` is READ OFF the final chunk -- held iff the chunk reached the extent.
        return r_block, r_block >= rdim

    @classmethod
    def _secondary_red_values(
        cls,
        env: CompileEnvironment,
        spec: ConfigSpec,
        fact: ReductionFact,
        m_block: int,
        exclude_block_id: int | None,
    ) -> dict[int, int]:
        """Per-axis ``block_id -> r_block`` for every TUNABLE reduction the kernel sizes besides
        ``exclude_block_id`` (the axis the caller seeds separately).

        Driven by the Stage-1 ``ReductionKernelFact`` (PROMPT §2.3): the sized descriptors
        (FULL_SLICE / FULL_GRID / USER_TILE) that have a tunable ``block_sizes`` slot. A
        GRID_TILE (grid-parallelized partial) is NOT sized as a reduction (it stays a grid row),
        so it is absent -- the faithful fix to the legacy
        ``secondary_reduction_block_ids ∩ block_sizes`` which wrongly included it. Each axis is
        chunked against its OWN extent (sequential / not co-resident). Falls back to the legacy
        ``secondary_reduction_block_ids`` only if the kernel fact is absent (defensive).
        """
        valid = spec.block_sizes.valid_block_ids()
        kf = spec.reduction_kernel_fact
        if kf is not None:
            sized_bids = {
                d.block_id
                for d in kf.reductions
                if d.category in SIZED_REDUCTION_CATEGORIES
            }
            bids = [
                b
                for b in sized_bids
                if b in valid and b != exclude_block_id
            ]
        else:
            bids = [
                b for b in fact.secondary_reduction_block_ids if b in valid
            ]
        red_values: dict[int, int] = {}
        for bid in sorted(bids):
            axis_r_block, _ = cls._reduction_rblock(
                env, fact, m_block, rnumel=env.block_sizes[bid].size_hint()
            )
            red_values[bid] = axis_r_block
        return red_values

    @classmethod
    def _grad_collapse_group(
        cls, spec: ConfigSpec, device_ir: DeviceIR
    ) -> tuple[int, ...] | None:
        """Detect the grad-parameter M-collapse SHAPE from the Stage-1 taxonomy (PROMPT §6 Q4):
        a co-residency group holding a FULL-EXTENT feature reduction (the ``.mean(-1)`` /
        feature ``.sum`` materialized over N) CO-RESIDENT with a NON-full-extent inner re-tile
        (the cross-row grad accumulation over the inner-M range) — the norm-backward family
        (rms/layer/instance/group bwd), NO kernel identity.

        Returns the inner re-tile block_ids (the axes the byte-capped inner tile sizes) when the
        shape holds, else ``None``. This is the faithful replacement for the
        ``per_feature_accumulator`` recognizer: "a full-slice reduction co-resident with a
        partial grid-tile / user-tile reduction," read off ``category`` + ``coresidency_groups``.

        P2: built + validated to reproduce the override's inner-tile-id set; the override still
        gates on ``fact.per_feature_accumulator`` (P3 swaps to this).
        """
        kf = spec.reduction_kernel_fact
        if kf is None:
            return None
        grid_ids = {b for bids in device_ir.grid_block_ids for b in bids}
        valid = set(spec.block_sizes.valid_block_ids())
        for g in kf.coresidency_groups:
            descs = [kf.reductions[i] for i in g.descriptor_indices]
            has_full_extent = any(
                d.category in FULL_EXTENT_CATEGORIES for d in descs
            )
            # the inner re-tile: a co-resident reduction that is NOT full-extent (a partial
            # grid-tile or an inner user-tile), with a tunable slot, not a grid axis. EXCLUDE a
            # CARRIED-2D reduction: a [M_BLOCK, R_BLOCK] loop-carried accumulator is an ordinary
            # user-tiled reduction (its own carried byte cap sizes it), NOT a grad-collapse
            # cross-row re-tile. Without this exclusion the collapse MISFIRES on a full-slice +
            # carried-2D co-resident kernel (carried2d_plus_fullslice), overriding the correct
            # carried-cap config and regressing ~1.36x. The norm-bwd inner re-tile (the
            # `.sum(0)` grad accum) is NOT carried_2d, so they are unaffected.
            inner = [
                d.block_id
                for d in descs
                if d.category not in FULL_EXTENT_CATEGORIES
                and not d.carried_2d
                and d.block_id in valid
                and d.block_id not in grid_ids
            ]
            if has_full_extent and inner:
                return tuple(sorted(inner))
        return None

    # Warp ceiling for a primary reduction CO-RESIDENT with another sized reduction. A
    # co-resident multi-reduction program already does heavy per-CTA work (two resident tiles),
    # so the extent-keyed warp ramp -- tuned for a SINGLE reduction row -- over-provisions warps;
    # measured w8->w4 wins ~1.1-1.75x on p2 (FULL_SLICE+GRID_TILE) and p8 (FULL_SLICE+FULL_GRID).
    CORESIDENT_MAX_WARPS = 4

    @classmethod
    def _coresident_with_other_sized(
        cls, spec: ConfigSpec, primary_block_id: int
    ) -> bool:
        """True iff the primary reduction shares a co-residency group with ANOTHER reduction of
        ANY category (PROMPT §2.2/§6.2.1) -- a second resident reduction tile makes the program
        heavy regardless of whether that second reduction is *sized as* a reduction (a GRID_TILE
        cross-row accum still costs residency, p2). Corpus-safe: every corpus kernel with a
        multi-descriptor group routes through the grad-collapse path (which sets num_warps
        itself), so this only caps the off-corpus multi-reduction kernels' warps. Read off the
        kernel fact; False if absent.
        """
        kf = spec.reduction_kernel_fact
        if kf is None:
            return False
        for g in kf.coresidency_groups:
            if any(
                kf.reductions[i].block_id == primary_block_id
                for i in g.descriptor_indices
            ):
                return len(g.descriptor_indices) > 1
        return False

    @classmethod
    def _m_collapse_grid_block(
        cls, env: CompileEnvironment, fact: ReductionFact, cap: int | None = None
    ) -> int:
        """Occupancy-sized grid M block for a grad-parameter M-collapse (rms/ln/instance/group
        backward on the standard track; bias_grad/dyt on the user-tiled track). The
        grad-parameter (``grad_weight[N]`` / ``grad_bias[N]``) is summed across the grid rows
        into a per-CTA partial finalized by a cross-CTA ``sum(0)``; with the block floored to 1
        that is a grid-wide M-way collapse (one partial/row), so size it to ~one SM wave
        (``next_pow2(grid_rows // num_sm)``) to cut it to ~``num_sm`` partials.

        ``cap`` bounds the block when it ALSO bears the reduction slab (user-tiled pure
        collapse, capped at ``M_COLLAPSE_MAX_CTA``); the standard track leaves it uncapped
        because the resident ``[inner, feature]`` set rides a separate inner re-tile, not this
        block. A dynamic/unbacked grid (``grid_rows == 0``) collapses to 1.
        """
        from ..._utils import next_power_of_2 as _np2
        from ...runtime import get_num_sm

        grid_rows = _grid_rows(env, fact.m_block_ids)
        num_sm = max(1, get_num_sm(env.device))
        block = _np2(max(1, grid_rows // num_sm))
        return max(1, block if cap is None else min(cap, block))

    @classmethod
    def _m_collapse_inner_byte_cap(cls, feat_bytes: int) -> int:
        """Largest pow2 inner reduction tile whose resident ``[inner, feature]`` fp32 set fits
        ``M_COLLAPSE_TILE_BYTES``, given the per-row feature footprint ``feat_bytes``. Both
        M-collapse tracks pass ``fact.feature_footprint * itemsize`` (the PRODUCT of the
        materialized feature axes); for a 2-D norm that product is the single feature axis.
        """
        from ..._utils import next_power_of_2 as _np2

        return max(1, _np2(max(1, cls.M_COLLAPSE_TILE_BYTES // max(1, feat_bytes))))


class TritonStandardReductionHeuristic(_TritonReductionSeedBase):
    """standard (Helion-rolled rdim) inner-reduction seed: Helion rolls the reduction axis
    into a ``reduction_loops`` loop from a single ``.sum(-1)``-style op — sum, long_sum,
    rms_norm, layer_norm, softmax-row, cross_entropy. Triton analog of
    ``CuteReductionTileHeuristic`` (keeps its registry name), deepening the original
    one-row/persistent/``['last']`` seed with the num_warps ramp, persistent-vs-looped,
    and per-slot eviction.

    Gated by ``_triton_reduction_eligible`` (standard track) — broader than upstream
    ``is_canonical_row_reduction`` (also multi-axis rollable rows and raised-``autotuner_min``
    large-M shapes). Off sm90 the H100-tuned levers are unvalidated, so it falls back to
    ``_narrow_seed`` (pre-existing behavior preserved).
    """

    name = "triton_reduction_tile"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _triton_reduction_eligible(env, device_ir):
            return False
        spec = env.config_spec
        fact = _primary_fact(env)
        return fact is not None and _is_standard_reduction(spec, fact)

    @classmethod
    def _narrow_seed(cls, env: CompileEnvironment) -> Config:
        """The upstream conservative standard seed (one row/program, single persistent pass,
        ``['last']`` eviction where supported). A verbatim port used off sm90 so non-sm90
        behavior is unchanged.
        """
        spec = env.config_spec
        seed: dict[str, Any] = {
            "block_sizes": [1],
            "reduction_loops": [None],
        }
        # Emit 'last' only where the backend supports it; backends that restrict
        # eviction to ("",) keep the spec default so the seed stays valid.
        eviction = spec.load_eviction_policies
        if (
            eviction.length
            and isinstance(eviction.inner, EnumFragment)
            and "last" in eviction.inner.choices
        ):
            seed["load_eviction_policies"] = ["last"] * eviction.length
        return Config(**seed)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # Off the H100-validated target: keep the upstream conservative seed.
            return cls._narrow_seed(env)
        from ...runtime import get_num_sm

        spec = env.config_spec
        fact = _primary_fact(env)
        if fact is None:
            return None
        # standard rides persistent-vs-looped on reduction_loops (sized by the shared _reduction_rblock).
        # footprint_factor=body_live_tiles routes a heavy body that would overflow the register file
        # persistent (e.g. fused_linear_jsd) to the looped path instead.
        m_block = cls._m_block_product(spec, fact)
        r_block, persistent = cls._reduction_rblock(
            env,
            fact,
            m_block,
            footprint_factor=fact.body_live_tiles,
        )
        num_warps = cls._num_warps(
            fact, max(1, get_num_sm(env.device)), _grid_rows(env, fact.m_block_ids)
        )
        # CO-RESIDENT multi-reduction cap (PROMPT §6.2.1): when the primary shares a budget with
        # another sized reduction, the program is already heavy -> cap the extent-keyed warp ramp
        # (corpus-safe: such corpus kernels take the grad-collapse path below, which sets warps).
        if cls._coresident_with_other_sized(spec, fact.primary_reduction_block_id):
            num_warps = min(num_warps, cls.CORESIDENT_MAX_WARPS)

        # A standard reduction may be followed by a normalize loop (e.g. `s = x.sum(); out =
        # x/s`); its extra block_sizes tile(s) are sized by _build_block_sizes (matched to
        # the reduction tile). Only a seed (a worse tile costs autotuning time, never
        # correctness), so emit and let the autotuner refine.
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)

        # red_block_id=None: rdim is not a block_sizes entry, so every entry is a grid axis (floored)
        # or a normalize-loop tile. MATERIALIZED rdim (rms/ln/instance bwd, the roller declined to roll
        # it): emit an EMPTY reduction_loops -- already full-width persistent, and a length-1 list would
        # fail normalize against the 0-length spec.
        is_materialized = (
            fact.primary_reduction_block_id
            not in spec.reduction_loops.valid_block_ids()
        )
        reduction_loops: list[int | None]
        if is_materialized:
            reduction_loops = []
        elif len(spec.reduction_loops) <= 1:
            # Single rolled reduction (every corpus kernel): byte-identical to before.
            reduction_loops = [None] if persistent else [r_block]
        else:
            # MULTI rolled reduction (the relaxed gate, e.g. two SEQUENTIAL rolled reductions in
            # separate graphs / co-residency groups). Size EACH rolled spec independently against
            # its OWN extent (sequential -> not co-resident -> each gets the full budget), one
            # ``reduction_loops`` entry per spec in spec order. The primary spec reuses the
            # (r_block, persistent) computed above; the rest are sized from their own extent.
            reduction_loops = []
            for rl_spec in spec.reduction_loops:
                bid = rl_spec.block_ids[0]
                if bid == fact.primary_reduction_block_id:
                    reduction_loops.append(None if persistent else r_block)
                else:
                    rb, pers = cls._reduction_rblock(
                        env,
                        fact,
                        m_block,
                        rnumel=env.block_sizes[bid].size_hint(),
                    )
                    reduction_loops.append(None if pers else rb)
        # The PRIMARY rdim rides reduction_loops, NOT a block_sizes entry, so it is never in
        # red_values. But a kernel may ROLL its primary yet hand-write a SECONDARY reduction over a
        # tunable axis (a block_sizes entry, recorded in fact.secondary_reduction_block_ids by
        # build_reduction_facts). Size each such secondary as a reduction (its own r_block via
        # _reduction_rblock) instead of letting it fall to the catch-all generic widen -- the
        # standard-track analogue of the user-tiled multi-reduction sizer. Single-reduction kernels
        # have secondary_reduction_block_ids=() -> red_values stays empty -> every block_sizes axis
        # floors, byte-identical to before. (The primary is never in this set, so no skip needed.)
        # Per-axis r_block for the tunable SECONDARY reductions, driven by the Stage-1 kernel
        # fact (the rolled primary rides reduction_loops, so it is excluded). Single-reduction
        # kernels get {} -> byte-identical.
        red_values = cls._secondary_red_values(
            env, spec, fact, m_block, exclude_block_id=fact.primary_reduction_block_id
        )
        block_sizes = cls._build_block_sizes(
            env,
            spec,
            fact,
            red_values or None,
            non_reduction_loop_ids=non_reduction_loop_ids,
        )
        # Dual-axis grad-parameter M-collapse (rms/ln/instance/group bwd): the grid M block is re-tiled
        # by an inner loop and feeds a per-feature grad accumulator finalized across CTAs.
        # _build_block_sizes floors it to 1 (leaving a grid-wide finalize); size it for occupancy so the
        # finalize shrinks to ~num_sm partials.
        #
        # Keyed on the TAXONOMY (PROMPT §6 Q4), NOT the ``per_feature_accumulator`` recognizer:
        # ``_grad_collapse_group`` returns the inner re-tile ids of a co-residency group holding a
        # full-extent feature reduction co-resident with a non-full-extent inner re-tile -- "a
        # FULL_SLICE reduction co-resident with a partial GRID_TILE/USER_TILE reduction," read off
        # category + coresidency_groups. This replaces BOTH the ``pfa`` gate (Defect #1/#6 the
        # recognizer) AND the subtractive ``inner_tile_ids`` filter (Defect #2 "an axis defined by
        # what it is NOT"). The 9 standard + 8 transfer kernels have no such group -> None -> their
        # seeds stay byte-identical.
        inner_tile_ids = cls._grad_collapse_group(spec, device_ir)
        if inner_tile_ids:
            # Occupancy-size the grid M block (the dominant lever), byte-cap the inner
            # re-tile to the feature footprint, and drop the narrow-w1 warps lever: it keys
            # on rdim extent alone, but the resident tile here is [inner, feature]-wide, so
            # the plain extent ramp (>=4 warps) is faithful. Only instance_norm is affected.
            m_cta = cls._m_collapse_grid_block(env, fact)
            for mbid in fact.m_block_ids:
                # A grid-PINNED M axis (block_size=1) has no tunable Config slot and cannot be
                # raised to an occupancy block -- skip it. Only unpinned grid tiles re-tile.
                if mbid in spec.block_sizes.valid_block_ids():
                    block_sizes[spec.block_sizes.block_id_to_index(mbid)] = m_cta
            inner = cls._m_collapse_inner_byte_cap(
                max(1, fact.feature_footprint) * max(1, fact.itemsize)
            )
            for bid in inner_tile_ids:
                block_sizes[spec.block_sizes.block_id_to_index(bid)] = inner
            num_warps = cls._num_warps(
                fact
            )  # num_sm/grid_rows default 0 -> no narrow-w1
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": 1,
            # 'flat': these reductions are grid-saturated at the M-grid.
            "pid_type": "flat",
        }
        # Eviction: streamed input -> 'first' everywhere; looped re-read -> first load
        # 'last', rest 'first'. PERSISTENT rows are left at default ON PURPOSE (the `not
        # persistent` gate below): a rolled persistent reduction fuses the reduce + apply to a
        # SINGLE HBM load of the row (profiler-confirmed), so a 'last' hint is a no-op and
        # actually regresses wide rows by pinning x and evicting weight/store lines. This is
        # the opposite of the user-tiled track, where softmax_two_pass has two PHYSICAL
        # reduction loops (two loads) so 'last' helps even when persistent.
        evict = None
        if fact.num_load == 1:
            evict = cls._eviction_policies(env, "stream")
        elif fact.row_reread and not persistent:
            # Re-read row's eviction slot read directly from the fact (its load's
            # MemoryOpFact.eviction_index), not a per-config codegen re-walk.
            evict = cls._eviction_policies(env, "reread", fact.reread_eviction_index)
        if evict is not None:
            seed["load_eviction_policies"] = evict
        return Config(**seed)


class TritonUserTiledReductionHeuristic(_TritonReductionSeedBase):
    """user-tiled inner-reduction seed: fires when the user hand-writes the ``hl.tile`` loop
    over the reduction axis (so the rdim is an ordinary ``block_sizes`` entry, e.g.
    ``hl.tile(n, block_size=R_BLOCK)``), which the upstream gate rejects entirely.
    R_BLOCK starts at the shared ``_reduction_rblock`` (M_BLOCK-aware footprint cap), then
    INDEPENDENT band predicates layer on via ``min`` (a kernel gets every cap it matches;
    today's kernels each match exactly one):

    - **plain user-tiled** (softmax_two_pass): no extra cap -- persistent full-pow2 R_BLOCK,
      standard-style reread-eviction for wide looped rows.
    - **carried 2-D tiles** (kl_div, jsd): carry ``[M_BLOCK, R_BLOCK]`` accumulator tiles
      across the loop, so R_BLOCK is capped by ``CARRIED_TILE_MAX_BYTES / (itemsize *
      num_carried_2d_tiles)`` -- folded into the shared ``_reduction_rblock`` decision.
    - **reduce-then-apply** (welford, ``non_reduction_loop_block_ids`` non-empty): no combine
      floor. Its normalize/apply tile starts at the reduction tile and gets the SAME
      M_BLOCK-aware footprint cap; see ``_build_block_sizes``.

    TODO(reductions): as more structured families land, promote each band into its own
    fact-keyed ``AutotunerHeuristic`` subclass rather than growing this method.
    """

    name = "triton_reduction_user_tile"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _triton_reduction_eligible(env, device_ir):
            return False
        spec = env.config_spec
        return not _is_standard_reduction(spec, spec.reduction_facts[0])

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # Off sm90: upstream never fired on user-tiled, so no prior seed to preserve. Decline.
            return None
        from ...runtime import get_num_sm

        spec = env.config_spec
        fact = spec.reduction_facts[0]
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)
        m_block = cls._m_block_product(spec, fact)

        # user-tiled: rdim IS a block_sizes entry (no reduction_loops knob); persistent == R_BLOCK >=
        # next_pow2(N), other axes floored (u0*u1 <= 2**20). The shared lever sizes R_BLOCK from
        # residency (single-tile footprint + the folded-in carried-2-D cap for kl_div/jsd) and returns
        # it directly; _persistent is unused on this track.
        r_block, _persistent = cls._reduction_rblock(env, fact, m_block)
        # M-COLLAPSE (grad-parameter reduction, e.g. bias_grad/dyt): collapse the grid/row axis into a
        # per-feature accumulator, sizing the grid CTA for occupancy instead of T2's floored grid.
        m_collapse_block: int | None = None
        # Faithful signature: per_feature_accumulator -- a loop-carried accumulator over ALL
        # the materialized feature axis (bias_grad/dyt); per-row / 2-D accumulators are excluded.
        is_m_collapse = fact.per_feature_accumulator
        if is_m_collapse:
            # (a) grid CTA -> OCCUPANCY (_m_collapse_grid_block), capped at M_COLLAPSE_MAX_CTA since the
            #     grid block also bears the reduction slab (sum(0) finalize over ~num_sm partials). An
            #     unbacked/AOT grid (grid_rows == 0) falls through to block 1 -- a worse seed, not a bug.
            m_collapse_block = cls._m_collapse_grid_block(
                env, fact, cap=cls.M_COLLAPSE_MAX_CTA
            )
            # (b) inner reduction tile: depends on whether the collapse has PER-ROW WORK,
            #     which ``body_live_tiles`` measures (peak simultaneously-live full-width tiles).
            if fact.body_live_tiles <= 1:
                # PURE collapse (bias_grad: read + sum, ONE resident tile): a big inner tile is
                # cheap and cuts loop overhead, so reduce the whole CTA wave in one slab (bounded
                # by the grid block + 256 cap).
                r_block = m_collapse_block
            else:
                # Collapse WITH per-row work (dyt: full-width grad_x store + tanh intermediates):
                # a big inner tile spills, so byte-cap the resident [inner, feature] footprint
                # tight (~2-8 rows) for occupancy.
                feat_bytes = max(1, fact.feature_footprint) * max(1, fact.itemsize)
                inner_cap = cls._m_collapse_inner_byte_cap(feat_bytes)
                r_block = max(1, min(m_collapse_block, inner_cap))

        num_warps = cls._num_warps(
            fact, max(1, get_num_sm(env.device)), _grid_rows(env, fact.m_block_ids)
        )

        # Per-axis r_block map. The primary rdim always gets the r_block computed above (its
        # m_collapse override included). A multi-reduction kernel (rms_norm_per_block: a tunable
        # 5120-wide RMS sum + a pinned 128-wide group amax) ALSO sizes each SECONDARY reducing
        # axis to its OWN extent, so the non-dominant reduction is no longer floored to 1. Only
        # TUNABLE axes (a block_sizes slot) are inserted -- a PINNED/materialized reducing axis is
        # already full-extent resident and has no slot (the Error-1 guard, mirrored here). The
        # passes are sequential / not co-resident, so each axis is chunked against its own extent
        # (see _reduction_rblock's rnumel). Single-reduction kernels have an empty
        # secondary_reduction_block_ids, so this map is exactly {fact.primary_reduction_block_id:
        # r_block} -- byte-identical. The primary is seeded above and is never in the secondary set.
        red_values: dict[int, int] = {fact.primary_reduction_block_id: r_block}
        # The tunable SECONDARY reductions (driven by the Stage-1 kernel fact), each chunked
        # against its own extent. The primary is seeded above (with its m_collapse override), so
        # exclude it. Single-reduction kernels add nothing -> byte-identical.
        red_values.update(
            cls._secondary_red_values(
                env,
                spec,
                fact,
                m_block,
                exclude_block_id=fact.primary_reduction_block_id,
            )
        )

        block_sizes = cls._build_block_sizes(
            env,
            spec,
            fact,
            red_values,
            non_reduction_loop_ids=non_reduction_loop_ids,
        )
        if m_collapse_block is not None:
            # Raise the grid CTA tile(s) from the floor to the occupancy block (the reduction
            # tile was already set to m_collapse_block via r_block above). A grid-PINNED M axis
            # (block_size=1) has no tunable Config slot and cannot be raised -- skip it.
            for mbid in fact.m_block_ids:
                if mbid in spec.block_sizes.valid_block_ids():
                    idx = spec.block_sizes.block_id_to_index(mbid)
                    block_sizes[idx] = m_collapse_block
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "num_warps": num_warps,
            "num_stages": 1,
            "pid_type": "flat",  # see the standard branch.
        }
        # Reread eviction: keep the re-read row L2-resident ('last' on its load slot) whenever
        # it is re-read — welford (reduce-then-apply across combine + normalize) AND plain
        # user-tiled (softmax_two_pass loads x twice). Applies even when PERSISTENT: the second
        # pass still re-fetches x from HBM (profiler-confirmed), so 'last' cuts that re-read
        # traffic. kl_div/jsd (row_reread=False) unaffected.
        if non_reduction_loop_ids or fact.row_reread:
            # Re-read row's eviction slot read directly from the fact (its load's
            # MemoryOpFact.eviction_index), not a per-config codegen re-walk.
            ev = cls._eviction_policies(env, "reread", fact.reread_eviction_index)
            if ev is not None:
                seed["load_eviction_policies"] = ev
        return Config(**seed)


class TritonMatmulReductionEpilogueHeuristic(AutotunerHeuristic):
    """Seed for a fused matmul + reduction-over-output-axis epilogue (matmul_rms_norm /
    matmul_layernorm / matmul_softmax / matmul_l2_normalize / matmul_sum / ...): a single
    grid loop over M does an inner K-loop ``addmm`` into a register-resident ``[M_BLOCK, N]``
    fp32 accumulator, then reduces over the matmul's N (output) axis on that accumulator. N
    is ``hl.specialize``'d (never tiled), so BOTH the ``[M_BLOCK, N]`` accumulator AND the
    ``[K_BLOCK, N]`` y-operand tile scale with N -> the kernel is SMEM/register-footprint
    bound and the win regime is small N (where a productive tile fits).

    Fires on the composed ``MatmulWithReductionEpilogueFact`` (a MatmulFact + an epilogue
    ReductionFact in one kernel) -- never on a pure matmul or a pure reduction, so those stay
    byte-identical. This sizes M_BLOCK by the resident fp32-accumulator footprint.
    """

    name = "triton_matmul_reduction_epilogue"
    backend = "triton"
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    # The resident [M_BLOCK, N] fp32 accumulator must fit a per-program byte budget; M_BLOCK is the
    # largest pow2 under it, capped at MAX_M_BLOCK (an occupancy/register ceiling). ~128 KiB gives
    # the answer-key tile: M_BLOCK=64 at N<=512, 32 at N=1024, 16 at N=2048 (where the win vanishes).
    ACC_BUDGET_BYTES = 131072
    MAX_M_BLOCK = 64
    # Inner K tile (min 16 by the matmul min_dot_size; normalize clamps to <=K).
    K_BLOCK = 32
    # num_stages: pipeline the K-loop addmm (a matmul knob; the answer key uses 3).
    NUM_STAGES = 3
    # num_warps ramps with the resident accumulator elements (M_BLOCK * N).
    NUM_WARPS_ELEM_BREAK = 16384
    # Staged matmul-operand SMEM budget (sm90/H100 has ~227 KiB/SM). The [K_BLOCK, N]
    # y-operand x num_stages must fit this; past it the shipped [.,32]/st3 OOMs.
    # Calibrated to the measured feasibility boundary (KB=32/st3 fits N<=1024 bf16 /
    # N<=512 fp32; KB=16/st3 fits N<=2048 / N<=1024). The byte-cap (get_seed_config)
    # drops K_BLOCK 32->16 FIRST -- it halves the staged bytes AND avoids the measured
    # non-monotonic KB=32 ptxas cliffs -- keeping full stages; only past KB=16/st3 does
    # it drop num_stages (cliff-free once KB=16).
    SMEM_STAGED_BUDGET_BYTES = 196608  # 192 KiB

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return False
        # Resident-only: fire when the composed fact's N axis is hl.specialize'd
        # (n_block_id is None). The looped/tiled-N shape is left to the default config.
        facts = env.config_spec.matmul_reduction_epilogue_facts
        return len(facts) == 1 and facts[0].matmul.n_block_id is None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return None
        from ..._utils import prev_power_of_2

        spec = env.config_spec
        fact = spec.matmul_reduction_epilogue_facts[0]
        n = max(1, fact.n_extent)
        # Per-program row ceiling: MAX_M_BLOCK at 2 bytes (bf16/fp16 tensor core),
        # scaled DOWN as the input dtype widens (fp32 = ~2x regs/elem -> //2 -> 32).
        # The factor only lowers the ceiling; MAX_M_BLOCK is the hard occupancy cap,
        # so a 1-byte dtype (fp8) is pinned to it by min(), not pushed above it.
        input_itemsize = fact.matmul.lhs_dtype.itemsize
        max_m = max(1, min(cls.MAX_M_BLOCK, cls.MAX_M_BLOCK * 2 // input_itemsize))

        # Resident N (hl.specialize'd, n_block_id is None -- guaranteed by is_eligible):
        # M_BLOCK = largest pow2 [M_BLOCK, N] fp32 accumulator under the ACC budget, capped
        # at max_m. The staged [K_BLOCK, N] operand is bounded separately by the SMEM
        # byte-cap below, which is what sets the feasible-N ceiling.
        m_block = max(
            1, min(max_m, prev_power_of_2(max(1, cls.ACC_BUDGET_BYTES // (n * 4))))
        )
        num_warps = 4 if m_block * n <= cls.NUM_WARPS_ELEM_BREAK else 8

        # K_BLOCK + num_stages via a priority-ordered footprint byte-cap. The staged
        # [K_BLOCK, N] y-operand (x num_stages) must fit SMEM; in the shipped small-N
        # regime [K_BLOCK=32, num_stages=3] fits, but past it (large N) it overflows.
        # Reduce K_BLOCK 32->16 FIRST -- it halves the staged bytes AND avoids the measured
        # non-monotonic K_BLOCK=32 ptxas cliffs, while keeping full stages -- then, only if
        # [16, st=3] still overflows (very large N), drop num_stages (cliff-free once
        # K_BLOCK=16). This EXTENDS the feasible N (KB=32/st3 to N<=1024 bf16, then KB=16/st3
        # to N<=2048) instead of OOMing into the bad default; small-N stays byte-identical.
        k_hint = next(
            (
                cast("BlockSizeSpec", spec.block_sizes[i]).size_hint
                for i in range(len(spec.block_sizes))
                if cast("BlockSizeSpec", spec.block_sizes[i]).block_id
                == fact.k_block_id
            ),
            cls.K_BLOCK,
        )
        k_block = min(cls.K_BLOCK, k_hint)
        num_stages = cls.NUM_STAGES
        if num_stages * k_block * n * input_itemsize > cls.SMEM_STAGED_BUDGET_BYTES:
            k_block = min(k_block, 16)
            while (
                num_stages > 1
                and num_stages * k_block * n * input_itemsize
                > cls.SMEM_STAGED_BUDGET_BYTES
            ):
                num_stages -= 1

        block_sizes: list[int] = []
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            bid = bs_spec.block_id
            if bid == fact.m_block_id:
                block_sizes.append(max(bs_spec.min_size, m_block))
            elif bid == fact.k_block_id:
                block_sizes.append(
                    max(bs_spec.min_size, min(k_block, bs_spec.size_hint))
                )
            else:
                block_sizes.append(max(1, bs_spec.min_size, bs_spec.autotuner_min))

        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            # The epilogue reduction is materialized on the resident accumulator, so
            # there is no reduction_loops knob to set.
            "reduction_loops": [],
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        return Config(**seed)
