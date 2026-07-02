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
from ...autotuner.config_spec import ReductionCategory
from ...runtime.config import Config
from .common import REDUCTION_TARGET_NAMES
from .common import clamp_block_size_targets
from .common import matches_hardware
from .common import op_name_parts
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...autotuner.config_spec import BlockSizeSpec
    from ...autotuner.config_spec import ConfigSpec
    from ...autotuner.config_spec import MatmulFact
    from ...autotuner.config_spec import ReductionDescriptor
    from ...autotuner.config_spec import ReductionKernelFact
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
    genuinely multi-reduction kernels (e.g. two sequential rolled reductions). A reduction with NO
    sized member (only GRID_TILE / DECLINED) still declines, as today.

    Keyed PURELY on the Stage-1 kernel fact -- no ``ReductionFact`` read. ``build_reduction_kernel_fact``
    runs unconditionally on every live compile, so the fact is absent only for a bare-spec unit
    test or a kernel with genuinely no reduction; both correctly decline (no sized reduction).
    """
    spec = env.config_spec
    if spec.matmul_facts:
        return False
    kf = spec.reduction_kernel_fact
    if kf is None:
        return False
    return any(d.category in SIZED_REDUCTION_CATEGORIES for d in kf.reductions)


def _primary_descriptor_selected(env: CompileEnvironment) -> ReductionDescriptor | None:
    """Primary reduction descriptor: max ROW-BYTES (size_hint*input_load_itemsize) over BACKED
    sized descriptors (§6.2.1). NOT category tier-order (would flip rms_norm_per_block_quant).
    None if no sized reduction / no kernel fact present. Zero-divergence from the legacy
    max-size_hint primary across the corpus.

    This is the single Stage-1 source the reduction tracks read every scalar lever off
    (num_warps / persistence / footprint caps) -- the descriptor IS the primary, so no
    ``ReductionFact`` is consulted (the legacy flat fact stays built for the matmul-epilogue +
    the eligibility gate, but is unread here). On the live compile path the kernel fact is
    never absent when there is a sized reduction (``build_reduction_kernel_fact`` runs
    unconditionally), so ``None`` is the TEST-ONLY / no-sized-reduction case.
    """
    from torch._inductor.utils import free_unbacked_symbols

    kf = env.config_spec.reduction_kernel_fact
    if kf is None:
        return None
    sized = [d for d in kf.reductions if d.category in SIZED_REDUCTION_CATEGORIES]
    if not sized:
        return None
    backed = [
        d for d in sized if not free_unbacked_symbols(env.block_sizes[d.block_id].size)
    ]
    pool = backed or sized
    return max(
        pool, key=lambda d: (d.size_hint * max(1, d.input_load_itemsize), d.size_hint)
    )


def _is_standard_reduction(pd: ReductionDescriptor) -> bool:
    """standard vs user-tiled discriminator, keyed on the Stage-1 TAXONOMY (PROMPT §2.1/§4
    ACCESS): standard iff the primary reduction's category is FULL_SLICE (a rolled rdim OR a
    materialized full-width rdim the roller declined) or FULL_GRID; user-tiled is the USER_TILE
    (rdim-is-a-block_sizes-entry) case. Reads the primary descriptor's category directly --
    proven equivalent to the legacy ``primary ∉ block_sizes`` proxy across the 447-cell corpus +
    13 probes (``_lab/redesign/validate_kernel_fact.py``).
    """
    return pd.category in FULL_EXTENT_CATEGORIES


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


class _TileAllocation(NamedTuple):
    """The result of :meth:`_TritonReductionSeedBase.size_reduction_tiles` — the ONE
    per-co-residency-group BUDGET allocation (PROMPT §2.3/§6.2.1), the single source of every
    tile size the seed emits.

    There is ONE allocator and ONE budget. Per co-residency group it forms a register/byte
    capacity, then seats axes in priority order (full-extent reductions → user-tile reductions →
    grid-tile reductions → the grid-M rows), each taking first crack then FLOORED by the budget
    that remains after everything already seated. Earlier groups' assignments are held FIXED as
    inputs to later groups; the non-reduction loops are sized LAST against the remaining headroom.
    Floor-vs-resident and collapse-vs-widen are budget OUTCOMES (no recognizer, no cdiv branch).

    - ``block_sizes``: the FULL ``Config.block_sizes`` vector — every tunable axis sized.
    - ``red_values``: ``{block_id -> r_block}`` for every TUNABLE sized reduction (the user-tiled
      reduction axes that ride a ``block_sizes`` slot). The standard track's rolled primary rides
      ``reduction_loops`` instead, so it is surfaced via ``primary_r_block``/``persistent`` rather
      than here. EMISSION routing is the ONLY standard-vs-user difference — every reduction
      (rolled included) gets a size from the SAME budget; rolled ones happen to land on the
      ``reduction_loops`` knob instead of a ``block_sizes`` slot.
    - ``primary_r_block`` / ``persistent``: the primary reduction's chunk + persistence verdict
      (BYTE budget admits the full extent AND the row is re-read), an OUTCOME of the budget.
    - ``rolled_loop_sizes``: ``{block_id -> (r_block, persistent)}`` for every ROLLED reduction
      axis OTHER than the primary (a multi-graph kernel that rolls >1 reduction into separate
      ``reduction_loops`` subgraphs — each a SEQUENTIAL pass / its own group, sized against its own
      budget). Empty for the corpus (single rolled reduction) and for the user-tiled track.
    """

    block_sizes: list[int]
    red_values: dict[int, int]
    primary_r_block: int
    persistent: bool
    rolled_loop_sizes: dict[int, tuple[int, bool]]


class _TritonReductionSeedBase(AutotunerHeuristic):
    """Shared base for the two Triton inner-reduction seed heuristics. Both consume the Stage-1
    ``ReductionKernelFact`` through ONE budget allocator (:meth:`size_reduction_tiles`); the
    subclasses differ ONLY in how they map the allocation onto knobs (EMISSION routing):

    - **standard** (:class:`TritonStandardReductionHeuristic`): Helion rolls the rdim into a
      ``reduction_loops`` loop, so the primary reduction's size lands on that knob.
    - **user-tiled** (:class:`TritonUserTiledReductionHeuristic`): the user hand-writes the
      ``hl.tile`` loop, so each reduction axis is a ``block_sizes`` entry.

    Not registered; only the subclasses are.
    """

    backend = "triton"
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    # ----- THE BUDGET (a register/byte capacity; everything else is a per-axis desire) -----
    # Per-program persistent byte ceiling: the group's resident working set — the SUM over its
    # actual live tiles of ``itemsize × ∏(tile dims)`` (``CoResidencyGroup.live_tiles``) — must fit
    # this, else a tile floors. ~240 KiB, just over H100 SMEM. This SUBSUMES the old scattered
    # footprint caps (``_resident_tile_cap``, ``_carried_*``, ``M_COLLAPSE_TILE_BYTES`` inner cap,
    # ``_pinned_inner_resident_elems``): ONE budget + a group-total Σ-over-live-tiles footprint.
    ROW_PERSIST_MAX_BYTES = 245760
    # The tighter byte ceiling for a CARRIED reduction (an accumulator whose last dim is the rdim —
    # kl_div/jsd's ``[grid_M, R]``, the grad-parameter norm-bwd's ``[inner, N]``): that tile is held
    # resident across the WHOLE inner loop (not streamed-and-released like a persistent row), a
    # heavier steady-state pressure, so the chunk sharing SRAM with it wants a smaller extent. HALF
    # of ROW_PERSIST. This is the SOLE place the carried-vs-streamed distinction lives (one uniform
    # Σ-over-live-tiles footprint, two budget constants): kl_div R 4096 and layer_norm_bwd inner 2
    # fall out of the SAME footprint at this budget — no ``num_live`` / buffer-count multiplier.
    CARRIED_PERSIST_MAX_BYTES = 245760 // 2
    # The PERSISTENCE-HOLD ceiling — the (looser) byte watermark a RE-READ row may hold its FULL
    # extent under (vs the CHUNK budget, which sizes a streamed/looped tile). Only ``row_reread AND
    # carried_2d_count==0`` reductions reach the hold, so it never loosens a carried tile (kl_div/jsd
    # stay chunked). TWO ceilings, selected by ``_has_store_only_row_reread`` — an ADMITTED PROXY (see
    # that method) for "does persist's avoided HBM re-read live in the small L2 working set or the
    # large register file". The true cutoff is not a single faithful byte budget (softmax flips at
    # ~128-160 KiB, cross_entropy at ~256-384 KiB with the SAME footprint — measured), so these are
    # two CALIBRATED buckets that fit the curriculum, not a physical constant:
    #  - single fused reduce, no store-only re-read (``False`` — cross_entropy/cross_entropy_ls/sum):
    #    the row's reuse is register-resident, so a high resident watermark is a win far out — persist
    #    to ~600 KiB (cross_entropy N=50257 measured +7%, N=32000 +47%). 3× ROW.
    #  - a store-only re-reading pass exists (``True`` — softmax/rms/layer_norm/welford): the row is
    #    re-swept from L2; beyond ~32 K fp32 elems streaming beats holding it (softmax N=32768 persist
    #    +30%, N=49152 persist -34%). Tighter ceiling -> chunks at N >= ~49152. ~1.2× ROW (admits
    #    softmax N=32768 = 262156 B, rejects N=49152 = 393228 B). CAVEAT: rms_norm actually flips at
    #    ~160 KiB (earlier than this ceiling's 288 KiB implies) — the proxy is coarse; it buckets
    #    correctly on the corpus but the bucket VALUE is not a faithful per-kernel cutoff.
    PERSIST_HOLD_MAX_BYTES = 3 * 245760
    USER_TILE_PERSIST_HOLD_MAX_BYTES = 294912
    # Looped-fallback reduction chunk (pow2) for a row that does not fit the persistent budget.
    LOOPED_CHUNK = 16384
    # Occupancy floor for the grid-M widen: keep the post-tile grid >= num_sm * MIN_WAVES so
    # collapsing a fan-out sibling never under-occupies (mirrors the pointwise seed's MIN_WAVES).
    MIN_WAVES = 8
    # Diminishing-returns ceiling on the grid-M WIDEN (rows/program): a memory-bound reduction does
    # not amortize past a handful of batched rows, and widening only trades away grid parallelism
    # (measured g8 optimum; g64/g128 regress softmax ~1.4x and per_token_group ~1.1x). Bounds the
    # widen that the byte/occupancy caps alone would permit on a small-row huge-M kernel. Does NOT
    # bound the grad-param COLLAPSE branch (which intentionally batches many rows to cut the
    # cross-grid finalize) nor a raised autotuner_min floor (max(floor, ...) still wins).
    WIDEN_MAX_ROWS = 8

    # num_warps levers (kept OUTSIDE the budget — a scalar keyed on the primary's resident ROW
    # BYTES, §6.2.1). NARROW-row single-warp: a narrow reduction row wants ONE warp (the cross-warp
    # reduction tree is pure overhead). Gated on a row-byte cap AND an occupancy cap, both keyed on
    # ``input_load_itemsize`` (the HBM-load element width — faithful, dtype-agnostic).
    NARROW_W1_MAX_BYTES = 2048
    NARROW_W1_OCC_BYTE_LIMIT = 262144

    # =============================== Stage-1 fact accessors ================================= #
    @classmethod
    def _grid_axis_block_ids(cls, spec: ConfigSpec) -> tuple[int, ...]:
        """The parallel grid (M) axes -- grid block_ids with NO reduction SIZED over them. Read
        off the Stage-1 ``ReductionKernelFact.grid_axis_block_ids`` (PROMPT §2.1). The kernel fact
        is never absent on a reachable call (every caller runs only after
        ``_primary_descriptor_selected`` returned non-None, which requires a kernel fact present).
        """
        kf = spec.reduction_kernel_fact
        assert kf is not None
        return kf.grid_axis_block_ids

    @classmethod
    def _non_reduction_loop_ids(cls, spec: ConfigSpec) -> tuple[int, ...]:
        """The non-reduction user-tiled loops (welford's normalize pass) -- sized as a separate
        apply pass, NOT reduction-sized. Read off ``ReductionKernelFact.non_reduction_loop_block_ids``.
        """
        kf = spec.reduction_kernel_fact
        assert kf is not None
        return kf.non_reduction_loop_block_ids

    @classmethod
    def _reduction_block_ids(cls, spec: ConfigSpec) -> set[int]:
        """The set of REDUCTION-axis block_ids (the Stage-1 kernel fact's reductions). The
        MEMBERSHIP key for classifying an accumulator's dims (a dim is an rdim iff it is in this
        set -- NOT inferred from POSITION). Empty when no kernel fact (a bare-spec unit test)."""
        kf = spec.reduction_kernel_fact
        return {d.block_id for d in kf.reductions} if kf is not None else set()

    @classmethod
    def _resident_block_ids(cls, spec: ConfigSpec) -> set[int]:
        """The union of block_ids that appear (as a resolved dim) in SOME co-residency group's
        live-tile set — the faithful "is this axis register-resident?" test (PROMPT §4). The SINGLE
        definition of residency, shared by the grid-M widen (a resident grid axis widens into the
        byte budget; a NON-resident one is reduced away -> collapses) and ``_has_reduced_away_grid``
        (the num_warps M-collapse floor). Empty if no kernel fact (a bare-spec unit test)."""
        kf = spec.reduction_kernel_fact
        if kf is None:
            return set()
        resident: set[int] = set()
        for g in kf.coresidency_groups:
            for tile in g.live_tiles:
                resident.update(d for d in tile if d is not None)
        return resident

    @classmethod
    def _has_reduced_away_grid(cls, spec: ConfigSpec) -> bool:
        """True iff some grid axis is REDUCED AWAY — a grid block_id that appears in NO live tile,
        i.e. a sequential cross-grid reduction loop whose partial is finalized by a later
        ``.sum(0)`` (the grad-parameter M-collapse idiom). Uses the shared ``_resident_block_ids``
        residency test. False if no kernel fact."""
        kf = spec.reduction_kernel_fact
        if kf is None:
            return False
        resident = cls._resident_block_ids(spec)
        return any(g not in resident for g in kf.grid_axis_block_ids)

    @staticmethod
    def _max_group_footprint(
        kf: ReductionKernelFact,
        axis: int,
        footprint_terms: Callable[
            [tuple[tuple[int | None, ...], ...], int], tuple[int, int]
        ],
        default_tiles: tuple[tuple[int | None, ...], ...],
    ) -> tuple[int, int]:
        """The ``(scale, flat)`` footprint for sizing ``axis`` taken from the HEAVIEST co-residency
        group that SPANS it (the group whose live tiles give the largest ``scale``). A reduction
        axis is tiled the same width everywhere, so it must fit the worst group that uses it — jsd's
        rdim is shared across its KL group and its two heavier V-reduction groups; sizing it against
        the max group lands R at the measured optimum without a ``carried_2d_count`` fudge. ``flat``
        comes from that SAME max group (mixing scale/flat across groups breaks the chunk solve). If
        the axis spans no group's tiles (a bare-spec / degenerate case), fall back to
        ``default_tiles`` (this descriptor's own group)."""
        best = None
        for g in kf.coresidency_groups:
            if not any(axis in t for t in g.live_tiles):
                continue
            scale, flat = footprint_terms(g.live_tiles, axis)
            if best is None or scale > best[0]:
                best = (scale, flat)
        return best if best is not None else footprint_terms(default_tiles, axis)

    @classmethod
    def _has_store_only_row_reread(cls, spec: ConfigSpec, pd: ReductionDescriptor) -> bool:
        """True iff the primary reduction's row tensor is ALSO loaded by a STORE-ONLY pass — a load
        of that tensor that feeds a store and NO reduction (``stores_fed and not reductions_fed``).
        This is the literal, exact thing measured; the name says the mechanism, not an interpretation.

        WHY IT EXISTS + THE FAITHFULNESS CAVEAT (read this before trusting it): we WANT to know
        "does persistence's benefit (avoiding the row's HBM re-read) get served from the small L2
        working set (=> a TIGHTER persist-hold ceiling, chunk sooner) or from the larger register
        file (=> a LOOSER ceiling)?" — the physical quantity that sets the persist->chunk cutoff.
        MEASURED (this session, ncu + emitted Triton), that quantity is NOT cleanly recoverable from
        any seed-time signal we found:
          - softmax cutoff ~128-160 KiB, cross_entropy ~256-384 KiB (fp32) — SAME byte footprint,
            ~2x-different cutoff, so NO single byte budget is faithful.
          - persistent cross_entropy AND persistent rms_norm both emit ONE ``tl.load(x)`` (register-
            resident reuse), yet rms_norm flips at ~160 KiB and cross_entropy at ~256 KiB — so
            load-count / "# physical passes" does NOT predict the cutoff either.
          - rms_norm has a full-width [m,N] output tile (heavier resident set) and cross_entropy a
            scalar; but the GPU-verified scalar-output 2-pass adversarial kernel ALSO flips at
            softmax's ~160 KiB — so ``full_width_output`` is not the axis either.
        No graph-count / load-count / output-width / byte-budget proxy separated all cases. So this
        predicate is an ADMITTED PROXY that happens to classify the CURRICULUM correctly (softmax /
        rms_norm / layer_norm / welford -> SMALL; cross_entropy / sum / kl_div / jsd -> BIG) — it is
        NOT a faithful measure of the underlying cache-tier/working-set question and is known to be
        fooled off-corpus (an adversarial 2-pass kernel whose 2nd pass REDUCES instead of STORING
        re-reads the row identically but is classified False -> BIG -> measured 2.24x too-loose). If
        a future kernel regresses on the persist ceiling, THIS proxy is the first suspect; the real
        fix is a working-set-vs-cache-tier signal we could not compute at seed time. See
        ``_lab/redesign/APPLY_REREAD_ADVERSARIAL_CANDIDATES.md`` + the CF-Step notes.

        Detected from the walker ``MemoryOpFact`` list (already on the spec — no re-walk). NOT the
        same as ``non_reduction_loop_block_ids`` (softmax's 2nd pass reduces over the SAME axis so
        that set is empty for it). Empty facts / no kernel fact -> False."""
        facts = spec.memory_op_facts
        if not facts:
            return False
        red_tensors = {
            f.tensor_name
            for f in facts
            if f.kind == "load"
            and f.tensor_name is not None
            and any(ax == pd.block_id for ax, _ in f.reductions_fed)
        }
        if not red_tensors:
            return False
        return any(
            f.kind == "load"
            and f.tensor_name in red_tensors
            and f.stores_fed
            and not f.reductions_fed
            for f in facts
        )

    # =============================== scalar levers (outside the budget) ===================== #
    @classmethod
    def _num_warps(
        cls, pd: ReductionDescriptor, num_sm: int = 0, grid_rows: int = 0
    ) -> int:
        """Scale num_warps with the reduction extent (pow2): rnumel <= 1024 -> 4, <= 4096 -> 8,
        <= 16384 -> 16, > 16384 -> 32. NARROW-row single-warp refinement at the low end (a narrow
        row at low/moderate occupancy wants ONE warp). Keyed on ``input_load_itemsize`` (faithful,
        no dtype-kind branch); needs ``num_sm`` + a static grid (0 disables it)."""
        rnumel = pd.size_hint
        ils = pd.input_load_itemsize
        row_bytes = rnumel * ils
        have_enough_information = num_sm > 0 and ils > 0 and grid_rows > 0
        if have_enough_information:
            occ = grid_rows // num_sm
            if (
                pd.carried_2d_count == 0
                and row_bytes <= cls.NARROW_W1_MAX_BYTES
                and occ * row_bytes <= cls.NARROW_W1_OCC_BYTE_LIMIT
            ):
                return 1
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
        """The smallest valid block size for an entry (honors a raised ``autotuner_min`` for
        large-M shapes rather than emitting an invalid ``block_size=1``)."""
        return max(1, bs_spec.min_size, bs_spec.autotuner_min)

    @classmethod
    def _m_axis_block_size(cls, spec: ConfigSpec, mbid: int) -> int:
        """Seed block size (rows/program) for one M-axis (grid) block_id, whether or not it is a
        tunable ``block_sizes`` entry. A grid-PINNED axis (``hl.tile(M, block_size=1)``) has no
        tunable slot and lives solely on the program grid -- read its FIXED value off
        ``env.block_sizes`` (the grid-pinned-M idiom every vLLM quant kernel uses)."""
        if mbid in spec.block_sizes.valid_block_ids():
            m_idx = spec.block_sizes.block_id_to_index(mbid)
            return cls._block_floor(cast("BlockSizeSpec", spec.block_sizes[m_idx]))
        from ...runtime.config import Config as _Config
        from ..compile_environment import CompileEnvironment

        env = CompileEnvironment.current()
        value = env.block_sizes[mbid].from_config(_Config(block_sizes=[]))
        if isinstance(value, (int, torch.SymInt)):
            return max(1, int(value))
        log.warning(
            "reduction seed: M-axis block_id=%s resolved to a non-static block size %r; "
            "falling back to block_size=1 (this should not happen for a pinned grid axis)",
            mbid,
            value,
        )
        return 1

    @classmethod
    def _eviction_policies(
        cls,
        env: CompileEnvironment,
        kind: str,
        reread_slot: int | None = None,
    ) -> list[str] | None:
        """``load_eviction_policies`` list (spec length); None leaves the autotuner default.
        - ``"stream"`` — single streamed input (read once): every load -> ``'first'`` (frees L2).
        - ``"reread"`` — the row is re-read across passes: its first load -> ``'last'``
          (L2-resident), rest -> ``'first'``. ``reread_slot`` from ``reread_eviction_index``."""
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

    # ================================ THE BUDGET ALLOCATOR ============================== #
    @classmethod
    def size_reduction_tiles(
        cls,
        env: CompileEnvironment,
        spec: ConfigSpec,
        device_ir: DeviceIR,
        pd: ReductionDescriptor,
    ) -> _TileAllocation:
        """THE allocator: a per-co-residency-group BUDGET over the group's ACTUAL resident live
        tiles (``CoResidencyGroup.live_tiles``) assigns every tile size, in TWO passes.

        The footprint is faithful: ``resident_bytes = itemsize × Σ over the group's live tiles of
        ∏(tile dim widths)``. Sizing an axis A splits that sum into ``(scale, flat)`` — tiles
        CONTAINING A scale with ``block(A)``, tiles WITHOUT A are constant — and the budget test is
        ``itemsize × (scale × block(A) + flat) <= budget`` (the constant term SUBTRACTED, never
        divided). No ``num_live`` multiplier, no separate accumulator sum, no feature-extent
        reconstruction: the live tiles ARE the resident set (accumulators captured inline at real
        shape, scalar carries as rank-1 constant tiles).

        For each co-residency group:

          PASS 1 — seat the reductions with the grid axes pinned at their FLOOR (full-extent ->
            user-tile -> grid-tile). A re-read full-slice raises its floor to the full extent
            (PERSISTENCE) iff its resident tile fits the budget; else it chunks to
            ``min(LOOPED_CHUNK, byte budget, extent)``. A carried reduction (kl_div/norm-bwd) sizes
            against the tighter ``CARRIED_PERSIST`` budget.

          PASS 2 — the grid-M rows take the REMAINDER. A grid axis that is RESIDENT (appears in some
            live tile -> its row co-occupies the working set) WIDENS into the byte remainder (capped
            by occupancy + WIDEN_MAX_ROWS + extent) and FLOORS when the budget is spent. A grid axis
            in NO live tile is REDUCED AWAY (a sequential cross-grid ``.sum(0)`` finalize, holds no
            bytes) -> its floor raises to ``grid_rows / num_sm`` (collapse the finalize to ~1 SM
            wave). Both are pure per-axis MEMBERSHIP outcomes — no ``cdiv`` branch, no recognizer.

        Then the non-reduction loops LAST (welford's normalize, rms_norm_per_block's groups_per_row)
        — a separate pass co-resident with nothing in the group, sized against its own headroom.

        EMISSION is the ONLY standard-vs-user difference: a reduction's computed size is WRITTEN to
        ``reduction_loops`` (rolled/standard) or a ``block_sizes`` slot (user-tiled). Every
        reduction gets a size from the SAME budget; the split is codegen routing, not a different
        way to compute.
        """
        from ..._utils import next_power_of_2 as _np2
        from ..._utils import prev_power_of_2 as _pp2
        from ...runtime import get_num_sm

        num_sm = max(1, get_num_sm(env.device))
        occ_floor = num_sm * cls.MIN_WAVES
        itemsize = max(1, pd.itemsize)
        valid = set(spec.block_sizes.valid_block_ids())
        grid_ids = set(cls._grid_axis_block_ids(spec))
        non_reduction_loop_ids = set(cls._non_reduction_loop_ids(spec))
        reduction_ids = cls._reduction_block_ids(spec)
        kf = spec.reduction_kernel_fact
        assert kf is not None

        # Extent (pow2-padded) per block_id, read from STORED hints so the allocator runs without
        # an active CompileEnvironment (the unit tests call get_seed_config outside the env ctx):
        # a reduction's extent is its descriptor ``size_hint``; a tunable axis's is its
        # ``BlockSizeSpec.size_hint``; a non-tunable axis (a pinned grid / materialized feature) is
        # read off ``env.block_sizes`` (guarded — only the live compile path has such axes).
        _desc_extent = {d.block_id: d.size_hint for d in kf.reductions}
        _spec_extent = {
            cast("BlockSizeSpec", spec.block_sizes[i]).block_id: cast(
                "BlockSizeSpec", spec.block_sizes[i]
            ).size_hint
            for i in range(len(spec.block_sizes))
        }

        def extent_of(bid: int) -> int:
            if bid in _spec_extent:
                return _np2(_spec_extent[bid])
            if bid in _desc_extent:
                return _np2(_desc_extent[bid])
            return _np2(env.block_sizes[bid].size_hint())

        # The persistence/chunk BUDGET a reduction sizes against — the ONLY place the regime enters
        # (the footprint FORMULA is identical everywhere; only this number changes). TWO budgets,
        # keyed PER-REDUCTION on ONE faithful flag (does THIS reduction carry a >=2-D tile?):
        #  - CARRIED (``carried_2d_count > 0`` — the reduction's own ``[grid_M, R]`` accumulator is
        #    held resident across the WHOLE inner loop, kl_div/jsd): the TIGHTER budget. That tile is
        #    a heavier steady-state pressure than a streamed row -> smaller chunk (kl_div R 4096).
        #  - STREAMED (``carried_2d_count == 0``): the ROW budget. CRUCIAL that this is PER-REDUCTION,
        #    not kernel-wide: a grad-parameter norm-bwd carries its N accumulator on the MATERIALIZED
        #    N axis (c2d=3), but the co-resident INNER tile it sizes is itself NON-carried (c2d=0) and
        #    wants the looser ROW budget (inner 2, measured — a kernel-wide CARRIED flag floored it to
        #    1 and cost ~20% on layer_norm_bwd 8192x4096). A per-row SCALAR carry (welford mean/M2
        #    ``[grid_M]``) has c2d=0, so it stays STREAMED.
        def persist_budget_for(d: object) -> int:
            return (
                cls.CARRIED_PERSIST_MAX_BYTES
                if d.carried_2d_count > 0
                else cls.ROW_PERSIST_MAX_BYTES
            )

        # The PERSISTENCE-HOLD ceiling. IDEALLY keyed on "will persist's avoided HBM re-read be
        # served from the small L2 working set (=> tighter SMALL ceiling, chunk sooner) or the larger
        # register file (=> looser BIG ceiling)". That physical quantity is NOT faithfully recoverable
        # at seed time (see ``_has_store_only_row_reread`` — softmax and cross_entropy have the same
        # byte footprint but ~2x-different cutoffs, and neither load-count nor output-width nor a byte
        # budget separates them). So we use ``_has_store_only_row_reread`` as an ADMITTED PROXY that
        # classifies the CURRICULUM correctly (store-only re-read pass -> SMALL; single fused reduce
        # -> BIG) but is known-unfaithful off-corpus. Keyed on the PRIMARY; gates the hold, which only
        # fires for the primary's full-slice/user-tile reduction.
        hold_ceiling = (
            cls.USER_TILE_PERSIST_HOLD_MAX_BYTES
            if cls._has_store_only_row_reread(spec, pd)
            else cls.PERSIST_HOLD_MAX_BYTES
        )

        # The static grid-row count (program count before any widen), the occupancy numerator.
        from ..compile_environment import NoCurrentEnvironment

        grid_rows = 1
        try:
            for gbid in grid_ids:
                size = env.block_sizes[gbid].size
                if isinstance(size, (int, torch.SymInt)):
                    grid_rows *= env.size_hint(size)
                else:
                    grid_rows = 0  # dynamic grid -> no compile-time occupancy
                    break
        except (NoCurrentEnvironment, AttributeError, TypeError):
            grid_rows = 0

        # ``seated`` holds every tile assigned so far (held fixed for later sizing); ``sizes`` is
        # the subset that lands in tunable ``block_sizes`` slots. PASS 1 seats every grid axis at
        # its FLOOR; the reductions are sized against that floored grid, then PASS 2 widens the grid
        # into whatever budget the seated reductions left (the two-pass structure — reductions
        # first with the grid pinned low, then the grid).
        seated: dict[int, int] = {}
        for gbid in sorted(grid_ids):
            seated[gbid] = cls._m_axis_block_size(spec, gbid)
        sizes: dict[int, int] = {}
        red_values: dict[int, int] = {}
        rolled_loop_sizes: dict[int, tuple[int, bool]] = {}
        primary_r_block = 1
        persistent = False

        # The faithful "is this axis register-resident?" test (PROMPT §4), shared with the num_warps
        # M-collapse floor via ``_resident_block_ids``. A grid axis IN a live tile widens into the
        # byte budget (softmax/sum/kl_div — its row co-occupies the working set); a grid axis in NO
        # live tile is a sequential cross-grid reduction loop finalized by a later ``.sum(0)``
        # (bias_grad/norm-bwd's outer M) — it holds no bytes, so the byte budget cannot size it (it
        # collapses to ~1 SM wave instead). Replaced the old ``in_accumulator(g)`` gate, which got
        # ``sum`` wrong (sum's m IS a resident read-tile dim but is in no accumulator).
        resident_block_ids = cls._resident_block_ids(spec)

        # A kernel with a loop-carried >=2-D accumulator (``carried_2d_count >= 1`` on any reduction
        # — kl_div/jsd) pins that ``[grid_M, R]`` state in registers across the whole inner loop.
        # Widening its resident grid multiplies the pinned register footprint and trips the CTA-per-SM
        # register-occupancy cliff, which the leftover-byte widen + program-count ``occ_widen`` cannot
        # see (Pass 2 §S4-2b). So a carried kernel keeps its resident grid at FLOOR (no widen).
        carried_kernel = any(d.carried_2d_count > 0 for d in kf.reductions)

        def footprint_terms(
            tiles: tuple[tuple[int | None, ...], ...],
            axis: int,
        ) -> tuple[int, int]:
            """The faithful group footprint as ``(scale, flat)``: resident BYTES while sizing
            ``axis`` = ``itemsize × (scale × block(axis) + flat)`` — an R-SCALING term plus a
            CONSTANT term, kept SEPARATE (they ADD; folding the constant into a per-element
            coefficient and multiplying by the extent over-counts it, which wrongly denies
            persistence). We simply SUM ``∏(dim widths)`` over the group's ACTUAL resident live
            tiles (``CoResidencyGroup.live_tiles`` — each an actual ``dim_block_ids`` tuple). A tile
            CONTAINING ``axis`` scales with it (its ``∏(other dims)`` adds to ``scale``); a tile
            WITHOUT ``axis`` is constant (its ``∏(dims)`` adds to ``flat``). A ``None`` dim is a
            size-1 broadcast (grounding CF-Step 7: in an attributed group tile a ``None`` is never a
            resident full-width feature — the feature is always a RESOLVED block_id whose width
            ``extent_of`` gives). No ``num_live`` multiplier, no separate accumulator sum, no
            feature reconstruction: the tiles ALREADY are the resident set (accumulators captured
            inline at real shape, scalar carries as their own rank-1 constant tiles)."""
            scale = 0
            flat = 0
            for tile in tiles:
                contains_axis = axis in tile
                prod = 1
                for d in tile:
                    if d is None or d == axis:
                        continue
                    prod *= _tile_dim_width(d)
                if contains_axis:
                    scale += prod
                else:
                    flat += prod
            return max(1, scale), flat

        def _tile_dim_width(bid: int) -> int:
            """One resident dim's tile width by MEMBERSHIP: a seated reduction/grid tile (else its
            full extent). A grid axis's width is its seated rows (floor in pass 1, widened in pass 2)."""
            return max(1, seated.get(bid, extent_of(bid)))

        for g in kf.coresidency_groups:
            descs = [kf.reductions[i] for i in g.descriptor_indices]
            sized = [d for d in descs if d.category in SIZED_REDUCTION_CATEGORIES]
            if not sized:
                continue
            tiles = g.live_tiles

            # ---- PASS 1: seat the reductions (full-extent -> user-tile -> grid-tile) against the
            # group's live-tile footprint with the grid axes at their floor. ----
            order = sorted(
                sized,
                key=lambda d: (
                    0
                    if d.category in FULL_EXTENT_CATEGORIES
                    else (1 if d.category is ReductionCategory.USER_TILE else 2),
                    -d.size_hint,
                ),
            )
            # ``order`` is ``sized`` (SIZED_REDUCTION_CATEGORIES only): FULL_SLICE / FULL_GRID /
            # USER_TILE. A GRID_TILE reduction (jsd's grid amax) is NOT sized here — it is a grid
            # axis, seated at its floor in the grid loop above and widened in PASS 2 like any grid
            # row. So this loop never sees a GRID_TILE.
            for d in order:
                raw_ext = d.size_hint  # the true reduction extent (NOT pow2-padded)
                ext = extent_of(d.block_id)  # pow2-padded — the seated tile width
                materialized_full_width = (
                    d.category is ReductionCategory.FULL_SLICE
                    and d.block_id not in valid
                    and d.block_id not in spec.reduction_loops.valid_block_ids()
                )
                if d.category is ReductionCategory.FULL_GRID or materialized_full_width:
                    # FULL_GRID (cdiv == 1) or a MATERIALIZED full-width FULL_SLICE (the roller
                    # declined to roll it AND it has no tunable block_sizes slot — the grad-parameter
                    # ``grad_weight[N]`` accumulator axis that norm-bwd holds resident at full width,
                    # or per_token_group's specialized ``group_size``): the whole axis is ONE
                    # program's tile, full-extent resident BY DEFINITION. Seat at the full extent,
                    # never chunk it through the byte budget — it cannot be split across programs AND
                    # it has nowhere to emit a chunk. CRUCIAL: seating it full-width (not chunked to
                    # 1) is what lets the co-resident inner tile SEE the real N (else the inner tile
                    # reads N as 1 and grows to the full extent — a spill).
                    seated[d.block_id] = ext
                    if d.block_id == pd.block_id:
                        primary_r_block = ext
                        # The axis is seated at its FULL extent here (r == ext), so it is persistent
                        # by the uniform ``persistent = (r >= ext)`` rule the normal sizing path
                        # below also uses. (This is the honest definition — "the r-block covers the
                        # whole reduction extent" — replacing the old ``is FULL_GRID`` special-case,
                        # which was a misleading proxy: for a FULL_GRID axis ``persistent`` is inert
                        # anyway — ``reduction_loops=[]`` via ``is_materialized`` + eviction short-
                        # circuits on ``num_load==1`` — and for a materialized grad-param full-slice
                        # it dropped the re-read eviction hint via the ``not persistent`` gate. That
                        # hint is a small, SIZE-dependent L2 nudge; the faithful fix is a
                        # pinned-tile-bytes eviction gate, NOT overloading ``persistent`` — see
                        # ``_lab/redesign/SUGGESTED_CHANGES.md``.)
                        persistent = True
                    if d.block_id in valid:
                        red_values[d.block_id] = ext
                    continue
                # Resident BYTES(R) = itemsize × (scale × R + flat) over the live tiles. A reduction
                # axis is tiled the SAME width everywhere it appears, so it must fit the HEAVIEST
                # co-residency group that spans it — take the footprint from the max-``scale`` group
                # over ``d.block_id``, not just this descriptor's own group. On the corpus this only
                # bites jsd: its rdim (bid0) is shared across the KL group (scale 5) and the two
                # V-reduction groups (scale 9 each), so max=9 -> R 2048 (the measured optimum, for the
                # RIGHT reason — the heaviest spanning group — replacing the old
                # ``scale *= carried_2d_count`` fudge). kl_div's rdim spans ONE group (scale 6) ->
                # max==6 -> R 4096, unchanged. ``flat`` is taken from the SAME max group (mixing terms
                # across groups would break the chunk arithmetic).
                scale, flat = cls._max_group_footprint(
                    kf, d.block_id, footprint_terms, default_tiles=tiles
                )
                # PERSISTENCE: hold the full extent iff the row is re-read (a persistent pass fuses
                # reduce+apply to one HBM load) AND there is no carried 2-D tile (a carried tile is
                # held resident the whole loop — it chunks, never persists) AND the single resident
                # tile fits the persist ``hold_ceiling`` (apply-reread-keyed, computed above) AND the
                # extent clears the per-program element limit. The byte test uses the RAW extent (true
                # resident element count, not pow2-padded).
                element_cap = env.backend.max_tensor_numel
                ext_held = (
                    d.row_reread
                    and d.carried_2d_count == 0
                    and (element_cap is None or raw_ext <= element_cap)
                    and itemsize * (scale * raw_ext + flat) <= hold_ceiling
                )
                if ext_held:
                    r = ext
                else:
                    # stream/chunk: the largest pow2 R whose resident bytes fit the persist budget,
                    # solving ``itemsize × (scale × R + flat) <= budget`` for R (the CONSTANT term is
                    # SUBTRACTED, not divided). Capped by LOOPED_CHUNK and the extent. A CARRIED
                    # reduction (kl_div/jsd) sizes here against the tighter carried budget; a
                    # non-carried inner tile (grad-param) against ROW.
                    avail = persist_budget_for(d) // itemsize - flat
                    byte_budget = _pp2(max(1, avail // scale))
                    r = max(1, min(cls.LOOPED_CHUNK, byte_budget, ext))
                seated[d.block_id] = r
                # THREE independent routing checks (the block_sizes and reduction_loops namespaces
                # are DISJOINT — an axis is a ``block_sizes`` tile XOR a rolled ``reduction_loops``
                # axis, never both — so these are plain ``if``s, not an if/elif chain):
                # (A) the PRIMARY's scalar levers (num_warps ramp + standard-track reduction_loops).
                if d.block_id == pd.block_id:
                    primary_r_block = r
                    persistent = r >= ext and d.category in FULL_EXTENT_CATEGORIES
                # (B) a tunable ``block_sizes`` reduction (user-tiled) -> its block_sizes slot.
                if d.block_id in valid:
                    red_values[d.block_id] = r
                # (C) a ROLLED NON-primary reduction -> surface its size for the standard track's
                # reduction_loops emission. ``!= pd.block_id`` excludes the ROLLED PRIMARY (whose
                # size is emitted via ``primary_r_block`` in (A) instead — it would otherwise be
                # double-routed here). CORPUS-DARK (no corpus kernel rolls >1 reduction).
                if (
                    d.block_id != pd.block_id
                    and d.block_id in spec.reduction_loops.valid_block_ids()
                ):
                    rolled_loop_sizes[d.block_id] = (
                        r,
                        r >= ext and d.category in FULL_EXTENT_CATEGORIES,
                    )

            # ---- PASS 2: the grid-M rows take the remainder (widen / floor / collapse). ----
            for mbid in sorted(grid_ids):
                if mbid not in valid:
                    continue  # a grid-PINNED axis (FixedBlockSizeSource) -> fixed, not sized.
                ext = extent_of(mbid)
                floor = cls._block_floor(
                    cast(
                        "BlockSizeSpec",
                        spec.block_sizes[spec.block_sizes.block_id_to_index(mbid)],
                    )
                )
                if mbid not in resident_block_ids:
                    # a sequential cross-grid reduction loop (grad-param .sum(0)): in NO live tile ->
                    # NOT resident, holds no bytes. The byte budget cannot size it; raise the floor
                    # to ~1 SM wave to collapse the cross-grid finalize.
                    collapse = _np2(max(1, grid_rows // num_sm)) if grid_rows > 0 else 1
                    blk = max(floor, min(collapse, ext))
                elif carried_kernel:
                    # REGISTER-OCCUPANCY GUARD: a kernel with a loop-carried >=2-D accumulator
                    # (``carried_2d_count >= 1`` — kl_div/jsd) pins that ``[grid_M, R]`` state in
                    # registers across the WHOLE inner loop, so widening the grid MULTIPLIES the
                    # pinned register footprint and trips the CTA-per-SM occupancy cliff (jsd grid=2
                    # measured ~1.1x; register-occupancy, NOT a byte spill — n_spills=0). The
                    # leftover-byte widen can't see that (it is a LOOSE occupancy proxy when sizing a
                    # residual, vs a TIGHT one when Pass 1 sizes R to FILL the budget), and
                    # ``occ_widen`` only guards PROGRAM-COUNT occupancy (jsd grid=2 leaves 4096 >>
                    # 1056 programs, so it permits it). So a carried kernel KEEPS its resident grid at
                    # FLOOR. Replaces the old ``scale_w *= carried_2d_count`` fudge. Corpus: jsd
                    # (grid stays 1) + kl_div (already at floor 1) are the only carried kernels.
                    blk = floor
                else:
                    # resident parallel rows: widen into the byte remainder (same faithful
                    # ``scale × block + flat`` footprint over the live tiles — a wider grid row
                    # scales every tile CONTAINING the grid axis), capped by occupancy (keep the
                    # post-widen grid >= num_sm·MIN_WAVES), a diminishing-returns ROWS ceiling, and
                    # the extent; floors when the budget is full.
                    scale_w, flat_w = footprint_terms(tiles, mbid)
                    avail_w = persist_budget_for(pd) // itemsize - flat_w
                    byte_widen = _pp2(max(1, avail_w // scale_w))
                    if grid_rows > 0:
                        occ_widen = _pp2(max(1, grid_rows // occ_floor))
                    else:
                        occ_widen = (
                            1  # dynamic grid -> no compile-time occupancy -> no widen
                        )
                    # ROWS ceiling: batching more than WIDEN_MAX_ROWS reduction ROWS/program only
                    # trades away grid parallelism for a resident-row reduction (softmax/rms_norm:
                    # memory-bound, does not amortize past ~8 rows). Does NOT apply when the primary
                    # is FULL_GRID (the grid axis batches tiny grid-resident per-group reductions —
                    # per_token_group's groups_per_row — which wants the wide occupancy-bound widen).
                    rows_ceiling = (
                        ext
                        if pd.category is ReductionCategory.FULL_GRID
                        else cls.WIDEN_MAX_ROWS
                    )
                    blk = max(floor, min(byte_widen, occ_widen, rows_ceiling, ext))
                seated[mbid] = blk
                sizes[mbid] = blk

        # ---- the non-reduction / independent loops LAST (own budget vs the headroom) ----
        # welford's normalize loop / rms_norm_per_block's groups_per_row. Co-resident with nothing
        # in a group's reduction tile (a separate sequential pass), so each gets a FRESH budget
        # against its own extent capped by the streamed ROW budget.
        loop_budget = _pp2(max(1, cls.ROW_PERSIST_MAX_BYTES // itemsize))
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            bid = bs_spec.block_id
            if bid in red_values or bid in grid_ids or bid in reduction_ids:
                continue
            if bid in non_reduction_loop_ids or bid not in seated:
                # a non-reduction apply loop OR an independent standalone tiled loop: size it to
                # its own extent capped by the headroom (flooring it to 1 would serialize the pass).
                sizes[bid] = max(1, min(extent_of(bid), loop_budget))

        # ---- assemble the full block_sizes vector ----
        block_sizes: list[int] = []
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            bid = bs_spec.block_id
            if bid in sizes:
                block_sizes.append(sizes[bid])
            elif bid in red_values:
                block_sizes.append(red_values[bid])
            else:
                block_sizes.append(cls._block_floor(bs_spec))

        return _TileAllocation(
            block_sizes=block_sizes,
            red_values=red_values,
            primary_r_block=primary_r_block,
            persistent=persistent,
            rolled_loop_sizes=rolled_loop_sizes,
        )


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
        pd = _primary_descriptor_selected(env)
        return pd is not None and _is_standard_reduction(pd)

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
        pd = _primary_descriptor_selected(env)
        if pd is None:
            return None
        # ONE allocator sizes every axis from the per-co-residency-group budget (PROMPT §2.3/
        # §6.2.1): the reduction chunk(s), the grid M (the remainder — widen / floor / collapse,
        # all budget OUTCOMES), and the apply/independent loops, in one coherent pass. The standard
        # track then maps the sizing onto the rolled ``reduction_loops`` knob + the num_warps ramp
        # + eviction below (EMISSION routing only).
        alloc = cls.size_reduction_tiles(env, spec, device_ir, pd)
        block_sizes = alloc.block_sizes
        r_block, persistent = alloc.primary_r_block, alloc.persistent
        num_warps = cls._num_warps(
            pd,
            max(1, get_num_sm(env.device)),
            _grid_rows(env, cls._grid_axis_block_ids(spec)),
        )
        # GRAD-PARAMETER M-COLLAPSE warp FLOOR: a kernel that reduces its grid-M axis AWAY (finalized
        # by a later ``.sum(0)`` — a grid block_id in NO live tile, the faithful §4 collapse signal)
        # batches many M-rows per program and accumulates a wide ``[inner, N]`` gradient. That
        # cross-warp-parallelizable work wants >=8 warps even when the primary reduction's extent is
        # small (MEASURED: layer_norm_bwd 8192x4096 w4 is ~1.16x slower than w8; bias_grad_bwd wants
        # w16, w8 is +9.5%). A FLOOR, so it never lowers a large-rdim ramp (bias_grad stays 16). This
        # is INDEPENDENT of co-residency (bias_grad/dyt are collapse but single-reduction), so it is
        # NOT gated on a co-resident sibling. (Replaced the old ``max(floor, num_warps//2)`` block:
        # the ``//2`` was corpus-inert here — every collapse kernel has base ramp <=8 — AND measured
        # 2-5x HARMFUL on a co-resident large-rdim kernel; the only halving win was on the p2/p8
        # stress-probes, not the real corpus. See _lab/redesign for the full measurement trail.)
        if cls._has_reduced_away_grid(spec):
            num_warps = max(8, num_warps)

        # standard rides persistent-vs-looped on the rolled ``reduction_loops`` knob (the primary
        # rdim is NOT a block_sizes entry). MATERIALIZED rdim (rms/ln/instance bwd, the roller
        # declined to roll it): emit an EMPTY reduction_loops -- already full-width persistent, and
        # a length-1 list would fail normalize against the 0-length spec.
        is_materialized = pd.block_id not in spec.reduction_loops.valid_block_ids()
        reduction_loops: list[int | None]
        if is_materialized:
            reduction_loops = []
        elif len(spec.reduction_loops) <= 1:
            # Single rolled reduction (every corpus kernel): byte-identical to before.
            reduction_loops = [None] if persistent else [r_block]
        else:
            # MULTI rolled reduction (the relaxed gate, e.g. two SEQUENTIAL rolled reductions in
            # separate graphs / co-residency groups). One ``reduction_loops`` entry per spec in
            # spec order, mapping the allocator's sizing onto the knob: the primary spec uses
            # (r_block, persistent), the OTHER rolled specs use ``alloc.rolled_loop_sizes`` (each
            # sized against its OWN extent by the allocator -- a rolled axis has no block_sizes
            # slot, so the allocator surfaces it here rather than in red_values).
            reduction_loops = []
            for rl_spec in spec.reduction_loops:
                bid = rl_spec.block_ids[0]
                if bid == pd.block_id:
                    reduction_loops.append(None if persistent else r_block)
                else:
                    rb, pers = alloc.rolled_loop_sizes[bid]
                    reduction_loops.append(None if pers else rb)
        seed: dict[str, Any] = {
            "block_sizes": block_sizes,
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": 1,
            # 'flat': these reductions are grid-saturated at the M-grid.
            "pid_type": "flat",
        }
        # Eviction: streamed input -> 'first' everywhere; a re-read row RELOADED across a grid-
        # COLLAPSE loop -> pin it 'last' (first load), rest 'first'. Gated on
        # ``_has_reduced_away_grid`` (the grad-parameter ``.sum(0)`` M-collapse idiom — the program
        # batches MANY M-rows and reloads the row per row, so x is genuinely re-fetched from L2
        # across the collapse loop and pinning it pays), NOT on ``not persistent``. MEASURED (this
        # session, single-process): pinning helps a COLLAPSE reread (layer_norm_bwd 8192x4096 +2.9%,
        # rms_norm_bwd +1.9%) but HURTS a single fused persistent row that does NOT reload from L2
        # (cross_entropy V=30522 +2.2%, V=65536 +6.9% — the pin only oversubscribes L2 and evicts
        # store lines). ``not persistent`` was a proxy that (a) dropped the hint on the grad-param
        # collapse it should keep (exposed by the ``persistent=(r>=ext)`` cleanup) and (b) would have
        # pinned fused persistent cross_entropy. The audit doc's tile-bytes gate is REFUTED (non-
        # monotonic: grad-param 128 KiB helps, cross_entropy 122 KiB hurts — the reload STRUCTURE
        # decides, not size); ``_has_store_only_row_reread`` also fails to separate them (both False —
        # grad-param's reread load feeds BOTH a reduction and a store). ``_has_reduced_away_grid`` is
        # the discriminator here. (per_token_group hits the ``num_load==1`` stream branch first.)
        evict = None
        if pd.num_load == 1:
            evict = cls._eviction_policies(env, "stream")
        elif pd.row_reread and cls._has_reduced_away_grid(spec):
            # Re-read row's eviction slot read directly from the descriptor (its load's
            # MemoryOpFact.eviction_index), not a per-config codegen re-walk.
            evict = cls._eviction_policies(env, "reread", pd.reread_eviction_index)
        if evict is not None:
            seed["load_eviction_policies"] = evict
        return Config(**seed)


class TritonUserTiledReductionHeuristic(_TritonReductionSeedBase):
    """user-tiled inner-reduction seed: fires when the user hand-writes the ``hl.tile`` loop
    over the reduction axis (so the rdim is an ordinary ``block_sizes`` entry, e.g.
    ``hl.tile(n, block_size=R_BLOCK)``), which the upstream gate rejects entirely.

    Every axis (the reduction r_block(s), the grid rows, the apply loops) is sized by the shared
    :meth:`size_reduction_tiles` ONE budget allocator — there are NO per-band branches. The kernel
    families this track covers (plain user-tiled softmax, carried-2-D kl_div/jsd, reduce-then-apply
    welford, grad-parameter bias_grad/dyt) differ only in their Stage-1 facts (carried accumulators,
    non-reduction loops, materialized features), which the budget consumes uniformly; the
    floor-vs-resident and chunk-vs-persistent decisions are budget OUTCOMES. This track maps the
    allocation onto its knobs (every reduction axis is a ``block_sizes`` entry; no
    ``reduction_loops``) + num_warps + reread eviction below.
    """

    name = "triton_reduction_user_tile"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _triton_reduction_eligible(env, device_ir):
            return False
        pd = _primary_descriptor_selected(env)
        return pd is not None and not _is_standard_reduction(pd)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # Off sm90: upstream never fired on user-tiled, so no prior seed to preserve. Decline.
            return None
        from ...runtime import get_num_sm

        spec = env.config_spec
        pd = _primary_descriptor_selected(env)
        if pd is None:
            return None
        # ONE allocator sizes every axis from the per-co-residency-group budget (PROMPT §2.3/
        # §6.2.1): the user-tiled reduction chunk(s) on their block_sizes slots, the grid M (the
        # remainder), the apply loops, and the grad-parameter M-collapse (bias_grad/dyt) override,
        # all in one coherent pass — no r_block sized in isolation then block sizes in a separate
        # pass. The user-tiled track then maps the sizing onto num_warps + eviction below (no
        # reduction_loops knob; the rdim rides a block_sizes entry).
        alloc = cls.size_reduction_tiles(env, spec, device_ir, pd)
        block_sizes = alloc.block_sizes
        num_warps = cls._num_warps(
            pd,
            max(1, get_num_sm(env.device)),
            _grid_rows(env, cls._grid_axis_block_ids(spec)),
        )
        non_reduction_loop_ids = set(cls._non_reduction_loop_ids(spec))
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
        if non_reduction_loop_ids or pd.row_reread:
            # Re-read row's eviction slot read directly from the descriptor (its load's
            # MemoryOpFact.eviction_index), not a per-config codegen re-walk.
            ev = cls._eviction_policies(env, "reread", pd.reread_eviction_index)
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
