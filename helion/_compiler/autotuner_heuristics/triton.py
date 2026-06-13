from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ...autotuner.config_fragment import EnumFragment
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


# B200 / sm100 hardware target. The dedicated B200 reduction heuristics gate on this; the
# sm90 reduction heuristics DECLINE when it matches, so on sm100 exactly one reduction seed
# is collected (the B200-tuned one) — never a competing sm90 seed alongside it. Sibling
# precedent: ``TritonB200MatmulHeuristic`` above is already ``(("cuda", "sm100"),)``.
_B200_TARGET: tuple[tuple[str, str | None], ...] = (("cuda", "sm100"),)


def _triton_reduction_eligible(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    """Gate: exactly one ``ReductionFact`` and no ``matmul_facts``. Admits both tracks
    (T1 rollable, T2 user-tiled); excludes GEMMs and multi-axis manual reductions."""
    spec = env.config_spec
    return len(spec.reduction_facts) == 1 and not spec.matmul_facts


def _is_t1_reduction(spec: ConfigSpec, fact: ReductionFact) -> bool:
    """T1 vs T2 discriminator: T1 iff the rdim is a rollable ``reduction_loops`` entry,
    else T2 (a ``block_sizes`` entry). Exhaustive over eligible reductions (the two
    device_ir populators are mutually exclusive).
    """
    return fact.block_id in spec.reduction_loops.valid_block_ids()


def _grid_rows(env: CompileEnvironment, m_block_ids: tuple[int, ...]) -> int:
    """Product of the static M-axis (non-reduction grid) extents — the program count the
    reduction launches, the numerator of the occupancy ``grid_rows // num_sm``. 0 if any
    extent is not a statically-resolvable size (a dynamic/jagged grid has no compile-time
    occupancy, so the occupancy-gated narrow-w1 lever declines). A pure function of
    ``m_block_ids`` + env, computed on demand by the lever rather than stored on the fact.
    """
    grid_rows = 1
    for mbid in m_block_ids:
        size = env.block_sizes[mbid].size
        if not isinstance(size, (int, torch.SymInt)):
            return 0
        grid_rows *= env.size_hint(size)
    return grid_rows


class _TritonReductionSeedBase(AutotunerHeuristic):
    """Shared base for the two Triton inner-reduction seed heuristics. Both share the
    workload facts (``ReductionFact``), the persistent-vs-looped lever
    (``_persistent_looped``), the ``num_warps`` ramp, eviction provenance, and the
    block-size builders; the subclasses differ only in mapping that decision onto knobs:

    - **T1** (:class:`TritonReductionTileHeuristic`): rollable rdim, rides
      ``reduction_loops``.
    - **T2** (:class:`TritonReductionUserTileHeuristic`): user-tiled, the reduction axis
      is a ``block_sizes`` entry (plain-T2 softmax, Band-B kl_div/jsd, Band-C welford).

    Cloned from ``cute.CuteReductionTileHeuristic`` for triton (drops the CuTe-only
    knobs, adds ``num_warps`` / ``num_stages``). Not registered; only the subclasses are.
    """

    backend = "triton"
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    # Looped-fallback chunk (pow2) for rows above the structural cap (rnumel > 2**20).
    LOOPED_CHUNK = 16384
    # num_warps for the looped streaming branch.
    LOOPED_NUM_WARPS = 32
    # Band-B (T2 carrying [M_BLOCK, R_BLOCK] 2-D accumulators: kl_div, jsd) R_BLOCK cap, as a
    # per-program footprint R_BLOCK * itemsize * n_carried; bytes (via itemsize) for dtype-generality.
    BANDB_R_BLOCK_BYTES = 16384
    # Per-row persistent byte ceiling (size_hint * itemsize); above it a wide resident row spills
    # register/SMEM, so the reduction loops a fixed chunk instead. ~240 KiB, just over H100 SMEM.
    ROW_PERSIST_MAX_BYTES = 245760
    # Per-row ELEMENT ceiling for a FULL-WIDTH-output row (stores the whole [M, N] row back): its
    # resident tile is fp32-promoted, so it spills at a row WIDTH independent of input dtype, which
    # the byte cap above (input bytes) undercounts 2x for a half-precision row. Gates only
    # full_width_output rows; a no-op for the existing kernels (they reduce x.to(fp32) and top out
    # at N=16384 or already loop), steering half-precision full-width T1 rows onto the looped path.
    FULL_WIDTH_PERSIST_MAX_ELEMS = 81920
    # Band-C (welford reduce-then-apply) combine-tile FLOOR (32 KiB / itemsize elems): the serial
    # scalar recurrence (count/mean/M2) prefers a persistent combine; spill-safe at huge M_BLOCK.
    STRUCTURED_COMBINE_CAP_BYTES = 32768
    # Band-C combine cap is M_BLOCK-AWARE (raise-only): the spill driver is the per-program
    # footprint M_BLOCK * tile * itemsize, so the budget is divided by M_BLOCK (a small M_BLOCK
    # affords a wider combine tile; a raised M_BLOCK keeps the floor above). Never below the floor.
    STRUCTURED_COMBINE_PROG_BYTES = 262144
    # Apply/normalize stream chunk (bytes) for a MULTI-ROW-per-program reduce-then-apply
    # (M_BLOCK > 1): it holds the [M_BLOCK, tile] tile resident and would spill a wide tile, so it
    # streams a fixed per-row chunk. One-row-per-program applies keep the wide tile (no spill).
    APPLY_LOOP_STREAM_BYTES = 8192
    # M_BLOCK threshold at/above which the apply-stream cap fires. sm90: 2 (only multi-row
    # applies; single-row keeps the wide tile). The B200 T2 subclass lowers this to 1 (cap
    # even single-row applies — measured: B200 welford M_BLOCK==1 wants the 2048-elem
    # streamed tile, the wide tile costs ~25-30% there).
    APPLY_STREAM_CAP_MIN_M_BLOCK = 2

    # NARROW-row single-warp (occupancy-gated): a narrow reduction extent wants ONE warp (the
    # cross-warp reduction tree is pure overhead; w1 reduces in-register via shuffle). The win
    # inverts past an occupancy ceiling (the SMs saturate), so it is gated on a row-byte cap AND an
    # occupancy cap, both keyed on input_load_itemsize (the HBM-load element width — faithful and
    # dtype-agnostic, unlike the fp32-promoted accumulator itemsize which is 4 at both dtypes):
    #   - row cap: rnumel * input_load_itemsize <= NARROW_W1_MAX_BYTES.
    #   - occ cap: occ * row_bytes <= NARROW_W1_OCC_BYTE_LIMIT (a wider row saturates at lower
    #     occupancy, so the ceiling is on the product, not a flat occ).
    NARROW_W1_MAX_BYTES = 2048
    NARROW_W1_OCC_BYTE_LIMIT = 262144

    @classmethod
    def _bandb_r_block_cap(cls, fact: ReductionFact) -> int:
        """Pow2 R_BLOCK ceiling for a Band-B (carried 2-D tile) reduction: the per-program
        footprint ``BANDB_R_BLOCK_BYTES`` split across the accumulator itemsize and the
        carried-tile count. ``max(1, ..)`` guards a zero itemsize / tile count.
        """
        from ..._utils import next_power_of_2 as _np2

        cap = cls.BANDB_R_BLOCK_BYTES // (
            max(1, fact.itemsize) * max(1, fact.num_carried_2d_tiles)
        )
        return _np2(max(1, cap))

    @classmethod
    def _warp_ramp(cls, extent: int) -> int:
        """The streaming num_warps ramp at an arbitrary (pow2) reduction extent (per
        NumWarpsFragment): <=1024->4, <=4096->8, <=16384->16, >16384->32. Too few
        under-occupies the SM, too many wastes the cross-warp reduction tree. ``> 16384``
        (not ``>=``) keeps sum's widest in-sample row (16384) at w16, excluding the
        tiny-rnumel w32 regression.

        The single shared wide-row ladder behind BOTH ``_num_warps`` (keyed on the
        persistent extent) and the B200 Band-B re-key (keyed on the capped R_BLOCK), so
        the ladder lives in one place and cannot drift between the two call sites.
        """
        if extent > 16384:
            return 32
        if extent <= 1024:
            return 4
        if extent <= 4096:
            return 8
        return 16

    @classmethod
    def _num_warps(
        cls, fact: ReductionFact, extent: int, num_sm: int = 0, grid_rows: int = 0
    ) -> int:
        """num_warps for the persistent path: ``_warp_ramp(extent)`` with a NARROW-row
        single-warp refinement layered on at the LOW end. ``extent`` is the keyed
        reduction extent, passed EXPLICITLY by the caller (the persistent seed path keys
        on ``fact.size_hint``). The B200 Band-B re-key bypasses this method and calls
        ``_warp_ramp`` on the capped R_BLOCK directly — its narrow-w1 refinement would be
        disabled there anyway (gated ``num_carried_2d_tiles == 0``).

        NARROW-row single-warp (the occupancy-gated lever): a narrow row at low/moderate
        occupancy wants ONE warp (the cross-warp reduction tree is pure overhead — see
        ``NARROW_W1_MAX_BYTES``). Fires only when the row-byte cap AND the resident-pressure
        cap (``occ * row_bytes <= NARROW_W1_OCC_BYTE_LIMIT``) hold; both key on
        ``input_load_itemsize`` (faithful, no dtype-kind branch) and the occ ceiling scales
        DOWN as the row grows (a wider row cliffs at lower occupancy). Needs ``num_sm``
        (0 disables it, e.g. an off-device caller). Disjoint from the wide-row ramp
        (``NARROW_W1_MAX_BYTES`` << the extent>16384 region), so the two never interact.
        """
        ils = fact.input_load_itemsize
        row_bytes = extent * ils
        # NARROW-row single-warp (see NARROW_W1_MAX_BYTES); needs a known device + static grid.
        have_enough_information = num_sm > 0 and ils > 0 and grid_rows > 0
        if have_enough_information:
            occ = grid_rows // num_sm
            if (
                fact.num_carried_2d_tiles == 0  # not Band-B (kl_div/jsd)
                and row_bytes <= cls.NARROW_W1_MAX_BYTES
                and occ * row_bytes <= cls.NARROW_W1_OCC_BYTE_LIMIT
            ):
                return 1
        return cls._warp_ramp(extent)

    @classmethod
    def _block_floor(cls, bs_spec: BlockSizeSpec) -> int:
        """The smallest valid block size for an entry, used for every non-reduction axis
        the seed does not widen. Prefers one row/program but honors a raised
        ``autotuner_min`` (large-M shapes) rather than emitting an invalid ``block_size=1``.
        """
        return max(1, bs_spec.min_size, bs_spec.autotuner_min)

    @classmethod
    def _m_block_product(cls, spec: ConfigSpec, fact: ReductionFact) -> int:
        """Product of the seed's floored M-axis (grid) block sizes — the number of rows each
        program processes (1 unless a huge-M shape raised ``autotuner_min``). Shared by the
        apply-loop stream cap (``_build_block_sizes``) and the Band-C combine cap so they read
        the same M_BLOCK.
        """
        m_block = 1
        for mbid in fact.m_block_ids:
            m_idx = spec.block_sizes.block_id_to_index(mbid)
            m_block *= cls._block_floor(cast("BlockSizeSpec", spec.block_sizes[m_idx]))
        return m_block

    @classmethod
    def _build_block_sizes(
        cls,
        spec: ConfigSpec,
        fact: ReductionFact,
        red_block_id: int | None,
        red_value: int | None,
        non_reduction_loop_ids: frozenset[int] | set[int] = frozenset(),
    ) -> list[int]:
        """Build the ``block_sizes`` list: the reduction axis gets ``red_value``, each
        non-reduction loop tile (``non_reduction_loop_ids``, disjoint from the reduction
        block_id) gets ``loop_block``, every other axis its ``_block_floor``.
        ``red_block_id`` is None for T1 (the reduction rides ``reduction_loops``, not a
        block_sizes entry).

        The non-reduction loop tile matches the reduction tile — ``red_value`` (T2) or
        ``next_pow2(size_hint)`` (T1, where ``red_value`` is None). The normalize pass
        carries no accumulator, so this tile is a pure seed (a sane non-size-1 start, never
        a correctness constraint); the autotuner refines it from there.
        """
        from ..._utils import next_power_of_2 as _np2

        loop_block: int | None = None
        if non_reduction_loop_ids:
            # Match the reduction tile: red_value (T2) or next_pow2(size_hint) (T1, where
            # red_value is None). One rule, no byte-keyed widening — a sane non-size-1 seed.
            loop_block = red_value if red_value is not None else _np2(fact.size_hint)
            # ...except a MULTI-ROW-per-program apply (M_BLOCK > 1, the raised autotuner_min
            # at huge M) holds the [M_BLOCK, tile] tile resident and spills a wide tile, so it
            # streams a fixed per-row chunk instead (see APPLY_LOOP_STREAM_BYTES — welford fp32
            # huge-M wide-N otherwise cliffs to <=50% of torch.compile). On sm90 a
            # one-row-per-program apply (M_BLOCK == 1) keeps the wide tile
            # (APPLY_STREAM_CAP_MIN_M_BLOCK == 2); the B200 T2 subclass lowers the threshold
            # to 1 (cap even single-row applies — B200 wants the streamed tile there too).
            if cls._m_block_product(spec, fact) >= cls.APPLY_STREAM_CAP_MIN_M_BLOCK:
                chunk = max(1, cls.APPLY_LOOP_STREAM_BYTES // max(1, fact.itemsize))
                loop_block = min(loop_block, _np2(chunk))

        red_idx = (
            spec.block_sizes.block_id_to_index(red_block_id)
            if red_block_id is not None
            else None
        )
        out: list[int] = []
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            if i == red_idx:
                out.append(cast("int", red_value))
            elif bs_spec.block_id in non_reduction_loop_ids and loop_block is not None:
                out.append(loop_block)
            else:
                out.append(cls._block_floor(bs_spec))
        return out

    @classmethod
    def _eviction_policies(
        cls,
        env: CompileEnvironment,
        kind: str,
        reread_slots: tuple[int, ...] = (),
    ) -> list[str] | None:
        """``load_eviction_policies`` list (spec length), keyed on per-load residency;
        None leaves the autotuner default.

        - ``"stream"`` — single streamed input (``num_load == 1``: sum, long_sum), read
          once: every load -> ``'first'`` (frees L2).
        - ``"reread"`` — one or more rows re-read across passes: EACH re-read load -> ``'last'``
          (L2-resident), the rest -> ``'first'``. ``reread_slots`` are those loads' actual
          slots, read directly from ``ReductionFact.reread_eviction_indices`` (each re-read
          load's ``MemoryOpFact.eviction_index``), not guessed or re-walked per config.

        Other kinds leave the default until a per-slot win is confirmed.
        """
        n = env.config_spec.load_eviction_policies.length
        if n <= 0:
            return None
        if kind == "stream":
            return ["first"] * n
        if kind == "reread":
            valid = [i for i in reread_slots if 0 <= i < n]
            if not valid:
                return None
            policy = ["first"] * n
            for i in valid:
                policy[i] = "last"
            return policy
        return None

    @classmethod
    def _persistent_looped(
        cls, env: CompileEnvironment, fact: ReductionFact
    ) -> tuple[bool, int]:
        """The shared first lever (both tracks); returns ``(persistent, extent)``.
        Persistent for every row the backend compiles in one pass (up to
        max_tensor_numel = 2**20 elems) AND within the per-row byte ceiling
        (ROW_PERSIST_MAX_BYTES, the register/SMEM spill limit); else loop a fixed chunk.
        ``num_warps`` is computed separately by ``_seed_num_warps`` (one explicit
        warp-keying path shared by T1 and T2), not bundled into this tuple.

        The byte cap is unconditional but config-neutral off the re-read kernels (a
        single-load looped chunk ties the persistent pass; the Band-B cap is already
        tighter), so only rms_norm/layer_norm/softmax/cross_entropy/welford are steered.
        """
        from ..._utils import next_power_of_2 as _np2

        # Persistent iff ALL (independent): the element cap (None => compile limit), the per-row
        # byte ceiling (residency — correct for a scalar-output re-read row like cross_entropy),
        # AND — for a full_width_output row — a per-row ELEMENT ceiling, since its fp32-promoted
        # resident tile spills at a WIDTH the input-byte cap undercounts 2x at half precision
        # (see FULL_WIDTH_PERSIST_MAX_ELEMS).
        element_cap = env.backend.max_tensor_numel
        can_persist = (
            (element_cap is None or fact.size_hint <= element_cap)
            and (fact.size_hint * max(1, fact.itemsize) <= cls.ROW_PERSIST_MAX_BYTES)
            and (
                not fact.full_width_output
                or fact.size_hint <= cls.FULL_WIDTH_PERSIST_MAX_ELEMS
            )
        )

        if can_persist:
            # Persistent: T1 encodes the extent as reduction_loops=None; T2 as the full
            # pow2 R_BLOCK so the inner `for tile_n` runs once.
            return True, _np2(fact.size_hint)
        # Looped: exceeds the 2**20 or byte cap. Fixed chunk.
        return False, cls.LOOPED_CHUNK

    @classmethod
    def _seed_num_warps(
        cls, env: CompileEnvironment, fact: ReductionFact, persistent: bool
    ) -> int:
        """num_warps for the persistent-vs-looped seed path — the ONE explicit
        warp-keying path both T1 and T2 ``get_seed_config`` funnel through.

        - Persistent: the extent-keyed ramp (``_num_warps``), keyed EXPLICITLY on
          ``fact.size_hint``. ``num_sm`` + ``grid_rows`` (the product of static M extents,
          computed on demand — a pure function of ``m_block_ids`` + env, not stored on the
          fact) feed the occupancy-gated narrow-row w1 refinement inside ``_num_warps``.
        - Looped (exceeds the 2**20 / byte cap): the fixed high streaming warp count.
        """
        if not persistent:
            return cls.LOOPED_NUM_WARPS
        from ...runtime import get_num_sm

        num_sm = max(1, get_num_sm(env.device))
        grid_rows = _grid_rows(env, fact.m_block_ids)
        return cls._num_warps(fact, fact.size_hint, num_sm, grid_rows)


class TritonReductionTileHeuristic(_TritonReductionSeedBase):
    """T1 (rollable-rdim) inner-reduction seed: sum, long_sum, rms_norm, layer_norm,
    softmax-row, cross_entropy. Triton analog of ``CuteReductionTileHeuristic`` (keeps
    its registry name), deepening the original one-row/persistent/``['last']`` seed with
    the num_warps ramp, persistent-vs-looped, and per-slot eviction.

    Gated by ``_triton_reduction_eligible`` (T1 track) — broader than upstream
    ``is_canonical_row_reduction`` (also multi-axis rollable rows, raised-``autotuner_min``
    large-M shapes). Off sm90 the H100-tuned levers are unvalidated, so it falls back to
    ``_narrow_seed`` (pre-existing behavior preserved).
    """

    name = "triton_reduction_tile"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _triton_reduction_eligible(env, device_ir):
            return False
        spec = env.config_spec
        return _is_t1_reduction(spec, spec.reduction_facts[0])

    @classmethod
    def _narrow_seed(cls, env: CompileEnvironment) -> Config:
        """The upstream conservative T1 seed (one row/program, single persistent pass,
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
        if cls.HARDWARE_TARGETS != _B200_TARGET and matches_hardware(env, _B200_TARGET):
            # On sm100, defer to the dedicated B200 subclass (else two T1 seeds would be
            # collected — this narrow one plus the B200-tuned one). The B200 subclass sets
            # HARDWARE_TARGETS == _B200_TARGET so it SKIPS this guard and runs the rich
            # branch below. DEAD CODE off sm100: matches_hardware is an exact
            # compute-capability match (no arch fallback), so on sm90 this is always False
            # and the sm90 path is byte-identical to before.
            return None
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # Off the H100-validated target: keep the upstream conservative seed.
            return cls._narrow_seed(env)
        spec = env.config_spec
        fact = spec.reduction_facts[0]
        # T1 rides persistent-vs-looped on `reduction_loops`, so the lever's `extent`
        # (the T2 R_BLOCK) is unused here; warps key on the persistent extent.
        persistent, _extent = cls._persistent_looped(env, fact)
        num_warps = cls._seed_num_warps(env, fact, persistent)

        # A T1 reduction may be followed by a normalize loop (e.g. `s = x.sum(); out =
        # x/s`); its extra block_sizes tile(s) are sized by _build_block_sizes (matched to
        # the reduction tile). Only a seed (a worse tile costs autotuning time, never
        # correctness), so emit and let the autotuner refine.
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)

        # red_block_id=None: the rdim is not a block_sizes entry, so every entry is a
        # grid axis (floored) or a normalize loop tile (sized to the reduction tile). None
        # loop => the single grid block at its floor, as before.
        reduction_loops: list[int | None] = [None] if persistent else [cls.LOOPED_CHUNK]
        seed: dict[str, Any] = {
            "block_sizes": cls._build_block_sizes(
                spec, fact, None, None, non_reduction_loop_ids=non_reduction_loop_ids
            ),
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": 1,
            # 'flat': these reductions are grid-saturated at the M-grid.
            "pid_type": "flat",
        }
        # Eviction: streamed input -> 'first' everywhere; looped re-read -> first load
        # 'last', rest 'first'. Persistent rows stay resident, so left at default.
        evict = None
        if fact.num_load == 1:
            evict = cls._eviction_policies(env, "stream")
        elif fact.row_reread and not persistent:
            # Re-read rows' eviction slots read directly from the fact (each load's
            # MemoryOpFact.eviction_index), not a per-config codegen re-walk.
            evict = cls._eviction_policies(env, "reread", fact.reread_eviction_indices)
        if evict is not None:
            seed["load_eviction_policies"] = evict
        return Config(**seed)


class TritonReductionUserTileHeuristic(_TritonReductionSeedBase):
    """T2 (user-tiled) inner-reduction seed: fires on a T2 reduction (the reduction
    axis is an ordinary ``block_sizes`` entry, i.e. a user
    ``hl.tile(n, block_size=R_BLOCK)``), which the upstream gate rejects entirely.
    Covers three mutually-exclusive sub-regimes in one linear path: R_BLOCK starts at the
    shared persistent-vs-looped extent, then is capped by this workload's live state:

    - **plain T2** (softmax_two_pass): no cap — persistent full-pow2 R_BLOCK, T1-style
      reread-eviction for wide looped rows.
    - **Band B** (kl_div, jsd): carries ``[M_BLOCK, R_BLOCK]`` 2-D tiles, so a full-N
      R_BLOCK spills — cap by ``BANDB_R_BLOCK_BYTES / (itemsize * num_carried_2d_tiles)``.
    - **Band C** (welford, ``non_reduction_loop_block_ids`` non-empty): reduce-then-apply
      — cap the combine tile by ``STRUCTURED_COMBINE_CAP_BYTES``, size normalize tile(s)
      to match the reduction tile (see ``_build_block_sizes``).

    TODO(reductions): as more structured families land, promote each band into its own
    fact-keyed ``AutotunerHeuristic`` subclass rather than growing this method.
    """

    name = "triton_reduction_user_tile"

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not _triton_reduction_eligible(env, device_ir):
            return False
        spec = env.config_spec
        return not _is_t1_reduction(spec, spec.reduction_facts[0])

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if cls.HARDWARE_TARGETS != _B200_TARGET and matches_hardware(env, _B200_TARGET):
            # On sm100, defer to the dedicated B200 subclass (which sets HARDWARE_TARGETS
            # == _B200_TARGET and so SKIPS this guard). DEAD CODE off sm100 (exact-match
            # matches_hardware), so the sm90 path is byte-identical to before.
            return None
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # Off sm90: upstream never fired on T2, so no prior seed to preserve. Decline.
            return None
        from ..._utils import next_power_of_2 as _np2

        spec = env.config_spec
        fact = spec.reduction_facts[0]
        persistent, extent = cls._persistent_looped(env, fact)
        num_warps = cls._seed_num_warps(env, fact, persistent)

        # T2: the rdim IS a block_sizes entry (no reduction_loops knob); persistent ==
        # R_BLOCK >= next_pow2(N). Other axes stay at floor (keeps Band-B M_BLOCK at 1,
        # required by the u0*u1 <= 2**20 constraint). R_BLOCK starts at the lever extent,
        # then is capped by live state (the three sub-regimes are mutually exclusive):
        r_block = extent
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)
        if fact.num_carried_2d_tiles >= 1:
            # Band B (kl_div, jsd): a full-N R_BLOCK over-allocates the carried 2-D tiles and
            # spills, so cap the footprint (R_BLOCK * itemsize * n_carried) via _bandb_r_block_cap.
            r_block = min(r_block, cls._bandb_r_block_cap(fact))
        elif non_reduction_loop_ids:
            # Band C (welford, groupnorm): cap the combine tile (M_BLOCK-aware, raise-only; see
            # STRUCTURED_COMBINE_PROG_BYTES). Normalize tile(s) sized to match in _build_block_sizes.
            itemsize = max(1, fact.itemsize)
            m_block = cls._m_block_product(spec, fact)
            floor_elems = cls.STRUCTURED_COMBINE_CAP_BYTES // itemsize
            budget_elems = (
                cls.STRUCTURED_COMBINE_PROG_BYTES // max(1, m_block) // itemsize
            )
            cap = max(floor_elems, budget_elems)
            r_block = min(r_block, _np2(max(1, cap)))

        seed: dict[str, Any] = {
            "block_sizes": cls._build_block_sizes(
                spec,
                fact,
                fact.block_id,
                r_block,
                non_reduction_loop_ids=non_reduction_loop_ids,
            ),
            "num_warps": num_warps,
            "num_stages": 1,
            "pid_type": "flat",  # see the T1 branch.
        }
        # Reread eviction: welford (reduce-then-apply) always re-reads across combine +
        # normalize, so 'last' regardless of persistence; plain-T2 (softmax_two_pass) only
        # when re-read AND looped. kl_div/jsd (row_reread=False, no normalize) unaffected.
        if non_reduction_loop_ids or (fact.row_reread and not persistent):
            # Re-read rows' eviction slots read directly from the fact (each load's
            # MemoryOpFact.eviction_index), not a per-config codegen re-walk.
            ev = cls._eviction_policies(env, "reread", fact.reread_eviction_indices)
            if ev is not None:
                seed["load_eviction_policies"] = ev
        return Config(**seed)


# ===========================================================================
# B200 (sm100) dedicated reduction seed heuristics.
#
# These subclass the H100/sm90 T1/T2 heuristics but gate on sm100 and are the promoted
# compiler default there. The sm90 classes above DECLINE on sm100 (their B200 guard), so
# on B200 exactly one reduction seed is collected — the B200 one. Sibling precedent:
# ``TritonB200MatmulHeuristic`` in this file is already sm100-gated and registered.
#
# Because ``HARDWARE_TARGETS == _B200_TARGET`` here, the inherited ``get_seed_config``
# SKIPS the sm90 classes' B200-defer guard and runs the rich branch (the
# ``matches_hardware(env, cls.HARDWARE_TARGETS)`` check passes on B200) — so the full
# persistent-vs-looped / num_warps-ramp / eviction / band logic fires on sm100. The shared
# ``_num_warps`` lever already reads ``get_num_sm(env.device)`` (148 on B200) and
# ``input_load_itemsize`` (the true HBM-load width, 2 at halves), so it is already
# hardware- and dtype-aware; the B200 constants are re-tuned from here by the climb.
#
# B200 (sm100) constants start from the inherited H100 values as the STARTING HYPOTHESIS
# and are re-tuned per the method (oracle answer-key -> field-diff -> A/B -> gates). When a
# B200 constant diverges from H100, it is overridden on the subclass (leaving the sm90
# value untouched on the base).
# ===========================================================================
class TritonB200ReductionTileHeuristic(TritonReductionTileHeuristic):
    """B200 (sm100) T1 inner-reduction seed (sum, long_sum, rms_norm, layer_norm,
    softmax-row, cross_entropy). Inherits the T1 logic; gates on sm100 and is promoted to
    the compiler default there. Constants re-tuned for B200 (148 SMs, 2B-or-4B reduction
    widths at half precision) as the climb shows what must change."""

    name = "triton_b200_reduction_tile"
    promote_seed_to_default = True
    HARDWARE_TARGETS = _B200_TARGET

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        # Hardware-gate the B200 class itself, so on a non-sm100 box it never fires (the
        # sm90 sibling serves there). The reduction-track check is the inherited one.
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return False
        return super().is_eligible(env, device_ir)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        from ...runtime import get_num_sm

        config = super().get_seed_config(env, device_ir)
        if config is None:
            return config
        spec = env.config_spec
        fact = spec.reduction_facts[0]
        d = dict(config.config)
        changed = False

        # (A) TWO-PASS full-width row -> fewer warps. A full-width-output reduction whose
        # resident input row feeds >= 2 reduction passes (layer_norm: mean-sum then
        # variance-sum) holds the row live across TWO serialized cross-warp reduction trees;
        # the streaming warp ramp (keyed on element count) over-provisions and pays the
        # second tree's shuffle/barrier latency on a register-heavy resident row. Halve the
        # ramp warps (one pow2 step). DTYPE-INDEPENDENT (the second-tree latency is not a
        # byte-rate effect): layer_norm (2048,14336) wants w8 at fp32 (0.98->1.31) AND bf16/
        # fp16 (0.74->1.06), which the load-width law could not explain. A single-pass row
        # (rms_norm: x^2-sum, row_reduction_passes==1) keeps the ramp (measured: rms_norm
        # (2048,16384) bf16 w16=1.205 > w8=1.127). Co-gated on full_width_output so a
        # scalar-output two-pass row (cross_entropy: passes==2 but full_width False, which is
        # reduction-tree-bound and wants MORE warps) is excluded. Keyed on the reduction-pass
        # COUNT (a faithful structural count, not kernel identity / num_load — a 2-load
        # two-pass row would still fire, a 3-load one-pass row would not).
        warps = d.get("num_warps")
        if (
            fact.full_width_output
            and fact.row_reduction_passes >= 2
            and isinstance(warps, int)
            and warps > 1
        ):
            d["num_warps"] = max(1, warps // 2)
            changed = True

        # (B) Grid-starved LOOPED row -> persistent_interleaved pid_type. A looped wide-N
        # reduction launches only `grid_rows` programs (one per kept row); when grid_rows < num_sm
        # the flat grid is under one wave on 148 SMs. At the high warp count this wide-N seed
        # selects (the looped branch fixes num_warps=32), those `grid_rows` programs are
        # maximally-fat CTAs, and a flat sub-one-wave launch of fat CTAs schedules pathologically
        # on B200. Declaring a PERSISTENT (hardware-sized) grid fixes the launch — a pure
        # launch-grid + loop-wrap codegen change (program_id.py), each program still reducing
        # WHOLE rows (no split-k, no source change). With block_size = cdiv(grid_rows,
        # num_sm) = 1 (the gate guarantees grid_rows < num_sm), persistent_blocked and
        # persistent_interleaved assign program p -> row p IDENTICALLY (verified in the lowered
        # Triton + equal DRAM bytes per ncu) — but interleaved's strided `range(pid, total,
        # num_sm)` issue order schedules the DRAM request stream far better than blocked's
        # contiguous `range(start, end)`: ncu shows 47%->77% of peak memory throughput on the
        # firing cells with IDENTICAL bytes read and occupancy (a DRAM-access-scheduling /
        # bank-level-parallelism effect, NOT L2 locality — L2 hit rate ~0 both). The win
        # concentrates where the inner-loop trip count (ceil(N/16384)) is NOT a power of two
        # (the chunk stride then de-aliases the pow2 DRAM bank interleave); it is a dead tie at
        # pow2 trip counts (both issue orders reach the same bank steady state). Persisted A/B
        # over the firing cells (do_bench headline + CUDA-graph cross-check, both metrics on
        # every cell — no per-cell metric cherry-picking): interleaved has FEWER below-floor
        # cells than blocked on BOTH metrics (do_bench 0 vs 1; CUDA-graph 1 vs 6 — interleaved
        # rescues the half-precision cells blocked leaves below floor, e.g. (96,393216) bf16/fp16
        # cudagraph 0.62/0.63->1.07/1.06; (64,786432) fp32 0.95->1.56). The lone genuine trade is
        # (96,393216) fp32: a bounded ABOVE-floor nick (do_bench 0.92->0.87, cudagraph 0.81->0.75
        # ~at floor) where the chunks=24 verdict flips by dtype. No faithful key separates that
        # fp32 nick from the fp32 chunks=12/40/48 WINS without a curriculum fence (fp32 is
        # non-monotonic in chunks), so it is an accepted §3 net-positive trade — one bounded
        # above-floor nick vs many large wins and a net REDUCTION in below-floor cells. (The
        # extreme-low-M cells like (16,2097152)/(64,655360), M<<148, are the accepted
        # non-realistic grid-starvation corner; they also tie/improve but are not the
        # justification.) Falls back to persistent_blocked if interleaved is unavailable.
        # Gated on the OCCUPANCY (grid_rows < num_sm — a faithful hardware-unit property, never
        # dtype/identity) AND the looped branch (a persistent single-pass ROW already saturates).
        red_loops = d.get("reduction_loops")
        is_looped = bool(red_loops) and red_loops != [None]
        if is_looped:
            allowed = spec.allowed_pid_types
            pid = None
            if "persistent_interleaved" in allowed:
                pid = "persistent_interleaved"
            elif "persistent_blocked" in allowed:
                pid = "persistent_blocked"
            if pid is not None:
                num_sm = max(1, get_num_sm(env.device))
                grid_rows = _grid_rows(env, fact.m_block_ids)
                # occ == 0 (grid_rows < num_sm): the flat grid cannot fill one wave.
                if 0 < grid_rows < num_sm:
                    d["pid_type"] = pid
                    changed = True

        return Config(**d) if changed else config


class TritonB200ReductionUserTileHeuristic(TritonReductionUserTileHeuristic):
    """B200 (sm100) T2 user-tiled inner-reduction seed (softmax_two_pass, kl_div, jsd,
    welford). Inherits the T2 logic; gates on sm100 and is promoted to the compiler default
    there. Constants re-tuned for B200 as the climb shows what must change."""

    name = "triton_b200_reduction_user_tile"
    promote_seed_to_default = True
    HARDWARE_TARGETS = _B200_TARGET

    # Cap the apply/normalize stream tile even for a SINGLE-row-per-program reduce-then-apply
    # (M_BLOCK == 1), unlike sm90 which keeps the wide tile there. On B200 a single-row
    # welford apply with the full-width tile is ~25-30% slower than the 2048-elem streamed
    # tile: the cap (APPLY_LOOP_STREAM_BYTES // itemsize = 8192//4 = 2048 elems, since
    # welford's reduction itemsize is the fp32-promoted accumulator width 4 at ALL dtypes)
    # holds across fp32/bf16/fp16. Measured (in-process A/B vs the flat seed): welford
    # (8192,8192) fp32 0.715->0.938, (8192,5120) bf16 0.706->0.812 (both clear floor); 2048
    # is the elem optimum (1024 and 4096 both worse). Keyed on the resident-tile footprint
    # (bytes via the accumulator itemsize), never dtype/identity.
    APPLY_STREAM_CAP_MIN_M_BLOCK = 1

    # Band-C (reduce-then-apply with a carried SCALAR combine: welford/groupnorm-style,
    # non_reduction_loop_ids non-empty AND num_carried_2d_tiles == 0) scales num_warps with
    # the HBM LOAD WIDTH, not the element count. The streaming ramp keys num_warps on rnumel
    # (elements), but Band-C's serial count/mean/M2 recurrence does not parallelize across
    # warps — what matters is feeding the memory pipeline, whose byte-rate per element is
    # input_load_itemsize. So the faithful warp count balances byte-rate:
    #     num_warps_eff = ramp_warps * input_load_itemsize // FP32_ITEMSIZE   (floored at 1)
    # fp32 (4B) -> ×4//4 = ramp unchanged; bf16/fp16 (2B) -> ×2//4 = half; a future fp8 (1B)
    # -> quarter. This is WIDTH-CONTINUOUS over the field's physical range {1,2,4,8} with NO
    # threshold literal — it is not the {<=2} dtype-step fence (Gate D), and it is not a
    # total-BYTE ramp (Gate D showed the effect is load-WIDTH, not size×width: bf16 (8192,
    # 12288)=24576B wants halving while fp32 (16384,4096)=16384B keeps the ramp — byte order
    # inverted, so a byte threshold mis-keys; width is the real property). Measured wins
    # (in-process A/B): welford (32768,8192) bf16 w16->w8 0.67->1.01, (262144,2048) bf16/fp16
    # 0.82->1.16, (8192,5120) bf16/fp16 ~+34%; fp32 byte-identical (×1, zero regression).
    # The same shape wants opposite warps by width — (8192,12288) fp32 w16=0.99 vs bf16
    # w8=1.03 — which is exactly the byte-rate law deciding, faithfully, by the real property.
    FP32_ITEMSIZE = 4

    # Band-C resident M-tile footprint cap (bytes). At HUGE M the grid-block floor
    # (raise_grid_block_minimums, spec-level) raises M_BLOCK so a program holds an
    # [M_BLOCK, R_BLOCK] resident tile of M_BLOCK * R_BLOCK * itemsize bytes; past ~64 KiB
    # the per-program register/SMEM footprint spills and the reduce-then-apply cliffs. The
    # spec only raises the autotuner's SEARCH floor (autotuner_min) — a SEED config may
    # legally emit a smaller M_BLOCK (verified: normalize clamps only against min_size, not
    # autotuner_min). So cap the seed's M_BLOCK (pow2, lower-only) to fit this budget.
    # Measured: welford (262144,2048) fp32 M_BLOCK 16->8 (131072B->65536B) G 0.71->0.85
    # (clears floor); byte-exact no-op at every other welford cell (next-largest resident is
    # 65536B at M_BLOCK<=4, already within budget) so it cannot cause a regression. Keyed on
    # the resident-tile footprint (bytes), sibling to STRUCTURED_COMBINE_PROG_BYTES /
    # APPLY_LOOP_STREAM_BYTES — never a shape/dtype literal.
    BANDC_RESIDENT_MTILE_CAP_BYTES = 65536

    # Reduction-extent threshold (elems) below which a Band-C reduction takes ONE warp (the
    # serial scalar combine has negligible per-warp work at tiny extent). 1024 = the base
    # num_warps ramp's smallest band edge (rnumel<=1024 -> w4); Band-C goes one further to
    # w1. Keyed on the raw reduction extent (size_hint), reusing the ramp's own band edge.
    NARROW_BANDC_W1_MAX_ELEMS = 1024

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return False
        return super().is_eligible(env, device_ir)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        config = super().get_seed_config(env, device_ir)
        if config is None:
            return config
        spec = env.config_spec
        fact = spec.reduction_facts[0]
        red_idx = spec.block_sizes.block_id_to_index(fact.block_id)

        # Band C = reduce-then-apply with a carried SCALAR combine (non-reduction loop tiles
        # present, no carried 2-D tile). Band B (handled above) and plain-T2 (softmax: no
        # non_reduction loops) are unaffected by either Band-C lever.
        is_band_c = (
            bool(fact.non_reduction_loop_block_ids) and fact.num_carried_2d_tiles == 0
        )
        if not is_band_c:
            return config
        d = dict(config.config)
        changed = False

        # (1) Width-continuous byte-rate-balancing warp count (see class comment).
        ils = fact.input_load_itemsize
        warps = d.get("num_warps")
        if ils > 0 and isinstance(warps, int) and warps > 1:
            eff = max(1, warps * ils // cls.FP32_ITEMSIZE)
            if eff != warps:
                d["num_warps"] = eff
                changed = True

        # (1b) NARROW Band-C -> single warp. At a tiny reduction extent (size_hint in the
        # ramp's smallest band) the serial count/mean/M2 combine has almost no per-warp work,
        # so even fp32 (which the width law (1) leaves at the ramp floor) wants ONE warp; the
        # cross-warp reduction tree is pure overhead. This is the Band-C analog of the
        # streaming narrow-w1 lever, keyed on the reduction extent (a raw faithful shape dim)
        # at the SAME small-band threshold the base ramp already uses (1024). Measured
        # welford (16384,768) fp32 w4->w1 0.71->0.84 (median-of-9 sweep; clears floor — the
        # last below-floor welford cell; persisted re-check confirms w4=0.705 stable across 5
        # reads, NOT noise), (4096,1025) fp32 0.90->1.13; neutral where already >=w-effective
        # ((16384,1024) fp32). The halves already reach w1 here via (1)/the streaming lever,
        # so this only newly affects fp32.
        warps = d.get("num_warps")
        if (
            fact.size_hint <= cls.NARROW_BANDC_W1_MAX_ELEMS
            and isinstance(warps, int)
            and warps > 1
        ):
            d["num_warps"] = 1
            changed = True

        # (2) Resident M-tile footprint cap (see BANDC_RESIDENT_MTILE_CAP_BYTES).
        block_sizes = list(d.get("block_sizes", []))
        red_idx = spec.block_sizes.block_id_to_index(fact.block_id)
        if 0 <= red_idx < len(block_sizes):
            r_block = block_sizes[red_idx]
            itemsize = max(1, fact.itemsize)
            m_block = cls._m_block_product(spec, fact)
            footprint = m_block * max(1, r_block) * itemsize
            if footprint > cls.BANDC_RESIDENT_MTILE_CAP_BYTES and m_block > 1:
                budget_m = max(
                    1,
                    cls.BANDC_RESIDENT_MTILE_CAP_BYTES // (max(1, r_block) * itemsize),
                )
                from ..._utils import next_power_of_2 as _np2

                # Largest pow2 m_block <= budget_m (and <= current m_block).
                target_m = min(m_block, _np2(budget_m + 1) // 2 if budget_m >= 1 else 1)
                target_m = max(1, target_m)
                if target_m < m_block:
                    # Shrink the M-tile product to target_m by dividing ONE M-axis entry by
                    # the full pow2 ratio (not every entry — dividing all of them shrinks the
                    # product by ratio**n, overshooting for n>=2). Pick the first entry that can
                    # absorb the whole ratio (>= ratio); fall back to the first divisible entry.
                    # The common case is a single M-axis block (welford's m_block_ids has one
                    # entry), where this is exactly block_sizes[m] // ratio == target_m.
                    ratio = m_block // target_m
                    m_indices = [
                        mi
                        for mbid in fact.m_block_ids
                        if 0
                        <= (mi := spec.block_sizes.block_id_to_index(mbid))
                        < len(block_sizes)
                        and block_sizes[mi] > 1
                    ]
                    absorber = next(
                        (mi for mi in m_indices if block_sizes[mi] >= ratio),
                        m_indices[0] if m_indices else None,
                    )
                    if absorber is not None:
                        block_sizes[absorber] = max(1, block_sizes[absorber] // ratio)
                        d["block_sizes"] = block_sizes
                        changed = True

        return Config(**d) if changed else config
