from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

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
    # num_warps for the looped branch (the huge-rnumel streaming class wants 32).
    LOOPED_NUM_WARPS = 32
    # Band-B (T2 carrying [M_BLOCK, R_BLOCK] 2-D accumulators: kl_div, jsd) R_BLOCK cap,
    # in bytes/accumulator: a full-N persistent R_BLOCK spills, so cap the footprint at
    # R_BLOCK * itemsize * n_carried. Bytes (via itemsize) for dtype-generality.
    BANDB_R_BLOCK_BYTES = 16384
    # Per-row persistent byte ceiling, above which the reduction loops a fixed chunk (a
    # wide resident row spills register/SMEM). Caps on real row bytes (size_hint *
    # itemsize), not next_pow2 (vocab sizes sharing a next_pow2 split at the crossover).
    # 240 KiB sits in the measured dead-zone (most impactful on re-read kernels like CE).
    ROW_PERSIST_MAX_BYTES = 245760
    # Per-row ELEMENT ceiling for a FULL-WIDTH-output row (one that stores the whole [M, N] row
    # back): above this width the persistent fp32 row tile + the full-width store spill, so the
    # row must loop. Keyed on ELEMENTS (the resident fp32 tile width), NOT the HBM-input bytes —
    # the byte cap (size_hint*itemsize) undercounts a half-precision full-width T1 row 2x (its
    # resident tile is fp32-promoted), which is exactly why log_softmax bf16 at N~98304 wrongly
    # persisted (196 KB input row < the 240 KB byte cap) and spilled ~16x.
    #   81920 is the MEASURED stable bf16 persist-vs-loop crossover: persist wins at and below
    # 81920 (N=73728 persist +37% stable over 4 fresh procs, N=81920 +16%), loop wins decisively
    # at and above 86016 (N=86016 loop +36%, N=98304 +135%, N=131072 +313%). Above it the
    # persistent fp32 row tile (81920 elems * 4 B = 320 KiB resident) — already past the H100's
    # 228 KiB max shared memory, so it lives in registers and spills as the width grows — loses
    # to the streamed looped chunk. It is keyed to that measured spill crossover, not to the
    # curriculum (no curriculum log_softmax shape lies in the (65536, 81920] band; the band is
    # honored only so a non-curriculum full-width kernel there still gets the faster persist).
    # The fp32 crossover is a touch lower (~77k) so 81920 lets fp32 N=81920 persist when loop is
    # ~9% faster — a small, bounded edge trade on a non-curriculum width (fp32 wide full-width is
    # already looped by the byte cap at the curriculum's N>=98304). Only gates full_width_output
    # rows (scalar re-read rows like cross_entropy persist far past this, read-bound); the
    # existing full-width 9 reduce x.to(fp32) (itemsize 4) and top out at N=16384 or already loop,
    # so it is a no-op for them and steers only half-precision full-width T1 rows (log_softmax).
    FULL_WIDTH_PERSIST_MAX_ELEMS = 81920
    # Band-C (welford reduce-then-apply) combine-tile cap, in bytes. The combine is a
    # serial scalar recurrence (count/mean/M2) that prefers persistent; this is the
    # FLOOR of the combine tile (32 KiB / itemsize = 8192 elems) — the spill-safe budget
    # validated at the raised-M_BLOCK (huge-M) welford shapes.
    STRUCTURED_COMBINE_CAP_BYTES = 32768
    # Band-C combine cap is M_BLOCK-AWARE (raise-only). The real spill driver is the
    # per-program footprint ``M_BLOCK * combine_tile * itemsize`` (the combine carries one
    # [M_BLOCK]-wide scalar per stat), NOT the per-row tile bytes alone — so the prior flat
    # ``STRUCTURED_COMBINE_CAP_BYTES`` cap was too tight at SMALL M_BLOCK (it throttled the
    # combine tile of a small-M wide-N row that has register headroom to spare). A small
    # M_BLOCK affords a WIDER combine tile (fewer combine-loop trips, more ILP); a raised
    # M_BLOCK (huge-M) must stay narrow. So the combine tile is bounded by this per-program
    # byte budget DIVIDED by M_BLOCK, but never BELOW the validated floor above (raise-only,
    # so huge-M is byte-for-byte unchanged — the welford(262144,5120) 7.3x valley is
    # untouched). 256 KiB matches the resident-pressure scale of NARROW_W1_OCC_BYTE_LIMIT;
    # it is a hardware footprint ceiling, not a curriculum value. The combine tile shrinks
    # 65536->16384->8192 as M_BLOCK grows 1->4->8 (footprint pinned at 256 KiB), reaching the
    # 8192-elem floor at M_BLOCK>=8 (huge-M, where it is byte-identical to the old flat cap).
    # CUDA-graph-validated: at M_BLOCK=1 wide-N, welford +1-11% and groupnorm +1-16.5%; the
    # M_BLOCK=4/8 transition tiles are flat-to-faster; welford huge-M (M_BLOCK>=16) zero-regression.
    STRUCTURED_COMBINE_PROG_BYTES = 262144

    # A wide half-precision REDUCTION-BOUND row (re-read for multiple reductions, no
    # full-width output store) whose reduction-tree footprint is <= this many bytes prefers
    # w8 over w32: the cross-warp shared-memory reduction tree dominates and grows ~linearly
    # with warp count, so fewer warps win once the half-precision row is small enough in
    # bytes for 8 warps to cover. cross_entropy bf16 V<=~50k lives here (+35-49%).
    REREAD_W8_MAX_BYTES = 102400

    # Band-B (carries [M_BLOCK, R_BLOCK] 2-D accumulator tiles: kl_div, jsd) at a WIDE
    # half-precision row prefers w8 over the ramp's w32 — the SAME reduction-tree-overhead
    # mechanism as the re-read scalar-output case above, but reached via a different kernel
    # structure (a streaming 2-D-tile reduction, NOT re-read). Gated on the half-precision
    # INPUT-LOAD width (``input_load_itemsize <= 2``): at fp32 the 2-D-tile footprint is
    # heavier and w32 stays optimal (kl_div fp32 wide-V w8 regresses 5.6-7.3%) — so the byte
    # signal faithfully excludes fp32 without a dtype-kind branch. jsd/kl_div bf16 at V>=16384
    # are ~+6-15% faster at w8 (oracle-confirmed; seed w32 avg +10.8% off-optimal).
    BANDB_W8_MAX_INPUT_ITEMSIZE = 2

    # NARROW-row single-warp (occupancy-gated). A narrow reduction extent wants ONE warp:
    # the cross-warp reduction tree (shared-mem + __syncthreads) is pure overhead when each
    # warp's slice is tiny, and w1 reduces in-register via shuffle (0 shared traffic, 0
    # barriers). The win INVERTS to a >10% (up to 6x) regression past an occupancy ceiling
    # (the SMs saturate, so more warps amortize), so it is gated on BOTH a row-byte cap and
    # an occupancy cap, both keyed on the INPUT-LOAD byte width (``input_load_itemsize``:
    # 2 bf16/fp16, 4 fp32) — faithful and dtype-AGNOSTIC (the HBM-load element size IS a
    # workload property), NOT ``itemsize`` (the fp32-promoted accumulator width = 4 at BOTH
    # dtypes for the norm/softmax family, which cannot discriminate — the bug that admitted
    # softmax fp32 (32768,512) at +19.6% in an earlier fixed-occ attempt).
    #   - row cap: ``rnumel * input_load_itemsize <= NARROW_W1_MAX_BYTES`` (bf16 rnumel<=1024,
    #     fp32 rnumel<=512). Above it the row is wide enough that 1 warp under-utilizes.
    #   - occ cap: the per-program saturation point DROPS as the resident row grows (a wider
    #     row saturates the SMs' latency-hiding at lower occupancy — measured: a 1 KiB row is
    #     safe to occ~496 but a 2 KiB row cliffs by occ~200). So the cap is on the PRODUCT
    #     ``occ * row_bytes <= NARROW_W1_OCC_BYTE_LIMIT`` (i.e. ``occ <= LIMIT // row_bytes``),
    #     not a flat occ: 256 KiB gives a 512-byte row occ<=512, a 1 KiB row occ<=256, a 2 KiB
    #     row occ<=128 (below softmax's ~200 cliff at 2 KiB). A flat occ cap mis-fired w1 into
    #     that 2 KiB cliff (softmax bf16 (32768,1024) +18-30%).
    # CUDA-graph-fit across softmax/rms_norm/layer_norm/sum/welford/cross_entropy x bf16/fp32:
    # in the fired zone w1 beats the ramp by up to +62% (softmax bf16) with worst regression
    # ~4.7% vs the cell-best (a small upper-edge trade; no fact separates the edge from the
    # core). Disabled when input_load_itemsize==0 (kl_div/jsd: 2-D carried tiles, no single
    # reduction-fed row load — and forcing w1 there regresses up to +46%).
    NARROW_W1_MAX_BYTES = 2048
    # occ * row_bytes ceiling (256 KiB): the resident-pressure product above which w1 cliffs.
    NARROW_W1_OCC_BYTE_LIMIT = 262144

    @classmethod
    def _num_warps(cls, fact: ReductionFact, num_sm: int = 0) -> int:
        """Scale num_warps with the reduction extent (pow2, per NumWarpsFragment):
        rnumel <= 1024 -> 4, <= 4096 -> 8, <= 16384 -> 16, > 16384 -> 32. Too few
        under-occupies the SM, too many wastes the reduction tree.

        NARROW-row single-warp refinement at the LOW end (the occupancy-gated lever): a
        narrow row at low/moderate occupancy wants ONE warp (the cross-warp reduction tree
        is pure overhead — see ``NARROW_W1_MAX_BYTES``). Fires only when the row-byte cap AND
        the resident-pressure cap (``occ * row_bytes <= NARROW_W1_OCC_BYTE_LIMIT``) hold; both
        key on ``input_load_itemsize`` (faithful, no dtype-kind branch) and the occ ceiling
        scales DOWN as the row grows (a wider row cliffs at lower occupancy). Needs ``num_sm``
        (0 disables it, e.g. an off-device caller). Disjoint from the wide-row w8/w32 branch
        below (``NARROW_W1_MAX_BYTES`` << the rnumel>16384 region), so the two never interact.

        One dtype-aware refinement at the WIDE end (rnumel > 16384): a REDUCTION-BOUND row
        — re-read (``row_reread``, i.e. live across >=2 reductions, e.g. cross_entropy's
        amax+sum) AND with a per-row SCALAR output (``not full_width_output``: it does NOT
        store the whole [M, N] row back, so the kernel is not store/occupancy-bound) — over
        a wide half-precision row whose reduction-tree footprint is <= ``REREAD_W8_MAX_BYTES``
        (size_hint * itemsize) prefers w8 over w32. ncu: at w32 the cross-warp shared-memory
        reduction tree is ~4x costlier (bank conflicts) and throttles the read pipeline;
        fewer warps win because the kernel is reduction-tree-bound on a tiny output.
        CUDA-graph-validated M-stable (occ 15-124) and zero-regression: cross_entropy bf16 at
        V=32000/50257 is ~35-49% faster. Excluded by faithful facts (NOT kernel identity):
        sum/long_sum (not re-read -> single reduction, want w32); layer_norm/softmax/welford/
        rms_norm (full_width_output -> store/occupancy-bound, want w32 — keying on num_load
        instead regressed layer_norm ~32%); kl_div/jsd (streaming, not re-read). The mechanism
        is dtype-AGNOSTIC (not an fp32-vs-half switch): the byte cap simply means a 4-byte fp32
        row hits the cap at half the extent of a 2-byte row, so realistic fp32 vocabs (V>=30522
        -> >cap) stay w32 while the common bf16 vocabs (V~32k/50k -> <=cap) get w8; fp32 CE in
        V in [16384,25600] does fire w8 and also benefits. (The NARROW-row fewer-warps want,
        the opposite low-extent regime, is the occupancy-gated w1 branch above.)
        """
        rnumel = fact.size_hint
        # NARROW-row single-warp: a narrow row at low/moderate occupancy. Both caps key on
        # the input-load byte width (dtype-faithful), and the occupancy needs num_sm; an
        # input_load_itemsize of 0 (kl_div/jsd, or unknown) disables it.
        ils = fact.input_load_itemsize
        row_bytes = rnumel * ils
        if (
            num_sm > 0
            and ils > 0
            and fact.grid_rows > 0
            and row_bytes <= cls.NARROW_W1_MAX_BYTES
        ):
            # grid_rows==0 means a dynamic/jagged M-grid: occupancy is unknown at compile
            # time, so the lever DECLINES (an unknown occ must not assume the best-case low
            # occ — that would wrongly fire w1 on a possibly-saturated grid).
            # Cap the resident-pressure PRODUCT occ * row_bytes (a wider row saturates the SM
            # latency-hiding at lower occupancy), so the safe occupancy ceiling scales down as
            # the row grows: occ <= LIMIT // row_bytes.
            occ = fact.grid_rows // num_sm
            if occ * row_bytes <= cls.NARROW_W1_OCC_BYTE_LIMIT:
                return 1
        # >16384 (not >=) so sum's widest in-sample row (16384) stays w16, excluding the
        # tiny-rnumel w32 regression.
        warps32_min_elems = 16384
        if rnumel > warps32_min_elems:
            # Reduction-bound (re-read, scalar output) wide rows want fewer warps; the byte cap
            # caps the resident-row footprint (so wide fp32, 4 B/elem, hits it sooner than bf16)
            # and full-width/non-reread kernels stay on the w32 default.
            if (
                fact.row_reread
                and not fact.full_width_output
                and rnumel * max(1, fact.itemsize) <= cls.REREAD_W8_MAX_BYTES
            ):
                return 8
            # Band-B (2-D-tile streaming reduction: kl_div, jsd) at a wide half-precision row
            # — same reduction-tree-overhead win, reached via a different structure. Gated on
            # the input-load width so fp32 (heavier tiles, w32-optimal) is faithfully excluded.
            if (
                fact.num_carried_2d_tiles >= 1
                and 0 < fact.input_load_itemsize <= cls.BANDB_W8_MAX_INPUT_ITEMSIZE
            ):
                return 8
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
        block_id) gets the widened size derived from ``fact``, every other axis its
        ``_block_floor``. ``red_block_id`` is None for T1 (the reduction rides
        ``reduction_loops``, not a block_sizes entry).

        A non-reduction loop tile is a pure perf lever (the normalize pass carries no
        accumulator): persistent ``next_pow2(N)`` when per-row valid work is small,
        looped above. Keys on per-row valid bytes (``n_valid * itemsize``) so rows
        sharing a next_pow2 separate, and is NOT tied to ``red_value`` (welford wants a
        narrower normalize tile than its combine tile).

        Fallback when the extent is not static (``static_rnumel is None``): the byte cap
        has no extent to key on, so match the reduction tile — ``red_value`` (T2) or
        ``next_pow2(size_hint)`` (T1). NOT tuned on any kernel (none has a dynamic-extent
        non-reduction loop).
        """
        from ..._utils import next_power_of_2 as _np2

        # Per-row valid bytes below which the loop tile stays persistent at next_pow2(N).
        persist_max_bytes = 12288
        # Looped chunk (bytes) above that threshold — the FLOOR of the widened chunk below.
        # Do NOT lower without re-gating: a 4096-elem chunk is a ~6x regression valley at the
        # large-M / N~5120 welford class (M_BLOCK=16).
        loop_chunk_bytes = 8192
        # Half-precision wide looped rows afford a WIDER normalize chunk (M-aware, raise-only).
        # Keyed on input_load_itemsize<=2 (the bf16/fp16 HBM-load width — the SAME dtype-faithful
        # workload signal the w8/narrow-warp levers use, NOT a dtype-kind branch). At fp32
        # (input_load_itemsize 4) the normalize-tile optimum is NON-MONOTONIC in width (measured:
        # helps N~16k, regresses N~20k), so it is not faithfully width-keyable and is left at the
        # floor for the autotuner to refine. The footprint is M_BLOCK*chunk*itemsize, so the
        # budget divides by M_BLOCK (huge-M keeps the floor — the 7.3x valley is untouched), and
        # it only fires when the row spans >= 2 widened chunks (np2_n > raised), so a row that
        # would collapse to a single persistent chunk stays at the floor. CUDA-graph-validated:
        # welford+groupnorm bf16/fp16 small-M wide-N -2 to -20%, zero welford regression.
        normalize_prog_bytes = 16384

        loop_block: int | None = None
        if non_reduction_loop_ids:
            if fact.static_rnumel is None:
                # Untuned dynamic-extent default: match the reduction tile (``red_value``
                # for T2; ``next_pow2(size_hint)`` for T1, where red_value is None).
                loop_block = (
                    red_value if red_value is not None else _np2(fact.size_hint)
                )
            else:
                np2_n = _np2(fact.size_hint)
                n_valid = fact.static_rnumel
                itemsize = max(1, fact.itemsize)
                if n_valid * itemsize <= persist_max_bytes:
                    loop_block = np2_n
                else:
                    loop_chunk_elems = max(1, loop_chunk_bytes // itemsize)
                    if 0 < fact.input_load_itemsize <= 2:
                        m_block = 1
                        for mbid in fact.m_block_ids:
                            m_idx = spec.block_sizes.block_id_to_index(mbid)
                            m_block *= cls._block_floor(
                                cast("BlockSizeSpec", spec.block_sizes[m_idx])
                            )
                        raised = max(
                            1, normalize_prog_bytes // max(1, m_block) // itemsize
                        )
                        # Only widen when the row spans >= 2 widened chunks (else a wide
                        # chunk just collapses the loop to one persistent pass, which is slower
                        # for these half-precision rows — measured at np2_n == raised).
                        if np2_n > raised:
                            loop_chunk_elems = max(loop_chunk_elems, raised)
                    loop_block = min(np2_n, _np2(loop_chunk_elems))

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
        reread_slot: int | None = None,
    ) -> list[str] | None:
        """``load_eviction_policies`` list (spec length), keyed on per-load residency;
        None leaves the autotuner default.

        - ``"stream"`` — single streamed input (``num_load == 1``: sum, long_sum), read
          once: every load -> ``'first'`` (frees L2).
        - ``"reread"`` — the row is re-read across passes: its first load -> ``'last'``
          (L2-resident), rest -> ``'first'``. ``reread_slot`` is that load's actual slot,
          resolved for the config via ``reread_eviction_slot_for_config``, not guessed.

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
    def _persistent_looped(
        cls, env: CompileEnvironment, fact: ReductionFact
    ) -> tuple[bool, int, int]:
        """The shared first lever (both tracks); returns ``(persistent, extent,
        num_warps)``. Persistent for every row the backend compiles in one pass (up to
        max_tensor_numel = 2**20 elems) AND within the per-row byte ceiling
        (ROW_PERSIST_MAX_BYTES, the register/SMEM spill limit); else loop a fixed chunk.

        The byte cap is unconditional but config-neutral off the re-read kernels (a
        single-load looped chunk ties the persistent pass; the Band-B cap is already
        tighter), so only rms_norm/layer_norm/softmax/cross_entropy/welford are steered.
        """
        from ..._utils import next_power_of_2 as _np2
        from ...runtime import get_num_sm

        # Persistent iff ALL of: the element cap (None => no cap, a compile limit), the byte
        # ceiling (a residency limit), AND — for a FULL-WIDTH-output row — a per-row ELEMENT
        # ceiling (a distinct spill limit). These are independent.
        #
        # The byte ceiling (``size_hint * itemsize``) keys on the HBM re-read footprint and is
        # correct for a SCALAR-output re-read row (cross_entropy holds only a per-row scalar
        # accumulator, so its residency is read-bound and it persists fine well past a 240 KB
        # input row — measured: CE bf16 N~98304 persistent is ~40% FASTER than looped). But a
        # FULL-WIDTH-output row (layer_norm/softmax/welford/log_softmax store the whole [M, N]
        # row) holds the fp32-promoted row tile resident to feed the store, so it spills at a
        # row WIDTH (element count), independent of the input dtype — the persist-vs-loop
        # crossover is faithfully ELEMENT-keyed (~80k elems, measured monotonic across bf16 +
        # fp32), NOT input-byte-keyed. The byte cap alone undercounts a half-precision full-width
        # T1 row (itemsize 2) 2x: log_softmax bf16 at N~98304 had a 196 KB *input* row under the
        # 240 KB byte cap, so it persisted — but its fp32 resident tile is ~384 KB and spilled
        # ~16x (2.4x slower than the looped oracle). So a full-width row also caps at
        # ``FULL_WIDTH_PERSIST_MAX_ELEMS`` elements. Gated on ``full_width_output`` so the scalar
        # re-read kernels are untouched; every existing full-width kernel reduces ``x.to(fp32)``
        # and tops out at N=16384 (rms/ln/welford) or already loops (softmax T2 at N>61440), so
        # this is a no-op for the 9 and steers only the half-precision full-width T1 rows
        # (log_softmax) onto the looped path the oracle confirms is up to 2.4x faster there.
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
            # num_sm feeds the occupancy-gated narrow-row w1 branch in _num_warps.
            num_sm = max(1, get_num_sm(env.device))
            # Persistent: T1 encodes the extent as reduction_loops=None; T2 as the full
            # pow2 R_BLOCK so the inner `for tile_n` runs once.
            return True, _np2(fact.size_hint), cls._num_warps(fact, num_sm)
        # Looped: exceeds the 2**20 or byte cap. Fixed chunk + high streaming warps.
        return False, cls.LOOPED_CHUNK, cls.LOOPED_NUM_WARPS


class TritonReductionTileHeuristic(_TritonReductionSeedBase):
    """T1 (rollable-rdim) inner-reduction seed: sum, long_sum, rms_norm, layer_norm,
    softmax-row, cross_entropy. Triton analog of ``CuteReductionTileHeuristic`` (keeps
    its registry name), deepening the original one-row/persistent/``['last']`` seed with
    the num_warps ramp, persistent-vs-looped, per-slot eviction, and a
    ``persistent_interleaved`` grid for wide looped re-read rows.

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
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # Off the H100-validated target: keep the upstream conservative seed.
            return cls._narrow_seed(env)
        from ..._utils import next_power_of_2 as _np2

        spec = env.config_spec
        fact = spec.reduction_facts[0]
        # T1 rides persistent-vs-looped on `reduction_loops`, so the lever's `extent`
        # (the T2 R_BLOCK) is unused here.
        persistent, _extent, num_warps = cls._persistent_looped(env, fact)

        # A T1 reduction may be followed by a normalize loop (e.g. `s = x.sum(); out =
        # x/s`); its extra block_sizes tile(s) are widened by _build_block_sizes. NOT
        # perf-validated (no curriculum kernel), but it is only a seed (a worse tile
        # costs autotuning time, never correctness), so emit and let the autotuner refine.
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)

        # red_block_id=None: the rdim is not a block_sizes entry, so every entry is a
        # grid axis (floored) or a normalize loop tile (widened). None loop => the single
        # grid block at its floor, as before.
        reduction_loops: list[int | None] = [None] if persistent else [cls.LOOPED_CHUNK]
        seed: dict[str, Any] = {
            "block_sizes": cls._build_block_sizes(
                spec, fact, None, None, non_reduction_loop_ids=non_reduction_loop_ids
            ),
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": 1,
            # 'flat': these reductions are grid-saturated at the M-grid. The
            # wide-looped-reread path below is the one exception.
            "pid_type": "flat",
        }
        # Eviction: streamed input -> 'first' everywhere; looped re-read -> first load
        # 'last', rest 'first'. Persistent rows stay resident, so left at default.
        evict = None
        if fact.num_load == 1:
            evict = cls._eviction_policies(env, "stream")
        elif fact.row_reread and not persistent:
            slot = device_ir.reread_eviction_slot_for_config(
                fact.reread_buffer_name, Config(**seed), env
            )
            evict = cls._eviction_policies(env, "reread", slot)
        if evict is not None:
            seed["load_eviction_policies"] = evict
        # OVERFIT NOTE: this ``persistent_interleaved`` grid is tuned for the wide-reread
        # few-long-rows class (cross_entropy at large V) and is NOT generalized — it beats
        # 'flat' there, but the sm_mult formula and maxnreg=64 are unvalidated beyond it;
        # re-validate them before relying on it elsewhere.
        # The win is GRID-TAIL QUANTIZATION (ncu-confirmed): num_sm_multiplier rounds a ragged
        # final wave to whole even waves, which only pays off at low occupancy. WS1 measured a
        # regression at high occupancy (CE bf16 V=131072 +18-22% at >=46 waves) — but an
        # occupancy gate to fix it is NOT yet safe: the interleaved-vs-flat crossover is
        # DTYPE-dependent (at 62 waves CE bf16/fp16 prefer flat by ~1.5-2.6% but CE fp32 prefers
        # interleaved by ~24%), so a flat wave threshold regresses fp32. The faithful fix needs a
        # dtype/footprint-aware boundary (occ * row_bytes, via input_load_itemsize) — deferred to
        # WS2; the regression is off the WS1 curriculum (V=131072 large-M is not a CE shape).
        # ~one resident program per row + a register cap to hide the re-read latency;
        # num_sm_multiplier sizes the grid to the row count.
        if fact.row_reread and not persistent:
            # local import: helion.runtime imports the heuristics (circular at module level).
            from ...runtime import get_num_sm

            # grid_rows = product of M-axis extents (env.size_hint needs env current).
            grid_rows = 1
            for _mbid in fact.m_block_ids:
                # pyrefly: ignore [bad-argument-type]
                grid_rows *= env.size_hint(env.block_sizes[_mbid].size)
            num_sm = max(1, get_num_sm(env.device))
            sm_mult = min(32, max(1, _np2(-(-grid_rows // num_sm))))
            seed["pid_type"] = "persistent_interleaved"
            seed["num_sm_multiplier"] = sm_mult
            seed["maxnreg"] = 64
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
      — cap the combine tile by ``STRUCTURED_COMBINE_CAP_BYTES``, widen normalize tile(s)
      separately (see ``_build_block_sizes``).

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
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            # Off sm90: upstream never fired on T2, so no prior seed to preserve. Decline.
            return None
        from ..._utils import next_power_of_2 as _np2

        spec = env.config_spec
        fact = spec.reduction_facts[0]
        persistent, extent, num_warps = cls._persistent_looped(env, fact)

        # T2: the rdim IS a block_sizes entry (no reduction_loops knob); persistent ==
        # R_BLOCK >= next_pow2(N). Other axes stay at floor (keeps Band-B M_BLOCK at 1,
        # required by the u0*u1 <= 2**20 constraint). R_BLOCK starts at the lever extent,
        # then is capped by live state (the three sub-regimes are mutually exclusive):
        r_block = extent
        non_reduction_loop_ids = set(fact.non_reduction_loop_block_ids)
        if fact.num_carried_2d_tiles >= 1:
            # Band B (kl_div, jsd): a full-N R_BLOCK over-allocates the carried 2-D tiles
            # and spills, so cap the footprint (R_BLOCK * itemsize * n_carried)
            # SM-resident. num_carried_2d_tiles routes (>=1) and sizes; max(1,..) guards 0.
            cap = cls.BANDB_R_BLOCK_BYTES // (
                max(1, fact.itemsize) * max(1, fact.num_carried_2d_tiles)
            )
            r_block = min(r_block, _np2(max(1, cap)))
        elif non_reduction_loop_ids:
            # Band C (welford, groupnorm): the combine is a serial scalar recurrence that
            # prefers persistent, so cap its tile; normalize tile(s) are widened separately
            # in _build_block_sizes.
            #
            # The cap is M_BLOCK-AWARE: the spill driver is the per-program combine
            # footprint ``M_BLOCK * tile * itemsize`` (one [M_BLOCK]-wide scalar per
            # combine stat), so the byte budget is divided by M_BLOCK. A small M_BLOCK
            # (small-M wide-N) affords a WIDER combine tile (fewer combine trips); a raised
            # M_BLOCK (huge-M) keeps the narrow validated floor. Raise-only via max(floor,
            # budget // M_BLOCK), so huge-M is byte-identical to before (the
            # welford(262144,5120) 7.3x valley is untouched). M_BLOCK is the seed's floored
            # block_sizes[0] (= the raised autotuner_min), known here before _build_block_sizes.
            itemsize = max(1, fact.itemsize)
            m_block = 1
            for mbid in fact.m_block_ids:
                m_idx = spec.block_sizes.block_id_to_index(mbid)
                m_block *= cls._block_floor(
                    cast("BlockSizeSpec", spec.block_sizes[m_idx])
                )
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
            slot = device_ir.reread_eviction_slot_for_config(
                fact.reread_buffer_name, Config(**seed), env
            )
            ev = cls._eviction_policies(env, "reread", slot)
            if ev is not None:
                seed["load_eviction_policies"] = ev
        return Config(**seed)
