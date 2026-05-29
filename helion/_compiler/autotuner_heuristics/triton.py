from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from ...runtime.config import Config
from .common import clamp_block_size_targets
from .common import matches_hardware
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ...autotuner.config_spec import BlockSizeSpec
    from ...autotuner.config_spec import ReductionFact
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


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


def _triton_reduction_eligible(
    env: CompileEnvironment, device_ir: DeviceIR
) -> bool:
    """Gate for the Triton Band-A reduction seed (T1 rollable, single rdim).

    Mirrors the CuTe template's structural gate: single non-reduction tile +
    single reduction dim, no matmul facts (this seeds reductions, not GEMMs),
    and a populated ReductionFact for the rdim.

    Unlike the CuTe template we do NOT additionally require the M-axis floor to
    be ``<= 1``. Triton's autotuner raises ``autotuner_min`` to 2+ for LARGE-M
    shapes (``raise_grid_block_minimums``: a tiny block on a 32768-row axis
    makes an enormous grid) — that floor is an autotuner-search-efficiency knob,
    NOT a correctness limit on block=1. We accept any floor and seed the M-block
    AT that floor (see ``_m_block_size``) rather than forcing 1, which is what
    lets the small-N / large-M shapes (e.g. 32768x256) get a reduction seed
    instead of being silently skipped.

    NOTE: ``len(reduction_loops)==1`` matches T1 (rollable rdim) ONLY. T2
    manual-tile reductions (softmax_two_pass, kl_div, jsd) are a ``block_sizes``
    entry with ``reduction=True`` and are NOT in ``reduction_loops`` — this gate
    must be BROADENED for those later (Band B / T2).
    """
    spec = env.config_spec
    return (
        len(spec.block_sizes) == 1
        and len(spec.reduction_loops) == 1
        and len(spec.reduction_facts) == 1
        and not spec.matmul_facts
    )


class TritonReductionHeuristic(AutotunerHeuristic):
    """Band-A inner-reduction seed for the Triton backend.

    Cloned from ``cute.CuteReductionTileHeuristic`` but for ``backend="triton"``:
    drops the CuTe-only knobs (``num_threads``, ``cute_vector_widths``) and adds
    the global scalar knobs ``num_warps`` / ``num_stages`` (the CuTe template
    left those to the spec default).

    The seed targets canonical scalar-accumulator inner reductions (sum,
    rms_norm, layer_norm, softmax-row, long_sum): the M-axis tile is the
    autotuner's floor (``_m_block_size``, typically 1 row per program) so the
    threads cooperate on the contiguous last-dim reduction, with the two-pass
    loads fused so x isn't reloaded.

    Persistent-vs-looped (the first lever, branched on the WORKLOAD's reduction
    extent ``size_hint`` = rnumel, NOT kernel identity):

    - The un-seeded Triton ``default_config`` goes **looped** with chunk
      ``min(next_pow2(rnumel), 4096)`` once ``rnumel > reduction_loop_force_
      threshold`` (None on Triton ⇒ effectively persistent up to 4096, looped
      4096 above). Empirically (rms_norm fwd, fp32, H100) that looped default
      LOSES ~23-25% to torch.compile-default / Helion-max at rnumel ∈
      {8192, 16384} because tc/Helion-max keep the reduction **persistent**
      (whole contiguous row in registers/SMEM, single pass, no roffset loop).
    - So we seed **persistent** (``reduction_loops=[None]``) whenever the row
      plausibly fits a persistent reduction, i.e.
      ``rnumel * itemsize <= PERSIST_MAX_BYTES``; above that we fall back to a
      looped chunk. The threshold is in BYTES (via ``itemsize``) so it
      generalizes across dtypes — see the constant.

    ``num_warps`` scales with the reduction extent: more independent lane work
    ⇒ more warps amortize the cross-lane reduction tree and keep the SMs fed.
    """

    name = "triton_reduction_tile"
    backend = "triton"

    # Persistent-reduction ceiling for a single fp32 contiguous row, in
    # ELEMENTS. Above this the row no longer fits a single-pass persistent
    # reduction comfortably and a looped chunk wins. Starting value 16384 (=
    # 64 KiB at fp32) — chosen so the rms_norm in-sample range (rnumel up to
    # 16384) stays persistent, matching what tc/Helion-max do. This is the
    # primary lever to A/B against the oracle field-diff; expressed in BYTES
    # via itemsize so it generalizes across dtypes (NOT hardcoded to fp32).
    PERSIST_MAX_BYTES = 65536
    # Looped fallback chunk for rows above the persistent ceiling (power of 2).
    LOOPED_CHUNK = 4096
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return _triton_reduction_eligible(env, device_ir)

    @classmethod
    def _num_warps(cls, fact: ReductionFact) -> int:
        """Scale num_warps with the reduction extent (in elements).

        A wider row gives each warp more independent lane work and more memory
        traffic to overlap; too few warps under-occupies the SM, too many waste
        the cross-lane reduction tree. These breakpoints are the spec default
        (4) for small rows, stepping up for the wide rows where the persistent
        single-pass reduction is bandwidth-bound. Power-of-2 (NumWarpsFragment
        requires it). To be A/B'd against the oracle's num_warps.
        """
        rnumel = fact.size_hint
        if rnumel <= 1024:
            return 4
        if rnumel <= 4096:
            return 8
        return 16

    @classmethod
    def _m_block_size(cls, env: CompileEnvironment) -> int:
        """M-axis (non-reduction) block size = the autotuner's floor.

        Prefer 1 row per program (so the reduction recruits the whole block's
        threads). But for large-M shapes the autotuner raises ``autotuner_min``
        above 1; we honor that floor (it's a valid block size and keeps the grid
        sane) rather than emitting an invalid block_size=1.
        """
        bs_spec = cast("BlockSizeSpec", env.config_spec.block_sizes[0])
        return max(1, bs_spec.min_size, bs_spec.autotuner_min)

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return None
        spec = env.config_spec
        fact = spec.reduction_facts[0]
        rnumel_bytes = fact.size_hint * fact.itemsize
        if rnumel_bytes <= cls.PERSIST_MAX_BYTES:
            # Persistent (single-pass) reduction. normalize() realizes None as
            # the full power-of-2 extent at codegen (no `for roffset` loop).
            reduction_loops: list[int | None] = [None]
        else:
            reduction_loops = [cls.LOOPED_CHUNK]
        seed: dict[str, Any] = {
            "block_sizes": [cls._m_block_size(env)],
            "reduction_loops": reduction_loops,
            "num_warps": cls._num_warps(fact),
            "num_stages": 1,
        }
        return Config(**seed)
