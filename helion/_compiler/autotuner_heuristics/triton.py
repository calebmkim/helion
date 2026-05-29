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

    # Persistent-reduction ceiling for a contiguous row, in BYTES (NOT
    # hardcoded to fp32 — multiplied by the ReductionFact's itemsize so it
    # generalizes to bf16/fp16). Below it we keep the reduction single-pass
    # (persistent); above it a looped chunk wins.
    #
    # EVIDENCE (synthetic persistent-vs-looped crossover sweep on H100/fp32,
    # rms_norm_fwd, best-persistent vs best-looped over warps/stages, median
    # do_bench — _lab/harness/crossover_sweep.py):
    #   rnumel(elem) | KiB | M=8 pers/loop | M=1024 | M=4096   (>1 ⇒ looped wins)
    #     16384      |  64 |    0.95 pers  |  0.98  |  0.98
    #     32768      | 128 |    0.97 pers  |  0.80  |  0.78
    #     49152      | 192 |    0.96 pers  |  0.76  |  0.74
    #     65536      | 256 |    1.06 LOOP  |  0.86  |  0.87   ← occupied still persist
    #     98304      | 384 |    2.15 LOOP  |  2.94  |  3.07   ← looped wins big
    #    131072      | 512 |    3.10 LOOP  |  3.45  |  3.67
    #    262144      |1024 |    2.93 LOOP  |  3.60  |  3.68
    # So persistent keeps winning to ~256 KiB/row for grid-OCCUPIED shapes
    # (M≥1024) and to ~192 KiB for grid-starved tiny-M; the crossover is
    # ~64–256 KiB, grid-occupancy-modulated. We pick the OCCUPIED crossover
    # (256 KiB = 65536 fp32 elems): rnumel ≤ that → persistent (right for the
    # common case; tiny-M loses only ~6% exactly at 65536); rnumel > that →
    # looped (where looped wins 2–3.7×). This is ~4× higher than the prior
    # 65536-BYTE fence (=16384 elems), which was set at the in-sample max row
    # and made the looped branch fire ~4× too early.
    PERSIST_MAX_BYTES = 262144  # 256 KiB; 65536 fp32 elems
    # Looped fallback chunk for rows above the persistent ceiling (power of 2).
    # EVIDENCE (_lab/harness/looped_chunk_probe.py, looped-winning region): in
    # the looped regime a LARGER R_BLOCK keeps the reduction efficient — per
    # chunk best-us at M=8/rnumel=131072: 2048→36.4, 4096→25.1, 8192→23.5,
    # 16384→22.0; same ordering at M=4/8/16 and M=1024. 16384 beats the old
    # 4096 by ~15–25%. (Chunk is always < rnumel here since looped only fires
    # above 65536 elems.)
    LOOPED_CHUNK = 16384
    # num_warps for the LOOPED branch (huge rnumel). EVIDENCE: in the looped
    # region warps=32 dominates for the (tiny-M, huge-rnumel) long_sum regime
    # (best (w,s) was (32,1) at essentially every looped case in the probe) —
    # few programs ⇒ each must extract max ILP/parallelism over the long row.
    # The persistent branch keeps the lower rnumel-scaled warps (_num_warps).
    LOOPED_NUM_WARPS = 32
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
            # num_warps scales with the (bounded) reduction extent.
            reduction_loops: list[int | None] = [None]
            num_warps = cls._num_warps(fact)
        else:
            # Looped reduction over a fixed R_BLOCK chunk. The row is too large
            # for a single persistent pass; loop over LOOPED_CHUNK-element
            # chunks. Here the row is huge, so use the high looped warp count
            # (see LOOPED_NUM_WARPS) rather than the persistent rnumel ramp.
            reduction_loops = [cls.LOOPED_CHUNK]
            num_warps = cls.LOOPED_NUM_WARPS
        seed: dict[str, Any] = {
            "block_sizes": [cls._m_block_size(env)],
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": 1,
        }
        return Config(**seed)
