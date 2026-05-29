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
      ``min(next_pow2(rnumel), 4096)`` once ``rnumel > 4096``.  Empirically
      (H100/fp32) that looped default LOSES big at wide rows because
      tc/Helion-max keep the reduction **persistent** (whole contiguous row in
      registers/SMEM, single pass, no roffset loop).
    - So we seed **persistent** (``reduction_loops=[None]``) for EVERY row the
      Triton backend can actually compile as a persistent reduction — i.e. up to
      the backend's per-tile element cap ``max_tensor_numel`` (Triton's hard
      ``TRITON_MAX_TENSOR_NUMEL`` = 2**20 elems; above that ``tl.arange`` over
      the whole row is rejected at codegen).  Only above that *structural*
      ceiling do we fall back to a looped chunk — that is the ONLY regime where
      persistent is not even an option.

    WHY no byte-based persist ceiling below the structural cap (the v2→v3 fix):
    a warps-HELD-EQUAL synthetic sweep (``_lab/harness/v3_crossover_sweep.py``,
    sum_kernel = num_load=1 = same class as long_sum, persistent vs looped both
    at warps∈{16,32}) shows **persistent wins or ties at every feasible byte
    size** up to the 1 MiB / 2**20-elem structural cap:

        rnumel | KiB  | bestP/bestL (>1 ⇒ looped wins)         | verdict
        131072 |  512 | 0.92–1.00 across M∈{1,4,16,64,256}     | PERSISTENT
        262144 | 1024 | 0.87–1.01                              | PERSISTENT
        393216 | 1536 | 0.52–0.97                              | PERSISTENT
        524288 | 2048 | 1.20–1.22 at M≤16, ~1.0 at M≥64        | (looped@small-M*)
        786432 | 3072 | 0.54–1.08                              | PERSISTENT
       1048576 | 4096 | 1.00–1.10 (noisy ~tie)                 | PERSISTENT(cap)
      >1048576 |      | persistent FAILS to compile            | LOOPED ONLY

    The v2 byte fence (256 KiB = ``PERSIST_MAX_BYTES``) was WRONG: it sent every
    in-sample long_sum row (128 KiB–1 MiB) to the looped path, but a controlled
    A/B (``AUDIT_seed_vs_p32.py`` / reproduced here) shows persistent/w32 BEATS
    the v2 looped/w32 seed on ALL 5 in-sample long_sum shapes (1.01–1.16×).  The
    v2 long_sum "win" was entirely from **num_warps=32**, not the looped or
    grid-occupancy branches — those were net-harmful and effectively fenced
    long_sum's shapes.  v3 deletes both and moves the warps=32 lever into the
    PERSISTENT path (see ``_num_warps``).

    (* the isolated 524288/small-M looped win is real but non-monotone — it is
    surrounded by persistent-wins at 393216 and 786432 — so it is NOT keyed; see
    the notebook "looped tail" disclosure.  No in-sample shape reaches even the
    structural cap, so the looped branch has NO in-sample coverage — it is a
    synthetic/structural generalization tail, disclosed in the notebook.)

    ``num_warps`` scales with the reduction extent (``rnumel``) ALONE: at huge
    rnumel the persistent pass is bandwidth-bound and ramps up to 32 warps,
    regardless of ``num_load``.  A matched-pair A/B (v4) showed the w32 win
    tracks rnumel for both num_load==1 and num_load>=2; the earlier num_load
    fence was inert in-sample, false on the physics, and harmful out-of-sample,
    so it was deleted.  See ``_num_warps``.
    """

    name = "triton_reduction_tile"
    backend = "triton"

    # Looped fallback chunk for rows ABOVE the structural persistent cap (power
    # of 2). Only reached for rnumel > max_tensor_numel (2**20 elems), i.e. when
    # a single persistent pass cannot compile at all. No in-sample shape reaches
    # this; the chunk is set by the v2 looped_chunk_probe (16384 best in the
    # looped regime) and re-confirmed adequate for the >1 MiB rows in
    # _lab/harness/v3_crossover_sweep.py. DISCLOSURE: synthetic-evidence-only.
    LOOPED_CHUNK = 16384
    # num_warps for the LOOPED branch (huge rnumel beyond the persistent cap).
    # In that regime warps=32 dominates (long_sum num_load=1 streaming class).
    LOOPED_NUM_WARPS = 32
    # rnumel breakpoint (in ELEMENTS) above which a reduction wants the maximum
    # warp count (32) in the PERSISTENT path. Gated on rnumel ALONE (NOT
    # num_load — see _num_warps; the matched-pair A/B shows w32 is rnumel-driven
    # for every num_load). EVIDENCE (_lab/harness/v3_persist_warps_ramp.py, best
    # num_warps per rnumel): w32 dominates from rnumel=32768 up (e.g. rnumel
    # 262144/M=1: w4=47.9us → w32=11.1us = 4.3×). The breakpoint is set STRICTLY
    # ABOVE 16384 so that sum's max in-sample row (rnumel=16384) is BYTE-FOR-BYTE
    # unchanged (stays at the _num_warps ramp's w16) — no-regression on sum. The
    # tiny-rnumel w32 catastrophe ((32768,256): w16=570us->w32=1174us) is at
    # rnumel=256, already excluded by this guard — an rnumel guard, not num_load.
    STREAM_WARPS32_MIN_ELEMS = 16384
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return _triton_reduction_eligible(env, device_ir)

    @classmethod
    def _num_warps(cls, fact: ReductionFact) -> int:
        """Scale num_warps with the reduction extent (elems) ALONE.

        A wider row gives each warp more independent lane work and more memory
        traffic to overlap; too few warps under-occupies the SM, too many waste
        the cross-lane reduction tree. Power-of-2 (NumWarpsFragment requires it).

        The ramp keys on ``rnumel`` ONLY (NOT num_load, NOT kernel identity):

            rnumel <= 1024  -> 4
            rnumel <= 4096  -> 8
            rnumel <= 16384 -> 16
            rnumel >  16384 -> 32

        EVIDENCE that the w32 step is rnumel-driven, not num_load-driven:

        - [v4] A matched-pair synthetic A/B (num_load=1 vs num_load=2, IDENTICAL
          structure) shows the w32 benefit tracks ``rnumel`` for BOTH: at
          rnumel=131072 w32/w16=0.57 even for num_load=2. So num_load is NOT the
          lever — the persistent pass is bandwidth-bound on a huge row
          regardless of how many times each element is read.
          (_lab/harness/AUDITOR_numload_warps_ab.py)
        - The persistent w32 ramp (single-stream sum/long_sum) is confirmed in
          _lab/harness/v3_persist_warps_ramp.py: w32 best from rnumel 32768 up
          (262144/M=1: w4 47.9us → w32 11.1us).
        - Real multi-load reductions ALSO want w32 at large rnumel: real
          rms_norm (num_load=2) (1,131072) w32/w16=0.725, and layer_norm
          (num_load=3) (1,131072) is ~34% slower at w16 than w32.
          (_lab/harness/AUDITOR_rmsnorm_largeN_warps.py)

        WHY the earlier num_load fence was deleted (v3 -> v4): the
        ``num_load==1`` condition was a curriculum-split fence dressed as
        physics. It was INERT in-sample (no in-sample num_load>=2 kernel has
        rnumel>16384, so the condition never fired — 0/27 seeds change if you
        drop it; AUDITOR_gate_inert_proof.py), FALSE on the matched-pair physics
        above, and HARMFUL out-of-sample (it denied real rms_norm/layer_norm a
        real 30-40% w32 win at large rnumel). The tiny-rnumel w32 catastrophe
        ((32768,256): w16=570us -> w32=1174us) is at rnumel=256, already
        excluded by the ``> 16384`` guard — an rnumel guard, not a num_load one.
        """
        rnumel = fact.size_hint
        if rnumel > cls.STREAM_WARPS32_MIN_ELEMS:
            # Huge row: the bandwidth-bound persistent pass wants the max warp
            # count to keep the SM fed and amortize the reduction tree. This is
            # gated on rnumel ALONE — the w32 win is driven by the reduction
            # extent, NOT by num_load. A matched-pair A/B (num_load=1 vs
            # num_load=2, identical structure) shows BOTH want w32 at large
            # rnumel (rnumel=131072: w32/w16=0.57 even for num_load=2). The
            # tiny-rnumel w32 catastrophe (e.g. (32768,256): w16=570us ->
            # w32=1174us) is at rnumel=256, already excluded by `> 16384` — it
            # is an rnumel guard, not a num_load one.
            return 32
        if rnumel <= 1024:
            return 4
        if rnumel <= 4096:
            return 8
        return 16

    @classmethod
    def _m_extent(cls, env: CompileEnvironment, fact: ReductionFact) -> int:
        """Total non-reduction (row) extent = the M-axis grid size.

        Product of the kept (non-reduction) tile size_hints. With the M-block at
        its floor (~1 row/program) this is the number of programs launched along
        the row axis = how much of the GPU the kernel fills.

        DIAGNOSTIC ONLY in v3: no branch keys on it anymore. The v2
        grid-occupancy branch that used it was DELETED — its premise ("looped
        wins at small M") was a confound (it compared persistent/w16 vs
        looped/w32; at equal warps persistent wins). Kept for trace/audit
        scripts that report grid size.
        """
        spec = env.config_spec
        extent = 1
        for mid in fact.m_block_ids:
            bs = spec.block_sizes.block_id_lookup(mid)
            extent *= bs.size_hint
        return extent

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
        # Persistent is the workhorse for EVERY row the backend can compile as a
        # single pass. The ONLY structural limit is the backend's per-tile
        # element cap (Triton's max_tensor_numel = 2**20 elems); above it the
        # whole-row `tl.arange` is rejected at codegen, so we MUST loop. There is
        # no perf-based byte ceiling below the cap: a warps-held-equal sweep
        # (_lab/harness/v3_crossover_sweep.py) shows persistent wins or ties at
        # every feasible byte size up to the cap. (The v2 byte fence + grid-
        # occupancy branch were a confound — see the class docstring.)
        persist_cap = env.backend.max_tensor_numel  # None ⇒ no element cap
        can_persist = persist_cap is None or fact.size_hint <= persist_cap
        if can_persist:
            # Persistent (single-pass) reduction. normalize() realizes None as
            # the full power-of-2 extent at codegen (no `for roffset` loop).
            # num_warps scales with rnumel ALONE (see _num_warps).
            reduction_loops: list[int | None] = [None]
            num_warps = cls._num_warps(fact)
        else:
            # Row exceeds the backend's persistent element cap — a single pass
            # cannot compile, so loop over a fixed R_BLOCK chunk with the high
            # streaming warp count. NOTE: no in-sample shape reaches this cap;
            # this branch is a synthetic/structural generalization tail (see the
            # notebook "looped tail" disclosure).
            reduction_loops = [cls.LOOPED_CHUNK]
            num_warps = cls.LOOPED_NUM_WARPS
        seed: dict[str, Any] = {
            "block_sizes": [cls._m_block_size(env)],
            "reduction_loops": reduction_loops,
            "num_warps": num_warps,
            "num_stages": 1,
        }
        return Config(**seed)
