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

    ``num_warps`` scales with the reduction extent AND ``num_load`` (a workload
    arith-intensity fact, NEVER kernel identity): single-stream reductions
    (``num_load==1``: sum, long_sum) ramp up to 32 warps at huge rnumel where
    the persistent pass is purely bandwidth-bound; re-reading reductions
    (``num_load==2``: rms_norm) do NOT — high warps hurt them.  See
    ``_num_warps``.
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
    # rnumel breakpoint (in ELEMENTS) above which a single-stream (num_load==1)
    # reduction wants the maximum warp count (32) in the PERSISTENT path. EVIDENCE
    # (_lab/harness/v3_persist_warps_ramp.py, sum_kernel persistent path, best
    # num_warps per rnumel): w32 dominates from rnumel=32768 up (e.g. rnumel
    # 262144/M=1: w4=47.9us → w32=11.1us = 4.3×). The breakpoint is set STRICTLY
    # ABOVE 16384 so that sum's max in-sample row (rnumel=16384) is BYTE-FOR-BYTE
    # unchanged (stays at the _num_warps ramp's w16) — no-regression on sum.
    STREAM_WARPS32_MIN_ELEMS = 16384
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return _triton_reduction_eligible(env, device_ir)

    @classmethod
    def _num_warps(cls, fact: ReductionFact) -> int:
        """Scale num_warps with the reduction extent (elems) AND ``num_load``.

        A wider row gives each warp more independent lane work and more memory
        traffic to overlap; too few warps under-occupies the SM, too many waste
        the cross-lane reduction tree. Power-of-2 (NumWarpsFragment requires it).

        Two regimes, keyed on the WORKLOAD fact ``num_load`` (arith intensity),
        NOT on kernel identity:

        - ``num_load == 1`` (single-stream reductions: sum, long_sum): the
          persistent pass is purely bandwidth-bound on a huge row, so the warp
          count ramps all the way to 32. EVIDENCE
          (_lab/harness/v3_persist_warps_ramp.py): w32 best from rnumel 32768 up
          (262144/M=1: w4 47.9us → w32 11.1us). The w32 step sits STRICTLY above
          16384 so sum's max in-sample row (16384) is unchanged.
        - ``num_load >= 2`` (re-reading reductions: rms_norm reloads x for the
          normalize pass): high warps HURT. EVIDENCE
          (_lab/harness/v3_rmsnorm_warps_ab.py): rms_norm NEVER prefers w32 — at
          (32768,256) w16=574us and w32=1182us (catastrophic at large-M/tiny-N
          where warps couple badly with the small M-block); at its wide rows
          warps barely matter. So multi-load reductions keep the conservative
          v1 ramp (caps at 16), preserving the rms_norm champion.

        This is the generalizable distinction the v2 auditor demanded: the
        warps=32 win is a num_load=1 property, not a generic huge-rnumel one.
        """
        rnumel = fact.size_hint
        if fact.num_load <= 1 and rnumel > cls.STREAM_WARPS32_MIN_ELEMS:
            # Single-stream, huge row: bandwidth-bound persistent pass wants the
            # max warp count to keep the SM fed and amortize the reduction tree.
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
            # num_warps scales with rnumel AND num_load (see _num_warps).
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
