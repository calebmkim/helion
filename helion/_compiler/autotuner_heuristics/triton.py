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
    from ...autotuner.config_spec import ConfigSpec
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


def _triton_reduction_eligible(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    """Gate for the Triton reduction seed (single inner reduction, no GEMM).

    Keys on the WORKLOAD invariant the seed needs: ONE registered
    ``ReductionFact`` (one inner reduction axis) and NO ``matmul_facts``. That
    single condition admits BOTH tracks (and excludes GEMMs and multi-axis manual
    reductions, which leave ``reduction_facts`` empty):

    - **T1** (rollable rdim): a ``reduction=True`` block with a
      ``ReductionLoopSpec``; the fact is built in ``register_rollable_reductions``.
    - **T2** (user-tiled / manually-looped: softmax_two_pass, kl_div, jsd): both
      ``hl.tile`` axes are ordinary ``block_sizes`` entries; the fact is built in
      ``register_user_tiled_reductions`` (guarded to fire only when T1 did not, so
      the fact count stays 1).

    We seed the M-block at its autotuner floor (see ``_block_floor``) rather than
    requiring ``<= 1``: Triton raises ``autotuner_min`` above 1 for large-M shapes
    (a search-efficiency knob, not a correctness limit), and we honor that.
    """
    spec = env.config_spec
    return len(spec.reduction_facts) == 1 and not spec.matmul_facts


class TritonReductionHeuristic(AutotunerHeuristic):
    """Band-A inner-reduction seed for the Triton backend.

    Cloned from ``cute.CuteReductionTileHeuristic`` but for ``backend="triton"``:
    drops the CuTe-only knobs (``num_threads``, ``cute_vector_widths``) and adds
    the global scalar knobs ``num_warps`` / ``num_stages``.

    Targets canonical scalar-accumulator inner reductions (sum, rms_norm,
    layer_norm, softmax-row, long_sum): the M-axis tile is the autotuner's floor
    (``_block_floor``, typically 1 row per program) so the threads cooperate on
    the contiguous last-dim reduction, with the two-pass loads fused.

    Persistent-vs-looped (the first lever, branched on the reduction extent
    ``size_hint`` = rnumel, NOT kernel identity): we seed **persistent**
    (``reduction_loops=[None]``) for every row the backend can compile as a
    single pass — up to ``max_tensor_numel`` (Triton's ``TRITON_MAX_TENSOR_NUMEL``
    = 2**20 elems, above which a whole-row ``tl.arange`` is rejected at codegen).
    Only above that *structural* ceiling do we fall back to a looped chunk.

    There is no perf byte fence below the structural cap for single-load
    reductions: a warps-held-equal sweep (``_lab/harness/v3_crossover_sweep.py``,
    num_load=1) shows persistent wins-or-ties at every feasible byte size up to
    the 2**20-elem cap. (An earlier 256 KiB byte fence was wrong — its long_sum
    "win" was entirely from ``num_warps=32``, which v3 moved into the persistent
    path; see ``_num_warps``.) The looped branch therefore has no in-sample
    coverage — it is a structural-generalization tail (see the notebook).

    ``num_warps`` scales with ``rnumel`` ALONE (a matched-pair A/B showed the w32
    win tracks rnumel for every num_load); see ``_num_warps``.
    """

    name = "triton_reduction_tile"
    backend = "triton"

    # Looped fallback chunk (pow2) for rows ABOVE the structural persistent cap
    # (rnumel > max_tensor_numel = 2**20), where a single pass cannot compile. No
    # in-sample shape reaches this; 16384 was best in the looped regime
    # (synthetic evidence only — _lab/harness/v3_crossover_sweep.py).
    LOOPED_CHUNK = 16384
    # num_warps for the LOOPED branch: w32 dominates the huge-rnumel streaming class.
    LOOPED_NUM_WARPS = 32
    # rnumel breakpoint (ELEMENTS) above which the PERSISTENT path wants max warps
    # (32). Gated on rnumel ALONE (the matched-pair A/B shows w32 is rnumel-driven
    # for every num_load). w32 dominates from rnumel=32768 up (262144/M=1: w4 47.9us
    # -> w32 11.1us). Set strictly ABOVE 16384 so sum's widest in-sample row
    # (rnumel=16384) stays byte-identical at the ramp's w16 (no regression on sum);
    # the tiny-rnumel w32 catastrophe ((32768,256) at rnumel=256) is excluded here too.
    STREAM_WARPS32_MIN_ELEMS = 16384
    # Band-B (T2 with a [M_BLOCK, R_BLOCK] 2D accumulator carried across the inner
    # loop: kl_div, jsd) R_BLOCK cap in BYTES (per accumulator). A full-N persistent
    # R_BLOCK over-allocates the live state and spills — at the widest in-sample rows
    # the persistent seed is 1.2-9.7x slower than a small looped chunk. 16 KiB
    # (= 4096 fp32 elems) is best-or-tied at every in-sample Band-B row (narrow: no
    # regression; wide: recovers the spill). In BYTES (via itemsize) so it
    # generalizes across dtypes. Scalar/row-accumulator reductions
    # (num_carried_accumulators==0) are unaffected — they stay persistent.
    BANDB_R_BLOCK_BYTES = 16384
    # Persistent byte ceiling for RE-READ reductions (gated by ``fact.row_reread``;
    # see the gate note in get_seed_config), in BYTES per row (via itemsize, dtype-
    # general). Above this a re-read reduction loops over a fixed chunk.
    #
    # Gated on re-reading the row because the crossover is num_load-dependent: a
    # matched-warps persistent-vs-looped A/B shows num_load=1 (sum) ties P==L at every
    # rnumel up to 1 MiB (so single-load stays persistent to the structural cap), but
    # num_load>=2 (rms_norm/layer_norm/cross_entropy) wins persistent only up to
    # ~256 KiB then loses 1.6-4x — the multi-load reduction re-streams the wide row
    # each pass, so a persistent kernel holding the whole row resident spills while a
    # looped chunk streams.
    #
    # CAP = 240 KiB (245760 B). A real-cross_entropy oracle + matched-lever A/B
    # (run3_ce_persist_ab.py) places the crossover between 224 KiB (persist wins ~10%,
    # beats tc) and 256 KiB (looped wins ~10%, persist spills); 240 KiB sits in that
    # dead-zone. Caps on ACTUAL row bytes (size_hint*itemsize), NOT next_pow2 — V in
    # {50304,57344,65536} all share next_pow2=65536 yet split across the crossover, so
    # a next_pow2 cap could not separate them. (The run-2 cap of 128 KiB was ~1.75x too
    # low and looped the ~50K-vocab boundary to a ~1.5-1.7x-slower seed.)
    #
    # In-sample effect: rms_norm/layer_norm/softmax_two_pass (rnumel<=16384, <=64 KiB)
    # stay persistent byte-identically; cross_entropy V in {49152,50257,50304,57344}
    # (196-229 KiB) flip looped->persistent (recover 1.08-1.68x); V>=65536 stay looped
    # (genuinely spill). num_load=1 and Band-B (tighter R_BLOCK cap dominates) unaffected.
    # SEPARATE OPEN ISSUE (notebook "P2"): at V>=98304 the looped seed is itself ~2x
    # slower than tc — a distinct lever, independent of where this boundary sits.
    MULTILOAD_PERSIST_MAX_BYTES = 245760
    # Band-C STRUCTURED-COMBINE (welford-like reduce-then-apply) tile caps, in BYTES
    # (via itemsize, dtype-general). The welford source bug (`Tn=chunk.size(-1)`
    # counted the padded tile width) is fixed (`Tn=(tile_n.index<n).sum()`), so the
    # combine tile no longer has to be a pow2 DIVISOR of N — a byte-capped tile is
    # correct at any N. The two tiles are sized by INDEPENDENT byte caps.
    #
    # COMBINE tile cap. The combine pass runs the multi-statistic (count/mean/M2)
    # combine as a SERIAL scalar recurrence, so it prefers to stay persistent
    # (single pass, next_pow2(N) tile) — looping it pays the recurrence overhead and
    # regresses. 32 KiB (= 8192 fp32) keeps it persistent across the welford
    # curriculum (all N<=8192). Well-factored 1024/2048/4096 stay byte-identical to
    # the old pow2-divisor version. At N>=32768 (out of curriculum) a persistent
    # 32 KiB combine spills and would need looping — a documented future-scope limit.
    STRUCTURED_COMBINE_CAP_BYTES = 32768
    # APPLY (normalize) tile. The apply pass (`y=(x-mean)*rstd*w+b`) carries no
    # accumulator and masks its output write (correct at any width), so this is a pure
    # perf lever: persistent next_pow2(N) when the per-row work is small, looped when
    # large. Keys on PER-ROW VALID BYTES (n_valid*itemsize), NOT next_pow2(N):
    # (262144,2560) (10 KiB/row) wants persistent 4096 while (262144,4096) (16 KiB/row,
    # same np2=4096) wants looped 2048 — a np2-byte cap could not separate them.
    # Threshold 12 KiB sits between the two.
    STRUCTURED_APPLY_PERSIST_MAX_BYTES = 12288
    # Looped-apply chunk (bytes) when per-row apply work exceeds the persist threshold.
    # 8 KiB (= 2048 fp32) is best/near-best at the wide in-sample rows. DO NOT raise
    # naively: raising to 16384 (apply 4096) was REJECTED — apply=4096 is a pathological
    # valley at large-M/N~5120 (welford(262144,5120) regressed 4-7x). A wider apply only
    # pays coupled with a bigger combine + M_block; any future raise must be gated so
    # apply stays <=2048 at the large-M/N=5120 class.
    STRUCTURED_APPLY_LOOP_CHUNK_BYTES = 8192
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    @classmethod
    def is_eligible(cls, env: CompileEnvironment, device_ir: DeviceIR) -> bool:
        return _triton_reduction_eligible(env, device_ir)

    @classmethod
    def _num_warps(cls, fact: ReductionFact) -> int:
        """Scale num_warps with the reduction extent (elems) ALONE.

        A wider row gives each warp more lane work and memory traffic to overlap;
        too few under-occupies the SM, too many waste the reduction tree.
        Power-of-2 (NumWarpsFragment requires it). Keys on ``rnumel`` ONLY (NOT
        num_load, NOT kernel identity):

            rnumel <= 1024  -> 4
            rnumel <= 4096  -> 8
            rnumel <= 16384 -> 16
            rnumel >  16384 -> 32

        The w32 step is rnumel-driven, not num_load-driven: a matched-pair A/B
        (num_load=1 vs 2, identical structure) shows both want w32 at large rnumel
        (131072: w32/w16=0.57 even for num_load=2). An earlier ``num_load==1`` fence
        was deleted — inert in-sample, false on the physics, and harmful
        out-of-sample (it denied real rms_norm/layer_norm a 30-40% w32 win). The
        tiny-rnumel w32 catastrophe ((32768,256) at rnumel=256) is excluded by the
        ``> 16384`` guard — an rnumel guard, not a num_load one.
        """
        rnumel = fact.size_hint
        if rnumel > cls.STREAM_WARPS32_MIN_ELEMS:
            return 32
        if rnumel <= 1024:
            return 4
        if rnumel <= 4096:
            return 8
        return 16

    @classmethod
    def _block_floor(cls, bs_spec: BlockSizeSpec) -> int:
        """The autotuner floor for a single block_sizes entry — the smallest
        valid block size (``>= 1``, ``>= min_size``, ``>= autotuner_min``).

        Used for every non-reduction axis the seed does not widen, including a T1
        reduction's M/row axis. Prefers 1 row per program, but honors the raised
        ``autotuner_min`` for large-M shapes rather than emitting an invalid
        ``block_size=1``.
        """
        return max(1, bs_spec.min_size, bs_spec.autotuner_min)

    @classmethod
    def _build_block_sizes(
        cls,
        spec: ConfigSpec,
        red_block_id: int,
        red_value: int,
        apply_ids: frozenset[int] | set[int] = frozenset(),
        apply_value: int | None = None,
    ) -> list[int]:
        """Build the ``block_sizes`` list for a T2 / Band-C seed: the reduction axis
        gets ``red_value``, any apply tile (Band-C) gets ``apply_value``, every other
        axis stays at its ``_block_floor``. The reduction axis is selected by INDEX,
        apply tiles by block_id (these never collide — apply_ids excludes the
        reduction block_id; see device_ir.register_user_tiled_reductions).
        """
        red_idx = spec.block_sizes.block_id_to_index(red_block_id)
        out: list[int] = []
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            if i == red_idx:
                out.append(red_value)
            elif bs_spec.block_id in apply_ids and apply_value is not None:
                out.append(apply_value)
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
        """``load_eviction_policies`` list (length == the live spec's, built EXACTLY
        — ``normalize`` does NOT length-validate it), keyed on per-load cache
        residency. Returns None to leave the autotuner default ('').

        - ``"stream"`` — a single streamed reduction input (``num_load == 1``: sum,
          long_sum) is read once and never reused, so every load -> ``'first'``
          (evict_first frees L2).
        - ``"reread"`` — the reduction-input row is re-read across >= 2 passes
          (``fact.row_reread``). The re-read ROW's FIRST load -> ``'last'`` (keep it
          L2-resident for the re-read); every other slot -> ``'first'`` (stream the
          final uses). The ``'last'`` slot is keyed on WHICH host buffer is the
          re-read row (``_compute_reread_buffer_slots``), NOT the run-2 positional
          slot 0 — which mis-placed ``'last'`` for cross_entropy (its row ``logits``
          first loads at slot 2, behind label/gather loads). A/B: CE wide-V
          1.31x/1.19x; welford 1.28x/1.27x (== the run-2 positional rule, whose row
          IS slot 0, so byte-identical there).

        Multi-load reductions with reused broadcast operands (rms_norm/layer_norm)
        pass no kind until a clean per-slot win is A/B-confirmed.
        """
        n = env.config_spec.load_eviction_policies.length
        if n <= 0:
            return None
        if kind == "stream":
            return ["first"] * n
        if kind == "reread":
            # Row's first load -> 'last' (keep L2-resident for the re-read), every
            # other slot -> 'first'. Guard: slots in range of the live spec length.
            slots = [s for s in reread_slots if 0 <= s < n]
            if not slots:
                return None
            policy = ["first"] * n
            policy[slots[0]] = "last"
            return policy
        return None

    @classmethod
    def get_seed_config(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> Config | None:
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return None
        spec = env.config_spec
        fact = spec.reduction_facts[0]

        # Persistent-vs-looped, identical for both tracks: keys on the reduction
        # extent (rnumel) and the backend's per-tile element cap, NOT the knob
        # mechanism. Persistent is the workhorse for every row the backend can
        # compile as a single pass (up to max_tensor_numel = 2**20 elems, above
        # which a whole-row `tl.arange` is rejected). Single-load reductions have no
        # perf byte fence below that cap; RE-READ reductions get an additional perf
        # ceiling (MULTILOAD_PERSIST_MAX_BYTES), because reuse of the row across the
        # reduction boundary pins it resident and a persistent seed spills once it
        # exceeds the register/SMEM budget.
        #
        # GATE = ``fact.row_reread`` — the faithful "reduction-input row is reused /
        # live across the reduction boundary" property (computed in device_ir's
        # ``_compute_row_reread`` by a consumer-dataflow trace). It is liveness, not a
        # load-op count (which over-counts cross_entropy's scalar label gather) nor a
        # ReductionLowering count (which under-counts rms_norm/layer_norm's apply-pass
        # re-read -> would exempt their wide rows -> spill). Right set: sum/long_sum
        # False; rms_norm/layer_norm/softmax/cross_entropy True; kl_div/jsd False (two
        # distinct inputs each read once; Band-B's R_BLOCK cap dominates them anyway);
        # welford True (Band-C caps dominate).
        persist_cap = env.backend.max_tensor_numel  # None ⇒ no element cap
        can_persist = persist_cap is None or fact.size_hint <= persist_cap
        if fact.row_reread:
            row_bytes = fact.size_hint * max(1, fact.itemsize)
            if row_bytes > cls.MULTILOAD_PERSIST_MAX_BYTES:
                can_persist = False
        from ..._utils import next_power_of_2 as _np2

        if can_persist:
            # Persistent (single-pass). For T1 the persistent extent is encoded as
            # reduction_loops=None; for T2 it is the full pow2 R_BLOCK so the inner
            # `for tile_n` loop runs exactly once.
            extent = _np2(fact.size_hint)
            num_warps = cls._num_warps(fact)
            persistent = True
        else:
            # Looped (chunked) reduction, reached EITHER because the row exceeds the
            # 2**20 structural cap (a single pass cannot compile — a structural tail,
            # no in-sample coverage) OR because a re-read row exceeds
            # MULTILOAD_PERSIST_MAX_BYTES (the cross_entropy perf crossover; fires
            # in-sample for V=131072). Either way: a fixed chunk + high streaming warps.
            extent = cls.LOOPED_CHUNK
            num_warps = cls.LOOPED_NUM_WARPS
            persistent = False

        is_t1 = fact.block_id in spec.reduction_loops.valid_block_ids()
        if is_t1:
            # T1 (rollable rdim): the persistent-vs-looped choice rides on the
            # `reduction_loops` knob (None = persistent). The single M-block is
            # seeded at its autotuner floor.
            reduction_loops: list[int | None] = (
                [None] if persistent else [cls.LOOPED_CHUNK]
            )
            seed: dict[str, Any] = {
                "block_sizes": [
                    cls._block_floor(cast("BlockSizeSpec", spec.block_sizes[0]))
                ],
                "reduction_loops": reduction_loops,
                "num_warps": num_warps,
                "num_stages": 1,
                # 'flat' is the principled default: persistent/narrow-looped
                # reductions are grid-saturated at the M-grid (a matched-lever A/B
                # showed flat dominates 1.5-4x). EDIT-PID below is the one measured
                # exception (wide looped re-read rows), where it overrides this.
                "pid_type": "flat",
            }
            # Eviction (keyed on the faithful re-read property):
            #  - num_load==1 (sum/long_sum): single streamed input -> 'first' everywhere.
            #  - row_reread AND LOOPED (cross_entropy wide-V: logits re-read across the
            #    amax + exp-sum passes, streamed from HBM each pass): the re-read
            #    buffer's first load -> 'last', rest -> 'first'. CE wide-V 1.31x/1.19x.
            #    Gated on `not persistent` because eviction only affects HBM-streamed
            #    loads — a persistent row is register/SMEM-resident across the passes
            #    (A/B confirms eviction neutral on the persistent CE boundary), so this
            #    leaves at-floor re-read kernels byte-identical.
            #  - else: default.
            evict = None
            if fact.num_load == 1:
                evict = cls._eviction_policies(env, "stream")
            elif fact.row_reread and not persistent:
                evict = cls._eviction_policies(env, "reread", fact.reread_buffer_slots)
            if evict is not None:
                seed["load_eviction_policies"] = evict
            # EDIT-PID: a wide LOOPED RE-READ T1 reduction (gate identical to the
            # reread-eviction above) is a FEW-LONG-ROWS workload — M=2048-8192 rows
            # (<< the machine's program capacity) each grinding a heavy looped
            # multi-pass re-read. A ``persistent_interleaved`` grid of
            # ``get_num_sm * num_sm_multiplier`` resident programs + a register cap
            # beats the default 'flat' here (A/B vs flat: CE 1.23x/1.25x/1.05x).
            # PERSISTENT rows stay 'flat'; welford (Band-C) and softmax (T2) take
            # other branches, so only T1 fires.
            #   - num_sm_multiplier: physics-derived from the workload (M rows) and
            #     hardware (SM count), NOT the oracle's value — size the grid to ~= the
            #     row count so each row maps to ~one resident program =
            #     clamp(np2(ceil(M / num_sm)), 1, 32). Degenerate M=1 -> 1 (a measured
            #     tie). The oracle's own pick varies (32/32/4), so a derived M-scaled
            #     value is more principled than a constant.
            #   - maxnreg=64: a high-occupancy register cap to hide the looped-reread
            #     memory latency. Load-bearing — isolating it, the gain ~halves.
            if fact.row_reread and not persistent:
                # local import: ``helion.runtime`` imports the heuristics, so a
                # module-level import here is circular.
                from ...runtime import get_num_sm

                # grid_rows = product of the M-axis extents (the rows the flat grid
                # launches), via env.size_hint (BlockSizeInfo.size_hint() would need
                # env to be the current context, not guaranteed at seed-emit).
                grid_rows = 1
                for _mbid in fact.m_block_ids:
                    grid_rows *= env.size_hint(env.block_sizes[_mbid].size)
                num_sm = max(1, get_num_sm(env.device))
                sm_mult = min(32, max(1, _np2(-(-grid_rows // num_sm))))
                seed["pid_type"] = "persistent_interleaved"
                seed["num_sm_multiplier"] = sm_mult
                seed["maxnreg"] = 64
            return Config(**seed)

        if fact.is_structured_combine:
            # BAND C: a two-pass STRUCTURED COMBINE (welford-like reduce-then-apply).
            # The same inner extent is tiled by a combine pass (the reduction axis,
            # carrying per-row count/mean/M2 as a scalar recurrence) AND one-or-more
            # apply/normalize passes (fact.apply_block_ids, no reduction). A
            # single-axis seed would floor the apply tile(s) to width 1 (~10-20x
            # slower), so we widen them. The two tiles are sized by independent byte
            # caps (STRUCTURED_* constants), NOT N's factorization — the welford source
            # bug is fixed (masked valid count, not the constexpr tile width), so a
            # byte-capped tile is correct at any N (the old pow2-divisor constraint is
            # gone).
            np2_n = _np2(fact.size_hint)
            n_valid = (
                fact.static_rnumel if fact.static_rnumel is not None else fact.size_hint
            )
            itemsize = max(1, fact.itemsize)
            # Combine tile: full next_pow2(N) up to the spill-safe byte cap; looped above.
            combine_cap_elems = max(1, cls.STRUCTURED_COMBINE_CAP_BYTES // itemsize)
            combine_block = min(np2_n, _np2(combine_cap_elems))
            # Apply (normalize) tile: persistent (single masked pass) when per-row valid
            # work is small, looped (capped chunk) when large; keys on per-row valid bytes.
            if n_valid * itemsize <= cls.STRUCTURED_APPLY_PERSIST_MAX_BYTES:
                apply_block = np2_n
            else:
                apply_cap_elems = max(
                    1, cls.STRUCTURED_APPLY_LOOP_CHUNK_BYTES // itemsize
                )
                apply_block = min(np2_n, _np2(apply_cap_elems))
            sc_seed: dict[str, Any] = {
                "block_sizes": cls._build_block_sizes(
                    spec,
                    fact.block_id,
                    combine_block,
                    apply_ids=set(fact.apply_block_ids),
                    apply_value=apply_block,
                ),
                "num_warps": num_warps,
                "num_stages": 1,
                "pid_type": "flat",  # principled constant — see the T1 branch.
            }
            # Eviction: the structured combine re-reads the reduction input (x in the
            # combine pass, re-read in the apply pass) -> 'last' on x's first load,
            # 'first' on its re-reads. The 'last' slot is keyed on which host buffer is
            # re-read (provenance); for welford that IS slot 0, so byte-identical to the
            # run-2 positional win but faithful.
            sc_evict = cls._eviction_policies(env, "reread", fact.reread_buffer_slots)
            if sc_evict is not None:
                sc_seed["load_eviction_policies"] = sc_evict
            return Config(**sc_seed)

        # T2 (user-tiled / manually-looped): the reduction axis IS a block_sizes
        # entry (the inner `hl.tile(n, block_size=R_BLOCK)`); there is no
        # `reduction_loops` knob. Persistent == R_BLOCK >= next_pow2(N) so the inner
        # loop runs once. Every other block_size (the grid/row axes) stays at its
        # floor — for the Band-B loss kernels this keeps M_BLOCK at 1, required by the
        # u0*u1 <= 2**20 numel constraint.
        r_block = extent
        if fact.num_carried_accumulators >= 1:
            # BAND B: this T2 reduction carries one-or-more [M_BLOCK, R_BLOCK] 2D
            # accumulators across the inner loop (kl_div: loss_sum; jsd:
            # intermediate_loss + intermediate_dX). A full-N persistent R_BLOCK
            # over-allocates that live state and SPILLS (the persistent seed is
            # 1.2-9.7x slower than a small looped chunk at the widest in-sample rows).
            # Cap R_BLOCK so the live footprint stays SM-resident: the loop holds
            # num_carried_accumulators tiles at the same R_BLOCK, so the footprint is
            # R_BLOCK * itemsize * n_carried; hold that to BANDB_R_BLOCK_BYTES =>
            # R_BLOCK <= budget / (itemsize * n_carried).
            #
            # num_carried_accumulators is the SINGLE Band-B signal — it both ROUTES
            # (>= 1 here) and SIZES (the divisor). It is the faithful count of [M,R]
            # tiles genuinely resident across the loop: jsd carries 2 -> 2048, kl_div
            # carries 1 (its [M,R] kl_loss is in-loop scratch, excluded) -> 4096. NOT a
            # raw ReductionLowering count (the rejected ÷nro proxy, which equals the
            # carried count only under a 1:1 reduction<->accumulator structure). Gated
            # on the workload property, not kernel identity; jsd is the sole firer by
            # curriculum incidence. Scalar/row-accumulator T2 (softmax_two_pass) ->
            # carried==0 -> stays persistent. max(1, ...) is a 0-divide guard.
            bandb_cap = max(
                1,
                cls.BANDB_R_BLOCK_BYTES
                // (max(1, fact.itemsize) * max(1, fact.num_carried_accumulators)),
            )
            r_block = min(r_block, _np2(bandb_cap))

        seed = {
            "block_sizes": cls._build_block_sizes(spec, fact.block_id, r_block),
            "num_warps": num_warps,
            "num_stages": 1,
            "pid_type": "flat",  # principled constant — see the T1 branch.
        }
        # EDIT#6: the same faithful reread-eviction as T1/Band-C, on the T2 plain path.
        # A T2 reduction that re-reads its row across passes and runs looped
        # (softmax_two_pass: x re-read in the max + exp-sum passes; wide rows) wants the
        # row L2-resident: 'last' on its first load, 'first' elsewhere. Same gate as the
        # T1 eviction. softmax (1024,65536) 1.36x / (512,131072) 1.10x. kl_div/jsd are
        # row_reread=False -> unaffected; narrow/persistent softmax stays byte-identical.
        if fact.row_reread and not persistent:
            ev = cls._eviction_policies(env, "reread", fact.reread_buffer_slots)
            if ev is not None:
                seed["load_eviction_policies"] = ev
        return Config(**seed)
