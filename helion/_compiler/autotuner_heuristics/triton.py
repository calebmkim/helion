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


def _triton_reduction_eligible(env: CompileEnvironment, device_ir: DeviceIR) -> bool:
    """Gate for the Triton reduction seed (single inner reduction, no GEMM).

    Keys on exactly the WORKLOAD invariant the seed needs: ONE registered
    ``ReductionFact`` (one inner reduction axis) and NO ``matmul_facts`` (this
    seeds reductions, not GEMMs). That single condition admits BOTH tracks:

    - **T1** (rollable rdim): a ``reduction=True`` block with a
      ``ReductionLoopSpec``; the fact is built in ``register_rollable_reductions``.
      Here ``len(reduction_loops)==1`` and ``len(block_sizes)==1``.
    - **T2** (user-tiled / manually-looped: softmax_two_pass, kl_div, jsd): BOTH
      ``hl.tile`` axes are ordinary ``block_sizes`` entries (no ``reduction=True``
      block, ``reduction_loops`` empty); the fact is built in
      ``register_user_tiled_reductions`` (guarded so it only fires when T1 did
      not, keeping the fact count at 1). Here ``len(block_sizes)>=2`` and
      ``len(reduction_loops)==0``.

    The earlier ``len(block_sizes)==1 and len(reduction_loops)==1`` conditions
    were a T1-ONLY shape signature; they EXCLUDED T2 (which has 2 block_sizes / 0
    reduction_loops). Replacing them with ``len(reduction_facts)==1`` generalizes
    on the workload (a single inner reduction) rather than on the rolling
    mechanism, while still excluding GEMMs and multi-axis manual reductions
    (which leave ``reduction_facts`` at 0).

    We do NOT require the M-axis floor to be ``<= 1`` (the CuTe template did).
    Triton's autotuner raises ``autotuner_min`` to 2+ for LARGE-M shapes
    (``raise_grid_block_minimums``) — a search-efficiency knob, NOT a correctness
    limit. We seed the M-block AT that floor (see ``_m_block_size``) so small-N /
    large-M shapes still get a seed.
    """
    spec = env.config_spec
    return len(spec.reduction_facts) == 1 and not spec.matmul_facts


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
    # Band-B (T2 with a [M_BLOCK, R_BLOCK] 2D accumulator carried across the inner
    # loop: kl_div, jsd) R_BLOCK cap in BYTES (per accumulator). A full-N
    # persistent R_BLOCK over-allocates the live state for these kernels and
    # spills — at the widest in-sample rows the persistent seed is 1.2–9.7×
    # SLOWER than a small looped chunk (matched A/B, M_BLOCK at floor: e.g. jsd
    # (8192,65536) persistent=19.8ms vs loop4096=2.4ms). Capping R_BLOCK at 16 KiB
    # (= 4096 fp32 elems) is best-or-tied at EVERY in-sample Band-B row (narrow
    # rows: persist/cap=0.996–1.017, i.e. no regression; wide rows: recovers the
    # spill). Expressed in BYTES (via itemsize) so it generalizes across dtypes.
    # Scalar-accumulator reductions (num_tiled_accumulators==0: every T1 kernel +
    # softmax_two_pass, which carries only a [M_BLOCK] row state) are UNAFFECTED —
    # they stay persistent to the structural cap. EVIDENCE:
    # _lab/harness/t2_bandb_chunk_sweep.py + t2_bandb_narrow_check.py.
    BANDB_R_BLOCK_BYTES = 16384
    # Persistent byte ceiling for RE-READ reductions (gated by the faithful
    # ``fact.row_reread`` property — the reduction-input row is reused/live across
    # the reduction boundary; see the gate note in get_seed_config), in BYTES (per
    # row, via itemsize so it generalizes across dtypes). Above this a re-read
    # reduction loops over a fixed chunk instead of staying persistent.
    #
    # WHY this is gated on re-reading the row (the cross_entropy widening). The
    # champion's "persistent to the 2**20 structural cap" was validated ONLY on
    # num_load=1 (sum_kernel, v3_crossover_sweep). A direct persistent-vs-looped
    # A/B at MATCHED warps across num_load (_lab/harness/persist_crossover_by_numload.py
    # + ce_crossover_tight.py, H100/fp32) shows the crossover is num_load-DEPENDENT:
    #
    #   rnumel(KiB) | sum nl=1 P/L | rms_norm nl=2 P/L | cross_entropy nl=3 P/L
    #     64 (16K)  |    1.01      |      0.99         |       0.98
    #    128 (32K)  |    1.00      |      0.96         |       0.98
    #    256 (64K)  |    1.00      |      0.99         |       1.04 (~tie)
    #    288 (72K)  |     -        |      2.75         |       1.09
    #    384 (96K)  |    1.01      |      3.00         |       1.61
    #    512(128K)  |    1.00      |      2.91         |       3.97
    #
    # num_load=1 (sum) ties persistent==looped at EVERY rnumel up to 1 MiB (P/L~1.0
    # at M=8 AND M=1024) — so the structural-cap policy stays correct for it (no
    # regression). num_load>=2 (rms_norm, layer_norm, cross_entropy) persistent
    # wins/ties only up to ~256 KiB then LOSES decisively (looped 1.6-4x faster) —
    # a multi-load reduction re-streams the wide row on each load pass, and a
    # persistent kernel that holds the whole row resident spills, while a looped
    # chunk streams. The crossover is SHARP at 256->288 KiB and holds at both
    # grid-occupied (M=1024) and grid-starved (M=8). Cap at 256 KiB: persistent at
    # the boundary (256 KiB) is still a tie/win, looped only from 288 KiB up.
    #
    # CAP = 240 KiB (RUN-3 RE-DERIVATION; was 128 KiB in run-2). The run-2 cap of
    # 128 KiB was ~1.75x too LOW: it looped EVERY cross_entropy row with actual
    # bytes > 128 KiB (V > 32768), but a fresh per-shape oracle + a matched-lever
    # persistent-vs-looped A/B on the REAL cross_entropy kernel at the REAL train
    # vocabs (NOT the run-2 synthetic ce_crossover proxy) show persistent WINS and
    # BEATS torch.compile-default well past 128 KiB. The run-2 cap sent the
    # ~50K-vocab boundary to a looped seed that is ~1.5-1.7x SLOWER than persistent
    # AND ~2x slower than tc-default (Phase-1 floor sweep: CE V=50257 G=0.65).
    #
    # The crossover (run-3 _lab/harness/run3_ce_persist_ab.py, H100/fp32,
    # median-of-7, persist[rl=None,w32] / seed_looped[rl=16384,w32], IDENTICAL at
    # M in {2048,4096,8192} -> a per-PROGRAM working-set property, not M-dependent):
    #
    #   V       | actual row KiB | persist/looped | verdict
    #   50304   |  196           | 1.56           | PERSIST (beats tc 1.07)
    #   57344   |  224           | 1.08-1.11      | PERSIST (beats tc, all M)
    #   65536   |  256           | 0.88-0.92      | LOOPED  (persist spills ~10%)
    #   73728   |  288           | 0.98           | LOOPED  (~tie)
    #   81920   |  320           | 0.70           | LOOPED  (persist spills 30%)
    #   98304+  |  384+          | 0.24-0.47      | LOOPED  (persist spills 2-4x)
    #
    # The crossover is between 224 KiB (persist wins ~10%) and 256 KiB (looped wins
    # ~10%): persistent holds the whole valid row resident across the 2-pass
    # (amax-pass then exp-sum-pass) RE-READ and wins until that resident row + its
    # working set SPILLS (~256 KiB on sm_90 for this multi-load pattern), at which
    # point a looped chunk that streams the row wins. So CAP = 240 KiB (245760 B)
    # sits in the measured dead-zone between the last confirmed persist-win (224
    # KiB) and the first looped-win (256 KiB): <=240 KiB persistent, >240 KiB
    # looped for a multi-load reduction. NOTE this caps on the ACTUAL row bytes
    # (fact.size_hint*itemsize), not next_pow2 — the crossover tracks the real
    # resident working set, and 50304/57344/65536 all share next_pow2=65536 yet
    # split across the crossover, so a next_pow2 cap could not separate them.
    #
    # In-sample EFFECT: rms_norm/layer_norm/softmax_two_pass (num_load>=2) all have
    # rnumel <= 16384 (<=64 KiB) << this cap -> stay persistent BYTE-IDENTICALLY
    # (no-regression). Now fires only for cross_entropy V >= 65536 (256 KiB+), which
    # genuinely want looped (persist spills there). cross_entropy V in
    # {49152,50257,50304,57344} (196-229 KiB) FLIP from looped(slow) -> persistent
    # (recovers 1.08-1.68x). num_load=1 (sum/long_sum) and Band-B (kl_div/jsd, whose
    # tighter 16 KiB R_BLOCK cap dominates) are unaffected.
    #
    # SEPARATE OPEN ISSUE (NOT addressed by this cap — see notebook "P2"): at
    # V >= 98304 the looped seed (chunk 16384, w32) is itself ~2x slower than
    # tc-default; the looped PARAMS (chunk/warps) and/or a 2-pass source ceiling are
    # a distinct Phase-2 lever, independent of WHERE the persist->loop boundary sits.
    MULTILOAD_PERSIST_MAX_BYTES = 245760
    # Band-C STRUCTURED-COMBINE (welford-like reduce-then-apply) tile caps, in BYTES
    # (per element via itemsize so they generalize across dtypes). RUN-2 RE-DERIVATION:
    # the welford source bug (`Tn=chunk.size(-1)` counted the padded tile width, not
    # the valid columns) is FIXED (`Tn=(tile_n.index<n).sum()`), so the combine tile
    # NO LONGER has to be a power-of-2 DIVISOR of N — a full byte-capped tile is correct
    # at ANY N (incl. odd/prime). This deletes the v8 `largest_pow2_div` constraint and
    # the apply<->combine coupling; the two tiles are now sized by INDEPENDENT byte
    # caps keyed on their own working-set / per-row-work physics.
    #
    # COMBINE tile cap. The combine pass loads a [M_BLOCK, R_BLOCK] chunk, materializes
    # chunk*chunk, and runs the multi-statistic (count/mean/M2) combine over it as a
    # SERIAL scalar recurrence across inner iterations. The combine therefore strongly
    # prefers to stay PERSISTENT (a single inner pass, next_pow2(N) tile) — LOOPING it
    # (capping below np2(N)) pays the serial-recurrence overhead and REGRESSES:
    # (262144,5120) combine 8192(persist)=G0.721 vs 4096(loop,2 iters)=0.636. So the
    # cap must be large enough to keep the combine persistent across the welford
    # curriculum (in-sample-v2 + TEST are all N<=8192): 32 KiB (= 8192 fp32) does that.
    # EVIDENCE (run2 _lab/harness/run2_wf_sweep.py, H100/fp32, M=262144): removing the
    # v8 pow2-divisor constraint recovers the non-pow2 cliff — (262144,2560) combine
    # 512->4096 G 0.857->0.973; (262144,1536) 512->2048 0.924->0.986; (262144,5120)
    # 1024->8192 0.693->0.721; well-factored 1024/2048/4096 stay BYTE-IDENTICAL to v8
    # (largest_pow2_div==N there). The combine is perf-flat at large N (N=8192:
    # c256..c8192 all 0.673-0.685). At N>=32768 (OUT of the welford curriculum/TEST,
    # all <=7168) a persistent 32 KiB combine spills (run-1 wf_combine_largeN_safety:
    # >16 KiB regresses at N>=32768); a looped combine would be needed there — a
    # documented future-scope limit, not a current-curriculum regression.
    STRUCTURED_COMBINE_CAP_BYTES = 32768
    # APPLY (normalize) tile. The apply pass (`y=(x-mean)*rstd*w+b; out[...]=y`) carries
    # NO accumulator and a MASKED output write (correct at ANY tile width), so this is a
    # PURE perf lever. Single-pass (persistent next_pow2(N)) when the per-row apply work
    # is small; LOOPED (capped chunk) when it is large (better occupancy / less register
    # pressure). The crossover keys on the PER-ROW VALID BYTES (n_valid*itemsize), NOT
    # next_pow2(N): (262144,2560) (10 KiB/row) wants a PERSISTENT 4096 apply (G
    # 0.916->0.973) while (262144,4096) (16 KiB/row, SAME np2=4096) wants a LOOPED 2048
    # apply (0.730->0.760). A pure np2-byte cap cannot separate two shapes with the same
    # np2; the valid-row-bytes threshold can. Threshold 12 KiB sits between 2560's 10 KiB
    # (persistent) and 4096's 16 KiB (looped). EVIDENCE: run2_wf_sweep.py apply sweeps.
    STRUCTURED_APPLY_PERSIST_MAX_BYTES = 12288
    # Looped-apply chunk (bytes) used when the per-row apply work exceeds the persist
    # threshold. 8 KiB (= 2048 fp32) is best/near-best at the wide in-sample-v2 rows
    # (N=4096/5120 apply 2048 best; N=8192 nearly flat 512-2048). Same byte discipline.
    # DO NOT raise this naively: EDIT#4 (run-3) tried 16384 (apply 2048->4096) for a
    # +1.05-1.07x at welford(4096,16384)/(32768,8192), but the results-referee REJECTED
    # it -- apply=4096 is a PATHOLOGICAL VALLEY at large-M / N~5120: welford(262144,5120)
    # regressed 4-7x (33.5ms vs 4.6ms, reproduced 5x/3 codepaths/2 seeds), and (8192,4096)
    # /(16384,4096) -2.3%. The cap is NOT a clean Band-C-only single lever -- a wider apply
    # only pays WITH a bigger combine + M_block (the coupled welford full-close = EDIT#4b);
    # any future raise must be GATED so apply does not exceed 2048 at the large-M/N=5120
    # class. Reverted to 8192.
    STRUCTURED_APPLY_LOOP_CHUNK_BYTES = 8192
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
    def _block_floor(cls, bs_spec: BlockSizeSpec) -> int:
        """The autotuner floor for a single block_sizes entry."""
        return max(1, bs_spec.min_size, bs_spec.autotuner_min)

    @classmethod
    def _eviction_policies(
        cls,
        env: CompileEnvironment,
        kind: str,
        reread_slots: tuple[int, ...] = (),
    ) -> list[str] | None:
        """``load_eviction_policies`` list (length == the live spec's, built EXACTLY
        — the field is NOT length-validated by ``normalize``), keyed on per-load
        cache RESIDENCY. Returns None to leave the autotuner default ('').

        - ``"stream"`` — a single streamed reduction input (``num_load == 1``:
          sum, long_sum) is read ONCE and never reused, so EVERY load is
          ``'first'`` (evict_first frees L2). e.g. sum (512,8192) G 0.925->1.451,
          (2048,16384) 0.931->1.087.
        - ``"reread"`` — the reduction-input row is RE-READ across >= 2 passes
          (``fact.row_reread``: welford combine+apply re-read x; cross_entropy
          amax-pass + exp-sum-pass re-read logits). The FIRST load of the re-read
          ROW wants ``'last'`` (kept L2-resident for the later re-read); EVERY OTHER
          slot -> ``'first'`` (stream/evict-first frees L2 — both the row's own
          re-reads, which are final uses, and all streamed-once operands). RUN-3
          DE-HACK: the slot getting ``'last'`` is derived from WHICH host buffer is
          the re-read ROW (``device_ir``'s ``_compute_reread_buffer_slots``: HBM-
          re-read AND reduction-input), NOT the run-2 POSITIONAL ``slot 0`` — which
          only happened to be right for welford (its row's first load IS slot 0) and
          mis-places ``'last'`` for cross_entropy (whose row ``logits`` first loads at
          slot 2, behind a labels + a target-gather load). A/B
          (run3_reread_rulefix_ab.py): CE wide-V 1.31x/1.19x; welford 1.28x/1.27x ==
          the run-2 positional rule (its row IS slot 0, so streaming the rest matches).
          The earlier "row re-reads -> 'first', OTHERS default ''" variant regressed
          welford 3-6% (under-streamed weight/bias); positional ``'last'`` on slot 0
          FAILS CE (0.99/1.04, marks labels resident) — so the win is the buffer-
          identity ``'last'`` placement, not "some eviction."

        Multi-load reductions with REUSED broadcast operands (rms_norm/layer_norm:
        a streamed x PLUS a weight/bias reused across the grid) historically had
        no clean per-slot rule under the positional rule (run-1); whether the
        provenance-faithful policy gives them a clean win is a SEPARATE measured
        question — those callers still pass no kind until A/B-confirmed.
        """
        n = env.config_spec.load_eviction_policies.length
        if n <= 0:
            return None
        if kind == "stream":
            return ["first"] * n
        if kind == "reread":
            # Faithful re-read policy: the re-read reduction ROW's FIRST load (from the
            # fact's reread_buffer_slots, which identify WHICH buffer is the re-read
            # row via provenance) -> 'last' (keep it L2-resident for its later re-read);
            # EVERY OTHER slot -> 'first' (stream/evict-first frees L2). This keeps the
            # buffer-IDENTITY de-hack for the single 'last' slot (the fact-integrity
            # caveat: 'last' goes on the verified reduction row, NOT the run-2 positional
            # slot 0 -- for cross_entropy that's logits@slot2, not labels@slot0) while
            # streaming all other loads, which is the run-2 positional behavior that was
            # actually RIGHT for the non-row slots. A/B (run3_reread_rulefix_ab.py):
            #   CE wide-V: ['first','first','last','first','first'] -> 1.31x/1.19x (ties
            #     my earlier ['','','last',...] -- streaming the labels/logits_flat
            #     gathers is marginally better, never worse).
            #   welford: row x@slot0 -> ['last','first','first','first'] == the run-2
            #     positional rule -> reproduces its win (1.28x/1.27x); BYTE-IDENTICAL to
            #     the shipping champion for welford (no regression).
            # The earlier "row slots -> first, OTHERS default ''" variant REGRESSED
            # welford 3-6% (it under-streamed weight/bias); this streams them.
            # Positional ['last',...] FAILS on CE (0.99/1.04) because it marks labels
            # 'last' -- so the win is the buffer-identity de-hack, not "some eviction."
            # Guard: slots must be in range of the live spec length.
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

        # The persistent-vs-looped lever is identical for BOTH tracks: it keys on
        # the reduction extent (fact.size_hint = rnumel) and the backend's
        # per-tile element cap, NOT on the knob mechanism. Persistent is the
        # workhorse for EVERY row the backend can compile as a single pass; the
        # structural limit is Triton's max_tensor_numel (2**20 elems), above which
        # the whole-row `tl.arange` is rejected at codegen.
        #
        # For SINGLE-LOAD reductions (num_load==1: sum, long_sum) a
        # warps-held-equal sweep (_lab/harness/v3_crossover_sweep.py) shows
        # persistent wins or ties at every feasible byte size up to that cap, so
        # there is no perf byte fence below it.
        #
        # For RE-READ reductions the reduction-input row is REUSED ACROSS THE
        # REDUCTION BOUNDARY — it feeds the reduction AND a downstream consumer
        # (cross_entropy amax-pass + exp-sum-pass; softmax max-pass + exp-sum-pass;
        # rms_norm/layer_norm reduction + the APPLY/normalize that reuses the row).
        # That reuse pins the WHOLE ROW LIVE/RESIDENT across the boundary whether the
        # consumer re-reads it from HBM (CE) or holds it in registers (rms_norm) — the
        # liveness, not the load mechanism, is what matters. A persistent seed holds
        # the whole row resident and WINS while it fits, but SPILLS once the resident
        # row exceeds the register/SMEM budget (~256 KiB on sm_90), where a looped
        # chunk that streams the row per pass wins. So a re-read reduction gets an
        # additional PERF byte ceiling (MULTILOAD_PERSIST_MAX_BYTES, run-3-re-derived
        # to 240 KiB; see the constant) below the structural one.
        #
        # GATE = ``fact.row_reread`` (EDIT-GATE-v2). The FAITHFUL "the reduction-input
        # row is REUSED / LIVE across the reduction boundary" property, computed in
        # device_ir (``_compute_row_reread``) by a CONSUMER-DATAFLOW trace cutting AT
        # the reduction: a loaded reduction-input tile's value feeds >= 2 distinct
        # reduction lowerings (softmax/cross_entropy: max + sum) OR feeds a reduction
        # AND reaches a store BYPASSING the reduction (rms_norm/layer_norm: x feeds the
        # sum-of-squares AND the x*rstd*w apply). Liveness, NOT a load-op count and NOT
        # a read-region count — right for in-register reuse (rms_norm: one hl.load of
        # x, value used twice) and immune to the single-pass-double-load false positive.
        # softmax's max + sum passes both load x. A single-pass stream (sum/long_sum)
        # touches its row in only ONE loop graph -> 1. Two DISTINCT inputs each read
        # once (kl_div yp/yt, jsd) -> each 1 -> False. Dataflow-, not load-op-count- or
        # coding-style-dependent (softmax_decomposed=True too). It REPLACES two rejected
        # proxies: ``num_load>=2`` (OVER-counts —
        # cross_entropy's scalar label gather inflates num_load to 3) and
        # ``num_reduction_ops>=2`` (UNDER-counts — rms_norm/layer_norm's apply-pass
        # re-read is NOT a ReductionLowering, so num_reduction_ops==1, which would
        # EXEMPT their wide 512 KiB rows -> persistent -> ~2.9x spill). row_reread
        # gives the exact right set: sum/long_sum=False (single stream -> persist to
        # the structural cap even at >240 KiB); rms_norm/layer_norm/softmax/
        # cross_entropy=True (re-read -> governed by this byte cap); kl_div/jsd=False
        # (two DISTINCT inputs each read once — and Band-B's R_BLOCK cap dominates
        # them anyway); welford=True (re-reads x; Band-C caps dominate). Verified on
        # all 9 kernels + flip-set across train+val+test+ROBUSTNESS (the widest rows
        # live in robustness; the cap fires on byte width). See
        # ReductionFact.row_reread + _lab/harness/run3_row_reread_probe.py.
        persist_cap = env.backend.max_tensor_numel  # None ⇒ no element cap
        can_persist = persist_cap is None or fact.size_hint <= persist_cap
        if fact.row_reread:
            row_bytes = fact.size_hint * max(1, fact.itemsize)
            if row_bytes > cls.MULTILOAD_PERSIST_MAX_BYTES:
                can_persist = False
        from ..._utils import next_power_of_2 as _np2

        if can_persist:
            # Persistent (single-pass) reduction. num_warps scales with rnumel
            # ALONE (see _num_warps). For T1 the persistent extent is encoded as
            # reduction_loops=None; for T2 it is the full power-of-2 R_BLOCK so
            # the inner `for tile_n` loop runs exactly once.
            extent = _np2(fact.size_hint)
            num_warps = cls._num_warps(fact)
            persistent = True
        else:
            # Looped (chunked) reduction. Reached for EITHER reason:
            #  (a) the row exceeds the backend's persistent element cap (2**20),
            #      so a single whole-row pass cannot compile — a structural tail
            #      with no in-sample coverage (the notebook "looped tail"
            #      disclosure); OR
            #  (b) num_load>=2 AND the row exceeds MULTILOAD_PERSIST_MAX_BYTES —
            #      a PERF crossover (the cross_entropy widening: wide multi-load
            #      rows are 1.6-4x faster looped). This DOES fire in-sample, for
            #      cross_entropy V=131072 (512 KiB).
            # Either way: loop over a fixed R_BLOCK chunk with the high streaming
            # warp count (looped wide rows want chunk 16384 / w32 — A/B in
            # _lab/harness/ce_persist_vs_loop.py).
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
                "block_sizes": [cls._m_block_size(env)],
                "reduction_loops": reduction_loops,
                "num_warps": num_warps,
                "num_stages": 1,
                # pid_type: 'flat' is the PRINCIPLED DEFAULT for the COMMON case
                # (persistent / narrow-looped reductions are grid-saturated at the
                # M-grid; run-1's matched-lever A/B showed flat dominates 1.5-4x and
                # the oracle's persistent pid was a confounded passenger). EDIT-PID
                # (below) is the ONE measured EXCEPTION: a wide LOOPED RE-READ row
                # (row_reread AND not persistent) is NOT grid-saturated — it's a few
                # long rows (CE M=2048-8192 << machine) each grinding a heavy looped
                # multi-pass re-read, where a persistent_interleaved grid + register
                # cap measurably beats flat (1.05-1.25x). Set 'flat' here; EDIT-PID
                # overrides it in that regime.
                "pid_type": "flat",
            }
            # Eviction (keyed on the FAITHFUL re-read property, RUN-3):
            #  - num_load==1 (sum/long_sum): a single streamed input read ONCE ->
            #    'first' everywhere (evict_first frees L2).
            #  - else row_reread AND LOOPED (cross_entropy wide-V: logits re-read
            #    across amax + exp-sum passes, the row STREAMED from HBM each pass):
            #    the re-read buffer's first load -> 'last' (keep L2-resident for the
            #    re-read), its re-reads -> 'first'; from provenance (WHICH buffer is
            #    re-read), not a positional slot. CE wide-V 1.31x/1.19x/1.09x.
            #    GATED on `not persistent` because eviction policy only affects
            #    HBM-STREAMED loads: a PERSISTENT row is held in registers/SMEM across
            #    the passes (no HBM re-stream), so its load eviction is moot — A/B
            #    confirms eviction is NEUTRAL on the persistent CE boundary (50304:
            #    1.001). This scopes the change to the looped re-read regime (the win)
            #    and leaves persistent re-read kernels (rms/ln/softmax at floor, CE
            #    boundary) BYTE-IDENTICAL — avoiding an unmeasured blanket change to
            #    at-floor kernels. (rms/ln only loop at >240KiB robustness rows, where
            #    the same wide-row physics applies.)
            #  - else: default.
            if fact.num_load == 1:
                evict = cls._eviction_policies(env, "stream")
            elif fact.row_reread and not persistent:
                evict = cls._eviction_policies(env, "reread", fact.reread_buffer_slots)
            else:
                evict = None
            if evict is not None:
                seed["load_eviction_policies"] = evict
            # EDIT-PID: a wide LOOPED RE-READ T1 reduction (gate identical to the
            # reread-eviction above: ``fact.row_reread and not persistent``) is a
            # FEW-LONG-ROWS workload — M=2048-8192 rows (<< the machine's program
            # capacity) each grinding a heavy looped multi-pass re-read. A
            # ``persistent_interleaved`` grid of ``get_num_sm * num_sm_multiplier``
            # resident programs (striding the M-grid) + a register cap measurably
            # beats the default ``flat`` here (matched-lever A/B vs flat on the
            # EDIT#3-evicted seed: CE (4096,98304) 1.23x / (8192,128256) 1.25x /
            # (2048,256000) 1.05x; run3_pid_derived_ab.py). The eviction-gate scoping
            # holds: PERSISTENT rows stay 'flat' (grid-saturated); welford (Band-C)
            # and softmax (T2) take OTHER branches so are untouched; only T1 fires.
            #   - num_sm_multiplier: PHYSICS-derived from the WORKLOAD (M rows) and
            #     the HARDWARE (SM count), NOT the oracle's value: size the persistent
            #     grid to ~= the row count so each row maps to ~one resident program
            #     (fills the SMs the under-filling M-grid leaves idle, without
            #     excessive per-program row-looping). = clamp(np2(ceil(M / num_sm)),
            #     1, 32). Degenerate M=1 (rms/ln >240KiB robustness) -> 1 (no
            #     over-subscription), where pid is a measured tie (1.008/1.010). The
            #     oracle's own pick VARIES (32/32/4 across the 3 CE), so a derived
            #     M-scaled value is MORE principled than any constant.
            #   - maxnreg=64: a high-occupancy register cap — capping regs/thread
            #     raises resident-warp count to hide the looped-reread memory latency.
            #     LOAD-BEARING (not a passenger): isolating it, the gain ~halves
            #     without it (1.05/1.15/1.00 vs 1.23/1.25/1.05) and 256000 ties flat
            #     -> maxnreg=64 is what makes the widest net-positive. 64 = a standard
            #     ~2x-occupancy cap for a heavy-accumulator persistent kernel.
            if fact.row_reread and not persistent:
                # local import: ``helion.runtime`` imports the heuristics, so a
                # module-level import here is circular.
                from ...runtime import get_num_sm

                # grid_rows = product of the M-axis extents (the rows the flat grid
                # launches). Use the passed ``env``'s ``size_hint`` on each block's
                # static size (``BlockSizeInfo.size_hint()`` would need the env to be
                # the CURRENT context, not guaranteed at seed-emit).
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
            # The SAME inner extent is tiled by a combine pass (the reduction axis,
            # fact.block_id, carrying per-row count/mean/M2 as a scalar recurrence)
            # AND one-or-more apply/normalize passes (fact.apply_block_ids, no
            # reduction). A single-axis seed would floor the apply tile(s) to width 1
            # (~10-20x slower), so we widen them. The two tiles are sized by
            # INDEPENDENT byte caps (STRUCTURED_* constants) keyed on their own
            # physics — NOT on N's factorization.
            #
            # RUN-2 SIMPLIFICATION: the welford source bug is fixed (the per-chunk
            # count is the masked valid count `(tile_n.index < n).sum()`, not the
            # constexpr tile width), so a full byte-capped combine tile is correct at
            # ANY N (incl. odd/prime). That DELETES the v8 `largest_pow2_div`
            # correctness constraint (which collapsed the combine tile at non-pow2 N —
            # to 1 at prime N) AND the apply<->combine coupling. Both tiles are now
            # plain byte-capped next_pow2(N) reductions like any other.
            np2_n = _np2(fact.size_hint)
            n_valid = (
                fact.static_rnumel if fact.static_rnumel is not None else fact.size_hint
            )
            itemsize = max(1, fact.itemsize)
            # Combine tile: full next_pow2(N) up to the spill-safe byte cap; looped
            # above. (262144,2560) 512->4096 G 0.857->0.973; (262144,1536) 512->2048.
            combine_cap_elems = max(1, cls.STRUCTURED_COMBINE_CAP_BYTES // itemsize)
            combine_block = min(np2_n, _np2(combine_cap_elems))
            # Apply (normalize) tile: PERSISTENT (single masked pass) when the per-row
            # valid work is small; LOOPED (capped chunk) when large. Keys on per-row
            # VALID bytes, not np2(N) (so (262144,2560) gets a persistent apply while
            # (262144,4096), same np2, gets a looped one).
            if n_valid * itemsize <= cls.STRUCTURED_APPLY_PERSIST_MAX_BYTES:
                apply_block = np2_n
            else:
                apply_cap_elems = max(
                    1, cls.STRUCTURED_APPLY_LOOP_CHUNK_BYTES // itemsize
                )
                apply_block = min(np2_n, _np2(apply_cap_elems))
            apply_ids = set(fact.apply_block_ids)
            red_idx_sc = spec.block_sizes.block_id_to_index(fact.block_id)
            sc_block_sizes: list[int] = []
            for i in range(len(spec.block_sizes)):
                bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
                bid = bs_spec.block_id
                if i == red_idx_sc:
                    sc_block_sizes.append(combine_block)
                elif bid in apply_ids:
                    sc_block_sizes.append(apply_block)
                else:
                    sc_block_sizes.append(cls._block_floor(bs_spec))
            sc_seed: dict[str, Any] = {
                "block_sizes": sc_block_sizes,
                "num_warps": num_warps,
                "num_stages": 1,
                "pid_type": "flat",  # principled constant — see the T1 branch.
            }
            # Eviction: the structured combine RE-READS the reduction input (x in
            # the combine pass, re-read in the apply pass) -> 'last' on x's FIRST
            # load (keep L2-resident), 'first' on its re-reads. RUN-3 DE-HACK: the
            # 'last' slot is now derived from WHICH host buffer is re-read (device_ir
            # provenance), not the POSITIONAL slot 0 — for welford the re-read buffer
            # IS slot 0 so this is byte-identical to the run-2 win (welford
            # (262144,4096) G 0.759->0.950, (5120) 0.696->0.807), but faithful.
            sc_evict = cls._eviction_policies(env, "reread", fact.reread_buffer_slots)
            if sc_evict is not None:
                sc_seed["load_eviction_policies"] = sc_evict
            return Config(**sc_seed)

        # T2 (user-tiled / manually-looped): the reduction axis IS a block_sizes
        # entry (the inner `hl.tile(n, block_size=R_BLOCK)`); there is no
        # `reduction_loops` knob. Persistent == R_BLOCK >= next_pow2(N) so the
        # inner loop runs once. Every OTHER block_size (the grid/row axes) stays
        # at its floor — for the Band-B loss kernels (kl_div/jsd) this keeps
        # M_BLOCK at 1, which is required by the u0*u1 <= 2**20 numel constraint
        # (the inner loop carries [M_BLOCK, R_BLOCK] live accumulators; a full-N
        # R_BLOCK only survives at M_BLOCK=1).
        r_block = extent
        if fact.num_tiled_accumulators >= 1:
            # BAND B: this T2 reduction carries one-or-more [M_BLOCK, R_BLOCK] 2D
            # accumulators across the inner loop (kl_div: loss_sum; jsd:
            # intermediate_loss + intermediate_dX). A full-N persistent R_BLOCK
            # over-allocates that live state and SPILLS — at the widest in-sample
            # rows the persistent seed is 1.2–9.7× SLOWER than a small looped
            # chunk (matched A/B vs persistent, M_BLOCK at floor). Cap R_BLOCK so
            # the LIVE carried-accumulator footprint stays SM-resident: the inner
            # loop holds num_carried_accumulators independent [M_BLOCK, R_BLOCK]
            # tiles resident SIMULTANEOUSLY (the loop carry set), all at the SAME
            # R_BLOCK, so the resident footprint is R_BLOCK * itemsize *
            # num_carried_accumulators. Hold that to a single BANDB_R_BLOCK_BYTES
            # budget => R_BLOCK <= budget / (itemsize * n_carried).
            # EDIT#5 (v2 — re-keyed onto num_carried_accumulators per fact-integrity):
            # jsd carries 2 (intermediate_loss + intermediate_dX) -> 16384/(4*2)=2048;
            # kl_div carries 1 (loss_sum; its [M,R] kl_loss is in-loop SCRATCH, NOT
            # carried) -> 16384/(4*1)=4096, byte-IDENTICAL to before. Closes jsd to
            # oracle parity (full oracle 1.21@narrow + 1.03@wide both want R_BLOCK
            # 2048; V-INDEPENDENT, no-regression wide; A/B +19%/+10%/+2.4%,
            # run3_bandb_nro_ab). The divisor is num_carried_accumulators — the
            # FAITHFUL count of [M,R] tiles in the reduction loop's carry set — NOT
            # num_reduction_ops (a ReductionLowering count that equals the carried
            # count only under a 1:1 reduction<->accumulator structure; it mis-sizes
            # N reductions on ONE carried tile [under-sizes] or M tiles reduced fewer
            # times [over-sizes -> spill]) and NOT num_tiled_accumulators (over-counts
            # in-loop scratch like kl_loss -> would over-divide). Gated on the WORKLOAD
            # properties num_tiled_accumulators>=1 (Band-B routing) + carried-count
            # divisor, NOT kernel identity — any Band-B reduction's chunk scales as
            # 1/n_carried (the TRANSFER tv_distance probe carries 1 -> stays 4096 like
            # kl_div). jsd is the sole firer by curriculum incidence, not a fence.
            # Scalar-/row-accumulator T2 (softmax_two_pass, num_tiled_accumulators==0)
            # stays persistent to the structural cap.
            bandb_cap = max(
                1,
                cls.BANDB_R_BLOCK_BYTES
                // (max(1, fact.itemsize) * max(1, fact.num_carried_accumulators)),
            )
            r_block = min(r_block, _np2(bandb_cap))

        red_idx = spec.block_sizes.block_id_to_index(fact.block_id)
        block_sizes_list: list[int] = []
        for i in range(len(spec.block_sizes)):
            bs_spec = cast("BlockSizeSpec", spec.block_sizes[i])
            if i == red_idx:
                block_sizes_list.append(r_block)
            else:
                block_sizes_list.append(cls._block_floor(bs_spec))
        seed = {
            "block_sizes": block_sizes_list,
            "num_warps": num_warps,
            "num_stages": 1,
            "pid_type": "flat",  # principled constant — see the T1 branch.
        }
        # EDIT#6: the SAME faithful reread-eviction as T1/Band-C, now on the T2 plain
        # path. A T2 user-tiled reduction that RE-READS its row across passes and runs
        # LOOPED (softmax_two_pass: x re-read in the max pass + the exp-sum pass; wide
        # rows > the persist cap) wants the re-read row L2-resident: 'last' on the row's
        # first load (reread_buffer_slots[0]), 'first' elsewhere. Gate is IDENTICAL to
        # the T1 eviction (``fact.row_reread and not persistent``). Full-oracle: softmax
        # (1024,65536) 1.36x / (512,131072) 1.10x, BEATS tc; lever-decomp isolates the
        # eviction as the sole carrier (chunk/warps passengers; run3_softmax_decomp_ab).
        # The fact already computes softmax's slots=(0,1) — this is purely a new
        # CONSUMER of the existing reread_buffer_slots, no fact change. Other T2 kernels
        # (kl_div/jsd) are row_reread=False -> unaffected; narrow/persistent softmax
        # stays default (byte-identical). The reread rule now spans all 3 tracks.
        if fact.row_reread and not persistent:
            ev = cls._eviction_policies(env, "reread", fact.reread_buffer_slots)
            if ev is not None:
                seed["load_eviction_policies"] = ev
        return Config(**seed)

    @classmethod
    def get_seed_configs(
        cls, env: CompileEnvironment, device_ir: DeviceIR
    ) -> list[Config] | None:
        """Run-2 Goal 3b — opt-in MULTI-seed PORTFOLIO for BEAT-MAX-EFFORT autotune.

        Returns None (single-seed: Product-A + Goal-3a unchanged) UNLESS the env
        flag ``HELION_REDUCTION_SEED_PORTFOLIO`` is set. When set, returns the best
        deterministic seed PLUS a few structurally-distinct variants, each a
        PRE-REGISTERED falsifiable hypothesis about a coupling the bounded
        ``LFBOTreeSearch`` under-samples run-to-run (see ``_lab/run2_notebook.md``
        "Goal 3b portfolio"). The autotuner injects ALL into gen-0 and explores
        around them, so the seeded best-of-N reaches a hard coupling the unseeded
        search reaches only by luck. Variants are derived from PRINCIPLE (the
        levers + Goal-2 findings), NOT fit to observed unseeded winners (p-hacking,
        banned). ``dedupe_configs`` removes exact duplicates.
        """
        import os

        if not os.environ.get("HELION_REDUCTION_SEED_PORTFOLIO"):
            return None
        base = cls.get_seed_config(env, device_ir)
        if base is None:
            return None
        from .common import dedupe_configs

        bd: dict[str, Any] = dict(base)
        variants: list[dict[str, Any]] = []
        # H1/H2 — WARP portfolio. The rnumel ramp picks one num_warps; the optimum
        # can sit one step away (register-heavy combines want fewer warps; streamed
        # wide rows want more) and the bounded search may not reliably land it.
        # Inject the full ramp {4,8,16,32}.
        for w in (4, 8, 16, 32):
            if w != bd.get("num_warps"):
                variants.append({**bd, "num_warps": w})
        # H3 — EVICTION coupling (Goal 2). The per-load eviction space is large and
        # the search under-samples it at bounded budget; seed BOTH the default
        # (no eviction) and all-'last' variants alongside the rule already in base.
        nev = env.config_spec.load_eviction_policies.length
        if nev > 0:
            variants.extend(
                [
                    {**bd, "load_eviction_policies": [""] * nev},
                    {**bd, "load_eviction_policies": ["last"] * nev},
                ]
            )
        # H4 — num_stages pipelining (usually inert for memory-bound reductions; a
        # cheap extra probe in case an inner op overlaps).
        if bd.get("num_stages") != 2:
            variants.append({**bd, "num_stages": 2})
        configs: list[Config] = [base, *(Config(**v) for v in variants)]
        return dedupe_configs(configs)
