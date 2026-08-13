from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

import helion
from helion._compiler.autotuner_heuristics.triton import _B200_MATMUL_HEURISTICS_PATH
from helion._compiler.autotuner_heuristics.triton import (
    TritonB200FormulaMatmulHeuristic,
)
from helion._compiler.autotuner_heuristics.triton import TritonB200MatmulHeuristic
from helion._compiler.autotuner_heuristics.triton import (
    TritonB200MultiMatmulHeuristic as _MULTI,
)
from helion._compiler.autotuner_heuristics.triton import TritonH100MatmulHeuristic
from helion._compiler.autotuner_heuristics.triton import _batched_static_matmul_fact
from helion._compiler.autotuner_heuristics.triton import _generalized_static_matmul_fact
from helion._compiler.autotuner_heuristics.triton import _seed_config_for_bucket
from helion._compiler.autotuner_heuristics.triton import _seed_config_for_config_spec
from helion.autotuner.config_fragment import EnumFragment
from helion.autotuner.config_fragment import IntegerFragment
from helion.autotuner.config_fragment import ListOf
from helion.autotuner.config_fragment import PowerOfTwoFragment
from helion.autotuner.config_spec import DotAxes
from helion.autotuner.config_spec import DotAxisKind
from helion.autotuner.config_spec import LiveTile
from helion.autotuner.config_spec import MatmulFact

_SHAPE_BUCKET_KEYS = {
    "dtype",
    "k_bucket",
    "m_bucket",
    "n_bucket",
    "k_value",
    "m_value",
    "n_value",
}


def _matmul_fact(
    *,
    static_m: int = 1024,
    static_n: int = 1024,
    static_k: int = 1024,
    lhs_ndim: int = 2,
    rhs_ndim: int = 2,
) -> MatmulFact:
    return MatmulFact(
        lhs_ndim=lhs_ndim,
        rhs_ndim=rhs_ndim,
        m_block_id=0,
        n_block_id=1,
        k_block_id=2,
        static_m=static_m,
        static_n=static_n,
        static_k=static_k,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )


def _matmul_config_spec(
    *,
    matmul_facts: list[MatmulFact] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        matmul_facts=[] if matmul_facts is None else matmul_facts,
        block_sizes=[object(), object(), object()],
        allowed_pid_types=("flat",),
        _base_default_config=lambda: helion.Config(
            block_sizes=[1, 1, 1],
            l2_groupings=[1],
            num_warps=4,
            num_stages=1,
            pid_type="flat",
        ),
        _flat_fields=lambda: {
            "block_sizes": ListOf(IntegerFragment(1, 4096, 1), length=3),
            "l2_groupings": ListOf(IntegerFragment(1, 64, 1), length=1),
            "num_warps": PowerOfTwoFragment(1, 32, 4),
            "num_stages": IntegerFragment(1, 8, 1),
            "pid_type": EnumFragment(("flat",)),
        },
        normalize=lambda raw, _fix_invalid=False: None,
        _shrink_for_numel_constraints=lambda config: None,
    )


def _bucket(m: int, n: int, k: int) -> dict[str, object]:
    return {
        "dtype": "fp16_bf16",
        "m_value": m,
        "n_value": n,
        "k_value": k,
    }


def test_matmul_heuristic_rules_have_unique_shape_buckets() -> None:
    data = json.loads(_B200_MATMUL_HEURISTICS_PATH.read_text())
    keys = [json.dumps(rule["shape_bucket"], sort_keys=True) for rule in data["rules"]]

    assert set(data) == {"rules"}
    assert len(keys) == len(set(keys))
    for rule in data["rules"]:
        assert set(rule) == {"shape_bucket", "templates"}
        assert set(rule["shape_bucket"]).issubset(_SHAPE_BUCKET_KEYS)
        for key in ("k_bucket", "m_bucket", "n_bucket"):
            value = rule["shape_bucket"].get(key)
            if value is not None:
                values = value if isinstance(value, list) else [value]
                assert all(isinstance(item, str) for item in values)
                assert all(item.startswith("(") for item in values)
                assert all(item.endswith(("]", ")")) for item in values)
        for key in ("k_value", "m_value", "n_value"):
            value = rule["shape_bucket"].get(key)
            if value is not None:
                values = value if isinstance(value, list) else [value]
                assert all(isinstance(item, int) for item in values)
        assert rule["templates"]
        assert all("template" not in template for template in rule["templates"])


def test_matmul_bucket_matching_generates_seed_config() -> None:
    seed = _seed_config_for_bucket(
        _bucket(1024, 1024, 1024),
        config_spec=_matmul_config_spec(),
    )

    assert seed is not None
    assert dict(seed)["block_sizes"] == [128, 64, 64]
    assert dict(seed)["l2_groupings"] == [2]

    assert (
        _seed_config_for_bucket(
            _bucket(128, 128, 128),
            config_spec=_matmul_config_spec(),
        )
        is None
    )


def test_matmul_fact_generates_compiler_seed_config() -> None:
    config_spec = _matmul_config_spec(matmul_facts=[_matmul_fact()])

    seed = _seed_config_for_config_spec(config_spec)

    assert seed is not None
    assert dict(seed)["block_sizes"] == [128, 64, 64]

    config_spec = _matmul_config_spec(
        matmul_facts=[_matmul_fact(), _matmul_fact()],
    )

    assert _seed_config_for_config_spec(config_spec) is None


def test_triton_b200_matmul_heuristic_gates_on_hardware() -> None:
    env = SimpleNamespace(device=None, config_spec=_matmul_config_spec())
    env.config_spec.matmul_facts.append(_matmul_fact())
    b200 = SimpleNamespace(
        device_kind="cuda",
        hardware_name="NVIDIA B200",
        compute_capability="sm100",
    )
    h100 = SimpleNamespace(
        device_kind="cuda",
        hardware_name="NVIDIA H100",
        compute_capability="sm90",
    )

    with patch(
        "helion._hardware.get_hardware_info",
        return_value=b200,
    ):
        assert TritonB200MatmulHeuristic.is_eligible(env, SimpleNamespace())
        seed = TritonB200MatmulHeuristic.get_seed_config(env, SimpleNamespace())

    assert seed is not None
    assert dict(seed)["block_sizes"] == [128, 64, 64]

    with patch(
        "helion._hardware.get_hardware_info",
        return_value=h100,
    ):
        assert not TritonB200MatmulHeuristic.is_eligible(env, SimpleNamespace())


def test_b200_formula_subsumes_table_promotion_wiring() -> None:
    # The sm100 FORMULA owns the compiler default; the TABLE is demoted to a search seed.
    assert TritonB200FormulaMatmulHeuristic.promote_seed_to_default is True
    assert TritonB200MatmulHeuristic.promote_seed_to_default is False
    assert TritonB200FormulaMatmulHeuristic.HARDWARE_TARGETS == (("cuda", "sm100"),)
    # The formula is a subclass of the H100 budget formula (inherits _matmul_tile).
    assert issubclass(TritonB200FormulaMatmulHeuristic, TritonH100MatmulHeuristic)
    # Registered AFTER the table so it wins the last-promote-wins default loop.
    from helion._compiler.autotuner_heuristics import get_heuristics

    order = [h.__name__ for h in get_heuristics("triton")]
    assert order.index("TritonB200FormulaMatmulHeuristic") > order.index(
        "TritonB200MatmulHeuristic"
    )


def test_h100_base_tile_is_unchanged_by_tmem_budget() -> None:
    # TMEM_BUDGET is None on the sm90 base (no tensor memory), so the TMEM growth step is skipped and
    # the H100 formula stays byte-identical: a big compute-bound cube keeps the register-budget wide-N
    # [128, 256, 64] w8 s4 tile (num_sm=132).
    assert TritonH100MatmulHeuristic.TMEM_BUDGET is None
    assert TritonH100MatmulHeuristic._matmul_tile(4096, 4096, 4096, 2, 132, 1) == (
        128,
        256,
        64,
        8,
        4,
        1,
    )


def test_b200_tile_grows_to_fill_tmem_budget() -> None:
    # On sm100 the tile is grown against the TENSOR-MEMORY budget (the accumulator lives there, not in
    # registers): double whichever axis still fits, N first since it is the coalesced store axis. The
    # reservation includes the A operand at the largest bk the formula can emit, so the [256,256] fp32
    # accumulator -- which alone exactly fills tensor memory -- can never be reached, and the wide
    # [128,256] tile is the largest that fits.
    cls = TritonB200FormulaMatmulHeuristic
    sm = 148
    for m, n, k in ((2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)):
        assert cls._matmul_tile(m, n, k, 2, sm, 1)[:2] == (128, 256), (m, n, k)
    # Non-saturated batched dots grow too -- tensor memory is theirs to use as well.
    assert cls._matmul_tile(4096, 4096, 4096, 2, sm, 4)[:2] == (128, 256)
    # ...but a SATURATED batched dot keeps the small occupancy tile step (2.5) chose for it; growing it
    # back would undo that (measured: it inflates mamba-shaped tiles from [32,64] to [32,1024]).
    bm, bn = cls._matmul_tile(32, 1024, 4096, 2, sm, 1000)[:2]
    assert bm <= cls.SAT_TILE_BM and bn <= cls.SAT_TILE_BN, (bm, bn)


def test_b200_tmem_budget_matches_measured_hardware() -> None:
    # _tmem_bytes is a deliberate OVER-estimate in BYTES, checked against TMEM_BUDGET exactly like
    # _smem_bytes is checked against SMEM_BUDGET. Hardware reports tensor memory in COLUMNS (limit
    # 512), so the ground truth below is measured tmem_size in columns and what must agree is the
    # VERDICT: fits / does not fit. Ground truth is tmem_size read off compiled sm100 kernel metadata
    # in the worst case, i.e. with the A operand promoted into tensor memory.
    cls = TritonB200FormulaMatmulHeuristic
    hw_columns = {  # (bm, bn, bk) bf16 -> measured tmem columns, A promoted; hardware limit is 512
        (64, 64, 16): 128,
        (64, 128, 16): 256,
        (64, 256, 16): 512,
        (64, 512, 16): 512,
        (64, 1024, 16): 520,
        (128, 128, 16): 256,
        (128, 256, 16): 512,
        (128, 512, 16): 520,
        (128, 512, 32): 528,
        (128, 512, 64): 544,
        (128, 1024, 16): 1032,
        (256, 128, 16): 512,
        (256, 256, 16): 528,
        (
            256,
            256,
            32,
        ): 544,  # the CI failure: 512 for the acc + 32 for the promoted A operand
        (256, 256, 64): 576,
        (256, 256, 128): 640,
        (256, 512, 16): 1040,
    }
    for (bm, bn, bk), cols in hw_columns.items():
        fits_model = cls._tmem_bytes(bm, bn, bk, 2) <= cls.TMEM_BUDGET
        fits_hardware = cols <= 512
        assert fits_model == fits_hardware, (bm, bn, bk, cols)
    # Specifically: the [256,256] square is rejected, the wide [128,256] tile accepted.
    assert cls._tmem_bytes(256, 256, 16, 2) > cls.TMEM_BUDGET
    assert cls._tmem_bytes(128, 256, 64, 2) <= cls.TMEM_BUDGET
    # The budget is the real capacity: 128 lanes x 512 columns x 32 bit.
    assert cls.TMEM_BUDGET == 128 * 512 * 4
    # ...and a [256,256] fp32 accumulator alone exactly fills it, which is why no promoted A operand
    # can ever coexist with that square.
    assert cls.TMEM_BUDGET == 256 * 256 * 4
    # A tile too small for tcgen05 uses no tensor memory at all (measured tmem_size == 0), so it must
    # be charged nothing -- otherwise tiny decode tiles get rejected for a resource they never touch.
    assert cls._tmem_bytes(32, 4096, 16, 2) == 0
    assert cls._tmem_bytes(16, 512, 32, 4) == 0


def test_b200_smem_bytes_bounds_measured_hardware() -> None:
    # The epilogue's accumulator staging buffer (bm*bn*4) is invisible to the operand-ring formula and
    # is independent of num_stages, so a tile can fit the ring and still exceed SMEM. Values are
    # measured `shared` from compiled sm100 metadata; the model must be an UPPER bound on each.
    cls = TritonB200FormulaMatmulHeuristic
    # (bm, bn, bk, itemsize, num_stages) -> measured shared bytes (worst case over epilogue variants)
    measured = {
        (64, 64, 64, 2, 6): 32784,
        (128, 128, 64, 2, 6): 65552,
        (128, 256, 64, 2, 6): 131072,
        (256, 128, 32, 2, 6): 131072,
        (256, 256, 32, 2, 6): 262144,
        (
            256,
            256,
            16,
            2,
            6,
        ): 262144,  # ring shrinks with bk, the epilogue term does NOT
        (128, 128, 128, 4, 6): 786448,
    }
    for (bm, bn, bk, itemsize, ns), shared in measured.items():
        assert cls._smem_bytes(bm, bn, bk, itemsize, ns) >= shared, (bm, bn, bk, ns)
    # [256,256] is rejected by the epilogue term alone, at ANY bk -- the ring cannot rescue it.
    for bk in (16, 32, 64):
        assert cls._smem_bytes(256, 256, bk, 2, 1) > cls.SMEM_BUDGET
    # The slack is load-bearing: without it an otherwise-exact bound misses by the 16-byte mbarriers.
    assert cls.SMEM_SLACK >= 16


def test_sm90_conservative_accounting_is_inert() -> None:
    # sm90 must stay byte-identical, but NOT because the epilogue term is Blackwell-specific -- it is
    # not. Measured on an sm90 target, a [128,256,64] bf16 dot with an fp32 output reports
    # shared=131072 == bm*bn*4, exactly as sm100 does, so EPILOGUE_ACC_ITEMSIZE is set on the base.
    # What is sm100-only is ENFORCING the budget by shrinking the tile.
    cls = TritonH100MatmulHeuristic
    assert cls.TMEM_BUDGET is None  # no tensor memory on sm90
    assert cls._tmem_bytes(256, 256, 32, 2) == 0
    assert cls.EPILOGUE_ACC_ITEMSIZE == 4  # the term is arch-independent...
    assert (
        cls.ENFORCE_SMEM_BUDGET is False
    )  # ...but sm90 does not enforce it (model over-estimates)
    assert cls.SMEM_SLACK == 0

    # The epilogue term is a NO-OP on sm90 for an arithmetic reason: the accumulator lives in the
    # register file, so ACC_BUDGET caps the emittable tile at bm*bn == ACC_BUDGET. Binding would need
    # bm*bn * 4 > SMEM_BUDGET -- unreachable. Pin both halves of that argument so a future ACC_BUDGET
    # bump cannot silently make the term start binding on sm90 unnoticed.
    assert cls.ACC_BUDGET * cls.EPILOGUE_ACC_ITEMSIZE <= cls.SMEM_BUDGET
    emitted = {
        cls._matmul_tile(m, n, k, itemsize, 132, pinned_grid)[:2]
        for m in (1, 16, 256, 4096, 65536)
        for n in (1, 16, 256, 4096, 65536)
        for k in (16, 256, 4096)
        for itemsize in (1, 2, 4)
        for pinned_grid in (1, 4, 1000)
    }
    assert max(bm * bn for bm, bn in emitted) <= cls.ACC_BUDGET

    # And the sm90 tile is unchanged: still the register-budget [128, 256].
    assert cls._matmul_tile(4096, 4096, 4096, 2, 132, 1)[:3] == (128, 256, 64)


# ---------------------------------------------------------------------------
# Generalized axis freedom, graded occupancy, and whole-kernel resources.
#
# These pin the mandatory Section-3 capabilities individually, so a poor
# curriculum result is attributable to policy rather than to an implementation
# bug in the fact layer, the projection, the ranking, or the resource model.
# ---------------------------------------------------------------------------

_FML = TritonB200FormulaMatmulHeuristic


def _axes(
    m: DotAxisKind = DotAxisKind.TUNABLE_TILED,
    n: DotAxisKind = DotAxisKind.TUNABLE_TILED,
    k: DotAxisKind = DotAxisKind.TUNABLE_TILED,
    *,
    m_extent: int | None = 1024,
    n_extent: int | None = 1024,
    k_extent: int | None = 1024,
) -> DotAxes:
    return DotAxes(m, n, k, m_extent, n_extent, k_extent)


class _BlockSizesStub(list):
    """A ``ConfigSpec.block_sizes`` stand-in: indexable + sized like the real one, plus
    ``valid_block_ids()``."""

    def __init__(self, block_ids: list[int]) -> None:
        super().__init__(
            SimpleNamespace(block_id=b, min_size=1, max_size=4096, autotuner_min=1)
            for b in block_ids
        )
        self._ids = list(block_ids)

    def valid_block_ids(self) -> list[int]:
        return list(self._ids)


def _block_sizes_stub(block_ids: list[int]) -> _BlockSizesStub:
    return _BlockSizesStub(block_ids)


def _generalized_spec(
    fact: MatmulFact,
    axes: DotAxes,
    *,
    valid_block_ids: list[int],
    grid_block_ids: tuple[int, ...] = (),
) -> SimpleNamespace:
    mm = SimpleNamespace(
        matmuls=(fact,),
        axes=(axes,),
        sites=(SimpleNamespace(graph_id=0, loop_trips=1, updates_carry=False),),
        knob_users=(),
        outer_grid=1,
        sequential_loop_trips=1,
        live_tiles=(),
        live_dot_outputs=(),
        pipelined_regions=(),
        resident_regions=(),
        n_dot_nodes=1,
        attribution_complete=True,
    )
    return SimpleNamespace(
        matmul_facts=[fact],
        multi_matmul_fact=mm,
        block_sizes=_block_sizes_stub(valid_block_ids),
        grid_block_ids=grid_block_ids,
    )


def test_generalized_gate_admits_a_fixed_contraction_axis() -> None:
    """A dot whose K is a specialized full extent has no ``block_k`` to set. The incumbent
    gate declines it for that reason alone; the generalized gate admits it, because a fixed
    axis is a smaller set of knobs and not a smaller problem."""
    fact = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=0,
        n_block_id=1,
        k_block_id=4,  # registered but NOT tunable
        static_m=256,
        static_n=256,
        static_k=64,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )
    spec = _generalized_spec(
        fact,
        _axes(k=DotAxisKind.FIXED_FULL_EXTENT, k_extent=64),
        valid_block_ids=[0, 1],
    )
    assert _generalized_static_matmul_fact(spec) is fact
    # ...and the incumbent gate still declines it, which is what made this widening needed.
    assert _batched_static_matmul_fact(spec) is None


def test_generalized_gate_admits_zero_tunable_dot_axes() -> None:
    """A kernel that exposes no tile at all still wants num_warps / num_stages; the
    alternative is the bare fragment default."""
    fact = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=None,
        n_block_id=None,
        k_block_id=None,
        static_m=64,
        static_n=64,
        static_k=64,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )
    spec = _generalized_spec(
        fact,
        _axes(
            DotAxisKind.FIXED_FULL_EXTENT,
            DotAxisKind.FIXED_FULL_EXTENT,
            DotAxisKind.FIXED_FULL_EXTENT,
            m_extent=64,
            n_extent=64,
            k_extent=64,
        ),
        valid_block_ids=[],
    )
    assert _generalized_static_matmul_fact(spec) is fact


def test_generalized_gate_declines_two_axes_sharing_one_knob() -> None:
    """Two tunable axes on one block id is a genuine conflict that must be RANKED, which is
    front end 2's job -- front end 1 has no way to arbitrate it."""
    fact = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=0,
        n_block_id=1,
        k_block_id=1,  # same knob as N
        static_m=256,
        static_n=256,
        static_k=256,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )
    spec = _generalized_spec(fact, _axes(), valid_block_ids=[0, 1])
    assert _generalized_static_matmul_fact(spec) is None


def test_generalized_gate_declines_an_unknown_extent() -> None:
    """No static extent means nothing can be sized; a dynamic/jagged dot must not be
    silently configured from a guess."""
    fact = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=0,
        n_block_id=1,
        k_block_id=2,
        static_m=256,
        static_n=256,
        static_k=None,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )
    spec = _generalized_spec(
        fact, _axes(k=DotAxisKind.UNKNOWN, k_extent=None), valid_block_ids=[0, 1, 2]
    )
    assert _generalized_static_matmul_fact(spec) is None


def test_tmem_is_counted_in_columns_and_sums_over_live_accumulators() -> None:
    """tcgen05 tensor memory is allocated as 128 lanes x N columns of 32 bits, so a tile
    costs ``ceil(bm/128) * bn`` COLUMNS and a bm<128 accumulator costs the same as a
    full-lane one. Measured on B200: a kernel's ``tmem_size`` equals its accumulator's N
    extent exactly ([64,64] -> 64, [128,128] -> 128, [128,256] -> 256)."""
    assert _FML._tmem_columns([(64, 64, 2)]) == 64
    assert _FML._tmem_columns([(128, 128, 2)]) == 128
    assert _FML._tmem_columns([(128, 256, 2)]) == 256
    # A byte model divides by the lanes a narrow accumulator does not use; the column model
    # does not, which is the whole point.
    assert _FML._tmem_columns([(64, 256, 2)]) == 256
    # bm past a lane group needs a second one.
    assert _FML._tmem_columns([(256, 256, 2)]) == 512
    # Live accumulators ADD -- this is the measured failure
    # (``tensor memory, Required: 768, limit 512``) reproduced as arithmetic.
    assert _FML._tmem_columns([(128, 256, 2)] * 3) == 768
    assert _FML._tmem_columns([(128, 256, 2)] * 3) > _FML.TMEM_COLUMN_BUDGET
    # ...and one accumulator under the incumbent caps never binds, so the single-GEMM path
    # is unaffected by adding this check.
    assert _FML._tmem_columns([(128, _FML.BASE_BN_CAP, 2)]) <= _FML.TMEM_COLUMN_BUDGET
    # Below the tcgen05 minimum the dot uses no tensor memory at all (measured
    # ``tmem_size == 0``), so charging it would reject a tiny tile for a resource it never
    # touches.
    assert _FML._tmem_columns([(32, 256, 2)]) == 0
    # sm90 has no tensor memory: the check must be completely inert there.
    assert TritonH100MatmulHeuristic._tmem_columns([(128, 256, 2)] * 8) == 0


def test_register_accumulator_charge_depends_on_the_warpgroup() -> None:
    """tcgen05 MMA issues per warpgroup, so below 4 warps that path is unavailable and EVERY
    accumulator falls back into the register file -- the budget the tensor-memory model
    assumed away. Measured spill counts across a warp sweep at a byte-identical tile stop
    exactly where the warpgroup becomes available (1: 1148 spills, 2: 84, 4: 0, 8: 0)."""
    tiles = [(64, 64, 2), (64, 64, 2)]
    assert _FML._register_acc_bytes(tiles, 1) == 2 * 64 * 64 * 4
    assert _FML._register_acc_bytes(tiles, 2) == 2 * 64 * 64 * 4
    assert _FML._register_acc_bytes(tiles, 4) == 0
    # A dot below the tcgen05 minimum stays register-resident at ANY warp count.
    assert _FML._register_acc_bytes([(32, 64, 2)], 8) == 32 * 64 * 4


def test_warps_rise_with_the_register_resident_live_set() -> None:
    """The property that decides the one-warp penalty is total live fp32 dot-output bytes
    against the one-warp register file (32 x 255 x 4 = 32640 B), NOT the contraction count:
    measured at matched live bytes, matched MMA FLOPs and matched HBM traffic, the one-warp
    penalty is 8.29x for ONE contraction, 2.69x for two and 4.75x for four, and a dot-free
    control spills 490 registers and loses 4.0x with no MMA anywhere.

    The ladder climbs 1 -> 2 -> 4 -> 8. Losing tcgen05 below a warpgroup is confirmed in the
    emitted PTX but is not itself a penalty -- at 16 KiB live, one warp on ``mma.sync`` beats
    four warps on tcgen05 -- and two warps is the hand-tuned answer in 11 of 18
    ``chunk_cumsum_gc`` cells."""
    # ONE 64x64 fp32 accumulator is 16 KiB and fits one warp: leave the draft alone.
    assert _FML._warps_for_live_set(1, [(64, 64, 2)]) == 1
    # TWO of them are 32 KiB against a 31.9 KiB file: one step up the ladder, not a jump.
    assert _FML._warps_for_live_set(1, [(64, 64, 2)] * 2) == 2
    # Accumulators are charged at FACE VALUE (a dot's output extent is exact), unlike the
    # FX-derived non-accumulator set. Discounting them too left one curriculum case at one
    # warp with 430 spills, 1.62x slower than the same tile at four warps.
    assert _FML._warps_for_live_set(1, [(64, 64, 2)] * 4) == 4
    # No accumulators recorded -> no opinion, so the incumbent ramp is preserved.
    assert _FML._warps_for_live_set(1, []) == 1
    assert _FML._warps_for_live_set(8, []) == 8
    # Never lowers what the tile ramp asked for.
    assert _FML._warps_for_live_set(8, [(64, 64, 2)] * 4) == 8
    # Where tensor memory cannot rescue it (bm below the tcgen05 minimum), it keeps climbing.
    assert _FML._warps_for_live_set(1, [(32, 2048, 2)] * 4) == _FML.MAX_NUM_WARPS
    # The NON-accumulator live set comes from the FX peak, which over-counts, so it is
    # discounted by the calibrated factor before the comparison.
    assert _FML._warps_for_live_set(1, [], other_register_bytes=400000) == 8
    assert _FML._warps_for_live_set(1, [], other_register_bytes=1024) == 1


def test_graded_stage_depth_falls_off_with_outer_parallelism() -> None:
    """Depth is bought with shared memory per CTA, and shared memory per CTA is what limits
    how many CTAs an SM holds. Below one wave there is no co-residency to protect and depth
    is the only latency hiding available; above it, every extra stage evicts a CTA. So the
    depth must fall off GRADUALLY with the grid -- which the incumbent single threshold at
    ``SAT_WAVES * num_sm`` cannot express, and which matches the hand-tuned corpus (outer
    grid 32 -> 8-11 stages, 96 -> 3-4, 256 -> 2-4, >=1024 -> 2)."""
    per_stage = 16 * 1024

    def smem_of(stages: int) -> int:
        return per_stage * stages

    depths = [
        _FML._graded_stage_depth(smem_of, loop_trips=256, grid=grid, num_sm=148)
        for grid in (32, 96, 148, 296, 1024, 16384)
    ]
    assert depths == sorted(depths, reverse=True), depths
    assert depths[0] > depths[-1]
    # The divisor is CLAMPED at GRADED_MAX_CTAS_PER_SM, so the gradient SATURATES rather
    # than running to the floor: a grid far above the machine size does not demand a
    # matching number of simultaneously-resident CTAs (the excess queues), and dividing by
    # the raw wave count instead collapsed every large-grid kernel to a single stage
    # (measured: an outer grid of 8192 on 148 SMs gives a 4 KiB per-CTA share).
    assert depths[-1] == depths[-2]
    assert (
        _FML._graded_stage_depth(smem_of, loop_trips=256, grid=10**6, num_sm=148)
        == depths[-1]
    )
    # An empty machine reaches depths the incumbent MAX_STAGES ceiling cannot express.
    assert depths[0] > _FML.MAX_STAGES or _FML.HW_MAX_STAGES <= _FML.MAX_STAGES
    assert depths[0] <= _FML.HW_MAX_STAGES


def test_graded_stage_depth_is_capped_by_the_loop_it_pipelines() -> None:
    """There is no point pipelining deeper than the loop is long."""

    def cheap(stages: int) -> int:
        return 1024 * stages

    assert _FML._graded_stage_depth(cheap, loop_trips=3, grid=1, num_sm=148) == 3
    assert _FML._graded_stage_depth(cheap, loop_trips=1, grid=1, num_sm=148) == 2


def test_multi_matmul_ranking_prefers_a_carried_accumulator_then_work() -> None:
    """A dot feeding a loop-carried accumulator holds that accumulator resident for the whole
    loop, so its tile sets the kernel's whole-loop footprint -- hence ranking priority. But
    it is a PREFERENCE: dimensions and execution count must also matter, and a kernel with no
    carried accumulator has to rank purely on work."""
    big = MatmulFact(2, 2, 0, 1, 2, 256, 256, 256, torch.bfloat16, torch.bfloat16)
    small = MatmulFact(2, 2, 0, 1, 2, 64, 64, 64, torch.bfloat16, torch.bfloat16)

    def mm(carry: tuple[bool, ...], trips: tuple[int, ...]) -> SimpleNamespace:
        facts = (big, small)
        ns = SimpleNamespace(
            matmuls=facts,
            axes=(_axes(), _axes()),
            sites=tuple(
                SimpleNamespace(graph_id=0, loop_trips=t, updates_carry=c)
                for c, t in zip(carry, trips, strict=True)
            ),
            attribution_complete=True,
        )
        ns.dot_work = lambda i: (
            (facts[i].static_m or 1)
            * (facts[i].static_n or 1)
            * (facts[i].static_k or 1)
            * max(1, ns.sites[i].loop_trips)
        )
        return ns

    # The carried dot wins even though it does far less work.
    f = mm((False, True), (1, 1))
    assert _MULTI._rank_key(f, 1) > _MULTI._rank_key(f, 0)
    # With no carry anywhere, work decides.
    f = mm((False, False), (1, 1))
    assert _MULTI._rank_key(f, 0) > _MULTI._rank_key(f, 1)
    # Execution count is part of work, so a small dot run many times can outrank a big one.
    f = mm((False, False), (1, 4096))
    assert _MULTI._rank_key(f, 1) > _MULTI._rank_key(f, 0)
    # Untrusted attribution must collapse the carry term for EVERY dot equally, so ranking
    # degrades to pure work rather than to an arbitrary order.
    f = mm((False, True), (1, 1))
    f.attribution_complete = False
    assert _MULTI._rank_key(f, 0) > _MULTI._rank_key(f, 1)


def test_projection_is_a_no_op_for_a_clean_gemm() -> None:
    """With three tunable axes and no extra live accumulator, ``_tile_for_dot`` must return
    the incumbent proposal untouched -- that is what makes the whole widening safe for the
    GEMM/BMM/split-K workloads it was not measured on."""
    fact = _matmul_fact(static_m=4096, static_n=4096, static_k=4096)
    spec = _generalized_spec(fact, _axes(), valid_block_ids=[0, 1, 2])
    env = SimpleNamespace(config_spec=spec, block_sizes=[])
    proposal = _FML._matmul_tile(4096, 4096, 4096, 2, 148, 1)
    assert _FML._tile_for_dot(env, fact, _axes(), 2, 148, 1) == proposal


def test_sm90_keeps_the_incumbent_gate_and_every_switch_off() -> None:
    """Every measurement behind the generalized machinery is B200, so sm90 must be a
    byte-identical freeze: same eligibility precondition, no graded stages, no work-aware
    warps, no tensor-memory column budget."""
    cls = TritonH100MatmulHeuristic
    assert cls.GENERALIZED_AXES is False
    assert cls.GRADED_STAGES is False
    assert cls.WORK_AWARE_WARPS is False
    assert cls.TMEM_COLUMN_BUDGET is None
    fixed = MatmulFact(
        lhs_ndim=2,
        rhs_ndim=2,
        m_block_id=0,
        n_block_id=1,
        k_block_id=4,
        static_m=256,
        static_n=256,
        static_k=64,
        lhs_dtype=torch.bfloat16,
        rhs_dtype=torch.bfloat16,
    )
    spec = _generalized_spec(
        fixed,
        _axes(k=DotAxisKind.FIXED_FULL_EXTENT, k_extent=64),
        valid_block_ids=[0, 1],
    )
    # sm90 routes through the incumbent gate, which declines the fixed-axis dot.
    assert cls._eligible_fact(spec) is None
    # ...while sm100 admits it.
    assert _FML._eligible_fact(spec) is fixed


def test_multi_matmul_front_end_declines_whatever_front_end_one_owns() -> None:
    """Exactly one of the two front ends may own a kernel, so promotion is unambiguous
    without either of them having to know the other's policy."""
    fact = _matmul_fact()
    spec = _generalized_spec(fact, _axes(), valid_block_ids=[0, 1, 2])
    env = SimpleNamespace(
        config_spec=spec,
        device=torch.device("cuda"),
        settings=SimpleNamespace(),
    )
    with patch(
        "helion._compiler.autotuner_heuristics.triton.matches_hardware",
        return_value=True,
    ):
        assert _MULTI.is_eligible(env, None) is False
    # Registration order must put the multi front end AFTER the single one, since the
    # compiler-default loop is last-promote-wins.
    from helion._compiler.autotuner_heuristics import HEURISTICS_BY_BACKEND

    triton_order = HEURISTICS_BY_BACKEND["triton"]
    assert triton_order.index(_MULTI) > triton_order.index(_FML)


def test_warp_floor_is_conditioned_on_the_live_set_not_the_dot_count() -> None:
    """The multi-contraction warp floor must NOT fire on dot count alone.

    Adversarial review built a 4-contraction kernel chained so only ONE output is ever live:
    it spills nothing at any warp count and is fastest at ONE warp by +7.0%. An unconditional
    floor overrides that, and costs +72.1% on a curriculum cell whose hand-tuned num_warps is
    1 in all 10 of its cases. Conditioned on the live-set estimate instead, the floor agrees
    with the answer key on 23 of the 24 cases it touches rather than 13, and changes nothing
    on the other 408. So: one live accumulator -> never floor, however many dots exist."""
    env = SimpleNamespace(
        config_spec=SimpleNamespace(
            multi_matmul_fact=SimpleNamespace(
                live_dot_outputs=(), live_tiles=(), matmuls=(), axes=()
            ),
            block_sizes=_block_sizes_stub([]),
        ),
        block_sizes=[],
    )
    # No recorded accumulators at all -> no opinion.
    assert _MULTI._needs_warp_floor(env, []) is False

    def env_with(tiles: tuple) -> SimpleNamespace:
        from helion.autotuner.config_spec import LiveTile

        outs = tuple(
            LiveTile(
                dim_block_ids=(None, None),
                static_dims=(rows, cols),
                itemsize=4,
                kind="dot_out",
            )
            for rows, cols in tiles
        )
        return SimpleNamespace(
            config_spec=SimpleNamespace(
                multi_matmul_fact=SimpleNamespace(
                    live_dot_outputs=outs, live_tiles=(), matmuls=(), axes=()
                ),
                block_sizes=_block_sizes_stub([]),
            ),
            block_sizes=[],
        )

    # ONE live 64x64 fp32 accumulator fits one warp: no floor, no matter the dot count.
    assert _MULTI._needs_warp_floor(env_with(((64, 64),)), []) is False
    # Two of them are 32 KiB against the 31.9 KiB one-warp file: floor applies.
    assert _MULTI._needs_warp_floor(env_with(((64, 64), (64, 64))), []) is True
    # Two TINY live accumulators still fit, so the count alone must not trigger it.
    assert _MULTI._needs_warp_floor(env_with(((16, 16), (16, 16))), []) is False


def _live(kind: str, *block_ids: int | None) -> LiveTile:
    return LiveTile(
        dim_block_ids=tuple(block_ids),
        static_dims=tuple(None for _ in block_ids),
        itemsize=2,
        kind=kind,
    )


def _knob_spec(
    *,
    knob_users: tuple[tuple[int, tuple[tuple[int, str], ...]], ...],
    block_ids: list[int],
    extents: dict[int, int],
    grid_block_ids: tuple[int, ...] = (),
    pipelined_regions: tuple[tuple[LiveTile, ...], ...] = (),
) -> SimpleNamespace:
    """An ``env`` stub carrying just what ``_apply_knob_roles`` reads."""
    mm = SimpleNamespace(
        matmuls=(),
        axes=(),
        sites=(),
        knob_users=knob_users,
        outer_grid=1,
        sequential_loop_trips=1,
        live_tiles=(),
        live_dot_outputs=(),
        pipelined_regions=pipelined_regions,
        resident_regions=(),
        n_dot_nodes=len(knob_users),
        attribution_complete=True,
    )
    spec = SimpleNamespace(
        matmul_facts=[],
        multi_matmul_fact=mm,
        block_sizes=_block_sizes_stub(block_ids),
        grid_block_ids=grid_block_ids,
        _base_default_config=lambda: SimpleNamespace(config={}),
    )
    env = SimpleNamespace(
        config_spec=spec,
        block_sizes=[
            SimpleNamespace(
                size=extents.get(b, 1),
                block_size_source=SimpleNamespace(from_config=lambda *_a: None),
            )
            for b in range(max(extents or {0: 0}) + 1)
        ],
        size_hint=lambda v: int(v),
    )
    return env, mm


def test_launch_grid_counts_only_grid_axes() -> None:
    """A knob that is walked by a SEQUENTIAL loop contributes iterations, not programs. The
    budget formula's wave model counts every M/N tile as a program, which reports a saturated
    machine for a kernel that launches a handful of CTAs."""
    env, _mm = _knob_spec(
        knob_users=((0, ((0, "n"),)), (1, ((0, "m"),))),
        block_ids=[0, 1],
        extents={0: 128, 1: 128},
        grid_block_ids=(1,),
    )
    # Only block id 1 is a grid axis: halving the LOOP axis 0 must not change the grid,
    # halving the grid axis must double it.
    assert _MULTI._launch_grid(env, [128, 128]) == 1
    assert _MULTI._launch_grid(env, [32, 128]) == 1
    assert _MULTI._launch_grid(env, [128, 32]) == 4


def test_knob_amortization_separates_reuse_free_from_reuse_bearing() -> None:
    """The discriminator for whether growing a tile buys anything: does its loop region stage
    a load the knob does NOT span? If every load spans the knob, bytes and MMA work both
    scale with the tile and arithmetic intensity is constant in it."""
    reuse_free = (_live("load", 9, 4), _live("store", 9, 4))
    reuse_bearing = (_live("load", 9, 4), _live("load", 9, 3), _live("store", 9, 4))
    _env, mm = _knob_spec(
        knob_users=((4, ((0, "n"),)),),
        block_ids=[4],
        extents={4: 128},
        pipelined_regions=(reuse_free,),
    )
    assert _MULTI._knob_amortizes(mm, 4) is False
    _env, mm = _knob_spec(
        knob_users=((4, ((0, "n"),)),),
        block_ids=[4],
        extents={4: 128},
        pipelined_regions=(reuse_bearing,),
    )
    # Block id 3 is re-fetched for every iteration of 4, so growing 4 amortizes it.
    assert _MULTI._knob_amortizes(mm, 4) is True
    # ...and symmetrically, 3's own loop re-fetches the 4-spanning load, so 3 amortizes too.
    assert _MULTI._knob_amortizes(mm, 3) is True


def test_reuse_free_output_knob_drops_to_the_allocation_floor() -> None:
    """A reuse-free knob that sizes a dot OUTPUT extent buys no arithmetic intensity while
    the fp32 accumulator, the register-resident intermediates and the store staging all scale
    with it, so it goes to the tcgen05 allocation granularity. A reuse-BEARING knob in the
    same position must be left alone."""
    env, _mm = _knob_spec(
        knob_users=((4, ((0, "n"),)),),
        block_ids=[4],
        extents={4: 128},
        pipelined_regions=((_live("load", 9, 4), _live("store", 9, 4)),),
    )
    block_sizes = [128]
    _MULTI._apply_knob_roles(env, env.config_spec.multi_matmul_fact, block_sizes, 148)
    assert block_sizes == [_MULTI.TMEM_ALLOC_COLUMNS]

    env, _mm = _knob_spec(
        knob_users=((4, ((0, "n"),)),),
        block_ids=[4],
        extents={4: 128},
        pipelined_regions=((_live("load", 9, 4), _live("load", 9, 3)),),
    )
    block_sizes = [128]
    _MULTI._apply_knob_roles(env, env.config_spec.multi_matmul_fact, block_sizes, 148)
    assert block_sizes == [128]


def test_contraction_only_knob_is_left_where_the_formula_put_it() -> None:
    """Growing a contraction-only knob to its extent measured NET NEGATIVE on the curriculum
    (right on some kernels, 0.84x-0.89x on others), so the rule is off and a knob no dot
    claims as M or N must come out unchanged."""
    assert _MULTI.ROLE_CONTRACTION_GROW is False
    env, _mm = _knob_spec(
        knob_users=((4, ((0, "k"), (1, "k"))),),
        block_ids=[4],
        extents={4: 128},
        pipelined_regions=((_live("load", 9, 4),),),
    )
    block_sizes = [32]
    _MULTI._apply_knob_roles(env, env.config_spec.multi_matmul_fact, block_sizes, 148)
    assert block_sizes == [32]


def test_grid_knob_shrinks_until_the_launch_grid_fills_a_wave() -> None:
    """A knob that IS the grid trades tile area against occupancy, so it is sized by the
    machine: shrink while the launch grid is under one wave, and stop at the allocation
    granularity rather than at the dot minimum."""
    env, _mm = _knob_spec(
        knob_users=((0, ((0, "m"), (1, "n"))),),
        block_ids=[0],
        extents={0: 8192},
        grid_block_ids=(0,),
        pipelined_regions=((_live("load", 9, 0), _live("load", 9, 7)),),
    )
    block_sizes = [8192]
    _MULTI._apply_knob_roles(env, env.config_spec.multi_matmul_fact, block_sizes, 148)
    # 8192 / 64 = 128 programs >= 0.8 * 148, so the shrink stops there rather than
    # continuing to the floor.
    assert block_sizes == [64]
    # When even the floor cannot fill a wave, the granularity floor wins: a narrower
    # accumulator reserves the same tensor memory while issuing less MMA work.
    env, _mm = _knob_spec(
        knob_users=((0, ((0, "m"), (1, "n"))),),
        block_ids=[0],
        extents={0: 1024},
        grid_block_ids=(0,),
        pipelined_regions=((_live("load", 9, 0), _live("load", 9, 7)),),
    )
    block_sizes = [1024]
    _MULTI._apply_knob_roles(env, env.config_spec.multi_matmul_fact, block_sizes, 148)
    assert block_sizes == [_MULTI.TMEM_ALLOC_COLUMNS]
    # A grid that already covers the machine is left alone.
    env, _mm = _knob_spec(
        knob_users=((0, ((0, "m"),)),),
        block_ids=[0],
        extents={0: 65536},
        grid_block_ids=(0,),
        pipelined_regions=((_live("load", 9, 0), _live("load", 9, 7)),),
    )
    block_sizes = [128]
    _MULTI._apply_knob_roles(env, env.config_spec.multi_matmul_fact, block_sizes, 148)
    assert block_sizes == [128]
