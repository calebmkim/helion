from __future__ import annotations

import enum
import functools
import hashlib
import itertools
import logging
import math
import operator
import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import NamedTuple
from typing import cast

import torch
from torch._inductor.runtime.runtime_utils import next_power_of_2
import torch.distributed as dist

from .._compat import _regs_per_block
from .._compat import device_num_sm
from .._compat import num_compute_units
from .._compat import supports_amd_cdna_tunables
from .._compat import supports_maxnreg
from .._compat import supports_tensor_descriptor
from .._compat import target_device_capability as get_target_device_capability
from .._compat import warps_to_threads
from .._compiler.cute.cute_flash import FLASH_CAUSAL_KV_ORDER_KEY
from .._compiler.cute.cute_flash import FLASH_CAUSAL_LOOP_SPLIT_KEY
from .._compiler.cute.cute_flash import FLASH_CONFIG_KEYS
from .._compiler.cute.cute_flash import FLASH_CORR_REGS_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_FREQ_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_OFFSET0_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_OFFSET_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_RES_KEY
from .._compiler.cute.cute_flash import FLASH_E2E_SCHEDULE_KEY
from .._compiler.cute.cute_flash import FLASH_EPI_TMA_KEY
from .._compiler.cute.cute_flash import FLASH_EXP2_IMPL_KEY
from .._compiler.cute.cute_flash import FLASH_MASKED_E2E_SCHEDULE_KEY
from .._compiler.cute.cute_flash import FLASH_OTHER_REGS_KEY
from .._compiler.cute.cute_flash import FLASH_ROLE_MAP_KEY
from .._compiler.cute.cute_flash import FLASH_SOFTMAX_REGS_KEY
from .._compiler.cute.cute_flash import FLASH_TOPOLOGY_KEY
from .._compiler.cute.cute_flash import _flash_causal_hd64_seed_num_kv_supported
from .._compiler.cute.cute_flash import _flash_causal_hd64_seed_offset0
from .._compiler.cute.cute_flash import _flash_causal_hd64_seed_params
from .._compiler.cute.cute_flash import _flash_e2e_offset_period
from .._compiler.cute.cute_flash import _flash_e2e_schedule_default
from .._compiler.cute.cute_flash import _flash_masked_e2e_schedule_params
from .._compiler.cute.cute_flash import _flash_normalize_e2e_offset
from .._compiler.cute.cute_flash import _flash_normalize_e2e_params
from .._compiler.cute.cute_flash import _flash_parse_e2e_schedule
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_STRATEGY_CONFIG_KEYS
from .._compiler.cute.tcgen05_config import CUTE_TCGEN05_TUNABLE_KEYS
from .._compiler.cute.tcgen05_config import CuteTcgen05Config
from .._compiler.cute.tcgen05_config import Tcgen05AbStagesThreeSearchConstraints
from .._compiler.cute.tcgen05_config import Tcgen05ClusterM2SearchConstraints
from .._compiler.cute.tcgen05_constants import TCGEN05_TWO_CTA_MAX_K_TILES
from .._compiler.cute.tv_layout import NDTILE_TV_CHOICES
from .._compiler.cute.tv_layout import ONLINE_DEFER_CHOICES
from .._compiler.cute.tv_layout import REDUCTION_RELOAD_CHOICES
from .._compiler.cute.tv_layout import ROW_RESIDENCY_CHOICES
from .._compiler.cute.tv_layout import ROW_RESIDENCY_GMEM
from .._compiler.cute.tv_layout import THREADS_PER_ROW_CHOICES
from .._compiler.cute.tv_layout import ndtile_tv_for
from .._compiler.cute.tv_layout import online_defer_for
from .._compiler.cute.tv_layout import row_residency_for
from .._compiler.cute.tv_layout import row_residency_from_legacy
from .._compiler.cute.tv_layout import threads_per_row_for
from ..exc import InvalidConfig
from ..runtime.triton.launcher import get_num_xcd
from .block_id_sequence import BlockIdSequence
from .block_id_sequence import _BlockIdItem
from .block_id_sequence import _PowerOfTwoBlockIdItem
from .config_fragment import BlockSizeFragment
from .config_fragment import BooleanFragment
from .config_fragment import ConfigSpecFragment
from .config_fragment import EnumFragment
from .config_fragment import IntegerFragment
from .config_fragment import ListOf
from .config_fragment import NumThreadsFragment
from .config_fragment import NumWarpsFragment
from .config_fragment import PermutationFragment
from .config_fragment import PowerOfTwoFragment
from .config_fragment import assert_integer_power_of_two
import helion

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from .._compiler.backend import Backend
    from ..runtime.config import IndexingLiteral
    from ..runtime.config import PidTypeLiteral
    from .config_generation import ConfigGeneration

log = logging.getLogger(__name__)

_TARGET_DEVICE_CAPABILITY_UNSET = object()


class TensorNumelConstraint(NamedTuple):
    """Tensor element count must stay within Triton's max numel limit."""

    check_fn: Callable[..., bool]
    block_indices: tuple[int, ...]
    expr_str: str


class MatmulFact(NamedTuple):
    """Shape facts recorded when matmul requirements are applied."""

    lhs_ndim: int
    rhs_ndim: int
    m_block_id: int | None
    n_block_id: int | None
    k_block_id: int | None
    static_m: int | None
    static_n: int | None
    static_k: int | None
    lhs_dtype: torch.dtype
    rhs_dtype: torch.dtype


class ReductionCategory(enum.Enum):
    """How a reduction axis maps onto the program grid — one category per reduction.

    - ``FULL_SLICE`` — the whole axis is reduced within one program (``x[m, :]``); not on the grid.
    - ``FULL_GRID`` — a full-extent axis on the grid, block == extent (fully resident per program).
    - ``GRID_TILE`` — a grid axis reduced over but NOT full-extent: the grid parallelizes the
      reduction across programs, so the whole-axis size is not a per-program extent.
    - ``USER_TILE`` — an inner sequential ``hl.tile`` the user wrote over the reduction axis.
    - ``DECLINED`` — no static extent (e.g. jagged / data-dependent); recorded but never sized.
    """

    FULL_SLICE = "full_slice"
    FULL_GRID = "full_grid"
    GRID_TILE = "grid_tile"
    USER_TILE = "user_tile"
    DECLINED = "declined"


# Categories the seed sizes a per-program reduction extent for. GRID_TILE (grid-parallelized
# partial) stays a grid row and DECLINED (no static extent) falls back to the default, so neither
# is sized as a reduction.
SIZED_REDUCTION_CATEGORIES = frozenset(
    {
        ReductionCategory.FULL_SLICE,
        ReductionCategory.FULL_GRID,
        ReductionCategory.USER_TILE,
    }
)
# Categories that occupy the full reduction extent within one program.
FULL_EXTENT_CATEGORIES = frozenset(
    {ReductionCategory.FULL_SLICE, ReductionCategory.FULL_GRID}
)


class ReductionDescriptor(NamedTuple):
    """One reduction OCCURRENCE: a (``graph_id``, ``block_id``) reduction on the ORIGINAL
    (pre-roll) device graphs. Stage 1 emits a list of these; the Stage-2 allocator consumes them.

    A reduction axis may occur in more than one original graph (e.g. a kernel that reduces the
    same axis in two separate passes) — each occurrence is its own descriptor, so sequential
    passes over one axis are NOT collapsed.

    Descriptors with the same ``graph_id`` are co-resident (the compiler fused them into one graph
    -> shared resident working set). ``graph_id`` is read off the ORIGINAL graphs only (rolled
    ``ReductionLoopGraphInfo`` subgraphs excluded), so it is invariant to the autotuner flipping a
    ``reduction_loops`` knob.

    Fields:
    - ``category``: the :class:`ReductionCategory`.
    - ``block_id`` / ``graph_id``: the reduction axis + its original-graph co-residency key.
    - ``size_hint`` / ``itemsize`` / ``input_load_itemsize``: extent (element count), the
      fp32-promoted accumulator itemsize, and the HBM-load element width feeding it.
    - ``carried_2d_count``: the NUMBER of >=2-D ``[M_BLOCK, R_BLOCK]`` loop-carried accumulators
      whose last dim is this rdim (e.g. kl_div=1, jsd=2); those tiles stay resident the whole loop.
      A count (not a bool) because the carried byte cap divides the budget by it. 0 = none here.
    - ``row_reread`` / ``reread_eviction_index`` / ``num_load``: per-reduction memory-op signals.
    """

    category: ReductionCategory
    block_id: int
    graph_id: int
    size_hint: int
    itemsize: int
    input_load_itemsize: int = 0
    carried_2d_count: int = 0
    row_reread: bool = False
    reread_eviction_index: int | None = None
    num_load: int = 0


class CoResidencyGroup(NamedTuple):
    """A ``graph_id`` equivalence class of reductions whose working tiles are live at the same
    time, so ONE budget must fit them all. ``descriptor_indices`` indexes into
    ``ReductionKernelFact.reductions``.

    ``live_tiles`` is the group's resident tile set — one ``dim_block_ids`` tuple per
    register-resident tile (the block id each dim spans, ``None`` for a static/broadcast dim). It
    is the peak live set of the group's home graph, combined with the for-loop bodies the group
    drives and its If/Else branch siblings (see device_ir ``_group_live_tiles``). Each loop-carried
    accumulator is captured inline at its real shape, so the Stage-2 footprint can sum ``∏(dim
    widths)`` per actual tile. Empty when the fact is built without a live env (a bare-spec test).
    """

    graph_id: int
    descriptor_indices: tuple[int, ...]
    live_tiles: tuple[tuple[int | None, ...], ...] = ()


class ReductionKernelFact(NamedTuple):
    """The per-kernel Stage-1 product that Stage 2 consumes: the list of reduction descriptors,
    their co-residency groups (``graph_id`` classes), the non-reduction user-tiled loops (sized as
    a separate pass), and the parallel grid axes (rows with no reduction over them).

    Built by ``build_reduction_kernel_fact``. ``reductions`` may be empty (a kernel with only
    GRID_TILE / DECLINED reductions, or none) — the seed then declines.
    """

    reductions: tuple[ReductionDescriptor, ...]
    coresidency_groups: tuple[CoResidencyGroup, ...]
    non_reduction_loop_block_ids: tuple[int, ...] = ()
    grid_axis_block_ids: tuple[int, ...] = ()


class MatmulWithReductionEpilogueFact(NamedTuple):
    """A fused matmul + reduction-over-output-axis epilogue, recorded when a ``MatmulFact`` and
    a register-resident epilogue reduction co-occur in one kernel (e.g.
    ``matmul_rms_norm``: ``acc = x @ y`` then a reduction over N on the carried ``[M_BLOCK,
    N]`` accumulator, then write-back). A COMPOSED fact: it holds the matmul fact plus the few
    derived fields the seed keys on. ``TritonMatmulReductionEpilogueHeuristic`` branches on it.

    - ``matmul``: the composed matmul sub-fact.
    - ``n_extent``: the specialized output width N (= the epilogue reduction's ``size_hint``); N
      is ``hl.specialize``'d (never tiled), so both the ``[M_BLOCK, N]`` accumulator and the
      ``[K_BLOCK, N]`` operand tile scale with N — the resident-footprint signal the
      footprint-aware tile chooser keys on.
    - ``m_block_id`` / ``k_block_id``: the grid M tile and the K tile the seed sizes
      (there is no ``n_block_id`` — N is specialized, not a block_size).
    """

    matmul: MatmulFact
    n_extent: int
    m_block_id: int | None
    k_block_id: int | None


class MemoryOpFact(NamedTuple):
    """Metadata linking one ``Config.indexing`` slot to its graph memory op, one entry per
    load/store in graph-traversal order (so ``memory_op_facts[i]`` describes ``config.indexing[i]``).
    Lets heuristics reason about *which* load/store a slot is, not a bare positional index.

    The reduction-fact builders consume the enrichment fields below (all reduction-AGNOSTIC — no
    notion of which axis is "the" reduction; the builders index them by the reduction's ``block_id``).
    """

    indexing_index: int  # slot in Config.indexing (== position in this list)
    kind: str  # "load" | "store"
    eviction_index: int | None  # slot in Config.load_eviction_policies, else None
    tensor_name: str | None  # host buffer name being accessed, e.g. "x", "weight"
    dtype: torch.dtype | None  # element dtype of the accessed tensor
    ndim: int  # rank of the accessed tensor
    num_reuses: int  # downstream FX consumers of the load (0 for stores)
    matmul_operand: str | None  # matmul/dot operand: "lhs" | "rhs" | None
    # --- reduction-fact enrichment (reduction-agnostic; () / False / None for stores) ---
    # device_ir.graphs index this op lives in (scopes per-graph counts).
    graph_id: int = -1
    # per-axis count of reductions this load FEEDS: ((reduction_axis_block_id, count), ...).
    reductions_fed: tuple[tuple[int, int], ...] = ()
    # stores the load's value reaches WITHOUT passing through a reduction, each keyed by the
    # store's full subscript-axis tuple (store[tile_m, tile_n] -> (id_m, id_n)); the forward
    # walk cuts at both reductions and stores. Empty == value never bypasses a reduction.
    stores_fed: tuple[tuple[int | None, ...], ...] = ()
    # block-id per non-bare-int subscript, from the accessed tensor's SHAPE dims (``None`` where
    # unresolvable). Shape-resolved fallback for the gates below (the plain-slice case,
    # e.g. ``out[tile_m, :]``).
    indexed_block_ids: tuple[int | None, ...] = ()
    # inner-dim extent for a rank>=2 op (a reduction-width signal; gates use subscript_block_ids).
    inner_extent: int | None = None
    # AXIS the op's INDEX subscripts address (block-id per non-bare-int position, from the tile/offset
    # subscript so it is reduction-AGNOSTIC; ``None`` for a plain slice). The faithful axis key for
    # the full_width_output / input_load_itemsize gates.
    subscript_block_ids: tuple[int | None, ...] = ()
    # Element stride of the accessed tensor along each subscript position, aligned 1:1 with
    # ``subscript_block_ids`` (from ``.stride()``). A stride-1 position is the contiguous (coalescing)
    # axis — the last subscript for a row-major tensor, a different one for a transposed/strided view.
    subscript_strides: tuple[int, ...] = ()
    # DISTINCT HBM elements the op's accessed tensor touches: product of its size-hinted shape dims
    # over NON-broadcast dims (``stride != 0``); a stride-0 dim contributes factor 1 (``0`` if no
    # resolvable fake tensor). A FULL-EXTENT op has ``accessed_numel`` == the problem numel; a
    # BROADCAST operand has a STRICTLY SMALLER count — whether it is a small tensor (``bias[N]``,
    # ``[M,1]``, ``[1,N]``) OR a full-SIZE ``.expand()``/``broadcast_tensors`` view with a stride-0
    # dim. The faithful signal for per-element HBM traffic at ANY rank/stride, unlike a bare ``ndim``
    # check (a full-rank ``[M,1]`` broadcast passes ndim) or a shape-only product (a stride-0 expand
    # passes shape).
    accessed_numel: int = 0


class AccumulatorFact(NamedTuple):
    """One loop-carried tensor accumulator in a reduction loop, recorded at compile time.
    Reduction-AGNOSTIC (like ``MemoryOpFact``): ``dim_block_ids`` is the per-dim block-id
    provenance (``None`` for a static dim), ``itemsize`` the element size. Accumulators whose last
    dim is the reduction axis feed ``ReductionDescriptor.carried_2d_count`` (a 1-D ``[M_BLOCK]``
    scalar accumulator counts as 0).
    """

    dim_block_ids: tuple[int | None, ...]
    itemsize: int


class PointwiseElementwiseFact(NamedTuple):
    """Workload facts for a PURE elementwise/pointwise kernel — DEFINED by the absence of any
    reduction/matmul/accumulator fact (the disjointness rule): if one of those fired, the kernel
    belongs to that family and this fact is never built. Bandwidth-bound; the compiler defaults it to
    ``block_size=32`` (which starves HBM), so the seed sizes a saturating tile from these fields
    (derived from the walker ``MemoryOpFact`` list + block-size specs, plus one graph walk).

    - ``total_numel``: product of the tiled block dims' ``size_hint``s (the problem element count,
      M*N); the occupancy / grid-saturation input.
    - ``slab_numel``: the untiled inner slab, in ELEMENTS, that full-extent ops drag per tiled element
      = ``sum(accessed_numel // total_numel)`` over those ops (flat kernel: 1 per op; rope:
      heads*head_dim). A BROADCAST operand (``bias[N]``, ``[M,1]``, stride-0) has
      ``accessed_numel < total_numel`` → amortized → excluded; an OVERSIZED operand still touches the
      full problem → counted (hence ``>=``).
    - ``storage_itemsize`` / ``compute_itemsize``: the STORAGE (HBM) and widest COMPUTE (fp32) byte
      widths that scale slab_numel into the two budgets — ``slab_numel * storage`` = bandwidth traffic,
      ``slab_numel * compute`` = register cap (compute reads the promoted dtype; a memory op knows only
      storage). NOTE: the register cap is a COARSE proxy (blind to compute temporaries), but benign —
      pointwise is memory-bound; its only jobs are relaxing the floor for a heavy slab and capping the
      transpose-conflict tile. (Mixed-dtype storage uses the max width — a minor seed approximation.)
    - ``contig_block_ids``: TILED block-ids that are the stride-1 axis of some full-extent op (from
      ``subscript_strides``, no graph walk). Row-major → the last dim (seed unchanged); transposed →
      a different dim; two+ entries = a load-vs-store CONFLICT → the seed emits a BALANCED tile.
    - ``sfu_ops``: count of transcendental (SFU) ops. SFU ops are latency-bound on a distinct unit, so
      a transcendental-heavy tile wants more warps while an all-FMA tile of the same op count does not
      — so SFU count (not total op count) drives the num_warps ramp.
    """

    total_numel: int
    slab_numel: int
    storage_itemsize: int
    compute_itemsize: int
    contig_block_ids: tuple[int, ...] = ()
    sfu_ops: int = 0


def shrink_block_sizes_for_numel_constraints(
    constraints: list[TensorNumelConstraint],
    block_sizes: list[int],
    min_sizes: list[int],
) -> None:
    """Shrink *block_sizes* in-place so every *constraint* is satisfied.

    Halves the largest involved block size first for balanced tiles.
    Fixed-point loop handles cross-constraint interactions.
    """
    prev = list(block_sizes)
    while True:
        for constraint in constraints:
            while not constraint.check_fn(
                *(block_sizes[i] for i in constraint.block_indices)
            ):
                best_idx: int | None = None
                best_val = -1
                for i in constraint.block_indices:
                    can_halve = block_sizes[i] // 2 >= min_sizes[i]
                    if can_halve and block_sizes[i] > best_val:
                        best_val = block_sizes[i]
                        best_idx = i
                if best_idx is None:
                    log.warning(
                        "tensor numel constraint unsatisfiable at minimum "
                        "block sizes: %s",
                        constraint.expr_str,
                    )
                    break
                block_sizes[best_idx] //= 2
        if block_sizes == prev:
            break
        prev = list(block_sizes)


DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 1

# Upper bound (power of two) that a matmul tile dimension's block size may reach
# even when the dimension itself is smaller. Applied only to dimensions that
# feed an hl.dot (see enforce_dot_requirements), so the autotuner can mask-
# overshoot a small matmul dimension up to a hardware-friendly tile (e.g. an M
# tile matching the native MMA shape). We restrict this to matmuls because such
# kernels are memory/MMA-bound on the small dimension -- the masked-off rows/cols
# are effectively free and the larger, hardware-aligned tile runs faster -- while
# for elementwise/reduction kernels a larger-than-dimension tile is pure waste.
SMALL_DIM_BLOCK_SIZE_OVERSHOOT = 64

# Base backend tunable keys (public)
_BASE_BACKEND_TUNABLE_KEYS: frozenset[str] = frozenset(
    {
        "waves_per_eu",
        "matrix_instr_nonkdim",
        "num_ctas",
        "occupancy",
        "pallas_worklist_grouping",
        "pallas_loop_type",
        "pallas_pre_broadcast",
        *CUTE_TCGEN05_TUNABLE_KEYS,
    }
)
_BACKEND_DIAGNOSTIC_CONFIG_KEYS = CUTE_TCGEN05_DIAGNOSTIC_CONFIG_KEYS


def _get_backend_tunable_keys() -> frozenset[str]:
    """Get all backend tunable keys, including FB-private ones if available."""
    try:
        from ..fb.mtia_tunables import MTIA_TUNABLES  # pyrefly: ignore [missing-import]

        return _BASE_BACKEND_TUNABLE_KEYS | frozenset(MTIA_TUNABLES)
    except ImportError:
        return _BASE_BACKEND_TUNABLE_KEYS


BACKEND_TUNABLE_KEYS: frozenset[str] = _get_backend_tunable_keys()
_BACKEND_STRATEGY_CONFIG_KEYS = CUTE_TCGEN05_STRATEGY_CONFIG_KEYS
# All config keys whose support depends on the backend.  The base Backend
# class rejects these by default; each backend subclass opts in selectively.
BACKEND_SPECIFIC_KEYS: frozenset[str] = (
    BACKEND_TUNABLE_KEYS
    | _BACKEND_DIAGNOSTIC_CONFIG_KEYS
    | _BACKEND_STRATEGY_CONFIG_KEYS
    | frozenset(FLASH_CONFIG_KEYS)
    | {
        "num_threads",
        "cute_vector_widths",
        "cute_threads_per_row",
        # ⚠ ACCEPTED BUT NOT A KNOB (task 1): an input spelling with no spec sequence and
        # no search slot, translated into ``cute_row_residency`` and stripped by
        # ``_normalize_cute_row_residency``.  Listed here so an old config migrates
        # silently instead of raising ``Unsupported config keys``.
        "cute_reduction_reload",
        # ⭐ The ONE three-way row-residency axis (registers / smem / gmem) that
        # SUPERSEDED the ``cute_reduction_reload`` + ``cute_tv_sweep_cache``
        # conjunction as the SELECTOR.  cute-only for the same reason as the four
        # above: it names CuTe emission arms.
        "cute_row_residency",
        # Bounds the ``smem`` arm of ``cute_row_residency``: the whole-kernel SMEM budget
        # its staging tiles may occupy.  cute-only for the same reason as the four above.
        # The two AST-pass knobs.  cute-only, so they MUST be listed here: the
        # base ``Backend.supports_config_key`` rejects every member of this set,
        # which is what makes ``ConfigSpec._normalize`` raise on a config that
        # names them for triton/pallas/metal instead of silently ignoring them.
        "cute_ndtile_tv",
        "cute_online_defer",
        "load_cache_modifiers",
        "store_cache_modifiers",
        "pallas_loop_type",
        "pallas_load_buffer_count",
        "pallas_pre_broadcast",
        "xcd_remap",
    }
)
VALID_KEYS: frozenset[str] = frozenset(
    [
        "block_sizes",
        "num_threads",
        "loop_orders",
        "l2_groupings",
        "reduction_loops",
        "flatten_loops",
        "range_unroll_factors",
        "range_warp_specializes",
        "range_num_stages",
        "range_multi_buffers",
        "range_flattens",
        "static_ranges",
        "num_warps",
        "num_stages",
        "pid_type",
        "num_sm_multiplier",
        "maxnreg",
        "indexing",
        "atomic_indexing",
        "load_eviction_policies",
        "load_cache_modifiers",
        "store_cache_modifiers",
        "pallas_loop_type",
        "pallas_load_buffer_count",
        "pallas_pre_broadcast",
        "cute_vector_widths",
        "cute_threads_per_row",
        "cute_reduction_reload",
        "cute_row_residency",
        "cute_ndtile_tv",
        "cute_online_defer",
        *BACKEND_TUNABLE_KEYS,
        "advanced_controls_file",
        "epilogue_subtile",
        "xcd_remap",
        *_BACKEND_DIAGNOSTIC_CONFIG_KEYS,
        *_BACKEND_STRATEGY_CONFIG_KEYS,
        *FLASH_CONFIG_KEYS,
    ]
)
# Loop types the autotuner searches by default for every Pallas inner loop.
AUTOTUNED_PALLAS_LOOP_TYPES = ("emit_pipeline", "unroll", "fori_loop")
VALID_PALLAS_LOOP_TYPES = AUTOTUNED_PALLAS_LOOP_TYPES
VALID_PALLAS_WORKLIST_GROUPINGS = (0, 1, 2)
VALID_PID_TYPES = (
    "flat",
    "xyz",
    "persistent_blocked",
    "persistent_interleaved",
)
MIN_NUM_SM_MULTIPLIER = 1
MAX_NUM_SM_MULTIPLIER = 128
DEFAULT_NUM_SM_MULTIPLIER = 1
EPILOGUE_SUBTILE_EXTENDED_CHOICES = (None, 2, 4)
EPILOGUE_SUBTILE_DEFAULT_CHOICES = (None, 2)
EPILOGUE_SUBTILE_MIN_K_HINT = 1024
EPILOGUE_SUBTILE_MIN_K_HINT_EXTENDED = 16384
# maxnreg values: None means no limit, otherwise limit to this many registers per thread
# Lower values allow higher occupancy but may hurt performance for register-heavy kernels
VALID_MAXNREG = (None, 32, 64, 128, 256)
DEFAULT_MAXNREG = None
_CUTE_IMPLICIT_DEFAULT_KEYS: frozenset[str] = frozenset(
    {
        "loop_orders",
        "flatten_loops",
        "l2_groupings",
        "range_unroll_factors",
        "range_warp_specializes",
        "range_num_stages",
        "range_multi_buffers",
        "range_flattens",
        "static_ranges",
        "load_eviction_policies",
        "indexing",
        "atomic_indexing",
        "num_warps",
        "num_stages",
        "pid_type",
        "num_sm_multiplier",
        "maxnreg",
    }
)


# For tileir backend or AMD ROCM, eviction policies are not supported.
# Keep this uncached: some tests patch the AMD capability helper, and caching
# only on backend name can poison later Triton ConfigSpec construction inside
# the same worker process.
def get_valid_eviction_policies(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and not supports_amd_cdna_tunables():
        return ("", "first", "last")
    return ("",)


def get_valid_load_cache_modifiers(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and supports_amd_cdna_tunables():
        return ("", ".cg")
    return ("",)


def get_valid_store_cache_modifiers(backend_name: str) -> tuple[str, ...]:
    if backend_name == "triton" and supports_amd_cdna_tunables():
        return ("", ".cs", ".wt")
    return ("",)


def _config_fingerprint(config: Mapping[str, object]) -> str:
    """A content key for a config dict, stable across copies of it.

    Used only by the ``cute_row_residency`` provenance sets (see
    ``ConfigSpec._record_cute_row_residency_provenance``).  ⚠ CONTENT AND NOT ``id()``,
    because ``BoundKernel.to_code`` deliberately COPIES the config before normalizing it,
    so the object codegen sees is never the object that was recorded.

    ⚠ NOT ``hash()``: ``Config.__hash__`` exists but this is handed plain dicts as well,
    and a repr-based key keeps the sets debuggable when a provenance answer is surprising.
    Sorted so key order cannot change the answer.
    """
    return repr(sorted((k, repr(v)) for k, v in config.items()))


class ConfigSpec:
    def __init__(
        self,
        *,
        backend: Backend,
        user_defined_tunables: Mapping[str, ConfigSpecFragment] | None = None,
        target_device_capability: tuple[int, int]
        | object
        | None = _TARGET_DEVICE_CAPABILITY_UNSET,
        device: torch.device | None = None,
        num_sm: int | None = None,
    ) -> None:
        self.backend = backend
        self.backend_name = backend.name
        self.max_reduction_threads = backend.max_reduction_threads()
        self.max_reduction_loop = backend.max_reduction_loop()
        self.reduction_loop_force_threshold = self.max_reduction_threads
        self.cute_indexed_reduction_block_ids: set[int] = set()
        self.user_defined_tunables = (
            {} if user_defined_tunables is None else dict(user_defined_tunables)
        )
        # Bound kernels pass an explicit target capability. Direct CuTe specs
        # use the current CUDA device so validation still enforces arch gates.
        if target_device_capability is _TARGET_DEVICE_CAPABILITY_UNSET:
            self.target_device_capability: tuple[int, int] | None = (
                get_target_device_capability() if self.backend_name == "cute" else None
            )
        else:
            self.target_device_capability = cast(
                "tuple[int, int] | None",
                target_device_capability,
            )

        # XCD count for the *compile* device, captured once so xcd_remap's
        # support/search/normalize decisions match the device used in codegen
        # (rather than the current device).  1 disables/no-ops xcd_remap.
        self.num_xcd: int = get_num_xcd(device)
        # Persistent grid SM/CU count of the compile device (after reserved_sms);
        # used to check XCD-alignment of the persistent_interleaved grid stride.
        # Defaults to the device CU count (consistent with num_xcd) when not
        # passed explicitly by the compile path.
        self.num_sm: int = num_sm if num_sm is not None else device_num_sm(device)

        self.block_sizes: BlockIdSequence[BlockSizeSpec] = BlockIdSequence()
        self.num_threads: BlockIdSequence[NumThreadsSpec] = BlockIdSequence()
        self.loop_orders: BlockIdSequence[LoopOrderSpec] = BlockIdSequence()
        self.l2_groupings: BlockIdSequence[L2GroupingSpec] = BlockIdSequence()
        self.flatten_loops: BlockIdSequence[FlattenLoopSpec] = BlockIdSequence()
        self.reduction_loops: BlockIdSequence[ReductionLoopSpec] = BlockIdSequence()
        self.cute_vector_widths: BlockIdSequence[CuteVectorWidthSpec] = (
            BlockIdSequence()
        )
        # The TV-layout knobs are registered together (one slot per reduction
        # block) so a reduction's layout is fully described by block size,
        # threads per row, vector width, and row residency.
        self.cute_threads_per_row: BlockIdSequence[CuteThreadsPerRowSpec] = (
            BlockIdSequence()
        )
        # ⭐ The ONE three-way residency axis (registers / smem / gmem) -- and after
        # task 1 it is the ONLY key carrying that decision.  ``cute_reduction_reload``
        # used to be declared right here on the same domain, which is what let one
        # decision wear two keys; it is now an accepted INPUT spelling only, translated
        # and stripped by ``_normalize_cute_row_residency``.  It is deliberately NOT
        # grouped with
        # ``cute_tv_sweep_cache`` below: that key's per-device-loop domain is right
        # for a pass's register budget and wrong for a row's residency, and having
        # one decision live on two domains is what this axis removes.
        self.cute_row_residency: BlockIdSequence[CuteRowResidencySpec] = (
            BlockIdSequence()
        )
        # The two AST-pass knobs, one slot per DEVICE LOOP (see
        # ``CuteOnlineDeferSpec`` for why that domain differs from the three
        # TV-layout knobs above, which are per reduction block).  Registered from
        # ``lower_to_device_ir`` rather than from ``register_rollable_reductions``,
        # because the latter early-returns when a kernel has no reduction block --
        # and ``cross_entropy_online``, the kernel ``cute_online_defer`` controls,
        # is exactly that kernel.
        self.cute_ndtile_tv: BlockIdSequence[CuteNDTileTvSpec] = BlockIdSequence()
        self.cute_online_defer: BlockIdSequence[CuteOnlineDeferSpec] = BlockIdSequence()
        # ⭐ USER-PROVENANCE for ``cute_row_residency``, as two sets of config
        # FINGERPRINTS (see ``_config_fingerprint`` for why content and not ``id()``):
        #
        #   _cute_row_residency_explicit     the caller WROTE the key in a config handed
        #                                    to ``normalize`` (written there, the only
        #                                    place the answer is still knowable).
        #                                    ⚠ Keyed by ``id()`` of the config DICT, and
        #                                    content would be WRONG -- an explicit request,
        #                                    a ladder fill and the legacy spelling all
        #                                    normalize to the SAME bytes, which is exactly
        #                                    the collision this carrier exists to see
        #                                    through.  Sound because ``to_code`` normalizes
        #                                    the very copy it then hands to codegen.
        #   _cute_row_residency_synthesised  the config was MINTED by ``flat_config`` --
        #                                    a default or an autotuner draw -- so its keys
        #                                    were never typed by a human.  ⚠ Keyed by
        #                                    CONTENT, because this one MUST survive
        #                                    ``to_code``'s defensive copy.  Wins over the
        #                                    set above; ``cute_row_residency_is_explicit``
        #                                    documents why the order matters.
        #
        # ⚠ NEITHER IS the question ``TileStrategy._cute_row_residency_requested_by_block``
        # answers.  That dict records "a strategy resolved a residency for this block", off
        # the POST-normalize config, so it cannot see the distinction these sets exist for
        # -- the measurement is in ``_record_cute_row_residency_provenance``'s docstring.
        self._cute_row_residency_explicit: set[int] = set()
        self._cute_row_residency_synthesised: set[str] = set()
        self.range_unroll_factors: BlockIdSequence[RangeUnrollFactorSpec] = (
            BlockIdSequence()
        )
        self.range_warp_specialize: BlockIdSequence[RangeWarpSpecializeSpec] = (
            BlockIdSequence()
        )
        self.range_num_stages: BlockIdSequence[RangeNumStagesSpec] = BlockIdSequence()
        self.range_multi_buffers: BlockIdSequence[RangeMultiBufferSpec] = (
            BlockIdSequence()
        )
        self.range_flattens: BlockIdSequence[RangeFlattenSpec] = BlockIdSequence()
        self.static_ranges: BlockIdSequence[StaticRangeSpec] = BlockIdSequence()

        self.allowed_pid_types: tuple[PidTypeLiteral, ...] = tuple(VALID_PID_TYPES)
        self.max_num_sm_multiplier: int = MAX_NUM_SM_MULTIPLIER
        self.grid_block_ids: list[int] = []
        self.tensor_numel_constraints: list[TensorNumelConstraint] = []
        self.load_eviction_policies = ListOf(
            EnumFragment(choices=get_valid_eviction_policies(self.backend_name)),
            length=0,
        )
        self.load_cache_modifiers = ListOf(
            EnumFragment(choices=get_valid_load_cache_modifiers(self.backend_name)),
            length=0,
        )
        self.store_cache_modifiers = ListOf(
            EnumFragment(choices=get_valid_store_cache_modifiers(self.backend_name)),
            length=0,
        )
        self.indexing = ListOf(
            EnumFragment(choices=self.valid_indexing_types()),
            length=0,
        )
        self.atomic_indexing = ListOf(
            EnumFragment(choices=self.valid_atomic_indexing_types()),
            length=0,
        )
        self.pallas_load_buffer_count = ListOf(
            IntegerFragment(1, 2, 1),
            length=0,
        )
        self.epilogue_subtile_candidate_enabled: bool = False
        self.epilogue_subtile_autotune_choices: tuple[int | None, ...] | None = None
        self.epilogue_subtile_k_hint: int = 0
        self.has_pallas_inner_loops: bool = False
        self.has_symbolic_or_data_dependent_bounds: bool = False
        self._cute_tcgen05_config = CuteTcgen05Config(self)
        # CuTe flash-attention autotune surface gating (Tasks #25 + #28).
        # Default False so the flash knobs never appear in the search surface
        # and behavior is byte-identical to the env-only path. Set True when the
        # flash detector fires (see ``lower_to_device_ir``). The shape needed to
        # build the fragments (head_dim / num_kv) is captured at the same time.
        self.cute_flash_search_enabled: bool = False
        self._cute_flash_head_dim: int | None = None
        self._cute_flash_num_kv: int | None = None
        self._cute_flash_dtype: torch.dtype = torch.float16
        self._cute_flash_is_causal: bool = False
        self._cute_flash_has_kv_tile_pruning: bool = False
        self._cute_flash_requires_ws_overlap: bool = False
        self._cute_flash_small_biased_candidate: bool = False
        self._cute_flash_block_size_targets: dict[int, int] = {}
        self.compiler_default_config: helion.Config | None = None
        self.compiler_seed_configs: list[helion.Config] = []
        self.autotuner_heuristics: list[str] = []
        self.matmul_facts: list[MatmulFact] = []
        # The Stage-1 categorizing product the reduction seed + allocator consume.
        self.reduction_kernel_fact: ReductionKernelFact | None = None
        self.matmul_reduction_epilogue_facts: list[MatmulWithReductionEpilogueFact] = []
        self.accumulator_facts: list[AccumulatorFact] = []
        self.pointwise_facts: list[PointwiseElementwiseFact] = []
        self.store_indices: list[int] = []
        self.memory_op_facts: list[MemoryOpFact] = []
        self.backend_tunable_fragments = self.backend.tunable_fragments()
        unknown_tunables = set(self.backend_tunable_fragments) - BACKEND_TUNABLE_KEYS
        if unknown_tunables:
            raise RuntimeError(
                f"Backend {self.backend_name!r} returned unknown tunables: {sorted(unknown_tunables)!r}"
            )

    def _should_keep_epilogue_subtile_for_autotune(self) -> bool:
        if self.epilogue_subtile_autotune_choices is None:
            return False
        return supports_tensor_descriptor()

    def fix_epilogue_subtile_store_indexing(self, config: dict[str, object]) -> None:
        """Force subtiled store indexing to tensor_descriptor for correctness."""
        if (
            not self.epilogue_subtile_candidate_enabled
            or "epilogue_subtile" not in config
        ):
            return
        indexing = config.get("indexing")
        if isinstance(indexing, list):
            for i in self.store_indices:
                indexing[i] = "tensor_descriptor"

    @staticmethod
    def _infer_epilogue_subtile_k_hint(args: Sequence[object]) -> int:
        def _as_concrete_dim(dim: object) -> int | None:
            return dim if type(dim) is int else None

        tensor_args = [
            arg for arg in args if isinstance(arg, torch.Tensor) and arg.ndim >= 2
        ]
        best = 0
        for lhs, rhs in itertools.combinations(tensor_args, 2):
            candidates: list[int] = []
            lhs_last = _as_concrete_dim(lhs.shape[-1])
            lhs_prev = _as_concrete_dim(lhs.shape[-2])
            rhs_last = _as_concrete_dim(rhs.shape[-1])
            rhs_prev = _as_concrete_dim(rhs.shape[-2])
            if lhs_last is not None and rhs_prev is not None and lhs_last == rhs_prev:
                candidates.append(lhs_last)
            if lhs_prev is not None and rhs_last is not None and lhs_prev == rhs_last:
                candidates.append(lhs_prev)
            if candidates:
                best = max(best, *candidates)
        return best

    def configure_epilogue_subtile_autotune(self, args: Sequence[object]) -> None:
        self.epilogue_subtile_k_hint = self._infer_epilogue_subtile_k_hint(args)
        arch = self.target_device_capability
        if arch is None:
            self.epilogue_subtile_autotune_choices = None
            return

        if arch >= (10, 0):
            arch_enabled = (
                self.epilogue_subtile_candidate_enabled and supports_tensor_descriptor()
            )
        else:
            arch_enabled = False

        enabled = (
            arch_enabled and self.epilogue_subtile_k_hint >= EPILOGUE_SUBTILE_MIN_K_HINT
        )
        if not enabled:
            self.epilogue_subtile_autotune_choices = None
        elif (
            arch >= (10, 0)
            and self.epilogue_subtile_k_hint >= EPILOGUE_SUBTILE_MIN_K_HINT_EXTENDED
        ):
            self.epilogue_subtile_autotune_choices = EPILOGUE_SUBTILE_EXTENDED_CHOICES
        else:
            self.epilogue_subtile_autotune_choices = EPILOGUE_SUBTILE_DEFAULT_CHOICES

    def valid_indexing_types(self) -> tuple[IndexingLiteral, ...]:
        if supports_tensor_descriptor():
            return ("pointer", "tensor_descriptor")
        if not self.backend.supports_block_ptr_indexing():
            return ("pointer",)
        return ("pointer", "block_ptr")

    def valid_atomic_indexing_types(self) -> tuple[IndexingLiteral, ...]:
        """Atomic ops only support pointer and tensor_descriptor (no block_ptr)."""
        if supports_tensor_descriptor():
            return ("pointer", "tensor_descriptor")
        return ("pointer",)

    def _remove_duplicates(self) -> None:
        self.num_threads._remove_duplicates()
        self.loop_orders._remove_duplicates()
        self.l2_groupings._remove_duplicates()
        self.flatten_loops._remove_duplicates()
        self.range_unroll_factors._remove_duplicates()
        self.range_warp_specialize._remove_duplicates()
        self.range_num_stages._remove_duplicates()
        self.range_multi_buffers._remove_duplicates()
        self.range_flattens._remove_duplicates()
        self.static_ranges._remove_duplicates()

    def disallow_pid_type(self, pid_type: PidTypeLiteral) -> None:
        """Disallow a pid_type from being used in the config."""

        self.allowed_pid_types = tuple(
            [x for x in self.allowed_pid_types if x != pid_type]
        )
        assert self.allowed_pid_types

    @property
    def cute_tcgen05_search_enabled(self) -> bool:
        return self._cute_tcgen05_config.search_enabled

    @cute_tcgen05_search_enabled.setter
    def cute_tcgen05_search_enabled(self, value: bool) -> None:
        self._cute_tcgen05_config.search_enabled = value

    @property
    def cute_tcgen05_aux_kernel_detected(self) -> bool:
        return self._cute_tcgen05_config.aux_kernel_detected

    @cute_tcgen05_aux_kernel_detected.setter
    def cute_tcgen05_aux_kernel_detected(self, value: bool) -> None:
        self._cute_tcgen05_config.aux_kernel_detected = value

    @property
    def cute_tcgen05_exact_shape_aux_kernel_detected(self) -> bool:
        return self._cute_tcgen05_config.exact_shape_aux_kernel_detected

    @cute_tcgen05_exact_shape_aux_kernel_detected.setter
    def cute_tcgen05_exact_shape_aux_kernel_detected(self, value: bool) -> None:
        self._cute_tcgen05_config.exact_shape_aux_kernel_detected = value

    @property
    def cute_tcgen05_matmul_has_non_tcgen05_operand(self) -> bool:
        return self._cute_tcgen05_config.matmul_has_non_tcgen05_operand

    @cute_tcgen05_matmul_has_non_tcgen05_operand.setter
    def cute_tcgen05_matmul_has_non_tcgen05_operand(self, value: bool) -> None:
        self._cute_tcgen05_config.matmul_has_non_tcgen05_operand = value

    def _normalize_cute_flash(
        self, config: dict[str, object], *, fix_invalid: bool
    ) -> None:
        """Normalize the flash-attention knobs (Tasks #25 + #28).

        Only runs when ``cute_flash_search_enabled`` is set (the flash detector
        fired). Mirrors ``CuteTcgen05Config.normalize_strategy``: each key in
        ``FLASH_CONFIG_KEYS`` is validated against its fragment's choices and
        defaulted (to the fragment default = env/shape-resolved current value)
        when absent. When the flag is off this is a no-op so configs never grow
        the flash keys and behavior is byte-identical to today.
        """
        if not self.cute_flash_search_enabled:
            return
        assert self._cute_flash_head_dim is not None
        assert self._cute_flash_num_kv is not None
        from .._compiler.cute.cute_flash import flash_autotune_fragments

        block_size_targets = self._cute_flash_block_size_target_list()
        if fix_invalid:
            config["block_sizes"] = list(block_size_targets)
            config["pid_type"] = "flat"
            self._normalize_cute_flash_default_sequence(config, "l2_groupings", 1)
            self._normalize_cute_flash_default_sequence(config, "num_threads", 0)
            self._normalize_cute_flash_default_sequence(config, "cute_vector_widths", 1)
            self._normalize_cute_flash_default_loop_orders(config)
            config.pop("epilogue_subtile", None)
        elif not self._is_cute_flash_config_envelope(config, block_size_targets):
            return

        if self._cute_flash_requires_ws_overlap:
            config[FLASH_TOPOLOGY_KEY] = "ws_overlap"
            topology_override = "ws_overlap"
        else:
            valid_manual_topologies = {"fa4", "ws_overlap"}
            topology_value = config.get(FLASH_TOPOLOGY_KEY)
            topology_override = (
                topology_value if topology_value in valid_manual_topologies else None
            )
        fragments = flash_autotune_fragments(
            self._cute_flash_head_dim,
            self._cute_flash_num_kv,
            dtype=self._cute_flash_dtype,
            is_causal=self._cute_flash_is_causal,
            has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
            requires_ws_overlap=self._cute_flash_requires_ws_overlap,
            small_biased_candidate=self._cute_flash_small_biased_candidate,
            topology_override=cast("str | None", topology_override),
        )
        e2e_offset_was_present = FLASH_E2E_OFFSET_KEY in config
        e2e_offset0_was_present = FLASH_E2E_OFFSET0_KEY in config
        e2e_offset_keys = (FLASH_E2E_OFFSET_KEY, FLASH_E2E_OFFSET0_KEY)
        for key, fragment in fragments.items():
            choices = cast("EnumFragment", fragment).choices
            if key in config:
                if config[key] not in choices:
                    if key in e2e_offset_keys:
                        # Legacy explicit e2e frequency overrides can make offsets
                        # outside the autotune fragment valid. Validate the effective
                        # cadence after the e2e keys have been normalized below.
                        pass
                    elif fix_invalid:
                        config[key] = fragment.default()
                    else:
                        raise InvalidConfig(
                            f"{key} must be one of {list(choices)!r}, "
                            f"got {config[key]!r}"
                        )
            else:
                if key not in e2e_offset_keys:
                    config[key] = fragment.default()
        effective_topology = cast("str", config[FLASH_TOPOLOGY_KEY])
        if effective_topology == "fa4" and self._cute_flash_num_kv % 2 != 0:
            effective_topology = "ws_overlap"
        if fix_invalid:
            config[FLASH_TOPOLOGY_KEY] = effective_topology
        if effective_topology != "fa4":
            config[FLASH_ROLE_MAP_KEY] = "helion"
            config[FLASH_EPI_TMA_KEY] = False
            config[FLASH_MASKED_E2E_SCHEDULE_KEY] = "inherit"
            config[FLASH_CAUSAL_KV_ORDER_KEY] = "ascending"
            config[FLASH_CAUSAL_LOOP_SPLIT_KEY] = False
        causal_kv_order = config.get(FLASH_CAUSAL_KV_ORDER_KEY)
        if not self._cute_flash_is_causal or causal_kv_order != "descending":
            config[FLASH_CAUSAL_LOOP_SPLIT_KEY] = False
        e2e_schedule_default = (
            "8/2"
            if (
                effective_topology == "fa4"
                and self._cute_flash_is_causal
                and self._cute_flash_head_dim == 64
                and _flash_causal_hd64_seed_num_kv_supported(self._cute_flash_num_kv)
            )
            else _flash_e2e_schedule_default(
                effective_topology, self._cute_flash_head_dim
            )
        )
        exp2_impl, e2e_freq, e2e_res = _flash_parse_e2e_schedule(
            str(config[FLASH_E2E_SCHEDULE_KEY]), e2e_schedule_default
        )
        if FLASH_EXP2_IMPL_KEY in config:
            exp2_impl = str(config[FLASH_EXP2_IMPL_KEY])
        if FLASH_E2E_FREQ_KEY in config:
            e2e_freq = cast("int", config[FLASH_E2E_FREQ_KEY])
        if FLASH_E2E_RES_KEY in config:
            e2e_res = cast("int", config[FLASH_E2E_RES_KEY])
        _impl, e2e_freq, e2e_res, _schedule = _flash_normalize_e2e_params(
            exp2_impl,
            e2e_freq,
            e2e_res,
            e2e_schedule_default,
        )
        masked_e2e_schedule = str(config.get(FLASH_MASKED_E2E_SCHEDULE_KEY, "inherit"))
        _masked_schedule, masked_e2e_freq, masked_e2e_res = (
            _flash_masked_e2e_schedule_params(
                masked_e2e_schedule,
                e2e_schedule_default,
                e2e_freq,
                e2e_res,
            )
        )
        if not self._cute_flash_is_causal:
            masked_e2e_freq = e2e_freq
            masked_e2e_res = e2e_res
        e2e_offset_period = _flash_e2e_offset_period(
            e2e_freq,
            e2e_res,
            masked_e2e_freq,
            masked_e2e_res,
        )
        if (
            e2e_offset_period > 0
            and effective_topology == "fa4"
            and self._cute_flash_head_dim == 64
        ):
            if self._cute_flash_is_causal and _flash_causal_hd64_seed_num_kv_supported(
                self._cute_flash_num_kv
            ):
                schedule_default_offset = (
                    _flash_causal_hd64_seed_params(self._cute_flash_num_kv)[0]
                    % e2e_offset_period
                )
            else:
                split_default_freq = e2e_freq if e2e_res > 0 else masked_e2e_freq
                schedule_default_offset = split_default_freq // 8
        else:
            schedule_default_offset = 0
        default_offset = schedule_default_offset
        env_offset = os.environ.get("HELION_CUTE_FLASH_E2E_OFFSET")
        if env_offset is not None:
            default_offset = int(env_offset)
            if e2e_offset_period == 0:
                default_offset = 0
            elif default_offset < 0:
                default_offset = schedule_default_offset
            else:
                default_offset %= e2e_offset_period
        if not e2e_offset_was_present:
            config[FLASH_E2E_OFFSET_KEY] = default_offset
        default_offset0 = (
            _flash_causal_hd64_seed_offset0(self._cute_flash_num_kv)
            if (
                e2e_offset_period > 0
                and effective_topology == "fa4"
                and self._cute_flash_is_causal
                and self._cute_flash_head_dim == 64
                and _flash_causal_hd64_seed_num_kv_supported(self._cute_flash_num_kv)
            )
            else 0
        )
        env_offset0 = os.environ.get("HELION_CUTE_FLASH_E2E_OFFSET0")
        if env_offset0 is not None:
            env_offset0_value = int(env_offset0)
            if e2e_offset_period == 0:
                default_offset0 = 0
            elif env_offset0_value < 0:
                default_offset0 %= e2e_offset_period
            else:
                default_offset0 = env_offset0_value % e2e_offset_period
        if not e2e_offset0_was_present:
            config[FLASH_E2E_OFFSET0_KEY] = default_offset0
        for key, default in (
            (FLASH_E2E_OFFSET_KEY, default_offset),
            (FLASH_E2E_OFFSET0_KEY, default_offset0),
        ):
            e2e_offset_value = config[key]
            if not isinstance(e2e_offset_value, int):
                if fix_invalid:
                    config[key] = default
                    e2e_offset_value = default
                else:
                    raise InvalidConfig(
                        f"{key} must be an integer, got {e2e_offset_value!r}"
                    )
            e2e_offset = e2e_offset_value
            e2e_offset_invalid = (
                e2e_offset != 0
                if e2e_offset_period == 0
                else e2e_offset < 0 or e2e_offset >= e2e_offset_period
            )
            if e2e_offset_invalid:
                if fix_invalid:
                    config[key] = _flash_normalize_e2e_offset(
                        e2e_offset, default, e2e_offset_period
                    )
                else:
                    expected = (
                        [0]
                        if e2e_offset_period == 0
                        else list(range(e2e_offset_period))
                    )
                    raise InvalidConfig(
                        f"{key} must be one of {expected!r} for "
                        f"{FLASH_E2E_SCHEDULE_KEY}={config[FLASH_E2E_SCHEDULE_KEY]!r}, "
                        f"got {e2e_offset!r}"
                    )
        self._normalize_cute_flash_register_budget(
            config,
            fragments,
            effective_topology,
            fix_invalid=fix_invalid,
        )

    def _normalize_cute_flash_register_budget(
        self,
        config: dict[str, object],
        fragments: Mapping[str, ConfigSpecFragment],
        effective_topology: str,
        *,
        fix_invalid: bool,
    ) -> None:
        if effective_topology != "fa4":
            return
        softmax_regs = config.get(FLASH_SOFTMAX_REGS_KEY)
        corr_regs = config.get(FLASH_CORR_REGS_KEY)
        other_regs = config.get(FLASH_OTHER_REGS_KEY)
        if (
            type(softmax_regs) is not int
            or type(corr_regs) is not int
            or type(other_regs) is not int
        ):
            return
        budget = 2 * softmax_regs + corr_regs + other_regs
        if budget <= 512:
            return
        message = (
            "FA4 register budget exceeds 512: "
            f"2 * {FLASH_SOFTMAX_REGS_KEY} ({softmax_regs}) + "
            f"{FLASH_CORR_REGS_KEY} ({corr_regs}) + "
            f"{FLASH_OTHER_REGS_KEY} ({other_regs}) = {budget}"
        )
        if not fix_invalid:
            raise InvalidConfig(message)

        def _int_choices(key: str) -> tuple[int, ...]:
            fragment = cast("EnumFragment", fragments[key])
            return tuple(value for value in fragment.choices if type(value) is int)

        current = (softmax_regs, corr_regs, other_regs)
        best: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
        for softmax_candidate in _int_choices(FLASH_SOFTMAX_REGS_KEY):
            for corr_candidate in _int_choices(FLASH_CORR_REGS_KEY):
                for other_candidate in _int_choices(FLASH_OTHER_REGS_KEY):
                    candidate = (softmax_candidate, corr_candidate, other_candidate)
                    if 2 * softmax_candidate + corr_candidate + other_candidate > 512:
                        continue
                    score = (
                        sum(a != b for a, b in zip(candidate, current, strict=True)),
                        sum(
                            abs(a - b) for a, b in zip(candidate, current, strict=True)
                        ),
                        -softmax_candidate,
                    )
                    if best is None or score < best[0]:
                        best = (score, candidate)
        if best is None:
            raise InvalidConfig(message)
        _score, (softmax_regs, corr_regs, other_regs) = best
        config[FLASH_SOFTMAX_REGS_KEY] = softmax_regs
        config[FLASH_CORR_REGS_KEY] = corr_regs
        config[FLASH_OTHER_REGS_KEY] = other_regs

    def enable_cute_flash_search(
        self,
        *,
        head_dim: int,
        num_kv: int,
        dtype: torch.dtype = torch.float16,
        block_size_targets: Mapping[int, int],
        is_causal: bool = False,
        has_kv_tile_pruning: bool = False,
        requires_ws_overlap: bool = False,
        small_biased_candidate: bool = False,
    ) -> None:
        self.cute_flash_search_enabled = True
        self._cute_flash_head_dim = head_dim
        self._cute_flash_num_kv = num_kv
        self._cute_flash_dtype = dtype
        self._cute_flash_is_causal = is_causal
        self._cute_flash_has_kv_tile_pruning = has_kv_tile_pruning
        self._cute_flash_requires_ws_overlap = requires_ws_overlap
        self._cute_flash_small_biased_candidate = small_biased_candidate
        self._cute_flash_block_size_targets = dict(block_size_targets)
        for block_id, target in block_size_targets.items():
            spec = self.block_sizes.block_id_lookup(block_id)
            spec.autotuner_min = target
            spec.max_size = target

    def _pre_normalize_cute_flash_block_sizes(self, config: dict[str, object]) -> None:
        if not self.cute_flash_search_enabled or "block_sizes" not in config:
            return
        block_size_targets = self._cute_flash_block_size_target_list()
        value = config["block_sizes"]
        raw_block_sizes = [*value] if isinstance(value, (list, tuple)) else [value]
        if raw_block_sizes == block_size_targets:
            return
        config["block_sizes"] = list(block_size_targets)

    def _cute_flash_block_size_target_list(self) -> list[int]:
        targets: list[int | None] = [None] * len(self.block_sizes)
        for block_id, target in self._cute_flash_block_size_targets.items():
            targets[self.block_sizes.block_id_to_index(block_id)] = target
        if any(target is None for target in targets):
            raise InvalidConfig(
                "CuTe flash attention search has incomplete block sizes"
            )
        return [target for target in targets if target is not None]

    def _normalize_cute_flash_default_sequence(
        self,
        config: dict[str, object],
        key: str,
        default: object,
    ) -> None:
        value = config.get(key)
        if not value:
            config.pop(key, None)
            return
        if not isinstance(value, list) or any(item != default for item in value):
            config.pop(key, None)
            return
        config.pop(key, None)

    def _normalize_cute_flash_default_loop_orders(
        self, config: dict[str, object]
    ) -> None:
        value = config.get("loop_orders")
        if not value:
            config.pop("loop_orders", None)
            return
        defaults = [spec._fill_missing() for spec in self.loop_orders]
        if value != defaults:
            config.pop("loop_orders", None)
            return
        config.pop("loop_orders", None)

    def _is_cute_flash_config_envelope(
        self, config: dict[str, object], block_size_targets: list[int]
    ) -> bool:
        if config.get("block_sizes") != block_size_targets:
            return False
        if config.get("pid_type", "flat") != "flat":
            return False
        if "epilogue_subtile" in config:
            return False
        for key, default in (
            ("l2_groupings", 1),
            ("num_threads", 0),
            ("cute_vector_widths", 1),
        ):
            value = config.get(key)
            if value and (
                not isinstance(value, list) or any(item != default for item in value)
            ):
                return False
        loop_orders = config.get("loop_orders")
        if loop_orders:
            defaults = [spec._fill_missing() for spec in self.loop_orders]
            if loop_orders != defaults:
                return False
        return True

    @property
    def _tcgen05_cluster_m_search_choices(self) -> tuple[int, ...] | None:
        return self._cute_tcgen05_config.cluster_m_search_choices

    @_tcgen05_cluster_m_search_choices.setter
    def _tcgen05_cluster_m_search_choices(self, value: tuple[int, ...] | None) -> None:
        self._cute_tcgen05_config.cluster_m_search_choices = value

    @property
    def _tcgen05_cluster_m2_search_constraints(
        self,
    ) -> Tcgen05ClusterM2SearchConstraints | None:
        return self._cute_tcgen05_config.cluster_m2_search_constraints

    @_tcgen05_cluster_m2_search_constraints.setter
    def _tcgen05_cluster_m2_search_constraints(
        self, value: Tcgen05ClusterM2SearchConstraints | None
    ) -> None:
        self._cute_tcgen05_config.cluster_m2_search_constraints = value

    @property
    def _tcgen05_ab_stages_three_search_constraints(
        self,
    ) -> Tcgen05AbStagesThreeSearchConstraints | None:
        return self._cute_tcgen05_config.ab_stages_three_search_constraints

    @_tcgen05_ab_stages_three_search_constraints.setter
    def _tcgen05_ab_stages_three_search_constraints(
        self, value: Tcgen05AbStagesThreeSearchConstraints | None
    ) -> None:
        self._cute_tcgen05_config.ab_stages_three_search_constraints = value

    @property
    def _tcgen05_num_epi_warps_search_choices(self) -> tuple[int, ...] | None:
        return self._cute_tcgen05_config.num_epi_warps_search_choices

    @_tcgen05_num_epi_warps_search_choices.setter
    def _tcgen05_num_epi_warps_search_choices(
        self, value: tuple[int, ...] | None
    ) -> None:
        self._cute_tcgen05_config.num_epi_warps_search_choices = value

    @property
    def _tcgen05_num_epi_warps_validation_choices(self) -> tuple[int, ...] | None:
        return self._cute_tcgen05_config.num_epi_warps_validation_choices

    @_tcgen05_num_epi_warps_validation_choices.setter
    def _tcgen05_num_epi_warps_validation_choices(
        self, value: tuple[int, ...] | None
    ) -> None:
        self._cute_tcgen05_config.num_epi_warps_validation_choices = value

    def _tcgen05_full_tile_direct_entry_seed_eligible(self) -> bool:
        return self._cute_tcgen05_config.full_tile_direct_entry_seed_eligible()

    def _tcgen05_full_tile_direct_entry_seed_bk(self) -> int | None:
        return self._cute_tcgen05_config.full_tile_direct_entry_seed_bk()

    def _tcgen05_full_tile_direct_entry_seed_config(self) -> helion.Config | None:
        return self._cute_tcgen05_config.full_tile_direct_entry_seed_config()

    def register_cute_tcgen05_mma_analysis(
        self,
        *,
        m_block_id: int,
        n_block_id: int,
        k_block_id: int,
        input_dtype: torch.dtype,
        has_leading_passthrough: bool,
        explicit_epi_tile_compatible: bool,
    ) -> None:
        self._cute_tcgen05_config.register_mma_analysis(
            m_block_id=m_block_id,
            n_block_id=n_block_id,
            k_block_id=k_block_id,
            input_dtype=input_dtype,
            has_leading_passthrough=has_leading_passthrough,
            explicit_epi_tile_compatible=explicit_epi_tile_compatible,
        )

    def _tcgen05_matmul_block_fragments(
        self,
    ) -> tuple[BlockSizeFragment, BlockSizeFragment, BlockSizeFragment] | None:
        return self._cute_tcgen05_config._matmul_block_fragments()

    def _tcgen05_matmul_seed_block_sizes(
        self, *, bm: int, bn: int, bk: int
    ) -> list[int] | None:
        return self._cute_tcgen05_config._matmul_seed_block_sizes(
            bm=bm,
            bn=bn,
            bk=bk,
        )

    def restrict_tcgen05_cluster_m_search(self, choices: tuple[int, ...]) -> None:
        self._cute_tcgen05_config.restrict_cluster_m_search(choices)

    def allow_tcgen05_cluster_m2_search(
        self,
        *,
        static_k: int,
        max_k_tiles: int = TCGEN05_TWO_CTA_MAX_K_TILES,
        allow_edge_k_tail_family: bool = False,
    ) -> None:
        self._cute_tcgen05_config.allow_cluster_m2_search(
            static_k=static_k,
            max_k_tiles=max_k_tiles,
            allow_edge_k_tail_family=allow_edge_k_tail_family,
        )

    @staticmethod
    def _tcgen05_cluster_m2_bk_is_valid(
        bk: int, constraints: Tcgen05ClusterM2SearchConstraints
    ) -> bool:
        return CuteTcgen05Config.cluster_m2_bk_is_valid(bk, constraints)

    def _tcgen05_c_input_seed_config(self) -> helion.Config | None:
        return self._cute_tcgen05_config._c_input_seed_config()

    def autotune_seed_configs(self) -> list[helion.Config]:
        seeds = self._cute_tcgen05_config.autotune_seed_configs()
        if self.backend_name == "cute" and self.cute_flash_search_enabled:
            from .._compiler.cute.cute_flash import flash_attention_seed_configs

            assert self._cute_flash_head_dim is not None
            seeds.extend(
                flash_attention_seed_configs(
                    self._cute_flash_head_dim,
                    self._cute_flash_num_kv,
                    dtype=self._cute_flash_dtype,
                    is_causal=self._cute_flash_is_causal,
                    has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
                    requires_ws_overlap=self._cute_flash_requires_ws_overlap,
                    small_biased_candidate=self._cute_flash_small_biased_candidate,
                    block_size_targets=self._cute_flash_block_size_target_list(),
                )
            )
        return seeds

    def _fix_tcgen05_cluster_m2_search_config(self, config: dict[str, object]) -> None:
        self._cute_tcgen05_config._fix_cluster_m2_search_config(config)

    def allow_tcgen05_ab_stages_three_search(
        self,
        *,
        dtype_bytes: int,
        device: torch.device,
    ) -> None:
        self._cute_tcgen05_config.allow_ab_stages_three_search(
            dtype_bytes=dtype_bytes,
            device=device,
        )

    @staticmethod
    def _cute_per_cta_ab_smem_budget_bytes(device: torch.device) -> int:
        return CuteTcgen05Config.per_cta_ab_smem_budget_bytes(device)

    def _tcgen05_ab_stages_three_fits(
        self,
        *,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
    ) -> bool:
        return self._cute_tcgen05_config.ab_stages_three_fits(
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=cluster_m,
        )

    def _tcgen05_grouped_dynamic_ab4_fits_for_target(
        self,
        *,
        dtype_bytes: int,
        device: torch.device,
        bm: int,
        bn: int,
        bk: int,
        cluster_m: int,
        c_stages: int,
    ) -> bool:
        return self._cute_tcgen05_config.grouped_dynamic_ab4_fits_for_target(
            dtype_bytes=dtype_bytes,
            device=device,
            bm=bm,
            bn=bn,
            bk=bk,
            cluster_m=cluster_m,
            c_stages=c_stages,
        )

    def _fix_tcgen05_ab_stages_three_search_config(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._fix_ab_stages_three_search_config(config)

    def _fix_tcgen05_with_scheduler_search_config(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._fix_with_scheduler_search_config(config)

    def _fix_tcgen05_cluster_m1_persistent_search_config(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._fix_cluster_m1_persistent_search_config(config)

    def restrict_tcgen05_num_epi_warps_search(self, choices: tuple[int, ...]) -> None:
        self._cute_tcgen05_config.restrict_num_epi_warps_search(choices)

    def restrict_tcgen05_num_epi_warps_validation(
        self, choices: tuple[int, ...]
    ) -> None:
        self._cute_tcgen05_config.restrict_num_epi_warps_validation(choices)

    def narrow_tcgen05_autotune_to_validated_configs(
        self,
        *,
        allow_persistent_pid_types: bool = False,
        allow_cluster_m2_search: bool = False,
        cluster_m2_static_k: int | None = None,
        allow_cluster_m2_edge_k_tail_family: bool = False,
        allow_cluster_m2_fp8_small_grid: bool = False,
        ab_stages_three_dtype_bytes: int | None = None,
        ab_stages_three_device: torch.device | None = None,
    ) -> None:
        self._cute_tcgen05_config.narrow_autotune_to_validated_configs(
            allow_persistent_pid_types=allow_persistent_pid_types,
            allow_cluster_m2_search=allow_cluster_m2_search,
            cluster_m2_static_k=cluster_m2_static_k,
            allow_cluster_m2_edge_k_tail_family=allow_cluster_m2_edge_k_tail_family,
            allow_cluster_m2_fp8_small_grid=allow_cluster_m2_fp8_small_grid,
            ab_stages_three_dtype_bytes=ab_stages_three_dtype_bytes,
            ab_stages_three_device=ab_stages_three_device,
        )

    def _record_cute_row_residency_provenance(
        self,
        config: dict[str, object],
        provided_keys: set[str],
        fix_invalid: bool,
    ) -> None:
        """⭐ DID THE CALLER *WRITE* ``cute_row_residency``, or did the ladder fill it?

        Records the answer on ``self`` so codegen can ask it.  This is the carrier that
        makes "a named residency is honoured or raises" implementable at all, and it did
        not exist before -- which is worth stating plainly, because the task statement
        said it did:

            "⭐ The provenance machinery already exists -- item 4 built
             ``_cute_row_residency_requested_by_block`` for exactly this 'explicit vs
             default' distinction.  Reuse it; do not re-derive from the value."

        ⛔ MEASURED FALSE.  That dict stores whatever ``_cute_row_residency_config``
        returned, and THAT reads the POST-normalize config (``tile_strategy`` ~:3834
        ``config_get``), i.e. after ``_fill_missing`` has already run.  A ladder-filled
        ``smem`` and a user-written ``smem`` are the same bytes by then, so the dict
        cannot tell them apart.  What it actually distinguishes is "did a STRATEGY
        resolve a residency for this block" vs "the base-class sentinel" -- compiler-side
        provenance, which is strictly weaker.

        The only true user-provenance in the tree is ``provided_keys`` (``normalize``'s
        own local, snapshotted before the ``_fill_missing`` loop), and it used to die at
        the end of that call.  This method is what makes it outlive it.

        ⚠ WHY IT IS KEYED ON ``id(config)`` AND CLEARED, RATHER THAN STORED ON THE
        CONFIG.  Two constraints, both load-bearing:

        * A reserved key inside the config dict (``_cute_row_residency_explicit``) is not
          available: the config is hashed, compared, JSON-round-tripped, cached and
          recorded into ``frozen_configs.json``, so an extra key would change every
          recorded artifact and every cache key, and ``Config.__eq__`` would call two
          equal configs different.
        * An attribute on ``ConfigSpec`` alone is not enough either, because ONE spec
          normalizes MANY configs (the autotuner normalizes thousands against the same
          spec), so a single flag would leak one draw's provenance into the next.

        ⇒ per-config, on the spec, keyed by identity, and ``BoundKernel.to_code`` hands
        codegen the very object it normalized (``kernel.py`` ~:820 copies FIRST and then
        normalizes the copy), so the identity is still valid at codegen.

        ⚠ ``fix_invalid=True`` MEANS "THE AUTOTUNER BUILT THIS", AND IT IS RECORDED AS
        *NOT EXPLICIT* REGARDLESS OF WHICH KEYS ARE PRESENT.  That is the whole reason
        the raise is safe.  ``ConfigGeneration.unflatten`` -> ``flat_config`` writes EVERY
        key and then normalizes with ``_fix_invalid=True``, so a searched config always
        "names" the residency and a provenance test over ``provided_keys`` alone would
        read every draw as explicit -- which is exactly how an earlier attempt at this
        (recorded at ``_reconcile_cute_residency_budget``) came to fire a raise inside the
        autotuner and had to be replaced by a reconcile.  ``_fix_invalid`` is already the
        tree's own "this config was synthesised, silently fix it" signal, so keying on it
        reuses an existing distinction instead of inventing a parallel one.
        """
        if self.backend_name != "cute":
            return
        if fix_invalid or "cute_row_residency" not in provided_keys:
            # ⭐⭐ A LADDER FILL IS RECORDED **CONTENT-KEYED**, SO IT SURVIVES A COPY.
            #
            # ⛔ THE DEFECT THIS CLOSES, AND IT IS NOT HYPOTHETICAL.
            # ``BoundKernel.compile_config`` calls ``_normalized_config_copy``, which does
            # ``Config(**normalized.config)`` and then normalizes AGAIN.  By that second
            # call ``_fill_missing`` has already written ``cute_row_residency``, so it IS
            # in ``provided_keys`` -- and the ``id()``-keyed ``explicit`` set below records
            # the copy as an explicit user request.  The comment further down explains why
            # ``id()`` is right for a config the caller really did write every key of; what
            # it assumed is that ``to_code`` "copies FIRST and then normalizes the copy".
            # ``compile_config`` now normalizes, copies, and normalizes again, so the
            # assumption no longer holds and the carrier reports a LADDER FILL AS EXPLICIT.
            #
            # MEASURED on ``Config(block_sizes=[1, 128, 128])`` (a biased-attention kernel
            # naming no residency at all):
            #
            #     after ONE normalize                  cute_row_residency=['registers']
            #                                          is_explicit = False   ✅
            #     after a COPY + a SECOND normalize    cute_row_residency=['registers']
            #                                          is_explicit = True    ⛔
            #
            # Downstream, any honour-or-error rule that trusts that answer fails a kernel
            # over a default the user never saw.  MEASURED: recording a decline for a
            # no-plan loop-free reduction without this took the cute suite from 13 failures
            # to 84, including EVERY flash-attention test -- the third time an
            # unconditional raise at a decline site has broken the attention examples.
            #
            # ⇒ mark the fill in ``synthesised``, which is CONTENT-keyed precisely so it
            # survives a copy, and which ``cute_row_residency_is_explicit`` already checks
            # FIRST and lets win over ``explicit``.  So the second normalize's (correct,
            # from its own point of view) "the caller passed every key" is overruled by the
            # older, stronger fact: no human ever typed this value.
            #
            # ⛔⛔ KNOWN LIMITATION, MEASURED, AND STATED HERE BECAUSE IT IS EASY TO GET
            # WRONG: THE CONTENT-COLLISION ARGUMENT BELOW **DOES** REACH THIS DIRECTION.
            #
            # An earlier version of this comment claimed it did not, reasoning that a later
            # same-bytes call must be "a request for the value the ladder would have chosen
            # anyway, so the two are indistinguishable in effect".  ⛔ THAT REASONING IS
            # WRONG.  Whether the ladder AGREES with a request has no bearing on whether the
            # EMISSION HONOURED it, which is the entire point of the honour-or-error
            # contract: a ``registers`` request that lands on a ``gmem`` kernel must be
            # refused however the value was arrived at.
            #
            # MEASURED consequence, ``N=512 tpr=128 cute_vector_widths=[2,1]
            # reduction_loops=[None] cute_row_residency=["registers"]`` -- the ladder's own
            # value at N=512 -- in ONE process:
            #
            #     the EXPLICIT config normalized FIRST     -> refused    ✅
            #     a NO-KEY config of the same shape first  -> compiled   ⛔ silently
            #
            # i.e. an explicit request for the ladder's own value is swallowed if a default
            # config for the same shape was normalized earlier in the process.
            #
            # ⚠ WHY IT IS ACCEPTED RATHER THAN FIXED.  The alternative is worse and it is
            # also measured: WITHOUT this record, a config that names NO residency at all
            # hard-raises ``CuteRowResidencyUnavailable`` through ``compile_config`` -- seven
            # tests, including every flash-attention one.  A user who wrote nothing getting
            # an exception is worse than a user who wrote something losing a diagnostic.
            #
            # ⚠ FOUR CARRIERS WERE BUILT AND MEASURED; ALL FAIL, so this is a structural
            # limit rather than an unfinished thought:
            #   * last-writer-wins (also write ``False`` from the explicit branch below) --
            #     breaks the config COPY this record exists to survive, because the copy's
            #     own normalize sees the key present;
            #   * check the ``id()``-keyed set FIRST -- same failure, the copy's own id lands
            #     in it;
            #   * key on the residency LIST's identity -- the copy DOES pass the list object
            #     through, but ``BlockIdSequence._normalize`` does ``values = [*(values or
            #     ())]``, rebuilding the list on EVERY normalize, so no list identity
            #     survives one;
            #   * stop ``compile_config`` re-normalizing (``_normalized_config_copy``) -- this
            #     is the correct upstream fix and it works, but it also stops the
            #     pre-existing LOOPED-path decline from firing, i.e. a behaviour change in
            #     ``runtime/kernel.py`` far outside this change's scope.
            #
            # ⇒ a real fix belongs upstream: either ``compile_config`` not re-normalizing an
            # already-normalized config, or provenance travelling ON the ``Config`` object
            # instead of in a side table beside it.
            if "cute_row_residency" not in provided_keys:
                self._cute_row_residency_synthesised.add(_config_fingerprint(config))
                if len(self._cute_row_residency_synthesised) > 8192:
                    self._cute_row_residency_synthesised.clear()
                    self._cute_row_residency_synthesised.add(
                        _config_fingerprint(config)
                    )
            return
        # ⚠ RECORD ONLY THE POSITIVE.  The reader's default is False, so an absent entry
        # already means "not explicit"; writing False entries would only grow the set and
        # would let a stale entry flip a real request to silent.
        #
        # ⚠⚠ KEYED BY ``id()`` OF THE CONFIG DICT, AND CONTENT WOULD BE **WRONG** HERE --
        # not merely slower.  MEASURED: normalizing three configs that differ only in what
        # the CALLER wrote --
        #     (a) cute_row_residency=["smem"]      explicit
        #     (b) nothing                          ladder fills "smem"
        #     (c) cute_reduction_reload=["smem"]   legacy spelling, translated
        # -- yields three FULLY-NORMALIZED configs that are byte-for-byte identical, so one
        # content fingerprint names all three and the answer for (a) leaks onto (b) and (c).
        # That collision is the very thing this carrier exists to see through: the whole
        # point is that the VALUE cannot distinguish these and the PROVENANCE can.
        #
        # ⭐ And ``id()`` is exactly right for this half: ``BoundKernel.to_code`` copies the
        # config and then normalizes THE COPY (``kernel.py`` ~:820-821), and it is that copy
        # which reaches ``generate_ast`` -- so the dict recorded here IS the dict codegen
        # reads back.  The synthesised set below is the one that has to survive a copy, and
        # it is content-keyed for that reason.
        self._cute_row_residency_explicit.add(id(config))
        if len(self._cute_row_residency_explicit) > 8192:
            # ⚠ ``id()`` IS REUSED AFTER A DICT IS FREED, so this set can hold a stale key
            # that a future unrelated config lands on -- which would read as EXPLICIT and
            # could raise.  Clearing bounds the exposure; the residual risk is accepted
            # because the alternative (holding the dicts alive to keep ids unique) leaks
            # every config in a search.  A false EXPLICIT can only ever fire when the
            # residency ALSO could not be honoured, i.e. on a config that was already
            # emitting the wrong residency silently.
            self._cute_row_residency_explicit.clear()

    def cute_row_residency_is_explicit(self, config: dict[str, object]) -> bool:
        """Did the CALLER write ``cute_row_residency`` in this config?  See above.

        ⚠⚠ FALSE IS THE FAIL-SAFE ANSWER, AND THE TWO CHECKS ARE ORDERED ACCORDINGLY.
        A config that never went through this spec's ``normalize`` has no provenance, and
        reading "no record" as EXPLICIT would turn every such path into a raise -- so the
        polarity is chosen such that LOSING the record degrades to today's behaviour (a
        silent decline), never to a new failure.

        ⭐ ``synthesised`` WINS OVER ``explicit``, and the order is load-bearing.  A
        machine-built config is re-normalized on the user path (``BoundKernel.to_code``
        calls ``normalize`` a second time with ``_fix_invalid=False``), which records it as
        explicit -- correctly, from that call's point of view, since the caller really did
        pass every key.  The ``synthesised`` set is the older, stronger fact: it says the
        keys were never typed by a human in the first place.  MEASURED: with the checks in
        the other order, ``HELION_AUTOTUNE_EFFORT=none`` raises on two phase-A gate tests.
        """
        if _config_fingerprint(config) in self._cute_row_residency_synthesised:
            return False
        return id(config) in self._cute_row_residency_explicit

    def _normalize_cute_row_residency(
        self, config: dict[str, object], provided_keys: set[str]
    ) -> None:
        """⭐ Reconcile ``cute_row_residency`` with the two knobs it supersedes.

        THE COMPATIBILITY RULE, and it is the thing that keeps the 40 frozen cells
        byte-identical: whichever key the CALLER actually wrote is authoritative.

        * the caller named ``cute_row_residency``     -> honour it, untouched;
        * the caller named only the OLD keys          -> derive the axis from them via
          ``row_residency_from_legacy``, so an existing config keeps meaning exactly
          what it meant (and ``.expected`` goldens do not move);
        * the caller named NEITHER                    -> ``_fill_missing`` already put
          the per-shape ladder in, which reproduces the two old ladders composed.

        ⚠ WHY ``provided_keys`` AND NOT "is the value falsy".  By the time this runs
        every one of the three keys is present and filled, because ``_normalize``
        above ran on all of them -- so "did the caller ask for this?" is no longer
        readable off the values.  ``provided_keys`` is snapshotted BEFORE that loop
        for exactly this reason.  Reading the filled values instead would make an
        explicit ``cute_row_residency=["gmem"]`` indistinguishable from a ladder fill
        and silently overwrite it from the old keys.

        ⚠ ``cute_tv_sweep_cache`` IS NOT REWRITTEN OR REMOVED.  It stays in the config as
        the caller left it, because it still has a real job -- a per-thread register BUDGET
        that can make ``"registers"`` DECLINE.  It is no longer a selector, which is the
        whole content of FIXLIST item 2.

        ⭐⭐ ``cute_reduction_reload`` IS STRIPPED (task 1), AND THE STRIP IS THE FIX.
        Before task 1 this method filled the new key and left the old one in place, so the
        config that reached codegen carried the decision TWICE and
        ``TileStrategy._cute_row_residency_config`` still read the old pair as a live
        fallback -- i.e. the two-encodings bug the axis exists to remove was still
        expressible.  Now the translation happens exactly once, here, at the door:

            in :  cute_reduction_reload=[...]  (+ optional cute_tv_sweep_cache)
            out:  cute_row_residency=[...]     and NO cute_reduction_reload

        so after ``normalize()`` there is exactly ONE key carrying the residency, no
        recorded artifact can spell it the old way, and nothing downstream reads the old
        name.  What survives is an accepted INPUT spelling, which is what FIXLIST item 1
        asks for ("``row_residency_from_legacy()`` a one-shot config migration rather than
        a live translation").

        ⚠ WHY AN ACCEPTED INPUT AT ALL, RATHER THAN A HARD REMOVAL.  Two reasons, one of
        them a hard constraint: (a) ``_notes/tests/test_staging_on_persistent.py`` is a
        GRADED contract test (level 8 forbids editing it) and it passes
        ``cute_reduction_reload=["smem"]`` to ``helion.Config`` directly, so a hard removal
        turns a green contract test red; (b) a translation-at-the-door is where a
        compatibility shim belongs, and it costs nothing downstream because the key is
        gone from the config by the time anything reads it.
        """
        # ⭐⭐ THE DEMOTION (task 2, FIXLIST item 2): ``cute_tv_sweep_cache`` IS NOT IN
        # THIS SET.  It used to be, and that made the BUDGET a SELECTOR of the residency
        # axis -- the exact defect item 2 names ("``sweep_cache`` re-decides the rmem arm
        # from a second place ... nothing rejects the pair and the budget wins, so the
        # emitted kernel contradicts the config").
        #
        # MEASURED before this edit, on ``rms_norm`` M=2048 N=8192 bf16, normalizing a
        # config that names ONLY the budget:
        #
        #     cute_tv_sweep_cache=[0]    ->  cute_row_residency=['gmem']       ⬅ THE BUG
        #     cute_tv_sweep_cache=[16]   ->  cute_row_residency=['registers']
        #     cute_tv_sweep_cache=[128]  ->  cute_row_residency=['registers']
        #     (nothing named)            ->  cute_row_residency=['smem']  (the ladder)
        #
        # i.e. an integer whose documented job is "how many registers may the cache spend"
        # was choosing WHICH OF THREE KERNELS to emit, through
        # ``row_residency_from_legacy``'s ``cache_slots > 0`` test -- a mode switch wearing
        # a budget's name.  After this edit the axis is decided by ``cute_row_residency``
        # (explicit) or by its own per-shape ladder (absent), and the budget only ever
        # CAPS the ``registers`` arm at the pass.
        #
        # ⚠⚠ THIS IS A BEHAVIOUR CHANGE, NOT A REACHABILITY CHANGE, AND IT IS THE ONLY
        # ONE IN TASKS 1-3.  A config that names ONLY ``cute_tv_sweep_cache=[0]`` used to
        # compile to a ``gmem`` kernel and now compiles to whatever the ladder asks for.
        # That is the intended content of the demotion -- "0 registers" and "do not use
        # the register mechanism" are different statements and only one of them is this
        # key's job -- but it must be said out loud rather than discovered.
        #
        # ⭐ IT MOVES NO FROZEN CELL, and that is a dividend of doing task 1 first: all 25
        # cells that used to lean on the legacy encoding now name ``cute_row_residency``
        # explicitly, so the two cells that DO name a budget
        # (``layer_norm/8192x100000`` cache=[32] -> residency=['registers'],
        #  ``rms_norm/8192x100000`` cache=[0] -> residency=['smem']) take their residency
        # from the axis and not from the budget.  VERIFIED: level 1, 40/40 byte-identical.
        #
        # ``cute_reduction_reload`` stays in the set because it genuinely IS a residency
        # spelling (task 1 keeps it as a translated-and-stripped input alias).
        legacy_named = provided_keys & {"cute_reduction_reload"}
        if "cute_row_residency" in provided_keys or not legacy_named:
            # ⚠ STILL STRIP, even on the path that does not translate.  A caller who
            # names BOTH keys gets the new one honoured (it is authoritative) and the old
            # one removed, so the two cannot disagree on the emitted artifact.
            config.pop("cute_reduction_reload", None)
            return
        reload_values = config.get("cute_reduction_reload")
        residency = list(cast("list[str]", config.get("cute_row_residency") or []))
        # ⭐ VALIDATE THE LEGACY INPUT LOUDLY, because deleting ``CuteReductionReloadSpec``
        # deleted the ``_normalize`` that used to do it.  Without this an illegal value
        # would be silently COERCED: ``row_residency_from_legacy`` tests ``== "smem"``, so
        # ``cute_reduction_reload=["banana"]`` would quietly mean ``registers``.  A typo
        # that compiles to a plausible kernel makes every recorded config unverifiable --
        # this repo's enumeration antipattern, and exactly what ``CuteRowResidencySpec
        # ._normalize`` refuses to do for the new key.  The message keeps the old spec's
        # wording so an existing config's error is unchanged.
        if isinstance(reload_values, (list, tuple)):
            for i, value in enumerate(reload_values):
                if value not in REDUCTION_RELOAD_CHOICES:
                    raise InvalidConfig(
                        f"config[cute_reduction_reload][{i}] must be one of "
                        f"{REDUCTION_RELOAD_CHOICES}, got {value!r}"
                    )
        for index, spec in enumerate(self.cute_row_residency):
            if index >= len(residency):
                break
            block_id = spec.block_id
            # ⚠ READ THROUGH ``cute_row_residency``'s OWN INDEX MAP, not through a
            # ``cute_reduction_reload`` sequence -- that sequence no longer exists (task 1).
            # This is sound rather than a coincidence: both keys were registered from the
            # SAME loop over the same ``rdims`` in ``DeviceIR._register_cute_tv_layout_slots``,
            # one slot per reduction block, so their block_id -> index maps were identical by
            # construction.  ⇒ the legacy list is positionally interchangeable with this one.
            reload_from = self.cute_row_residency.config_get(
                cast("list[str | None]", reload_values or []), block_id, None
            )
            # ⚠ ``cache_slots=None`` ALWAYS, because ``cute_tv_sweep_cache`` no longer
            # exists (task 1 steps 2-3 deleted it).  ``None`` means "no budget stated",
            # which ``row_residency_from_legacy`` maps to ``registers`` for
            # ``reload=None`` -- the same answer the old positive default gave.  ⇒ the
            # legacy spelling keeps meaning exactly what it meant; what is gone is the
            # ability of a budget to turn that into ``gmem``.
            residency[index] = row_residency_from_legacy(
                cast("str | None", reload_from), None
            )
        config["cute_row_residency"] = residency
        # ⭐ THE ONE-SHOT MIGRATION, COMPLETED: the old spelling has been read, translated
        # and is now removed.  Popping it here rather than leaving it is what makes this a
        # migration instead of a second encoding -- see the docstring.
        config.pop("cute_reduction_reload", None)

    def supports_config_key(self, key: str) -> bool:
        return self.backend.supports_config_key(key)

    def supported_config_keys(self) -> frozenset[str]:
        return frozenset(key for key in VALID_KEYS if self.supports_config_key(key))

    def _default_num_stages(self) -> int:
        return DEFAULT_NUM_STAGES

    def _num_stages_fragment(self) -> ConfigSpecFragment:
        if self.backend_name == "tileir":
            return EnumFragment(choices=tuple(range(1, 11)))
        if supports_amd_cdna_tunables():
            return IntegerFragment(1, 4, self._default_num_stages())
        if self.backend_name == "metal":
            return IntegerFragment(1, 1, 1)
        return IntegerFragment(1, 8, self._default_num_stages())

    def _tcgen05_optional_fragments(
        self, *, for_search: bool = False
    ) -> dict[str, ConfigSpecFragment]:
        return self._cute_tcgen05_config.optional_fragments(for_search=for_search)

    def _tcgen05_strategy_autotune_fragments(
        self,
    ) -> dict[str, ConfigSpecFragment]:
        return self._cute_tcgen05_config.strategy_autotune_fragments()

    def _tcgen05_strategy_validation_fragments(
        self,
    ) -> dict[str, ConfigSpecFragment]:
        return self._cute_tcgen05_config.strategy_validation_fragments()

    @staticmethod
    def _tcgen05_strategy_field_default(key: str, *, pid_type: object = None) -> object:
        return CuteTcgen05Config.strategy_field_default(key, pid_type=pid_type)

    def _validate_tcgen05_strategy_invariants_in_normalize(
        self,
        config: dict[str, object],
        *,
        _fix_invalid: bool,
    ) -> None:
        self._cute_tcgen05_config.validate_strategy_invariants(
            config,
            fix_invalid=_fix_invalid,
        )

    @staticmethod
    def _validate_optional_fragment_value(
        name: str, fragment: ConfigSpecFragment, value: object
    ) -> object:
        return CuteTcgen05Config._validate_optional_fragment_value(
            name,
            fragment,
            value,
        )

    def _clamp_tcgen05_l2_swizzle_size_to_shape(
        self, config: dict[str, object]
    ) -> None:
        self._cute_tcgen05_config._clamp_l2_swizzle_size_to_shape(config)

    def unsupported_config_keys(self, config: Mapping[str, object]) -> list[str]:
        return sorted(
            key
            for key in config
            if key in VALID_KEYS and not self.supports_config_key(key)
        )

    def is_supported_config(self, config: Mapping[str, object]) -> bool:
        return not self.unsupported_config_keys(config)

    def normalize(
        self, config: helion.Config | dict[str, object], *, _fix_invalid: bool = False
    ) -> None:
        """Normalize the config to match the block_sizes and validate the config.

        Args:
            config: The config to normalize (modified in place).
            _fix_invalid: If True, silently fix invalid combinations instead of raising
                errors. Used internally during autotuning config generation.
        """
        if isinstance(config, helion.Config):
            self.normalize(config.config, _fix_invalid=_fix_invalid)
            return

        for name in (
            "block_size",
            "loop_order",
            "reduction_loop",
            "l2_grouping",
            "flatten_loop",
            "range_unroll_factor",
            "range_warp_specialize",
            "range_num_stage",
            "range_multi_buffer",
            "range_flatten",
            "static_range",
        ):
            if name in config:
                names = f"{name}s"
                if names in config:
                    raise InvalidConfig(f"Cannot specify both {name} and {names}")
                value = config.pop(name)
                if name == "reduction_loop" and len(self.reduction_loops) > 1:
                    # Apply the same reduction_loop setting to every
                    # reduction dimension so a single scalar value works
                    # when multiple dims can be rolled.
                    config[names] = [value for _ in range(len(self.reduction_loops))]
                else:
                    config[names] = [value]

        if unsupported := self.unsupported_config_keys(config):
            # Separate backend-specific keys (e.g. AMD tunables, TileIR tunables)
            # from common keys (e.g. num_warps, num_stages, indexing).
            # Backend-specific keys should raise errors; common keys are
            # silently stripped so configs are portable across backends.
            backend_specific = [k for k in unsupported if k in BACKEND_SPECIFIC_KEYS]
            common = [k for k in unsupported if k not in BACKEND_SPECIFIC_KEYS]
            for key in common:
                config.pop(key, None)
            if backend_specific:
                if _fix_invalid:
                    for key in backend_specific:
                        config.pop(key, None)
                else:
                    raise InvalidConfig(
                        f"Unsupported config keys for backend {self.backend_name!r}: {backend_specific}"
                    )
        if self.backend_name == "cute":
            self._cute_tcgen05_config.prepare_normalization(
                config, fix_invalid=_fix_invalid
            )
        provided_keys = set(config)
        if _fix_invalid:
            self._pre_normalize_cute_flash_block_sizes(config)

        for name, mapping, flatten in [
            ("block_sizes", self.block_sizes, True),
            ("num_threads", self.num_threads, True),
            ("flatten_loops", self.flatten_loops, True),
            ("l2_groupings", self.l2_groupings, True),
            ("loop_orders", self.loop_orders, False),
            ("reduction_loops", self.reduction_loops, True),
            ("cute_vector_widths", self.cute_vector_widths, True),
            ("cute_threads_per_row", self.cute_threads_per_row, True),
            ("cute_row_residency", self.cute_row_residency, True),
            ("cute_ndtile_tv", self.cute_ndtile_tv, True),
            ("cute_online_defer", self.cute_online_defer, True),
            ("range_unroll_factors", self.range_unroll_factors, True),
            ("range_warp_specializes", self.range_warp_specialize, True),
            ("range_num_stages", self.range_num_stages, True),
            ("range_multi_buffers", self.range_multi_buffers, True),
            ("range_flattens", self.range_flattens, True),
            ("static_ranges", self.static_ranges, True),
        ]:
            if not self.supports_config_key(name):
                if name in config:
                    raise InvalidConfig(
                        f"{name} is not supported on backend {self.backend_name!r}"
                    )
                config.pop(name, None)
                continue
            config[name] = mapping._normalize(
                name, config.get(name, ()), flatten=flatten
            )

        if self.supports_config_key("cute_row_residency") and len(
            self.cute_row_residency
        ):
            self._normalize_cute_row_residency(config, provided_keys)
            # ⭐ Task 2's reconciliation, AFTER the translation above so it sees the one
            # authoritative residency rather than an old spelling.  ⚠ It reads the
            # POST-translation residency deliberately -- an explicit request is explicit
            # however it was spelled -- so it needs no ``provided_keys``.

        # Clamp inner block sizes that are bounded by an outer block
        # (e.g. ``hl.tile(outer.begin, outer.end)``): at this point the
        # outer's concrete block size for this config is known, and the
        # inner extent can never exceed it.
        block_sizes_list = config.get("block_sizes")
        if isinstance(block_sizes_list, list):
            changed = False
            new_block_sizes = list(block_sizes_list)
            for i, spec in enumerate(self.block_sizes):
                bb = spec.bounded_by_block_id
                if (
                    bb is None
                    or i >= len(new_block_sizes)
                    or new_block_sizes[i] is None
                ):
                    continue
                try:
                    outer_index = self.block_sizes.block_id_to_index(bb)
                except KeyError:
                    continue
                outer_val = (
                    new_block_sizes[outer_index]
                    if outer_index < len(new_block_sizes)
                    else None
                )
                if (
                    isinstance(outer_val, int)
                    and isinstance(new_block_sizes[i], int)
                    and new_block_sizes[i] > outer_val
                ):
                    new_block_sizes[i] = outer_val
                    changed = True
            if changed:
                config["block_sizes"] = new_block_sizes
                num_threads = config.get("num_threads")
                if isinstance(num_threads, list):
                    new_num_threads = list(num_threads)
                    for i, (block_size, num_thread) in enumerate(
                        zip(new_block_sizes, new_num_threads, strict=False)
                    ):
                        if (
                            type(block_size) is not int
                            or type(num_thread) is not int
                            or num_thread <= 0
                        ):
                            continue
                        if num_thread > block_size:
                            num_thread = 1 << (max(block_size, 1).bit_length() - 1)
                        while num_thread > 1 and block_size % num_thread != 0:
                            num_thread //= 2
                        new_num_threads[i] = max(num_thread, 1)
                    config["num_threads"] = new_num_threads

        if self.supports_config_key("num_threads"):
            num_threads = cast("list[int]", config.get("num_threads", []))
            if all(value == 0 for value in num_threads):
                config.pop("num_threads", None)
        else:
            config.pop("num_threads", None)

        # Cap reduction loops at the backend's max loop chunk, while using the
        # live reduction thread threshold to decide when a persistent reduction
        # must be rolled.
        if self.max_reduction_threads is not None and self.reduction_loops:
            force_threshold = self.reduction_loop_force_threshold
            max_loop = self.max_reduction_loop
            reduction_loops = config.get("reduction_loops", [])
            if force_threshold is not None and isinstance(reduction_loops, list):
                new_loops = list(reduction_loops)
                changed = False
                for i, spec in enumerate(self.reduction_loops):
                    if i >= len(new_loops):
                        break
                    # Indexed reductions (argmin/argmax) on CuTe must keep
                    # the persistent thread count or rolled chunk within a
                    # single warp, since cute.arch.warp_reduction only
                    # supports threads_in_group<=32.
                    block_threshold = force_threshold
                    if (
                        self.backend_name == "cute"
                        and spec.block_id in self.cute_indexed_reduction_block_ids
                    ):
                        block_threshold = min(block_threshold, 32)
                    # ⭐⭐ THE ``[None]`` ARM IS GONE (A1, site 1).  It used to read
                    # ``if new_loops[i] is None and per_cta_hint > block_threshold:
                    # new_loops[i] = min(spec.size_hint, block_threshold)`` -- i.e. a
                    # PERSISTENT request whose per-CTA extent exceeded the thread budget was
                    # silently answered with a LOOPED reduction.
                    #
                    # It existed to dodge a real correctness bug, not a hardware limit: with
                    # the thread count capped at ``max_reduction_threads`` the synthetic lane
                    # loop was never created above 1024, so the reduction visited only the
                    # first 1024 elements (fixed in ``de0267822`` -- ``needs_synthetic`` now
                    # compares against the REAL extent).  With that fixed, persistent is
                    # bit-exact at every N measured up to 32768, so the substitution is no
                    # longer buying anything and a ``[None]`` request is honoured.
                    #
                    # ⚠ The ``elif`` below is UNTOUCHED and still load-bearing: it caps an
                    # EXPLICIT chunk at ``max_reduction_loop`` (and at 32 for an indexed
                    # reduction, where ``warp_reduction`` only supports
                    # ``threads_in_group <= 32``).  That is a cap on a value the caller
                    # actually named, not a substitution for one they did not.
                    if (
                        new_loops[i] is not None
                        and max_loop is not None
                        and (
                            new_loops[i] > max_loop
                            or (
                                self.backend_name == "cute"
                                and spec.block_id
                                in self.cute_indexed_reduction_block_ids
                                and new_loops[i] > 32
                            )
                        )
                    ):
                        new_loops[i] = min(
                            new_loops[i] if max_loop is None else max_loop,
                            block_threshold,
                        )
                        changed = True
                if changed:
                    config["reduction_loops"] = new_loops

        # ⭐⭐ A1 (site 2): THE ``available``-BASED FORCE-ROLL AND ITS RAISE ARE DELETED.
        #
        # This block asked "do the non-reduction thread axes leave room for the reduction
        # axis?" -- by multiplying every ``num_threads`` slot's thread count (falling back to
        # that axis's block size for the ``0`` = "auto" sentinel) -- and if not, either
        # rewrote a PERSISTENT ``reduction_loops=[None]`` request into a looped chunk or
        # raised ``InvalidConfig``.  Both halves are obsolete:
        #
        # 1. The ROLL dodged a real bug (above ``max_reduction_threads`` the synthetic lane
        #    loop was never built, so the reduction visited only the first 1024 elements),
        #    fixed in ``de0267822``.  With that fixed the substitution buys nothing.
        #
        # 2. The RAISE was kept at first as the honouring-compatible half, then MEASURED to
        #    protect nothing: ``sum(x[tm,:])`` at ``block_sizes=[1024] num_threads=[1024]``
        #    (so ``available == 1``) with ``reduction_loops=[None]`` is **bit-exact** at
        #    ``cute_vector_widths`` 1 and 8 once the raise is removed ({-1,+1} integer data,
        #    so a correct kernel MUST be bit-exact), and ``InvalidConfig`` with it in place.
        #    ⇒ it was rejecting working configs.
        #
        # ⛔ AND THE ARITHMETIC WAS WRONG ON MULTI-NEST KERNELS, which is how this was found.
        # A kernel's loop nests are not simultaneously live.  MEASURED on
        # ``examples/split_k_barrier`` at ``block_sizes=[16, 8, 16, 16, 16]``: block_ids
        # 0/1/3 belong to the split-K matmul and 4/5 to the epilogue that sums the partials,
        # separated by ``hl.barrier()`` -- yet the product was ``16*8*16*16*16 = 524288``
        # against a 1024-thread budget, so ``available`` collapsed to 1 and this raised
        # ``InvalidConfig: reduction axis 0 has no thread budget left (non-reduction axes use
        # 524288 of 1024 threads)`` on a kernel that launches fine.  (The reduction there is
        # the epilogue's ``tmp[tm,tn,:].sum(-1)`` over ``split_k=64`` -- an ordinary
        # reduction needing 64 threads, not the K axis; split-K itself is a plain tiled
        # matmul.)  It was LATENT before A1: the roll consumed the nonsense value and still
        # compiled, so the bad arithmetic never became visible.
        #
        # ⇒ DELETED rather than repaired.  Repairing it would mean making the thread
        # accounting nest-aware in order to serve a rule that no longer has a job.
        # ``adjust_reduction_thread_count`` already shrinks the reduction thread count to fit
        # the real launch budget at codegen time, where the nest structure is known -- that is
        # the right place for the question, and it is already answered there.

        # ⭐⭐ A1: THE PERSISTENT FORCE-ROLL IS GONE.  It used to live here.
        #
        # It rewrote ``reduction_loops[i] = size_hint // 2`` whenever the caller asked for
        # ``None`` (persistent) with ``cute_vector_widths[bid] > 1``, i.e. it answered a
        # persistent request with a LOOPED reduction and said nothing.  Its justification was
        # real at the time: ``codegen_preamble`` built exactly ONE lane nest, so a second
        # dependent reduction had nowhere to open its own, and persistent + TV raised
        # ``BackendUnsupported: a lane reduction ... still owing its lane fold and its
        # cross-thread combine``.  Rolling dodged that by routing the shape to the path whose
        # ``ReductionRoller`` already gives one subgraph per dependency layer.
        #
        # ⇒ FIXED AT LOWERING INSTEAD, which is where the missing capability was:
        # ``DeviceGridState.prebuilt_lane_nest_factory`` makes the TV nest REBUILDABLE, so
        # ``_wrap_segmented_body`` mints one nest per sealed segment and the existing seals
        # mechanism covers dependent reductions on the TV path too.  A ``[None]`` request now
        # yields a persistent reduction, or a loud error -- never a silent substitution.
        #
        # MEASURED after the change: the 108-config loop-free grid goes from
        # ``{0: 9, 32: 45, 64: 36, 128: 18}`` back to ``{32: 36, 64: 36, 128: 36}`` (the
        # as-received widths, 0 at the scalar floor, 0 configs narrower), 81 of 108 configs
        # are now genuinely ``PersistentReductionStrategy``, the dependent ``amax`` ->
        # ``sum(exp(v - amax))`` pair compiles and is correct, and all 40 frozen cells are
        # byte-identical.
        #
        # ⚠ Sites 1 and 2 above still roll when the extent exceeds the available reduction
        # threads; that is a SEPARATE limit (the synthetic-lane accumulator, see their own
        # comments) and is why N > 1024 still rolls.  Only this TV-width roll is deleted.

        # Disable range_* configs for static ranges
        static_range_block_ids = [
            block_id
            for block_id in self.static_ranges.valid_block_ids()
            if self.static_ranges.config_get(
                cast("list[bool]", config.get("static_ranges", [])),
                block_id,
            )
        ]
        if static_range_block_ids:
            for name, mapping in (
                ("range_unroll_factors", self.range_unroll_factors),
                ("range_warp_specializes", self.range_warp_specialize),
                ("range_num_stages", self.range_num_stages),
                ("range_multi_buffers", self.range_multi_buffers),
                ("range_flattens", self.range_flattens),
            ):
                config[name] = mapping._reset_config_to_default(
                    name, config.get(name, ()), block_ids=static_range_block_ids
                )

        for name in (
            "loop_orders",
            "l2_groupings",
            "flatten_loops",
            "reduction_loops",
            "cute_vector_widths",
            "cute_threads_per_row",
            "cute_row_residency",
            # Stripped only when the LIST is empty (no slot registered at all,
            # e.g. a kernel with no device loop, or a non-cute backend).  A
            # registered slot holding ``False`` / ``0`` survives, because
            # ``[False]`` and ``[0]`` are non-empty lists and therefore truthy --
            # which matters here as those are exactly the values that turn a pass
            # OFF, and a key that vanished would be re-filled with the ladder's
            # ``True`` / ``128`` and silently turn it back on.
            "cute_ndtile_tv",
            "cute_online_defer",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "load_cache_modifiers",
            "store_cache_modifiers",
            "indexing",
            "atomic_indexing",
        ):
            if not config.get(name):
                config.pop(name, None)

        # Remove unsupported keys before setting defaults
        for name in (
            "num_warps",
            "num_stages",
            "load_eviction_policies",
            "load_cache_modifiers",
            "store_cache_modifiers",
            "indexing",
            "atomic_indexing",
            "pallas_load_buffer_count",
            "pid_type",
            "num_sm_multiplier",
            "maxnreg",
        ):
            if not self.supports_config_key(name):
                config.pop(name, None)

        if self.supports_config_key("num_warps"):
            config.setdefault("num_warps", DEFAULT_NUM_WARPS)
        if self.supports_config_key("num_stages"):
            config.setdefault("num_stages", self._default_num_stages())
        if self.supports_config_key("load_eviction_policies"):
            config.setdefault(
                "load_eviction_policies", self.load_eviction_policies.default()
            )
        if (
            self.supports_config_key("load_cache_modifiers")
            and self.load_cache_modifiers.length > 0
        ):
            config.setdefault(
                "load_cache_modifiers", self.load_cache_modifiers.default()
            )
        if (
            self.supports_config_key("store_cache_modifiers")
            and self.store_cache_modifiers.length > 0
        ):
            config.setdefault(
                "store_cache_modifiers", self.store_cache_modifiers.default()
            )
        if self.supports_config_key("indexing"):
            config.setdefault("indexing", self.indexing.default())
        if self.supports_config_key("atomic_indexing"):
            config.setdefault("atomic_indexing", self.atomic_indexing.default())
        for key, fragment in self.backend_tunable_fragments.items():
            config.setdefault(key, fragment.default())
        if self.backend_name == "cute":
            self._cute_tcgen05_config.normalize_pre_pid_type(
                config,
                fix_invalid=_fix_invalid,
            )
        if self.has_pallas_inner_loops:
            if self.has_symbolic_or_data_dependent_bounds:
                # "unroll" uses Python range() which can't handle traced bounds.
                # Between the remaining options, prefer "fori_loop": it handles
                # both DMA-aligned and unaligned inner blocks, while
                # "emit_pipeline" fails on unaligned dims.
                config.setdefault("pallas_loop_type", "fori_loop")
            else:
                config.setdefault("pallas_loop_type", VALID_PALLAS_LOOP_TYPES[0])
        if (
            self.supports_config_key("pallas_load_buffer_count")
            and self.has_pallas_inner_loops
            and config.get("pallas_loop_type") == "fori_loop"
        ):
            values = config.setdefault(
                "pallas_load_buffer_count", self.pallas_load_buffer_count.default()
            )
            expected = self.pallas_load_buffer_count.length
            if (
                not isinstance(values, list)
                or len(values) != expected
                or any(
                    type(value) is not int or value not in (1, 2) for value in values
                )
            ):
                raise InvalidConfig(
                    "pallas_load_buffer_count must be a list containing one "
                    "buffer count (1 or 2) per input tensor "
                    f"(expected {expected}, got {values!r})"
                )
            if expected == 0:
                config.pop("pallas_load_buffer_count")
        else:
            config.pop("pallas_load_buffer_count", None)

        if (
            self.supports_config_key("pallas_pre_broadcast")
            and self.has_pallas_inner_loops
            and config.get("pallas_loop_type") not in ("fori_loop", "emit_pipeline")
        ):
            # The transform widens loop-carried VMEM scratch, so it only applies
            # to the streaming lowerings.  "unroll" carries values through the
            # jax.lax.fori_loop tuple and allocates no scratch to widen; pin the
            # flag off there so both settings do not autotune as distinct configs.
            config.pop("pallas_pre_broadcast", None)

        if self.supports_config_key("pid_type"):
            if "pid_type" in config:
                if config["pid_type"] not in VALID_PID_TYPES:
                    raise InvalidConfig(
                        f"Invalid value for 'pid_type': {config['pid_type']!r} must be one of {list(VALID_PID_TYPES)!r}"
                    )
            else:
                config["pid_type"] = VALID_PID_TYPES[0]

        if self.supports_config_key("xcd_remap"):
            if "xcd_remap" in config:
                if not isinstance(config["xcd_remap"], bool):
                    raise InvalidConfig(
                        f"Invalid value for 'xcd_remap': {config['xcd_remap']!r} must be a bool"
                    )
                if config["xcd_remap"]:
                    pid_type = config.get("pid_type", "flat")
                    if self.num_xcd <= 1:
                        # No-op on single-XCD devices: silently disable rather
                        # than reject (the remap is the identity at NUM_XCDS=1).
                        config["xcd_remap"] = False
                    elif pid_type not in (
                        "flat",
                        "persistent_blocked",
                        "persistent_interleaved",
                    ):
                        # xcd_remap is only defined for flat and the persistent
                        # (blocked / interleaved) PID strategies.
                        if _fix_invalid:
                            config["xcd_remap"] = False
                        else:
                            raise InvalidConfig(
                                "xcd_remap=True requires pid_type in "
                                "{'flat', 'persistent_blocked', 'persistent_interleaved'}"
                            )
                    elif pid_type == "persistent_interleaved":
                        # interleaved remaps each virtual pid, so it needs the
                        # persistent grid stride to be XCD-aligned (this can be
                        # broken by reserved_sms); otherwise a worker spans
                        # multiple XCD regions.  Silently disable (perf no-op).
                        mult = config.get("num_sm_multiplier", 1)
                        if not isinstance(mult, int) or mult < 1:
                            mult = 1
                        if (self.num_sm * mult) % self.num_xcd != 0:
                            config["xcd_remap"] = False
            else:
                config["xcd_remap"] = False
        else:
            config.pop("xcd_remap", None)

        if _fix_invalid and self.backend_name == "cute":
            self._cute_tcgen05_config.fix_search_config(config)

        if self.backend_name == "cute":
            self._cute_tcgen05_config.normalize_strategy(
                config,
                fix_invalid=_fix_invalid,
            )
            self._normalize_cute_flash(config, fix_invalid=_fix_invalid)

        if self.supports_config_key("num_sm_multiplier"):
            # Validate num_sm_multiplier is a power of two in range
            if "num_sm_multiplier" in config:
                val = config["num_sm_multiplier"]
                if (
                    not isinstance(val, int)
                    or val < MIN_NUM_SM_MULTIPLIER
                    or val > MAX_NUM_SM_MULTIPLIER
                    or (val & (val - 1)) != 0  # not a power of two
                ):
                    raise InvalidConfig(
                        f"Invalid value for 'num_sm_multiplier': {val!r} must be a power of two between {MIN_NUM_SM_MULTIPLIER} and {MAX_NUM_SM_MULTIPLIER}"
                    )
            else:
                config["num_sm_multiplier"] = DEFAULT_NUM_SM_MULTIPLIER

        # Only validate maxnreg on CUDA devices (not supported on AMD and Intel GPU)
        if self.supports_config_key("maxnreg") and supports_maxnreg():
            if "maxnreg" in config:
                if config["maxnreg"] not in VALID_MAXNREG:
                    raise InvalidConfig(
                        f"Invalid value for 'maxnreg': {config['maxnreg']!r} must be one of {list(VALID_MAXNREG)!r}"
                    )
            else:
                config["maxnreg"] = VALID_MAXNREG[0]

            # Cap maxnreg so that maxnreg * threads_per_block doesn't exceed
            # the register file.  On sm100+ ptxas honours .maxnreg over
            # .reqntid, so an uncapped value causes "out of resource: threads"
            # at load.
            maxnreg = cast("int | None", config.get("maxnreg"))
            num_warps = config.get("num_warps", DEFAULT_NUM_WARPS)
            if maxnreg is not None and isinstance(num_warps, int):
                limit = _regs_per_block() // warps_to_threads(num_warps)
                if maxnreg > limit:
                    if _fix_invalid:
                        valid = [
                            v for v in VALID_MAXNREG if v is not None and v <= limit
                        ]
                        if valid:
                            config["maxnreg"] = max(valid)
                        else:
                            config.pop("maxnreg", None)
                    else:
                        raise InvalidConfig(
                            f"maxnreg={maxnreg} exceeds register budget for "
                            f"num_warps={num_warps} (max {limit})"
                        )
        else:
            # Remove maxnreg if not supported
            config.pop("maxnreg", None)

        # Handle num_sm_multiplier and maxnreg for non-persistent pid_types
        # These options only make sense for persistent kernels
        pid_type = config.get("pid_type")
        if pid_type in ("flat", "xyz"):
            # Handle num_sm_multiplier
            num_sm_multiplier = config.get(
                "num_sm_multiplier", DEFAULT_NUM_SM_MULTIPLIER
            )
            if num_sm_multiplier != DEFAULT_NUM_SM_MULTIPLIER:
                if _fix_invalid:
                    # Silently fix during autotuning config generation
                    config.pop("num_sm_multiplier", None)
                else:
                    # Raise error for user-specified invalid combinations
                    raise InvalidConfig(
                        f"num_sm_multiplier={num_sm_multiplier} can only be used with persistent "
                        f"pid_type ('persistent_blocked' or 'persistent_interleaved'), "
                        f"got pid_type={pid_type!r}"
                    )
            else:
                # Remove default value from config
                config.pop("num_sm_multiplier", None)

            # Handle maxnreg - only makes sense for persistent kernels (and only on non-AMD and non-Intel GPU)
            if supports_maxnreg():
                maxnreg = config.get("maxnreg", DEFAULT_MAXNREG)
                if maxnreg != DEFAULT_MAXNREG:
                    if _fix_invalid:
                        # Silently fix during autotuning config generation
                        config.pop("maxnreg", None)
                    else:
                        # Raise error for user-specified invalid combinations
                        raise InvalidConfig(
                            f"maxnreg={maxnreg} can only be used with persistent "
                            f"pid_type ('persistent_blocked' or 'persistent_interleaved'), "
                            f"got pid_type={pid_type!r}"
                        )
                else:
                    # Remove default value from config
                    config.pop("maxnreg", None)

        if "advanced_controls_file" in config:
            value = config.get("advanced_controls_file") or ""
            if not isinstance(value, str):
                raise InvalidConfig(
                    f"advanced_controls_file must be a string path, got {value!r}"
                )
            config["advanced_controls_file"] = value

        if "epilogue_subtile" in config:
            val = config["epilogue_subtile"]
            # Normalize bool to int for backward compat
            if val is True:
                config["epilogue_subtile"] = 2
            elif not val:
                config.pop("epilogue_subtile", None)
            elif val not in EPILOGUE_SUBTILE_EXTENDED_CHOICES:
                raise InvalidConfig(
                    f"epilogue_subtile must be one of {EPILOGUE_SUBTILE_EXTENDED_CHOICES!r}, got {val!r}"
                )
            elif _fix_invalid and not self._should_keep_epilogue_subtile_for_autotune():
                config.pop("epilogue_subtile", None)
            # Epilogue subtiling is incompatible with flatten_loops because
            # FlattenedTileStrategy does not support offset_var needed by
            # the epilogue store codegen path.
            flatten_loops = config.get("flatten_loops")
            if (
                "epilogue_subtile" in config
                and isinstance(flatten_loops, list)
                and any(flatten_loops)
            ):
                if _fix_invalid:
                    config.pop("epilogue_subtile", None)
                else:
                    raise InvalidConfig(
                        "epilogue_subtile is incompatible with flatten_loops=True"
                    )

        # Set default values for grid indices when pid_type is not persistent
        if pid_type in ("flat", "xyz") and self.grid_block_ids:
            for name, mapping in (
                ("range_unroll_factors", self.range_unroll_factors),
                ("range_warp_specializes", self.range_warp_specialize),
                ("range_num_stages", self.range_num_stages),
                ("range_multi_buffers", self.range_multi_buffers),
                ("range_flattens", self.range_flattens),
            ):
                config[name] = mapping._reset_config_to_default(
                    name, config.get(name, ()), block_ids=self.grid_block_ids
                )

        range_warp_specializes = cast(
            "list[bool | None]", config.get("range_warp_specializes", [])
        )

        if range_warp_specializes and any(range_warp_specializes):
            # Only one range_warp_specializes is allowed, take the first one
            # Prefer warp specialize on outermost loop
            first_idx = range_warp_specializes.index(True)
            for i in range(first_idx + 1, len(range_warp_specializes)):
                range_warp_specializes[i] = None

            range_unroll_factors = cast(
                "list[int]", config.get("range_unroll_factors", [])
            )
            if range_unroll_factors and range_unroll_factors[first_idx] > 1:
                if range_unroll_factors[first_idx]:
                    range_unroll_factors[first_idx] = 0

                config["range_unroll_factors"] = range_unroll_factors

        if self.supports_config_key("range_warp_specializes"):
            config["range_warp_specializes"] = range_warp_specializes

        if self.backend_name == "cute":
            preserve_keys = self._cute_tcgen05_config.implicit_default_keys_to_preserve(
                config
            )
            for key in _CUTE_IMPLICIT_DEFAULT_KEYS - provided_keys - preserve_keys:
                config.pop(key, None)

        # Allow tunable parameter keys in addition to backend-supported keys.
        allowed_keys = self.supported_config_keys() | {
            *self.user_defined_tunables.keys()
        }
        if invalid_keys := ({*config} - allowed_keys):
            raise InvalidConfig(f"Invalid config keys {sorted(invalid_keys)!r}")

        # ⭐ LAST, because the fingerprint must key the config AS CODEGEN WILL SEE IT.
        # ``normalize`` mutates in place (it fills every missing slot), so a fingerprint
        # taken at the top -- where ``provided_keys`` is snapshotted -- names a dict that no
        # longer exists by the time anything reads it back.  MEASURED: recorded at the top,
        # the key went from 6 entries to 11 across this method and the lookup missed 100%
        # of the time, silently reverting the whole feature to "never explicit".
        # ``provided_keys`` is still the ANSWER; only the KEY is taken here.
        self._record_cute_row_residency_provenance(config, provided_keys, _fix_invalid)

    def raise_grid_block_minimums(self) -> None:
        """Raise min_size for grid block dimensions based on problem size.

        Very small block sizes produce enormous grids that the autotuner
        wastes time exploring.  This heuristic sets a floor so the total
        number of blocks per dimension stays within a reasonable range
        derived from ``num_compute_units``.

        The raised minimum never exceeds the default block size that
        ``_fragment`` would compute, so memory and shared-memory
        constraints from non-tiled dimensions are respected.
        """
        if not self.grid_block_ids:
            return

        n_cus = num_compute_units()
        n_dims = len(self.grid_block_ids)
        max_blocks_per_dim = math.ceil((n_cus * 64) ** (1.0 / n_dims))

        for grid_bid in self.grid_block_ids:
            try:
                spec = self.block_sizes.block_id_lookup(grid_bid)
            except KeyError:
                continue
            if spec.size_hint <= 0:
                continue
            default = spec._fragment(self).default_val
            min_block = spec.size_hint // max_blocks_per_dim
            min_block = min(min_block, default)
            if min_block >= 2:
                min_block = 1 << (min_block.bit_length() - 1)
                min_block = min(min_block, spec.max_size)
                spec.autotuner_min = assert_integer_power_of_two(
                    max(min_block, spec.autotuner_min)
                )

    def create_config_generation(
        self,
        *,
        overrides: Mapping[str, object] | None = None,
        advanced_controls_files: list[str] | None = None,
        process_group_name: str | None = None,
    ) -> ConfigGeneration:
        from .config_generation import ConfigGeneration

        return ConfigGeneration(
            self,
            overrides=overrides,
            advanced_controls_files=advanced_controls_files,
            process_group_name=process_group_name,
        )

    def flatten_missing_field_default(
        self,
        key: str,
        config: dict[str, object],
    ) -> tuple[bool, object]:
        if self.backend_name == "cute":
            return self._cute_tcgen05_config.flatten_missing_field_default(key, config)
        return False, None

    def prepare_override_normalization(
        self,
        config: dict[str, object],
        overrides: Mapping[str, object],
    ) -> None:
        if self.backend_name == "cute":
            self._cute_tcgen05_config.prepare_override_normalization(
                config,
                overrides,
            )

    def _base_default_config(self) -> helion.Config:
        config = self.flat_config(lambda x: x.default())
        self._shrink_for_numel_constraints(config)
        return config

    def default_config(self) -> helion.Config:
        if self.compiler_default_config is None:
            return self._base_default_config()
        # A promoted seed only specifies the knobs it cares about (e.g. block_sizes); layer it over
        # the full base defaults so every other key — including user register_tunable defaults — is
        # preserved rather than dropped.
        merged = dict(self._base_default_config().config)
        merged.update(self.compiler_default_config.config)
        config = helion.Config.from_dict(merged)
        # Then normalize, so a promoted compiler default has the same canonical field set as the
        # ``_base_default_config`` path: without this its ``repr``/equality differs from its own
        # flatten/unflatten round-trip, which breaks callers that key on the config identity
        # (e.g. benchmark result maps).
        self.normalize(config, _fix_invalid=True)
        self._shrink_for_numel_constraints(config)
        return config

    def _shrink_for_numel_constraints(self, config: helion.Config) -> None:
        """Shrink block_sizes in *config* in-place so every tensor numel
        constraint is satisfied.
        """
        block_sizes = config.config.get("block_sizes")
        if (
            not isinstance(block_sizes, list)
            or not block_sizes
            or not self.tensor_numel_constraints
        ):
            return
        min_sizes = [
            max(self.block_sizes[i].min_size, 1) for i in range(len(block_sizes))
        ]
        shrink_block_sizes_for_numel_constraints(
            self.tensor_numel_constraints, block_sizes, min_sizes
        )

    def _flat_fields(
        self,
    ) -> dict[str, BlockIdSequence[Any] | ConfigSpecFragment]:
        """Return {key: field} for all tunable fields in flat_config() order.

        This is the single source of truth for field ordering.
        """
        fields: dict[str, BlockIdSequence[Any] | ConfigSpecFragment] = {
            "block_sizes": self.block_sizes,
        }
        if self.backend_name == "cute":
            if self.cute_tcgen05_search_enabled:
                fields.update(self._cute_tcgen05_config.flat_fields())
            elif self.cute_flash_search_enabled:
                from .._compiler.cute.cute_flash import flash_autotune_fragments

                assert self._cute_flash_head_dim is not None
                assert self._cute_flash_num_kv is not None
                fields.update(
                    flash_autotune_fragments(
                        self._cute_flash_head_dim,
                        self._cute_flash_num_kv,
                        dtype=self._cute_flash_dtype,
                        is_causal=self._cute_flash_is_causal,
                        has_kv_tile_pruning=self._cute_flash_has_kv_tile_pruning,
                        requires_ws_overlap=self._cute_flash_requires_ws_overlap,
                        small_biased_candidate=(
                            self._cute_flash_small_biased_candidate
                        ),
                    )
                )
            elif self.supports_config_key("num_threads"):
                fields["num_threads"] = self.num_threads
                # Universal pid emission honors ``loop_orders`` and the
                # better order is shape-dependent. tcgen05 exposes the same
                # field from CuteTcgen05Config.flat_fields().
                if (
                    self.supports_config_key("loop_orders")
                    and len(self.loop_orders) > 0
                ):
                    fields["loop_orders"] = self.loop_orders
                # Expose ``cute_vector_widths`` per-block so the
                # autotuner can vary V in {1, 2, 4, 8} for lane-loop
                # vec loads (and for ``LoopedReductionStrategy`` rolled
                # reductions).  Without this entry, ``flatten`` strips V
                # back to the default of 1, defeating the seed
                # heuristics that try to bias toward LDG.128 lattices.
                if (
                    self.supports_config_key("cute_vector_widths")
                    and len(self.cute_vector_widths) > 0
                ):
                    fields["cute_vector_widths"] = self.cute_vector_widths
                # The CuTe reduction knobs: the TV-layout ones (per reduction
                # block, now including ``cute_row_residency``) and two AST-pass
                # ones (per device loop).
                #
                # ⭐ ONE BLOCK, NOT TWO.  These used to be registered separately
                # because the TV-layout three were routed through
                # ``_tv_search_choices`` -- a helper that pinned their *search* to
                # their own default while codegen did not yet read them, so varying
                # them would only have spent autotuner population on identical
                # kernels.  All five are read by codegen now (see the note above
                # ``CuteThreadsPerRowSpec``), so every one of them reaches a
                # genuinely different kernel and the search must be able to vary all
                # five.  With the distinction gone the two loops were textually the
                # same loop, and keeping them apart would have preserved a
                # difference that no longer exists.
                # ⛔ ``cute_reduction_reload`` IS DELIBERATELY ABSENT (task 1).  It is no
                # longer a searchable axis: ``cute_row_residency`` below is the one key
                # carrying that decision, and offering both would make the autotuner move
                # TWO keys on TWO block-id domains to walk ONE one-dimensional choice --
                # complaint 3 in ``CuteRowResidencySpec``'s docstring, and the reason the
                # axis was introduced.  The old key survives only as an accepted INPUT
                # spelling that ``_normalize_cute_row_residency`` folds in and strips, so
                # a search that could set it would be setting a key that no longer exists
                # by the time codegen runs.
                fields.update(
                    {
                        cute_key: cute_seq
                        for cute_key, cute_seq in (
                            ("cute_threads_per_row", self.cute_threads_per_row),
                            ("cute_row_residency", self.cute_row_residency),
                            ("cute_ndtile_tv", self.cute_ndtile_tv),
                            ("cute_online_defer", self.cute_online_defer),
                        )
                        if self.supports_config_key(cute_key) and len(cute_seq) > 0
                    }
                )
            if (
                not self.cute_flash_search_enabled
                and self.epilogue_subtile_autotune_choices is not None
            ):
                fields["epilogue_subtile"] = EnumFragment(
                    choices=self.epilogue_subtile_autotune_choices
                )
            fields.update(self.user_defined_tunables)
            return fields

        # Only add sequence keys that the backend supports
        fields.update(
            {
                name: seq
                for name, seq in [
                    ("loop_orders", self.loop_orders),
                    ("flatten_loops", self.flatten_loops),
                    ("l2_groupings", self.l2_groupings),
                    ("reduction_loops", self.reduction_loops),
                    ("range_unroll_factors", self.range_unroll_factors),
                    ("range_warp_specializes", self.range_warp_specialize),
                    ("range_num_stages", self.range_num_stages),
                    ("range_multi_buffers", self.range_multi_buffers),
                    ("range_flattens", self.range_flattens),
                    ("static_ranges", self.static_ranges),
                ]
                if self.supports_config_key(name)
            }
        )

        # Scalar fields (ConfigSpecFragment)
        is_tileir = self.backend_name == "tileir"
        if is_tileir:
            # TileIR: num_warps is unused (fixed at 4), num_stages has wider range
            num_warps_fragment: ConfigSpecFragment = NumWarpsFragment(4, 4)
        elif supports_amd_cdna_tunables():
            num_warps_fragment = NumWarpsFragment(1, 16, DEFAULT_NUM_WARPS)
        else:
            num_warps_fragment = NumWarpsFragment(1, 32, DEFAULT_NUM_WARPS)
        num_stages_fragment = self._num_stages_fragment()

        if self.supports_config_key("num_warps"):
            fields["num_warps"] = num_warps_fragment
        if self.supports_config_key("num_stages"):
            fields["num_stages"] = num_stages_fragment
        if self.supports_config_key("indexing"):
            fields["indexing"] = self.indexing
        if self.supports_config_key("atomic_indexing"):
            fields["atomic_indexing"] = self.atomic_indexing
        if (
            self.supports_config_key("pallas_load_buffer_count")
            and self.has_pallas_inner_loops
            and self.pallas_load_buffer_count.length > 0
        ):
            fields["pallas_load_buffer_count"] = self.pallas_load_buffer_count
        if self.supports_config_key("pid_type"):
            fields["pid_type"] = EnumFragment(self.allowed_pid_types)
        if self.supports_config_key("xcd_remap") and self.num_xcd > 1:
            fields["xcd_remap"] = BooleanFragment()
        if self.supports_config_key("num_sm_multiplier"):
            fields["num_sm_multiplier"] = PowerOfTwoFragment(
                MIN_NUM_SM_MULTIPLIER,
                self.max_num_sm_multiplier,
                DEFAULT_NUM_SM_MULTIPLIER,
            )
        if self.supports_config_key("load_eviction_policies"):
            fields["load_eviction_policies"] = self.load_eviction_policies
        if (
            self.supports_config_key("load_cache_modifiers")
            and self.load_cache_modifiers.length > 0
        ):
            fields["load_cache_modifiers"] = self.load_cache_modifiers
        if (
            self.supports_config_key("store_cache_modifiers")
            and self.store_cache_modifiers.length > 0
        ):
            fields["store_cache_modifiers"] = self.store_cache_modifiers
        if self.supports_config_key("num_threads"):
            fields["num_threads"] = self.num_threads
        if is_tileir:
            fields["num_ctas"] = self.backend_tunable_fragments["num_ctas"]
            fields["occupancy"] = self.backend_tunable_fragments["occupancy"]
        else:
            fields.update(self.backend_tunable_fragments)
        if self.has_pallas_inner_loops:
            choices = AUTOTUNED_PALLAS_LOOP_TYPES
            if self.has_symbolic_or_data_dependent_bounds:
                # Exclude "unroll" (uses Python range(), can't handle traced
                # bounds) and put "fori_loop" first: it handles both DMA-aligned
                # and unaligned inner blocks, while "emit_pipeline" fails on
                # unaligned dims.
                # TODO(thcmbs): Also exclude "emit_pipeline" when has_pallas_dma_unaligned
                # is set, to avoid wasted autotuning effort. See PR #1969 review discussion.
                choices = ("fori_loop", "emit_pipeline")
                if self.grid_block_ids:
                    # Owner hl.grid + jagged bounds may be compactable. The full
                    # detector remains authoritative, so residual mismatches are
                    # autotuner-skippable InvalidConfig candidates.
                    choices = (*choices, "unroll")
                    fields["pallas_worklist_grouping"] = EnumFragment(
                        choices=VALID_PALLAS_WORKLIST_GROUPINGS
                    )
            fields["pallas_loop_type"] = EnumFragment(choices=choices)
            if self.supports_config_key("pallas_pre_broadcast"):
                fields["pallas_pre_broadcast"] = BooleanFragment()
        # Only include maxnreg on CUDA devices (not supported on AMD and Intel GPU)
        if self.supports_config_key("maxnreg") and supports_maxnreg():
            fields["maxnreg"] = EnumFragment(VALID_MAXNREG)
        if self.epilogue_subtile_autotune_choices is not None:
            fields["epilogue_subtile"] = EnumFragment(
                choices=self.epilogue_subtile_autotune_choices
            )
        # Add tunable parameters
        fields.update(self.user_defined_tunables)
        return fields

    def structural_fingerprint(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> tuple[tuple[str | int, ...], ...]:
        """Return a hashable structural description of this ConfigSpec's search space.

        Captures field names, sequence lengths, per-item block_ids lengths
        (for PermutationFragment), ListOf inner lengths, and optional ACF slot
        presence.  Two ConfigSpecs with the same fingerprint can safely exchange
        FlatConfig values.
        """
        result: list[tuple[str | int, ...]] = [
            (key, *field.fingerprint()) for key, field in self._flat_fields().items()
        ]
        acf_fragment = self._advanced_controls_file_fragment(advanced_controls_files)
        if acf_fragment is not None:
            result.append(
                (
                    "advanced_controls_file",
                    *cast("tuple[str, ...]", acf_fragment.choices),
                )
            )
        return tuple(result)

    def structural_fingerprint_hash(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> str:
        """Return a hex-digest SHA-256 hash of the structural fingerprint."""
        return hashlib.sha256(
            repr(
                self.structural_fingerprint(
                    advanced_controls_files=advanced_controls_files
                )
            ).encode("utf-8")
        ).hexdigest()

    def _advanced_controls_file_fragment(
        self, advanced_controls_files: list[str] | None
    ) -> EnumFragment | None:
        # Empty list means no autotuning with ACFs.
        if not advanced_controls_files:
            return None
        files = advanced_controls_files
        # When non-empty list is provided then ensure default -O3 is considered.
        if "" not in files:
            files = [*files, ""]
        return EnumFragment(tuple(files))

    def flat_key_layout(
        self, *, advanced_controls_files: list[str] | None = None
    ) -> list[tuple[str, int, bool]]:
        """Return (key_name, num_flat_entries, is_sequence) for each field.

        is_sequence is True for BlockIdSequence keys whose list values
        are spread across individual flat slots.
        """
        result = [
            (key, *field._flat_key_info()) for key, field in self._flat_fields().items()
        ]
        if self._advanced_controls_file_fragment(advanced_controls_files) is not None:
            result.append(("advanced_controls_file", 1, False))
        return result

    def flat_config(
        self,
        fn: Callable[[ConfigSpecFragment], object],
        *,
        advanced_controls_files: list[str] | None = None,
    ) -> helion.Config:
        """Map a flattened version of the config using the given function."""
        config: dict[str, Any] = {}
        for key, field in self._flat_fields().items():
            config[key] = field._flat_config(self, fn)

        for name in (
            "loop_orders",
            "num_threads",
            "flatten_loops",
            "reduction_loops",
            "l2_groupings",
            "range_unroll_factors",
            "range_warp_specializes",
            "range_num_stages",
            "range_multi_buffers",
            "range_flattens",
            "static_ranges",
            "load_eviction_policies",
            "load_cache_modifiers",
            "store_cache_modifiers",
            "indexing",
            "atomic_indexing",
            "pallas_load_buffer_count",
        ):
            if not config.get(name):
                config.pop(name, None)
        acf_fragment = self._advanced_controls_file_fragment(advanced_controls_files)
        if acf_fragment is not None:
            config["advanced_controls_file"] = fn(acf_fragment)
        self.normalize(config, _fix_invalid=True)
        out = helion.Config(**config)
        # ⭐ EVERY CONFIG BUILT HERE IS MACHINE-SYNTHESISED, AND MUST STAY THAT WAY WHEN IT
        # IS RE-NORMALIZED LATER.  This is the ONE construction site for both
        # ``default_config()`` (via ``_base_default_config``) and the autotuner's
        # ``ConfigGeneration.unflatten``, and it writes EVERY key from the fragments -- so
        # the result "names" ``cute_row_residency`` even though no human did.
        #
        # ⛔ MEASURED: without this, ``HELION_AUTOTUNE_EFFORT=none`` (which routes through
        # ``BoundKernel._user_provided_config`` -> ``default_config()`` -> ``to_code``,
        # i.e. a SECOND ``normalize`` on the USER path with ``_fix_invalid=False``) made
        # two phase-A gate tests raise ``CuteRowResidencyUnavailable`` --
        # ``class5_int32_no_wrap`` and ``class7a_store_between_reductions_survives``.  The
        # ``_fix_invalid`` flag alone cannot see it: on that second pass it is False, and
        # correctly so -- the caller really did hand in a fully-populated config.
        #
        # ⚠⚠ FINGERPRINTED BY CONTENT, NOT BY ``id()``, AND THAT IS FORCED.
        # ``BoundKernel.to_code`` makes a DEFENSIVE COPY before it normalizes
        # (``kernel.py`` ~:820, ``config = Config(**config.config)``, with its own comment
        # explaining why), so the object codegen sees is NOT the object minted here and an
        # identity key misses every time.  A content fingerprint survives that copy, and
        # also survives ``flatten -> unflatten``.
        #
        # ⭐ AND THE COLLISION IS BENIGN, WHICH IS WHY CONTENT IS SOUND HERE: two configs
        # with the same fingerprint have the same keys and the same values, so a
        # hand-written config that is byte-for-byte a machine default is treated as
        # synthesised.  That is the RIGHT answer -- it names exactly what the ladder would
        # have chosen, so "honour it or decline" is the same question the ladder faces, and
        # the ladder is explicitly allowed to decline (it cannot predict the decline).
        self._cute_row_residency_synthesised.add(_config_fingerprint(config))
        if len(self._cute_row_residency_synthesised) > 8192:
            # Bounded.  ⚠ Dropping an entry degrades toward RAISING, which is the unsafe
            # direction, so the cap is set far above any real search population (a
            # generation is O(100) configs) and clearing is a last resort rather than a
            # routine eviction.
            self._cute_row_residency_synthesised.clear()
            self._cute_row_residency_synthesised.add(_config_fingerprint(config))
        return out


class LoopOrderSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> PermutationFragment:
        return PermutationFragment(len(self.block_ids))

    def _normalize(self, name: str, value: object) -> list[int]:
        if type(value) is not list:
            if not isinstance(value, tuple):
                raise InvalidConfig(f"{name} must be a list, got {value!r}")
            value = [*value]
        length = len(self.block_ids)
        if len(value) != length:
            raise InvalidConfig(f"{name} must be length {length}, got {len(value)}")
        if {*value} != {*range(length)}:
            raise InvalidConfig(f"{name} must be permutation, got {value!r}")
        return value

    def _fill_missing(self) -> list[int]:
        """Provide a value when not provided by the user."""
        return [*range(len(self.block_ids))]


class L2GroupingSpec(_PowerOfTwoBlockIdItem):
    def _fragment(self, base: ConfigSpec) -> PowerOfTwoFragment:
        return PowerOfTwoFragment(1, 64, 1)

    def _fill_missing(self) -> int:
        return 1


class BlockSizeSpec(_PowerOfTwoBlockIdItem):
    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
        min_size: int = 1,
        max_size: int | None = None,
        bounded_by_block_id: int | None = None,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

        # TODO(shunting): it's a bit conservative since not every block is split
        # for different ranks.
        bounded_hint = size_hint
        if dist.is_initialized():
            world_size = dist.get_world_size()
            bounded_hint = bounded_hint // world_size

        bounded_hint = max(bounded_hint, 1)
        self.min_size: int = min_size
        self.autotuner_min: int = min_size
        # Largest power-of-two block that fits inside the dimension. allow_overshoot
        # may raise max_size above this for matmul dims, but the default block size
        # stays clamped to dim_max_size (see _fragment).
        self.dim_max_size: int = (
            next_power_of_2(bounded_hint) if max_size is None else max_size
        )
        self.max_size: int = self.dim_max_size
        # Outer block_id whose tile extent caps this block's size in normalize().
        self.bounded_by_block_id: int | None = bounded_by_block_id
        if self.max_size < self.min_size:
            self.max_size = self.min_size
        assert self.min_size <= self.max_size

    def __repr__(self) -> str:
        fields: list[str] = []
        for field, default in (
            ("block_id", None),
            ("size_hint", None),
            ("min_size", 1),
            ("max_size", next_power_of_2(self.size_hint)),
            ("bounded_by_block_id", None),
        ):
            value = getattr(self, field)
            if value != default:
                fields.append(f"{field}={value!r}")
        return f"BlockSizeSpec({', '.join(fields)})"

    def _normalize(self, name: str, value: object) -> int | None:
        result = super()._normalize(name, value)
        if isinstance(result, int) and result < self.min_size:
            result = self.min_size
        return result

    def update_min(self, value: int) -> None:
        self.min_size = assert_integer_power_of_two(max(value, self.min_size))
        if self.max_size < self.min_size:
            self.max_size = self.min_size

    def update_max(self, value: int) -> None:
        clamped = max(value, 1)
        self.max_size = assert_integer_power_of_two(min(clamped, self.max_size))

    def allow_overshoot(self, ceiling: int) -> None:
        """Raise the autotuner search ceiling above the dimension size.

        Used for matmul tile dimensions: a block larger than a small dimension
        (with the extra rows/cols masked off) can map to a more efficient MMA
        tile and run faster. Only the search ceiling grows; the default block
        size stays clamped to the dimension (see _fragment). Dimensions bounded
        by an outer tile extent are left untouched.
        """
        if self.bounded_by_block_id is not None:
            return
        self.max_size = max(self.max_size, next_power_of_2(max(ceiling, 1)))

    def update_hint(self, value: int) -> None:
        self.size_hint = value
        self.update_max(next_power_of_2(max(value, 1)))

    def _fragment(self, base: ConfigSpec) -> BlockSizeFragment:
        total_ndim = len(base.block_sizes)
        reduction_numel = _product(
            [next_power_of_2(spec.size_hint) for spec in base.reduction_loops]
        )
        if total_ndim <= 2 and reduction_numel <= 128:
            default = 32
        elif total_ndim >= 3 and reduction_numel > 1:
            # With 3+ tiled dimensions and a non-trivial reduction/full-slice
            # dimension, the total tensor numel (default^total_ndim *
            # reduction_numel) grows quickly and can cause Triton JIT
            # compilation to hang or exceed shared memory limits.
            # Compute a default that keeps total numel <= 32768 (safe for
            # 64KB shared memory with 2-byte elements like bf16).
            target = 32768
            per_dim = int((target / reduction_numel) ** (1.0 / total_ndim))
            default = max(1, 1 << (per_dim.bit_length() - 1)) if per_dim >= 1 else 1
        elif reduction_numel <= 256:
            default = 16
        else:
            default = 1
        low = min(max(self.min_size, self.autotuner_min), self.max_size)
        # Clamp the default within the dimension so allow_overshoot only widens
        # the autotuner *search*, never the default (non-autotuned) block size.
        # Needed for matmul dims smaller than the heuristic default (e.g. M<16),
        # where the default would otherwise overshoot to a masked tile.
        default = min(default, self.dim_max_size)
        return BlockSizeFragment(
            low,
            self.max_size,
            default,
        )


class NumThreadsSpec(_PowerOfTwoBlockIdItem):
    def __init__(self, *, block_id: int, size_hint: int) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _normalize(self, name: str, value: object) -> int | None:
        # 0 is a valid sentinel meaning "use block_size as thread count"
        if value == 0:
            return 0
        return super()._normalize(name, value)

    def _fragment(self, base: ConfigSpec) -> NumThreadsFragment:
        max_threads = min(max(self.size_hint, 1), 1024)
        default = next_power_of_2(max_threads)
        return NumThreadsFragment(default)

    def _fill_missing(self) -> int:
        return 0


class FlattenLoopSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> BooleanFragment:
        return BooleanFragment()

    def _normalize(self, name: str, value: object) -> bool:
        if not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}") from None
        return value

    def _fill_missing(self) -> bool:
        return False


class ReductionLoopSpec(_PowerOfTwoBlockIdItem):
    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _flat_fragment(self, base: ConfigSpec) -> BlockSizeFragment:
        # Shared by both directions:
        # - unflatten: flat integer -> Config value via _flat_config()
        # - flatten: Config value -> flat integer via _encode_flat_value()
        low = 8  # TODO(jansel): is smaller needed?
        high = next_power_of_2(max(low, self.size_hint))
        default = min(high, 4096)
        # Cap default at the backend's max reduction loop so that
        # large reductions default to looped rather than persistent.
        if base.max_reduction_loop is not None:
            force_threshold = base.reduction_loop_force_threshold
            if force_threshold is not None and self.size_hint > force_threshold:
                default = min(default, base.max_reduction_loop)
        return BlockSizeFragment(low, high, default)

    def _flat_config(
        self, base: ConfigSpec, fn: Callable[[ConfigSpecFragment], object]
    ) -> int | None:
        fragment = self._flat_fragment(base)
        low = fragment.low
        high = fragment.high
        value = fn(fragment)
        assert isinstance(value, int)
        if not (low <= value <= high):
            raise InvalidConfig(
                f"Invalid value for reduction loop {low} <= {value} <= {high}"
            )
        if value >= self.size_hint:
            return None  # max size becomes persistent reduction
        return value

    def _encode_flat_value(self, base: ConfigSpec, value: object) -> object:
        # Encode None ("persistent reduction") so the inverse ``_flat_config``
        # decodes it back to None. ``_flat_config`` returns None for any value
        # >= size_hint, so the encoding must also be >= size_hint: use the
        # fragment's ``high`` (always >= size_hint). The fragment *default* is
        # capped at max_reduction_loop and can fall below size_hint, which would
        # round-trip None into a slow looped config (e.g. size_hint=32000 ->
        # default 4096 -> reduction_loops=[4096]).
        if value is None:
            return self._flat_fragment(base).high
        return value

    def _normalize(self, name: str, value: object) -> int | None:
        if value is None:
            return None
        normalized = super()._normalize(name, value)
        # A looped chunk of 1 is degenerate: "hold the whole axis" is encoded as
        # ``None`` (persistent), not 1, and ``LoopedReductionStrategy`` rejects a
        # block size <= 1.  The autotuner search never proposes < 8 (its fragment
        # ``low`` is 8), but the reduction seed's byte budget can collapse the chunk
        # to 1 on a reduction co-resident with a wide feature.  Floor a stray 1 up to
        # that same search floor of 8; for a small extent the ``>= size_hint`` rule
        # below then collapses it to persistent ``None``, and 1 is the only
        # power-of-two chunk that can hit this (so legal chunks 2, 4, 8, ... are
        # left byte-identical).
        if isinstance(normalized, int) and normalized < 2:
            normalized = 8
        # A looped reduction whose chunk equals or exceeds the reduction
        # extent has only one iteration — it is semantically identical to a
        # persistent reduction, but the looped codegen path occasionally
        # produces subtly different results on the CuTe backend (e.g. when a
        # multi-pass kernel like layer_norm reuses the loaded inputs across
        # two reductions).  Collapsing to ``None`` here matches the
        # ``_flat_config`` behaviour and keeps the persistent/loop choice in
        # sync regardless of how the value was generated.
        if isinstance(normalized, int) and normalized >= self.size_hint:
            return None
        return normalized

    def _fill_missing(self) -> None:
        return None


_CUTE_VECTOR_WIDTH_CHOICES: tuple[int, ...] = (1, 2, 4, 8)


class CuteVectorWidthSpec(_BlockIdItem):
    """Per-reduction-block vector load width for the CuTe backend.

    V=1 disables vectorization (scalar loads). V=2/4/8 emits
    ``cute.arch.load(..., ir.VectorType.get([V], elem_dtype.mlir_type))``
    for the inner reduction load, lowering to LDG.64/LDG.128.
    """

    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        return EnumFragment(choices=_CUTE_VECTOR_WIDTH_CHOICES)

    def _normalize(self, name: str, value: object) -> int:
        if not isinstance(value, int):
            raise InvalidConfig(f"{name} must be an integer, got {value!r}")
        if value not in _CUTE_VECTOR_WIDTH_CHOICES:
            raise InvalidConfig(
                f"{name} must be one of {_CUTE_VECTOR_WIDTH_CHOICES}, got {value!r}"
            )
        return value

    def _fill_missing(self) -> int:
        return 1


# ⭐ ``_tv_search_choices`` USED TO LIVE HERE, together with the module flag
# ``tv_layout.TV_LAYOUT_KNOBS_SEARCHABLE`` that drove it.  Both are DELETED, and the
# deletion is the finding.
#
# The flag existed to pin TV-layout knobs' *search* to their own defaults
# while codegen did not yet read them: varying an unread knob only spends autotuner
# population on identical kernels.  It was then flipped to True for
# ``cute_reduction_reload`` while its comment continued to say that
# ``cute_threads_per_row`` was still unread by codegen.
#
# The remaining knobs are read by codegen today:
#
#   cute_threads_per_row  -> ``LoopedReductionStrategy.__init__``  (lowers the row's
#                            thread count when a wider copy is on the table)
#   cute_reduction_reload -> ``LoopedReductionStrategy._cute_reload_from_config``
#
# With the flag True, ``_tv_search_choices`` returned ``None`` unconditionally, which is
# ``EnumFragment``'s own default for ``search_choices`` — so it was a function that
# could only ever return the default, guarded by a flag with no reachable second value.
# Deleting both leaves the three fragments plain ``EnumFragment(choices=...)``, which is
# what the two knobs landed in ``fa9358c27`` already do, so the five TV/pass knobs now
# have ONE shape between them instead of two.


class CuteThreadsPerRowSpec(_PowerOfTwoBlockIdItem):
    """Per-reduction-block count of threads cooperating on one row (CuTe).

    This is the inner mode of the TV layout's thread layout
    (``make_ordered_layout((num_threads // threads_per_row, threads_per_row),
    order=(1, 0))``), i.e. quack's ``threads_per_row``
    (``quack/reduction_base.py:44``).  The CTA size is *derived* as
    ``rows_per_cta * threads_per_row`` rather than configured separately, so
    illegal (num_threads, threads_per_row) pairs are unrepresentable.

    The default is quack's threshold ladder as a pure function of the reduction
    extent (``quack/rmsnorm_config.py:52-56``), which makes ``default_config()``
    reproduce quack's verified layout table.
    """

    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _default(self) -> int:
        return threads_per_row_for(self.size_hint)

    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        default = self._default()
        # The fragment's first choice is its default (``EnumFragment.default``),
        # so rotate the ladder's answer to the front while keeping the full
        # menu representable.
        choices = (default, *(c for c in THREADS_PER_ROW_CHOICES if c != default))
        return EnumFragment(choices=choices)

    def _normalize(self, name: str, value: object) -> int:
        normalized = super()._normalize(name, value)
        if normalized not in THREADS_PER_ROW_CHOICES:
            raise InvalidConfig(
                f"{name} must be one of {THREADS_PER_ROW_CHOICES}, got {value!r}"
            )
        return normalized

    def _fill_missing(self) -> int:
        return self._default()


# ⛔ ``CuteReductionReloadSpec`` WAS DELETED HERE (task 1, FIXLIST item 1).
#
# It was a per-reduction-block enum over ``(None, "smem")`` naming where the second read
# of the reduced row came from.  ``CuteRowResidencySpec`` below is a strict SUPERSET of it
# -- ``("registers", "smem", "gmem")`` -- and the old key's ``None`` meant *registers* or
# *gmem* depending on a SECOND knob's budget, i.e. one enum value standing for two
# kernels.  Keeping both let the old encoding go on expressing the ambiguity the new axis
# exists to remove, and gave the autotuner two keys on two block-id domains to walk one
# one-dimensional choice.
#
# ``cute_reduction_reload`` survives ONLY as an accepted INPUT spelling:
# ``ConfigSpec._normalize_cute_row_residency`` translates it through
# ``tv_layout.row_residency_from_legacy`` and then STRIPS it, so no config that reaches
# codegen carries it and nothing downstream reads the name.  That is FIXLIST item 1's
# stated fix -- "a one-shot config migration rather than a live translation".
#
# ⚠ THE VALUE VALIDATION MOVED WITH IT, deliberately and not incidentally: this class's
# ``_normalize`` was the only thing rejecting an illegal value, and
# ``row_residency_from_legacy`` tests ``== "smem"``, so without a replacement
# ``cute_reduction_reload=["banana"]`` would have been silently COERCED to ``registers``.
# The check now lives at the translation site with the same message.
#
# ⚠ AND THE 40 FROZEN CELLS WERE MIGRATED FIRST, which is what made this deletion inert:
# ``_notes_codereview/migrate_frozen_task1.py`` rewrote all 25 cells that named the old key
# into ``cute_row_residency``, PROVEN byte-identical on 40/40 before any code moved.
# Deleting the codegen-side fallback first would have silently re-read those 25 as ``gmem``.


class CuteRowResidencySpec(_BlockIdItem):
    """⭐ Per-reduction-block: WHERE THE SECOND READ OF THE ROW COMES FROM (CuTe).

    ONE three-valued axis over ``("registers", "smem", "gmem")``
    (``cute/tv_layout.py::ROW_RESIDENCY_CHOICES``), mirroring quack's
    ``reload_from_vals = (None, "smem", "gmem")`` (``quack/rmsnorm_config.py``).

        ``registers``  the row's fragment lanes are cached across the sweeps, so
                       sweep 2 reads REGISTERS (``cute/fuse_tv_copy_sweeps.py``);
        ``smem``       sweep 1 publishes the row into a per-CTA staged tile and
                       sweep 2 reads SMEM (``cute/memory_ops.py``);
        ``gmem``       no cache at all -- sweep 2 issues a second
                       ``local_tile`` + ``cute.copy`` and re-reads GLOBAL.  This is
                       the BASELINE any residency claim must be measured against.

    ⭐ WHY IT IS ONE KEY AND NOT THE TWO IT SUPERSEDES.  Before this axis the choice
    was split across ``cute_reduction_reload`` (an enum, per reduction block) and
    ``cute_tv_sweep_cache`` (an int slot budget, per DEVICE LOOP), and the third
    value had no name -- ``gmem`` was reachable only as the CONJUNCTION
    ``cute_reduction_reload=None AND cute_tv_sweep_cache=0``.  Three consequences,
    all measured:

    1. ``cute_reduction_reload=None`` meant *registers* when the budget was positive
       and *gmem* when it was 0.  The key that reads like the residency selector was
       not one.
    2. "Exactly one mechanism is in effect" was not representable, so nothing
       enforced it -- and MEASURED, ``reload="smem"`` with ``cache=128`` emits the
       SMEM signature while the register cache never fires.  The 2x2 grid had only
       three reachable kernels, i.e. it was a three-valued axis wearing two keys.
    3. The autotuner had to move TWO keys on TWO block-id domains in concert to walk
       a one-dimensional choice, and could not name one third of it at all.

    ⭐⭐ AND IT IS NOW THE **ONLY** THING THAT DECIDES (task 1).  The two budget knobs
    that could overrule it -- ``cute_stage_smem_kb`` (a per-kernel SMEM ceiling) and
    ``cute_tv_sweep_cache`` (a per-thread register-slot ceiling) -- are DELETED.  They
    were *performance policy*, and a performance policy that silently changes which
    memory a row lives in makes every recorded config unverifiable: MEASURED on the
    frozen table, 13 of 40 cells named a residency the kernel did not use.

    ⇒ the only refusals left are the two a hardware compiler is entitled to:
      * **CAPACITY** -- the staged tile exceeds the device's shared memory, or the row's
        register footprint exceeds what a thread can hold;
      * **GEOMETRY** -- there is no usable row axis, a third thread axis is in play, or
        no TV plan was built at all.
    Both are honest reasons to refuse a named residency.  An *explicit* request that hits
    one of them RAISES (``exc.CuteRowResidencyUnavailable``); a request the per-shape
    ladder supplied still declines silently, because the ladder cannot predict the
    decline -- see ``cute/memory_ops.py::cute_emit_row_residency_marker``.

    ⚠ PER REDUCTION BLOCK, and the domain is not a detail.  Residency describes a
    ROW, which only exists where a reduction block does -- so this registers in
    ``DeviceIR._register_cute_tv_layout_slots`` rather than in the per-DEVICE-LOOP group
    beside ``cute_online_defer``.  MEASURED on a kernel with two reduction blocks in one
    device loop, the two domains genuinely differ: the per-block keys have length 2 there
    and the per-loop keys length 1.  A per-device-loop registration would collapse the two
    reductions onto one residency.
    """

    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
        ladder_default: bool = True,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint
        # ⛔⛔ ``ladder_default=False`` MEANS "DEFAULT TO ``gmem``", AND IT IS A CORRECTNESS
        # REQUIREMENT FOR ANY SLOT ON A PATH THAT CANNOT STAGE.
        #
        # ``row_residency_for`` is quack's per-extent ladder: ``smem`` at ``size_hint >= 4096``,
        # else ``registers`` -- it NEVER returns ``gmem``.  That is right for a reduction row,
        # where staging between the two sweeps is the whole point.  It is WRONG for a slot on a
        # path with no between-passes gap, because ``_fill_missing`` then requests staging on
        # every kernel that names no residency key at all.
        #
        # MEASURED when the ND-tile slot was first registered with the ladder default: a
        # DIFFERENT kernel on 48/48 (shape x block-size x dtype) cells with no residency key
        # anywhere -- a dead WRITE-ONLY SMEM buffer up to 128 KiB at n >= 4096 (both consumer
        # reads still read the rmem fragment), or at n <= 2048 a ``registers`` path that LOSES
        # the vectorised fold (``_helion_vfold_acc`` 6 -> 0).  Numerics stayed bit-exact, so
        # nothing failed; it was pure dead work, and invisible to 1051 tests.
        #
        # ⇒ a newly registered slot MUST default to the identity.  ``gmem`` -- "do not park the
        # row, re-read it" -- is that identity: it is the one value nothing can refuse.
        self.ladder_default = ladder_default

    def _default(self) -> str:
        if not self.ladder_default:
            return ROW_RESIDENCY_GMEM
        return row_residency_for(self.size_hint)

    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        default = self._default()
        # The fragment's first choice IS its default (``EnumFragment.default``), so
        # rotate the ladder's answer to the front while keeping all three values
        # representable.  Same shape as the four ladders beside it -- and the reason
        # ``ROW_RESIDENCY_CHOICES`` must never be handed to a fragment unrotated.
        choices = (default, *(c for c in ROW_RESIDENCY_CHOICES if c != default))
        return EnumFragment(choices=choices)

    def _normalize(self, name: str, value: object) -> str:
        # Membership, not a coercion.  A typo MUST raise and MUST name the offending
        # value: silently mapping an unknown string onto some residency would make
        # every recorded config unverifiable, which is the exact failure this axis
        # exists to remove.  ``ROW_RESIDENCY_CHOICES`` is the legal set AND the
        # search menu here -- unlike the two byte/slot budgets beside it, a residency
        # has no off-menu values because it names a code path, not a quantity.
        if value not in ROW_RESIDENCY_CHOICES:
            raise InvalidConfig(
                f"{name} must be one of {ROW_RESIDENCY_CHOICES}, got {value!r}"
            )
        return cast("str", value)

    def _fill_missing(self) -> str:
        # ⭐ "USE THE LADDER" -- and here that is the only implementable answer, not
        # merely the conservative one.
        #
        # ``row_residency_for`` reproduces what the two old ladders jointly produce
        # (``reload_from_for(n)`` composed with ``tv_sweep_cache_for(n)``), so a
        # config that names no residency -- every one of the 40 frozen cells --
        # compiles to the byte-identical kernel it did before this key existed.  That
        # is what makes promoting the two knobs to one axis a REACHABILITY change.
        #
        # ⚠ AND IT IS THE *REQUESTED* LADDER, NOT "today's effective choice".  The
        # latter is impossible HERE, not just inconvenient: ``_fill_missing(self)``
        # takes no arguments and this spec holds only ``block_ids`` and
        # ``size_hint``, while the effective residency depends on
        # ``ChunkTVPlan.lane_extent``, ``_loop_block_size``,
        # ``thread_block_dims()``, the emitted ``cluster_n`` and the running
        # whole-kernel SMEM charge -- none of which exist at normalize time (the
        # strategies are not even constructed yet).  A ladder that guessed the
        # decline would be wrong in both directions and would move the frozen table.
        # The DECLINE is reported where it happens, on the emitted artifact.
        return self._default()


class CuteOnlineDeferSpec(_BlockIdItem):
    """Per-device-loop: WHERE an online ``(max, sum-of-exp)`` recurrence combines
    across lanes (CuTe).

    ``True`` = after the loop, one cross-lane merge per row; ``False`` = inside the
    loop, ``2 * N/(nt*V)`` merges per row with each one a serial dependency between
    consecutive iterations.  Consumed by
    ``helion/_compiler/cute/defer_online_merge.py``.

    ⭐ WHY THIS IS A KNOB AND NOT A FIXED REWRITE.  The deferral is not free: it buys
    ``2 * N/(nt*V) - 1`` cross-lane merges at the price of ONE extra ``exp`` per
    thread (the final rescale onto the merged max).  ``examples/cross_entropy.py``
    states the consequence directly -- "whether that trade wins is a property of the
    backend, not of the algorithm: on a machine where the special-function pipe is
    the limiter the extra ``exp`` can cost more than the saved memory pass".  A
    decision with that shape belongs in the config, where the autotuner can settle
    it per shape, not in a module-level ``if``.

    It also PARTICIPATES in the search rather than sitting above it: enabling the
    deferral changed which ``num_threads`` wins (LEDGER E047 retired), because the
    cost of a cross-lane merge is what made a wide CTA expensive.  So the two knobs
    interact and must be searched jointly.

    ⚠ WHY THE DOMAIN IS AN ``EnumFragment`` OVER ``(True, False)`` AND NOT A
    ``BooleanFragment``.  ``BooleanFragment.default()`` is a hardcoded ``False``, so
    a boolean fragment would make the DEFAULT config the in-loop form -- a
    behaviour change on every cell the pass fires on, and exactly the regression
    the byte-identical-hash gate exists to catch.  ``EnumFragment.default()`` is
    ``choices[0]``, so ``ONLINE_DEFER_CHOICES = (True, False)`` is what preserves
    today's kernel while still offering both values to the search.

    ⚠ PER-DEVICE-LOOP, NOT PER-REDUCTION-BLOCK, and that distinction is load-bearing
    rather than stylistic.  ``CuteReductionReloadSpec`` above is registered once per
    *reduction* block, which is right for it -- it describes a row layout.  This pass
    fires on an ``hl.tile`` loop carrying a recurrence, and MEASURED on
    ``cross_entropy_online`` (the only kernel in the reduction table it fires on)
    that loop's block is a plain TILE block: the kernel has
    ``reduction_loops == []`` and ``block_sizes == [0, 1]``.  A per-reduction-block
    registration would therefore have created ZERO slots on the one kernel the knob
    controls, i.e. an unreachable knob that looks registered.
    """

    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _default(self) -> bool:
        return online_defer_for(self.size_hint)

    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        default = self._default()
        choices = (default, *(c for c in ONLINE_DEFER_CHOICES if c != default))
        return EnumFragment(choices=choices)

    def _normalize(self, name: str, value: object) -> bool:
        # ``type(...) is not bool`` rather than ``isinstance``: ``bool`` is a
        # subclass of ``int``, so ``isinstance(1, bool)`` is False but
        # ``isinstance(True, int)`` is True, and a config written as ``[1]``
        # instead of ``[True]`` must be told so rather than silently coerced.
        if type(value) is not bool:
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}")
        return value

    def _fill_missing(self) -> bool:
        # The ladder value is what every config in the tree already compiles to,
        # because the pass ran unconditionally before this knob existed.  Filling
        # ``False`` would mean every frozen config -- all of which omit this key --
        # silently switched to the slower in-loop form.  So "omitted" means "use
        # the ladder", and that is precisely what makes the promotion a pure
        # reachability change.
        return self._default()


class CuteNDTileTvSpec(_BlockIdItem):
    """Per-device-loop: does an explicit ``hl.tile`` loop load through a vectorized
    TV ``cute.copy`` (CuTe)?

    ``True`` = one 128-bit ``cute.copy`` per outer lane iteration; ``False`` = today's
    form, two ``cute.arch.load`` of ``vector<4 x Uint16>`` (two 64-bit LDGs) plus a
    per-element ``if vec_lane < 4 else ...`` select-and-bitcast on every consume.
    Read by ``tile_strategy.cute_ndtile_tv_enabled``.

    ⭐ WHY THIS IS A KNOB AND NOT A DEFAULT, and the argument is a MEASUREMENT.  This
    began as the ``HELION_CUTE_NDTILE_TV`` env var, which was the right instrument for
    one question ("is the trade a win?") and the wrong one for the answer, because the
    answer is *sometimes*.  MEASURED on all 8 ``cross_entropy_online`` cells the gate
    moves -- one gate arm per process, position-balanced, judged on the mean because
    ``cuda.Event.elapsed_time`` is quantized to ~2.04us on this box:

        lane extent 2      -> WINS up to -4.46% at the incumbent chunk
        lane extent 4      -> WINS ~4-7%   (the best rung on the narrow-chunk cells)
        lane extent 8-16   -> WINS -0.5% to -6.9%, and LOSES to extent 4 on 3 of the
                              4 cells measured in both arms

    ⛔ THIS TABLE WAS INVERTED UNTIL 2026-08-01 (run 2 T0).  It read "extent 2 loses
    ~2%; extent 8-16 WINS +81% to +110%".  The +81..110% magnitudes are the gate-**OFF**
    LOSSES from raising ``chunk`` (32768x8192 tv_OFF chunk2048 ext8: +80.084%), i.e. how
    much the TV arm rescues a wide chunk -- not a speedup over the incumbent.  Anyone
    tuning toward +81% pushes ``chunk`` to the extent-16 rung, which is measured the
    WORSE arm on most cells.  ⇒ the rule is "extent 4 on the narrow-chunk cells, 8-16
    only on the already-wide ones".

    The mechanism claim survives and explains the inversion: the legacy path hoists
    exactly TWO ``arch.load``s regardless of extent, so it cliffs ~1.8x between extent 4
    and 8 while the TV path stays flat -- that cliff IS the gate-OFF loss above.
    ⇒ three of the eight cells prefer ``False`` at their current geometry and ``True``
    once chunk/threads are re-tuned for it.  A global default forces one answer on all
    eight; this lets each loop record its own.  (8 cells were then re-frozen onto
    ``True`` with re-tuned chunk, -0.5%..-5.4%, confirmed at 7 rounds x 60 reps.)

    ⚠ AND THE KNOB CANNOT BE REPLACED BY A LADDER, which is the second reason it is a
    knob.  The carrier is lane extent = ``chunk / (num_threads * vec)`` -- i.e. a
    function of ``block_sizes`` and ``num_threads``, which are themselves being
    searched.  A ladder here would have to predict the outcome of the search it is
    part of.

    ⚠ WHY ``EnumFragment`` OVER ``(False, True)`` AND NOT ``BooleanFragment`` -- the
    same reason ``CuteOnlineDeferSpec`` gives, with the opposite polarity.
    ``EnumFragment.default()`` is ``choices[0]``, and this knob's inert value is
    ``False`` (the env gate defaults off), so ``NDTILE_TV_CHOICES = (False, True)``
    is what keeps every existing config byte-identical while still offering ``True``
    to the search.  ``BooleanFragment.default()`` is a hardcoded ``False`` and would
    coincidentally give the right default here -- but it cannot express the *ordered*
    domain, and relying on that coincidence would break the moment the polarity of
    some future knob differed.

    ⚠ PER-DEVICE-LOOP, NOT PER-REDUCTION-BLOCK, and this is load-bearing.  The gate is
    read at plan construction for a ``CuteNDTileStrategy``, whose loop is a plain TILE
    block: MEASURED, ``cross_entropy_online`` -- the only kernel in the reduction table
    on this path -- has ``reduction_loops == []`` and ``block_sizes == [0, 1]``.  A
    per-reduction-block registration would have created ZERO slots on the one kernel
    the knob controls, i.e. a knob that looks registered and controls nothing.  That is
    exactly why it shares ``_register_cute_ast_pass_specs``' domain with
    ``cute_online_defer``, which learned the same lesson.
    """

    def __init__(
        self,
        *,
        block_id: int,
        size_hint: int,
    ) -> None:
        super().__init__([block_id])
        self.size_hint = size_hint

    def _default(self) -> bool:
        return ndtile_tv_for(self.size_hint)

    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        default = self._default()
        choices = (default, *(c for c in NDTILE_TV_CHOICES if c != default))
        return EnumFragment(choices=choices)

    def _normalize(self, name: str, value: object) -> bool:
        # ``type(...) is not bool`` rather than ``isinstance``, for the reason
        # ``CuteOnlineDeferSpec`` records: ``bool`` subclasses ``int``, so a config
        # written ``[1]`` instead of ``[True]`` must be told so, not coerced.
        if type(value) is not bool:
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}")
        return value

    def _fill_missing(self) -> bool:
        # "USE THE LADDER", which here means ``False`` -- and unlike
        # ``CuteOnlineDeferSpec`` the ladder and the inert value coincide, so this is
        # the safe direction by construction rather than by argument.
        #
        # ⭐ THIS IS THE LINE THAT MAKES THE PROMOTION A PURE REACHABILITY CHANGE.
        # Every config in the tree -- all 40 frozen cells, every ``.expected`` golden,
        # every hand-written config -- omits this key, and the env gate they were
        # measured under was OFF.  Filling ``True`` would silently switch all of them
        # to the TV emission: 8 frozen cells move and 3 tests that pin the
        # ``cute.arch.load`` form go red.  Filling ``False`` keeps them byte-identical.
        return self._default()


class _OptionalIntSpec(_BlockIdItem):
    def _normalize(self, name: str, value: object) -> int:
        if not isinstance(value, int):
            raise InvalidConfig(f"{name} must be an integer, got {value!r}")
        return value

    def _fill_missing(self) -> int:
        """Provide a value when not provided by the user."""
        return 0


class _OptionalBoolSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> EnumFragment:
        return EnumFragment((None, False, True))

    def _normalize(self, name: str, value: object) -> bool | None:
        if value is not None and not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean or None, got {value!r}")
        return value

    def _fill_missing(self) -> None:
        """Provide a value when not provided by the user."""
        return None


class RangeUnrollFactorSpec(_OptionalIntSpec):
    def _fragment(self, base: ConfigSpec) -> IntegerFragment:
        return IntegerFragment(0, 4, 0)


class RangeWarpSpecializeSpec(_OptionalBoolSpec):
    pass


class RangeNumStagesSpec(_OptionalIntSpec):
    def _fragment(self, base: ConfigSpec) -> IntegerFragment:
        return IntegerFragment(0, 4, 0)


class RangeMultiBufferSpec(_OptionalBoolSpec):
    pass


class RangeFlattenSpec(_OptionalBoolSpec):
    pass


class StaticRangeSpec(_BlockIdItem):
    def _fragment(self, base: ConfigSpec) -> BooleanFragment:
        return BooleanFragment()

    def _normalize(self, name: str, value: object) -> bool:
        if not isinstance(value, bool):
            raise InvalidConfig(f"{name} must be a boolean, got {value!r}")
        return value

    def _fill_missing(self) -> bool:
        """Provide a value when not provided by the user."""
        return False


def _product(seq: Sequence[int]) -> int:
    """Return the product of the elements in the sequence."""
    return functools.reduce(operator.mul, seq, 1)
