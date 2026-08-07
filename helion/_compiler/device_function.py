from __future__ import annotations

import ast
from collections import defaultdict
import contextlib
import dataclasses
import enum
import itertools
import math
import threading
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import Protocol
from typing import TypeVar
from typing import cast

import sympy
import torch
from torch._dynamo.source import TensorProperty
from torch._dynamo.source import TensorPropertySource
from torch.fx.graph import _Namespace

from .. import exc
from .._compat import get_tensor_descriptor_fn_name
from .ast_extension import ExtendedAST
from .ast_extension import create
from .ast_extension import create_arg
from .ast_extension import create_arguments
from .ast_extension import expr_from_string
from .ast_extension import statement_from_string
from .ast_read_writes import ReadWrites
from .ast_read_writes import ast_rename
from .ast_read_writes import dead_assignment_elimination
from .ast_read_writes import dead_lane_loop_elimination
from .backend_registry import all_reserved_launch_param_names
from .compile_environment import CompileEnvironment
from .cute.device_state import CuteDeviceFunctionState
from .host_function import HostFunction
from .host_function import NoCurrentFunction
from .output_header import reserved_names
from .source_location import SyntheticLocation
from .variable_origin import BlockSizeOrigin
from .variable_origin import GridOrigin
from .variable_origin import Origin
from .variable_origin import TensorSizeOrigin
from .variable_origin import TileBeginOrigin

if TYPE_CHECKING:
    from ..runtime.config import Config
    from .device_ir import HelperFunctionGraphInfo
    from .generate_ast import GenerateAST
    from .indexing_strategy import IndexingStrategy
    from .program_id import ProgramIDs
    from helion._compiler.pallas.ordered_carry import CarryBoundaryTile
    from helion._compiler.pallas.ordered_carry import CarryScratchKey
    from helion._compiler.pallas.plan_tiling import DimensionTiling

    _P = TypeVar("_P", bound="TensorPropertyArg")

    class _TLS(Protocol):
        functions: list[DeviceFunction]


tls: _TLS = cast("_TLS", threading.local())

# "This block owns no slot in that config key", distinct from every legal knob value.
# A plain ``None`` will not do: ``None`` is itself a legal value for several
# per-block knobs (``cute_reduction_reload``, ``reduction_loops``), so a ``None``
# default from ``config_get`` cannot be told apart from a real one.
_MISSING: object = object()


class VarInfo(NamedTuple):
    """Information about a variable derived from a sympy expression."""

    name: str
    fx_node: torch.fx.Node


def find_block_size_symbols(
    expr: sympy.Expr,
) -> tuple[dict[sympy.Symbol, int], set[sympy.Symbol]]:
    """
    Find block size symbols in a sympy expression.

    Returns:
        tuple of (block_size_mapping, non_block_size_symbols) where:
        - block_size_mapping: dict mapping block size symbols to their block_id
        - non_block_size_symbols: set of symbols that are NOT block sizes
    """
    if not isinstance(expr, sympy.Expr):
        return {}, set()

    hf = HostFunction.current()
    block_sizes = {}
    non_block_size_symbols = set()

    for symbol in expr.free_symbols:
        origin_info = hf.expr_to_origin.get(symbol)
        if origin_info is None or not isinstance(origin_info.origin, BlockSizeOrigin):
            # pyrefly: ignore [bad-argument-type]
            non_block_size_symbols.add(symbol)
        else:
            # pyrefly: ignore [unsupported-operation]
            block_sizes[symbol] = origin_info.origin.block_id

    # pyrefly: ignore[bad-return]
    return block_sizes, non_block_size_symbols


def contains_only_block_size_symbols(expr: sympy.Expr) -> bool:
    """Check if expression contains only block size symbols (no other variables)."""
    _, non_block = find_block_size_symbols(expr)
    return len(non_block) == 0


@dataclasses.dataclass
class Argument:
    name: str  # in the device function

    def host_str(self) -> str:
        raise NotImplementedError

    def arg_def_node(self) -> ast.arg:
        return create_arg(self.name)

    def sort_key(self) -> tuple[object, ...]:
        return (_sort_order[type(self)],)


@dataclasses.dataclass
class TensorArg(Argument):
    fake_value: torch.Tensor
    _host_str: str | None

    def host_str(self) -> str:
        if self._host_str is None:
            raise RuntimeError("TensorArg has no host representation")
        return self._host_str


@dataclasses.dataclass
class TensorDescriptorArg(TensorArg):
    # Permutation applied to make stride==1 dimension last
    permutation: list[int] | None = None

    def host_str(self) -> str:
        if self._host_str is None:
            raise RuntimeError(
                "TensorDescriptorArg is device-only and has no host representation"
            )
        return self._host_str

    @property
    def inverse_permutation(self) -> list[int]:
        """Get the inverse permutation to undo the applied permutation."""
        if (permutation := self.permutation) is None:
            raise RuntimeError("TensorDescriptorArg.permutation is None")
        inverse_perm = [0] * len(permutation)
        for i, p in enumerate(permutation):
            inverse_perm[p] = i
        return inverse_perm


@dataclasses.dataclass
class TensorPropertyArg(Argument):
    tensor_arg: TensorArg
    dim: int

    def sort_key(self) -> tuple[object, ...]:
        return (_sort_order[type(self)], self.tensor_arg.name, self.dim)


class TensorSizeArg(TensorPropertyArg):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.size({self.dim})"


class TensorStrideArg(TensorPropertyArg):
    def host_str(self) -> str:
        return f"{self.tensor_arg.host_str()}.stride({self.dim})"


@dataclasses.dataclass
class NumericArgument(Argument):
    _host_str: str

    def host_str(self) -> str:
        return self._host_str


class ConstExprArg(NumericArgument):
    def arg_def_node(self) -> ast.arg:
        return create_arg(
            self.name, CompileEnvironment.current().backend.constexpr_type
        )


@dataclasses.dataclass
class SymbolArgument(NumericArgument):
    pass


class StaticShape(Argument):
    def __init__(self, val: int) -> None:
        super().__init__(repr(val))


_sort_order: dict[type[Argument], int] = {
    TensorDescriptorArg: 0,
    TensorArg: 0,
    TensorSizeArg: 1,
    TensorStrideArg: 2,
    SymbolArgument: 3,
    ConstExprArg: 4,
}


@dataclasses.dataclass
class ScratchArg:
    """A scratch memory buffer allocated in device memory (e.g., VMEM on TPU).

    scratch_type can be "vmem" (default) for VMEM buffers or "dma_semaphore"
    for DMA semaphores used with pltpu.make_async_copy.
    """

    name: str
    shape: tuple[int | torch.SymInt, ...]
    dtype: torch.dtype | None  # None for semaphores
    scratch_type: str = "vmem"  # "vmem" or "dma_semaphore"


def _is_literal_constexpr(arg: ConstExprArg) -> bool:
    """Check if a constexpr arg has a known literal value that can be inlined at module level."""
    host_str = arg.host_str()
    if host_str == arg.name:
        return False
    try:
        ast.literal_eval(host_str)
        return True
    except (ValueError, SyntaxError):
        return False


class PallasMemorySpace(enum.Enum):
    """TPU memory space for Pallas tensors."""

    HBM = "hbm"  # Pipeline body tensors (DMA)
    SMEM = "smem"  # Scalar-only access
    VMEM = "vmem"  # Vector/slice access (default)


class DeviceFunction:
    def __init__(
        self,
        name: str,
        config: Config,
        codegen: GenerateAST,
    ) -> None:
        super().__init__()
        self.name = name
        self.config = config
        self.codegen = codegen
        self.arguments: list[Argument] = []
        self.preamble: list[ast.AST] = []
        self.body: list[ast.AST] = []
        self._tensor_args: dict[torch.Tensor, TensorArg] = {}
        self._tensor_descriptor_args: dict[
            tuple[torch.Tensor, str], TensorDescriptorArg
        ] = {}
        self._expr_args: dict[sympy.Expr, SymbolArgument] = {}
        self._constexpr_args: dict[str, ConstExprArg] = {}
        self._constexpr_host_defs: set[str] = set()
        self._scratch_args: list[ScratchArg] = []
        self.wrapper_only_params: list[str] = []
        self._tensor_properties: dict[
            tuple[type[TensorPropertyArg], torch.Tensor, int], TensorPropertyArg
        ] = {}
        self._unique_counter: dict[str, itertools.count[int]] = defaultdict(
            itertools.count
        )
        self.pid: ProgramIDs | None = None
        self.namespace: _Namespace = _Namespace()
        self.namespace._used_names.update(reserved_names())

        self.namespace._used_names.update(all_reserved_launch_param_names())
        self.namespace._used_names.update(
            x.removeprefix("_triton_config_")
            for x in config
            if x.startswith("_triton_config_")
        )
        self._variable_renames: dict[str, list[str]] = {}
        self.dce_vars: list[str] = []
        # Names of matmul-fallback running-sum accumulators emitted as
        # ``acc = acc + product`` inside a constexpr V-loop.  The
        # ``hoist_warp_reduce`` pass reads this to reduce the FINAL value once
        # (instead of building a per-lane V-fold, which would double-count the
        # already-accumulated running sum).  See ``_emit_cute_matmul``.
        self.cute_matmul_running_sums: set[str] = set()
        # Arg names referenced only by fusion placeholder strings
        # (<STORE_OUTPUT_*>, <LOAD_INPUT_*>), not by the AST body.
        # DCE would incorrectly strip them without this exemption.
        self.placeholder_args: set[str] = set()
        # Sourceless prologue params (e.g. ones_like) that are fully inlined
        # by the prologue hook.  These should be DCE'd away and also removed
        # from the host function signature (populated by _codegen_prologue_fusion).
        self.sourceless_prologue_params: set[str] = set()
        self.block_size_var_cache: dict[tuple[int, ...], str] = {}
        self.expr_to_var_info: dict[sympy.Expr, VarInfo] = {}
        self.deferred_rdim_defs: list[tuple[str, sympy.Expr]] = []
        self._cute_state = CuteDeviceFunctionState()

        from .helper_function import HelperFunctionManager

        self.helper_manager = HelperFunctionManager()

        from .tile_dispatch import TileStrategyDispatch

        self.tile_strategy: TileStrategyDispatch = TileStrategyDispatch(self, config)

        # Store indexing config to lazily create strategies per load/store
        self._indexing_config = config.indexing
        self.indexing_strategies: list[IndexingStrategy] = []

        # Atomic indexing config (separate from load/store indexing)
        self._atomic_indexing_config = config.atomic_indexing
        self.atomic_indexing_strategies: list[IndexingStrategy] = []
        self.atomic_op_index = 0

        self.rng_seed_count = 0
        self.device_load_index = 0
        self.device_load_cache_modifier_index = 0
        self.device_store_index = 0
        self.device_store_cache_modifier_index = 0
        # Single counter for both loads and stores for indexing assignment
        self.device_memory_op_index = 0
        self.epilogue_subtile_store_indices: dict[str, int] = {}
        self.epilogue_subtile_atomic_indices: dict[str, int] = {}
        self.rng_seed_buffer_param_name = None

        # Pallas: id(fake_tensor) → [DimensionTiling], recorded during `plan_tiling`
        self.pallas_tensor_dim_tilings: dict[int, list[DimensionTiling]] = {}
        # Pallas: id(fake_tensor) → memory space, determined during
        # tracing (HBM for pipeline) and codegen (SMEM for scalar access).
        # NOTE: Currently each tensor can only have one memory space.
        # If a tensor needs both SMEM (scalar access) and VMEM (slice
        # access), it will need tensor duplication — passing the same
        # data as two separate args in different memory spaces. This
        # dict would then need to support multiple entries per tensor
        # or the tensor would get distinct arg IDs per memory space.
        self.pallas_memory_space: dict[int, PallasMemorySpace] = {}
        # Pallas: id(fake_tensor) → {dim: (block_id, extra_pad)} for dims
        # using pl.ds() that may need host-side padding.
        self.pallas_pad_info: dict[int, dict[int, tuple[int, int]]] = {}
        # Pallas ordered carry: jagged row block_id -> CarryBoundaryTile.  Filled by
        # the emit_pipeline codegen when the tile is a legal map axis; read by
        # the store codegen to stitch the boundary across neighbouring groups.
        self.carry_tiles: dict[int, CarryBoundaryTile] = {}
        # CarryScratchKey(row block_id, output name) -> carry scratch var name.
        # One scratch per output buffer (a tile may feed several stores),
        # allocated at the store.
        self.carry_scratch: dict[CarryScratchKey, str] = {}

    def allocate_store_index(self) -> int:
        """Bump store counters and return the indexing strategy slot."""
        self.device_store_index += 1
        idx = self.device_memory_op_index
        self.device_memory_op_index += 1
        return idx

    def get_indexing_strategy(self, index: int) -> IndexingStrategy:
        from .indexing_strategy import IndexingStrategy
        from .indexing_strategy import PointerIndexingStrategy

        # Expand strategies list if needed
        while len(self.indexing_strategies) <= index:
            idx = len(self.indexing_strategies)

            if isinstance(self._indexing_config, str):
                # Single string: all loads/stores use the same strategy
                if not self.indexing_strategies:
                    strategy = IndexingStrategy.select(self._indexing_config)
                else:
                    strategy = self.indexing_strategies[0]
            elif isinstance(self._indexing_config, list) and self._indexing_config:
                # List: one strategy per load/store
                assert idx < len(self._indexing_config), (
                    f"Load/Store operation {idx} exceeds indexing config length "
                    f"{len(self._indexing_config)}. Please specify indexing for all loads and stores."
                )
                strategy = IndexingStrategy.select(self._indexing_config[idx])
            else:
                # Empty/default: use pointer
                strategy = PointerIndexingStrategy()

            self.indexing_strategies.append(strategy)

        return self.indexing_strategies[index]

    def get_atomic_indexing_strategy(self, index: int) -> IndexingStrategy:
        from .indexing_strategy import IndexingStrategy
        from .indexing_strategy import PointerIndexingStrategy

        while len(self.atomic_indexing_strategies) <= index:
            idx = len(self.atomic_indexing_strategies)

            if isinstance(self._atomic_indexing_config, str):
                if not self.atomic_indexing_strategies:
                    strategy = IndexingStrategy.select(self._atomic_indexing_config)
                else:
                    strategy = self.atomic_indexing_strategies[0]
            elif (
                isinstance(self._atomic_indexing_config, list)
                and self._atomic_indexing_config
            ):
                assert idx < len(self._atomic_indexing_config), (
                    f"Atomic operation {idx} exceeds atomic_indexing config length "
                    f"{len(self._atomic_indexing_config)}. Please specify atomic_indexing for all atomic ops."
                )
                strategy = IndexingStrategy.select(self._atomic_indexing_config[idx])
            else:
                strategy = PointerIndexingStrategy()

            self.atomic_indexing_strategies.append(strategy)

        return self.atomic_indexing_strategies[index]

    def has_rng_ops(self) -> bool:
        """Check if this kernel uses any RNG operations."""
        return self.rng_seed_count > 0 and self.rng_seed_buffer_param_name is not None

    def reserve_rng_seed(self, seed_index: int) -> None:
        """Ensure the RNG seed buffer is available up to a specific index."""
        assert seed_index >= 0
        self.rng_seed_count = max(self.rng_seed_count, seed_index + 1)
        if self.rng_seed_buffer_param_name is None:
            # pyrefly: ignore [bad-assignment]
            self.rng_seed_buffer_param_name = self.new_var("rng_seed_buffer")

    def block_size_var(self, block_id: int) -> str | None:
        key = (block_id,)

        # Block size var could be used outside of a hl.tile loop, and at that point
        # no tile strategy has populated the cache yet, so we must lazily create
        # the constexpr argument here and lift it as device function argument;
        # later strategies will reuse the cached name or intentionally replace it
        # (e.g. flattened loops, reductions).
        if key not in self.block_size_var_cache:
            env = CompileEnvironment.current()
            block_value = env.block_sizes[block_id].from_config(self.config)

            if block_value is None:
                return None

            var_name = self.new_var(f"_BLOCK_SIZE_{block_id}")
            self.block_size_var_cache[key] = var_name
            self.constexpr_arg_with_host_def(var_name, block_value)

        return self.block_size_var_cache[key]

    def resolved_block_size(self, block_id: int) -> int | torch.SymInt | None:
        """Resolve a block_id to its concrete size for the current config."""
        env = CompileEnvironment.current()
        return env.block_sizes[block_id].from_config(self.config)

    def evaluate_constexpr_condition(self, test: object) -> bool | None:
        """Resolve a control-flow test to a concrete bool for the current config.

        Block sizes are symbolic during the (single) frontend pass but become
        concrete integers per-config at codegen time.  A branch condition that
        depends only on block sizes (and other constants) is therefore a
        compile-time constant here, even though it could not be folded earlier.
        Substituting the config's block sizes lets us decide the branch and emit
        only the live side, which is what makes shape-divergent branches legal.

        Returns the concrete bool, or ``None`` if the test is not a compile-time
        constant for this config (e.g. it depends on a runtime value).
        """
        if isinstance(test, (bool, int)):
            return bool(test)
        if not isinstance(test, torch.SymBool):
            return None
        # NB: a boolean expression (And/Or/Relational) is a sympy.Basic but not a
        # sympy.Expr, so we classify its free symbols directly rather than via
        # find_block_size_symbols (which only handles sympy.Expr).
        expr = test._sympy_()
        if not isinstance(expr, sympy.Basic):
            return None
        env = CompileEnvironment.current()
        hf = HostFunction.current()
        subs: dict[sympy.Basic, sympy.Basic] = {}
        for symbol in expr.free_symbols:
            origin_info = hf.expr_to_origin.get(symbol)
            if origin_info is None or not isinstance(
                origin_info.origin, BlockSizeOrigin
            ):
                return None
            value = env.block_sizes[origin_info.origin.block_id].from_config(
                self.config
            )
            if not isinstance(value, int):
                return None
            subs[symbol] = sympy.Integer(value)
        resolved = expr.xreplace(subs)
        if resolved.free_symbols:
            return None
        return bool(resolved)

    def try_map_block_symbols_to_vars(self, expr: sympy.Expr) -> sympy.Expr | None:
        """Try to map all block size symbols in expression to their variable names.

        Returns:
            - The expression with symbols replaced if ALL symbols are block sizes and have variables
            - None if the expression contains non-block symbols or unmapped block symbols
        """
        block_mapping, non_block_symbols = find_block_size_symbols(expr)

        # Can't map if there are non-block symbols
        if non_block_symbols:
            return None

        # No symbols to map - return as-is
        if not block_mapping:
            return expr

        # Try to map all block symbols to their variables
        var_map = {}
        for symbol, block_id in block_mapping.items():
            block_var = self.block_size_var(block_id)
            if not block_var:
                # Can't map this block symbol - fail
                return None
            var_map[symbol] = sympy.Symbol(block_var, integer=True)

        # Successfully mapped all symbols
        # pyrefly: ignore [bad-return]
        return expr.xreplace(var_map)

    def merge_variable_names(self, a: str, b: str) -> None:
        name_group = [
            *self._variable_renames.get(a, [a]),
            *self._variable_renames.get(b, [b]),
        ]
        for n in name_group:
            self._variable_renames[n] = name_group

    def variable_aliases(self, name: str) -> tuple[str, ...]:
        return tuple(self._variable_renames.get(name, [name]))

    @property
    def cute_state(self) -> CuteDeviceFunctionState:
        return self._cute_state

    def set_pid(self, pid: ProgramIDs) -> None:
        if self.pid is not None:
            raise exc.InvalidAPIUsage(
                "Multiple top-level grid loops are not supported with this config. "
                "Try using pid_type='persistent' or combining the loops into a single "
                "hl.tile/hl.grid call."
            )
        self.pid = pid

    def sympy_expr(self, expr: sympy.Expr) -> str:
        env = CompileEnvironment.current()
        with contextlib.suppress(Exception):
            expr = env.shape_env.simplify(expr)
        expr = env.specialize_expr(expr)
        if not expr.free_symbols:
            return env.backend.sympy_printer_expr(expr)
        if expr in self.expr_to_var_info:
            return self.expr_to_var_info[expr].name
        expr_to_origin = HostFunction.current().expr_to_origin
        if expr in expr_to_origin:
            return self._lift_sympy_arg(expr)
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda x: x.name):
            assert isinstance(sym, sympy.Symbol)
            if sym in self.expr_to_var_info:
                replacements[sym] = sympy.Symbol(
                    self.expr_to_var_info[sym].name, integer=True
                )
            else:
                assert sym in expr_to_origin, f"no origin found for {sym.name}"
                replacements[sym] = sympy.Symbol(
                    self._lift_sympy_arg(sym), integer=True
                )
        # pyrefly: ignore [bad-argument-type]
        return env.backend.sympy_printer_expr(expr.xreplace(replacements))

    def _lift_sympy_arg(self, expr: sympy.Expr) -> str:
        env = CompileEnvironment.current()
        origin = HostFunction.current().expr_to_origin[expr]
        if isinstance(origin.origin, TensorSizeOrigin):
            assert origin.fake_value is not None
            arg = self.tensor_size(
                origin.fake_value,
                origin.origin.key,
            )
            return arg.name
        if isinstance(origin.origin, BlockSizeOrigin):
            result = self.block_size_var(env.canonical_block_id(origin.origin.block_id))
            assert result is not None
            return result
        if isinstance(origin.origin, GridOrigin):
            # Only the tile's begin is the loop offset.  The other tile edges
            # render as compound expressions (``tile.end`` clamps to the loop
            # end, ``tile.count`` divides, ``tile.id`` shifts), so returning the
            # offset for them would silently substitute the begin.  This path is
            # reached whenever such a symbol survives into a sympy expression
            # rather than its own op -- e.g. ``min(n, tile.end)``, which
            # ``_builtin_min`` folds into ``sympy.Min`` at trace time.  The
            # result becomes a sympy Symbol name, so parenthesize it to keep
            # precedence intact inside the enclosing expression.
            resolved = env.resolve_codegen_block_id(
                origin.origin.block_id, self.codegen
            )
            if type(origin.origin) in (GridOrigin, TileBeginOrigin):
                return self.codegen.offset_var(resolved)
            # Render through the origin so each derived edge keeps its own
            # formula, but on the resolved live loop: host_str() reads
            # offset_var and active_device_loops by block_id, and an aliased
            # symbol's own block may have no active loop.
            derived = dataclasses.replace(origin.origin, block_id=resolved)
            return f"({derived.host_str()})"
        return self.expr_arg(expr, origin.origin).name

    def user_sympy_expr(self, expr: sympy.Expr) -> str:
        """A sympy expression that flows into user computations."""
        expr_to_origin = HostFunction.current().expr_to_origin
        replacements = {}
        for sym in sorted(expr.free_symbols, key=lambda s: s.name):
            assert isinstance(sym, sympy.Symbol)
            origin_info = expr_to_origin.get(sym)
            if origin_info is None:
                continue
            origin = origin_info.origin
            if isinstance(origin, BlockSizeOrigin):
                replacements[sym] = self.tile_strategy.user_size(origin.block_id)
        if replacements:
            # pyrefly: ignore [bad-assignment]
            expr = expr.xreplace(replacements)
        return self.sympy_expr(expr)

    def literal_expr(self, expr: object) -> str:
        if isinstance(expr, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            # SymBool's `_sympy_()` is a boolean, not an Expr that sympy_expr wants.
            # pyrefly: ignore [bad-argument-type]
            return self.sympy_expr(expr._sympy_())
        if isinstance(expr, sympy.Expr):
            return self.sympy_expr(expr)
        if isinstance(expr, float) and not math.isfinite(expr):
            return f"float('{expr}')"
        return repr(expr)

    def unique_name(self, prefix: str, dce: bool = False) -> str:
        return self.new_var(f"{prefix}_{next(self._unique_counter[prefix])}", dce=dce)

    def new_var(self, name: str, *, dce: bool = False) -> str:
        name = self.namespace.create_name(name, None)
        if dce:
            self.dce_vars.append(name)
        return name

    def tensor_arg(
        self, fake_value: torch.Tensor, prefer_name: str | None = None
    ) -> TensorArg:
        if fake_value not in self._tensor_args:
            origin = HostFunction.current().tensor_to_origin[fake_value]
            arg = TensorArg(
                self.new_var(prefer_name or origin.suggest_var_name()),
                fake_value,
                origin.host_str(),
            )
            self.arguments.append(arg)
            self._tensor_args[fake_value] = arg
        return self._tensor_args[fake_value]

    def tensor_descriptor_arg(
        self, fake_value: torch.Tensor, block_size: list[int | torch.SymInt]
    ) -> TensorDescriptorArg:
        host_function = HostFunction.current()
        block_size_expr = ", ".join(map(self.literal_expr, block_size))
        key = (fake_value, block_size_expr)
        if key not in self._tensor_descriptor_args:
            origin = host_function.tensor_to_origin[fake_value]
            desc_name = self.new_var(origin.suggest_var_name() + "_desc")
            env = CompileEnvironment.current()

            # Find which dimension has stride==1
            layout_signature = env.tensor_descriptor_layout_signature(fake_value)
            assert layout_signature is not None
            stride_one_dim = layout_signature[0]
            assert stride_one_dim is not None

            # Determine if we need permutation (stride==1 dimension is not last)
            permutation = None
            if stride_one_dim != fake_value.ndim - 1:
                # Create permutation to move stride==1 dimension to last position
                permutation = [*range(fake_value.ndim)]
                permutation.pop(stride_one_dim)
                permutation.append(stride_one_dim)

            # Create the regular tensor arg and size/stride args
            tensor_arg = self.tensor_arg(fake_value)
            size_args = [
                self.tensor_size(fake_value, i) for i in range(fake_value.ndim)
            ]
            stride_args = [
                self.tensor_stride(fake_value, i) for i in range(fake_value.ndim)
            ]

            # Apply permutation if needed
            if permutation is not None:
                size_args = [size_args[i] for i in permutation]
                stride_args = [stride_args[i] for i in permutation]
                block_size = [block_size[i] for i in permutation]
                # Update block_size_expr for the permuted order
                block_size_expr = ", ".join(map(self.literal_expr, block_size))

            descriptor_dims = (
                permutation if permutation is not None else [*range(fake_value.ndim)]
            )
            assert descriptor_dims[-1] == stride_one_dim
            # The descriptor permutation above makes the last descriptor
            # dimension the proven stride-one dimension. Triton checks this
            # predicate at JIT time, so emit it as a literal even when other
            # dynamic strides are runtime scalars.
            stride_args[-1] = StaticShape(1)

            # Add tl.make_tensor_descriptor call to preamble
            sizes = ", ".join([arg.name for arg in size_args])
            strides = ", ".join([arg.name for arg in stride_args])

            tensor_descriptor_fn_name = get_tensor_descriptor_fn_name()
            descriptor_stmt = statement_from_string(
                f"{desc_name} = {tensor_descriptor_fn_name}({tensor_arg.name}, [{sizes}], [{strides}], [{block_size_expr}])"
            )
            self.preamble.append(descriptor_stmt)

            arg = TensorDescriptorArg(
                desc_name,
                fake_value,
                None,  # No host_str since this is device-only
                permutation,
            )
            # Don't add to self.arguments since this is device-only
            self._tensor_descriptor_args[key] = arg
        return self._tensor_descriptor_args[key]

    def expr_arg(self, sym: sympy.Expr, origin: Origin) -> SymbolArgument:
        if sym not in self._expr_args:
            arg = SymbolArgument(
                name=self.new_var(origin.suggest_var_name()),
                _host_str=origin.host_str(),
            )
            self.arguments.append(arg)
            self._expr_args[sym] = arg
        return self._expr_args[sym]

    def constexpr_arg(self, name: str, value: object | None = None) -> bool:
        """Create a constexpr argument, returns True if created, False if already exists."""
        if name in self._constexpr_args:
            return False
        host_str = name if value is None else self._format_constexpr_value(value)
        self._constexpr_args[name] = rv = ConstExprArg(name, host_str)
        self.arguments.append(rv)
        return True

    def constexpr_arg_with_host_def(self, name: str, value: object) -> None:
        """Create a constexpr argument and add its host-side definition if needed."""
        created = self.constexpr_arg(name, value)
        host_expr = self._constexpr_args[name].host_str()
        if created or name not in self._constexpr_host_defs:
            self.codegen.host_statements.append(
                statement_from_string(f"{name} = {host_expr}")
            )
        self._constexpr_host_defs.add(name)

    def _format_constexpr_value(self, value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float, bool)):
            return repr(value)

        # Extract sympy expression from torch symbolic types
        if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            value = value._sympy_()

        # Handle sympy expressions
        if isinstance(value, sympy.Expr):
            return HostFunction.current().sympy_expr(value)

        return HostFunction.current().literal_expr(value)

    def _tensor_property(
        self,
        prop_cls: type[_P],
        fake_value: torch.Tensor,
        dim: int,
        prefix: str,
    ) -> _P:
        # TODO(jansel): dedupe based on sympy expressions
        key = (prop_cls, fake_value, dim)
        if key not in self._tensor_properties:
            arg = self.tensor_arg(fake_value)
            prop = prop_cls(f"{arg.name}_{prefix}_{dim}", arg, dim)
            self.arguments.append(prop)
            self._tensor_properties[key] = prop
        return cast("_P", self._tensor_properties[key])

    def tensor_size(self, fake_value: torch.Tensor, dim: int) -> Argument:
        if isinstance(v := fake_value.size(dim), int) or isinstance(
            v._sympy_(), sympy.Integer
        ):
            return StaticShape(int(v))
        return self._tensor_property(TensorSizeArg, fake_value, dim, "size")

    def tensor_stride(self, fake_value: torch.Tensor, dim: int) -> Argument:
        v = fake_value.stride(dim)
        env = CompileEnvironment.current()
        # Check if this stride was explicitly specialized
        source = env.tensor_input_source(fake_value)
        if (
            source is not None
            and TensorPropertySource(source, TensorProperty.STRIDE, dim)
            in env.specialized_strides
        ):
            return StaticShape(int(v))
        if isinstance(v, int):
            if env.settings.static_shapes:
                return StaticShape(v)
        return self._tensor_property(TensorStrideArg, fake_value, dim, "stride")

    def sorted_args(self) -> list[Argument]:
        self.arguments.sort(key=lambda arg: arg.sort_key())
        return self.arguments

    def _cute_pass_knob_by_offset_var(self, key: str) -> dict[str, object]:
        """``{loop offset variable name: knob value}`` for a per-device-loop knob.

        The bridge between the two CuTe AST-pass knobs (``cute_online_defer``,
        ``cute_tv_sweep_cache``) and the passes that consume them.  The knobs are
        registered per device-loop BLOCK ID, but the passes run on emitted source
        where a block id no longer exists -- only the loop's induction variable does.
        This resolves one to the other through the tile strategies, which are the
        single owner of that naming (``TileStrategy.offset_var``).

        Resolved HERE rather than inside each pass so the passes stay pure AST
        functions of their arguments: unit-testable on a hand-written body, with no
        ``CompileEnvironment`` and no ``Config`` in scope.

        Missing slots are simply absent from the result: the specs' ``_fill_missing``
        supplies the ladder value when a config omits the key, so an absent entry
        means "this loop has no slot at all" and every pass treats that as its
        default.  It is never a silent off.
        """
        env = CompileEnvironment.current()
        spec_seq = getattr(env.config_spec, key, None)
        if spec_seq is None or not len(spec_seq):
            return {}
        values = self.config.config.get(key)
        if not isinstance(values, (list, tuple)):
            return {}
        out: dict[str, object] = {}
        for strategy in self.tile_strategy.strategies:
            for block_id in strategy.block_ids:
                # ``config_get`` maps block_id -> flat slot index, and returns the
                # sentinel when the block owns no slot.  ``_MISSING`` rather than
                # ``None`` because ``None`` is a legal value for other knobs of this
                # shape and would be indistinguishable from "no slot".
                value = spec_seq.config_get(list(values), block_id, _MISSING)
                if value is _MISSING:
                    continue
                try:
                    offset_var = strategy.offset_var(block_id)
                except NotImplementedError:
                    # ``FlattenedTileStrategy`` has no per-block offset variable.
                    # Its blocks are flattened into one launch axis, so they are not
                    # device loops and neither pass can fire on them.
                    continue
                out[offset_var] = value
        return out

    def codegen_function_def(self) -> list[ast.stmt]:
        prefix = []
        if self._tensor_descriptor_args:
            prefix.append(
                statement_from_string("helion.runtime.set_triton_allocator()")
            )

        backend = CompileEnvironment.current().backend
        sorted_arguments = self.sorted_args()

        # Separate constexpr args: inline those with known literal values at
        # module level, keep dynamic ones as function parameters
        constexpr_to_inline = [
            arg
            for arg in sorted_arguments
            if isinstance(arg, ConstExprArg) and _is_literal_constexpr(arg)
        ]
        inlined_names = {arg.name for arg in constexpr_to_inline}
        param_args = [
            arg
            for arg in sorted_arguments
            if not isinstance(arg, ConstExprArg) or arg.name not in inlined_names
        ]

        args = [arg.arg_def_node() for arg in param_args]
        # Ordering invariant:
        # [param_args, extra_params, rng_seed, scratch_args, wrapper_only_params].
        # codegen_function_call must match this order — it builds positional args
        # from param_args, extends with extra_params, then build_launcher_args
        # appends rng_seed_buffer.
        args.extend(create_arg(name) for name in self.codegen._extra_params)
        if self.has_rng_ops():
            # Add the seed buffer as a pointer parameter to kernel signature
            assert self.rng_seed_buffer_param_name is not None
            args.append(create_arg(self.rng_seed_buffer_param_name))

        # Add scratch memory parameters (for emit_pipeline on Pallas/TPU)
        for scratch_arg in self._scratch_args:
            args.append(create_arg(scratch_arg.name))
        args.extend(create_arg(name) for name in self.wrapper_only_params)

        # Generate inlined constexpr assignments at module level
        # (e.g., _BLOCK_SIZE_0 = tl.constexpr(256))
        # Use SyntheticLocation to suppress source origin comments on these statements
        with SyntheticLocation():
            for arg in constexpr_to_inline:
                self.codegen.module_statements.append(
                    statement_from_string(
                        backend.inline_constexpr(arg.name, arg.host_str())
                    )
                )

        # Generate preamble to dereference scalar refs (e.g., Pallas 0-dim tensors)
        scalar_preamble: list[ast.AST] = []
        for arg in param_args:
            scalar_preamble.extend(backend.scalar_arg_preamble(arg))

        function_decorator = backend.function_decorator_for_args(param_args)
        kernel_body: list[ast.stmt] = cast(
            "list[ast.stmt]",
            [
                *scalar_preamble,
                *self.preamble,
                *self.body,
            ],
        )
        if backend.name == "cute":
            from .cute.fuse_two_pass_loads import fuse_two_pass_loads

            # Collect static integer values for constexpr names so the
            # fusion pass can resolve range(..., step=cutlass.Int32(NAME))
            # trip counts. Three sources: literal constexpr inlined args,
            # host-side literal assignments to constexpr-named variables,
            # and the inlined module-level constexpr decls.
            constexpr_values: dict[str, int] = {}
            for arg in constexpr_to_inline:
                try:
                    value = int(arg.host_str())
                except (TypeError, ValueError):
                    continue
                constexpr_values[arg.name] = value
            for stmt in self.codegen.host_statements:
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, int)
                ):
                    constexpr_values[stmt.targets[0].id] = stmt.value.value
            # Pass the per-axis thread dims so the fuser can size
            # SMEM-backed caches correctly and build a per-thread
            # linear slot index when ``cache_size`` exceeds the
            # register-fragment threshold (opt-in via
            # ``HELION_FUSER_MODE=smem``).
            try:
                thread_dims = self.tile_strategy.thread_block_dims()
                thread_block_dims: tuple[int, int, int] = (
                    int(thread_dims[0]),
                    int(thread_dims[1]),
                    int(thread_dims[2]),
                )
            except Exception:
                thread_block_dims = (1, 1, 1)
            kernel_body = fuse_two_pass_loads(
                kernel_body,
                constexpr_values,
                thread_block_dims=thread_block_dims,
            )
            # The same idea for the TV-layout path, whose gmem read is a
            # ``cute.copy`` STATEMENT into a register fragment rather than an
            # assignment of ``ptr.load()`` -- so ``fuse_two_pass_loads``
            # structurally cannot see it.  ``layer_norm`` needs ``mean`` before
            # ``centered = x - mean``, so it sweeps the row THREE times and
            # re-reads ``x`` from gmem each time; quack reads it ONCE and
            # reloads from registers (``reload_from=None``).  This caches the
            # fragment lanes across the sweeps so the later ones hit registers.
            # MEASURED at 32768x1024 (LEDGER E055): 6 copies -> 4, ratio vs
            # quack 0.900 -> 1.066.
            #
            # The per-thread register budget it may spend is the
            # ``cute_tv_sweep_cache`` knob (``CuteTvSweepCacheSpec``), keyed by the
            # sweep loop's offset variable; ``0`` declines.  A loop with no slot is
            # simply absent from the map, and the pass falls back to its own
            # default for those -- so an empty map is today's behaviour exactly.
            # ⭐ ``cute_row_residency=["gmem"]`` VETOES the cache by zeroing its budget.
            # ``gmem`` means "no mechanism -- sweep 2 re-reads global", so it must
            # suppress the register cache; expressed as a budget of 0 because that is
            # already the pass's own spelling for "decline this loop", so the veto adds
            # no second mechanism.  Only an EXPLICIT ``gmem`` vetoes -- a refused
            # ``smem`` request still falls back to the cache exactly as it does today
            # (see ``cute_row_residency_forbids_sweep_cache`` for the measurement).
            #
            # Resolved through the SAME ``_cute_pass_knob_by_offset_var`` bridge the
            # budget itself uses, rather than by asking each strategy: that helper
            # already owns block-id -> offset-var naming and already skips a strategy
            # with no per-block offset variable.  It reads the key POST-NORMALIZE, so
            # the legacy derivation (``_normalize_cute_row_residency``) has already
            # run and this sees the one authoritative residency per block.
            residency_veto = {
                offset_var
                for offset_var, value in self._cute_pass_knob_by_offset_var(
                    "cute_row_residency"
                ).items()
                if value == "gmem"
            }
            # ⭐ AND THE SAME VETO, ASKED OF THE STRATEGIES THAT RESOLVED A RESIDENCY
            # WITHOUT OWNING A CONFIG SLOT.  The bridge above reads the config KEY, so it
            # can only see a block that owns a ``cute_row_residency`` slot -- and a TILED
            # reduction axis owns none (those slots are registered per REDUCTION BLOCK by
            # ``DeviceIR._register_cute_tv_layout_slots``).  MEASURED on a tile-regime
            # two-sweep kernel whose strategy resolved ``gmem``: the bridge returned ``{}``,
            # the veto never fired, and ``fuse_tv_copy_sweeps`` emitted 10 ``_tv_sweep_cache``
            # statements -- i.e. ``gmem`` and ``registers`` compiled to the SAME kernel, which
            # is precisely the collapse ``cute_row_residency_forbids_sweep_cache`` exists to
            # prevent (it is the one thing that makes ``gmem`` a distinct value rather than a
            # synonym).
            #
            # ⚠ ASKED, NOT INFERRED, and asked of the capability rather than the class:
            # ``cute_row_residency_forbids_sweep_cache`` is on ``TileStrategy``, so every
            # strategy can answer and one that never resolved a residency answers False.
            # ⚠ A UNION, never a replacement: the config-key path stays authoritative for
            # every block that DOES own a slot, so this cannot un-veto a rolled reduction --
            # it only adds loops the key could not name.  That keeps the edit inert wherever
            # the old path already had an answer.
            #
            # ⛔⛔ AND IT MUST ASK "WAS A RESIDENCY *ASKED FOR*", NOT ONLY "IS IT gmem".
            # ``cute_row_residency_forbids_sweep_cache()`` is ``requested == "gmem"``, and
            # ``gmem`` is ALSO the base-class default -- it means "no mechanism", which is the
            # honest answer for a strategy that never participated in the axis.  So the
            # predicate alone cannot tell an EXPLICIT ``gmem`` (which must veto, because that
            # is the only thing making ``gmem`` a distinct kernel rather than a synonym for
            # ``registers``) from an ABSENT request (which must NOT veto, or every kernel in
            # the tree silently loses ``fuse_tv_copy_sweeps``).
            #
            # That distinction is exactly what ``cute_row_residency_forbids_sweep_cache``'s own
            # docstring is about: it vetoes on ``gmem`` ALONE and the asymmetry is measured, not
            # tidiness -- vetoing on a REFUSED ``smem`` request would suppress the fallback that
            # is today's behaviour for every cell whose staging is declined.  The same argument
            # applies one step out: vetoing on a DEFAULT ``gmem`` would suppress it everywhere.
            #
            # ⇒ the request must be one this strategy actually RESOLVED, which is what the
            # per-block record answers.  A strategy that never resolved one is skipped.
            #
            # ⭐⭐ ASKED OF THE STRATEGY, NOT REIMPLEMENTED HERE (task 4, FIXLIST item 4).
            # This loop used to inline the predicate itself -- ``if value != "gmem":
            # continue`` over ``getattr(strategy, "_cute_row_residency_requested_by_block",
            # None)`` -- while ``TileStrategy.cute_row_residency_forbids_sweep_cache``, whose
            # docstring is the SPECIFICATION for this veto, was NEVER CALLED (measured: one
            # grep hit, the ``def``).  So the veto had two live implementations and one dead
            # specification, which is item 4's complaint in its sharpest form: a documented
            # decision site that decides nothing.  ``cute_row_residency_veto_offset_vars``
            # now RETURNS the answer, so the docstring governs the code that runs and the
            # provenance rule ("was a residency actually RESOLVED") lives in one place.
            for strategy in self.tile_strategy.strategies:
                residency_veto |= strategy.cute_row_residency_veto_offset_vars()
            # ⭐ THE VETO IS THE ONLY THING THIS SITE HANDS THE PASS (task 1 steps 2-3).
            #
            # It used to also pass ``slot_budgets`` -- the ``cute_tv_sweep_cache`` per-thread
            # register budget -- and that knob is DELETED.  A budget could make the
            # ``registers`` arm decline, i.e. a *performance* ceiling could turn a config's
            # named residency into a different emitted one; MEASURED on the frozen table,
            # that is how 13 of 40 cells came to record a residency the kernel never used.
            #
            # ⚠ AND THE VETO IS NOW ITS OWN CHANNEL RATHER THAN ``budget=0``.  Encoding "the
            # user asked for no mechanism" as a quantity of zero is what let
            # ``HELION_TV_SWEEP_FUSE_SLOTS`` silently un-veto three explicit-``gmem`` cells
            # (measured): the override replaced the whole map, zero included.  A set cannot be
            # raised by an override that only knows about numbers.
            # ⭐⭐ ``fuse_tv_copy_sweeps`` IS GONE.  The ``registers`` residency is now decided
            # and emitted AT LOWERING -- ``memory_ops._cute_tv_partition_hoist``'s rmem branch
            # publishes the first read's lanes into a per-thread cache and serves every later
            # read of the SAME TILE from it, with no gmem copy.
            #
            # ⛔ DELETED BECAUSE NOTHING REACHED IT, not because something reimplemented it.
            # MEASURED before removal, on two independent populations:
            #   * the 40 FROZEN CELLS  -- 40/40 compile, 40 pass calls, **0 fusions**;
            #   * 10 cute test files   -- 709 passed, 798 pass calls, **0 fusions**.
            # The last survivor was a two-multi-read-tensor kernel where ``smem`` staging takes
            # one tensor and refused the other; the lowering arm now picks that one up too.
            #
            # ⚠ ``residency_veto`` above is retained: it is still the channel by which an
            # explicit ``gmem`` request suppresses the register mechanism.
            # Hoist warp reductions out of constexpr V-loops to collapse
            # 4 per-V-lane warp reductions into 1 V-fold + 1 warp reduce.
            # For online softmax style kernels this drops per-row reductions
            # from ~396 to ~99 (4x fewer SHFL trees).
            from .cute.hoist_warp_reduce import hoist_warp_reduce_from_vloop

            kernel_body = hoist_warp_reduce_from_vloop(
                kernel_body, running_sum_accumulators=self.cute_matmul_running_sums
            )
            # Merge adjacent constexpr V-loops that share an identical
            # statement prefix.  Caches the last common per-V-lane value
            # into a register fragment so V-loop 2's bitcast/cast chain
            # disappears and the SASS scheduler can issue V-loop 2's
            # arithmetic without waiting for V-loop 1's results.
            from .cute.merge_sibling_v_loops import merge_sibling_v_loops

            kernel_body = merge_sibling_v_loops(kernel_body)
            # Fuse adjacent per-lane fp8 decodes in the SIMT matmul V-loop
            # into one ``cvt.rn.f16x2.e4m3x2`` (decode 2 e4m3 bytes per
            # instruction) — halves the decode instruction count on the
            # skinny-M fp8 GEMV path.
            from .cute.fuse_fp8_pair_decode import fuse_fp8_pair_decode

            kernel_body = fuse_fp8_pair_decode(kernel_body)
            # Hoist loop-invariant floating-point divisions out of inner
            # tile loops, replacing each ``x / scalar`` with a hoisted
            # ``inv = 1.0 / scalar`` + ``x * inv`` in the loop body.
            # B200 div is ~22 cycles vs ~2 for multiply, so the softmax
            # consume sweep (~12672 divides per row) sees a measured
            # +20% bench gain on (4096, 12672) fp16.
            from .cute.hoist_loop_invariant_recip import hoist_loop_invariant_recips

            # Pass the post-renames map so the invariance analysis can
            # treat ``v_1_0`` (which will be renamed to ``mi`` by
            # ast_rename below) as an assignment to ``mi`` for the
            # purpose of LICM.  Without this the FMA hoist would
            # mistakenly classify ``mi`` as loop-invariant in the reduce
            # loop and capture its stale initial value.
            rename_groups = {k: v[0] for k, v in self._variable_renames.items()}
            kernel_body = hoist_loop_invariant_recips(
                kernel_body, rename_groups=rename_groups
            )
            # P18: software-pipeline the per-iteration vec load by one
            # stage.  Pre-issue iter 0's load above the loop and, inside
            # the body, issue iter N+1's load BEFORE iter N's compute
            # runs.  The B200 SASS scheduler can then keep multiple
            # ld.global instructions in flight, hiding HBM round-trip
            # latency on softmax/online-reduction inner loops where the
            # ``load -> compute(mi, di) -> next iter`` sequential
            # dependency chain dominates the per-iter stall budget.
            from .cute.pipeline_inner_loads import pipeline_inner_loads

            # Pass the post-rename canonical map so the loop-carried-write
            # gate can correctly identify writes whose pre-rename target
            # is an alias (e.g. ``v_1_0 = v_1`` will be renamed to
            # ``mi = v_1``).  Without this, the gate would mis-classify
            # the softmax reduce sweep as having no loop-carried write
            # and incorrectly skip pipelining.
            kernel_body = pipeline_inner_loads(
                kernel_body, constexpr_values, rename_groups=rename_groups
            )
            # ⭐ THE ONE canonical row-residency statement, emitted LAST so it is a
            # function of the FINAL body.  It must come after ``fuse_tv_copy_sweeps``
            # in particular: that pass is what decides ``registers`` vs ``gmem`` (it
            # can still decline on its register budget), so a marker written any
            # earlier could only say "gmem/registers" and name two values where the
            # reader needs one.  Reading the emitted instructions is what makes
            # "stated == observed" true by construction.
            #
            # ⚠ UNDER ``SyntheticLocation``, and that is not cosmetic.  A statement
            # built outside it inherits whatever source location was current, so the
            # unparser emits a ``# src[...]`` origin comment for it AND shifts the
            # neighbouring annotations -- which turns a comment-only addition into a
            # multi-line textual diff on every reduction cell.  MEASURED: without this
            # the 40-cell hash gate reports 29 changed cells whose entire diff is
            # comment noise.  The marker describes the whole kernel and has no source
            # line, so a synthetic location is also the honest answer.
            from .cute.memory_ops import cute_emit_row_residency_marker

            with SyntheticLocation():
                residency_marker = cute_emit_row_residency_marker(self, kernel_body)
            if residency_marker is not None:
                self.codegen.module_statements.append(residency_marker)
        return [
            *prefix,
            ast_rename(
                create(
                    ast.FunctionDef,
                    name=self.name,
                    args=create_arguments(args),
                    body=kernel_body,
                    decorator_list=[expr_from_string(function_decorator)]
                    if function_decorator
                    else [],
                    type_params=[],
                ),
                {k: v[0] for k, v in self._variable_renames.items()},
            ),
        ]

    def codegen_function_call(self) -> ast.AST:
        env = CompileEnvironment.current()
        backend = env.backend

        args: list[str] = []
        tensor_host_args: list[str] = []
        arg_objects: list[Argument] = []
        for arg in self.sorted_args():
            # Skip constexpr args that are inlined at module level
            if isinstance(arg, ConstExprArg) and _is_literal_constexpr(arg):
                continue
            if isinstance(arg, ConstExprArg) and arg.name in self._constexpr_host_defs:
                host_arg = arg.name
            else:
                host_arg = arg.host_str()
            if isinstance(arg, TensorArg):
                tensor_host_args.append(host_arg)
            host_arg = backend.transform_host_arg(arg, host_arg, tensor_host_args)
            args.append(host_arg)
            arg_objects.append(arg)

        pid = self.pid
        assert pid is not None

        call_grid_expr = pid.codegen_grid()
        # ── the CLUSTER launch (PORT_SPEC_layout.md §9) ─────────────────────────
        #
        # A clustered row reduction splits ONE row across ``cluster_n`` CTAs, which
        # quack expresses as ``grid=[ceil_div(M, tiler_m), cluster_n, num_heads]`` plus
        # ``cluster=[1, cluster_n, 1]`` (``rmsnorm.py:137-142``).  The ``cluster=``
        # half already flows end-to-end via ``cute_state.cluster_shape``
        # (``generate_ast.py`` -> ``runtime/__init__.py:3292``); only grid.y was
        # missing, so this widens the grid tuple to match.
        #
        # ⚠ THE LAUNCH-FAILURE / HANG CONSTRAINT.  CUDA requires the grid to be a
        # multiple of the cluster shape.  Making grid.y *be* ``cluster_n`` satisfies
        # that identically rather than by arithmetic luck -- there is no rounding here
        # that could leave a partial cluster, which is the case that fails to launch
        # (or, with a barrier in flight, hangs).  The assert makes that structural
        # rather than incidental.
        cluster_shape = self._cute_state.cluster_shape
        if (
            cluster_shape is not None
            and env.backend.name == "cute"
            and cluster_shape[0] == 1
            and cluster_shape[2] == 1
            and cluster_shape[1] > 1
        ):
            cluster_n = cluster_shape[1]
            grid_src = ast.unparse(call_grid_expr)
            # Only the single-axis grid shape is handled; anything else already uses
            # grid.y for its own purpose and must not have the cluster folded in.
            if grid_src.endswith(",)"):
                call_grid_expr = expr_from_string(f"({grid_src[1:-2]}, {cluster_n}, 1)")
        # Extra params are positional and must come before any keyword args that
        # build_launcher_args appends (e.g. num_warps=, num_stages=).
        args.extend(self.codegen._extra_params)
        call_args = backend.build_launcher_args(
            args,
            tensor_host_args=tensor_host_args,
            has_rng_ops=self.has_rng_ops(),
            config=self.config,
            has_barrier=env.has_barrier,
            sorted_args=arg_objects,
        )
        # Check if the backend wants to capture return values for output-only tensors.
        output_only_names = getattr(backend, "_output_only_names", [])
        launcher_call = (
            f"_launcher({self.name}, {{call_grid_expr}}, {', '.join(call_args)})"
        )
        if output_only_names:
            if len(output_only_names) == 1:
                assign_target = output_only_names[0]
            else:
                assign_target = ", ".join(output_only_names)
            call_statement = statement_from_string(
                f"{assign_target} = {launcher_call}",
                call_grid_expr=call_grid_expr,
            )
        else:
            call_statement = statement_from_string(
                launcher_call,
                call_grid_expr=call_grid_expr,
            )
        assert isinstance(call_statement, ExtendedAST)
        # Mark the kernel call so we can find it in codegen_precompile_def
        call_statement._is_kernel_call = True
        return call_statement

    def dead_code_elimination(self) -> None:
        """
        Remove variables that are not used in the function body.
        """

        for _ in range(8):
            rw = ReadWrites.from_list([*self.preamble, *self.body])
            dead_assignment_elimination(self.body, self.dce_vars, 1, rw)
            dead_assignment_elimination(self.preamble, self.dce_vars, 1, rw)
            dead_lane_loop_elimination(self.body)
            dead_lane_loop_elimination(self.preamble)
        rw = ReadWrites.from_list([*self.preamble, *self.body])

        # Drop unused args, but keep placeholder_args (fusion-injected tensor
        # pointers referenced only by placeholder strings, not the AST body).
        # sourceless_prologue_params are intentionally NOT exempted — they are
        # fully inlined by the prologue hook and should be removed by DCE.
        args_to_remove = {
            arg.name
            for arg in self.arguments
            # pyrefly: ignore [unbound-name]
            if arg.name not in rw.reads and arg.name not in self.placeholder_args
        }
        if args_to_remove:
            self.arguments = [
                arg for arg in self.arguments if arg.name not in args_to_remove
            ]
            for cache in cast(
                "list[dict[object, Argument]]",
                [
                    self._tensor_args,
                    self._tensor_descriptor_args,
                    self._expr_args,
                    self._tensor_properties,
                ],
            ):
                for k, v in [*cache.items()]:
                    if v.name in args_to_remove:
                        del cache[k]

    def register_helper_function(
        self, helper_graph_info: HelperFunctionGraphInfo
    ) -> None:
        """Register a helper function to be generated at global scope."""
        name = self.namespace.create_name(helper_graph_info.name, None)
        self.helper_manager.register_helper_function(helper_graph_info, name)

    def codegen_helper_functions(self) -> list[ast.stmt]:
        """Generate helper function definitions at global scope."""
        return self.helper_manager.codegen_helper_functions()

    def flush_deferred_rdim_defs(self, codegen: GenerateAST) -> None:
        """Add all deferred RDIM definitions to host statements."""
        backend = CompileEnvironment.current().backend
        for var_name, expr in self.deferred_rdim_defs:
            expr_str = HostFunction.current().sympy_expr(expr)
            stmt = statement_from_string(
                f"{var_name} = {backend.dynamic_rdim_size_expr(expr_str)}"
            )
            codegen.host_statements.append(stmt)
        self.deferred_rdim_defs.clear()

    def register_scratch(
        self,
        shape: tuple[int | torch.SymInt, ...],
        dtype: torch.dtype | None,
        name_hint: str = "scratch",
        scratch_type: str = "vmem",
    ) -> str:
        """Register a scratch memory buffer and return its variable name."""
        if CompileEnvironment.current().backend_name != "pallas":
            raise NotImplementedError(
                "register_scratch is only supported by the Pallas backend"
            )
        name = self.new_var(name_hint)
        self._scratch_args.append(ScratchArg(name, shape, dtype, scratch_type))
        return name

    def scratch_read_slice(self, name: str) -> str | None:
        """Return the index expression for reading logical data from a padded scratch.

        Returns None if no padding was applied.
        """
        return None

    def register_dma_semaphore(
        self, name_hint: str = "sem", shape: tuple[int, ...] = ()
    ) -> str:
        """Register a DMA semaphore scratch buffer and return its variable name."""
        return self.register_scratch(
            shape, None, name_hint=name_hint, scratch_type="dma_semaphore"
        )

    def get_tensor_read_write_names(self) -> tuple[set[str], set[str]]:
        """Returns AST names of read and written tensors"""
        from helion.language import memory_ops
        from helion.language import tile_index
        from helion.language.atomic_ops import ATOMIC_OPS

        read_names: set[str] = set()
        write_names: set[str] = set()
        for graph in self.codegen.codegen_graphs:
            for node in graph.graph.nodes:
                if node.op != "call_function":
                    continue

                def _get_tensor_name(node: torch.fx.Node) -> str | None:
                    tensor_arg = node.args[0]
                    assert isinstance(tensor_arg, torch.fx.Node)
                    # tile.index loads operate on a synthesized FakeTensor
                    # that is not registered in ``tensor_to_origin``; they
                    # are materialized inline by the load codegen rather
                    # than referencing a kernel-arg tensor.
                    if (
                        tensor_arg.op == "call_function"
                        and tensor_arg.target == tile_index
                    ):
                        return None
                    tensor_val = tensor_arg.meta.get("val")
                    assert isinstance(tensor_val, torch.Tensor)
                    return self.tensor_arg(tensor_val).name

                if node.target is memory_ops.load:
                    name = _get_tensor_name(node)
                    if name is not None:
                        read_names.add(name)
                elif node.target is memory_ops.store:
                    name = _get_tensor_name(node)
                    if name is not None:
                        write_names.add(name)
                elif node.target in ATOMIC_OPS:
                    name = _get_tensor_name(node)
                    if name is not None:
                        read_names.add(name)
                        write_names.add(name)
        return read_names, write_names

    def __enter__(self) -> None:
        try:
            tls.functions.append(self)
        except AttributeError:
            tls.functions = [self]

    def __exit__(self, *args: object) -> None:
        tls.functions.pop()

    @staticmethod
    def current() -> DeviceFunction:
        try:
            return tls.functions[-1]
        except (AttributeError, IndexError):
            raise NoCurrentFunction from None
