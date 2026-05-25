# Helion Reduction Heuristics — v1 Implementation Plan

**Author context:** Standalone plan. A fresh agent should be able to execute
this without reading the three companion files (`helion_autotuner_overview.md`,
`triton_reduction_heuristics.md`, `list_of_kernels.md`). Those files can still
be given to an agent as optional background, but they are not prerequisites:
all required implementation steps, benchmark shapes, split discipline, and
source-code oracle references are duplicated here.

**Note on the low-level code in §5–§6:** the snippets reflect verified Helion
APIs as of this writing, but several signatures are subtle and change
periodically — e.g. `ConfigSpec.normalize` returns `None` (mutates in place,
not "returns the dict"); `BlockIdSequence` exposes `block_id_lookup`, not
`get_spec`; `BlockSizeSpec` inherits `_PowerOfTwoBlockIdItem` whose
`_normalize` raises `InvalidConfig` on non-pow2 values; fragments expose
`default` as an attribute, not a method on all spec types. **Treat the code
as algorithmic intent: verify every API call against the current source
before pasting, and adjust as needed.** The load-bearing part of §6 is the
recipe formulas and constants, not the surrounding glue.

**Hardware target:** NVIDIA H100, `sm90`, `fp32`.
**Baseline to beat:** Helion's current `default_config()` (P0).
**Stretch (INNER-classified kernels only — see §1):** match or approach
`torch.compile` on most shapes (P1); outperform everywhere (P2).

**Repo variables (set these for the implementation environment):**

- Helion: set `HELION_REPO` to the Helion clone path.
- PyTorch (Inductor oracle): set `PYTORCH_REPO` to the PyTorch clone path.
- Worktree destination: set `WORKTREE_ROOT` to a directory where git
  worktrees may be created, and `HELION_WORKTREE` to
  `$WORKTREE_ROOT/helion-reduction-heuristics`.
- Logs/cache: set `LOG_DIR` and `CACHE_ROOT` to ignored scratch directories.

---

## 0. GitHub / File Management Plan

**Before any code changes, create a dedicated Helion worktree.** The agent
doing the implementation should make this worktree, not require the user to do
it manually. The currently checked-out Helion branch is irrelevant for this
project; do not base the work on it unless the user explicitly says it contains
needed prerequisites. Base the stack on `origin/main` by default.

Recommended setup:

```bash
export HELION_REPO="<helion-clone>"
export PYTORCH_REPO="<pytorch-clone>"
export WORKTREE_ROOT="<worktree-root>"
export HELION_WORKTREE="$WORKTREE_ROOT/helion-reduction-heuristics"
export LOG_DIR="<ignored-log-dir>"
export CACHE_ROOT="<ignored-cache-root>"

cd "$HELION_REPO"
git fetch origin fork
git worktree add -b rh/compiler-seeds-effort-none \
  "$HELION_WORKTREE" \
  origin/main
cd "$HELION_WORKTREE"
```

Push branches to the fork remote:

```bash
git push -u fork rh/compiler-seeds-effort-none
```

**Use a shallow stacked PR series.** This project has reviewable boundaries;
stacking keeps each PR small without hiding the dependency chain.

1. **PR 1 — compiler seed plumbing**
   - Branch: `rh/compiler-seeds-effort-none`
   - Scope: `HELION_USE_COMPILER_SEEDS`, the three `effort=none` call sites,
     seed normalization/fallback, and targeted tests.
   - Expected behavior: default-off, so CI and user-visible behavior should
     remain unchanged unless the flag is set.

2. **PR 2 — reduction fact collection**
   - Branch: `rh/reduction-facts`, based on PR 1.
   - Scope: `ReductionHint`, `ReductionFact`, fact population, and focused
     tests for sum, cross-entropy counts, and non-power-of-two shapes.
   - Keep this separate from PR 3. The fact bag is the correctness contract
     for the heuristic: eligibility, INNER/OUTER classification, `r_size_hint`
     vs `static_rnumel`, `x_block_id` pairing, and load/store/reduction counts
     should be reviewed and tested before performance recipes consume them.
   - Expected behavior: inert without heuristic registration.

3. **PR 3 — Triton reduction heuristic**
   - Branch: `rh/triton-reduction-heuristic`, based on PR 2.
   - Scope: `TritonReductionHeuristic`, INNER recipes A/B, heuristic
     registration, valid seed-config tests, and first end-to-end validation on
     `sum_kernel`.

4. **PR 4 — benchmark refinement / enablement decision**
   - Branch: `rh/reduction-heuristic-benchmarks`, based on PR 3.
   - Scope: recipe refinements from train/validation results, any durable
     benchmark helpers worth checking in, and the PR report with validation
     and final test-set numbers. Do not flip the seed flag default-on until a
     later PR after the benchmark evidence is solid.

**Commit discipline:** commit frequently at working, reviewable checkpoints.
Good commit boundaries are: helper added + tested; one call-site family wired
and tested; fact type added; fact populator passes `sum`; cross-entropy count
test passes; non-pow2 fact test passes; heuristic emits a valid config; one
kernel validates end-to-end. Avoid giant end-of-project commits, but also avoid
commits that do not import, compile, or pass their focused tests unless they
are explicitly marked temporary and immediately followed by the fix before
pushing for review.

**File hygiene:**
- Keep code edits inside the dedicated Helion worktree.
- Treat `$PYTORCH_REPO` as read-only oracle material.
- The planning docs (`reduction_heuristics_plan.md`, `list_of_kernels.md`,
  etc.) may live outside the Helion repo. Do not assume edits to them are part
  of a GitHub PR unless intentionally copied into the Helion worktree or
  summarized in a PR description.
- Put raw benchmark logs in `$LOG_DIR` or another ignored location. Commit only
  durable scripts, small fixtures, or summarized results.
- Use a fresh `HELION_CACHE_DIR` per benchmark iteration as described in §7.2.

---

## 1. TL;DR

Port Inductor's `_persistent_reduction_configs` + `_reduction_configs` recipes
into Helion as a new compile-time `AutotunerHeuristic` (`TritonReductionHeuristic`).
The heuristic reads a new `ReductionFact` (mirroring the existing `MatmulFact`)
that is populated during `register_rollable_reductions`. For `effort=none`, the
seed becomes the actual launch config; for `quick`/`full`, it seeds the autotuner's
initial population.

**v1 scope is INNER-classified reductions only.** The seven v1 kernels
(rms_norm, layer_norm, softmax variants, sum, cross_entropy, longsum) all
reduce over a stride-1 last dim and classify as `ReductionHint.INNER`. The
`is_eligible` filter restricts the heuristic to `hint == INNER` for v1; for
`OUTER` and `DEFAULT` hints the kernel falls back to `default_config()`
exactly as it does today. Recipes C (OUTER) and D (DEFAULT) are still
sketched in §6 because the populator emits the hint and we want the recipe
to be ready when v2 adds OUTER-bearing kernels — but they are **not
exercised by the v1 kernel set, not benchmarked, and not landing-gated**.

v2 expands the eligibility filter to OUTER+DEFAULT (with Recipe C/D
benchmarking), to user-tiled reductions (kl_div, jsd, welford), and to
backward kernels.

---

## 2. Problem Statement

### 2.1 What's wrong with `default_config()` today

`ConfigSpec.default_config()` (`helion/autotuner/config_spec.py:2566`) calls
`flat_config(lambda x: x.default())` — every fragment returns its own default,
independent of kernel shape. For reductions:

- `ReductionLoopSpec._flat_fragment` (`config_spec.py:2983–2995`) defaults to
  `min(next_power_of_2(size_hint), 4096)`. For a 32k-wide reduction (cross
  entropy), this picks `R0_BLOCK = 4096` and never explores `R0_BLOCK = 2048`
  even though Inductor caps `MAX_R0_BLOCK` at 2048 on H100 specifically to
  avoid register spills.
- `num_warps` defaults to `4` (`config_spec.py:235`, `DEFAULT_NUM_WARPS`).
  Inductor's rule is `num_warps = R0_BLOCK // 128` (i.e. 16 for `R0_BLOCK = 2048`).
  A 4×–8× warps deficit is typical.
- `block_sizes[m_block_id]` defaults to a fragment middle, not to the
  reduction-paired `XBLOCK` Inductor would pick (e.g. `XBLOCK = 8` for
  `rnumel = 256` persistent).
- For small persistent inner reductions (`rnumel <= 256`), Inductor uses
  `num_warps = 1` (warp-level reduction, no shared memory). Helion's default
  always crosses warp boundaries.

Under `effort=none`, kernels launch with this `default_config()` directly. Under
`quick`/`full`, the autotuner starts from `default_flat()` (`config_generation.py:492`),
so even with autotuning the search starts from a bad point and may not converge
in budget.

### 2.2 Why heuristic seeds help

Inductor's reduction heuristics encode 5+ years of empirical tuning across
thousands of kernels. They are not "the optimum" but they are a strong prior:
within a small constant of optimum on most shapes, much better than any
shape-agnostic default. Porting them gives Helion a baseline floor that is
already close enough to optimum that autotuning becomes a refinement step
rather than a discovery step.

### 2.3 Scope deliberately excluded from v1

| Excluded | Why | Plan |
|---|---|---|
| User-tiled reductions (kl_div, jsd, welford, softmax_two_pass) | Different config knob: `BlockSizeSpec` (shared with other ops), not `ReductionLoopSpec`. Needs separate fact populator + recipe. | v2 sub-track (§9). |
| Backward kernels (rms_norm_bwd, layer_norm_bwd) | Two reductions per kernel (per-row N + cross-CTA M for param grad). Needs multi-fact handling. | v2 sub-track (§9). |
| Cooperative reductions (Inductor's `cooperative_reduction`) | Helion has no emit path. Tiny-x huge-r SM starvation requires manual `hl.barrier()` source rewrite. | Out of scope permanently. |
| TMA filter (Inductor's `_maybe_filter_configs_for_tma_restrictions`) | Helion **does** emit TMA loads (via `indexing="tensor_descriptor"`). When a load violates the 16-byte minimum, `TensorDescriptorIndexingStrategy.codegen_load` (`indexing_strategy.py:574-585`) falls back to `PointerIndexingStrategy` — kernel still compiles and runs. So TMA is a *performance* concern, not a correctness one. v1 Recipe C applies an XBLOCK floor to preserve the TMA option for fp32 OUTER kernels (see §5.3). | No filter port needed; recipes preserve TMA reachability as a performance optimization. |
| Split reductions / `OUTER_TINY` hint | Inductor synthesizes these via `Reduction.num_splits` multilayer logic. Helion has no equivalent. | Skip — never classified as `OUTER_TINY`. |

---

## 3. Architectural Background

### 3.1 Helion's reduction config space

Two relevant knobs per reduction dim:

| Knob | Type | Owner | Lives in |
|---|---|---|---|
| `reduction_loops[r_block_id]` | `int \| None` | Compiler-managed reductions (the roller fires) | `config_spec.reduction_loops: BlockIdSequence[ReductionLoopSpec]` |
| `block_sizes[block_id]` | `int` (power of two) | User-written `hl.tile(...)` loops | `config_spec.block_sizes: BlockIdSequence[BlockSizeSpec]` |

**Critical asymmetry:** a compiler-allocated reduction `block_id` puts a
`BlockSizeInfo` into `env.block_sizes` (the list, not the spec) but does
**not** put a `BlockSizeSpec` into `config_spec.block_sizes`. So for compiler-
managed reductions, `reduction_loops` is the only per-axis tunable knob.

`ReductionLoopSpec` (`config_spec.py:2973`) encoding:

```python
# autotuner picks an integer in [low=8, high=next_power_of_2(size_hint)]
if value >= self.size_hint:
    return None   # persistent reduction
return value      # looped, R0_BLOCK = value
```

So `reduction_loops[r_block_id] = None` ↔ persistent kernel; any int ↔ looped
with `R0_BLOCK = value`.

### 3.2 Reduction strategy selection

`Backend.create_reduction_strategy` (`helion/_compiler/backend.py:631-647`):

```python
if reduction_loop is None:
    return PersistentReductionStrategy(fn, block_id)
return LoopedReductionStrategy(fn, block_id, reduction_loop)
```

Selection is purely a function of the picked `reduction_loops[i]` value. There
is no separate codegen decision — the config drives lowering. (Contrast
Inductor where lowering picks the decorator before config gen.)

### 3.3 Where reductions become tunable

`register_rollable_reductions` (`helion/_compiler/device_ir.py:748-826`)
runs the `ReductionRoller` (`helion/_compiler/roll_reduction.py`) as analysis
only. For each reduction dim it can roll, it appends a `ReductionLoopSpec` to
`env.config_spec.reduction_loops`. If the roller bails (matmul-fed K, indirect
indexing, etc.), no spec is registered and the reduction is forced persistent
over the full rnumel with no tunable knob.

### 3.4 Fact-bag pattern (the model to follow)

`MatmulFact` (`helion/autotuner/config_spec.py:160-172`) is a NamedTuple that
matmul lowering writes into `config_spec.matmul_facts` (one per matmul op).
The `TritonSkinnyGemmHeuristic` (`helion/_compiler/autotuner_heuristics/triton.py:18-94`)
reads `facts = env.config_spec.matmul_facts`, checks eligibility, and returns
a `Config(block_sizes=[...])` seed. We mirror this exactly for reductions.

### 3.5 Seed-config flow

```
KernelCompiler.compile  (runtime/kernel.py:484)
    ↓
env.config_spec built; reduction_loops, block_sizes populated
    ↓
compiler_seed_configs(env, device_ir)  (runtime/kernel.py:495)
    ↓
    for each heuristic in HEURISTICS_BY_BACKEND[env.backend_name]:
        if heuristic.is_eligible(env, device_ir):
            configs.append(heuristic.get_seed_config(env, device_ir))
    ↓
env.config_spec.compiler_seed_configs = [...]
    ↓
    autotuner reads seeds via autotuner/config_generation.py:407
    and injects into initial population
    ↓
effort=none currently BYPASSES seeds — falls back to default_config()
at backend.py:676, kernel.py:986, kernel.py:1003   ← Step 1 closes this
```

---

## 4. Inductor → Helion Mapping (Cheat Sheet)

| Inductor concept | Helion equivalent | Notes |
|---|---|---|
| `@persistent_reduction` decorator | `reduction_loops[i] = None` (→ `PersistentReductionStrategy`) | Config-driven, not lowering-driven |
| `@reduction` (looped) decorator | `reduction_loops[i] = R0_BLOCK` (→ `LoopedReductionStrategy`) | |
| `R0_BLOCK` value | Same — the integer value of `reduction_loops[i]` | |
| `XBLOCK` | `block_sizes[m_block_id]` of the paired parallel `hl.tile` axis | |
| `num_warps` | `num_warps` config field | Same semantics |
| `num_stages` | `num_stages` config field | Inductor pins `num_stages=1` for reductions |
| `ReductionHint` (INNER/OUTER/OUTER_TINY/DEFAULT) | **New: `ReductionHint` enum on Helion side** | Net-new; closest precedent is `reduction_loops` polarity but it doesn't encode coalescing |
| `size_hints["x"]`, `size_hints["r0_"]` | `BlockSizeInfo.size_hint()` for the m and r block_ids | |
| `inductor_meta["num_load"/"num_reduction"/"num_store"]` | **New: count by walking reduction's FX subgraph** | Mirror the count Inductor builds in codegen |
| `Reduction.num_splits` stride classifier | **New: read `fake_tensor.stride()[rdim_axis] == 1` directly** | Trivial in Helion (§5.2) |
| `MAX_R0_BLOCK = 1024 if device_major >= 10 else 2048` | Same logic; H100 → 2048 | Read via `helion._hardware.get_hardware_info()` |
| Register-intensive trigger: `x>=1024 and num_load+num_reduction>=10` | Same logic: lowers `MAX_R0_BLOCK` to 1024 AND halves `NW_MAX` (see `_compute_limits` in §6.0). Inductor at `triton_heuristics.py:3834-3856` + `:3087-3096` | |
| Persistent base threshold `INNER<=1024, else <=64` | Same logic; controls hint dispatch in heuristic | |
| `cooperative_reduction` | **None — out of scope** | Helion has no equivalent emit path |
| TMA filter (per-config) | **Per-load fallback to pointer** at `indexing_strategy.py:574-585` | Helion DOES emit TMA. Unsupported loads degrade to pointer codegen automatically — performance concern, not correctness. Heuristic preserves TMA reachability via XBLOCK floors (§5.3). |
| `filter_reduction_configs_for_determinism` | **Defer to v2** | Helion has no `deterministic` config knob yet |

---

## 5. Implementation Plan

### Step 1: Wire `effort=none` to read `compiler_seed_configs` *(P0 prerequisite, gated)*

**Goal:** Make seeds usable as the default launch config when no autotuning
runs. This is the fast iteration loop.

**Gating with `HELION_USE_COMPILER_SEEDS` (opt-in):** Wire the new behavior
behind an env var so the PR does *not* change `effort=none` behavior for any
existing user by default. Only when `HELION_USE_COMPILER_SEEDS=1` does the
effort=none path consult `compiler_seed_configs`; otherwise it falls back to
`default_config()` exactly as it does today.

Rationale: this dramatically shrinks the blast radius of the wiring change.
`.expected` golden files don't shift unless someone opts in, CI stays green,
and we can develop / benchmark with the flag set without forcing the new path
on every user the moment the PR lands. Flip the default to "on" in a separate,
clean PR once the heuristic is mature.

**Edits (3 sites, same pattern):**

```python
# Helper: add to existing helion/runtime/settings.py (do NOT create a new
# _env.py — settings.py is the established home for runtime env-var reads).
def _use_compiler_seeds() -> bool:
    import os
    return os.environ.get("HELION_USE_COMPILER_SEEDS", "0") == "1"

# Before
config = ...config_spec.default_config()
# After
seeds = ...config_spec.compiler_seed_configs
if _use_compiler_seeds() and seeds:
    try:
        # Validate the seed up front. Bad recipe seeds (non-pow2
        # reduction_loops, value >= size_hint, missing block_sizes
        # entries, etc.) would otherwise raise InvalidConfig deep in
        # BoundKernel.to_code at kernel.py:610 and surface as a user-
        # facing compile failure. Catching here keeps effort=none safe.
        config = ...config_spec.normalize(seeds[0], _fix_invalid=False)
    except InvalidConfig as e:
        log.warning(
            "compiler_seed_configs[0] failed normalize for kernel %s: %s; "
            "falling back to default_config()", kernel_name, e,
        )
        config = ...config_spec.default_config()
else:
    config = ...config_spec.default_config()
```

- [ ] Add `_use_compiler_seeds()` to `helion/runtime/settings.py`; import at
  all 3 sites.
- [ ] `helion/_compiler/backend.py:676` — inside `Backend.autotune` when
  `autotune_effort == "none"`.
- [ ] `helion/runtime/kernel.py:986` — `_fixed_config_for_td_layout_guards`.
- [ ] `helion/runtime/kernel.py:1003` — `_user_provided_config`.

**Why the explicit normalize:** `compiler_seed_configs`
(`__init__.py:38-54`) only catches exceptions thrown while *building* a
seed; it does not validate the returned `Config`. The autotune path
(`config_generation.py:407-424`) has its own try/except that drops bad
seeds, but the `effort=none` path bypasses that and would otherwise raise
on bad seeds at `to_code` time. The proactive `normalize` here matches the
autotune path's tolerance: bad seed → warn + fall back, never crash.

**Verification:**
1. **Default-off check:** without `HELION_USE_COMPILER_SEEDS=1`, run any
   kernel under `HELION_AUTOTUNE_EFFORT=none`. Codegen must be byte-identical
   to pre-PR behavior. Re-run a handful of existing `.expected` tests to
   confirm.
2. **Opt-in check:** with `HELION_USE_COMPILER_SEEDS=1`, pick any kernel that
   today matches `TritonSkinnyGemmHeuristic` (skinny GEMM, sm90/gfx950) and
   run under `HELION_AUTOTUNE_EFFORT=none`. Confirm via
   `HELION_PRINT_OUTPUT_CODE=1` that the launch config matches the heuristic's
   seed, not `default_config()`.
3. **Opt-in, no-heuristic-match:** with the flag on but without
   `TritonReductionHeuristic` registered, *non*-matmul kernels under
   `effort=none` should still launch with `default_config()` (empty seeds list
   → falls back).

**Test breakage to expect:** none in CI by default — the flag defaults to off.
When *running tests with the flag on*, `.expected` golden files that pin
`default_config()` codegen for kernels matching a seed-producing heuristic
will shift. Iterate on those test files individually with
`pytest test/<file> -k <name> -x -vv -s 2>&1 | tee "$LOG_DIR/pytest.out"`.

**Do NOT** run the full suite under `HELION_AUTOTUNE_EFFORT=none` — Helion's
`CLAUDE.md` explicitly warns this breaks tests by changing execution paths.

**Ranking (when multiple heuristics seed):** `compiler_seed_configs`
(`helion/_compiler/autotuner_heuristics/__init__.py:33-56`) just appends seeds
today. For `effort=none`, picking `seeds[0]` is fine for v1 (only one reduction
heuristic registered, plus the matmul skinny-gemm one which has disjoint
eligibility). If a kernel ever matches multiple, add a `priority: int` class
attribute and sort before slicing.

**Flag lifecycle:** the flag is a development scaffold, not a permanent API.
Once v1's recipes hit P1 across the v1 kernel set, flip the default to "on" in
a follow-up PR (and update the affected `.expected` files in that same PR).
The flag itself can stay as a kill-switch for one more release, then be
removed.

---

### Step 2: Introduce `ReductionHint` enum + `ReductionFact` + populator

**Goal:** Provide the heuristic the same scalar/enum signals Inductor's Stage B
consumes (hint, counts, dtype, static shapes, rollable flag).

#### 2a. Add `ReductionHint` enum

New module: `helion/_compiler/reduction_hint.py`

```python
from __future__ import annotations
from enum import Enum

class ReductionHint(Enum):
    """Classifies a compiler-managed reduction by load-stride coalescing.

    Mirrors torch._inductor.runtime.hints.ReductionHint but is independent
    so we can evolve semantics if needed.
    """
    INNER = 0     # reduced dim is stride-1 along all rdim loads
    OUTER = 1     # reduced dim is NOT stride-1 (typically stride-M)
    DEFAULT = 2   # mixed or unclassifiable
```

`OUTER_TINY` is omitted: Inductor only assigns it via split-reduction
synthesis (which Helion doesn't have).

#### 2b. Add `ReductionFact` NamedTuple

Insert into `helion/autotuner/config_spec.py` immediately after `MatmulFact`
(around line 173):

```python
class ReductionFact(NamedTuple):
    """Shape facts recorded when a rollable reduction is registered.

    One entry per (registered) ReductionLoopSpec. Compiler-managed
    reductions only; user-tiled reductions (BlockReductionStrategy)
    are not in this list.

    Three distinct "R" values are tracked because they have different
    semantics and conflating them produces silent miscompiles:

    - static_rnumel: the actual reduction extent. May be non-power-of-two
      (e.g. 1023, 2047). This is what the kernel actually reduces over.
    - r_size_hint: the value `ReductionLoopSpec` uses for its persistent
      decode boundary. Per `config_spec.py:3009`, `if value >= size_hint:
      return None`. So any r0_block we emit MUST be strictly less than
      r_size_hint, or the flat-config autotune path silently reinterprets
      it as persistent. Typically `r_size_hint = next_power_of_2(rnumel)`,
      but read it from the live `ReductionLoopSpec`, do not recompute.
    - effective r0_block for looped: computed in the recipe (§6.2/6.3).
      Must be a power of two (else `_PowerOfTwoBlockIdItem` raises
      InvalidConfig) and strictly less than r_size_hint.
    """
    r_block_id: int                # reduction dim block_id
    x_block_id: int | None         # paired parallel hl.tile dim (None if absent)
    static_rnumel: int | None      # int if reduction dim is statically known
    r_size_hint: int               # ReductionLoopSpec.size_hint (pow2-rounded)
    static_xnumel: int | None      # int if parallel dim is statically known
    hint: ReductionHint            # INNER/OUTER/DEFAULT
    num_load: int                  # whole-kernel load count (matches Inductor)
    num_reduction: int             # whole-kernel reduction count (matches Inductor)
    num_store: int                 # whole-kernel store count (matches Inductor)
    accum_dtype: torch.dtype       # accumulator dtype (Inductor's src_dtype)
    is_rollable: bool              # True iff r_block_id appears in reduction_loops
```

And initialize in `ConfigSpec.__init__` next to `matmul_facts`
(`config_spec.py:519`):

```python
self.reduction_facts: list[ReductionFact] = []
```

#### 2c. Populate facts after `register_rollable_reductions`

`register_rollable_reductions` runs at `helion/_compiler/device_ir.py:748`,
called from `_to_device_ir` at line 2207. Add a sibling method
`register_reduction_facts(self) -> None` and call it immediately after:

```python
# helion/_compiler/device_ir.py — near line 2207
device_ir.register_rollable_reductions()
device_ir.register_reduction_facts()   # ← new
```

The populator walks each reduction dim that's now in
`env.config_spec.reduction_loops`, scans the relevant graphs for
`helion_language_memory_ops_load` / `store` nodes and `ReductionLowering`
nodes, and emits one `ReductionFact` per dim.

**Stride classification (the INNER/OUTER vote):** see §5.2.

**Counts (whole-kernel, matching Inductor):** these are *whole-kernel
counts*, not "ops touching the rdim". Inductor increments num_load /
num_store / num_reduction on **every** load, store, and reduction the
generated kernel performs, with no axis check (see
`pytorch/torch/_inductor/codegen/common.py:2860, 2879, 2893, 2903`).
Reproducing this is load-bearing because the `register_intensive`
threshold is `num_load + num_reduction >= 10` over the whole kernel
(see `triton_heuristics.py:3835-3842`).

Extend the existing `_count_device_loads_and_stores` walk at
`helion/_compiler/device_ir.py:1994-2041`:

- `num_load`: count *all* `node.target is memory_ops.load` nodes in the
  kernel body. Do not filter by axis.
- `num_store`: count *all* `node.target is memory_ops.store` nodes,
  including output stores along the X axis (Inductor counts these too —
  reduction outputs flow through `store_reduction` at `common.py:2888-2894`
  which bumps `num_store`).
- `num_reduction`: count *all* nodes whose lowering is a `ReductionLowering`
  (`helion/_compiler/inductor_lowering.py:645`). Uniform check via
  `isinstance(node.meta.get("lowering"), ReductionLowering)` catches
  sum / amax / mean / var_mean / Welford / argmax / etc. For
  `cross_entropy` this correctly yields `num_reduction = 2` per row
  (amax + sum), pushing it past the `>= 10` register-intensive boundary
  along with its load count.

**Avoid double-counting:** `register_rollable_reductions` clones the
reduction body into a `ReductionLoopGraphInfo` subgraph. The counting
walk needs to skip those clones (or count parent vs subgraph and pick
one), else the reductions/loads get counted twice. ~10 LOC of filter.

**Semantic difference from Inductor (worth a code comment):** Inductor
counts at op-handler emission time, so its CSE folds duplicate loads of
the same tile into one count. Helion's walk is per-FX-node post-DCE, so
kernels that re-load the same tile will report slightly higher load
counts than Inductor would. For the `>= 10` threshold this is fine
(rankings are preserved), but flag it in a comment.

**`static_rnumel`/`static_xnumel`:** from `BlockSizeInfo.size_hint()`. Set
`static_rnumel = int(numel)` when `numel` is a `sympy.Integer`, else `None`.

**`r_size_hint`:** read directly from `ReductionLoopSpec.size_hint` for the
rdim's block_id — do NOT recompute. This is the pow2-rounded value that
governs the persistent decode boundary (`config_spec.py:3009`).

**`accum_dtype`:** read from the reduced load's `node.meta["val"].dtype`.

**`x_block_id`:** the m-side (parallel) tile axis. Walk the same graph for
`hl.tile`-emitted block_ids that index the same tensors as the rdim's loads.
For row-reduction kernels this is unambiguous (one outer `hl.tile(m)`); for
ambiguous cases return `None`.

**Defensive:** wrap the whole populator in a try/except that logs and clears
the fact list — never crash compilation. Heuristics already tolerate a missing
fact bag (eligibility just returns False).

**Verification:** add a unit test
`test/test_reduction_facts.py::test_sum_kernel_facts` that compiles `sum_kernel`
on shape (2048, 1024) with `HELION_AUTOTUNE_EFFORT=none` and asserts:
- `len(env.config_spec.reduction_facts) == 1`
- `facts[0].static_rnumel == 1024 and facts[0].static_xnumel == 2048`
- `facts[0].r_size_hint == 1024` (already a power of two)
- `facts[0].hint == ReductionHint.INNER`
- `facts[0].is_rollable is True`
- `facts[0].num_load == 1 and facts[0].num_reduction == 1 and facts[0].num_store == 1`
  (these are whole-kernel counts; sum_kernel happens to have exactly
  one of each, but the assertion is about the count, not "ops touching rdim")

Add a second test for `cross_entropy` on `(4096, 4096)` asserting
`num_reduction == 2` (amax + sum per row) to lock in the whole-kernel
counting semantics. Also add a non-pow2 case: `sum_kernel` on `(2048, 1023)`
asserting `r_size_hint == 1024` (so the recipe can safely emit `r0_block = 512`
without hitting the persistent decode boundary).

### Step 5.2: How to classify INNER vs OUTER from `fake_tensor.stride()`

For each load node feeding the reduction:
1. Resolve its host tensor: `host_tensor_node = load_node.args[0]` →
   `fake = host_tensor_node.meta["val"]` is a `torch.Tensor` with
   `.shape` and `.stride()`.
2. Find the rdim axis: walk `enumerate(fake.size())` and find the position
   `i` where `env.get_block_id(fake.size()[i]) == rdim_block_id`. If no
   axis matches, this load isn't touching the rdim — skip.
3. Vote: `inner_vote += 1` if `fake.stride()[i] == 1`, else `outer_vote += 1`.

Aggregate:
- `num_inner > num_outer` → `ReductionHint.INNER`
- `num_outer > num_inner` → `ReductionHint.OUTER`
- equal (including 0/0) → `ReductionHint.DEFAULT`

This is the Helion equivalent of `Reduction.num_splits`
(`pytorch/torch/_inductor/ir.py:1537-1561`) but skips all the sympy machinery
because the strides are concretely known on the fake input tensor.

ORACLE: `pytorch/torch/_inductor/ir.py:1355` for the original logic; also
`pytorch/torch/_inductor/sizevars.py:1250` for `stride_hints` (not needed in
Helion — we read strides directly).

### Step 5.3: TMA (tensor_descriptor) interaction — performance-only, not correctness

Helion emits TMA loads via the `indexing="tensor_descriptor"` config knob
(see `helion/_compiler/indexing_strategy.py:228` for the strategy dispatch,
`:443-572` for `TensorDescriptorIndexingStrategy`, and `:497-499` for the
16-byte minimum). The constraint is enforced **per-load at codegen-time**,
not as a post-filter on the config pool.

**Correctness is not at risk** even if a recipe emits a sub-16-byte XBLOCK.
`TensorDescriptorIndexingStrategy.codegen_load` at `indexing_strategy.py:574-585`
falls back to `PointerIndexingStrategy().codegen_load(...)` when the load
is unsupported. So the kernel still compiles and runs correctly — it just
silently degrades to a pointer load on that site, losing the TMA path.

**Performance is at risk.** When the autotuner selects
`indexing="tensor_descriptor"` globally, individual loads that violate the
16-byte rule become pointer loads while their siblings remain TMA. We lose
the TMA win on the violating loads without losing the config.

**Implication for our recipes (fp32, H100):**
- 16 bytes ÷ 4 bytes/fp32 element = **4-element minimum** on the
  contiguous axis.
- Recipe A (persistent INNER, R ≤ 1024): the contiguous axis is N (= rdim).
  Block along N is full R (≥ 256 in the small-persistent branch); always
  ≥ 4 elements. **TMA reachable on input loads.**
- Recipe B (looped INNER, R > 1024): N still contiguous, R0_BLOCK = 2048
  or smaller pow2 ≥ 1024. **TMA reachable on input loads.**
- Recipe C (OUTER): M is contiguous, XBLOCK ∈ {2, 8, 16, ...}. Minimum
  XBLOCK = 2 (fp32: 8 bytes — **below the 16-byte rule**, so TMA degrades
  to pointer on the M-axis loads). Recipe C applies a `min_xblock_for_tma`
  floor (XBLOCK ≥ 4 for fp32) to *preserve the TMA option* — this is a
  performance optimization, not a correctness requirement.

**For future fp16/bf16 expansion:** 16-byte minimum becomes 8 elements.
Recipe A persistent branch with XBLOCK ∈ {1, 8} on a contiguous M axis
would have XBLOCK=1 silently degrade. Add a dtype-aware floor in the
recipe.

**Heuristic doesn't need to filter `indexing`.** Leave it unset in the
seed Config — autotuner explores `("pointer", "tensor_descriptor",
"block_ptr")` per load and skips invalid combos.

---

### Step 3: Implement `TritonReductionHeuristic`

New file: `helion/_compiler/autotuner_heuristics/triton_reduction.py`

```python
from __future__ import annotations
from typing import TYPE_CHECKING
from ...runtime.config import Config
from ..reduction_hint import ReductionHint
from .common import matches_hardware
from .registry import AutotunerHeuristic

if TYPE_CHECKING:
    from ..compile_environment import CompileEnvironment
    from ..device_ir import DeviceIR


class TritonReductionHeuristic(AutotunerHeuristic):
    name = "triton_reduction"
    backend = "triton"
    # v1: sm90 only. Recipes in §6 were derived from H100 (`MAX_R0_BLOCK=2048`,
    # 132 SMs, 16-byte TMA floor). Do NOT add sm100 until v1 hits its P1 target
    # on sm90 — Hopper vs Blackwell differ in SM count, register file, TMA
    # capabilities, and likely warp-count sweet spots, so the recipe constants
    # would need re-tuning before they can be trusted on sm100.
    HARDWARE_TARGETS = (("cuda", "sm90"),)

    @classmethod
    def is_eligible(cls, env, device_ir):
        if not matches_hardware(env, cls.HARDWARE_TARGETS):
            return False
        facts = env.config_spec.reduction_facts
        # v1: exactly one rollable reduction with static dims
        if len(facts) != 1:
            return False
        fact = facts[0]
        # v1 scope: INNER hint only. OUTER and DEFAULT are deliberately
        # excluded because Recipe C/D are not benchmarked against the v1
        # kernel set (which is all-INNER). Returning False here means
        # the kernel falls back to default_config() — same behavior as
        # pre-PR. v2 will lift this restriction once Recipe C/D land
        # with their own OUTER-bearing benchmark coverage.
        if fact.hint is not ReductionHint.INNER:
            return False
        return (
            fact.is_rollable
            and fact.static_rnumel is not None
            and fact.x_block_id is not None
            and fact.static_xnumel is not None
        )

    @classmethod
    def get_seed_config(cls, env, device_ir):
        fact = env.config_spec.reduction_facts[0]
        recipe = _select_recipe(env, fact)
        return _emit_config(env, fact, recipe)
```

`_select_recipe` and `_emit_config` apply the recipes in §6.

**`_emit_config` must build a complete `block_sizes` list, not partial.**
Unlike scalar knobs and `reduction_loops` (which fill in via `setdefault` /
`_fill_missing → None`), `BlockSizeSpec` inherits `_BlockIdItem._fill_missing`
which raises `NotImplementedError` — and `BlockIdSequence._normalize`
(`block_id_sequence.py:222-232`) re-raises that as `InvalidConfig`. So if
the kernel has e.g. two `hl.tile` axes and the seed only sets one entry of
`block_sizes`, normalize fails and the seed is dropped.

Emit `block_sizes` as a full list in `env.config_spec.block_sizes` order:

```python
def _emit_config(env, fact, recipe) -> Config:
    spec_block_ids = [b.block_id for b in env.config_spec.block_sizes]
    block_sizes: list[int] = []
    for block_id in spec_block_ids:
        if block_id == fact.x_block_id:
            block_sizes.append(recipe.xblock)
        else:
            # use the spec's own default for this axis (typically the dim's
            # full size or a small constant; we don't have an opinion on
            # axes the heuristic isn't targeting)
            spec = env.config_spec.block_sizes.get_spec(block_id)
            block_sizes.append(spec.default())

    return Config(
        block_sizes=block_sizes,
        reduction_loops=recipe.reduction_loops,
        num_warps=recipe.num_warps,
        num_stages=1,
    )
```

Register the heuristic in
`helion/_compiler/autotuner_heuristics/__init__.py`:

```python
from .triton_reduction import TritonReductionHeuristic
HEURISTICS_BY_BACKEND["triton"] = (
    TritonSkinnyGemmHeuristic,
    TritonReductionHeuristic,
)
```

---

### Step 4: Validate end-to-end on a single kernel

Pick `sum_kernel` (the simplest target — no epilogue, no second pass) and run
the full triple-signal loop from §7. Land Step 1+2+3 + this validation as a
single PR before expanding.

---

## 6. Per-Knob Porting Recipes (the actual heuristic logic)

**Notation:** `R = static_rnumel`, `X = static_xnumel`, `MAX_R0_BLOCK` per §6.0.

### 6.0 H100/sm90 constants and per-fact derived limits

```python
def _compute_limits(fact: ReductionFact) -> tuple[int, int, int, bool]:
    """Returns (MAX_R0_BLOCK, NW_MIN, NW_MAX, register_intensive).

    Mirrors triton_heuristics.py:3834-3856:
      register_intensive = False
      MAX_R0_BLOCK = 1024 if major >= 10 else 2048   # Blackwell → 1024
      if size_hints["x"] >= 1024 and (num_load + num_reduction) >= 10:
          MAX_R0_BLOCK = 1024
          register_intensive = True

    On H100 (sm90), MAX_R0_BLOCK starts at 2048 and drops to 1024 under
    register pressure. register_intensive ALSO halves NW_MAX (see
    `_num_warps` at triton_heuristics.py:3087-3096:
    `max_num_warps = max_num_warps // 2` when register_intensive).
    """
    # H100/sm90 (v1 hardware target):
    MAX_R0_BLOCK = 2048
    NW_MAX = 16 if (fact.static_rnumel or 0) <= 8192 else 32
    NW_MIN = 2

    register_intensive = (
        (fact.static_xnumel or 0) >= 1024
        and (fact.num_load + fact.num_reduction) >= 10
    )
    if register_intensive:
        MAX_R0_BLOCK = 1024
        NW_MAX = max(NW_MIN, NW_MAX // 2)

    return MAX_R0_BLOCK, NW_MIN, NW_MAX, register_intensive
```

ORACLE: `pytorch/torch/_inductor/runtime/triton_heuristics.py:3834-3856` for
the `MAX_R0_BLOCK` switch + `register_intensive` trigger, and `:3087-3096`
for the `_num_warps` halving.

### 6.1 Recipe A — INNER persistent (R ≤ 1024)

```python
MAX_R0_BLOCK, NW_MIN, NW_MAX, register_intensive = _compute_limits(fact)
R = fact.static_rnumel
X = fact.static_xnumel

# reduction kind
reduction_loops = [None]      # persistent

# XBLOCK selection — from triton_heuristics.py:4444-4470
if R >= 256 and R <= 1024 and X // 8 >= 128:
    # CUDA small-persistent-inner branch
    xblock = min(1024 // R, 8)        # clamp to 8
    num_warps = 1                     # warp-level reduction, no smem
else:
    # General persistent sweep: pick the largest XBLOCK with R*XBLOCK <= 4096
    candidates = [128, 32, 8, 1]
    for cand in candidates:
        if cand <= X and R * cand <= 4096:
            xblock = cand
            break
    else:
        xblock = 1
    # num_warps via Inductor formula (persistent: full R contributes to tile)
    tile = xblock * R
    num_warps = _next_pow2_clamped(tile // 128, NW_MIN, NW_MAX)
```

ORACLE: `pytorch/torch/_inductor/runtime/triton_heuristics.py:4347-4477` for
`_persistent_reduction_configs` and the INNER pruning.

### 6.2 Recipe B — INNER looped (R > 1024)

The looped r0_block has to satisfy two Helion invariants that don't exist
in Inductor:

1. **Power of two** — `ReductionLoopSpec` extends `_PowerOfTwoBlockIdItem`
   (`config_spec.py:2973`), and `_normalize` calls `assert_integer_power_of_two`
   on the value (`block_id_sequence.py:256-263`). Non-pow2 values raise
   `InvalidConfig`.
2. **Strictly less than `r_size_hint`** — flat-config decode at
   `config_spec.py:3009` reads any `value >= self.size_hint` as `None`
   (persistent). So if we emit `r0_block == r_size_hint`, the
   autotune-seed-rank path silently converts our "looped" seed into a
   persistent one, and we benchmark a different config than the recipe
   intended.

```python
MAX_R0_BLOCK, NW_MIN, NW_MAX, register_intensive = _compute_limits(fact)
R = fact.static_rnumel

# Compute the largest legal looped r0_block:
#   - pow2 (Helion invariant 1)
#   - <= MAX_R0_BLOCK (register pressure cap)
#   - < r_size_hint (Helion invariant 2 — else decoded as persistent)
pow2_cap = min(_prev_pow2(R), MAX_R0_BLOCK)
if pow2_cap >= fact.r_size_hint:
    pow2_cap //= 2     # back off to keep looped
r0_block = max(pow2_cap, 1)
reduction_loops = [r0_block]

# contiguous_config XBLOCK selection
xblock = 2 if R <= 2048 else 1

# num_warps for INNER hint: r // 128, where r is the CAPPED reduction block,
# NOT the full R. Inductor passes `min(rnumel, MAX_R0_BLOCK)` to
# triton_config_reduction (triton_heuristics.py:3956) and num_warps formula
# at :3320 uses that capped value. Using full R systematically over-warps
# long reductions (e.g. R=130k would request 32 warps instead of 16).
num_warps = _next_pow2_clamped(r0_block // 128, NW_MIN, NW_MAX)
```

Worked examples (verify on paper before coding):

| R | r_size_hint | _prev_pow2(R) | pow2_cap | r0_block | xblock | num_warps |
|---|---|---|---|---|---|---|
| 1024 | 1024 | 1024 | 1024 | 512 (backed off) | 2 | 4 |
| 2047 | 2048 | 1024 | 1024 | 1024 | 2 | 8 |
| 2048 | 2048 | 2048 | 2048 | 1024 (backed off) | 2 | 8 |
| 4096 | 4096 | 4096 | 2048 | 2048 | 1 | 16 |
| 8192 | 8192 | 8192 | 2048 | 2048 | 1 | 16 |
| 130000 | 131072 | 65536 | 2048 | 2048 | 1 | 16 |

ORACLE: `_reduction_configs` and `contiguous_config` at
`pytorch/torch/_inductor/runtime/triton_heuristics.py:3956-3960` for the
`min(rnumel, MAX_R0_BLOCK)` pattern, and `:3320` for the
`num_warps = r // 128` rule on the capped r.

### 6.3 Recipe C — OUTER looped (`outer_config_opt` port)

**v1 status: SKETCH ONLY — not gated by `is_eligible` in v1.** The v1
kernel set is all-INNER, so this recipe is not exercised end-to-end and
not benchmarked. It is written here so that v2 (which adds OUTER-bearing
kernels) can light it up by relaxing the `is_eligible` hint filter. Until
that happens, treat the code below as a design draft — the formulas
follow Inductor's `outer_config_opt` but no v1 validation backs them up.

When enabled, apply only for `hint == OUTER`. Complex multi-branch logic;
for v2 implement the small/medium static-`X` branches first and fall back
to Recipe B for large `X`. Defer dynamic-shape branches.

```python
MAX_R0_BLOCK, NW_MIN, NW_MAX, register_intensive = _compute_limits(fact)
R, X = fact.static_rnumel, fact.static_xnumel

# Translated from outer_config_opt at triton_heuristics.py:3908+:
if X <= 1024:
    xblock = max(min(X // 128, 8), 2)
    rblock = min(R, 64)
elif X // 4096 <= 8:
    xblock = 16
    rblock = 512 // xblock
else:
    xblock_unclamped = _next_pow2_clamped(X // 4096, 1, 256)
    xblock = xblock_unclamped
    rblock = 512 // xblock

# TMA-reachability floor (§5.3, performance-only — not correctness):
# Helion's TensorDescriptorIndexingStrategy.codegen_load falls back to
# PointerIndexingStrategy when the load is unsupported
# (indexing_strategy.py:574-585), so the kernel will still compile and
# run if we emit a sub-floor XBLOCK; we just give up TMA. Apply this
# floor anyway to preserve the autotuner's option to pick
# indexing="tensor_descriptor".
min_xblock_for_tma = max(1, 16 // fact.accum_dtype.itemsize)   # fp32 -> 4
xblock = max(xblock, min_xblock_for_tma)

# r0_block must satisfy the same pow2 + size_hint rules as Recipe B:
pow2_rblock = min(_prev_pow2(rblock), MAX_R0_BLOCK)
if pow2_rblock >= fact.r_size_hint:
    pow2_rblock //= 2
r0_block = max(pow2_rblock, 1)
reduction_loops = [r0_block]

# num_warps for non-INNER: tile_product // 128 with capped r
tile = xblock * r0_block
num_warps = _next_pow2_clamped(tile // 128, NW_MIN, NW_MAX)
```

ORACLE: `pytorch/torch/_inductor/runtime/triton_heuristics.py:3908+` for
the full `outer_config_opt` source, including the dynamic-shape branches
(`num_dynamic == 0/1/>1`) that we're not porting in v1.

### 6.4 Recipe D — DEFAULT hint

**v1 status: SKETCH ONLY — not gated by `is_eligible` in v1.** Like Recipe C,
this is written so v2 can light it up without re-deriving the strategy, but
v1's all-INNER kernel set never exercises it.

When enabled: fall back to Recipe B (INNER looped) and let the autotuner
refine. The DEFAULT hint means we couldn't make a confident classification;
pinning a recipe risks being worse than letting the search run. **Note that
Recipe B bakes in INNER-specific assumptions** (XBLOCK = 2 or 1 because N is
contiguous; warps clamped against `r0_block // 128`); for DEFAULT kernels
those assumptions may be wrong on the contiguous axis. v2 should benchmark
this fallback against an "emit no seed for DEFAULT" baseline before locking
the recipe choice in.

### 6.5 `num_stages`

Pin to `num_stages=1` in every emitted seed. **This matches Inductor's
default for reductions** (Inductor's `triton_config_reduction` at
`pytorch/torch/_inductor/runtime/triton_heuristics.py:3283` defaults
`num_stages=1`). Cost is near-zero; constrains autotuner slightly but
keeps seed deterministic.

### 6.6 What we deliberately don't pin

- `pid_type`, `indexing`, `atomic_indexing`, `num_sm_multiplier`,
  `loop_orders`, `flatten_loops`, `l2_groupings`, `range_*` — autotuner
  explores these.
- `epilogue_subtile`, `maxnreg` — not relevant for the v1 row-reduction
  kernels.

### 6.7 Helpers: power-of-two utilities

```python
def _next_pow2_clamped(value: int, lo: int, hi: int) -> int:
    """Match Inductor's `next_power_of_2(min(max(v, lo), hi))` semantics.

    Rounds UP to next power of two after clamping into [lo, hi]. Used by
    Inductor's `_num_warps` at triton_heuristics.py:3087-3096 — porting
    that rounding direction is load-bearing because rounding down (e.g.
    7 → 4 instead of 8) systematically under-warps INNER reductions and
    is the kind of off-by-power-of-two bug that's invisible in geomean
    but obvious in per-shape regressions.
    """
    value = max(value, lo)
    value = min(value, hi)
    if value <= 0:
        return lo
    if value & (value - 1) == 0:
        return value
    return 1 << value.bit_length()


def _prev_pow2(value: int) -> int:
    """Largest power of two ≤ value. Used for r0_block selection."""
    if value <= 0:
        return 1
    return 1 << (value.bit_length() - 1)
```

---

## 7. Benchmarking & Iteration Strategy

### 7.1 Three validation signals

| Signal | What it tells us | Cost |
|---|---|---|
| **(a) `effort=none` speedup vs `default_config()`** | "Heuristic beats the baseline floor" — primary fast signal | seconds per kernel |
| **(b) Heuristic-seed rank in `quick` autotune final population** | "Heuristic is near-optimal, not just lucky on these shapes" | minutes per kernel |
| **(c) `benchmarks/run.py` vs `torch.compile` on tritonbench shapes** | "We're competitive with the production alternative" — landing gate | minutes-to-hours per kernel |

**Iterate on (a) against train.** Use (b) as a sanity check before declaring a
recipe done. Gate landing on (c) crossing the P1 bar on validation (§7.4) and
on the test-set comparison in §7.7.

### 7.2 Real commands

#### Signal (a): direct `effort=none` benchmark

Write a small driver script (one per kernel) that times the kernel with
`HELION_AUTOTUNE_EFFORT=none` before vs after the heuristic is registered.
Pattern from `helion/pretuned_kernels/rms_norm/rms_norm.py:80-126` which
uses `triton.testing.do_bench`. Example skeleton:

```python
# helion/scripts/bench_reduction_heuristic.py (suggested location).
# IMPORT NOTE: Helion's `examples/` lives at the repo root (not inside the
# `helion/` package), so the import is `from examples.sum import ...`,
# NOT `from helion.examples.sum import ...`. This matches how
# benchmarks/run.py:413 (`KERNEL_MAPPINGS`) imports examples.
import torch, helion, triton
from examples.sum import sum_kernel

shapes = [(2048, 1024), (2048, 4096), (4096, 5120),
          (8192, 4096), (32768, 256)]  # train shapes from §10.2
for shape in shapes:
    x = torch.randn(*shape, device="cuda", dtype=torch.float32)
    # The kernel is run with HELION_AUTOTUNE_EFFORT=none + HELION_USE_COMPILER_SEEDS=1
    # in the env; the heuristic's seed becomes the launch config.
    ms = triton.testing.do_bench(lambda: sum_kernel(x))
    print(f"{shape}: {ms:.4f} ms")
```

Run with (cwd MUST be the helion repo root so `examples.sum` resolves):
```bash
mkdir -p "$LOG_DIR"
cd "$HELION_WORKTREE" && \
HELION_AUTOTUNE_EFFORT=none HELION_USE_COMPILER_SEEDS=1 HELION_PRINT_OUTPUT_CODE=1 \
    python scripts/bench_reduction_heuristic.py 2>&1 | tee "$LOG_DIR/bench_a.txt"
```

#### Signal (b): autotune-seed rank check

Use `HELION_LOGS=+autotune` to log the final-population fitness ranking:

```bash
mkdir -p "$LOG_DIR"
cd "$HELION_WORKTREE" && \
HELION_AUTOTUNE_EFFORT=quick HELION_USE_COMPILER_SEEDS=1 HELION_LOGS=+autotune \
    python scripts/bench_reduction_heuristic.py 2>&1 | tee "$LOG_DIR/bench_b.txt"
```

In the log output, locate the heuristic's seed config (printed by
`config_generation.py:419` if it failed to transfer, or by autotuner search
logs otherwise) and confirm it ranks in the top 25% of the final population.
If it ranks bottom half, the heuristic is wrong for that shape — debug
before landing.

#### Signal (c): tritonbench end-to-end

The Helion benchmarks entry point is
`benchmarks/run.py` under `$HELION_WORKTREE`. From
`helion/benchmarks/README.md`:

All commands below require `cwd = "$HELION_WORKTREE"` (the benchmarks runner
resolves operator modules relative to the repo root, and TritonBench operator
imports break from any other cwd).

```bash
cd "$HELION_WORKTREE"

# Single kernel, operator's default sweep (use as the train sweep).
# --precision fp32 is REQUIRED: TritonBench operator defaults vary
# (e.g., softmax DEFAULT_PRECISION="fp16" at
# helion/benchmarks/tritonbench/tritonbench/operators/softmax/operator.py:162).
# Without --precision fp32 we would benchmark a dtype the heuristic was
# never tuned for.
HELION_USE_COMPILER_SEEDS=1 \
    python benchmarks/run.py --precision fp32 --metrics speedup,accuracy --kernel sum

# Multiple kernels at once:
HELION_USE_COMPILER_SEEDS=1 \
    python benchmarks/run.py --precision fp32 --metrics speedup,accuracy \
    --kernel sum,rms_norm,layer_norm,softmax,cross_entropy

# Specific backend:
HELION_USE_COMPILER_SEEDS=1 \
    python benchmarks/run.py --precision fp32 --helion-backend triton \
    --metrics speedup,accuracy --kernel sum
```

The `KERNEL_MAPPINGS` table in `benchmarks/run.py:340-560` registers each
kernel. Reduction-relevant entries: `"sum"`, `"softmax"`, `"softmax-bwd"`,
`"rms_norm"`, `"rms_norm-bwd"`, `"layer_norm"`, `"cross_entropy"`, `"jsd"`,
`"kl_div"`, `"welford"`.

**Cache hygiene during iteration (load-bearing for honest Signal (a) numbers):**

Helion's compile cache keys on the kernel source + settings + config. When
iterating on the heuristic recipe, the **seed contents change** between runs
but the kernel source and settings often don't, so cached compile artifacts
from earlier recipe versions can be served instead of recompiling against
the new seed. Signal (a) numbers measured against a polluted cache look
"better than they are" because part of what's being timed is the launch of
a stale-config kernel.

**Per-iteration discipline:**

- Set a fresh cache dir per recipe iteration:
  `HELION_CACHE_DIR="$CACHE_ROOT/helion_cache_recipe_$(git rev-parse --short HEAD)"`
  is a clean default when `CACHE_ROOT` points at an ignored scratch directory
  — it changes whenever the recipe code commits change. Alternatively, clear
  the active Helion cache directory between runs.
- When comparing two recipe versions head-to-head, **always** use distinct
  cache dirs (one per version). Otherwise the "second" run is partly served
  from the "first" run's artifacts.
- When comparing two strategies under §7.7, do one full-cold run per
  strategy per shape (cleared cache) before recording the number. This is
  expensive but only happens once per strategy at the final comparison
  step; the cost is small relative to the cost of shipping the wrong
  strategy.
- Before publishing the §7.7 test-set number, verify it's reproducible
  cold: clear the cache, rerun, confirm the number doesn't shift by more
  than benchmark noise. If it does shift, something is non-deterministic
  and needs to be tracked down before the report is honest.

This is iteration hygiene, not a correctness requirement — but stale-cache
contamination of Signal (a) is by far the most common source of "I thought
this recipe was better but it isn't" confusion during dev.

**Kernels that don't exercise the heuristic out-of-the-box:**
- `long_sum` is **not** registered in `KERNEL_MAPPINGS` *and* the `longsum`
  variant ships with a hardcoded config (`examples/long_sum.py:47`,
  `reduction_loops=[None]`, `num_warps=32`, `num_stages=4`,
  `indexing="block_ptr"`). That fixed config bypasses heuristic seeding
  entirely. To exercise the heuristic on long-N reductions, write a
  hand-written driver that calls `longsum`/`longsum_w_red_loop` via a
  *separate* `@helion.kernel`-decorated wrapper without a `config=` arg
  (or strip the decorator locally for benchmarking). Note this in the
  per-kernel results.
- `cross_entropy` *does* fit v1: it has one rdim (V) with two reductions
  per row (amax + sum) — the populator emits `len(facts) == 1` with
  `num_reduction == 2`. The trailing `losses.mean()` at
  `examples/cross_entropy.py:79` is *outside* the `hl.tile(n)` loop
  (host-side post-kernel op), so it does NOT add a second fact.

Validation and test shapes are listed directly in §10.2. They are not in
tritonbench operators by default — pass them via the operator's shape-args or
extend the operator. For v1, run the operator with its default shapes during
dev iteration, then run a hand-written driver on the validation shapes and
(separately, once) the test shapes (same pattern as Signal (a)).

### 7.3 Train / validation / test shape splits

We will likely try **multiple heuristic strategies** (different recipe
families, different classifier cutoffs, different default-warp policies, etc.)
and need to pick the best one. That requires a **test set** the candidate
strategies have not been tuned or selected against. Use a three-way split:

| Split | Source | When touched | Purpose |
|---|---|---|---|
| **Train** | §10.2 `Train shapes` column | Every dev iteration (Signal (a) hill-climbing) | Recipe authoring + per-knob tuning |
| **Validation** | §10.2 `Validation shapes` column | After each substantial recipe change, on every candidate strategy | Generalization signal; the metric used to pick a candidate to ship |
| **Test** | §10.2 `Test shapes` column + any newly added held-out shapes | **Exactly once, at the very end of the entire project**, on the single chosen winning strategy | Final report number; does NOT influence which strategy ships |

The split in §10.2 is the contract. Do not relabel shapes after heuristic
development begins unless you are deliberately declaring the previous test set
contaminated and replacing it with a new held-out set.

**Discipline:**
- Train shapes can be iterated on freely.
- Validation shapes can be inspected as often as you want during dev. The
  **winning candidate strategy is chosen using validation results, not test
  results.** All cross-strategy comparison happens on validation.
- **The test set is touched exactly once, at the end of the whole project**,
  after every candidate has been refined against train+validation and a
  single winner has been picked. The test-set run is a *report number*,
  not a decision input. If the test number is disappointing, you do not
  get to revise the heuristic and re-run — that would contaminate the test
  set. (If the test number is so bad it blocks shipping, the honest move is
  to designate a fresh held-out set as the new test, then go back to
  refining against the now-augmented validation pool.)
- **Bucket "tiny-M" shapes separately.** Very small `M` (or `BT` for
  cross_entropy) — roughly `M < 16` — shifts the kernel into a
  launch-dominated / latency-bound regime where kernel-launch overhead and
  fixed per-call costs swamp the compute that any heuristic config knob
  controls. Those shapes should be present in the bench sweeps as a
  **correctness check** (heuristic must not crash, must not emit
  `InvalidConfig`, must produce numerically correct output) but should
  **not** be used as tuning signals. They will drag the geomean around
  without representing the kind of shape the heuristic exists to optimize,
  and chasing them risks distorting recipes that work on the real target
  regime. Report tiny-M results in a separate column ("launch-bound") in
  the §7.7 strategy comparison, not in the headline geomean.

**Why this matters:** we will try multiple recipe families, classifier
cutoffs, default-warp policies, etc. If even *one* of those decisions is
informed by test results, the test set has been turned into validation and
the final number stops being an honest generalization estimate.

### 7.4 Per-kernel progress metric

For each kernel, compute on its train sweep (and on validation/test per §7.3
discipline):

| Metric | Definition | Target |
|---|---|---|
| `geomean(default_speedup)` | geo mean of `t_default / t_heuristic` over the sweep | **Large** — `default_config()` is known to be very bad on these shapes, so expect a substantial improvement. We do not set a numeric P0 bar here because the headroom over default is huge; if the geomean is anything less than "obviously much better," the heuristic is broken. |
| `geomean(torch_compile_ratio)` | geo mean of `t_torch_compile / t_heuristic` over the sweep | P1: > 0.9 (within 10% of torch.compile, geometric mean) |
| `worst_shape_regression_vs_default` | min over shapes of `t_default / t_heuristic` | > 0.95 (no shape worse than 5% slower than default) |
| `worst_shape_regression_vs_torch_compile` | min over shapes of `t_torch_compile / t_heuristic` | > 0.7 (no shape worse than ~30% slower than torch.compile) |
| `val_geomean(torch_compile_ratio)` | same as P1 but on validation set | > 0.85 |
| `test_geomean(torch_compile_ratio)` | same as P1 but on test set | reported, not gated; used to compare strategies (§7.3) |

A heuristic that improves geomean substantially but tanks one shape by 3× is
**not landing-ready** — uniform improvement is the explicit goal. The geomean
focus catches average-case wins; the `worst_shape_regression` columns catch
the "tanked one shape" failure mode.

Why no numeric P0 bar on `default_speedup`: the current `default_config()`
behavior on reduction kernels is bad enough that any reasonable heuristic
should clear it by a wide margin. Holding a literal threshold there encourages
gaming the easy comparison; we'd rather spend the discipline on the
torch.compile and worst-shape metrics, which are the ones that actually
determine whether the kernel is shippable.

### 7.5 Hill-climbing workflow

1. **Implement** the recipe per §6 for the kernel under study.
2. **Run Signal (a)** on the kernel's **train** sweep.
3. **For each shape where Signal (a) shows < 1.0× (regression vs default):**
   a. Capture generated code with `HELION_PRINT_OUTPUT_CODE=1`.
   b. Compare to what Inductor emits for the same logical op via
      `TORCH_LOGS=output_code python -c "import torch; ..."`.
   c. Identify which knob differs (almost always: `num_warps`, `R0_BLOCK`,
      or `XBLOCK`).
   d. Adjust the recipe — never special-case a shape; adjust the rule
      that produced it.
4. **Re-run Signal (a) on train.** When the geomean is clearly large (a much
   better launch config than default — see §7.4 for why no numeric bar) and
   no train shape regresses below the `worst_shape_regression_vs_default`
   threshold, advance to Signal (b).
5. **Run Signal (b) on train.** If heuristic seed rank is poor on shape X,
   either the recipe is wrong (fix it) or the autotuner is finding something
   surprising — if the latter, check whether the surprising config is stable
   across reruns or noise.
6. **Run Signal (a) + (c) on validation.** Confirms the recipe generalizes
   beyond the shapes it was authored against. Iterate back to step 3 if
   validation reveals a systematic gap (not a single-shape outlier).
7. **Keep iterating past the minimum bar.** Once §7.4's P1 floor is cleared,
   do not stop — keep refining as long as you're closing the gap to
   torch.compile (or pulling ahead). Stop only when §7.6's halt-and-reassess
   or hard-ceiling conditions are met. The goal is to beat torch.compile on
   *every* validation shape, not just on the geomean.
8. **Move to next kernel** when §7.6's stopping criteria are met. Do not
   touch the test set yet.

### 7.6 Stopping criteria

For a given kernel:

- **Minimum bar to advance to the next kernel:** train + validation both
  clear §7.4's thresholds (large `geomean(default_speedup)`, P1 geomean ≥ 0.9
  on train and ≥ 0.85 on validation, no regression worse than the bars in
  §7.4). This is the *floor*, not the goal.
- **Keep iterating past the minimum bar whenever there is room:**
  - If `worst_shape_regression_vs_torch_compile` on validation is < 1.0 on
    *any* shape, keep refining — the explicit goal is to **beat torch.compile
    on every shape**, not just on the geomean. A 1.05× geomean win that
    leaves three shapes at 0.85× of torch.compile is worse than a 1.02×
    geomean win that has every shape ≥ 1.0×.
  - If validation geomean is climbing run-over-run, keep going. Don't stop
    because you hit a numeric threshold — stop because the gradient flattens.
- **Halt and reassess** when: 3+ iterations of recipe tweaks produce no
  measurable improvement on either geomean or the worst-shape ratio. At
  that point, revisit the classifier (maybe the wrong hint is being
  emitted) or the recipe family (maybe the Inductor recipe doesn't
  transfer for this shape regime).
- **Give up on the kernel** after **2 halt-and-reassess cycles** with no
  progress. Ship whatever the best version was, record the remaining gap
  in the PR, and file the kernel as a v2 follow-up. The point of 2 cycles
  is: cycle 1 reassesses the recipe, cycle 2 reassesses the classifier
  (or vice versa) — if neither shifts the numbers, the next blocker is
  almost certainly outside the v1 scope (e.g., a missing fact signal, a
  knob v1 deliberately left unpinned, a kernel structure v1 doesn't
  handle).
- **Hard wall-clock ceiling:** ~1 day per kernel. The reassess-count rule
  above will typically fire first; the wall-clock is just a safety net for
  the case where each iteration is unusually cheap.

**Project-level stopping:** the whole project halts when every v1 kernel
has cleared the minimum bar AND we have spent the per-kernel budget pushing
past it. Then run §7.7 — strategy comparison on validation, then the
single test-set report.

### 7.7 Strategy comparison (on validation) and final test-set report

**Strategy comparison happens on validation, not test.** For each candidate
strategy that has cleared §7.6, record:
- `val_geomean(torch_compile_ratio)` per kernel and aggregated across kernels
- `worst_shape_regression_vs_torch_compile` per kernel on validation
- Number of validation shapes below the 0.7 worst-shape floor

Pick the winning strategy off this table. The winner is the strategy with
the best aggregated `val_geomean(torch_compile_ratio)` that also has zero
shapes below the worst-shape floor (or the fewest, if no strategy is clean).

**Final test-set report (exactly once, only after the winner is picked):**
run the chosen strategy on the test set and report:
- `test_geomean(torch_compile_ratio)` per kernel and aggregated
- `worst_shape_regression_vs_torch_compile` per kernel on test
- Number of test shapes below the 0.7 worst-shape floor

This number goes in the PR description as the honest generalization estimate.
It is **not** an input to which strategy ships — that decision was already
made on validation. If the test number embarrasses the validation result by
a wide margin, that is itself an interesting signal (validation shapes were
unrepresentative) and should be discussed in the PR, but the heuristic still
ships as picked.

---

## 8. Risk & Fallback

| Risk | Likelihood | Mitigation |
|---|---|---|
| `.expected` goldens churn massively under Step 1 | Low (with flag) | Step 1 is gated behind `HELION_USE_COMPILER_SEEDS=1` (default off), so CI golden files don't shift on land. When flipping the flag to default-on later, regenerate goldens in that PR. |
| Step 1 changes user-visible behavior (effort=none output differs) | Low (with flag) | Default-off flag means landing is a no-op for users. Flip default in a follow-up PR once recipes are mature. |
| `ReductionFact` populator crashes on some kernel | Medium | Wrap entire populator in try/except; log warning; clear fact list on failure. Heuristic falls back to ineligible → no seed → existing default. Kernel itself still compiles. **Inspect the logged warnings periodically** — each one is a kernel structure the populator doesn't handle yet, and worth investigating to expand coverage (or to document why it's intentionally skipped). |
| Heuristic produces invalid `Config` for some kernel (block_size below floor, etc.) | Medium | `compiler_seed_configs` already wraps `get_seed_config` in try/except (`__init__.py:46-55`). Invalid config is silently skipped — no compilation failure. |
| INNER classification wrong for transposed inputs | Medium | Stride read from runtime fake tensor handles this; OUTER hint correctly emitted. Test explicitly with `x.t()` shapes in validation. |
| Heuristic regresses pretuned kernels | Medium | Pretuned configs continue to win where present (they're checked before `compiler_seed_configs` in `BoundKernel._maybe_use_pretuned_config` — verify this is the order). Heuristic only applies to non-pretuned. |
| `num_warps = R // 128` on rnumel=130k overflows clamp | Low | Explicit clamp `_clamp_pow2(R // 128, NW_MIN, NW_MAX)` handles this. |
| Conflicts with `autotune_force_persistent` setting (`runtime/settings.py:369`) | Low | When the setting forces persistent, heuristic should still emit a persistent seed (or skip). Check `env.settings.autotune_force_persistent` in `is_eligible`. |
| Multiple reductions per kernel (bwd) match v1 eligibility filter | Low | v1 `is_eligible` requires `len(facts) == 1`; bwd kernels are filtered out. |

**Hard rollback:** revert Step 1 (the 3-site edit) and Step 3 (heuristic
registration). Steps 2 (facts + enum) are inert without the heuristic and
can be left in place.

---

## 9. Expansion Path (v2, immediately after v1)

The user has signaled that user-tiled + backwards should follow v1 *soon*.
Design the v1 to make these additive, not breaking.

### 9.1 v2a: backward kernels (rms_norm_bwd, layer_norm_bwd)

These have **two reductions per kernel**:
- per-row reduction over N (compiler-managed, same as fwd)
- per-column / cross-CTA reduction over M for parameter gradients

Plan:
- Relax `is_eligible` to `len(facts) >= 1` (drop the `== 1` restriction).
- For each fact, apply the recipe independently.
- Emit a Config with `reduction_loops` as a list of N values (one per
  reduction dim, in the order `config_spec.reduction_loops` enumerates them).
- Add a unit test with both reductions present; verify both are seeded
  coherently.

### 9.2 v2b: user-tiled reductions (kl_div, jsd, welford, softmax_two_pass)

Different code path. The knob is `block_sizes[user_tile_block_id]`, not
`reduction_loops`. Sketch:

1. Add `UserTiledReductionFact` (separate from `ReductionFact` — different
   semantics, different fields).
2. Populator: walk the FX graph for `hl.tile(N, block_size=...)` loops where
   the loop body contains an accumulator pattern (`hl.zeros(...)` outside,
   `+=` inside, store after). Recognize the accumulator's reduction-axis
   block_id.
3. New heuristic class `TritonUserTiledReductionHeuristic` consuming
   `user_tiled_reduction_facts`.
4. Seed `Config(block_sizes=[..., n_block, ..., m_block, ...])`. Recipes
   resemble Recipe B (INNER looped) but the `R0_BLOCK` equivalent is now a
   `BlockSizeSpec` value with its own min/max bounds.

The framework from v1 (registry, `compiler_seed_configs`, ranking) is
reused as-is.

### 9.3 Not in v2

- Cooperative reductions — permanently out.
- Coordinate-descent post-search — orthogonal to seed quality.
- A separate TMA post-filter — Helion already enforces the 16-byte minimum
  per-load at `helion/_compiler/indexing_strategy.py:497-499`. The Inductor
  pattern (build pool → filter) doesn't apply to Helion's architecture
  (per-load eligibility checks inside the indexing strategy). If a future
  reduction-specific TMA concern surfaces (e.g. reduction-axis TMA loads
  needing alignment guarantees beyond 16 bytes), revisit then — but it
  belongs in `TensorDescriptorIndexingStrategy`, not in the heuristic.

---

## 10. Appendix

### 10.1 Repo-relative file-path quick reference

**Helion — files you will modify:**
| File | Purpose |
|---|---|
| `helion/_compiler/backend.py:676` | Step 1 edit |
| `helion/runtime/kernel.py:986` | Step 1 edit |
| `helion/runtime/kernel.py:1003` | Step 1 edit |
| `helion/_compiler/reduction_hint.py` | **NEW** — Step 2a |
| `helion/autotuner/config_spec.py:160` (after `MatmulFact`) | Add `ReductionFact` |
| `helion/autotuner/config_spec.py:519` (after `matmul_facts`) | Initialize `reduction_facts` |
| `helion/_compiler/device_ir.py:748` (sibling of `register_rollable_reductions`) | Add `register_reduction_facts` |
| `helion/_compiler/device_ir.py:2207` | Call new method |
| `helion/_compiler/autotuner_heuristics/triton_reduction.py` | **NEW** — Step 3 |
| `helion/_compiler/autotuner_heuristics/__init__.py:18` | Register heuristic |

**Helion — files you will read for reference (do not modify):**
| File | Why |
|---|---|
| `helion/_compiler/autotuner_heuristics/triton.py` | Pattern for `TritonSkinnyGemmHeuristic` |
| `helion/_compiler/autotuner_heuristics/common.py` | `matches_hardware`, `dedupe_configs`, `clamp_block_size_targets` |
| `helion/_compiler/autotuner_heuristics/registry.py` | `AutotunerHeuristic` base class |
| `helion/_compiler/compile_environment.py:1235` | `BlockSizeInfo` |
| `helion/_compiler/compile_environment.py:106` (`get_block_id`) | rdim → tensor axis lookup |
| `helion/_compiler/reduction_strategy.py:366, 567, 743` | `PersistentReductionStrategy`, `LoopedReductionStrategy`, `BlockReductionStrategy` |
| `helion/_compiler/device_ir.py:575, 696, 748` | `RolledReductionInfo`, `DeviceIR`, `register_rollable_reductions` |
| `helion/_compiler/roll_reduction.py:183-204` | rdim→axis pattern to mirror |
| `helion/_compiler/inductor_lowering.py:645` | `ReductionLowering` |
| `helion/_compiler/indexing_strategy.py:228, 443, 497-499` | `TensorDescriptorIndexingStrategy` dispatch + 16-byte TMA minimum (§5.3) |
| `helion/_compat.py:207, 339` | `supports_tensor_descriptor()` gate |
| `helion/language/matmul_ops.py:292` | `MatmulFact` populator pattern to mirror |
| `helion/autotuner/config_spec.py:2973` | `ReductionLoopSpec` encoding |
| `helion/autotuner/config_generation.py:407` | Where seeds are injected into autotuner |
| `helion/autotuner/effort_profile.py` | `none`/`quick`/`full` profile definitions |
| `helion/runtime/settings.py:369` | `autotune_force_persistent` |
| `helion/_hardware.py` | `get_hardware_info()` |
| `helion/benchmarks/run.py:340-560` | `KERNEL_MAPPINGS` |
| `helion/benchmarks/README.md` | Bench invocation reference |
| `helion/CLAUDE.md` | Agent guidance (testing, lint, env vars) |

**Inductor — oracle sources (read-only reference):**
| File | Symbol | Line |
|---|---|---|
| `pytorch/torch/_inductor/runtime/triton_heuristics.py` | `_reduction_configs` | 3817 |
| same | `_persistent_reduction_configs` | 4347 |
| same | `triton_config_reduction` | 3283 |
| same | `outer_config_opt` (nested in `_reduction_configs`) | 3908 |
| same | `MAX_R0_BLOCK = 1024 if device_major is not None and device_major >= 10 else 2048` | 3841 |
| same | register-intensive trigger | 3842 |
| same | `_get_nd_reduction_numels` | 3245 |
| same | `_maybe_filter_configs_for_tma_restrictions` (out of scope) | 3629 |
| same | `filter_reduction_configs_for_determinism` (defer to v2) | 4126 |
| same | `cached_autotune` | 2853 |
| same | `cooperative_reduction` (out of scope) | 4293 |
| `pytorch/torch/_inductor/choices.py` | `should_use_persistent_reduction` | 412 |
| same | `should_use_cooperative_reduction` (out of scope) | 390 |
| `pytorch/torch/_inductor/ir.py` | `Reduction.num_splits` | 1355 |
| same | classification block (verbatim INNER/OUTER vote) | 1537–1561 |
| same | `_multilayer_second_step_hint` (n/a for Helion) | 1861 |
| `pytorch/torch/_inductor/codegen/simd_kernel_features.py` | `SIMDKernelFeatures.get_reduction_hint` | 152 |
| `pytorch/torch/_inductor/runtime/hints.py` | `ReductionHint` enum | 42 |
| same | `TRITON_MAX_RSPLIT = 64` (out of scope) | 22 |
| `pytorch/torch/_inductor/codegen/triton.py` | `inductor_meta` dict assembly inside `codegen_kernel` | 5956 / 6091 |

### 10.2 v1 kernel targets (compiler-managed forward)

This is the canonical shape list for v1. It is copied here so the plan does
not depend on `list_of_kernels.md`.

Shape conventions:
- Row-wise reductions use `(M, N)`, where `N` is the reduced/feature dimension.
- Cross entropy uses `(BT, V)`, where `BT = batch * sequence_length` and `V`
  is the vocabulary/class dimension.
- `Validation shapes` are used for strategy selection. `Test shapes` are run
  exactly once at the end, after the winning strategy is chosen.

#### `sum_kernel`

Source: `helion/examples/sum.py`

Train shapes:
`(2048,1024), (2048,4096), (2048,16384), (4096,1536), (4096,5120),
(8192,256), (8192,4096), (32768,256), (32768,1024)`

Validation shapes:
`(16,4096), (2048,1023), (4096,6144), (512,65536)`

Test shapes:
`(2048,2047), (1024,32768), (262144,256)`

Launch-bound correctness shapes, excluded from headline metrics:
`(1,4096)`

#### `rms_norm_fwd`

Source: `helion/examples/rms_norm.py`

Train shapes:
`(2048,1024), (2048,2048), (2048,4096), (2048,8192), (2048,16384),
(4096,1536), (4096,3584), (4096,5120), (4096,7168), (8192,4096),
(8192,8192), (32768,256), (32768,1024)`

Validation shapes:
`(16,4096), (2048,1023), (2048,3072), (4096,12288), (262144,256)`

Test shapes:
`(128,4096), (2048,2047), (2048,6144), (1024,32768), (589824,256)`

#### `layer_norm_fwd`

Source: `helion/examples/layer_norm.py`

Train shapes:
`(4096,1024), (4096,2048), (4096,4096), (4096,8192), (4096,12288),
(4096,15872), (2048,3584), (2048,8192), (8192,4096), (8192,5120),
(8192,7168)`

Validation shapes:
`(16,4096), (2048,1023), (2048,1536), (4096,6144), (1024,32768)`

Test shapes:
`(128,4096), (2048,2047), (1024,36864), (1152,36864), (262144,256)`

#### `softmax` / `softmax_decomposed`

Source: `helion/examples/softmax.py`

Train shapes:
`(4096,256), (4096,512), (4096,1024), (4096,2048), (4096,4096),
(4096,8192), (4096,12288), (4096,16384), (32768,256), (32768,1024)`

Validation shapes:
`(16,4096), (2048,1023), (2048,32768), (128,131072)`

Test shapes:
`(128,4096), (2048,2047), (512,65536), (262144,256)`

#### `cross_entropy`

Source: `helion/examples/cross_entropy.py`

Train shapes:
`(4096,4096), (4096,16384), (8192,32768), (16384,32768),
(8192,65536), (16384,65536), (8192,131072)`

Validation shapes:
`(2048,32000), (8192,128000), (4096,129280), (1024,256000)`

Test shapes:
`(4096,32000), (2048,128256), (2048,151936)`

#### `longsum` (compiler-managed persistent variant)

Source: `helion/examples/long_sum.py`

Train shapes:
`(1,32768), (2,65536), (4,130000), (8,131072), (16,262144)`

Validation shapes:
`(1,100000), (4,262143)`

Test shapes:
`(1,1048576), (32,65536)`

Out-of-sample shapes that are too large for local iteration: shrink `M` /
`BT` while keeping `N` / `V` fixed. **Caveat:** shrinking `M` past ~16 changes
the performance problem — the kernel becomes launch-dominated and no config
choice meaningfully affects wall time. Treat any resulting `M < ~16` shape as
a tiny-M correctness check per §7.3, not as a tuning signal. If a shape that's
"too large for local iteration" can only be shrunk by collapsing `M` into the
launch-bound regime, drop it from local iteration entirely and only run it on
the target hardware at §7.7 reporting time.

### 10.3 Glossary

| Term | Meaning |
|---|---|
| **Compiler-managed reduction** | Kernel where the user writes `x[tile_m, :].sum(dim=-1)` and Helion's compiler owns the reduction lowering (persistent vs looped is config-driven via `ReductionLoopSpec`). |
| **User-tiled reduction** | Kernel where the user wrote `for tile_n in hl.tile(N, block_size=...)` over the reduction dim explicitly. The chunk size is a `BlockSizeSpec` shared with other ops; no persistent option. |
| **rdim** | Reduction dimension. Has a `block_id` and a `BlockSizeInfo` in `env.block_sizes`. |
| **R0_BLOCK** (Inductor) ↔ value of `reduction_loops[rdim_block_id]` (Helion) | The looped chunk size. `None` means persistent. |
| **XBLOCK** (Inductor) ↔ value of `block_sizes[m_block_id]` (Helion) | The parallel-axis tile size paired with the reduction. |
| **Persistent reduction** | The whole rdim fits in registers; single Triton block does the entire reduction. No `R0_BLOCK` kwarg. |
| **Looped reduction** | rdim chunked into `R0_BLOCK`-sized pieces, accumulated over chunks. |
| **`ReductionHint`** | Inductor's classifier: INNER (stride-1 along rdim), OUTER (non-coalesced), OUTER_TINY (split-reduction second stage), DEFAULT. |
| **Rollable** | A reduction is rollable iff `ReductionRoller` can rewrite the persistent-style FX graph into a looped one. Non-rollable reductions get no `ReductionLoopSpec` (engineering gaps in roll_reduction.py + matmul-fed K). |
| **Stage A / Stage B / Stage C** (Inductor) | A: kernel-kind selection (persistent/looped/coop). B: config gen for the chosen kind. C: filters + autotune. Helion's heuristic operates at Stage B's abstraction level. |
| **Fact bag** | `env.config_spec.matmul_facts` / `reduction_facts` — read-only per-kernel facts populated by compile passes, consumed by heuristics. |
