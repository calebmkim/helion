# B200 Matmul Heuristic: High-Level Trace

This document describes the current B200 Triton matmul heuristic at a high
level. It focuses on the order of decisions: what each stage knows, what it
sets, and what later stages may correct.

The relevant implementation is primarily in:

- `helion/_compiler/autotuner_heuristics/triton.py`
- `helion/_compiler/autotuner_heuristics/__init__.py`

Reduction and pointwise heuristics are separate paths and are not covered here.

## Executive summary

The heuristic is best understood as:

1. Classify the kernel and choose either the single-contraction or
   multi-contraction front end.
2. Use a hardware-budget formula to draft one or more matmul tiles.
3. Map those abstract tiles onto the block-size knobs the kernel actually has.
4. Correct block-size knobs according to their actual dataflow and launch role.
5. Recompute pipeline depth and warp count from the resulting real kernel.
6. Shrink cheaper knobs until shared-memory and tensor-memory constraints are
   satisfied.
7. Emit the corrected config as both an autotuner seed and the no-autotune
   compiler default.

The important design principle is that the formula's first answer is only a
proposal. Axis projection, whole-kernel work, launch geometry, and aggregate
resource use can all change the final answer.

## The knobs

The promoted formula actively chooses:

| Knob | Purpose |
|---|---|
| `block_sizes` | Per-program tile sizes for tunable kernel axes |
| `num_warps` | Execution regime and threads assigned to each CTA |
| `num_stages` | Pipeline depth and number of in-flight operand stages |
| `l2_groupings` | PID grouping for L2 reuse, on the single-contraction path |

Other config fields retain their normal base defaults. The older B200 lookup
table can still contribute a richer autotuner search seed, but it no longer
owns the compiler default.

## Stage 1: Build structural facts

Before the heuristic runs, the compiler records facts about the kernel:

- Each dot's M, N, and K extents.
- Whether each axis is tunable, fixed at its full extent, or unknown.
- Which dots share the same block-size knob.
- Which axes form the launch grid and which are sequential loops.
- Which dots execute repeatedly and which update loop-carried state.
- Pipelined load/store regions and live intermediate tiles.

These facts let the heuristic reason about the kernel's structure rather than
matching a kernel name.

## Stage 2: Select the front end

There are two promoted B200 paths.

### Single-contraction front end

This path owns a kernel when it has one understandable contraction:

- Every M/N/K axis has a known per-program extent.
- Each tunable contraction axis has its own knob.
- Any other tunable axes are understood outer-grid axes.

An axis may be fixed rather than tunable. A kernel with no tunable dot axes can
still use this path because `num_warps` and `num_stages` remain meaningful.

### Multi-contraction front end

This path owns kernels with multiple dots, shared knobs, or other contraction
structure that cannot be represented as one independent GEMM. It runs only
when the single-contraction path does not apply.

At least one dot must have known M/N/K extents. Fully FP8 contractions are
currently declined by both paths because the promoted Triton configuration
cannot yet guarantee the desired accumulation precision.

## Stage 3: Draft a tile for one dot

Both front ends use the same budget formula to propose:

```text
(block_m, block_n, block_k, num_warps, num_stages, l2_grouping)
```

At a high level, the formula does the following:

1. Choose power-of-two M and N tiles, bounded by the problem shape and an
   accumulator budget. It generally favors N because N provides coalesced
   stores and B-operand reuse.
2. If the outer grid already supplies abundant parallelism, cap the tile so
   more CTAs can remain available.
3. Otherwise, grow the tile while tensor-memory capacity and launch-wave
   occupancy allow it.
4. If the launch would underfill the GPU, shrink a tile dimension when doing so
   strictly improves wave utilization.
5. Choose K tile size and an initial stage depth together so the operand
   pipeline fits shared memory.
6. Choose an initial warp count from tile size.
7. If necessary, shrink the draft tile until its local shared-memory and
   tensor-memory estimates fit.
8. Enable L2 grouping when the M/N grid is tall enough for multiple M tiles to
   reuse B effectively.

This is still an abstract GEMM-shaped proposal. The next stages adapt it to the
actual kernel.

## Stage 4A: Single-contraction assembly

The single-contraction path turns the proposal into a real config in the
following order.

### 4A.1 Project onto real axes

- A fixed axis takes its fixed extent.
- A tunable axis takes the proposed value, clamped to that knob's legal range.
- A grid-partitioned K loop may apply a smaller M/N ceiling because it launches
  many independent partial-output CTAs.

This produces the tile that the generated kernel will actually execute.

### 4A.2 Recompute launch and loop facts

The heuristic resolves the candidate facts needed to judge the projected tile:

- The actual number of launched programs.
- Candidate-specific sequential loop trip counts.

This prevents later decisions from being based on the pre-projection tile.

### 4A.2a Correct knobs according to their actual role

Before selecting stages or warps, the single-contraction path applies the same
structural block-size correction used by the multi-contraction path:

- A non-grid output knob with no cross-axis load reuse shrinks toward the
  tensor-memory allocation floor. Making that tile larger increases resident
  state and staging without improving arithmetic intensity.
- A grid output knob may shrink while the launch is below roughly one GPU wave,
  but only when the shrink **strictly improves launch-wave utilization**.

For a launch of `g` programs on `num_sm` SMs, the comparison is based on:

```text
wave_utilization = g / (ceil(g / num_sm) * num_sm)
```

This accepts a transition such as `64 -> 128` programs on 148 SMs, because both
launches occupy one wave and utilization doubles. It rejects `86 -> 172` or
`96 -> 192`: those launches cross into a second wave with exactly the same
average utilization, so the extra CTAs would only sacrifice operand reuse and
increase total CTA overhead.

If a block size changes, the actual launch and sequential-loop facts are
recomputed before `num_stages` and `num_warps` are selected.

### 4A.3 Recompute `num_stages`

For a normal tunable K loop, the formula's K-aware stage choice remains the
starting point.

When K is fixed and the useful pipeline is an enclosing sequential loop, a
graded stage model takes over:

- More loop work creates an opportunity for deeper pipelining.
- More independently resident CTAs reduce the need for deep pipelining.
- Shared memory limits how many stages can coexist with the desired CTA
  residency.
- A one-trip loop normally prefers a second stage only when it does not create
  excessive register pressure or collapse residency from multiple CTAs to one.

The result is the deepest useful stage count that fits the candidate's
occupancy and shared-memory situation.

### 4A.4 Recompute `num_warps`

B200 selects a warp regime from the real candidate rather than retaining the
tile-size ramp:

- Very large tensor-core work can justify eight warps.
- Enough tcgen05-eligible work can justify entering the four-warp warpgroup
  regime.
- Substantial work can justify two warps.
- Small work starts at one or two warps depending on register pressure.

Register pressure is then used as a guardrail:

- A register-MMA tile may add warps to relieve serious pressure.
- Crossing from two warps into the four-warp tcgen05 regime requires stronger
  evidence because that transition can reduce CTA residency.
- Moving from four to eight warps is allowed only when the work justifies it or
  the pressure correction does not reduce estimated residency.
- If dynamic loop work is uncertain, that uncertainty may prevent lowering a
  conservative prior choice; it cannot be used as evidence to lower it.

## Stage 4B: Multi-contraction assembly

The multi-contraction front end cannot simply accept one dot's config. It
builds a whole-kernel draft.

### 4B.1 Propose a tile for every understandable dot

Each dot receives the same budget proposal, axis projection, and
candidate-aware stage/warp treatment described above. The role correction is
deliberately skipped at this point: shared knobs have not yet been resolved, so
correcting each dot independently could apply the same kernel-level decision
more than once. These are per-dot preferences, not yet kernel configs.

### 4B.2 Rank the dots

Dots are ranked by:

1. Whether the dot updates loop-carried state.
2. Its dynamic amount of dot work.
3. Its output area.

This makes a long-lived or expensive contraction win conflicts over a less
important one.

### 4B.3 Resolve every block-size knob

For each tunable knob:

- If one or more dots use it, take the corresponding M/N/K value from the
  highest-ranked user.
- If it is only an outer parallel grid axis, pin it to its floor.
- If no dot or understood grid role claims it, preserve the base default.

### 4B.4 Correct knobs according to their actual role

The GEMM formula assumes that growing an output tile improves reuse. That is
not true for every surrounding kernel. After shared knobs are resolved, the
same two structural corrections used by the single-contraction path apply:

- A non-grid output knob with no cross-axis load reuse is shrunk toward the
  tensor-memory allocation floor. Growing it increases state and staging
  without improving arithmetic intensity.
- A grid knob is shrinkable while the launch is below roughly one GPU wave,
  but only if halving it strictly improves launch-wave utilization. A shrink
  that merely turns one partial wave into two equally partial waves is rejected.
- Once no legal shrink improves utilization, the tile is left alone to
  preserve operand reuse and avoid extra CTA overhead.

### 4B.5 Recompute kernel-global scalar knobs

The front end then chooses one `num_stages` and one `num_warps` for the whole
kernel:

- `num_stages` uses aggregate pipeline regions, loop trips, shared-memory use,
  launch demand, and estimated resident CTAs.
- Stage depth is recomputed from the final role-corrected block sizes, so its
  loop trips, shared-memory ring, launch grid, and residency inputs describe
  the tile that will actually be emitted.
- `num_warps` uses aggregate candidate dot work and whole-kernel register
  pressure, through the same B200 regime selector used by the single-dot path.

## Stage 5: Whole-kernel resource fix-up

Both paths treat the assembled config as provisional and enforce hard resource
limits.

The correction order is deliberate:

1. If shared memory is over budget, reduce `num_stages` first. Pipeline depth is
   cheaper to surrender than tile area.
2. If the config is still over shared-memory or tensor-memory limits, halve a
   legal tunable tile knob.
3. Never shrink a fixed axis. The kernel author fixed it, so another knob must
   absorb the correction.
4. Recompute `num_warps` after tile changes because work, register pressure,
   tcgen05 eligibility, launch size, and residency may all have changed.
5. Repeat until the config fits or no understood knob can move further.

The single-contraction path preferentially shrinks N then M after exhausting
stage depth. The multi-contraction path shrinks the largest dot-claimed knob,
which may be shared by several contractions.

## Stage 6: Emit primary and alternate seeds

The final corrected config is rank 0:

- It becomes the compiler default when autotuning is disabled.
- It is also the first compiler seed when autotuning is enabled.

The single-contraction path may add:

- A transposed M/N aspect-ratio alternative, if legal.
- A one-stage-shallower alternative.

The multi-contraction path may add one shallower-stage neighbor.

The old B200 shape-table heuristic remains registered as an unpromoted search
seed. It can help autotuning explore additional config fields, but it cannot
replace the formula-selected no-autotune default.

## Knob trace at a glance

| Knob | Initial choice | Main corrections | Final enforcement |
|---|---|---|---|
| `block_sizes` | Per-dot budget formula | Axis projection; shared-knob ranking; role-aware shrinking with a strict wave-utilization guard | Shrunk only when stages cannot resolve SMEM or when TMEM is over budget |
| `num_stages` | K-loop length and operand-ring SMEM | Graded from real loop trips, launch demand, residency, and whole-kernel pipeline regions | Reduced first on SMEM overflow |
| `num_warps` | Coarse tile-size proposal | Re-solved from candidate work, tcgen05 regime, register pressure, and residency | Re-solved after every tile correction |
| `l2_groupings` | Tall M/N grid enables B reuse | No later correction | Emitted by the single-contraction formula; multi-contraction leaves the base setting |

## Mental model

The shortest accurate mental model is:

> Propose like a GEMM, project onto the kernel's real axes, resolve shared
> choices, correct each knob by its actual role, then repeatedly recompute
> global scalars and resources until the config is legal.

That ordering is the core of the current heuristic. The initial tile formula
provides direction; the candidate-real and whole-kernel correction stages
decide what is actually emitted.

## Appendix: Stage-depth limitation

The current stage solver is primarily a **feasibility solver**. Given the final
block sizes, it determines the available pipeline regions, their loop trips,
the estimated shared-memory ring, launch demand, and resident CTA count. It
then selects the deepest stage count that those constraints permit. In that
sense, the policy is approximately:

> Use as many stages as possible, subject to useful-loop-depth, occupancy, and
> hard-resource limits.

What it does **not** model is whether the next feasible stage is profitable:

> Does another stage overlap enough useful loading to justify its additional
> scheduling and resource cost?

Estimated occupancy is already a constraint: the chosen depth normally must fit
the per-CTA SMEM share for the residency available at stage one, and the
one-trip check rejects a stage-two transition that collapses residency from
multiple CTAs to one. The missing question begins after that guard has passed.

That missing question includes several effects:

- additional async copies, barriers, and pipeline prologue/epilogue work;
- occupancy effects missed by the resource estimate or its discrete thresholds;
- register-allocation or spill changes in the generated schedule;
- a possible change in the selected warp or tcgen05 regime;
- whether the staged loads are a substantial part of execution time;
- whether operands are reused across multiple dots, making the loop
  compute-heavy despite having several trips;
- work outside the pipelined loop, which cannot benefit from another stage.

The distinction explains why the same nominal transition can have opposite
results. Long streaming or carried loops can benefit from deeper staging when
residency is unchanged. Short, compute-heavy, or operand-reusing loops may pay
pipeline overhead without hiding meaningful latency. Resource estimates can
also predict an occupancy loss that the compiled allocation does not actually
incur.

A general profitability model is difficult because these effects are
backend-dependent, coupled, and sometimes discontinuous. Stage depth can alter
the generated schedule, spills, tensor-core regime, or warp choice rather than
merely adding one more identical buffer. Until a reliable structural signal is
available, the heuristic intentionally retains the simpler deepest-feasible
policy and treats profitability as a known limitation rather than encoding
kernel-specific exceptions.

### Multiple scalar priors remain a future opportunity

FE2 has two defensible starting points for its kernel-global scalar solve: a
whole-kernel prior conditioned on each dot, and a merged-only prior derived
after shared block sizes are resolved. GPU-1 measurements found substantial
wins in both directions: the preconditioned result was up to 1.58x faster,
while the merged-only result was up to 1.43x faster. Neither prior dominates.

That makes both results plausible future autotuner seeds after final
whole-kernel validation. The current implementation intentionally emits only
the whole-kernel-conditioned path, however. Keeping a second merged-only path
would add policy and validation surface to an already large heuristic; it is
not retained solely to preserve this future search opportunity.

### Joint BK, stage, and warp selection

The current policy splits two useful properties across different paths:

- A tunable K loop gets a joint BK and `num_stages` proposal from
  `_matmul_tile`, but that proposal uses dot-local SMEM and only coarse,
  binary occupancy signals.
- A fixed-full-extent K gets the candidate-real graded stage model, which uses
  whole-kernel SMEM and estimated CTA residency, but has no BK decision to
  make.

A possible improvement is to unify these after axis projection. For each legal
BK, stage depth, and likely warp count, construct the complete kernel block
vector, reject hard-resource failures, and compute whole-kernel residency.
Fixed K would be the same search with a single BK candidate.

The dependency is circular: BK and stages determine SMEM; warps affect thread
and register limits; those resources determine residency; and the desired
residency in turn changes the acceptable BK and stage depth. The candidate
space is small enough to enumerate. The difficult part is ranking the feasible
frontier: maximum residency is not always fastest, because a long K loop may
profit enough from a larger BK or deeper pipeline to justify fewer resident
CTAs.

The tunable-BK graded-stage ablation did not test this joint policy: it held BK
fixed and replaced only the stage choice. Its opposing results--large regular
GEMM regressions but substantial split-K and Mamba wins--motivate the joint
experiment rather than refute it. This remains a future heuristic improvement,
not part of the current implementation.

## Appendix: Register-load accounting limitation

The register-pressure model currently excludes every value produced by a load.
Those bytes are instead assigned to the shared-memory model. This is a useful
approximation for pipelined matmul operands that Triton promotes into an SMEM
ring, but `LiveTile(kind="load")` describes every load, not only promoted dot
operands.

For example:

```python
value = y[0, tile_m, tile_n]
x[tile_m, tile_n] = value
```

`x` remains an HBM destination, but `value` must pass through registers on the
way from the load to the store. Because it is classified only as a load, the
current model assigns it zero register bytes. The same undercount can occur for
ordinary pointwise loads and other values that are not SMEM-promoted.

The underlying limitation is that the model knows an operation is a load but
does not encode its eventual storage role. A more precise model would
distinguish SMEM-promoted pipeline operands from register-resident loads,
charging the latter to register liveness. Simply charging every load to both
budgets would also be too crude: promoted operands need different accounting,
and compiler scheduling may stream or reuse registers rather than materialize
the full logical tile at once.
