# Handoff: `reread_buffer_slots` / re-read eviction-slot resolution

> Context for a synthesizer agent merging diffs from several conversations. Assumes
> background on the reduction-seed heuristic project. This doc is specifically about
> the `ReductionFact.reread_buffer_slots` field and the load-eviction-policy slot bug.
>
> **Note on scope:** the synthesizer is expected to add a **compiler analysis that
> walks memory ops once and stores per-load info in a struct**. If so, that analysis
> can/should **subsume everything below** — the "mimic the codegen walk" implementation
> described here is a stopgap that exists only because there was no such analysis. Read
> this for the *problem definition and the invariants*, not as a prescription to keep
> the mimic.

---

## 1. What the field is for

The reduction seed heuristic emits a `load_eviction_policies` config: a flat
`list[str]` over `{"", "first", "last"}`, **indexed positionally by a load-only
counter** (`device_function.device_load_index`, bumped once per emitted load in the
triton `load` codegen, `helion/language/memory_ops.py` ~line 6201; stores have a
separate index). For re-read reductions the seed wants the re-read row's **first**
load marked `'last'` (keep it L2-resident across passes) and its other loads `'first'`.
`reread_buffer_slots` was the field telling the seed **which list index** to mark.

Per-kernel: only cross_entropy (`logits`), welford (`x`), softmax (`x`) emit a `'last'`
on the curriculum (48 of 259 shapes). rms_norm/layer_norm are `row_reread=True` but
deliberately left at eviction-default; sum/long_sum are `'first'`-only; kl_div/jsd have
no re-read (`row_reread=False`).

## 2. The bug (the thing any solution must fix)

`reread_buffer_slots` stored a slot index **predicted at fact-build time** by
enumerating the *pre-rolling* `self.graphs` in `graph_id × find_nodes(sort=False)`
order. But codegen assigns policies positionally by `device_load_index`, which is a
**root-first walk over the *rolled* codegen graphs** (`build_codegen_graphs(config)`),
descending into control-flow subgraphs in emission order. These are **two independent
enumerations** that can disagree:
- a manual inner-tile reduction allocates its loop subgraphs *before* the root (so
  graph_id order reverses codegen's root-first order);
- reduction rolling can add/duplicate loads.

So the predicted slot could land `'last'` on the **wrong load** (a perf-correctness
bug — eviction is a cache hint, never affects numerics, so it degrades silently rather
than crashing).

**Concrete evidence it was a real latent bug, not just inelegant:** on
cross_entropy the old method computed slots `[2,3,4]` — implying a phantom **5th**
load of `logits` — while real codegen emits only **4** loads. They agreed on
`slots[0]=2` (the only index actually used) *by luck*. The prediction was genuinely
wrong about the index space; it was right at position 0 coincidentally.

It was correct on the **entire curriculum**, but by a guard + coincidence, not by
construction: 7/9 kernels early-return (no re-read buffer), and the 2 that fire (CE,
welford) keep their re-read load in plain `_for_loop`/reduction-loop bodies — never
under an `if`/`while`/helper. An off-curriculum **masked reduction** (re-read row
loaded inside a tensor-predicated `_if` branch, with any load before it) would break it.

## 3. The correct mental model (key insight, survives any implementation)

`device_load_index` order is **NOT a static graph property** — it is *emergent* from
the real `GraphInterpreter` lowering nodes to AST during a `generate_ast` pass. The
counter is bumped as a side effect when codegen reaches each load node; recursion into
subgraphs is itself a lowering side effect (the `_for_loop`/`_if`/`_while` codegen calls
`codegen_call_with_graph` → `GraphInterpreter(subgraph).run(...)`). There is **no
pre-existing pure walk** anywhere that reproduces this order. (Verified: ~6 graph walks
exist in the codebase; all disagree on descent set / graph list.)

Therefore the only two faithful ways to know the slot are:
- **(a) re-derive it** by hand-walking the rolled graphs in codegen's emission order
  (the stopgap I shipped — must re-implement codegen's control-flow descent and is
  drift-prone), or
- **(b) observe it** by capturing `device_load_index → buffer` during a *real* codegen
  pass (structurally faithful, can't drift, but heavier).

**A compiler analysis that records per-memory-op info in a struct during the existing
lowering/codegen traversal is essentially (b) done properly** — it reads the same
counter at the same site, so it eliminates the entire drift class by construction. That
is the right direction and supersedes the stopgap.

## 4. What I actually shipped (the stopgap — likely to be replaced)

Data-model swap + resolve-at-emit. **The `(a)∧(b)` buffer-selection logic is unchanged;
only the slot-from-buffer step moved.**

- `ReductionFact.reread_buffer_slots: tuple[int,...]` → **`reread_buffer_name: str | None`**
  (config-independent dataflow fact: *which buffer*, not *which slot*). Selection rule
  preserved: the buffer that is both **(a)** HBM-re-read (loaded in ≥2 loop graphs)
  **and (b)** a reduction input (its value feeds a `ReductionLowering` over the rdim).
  The `(b)` reduction-input AND-condition is the deliberately-kept out-of-sample guard.
- `DeviceIR._compute_reread_buffer_slots` → **`_compute_reread_buffer_name`** (returns
  the name; drops the slot enumeration).
- New **`DeviceIR.reread_eviction_slots_for_config(name, config, env)`**: builds
  `build_codegen_graphs(config)` and walks it root-first, incrementing a `load_index`
  per load, **descending control-flow subgraphs in codegen's emission order** —
  `_for_loop`/`_for_loop_step` body, both branches of `_if` (if-graph then else-graph),
  `_while_loop` (condition then body). Returns the slots of the named buffer.
- The heuristic (`autotuner_heuristics/triton.py`, 3 reread call sites: T1, Band-C,
  T2-plain) calls a thin `_reread_slots(env, device_ir, fact, seed)` that builds a
  partial `Config(**seed)` and delegates to the resolver, then feeds the result to the
  existing `_eviction_policies(..., slots)` (which still sets `policy[slots[0]]='last'`).

Two non-obvious things any reimplementation will hit:
- **Context self-sufficiency.** The resolver needs both `CompileEnvironment.current()`
  and `HostFunction.current()`. At seed-emit time **env is current but host is NOT**,
  and `compiler_seed_configs` **silently swallows exceptions** — so a context error
  makes the eviction policy *vanish*, not crash. Fix: `DeviceIR` captures its owning
  `HostFunction` (set in `lower_to_device_ir`), and the resolver enters env/host
  contexts *defensively* (only if not already current — entering an already-active env
  asserts). A struct-based analysis computed during the main compile sidesteps this
  entirely (host/env already active there).
- **`visited` dedup vs emit-per-call.** My walk dedups subgraph visits; codegen emits a
  subgraph each time it's called. Safe today (each subgraph has one call site) but a
  semantic mismatch — documented as a guard.

Files touched: `helion/autotuner/config_spec.py` (field), `helion/_compiler/device_ir.py`
(resolver + name computation + host capture), `helion/_compiler/autotuner_heuristics/triton.py`
(3 call sites + `_reread_slots`). **`kernel.py` is NOT touched** (an earlier attempt to
enter context at the caller failed — env already active there).

## 5. Invariants any solution must satisfy (use these to judge competing diffs)

1. **`CONFIG diffs=0` on the 259-shape recorder.** `_lab/harness/run3_task1_verify_after_edit.py`
   records the seed config for all 9 kernels × train/val/test before/after; the emitted
   configs must be **byte-identical** to the pre-change baseline. (Only `ReductionFact`'s
   repr is allowed to change — a fact diff with config diff 0 is fine and expected.)
   Both my original and hardened versions pass this.
2. **Emitted Triton lands `evict_last` on the re-read row**: cross_entropy→`logits`,
   welford→`x`, softmax→`x`. (Verified by reading generated code, not just the config.)
3. **Tests pass**: `test_reductions.py`, `test_examples.py` (reduction kernels),
   `test_autotuner_heuristics.py`.
4. **Off-curriculum faithfulness** (the actual point of the fix): a masked / branched /
   while-loop reduction whose re-read row sits under an `_if`/`_while` must still get the
   correct slot. The mimic handles this by descending those control-flow ops; a
   struct-from-real-codegen analysis handles it for free.

## 6. Recommendation to the synthesizer

If you are adding a **compiler memory-op analysis that stores per-load info (buffer
identity + emission index) in a struct during the real lowering/codegen traversal**,
that is the structurally-faithful answer and should **replace** both the predicted-slot
field *and* the hand-mimic resolver: the seed can then look up the re-read row's slot
directly from that struct (computed by the same pass that assigns `device_load_index`),
with no re-derivation and no drift. Keep the `(a)∧(b)` buffer-selection semantics
(which buffer is the re-read row) — that part is orthogonal and correct. Validate the
result against the four invariants in §5 (the 259-shape `CONFIG diffs=0` recorder is the
fast, GPU-light gate).
