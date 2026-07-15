# Spec: data collection — restricted-knob vs full-space CuTe fp8 autotuning

> Written 2026-07-15 for the agent who will implement + run this. Single-GPU (B200-class); GPU work is
> SERIAL — never run two autotune/bench jobs at once.
>
> **This kit is self-contained under `experiments/cute_knob_restriction/`** (this SPEC, `train_set.json`,
> and `scripts/`). It rides on the `cute-subproc-recompile-fix` branch alongside the recompile fix it
> depends on.

## (0) Setup on the run box (do this first)

The kit scripts were written for a specific machine and **hardcode `/home/dev/local/helion-rank0` and
`/home/dev/helion-env/bin/python`**. On a different box you MUST fix these before running:
- `scripts/compare_final.py`, `scripts/rank0_verify_selection.py`, `scripts/cudagraph_cobench.py`:
  the `WORKTREE = "/home/dev/local/helion-rank0"` constant near the top → set to this checkout's path.
- `scripts/compare_final_run.sh`: `PY=`, `LAB=`, `PYTHONPATH=` at the top → point at your interpreter
  and this kit dir.
Also required on the box: the CuTe backend installed (`nvidia-cutlass-dsl[cu13]` + `apache-tvm-ffi`) and
a tcgen05-capable GPU. Verify the checkout is the fix branch: `HELION_FUSED_ACCURACY_CHECK` must exist in
`helion/runtime/settings.py` and `_reattach_cute_source_hash` in `helion/autotuner/precompile_future.py`.
The `fp8_matmul` kernel is `examples/fp8_matmul.py`; the arg builder is
`benchmarks/cute/compare_matmul_backends.py` (both tracked upstream files, already in the checkout).

## (a) What this is

A data-collection experiment. Run fp8 scaled_mm autotuning two ways on the same shapes —
**(A) the current full-space search** and **(B) a search restricted to a subset of knobs** (the rest
pinned to fixed values) — and record what each produces. The open question is whether restricting the
search changes the resulting perf and/or its run-to-run consistency, in either direction.

This is a hypothesis, not a foregone conclusion. Restriction might help, hurt, or do nothing; a null
or negative result is a complete, valid outcome. Collect and report the numbers either way — do not
tune the experiment toward a win, and do not treat any shape as one we "should" improve.

One factual note (not a prediction): for the same per-config cost, arm (B) searches a smaller space
than (A), so it will cover more distinct configs in a fixed budget. Whether more coverage of a smaller
space yields a better, worse, or equal winning config is exactly what the data should show.

## (b) Shapes to test — 5 shapes (DECIDED with user 2026-07-15)

From the 8-shape fp8 train set (`train_set.json` in this kit, dtype float8_e4m3fn, scaled_mm epilogue).
Chosen to span distinct regimes and to exercise the knobs kept tunable in the restricted arm:

| shape | regime |
|---|---|
| `m64_k6144_n2048`   | deep-K narrow-N |
| `m64_k4096_n24576`  | wide-N ~1-wave |
| `m64_k5120_n51200`  | huge-N multi-wave BW |
| `m512_k4096_n4096`  | above-wave device-fill |
| `m64_k2048_n2048`   | extreme sub-wave |

Why these five specifically (regime coverage, not expected outcome): `m512_k4096_n4096` is the only
train-set shape whose known-good config uses `l2_groupings=[16]` / `persistent_interleaved`, and
`m64_k2048_n2048` the only one using `pid_type=flat` + `non_persistent` — including both ensures the
restricted arm's tunable set is actually exercised across the pid/persistence/l2 axes rather than
collapsing to one corner. `train_set.json` also carries a per-shape `golden_us` /
`stock_winner_us` — use these only as post-hoc reference points when reporting, not as targets to hit.

Excluded: `m64_k25600_n5120` (split-K, out of scope for this kernel path), `m512_k2048_n4096` and
`m64_k5120_n5120` (regimes already represented by the five; add only if results are ambiguous).

## (c) How to restrict the search — which knobs to pin, and the mechanism

### The pin/tune split (MODERATE pin-set, DECIDED with user)

Derived from two data sources: (1) which knobs *vary* across the 8 goldens in `train_set.json`, and
(2) the `ablate_valley*` results (one-knob-out cudagraph timing at the valley shape), which measured
that the pinned knobs move perf by ≤0.03µs (~0.2% = noise).

**TUNE (~6 knobs — these took ≥2 distinct values across the 8 train-set goldens):**
- `block_sizes` `[bm, bn, bk]` — goldens span `[64,16,256]`/`[64,32,256]`/`[64,64,128]`/`[128,128,128]`
- `tcgen05_ab_stages` — goldens use 6 / 8 / 12
- `pid_type` — goldens use flat / persistent_blocked / persistent_interleaved
- `tcgen05_persistence_model` — static_persistent / non_persistent (co-varies with pid_type)
- `tcgen05_acc_stages` — 1 / 2
- `tcgen05_cluster_m` / `tcgen05_cluster_n` — KEEP TUNABLE. cluster_m is SMEM-coupled to block size
  (cm2 halves per-CTA SMEM → changes the ab-stages ceiling ~6→12; see RANK0_PROGRESS.md T5b guard-safety),
  so pinning it would silently alter the reachable ab range. All 5 chosen shapes' goldens happen to be
  cm1, but leave it tunable rather than pin a coupled knob.
- `l2_groupings` + `tcgen05_l2_swizzle_size` — KEEP TUNABLE (they co-vary; `m512_k4096_n4096`'s golden
  uses `l2_groupings=[16]`/swizzle 8). Ablation rated these near-neutral but they're cheap to keep.

**PIN (constant across all 8 goldens AND ablation-confirmed neutral):**
- `tcgen05_c_stages = 2`
- `tcgen05_num_epi_warps = 4`
- `tcgen05_strategy = "role_local_monolithic"`
- `tcgen05_layout_strategy = "default"`
- all `tcgen05_warp_spec_*` (ab_load=1, mma=1, others 0; register_decrease=120, register_increase=256)
- `cute_vector_widths = [1, 1, 1]`
- `indexing` (leave at the tcgen05 default the seed produces)
- all `tcgen05_layout_overrides_*` = None

### The mechanism (this is option B — fragment-range narrowing, NOT `configs=`)

`helion.kernel(configs=[...])` only accepts a whitelist of FULL configs (verified: `kernel.py:189`
builds complete `Config` objects) — it is all-or-nothing and cannot express "pin these, tune those."
Do NOT use it for the pinned-subset arm.

The correct hook is **`Tcgen05Config.optional_fragments(for_search=True)`** in
`helion/_compiler/cute/tcgen05_config.py:1821`. That method returns the per-knob search fragments; the
search only draws from each fragment's active choices. There is ALREADY precedent for pinning here:
`cluster_n_choices = (1,)` when `for_search` (line 1828), and the FFI-launch knob "collapses to a
single True choice" (lines 1892-1896). **Pinning a knob = collapsing its fragment to a singleton.**

Concretely, the implementation should:
- For `EnumFragment` knobs (c_stages, strategy, layout_strategy, num_epi_warps, warp_spec_* enums,
  cluster when pinned): pass a 1-tuple of the pinned value, or set `search_choices` to the singleton
  (EnumFragment already supports `search_choices` as a search-only subset of `choices` — line 256; this
  keeps the validation surface intact so an explicit user config still validates).
- For the `block_sizes`/`indexing`/`vector_widths`/`layout_overrides` knobs (declared elsewhere in
  the config_spec, not in `optional_fragments`): pin via the same singleton-fragment approach in their
  respective fragment declarations.

**Recommended packaging:** gate it behind a new env var, e.g. `HELION_RANK0_PIN_KNOBS=1`, following the
existing `HELION_RANK0_*` env-gated-arm pattern already in this tree (`base_search.py`,
`config_generation.py`, and the `HELION_RANK0_BK256`/`AB_CEIL`/`AB_BIAS` arms). When the flag is set,
`optional_fragments(for_search=True)` (and the block_sizes/indexing fragment sites) collapse the PIN
list to singletons and leave the TUNE list at full range. Default off = byte-identical to current.
This makes the A/B a clean flag flip on one branch.

IMPORTANT — pin to the RIGHT value: for enum/scalar knobs the pinned value is the constant seen across
goldens (listed above). Verify against `train_set.json` before hardcoding; do not invent values.

## (d) Runs and budget

- **Budget: 600s per run.** At 600s all full-space CuTe cells already exhaust the wall-clock budget
  (measured — none reach final verification), so both arms run under the same binding constraint. A
  900s follow-up is optional if the 600s data is ambiguous; keep 600s as the primary so the two arms
  are compared under identical conditions.
- **Arms (2): full-space (flag off) vs pinned-subset (flag on).** Flag off = current behavior.
- **Seeds: 2 per (shape, arm)** — seeds 2000 and 2001. A second seed is needed to distinguish a real
  arm-to-arm difference from seed variance; two is enough to see gross variance without exploding cost.
- **Total: 5 shapes × 2 arms × 2 seeds = 20 runs × 600s ≈ 3.5 GPU-hours** (serial). Resumable; skip a
  cell whose out.json is already `status: ok` (see `compare_final_run.sh` for the skip pattern).
- **Run on the pushed fix branch** `cute-subproc-recompile-fix` (or cherry-pick the two fixes) with
  `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1` + `HELION_FUSED_ACCURACY_CHECK=1` and
  `HELION_AUTOTUNE_EFFORT=full`, so both arms run on the same up-to-date benchmarking path.

## (e) Measurement, pitfalls, and what to watch

**Judge with the cudagraph cold-L2 ruler, NOT the in-search timer.** The in-search timer has a
dispatch-floor bias on M=64 shapes (see the `selgap-timer-not-rebench` memory note: the final k=32
rebench picks the WRONG config; only cudagraph timing picks right). Re-time each arm's crowned winner
with the existing cudagraph harness (`scripts/cudagraph_cobench.py` / `rank0_verify_selection.py`).

**Record EVERY config tried in each run (not just the winner).** The `HELION_AUTOTUNE_LOG` tally
mechanism already does this: set `HELION_AUTOTUNE_LOG=<path>` per run (the harness scripts do) and it
writes `<path>.csv` with one `started` + one `ok`/`error` row PER config, columns
`run_id,timestamp_s,config_id,generation,status,perf_ms,compile_time_s,config` — the `config` column is
the full Config string (every knob value). **Retain all 20 per-run CSVs as first-class artifacts**
(not just the summary JSONs); they are the raw record of the search trajectory and the whole point of
being able to analyze the range of configs each arm explored. Do not delete/overwrite them between
cells — give each (shape, arm, seed) its own log path.

From those CSVs, produce a **per-knob distribution of values tried**, per (shape, arm): e.g. for each
knob in the TUNE set, the histogram of values sampled across the run; and confirm the PIN set shows a
single value (the pin-verification check in pitfall #1). This is what lets you compare the *range* of
configs the two arms explored, not just their winners.

**Also record per (shape, arm, seed)** — collect all of these regardless of which way they come out:
- crowned config, in-search µs, cudagraph µs, and cudagraph ratio to the `train_set.json` `golden_us`
  reference (reference point only, not a pass/fail bar)
- distinct configs searched, generations reached, whether the run hit the wall-clock budget
- across the 2 seeds per (shape, arm): the spread (min/median/max) of the cudagraph µs

**Pitfalls / things to verify:**
1. **Confirm the pin actually takes.** Before the full sweep, do one short (150s) pinned run and grep
   the tally CSV `config` column: the pinned knobs must show a SINGLE value across all rows; the tuned
   knobs must vary. If a "pinned" knob still varies, the fragment collapse didn't reach that knob.
2. **cluster_m coupling.** cluster_m is kept tunable because pinning it would change the reachable ab
   range. If you later try an aggressive arm that pins cluster_m=1, the ab ceiling drops to ~6 at cm1 —
   that's expected, not a bug (RANK0_PROGRESS.md T5b guard-safety).
3. **Reachability sanity, not a target.** `m64_k2048_n2048`'s reference config uses
   `pid_type=flat`+`non_persistent` and `m512_k4096_n4096`'s uses `l2_groupings=[16]` — both knobs are
   in the TUNE set, so those configs remain reachable in the restricted arm. If a restricted run cannot
   in principle reach a shape's reference config, that's a spec bug (a needed knob got pinned), distinct
   from the run simply not finding it — flag which case it is.
4. **Single-seed goldens caveat.** The `train_set.json` goldens are single-seed; some pinned "constants"
   could be ties rather than true optima. Report the pinned values used and note this when interpreting.
5. **Don't push `_lab/`** if you commit anything (675MB, not gitignored). Commit only `helion/` + `test/`.

**Reuse existing harness:** `scripts/compare_final.py` (per-run tally→JSON with generations/per-config
time; now also records `budget_exceeded`/`ran_final_verification`) and `compare_final_run.sh` (resumable
multi-cell runner) are the closest fit — extend the cell roster to the 5 shapes × 2 arms × 2 seeds and
add the `HELION_RANK0_PIN_KNOBS` env toggle per arm.

## Deliverable
1. The 20 retained per-run tally CSVs (every config tried, full Config string per row).
2. A per-(shape, arm) summary of the config range explored: for each TUNE knob, the distribution of
   values sampled; and confirmation the PIN knobs were single-valued.
3. A table: per (shape, arm, seed) → cudagraph µs, ratio to `golden_us`, distinct configs, gens,
   budget-hit; and the per-(shape, arm) seed spread.
4. A plain description of what the data shows — for each shape, how the restricted arm compares to
   full-space on both perf and seed-to-seed spread — without presuming a direction. A result of "no
   meaningful difference" or "restriction is worse" is a valid, complete finding to report as-is.
