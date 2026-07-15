# CuTe subprocess-autotune recompile fix

Two fixes that remove redundant cutlass-DSL compiles from **subprocess-mode** autotuning of
CuTe fp8 kernels. Net effect on `4096³` fp8 (effort=full, 600s budget): subprocess mode goes
from **79 → 169** distinct configs searched — parity with in-process (170) — while keeping the
killable-worker isolation that subprocess mode exists for.

> Branch: `cute-subproc-recompile-fix` (off `upstream/main`). Both fixes are CuTe-specific; they
> are inert on the Triton backend and on the in-process autotune path.

## What's in here

1. **Fuse benchmark + accuracy into one worker job** (`HELION_FUSED_ACCURACY_CHECK`, default **off**).
   In subprocess mode the parent used to send the worker two jobs per config — `BenchmarkJob` then
   `AccuracyCheckJob` — each of which re-`exec`s the kernel source into a fresh module and so pays
   its own cold DSL compile. This fuses them into one `BenchmarkAndAccuracyJob` that loads/compiles
   once and reuses it for both timing and the accuracy check. ~2× per-config in the benchmark phase.
   Files: `helion/autotuner/benchmark_job.py`, `benchmark_provider.py`, `runtime/settings.py`.

2. **Give the worker a disk-cache key** (always on; no flag).
   The on-disk CuTe DSL cache key derives from `cute_kernel._helion_cute_source_hash`, which the
   parent sets in `annotate_compiled_module` (only reached from `BoundKernel.compile_config`). The
   worker loads kernels via `_load_compiled_fn`'s bare `exec`, which never set that attribute, so
   every worker launcher got `cache_key=None` → the disk cache was fully disabled in the worker →
   every worker compile was cold, *including the higher-accuracy `rebenchmark_population` pass that
   re-times the same configs it just compiled*. `_reattach_cute_source_hash` recomputes the same
   `sha256(source)` in the worker and reattaches it, so a repeat compile of the same config is a
   warm disk reload (~0.05s) instead of a cold recompile (~3.5s). This is what closes the
   config-count gap (the rebench pass was ~130s of the 600s budget on `4096³`).
   File: `helion/autotuner/precompile_future.py`.

Both fixes assume subprocess mode; within a single autotune run they work with no extra flags
because the ephemeral `CUTE_DSL_CACHE_DIR` lives for the whole search and the long-lived worker
inherits it.

## How to run an autotune job with the improvement

**Subprocess mode MUST be ON** for either fix to do anything. On some boxes it is force-disabled;
check for `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0` in the venv's `activate` and override it.

```bash
export HELION_BACKEND=cute
export HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=1   # REQUIRED — the fixes only touch the worker path
export HELION_FUSED_ACCURACY_CHECK=1            # fix #1 (off by default)
export HELION_AUTOTUNE_EFFORT=full
export HELION_AUTOTUNE_BUDGET_SECONDS=600
# fix #2 (disk-cache key) needs no flag.
python your_autotune_job.py
```

- If `HELION_AUTOTUNE_BENCHMARK_SUBPROCESS=0` (in-process), **both fixes are inert by design** —
  in-process already reuses the compiled kernel in memory. You will see no change; that's expected.
- `HELION_KEEP_CACHE` is **not** required for the intra-run win (the win is within one search).

## Verifying the fix works on a new box (optional probe)

An env-gated probe logs every CuTe kernel materialization (cold compile vs warm disk reload) with
its pid, so you can confirm the worker's rebench compiles turned warm. It is **inert unless the env
var is set** — zero cost otherwise. (`_CompiledCuteLauncher.__call__` in `helion/runtime/__init__.py`.)

```bash
export HELION_RANK0_COMPILE_PROBE=/tmp/compile_probe.jsonl
# ... run the autotune job as above ...
```

Then inspect: with the fix you should see the worker's rebench configs as `disk_reload` (~0.05s),
not `cold_compile` (~3.5s), and every row's `cache_key` should be non-`None`:

```bash
python - <<'PY'
import json, collections
rows = [json.loads(l) for l in open("/tmp/compile_probe.jsonl") if l.strip()]
print("outcomes:", collections.Counter(r["outcome"] for r in rows))
print("cache_key None:", sum(r["cache_key"] == "None" for r in rows), "/", len(rows))
PY
```

Expected (fix working): a mix of `cold_compile` (first time each config is seen) + `disk_reload`
(rebench of already-seen configs), and `cache_key None: 0`. Before the fix it was 100% cold /
100% None.

## Measured baseline (single B200, effort=full, 600s, seed=2000)

`4096×4096×4096` fp8, distinct configs searched (all four CuTe cells hit the wall-clock budget):

| mode | before | with fix |
|---|--:|--:|
| subprocess | 79 | **169** |
| in-process (fix inert) | 166 | 170 |

Triton fp8 (same kernel, different backend) searches ~576 in the same budget — the residual gap to
Triton is inherent cutlass-DSL compile cost (~3.4s/config vs ~0.27s), not the autotuner.

## Notes / TODO before upstreaming

- The probe (`HELION_RANK0_COMPILE_PROBE`) is measurement scaffolding — fine to keep for now, remove
  in a follow-up PR.
- Not yet run: the full `cute-verify` suite. Do that before opening a PR.
- Fix #1 defaults off (byte-identical to current when unset); fix #2 is always on but inert off the
  subprocess+CuTe path.
