# Seed-config manifest — what the heuristic emits, per cell (change-detection tool)

For every `(corpus, kernel, dtype, shape)` over the **full** shape matrix (all splits —
train/val/test/robustness, both dtypes; 892 cells), this records the config the reduction-seed
heuristic actually emits. Use it to answer "did my heuristic change flip any configs, and which?"

## Files
- **`config_manifest.json`** — the machine-readable record. Per cell: `raw_seed` (what the
  heuristic decided, pre-normalize), `normalized_seed` (what actually *runs* after
  `spec.normalize` — this is the field the differ keys on), `base_default` (the unseeded compiler
  base, normalized), `configs_differ`, `fired_heuristics`, `classification`
  (T1_rolled / T2_usertiled / materialized / no_reduction_fact / gemm), `reduction_fact` (the
  fact fields the heuristic keyed on), `n_seeds` (0 ⇒ declined).
- **`CONFIG_MANIFEST.txt`** — a grep-friendly one-line-per-cell view (regenerate from the JSON
  with `manifest_index.py`).

## Regenerate the baseline manifest
Front-end only (bind → `compiler_seed_configs` → `normalize`); NO codegen/ptxas/timing, so it
runs in a few minutes over the whole matrix and is immune to the ptxas hang that bites timing.

```
cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
  PYTHONPATH=/home/dev/local/helion-redesign \
  /home/dev/helion/.venv/bin/python \
  /home/dev/local/helion-redesign/_lab/perf_report/config_manifest.py \
  --out /home/dev/local/prompts-lab/perf-report/results/config_manifest.json
# then refresh the text index:
/home/dev/helion/.venv/bin/python \
  /home/dev/local/helion-redesign/_lab/perf_report/manifest_index.py \
  .../config_manifest.json > .../CONFIG_MANIFEST.txt
```
Subset while iterating: `--corpus curriculum,vllm --kernels rms_norm,softmax`.

## Did my heuristic change flip any configs?
1. Record a BEFORE manifest (or reuse the committed `config_manifest.json` as the baseline).
2. Make your heuristic edit.
3. Record an AFTER manifest to a new path.
4. Diff:
```
... config_manifest.py --diff BEFORE.json AFTER.json
```
- **ZERO-DIFF** (exit 0): no cell's config, fired-heuristic, or classification changed —
  a behavior-preserving edit (perf verdicts transfer).
- **CHANGED** (exit 1): prints each changed cell with the field and `before -> after`. This is
  your re-bench worklist (the changed cells are exactly the ones whose perf can move).

The differ keys on `normalized_seed`, `raw_seed`, `base_default`, `fired_heuristics`,
`classification`, `n_seeds`, `configs_differ` — so a change that only alters the *raw* seed but
normalizes to the same running config is still surfaced (as a `raw_seed` diff) but distinguished
from a `normalized_seed` diff.

## Baseline provenance
Recorded on branch `reduction-redesign` @ `4c201983`, one dedicated H100. 892 cells ok, 2 errored
(`synth_store_bandwidth`, `synth_arith_intensity` — frozen-flag kernel-source compile errors, not
the heuristic). 1 cell declines (`oos1-jagged-declined`, n_seeds=0 — intended: data-dependent
extent is out of scope). Seed config differs from the unseeded default in 888/892 cells.
