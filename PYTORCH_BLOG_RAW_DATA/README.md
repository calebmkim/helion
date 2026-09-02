# Per-Cell Normalized Data

These CSV files are slim exports of the per-cell measurements behind the
linear-attention and matmul/multi-matmul results. They contain absolute
latencies in microseconds and normalized speedups for every cell.

For every normalized column:

```text
normalized speedup = baseline latency / arm latency
```

The baseline is therefore `1.0`, and values above `1.0` are faster than the
baseline. The CSVs retain the source JSON's floating-point precision and row
order. They are per-cell measurements, not aggregate geomeans.

## Files

| File | Cells | Baseline | Other arms |
|---|---:|---|---|
| [linear_attention_e2e_per_cell.csv](linear_attention_e2e_per_cell.csv) | 96 | Pre-change compiler-selected default | New heuristic, full-autotuned AOT config, handwritten FLA Triton |
| [matmul_multimatmul_on_corpus_per_cell.csv](matmul_multimatmul_on_corpus_per_cell.csv) | 149 | Raw compiler base default | New heuristic, per-cell pre-tuned reference |
| [matmul_multimatmul_held_out_per_cell.csv](matmul_multimatmul_held_out_per_cell.csv) | 75 | Pre-change compiler selection | New heuristic, quick-autotuned config |

## Baseline Details

The word "default" has different precise meanings in the source experiments:

- **End-to-end linear attention:** the baseline is the configuration selected
  by the pre-change compiler. It is not necessarily the raw unseeded
  `_base_default_config()`.
- **On-corpus constituent kernels:** all 149 baselines have
  `config_origin=default_config`, so this is the raw compiler base default.
- **Held-out constituent kernels:** the baseline is the pre-change compiler
  selection. It is the raw base default for 60 cells and an existing formula
  matmul heuristic for 15 cells. The CSV includes `baseline_origin` and
  `baseline_heuristic_name` so those cases remain explicit.

The held-out export uses the complete three-arm
`off-corpus-main-autotune/1` replay, where all 75 arms were timed under one
compatible protocol. A later unseeded full-autotune study is not merged into
this table because it has no same-protocol default arm; normalizing that run
to this run's default latency would create a cross-run ratio.

## Columns

Each CSV starts with stable cell identity and shape metadata, followed by:

- arm scores normalized to the file's baseline;
- useful direct pairwise ratios, such as pre-tuned versus heuristic;
- the absolute latency for each arm in microseconds;
- baseline and heuristic provenance where applicable.

The companion [shape inventory](../PYTORCH_BLOG_BENCHMARK_SHAPES.md) expands
the shape names and jagged sequence-length lists.

## Source Snapshots

| Export | Source schema | Source SHA-256 |
|---|---|---|
| End-to-end linear attention | `linear-attention-e2e-fla-results/3` | `41991df359b93d8b1076666eacd11418b3e505f21af92bc0e675821690ad35e5` |
| On-corpus constituent kernels | `matmul-heuristic-pretuned-results/1` | `ab05704307f4defc2add92642f0f3995e8b4750519537245334fe9878419e7a0` |
| Held-out constituent kernels | `off-corpus-main-autotune-results/1` | `8c9be14d79c4e87e7417fd79add22ebdc11dded95338b3674e9dd6d4c0c0dfdc` |
