# B200 Pointwise-Heuristic Audit

This directory reproduces a focused B200 pointwise experiment on current
Helion. See [PLAN.md](PLAN.md) for the kernel classification, shape curriculum,
AOT provenance, and timing contract.

## Run

All GPU commands are pinned to physical GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 python perf-repro/b200-pointwise-audit/run_all.py --smoke
CUDA_VISIBLE_DEVICES=1 python perf-repro/b200-pointwise-audit/run_all.py
python perf-repro/b200-pointwise-audit/aggregate.py
python perf-repro/b200-pointwise-audit/plot_performance.py
```

The driver is resumable. Pass `--force` to rerun completed rows or `--kernels`
with a comma-separated list to select kernels. The per-cell timeout defaults to
300 seconds and can be changed with `--timeout-seconds`.

## Arms

- `default`: Helion's unseeded base compiler default.
- `seed`: the current `triton_pointwise` heuristic.
- `torch_compile`: bare `torch.compile()` in default mode.
- `aot`: the selected pretuned config where available.
- `default_null`: a duplicate default arm used only to quantify timing noise.

Reported performance is `default latency / arm latency`; higher is faster.

## Results

After a full run:

- `results/full/raw/`: one complete JSON row per shape.
- `results/full/SUMMARY.md`: generated tables and diagnostics.
- `results/charts/`: separate cohort charts and a combined chart.
