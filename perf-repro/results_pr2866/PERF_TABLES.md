# PR #2866 pointwise seed heuristic — per-(kernel, dtype) perf summary

**Shapes run per kernel:** [`../shapes.json`](../shapes.json) — the exact shape list for every
kernel (which shapes reproduce the PR's numbers, `pointwise`, vs. which are out-of-sample,
`pointwise_gen`). Full method + reproduction detail: [`REPORT.md`](REPORT.md); full per-cell
breakdown: [`SUMMARY.md`](SUMMARY.md).

**What the numbers mean.** Each value is a geomean over that kernel's shapes of a latency ratio
where **> 1 means the heuristic's seeded config is faster** than the baseline:
- **vs. default** — the *unseeded* Helion compiler default (`_base_default_config`, block_size=32);
  this is what the seed heuristic buys over doing nothing.
- **vs. torch.compile** — `torch.compile` in **default mode** (NOT `max-autotune`) of the
  equivalent standalone PyTorch reference (pre-projected `[M,N]` tensors — not a full MLP wrapper).

| Kernel | dtype | source | vs. default | vs. torch.compile |
|---|---|---|---|---|
| swiglu | bf16 | `examples/swiglu.py` | 9.28 | 1.00 |
| geglu | bf16 | `examples/geglu.py` | 9.27 | 1.00 |
| residual_add | bf16 | `examples/add.py` | 1.28 | 0.98 |
| rope_fwd | bf16 | `examples/rope.py` | 16.80 | 1.09 |
| relu_squared | bf16 | [authored](../deps/pointwise_kernels.py) | 13.67 | 1.00 |
| bias_gelu | bf16 | [authored](../deps/pointwise_kernels.py) | 1.13 | 0.96 |
| dyt | bf16 | [authored](../deps/pointwise_kernels.py) | 1.15 | 0.94 |
| heavy_transcendental_1d | bf16 | [authored](../deps/pointwise_kernels.py) | 2.85 | 1.01 |
| transposed_out_add | bf16 | [authored](../deps/pointwise_kernels.py) | 1.17 | 0.97 |

_`rope_fwd` vs. torch.compile is over 4 of its 8 shapes: on the other 4, `torch.compile` itself hit
a 150s Inductor compile timeout (the Helion seed compiled and ran on all 8, beating default
13–19x). `vs. default` is over all 8. See [`REPORT.md`](REPORT.md) §5._
