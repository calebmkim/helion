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

Each row fuses that kernel's **in-sample** (PR-reproduce) shapes **and** its **out-of-sample**
(never-fitted) shapes into one geomean. All kernels are bf16. Metric = cold-L2 **interleaved**
median-of-9 event timing; the 4 cells below the eager timing floor (see [`REPORT.md`](REPORT.md) §5)
are scored on cudagraph device time.

| Kernel | dtype | source | vs. default | vs. torch.compile | shapes (in + out) |
|---|---|---|---|---|---|
| swiglu | bf16 | `examples/swiglu.py` | 9.28 | 1.00 | 30 (23 + 7) |
| geglu | bf16 | `examples/geglu.py` | 9.27 | 1.00 | 22 (17 + 5) |
| residual_add | bf16 | `examples/add.py` | 1.28 | 0.98 | 21 (16 + 5) |
| rope_fwd | bf16 | `examples/rope.py` | 16.80 | 1.09 | 8 (2 + 6) |
| relu_squared | bf16 | extra (authored) | 13.67 | 1.00 | 16 (12 + 4) |
| bias_gelu | bf16 | extra (authored) | 1.13 | 0.96 | 20 (15 + 5) |
| dyt | bf16 | extra (authored, held-out kernel) | 1.15 | 0.94 | 18 (0 + 18) |
| heavy_transcendental_1d | bf16 | extra (authored, SFU-ramp lever) | 2.85 | 1.01 | 6 (1 + 5) |
| transposed_out_add | bf16 | extra (authored, coalescing lever) | 1.17 | 0.97 | 7 (1 + 6) |

**Source column.** `examples/*` = the in-tree Helion example kernel (the shipping op). `extra
(authored)` = a standalone kernel the PR #2866 benchmarks but that does not live in `examples/`;
these are vendored from the PR's lab branch ([`../deps/pointwise_kernels.py`](../deps/pointwise_kernels.py)):
`relu_squared` / `bias_gelu` are two of the five rows in the PR's headline flat-family table, `dyt`
is the PR's held-out-kernel generalization claim (PR: 1.16x), and `heavy_transcendental_1d` /
`transposed_out_add` are the exact kernels the PR names for the SFU-`num_warps`-ramp and
contiguity/coalescing levers.

_`rope_fwd` vs. torch.compile is over 4 of its 8 shapes: on the other 4, `torch.compile` itself hit
a 150s Inductor compile timeout (the Helion seed compiled and ran on all 8, beating default
13–19x). `vs. default` is over all 8. See [`REPORT.md`](REPORT.md) §5._
