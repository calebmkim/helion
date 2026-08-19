# H100 Reduction-Heuristic Audit Analysis

The fixed 114-cell audit ran on one NVIDIA H100 80GB (`sm90`) at Helion
`2be976894bb434724726cd1064c7eee57210faa6`. Ratios are comparison latency
divided by seed latency, so values above one favor the H100 reduction seed.

| Cohort | Cells | default / seed | torch.compile / seed | AOT / seed |
|---|---:|---:|---:|---:|
| General AOT | 24 | 1.135 (24 valid) | 1.054 (24 valid) | 0.936 (24 valid) |
| Original | 36 | 6.045 (35 valid) | 1.038 (35 valid) | n/a |
| vLLM | 54 | 2.016 (54 valid) | 1.386 (53 valid) | 0.965 (54 valid) |

## Findings

- The seed is a large improvement over the unseeded Helion default. It wins
  16/24 general-AOT cells, all 35 valid original-kernel cells, and 49/54 vLLM
  cells.
- Against `torch.compile`, the seed wins 13/24 general cells, 21/35 valid
  original cells, and 49/53 valid vLLM cells. JSD is the clearest exception:
  its `G_tc` geomean is 0.822, so compiled torch is consistently faster.
- Checked-in H100 tuning still has value. AOT latency is 0.936x seed in the
  general cohort and 0.965x seed in the vLLM cohort. The seed wins only 8/24
  and 16/54 of those comparisons, respectively.
- Cross Entropy is the main general-kernel gap (`G_aot` geomean 0.798). The
  seed often uses 32 warps where AOT uses 16. At vocabulary 152064 the seed
  also emits 372-byte spill stores and 460-byte spill loads, while the AOT
  reduction tile emits no spills.
- Large vLLM losses come from structural config differences. At the largest
  shapes, dynamic quantization uses 16 seed warps versus 8 AOT warps, RMSNorm
  per-block quant uses 16 versus 4, and SiLU per-block quant uses block 64 with
  93-96 registers versus AOT block 8 with 32 registers.
- Fused QK Norm + RoPE `(8192, 64, 8)` uses a flat 4-warp seed kernel with 232
  registers. AOT selects a 1-warp persistent kernel with 56 registers and has
  0.872x the seed latency.

## Correctness

There were no cell, compile, capture, graph-break, heuristic no-fire, or AOT
selector failures. Three arm checks failed and remain excluded from ratios:

- RMSNorm backward `(2048, 11008)`: seed and default each miss tolerance on 21
  of 22,544,384 `grad_x` values (0.0000931%). Maximum absolute error is 0.0625;
  mean absolute error is 0.00205.
- Compiled Fused QK Norm + RoPE `(8192, 32, 8)`: 1 of 50,331,648 QKV values
  misses tolerance (0.00000199%). Maximum absolute error is 0.03125.

These diagnostics do not change the raw failure status or headline results.

## Noise

Forty-two arm measurements remain above 5% relative spread after escalation to
15 rounds. Every one is a microsecond-scale result at or below 11.904 us; the
maximum spread is 10.33%. All other arms use the specified nine-round protocol.

See `REPORT.md` and `summary.json` for the complete results. `INVESTIGATION.md`,
`investigation.json`, and `investigation_codegen/` contain the post-run config,
register, spill, resource, launcher, and PTX inspection.
