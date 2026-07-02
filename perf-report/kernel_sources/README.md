# Kernel sources — reduction-seed perf report

Source for every kernel the reduction seed heuristic is tested on. Grouped by corpus. See
`../KERNEL_INVENTORY.md` for shape counts, dtypes, and the shape-definition files.

## Where each corpus's source lives

- **curriculum/ (the 9 original reduction kernels)** — NOT copied here; they are upstream Helion
  examples. Point to `helion/examples/`:
  `rms_norm.py`, `layer_norm.py`, `softmax.py`, `welford.py`, `sum.py`, `cross_entropy.py`,
  `kl_div.py`, `jsd.py`, `long_sum.py`. Kernel→fn map: `helion-redesign/_lab/harness/run2_measure_g.py`.
- **transfer/** — `transfer_kernels.py` holds 6 (`fused_add_rmsnorm`, `fused_add_layernorm`,
  `gated_rmsnorm`, `scaled_masked_softmax`, `cross_entropy_ls_zloss`, `dynamic_quant`); the other 2
  (`fused_linear_jsd`, `grpo`) are upstream `helion/examples/{fused_linear_jsd,grpo_loss}.py`.
  `shapes_transfer.py` = the shape lists.
- **mreduction/** — `mreduction_styles_view_only.py` holds `bias_grad_bwd`, `dyt_bwd`,
  `group_norm_bwd`, `instance_norm_bwd`; `rms_norm_bwd`/`layer_norm_bwd` are upstream
  `helion/examples/{rms_norm,layer_norm}.py`.
- **vllm/kut/** — the 5 quantization kernels under test (`silu_mul_fp8`,
  `dynamic_per_token_scaled_fp8_quant`, `rms_norm_dynamic_per_token_quant`,
  `per_token_group_fp8_quant`, `rms_norm_per_block_quant`). `refs.py` = the torch reference impls.
- **synthetic_probes/** — 13 categorization stress-test kernels (11 `p*` + 2 `oos*`), each a dir with
  `kernel.py`. Correctness/generality probes, NOT perf workloads (see inventory caveat).
- **adversarial_synth/** — 7 heuristic-generality probes; meant to be swept over N (persist-vs-chunk),
  not run at one shape.

These are COPIES (snapshot at report time). The live source is in the respective worktrees /
`prompts-lab/` dirs named above.
