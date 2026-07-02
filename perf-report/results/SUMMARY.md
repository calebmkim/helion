# Reduction-seed perf report — SUMMARY

3 arms per real cell: **seed** (heuristic) / **default** (unseeded base) / **tc** (torch.compile default). Metric = cold-L2 median-of-9 `do_bench`, forward-only, single-process, same tensors.

- `G_tc = tc_us / seed_us` (>1 ⇒ seed beats torch.compile)
- `G_def = default_us / seed_us` (>1 ⇒ seed beats the unseeded default — "what the heuristic buys")


## Headline aggregate (geomean over cells with both arms valid)

| scope | geo G_tc | geo G_def | n(G_tc) | n(G_def) |
|---|---|---|---|---|
| overall | 1.022 | 1.887 | 324 | 324 |
| bf16 | 0.995 | 1.821 | 146 | 146 |
| fp32 | 0.986 | 1.939 | 161 | 161 |
| native | 1.821 | 1.980 | 17 | 17 |

### Per-corpus

| corpus | geo G_tc | geo G_def | n(G_tc) | n(G_def) |
|---|---|---|---|---|
| curriculum | 1.003 | 2.203 | 120 | 120 |
| transfer | 1.005 | 1.357 | 160 | 160 |
| mreduction | 0.858 | 6.483 | 27 | 27 |
| vllm | 1.821 | 1.980 | 17 | 17 |

## Per-shape disasters (realistic shape with G_tc < 0.75)

| corpus | kernel | shape | dtype | G_tc |
|---|---|---|---|---|
| mreduction | bias_grad_bwd | [2048, 1024] | bf16 | 0.369 |
| mreduction | instance_norm_bwd | [128, 32, 256] | bf16 | 0.379 |
| mreduction | instance_norm_bwd | [64, 16, 128] | bf16 | 0.389 |
| mreduction | dyt_bwd | [2048, 1024] | bf16 | 0.400 |
| curriculum | long_sum | [8, 2097152] | fp32 | 0.475 |
| curriculum | cross_entropy | [4096, 151936] | fp32 | 0.534 |
| curriculum | long_sum | [8, 2097152] | bf16 | 0.538 |
| curriculum | cross_entropy | [1024, 250000] | fp32 | 0.553 |
| curriculum | cross_entropy | [1024, 128256] | fp32 | 0.556 |
| curriculum | cross_entropy | [4096, 114688] | fp32 | 0.562 |
| vllm | silu_mul_fp8 | [128, 14336] | native | 0.575 |
| mreduction | dyt_bwd | [2048, 1024] | fp32 | 0.577 |
| vllm | silu_mul_fp8 | [32, 14336] | native | 0.583 |
| mreduction | group_norm_bwd | [128, 64, 64, 8] | bf16 | 0.590 |
| mreduction | instance_norm_bwd | [64, 16, 128] | fp32 | 0.592 |
| mreduction | instance_norm_bwd | [128, 32, 256] | fp32 | 0.599 |
| mreduction | bias_grad_bwd | [2048, 1024] | fp32 | 0.604 |
| transfer | cross_entropy_ls_zloss | [4096, 102400] | fp32 | 0.625 |
| transfer | cross_entropy_ls_zloss | [2048, 151936] | fp32 | 0.630 |
| transfer | cross_entropy_ls_zloss | [2048, 256000] | fp32 | 0.647 |
| transfer | cross_entropy_ls_zloss | [8192, 128256] | fp32 | 0.650 |
| transfer | cross_entropy_ls_zloss | [4096, 128256] | fp32 | 0.651 |
| transfer | cross_entropy_ls_zloss | [2048, 256000] | bf16 | 0.655 |
| mreduction | layer_norm_bwd | [2048, 4096] | bf16 | 0.659 |
| curriculum | long_sum | [48, 786432] | bf16 | 0.704 |
| curriculum | cross_entropy | [4096, 151936] | bf16 | 0.717 |
| transfer | scaled_masked_softmax | [1024, 65536] | fp32 | 0.724 |
| transfer | scaled_masked_softmax | [512, 131072] | fp32 | 0.733 |
| transfer | scaled_masked_softmax | [512, 131072] | bf16 | 0.741 |
| mreduction | bias_grad_bwd | [4096, 8192] | bf16 | 0.742 |

## (A) Real workloads — per-(kernel, dtype) geomeans


### curriculum

| kernel | dtype | geo G_tc | geo G_def | shapes | acc-pass | min G_tc |
|---|---|---|---|---|---|---|
| cross_entropy | bf16 | 1.098 | 1.245 | 7 | 7 | 0.717 |
| cross_entropy | fp32 | 0.729 | 1.360 | 7 | 7 | 0.534 |
| jsd | bf16 | 0.813 | 3.932 | 7 | 7 | 0.789 |
| jsd | fp32 | 1.034 | 3.874 | 7 | 7 | 0.991 |
| kl_div | bf16 | 1.085 | 7.412 | 7 | 7 | 0.928 |
| kl_div | fp32 | 1.085 | 5.806 | 7 | 7 | 1.000 |
| layer_norm | bf16 | 1.038 | 1.028 | 8 | 8 | 0.925 |
| layer_norm | fp32 | 1.002 | 1.083 | 8 | 8 | 0.967 |
| long_sum | bf16 | 0.934 | 2.601 | 7 | 7 | 0.538 |
| long_sum | fp32 | 0.933 | 2.438 | 7 | 7 | 0.475 |
| rms_norm | bf16 | 1.034 | 1.032 | 8 | 8 | 0.975 |
| rms_norm | fp32 | 1.000 | 1.082 | 8 | 8 | 0.980 |
| softmax | bf16 | 1.256 | 4.810 | 8 | 8 | 0.882 |
| softmax | fp32 | 1.143 | 4.057 | 8 | 8 | 0.938 |
| sum | bf16 | 1.000 | 1.119 | 7 | 2 | 0.992 |
| sum | fp32 | 0.971 | 1.048 | 7 | 7 | 0.934 |
| welford | bf16 | None | None | 7 | 0 | None |
| welford | fp32 | 0.956 | 2.603 | 7 | 7 | 0.908 |

### transfer

| kernel | dtype | geo G_tc | geo G_def | shapes | acc-pass | min G_tc |
|---|---|---|---|---|---|---|
| cross_entropy_ls_zloss | bf16 | 0.973 | 1.632 | 11 | 11 | 0.655 |
| cross_entropy_ls_zloss | fp32 | 0.938 | 1.778 | 11 | 11 | 0.625 |
| dynamic_quant | bf16 | 1.121 | 1.146 | 8 | 8 | 1.001 |
| dynamic_quant | fp32 | 0.979 | 1.189 | 8 | 8 | 0.902 |
| fused_add_layernorm | bf16 | 1.074 | 1.081 | 12 | 12 | 0.910 |
| fused_add_layernorm | fp32 | 0.988 | 1.139 | 12 | 12 | 0.948 |
| fused_add_rmsnorm | bf16 | 0.994 | 1.115 | 12 | 12 | 0.986 |
| fused_add_rmsnorm | fp32 | 1.004 | 1.133 | 12 | 12 | 0.990 |
| fused_linear_jsd | bf16 | 0.873 | 1.421 | 7 | 7 | 0.770 |
| fused_linear_jsd | fp32 | 1.168 | 1.760 | 7 | 7 | 0.865 |
| gated_rmsnorm | bf16 | 0.992 | 1.026 | 12 | 12 | 0.852 |
| gated_rmsnorm | fp32 | 0.992 | 1.088 | 12 | 12 | 0.914 |
| grpo | bf16 | 1.105 | 2.992 | 7 | 7 | 0.818 |
| grpo | fp32 | 0.947 | 4.271 | 7 | 7 | 0.845 |
| scaled_masked_softmax | bf16 | 1.064 | 1.193 | 11 | 11 | 0.741 |
| scaled_masked_softmax | fp32 | 0.945 | 1.185 | 11 | 11 | 0.724 |

### mreduction

| kernel | dtype | geo G_tc | geo G_def | shapes | acc-pass | min G_tc |
|---|---|---|---|---|---|---|
| bias_grad_bwd | bf16 | 0.612 | 1.176 | 3 | 3 | 0.369 |
| bias_grad_bwd | fp32 | 0.794 | 1.082 | 3 | 3 | 0.604 |
| dyt_bwd | bf16 | 0.873 | 7.421 | 3 | 3 | 0.400 |
| dyt_bwd | fp32 | 1.077 | 7.348 | 3 | 3 | 0.577 |
| group_norm_bwd | bf16 | 0.590 | 6.473 | 2 | 2 | 0.590 |
| group_norm_bwd | fp32 | 0.876 | 11.656 | 2 | 2 | 0.876 |
| instance_norm_bwd | bf16 | 0.384 | 10.093 | 2 | 2 | 0.379 |
| instance_norm_bwd | fp32 | 0.595 | 15.683 | 2 | 2 | 0.592 |
| layer_norm_bwd | bf16 | 1.040 | 16.039 | 3 | 3 | 0.659 |
| layer_norm_bwd | fp32 | 1.317 | 14.018 | 3 | 3 | 1.162 |
| rms_norm_bwd | bf16 | None | None | 3 | 0 | None |
| rms_norm_bwd | fp32 | 1.338 | 10.486 | 3 | 3 | 1.281 |

### vllm

| kernel | dtype | geo G_tc | geo G_def | geo G_vllm | shapes | acc-pass | min G_tc |
|---|---|---|---|---|---|---|---|
| dynamic_per_token_scaled_fp8_quant | native | 2.698 | 2.459 | 1.017 | 4 | 4 | 2.368 |
| per_token_group_fp8_quant | native | 1.097 | 0.989 | 1.034 | 4 | 1 | 1.097 |
| rms_norm_dynamic_per_token_quant | native | 2.342 | 5.267 | 1.039 | 4 | 4 | 2.114 |
| rms_norm_per_block_quant | native | 2.600 | 1.340 | 1.004 | 4 | 4 | 2.368 |
| silu_mul_fp8 | native | 0.760 | 1.053 | 1.031 | 4 | 4 | 0.575 |

(`geo G_vllm` = seed vs vLLM's own shipped H100-tuned config, nearest-shape lookup; >1 ⇒ seed beats vLLM's tuned config.)

## (B) Generality diagnostics (seed vs default, G_def only — NOT headline perf)

| corpus | kernel | shape | G_def | status |
|---|---|---|---|---|
| adversarial_synth | synth_l2_vs_reg_twograph_scalar | [2048, 49152] | 7.105 | ok |
| adversarial_synth | synth_livecount_scalar_out | [2048, 32768] | 7.285 | ok |
| adversarial_synth | synth_reread_softmax_VERIFIED_wrongcap | [2048, 49152] | 6.006 | ok |
| adversarial_synth | synth_reread_variance_NOT_wrongcap | [2048, 49152] | 7.444 | ok |
| adversarial_synth | synth_working_set_undercount | [2048, 40960] | 10.823 | ok |
| synthetic_probes | oos1-jagged-declined | [78991, 128] | n/a | declined |
| synthetic_probes | oos2-strided-dim0 | [8192, 32000] | 1.043 | ok |
| synthetic_probes | p1-outer-product-coresident | [2048, 64, 128] | 0.995 | ok |
| synthetic_probes | p10-usertile-and-gridtile | [2048, 4096] | 25.480 | ok |
| synthetic_probes | p11-fullextent-then-nonred-loop | [2048, 4096] | 31.858 | ok |
| synthetic_probes | p2-feature-plus-rowaccum-offcorpus | [2048, 4096] | 1.031 | ok |
| synthetic_probes | p3-full-grid-nonquant | [8192, 32, 128] | 1.154 | ok |
| synthetic_probes | p4-two-rollable-sequential | [2048, 4096] | 1.015 | ok |
| synthetic_probes | p5-3d-reduction-tile | [4096, 64, 64] | 1.627 | ok |
| synthetic_probes | p6-mixed-coresident-plus-sequential | [2048, 4096] | 0.870 | ok |
| synthetic_probes | p7-gridtile-then-usertile | [2048, 4096] | 2.023 | ok |
| synthetic_probes | p8-fullgrid-plus-usertile | [8192, 32, 128] | 2.899 | ok |
| synthetic_probes | p9-nonred-loop-then-fullextent | [2048, 4096] | 23.728 | ok |

## Acc-fail cells — perf still measured (18 cells)

These cells fail the accuracy gate (excluded from the headline geomeans, footgun #6a) but timing is recorded anyway. `both_fail=yes` ⇒ seed AND default fail the gate identically, so the perf ratios below are still an apples-to-apples comparison (the failure is a kernel-source fact — bf16 accumulator / fp8 boundary — not a seed-specific wrong answer).

| corpus | kernel | shape | dtype | seed_us | def_us | tc_us | perf seed/tc | perf seed/def | both_fail |
|---|---|---|---|---|---|---|---|---|---|
| curriculum | sum | [16384, 1280] | bf16 | 21.9 | 23.9 | 24.9 | 1.136 | 1.091 | yes |
| curriculum | sum | [8192, 2560] | bf16 | 21.6 | 24.0 | 23.8 | 1.104 | 1.110 | yes |
| curriculum | sum | [8192, 7168] | bf16 | 47.3 | 51.6 | 45.6 | 0.965 | 1.091 | NO — ok |
| curriculum | sum | [4096, 10240] | bf16 | 35.6 | 40.7 | 34.9 | 0.981 | 1.145 | NO — ok |
| curriculum | sum | [16384, 3072] | bf16 | 41.6 | 45.9 | 42.0 | 1.011 | 1.104 | yes |
| curriculum | welford | [16384, 896] | bf16 | 27.6 | 66.8 | 26.5 | 0.957 | 2.418 | NO — ok |
| curriculum | welford | [16384, 1280] | bf16 | 42.0 | 95.9 | 50.2 | 1.196 | 2.282 | NO — ok |
| curriculum | welford | [8192, 3584] | bf16 | 46.7 | 209.6 | 52.7 | 1.127 | 4.485 | NO — ok |
| curriculum | welford | [16384, 5120] | bf16 | 158.9 | 366.9 | 122.5 | 0.771 | 2.309 | NO — ok |
| curriculum | welford | [8192, 14336] | bf16 | 201.2 | 817.8 | 161.8 | 0.804 | 4.064 | NO — ok |
| curriculum | welford | [32768, 2560] | bf16 | 162.6 | 305.7 | 154.7 | 0.951 | 1.880 | NO — ok |
| curriculum | welford | [16384, 7168] | bf16 | 192.2 | 510.5 | 160.7 | 0.836 | 2.656 | NO — ok |
| mreduction | rms_norm_bwd | [2048, 4096] | bf16 | 56.5 | 685.0 | 43.9 | 0.777 | 12.129 | yes |
| mreduction | rms_norm_bwd | [8192, 4096] | bf16 | 95.9 | 1329.4 | 136.9 | 1.427 | 13.858 | yes |
| mreduction | rms_norm_bwd | [4096, 8192] | bf16 | 96.3 | 1867.5 | 136.1 | 1.414 | 19.402 | yes |
| vllm | per_token_group_fp8_quant | [8192, 4096, 128] | native | 190.0 | 117.9 | 117.3 | 0.618 | 0.621 | yes |
| vllm | per_token_group_fp8_quant | [8192, 7168, 128] | native | 194.3 | 200.7 | 192.4 | 0.990 | 1.033 | yes |
| vllm | per_token_group_fp8_quant | [2048, 8192, 128] | native | 65.5 | 66.6 | 66.5 | 1.015 | 1.016 | yes |

## x / n/a cells (33 arm-level entries)

| corpus | kernel | shape | dtype | arm | reason |
|---|---|---|---|---|---|
| adversarial_synth | synth_arith_intensity | None | None | bind/build | IndexError: list index out of range |
| adversarial_synth | synth_store_bandwidth | None | None | bind/build | ControlFlowTensorMismatch: Tensor mismatch in control flow for variable 'out': r |
| curriculum | sum | [16384, 1280] | bf16 | seed | acc-fail (maxabs=1.000e+00) |
| curriculum | sum | [16384, 1280] | bf16 | default | acc-fail (maxabs=1.000e+00) |
| curriculum | sum | [8192, 2560] | bf16 | seed | acc-fail (maxabs=1.500e+00) |
| curriculum | sum | [8192, 2560] | bf16 | default | acc-fail (maxabs=1.500e+00) |
| curriculum | sum | [8192, 7168] | bf16 | seed | acc-fail (maxabs=3.000e+00) |
| curriculum | sum | [4096, 10240] | bf16 | seed | acc-fail (maxabs=4.000e+00) |
| curriculum | sum | [16384, 3072] | bf16 | seed | acc-fail (maxabs=2.000e+00) |
| curriculum | sum | [16384, 3072] | bf16 | default | acc-fail (maxabs=2.000e+00) |
| curriculum | welford | [16384, 896] | bf16 | seed | acc-fail (maxabs=7.812e-02) |
| curriculum | welford | [16384, 1280] | bf16 | seed | acc-fail (maxabs=7.812e-02) |
| curriculum | welford | [8192, 3584] | bf16 | seed | acc-fail (maxabs=7.422e-02) |
| curriculum | welford | [16384, 5120] | bf16 | seed | acc-fail (maxabs=7.031e-02) |
| curriculum | welford | [8192, 14336] | bf16 | seed | acc-fail (maxabs=5.859e-02) |
| curriculum | welford | [32768, 2560] | bf16 | seed | acc-fail (maxabs=8.203e-02) |
| curriculum | welford | [16384, 7168] | bf16 | seed | acc-fail (maxabs=7.031e-02) |
| mreduction | group_norm_bwd | [256, 128, 128, 16] | fp32 | default | compile-fail:timeout |
| mreduction | group_norm_bwd | [256, 128, 128, 16] | fp32 | tc | compile-fail:InductorError |
| mreduction | group_norm_bwd | [256, 128, 128, 16] | bf16 | default | compile-fail:timeout |
| mreduction | group_norm_bwd | [256, 128, 128, 16] | bf16 | tc | compile-fail:InductorError |
| mreduction | rms_norm_bwd | [2048, 4096] | bf16 | seed | acc-fail (maxabs=1.000e+00) |
| mreduction | rms_norm_bwd | [2048, 4096] | bf16 | default | acc-fail (maxabs=1.000e+00) |
| mreduction | rms_norm_bwd | [8192, 4096] | bf16 | seed | acc-fail (maxabs=2.000e+00) |
| mreduction | rms_norm_bwd | [8192, 4096] | bf16 | default | acc-fail (maxabs=2.000e+00) |
| mreduction | rms_norm_bwd | [4096, 8192] | bf16 | seed | acc-fail (maxabs=1.000e+00) |
| mreduction | rms_norm_bwd | [4096, 8192] | bf16 | default | acc-fail (maxabs=1.000e+00) |
| vllm | per_token_group_fp8_quant | [8192, 4096, 128] | native | seed | acc-fail (out_q:exact=97.0%/rel=0.333 scale:exact=100.0%/rel=0) |
| vllm | per_token_group_fp8_quant | [8192, 4096, 128] | native | default | acc-fail (out_q:exact=97.0%/rel=0.333 scale:exact=100.0%/rel=0) |
| vllm | per_token_group_fp8_quant | [8192, 7168, 128] | native | seed | acc-fail (out_q:exact=97.0%/rel=1 scale:exact=100.0%/rel=0) |
| vllm | per_token_group_fp8_quant | [8192, 7168, 128] | native | default | acc-fail (out_q:exact=97.0%/rel=1 scale:exact=100.0%/rel=0) |
| vllm | per_token_group_fp8_quant | [2048, 8192, 128] | native | seed | acc-fail (out_q:exact=97.0%/rel=1 scale:exact=100.0%/rel=0) |
| vllm | per_token_group_fp8_quant | [2048, 8192, 128] | native | default | acc-fail (out_q:exact=97.0%/rel=1 scale:exact=100.0%/rel=0) |
