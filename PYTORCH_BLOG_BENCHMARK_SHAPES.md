# PyTorch Blog: Benchmark Shape Inventory

This companion to `PYTORCH_BLOG_HEURISTICS_RESULTS.md` records the shapes used
for the linear-attention and matmul/multi-matmul results. It separates three
different populations that should not be conflated:

- 96 end-to-end linear-attention operation, mode, and shape cells.
- 149 on-corpus constituent-kernel cells where the formula-matmul or
  multi-matmul heuristic fired.
- 75 held-out (called `off_corpus` in the artifacts) cells across 15 kernel
  bodies.

All runs target NVIDIA B200 (`sm100`). The end-to-end linear-attention run is
BF16 throughout. Dtypes for the held-out cases are shown per row.

## 1. End-to-End Linear Attention

The seven dense variants (`vanilla_linear_attn`, `simple_gla`, `retention`,
`full_gla`, `delta_rule`, `gated_delta_rule`, and `kda`) run forward and
forward-plus-backward on all six dense shapes. `kda_fused` runs forward on the
same six shapes. `kda_varlen` runs forward on the six variable-length shapes.

### Dense Shapes

The runtime tensor tuple is `[B, H, T, D, DV]`, with `D == DV`.

| Name | B | H | T | D (= DV) |
|---|---:|---:|---:|---:|
| `B1_T8192_H96_D128` | 1 | 96 | 8192 | 128 |
| `B2_T16384_H16_D128` | 2 | 16 | 16384 | 128 |
| `B4_T2048_H16_D128` | 4 | 16 | 2048 | 128 |
| `B4_T4096_H64_D128` | 4 | 64 | 4096 | 128 |
| `B8_T2048_H32_D256` | 8 | 32 | 2048 | 256 |
| `B8_T1024_H8_D64` | 8 | 8 | 1024 | 64 |

### Variable-Length Shapes

Each case contains 8192 tokens. The ragged and uniform lists are the actual
per-sequence lengths, not buckets.

| Name | Sequence lengths | H | D |
|---|---|---:|---:|
| `fixed_T8192_H96_D128` | `[8192]` | 96 | 128 |
| `fixed_T8192_H64_D128` | `[8192]` | 64 | 128 |
| `ragged_T8192_H96_D128` | `[1300, 547, 2048, 963, 271, 3063]` | 96 | 128 |
| `ragged_T8192_H64_D128` | `[1300, 547, 2048, 963, 271, 3063]` | 64 | 128 |
| `uniform_T8192_H96_D128` | `[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]` | 96 | 128 |
| `uniform_T8192_H64_D128` | `[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]` | 64 | 128 |

This gives `7 x 6 x 2 + 6 + 6 = 96` cells.

## 2. On-Corpus Constituent Kernels

The constituent-kernel report includes 149 cells where
`triton_b200_formula_matmul` or `triton_b200_multi_matmul` fired: 124 Helion
linear-attention cells and 25 SGLang KDA cells. The broader harness measured
20 additional reduction-only cells, but those are excluded here because they
are not part of the matmul/multi-matmul result.

### Helion Linear Attention

Twenty dense kernel bodies use five cases each:

| Name | B | H | T | D |
|---|---:|---:|---:|---:|
| `B8_T1024_H8_D64` | 8 | 8 | 1024 | 64 |
| `B4_T2048_H16_D128` | 4 | 16 | 2048 | 128 |
| `B2_T16384_H16_D128` | 2 | 16 | 16384 | 128 |
| `B8_T2048_H32_D256` | 8 | 32 | 2048 | 256 |
| `B1_T8192_H96_D128` | 1 | 96 | 8192 | 128 |

Five variable-length kernel bodies use the following selection. Four bodies
have all five cases; `chunk_fwd_A_diag_anchored_varlen_helion` has only the
three H64 cases and the H96 ragged case because no tuned reference existed for
the H96 fixed or uniform cases.

| Profile | Sequence lengths | H | D |
|---|---|---:|---:|
| `fixed_T8192_H64_D128` | `[8192]` | 64 | 128 |
| `uniform_T8192_H64_D128` | `[1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024]` | 64 | 128 |
| `ragged_T8192_H64_D128` | `[1300, 547, 2048, 963, 271, 3063]` | 64 | 128 |
| `fixed_T8192_H96_D128` | `[8192]` | 96 | 128 |
| `ragged_T8192_H96_D128` | `[1300, 547, 2048, 963, 271, 3063]` | 96 | 128 |

The total is `20 x 5 + (5 + 4 + 5 + 5 + 5) = 124` cells.

### SGLang KDA

Four in-scope prefill bodies (`_intra_matrices_wide`,
`_intra_solve_recompute`, `_chunk_state`, and `_chunk_output`) use five cases
each. Activations are BF16; `_chunk_state` uses the state dtype shown below.

| Case/profile | Layout and shape | State dtype for `_chunk_state` |
|---|---|---|
| `fixed_h16_t512` | fixed: B=1, H=16, T=512, K=128, V=128 | FP32 |
| `fixed_h32_t8192` | fixed: B=1, H=32, T=8192, K=128, V=128 | BF16 |
| `packed_h12_t2048` | packed varlen: B=1, H=12, T=2048, K=128, V=128 | FP32 |
| `ragged_h16_t2048` | packed varlen: B=4, H=16, lengths=`[269, 461, 563, 755]`, K=128, V=128 | BF16 |
| `ragged_h32_t8192` | packed varlen: B=4, H=32, lengths=`[1037, 1805, 2291, 3059]`, K=128, V=128 | BF16 |

The in-scope ReplaySSM decode body uses five more cases:

| Case | B | H | K | V | Cache | Activation | State |
|---|---:|---:|---:|---:|---:|---|---|
| `float32_small_head_b1` | 1 | 12 | 128 | 128 | 16 | BF16 | FP32 |
| `bfloat16_small_head_b16` | 16 | 12 | 128 | 128 | 16 | BF16 | BF16 |
| `bfloat16_small_head_b128` | 128 | 12 | 128 | 128 | 16 | BF16 | BF16 |
| `bf16_regular_h16_b64` | 64 | 16 | 128 | 128 | 16 | BF16 | BF16 |
| `bf16_regular_h32_b128` | 128 | 32 | 128 | 128 | 16 | BF16 | BF16 |

This gives `4 x 5 + 5 = 25` SGLang cells.

## 3. Held-Out Matmul/Multi-Matmul Corpus

The held-out corpus has five cases for each of 15 kernel bodies: two
state-space bodies, eight attention bodies, and five matmul-like bodies. For
jagged cases, offsets are zero followed by cumulative sums of the listed
lengths, so the redundant offsets are omitted here.

### `off.state_space.gdn_fwd_h`

| Case | Shape | Dtype |
|---|---|---|
| `gdn_b1_t8192_h64_c64_d64_s128` | batch=1, seqlen=8192, nheads=64, chunk_size=64, dhead=64, dstate=128 | `bfloat16` |
| `gdn_b2_t8192_h32_c128_d128_s256` | batch=2, seqlen=8192, nheads=32, chunk_size=128, dhead=128, dstate=256 | `bfloat16` |
| `gdn_b4_t4096_h64_c128_d64_s128` | batch=4, seqlen=4096, nheads=64, chunk_size=128, dhead=64, dstate=128 | `bfloat16` |
| `gdn_b8_t2048_h32_c256_d64_s128` | batch=8, seqlen=2048, nheads=32, chunk_size=256, dhead=64, dstate=128 | `bfloat16` |
| `gdn_b8_t4096_h80_c256_d64_s128` | batch=8, seqlen=4096, nheads=80, chunk_size=256, dhead=64, dstate=128 | `bfloat16` |

### `off.state_space.mamba2_chunk_scan`

| Case | Shape | Dtype |
|---|---|---|
| `mamba_scan_b1_h64_g8_t8192_c64_d64_s128` | batch=1, nheads=64, ngroups=8, seqlen=8192, chunk_size=64, headdim=64, dstate=128 | `bfloat16` |
| `mamba_scan_b2_h32_g8_t8192_c128_d128_s256` | batch=2, nheads=32, ngroups=8, seqlen=8192, chunk_size=128, headdim=128, dstate=256 | `bfloat16` |
| `mamba_scan_b4_h64_g8_t4096_c128_d64_s128` | batch=4, nheads=64, ngroups=8, seqlen=4096, chunk_size=128, headdim=64, dstate=128 | `bfloat16` |
| `mamba_scan_b8_h32_g4_t2048_c256_d64_s128` | batch=8, nheads=32, ngroups=4, seqlen=2048, chunk_size=256, headdim=64, dstate=128 | `bfloat16` |
| `mamba_scan_b8_h80_g1_t4096_c256_d64_s128` | batch=8, nheads=80, ngroups=1, seqlen=4096, chunk_size=256, headdim=64, dstate=128 | `bfloat16` |

### `off.attention.dense_forward`

| Case | Shape | Dtype |
|---|---|---|
| `attention_b1_h4_m512_n512_d64_f16` | B=1, H=4, M=512, N=512, D=64 | `float16` |
| `attention_b2_h32_m1024_n1024_d64_f16` | B=2, H=32, M=1024, N=1024, D=64 | `float16` |
| `attention_b8_h16_m2048_n2048_d64_bf16` | B=8, H=16, M=2048, N=2048, D=64 | `bfloat16` |
| `attention_b4_h32_m4096_n4096_d128_bf16` | B=4, H=32, M=4096, N=4096, D=128 | `bfloat16` |
| `attention_b4_h16_m128_n4096_d64_bf16` | B=4, H=16, M=128, N=4096, D=64 | `bfloat16` |

### `off.attention.causal_forward`

| Case | Shape | Dtype |
|---|---|---|
| `causal_attention_b1_h8_s512_d64_f16` | B=1, H=8, S=512, D=64 | `float16` |
| `causal_attention_b2_h32_s1024_d64_f16` | B=2, H=32, S=1024, D=64 | `float16` |
| `causal_attention_b8_h16_s2048_d64_bf16` | B=8, H=16, S=2048, D=64 | `bfloat16` |
| `causal_attention_b4_h32_s4096_d128_bf16` | B=4, H=32, S=4096, D=128 | `bfloat16` |
| `causal_attention_b1_h16_s8192_d128_bf16` | B=1, H=16, S=8192, D=128 | `bfloat16` |

### `off.attention.biased_forward`

| Case | Shape | Dtype |
|---|---|---|
| `biased_attention_b1_h4_m128_n128_d64_f16` | B=1, H=4, M=128, N=128, D=64 | `float16` |
| `biased_attention_b2_h8_m512_n512_d64_f16` | B=2, H=8, M=512, N=512, D=64 | `float16` |
| `biased_attention_b2_h16_m1024_n1024_d64_f16` | B=2, H=16, M=1024, N=1024, D=64 | `float16` |
| `biased_attention_b1_h16_m1024_n2048_d128_bf16` | B=1, H=16, M=1024, N=2048, D=128 | `bfloat16` |
| `biased_attention_b4_h8_m256_n1024_d64_bf16` | B=4, H=8, M=256, N=1024, D=64 | `bfloat16` |

### `off.attention.backward`

| Case | Shape | Dtype |
|---|---|---|
| `attention_backward_b1_h4_s256_d64_f16` | B=1, H=4, S=256, D=64 | `float16` |
| `attention_backward_b2_h16_s512_d64_f16` | B=2, H=16, S=512, D=64 | `float16` |
| `attention_backward_b2_h32_s1024_d64_f16` | B=2, H=32, S=1024, D=64 | `float16` |
| `attention_backward_b4_h32_s2048_d128_bf16` | B=4, H=32, S=2048, D=128 | `bfloat16` |
| `attention_backward_b1_h16_s4096_d128_bf16` | B=1, H=16, S=4096, D=128 | `bfloat16` |

### `off.attention.blackwell_forward`

| Case | Shape | Dtype |
|---|---|---|
| `blackwell_attention_b1_h8_s512_d64_f16` | B=1, H=8, S=512, D=64 | `float16` |
| `blackwell_attention_b2_h16_s1024_d128_bf16` | B=2, H=16, S=1024, D=128 | `bfloat16` |
| `blackwell_attention_b4_h32_s2048_d64_bf16` | B=4, H=32, S=2048, D=64 | `bfloat16` |
| `blackwell_attention_b4_h32_s4096_d128_bf16` | B=4, H=32, S=4096, D=128 | `bfloat16` |
| `blackwell_attention_b1_h16_s8192_d128_bf16` | B=1, H=16, S=8192, D=128 | `bfloat16` |

### `off.attention.blackwell_backward`

| Case | Shape | Dtype |
|---|---|---|
| `blackwell_attention_backward_b1_h8_s256_d64_f16` | B=1, H=8, S=256, D=64 | `float16` |
| `blackwell_attention_backward_b2_h16_s512_d128_bf16` | B=2, H=16, S=512, D=128 | `bfloat16` |
| `blackwell_attention_backward_b2_h32_s1024_d64_f16` | B=2, H=32, S=1024, D=64 | `float16` |
| `blackwell_attention_backward_b4_h32_s2048_d128_bf16` | B=4, H=32, S=2048, D=128 | `bfloat16` |
| `blackwell_attention_backward_b1_h16_s4096_d128_bf16` | B=1, H=16, S=4096, D=128 | `bfloat16` |

### `off.attention.flex_forward`

| Case | Shape | Dtype |
|---|---|---|
| `flex_dense_b2_h32_s1024_d64_f16` | B=2, Hq=32, Hkv=32, M=1024, N=1024, D=64, dense mask | `float16` |
| `flex_causal_b2_h32_s2048_d64_f16` | B=2, Hq=32, Hkv=32, M=2048, N=2048, D=64, causal mask | `float16` |
| `flex_window_b4_h16_s4096_d128_bf16` | B=4, Hq=16, Hkv=16, M=4096, N=4096, D=128, causal sliding window=512 | `bfloat16` |
| `flex_gqa_b2_hq32_hkv8_s2048_d128_bf16` | B=2, Hq=32, Hkv=8, M=2048, N=2048, D=128, causal mask, GQA | `bfloat16` |
| `flex_softcap_cross_b2_h16_m512_n4096_d64_bf16` | B=2, Hq=16, Hkv=16, M=512, N=4096, D=64, dense mask, tanh softcap=30 | `bfloat16` |

### `off.attention.jagged_hstu`

| Case | Shape | Dtype |
|---|---|---|
| `jagged_hstu_b4_l320_max128_h8_d64` | H=8, D=64, max_seq_len=128, lengths=`[128, 96, 64, 32]` | `bfloat16` |
| `jagged_hstu_b16_l3440_max512_h16_d64` | H=16, D=64, max_seq_len=512, lengths=`[512, 480, 448, 384, 320, 256, 224, 192, 160, 128, 96, 80, 64, 48, 32, 16]` | `bfloat16` |
| `jagged_hstu_b8_l6656_max2048_h32_d128` | H=32, D=128, max_seq_len=2048, lengths=`[2048, 1536, 1024, 768, 512, 384, 256, 128]` | `bfloat16` |
| `jagged_hstu_b32_l2485_max256_h8_d128` | H=8, D=128, max_seq_len=256, lengths=`[256, 240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 72, 64, 56, 48, 40, 32, 28, 24, 20, 16, 14, 12, 10, 8, 8, 6, 4, 4, 2, 1]` | `bfloat16` |
| `jagged_hstu_b2_l7168_max4096_h16_d64` | H=16, D=64, max_seq_len=4096, lengths=`[4096, 3072]` | `bfloat16` |

### `off.matmul.broadcast`

| Case | Shape | Dtype |
|---|---|---|
| `broadcast_matmul_b32_m1_k4096_n4096` | B=32, M=1, K=4096, N=4096 | `bfloat16` |
| `broadcast_matmul_b8_m128_k4096_n11008` | B=8, M=128, K=4096, N=11008 | `bfloat16` |
| `broadcast_matmul_b16_m512_k768_n1024` | B=16, M=512, K=768, N=1024 | `bfloat16` |
| `broadcast_matmul_b4_m1024_k4096_n4096` | B=4, M=1024, K=4096, N=4096 | `bfloat16` |
| `broadcast_matmul_b64_m16_k1024_n4096` | B=64, M=16, K=1024, N=4096 | `bfloat16` |

### `off.matmul.gather_gemv`

| Case | Shape | Dtype |
|---|---|---|
| `gather_gemv_b8_s2048_n2` | B=8, S=2048, N=2 | `bfloat16` |
| `gather_gemv_b8_s4096_n2` | B=8, S=4096, N=2 | `bfloat16` |
| `gather_gemv_b8_s8192_n2` | B=8, S=8192, N=2 | `bfloat16` |
| `gather_gemv_b8_s14336_n2` | B=8, S=14336, N=2 | `bfloat16` |
| `gather_gemv_b64_s4096_n8` | B=64, S=4096, N=8 | `bfloat16` |

### `off.matmul.jagged_dense_bmm`

| Case | Shape | Dtype |
|---|---|---|
| `jagged_bmm_b8_l255_d128_k64_bias` | D=128, K=64, bias=true, lengths=`[1, 2, 4, 8, 16, 32, 64, 128]` | `bfloat16` |
| `jagged_bmm_b16_l2176_d256_k128_bias` | D=256, K=128, bias=true, lengths=`[256, 240, 224, 208, 192, 176, 160, 144, 128, 112, 96, 80, 64, 48, 32, 16]` | `bfloat16` |
| `jagged_bmm_b16_l2522_d256_k512_no_bias` | D=256, K=512, bias=false, lengths=`[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 512, 1024]` | `bfloat16` |
| `jagged_bmm_b32_l550_d768_k256_bias` | D=768, K=256, bias=true, lengths=`[64, 56, 48, 40, 36, 32, 28, 24, 22, 20, 18, 16, 15, 14, 13, 12, 11, 10, 9, 8, 8, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 1]` | `bfloat16` |
| `jagged_bmm_b4_l10240_d512_k512_no_bias` | D=512, K=512, bias=false, lengths=`[1024, 2048, 3072, 4096]` | `bfloat16` |

### `off.matmul.squeeze_excitation_forward`

| Case | Shape | Dtype |
|---|---|---|
| `se_net_m256_n256_k16` | M=256, N=256, K=16 | `bfloat16` |
| `se_net_m256_n512_k32` | M=256, N=512, K=32 | `bfloat16` |
| `se_net_m128_n1024_k64` | M=128, N=1024, K=64 | `bfloat16` |
| `se_net_m64_n2048_k128` | M=64, N=2048, K=128 | `bfloat16` |
| `se_net_m1024_n1024_k256` | M=1024, N=1024, K=256 | `bfloat16` |

### `off.matmul.bf16xint16`

| Case | Shape | Input dtypes |
|---|---|---|
| `bf16xint16_m1_k4096_n4096` | M=1, K=4096, N=4096 | BF16 x INT16 |
| `bf16xint16_m32_k4096_n11008` | M=32, K=4096, N=11008 | BF16 x INT16 |
| `bf16xint16_m512_k4096_n4096` | M=512, K=4096, N=4096 | BF16 x INT16 |
| `bf16xint16_m4096_k4096_n11008` | M=4096, K=4096, N=11008 | BF16 x INT16 |
| `bf16xint16_m65536_k1024_n1280` | M=65536, K=1024, N=1280 | BF16 x INT16 |

This is `15 x 5 = 75` held-out cells.

## Provenance

The lists above were transcribed from:

- `benchmarks/run_linattn.py` and `scripts/linear_attention_e2e_fla_v3.py`
  for the end-to-end shapes.
- `LINATTN_HEURISTIC_CURRICULUM.json` (SHA-256
  `8e0a5fd9925a6043359db8b43db7b39911c11a06d447acec403e8e06f8e23f3a`)
  for constituent linear-attention case definitions.
- `MATMUL_HEURISTIC_PERF_CURRICULUM.json` (SHA-256
  `a69cca1b8694f2c76adef77731e15709502dc15fbc96475a57cb6aeddca5906b`)
  for the exact 169-cell on-corpus selection and all 75 held-out cases.
