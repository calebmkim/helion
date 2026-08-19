# H100 Reduction-Heuristic Audit

Generated solely from raw JSONL on 2026-08-19T06:46:10.721427+00:00.

## Environment

- Helion: `2be976894bb434724726cd1064c7eee57210faa6`
- PyTorch: `2.12.0+cu132` (`7661cd9c6b841b62b7f411aa52ec51f05457263b`)
- Triton: `3.7.0`
- CUDA runtime/driver: `13.2` / `595.71.05`
- GPU: `NVIDIA H100 80GB HBM3` (`sm90`)
- Cells: `114`

Ratios above one mean the H100 reduction seed is faster.

## Cohort Summary

| Cohort | Cells | G_default geo [min,max] | G_tc geo [min,max] | G_aot geo [min,max] |
|---|---:|---:|---:|---:|
| general_aot | 24 | 1.135 [0.813,1.788] (24 valid) | 1.054 [0.763,1.752] (24 valid) | 0.936 [0.707,1.103] (24 valid) |
| original | 36 | 6.045 [1.174,50.865] (35 valid) | 1.038 [0.653,1.586] (35 valid) | - [-,-] (0 valid) |
| vllm | 54 | 2.016 [0.870,15.278] (54 valid) | 1.386 [0.979,3.949] (53 valid) | 0.965 [0.714,1.082] (54 valid) |

## Per-Kernel Summary

| Cohort | Kernel | Cells | G_default geo [min,max] | G_tc geo [min,max] | G_aot geo [min,max] |
|---|---|---:|---:|---:|---:|
| general_aot | rms_norm | 6 | 1.050 [0.813,1.355] | 0.997 [0.818,1.147] | 0.963 [0.822,1.047] |
| general_aot | layer_norm | 6 | 1.271 [1.001,1.779] | 1.121 [0.999,1.577] | 0.981 [0.875,1.075] |
| general_aot | softmax | 6 | 1.253 [0.999,1.788] | 1.197 [0.941,1.752] | 1.017 [0.928,1.103] |
| general_aot | cross_entropy | 6 | 0.992 [0.836,1.360] | 0.922 [0.763,1.276] | 0.798 [0.707,0.895] |
| original | kl_div | 6 | 8.515 [2.947,24.027] | 1.109 [0.935,1.545] | - [-,-] |
| original | jsd | 6 | 4.369 [1.903,12.554] | 0.822 [0.786,0.899] | - [-,-] |
| original | fused_linear_jsd | 6 | 1.432 [1.174,2.798] | 1.037 [0.937,1.129] | - [-,-] |
| original | grpo | 6 | 3.142 [1.270,6.907] | 0.971 [0.653,1.095] | - [-,-] |
| original | rms_norm_bwd | 6 | 15.668 [7.281,34.140] | 1.099 [0.992,1.350] | - [-,-] |
| original | layer_norm_bwd | 6 | 21.811 [14.584,50.865] | 1.251 [1.090,1.586] | - [-,-] |
| vllm | dynamic_per_token_scaled_fp8_quant | 9 | 3.522 [2.033,6.595] | 1.071 [1.001,1.297] | 0.979 [0.817,1.027] |
| vllm | per_token_group_fp8_quant | 9 | 1.036 [0.907,1.402] | 1.369 [1.084,1.687] | 0.961 [0.884,1.011] |
| vllm | rms_norm_dynamic_per_token_quant | 9 | 7.737 [4.188,15.278] | 1.106 [0.979,1.554] | 0.976 [0.714,1.082] |
| vllm | rms_norm_per_block_quant | 9 | 2.040 [1.516,2.525] | 1.660 [1.287,2.663] | 0.967 [0.795,1.063] |
| vllm | silu_and_mul_per_block_quant | 9 | 1.096 [0.870,1.266] | 1.592 [1.247,2.167] | 0.932 [0.768,1.027] |
| vllm | fused_qk_norm_rope | 9 | 1.063 [1.004,1.212] | 1.688 [1.176,3.949] | 0.974 [0.872,1.046] |

## Per-Shape Results

| Cohort | Kernel | Shape | seed us | default us | tc us | aot_sm90 us | G_default | G_tc | G_aot | Seed config | AOT config |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| general_aot | rms_norm | `(2048, 48)` | 7.200 | 5.856 | 5.888 | 5.920 | 0.813 | 0.818 | 0.822 | `b=[1];r=[None];w=4;pid=flat` | `b=[4];r=[None];w=8;pid=flat` |
| general_aot | rms_norm | `(2048, 1023)` | 8.896 | 8.960 | 8.896 | 8.448 | 1.007 | 1.000 | 0.950 | `b=[1];r=[None];w=4;pid=flat` | `b=[4];r=[None];w=8;pid=flat` |
| general_aot | rms_norm | `(2048, 4096)` | 15.648 | 16.384 | 17.952 | 16.384 | 1.047 | 1.147 | 1.047 | `b=[1];r=[None];w=8;pid=flat` | `b=[1];r=[None];w=16;pid=flat` |
| general_aot | rms_norm | `(4096, 7168)` | 43.520 | 50.272 | 45.312 | 42.656 | 1.155 | 1.041 | 0.980 | `b=[2];r=[None];w=16;pid=flat` | `b=[1];r=[None];w=16;pid=flat` |
| general_aot | rms_norm | `(16384, 8192)` | 181.296 | 245.648 | 182.240 | 180.720 | 1.355 | 1.005 | 0.997 | `b=[2];r=[None];w=16;pid=flat` | `b=[1];r=[None];w=16;pid=flat` |
| general_aot | rms_norm | `(589824, 256)` | 204.224 | 204.160 | 204.128 | 204.160 | 1.000 | 1.000 | 1.000 | `b=[16];r=[None];w=4;pid=flat` | `b=[16];r=[None];w=4;pid=flat` |
| general_aot | layer_norm | `(4096, 1024)` | 11.040 | 11.808 | 11.232 | 11.872 | 1.070 | 1.017 | 1.075 | `b=[2];r=[None];w=4;pid=flat` | `b=[1];r=[None];w=4;pid=flat` |
| general_aot | layer_norm | `(4096, 3072)` | 21.952 | 21.984 | 34.624 | 21.824 | 1.001 | 1.577 | 0.994 | `b=[2];r=[None];w=8;pid=flat` | `b=[1];r=[None];w=8;pid=flat` |
| general_aot | layer_norm | `(8192, 5120)` | 65.696 | 69.952 | 73.312 | 62.560 | 1.065 | 1.116 | 0.952 | `b=[2];r=[None];w=16;pid=flat` | `b=[1];r=[None];w=8;pid=flat` |
| general_aot | layer_norm | `(4096, 12288)` | 74.528 | 114.144 | 74.432 | 74.592 | 1.532 | 0.999 | 1.001 | `b=[1];r=[None];w=16;pid=flat` | `b=[1];r=[None];w=16;pid=flat` |
| general_aot | layer_norm | `(4096, 16384)` | 93.664 | 166.592 | 95.360 | 93.744 | 1.779 | 1.018 | 1.001 | `b=[1];r=[None];w=16;pid=flat` | `b=[1];r=[None];w=16;pid=flat` |
| general_aot | layer_norm | `(1024, 36864)` | 73.488 | 99.744 | 80.160 | 64.320 | 1.357 | 1.091 | 0.875 | `b=[1];r=[16384];w=32;pid=flat` | `b=[1];r=[8192];w=32;pid=flat` |
| general_aot | softmax | `(4096, 256)` | 7.424 | 8.576 | 7.136 | 8.192 | 1.155 | 0.961 | 1.103 | `b=[2];r=[None];w=4;pid=flat` | `b=[1];r=[None];w=1;pid=persistent_interleaved` |
| general_aot | softmax | `(4096, 384)` | 7.904 | 8.928 | 8.000 | 8.256 | 1.130 | 1.012 | 1.045 | `b=[2];r=[None];w=4;pid=flat` | `b=[1];r=[None];w=1;pid=persistent_interleaved` |
| general_aot | softmax | `(4096, 768)` | 9.760 | 9.920 | 9.184 | 9.056 | 1.016 | 0.941 | 0.928 | `b=[2];r=[None];w=4;pid=flat` | `b=[1];r=[None];w=1;pid=persistent_interleaved` |
| general_aot | softmax | `(4096, 4096)` | 25.792 | 25.760 | 45.184 | 26.336 | 0.999 | 1.752 | 1.021 | `b=[2];r=[None];w=8;pid=flat` | `b=[1];r=[None];w=8;pid=flat` |
| general_aot | softmax | `(4096, 16384)` | 95.280 | 170.368 | 144.576 | 95.280 | 1.788 | 1.517 | 1.000 | `b=[1];r=[None];w=16;pid=flat` | `b=[1];r=[None];w=16;pid=flat` |
| general_aot | softmax | `(2048, 32768)` | 113.696 | 185.920 | 137.088 | 115.424 | 1.635 | 1.206 | 1.015 | `b=[1];r=[None];w=32;pid=flat` | `b=[1];r=[None];w=16;pid=flat` |
| general_aot | cross_entropy | `(2048, 32000)` | 81.504 | 110.832 | 104.000 | 57.648 | 1.360 | 1.276 | 0.707 | `b=[1];r=[None];w=32;pid=flat` | `b=[1];r=[None];w=16;pid=flat` |
| general_aot | cross_entropy | `(1024, 256000)` | 409.888 | 386.816 | 356.704 | 366.688 | 0.944 | 0.870 | 0.895 | `b=[1];r=[16384];w=32;pid=flat` | `b=[1];r=[32768];w=32;pid=flat` |
| general_aot | cross_entropy | `(2048, 128256)` | 394.720 | 390.272 | 359.904 | 314.080 | 0.989 | 0.912 | 0.796 | `b=[1];r=[None];w=32;pid=flat` | `b=[1];r=[None];w=16;pid=flat` |
| general_aot | cross_entropy | `(8192, 128000)` | 1472.288 | 1414.432 | 1367.808 | 1168.864 | 0.961 | 0.929 | 0.794 | `b=[1];r=[None];w=32;pid=flat` | `b=[1];r=[None];w=16;pid=flat` |
| general_aot | cross_entropy | `(4096, 152064)` | 1092.000 | 913.152 | 833.632 | 788.608 | 0.836 | 0.763 | 0.722 | `b=[1];r=[None];w=32;pid=flat` | `b=[1];r=[32768];w=32;pid=flat` |
| general_aot | cross_entropy | `(2048, 256000)` | 796.912 | 743.792 | 681.376 | 711.536 | 0.933 | 0.855 | 0.893 | `b=[1];r=[16384];w=32;pid=flat` | `b=[1];r=[32768];w=32;pid=flat` |
| original | kl_div | `(8192, 32768)` | 448.128 | 1320.848 | 419.072 | - | 2.947 | 0.935 | - | `b=[4096, 1];r=None;w=32;pid=flat` | `-` |
| original | kl_div | `(2048, 50257)` | 218.064 | 1513.552 | 248.944 | - | 6.941 | 1.142 | - | `b=[4096, 1];r=None;w=32;pid=flat` | `-` |
| original | kl_div | `(4096, 114688)` | 696.192 | 4711.680 | 717.280 | - | 6.768 | 1.030 | - | `b=[4096, 1];r=None;w=32;pid=flat` | `-` |
| original | kl_div | `(1024, 128256)` | 219.744 | 5279.888 | 339.488 | - | 24.027 | 1.545 | - | `b=[4096, 1];r=None;w=32;pid=flat` | `-` |
| original | kl_div | `(4096, 151936)` | 964.320 | 6240.736 | 918.144 | - | 6.472 | 0.952 | - | `b=[4096, 1];r=None;w=32;pid=flat` | `-` |
| original | kl_div | `(1024, 250000)` | 409.760 | 7256.288 | 470.880 | - | 17.709 | 1.149 | - | `b=[4096, 1];r=None;w=32;pid=flat` | `-` |
| original | jsd | `(8192, 32768)` | 671.568 | 1277.904 | 528.112 | - | 1.903 | 0.786 | - | `b=[2048, 1];r=None;w=32;pid=flat` | `-` |
| original | jsd | `(2048, 50257)` | 307.328 | 1518.432 | 276.416 | - | 4.941 | 0.899 | - | `b=[2048, 1];r=None;w=32;pid=flat` | `-` |
| original | jsd | `(4096, 114688)` | 1129.760 | 4512.368 | 925.088 | - | 3.994 | 0.819 | - | `b=[2048, 1];r=None;w=32;pid=flat` | `-` |
| original | jsd | `(2048, 128256)` | 683.616 | 4992.192 | 551.584 | - | 7.303 | 0.807 | - | `b=[2048, 1];r=None;w=32;pid=flat` | `-` |
| original | jsd | `(8192, 151936)` | 3119.840 | 6298.752 | 2496.208 | - | 2.019 | 0.800 | - | `b=[2048, 1];r=None;w=32;pid=flat` | `-` |
| original | jsd | `(1024, 250000)` | 674.352 | 8466.080 | 555.168 | - | 12.554 | 0.823 | - | `b=[2048, 1];r=None;w=32;pid=flat` | `-` |
| original | fused_linear_jsd | `(8192, 32000)` | 2619.584 | 3075.232 | 2943.520 | - | 1.174 | 1.124 | - | `b=[1];r=[8192];w=32;pid=flat` | `-` |
| original | fused_linear_jsd | `(4096, 50257)` | 3535.200 | 9890.752 | 3311.328 | - | 2.798 | 0.937 | - | `b=[1];r=[8192];w=32;pid=flat` | `-` |
| original | fused_linear_jsd | `(8192, 128256)` | 12379.264 | 16132.896 | 12156.000 | - | 1.303 | 0.982 | - | `b=[1];r=[8192];w=32;pid=flat` | `-` |
| original | fused_linear_jsd | `(2048, 151936)` | 3840.544 | 5223.200 | 3932.672 | - | 1.360 | 1.024 | - | `b=[1];r=[8192];w=32;pid=flat` | `-` |
| original | fused_linear_jsd | `(2048, 256000)` | 6474.656 | 8142.144 | 6734.560 | - | 1.258 | 1.040 | - | `b=[1];r=[8192];w=32;pid=flat` | `-` |
| original | fused_linear_jsd | `(16384, 32000)` | 5195.392 | 6114.336 | 5863.712 | - | 1.177 | 1.129 | - | `b=[1];r=[8192];w=32;pid=flat` | `-` |
| original | grpo | `(8, 1024, 32000)` | 340.160 | 1392.608 | 323.776 | - | 4.094 | 0.952 | - | `b=[1, 8, 2048];r=None;w=32;pid=flat` | `-` |
| original | grpo | `(8, 2048, 64000)` | 1235.936 | 2777.888 | 1297.120 | - | 2.248 | 1.050 | - | `b=[1, 16, 1024];r=None;w=32;pid=flat` | `-` |
| original | grpo | `(4, 2048, 128256)` | 1228.464 | 4061.296 | 1345.616 | - | 3.306 | 1.095 | - | `b=[1, 16, 1024];r=None;w=32;pid=flat` | `-` |
| original | grpo | `(8, 4096, 128256)` | 4805.648 | 6101.728 | 5223.584 | - | 1.270 | 1.087 | - | `b=[1, 16, 1024];r=None;w=32;pid=flat` | `-` |
| original | grpo | `(16, 1024, 50257)` | 1858.496 | 6708.480 | 1213.984 | - | 3.610 | 0.653 | - | `b=[1, 8, 2048];r=None;w=32;pid=flat` | `-` |
| original | grpo | `(4, 1024, 256000)` | 1191.648 | 8231.104 | 1287.936 | - | 6.907 | 1.081 | - | `b=[1, 8, 2048];r=None;w=32;pid=flat` | `-` |
| original | rms_norm_bwd | `(2048, 4096)` | 35.392 | 695.936 | 40.096 | - | 19.664 | 1.133 | - | `b=[16, 2];r=None;w=8;pid=flat` | `-` |
| original | rms_norm_bwd | `(8192, 4096)` | 95.344 | 1324.160 | 100.832 | - | 13.888 | 1.058 | - | `b=[64, 2];r=None;w=8;pid=flat` | `-` |
| original | rms_norm_bwd | `(4096, 8192)` | 94.336 | 3220.640 | 127.360 | - | 34.140 | 1.350 | - | `b=[32, 1];r=None;w=16;pid=flat` | `-` |
| original | rms_norm_bwd | `(16384, 4096)` | 172.352 | 2396.960 | 172.160 | - | 13.907 | 0.999 | - | `b=[128, 2];r=None;w=8;pid=flat` | `-` |
| original | rms_norm_bwd | `(8192, 2048)` | 56.992 | 414.960 | 56.560 | - | 7.281 | 0.992 | - | `b=[64, 4];r=None;w=8;pid=flat` | `-` |
| original | rms_norm_bwd | `(2048, 11008)` | - | - | 96.704 | - | - | - | - | `b=[16, 1];r=None;w=16;pid=flat` | `-` |
| original | layer_norm_bwd | `(2048, 4096)` | 43.360 | 868.112 | 51.616 | - | 20.021 | 1.190 | - | `b=[16, 2];r=None;w=8;pid=flat` | `-` |
| original | layer_norm_bwd | `(8192, 4096)` | 102.880 | 1577.088 | 122.704 | - | 15.329 | 1.193 | - | `b=[64, 2];r=None;w=8;pid=flat` | `-` |
| original | layer_norm_bwd | `(4096, 8192)` | 102.848 | 2979.600 | 163.104 | - | 28.971 | 1.586 | - | `b=[32, 1];r=None;w=16;pid=flat` | `-` |
| original | layer_norm_bwd | `(16384, 4096)` | 181.120 | 2956.576 | 197.952 | - | 16.324 | 1.093 | - | `b=[128, 2];r=None;w=8;pid=flat` | `-` |
| original | layer_norm_bwd | `(8192, 2048)` | 63.872 | 931.488 | 69.600 | - | 14.584 | 1.090 | - | `b=[64, 8];r=None;w=8;pid=flat` | `-` |
| original | layer_norm_bwd | `(2048, 11008)` | 76.544 | 3893.376 | 109.376 | - | 50.865 | 1.429 | - | `b=[16, 1];r=None;w=16;pid=flat` | `-` |
| vllm | dynamic_per_token_scaled_fp8_quant | `(1, 2048)` | 5.824 | 11.840 | 5.984 | 5.824 | 2.033 | 1.027 | 1.000 | `b=[2048, 2048];r=None;w=8;pid=flat` | `b=[2048, 2048];r=None;w=8;pid=flat` |
| vllm | dynamic_per_token_scaled_fp8_quant | `(1, 4096)` | 6.016 | 19.040 | 6.240 | 6.176 | 3.165 | 1.037 | 1.027 | `b=[4096, 4096];r=None;w=8;pid=flat` | `b=[4096, 4096];r=None;w=8;pid=flat` |
| vllm | dynamic_per_token_scaled_fp8_quant | `(1, 5120)` | 6.368 | 21.696 | 6.528 | 6.368 | 3.407 | 1.025 | 1.000 | `b=[8192, 8192];r=None;w=16;pid=flat` | `b=[8192, 1024];r=None;w=16;pid=flat` |
| vllm | dynamic_per_token_scaled_fp8_quant | `(128, 2048)` | 6.112 | 12.672 | 6.240 | 6.112 | 2.073 | 1.021 | 1.000 | `b=[2048, 2048];r=None;w=8;pid=flat` | `b=[2048, 1024];r=None;w=8;pid=flat` |
| vllm | dynamic_per_token_scaled_fp8_quant | `(128, 4096)` | 6.400 | 19.328 | 6.560 | 6.448 | 3.020 | 1.025 | 1.008 | `b=[4096, 4096];r=None;w=8;pid=flat` | `b=[4096, 2048];r=None;w=16;pid=flat` |
| vllm | dynamic_per_token_scaled_fp8_quant | `(128, 5120)` | 6.656 | 22.528 | 6.944 | 6.656 | 3.385 | 1.043 | 1.000 | `b=[8192, 8192];r=None;w=16;pid=flat` | `b=[8192, 1024];r=None;w=16;pid=flat` |
| vllm | dynamic_per_token_scaled_fp8_quant | `(8192, 2048)` | 24.128 | 129.088 | 28.864 | 23.008 | 5.350 | 1.196 | 0.954 | `b=[2048, 2048];r=None;w=8;pid=flat` | `b=[512, 512];r=None;w=1;pid=flat` |
| vllm | dynamic_per_token_scaled_fp8_quant | `(8192, 4096)` | 38.208 | 251.968 | 49.568 | 39.104 | 6.595 | 1.297 | 1.023 | `b=[4096, 4096];r=None;w=8;pid=flat` | `b=[4096, 4096];r=None;w=8;pid=flat` |
| vllm | dynamic_per_token_scaled_fp8_quant | `(8192, 5120)` | 61.824 | 314.368 | 61.888 | 50.496 | 5.085 | 1.001 | 0.817 | `b=[8192, 8192];r=None;w=16;pid=flat` | `b=[8192, 1024];r=None;w=8;pid=flat` |
| vllm | per_token_group_fp8_quant | `(1, 2048, 128)` | 6.080 | 6.016 | 7.200 | 5.888 | 0.989 | 1.184 | 0.968 | `b=[1];r=None;w=4;pid=flat` | `b=[4];r=None;w=4;pid=flat` |
| vllm | per_token_group_fp8_quant | `(1, 4096, 128)` | 6.016 | 6.400 | 7.872 | 6.080 | 1.064 | 1.309 | 1.011 | `b=[1];r=None;w=4;pid=flat` | `b=[1];r=None;w=1;pid=flat` |
| vllm | per_token_group_fp8_quant | `(1, 5120, 128)` | 6.080 | 6.400 | 7.360 | 6.016 | 1.053 | 1.211 | 0.989 | `b=[1];r=None;w=4;pid=flat` | `b=[1];r=None;w=1;pid=flat` |
| vllm | per_token_group_fp8_quant | `(128, 2048, 128)` | 7.200 | 6.528 | 7.808 | 6.368 | 0.907 | 1.084 | 0.884 | `b=[1];r=None;w=4;pid=flat` | `b=[8];r=None;w=4;pid=flat` |
| vllm | per_token_group_fp8_quant | `(128, 4096, 128)` | 7.328 | 6.784 | 9.248 | 6.720 | 0.926 | 1.262 | 0.917 | `b=[2];r=None;w=4;pid=flat` | `b=[8];r=None;w=2;pid=flat` |
| vllm | per_token_group_fp8_quant | `(128, 5120, 128)` | 7.168 | 7.424 | 10.560 | 6.880 | 1.036 | 1.473 | 0.960 | `b=[4];r=None;w=4;pid=flat` | `b=[8];r=None;w=4;pid=flat` |
| vllm | per_token_group_fp8_quant | `(8192, 2048, 128)` | 24.032 | 24.416 | 39.424 | 23.744 | 1.016 | 1.640 | 0.988 | `b=[16];r=None;w=4;pid=flat` | `b=[8];r=None;w=2;pid=flat` |
| vllm | per_token_group_fp8_quant | `(8192, 4096, 128)` | 41.440 | 41.456 | 69.920 | 40.704 | 1.000 | 1.687 | 0.982 | `b=[32];r=None;w=4;pid=flat` | `b=[8];r=None;w=2;pid=flat` |
| vllm | per_token_group_fp8_quant | `(8192, 5120, 128)` | 51.040 | 71.552 | 82.176 | 48.896 | 1.402 | 1.610 | 0.958 | `b=[64];r=None;w=4;pid=flat` | `b=[8];r=None;w=2;pid=flat` |
| vllm | rms_norm_dynamic_per_token_quant | `(1, 2048)` | 6.432 | 27.840 | 6.368 | 6.480 | 4.328 | 0.990 | 1.007 | `b=[2048, 2048, 2048];r=None;w=8;pid=flat` | `b=[2048, 2048, 2048];r=None;w=8;pid=flat` |
| vllm | rms_norm_dynamic_per_token_quant | `(1, 4096)` | 6.816 | 46.272 | 6.784 | 6.784 | 6.789 | 0.995 | 0.995 | `b=[4096, 4096, 4096];r=None;w=8;pid=flat` | `b=[4096, 4096, 4096];r=None;w=16;pid=persistent_interleaved` |
| vllm | rms_norm_dynamic_per_token_quant | `(1, 5120)` | 7.168 | 59.616 | 7.168 | 7.328 | 8.317 | 1.000 | 1.022 | `b=[8192, 8192, 8192];r=None;w=16;pid=flat` | `b=[8192, 8192, 2048];r=None;w=32;pid=flat` |
| vllm | rms_norm_dynamic_per_token_quant | `(128, 2048)` | 6.656 | 27.872 | 7.280 | 7.200 | 4.188 | 1.094 | 1.082 | `b=[2048, 2048, 2048];r=None;w=8;pid=flat` | `b=[2048, 2048, 2048];r=None;w=8;pid=flat` |
| vllm | rms_norm_dynamic_per_token_quant | `(128, 4096)` | 7.136 | 47.456 | 7.680 | 7.584 | 6.650 | 1.076 | 1.063 | `b=[4096, 4096, 4096];r=None;w=8;pid=flat` | `b=[4096, 4096, 4096];r=None;w=16;pid=flat` |
| vllm | rms_norm_dynamic_per_token_quant | `(128, 5120)` | 7.680 | 58.688 | 7.520 | 7.488 | 7.642 | 0.979 | 0.975 | `b=[8192, 8192, 8192];r=None;w=16;pid=flat` | `b=[8192, 8192, 2048];r=None;w=8;pid=flat` |
| vllm | rms_norm_dynamic_per_token_quant | `(8192, 2048)` | 30.304 | 372.544 | 47.104 | 27.936 | 12.294 | 1.554 | 0.922 | `b=[2048, 2048, 2048];r=None;w=8;pid=flat` | `b=[2048, 2048, 1024];r=None;w=4;pid=flat` |
| vllm | rms_norm_dynamic_per_token_quant | `(8192, 4096)` | 49.536 | 756.832 | 59.680 | 52.640 | 15.278 | 1.205 | 1.063 | `b=[4096, 4096, 4096];r=None;w=8;pid=flat` | `b=[4096, 4096, 2048];r=None;w=8;pid=persistent_interleaved` |
| vllm | rms_norm_dynamic_per_token_quant | `(8192, 5120)` | 91.008 | 926.048 | 105.952 | 65.024 | 10.175 | 1.164 | 0.714 | `b=[8192, 8192, 8192];r=None;w=16;pid=flat` | `b=[8192, 8192, 2048];r=None;w=8;pid=flat` |
| vllm | rms_norm_per_block_quant | `(1, 2048, 128)` | 6.880 | 11.392 | 10.144 | 6.848 | 1.656 | 1.474 | 0.995 | `b=[2048, 16];r=None;w=8;pid=flat` | `b=[2048, 16];r=None;w=8;pid=flat` |
| vllm | rms_norm_per_block_quant | `(1, 4096, 128)` | 7.008 | 16.448 | 9.888 | 6.976 | 2.347 | 1.411 | 0.995 | `b=[4096, 32];r=None;w=8;pid=flat` | `b=[4096, 32];r=None;w=16;pid=persistent_interleaved` |
| vllm | rms_norm_per_block_quant | `(1, 5120, 128)` | 7.808 | 19.712 | 10.048 | 7.968 | 2.525 | 1.287 | 1.020 | `b=[8192, 64];r=None;w=16;pid=flat` | `b=[8192, 64];r=None;w=32;pid=flat` |
| vllm | rms_norm_per_block_quant | `(128, 2048, 128)` | 8.000 | 12.128 | 11.232 | 7.136 | 1.516 | 1.404 | 0.892 | `b=[2048, 16];r=None;w=8;pid=flat` | `b=[2048, 16];r=None;w=8;pid=flat` |
| vllm | rms_norm_per_block_quant | `(128, 4096, 128)` | 8.160 | 17.280 | 11.904 | 8.672 | 2.118 | 1.459 | 1.063 | `b=[4096, 32];r=None;w=8;pid=flat` | `b=[4096, 32];r=None;w=16;pid=flat` |
| vllm | rms_norm_per_block_quant | `(128, 5120, 128)` | 9.440 | 20.896 | 14.560 | 9.440 | 2.214 | 1.542 | 1.000 | `b=[8192, 64];r=None;w=16;pid=flat` | `b=[8192, 64];r=None;w=32;pid=flat` |
| vllm | rms_norm_per_block_quant | `(8192, 2048, 128)` | 46.176 | 94.016 | 95.232 | 46.080 | 2.036 | 2.062 | 0.998 | `b=[2048, 16];r=None;w=8;pid=flat` | `b=[2048, 16];r=None;w=8;pid=flat` |
| vllm | rms_norm_per_block_quant | `(8192, 4096, 128)` | 86.864 | 189.536 | 179.264 | 84.752 | 2.182 | 2.064 | 0.976 | `b=[4096, 32];r=None;w=8;pid=flat` | `b=[8192, 32];r=None;w=16;pid=flat` |
| vllm | rms_norm_per_block_quant | `(8192, 5120, 128)` | 132.512 | 261.760 | 352.816 | 105.376 | 1.975 | 2.663 | 0.795 | `b=[8192, 64];r=None;w=16;pid=flat` | `b=[8192, 64];r=None;w=4;pid=flat` |
| vllm | silu_and_mul_per_block_quant | `(1, 6144, 128)` | 6.016 | 7.168 | 8.064 | 6.048 | 1.191 | 1.340 | 1.005 | `b=[2];r=None;w=4;pid=flat` | `b=[4];r=None;w=4;pid=flat` |
| vllm | silu_and_mul_per_block_quant | `(1, 12288, 128)` | 6.368 | 7.328 | 8.032 | 6.304 | 1.151 | 1.261 | 0.990 | `b=[4];r=None;w=4;pid=flat` | `b=[4];r=None;w=4;pid=flat` |
| vllm | silu_and_mul_per_block_quant | `(1, 25600, 128)` | 6.208 | 7.488 | 7.744 | 6.176 | 1.206 | 1.247 | 0.995 | `b=[8];r=None;w=4;pid=flat` | `b=[4];r=None;w=4;pid=flat` |
| vllm | silu_and_mul_per_block_quant | `(128, 6144, 128)` | 7.200 | 7.680 | 10.720 | 7.392 | 1.067 | 1.489 | 1.027 | `b=[4];r=None;w=4;pid=flat` | `b=[8];r=None;w=4;pid=flat` |
| vllm | silu_and_mul_per_block_quant | `(128, 12288, 128)` | 8.352 | 9.536 | 12.768 | 8.288 | 1.142 | 1.529 | 0.992 | `b=[8];r=None;w=4;pid=flat` | `b=[8];r=None;w=4;pid=flat` |
| vllm | silu_and_mul_per_block_quant | `(128, 25600, 128)` | 11.488 | 12.608 | 21.696 | 10.816 | 1.097 | 1.889 | 0.942 | `b=[16];r=None;w=4;pid=flat` | `b=[8];r=None;w=4;pid=flat` |
| vllm | silu_and_mul_per_block_quant | `(8192, 6144, 128)` | 93.856 | 118.784 | 203.408 | 88.352 | 1.266 | 2.167 | 0.941 | `b=[64];r=None;w=4;pid=flat` | `b=[16];r=None;w=8;pid=flat` |
| vllm | silu_and_mul_per_block_quant | `(8192, 12288, 128)` | 221.632 | 192.736 | 402.400 | 171.776 | 0.870 | 1.816 | 0.775 | `b=[64];r=None;w=4;pid=flat` | `b=[8];r=None;w=4;pid=flat` |
| vllm | silu_and_mul_per_block_quant | `(8192, 25600, 128)` | 451.616 | 424.608 | 833.216 | 347.040 | 0.940 | 1.845 | 0.768 | `b=[64];r=None;w=4;pid=flat` | `b=[8];r=None;w=4;pid=flat` |
| vllm | fused_qk_norm_rope | `(1, 16, 8)` | 6.560 | 7.296 | 7.712 | 6.656 | 1.112 | 1.176 | 1.015 | `b=[1];r=None;w=4;pid=flat` | `b=[1];r=None;w=1;pid=persistent_blocked` |
| vllm | fused_qk_norm_rope | `(1, 32, 8)` | 6.528 | 7.200 | 9.216 | 6.464 | 1.103 | 1.412 | 0.990 | `b=[1];r=None;w=4;pid=flat` | `b=[1];r=None;w=1;pid=flat` |
| vllm | fused_qk_norm_rope | `(1, 64, 8)` | 6.848 | 6.912 | 8.192 | 6.400 | 1.009 | 1.196 | 0.935 | `b=[2];r=None;w=4;pid=flat` | `b=[2];r=None;w=2;pid=flat` |
| vllm | fused_qk_norm_rope | `(128, 16, 8)` | 7.520 | 7.616 | 10.368 | 7.200 | 1.013 | 1.379 | 0.957 | `b=[2];r=None;w=4;pid=flat` | `b=[4];r=None;w=1;pid=flat` |
| vllm | fused_qk_norm_rope | `(128, 32, 8)` | 7.616 | 7.936 | 10.464 | 7.360 | 1.042 | 1.374 | 0.966 | `b=[4];r=None;w=4;pid=flat` | `b=[4];r=None;w=2;pid=flat` |
| vllm | fused_qk_norm_rope | `(128, 64, 8)` | 8.576 | 9.280 | 14.400 | 8.640 | 1.082 | 1.679 | 1.007 | `b=[8];r=None;w=4;pid=flat` | `b=[4];r=None;w=2;pid=flat` |
| vllm | fused_qk_norm_rope | `(8192, 16, 8)` | 45.600 | 45.760 | 180.064 | 47.680 | 1.004 | 3.949 | 1.046 | `b=[32];r=None;w=4;pid=flat` | `b=[2];r=None;w=1;pid=persistent_blocked` |
| vllm | fused_qk_norm_rope | `(8192, 32, 8)` | 74.992 | 90.912 | - | 74.208 | 1.212 | - | 0.990 | `b=[64];r=None;w=4;pid=flat` | `b=[2];r=None;w=1;pid=persistent_blocked` |
| vllm | fused_qk_norm_rope | `(8192, 64, 8)` | 144.800 | 145.408 | 383.552 | 126.208 | 1.004 | 2.649 | 0.872 | `b=[128];r=None;w=4;pid=flat` | `b=[2];r=None;w=1;pid=persistent_blocked` |

## Failures And Noise

### Cell Errors (0)

None.

### Heuristic No Fires (0)

None.

### Torch Compile Graph Breaks (0)

None.

### Arm Failures (0)

None.

### Accuracy Failures (3)

- `{"accuracy": {"outputs": {"grad_weight": {"actual_dtype": "bfloat16", "atol": 0.03, "exact_fraction": 0.9997274875640869, "exact_required": false, "expected_dtype": "bfloat16", "max_abs": 0.03125, "max_rel": 0.007042253389954567, "pass": true, "rtol": 0.03, "shape": [11008]}, "grad_x": {"actual_dtype": "bfloat16", "atol": 0.03, "exact_fraction": 0.42630434036254883, "exact_required": false, "expected_dtype": "bfloat16", "max_abs": 0.0625, "max_rel": 3014657.0, "pass": false, "rtol": 0.03, "shape": [2048, 11008]}}, "pass": false}, "arm": "default", "kernel": "rms_norm_bwd", "shape": [2048, 11008], "source": "rms_norm_bwd.jsonl:6"}`
- `{"accuracy": {"outputs": {"grad_weight": {"actual_dtype": "bfloat16", "atol": 0.03, "exact_fraction": 0.9997274875640869, "exact_required": false, "expected_dtype": "bfloat16", "max_abs": 0.03125, "max_rel": 0.007042253389954567, "pass": true, "rtol": 0.03, "shape": [11008]}, "grad_x": {"actual_dtype": "bfloat16", "atol": 0.03, "exact_fraction": 0.42630425095558167, "exact_required": false, "expected_dtype": "bfloat16", "max_abs": 0.0625, "max_rel": 3014657.0, "pass": false, "rtol": 0.03, "shape": [2048, 11008]}}, "pass": false}, "arm": "seed", "kernel": "rms_norm_bwd", "shape": [2048, 11008], "source": "rms_norm_bwd.jsonl:6"}`
- `{"accuracy": {"outputs": {"qkv": {"actual_dtype": "bfloat16", "atol": 0.02, "exact_fraction": 0.6045575141906738, "exact_required": false, "expected_dtype": "bfloat16", "max_abs": 0.03125, "max_rel": 17333985280.0, "pass": false, "rtol": 0.02, "shape": [8192, 6144]}}, "pass": false}, "arm": "torch_compile", "kernel": "fused_qk_norm_rope", "shape": [8192, 32, 8], "source": "fused_qk_norm_rope.jsonl:8"}`

### High Spread (42)

- `{"arm": "seed", "kernel": "rms_norm", "relative_spread": 0.05333335230471944, "shape": [2048, 48], "source": "rms_norm.jsonl:1"}`
- `{"arm": "torch_compile", "kernel": "rms_norm", "relative_spread": 0.05978258806439286, "shape": [2048, 48], "source": "rms_norm.jsonl:1"}`
- `{"arm": "seed", "kernel": "softmax", "relative_spread": 0.05172415415269939, "shape": [4096, 256], "source": "softmax.jsonl:1"}`
- `{"arm": "aot_sm90", "kernel": "dynamic_per_token_scaled_fp8_quant", "relative_spread": 0.05494507251770916, "shape": [1, 2048], "source": "dynamic_per_token_scaled_fp8_quant.jsonl:1"}`
- `{"arm": "aot_sm90", "kernel": "dynamic_per_token_scaled_fp8_quant", "relative_spread": 0.052884598903387245, "shape": [128, 5120], "source": "dynamic_per_token_scaled_fp8_quant.jsonl:6"}`
- `{"arm": "aot_sm90", "kernel": "per_token_group_fp8_quant", "relative_spread": 0.05978258806439286, "shape": [1, 2048, 128], "source": "per_token_group_fp8_quant.jsonl:1"}`
- `{"arm": "default", "kernel": "per_token_group_fp8_quant", "relative_spread": 0.06382980699671545, "shape": [1, 2048, 128], "source": "per_token_group_fp8_quant.jsonl:1"}`
- `{"arm": "aot_sm90", "kernel": "per_token_group_fp8_quant", "relative_spread": 0.08040196338240913, "shape": [128, 2048, 128], "source": "per_token_group_fp8_quant.jsonl:4"}`
- `{"arm": "default", "kernel": "per_token_group_fp8_quant", "relative_spread": 0.058823550392035025, "shape": [128, 2048, 128], "source": "per_token_group_fp8_quant.jsonl:4"}`
- `{"arm": "seed", "kernel": "per_token_group_fp8_quant", "relative_spread": 0.06666665804330935, "shape": [128, 2048, 128], "source": "per_token_group_fp8_quant.jsonl:4"}`
- `{"arm": "aot_sm90", "kernel": "per_token_group_fp8_quant", "relative_spread": 0.06190474870574572, "shape": [128, 4096, 128], "source": "per_token_group_fp8_quant.jsonl:5"}`
- `{"arm": "seed", "kernel": "per_token_group_fp8_quant", "relative_spread": 0.05676854701984398, "shape": [128, 4096, 128], "source": "per_token_group_fp8_quant.jsonl:5"}`
- `{"arm": "default", "kernel": "per_token_group_fp8_quant", "relative_spread": 0.05172415415269939, "shape": [128, 5120, 128], "source": "per_token_group_fp8_quant.jsonl:6"}`
- `{"arm": "seed", "kernel": "per_token_group_fp8_quant", "relative_spread": 0.05803576561880102, "shape": [128, 5120, 128], "source": "per_token_group_fp8_quant.jsonl:6"}`
- `{"arm": "aot_sm90", "kernel": "rms_norm_per_block_quant", "relative_spread": 0.09633027400455221, "shape": [1, 4096, 128], "source": "rms_norm_per_block_quant.jsonl:2"}`
- `{"arm": "seed", "kernel": "rms_norm_per_block_quant", "relative_spread": 0.0958904127793727, "shape": [1, 4096, 128], "source": "rms_norm_per_block_quant.jsonl:2"}`
- `{"arm": "torch_compile", "kernel": "rms_norm_per_block_quant", "relative_spread": 0.0711973890567147, "shape": [1, 4096, 128], "source": "rms_norm_per_block_quant.jsonl:2"}`
- `{"arm": "seed", "kernel": "rms_norm_per_block_quant", "relative_spread": 0.0679999906867747, "shape": [128, 2048, 128], "source": "rms_norm_per_block_quant.jsonl:4"}`
- `{"arm": "aot_sm90", "kernel": "rms_norm_per_block_quant", "relative_spread": 0.10332106927259432, "shape": [128, 4096, 128], "source": "rms_norm_per_block_quant.jsonl:5"}`
- `{"arm": "torch_compile", "kernel": "rms_norm_per_block_quant", "relative_spread": 0.06720426342940283, "shape": [128, 4096, 128], "source": "rms_norm_per_block_quant.jsonl:5"}`
- `{"arm": "aot_sm90", "kernel": "silu_and_mul_per_block_quant", "relative_spread": 0.08121830110896673, "shape": [1, 12288, 128], "source": "silu_and_mul_per_block_quant.jsonl:2"}`
- `{"arm": "default", "kernel": "silu_and_mul_per_block_quant", "relative_spread": 0.06986902005241231, "shape": [1, 12288, 128], "source": "silu_and_mul_per_block_quant.jsonl:2"}`
- `{"arm": "seed", "kernel": "silu_and_mul_per_block_quant", "relative_spread": 0.05025127281725661, "shape": [1, 12288, 128], "source": "silu_and_mul_per_block_quant.jsonl:2"}`
- `{"arm": "torch_compile", "kernel": "silu_and_mul_per_block_quant", "relative_spread": 0.059760949707883336, "shape": [1, 12288, 128], "source": "silu_and_mul_per_block_quant.jsonl:2"}`
- `{"arm": "seed", "kernel": "silu_and_mul_per_block_quant", "relative_spread": 0.056701014688583753, "shape": [1, 25600, 128], "source": "silu_and_mul_per_block_quant.jsonl:3"}`
- `{"arm": "aot_sm90", "kernel": "fused_qk_norm_rope", "relative_spread": 0.0769231038312045, "shape": [1, 16, 8], "source": "fused_qk_norm_rope.jsonl:1"}`
- `{"arm": "default", "kernel": "fused_qk_norm_rope", "relative_spread": 0.05701753014304724, "shape": [1, 16, 8], "source": "fused_qk_norm_rope.jsonl:1"}`
- `{"arm": "seed", "kernel": "fused_qk_norm_rope", "relative_spread": 0.0634146189105471, "shape": [1, 16, 8], "source": "fused_qk_norm_rope.jsonl:1"}`
- `{"arm": "aot_sm90", "kernel": "fused_qk_norm_rope", "relative_spread": 0.09900986461387645, "shape": [1, 32, 8], "source": "fused_qk_norm_rope.jsonl:2"}`
- `{"arm": "default", "kernel": "fused_qk_norm_rope", "relative_spread": 0.08444444214488249, "shape": [1, 32, 8], "source": "fused_qk_norm_rope.jsonl:2"}`
- `{"arm": "seed", "kernel": "fused_qk_norm_rope", "relative_spread": 0.09313732378718166, "shape": [1, 32, 8], "source": "fused_qk_norm_rope.jsonl:2"}`
- `{"arm": "torch_compile", "kernel": "fused_qk_norm_rope", "relative_spread": 0.05902771988170506, "shape": [1, 32, 8], "source": "fused_qk_norm_rope.jsonl:2"}`
- `{"arm": "aot_sm90", "kernel": "fused_qk_norm_rope", "relative_spread": 0.06250004092726262, "shape": [1, 64, 8], "source": "fused_qk_norm_rope.jsonl:3"}`
- `{"arm": "default", "kernel": "fused_qk_norm_rope", "relative_spread": 0.06018517177356923, "shape": [1, 64, 8], "source": "fused_qk_norm_rope.jsonl:3"}`
- `{"arm": "seed", "kernel": "fused_qk_norm_rope", "relative_spread": 0.060747650205685205, "shape": [1, 64, 8], "source": "fused_qk_norm_rope.jsonl:3"}`
- `{"arm": "aot_sm90", "kernel": "fused_qk_norm_rope", "relative_spread": 0.057777765992522774, "shape": [128, 16, 8], "source": "fused_qk_norm_rope.jsonl:4"}`
- `{"arm": "default", "kernel": "fused_qk_norm_rope", "relative_spread": 0.054621838206543656, "shape": [128, 16, 8], "source": "fused_qk_norm_rope.jsonl:4"}`
- `{"arm": "seed", "kernel": "fused_qk_norm_rope", "relative_spread": 0.0553191368150638, "shape": [128, 16, 8], "source": "fused_qk_norm_rope.jsonl:4"}`
- `{"arm": "aot_sm90", "kernel": "fused_qk_norm_rope", "relative_spread": 0.06956523939798095, "shape": [128, 32, 8], "source": "fused_qk_norm_rope.jsonl:5"}`
- `{"arm": "default", "kernel": "fused_qk_norm_rope", "relative_spread": 0.08467741982804096, "shape": [128, 32, 8], "source": "fused_qk_norm_rope.jsonl:5"}`
- `{"arm": "default", "kernel": "fused_qk_norm_rope", "relative_spread": 0.09310345588923041, "shape": [128, 64, 8], "source": "fused_qk_norm_rope.jsonl:6"}`
- `{"arm": "seed", "kernel": "fused_qk_norm_rope", "relative_spread": 0.06529846086349327, "shape": [128, 64, 8], "source": "fused_qk_norm_rope.jsonl:6"}`
