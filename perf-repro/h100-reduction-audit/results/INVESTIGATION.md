# H100 Material AOT-Loss Inspection

Post-run inspection only. Headline measurements were not modified.

Included cells have `G_aot < 0.900`; lower ratios mean a larger checked-in H100 AOT advantage.

| Kernel | Shape | G_aot | Seed config | AOT config | Seed resources | AOT resources |
|---|---|---:|---|---|---|---|
| cross_entropy | `(2048, 32000)` | 0.707 | `b=[1];r=[None];w=32;pid=flat` | `b=[1];r=[None];w=16;pid=flat` | reg=41; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=52; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| rms_norm_dynamic_per_token_quant | `(8192, 5120)` | 0.714 | `b=[8192, 8192, 8192];r=None;w=16;pid=flat` | `b=[8192, 8192, 2048];r=None;w=8;pid=flat` | reg=32; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=32; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| cross_entropy | `(4096, 152064)` | 0.722 | `b=[1];r=[None];w=32;pid=flat` | `b=[1];r=[32768];w=32;pid=flat` | reg=64; spill=372/460 B; stack=336 B; local=0 B; shared=1024 B | reg=59; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| silu_and_mul_per_block_quant | `(8192, 25600, 128)` | 0.768 | `b=[64];r=None;w=4;pid=flat` | `b=[8];r=None;w=4;pid=flat` | reg=96; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=32; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| silu_and_mul_per_block_quant | `(8192, 12288, 128)` | 0.775 | `b=[64];r=None;w=4;pid=flat` | `b=[8];r=None;w=4;pid=flat` | reg=93; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=32; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| cross_entropy | `(8192, 128000)` | 0.794 | `b=[1];r=[None];w=32;pid=flat` | `b=[1];r=[None];w=16;pid=flat` | reg=64; spill=148/188 B; stack=136 B; local=0 B; shared=1024 B | reg=128; spill=176/184 B; stack=112 B; local=0 B; shared=1024 B |
| rms_norm_per_block_quant | `(8192, 5120, 128)` | 0.795 | `b=[8192, 64];r=None;w=16;pid=flat` | `b=[8192, 64];r=None;w=4;pid=flat` | reg=46; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=112; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| cross_entropy | `(2048, 128256)` | 0.796 | `b=[1];r=[None];w=32;pid=flat` | `b=[1];r=[None];w=16;pid=flat` | reg=64; spill=148/188 B; stack=136 B; local=0 B; shared=1024 B | reg=128; spill=176/184 B; stack=112 B; local=0 B; shared=1024 B |
| dynamic_per_token_scaled_fp8_quant | `(8192, 5120)` | 0.817 | `b=[8192, 8192];r=None;w=16;pid=flat` | `b=[8192, 1024];r=None;w=8;pid=flat` | reg=32; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=20; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| rms_norm | `(2048, 48)` | 0.822 | `b=[1];r=[None];w=4;pid=flat` | `b=[4];r=[None];w=8;pid=flat` | reg=16; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=17; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| fused_qk_norm_rope | `(8192, 64, 8)` | 0.872 | `b=[128];r=None;w=4;pid=flat` | `b=[2];r=None;w=1;pid=persistent_blocked` | reg=232; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=56; spill=0/0 B; stack=0 B; local=0 B; shared=0 B |
| layer_norm | `(1024, 36864)` | 0.875 | `b=[1];r=[16384];w=32;pid=flat` | `b=[1];r=[8192];w=32;pid=flat` | reg=32; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=32; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| per_token_group_fp8_quant | `(128, 2048, 128)` | 0.884 | `b=[1];r=None;w=4;pid=flat` | `b=[8];r=None;w=4;pid=flat` | reg=18; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=21; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| rms_norm_per_block_quant | `(128, 2048, 128)` | 0.892 | `b=[2048, 16];r=None;w=8;pid=flat` | `b=[2048, 16];r=None;w=8;pid=flat` | reg=32; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=32; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| cross_entropy | `(2048, 256000)` | 0.893 | `b=[1];r=[16384];w=32;pid=flat` | `b=[1];r=[32768];w=32;pid=flat` | reg=43; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=60; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |
| cross_entropy | `(1024, 256000)` | 0.895 | `b=[1];r=[16384];w=32;pid=flat` | `b=[1];r=[32768];w=32;pid=flat` | reg=43; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B | reg=60; spill=0/0 B; stack=0 B; local=0 B; shared=1024 B |

The adjacent `investigation_codegen/` directory contains the emitted Python/Triton launchers and PTX for every arm in this table. `investigation.json` contains full configs, cache keys, Triton metadata, raw `ptxas -v` output, and raw `cuobjdump --dump-resource-usage` output.
