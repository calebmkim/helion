"""Real-world shape curriculum for the reduction-heuristic TRANSFER test.

Every shape is a real model config (hidden/vocab/expert/context/token counts from
Llama/GPT/Qwen/Gemma/Mixtral/DeepSeek/CLIP). These kernels are NOT in the merged
heuristic's training curriculum, so all of these are genuine transfer measurements.

Convention:
  * norms / quant : (M=tokens, N=hidden)
  * softmax       : (M=batch*heads*query, N=key_len)
  * CE / FLCE     : (M=tokens, N=vocab)
  * GRPO          : (B, L, V)
"""
from __future__ import annotations

# Hidden-dim envelope (real model hidden sizes) shared by the norm + quant kernels.
_NORM = [
    (8192, 768), (8192, 1024), (8192, 2048), (8192, 4096),   # GPT2/BERT, Llama-7B
    (4096, 5120), (4096, 7168), (4096, 8192),                 # Llama-13B/Yi-34B/Llama-70B
    (2048, 12288), (2048, 16384),                             # GPT-3 / MLP
    (16384, 4096), (8192, 3072), (32768, 2048),               # M-variation
]

SHAPES = {
    # ---- Bucket 1: reduction + pointwise fused ---------------------------------
    "fused_add_rmsnorm": _NORM,
    "fused_add_layernorm": [
        (8192, 768), (8192, 1024), (8192, 2048), (4096, 2560), (8192, 3072),
        (4096, 4096), (4096, 5120), (2048, 8192), (2048, 12288), (2048, 16384),
        (16384, 1024), (32768, 1024),
    ],
    "gated_rmsnorm": _NORM,
    "scaled_masked_softmax": [
        (262144, 128), (131072, 256), (131072, 512),          # short-ctx (tiny-N, huge-M)
        (16384, 1024), (16384, 2048), (8192, 4096),           # mid context
        (4096, 8192), (2048, 16384), (2048, 32768),           # long context
        (1024, 65536), (512, 131072),                         # very long context
    ],
    "cross_entropy_ls_zloss": [
        (8192, 32000), (8192, 32064), (8192, 50257),          # Llama2, Phi3, GPT2
        (4096, 50304), (8192, 49152), (4096, 128256),         # NeoX, -, Llama3
        (8192, 128256), (2048, 151936), (2048, 256000),       # Llama3 big-M, Qwen2, Gemma
        (16384, 32000), (4096, 102400),                       # Llama2 big-M, DeepSeek-LLM/V2
    ],

    # ---- Bucket 2: new kernels --------------------------------------------------
    "dynamic_quant": [
        (16384, 4096), (8192, 8192), (4096, 16384), (8192, 2048),
        (16384, 1024), (2048, 16384), (8192, 768), (4096, 5120),
    ],
    "fused_linear_jsd": [   # (chunk_tokens, vocab) — kernel sees precomputed logits
        (8192, 32000), (4096, 50257), (8192, 128256), (4096, 128256),
        (2048, 151936), (2048, 256000), (16384, 32000),
    ],
    "grpo": [               # (B, L, V)
        (8, 1024, 32000), (8, 2048, 64000), (4, 2048, 128256), (8, 4096, 128256),
        (16, 1024, 50257), (4, 1024, 256000), (8, 2048, 151936),
    ],
}

# A small correctness-only smoke set per kernel family (cheap shapes for the verifier).
SMOKE = {
    "fused_add_rmsnorm": [(4096, 4096), (2048, 8192), (8192, 768)],
    "fused_add_layernorm": [(4096, 4096), (2048, 8192), (8192, 1024)],
    "gated_rmsnorm": [(4096, 4096), (2048, 8192), (8192, 768)],
    "scaled_masked_softmax": [(8192, 1024), (4096, 8192), (1024, 65536)],
    "cross_entropy_ls_zloss": [(4096, 32000), (2048, 128256)],
    "dynamic_quant": [(8192, 4096), (4096, 16384), (8192, 768)],
    "fused_linear_jsd": [(4096, 32000), (2048, 128256)],
    "grpo": [(4, 512, 32000), (2, 1024, 50257)],
}
