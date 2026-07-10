"""Shape curriculum for the H100 matmul-seed hill-climb (DRAFT).

Three kernels (this run): `matmul` (clean 2D GEMM, examples/matmul.py),
`mamba2_chunk_state` (fused batched GEMM-with-decay, examples/mamba2_chunk_state.py),
and `fp8_gemm` (fp8 e4m3 GEMM, examples/fp8_gemm.py — the 8-bit band; the autotune
spawn-worker blocker is FIXED, commit abc10a00 on reduction-4pr-stack).

Splits: TRAIN / VAL / TEST, with an ACCESS FIREWALL for honesty:
- TRAIN — the main agent hill-climbs on these (reads, benches, tunes freely).
- VAL  — an **adversarial referee subagent is the ONLY one allowed to run these**;
  the main agent must NEVER read or bench a VAL shape. At milestones the referee
  evaluates the current heuristic on VAL and returns a **mechanistic diagnosis**
  (the mis-served property class + WHY + a fix direction — e.g. "wide-N tile
  starved by the occupancy cap"), but NEVER the exact shape/ceiling-config, so the
  main agent fixes the mechanism generally and cannot memorize VAL. A genuinely
  out-of-sample regime is closed by adding a real named TRAIN shape, not a fence.
- TEST — the final firewall: read **once**, at the very end, by the referee.
Each split spans every regime + width so coverage is proven, not assumed. Shapes
are REALISTIC (real-model dims, cited) with a few plausible-but-unusual STRESS
shapes; nothing diabolical.

Width is the dtype key (operand bit-width, not format): bf16==fp16 = one 16-bit
band; fp8 = 8-bit; fp32 = 32-bit. int4 (4-bit, mixed-width int4×fp16) is DEFERRED
(needs the int4_gemm kernel). A few fp16 shapes duplicate bf16 ones on purpose, to
*empirically confirm* the 16-bit merge (fp16 and bf16 should want the same config).

matmul tuple = (M, K, N, dtype)   for  matmul(x[M,K], y[K,N]) -> [M,N]
  the dot's MatmulFact is (m=M, n=N, k=K); width = itemsize of the inputs.

mamba2_chunk_state tuple = (batch, seqlen, nheads, chunk, headdim, dstate, dtype)
  ngroups=1 (B shared across heads, as in the experiment). nchunks = seqlen/chunk.
  the inner dot is  [headdim, chunk] @ [chunk, dstate] -> [headdim, dstate],
  batched over (batch * nchunks * nheads).  So the dot's (M, N, K) = (headdim,
  dstate, chunk) — all SMALL (the regime B200 misses) — and batch/seqlen/nheads
  set the grid (occupancy/pid), not the per-dot tile.
"""

from __future__ import annotations

import torch

DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32,
         "fp8": torch.float8_e4m3fn}
WIDTH_BITS = {"bf16": 16, "fp16": 16, "fp32": 32, "fp8": 8}  # the heuristic key axis


# ----------------------------------------------------------------------------
# matmul  (M, K, N, dtype)
# ----------------------------------------------------------------------------
MATMUL_TRAIN = [
    # --- compute-bound square / training (16-bit) ---
    (2048, 2048, 2048, "bf16"),     # small training cube
    (4096, 4096, 4096, "bf16"),     # 7B hidden, attn QKV/O proj
    (8192, 8192, 8192, "bf16"),     # large training cube
    (8192, 4096, 4096, "bf16"),     # 8k tokens x 7B attn proj
    # --- FFN / MLP rectangular (16-bit) ---
    (4096, 4096, 11008, "bf16"),    # Llama-2-7B gate/up   (hidden->ffn)
    (4096, 11008, 4096, "bf16"),    # Llama-2-7B down      (ffn->hidden)
    (4096, 4096, 14336, "bf16"),    # Llama-3-8B up
    (4096, 14336, 4096, "bf16"),    # Llama-3-8B down
    (8192, 8192, 28672, "bf16"),    # large / Mixtral-style FFN up
    # --- attention QKV / O proj (16-bit) ---
    (4096, 4096, 12288, "bf16"),    # fused QKV (3x hidden), 7B
    (8192, 5120, 5120, "bf16"),     # 13B hidden=5120 attn proj
    # --- vocab / LM head (huge N) ---
    (4096, 4096, 128256, "bf16"),   # Llama-3 vocab
    (2048, 4096, 32000, "bf16"),    # Llama-2 vocab
    # --- decode / GEMV (tiny M, memory-bound) ---
    (1, 4096, 4096, "bf16"),        # single-token decode proj
    (8, 4096, 14336, "bf16"),       # batch-8 decode FFN
    (32, 4096, 4096, "bf16"),       # batch-32 decode proj
    # --- small / prefill-chunk ---
    (128, 4096, 4096, "bf16"),
    (256, 4096, 4096, "bf16"),
    # --- width: fp32 (32-bit band) ---
    (2048, 2048, 2048, "fp32"),
    (4096, 4096, 4096, "fp32"),
    (8, 4096, 4096, "fp32"),        # fp32 decode
    # --- width: fp16 (must match bf16 config -> validates the 16-bit merge) ---
    (4096, 4096, 4096, "fp16"),
    (4096, 4096, 14336, "fp16"),
    # --- stress (plausible-but-unusual; nothing diabolical) ---
    (16384, 8192, 512, "bf16"),     # very tall, small N
    (512, 4096, 16384, "bf16"),     # small M, very wide N
    (3072, 3072, 3072, "bf16"),     # non-pow2 cube
]

# VAL — adversarial-referee-only (the main agent must NOT read/run these)
MATMUL_VAL = [
    (4096, 4096, 8192, "bf16"),     # mid FFN-ish
    (6144, 4096, 4096, "bf16"),     # 6k tokens
    (4, 4096, 4096, "bf16"),        # decode batch-4
    (4096, 8192, 8192, "bf16"),     # 70B-ish hidden=8192
    (2048, 11008, 4096, "bf16"),    # FFN down, 2k tokens
    (256, 8192, 8192, "bf16"),      # small M, big hidden
    (4096, 4096, 4096, "fp16"),     # width interpolation
]

MATMUL_TEST = [  # HELD OUT — never climbed; the generalization firewall
    # realistic, one per regime, distinct from TRAIN:
    (4096, 4096, 16384, "bf16"),    # FFN up (8B variants)
    (8192, 14336, 4096, "bf16"),    # FFN down, 8k tokens (Llama-3)
    (16, 8192, 8192, "bf16"),       # decode batch-16, 70B
    (2, 4096, 4096, "bf16"),        # decode batch-2
    (4096, 4096, 152064, "bf16"),   # Qwen2 vocab
    (8192, 5120, 13824, "bf16"),    # 13B FFN
    (512, 4096, 4096, "bf16"),      # small / prefill
    (4096, 8192, 1024, "bf16"),     # narrow-N projection
    # width held-out:
    (1024, 1024, 1024, "fp32"),     # fp32 small
    (4096, 4096, 11008, "fp16"),    # fp16 FFN (== bf16 config?)
    # stress held-out:
    (32768, 4096, 256, "bf16"),     # extreme tall-skinny
    (256, 16384, 256, "bf16"),      # deep-K, small M·N
    (4000, 4096, 4096, "bf16"),     # non-tile-multiple M (tail handling)
]


# ----------------------------------------------------------------------------
# mamba2_chunk_state  (batch, seqlen, nheads, chunk, headdim, dstate, dtype)
# ----------------------------------------------------------------------------
MAMBA_TRAIN = [
    (2, 4096, 64, 256, 64, 128, "fp16"),     # Mamba-2-1.3B
    (8, 4096, 80, 256, 64, 128, "fp16"),     # Mamba-2-2.7B (the experiment shape)
    (4, 2048, 32, 256, 64, 128, "fp16"),     # Mamba-2-370M
    (8, 8192, 128, 256, 64, 128, "fp16"),    # 7B-ish, long seq
    (2, 4096, 80, 256, 64, 256, "fp16"),     # dstate=256 (bigger N)
    (4, 4096, 64, 128, 64, 128, "fp16"),     # chunk=128 (smaller K)
    (8, 2048, 64, 256, 128, 128, "fp16"),    # headdim=128 (bigger M)
    (8, 4096, 80, 256, 64, 128, "bf16"),     # bf16 width check (== fp16 config?)
]

# VAL — adversarial-referee-only
MAMBA_VAL = [
    (4, 4096, 80, 256, 64, 128, "fp16"),
    (2, 8192, 64, 256, 64, 128, "fp16"),
    (8, 4096, 48, 256, 64, 128, "fp16"),     # ~1.5B nheads=48
]

MAMBA_TEST = [  # HELD OUT
    (1, 4096, 64, 256, 64, 128, "fp16"),     # batch-1 (decode-ish grid)
    (8, 16384, 80, 256, 64, 128, "fp16"),    # long-context (real, stressful grid)
    (2, 4096, 128, 256, 128, 256, "fp16"),   # big headdim+dstate (stress dot)
    (4, 4096, 24, 256, 64, 128, "fp16"),     # Mamba-2-130M (small)
    (16, 2048, 64, 256, 64, 128, "fp16"),    # batch-16 (training grid)
    (8, 4096, 64, 128, 128, 128, "bf16"),    # chunk=128 + headdim=128, bf16
]


# ----------------------------------------------------------------------------
# fp8_gemm  (M, K, N, dtype) — fp8_e4m3 GEMM, the 8-bit band (examples/fp8_gemm.py)
# NOTE: tc here is cuBLAS `_scaled_mm` (~2.5x faster) — fp8 has a Helion codegen
# ceiling, so the bar is beat-default + approach the *Helion* ceiling, NOT tc.
# Large-K fp8 can hit the known slow-ptxas pathology — set a per-config compile
# timeout (HELION_AUTOTUNE_COMPILE_TIMEOUT); the search skips those configs.
# ----------------------------------------------------------------------------
FP8_TRAIN = [
    (2048, 2048, 2048, "fp8"),       # small cube
    (4096, 4096, 4096, "fp8"),       # attn proj cube
    (8192, 8192, 8192, "fp8"),       # large
    (8192, 4096, 4096, "fp8"),       # 8k tokens
    (4096, 4096, 11008, "fp8"),      # Llama-2-7B FFN up
    (4096, 14336, 4096, "fp8"),      # Llama-3-8B FFN down (large K)
    (4096, 4096, 14336, "fp8"),      # Llama-3-8B FFN up
    (256, 4096, 4096, "fp8"),        # small / prefill
    (8, 4096, 14336, "fp8"),         # decode FFN
    (32, 4096, 4096, "fp8"),         # decode batch-32
    (4096, 4096, 28672, "fp8"),      # big FFN up
]
FP8_VAL = [  # VAL — adversarial-referee-only
    (4096, 4096, 8192, "fp8"),       # mid FFN
    (4, 4096, 4096, "fp8"),          # decode batch-4
    (16384, 4096, 4096, "fp8"),      # 16k tokens
]
FP8_TEST = [  # HELD OUT
    (4096, 4096, 16384, "fp8"),      # FFN up
    (8192, 14336, 4096, "fp8"),      # FFN down, 8k tokens (large K)
    (16, 8192, 8192, "fp8"),         # decode batch-16
    (4096, 4096, 128256, "fp8"),     # fp8 LM head / vocab
    (512, 4096, 4096, "fp8"),        # small
    (8192, 5120, 5120, "fp8"),       # 13B attn proj
]


SPLITS = {  # train / val / test  (val is adversarial-referee-only — see module docstring)
    "matmul": {"train": MATMUL_TRAIN, "val": MATMUL_VAL, "test": MATMUL_TEST},
    "mamba2_chunk_state": {"train": MAMBA_TRAIN, "val": MAMBA_VAL, "test": MAMBA_TEST},
    "fp8_gemm": {"train": FP8_TRAIN, "val": FP8_VAL, "test": FP8_TEST},
}


def matmul_mnk_width(shape: tuple) -> tuple[int, int, int, int]:
    """(M, K, N, dtype) -> the dot's (M, N, K, width_bits) — the heuristic key."""
    m, k, n, dt = shape
    return m, n, k, WIDTH_BITS[dt]


def mamba_dot_mnk_width(shape: tuple) -> tuple[int, int, int, int]:
    """(b, seq, nh, chunk, hd, ds, dtype) -> the inner dot's (M, N, K, width_bits).
    M=headdim, N=dstate, K=chunk; batch = b * (seq//chunk) * nh (grid, not tile)."""
    b, seq, nh, chunk, hd, ds, dt = shape
    return hd, ds, chunk, WIDTH_BITS[dt]


def summary() -> None:
    for kern, sp in SPLITS.items():
        parts = " ".join(f"{k}={len(v)}" for k, v in sp.items())
        tot = sum(len(v) for v in sp.values())
        print(f"{kern}: {parts}  total={tot}")
    # sanity invariants (no GPU): mamba chunk divides seqlen
    for s in MAMBA_TRAIN + MAMBA_VAL + MAMBA_TEST:
        b, seq, nh, chunk, hd, ds, dt = s
        assert seq % chunk == 0, f"chunk !| seqlen: {s}"


if __name__ == "__main__":
    summary()
