"""CANONICAL pointwise/elementwise-kernel shape curriculum — DRAFT.

Drop-in companion to shapes_v3_draft.py (the reduction family), same five-bucket
discipline:
    train / val / test  -> MEASURABLE realistic shapes (the headline G suites)
    robustness          -> correctness-only canaries (decode / odd / non-pow2). NO G claim.

THE DEFINING PROPERTY: every kernel here is reduction-FREE in the forward pass. A
kernel whose forward carries a reduction (amax/sum/var/welford) is reduction-family,
NOT pointwise -- if a ReductionFact fires on it, leave it to the reduction heuristic
(don't claw it back; see the task file's disjointness rule + Gate-D divergence).

WHY these kernels (ranked):
  build-now : swiglu (SiLU(gate)*up), geglu (GELU_tanh(gate)*up)  -- the GLU MLP
              activation of ~every modern decoder LLM; N = intermediate_size.
  strong    : residual_add (h+residual, N=hidden_size, pure-BW floor).
  optional  : relu_squared (ungated, 2-traffic CONTRAST), bias_gelu (non-gated FFN,
              broadcast bias[N]), dyt (Dynamic-Tanh norm-replacement, FORWARD-ONLY:
              fwd is reduction-free, bwd reduces over rows -> bwd out of scope).
  EXCLUDED  : fp8 activation quant -- the realistic form is per-TOKEN (scale =
              amax(row)/448) = a REDUCTION over N -> reduction family. Only per-tensor
              STATIC quant is reduction-free, and that is not the real activation path.

Design rules (identical to the reduction curriculum):
  1. train covers every N-regime val/test probe -> test = interpolation, not a new regime.
  2. measurable splits clear the ~20us do_bench floor.  NOTE: pointwise is bf16
     (itemsize=2), so the floor math uses 2 bytes * TRAFFIC (STRICTER than the
     reduction file's *4).  Small-N bands are paired with LARGE M to stay measurable.
  3. sizes anchored to real model configs (verified vs HF config.json this session):
       GLU (swiglu/geglu/relu_squared) : N = intermediate_size
       residual_add / dyt             : N = hidden_size
       bias_gelu (non-gated FFN)      : N = 4*hidden_size
  4. train/val/test pairwise disjoint; robustness disjoint from all.
  5. dtype = bf16 (real LLM serving).  The tritonbench geglu / vector_add operators
     have NO DEFAULT_PRECISION (inherit fp32) -> pass --precision bf16 explicitly.

CRITICAL EVAL FOOTGUN: do NOT measure through the swiglu/geglu *tritonbench operators*
-- they bench the FULL matmul-heavy MLP (down_proj(act(gate_proj)*up_proj)) and the
elementwise win is a tiny fraction of wall time.  Measure the STANDALONE elementwise
op (examples/swiglu.py _swiglu_fwd(a,b) / geglu / add) with cold-L2 do_bench + gbps.
Robustness shapes (M<=512) fit in H100 L2 -> cold-L2 only; timings informational.

Convention: (M, N) with M = tokens = batch*seq_len.
"""
from __future__ import annotations

# Per-kernel traffic factor in units of (M*N*itemsize) bytes.  bf16 itemsize=2.
#   swiglu/geglu/residual_add : read 2 + write 1  -> 3  (gated act / add)
#   relu_squared/bias_gelu/dyt: read 1 + write 1  -> 2  (unary; the bytes-aware probe)
TRAFFIC = {
    "swiglu": 3, "geglu": 3, "residual_add": 3,
    "relu_squared": 2, "bias_gelu": 2, "dyt": 2,
}
ITEMSIZE = 2          # bf16
HBM_BYTES_PER_S = 2.0e12


SHAPES = {

    # ---- swiglu : SiLU(gate)*up. N = intermediate_size. -----------------------
    "swiglu": {
        "train": [
            (32768, 1536), (16384, 2048), (16384, 2880), (8192, 3072),   # Qwen3-235B/DSV2 expert, DSV3 expert, GPT-OSS, Qwen3-0.6B
            (16384, 8192), (8192, 11008), (8192, 12288), (8192, 13824),  # Phi-3-mini, Llama-2-7B, Qwen3-8B, Qwen2.5-14B
            (8192, 14336), (4096, 17408), (4096, 17920), (4096, 18432),  # Llama-3-8B/Mistral/Mixtral-7B-expert, Qwen3-14B, Phi-3-medium/Phi-4, DeepSeek-V3 dense
            (4096, 18944), (4096, 25600), (4096, 28672), (2048, 29568),  # Qwen2.5-7B, Qwen3-32B, Llama-3-70B, Qwen2.5-72B
        ],
        "val": [
            (32768, 2048), (16384, 9728), (16384, 12288), (8192, 16384),  # DSV3 expert, Qwen3-4B, Qwen3-8B, Mixtral-8x22B-expert
            (8192, 20480), (4096, 27648), (4096, 29568),                  # Yi-34B, Qwen2.5-32B, Qwen2.5-72B
        ],
        "test": [
            (8192, 2880), (16384, 8960), (16384, 11008), (4096, 15360),   # GPT-OSS, Qwen2-1.5B, Llama-2-7B, mid interp
            (8192, 22016), (2048, 24576), (8192, 28672),                  # Llama-1-65B, large interp, Llama-3-70B
        ],
        "robustness": [
            (1, 11008), (7, 14336), (33, 14336), (64, 11008),            # decode M=1, odd/prime M
            (128, 18944), (256, 14336),                                  # busy-serving decode concurrency (<20us)
            (2048, 11007), (512, 14337),                                 # non-pow2 N (partial last tile)
        ],
    },

    # ---- geglu : GELU_tanh(gate)*up. N = intermediate_size (Gemma/PaLM). ------
    "geglu": {
        "train": [
            (16384, 2880), (8192, 3072), (8192, 6912),                   # small-N coverage, coverage, Gemma-3-1B
            (8192, 9216), (8192, 10240), (8192, 14336), (8192, 15360),   # Gemma-2-2B, Gemma-3-4B, Gemma-2-9B, Gemma-3-12B
            (4096, 16384), (4096, 21504), (4096, 24576),                 # Gemma-1-2B, Gemma-3-27B, Gemma-1-7B
            (4096, 36864), (2048, 36864),                                # Gemma-2-27B (two M bound the band)
        ],
        "val": [
            (16384, 6912), (16384, 10240), (8192, 16384),                # Gemma-3-1B, Gemma-3-4B, Gemma-1-2B
            (8192, 21504), (4096, 30720),                                # Gemma-3-27B, wide interp
        ],
        "test": [
            (16384, 9216), (4096, 14336), (2048, 21504),                 # Gemma-2-2B, Gemma-2-9B, Gemma-3-27B
            (2048, 24576), (8192, 36864),                                # Gemma-1-7B, Gemma-2-27B
        ],
        "robustness": [
            (1, 9216), (7, 16384), (33, 10240), (64, 9216),             # decode + odd M
            (128, 36864), (256, 14336),                                 # decode concurrency at wide-N
            (2048, 16383), (512, 9217),                                 # non-pow2 N
        ],
    },

    # ---- residual_add : h + residual. N = hidden_size. Pure-BW floor. --------
    "residual_add": {
        "train": [
            (32768, 768), (16384, 1024), (16384, 1536), (16384, 2048),   # GPT-2, Qwen3-0.6B, Qwen2-1.5B, Llama-3.2-1B
            (16384, 2560), (8192, 3072), (8192, 3584), (8192, 4096),     # Gemma-3-4B, Llama-3.2-3B, Qwen2-7B, Llama-3-8B/Mistral
            (8192, 5120), (8192, 7168), (4096, 8192),                    # Llama-2-13B/Qwen2.5-14B, DeepSeek-V3, Llama-3-70B/Qwen2.5-72B
        ],
        "val": [
            (32768, 1024), (8192, 2048), (16384, 3584),                  # Qwen3-0.6B, Llama-3.2-1B, Qwen2-7B
            (16384, 5120), (8192, 8192),                                 # Llama-2-13B, Llama-3-70B
        ],
        "test": [
            (32768, 1536), (4096, 2560), (16384, 4096),                  # Qwen2-1.5B, Gemma-3-4B, Llama-3-8B
            (4096, 7168), (16384, 8192),                                 # DeepSeek-V3, Llama-3-70B
        ],
        "robustness": [
            (1, 4096), (7, 4096), (33, 8192), (64, 4096),               # decode M=1, odd M
            (128, 8192), (256, 5120),                                   # decode concurrency
            (2048, 4095), (512, 8191),                                  # non-pow2 N
        ],
    },

    # ---- relu_squared : max(x,0)^2 (ungated, 1 input). N=intermediate. --------
    #      CONTRAST op: 2-tensor traffic vs GLU's 3 -> probes byte-awareness.
    "relu_squared": {
        "train": [
            (32768, 3072), (16384, 8192), (8192, 11008),                 # Qwen3-0.6B, Phi-3, Llama-2-7B
            (8192, 14336), (4096, 16384), (4096, 18944),                 # Llama-3-8B, Gemma-1-2B, Qwen2.5-7B
            (4096, 28672), (4096, 20480),                                # 70B-FFN, OPT-13B/Yi-34B
        ],
        "val": [
            (32768, 8192), (16384, 11008), (8192, 16384), (8192, 28672), # Phi-3, Llama-2-7B, Gemma-1-2B, 70B FFN
        ],
        "test": [
            (16384, 3072), (4096, 14336), (8192, 18944), (8192, 20480),  # Qwen3-0.6B, Llama-3-8B, Qwen2.5-7B, OPT-13B
        ],
        "robustness": [
            (1, 11008), (7, 14336), (64, 16384), (256, 11008),
            (2048, 11007),
        ],
    },

    # ---- bias_gelu : GELU(x + bias[N]) non-gated FFN epilogue. N=4*hidden. ----
    #      BROADCAST input (bias[N] over M). GPT-2/BERT/Falcon/OPT regime.
    #      AUTHOR on a PRE-PROJECTED x[M,N]: GELU(x+bias), NOT GELU(x@W+b)
    #      (else a MatmulFact fires and it is no longer pointwise).
    "bias_gelu": {
        "train": [
            (32768, 3072), (16384, 4096), (16384, 5120), (16384, 6400),  # GPT-2-small, BERT-large/GPT-2-med, GPT-2-large, GPT-2-xl
            (8192, 8192), (8192, 10240), (8192, 16384),                  # non-gated proxy, Phi-2, mid FFN
            (8192, 18176), (4096, 20480), (4096, 32768),                 # Falcon-7B, OPT-13B, Falcon-40B/inductor max
        ],
        "val": [
            (32768, 4096), (8192, 6400), (16384, 10240),                 # BERT-large, GPT-2-xl, Phi-2
            (16384, 16384), (8192, 32768),                               # mid FFN, Falcon-40B
        ],
        "test": [
            (32768, 5120), (16384, 8192), (4096, 18176),                 # GPT-2-large, proxy, Falcon-7B
            (8192, 20480), (2048, 32768),                                # OPT-13B, Falcon-40B
        ],
        "robustness": [
            (1, 4096), (7, 4096), (64, 8192), (256, 16384),
            (2048, 4095), (512, 8191),
        ],
    },

    # ---- dyt : tanh(alpha*x)*gamma[N] (+beta[N]) FORWARD ONLY. N=hidden. ------
    #      Norm-replacement; per-feature gamma/beta broadcast over rows. The FWD is
    #      reduction-free (the whole point of DyT); the BWD reduces -> fwd only.
    "dyt": {
        "train": [
            (32768, 768), (16384, 1024), (16384, 1536), (16384, 2048),   # GPT-2, Qwen3-0.6B, Qwen2-1.5B, Llama-3.2-1B
            (16384, 2560), (8192, 3072), (8192, 4096),                   # Gemma-3-4B, Llama-3.2-3B, Llama-3-8B
            (8192, 5120), (8192, 7168), (4096, 8192),                    # Llama-2-13B, DeepSeek-V3, Llama-3-70B
        ],
        "val": [
            (32768, 1024), (8192, 2048), (16384, 4096), (16384, 7168),   # Qwen3-0.6B, Llama-3.2-1B, Llama-3-8B, DeepSeek-V3
        ],
        "test": [
            (32768, 1536), (8192, 2560), (16384, 5120), (8192, 8192),    # Qwen2-1.5B, Gemma-3-4B, Llama-2-13B, Llama-3-70B
        ],
        "robustness": [
            (1, 4096), (7, 4096), (64, 4096), (256, 5120),
            (2048, 4095), (512, 8191),
        ],
    },
}


def _est_us(kernel, m, n):
    return m * n * ITEMSIZE * TRAFFIC[kernel] / HBM_BYTES_PER_S * 1e6


def validate():
    from itertools import combinations
    NOISE_US = 20.0
    problems = 0
    meas = ("train", "val", "test")
    print(f"{'kernel':14} {'split':10} {'n':>2} {'M-range':>16} {'N-range':>16} {'min_est_us':>11}")
    for k, splits in SHAPES.items():
        train_n = [n for _, n in splits["train"]]
        for sp in ("train", "val", "test", "robustness"):
            s = splits[sp]
            mr = (min(m for m, _ in s), max(m for m, _ in s))
            nr = (min(n for _, n in s), max(n for _, n in s))
            mn = min(_est_us(k, m, n) for m, n in s)
            print(f"{k:14} {sp:10} {len(s):>2} {str(mr):>16} {str(nr):>16} {mn:>11.1f}")
        # 1. pairwise disjoint among train/val/test + robustness
        allsp = meas + ("robustness",)
        for a, b in combinations(allsp, 2):
            ov = set(map(tuple, splits[a])) & set(map(tuple, splits[b]))
            if ov:
                print(f"  !! {k}: {a}&{b} OVERLAP {sorted(ov)}"); problems += 1
        # 2. val/test N within train envelope (interpolation)
        for sp in ("val", "test"):
            for m, n in splits[sp]:
                if not (min(train_n) <= n <= max(train_n)):
                    print(f"  !! {k}: {sp} ({m},{n}) N OUTSIDE train envelope "
                          f"[{min(train_n)},{max(train_n)}]"); problems += 1
        # 3. measurable splits clear the noise floor
        for sp in meas:
            for m, n in splits[sp]:
                if _est_us(k, m, n) < NOISE_US:
                    print(f"  !! {k}: {sp} ({m},{n}) est {_est_us(k,m,n):.1f}us "
                          f"< {NOISE_US}us NOISE FLOOR"); problems += 1
        # 4. balance
        nt, nv, nte = len(splits["train"]), len(splits["val"]), len(splits["test"])
        if not (8 <= nt <= 16): print(f"  ?? {k}: train n={nt} (want 8-16)")
        if not (4 <= nv <= 9):  print(f"  ?? {k}: val n={nv} (want 4-9)")
        if not (4 <= nte <= 9): print(f"  ?? {k}: test n={nte} (want 4-9)")
        print()
    tot = sum(len(s) for v in SHAPES.values() for s in v.values())
    print(f"{'PASS' if problems == 0 else 'FAIL'}: {problems} problem(s). "
          f"kernels={len(SHAPES)} all-bucket shapes={tot}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if validate() else 0)
