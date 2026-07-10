"""ADVERSARIAL pointwise curriculum — the 5 gap classes, anchored to REAL named workloads.

Each entry names the PROBE kernel (structural detector, from local/rope_probe/adv/shards) and the REAL
op whose structure it stands in for (Gate E firewall: a synthetic probe is a detector, not a faithful
curriculum member — so shapes are REAL model dims, and a held-out variant + the `test` split are read
only once at freeze).

Five-bucket discipline (mirrors shapes_pointwise_draft.py):
  train / val / test  -> MEASURABLE realistic shapes (headline G suites; > ~1M elems, > L2 where a
                         bandwidth claim is made). val/test N within the train envelope (interpolation).
  robustness          -> correctness-only canaries (small / non-pow2 / skinny). NO G claim.

Real dims used (verified against common model configs):
  hidden_size H     : 2048, 4096, 5120, 6144, 8192 (Llama/GPT/Gemma families)
  intermediate FFN  : 11008 (Llama-7B), 13824, 14336 (Llama-8B), 8192
  tokens M          : prefill 2048-8192 (batch*seq); decode 1-256 (skinny-M)
  attention/KV      : head_dim 64/128; a permuted [.., S, D] view is a transposed elementwise target
"""
from __future__ import annotations

# name -> {shard, kernel, cls, real_anchor, splits, boundary}
CURRICULUM = {
    # ============ CLASS 1: transposed / strided access (Lever 1: contiguity) ============
    # in-only transpose = the realistic case (an elementwise act on a .transpose()/permute view of an
    # activation or KV-cache); both-transposed is borderline (kept as the single-contig probe).
    "transposed_in_relu2": {
        "shard": "shard_transpose.json", "cls": "transpose",
        "real_anchor": "relu^2 activation applied to a .transpose(-1,-2)'d / permuted activation view "
                       "(e.g. attention out [B,H,S,D]->[B,S,H,D] contiguous-read of a transposed src)",
        "boundary": "CONFLICT (transposed load dim0, contiguous store dim1) -> balanced tile",
        "train": [(4096, 4096), (8192, 8192), (2048, 8192), (8192, 2048), (4096, 8192), (2048, 4096), (6144, 6144)],
        "val": [(5120, 5120), (6144, 4096)],
        "test": [(8192, 4096), (3072, 6144)],
        "robustness": [(1024, 1024), (512, 2048), (2048, 512), (4096, 4090)],
    },
    "transposed_both_scale": {
        "shard": "shard_transpose.json", "cls": "transpose",
        "real_anchor": "scale (RoPE-free) of a transposed activation written back transposed — a fused "
                       "transpose+affine (single contiguous axis = dim0 for both read and write)",
        "boundary": "SINGLE contiguous axis (both in&out stride-1 on dim0) -> [budget,1]",
        "train": [(4096, 4096), (8192, 2048), (2048, 8192), (8192, 8192), (6144, 6144)],
        "val": [(5120, 5120)],
        "test": [(4096, 8192)],
        "robustness": [(1024, 1024), (512, 2048), (4096, 4090)],
    },
    # ============ CLASS 2: compute-bound wide rows / num_warps starvation (Lever 2) ============
    "wide_row_trig_chain": {
        "shard": "shard_warps.json", "cls": "compute",
        "real_anchor": "transcendental-heavy fused elementwise (a deep SiLU/tanh/exp activation chain) "
                       "on a wide FFN activation [tokens, intermediate]",
        "boundary": "HIGH arith-intensity (sfu=12) big tile -> num_warps 16",
        "train": [(2048, 2048), (512, 8192), (4096, 4096), (2048, 11008), (8192, 4096), (512, 16384), (1024, 14336)],
        "val": [(4096, 8192), (2048, 8192)],
        "test": [(3072, 6144), (256, 16384)],
        "robustness": [(256, 256), (127, 4096), (2048, 2050)],
    },
    "wide_row_moderate_chain": {
        "shard": "shard_warps.json", "cls": "compute",
        "real_anchor": "moderate fused activation (a 2-3 op SiLU+gelu tail) on a wide activation — the "
                       "num_warps ramp's LOW-intensity boundary (must NOT over-warp)",
        "boundary": "MODERATE intensity (sfu=4) — near the ramp breakpoint (w4 vs w8)",
        "train": [(2048, 2048), (1024, 4096), (3072, 3072), (4096, 4096), (2048, 8192)],
        "val": [(2048, 4096)],
        "test": [(4096, 2048)],
        "robustness": [(256, 256), (2048, 2050)],
    },
    "skinny_m_wide_n_chain": {
        "shard": "shard_warps.json", "cls": "compute",
        "real_anchor": "transcendental elementwise during DECODE (tiny M, huge N) — small grid AND "
                       "under-warped tiles",
        "boundary": "skinny-M wide-N (small grid) + high intensity",
        "train": [(64, 65536), (128, 32768), (32, 131072), (64, 32768)],
        "val": [(128, 65536)],
        "test": [(96, 49152)],
        "robustness": [(1, 32768), (8, 8192)],
    },
    "manydiff_chains": {
        "shard": "shard_temporaries.json", "cls": "compute+temporaries",
        "real_anchor": "heavy fused activation with many simultaneously-live intermediates (a high-order "
                       "GELU-tanh polynomial / multi-branch activation) on FFN activations",
        "boundary": "HIGHEST intensity (sfu=20, all live) — smaller tile + more warps",
        "train": [(4096, 4096), (8192, 8192), (2048, 11008), (4096, 8192), (2048, 14336), (8192, 2048)],
        "val": [(6144, 6144), (4096, 11008)],
        "test": [(8192, 4096), (3072, 12288)],
        "robustness": [(128, 256), (127, 4096), (256, 256)],
    },
    "horner_poly24": {
        "shard": "shard_temporaries.json", "cls": "compute",
        "real_anchor": "a high-degree polynomial approximation (Horner form) — 0 SFU but a deep FMA "
                       "dependency chain; the ramp's op-count-vs-SFU divergence probe",
        "boundary": "0 SFU, 77 cheap FMAs in a depth-24 chain (op-count high, SFU 0)",
        "train": [(4096, 4096), (8192, 8192), (2048, 8192), (4096, 2048)],
        "val": [(6144, 6144)],
        "test": [(8192, 4096)],
        "robustness": [(64, 128), (256, 256)],
    },
    # ============ CLASS 3: high fan-in (Lever 3: per-operand byte budget) ============
    "fanin64_2d_add": {
        "shard": "shard_fanin.json", "cls": "fanin",
        "real_anchor": "multi-way residual / MoE expert-combine: sum of K contributions "
                       "(K-way residual stream add) on [tokens, hidden]",
        "boundary": "K=64 full-extent operands inflate bytes_per_elem -> tile starves without a "
                    "per-operand view",
        "train": [(4096, 4096), (8192, 2048), (2048, 8192), (4096, 8192)],
        "val": [(6144, 4096)],
        "test": [(8192, 4096)],
        "robustness": [(1024, 1024), (256, 4096)],
    },
    "fanin32_2d_silu": {
        "shard": "shard_fanin.json", "cls": "fanin",
        "real_anchor": "32-way expert combine + SiLU (a realistic MoE combine with an activation)",
        "boundary": "K=32 operands + per-element SiLU",
        "train": [(4096, 4096), (2048, 8192), (8192, 2048), (4096, 8192)],
        "val": [(6144, 4096)],
        "test": [(8192, 4096)],
        "robustness": [(1024, 1024), (256, 4096)],
    },
    "fanin96_bf16_2d": {
        "shard": "shard_fanin.json", "cls": "fanin",
        "real_anchor": "very-high fan-in bf16 combine (K=96) — the fan-in boundary in bf16 storage",
        "boundary": "K=96 bf16 operands; both budget & reg cap agree on a sub-default tile",
        "train": [(4096, 4096), (8192, 4096), (2048, 8192)],
        "val": [(6144, 6144)],
        "test": [(8192, 8192)],
        "robustness": [(1024, 1024)],
    },
    # ============ CLASS 4: broadcast + compute (subsumed by contiguity/bytes; Lever 2 warps too) ==
    "bcast_rowvec_transcend": {
        "shard": "shard_broadcast.json", "cls": "broadcast",
        "real_anchor": "per-column affine + activation (LayerNorm-affine-like: x*gamma[N]+beta[N] then "
                       "a transcendental), reduction-free",
        "boundary": "per-row broadcast operands excluded from byte model + heavy compute",
        "train": [(8192, 8192), (2048, 512), (4096, 4096), (2048, 8192)],
        "val": [(6144, 6144)],
        "test": [(8192, 4096)],
        "robustness": [(1024, 1024), (256, 4096)],
    },
}

# Real-workload anchor set held OUT for Gate E (read once at freeze): the `test` split of every kernel
# + one whole gap-class variant kernel per class (the synthetic-probe holdout).
HELDOUT_KERNELS = ["wide_row_moderate_chain", "fanin32_2d_silu"]  # variants; fit uses the primaries


def validate() -> int:
    problems = 0
    for name, spec in CURRICULUM.items():
        splits = {k: spec[k] for k in ("train", "val", "test", "robustness")}
        allm = [tuple(s) for v in splits.values() for s in v]
        # 1. pairwise disjoint train/val/test/robustness
        seen = set()
        for sp in ("train", "val", "test", "robustness"):
            for s in splits[sp]:
                t = tuple(s)
                if t in seen:
                    print(f"  !! {name}: {sp} shape {t} duplicated across splits"); problems += 1
                seen.add(t)
        # 2. val/test N within train envelope (interpolation, not extrapolation)
        train_ns = [n for (_, n) in splits["train"]]
        lo, hi = min(train_ns), max(train_ns)
        for sp in ("val", "test"):
            for (m, n) in splits[sp]:
                if not (lo <= n <= hi):
                    print(f"  ?? {name}: {sp} ({m},{n}) N outside train N-envelope [{lo},{hi}]"); problems += 1
        # 3. measurable splits clear ~1M elements (a G claim needs a non-noise size)
        for sp in ("train", "val", "test"):
            for (m, n) in splits[sp]:
                if m * n < 1_000_000:
                    print(f"  ?? {name}: {sp} ({m},{n}) = {m*n} elems < 1M (measurability floor)"); problems += 1
        # 4. train size sane
        nt = len(splits["train"])
        if not (3 <= nt <= 16):
            print(f"  ?? {name}: train n={nt} (want 3-16)")
    tot = sum(len(v[s]) for v in CURRICULUM.values() for s in ("train", "val", "test", "robustness"))
    print(f"VALIDATE: {'OK' if problems == 0 else str(problems)+' PROBLEMS'} | "
          f"kernels={len(CURRICULUM)} all-bucket shapes={tot} heldout_kernels={HELDOUT_KERNELS}")
    return problems


if __name__ == "__main__":
    validate()
