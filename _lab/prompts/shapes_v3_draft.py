"""CANONICAL reduction-kernel shape curriculum — v3 DRAFT (single source of truth).

Design rules (agreed with user):
  1. Per kernel, FIVE buckets:
       train / val / test   -> MEASURABLE realistic shapes (the headline G suites)
       robustness           -> correctness-only canaries (tiny, prime, non-pow2,
                               extreme extrapolation, grid-bound). NO G claim.
       (transfer lives in TRANSFER below: kernels the heuristic was NOT tuned on.)
  2. train MUST cover every N-regime (band) that val/test probe -> test is
     INTERPOLATION within the trained envelope, never a regime train never saw.
  3. Measurable splits must clear the do_bench noise floor (est >= ~20us). To probe
     a "small" regime, make only ONE dim small (tiny-N x large-M) so it stays
     measurable; genuine tiny x tiny goes to robustness (correctness only).
  4. Sizes anchored to real model configs (hidden dims / vocab / token batches).
  5. train/val/test pairwise disjoint; balanced ~12-16 / ~7 / ~7 per kernel.

Every harness should IMPORT splits from here instead of re-listing shapes.
Run `python shapes_v3_draft.py` to validate all invariants.

Convention: row-reductions (M, N) with N reduced; losses (BT, V); welford (S, D).
Inline comment = real-model anchor where applicable.
"""
from __future__ import annotations

# Memory-traffic factor (x M*N*4 bytes) used only for the noise-floor ESTIMATE.
TRAFFIC = {
    "rms_norm": 2, "layer_norm": 2, "softmax": 2, "welford": 2,  # read x + write y
    "sum": 1, "long_sum": 1, "cross_entropy": 1,                  # read-dominated
    "kl_div": 2, "jsd": 2,                                        # read two MxN inputs
    # transfer kernels (fact-distinct probes):
    "tv_distance": 2, "argmax": 1, "l2_norm": 1, "minmax_normalize": 2,
}
HBM_BYTES_PER_S = 2.0e12  # conservative effective H100 HBM bandwidth


# N-band edges per kernel family (upper-inclusive thresholds). A val/test shape's
# N-band must also appear in train.
NORM_BANDS = [2048, 4096, 8192, 16384, 1 << 30]      # small/med/large/xlarge/huge
SUM_BANDS = [2048, 4096, 8192, 16384, 1 << 30]
LONGSUM_BANDS = [131072, 524288, 1 << 30]            # long/xlong/huge
VOCAB_BANDS = [50304, 128256, 1 << 30]               # small/med/large vocab

BANDS = {
    "rms_norm": NORM_BANDS, "layer_norm": NORM_BANDS, "softmax": NORM_BANDS,
    "welford": NORM_BANDS, "sum": SUM_BANDS, "long_sum": LONGSUM_BANDS,
    "cross_entropy": VOCAB_BANDS, "kl_div": VOCAB_BANDS, "jsd": VOCAB_BANDS,
}


# =========================================================================== #
#  MEASURABLE SPLITS  (train / val / test)  +  robustness (correctness-only)
# =========================================================================== #

SHAPES = {

    # ---- rms_norm : N = hidden dim. M = tokens (batch*seq). -------------------
    "rms_norm": {
        "train": [
            (8192, 768), (8192, 1024), (8192, 1536), (4096, 2048),     # small  (GPT2/BERT, Qwen2-1.5B, Gemma-2B)
            (8192, 2560), (8192, 3072), (4096, 3584), (8192, 4096),    # med    (Phi-2, Phi-3/Gemma-7B, Qwen2-7B, Llama-7B)
            (4096, 5120), (4096, 7168), (4096, 8192),                  # large  (Llama-13B, Yi-34B, Llama-70B)
            (2048, 12288), (2048, 14336), (2048, 16384),               # xlarge (GPT-3, Llama-MLP)
            (16384, 4096), (32768, 2048),                              # M-variation
        ],
        "val": [
            (16384, 1024), (8192, 2048), (8192, 3584), (4096, 4096),
            (2048, 5120), (2048, 8192), (4096, 12288), (16384, 2048),
        ],
        "test": [
            (16384, 896), (8192, 1280), (16384, 1536), (4096, 2560),   # Qwen0.5B, GPT2-large (M bumped off noise floor)
            (2048, 4096), (2048, 6144), (2048, 7168), (2048, 10240),
        ],
        "robustness": [
            (1, 4096), (16, 4096), (128, 8192),                        # tiny-M (inference)
            (2048, 1025), (2048, 2047), (2048, 8191),                  # non-pow2 + Mersenne prime
            (262144, 256), (1, 131072),                                # grid-bound, tiny-M-huge-N
        ],
    },

    # ---- layer_norm : same regime as rms_norm (distinct M/N picks). -----------
    "layer_norm": {
        "train": [
            (8192, 768), (8192, 1024), (8192, 1536), (8192, 2048),
            (4096, 2560), (8192, 3072), (8192, 3584), (4096, 4096),
            (4096, 5120), (4096, 7168), (4096, 8192),
            (2048, 12288), (2048, 14336), (2048, 16384),
            (16384, 2048), (32768, 1024),
        ],
        "val": [
            (16384, 1024), (4096, 1536), (8192, 2560), (8192, 4096),
            (2048, 5120), (2048, 8192), (4096, 12288), (16384, 4096),
        ],
        "test": [
            (16384, 896), (8192, 1280), (4096, 2048), (4096, 3584),    # M bumped off noise floor
            (2048, 4096), (2048, 6144), (2048, 7168), (2048, 10240),
        ],
        "robustness": [
            (1, 4096), (16, 4096), (128, 8192),
            (2048, 1025), (2048, 2047), (2048, 8191),
            (262144, 256), (1, 131072),
        ],
    },

    # ---- softmax : generic row softmax; N spans attn/feature to long-context. -
    "softmax": {
        "train": [
            (262144, 128), (131072, 256), (16384, 512),                # tiny-N attention (short ctx), grid-occupancy
            (8192, 1024), (8192, 2048),                                # small
            (8192, 2560), (4096, 3072), (4096, 4096),                  # med
            (4096, 5120), (4096, 8192),                                # large
            (4096, 16384), (2048, 24576), (2048, 32768),               # xlarge
            (1024, 65536), (512, 131072),                              # huge (long-context)
        ],
        "val": [
            (262144, 256), (8192, 768), (8192, 1536), (8192, 4096), (4096, 6144),
            (2048, 12288), (2048, 49152), (1024, 98304),
        ],
        "test": [
            (131072, 128), (8192, 896), (8192, 1280), (8192, 3072), (4096, 7168),
            (2048, 16384), (2048, 40960), (512, 98304),
        ],
        "robustness": [
            (16, 4096), (128, 4096), (2048, 1023), (2048, 2047),
            (262144, 257), (1, 131072), (4096, 8191), (1, 262144),
        ],
    },

    # ---- welford : N = hidden (norms regime). FIX: train now covers wide N. ---
    "welford": {
        "train": [
            (16384, 768), (16384, 1024), (16384, 1536), (16384, 2048),  # small
            (16384, 2560), (8192, 3072), (16384, 4096),                # med
            (8192, 5120), (8192, 7168), (8192, 8192),                  # large
            (8192, 12288), (4096, 16384),                              # xlarge
            (262144, 2048), (65536, 4096), (32768, 8192),              # M-variation (huge-M legacy)
        ],
        "val": [
            (8192, 768), (8192, 4096), (8192, 6144), (16384, 3072),
            (4096, 10240), (8192, 2048), (4096, 12288),
        ],
        "test": [
            (16384, 896), (16384, 1280), (8192, 3584), (16384, 5120),
            (8192, 14336), (32768, 2560), (16384, 7168),
        ],
        "robustness": [
            (262144, 768), (262144, 1543), (131072, 2048),            # grid-bound, PRIME canary
            (262144, 5120), (8192, 2049), (4096, 1025),               # wide + non-pow2
            (65536, 16384), (262144, 7168),                           # wide-N (mem-heavy)
        ],
    },

    # ---- sum : generic row reduction; N up to MLP-intermediate. ---------------
    "sum": {
        "train": [
            (16384, 1024), (16384, 2048),                             # small
            (16384, 2560), (8192, 3072), (8192, 4096),                # med
            (8192, 5120), (8192, 8192),                               # large
            (4096, 11008), (4096, 14336), (4096, 16384),              # xlarge (Llama-2-7B / Llama-3-8B MLP)
            (4096, 28672), (2048, 18944),                             # huge (Llama-2-70B FFN / Qwen2-7B FFN)
            (32768, 1024), (16384, 4096),                             # M-variation
        ],
        "val": [
            (16384, 1536), (8192, 3584), (8192, 6144), (4096, 12288),
            (2048, 28672), (16384, 5120), (8192, 2048),
        ],
        "test": [
            (16384, 1280), (8192, 2560), (8192, 7168), (4096, 10240),
            (4096, 18432), (2048, 24576), (16384, 3072),
        ],
        "robustness": [
            (1, 4096), (16, 4096), (2048, 1023), (2048, 2047),
            (262144, 256), (1, 262144), (8192, 8191), (4, 1048576),
        ],
    },

    # ---- long_sum : large-N looped reduction; M bumped to stay measurable. ----
    "long_sum": {
        "train": [
            (256, 65536), (256, 98304),                               # long
            (256, 131072), (128, 131072), (128, 196608),              # long/xlong
            (128, 262144), (64, 262144), (96, 393216),                # xlong
            (64, 524288), (64, 786432), (64, 1048576),                # huge
            (16, 2097152),                                            # >2^20 looped tail (measurable: M lifted)
        ],
        "val": [
            (192, 131072), (96, 262144), (128, 163840),
            (64, 393216), (96, 524288), (64, 655360),
        ],
        "test": [
            (160, 131072), (128, 229376), (96, 196608),
            (64, 294912), (128, 524288), (48, 786432), (8, 2097152),
        ],
        "robustness": [
            (1, 32768), (2, 65536), (4, 131072), (8, 262144),         # few-row (noise floor)
            (1, 1048576), (4, 262143), (1, 1000003), (16, 65537),     # non-pow2 + primes
        ],
    },

    # ---- cross_entropy : (BT, V). V = real vocabs. ---------------------------
    "cross_entropy": {
        "train": [
            (8192, 30522), (8192, 32000), (8192, 32064), (8192, 50257),   # BERT, Llama2, Phi3, GPT2
            (4096, 50304), (4096, 65536), (8192, 49152),                  # NeoX-padded, -, -
            (4096, 98304), (4096, 128000), (4096, 128256),                # -, -, Llama3
            (8192, 128256), (2048, 151936), (2048, 256000),               # Llama3 large-M (online-vs-2pass), Qwen2, Gemma
            (16384, 32000),                                               # M-variation
        ],
        "val": [
            (4096, 32000), (8192, 40960), (4096, 50257), (2048, 100352),
            (2048, 128256), (1024, 151936), (2048, 200000),
        ],
        "test": [
            (8192, 32768), (4096, 49152), (2048, 50257), (4096, 114688),
            (1024, 128256), (4096, 151936), (1024, 250000),
        ],
        "robustness": [
            (1, 32000), (16, 128256), (128, 151936),                  # tiny-M
            (2048, 32003), (2048, 50261), (4, 256000),                # prime V
            (262144, 4096), (8192, 1024),                             # small-V grid-bound
        ],
    },

    # ---- kl_div : (BT, V), reads two MxV inputs. -----------------------------
    "kl_div": {
        "train": [
            (8192, 30522), (8192, 32000), (4096, 32064), (8192, 50257),
            (4096, 50304), (4096, 65536), (4096, 49152),
            (2048, 98304), (4096, 128256), (2048, 128000),
            (2048, 151936), (1024, 256000),
            (16384, 32000),
        ],
        "val": [
            (4096, 32000), (8192, 40960), (4096, 50257), (2048, 100352),
            (2048, 128256), (1024, 151936), (2048, 200000),
        ],
        "test": [
            (8192, 32768), (8192, 49152), (2048, 50257), (4096, 114688),
            (1024, 128256), (4096, 151936), (1024, 250000),
        ],
        "robustness": [
            (1, 32000), (16, 128256), (128, 151936),
            (2048, 32003), (2048, 50261), (4, 256000),
            (262144, 4096), (8192, 1024),
        ],
    },

    # ---- jsd : (BT, V), heavy epilogue (Band-B). -----------------------------
    "jsd": {
        "train": [
            (8192, 30522), (8192, 32000), (8192, 32064), (8192, 50257),
            (8192, 50304), (8192, 65536), (4096, 49152),
            (4096, 98304), (8192, 128256), (4096, 128000),
            (4096, 151936), (2048, 256000),
            (16384, 32000),
        ],
        "val": [
            (4096, 32000), (8192, 40960), (4096, 50257), (2048, 100352),
            (4096, 128256), (2048, 151936), (2048, 200000),
        ],
        "test": [
            (8192, 32768), (8192, 49152), (2048, 50257), (4096, 114688),
            (2048, 128256), (8192, 151936), (1024, 250000),
        ],
        "robustness": [
            (1, 32000), (16, 128256), (128, 151936),
            (2048, 32003), (2048, 50261), (4, 256000),
            (262144, 4096), (8192, 1024),
        ],
    },
}


# =========================================================================== #
#  TRANSFER suite — NEW kernels the heuristic is NOT tuned on (Goal 5).
#  Reported as a SEPARATE "transfer to unseen kernel" number, never folded into
#  the per-kernel shape-generalization headline. No train split (by definition).
# =========================================================================== #

TRANSFER = {
    # Band-B accumulator probe: NEW loss, THINNER epilogue than kl/jsd (tv = 0.5*sum|p-q|).
    # Tests whether the R_BLOCK byte-cap generalizes to a light-epilogue Band-B kernel.
    # Includes narrow-V (persistent Band-B, accumulator below the 16KiB cap).
    "tv_distance": [
        (8192, 32000), (8192, 50257), (4096, 128256), (4096, 151936),
        (8192, 65536), (2048, 256000), (131072, 256), (65536, 4096),
    ],
    # op-variety probe: row argmax — a NON-additive reduction op (max+index). Flips
    # num_reduction_ops / the op-agnostic claim — no in-sample kernel uses argmax.
    "argmax": [
        (16384, 1024), (8192, 4096), (8192, 8192), (4096, 16384),
        (8192, 32000), (4096, 128256), (32768, 2048), (8192, 2560),
    ],
    # single-pass STREAMED num_load=1 reduction with NO apply pass — distinct from
    # the 2-load norms and from welford/standardize (which both have an apply pass).
    # Probes the streamed-load eviction='first' policy on an unseen kernel.
    "l2_norm": [
        (16384, 1024), (8192, 4096), (8192, 8192), (4096, 16384),
        (2048, 32768), (8192, 2560), (4096, 5120), (16384, 768),
    ],
    # structured-combine probe with a NON-additive (max,min) combine — different
    # combine arity/recurrence than welford(3-stat) and standardize(2-moment).
    "minmax_normalize": [
        (16384, 1024), (16384, 4096), (8192, 8192), (4096, 16384),
        (8192, 2560), (8192, 5120), (32768, 2048), (16384, 1536),
    ],
}


# =========================================================================== #
#  VALIDATOR
# =========================================================================== #

def _est_us(kernel, m, n):
    return m * n * 4 * TRAFFIC[kernel] / HBM_BYTES_PER_S * 1e6


def _band(n, edges):
    for i, e in enumerate(edges):
        if n <= e:
            return i
    return len(edges)


def validate():
    from itertools import combinations
    NOISE_US = 20.0
    problems = 0
    print(f"{'kernel':14} {'split':10} {'n':>2} {'M-range':>16} {'N-range':>16} "
          f"{'min_est_us':>11}")
    for k, splits in SHAPES.items():
        edges = BANDS[k]
        train = splits["train"]
        train_bands = {_band(n, edges) for _, n in train}
        train_n = [n for _, n in train]
        # 'coverage' = MEASURABLE shapes that exercise a real heuristic branch but
        # are NOT standard realistic sizes; trained on + measured, but reported
        # SEPARATELY from the realistic train/val/test headline. Optional per kernel.
        meas_splits = [s for s in ("train", "val", "test", "coverage") if s in splits]
        check_splits = [s for s in ("val", "test", "coverage") if s in splits]
        for sp in meas_splits + ["robustness"]:
            s = splits[sp]
            mr = (min(m for m, _ in s), max(m for m, _ in s))
            nr = (min(n for _, n in s), max(n for _, n in s))
            mn = min(_est_us(k, m, n) for m, n in s)
            print(f"{k:14} {sp:10} {len(s):>2} {str(mr):>16} {str(nr):>16} "
                  f"{mn:>11.1f}")

        # --- invariant checks ---
        meas = [t for sp in meas_splits for t in splits[sp]]
        # 1. pairwise disjoint among all measurable splits (train/val/test/coverage)
        for a, b in combinations(meas_splits, 2):
            ov = set(map(tuple, splits[a])) & set(map(tuple, splits[b]))
            if ov:
                print(f"  !! {k}: {a}&{b} OVERLAP {sorted(ov)}"); problems += 1
        # robustness should also be disjoint from measurable splits
        ovr = set(map(tuple, splits["robustness"])) & set(map(tuple, meas))
        if ovr:
            print(f"  !! {k}: robustness overlaps measurable {sorted(ovr)}")
            problems += 1
        # 2. train covers every N-band that val/test/coverage probe
        for sp in check_splits:
            for m, n in splits[sp]:
                if _band(n, edges) not in train_bands:
                    print(f"  !! {k}: {sp} ({m},{n}) N-band not in train "
                          f"(train N {min(train_n)}..{max(train_n)})"); problems += 1
        # 3. N envelope: val/test/coverage N within train N range
        for sp in check_splits:
            for m, n in splits[sp]:
                if not (min(train_n) <= n <= max(train_n)):
                    print(f"  !! {k}: {sp} ({m},{n}) N OUTSIDE train envelope")
                    problems += 1
        # 4. measurable splits clear the noise floor
        for sp in meas_splits:
            for m, n in splits[sp]:
                if _est_us(k, m, n) < NOISE_US:
                    print(f"  !! {k}: {sp} ({m},{n}) est {_est_us(k,m,n):.1f}us "
                          f"< {NOISE_US}us NOISE FLOOR"); problems += 1
        # 5. balance
        nt, nv, nte = len(train), len(splits["val"]), len(splits["test"])
        if not (12 <= nt <= 16): print(f"  ?? {k}: train n={nt} (want 12-16)")
        if not (6 <= nv <= 9):   print(f"  ?? {k}: val n={nv} (want 6-9)")
        if not (6 <= nte <= 9):  print(f"  ?? {k}: test n={nte} (want 6-9)")
        print()

    # transfer: just check measurable + report
    print("--- TRANSFER (unseen kernels; measured separately) ---")
    for k, s in TRANSFER.items():
        mn = min(_est_us(k, m, n) for m, n in s)
        nf = [(m, n) for m, n in s if _est_us(k, m, n) < NOISE_US]
        print(f"{k:16} n={len(s):>2} min_est_us={mn:>7.1f}"
              + (f"  NOISE-FLOOR:{nf}" if nf else ""))
        if nf:
            problems += 1

    print(f"\n{'PASS' if problems == 0 else 'FAIL'}: {problems} problem(s).")
    # totals
    tot = sum(len(s) for v in SHAPES.values() for s in v.values())
    ncov = sum(len(v.get("coverage", [])) for v in SHAPES.values())
    print(f"Total kernels={len(SHAPES)} all-bucket shapes={tot} "
          f"(coverage={ncov}) transfer kernels={len(TRANSFER)}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if validate() else 0)
