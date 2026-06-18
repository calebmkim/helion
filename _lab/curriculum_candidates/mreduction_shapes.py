"""M-reduction curriculum shapes — train/val/test (+ optional robustness).

Modeled after `_lab/prompts/shapes_v3_draft.py`, adapted for the M-reduction
(grad_w / param-gradient) kernels, whose shapes are heterogeneous:

  bias_grad / dyt   : (M, N)        M = collapsed rows (tokens), N = feature width
  group_norm        : (N, C, S, G)  N = batch (grid/collapse), C = channels, S = spatial, G = groups
  instance_norm     : (B, C, S)     B = batch (grid/collapse), C = channels, S = spatial

Banding axis = the per-row FEATURE FOOTPRINT `F` (what the byte-cap / warp-ramp key on):
  bias_grad / dyt   : F = N
  group_norm        : F = C * S      (the resident [inner, C, S] tile)
  instance_norm     : F = C * S

Design rules (same spirit as shapes_v3_draft.py):
  1. train / val / test = MEASURABLE realistic shapes (the headline G suites);
     robustness = correctness-only canaries (OPTIONAL here per user).
  2. train MUST cover every F-band that val/test probe -> test is INTERPOLATION
     within the trained envelope, never an F-regime train never saw.
  3. Measurable splits clear the do_bench noise floor (est >= ~20us).
  4. train/val/test pairwise disjoint; ~12-16 train / ~6-9 val / ~6-9 test.

Run `python mreduction_shapes.py` to validate all invariants.
"""
from __future__ import annotations

HUGE = 1 << 30
HBM_BYTES_PER_S = 2.0e12  # conservative effective H100 HBM bandwidth (noise-floor est only)

# Per-kernel metadata: arity (len of shape tuple), HBM traffic factor (x elems*4 B),
# and the F-band edges (upper-inclusive) the curriculum must keep train-covered.
#   bias_grad : read grad_out (M*N), write [N]                 -> ~1x
#   dyt       : read x+grad_out (2 M*N), write grad_x (M*N)    -> ~3x
#   group/inst: read x+grad_out (2 N*C*S), write grad_x        -> ~3x
META = {
    "bias_grad":     {"arity": 2, "traffic": 1, "bands": [2048, 4096, 8192, 16384, HUGE]},
    "dyt":           {"arity": 2, "traffic": 3, "bands": [2048, 4096, 8192, 16384, HUGE]},
    "group_norm":    {"arity": 4, "traffic": 3, "bands": [8192, 32768, 131072, HUGE]},
    "instance_norm": {"arity": 3, "traffic": 3, "bands": [8192, 32768, 131072, HUGE]},
}


def feature(kernel: str, shape: tuple) -> int:
    """Per-row feature footprint F (the byte-cap / warp-ramp axis)."""
    if kernel in ("bias_grad", "dyt"):
        return shape[1]  # N
    if kernel == "group_norm":
        _, c, s, _ = shape
        return c * s
    if kernel == "instance_norm":
        _, c, s = shape
        return c * s
    raise KeyError(kernel)


def collapse(kernel: str, shape: tuple) -> int:
    """The reduced/grid (M-like) extent."""
    return shape[0]  # M (bias/dyt), N (group), B (instance)


def total_elems(kernel: str, shape: tuple) -> int:
    return collapse(kernel, shape) * feature(kernel, shape)


# =========================================================================== #
#  SHAPES  (train / val / test  +  optional robustness)
# =========================================================================== #

SHAPES = {

    # ---- bias_grad : grad_bias = sum_M grad_out -> [N]. F = N. -----------------
    #   M = tokens (collapsed), N = hidden dim. Large M favors the split/occupancy.
    "bias_grad": {
        "train": [
            (16384, 1024), (16384, 1536), (16384, 2048),              # b0  (N<=2048)
            (8192, 2560), (8192, 3072), (8192, 4096),                 # b1  (<=4096)
            (8192, 5120), (8192, 6144), (8192, 8192),                 # b2  (<=8192)
            (4096, 12288), (4096, 16384),                             # b3  (<=16384)
            (32768, 2048), (65536, 1024), (16384, 4096),              # M-variation (occupancy)
        ],
        "val": [
            (8192, 1280), (16384, 2560), (8192, 3584), (4096, 7168),
            (8192, 11008), (32768, 1024), (16384, 8192),
        ],
        "test": [
            (32768, 1536), (16384, 3072), (4096, 5120), (8192, 2048),
            (4096, 10240), (16384, 6144), (8192, 14336),
        ],
        "robustness": [
            (1, 4096), (128, 8192), (262144, 256), (8192, 2047),
        ],
    },

    # ---- dyt (Dynamic Tanh) : grad_w/grad_b collapse M; grad_x elementwise. F=N -
    "dyt": {
        "train": [
            (16384, 1024), (16384, 1536), (8192, 2048),               # b0
            (8192, 2560), (8192, 3072), (8192, 4096),                 # b1
            (8192, 5120), (4096, 6144), (8192, 8192),                 # b2
            (4096, 11008), (4096, 16384),                             # b3
            (32768, 1536), (65536, 1024), (16384, 4096),              # M-variation
        ],
        "val": [
            (8192, 1280), (16384, 2048), (8192, 3584), (4096, 7168),
            (4096, 12288), (32768, 2048), (16384, 8192),
        ],
        "test": [
            (32768, 1024), (16384, 3072), (8192, 7168), (4096, 2560),
            (4096, 10240), (8192, 6144), (8192, 14336),
        ],
        "robustness": [
            (1, 4096), (128, 8192), (262144, 256), (8192, 2047),
        ],
    },

    # ---- group_norm : (N, C, S, G), G=32. F = C*S (resident [inner, C, S]). -----
    #   Vision / diffusion-UNet GroupNorm. N = batch (collapsed over grid + S).
    "group_norm": {
        "train": [
            (512, 128, 64, 32), (1024, 64, 128, 32),                  # b0  (F<=8192)
            (256, 64, 256, 32), (256, 128, 128, 32), (128, 128, 256, 32),  # b1  (<=32768)
            (128, 256, 256, 32), (64, 512, 128, 32), (32, 128, 1024, 32),  # b2  (<=131072)
            (32, 256, 512, 32),                                       # b2
            (16, 256, 1024, 32), (16, 512, 512, 32), (16, 512, 1024, 32),  # b3  (huge)
            (8, 256, 4096, 32), (64, 256, 256, 32),                   # b3 / M-variation
        ],
        "val": [
            (512, 64, 128, 32), (256, 128, 256, 32),
            (96, 256, 256, 32), (48, 256, 512, 32),
            (24, 256, 1024, 32), (12, 512, 1024, 32),
        ],
        "test": [
            (768, 128, 64, 32), (192, 128, 256, 32),
            (96, 512, 128, 32), (48, 128, 1024, 32),
            (24, 512, 512, 32), (8, 512, 1024, 32),
        ],
        "robustness": [
            (8, 64, 64, 32), (256, 96, 49, 32), (16, 320, 256, 32),
        ],
    },

    # ---- instance_norm : (B, C, S). F = C*S (resident [inner, C, S]). -----------
    #   InstanceNorm over spatial S per (b, c); weight/bias per channel [C].
    "instance_norm": {
        "train": [
            (512, 64, 128), (1024, 32, 256),                         # b0  (F<=8192)
            (256, 64, 256), (256, 128, 128), (128, 256, 128),        # b1  (<=32768)
            (128, 128, 512), (64, 256, 256), (32, 128, 1024),        # b2  (<=131072)
            (32, 256, 512),                                          # b2
            (16, 256, 1024), (16, 512, 512), (16, 512, 1024),        # b3  (huge)
            (8, 256, 4096), (256, 256, 256),                         # b3 / M-variation
        ],
        "val": [
            (512, 32, 256), (256, 128, 256),
            (96, 256, 256), (48, 256, 512),
            (24, 256, 1024), (12, 512, 1024),
        ],
        "test": [
            (768, 64, 128), (192, 256, 128),
            (96, 128, 512), (48, 128, 1024),
            (24, 512, 512), (8, 512, 1024),
        ],
        "robustness": [
            (8, 64, 64), (256, 96, 49), (16, 320, 256),
        ],
    },
}


# =========================================================================== #
#  VALIDATOR
# =========================================================================== #

def _est_us(kernel: str, shape: tuple) -> float:
    return total_elems(kernel, shape) * 4 * META[kernel]["traffic"] / HBM_BYTES_PER_S * 1e6


def _band(f: int, edges: list[int]) -> int:
    for i, e in enumerate(edges):
        if f <= e:
            return i
    return len(edges)


def validate() -> int:
    from itertools import combinations
    NOISE_US = 20.0
    problems = 0
    print(f"{'kernel':14} {'split':10} {'n':>2} {'F-range':>16} {'min_est_us':>11}")
    for k, splits in SHAPES.items():
        meta = META[k]
        edges = meta["bands"]
        arity = meta["arity"]
        train = splits["train"]
        train_bands = {_band(feature(k, s), edges) for s in train}
        train_f = [feature(k, s) for s in train]
        meas = ["train", "val", "test"]
        check = ["val", "test"]
        all_splits = meas + (["robustness"] if "robustness" in splits else [])

        for sp in all_splits:
            s = splits[sp]
            fr = (min(feature(k, x) for x in s), max(feature(k, x) for x in s))
            mn = min(_est_us(k, x) for x in s)
            print(f"{k:14} {sp:10} {len(s):>2} {str(fr):>16} {mn:>11.1f}")

        # 0. arity + (group_norm) C%G sanity
        for sp in all_splits:
            for x in splits[sp]:
                if len(x) != arity:
                    print(f"  !! {k}: {sp} {x} arity {len(x)} != {arity}"); problems += 1
                if k == "group_norm" and len(x) == 4 and x[1] % x[3] != 0:
                    print(f"  !! {k}: {sp} {x} C={x[1]} not divisible by G={x[3]}"); problems += 1

        # 1. pairwise disjoint among measurable splits
        for a, b in combinations(meas, 2):
            ov = set(map(tuple, splits[a])) & set(map(tuple, splits[b]))
            if ov:
                print(f"  !! {k}: {a}&{b} OVERLAP {sorted(ov)}"); problems += 1
        if "robustness" in splits:
            allmeas = {tuple(x) for sp in meas for x in splits[sp]}
            ovr = {tuple(x) for x in splits["robustness"]} & allmeas
            if ovr:
                print(f"  !! {k}: robustness overlaps measurable {sorted(ovr)}"); problems += 1

        # 2. train covers every F-band val/test probe
        for sp in check:
            for x in splits[sp]:
                if _band(feature(k, x), edges) not in train_bands:
                    print(f"  !! {k}: {sp} {x} F={feature(k,x)} band not in train"); problems += 1
        # 3. F envelope: val/test F within train F range
        for sp in check:
            for x in splits[sp]:
                if not (min(train_f) <= feature(k, x) <= max(train_f)):
                    print(f"  !! {k}: {sp} {x} F={feature(k,x)} OUTSIDE train envelope "
                          f"[{min(train_f)},{max(train_f)}]"); problems += 1
        # 4. measurable splits clear the noise floor
        for sp in meas:
            for x in splits[sp]:
                if _est_us(k, x) < NOISE_US:
                    print(f"  !! {k}: {sp} {x} est {_est_us(k,x):.1f}us < {NOISE_US}us NOISE FLOOR")
                    problems += 1
        # 5. balance
        nt, nv, nte = len(train), len(splits["val"]), len(splits["test"])
        if not (12 <= nt <= 16): print(f"  ?? {k}: train n={nt} (want 12-16)")
        if not (6 <= nv <= 9):   print(f"  ?? {k}: val n={nv} (want 6-9)")
        if not (6 <= nte <= 9):  print(f"  ?? {k}: test n={nte} (want 6-9)")
        print()

    tot = sum(len(s) for v in SHAPES.values() for s in v.values())
    print(f"\n{'PASS' if problems == 0 else 'FAIL'}: {problems} problem(s).  "
          f"kernels={len(SHAPES)} all-bucket shapes={tot}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if validate() else 0)
