"""RUN-3 EDIT#5 — full 4-split FLIP-SET verification (compile-only, NO autotune, NO do_bench).

EDIT#5 = the Band-B R_BLOCK footprint cap divides ALSO by num_reduction_ops:
    bandb_cap = BANDB_R_BLOCK_BYTES // (itemsize * max(1, num_reduction_ops))
jsd (n_acc=2) -> R_BLOCK 4096->2048; kl_div (n_acc=1) -> 4096 unchanged (byte-identical).

This emits the ACTUAL seed (working tree = post-EDIT#5) for EVERY jsd + kl_div shape across
ALL FOUR splits (train/val/test/robustness — the EDIT#2 lesson: scan all splits, widest rows
live in robustness), PLUS a T1 (sum) and a non-Band-B T2 (softmax) contrast. For each shape it
reads the EMITTED R_BLOCK (block_sizes[reduction_index]) and INDEPENDENTLY recomputes the OLD
(itemsize-only) vs NEW (itemsize*nro) cap to label FLIP vs BYTE-IDENTICAL. The post-edit emitted
R_BLOCK must equal the NEW cap everywhere; jsd must flip 4096->2048; kl_div/sum/softmax identical.

Pure bind() + seed-emit — NO timing, does NOT enter the GPU timing queue.

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_edit5_flipset.py
"""

from __future__ import annotations

import json
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)
_PROMPTS_DIR = os.path.join(os.path.dirname(_HARNESS_DIR), "prompts")
if _PROMPTS_DIR not in sys.path:
    sys.path.insert(0, _PROMPTS_DIR)

import helion  # noqa: E402

import shapes_v3_draft as S  # noqa: E402
from run2_measure_g import KERNELS, get_seed  # noqa: E402

BANDB_R_BLOCK_BYTES = 16384  # the constant the cap budgets against (triton.py)


def _np2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _reduction_block_size(seed_cfg: dict, fact) -> int | None:
    """The emitted R_BLOCK = block_sizes entry at the reduction axis index (T2)."""
    bs = seed_cfg.get("block_sizes")
    if bs is None:
        return None
    # T2 Band-B: reduction axis is a block_sizes entry. Find its index generically
    # via the bound spec (block_id -> index); but here we only have the cfg, so we
    # locate it by the fact's block_id resolved against the bound spec in probe().
    return bs  # caller resolves the index


def probe(kernel: str, M: int, N: int, split: str):
    fn, builder, _ = KERNELS[kernel]
    args, _, _ = builder(M, N)
    bound = fn.bind(args)
    spec = bound.env.config_spec
    facts = spec.reduction_facts
    if not facts:
        return {"kernel": kernel, "M": M, "N": N, "split": split,
                "note": "NO reduction_facts"}
    fact = facts[0]
    seed_cfg, _ = get_seed(fn, args)
    nro = fact.num_reduction_ops
    n_tiled = fact.num_tiled_accumulators
    itemsize = fact.itemsize

    # Independently recompute OLD vs NEW cap (the flip predictor).
    old_cap = _np2(max(1, BANDB_R_BLOCK_BYTES // max(1, itemsize)))
    new_cap = _np2(max(1, BANDB_R_BLOCK_BYTES // (max(1, itemsize) * max(1, nro))))

    # The EMITTED R_BLOCK: for a Band-B T2 the reduction axis is a block_sizes entry.
    emitted_rblock = None
    bs = seed_cfg.get("block_sizes")
    is_t2 = bs is not None
    if is_t2:
        try:
            red_idx = spec.block_sizes.block_id_to_index(fact.block_id)
            emitted_rblock = bs[red_idx]
        except Exception:  # noqa: BLE001
            emitted_rblock = None

    # For Band-B (n_tiled>=1) the emitted R_BLOCK is min(extent, cap); for narrow V
    # the extent (np2(N)) may be < cap so neither old nor new fires (persistent).
    extent_np2 = _np2(N)
    old_rblock_expected = min(extent_np2, old_cap) if n_tiled >= 1 else emitted_rblock
    new_rblock_expected = min(extent_np2, new_cap) if n_tiled >= 1 else emitted_rblock

    flip = (n_tiled >= 1) and (old_rblock_expected != new_rblock_expected)
    # Consistency: emitted must equal NEW expected (the working tree has EDIT#5).
    consistent = (emitted_rblock == new_rblock_expected) if is_t2 and n_tiled >= 1 \
        else True

    return {
        "kernel": kernel, "M": M, "N": N, "split": split,
        "n_tiled": n_tiled, "nro": nro, "itemsize": itemsize,
        "extent_np2": extent_np2,
        "old_cap": old_cap, "new_cap": new_cap,
        "old_rblock": old_rblock_expected, "new_rblock": new_rblock_expected,
        "emitted_rblock": emitted_rblock,
        "FLIP": flip, "consistent": consistent,
        "num_warps": seed_cfg.get("num_warps"),
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} helion={helion.__file__} (bind-only, no do_bench)\n", flush=True)
    results = []
    cases: list[tuple[str, int, int, str]] = []
    # jsd + kl_div: ALL 4 splits (the firing kernels + their byte-identical peer).
    for kernel in ("jsd", "kl_div"):
        for split in ("train", "val", "test", "robustness"):
            for (M, N) in S.SHAPES[kernel][split]:
                cases.append((kernel, M, N, split))
    # Contrast: sum (T1, n_tiled=0 -> never reaches the Band-B branch) + softmax
    # (T2 non-Band-B, n_tiled=0) — must be untouched / structurally excluded.
    for (M, N) in S.SHAPES["sum"]["train"][:3]:
        cases.append(("sum", M, N, "train"))
    for (M, N) in S.SHAPES["softmax"]["train"][:3]:
        cases.append(("softmax", M, N, "train"))

    n_flip = n_ident = n_inconsistent = 0
    flips = []
    for kernel, M, N, split in cases:
        try:
            r = probe(kernel, M, N, split)
        except Exception as e:  # noqa: BLE001
            r = {"kernel": kernel, "M": M, "N": N, "split": split,
                 "ERR": f"{type(e).__name__}: {e}"[:150]}
        results.append(r)
        if "ERR" in r:
            print(f"[ERR] {kernel}({M},{N})[{split}]: {r['ERR']}", flush=True)
            continue
        tag = "FLIP 4096->2048" if r.get("FLIP") else "byte-id"
        if r.get("FLIP"):
            n_flip += 1
            flips.append(f"{kernel}({M},{N})[{split}]")
        else:
            n_ident += 1
        if not r.get("consistent", True):
            n_inconsistent += 1
            tag += "  *** INCONSISTENT (emitted != new_cap) ***"
        print(
            f"{r['kernel']:>9}({r['M']:>6},{r['N']:>7})[{r['split']:>10}] "
            f"n_tiled={r.get('n_tiled')} nro={r.get('nro')} "
            f"old_R={r.get('old_rblock')} new_R={r.get('new_rblock')} "
            f"emitted_R={r.get('emitted_rblock')} w={r.get('num_warps')}  -> {tag}",
            flush=True,
        )

    print(f"\nSUMMARY: {n_flip} FLIP, {n_ident} byte-identical, "
          f"{n_inconsistent} INCONSISTENT (must be 0).", flush=True)
    print(f"FLIPS ({n_flip}): {flips}", flush=True)
    # Assertions for the gate receipt.
    jsd_flips = [f for f in flips if f.startswith("jsd")]
    kl_flips = [f for f in flips if f.startswith("kl_div")]
    print(f"\njsd flips: {len(jsd_flips)} (expect: all jsd Band-B shapes with "
          f"np2(N) > 2048)", flush=True)
    print(f"kl_div flips: {len(kl_flips)} (expect 0 — nro=1, byte-identical)",
          flush=True)
    assert n_inconsistent == 0, "emitted R_BLOCK != NEW cap somewhere — edit not live!"
    assert len(kl_flips) == 0, "kl_div FLIPPED — should be byte-identical (nro=1)!"

    out = os.path.join(os.path.dirname(_HARNESS_DIR), "logs", "run3",
                       "edit5_flipset.json")
    with open(out, "w") as f:
        json.dump({"results": results, "n_flip": n_flip, "n_ident": n_ident,
                   "flips": flips}, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
