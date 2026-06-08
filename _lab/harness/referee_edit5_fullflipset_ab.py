"""RESULTS-REFEREE (EDIT#5, d2ff878a) — INDEPENDENT full-flip-set no-regression A/B.

The EDIT#4 lesson: a config-flipping cap's no-regression A/B must cover the FULL
flip-set (every shape it moves) at MID *and* EXTREME M — the worker's 3-shape A/B
missed a pathological large-M valley. So this referee command sweeps the FULL jsd
flip-set the cap moves R_BLOCK 4096->2048 on: narrow-V, wide-V, AND large-M, with the
hub-named shapes explicitly included:
    (16384,32000) large-M ; (4096,128256) (4096,151936) (8192,128256) wide-V.

For EACH jsd shape: build the LIVE seed, then bench TWO arms that differ ONLY in
R_BLOCK (block_sizes[reduction_index]):
    new = R_BLOCK=2048  (what EDIT#5 emits)
    old = R_BLOCK=4096  (the pre-EDIT#5 seed)
All other config fields (num_warps, M_BLOCK, stages, indexing, ...) held EQUAL — this
isolates the cap's effect. Accuracy ON: each arm checked vs the operator fp32 ref
(rtol=1e-3, atol=1e-4) BEFORE timing (configs=[seed] bypasses the autotuner's own
accuracy check, so we run our own). Timing = median-of-N independent do_bench(median)
samples, N>=5, spread reported. SERIAL (one do_bench at a time).

Accept rule (per shape): new(2048) is NOT slower than old(4096) beyond noise, i.e.
    speedup = old_us / new_us >= 1 - NOISE  (NOISE band = max observed spread, ~1%).
A 2048-slower-than-4096 on ANY jsd shape (speedup << 1) = a regression = FAIL.

kl_div: assert the live seed R_BLOCK == 4096 (byte-identical, nro=1) at narrow+wide+
large-M; emit the config and confirm new==old (the cap is a no-op).

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/referee_edit5_fullflipset_ab.py
"""

from __future__ import annotations

import json
import os
import sys

import torch

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import helion  # noqa: E402

from run2_measure_g import (  # noqa: E402
    KERNELS,
    check_correct,
    codegen_kind,
    get_seed,
)
from triton.testing import do_bench  # noqa: E402

N_RUNS = 7  # median-of-7 independent do_bench(median) samples per arm
SEED = 0  # fixed RNG seed for reproducible inputs
LOG = os.path.join(os.path.dirname(_HARNESS_DIR), "logs", "run3",
                   "referee_edit5_fullflipset_ab.json")

# FULL jsd flip-set the cap moves 4096->2048, spanning narrow/mid/wide V and
# small..LARGE M. The hub-named must-cover shapes are marked.
JSD_SHAPES = [
    (8192, 30522),    # narrow-V headline (the 1.213 gap)
    (8192, 32000),    # narrow-V headline
    (8192, 50257),    # mid-V
    (8192, 65536),    # mid-V
    (4096, 98304),    # wide-V, mid-M
    (8192, 128256),   # *** HUB-NAMED wide-V ***
    (4096, 128256),   # *** HUB-NAMED wide-V ***
    (4096, 151936),   # *** HUB-NAMED wide-V ***
    (2048, 256000),   # widest-V (the 1.034 wide gap)
    (16384, 32000),   # *** HUB-NAMED large-M (the EDIT#4-style valley risk) ***
]

# kl_div: must stay R_BLOCK=4096 (byte-identical). narrow + wide + large-M.
KL_SHAPES = [
    (8192, 30522),
    (8192, 128256),
    (4096, 151936),
    (16384, 32000),   # large-M sanity (peer of the jsd large-M shape)
    (1024, 256000),
]


def _np2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def _reduction_index(bound, fact):
    return bound.env.config_spec.block_sizes.block_id_to_index(fact.block_id)


def _bench(fn, n=N_RUNS):
    """median-of-n independent do_bench(median) samples, in ms; + relative spread."""
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    spread = (s[-1] - s[0]) / med if med else float("nan")
    return med, spread, s


def _build_and_check(fn_obj, args, ref, out_extract, cfg):
    """Build kernel with explicit config, run once, check accuracy vs fp32 ref."""
    k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    ok, maxerr = check_correct(out_extract(b(*args)), ref)
    return b, ok, maxerr


def run_jsd(M, N):
    fn_obj, builder, _ = KERNELS["jsd"]
    torch.manual_seed(SEED)
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"jsd{(M, N)} non-fp32 {a.dtype}"

    seed_cfg, bound = get_seed(fn_obj, args)
    fact = bound.env.config_spec.reduction_facts[0]
    ridx = _reduction_index(bound, fact)
    seed_bs = list(seed_cfg["block_sizes"])
    emitted_r = seed_bs[ridx]

    extent = _np2(N)
    new_r = min(2048, extent)  # EDIT#5 cap
    old_r = min(4096, extent)  # pre-EDIT#5 cap

    def mk(rblock):
        bs = list(seed_bs)
        bs[ridx] = rblock
        c = dict(seed_cfg)
        c["block_sizes"] = bs
        return c

    arms = {}
    # NEW arm (what the live heuristic emits)
    b_new, ok_new, err_new = _build_and_check(fn_obj, args, ref, out_extract, mk(new_r))
    cg_new = codegen_kind(b_new)
    # OLD arm (pre-edit seed)
    b_old, ok_old, err_old = _build_and_check(fn_obj, args, ref, out_extract, mk(old_r))
    cg_old = codegen_kind(b_old)

    # Time SERIALLY: finish one do_bench loop before starting the next.
    new_med, new_sp, new_s = (_bench(lambda: b_new(*args)) if ok_new
                              else (None, None, None))
    old_med, old_sp, old_s = (_bench(lambda: b_old(*args)) if ok_old
                              else (None, None, None))

    speedup = (old_med / new_med) if (new_med and old_med) else None
    arms = {
        "new_2048": {"r_block": new_r, "us": new_med * 1000 if new_med else None,
                     "spread": new_sp, "correct": ok_new, "maxerr": err_new,
                     "codegen": cg_new, "samples_ms": new_s},
        "old_4096": {"r_block": old_r, "us": old_med * 1000 if old_med else None,
                     "spread": old_sp, "correct": ok_old, "maxerr": err_old,
                     "codegen": cg_old, "samples_ms": old_s},
    }
    return {
        "kernel": "jsd", "shape": [M, N], "extent_np2": extent,
        "emitted_R": emitted_r, "num_reduction_ops": fact.num_reduction_ops,
        "num_tiled_accumulators": fact.num_tiled_accumulators,
        "num_warps": seed_cfg.get("num_warps"),
        "speedup_old_over_new": speedup, "arms": arms,
    }


def run_kl(M, N):
    fn_obj, builder, _ = KERNELS["kl_div"]
    torch.manual_seed(SEED)
    args, ref, out_extract = builder(M, N)
    seed_cfg, bound = get_seed(fn_obj, args)
    fact = bound.env.config_spec.reduction_facts[0]
    ridx = _reduction_index(bound, fact)
    emitted_r = list(seed_cfg["block_sizes"])[ridx]
    # confirm the live seed is R_BLOCK 4096 (or the persistent extent if narrower)
    expect_r = min(4096, _np2(N))
    b, ok, err = _build_and_check(fn_obj, args, ref, out_extract, dict(seed_cfg))
    return {
        "kernel": "kl_div", "shape": [M, N], "extent_np2": _np2(N),
        "emitted_R": emitted_r, "expect_R_4096cap": expect_r,
        "byte_identical_4096": emitted_r == expect_r,
        "num_reduction_ops": fact.num_reduction_ops,
        "correct": ok, "maxerr": err, "num_warps": seed_cfg.get("num_warps"),
    }


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} helion={helion.__file__}", flush=True)
    print(f"N_RUNS={N_RUNS} (median-of-{N_RUNS} do_bench), SEED={SEED}\n", flush=True)

    out = {"gpu": gpu, "n_runs": N_RUNS, "seed": SEED, "jsd": [], "kl_div": []}

    print("=" * 92)
    print("JSD FULL FLIP-SET A/B  (new R_BLOCK=2048 vs old R_BLOCK=4096; all else equal)")
    print("=" * 92, flush=True)
    worst = (None, 9.9)
    for M, N in JSD_SHAPES:
        r = run_jsd(M, N)
        out["jsd"].append(r)
        a_new, a_old = r["arms"]["new_2048"], r["arms"]["old_4096"]
        sp = r["speedup_old_over_new"]
        if sp is not None and sp < worst[1]:
            worst = (r["shape"], sp)
        nm = (f"{a_new['us']:8.1f}us(sp{a_new['spread']*100:4.1f}%,{a_new['codegen'][:4]})"
              if a_new["us"] else f"  build/acc FAIL err={a_new['maxerr']}")
        om = (f"{a_old['us']:8.1f}us(sp{a_old['spread']*100:4.1f}%,{a_old['codegen'][:4]})"
              if a_old["us"] else f"  build/acc FAIL err={a_old['maxerr']}")
        flag = ""
        if sp is not None:
            flag = " <<< REGRESSION (2048 slower)" if sp < 0.985 else \
                   (" WIN" if sp > 1.015 else " tie")
        print(f"jsd({M:>6},{N:>7}) emR={r['emitted_R']} w{r['num_warps']} | "
              f"new2048 {nm} | old4096 {om} | "
              f"old/new={sp if sp is None else round(sp,3)}{flag}", flush=True)

    print("\n" + "=" * 92)
    print("KL_DIV BYTE-IDENTITY  (nro=1 -> R_BLOCK must stay 4096; cap is a no-op)")
    print("=" * 92, flush=True)
    kl_all_4096 = True
    for M, N in KL_SHAPES:
        r = run_kl(M, N)
        out["kl_div"].append(r)
        kl_all_4096 = kl_all_4096 and r["byte_identical_4096"] and r["correct"]
        print(f"kl_div({M:>6},{N:>7}) emR={r['emitted_R']} expect={r['expect_R_4096cap']} "
              f"nro={r['num_reduction_ops']} byte_id={r['byte_identical_4096']} "
              f"correct={r['correct']} maxerr={r['maxerr']:.2e}", flush=True)

    print("\n" + "=" * 92)
    print("VERDICT INPUTS")
    print("=" * 92, flush=True)
    print(f"jsd worst-case shape (lowest old/new): {worst[0]} @ old/new={round(worst[1],3)}",
          flush=True)
    jsd_ok = all(
        (r["arms"]["new_2048"]["correct"] and r["arms"]["old_4096"]["correct"]
         and r["speedup_old_over_new"] is not None
         and r["speedup_old_over_new"] >= 0.985)
        for r in out["jsd"]
    )
    print(f"jsd no-regression (all old/new >= 0.985 AND both arms correct): {jsd_ok}",
          flush=True)
    print(f"kl_div byte-identical@4096 + correct (all shapes): {kl_all_4096}", flush=True)
    out["verdict_inputs"] = {
        "jsd_worst_shape": worst[0], "jsd_worst_old_over_new": worst[1],
        "jsd_no_regression": jsd_ok, "kl_div_byte_identical_4096": kl_all_4096,
    }

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {LOG}]", flush=True)


if __name__ == "__main__":
    main()
