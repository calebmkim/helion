"""RESULTS-REFEREE (EDIT#5) — tighter re-bench of the HIGH-SPREAD / LOW-MARGIN jsd shapes.

The full-flip-set A/B showed 2048 faster on every jsd shape, but several wide-V/large-M
shapes had 17-24% do_bench spread, which could mask a small regression at the ~2-3% margin
shapes. This re-bench targets those shapes with:
  - N=11 samples per arm (more statistics)
  - INTERLEAVED arms (new, old, new, old, ...) per round so transient thermal/clock drift
    hits both arms equally
  - report BOTH median-of-medians AND MIN-of-medians (min = the cleanest least-perturbed
    estimate; if 2048's min < 4096's min, 2048 is genuinely at-least-as-fast).
A regression would be: 2048's best-case (min) is SLOWER than 4096's best-case.

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/referee_edit5_highspread_recheck.py
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

from run2_measure_g import KERNELS, check_correct, get_seed  # noqa: E402
from triton.testing import do_bench  # noqa: E402

N_ROUNDS = 11
SEED = 0
LOG = os.path.join(os.path.dirname(_HARNESS_DIR), "logs", "run3",
                   "referee_edit5_highspread_recheck.json")

# the high-spread / low-margin shapes from the full-flip-set A/B + the large-M risk shape.
SHAPES = [
    (8192, 50257),    # 24% spread in first pass
    (16384, 32000),   # large-M (the EDIT#4-style risk), 22% spread
    (8192, 128256),   # wide-V, 20% spread
    (4096, 128256),   # wide-V, 18% spread
    (4096, 151936),   # wide-V, 2.9% margin
    (2048, 256000),   # widest-V, 2.6% margin (lowest)
]


def _np2(x: int) -> int:
    p = 1
    while p < x:
        p <<= 1
    return p


def mk(seed_cfg, ridx, rblock):
    bs = list(seed_cfg["block_sizes"])
    bs[ridx] = rblock
    c = dict(seed_cfg)
    c["block_sizes"] = bs
    return c


def build(fn_obj, args, ref, out_extract, cfg):
    k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
    b = k.bind(args)
    b.ensure_config_exists(args)
    ok, err = check_correct(out_extract(b(*args)), ref)
    return b, ok, err


def main():
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} N_ROUNDS={N_ROUNDS} SEED={SEED} (interleaved new/old)\n", flush=True)
    fn_obj, builder, _ = KERNELS["jsd"]
    out = []
    for M, N in SHAPES:
        torch.manual_seed(SEED)
        args, ref, out_extract = builder(M, N)
        seed_cfg, bound = get_seed(fn_obj, args)
        fact = bound.env.config_spec.reduction_facts[0]
        ridx = bound.env.config_spec.block_sizes.block_id_to_index(fact.block_id)
        extent = _np2(N)
        new_r, old_r = min(2048, extent), min(4096, extent)
        b_new, ok_n, err_n = build(fn_obj, args, ref, out_extract, mk(seed_cfg, ridx, new_r))
        b_old, ok_o, err_o = build(fn_obj, args, ref, out_extract, mk(seed_cfg, ridx, old_r))
        assert ok_n and ok_o, f"accuracy FAIL jsd{(M,N)} new={err_n} old={err_o}"

        new_s, old_s = [], []
        torch.cuda.synchronize()
        for _ in range(N_ROUNDS):
            # interleave: time new then old in the SAME round (serial; one at a time)
            new_s.append(float(do_bench(lambda: b_new(*args), return_mode="median")))
            old_s.append(float(do_bench(lambda: b_old(*args), return_mode="median")))
        new_s.sort(); old_s.sort()
        n_med, n_min = new_s[len(new_s)//2], new_s[0]
        o_med, o_min = old_s[len(old_s)//2], old_s[0]
        sp_med = o_med / n_med
        sp_min = o_min / n_min
        rec = {
            "shape": [M, N], "new_r": new_r, "old_r": old_r,
            "new_med_us": n_med*1000, "new_min_us": n_min*1000,
            "old_med_us": o_med*1000, "old_min_us": o_min*1000,
            "speedup_med": sp_med, "speedup_min": sp_min,
            "new_spread": (new_s[-1]-new_s[0])/n_med,
            "old_spread": (old_s[-1]-old_s[0])/o_med,
        }
        out.append(rec)
        verdict = "REGRESS" if (sp_min < 0.99 or sp_med < 0.99) else \
                  ("WIN" if sp_min > 1.01 else "tie")
        print(f"jsd({M:>6},{N:>7}) | new2048 med={n_med*1000:8.1f} min={n_min*1000:8.1f}us "
              f"| old4096 med={o_med*1000:8.1f} min={o_min*1000:8.1f}us "
              f"| old/new med={sp_med:.3f} min={sp_min:.3f}  {verdict}", flush=True)

    print()
    worst_med = min(r["speedup_med"] for r in out)
    worst_min = min(r["speedup_min"] for r in out)
    no_regress = all(r["speedup_min"] >= 0.99 and r["speedup_med"] >= 0.99 for r in out)
    print(f"worst old/new (median)={worst_med:.3f}  worst old/new (min)={worst_min:.3f}")
    print(f"NO-REGRESSION (all median>=0.99 AND all min>=0.99): {no_regress}", flush=True)

    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w") as f:
        json.dump({"gpu": gpu, "n_rounds": N_ROUNDS, "results": out,
                   "worst_speedup_med": worst_med, "worst_speedup_min": worst_min,
                   "no_regression": no_regress}, f, indent=2)
    print(f"[wrote {LOG}]", flush=True)


if __name__ == "__main__":
    main()
