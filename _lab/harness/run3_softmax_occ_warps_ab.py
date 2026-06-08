"""RUN-3 #15 — softmax small-N OCCUPANCY-vs-warps A/B (NO autotune).

The run-2 OVERFITTING TRAP: softmax small-N warps is NOT a clean rnumel rule. Prior A/B
(notebook ~647): softmax(131072,256)->w8, (262144,128)->w8/16, BUT (16384,512)->w4 (w16
CATASTROPHIC 0.514). The rnumel ramp (rnumel<=1024->w4) can't separate them. Hypothesis:
the lever is GRID OCCUPANCY (grid_rows / num_sm), the same quantity EDIT-PID already
computes (triton.py:680-692). But occupancy is CONFOUNDED with rnumel in the prior 3
shapes (the w4 shape also has 2x the rnumel). This sweep DISAMBIGUATES by varying
grid_rows/SM and rnumel INDEPENDENTLY.

H100 = 132 SMs. softmax is T2 persistent at these N (256/512 fp32 << persist cap), grid =
grid_rows (one program per row, M_BLOCK=1 expected — confirm in output).

Shape grid (the 2 DISCRIMINATORS are starred):
  hiOcc_tinyN   (131072,256)  occ~993  rnumel256  [known: wants w8]
  hiOcc_tinierN (262144,128)  occ~1986 rnumel128  [known: wants w8/16]
  loOcc_smallN  (16384,512)   occ~124  rnumel512  [known: wants w4; w16 catastrophic]
  *hiOcc_smallN  (131072,512) occ~993  rnumel512  -> w8? occupancy governs : w4? rnumel governs
  *loOcc_tinyN   (4096,256)   occ~31   rnumel256  -> w4? occupancy governs : w8? rnumel governs
  midOcc_tinyN  (32768,256)   occ~248  rnumel256  [find the contour between 124 and 993]

Reads seed/arm for warps {4,8,16}. CONCLUSION test: if hiOcc_smallN wants w8 AND
loOcc_tinyN wants w4 -> grid_rows//num_sm is the lever (NOT rnumel) -> a principled
occupancy fact. If hiOcc_smallN wants w4 (rnumel governs) -> the lever is rnumel-AND-M,
murkier. Reports the per-cell best warp + occ + rnumel so the separating contour is visible.

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_softmax_occ_warps_ab.py
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
    N_RUNS,
    get_seed,
    check_correct,
    codegen_kind,
)
from triton.testing import do_bench  # noqa: E402

LOG_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "logs", "run3"))

# (label, M, N)
CASES = [
    ("hiOcc_tinyN", 131072, 256),
    ("hiOcc_tinierN", 262144, 128),
    ("loOcc_smallN", 16384, 512),
    ("hiOcc_smallN", 131072, 512),   # DISCRIMINATOR
    ("loOcc_tinyN", 4096, 256),      # DISCRIMINATOR
    ("midOcc_tinyN", 32768, 256),
]
WARPS = [4, 8, 16]


def _bench(fn, n=N_RUNS):
    torch.cuda.synchronize()
    s = sorted(float(do_bench(fn, return_mode="median")) for _ in range(n))
    med = s[len(s) // 2]
    return med, ((s[-1] - s[0]) / med if med else None)


def _occ_and_mblock(fn_obj, args, seed):
    """grid_rows = prod(size_hint(m_block_ids)); occ = grid_rows / num_sm. Also report
    the seed's M_BLOCK (the non-reduction block_sizes) to confirm grid == grid_rows."""
    from helion.runtime import get_num_sm
    k = helion.kernel(fn_obj.fn)
    b = k.bind(args)
    env = b.env
    spec = env.config_spec
    f = spec.reduction_facts[0]
    ridx = spec.block_sizes.block_id_to_index(f.block_id)
    grid_rows = 1
    for mbid in f.m_block_ids:
        grid_rows *= env.size_hint(env.block_sizes[mbid].size)
    num_sm = max(1, get_num_sm(env.device))
    seed_bs = list(seed["block_sizes"])
    m_blocks = [seed_bs[i] for i in range(len(seed_bs)) if i != ridx]
    return grid_rows, num_sm, grid_rows / num_sm, ridx, m_blocks


def _run(fn_obj, args, ref, out_extract, cfg):
    try:
        k = helion.kernel(fn_obj.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        ok, err = check_correct(out_extract(b(*args)), ref)
        med, sp = (_bench(lambda: b(*args)) if ok else (None, None))
        return {"us": med * 1000 if med else None, "spread": sp, "correct": ok,
                "codegen": codegen_kind(b), "w": dict(b._config).get("num_warps")}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:160]}


def run_case(label, M, N):
    fn, builder, tc_ref = KERNELS["softmax"]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"softmax{(M, N)} non-fp32"
    seed = dict(get_seed(fn, args)[0])
    grid_rows, num_sm, occ, ridx, m_blocks = _occ_and_mblock(fn, args, seed)
    seed_w = seed.get("num_warps")

    arms = {f"w{w}": {**seed, "num_warps": w} for w in WARPS}

    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    assert check_correct(out_extract(tc(args)), ref)[0]
    tc_med, _ = _bench(lambda: tc(args))

    res = {n: _run(fn, args, ref, out_extract, c) for n, c in arms.items()}
    # seed/arm uses the seed-warp arm as the reference baseline
    ref_arm = res.get(f"w{seed_w}", {})
    base = ref_arm.get("us")
    best = min((r["us"] for r in res.values() if r.get("us")), default=None)
    best_w = next((nm for nm, r in res.items() if r.get("us") == best), "?")
    print(f"\n=== softmax({M},{N}) [{label}] === tc={tc_med*1000:.1f}us  "
          f"grid_rows={grid_rows} num_sm={num_sm} OCC={occ:.0f} rnumel={N}  "
          f"M_BLOCK={m_blocks} seed_w={seed_w}  BEST={best_w}", flush=True)
    for w in WARPS:
        r = res[f"w{w}"]
        if r.get("us"):
            sa = base / r["us"] if base else float("nan")
            star = " <-BEST" if r["us"] == best else ""
            print(f"  w{w:<3} {r['us']:>9.1f}us  seedw/this={sa:>6.3f}  "
                  f"this/tc={tc_med*1000/r['us']:.3f}  {'OK' if r['correct'] else 'BAD'}"
                  f"{star}", flush=True)
        else:
            print(f"  w{w:<3} -> {r}", flush=True)
    return {"label": label, "shape": [M, N], "grid_rows": grid_rows, "num_sm": num_sm,
            "occ": occ, "rnumel": N, "m_block": m_blocks, "seed_w": seed_w,
            "best_w": best_w, "tc_us": tc_med * 1000, "arms": res}


def main():
    print(f"GPU={os.environ.get('CUDA_VISIBLE_DEVICES','?')} helion={helion.__file__}",
          flush=True)
    out = []
    for label, M, N in CASES:
        try:
            out.append(run_case(label, M, N))
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {label} softmax({M},{N}): {type(e).__name__}: {e}"[:200],
                  flush=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "softmax_occ_warps_ab.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[wrote {os.path.join(LOG_DIR, 'softmax_occ_warps_ab.json')}]", flush=True)
    # CONCLUSION helper: print the occ x best-w table sorted by occ
    print("\n--- OCC vs BEST-WARP (sorted by occupancy) ---", flush=True)
    for r in sorted(out, key=lambda x: x["occ"]):
        print(f"  occ={r['occ']:>6.0f}  rnumel={r['rnumel']:>4}  "
              f"best={r['best_w']:<3}  [{r['label']}]", flush=True)


if __name__ == "__main__":
    main()
