"""GATE A — independent reproduction (authored fresh, NOT copied from ab_flj.py).

Skeptic's own-script repro of the CLAIM: the body_live_tiles liveness ceiling flips
fused_linear_jsd narrow-V from persistent reduction_loops=[None] to looped [8192], turning a
sub-1.0x loss vs grad-fair torch.compile into a >=1.0x win.

This script attacks the two axes ab_flj.py leaves open:
  (1) ARM-EQUIVALENCE (footgun #6c): the tc baseline computes BOTH the loss [M] AND the
      grad [M,V] fp32 — exactly the two tensors jsd_kernel returns — so neither arm is doing
      less work. (A loss-ONLY baseline would bias AGAINST the seed; this is the fair version.)
  (2) SEED-IS-THE-[8192]-TIMED: we do NOT hand-pick the looped chunk. We ask the live heuristic
      for the ACTUAL emitted seed (compiler_seed_configs) and ASSERT its reduction_loops==[8192]
      on each narrow-V shape. The AFTER arm runs THAT emitted seed verbatim. The BEFORE arm is
      the same seed with reduction_loops forced to [None] (the persistent config the liveness
      ceiling is claimed to replace). So BEFORE/AFTER differ ONLY in reduction_loops.

Method (independent of ab_flj.py — written from harness primitives):
  * fixed inputs (one allocation, reused for BOTH arms + tc), accuracy gate ON (loss allclose),
  * single process, median-of-9 do_bench medians, working set 1-2.5 GB >> 50 MB L2 -> cold,
  * torch._dynamo.reset() before compiling tc so tc is not penalised by a stale dynamo cache,
  * same dtype on both arms; grad output is fp32 in BOTH the kernel and the tc baseline.

Run (driver, foreground-serial, ONE GPU):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-3stage \
    python /home/calebkim/helion-new-heuristics/helion-3stage/_lab/stage1_liveness/gate_A_repro.py

PASS criteria (the claim reproduces) iff, for every shape:
  - the emitted seed's reduction_loops == [8192]            (seed-is-the-timed-config)
  - acc==True for both [None] and [8192]                    (not an accuracy-fail mirage)
  - G_before (=[None]) < 1.0 and G_after (=[8192]) >= 1.0   (loss flips to a win)
  - G_after / G_before clears the ~5% noise band            (real move, not wobble)
"""

from __future__ import annotations

import os
import sys

import torch
from triton.testing import do_bench

import helion
from helion._compiler.autotuner_heuristics import compiler_seed_configs

_WT = "/home/calebkim/helion-new-heuristics/helion-3stage"
assert os.path.realpath(helion.__file__).startswith(_WT), helion.__file__
sys.path.insert(0, _WT)
from examples.fused_linear_jsd import jsd_kernel  # noqa: E402

N_MED = 9
BETA, TEMP, IGNORE = 0.5, 1.0, -100
# shapes: (M, V, dtype). narrow-V is the target (the seed must emit [8192]); the two wide-V
# rows are sanity controls — at (2048,128256) the seed already looped on base, so it is NOT
# required to be [8192] and is reported only.
NARROW = [
    (4096, 32000, torch.bfloat16),
    (4096, 50257, torch.bfloat16),
    (8192, 32000, torch.bfloat16),
    (4096, 32000, torch.float32),
    (4096, 50257, torch.float32),
]


def median_us(fn) -> float:
    torch.cuda.synchronize()
    samples = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_MED))
    return samples[N_MED // 2] * 1e3  # ms -> us


def grad_fair_tc(sl: torch.Tensor, tl: torch.Tensor):
    """ARM-FAIR baseline (footgun #6c): returns BOTH loss [M] AND grad [M,V] fp32 — the same
    two outputs jsd_kernel returns. fp32 math throughout, matching the kernel's .to(float).
    """

    def f():
        ss, ts = sl.float() / TEMP, tl.float() / TEMP
        sp = torch.softmax(ss, -1)
        tp = torch.softmax(ts, -1)
        slp = torch.log_softmax(ss, -1)
        tlp = torch.log_softmax(ts, -1)
        m = (1 - BETA) * sp + BETA * tp
        logm = torch.log(m)
        skl = (sp * (slp - logm)).sum(-1)
        tkl = (tp * (tlp - logm)).sum(-1)
        loss = (1 - BETA) * skl + BETA * tkl
        grad = ((1 - BETA) / TEMP) * (sp - m)
        return loss, grad

    return f


def run_one(m: int, v: int, dt: torch.dtype) -> dict:
    torch.manual_seed(0)
    sl = torch.randn(m, v, device="cuda", dtype=dt)
    tl = torch.randn(m, v, device="cuda", dtype=dt)
    args = (BETA, IGNORE, TEMP, sl, tl)

    # 1) the ACTUAL emitted seed from the live heuristic (no hand-picking)
    bound = jsd_kernel.bind(args)
    seed = compiler_seed_configs(bound.env, bound.host_function.device_ir)[0]
    base = dict(seed.config)
    emitted_rl = base.get("reduction_loops")

    # 2) grad-fair tc reference + timing (loss AND grad)
    ref_loss, _ref_grad = grad_fair_tc(sl, tl)()
    torch._dynamo.reset()
    tcf = torch.compile(grad_fair_tc(sl, tl))
    tcf()  # warm
    t_tc = median_us(tcf)

    out: dict = {
        "shape": (m, v),
        "dtype": str(dt).replace("torch.", ""),
        "emitted_reduction_loops": emitted_rl,
        "t_tc_us": round(t_tc, 1),
        "arms": {},
    }

    # 3) BEFORE = persistent [None]; AFTER = the emitted seed (must be looped [8192] for narrow-V).
    #    Same config in every other field -> only reduction_loops differs.
    for tag, rl in (("before_None", [None]), ("after_seed", emitted_rl)):
        cfg = dict(base)
        cfg["reduction_loops"] = rl
        k = helion.kernel(jsd_kernel.fn, config=helion.Config(**cfg), static_shapes=True)
        loss, grad = k(*args)
        acc = bool(
            torch.allclose(loss.float(), ref_loss.float(), rtol=3e-2, atol=3e-2)
        )
        t = median_us(lambda: k(*args))
        out["arms"][tag] = {
            "reduction_loops": rl,
            "acc": acc,
            "lat_us": round(t, 1),
            "G_vs_fairtc": round(t_tc / t, 3),
        }

    del sl, tl
    torch.cuda.empty_cache()
    return out


def main() -> None:
    print(f"helion={helion.__file__}")
    print(f"device={torch.cuda.get_device_name()}  N_med={N_MED}  grad-fair tc (loss+grad)\n")
    all_pass = True
    for (m, v, dt) in NARROW:
        r = run_one(m, v, dt)
        b = r["arms"]["before_None"]
        a = r["arms"]["after_seed"]
        seed_ok = r["emitted_reduction_loops"] == [8192]
        ratio = a["G_vs_fairtc"] / b["G_vs_fairtc"] if b["G_vs_fairtc"] else float("inf")
        flips = b["G_vs_fairtc"] < 1.0 <= a["G_vs_fairtc"]
        clears_noise = ratio >= 1.05
        ok = seed_ok and b["acc"] and a["acc"] and flips and clears_noise
        all_pass = all_pass and ok
        print(
            f"flj {r['dtype']:5} {str(r['shape']):14} emitted_rl={r['emitted_reduction_loops']} "
            f"seed==[8192]:{seed_ok}\n"
            f"    tc(loss+grad)={r['t_tc_us']:8.1f}us\n"
            f"    [None]  acc={b['acc']} lat={b['lat_us']:8.1f}us G={b['G_vs_fairtc']:.3f}\n"
            f"    [seed]  acc={a['acc']} lat={a['lat_us']:8.1f}us G={a['G_vs_fairtc']:.3f}  "
            f"(after/before={ratio:.2f}x)  -> {'PASS' if ok else 'FAIL'}"
        )
    print(f"\n=== CLAIM REPRODUCED: {all_pass} ===")
    print("(PASS = seed emits [8192] AND both arms accurate AND G flips <1 -> >=1 "
          "AND the move clears the 5% noise band, on every narrow-V shape.)")


if __name__ == "__main__":
    main()
