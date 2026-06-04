"""TASK 2 — three-arm head-to-head, ALL ARMS ON UP-TO-DATE MAIN.

The fair comparison (user's "config oracle" idea): run everything on the SAME
compiler substrate (origin/main, which has the just-landed baseline reduction
heuristic, PR #2648), so the ONLY thing that varies between the heuristic arms
is the CONFIG.

Three arms, per (kernel, shape) on the TEST split (all 9 kernels):
  - baseline : main's live heuristic config IF it fires (compiler_seed_configs
               non-empty), ELSE config_spec.default_config().  RECORD which
               ('heuristic' vs 'default').
  - mine     : caleb's heuristic config, REPLAYED from the recorded oracle
               (_lab/logs/run3/task1_seed_configs.json, raw_seed) — caleb's
               heuristic cannot run on main (needs facts absent there), so we
               replay its recorded output.  Always 'heuristic_replayed'.
  - tc       : torch.compile DEFAULT mode of the fp32 reference.

Every arm: configs=[cfg] (NO autotune), fp32 (asserted), correctness-gated
(rtol=1e-3/atol=1e-4, NOT loosened), median-of-N do_bench, GPU idle-gated before
each shape, NORMALIZED running config recorded (proves the intended config ran).

MUST be run with PYTHONPATH pointing at the MAIN checkout (helion-main-baseline),
so `import helion` + `from examples...` resolve to MAIN:

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-main-baseline \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_task2_replay_bench.py \
    [--kernels rms_norm,sum,...] [--smoke]

--smoke : correctness + config-transfer only (no timing), 1st test shape/kernel.
Resumable: results keyed (kernel,split,M,N) in the out JSON; existing rows skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback

import torch

import helion

# ----- paths (this script lives in caleb's branch _lab; the ORACLE json too) -- #
_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_BRANCH_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
_PROMPTS_DIR = os.path.join(_BRANCH_ROOT, "_lab", "prompts")
if _PROMPTS_DIR not in sys.path:
    sys.path.insert(0, _PROMPTS_DIR)

ORACLE = os.path.join(_BRANCH_ROOT, "_lab", "logs", "run3", "task1_seed_configs.json")
OUT = os.path.join(_BRANCH_ROOT, "_lab", "logs", "run3", "task2_replay_bench.json")

# helion MUST resolve to a checkout that is NOT caleb's branch (we want MAIN).
assert not os.path.abspath(helion.__file__).startswith(_BRANCH_ROOT + os.sep), (
    f"helion ({helion.__file__}) resolves to caleb's branch — point PYTHONPATH "
    "at the MAIN checkout (helion-main-baseline) so all arms share main's substrate."
)

from triton.testing import do_bench  # noqa: E402

from shapes_v3_draft import SHAPES  # noqa: E402

from helion._compiler.autotuner_heuristics import compiler_seed_configs  # noqa: E402

# Builders + fp32 references — logic copied verbatim from run2_measure_g (the
# established harness); we inline them rather than import (run2_measure_g asserts
# helion is under caleb's branch, which is false here by design).
from examples.rms_norm import rms_norm_fwd, rms_norm_pytorch  # noqa: E402
from examples.sum import sum_kernel  # noqa: E402
from examples.long_sum import longsum  # noqa: E402
from examples.layer_norm import layer_norm_fwd  # noqa: E402
from examples.softmax import softmax_two_pass  # noqa: E402
from examples.kl_div import kl_div_forward  # noqa: E402
from examples.jsd import jsd_forward, TorchJSDBaseline  # noqa: E402
from examples.cross_entropy import cross_entropy  # noqa: E402
from examples.welford import welford, eager_layer_norm  # noqa: E402

EPS = 1e-5
N_RUNS = 7
LONG = torch.int64
KERNEL_ORDER = [
    "rms_norm", "layer_norm", "softmax", "welford", "sum", "long_sum",
    "cross_entropy", "kl_div", "jsd",
]


def _first(o):
    return o[0] if isinstance(o, tuple) else o


def build_rms_norm(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    return (x, w, EPS), rms_norm_pytorch(x, w, EPS), _first


def build_layer_norm(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    w = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    return ((x, [n], w, b, EPS),
            torch.nn.functional.layer_norm(x, [n], w, b, EPS), _first)


def build_welford(m, n):
    weight = torch.rand(n, device="cuda", dtype=torch.float32)
    bias = torch.rand(n, device="cuda", dtype=torch.float32)
    x = torch.rand(m, n, device="cuda", dtype=torch.float32)
    args = (weight, bias, x, EPS)
    return args, eager_layer_norm(*args), _first


def build_softmax(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    return (x,), torch.nn.functional.softmax(x, dim=1), _first


def build_cross_entropy(m, n):
    logits = torch.randn(m, n, device="cuda", dtype=torch.float32)
    labels = torch.randint(0, n, (m,), device="cuda", dtype=LONG)
    return (logits, labels), torch.nn.functional.cross_entropy(logits, labels), _first


def build_kl_div(m, n):
    yp = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1)
    yt = torch.randn(m, n, device="cuda", dtype=torch.float32).softmax(-1)
    args = (yp, yt, False, "batchmean", 1e-10)
    ref = torch.nn.KLDivLoss(reduction="batchmean", log_target=False).to("cuda")(yp, yt)
    return args, ref, _first


_JSD_BASELINE = TorchJSDBaseline(beta=0.5, ignore_index=-100)


def build_jsd(m, n):
    lq = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1)
    lp = torch.randn(m, n, device="cuda", dtype=torch.float32).log_softmax(-1)
    args = (lq, lp, None, 0.5, -100)
    return args, _JSD_BASELINE(lq, lp), _first


def build_sum(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    return (x,), torch.sum(x, dim=-1), _first


def build_long_sum(m, n):
    x = torch.randn(m, n, device="cuda", dtype=torch.float32)
    return (x,), torch.sum(x, dim=-1), _first


KERNELS = {
    "rms_norm": (rms_norm_fwd, build_rms_norm, lambda a: rms_norm_pytorch(*a)),
    "layer_norm": (layer_norm_fwd, build_layer_norm,
                   lambda a: torch.nn.functional.layer_norm(a[0], a[1], a[2], a[3], a[4])),
    "welford": (welford, build_welford, lambda a: eager_layer_norm(*a)),
    "softmax": (softmax_two_pass, build_softmax,
                lambda a: torch.nn.functional.softmax(a[0], dim=1)),
    "cross_entropy": (cross_entropy, build_cross_entropy,
                      lambda a: torch.nn.functional.cross_entropy(a[0], a[1])),
    "kl_div": (kl_div_forward, build_kl_div,
               lambda a: torch.nn.KLDivLoss(reduction="batchmean",
                                            log_target=False).to("cuda")(a[0], a[1])),
    "jsd": (jsd_forward, build_jsd, lambda a: _JSD_BASELINE(a[0], a[1])),
    "sum": (sum_kernel, build_sum, lambda a: torch.sum(a[0], dim=-1)),
    "long_sum": (longsum, build_long_sum, lambda a: torch.sum(a[0], dim=-1)),
}


# --------------------------------------------------------------------------- #
# GPU idle-gate (the welford-owner can grab the GPU mid-sweep).
# --------------------------------------------------------------------------- #
def _external_busy_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return 0
    me = os.getpid()
    busy = 0
    for line in filter(None, (ln.strip() for ln in out.splitlines())):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid, mib = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if pid != me:
            busy += mib
    return busy


def _wait_idle(label, thresh=1024, tries=30, sleep_s=10.0):
    for _ in range(tries):
        b = _external_busy_mib()
        if b <= thresh:
            return
        print(f"  [idle-gate] {label}: external {b} MiB; waiting {sleep_s:.0f}s",
              flush=True)
        time.sleep(sleep_s)
    print(f"  [idle-gate] {label}: still busy after {tries} — benching anyway", flush=True)


# --------------------------------------------------------------------------- #
# Measurement primitives.
# --------------------------------------------------------------------------- #
def median_do_bench(fn):
    torch.cuda.synchronize()
    samples = [float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS)]
    return sorted(samples)[len(samples) // 2]


def check_correct(out, ref):
    o = torch.as_tensor(out).to(torch.float32)
    r = torch.as_tensor(ref).to(torch.float32)
    ok = bool(torch.allclose(o, r, rtol=1e-3, atol=1e-4))
    maxerr = float((o - r).abs().max())
    return ok, maxerr


def _norm_running_cfg(bound):
    """The actually-running normalized config dict (proves what ran)."""
    return {k: v for k, v in dict(bound._config).items()}


def run_arm(fn, args, ref, out_extract, cfg_dict, *, timed):
    """Build kernel with configs=[cfg], check correctness, optionally time.

    Returns dict: {normalized_cfg, correct, maxerr, lat_us|None, error|None}.
    """
    try:
        seeded = helion.kernel(fn.fn, configs=[helion.Config(**dict(cfg_dict))])
        bound = seeded.bind(args)
        bound.ensure_config_exists(args)
        normalized = _norm_running_cfg(bound)
        out = out_extract(bound(*args))
        correct, maxerr = check_correct(out, ref)
        lat = None
        if timed and correct:
            lat = median_do_bench(lambda: bound(*args)) * 1000.0  # ms->us
        return {"normalized_cfg": normalized, "correct": correct,
                "maxerr": maxerr, "lat_us": lat, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"normalized_cfg": None, "correct": False, "maxerr": None,
                "lat_us": None, "error": f"{type(e).__name__}: {e}"}


def run_tc(tc_ref, args, ref, out_extract, *, timed):
    try:
        torch._dynamo.reset()
        tc = torch.compile(tc_ref)
        out = out_extract(tc(args))
        correct, maxerr = check_correct(out, ref)
        lat = median_do_bench(lambda: tc(args)) * 1000.0 if timed else None
        return {"correct": correct, "maxerr": maxerr, "lat_us": lat, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"correct": False, "maxerr": None, "lat_us": None,
                "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default="", help="comma list; default all")
    ap.add_argument("--smoke", action="store_true",
                    help="correctness+config-transfer only (no timing), 1 shape/kernel")
    args_ns = ap.parse_args()
    only = set(args_ns.kernels.split(",")) if args_ns.kernels else set(KERNEL_ORDER)
    timed = not args_ns.smoke

    print(f"helion={helion.__file__}", flush=True)
    print(f"oracle={ORACLE}\nout={OUT}  timed={timed}\n", flush=True)

    # oracle: caleb's recorded configs, keyed (kernel, M, N) -> raw_seed
    oracle_rows = json.load(open(ORACLE))["rows"]
    mine_cfg = {(r["kernel"], r["M"], r["N"]): r.get("raw_seed")
                for r in oracle_rows if "raw_seed" in r}

    # resume: load existing results
    results = {}
    if os.path.exists(OUT) and not args_ns.smoke:
        try:
            for row in json.load(open(OUT))["rows"]:
                results[(row["kernel"], row["split"], row["M"], row["N"])] = row
        except Exception:
            results = {}

    for kn in KERNEL_ORDER:
        if kn not in only:
            continue
        fn, builder, tc_ref = KERNELS[kn]
        test_shapes = SHAPES[kn]["test"]
        if args_ns.smoke:
            test_shapes = test_shapes[:1]
        for (M, N) in test_shapes:
            rkey = (kn, "test", M, N)
            if rkey in results and not args_ns.smoke:
                print(f"  -- skip {kn}(test {M},{N}) (already done)", flush=True)
                continue
            tag = f"{kn}(test {M},{N})"
            _wait_idle(tag)
            try:
                a_args, ref, out_extract = builder(M, N)
                # fp32 assertion on the primary input tensor
                prim = a_args[0]
                assert torch.is_tensor(prim) and prim.dtype == torch.float32, \
                    f"{tag}: primary input not fp32 ({getattr(prim,'dtype',None)})"

                # --- baseline arm: main heuristic if fires, else default ---
                b = fn.bind(a_args)
                seeds = compiler_seed_configs(b.env, b.host_function.device_ir)
                if seeds:
                    base_cfg = dict(seeds[0])
                    base_source = "heuristic"
                else:
                    base_cfg = dict(b.env.config_spec.default_config())
                    base_source = "default"
                base_fired = list(getattr(b.env.config_spec, "autotuner_heuristics", []))
                del b
                torch.cuda.empty_cache()

                base = run_arm(fn, a_args, ref, out_extract, base_cfg, timed=timed)

                # --- mine arm: replay caleb's recorded config ---
                my_cfg = mine_cfg.get((kn, M, N))
                if my_cfg is None:
                    mine = {"normalized_cfg": None, "correct": False, "maxerr": None,
                            "lat_us": None, "error": "no oracle config for shape"}
                else:
                    mine = run_arm(fn, a_args, ref, out_extract, my_cfg, timed=timed)

                # --- tc arm ---
                tc = run_tc(tc_ref, a_args, ref, out_extract, timed=timed)

                def _ratio(num, den):
                    return (num / den) if (num and den) else None

                row = {
                    "kernel": kn, "split": "test", "M": M, "N": N,
                    "baseline_source": base_source,        # 'heuristic' | 'default'
                    "baseline_fired": base_fired,
                    "baseline_intended_cfg": base_cfg,
                    "baseline_normalized_cfg": base["normalized_cfg"],
                    "baseline_correct": base["correct"], "baseline_maxerr": base["maxerr"],
                    "baseline_us": base["lat_us"], "baseline_error": base["error"],
                    "mine_source": "heuristic_replayed",
                    "mine_intended_cfg": my_cfg,
                    "mine_normalized_cfg": mine["normalized_cfg"],
                    "mine_correct": mine["correct"], "mine_maxerr": mine["maxerr"],
                    "mine_us": mine["lat_us"], "mine_error": mine["error"],
                    "tc_correct": tc["correct"], "tc_maxerr": tc["maxerr"],
                    "tc_us": tc["lat_us"], "tc_error": tc["error"],
                    # ratios (<1 => first is faster)
                    "mine_over_baseline": _ratio(mine["lat_us"], base["lat_us"]),
                    "mine_over_tc": _ratio(mine["lat_us"], tc["lat_us"]),
                    "baseline_over_tc": _ratio(base["lat_us"], tc["lat_us"]),
                }
                results[rkey] = row
                mob = row["mine_over_baseline"]
                print(
                    f"[OK] {tag:30s} base[{base_source}]="
                    f"{(base['lat_us'] or float('nan')):8.1f}us "
                    f"mine={(mine['lat_us'] or float('nan')):8.1f}us "
                    f"tc={(tc['lat_us'] or float('nan')):8.1f}us "
                    f"mine/base={mob if mob else float('nan'):.3f} "
                    f"corr(b/m/tc)={int(base['correct'])}/{int(mine['correct'])}/{int(tc['correct'])}",
                    flush=True)
                if base["error"]:
                    print(f"     baseline_error: {base['error']}", flush=True)
                if mine["error"]:
                    print(f"     mine_error: {mine['error']}", flush=True)
                if tc["error"]:
                    print(f"     tc_error: {tc['error']}", flush=True)

                del a_args, ref
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                print(f"[ERR] {tag}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                results[rkey] = {"kernel": kn, "split": "test", "M": M, "N": N,
                                 "fatal_error": f"{type(e).__name__}: {e}"}
                torch.cuda.empty_cache()

            if not args_ns.smoke:
                json.dump({"rows": list(results.values())}, open(OUT, "w"), indent=1)

    if not args_ns.smoke:
        json.dump({"rows": list(results.values())}, open(OUT, "w"), indent=1)
        print(f"\nwrote {OUT}  ({len(results)} rows)", flush=True)
    else:
        print("\n=== SMOKE DONE (no file written) ===", flush=True)


if __name__ == "__main__":
    main()
