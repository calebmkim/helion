"""RUN-3 / RUN-2 matrix: the autotune comparison across TEST shapes.

Per (kernel, TEST shape) it produces the 4 Helion autotune arms + tc-max + a
fair re-bench of every winner, fully CHECKPOINTED and resumable:

  C2 = unseeded quick     C3 = unseeded full(max)
  C5 = seeded   quick     C6 = seeded   full(max)
  C8 = torch.compile max-autotune  (do_bench median-of-7)

For each Helion arm it runs a COLD-CACHE autotune in a fresh subprocess
(run2_productB_driver.py) writing the per-generation CSV (HELION_AUTOTUNE_LOG),
then:
  - parses per-generation best-so-far (for the 4-arm x generation table),
  - extracts the winning Config and FAIR-RE-BENCHES it with do_bench median-of-7
    (so the arm latencies are comparable to Run-1's C1/C4/C7 and to C8).

CHECKPOINTING:
  - autotune cell = a CSV+.out in <out>/pb/ ; skipped if the .out already has
    "[driver] DONE" (complete run). Interrupt-safe (cold-cache fresh subprocess).
  - per-(kernel,shape) summary written to <out>/run2_<kernel>_<M>x<N>.json after
    every shape, with the re-benched winners + per-gen arrays + C8.

ORDERING: pass --efforts quick first (all shapes), then --efforts full — so an
interrupt leaves complete quick coverage. --reps controls autotune repeats/arm.

Run from a non-checkout cwd, one GPU per process (disjoint kernels):
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/.../wt-reduction-2 python run3_run2_matrix.py \
      --kernels sum,rms_norm --efforts quick --reps 1 --gpu 1
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import time

import torch

import helion

WT = "/home/calebkim/helion-new-heuristics/wt-reduction-2/"
assert helion.__file__.startswith(WT), f"WRONG helion (prefix trap?): {helion.__file__}"

from run2_measure_g import KERNELS, check_correct, median_do_bench  # noqa: E402
from run2_productB_analyze import best_vs_gen, parse_csv  # noqa: E402
from run3_run1_matrix import TEST_SHAPES  # noqa: E402

DRIVER = os.path.join(os.path.dirname(__file__), "run2_productB_driver.py")
PY = sys.executable

# run3/measure_g kernel name -> run2_productB_driver kernel name
DRIVER_NAME = {"softmax": "softmax_two_pass"}


def driver_kernel(k):
    return DRIVER_NAME.get(k, k)


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return math.exp(sum(math.log(v) for v in xs) / len(xs)) if xs else None


def _save(path, blob):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(blob, f, indent=2, default=str)
    os.replace(tmp, path)


def _load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def autotune_cell(kernel, m, n, mode, effort, rep, pbdir, gpu):
    """Run ONE cold-cache autotune (subprocess). Skip if already complete.
    Returns (csv_path, out_path, winner_dict_or_None)."""
    dk = driver_kernel(kernel)
    tag = f"{dk}_{m}x{n}_{mode}_{effort}_s{rep}"
    prefix = os.path.join(pbdir, tag)
    csv_path, out_path = prefix + ".csv", prefix + ".out"
    if os.path.exists(out_path) and "[driver] DONE" in open(out_path).read():
        return csv_path, out_path, _winner_from_out(out_path)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = WT.rstrip("/")
    env["HELION_FORCE_AUTOTUNE"] = "1"
    env["HELION_AUTOTUNE_EFFORT"] = effort
    env["HELION_AUTOTUNE_RANDOM_SEED"] = str(rep)
    env["HELION_AUTOTUNE_LOG"] = prefix
    env["HELION_BENCHMARK_DISABLE_LOGGING"] = "1"
    if mode == "unseeded":
        env["HELION_DISABLE_AUTOTUNER_HEURISTICS"] = "1"
    else:
        env.pop("HELION_DISABLE_AUTOTUNER_HEURISTICS", None)
    with open(out_path, "w") as out:
        subprocess.run(
            [PY, DRIVER, "--kernel", dk, "--M", str(m), "--N", str(n),
             "--mode", mode, "--rand-seed", str(rep), "--log", prefix],
            env=env, stdout=out, stderr=subprocess.STDOUT, check=False,
        )
    return csv_path, out_path, _winner_from_out(out_path)


def _winner_from_out(out_path):
    try:
        for line in open(out_path):
            if "[driver] DONE best_config=" in line:
                return ast.literal_eval(line.split("best_config=", 1)[1].strip())
    except (OSError, ValueError, SyntaxError):
        pass
    return None


def rebench_config(kernel, m, n, cfg):
    """Fair do_bench median-of-7 of an explicit Config (comparable to Run-1)."""
    if cfg is None:
        return None, None, "no-winner"
    fn, builder, _ = KERNELS[kernel]
    args, ref, out_extract = builder(m, n)
    try:
        k = helion.kernel(fn.fn, configs=[helion.Config(**cfg)])
        b = k.bind(args)
        b.ensure_config_exists(args)
        ok, _ = check_correct(out_extract(b(*args)), ref)
        lat = median_do_bench(lambda: b(*args)) if ok else None
        return ((lat * 1000) if lat else None), ok, None
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {str(e)[:160]}"


def measure_c8(kernel, m, n):
    """torch.compile max-autotune do_bench median-of-7."""
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(m, n)
    try:
        torch._dynamo.reset()
        tc = torch.compile(tc_ref, mode="max-autotune-no-cudagraphs")
        ok, _ = check_correct(out_extract(tc(args)), ref)
        lat = median_do_bench(lambda: tc(args)) if ok else None
        return ((lat * 1000) if lat else None), ok, None
    except Exception as e:  # noqa: BLE001
        return None, None, f"{type(e).__name__}: {str(e)[:160]}"


def per_gen_median(csv_paths):
    """median over reps of best-so-far at each generation -> {gen: ms}."""
    per_rep = [best_vs_gen(parse_csv(p)) for p in csv_paths]
    per_rep = [d for d in per_rep if d]
    if not per_rep:
        return {}
    gens = sorted({g for d in per_rep for g in d})
    out = {}
    for g in gens:
        vals = sorted(d[g] for d in per_rep if g in d)
        out[g] = vals[len(vals) // 2]
    return out


def do_shape(kernel, m, n, efforts, reps, pbdir, out_dir, gpu):
    cell_path = os.path.join(out_dir, f"run2_{kernel}_{m}x{n}.json")
    blob = _load(cell_path) or {"kernel": kernel, "shape": [m, n], "gpu": gpu,
                                "arms": {}}
    for effort in efforts:
        for mode in ("unseeded", "seeded"):
            arm = f"{mode}_{effort}"
            csvs, winners = [], []
            for rep in range(reps):
                t0 = time.time()
                csv_p, out_p, win = autotune_cell(kernel, m, n, mode, effort,
                                                  rep, pbdir, gpu)
                csvs.append(csv_p)
                winners.append(win)
                print(f"[AT ] {kernel}({m},{n}) {arm} s{rep} "
                      f"({time.time() - t0:.0f}s) winner={'ok' if win else 'NONE'}",
                      flush=True)
            # re-bench each rep's winner (fair do_bench)
            reb = []
            for win in winners:
                lat, ok, err = rebench_config(kernel, m, n, win)
                reb.append({"lat_us": lat, "correct": ok, "err": err,
                            "config": win})
            lats = [r["lat_us"] for r in reb if r["lat_us"]]
            blob["arms"][arm] = {
                "rebench_lat_us_median": (sorted(lats)[len(lats) // 2]
                                          if lats else None),
                "reps": reb,
                "per_gen_best_ms": per_gen_median(csvs),
            }
            _save(cell_path, blob)  # checkpoint after each arm
            print(f"[REB] {kernel}({m},{n}) {arm} "
                  f"rebench_us={blob['arms'][arm]['rebench_lat_us_median']}",
                  flush=True)
    # C8 (tc max-autotune) once per shape
    if "c8_tc_max_us" not in blob:
        t0 = time.time()
        c8, ok, err = measure_c8(kernel, m, n)
        blob["c8_tc_max_us"] = c8
        blob["c8_correct"] = ok
        blob["c8_err"] = err
        _save(cell_path, blob)
        print(f"[C8 ] {kernel}({m},{n}) tc_max_us={c8} ({time.time()-t0:.0f}s)",
              flush=True)
    return cell_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernels", default=",".join(TEST_SHAPES))
    ap.add_argument("--efforts", default="quick,full")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--shape", default=None, help="single MxN override e.g. 4096x12288")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "..", "logs", "run3"))
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out)
    pbdir = os.path.join(out_dir, "pb")
    os.makedirs(pbdir, exist_ok=True)
    efforts = [e.strip() for e in a.efforts.split(",") if e.strip()]
    kernels = [k.strip() for k in a.kernels.split(",") if k.strip()]
    override = None
    if a.shape:
        override = tuple(int(x) for x in a.shape.lower().split("x"))
    print(f"GPU={a.gpu} helion={helion.__file__}\nkernels={kernels} "
          f"efforts={efforts} reps={a.reps} out={out_dir}\n", flush=True)
    # GLOBAL passes: ALL shapes at quick first, THEN all shapes at full, so an
    # interrupt leaves complete quick coverage. Per-shape try/except so one bad
    # shape never aborts the autonomous run.
    for effort in efforts:
        print(f"\n========== GLOBAL PASS: effort={effort} ==========\n", flush=True)
        for k in kernels:
            shapes = [override] if override else TEST_SHAPES[k]
            for (m, n) in shapes:
                try:
                    do_shape(k, m, n, [effort], a.reps, pbdir, out_dir, a.gpu)
                except Exception as e:  # noqa: BLE001
                    import traceback
                    print(f"[FATAL-shape] {k}({m},{n}) {effort}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    torch.cuda.empty_cache()
        print(f"\n========== PASS DONE: effort={effort} ==========\n", flush=True)


if __name__ == "__main__":
    main()
