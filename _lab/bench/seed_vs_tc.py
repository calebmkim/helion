"""Clean seeded-Helion vs torch.compile-default, PURE TritonBench.

Per kernel: ONE run.py process (via run_seeded.py, which promotes the reduction
seed to default_config so effort=none runs the seed). TritonBench resets dynamo +
kernel state per input (the isolation our hand-rolled scripts lacked), so the
torch.compile baseline is measured cleanly. From that single isolated table we read
both columns:
  helion_<k>-latency   = seeded Helion
  torch_compile_*-lat   = torch.compile DEFAULT mode (no max-autotune)
and report G_seed = tc_default_latency / helion_seeded_latency  (>1 = seed beats tc).

Shapes: kernels that accept --shapes get the test split; welford/kl_div/jsd run
their operator's built-in sweep (no --shapes support). All TritonBench-native.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys

PR = "/home/dev/local/helion-pr-edit"
PY = "/home/dev/helion/.venv/bin/python"
sys.path.insert(0, "/home/dev/local/helion-reduction-heuristics-run2/_lab/prompts")
import shapes_v3_draft as SH  # noqa: E402

# kernel -> (run.py --kernel, helion latency col, tc-default latency col,
#            supports --shapes, extra args)
KCFG = {
    "rms_norm": ("rms_norm", "helion_rms_norm_tritonbench-latency",
                 "torch_compile_no_autotune_rms-latency", True, []),
    "layer_norm": ("layer_norm", "helion_layer_norm_tritonbench-latency",
                   "torch_compile_no_autotune_layer_norm-latency", True, []),
    "softmax": ("softmax", "helion_softmax_tritonbench-latency",
                "torch_compile_no_autotune_softmax-latency", True, []),
    "sum": ("sum", "helion_sum_tritonbench-latency",
            "torch_compile_no_autotune_sum-latency", True, ["--reduce-dim", "1"]),
    "cross_entropy": ("cross_entropy", "helion_cross_entropy-latency",
                      "torch_compile_no_autotune_cross_entropy_loss-latency", True, []),
    "welford": ("welford", "helion_welford_tritonbench-latency",
                "torch_compile_welford_default-latency", False, []),
    "kl_div": ("kl_div", "helion_kl_div_tritonbench-latency",
               "torch_compile_kl_div_default-latency", False, []),
    "jsd": ("jsd", "helion_jsd_tritonbench-latency",
            "torch_compile_jsd_default-latency", False, []),
    # long_sum reuses the sum operator; needs --reduce-dim 1 (row reduction) and the
    # long-N test shapes via --shapes. helion col is the longsum hook's method name.
    "long_sum": ("long_sum", "helion_longsum_tritonbench-latency",
                 "torch_compile_no_autotune_sum-latency", True, ["--reduce-dim", "1"]),
}


def _foreign_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=10).stdout
    except Exception:  # noqa: BLE001
        return 0
    m = 0
    for line in out.splitlines():
        p = [c.strip() for c in line.split(",")]
        if len(p) == 2 and p[0].isdigit():
            m = max(m, int(p[1]) if p[1].isdigit() else 0)
    return m


def _run_seeded(kernel: str) -> tuple[list[dict], int]:
    run_name, hl_col, tc_col, supports_shapes, extra = KCFG[kernel]
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "0", "HELION_AUTOTUNE_EFFORT": "none",
                "PYTHONPATH": PR, "HELION_PROMOTE_REDUCTION_SEED": "1"})
    cmd = [PY, f"{PR}/_lab/bench/run_seeded.py", "--kernel", run_name,
           "--metrics", "latency,accuracy", "--precision", "fp32",
           "--csv", "--output-dir", "/tmp/seedtc_out", *extra]
    if supports_shapes:
        shapes = ";".join(f"{m},{n}" for m, n in SH.SHAPES[kernel]["test"])
        cmd += ["--shapes", shapes]
    f_before = _foreign_mib()
    out = subprocess.run(cmd, cwd=PR, env=env, capture_output=True, text=True)
    foreign = max(f_before, _foreign_mib())
    text = out.stdout + "\n" + out.stderr
    header, rows = None, []
    for line in text.splitlines():
        cells = line.split(";")
        if len(cells) < 3:
            continue
        first = cells[0].strip()
        if first in ("(M, H)", "x_val") or (first.startswith("(") and "latency" in line):
            header = [c.strip() for c in cells]
        elif header and (re.match(r"^\(\d+", first) or re.match(r"^\d+$", first)):
            rows.append(dict(zip(header, cells)))
    if not rows:
        sys.stderr.write(text[-3000:])
        raise RuntimeError(f"no table for {kernel}")
    return rows, foreign


def _num(x: str) -> float:
    m = re.match(r"\s*([0-9.]+)", x or "")
    return float(m.group(1)) if m else float("nan")


def _col(rows: list[dict], name: str) -> str:
    keys = list(rows[0])
    if name in keys:
        return name
    hits = [c for c in keys if name in c]
    if len(hits) != 1:
        raise RuntimeError(f"col {name!r} -> {hits}")
    return hits[0]


def bench(kernel: str) -> dict:
    _, hl_sub, tc_sub, _, _ = KCFG[kernel]
    rows, foreign = _run_seeded(kernel)
    hl, tc = _col(rows, hl_sub), _col(rows, tc_sub)
    acc = _col(rows, hl_sub.replace("-latency", "-accuracy"))
    shape_key = list(rows[0])[0]
    out_rows = []
    for r in rows:
        if r[shape_key].strip() == "average":
            continue
        h, t = _num(r[hl]), _num(r[tc])
        out_rows.append({
            "shape": r[shape_key].strip(),
            "helion_seed_ms": h, "tc_default_ms": t,
            "G_seed": round(t / h, 4) if h else None,
            "acc": r.get(acc, "?").strip(),
        })
    gs = [r["G_seed"] for r in out_rows if r["G_seed"]]
    return {
        "kernel": kernel, "rows": out_rows,
        "geo_G_seed": round(statistics.geometric_mean(gs), 4),
        "median_G_seed": round(statistics.median(gs), 4),
        "min_G_seed": round(min(gs), 4), "max_G_seed": round(max(gs), 4),
        "any_acc_fail": any(r["acc"] not in ("1", "1.0", "?") for r in out_rows),
        "foreign_mib": foreign,
    }


def main() -> None:
    os.makedirs("/tmp/seedtc_out", exist_ok=True)
    kernels = sys.argv[1:] or list(KCFG)
    results = []
    for k in kernels:
        sys.stderr.write(f"\n===== {k} =====\n")
        try:
            r = bench(k)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"  {k} FAILED: {e}\n")
            results.append({"kernel": k, "error": str(e)})
            continue
        sys.stderr.write(
            f"  G_seed median={r['median_G_seed']} geo={r['geo_G_seed']} "
            f"({r['min_G_seed']}-{r['max_G_seed']}) acc_fail={r['any_acc_fail']} "
            f"foreign={r['foreign_mib']}MiB\n")
        results.append(r)
        with open("/tmp/seedtc_out/results.json", "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
