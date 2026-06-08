"""Drive the 3-arm reduction-seed benchmark across all curriculum kernels via
TritonBench (benchmarks/run.py), then summarize with a noise-robust ratio.

Per kernel we run benchmarks/run.py TWICE at HELION_AUTOTUNE_EFFORT=none:
  * SEEDED  : HELION_PROMOTE_REDUCTION_SEED=1  (default_config() == PR seed)
  * DEFAULT : HELION_DISABLE_AUTOTUNER_HEURISTICS=1 (default_config() == base)
Both processes also bench the operator's torch.compile DEFAULT-mode baseline
(`torch_compile_*_default` / `torch_compile_no_autotune_*`), which is the stable
cross-process ANCHOR. Because small-kernel cross-process do_bench jitter is ~5-10%,
the headline seed effect is the WITHIN-process tc-ratio lift:
    seed_effect = (seed_helion / tc)^-1  /  (default_helion / tc)^-1
                = (default_helion/tc) / (seed_helion/tc)   [latency form]
We also report the raw seeded/default and each arm's speedup vs tc-default.

This file only ORCHESTRATES + PARSES; all timing/accuracy/inputs are TritonBench's.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time

PR = "/home/dev/local/helion-pr-edit"
PY = "/home/dev/helion/.venv/bin/python"
SHAPES_DIR = "/home/dev/local/helion-reduction-heuristics-run2/_lab/prompts"

sys.path.insert(0, SHAPES_DIR)
import shapes_v3_draft as SH  # noqa: E402

# kernel -> (run.py --kernel name, helion latency col substr, tc-default col substr,
#            extra run.py args)
KCFG = {
    "rms_norm": ("rms_norm", "helion_rms_norm_tritonbench-latency",
                 "torch_compile_no_autotune_rms-latency", []),
    "layer_norm": ("layer_norm", "helion_layer_norm_tritonbench-latency",
                   "torch_compile_no_autotune_layer_norm-latency", []),
    "softmax": ("softmax", "helion_softmax_tritonbench-latency",
                "torch_compile_no_autotune_softmax-latency", []),
    "sum": ("sum", "helion_sum_tritonbench-latency",
            "torch_compile_no_autotune_sum-latency", ["--reduce-dim", "1"]),
    "cross_entropy": ("cross_entropy", "helion_cross_entropy-latency",
                      "torch_compile_no_autotune_cross_entropy_loss-latency", []),
    "welford": ("welford", "helion_welford_tritonbench-latency",
                "torch_compile_welford_default-latency", []),
    "kl_div": ("kl_div", "helion_kl_div_tritonbench-latency",
               "torch_compile_kl_div_default-latency", []),
    "jsd": ("jsd", "helion_jsd_tritonbench-latency",
            "torch_compile_jsd_default-latency", []),
    "long_sum": ("long_sum", "helion_longsum_tritonbench-latency",
                 "torch_compile_no_autotune_sum-latency", ["--reduce-dim", "1"]),
}


# Max foreign-GPU-process MiB observed during the most recent _run() call.
# >0 means another process shared the GPU => timings for that run are suspect.
_LAST_FOREIGN = 0
# A run is treated as contaminated if a foreign process used more than this many
# MiB at any sample (a few hundred MiB of transient context is tolerable; a real
# co-tenant kernel is GBs).
FOREIGN_MIB_THRESHOLD = 300


def _shapes_arg(kernel: str, split: str) -> str:
    return ";".join(f"{m},{n}" for m, n in SH.SHAPES[kernel][split])


def _gpu_pids() -> dict[int, int]:
    """{pid: used_MiB} of every process on the GPU right now."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:  # noqa: BLE001
        return {}
    pids = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            pids[int(parts[0])] = int(parts[1]) if parts[1].isdigit() else 0
    return pids


def _descendants(root: int) -> set[int]:
    """All PIDs in the process tree rooted at `root` (so we don't flag our own
    benchmark subprocess + its inductor compile workers as foreign)."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,ppid"], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return {root}
    children: dict[int, list[int]] = {}
    for line in out.splitlines()[1:]:
        f = line.split()
        if len(f) == 2 and f[0].isdigit() and f[1].isdigit():
            children.setdefault(int(f[1]), []).append(int(f[0]))
    seen, stack = set(), [root]
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        stack.extend(children.get(p, []))
    return seen


def _run(kernel_run_name: str, shapes: str, extra: list[str], env_extra: dict) -> list[dict]:
    """Run benchmarks/run.py and parse the printed result table (semicolon CSV
    on stdout). run.py's temp CSV is auto-deleted, so we parse the table it prints.
    Returns a list of per-shape dict rows keyed by the table header columns.

    A background sampler watches for FOREIGN GPU processes (not in our subprocess
    tree) during the run; the max foreign MiB is recorded on each row dict via the
    module-level _LAST_FOREIGN so the caller can flag/retry a contaminated kernel.
    """
    global _LAST_FOREIGN
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "HELION_AUTOTUNE_EFFORT": "none",
        "PYTHONPATH": PR,
    })
    env.update(env_extra)
    seeded = env_extra.get("HELION_PROMOTE_REDUCTION_SEED") == "1"
    script = f"{PR}/_lab/bench/run_seeded.py" if seeded else f"{PR}/benchmarks/run.py"
    # --csv makes run.py print the semicolon-delimited table (header "(M, H);...")
    # to stdout, which we parse. The temp CSV file itself is auto-deleted.
    cmd = [PY, script, "--kernel", kernel_run_name,
           "--metrics", "latency,accuracy", "--precision", "fp32",
           "--shapes", shapes, "--csv", "--output-dir", "/tmp/sweep_out", *extra]
    proc = subprocess.Popen(cmd, cwd=PR, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    # contention sampler: any GPU PID not in our (proc) tree is foreign.
    foreign_max = {"mib": 0}
    stop = threading.Event()

    def _watch() -> None:
        while not stop.is_set():
            mine = _descendants(proc.pid) | {os.getpid()}
            for pid, mib in _gpu_pids().items():
                if pid not in mine:
                    foreign_max["mib"] = max(foreign_max["mib"], mib)
            time.sleep(1.0)

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    stdout, stderr = proc.communicate()
    stop.set()
    t.join(timeout=2)
    _LAST_FOREIGN = foreign_max["mib"]
    text = stdout + "\n" + stderr
    # The --csv table header is either "(M, H);..." (norm/softmax operators) or
    # "x_val;..." (sum operator). Data rows start with a shape tuple "(N, N);" or a
    # bare numeric x_val "N;"; the trailing "average;" row is skipped.
    header = None
    rows: list[dict] = []
    for line in text.splitlines():
        cells = line.split(";")
        if len(cells) < 3:
            continue
        first = cells[0].strip()
        if first in ("(M, H)", "x_val") or (
            first.startswith("(") and "latency" in line
        ):
            header = [c.strip() for c in cells]
        elif header and (
            re.match(r"^\(\d+,\s*\d+\)$", first) or re.match(r"^\d+$", first)
        ):
            rows.append(dict(zip(header, cells)))
    if not rows:
        sys.stderr.write(text[-2500:])
        raise RuntimeError(f"no result table parsed for {kernel_run_name}")
    return rows


def _num(x: str) -> float:
    m = re.match(r"\s*([0-9.]+)", x or "")
    return float(m.group(1)) if m else float("nan")


def _col(rows: list[dict], name: str) -> str | None:
    # exact match first (header cells are already stripped), then unique substring.
    keys = list(rows[0].keys())
    if name in keys:
        return name
    hits = [c for c in keys if name in c]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f"ambiguous column {name!r}: {hits}")
    return None


def bench_kernel(kernel: str, split: str) -> dict:
    run_name, hl_sub, tc_sub, extra = KCFG[kernel]
    shapes = _shapes_arg(kernel, split)
    seed_rows = _run(run_name, shapes, extra, {"HELION_PROMOTE_REDUCTION_SEED": "1"})
    foreign_seed = _LAST_FOREIGN
    def_rows = _run(run_name, shapes, extra, {"HELION_DISABLE_AUTOTUNER_HEURISTICS": "1"})
    foreign_def = _LAST_FOREIGN
    foreign_mib = max(foreign_seed, foreign_def)

    hl_s, tc_s = _col(seed_rows, hl_sub), _col(seed_rows, tc_sub)
    hl_d, tc_d = _col(def_rows, hl_sub), _col(def_rows, tc_sub)
    acc_s = _col(seed_rows, hl_sub.replace("-latency", "-accuracy"))
    acc_d = _col(def_rows, hl_sub.replace("-latency", "-accuracy"))
    shape_key = list(seed_rows[0].keys())[0]  # the "(M, H)" column (single cell)

    rows = []
    for rs, rd in zip(seed_rows, def_rows):
        if rs[shape_key].strip() == "average":
            continue
        sh = rs[shape_key].strip()
        s_hl, d_hl = _num(rs[hl_s]), _num(rd[hl_d])
        s_tc, d_tc = _num(rs[tc_s]), _num(rd[tc_d])
        # G = tc_default_latency / helion_latency, computed WITHIN each process
        # (ledger convention). G>=1 => helion matches/beats torch.compile-default.
        g_seed = s_tc / s_hl if s_hl else None
        g_default = d_tc / d_hl if d_hl else None
        rows.append({
            "shape": sh,
            "lat_seed_helion": s_hl, "lat_default_helion": d_hl,
            "lat_tc_seedproc": s_tc, "lat_tc_defproc": d_tc,
            "G_seed": round(g_seed, 4) if g_seed else None,
            "G_default": round(g_default, 4) if g_default else None,
            # headline: how much the seed lifts helion's tc-ratio (noise-robust,
            # within-process anchored). >1 => seed faster than the unseeded default.
            "seed_lift": round(g_seed / g_default, 4)
            if (g_seed and g_default) else None,
            "seed_vs_default_raw": round(d_hl / s_hl, 4) if s_hl else None,
            "acc_seed": rs.get(acc_s, "").strip() if acc_s else "?",
            "acc_default": rd.get(acc_d, "").strip() if acc_d else "?",
        })

    def geo(key):
        vals = [r[key] for r in rows if r[key] and r[key] == r[key] and r[key] > 0]
        return round(statistics.geometric_mean(vals), 4) if vals else None

    return {
        "kernel": kernel, "split": split, "rows": rows,
        "geo_G_seed": geo("G_seed"),
        "geo_G_default": geo("G_default"),
        "geo_seed_lift": geo("seed_lift"),
        "geo_seed_vs_default_raw": geo("seed_vs_default_raw"),
        "any_acc_fail": any(
            r["acc_seed"] not in ("1", "1.0", "?")
            or r["acc_default"] not in ("1", "1.0", "?")
            for r in rows
        ),
        "foreign_mib": foreign_mib,
        "contaminated": foreign_mib > FOREIGN_MIB_THRESHOLD,
    }


def main() -> None:
    kernels = sys.argv[1:] or list(KCFG)
    split = "test"
    os.makedirs("/tmp/sweep_out", exist_ok=True)
    results = []
    for k in kernels:
        sys.stderr.write(f"\n===== {k} ({split}) =====\n")
        r = None
        # Up to 3 attempts: retry if a foreign GPU process contaminated the run.
        for attempt in range(1, 4):
            try:
                r = bench_kernel(k, split)
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"  {k} FAILED (attempt {attempt}): {e}\n")
                r = {"kernel": k, "error": str(e)}
                break
            if not r["contaminated"]:
                break
            sys.stderr.write(
                f"  ⚠ CONTAMINATED (foreign {r['foreign_mib']} MiB on GPU, "
                f"attempt {attempt}) — re-running {k}\n"
            )
        if r is not None and "error" not in r:
            sys.stderr.write(
                f"  G_seed={r['geo_G_seed']}  G_default={r['geo_G_default']}  "
                f"seed_lift={r['geo_seed_lift']}  raw_seed/def={r['geo_seed_vs_default_raw']}"
                f"  acc_fail={r['any_acc_fail']}  foreign_mib={r['foreign_mib']}"
                f"  contaminated={r['contaminated']}\n"
            )
        results.append(r)
        # write incrementally so a mid-sweep kill never loses completed kernels
        with open("/tmp/sweep_out/results.json", "w") as f:
            json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
