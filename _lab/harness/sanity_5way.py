"""5-way rms_norm sanity driver (Step 1, harness-integrity).

Runs benchmarks/run.py at the three Helion effort levels (none/quick/full) and
parses the per-variant tritonbench CSV to extract, for each shape:

  Helion-default (effort=none), Helion-quick, Helion-max (full),
  torch.compile-default (new variant), torch.compile-max (existing).

tc-* variants do not depend on HELION_AUTOTUNE_EFFORT, so we read them from the
effort=none run (they are identical across runs up to noise).

Emits a clean ms + speedup-vs-eager table. fp32 asserted via --precision fp32 and
the operator's tensor dtype (the JSON metadata prints dtype=torch.float32).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

WT = "/home/calebkim/helion-new-heuristics/wt-reduction"
PY = "/home/calebkim/.conda/envs/helion/bin/python"

# variant column prefixes in the tritonbench CSV
HELION = "helion_rms_norm_tritonbench"
TC_DEFAULT = "torch_compile_rms_norm_default"
TC_MAX = "torch_compile_rms"
EAGER = "llama_rms"  # baseline=True


def run_one(effort: str, m: int, h: int) -> dict[str, dict[str, float]]:
    """Run run.py at one effort/shape; parse the stdout table.

    The tritonbench temp CSV is deleted on process exit, so we parse the
    tabulated stdout instead (deterministic, whitespace-separated)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PYTHONPATH"] = WT
    env["HELION_AUTOTUNE_EFFORT"] = effort
    cmd = [
        PY,
        "benchmarks/run.py",
        "--kernel",
        "rms_norm",
        "--metrics",
        "latency,accuracy,speedup",
        "--precision",
        "fp32",
        "--M",
        str(m),
        "--H",
        str(h),
    ]
    p = subprocess.run(
        cmd, cwd=WT, env=env, capture_output=True, text=True, timeout=1800
    )
    # assert fp32 actually used (dtype in the input-metadata warning on stderr)
    assert "torch.float32" in p.stderr, "fp32 not asserted in run!"
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-2000:])
        sys.stderr.write(p.stderr[-2000:])
        raise RuntimeError(f"run.py failed effort={effort} ({m},{h})")
    return parse_stdout_table(p.stdout)


def parse_stdout_table(stdout: str) -> dict[str, dict[str, float]]:
    """Parse the tabulated run.py stdout into {variant: {metric: value}}."""
    lines = stdout.splitlines()
    header_line = None
    data_line = None
    for i, ln in enumerate(lines):
        if "-latency" in ln and ("(M" in ln or "x_val" in ln):
            header_line = ln
            # next non-separator line that starts with a shape tuple is the data
            for j in range(i + 1, len(lines)):
                if lines[j].strip().startswith("("):
                    data_line = lines[j]
                    break
            break
    assert header_line and data_line, "could not find table in stdout"
    headers = header_line.split()
    # the data line has "(M, H)" as first 2 whitespace tokens, then values
    # which may include "(±x%)" parentheses -> regroup by stripping those.
    # Easiest: drop "(±...%)" tokens, keep numeric tokens aligned to headers.
    data_clean = re.sub(r"\(±[^)]*\)", "", data_line)
    toks = data_clean.split()
    # first token is the shape "(M," , second "H)" -> the x_val column.
    # headers[0] is "(M," headers[1] is "H)"? No: header is "(M, H)" split into
    # "(M," and "H)". Align: skip first 2 header tokens + first 2 data tokens.
    h_vals = headers[2:]
    d_vals = toks[2:]
    out: dict[str, dict[str, float]] = {}
    for col, val in zip(h_vals, d_vals):
        mt = re.match(r"(.+)-(latency|accuracy|speedup)$", col.strip())
        if not mt:
            continue
        variant, metric = mt.group(1), mt.group(2)
        try:
            out.setdefault(variant, {})[metric] = float(val)
        except ValueError:
            pass
    return out


def main() -> None:
    shapes = [(4096, 8192), (8192, 8192)]
    results: dict[tuple, dict] = {}
    for (m, h) in shapes:
        per_effort = {}
        for effort in ("none", "quick", "full"):
            per_effort[effort] = run_one(effort, m, h)
            sys.stderr.write(f"done effort={effort} shape=({m},{h})\n")
        results[(m, h)] = per_effort

    # build the table
    print("\n=== 5-WAY rms_norm SANITY (fp32, GPU2, do_bench median) ===")
    header = (
        f"{'shape':>13} | {'variant':<22} | {'latency_ms':>11} | "
        f"{'speedup_vs_eager':>16} | {'accuracy':>8}"
    )
    print(header)
    print("-" * len(header))
    for (m, h) in shapes:
        pe = results[(m, h)]
        eager_lat = pe["none"][EAGER]["latency"]
        rows = [
            ("Helion-default", pe["none"][HELION]),
            ("Helion-quick", pe["quick"][HELION]),
            ("Helion-max(full)", pe["full"][HELION]),
            ("tc-default", pe["none"][TC_DEFAULT]),
            ("tc-max", pe["none"][TC_MAX]),
            ("eager(llama_rms)", pe["none"][EAGER]),
        ]
        for name, d in rows:
            lat = d.get("latency", float("nan"))
            spd = eager_lat / lat if lat else float("nan")
            acc = d.get("accuracy", float("nan"))
            print(
                f"{f'({m},{h})':>13} | {name:<22} | {lat:>11.5f} | "
                f"{spd:>16.3f} | {acc:>8.3f}"
            )
        print("-" * len(header))


if __name__ == "__main__":
    main()
