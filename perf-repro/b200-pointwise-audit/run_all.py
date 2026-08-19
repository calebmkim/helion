"""Resumable process-isolated driver for the B200 pointwise audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from matrix import SPEC_BY_NAME
from matrix import iter_cells

AUDIT_DIR = Path(__file__).resolve().parent
ROOT = AUDIT_DIR.parents[1]


def _parse_kernels(value: str) -> set[str] | None:
    if not value:
        return None
    kernels = {item.strip() for item in value.split(",") if item.strip()}
    unknown = kernels - set(SPEC_BY_NAME)
    if unknown:
        raise ValueError(f"unknown kernels: {sorted(unknown)}")
    return kernels


def _is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return "arms" in row or "fatal_error" in row


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--kernels", default="")
    parser.add_argument("--results-dir", default="")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("run_all.py requires CUDA_VISIBLE_DEVICES=1")

    kernels = _parse_kernels(args.kernels)
    cells = iter_cells(kernels)
    if args.smoke:
        cells = [cell for cell in cells if cell[1] == 0]

    default_name = "smoke" if args.smoke else "full"
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else AUDIT_DIR / "results" / default_name
    )
    raw_dir = results_dir / "raw"
    log_dir = results_dir / "logs"
    cache_dir = results_dir / "helion-cache"
    for directory in (raw_dir, log_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env["HELION_AUTOTUNE_EFFORT"] = "none"
    env["HELION_CACHE_DIR"] = str(cache_dir)
    if args.smoke:
        env.update(
            {
                "AUDIT_ROUNDS": "2",
                "AUDIT_ROUNDS_HIGH": "2",
                "AUDIT_COLD_BATCH": "4",
                "AUDIT_COLD_ITERS": "3",
                "AUDIT_DYNAMO_EXPLAIN": "1",
            }
        )

    failures = 0
    started = time.time()
    for ordinal, (spec, shape_index, shape) in enumerate(cells, start=1):
        stem = f"{spec.cohort}__{spec.kernel}__{shape_index:02d}"
        output = raw_dir / f"{stem}.json"
        log = log_dir / f"{stem}.log"
        if not args.force and _is_complete(output):
            print(
                f"[{ordinal:03}/{len(cells):03}] skip {spec.kernel} {shape}",
                flush=True,
            )
            continue
        command = [
            sys.executable,
            str(AUDIT_DIR / "bench.py"),
            "--kernel",
            spec.kernel,
            "--shape-index",
            str(shape_index),
            "--output",
            str(output),
        ]
        print(
            f"[{ordinal:03}/{len(cells):03}] run  {spec.kernel} {shape}",
            flush=True,
        )
        begin = time.time()
        try:
            with log.open("w") as stream:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
                try:
                    returncode = process.wait(timeout=args.timeout_seconds)
                except BaseException:
                    _terminate_process_group(process)
                    raise
            elapsed = time.time() - begin
            if returncode:
                failures += 1
                print(
                    f"  FAIL rc={returncode} elapsed={elapsed:.1f}s log={log}",
                    flush=True,
                )
            else:
                print(f"  done elapsed={elapsed:.1f}s", flush=True)
        except subprocess.TimeoutExpired:
            failures += 1
            elapsed = time.time() - begin
            output.write_text(
                json.dumps(
                    {
                        "cohort": spec.cohort,
                        "kernel": spec.kernel,
                        "shape_index": shape_index,
                        "shape": list(shape),
                        "dtype": spec.dtype,
                        "fatal_error": (f"timeout after {args.timeout_seconds}s"),
                    },
                    indent=2,
                )
                + "\n"
            )
            print(
                f"  TIMEOUT elapsed={elapsed:.1f}s log={log}",
                flush=True,
            )

    elapsed = time.time() - started
    print(
        f"completed {len(cells)} scheduled cells in {elapsed / 60:.1f} min; "
        f"process failures={failures}; results={results_dir}",
        flush=True,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
