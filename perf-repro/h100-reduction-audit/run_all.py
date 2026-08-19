#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from workloads import all_kernel_names
from workloads import ROOT


AUDIT_DIR = Path(__file__).resolve().parent


def _run(command: list[str], env: dict[str, str]) -> int:
    print("+ " + " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "full"))
    parser.add_argument(
        "--results-dir",
        default=str(AUDIT_DIR / "results"),
    )
    parser.add_argument(
        "--kernels",
        default=",".join(all_kernel_names()),
        help="comma-separated kernel names",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    kernels = [name.strip() for name in args.kernels.split(",") if name.strip()]
    unknown = sorted(set(kernels) - set(all_kernel_names()))
    if unknown:
        raise SystemExit(f"unknown kernels: {unknown}")

    results_dir = Path(args.results_dir)
    raw_dir = results_dir / ("smoke_raw" if args.mode == "smoke" else "raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HELION_AUTOTUNE_EFFORT"] = "none"
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    manifest = {
        "mode": args.mode,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "kernels": kernels,
        "raw_dir": str(raw_dir),
        "resume": args.resume,
        "commands": [],
    }
    started = time.perf_counter()
    failures: list[dict[str, object]] = []
    for kernel in kernels:
        command = [
            sys.executable,
            str(AUDIT_DIR / "audit.py"),
            "--kernel",
            kernel,
            "--out-dir",
            str(raw_dir),
        ]
        if args.mode == "smoke":
            command.append("--smoke")
        if args.resume:
            command.append("--resume")
        manifest["commands"].append(command)
        returncode = _run(command, env)
        if returncode:
            failures.append({"kernel": kernel, "returncode": returncode})

    manifest.update(
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "process_failures": failures,
        }
    )
    manifest_path = results_dir / f"{args.mode}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}", flush=True)

    if args.mode == "full":
        returncode = _run(
            [
                sys.executable,
                str(AUDIT_DIR / "aggregate.py"),
                "--raw-dir",
                str(raw_dir),
                "--out-dir",
                str(results_dir),
            ],
            env,
        )
        if returncode:
            failures.append({"kernel": "aggregate", "returncode": returncode})

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
