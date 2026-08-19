#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any


os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

from audit import _extract_configs  # noqa: E402
from audit import _select_aot  # noqa: E402
from audit import config_dict  # noqa: E402
from workloads import build_workload  # noqa: E402


AUDIT_DIR = Path(__file__).resolve().parent
RESOURCE_PATTERN = re.compile(
    r"Function (?P<name>\S+):\s+"
    r"REG:(?P<registers>\d+)\s+"
    r"STACK:(?P<stack_bytes>\d+)\s+"
    r"SHARED:(?P<shared_bytes>\d+)\s+"
    r"LOCAL:(?P<local_bytes>\d+)"
)
SPILL_PATTERN = re.compile(
    r"(?P<stack_bytes>\d+) bytes stack frame, "
    r"(?P<spill_store_bytes>\d+) bytes spill stores, "
    r"(?P<spill_load_bytes>\d+) bytes spill loads"
)
REGISTER_PATTERN = re.compile(r"Used (?P<registers>\d+) registers")
PTX_VERSION_PATTERN = re.compile(r"^\.version (?P<version>\d+\.\d+)$", re.MULTILINE)


def _load_rows(raw_dir: Path, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                ratio = row.get("ratios", {}).get("G_aot")
                if ratio is not None and ratio < threshold:
                    rows.append(row)
    rows.sort(key=lambda row: row["ratios"]["G_aot"])
    return rows


def _artifact(cache_dir: Path, suffix: str) -> Path:
    matches = [
        path
        for path in cache_dir.glob(f"*{suffix}")
        if not path.name.startswith("__grp__")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {suffix} artifact in {cache_dir}, found {matches}"
        )
    return matches[0]


def _run(command: list[str]) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.stdout + result.stderr


def _resource_usage(cubin: Path) -> tuple[list[dict[str, Any]], str]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        raise RuntimeError("cuobjdump is unavailable")
    output = _run([cuobjdump, "--dump-resource-usage", str(cubin)])
    resources = [
        {
            key: int(value) if key != "name" else value
            for key, value in match.groupdict().items()
        }
        for match in RESOURCE_PATTERN.finditer(output)
    ]
    if not resources:
        raise RuntimeError(f"could not parse cuobjdump output for {cubin}")
    return resources, output


def _ptxas_candidates(ptx: Path) -> list[Path]:
    from triton.backends.nvidia.compiler import get_ptxas

    match = PTX_VERSION_PATTERN.search(ptx.read_text())
    if match is None:
        raise RuntimeError(f"could not read PTX version from {ptx}")
    ptx_version = float(match.group("version"))
    triton_ptxas = Path(get_ptxas(90).path)
    torch_ptxas = Path(torch.__file__).resolve().parent / "bin" / "ptxas"
    candidates = (
        [torch_ptxas, triton_ptxas]
        if ptx_version >= 9.0
        else [triton_ptxas, torch_ptxas]
    )
    fallback = shutil.which("ptxas")
    if fallback is not None:
        candidates.append(Path(fallback))
    return list(dict.fromkeys(path for path in candidates if path.is_file()))


def _ptxas_usage(ptx: Path) -> tuple[dict[str, Any], str]:
    outputs: list[str] = []
    with tempfile.TemporaryDirectory(prefix="h100_audit_ptxas_") as tmp:
        for ptxas in _ptxas_candidates(ptx):
            result = subprocess.run(
                [
                    str(ptxas),
                    "-v",
                    "--gpu-name",
                    "sm_90a",
                    str(ptx),
                    "-o",
                    str(Path(tmp) / "kernel.cubin"),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = result.stdout + result.stderr
            outputs.append(f"$ {ptxas}\n{output}")
            if result.returncode == 0:
                break
        else:
            raise RuntimeError(
                f"no compatible ptxas for {ptx}:\n" + "\n".join(outputs)
            )
    spill = SPILL_PATTERN.search(output)
    registers = REGISTER_PATTERN.search(output)
    if spill is None or registers is None:
        raise RuntimeError(f"could not parse ptxas output for {ptx}")
    usage: dict[str, Any] = {
        key: int(value) for key, value in spill.groupdict().items()
    }
    usage["registers"] = int(registers.group("registers"))
    usage["binary"] = str(ptxas)
    return usage, outputs[-1]


def _compile_arm(
    workload: Any,
    name: str,
    config: Any,
    code_dir: Path,
) -> dict[str, Any]:
    state = workload.make_state()
    state.restore()
    bound = workload.kernel_fn.bind(state.args)
    emitted_code = bound.to_triton_code(config)
    code_path = code_dir / f"{name}.py"
    code_path.write_text(emitted_code)
    compiled = bound.compile_config(config, allow_print=False)
    output = compiled(*state.args)
    torch.cuda.synchronize()
    del output

    cache_key = bound.backend_cache_key(config)
    if cache_key is None:
        raise RuntimeError(f"no Triton cache key for {workload.kernel} {name}")
    cache_root = Path(os.environ["TRITON_CACHE_DIR"])
    cache_dir = cache_root / cache_key
    metadata_path = _artifact(cache_dir, ".json")
    ptx_path = _artifact(cache_dir, ".ptx")
    cubin_path = _artifact(cache_dir, ".cubin")
    metadata = json.loads(metadata_path.read_text())
    resources, cuobjdump_output = _resource_usage(cubin_path)
    ptxas, ptxas_output = _ptxas_usage(ptx_path)
    saved_ptx = code_dir / f"{name}.ptx"
    shutil.copy2(ptx_path, saved_ptx)

    result = {
        "config": config_dict(config),
        "backend_cache_key": cache_key,
        "emitted_code": {
            "path": str(code_path),
            "sha256": hashlib.sha256(emitted_code.encode()).hexdigest(),
            "lines": len(emitted_code.splitlines()),
            "bytes": len(emitted_code.encode()),
        },
        "ptx": {
            "path": str(saved_ptx),
            "sha256": hashlib.sha256(saved_ptx.read_bytes()).hexdigest(),
            "bytes": saved_ptx.stat().st_size,
        },
        "triton_metadata": metadata,
        "cuobjdump_resources": resources,
        "cuobjdump_output": cuobjdump_output,
        "ptxas": ptxas,
        "ptxas_output": ptxas_output,
    }
    del compiled, bound, state
    return result


def _compact_config(config: dict[str, Any]) -> str:
    return (
        f"b={config.get('block_sizes')};"
        f"r={config.get('reduction_loops')};"
        f"w={config.get('num_warps')};"
        f"pid={config.get('pid_type')}"
    )


def _resource_cell(arm: dict[str, Any]) -> str:
    ptxas = arm["ptxas"]
    resources = arm["cuobjdump_resources"]
    local_bytes = max(item["local_bytes"] for item in resources)
    shared_bytes = max(item["shared_bytes"] for item in resources)
    return (
        f"reg={ptxas['registers']}; "
        f"spill={ptxas['spill_store_bytes']}/{ptxas['spill_load_bytes']} B; "
        f"stack={ptxas['stack_bytes']} B; local={local_bytes} B; "
        f"shared={shared_bytes} B"
    )


def _report(cases: list[dict[str, Any]], threshold: float) -> str:
    lines = [
        "# H100 Material AOT-Loss Inspection",
        "",
        "Post-run inspection only. Headline measurements were not modified.",
        "",
        f"Included cells have `G_aot < {threshold:.3f}`; lower ratios mean a "
        "larger checked-in H100 AOT advantage.",
        "",
        "| Kernel | Shape | G_aot | Seed config | AOT config | Seed resources | "
        "AOT resources |",
        "|---|---|---:|---|---|---|---|",
    ]
    for case in cases:
        seed = case["arms"]["seed"]
        aot = case["arms"]["aot_sm90"]
        lines.append(
            f"| {case['kernel']} | `{tuple(case['shape'])}` | "
            f"{case['G_aot']:.3f} | `{_compact_config(seed['config'])}` | "
            f"`{_compact_config(aot['config'])}` | "
            f"{_resource_cell(seed)} | {_resource_cell(aot)} |"
        )
    lines.extend(
        [
            "",
            "The adjacent `investigation_codegen/` directory contains the emitted "
            "Python/Triton launchers and PTX for every arm in this table. "
            "`investigation.json` contains full configs, cache keys, Triton metadata, "
            "raw `ptxas -v` output, and raw `cuobjdump --dump-resource-usage` output.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default=str(AUDIT_DIR / "results" / "raw"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(AUDIT_DIR / "results"),
    )
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise SystemExit("CUDA_VISIBLE_DEVICES must be exactly 0")
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    code_root = out_dir / "investigation_codegen"
    code_root.mkdir(parents=True, exist_ok=True)
    selected = _load_rows(raw_dir, args.threshold)
    cases: list[dict[str, Any]] = []
    for row in selected:
        kernel = row["kernel"]
        shape = tuple(row["shape"])
        print(f"[inspect] {kernel} {shape}", flush=True)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        workload = build_workload(kernel, shape)
        seed_config, _, _, _ = _extract_configs(workload)
        aot_config, aot_selector = _select_aot(workload)
        case_dir = code_root / (
            kernel + "__" + "x".join(str(value) for value in shape)
        )
        case_dir.mkdir(parents=True, exist_ok=True)
        arms = {
            "seed": _compile_arm(workload, "seed", seed_config, case_dir),
            "aot_sm90": _compile_arm(
                workload, "aot_sm90", aot_config, case_dir
            ),
        }
        cases.append(
            {
                "kernel": kernel,
                "shape": list(shape),
                "G_aot": row["ratios"]["G_aot"],
                "seed_us": row["arms"]["seed"]["timing"]["median_us"],
                "aot_sm90_us": row["arms"]["aot_sm90"]["timing"]["median_us"],
                "aot_selector": aot_selector,
                "arms": arms,
            }
        )
        del workload
        gc.collect()
        torch.cuda.empty_cache()

    document = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dir": str(raw_dir),
        "selection": {"metric": "G_aot", "threshold_lt": args.threshold},
        "cases": cases,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "investigation.json"
    report_path = out_dir / "INVESTIGATION.md"
    json_path.write_text(json.dumps(document, indent=2) + "\n")
    report_path.write_text(_report(cases, args.threshold))
    print(f"wrote {json_path} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
