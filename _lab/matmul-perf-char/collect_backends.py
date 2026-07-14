"""Collect the torch.compile(max-autotune) SELECTED backend for the tc arm of every
cell (§1/§4/§5(d)/§7 'log the selected backend per shape'), then merge `winner` +
`winner_kind` into the tc_max_autotune arm of every record in the JSONL.

Runs the mmperf.py `backend` subcommand once per unique (kernel,shape,dtype), FRESH
inductor cache per probe (else cache-hit skips autotune -> no table), foreground.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
MMPERF = os.path.join(HERE, "mmperf.py")
_DEFAULT_WT = os.path.dirname(os.path.dirname(HERE))
WORKTREE = os.environ.get("MMPERF_WORKTREE", _DEFAULT_WT)


def main():
    jsonl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "h100_results.jsonl")
    recs = [json.loads(l) for l in open(jsonl)]
    cells = {}
    for r in recs:
        cells[(r["kernel"], tuple(r["shape"]), r["dtype"])] = r["shape"]

    # resumable: cache winners to disk so a timeout doesn't lose the expensive probes
    cache_path = os.path.join(HERE, ".backend_winners.json")
    winners = {}
    if os.path.exists(cache_path):
        for k, v in json.load(open(cache_path)).items():
            parts = k.split("|")
            winners[(parts[0], tuple(int(x) for x in parts[1].split(",")), parts[2])] = tuple(v)

    def save_cache():
        out = {}
        for (kn, sh, dt), v in winners.items():
            out[f"{kn}|{','.join(str(x) for x in sh)}|{dt}"] = list(v)
        json.dump(out, open(cache_path, "w"))

    tmp = "/tmp/backend_probe.json"
    for i, (kernel, shape, dtype) in enumerate(sorted(cells)):
        if (kernel, shape, dtype) in winners and winners[(kernel, shape, dtype)][0] not in (
                None, "probe_error", "unknown (no 100.0% line parsed)"):
            print(f"[{i+1}/{len(cells)}] {kernel} {list(shape)} {dtype}: "
                  f"CACHED {winners[(kernel, shape, dtype)]}", flush=True)
            continue
        shape_str = ",".join(str(x) for x in shape)
        env = dict(os.environ)
        env["MMPERF_WORKTREE"] = WORKTREE
        env.setdefault("PYTHONPATH", WORKTREE)
        env["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/ind_bk_{int(time.time()*1e6)}_{i}"
        env["TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC"] = "0"
        if os.path.exists(tmp):
            os.remove(tmp)
        proc = subprocess.run(
            [PY, MMPERF, "backend", "--kernel", kernel, "--shape", shape_str,
             "--dtype", dtype, "--out", tmp],
            env=env, capture_output=True, text=True, timeout=300)
        try:
            d = json.load(open(tmp))
            w, k = d.get("winner"), d.get("winner_kind")
        except Exception:
            w, k = "probe_error", None
        winners[(kernel, shape, dtype)] = (w, k)
        save_cache()
        print(f"[{i+1}/{len(cells)}] {kernel} {list(shape)} {dtype}: winner={w} kind={k}", flush=True)

    # merge into records
    for r in recs:
        key = (r["kernel"], tuple(r["shape"]), r["dtype"])
        w, k = winners.get(key, (None, None))
        tc = r["arms"].get("tc_max_autotune")
        if tc is not None:
            tc["winner"] = w
            tc["winner_kind"] = k

    with open(jsonl, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, default=str) + "\n")
    print(f"\nmerged winners into {len(recs)} records in {jsonl}")


if __name__ == "__main__":
    main()
