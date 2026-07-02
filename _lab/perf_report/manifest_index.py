"""Render a config_manifest.json into a compact, grep-friendly one-line-per-cell text index.

Usage: manifest_index.py <config_manifest.json> [> CONFIG_MANIFEST.txt]
"""
import json
import sys


def cfg_short(c):
    if not c:
        return "None"
    keep = ("block_sizes", "reduction_loops", "num_warps", "num_stages", "pid_type",
            "indexing", "load_eviction_policies")
    parts = []
    for k in keep:
        if k not in c:
            continue
        v = c[k]
        # collapse the all-'pointer' / all-'' vectors that clutter the line
        if k == "indexing" and isinstance(v, list) and len(set(v)) <= 1:
            v = f"[{v[0]}x{len(v)}]" if v else "[]"
        if k == "load_eviction_policies" and isinstance(v, list) and set(v) <= {""}:
            continue
        parts.append(f"{k.replace('_sizes','').replace('reduction_loops','rl').replace('num_warps','w').replace('num_stages','st').replace('pid_type','pid')}={v}")
    return " ".join(parts)


def main():
    path = sys.argv[1]
    d = json.load(open(path))
    rows = d["rows"]
    for r in rows:
        key = f"{r['corpus']}/{r['kernel']}/{r.get('dtype')}/{r.get('shape')}"
        if "error" in r:
            print(f"{key:70s} ERROR: {r['error'][:60]}")
            continue
        fired = ",".join(r.get("fired_heuristics") or []) or "-none-"
        differ = "DIFF" if r.get("configs_differ") else "same"
        print(f"{key:70s} [{r.get('classification','?'):14s}] {differ} "
              f"fired={fired:26s} SEED: {cfg_short(r.get('normalized_seed'))}")


if __name__ == "__main__":
    main()
