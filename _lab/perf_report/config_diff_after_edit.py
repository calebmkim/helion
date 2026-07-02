"""Derive the AFFECTED SET after the NARROW_W1 removal: re-extract every required cell's seed
config (post-edit) and diff against the seed_config stored in results/*.json (pre-edit).

Bind-only (no timing) so it's cheap. Prints the cells whose normalized seed config CHANGED —
those are exactly the ones to re-bench. Any change outside the expected w1->ramp set is a
red flag to investigate before trusting a targeted rebench.

Run (from /tmp):
  HELION_AUTOTUNE_EFFORT=none PYTHONPATH=/home/dev/local/helion-redesign \
    /home/dev/helion/.venv/bin/python .../config_diff_after_edit.py --results <RESULTS_DIR>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import perf_report_bench as PB  # noqa: E402
import torch  # noqa: E402


def _after_seed_cfg(corpus, kernel, shape, dtype):
    """Bind the cell post-edit and return the normalized seed config dict (or None)."""
    dt = PB._DT.get(dtype)
    if corpus == "curriculum":
        kfn, args, *_ = PB._cur_build(kernel, shape[0], shape[1], dt)
    elif corpus == "transfer":
        kfn, args, *_ = PB._transfer_build(kernel, shape, dt)
    elif corpus == "mreduction":
        kfn, args, *_ = PB._mred_build(kernel, tuple(shape), dt)
    elif corpus == "vllm":
        import bench_arms as B
        import importlib
        tok, hidden = shape[0], shape[1]
        group = shape[2] if len(shape) > 2 else None
        mod_name, kern_attr, builder, _s, _k = B.SPECS[kernel]
        mod = importlib.import_module(mod_name)
        kfn = getattr(mod, kern_attr)
        args = builder(tok, hidden, group)[0]
    else:
        raise KeyError(corpus)
    seed_cfg, _base, _fired = PB._extract_configs(kfn, args)
    PB._cleanup()
    return PB._cfg_dict(seed_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    changed = []
    same = 0
    errors = []
    for path in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        base = os.path.basename(path)
        if base in ("summary.json", "narrow_w1_ab.json"):
            continue
        corpus_kernel = base[:-5]  # strip .json
        d = json.load(open(path))
        for r in d.get("rows", []):
            if "arms" not in r or "seed_config" not in r:
                continue
            corpus, kernel, shape, dtype = r["corpus"], r["kernel"], r.get("shape"), r.get("dtype")
            before = r["seed_config"]
            try:
                after = _after_seed_cfg(corpus, kernel, shape, dtype)
            except Exception as e:  # noqa: BLE001
                errors.append((corpus, kernel, shape, dtype, f"{type(e).__name__}: {e}"))
                continue
            if before != after:
                bw = (before or {}).get("num_warps")
                aw = (after or {}).get("num_warps")
                # only-num_warps-changed?
                only_warps = (before and after
                              and {k: v for k, v in before.items() if k != "num_warps"}
                              == {k: v for k, v in after.items() if k != "num_warps"})
                changed.append({"corpus": corpus, "kernel": kernel, "shape": shape,
                                "dtype": dtype, "before_warps": bw, "after_warps": aw,
                                "only_num_warps_changed": only_warps,
                                "before": before, "after": after})
            else:
                same += 1
    print(f"CHANGED {len(changed)} cells; {same} unchanged; {len(errors)} bind-errors\n")
    for c in changed:
        flag = "" if c["only_num_warps_changed"] else "  <<< NON-WARP CHANGE!"
        print(f"  {c['corpus']}/{c['kernel']} {c['shape']} {c['dtype']}: "
              f"w{c['before_warps']} -> w{c['after_warps']}{flag}")
    if errors:
        print(f"\nbind-errors ({len(errors)}):")
        for e in errors:
            print(f"  {e[0]}/{e[1]} {e[2]} {e[3]}: {e[4][:80]}")
    if a.out:
        json.dump({"changed": changed, "n_same": same, "errors": errors},
                  open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    # exit nonzero if any NON-warp change (a surprise that invalidates the targeted-rebench assumption)
    surprises = [c for c in changed if not c["only_num_warps_changed"]]
    if surprises:
        print(f"\nWARNING: {len(surprises)} cells changed a field OTHER than num_warps.")


if __name__ == "__main__":
    main()
