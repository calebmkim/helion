"""RUN-3 — RE-STAMP oracle-cache source_hashes to the new DETERMINISTIC recipe.

Hub/ledger-keeper-approved (option ii): the source_hash recipe was made cross-process
deterministic (component (c) now skips *_facts + seed/heuristic + address-reprs). The
cached oracle LATENCIES are unchanged/valid (the fix only realigns the KEY) — re-stamp
each entry's `source_hash` to the new recipe value, touching NOTHING else.

AUDIT (hub conditions): prints per-entry old->new hash, and ASSERTS the only field that
changes is `source_hash` (a deep-equality check of the entry minus source_hash before/after).
Bind-only (no do_bench / no GPU token). Writes the cache back only if --apply is passed.

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_restamp_oracle_hashes.py [--apply]
"""

from __future__ import annotations

import copy
import json
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

from run2_measure_g import KERNELS  # noqa: E402
from run3_oracle import source_hash  # noqa: E402

import helion  # noqa: E402

CACHE = os.path.abspath(
    os.path.join(_HARNESS_DIR, "..", "logs", "run3", "oracle_cache.json")
)
# IN-SCOPE only: welford is another agent's; don't re-bind/re-stamp its entries.
INSCOPE = {
    "rms_norm", "layer_norm", "sum", "long_sum", "cross_entropy",
    "softmax", "kl_div", "jsd",
}


def main() -> None:
    apply = "--apply" in sys.argv
    blob = json.load(open(CACHE))
    entries = blob["entries"]
    print(f"helion={helion.__file__}  apply={apply}\n", flush=True)

    changed = 0
    skipped_oos = 0
    err = 0
    for key in sorted(entries):
        ent = entries[key]
        kn = ent["kernel"]
        if kn not in INSCOPE:
            print(f"  -- skip {key} (out of scope: {kn}; leaving its hash as-is)",
                  flush=True)
            skipped_oos += 1
            continue
        M, N = ent["shape"]
        old_hash = ent.get("source_hash")
        try:
            fn, builder, _ = KERNELS[kn]
            args, _, _ = builder(M, N)
            b = fn.bind(args)
            new_hash = source_hash(kn, b)
        except Exception as e:
            print(f"  !! {key}: bind/hash ERR {type(e).__name__}: {e}"[:160],
                  flush=True)
            err += 1
            continue
        # AUDIT: prove ONLY source_hash changes — deep-compare entry minus source_hash.
        before = copy.deepcopy(ent)
        before.pop("source_hash", None)
        after = copy.deepcopy(ent)
        after["source_hash"] = new_hash
        after.pop("source_hash", None)
        assert before == after, f"{key}: NON-HASH FIELD CHANGED — abort"
        status = "SAME" if old_hash == new_hash else "CHANGED"
        print(f"  {key:28s} {old_hash} -> {new_hash}  [{status}] "
              f"(oracle_us={ent.get('oracle_us')}, untouched)", flush=True)
        if old_hash != new_hash:
            changed += 1
        if apply:
            ent["source_hash"] = new_hash

    print(f"\nSUMMARY: {changed} hashes CHANGED, {skipped_oos} out-of-scope skipped, "
          f"{err} errors. apply={apply}", flush=True)
    if apply and err == 0:
        blob["updated"] = blob.get("updated", "") + " | hashes re-stamped (det recipe)"
        with open(CACHE, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"WROTE {CACHE} (source_hash fields only; latencies untouched)",
              flush=True)
    elif apply:
        print("NOT writing — errors present; resolve first.", flush=True)
    else:
        print("DRY-RUN (no --apply). Re-run with --apply to write.", flush=True)


if __name__ == "__main__":
    main()
