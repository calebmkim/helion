"""RUN-3 oracle-cache RE-KEY (ledger-keeper recipe correction; NO do_bench).

Batch-1 entries were keyed with the OLD recipe (which WRONGLY included the heuristic
triton.py + __init__/registry). The ledger-keeper recipe DROPS the heuristic — the
oracle is what the SEARCH finds, independent of the seed. This re-stamps every cached
entry's ``source_hash`` under the CORRECTED recipe (run3_oracle.source_hash):
    sha256( read(examples/<kernel>.py)
            + to_triton_code(default_config for that shape)   # codegen
            + repr(config_spec knob/range dump) )             # search space
Records old_source_hash -> source_hash per entry + the recipe id, so the ledger-keeper
can audit. Uses bind + to_triton_code (CODEGEN only — no do_bench, no kernel launch),
so it does NOT enter the one-GPU timing queue.

Run from /tmp with the canonical wiring.
"""

from __future__ import annotations

import json
import os
import sys

import torch

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import helion  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run3_oracle", os.path.join(_HARNESS_DIR, "run3_oracle.py")
)
run3_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run3_oracle)

from run2_measure_g import KERNELS  # noqa: E402

CACHE_PATH = run3_oracle.CACHE_PATH
RECIPE_ID = "ledger_v2_2026-06-03: sha256(src + to_triton_code(default_cfg) + knobdump); heuristic EXCLUDED"


def main():
    with open(CACHE_PATH) as f:
        cache = json.load(f)
    entries = cache["entries"]
    print(f"Re-keying {len(entries)} cache entries under the corrected recipe.\n", flush=True)
    changed = 0
    for key, e in entries.items():
        kernel = e["kernel"]
        M, N = e["shape"]
        fn = KERNELS[kernel][0]
        builder = KERNELS[kernel][1]
        args, _, _ = builder(M, N)
        bk = helion.kernel(fn.fn).bind(args)
        new_hash = run3_oracle.source_hash(kernel, bk)
        old_hash = e.get("source_hash")
        if new_hash != old_hash:
            e["old_source_hash"] = old_hash
            e["source_hash"] = new_hash
            e["source_hash_recipe"] = RECIPE_ID
            changed += 1
        print(f"  {key}: {old_hash} -> {new_hash}"
              f"{'  (unchanged)' if new_hash == old_hash else ''}", flush=True)
        del args
        torch.cuda.empty_cache()
    cache["source_hash_recipe"] = RECIPE_ID
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, default=str)
    print(f"\nRe-keyed {changed}/{len(entries)} entries (recipe={RECIPE_ID}).", flush=True)
    print(f"[wrote {CACHE_PATH}]", flush=True)


if __name__ == "__main__":
    main()
