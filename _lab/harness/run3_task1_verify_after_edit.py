"""Re-record task-1 seed configs on the CURRENT source, to a SEPARATE file, so
the committed baseline (task1_seed_configs.json) is never overwritten. Reuses
run3_task1_seed_configs.record_one verbatim (same bind / seed / normalize path).

  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_task1_verify_after_edit.py
"""

from __future__ import annotations

import json
import os
import sys
import traceback

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROMPTS_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "prompts"))
for _d in (_HARNESS_DIR, _PROMPTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import torch  # noqa: E402

import helion  # noqa: E402

from run3_task1_seed_configs import KERNEL_ORDER  # noqa: E402
from run3_task1_seed_configs import SPLITS  # noqa: E402
from run3_task1_seed_configs import record_one  # noqa: E402
from shapes_v3_draft import SHAPES  # noqa: E402

_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep), (
    f"helion ({helion.__file__}) not under harness worktree ({_WT_ROOT})"
)

OUT = os.path.join(_HARNESS_DIR, "..", "logs", "run3", "task1_seed_configs_AFTER.json")
OUT = os.path.abspath(OUT)


def main() -> None:
    print(f"helion={helion.__file__}", flush=True)
    print(f"out={OUT}\n", flush=True)
    rows = []
    n_ok = n_err = 0
    for kernel_name in KERNEL_ORDER:
        splits = SHAPES[kernel_name]
        for split in SPLITS:
            for (m, n) in splits[split]:
                tag = f"{kernel_name:14s} {split:5s} ({m},{n})"
                try:
                    rec = record_one(kernel_name, m, n)
                except Exception as e:  # noqa: BLE001
                    n_err += 1
                    print(f"[ERR ] {tag}: {type(e).__name__}: {e}", flush=True)
                    traceback.print_exc()
                    rows.append({"kernel": kernel_name, "split": split,
                                 "M": m, "N": n,
                                 "error": f"{type(e).__name__}: {e}"})
                    torch.cuda.empty_cache()
                    continue
                rec["split"] = split
                rows.append(rec)
                n_ok += 1
                json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print(f"\n=== DONE: {n_ok} recorded, {n_err} errored ===", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
