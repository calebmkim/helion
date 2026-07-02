"""One-shot kernel launch for ncu profiling: build a curriculum kernel at (M,N) pinned to a
config, run it ONCE (warmup already done by helion compile), exit. ncu wraps this.

Usage:
  ncu --set basic --launch-count 1 -k <regex-or-blank> \
    python oneshot_launch.py --kernel jsd --shape 8192,32000 \
      --config '{"block_sizes":[2048,1],"num_warps":32,"num_stages":1,"pid_type":"flat"}'
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))
for _d in (
    os.path.join(_HARNESS_DIR, "..", "harness"),
    os.path.join(_HARNESS_DIR, "..", "prompts"),
    os.path.join(_WT_ROOT, "examples"),
):
    _d = os.path.abspath(_d)
    if _d not in sys.path:
        sys.path.insert(0, _d)

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402

from run2_measure_g import KERNELS  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True)
    p.add_argument("--shape", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--warmup", type=int, default=3)
    args = p.parse_args()
    M, N = (int(x) for x in args.shape.split(","))
    fn, builder, _ref = KERNELS[args.kernel]
    kargs = builder(M, N)[0]
    cfg = helion.Config(**json.loads(args.config))
    k = helion.kernel(fn.fn, config=cfg, static_shapes=False)
    # compile + warmup (not profiled if ncu --launch-count targets the last launch)
    for _ in range(args.warmup):
        k(*kargs)
    torch.cuda.synchronize()
    # the profiled launch
    k(*kargs)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
