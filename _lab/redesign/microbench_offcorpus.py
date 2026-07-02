"""Off-corpus micro-bench: build a curriculum kernel at an ARBITRARY (M,N) via its KERNELS
builder and A/B two pinned configs (median-of-9 do_bench), for shapes replay_bench's corpus
adapter doesn't contain (e.g. rms_norm at N=32768/49152, the S2 verification gap).

Usage (from /tmp, FOREGROUND, one shape/process):
  cd /tmp && HELION_CACHE_DIR=$(mktemp -d) PYTHONPATH=/home/dev/local/helion-redesign \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/redesign/microbench_offcorpus.py \
    --kernel rms_norm --shape 8192,49152 \
    --before '{"block_sizes":[1],"reduction_loops":[null],"num_warps":32,"num_stages":1,"pid_type":"flat"}' \
    --after  '{"block_sizes":[1],"reduction_loops":[16384],"num_warps":32,"num_stages":1,"pid_type":"flat"}'
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
from triton.testing import do_bench  # noqa: E402

import helion  # noqa: E402

from run2_measure_g import KERNELS  # noqa: E402

N_RUNS = 9


def _med(fn) -> float:
    torch.cuda.synchronize()
    samples = sorted(float(do_bench(fn, return_mode="median")) for _ in range(N_RUNS))
    return samples[len(samples) // 2] * 1000.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True)
    p.add_argument("--shape", required=True)
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    args = p.parse_args()
    M, N = (int(x) for x in args.shape.split(","))
    fn, builder, _ref = KERNELS[args.kernel]
    kargs = builder(M, N)[0]

    def bind(cfg_dict):
        cfg = helion.Config(**json.loads(cfg_dict))
        k = helion.kernel(fn.fn, config=cfg, static_shapes=False)
        return lambda: k(*kargs)

    f_b, f_a = bind(args.before), bind(args.after)
    ob, oa = f_b(), f_a()

    def first(o):
        return o[0] if isinstance(o, (tuple, list)) else o

    same = None
    if torch.is_tensor(first(ob)) and torch.is_tensor(first(oa)):
        same = bool(
            torch.allclose(first(ob).float(), first(oa).float(), rtol=2e-2, atol=2e-2)
        )
    tb, ta = _med(f_b), _med(f_a)
    print(
        json.dumps(
            {
                "cell": f"{args.kernel}/({M}, {N})",
                "before_us": round(tb, 3),
                "after_us": round(ta, 3),
                "ratio_after_over_before": round(ta / tb, 4),
                "outputs_match": same,
                "regression_past_10pct": bool(ta / tb > 1.10),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
