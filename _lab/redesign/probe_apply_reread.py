"""S1/S2 grounding — classify every corpus kernel by the APPLY-REREAD signal + dump the
pinned-tile-bytes of the primary's re-read load. The shared detector for S1 (eviction gate) and
S2 (persist-hold ceiling).

apply_reread = there is a LOAD of a tensor that ALSO feeds the primary reduction, but THIS load
feeds a STORE and NO reduction (a separate pass re-reading the reduction row to write output).

Usage (from /tmp):
  cd /tmp && HELION_AUTOTUNE_EFFORT=none HELION_CACHE_DIR=$(mktemp -d) \
    PYTHONPATH=/home/dev/local/helion-redesign /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-redesign/_lab/redesign/probe_apply_reread.py
"""

from __future__ import annotations

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

import ground_live_tiles as G  # noqa: E402


def classify(fn, args, label):
    from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
        _primary_descriptor_selected,
    )

    bound = fn.bind(args)
    env = bound.env
    dev = bound.host_function.device_ir
    env.__enter__()
    bound.host_function.__enter__()
    try:
        spec = env.config_spec
        pd = _primary_descriptor_selected(env)
        if pd is None:
            print(f"{label:44} pd=None (declines)")
            return
        facts = spec.memory_op_facts
        # tensors that feed the primary reduction (loads whose reductions_fed includes pd axis)
        red_tensors = {
            f.tensor_name
            for f in facts
            if f.kind == "load"
            and any(ax == pd.block_id for ax, _ in f.reductions_fed)
        }
        # apply-reread: a load of one of those tensors that feeds a store and NO reduction
        apply_reread = any(
            f.kind == "load"
            and f.tensor_name in red_tensors
            and f.stores_fed
            and not f.reductions_fed
            for f in facts
        )
        # pinned-tile-bytes of the re-read load (at pd.reread_eviction_index)
        pinned_bytes = None
        if pd.reread_eviction_index is not None:
            rr = next(
                (
                    f
                    for f in facts
                    if f.kind == "load"
                    and f.eviction_index == pd.reread_eviction_index
                ),
                None,
            )
            if rr is not None and rr.inner_extent and rr.dtype is not None:
                # m_block unknown at fact time; report per-(m_block=1) tile bytes = inner_extent×itemsize
                pinned_bytes = rr.inner_extent * rr.dtype.itemsize
        print(
            f"{label:44} cat={pd.category.value:10} reread={int(pd.row_reread)} "
            f"apply_reread={int(apply_reread)} "
            f"pinned/m=({pinned_bytes}) red_tensors={sorted(t for t in red_tensors if t)}"
        )
    finally:
        bound.host_function.__exit__(None, None, None)
        env.__exit__(None, None, None)
        del bound
        torch.cuda.empty_cache()


def main():
    print(f"helion={helion.__file__}\n", flush=True)
    for k in [
        "sum",
        "long_sum",
        "softmax",
        "kl_div",
        "jsd",
        "rms_norm",
        "layer_norm",
        "welford",
        "cross_entropy",
    ]:
        fn, a = G._curriculum_fn_args(k)
        classify(fn, a, f"curriculum/{k}")
    for rec in [
        "transfer:cross_entropy_ls_zloss",
        "transfer:gated_rmsnorm",
        "transfer:fused_add_rmsnorm",
        "transfer:fused_add_layernorm",
        "transfer:scaled_masked_softmax",
        "transfer:dynamic_quant",
        "mreduction:layer_norm_bwd",
        "mreduction:rms_norm_bwd",
        "mreduction:bias_grad_bwd",
        "mreduction:group_norm_bwd",
        "mreduction:instance_norm_bwd",
        "mreduction:dyt_bwd",
        "vllm:per_token_group_fp8_quant",
        "vllm:rms_norm_per_block_quant",
    ]:
        c, _, kn = rec.partition(":")
        for label, fn, a in list(G._recorder_targets(c, {kn}))[:1]:
            classify(fn, a, label)


if __name__ == "__main__":
    main()
