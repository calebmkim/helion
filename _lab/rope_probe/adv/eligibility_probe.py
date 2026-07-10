"""Eligibility + seed/default block-size probe (NO benchmarking).

Binds a candidate kernel from a shard JSON, checks the pointwise fact fires,
and prints the seed vs default block_sizes plus the derived fact fields, so we
can validate structural choices for the partial-tiling lens before the real
scoring harness runs on GPU.

argv: shard_json entry_index shape_index
"""
from __future__ import annotations

import json
import sys

_WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
sys.path.insert(0, _WT)

import torch  # noqa: E402
import helion  # noqa: E402

assert helion.__file__.startswith(_WT), helion.__file__
import helion.language as hl  # noqa: F401,E402
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    TritonPointwiseSeedHeuristic,
)


def main() -> None:
    shard, ei, si = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    entry = json.load(open(shard))[ei]
    name = entry["name"]
    shape = entry["shapes"][si]
    out = {"name": name, "shape": shape}

    import importlib.util
    import os

    gen_dir = os.path.join(os.path.dirname(shard), "_gen_probe")
    os.makedirs(gen_dir, exist_ok=True)
    mod_path = os.path.join(gen_dir, f"advk_{ei}_{name}.py")
    header = (
        "from __future__ import annotations\n"
        "import torch\nimport helion\nimport helion.language as hl\n\n"
    )
    with open(mod_path, "w") as f:
        f.write(header + entry["code"])
    spec_ = importlib.util.spec_from_file_location(f"advk_{ei}_{name}", mod_path)
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    fn = getattr(mod, name)
    args = tuple(mod.make_inputs(tuple(shape) if isinstance(shape, list) else shape))

    bk = fn.bind(args)
    env, dev_ir, spec = bk.env, bk.host_function.device_ir, bk.config_spec
    out["pointwise_eligible"] = TritonPointwiseSeedHeuristic.is_eligible(env, dev_ir)
    out["has_reduction_facts"] = bool(spec.reduction_facts)
    out["has_matmul_facts"] = bool(spec.matmul_facts)
    out["has_accumulator_facts"] = bool(spec.accumulator_facts)
    if not out["pointwise_eligible"]:
        print(json.dumps(out))
        return
    seed_cfg = TritonPointwiseSeedHeuristic.get_seed_config(env, dev_ir)
    default_cfg = spec.default_config()
    out["seed_block_sizes"] = seed_cfg.config.get("block_sizes")
    out["default_block_sizes"] = default_cfg.config.get("block_sizes")
    fact = spec.pointwise_facts[0]
    out["fact_total_numel"] = fact.total_numel
    out["fact_bytes_per_elem"] = fact.bandwidth_bytes_per_elem
    out["fact_reg_bytes_per_elem"] = fact.register_bytes_per_elem
    # derived budgets
    from helion.runtime import get_num_sm

    num_sm = max(1, get_num_sm(env.device))
    H = TritonPointwiseSeedHeuristic
    bpe = max(1, fact.bandwidth_bytes_per_elem)
    rbpe = max(1, fact.register_bytes_per_elem)
    out["num_sm"] = num_sm
    out["budget_target"] = max(1, H.TILE_BYTES // bpe)
    out["reg_cap"] = max(1, H.REGISTER_BYTES // rbpe)
    out["occ_cap"] = max(1, fact.total_numel // (num_sm * H.MIN_WAVES))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
