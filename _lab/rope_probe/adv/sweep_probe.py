"""Fast bind-only sweep: for a given kernel code + a list of shapes, print seed vs
default block_sizes and the derived budgets. No benchmarking (low GPU contention)."""
from __future__ import annotations

import sys

_WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"
sys.path.insert(0, _WT)

import torch  # noqa: E402
import helion  # noqa: E402

assert helion.__file__.startswith(_WT), helion.__file__
import helion.language as hl  # noqa: F401,E402
from helion._compiler.autotuner_heuristics.triton import (  # noqa: E402
    TritonPointwiseSeedHeuristic as H,
)
from helion.runtime import get_num_sm  # noqa: E402


# ---- rotate kernel with N intermediate temporaries (Direction B stressor) ----
@helion.kernel()
def rotate_manytemp(x, cos, sin):
    batch, heads, seq, hd = x.size()
    half = hd // 2
    out = torch.empty_like(x)
    for tb, tt in hl.tile([batch, seq]):
        cos_pair = cos[tb, tt, :].to(torch.float32).reshape([tb, tt, 2, half]).permute(0, 1, 3, 2)
        sin_pair = sin[tb, tt, :].to(torch.float32).reshape([tb, tt, 2, half]).permute(0, 1, 3, 2)
        cos_a, cos_b = hl.split(cos_pair)
        sin_a, sin_b = hl.split(sin_pair)
        xp = x[tb, :, tt, :].to(torch.float32).reshape([tb, heads, tt, 2, half]).permute(0, 1, 2, 4, 3)
        x_a, x_b = hl.split(xp)
        # many simultaneously-live fp32 temporaries (register-model blind spot)
        t0 = x_a * cos_a[:, None, :, :]
        t1 = x_b * sin_a[:, None, :, :]
        t2 = x_b * cos_b[:, None, :, :]
        t3 = x_a * sin_b[:, None, :, :]
        t4 = t0 - t1
        t5 = t2 + t3
        y_a = t4 + (t0 * 0.0)
        y_b = t5 + (t2 * 0.0)
        out[tb, :, tt, :] = hl.join(y_a, y_b).permute(0, 1, 2, 4, 3).reshape([tb, heads, tt, hd]).to(out.dtype)
    return out


def probe(name, fn, shape, mk):
    args = mk(shape)
    bk = fn.bind(tuple(args))
    env, dev_ir, spec = bk.env, bk.host_function.device_ir, bk.config_spec
    elig = H.is_eligible(env, dev_ir)
    if not elig:
        print(f"{name} {shape}: NOT ELIGIBLE red={bool(spec.reduction_facts)} acc={bool(spec.accumulator_facts)}")
        return
    seed = H.get_seed_config(env, dev_ir).config.get("block_sizes")
    default = spec.default_config().config.get("block_sizes")
    fact = spec.pointwise_facts[0]
    num_sm = max(1, get_num_sm(env.device))
    bpe = max(1, fact.bandwidth_bytes_per_elem)
    rbpe = max(1, fact.register_bytes_per_elem)
    bt = max(1, H.TILE_BYTES // bpe)
    rc = max(1, H.REGISTER_BYTES // rbpe)
    oc = max(1, fact.total_numel // (num_sm * H.MIN_WAVES))
    print(f"{name} {shape}: seed={seed} default={default} tot={fact.total_numel} "
          f"bpe={bpe} rbpe={rbpe} bt={bt} rc={rc} oc={oc}")


def mk_bhsd(shape):
    batch, heads, seq, hd = shape
    x = torch.randn(batch, heads, seq, hd, device="cuda", dtype=torch.bfloat16)
    ang = torch.randn(batch, seq, hd, device="cuda", dtype=torch.bfloat16)
    return (x, torch.cos(ang), torch.sin(ang))


if __name__ == "__main__":
    shapes = [
        [16, 4, 4096, 128],
        [8, 4, 8192, 128],
        [32, 2, 4096, 64],
        [16, 2, 8192, 64],
        [64, 1, 8192, 64],
        [32, 4, 4096, 128],
        [16, 8, 4096, 64],
    ]
    for s in shapes:
        probe("rotate_manytemp", rotate_manytemp, s, mk_bhsd)
