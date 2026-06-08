"""Find CORRECT welford configs at the in-sample shapes, esp. the non-pow2 canary
(262144,1536). The correctness rule (from codegen): the welford-COMBINE tile
(block 1) width must DIVIDE N exactly (so Tn=tile_width == valid count, no mask).
The normalize tile (block 2) just writes out -- masking is fine there (it writes
only valid lanes), so it can be persistent next_pow2.

We test, per shape, several (R1, R2) patterns + measure correctness (maxabs/maxrel
vs eager layer_norm) and whether the config even compiles. M_BLOCK at the
autotuner floor (we read it).
"""
from __future__ import annotations

import sys
import torch
import helion

WORKTREE = "/home/calebkim/helion-new-heuristics/wt-reduction"
assert helion.__file__.startswith(WORKTREE), helion.__file__
sys.path.insert(0, WORKTREE)

from triton.testing import do_bench  # noqa: E402
from examples.welford import welford, eager_layer_norm  # noqa: E402
from helion._utils import next_power_of_2 as np2  # noqa: E402

EPS = 1e-5
IN_SAMPLE = [(262144, 1024), (262144, 1536), (262144, 2048), (262144, 4096)]


def args(m, n):
    return (torch.rand(n, device="cuda"), torch.rand(n, device="cuda"),
            torch.rand(m, n, device="cuda"), EPS)


def med(fn, reps=4):
    torch.cuda.synchronize()
    s = [float(do_bench(fn, return_mode="median")) for _ in range(reps)]
    return sorted(s)[len(s) // 2]


def mfloor(a):
    b = welford.bind(a)
    bs = b.config_spec.block_sizes[0]
    return max(1, bs.min_size, bs.autotuner_min)


def run(a, bs, w, ref):
    try:
        k = helion.kernel(welford.fn, configs=[helion.Config(
            block_sizes=bs, num_warps=w, num_stages=1)])
        b = k.bind(a)
        b.ensure_config_exists(a)
        out = b(*a)
        out = out[0] if isinstance(out, tuple) else out
        maxabs = float((out.float() - ref.float()).abs().max())
        denom = ref.float().abs().clamp_min(1e-6)
        maxrel = float(((out.float() - ref.float()).abs() / denom).max())
        ok = bool(torch.allclose(out.float(), ref.float(), rtol=1e-3, atol=1e-3))
        lat = med(lambda: b(*a)) * 1000
        return dict(lat=round(lat, 1), ok=ok, maxabs=maxabs, maxrel=maxrel)
    except Exception as e:  # noqa: BLE001
        return dict(err=f"{type(e).__name__}: {str(e)[:60]}")


def main():
    print(f"helion={helion.__file__}\n")
    for (m, n) in IN_SAMPLE:
        a = args(m, n)
        ref = eager_layer_norm(*a)
        mf = mfloor(a)
        P = np2(n)
        nonpow2 = (P != n)
        print(f"=== ({m},{n})  next_pow2={P}  nonpow2={nonpow2}  m_floor={mf} ===")
        # candidate (R1_combine, R2_normalize) patterns
        cands = [
            ("persist-np2-both", [mf, P, P]),         # masked combine if nonpow2 -> WRONG
            ("persist-exactN-combine", [mf, n, P]),   # combine = exact N (divides), normalize np2
            ("persist-exactN-both", [mf, n, n]),      # both exact N
        ]
        # divisor combine chunk (largest pow2 dividing N), normalize np2
        div = n
        d = P
        while d > 1 and n % d != 0:
            d //= 2
        cands.append((f"div-combine-{d}", [mf, d, P]))
        for w in (8, 16):
            for label, bs in cands:
                r = run(a, bs, w, ref)
                tag = "OK " if r.get("ok") else ("ERR" if "err" in r else "BAD")
                if "err" in r:
                    print(f"  [{tag}] {label:24s} w{w} bs={bs}: {r['err']}")
                else:
                    print(f"  [{tag}] {label:24s} w{w} bs={bs}: "
                          f"{r['lat']}us maxabs={r['maxabs']:.2e} maxrel={r['maxrel']:.2e}")
        print()


if __name__ == "__main__":
    main()
