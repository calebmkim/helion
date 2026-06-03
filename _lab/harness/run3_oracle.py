"""RUN-3 Phase-2 ORACLE builder + seed/oracle field-diff (machine-portable).

For one (kernel, M, N): run the Helion autotuner FRESH (force=True), fair-re-bench
its winning config with do_bench (the autotuner's internal perf_ms is NOT comparable
to a do_bench median — a documented trap), measure the LIVE seed with do_bench in
the SAME process (so seed/oracle is noise-robust: both arms timed identically), gate
BOTH for correctness vs the fp32 reference, field-diff the seed against the oracle's
winning config (the differing fields ARE the Phase-2 worklist), and CACHE the result
keyed by a kernel-source-hash in `_lab/logs/run3/oracle_cache.json`.

VICTORY bar: seed/oracle <= 1+eps (eps 3-5%). FLOOR (re-confirmed): G=tc/seed >= 1-eps.
A cached entry is FRESH only while its source_hash matches the current source. KEY recipe
(ledger-keeper guardian, see source_hash): kernel source + generated Triton of the DEFAULT
config (codegen reference) + config_spec knob/range dump. EXCLUDES the heuristic/seed code
(the oracle is what the SEARCH finds, not what the seed emits). Safety net: victory-confirm
ALWAYS re-runs a FRESH FULL oracle, so cache under-invalidation can't taint a done-verdict.

The autotuner budget is set by HELION_AUTOTUNE_EFFORT (quick to iterate, full to
confirm) — pass it in the environment. HELION_FORCE_AUTOTUNE is implied by force=True;
the ephemeral Triton cache (default) keeps the oracle fresh.

fp32 everywhere. Pin CUDA_VISIBLE_DEVICES; confirm GPU idle. ONE GPU => never run two
do_bench / two autotunes concurrently.

Canonical invocation (run from /tmp):
  cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=quick \
    PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
    /home/dev/helion/.venv/bin/python \
    /home/dev/local/helion-reduction-heuristics-run2/_lab/harness/run3_oracle.py \
    --kernel cross_entropy --M 8192 --N 128256
  # or a batch from a JSON list of {kernel,M,N}:
  #   ... run3_oracle.py --batch /path/to/shapes.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

from run2_measure_g import KERNELS  # noqa: E402
from run2_measure_g import N_RUNS  # noqa: E402
from run2_measure_g import check_correct  # noqa: E402
from run2_measure_g import codegen_kind  # noqa: E402
from run2_measure_g import get_seed  # noqa: E402
from triton.testing import do_bench  # noqa: E402

import helion  # noqa: E402

EPS = 0.05
LOG_DIR = os.path.abspath(os.path.join(_HARNESS_DIR, "..", "logs", "run3"))
CACHE_PATH = os.path.join(LOG_DIR, "oracle_cache.json")

# Worktree root (derived, machine-portable) for the source-hash inputs.
_WT = os.path.abspath(os.path.join(_HARNESS_DIR, "..", ".."))

# kernel name -> the examples source file(s) that define its computation.
_KERNEL_SRC = {
    "rms_norm": ["examples/rms_norm.py"],
    "layer_norm": ["examples/layer_norm.py"],
    "welford": ["examples/welford.py"],
    "softmax": ["examples/softmax.py"],
    "cross_entropy": ["examples/cross_entropy.py"],
    "kl_div": ["examples/kl_div.py"],
    "jsd": ["examples/jsd.py"],
    "sum": ["examples/sum.py"],
    "long_sum": ["examples/long_sum.py"],
}


def source_hash(kernel: str, bound) -> str:
    """Oracle-cache key (LEDGER-KEEPER guardian recipe, 2026-06-03):
        sha256( read(examples/<kernel>.py)
                + to_triton_code(DEFAULT_config for this shape)  # codegen, per-shape
                + repr(sorted config_spec knobs+ranges) )        # search space
    EXCLUDES the heuristic file + any seed-only fact code: the oracle is what the
    autotuner SEARCH finds — it does NOT depend on the seed the heuristic emits, so
    keying on the heuristic would invalidate every cached oracle on every heuristic
    edit and defeat the cache's cheap-first purpose. The generated Triton of the
    DEFAULT config is a fixed, heuristic-independent per-shape reference that moves
    iff codegen that actually alters THIS kernel's Triton changes (it naturally
    takes the relevant path: looped at wide N, persistent at narrow). Safety net:
    victory-confirm ALWAYS re-runs a FRESH FULL oracle, so under-invalidation can
    only mislead during iteration, never at a done-verdict."""
    h = hashlib.sha256()
    # (a) kernel source
    for rel in _KERNEL_SRC[kernel]:
        with open(os.path.join(_WT, rel), "rb") as f:
            h.update(rel.encode())
            h.update(b"\0")
            h.update(f.read())
            h.update(b"\0")
    # (b) generated Triton of the DEFAULT config for this shape (codegen reference)
    try:
        default_cfg = bound.config_spec.default_config()
        h.update(b"TRITON\0")
        h.update(bound.to_triton_code(default_cfg).encode())
        h.update(b"\0")
    except Exception as e:
        h.update(f"TRITON_ERR:{type(e).__name__}\0".encode())
    # (c) config_spec knob/range dump (search space)
    try:
        spec = bound.config_spec
        knobs = []
        for attr in sorted(dir(spec)):
            if attr.startswith("_"):
                continue
            try:
                v = getattr(spec, attr)
            except Exception:
                continue
            if callable(v):
                continue
            knobs.append(f"{attr}={v!r}")
        h.update(("KNOBS\0" + "\n".join(knobs)).encode())
    except Exception as e:
        h.update(f"KNOBS_ERR:{type(e).__name__}\0".encode())
    return h.hexdigest()[:16]


def _bench_samples(fn, n=N_RUNS):
    torch.cuda.synchronize()
    return [float(do_bench(fn, return_mode="median")) for _ in range(n)]


def _stats(samples):
    s = sorted(samples)
    med = s[len(s) // 2]
    return {
        "median_ms": med,
        "min_ms": s[0],
        "max_ms": s[-1],
        "spread": (s[-1] - s[0]) / med if med > 0 else None,
        "n": len(samples),
        "samples_ms": samples,
    }


def field_diff(seed_cfg: dict, oracle_cfg: dict):
    """Per-field diff seed vs oracle (the Phase-2 worklist). Union of keys."""
    diff = {}
    for k in sorted(set(seed_cfg) | set(oracle_cfg)):
        sv, ov = seed_cfg.get(k), oracle_cfg.get(k)
        if sv != ov:
            diff[k] = {"seed": sv, "oracle": ov}
    return diff


def run_one(kernel: str, M: int, N: int, effort: str):
    fn, builder, tc_ref = KERNELS[kernel]
    args, ref, out_extract = builder(M, N)
    for a in args:
        if torch.is_tensor(a) and a.is_floating_point():
            assert a.dtype == torch.float32, f"{kernel}{(M, N)} non-fp32 {a.dtype}"

    # --- live seed (normalized) + correctness + codegen kind ---
    seed_raw, _ = get_seed(fn, args)
    seeded = helion.kernel(fn.fn, configs=[helion.Config(**dict(seed_raw))])
    bound_s = seeded.bind(args)
    bound_s.ensure_config_exists(args)
    seed_norm = dict(bound_s._config)
    seed_codegen = codegen_kind(bound_s)
    out_s = out_extract(bound_s(*args))
    seed_ok, seed_maxerr = check_correct(out_s, ref)

    # --- ORACLE: fresh autotune (force=True; ephemeral triton cache) ---
    t0 = time.time()
    k_at = helion.kernel(fn.fn)
    bk = k_at.bind(args)
    oracle_cfg_obj = bk.autotune(args, force=True)
    autotune_s = time.time() - t0
    oracle_cfg = dict(oracle_cfg_obj)

    # --- fair re-bench of the oracle winner with do_bench (fresh kernel) ---
    oracle_k = helion.kernel(fn.fn, configs=[helion.Config(**oracle_cfg)])
    bound_o = oracle_k.bind(args)
    bound_o.ensure_config_exists(args)
    oracle_norm = dict(bound_o._config)
    oracle_codegen = codegen_kind(bound_o)
    out_o = out_extract(bound_o(*args))
    oracle_ok, oracle_maxerr = check_correct(out_o, ref)

    # --- torch.compile default (floor reference) ---
    torch._dynamo.reset()
    tc = torch.compile(tc_ref)
    out_tc = out_extract(tc(args))
    ok_tc, _ = check_correct(out_tc, ref)
    assert ok_tc, f"tc reference FAIL {kernel} {(M, N)}"

    # --- SAME-PROCESS do_bench of all three (noise-robust ratios) ---
    seed_st = _stats(_bench_samples(lambda: bound_s(*args))) if seed_ok else None
    oracle_st = _stats(_bench_samples(lambda: bound_o(*args))) if oracle_ok else None
    tc_st = _stats(_bench_samples(lambda: tc(args)))

    seed_med = seed_st["median_ms"] if seed_st else None
    oracle_med = oracle_st["median_ms"] if oracle_st else None
    tc_med = tc_st["median_ms"]

    seed_oracle = (seed_med / oracle_med) if (seed_med and oracle_med) else None
    g_floor = (tc_med / seed_med) if seed_med else None
    oracle_vs_tc = (tc_med / oracle_med) if oracle_med else None

    entry = {
        "kernel": kernel,
        "shape": [M, N],
        "source_hash": source_hash(kernel, bk),
        "effort": effort,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "autotune_s": round(autotune_s, 1),
        "seed_cfg": seed_norm,
        "seed_codegen": seed_codegen,
        "seed_correct": seed_ok,
        "seed_maxerr": seed_maxerr,
        "oracle_cfg": oracle_norm,
        "oracle_codegen": oracle_codegen,
        "oracle_correct": oracle_ok,
        "oracle_maxerr": oracle_maxerr,
        "seed_us": seed_med * 1000 if seed_med else None,
        "oracle_us": oracle_med * 1000 if oracle_med else None,
        "tc_us": tc_med * 1000,
        "seed_oracle": seed_oracle,  # VICTORY: <= 1+eps
        "G_floor": g_floor,  # FLOOR: >= 1-eps
        "oracle_vs_tc": oracle_vs_tc,  # >1 => oracle beats tc
        "seed_dist": seed_st,
        "oracle_dist": oracle_st,
        "tc_dist": tc_st,
        "field_diff_seed_vs_oracle": field_diff(seed_norm, oracle_norm),
    }
    return entry


def _load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {"entries": {}}


def _cache_key(kernel, M, N):
    return f"{kernel}:{M}x{N}"


def _save_entry(entry):
    os.makedirs(LOG_DIR, exist_ok=True)
    cache = _load_cache()
    key = _cache_key(entry["kernel"], entry["shape"][0], entry["shape"][1])
    cache["entries"][key] = entry
    cache["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2, default=str)


def _print_entry(e):
    so = f"{e['seed_oracle']:.3f}" if e["seed_oracle"] is not None else "None"
    gf = f"{e['G_floor']:.3f}" if e["G_floor"] is not None else "None"
    ot = f"{e['oracle_vs_tc']:.3f}" if e["oracle_vs_tc"] is not None else "None"
    verdict = (
        "VICTORY"
        if (e["seed_oracle"] is not None and e["seed_oracle"] <= 1 + EPS)
        else "GAP"
    )
    print(
        f"\n=== {e['kernel']}({e['shape'][0]},{e['shape'][1]}) [{verdict}] "
        f"effort={e['effort']} autotune={e['autotune_s']}s ===",
        flush=True,
    )
    print(
        f"  seed/oracle={so}  G_floor(tc/seed)={gf}  oracle/tc(tc/oracle)={ot}",
        flush=True,
    )
    print(
        f"  seed_us={e['seed_us']:.1f} ({e['seed_codegen']}) "
        f"oracle_us={e['oracle_us']:.1f} ({e['oracle_codegen']}) "
        f"tc_us={e['tc_us']:.1f}",
        flush=True,
    )
    print(
        f"  seed_correct={e['seed_correct']} (err {e['seed_maxerr']:.1e})  "
        f"oracle_correct={e['oracle_correct']} (err {e['oracle_maxerr']:.1e})",
        flush=True,
    )
    print("  FIELD DIFF seed->oracle (the worklist):", flush=True)
    if e["field_diff_seed_vs_oracle"]:
        for k, v in e["field_diff_seed_vs_oracle"].items():
            print(f"    {k}: seed={v['seed']!r}  oracle={v['oracle']!r}", flush=True)
    else:
        print("    (seed config == oracle config — already matched)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", choices=list(KERNELS))
    ap.add_argument("--M", type=int)
    ap.add_argument("--N", type=int)
    ap.add_argument("--batch", help="JSON file: list of {kernel,M,N}")
    a = ap.parse_args()
    effort = os.environ.get("HELION_AUTOTUNE_EFFORT", "quick")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"GPU={gpu} helion={helion.__file__} effort={effort}", flush=True)

    if a.batch:
        with open(a.batch) as f:
            shapes = json.load(f)
    else:
        assert a.kernel and a.M and a.N, "need --kernel/--M/--N or --batch"
        shapes = [{"kernel": a.kernel, "M": a.M, "N": a.N}]

    for sp in shapes:
        kernel, M, N = sp["kernel"], int(sp["M"]), int(sp["N"])
        tag = f"{kernel}({M},{N})"
        try:
            e = run_one(kernel, M, N, effort)
        except torch.cuda.OutOfMemoryError as ex:
            torch.cuda.empty_cache()
            print(f"[OOM ] {tag}: {type(ex).__name__}", flush=True)
            continue
        except Exception as ex:
            print(f"[ERR ] {tag}: {type(ex).__name__}: {ex}"[:300], flush=True)
            continue
        _save_entry(e)
        _print_entry(e)
    print(f"\n[cache -> {CACHE_PATH}]", flush=True)


if __name__ == "__main__":
    main()
