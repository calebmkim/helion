"""Accuracy-only sweep (NO timing): for every (corpus,kernel,shape,dtype) build the cell, compile
the seed + default configs, run once, and report accuracy vs the eager reference + max-abs error.
Fast (compile-bound) — use it to find which cells fail the gate before deciding fixes.

Run: PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
       python perf-repro/tools/acc_sweep.py [--corpus C] [--kernel K] [--max-shapes N]
"""
from __future__ import annotations
import argparse, importlib.util, os, sys

_TOOLS = os.path.dirname(os.path.abspath(__file__))
_PERF = os.path.dirname(_TOOLS)
sys.path.insert(0, _PERF)
os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")
_spec = importlib.util.spec_from_file_location("prb", os.path.join(_PERF, "perf_report_bench.py"))
prb = importlib.util.module_from_spec(_spec); sys.modules["prb"] = prb; _spec.loader.exec_module(prb)
import torch  # noqa: E402


def _acc_cell(corpus, kernel, shape, dtype):
    family = corpus[:-4] if corpus.endswith("_gen") else corpus
    dt = prb._DT.get(dtype)
    if family == "vllm":
        return prb.run_vllm_cell(kernel, shape, corpus=corpus)  # already runs acc; reuse its row
    if family == "curriculum":
        kfn, args, ref, acc, _ = prb._cur_build(kernel, shape[0], shape[1], dt)
    elif family == "transfer":
        kfn, args, ref, acc, _ = prb._transfer_build(kernel, shape, dt)
    elif family == "mreduction":
        kfn, args, ref, acc, _ = prb._mred_build(kernel, tuple(shape), dt)
    else:
        raise KeyError(corpus)
    torch._dynamo.reset()
    seed_cfg, base_cfg, fired = prb._extract_configs(kfn, args)
    out = {"fired": fired, "arms": {}}
    for name, cfg in (("seed", seed_cfg), ("default", base_cfg)):
        k = prb._replay(kfn, cfg)
        if k is None:
            out["arms"][name] = ("no-config", "")
            continue
        try:
            o = k(*args)
            ok, detail = acc(o, ref)
            out["arms"][name] = ("PASS" if ok else "FAIL", detail)
        except Exception as e:  # noqa: BLE001
            out["arms"][name] = (f"ERR:{type(e).__name__}", str(e)[:60])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--kernel", default=None)
    ap.add_argument("--max-shapes", type=int, default=2, help="test up to N (largest-N) shapes/kernel")
    a = ap.parse_args()
    SH = prb._load_shapes()
    fails = 0
    for corpus, c in SH["corpora"].items():
        if a.corpus and corpus != a.corpus:
            continue
        split = c["required_splits"][0]
        for kernel, kd in c["kernels"].items():
            if a.kernel and kernel != a.kernel:
                continue
            shapes = kd["shapes"][split]
            # test the largest-reduction-width shapes (worst case for accumulation error)
            shapes = sorted(shapes, key=lambda s: -(s[1] if len(s) > 1 else s[0]))[:a.max_shapes]
            for shape in shapes:
                for dtype in c["dtypes"]:
                    try:
                        r = _acc_cell(corpus, kernel, tuple(shape), dtype)
                    except Exception as e:  # noqa: BLE001
                        print(f"[BUILD-ERR] {corpus}/{kernel} {shape} {dtype}: {type(e).__name__}: {str(e)[:80]}")
                        continue
                    if corpus.startswith("vllm"):
                        arms = {n: (v.get("status"), v.get("acc_detail", "")) for n, v in r["arms"].items()
                                if n in ("seed", "default")}
                        badv = [n for n, (s, _) in arms.items() if s not in ("ok",)]
                    else:
                        arms = r["arms"]
                        badv = [n for n, (s, _) in arms.items() if s != "PASS"]
                    flag = "  <-- FAIL" if badv else ""
                    if badv:
                        fails += 1
                        det = "; ".join(f"{n}={s}({d})" for n, (s, d) in arms.items())
                        print(f"[{corpus}/{kernel}] {shape} {dtype}: {det}{flag}")
    print(f"\n=== {fails} failing cell(s) across sampled shapes ===")


if __name__ == "__main__":
    main()
