# LOCAL SETUP — devserver (devvm272.dkl0.facebook.com), this run (concrete facts)

`hillclimb-method.md` holds the durable principles; this file holds the specifics it defers to.
This is the **devserver** analogue of `local-setup.md` (which describes a *different* machine — the
`/home/dev/...` venv box with a single H100). **Everything below was probed and run live on
2026-06-08**, not assumed — every canonical command in §7 was executed and produced the result quoted.
Still: reconcile against the live tree before trusting any path/SHA here (they drift). If
`helion.__file__` or a path doesn't resolve, re-discover it (`git rev-parse --show-toplevel`,
`pip show helion`) — don't trust this blindly.

## Paths & interpreter
- **Worktree (source of truth):** `/home/calebkim/helion-new-heuristics/helion-fork-pr-with-lab`
  (branch `reduction-pr-with-lab`). The heuristic is at
  `helion/_compiler/autotuner_heuristics/triton.py`; facts at `helion/autotuner/config_spec.py`;
  examples at `examples/`; the lab at `_lab/`. Confirm with `git rev-parse --show-toplevel`.
- **Interpreter:** `/home/calebkim/.conda/envs/helion/bin/python` (conda env `helion`, Python 3.12.13).
  **Verified deps:** torch `2.12.0.dev20260408+cu128` (cuda build 12.8), triton `3.7.0`,
  `packaging 26.0`, `torch.cuda.is_available() == True`, 4 devices. **Never `pip install`** (repo rule;
  the env is already complete). The system `python3` (`/usr/local/bin/python3`, 3.12+meta) **lacks the
  deps** (no `torch`, no `packaging`) — never use it.
- **Run scripts from `cwd=/tmp`** (or any dir that is NOT a checkout root) **with
  `PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-fork-pr-with-lab`** so `import helion`
  resolves to the worktree (see §2 — by default it resolves *elsewhere*). Assert it at the top of
  every script:
  `assert helion.__file__.startswith("/home/calebkim/helion-new-heuristics/helion-fork-pr-with-lab")`.
  Verified live: with PYTHONPATH set, `helion.__file__` and `examples.rms_norm.__file__` both resolve
  under the worktree; without it, `import helion` →
  `/home/calebkim/helion-new-heuristics/local/helion/helion/__init__.py` (the wrong checkout).

## The two checkouts + the import-wiring gotcha (READ — this is the #1 footgun)
There are several checkouts on this box; two matter, and they are **linked git worktrees of one repo**
(`.git` here is an 85-byte *file*; `git rev-parse --git-common-dir` →
`/home/calebkim/helion-new-heuristics/local/helion/.git`):

| Path | What it is | Role |
|---|---|---|
| `/home/calebkim/helion-new-heuristics/helion-fork-pr-with-lab` | **your worktree** (branch `reduction-pr-with-lab`, HEAD `02046a83`) | edit `helion/` + `examples/` **here**; `_lab/` lives here |
| `/home/calebkim/helion-new-heuristics/local/helion` | the **main worktree** (branch `main`, HEAD `ed36077d`) | holds the **editable install** + the **tritonbench operators**; edit operators **here** |

(Other worktrees exist — `helion-fork-seed-run2`, `helion-scratch-1/2/3` — ignore them. `git worktree
list` shows all.)

`helion` is installed **editable as a plain `.pth` path entry** (`_editable_impl_helion.pth` →
`/home/calebkim/helion-new-heuristics/local/helion`), so by default `import helion` resolves to the
MAIN checkout, **not** your worktree. `PYTHONPATH=<worktree>` comes earlier in `sys.path`, so it
shadows the `.pth` — **that's the fix, every run** (verified live, §1).

- **`benchmarks/run.py` IS in the worktree** (`<worktree>/benchmarks/run.py`, 86 KB) → it runs from
  your worktree. ✅
- **`tritonbench` resolves to the MAIN checkout even with the worktree on `PYTHONPATH`.** It's a
  `MetaPathFinder` editable (`__editable___tritonbench_0_0_1_finder.py`) whose `MAPPING` is
  **hardcoded** to `/home/calebkim/helion-new-heuristics/local/helion/benchmarks/tritonbench/tritonbench`,
  and the worktree has **no `benchmarks/tritonbench/` dir** to shadow it (verified). ⇒ **TritonBench
  operator edits (`operators/<op>/operator.py`, new baselines like `torch_compile_<op>_default`) go in
  the MAIN checkout** (`local/helion/benchmarks/tritonbench/...`), while the Helion kernel under test +
  the `examples.*` wrapper run from your worktree. Prove both with a sentinel print after any edit.

## GPU — 4× H100, shared box (all idle this run)
- **4× NVIDIA H100 (~97 GB each), indices 0–3**, all idle at probe time (4 MiB, 0% util). Driver
  580.126.09, CUDA 13.0. `CUDA_VISIBLE_DEVICES` was unset.
- **Pin every run** with `CUDA_VISIBLE_DEVICES=<idx>` (the portable bench scripts pin `0` internally).
  **Re-check `nvidia-smi` before any trusted timing** — this is a shared machine; rule out co-tenants
  before believing a delta. Plenty of GPUs ⇒ you *can* run on a free index while another is busy, but
  never two `do_bench` runs on the SAME device at once.
- **NEVER launch a detached/background GPU job** (they die silently and never notify). Long oracles:
  foreground, one shape at a time, JSON-checkpointed.

## Git — commit frequently; remote naming is INVERTED vs the old box
- Worktree branch `reduction-pr-with-lab` (carries `_lab/`-only commits on top of the PR head — never
  push this lab branch onto the PR ref). Because all checkouts share one repo, **remotes are shared**:
  - **`origin` = `org-21003710@github.com:pytorch/helion.git`** — this is **UPSTREAM** (pytorch/helion).
    **NEVER push to `origin` here.** (On the old `/home/dev` box `origin` was the *fork* — the meaning
    is flipped on this box, so don't carry muscle memory over.)
  - **`fork` = `https://github.com/calebmkim/helion.git`** — the personal fork.
- **Commit to your own branch freely and often** — bank each green step. Do **not** `git push` (the
  human updates the PR; repo rule).

## Key scripts (reuse) — paths not guaranteed portable; use judgement
**These lab scripts were authored on prior machines, so some hardcode paths / `startswith` asserts /
default interpreters that don't match this box.** Don't expect 100% portability — repoint the one you
need when you need it. Two tiers:

- **Headline bench scripts — portable, BUT default to the old interpreter** (`seed_vs_tc.py`,
  `sweep.py`, `bare_fwd_seed_vs_tc.py`): they derive the worktree root from `__file__` and honor
  `HELION_WORKTREE` / `HELION_PY`. **On this box you MUST set
  `HELION_PY=/home/calebkim/.conda/envs/helion/bin/python`** (their built-in default is the old box's
  `/home/dev/helion/.venv/bin/python`, which doesn't exist here). `HELION_WORKTREE` auto-derives
  correctly but set it for safety. The `run3_oracle.py` / `run3_task1_verify_after_edit.py` *bodies*
  are root-from-`__file__` and assert `helion.__file__` under the derived root (portable — verified);
  only their docstring example commands cite the dead `/home/dev/...` PYTHONPATH/interpreter — ignore
  those lines.
- **The many per-lever probe scripts** (`_lab/harness/AUDITOR_*`, `AUDIT_*`, `*_ab.py`, etc.) still
  hardcode old roots (`/home/dev/local/...`, `/home/calebkim/helion-new-heuristics/wt-reduction`, etc.).
  They're throwaway A/B probes — repoint the one you need, don't bulk-fix.

### ⚠️ `seed_vs_tc.py` ↔ this tritonbench version mismatch (verified failure)
`seed_vs_tc.py`'s `KCFG` marks several kernels as `--shapes`-capable, but **this checkout's tritonbench
`rms_norm` operator only accepts `--M`/`--H`** (it rejects `--shapes` → the script raises
`no table for rms_norm`). So the portable 3-arm script fails on rms_norm here as-is. Workarounds: drive
the seeded arm directly via `run_seeded.py --kernel rms_norm --M <m> --H <h> --num-inputs <k>` (verified
working, §7), or repoint `KCFG[rms_norm]`'s `supports_shapes` flag. Verify `--shapes` vs `--M/--H`
per-kernel before trusting the script on any given operator.

## Key scripts — what they do (same roles as `local-setup.md`)
- **Behavior oracle (compile-time, no GPU):** `_lab/harness/run3_task1_verify_after_edit.py` — records
  the emitted seed config across the curriculum; BEFORE/AFTER diff, 0 config-diffs = behavior-preserving.
- **Headline bench (tritonbench path):** `_lab/bench/run_seeded.py` (promotes seed → default at
  `HELION_AUTOTUNE_EFFORT=none` when `HELION_PROMOTE_REDUCTION_SEED=1`), driven by
  `_lab/bench/seed_vs_tc.py` / `_lab/bench/sweep.py` (3-arm: seeded / unseeded-default / tc-default).
  `_lab/bench/bare_fwd_seed_vs_tc.py` is the bare-forward cross-check.
- **Curriculum:** `_lab/prompts/shapes_v3_draft.py` (`SHAPES` + `TRANSFER`; `validate()` enforces the
  band/noise-floor invariants). Verified: `validate()` → `PASS: 0 problem(s)`, 9 kernels, 331 shapes,
  4 transfer kernels.
- **Oracle / quick-autotune / full-autotune knob:** `HELION_AUTOTUNE_EFFORT=none|quick|full` (real
  helion setting, `helion/runtime/settings.py:534`). `none` = seed-only, no search; `quick` = fast
  triage oracle; `full` = converged oracle. Oracle runner: `_lab/harness/run3_oracle.py`
  (`--kernel <k> --M <m> --N <n>`; caches winners to `_lab/logs/run3/oracle_cache.json`). Force a
  re-tune with `HELION_FORCE_AUTOTUNE=1` (real setting, `settings.py:520`).
- **Control / read-codegen flags (verified real settings):**
  `HELION_DISABLE_AUTOTUNER_HEURISTICS=1` (the unseeded-default control arm, `settings.py:528`),
  `HELION_PRINT_OUTPUT_CODE=1` / `TORCH_LOGS=output_code` (read generated Triton).
- **Seeded arm:** `HELION_PROMOTE_REDUCTION_SEED=1` is **NOT a helion setting** (confirmed: it does not
  appear anywhere in `helion/`). It is read only by the `_lab/bench/run_seeded.py` wrapper, which sets
  `promote_seed_to_default=True` on the two reduction heuristics. The seeded arm therefore works **only
  when driven through `run_seeded.py`** — setting it on a bare `benchmarks/run.py` does nothing.

## Canonical commands (copy-paste — all VERIFIED working 2026-06-08)

**Import / GPU smoke (proves wiring resolves to the worktree):**
```
cd /tmp && CUDA_VISIBLE_DEVICES=0 \
  PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-fork-pr-with-lab \
  /home/calebkim/.conda/envs/helion/bin/python -c \
  "import helion; print(helion.__file__); import torch; print(torch.cuda.get_device_name(0))"
```

**Bare seed (Product-A; no autotune) — codegen + correctness + timing:**
```
cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
  PYTHONPATH=/home/calebkim/helion-new-heuristics/helion-fork-pr-with-lab \
  /home/calebkim/.conda/envs/helion/bin/python <your_script>
```
(Verified on rms_norm(2048,4096) fp32: default/seed config = persistent `reduction_loops=[None]`,
`block_sizes=[1]`, `num_warps=4`; `max_abs_err = 1.91e-6`; seed latency 35.3 µs.)

**TritonBench — the HEADLINE harness (3-arm table + accuracy gate + CSV):**
```
cd /home/calebkim/helion-new-heuristics/helion-fork-pr-with-lab && \
CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none PYTHONPATH=$PWD \
  /home/calebkim/.conda/envs/helion/bin/python benchmarks/run.py \
  --kernel rms_norm --metrics latency,accuracy,speedup --precision fp32 --M 16384 --H 8192 --num-inputs 1
```
(rms_norm uses `--M/--H`, **not** `--shapes`, in this tritonbench version. Verified: produces the
helion-vs-torch.compile-vs-liger-vs-triton table with `±` spreads + accuracy gate.)

**Seeded arm directly (seed promoted → default, via the wrapper):**
```
cd /home/calebkim/helion-new-heuristics/helion-fork-pr-with-lab && \
CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none HELION_PROMOTE_REDUCTION_SEED=1 PYTHONPATH=$PWD \
  /home/calebkim/.conda/envs/helion/bin/python _lab/bench/run_seeded.py \
  --kernel rms_norm --metrics latency,accuracy --precision fp32 --M 16384 --H 8192 --num-inputs 1
```
(Verified: prints `[run_seeded] reduction seed promoted to default config (seeded arm)`.)

**Portable 3-arm driver (set `HELION_PY`!) — works for kernels whose operator doesn't need `--shapes`:**
```
cd /home/calebkim/helion-new-heuristics/helion-fork-pr-with-lab && \
HELION_PY=/home/calebkim/.conda/envs/helion/bin/python HELION_WORKTREE=$PWD \
  /home/calebkim/.conda/envs/helion/bin/python _lab/bench/seed_vs_tc.py <kernel...>
```
(See the §"mismatch" note — fails on rms_norm in this tritonbench version.)

## Verified results — sanity benchmark (rms_norm, fp32, H100 idx 0, do_bench)
Big memory-bound shape so launch overhead (~few µs) is noise; spreads confirm it (±0.3–1.3% at the big
shape vs ±10% at tiny (2048,1024)):

| shape (M,H) | helion **unseeded default** | helion **seeded** | torch.compile default | liger | hand triton |
|---|---|---|---|---|---|
| (16384, 8192) fp32 | 0.682 ms | **0.499 ms** (±0.5%) | 0.494 ms | 0.496 ms | 0.499 ms |

All accuracy = 1. The reduction-seed heuristic improves rms_norm at this shape **0.682 → 0.499 ms
(~1.37×)**, reaching torch.compile parity. (Tiny-shape reference: at (2048,2048) helion seeded was
0.0146 ms / 4.12× speedup over the llama_rms baseline, but with ±10% spread — don't draw conclusions
from sub-0.1 ms shapes.)

## Precision / robustness footguns (verified elsewhere; re-confirm per kernel)
- **Some operators override `--precision`** (welford has historically defaulted to bf16, softmax to
  fp16 even with `--precision fp32`). Assert the dtype actually took on every call — don't trust the
  flag alone.
- **Cold compile is slow** (tens of seconds; first-input compiles dominate). Don't mistake it for a
  hang. Print parseable sentinel lines (`RESULT_*`) so headline numbers survive log noise.
- **Trust TritonBench `do_bench` (±~2%)** over naive CUDA-event timing for sub-0.1 ms shapes (~±12%) —
  the referee measures via TritonBench.

## Resume state — the gated log (read these to pick up where the last run left off)
Source of truth (method §6.1) — trust them over assumptions about "what we found":
- **Worker notebook (reasoning trace, per-shape status):** `_lab/run3_notebook.md` (prior run);
  `_lab/dtype_notebook.md` + `_lab/DTYPE_REPORT.md` for the bf16/fp16 dtype climb.
- **Hub log (orchestration state / next action):** `_lab/HUB_BATON.md` + `_lab/HUB_LOG.md`.
- **Ledger:** `_lab/ledger.json` (key `run3`: `champion_advances`, `gate_verdicts`, `results`,
  `PARITY_gap_list`; `oracle_cache` is a TOP-LEVEL key). On-disk oracle cache:
  `_lab/logs/run3/oracle_cache.json` (written by `run3_oracle.py`).
- Gate frames to reuse verbatim: `_lab/prompts/gate-prompts.md`.

If you start a NEW run (different heuristic / fresh slate), create your own `*_notebook.md` + ledger
rather than overwriting these.

## Reference (optional)
`_lab/SETUP.md` is the closest prior devserver-family doc (same conda env + `/home/calebkim/...` paths,
but an older worktree `wt-reduction` / branch `reduction-heuristics-autotuner`). `local-setup.md` and
`server-specific-setup.md` describe the **other** machine (`/home/dev/...` venv box, single GPU,
`origin`=fork). Where any report disagrees with the live code, **the code wins** — read `triton.py` /
`config_spec.py` for ground truth.
