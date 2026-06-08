# Server-Specific Setup (companion to `reduction-heuristics-standalone.md`)

> The work-order prompt is deliberately **path-free** so it stays portable. THIS file holds the
> machine-specific facts for the server it was written on (probed live **2026-06-03**). If `helion.__file__`
> or the paths below don't resolve on YOUR machine, re-discover them per the prompt's Step 0 — do **not**
> trust these blindly. Everything here was verified by running it, not assumed.

---

## 1. Interpreter / environment

- **Python:** `/home/dev/helion/.venv/bin/python` (a **venv**, py 3.12.3 — NOT conda). The system
  `python`/`python3` lack the deps.
- **Deps (verified import):** torch `2.13.0.dev…+cu130`, triton `3.7.0`, `torch.cuda.is_available() == True`.
- **NEVER `pip install`** (repo rule + the env is already complete). To use the worktree's code, override
  `PYTHONPATH`, never reinstall.

## 2. GPU

- **1× NVIDIA H100 80GB, index 0, idle** (0 MiB used at probe time). `CUDA_VISIBLE_DEVICES` was unset.
- **Pin every run** with `CUDA_VISIBLE_DEVICES=0`, and `nvidia-smi` to confirm it's idle before trusting a
  number. **One GPU ⇒ time serially** (the prompt's timing-queue rule); never two `do_bench` runs at once.

## 3. The two checkouts + the import-wiring gotcha (READ — this silently bit run 1)

There are **two** helion checkouts on this box, and they serve different roles:

| Path | What it is | Role |
|---|---|---|
| `/home/dev/local/helion-reduction-heuristics-run2` | **your worktree** (branch `reduction-heuristics-run2`, HEAD `a99512e1`) | the heuristic + `_lab/` live here; **edit `helion/` + `examples/` here** |
| `/home/dev/local/helion` | the **original checkout** (branch `reduction-heuristic-plan`, HEAD `1ec3193d`) | holds the **editable install** + the **tritonbench operators**; edit operators **here** |

`helion` is installed **editable as a plain path entry** pointing at the original checkout, so by default
`import helion` resolves to the ORIGINAL, not your worktree. **The fix (verified):**

```
cd /tmp   # a cwd that is NOT a checkout root (a checkout-root cwd shadows everything via sys.path[0])
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 \
  /home/dev/helion/.venv/bin/python <script-or-benchmarks/run.py ...>
```

- **Verified:** with the above, `helion.__file__` → `…/helion-reduction-heuristics-run2/helion/__init__.py`
  (your worktree). ✅
- **But `tritonbench` resolves to `/home/dev/local/helion/benchmarks/tritonbench/…` EVEN WITH the worktree
  on `PYTHONPATH`** (separate editable; the worktree has no `benchmarks/tritonbench/` to shadow it). ✅ confirmed.
  ⇒ **Operator-level edits (TritonBench `operators/…`, new baselines, new kernels) go in the ORIGINAL
  checkout**, while the Helion kernel-under-test runs from your worktree. Always re-prove both with a
  sentinel print after any edit (Step 0).

## 4. Git facts (verified — do not trust hard-coded SHAs elsewhere; re-`git log`)

- Your worktree: branch **`reduction-heuristics-run2`**, HEAD **`a99512e1`** (`.git` is a *file* →
  `gitdir: /home/dev/local/helion/.git/worktrees/…`).
- Remotes (note the inversion): **`origin` = `git@github.com:calebmkim/helion.git`** (a fork);
  **`upstream` = `https://github.com/pytorch/helion.git`**.
- **Commit early/often locally; NEVER `git push`** (the human handles pushes; "yolo" is local-only).

## 5. Canonical commands (copy-paste, verified working)

**Bare seed (Product-A measurement; no autotune):**
```
cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
  PYTHONPATH=/home/dev/local/helion-reduction-heuristics-run2 /home/dev/helion/.venv/bin/python <your_script>
```
(`helion.kernel(fn, config=seed)` → runs the seed, no search; verified correct on rms_norm(2048,4096),
max_abs_err 1.9e-6.)

**TritonBench (the HEADLINE harness — verified working):**
```
cd /home/dev/local/helion-reduction-heuristics-run2 && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
  PYTHONPATH=$PWD /home/dev/helion/.venv/bin/python benchmarks/run.py \
  --kernel rms_norm --metrics latency,accuracy,speedup --precision fp32 --num-inputs 2
```
Produces the Helion-vs-torch.compile-vs-liger table with `±` spread + accuracy gate + CSV.

**Oracle (quick autotune; ~5 min/shape — it's the expensive resource, CACHE it):**
```
… HELION_FORCE_AUTOTUNE=1 HELION_AUTOTUNE_EFFORT=quick … <script that builds the kernel and calls it>
```
Verified: 3 generations ran, GPU ~91%, returned a winning config (`pid_type='flat'`).

## 6. Benchmarking state — ALL 9 curriculum kernels now wired (fixes applied 2026-06-03)

Probed live via `benchmarks/run.py`, then the gaps were **fixed and re-verified** (every row below
re-ran to `helion_*-accuracy == 1` with a tc-default baseline present). The transfer kernels remain for the
agent to write (Goal 5).

| Kernel | TritonBench | tc-**default** baseline (the `G` floor) | Notes |
|---|---|---|---|
| rms_norm, layer_norm, softmax, sum, cross_entropy | ✅ | ✅ (`torch_compile_no_autotune_*`) | upstream — untouched |
| **welford** | ✅ | ✅ **added** `torch_compile_welford_default` | precision footgun — see §7 |
| **kl_div** | ✅ | ✅ **added** `torch_compile_kl_div_default` | — |
| **jsd** | ✅ | ✅ **added** `torch_compile_jsd_default` | — |
| **long_sum** | ✅ **newly wired** | ✅ (reuses `sum` operator's `torch_compile_no_autotune_sum`) | **MUST pass `--reduce-dim 1`** — see below |
| transfer (`tv_distance`, `argmax`, `l2_norm`, `minmax_normalize`) | ❌ (expected) | ❌ | you write+wire these (Goal 5) |

### What was changed (so you can reproduce / extend it)

**(a) tc-default baselines for welford, kl_div, jsd** — the objective is `G = tc_default / seed`, but these
three operators shipped only `torch_compile_*` at `mode="max-autotune-no-cudagraphs"` (that's the *oracle*
side, not the floor). Added a `torch_compile_<op>_default` method to each operator — an exact clone of the
existing `torch_compile_<op>` with the `mode=` kwarg removed (so Inductor uses its default codegen path).
- **Edited in the ORIGINAL checkout** (operators live there — see §3):
  `/home/dev/local/helion/benchmarks/tritonbench/tritonbench/operators/{welford,kl_div,jsd}/operator.py`.
- Patches saved to `_lab/harness/patches/torch_compile_{welford,kl_div,jsd}_default.patch` (alongside run-1's
  `torch_compile_rms_norm_default.patch`). These are **NOT in the worktree's git** (operators are a separate
  checkout) — **re-apply them if the original checkout is reset/re-cloned.**

**(b) long_sum wired into the headline harness** — it had `examples/long_sum.py` but no `KERNEL_MAPPINGS`
entry and no operator, so run 1/2 only ever benched it via custom scripts. It is a 2D `[m,n]→[m]` row-sum,
structurally identical to `sum`, so it now **reuses the `sum` tritonbench operator** (free tc-default +
accuracy gate). Two edits, both **in the worktree** (tracked by the worktree's own git — commit them):
- `examples/long_sum.py`: added a `longsum_tritonbench(tb_op, x)` hook (mirrors `sum_tritonbench`; binds the
  persistent `longsum` variant) + a `from typing import Callable` import.
- `benchmarks/run.py`: added a `"long_sum"` `KERNEL_MAPPINGS` entry pointing at
  `tritonbench.operators.sum.operator` + the new hook; **and fixed `run_kernel_variants` to derive the
  tritonbench `--op` name from the MODULE PATH, not the dict key** (it previously did
  `kernel_name.removesuffix("-bwd")`, which sent `--op long_sum` → "operator not found"). The fix:
  `tritonbench.operators.<op>.operator → <op>`, falling back to the de-bwd'd key. Behavior-neutral for every
  existing kernel (their key already equals their op-dir name); only long_sum needed the divergence.

### ⚠️ long_sum invocation requirement (non-obvious — it silently fails accuracy otherwise)

The `sum` operator's baseline is `torch.sum(x, dim=self.reduce_dim)` with **`reduce_dim` defaulting to `None`
(reduce-to-scalar)**. `longsum` is a **row** reduction, so you **MUST pass `--reduce-dim 1`** or the
operator's reference computes a scalar while Helion returns a row vector → shape mismatch →
`helion_longsum_tritonbench-accuracy = 0` (a false failure; the kernel is numerically correct — verified
independently). Drive large N with `--shapes` (the operator ignores the curriculum's M,N otherwise). Verified
working:
```
cd /home/dev/local/helion-reduction-heuristics-run2 && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none \
  PYTHONPATH=$PWD /home/dev/helion/.venv/bin/python benchmarks/run.py --kernel long_sum \
  --metrics latency,accuracy --precision fp32 --reduce-dim 1 --shapes "256,65536;64,1048576"
```
(Result: `helion_longsum_tritonbench-accuracy = 1`, tc-default `torch_compile_no_autotune_sum` present.
Ignore `triton_sum-accuracy = 0` — that's the operator's *own* hand-written Triton kernel not supporting this
shape, unrelated to the Helion path.)

## 7. Precision / robustness footguns (verified)

- **Some operators override `--precision`.** Verified: **welford defaulted to bfloat16 inputs** even with
  `--precision fp32`; softmax historically defaults to fp16. **Assert fp32 on every call and confirm it took**
  — don't trust the flag alone.
- **Cold-compile is slow:** ~47 s for a single welford input first-run; tens of seconds is normal cold. Don't
  mistake slow-compile for a hang. (Autotune logs progress bars to stderr; long jobs auto-background — read
  the result file, print parseable sentinel lines like `RESULT_*` so they survive log noise.)
- **Trust TritonBench `do_bench` (±~2%)** over naive CUDA-event timing for sub-0.1 ms shapes (~±12%) — the
  referee measures via TritonBench.

## 8. Orchestration primitives (verified on this Claude Code Agent/Team harness, 2026-06-03)

(Full detail is in the prompt's Step-0 "VERIFIED" block; summary here.)
- **Continue-in-place ✅** — `SendMessage` to a "completed" background agent resumes its SAME context. Realize
  the persistent worker this way.
- **Nesting ❌** — a spawned agent (even a teammate) has **no `Agent` tool**; it cannot spawn. ⇒ the hub
  stands up investigators as **peers up front**; the worker messages them.
- **Teams/peer DMs ✅ + shared task list ✅.** Build it as one team: hub=lead (only spawner), worker=persistent
  peer, investigators=standing peers, gates=hub spawns fresh per claim.
- **Pass `model:"opus"` on EVERY spawn** — omitting it silently downgrades to opus-4-7 (does NOT inherit the
  lead's model). `"opus"` resolves to the full `us.anthropic.claude-opus-4-8[1m]` (1M context).
- **No per-spawn effort knob** — encode rigor in the prompt; max-effort is a session-level setting, not an
  `Agent` param. `Workflow` is hub-only.
- Peer replies are **async** — "my turn ended, no reply yet" ≠ "no answer."
