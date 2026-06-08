# LOCAL SETUP — this machine, this run (concrete facts)

`hillclimb-method.md` holds the durable principles; this file holds the specifics it defers to.
Reconcile against the live tree before trusting any path/SHA here (these drift).

## Paths & interpreter
- **Worktree (source of truth):** `/home/dev/local/helion-pr-with-lab`. The heuristic is at
  `helion/_compiler/autotuner_heuristics/triton.py`; facts at `helion/autotuner/config_spec.py`;
  examples at `examples/`. Confirm with `git rev-parse --show-toplevel`; don't hardcode downstream.
- **Interpreter:** `/home/dev/helion/.venv/bin/python` (shared venv, Python 3.12, has every dep —
  torch nightly, triton dev, tritonbench). **Never `pip install`.**
- **Run scripts from `cwd=/tmp`** with `PYTHONPATH=/home/dev/local/helion-pr-with-lab` so
  `import helion` resolves to the worktree. Assert `helion.__file__` is under the worktree at the
  top of every script.

## GPU — DEDICATED this run
- One **NVIDIA H100 80GB**. **Not shared by default** for this job: the human will tell you
  explicitly if another agent needs it. So you may **run timings back-to-back without idle-gating**;
  the contention guard / `_wait_idle` machinery in the bench scripts is unnecessary here (a single
  `nvidia-smi` before a headline timing is fine cheap insurance).
- **Still NEVER launch a detached/background GPU job** (the §1 hard rule stands regardless of
  sharing — they die silently and never notify). Long oracles: foreground, one shape at a time,
  JSON-checkpointed.

## Git — commit frequently
- Branch `reduction-pr-with-lab` (carries one `_lab/`-only commit on top of the PR head — never
  push this lab branch onto the PR ref). `origin` = `git@github.com:calebmkim/helion.git` (the
  fork), `upstream` = `pytorch/helion` (**never push upstream**).
- **Commit to your own branch freely and often** — no need to ask. Bank each green step. Do **not**
  `git push` (the human updates the PR).

## Key scripts (reuse) — paths are not guaranteed portable; use your judgement
**These lab scripts were authored on prior machines, so some hardcode paths/`startswith` asserts
that won't match every server. Don't expect 100% portability — if a script fails on an import or a
path, just repoint it (`git rev-parse --show-toplevel`, or set `HELION_WORKTREE`) and move on; it
doesn't have to be perfect, try your best.** Two tiers:
- **Headline bench scripts — already made portable** (`seed_vs_tc.py`, `sweep.py`,
  `bare_fwd_seed_vs_tc.py`): they derive the worktree root from `__file__` and honor
  `HELION_WORKTREE` / `HELION_PY`. Run as-is here. The `run3_oracle.py` /
  `run3_task1_verify_after_edit.py` *bodies* are likewise root-from-`__file__` (only their docstring
  example commands cite a dead `PYTHONPATH` — ignore those).
- **The many per-lever probe scripts** (`_lab/harness/AUDITOR_*`, `PERF_INV_*`, `run3_*_ab.py`, etc.)
  still hardcode old roots (`/home/dev/local/helion-pr-edit`, `.../helion-reduction-heuristics-run2`,
  `/home/calebkim/...`). They're throwaway A/B probes, not the headline — repoint the one you need
  when you need it, don't bulk-fix them.

- **Behavior oracle (compile-time, no GPU):** `_lab/harness/run3_task1_verify_after_edit.py` —
  records the emitted seed config across the curriculum; BEFORE/AFTER diff, 0 config-diffs =
  behavior-preserving. (Needs a dtype axis added for this task — see `dtype-task.md`.)
- **Headline bench (tritonbench path):** `_lab/bench/run_seeded.py` (promotes seed → default at
  `HELION_AUTOTUNE_EFFORT=none`), driven by `_lab/bench/seed_vs_tc.py` / `_lab/bench/sweep.py`
  (3-arm: seeded / unseeded-default / tc-default). `_lab/bench/bare_fwd_seed_vs_tc.py` is the
  bare-forward cross-check.
- **Curriculum:** `_lab/prompts/shapes_v3_draft.py` (`SHAPES` + `TRANSFER`; `validate()` enforces
  the band/noise-floor invariants). fp32-baked today — add a dtype axis per the task file.
- **The oracle / quick-autotune / full-autotune knob:** `HELION_AUTOTUNE_EFFORT=none|quick|full`
  (real helion setting, `helion/runtime/settings.py`). `none` = seed-only, no search (Product A);
  **`quick`** = the fast triage oracle (Step 2 cheap-first); **`full`** = the converged oracle you
  bank against (Step 3/4, anti-giving-up). When the method says "run the oracle," this is the knob.
  The oracle **runner** is `_lab/harness/run3_oracle.py` (`--kernel <k> --M <m> --N <n>`, reads
  `HELION_AUTOTUNE_EFFORT` from env, caches winners to `_lab/logs/run3/oracle_cache.json`).
- **Other env flags:** `HELION_PRINT_OUTPUT_CODE=1` / `TORCH_LOGS=output_code` (read generated
  Triton), `HELION_DISABLE_AUTOTUNER_HEURISTICS=1` (the unseeded-default control arm).
- **Seeded arm:** `HELION_PROMOTE_REDUCTION_SEED=1` is **NOT a helion setting** — it is read only by
  the `_lab/bench/run_seeded.py` wrapper, which monkey-patches `promote_seed_to_default=True`. The
  seeded arm therefore works **only when driven through `run_seeded.py`**; setting it on a bare
  `benchmarks/run.py` does nothing.

## Resume state — the gated log (read these to pick up where the last run left off)
These are the source of truth (method §6.1) — trust them over any assumption about "what we found":
- **Worker notebook (reasoning trace, per-shape status, deferred hard-pile):**
  `_lab/run3_notebook.md` (the live worker source-of-truth from the prior run).
- **Hub log (orchestration state, phase, what's banked, next action):** `_lab/HUB_BATON.md` (the
  save-game pointer) + `_lab/HUB_LOG.md` (fuller history).
- **Ledger:** `_lab/ledger.json`. Key `run3` holds `champion_advances`, `gate_verdicts`, `results`,
  `PARITY_gap_list`. The **`oracle_cache` is a TOP-LEVEL key** (not under `run3`); the on-disk oracle
  cache is `_lab/logs/run3/oracle_cache.json` (written by `run3_oracle.py`).
- Gate frames to reuse verbatim: `_lab/prompts/gate-prompts.md`.

If you start a NEW run (different heuristic / fresh slate), create your own `*_notebook.md` +
ledger rather than overwriting these; keep them as the prior lineage.

## Reference (optional, not required)
The deep `_lab/` `FINAL_*.md` / `HANDOFF_*.md` are *optional* lineage. Where any of them disagree
with the live code, **the code wins** — read `triton.py` / `config_spec.py` for ground truth, not
the reports.
