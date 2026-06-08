# Agent handoff: (A) verify your change is config-safe, (B) push to the PR-diff branch

> For any agent editing `helion/` in this repo. Two procedures: **(A)** prove your change
> didn't alter the seed configs the heuristic emits (the 259-shape diff), and **(B)** reflect
> your change on the clean PR-diff branch on the fork. Both verified working 2026-06-05.

## Non-negotiables (read first)
- **GPU is shared and may be busy.** The verify step **binds kernels → allocates CUDA memory**
  (up to ~8 GB for the widest shapes). It does NOT autotune/do_bench, but a bind still allocates.
  **`nvidia-smi` first; only run when idle. Run foreground, one process, serial.** Never launch a
  detached/background GPU job here (it gets killed silently).
- **`origin` = the fork (`git@github.com:calebmkim/helion.git`). `upstream` = pytorch/helion.**
  **NEVER push to `upstream`.** Only ever push to `origin`.
- **Interpreter:** `/home/dev/helion/.venv/bin/python` (shared venv; has all deps). **Never `pip install`.**
- Run from **`cwd=/tmp`** (a non-checkout dir) so `import helion` resolves to the worktree via
  `PYTHONPATH`, not the editable install.
- Derive the worktree root; don't hardcode it: `WT=$(git -C <your-checkout> rev-parse --show-toplevel)`.

---

## (A) The 259-shape config diff — prove your change is behavior-preserving

The recorder binds all 259 curriculum shapes (9 kernels × train/val/test) and records, per shape,
the **raw seed config** + **normalized config** the heuristic emits (compile-time only). A
behavior-preserving change must produce **0 config diffs**. Configs are **shape-only & deterministic**
(random input values don't matter), so a clean change reproduces them exactly.

The recorder is committed at `_lab/harness/run3_task1_verify_after_edit.py` and always writes to
`_lab/logs/run3/task1_seed_configs_AFTER.json`.

```bash
WT=$(git rev-parse --show-toplevel)               # run from inside your checkout
PY=/home/dev/helion/.venv/bin/python
REC="$WT/_lab/harness/run3_task1_verify_after_edit.py"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # MUST be idle before proceeding

# 1) BASELINE — record on a CLEAN tree (BEFORE your change), then stash the result aside.
#    (If you've already edited: `git stash` first, record, then `git stash pop`.)
cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none PYTHONPATH="$WT" "$PY" "$REC"
cp "$WT/_lab/logs/run3/task1_seed_configs_AFTER.json" /tmp/task1_BEFORE.json

# 2) Apply your source change. Then record AGAIN (overwrites the AFTER file).
cd /tmp && CUDA_VISIBLE_DEVICES=0 HELION_AUTOTUNE_EFFORT=none PYTHONPATH="$WT" "$PY" "$REC"

# 3) DIFF baseline vs after.
"$PY" - "$WT" <<'PYEOF'
import json, sys
WT = sys.argv[1]
a = {(r["kernel"],r["M"],r["N"]): r for r in json.load(open("/tmp/task1_BEFORE.json"))["rows"]}
b = {(r["kernel"],r["M"],r["N"]): r for r in
     json.load(open(f"{WT}/_lab/logs/run3/task1_seed_configs_AFTER.json"))["rows"]}
assert set(a)==set(b), f"shape-set changed: {set(a)^set(b)}"
cfg=fact=0
for k in sorted(a):
    for f in ("raw_seed","normalized_cfg","n_seeds","heuristics_fired","classification"):
        if a[k].get(f)!=b[k].get(f):
            cfg+=1; print(f"CONFIG DIFF {k}.{f}:\n  BEFORE={a[k].get(f)}\n  AFTER ={b[k].get(f)}")
    if (a[k].get("reduction_fact") or {})!=(b[k].get("reduction_fact") or {}):
        fact+=1
print(f"\nrows={len(a)}  CONFIG diffs={cfg}  reduction_fact diffs={fact}")
print("VERDICT:", "CONFIGS UNCHANGED ✅" if cfg==0 else "CONFIGS CHANGED ❌ — investigate before pushing")
PYEOF
```

**Reading the result:**
- `CONFIG diffs=0` → your change preserved every emitted seed. Safe.
- `CONFIG diffs>0` → your change altered seeds. If that's **intended** (you meant to change behavior),
  inspect each diff and confirm it's what you wanted. If **unintended**, you have a bug — fix before pushing.
- `reduction_fact diffs>0` with `CONFIG diffs=0` is normal **only** if you intentionally renamed/added/
  removed a `ReductionFact` field; the seeds are what ship, the facts are inputs.

> Why baseline-on-your-own-clean-tree (not the committed `*_AFTER.json`): it sidesteps schema drift —
> baseline and after are recorded with the **same** field schema (your tree ± your one change), so the
> only delta is your edit. Shortcut if you trust the committed artifact is current with `HEAD`:
> `git show HEAD:_lab/logs/run3/task1_seed_configs_AFTER.json > /tmp/task1_BEFORE.json` instead of step 1.

---

## (B) Push to the PR-diff branch (`reduction-seed-heuristic-run2`)

**Model:** the PR-diff branch is a **single squashed commit** on the fork ("what the PR looks like now"),
holding the same `helion/`+`test/` source as the dev branch `reduction-heuristics-run2` but **without the
`_lab/` cruft**. To update it you **mirror the dev source into it and amend the squash commit**, preserving
its message + author/date, then **force-push with a lease**.

**Prereq:** your change is committed on **dev** (`reduction-heuristics-run2`) and pushed (`git push origin
reduction-heuristics-run2` — a plain fast-forward, no force). The PR branch should always mirror dev's source.

```bash
WT=$(git rev-parse --show-toplevel); PY=/home/dev/helion/.venv/bin/python
git -C "$WT" fetch origin reduction-seed-heuristic-run2
LEASE=$(git -C "$WT" rev-parse origin/reduction-seed-heuristic-run2)   # current remote tip = lease target

# Use a TEMP worktree on the PR branch — NEVER `git checkout` the PR branch in your working tree
# (a stray checkout clobbers your tree; this is a hard-won lesson).
PRWT=/tmp/pr-wt-$$; rm -rf "$PRWT"
git -C "$WT" worktree add "$PRWT" reduction-seed-heuristic-run2

# Mirror dev's source into the PR branch (overwrites helion/+test/ with dev's exact versions).
# Idempotent + order-independent: PR source becomes == dev source regardless of prior PR state.
git -C "$PRWT" checkout reduction-heuristics-run2 -- helion/ test/

# Sanity: confirm only intended files changed, it compiles, it lints.
git -C "$PRWT" status --short
"$PY" -m py_compile $(git -C "$PRWT" diff --name-only | grep '\.py$' | sed "s|^|$PRWT/|")
/home/dev/helion/.venv/bin/ruff check "$PRWT/helion/" 2>&1 | tail -2

# Amend the squash commit (keeps Caleb's message + author/date) and force-push WITH LEASE.
git -C "$PRWT" add -A
git -C "$PRWT" commit --amend --no-edit
git -C "$PRWT" push --force-with-lease=reduction-seed-heuristic-run2:$LEASE origin reduction-seed-heuristic-run2

# Clean up the temp worktree.
git -C "$WT" worktree remove "$PRWT" --force
```

**Verify it landed:**
```bash
git -C "$WT" fetch origin reduction-seed-heuristic-run2
for f in $(git -C "$WT" diff --name-only reduction-heuristics-run2 origin/reduction-seed-heuristic-run2 -- helion/ test/); do
  echo "MISMATCH: $f"; done; echo "(no MISMATCH lines = PR source == dev source ✅)"
```

### ⚠️ Multi-agent coordination (important)
- **Commit all source changes to the shared dev branch `reduction-heuristics-run2` first.** The PR branch
  is derived from dev; if your change isn't on dev, the mirror step won't include it.
- **If several agents push the PR branch, serialize.** Each push amends+rewrites the single squash commit.
  `--force-with-lease` makes a racing push **fail loudly** (not silently clobber) — if it fails,
  `git fetch`, re-capture `$LEASE`, redo the mirror+amend, retry. Because the mirror step copies dev's
  source wholesale, whoever pushes last yields PR source == dev source, provided everyone's change is on dev.
- **Never `git push` to `upstream`.** Only `origin`.
```
