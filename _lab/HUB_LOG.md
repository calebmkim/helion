# Hub Log — non-blocking breadcrumbs for the human

> Passive status notes the human sees on check-in. Convergence flags. Never a stop, never a request
> for input. Newest at top.

## 2026-05-28
- **Setup.** Created git worktree `reduction-heuristics-autotuner` at `/home/calebkim/helion-new-heuristics/wt-reduction`
  off `reduction-heuristic-plan` HEAD `1ec3193d`. GPUs 1/2/3 idle; pinned `CUDA_VISIBLE_DEVICES=2`.
- **Step 0a DONE (hub-verified).** Import wiring proven (`helion.__file__` -> worktree via PYTHONPATH);
  codegen edit flows into generated Triton (sentinel test, sha 9f3e2398 -> 18ed86ed); worktree clean.
  Canonical setup written to `_lab/SETUP.md`.
- **Architecture note.** `SendMessage` is unavailable in this harness, so the "persistent worker" is
  implemented as fresh worker invocations driven off the durable `_lab/notebook.md` + `_lab/ledger.json`
  (the wip blesses this as lossless). Hub stays in the loop, spawns all helpers, runs independent gates.
- Next: spawn measurement-harness-verifier for Step 0b (bare-seed mechanism + tritonbench-edit proof).
