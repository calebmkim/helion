# Originating prompts / context (dev reference, not shipped)

These files are the originating prompts and shape-curriculum context for the
reduction-heuristics work, copied verbatim from the dev box's `local/` dir
(outside any git checkout) so the branch is self-contained for handoff to
another machine. They are **dev context only** — not part of the shippable diff.

- `reduction-heuristics-run2-prompt.md` — the run-2 driving prompt.
- `reduction-heuristics-wip.md` — running WIP notes/scope.
- `shapes_v3_draft.py` — the v3 shape curriculum (train/val/test/coverage/
  robustness/transfer shapes for the 9 reduction kernels).

**`shapes_v3_draft.py` supersedes the earlier shape lists** referenced in the
two prompt `.md` files — treat it as the current source of truth for shapes.

See `_lab/HANDOFF_run2.md` for the run-2 orientation and `_lab/HANDOFF_run3_perf_dig.md`
for the in-progress run-3 perf dig.
