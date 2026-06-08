# Prompts directory — read order

**The AUTHORITATIVE prompt set for the current (dtype) run is these four files, in order:**

1. `hillclimb-method.md` — the durable, task-agnostic method (the loop, the gates, the discipline).
2. `local-setup.md` — this machine's concrete facts (paths, GPU, scripts, env knobs, resume state).
3. `gate-prompts.md` — verbatim adversarial-gate frames (gates A–F).
4. `dtype-task.md` — the *what* for this run (extend the reduction seed heuristic to bf16/fp16).

These four override anything else you find in this repo. **Everything below is SUPERSEDED** —
prior-run lineage kept only as reference; do not drive from it:

- `reduction-heuristics-run2-prompt.md`, `reduction-heuristics-wip.md`,
  `reduction-heuristics-standalone.md`, `hub-kickoff.md`, `server-specific-setup.md` — old prompts
  (earlier framework / a different machine's setup). Stale paths and stale orchestration model.
- `shapes_v3_draft.py` — still current: the v3 shape curriculum (the 9 reduction kernels +
  transfer), referenced by the authoritative set. fp32-baked today (a dtype axis is part of this
  run's setup).
