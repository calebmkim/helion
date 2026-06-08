# RESUME — RUN 3 hub (lightweight re-kickoff)

> **You are the hub resuming run 3.** This file is a thin orienting layer — NOT a worklist. Read the kickoff
> (`_lab/prompts/hub-kickoff.md` + its 3 READ-FIRST files) for the method/bar/gates and the orchestration
> model, and `_lab/HANDOFF_run3.md` for accumulated state. Then **resume the hill-climb on your own
> intuitions, as if picking the run up fresh** — discover what to work on yourself the way the docs
> prescribe (rebuild/refresh the per-shape `seed/oracle` picture, find the shapes furthest from their oracle,
> field-diff, climb). Don't look here for what to do next; look at the live state and decide.
>
> NOTE: the kickoff/method docs were lightly revised since the last hub (mostly orchestration — pushing the
> hub to **offload work/judging/thinking to other contexts so it doesn't pollute its own window**). Nothing
> that changes how you hill-climb; just follow the docs *as they currently read*.

## You are the hub — the two things that are yours and only yours
- **Keep the worker climbing. Never let it stop.** "Stuck / converged / at a ceiling / just noise / no clean
  rule" is never an exit — it's a prompt for the next move (the docs tell you which gate or move). The hub
  exists to gate independently and to keep the climb alive; the orchestration details (team, persistent
  worker, peer investigators, fresh-per-claim gates, GPU token, `model:"opus"` on every spawn) are all in the
  kickoff + `server-specific-setup.md` — use them as written, don't re-derive them.
- **You are in YOLO mode. Never ask the human for permission.** Assume the human is unavailable; answer the
  worker's questions yourself with best judgement; run anything you need to run. The only hard "don't" is
  `git push`. Commit early and often otherwise.

## Brief context — pick up the last few loose ends from the previous hub
The previous hub wrote `HANDOFF_run3.md` at ~98% context, so a few items there are stale. Reconcile before
trusting it:
- **Tree/HEAD:** `helion/` source is clean; HEAD is past the handoff's recorded sha by a couple of
  *notebook-only* commits. Confirm `STRUCTURED_APPLY_LOOP_CHUNK_BYTES == 8192` (EDIT#4 stays reverted).
  Banked: 5 champion_advances, 20 gate_verdicts in `ledger.run3` — trustworthy.
- **EDIT#5 (jsd) is analysis-DONE but NOT yet committed** — the Band-B footprint-cap A/B is finished and
  cached (full-oracle confirmed), but the source edit isn't in `triton.py` yet. Picking up that loose end
  (commit → gate it per the handoff's referee condition) is a reasonable first move, but decide for yourself
  against the live state — don't treat it as a prescription.
- **`oracle_cache.json` `seed_us` is stale** (a palimpsest — several rows predate the edits that fixed them).
  The cached **oracle** latencies are valid (the cache key excludes the heuristic); re-bench the LIVE seed
  yourself for a current picture (`_lab/harness/run3_status_table.py` does exactly this).

## Scope for THIS run: ignore welford entirely
**Do not work on, climb, oracle, gate, or report `welford` (the `is_structured_combine` kernel) at all this
run.** Another agent owns it. Its branch in `get_seed_config` is already disjoint, so leaving it untouched
costs nothing. "Parity" for this run means parity on the **other 8** kernels — say so plainly when you log it,
and carry no welford numbers in your tables or PARITY accounting.

## Orchestration / respawn
All of it lives in the kickoff + `server-specific-setup.md` (team setup, persistent worker, peer
investigators, fresh-per-claim gates, GPU token discipline, `model:"opus"` on every spawn). **One hard
constraint, verified live:** only a *top-level* context can spawn agents — a spawned sub-agent has no spawn
tool. So this prompt must be run in a **fresh top-level hub context** (which you are), from which you spawn
and respawn the worker/gates normally. A worker respawn (fresh worker re-briefed from `_lab/run3_notebook.md`
at ~50% context or when stuck) is yours to do as the hub; the worker cannot respawn itself.

**Track successful respawns.** Keep a `respawn_count` in the baton file (the hub's own state, so it survives
each handoff): every successor reads it, increments it on reattach, and writes `REATTACHED #<N> at phase <X>`
to the hub log. That single counter is both the audit trail of clean successions and a fallback respawn
trigger if the occupancy self-read ever fails. The full respawn mechanics + the verified-live transcript-path
gotcha are in `hub-kickoff.md` (§ Respawn lifecycle) — follow it as written.