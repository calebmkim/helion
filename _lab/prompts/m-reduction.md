# TASK (WS2): extend the reduction seed heuristic to M-axis (parameter-gradient) reductions — backward norms

**Read in order (same dir): `hillclimb-method.md` → `local-setup-devserver.md` (THIS box — NOT
`local-setup.md`, which describes a *different* H100 machine) → `gate-prompts.md` → this file.** These four are authoritative; where any other doc/notebook/comment —
or this file — disagrees with them or the live code, those win.

**Resume context — START FRESH; ignore any prior m-reduction climb.** A previous WS2 / m-reduction
hill-climb has happened, and you may encounter its leftovers — a `_lab/ws2_notebook.md`, a `ws2` ledger
key, prior branches/commits/reports/logs. **Treat ALL of it as stale and IGNORE it:** do not read it,
resume from it, or let it shape your hypotheses. This is a genuinely **new** climb — it **overrides the
method's START-HERE "picking up an in-progress run" path** for this run. Start a fresh notebook + ledger
under a **NEW key** (e.g. `ws2-fresh`) so nothing auto-resumes the old run; don't overwrite the
run3/dtype/ws1 lineage either.

**Base:** branch off **`calebmkim/stack/3`** — the head of the submitted reduction PR stack (currently
commit `d6ad1156`, "[autotuner] Triton reduction seed heuristic (generalizable core)", on the current
main). It carries the fact layer (`ReductionFact` + `MemoryOpFact` + `AccumulatorFact` / the
walker-derived split, `reread_eviction_index`, `accumulator_facts`) **and** the generalizable-core
heuristic you extend. (Do **not** use `fork/reduction-seed-heuristic` / `8c3bb634` — that's an older
snapshot on a stale main.) Run on your **own branch/worktree**.

**Deliverable shape:** produce a clean set of changes *on top of* `calebmkim/stack/3` that the human will later
edit to taste and **compress to a single commit** above pr-3. So keep the work clean and **additive**
(your new fact + builder + role-based selection are additive; keep the eligibility-gate /
fact-registration rework minimal), and **leave good information for the squash** — a tight
notebook/ledger trail + the deliverable report (what changed, why, the fact design you chose) so the
commit is easy to curate.

**The existing forward heuristic (9 kernels × 3 dtypes) is your DO-NOT-REGRESS baseline** — your
populator/gate changes can touch it, so guard it with **Gate R (the regression-referee)**, whose
full-matrix config-recorder sweep flags any changed forward cell and re-benches it to floor (below).

This file is a **map, not a checklist** — the *what* + the facts that cost real time to discover. The
*how* (the fact design, the fixes, the order you do things) is **yours** (method §2, be aggressive on
the *infra*). The perf *bar* is the method's, not a maximal one: clear §3's disaster-avoidance floor +
per-(kernel, dtype) geomean; beating tc / oracle-parity is overtime you climb into once the bar holds
(§Step 4) — not the entry bar.

---

## Background — a genuinely new reduction TYPE

The seed heuristic serves single-axis **forward** row-reductions. Backward norm kernels
(`rms_norm_bwd`, `layer_norm_bwd`) reduce over **two** axes:
- **grad_x over N** — per-row, structurally *like* the forward row-reduction. **CONFIRMED by compile
  (2026-06-08): it IS a real `reduction=True` rdim (size=N), but gets NO fact and NO seed today** —
  *precisely because* grad_w shares the body (see KEY MECHANISM). The existing machinery does **not**
  seed grad_x once a second reduction is present; the two-reduction interaction is exactly what breaks
  it.
- **grad_w over M** — the **parameter-gradient**: collapse many rows (the token/batch axis) → a tiny
  `[N]` per-feature gradient. A **"tall→tiny" reduction that is NOT modeled today.**

**KEY MECHANISM — CONFIRMED by compile (`EFFORT=none` fact-dump, 2026-06-08).** `rms_norm_bwd` and
`layer_norm_bwd` register **ZERO `ReductionFact`s** and fall back to the **generic
`_base_default_config()`** (`block_sizes=[32,32], num_warps=4, num_stages=1, pid_type=flat`) — **both
axes UNSEEDED.** Three things conspire, all because the kernel has **two distinct inner reductions in
one body** (N-axis grad_x + M-axis grad_w):
1. **grad_w (M-axis)** reduces over an inner `hl.tile` block (`reduction=False`) accumulated into a
   carried buffer — a non-fact tile-accumulation (no `ReductionFact`).
2. **grad_x (N-axis)** IS a real `reduction=True` rdim (size=N) — but **T1 declines to register it**
   because a second reduction is interleaved (a one-rolled-rdim-per-graph constraint; pin the exact
   decline path in Step 0).
3. **T2** then declines too: its `inner_red` set has TWO non-grid `ReductionLowering` block_ids
   (`len != 1`).

→ 0 facts → `_triton_reduction_eligible` (needs exactly 1) declines → generic default. **`softmax_bwd`
is the clean 1-fact control** (only ONE reduction → a T2 seed `block_sizes=[1,2048,2048], w8`, matching
forward `softmax_two_pass`).
Upside: rms/ln bwd sit on a fully generic seed today, so the headroom is large.

Your job: build the machinery so the **grad_w (M-axis) tile-accumulation reduction** is recognized as a
seedable reduction and gets a good seed, **composed with grad_x's seed into ONE Config**, clearing the
method's bar (§3) on the backward norm kernels.

## Step 0 — confirm the fact-structure baseline (mostly done; don't rediscover it)

The fact-structure half is **already established** (see KEY MECHANISM): rms/ln bwd = 0 facts → generic
default; softmax_bwd = 1 fact → T2. You do NOT need to rediscover it — but DO **re-confirm cheaply**
(`EFFORT=none`, dump `config_spec.reduction_facts` + `accumulator_facts`) after each `device_ir` change,
since flipping that count 0 → {the facts you intend} is the signal your new registration works. (Your
base — the pr-stack 3rd commit — already carries the walker/derived fact layer; confirm
`ReductionFact` / `MemoryOpFact` / `AccumulatorFact` are present at your base commit, then derive
grad_w's fact from `AccumulatorFact` per the machinery section.)

Early on, **profile each bwd kernel's grad_x-vs-grad_w runtime split** (you'll build the backward
harness first — see Where everything lives) — it bounds the prize and tells you which axis to attack.

## The new machinery — from-scratch infra IS the job (method §2)

Build a recognizer for the **"tile-accumulation reduction"** (a `.sum` over an inner device-loop tile
accumulated into a carried buffer — the grad_w pattern). The design is **open-ended and yours**, bounded
by the gates + the hard constraints below. **The fact-doctrine (method §2) is the load-bearing
constraint here — read it before you design:**

- **Obey the fact-doctrine (method §2; the Fact-gate, Gate D, enforces it).** In short: only **walker
  facts** may walk the graph (once, in the collector); your recognizer must be a **derived fact** that
  reads walker facts + trivial structural reads and **never walks the graph itself**. Don't re-derive
  the doctrine here — read §2.
- **grad_w is *literally* a loop-carried accumulator** (a `.sum` over an inner tile into a carried
  `[N]` buffer) → it's exactly what the walker **`AccumulatorFact`** models (one per carried accumulator:
  dim block-ids + itemsize), which **your base already carries** (the pr-stack facts layer). So **derive
  your M-reduction fact from `AccumulatorFact` + structural reads — no new graph walk.** Only if grad_w
  needs graph info `AccumulatorFact` doesn't expose, add a **general, consumer-agnostic field to the
  walker fact** (must survive Gate D Part 1's different-consumer test) — never a bespoke walk in your
  derived builder.
- **The fact's *shape* is yours — scaffolding, not a mandate (constrained only by the fact-doctrine +
  Gate D).** Any of these is fine; pick what's cleanest and most maintainable: a **new standalone
  dataclass** (`MReductionFact`); a **subclass of `ReductionFact`** (`class MReductionFact(ReductionFact)`);
  **just extending `ReductionFact`** with the new fields directly; or a **per-reduction-loop fact**
  unifying T1/T2/M into one list. (NB the existing reread fields — `row_reread`, `reread_eviction_index` — are kernel-/op-wide, not cleanly per-loop, so a naive per-loop
  split is wrong; weigh that against a flat unify.) Whatever the shape, it is a **derived fact**: built
  in a derived-fact builder (e.g. `register_m_reduction_facts`; "**after** T1/T2" is just an ordering
  requirement), reading **walker facts + trivial structural reads** — not other derived facts' outputs.
  Let **Gate D** judge it.
- **HARD CONSTRAINT — all reductions' seeds COMPOSE INTO ONE `Config`.** A kernel emits a single Config
  — shared scalars (`num_warps`, `num_stages`, `pid_type`) + positional `block_sizes`/`reduction_loops`
  lists. So grad_x's preferred `num_warps` and grad_w's must be **reconciled into one value** — a real
  design decision (max? N-dominant? a new fact-keyed rule?). Pick it, justify it, **measure it**, and
  put it through **Gate H** (is the reconciliation rule a faithful, general key, or a curriculum fit?).
- **You must register BOTH reductions, then seed both — THREE blockers to rework (confirmed in Step
  0).** Today the kernel gets 0 facts because: (1) **T1 declines the real N-axis rdim** when a 2nd
  reduction is interleaved (the one-rolled-rdim-per-graph path — find + relax it so the N-rdim
  registers); (2) **T2 self-declines** on `len(inner_red) != 1`; (3) the **M-axis is a non-fact
  tile-accumulation** (your new derived fact handles this). Then the `==1` gate
  (`_triton_reduction_eligible`) and the `reduction_facts[0]` hard-indexing in both `get_seed_config`s
  must move to **role-based selection over a multi-fact list** (the "exactly one ReductionFact"
  invariant is a LIMITATION to redesign, not a requirement). **NB "just relax the gate to ≥1" is
  INSUFFICIENT** — with 0 facts registered, relaxing the gate changes nothing; the populators are the
  real blockers.
- **BUILD IT AS A CLEAN ABSTRACTION:** make the tile-accumulation recognizer a tidy, self-contained
  piece — don't gratuitously hard-code assumptions that only hold for the M-axis case where a little
  generality is free. (A later effort may extend it to other tile-accumulation reductions; you don't
  need to handle those, just don't wall them off.)
- **You ARE free to write throwaway one-off PROBE kernels** while developing the heuristic — to check a
  workload property or de-risk a design (e.g. a minimal `sum`-over-dim-0 in the tile-accumulation idiom
  to build the recognizer on the isolated single-reduction case before the N+M interleaving). These are
  *unit probes*: no baseline, no curriculum entry, no autograd Function, no perf target. This is
  DISTINCT from authoring full **curriculum** kernels (baselines + shapes + perf targets): WS2 targets
  only the existing rms/ln backward (+ softmax_bwd control).

## `grad_w` — the levers + the likely codegen ceiling

`grad_w` collapses many rows → `[N]` (per-M-tile partials + a finalize); its seed levers are the M-tile
size, the partials grid, and `num_warps`/`num_stages` for the partial accumulation — pull the oracle
answer-key freely here (method §2/§3).

**Likely codegen-bound — flagged so you don't flail or wrongly exempt.** It's the same "reduce over the
batch axis → tiny output" shape that made `long_sum` codegen-bound vs tc's split-reduction (which Helion
can't express). The oracle is the arbiter; flagging the mechanism up front just saves you discovering it.
If a **fresh converged** oracle can't beat tc, the floor **retargets to `0.75 × oracle`** (**not exempt**
— stays live until `seed ≥ 0.75 × oracle`; route the ceiling claim through **Gate B**, a hypothesis to
test, never self-certified).

## Goal — the method's disaster-avoidance bar, across the backward norms × 3 dtypes

The bars are method §3's, applied to the backward kernels (greenfield — no inherited champion, all 3
dtypes tuned together):
- **Per-shape floor:** no *realistic* shape on `rms_norm_bwd` / `layer_norm_bwd` below `G = 0.75` vs tc
  (or `0.75 × oracle` where grad_w is codegen-bound, Step 2b).
- **Per-(kernel, dtype) geomean ≥ 0.85** — each (bwd kernel, dtype) pair clears it on its own.
- Beating tc / oracle-parity is **overtime** (§Step 4) — never-stop keeps climbing past the bar, but
  it's not the bankable bar.
- **`softmax_bwd` is the 1-fact control** (already gets a T2 seed — it should keep clearing the bar).
- **Do-not-regress the 9 forward kernels** at any dtype — your populator/gate/`device_ir` changes could
  perturb them, so guard with **Gate R** (the disaster-avoidance MUST): it sweeps the full matrix with
  the config-recorder and re-benches any changed forward cell to floor (a 0-diff = nothing changed; a
  changed-but-above-floor forward cell is Gate R's to adjudicate, not an auto-fail).

## Where everything lives

- **Kernels — the TARGETS already exist; you do NOT author *curriculum* kernels here** (throwaway
  property-probe kernels ARE fine — see the machinery section): `examples/rms_norm.py` (`rms_norm_bwd` +
  `RMSNormFunction`), `examples/layer_norm.py` (`layer_norm_bwd`), `examples/softmax.py` (`softmax_bwd`
  — the control). All have autograd `Function`s + `*_tritonbench` wrappers.
- **Baseline / yardstick (tc):** tritonbench has `-bwd` + torch.compile baselines for `rms_norm-bwd`,
  `layer_norm-bwd`, `softmax-bwd` (`benchmarks/run.py`, via the `-bwd` name suffix → `--bwd` flag).
  **CAVEAT:** rms/ln-bwd strip `--cudagraph` (OOM, issue 711). **Use tritonbench's own measure mode
  (default `triton_do_bench`); deviate to a device-time mode ONLY if you MEASURE a host-overhead
  artifact** — backward is heavier / higher arithmetic-intensity so that artifact likely doesn't apply.
  Don't prescribe a bespoke metric blind.
- **Harness:** the lab harness is **FORWARD-ONLY** (`requires_grad=False`). You must **extend it to a
  backward path** — `benchmarks/run.py` already supports the `-bwd` suffix, so extend
  `_lab/bench/seed_vs_tc.py` to drive it rather than building backward timing from scratch.
- **TRITONBENCH RELIABILITY GUARDRAILS** (method §4's footguns, as they bite the *backward* path — a
  prior climb logged tritonbench producing FALSE results; "reliable" means "used correctly"):
  1. **Gate ACCURACY first** — tritonbench reports perf on accuracy-FAILING kernels and emits false
     `acc=0`. Verify backward accuracy vs the autograd reference at the SAME dtype (upcast to fp32 for
     `allclose`; near-zero-output "fails" are tolerance traps → max_abs, not max_rel).
  2. **Time BOTH arms identically.** Backward intrinsically builds a grad graph, so the fwd "no-grad"
     rule is replaced by **"same grad + measurement setup on both arms"** — a grad-mode / autograd-
     wrapper mismatch fabricates artifacts. Force + assert the dtype (some operators default to fp16).
  3. **`do_bench` default mis-times low-M / bandwidth-bound**; backward is heavier so it likely doesn't
     bite — but if a low-M bwd shape looks like a loss, re-check on a device-time mode
     (`gpu_events`/profiler; `--cudagraph` is stripped for rms/ln-bwd).
  4. **The rest of method §4's footguns apply unchanged** — one isolated process per kernel (batching
     corrupts the tc baseline), baseline = `tc_default` (not max-autotune / reduce-overhead), the timer
     isn't itself biased (re-bench the FULL verbatim config; `<25µs` = noise floor), and never cudagraph
     a `reduce-overhead` baseline.
- **Heuristic / facts / populators:** `triton.py` (the eligibility gate + the two
  `is_eligible`/`get_seed_config`), `config_spec.py` (`ReductionFact`, `MemoryOpFact`, `AccumulatorFact`
  + `accumulator_facts` — the pr-stack base carries the walker/derived layer),
  `device_ir.py` (`register_rollable_reductions`, `register_user_tiled_reductions`, **the T1/T2
  mutual-exclusion / one-rolled-rdim-per-graph path you must relax for blocker 2**, the walker-fact
  collector + the reduction/accumulator fact builders).
  **Anchor by symbol name — line numbers drift.**
- **Behavior recorder (compile-time, no GPU):** the config-recorder — the **tool Gate R uses** to prove
  your `device_ir`/populator changes left the **forward** kernels' emitted seeds byte-identical (a
  0-diff on the 9 forward kernels = you didn't perturb them). `local-setup-devserver.md` names the
  script.

## Correctness traps (method §4)

- **fp32-accumulate** — the grad_w accumulator must be fp32.
- **Your populator/gate changes can mis-seed or break the 9 forward kernels** — discharge with Step-0 +
  **Gate R** (its config-recorder sweep flags + re-benches any changed forward cell to floor).
- **Backward correctness:** gate vs the **autograd reference** (torch's rms_norm/layer_norm backward),
  same dtype, upcast both to fp32 before `allclose`.

## Deliverable (DoD — a milestone to BANK, then keep climbing, §6.0)

1. The grad_w tile-accumulation reduction **recognized + seeded** (as a derived fact reading walker
   facts), composed with grad_x into **one Config**; `rms_norm_bwd` + `layer_norm_bwd` clear the bar
   (per-shape floor + per-(kernel, dtype) geomean ≥ 0.85; floor retargeted to `0.75 × oracle` where
   grad_w is codegen-bound), all 3 dtypes; `softmax_bwd` control holds. Each bar-clear (and any
   beating-tc overtime) is banked only after **adversarial-verify (Gate A)** — N independent skeptics +
   the independent own-script reproduction, reading the focal cell from Gate R's step-4b sweep.
2. The new fact + builder passes the **Fact-gate (Gate D)** (walker/derived placement + the divergence
   test), and the new levers (incl. the `num_warps` reconciliation) pass the **generality gate
   (Gate H)**; the populator/gate rework does **not regress** the 9 forward kernels (**Gate R** +
   its config-recorder sweep re-benches any changed forward cell to floor — plus **Gate E** at freeze).
3. The tile-accumulation recognizer left as a **clean, reusable abstraction** (a later effort may
   extend it — don't wall that off).
4. Report: per-kernel bwd `G` vs tc/oracle at 3 dtypes; the grad_x-vs-grad_w perf split; the fact design
   you chose + why (and how it obeys the walker/derived doctrine); the `num_warps` reconciliation
   policy; what grad_w is/isn't codegen-bound on.

**Scope boundary:** validated lab result in the worktree (changes + proofs + report) — bank it and keep
climbing (§6.0). Do **not** `git push` / touch the PR. Never stop on your own; never ask the human
mid-run — resolve, log, and continue (§6.0).
