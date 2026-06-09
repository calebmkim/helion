# TASK (WS2): extend the reduction seed heuristic to M-axis (parameter-gradient) reductions — backward norms

**Read in order (same dir): `hillclimb-method.md` → `local-setup.md` → `gate-prompts.md` → this
file.** These four are authoritative; where any other doc/notebook/comment disagrees, they win — and
so does the live code.

**Resume context — fresh climb, NEW machinery.** No prior log for this workstream. Create a fresh
notebook `_lab/ws2_notebook.md` + ledger key `ws2`; don't overwrite run3/dtype/ws1 lineage. Run on
your **own branch/worktree**. **The existing forward heuristic (9 kernels, 3 dtypes) is your DO-NOT-REGRESS baseline;
your populator/gate changes can touch it, so guard it carefully.**

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
`layer_norm_bwd` register **ZERO `ReductionFact`s** (not ≤1 — this corrects the earlier guess) and
fall back to the **generic `_base_default_config()`** (`block_sizes=[32,32], num_warps=4,
num_stages=1, pid_type=flat`) — **both axes UNSEEDED.** Three things conspire, all because the kernel
has **two distinct inner reductions in one body** (N-axis grad_x + M-axis grad_w):
1. **grad_w (M-axis)** reduces over an inner `hl.tile` block (`reduction=False`) accumulated into a
   carried buffer — a non-fact tile-accumulation (no `ReductionFact`). *(Confirmed as hypothesized.)*
2. **grad_x (N-axis)** IS a real `reduction=True` rdim (size=N) — but **T1 declines to register it**
   because a second reduction is interleaved (a one-rolled-rdim-per-graph constraint; pin the exact
   decline path in Step 0).
3. **T2** then declines too: its `inner_red` set has TWO non-grid `ReductionLowering` block_ids
   (`len != 1`, `device_ir.py:1081`).

→ 0 facts → `_triton_reduction_eligible` (needs exactly 1) declines → generic default. **`softmax_bwd`
is the clean 1-fact control** (only ONE reduction → T2 seed `block_sizes=[1,2048,2048], w8`, matching
forward `softmax_two_pass`). Upside: rms/ln bwd sit on a fully generic seed today, so the headroom is large.

Your job: build the machinery so the **grad_w (M-axis) tile-accumulation reduction** is recognized as
a seedable reduction and gets a good seed, **composed with grad_x's seed into ONE Config**, beating
tc on the backward norm kernels.

## Step 0 — baseline (fact-structure HALF already done; perf-split half remains)

The fact-structure half is **already established** (see KEY MECHANISM): rms/ln bwd = 0 facts → generic
default; softmax_bwd = 1 fact → T2. You do NOT need to rediscover it — but DO **re-confirm cheaply**
(`EFFORT=none`, dump `config_spec.reduction_facts`) after each `device_ir` change, since flipping that
count 0 → {the facts you intend} is the signal your new registration works. The REMAINING Step-0 work
is the **perf SPLIT**: how much of each bwd kernel's runtime is grad_x vs grad_w — it bounds the prize
and tells you which axis to attack first. That needs the backward harness (not built yet — see Where
everything lives). Delegate both the re-confirm and the split measurement to an ephemeral
code/perf-investigator (method §6.2); keep the raw dumps out of your own context.

## The new machinery — from-scratch infra IS the job (method §2)

Build a recognizer for the **"tile-accumulation reduction"** (a `.sum` over an inner device-loop
tile accumulated into a carried buffer — the grad_w pattern). The design is **open-ended and yours**,
bounded only by the gates + the hard constraints below:

- **Fact GRANULARITY is open — here is a suggestion AND its wrinkle, you decide.** A *per-reduction-
  loop* fact (a "`ReductionLoopFact`" — one per reduction loop, eagerly assuming rollable dims will
  roll) would unify T1/T2 and make N+M just a list. **BUT** some existing `ReductionFact` fields
  (`row_reread`, `reread_buffer_name`) are **kernel-wide, not cleanly per-loop** — so a naive per-loop
  split is wrong. Don't adopt it reflexively; design what's cleanest and let **Gate D
  (fact-integrity)** judge it.
- **ALL-NEW-DATASTRUCTS:** **prefer a NEW dataclass** (e.g. an `MReductionFact` / tile-accumulation
  fact) **over adding fields to `ReductionFact`** — modular, and it avoids merge conflicts. Populate it in `device_ir` with a **new builder** (e.g.
  `register_m_reduction_facts`) run **after** T1/T2.
- **HARD CONSTRAINT (survives any granularity choice): all reductions' seeds must COMPOSE INTO ONE
  `Config`.** A kernel emits a single Config — shared scalars (`num_warps`, `num_stages`, `pid_type`)
  + positional `block_sizes`/`reduction_loops` lists. So grad_x's preferred `num_warps` and grad_w's
  must be **reconciled into one value** — a real design decision (max? N-dominant? a new fact-keyed
  rule?). Pick it, justify it, **measure it**.
- **You must register BOTH reductions, then seed both — THREE blockers to rework (confirmed in Step
  0).** Today the kernel gets 0 facts because: (1) **T1 declines the real N-axis rdim** when a 2nd
  reduction is interleaved (the one-rolled-rdim-per-graph path — find + relax it so the N-rdim
  registers); (2) **T2 self-declines** on `len(inner_red)!=1` (`device_ir.py:1081`); (3) the **M-axis
  is a non-fact tile-accumulation** (your new builder handles this). Then the `==1` gate
  (`_triton_reduction_eligible`) and the `reduction_facts[0]` hard-indexing in both `get_seed_config`s
  must move to **role-based selection over a multi-fact list** (the "exactly one ReductionFact"
  invariant at `config_spec.py:90` is a LIMITATION to redesign, not a requirement). **NB "just relax
  the gate to ≥1" is INSUFFICIENT** — with 0 facts registered, relaxing the gate changes nothing; the
  populators are the real blockers.
- **BUILD IT AS A CLEAN ABSTRACTION:** make the tile-accumulation recognizer a tidy, self-contained
  piece — don't gratuitously hard-code assumptions that only hold for the M-axis case where a little
  generality is free. (A later effort may extend it to other tile-accumulation reductions; you don't
  need to handle those, just don't wall them off.)
- **You ARE free to write throwaway one-off PROBE kernels** while developing the heuristic — to check
  a specific workload property or de-risk a design (e.g. a minimal `sum`-over-dim-0 in the
  tile-accumulation idiom to build the recognizer on the isolated single-reduction case before the N+M
  interleaving; or a small kernel to watch how a fact populates). These are *unit probes*: no baseline,
  no curriculum entry, no autograd Function, no perf target — write them liberally (you do this
  naturally; it's stated so the "don't author curriculum kernels" rule below isn't misread as "write
  no kernels at all"). This is DISTINCT from authoring full **curriculum** kernels (baselines + shapes
  + perf targets): WS2 targets only the existing rms/ln backward (+ softmax_bwd control).

## The perf lever (the tall→tiny reduction)

grad_w collapses many rows → `[N]`. The example expresses it as per-M-tile partials + a finalize. The
seed levers: the M-tile size, the partials grid, `num_warps`/`num_stages` for the partial
accumulation. Use the **oracle answer-key** (method §2/§3) to find which config dimensions actually
move grad_w. **NOTE:** this is the same "reduce over the batch axis → tiny output" shape that made
`long_sum` codegen-bound vs tc's split-reduction — so grad_w **may be partly codegen-bound**. Per the
oracle-retarget rule (method §2a, §3): **if the Helion oracle can't beat tc on grad_w, RETARGET the
seed to the oracle (`seed ≤ oracle×1.05`) — do NOT exempt the shape.** A shape is exempt only when
`seed ≈ oracle`.

## Goal

**This is the AGGRESSIVE, EXHAUSTIVE climb — the full method (§3/§4)**.
Per shape, across **all 3 dtypes tuned together**
(greenfield — no inherited fp32 champion): beat tc on `rms_norm_bwd` + `layer_norm_bwd` per-shape
where reachable, oracle-retarget where grad_w is codegen-bound, then push toward **oracle parity**
(§4) — never bounded by "good enough," never stopping short of the per-shape bar. Use **`softmax_bwd`
as the 1-fact control** (it already gets a T2 seed). **Do-not-regress the 9 forward kernels** at any
dtype — your populator/gate/`device_ir`
changes could perturb them, so guard with the no-regression backstop + the behavior-recorder.

## Where everything lives

- **Kernels — the TARGETS already exist; you do NOT author *curriculum* kernels here** (throwaway
  property-probe kernels ARE fine — see the machinery section; new curriculum kernels are the separate
  validate step): `examples/rms_norm.py` (`rms_norm_bwd` + `RMSNormFunction`), `examples/layer_norm.py`
  (`layer_norm_bwd`), `examples/softmax.py` (`softmax_bwd` — the control). All have autograd
  `Function`s + `*_tritonbench` wrappers.
- **Baseline to beat:** tritonbench has `-bwd` + torch.compile baselines for `rms_norm-bwd`,
  `layer_norm-bwd`, `softmax-bwd` (`benchmarks/run.py`, via the `-bwd` name suffix → `--bwd` flag).
  **CAVEAT:** rms/ln-bwd strip `--cudagraph` (OOM, issue 711). **Use tritonbench's own measure mode
  (default `triton_do_bench`); deviate to a device-time mode ONLY if you MEASURE a host-overhead
  artifact** — the fwd climb *earned* its cudagraph deviation that way, but backward is heavier /
  higher arithmetic-intensity so that artifact likely doesn't apply. Don't prescribe a bespoke metric
  blind.
- **Harness:** the lab harness is **FORWARD-ONLY** (`bare_fwd_dtype.py`: `requires_grad=False`). You
  must **extend it to a backward path** — `benchmarks/run.py` already supports the `-bwd` suffix, so
  extend `_lab/bench/seed_vs_tc.py` to drive it rather than building backward timing from scratch.
- **TRITONBENCH RELIABILITY GUARDRAILS** — the prior climb logged tritonbench producing FALSE results;
  "reliable" means "used correctly." These apply to the backward path too:
  1. **Gate ACCURACY first — tritonbench reports perf on accuracy-FAILING kernels and emits false
     `acc=0`.** Verify backward accuracy vs the autograd reference at the SAME dtype (upcast to fp32 for
     `allclose`); near-zero-output "fails" are tolerance traps (max_abs, not max_rel).
  2. **One isolated process per kernel; one timing run per GPU** — batching compiles in a long-lived
     process corrupts the tc baseline (this fabricated a bogus 2.18× once).
  3. **Time BOTH arms identically.** Backward intrinsically builds a grad graph (so the fwd
     "no-grad" rule is replaced by "same grad + measurement setup on both arms") — a grad-mode /
     autograd-wrapper mismatch fabricated fwd artifacts (rms_norm 0.79, welford 3.19). Force + assert
     the dtype (some operators default to fp16).
  4. **tritonbench's baseline is `tc_default`** (not max-autotune, not reduce-overhead); seed-vs-
     unseeded-Helion-default is a different, inflated axis.
  5. **`do_bench` default mis-times low-M / bandwidth-bound** (host-enqueue artifact). Backward is
     heavier so it likely doesn't bite — but if a low-M bwd shape looks like a loss, re-check on a
     device-time mode (`gpu_events`/profiler; `--cudagraph` is stripped for rms/ln-bwd). **Never
     cudagraph a `reduce-overhead` baseline** (double-wrap garbles it).
  6. **The timer is NOT itself biased** — anomalies usually trace to analysis (a field-diff re-benching
     a fabricated config → bogus 33× artifact); re-bench the FULL verbatim config; <25µs = noise floor.
- **Heuristic / facts / populators:** `triton.py` (gate `_triton_reduction_eligible` ~299; the two
  `is_eligible`/`get_seed_config`), `config_spec.py` (`ReductionFact` ~88), `device_ir.py`
  (`register_rollable_reductions` ~889, `register_user_tiled_reductions` ~1050, the mutual-exclusion
  ~3041, fact builders ~999/~1130).
- **Behavior recorder (compile-time, no GPU):** `_lab/harness/run3_task1_verify_after_edit.py` —
  use it to prove your `device_ir`/populator changes left the **forward** kernels' emitted seeds
  byte-identical (a 0-diff on the 9 forward kernels = you didn't perturb them). Post-edit check, not a
  during-climb gate.

## Correctness traps (method §4)

- **fp32-accumulate** (the grad_w accumulator must be fp32).
- **Your populator/gate changes can mis-seed or break the 9 forward kernels** — discharge with Step-0
  + the no-regression backstop + the behavior-recorder 0-diff on forward.
- **Backward correctness:** gate vs the **autograd reference** (torch's rms_norm/layer_norm backward),
  same dtype, upcast both to fp32 before `allclose`.

## Deliverable (DoD — a milestone to BANK, then keep climbing, §6.0)

1. The grad_w tile-accumulation reduction **recognized + seeded**, composed with grad_x into **one
   Config**; `rms_norm_bwd` + `layer_norm_bwd` beat tc per-shape where reachable (oracle-retarget
   where not), all 3 dtypes; `softmax_bwd` control holds.
2. The new fact + builder passes **Gate D** (fact-integrity divergence test); the populator/gate
   rework does **not regress** the 9 forward kernels (no-regression backstop + behavior-recorder
   0-diff on forward + **Gate E** at freeze).
3. The tile-accumulation recognizer left as a **clean, reusable abstraction** (a later effort may extend it — don't wall that off).
4. Report: per-kernel bwd G vs tc/oracle at 3 dtypes; the grad_x-vs-grad_w perf split; the fact design
   you chose + why; the `num_warps` reconciliation policy; what grad_w is/isn't codegen-bound on.

**Scope boundary:** validated lab result in the worktree (changes + proofs + report) — bank it and
keep climbing (§6.0). Do **not** `git push` / touch the PR.
