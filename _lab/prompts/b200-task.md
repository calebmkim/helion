# TASK: bring the reduction seed heuristic to B200 (sm100) at fp32 + bf16 + fp16

**Read in order (same dir): `hillclimb-method.md` (the durable method, gates, footguns,
orchestration) → `local-setup-b200.md` (THIS machine: B200/sm100, the correct interpreter, the
dormant-seed fact, scripts, env knobs, resume state) → `gate-prompts.md` (verbatim adversarial-gate
frames) → this file (the *what* for this run).**

⚠️ **Use `local-setup-b200.md`, NOT `local-setup.md`** (the latter describes the old H100 box and has
stale facts). `dtype-task.md` is the **H100/sm90 dtype** task, superseded by this file — its dtype
reasoning transfers verbatim, but its "the run3 champion is a working fp32 baseline" framing is
**false on B200** (see below). Where anything disagrees with these four docs or the live code, those
win.

This file is a **map, not a checklist**. It gives you the *what* and the facts that cost real time to
discover; the *how* — the architecture, the fixes, the fact/lever changes, the order you do things —
is **yours to figure out** (method §2, be aggressive). Don't expect step-by-step instructions; the
method file owns the loop and Step 0, and you own the decisions.

> ## ⚠️ EXPECT DISCREPANCIES — this lineage was written for a DIFFERENT machine.
> Everything here (docs, harness, curriculum, lab scripts, logs) was **originally written for an H100**
> and only partly retargeted to this B200 box, so **more stale H100-isms will surface** — a dead path
> or venv, an assert against an old root, an fp32 hardcode, a script assuming the old SM count, a
> drifted tritonbench arg. Expected, not a blocker: work around it and log it (method §6.0/§6.2). **The
> live code and your own fresh measurements are the ground truth; any doc (this one included) may be
> out of date.**

---

## The one fact that reframes this run — READ FIRST

**On B200 the reduction seed is DORMANT.** Both reduction heuristics are gated to `sm90`, and the
hardware match is *exact* (no arch-fallback), so on this `sm100` box neither fires: the T1 kernels
(sum, long_sum, rms_norm, layer_norm, softmax-row, cross_entropy) fall back to the upstream
conservative seed, and the T2 kernels (softmax_two_pass, kl_div, jsd, welford) get **no seed at
all**. Confirm this yourself in Step 0 (dump `compiler_seed_configs` across the curriculum × dtypes)
— it's the cheapest, most important fact you'll establish, and the details of exactly what each path
emits are worth seeing firsthand.

**Consequence:** this is **not** "extend a working B200/fp32 heuristic to halves." On B200 *even fp32
reductions are unseeded today.* Your job is to make the H100-tuned reduction seed **fire on B200**,
validate it there, and tune it for **all three dtypes at once** (fp32, bf16, fp16). The H100-tuned
heuristic is **probably a good starting point** for B200 — but a starting point you'll likely need to
**tweak**, not a finished answer to port verbatim. Its **constants** are the likeliest thing to re-tune
(sized for the old SM count and a 4B accumulator; B200 has 148 SMs and 2B-or-4B accumulators at halves)
— but **don't assume constants are the only change**: a lever, a band boundary, a persistent/looped
crossover, or a new B200-specific rule may also be needed. Treat the H100 heuristic as the starting
hypothesis and let the climb (oracle answer-key → field-diff → A/B → gate) show you what actually has to
change. Verify; don't assume.

---

## Architecture: a dedicated sm100 reduction heuristic (the chosen approach)

**Decision: add B200 as a separate, sm100-gated reduction heuristic** — a sibling to
`TritonB200MatmulHeuristic` (already sm100-gated and registered), not hardware branches bolted into the
H100 heuristic. The deciding reason is the hard constraint below: **a separate class keeps the frozen
H100/sm90 path untouched by construction** — the cleanest possible guarantee. Reuse the shared
machinery (subclass/share `_TritonReductionSeedBase`, the `ReductionFact` reads, the persistent/looped
+ warp-ramp + byte-cap + eviction + band logic) — the *shape* of the heuristic is likely right; what's
B200-specific is the **constants** (148 SMs, 2B-or-4B accumulators). Give the sm100 class its own
constants from the start; don't inherit H100-tuned magic numbers blindly. (If sharing the base would
entangle the sm90 path, prefer duplicating the small amount of logic into the sm100 class over risking
the freeze.)

**The one hard constraint: freeze the H100/sm90 path byte-identical.** You can't measure H100 here, so
the only safe guarantee is that the sm90 path emits exactly what it does today. The separate sm100
class gets you most of the way **by construction** (don't refactor shared sm90 code and hope it's
equivalent); still **prove it** with the behavior recorder at fp32 (0 config-diffs vs pre-change HEAD)
plus the structural argument that no sm90 codepath changed. **Wiring to get right:** the seed collector
runs *every* eligible heuristic, and T1's eligibility is hardware-independent — so registration,
duplicate-seed suppression, and the `run_seeded.py` promotion all apply to the new class. Work the
mechanics out from the live code; just don't let the new sm100 path silently emit a competing or stale
seed alongside the sm90 one.

Gate note (B200-specific): a *hardware* (`sm100`) gate is not a workload fact (so it's exempt from the
Fact-gate's divergence test), but a constant **inside** the sm100 class that fences the curriculum's
shapes is still overfit (Gate H / Gate E). (Dtype generality via `itemsize`/bytes, never a dtype-kind
branch, is standard gate doctrine, §5 — not repeated here.)

---

## Goal — disaster-avoidance + pretty-good across the whole B200 matrix (method §3)

The bars are method §3's, unchanged — per-shape **floor** (no realistic shape below `G = 0.75` vs tc,
or `0.75 × oracle` if codegen-bound) **+** per-(kernel, dtype) **geomean ≥ 0.85**; beating tc / oracle
parity are overtime. Apply across **{fp32, bf16, fp16} × 9 kernels × curriculum** on B200, all active
at once:

- **fp32 on B200** is real new work (the seed doesn't fire today), not a re-anchor. The H100 run3
  answer-keys are a *hint* about which levers matter, not a baseline you inherit.
- **bf16** is untuned everywhere — likely the biggest, easiest gains.
- **fp16** — hypothesis to *test, not assume*: perf transfers from bf16 (both itemsize=2, the
  heuristic reads only `itemsize`+counts), leaving fp16 mostly a correctness question (narrow
  exponent, max ≈ 65504 → overflow risk). Confirm the config-transfer and re-tune only where the
  evidence demands.
- **No regression across the active B200 matrix:** the backstop (method §3) binds every *realistic*
  shape on every active dtype — Gate R (the disaster-avoidance MUST) owns the realistic↔diabolical
  call. The sm90/H100 cells are frozen — out of the trade space entirely, and untouched by
  construction (separate sm100 heuristic).

No finish line — bank the deliverable as a milestone and keep climbing (method §6.0).

---

## The workload is FIXED — do not change it

Identical to the H100 run, non-negotiable (it's what keeps the numbers comparable): **forward only**
(no grad graph either arm; time the bare forward), **same shapes** (the v3 curriculum
`shapes_v3_draft.py` — you may *add* dtype stressors like an fp16-overflow canary, but don't drop or
alter the existing sets; keep the splits disjoint + the TEST firewall), **same 9 kernels**. The only
kernel edits in scope are **minimal, workload-preserving half-precision compile fixes** (below) — not
algorithmic changes.

---

## The one dtype fact to internalize (from `dtype-task.md`, confirmed on B200)

**Helion accumulates reductions in fp32 even when the input is bf16/fp16** (the "promote fp16→fp32"
path in `reduction_strategy.py`). So at half precision the input width and the resident-accumulator
width disagree — and that's already visible in the fact you branch on: **`ReductionFact.itemsize` is
not uniform across kernels at halves** — it reads the *reduction-input* width, which is the upcast
fp32 (4B) for kernels that `.to(float32)` before reducing and the native half (2B) for the rest.
(You'll see exactly which is which in the Step-0 dump.)

So every **byte**-reasoning lever (residency caps, byte budgets) was sized for 4B — which mis-size at
2B (or for 148 SMs) is part of the climb (don't blanket-edit; see Consequence above). Handy check: a
correctness-only fix that's a true fp32 no-op shows **0 config-diffs at fp32** in the recorder — proof
you only touched low precision.

---

## Where everything lives

- **Heuristic:** `helion/_compiler/autotuner_heuristics/triton.py` (base `_TritonReductionSeedBase` +
  T1 `TritonReductionTileHeuristic` + T2 `TritonReductionUserTileHeuristic`; currently sm90-gated).
  Registered in `autotuner_heuristics/__init__.py`. Hardware match via `common.matches_hardware` +
  `_hardware.get_hardware_info` (returns `sm100` here).
- **Facts:** `helion/autotuner/config_spec.py` (`ReductionFact`, `itemsize` already exists),
  populated in `helion/_compiler/device_ir.py`'s fact builders. Adding/extending a fact (and its
  populator) is fair game (method §2).
- **The 9 kernels:** `examples/`. Read the source when reasoning about residency (which accumulators
  a kernel carries, whether it re-reads a row, whether an accumulator is explicitly typed).
- **Harness, oracle, recorder, curriculum, logs, env knobs + the known frictions:** all in
  `local-setup-b200.md` (threading a dtype axis through them is Step-0 setup, below).

---

## Setup is the part that matters most here

The human's top priority: **get the harness running correctly across all three dtypes before you
climb** — this box has more setup friction than the H100 run, and a miscalibrated harness produces
plausible-but-wrong numbers (method §4). The B200-specific Step-0 work: confirm the dormancy + the
itemsize map yourself, thread a dtype axis through the bench/oracle/recorder/curriculum, set
half-precision tolerances from a *measured* per-kernel floor (logged), and start a **fresh notebook +
ledger key** (e.g. `b200`) — don't overwrite the run3 H100 lineage (mine it as reference only).

One known kernel issue to fold into setup: **jsd does not compile at bf16/fp16** (a control-flow
dtype mismatch on its `dX` accumulator, ~`examples/jsd.py:116`). A minimal workload-preserving fix is
in scope, documented as a *kernel-authoring* change (prove it's an fp32 no-op via the recorder). Any
kernel you genuinely can't fix without changing its workload is characterized, flagged, and excluded
from that dtype's deliverable — don't misattribute a kernel-authoring accuracy hit to your seed.

---

## Correctness traps (find where each bites — don't assume)

- **Half-precision tolerance:** an fp32-tight `allclose` falsely fails a correct half kernel.
  Multi-pass kernels (welford, layer_norm, the two-pass softmax/CE/kl/jsd) compound rounding. Measure
  the per-kernel floor, log it.
- **fp16 overflow (max ≈ 65504)** where bf16/fp32 don't — `exp()` in softmax/CE/jsd, sums-of-squares
  in the norms. Work out which kernels are at risk and whether the fp32-accumulator promotion actually
  saves them (the reduction is promoted, but an intermediate computed at input width can still
  overflow).
- **Non-promoted accumulators:** if a kernel source genuinely accumulates at the input width, that's
  a kernel-authoring accuracy question, not a heuristic one — characterize it.
- **"Dormant" ≠ "bad" seed:** before your sm100 path exists, the "seeded" arm is just the
  fallback/default. Measuring *that* vs tc tells you about the fallback, not the seed you're building.
  Make the seed fire first.

---

## Deliverable (a milestone to BANK, then keep climbing)

A valid, freezable champion — when met and verified, freeze + bank (commit + report), then keep
climbing (method §6.0). It comprises:

1. The reduction seed **fires on B200 (sm100)** via a **dedicated sm100 reduction heuristic**, with the
   **H100/sm90 path frozen byte-identical** (untouched by construction + recorder 0-diff at fp32 + the
   structural argument).
2. **Correctness** for all 9 kernels at fp32/bf16/fp16, at a measured/justified tolerance, with any
   unfixable/non-promoted-accumulator kernel characterized (jsd fixed or flagged).
3. **Perf to the bankable bar, per dtype:** every *realistic* shape ≥ its floor (`G ≥ 0.75` vs tc, or
   `seed ≥ 0.75 × oracle` for codegen-bound shapes — retargeted, not exempted, via a *fresh converged*
   oracle + Gate B) **and** every (kernel, dtype) geomean ≥ 0.85 — a per-shape seed/oracle/tc table per
   dtype. Overtime beyond (beating tc, oracle parity) is logged as bonus. fp16 handled
   (transfer-or-tune, with the config-transfer check shown).
4. **No regression across the active B200 matrix**; any accepted trade meets the method §3 bar and is
   logged with per-shape deltas.
5. Every fact/heuristic/lever change **justified by measurement and passing the §5 gates** (faithful
   property + faithful key, reproduced, belongs in the core, no realistic shape below floor, not
   overfit, surprising wins mechanism-explained).
6. A short **report**: per-kernel G vs tc and vs unseeded-default per dtype, the parity table per
   dtype, what transferred free vs needed re-tuning, the architecture decision and why, the
   kernel-authoring fixes, and the open questions you resolved.

**Scope boundary** (not a stop signal): the deliverable lives as a validated lab result in the
worktree — bank it and keep climbing. Do **not** `git push` / touch the PR; the human decides that.
Never stop on your own (method §6.0).
