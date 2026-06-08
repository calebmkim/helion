# TASK: extend the reduction seed heuristic to bf16/fp16

**Read in order (same dir): `hillclimb-method.md` (the durable method, gates, footguns,
orchestration; its START-HERE block tells you how to resume-vs-start-fresh) → `local-setup.md`
(this machine: paths, dedicated GPU, frequent-commit policy, log + script locations) →
`gate-prompts.md` (verbatim adversarial-gate frames) → this file (the *what* for this run).**
These four files are authoritative; where any other doc/notebook/comment you find disagrees with
them, they win.

This file is deliberately big-picture. The method tells you *how* to climb; this tells you *what*
you're climbing and points you at the code, kernels, shapes, and harness. The actual fixes,
fact/heuristic changes, and lever re-tunings are **yours to discover and make** (be aggressive —
method §2). Don't expect a checklist here; expect a map.

**Resume context — read this so you don't chase a dead worklist.** A prior fp32 climb (`run3`)
produced the current heuristic and was **stopped by the human** mid-edit; its logs
(`HUB_BATON.md`, `run3_notebook.md`, ledger key `run3`) end on an un-built fp32 candidate ("EDIT#7").
This is a **NEW dtype climb**, not a resume of that. So: do **NOT** execute the EDIT#7 worklist;
**treat the committed `run3` champion (current `triton.py` @ branch HEAD) as your inherited fp32
baseline that you must not regress**; start a **fresh dtype notebook + ledger key** (don't overwrite
the run3 lineage); and mine the run3 logs only as *reference* (what was tried, what the gates
falsified, the per-shape fp32 status) — the method's START-HERE "resume vs scratch" framing maps to
"fresh dtype climb on an inherited baseline" here.

---

## Background: what bf16/fp16 are, and why this is the next frontier

The reduction seed heuristic was built and tuned entirely at **fp32**. The current fp32 results are
**strong but NOT finished** — the seed does not yet reach parity with torch.compile on every shape,
let alone the autotuner oracle. So fp32 is *not* "done, don't touch": it's fair game to keep
improving. The reason this run leads with bf16/fp16 is that those dtypes were **never tuned at all**,
so the easy wins are likely larger there.

**The dtypes:**
- **bf16** (bfloat16, 2 bytes): same 8-bit exponent as fp32 (so the same dynamic range, no overflow
  surprises) but only 7 mantissa bits — coarse rounding. The dtype modern training runs in.
- **fp16** (float16, 2 bytes): more mantissa (10 bits, finer rounding) but only a 5-bit exponent →
  **max ≈ 65504**, so it can *overflow* where bf16/fp32 wouldn't (think `exp()` in softmax/CE, or
  sums-of-squares in the norms).
- **fp8 is OUT OF SCOPE** — not a code limit but a workload one: these are norm/loss/statistics
  kernels whose inputs live in bf16/fp32 in real models (fp8 is an output-cast feeding the next
  GEMM, not the reduction dtype), and fp8 always needs amax *scaling* (scale inputs + dequant) which
  makes it a different kernel. Leave it.

**Why bf16 and fp16 may be close to each other:** both are itemsize=2, the heuristic reads only
`itemsize` + element counts (never the dtype *kind*), and on H100 both upcast to fp32 at identical
hardware cost. So a reasonable hypothesis — *for you to test, not assume* — is that they want
near-identical configs and bf16 perf work transfers to fp16, leaving fp16 mostly a **correctness**
question (its overflow-prone exponent). Verify that hypothesis rather than trusting it.

---

## Goal

Per the method's goal hierarchy (§3): **beat torch.compile first** (the portable bar), per active
dtype; use the oracle only as a reachability-test + answer-key; push toward oracle parity once tc is
beaten. Apply that across dtypes:

- Re-anchor and (where you can) **improve fp32** — it's not at parity yet.
- Bring **bf16** up the same ladder (likely the biggest, easiest gains — untuned).
- Handle **fp16** (probably transfers from bf16 on perf; needs its own correctness check).
- **Never regress an already-banked dtype/shape** — the no-regression backstop spans the whole
  active matrix (method §3).

There's no finish line — bank the deliverable as a milestone and keep climbing (method §6.0).

---

## The code

Worktree `/home/dev/local/helion-pr-with-lab` (confirm via `local-setup.md`).

- **Heuristic:** `helion/_compiler/autotuner_heuristics/triton.py` — a shared base
  `_TritonReductionSeedBase` plus T1 `TritonReductionTileHeuristic` (rollable rdim) and T2
  `TritonReductionUserTileHeuristic` (user-tiled; covers softmax_two_pass, the Band-B kl_div/jsd,
  and Band-C welford). H100/sm90-gated.
- **Facts:** `helion/autotuner/config_spec.py`, `ReductionFact` (~L88), populated in `device_ir.py`'s
  fact builders. **`itemsize` already exists** and is read from the reduction's *input* tensor.
- All of the above is **yours to change aggressively** (method §2): add fields/methods/facts/levers,
  restructure classes — whatever the perf demands, subject only to the gates.

**Prime directive:** drive dtype generality through faithful *workload properties* (bytes,
itemsize, residency, accumulator structure), **never** a `dtype`-kind branch or an `itemsize==2`
special-case (that's a dtype branch in disguise — the fact-integrity gate hunts exactly this). If
you reach for "if dtype is bf16…", you've mis-modeled; find the property.

---

## The one fact to internalize before tuning bf16

**Helion accumulates reductions in fp32 even when the input is bf16/fp16** (it promotes the
accumulator — `get_computation_dtype` / the "promote fp16 to fp32" path in `reduction_strategy.py`;
confirm empirically). So at half precision the *input* width and the *resident-accumulator* width no
longer agree.

Carry that fact into the heuristic and **audit it yourself**: the heuristic's byte-budget caps, its
residency reasoning, and its fp32-tuned perf levers (warps, registers, chunk sizes, crossovers) were
all written and tuned when input==accumulator==4B. Which of them are now mis-sized or mis-tuned at
2B, which are fine, and what the right correction is — that's the work; the climb (oracle answer-key
→ field-diff → A/B → gate) is how you find out. Don't blanket-edit; reason about what each lever
actually governs. (Tip: a correctness-only fix that's a true no-op at fp32 will show 0 config-diffs
at fp32 in the behavior recorder — a cheap way to prove you only touched low precision.)

---

## Where everything lives (this run is standalone — here are the pointers)

- **The 9 reduction kernels** are runnable examples in `examples/`: `sum.py`, `long_sum.py`,
  `rms_norm.py`, `layer_norm.py`, `softmax.py` (row + `softmax_two_pass`), `cross_entropy.py`,
  `kl_div.py`, `jsd.py`, `welford.py`. Read the kernel source when reasoning about what's resident
  (e.g. which accumulators a kernel carries, whether it re-reads a row, whether an accumulator is
  explicitly typed) — `welford.py` and `long_sum.py` in particular have accumulator choices worth
  understanding before you trust a cap.
- **The shape curriculum** is `_lab/prompts/shapes_v3_draft.py` (`SHAPES` = 9 kernels ×
  train/val/test + robustness; `TRANSFER` = off-curriculum kernels; `validate()` enforces the
  band/noise-floor invariants). It is **fp32-baked today** — at minimum the noise-floor estimate is
  fp32-byte; growing a real dtype axis (so bf16 shapes stay above the noise floor, references run
  at-dtype, tolerances loosen appropriately for half precision) is part of the work. The shape
  *lists* are largely dtype-agnostic; add the dtype-specific stressors the climb reveals you need.
- **Benchmark harness** (tritonbench-based, the trustworthy path): `_lab/bench/seed_vs_tc.py`,
  `run_seeded.py`, `sweep.py`; the bare-forward cross-check `bare_fwd_seed_vs_tc.py`. These build
  inputs and drive the 3 arms — they currently hardcode fp32; threading a dtype through is part of
  the setup. (Heed the method §4 footguns: forward-only, dynamo-reset, single-process,
  loosen-tolerance-with-logged-justification at half precision.)
- **Behavior recorder** (POST-CLIMB only, method §5): `_lab/harness/run3_task1_verify_after_edit.py`
  — records the emitted config per curriculum shape. Use it at the end to prove a cosmetic/refactor
  edit is 0-diff at fp32 (and to test the bf16≡fp16-config hypothesis). Not a during-climb gate.

You will need to extend the harness/curriculum/recorder with a dtype axis early — that's Step-0/setup
work, not the climb.

---

## Correctness traps that can mislead the oracle/gates (find where each bites)

- **Half-precision tolerance:** an fp32-tight `allclose` *falsely fails* a correct bf16/fp16 kernel
  (its round-trip floor is far higher, and compounds across multi-pass kernels). Set the gate from a
  *measured* per-kernel floor and log the justification (method §4); never silently widen it. The
  current fp32 tolerance is hardcoded in the harness — find every site.
- **fp16's narrow exponent (max ≈ 65504) can overflow** where bf16/fp32 don't. Work out which of the
  9 kernels are at risk and whether the fp32 promotion actually saves them — don't assume.
- **An input-dtype (non-promoted) accumulator is possible** in some kernel sources. If a kernel
  genuinely accumulates at the input width, its accuracy hit is a *kernel-authoring* question, not a
  heuristic one — characterize it so a failed correctness check doesn't get blamed on your seed.

---

## Deliverable (definition of done) — a milestone to BANK, then keep climbing

A **valid, freezable champion**, not a stop signal: when met and verified, freeze + bank (commit +
report), then keep climbing (method §6.0). It comprises:

1. **Correctness** for all 9 kernels at bf16 and fp16, at a measured/justified half-precision
   tolerance, with any genuinely-bf16-accumulator kernel characterized.
2. **bf16 perf** up the goal-hierarchy ladder: seed ≥ tc on every reachable shape (codegen-bound
   exemptions flagged via a fresh converged oracle), then oracle parity where reachable — per-shape
   seed/oracle/tc table. **fp16** handled (transfer-or-tune as the evidence dictates).
3. **fp32 held or improved** — no regression on any banked fp32 shape; fp32 config changes that were
   meant to be no-ops proven 0-diff via the recorder.
4. Whatever fact/heuristic/lever changes you made, each justified by measurement and passing the
   gates (faithful property, no dtype-kind/identity branch, reproduced).
5. A short report: per-kernel bf16/fp16 G vs tc and vs unseeded default, the parity table, what
   transferred free vs needed re-tuning, and the open questions you resolved.

**Scope boundary** (not a stop signal): the deliverable lives as a validated lab result in the
worktree (changes applied, proofs + report) — bank it and keep climbing. Do **not** `git push` /
touch the PR; the human decides that.
