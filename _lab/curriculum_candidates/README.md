# M-reduction curriculum candidates — handoff

> **STATUS (completed @ commit 56335324, branch ws2-mreduction):** the m-reduction seed has been
> EXTENDED to these 4 kernels + fully gate-validated. Read `WS2_CURRICULUM_REPORT.md` (deliverable) +
> `CURRICULUM_NOTEBOOK.md` (worker log) — they SUPERSEDE the draft notes below.
> **What landed:** (a) lever fix — `feature_extent` = full resident footprint C·S (was C); (b) role-based
> re-route bias_grad/dyt T2→m_reduction; (c) non-pow2 gw-accumulator match. Gates R/D/H/A/E/F all PASS;
> forward 739/0 + rms/ln/softmax bwd byte-identical.
> **Bar status:** dyt CLEARS all dtypes (geo 1.37/1.00/1.00); group/instance CLEAR every kernel-tractable
> (C·S≤32768) shape (geo 1.7–2.8, 0 below floor, train+TEST). Below bar = structural, documented:
> HRQ #1 wide-S group/instance kernel-bound (no S-tiling); HRQ #2 bias_grad codegen-bound pure sum
> (Product B); HRQ #3 non-pow2-C crash (curriculum-kernel bug). The note below ("the LEVER bug") is the
> ORIGINAL handoff and is now RESOLVED (it was item (a)).

---

Four backward kernels drafted to extend the m-reduction seed past the rms/ln baseline.
Definition in scope: **M-reduction = a reduction over the GRID axis** (the per-CTA
tile-accumulation half), authored grad_w-style (grid over M, per-CTA `[feature]` partial,
separate finalize `.sum(0)`) so they stay clear of the split-K *combine* track.

File: `mreduction_styles.py` (kernels + torch references + a correctness gate).

## Structural classes (relative to rms/ln = "class C: feature reduction over the SAME axis as the param")

| Kernel | Class | What it varies | grad_x feature reduction |
|---|---|---|---|
| `bias_grad_bwd` | **A** | pure collapse, no feature reduction | none |
| `dyt_bwd` (Dynamic Tanh) | **A** | pure collapse + a data term; grad_x is elementwise | none |
| `group_norm_bwd` | **B** | decoupled axis: grad_w over `C`, grad_x normalizes over `group_channels` | over group (≠ C) |
| `instance_norm_bwd` | **B** | decoupled axis: grad_w over `C`, grad_x normalizes over spatial `S` | over S (≠ C) |

## Curriculum shapes — `mreduction_shapes.py` (validates: PASS)

`mreduction_shapes.py` is the shape curriculum (modeled on `_lab/prompts/shapes_v3_draft.py`):
`SHAPES[kernel] = {train, val, test, robustness}`, 14 / 6–7 / 6–7 / 3–4 each (122 total).
Banding axis = the per-row feature footprint `F` (the byte-cap / warp-ramp key): `F=N` for
bias_grad/dyt, `F=C·S` for group/instance norm. `validate()` enforces train-covers-val/test
bands, F-envelope, pairwise-disjoint splits, and the ~20µs noise floor — run
`python mreduction_shapes.py` (exits 0 = PASS). Shape tuples match the kernel signatures in
`mreduction_styles.py` (`(M,N)`, `(N,C,S,G)`, `(B,C,S)`). Harnesses should `import SHAPES` from here.

## Correctness — VERIFIED (GPU 2, EFFORT=none)

All four pass vs a torch autograd reference, **fp32 and bf16**, grad_x + grad_weight + grad_bias:
- fp32 rel err ~1e-7 (machine precision)
- bf16 rel err ~2e-3–4e-3 (bf16 precision; tol 2e-2)

Re-run: `cd /tmp && PYTHONPATH=<worktree> CUDA_VISIBLE_DEVICES=<idx> HELION_AUTOTUNE_EFFORT=none python <worktree>/_lab/curriculum_candidates/mreduction_styles.py`

## Current seed state — ALL FOUR ALREADY FIRE (corrected via factdump)

An earlier note here said "all four decline → generic [32,32]". That was WRONG — the `[32,32]`
seen at runtime is an `EFFORT=none` artifact (reduction/m-reduction heuristics don't set
`promote_seed_to_default`, so at effort=none the kernel runs the generic default while the real
seed lives in `compiler_seed_configs`). The actual seed layer fires on all four:

| Kernel | reduction_facts | m_reduction_facts | seed block_sizes |
|---|---|---|---|
| bias_grad | 1 | 0 | `[1, 8192]`  (T1/T2) |
| dyt | 1 | 0 | `[1, 8192]`  (T1/T2) |
| group_norm (3D) | 0 | **1** | `[1, 128]`  (**m-reduction fires**) |
| instance_norm (3D) | 0 | **1** | `[1, 512]`  (**m-reduction fires**) |

So the recognizer GENERALIZES to canonical group/instance norm — good news. The class-B
"decoupled axis" decline predicted earlier does NOT occur for these authorings: the gw/gb `[C]`
accumulator matches the materialized `reduction=True` C block (`dim_block_ids==(C,)`), so the
fingerprint fires.

## The real gap is the LEVER, not the recognizer (perf — the broadening target)

`MReductionFact.feature_extent` is read from the materialized reduction block = **C only**, but
the 3D norms' resident tile is `[inner, C, S]`. So the byte-cap undercounts the row footprint by
a factor of **S**:

- group_norm (C=64, S=64): `feature_extent=64`, byte-cap sizes `inner = np2(32768/(64·4)) = 128`,
  but to fit 32 KiB it must be `np2(32768/(64·64·4)) = 2`. → `inner` is **64× too large** → the
  `[128, C, S]` resident tile spills *worse than the generic default*.
- instance_norm (C=16, S=128): `feature_extent=16` → `inner=512` vs correct ~1; **128× too large**.

Real vision group_norm has S = H·W = thousands, so this mis-sizing is severe. **The broadening
target is the lever:** `feature_extent` (or the byte-cap input) must account for the full resident
feature footprint (here `C·S`), not just the single materialized reduction block. Recognition is
already general; lever transfer to multi-dim (spatial) resident tiles is the fix.

(Class A — bias_grad/dyt — fire T1/T2, not the m-reduction seed; whether their T1/T2 seed is
*good* for a tall→tiny collapse is a separate seed-quality question, not a decline.)

NB the 3D group/instance kernels spill hard at large shapes under the generic default (and under
the current mis-sized m-reduction seed), so keep test shapes small until the lever is fixed.
