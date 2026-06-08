# Reduction-heuristic gates & levers — necessity notes

Verified 2026-06-05 on the merged tree (`/tmp/v2-wt`, branch `reduction-seed-heuristic-v2`), matched-lever,
median-of-7, fp32, H100. No code changed — these record which gates/levers are load-bearing vs simplifiable.
Evidence JSONs are in this dir.

---

## 1. MULTILOAD persist-cap `if fact.row_reread:` guard — UNGATABLE (perf-neutral) — ✅ DONE (ungated 2026-06-05, pushed in 5b3d5fdf)

`_TritonReductionSeedBase._persistent_looped`:
```python
if fact.row_reread:
    if row_bytes > MULTILOAD_PERSIST_MAX_BYTES (240KiB): can_persist = False
```
Q: could the cap fire for ALL kernels (drop the `row_reread` guard)?
**A: yes, perf-neutral.** The only `row_reread=False` kernels affected:
- sum/long_sum (T1 single-load): forced-looped TIES persistent **0.985–1.008x** (wash) — `cap_gating_verify.json`.
- kl_div/jsd (T2 Band-B): ungating is **BYTE-IDENTICAL** (0/8 shapes differ) — `cap_ungate_kldiv_jsd.json`.
  Band-B R_BLOCK cap (4096/2048) < LOOPED_CHUNK(16384) and < np2(N), so `min(np2(N),cap)==min(16384,cap)==cap`;
  num_warps 32 both; eviction inert (row_reread=False).
**Caveat:** the `row_reread` FIELD is still required by levers #3 below (reread-eviction + EDIT-PID), so
ungating the cap ≠ deleting the field. Decision deferred — pure code-clarity call, perf-neutral either way.

---

## 2. Band-B `if fact.num_carried_accumulators >= 1:` gate — LOAD-BEARING (do not simplify)

T2-plain branch:
```python
r_block = extent
if fact.num_carried_accumulators >= 1:
    r_block = min(r_block, _np2(bandb_cap))   # bandb_cap = 16384/(itemsize*n_carried): kl_div 4096, jsd 2048
```
Q: collapse to one unconditional rule — (a) `r_block=extent` always, or (b) `min(extent,bandb_cap)` always?
**A: NO — each class regresses under the other's rule** (`bandb_gate_test.json`):
- (a) no-cap on kl_div/jsd → full-N R_BLOCK SPILLS: kl_div up to **3.94x** slower, jsd up to **29x**
  (2 accumulators spill ~3–8x harder than 1, matching footprint = R_BLOCK·itemsize·n_carried).
- (b) always-cap on softmax (carried=0) → R_BLOCK forced ≤4096 → **1.06–1.36x** slower (wants 8192–16384).
So carried≥1 needs the cap, carried=0 needs no-cap. Gate keys on a faithful workload property; necessary
both directions.

---

## 3. Reread load-eviction `_eviction_policies(env,"reread",reread_buffer_slots)` — load-bearing (no downside)

`'last'` on the re-read row's FIRST load (keep L2-resident for the re-read pass), `'first'` elsewhere. Slot is
**provenance-keyed** to the re-read host buffer, not positional (fixed a CE bug: logits first-loads at slot 2).
Fires on: cross_entropy T1 looped wide-V (15), welford Band-C (all 15), softmax T2 looped widest (2).
kl_div/jsd never (row_reread=False).
**Benefit** (matched-lever, default `['']` vs reread; `reread_evict_verify.json`):
- cross_entropy: ~1.1–**1.43x** (geomean ~1.29x) — big win, hundreds of µs on wide-V.
- softmax: ~1.08–**1.30x** (geomean ~1.18x) — strong on widest looped.
- welford: ~1.0–**1.24x** (geomean ~1.10x) — WIDTH-GATED (1.24x widest looped, inert 1.01x narrow persistent).
**Zero regressions** (worst 1.008x). Mechanism: re-read kernels read the row twice; default lets L2 evict it →
pass 2 re-streams from HBM; `'last'` keeps it resident. Free + never hurts → keep across all three tracks.

---

### Evidence files (this dir)
`cap_gating_verify.json`, `cap_ungate_kldiv_jsd.json` (#1) · `bandb_gate_test.json` (#2) ·
`reread_evict_verify.json` (#3). Related: `perf_ab_merged.json`, `t2_train_3way.json`, `editpid_verify.json`.
