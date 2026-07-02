# `_apply_reread` adversarial candidates — full handoff (2026-07-02)

## GPU-VERIFIED RESULTS (2 of 11 benched serially, H100)

1. **Candidate 1 `reread_softmax_denominator_check` = CONFIRMED WRONG CAP (2.24×).**
   File: `adversarial_synth/synth_reread_softmax_VERIFIED_wrongcap.py`. At N=49152 fp32:
   `apply_reread=False -> BIG -> seed persists [1,65536]`; MEASURED persist 708µs vs **chunk 316µs
   (chunk 2.24× faster)**. `row_reread=True` (armed, not inert). This is the real false-negative:
   softmax-isomorphic 2-pass kernel whose pass-2 REDUCES (entropy) instead of STORING, so its reread
   load is `(R,-)` not `(-,S)` and evades the clause, while the refetch cliff is unchanged.

2. **Candidate 0 `reread_variance_of_shifted_row` = NOT a wrong cap.**
   File: `adversarial_synth/synth_reread_variance_NOT_wrongcap.py`. `row_reread=False` (the
   `ones_like().sum` count constant-folded → hold inert) AND measured true preference is PERSIST
   (+31%). So a cheap `(x-mean)²` pass-2 does NOT reproduce; only the heavy softmax-isomorphic pass-2
   (#1) does. Good boundary: it's not "any reduce-then-reread," it's specifically the refetch-heavy one.

NEXT (unverified, ranked worklist at bottom): candidates 5,6,8 (BIG→SMALL, high conf) and the
false-positive set 2,4 (SMALL→BIG). Candidate 9 self-declared INERT (drop). Candidates 3,10 are a
DIFFERENT bug (bf16 itemsize miscalibration within a bucket, not a SMALL↔BIG flip).

---

> The adversarial workflow (`_lab/redesign/` … run this session) produced 11 synthetic Helion
> reduction kernels intended to make `_apply_reread` pick the WRONG persistence-hold ceiling
> (SMALL=294912 vs BIG=737280). Two were GPU-verified serially (see VERIFIED section at bottom).
> Verify any candidate with `_lab/redesign/verify_synth_kernel.py --mod <file> --persist '<cfg>' --chunk '<cfg>'`
> (FOREGROUND, one kernel/process; it prints pd + apply_reread + ceiling + seed reduction_loops, then
> A/Bs persist vs chunk median-of-9). GATE every candidate on `row_reread==True` first — if False the
> persistence hold is inert and the 'wrong cap' never changes the emitted config.
>
> Heuristic under test: `_apply_reread` = ANY load of a primary-reduction tensor with
> `stores_fed AND NOT reductions_fed` (a store-only 2nd-pass load). TRUE->SMALL, FALSE->BIG.
> Causal model (measured): a 2nd physical pass that REFETCHES the row from HBM wants SMALL (softmax
> N=49152 persist -34%); a single fused pass only pays chunk LOOP OVERHEAD, wants BIG (cross_entropy
> persist +47%). The clause keys on the reread load's SINK, which correlates with '2nd pass' on the
> corpus but is NOT identical — a normalize-THEN-reduce 2nd pass evades it.


## Candidate 0: reread_variance_of_shifted_row (two-pass: mean-pass forks, variance-pass re-reduces the reread row)

- **failure_mode:** false_negative_BIG_should_be_SMALL  | **heuristic picks:** BIG  | **true preference:** SMALL  | **helion_expressible:** True  | **confidence:** high
- **apply_reread predicted:** False
- **why wrong:** PHYSICS: this is a genuine two-physical-pass kernel over the same row x[tile_m,:] — identical to softmax/welford's read structure. Pass 1 streams the row (mean+count), Pass 2 RE-READS the same row (sum of squared deviations from the mean). If persistence holds the full row on-chip, Pass 2 reads it for free (~1 HBM read total, softmax measured +30% at N=32768); if it chunks, the row is evicted between the two loops and Pass 2 REFETCHES it from HBM (~2 reads) — the exact steep refetch cliff the SMALL ceiling exists to cap at N>=~49K where holding regresses (softmax N=49152 persist -34%). So the true preference is SMALL, identical to welford (a real SMALL corpus kernel with the SAME two-loop shape). HEURISTIC ERROR: welford/softmax get SMALL only because their Pass-2 load is a PURE store ((-,S) -> _apply_reread=True). Here Pass 2's load feeds a reduction before any store, so its signature is (R,-), never (-,S). _apply_reread scans for a `stores_fed AND NOT reductions_fed` load and finds none -> returns False -> BIG=737280. Meanwhile row_reread survives via Pass-1's cnt>=2 fork (one x load -> two sum() reductions on axis N -> reductions_fed=((N,2),)), so the persistence hold IS attempted and the ceiling DOES govern. Net: identical refetch physics to welford, opposite ceiling. The 2nd pass's own reduction residency does NOT rescue BIG: its accumulator is a scalar [M_BLOCK] carry (carried_2d_count=0, so the hold is not blocked), it holds no wide resident tile of its own, and it does not change the row's residency economics one bit — the row is the thing being refetched, exactly as in welford. So BIG lets the row persist past ~32K where softmax/welford measurably regress -34%: a true wrong-cap.

```python
@helion.kernel(static_shapes=False)
def reread_variance(x):  # x: [M, N] fp32, output [M] = var of (x - rowmean)
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    bm = hl.register_block_size(m)
    bn = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=bm):
        s   = hl.zeros([tile_m], dtype=torch.float32)   # scalar carry [M_BLOCK]
        cnt = hl.zeros([tile_m], dtype=torch.float32)
        # ---- PASS 1: reduce over the row. ONE load of x forks into TWO reductions
        #      on the SAME axis N -> reductions_fed = ((N, 2),) -> cnt>=2 -> row_reread=True,
        #      and this load is (R,-), never a pure (-,S).
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            s   += v.sum(dim=1)                          # reduction #1 on axis N
            cnt += (v * 0 + 1).sum(dim=1)                # reduction #2 on axis N (same load)
        mean = s / cnt
        # ---- PASS 2: a SEPARATE hl.tile(n) loop RE-READS x from HBM/L2 and REDUCES it
        #      (sum of squared deviations). Its load feeds a reduction -> (R,-), NOT (-,S).
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=bn):
            v2 = x[tile_m, tile_n]                       # SECOND physical read of the row
            d  = v2 - mean[:, None]
            acc += (d * d).sum(dim=1)                    # reduction on axis N -> (R,-)
        out[tile_m] = acc / cnt
    return out
```

## Candidate 1: reread_softmax_denominator_check (softmax pass-1 fork; pass-2 re-reduces exp instead of pure-store)

- **failure_mode:** false_negative_BIG_should_be_SMALL  | **heuristic picks:** BIG  | **true preference:** SMALL  | **helion_expressible:** True  | **confidence:** high
- **apply_reread predicted:** False
- **why wrong:** PHYSICS: byte-for-byte the SAME two-pass read pattern as softmax_two_pass (Pass 1 = online max+denominator, Pass 2 = re-sweep the row to consume the normalized probabilities). The ONLY change from real softmax is Pass 2's sink: instead of `out[tile_m, tile_n] = p` (a store), it reduces the probs into an entropy scalar. The row-residency economics are unchanged — Pass 2 still physically re-reads x[tile_m,:], so holding the row on-chip makes Pass 2 free (1 read) and chunking evicts+refetches it (2 reads), the identical cliff that makes real softmax want SMALL past N~=32K. HEURISTIC ERROR: softmax_two_pass gets _apply_reread=True purely because Pass 2's load flows to a STORE with no intervening reduction ((-,S)). Redirecting that load through a reduction makes it (R,-); the `not f.reductions_fed` clause in _apply_reread (triton.py:726) now fails, so _apply_reread=False -> BIG=737280. row_reread is still True (Pass-1's amax+sum fork gives cnt=2 on axis N). So the persistence hold is armed with the WRONG (2.5x too loose) ceiling. RESIDENCY-DEFENSE for SMALL: Pass 2's accumulator (`ent`) is a [M_BLOCK] scalar (carried_2d_count=0, hold not blocked); it adds no wide resident tile, so it cannot 'want BIG' on its own account. It is a pure consumer of the reread row exactly like softmax's normalize store. Therefore BIG lets the softmax row persist to ~600KiB where real softmax measurably regresses (-34% at N=49152): a wrong-cap by the same cliff, this kernel being maximally isomorphic to the corpus SMALL kernel it mimics. This is my highest-confidence pick because the ONLY delta from a MEASURED SMALL kernel (softmax_two_pass) is the Pass-2 sink kind, which is exactly the bit _apply_reread keys on while being causally irrelevant to row residency.

```python
@helion.kernel(static_shapes=False)
def reread_logsumexp_and_maxprob(x):  # x: [M,N] fp32 -> out[M] scalar summary
    m, n = x.size()
    out = torch.empty([m], dtype=torch.float32, device=x.device)
    bm = hl.register_block_size(m); bn = hl.register_block_size(n)
    for tile_m in hl.tile(m, block_size=bm):
        mi = hl.full([tile_m], float('-inf'), dtype=torch.float32)   # scalar carry
        di = hl.zeros([tile_m], dtype=torch.float32)                 # scalar carry
        # PASS 1 (online softmax stats): ONE x load forks -> amax + running sum.
        # reductions_fed=((N,2),) -> row_reread=True; load is (R,-), not (-,S).
        for tile_n in hl.tile(n, block_size=bn):
            v = x[tile_m, tile_n]
            a = torch.amax(v, dim=1)                 # reduction #1
            mn = torch.maximum(mi, a)
            di = di * torch.exp(mi - mn) + torch.exp(v - mn[:, None]).sum(dim=1)  # reduction #2
            mi = mn
        # PASS 2: SEPARATE loop RE-READS x and REDUCES the normalized probs (e.g. entropy /
        # max-prob) -> its load feeds a reduction -> (R,-), NOT a pure store.
        ent = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(n, block_size=bn):
            v2 = x[tile_m, tile_n]                   # SECOND physical read of the row
            p  = torch.exp(v2 - mi[:, None]) / di[:, None]
            ent += -(p * torch.log(p + 1e-9)).sum(dim=1)   # reduction on axis N -> (R,-)
        out[tile_m] = ent
    return out
```

## Candidate 2: logsumexp_normalize_at_argmax (tiny-slice second pass)

- **failure_mode:** false_positive_SMALL_should_be_BIG  | **heuristic picks:** SMALL  | **true preference:** BIG  | **helion_expressible:** True  | **confidence:** medium
- **apply_reread predicted:** True
- **why wrong:** The claim 'apply_reread=True <=> chunking refetches the whole row (steep cliff)' assumes pass 2 re-reads x at FULL row width and pays a second HBM pass. In this kernel the physical second pass is written to touch only a NEGLIGIBLE slice of x (the argmax column / a narrow band), but the (-,S) load STILL has stores_fed and NOT reductions_fed, so _apply_reread returns True and picks SMALL. Yet the dominant reduction here is the two-loop max+exp-sum over the FULL row, and the chunk-vs-persist economics are governed by that primary loop: persisting holds the row so the max pass and the exp-sum pass are served on-chip; chunking re-loops with partial accumulators and index recompute (the measured cross_entropy +47% at N=32000 / +7% at N=50257 pure loop overhead), while the tiny pass-2 refetch is ~0. Net: in the N in (73728, 184320] fp32 window the heuristic chunks at ~73728 (SMALL) and leaves the whole BIG-side persistence gain (up to ~+40%) on the table. The prompt's own SIBLING probe would flag this too (2 sibling passes), but the SIZE the second pass touches is what the causal model got wrong — a real (-,S) pass does not imply a costly refetch.

```python
@helion.kernel(static_shapes=False)
def logsumexp_scatter(x, idx):  # x:[M,N] fp32, idx:[M] int64
    M, N = x.shape
    out = torch.empty([M, N], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(M):
        # PASS 1 (primary reduction over N): full-row log-sum-exp
        m_acc = hl.full([tile_m], -inf, dtype=fp32)
        for tile_n in hl.tile(N):
            m_acc = torch.maximum(m_acc, x[tile_m, tile_n].amax(dim=1))
        s_acc = hl.zeros([tile_m], dtype=fp32)
        for tile_n in hl.tile(N):
            s_acc += torch.exp(x[tile_m, tile_n] - m_acc[:, None]).sum(dim=1)
        lse = m_acc + torch.log(s_acc)          # [tile_m]
        # PASS 2 (separate (-,S) re-read of x -> STORE, no reduction):
        # re-read x ONLY at the argmax column region and write a normalized slice.
        for tile_n in hl.tile(N):
            col = idx[tile_m]                    # [tile_m] chosen columns
            # gather a TINY slice of x back (or just the single argmax col),
            # normalize by lse, and store it. The (-,S) load below re-reads x.
            xr = x[tile_m, tile_n]               # <-- (-,S) load of primary tensor
            out[tile_m, tile_n] = xr - lse[:, None]  # store, no reduction
    return out
```

## Candidate 3: bf16_softmax_with_narrow_apply (small-itemsize refetch is cheap)

- **failure_mode:** false_positive_SMALL_should_be_BIG  | **heuristic picks:** SMALL  | **true preference:** BIG  | **helion_expressible:** True  | **confidence:** low
- **apply_reread predicted:** True
- **why wrong:** The hold-ceiling test is itemsize*(scale*raw_ext + flat) <= ceiling. The causal model calibrated SMALL vs BIG on fp32 softmax (N=32768 persist +30%, N=49152 persist regresses). But the ceilings are BYTE ceilings and _apply_reread carries NO dtype term, so bf16 gets the SAME 294912-byte SMALL ceiling -> persists only to raw_ext <= 147456 bf16 elems, chunks above. The physical refetch that motivates SMALL (pass 2 re-reads the row from L2/HBM) is HALF the bytes per element in bf16: the persist-vs-chunk cliff that made SMALL correct at fp32 is shifted OUT by ~2x in element count because both the refetch cost and the residency both scale with bytes, but the persistence element-cap (Triton max_tensor_numel=2^20) is far away and the L2 working-set the second pass re-reads is only itemsize*N bytes. Concretely: at N in (147456, 368640] bf16 the true row fits comfortably in the same L2 footprint fp32 softmax used at N in (73728, 184320], so persistence is still the win there — but SMALL forces a chunk at 147456. The heuristic conflates 'byte residency' with 'element residency': it should have shifted the ceiling for the cheaper-per-element refetch. BIG (368640 bf16) is the calibrated-correct element window.

```python
@helion.kernel(static_shapes=False)
def softmax_bf16_2pass(x):  # x:[M,N] bf16 (itemsize 2)
    M, N = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(M):
        # PASS 1: primary reduction over N (max then sum-exp), fp32 accum
        mx = hl.full([tile_m], -inf, dtype=fp32)
        for tile_n in hl.tile(N):
            mx = torch.maximum(mx, x[tile_m, tile_n].to(fp32).amax(dim=1))
        se = hl.zeros([tile_m], dtype=fp32)
        for tile_n in hl.tile(N):
            se += torch.exp(x[tile_m, tile_n].to(fp32) - mx[:, None]).sum(dim=1)
        # PASS 2: separate (-,S) re-read of x -> normalized store, no reduction
        for tile_n in hl.tile(N):
            xr = x[tile_m, tile_n].to(fp32)      # <-- (-,S) load of primary tensor
            out[tile_m, tile_n] = (torch.exp(xr - mx[:, None]) / se[:, None]).to(bf16)
    return out
```

## Candidate 4: poly_activation_reduce (compute-bound primary, cheap re-read)

- **failure_mode:** false_positive_SMALL_should_be_BIG  | **heuristic picks:** SMALL  | **true preference:** BIG  | **helion_expressible:** True  | **confidence:** medium
- **apply_reread predicted:** True
- **why wrong:** _apply_reread is purely a DATAFLOW predicate (a (-,S) sibling load of the primary tensor exists) -> True -> SMALL. But the persist-vs-chunk decision is a PERFORMANCE tradeoff, and here the primary pass is COMPUTE-bound (a long transcendental chain per element), so the machine spends far more cycles in ALUs than on the row's DRAM traffic. Chunking that compute-bound loop adds loop/index/partial-accumulator overhead per chunk (the same +47% mechanism measured on cross_entropy, amplified because each chunk boundary re-issues the setup for a costly body) while the ONLY thing SMALL 'buys' is avoiding a second read of the row in pass 2 — but that read is trivial next to the pass-1 compute, and it re-reads a warm row from L2 anyway. The heuristic assumes the row-refetch cost is the dominant cliff term; when the primary reduction is arithmetic-heavy, the loop-overhead term dominates and monotonically favors persisting to the largest resident extent (BIG). So in the N in (73728, 184320] fp32 window the heuristic chunks at ~73728 (SMALL) and forfeits the persistence win. This is the purest form of the false positive: the predicate is structurally correct (there IS a second pass) but the economics it stands in for are inverted by the compute intensity of pass 1.

```python
@helion.kernel(static_shapes=False)
def expensive_reduce_then_scale(x):  # x:[M,N] fp32
    M, N = x.shape
    out = torch.empty([M, N], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(M):
        # PASS 1: primary reduction over N, but with a HEAVY per-element transcendental
        # chain BEFORE the sum (many exp/log/erf ops) -> the loop is COMPUTE-bound,
        # not memory-bound. This is the dominant reduction pd.
        acc = hl.zeros([tile_m], dtype=fp32)
        for tile_n in hl.tile(N):
            v = x[tile_m, tile_n]
            v = torch.erf(torch.exp(v) * torch.log1p(v*v) + torch.tanh(v))  # heavy
            acc += v.sum(dim=1)
        norm = acc                                # [tile_m]
        # PASS 2: separate (-,S) re-read of x -> scaled store, no reduction, CHEAP
        for tile_n in hl.tile(N):
            out[tile_m, tile_n] = x[tile_m, tile_n] / norm[:, None]  # (-,S) load
    return out
```

## Candidate 5: row_multistat_fused (single pass, K independent scalar accumulators + heavy transcendental epilogue over one full-slice fp32 row)

- **failure_mode:** false_negative_BIG_should_be_SMALL  | **heuristic picks:** BIG  | **true preference:** SMALL  | **helion_expressible:** True  | **confidence:** high
- **apply_reread predicted:** False
- **why wrong:** There is exactly ONE physical load of the row (`x[tile_m, :]`); it forks to BOTH the amax/sum reductions AND (through the derived scalars) the store, so it is an (R,S) load, never a separate (-,S) pass. `_apply_reread` requires a load with `stores_fed AND NOT reductions_fed`; none exists => returns False => hold_ceiling = PERSIST_HOLD_MAX_BYTES = 737280 (BIG). The persist gate `itemsize*(scale*raw_ext+flat) <= 737280` is the ONLY test. The 8 accumulators are rank-1 `[tile_m]` tiles, so by `_graph_peak_live_tiles`' rule 'one rank-2 tile beats any number of rank-1 tiles' the peak-live set is the single `[m,V]` read tile: scale=m_block=1, and the 8 carries land in `flat` at 1 elem each (flat~=8). Footprint = 4*(1*65536+8) ~= 262 KiB << 737280 => the heuristic PERSISTS the full 65536-wide row. But the TRUE resident state per program is the 262 KiB row PLUS 8 live fp32 accumulators per lane PLUS the transcendental epilogue's high register liveness (exp/log1p over the held row), which drives register-file occupancy to a fraction of one CTA/SM well before N reaches the ceiling. cross_entropy N=40960 (a 2-accumulator instance of THIS exact shape at only 22% of the ceiling) already measures chunk +13% FASTER purely from occupancy. This kernel deliberately multiplies the per-element register state (8 carries + heavy epilogue) so the occupancy cliff bites at an even smaller N-fraction; chunking caps the resident row to LOOPED_CHUNK width (keeping many CTAs/SM) and pays only loop/partial-accumulator overhead, which is cheap when there is no refetch. The byte ceiling models per-CTA footprint but has NO term for CTAs-per-SM (occupancy) or accumulator-multiplicity register pressure — the exact axis the prompt names.

```python
@helion.kernel(static_shapes=False)
def row_multistat_fused(x, w0, w1, w2, w3, w4, w5, w6, w7):  # x: [M, V] fp32, wk: [V] fp32
    M, V = x.shape
    out = torch.empty([M, 8], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):        # grid-pinned row => m_block = 1 => scale = 1
        row = x[tile_m, :]                          # ONE FULL_SLICE load of the whole V-wide row (row_reread=True)
        mx  = torch.amax(row, dim=-1)               # secondary reduction, reuses the SAME loaded row
        z   = row - mx[:, None]
        e   = torch.exp(z)
        den = torch.sum(e, dim=-1)                   # PRIMARY reduction (log-sum-exp denominator)
        # 8 independent weighted moments of the SAME resident row, each a [tile_m] scalar carry:
        s0 = torch.sum(e * w0[None, :], dim=-1)
        s1 = torch.sum(e * w1[None, :] * z,        dim=-1)
        s2 = torch.sum(e * w2[None, :] * z * z,    dim=-1)
        s3 = torch.sum(e * w3[None, :] * torch.exp(z * 0.5), dim=-1)
        s4 = torch.sum(e * w4[None, :] * z * z * z, dim=-1)
        s5 = torch.sum(row * w5[None, :],          dim=-1)
        s6 = torch.sum((row * row) * w6[None, :],  dim=-1)
        s7 = torch.sum(torch.log1p(e) * w7[None, :], dim=-1)
        # epilogue writes 8 SCALARS derived FROM the reductions (no second x load => no (-,S) load):
        out[tile_m, 0] = s0 / den
        out[tile_m, 1] = s1 / den
        out[tile_m, 2] = s2 / den
        out[tile_m, 3] = s3 / den
        out[tile_m, 4] = s4 / den
        out[tile_m, 5] = s5
        out[tile_m, 6] = s6
        out[tile_m, 7] = s7
    return out
# Sizing point: fp32, V = 65536 (footprint 4*(1*65536+flat) ~= 262 KiB, 36% of BIG=737280). Also runs V in [40960, 184320].
```

## Candidate 6: wide_logsumexp_score (minimal single-pass full-slice fp32 reduction scaled past the measured occupancy cliff)

- **failure_mode:** false_negative_BIG_should_be_SMALL  | **heuristic picks:** BIG  | **true preference:** SMALL  | **helion_expressible:** True  | **confidence:** high
- **apply_reread predicted:** False
- **why wrong:** Identical dataflow to corpus cross_entropy: one load of the row forks to amax + exp/sum + the scalar store => (R,S), no (-,S) load => `_apply_reread` False => BIG=737280. Persist gate: 4*(1*98304) = 393216 <= 737280 => the heuristic PERSISTS a 98304-wide fp32 row = 384 KiB/program. That is ~6x the H100 SM register file (65536 x 32-bit = 256 KiB) held by a SINGLE program => heavy spilling and fractional CTAs/SM. The corpus already proves the cliff is INSIDE the BIG range for this exact kernel family: cross_entropy chunk is +13% FASTER at V=40960 (only 22% of the ceiling, footprint 164 KiB). At V=98304 (52% of ceiling) the persistent tile is ~2.4x larger and the occupancy deficit is strictly worse, yet the heuristic still says persist because the byte gate has no occupancy term. Truth wants SMALL/chunk (LOOPED_CHUNK keeps resident registers bounded, occupancy high; the only cost of chunking a single-fused-pass reduction is loop overhead, since there is NO row refetch — chunk reads the same DRAM once). Highest-confidence candidate: it is a pure rescale of a measured corpus exception, so the causal argument needs no new physics.

```python
@helion.kernel(static_shapes=False)
def wide_logsumexp_score(logits, targets):  # logits: [M, V] fp32, targets: [M] int
    M, V = logits.shape
    logits_flat = logits.view(-1)
    out = torch.empty([M], dtype=torch.float32, device=logits.device)
    for tile_m in hl.tile(M, block_size=1):          # grid-pinned row => scale = 1
        row = logits[tile_m, :]                       # ONE FULL_SLICE load of the V-wide row
        mx  = torch.amax(row, dim=-1)                 # secondary reduction (reuses same row)
        lse = mx + torch.log(torch.sum(torch.exp(row - mx[:, None]), dim=-1))   # PRIMARY reduction
        tgt = hl.load(logits_flat, [tile_m.index * V + targets[tile_m]])
        out[tile_m] = lse - tgt                       # scalar store derived from the reductions
    return out
# This is cross_entropy per-row, but sized at V = 98304 (fp32) — footprint 4*98304 = 384 KiB, 52% of BIG=737280.
```

## Candidate 7: row_gated_energy (single (R,S)-fork load: same resident row feeds a reduction AND an in-pass normalized elementwise store, heavy per-element epilogue)

- **failure_mode:** false_negative_BIG_should_be_SMALL  | **heuristic picks:** BIG  | **true preference:** SMALL  | **helion_expressible:** True  | **confidence:** medium
- **apply_reread predicted:** False
- **why wrong:** This is the (R,S)-fork trap. The single load `x[tile_m, :]` forks to BOTH the `sum(e*e)` reduction AND the elementwise store `out[tile_m,:]` — WITHOUT the store's value passing through a separate re-read of x. So that one load has reductions_fed non-empty AND stores_fed non-empty => it is (R,S), and `_apply_reread`'s predicate `stores_fed AND NOT reductions_fed` is FALSE for it (and there is no other load of x). => `_apply_reread` False => BIG. Contrast the corpus SMALL kernels (softmax/rms/layer_norm): those use a SEPARATE second `hl.tile(N)` pass that RE-READS x to normalize — that second load is (-,S) => True => SMALL. Here the normalize is FUSED into the same pass on the resident row, which is what SILENCES the signal. Persist gate: 4*(1*131072) = 524288 <= 737280 => persist the 512 KiB row (8x the SM register file). But physically this is a full-width [m,V] rank-2 output tile ALIVE simultaneously with the [m,V] input and the sigmoid/rsqrt intermediates — the true resident set is ~2-3 full V-wide tiles, and the register-occupancy cliff arrives far below the ceiling. NOTE the honest caveat: because the OUTPUT is a full-width [m,V] store, the peak-live tile set genuinely contains a second rank-2 tile, so `scale` here is ~2 (not 1) — the footprint is LESS undercounted than candidates 1/2, and at scale=2 the gate would already reject V=131072 (4*2*131072=1.05MB>737280) and pass only V<=92160. So the CLEAN false-negative window is V in [~46080, 92160]: gate says persist (scale=2), occupancy says chunk. This is the lower-confidence variant precisely because the fused full-width store partly re-enters the footprint the heuristic models — included to show the boundary where the (R,S)-fork evasion still leaves an occupancy gap but the byte model is no longer blind.

```python
@helion.kernel(static_shapes=False)
def row_gated_energy(x, gate):  # x: [M, V] fp32, gate: [V] fp32
    M, V = x.shape
    out = torch.empty([M, V], dtype=torch.float32, device=x.device)
    for tile_m in hl.tile(M, block_size=1):          # scale = 1
        row = x[tile_m, :]                            # ONE load, kept resident
        g   = gate[None, :]
        e   = row * torch.sigmoid(row) * g            # heavy per-element transform of the held row
        energy = torch.sum(e * e, dim=-1)             # PRIMARY reduction over the SAME resident row
        # normalized output written from the SAME loaded row (fused, NOT a re-read):
        out[tile_m, :] = e * torch.rsqrt(energy[:, None] + 1e-6)
    return out
# fp32, V = 131072: footprint 4*(1*131072) = 512 KiB, 71% of BIG=737280.
```

## Candidate 8: sibling_max_reread_then_store (false-negative: BIG picked, wants SMALL)

- **failure_mode:** false_negative_BIG_should_be_SMALL  | **heuristic picks:** BIG  | **true preference:** SMALL  | **helion_expressible:** True  | **confidence:** high
- **apply_reread predicted:** False
- **why wrong:** This is the KNOWN-evasion seed weaponized. The primary pd = the sum over the row (rolled, FULL_SLICE). red_tensors={'x'} (x feeds the primary sum). _apply_reread scans for a load with (stores_fed AND NOT reductions_fed). PASS 3's load of x IS such a load (feeds only the store, no reduction) -> _apply_reread would fire True (SMALL) via PASS 3. BUT if you REMOVE PASS 3 and keep only PASS 1 (sum) + PASS 2 (max->store amax): PASS 2's load of x has reductions_fed={max-axis} AND stores_fed={amax scalar}. The gate requires `not f.reductions_fed`, so PASS 2's load is (R,S), NOT (-,S) -> filtered out. No load qualifies -> _apply_reread=False -> BIG ceiling. Physically PASS 2 is a genuine 2nd sweep over the whole row from HBM/L2 (identical eviction physics to softmax's max pass), so past ~32K fp32 elems streaming beats holding: persist regresses exactly like softmax N=49152 (-34% measured). The heuristic conflates 'the reread load also happens to feed a reduction' with 'the row's reuse is register-resident within one fused pass' — but here the reduction in PASS 2 is over the SAME axis in a SEPARATE graph (sibling), so it is NOT register-resident, it is an L2 round-trip. LOAD-BEARING: primary row_reread=True (PASS 2's load feeds the primary sum? no — but PASS 1's/PASS 3's do). Concretely PASS1 load feeds sum (cnt>=1) and there is a bypass store in PASS3 -> row_reread=True on the primary axis; carried_2d_count=0; so ext_held is gated purely by the ceiling. At N=49152 fp32 the row is 49152*4=196608 B: <= BIG(737280) so persist=True, but > SMALL(294912)? 196608 < 294912 so even SMALL persists here — must push to N=65536 (262144 B < 294912, still persists both) so pick N=131072 fp32: 524288 B, > SMALL(294912) rejects (chunk, CORRECT), <= BIG(737280) accepts (persist, WRONG). At N=131072 the true optimum is chunk (streaming the 2-pass row), BIG makes the seed emit reduction_loops=[None] (persist) -> the measured -30%..-34% regression. The pick is LOAD-BEARING: persistent flips reduction_loops [None]<->[r_block].

```python
@helion.kernel(static_shapes=False)
def k(x):  # x: [M, N] fp32, N in {32768, 49152, 65536}
    M, N = x.shape
    out = torch.empty_like(x)
    amax = x.new_empty([M])
    for tile_m in hl.tile(M):
        # PASS 1 (PRIMARY reduction): sum over the row -> feeds a reduction, no store
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(N):
            acc += x[tile_m, tile_n].sum(dim=1)
        denom = acc[:, None]
        # PASS 2 (SEPARATE physical re-read of the SAME row x): a SECONDARY reduction (max)
        # then STORE the scalar. This load feeds a reduction (max) AND a store (amax scalar).
        mrow = hl.full([tile_m], float('-inf'), dtype=torch.float32)
        for tile_n in hl.tile(N):
            mrow = torch.maximum(mrow, x[tile_m, tile_n].max(dim=1).values)
        amax[tile_m] = mrow
        # PASS 3 (SEPARATE physical re-read of the SAME row x): normalize+store using denom only
        for tile_n in hl.tile(N):
            out[tile_m, tile_n] = x[tile_m, tile_n] / denom
    return out, amax
```

## Candidate 9: cross_tensor_reread_B_not_A (false-negative via red_tensors membership filter)

- **failure_mode:** false_negative_BIG_should_be_SMALL  | **heuristic picks:** BIG  | **true preference:** SMALL  | **helion_expressible:** True  | **confidence:** medium
- **apply_reread predicted:** False
- **why wrong:** This attacks the red_tensors MEMBERSHIP filter directly (prompt idea (a)). red_tensors = {tensors LOADED that feed the primary} = {'x'} only (y is never loaded in the primary sum pass). _apply_reread then looks for a (-,S) load whose tensor is IN red_tensors: PASS 2's load is of 'y', which is NOT in red_tensors, so it is filtered out. No qualifying load -> _apply_reread=False -> BIG. But the persistence economics of the standard track are governed by the PRIMARY's row (x): is x's row held on-chip across passes? x is read ONCE (PASS 1 only), fully consumed from registers into acc — x is NEVER re-read. So for X specifically, a single fused pass IS correct and BIG is arguably right FOR X. HOWEVER: the interesting failure is the DUAL — swap which tensor the primary reduces. If instead the PRIMARY sum is over y (the tensor that is ALSO re-read in a store-only pass) and x is a one-shot side input, then red_tensors={'y'}, PASS-2 y-load is (-,S) in red_tensors -> _apply_reread=True -> SMALL correctly. The asymmetry exposes that the heuristic's verdict depends on WHICH tensor is nominally 'primary', not on whether ANY row is physically re-read. In the pseudocode as written the reread is of y and the persistence-hold is applied to x's row (the primary). x's row is NOT re-read so its hold verdict (BIG, persist far) does not directly regress — BUT it is INERT/HARMLESS here: x is single-pass so persist is genuinely fine, and y's reread residency is a SEPARATE non-reduction apply loop sized by the LAST-pass loop_budget (triton.py:1188, always ROW_PERSIST-capped, never the hold ceiling), so y never reaches the hold at all. VERDICT: the membership filter is exploitable to MISDIRECT the ceiling onto the wrong tensor, but in THIS topology the misfire is INERT because x genuinely wants BIG and y is sized off a different knob. It becomes a real regression only if x itself is also re-read in yet another store-only pass of x — which collapses back to candidate 1. So this candidate documents that the cross-tensor membership evasion is inert unless the primary tensor is itself re-read.

```python
@helion.kernel(static_shapes=False)
def k(x, y):  # x,y: [M, N] fp32; PRIMARY reduces x, a SEPARATE pass re-reads y and stores
    M, N = x.shape
    out = torch.empty_like(y)
    for tile_m in hl.tile(M):
        # PASS 1 (PRIMARY): rolled sum over x's row -> denom. Only x feeds the primary.
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(N):
            acc += x[tile_m, tile_n].sum(dim=1)
        denom = acc[:, None]
        # PASS 2 (SEPARATE physical pass): re-read y's row (a DIFFERENT tensor) and STORE.
        # y is NEVER read in PASS 1, so y does NOT feed the primary reduction.
        for tile_n in hl.tile(N):
            out[tile_m, tile_n] = y[tile_m, tile_n] * denom
    return out
```

## Candidate 10: bf16_input_fp32_accum_ceiling_mismatch (false-positive/negative via itemsize)

- **failure_mode:** false_positive_SMALL_should_be_BIG  | **heuristic picks:** SMALL  | **true preference:** BIG  | **helion_expressible:** True  | **confidence:** medium
- **apply_reread predicted:** True
- **why wrong:** This attacks the FIXED BYTE ceiling vs the true tile-limit cliff under a dtype mismatch (prompt idea (c)). _apply_reread fires True correctly (PASS 2 re-reads x -> (-,S) load in red_tensors={'x'}) -> SMALL(294912). The persistence-hold byte test at triton.py:1088 uses `itemsize * (scale*raw_ext + flat)` where itemsize = pd.itemsize = the fp32-PROMOTED accumulator itemsize (4), NOT the bf16 HBM width. But the PHYSICAL cliff the SMALL ceiling encodes ('past ~32K elements the re-read row won't stay L2-resident') is about the HBM-resident bytes of the actual x row = N * 2 (bf16), while the L2/SRAM footprint that governs whether persist beats chunk is the on-chip working set. The claim behind SMALL is 'softmax regresses past ~32K ELEMENTS (N=49152)'. With bf16 input the SAME 32K-element cliff sits at 32768 elements = 65536 HBM bytes, but the seed's byte test on the fp32 accumulator gives 32768*4=131072 B, and SMALL admits up to 294912/4 = 73728 fp32 elements. So the seed lets a bf16 row PERSIST all the way to N=73728 ELEMENTS (using the fp32-elem budget) — FAR past the ~32K-49K element cliff where softmax's re-read row actually stops fitting L2. The element cliff is dtype-INDEPENDENT (it is an occupancy/L2-capacity limit in elements, per the corpus note softmax N=49152 persist -34%), but the byte ceiling is applied to fp32-promoted bytes, so a bf16 kernel gets ~2x the element headroom the fp32-tuned SMALL ceiling intended. Result: at N in (49152, 73728) the seed picks persist (SMALL admits it because 49152*4=196608 < 294912) when the true 2-pass optimum is CHUNK — a false-persist within the SMALL bucket. This is a SMALL-bucket internal miscalibration rather than SMALL-vs-BIG, but it shows the fixed byte ceiling on promoted itemsize does not track the element cliff for narrow inputs. LOAD-BEARING: persistent flips reduction_loops [None]<->[r_block]; at N=49152 bf16 it wrongly emits [None] (persist) mirroring softmax's measured -30%..-34%. Note the element_cap (backend.max_tensor_numel) also gates at 1088 and may pre-empt persist at very large N regardless.

```python
@helion.kernel(static_shapes=False)
def k(x):  # x: [M, N] bf16 (itemsize 2), fp32 accumulator; softmax-style 2-pass
    M, N = x.shape
    out = torch.empty_like(x)
    for tile_m in hl.tile(M):
        # PASS 1 (PRIMARY): rolled max+sum over the bf16 row (fed to a reduction)
        acc = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(N):
            acc += x[tile_m, tile_n].to(torch.float32).exp().sum(dim=1)
        denom = acc[:, None]
        # PASS 2 (SEPARATE re-read of x's row -> normalize+store): (-,S) load -> apply_reread True
        for tile_n in hl.tile(N):
            out[tile_m, tile_n] = (x[tile_m, tile_n].to(torch.float32).exp() / denom).to(torch.bfloat16)
    return out
```

---

## RANKED VERIFICATION WORKLIST (lead reviewer)

There's already a harness. I have enough to produce the verified worklist. Let me do a final static check on the two highest-confidence false-negative-BIG-should-be-SMALL candidates (1 and 2), which turn on whether Pass-2's reducing re-read load is really classified `(R,-)` and thus suppresses `_apply_reread`, while `row_reread` survives via the Pass-1 fork.

For candidate 1 (`reread_variance`): Pass-1 has `v.sum` and `(v*0+1).sum` — two reductions on axis N from one load → `reductions_fed = ((N,2),)` → cnt=2 → row_reread=True. `_apply_reread`: red_tensors={x} (Pass-1 load feeds primary). Is there any load of x with `stores_fed and not reductions_fed`? Pass-2's load feeds `acc += (d*d).sum` — a reduction → `reductions_fed` non-empty → filtered out. Pass-1's load: no store. So `_apply_reread=False` → BIG. row_reread=True, carried_2d_count=0 → hold gate active. **The mechanism is exactly as claimed.** This is a genuine 2-physical-pass kernel (same as welford/softmax refetch physics) getting BIG. Sound.

Candidate 2 (`reread_softmax`): identical structure, isomorphic to softmax_two_pass except Pass-2 reduces instead of stores. Same verdict. Highest fidelity to a measured SMALL corpus kernel.

One caveat to verify: whether Helion's cnt>=2 counting actually fires for `(v*0+1).sum` (candidate 1) vs the cleaner `amax`+`sum` fork (candidate 2). Candidate 2's fork is two genuine distinct reduction ops (`amax` and the `exp().sum`), which is exactly softmax's own Pass-1 structure — high confidence it registers cnt=2. Candidate 1's `(v*0+1)` may get constant-folded. So candidate 2 is more robust than 1 on the `row_reread` survival.

I have everything needed to rank.

## VERIFICATION WORKLIST — `_apply_reread` wrong-cap candidates

Verified against source: `PERSIST_HOLD_MAX_BYTES = 3*245760 = 737280` (BIG), `USER_TILE_PERSIST_HOLD_MAX_BYTES = 294912` (SMALL) [triton.py:589-590]. `_apply_reread` predicate = a load `stores_fed AND NOT reductions_fed` whose tensor also feeds the primary [726]. Hold gate `ext_held` requires `row_reread AND carried_2d_count==0 AND itemsize*(scale*raw_ext+flat) <= hold_ceiling` [1084-1088]. `row_reread` fires iff some load has `cnt>=2` OR `cnt>=1 AND stores_fed` on the primary axis [device_ir.py:1776]. Footprint is a faithful Σ over live tiles — scalar carries land in `flat` at ~1 elem each; a full-width rank-2 store IS a live tile (enters `scale`) [triton.py:969-999]. Confirmed: the byte gate has NO occupancy/CTA-per-SM term and NO accumulator-multiplicity term.

### RANKED SHORTLIST (real, expressible, sound; most-likely-real first)

**1. reread_softmax_denominator_check (cand #2) — false-neg, BIG picked, wants SMALL.**
Byte-isomorphic to the MEASURED SMALL kernel softmax_two_pass; only delta is Pass-2 sink (reduce vs store), which is exactly the `not reductions_fed` bit `_apply_reread` keys on but is causally irrelevant to row residency. row_reread survives via Pass-1 amax+sum fork (two genuine distinct reduction ops → cnt=2, robust). Not inert.
What to measure: fp32, M large, **N=49152**. A/B `reduction_loops=[null]` (persist, seed's BIG pick) vs `[16384]` (chunk). Expect chunk to win (~ softmax N=49152 persist −34%). Also confirm at N=32768 persist still wins (SMALL boundary correct).
Prior real wrong-cap: **~0.75.**

**2. reread_variance_of_shifted_row (cand #1) — false-neg, BIG→SMALL.**
Same mechanism, genuine two-physical-pass (mean-pass forks, variance-pass re-reduces the reread row = welford refetch physics). One risk vs #2: row_reread relies on the Pass-1 fork `sum` + `(v*0+1).sum`; the count-of-ones may constant-fold, dropping cnt to 1 with no store → row_reread=False → hold INERT (chunks regardless, cap never applies). Verify row_reread=True first.
What to measure: fp32, **N=49152**, persist vs chunk. Gate: first print `pd.row_reread` — if False, candidate is INERT (drop). If True, expect chunk win.
Prior: **~0.6** (docked for the row_reread constant-fold risk).

**3. sibling_max_reread_then_store (cand #8 in list) — false-neg, BIG→SMALL.**
Weaponized known-evasion: 3 sibling passes; drop Pass-3 so only Pass-1(sum) + Pass-2(max→store amax) remain. Pass-2's load is (R,S) → filtered → `_apply_reread=False` → BIG, yet Pass-2 is a genuine 2nd HBM sweep. Author correctly notes it's load-bearing only at **N=131072 fp32** (524288 B: > SMALL 294912 rejects=chunk, ≤ BIG 737280 accepts=persist). At N≤65536 BOTH ceilings persist → INERT. Must use the 2-pass (no Pass-3) variant AND N=131072; row_reread via Pass-1 sum + (if Pass-3 kept) bypass store.
Caveat: with Pass-3 removed, row_reread needs Pass-1's cnt>=1 AND a stores_fed on x — but Pass-2 stores `amax`, a scalar reduced from x, so that load is (R,S), cnt>=1+store → row_reread=True. OK.
What to measure: 2-pass variant, **N=131072 fp32**, persist vs chunk.
Prior: **~0.55.**

**4. wide_logsumexp_score (cand #7) — false-neg (occupancy), BIG→SMALL.**
Pure rescale of the MEASURED cross_entropy occupancy exception (chunk +13% faster at N=40960, only 22% of BIG). At **N=98304 fp32** (52% of ceiling, 384 KiB/program, grid-pinned block_size=1 → scale=1) the byte gate persists but occupancy is strictly worse than the measured-losing point. row_reread=True via amax+sum fork (cnt=2). Sound; the ONE unknown is whether the +13% occupancy effect grows monotonically with N or the N=40960 point was a local occupancy sweet-spot (cross_entropy N=50257 persist was +7%, i.e. persist WON just above 40960 — non-monotone!). That non-monotonicity is the main risk.
What to measure: single-pass, grid-pinned bm=1, **N=98304 fp32**, persist vs chunk. Also spot N=65536, 131072 to map the occupancy curve.
Prior: **~0.45** (docked hard for the measured non-monotonicity right next to the exception point).

**5. row_multistat_fused (cand #6) — false-neg (occupancy + register multiplicity), BIG→SMALL.**
Deliberately multiplies register state (8 scalar carries + heavy transcendental epilogue) to bite the occupancy cliff at smaller N. Mechanism-correct that the byte model is blind to both accumulator multiplicity (8 carries → flat≈8, negligible) and epilogue register liveness. BUT: this is a NEW physics claim (register-file occupancy from carries+epilogue), not a rescale of a measured point, so the size at which it bites is unknown. At V=65536 the footprint is only 36% of BIG — needs the register cliff to arrive that early, unproven.
What to measure: fp32, grid-pinned bm=1, **V=65536**, persist vs chunk; sweep V∈{40960, 98304, 131072}. Watch `ncu` achieved occupancy / registers-per-thread, not just DRAM.
Prior: **~0.4.**

### LOWER PRIORITY / borderline

**6. poly_activation_reduce (cand #5) — false-pos (compute-bound), SMALL→BIG.** Genuine 2nd (−,S) pass so `_apply_reread=True`→SMALL correctly by dataflow, but claims compute-bound Pass-1 inverts the economics. Plausible but the (−,S) Pass-2 refetch is real; whether compute dominance flips it is a live empirical question. Prior **~0.35.** What to measure: fp32, **N=98304**, persist([null]) vs chunk([16384]); heavy erf/exp/log1p body.

**7. logsumexp_normalize_at_argmax (cand #3) — false-pos (tiny-slice 2nd pass), SMALL→BIG.** Claims Pass-2 touches only a negligible x slice so no real refetch. Weakness: as written Pass-2 loops full `hl.tile(N)` and reads `x[tile_m,tile_n]` full-width — NOT actually a tiny slice, so the (−,S) refetch IS full and SMALL may be right. To be real the second pass must genuinely subset x, which is awkward to express. Prior **~0.3.** Likely mis-specified as written.

### FLAG — INERT / likely-inert (do NOT burn GPU without the guard checks)

- **cross_tensor_reread_B_not_A (cand #9):** author self-declares INERT in the written topology — primary x is single-pass (genuinely wants BIG), reread-y is sized off the last-pass loop_budget, never the hold ceiling. Only becomes real if x itself is re-read (collapses to cand #1). **DROP.**
- **bf16_softmax_with_narrow_apply (cand #4) / bf16_input_fp32_accum_ceiling_mismatch (cand #10):** SMALL-bucket *internal* miscalibrations (element-vs-byte, promoted-itemsize), not SMALL↔BIG flips. `_apply_reread`=True→SMALL is picked correctly; the claim is the byte value is off for bf16. These are a *different* bug (itemsize term) than "picks wrong ceiling" and lower-confidence per author. Deprioritize unless specifically auditing the itemsize term. Prior each **~0.3.**
- **INERT-RISK on cand #1 and #8:** both must have `pd.row_reread==True` confirmed first (cand #1 = ones-count fold risk; cand #8 = must drop Pass-3 AND hit N=131072 or both ceilings persist). If row_reread is False, or footprint fits both ceilings at the chosen N, the cap never changes the emitted `reduction_loops` → INERT.

### Suggested GPU order
Run **#1 (cand2)** and **#4 (cand7)** first — highest priors and cheapest to falsify (cand7 is a pure rescale of a corpus point). Then #2, #3. Gate #1/#3/#5/#6 on a `pd.row_reread` + `ext_held` print (via `_lab/redesign/verify_synth_kernel.py`) BEFORE benching. Harness already exists: `/home/dev/local/helion-redesign/_lab/redesign/verify_synth_kernel.py` (prints pd, `_apply_reread`, ceiling, seed `reduction_loops`, and A/Bs persist vs chunk).

Key source refs: `/home/dev/local/helion-redesign/helion/_compiler/autotuner_heuristics/triton.py:695-728` (`_apply_reread`), `:1084-1100` (hold gate + chunk fallback), `:969-999` (Σ-live-tiles footprint), `/home/dev/local/helion-redesign/helion/_compiler/device_ir.py:1770-1788` (`row_reread`).