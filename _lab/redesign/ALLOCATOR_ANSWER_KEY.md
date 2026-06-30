# Corpus answer-key for the budget allocator (probed at 34ae072e)

Every DISTINCT reduction structure in the corpus, with its Stage-1 facts + the CURRENT seed config.
Notation: RED `cat bid sh(extent) isz ilsz c2d(carried) fwo blt(body_live) reread`. ACC `dims (sizes)`.
"grid" = grid_axis_block_ids. Configs are `block_sizes` (positional by valid bid) + reduction_loops + warps.
num_sm(H100)=132, MIN_WAVES=8 → occupancy floor num_sm·MIN_WAVES=1056.

## STRUCTURE 1 — FULL_SLICE rolled, standard track, persistent (rms_norm, sum, long_sum, cross_entropy, most transfer)
Pattern: grid=(0,) [rows], RED FULL_SLICE bid1 over features, rl_valid=[1] (rolled). ACC [0,None] per-row scalar.
- rms_norm (8192,768) fp32: RED sh=768 blt=3 reread=T fwo=T. seed bs=[4] rl=[None] w=4.  (persistent, grid-M widen=4)
- rms_norm (2048,16384): sh=16384. bs=[1] rl=[None] w=16.  (grid-M floors to 1; wide row)
- rms_norm (32768,2048): sh=2048. bs=[8] rl=[None] w=8.  (grid-M widen=8)
- rms_norm (16384,4096): sh=4096. bs=[4] w=8.
- sum (16384,1024): FULL_SLICE blt=2 reread=F fwo=F. bs=[8] rl=[None] w=4 evict=stream.  (num_load==1 stream)
- long_sum (256,65536): sh=65536 blt=2 reread=F. bs=[1] rl=[16384] w=32.  (NOT persistent → looped chunk 16384)
- cross_entropy (8192,30522): sh=30522 blt=2 reread=T fwo=F. bs=[1] rl=[None] w=32.  (persistent, grid floors 1)
GRID-M widen rule (independent rows): M_BLOCK = min(occupancy_cap=grid_rows/1056, byte_widen, extent), floor 1.
  byte_widen ≈ ROW_PERSIST_MAX_BYTES / (R_BLOCK_resident · blt · itemsize). reread→persistent (rl=[None]).
  long_sum: not reread → looped, chunk=min(LOOPED_CHUNK=16384, byte_budget, extent).

## STRUCTURE 2 — USER_TILE persistent, single reduction (softmax, dynamic_per_token, dynamic_quant)
Pattern: grid=(0,), RED USER_TILE bid1, rdim IS a block_sizes slot. r_block seated full-extent (persistent).
- softmax (262144,128) fp32: sh=128 blt=2 reread=T fwo=T. bs=[16,128] w=4.  (r_block=128 full, grid-M=16)
- softmax (16384,512): sh=512. bs=[8,512] w=1.  (narrow-w1: row_bytes=512·4=2048 ≤ NARROW_W1_MAX_BYTES)
- softmax (2048,32768): sh=32768. bs=[1,32768] w=32.
- dynamic_per_token_scaled_fp8 (128,8192) bf16: USER_TILE sh=8192 blt=2. bs=[8192,8192] w=16.  (r_block=8192 full; non-red loop bid2 also 8192)
GRID-M (rows, bid0): widens into remaining budget, occupancy-capped, floors when r_block fills budget.

## STRUCTURE 3 — carried-2D accumulator (kl_div, jsd)
Pattern: grid=(rowaxis,), RED USER_TILE bidR with c2d≥1; ACC [grid_M, rdim] resident the whole loop.
- kl_div (8192,30522) fp32: RED USER_TILE bid0 sh=30522 c2d=1 blt=6. grid=(1,). ACC [1,0] (8192,30522). bs=[4096,1] w=32.
    → r_block(bid0)=4096 (carried byte cap: CARRIED_TILE_MAX_BYTES=16384/(itemsize4·c2d1)=4096). grid bid1 floors 1.
- jsd (8192,30522): THREE groups (g1,g3 GRID_TILE bid1; g5 USER_TILE bid0 sh=30522 c2d=2). bs=[2048,1] w=32.
    → r_block=2048 (16384/(4·2)). The 2 GRID_TILE reductions over bid1 FLOOR (grid-parallelized). grid bid1=1.
  *** jsd's GRID_TILE = the FLOOR-vs-resident contrast: grid_tile reduction is parallelized → claims ~1.

## STRUCTURE 4 — FULL_GRID resident (per_token_group) — the RESIDENT contrast to jsd's floor
- per_token_group_fp8 (128,4096,128) bf16: grid=(0,1), RED FULL_GRID bid2 sh=128 pinned=T reread=T fwo=T blt=3.
    bs_valid=[1] only (bid2 is pinned full-extent, no tunable slot; bid0,bid1 are grid). seed bs=[2] rl=[] w=1.
    → bid2 (group_size=128) seated RESIDENT (pinned full-grid). bid1 (groups_per_row) = grid sibling WIDENS to 2
      (occupancy: tok·(hidden/g)/g ≥ 1056 → g≤2). THE per_token_group 2× widen — grid sibling inherits remainder.

## STRUCTURE 5 — reduce-then-apply, non_reduction_loop (welford)
- welford (16384,768) fp32: grid=(0,), RED USER_TILE bid1 sh=768 blt=3 reread=T fwo=F. non_red_loop=(2,).
    ACC [0] scalars + [0,None]. bs=[8,1024,1024] w=4.  → r_block(bid1)=1024(? next_pow2(768)=1024), grid bid0=8,
      non-red-loop bid2 = 1024 (sized last, matches reduction tile, capped by resident budget).

## STRUCTURE 6 — multi-reduction SEQUENTIAL (rms_norm_per_block, rms_norm_dynamic_per_token)
- rms_norm_per_block_quant (128,4096,128) bf16: TWO groups g0(USER_TILE bid1 sh=4096), g1(FULL_SLICE bid3 sh=128).
    bs_valid=[1,2]. ACC [0](128), [0,None,None](128). seed bs=[4096,32] w=8.
    → bid1 (group g0) r_block=4096 full. bid3 is FULL_SLICE in SEPARATE group g1 (sequential, own budget). bid2
      (groups_per_row, an independent loop) sized=32 (own extent capped by resident budget). NOTE bid3 has no tunable slot here? bs_valid=[1,2] not 3 — bid3 pinned/materialized.
- rms_norm_dynamic_per_token (128,8192): g0(USER_TILE bid1 sh=8192), g1(USER_TILE bid2 sh=8192). bs=[8192,8192,8192].
    → both reductions in SEPARATE groups (sequential), each r_block=full extent. non-red loop bid3=8192.

## STRUCTURE 7 — grad-param M-COLLAPSE (bias_grad, dyt, rms/layer/instance/group _bwd) ← THE CRUX
Pattern: grid=(0,) [rows]. The grid-M parallelizes a CROSS-ROW reduction whose result is a grad-param
accumulator finalized ACROSS the grid (gb_blocks.sum(0)). Flooring M_BLOCK to 1 = the [1,1] catastrophe
(one partial per row → huge finalize). MUST collapse M_BLOCK toward ~num_sm (≈1 wave). CURRENT code uses the
_is_per_feature_accumulator / _grad_collapse_group RECOGNIZER (the thing being DELETED).
- bias_grad (2048,1024) fp32: RED USER_TILE bid1 sh=8192(reg block hint) blt=1(pure collapse). ACC [2]=feature(1024).
    grid=(0,) bid0=CTA tile. bs_valid=[0,1]. seed bs=[16,16] w=16. → m_collapse=next_pow2(grid_rows2048/num_sm132)=16; both bid0(CTA) & bid1(inner)=16.
- bias_grad (8192,4096): bs=[64,64]. (8192/132=62→64). (4096,8192): bs=[32,32].
- dyt_bwd (2048,1024): RED USER_TILE bid2 sh=8192 blt=3 reread=T. ACC [None,1]=feat,[1],[1]. bs=[16,8] w=16.
    → bid0(CTA)=16 (collapse), bid2(inner)=8 (BYTE-capped: blt=3 per-row work → M_COLLAPSE_TILE_BYTES=32768/(feat1024·4)=8).
- rms_norm_bwd (2048,4096) fp16: grid=(0,), RED USER_TILE bid1 sh=8192 blt=5; RED FULL_SLICE bid2 sh=4096 c2d=1 blt=7.
    SAME group g0 (co-resident: feature .sum + cross-row .sum). bs_valid=[0,1]. seed bs=[16,2] w=8.
    → bid0(grid CTA)=16 collapse; bid1(inner re-tile)=2 (byte cap). The FULL_SLICE feature (bid2) is materialized (no slot).
- layer_norm_bwd (2048,4096) fp16: RED USER_TILE bid2 sh=8192 blt=6; FULL_SLICE bid1 sh=4096 c2d=3 blt=7. bs=[16,2] w=8.
- group_norm_bwd (128,64,64,8) fp32: RED FULL_SLICE bid1 sh=64 c2d=5 blt=6; USER_TILE bid2 sh=8192; FULL_SLICE bid3 sh=8 c2d=3.
    rank-4 ACCs [2,3,3,1] etc. bs_valid=[0,2]. seed bs=[1,16] rl=[].  (bid0 grid=1?? collapse cap; bid2 inner=16)
- instance_norm_bwd (64,16,128): FULL_SLICE bid3 sh=128 blt=3; USER_TILE bid2 sh=8192. bs=[1,4].
- THE 2 KNOWN MOVERS: rms/layer_norm_bwd (4096,8192) fp16: [1,1]→[32,1] (collapse fix already in redesign).

### THE CRUX QUESTION (grid-M floor-vs-collapse, no recognizer)
rms_norm grid-M (STRUCT 1): independent rows → widening only LOSES parallelism → occupancy CAPS it (floor-ish).
bias_grad grid-M (STRUCT 7): parallelizes a cross-grid reduction (grad-param finalized via .sum(0)) → widening
  SAVES finalize work → push toward ~num_sm (1 wave), occupancy must NOT floor it.
Same axis, OPPOSITE occupancy treatment. The faithful distinguishing property: is there a loop-carried
accumulator FINALIZED ACROSS the grid-M axis (grad-param: dims are feature-only, finalized by .sum(0))?
That property IS what _is_per_feature_accumulator detects. Open: express it as a faithful budget/occupancy
input (continuous, uniform) NOT a recognized-shape branch.
