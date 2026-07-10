# PR #2866 — pointwise seed heuristic: adversarial lab bundle

This branch (`pointwise-2866-lab`) is the exact PR #2866 commit
(`[autotuner] pointwise seed heuristic for elementwise + partially-tiled kernels`)
plus the adversarial-hunt lab that produced it. `_lab/` is normally git-ignored;
it is force-added here so another machine can `git checkout` and inspect/rerun it.

## Layout

- `_lab/adversarial/` — the write-up
  - `REPORT.md`, `NOTEBOOK.md` — findings + running log of the gap hunt
  - `shapes_adversarial.py` — the CURRICULUM dict (name → shard, kernel class, real anchor, splits)
  - `ledger.json`, `cfg_recorder.py` — result ledger + config-recording helper
- `_lab/rope_probe/adv/` — the runnable hunt (copied from `local/rope_probe/adv/`)
  - `shards/shard_*.json` — **self-contained**: each entry has the kernel `code` **and** its `shapes`
  - `shards/_gen/*.py` — the kernel sources extracted from the shards
  - `harness_one.py` — trusted per-process measurement (seed config vs compiler default, acc-gated, cold-L2)
  - `oracle_vs_tc.py`, `results*.jsonl` — oracle-vs-torch.compile probe + raw results

## The two kernels called out

| kernel | source | shapes (shard) |
|---|---|---|
| `transposed_out_add` | `_lab/rope_probe/adv/shards/_gen/advk_0_transposed_out_add.py` | `shards/shard_transpose.json` |
| `heavy_transcendental_1d` | `_lab/rope_probe/adv/shards/_gen/advk_0_heavy_transcendental_1d.py` | `shards/shard_compute.json` |

## Path caveat (if you want to RUN it, not just read it)

`rope_probe/adv/harness_one.py` hardcodes
`_WT = "/home/calebkim/helion-new-heuristics/helion-pointwise"` to force-import the
PR #2866 Helion checkout. On another box, point that at your checkout of this branch
(or any tree with the #2866 pointwise seed) before running. The shard JSONs and
`_gen/*.py` kernels are path-independent.
