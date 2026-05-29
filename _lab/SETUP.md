# Canonical Working Setup (verified by hub, 2026-05-28)

This is the **single source of truth** for how to run anything in this worktree.
All agents must use exactly this. If something here turns out wrong, fix it here.

## Paths
- **Worktree (YOUR code; edits here are what run):** `/home/calebkim/helion-new-heuristics/wt-reduction`
- **Original checkout (do NOT edit; the editable install points here):** `/home/calebkim/helion-new-heuristics/local/helion`
- Branch: `reduction-heuristics-autotuner` (git worktree off `reduction-heuristic-plan` HEAD `1ec3193d`).
- Lab dir: `_lab/` (this dir). Logs: `logs/`. Harness scripts: `_lab/harness/`.

## Interpreter
- `/home/calebkim/.conda/envs/helion/bin/python` (conda env `helion`: torch 2.12.0.dev cu128, triton 3.7.0, cuda 12.8).
- System `python`/`python3` LACK the deps — never use them.

## Import wiring (VERIFIED)
- `helion` is an editable *plain-path* install pointing at the ORIGINAL checkout. To run YOUR
  worktree code, set `PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction` AND run from a
  cwd that is NOT the original checkout root.
- **Always assert** at the top of any script: `assert helion.__file__.startswith("/home/calebkim/helion-new-heuristics/wt-reduction")`.
- Verified: with PYTHONPATH set (cwd=/tmp or the worktree root), `helion.__file__` =
  `.../wt-reduction/helion/__init__.py`. Without it, it resolves to the original checkout.
- Codegen edits flow through: a sentinel injected into `helion/runtime/kernel.py` `to_code` changed
  `bound.to_triton_code(cfg)` output (sha 9f3e2398 -> 18ed86ed). Reverted; worktree clean.
- **tritonbench uses a DIFFERENT import-hook editable** (`benchmarks/run.py` is at the worktree repo
  ROOT, i.e. `.../wt-reduction/benchmarks/run.py`). PYTHONPATH alone may NOT shadow it — VERIFY with a
  sentinel before trusting any tritonbench operator edit; point at the worktree's tritonbench / adjust
  its finder. NO `pip install`. (measurement-harness-verifier owns proving this.)

## GPU (shared machine)
- 4x H100 (sm_90). GPU 0 often has co-tenants. As of 2026-05-28 GPUs 1,2,3 idle.
- **Pin `CUDA_VISIBLE_DEVICES=2`** for this run (re-check `nvidia-smi` before any trusted measurement;
  rule out co-tenants before believing a delta).

## Canonical invocation (codegen / bare-seed dev)
```
cd /home/calebkim/helion-new-heuristics/wt-reduction && \
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/calebkim/helion-new-heuristics/wt-reduction \
/home/calebkim/.conda/envs/helion/bin/python <script.py>
```

## Bare-seed mechanism (Product A; from the Technical Appendix)
- `kern = helion.kernel(fn, config=seed); kern(*args)` runs `seed` with NO autotune (the
  `len(configs)==1` short-circuit). Or `configs=[seed]` so a bad seed RAISES instead of being
  silently dropped.
- `bound = fn.bind(args); cfg = bound.config_spec.default_config()` gives a starting Config.
- `Config` is an immutable Mapping; build variants: `helion.Config(**{**dict(cfg), "num_warps":8, ...})`.
- `bound.to_triton_code(cfg)` shows codegen (distinct config -> distinct Triton).
- Caveats: configs=[seed] BYPASSES the autotuner accuracy check -> run your OWN correctness check;
  `normalize()` may mutate the seed (forces persistent when value>=size_hint, caps by size_hint) ->
  inspect/benchmark the NORMALIZED `bound_kernel._config`, not the raw seed.

## Precision
- **fp32 fixed for the whole hill-climb.** Assert dtype on every benchmark call (softmax defaults to
  fp16 — override). The heuristic reads dtype/itemsize as a ReductionFact field, never hardcodes fp32.

## rms_norm reference (verified)
- `from examples.rms_norm import rms_norm_fwd` (the `@helion.kernel` fn). Args: x [M,N] fp32, weight [N] fp32, eps.
- default_config at (2048,4096): `reduction_loops=[None]` (persistent), `block_sizes=[1]`, num_warps=4, num_stages=1.
- `rms_norm_pytorch(x, weight, eps)` is the fp32 reference in examples/rms_norm.py.

## Measurement
- Trust TritonBench's own latency (do_bench, ~+/-2%) over naive CUDA-event timing (~+/-12% at sub-0.1ms).
  The results-referee measures via TritonBench, not hand timing.
