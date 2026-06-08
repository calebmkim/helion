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

## tritonbench edit wiring (VERIFIED by measurement-harness-verifier, 2026-05-28)
**The two editables resolve DIFFERENTLY — proven with sentinels:**
- `helion` editable = a plain `.pth` (`_editable_impl_helion.pth`) adding the ORIGINAL checkout to
  `sys.path`. PYTHONPATH=worktree comes earlier in `sys.path`, so **`import helion` AND
  `import examples.*` resolve to the WORKTREE.** Sentinel in `examples/rms_norm.py:rms_norm_tritonbench`
  fired from the worktree copy. So the **helion-side wrapper + the kernel under test ARE editable via
  PYTHONPATH** — no special handling.
- `tritonbench` editable = a `MetaPathFinder` (`__editable___tritonbench_0_0_1_finder.py`, appended to
  `sys.meta_path`) whose `MAPPING` is HARDCODED to
  `/home/calebkim/helion-new-heuristics/local/helion/benchmarks/tritonbench/tritonbench`. The worktree has
  **no `benchmarks/tritonbench/` dir**, so PYTHONPATH cannot shadow it. Sentinel in the tritonbench
  rms_norm `operator.py` fired from the **ORIGINAL checkout**, confirming the gotcha.
- **Net for the rms_norm benchmark path:** the Helion kernel under test runs through the WORKTREE's
  `examples.rms_norm` (good — that's our code); the tritonbench operator (input gen, eager/torch.compile
  baselines, accuracy) runs from the ORIGINAL checkout.

**How to make a tritonbench-operator edit take effect (NO `pip install`), in priority order:**
1. **PRIMARY (proven, simplest):** edit the operator IN the original checkout
   `/home/calebkim/helion-new-heuristics/local/helion/benchmarks/tritonbench/.../operators/<op>/operator.py`.
   Pure-Python, picked up immediately (no rebuild). This is where the editable resolves, so the edit runs.
   Downside: lives outside the worktree's git — fine for third-party benchmark-harness code (e.g. adding
   the `torch_compile_<op>_default` Product-A baseline variant). **Verify with a one-line sentinel print
   before trusting any operator edit**, then remove it.
2. **ALT (in-worktree, version-controlled):** install a targeted `sys.meta_path` finder (insert at index
   0, before both `PathFinder` and the editable `_EditableFinder`) that redirects only
   `tritonbench.operators.<op>.operator` to a worktree overlay file. Tested working — BUT the overlay must
   be a COMPLETE operator module (its `__init__` does `from .operator import Operator`), so the overlay
   should `from <original module> import *` then add the new variant. More moving parts; use only if the
   edit must be in worktree git.
   - A FULL `tritonbench/` package on PYTHONPATH does win over the editable finder (PathFinder@idx3 beats
     _EditableFinder@idx4), but that shadows ALL of tritonbench (components/kernels/data) — do NOT do a
     partial full-package overlay. And do NOT vendor under `benchmarks/tritonbench/`: `run.py`'s
     `check_and_setup_tritonbench()` treats that path as a "local checkout" and may `git clone` over it.

## Bare-seed measurement (VERIFIED; canonical scripts in `_lab/harness/`)
- `_lab/harness/bare_seed_run.py` — `run_bare_seed(fn, build_args, reference, shape, seed_dict, ...)`:
  asserts worktree helion; runs `helion.kernel(fn.fn, configs=[seed])` (len==1 short-circuit -> NO
  autotune); inspects normalized `bound._config`; confirms seed used (persistent-vs-looped + num_warps in
  generated Triton); correctness vs reference (fp32 tol params, default rtol=1e-3/atol=1e-4); median-of-N
  `triton.testing.do_bench` + spread. dtype is a param (default fp32, NOT hardcoded). CLI demo = rms_norm
  (2048,4096) fp32.
- `_lab/harness/evidence_block.py` — `format_evidence_block(EvidenceFields(...))` emits the FIXED receipt
  every agent reuses.
- Mechanism facts proven: (a) no CSV at `HELION_AUTOTUNE_LOG` => no real search; (b) `configs=[seed]` +
  eager `config_spec.normalize(seed)` makes a structurally-invalid seed RAISE (e.g. `num_warps=7` ->
  `AssertionError: num_warps must be a power of 2`) instead of being silently dropped; (c) distinct config
  -> distinct Triton (persistent `reduction_loops=[None]` has NO `for roffset` loop; looped
  `reduction_loops=[512]` HAS one; num_warps reflected in the launcher call). Note: `config_spec.normalize`
  does NOT collapse `reduction_loops` value>=size_hint to `None` at the spec level — the
  persistent-vs-looped choice is realized at CODEGEN, so always inspect the generated Triton, not just the
  normalized dict.

## Measurement
- Trust TritonBench's own latency (do_bench, ~+/-2%) over naive CUDA-event timing (~+/-12% at sub-0.1ms).
  The results-referee measures via TritonBench, not hand timing.
