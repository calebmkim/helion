# tritonbench operator patches (live OUTSIDE the worktree git)

These patches modify the **tritonbench** operators. Per `_lab/SETUP.md`
("tritonbench edit wiring (VERIFIED)"), the tritonbench editable resolves via a
hardcoded `MetaPathFinder` to the **ORIGINAL checkout**, NOT the worktree:

    /home/calebkim/helion-new-heuristics/local/helion/benchmarks/tritonbench/tritonbench/operators/<op>/operator.py

So `PYTHONPATH=<worktree>` does NOT shadow these files; the live edit must be made
directly in the original checkout. That directory is its own nested git repo
(git-ignored by the parent helion repo), so the edit is invisible to the worktree's
git. We therefore save the diff here, in the worktree, so the change is reproducible
and version-controlled even though the file it touches is not.

## torch_compile_rms_norm_default.patch

Adds a `torch_compile_rms_norm_default` benchmark variant to the rms_norm operator:
`torch.compile(module)` with **DEFAULT mode** (no `mode=` kwarg), i.e. NOT
`max-autotune-no-cudagraphs`. This is Product A's baseline-to-beat (torch.compile
DEFAULT). The operator previously only shipped `torch_compile_rms`
(`mode="max-autotune-no-cudagraphs"`).

Verified (2026-05-28): a one-line stderr sentinel in this operator fired from the
ORIGINAL checkout copy, confirming the edit is the file that actually runs; the new
`torch_compile_rms_norm_default-{latency,accuracy,speedup}` columns appear in
`benchmarks/run.py --kernel rms_norm` output, accuracy=1.

### To (re-)apply after a fresh tritonbench checkout

    cd /home/calebkim/helion-new-heuristics/local/helion/benchmarks/tritonbench
    git apply /home/calebkim/helion-new-heuristics/wt-reduction/_lab/harness/patches/torch_compile_rms_norm_default.patch

(Paths in the patch are repo-relative to that nested tritonbench repo.)
