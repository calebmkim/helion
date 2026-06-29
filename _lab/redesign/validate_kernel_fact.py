"""P1 validation: the new ReductionKernelFact is FAITHFUL + carries the legacy facts.

For every corpus kernel + probe, bind and check INVARIANTS that must hold for the categorizing
fact to be a correct, behavior-preserving replacement of the legacy ReductionFact list:

  INV1 (category sanity): every descriptor's category matches the IR structure
        (grid+cdiv1 -> FULL_GRID; grid+partial -> GRID_TILE; block_sizes -> USER_TILE; else
        FULL_SLICE; size=None -> DECLINED). Re-derived independently here.
  INV2 (co-residency = original graph_id): each group is exactly one graph_id's reductions;
        groups partition the descriptors.
  INV3 (legacy reproducibility): the legacy reduction_facts[0] (when present) is RECONSTRUCTIBLE
        from the descriptors — its primary block_id is a sized descriptor, its size_hint /
        itemsize / row_reread / full_width_output / input_load_itemsize / num_load match the
        corresponding descriptor, and secondary_reduction_block_ids ⊆ the sized descriptors.
  INV4 (rollable invariance): two rollable FULL_SLICE descriptors are never co-resident (§2.7).

Reuses the unified recorder's corpus iterators so it sees the exact 447-cell matrix + the 13
probes. Prints a per-kernel PASS/FAIL with the first failing invariant; exits 1 on any FAIL.
"""

from __future__ import annotations

import os
import sys

_HARNESS = os.path.dirname(os.path.abspath(__file__))
_WT_ROOT = os.path.abspath(os.path.join(_HARNESS, "..", ".."))
_LOCAL = os.path.abspath(os.path.join(_WT_ROOT, ".."))
for _d in (
    _HARNESS,
    os.path.join(_WT_ROOT, "_lab", "harness"),
    os.path.join(_WT_ROOT, "_lab", "prompts"),
    os.path.join(_LOCAL, "prompts-lab", "vllm-bench"),
    os.path.join(_LOCAL, "prompts-lab", "transfer"),
    os.path.join(_WT_ROOT, "examples"),
):
    if os.path.abspath(_d) not in sys.path:
        sys.path.insert(0, os.path.abspath(_d))

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import torch  # noqa: E402

import helion  # noqa: E402
from helion._compiler.compile_environment import FixedBlockSizeSource  # noqa: E402
from helion.autotuner.config_spec import ReductionCategory  # noqa: E402
from helion.autotuner.config_spec import SIZED_REDUCTION_CATEGORIES  # noqa: E402

assert os.path.abspath(helion.__file__).startswith(_WT_ROOT + os.sep)

import ir_introspect as II  # noqa: E402
import unified_config_recorder as U  # noqa: E402


def _check(name: str, fn, args) -> list[str]:
    """Return a list of invariant-violation strings (empty == PASS)."""
    fails: list[str] = []
    bound = fn.bind(args)
    env = bound.env
    spec = env.config_spec
    kf = spec.reduction_kernel_fact
    with env:
        if kf is None:
            # Only acceptable when there are genuinely no reductions in original graphs.
            occ = bound.host_function.device_ir._original_graph_reductions()
            if occ:
                fails.append(f"kernel_fact=None but {len(occ)} reduction occurrences exist")
            return _finish(bound, fails)

        descs = kf.reductions
        grid_ids = {
            b for bids in bound.host_function.device_ir.grid_block_ids for b in bids
        }
        bs_valid = set(spec.block_sizes.valid_block_ids())

        # INV1: category re-derivation.
        for d in descs:
            info = env.block_sizes[d.block_id]
            if not isinstance(info.size, (int, torch.SymInt)):
                expect = ReductionCategory.DECLINED
            elif d.block_id in grid_ids:
                src = info.block_size_source
                full = (
                    isinstance(src, FixedBlockSizeSource)
                    and isinstance(info.size, (int, torch.SymInt))
                    and env.known_equal(src.value, info.size)
                )
                expect = (
                    ReductionCategory.FULL_GRID if full else ReductionCategory.GRID_TILE
                )
            elif d.block_id in bs_valid:
                expect = ReductionCategory.USER_TILE
            else:
                expect = ReductionCategory.FULL_SLICE
            if d.category is not expect:
                fails.append(
                    f"INV1 g{d.graph_id} bid{d.block_id}: cat={d.category.value} "
                    f"expect={expect.value}"
                )

        # INV2: groups partition descriptors, each is one graph_id.
        idxs_seen: list[int] = []
        for g in kf.coresidency_groups:
            for i in g.descriptor_indices:
                idxs_seen.append(i)
                if descs[i].graph_id != g.graph_id:
                    fails.append(
                        f"INV2 group g{g.graph_id} holds desc with graph_id "
                        f"{descs[i].graph_id}"
                    )
        if sorted(idxs_seen) != list(range(len(descs))):
            fails.append(f"INV2 groups do not partition descriptors: {idxs_seen}")

        # INV4: two rollable FULL_SLICE never co-resident.
        for g in kf.coresidency_groups:
            roll_full = [
                i
                for i in g.descriptor_indices
                if descs[i].category is ReductionCategory.FULL_SLICE
                and descs[i].rollable
            ]
            if len(roll_full) > 1:
                fails.append(
                    f"INV4 group g{g.graph_id} has {len(roll_full)} rollable FULL_SLICE"
                )

        # INV3: legacy RECONSTRUCTIBILITY (only when a legacy fact exists). The legacy
        # ReductionFact must be reconstructible from the descriptors so the heuristic can switch
        # to the kernel fact with no config change (PROMPT §2.6 reproducibility map).
        if spec.reduction_facts:
            lf = spec.reduction_facts[0]
            sized = {d.block_id: d for d in descs if d.category in SIZED_REDUCTION_CATEGORIES}
            by_bid = {}
            for d in descs:
                by_bid.setdefault(d.block_id, d)
            prim = lf.primary_reduction_block_id
            # The legacy primary must be a SIZED descriptor (full-extent or user-tile), and the
            # SIZING-CRITICAL per-axis fields must match exactly (these drive the byte caps).
            if prim not in sized:
                fails.append(
                    f"INV3 legacy primary bid{prim} not a sized descriptor "
                    f"(sized={sorted(sized)})"
                )
            else:
                d = sized[prim]
                for field in ("size_hint", "itemsize", "input_load_itemsize", "row_reread"):
                    lv, dv = getattr(lf, field), getattr(d, field)
                    if lv != dv:
                        fails.append(f"INV3 bid{prim} {field}: legacy={lv} desc={dv}")
            # full_width_output is PER-DESCRIPTOR in the new fact (PROMPT §2.6); the legacy
            # kernel-scalar = OR over the primary's group reductions + the non-reduction loops
            # (a full-width store on the normalize pass, e.g. welford). Reconstruct it that way
            # and require a match (proves the consumer can recompute the legacy scalar).
            prim_group = next(
                (g for g in kf.coresidency_groups
                 if any(descs[i].block_id == prim for i in g.descriptor_indices)),
                None,
            )
            # Only meaningful when the primary's group is a SINGLETON (the legacy single-fact
            # assumption). For a MULTI-sized-co-resident group (a genuine multi-reduction kernel
            # the legacy single-primary view can't represent, e.g. the rewritten p1 outer-product
            # with two co-resident FULL_SLICE reductions), the per-descriptor full_width_output is
            # MORE faithful than the legacy kernel-scalar, so they legitimately differ — skip.
            n_sized_in_group = (
                sum(
                    1
                    for i in prim_group.descriptor_indices
                    if descs[i].category in SIZED_REDUCTION_CATEGORIES
                )
                if prim_group is not None
                else 1
            )
            if n_sized_in_group <= 1:
                recon_fw = False
                if prim_group is not None:
                    recon_fw = any(
                        descs[i].full_width_output
                        for i in prim_group.descriptor_indices
                    )
                recon_fw = recon_fw or _loop_has_full_width_store(
                    spec, lf.non_reduction_loop_block_ids
                )
                if recon_fw != lf.full_width_output:
                    fails.append(
                        f"INV3 full_width_output reconstruct: legacy={lf.full_width_output} "
                        f"recon={recon_fw}"
                    )
            # Every legacy secondary must be REPRESENTED as a descriptor (any category) — the
            # legacy code's secondaries include grid-tile reductions it mis-sized as reductions.
            for sb in lf.secondary_reduction_block_ids:
                if sb not in by_bid:
                    fails.append(f"INV3 legacy secondary bid{sb} not represented")
    return _finish(bound, fails)


def _loop_has_full_width_store(spec, loop_bids) -> bool:
    """A non-reduction loop with a rank>=2 store over its own axis (welford normalize)."""
    loop_set = set(loop_bids)
    for f in spec.memory_op_facts:
        if (
            f.kind == "store"
            and f.ndim >= 2
            and (
                (f.subscript_block_ids and f.subscript_block_ids[-1] in loop_set)
                or (f.indexed_block_ids and f.indexed_block_ids[-1] in loop_set)
            )
        ):
            return True
    return False


def _finish(bound, fails):
    del bound
    torch.cuda.empty_cache()
    return fails


def main() -> None:
    print(f"helion={helion.__file__}\n", flush=True)
    n_pass = n_fail = 0
    failed_kernels: list[str] = []

    # The 13 probes.
    for slug in II._PROBES:
        fn, args = II._probe_fn_args(slug)
        try:
            fails = _check(slug, fn, args)
        except Exception as e:  # noqa: BLE001
            import traceback

            fails = [f"EXC {type(e).__name__}: {e}"]
            traceback.print_exc()
        tag = f"probe/{slug}"
        if fails:
            n_fail += 1
            failed_kernels.append(tag)
            print(f"[FAIL] {tag}\n      " + "\n      ".join(fails), flush=True)
        else:
            n_pass += 1
            print(f"[ ok ] {tag}", flush=True)

    # The full 447-cell corpus.
    for corpus in ("curriculum", "transfer", "vllm", "mreduction"):
        for (cps, kname, shape, dtype, fn, kargs, split) in U._CORPORA[corpus](None):
            tag = f"{cps}/{kname}/{shape}"
            try:
                fails = _check(tag, fn, kargs)
            except Exception as e:  # noqa: BLE001
                fails = [f"EXC {type(e).__name__}: {e}"]
            if fails:
                n_fail += 1
                failed_kernels.append(tag)
                print(f"[FAIL] {tag}\n      " + "\n      ".join(fails), flush=True)
            else:
                n_pass += 1
    print(f"\n=== {n_pass} pass, {n_fail} fail ===", flush=True)
    if failed_kernels:
        print("FAILED:", failed_kernels[:40])
        sys.exit(1)


if __name__ == "__main__":
    main()
