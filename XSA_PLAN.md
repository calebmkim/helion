# Plan: Exclusive Self-Attention (XSA) Helion kernel + TritonBench benchmark

## 0. Scope for the first pass

The paper defines XSA on causal self-attention, but this first benchmark should
intentionally match the current non-causal attention path. The implementation
scope is:

- non-causal attention
- forward only
- pre-projected `Q/K/V` inputs with shape `(B, H, T, D)`
- self-attention only, so `Q`, `K`, and `V` share sequence length `T`
- no output projection `W_o`

Mechanically:

```python
Y = scaled_dot_product_attention(Q, K, V, is_causal=False)
Vn = normalize(V, dim=-1, eps=eps)
Z = Y - (Y * Vn).sum(dim=-1, keepdim=True) * Vn
```

The fused Helion implementation still reads `V_i` once in the epilogue. The
win is that `Y` is never materialized to HBM and then read back by a separate
epilogue kernel.

---

## 1. Benchmarking source of truth

Use `benchmarks/tritonbench` as the source of truth for benchmarking behavior.
`examples/attention.py` is useful for Helion kernel structure, but the
benchmark operator, input generation, accuracy checks, and backend registration
should follow TritonBench patterns.

There are two integration layers:

1. The raw TritonBench operator lives under
   `benchmarks/tritonbench/tritonbench/operators/xsa` and should run on its own
   with `benchmarks/tritonbench/run.py --op xsa`. This validates input
   generation, the PyTorch baseline, and the `torch_compile_xsa` row.
2. The Helion comparison path is `benchmarks/run.py --kernel xsa`. That wrapper
   dynamically registers `examples.xsa.xsa_tritonbench` as a TritonBench backend
   using `KERNEL_MAPPINGS`; raw TritonBench will not discover Helion by itself.

The closest template is:

- `benchmarks/tritonbench/tritonbench/operators/flash_attention/operator.py`
- `benchmarks/tritonbench/tritonbench/operators/flash_attention/generate_inputs.py`

Important conventions to copy:

- Parse operator-specific args in `parse_op_args`.
- Use `DEFAULT_PRECISION = "bf16"`.
- Use TritonBench attention input generators rather than a hand-written shape
  list.
- Return list outputs from benchmark methods through a `multi_input_wrapper`.
  This matters because `generate_inputs.py` can emit multiple `Q/K/V` triples
  for one benchmark input when `--gen-cache-size-inputs` is used.
- Implement `get_num_inputs_per_iter` and `get_latency_scale` so multi-triple
  inputs report per-`Q/K/V` latency and CUDA graph configuration sees the right
  number of logical inputs.
- Override `accuracy` like attention does, so list outputs are compared against
  the baseline with attention-style tolerances and NaN checks.
- Use `@register_x_val(label="(Batch, Heads, SeqLen, SeqLen_KV, Dhead)")` to
  match attention dashboard shape labels.

---

## 2. Files to add or edit

| # | Path | What it does |
|---|---|---|
| 1 | `benchmarks/tritonbench/tritonbench/operators/xsa/__init__.py` | Package marker. |
| 2 | `benchmarks/tritonbench/tritonbench/operators/xsa/operator.py` | New forward-only TritonBench operator. |
| 3 | `examples/xsa.py` | Helion fused-XSA kernel plus `xsa_tritonbench` wrapper. |
| 4 | `benchmarks/run.py` | Add `KERNEL_MAPPINGS["xsa"]` and `KERNEL_METRIC_MAPPINGS["xsa"]`. |

No backward pass in this round.

---

## 3. TritonBench operator

The XSA operator should be a slim attention-style TritonBench operator with two
PyTorch rows:

1. `eager_xsa`: pure PyTorch eager matmul/softmax/matmul plus the XSA epilogue.
   This mirrors `flash_attention.operator.aten` with the extra XSA projection
   and is the `baseline=True` denominator.
2. `torch_compile_xsa`: `torch.compile` of the same XSA function. This is the
   generated-Triton comparison.

Do not add a handwritten Triton XSA row in the first pass. That keeps the new
operator close to TritonBench attention structurally without taking on a second
kernel implementation.

Skeleton:

```python
from __future__ import annotations

import argparse
from typing import Callable
from typing import Generator

import torch
import torch.nn.functional as F
from tritonbench.utils.triton_op import BenchmarkOperator
from tritonbench.utils.triton_op import register_benchmark
from tritonbench.utils.triton_op import register_x_val

from ..flash_attention.generate_inputs import additional_inputs
from ..flash_attention.generate_inputs import ragged_inputs
from ..flash_attention.generate_inputs import standard_inputs
from ..flash_attention.generate_inputs import sweep_inputs


def parse_op_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument("--n-heads", type=int, default=48, help="Number of heads")
    parser.add_argument("--d-head", type=int, default=64, help="Head dimension")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--input-types",
        type=str,
        default="STANDARD_SHAPES",
        choices=(
            "STANDARD_SHAPES",
            "RAGGED_SHAPES",
            "ADDITIONAL_SHAPES",
            "SWEEP_SHAPES",
        ),
    )
    parser.add_argument(
        "--gen-cache-size-inputs",
        action="store_true",
        help="Generate inputs as large as the GPU L2 cache size.",
    )
    return parser.parse_args(args)


def preproc_noop(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return args


def unpack_inputs(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.detach() for t in args)


def multi_input_wrapper(fn: Callable) -> Callable:
    def wrapper(
        self: Operator,
        *args: torch.Tensor,
    ) -> Callable[[], list[torch.Tensor]]:
        preproc_fn, benchmark_fn = fn(self, *args)
        assert len(args) % 3 == 0
        inputs = []
        all_inputs = []
        for i in range(0, len(args), 3):
            q, k, v = args[i : i + 3]
            inp = preproc_fn(q, k, v)
            all_inputs += [*unpack_inputs(*inp)]
            inputs.append(inp)

        def multi_input_fn() -> list[torch.Tensor]:
            return [benchmark_fn(*inp) for inp in inputs]

        self.optims[multi_input_fn] = torch.optim.SGD(all_inputs)
        return multi_input_fn

    wrapper.__name__ = fn.__name__
    return wrapper


def _xsa_epilogue(y: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    # Match F.normalize semantics: divide by max(norm, eps).
    y_float = y.float()
    vn = F.normalize(v.float(), dim=-1, eps=eps)
    z = y_float - (y_float * vn).sum(dim=-1, keepdim=True) * vn
    return z.to(y.dtype)


def _torch_eager_xsa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, sm_scale: float, eps: float
) -> torch.Tensor:
    assert q.shape[-2] == k.shape[-2] == v.shape[-2]
    p = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    y = torch.matmul(p, v)
    return _xsa_epilogue(y, v, eps)


class Operator(BenchmarkOperator):
    DEFAULT_PRECISION = "bf16"
    FWD_ONLY = True

    def __init__(
        self,
        tb_args: argparse.Namespace,
        extra_args: list[str] | None = None,
    ) -> None:
        super().__init__(tb_args, extra_args)
        args = parse_op_args(self.extra_args)
        self.BATCH = args.batch
        self.SEQ_LEN = args.seq_len
        self.H = args.n_heads
        self.D_HEAD = args.d_head
        self.eps = args.eps
        self.input_types = args.input_types
        self.gen_cache_size_inputs = args.gen_cache_size_inputs
        self.sm_scale = 1.0 / (self.D_HEAD**0.5)
        self.optims = {}

    @register_benchmark(baseline=True)
    @multi_input_wrapper
    def eager_xsa(self, *args: torch.Tensor) -> tuple[Callable, Callable]:
        def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return _torch_eager_xsa(q, k, v, self.sm_scale, self.eps)

        return preproc_noop, run

    @register_benchmark()
    @multi_input_wrapper
    def torch_compile_xsa(self, *args: torch.Tensor) -> tuple[Callable, Callable]:
        def xsa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return _torch_eager_xsa(q, k, v, self.sm_scale, self.eps)

        compiled = torch.compile(
            xsa,
            fullgraph=True,
            backend="inductor",
            mode="max-autotune",
        )

        def run(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return compiled(q, k, v)

        return preproc_noop, run

    def get_input_iter(self) -> Generator:
        shape = (self.BATCH, self.H, self.SEQ_LEN, self.SEQ_LEN, self.D_HEAD)
        if self.input_types == "RAGGED_SHAPES":
            return ragged_inputs(
                dtype=self.dtype,
                device=self.device,
                gen_cache_size_inputs=self.gen_cache_size_inputs,
            )
        if self.input_types == "STANDARD_SHAPES":
            return standard_inputs(
                shape=shape,
                num_inputs=self.tb_args.num_inputs,
                dtype=self.dtype,
                device=self.device,
                gen_cache_size_inputs=self.gen_cache_size_inputs,
            )
        if self.input_types == "ADDITIONAL_SHAPES":
            return additional_inputs(
                shape=shape,
                num_inputs=self.tb_args.num_inputs,
                dtype=self.dtype,
                device=self.device,
                add_production_shapes=self.add_production_shapes,
                name=self.name,
                shuffle_shapes=self.tb_args.shuffle_shapes,
                gen_cache_size_inputs=self.gen_cache_size_inputs,
            )
        if self.input_types == "SWEEP_SHAPES":
            return sweep_inputs(
                dtype=self.dtype,
                device=self.device,
                gen_cache_size_inputs=self.gen_cache_size_inputs,
            )
        raise AssertionError(f"Unknown input type {self.input_types}")

    @register_x_val(label="(Batch, Heads, SeqLen, SeqLen_KV, Dhead)")
    def get_x_val(self, example_inputs: tuple[torch.Tensor, ...]) -> tuple[int, ...]:
        q, k, _ = example_inputs[0:3]
        B, H, S, D = q.shape
        _, _, S_KV, _ = k.shape
        return (B, H, S, S_KV, D)

    def get_num_inputs_per_iter(
        self, example_inputs: tuple[torch.Tensor, ...]
    ) -> int:
        assert len(example_inputs) % 3 == 0
        return len(example_inputs) // 3

    def get_latency_scale(self, example_inputs: tuple[torch.Tensor, ...]) -> int:
        return self.get_num_inputs_per_iter(example_inputs)

    def accuracy(self, fn: Callable, baseline_fn: Callable) -> bool:
        output_list = fn()
        baseline_output_list = baseline_fn()

        if len(output_list) != len(baseline_output_list):
            return False

        for output, baseline_output in zip(
            output_list, baseline_output_list, strict=True
        ):
            if torch.isnan(output).any():
                return False

            if output.dtype in (torch.bfloat16, torch.float16):
                default_rtol = 1e-2
                default_atol = 2e-2
            else:
                default_rtol = 1e-5
                default_atol = 1e-8

            rtol = self.tb_args.rtol if self.tb_args.rtol is not None else default_rtol
            atol = self.tb_args.atol if self.tb_args.atol is not None else default_atol
            try:
                torch.testing.assert_close(
                    output,
                    baseline_output,
                    rtol=rtol,
                    atol=atol,
                )
            except Exception:
                return False

        return True
```

This keeps the XSA benchmark close to TritonBench attention while respecting the
decision to have only two PyTorch rows.

---

## 4. Helion kernel and TritonBench wrapper

Use `examples/attention.py` as the Helion kernel template, but not as the
benchmarking template. Start from its non-causal online-softmax loop and add the
XSA epilogue before storing.

Important implementation choices:

- Assert self-attention: `q`, `k`, and `v` share sequence length.
- Keep the attention loop equivalent to `examples/attention.py`.
- Match `F.normalize(v.float(), dim=-1, eps=eps)` semantics:
  `vn = v / max(norm(v), eps)`.
- Keep the epilogue in fp32 and cast only at the final store.

The `xsa_tritonbench` wrapper should return list outputs to match the
TritonBench attention operator's `multi_input_wrapper` behavior. This is
important for correctness when `--gen-cache-size-inputs` causes one
TritonBench input to contain multiple `Q/K/V` triples. This wrapper is not a
TritonBench-decorated method; `benchmarks/run.py` imports it and registers it
under the method name `helion_xsa_tritonbench`.

```python
def xsa_tritonbench(
    tb_op: object,
    *args: torch.Tensor,
) -> Callable[[], list[torch.Tensor]]:
    assert len(args) % 3 == 0
    eps = getattr(tb_op, "eps", 1e-6)

    def run() -> list[torch.Tensor]:
        outputs = []
        for i in range(0, len(args), 3):
            q, k, v = args[i : i + 3]
            outputs.append(xsa_kernel(q, k, v, eps))
        return outputs

    return run
```

The example `main()` should test `xsa_kernel` directly against a pure torch
eager tensor reference:

```python
def ref_xsa(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, eps: float
) -> torch.Tensor:
    sm_scale = 1.0 / math.sqrt(q.shape[-1])
    p = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    p = torch.softmax(p.float(), dim=-1).to(q.dtype)
    y = torch.matmul(p, v)
    vn = torch.nn.functional.normalize(v.float(), dim=-1, eps=eps)
    z = y.float() - (y.float() * vn).sum(-1, keepdim=True) * vn
    return z.to(y.dtype)
```

---

## 5. `benchmarks/run.py` entries

Add to `KERNEL_MAPPINGS`:

```python
"xsa": (
    "tritonbench.operators.xsa.operator",
    "examples.xsa",
    "xsa_tritonbench",
    {
        # Runtime cap for all-kernel/default runs. This mirrors the existing
        # Helion flash_attention cap; it is not a TritonBench operator default.
        "num_inputs": 6,
    },
),
```

Do not set `d_head` in `KERNEL_MAPPINGS`. The XSA operator's own
`parse_op_args` should default to `--d-head 64`, matching TritonBench
`flash_attention`.

Add to `KERNEL_METRIC_MAPPINGS`:

```python
"xsa": {
    "eager_xsa": "baseline",
    "torch_compile_xsa-speedup": "torch_compile_speedup",
    "torch_compile_xsa-accuracy": "torch_compile_accuracy",
    "torch_compile_xsa-latency": "torch_compile_latency_ms",
    "helion_xsa_tritonbench-speedup": "helion_speedup",
    "helion_xsa_tritonbench-accuracy": "helion_accuracy",
    "helion_xsa_tritonbench-latency": "helion_latency_ms",
},
```

---

## 6. Correctness-first workflow

Assume the `helion` conda environment is active for all commands:

```bash
conda activate helion
```

Start in quick mode: one small shape, no autotune, correctness only. This
checks Helion against the pure torch eager reference in `examples/xsa.py`.

```bash
HELION_AUTOTUNE_EFFORT=none python examples/xsa.py
```

Then check the raw TritonBench operator without Helion registration. This
validates the operator, input generator, baseline, and `torch.compile` row:

```bash
python benchmarks/tritonbench/run.py \
  --op xsa \
  --metrics accuracy \
  --batch 2 \
  --n-heads 4 \
  --seq-len 128 \
  --d-head 64 \
  --num-inputs 1
```

Then check the Helion wrapper through `benchmarks/run.py`, still with no
autotune:

```bash
HELION_AUTOTUNE_EFFORT=none python benchmarks/run.py \
  --kernel xsa \
  --metrics accuracy \
  --batch 2 \
  --n-heads 4 \
  --seq-len 128 \
  --d-head 64 \
  --num-inputs 1 \
  --output /tmp/xsa_quick.json
```

After that, sweep correctness across TritonBench's standard attention shapes
with no autotune:

```bash
HELION_AUTOTUNE_EFFORT=none python benchmarks/run.py \
  --kernel xsa \
  --metrics accuracy \
  --input-types STANDARD_SHAPES \
  --num-inputs 5 \
  --output /tmp/xsa_correctness_sweep.json
```

Only after those pass should we run latency and speedup numbers with autotuning
enabled:

```bash
python benchmarks/run.py \
  --kernel xsa \
  --metrics speedup,accuracy,latency \
  --input-sample-mode equally-spaced-k \
  --num-inputs 3 \
  --output /tmp/xsa.json
```

Do not keep `HELION_AUTOTUNE_EFFORT=none` set for final performance numbers.

---

## 7. Validation notes

- The first version is non-causal by design. A causal variant should be a
  separate follow-up because it needs both tile skipping and diagonal masking.
- Accuracy comparisons should use the same `eps`, fp32 epilogue, and
  `F.normalize` semantics in both PyTorch and Helion.
- Keep first-pass input sets self-attention-only. If a future production-shape
  loader can emit `S != S_KV`, filter those shapes or fail clearly instead of
  benchmarking cross-attention under the XSA name.
- Test a near-zero `V_i` row manually before trusting the sweep. The output
  should remain finite and should match the `F.normalize(..., eps=eps)`
  reference.
- The default shape behavior now comes from TritonBench attention:
  `STANDARD_SHAPES` starts at sequence length `2**7` and grows by powers of
  two. If `--seq-len` is fixed and `--num-inputs N` is provided, the generator
  emits `N` shapes by repeatedly doubling the sequence length, just like
  `flash_attention`.

---

## 8. Later follow-ups

- Add causal XSA once the non-causal benchmark is stable.
- Consider adding an SDPA-plus-epilogue row only if we specifically want a
  practical fused-attention PyTorch comparison. For the first pass, the
  baseline is the manual torch eager row that mirrors TritonBench attention's
  `aten` baseline.
- Consider measuring output projection separately only if the benchmark goal
  changes from "attention plus XSA epilogue" to a fuller transformer block
  fragment.
