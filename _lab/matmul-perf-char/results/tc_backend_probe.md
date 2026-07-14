# tc_max_autotune actual backend per shape (CUDA profiler-verified, 2026-07-14)

Determined by profiling the compiled tc callable and reading the dominant GPU kernel name
(nvjet_sm100_* = cuBLAS/cuBLASLt Blackwell; triton_* = Inductor Triton template). This CORRECTS
the harness's hardcoded `winner` label (which assumed cuBLAS for all GEMMs).

| kernel | shape | actual tc backend |
|---|---|---|
| matmul | [1,4096,4096] (decode M=1) | **TRITON** (triton_red_fused_mm — a GEMV, NOT cuBLAS) |
| matmul | [32,8192,8192] (decode M=32) | cuBLAS (nvjet_sm100_tst) |
| matmul | all other 20 cells | cuBLAS (nvjet_sm100_*) |
| fp8_gemm | ALL 10 cells (incl. M=1) | cuBLAS/cuBLASLt (nvjet_sm100_qqhsh — _scaled_mm) |
| bmm | ALL 8 cells (incl. tiny) | cuBLAS batched (nvjet_sm100_tst) |
| mamba2_chunk_state | all 8 | Triton (torch.compile of naive einsum ref — NO cuBLAS analog; by design) |

**Net correction:** exactly ONE cell — matmul[1,4096,4096] — has G_vs_tc measured against Triton,
not cuBLAS. Its G=1.40 "seed faster" is a win over a Triton GEMV, NOT over cuBLAS. Every other GEMM
cell's tc arm is genuinely cuBLAS/cuBLASLt (verified nvjet). The adversarial lens over-claimed that
M=32 and small fp8/bmm cells were also Triton; profiling shows they are cuBLAS.
