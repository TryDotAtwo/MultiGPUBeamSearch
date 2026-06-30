# Stream1 Piece Transformer Optimization 2026-06-30

## Scope

- Branch: `codex/stream1-piece-transformer`.
- Backend touched: Stream1 `piece_transformer` only.
- MLP path: no semantic changes intended; common `cuda/stream1.cu` still builds old MLP CUDA tests.
- Profiling target: local Docker image `gpu-dev-cutlass-nsight:2026-05-24` on RTX 3070 Laptop GPU, SM86 build.

## Changes

- Avoided per-thread `State128` value copies in transformer input token construction; the kernel now reads state bytes through a pointer.
- Added benchmark filters:
  - `BEAM_STREAM1_TRANSFORMER_B_MICRO`
  - `BEAM_STREAM1_TRANSFORMER_CONCURRENCY`
- Added Stream1 transformer LayerNorm fast path for `cols == 256`, using warp reductions and 8-float shared scratch instead of the generic 256-float reduction buffer.
- Added exact-dimension attention fast path for the current p900 transformer shape: `seq_len=51`, `d_model=256`, `nhead=8`, `head_dim=32`.

## Profiling Decisions

- `attention_threads=128` was tested and rejected: `344481.1` candidates/s at `B_MICRO=1024, concurrency=1`, worse than `256` threads.
- K/V shared-memory staging was tested and rejected as default: non-Nsight run had small noisy gain, but Nsight showed the staged attention kernel itself was slightly slower than exact non-staged attention.
- Final default keeps `attention_threads=256` and exact non-staged attention.

## Benchmark Results

Baseline from earlier local profiling at `B_MICRO=1024, concurrency=1`: about `304663.2` candidates/s.

Final filtered `1024x1`:

- `ms_per_launch_group=65.3703`
- `parents_per_sec=15664.6`
- `candidates_per_sec=375950.6`
- report: `test_results/stream1_transformer_final_1024x1_2026-06-30.md`

Final full sweep best:

- `B_MICRO=2048`, `concurrency=1`
- `ms_per_launch_group=120.1118`
- `parents_per_sec=17050.8`
- `candidates_per_sec=409218.8`
- `scratch_bytes=481394688`
- report: `test_results/stream1_transformer_final_sweep_2026-06-30.md`

## Nsight Final 2048x1 Kernel Split

Profile command generated:

- `test_results/nsys_stream1_transformer_final_2048x1_2026-06-30.nsys-rep`
- `test_results/nsys_stream1_transformer_final_2048x1_2026-06-30.sqlite`

GPU kernel time summary:

- `stream1_transformer_attention51_exact_kernel`: `53.0%`
- CUTLASS GEMM kernels: `29.5%`
- `stream1_transformer_layernorm256_copy_kernel`: `7.5%`
- `stream1_transformer_bias_silu_kernel`: `6.1%`
- `stream1_transformer_residual_bias_add_kernel`: `3.1%`
- `stream1_transformer_build_input_kernel`: `0.7%`

## Verification

Passed:

```text
cmake --build build-stream1-opt --target stream1_transformer_cuda_tests stream_benchmark -j2
./build-stream1-opt/stream1_transformer_cuda_tests
```

Passed full CUDA test executables after building the full tree except `contract_tests`:

```text
stream1_cuda_tests passed
stream1_transformer_cuda_tests passed
stream2_cuda_tests passed
stream3_cuda_tests passed
stream4_cuda_tests passed
stream5_cuda_tests passed
final_cuda_tests passed
threshold_cuda_tests passed
stitched_cuda_tests passed
dispatcher_cuda_tests passed
static_memory_cuda_tests passed
history_tests passed
```

Passed filtered CTest after excluding the fixture-dependent contract test:

```text
ctest --test-dir build-stream1-opt -E contract_tests --output-on-failure
100% tests passed, 0 tests failed out of 12
```

`ctest --test-dir build-stream1-opt --output-on-failure` had one environment/data failure:

```text
contract_tests=fail error=cannot open required text file: stream1_weights/manifest.json
```

This is a missing local test fixture, not a CUDA assertion failure.