# Stream1 Transformer SM80 Tensor-Core Dispatch and Fused LN Check

Date: 2026-07-08

Scope:

- Optimize only the native CUDA/CUTLASS `piece_transformer` Stream1 path.
- Keep MLP/default path separate.
- Preserve explicit `BEAM_STREAM1_TRANSFORMER_BLOCK51=1` and `BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1` behavior.
- No fallback, no distillation, no architecture change outside Stream1 transformer kernels.

Changes:

- Added rounded fused `bias + LayerNorm` for the exact p900 `block51` path where it replaces `bias_add -> layernorm_copy` while preserving the old half/bfloat rounding point.
- Marked hot scalar helpers `__forceinline__` after a first local run showed the new rounding helper could compile into a severe device-call performance regression.
- Switched fp16 transformer GEMM wrappers to runtime architecture dispatch:
  - SM80+ uses CUTLASS `arch::Sm80` with `GemmShape<16,8,16>` tensor ops.
  - SM75/T4 stays on CUTLASS `arch::Sm75` with `GemmShape<16,8,8>`.
- Updated `stream_benchmark` report text to describe runtime SM80/SM75 tensor-op dispatch.

Local Docker build:

```text
cmake --build build-fused-ln-local --target stream_benchmark stream1_transformer_cuda_tests production_runner -j2
```

Result:

```text
[5/5] Linking CXX executable production_runner
```

Full verification:

```text
ctest --test-dir build-fused-ln-local --output-on-failure
```

Result:

```text
100% tests passed, 0 tests failed out of 13
```

Performance evidence on local SM86 GPU, real `weights/megaminx_vlad_transformer_fp16`, graph replay, `b_micro=512`, `concurrency=2`:

| Build | ms/group | candidates/sec | checksum | digest |
|---|---:|---:|---:|---:|
| old binary from `build-final-cls-check` | 27.8192 | 883419.4 | 1242545152 | 3860563098260702083 |
| fused rounded LN before forceinline | 471.8097 | 52088.8 | 1242545152 | 3860563098260702083 |
| fused rounded LN after forceinline | 27.5348 | 892541.7 | 1242545152 | 3860563098260702083 |
| fused rounded LN + fp16 SM80 dispatch | 25.7323 | 955065.5 | 1242545152 | 3860563098260702083 |

The bad intermediate run proves the first fused kernel was mathematically correct but performance-broken by the non-inlined device rounding helper. The final run preserves the same score digest and improves local throughput versus the old binary by about 8.1%.

Pipeline smoke:

```text
stream_pipeline_benchmark mode=stream123 window=32 b_micro=512 concurrency=2 ring_slots=8 stream3_batch=98304 graph_window_jobs=64 physical_jobs=256 frontier_size=131072 ring_slot_jobs=256 stream3_jobs=32 stream4_jobs=0 candidates=3145728 depth_like_ms=3801.95 candidates_per_sec=827399 shard_capacity=1048576 allocation_bytes=3281187584 status=OK
```

Notes:

- This should help A100/H100 more directly than T4 because previous fp16 transformer GEMMs were forced through the SM75 instruction shape even on SM80 GPUs.
- The exact A100 cluster delta still needs a short isolated Stream1 or depth-10 job after pushing the branch.