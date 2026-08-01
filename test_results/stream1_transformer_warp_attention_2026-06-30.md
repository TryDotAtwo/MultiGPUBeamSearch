# Stream1 Transformer Warp-Attention Pass 2026-06-30

## Reason

The shared-memory attention patch improved the initial transformer backend but was still too slow. The remaining attention kernel serialized all 51 queries inside one `(row, head)` block.

## Change

- Replaced the transformer attention kernel with a warp-specialized short-sequence kernel.
- One block owns one `(row, head)` tile.
- Each warp owns one query at a time and reduces `head_dim=32` with `__shfl_down_sync`.
- Softmax scores stay in per-warp shared memory.
- The global attention score/probability scratch remains removed.
- MLP Stream1 path and Stream2/3/4 are unchanged.

## Local Verification

Main benchmark command:

```bash
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-task6-sm86 --target stream1_transformer_cuda_tests stream_benchmark -j2 && ./build-task6-sm86/stream1_transformer_cuda_tests && BEAM_WEIGHT_DIR=test_results/stream1_transformer_reference/weights_fp16 ./build-task6-sm86/stream_benchmark 0 | tee test_results/stream1_transformer_warp_attention_local_benchmark_2026-06-30.log"
```

Result:

```text
stream1_transformer_cuda_tests=pass
best_local_row=b_micro=1024 concurrency=1 candidates_per_sec=304663.2 scratch_bytes=240697344
```

Additional checks:

```text
contract_tests=pass
dispatcher_cuda_tests=pass
```

## Speed Interpretation

Local progression for comparable small rows:

- Original Kaggle 2xT4 baseline: about `113753.8` to `118966.7` candidates/s per T4 at `B_MICRO=512, concurrency=1`.
- Shared-attention local pass: `195734.1` candidates/s at `B_MICRO=512, concurrency=1`.
- Warp-attention local pass: `300068.8` candidates/s at `B_MICRO=512, concurrency=1`, best `304663.2` at `B_MICRO=1024, concurrency=1`.

The backend is now much faster, but still far behind the MLP path. The later PyTorch fast-path evidence contradicts treating the gap as an intrinsic `~20x` model-cost limit. Keep the empirical target at roughly the PyTorch-observed `~5x` envelope and treat the remaining native gap as an implementation/performance issue: too many unfused kernels, scalar/custom attention versus optimized SDPA-style kernels, and repeated CUTLASS GEMM setup around small fixed transformer shapes.