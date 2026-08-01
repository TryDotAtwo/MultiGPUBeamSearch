# Stream1 Piece Transformer Clean Tensor Attention - 2026-07-01

## Change

Rewrote the Stream1 piece-transformer attention path away from the previous pointer-array `GemmArray` implementation.

Current attention layout:

- QKV projection stays as the existing CUTLASS linear GEMM.
- QKV bias is applied in-place to the QKV scratch.
- QK uses CUTLASS `GemmBatched` per head over a fixed padded score/prob scratch.
- Softmax uses the fixed `seq_len=51` warp kernel over the score rows.
- V is packed into the same attention scratch for tensor-op alignment.
- PV uses CUTLASS `GemmBatched` per head.
- Removed transformer pointer-table scratch (`attention_q_ptrs`, `attention_k_ptrs`, score/prob/context pointer arrays); MLP Stream1 path is unchanged.

This is not a fallback path and does not change model weights or outputs.

## Verification

Docker image: `gpu-dev-cutlass-nsight:2026-05-24`.

```bash
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace gpu-dev-cutlass-nsight:2026-05-24 bash -lc "cmake --build build-stream1-opt --target stream1_transformer_cuda_tests stream_benchmark -j2 && ./build-stream1-opt/stream1_transformer_cuda_tests"
```

Result:

```text
stream1_transformer_cuda_tests=pass
```

## Local Benchmark

Weights: `test_results/stream1_transformer_reference/weights_fp16`.
Puzzle: `0`.

| B_MICRO | concurrency | ms/group | parents/s | candidates/s | scratch bytes | report |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 1 | 21.2628 | 24079.6 | 577909.4 | 163864576 | `test_results/stream1_transformer_clean_tensor_attention_512x1_2026-07-01.md` |
| 1024 | 1 | 41.4473 | 24706.1 | 592946.4 | 327729152 | `test_results/stream1_transformer_clean_tensor_attention_1024x1_2026-07-01.md` |
| 512 | 2 | 40.7648 | 25119.7 | 602873.7 | 327729152 | `test_results/stream1_transformer_clean_tensor_attention_512x2_2026-07-01.md` |

For comparison, the previous local `512x1` GemmArray report in this worktree showed `565930.9` candidates/s with `164126720` scratch bytes, so this clean tensor path is slightly faster locally and uses less scratch in that point.

## Notes

A first scalar warp-only rewrite was tested and rejected because it passed correctness but dropped to `51780.6` candidates/s at `512x1`. The final implementation keeps tensor-core QK/PV while removing the pointer-array attention machinery.
Additional verification after scratch struct change:

```bash
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace gpu-dev-cutlass-nsight:2026-05-24 bash -lc "cmake --build build-stream1-opt --target contract_tests dispatcher_cuda_tests -j2 && ./build-stream1-opt/contract_tests && ./build-stream1-opt/dispatcher_cuda_tests"
```

Result:

```text
contract_tests=pass
dispatcher_cuda_tests=pass
```
