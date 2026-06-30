# Stream1 Transformer Shared-Attention Pass 2026-06-30

## Reason

The first Kaggle 2xT4 transformer benchmark was far slower than expected: best aggregate was about `232720.5` candidates/s, while the MLP Stream1 path has local smoke rows around `10-12M` candidates/s per GPU and the model FLOP ratio is only about `5x`.

## Change

- Replaced the transformer attention scratch path with a shared-memory attention kernel.
- Removed the global `attention_scores_probs` transformer scratch allocation from sizing/allocation.
- Removed redundant transformer scratch clears from the steady Stream1 forward launch.
- MLP Stream1 and Stream2/3/4 code paths were not changed.

## Local Verification

Command:

```bash
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-task6-sm86 --target stream1_transformer_cuda_tests stream_benchmark -j2 && ./build-task6-sm86/stream1_transformer_cuda_tests && BEAM_WEIGHT_DIR=test_results/stream1_transformer_reference/weights_fp16 ./build-task6-sm86/stream_benchmark 0 | tee test_results/stream1_transformer_shared_attention_local_benchmark_2026-06-30.log"
```

Result:

```text
stream1_transformer_cuda_tests=pass
best_local_row=b_micro=512 concurrency=2 candidates_per_sec=211632.4 scratch_bytes=240697344
```

Additional checks:

```text
contract_tests=pass
dispatcher_cuda_tests=pass
```

## Interpretation

The patch is directionally correct: local `b_micro=512, concurrency=1` improved from the prior Kaggle-style `~108ms/group` reference point to `62.7791ms/group` on the local GPU, and scratch for the same row group dropped from `162963456` to `120348672` bytes. This is still not enough; the remaining backend is still much less efficient than MLP and needs deeper transformer-specific optimization.