# Stream1 Transformer Tensor Attention Rewrite - 2026-06-30

## Scope
- Branch/worktree: `codex/stream1-piece-transformer` at `D:\100XH100\.worktrees\stream1-piece-transformer`.
- Rewrote the Stream1 piece-transformer attention execution path away from the scalar/warp attention kernel.
- No fallback backend or distillation was added. MLP Stream1 path is untouched.

## Implementation
- `cuda/stream1.cu`
  - Added CUTLASS `GemmArray` tensor-core QK/PV attention path for the fixed transformer shape `seq_len=51`, `d_model=256`, `nhead=8`, `head_dim=32`.
  - Attention now runs as: QKV bias add -> QK `GemmArray` -> V pack into padded 64x32 scratch -> softmax over 51 keys with zero-padded columns -> PV `GemmArray` into normal context layout.
  - Added device pointer-array setup and chunked `GemmArray` launches with max batch chunk 32768 so `B_MICRO=8192` no longer fails at `8192*8=65536` attention matrices.
- `cuda/stream1.hpp`, `tools/stream1_weight_io.hpp`
  - Added transformer attention pointer-array scratch and runtime scratch sizing/allocation/view offsets.
  - Changed attention score/prob scratch to padded half storage: per `(row, head)` stride is `seq_len*align16(seq_len) + align16(seq_len)*head_dim`.

## Verification
Command:
```bash
docker run --rm --gpus all -v "${PWD}:/workspace" -w /workspace gpu-dev-cutlass-nsight:2026-05-24 bash -lc "cmake --build build-stream1-opt --target stream1_transformer_cuda_tests stream_benchmark -j2 && ./build-stream1-opt/stream1_transformer_cuda_tests"
```
Result:
```text
stream1_transformer_cuda_tests=pass
```

Large micro-batch check:
```text
B_MICRO=8192 concurrency=1 candidates_per_sec=488477.2 scratch_bytes=2626027520
```
This previously failed with `Stream1 piece_transformer QK GemmArray launch failed` at batch_count 65536; chunking fixed the failure.

## Local T4 Benchmark Results
Best point from `test_results/stream1_transformer_gemmarray_attention_sweep_chunked_2026-06-30.md`:
```text
b_micro=512 concurrency=1 candidates_per_sec=583142.3 scratch_bytes=164126720
```
Other useful points:
```text
b_micro=1024 concurrency=1 candidates_per_sec=565197.9
b_micro=1024 concurrency=2 candidates_per_sec=539804.0
b_micro=2048 concurrency=1 candidates_per_sec=446844.9
```

## Nsight Summary
New Nsight report: `test_results/nsys_stream1_transformer_gemmarray_attention_2048x1_2026-06-30.*`.
Kernel time split after rewrite:
- CUTLASS linear GEMMs: ~44.4%
- layernorm copy: ~14.4%
- bias SiLU: ~11.6%
- QKV bias add: ~6.7%
- softmax51: ~5.3%
- residual bias add: ~5.3%
- QK GemmArray: ~3.8%
- pack V: ~3.7%
- PV GemmArray: ~3.4%

Conclusion: the original scalar attention bottleneck is gone. Remaining overhead is now dominated by transformer block linear GEMMs and unfused elementwise kernels, especially layernorm/SILU/QKV bias/softmax/pack.
Additional regression checks:
```text
contract_tests=pass
dispatcher_cuda_tests=pass
```
Final cleanup verification after removing unused scalar/per-head attention code:
```text
stream1_transformer_cuda_tests=pass
contract_tests=pass
dispatcher_cuda_tests=pass
```

Final single-point benchmark after cleanup:
```text
b_micro=512 concurrency=1 candidates_per_sec=565930.9 scratch_bytes=164126720
```