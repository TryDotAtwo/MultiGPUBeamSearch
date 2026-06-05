# Beam Solver CUDA Scaffold

Agent-centered C++20/CUDA project scaffold for the multi-stream beam-search architecture in `ARCHITECTURE_NEED.md`.

## Current Scope
- Contract types: `State128`, `Hash128`, `CandidateMeta`, `FinalRequest`, `FinalResponse`.
- Config derivation: `RING_SLOT_COUNT`, `BEAM_WIDTH_ALIGNMENT`, `GLOBAL_BEAM_WIDTH_EFFECTIVE`.
- State operations: padding cleanup, final response byte-pack, move application.
- Hash operations: deterministic `Hash128` Zobrist reference with zero padding rows.
- Stream 3 reference: threshold, compact semantics, Hash128 sort/dedup, owner split, remote grouping.
- Stream 4 reference: threshold plus CUB/fixed-temp Hash128 sort/reduce dedup with deterministic tie-break.
- CPU reference depth expansion: small deterministic correctness harness, not production hot path.
- CUDA Stream 2 contract kernel: local child materialization, Hash128 generation, goal flag writes.
- CUDA Stream 3 contract kernel: threshold plus compact pack preserving original payload id.
- CUDA Stream 4 kernel: threshold + compact + batched CUB/fixed-temp Hash128 sort/reduce + clean-region rewrite.
- CUDA Stream 5 contract kernel: single-rank exchange contract.
- CUDA Stream 5 threshold path: reads fixed per-shard score histograms and computes global threshold state.
- CUDA final materialize kernel: FinalRequest to FinalResponse to `next_frontier_states_tmp`.
- Stitched single-rank CUDA contract test: Stream 1 score, Stream 2 goal/hash, Stream 3 compact, Stream 4 dedup, final materialize.

## Current Non-Scope
- Production multi-process multi-GPU orchestration.
- Multi-rank Stream 5 payload exchange validation on separate devices.
- CPU history reconstruction for full solved-path reporting beyond current production runner state transfer.

## Build
```powershell
cmake -S . -B build
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Docker GPU Tests
```powershell
docker compose build beam-tests
docker compose run --rm beam-tests
```

## Benchmark And Nsight Profiling
```powershell
docker compose run --rm nsight-profile
```

`nsight-profile` uses only NVIDIA Nsight Systems (`nsys profile`). Runtime framework profiler code is absent.

Latest Docker GPU result on local RTX 3070:
```text
contract_tests=pass
stream1_cuda_tests=pass
stream2_cuda_tests=pass
stream3_cuda_tests=pass
stream4_cuda_tests=pass
stream5_cuda_tests=pass
final_cuda_tests=pass
stitched_cuda_tests=pass
```

## Test Artifacts
- Root artifacts: `test_results/`.
- CTest working-directory artifacts: `build/test_results/`.
- Docker CTest artifacts: `build-docker/test_results/`.

## Next Implementation Slice
1. Reduce Stream3 owner split and scatter cost.
2. Reduce Stream1 folded-input and Stream2 hash/move overlap contention.
3. Add multi-rank NCCL Stream5 validation once local Docker/NVIDIA setup supports multi-process GPU tests.
