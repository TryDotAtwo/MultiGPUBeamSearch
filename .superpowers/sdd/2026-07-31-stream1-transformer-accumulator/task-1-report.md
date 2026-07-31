# Task 1 Report: Stream1 Transformer Accumulator Contract

## Delivered

- Added RuntimeConfig accumulator and Stream5 exchange physical-buffer fields. Unset fields preserve legacy ring_count sizing.
- Added runtime environment parsing for BEAM_ACCUMULATOR_BUFFER_COUNT and BEAM_STREAM5_EXCHANGE_BUFFER_COUNT; explicitly present zero values fail closed.
- Static-memory planning now sizes score/hash/parent/count by physical accumulator buffers, Stream5 send/receive by physical exchange buffers, and Stream3 scratch by macro batch.
- Added focused coverage for the 384 x 24 x 192 = 1,769,472 macro shape, two physical buffers, legacy fallback, invalid divisibility, and zero fallback.
- Did not change dispatcher scheduling; Task 2 owns buffer reuse and rebinding semantics.

## TDD evidence

The focused test was added before the production fields. A local CMake/Ninja build attempt could not compile any test because CUDA 12.5 nvcc cannot find cl.exe; Visual Studio BuildTools developer prompt also did not expose an MSVC compiler.

## Verification

- git diff --check: pass.
- Focused CUDA test executable: not built or run locally due missing cl.exe; executable GREEN remains pending.

## Concern

The source change has not had executable CUDA validation in this worktree. Run static_memory_cuda_tests on an MSVC/CUDA-capable host before merge.
