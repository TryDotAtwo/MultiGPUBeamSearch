# Task 1 Transformer Accumulator Contract Test Evidence

- Focused test: tests/static_memory_cuda_tests.cu run_accumulator_buffer_contract_test.
- Required shape: B_MICRO=384, MOVE_COUNT=24, macro Stream3 batch 1,769,472, 192 logical chunks, two physical accumulator buffers, and two physical Stream5 exchange buffers.
- Assertions cover bounded score/hash/parent/count, macro-sized Stream3 scratch, bounded Stream5 slots, legacy ring_count fallback, and fail-closed non-divisible and zero dimensions.
- RED record: the test was written before the contract fields. Local compilation is blocked before source compilation because CUDA 12.5 nvcc cannot find cl.exe.
- GREEN status: source implementation complete; executable GREEN pending an MSVC/CUDA-capable host.
