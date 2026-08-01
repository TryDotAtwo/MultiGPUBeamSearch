# Stream1 Transformer Forward Verification - 2026-06-29

## Scope
- Added a standalone CUDA Stream1 piece-transformer forward entrypoint without production dispatcher/runner wiring.
- Added separate `Stream1TransformerNetworkView` and `Stream1TransformerScratchView` API surfaces.
- Implemented p900 input token construction from `fast_slot_projected`, `fast_piece_static`, `piece_positions`, and `piece_mask`.
- Implemented input/output LayerNorm, pre-norm transformer blocks, CUTLASS linear projections, SiLU FFN activation, residual adds, and output score-key quantization.
- Implemented a fused row-wise tiled attention kernel for the current `seq_len=51`, using the existing reusable attention score/probability buffer in place, with explicit scratch initialization before the standalone forward. Full external FlashAttention 2 integration is deferred; no new external dependency was added.
- Kept production runner and benchmark fail-closed transformer guards unchanged.

## Red Check

```powershell
docker run --rm -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake -S . -B build-stream1-transformer-red -G Ninja -DBEAM_CUDA_ARCHITECTURES=75 >/tmp/cmake-red.log && cmake --build build-stream1-transformer-red --target stream1_transformer_cuda_tests -j2"
```

Result: FAIL before implementation as expected. The new test referenced missing `TransformerNetworkViewHolder`, `transformer_network_view`, `Stream1TransformerScratchView`, `transformer_scratch_view`, and `stream1_transformer_inference_cuda` symbols.

## Final Verification

```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-stream1-transformer-red --target stream1_transformer_cuda_tests -j2 >/tmp/build.log && ./build-stream1-transformer-red/stream1_transformer_cuda_tests"
```

Result: PASS, `stream1_transformer_cuda_tests=pass`. The real exporter fixture under `test_results/stream1_transformer_reference/` ran 8 p900 reference states. Max score-key error versus fp32 PyTorch reference was 1573 in the final single run, under the 3072 fp16/TensorCore tolerance. A five-run sample after scratch initialization produced max score-key errors of 1292, 1382, 1290, 1338, and 1030.

```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-stream1-transformer-red --target stream1_cuda_tests -j2 >/tmp/build-stream1.log && ./build-stream1-transformer-red/stream1_cuda_tests"
```

Result: PASS, `stream1_cuda_tests=pass`.

```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-stream1-transformer-red --target dispatcher_cuda_tests -j2"
```

Result: PASS compile. This validates the existing dispatcher test fixture against the expanded MLP network view shape; it does not wire transformer dispatch.

## Task 5 Quality Fix Verification

Scope: fixed `tools/stream_benchmark.cu` MLP `Stream1NetworkView` construction to include the expanded input/hidden/residual LayerNorm pointer fields. No transformer forward semantics changed, and production transformer dispatch remains unwired.

Red compile check before fix:

```powershell
docker run --rm -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-stream1-transformer-red --target stream_benchmark -j2"
```

Result: FAIL as expected. `tools/stream_benchmark.cu` attempted to initialize residual pointer tables into the new scalar LayerNorm pointer fields and failed with CUDA compile errors at lines 347-353.

Final `stream_benchmark` compile check:

```powershell
docker run --rm -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-stream1-transformer-red --target stream_benchmark -j2"
```

Result: PASS. The target rebuilt and linked `stream_benchmark` successfully.

Transformer CUDA regression check:

```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-stream1-transformer-red --target stream1_transformer_cuda_tests -j2 && ./build-stream1-transformer-red/stream1_transformer_cuda_tests"
```

Result: PASS, `stream1_transformer_cuda_tests=pass`.

## Task 5 Spec-Fix: Clean Checkout Fixture Skip

Scope: made the registered `stream1_transformer_cuda_tests` safe when the ignored reference fixture is absent from a clean checkout. The test now checks for both `test_results/stream1_transformer_reference/weights_fp16/manifest.json` and `test_results/stream1_transformer_reference/reference.json` before CUDA initialization. If either file is absent, it prints and records `stream1_transformer_cuda_tests=skip missing_reference_fixture` and exits 0. When the fixture exists, the full real p900 reference test is unchanged.

Missing-fixture verification without deleting the real fixture:

```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "cmake --build build-stream1-transformer-red --target stream1_transformer_cuda_tests -j2 && BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR=test_results/stream1_transformer_reference_missing ./build-stream1-transformer-red/stream1_transformer_cuda_tests && cat test_results/stream1_transformer_cuda_tests_2026-06-29.md"
```

Result: PASS skip behavior. The binary returned 0, printed `stream1_transformer_cuda_tests=skip missing_reference_fixture`, and wrote the same skip line plus missing fixture paths to `test_results/stream1_transformer_cuda_tests_2026-06-29.md`.

Fixture-present verification:

```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "./build-stream1-transformer-red/stream1_transformer_cuda_tests && cat test_results/stream1_transformer_cuda_tests_2026-06-29.md"
```

Result: PASS, `stream1_transformer_cuda_tests=pass`. The real exporter fixture under `test_results/stream1_transformer_reference/` ran 8 p900 reference states. Max score-key error was 745, under the unchanged 3072 fp16/TensorCore tolerance.

Registered CTest verification with the missing-fixture override:

```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:cuda128-sm120 bash -lc "BEAM_STREAM1_TRANSFORMER_REFERENCE_DIR=test_results/stream1_transformer_reference_missing ctest --test-dir build-stream1-transformer-red -R '^stream1_transformer_cuda_tests$' --output-on-failure"
```

Result: PASS, `100% tests passed, 0 tests failed out of 1`; `stream1_transformer_cuda_tests` completed in 0.02 sec.
