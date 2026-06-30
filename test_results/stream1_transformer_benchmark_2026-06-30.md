# Stream1 Transformer Benchmark Task 1 - 2026-06-30

## Status
DONE

## Implementation
- Kept `tools/stream_benchmark.cu` as the small benchmark entrypoint and split backend code into `tools/stream_benchmark_mlp.cu`, `tools/stream_benchmark_transformer.cu`, and shared helpers in `tools/stream_benchmark_common.hpp`.
- The entrypoint dispatches `STREAM1_BACKEND_PIECE_TRANSFORMER` to `benchmark_stream1_transformer(...)` and returns after writing the report; the existing MLP benchmark path dispatches to `benchmark_stream1_mlp(...)` plus Stream2/3/4 in the MLP file.
- Transformer benchmark constructs `stream1_weights::TransformerNetworkViewHolder` from uploaded transformer weights, allocates transformer scratch with `stream1_weights::alloc_stream1_scratch(stream1_model, b_micro, concurrency)`, allocates `parent_base`, `count`, and `score_ring` device buffers, and times `stream1_transformer_inference_cuda(...)` with CUDA events. Per-config CUDA streams, scratch, and device buffers are owned by a cleanup guard so failures do not leak resources into later sweep points.
- Sweep set: `B_MICRO={512,1024,2048,4096,8192}` and `STREAM1_CONCURRENCY={1,2,4}`.
- Configs whose estimated scratch + IO + safety margin exceed available GPU memory are skipped with an explicit report/stdout line. No fallback backend is used.

## Verification

### Build
Command:
```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:2026-05-24 bash -lc "cmake -S /workspace -B /tmp/beam-stream1-bench-build -G Ninja && cmake --build /tmp/beam-stream1-bench-build --target stream_benchmark"
```
Result: pass. `stream_benchmark` linked successfully. Existing warnings were from CUTLASS constexpr diagnostics and an unused Stream3 debug kernel.

Local Windows CMake configure was also attempted and failed before build because CUDA `nvcc` could not find `cl.exe` in `PATH`; Docker was used for the actual build verification.

### Transformer Benchmark Run
Command:
```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace -e BEAM_WEIGHT_DIR=/workspace/test_results/stream1_transformer_reference/weights_fp16 -e BEAM_STREAM_BENCH_REPORT=/workspace/test_results/stream1_transformer_benchmark_run_2026-06-30.md gpu-dev-cutlass-nsight:2026-05-24 /workspace/build-stream1-transformer-docker/stream_benchmark
```
Result: pass against the local fixture transformer weights. GPU reported `gpu_total_bytes=8589410304`, `gpu_free_before_bytes=7463763968`.

| b_micro | concurrency | rows_per_launch_group | ms_per_launch_group | parents_per_sec | candidates_per_sec | scratch_bytes |
|---:|---:|---:|---:|---:|---:|---:|
|512|1|512|98.0405|5222.3|125336.0|162963456|
|512|2|1024|203.8431|5023.5|120563.3|325926912|
|512|4|2048|407.4517|5026.4|120632.7|651853824|
|1024|1|1024|195.1863|5246.3|125910.5|325926912|
|1024|2|2048|408.4898|5013.6|120326.1|651853824|
|1024|4|4096|815.9496|5019.9|120478.0|1303707648|
|2048|1|2048|389.7066|5255.2|126125.7|651853824|
|2048|2|4096|881.2358|4648.0|111552.4|1303707648|
|2048|4|8192|3608.8718|2270.0|54479.1|2607415296|
|4096|1|4096|2005.2882|2042.6|49022.4|1303707648|
|4096|2|8192|3906.3069|2097.1|50330.9|2607415296|
|4096|4|16384|7816.5366|2096.1|50305.7|5214830592|
|8192|1|8192|3998.0593|2049.0|49175.9|2607415296|
|8192|2|16384|7815.9243|2096.2|50309.6|5214830592|
|8192|4|32768|skip|skip|skip|10429661184: estimated allocation exceeds available GPU memory free_bytes=7449083904 io_bytes=3538944|


### Post-Review Split Build
Command:
```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace gpu-dev-cutlass-nsight:2026-05-24 bash -lc "cmake -S /workspace -B /tmp/beam-stream1-split-build -G Ninja && cmake --build /tmp/beam-stream1-split-build --target stream_benchmark -j 2"
```
Result: pass. `stream_benchmark` linked from separate entrypoint, MLP, and transformer source files.

### MLP Split Smoke
Command:
```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace -e BEAM_WEIGHT_DIR=/workspace/stream1_weights -e BEAM_STREAM_MICRO_ONLY=1 -e BEAM_STREAM_BENCH_REPORT=/workspace/test_results/stream1_mlp_split_smoke_2026-06-30.md -e CUDA_LAUNCH_BLOCKING=1 gpu-dev-cutlass-nsight:2026-05-24 bash -lc "set -o pipefail; cmake -S /workspace -B /tmp/beam-stream1-split-smoke-build -G Ninja >/tmp/cmake.log && cmake --build /tmp/beam-stream1-split-smoke-build --target stream_benchmark -j 2 >/tmp/build.log && /tmp/beam-stream1-split-smoke-build/stream_benchmark 0 2>&1 | tee /workspace/test_results/stream1_mlp_split_smoke_console_2026-06-30.log"
```
Result: partial pass. MLP path entered `benchmark_stream1_mlp` and produced Stream1 rows through `B_MICRO=65536, concurrency=8`. The run was manually stopped before the last long sweep points because this smoke only needed to verify split-path execution. Console log: `test_results/stream1_mlp_split_smoke_console_2026-06-30.log`.

## Self-review Findings
- MLP execution path is isolated in `tools/stream_benchmark_mlp.cu` and remains the existing Stream1/2/3/4 benchmark flow; the transformer branch in the entrypoint returns before Stream2/3/4.
- No fallback backend path was added.
- No Stream3, Stream4, dispatcher, runtime config, or production runner source files were changed.
