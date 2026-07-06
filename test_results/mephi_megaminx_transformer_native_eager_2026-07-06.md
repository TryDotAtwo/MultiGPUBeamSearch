# MEPhI Megaminx Transformer Native Eager Executor

Date: 2026-07-06

Problem:

- Cluster job `31733` tried Megaminx Vlad `piece_transformer` with `BEAM_STREAM1_EXECUTOR=native_cuda_graph`, `BEAM_B_MICRO=512`, concurrency `2`, beam `700M`.
- Static memory planning passed, then ring-slot CUDA Graph instantiation failed with `cudaGraphInstantiate: out of memory`.
- With `B_MICRO=512`, one ring slot covers `512 * 24 = 12288` candidates. At local beam about `87.5M`, this requires many ring-slot graph execs. The isolated Stream1 benchmark chose this tiny batch for throughput, but production graph template count becomes much larger than the earlier MLP-style `B_MICRO=8192` runs.
- The user-provided `nvidia-smi` AWK used CSV field `$6`, which is total memory. In the MEPhI monitor format used memory is `$5`; the log still showed about `40324 MiB` used before the graph-instantiation OOM.

Change:

- Added explicit `BEAM_STREAM1_EXECUTOR=native_eager` / `native_no_graph` support to `production_runner`.
- `native_eager` uses the same native CUDA/CUTLASS Stream1 kernels and existing Stream2 hash/goal kernel, but launches Stream1/2 ring slots through the dispatcher hook instead of pre-instantiating ring-slot CUDA Graph templates.
- Stream3, Stream4, Stream5, final materialization, memory planning, and graph templates for non-ring-slot phases are unchanged.
- This is an explicit executor selection, not fallback behavior.
- Fixed the LibTorch ring-slot launcher to call Stream2 with job-relative `ring=0, ring_slot=0`, matching graph-capture behavior when `parent_base/count` pointers are already offset to the selected job.
- Updated MEPhI launchers to accept `MEGAMINX_STREAM1_BACKEND=native_eager` and `native_no_graph`.

Verification:

```text
docker run --rm -v D:\100XH100\.worktrees\stream1-piece-transformer:/work -w /work cmz-native-dev:2026-05-26 bash -lc "cmake -S . -B build-native-eager-check -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75 >/tmp/cmake.log && cmake --build build-native-eager-check --target production_runner contract_tests -j2 && ./build-native-eager-check/contract_tests"
```

Result:

```text
[100%] Built target production_runner
[100%] Built target contract_tests
contract_tests=pass
```

Shell syntax check:

```text
docker run --rm -v D:\100XH100\.worktrees\stream1-piece-transformer:/work -w /work bash:5.2 bash -lc "bash -n hpc/start_8xa100_libtorch_megaminx.sh hpc/solve_then_reflect.sh hpc/bench_8xa100_megaminx_transformer.sh"
```

Result: exit code `0`.

Next cluster test:

- Re-run the same 700M puzzle 990 smoke with `MEGAMINX_STREAM1_BACKEND=native_eager`, `BEAM_B_MICRO=512`, and `BEAM_STREAM1_CONCURRENCY=2`.
- If launch overhead is too high, implement bounded graph-window caching as a separate optimization: instantiate only a limited moving set of ring-slot graph execs instead of all jobs at startup.