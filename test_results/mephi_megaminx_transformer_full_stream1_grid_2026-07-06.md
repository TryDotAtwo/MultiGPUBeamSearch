# MEPhI Megaminx Transformer Full Stream1 Grid

Date: 2026-07-06

Scope:

- Make the isolated MEPhI Stream1 transformer benchmark sweep the same `B_MICRO x concurrency` grid for every explicit backend family.
- Keep production selection fail-closed: PyTorch remains reference timing only; production best-env selects native CUDA graph or LibTorch eager.
- Preserve the existing 900M pipeline config; this changes Stream1 benchmarking only.

Grid now covered by default in `hpc/bench_8xa100_megaminx_transformer.sh`:

```text
B_MICRO: 512 1024 2048 4096 8192 12288 16384
concurrency: 1 2 4 8
backend modes: pytorch:eager, libtorch:eager, libtorch:cuda_graph, native_cutlass:graph
```

Implementation notes:

- `tools/stream1_transformer_torch_benchmark.py` accepts `--concurrency`, runs `batch * concurrency` parent rows per measured group, and writes `rows_per_launch_group`.
- `tools/stream1_transformer_libtorch_benchmark.cpp` accepts `--concurrency` and measures eager/CUDA-graph groups the same way.
- `tools/stream1_transformer_backends.py` forwards concurrency to all three backend families.
- The MEPhI summary parser now matches both `candidates_per_sec=` and PyTorch `candidates_per_s=` lines.

Verification:

```text
python tests\test_stream1_transformer_backends.py
.....
Ran 5 tests in 0.001s
OK

python tests\test_stream1_transformer_parity.py
.......
Ran 7 tests in 0.000s
OK

python ast syntax check for tools/stream1_transformer_torch_benchmark.py and tools/stream1_transformer_backends.py
ast_syntax_ok=1

docker run ... bash -lc "bash -n hpc/mephi_8xa100_common.sh hpc/bench_8xa100_megaminx_transformer.sh hpc/start_8xa100_libtorch_megaminx.sh"
exit=0

docker run ... bash -lc "cmake --build build-transformer-plan-libtorch2 --target stream1_transformer_libtorch_benchmark -j2"
[100%] Built target stream1_transformer_libtorch_benchmark

git diff --check
exit=0, only CRLF normalization warnings
```