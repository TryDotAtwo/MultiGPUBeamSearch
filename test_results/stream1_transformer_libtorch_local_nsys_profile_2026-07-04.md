# Stream1 LibTorch Transformer Local Docker Nsight Profile

Date: 2026-07-04

Environment:

- Host GPU visible from Docker: NVIDIA GeForce RTX 3070 Laptop, driver 572.70, CUDA 12.8.
- Docker image used: `cmz-native-dev:2026-05-26`.
- Image already contains both LibTorch and Nsight Systems:
  - `torch=2.7.1+cu128`
  - `NVIDIA Nsight Systems version 2025.6.3.541-256337736014v0`
- No Dockerfile change was required for this profile run.

Build/profile command shape:

```bash
docker run --rm --gpus all --privileged \
  -v D:\100XH100\.worktrees\stream1-piece-transformer:/workspace \
  -w /workspace cmz-native-dev:2026-05-26 /bin/bash -lc '...
cmake -S /workspace -B /tmp/build-libtorch-nsys -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/usr/local/lib/python3.10/dist-packages/torch/share/cmake \
  -DBEAM_ENABLE_LIBTORCH_STREAM1=ON \
  -DCUTLASS_DIR=/opt/cutlass \
  -DBEAM_CUDA_ARCHITECTURES=86
cmake --build /tmp/build-libtorch-nsys --target stream1_transformer_libtorch_benchmark -j2
nsys profile --trace=cuda,nvtx,osrt,cublas --sample=none --cpuctxsw=none --stats=true --force-overwrite=true ...'
```

Weights:

`test_results/kaggle_libtorch_transformer_benchmark_v1_2026-07-04/stream1_transformer_weights_fp16`

Benchmark results on local RTX 3070 Laptop, batch 384, 50 timed iterations:

| Mode | elapsed_ms | parents/s | candidates/s | checksum |
|---|---:|---:|---:|---:|
| eager | 796.211 | 24114.2 | 578741 | 465948672 |
| cuda_graph | 769.746 | 24943.3 | 598639 | 465948672 |

Graph/eager on this local GPU: `1.0344x` by elapsed time, `1.0344x` by candidates/s.

Nsight Systems evidence:

- Eager profile generated:
  - `test_results/nsight_libtorch_transformer_2026-07-04/eager_b384.nsys-rep`
  - `test_results/nsight_libtorch_transformer_2026-07-04/eager_b384.sqlite`
  - `test_results/nsight_libtorch_transformer_2026-07-04/eager_b384.log`
- Graph profile generated:
  - `test_results/nsight_libtorch_transformer_2026-07-04/graph_b384.nsys-rep`
  - `test_results/nsight_libtorch_transformer_2026-07-04/graph_b384.sqlite`
  - `test_results/nsight_libtorch_transformer_2026-07-04/graph_b384.log`

CUDA Graph confirmation from `graph_b384.log`:

```text
cudaGraphLaunch_v10000  Num Calls=51
cudaGraphInstantiateWithFlags_v11040  Num Calls=1
cudaStreamBeginCapture_v10000  Num Calls=1
cudaStreamEndCapture_v10000  Num Calls=1
```

Eager hot kernel mix from `eager_b384.log` GPU kernel stats:

| Kernel family | GPU time share |
|---|---:|
| cuBLAS Lt GEMM 64x96x32 | 30.3% |
| cuBLAS Lt GEMM 160x128x32 | 17.3% |
| PyTorch FlashAttention forward | 12.7% |
| LayerNorm | 10.5% |
| elementwise mul / SiLU / add | about 22.0% combined |
| token-build indexSelect half | 3.9% |
| token-build cat | 0.9% |

Interpretation:

- The graph path really captures and replays. On this local RTX 3070 it is modestly faster than eager, unlike the previous Kaggle 2xT4 run where graph was slower.
- The main performance wall is not launch overhead anymore; it is the ATen/LibTorch kernel mix: many separate layernorm, elementwise, GEMM, FlashAttention, and token-build kernels.
- The largest remaining win is a fused/custom transformer block path, especially fusing LayerNorm + Linear epilogues and SiLU/gating/residual elementwise work. Token-build is visible but not the dominant cost.
- CUDA Graph should stay as an explicit graph backend/mode, but it is not a substitute for fusing the block.

Nsight Compute attempt:

A bounded NCU run against the dominant cuBLAS GEMM created `test_results/nsight_libtorch_transformer_2026-07-04/ncu_eager_gemm64x96.ncu-rep`, but metrics were unavailable on this host:

```text
No metrics to collect found in sections.
Cuda driver is not compatible with Nsight Compute.
Failed to load Nsight Compute CUDA modules.
```

So detailed occupancy/tensor-core counters need a driver/Nsight Compute compatible GPU host. Nsight Systems data from this Docker run is valid and sufficient for the next architectural decision.