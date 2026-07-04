# Stream1 LibTorch CUDA Graph Research Notes

Date: 2026-07-04

Summary:

- C++ LibTorch exposes `ATen/cuda/CUDAGraph.h`; the current Docker Torch image has `at::cuda::CUDAGraph`, `capture_begin`, `capture_end`, `replay`, and graph-private memory pool support.
- The graph-capable Stream1 transformer runner should be an explicit backend/mode, not a fallback.
- Required production contract: fixed batch, fixed state length, fixed dtype, fixed device, fixed model shape, static input tensor address, static output tensor address, warmup before capture, and long-lived model/graph/tensors.
- If capture or shape validation fails, abort loudly.

Sources checked:

- PyTorch CUDA Graph notes: https://docs.pytorch.org/docs/2.12/notes/cuda.html#cuda-graphs
- PyTorch CUDAGraph API: https://docs.pytorch.org/docs/2.12/generated/torch.cuda.CUDAGraph.html
- ATen C++ CUDAGraph header in the local Torch image: `/usr/local/lib/python3.10/dist-packages/torch/include/ATen/cuda/CUDAGraph.h`
- NVIDIA CUDA Runtime stream capture API: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
- NVIDIA cuBLAS CUDA Graph support: https://docs.nvidia.com/cuda/cublas/index.html#cuda-graphs-support
- PyTorch SDPA API: https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html

Profiler status:

- Kaggle 2xT4 image did not have `nsys`: `NSYS_PROFILE_UNAVAILABLE=nsys_not_found` in v6 log.
- Local Docker image has Nsight tooling but no NVIDIA driver/GPU, so it cannot produce a useful GPU timeline locally.
- Next real profiler step should be on a GPU host with Nsight Systems installed:
  `nsys profile --trace=cuda,nvtx,osrt,cublas --stats=true --force-overwrite=true -o libtorch_s1_graph ./stream1_transformer_libtorch_benchmark --weight-dir ... --batches 384 --warmup 20 --iters 200 --device cuda:0 --cuda-graph`.