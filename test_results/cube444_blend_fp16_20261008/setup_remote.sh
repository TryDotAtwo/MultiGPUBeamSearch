#!/bin/bash
set -euo pipefail
export PATH=/venv/main/bin:/usr/local/cuda/bin:$PATH
mkdir -p /tmp/blend-production /tmp/blend-validation/assets
cd /tmp/blend-production
tar -xzf /tmp/source.tar.gz
if [ ! -d /tmp/blend-validation/cutlass ]; then git clone --quiet https://github.com/NVIDIA/cutlass.git /tmp/blend-validation/cutlass; fi
git -C /tmp/blend-validation/cutlass checkout --quiet afa1772203677c5118fcd82537a9c8fefbcc7008
TORCH_PREFIX=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')
NCCL_ROOT=$(python -c 'import nvidia.nccl,pathlib;print(pathlib.Path(next(iter(nvidia.nccl.__path__))))')
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBEAM_CUDA_ARCHITECTURES=86 -DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_STATE_ALIGNMENT=16 -DBEAM_MOVE_COUNT=24 -DCUTLASS_DIR=/tmp/blend-validation/cutlass -DBEAM_ENABLE_LIBTORCH_STREAM1=ON -DBEAM_ENABLE_DEBUG=ON -DBEAM_ENABLE_DEBUG_LOGS=ON -DBEAM_ENABLE_DEPTH_LOGS=ON -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" -DNCCL_INCLUDE_DIR="$NCCL_ROOT/include" -DNCCL_LIBRARY="$NCCL_ROOT/lib/libnccl.so.2"
cmake --build build --target production_runner_libtorch_stream1 -j 2
