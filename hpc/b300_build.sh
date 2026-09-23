#!/usr/bin/env bash
# Native B300 (SM103) build gate. Uses the existing FP16 CUTLASS fallback path.
set -euo pipefail
repo=${B300_REPO_DIR:-/workspace/MGBFS}
build=${B300_BUILD_DIR:-/workspace/build-b300}
data=${B300_DATA_DIR:-/workspace}
cutlass=${B300_CUTLASS_DIR:-/workspace/cutlass}
nvcc=${B300_NVCC:-/usr/local/cuda/bin/nvcc}
cmake=${B300_CMAKE:-cmake}
smi=${B300_NVIDIA_SMI:-nvidia-smi}
nccl_include=${B300_NCCL_INCLUDE_DIR:-/usr/include}
nccl_library=${B300_NCCL_LIBRARY:-/usr/lib/x86_64-linux-gnu/libnccl.so}

version=$("$nvcc" --version)
if [[ ! "$version" =~ release[[:space:]]+([0-9]+)\.([0-9]+) ]] ||
   (( ${BASH_REMATCH[1]:-0} < 13 )); then
  echo 'B300 build requires CUDA Toolkit 13.0+ (native SM103)' >&2
  exit 2
fi
gpu_names=$("$smi" --query-gpu=name --format=csv,noheader)
[[ "$gpu_names" == *B300* ]] || { echo 'B300 GPU not found' >&2; exit 2; }
[[ -d "$cutlass/include" ]] || { echo "missing CUTLASS: $cutlass" >&2; exit 2; }
[[ -d "$nccl_include" && -f "$nccl_library" ]] || { echo 'missing NCCL headers/library' >&2; exit 2; }
[[ -f "$data/puzzle_info.json" ]] || { echo 'missing puzzle_info.json' >&2; exit 2; }

echo "gpu_names=$gpu_names"
echo "nvcc=$(printf '%s\n' "$version" | tail -n 1)"
echo "repo=$repo build=$build cutlass=$cutlass"
"$cmake" -S "$repo" -B "$build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$nvcc" \
  -DBEAM_CUDA_ARCHITECTURES=103 \
  -DCUTLASS_DIR="$cutlass" \
  -DNCCL_INCLUDE_DIR="$nccl_include" \
  -DNCCL_LIBRARY="$nccl_library" \
  -DBEAM_PUZZLE_INFO_JSON="$data/puzzle_info.json" \
  -DBEAM_ENABLE_DEBUG=ON -DBEAM_ENABLE_DEPTH_LOGS=ON
"$cmake" --build "$build" --target production_runner stream_benchmark stream1_transformer_cuda_tests -j "${B300_BUILD_JOBS:-8}"
if [[ -f "$build/production_runner" ]]; then
  sha256sum "$build/production_runner" "$build/stream_benchmark"
fi
