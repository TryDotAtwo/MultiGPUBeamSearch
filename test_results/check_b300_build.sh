#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
mkdir -p "$tmp/cutlass/include" "$tmp/nccl/include" "$tmp/data"
touch "$tmp/nccl/libnccl.so" "$tmp/data/puzzle_info.json"
printf '#!/usr/bin/env bash\ncat "$B300_TEST_NVCC_VERSION"\n' > "$tmp/nvcc"
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$*" >> "$B300_TEST_CMAKE_CALLS"\n' > "$tmp/cmake"
printf '#!/usr/bin/env bash\necho "NVIDIA B300"\n' > "$tmp/nvidia-smi"
chmod +x "$tmp/nvcc" "$tmp/cmake" "$tmp/nvidia-smi"
export B300_NVCC="$tmp/nvcc" B300_CMAKE="$tmp/cmake" B300_NVIDIA_SMI="$tmp/nvidia-smi"
export B300_TEST_NVCC_VERSION="$tmp/version" B300_TEST_CMAKE_CALLS="$tmp/cmake_calls"
export B300_CUTLASS_DIR="$tmp/cutlass" B300_NCCL_INCLUDE_DIR="$tmp/nccl/include"
export B300_NCCL_LIBRARY="$tmp/nccl/libnccl.so" B300_DATA_DIR="$tmp/data"
export B300_REPO_DIR="$PWD" B300_BUILD_DIR="$tmp/build"

printf 'Cuda compilation tools, release 12.8, V12.8.93\n' > "$tmp/version"
if bash hpc/b300_build.sh > "$tmp/old.log" 2>&1; then
  echo 'FAIL: CUDA 12.8 was accepted for SM103' >&2
  exit 1
fi
[[ ! -f "$tmp/cmake_calls" ]]

printf 'Cuda compilation tools, release 13.0, V13.0.88\n' > "$tmp/version"
bash hpc/b300_build.sh
grep -q -- '-DBEAM_CUDA_ARCHITECTURES=103' "$tmp/cmake_calls"
grep -q -- '-DBEAM_PUZZLE_INFO_JSON=' "$tmp/cmake_calls"
grep -q -- '--target production_runner stream_benchmark stream1_transformer_cuda_tests' "$tmp/cmake_calls"
echo 'PASS: B300 build rejects old CUDA and configures SM103 targets; mock compiler only'
