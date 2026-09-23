#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
mkdir -p "$tmp/bin" "$tmp/build" "$tmp/data" "$tmp/weights" "$tmp/cutlass/include" "$tmp/nccl"
cp test_results/hopper_runtime_v1/data/puzzle_info.json test_results/hopper_runtime_v1/data/test.csv "$tmp/data/"
archive_root=$(cd ../.. && pwd)
cp "$archive_root/test_results/paper_benchmarks/cube4_ours_t4_b4m_depth10/stream1_transformer_weights_fp16/"* "$tmp/weights/"
touch "$tmp/nccl/nccl.h" "$tmp/nccl/libnccl.so"
printf '#!/usr/bin/env bash\nif [[ $1 == --list-gpu-arch ]]; then echo compute_75; echo compute_80; echo compute_90; echo compute_103; else echo "Cuda compilation tools, release 13.0"; fi\n' > "$tmp/bin/nvcc"
printf '#!/usr/bin/env bash\nif [[ $* == *compute_cap* ]]; then printf "10.3\\n10.3\\n"; else printf "NVIDIA B300\\nNVIDIA B300\\n"; fi\n' > "$tmp/bin/nvidia-smi"
printf '#!/usr/bin/env bash\nprintf "%%s\\n" "$*" >> "$PORTABLE_TEST_CMAKE_CALLS"\n' > "$tmp/bin/cmake"
printf '#!/usr/bin/env bash\necho "rank=${RANK:-x} world=${WORLD_SIZE:-x} beam=${3:-x} cap=${BEAM_SHARD_CAPACITY_CANDIDATES:-x} qkv=${BEAM_STREAM1_TRANSFORMER_HOPPER_QKV:-x}"\n' > "$tmp/build/production_runner"
printf '#!/usr/bin/env bash\necho "micro=$BEAM_STREAM1_TRANSFORMER_B_MICRO lanes=$BEAM_STREAM1_TRANSFORMER_CONCURRENCY" > "$BEAM_STREAM_BENCH_REPORT"\n' > "$tmp/build/stream_benchmark"
chmod +x "$tmp/bin/"* "$tmp/build/"*
export PORTABLE_REPO_DIR="$PWD" PORTABLE_DATA_DIR="$tmp/data" PORTABLE_WEIGHT_DIR="$tmp/weights"
export PORTABLE_BUILD_DIR="$tmp/build" PORTABLE_RUN_ROOT="$tmp" PORTABLE_NVCC="$tmp/bin/nvcc"
export PORTABLE_SMI="$tmp/bin/nvidia-smi" PORTABLE_CMAKE="$tmp/bin/cmake"
export PORTABLE_CUTLASS_DIR="$tmp/cutlass" PORTABLE_NCCL_INCLUDE_DIR="$tmp/nccl"
export PORTABLE_NCCL_LIBRARY="$tmp/nccl/libnccl.so" PORTABLE_TEST_CMAKE_CALLS="$tmp/cmake_calls"
export PORTABLE_PYTHON=$(command -v python)
printf 'MemAvailable: 1000000000 kB\n' > "$tmp/meminfo"
export PORTABLE_MEMINFO_PATH="$tmp/meminfo"

probe=$(bash hpc/cube4_portable.sh probe)
grep -q 'arch=103 visible_gpus=2' <<< "$probe"
bash hpc/cube4_portable.sh build
grep -q -- '-DBEAM_CUDA_ARCHITECTURES=103' "$tmp/cmake_calls"
grep -q -- '-DBEAM_STATE_LOGICAL_BYTES=96' "$tmp/cmake_calls"
printf '#!/usr/bin/env bash\necho compute_90\n' > "$tmp/unsupported-nvcc"
chmod +x "$tmp/unsupported-nvcc"
if PORTABLE_NVCC="$tmp/unsupported-nvcc" bash hpc/cube4_portable.sh build > "$tmp/unsupported.log" 2>&1; then
  echo 'FAIL: unsupported compiler accepted' >&2; exit 1
fi
printf 'BEAM_CUDA_ARCHITECTURES:STRING=103\nBEAM_STATE_LOGICAL_BYTES:STRING=96\n' > "$tmp/build/CMakeCache.txt"
bash hpc/cube4_portable.sh s1
grep -q 'micro=128 lanes=2' "$tmp"/cube4_s1.*/gpu0.md
if PORTABLE_WORLD_SIZE=2 BEAM_WIDTH=256000000 bash hpc/cube4_portable.sh run > "$tmp/underspecified.log" 2>&1; then
  echo 'FAIL: underspecified beam profile accepted' >&2; exit 1
fi
if PORTABLE_WORLD_SIZE=2 BEAM_WIDTH=256000000 BEAM_B_MICRO=128 BEAM_STREAM1_CONCURRENCY=2 \
   BEAM_STREAM3_RING_SLOTS=4 BEAM_SHARD_COUNT=32 BEAM_HISTORY_RAM_BYTES=999999999999999 \
   BEAM_HISTORY_DISK_BYTES=1048576 bash hpc/cube4_portable.sh run > "$tmp/overcommit.log" 2>&1; then
  echo 'FAIL: host RAM overcommit accepted' >&2; exit 1
fi
PORTABLE_WORLD_SIZE=2 BEAM_WIDTH=256000000 BEAM_B_MICRO=128 BEAM_STREAM1_CONCURRENCY=2 \
  BEAM_STREAM3_RING_SLOTS=4 BEAM_SHARD_COUNT=32 BEAM_HISTORY_RAM_BYTES=1048576 \
  BEAM_HISTORY_DISK_BYTES=1048576 bash hpc/cube4_portable.sh run > "$tmp/run.out"
run_dir=$(sed -n 's/^run_dir=\([^ ]*\).*/\1/p' "$tmp/run.out")
[[ -f "$run_dir/rank0.log" && -f "$run_dir/rank1.log" ]]
grep -q 'world=2 beam=256000000 cap=4000768 qkv=off' "$run_dir/rank0.log"
echo 'PASS: portable architecture, Cube4 build, Stream1, explicit beam profile; mock binaries only'
