#!/usr/bin/env bash
# Native FP16 Cube4 entry point for a *detected*, homogeneous NVIDIA node.
set -euo pipefail
mode=${1:-}
[[ "$mode" == probe || "$mode" == build || "$mode" == s1 || "$mode" == run ]] || {
  echo 'usage: cube4_portable.sh {probe|build|s1|run}' >&2; exit 2;
}
repo=${PORTABLE_REPO_DIR:-/workspace/MGBFS}
data=${PORTABLE_DATA_DIR:-/workspace}
weights=${PORTABLE_WEIGHT_DIR:-/workspace/stream1_transformer_weights_fp16}
build=${PORTABLE_BUILD_DIR:-/workspace/build-cube4}
root=${PORTABLE_RUN_ROOT:-/workspace}
nvcc=${PORTABLE_NVCC:-/usr/local/cuda/bin/nvcc}
smi=${PORTABLE_SMI:-nvidia-smi}
cmake=${PORTABLE_CMAKE:-cmake}
python=${PORTABLE_PYTHON:-python3}

command -v "$smi" >/dev/null || { echo "nvidia-smi unavailable: $smi" >&2; exit 2; }
mapfile -t caps < <("$smi" --query-gpu=compute_cap --format=csv,noheader,nounits | sed 's/[[:space:]]//g')
visible=${#caps[@]}
(( visible > 0 )) || { echo 'no visible NVIDIA GPUs' >&2; exit 2; }
world=${PORTABLE_WORLD_SIZE:-1}
[[ "$world" =~ ^[1-9][0-9]*$ ]] && (( world <= visible )) || {
  echo "PORTABLE_WORLD_SIZE=$world exceeds visible GPUs=$visible" >&2; exit 2;
}
cap=${caps[0]}
[[ "$cap" =~ ^([0-9]+)\.([0-9])$ ]] || { echo "bad compute capability: $cap" >&2; exit 2; }
arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
for ((i=1; i<world; i++)); do
  [[ "${caps[i]}" == "$cap" ]] || { echo 'mixed GPU architectures require separate profiles' >&2; exit 2; }
done
if [[ "$mode" == probe ]]; then
  echo "arch=$arch visible_gpus=$visible selected_gpus=$world"
  "$smi" --query-gpu=name,memory.total --format=csv,noheader,nounits
  df -PB1 "$root"
  exit 0
fi

[[ -f "$data/puzzle_info.json" && -f "$data/test.csv" && -f "$weights/manifest.json" ]] || {
  echo 'missing Cube4 puzzle_info.json, test.csv, or FP16 manifest' >&2; exit 2;
}
"$python" "$repo/tools/verify_cube4_bundle.py" "$data" "$weights" "${PUZZLE_ID:-1000}"

if [[ "$mode" == build ]]; then
  command -v "$nvcc" >/dev/null || { echo "nvcc unavailable: $nvcc" >&2; exit 2; }
  "$nvcc" --list-gpu-arch | grep -Fxq "compute_$arch" || {
    echo "CUDA Toolkit cannot compile compute_$arch" >&2; exit 2;
  }
  cutlass=${PORTABLE_CUTLASS_DIR:-/workspace/cutlass}
  nccl_include=${PORTABLE_NCCL_INCLUDE_DIR:-/usr/include}
  nccl_library=${PORTABLE_NCCL_LIBRARY:-/usr/lib/x86_64-linux-gnu/libnccl.so}
  [[ -d "$cutlass/include" && -f "$nccl_include/nccl.h" && -f "$nccl_library" ]] || {
    echo 'CUTLASS or NCCL headers/library missing' >&2; exit 2;
  }
  "$cmake" -S "$repo" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$nvcc" \
    -DBEAM_CUDA_ARCHITECTURES="$arch" -DBEAM_STATE_LOGICAL_BYTES=96 -DBEAM_MOVE_COUNT=24 \
    -DBEAM_PUZZLE_INFO_JSON="$data/puzzle_info.json" \
    -DCUTLASS_DIR="$cutlass" -DNCCL_INCLUDE_DIR="$nccl_include" -DNCCL_LIBRARY="$nccl_library" \
    -DBEAM_ENABLE_DEBUG=ON -DBEAM_ENABLE_DEPTH_LOGS=ON
  "$cmake" --build "$build" --target production_runner stream_benchmark \
    stream1_transformer_cuda_tests -j "${PORTABLE_BUILD_JOBS:-8}"
  exit 0
fi

[[ -f "$build/CMakeCache.txt" ]] &&
grep -Fxq "BEAM_CUDA_ARCHITECTURES:STRING=$arch" "$build/CMakeCache.txt" &&
grep -Fxq 'BEAM_STATE_LOGICAL_BYTES:STRING=96' "$build/CMakeCache.txt" || {
  echo "build cache is not Cube4 state96 for compute_$arch" >&2; exit 2;
}
[[ -x "$build/production_runner" && -x "$build/stream_benchmark" ]] || {
  echo "missing production_runner or stream_benchmark in $build" >&2; exit 2;
}
export BEAM_WEIGHT_DIR="$weights" BEAM_GENERATOR_PATH="$data/puzzle_info.json"
export BEAM_PUZZLE_INFO_JSON="$data/puzzle_info.json" BEAM_TEST_CSV="$data/test.csv"
export BEAM_STREAM1_TRANSFORMER_HOPPER=off BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=off
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=off BEAM_STREAM1_TRANSFORMER_HOPPER_FF2=off
unset BEAM_HOPPER_NATIVE_FP8
export BEAM_STREAM1_TRANSFORMER_COMPACT57=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1
export BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1 BEAM_STREAM1_EXECUTOR=native_cuda_graph

if [[ "$mode" == s1 ]]; then
  gpu=${PORTABLE_GPU_INDEX:-0}
  [[ "$gpu" =~ ^[0-9]+$ ]] && (( gpu < visible )) || {
    echo "PORTABLE_GPU_INDEX=$gpu is outside visible GPUs=$visible" >&2; exit 2;
  }
  export BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1
  export BEAM_STREAM1_TRANSFORMER_B_MICRO=${PORTABLE_S1_MICRO:-128}
  export BEAM_STREAM1_TRANSFORMER_CONCURRENCY=${PORTABLE_S1_LANES:-2}
  export BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=${PORTABLE_S1_ITERS:-300}
  result_dir=$(mktemp -d "$root/cube4_s1.XXXXXX")
  echo "result_dir=$result_dir arch=$arch"
  CUDA_VISIBLE_DEVICES=$gpu BEAM_STREAM_BENCH_REPORT="$result_dir/gpu${gpu}.md" \
    timeout "${PORTABLE_S1_TIMEOUT_SEC:-180}" "$build/stream_benchmark" "${PUZZLE_ID:-1000}" \
    > "$result_dir/gpu${gpu}.log" 2>&1
  exit 0
fi

# A search profile cannot safely be inferred from GPU name or memory alone.
for name in BEAM_WIDTH BEAM_B_MICRO BEAM_STREAM1_CONCURRENCY BEAM_STREAM3_RING_SLOTS \
            BEAM_SHARD_COUNT BEAM_HISTORY_RAM_BYTES BEAM_HISTORY_DISK_BYTES; do
  [[ "${!name:-}" =~ ^[1-9][0-9]*$ ]] || { echo "set positive $name from a measured profile" >&2; exit 2; }
done
(( BEAM_STREAM3_RING_SLOTS > BEAM_STREAM1_CONCURRENCY )) || {
  echo 'ring slots must exceed Stream1 concurrency' >&2; exit 2;
}
(( BEAM_WIDTH <= 5000000000 && BEAM_SHARD_COUNT <= 256 )) || {
  echo 'beam or shard count exceeds launcher bounds' >&2; exit 2;
}
export BEAM_RUNTIME_CONFIG_MODE=manual BEAM_HISTORY_MODE=static_hybrid BEAM_HISTORY_CHUNKED_PIN=1
export BEAM_STREAM4_BATCH_ALIGNMENT=1024 BEAM_SHARD_BUFFER_COUNT=2
export BEAM_SHARD_CAPACITY_SCALE_PPM=1000000
export BEAM_SHARD_CAPACITY_CANDIDATES=$(( ((BEAM_WIDTH+world*BEAM_SHARD_COUNT*1024-1)/(world*BEAM_SHARD_COUNT*1024))*1024 ))
export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=$((world*1000000))
export BEAM_GPU_HEADROOM_BYTES=${BEAM_GPU_HEADROOM_BYTES:-4294967296}
export BEAM_ENABLE_DEBUG=1 BEAM_DEPTH_LOG_EVERY=1 WORLD_SIZE=$world NCCL_DEBUG=WARN
available_disk=$(df -PB1 "$root" | awk 'NR==2 {print $4}')
available_ram_kib=$(awk '/^MemAvailable:/ {print $2}' "${PORTABLE_MEMINFO_PATH:-/proc/meminfo}")
[[ "$available_disk" =~ ^[0-9]+$ && "$available_ram_kib" =~ ^[0-9]+$ ]] || {
  echo 'cannot read free disk or RAM' >&2; exit 2;
}
(( BEAM_HISTORY_DISK_BYTES < available_disk )) || { echo 'insufficient disk for requested history' >&2; exit 2; }
(( BEAM_HISTORY_RAM_BYTES < available_ram_kib*1024 )) || { echo 'insufficient host RAM for requested history' >&2; exit 2; }
run_dir=$(mktemp -d "$root/cube4_run.XXXXXX")
export BEAM_NCCL_ID_FILE="$run_dir/nccl.bin" BEAM_HISTORY_DIR="$run_dir/history"
echo "run_dir=$run_dir arch=$arch world_size=$world requested_beam=$BEAM_WIDTH effective_beam=$((BEAM_SHARD_CAPACITY_CANDIDATES*world*BEAM_SHARD_COUNT))"
pids=()
cleanup(){ for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
for ((rank=0; rank<world; rank++)); do
  RANK=$rank LOCAL_RANK=$rank "$build/production_runner" \
    "${PUZZLE_ID:-1000}" "${DEPTH_LIMIT:-9}" "$BEAM_WIDTH" > "$run_dir/rank${rank}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for ((remaining=world; remaining>0; remaining--)); do
  if wait -n; then :; else rc=$?; cleanup; break; fi
done
trap - EXIT INT TERM
echo "exit_code=$rc"
exit "$rc"
