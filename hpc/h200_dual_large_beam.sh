#!/usr/bin/env bash
# Single-node H200 launcher; defaults preserve the measured two-rank profile.
set -euo pipefail
cd "${H200_REPO_DIR:-/workspace/MGBFS}"
world=${H200_WORLD_SIZE:-2}
[[ "$world" =~ ^[1-8]$ ]] || { echo 'H200_WORLD_SIZE must be 1..8' >&2; exit 2; }
export H200_WORLD_SIZE="$world"
depth=${DEPTH_LIMIT:-9}
[[ "$depth" =~ ^[1-9][0-9]*$ && ${#depth} -le 10 ]] && ((depth <= 4294967295)) || {
  echo 'invalid DEPTH_LIMIT: expected integer 1..4294967295' >&2
  exit 2
}
run_dir=$(mktemp -d "${H200_RUN_ROOT:-/workspace}/h200_run.XXXXXX")
echo "run_dir=$run_dir"
export BEAM_NCCL_ID_FILE="$run_dir/nccl.bin"
export BEAM_NCCL_RUN_ID="$run_dir"
export BEAM_HISTORY_DIR="$run_dir/history"
export H200_DATA_DIR=${H200_DATA_DIR:-/workspace}
export BEAM_WEIGHT_DIR=${BEAM_WEIGHT_DIR:-/workspace/stream1_transformer_weights_fp16}
export BEAM_GENERATOR_PATH="$H200_DATA_DIR/puzzle_info.json"
export BEAM_PUZZLE_INFO_JSON="$H200_DATA_DIR/puzzle_info.json" BEAM_TEST_CSV="$H200_DATA_DIR/test.csv"
export BEAM_STREAM1_TRANSFORMER_HOPPER=off
export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=fp16_tma BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=fp16_tma
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF2=fp16_tma BEAM_STREAM1_TRANSFORMER_HOPPER_FF1_EPILOGUE=128x64
export BEAM_STREAM1_TRANSFORMER_FF2_POLICY=m128n128 BEAM_STREAM1_TRANSFORMER_COMPACT57=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1 BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
export BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN=0 BEAM_STREAM1_EXECUTOR=native_cuda_graph
export BEAM_RUNTIME_CONFIG_MODE=manual
export BEAM_B_MICRO=${BEAM_B_MICRO:-384} BEAM_STREAM1_CONCURRENCY=${BEAM_STREAM1_CONCURRENCY:-8}
# Mirror runtime_config's default explicitly for profile/environment binding.
export BEAM_STREAM1_TRANSFORMER_MICRO=${BEAM_STREAM1_TRANSFORMER_MICRO:-$BEAM_B_MICRO}
export BEAM_STREAM3_RING_SLOTS=${BEAM_STREAM3_RING_SLOTS:-12} BEAM_RING_GRAPH_EXECS_PER_LANE=32
export BEAM_SHARD_COUNT=${BEAM_SHARD_COUNT:-32} BEAM_SHARD_BUFFER_COUNT=2
export BEAM_STREAM4_BATCH_ALIGNMENT=1024
export BEAM_STREAM4_BATCH_CANDIDATES=${BEAM_STREAM4_BATCH_CANDIDATES:-262144}
export BEAM_STREAM4_TRIGGER_CANDIDATES=${BEAM_STREAM4_TRIGGER_CANDIDATES:-524288}
export BEAM_STREAM4_ACTIVE_SORT_SLOTS=${BEAM_STREAM4_ACTIVE_SORT_SLOTS:-4}
export BEAM_SHARD_CAPACITY_SCALE_PPM=1000000
beam=${BEAM_WIDTH:?set requested global beam}
export BEAM_WIDTH="$beam"
[[ "$beam" =~ ^[1-9][0-9]*$ && ${#beam} -le 10 ]] || { echo 'invalid BEAM_WIDTH' >&2; exit 2; }
[[ "$BEAM_SHARD_COUNT" =~ ^[1-9][0-9]*$ && ${#BEAM_SHARD_COUNT} -le 4 ]] || exit 2
export BEAM_SHARD_CAPACITY_CANDIDATES=$(( ((beam+world*BEAM_SHARD_COUNT*1024-1)/(world*BEAM_SHARD_COUNT*1024))*1024 ))
export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=65536 BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=$((world*1000000))
echo "world_size=$world requested_beam=$beam effective_beam=$((BEAM_SHARD_CAPACITY_CANDIDATES*world*BEAM_SHARD_COUNT)) shard_capacity=$BEAM_SHARD_CAPACITY_CANDIDATES"
export BEAM_GPU_HEADROOM_BYTES=4294967296
export BEAM_HISTORY_MODE=static_hybrid
export BEAM_HISTORY_CHUNKED_PIN=${BEAM_HISTORY_CHUNKED_PIN:-1}
# Global budgets, divided by WORLD_SIZE by the runner.
export BEAM_HISTORY_RAM_BYTES=${BEAM_HISTORY_RAM_BYTES:-137438953472}
export BEAM_HISTORY_DISK_BYTES=${BEAM_HISTORY_DISK_BYTES:-644245094400}
export BEAM_SOLVED_NEIGHBORHOOD_RADIUS=4 BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES=3000000
export BEAM_ENABLE_DEBUG=1 BEAM_DEPTH_LOG_EVERY=1 WORLD_SIZE=$world NCCL_DEBUG=WARN
unset BEAM_HOPPER_NATIVE_FP8
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64"
if [[ -n ${H200_PROFILE_PATH:-} ]]; then
  export H200_RESOLVED_PROFILE="$run_dir/resolved_profile.json"
  "${H200_PYTHON:-python3}" tools/cube4_profile.py --profile "$H200_PROFILE_PATH" --environment > "$H200_RESOLVED_PROFILE"
  cat "$H200_RESOLVED_PROFILE"
fi
pids=()
"${H200_PYTHON:-python3}" tools/verify_cube4_bundle.py "$H200_DATA_DIR" "$BEAM_WEIGHT_DIR" "${PUZZLE_ID:-1000}" > "$run_dir/bundle_preflight.log" 2>&1 || {
  cat "$run_dir/bundle_preflight.log" >&2
  exit 2
}
cat "$run_dir/bundle_preflight.log"
runner="${H200_BUILD_DIR:-/workspace/build}/production_runner"
"$runner" --build-info > "$run_dir/build_info.json"
"${H200_SMI:-nvidia-smi}" --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits > "$run_dir/hardware.csv"
"${H200_PYTHON:-python3}" tools/cube4_build_contract.py "$runner" "$run_dir/build_info.json" \
  --hardware-report "$run_dir/hardware.csv" --world-size "$world" > "$run_dir/build_preflight.json"
cat "$run_dir/build_preflight.json"
"${H200_PYTHON:-python3}" tools/nccl_library_observation.py "$runner" \
  > "$run_dir/nccl_library_observation.json"
"${H200_PYTHON:-python3}" tools/stream1_execution_environment.py --source-root . \
  > "$run_dir/stream1_execution_environment.json"
"${H200_PYTHON:-python3}" tools/host_memory_budget.py \
  --meminfo "${H200_MEMINFO_PATH:-/proc/meminfo}" \
  --cgroup-root "${BEAM_HOST_CGROUP_ROOT:-/sys/fs/cgroup}" \
  --history-ram-bytes "$BEAM_HISTORY_RAM_BYTES" \
  --headroom-bytes "${BEAM_HOST_RAM_HEADROOM_BYTES:-$(((world+2)*1073741824))}" \
  > "$run_dir/host_memory_preflight.json"
"${H200_PYTHON:-python3}" tools/host_disk_budget.py \
  --path "$run_dir" --history-disk-bytes "$BEAM_HISTORY_DISK_BYTES" \
  --headroom-bytes "${BEAM_HOST_DISK_HEADROOM_BYTES:-1073741824}" \
  > "$run_dir/disk_preflight.json"
# Fresh evidence only; incomplete numerical candidates are not admission.
# This currently fails closed until the independent coverage gate is complete.
"${H200_PYTHON:-python3}" tools/cube4_production_preflight.py \
  --runner "$runner" --probe "${H200_BUILD_DIR:-/workspace/build}/cuda_visible_device_probe" \
  --reference-dir "${BEAM_NUMERICAL_REFERENCE_DIR:-$H200_DATA_DIR/cube4_reference}" \
  --source-root "$PWD" --world-size "$world" \
  --output-root "$run_dir/numerical_preflight" > "$run_dir/numerical_preflight.log" 2>&1 || {
  cat "$run_dir/numerical_preflight.log" >&2
  exit 2
}
cleanup(){ for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
for ((rank=0; rank<world; rank++)); do
  RANK=$rank LOCAL_RANK=$rank "${H200_BUILD_DIR:-/workspace/build}/production_runner" \
    "${PUZZLE_ID:-1000}" "$depth" "$beam" > "$run_dir/rank${rank}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for ((remaining=world; remaining>0; remaining--)); do
  if wait -n; then :; else rc=$?; cleanup; break; fi
done
trap - EXIT INT TERM
echo "exit_code=$rc"
exit "$rc"
