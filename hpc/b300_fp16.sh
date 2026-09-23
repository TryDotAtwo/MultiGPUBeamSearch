#!/usr/bin/env bash
# B300 SM103 bring-up: native FP16 only. The Hopper TMA paths are intentionally disabled.
set -euo pipefail
mode=${1:-}
[[ "$mode" == s1 || "$mode" == run ]] || { echo 'usage: b300_fp16.sh {s1|run}' >&2; exit 2; }
world=${B300_WORLD_SIZE:-1}
[[ "$world" == 1 || "$world" == 2 ]] || { echo 'B300_WORLD_SIZE must be 1 or 2' >&2; exit 2; }
cd "${B300_REPO_DIR:-/workspace/MGBFS}"
root=${B300_RUN_ROOT:-/workspace}
build=${B300_BUILD_DIR:-/workspace/build-b300}
export BEAM_WEIGHT_DIR=${B300_WEIGHT_DIR:-/workspace/stream1_transformer_weights_fp16}
data=${B300_DATA_DIR:-/workspace}
export BEAM_GENERATOR_PATH="$data/puzzle_info.json" BEAM_PUZZLE_INFO_JSON="$data/puzzle_info.json"
export BEAM_TEST_CSV="$data/test.csv"
[[ -f "$data/puzzle_info.json" && -f "$data/test.csv" ]] || {
  echo "missing puzzle inputs in $data" >&2; exit 2;
}
[[ -f "$BEAM_WEIGHT_DIR/manifest.json" ]] || {
  echo "missing FP16 weight manifest in $BEAM_WEIGHT_DIR" >&2; exit 2;
}
[[ -x "$build/stream_benchmark" && -x "$build/production_runner" ]] || {
  echo "missing B300 executables in $build" >&2; exit 2;
}
export BEAM_STREAM1_TRANSFORMER_HOPPER=off
export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=off
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=off
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF2=off
unset BEAM_HOPPER_NATIVE_FP8
export BEAM_STREAM1_TRANSFORMER_COMPACT57=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1 BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
export BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN=0 BEAM_STREAM1_EXECUTOR=native_cuda_graph
export BEAM_STREAM1_TRANSFORMER_FF2_POLICY=m128n128

if [[ "$mode" == s1 ]]; then
  result_dir=$(mktemp -d "$root/b300_s1.XXXXXX")
  echo "result_dir=$result_dir"
  export BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1
  export BEAM_STREAM1_TRANSFORMER_B_MICRO=${B300_S1_MICRO:-384}
  export BEAM_STREAM1_TRANSFORMER_CONCURRENCY=${B300_S1_LANES:-8}
  export BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=${B300_S1_ITERS:-300}
  pids=()
  for ((gpu=0; gpu<world; gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu BEAM_STREAM_BENCH_REPORT="$result_dir/gpu${gpu}.md" \
      timeout "${B300_S1_TIMEOUT_SEC:-180}" "$build/stream_benchmark" "${PUZZLE_ID:-1000}" \
      > "$result_dir/gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
else
  export BEAM_RUNTIME_CONFIG_MODE=manual
  export BEAM_B_MICRO=${BEAM_B_MICRO:-384} BEAM_STREAM1_CONCURRENCY=${BEAM_STREAM1_CONCURRENCY:-8}
  export BEAM_STREAM3_RING_SLOTS=${BEAM_STREAM3_RING_SLOTS:-12} BEAM_RING_GRAPH_EXECS_PER_LANE=32
  export BEAM_SHARD_COUNT=${BEAM_SHARD_COUNT:-32} BEAM_SHARD_BUFFER_COUNT=2
  export BEAM_STREAM4_BATCH_ALIGNMENT=1024
  export BEAM_STREAM4_BATCH_CANDIDATES=${BEAM_STREAM4_BATCH_CANDIDATES:-262144}
  export BEAM_STREAM4_TRIGGER_CANDIDATES=${BEAM_STREAM4_TRIGGER_CANDIDATES:-524288}
  export BEAM_STREAM4_ACTIVE_SORT_SLOTS=${BEAM_STREAM4_ACTIVE_SORT_SLOTS:-4}
  export BEAM_SHARD_CAPACITY_SCALE_PPM=1000000
  beam=${BEAM_WIDTH:-256000000}
  [[ "$beam" =~ ^[1-9][0-9]*$ && ${#beam} -le 10 ]] || { echo 'invalid BEAM_WIDTH' >&2; exit 2; }
  [[ "$BEAM_SHARD_COUNT" =~ ^[1-9][0-9]*$ && ${#BEAM_SHARD_COUNT} -le 4 ]] || exit 2
  export BEAM_SHARD_CAPACITY_CANDIDATES=$(( ((beam+world*BEAM_SHARD_COUNT*1024-1)/(world*BEAM_SHARD_COUNT*1024))*1024 ))
  export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=65536
  export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=$((world*1000000))
  export BEAM_GPU_HEADROOM_BYTES=4294967296
  export BEAM_HISTORY_MODE=static_hybrid
  export BEAM_HISTORY_CHUNKED_PIN=1
  export BEAM_HISTORY_RAM_BYTES=${BEAM_HISTORY_RAM_BYTES:-137438953472}
  export BEAM_HISTORY_DISK_BYTES=${BEAM_HISTORY_DISK_BYTES:-68719476736}
  export BEAM_SOLVED_NEIGHBORHOOD_RADIUS=4 BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES=3000000
  export BEAM_ENABLE_DEBUG=1 BEAM_DEPTH_LOG_EVERY=1 WORLD_SIZE=$world NCCL_DEBUG=WARN
  run_dir=$(mktemp -d "$root/b300_run.XXXXXX")
  echo "run_dir=$run_dir world_size=$world requested_beam=$beam effective_beam=$((BEAM_SHARD_CAPACITY_CANDIDATES*world*BEAM_SHARD_COUNT))"
  export BEAM_NCCL_ID_FILE="$run_dir/nccl.bin" BEAM_HISTORY_DIR="$run_dir/history"
  pids=()
  for ((rank=0; rank<world; rank++)); do
    RANK=$rank LOCAL_RANK=$rank "$build/production_runner" \
      "${PUZZLE_ID:-1000}" "${DEPTH_LIMIT:-9}" "$beam" > "$run_dir/rank${rank}.log" 2>&1 &
    pids+=("$!")
  done
fi

cleanup(){ for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
rc=0
for p in "${pids[@]}"; do
  wait "$p" || { rc=$?; cleanup; break; }
done
trap - EXIT INT TERM
echo "exit_code=$rc"
exit "$rc"
