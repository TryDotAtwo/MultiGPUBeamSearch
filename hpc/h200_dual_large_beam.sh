#!/usr/bin/env bash
# Single-node H200 launcher; defaults preserve the measured two-rank profile.
set -euo pipefail
cd "${H200_REPO_DIR:-/workspace/MGBFS}"
world=${H200_WORLD_SIZE:-2}
[[ "$world" =~ ^[1-8]$ ]] || { echo 'H200_WORLD_SIZE must be 1..8' >&2; exit 2; }
run_dir=$(mktemp -d "${H200_RUN_ROOT:-/workspace}/h200_run.XXXXXX")
echo "run_dir=$run_dir"
export BEAM_NCCL_ID_FILE="$run_dir/nccl.bin"
export BEAM_HISTORY_DIR="$run_dir/history"
export BEAM_WEIGHT_DIR=/workspace/stream1_transformer_weights_fp16
export BEAM_GENERATOR_PATH=/workspace/puzzle_info.json
export BEAM_PUZZLE_INFO_JSON=/workspace/puzzle_info.json BEAM_TEST_CSV=/workspace/test.csv
export BEAM_STREAM1_TRANSFORMER_HOPPER=off
export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=fp16_tma BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=fp16_tma
export BEAM_STREAM1_TRANSFORMER_HOPPER_FF2=fp16_tma BEAM_STREAM1_TRANSFORMER_HOPPER_FF1_EPILOGUE=128x64
export BEAM_STREAM1_TRANSFORMER_FF2_POLICY=m128n128 BEAM_STREAM1_TRANSFORMER_COMPACT57=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1 BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
export BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN=0 BEAM_STREAM1_EXECUTOR=native_cuda_graph
export BEAM_RUNTIME_CONFIG_MODE=manual
export BEAM_B_MICRO=${BEAM_B_MICRO:-384} BEAM_STREAM1_CONCURRENCY=${BEAM_STREAM1_CONCURRENCY:-8}
export BEAM_STREAM3_RING_SLOTS=${BEAM_STREAM3_RING_SLOTS:-12} BEAM_RING_GRAPH_EXECS_PER_LANE=32
export BEAM_SHARD_COUNT=${BEAM_SHARD_COUNT:-32} BEAM_SHARD_BUFFER_COUNT=2
export BEAM_STREAM4_BATCH_ALIGNMENT=1024
export BEAM_STREAM4_BATCH_CANDIDATES=${BEAM_STREAM4_BATCH_CANDIDATES:-262144}
export BEAM_STREAM4_TRIGGER_CANDIDATES=${BEAM_STREAM4_TRIGGER_CANDIDATES:-524288}
export BEAM_STREAM4_ACTIVE_SORT_SLOTS=${BEAM_STREAM4_ACTIVE_SORT_SLOTS:-4}
export BEAM_SHARD_CAPACITY_SCALE_PPM=1000000
beam=${BEAM_WIDTH:?set requested global beam}
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
export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64
pids=()
cleanup(){ for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM
for ((rank=0; rank<world; rank++)); do
  RANK=$rank LOCAL_RANK=$rank "${H200_BUILD_DIR:-/workspace/build}/production_runner" \
    "${PUZZLE_ID:-1000}" "${DEPTH_LIMIT:-9}" "$beam" > "$run_dir/rank${rank}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for ((remaining=world; remaining>0; remaining--)); do
  if wait -n; then :; else rc=$?; cleanup; break; fi
done
trap - EXIT INT TERM
echo "exit_code=$rc"
exit "$rc"
