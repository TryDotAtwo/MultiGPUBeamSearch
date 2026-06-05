#!/bin/bash
#SBATCH --job-name=beam8a100-best
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${SCRIPT_DIR}/mephi_8xa100_common.sh"

beam_setup_paths
PUZZLE_ID="${PUZZLE_ID:-992}"
DEPTH_LIMIT="${DEPTH_LIMIT:-80}"
BEAM_WIDTH="${BEAM_WIDTH:-260000000}"

BEST_CONFIG_ENV="${BEST_CONFIG_ENV:-${LOG_DIR}/best_pipeline.env}"
if [ -f "${BEST_CONFIG_ENV}" ]; then
  source "${BEST_CONFIG_ENV}"
fi

SHARD_COUNT="${SHARD_COUNT:-16}"
STREAM4_BATCH_ALIGNMENT="${STREAM4_BATCH_ALIGNMENT:-1024}"
SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1250000}"
STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES:-524288}"
STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES:-1048576}"
BEAM_B_MICRO="${BEAM_B_MICRO:-8192}"
BEAM_STREAM1_CONCURRENCY="${BEAM_STREAM1_CONCURRENCY:-8}"
BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-8}"
if [ "${BEAM_STREAM3_RING_SLOTS}" -lt "${BEAM_STREAM1_CONCURRENCY}" ]; then
  BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM1_CONCURRENCY}"
fi
BEAM_STREAM4_ACTIVE_SORT_SLOTS="${BEAM_STREAM4_ACTIVE_SORT_SLOTS:-4}"
BEAM_SHARD_BUFFER_COUNT="${BEAM_SHARD_BUFFER_COUNT:-2}"

cleanup() {
  local rc=$?
  echo "cleanup_start rc=${rc} at $(date -Is)"
  beam_safe_clean_child "${BUILD_DIR}" "build"
  if [ "${rc}" -eq 0 ]; then
    beam_safe_clean_child "${HISTORY_DIR}" "history"
  else
    echo "cleanup_keep_history=${HISTORY_DIR}"
  fi
  echo "cleanup_done rc=${rc} at $(date -Is)"
  exit "${rc}"
}
trap cleanup EXIT

beam_preflight
beam_configure_build production_runner
beam_derive_shard_capacity
beam_validate_manual_config

echo "beam_width=${BEAM_WIDTH}"
echo "global_beam_width_effective=${GLOBAL_BEAM_WIDTH_EFFECTIVE}"
echo "local_beam_width=${LOCAL_BEAM_WIDTH}"
echo "shard_count=${SHARD_COUNT}"
echo "logical_shard_size=${LOGICAL_SHARD_SIZE}"
echo "shard_capacity_candidates=${SHARD_CAPACITY_CANDIDATES}"
echo "stream3_batch_candidates=${STREAM3_BATCH_CANDIDATES}"
echo "stream4_batch_candidates=${STREAM4_BATCH_CANDIDATES}"
echo "stream4_trigger_candidates=${STREAM4_TRIGGER_CANDIDATES}"

beam_export_common_runtime
export BEAM_PREDICT_STATS_VERBOSE="${BEAM_PREDICT_STATS_VERBOSE:-1}"
export BEAM_PREDICT_STATS_PATH="${BEAM_PREDICT_STATS_PATH:-${JOB_DIR}/predict_stats_p${PUZZLE_ID}_b${BEAM_WIDTH}_d${DEPTH_LIMIT}.jsonl}"
export BEAM_B_MICRO
export BEAM_STREAM1_CONCURRENCY
export BEAM_STREAM3_RING_SLOTS
export BEAM_SHARD_BUFFER_COUNT
beam_export_manual_config
beam_prepare_nccl_file "best"

RUN_LOG="${LOG_DIR}/production_runner_p${PUZZLE_ID}_d${DEPTH_LIMIT}_b${BEAM_WIDTH}_${SLURM_JOB_ID:-manual}.log"
beam_torchrun_production "best" "${RUN_LOG}"
echo "finished_at=$(date -Is)"
