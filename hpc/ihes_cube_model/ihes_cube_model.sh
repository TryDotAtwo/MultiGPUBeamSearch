#!/bin/bash
#SBATCH --job-name=ihes-model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/pool/6/vokirova/beam8a100/ihes_cube_model/logs/ihes-model-%j.out

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/pool/6/vokirova/beam8a100}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/repo}"
JOB_DIR="${JOB_DIR:-${BASE_DIR}/ihes_cube_model}"
RUN_DIR="${RUN_DIR:-${JOB_DIR}}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/data}"
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model.pth}"
WEIGHT_DIR="${WEIGHT_DIR:-${RUN_DIR}/stream1_weights_ihes_bf16}"
BUILD_DIR="${BUILD_DIR:-${RUN_DIR}/build-a100-${SLURM_JOB_ID:-manual}}"
HISTORY_DIR="${HISTORY_DIR:-${RUN_DIR}/history-${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
BEAM_COMMON_SH="${BEAM_COMMON_SH:-${REPO_DIR}/hpc/mephi_8xa100_common.sh}"

mkdir -p "${RUN_DIR}" "${DATA_DIR}" "${WEIGHT_DIR}" "${BUILD_DIR}" "${HISTORY_DIR}" "${LOG_DIR}"

if [ ! -f "${BEAM_COMMON_SH}" ]; then
  echo "missing_common_script=${BEAM_COMMON_SH}"
  exit 2
fi
source "${BEAM_COMMON_SH}"

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

beam_setup_paths
PUZZLE_ID="${PUZZLE_ID:-0}"
DEPTH_LIMIT="${DEPTH_LIMIT:-12}"
BEAM_WIDTH="${BEAM_WIDTH:-30000000}"

SHARD_COUNT="${SHARD_COUNT:-16}"
STREAM4_BATCH_ALIGNMENT="${STREAM4_BATCH_ALIGNMENT:-1024}"
SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1500000}"
STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES:-262144}"
STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES:-524288}"
BEAM_B_MICRO="${BEAM_B_MICRO:-8192}"
BEAM_STREAM1_CONCURRENCY="${BEAM_STREAM1_CONCURRENCY:-8}"
BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-8}"
if [ "${BEAM_STREAM3_RING_SLOTS}" -lt "${BEAM_STREAM1_CONCURRENCY}" ]; then
  BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM1_CONCURRENCY}"
fi
BEAM_STREAM4_ACTIVE_SORT_SLOTS="${BEAM_STREAM4_ACTIVE_SORT_SLOTS:-4}"
BEAM_SHARD_BUFFER_COUNT="${BEAM_SHARD_BUFFER_COUNT:-2}"
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-65536}"
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-2000000}"
BEAM_SOLVED_NEIGHBORHOOD_RADIUS="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS:-4}"
BEAM_GPU_HEADROOM_BYTES="${BEAM_GPU_HEADROOM_BYTES:-268435456}"
BEAM_HISTORY_RAM_BYTES="${BEAM_HISTORY_RAM_BYTES:-68719476736}"
BEAM_HISTORY_DISK_BYTES="${BEAM_HISTORY_DISK_BYTES:-274877906944}"

beam_preflight

if [ ! -f "${MODEL_PATH}" ]; then
  echo "missing_model=${MODEL_PATH}"
  echo "run_prepare_script=${REPO_DIR}/hpc/ihes_cube_model/prepare_ihes_cube_model.sh"
  exit 2
fi
if [ ! -f "${DATA_DIR}/puzzle_info.json" ] || [ ! -f "${DATA_DIR}/test.csv" ]; then
  echo "missing_ihes_data=${DATA_DIR}"
  echo "run_prepare_script=${REPO_DIR}/hpc/ihes_cube_model/prepare_ihes_cube_model.sh"
  exit 2
fi
if [ ! -f "${WEIGHT_DIR}/manifest.json" ]; then
  echo "missing_stream1_weights=${WEIGHT_DIR}/manifest.json"
  echo "run_prepare_script=${REPO_DIR}/hpc/ihes_cube_model/prepare_ihes_cube_model.sh"
  exit 2
fi

export BEAM_PUZZLE_INFO_JSON="${DATA_DIR}/puzzle_info.json"
export BEAM_GENERATOR_PATH="${DATA_DIR}/puzzle_info.json"
export BEAM_TEST_CSV="${DATA_DIR}/test.csv"
export BEAM_WEIGHT_DIR="${WEIGHT_DIR}"

beam_configure_build production_runner
beam_derive_shard_capacity
beam_validate_manual_config

echo "puzzle_id=${PUZZLE_ID}"
echo "depth_limit=${DEPTH_LIMIT}"
echo "beam_width=${BEAM_WIDTH}"
echo "global_beam_width_effective=${GLOBAL_BEAM_WIDTH_EFFECTIVE}"
echo "local_beam_width=${LOCAL_BEAM_WIDTH}"
echo "move_count_effective=${BEAM_MOVE_COUNT_EFFECTIVE}"
echo "shard_count=${SHARD_COUNT}"
echo "stream1_output_dim=${STREAM1_OUTPUT_DIM}"
echo "beam_b_micro_row_budget=${BEAM_B_MICRO}"
echo "beam_parent_batch_effective=${BEAM_PARENT_BATCH_EFFECTIVE}"
echo "stream1_rows_per_job_effective=${STREAM1_ROWS_PER_JOB_EFFECTIVE}"
echo "logical_shard_size=${LOGICAL_SHARD_SIZE}"
echo "shard_capacity_candidates=${SHARD_CAPACITY_CANDIDATES}"
echo "stream3_batch_candidates=${STREAM3_BATCH_CANDIDATES}"
echo "stream4_batch_candidates=${STREAM4_BATCH_CANDIDATES}"
echo "stream4_trigger_candidates=${STREAM4_TRIGGER_CANDIDATES}"
echo "final_chunk_candidates=${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}"
echo "final_exchange_scale_ppm=${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM}"

beam_export_common_runtime
export BEAM_PREDICT_STATS_VERBOSE="${BEAM_PREDICT_STATS_VERBOSE:-1}"
export BEAM_PREDICT_STATS_PATH="${BEAM_PREDICT_STATS_PATH:-${RUN_DIR}/predict_stats_p${PUZZLE_ID}_b${BEAM_WIDTH}_d${DEPTH_LIMIT}.jsonl}"
export BEAM_B_MICRO
export BEAM_STREAM1_CONCURRENCY
export BEAM_STREAM3_RING_SLOTS
export BEAM_SHARD_BUFFER_COUNT
beam_export_manual_config
beam_prepare_nccl_file "ihes"

RUN_LOG="${LOG_DIR}/production_runner_ihes_p${PUZZLE_ID}_d${DEPTH_LIMIT}_b${BEAM_WIDTH}_${SLURM_JOB_ID:-manual}.log"
beam_torchrun_production "ihes" "${RUN_LOG}"
echo "finished_at=$(date -Is)"
