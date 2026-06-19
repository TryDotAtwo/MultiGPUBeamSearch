#!/bin/bash
#SBATCH --job-name=ihes-segment-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail
ulimit -c 0

BASE_DIR="${BASE_DIR:-/mnt/pool/6/vokirova/beam8a100}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/repo}"
JOB_DIR="${JOB_DIR:-${BASE_DIR}/ihes_cube_model}"
RUN_DIR="${RUN_DIR:-${JOB_DIR}}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/data}"
REPAIR_TEST_CSV="${REPAIR_TEST_CSV:-${RUN_DIR}/segment_repair_test.csv}"
REPAIR_JOBS_CSV="${REPAIR_JOBS_CSV:-${RUN_DIR}/segment_repair_jobs.csv}"
MODEL_PATH="${MODEL_PATH:-${RUN_DIR}/model.pth}"
WEIGHT_DIR="${WEIGHT_DIR:-${RUN_DIR}/stream1_weights_ihes_bf16}"
BUILD_DIR="${BUILD_DIR:-${RUN_DIR}/build-a100-${SLURM_JOB_ID:-manual}}"
HISTORY_DIR="${HISTORY_DIR:-${RUN_DIR}/history-${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
BEAM_COMMON_SH="${BEAM_COMMON_SH:-${REPO_DIR}/hpc/mephi_8xa100_common.sh}"

mkdir -p "${RUN_DIR}" "${BUILD_DIR}" "${HISTORY_DIR}" "${LOG_DIR}"

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

if [ ! -f "${REPAIR_TEST_CSV}" ]; then
  echo "missing_repair_test_csv=${REPAIR_TEST_CSV}"
  exit 2
fi
if [ ! -f "${REPAIR_JOBS_CSV}" ]; then
  echo "missing_repair_jobs_csv=${REPAIR_JOBS_CSV}"
  exit 2
fi
if [ ! -f "${DATA_DIR}/puzzle_info.json" ]; then
  echo "missing_ihes_puzzle_info=${DATA_DIR}/puzzle_info.json"
  exit 2
fi
if [ ! -f "${WEIGHT_DIR}/manifest.json" ]; then
  echo "missing_stream1_weights=${WEIGHT_DIR}/manifest.json"
  exit 2
fi

REPAIR_ROW_INDEX="${REPAIR_ROW_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"
REPAIR_ID="${REPAIR_ID:-}"
if [ -z "${REPAIR_ROW_INDEX}" ] && [ -z "${REPAIR_ID}" ]; then
  echo "set REPAIR_ROW_INDEX, REPAIR_ID, or submit as a SLURM array"
  exit 2
fi

ENV_FILE="${RUN_DIR}/segment_repair_${SLURM_JOB_ID:-manual}_${REPAIR_ROW_INDEX:-id${REPAIR_ID}}.env"
python - "${REPAIR_JOBS_CSV}" "${ENV_FILE}" "${REPAIR_ROW_INDEX}" "${REPAIR_ID}" <<'PY'
import csv
import shlex
import sys

jobs_csv, env_file, row_index_text, repair_id_text = sys.argv[1:5]
selected = None
with open(jobs_csv, newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
if repair_id_text:
    for row in rows:
        if row["repair_id"] == repair_id_text:
            selected = row
            break
elif row_index_text:
    index = int(row_index_text)
    if index < 0 or index >= len(rows):
        raise SystemExit(f"repair row index out of range: {index} of {len(rows)}")
    selected = rows[index]
if selected is None:
    raise SystemExit("repair job row not found")

keys = [
    "repair_id",
    "puzzle_id",
    "solution_index",
    "start_step",
    "target_step",
    "old_segment_len",
    "original_length",
    "search_depth",
    "k1_radius",
    "prefix_path",
    "old_segment_path",
    "suffix_path",
    "target_state",
]
with open(env_file, "w", encoding="utf-8") as out:
    for key in keys:
        out.write(f"REPAIR_{key.upper()}={shlex.quote(selected[key])}\n")
PY
source "${ENV_FILE}"

PUZZLE_ID="${REPAIR_REPAIR_ID}"
DEPTH_LIMIT="${DEPTH_LIMIT:-${REPAIR_SEARCH_DEPTH}}"
BEAM_WIDTH="${BEAM_WIDTH:-900000000}"
BEAM_SOLVED_NEIGHBORHOOD_RADIUS="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS:-${REPAIR_K1_RADIUS}}"

SHARD_COUNT="${SHARD_COUNT:-32}"
STREAM4_BATCH_ALIGNMENT="${STREAM4_BATCH_ALIGNMENT:-1024}"
SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1000000}"
STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES:-262144}"
STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES:-1048576}"
BEAM_B_MICRO="${BEAM_B_MICRO:-8192}"
BEAM_STREAM1_CONCURRENCY="${BEAM_STREAM1_CONCURRENCY:-8}"
BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-8}"
BEAM_STREAM4_ACTIVE_SORT_SLOTS="${BEAM_STREAM4_ACTIVE_SORT_SLOTS:-4}"
BEAM_SHARD_BUFFER_COUNT="${BEAM_SHARD_BUFFER_COUNT:-2}"
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-88064}"
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-2000000}"
BEAM_GPU_HEADROOM_BYTES="${BEAM_GPU_HEADROOM_BYTES:-268435456}"
BEAM_HISTORY_RAM_BYTES="${BEAM_HISTORY_RAM_BYTES:-68719476736}"
BEAM_HISTORY_DISK_BYTES="${BEAM_HISTORY_DISK_BYTES:-274877906944}"

export BEAM_PUZZLE_INFO_JSON="${DATA_DIR}/puzzle_info.json"
export BEAM_GENERATOR_PATH="${DATA_DIR}/puzzle_info.json"
export BEAM_TEST_CSV="${REPAIR_TEST_CSV}"
export BEAM_WEIGHT_DIR="${WEIGHT_DIR}"
export BEAM_TARGET_STATE_TEXT="${REPAIR_TARGET_STATE}"

beam_preflight
beam_configure_build production_runner
beam_derive_shard_capacity
beam_validate_manual_config

echo "repair_id=${REPAIR_REPAIR_ID}"
echo "source_puzzle_id=${REPAIR_PUZZLE_ID}"
echo "solution_index=${REPAIR_SOLUTION_INDEX}"
echo "start_step=${REPAIR_START_STEP}"
echo "target_step=${REPAIR_TARGET_STEP}"
echo "old_segment_len=${REPAIR_OLD_SEGMENT_LEN}"
echo "original_length=${REPAIR_ORIGINAL_LENGTH}"
echo "depth_limit=${DEPTH_LIMIT}"
echo "beam_width=${BEAM_WIDTH}"
echo "k1_radius=${BEAM_SOLVED_NEIGHBORHOOD_RADIUS}"
echo "target_state_override=1"

beam_export_common_runtime
export BEAM_PREDICT_STATS_VERBOSE="${BEAM_PREDICT_STATS_VERBOSE:-0}"
export BEAM_PREDICT_STATS_PATH="${BEAM_PREDICT_STATS_PATH:-${RUN_DIR}/predict_stats_segment_repair_${REPAIR_REPAIR_ID}_${SLURM_JOB_ID:-manual}.jsonl}"
export BEAM_B_MICRO
export BEAM_STREAM1_CONCURRENCY
export BEAM_STREAM3_RING_SLOTS
export BEAM_SHARD_BUFFER_COUNT
beam_export_manual_config
beam_prepare_nccl_file "ihes_segment_repair_${REPAIR_REPAIR_ID}"

RUN_LOG="${LOG_DIR}/production_runner_ihes_segment_repair_${REPAIR_REPAIR_ID}_${SLURM_JOB_ID:-manual}.log"
beam_torchrun_production "segment_repair_${REPAIR_REPAIR_ID}" "${RUN_LOG}"

python - "${RUN_LOG}" "${REPAIR_PREFIX_PATH}" "${REPAIR_SUFFIX_PATH}" "${REPAIR_OLD_SEGMENT_LEN}" <<'PY'
import re
import sys

run_log, prefix, suffix, old_len_text = sys.argv[1:5]
old_len = int(old_len_text)
text = open(run_log, encoding="utf-8", errors="ignore").read()
matches = re.findall(r"puzzle_solved=1\b[^\n]*\bsolution_length=(\d+)\s+solution=([^\s]+)", text)
if not matches:
    print("segment_repair_solved=0")
    raise SystemExit(0)
segment_len, segment = matches[-1]
segment_len = int(segment_len)
parts = [item for item in (prefix, segment, suffix) if item]
candidate = ".".join(parts)
candidate_len = 0 if not candidate else len([item for item in candidate.split(".") if item])
print("segment_repair_solved=1")
print(f"segment_solution_length={segment_len}")
print(f"old_segment_len={old_len}")
print(f"segment_delta={segment_len - old_len}")
print(f"candidate_solution_length={candidate_len}")
print(f"candidate_solution={candidate}")
PY
echo "finished_at=$(date -Is)"
