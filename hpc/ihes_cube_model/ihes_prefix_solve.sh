#!/bin/bash
#SBATCH --job-name=ihes-prefix-solve
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/pool/6/vokirova/beam8a100/ihes_cube_model/logs/ihes-prefix-%A_%a.out

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/pool/6/vokirova/beam8a100}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/repo}"
JOB_DIR="${JOB_DIR:-${BASE_DIR}/ihes_cube_model}"
RUN_DIR="${RUN_DIR:-${JOB_DIR}}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-${RUN_DIR}/data}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/data-prefix-${SLURM_JOB_ID:-manual}}"
WEIGHT_DIR="${WEIGHT_DIR:-${RUN_DIR}/stream1_weights_ihes_bf16}"
BUILD_DIR="${BUILD_DIR:-${RUN_DIR}/build-a100-${SLURM_JOB_ID:-manual}}"
HISTORY_DIR="${HISTORY_DIR:-${RUN_DIR}/history-${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
SOLUTIONS_CSV="${SOLUTIONS_CSV:-${RUN_DIR}/solv_uniq.csv}"
PREFIX_PLAN_TSV="${PREFIX_PLAN_TSV:-${RUN_DIR}/prefix_solve_len24_plan.tsv}"
BEAM_COMMON_SH="${BEAM_COMMON_SH:-${REPO_DIR}/hpc/mephi_8xa100_common.sh}"

mkdir -p "${RUN_DIR}" "${DATA_DIR}" "${BUILD_DIR}" "${HISTORY_DIR}" "${LOG_DIR}"

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
  case "$(realpath -m "${DATA_DIR}")" in
    "$(realpath -m "${JOB_DIR}")"/data-prefix-*)
      rm -rf --one-file-system "${DATA_DIR}"
      echo "cleanup_remove_data=${DATA_DIR}"
      ;;
    *) echo "cleanup_skip_data=${DATA_DIR}" ;;
  esac
  echo "cleanup_done rc=${rc} at $(date -Is)"
  exit "${rc}"
}
trap cleanup EXIT

beam_setup_paths

DEPTH_LIMIT="${DEPTH_LIMIT:-30}"
BEAM_WIDTH="${BEAM_WIDTH:-900000000}"
PREFIX_PLAN_INDEX="${PREFIX_PLAN_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

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
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-98304}"
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-$(beam_default_final_exchange_scale_ppm)}"
BEAM_SOLVED_NEIGHBORHOOD_RADIUS="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS:-4}"
BEAM_GPU_HEADROOM_BYTES="${BEAM_GPU_HEADROOM_BYTES:-134217728}"
BEAM_HISTORY_RAM_BYTES="${BEAM_HISTORY_RAM_BYTES:-68719476736}"
BEAM_HISTORY_DISK_BYTES="${BEAM_HISTORY_DISK_BYTES:-4398046511104}"

beam_preflight

if [ ! -f "${SOURCE_DATA_DIR}/puzzle_info.json" ] || [ ! -f "${SOURCE_DATA_DIR}/test.csv" ]; then
  echo "missing_source_ihes_data=${SOURCE_DATA_DIR}"
  exit 2
fi
if [ ! -f "${WEIGHT_DIR}/manifest.json" ]; then
  echo "missing_stream1_weights=${WEIGHT_DIR}/manifest.json"
  exit 2
fi
if [ ! -f "${SOLUTIONS_CSV}" ]; then
  echo "missing_solutions_csv=${SOLUTIONS_CSV}"
  exit 2
fi

cp "${SOURCE_DATA_DIR}/puzzle_info.json" "${DATA_DIR}/puzzle_info.json"
cp "${SOURCE_DATA_DIR}/test.csv" "${DATA_DIR}/test.csv"

if [ ! -f "${PREFIX_PLAN_TSV}" ]; then
  "${NINJA_VENV_DIR}/bin/python" "${REPO_DIR}/tools/prepare_ihes_prefix_solve_plan.py" \
    --solutions "${SOLUTIONS_CSV}" \
    --test-csv "${SOURCE_DATA_DIR}/test.csv" \
    --puzzle-info "${SOURCE_DATA_DIR}/puzzle_info.json" \
    --out "${PREFIX_PLAN_TSV}" \
    --length 24
fi

PLAN_ROW="$("${NINJA_VENV_DIR}/bin/python" - "${PREFIX_PLAN_TSV}" "${PREFIX_PLAN_INDEX}" <<'PY'
import csv
import sys
path, wanted = sys.argv[1], int(sys.argv[2])
with open(path, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        if int(row["plan_index"]) == wanted:
            print("\t".join(row[k] for k in [
                "puzzle_id", "synthetic_id", "prefix_move", "prefix_count",
                "length24_solution_count", "synthetic_state"
            ]))
            raise SystemExit(0)
raise SystemExit(f"prefix plan index not found: {wanted}")
PY
)"
IFS=$'\t' read -r ORIGINAL_PUZZLE_ID PUZZLE_ID PREFIX_MOVE PREFIX_COUNT LENGTH24_COUNT SYNTHETIC_STATE <<< "${PLAN_ROW}"

printf '%s,"%s"\n' "${PUZZLE_ID}" "${SYNTHETIC_STATE}" >> "${DATA_DIR}/test.csv"

export BEAM_PUZZLE_INFO_JSON="${DATA_DIR}/puzzle_info.json"
export BEAM_GENERATOR_PATH="${DATA_DIR}/puzzle_info.json"
export BEAM_TEST_CSV="${DATA_DIR}/test.csv"
export BEAM_WEIGHT_DIR="${WEIGHT_DIR}"

beam_configure_build production_runner
beam_derive_shard_capacity
beam_validate_manual_config
beam_export_common_runtime
export BEAM_PREDICT_STATS_VERBOSE="${BEAM_PREDICT_STATS_VERBOSE:-1}"
export BEAM_PREDICT_STATS_PATH="${RUN_DIR}/predict_stats_prefix_${PUZZLE_ID}_${SLURM_JOB_ID:-manual}.jsonl"
export BEAM_B_MICRO
export BEAM_STREAM1_CONCURRENCY
export BEAM_STREAM3_RING_SLOTS
export BEAM_SHARD_BUFFER_COUNT
beam_export_manual_config

echo "prefix_plan_index=${PREFIX_PLAN_INDEX}"
echo "original_puzzle_id=${ORIGINAL_PUZZLE_ID}"
echo "synthetic_puzzle_id=${PUZZLE_ID}"
echo "prefix_move=${PREFIX_MOVE}"
echo "prefix_count=${PREFIX_COUNT}"
echo "length24_solution_count=${LENGTH24_COUNT}"
echo "depth_limit=${DEPTH_LIMIT}"
echo "beam_width=${BEAM_WIDTH}"

beam_prepare_nccl_file "ihes_prefix_${PUZZLE_ID}"
RUN_LOG="${LOG_DIR}/production_runner_ihes_prefix_${ORIGINAL_PUZZLE_ID}_${PUZZLE_ID}_${SLURM_JOB_ID:-manual}.log"
beam_native_production "ihes_prefix_${PUZZLE_ID}" "${RUN_LOG}"

"${NINJA_VENV_DIR}/bin/python" - "${RUN_LOG}" "${PREFIX_MOVE}" "${ORIGINAL_PUZZLE_ID}" "${PUZZLE_ID}" <<'PY'
import re
import sys
log_path, prefix, original_id, synthetic_id = sys.argv[1:5]
text = open(log_path, encoding="utf-8", errors="replace").read()
matches = re.findall(r"puzzle_solved=1\b[^\n]*\bsolution_length=(\d+)\s+solution=([^\s]+)", text)
if not matches:
    print("prefix_solve_solved=0")
    raise SystemExit(0)
suffix_len, suffix = matches[-1]
candidate = prefix + ("." + suffix if suffix else "")
print("prefix_solve_solved=1")
print(f"original_puzzle_id={original_id}")
print(f"synthetic_puzzle_id={synthetic_id}")
print(f"suffix_length={suffix_len}")
print(f"candidate_solution_length={len(candidate.split('.'))}")
print(f"candidate_solution={candidate}")
PY

echo "finished_at=$(date -Is)"
