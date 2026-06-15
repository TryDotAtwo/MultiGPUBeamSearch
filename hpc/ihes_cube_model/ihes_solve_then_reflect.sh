#!/bin/bash
#SBATCH --job-name=ihes-reflect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/pool/6/vokirova/beam8a100/ihes_cube_model/logs/ihes-reflect-%j.out

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

ORIGINAL_PUZZLE_ID="${PUZZLE_ID:-0}"
DEPTH_LIMIT="${DEPTH_LIMIT:-30}"
BEAM_WIDTH="${BEAM_WIDTH:-900000000}"
SYNTHETIC_PUZZLE_ID="${SYNTHETIC_PUZZLE_ID:-$((9000000 + ORIGINAL_PUZZLE_ID))}"

SHARD_COUNT="${SHARD_COUNT:-32}"
STREAM4_BATCH_ALIGNMENT="${STREAM4_BATCH_ALIGNMENT:-1024}"
SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1000000}"
STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES:-262144}"
STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES:-1048576}"
BEAM_B_MICRO="${BEAM_B_MICRO:-8192}"
BEAM_STREAM1_CONCURRENCY="${BEAM_STREAM1_CONCURRENCY:-8}"
BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-8}"
if [ "${BEAM_STREAM3_RING_SLOTS}" -lt "${BEAM_STREAM1_CONCURRENCY}" ]; then
  BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM1_CONCURRENCY}"
fi
BEAM_STREAM4_ACTIVE_SORT_SLOTS="${BEAM_STREAM4_ACTIVE_SORT_SLOTS:-4}"
BEAM_SHARD_BUFFER_COUNT="${BEAM_SHARD_BUFFER_COUNT:-2}"
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-98304}"
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-8000000}"
BEAM_SOLVED_NEIGHBORHOOD_RADIUS="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS:-4}"
BEAM_GPU_HEADROOM_BYTES="${BEAM_GPU_HEADROOM_BYTES:-134217728}"
BEAM_HISTORY_RAM_BYTES="${BEAM_HISTORY_RAM_BYTES:-68719476736}"
BEAM_HISTORY_DISK_BYTES="${BEAM_HISTORY_DISK_BYTES:-4398046511104}"

TEST_CSV_BACKUP=""

restore_test_csv() {
  if [ -n "${TEST_CSV_BACKUP}" ] && [ -f "${TEST_CSV_BACKUP}" ]; then
    cp "${TEST_CSV_BACKUP}" "${DATA_DIR}/test.csv"
    echo "restored_test_csv=${DATA_DIR}/test.csv"
  fi
}

cleanup() {
  local rc=$?
  echo "cleanup_start rc=${rc} at $(date -Is)"
  restore_test_csv
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

extract_solution_line() {
  local puzzle_id="$1"
  local run_log="$2"
  local rank_log_dir="$3"
  grep -hE "puzzle_solved=1 puzzle_id=${puzzle_id} .*solution_length=.* solution=" \
    "${run_log}" "${rank_log_dir}"/rank*.log 2>/dev/null | tail -1
}

solution_field() {
  sed -E 's/^.* solution=//'
}

solution_length_field() {
  sed -E 's/^.* solution_length=([0-9]+) solution=.*$/\1/'
}

run_solve() {
  local puzzle_id="$1"
  local tag="$2"
  LAST_SOLUTION_LINE=""
  PUZZLE_ID="${puzzle_id}"
  export PUZZLE_ID
  beam_safe_clear_history_contents
  beam_prepare_nccl_file "${tag}"
  local run_log="${LOG_DIR}/production_runner_ihes_p${PUZZLE_ID}_d${DEPTH_LIMIT}_b${BEAM_WIDTH}_${SLURM_JOB_ID:-manual}_${tag}.log"
  beam_torchrun_production "${tag}" "${run_log}"
  local rank_log_dir="${LOG_DIR}/ranks_${SLURM_JOB_ID:-manual}_${tag}"
  local line
  line="$(extract_solution_line "${puzzle_id}" "${run_log}" "${rank_log_dir}")"
  if [ -z "${line}" ]; then
    echo "missing_solution_line puzzle_id=${puzzle_id} run_log=${run_log} rank_log_dir=${rank_log_dir}"
    return 3
  fi
  LAST_SOLUTION_LINE="${line}"
  echo "solution_line_${tag}=${LAST_SOLUTION_LINE}"
}

write_reflected_puzzle() {
  local solution="$1"
  TEST_CSV_BACKUP="${LOG_DIR}/test_before_ihes_reflect_${SLURM_JOB_ID:-manual}.csv"
  cp "${DATA_DIR}/test.csv" "${TEST_CSV_BACKUP}"
  SOLUTION_TEXT="${solution}" \
  SYNTHETIC_PUZZLE_ID="${SYNTHETIC_PUZZLE_ID}" \
  DATA_DIR="${DATA_DIR}" \
  "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
solution = os.environ["SOLUTION_TEXT"]
synthetic_id = int(os.environ["SYNTHETIC_PUZZLE_ID"])
info = json.loads((data_dir / "puzzle_info.json").read_text())
central = list(info["central_state"])
generators = info["generators"]

def apply_path(state, path):
    out = list(state)
    for token in path.split("."):
        if token not in generators:
            raise RuntimeError(f"unknown move token: {token}")
        out = [out[i] for i in generators[token]]
    return out

def invert_token(token):
    return token[1:] if token.startswith("-") else "-" + token

def invert_path(path):
    return ".".join(invert_token(token) for token in reversed(path.split(".")))

reflected = apply_path(central, solution)
roundtrip = apply_path(reflected, invert_path(solution))
if roundtrip != central:
    raise RuntimeError("reflected-state roundtrip failed")

row = f'{synthetic_id},"{",".join(str(x) for x in reflected)}"\n'
test_csv = data_dir / "test.csv"
lines = test_csv.read_text().splitlines()
prefix = f"{synthetic_id},"
lines = [line for line in lines if not line.startswith(prefix)]
lines.append(row.rstrip("\n"))
test_csv.write_text("\n".join(lines) + "\n")
print(f"reflected_puzzle_id={synthetic_id}")
print(f"reflected_source_solution_length={len(solution.split('.'))}")
print("reflected_roundtrip_ok=1")
PY
}

verify_and_print_candidate() {
  local original_solution="$1"
  local reflected_solution="$2"
  ORIGINAL_PUZZLE_ID="${ORIGINAL_PUZZLE_ID}" \
  ORIGINAL_SOLUTION="${original_solution}" \
  REFLECTED_SOLUTION="${reflected_solution}" \
  DATA_DIR="${DATA_DIR}" \
  "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import csv
import json
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
original_id = int(os.environ["ORIGINAL_PUZZLE_ID"])
original_solution = os.environ["ORIGINAL_SOLUTION"]
reflected_solution = os.environ["REFLECTED_SOLUTION"]
info = json.loads((data_dir / "puzzle_info.json").read_text())
central = list(info["central_state"])
generators = info["generators"]

def apply_path(state, path):
    out = list(state)
    for token in path.split("."):
        if token not in generators:
            raise RuntimeError(f"unknown move token: {token}")
        out = [out[i] for i in generators[token]]
    return out

def invert_token(token):
    return token[1:] if token.startswith("-") else "-" + token

def invert_path(path):
    return ".".join(invert_token(token) for token in reversed(path.split(".")))

def load_state(puzzle_id):
    with (data_dir / "test.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["initial_state_id"]) == puzzle_id:
                return [int(x) for x in row["initial_state"].split(",")]
    raise RuntimeError(f"puzzle id not found: {puzzle_id}")

original_state = load_state(original_id)
candidate = invert_path(reflected_solution)
print(f"original_solution_length={len(original_solution.split('.'))}")
print(f"reflected_solution_length={len(reflected_solution.split('.'))}")
print(f"candidate_solution_length={len(candidate.split('.'))}")
print(f"original_solution_solves_original={int(apply_path(original_state, original_solution) == central)}")
print(f"candidate_solution_solves_original={int(apply_path(original_state, candidate) == central)}")
print(f"original_solution={original_solution}")
print(f"reflected_solution={reflected_solution}")
print(f"candidate_solution_for_original={candidate}")
PY
}

beam_setup_paths
beam_preflight

if [ ! -f "${MODEL_PATH}" ]; then
  echo "missing_model=${MODEL_PATH}"
  exit 2
fi
if [ ! -f "${DATA_DIR}/puzzle_info.json" ] || [ ! -f "${DATA_DIR}/test.csv" ]; then
  echo "missing_ihes_data=${DATA_DIR}"
  exit 2
fi
if [ ! -f "${WEIGHT_DIR}/manifest.json" ]; then
  echo "missing_stream1_weights=${WEIGHT_DIR}/manifest.json"
  exit 2
fi

export BEAM_PUZZLE_INFO_JSON="${DATA_DIR}/puzzle_info.json"
export BEAM_GENERATOR_PATH="${DATA_DIR}/puzzle_info.json"
export BEAM_TEST_CSV="${DATA_DIR}/test.csv"
export BEAM_WEIGHT_DIR="${WEIGHT_DIR}"

beam_configure_build production_runner
beam_derive_shard_capacity
beam_validate_manual_config

echo "original_puzzle_id=${ORIGINAL_PUZZLE_ID}"
echo "synthetic_puzzle_id=${SYNTHETIC_PUZZLE_ID}"
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
export BEAM_PREDICT_STATS_PATH="${BEAM_PREDICT_STATS_PATH:-${RUN_DIR}/predict_stats_reflect_p${ORIGINAL_PUZZLE_ID}_b${BEAM_WIDTH}_d${DEPTH_LIMIT}_${SLURM_JOB_ID:-manual}.jsonl}"
export BEAM_B_MICRO
export BEAM_STREAM1_CONCURRENCY
export BEAM_STREAM3_RING_SLOTS
export BEAM_SHARD_BUFFER_COUNT
beam_export_manual_config

run_solve "${ORIGINAL_PUZZLE_ID}" "original"
first_line="${LAST_SOLUTION_LINE}"
first_solution="$(printf '%s\n' "${first_line}" | solution_field)"
first_length="$(printf '%s\n' "${first_line}" | solution_length_field)"
echo "original_solution_line=${first_line}"
echo "original_solution_length_parsed=${first_length}"

write_reflected_puzzle "${first_solution}"

run_solve "${SYNTHETIC_PUZZLE_ID}" "reflected"
second_line="${LAST_SOLUTION_LINE}"
second_solution="$(printf '%s\n' "${second_line}" | solution_field)"
second_length="$(printf '%s\n' "${second_line}" | solution_length_field)"
echo "reflected_solution_line=${second_line}"
echo "reflected_solution_length_parsed=${second_length}"

verify_and_print_candidate "${first_solution}" "${second_solution}"
echo "finished_at=$(date -Is)"
