#!/bin/bash
#SBATCH --job-name=beam8a100-reflect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
BEAM_COMMON_SH="${BEAM_COMMON_SH:-${SCRIPT_DIR}/mephi_8xa100_common.sh}"
if [ ! -f "${BEAM_COMMON_SH}" ]; then
  echo "missing_common_script=${BEAM_COMMON_SH}"
  exit 2
fi
source "${BEAM_COMMON_SH}"

beam_setup_paths

ORIGINAL_PUZZLE_ID="${PUZZLE_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
DEPTH_LIMIT="${DEPTH_LIMIT:-80}"
BEAM_WIDTH="${BEAM_WIDTH:-900000000}"
SYNTHETIC_PUZZLE_ID="${SYNTHETIC_PUZZLE_ID:-$((9000000 + ORIGINAL_PUZZLE_ID))}"
MEGAMINX_STREAM1_BACKEND="${MEGAMINX_STREAM1_BACKEND:-${BEAM_STREAM1_EXECUTOR:-native_cuda_graph}}"

MODEL_DIR="${MODEL_DIR:-${JOB_DIR}/models/megaminx_vlad_transformer}"
REPO_WEIGHT_DIR="${REPO_DIR}/weights/megaminx_vlad_transformer_fp16"
if [ -z "${BEAM_WEIGHT_DIR:-}" ]; then
  if [ -f "${REPO_WEIGHT_DIR}/manifest.json" ]; then
    BEAM_WEIGHT_DIR="${REPO_WEIGHT_DIR}"
  else
    BEAM_WEIGHT_DIR="${JOB_DIR}/stream1_transformer_weights_fp16"
  fi
fi

BEST_CONFIG_ENV="${BEST_CONFIG_ENV:-/dev/null}"
if [ -f "${BEST_CONFIG_ENV}" ]; then
  source "${BEST_CONFIG_ENV}"
fi

SHARD_COUNT="${SHARD_COUNT:-32}"
STREAM4_BATCH_ALIGNMENT="${STREAM4_BATCH_ALIGNMENT:-1024}"
SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1250000}"
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
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-98304}"
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-8000000}"
BEAM_SOLVED_NEIGHBORHOOD_RADIUS="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS:-4}"
BEAM_GPU_HEADROOM_BYTES="${BEAM_GPU_HEADROOM_BYTES:-268435456}"
BEAM_HISTORY_RAM_BYTES="${BEAM_HISTORY_RAM_BYTES:-257698037760}"
BEAM_HISTORY_DISK_BYTES="${BEAM_HISTORY_DISK_BYTES:-1099511627776}"

TEST_CSV_BACKUP=""

restore_test_csv() {
  if [ -n "${TEST_CSV_BACKUP}" ] && [ -f "${TEST_CSV_BACKUP}" ]; then
    cp "${TEST_CSV_BACKUP}" "${REPO_DIR}/data/test.csv"
    echo "restored_test_csv=${REPO_DIR}/data/test.csv"
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

find_transformer_checkpoint() {
  if [ -n "${BEAM_TRANSFORMER_PTH:-}" ]; then
    printf '%s\n' "${BEAM_TRANSFORMER_PTH}"
    return 0
  fi
  if [ -d "${MODEL_DIR}" ]; then
    find "${MODEL_DIR}" -maxdepth 5 -type f \( -name '*.pth' -o -name '*.pt' \) | sort | head -n 1
  fi
}

prepare_stream1_weights() {
  if [ -f "${BEAM_WEIGHT_DIR}/manifest.json" ]; then
    echo "using_existing_weight_dir=${BEAM_WEIGHT_DIR}"
    return 0
  fi
  local checkpoint
  checkpoint="$(find_transformer_checkpoint || true)"
  if [ -z "${checkpoint}" ] || [ ! -f "${checkpoint}" ]; then
    echo "missing_stream1_weights=${BEAM_WEIGHT_DIR}/manifest.json"
    echo "missing_transformer_checkpoint=${MODEL_DIR}/*.pth"
    return 2
  fi
  echo "export_transformer_checkpoint=${checkpoint}"
  rm -rf -- "${BEAM_WEIGHT_DIR}.tmp"
  mkdir -p "${BEAM_WEIGHT_DIR}.tmp"
  "${NINJA_VENV_DIR}/bin/python" "${REPO_DIR}/tools/export_stream1.py" \
    --weights "${checkpoint}" \
    --out "${BEAM_WEIGHT_DIR}.tmp" \
    --dtype fp16 \
    --format piece-transformer
  rm -rf -- "${BEAM_WEIGHT_DIR}"
  mv "${BEAM_WEIGHT_DIR}.tmp" "${BEAM_WEIGHT_DIR}"
  echo "exported_weight_dir=${BEAM_WEIGHT_DIR}"
}

configure_stream1_runner() {
  case "${MEGAMINX_STREAM1_BACKEND}" in
    libtorch_eager)
      export BEAM_ENABLE_LIBTORCH_STREAM1=ON
      if [ -z "${CMAKE_PREFIX_PATH:-}" ]; then
        CMAKE_PREFIX_PATH="$("${NINJA_VENV_DIR}/bin/python" - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"
        export CMAKE_PREFIX_PATH
      fi
      TORCH_LIB_DIR="$("${NINJA_VENV_DIR}/bin/python" - <<'PY'
from pathlib import Path
import torch
print(Path(torch.__file__).resolve().parent / "lib")
PY
)"
      export LD_LIBRARY_PATH="${TORCH_LIB_DIR}:$(dirname "${NCCL_LIBRARY}"):${LD_LIBRARY_PATH:-}"
      beam_configure_build production_runner_libtorch_stream1
      export BEAM_PRODUCTION_RUNNER_PATH="${BUILD_DIR}/production_runner_libtorch_stream1"
      export BEAM_STREAM1_EXECUTOR=libtorch_eager
      ;;
    native_cuda_graph|cuda_graph)
      export BEAM_ENABLE_LIBTORCH_STREAM1=OFF
      export LD_LIBRARY_PATH="$(dirname "${NCCL_LIBRARY}"):${LD_LIBRARY_PATH:-}"
      beam_configure_build production_runner
      export BEAM_PRODUCTION_RUNNER_PATH="${BUILD_DIR}/production_runner"
      export BEAM_STREAM1_EXECUTOR=native_cuda_graph
      ;;
    *)
      echo "invalid_megaminx_stream1_backend=${MEGAMINX_STREAM1_BACKEND}"
      echo "allowed_megaminx_stream1_backend=libtorch_eager,native_cuda_graph"
      exit 2
      ;;
  esac
}
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
  local run_log="${LOG_DIR}/production_runner_p${PUZZLE_ID}_d${DEPTH_LIMIT}_b${BEAM_WIDTH}_${SLURM_JOB_ID:-manual}_${tag}.log"
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
  TEST_CSV_BACKUP="${LOG_DIR}/test_before_reflect_${SLURM_JOB_ID:-manual}.csv"
  cp "${REPO_DIR}/data/test.csv" "${TEST_CSV_BACKUP}"
  SOLUTION_TEXT="${solution}" \
  SYNTHETIC_PUZZLE_ID="${SYNTHETIC_PUZZLE_ID}" \
  "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

repo = Path(os.environ["REPO_DIR"])
solution = os.environ["SOLUTION_TEXT"]
synthetic_id = int(os.environ["SYNTHETIC_PUZZLE_ID"])
info = json.loads((repo / "data" / "puzzle_info.json").read_text())
central = list(info["central_state"])
generators = info["generators"]

def apply_path(state, path):
    out = list(state)
    for token in path.split("."):
        if token not in generators:
            raise RuntimeError(f"unknown move token: {token}")
        perm = generators[token]
        out = [out[i] for i in perm]
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
test_csv = repo / "data" / "test.csv"
lines = test_csv.read_text().splitlines()
prefix = f"{synthetic_id},"
lines = [line for line in lines if not line.startswith(prefix)]
lines.append(row.rstrip("\n"))
test_csv.write_text("\n".join(lines) + "\n")
print(f"reflected_puzzle_id={synthetic_id}")
print(f"reflected_solution_length={len(solution.split('.'))}")
print("reflected_roundtrip_ok=1")
print(f"reflected_state={','.join(str(x) for x in reflected)}")
PY
}

verify_and_print_candidate() {
  local original_solution="$1"
  local reflected_solution="$2"
  ORIGINAL_PUZZLE_ID="${ORIGINAL_PUZZLE_ID}" \
  ORIGINAL_SOLUTION="${original_solution}" \
  REFLECTED_SOLUTION="${reflected_solution}" \
  "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import csv
import json
import os
from pathlib import Path

repo = Path(os.environ["REPO_DIR"])
original_id = int(os.environ["ORIGINAL_PUZZLE_ID"])
original_solution = os.environ["ORIGINAL_SOLUTION"]
reflected_solution = os.environ["REFLECTED_SOLUTION"]
info = json.loads((repo / "data" / "puzzle_info.json").read_text())
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
    with (repo / "data" / "test.csv").open(newline="") as handle:
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

beam_preflight
prepare_stream1_weights
configure_stream1_runner
export BEAM_WEIGHT_DIR
beam_derive_shard_capacity
beam_validate_manual_config

if [ "${BEAM_STREAM1_EXECUTOR}" = "libtorch_eager" ] && [ "${STREAM1_OUTPUT_DIM}" -ne "${BEAM_MOVE_COUNT_EFFECTIVE}" ]; then
  echo "invalid_libtorch_output_dim=${STREAM1_OUTPUT_DIM} move_count=${BEAM_MOVE_COUNT_EFFECTIVE}"
  exit 2
fi

echo "original_puzzle_id=${ORIGINAL_PUZZLE_ID}"
echo "synthetic_puzzle_id=${SYNTHETIC_PUZZLE_ID}"
echo "beam_width=${BEAM_WIDTH}"
echo "global_beam_width_effective=${GLOBAL_BEAM_WIDTH_EFFECTIVE}"
echo "local_beam_width=${LOCAL_BEAM_WIDTH}"
echo "shard_count=${SHARD_COUNT}"
echo "megaminx_stream1_backend=${MEGAMINX_STREAM1_BACKEND}"
echo "stream1_executor=${BEAM_STREAM1_EXECUTOR}"
echo "stream1_weight_dir=${BEAM_WEIGHT_DIR}"
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
export BEAM_B_MICRO
export BEAM_STREAM1_CONCURRENCY
export BEAM_STREAM3_RING_SLOTS
export BEAM_SHARD_BUFFER_COUNT
beam_export_manual_config
export REPO_DIR

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
