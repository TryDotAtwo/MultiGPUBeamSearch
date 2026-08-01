#!/bin/bash
#SBATCH --job-name=megaminx-transformer-900m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${SCRIPT_DIR}/mephi_8xa100_common.sh"

beam_setup_paths

DEFAULT_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="$(beam_default_final_exchange_scale_ppm)"

PUZZLE_ID="${PUZZLE_ID:-991}"
DEPTH_LIMIT="${DEPTH_LIMIT:-120}"
BEAM_WIDTH="${BEAM_WIDTH:-900000000}"
MEGAMINX_STREAM1_BACKEND="${MEGAMINX_STREAM1_BACKEND:-${BEAM_STREAM1_EXECUTOR:-libtorch_eager}}"

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
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-${DEFAULT_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM}}"
BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM="${BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM:-1200000}"

MODEL_DIR="${MODEL_DIR:-${JOB_DIR}/models/megaminx_vlad_transformer}"
REPO_WEIGHT_DIR="${REPO_DIR}/weights/megaminx_vlad_transformer_fp16"
if [ -z "${BEAM_WEIGHT_DIR:-}" ]; then
  if [ -f "${REPO_WEIGHT_DIR}/manifest.json" ]; then
    BEAM_WEIGHT_DIR="${REPO_WEIGHT_DIR}"
  else
    BEAM_WEIGHT_DIR="${JOB_DIR}/stream1_transformer_weights_fp16"
  fi
fi

find_transformer_checkpoint() {
  if [ -n "${BEAM_TRANSFORMER_PTH:-}" ]; then
    printf '%s\n' "${BEAM_TRANSFORMER_PTH}"
    return 0
  fi
  if [ -d "${MODEL_DIR}" ]; then
    find "${MODEL_DIR}" -maxdepth 5 -type f \( -name '*.pth' -o -name '*.pt' \) | sort | head -n 1
  fi
}

prepare_libtorch_weights() {
  if [ -f "${BEAM_WEIGHT_DIR}/manifest.json" ]; then
    echo "using_existing_weight_dir=${BEAM_WEIGHT_DIR}"
    return 0
  fi
  local checkpoint
  checkpoint="$(find_transformer_checkpoint || true)"
  if [ -z "${checkpoint}" ] || [ ! -f "${checkpoint}" ]; then
    echo "missing_stream1_weights=${BEAM_WEIGHT_DIR}/manifest.json"
    echo "missing_transformer_checkpoint=${MODEL_DIR}/*.pth"
    echo "Set BEAM_WEIGHT_DIR to an exported piece_transformer weights directory, or set BEAM_TRANSFORMER_PTH to Vlad's Kaggle .pth with metadata/generators/pilgrim nearby."
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

CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
TORCH_LIB_DIR="${TORCH_LIB_DIR:-}"

prepare_libtorch_weights

case "${MEGAMINX_STREAM1_BACKEND}" in
  libtorch_eager)
    export BEAM_ENABLE_LIBTORCH_STREAM1=ON
    if [ -z "${CMAKE_PREFIX_PATH}" ]; then
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
  native_windowed_graph|native_graph_window)
    export BEAM_ENABLE_LIBTORCH_STREAM1=OFF
    export LD_LIBRARY_PATH="$(dirname "${NCCL_LIBRARY}"):${LD_LIBRARY_PATH:-}"
    export BEAM_RING_GRAPH_EXECS_PER_LANE="${BEAM_RING_GRAPH_EXECS_PER_LANE:-32}"
    beam_configure_build production_runner
    export BEAM_PRODUCTION_RUNNER_PATH="${BUILD_DIR}/production_runner"
    export BEAM_STREAM1_EXECUTOR=native_cuda_graph
    ;;
  native_eager|native_no_graph)
    export BEAM_ENABLE_LIBTORCH_STREAM1=OFF
    export LD_LIBRARY_PATH="$(dirname "${NCCL_LIBRARY}"):${LD_LIBRARY_PATH:-}"
    beam_configure_build production_runner
    export BEAM_PRODUCTION_RUNNER_PATH="${BUILD_DIR}/production_runner"
    export BEAM_STREAM1_EXECUTOR=native_eager
    ;;
  *)
    echo "invalid_megaminx_stream1_backend=${MEGAMINX_STREAM1_BACKEND}"
    echo "allowed_megaminx_stream1_backend=libtorch_eager,native_cuda_graph,native_windowed_graph,native_eager,native_no_graph"
    exit 2
    ;;
esac
export BEAM_WEIGHT_DIR

beam_derive_shard_capacity
beam_validate_manual_config

if [ "${BEAM_STREAM1_EXECUTOR}" = "libtorch_eager" ] && [ "${STREAM1_OUTPUT_DIM}" -ne "${BEAM_MOVE_COUNT_EFFECTIVE}" ]; then
  echo "invalid_libtorch_output_dim=${STREAM1_OUTPUT_DIM} move_count=${BEAM_MOVE_COUNT_EFFECTIVE}"
  exit 2
fi

cat <<EOF
puzzle_id=${PUZZLE_ID}
depth_limit=${DEPTH_LIMIT}
beam_width=${BEAM_WIDTH}
global_beam_width_effective=${GLOBAL_BEAM_WIDTH_EFFECTIVE}
local_beam_width=${LOCAL_BEAM_WIDTH}
shard_count=${SHARD_COUNT}
megaminx_stream1_backend=${MEGAMINX_STREAM1_BACKEND}
stream1_executor=${BEAM_STREAM1_EXECUTOR}
ring_graph_execs_per_lane=${BEAM_RING_GRAPH_EXECS_PER_LANE:-}
stream1_weight_dir=${BEAM_WEIGHT_DIR}
stream1_output_dim=${STREAM1_OUTPUT_DIM}
beam_b_micro_parent_budget=${BEAM_B_MICRO}
beam_parent_batch_effective=${BEAM_PARENT_BATCH_EFFECTIVE}
stream1_rows_per_job_effective=${STREAM1_ROWS_PER_JOB_EFFECTIVE}
logical_shard_size=${LOGICAL_SHARD_SIZE}
shard_capacity_candidates=${SHARD_CAPACITY_CANDIDATES}
stream3_batch_candidates=${STREAM3_BATCH_CANDIDATES}
stream4_batch_candidates=${STREAM4_BATCH_CANDIDATES}
stream4_trigger_candidates=${STREAM4_TRIGGER_CANDIDATES}
final_chunk_candidates=${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES}
final_exchange_scale_ppm=${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM}
torch_cmake_prefix=${CMAKE_PREFIX_PATH}
torch_lib_dir=${TORCH_LIB_DIR}
EOF

beam_export_common_runtime
export BEAM_PREDICT_STATS_VERBOSE="${BEAM_PREDICT_STATS_VERBOSE:-1}"
export BEAM_PREDICT_STATS_PATH="${BEAM_PREDICT_STATS_PATH:-${JOB_DIR}/predict_stats_${BEAM_STREAM1_EXECUTOR}_p${PUZZLE_ID}_b${BEAM_WIDTH}_d${DEPTH_LIMIT}.jsonl}"
export BEAM_B_MICRO
export BEAM_STREAM1_CONCURRENCY
export BEAM_STREAM3_RING_SLOTS
export BEAM_SHARD_BUFFER_COUNT
beam_export_manual_config
RUN_TAG="megaminx_${BEAM_STREAM1_EXECUTOR}"
beam_prepare_nccl_file "${RUN_TAG}"

RUN_LOG="${LOG_DIR}/production_runner_${BEAM_STREAM1_EXECUTOR}_p${PUZZLE_ID}_d${DEPTH_LIMIT}_b${BEAM_WIDTH}_${SLURM_JOB_ID:-manual}.log"
beam_torchrun_production "${RUN_TAG}" "${RUN_LOG}"
echo "finished_at=$(date -Is)"
