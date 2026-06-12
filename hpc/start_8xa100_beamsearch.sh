#!/bin/bash
#SBATCH --job-name=beam8a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail

JOB_DIR="${SLURM_SUBMIT_DIR:-/mnt/pool/6/vokirova/beam8a100}"
REPO_DIR="${JOB_DIR}/repo"
BUILD_DIR="${JOB_DIR}/build-a100"
CUTLASS_DIR="${CUTLASS_DIR:-/mnt/pool/3/vokirova/cutlass}"
NINJA_VENV_DIR="${NINJA_VENV_DIR:-/mnt/pool/3/vokirova/ninja-venv}"
NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-${NINJA_VENV_DIR}/lib/python3.13/site-packages/nvidia/nccl/include}"
NCCL_LIBRARY="${NCCL_LIBRARY:-${NINJA_VENV_DIR}/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2}"
HISTORY_DIR="${JOB_DIR}/history"
LOG_DIR="${JOB_DIR}/logs"
PREDICT_STATS_PATH="${JOB_DIR}/predict_stats_p992_b260m_d12.jsonl"
PUZZLE_ID="${PUZZLE_ID:-992}"
DEPTH_LIMIT="${DEPTH_LIMIT:-12}"
BEAM_WIDTH="${BEAM_WIDTH:-260000000}"
TORCHRUN_NNODES="${TORCHRUN_NNODES:-1}"
TORCHRUN_NPROC_PER_NODE="${TORCHRUN_NPROC_PER_NODE:-8}"
TORCHRUN_NODE_RANK="${TORCHRUN_NODE_RANK:-${SLURM_NODEID:-0}}"
TORCHRUN_RDZV_ENDPOINT="${TORCHRUN_RDZV_ENDPOINT:-127.0.0.1:29500}"
BEAM_B_MICRO="${BEAM_B_MICRO:-8192}"
BEAM_STREAM1_CONCURRENCY="${BEAM_STREAM1_CONCURRENCY:-8}"
BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-8}"
if [ "${BEAM_STREAM3_RING_SLOTS}" -lt "${BEAM_STREAM1_CONCURRENCY}" ]; then
  BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM1_CONCURRENCY}"
fi
SHARD_COUNT="${SHARD_COUNT:-16}"
STREAM4_BATCH_ALIGNMENT="${STREAM4_BATCH_ALIGNMENT:-1024}"
SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1250000}"
STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES:-524288}"
STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES:-524288}"

safe_remove_job_child_dir() {
  local target="$1"
  local label="$2"
  local resolved_job
  local resolved_target

  resolved_job="$(realpath -m "${JOB_DIR}")"
  resolved_target="$(realpath -m "${target}")"
  case "${resolved_target}" in
    "${resolved_job}/build-a100"|"${resolved_job}/history")
      if [ -d "${resolved_target}" ]; then
        echo "cleanup_remove_${label}=${resolved_target}"
        rm -rf --one-file-system "${resolved_target}"
      fi
      ;;
    *)
      echo "cleanup_skip_${label}=unsafe_path:${resolved_target}"
      ;;
  esac
}

cleanup() {
  local rc=$?
  echo "cleanup_start rc=${rc} at $(date -Is)"
  safe_remove_job_child_dir "${BUILD_DIR}" "build"
  if [ "${rc}" -eq 0 ]; then
    safe_remove_job_child_dir "${HISTORY_DIR}" "history"
  else
    echo "cleanup_keep_history=${HISTORY_DIR}"
  fi
  echo "cleanup_done rc=${rc} at $(date -Is)"
  exit "${rc}"
}
trap cleanup EXIT

round_up() {
  local value="$1"
  local alignment="$2"
  echo $(( ((value + alignment - 1) / alignment) * alignment ))
}

mkdir -p "${JOB_DIR}" "${BUILD_DIR}" "${HISTORY_DIR}" "${LOG_DIR}"
if [ -x "${NINJA_VENV_DIR}/bin/ninja" ]; then
  export PATH="${NINJA_VENV_DIR}/bin:${PATH}"
else
  echo "missing_ninja=${NINJA_VENV_DIR}/bin/ninja"
  exit 2
fi
if [ ! -x "${NINJA_VENV_DIR}/bin/python" ]; then
  echo "missing_python=${NINJA_VENV_DIR}/bin/python"
  exit 2
fi
if [ ! -f "${NCCL_INCLUDE_DIR}/nccl.h" ]; then
  echo "missing_nccl_header=${NCCL_INCLUDE_DIR}/nccl.h"
  exit 2
fi
if [ ! -f "${NCCL_LIBRARY}" ]; then
  echo "missing_nccl_library=${NCCL_LIBRARY}"
  exit 2
fi
if [ ! -d "${REPO_DIR}/.git" ]; then
  echo "missing_repo=${REPO_DIR}"
  exit 2
fi
if [ ! -d "${CUTLASS_DIR}/include" ]; then
  echo "missing_cutlass=${CUTLASS_DIR}"
  exit 2
fi
cd "${REPO_DIR}"

WORLD_SIZE_EFFECTIVE=$((TORCHRUN_NNODES * TORCHRUN_NPROC_PER_NODE))
BEAM_ALIGNMENT=$((WORLD_SIZE_EFFECTIVE * SHARD_COUNT * STREAM4_BATCH_ALIGNMENT))
GLOBAL_BEAM_WIDTH_EFFECTIVE="$(round_up "${BEAM_WIDTH}" "${BEAM_ALIGNMENT}")"
LOCAL_BEAM_WIDTH=$((GLOBAL_BEAM_WIDTH_EFFECTIVE / WORLD_SIZE_EFFECTIVE))
LOGICAL_SHARD_SIZE=$(( (LOCAL_BEAM_WIDTH + SHARD_COUNT - 1) / SHARD_COUNT ))
SHARD_CAPACITY_RAW=$(( (LOGICAL_SHARD_SIZE * SHARD_CAPACITY_SCALE_PPM + 999999) / 1000000 ))
SHARD_CAPACITY_CANDIDATES="$(round_up "${SHARD_CAPACITY_RAW}" "${STREAM4_BATCH_ALIGNMENT}")"
STREAM3_BATCH_CANDIDATES=$((BEAM_STREAM3_RING_SLOTS * BEAM_B_MICRO * 24))
if [ "${STREAM4_BATCH_CANDIDATES}" -gt "${SHARD_CAPACITY_CANDIDATES}" ]; then
  echo "invalid_stream4_batch=${STREAM4_BATCH_CANDIDATES} shard_capacity=${SHARD_CAPACITY_CANDIDATES}"
  exit 2
fi
if [ "${STREAM4_TRIGGER_CANDIDATES}" -lt "${STREAM4_BATCH_CANDIDATES}" ]; then
  echo "invalid_stream4_trigger=${STREAM4_TRIGGER_CANDIDATES} stream4_batch=${STREAM4_BATCH_CANDIDATES}"
  exit 2
fi
if [ "${STREAM4_TRIGGER_CANDIDATES}" -gt "${SHARD_CAPACITY_CANDIDATES}" ]; then
  echo "invalid_stream4_trigger=${STREAM4_TRIGGER_CANDIDATES} shard_capacity=${SHARD_CAPACITY_CANDIDATES}"
  exit 2
fi
if [ "${STREAM3_BATCH_CANDIDATES}" -gt "${SHARD_CAPACITY_CANDIDATES}" ]; then
  echo "invalid_stream3_batch=${STREAM3_BATCH_CANDIDATES} shard_capacity=${SHARD_CAPACITY_CANDIDATES}"
  exit 2
fi
if [ "${BEAM_STREAM1_CONCURRENCY}" -gt "${BEAM_STREAM3_RING_SLOTS}" ]; then
  echo "invalid_stream1_concurrency=${BEAM_STREAM1_CONCURRENCY} stream3_ring_slots=${BEAM_STREAM3_RING_SLOTS}"
  exit 2
fi

echo "job_dir=${JOB_DIR}"
echo "repo_dir=${REPO_DIR}"
git rev-parse HEAD
echo "started_at=$(date -Is)"
echo "beam_width=${BEAM_WIDTH}"
echo "torchrun_nnodes=${TORCHRUN_NNODES}"
echo "torchrun_nproc_per_node=${TORCHRUN_NPROC_PER_NODE}"
echo "torchrun_node_rank=${TORCHRUN_NODE_RANK}"
echo "torchrun_rdzv_endpoint=${TORCHRUN_RDZV_ENDPOINT}"
echo "world_size_effective=${WORLD_SIZE_EFFECTIVE}"
echo "beam_alignment=${BEAM_ALIGNMENT}"
echo "global_beam_width_effective=${GLOBAL_BEAM_WIDTH_EFFECTIVE}"
echo "local_beam_width=${LOCAL_BEAM_WIDTH}"
echo "logical_shard_size=${LOGICAL_SHARD_SIZE}"
echo "shard_capacity_candidates=${SHARD_CAPACITY_CANDIDATES}"
echo "stream3_batch_candidates=${STREAM3_BATCH_CANDIDATES}"
echo "stream4_batch_candidates=${STREAM4_BATCH_CANDIDATES}"
echo "stream4_trigger_candidates=${STREAM4_TRIGGER_CANDIDATES}"
ninja --version
"${NINJA_VENV_DIR}/bin/python" - <<'PY'
import torch
print("torch", torch.__version__)
PY
nvidia-smi

export LD_LIBRARY_PATH="$(dirname "${NCCL_LIBRARY}"):${LD_LIBRARY_PATH:-}"

cmake -S "${REPO_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_DIR="${CUTLASS_DIR}" \
  -DNCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR}" \
  -DNCCL_LIBRARY="${NCCL_LIBRARY}" \
  -DBEAM_CUDA_ARCHITECTURES=80 \
  -DBEAM_ENABLE_DEBUG=ON \
  -DBEAM_ENABLE_DEPTH_LOGS=ON \
  -DBEAM_ENABLE_DEBUG_LOGS=OFF \
  -DBEAM_DEBUG_STREAM_TIMING=OFF \
  -DBEAM_DEBUG_INFERENCE_TRACE=OFF \
  -DBEAM_DEBUG_PATH_TRACE=OFF \
  -DBEAM_DEBUG_FINAL_VALIDATE=OFF \
  -DBEAM_DEBUG_FINAL_EXCHANGE_TRACE=OFF \
  -DBEAM_DEBUG_FINAL_HISTOGRAM_TRACE=OFF \
  -DBEAM_DEBUG_STREAM4_HISTOGRAM_TRACE=OFF \
  -DBEAM_DEBUG_DEPTH_FLOW_TRACE=OFF

cmake --build "${BUILD_DIR}" --target production_runner -j 8

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=32

export BEAM_WEIGHT_DIR="${REPO_DIR}/stream1_weights"
export BEAM_DEPTH_LOG_EVERY=1
export BEAM_HISTORY_MODE=static_hybrid
export BEAM_HISTORY_SLOT_COUNT=2
export BEAM_HISTORY_WORKERS=4
export BEAM_HISTORY_RAM_BYTES=$((160 * 1024 * 1024 * 1024))
export BEAM_HISTORY_DISK_BYTES=$((384 * 1024 * 1024 * 1024))
export BEAM_HISTORY_DISK_PATH="${HISTORY_DIR}"
export BEAM_SOLVED_NEIGHBORHOOD_RADIUS=5
export BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES=3000000
export BEAM_STREAM2_SUFFIX_RADIUS=0
export BEAM_STREAM2_SUFFIX_BACKEND=composed_permutations
export BEAM_STREAM2_SUFFIX_MAX_COUNT=1
export BEAM_PREDICT_STATS_VERBOSE=1
export BEAM_PREDICT_STATS_PATH="${PREDICT_STATS_PATH}"

export BEAM_RUNTIME_CONFIG_MODE=manual
export BEAM_B_MICRO
export BEAM_STREAM1_CONCURRENCY
export BEAM_STREAM3_RING_SLOTS
export BEAM_SHARD_BUFFER_COUNT=2
export BEAM_SHARD_COUNT="${SHARD_COUNT}"
export BEAM_STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES}"
export BEAM_STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES}"
export BEAM_SHARD_CAPACITY_CANDIDATES="${SHARD_CAPACITY_CANDIDATES}"
export BEAM_SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM}"
export BEAM_STREAM4_ACTIVE_SORT_SLOTS=4
export BEAM_GLOBAL_SPILL_CAPACITY=0
export BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM=1200000
export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=1200000
export BEAM_GPU_HEADROOM_BYTES=$((3 * 1024 * 1024 * 1024))

RUN_LOG="${LOG_DIR}/production_runner_p${PUZZLE_ID}_d${DEPTH_LIMIT}_b${BEAM_WIDTH}_${SLURM_JOB_ID:-manual}.log"
RANK_LOG_DIR="${LOG_DIR}/ranks_${SLURM_JOB_ID:-manual}"
mkdir -p "${RANK_LOG_DIR}"
NCCL_ID_FILE="${JOB_DIR}/beam_solver_nccl_${SLURM_JOB_ID:-manual}.bin"
rm -f "${NCCL_ID_FILE}"
export BEAM_NCCL_ID_FILE="${NCCL_ID_FILE}"
export BEAM_RANK_LOG_DIR="${RANK_LOG_DIR}"
echo "run_log=${RUN_LOG}"
echo "rank_log_dir=${BEAM_RANK_LOG_DIR}"
echo "beam_nccl_id_file=${BEAM_NCCL_ID_FILE}"

"${NINJA_VENV_DIR}/bin/python" -m torch.distributed.run \
  --nnodes="${TORCHRUN_NNODES}" \
  --nproc-per-node="${TORCHRUN_NPROC_PER_NODE}" \
  --node-rank="${TORCHRUN_NODE_RANK}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${TORCHRUN_RDZV_ENDPOINT}" \
  --rdzv-id="beam8a100_${SLURM_JOB_ID:-manual}" \
  --no-python \
  /bin/bash -lc 'if [ "${RANK:-0}" = "0" ]; then exec "$@"; else exec "$@" > "${BEAM_RANK_LOG_DIR}/rank${RANK}.log" 2>&1; fi' \
  bash "${BUILD_DIR}/production_runner" "${PUZZLE_ID}" "${DEPTH_LIMIT}" "${BEAM_WIDTH}" 2>&1 | tee "${RUN_LOG}"

echo "finished_at=$(date -Is)"
