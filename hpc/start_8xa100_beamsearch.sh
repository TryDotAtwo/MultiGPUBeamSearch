#!/bin/bash
#SBATCH --job-name=beam8a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

set -euo pipefail

JOB_DIR="${SLURM_SUBMIT_DIR:-/mnt/pool/6/vokirova/beam8a100}"
REPO_DIR="${JOB_DIR}/repo"
BUILD_DIR="${JOB_DIR}/build-a100"
CUTLASS_DIR="${CUTLASS_DIR:-/mnt/pool/3/vokirova/cutlass}"
HISTORY_DIR="${JOB_DIR}/history"
LOG_DIR="${JOB_DIR}/logs"
PREDICT_STATS_PATH="${JOB_DIR}/predict_stats_p992_b260m_d12.jsonl"

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

mkdir -p "${JOB_DIR}" "${BUILD_DIR}" "${HISTORY_DIR}" "${LOG_DIR}"
if [ ! -d "${REPO_DIR}/.git" ]; then
  echo "missing_repo=${REPO_DIR}"
  exit 2
fi
if [ ! -d "${CUTLASS_DIR}/include" ]; then
  echo "missing_cutlass=${CUTLASS_DIR}"
  exit 2
fi
cd "${REPO_DIR}"

echo "job_dir=${JOB_DIR}"
echo "repo_dir=${REPO_DIR}"
git rev-parse HEAD
echo "started_at=$(date -Is)"
nvidia-smi

cmake -S "${REPO_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_DIR="${CUTLASS_DIR}" \
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
export BEAM_B_MICRO=4096
export BEAM_STREAM1_CONCURRENCY=2
export BEAM_STREAM3_RING_SLOTS=4
export BEAM_SHARD_BUFFER_COUNT=2
export BEAM_SHARD_COUNT=64
export BEAM_STREAM4_BATCH_CANDIDATES=524288
export BEAM_STREAM4_TRIGGER_CANDIDATES=1048576
export BEAM_SHARD_CAPACITY_CANDIDATES=1052672
export BEAM_STREAM4_ACTIVE_SORT_SLOTS=4
export BEAM_GLOBAL_SPILL_CAPACITY=0
export BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM=1200000
export BEAM_GPU_HEADROOM_BYTES=$((3 * 1024 * 1024 * 1024))

RUN_LOG="${LOG_DIR}/production_runner_p992_d12_b260m_${SLURM_JOB_ID:-manual}.log"
echo "run_log=${RUN_LOG}"

python3 -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=127.0.0.1:29500 \
  --rdzv-id="beam8a100_${SLURM_JOB_ID:-manual}" \
  --no-python \
  "${BUILD_DIR}/production_runner" 992 12 260000000 2>&1 | tee "${RUN_LOG}"

echo "finished_at=$(date -Is)"
