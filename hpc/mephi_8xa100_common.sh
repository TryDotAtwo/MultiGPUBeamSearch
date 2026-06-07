#!/bin/bash

beam_setup_paths() {
  JOB_DIR="${SLURM_SUBMIT_DIR:-/mnt/pool/6/vokirova/beam8a100}"
  REPO_DIR="${JOB_DIR}/repo"
  BUILD_DIR="${JOB_DIR}/build-a100"
  CUTLASS_DIR="${CUTLASS_DIR:-/mnt/pool/3/vokirova/cutlass}"
  NINJA_VENV_DIR="${NINJA_VENV_DIR:-/mnt/pool/3/vokirova/ninja-venv}"
  NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-${NINJA_VENV_DIR}/lib/python3.13/site-packages/nvidia/nccl/include}"
  NCCL_LIBRARY="${NCCL_LIBRARY:-${NINJA_VENV_DIR}/lib/python3.13/site-packages/nvidia/nccl/lib/libnccl.so.2}"
  HISTORY_DIR="${JOB_DIR}/history"
  LOG_DIR="${JOB_DIR}/logs"
  TUNING_DIR="${LOG_DIR}/tuning_${SLURM_JOB_ID:-manual}"
  TORCHRUN_NNODES="${TORCHRUN_NNODES:-1}"
  TORCHRUN_NPROC_PER_NODE="${TORCHRUN_NPROC_PER_NODE:-8}"
  TORCHRUN_NODE_RANK="${TORCHRUN_NODE_RANK:-${SLURM_NODEID:-0}}"
  TORCHRUN_RDZV_ENDPOINT="${TORCHRUN_RDZV_ENDPOINT:-127.0.0.1:29500}"
  WORLD_SIZE_EFFECTIVE=$((TORCHRUN_NNODES * TORCHRUN_NPROC_PER_NODE))
}

beam_preflight() {
  mkdir -p "${JOB_DIR}" "${BUILD_DIR}" "${HISTORY_DIR}" "${LOG_DIR}" "${TUNING_DIR}"
  if [ -f "${NINJA_VENV_DIR}/bin/activate" ]; then
    # MEPhI jobs need the prepared venv active before CMake/torchrun checks.
    source "${NINJA_VENV_DIR}/bin/activate"
  fi
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
  echo "job_dir=${JOB_DIR}"
  echo "repo_dir=${REPO_DIR}"
  echo "git_head=$(git rev-parse HEAD)"
  echo "started_at=$(date -Is)"
  echo "world_size_effective=${WORLD_SIZE_EFFECTIVE}"
  ninja --version
  "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import torch
print("torch", torch.__version__)
PY
  nvidia-smi
  export LD_LIBRARY_PATH="$(dirname "${NCCL_LIBRARY}"):${LD_LIBRARY_PATH:-}"
}

beam_configure_build() {
  local target="${1:-production_runner}"
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
  cmake --build "${BUILD_DIR}" --target "${target}" -j "${SLURM_CPUS_PER_TASK:-8}"
}

beam_round_up() {
  local value="$1"
  local alignment="$2"
  echo $(( ((value + alignment - 1) / alignment) * alignment ))
}

beam_stream1_output_dim() {
  local weight_dir="${BEAM_WEIGHT_DIR:-${REPO_DIR}/stream1_weights}"
  "${NINJA_VENV_DIR}/bin/python" - "${weight_dir}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
print(int(manifest["output_dim"]))
PY
}

beam_stream1_parent_batch_from_row_budget() {
  local row_budget="$1"
  local output_dim="$2"
  if [ "${output_dim}" -eq 1 ]; then
    if [ "${row_budget}" -lt 24 ]; then
      echo "BEAM_B_MICRO row budget is smaller than one 24-move expansion: ${row_budget}" >&2
      return 2
    fi
    echo $((row_budget / 24))
  else
    echo "${row_budget}"
  fi
}

beam_derive_shard_capacity() {
  STREAM1_OUTPUT_DIM="$(beam_stream1_output_dim)"
  BEAM_PARENT_BATCH_EFFECTIVE="$(beam_stream1_parent_batch_from_row_budget "${BEAM_B_MICRO}" "${STREAM1_OUTPUT_DIM}")"
  STREAM1_ROWS_PER_JOB_EFFECTIVE=$((BEAM_PARENT_BATCH_EFFECTIVE * (STREAM1_OUTPUT_DIM == 1 ? 24 : 1)))
  BEAM_ALIGNMENT=$((WORLD_SIZE_EFFECTIVE * SHARD_COUNT * STREAM4_BATCH_ALIGNMENT))
  GLOBAL_BEAM_WIDTH_EFFECTIVE="$(beam_round_up "${BEAM_WIDTH}" "${BEAM_ALIGNMENT}")"
  LOCAL_BEAM_WIDTH=$((GLOBAL_BEAM_WIDTH_EFFECTIVE / WORLD_SIZE_EFFECTIVE))
  LOGICAL_SHARD_SIZE=$(( (LOCAL_BEAM_WIDTH + SHARD_COUNT - 1) / SHARD_COUNT ))
  SHARD_CAPACITY_RAW=$(( (LOGICAL_SHARD_SIZE * SHARD_CAPACITY_SCALE_PPM + 999999) / 1000000 ))
  SHARD_CAPACITY_CANDIDATES="$(beam_round_up "${SHARD_CAPACITY_RAW}" "${STREAM4_BATCH_ALIGNMENT}")"
  STREAM3_BATCH_CANDIDATES=$((BEAM_STREAM3_RING_SLOTS * BEAM_PARENT_BATCH_EFFECTIVE * 24))
}

beam_validate_manual_config() {
  if [ "${STREAM4_BATCH_CANDIDATES}" -gt "${SHARD_CAPACITY_CANDIDATES}" ]; then
    echo "invalid_stream4_batch=${STREAM4_BATCH_CANDIDATES} shard_capacity=${SHARD_CAPACITY_CANDIDATES}"
    return 2
  fi
  if [ "${STREAM4_TRIGGER_CANDIDATES}" -lt "${STREAM4_BATCH_CANDIDATES}" ]; then
    echo "invalid_stream4_trigger=${STREAM4_TRIGGER_CANDIDATES} stream4_batch=${STREAM4_BATCH_CANDIDATES}"
    return 2
  fi
  if [ "${STREAM4_TRIGGER_CANDIDATES}" -gt "${SHARD_CAPACITY_CANDIDATES}" ]; then
    echo "invalid_stream4_trigger=${STREAM4_TRIGGER_CANDIDATES} shard_capacity=${SHARD_CAPACITY_CANDIDATES}"
    return 2
  fi
  if [ "${STREAM3_BATCH_CANDIDATES}" -gt "${SHARD_CAPACITY_CANDIDATES}" ]; then
    echo "invalid_stream3_batch=${STREAM3_BATCH_CANDIDATES} shard_capacity=${SHARD_CAPACITY_CANDIDATES}"
    return 2
  fi
  if [ "${BEAM_STREAM1_CONCURRENCY}" -gt "${BEAM_STREAM3_RING_SLOTS}" ]; then
    echo "invalid_stream1_concurrency=${BEAM_STREAM1_CONCURRENCY} stream3_ring_slots=${BEAM_STREAM3_RING_SLOTS}"
    return 2
  fi
}

beam_export_common_runtime() {
  export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export NCCL_ASYNC_ERROR_HANDLING=1
  export CUDA_DEVICE_MAX_CONNECTIONS=32
  export BEAM_WEIGHT_DIR="${BEAM_WEIGHT_DIR:-${REPO_DIR}/stream1_weights}"
  export BEAM_DEPTH_LOG_EVERY=1
  export BEAM_HISTORY_MODE="${BEAM_HISTORY_MODE:-static_hybrid}"
  export BEAM_HISTORY_SLOT_COUNT="${BEAM_HISTORY_SLOT_COUNT:-2}"
  export BEAM_HISTORY_WORKERS="${BEAM_HISTORY_WORKERS:-4}"
  export BEAM_HISTORY_RAM_BYTES="${BEAM_HISTORY_RAM_BYTES:-$((160 * 1024 * 1024 * 1024))}"
  export BEAM_HISTORY_DISK_BYTES="${BEAM_HISTORY_DISK_BYTES:-$((384 * 1024 * 1024 * 1024))}"
  export BEAM_HISTORY_DISK_PATH="${HISTORY_DIR}"
  export BEAM_SOLVED_NEIGHBORHOOD_RADIUS="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS:-5}"
  export BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES="${BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES:-3000000}"
  export BEAM_STREAM2_SUFFIX_RADIUS="${BEAM_STREAM2_SUFFIX_RADIUS:-0}"
  export BEAM_STREAM2_SUFFIX_BACKEND="${BEAM_STREAM2_SUFFIX_BACKEND:-composed_permutations}"
  export BEAM_STREAM2_SUFFIX_MAX_COUNT="${BEAM_STREAM2_SUFFIX_MAX_COUNT:-1}"
  export BEAM_PREDICT_STATS_VERBOSE="${BEAM_PREDICT_STATS_VERBOSE:-0}"
}

beam_export_manual_config() {
  export BEAM_RUNTIME_CONFIG_MODE=manual
  export BEAM_SHARD_COUNT="${SHARD_COUNT}"
  export BEAM_STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES}"
  export BEAM_STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES}"
  export BEAM_SHARD_CAPACITY_CANDIDATES="${SHARD_CAPACITY_CANDIDATES}"
  export BEAM_SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM}"
  export BEAM_STREAM4_ACTIVE_SORT_SLOTS="${BEAM_STREAM4_ACTIVE_SORT_SLOTS}"
  export BEAM_GLOBAL_SPILL_CAPACITY="${BEAM_GLOBAL_SPILL_CAPACITY:-0}"
  export BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM="${BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM:-1200000}"
  export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-0}"
  export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-1200000}"
  export BEAM_GPU_HEADROOM_BYTES="${BEAM_GPU_HEADROOM_BYTES:-$((3 * 1024 * 1024 * 1024))}"
}

beam_prepare_nccl_file() {
  NCCL_ID_FILE="${JOB_DIR}/beam_solver_nccl_${SLURM_JOB_ID:-manual}_${1:-run}.bin"
  rm -f "${NCCL_ID_FILE}"
  export BEAM_NCCL_ID_FILE="${NCCL_ID_FILE}"
}

beam_torchrun_production() {
  local run_tag="$1"
  local run_log="$2"
  local rank_log_dir="${LOG_DIR}/ranks_${SLURM_JOB_ID:-manual}_${run_tag}"
  local gpu_log="${TUNING_DIR}/nvidia_smi_${run_tag}.log"
  local gpu_monitor_pid=""
  mkdir -p "${rank_log_dir}"
  export BEAM_RANK_LOG_DIR="${rank_log_dir}"
  echo "run_tag=${run_tag}"
  echo "run_log=${run_log}"
  echo "rank_log_dir=${BEAM_RANK_LOG_DIR}"
  echo "beam_nccl_id_file=${BEAM_NCCL_ID_FILE}"
  echo "gpu_monitor_log=${gpu_log}"
  (
    while true; do
      date -Is
      nvidia-smi --query-gpu=index,name,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits
      sleep 5
    done
  ) > "${gpu_log}" 2>&1 &
  gpu_monitor_pid=$!
  set +e
  "${NINJA_VENV_DIR}/bin/python" -m torch.distributed.run \
    --nnodes="${TORCHRUN_NNODES}" \
    --nproc-per-node="${TORCHRUN_NPROC_PER_NODE}" \
    --node-rank="${TORCHRUN_NODE_RANK}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${TORCHRUN_RDZV_ENDPOINT}" \
    --rdzv-id="beam8a100_${SLURM_JOB_ID:-manual}_${run_tag}" \
    --no-python \
    /bin/bash -lc 'if [ "${RANK:-0}" = "0" ]; then exec "$@"; else exec "$@" > "${BEAM_RANK_LOG_DIR}/rank${RANK}.log" 2>&1; fi' \
    bash "${BUILD_DIR}/production_runner" "${PUZZLE_ID}" "${DEPTH_LIMIT}" "${BEAM_WIDTH}" 2>&1 | tee "${run_log}"
  local torchrun_rc=${PIPESTATUS[0]}
  set -e
  if [ -n "${gpu_monitor_pid}" ]; then
    kill "${gpu_monitor_pid}" >/dev/null 2>&1 || true
    wait "${gpu_monitor_pid}" >/dev/null 2>&1 || true
  fi
  return "${torchrun_rc}"
}

beam_safe_clean_child() {
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

beam_safe_clear_history_contents() {
  local resolved_job
  local resolved_history
  resolved_job="$(realpath -m "${JOB_DIR}")"
  resolved_history="$(realpath -m "${HISTORY_DIR}")"
  if [ "${resolved_history}" != "${resolved_job}/history" ]; then
    echo "history_clear_skip=unsafe_path:${resolved_history}"
    return 2
  fi
  mkdir -p "${resolved_history}"
  find "${resolved_history}" -mindepth 1 -maxdepth 1 -xdev -exec rm -rf -- {} +
}
