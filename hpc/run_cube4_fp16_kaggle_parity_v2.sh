#!/bin/bash
#SBATCH --partition=kaf12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/pool/6/vokirova/beam444-fp16-parity/parity-%j.out
set -euo pipefail
ROOT="${ROOT:-/mnt/pool/6/vokirova/beam444-fp16-parity}"
SOURCE="${ROOT}/source-a1db0e6"
MODEL_ARCHIVE="${ROOT}/cube4-full-transformer-inference.zip"
MODEL_ROOT="${ROOT}/model"
DATA_ROOT="${DATA_ROOT:-/mnt/pool/6/vokirova/beam444a100/cayley-py-444-cube}"
PIN=a1db0e6d9bb5458c8a842b37dfa99572d3025667
SCRIPT_SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
if [ "${1:-submit}" = submit ]; then
  mkdir -p "${ROOT}" "${MODEL_ROOT}"
  if [ ! -f "${MODEL_ARCHIVE}" ]; then
    curl -fL https://www.kaggle.com/api/v1/datasets/download/trydotatwo/cube4-full-transformer-inference -o "${MODEL_ARCHIVE}"
  fi
  if [ ! -f "${MODEL_ROOT}/.ready" ]; then
    unzip -q -o "${MODEL_ARCHIVE}" -d "${MODEL_ROOT}"
    touch "${MODEL_ROOT}/.ready"
  fi
  if [ ! -d "${SOURCE}/.git" ]; then
    git clone https://github.com/TryDotAtwo/MultiGPUBeamSearch.git "${SOURCE}"
  fi
  git -C "${SOURCE}" fetch origin "${PIN}"
  git -C "${SOURCE}" checkout --detach FETCH_HEAD
  test "$(git -C "${SOURCE}" rev-parse HEAD)" = "${PIN}"
  test -f "${DATA_ROOT}/puzzle_info.json"
  test -f "${DATA_ROOT}/test.csv"
  jid="$(sbatch --parsable "${SCRIPT_SOURCE}" run)"
  echo "JOB_ID=${jid}"
  echo "OUTPUT=${ROOT}/parity-${jid}.out"
  echo "LIVE_LOG=tail -n 100 -f ${ROOT}/parity-${jid}.out"
  exit 0
fi
test "${1:-}" = run
test -n "${SLURM_JOB_ID:-}"
test "$(git -C "${SOURCE}" rev-parse HEAD)" = "${PIN}"
PY=/mnt/pool/3/vokirova/ninja-venv/bin/python
CHECKPOINT="$(find "${MODEL_ROOT}" -type f -name model.pth -print -quit)"
METADATA="$(find "${MODEL_ROOT}" -type f -name model.json -print -quit)"
GENERATORS="$(find "${MODEL_ROOT}" -type f -name p002.json -print -quit)"
test -n "${CHECKPOINT}" && test -n "${METADATA}" && test -n "${GENERATORS}"
WEIGHTS="${ROOT}/weights-fp16"
if [ ! -f "${WEIGHTS}/manifest.json" ]; then
  "${PY}" "${SOURCE}/tools/export_stream1_transformer.py" \
    --weights "${CHECKPOINT}" --out "${WEIGHTS}" --dtype fp16 \
    --num-classes 6 --metadata "${METADATA}" --generators "${GENERATORS}" \
    --source-root "${MODEL_ROOT}"
fi
LAUNCHER_REPO="${LAUNCHER_REPO:-$(cd "$(dirname "${SCRIPT_SOURCE}")/.." && pwd)}"
source "${LAUNCHER_REPO}/hpc/mephi_8xa100_common.sh"
export JOB_DIR="${ROOT}" REPO_DIR="${SOURCE}" BUILD_DIR="${ROOT}/build-a100-fp16"
export HISTORY_DIR="${ROOT}/history" LOG_DIR="${ROOT}/logs"
export CUTLASS_DIR=/mnt/pool/3/vokirova/cutlass NINJA_VENV_DIR=/mnt/pool/3/vokirova/ninja-venv
export TORCHRUN_NNODES=1 TORCHRUN_NPROC_PER_NODE=2 TORCHRUN_NODE_RANK=0
export TORCHRUN_RDZV_ENDPOINT=127.0.0.1:29610
beam_setup_paths
export PUZZLE_ID=1000 DEPTH_LIMIT=50 BEAM_WIDTH=60000000
export SHARD_COUNT=4 STREAM4_BATCH_ALIGNMENT=1024 SHARD_CAPACITY_SCALE_PPM=1250000
export STREAM4_BATCH_CANDIDATES=196608 STREAM4_TRIGGER_CANDIDATES=196608
export BEAM_B_MICRO=8192 BEAM_STREAM1_CONCURRENCY=4 BEAM_STREAM3_RING_SLOTS=4
export BEAM_STREAM4_ACTIVE_SORT_SLOTS=4 BEAM_WEIGHT_DIR="${WEIGHTS}"
export BEAM_GENERATOR_PATH="${DATA_ROOT}/puzzle_info.json"
export BEAM_PUZZLE_INFO_JSON="${DATA_ROOT}/puzzle_info.json" BEAM_TEST_CSV="${DATA_ROOT}/test.csv"
export BEAM_SOLVED_NEIGHBORHOOD_RADIUS=4 BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES=3000000
export BEAM_HISTORY_MODE=static_hybrid
export BEAM_HISTORY_RAM_BYTES=$((64 * 1024 * 1024 * 1024))
export BEAM_HISTORY_DISK_BYTES=$((1024 * 1024 * 1024 * 1024))
export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=2000000
export BEAM_GPU_HEADROOM_BYTES=$((3 * 1024 * 1024 * 1024))
export BEAM_ENABLE_DEBUG=ON BEAM_ENABLE_DEPTH_LOGS=ON
export BEAM_DEBUG_INFERENCE_TRACE=ON BEAM_DEBUG_PATH_TRACE=ON
export BEAM_TRACK_SOLUTION_PATH='-f3.-f0.d1.r3.f1.-f3.-r2.-d0.-r0.d2.f1.-d2.f3.d2.-d0.r0.-f3.-r0.-f3.f0.f0.-r0.-r3.-d3.-d2.r2.-d1.f2.f1.f2.r1.f3.-r0.f0.-d1.-f1.r2.d1.-f2.d1.f2.-r2.f1.d3'
export BEAM_STOP_AFTER_TRACKED_PATH=1
export BEAM_STOP_AFTER_TRACKED_MISSING_EXTRA_DEPTHS=0
beam_preflight
TORCH_CMAKE_PREFIX="$("${PY}" -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake -S "${SOURCE}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_DIR="${CUTLASS_DIR}" \
  -DNCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR}" \
  -DNCCL_LIBRARY="${NCCL_LIBRARY}" \
  -DBEAM_CUDA_ARCHITECTURES=80 \
  -DBEAM_ENABLE_LIBTORCH_STREAM1=ON \
  -DCMAKE_PREFIX_PATH="${TORCH_CMAKE_PREFIX}"
beam_configure_build production_runner_libtorch_stream1
RUN_BUILD_DIR="${ROOT}/run-bin-a100-fp16"
mkdir -p "${RUN_BUILD_DIR}"
ln -sfn "${BUILD_DIR}/production_runner_libtorch_stream1" "${RUN_BUILD_DIR}/production_runner"
export BUILD_DIR="${RUN_BUILD_DIR}"
export BEAM_STREAM1_EXECUTOR=libtorch_eager
beam_derive_shard_capacity
beam_validate_manual_config
beam_export_common_runtime
beam_export_manual_config
beam_prepare_nccl_file parity
RUN_LOG="${LOG_DIR}/cube4_fp16_parity_p1000_b60000000_${SLURM_JOB_ID}.log"
beam_native_production cube4_fp16_parity "${RUN_LOG}"
echo PARITY_RUN_OK=1
