#!/bin/bash
#SBATCH --job-name=ihes-prebuild-runner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/pool/6/vokirova/beam8a100}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/repo}"
JOB_DIR="${JOB_DIR:-${BASE_DIR}/ihes_cube_model}"
RUN_DIR="${RUN_DIR:-${JOB_DIR}}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/data}"
WEIGHT_DIR="${WEIGHT_DIR:-${RUN_DIR}/stream1_weights_ihes_bf16}"
BUILD_DIR="${BUILD_DIR:-${RUN_DIR}/prebuilt-a100-ihes}"
HISTORY_DIR="${HISTORY_DIR:-${RUN_DIR}/prebuilt-history-check}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
BEAM_COMMON_SH="${BEAM_COMMON_SH:-${REPO_DIR}/hpc/mephi_8xa100_common.sh}"

mkdir -p "${RUN_DIR}" "${BUILD_DIR}" "${LOG_DIR}"

if [ ! -f "${BEAM_COMMON_SH}" ]; then
  echo "missing_common_script=${BEAM_COMMON_SH}"
  exit 2
fi
source "${BEAM_COMMON_SH}"

beam_setup_paths

for required in "${DATA_DIR}/puzzle_info.json" "${WEIGHT_DIR}/manifest.json"; do
  if [ ! -f "${required}" ]; then
    echo "missing_required_file=${required}"
    exit 2
  fi
done

export BEAM_PUZZLE_INFO_JSON="${DATA_DIR}/puzzle_info.json"
export BEAM_GENERATOR_PATH="${DATA_DIR}/puzzle_info.json"
export BEAM_WEIGHT_DIR="${WEIGHT_DIR}"

beam_preflight
beam_configure_build production_runner

RUNNER="${BUILD_DIR}/production_runner"
if [ ! -x "${RUNNER}" ]; then
  echo "missing_built_runner=${RUNNER}"
  exit 2
fi

echo "prebuilt_runner=${RUNNER}"
echo "prebuilt_runner_ready=1"
