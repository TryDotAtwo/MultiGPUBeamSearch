#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=ihes_smoke
#SBATCH --output=/mnt/pool/6/vokirova/beam8a100/ihes_cube_smoke/logs/ihes-smoke-%j.out

set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/pool/6/vokirova/beam8a100}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/repo}"
RUN_DIR="${RUN_DIR:-${BASE_DIR}/ihes_cube_smoke}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/data}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
BUILD_DIR="${BUILD_DIR:-${RUN_DIR}/build-${SLURM_JOB_ID:-manual}}"
WEIGHT_DIR="${WEIGHT_DIR:-${RUN_DIR}/dummy_weights}"
NINJA_VENV_DIR="${NINJA_VENV_DIR:-${VIRTUAL_ENV:-/mnt/pool/3/vokirova/ninja-venv}}"
CUTLASS_DIR="${CUTLASS_DIR:-${REPO_DIR}/external/cutlass}"
NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-}"
NCCL_LIBRARY="${NCCL_LIBRARY:-}"

mkdir -p "${RUN_DIR}" "${DATA_DIR}" "${LOG_DIR}" "${BUILD_DIR}"

if [ -f "${NINJA_VENV_DIR}/bin/activate" ]; then
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
if [ ! -d "${REPO_DIR}/.git" ]; then
  echo "missing_repo=${REPO_DIR}"
  exit 2
fi
if [ ! -d "${CUTLASS_DIR}/include" ]; then
  echo "missing_cutlass=${CUTLASS_DIR}"
  exit 2
fi

if [ -z "${NCCL_INCLUDE_DIR}" ] || [ -z "${NCCL_LIBRARY}" ]; then
  nccl_base="$("${NINJA_VENV_DIR}/bin/python" - <<'PY'
from pathlib import Path
import nvidia.nccl
print(Path(nvidia.nccl.__file__).resolve().parent)
PY
)"
  NCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR:-${nccl_base}/include}"
  NCCL_LIBRARY="${NCCL_LIBRARY:-${nccl_base}/lib/libnccl.so.2}"
fi
if [ ! -f "${NCCL_INCLUDE_DIR}/nccl.h" ]; then
  echo "missing_nccl_header=${NCCL_INCLUDE_DIR}/nccl.h"
  exit 2
fi
if [ ! -f "${NCCL_LIBRARY}" ]; then
  echo "missing_nccl_library=${NCCL_LIBRARY}"
  exit 2
fi
export LD_LIBRARY_PATH="$(dirname "${NCCL_LIBRARY}"):${LD_LIBRARY_PATH:-}"

echo "repo_dir=${REPO_DIR}"
echo "run_dir=${RUN_DIR}"
echo "build_dir=${BUILD_DIR}"
echo "git_head=$(git -C "${REPO_DIR}" rev-parse HEAD)"
echo "started_at=$(date -Is)"
nvidia-smi

if [ ! -f "${DATA_DIR}/puzzle_info.json" ] || [ ! -f "${DATA_DIR}/test.csv" ]; then
  echo "download_ihes_cube_data=1"
  "${NINJA_VENV_DIR}/bin/python" -m pip install --user kaggle --quiet || true
  kaggle competitions download -c cayleypy-ihes-cube -p "${DATA_DIR}" --force
  unzip -o "${DATA_DIR}/cayleypy-ihes-cube.zip" -d "${DATA_DIR}"
else
  echo "download_ihes_cube_data=0"
fi

"${NINJA_VENV_DIR}/bin/python" \
  "${REPO_DIR}/hpc/ihes_cube_smoke/make_dummy_weights.py" \
  "${WEIGHT_DIR}"

cmake -S "${REPO_DIR}" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUTLASS_DIR="${CUTLASS_DIR}" \
  -DNCCL_INCLUDE_DIR="${NCCL_INCLUDE_DIR}" \
  -DNCCL_LIBRARY="${NCCL_LIBRARY}" \
  -DBEAM_CUDA_ARCHITECTURES=80 \
  -DBEAM_PUZZLE_INFO_JSON="${DATA_DIR}/puzzle_info.json" \
  -DBEAM_ENABLE_DEBUG=ON \
  -DBEAM_ENABLE_DEPTH_LOGS=ON

cmake --build "${BUILD_DIR}" --target production_runner -j "${SLURM_CPUS_PER_TASK:-8}"

cd "${RUN_DIR}"
mkdir -p test_results

BEAM_GENERATOR_PATH=data/puzzle_info.json \
BEAM_WEIGHT_DIR="${WEIGHT_DIR}" \
BEAM_RUNTIME_CONFIG_MODE=manual \
BEAM_B_MICRO=64 \
BEAM_STREAM1_CONCURRENCY=1 \
BEAM_STREAM3_RING_SLOTS=1 \
BEAM_SHARD_COUNT=1 \
BEAM_SHARD_BUFFER_COUNT=2 \
BEAM_STREAM4_BATCH_CANDIDATES=2048 \
BEAM_STREAM4_TRIGGER_CANDIDATES=2048 \
BEAM_SHARD_CAPACITY_CANDIDATES=2048 \
BEAM_GLOBAL_SPILL_CAPACITY=0 \
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=1024 \
BEAM_GPU_HEADROOM_BYTES=268435456 \
"${BUILD_DIR}/production_runner" 0 2 512

echo "finished_at=$(date -Is)"
