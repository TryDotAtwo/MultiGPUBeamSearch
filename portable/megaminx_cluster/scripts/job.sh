#!/usr/bin/env bash
set -euo pipefail

: "${MEGAMINX_ARCHIVE_ROOT:?missing MEGAMINX_ARCHIVE_ROOT}"
: "${MEGAMINX_RUN_DIR:?missing MEGAMINX_RUN_DIR}"
: "${MEGAMINX_GPU_IDS:?missing MEGAMINX_GPU_IDS}"
: "${MEGAMINX_BEAM:?missing MEGAMINX_BEAM}"
: "${MEGAMINX_PUZZLE:?missing MEGAMINX_PUZZLE}"
: "${MEGAMINX_REFLECT:?missing MEGAMINX_REFLECT}"

if [ -f "${MEGAMINX_ARCHIVE_ROOT}/cluster.env" ]; then
  # User-owned, allowlisted by submit.py before sbatch.
  set -a
  source "${MEGAMINX_ARCHIVE_ROOT}/cluster.env"
  set +a
fi

GPU_CSV="${MEGAMINX_GPU_IDS//:/,}"
IFS=':' read -r -a GPU_ARRAY <<< "${MEGAMINX_GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
: "${CUDA_VISIBLE_DEVICES:?SLURM did not provide GPU visibility}"
export PYTHONPATH="${MEGAMINX_ARCHIVE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${MEGAMINX_ARCHIVE_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p -- "${MEGAMINX_RUN_DIR}/logs/ranks" "${MEGAMINX_RUN_DIR}/history"
MEGAMINX_DEPTH="${MEGAMINX_DEPTH:-120}"
export BEAM_RANK_LOG_DIR="${MEGAMINX_RUN_DIR}/logs/ranks"
export BEAM_NCCL_ID_FILE="${MEGAMINX_RUN_DIR}/nccl-${SLURM_JOB_ID:?missing SLURM_JOB_ID}.bin"
export BEAM_HISTORY_DISK_PATH="${MEGAMINX_RUN_DIR}/history"

python3 -m portable.megaminx_cluster.scripts.preflight_cli --hardware-only --archive-root "${MEGAMINX_ARCHIVE_ROOT}" --run-dir "${MEGAMINX_RUN_DIR}" --gpu-count "${GPU_COUNT}"
python3 -m portable.megaminx_cluster.auto_profile --archive-root "${MEGAMINX_ARCHIVE_ROOT}" --run-dir "${MEGAMINX_RUN_DIR}" --beam "${MEGAMINX_BEAM}"
python3 -m portable.megaminx_cluster.scripts.preflight_cli --archive-root "${MEGAMINX_ARCHIVE_ROOT}" --run-dir "${MEGAMINX_RUN_DIR}" --gpu-count "${GPU_COUNT}" --beam "${MEGAMINX_BEAM}"
source "${MEGAMINX_RUN_DIR}/selected_profile.env"

WORKFLOW_ARGS=(--archive-root "${MEGAMINX_ARCHIVE_ROOT}" --run-dir "${MEGAMINX_RUN_DIR}" --world-size "${GPU_COUNT}" --job-id "${SLURM_JOB_ID}" --puzzle "${MEGAMINX_PUZZLE}" --depth "${MEGAMINX_DEPTH}" --beam "${MEGAMINX_BEAM}" --reflect "${MEGAMINX_REFLECT}")
if [ -n "${MEGAMINX_ORIGINAL_SOLUTION:-}" ]; then WORKFLOW_ARGS+=(--original-solution "${MEGAMINX_ORIGINAL_SOLUTION}"); fi
START_NS="$(date +%s%N)"
python3 -m portable.megaminx_cluster.workflow "${WORKFLOW_ARGS[@]}"
END_NS="$(date +%s%N)"
WALL_US="$(( (END_NS - START_NS) / 1000 ))"

if [ -n "${MEGAMINX_RESULTS_URL:-}" ]; then
  : "${MEGAMINX_AUTHOR_NAME:?set MEGAMINX_AUTHOR_NAME in cluster.env}"
  : "${MEGAMINX_CLUSTER_NAME:?set MEGAMINX_CLUSTER_NAME in cluster.env}"
  python3 -m portable.megaminx_cluster.scripts.finalize_publication --archive-root "${MEGAMINX_ARCHIVE_ROOT}" --run-dir "${MEGAMINX_RUN_DIR}" --job-id "${SLURM_JOB_ID}" --cluster-name "${MEGAMINX_CLUSTER_NAME}" --author-name "${MEGAMINX_AUTHOR_NAME}" --search-mode "${MEGAMINX_REFLECT}" --depth "${MEGAMINX_DEPTH}" --wall-us "${WALL_US}"
  python3 -m portable.megaminx_cluster.scripts.validate_and_publish --run-dir "${MEGAMINX_RUN_DIR}" --url "${MEGAMINX_RESULTS_URL}" --poll
fi
