#!/usr/bin/env bash
set -euo pipefail

: "${MEGAMINX_ARCHIVE_ROOT:?missing MEGAMINX_ARCHIVE_ROOT}"
: "${MEGAMINX_RUN_DIR:?missing MEGAMINX_RUN_DIR}"
: "${MEGAMINX_GPU_IDS:?missing MEGAMINX_GPU_IDS}"
: "${MEGAMINX_BEAM:?missing MEGAMINX_BEAM}"
: "${MEGAMINX_PUZZLE:?missing MEGAMINX_PUZZLE}"
: "${MEGAMINX_REFLECT:?missing MEGAMINX_REFLECT}"

GPU_CSV="${MEGAMINX_GPU_IDS//:/,}"
IFS=':' read -r -a GPU_ARRAY <<< "${MEGAMINX_GPU_IDS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
export CUDA_VISIBLE_DEVICES="${GPU_CSV}"
export PYTHONPATH="${MEGAMINX_ARCHIVE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${MEGAMINX_ARCHIVE_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

mkdir -p -- "${MEGAMINX_RUN_DIR}/logs/ranks" "${MEGAMINX_RUN_DIR}/history"
MEGAMINX_DEPTH="${MEGAMINX_DEPTH:-120}"
export BEAM_RANK_LOG_DIR="${MEGAMINX_RUN_DIR}/logs/ranks"
export BEAM_NCCL_ID_FILE="${MEGAMINX_RUN_DIR}/nccl-${SLURM_JOB_ID:?missing SLURM_JOB_ID}.bin"
export BEAM_HISTORY_DISK_PATH="${MEGAMINX_RUN_DIR}/history"

python3 -m portable.megaminx_cluster.scripts.preflight_cli \
  --archive-root "${MEGAMINX_ARCHIVE_ROOT}" \
  --run-dir "${MEGAMINX_RUN_DIR}" \
  --gpu-count "${GPU_COUNT}" \
  --beam "${MEGAMINX_BEAM}"
source "${MEGAMINX_RUN_DIR}/selected_profile.env"

python3 -m torch.distributed.run \
  --nnodes=1 \
  --nproc-per-node="${GPU_COUNT}" \
  --node-rank=0 \
  --rdzv-backend=c10d \
  --rdzv-endpoint=127.0.0.1:29500 \
  --rdzv-id="megaminx-${SLURM_JOB_ID}" \
  --no-python \
  "${MEGAMINX_ARCHIVE_ROOT}/bin/production_runner" \
  "${MEGAMINX_PUZZLE}" "${MEGAMINX_DEPTH}" "${MEGAMINX_BEAM}" \
  2>&1 | tee "${MEGAMINX_RUN_DIR}/logs/production.log"
