#!/usr/bin/env bash
set -euo pipefail
: "${MEGAMINX_ARCHIVE_ROOT:?}"
: "${MEGAMINX_AUTOTUNE_RUN_DIR:?}"
: "${MEGAMINX_AUTOTUNE_GPU_IDS:?}"
export CUDA_VISIBLE_DEVICES="${MEGAMINX_AUTOTUNE_GPU_IDS//:/,}"
GPU_COUNT=$(awk -F: '{print NF}' <<<"${MEGAMINX_AUTOTUNE_GPU_IDS}")
cd "${MEGAMINX_ARCHIVE_ROOT}"
python3 -m portable.megaminx_cluster.scripts.verify_archive_payloads --archive-root "${MEGAMINX_ARCHIVE_ROOT}"
python3 -m portable.megaminx_cluster.scripts.preflight_cli --hardware-only --archive-root "${MEGAMINX_ARCHIVE_ROOT}" --run-dir "${MEGAMINX_AUTOTUNE_RUN_DIR}" --gpu-count "${GPU_COUNT}"
exec python3 -m portable.megaminx_cluster.autotune.controller