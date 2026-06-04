#!/bin/bash
#SBATCH --job-name=beam8a100-s1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
source "${SCRIPT_DIR}/mephi_8xa100_common.sh"

beam_setup_paths
beam_preflight
beam_configure_build stream_benchmark

export BEAM_WEIGHT_DIR="${REPO_DIR}/stream1_weights"
export BEAM_BENCHMARK_REPORT="${TUNING_DIR}/stream1_benchmark_${SLURM_JOB_ID:-manual}.md"
LOG="${TUNING_DIR}/stream1_benchmark_${SLURM_JOB_ID:-manual}.log"

"${BUILD_DIR}/stream_benchmark" "${PUZZLE_ID:-992}" 2>&1 | tee "${LOG}"

BEST_LINE="$(awk '
  /^stream1_micro / {
    cps = "";
    for (i = 1; i <= NF; ++i) {
      if ($i ~ /^candidates_per_sec=/) {
        split($i, a, "=");
        cps = a[2];
      }
    }
    if (cps != "") {
      printf "%020.6f\t%s\n", cps, $0;
    }
  }
' "${LOG}" | sort -nr | head -n 1 | cut -f2- || true)"
if [ -n "${BEST_LINE}" ]; then
  BEST_B_MICRO="$(echo "${BEST_LINE}" | sed -n 's/.*b_micro=\([0-9]*\).*/\1/p')"
  BEST_CONCURRENCY="$(echo "${BEST_LINE}" | sed -n 's/.*concurrent=\([0-9]*\).*/\1/p')"
  BEST_CAND_PER_SEC="$(echo "${BEST_LINE}" | sed -n 's/.*candidates_per_sec=\([0-9.]*\).*/\1/p')"
  {
    echo "export BEAM_B_MICRO=${BEST_B_MICRO}"
    echo "export BEAM_STREAM1_CONCURRENCY=${BEST_CONCURRENCY}"
    echo "export STREAM1_CANDIDATES_PER_SEC=${BEST_CAND_PER_SEC}"
  } > "${TUNING_DIR}/best_stream1.env"
  cp "${TUNING_DIR}/best_stream1.env" "${LOG_DIR}/best_stream1.env"
  echo "best_stream1_env=${LOG_DIR}/best_stream1.env"
  cat "${LOG_DIR}/best_stream1.env"
fi

echo "stream1_report=${BEAM_BENCHMARK_REPORT}"
echo "finished_at=$(date -Is)"
