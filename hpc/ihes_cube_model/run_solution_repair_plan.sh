#!/bin/bash

set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: run_solution_repair_plan.sh <plan.tsv> <runner> <beam_width> <history_base_dir> <job_dir>"
  exit 2
fi

PLAN_TSV="$1"
RUNNER_PATH="$2"
BEAM_WIDTH_ARG="$3"
HISTORY_BASE_DIR_ARG="$4"
JOB_DIR_ARG="$5"

if [ ! -f "${PLAN_TSV}" ]; then
  echo "missing_segment_plan=${PLAN_TSV}"
  exit 2
fi
if [ ! -x "${RUNNER_PATH}" ]; then
  echo "missing_runner=${RUNNER_PATH}"
  exit 2
fi

RANK_ID="${RANK:-0}"

tail -n +2 "${PLAN_TSV}" | while IFS=$'\t' read -r REPAIR_ID SEARCH_DEPTH WINDOW START_STEP TARGET_STEP OLD_SEGMENT_LEN OLD_SEGMENT_PATH TARGET_STATE; do
  export PUZZLE_ID="${REPAIR_ID}"
  export DEPTH_LIMIT="${SEARCH_DEPTH}"
  export BEAM_TARGET_STATE_TEXT="${TARGET_STATE}"
  export HISTORY_DIR="${HISTORY_BASE_DIR_ARG}/${REPAIR_ID}"
  export BEAM_HISTORY_DISK_PATH="${HISTORY_DIR}"
  export BEAM_PREDICT_STATS_VERBOSE=0
  export BEAM_PREDICT_STATS_PATH="${JOB_DIR_ARG}/predict_stats_solution_repair_${REPAIR_ID}_${SLURM_JOB_ID:-manual}.jsonl"
  export BEAM_NCCL_ID_FILE="${JOB_DIR_ARG}/beam_solver_nccl_${SLURM_JOB_ID:-manual}_ihes_solution_repair_${REPAIR_ID}.bin"

  if [ "${RANK_ID}" = "0" ]; then
    rm -f "${BEAM_NCCL_ID_FILE}"
    echo "segment_start repair_id=${REPAIR_ID} search_depth=${SEARCH_DEPTH} window=${WINDOW} start_step=${START_STEP} target_step=${TARGET_STEP} old_segment_len=${OLD_SEGMENT_LEN}"
  fi
  sleep 1

  set +e
  bash "${RUNNER_PATH}" "${REPAIR_ID}" "${SEARCH_DEPTH}" "${BEAM_WIDTH_ARG}"
  rc=$?
  set -e

  if [ "${RANK_ID}" = "0" ]; then
    echo "segment_done repair_id=${REPAIR_ID} rc=${rc}"
  fi
  if [ "${rc}" -ne 0 ]; then
    exit "${rc}"
  fi
done
