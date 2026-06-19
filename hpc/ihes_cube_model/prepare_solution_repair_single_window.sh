#!/bin/bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/mnt/pool/6/vokirova/beam8a100}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/repo}"
JOB_DIR="${JOB_DIR:-${BASE_DIR}/ihes_cube_model}"
DATA_DIR="${DATA_DIR:-${JOB_DIR}/data}"
SOLUTIONS_CSV="${SOLUTIONS_CSV:-${JOB_DIR}/solv_uniq.csv}"
K1_RADIUS="${K1_RADIUS:-5}"
SEARCH_DEPTH="${SEARCH_DEPTH:-7}"
BEAM_WIDTH="${BEAM_WIDTH:-900000000}"
PUZZLE_IDS="${PUZZLE_IDS:-}"

mkdir -p "${JOB_DIR}"

if [ ! -f "${DATA_DIR}/puzzle_info.json" ]; then
  echo "missing_puzzle_info=${DATA_DIR}/puzzle_info.json"
  exit 2
fi
if [ ! -f "${DATA_DIR}/test.csv" ]; then
  echo "missing_test_csv=${DATA_DIR}/test.csv"
  exit 2
fi
if [ ! -f "${SOLUTIONS_CSV}" ]; then
  echo "missing_solutions_csv=${SOLUTIONS_CSV}"
  exit 2
fi

cmd=(
  python3 "${REPO_DIR}/tools/generate_segment_repair_jobs.py"
  --puzzle-info "${DATA_DIR}/puzzle_info.json"
  --test-csv "${DATA_DIR}/test.csv"
  --solutions "${SOLUTIONS_CSV}"
  --out-test-csv "${JOB_DIR}/solution_repair_test.csv"
  --out-jobs-csv "${JOB_DIR}/solution_repair_segments.csv"
  --out-solution-jobs-csv "${JOB_DIR}/solution_repair_solutions.csv"
  --k1-radius "${K1_RADIUS}"
  --search-depth "${SEARCH_DEPTH}"
  --beam-width "${BEAM_WIDTH}"
)

if [ -n "${PUZZLE_IDS}" ]; then
  cmd+=(--puzzle-ids "${PUZZLE_IDS}")
fi

echo "solution_repair_single_window=1"
echo "k1_radius=${K1_RADIUS}"
echo "search_depth=${SEARCH_DEPTH}"
echo "window=$((K1_RADIUS + SEARCH_DEPTH))"
echo "beam_width=${BEAM_WIDTH}"
echo "solutions_csv=${SOLUTIONS_CSV}"
"${cmd[@]}"
