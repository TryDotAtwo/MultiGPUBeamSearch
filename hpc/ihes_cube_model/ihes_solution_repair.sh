#!/bin/bash
#SBATCH --job-name=ihes-solution-repair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

set -euo pipefail
ulimit -c 0

BASE_DIR="${BASE_DIR:-/mnt/pool/6/vokirova/beam8a100}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/repo}"
JOB_DIR="${JOB_DIR:-${BASE_DIR}/ihes_cube_model}"
RUN_DIR="${RUN_DIR:-${JOB_DIR}}"
DATA_DIR="${DATA_DIR:-${RUN_DIR}/data}"
REPAIR_TEST_CSV="${REPAIR_TEST_CSV:-${RUN_DIR}/solution_repair_test.csv}"
REPAIR_SEGMENTS_CSV="${REPAIR_SEGMENTS_CSV:-${RUN_DIR}/solution_repair_segments.csv}"
REPAIR_SOLUTIONS_CSV="${REPAIR_SOLUTIONS_CSV:-${RUN_DIR}/solution_repair_solutions.csv}"
WEIGHT_DIR="${WEIGHT_DIR:-${RUN_DIR}/stream1_weights_ihes_bf16}"
BUILD_DIR="${BUILD_DIR:-${RUN_DIR}/build-a100-${SLURM_JOB_ID:-manual}}"
HISTORY_BASE_DIR="${HISTORY_BASE_DIR:-${RUN_DIR}/history-${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
BEAM_COMMON_SH="${BEAM_COMMON_SH:-${REPO_DIR}/hpc/mephi_8xa100_common.sh}"

mkdir -p "${RUN_DIR}" "${BUILD_DIR}" "${HISTORY_BASE_DIR}" "${LOG_DIR}"

if [ ! -f "${BEAM_COMMON_SH}" ]; then
  echo "missing_common_script=${BEAM_COMMON_SH}"
  exit 2
fi
source "${BEAM_COMMON_SH}"

cleanup() {
  local rc=$?
  echo "cleanup_start rc=${rc} at $(date -Is)"
  beam_safe_clean_child "${BUILD_DIR}" "build"
  if [ "${rc}" -eq 0 ]; then
    beam_safe_clean_child "${HISTORY_BASE_DIR}" "history"
  else
    echo "cleanup_keep_history=${HISTORY_BASE_DIR}"
  fi
  echo "cleanup_done rc=${rc} at $(date -Is)"
  exit "${rc}"
}
trap cleanup EXIT

beam_setup_paths

BEAM_SOLUTION_REPAIR_MODE="${BEAM_SOLUTION_REPAIR_MODE:-plan}"
required_files=("${DATA_DIR}/puzzle_info.json" "${WEIGHT_DIR}/manifest.json")
if [ "${BEAM_SOLUTION_REPAIR_MODE}" = "resident" ]; then
  required_files+=("${DATA_DIR}/test.csv" "${SOLUTIONS_CSV:-${RUN_DIR}/solv_uniq.csv}")
else
  required_files+=("${REPAIR_TEST_CSV}" "${REPAIR_SEGMENTS_CSV}" "${REPAIR_SOLUTIONS_CSV}")
fi
for required in "${required_files[@]}"; do
  if [ ! -f "${required}" ]; then
    echo "missing_required_file=${required}"
    exit 2
  fi
done

SOLUTION_ROW_OFFSET="${SOLUTION_ROW_OFFSET:-0}"
if [ -z "${SOLUTION_ROW_INDEX:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  SOLUTION_ROW_INDEX=$((SOLUTION_ROW_OFFSET + SLURM_ARRAY_TASK_ID))
else
  SOLUTION_ROW_INDEX="${SOLUTION_ROW_INDEX:-}"
fi
SOLUTION_JOB_ID="${SOLUTION_JOB_ID:-}"
if [ "${BEAM_SOLUTION_REPAIR_MODE}" != "resident" ] && [ -z "${SOLUTION_ROW_INDEX}" ] && [ -z "${SOLUTION_JOB_ID}" ]; then
  echo "set SOLUTION_ROW_INDEX, SOLUTION_JOB_ID, or submit as a SLURM array"
  exit 2
fi

echo "solution_row_offset=${SOLUTION_ROW_OFFSET}"
echo "solution_row_index=${SOLUTION_ROW_INDEX}"
echo "solution_job_id_filter=${SOLUTION_JOB_ID:-}"

if [ "${BEAM_SOLUTION_REPAIR_MODE}" != "resident" ]; then
PLAN_TSV="${RUN_DIR}/solution_repair_plan_${SLURM_JOB_ID:-manual}_${SOLUTION_ROW_INDEX:-id${SOLUTION_JOB_ID}}.tsv"
META_ENV="${RUN_DIR}/solution_repair_${SLURM_JOB_ID:-manual}_${SOLUTION_ROW_INDEX:-id${SOLUTION_JOB_ID}}.env"
python - "${REPAIR_SOLUTIONS_CSV}" "${REPAIR_SEGMENTS_CSV}" "${META_ENV}" "${PLAN_TSV}" "${SOLUTION_ROW_INDEX}" "${SOLUTION_JOB_ID}" <<'PY'
import csv
import shlex
import sys

solutions_csv, segments_csv, env_file, plan_tsv, row_index_text, solution_job_id_text = sys.argv[1:7]
with open(solutions_csv, newline="", encoding="utf-8") as fh:
    solutions = list(csv.DictReader(fh))
selected = None
if solution_job_id_text:
    for row in solutions:
        if row["solution_job_id"] == solution_job_id_text:
            selected = row
            break
elif row_index_text:
    index = int(row_index_text)
    if index < 0 or index >= len(solutions):
        raise SystemExit(f"solution row index out of range: {index} of {len(solutions)}")
    selected = solutions[index]
if selected is None:
    raise SystemExit("solution job row not found")

with open(segments_csv, newline="", encoding="utf-8") as fh:
    segments = [
        row for row in csv.DictReader(fh)
        if row["solution_job_id"] == selected["solution_job_id"]
    ]
segments.sort(key=lambda row: (int(row["search_depth"]), int(row["start_step"]), int(row["target_step"])))

with open(env_file, "w", encoding="utf-8") as out:
    for key in ["solution_job_id", "puzzle_id", "solution_index", "original_length", "original_path"]:
        out.write(f"REPAIR_{key.upper()}={shlex.quote(selected[key])}\n")
    out.write(f"REPAIR_SEGMENT_COUNT={len(segments)}\n")

fields = [
    "repair_id",
    "search_depth",
    "window",
    "start_step",
    "target_step",
    "old_segment_len",
    "old_segment_path",
    "target_state",
]
with open(plan_tsv, "w", encoding="utf-8", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in segments:
        writer.writerow({key: row[key] for key in fields})
PY
source "${META_ENV}"
else
  REPAIR_SOLUTION_JOB_ID="resident"
  REPAIR_PUZZLE_ID="all"
  REPAIR_SOLUTION_INDEX="${SOLUTION_ROW_INDEX:-0}"
  REPAIR_ORIGINAL_LENGTH="0"
  REPAIR_ORIGINAL_PATH=""
  REPAIR_SEGMENT_COUNT=1
fi

if [ "${REPAIR_SEGMENT_COUNT}" -eq 0 ]; then
  echo "no_segments_for_solution_job=${REPAIR_SOLUTION_JOB_ID}"
  exit 0
fi

BEAM_WIDTH="${BEAM_WIDTH:-900000000}"
BEAM_SOLVED_NEIGHBORHOOD_RADIUS="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS:-5}"
SHARD_COUNT="${SHARD_COUNT:-32}"
STREAM4_BATCH_ALIGNMENT="${STREAM4_BATCH_ALIGNMENT:-1024}"
SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1000000}"
STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES:-262144}"
STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES:-1048576}"
BEAM_B_MICRO="${BEAM_B_MICRO:-8192}"
BEAM_STREAM1_CONCURRENCY="${BEAM_STREAM1_CONCURRENCY:-8}"
BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-8}"
BEAM_STREAM4_ACTIVE_SORT_SLOTS="${BEAM_STREAM4_ACTIVE_SORT_SLOTS:-4}"
BEAM_SHARD_BUFFER_COUNT="${BEAM_SHARD_BUFFER_COUNT:-2}"
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-98304}"
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-8000000}"
BEAM_GPU_HEADROOM_BYTES="${BEAM_GPU_HEADROOM_BYTES:-134217728}"
BEAM_HISTORY_RAM_BYTES="${BEAM_HISTORY_RAM_BYTES:-68719476736}"
BEAM_HISTORY_DISK_BYTES="${BEAM_HISTORY_DISK_BYTES:-4398046511104}"

export BEAM_PUZZLE_INFO_JSON="${DATA_DIR}/puzzle_info.json"
export BEAM_GENERATOR_PATH="${DATA_DIR}/puzzle_info.json"
export BEAM_TEST_CSV="${REPAIR_TEST_CSV}"
export BEAM_WEIGHT_DIR="${WEIGHT_DIR}"

beam_preflight
if [ -n "${BEAM_PREBUILT_RUNNER:-}" ]; then
  if [ ! -x "${BEAM_PREBUILT_RUNNER}" ]; then
    echo "missing_prebuilt_runner=${BEAM_PREBUILT_RUNNER}"
    exit 2
  fi
  ln -sf "${BEAM_PREBUILT_RUNNER}" "${BUILD_DIR}/production_runner"
  echo "using_prebuilt_runner=${BEAM_PREBUILT_RUNNER}"
else
  beam_configure_build production_runner
fi
beam_derive_shard_capacity
beam_validate_manual_config
beam_export_common_runtime
export BEAM_B_MICRO
export BEAM_STREAM1_CONCURRENCY
export BEAM_STREAM3_RING_SLOTS
export BEAM_SHARD_BUFFER_COUNT
beam_export_manual_config

echo "solution_job_id=${REPAIR_SOLUTION_JOB_ID}"
echo "source_puzzle_id=${REPAIR_PUZZLE_ID}"
echo "solution_index=${REPAIR_SOLUTION_INDEX}"
echo "original_length=${REPAIR_ORIGINAL_LENGTH}"
echo "segment_count=${REPAIR_SEGMENT_COUNT}"
echo "beam_width=${BEAM_WIDTH}"
echo "k1_radius=${BEAM_SOLVED_NEIGHBORHOOD_RADIUS}"
echo "build_dir=${BUILD_DIR}"

RESULT_TSV="${LOG_DIR}/solution_repair_segments_${REPAIR_SOLUTION_JOB_ID}_${SLURM_JOB_ID:-manual}.tsv"
printf "solution_job_id\trepair_id\tsearch_depth\twindow\tstart_step\ttarget_step\told_segment_len\tsegment_solved\tsegment_len\tsegment_delta\tsegment_path\n" > "${RESULT_TSV}"

echo "solution_repair_mode=${BEAM_SOLUTION_REPAIR_MODE}"

if [ "${BEAM_SOLUTION_REPAIR_MODE}" = "resident" ]; then
  export BEAM_TEST_CSV="${DATA_DIR}/test.csv"
  export BEAM_REPAIR_SOLUTIONS_CSV="${SOLUTIONS_CSV:-${RUN_DIR}/solv_uniq.csv}"
  export BEAM_REPAIR_K1_RADIUS="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS}"
  export BEAM_REPAIR_SEARCH_DEPTH="${BEAM_REPAIR_SEARCH_DEPTH:-7}"
  export BEAM_REPAIR_FIRST_SOLUTION="${BEAM_REPAIR_FIRST_SOLUTION:-${SOLUTION_ROW_INDEX:-0}}"
  export BEAM_REPAIR_MAX_SOLUTIONS="${BEAM_REPAIR_MAX_SOLUTIONS:-0}"
  export BEAM_REPAIR_RESULT_TSV="${LOG_DIR}/solution_repair_resident_${SLURM_JOB_ID:-manual}_${BEAM_REPAIR_FIRST_SOLUTION}_${BEAM_REPAIR_MAX_SOLUTIONS}.tsv"
  export PUZZLE_ID="${PUZZLE_ID:-1}"
  export DEPTH_LIMIT="${BEAM_REPAIR_SEARCH_DEPTH}"
  export BEAM_TARGET_STATE_TEXT=
  export BEAM_HISTORY_DISK_PATH="${HISTORY_BASE_DIR}/resident"
  beam_prepare_nccl_file "ihes_solution_repair_resident"
  RESIDENT_RUN_LOG="${LOG_DIR}/production_runner_ihes_solution_repair_resident_${SLURM_JOB_ID:-manual}.log"
  echo "repair_resident_solutions_csv=${BEAM_REPAIR_SOLUTIONS_CSV}"
  echo "repair_resident_result_tsv=${BEAM_REPAIR_RESULT_TSV}"
  echo "repair_resident_first_solution=${BEAM_REPAIR_FIRST_SOLUTION}"
  echo "repair_resident_max_solutions=${BEAM_REPAIR_MAX_SOLUTIONS}"
  beam_native_production "solution_repair_resident" "${RESIDENT_RUN_LOG}"
  exit 0
elif [ "${BEAM_SOLUTION_REPAIR_MODE}" = "plan" ]; then
  PLAN_RUN_LOG="${LOG_DIR}/production_runner_ihes_solution_repair_plan_${REPAIR_SOLUTION_JOB_ID}_${SLURM_JOB_ID:-manual}.log"
  beam_torchrun_segment_plan \
    "solution_repair_plan_${REPAIR_SOLUTION_JOB_ID}" \
    "${PLAN_RUN_LOG}" \
    "${PLAN_TSV}" \
    "${BUILD_DIR}/production_runner" \
    "${HISTORY_BASE_DIR}"
  python - "${PLAN_RUN_LOG}" "${PLAN_TSV}" "${REPAIR_SOLUTION_JOB_ID}" >> "${RESULT_TSV}" <<'PY'
import csv
import re
import sys

run_log, plan_tsv, solution_job_id = sys.argv[1:4]
text = open(run_log, encoding="utf-8", errors="ignore").read()
matches_by_repair = {}
for match in re.finditer(r"puzzle_solved=1\b[^\n]*\bpuzzle_id=(\d+)\b[^\n]*\bsolution_length=(\d+)\s+solution=([^\s]+)", text):
    matches_by_repair[match.group(1)] = (int(match.group(2)), match.group(3))

with open(plan_tsv, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        repair_id = row["repair_id"]
        old_len = int(row["old_segment_len"])
        segment_solved = 0
        segment_len = old_len
        segment_path = row["old_segment_path"]
        if repair_id in matches_by_repair:
            segment_solved = 1
            segment_len, segment_path = matches_by_repair[repair_id]
        delta = segment_len - old_len
        print("\t".join(map(str, [
            solution_job_id,
            repair_id,
            row["search_depth"],
            row["window"],
            row["start_step"],
            row["target_step"],
            old_len,
            segment_solved,
            segment_len,
            delta,
            segment_path,
        ])))
PY
elif [ "${BEAM_SOLUTION_REPAIR_MODE}" = "legacy" ] || [ "${BEAM_SOLUTION_REPAIR_MODE}" = "legacy_native" ]; then
tail -n +2 "${PLAN_TSV}" | while IFS=$'\t' read -r REPAIR_ID SEARCH_DEPTH WINDOW START_STEP TARGET_STEP OLD_SEGMENT_LEN OLD_SEGMENT_PATH TARGET_STATE; do
  export PUZZLE_ID="${REPAIR_ID}"
  export DEPTH_LIMIT="${SEARCH_DEPTH}"
  export BEAM_TARGET_STATE_TEXT="${TARGET_STATE}"
  export HISTORY_DIR="${HISTORY_BASE_DIR}/${REPAIR_ID}"
  export BEAM_HISTORY_DISK_PATH="${HISTORY_DIR}"
  export BEAM_PREDICT_STATS_VERBOSE=0
  export BEAM_PREDICT_STATS_PATH="${RUN_DIR}/predict_stats_solution_repair_${REPAIR_ID}_${SLURM_JOB_ID:-manual}.jsonl"
  beam_prepare_nccl_file "ihes_solution_repair_${REPAIR_ID}"
  RUN_LOG="${LOG_DIR}/production_runner_ihes_solution_repair_${REPAIR_ID}_${SLURM_JOB_ID:-manual}.log"

  echo "segment_start repair_id=${REPAIR_ID} search_depth=${SEARCH_DEPTH} window=${WINDOW} start_step=${START_STEP} target_step=${TARGET_STEP} old_segment_len=${OLD_SEGMENT_LEN}"
  set +e
  if [ "${BEAM_SOLUTION_REPAIR_MODE}" = "legacy_native" ]; then
    beam_native_production "solution_repair_${REPAIR_ID}" "${RUN_LOG}"
  else
    beam_torchrun_production "solution_repair_${REPAIR_ID}" "${RUN_LOG}"
  fi
  rc=$?
  set -e
  if [ "${rc}" -eq 0 ]; then
    beam_safe_clean_child "${HISTORY_DIR}" "segment_history"
  else
    echo "segment_keep_history=${HISTORY_DIR}"
  fi

  python - "${RUN_LOG}" "${OLD_SEGMENT_PATH}" "${OLD_SEGMENT_LEN}" "${REPAIR_SOLUTION_JOB_ID}" "${REPAIR_ID}" "${SEARCH_DEPTH}" "${WINDOW}" "${START_STEP}" "${TARGET_STEP}" "${rc}" >> "${RESULT_TSV}" <<'PY'
import re
import sys

run_log, old_segment, old_len_text, solution_job_id, repair_id, search_depth, window, start_step, target_step, rc_text = sys.argv[1:11]
old_len = int(old_len_text)
rc = int(rc_text)
segment_solved = 0
segment_len = old_len
segment_path = old_segment
if rc == 0:
    text = open(run_log, encoding="utf-8", errors="ignore").read()
    matches = re.findall(r"puzzle_solved=1\b[^\n]*\bsolution_length=(\d+)\s+solution=([^\s]+)", text)
    if matches:
        segment_solved = 1
        segment_len = int(matches[-1][0])
        segment_path = matches[-1][1]
delta = segment_len - old_len
print("\t".join(map(str, [
    solution_job_id,
    repair_id,
    search_depth,
    window,
    start_step,
    target_step,
    old_len,
    segment_solved,
    segment_len,
    delta,
    segment_path,
])))
PY
done
else
  echo "invalid_solution_repair_mode=${BEAM_SOLUTION_REPAIR_MODE}"
  echo "expected_solution_repair_mode=resident_plan_legacy_or_legacy_native"
  exit 2
fi

SUMMARY_TXT="${LOG_DIR}/solution_repair_summary_${REPAIR_SOLUTION_JOB_ID}_${SLURM_JOB_ID:-manual}.txt"
python - "${RESULT_TSV}" "${REPAIR_ORIGINAL_PATH}" "${REPAIR_ORIGINAL_LENGTH}" > "${SUMMARY_TXT}" <<'PY'
import csv
import sys
from collections import defaultdict

result_tsv, original_path, original_length_text = sys.argv[1:4]
original_length = int(original_length_text)
rows_by_depth = defaultdict(list)
with open(result_tsv, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh, delimiter="\t"):
        rows_by_depth[int(row["search_depth"])].append(row)

best = None
for depth in sorted(rows_by_depth):
    rows = sorted(rows_by_depth[depth], key=lambda r: int(r["start_step"]))
    parts = [row["segment_path"] for row in rows if row["segment_path"]]
    candidate = ".".join(parts)
    candidate_len = 0 if not candidate else len([item for item in candidate.split(".") if item])
    solved_segments = sum(1 for row in rows if row["segment_solved"] == "1")
    delta = candidate_len - original_length
    print(f"repair_depth={depth} candidate_solution_length={candidate_len} delta={delta} solved_segments={solved_segments}/{len(rows)} candidate_solution={candidate}")
    item = (candidate_len, depth, candidate)
    if best is None or item < best:
        best = item
if best is not None:
    print(f"best_candidate_solution_length={best[0]}")
    print(f"best_repair_depth={best[1]}")
    print(f"best_candidate_solution={best[2]}")
else:
    print(f"best_candidate_solution_length={original_length}")
    print(f"best_candidate_solution={original_path}")
PY
cat "${SUMMARY_TXT}"
echo "result_tsv=${RESULT_TSV}"
echo "summary_txt=${SUMMARY_TXT}"
echo "finished_at=$(date -Is)"
