#!/bin/bash
#SBATCH --job-name=ihes-solve-bucket
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
WORK_DATA_DIR="${WORK_DATA_DIR:-${RUN_DIR}/solve_bucket_data_${SLURM_JOB_ID:-manual}}"
SOLUTIONS_CSV="${SOLUTIONS_CSV:-${RUN_DIR}/solv_uniq.csv}"
WEIGHT_DIR="${WEIGHT_DIR:-${RUN_DIR}/stream1_weights_ihes_bf16}"
BUILD_DIR="${BUILD_DIR:-${RUN_DIR}/build-a100-${SLURM_JOB_ID:-manual}}"
HISTORY_DIR="${HISTORY_DIR:-${RUN_DIR}/history-${SLURM_JOB_ID:-manual}}"
LOG_DIR="${LOG_DIR:-${RUN_DIR}/logs}"
BEAM_COMMON_SH="${BEAM_COMMON_SH:-${REPO_DIR}/hpc/mephi_8xa100_common.sh}"

mkdir -p "${RUN_DIR}" "${WORK_DATA_DIR}" "${BUILD_DIR}" "${HISTORY_DIR}" "${LOG_DIR}"

if [ ! -f "${BEAM_COMMON_SH}" ]; then
  echo "missing_common_script=${BEAM_COMMON_SH}"
  exit 2
fi
source "${BEAM_COMMON_SH}"

clean_work_data_dir() {
  if [ -d "${WORK_DATA_DIR}" ] && [ "${WORK_DATA_DIR}" != "/" ]; then
    rm -rf "${WORK_DATA_DIR}"
  fi
}

cleanup() {
  local rc=$?
  echo "cleanup_start rc=${rc} at $(date -Is)"
  beam_safe_clean_child "${BUILD_DIR}" "build"
  clean_work_data_dir
  if [ "${rc}" -eq 0 ]; then
    beam_safe_clean_child "${HISTORY_DIR}" "history"
  else
    echo "cleanup_keep_history=${HISTORY_DIR}"
  fi
  echo "cleanup_done rc=${rc} at $(date -Is)"
  exit "${rc}"
}
trap cleanup EXIT

beam_setup_paths
beam_preflight

for required in \
  "${DATA_DIR}/puzzle_info.json" \
  "${DATA_DIR}/test.csv" \
  "${WEIGHT_DIR}/manifest.json" \
  "${SOLUTIONS_CSV}"; do
  if [ ! -f "${required}" ]; then
    echo "missing_required_file=${required}"
    exit 2
  fi
done

cp "${DATA_DIR}/puzzle_info.json" "${WORK_DATA_DIR}/puzzle_info.json"
cp "${DATA_DIR}/test.csv" "${WORK_DATA_DIR}/test.csv"

KNOWN_LENGTH="${KNOWN_LENGTH:-23}"
PUZZLE_OFFSET="${PUZZLE_OFFSET:-0}"
PUZZLE_LIMIT="${PUZZLE_LIMIT:-1}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  PUZZLE_OFFSET=$((PUZZLE_OFFSET + SLURM_ARRAY_TASK_ID * PUZZLE_LIMIT))
fi
SOLVE_BUCKET_PLAN="${LOG_DIR}/solve_bucket_plan_${SLURM_JOB_ID:-manual}_${PUZZLE_OFFSET}_${PUZZLE_LIMIT}.tsv"

"${NINJA_VENV_DIR}/bin/python" - "${SOLUTIONS_CSV}" "${KNOWN_LENGTH}" "${PUZZLE_OFFSET}" "${PUZZLE_LIMIT}" "${SOLVE_BUCKET_PLAN}" <<'PY'
import csv
import json
import sys
from pathlib import Path

solutions_csv, known_length_text, offset_text, limit_text, out_path = sys.argv[1:6]
known_length = int(known_length_text)
offset = int(offset_text)
limit = int(limit_text)
text = Path(solutions_csv).read_text(encoding="utf-8").strip()
rows = []
if text.startswith("{"):
    data = json.loads(text)
    for key, paths in data.items():
        puzzle_id = int(key)
        for idx, path in enumerate(paths):
            if not path:
                continue
            rows.append((puzzle_id, idx, path, len(path.split("."))))
else:
    with open(solutions_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            puzzle_id = int(row.get("initial_state_id") or row.get("puzzle_id") or row.get("id"))
            path = row.get("path") or row.get("solution") or row.get("moves") or ""
            if not path:
                continue
            rows.append((puzzle_id, idx, path, len(path.split("."))))

best = {}
for puzzle_id, idx, path, length in rows:
    current = best.get(puzzle_id)
    if current is None or length < current[2]:
        best[puzzle_id] = (idx, path, length)

selected = [
    (puzzle_id, idx, path, length)
    for puzzle_id, (idx, path, length) in best.items()
    if length == known_length
]
selected.sort(key=lambda item: item[0])
selected = selected[offset: offset + limit]

with open(out_path, "w", encoding="utf-8", newline="") as fh:
    writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
    writer.writerow(["puzzle_id", "solution_index", "known_length", "known_path"])
    writer.writerows(selected)

print(f"solve_bucket_known_length={known_length}")
print(f"solve_bucket_total_selected={len(selected)}")
print(f"solve_bucket_plan={out_path}")
for row in selected:
    print(f"solve_bucket_plan_row puzzle_id={row[0]} solution_index={row[1]} known_length={row[3]}")
PY

if [ "$(wc -l < "${SOLVE_BUCKET_PLAN}")" -le 1 ]; then
  echo "solve_bucket_no_puzzles=1"
  exit 0
fi

BEAM_WIDTH="${BEAM_WIDTH:-900000000}"
DEPTH_LIMIT="${DEPTH_LIMIT:-$((KNOWN_LENGTH - 1))}"
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
BEAM_SOLVED_RESULT_CAPACITY="${BEAM_SOLVED_RESULT_CAPACITY:-1048576}"

export BEAM_PUZZLE_INFO_JSON="${WORK_DATA_DIR}/puzzle_info.json"
export BEAM_GENERATOR_PATH="${WORK_DATA_DIR}/puzzle_info.json"
export BEAM_TEST_CSV="${WORK_DATA_DIR}/test.csv"
export BEAM_WEIGHT_DIR="${WEIGHT_DIR}"

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

export BEAM_SOLVE_BUCKET_MODE=1
export BEAM_SOLVE_BUCKET_EXTRA_DEPTHS="${BEAM_SOLVE_BUCKET_EXTRA_DEPTHS:-1}"
export BEAM_SOLVE_BUCKET_KNOWN_LENGTH="${KNOWN_LENGTH}"
export BEAM_HISTORY_DISK_PATH="${HISTORY_DIR}"
export BEAM_SOLVED_RESULT_CAPACITY

echo "solve_bucket_mode=1"
echo "known_length=${KNOWN_LENGTH}"
echo "depth_limit=${DEPTH_LIMIT}"
echo "beam_width=${BEAM_WIDTH}"
echo "global_beam_width_effective=${GLOBAL_BEAM_WIDTH_EFFECTIVE}"
echo "local_beam_width=${LOCAL_BEAM_WIDTH}"
echo "shard_count=${SHARD_COUNT}"
echo "k1_radius=${BEAM_SOLVED_NEIGHBORHOOD_RADIUS}"
echo "solved_result_capacity=${BEAM_SOLVED_RESULT_CAPACITY}"

invert_path_python='
import sys
path = sys.argv[1]
def inv(tok):
    return tok[1:] if tok.startswith("-") else "-" + tok
print(".".join(inv(tok) for tok in reversed(path.split("."))) if path else "")
'

write_reflected_puzzle() {
  local source_solution="$1"
  local synthetic_id="$2"
  SOLUTION_TEXT="${source_solution}" \
  SYNTHETIC_PUZZLE_ID="${synthetic_id}" \
  DATA_DIR="${WORK_DATA_DIR}" \
  "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import json
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
solution = os.environ["SOLUTION_TEXT"]
synthetic_id = int(os.environ["SYNTHETIC_PUZZLE_ID"])
info = json.loads((data_dir / "puzzle_info.json").read_text())
central = list(info["central_state"])
generators = info["generators"]

def apply_path(state, path):
    out = list(state)
    for token in path.split("."):
        out = [out[i] for i in generators[token]]
    return out

def invert_token(token):
    return token[1:] if token.startswith("-") else "-" + token

def invert_path(path):
    return ".".join(invert_token(token) for token in reversed(path.split(".")))

reflected = apply_path(central, solution)
if apply_path(reflected, invert_path(solution)) != central:
    raise RuntimeError("reflected-state roundtrip failed")

row = f'{synthetic_id},"{",".join(str(x) for x in reflected)}"\n'
test_csv = data_dir / "test.csv"
lines = test_csv.read_text().splitlines()
prefix = f"{synthetic_id},"
lines = [line for line in lines if not line.startswith(prefix)]
lines.append(row.rstrip("\n"))
test_csv.write_text("\n".join(lines) + "\n")
print(f"reflected_puzzle_id={synthetic_id}")
PY
}

extract_bucket_best_line() {
  local puzzle_id="$1"
  local result_tsv="$2"
  awk -F'\t' -v id="${puzzle_id}" '
    NR == 1 { next }
    $1 == id {
      if (best == "" || $4 + 0 < best_len) {
        best = $0
        best_len = $4 + 0
      }
    }
    END { if (best != "") print best }
  ' "${result_tsv}"
}

RESULT_TSV="${LOG_DIR}/solve_bucket_len${KNOWN_LENGTH}_${SLURM_JOB_ID:-manual}_${PUZZLE_OFFSET}_${PUZZLE_LIMIT}.tsv"
printf "puzzle_id\tvariant\tknown_length\tfound_length\tdelta\tsolution_path\tknown_path\n" > "${RESULT_TSV}"

tail -n +2 "${SOLVE_BUCKET_PLAN}" | while IFS=$'\t' read -r puzzle_id solution_index known_length known_path; do
  echo "solve_bucket_puzzle_start puzzle_id=${puzzle_id} solution_index=${solution_index} known_length=${known_length}"

  for variant in original reflected; do
    if [ "${variant}" = "original" ]; then
      run_puzzle_id="${puzzle_id}"
    else
      run_puzzle_id=$((9400000 + puzzle_id))
      write_reflected_puzzle "${known_path}" "${run_puzzle_id}"
    fi
    export PUZZLE_ID="${run_puzzle_id}"
    export DEPTH_LIMIT
    beam_safe_clear_history_contents
    beam_prepare_nccl_file "solve_bucket_${variant}_${puzzle_id}"
    export BEAM_SOLVE_BUCKET_RESULT_TSV="${LOG_DIR}/solve_bucket_${variant}_p${puzzle_id}_${SLURM_JOB_ID:-manual}.tsv"
    run_log="${LOG_DIR}/production_runner_ihes_solve_bucket_${variant}_p${puzzle_id}_${SLURM_JOB_ID:-manual}.log"
    beam_torchrun_production "solve_bucket_${variant}_${puzzle_id}" "${run_log}"

    best_line="$(extract_bucket_best_line "${run_puzzle_id}" "${BEAM_SOLVE_BUCKET_RESULT_TSV}" || true)"
    if [ -z "${best_line}" ]; then
      printf "%s\t%s\t%s\t-1\t0\t\t%s\n" "${puzzle_id}" "${variant}" "${known_length}" "${known_path}" >> "${RESULT_TSV}"
      continue
    fi
    found_length="$(printf '%s\n' "${best_line}" | awk -F'\t' '{print $4}')"
    found_path="$(printf '%s\n' "${best_line}" | awk -F'\t' '{print $8}')"
    if [ "${variant}" = "reflected" ]; then
      found_path="$("${NINJA_VENV_DIR}/bin/python" -c "${invert_path_python}" "${found_path}")"
    fi
    delta=$((found_length - known_length))
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "${puzzle_id}" "${variant}" "${known_length}" "${found_length}" "${delta}" "${found_path}" "${known_path}" >> "${RESULT_TSV}"
    echo "solve_bucket_variant_done puzzle_id=${puzzle_id} variant=${variant} found_length=${found_length} delta=${delta}"
  done
done

echo "solve_bucket_result_tsv=${RESULT_TSV}"
echo "finished_at=$(date -Is)"
