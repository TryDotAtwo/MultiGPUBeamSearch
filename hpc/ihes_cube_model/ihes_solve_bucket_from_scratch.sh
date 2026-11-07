#!/bin/bash
#SBATCH --job-name=ihes-bucket-fresh
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
WORK_DATA_DIR="${WORK_DATA_DIR:-${RUN_DIR}/solve_bucket_fresh_data_${SLURM_JOB_ID:-manual}}"
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

KNOWN_LENGTHS="${KNOWN_LENGTHS:-23,24}"
PUZZLE_OFFSET="${PUZZLE_OFFSET:-0}"
PUZZLE_LIMIT="${PUZZLE_LIMIT:-1}"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  PUZZLE_OFFSET=$((PUZZLE_OFFSET + SLURM_ARRAY_TASK_ID * PUZZLE_LIMIT))
fi
FRESH_RUN_TAG="${FRESH_RUN_TAG:-len${KNOWN_LENGTHS}_offset${PUZZLE_OFFSET}_limit${PUZZLE_LIMIT}}"
PROGRESS_DIR="${PROGRESS_DIR:-${LOG_DIR}/solve_bucket_fresh_progress_${FRESH_RUN_TAG}}"
mkdir -p "${PROGRESS_DIR}"
SOLVE_BUCKET_PLAN="${LOG_DIR}/solve_bucket_fresh_plan_${SLURM_JOB_ID:-manual}_${PUZZLE_OFFSET}_${PUZZLE_LIMIT}.tsv"

"${NINJA_VENV_DIR}/bin/python" - "${SOLUTIONS_CSV}" "${KNOWN_LENGTHS}" "${PUZZLE_OFFSET}" "${PUZZLE_LIMIT}" "${SOLVE_BUCKET_PLAN}" <<'PY'
import csv
import json
import sys
from pathlib import Path

solutions_csv, known_lengths_text, offset_text, limit_text, out_path = sys.argv[1:6]
known_lengths = {int(x) for x in known_lengths_text.replace(",", " ").split() if x}
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
    if length in known_lengths
]
selected.sort(key=lambda item: item[0])
selected = selected[offset: offset + limit]

with open(out_path, "w", encoding="utf-8", newline="") as fh:
    writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
    writer.writerow(["puzzle_id", "solution_index", "known_length", "baseline_path"])
    for puzzle_id, idx, path, length in selected:
        writer.writerow([puzzle_id, idx, length, path])

print(f"solve_bucket_known_lengths={','.join(str(x) for x in sorted(known_lengths))}")
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
DEPTH_LIMIT_DEFAULT="${DEPTH_LIMIT:-}"
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
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-$(beam_default_final_exchange_scale_ppm)}"
BEAM_GPU_HEADROOM_BYTES="${BEAM_GPU_HEADROOM_BYTES:-134217728}"
BEAM_HISTORY_RAM_BYTES="${BEAM_HISTORY_RAM_BYTES:-68719476736}"
BEAM_HISTORY_DISK_BYTES="${BEAM_HISTORY_DISK_BYTES:-4398046511104}"
BEAM_SOLVED_RESULT_CAPACITY="${BEAM_SOLVED_RESULT_CAPACITY:-1048576}"
BEAM_SOLVE_BUCKET_GATHER_SCRATCH_BYTES="${BEAM_SOLVE_BUCKET_GATHER_SCRATCH_BYTES:-2147483648}"
PUBLISH_RESULTS_REPO_URL="${PUBLISH_RESULTS_REPO_URL:-}"
PUBLISH_RESULTS_DIR="${PUBLISH_RESULTS_DIR:-${RUN_DIR}/public-results}"
PUBLISH_RESULTS_BRANCH="${PUBLISH_RESULTS_BRANCH:-main}"
PUBLISH_RESULTS_GIT_NAME="${PUBLISH_RESULTS_GIT_NAME:-beam-results-bot}"
PUBLISH_RESULTS_GIT_EMAIL="${PUBLISH_RESULTS_GIT_EMAIL:-beam-results-bot@users.noreply.github.com}"

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
export BEAM_HISTORY_DISK_PATH="${HISTORY_DIR}"
export BEAM_SOLVED_RESULT_CAPACITY
export BEAM_SOLVE_BUCKET_GATHER_SCRATCH_BYTES

echo "solve_bucket_mode=1"
echo "solve_bucket_fresh_original_reflect=1"
echo "known_lengths=${KNOWN_LENGTHS}"
echo "depth_limit_default=${DEPTH_LIMIT_DEFAULT:-auto}"
echo "beam_width=${BEAM_WIDTH}"
echo "global_beam_width_effective=${GLOBAL_BEAM_WIDTH_EFFECTIVE}"
echo "local_beam_width=${LOCAL_BEAM_WIDTH}"
echo "shard_count=${SHARD_COUNT}"
echo "k1_radius=${BEAM_SOLVED_NEIGHBORHOOD_RADIUS}"
echo "solved_result_capacity=${BEAM_SOLVED_RESULT_CAPACITY}"
echo "solve_bucket_gather_scratch_bytes=${BEAM_SOLVE_BUCKET_GATHER_SCRATCH_BYTES}"
echo "publish_results_repo_url=${PUBLISH_RESULTS_REPO_URL:-disabled}"
echo "publish_results_dir=${PUBLISH_RESULTS_DIR}"

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

normalize_known_path() {
  local puzzle_id="$1"
  local path="$2"
  PUZZLE_ID_TEXT="${puzzle_id}" \
  SOLUTION_TEXT="${path}" \
  DATA_DIR="${WORK_DATA_DIR}" \
  "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import csv
import json
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
puzzle_id = int(os.environ["PUZZLE_ID_TEXT"])
solution = os.environ["SOLUTION_TEXT"]
info = json.loads((data_dir / "puzzle_info.json").read_text())
central = list(info["central_state"])
generators = info["generators"]

def apply_path(state, path):
    out = list(state)
    for token in path.split(".") if path else []:
        out = [out[i] for i in generators[token]]
    return out

def invert_token(token):
    return token[1:] if token.startswith("-") else "-" + token

def invert_path(path):
    return ".".join(invert_token(token) for token in reversed(path.split("."))) if path else ""

def reverse_only(path):
    return ".".join(reversed(path.split("."))) if path else ""

def toggle_only(path):
    return ".".join(invert_token(token) for token in path.split(".")) if path else ""

with (data_dir / "test.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        if int(row["initial_state_id"]) == puzzle_id:
            initial = [int(x) for x in row["initial_state"].split(",")]
            break
    else:
        raise RuntimeError(f"puzzle id not found: {puzzle_id}")

variants = [
    ("as_is", solution),
    ("reverse_only", reverse_only(solution)),
    ("invert", invert_path(solution)),
    ("toggle_only", toggle_only(solution)),
]
for name, candidate in variants:
    if apply_path(initial, candidate) == central:
        print(candidate)
        print(f"known_path_orientation={name}", file=os.sys.stderr)
        raise SystemExit
raise RuntimeError(f"known solution does not solve puzzle {puzzle_id} in any supported orientation")
PY
}

solution_solves_original() {
  local puzzle_id="$1"
  local path="$2"
  PUZZLE_ID_TEXT="${puzzle_id}" \
  SOLUTION_TEXT="${path}" \
  DATA_DIR="${WORK_DATA_DIR}" \
  "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import csv
import json
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR"])
puzzle_id = int(os.environ["PUZZLE_ID_TEXT"])
solution = os.environ["SOLUTION_TEXT"]
info = json.loads((data_dir / "puzzle_info.json").read_text())
central = list(info["central_state"])
generators = info["generators"]

def apply_path(state, path):
    out = list(state)
    for token in path.split(".") if path else []:
        out = [out[i] for i in generators[token]]
    return out

with (data_dir / "test.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        if int(row["initial_state_id"]) == puzzle_id:
            initial = [int(x) for x in row["initial_state"].split(",")]
            break
    else:
        raise RuntimeError(f"puzzle id not found: {puzzle_id}")

raise SystemExit(0 if apply_path(initial, solution) == central else 1)
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

append_bucket_results() {
  local variant="$1"
  local puzzle_id="$2"
  local run_puzzle_id="$3"
  local known_length="$4"
  local runner_tsv="$5"
  local aggregate_tsv="$6"
  local source_index="$7"
  local source_solution="$8"
  local invert_flag="$9"
  VARIANT="${variant}" \
  PUZZLE_ID_TEXT="${puzzle_id}" \
  RUN_PUZZLE_ID_TEXT="${run_puzzle_id}" \
  KNOWN_LENGTH_TEXT="${known_length}" \
  RUNNER_TSV="${runner_tsv}" \
  AGGREGATE_TSV="${aggregate_tsv}" \
  SOURCE_INDEX_TEXT="${source_index}" \
  SOURCE_SOLUTION_TEXT="${source_solution}" \
  INVERT_FLAG="${invert_flag}" \
  DATA_DIR="${WORK_DATA_DIR}" \
  "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import csv
import json
import os
from pathlib import Path

variant = os.environ["VARIANT"]
puzzle_id = int(os.environ["PUZZLE_ID_TEXT"])
run_puzzle_id = int(os.environ["RUN_PUZZLE_ID_TEXT"])
known_length = int(os.environ["KNOWN_LENGTH_TEXT"])
runner_tsv = Path(os.environ["RUNNER_TSV"])
aggregate_tsv = Path(os.environ["AGGREGATE_TSV"])
source_index = int(os.environ["SOURCE_INDEX_TEXT"])
source_solution = os.environ["SOURCE_SOLUTION_TEXT"]
invert_flag = os.environ["INVERT_FLAG"] == "1"
data_dir = Path(os.environ["DATA_DIR"])

info = json.loads((data_dir / "puzzle_info.json").read_text())
central = list(info["central_state"])
generators = info["generators"]

def apply_path(state, path):
    out = list(state)
    for token in path.split(".") if path else []:
        out = [out[i] for i in generators[token]]
    return out

def invert_token(token):
    return token[1:] if token.startswith("-") else "-" + token

def invert_path(path):
    return ".".join(invert_token(token) for token in reversed(path.split("."))) if path else ""

initial = None
with (data_dir / "test.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        if int(row["initial_state_id"]) == puzzle_id:
            initial = [int(x) for x in row["initial_state"].split(",")]
            break
if initial is None:
    raise RuntimeError(f"puzzle id not found: {puzzle_id}")

best_len = None
count = 0
invalid = 0
if runner_tsv.exists():
    with runner_tsv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        with aggregate_tsv.open("a", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, delimiter="\t", lineterminator="\n")
            for row in reader:
                path = row.get("solution_path", "")
                if not path:
                    continue
                if invert_flag:
                    path = invert_path(path)
                found_len = len(path.split(".")) if path else 0
                valid = apply_path(initial, path) == central
                if not valid:
                    invalid += 1
                    print(f"solve_bucket_invalid_candidate puzzle_id={puzzle_id} variant={variant} found_length={found_len}")
                    continue
                delta = found_len - known_length
                writer.writerow([
                    puzzle_id,
                    variant,
                    source_index,
                    known_length,
                    found_len,
                    delta,
                    run_puzzle_id,
                    row.get("depth_index", ""),
                    row.get("found_depth", ""),
                    row.get("owner_rank", ""),
                    path,
                    source_solution,
                ])
                count += 1
                if best_len is None or found_len < best_len:
                    best_len = found_len
print(f"appended_bucket_results variant={variant} puzzle_id={puzzle_id} count={count} invalid={invalid} best_length={best_len if best_len is not None else -1}")
PY
}

best_aggregate_length() {
  local puzzle_id="$1"
  local aggregate_tsv="$2"
  awk -F'\t' -v id="${puzzle_id}" '
    NR == 1 { next }
    $1 == id && $5 >= 0 {
      if (best == "" || $5 + 0 < best) best = $5 + 0
    }
    END { if (best == "") print -1; else print best }
  ' "${aggregate_tsv}"
}
publish_puzzle_results() {
  local puzzle_id="$1"
  local known_length="$2"
  local result_tsv="$3"
  local best_len="$4"

  if [ -z "${PUBLISH_RESULTS_REPO_URL}" ] && [ ! -d "${PUBLISH_RESULTS_DIR}/.git" ]; then
    echo "publish_results_skip puzzle_id=${puzzle_id} reason=disabled"
    return 0
  fi

  mkdir -p "$(dirname "${PUBLISH_RESULTS_DIR}")"
  if [ ! -d "${PUBLISH_RESULTS_DIR}/.git" ]; then
    echo "publish_results_clone repo=${PUBLISH_RESULTS_REPO_URL} dir=${PUBLISH_RESULTS_DIR}"
    if ! git clone --depth 1 --branch "${PUBLISH_RESULTS_BRANCH}" "${PUBLISH_RESULTS_REPO_URL}" "${PUBLISH_RESULTS_DIR}"; then
      echo "publish_results_failed puzzle_id=${puzzle_id} stage=clone"
      return 0
    fi
  fi

  (
    if command -v flock >/dev/null 2>&1; then
      exec 9>"${RUN_DIR}/publish-results.lock"
      flock 9
    fi

    git -C "${PUBLISH_RESULTS_DIR}" config user.name "${PUBLISH_RESULTS_GIT_NAME}"
    git -C "${PUBLISH_RESULTS_DIR}" config user.email "${PUBLISH_RESULTS_GIT_EMAIL}"
    git -C "${PUBLISH_RESULTS_DIR}" checkout "${PUBLISH_RESULTS_BRANCH}" >/dev/null 2>&1 || git -C "${PUBLISH_RESULTS_DIR}" checkout -b "${PUBLISH_RESULTS_BRANCH}"
    git -C "${PUBLISH_RESULTS_DIR}" pull --rebase origin "${PUBLISH_RESULTS_BRANCH}" || true

    PUZZLE_ID_TEXT="${puzzle_id}" \
    KNOWN_LENGTH_TEXT="${known_length}" \
    BEST_LENGTH_TEXT="${best_len}" \
    RESULT_TSV="${result_tsv}" \
    RESULTS_REPO_DIR="${PUBLISH_RESULTS_DIR}" \
    FRESH_RUN_TAG_TEXT="${FRESH_RUN_TAG}" \
    SLURM_JOB_ID_TEXT="${SLURM_JOB_ID:-manual}" \
    SLURM_ARRAY_TASK_ID_TEXT="${SLURM_ARRAY_TASK_ID:-}" \
    BEAM_WIDTH_TEXT="${BEAM_WIDTH}" \
    K1_RADIUS_TEXT="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS}" \
    "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import csv
import os
from pathlib import Path

puzzle_id = os.environ["PUZZLE_ID_TEXT"]
known_length = int(os.environ["KNOWN_LENGTH_TEXT"])
best_len = int(os.environ["BEST_LENGTH_TEXT"])
result_tsv = Path(os.environ["RESULT_TSV"])
repo = Path(os.environ["RESULTS_REPO_DIR"])
base = repo / "data" / "ihes_cube"
puzzle_dir = base / "puzzles" / f"p{int(puzzle_id):04d}"
puzzle_dir.mkdir(parents=True, exist_ok=True)

rows = []
if result_tsv.exists():
    with result_tsv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = reader.fieldnames or []
        for row in reader:
            if row.get("puzzle_id") == puzzle_id:
                rows.append(row)
else:
    fieldnames = []

if not fieldnames:
    fieldnames = [
        "puzzle_id", "variant", "source_index", "known_length", "found_length",
        "delta", "run_puzzle_id", "depth_index", "found_depth", "owner_rank",
        "solution_path", "source_solution",
    ]

with (puzzle_dir / "solutions.tsv").open("w", encoding="utf-8", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

unique = {}
for row in rows:
    path = row.get("solution_path", "")
    if path and path not in unique:
        unique[path] = row
with (puzzle_dir / "unique_solutions.tsv").open("w", encoding="utf-8", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(unique.values())

variant_counts = {}
length_counts = {}
for row in rows:
    variant_counts[row.get("variant", "")] = variant_counts.get(row.get("variant", ""), 0) + 1
    length_counts[row.get("found_length", "")] = length_counts.get(row.get("found_length", ""), 0) + 1

metadata = {
    "puzzle_id": puzzle_id,
    "known_length": str(known_length),
    "best_length": str(best_len),
    "delta": str(best_len - known_length if best_len >= 0 else 0),
    "records": str(len(rows)),
    "unique_solutions": str(len(unique)),
    "fresh_run_tag": os.environ["FRESH_RUN_TAG_TEXT"],
    "slurm_job_id": os.environ["SLURM_JOB_ID_TEXT"],
    "slurm_array_task_id": os.environ["SLURM_ARRAY_TASK_ID_TEXT"],
    "beam_width": os.environ["BEAM_WIDTH_TEXT"],
    "k1_radius": os.environ["K1_RADIUS_TEXT"],
}
with (puzzle_dir / "metadata.env").open("w", encoding="utf-8") as out:
    for key, value in metadata.items():
        out.write(f"{key}={value}\n")

summary_lines = [
    f"# IHES Cube Puzzle {puzzle_id}",
    "",
    f"known_length: {known_length}",
    f"best_length: {best_len}",
    f"delta: {metadata['delta']}",
    f"records: {len(rows)}",
    f"unique_solutions: {len(unique)}",
    "",
    "## Variant Counts",
    "",
]
for key in sorted(variant_counts):
    summary_lines.append(f"- {key}: {variant_counts[key]}")
summary_lines += ["", "## Length Counts", ""]
for key in sorted(length_counts, key=lambda x: int(x) if x else -1):
    summary_lines.append(f"- {key}: {length_counts[key]}")
(puzzle_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

# Rebuild compact global indexes from per-puzzle metadata.
index_rows = []
for meta_path in sorted((base / "puzzles").glob("p*/metadata.env")):
    item = {}
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            item[k] = v
    if item:
        index_rows.append(item)
index_fields = ["puzzle_id", "known_length", "best_length", "delta", "records", "unique_solutions", "variants", "source_files", "fresh_run_tag", "slurm_job_id", "slurm_array_task_id", "beam_width", "k1_radius"]
with (base / "index.tsv").open("w", encoding="utf-8", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=index_fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in index_fields} for row in index_rows)
with (base / "improvements.tsv").open("w", encoding="utf-8", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=index_fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for item in index_rows:
        try:
            if int(item.get("delta", "0")) < 0:
                writer.writerow({field: item.get(field, "") for field in index_fields})
        except ValueError:
            pass
print(f"publish_results_prepared puzzle_id={puzzle_id} records={len(rows)} unique={len(unique)}")
PY

    git -C "${PUBLISH_RESULTS_DIR}" add "data/ihes_cube"
    if git -C "${PUBLISH_RESULTS_DIR}" diff --cached --quiet; then
      echo "publish_results_no_changes puzzle_id=${puzzle_id}"
      return 0
    fi
    git -C "${PUBLISH_RESULTS_DIR}" commit -m "Add IHES puzzle ${puzzle_id} bucket results"
    if git -C "${PUBLISH_RESULTS_DIR}" push origin "${PUBLISH_RESULTS_BRANCH}"; then
      echo "publish_results_pushed puzzle_id=${puzzle_id} repo=${PUBLISH_RESULTS_REPO_URL:-existing}"
    else
      echo "publish_results_failed puzzle_id=${puzzle_id} stage=push"
    fi
  ) || echo "publish_results_failed puzzle_id=${puzzle_id} stage=publish"
}

write_reflected_sources() {
  local puzzle_id="$1"
  local aggregate_tsv="$2"
  local out_tsv="$3"
  PUZZLE_ID_TEXT="${puzzle_id}" \
  AGGREGATE_TSV="${aggregate_tsv}" \
  OUT_TSV="${out_tsv}" \
  DATA_DIR="${WORK_DATA_DIR}" \
  "${NINJA_VENV_DIR}/bin/python" - <<'PY'
import csv
import hashlib
import json
import os
from pathlib import Path

puzzle_id = int(os.environ["PUZZLE_ID_TEXT"])
aggregate_tsv = Path(os.environ["AGGREGATE_TSV"])
out_tsv = Path(os.environ["OUT_TSV"])
data_dir = Path(os.environ["DATA_DIR"])
info = json.loads((data_dir / "puzzle_info.json").read_text())
central = list(info["central_state"])
generators = info["generators"]

def apply_path(state, path):
    out = list(state)
    for token in path.split(".") if path else []:
        out = [out[i] for i in generators[token]]
    return out

seen_path = set()
seen_state = set()
rows = []
if aggregate_tsv.exists():
    with aggregate_tsv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if int(row.get("puzzle_id", -1)) != puzzle_id or row.get("variant") != "original":
                continue
            path = row.get("solution_path", "")
            if not path or path in seen_path:
                continue
            seen_path.add(path)
            reflected_state = apply_path(central, path)
            state_key = ",".join(str(x) for x in reflected_state)
            state_hash = hashlib.sha256(state_key.encode("ascii")).hexdigest()[:16]
            if state_hash in seen_state:
                print(f"solve_bucket_fresh_skip_duplicate_reflected_state puzzle_id={puzzle_id} state_hash={state_hash}")
                continue
            seen_state.add(state_hash)
            rows.append((len(rows) + 1, state_hash, path))
with out_tsv.open("w", encoding="utf-8", newline="") as out:
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")
    writer.writerow(["source_index", "state_hash", "source_solution"])
    writer.writerows(rows)
print(f"solve_bucket_fresh_reflected_sources puzzle_id={puzzle_id} unique_states={len(rows)} unique_paths={len(seen_path)}")
PY
}

run_bucket_once() {
  local label="$1"
  local puzzle_id="$2"
  local run_puzzle_id="$3"
  local known_length="$4"
  local result_tsv="$5"
  local depth_limit="$6"
  export PUZZLE_ID="${run_puzzle_id}"
  export DEPTH_LIMIT="${depth_limit}"
  export BEAM_SOLVE_BUCKET_KNOWN_LENGTH="${known_length}"
  export BEAM_SOLVE_BUCKET_RESULT_TSV="${result_tsv}"
  beam_safe_clear_history_contents
  beam_prepare_nccl_file "${label}_${puzzle_id}"
  local run_log="${LOG_DIR}/production_runner_ihes_${label}_p${puzzle_id}_${SLURM_JOB_ID:-manual}.log"
  beam_torchrun_production "${label}_${puzzle_id}" "${run_log}"
}

RESULT_TSV="${LOG_DIR}/solve_bucket_fresh_${FRESH_RUN_TAG}.tsv"
if [ ! -f "${RESULT_TSV}" ]; then
  printf "puzzle_id\tvariant\tsource_index\tknown_length\tfound_length\tdelta\trun_puzzle_id\tdepth_index\tfound_depth\towner_rank\tsolution_path\tsource_solution\n" > "${RESULT_TSV}"
fi

tail -n +2 "${SOLVE_BUCKET_PLAN}" | while IFS=$'\t' read -r puzzle_id solution_index known_length baseline_path; do
  echo "solve_bucket_fresh_puzzle_start puzzle_id=${puzzle_id} solution_index=${solution_index} known_length=${known_length}"
  run_depth_limit="${DEPTH_LIMIT_DEFAULT:-${known_length}}"

  original_tsv="${LOG_DIR}/solve_bucket_fresh_original_p${puzzle_id}_${FRESH_RUN_TAG}.tsv"
  original_done="${PROGRESS_DIR}/p${puzzle_id}_original.done"
  if [ -f "${original_done}" ]; then
    echo "solve_bucket_fresh_resume_skip_original puzzle_id=${puzzle_id} done=${original_done}"
  else
    run_bucket_once "solve_bucket_fresh_original" "${puzzle_id}" "${puzzle_id}" "${known_length}" "${original_tsv}" "${run_depth_limit}"
    append_bucket_results "original" "${puzzle_id}" "${puzzle_id}" "${known_length}" "${original_tsv}" "${RESULT_TSV}" 0 "" 0
    printf "done_at=%s\n" "$(date -Is)" > "${original_done}"
  fi

  best_len="$(best_aggregate_length "${puzzle_id}" "${RESULT_TSV}")"
  echo "solve_bucket_fresh_original_done puzzle_id=${puzzle_id} best_len=${best_len} known_length=${known_length}"
  if [ "${best_len}" -ge 0 ] && [ "${best_len}" -lt "${known_length}" ]; then
    echo "solve_bucket_fresh_skip_reflected puzzle_id=${puzzle_id} reason=original_improved best_len=${best_len}"
    publish_puzzle_results "${puzzle_id}" "${known_length}" "${RESULT_TSV}" "${best_len}"
    continue
  fi

  sources_tsv="${LOG_DIR}/solve_bucket_fresh_sources_p${puzzle_id}_${FRESH_RUN_TAG}.tsv"
  write_reflected_sources "${puzzle_id}" "${RESULT_TSV}" "${sources_tsv}"
  if [ "$(wc -l < "${sources_tsv}")" -le 1 ]; then
    echo "solve_bucket_fresh_no_original_sources puzzle_id=${puzzle_id}"
    publish_puzzle_results "${puzzle_id}" "${known_length}" "${RESULT_TSV}" "${best_len}"
    continue
  fi

  tail -n +2 "${sources_tsv}" | while IFS=$'\t' read -r source_index state_hash source_solution; do
    if [ -z "${source_solution}" ]; then
      continue
    fi
    reflected_done="${PROGRESS_DIR}/p${puzzle_id}_reflected_${state_hash}.done"
    if [ -f "${reflected_done}" ]; then
      echo "solve_bucket_fresh_resume_skip_reflected puzzle_id=${puzzle_id} source_index=${source_index} state_hash=${state_hash}"
      continue
    fi
    run_puzzle_id=$((9500000 + puzzle_id * 1000 + source_index))
    write_reflected_puzzle "${source_solution}" "${run_puzzle_id}"
    reflected_tsv="${LOG_DIR}/solve_bucket_fresh_reflected_p${puzzle_id}_s${source_index}_${FRESH_RUN_TAG}.tsv"
    run_bucket_once "solve_bucket_fresh_reflected_s${source_index}" "${puzzle_id}" "${run_puzzle_id}" "${known_length}" "${reflected_tsv}" "${run_depth_limit}"
    append_bucket_results "reflected" "${puzzle_id}" "${run_puzzle_id}" "${known_length}" "${reflected_tsv}" "${RESULT_TSV}" "${source_index}" "${source_solution}" 1
    printf "done_at=%s\nstate_hash=%s\n" "$(date -Is)" "${state_hash}" > "${reflected_done}"
    best_len="$(best_aggregate_length "${puzzle_id}" "${RESULT_TSV}")"
    echo "solve_bucket_fresh_reflected_done puzzle_id=${puzzle_id} source_index=${source_index} state_hash=${state_hash} best_len=${best_len} known_length=${known_length}"
    if [ "${best_len}" -ge 0 ] && [ "${best_len}" -lt "${known_length}" ]; then
      echo "solve_bucket_fresh_stop_reflected puzzle_id=${puzzle_id} source_index=${source_index} best_len=${best_len}"
      break
    fi
  done
  best_len="$(best_aggregate_length "${puzzle_id}" "${RESULT_TSV}")"
  publish_puzzle_results "${puzzle_id}" "${known_length}" "${RESULT_TSV}" "${best_len}"
done

echo "solve_bucket_fresh_result_tsv=${RESULT_TSV}"
echo "finished_at=$(date -Is)"
