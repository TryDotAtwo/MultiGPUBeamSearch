#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PARTITION="${PARTITION:-}"
if [ -z "${PARTITION}" ]; then
  echo "missing_PARTITION: set PARTITION=YOUR_GPU_PARTITION"
  exit 2
fi

RUN_ROOT="${RUN_ROOT:-${PWD}/megaminx_transformer_bucket_run}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}}"
mkdir -p "${RUN_ROOT}" "${RUN_DIR}"
RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"
JOB_DIR="${JOB_DIR:-${RUN_DIR}}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/data}"
WEIGHT_DIR="${WEIGHT_DIR:-${REPO_DIR}/weights/megaminx_vlad_transformer_fp16}"
SOLUTIONS_CSV="${SOLUTIONS_CSV:-${RUN_DIR}/target_solutions.csv}"
KNOWN_LENGTHS="${KNOWN_LENGTHS:-75}"
ARRAY_SPEC="${ARRAY_SPEC:-0-0%1}"

mkdir -p "${RUN_DIR}/logs"

for required in \
  "${REPO_DIR}/hpc/mephi_8xa100_common.sh" \
  "${REPO_DIR}/hpc/ihes_cube_model/prepare_ihes_prebuilt_runner.sh" \
  "${REPO_DIR}/hpc/ihes_cube_model/ihes_solve_bucket_from_scratch.sh" \
  "${DATA_DIR}/puzzle_info.json" \
  "${DATA_DIR}/test.csv" \
  "${WEIGHT_DIR}/manifest.json" \
  "${SOLUTIONS_CSV}"; do
  if [ ! -f "${required}" ]; then
    echo "missing_required_file=${required}"
    exit 2
  fi
done

cp "${REPO_DIR}/hpc/ihes_cube_model/prepare_ihes_prebuilt_runner.sh" "${RUN_DIR}/"
cp "${REPO_DIR}/hpc/ihes_cube_model/ihes_solve_bucket_from_scratch.sh" "${RUN_DIR}/"
sed -i 's/\r$//' \
  "${RUN_DIR}/prepare_ihes_prebuilt_runner.sh" \
  "${RUN_DIR}/ihes_solve_bucket_from_scratch.sh"
bash -n \
  "${RUN_DIR}/prepare_ihes_prebuilt_runner.sh" \
  "${RUN_DIR}/ihes_solve_bucket_from_scratch.sh"
chmod +x \
  "${RUN_DIR}/prepare_ihes_prebuilt_runner.sh" \
  "${RUN_DIR}/ihes_solve_bucket_from_scratch.sh"

export BASE_DIR="${RUN_ROOT}"
export REPO_DIR
export RUN_DIR
export JOB_DIR
export DATA_DIR
export WEIGHT_DIR
export SOLUTIONS_CSV
export KNOWN_LENGTHS
export PUZZLE_OFFSET="${PUZZLE_OFFSET:-0}"
export PUZZLE_LIMIT="${PUZZLE_LIMIT:-1}"
export FRESH_RUN_TAG="${FRESH_RUN_TAG:-megaminx_len${KNOWN_LENGTHS}_bw${BEAM_WIDTH:-30000000}}"

export TORCHRUN_NNODES="${TORCHRUN_NNODES:-1}"
export TORCHRUN_NPROC_PER_NODE="${TORCHRUN_NPROC_PER_NODE:-8}"
WORLD_SIZE_EFFECTIVE=$((TORCHRUN_NNODES * TORCHRUN_NPROC_PER_NODE))

export MEGAMINX_STREAM1_BACKEND="${MEGAMINX_STREAM1_BACKEND:-native_cuda_graph}"
export BEAM_STREAM1_TRANSFORMER_BLOCK51="${BEAM_STREAM1_TRANSFORMER_BLOCK51:-1}"
export BEAM_STREAM1_TRANSFORMER_MICRO="${BEAM_STREAM1_TRANSFORMER_MICRO:-512}"
export BEAM_STREAM1_TRANSFORMER_OUTPUT_DIM="${BEAM_STREAM1_TRANSFORMER_OUTPUT_DIM:-24}"

export BEAM_WIDTH="${BEAM_WIDTH:-30000000}"
export SHARD_COUNT="${SHARD_COUNT:-4}"
export BEAM_B_MICRO="${BEAM_B_MICRO:-512}"
export BEAM_STREAM1_CONCURRENCY="${BEAM_STREAM1_CONCURRENCY:-2}"
export BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-8}"
unset BEAM_STREAM3_BATCH_CANDIDATES

export STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES:-65536}"
export STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES:-262144}"
export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-65536}"
export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-$((WORLD_SIZE_EFFECTIVE * 1000000))}"
export SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1000000}"

export BEAM_SOLVED_NEIGHBORHOOD_RADIUS="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS:-5}"
export BEAM_SOLVE_BUCKET_EXTRA_DEPTHS="${BEAM_SOLVE_BUCKET_EXTRA_DEPTHS:-2}"
export BEAM_SOLVED_RESULT_CAPACITY="${BEAM_SOLVED_RESULT_CAPACITY:-1048576}"
export BEAM_SOLVE_BUCKET_GATHER_SCRATCH_BYTES="${BEAM_SOLVE_BUCKET_GATHER_SCRATCH_BYTES:-2147483648}"
export BEAM_GPU_HEADROOM_BYTES="${BEAM_GPU_HEADROOM_BYTES:-536870912}"

unset BUILD_DIR
unset HISTORY_DIR
unset PUBLISH_RESULTS_REPO_URL
unset PUBLISH_RESULTS_DIR

PREBUILD_DIR="${PREBUILD_DIR:-${RUN_DIR}/prebuilt-a100-megaminx-transformer}"
export BUILD_DIR="${PREBUILD_DIR}"
export BEAM_PREBUILT_RUNNER="${PREBUILD_DIR}/production_runner"
PREBUILD_JOB="$(
  sbatch -p "${PARTITION}" \
    --chdir="${RUN_ROOT}" \
    --parsable \
    --export=ALL \
    "${RUN_DIR}/prepare_ihes_prebuilt_runner.sh"
)"
echo "PREBUILD_JOB=${PREBUILD_JOB}"

unset BUILD_DIR
unset HISTORY_DIR
ARRAY_JOB="$(
  sbatch -p "${PARTITION}" \
    --chdir="${RUN_ROOT}" \
    --dependency=afterok:${PREBUILD_JOB} \
    --array="${ARRAY_SPEC}" \
    --export=ALL \
    "${RUN_DIR}/ihes_solve_bucket_from_scratch.sh"
)"
echo "ARRAY_JOB=${ARRAY_JOB}"
echo "RUN_DIR=${RUN_DIR}"
echo "RESULT_TSV=${RUN_DIR}/logs/solve_bucket_fresh_${FRESH_RUN_TAG}.tsv"
echo "TAIL_EXAMPLE=tail -f ${RUN_ROOT}/slurm-<array_job>_0.out"
