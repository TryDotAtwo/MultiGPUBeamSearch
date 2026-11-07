#!/bin/bash
#SBATCH --job-name=beam1p4b-8xa100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail
ulimit -c 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "This script is intended to be submitted with sbatch."
  echo "Example: PARTITION=kaf12 PUZZLE_ID=992 sbatch -p kaf12 ${BASH_SOURCE[0]}"
  exit 2
fi

export REPO_DIR="${REPO_DIR}"
export BEAM_COMMON_SH="${REPO_DIR}/hpc/mephi_8xa100_common.sh"
export JOB_DIR="${JOB_DIR:-${REPO_DIR}}"
export BUILD_DIR="${BUILD_DIR:-${JOB_DIR}/build-a100-${SLURM_JOB_ID}}"
export HISTORY_DIR="${HISTORY_DIR:-${JOB_DIR}/history-${SLURM_JOB_ID}}"

if [ -z "${CUTLASS_DIR:-}" ]; then
  if [ -d "${REPO_DIR}/external/cutlass/include" ]; then
    export CUTLASS_DIR="${REPO_DIR}/external/cutlass"
  else
    echo "missing_cutlass_dir: set CUTLASS_DIR=/path/to/cutlass or install it at ${REPO_DIR}/external/cutlass"
    exit 2
  fi
fi

if [ -z "${NINJA_VENV_DIR:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ]; then
    export NINJA_VENV_DIR="${VIRTUAL_ENV}"
  elif [ -d "${REPO_DIR}/.venv" ]; then
    export NINJA_VENV_DIR="${REPO_DIR}/.venv"
  else
    echo "missing_ninja_venv: set NINJA_VENV_DIR=/path/to/venv with python, torch, ninja, and nvidia-nccl"
    exit 2
  fi
fi

export PUZZLE_ID="${PUZZLE_ID:-992}"
export DEPTH_LIMIT="${DEPTH_LIMIT:-80}"
export BEAM_WIDTH="${BEAM_WIDTH:-1400000000}"
export BEAM_WEIGHT_DIR="${BEAM_WEIGHT_DIR:-${REPO_DIR}/stream1_weights_artgor_bf16}"

export TORCHRUN_NNODES="${TORCHRUN_NNODES:-1}"
export TORCHRUN_NPROC_PER_NODE="${TORCHRUN_NPROC_PER_NODE:-8}"
WORLD_SIZE_EFFECTIVE=$((TORCHRUN_NNODES * TORCHRUN_NPROC_PER_NODE))

export BEAM_B_MICRO="${BEAM_B_MICRO:-8192}"
export BEAM_STREAM1_CONCURRENCY="${BEAM_STREAM1_CONCURRENCY:-8}"
export BEAM_STREAM3_RING_SLOTS="${BEAM_STREAM3_RING_SLOTS:-8}"

# 1.4B global beam on 8 GPUs is 175M/GPU. 64 shards keeps per-shard size
# close to the tested 700M/32-shard 40GB configuration while using 80GB cards.
export SHARD_COUNT="${SHARD_COUNT:-64}"
export SHARD_CAPACITY_SCALE_PPM="${SHARD_CAPACITY_SCALE_PPM:-1150000}"

export STREAM4_BATCH_CANDIDATES="${STREAM4_BATCH_CANDIDATES:-262144}"
export STREAM4_TRIGGER_CANDIDATES="${STREAM4_TRIGGER_CANDIDATES:-524288}"
export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES="${BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES:-88064}"
export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM="${BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM:-$((WORLD_SIZE_EFFECTIVE * 1000000))}"

export BEAM_SOLVED_NEIGHBORHOOD_RADIUS="${BEAM_SOLVED_NEIGHBORHOOD_RADIUS:-4}"
export BEAM_GPU_HEADROOM_BYTES="${BEAM_GPU_HEADROOM_BYTES:-268435456}"
export BEAM_HISTORY_RAM_BYTES="${BEAM_HISTORY_RAM_BYTES:-257698037760}"
export BEAM_HISTORY_DISK_BYTES="${BEAM_HISTORY_DISK_BYTES:-2199023255552}"

echo "portable_8xa100_80gb=1"
echo "repo_dir=${REPO_DIR}"
echo "job_dir=${JOB_DIR}"
echo "build_dir=${BUILD_DIR}"
echo "history_dir=${HISTORY_DIR}"
echo "cutlass_dir=${CUTLASS_DIR}"
echo "ninja_venv_dir=${NINJA_VENV_DIR}"
echo "puzzle_id=${PUZZLE_ID}"
echo "beam_width=${BEAM_WIDTH}"
echo "shard_count=${SHARD_COUNT}"

exec "${REPO_DIR}/hpc/solve_then_reflect.sh"
