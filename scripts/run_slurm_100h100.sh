#!/usr/bin/env bash
#SBATCH --job-name=cayley-beam-100h100
#SBATCH --nodes=13
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

export WORLD_SIZE="${WORLD_SIZE:-100}"
export GLOBAL_BEAM_WIDTH="${GLOBAL_BEAM_WIDTH:-1073741824}"
export B_MICRO="${B_MICRO:-131072}"
export SCORE_RING_DEPTH="${SCORE_RING_DEPTH:-64}"
export NET_RING_DEPTH="${NET_RING_DEPTH:-3}"
export BUCKET_CAP_PER_PEER="${BUCKET_CAP_PER_PEER:-65536}"
export USE_CUDA_GRAPHS="${USE_CUDA_GRAPHS:-1}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-dummy}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT
export INIT_METHOD="env://"

python scripts/h100_sizing.py --world-size "${WORLD_SIZE}" --global-beam-width "${GLOBAL_BEAM_WIDTH}" --bucket-cap-per-peer "${BUCKET_CAP_PER_PEER}"

srun --ntasks="${WORLD_SIZE}" --gpus-per-task=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-12}" \
  bash -lc 'export RANK=${SLURM_PROCID}; export LOCAL_RANK=${SLURM_LOCALID}; export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}; python beam_engine.py'
