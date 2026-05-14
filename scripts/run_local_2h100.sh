#!/usr/bin/env bash
set -euo pipefail

export WORLD_SIZE="${WORLD_SIZE:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export GLOBAL_BEAM_WIDTH="${GLOBAL_BEAM_WIDTH:-65536}"
export B_MICRO="${B_MICRO:-32768}"
export SCORE_RING_DEPTH="${SCORE_RING_DEPTH:-8}"
export NET_RING_DEPTH="${NET_RING_DEPTH:-3}"
export BUCKET_CAP_PER_PEER="${BUCKET_CAP_PER_PEER:-65536}"
export K_EXPAND_TILE="${K_EXPAND_TILE:-32768}"
export INFERENCE_PARALLELISM="${INFERENCE_PARALLELISM:-1}"
export USE_CUDA_GRAPHS="${USE_CUDA_GRAPHS:-1}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-fullbeamnice_static}"
export BETA="${BETA:-1.20}"
export HASH_LOAD_FACTOR="${HASH_LOAD_FACTOR:-0.45}"
export PROBE_LIMIT="${PROBE_LIMIT:-256}"
export MAX_DEPTH="${MAX_DEPTH:-100}"
export LOG_EVERY="${LOG_EVERY:-25}"
export TEST_START="${TEST_START:-0}"
export TEST_COUNT="${TEST_COUNT:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-PHB}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export RUN_SCRIPT="${RUN_SCRIPT:-scripts/solve_testcsv_2gpu.py}"

python scripts/h100_sizing.py \
  --world-size "${WORLD_SIZE}" \
  --global-beam-width "${GLOBAL_BEAM_WIDTH}" \
  --bucket-cap-per-peer "${BUCKET_CAP_PER_PEER}" \
  --b-micro "${B_MICRO}" \
  --k-expand-tile "${K_EXPAND_TILE}" \
  --score-ring-depth "${SCORE_RING_DEPTH}" \
  --net-ring-depth "${NET_RING_DEPTH}" \
  --max-depth "${MAX_DEPTH}" \
  --beta "${BETA}" \
  --hash-load-factor "${HASH_LOAD_FACTOR}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE}" \
  "${RUN_SCRIPT}"
