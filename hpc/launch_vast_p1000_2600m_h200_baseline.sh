#!/bin/bash
set -euo pipefail

export SLURM_SUBMIT_DIR=/workspace/MGBFS/hpc
export JOB_DIR=/workspace/beamjob_prod_baseline11m
export REPO_DIR=/workspace/MGBFS
export BUILD_DIR=/workspace/beam-build-hopper-fusion
export CUTLASS_DIR=/workspace/cutlass
export NINJA_VENV_DIR=/workspace/beam-tools
export NCCL_INCLUDE_DIR=/venv/main/lib/python3.12/site-packages/nvidia/nccl/include
export NCCL_LIBRARY=/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64
export TORCHRUN_NNODES=1
export TORCHRUN_NPROC_PER_NODE=8
export TORCHRUN_NODE_RANK=0
export TORCHRUN_RDZV_ENDPOINT=127.0.0.1:29751

export PUZZLE_ID=1000
export SYNTHETIC_PUZZLE_ID=9001000
export DEPTH_LIMIT=50
export BEAM_WIDTH=2600000000
export BEAM_WEIGHT_DIR=/workspace/stream1_transformer_weights_fp16
export BEAM_GENERATOR_PATH=/workspace/MGBFS/data/puzzle_info.json
export BEAM_PUZZLE_INFO_JSON=/workspace/MGBFS/data/puzzle_info.json
export BEAM_TEST_CSV=/workspace/MGBFS/data/test.csv

# Measured H200 FP16 baseline: 11.20M candidates/s at micro=192, 12 lanes.
export BEAM_STREAM1_TRANSFORMER_HOPPER=off
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1
export BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
export MEGAMINX_STREAM1_BACKEND=native_windowed_graph
export BEAM_STREAM1_EXECUTOR=native_cuda_graph
export BEAM_RING_GRAPH_EXECS_PER_LANE=32
export BEAM_B_MICRO=192
export BEAM_STREAM1_CONCURRENCY=12

export BEAM_STREAM3_RING_SLOTS=12
export SHARD_COUNT=8
export STREAM4_BATCH_ALIGNMENT=1024
export SHARD_CAPACITY_SCALE_PPM=1250000
export STREAM4_BATCH_CANDIDATES=524288
export STREAM4_TRIGGER_CANDIDATES=1048576
export BEAM_STREAM4_ACTIVE_SORT_SLOTS=4
export BEAM_SHARD_BUFFER_COUNT=2
export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=262144
export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=8000000
export BEAM_GPU_HEADROOM_BYTES=1073741824
export BEAM_SOLVED_NEIGHBORHOOD_RADIUS=4
export BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES=3000000

# 512 GiB RAM + 2 TiB sparse disk history, divided across eight ranks.
export BEAM_HISTORY_MODE=static_hybrid
export BEAM_HISTORY_RAM_BYTES=549755813888
export BEAM_HISTORY_DISK_BYTES=2199023255552
export BEAM_ENABLE_DEBUG=1
export BEAM_DEPTH_LOG_EVERY=1
export NCCL_DEBUG=WARN

mkdir -p "${JOB_DIR}/logs" "${JOB_DIR}/history" /workspace/MGBFS/data /workspace/beam-tools/bin
ln -sfn /venv/main/bin/python /workspace/beam-tools/bin/python
ln -sfn /usr/bin/ninja /workspace/beam-tools/bin/ninja
cp /workspace/cube4_kaggle/puzzle_info.json /workspace/MGBFS/data/puzzle_info.json
cp /workspace/cube4_kaggle/test.csv /workspace/MGBFS/data/test.csv

# The exact sm_90a runner is built separately. Avoid the cluster script's
# hard-coded sm_80 configure and its successful-run build cleanup.
bash <(sed \
  -e '/beam_configure_build production_runner$/d' \
  -e '/beam_safe_clean_child "${BUILD_DIR}" "build"/d' \
  /workspace/MGBFS/hpc/solve_then_reflect.sh)
