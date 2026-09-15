#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MGBFS
export BEAM_WEIGHT_DIR=/workspace BEAM_STREAM1_SYNTHETIC_STATES=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1
export BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
export BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1
export BEAM_STREAM1_TRANSFORMER_B_MICRO=${BEAM_STREAM1_TRANSFORMER_B_MICRO:-192}
export BEAM_STREAM1_TRANSFORMER_CONCURRENCY=${BEAM_STREAM1_TRANSFORMER_CONCURRENCY:-12}
export BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=100
tag=${H200_PROFILE_TAG:-profile_s1}
export BEAM_STREAM_BENCH_REPORT=/workspace/results/${tag}.md
nsys=/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys
"$nsys" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --cuda-graph-trace=node --force-overwrite=true -o /workspace/results/${tag} /workspace/build/stream_benchmark 1000 > /workspace/results/${tag}.log 2>&1
"$nsys" stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv /workspace/results/${tag}.nsys-rep > /workspace/results/${tag}_stats.csv
