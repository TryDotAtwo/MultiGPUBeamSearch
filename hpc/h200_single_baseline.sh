#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MGBFS
export BEAM_WEIGHT_DIR=/workspace
export BEAM_STREAM1_SYNTHETIC_STATES=1
export BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1
export BEAM_STREAM1_TRANSFORMER_HOPPER=off
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1
export BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
mkdir -p /workspace/results
for micro in 128 192 256 384 512 768; do
  for lanes in 4 8 12; do
    export BEAM_STREAM1_TRANSFORMER_B_MICRO=$micro
    export BEAM_STREAM1_TRANSFORMER_CONCURRENCY=$lanes
    export BEAM_STREAM_BENCH_REPORT=/workspace/results/s1_m${micro}_c${lanes}.md
    /workspace/build/stream_benchmark 1000 > /workspace/results/s1_m${micro}_c${lanes}.log 2>&1
  done
done
