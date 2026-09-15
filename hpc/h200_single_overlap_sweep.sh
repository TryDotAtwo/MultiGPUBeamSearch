#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MGBFS
export BEAM_WEIGHT_DIR=/workspace BEAM_STREAM1_SYNTHETIC_STATES=1
export BEAM_GENERATOR_PATH=/workspace/cube4_actions.json
export BEAM_PUZZLE_INFO_PATH=/workspace/puzzle_info.json BEAM_TEST_CSV_PATH=/workspace/test.csv
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1
export BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
export BEAM_STREAM1_TRANSFORMER_MICRO=192 BEAM_STREAM1_CONCURRENCY=12
export BEAM_PIPELINE_SMOKE_FRONTIER=1000000
export BEAM_RING_GRAPH_EXECS_PER_LANE=8
for outer in 192 768 3072; do
  for rings in 4 32; do
    for mode in stream12 stream123; do
      tag=o${outer}_r${rings}_${mode}
      export BEAM_B_MICRO=$outer BEAM_PIPELINE_SMOKE_RINGS=$rings
      export BEAM_STREAM3_RING_SLOTS=12 BEAM_PIPELINE_BENCH_MODE=$mode
      /workspace/build/stream_pipeline_benchmark 1000 > /workspace/results/overlap_${tag}.log 2>&1
    done
  done
done
