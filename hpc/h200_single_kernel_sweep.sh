#!/usr/bin/env bash
set -euo pipefail
cd /workspace/MGBFS
export BEAM_WEIGHT_DIR=/workspace BEAM_STREAM1_SYNTHETIC_STATES=1
export BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1
export BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=100
export BEAM_STREAM1_TRANSFORMER_B_MICRO=192 BEAM_STREAM1_TRANSFORMER_CONCURRENCY=12
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1
export BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
run() {
  local tag=$1; shift
  for rep in 1 2 3; do
    env "$@" BEAM_STREAM_BENCH_REPORT=/workspace/results/k_${tag}_${rep}.md \
      BEAM_STREAM1_TRANSFORMER_SCORE_DUMP=/workspace/results/k_${tag}_${rep}.bin \
      /workspace/build/stream_benchmark 1000 > /workspace/results/k_${tag}_${rep}.log 2>&1
  done
}
run baseline
for family in QKV FF1 ATTN_OUT FF2; do
  for tile in m64n128 m128n128; do
    if [[ $tile == m64n128 && ( $family == ATTN_OUT || $family == FF2 ) ]]; then continue; fi
    run ${family}_${tile} BEAM_STREAM1_TRANSFORMER_${family}_POLICY=$tile
  done
done
run fused BEAM_STREAM1_TRANSFORMER_ATTN_OUT_EPILOGUE=fused BEAM_STREAM1_TRANSFORMER_FF2_EPILOGUE=fused
run ln_persistent2 BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY=persistent BEAM_STREAM1_TRANSFORMER_LAYERNORM_PERSISTENT_BLOCKS_PER_SM=2
run ln_persistent4 BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY=persistent BEAM_STREAM1_TRANSFORMER_LAYERNORM_PERSISTENT_BLOCKS_PER_SM=4
run tma_qkv BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=fp16_tma
run tma_ff1 BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=fp16_tma
