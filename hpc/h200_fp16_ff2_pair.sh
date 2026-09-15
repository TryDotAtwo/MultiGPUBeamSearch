#!/usr/bin/env bash
set -euo pipefail
cd /workspace/hopper_runtime
result_dir=${H200_RESULTS_DIR:-/workspace/fp16_ff2_results}
build_dir=${H200_BUILD_DIR:-/workspace/build_ffn}
mkdir -p "$result_dir"
unset BEAM_HOPPER_NATIVE_FP8
export BEAM_WEIGHT_DIR=/workspace/stream1_transformer_weights_fp16
export BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=1 BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
export BEAM_STREAM1_TRANSFORMER_HOPPER_QKV=fp16_tma BEAM_STREAM1_TRANSFORMER_HOPPER_FF1=fp16_tma
export BEAM_STREAM1_TRANSFORMER_FF2_POLICY=m128n128 BEAM_STREAM1_TRANSFORMER_COMPACT57=1
export BEAM_STREAM1_TRANSFORMER_B_MICRO=384 BEAM_STREAM1_TRANSFORMER_CONCURRENCY=8
export BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN=0 BEAM_STREAM1_TRANSFORMER_HOPPER_FF1_EPILOGUE=128x64
export LD_LIBRARY_PATH=/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:/usr/local/cuda/lib64
run_one(){
  local mode=$1 tag=$2 puzzle=$3
  export BEAM_STREAM1_TRANSFORMER_HOPPER_FF2=$mode
  BEAM_STREAM_BENCH_REPORT="$result_dir/$tag.md" \
  BEAM_STREAM1_TRANSFORMER_SCORE_DUMP="$result_dir/$tag.bin" \
    "$build_dir/stream_benchmark" "$puzzle" > "$result_dir/$tag.log" 2>&1
}
# Numerical gate before timing. Fail closed; do not silently weaken to tolerance.
unset BEAM_STREAM1_SYNTHETIC_STATES
export BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=3
for puzzle in 1 2 3 4 5 6 7 8 9 10 1000; do
  for mode in off fp16_tma; do run_one "$mode" "quality_p${puzzle}_${mode}" "$puzzle"; done
  cmp "$result_dir/quality_p${puzzle}_off.bin" "$result_dir/quality_p${puzzle}_fp16_tma.bin"
done
export BEAM_STREAM1_SYNTHETIC_STATES=1 BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=300
for rep in 1 2 3 4 5 6 7; do
  order='off fp16_tma'; if ((rep%2==0)); then order='fp16_tma off'; fi
  for mode in $order; do run_one "$mode" "pair_${mode}_${rep}" 1000; done
  cmp "$result_dir/pair_off_${rep}.bin" "$result_dir/pair_fp16_tma_${rep}.bin"
done
