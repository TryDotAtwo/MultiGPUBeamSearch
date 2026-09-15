#!/usr/bin/env bash
set -euo pipefail
export H200_FP8_ARTIFACT=/workspace/hopper_offline_qkv_ffn_e4m3_v1
export H200_BUILD_DIR=/workspace/build_ffn
export H200_REPEATS=3 H200_BENCH_ITERS=150 H200_SKIP_QUALITY=1
for spec in '384 4' '384 8' '384 12' '768 4' '768 8' '1536 2' '1536 4' '3072 2'; do
  read -r micro lanes <<< "$spec"
  export BEAM_STREAM1_TRANSFORMER_B_MICRO=$micro BEAM_STREAM1_TRANSFORMER_CONCURRENCY=$lanes
  export H200_RESULTS_DIR=/workspace/fp8_ffn_sweep/m${micro}_c${lanes}
  bash /workspace/MGBFS/hpc/h200_native_fp8_pair.sh
done
