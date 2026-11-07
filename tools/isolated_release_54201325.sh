#!/usr/bin/env bash
set -eu
test "$(date +%s)" -lt 1791153617
cd /workspace
tar -xzf vast_paired_fixture_20261002.tar.gz -C /workspace
cd /workspace/release_audit_54201325
mkdir -p evidence/recovered_campaign_inputs_20261001 evidence/isolated_release_54201325
cp /workspace/test.csv /workspace/puzzle_info.json evidence/recovered_campaign_inputs_20261001/
ln -s /workspace/cube4_paired_reference_20261001 evidence/cube4_paired_reference_20261001
cp /workspace/dual_base_profile_54186691.json evidence/dual_base_profile.json
/venv/main/bin/python -m pip install --no-deps --target /workspace/nccl225_54201325 nvidia-nccl-cu12==2.25.1 > /workspace/release_nccl_54201325.log 2>&1
cmake --build build-release --target stream_benchmark -j 8 > /workspace/release_benchmark_build_54201325.log 2>&1
export BEAM_WEIGHT_DIR=/workspace/cube4_paired_reference_20261001/weights_fp16
export BEAM_GENERATOR_PATH=/workspace/release_audit_54201325/evidence/recovered_campaign_inputs_20261001/puzzle_info.json
export BEAM_PUZZLE_INFO_JSON="$BEAM_GENERATOR_PATH"
export BEAM_TEST_CSV=/workspace/release_audit_54201325/evidence/recovered_campaign_inputs_20261001/test.csv
export BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_SPLIT_QKV=0
export BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1
export BEAM_STREAM1_TRANSFORMER_BENCH_ITERS=20
for rank in 0 1; do
  for micro in 128 256 512; do
    test "$(date +%s)" -lt 1791153617
    case_dir=/workspace/release_audit_54201325/evidence/isolated_release_54201325/rank${rank}_micro${micro}_lanes4
    mkdir "$case_dir"
    (cd "$case_dir"; CUDA_VISIBLE_DEVICES="$rank" BEAM_STREAM1_TRANSFORMER_B_MICRO="$micro" BEAM_STREAM1_TRANSFORMER_CONCURRENCY=4 \
      timeout 90 /workspace/release_audit_54201325/build-release/stream_benchmark 1000 > raw.log 2>&1)
    grep 'stream1_transformer_micro ' "$case_dir/raw.log"
  done
done
sha256sum build-release/production_runner build-release/stream_benchmark
