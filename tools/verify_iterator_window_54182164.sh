#!/usr/bin/env bash
# Remote-only, owned instance 54182164; bounded by external/server deadline.
set -u
cd /workspace/network_regression_54182164 || exit 1
export CUTLASS_DIR=/workspace/cutlass
/venv/main/bin/python -m pytest -q > /workspace/full_iterator_binding_final_54182164.log 2>&1
code=$?; echo full_final_exit=$code; tail -8 /workspace/full_iterator_binding_final_54182164.log
if [ "$code" -ne 0 ]; then exit "$code"; fi
export BEAM_B_MICRO=32 BEAM_STREAM1_TRANSFORMER_MICRO=13 BEAM_STREAM1_CONCURRENCY=2
export BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
/venv/main/bin/python tools/cube4_production_preflight.py --runner build/production_runner \
  --probe build/cuda_visible_device_probe --reference-dir /workspace/cube4_paired_reference_20261001 \
  --source-root . --world-size 2 --output-root /workspace/preflight_iterator_54182164 \
  --timeout 180 > /workspace/preflight_iterator_54182164.log 2>&1
code=$?; echo native_preflight_exit=$code; tail -12 /workspace/preflight_iterator_54182164.log
if [ "$code" -ne 2 ] || [ ! -f /workspace/preflight_iterator_54182164/summary.json ]; then exit 1; fi
build/stream1_gemm_iterator_probe > /workspace/iterator_geometry_54182164.json || exit 1
for gpu in 0 1; do
  BEAM_TEST_GEMM_SCORE_OUTPUT=/workspace/preflight_iterator_54182164/device${gpu}/output \
  BEAM_TEST_GEMM_REFERENCE_DIR=/workspace/cube4_paired_reference_20261001 \
  BEAM_TEST_GEMM_ITERATOR_GEOMETRY=/workspace/iterator_geometry_54182164.json \
  /venv/main/bin/python -m pytest tests/test_gemm_iterator_collector_protocol.py -q \
    > /workspace/collector_fresh_gpu${gpu}_54182164.log 2>&1
  code=$?; echo fresh_protocol_gpu${gpu}_exit=$code; tail -5 /workspace/collector_fresh_gpu${gpu}_54182164.log
  if [ "$code" -ne 0 ]; then exit "$code"; fi
  LOCAL_RANK=$gpu RANK=$gpu WORLD_SIZE=1 BEAM_WEIGHT_DIR=/workspace/cube4_paired_reference_20261001/weights_fp16 \
  BEAM_STREAM1_EXECUTOR=native_cuda_graph timeout 180 compute-sanitizer --tool memcheck --leak-check full \
    --error-exitcode 77 build/production_runner --validate-stream1-scores \
    /workspace/cube4_paired_reference_20261001 /workspace/memcheck_iterator_54182164_gpu${gpu} \
    > /workspace/memcheck_iterator_54182164_gpu${gpu}.log 2>&1
  code=$?; echo memcheck_gpu${gpu}_exit=$code; tail -8 /workspace/memcheck_iterator_54182164_gpu${gpu}.log
  if [ "$code" -ne 0 ]; then exit "$code"; fi
done
sha256sum build/production_runner build/stream1_gemm_iterator_probe
