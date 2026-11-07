#!/usr/bin/env bash
# Remote-only bounded scorer/metadata verification,not production admission.
set -u
cd /workspace/network_regression_54186691 || exit 1
export CUTLASS_DIR=/workspace/cutlass
export BEAM_B_MICRO=32 BEAM_STREAM1_TRANSFORMER_MICRO=13 BEAM_STREAM1_CONCURRENCY=2
export BEAM_STREAM1_TRANSFORMER_FUSED_INPUT_LAYERNORM=1 BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY=1
export BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1
/venv/main/bin/python tools/cube4_production_preflight.py --runner build/production_runner \
  --probe build/cuda_visible_device_probe --reference-dir /workspace/cube4_paired_reference_20261001 \
  --source-root . --world-size 2 --output-root /workspace/preflight_ab_canonical_54186691 \
  --timeout 180 > /workspace/preflight_ab_canonical_54186691.log 2>&1
code=$?; echo canonical_preflight_exit=$code; tail -7 /workspace/preflight_ab_canonical_54186691.log
if [ "$code" -ne 2 ] || [ ! -f /workspace/preflight_ab_canonical_54186691/summary.json ]; then exit 1; fi
grep -E 'status|production_admitted|graph_gemm_mainloop_iterators_checked' /workspace/preflight_ab_canonical_54186691/summary.json
for gpu in 0 1; do
  LOCAL_RANK=$gpu RANK=$gpu WORLD_SIZE=1 BEAM_WEIGHT_DIR=/workspace/cube4_paired_reference_20261001/weights_fp16 \
  BEAM_STREAM1_EXECUTOR=native_cuda_graph timeout 180 compute-sanitizer --tool memcheck --leak-check full \
    --error-exitcode 77 build/production_runner --validate-stream1-scores \
    /workspace/cube4_paired_reference_20261001 /workspace/memcheck_ab_canonical_54186691_gpu${gpu} \
    > /workspace/memcheck_ab_canonical_54186691_gpu${gpu}.log 2>&1
  code=$?; echo canonical_memcheck_gpu${gpu}_exit=$code; tail -5 /workspace/memcheck_ab_canonical_54186691_gpu${gpu}.log
  if [ "$code" -ne 0 ]; then exit "$code"; fi
done
nvcc -std=c++17 -arch=sm_86 --expt-relaxed-constexpr -DBEAM_HAS_CUTLASS=1 \
  -Xcompiler=-fsanitize=undefined,-fno-sanitize-recover=undefined \
  -I. -Isrc -I/workspace/cutlass/include tests/graph_gemm_mainloop_native_probe.cu \
  -o /workspace/graph_gemm_mainloop_native_ubsan > /workspace/ab_ubsan_build.log 2>&1
code=$?; echo ubsan_build_exit=$code
if [ "$code" -ne 0 ]; then tail -10 /workspace/ab_ubsan_build.log; exit "$code"; fi
for gpu in 0 1; do
  /workspace/graph_gemm_mainloop_native_ubsan $gpu > /workspace/ab_ubsan_gpu${gpu}.log 2>&1
  code=$?; echo ubsan_gpu${gpu}_exit=$code; cat /workspace/ab_ubsan_gpu${gpu}.log
  if [ "$code" -ne 0 ]; then exit "$code"; fi
done
sha256sum build/production_runner build/stream1_gemm_iterator_probe
