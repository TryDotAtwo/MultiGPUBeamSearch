#!/usr/bin/env bash
set -euo pipefail
repo=${REPO_ROOT:-/workspace/repo}
cutlass=${CUTLASS_ROOT:-/workspace/cutlass}
out=${OUTPUT_ROOT:-/workspace/out}
mkdir -p "$out"
nvcc -std=c++20 -O3 -w --threads=4 --split-compile=4 \
  --split-compile-extended=4 --expt-relaxed-constexpr \
  -DSTREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON=1 \
  --generate-code=arch=compute_120a,code=sm_120a \
  "-I${repo}/cuda" "-I${cutlass}/include" \
  "-I${cutlass}/tools/util/include" \
  "${repo}/tests/stream1_transformer_sm120_nvfp4_cutlass_ff1_ff2_fused_numeric_tests.cu" \
  -o "${out}/ff1_ff2_fused_numeric"
sha256sum "${out}/ff1_ff2_fused_numeric"
