#!/usr/bin/env bash
set -euo pipefail
repo=${REPO_ROOT:-/workspace/repo}
cutlass=${CUTLASS_ROOT:-/workspace/cutlass}
out=${OUTPUT_ROOT:-/workspace/out}
opt=${OPT_LEVEL:--O0}
binary=${OUTPUT_NAME:-ff1_ff2_fused_numeric}
hidden=${FUSED_TEST_HIDDEN:-256}
threads=${FUSED_TEST_THREADS:-384}
direct_pack=${FUSED_DIRECT_PACK:-0}
no_relocation=${FUSED_NO_RELOCATION:-0}
accumulate_direct=${FUSED_ACCUMULATE_DIRECT:-0}
shared_carry=${FUSED_SHARED_CARRY:-0}
proxy_fence=${FUSED_PROXY_FENCE:-1}
handoff_trace=${FUSED_HANDOFF_TRACE:-0}
invalidate_barriers=${FUSED_INVALIDATE_BARRIERS:-1}
noinline_ff1=${FUSED_NOINLINE_FF1:-0}
wait_dependent_grids=${FUSED_WAIT_DEPENDENT_GRIDS:-1}
role_separated_4cta=${FUSED_ROLE_SEPARATED_4CTA:-0}
role_precompute=${FUSED_ROLE_PRECOMPUTE_FF1:-0}
role_dual_ff1_math_wg=${FUSED_ROLE_DUAL_FF1_MATH_WG:-0}
native_epilogue=${FUSED_NATIVE_EPILOGUE:-0}
cccl_include=${CUDA_CCCL_INCLUDE:-}
if [[ -z "$cccl_include" ]]; then
  for candidate in \
    /usr/local/cuda/include/cccl \
    /usr/local/lib/python*/site-packages/nvidia/cu*/include/cccl \
    /opt/pyvenv/lib/python*/site-packages/nvidia/cu*/include/cccl; do
    if [[ -f "$candidate/cuda/std/utility" ]]; then
      cccl_include=$candidate
      break
    fi
  done
fi
cccl_flags=()
if [[ -n "$cccl_include" ]]; then
  cccl_flags+=("-I${cccl_include}")
  echo "cuda_cccl_include=${cccl_include}"
fi
materialize_flags=(-DSTREAM1_CUTLASS_DIAG_DIRECT_FF1_MANUAL_MATERIALIZE=1)
if [[ "$native_epilogue" == "1" ]]; then
  materialize_flags=(-DSTREAM1_FUSED_NATIVE_EPILOGUE=1
    -DSTREAM1_CUTLASS_DIAG_DIRECT_FF1_EPILOGUE=1
    -DSTREAM1_HANDOFF_SOURCE_OWNED_STORE=1
    -DSTREAM1_HANDOFF_FIXED_AFFINE=1 -DSTREAM1_CUTLASS_SKIP_GLOBAL_SFD=1)
fi
mkdir -p "$out"
# The role-separated kernel embeds logical-singleton CUTLASS collectives in a
# larger physical cluster. Apply the audited, idempotent CUTLASS adaptations
# on every build so the compile flag cannot silently target an unpatched tree
# (notably the physical-self EMPTY-barrier release for producer rank 1).
python3 "$repo/test_results/molab_patch_cutlass_void_d_epilogue.py"
if [[ "$accumulate_direct" == "1" ]]; then
  python3 "$repo/test_results/molab_patch_cutlass_accumulator_carry.py" "$cutlass"
fi
nvcc -std=c++20 "$opt" -lineinfo --threads=4 --expt-relaxed-constexpr -w \
  -Xcompiler=-Wno-missing-field-initializers \
  -Xcompiler=-Wno-error=missing-field-initializers \
  -Xcompiler=-Wno-error \
  "-DSTREAM1_FUSED_TEST_HIDDEN=${hidden}" -Xptxas=-v \
  "-DSTREAM1_FUSED_TEST_THREADS=${threads}" \
  "-DSTREAM1_FUSED_DIRECT_PACK=${direct_pack}" \
  "-DSTREAM1_FUSED_NO_RELOCATION=${no_relocation}" \
  "-DSTREAM1_FUSED_ACCUMULATE_DIRECT=${accumulate_direct}" \
  "-DSTREAM1_FUSED_SHARED_CARRY=${shared_carry}" \
  "-DSTREAM1_FUSED_PROXY_FENCE=${proxy_fence}" \
  "-DSTREAM1_FUSED_HANDOFF_TRACE=${handoff_trace}" \
  "-DSTREAM1_FUSED_INVALIDATE_BARRIERS=${invalidate_barriers}" \
  "-DSTREAM1_FUSED_NOINLINE_FF1=${noinline_ff1}" \
  "-DSTREAM1_FUSED_WAIT_DEPENDENT_GRIDS=${wait_dependent_grids}" \
  "-DSTREAM1_ROLE_SEPARATED_4CTA=${role_separated_4cta}" \
  "-DSTREAM1_ROLE_PRECOMPUTE_FF1=${role_precompute}" \
  "-DSTREAM1_ROLE_DUAL_FF1_MATH_WG=${role_dual_ff1_math_wg}" \
  -DSTREAM1_PHYSICAL_CLUSTER_LOGICAL_SINGLETON=1 \
  -DSTREAM1_CUTLASS_DIAG_DIRECT_FF1_COLLECTIVES=1 \
  -DSTREAM1_CUTLASS_DIAG_DIRECT_FF1_PREFETCH=1 \
  -DSTREAM1_CUTLASS_DIAG_DIRECT_FF1_PIPELINE_CONSTRUCT=1 \
  -DSTREAM1_CUTLASS_DIAG_DIRECT_FF1_LOAD_INIT=1 \
  -DSTREAM1_CUTLASS_DIAG_DIRECT_FF1_MMA=1 \
  "${materialize_flags[@]}" \
  --generate-code=arch=compute_120a,code=sm_120a \
  "-I${repo}/cuda" "-I${cutlass}/include" \
  "-I${cutlass}/tools/util/include" \
  "${cccl_flags[@]}" \
  "${repo}/tests/stream1_transformer_sm120_nvfp4_cutlass_ff1_ff2_fused_numeric_tests.cu" \
  -o "${out}/${binary}"
sha256sum "${out}/${binary}"
