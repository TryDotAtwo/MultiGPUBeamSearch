cmake_minimum_required(VERSION 3.22)
set(BEAM_CUDA_ARCHITECTURES "75;86" CACHE STRING "CUDA architectures to build, e.g. 75;86 or 120")
include(cmake/BeamStateSizing.cmake)
beam_configure_state_sizing()
message(STATUS "Beam state logical bytes: ${BEAM_STATE_LOGICAL_BYTES}")
message(STATUS "Beam state physical bytes: ${BEAM_STATE_PHYSICAL_BYTES}")
message(STATUS "Beam state alignment: ${BEAM_STATE_ALIGNMENT}")
message(STATUS "Beam move count: ${BEAM_MOVE_COUNT}")
project(beam_solver_contracts LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(BEAM_ENABLE_DEBUG "Master switch for all debug instrumentation" OFF)
option(BEAM_ENABLE_DEPTH_LOGS "Compile production_runner per-depth logs when BEAM_ENABLE_DEBUG is ON" OFF)
option(BEAM_ENABLE_DEBUG_LOGS "Compile production_runner debug/memory logs when BEAM_ENABLE_DEBUG is ON" OFF)
option(BEAM_DEBUG_STREAM_TIMING "Compile CUDA stream timing diagnostics when BEAM_ENABLE_DEBUG is ON" OFF)
option(BEAM_DEBUG_INFERENCE_TRACE "Compile Stream1 inference/reference diagnostics when BEAM_ENABLE_DEBUG is ON" OFF)
option(BEAM_DEBUG_PATH_TRACE "Compile tracked solution path diagnostics when BEAM_ENABLE_DEBUG is ON" OFF)
option(BEAM_DEBUG_FINAL_VALIDATE "Compile final request validation diagnostics when BEAM_ENABLE_DEBUG is ON" OFF)
option(BEAM_DEBUG_FINAL_EXCHANGE_TRACE "Compile multi-rank final exchange diagnostics when BEAM_ENABLE_DEBUG is ON" OFF)
option(BEAM_DEBUG_FINAL_HISTOGRAM_TRACE "Compile final histogram-vs-survivor diagnostics when BEAM_ENABLE_DEBUG is ON" OFF)
option(BEAM_DEBUG_STREAM4_HISTOGRAM_TRACE "Compile Stream4 histogram A/B diagnostics when BEAM_ENABLE_DEBUG is ON" OFF)
option(BEAM_DEBUG_DEPTH_FLOW_TRACE "Compile per-depth flow counter diagnostics when BEAM_ENABLE_DEBUG is ON" OFF)
option(BEAM_ENABLE_LIBTORCH_STREAM1 "Build explicit LibTorch Stream1 transformer tools" OFF)

set(BEAM_ENABLE_DEPTH_LOGS_EFFECTIVE OFF)
set(BEAM_ENABLE_DEBUG_LOGS_EFFECTIVE OFF)
set(BEAM_DEBUG_STREAM_TIMING_EFFECTIVE OFF)
set(BEAM_DEBUG_INFERENCE_TRACE_EFFECTIVE OFF)
set(BEAM_DEBUG_PATH_TRACE_EFFECTIVE OFF)
set(BEAM_DEBUG_FINAL_VALIDATE_EFFECTIVE OFF)
set(BEAM_DEBUG_FINAL_EXCHANGE_TRACE_EFFECTIVE OFF)
set(BEAM_DEBUG_FINAL_HISTOGRAM_TRACE_EFFECTIVE OFF)
set(BEAM_DEBUG_STREAM4_HISTOGRAM_TRACE_EFFECTIVE OFF)
set(BEAM_DEBUG_DEPTH_FLOW_TRACE_EFFECTIVE OFF)
if(BEAM_ENABLE_DEBUG)
    if(BEAM_ENABLE_DEPTH_LOGS)
        set(BEAM_ENABLE_DEPTH_LOGS_EFFECTIVE ON)
    endif()
    if(BEAM_ENABLE_DEBUG_LOGS)
        set(BEAM_ENABLE_DEBUG_LOGS_EFFECTIVE ON)
    endif()
    if(BEAM_DEBUG_STREAM_TIMING)
        set(BEAM_DEBUG_STREAM_TIMING_EFFECTIVE ON)
    endif()
    if(BEAM_DEBUG_INFERENCE_TRACE)
        set(BEAM_DEBUG_INFERENCE_TRACE_EFFECTIVE ON)
    endif()
    if(BEAM_DEBUG_PATH_TRACE)
        set(BEAM_DEBUG_PATH_TRACE_EFFECTIVE ON)
    endif()
    if(BEAM_DEBUG_FINAL_VALIDATE)
        set(BEAM_DEBUG_FINAL_VALIDATE_EFFECTIVE ON)
    endif()
    if(BEAM_DEBUG_FINAL_EXCHANGE_TRACE)
        set(BEAM_DEBUG_FINAL_EXCHANGE_TRACE_EFFECTIVE ON)
    endif()
    if(BEAM_DEBUG_FINAL_HISTOGRAM_TRACE)
        set(BEAM_DEBUG_FINAL_HISTOGRAM_TRACE_EFFECTIVE ON)
    endif()
    if(BEAM_DEBUG_STREAM4_HISTOGRAM_TRACE)
        set(BEAM_DEBUG_STREAM4_HISTOGRAM_TRACE_EFFECTIVE ON)
    endif()
    if(BEAM_DEBUG_DEPTH_FLOW_TRACE)
        set(BEAM_DEBUG_DEPTH_FLOW_TRACE_EFFECTIVE ON)
    endif()
endif()
set(CUTLASS_DIR "" CACHE PATH "CUTLASS checkout root")
if(NOT CUTLASS_DIR AND DEFINED ENV{CUTLASS_DIR})
    set(CUTLASS_DIR "$ENV{CUTLASS_DIR}" CACHE PATH "CUTLASS checkout root" FORCE)
endif()
if(NOT CUTLASS_DIR)
    foreach(candidate /opt/cutlass "${CMAKE_SOURCE_DIR}/external/cutlass" /kaggle/working/cutlass)
        if(EXISTS "${candidate}/include")
            set(CUTLASS_DIR "${candidate}" CACHE PATH "CUTLASS checkout root" FORCE)
            break()
        endif()
    endforeach()
endif()

add_library(beam_core
    src/config.cpp
    src/state.cpp
    src/hash.cpp
    src/history.cpp
    src/history_budget.cpp
    src/native_input.cpp
    src/stream3.cpp
    src/stream4.cpp
    src/frontier_cpu.cpp
)
target_include_directories(beam_core PUBLIC src)
target_compile_definitions(beam_core PUBLIC
    BEAM_STATE_LOGICAL_BYTES=${BEAM_STATE_LOGICAL_BYTES}
    BEAM_STATE_PHYSICAL_BYTES=${BEAM_STATE_PHYSICAL_BYTES}
    BEAM_STATE_ALIGNMENT=${BEAM_STATE_ALIGNMENT}
    BEAM_MOVE_COUNT=${BEAM_MOVE_COUNT}
)

add_executable(contract_tests tests/contract_tests.cpp)
target_link_libraries(contract_tests PRIVATE beam_core)

add_executable(history_tests tests/history_tests.cpp)
target_link_libraries(history_tests PRIVATE beam_core)

add_executable(history_budget_tests tests/history_budget_tests.cpp)
add_executable(history_prefault_tests tests/history_prefault_tests.cpp)
target_include_directories(history_prefault_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(history_budget_tests PRIVATE beam_core)

add_executable(config_limits_tests tests/config_limits_tests.cpp)
target_link_libraries(config_limits_tests PRIVATE beam_core)

add_executable(unique_frontier_tests tests/unique_frontier_tests.cpp)
target_link_libraries(unique_frontier_tests PRIVATE beam_core)

add_executable(native_input_tests tests/native_input_tests.cpp)
target_link_libraries(native_input_tests PRIVATE beam_core)
add_executable(native_manifest_tests tests/native_manifest_tests.cpp)
target_link_libraries(native_manifest_tests PRIVATE beam_core)
add_executable(tensor_binding_tests tests/tensor_binding_tests.cpp)
target_link_libraries(tensor_binding_tests PRIVATE beam_core)

add_library(beam_cuda
    cuda/stream1.cu
    cuda/stream1_transformer.cu
    cuda/stream1_blend_readout.cu
    cuda/stream1_ensemble_readout.cu
    cuda/stream1_transformer_fmha.cu
    cuda/stream2.cu
    cuda/stream3.cu
    cuda/stream4.cu
    cuda/stream5.cu
    cuda/threshold.cu
    cuda/final_materialize.cu
    cuda/dispatcher.cu
    cuda/static_memory.cu
    cuda/runtime_config.cpp
)
target_include_directories(beam_cuda PUBLIC src cuda ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
if(CUTLASS_DIR AND EXISTS "${CUTLASS_DIR}/include")
    target_include_directories(beam_cuda PUBLIC "${CUTLASS_DIR}/include")
    if(EXISTS "${CUTLASS_DIR}/tools/util/include")
        target_include_directories(beam_cuda PUBLIC "${CUTLASS_DIR}/tools/util/include")
    endif()
    if(EXISTS "${CUTLASS_DIR}/examples/41_fused_multi_head_attention/kernel_forward.h")
        target_include_directories(beam_cuda PUBLIC "${CUTLASS_DIR}/examples/41_fused_multi_head_attention")
        target_compile_definitions(beam_cuda PUBLIC BEAM_HAS_CUTLASS_FMHA=1)
    else()
        target_compile_definitions(beam_cuda PUBLIC BEAM_HAS_CUTLASS_FMHA=0)
    endif()
    target_compile_definitions(beam_cuda PUBLIC BEAM_HAS_CUTLASS=1)
else()
    message(FATAL_ERROR "CUTLASS_DIR is required. Pass -DCUTLASS_DIR=/path/to/cutlass or set CUTLASS_DIR.")
endif()
target_link_libraries(beam_cuda PUBLIC beam_core)
target_compile_options(beam_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_definitions(beam_cuda PUBLIC
    BEAM_ENABLE_DEBUG=$<BOOL:${BEAM_ENABLE_DEBUG}>
    BEAM_DEBUG_STREAM_TIMING=$<BOOL:${BEAM_DEBUG_STREAM_TIMING_EFFECTIVE}>
    BEAM_DEBUG_INFERENCE_TRACE=$<BOOL:${BEAM_DEBUG_INFERENCE_TRACE_EFFECTIVE}>
    BEAM_DEBUG_PATH_TRACE=$<BOOL:${BEAM_DEBUG_PATH_TRACE_EFFECTIVE}>
    BEAM_DEBUG_FINAL_VALIDATE=$<BOOL:${BEAM_DEBUG_FINAL_VALIDATE_EFFECTIVE}>
    BEAM_DEBUG_FINAL_EXCHANGE_TRACE=$<BOOL:${BEAM_DEBUG_FINAL_EXCHANGE_TRACE_EFFECTIVE}>
    BEAM_DEBUG_FINAL_HISTOGRAM_TRACE=$<BOOL:${BEAM_DEBUG_FINAL_HISTOGRAM_TRACE_EFFECTIVE}>
    BEAM_DEBUG_STREAM4_HISTOGRAM_TRACE=$<BOOL:${BEAM_DEBUG_STREAM4_HISTOGRAM_TRACE_EFFECTIVE}>
    BEAM_DEBUG_DEPTH_FLOW_TRACE=$<BOOL:${BEAM_DEBUG_DEPTH_FLOW_TRACE_EFFECTIVE}>
)
find_path(NCCL_INCLUDE_DIR nccl.h REQUIRED)
find_library(NCCL_LIBRARY nccl REQUIRED)
target_include_directories(beam_cuda PUBLIC ${NCCL_INCLUDE_DIR})
target_link_libraries(beam_cuda PUBLIC ${NCCL_LIBRARY})
find_library(NVTOOLSEXT_LIBRARY nvToolsExt PATHS /usr/local/cuda/lib64 /usr/local/cuda-12.8/lib64)
if(NVTOOLSEXT_LIBRARY)
    target_link_libraries(beam_cuda PUBLIC ${NVTOOLSEXT_LIBRARY})
endif()
set_target_properties(beam_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}"
)

add_executable(stream2_cuda_tests tests/stream2_cuda_tests.cu)
target_link_libraries(stream2_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream2_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream1_cuda_tests tests/stream1_cuda_tests.cu)
target_link_libraries(stream1_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream1_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream1_transformer_cuda_tests tests/stream1_transformer_cuda_tests.cu)
add_executable(device_weight_verification_cuda_tests tests/device_weight_verification_cuda_tests.cu)
target_include_directories(device_weight_verification_cuda_tests PRIVATE cuda)
set_target_properties(device_weight_verification_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
add_executable(frontier_native_score_probe EXCLUDE_FROM_ALL tests/frontier_native_score_probe.cu)
target_link_libraries(frontier_native_score_probe PRIVATE beam_cuda)
set_target_properties(frontier_native_score_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
target_link_libraries(stream1_transformer_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream1_transformer_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream1_transformer_final_cls_cuda_tests tests/stream1_transformer_final_cls_cuda_tests.cu)
target_link_libraries(stream1_transformer_final_cls_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream1_transformer_final_cls_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream1_transformer_hopper_gemm_cuda_tests tests/stream1_transformer_hopper_gemm_cuda_tests.cu)
add_executable(stream1_hopper_epilogue_tests EXCLUDE_FROM_ALL tests/stream1_hopper_epilogue_tests.cu)
target_link_libraries(stream1_hopper_epilogue_tests PRIVATE beam_cuda)
target_compile_options(stream1_hopper_epilogue_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(stream1_hopper_epilogue_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
# Rejected performance experiment, kept explicitly buildable for reproduction.
add_executable(stream1_transformer_ln_qkv_fusion_cuda_tests EXCLUDE_FROM_ALL tests/stream1_transformer_ln_qkv_fusion_cuda_tests.cu)
add_executable(stream1_transformer_dual_ln_cuda_tests tests/stream1_transformer_dual_ln_cuda_tests.cu)
target_link_libraries(stream1_transformer_dual_ln_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream1_transformer_dual_ln_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
target_link_libraries(stream1_transformer_ln_qkv_fusion_cuda_tests PRIVATE beam_cuda)
target_compile_options(stream1_transformer_ln_qkv_fusion_cuda_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(stream1_transformer_ln_qkv_fusion_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
target_link_libraries(stream1_transformer_hopper_gemm_cuda_tests PRIVATE beam_cuda)
target_compile_options(stream1_transformer_hopper_gemm_cuda_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(stream1_transformer_hopper_gemm_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream1_transformer_gemm_policy_tests tests/stream1_transformer_gemm_policy_tests.cpp)
add_executable(stream1_resolved_gemm_tests tests/stream1_resolved_gemm_tests.cpp)
add_executable(production_score_request_tests tests/production_score_request_tests.cpp)
target_link_libraries(production_score_request_tests PRIVATE beam_core)
target_compile_features(production_score_request_tests PRIVATE cxx_std_17)
target_compile_features(stream1_resolved_gemm_tests PRIVATE cxx_std_17)
if(UNIX)
    add_executable(stream1_policy_snapshot_tests tests/stream1_policy_snapshot_tests.cpp)
    target_compile_features(stream1_policy_snapshot_tests PRIVATE cxx_std_17)
endif()
add_executable(stream1_transformer_hopper_policy_tests tests/stream1_transformer_hopper_policy_tests.cpp)
add_executable(stream1_layout_metadata_tests tests/stream1_layout_metadata_tests.cpp)
add_executable(stream1_loaded_execution_admission_tests tests/stream1_loaded_execution_admission_tests.cpp)
add_executable(stream1_execution_contract_tests tests/stream1_execution_contract_tests.cpp)
target_compile_features(stream1_execution_contract_tests PRIVATE cxx_std_17)
target_link_libraries(stream1_execution_contract_tests PRIVATE beam_core)
target_include_directories(stream1_execution_contract_tests PRIVATE cuda ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_features(stream1_loaded_execution_admission_tests PRIVATE cxx_std_17)
target_link_libraries(stream1_loaded_execution_admission_tests PRIVATE beam_core)
target_include_directories(stream1_loaded_execution_admission_tests PRIVATE cuda ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(stream1_layout_metadata_tests PRIVATE beam_core)
target_include_directories(stream1_layout_metadata_tests PRIVATE cuda ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(stream1_transformer_hopper_policy_tests PRIVATE cuda)
target_include_directories(stream1_transformer_gemm_policy_tests PRIVATE cuda)
target_compile_features(stream1_transformer_gemm_policy_tests PRIVATE cxx_std_17)
add_executable(stream1_transformer_attention_policy_tests tests/stream1_transformer_attention_policy_tests.cpp)
add_executable(stream1_transformer_shape_tests tests/stream1_transformer_shape_tests.cpp)
target_link_libraries(stream1_transformer_shape_tests PRIVATE beam_core)
add_executable(stream1_transformer_layernorm_policy_tests tests/stream1_transformer_layernorm_policy_tests.cpp)
add_executable(stream1_transformer_layernorm_split_slots_policy_tests tests/stream1_transformer_layernorm_policy_tests.cpp)
add_executable(stream1_transformer_autotune_contract_tests tests/stream1_transformer_autotune_contract_tests.cpp)
target_include_directories(stream1_transformer_attention_policy_tests PRIVATE cuda)
target_include_directories(stream1_transformer_shape_tests PRIVATE cuda)
target_include_directories(stream1_transformer_layernorm_policy_tests PRIVATE cuda)
target_include_directories(stream1_transformer_layernorm_split_slots_policy_tests PRIVATE cuda)
target_compile_definitions(stream1_transformer_layernorm_split_slots_policy_tests PRIVATE STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS=1)
target_compile_features(stream1_transformer_layernorm_policy_tests PRIVATE cxx_std_17)
target_compile_features(stream1_transformer_layernorm_split_slots_policy_tests PRIVATE cxx_std_17)
target_compile_features(stream1_transformer_attention_policy_tests PRIVATE cxx_std_17)
target_compile_features(stream1_transformer_shape_tests PRIVATE cxx_std_17)
target_include_directories(stream1_transformer_autotune_contract_tests PRIVATE cuda)
target_compile_features(stream1_transformer_autotune_contract_tests PRIVATE cxx_std_17)


add_executable(stream3_cuda_tests tests/stream3_cuda_tests.cu)
target_link_libraries(stream3_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream3_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

# Bounded owner-scatter acceptance: opt-in and never promoted by host checks.
add_executable(stream3_owner_scatter_cuda_tests EXCLUDE_FROM_ALL tests/stream3_owner_scatter_cuda_tests.cu)
target_link_libraries(stream3_owner_scatter_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream3_owner_scatter_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream4_cuda_tests tests/stream4_cuda_tests.cu)
target_link_libraries(stream4_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream4_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream4_logical_union_cuda_tests tests/stream4_logical_union_cuda_tests.cu)
target_link_libraries(stream4_logical_union_cuda_tests PRIVATE beam_cuda)
target_compile_definitions(stream4_logical_union_cuda_tests PRIVATE BEAM_TEST_LOGICAL_UNION=1)
set_target_properties(stream4_logical_union_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

# Separate opt-in scopes: single GPU must not masquerade as two-GPU acceptance.
foreach(union_scope IN ITEMS graph single_graph bounded_graph single_bounded_graph)
    set(union_target stream4_logical_union_${union_scope}_cuda_tests)
    add_executable(${union_target} EXCLUDE_FROM_ALL tests/stream4_logical_union_cuda_tests.cu)
    target_link_libraries(${union_target} PRIVATE beam_cuda)
    target_compile_definitions(${union_target} PRIVATE BEAM_TEST_LOGICAL_UNION=1 BEAM_TEST_UNION_GRAPH=1)
    if(union_scope MATCHES "^single_")
        target_compile_definitions(${union_target} PRIVATE BEAM_TEST_SINGLE_GPU=1)
    endif()
    if(union_scope MATCHES "bounded_graph$")
        target_compile_definitions(${union_target} PRIVATE BEAM_TEST_UNION_SORT_BOUND=1)
    endif()
    set_target_properties(${union_target} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
endforeach()

add_executable(union_memory_cuda_tests tests/union_memory_cuda_tests.cu)
add_executable(runtime_arithmetic_tests tests/runtime_arithmetic_tests.cpp)
target_link_libraries(runtime_arithmetic_tests PRIVATE beam_cuda)
target_link_libraries(union_memory_cuda_tests PRIVATE beam_cuda)
set_target_properties(union_memory_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream5_cuda_tests tests/stream5_cuda_tests.cu)
target_link_libraries(stream5_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream5_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(final_cuda_tests tests/final_cuda_tests.cu)
target_link_libraries(final_cuda_tests PRIVATE beam_cuda)
set_target_properties(final_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(threshold_cuda_tests tests/threshold_cuda_tests.cu)
target_link_libraries(threshold_cuda_tests PRIVATE beam_cuda)
set_target_properties(threshold_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(final_union_dispatcher_cuda_tests EXCLUDE_FROM_ALL tests/final_union_dispatcher_cuda_tests.cu)
add_executable(stream4_bounded_graph_cuda_tests tests/stream4_bounded_graph_cuda_tests.cu)
target_link_libraries(stream4_bounded_graph_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream4_bounded_graph_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
add_executable(final_snapshot_validation_cuda_tests tests/final_snapshot_validation_cuda_tests.cu)
add_executable(stream1_execution_shape_tests tests/stream1_execution_shape_tests.cpp)
add_executable(stream1_executor_contract_tests tests/stream1_executor_contract_tests.cpp)
add_executable(stream1_chunk_plan_tests tests/stream1_chunk_plan_tests.cpp)
target_include_directories(stream1_chunk_plan_tests PRIVATE cuda)
target_compile_features(stream1_chunk_plan_tests PRIVATE cxx_std_17)
add_test(NAME stream1_chunk_plan_tests COMMAND stream1_chunk_plan_tests)
add_executable(stream1_kernel_description_tests tests/stream1_kernel_description_tests.cpp)
target_include_directories(stream1_kernel_description_tests PRIVATE cuda)
target_compile_features(stream1_kernel_description_tests PRIVATE cxx_std_17)
add_test(NAME stream1_kernel_description_tests COMMAND stream1_kernel_description_tests)
add_executable(stream1_nonfinite_score_cuda_tests tests/stream1_nonfinite_score_cuda_tests.cu)
target_link_libraries(stream1_nonfinite_score_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream1_nonfinite_score_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
foreach(score_case IN ITEMS finite nan posinf neginf)
    add_test(NAME stream1_nonfinite_score_${score_case} COMMAND stream1_nonfinite_score_cuda_tests ${score_case})
    add_test(NAME stream1_nonfinite_graph_job_score_${score_case} COMMAND stream1_nonfinite_score_cuda_tests ${score_case} --graph-job)
endforeach()
target_compile_features(stream1_executor_contract_tests PRIVATE cxx_std_17)
target_compile_features(stream1_execution_shape_tests PRIVATE cxx_std_17)
add_executable(cuda_visible_device_probe EXCLUDE_FROM_ALL tools/cuda_visible_device_probe.cu)
set_target_properties(cuda_visible_device_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
add_executable(stream1_parameter_storage_probe EXCLUDE_FROM_ALL tools/stream1_parameter_storage_probe.cu)
target_link_libraries(stream1_parameter_storage_probe PRIVATE beam_core)
set_target_properties(stream1_parameter_storage_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
add_executable(stream1_gemm_parameter_probe EXCLUDE_FROM_ALL tools/stream1_gemm_parameter_storage_probe.cu)
target_include_directories(stream1_gemm_parameter_probe PRIVATE "${CUTLASS_DIR}/include")
target_compile_options(stream1_gemm_parameter_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(stream1_gemm_parameter_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
add_executable(stream1_gemm_iterator_probe EXCLUDE_FROM_ALL tools/stream1_gemm_iterator_probe.cu)
add_executable(stream1_transfer_geometry_probe EXCLUDE_FROM_ALL tools/stream1_transfer_geometry_probe.cu)
set_target_properties(stream1_transfer_geometry_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
add_executable(stream1_elementwise_arguments_probe EXCLUDE_FROM_ALL tools/stream1_elementwise_arguments_probe.cu)
add_executable(stream1_gemm_field_ledger_probe EXCLUDE_FROM_ALL tools/stream1_gemm_field_ledger_probe.cu)
add_executable(stream1_gemm_nested_storage_probe EXCLUDE_FROM_ALL tools/stream1_gemm_nested_storage_probe.cu)
target_include_directories(stream1_gemm_nested_storage_probe PRIVATE "${CUTLASS_DIR}/include")
set_target_properties(stream1_gemm_nested_storage_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
target_include_directories(stream1_gemm_field_ledger_probe PRIVATE "${CUTLASS_DIR}/include")
set_target_properties(stream1_gemm_field_ledger_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
add_executable(stream1_attention_field_ledger_probe EXCLUDE_FROM_ALL tools/stream1_attention_field_ledger_probe.cu)
target_include_directories(stream1_attention_field_ledger_probe PRIVATE "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/examples/41_fused_multi_head_attention")
set_target_properties(stream1_attention_field_ledger_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
target_include_directories(stream1_elementwise_arguments_probe PRIVATE "${CUTLASS_DIR}/include")
set_target_properties(stream1_elementwise_arguments_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
target_include_directories(stream1_gemm_iterator_probe PRIVATE "${CUTLASS_DIR}/include")
target_compile_options(stream1_gemm_iterator_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_target_properties(stream1_gemm_iterator_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
if(EXISTS "${CUTLASS_DIR}/examples/41_fused_multi_head_attention/kernel_forward.h")
    add_executable(stream1_attention_storage_probe EXCLUDE_FROM_ALL tools/stream1_attention_storage_probe.cu)
    target_include_directories(stream1_attention_storage_probe PRIVATE "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/examples/41_fused_multi_head_attention")
    target_compile_options(stream1_attention_storage_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(stream1_attention_storage_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
    add_executable(stream1_attention_parameter_probe EXCLUDE_FROM_ALL tools/stream1_attention_storage_probe.cu)
    target_include_directories(stream1_attention_parameter_probe PRIVATE "${CUTLASS_DIR}/include" "${CUTLASS_DIR}/examples/41_fused_multi_head_attention")
    target_compile_definitions(stream1_attention_parameter_probe PRIVATE STREAM1_ATTENTION_PARAMETER_LAYOUT_PROBE=1)
    target_compile_options(stream1_attention_parameter_probe PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(stream1_attention_parameter_probe PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
endif()
target_link_libraries(final_snapshot_validation_cuda_tests PRIVATE beam_cuda)
set_target_properties(final_snapshot_validation_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
target_link_libraries(final_union_dispatcher_cuda_tests PRIVATE beam_cuda)
set_target_properties(final_union_dispatcher_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(threshold_union_cuda_tests tests/threshold_union_cuda_tests.cu)
target_link_libraries(threshold_union_cuda_tests PRIVATE beam_cuda)
target_compile_definitions(threshold_union_cuda_tests PRIVATE BEAM_TEST_CONSERVATIVE=1)
set_target_properties(threshold_union_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stitched_cuda_tests tests/stitched_cuda_tests.cu)
target_link_libraries(stitched_cuda_tests PRIVATE beam_cuda)
set_target_properties(stitched_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(dispatcher_cuda_tests tests/dispatcher_cuda_tests.cu)
target_link_libraries(dispatcher_cuda_tests PRIVATE beam_cuda)
set_target_properties(dispatcher_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(static_memory_cuda_tests tests/static_memory_cuda_tests.cu)
target_link_libraries(static_memory_cuda_tests PRIVATE beam_cuda)
set_target_properties(static_memory_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(production_runner tools/production_runner.cu)
string(REPLACE ";" "," BEAM_BUILD_CUDA_ARCHES "${BEAM_CUDA_ARCHITECTURES}")
target_link_libraries(production_runner PRIVATE beam_cuda ${CMAKE_DL_LIBS})
target_compile_definitions(production_runner PRIVATE
    BEAM_BUILD_CUDA_ARCHES="${BEAM_BUILD_CUDA_ARCHES}"
    BEAM_ENABLE_DEPTH_LOGS=$<BOOL:${BEAM_ENABLE_DEPTH_LOGS_EFFECTIVE}>
    BEAM_ENABLE_DEBUG_LOGS=$<BOOL:${BEAM_ENABLE_DEBUG_LOGS_EFFECTIVE}>
)
set_target_properties(production_runner PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream_benchmark
    tools/stream_benchmark.cu
    tools/stream_benchmark_mlp.cu
    tools/stream_benchmark_transformer.cu)
target_link_libraries(stream_benchmark PRIVATE beam_cuda)
set_target_properties(stream_benchmark PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream_pipeline_benchmark tools/stream_pipeline_benchmark.cu)
target_link_libraries(stream_pipeline_benchmark PRIVATE beam_cuda)
set_target_properties(stream_pipeline_benchmark PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")
if(BEAM_ENABLE_LIBTORCH_STREAM1)
    find_package(Torch REQUIRED)
    add_executable(stream1_transformer_libtorch_benchmark
        tools/stream1_transformer_libtorch_benchmark.cpp
        tools/stream1_transformer_libtorch_backend.cpp)
    target_link_libraries(stream1_transformer_libtorch_benchmark PRIVATE "${TORCH_LIBRARIES}")
    target_include_directories(stream1_transformer_libtorch_benchmark PRIVATE "${TORCH_INCLUDE_DIRS}")
    target_compile_definitions(stream1_transformer_libtorch_benchmark PRIVATE BEAM_HAS_LIBTORCH_STREAM1=1)
    set_target_properties(stream1_transformer_libtorch_benchmark PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)

    add_executable(production_runner_libtorch_stream1
        tools/production_runner.cu
        tools/stream1_libtorch_ring_slot_launcher.cpp
        tools/stream1_transformer_libtorch_backend.cpp)
    add_executable(stream1_ensemble_benchmark tools/stream1_ensemble_benchmark.cpp)
    add_executable(stream1_native_mlp_benchmark tools/stream1_native_mlp_benchmark.cpp)
    add_executable(stream1_libtorch_mlp_benchmark tools/stream1_libtorch_mlp_benchmark.cpp)
    target_link_libraries(stream1_libtorch_mlp_benchmark PRIVATE beam_cuda "${TORCH_LIBRARIES}")
    target_include_directories(stream1_libtorch_mlp_benchmark PRIVATE "${TORCH_INCLUDE_DIRS}")
    set_target_properties(stream1_libtorch_mlp_benchmark PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    target_link_libraries(stream1_native_mlp_benchmark PRIVATE beam_cuda "${TORCH_LIBRARIES}")
    target_include_directories(stream1_native_mlp_benchmark PRIVATE "${TORCH_INCLUDE_DIRS}")
    set_target_properties(stream1_native_mlp_benchmark PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    target_link_libraries(stream1_ensemble_benchmark PRIVATE beam_cuda "${TORCH_LIBRARIES}")
    target_include_directories(stream1_ensemble_benchmark PRIVATE "${TORCH_INCLUDE_DIRS}")
    set_target_properties(stream1_ensemble_benchmark PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    target_link_libraries(production_runner_libtorch_stream1 PRIVATE beam_cuda "${TORCH_LIBRARIES}")
    target_include_directories(production_runner_libtorch_stream1 PRIVATE "${TORCH_INCLUDE_DIRS}")
    target_compile_definitions(production_runner_libtorch_stream1 PRIVATE
        BEAM_BUILD_CUDA_ARCHES="${BEAM_BUILD_CUDA_ARCHES}"
        BEAM_HAS_LIBTORCH_STREAM1=1
        BEAM_ENABLE_DEPTH_LOGS=$<BOOL:${BEAM_ENABLE_DEPTH_LOGS_EFFECTIVE}>
        BEAM_ENABLE_DEBUG_LOGS=$<BOOL:${BEAM_ENABLE_DEBUG_LOGS_EFFECTIVE}>
    )
    set_target_properties(production_runner_libtorch_stream1 PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}"
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF)
endif()

enable_testing()
add_test(NAME production_score_request_tests COMMAND production_score_request_tests)
add_test(NAME stream1_loaded_execution_admission_tests COMMAND stream1_loaded_execution_admission_tests)
add_test(NAME stream1_execution_contract_tests COMMAND stream1_execution_contract_tests)
add_test(NAME stream1_resolved_gemm_tests COMMAND stream1_resolved_gemm_tests)
if(UNIX)
    add_test(NAME stream1_policy_snapshot_tests COMMAND stream1_policy_snapshot_tests)
endif()
add_test(NAME stream1_layout_metadata_tests COMMAND stream1_layout_metadata_tests)
add_test(NAME contract_tests COMMAND contract_tests)
add_test(NAME history_tests COMMAND history_tests)
add_test(NAME history_budget_tests COMMAND history_budget_tests)
add_test(NAME history_prefault_tests COMMAND history_prefault_tests)
add_test(NAME native_input_tests COMMAND native_input_tests)
add_test(NAME native_manifest_tests COMMAND native_manifest_tests)
add_test(NAME production_runner_build_info COMMAND production_runner --build-info)
add_test(NAME tensor_binding_tests COMMAND tensor_binding_tests)
add_test(NAME config_limits_tests COMMAND config_limits_tests)
add_test(NAME unique_frontier_tests COMMAND unique_frontier_tests)
add_test(NAME stream1_cuda_tests COMMAND stream1_cuda_tests)
add_test(NAME stream1_transformer_cuda_tests COMMAND stream1_transformer_cuda_tests)
add_test(NAME device_weight_verification_cuda_tests COMMAND device_weight_verification_cuda_tests)
set_tests_properties(stream1_transformer_cuda_tests PROPERTIES SKIP_RETURN_CODE 77)
# A skipped optional CTest must never satisfy the required inference gate.
add_custom_target(check_stream1_reference_required
    COMMAND $<TARGET_FILE:stream1_transformer_cuda_tests> --require-reference
    DEPENDS stream1_transformer_cuda_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    USES_TERMINAL
)
add_test(NAME stream1_transformer_dual_ln_cuda_tests COMMAND stream1_transformer_dual_ln_cuda_tests)
add_test(NAME stream1_transformer_gemm_policy_tests COMMAND stream1_transformer_gemm_policy_tests)
add_test(NAME stream1_transformer_attention_policy_tests COMMAND stream1_transformer_attention_policy_tests)
add_test(NAME stream1_transformer_shape_tests COMMAND stream1_transformer_shape_tests)
add_test(NAME stream1_transformer_final_cls_cuda_tests COMMAND stream1_transformer_final_cls_cuda_tests)
add_test(NAME stream1_transformer_layernorm_policy_tests COMMAND stream1_transformer_layernorm_policy_tests)
add_test(NAME stream1_transformer_layernorm_split_slots_policy_tests COMMAND stream1_transformer_layernorm_split_slots_policy_tests)
add_test(NAME stream1_transformer_autotune_contract_tests COMMAND stream1_transformer_autotune_contract_tests)
add_test(NAME stream2_cuda_tests COMMAND stream2_cuda_tests)
add_test(NAME stream3_cuda_tests COMMAND stream3_cuda_tests)
add_test(NAME stream4_cuda_tests COMMAND stream4_cuda_tests)
add_test(NAME stream4_logical_union_cuda_tests COMMAND stream4_logical_union_cuda_tests)
add_test(NAME stream4_bounded_graph_cuda_tests COMMAND stream4_bounded_graph_cuda_tests)
add_test(NAME final_snapshot_validation_cuda_tests COMMAND final_snapshot_validation_cuda_tests)
add_test(NAME stream1_execution_shape_tests COMMAND stream1_execution_shape_tests)
add_test(NAME stream1_executor_contract_tests COMMAND stream1_executor_contract_tests)
set_tests_properties(stream1_executor_contract_tests PROPERTIES LABELS "host" TIMEOUT 30)
set_tests_properties(stream1_execution_shape_tests PROPERTIES LABELS "host" TIMEOUT 30)
set_tests_properties(final_snapshot_validation_cuda_tests PROPERTIES LABELS "host;cuda_linked" TIMEOUT 30)
add_test(NAME union_memory_cuda_tests COMMAND union_memory_cuda_tests)
add_test(NAME runtime_arithmetic_tests COMMAND runtime_arithmetic_tests)
set_tests_properties(union_memory_cuda_tests PROPERTIES LABELS "gpu" TIMEOUT 120)
set_tests_properties(stream4_logical_union_cuda_tests PROPERTIES LABELS "gpu;multi_gpu" TIMEOUT 120)
add_test(NAME stream5_cuda_tests COMMAND stream5_cuda_tests)
add_test(NAME final_cuda_tests COMMAND final_cuda_tests)
add_test(NAME threshold_cuda_tests COMMAND threshold_cuda_tests)
add_test(NAME threshold_union_cuda_tests COMMAND threshold_union_cuda_tests)
set_tests_properties(threshold_union_cuda_tests PROPERTIES LABELS "gpu;multi_gpu" TIMEOUT 120)
add_test(NAME stitched_cuda_tests COMMAND stitched_cuda_tests)
add_test(NAME dispatcher_cuda_tests COMMAND dispatcher_cuda_tests)
add_test(NAME static_memory_cuda_tests COMMAND static_memory_cuda_tests)

