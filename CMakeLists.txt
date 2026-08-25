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
option(BEAM_ENABLE_SM120_FP8 "Build the opt-in SM120 block-scaled FP8 Transformer path" OFF)
option(BEAM_BUILD_PRODUCTION_RUNNER_SHARED "Build the in-process production runner C ABI" OFF)

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

add_library(beam_cuda
    cuda/stream1.cu
    cuda/stream1_transformer.cu
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
find_library(CUBLASLT_LIBRARY NAMES cublasLt libcublasLt.so.13
    PATHS /usr/local/lib/python3.13/site-packages/nvidia/cu13/lib
          /tmp/uv-venv/lib/python3.13/site-packages/nvidia/cu13/lib
          /usr/local/cuda/lib64)
if(NOT CUBLASLT_LIBRARY)
    message(FATAL_ERROR "cuBLASLt is required for the Transformer FP16 benchmark path")
endif()
target_link_libraries(beam_cuda PUBLIC beam_core ${CUBLASLT_LIBRARY})
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
target_link_libraries(stream1_transformer_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream1_transformer_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream1_transformer_final_cls_cuda_tests tests/stream1_transformer_final_cls_cuda_tests.cu)
target_link_libraries(stream1_transformer_final_cls_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream1_transformer_final_cls_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

if(BEAM_ENABLE_SM120_FP8)
    target_sources(beam_cuda PRIVATE cuda/stream1_transformer_sm120_fp8.cu cuda/stream1_transformer_cublaslt_fp16.cu)
    target_compile_definitions(beam_cuda PUBLIC BEAM_ENABLE_SM120_FP8=1)
    add_executable(stream1_transformer_sm120_fp8_cuda_tests tests/stream1_transformer_sm120_fp8_cuda_tests.cu)
    target_link_libraries(stream1_transformer_sm120_fp8_cuda_tests PRIVATE beam_cuda)
    set_target_properties(stream1_transformer_sm120_fp8_cuda_tests PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}"
    )
    add_executable(stream1_transformer_sm120_fp8_benchmark tests/stream1_transformer_sm120_fp8_benchmark.cu)
    target_link_libraries(stream1_transformer_sm120_fp8_benchmark PRIVATE beam_cuda)
    set_target_properties(stream1_transformer_sm120_fp8_benchmark PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}"
    )
    add_executable(stream1_transformer_sm120_mxfp8_benchmark
        tests/stream1_transformer_sm120_mxfp8_benchmark.cu)
    target_link_libraries(stream1_transformer_sm120_mxfp8_benchmark PRIVATE beam_cuda)
    target_compile_options(stream1_transformer_sm120_mxfp8_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(stream1_transformer_sm120_mxfp8_benchmark PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}"
    )
    add_executable(stream1_transformer_sm120_sparse_mxfp8_benchmark
        tests/stream1_transformer_sm120_sparse_mxfp8_benchmark.cu)
    target_link_libraries(stream1_transformer_sm120_sparse_mxfp8_benchmark PRIVATE beam_cuda)
    target_include_directories(stream1_transformer_sm120_sparse_mxfp8_benchmark PRIVATE
        "${CMAKE_SOURCE_DIR}/tests/cutlass_test_stub"
        "${CUTLASS_DIR}/test/unit/gemm/device"
        "${CUTLASS_DIR}/test/unit/common")
    target_compile_options(stream1_transformer_sm120_sparse_mxfp8_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --threads=8>)
    set_target_properties(stream1_transformer_sm120_sparse_mxfp8_benchmark PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}"
    )
    add_executable(stream1_transformer_sm120_sparse_nvfp4_benchmark
        tests/stream1_transformer_sm120_sparse_mxfp8_benchmark.cu)
    target_link_libraries(stream1_transformer_sm120_sparse_nvfp4_benchmark PRIVATE beam_cuda)
    target_include_directories(stream1_transformer_sm120_sparse_nvfp4_benchmark PRIVATE
        "${CMAKE_SOURCE_DIR}/tests/cutlass_test_stub"
        "${CUTLASS_DIR}/test/unit/gemm/device"
        "${CUTLASS_DIR}/test/unit/common")
    target_compile_definitions(stream1_transformer_sm120_sparse_nvfp4_benchmark PRIVATE
        BEAM_BENCH_NVFP4=1)
    target_compile_options(stream1_transformer_sm120_sparse_nvfp4_benchmark PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --threads=8>)
    set_target_properties(stream1_transformer_sm120_sparse_nvfp4_benchmark PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}"
    )
    add_executable(stream1_transformer_sm120_fp8_export tools/stream1_transformer_sm120_fp8_export.cu)
    target_link_libraries(stream1_transformer_sm120_fp8_export PRIVATE beam_cuda)
    set_target_properties(stream1_transformer_sm120_fp8_export PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}"
    )
    add_executable(stream1_transformer_sm120_fp8_policy_tests
        tests/stream1_transformer_sm120_fp8_policy_tests.cpp)
    target_include_directories(stream1_transformer_sm120_fp8_policy_tests PRIVATE cuda)
    target_compile_features(stream1_transformer_sm120_fp8_policy_tests PRIVATE cxx_std_17)
else()
    target_compile_definitions(beam_cuda PUBLIC BEAM_ENABLE_SM120_FP8=0)
endif()

add_executable(stream1_transformer_gemm_policy_tests tests/stream1_transformer_gemm_policy_tests.cpp)
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
add_executable(sha256_tests tests/sha256_tests.cpp)
target_compile_features(sha256_tests PRIVATE cxx_std_17)
target_include_directories(stream1_transformer_autotune_contract_tests PRIVATE cuda)
target_compile_features(stream1_transformer_autotune_contract_tests PRIVATE cxx_std_17)


add_executable(stream3_cuda_tests tests/stream3_cuda_tests.cu)
target_link_libraries(stream3_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream3_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream4_cuda_tests tests/stream4_cuda_tests.cu)
target_link_libraries(stream4_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream4_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(stream5_cuda_tests tests/stream5_cuda_tests.cu)
target_link_libraries(stream5_cuda_tests PRIVATE beam_cuda)
set_target_properties(stream5_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(final_cuda_tests tests/final_cuda_tests.cu)
target_link_libraries(final_cuda_tests PRIVATE beam_cuda)
set_target_properties(final_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

add_executable(threshold_cuda_tests tests/threshold_cuda_tests.cu)
target_link_libraries(threshold_cuda_tests PRIVATE beam_cuda)
set_target_properties(threshold_cuda_tests PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

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
target_link_libraries(production_runner PRIVATE beam_cuda)
target_compile_definitions(production_runner PRIVATE
    BEAM_ENABLE_DEPTH_LOGS=$<BOOL:${BEAM_ENABLE_DEPTH_LOGS_EFFECTIVE}>
    BEAM_ENABLE_DEBUG_LOGS=$<BOOL:${BEAM_ENABLE_DEBUG_LOGS_EFFECTIVE}>
)
set_target_properties(production_runner PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}")

if(BEAM_BUILD_PRODUCTION_RUNNER_SHARED)
    set_target_properties(beam_core beam_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    add_library(production_runner_shared SHARED tools/production_runner.cu)
    target_link_libraries(production_runner_shared PRIVATE beam_cuda)
    target_compile_definitions(production_runner_shared PRIVATE
        BEAM_ENABLE_DEPTH_LOGS=$<BOOL:${BEAM_ENABLE_DEPTH_LOGS_EFFECTIVE}>
        BEAM_ENABLE_DEBUG_LOGS=$<BOOL:${BEAM_ENABLE_DEBUG_LOGS_EFFECTIVE}>
        BEAM_PRODUCTION_RUNNER_NO_MAIN=1
    )
    set_target_properties(production_runner_shared PROPERTIES
        OUTPUT_NAME beam_production_runner
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${BEAM_CUDA_ARCHITECTURES}"
    )
endif()

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
    target_link_libraries(production_runner_libtorch_stream1 PRIVATE beam_cuda "${TORCH_LIBRARIES}")
    target_include_directories(production_runner_libtorch_stream1 PRIVATE "${TORCH_INCLUDE_DIRS}")
    target_compile_definitions(production_runner_libtorch_stream1 PRIVATE
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
add_test(NAME contract_tests COMMAND contract_tests)
add_test(NAME history_tests COMMAND history_tests)
add_test(NAME stream1_cuda_tests COMMAND stream1_cuda_tests)
add_test(NAME stream1_transformer_cuda_tests COMMAND stream1_transformer_cuda_tests)
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
add_test(NAME stream5_cuda_tests COMMAND stream5_cuda_tests)
add_test(NAME final_cuda_tests COMMAND final_cuda_tests)
add_test(NAME threshold_cuda_tests COMMAND threshold_cuda_tests)
add_test(NAME stitched_cuda_tests COMMAND stitched_cuda_tests)
add_test(NAME dispatcher_cuda_tests COMMAND dispatcher_cuda_tests)
add_test(NAME static_memory_cuda_tests COMMAND static_memory_cuda_tests)
