#include "stream1_transformer_sm120_fp8.hpp"

#include "cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if BEAM_ENABLE_SM120_FP8
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#endif

namespace {

constexpr std::uint32_t kScaleGranularity = 128U;
constexpr float kE4m3Max = 448.0f;

void validate_common_shape(
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols) {
    if (rows == 0U || input_cols == 0U || output_cols == 0U) {
        throw std::invalid_argument("SM120 FP8 GEMM dimensions must be non-zero");
    }
    if (input_cols % kScaleGranularity != 0U ||
        output_cols % kScaleGranularity != 0U) {
        throw std::invalid_argument(
            "SM120 FP8 GEMM input/output columns must be divisible by 128");
    }
}

void validate_scale_multiplier(float scale_multiplier) {
    if (!std::isfinite(scale_multiplier) || scale_multiplier <= 0.0f) {
        throw std::invalid_argument("SM120 FP8 scale multiplier must be finite and positive");
    }
}

#if BEAM_ENABLE_SM120_FP8 && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using ElementD = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ElementCompute = float;
using MmaTileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<1, 128, 128>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassTensorOp,
    MmaTileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    void,
    LayoutD,
    AlignmentD,
    ElementD,
    LayoutD,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassTensorOp,
    ElementA,
    cute::tuple<LayoutA, LayoutSFA>,
    AlignmentA,
    ElementB,
    cute::tuple<LayoutB, LayoutSFB>,
    AlignmentB,
    ElementAccumulator,
    MmaTileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;

__device__ __forceinline__ float block_max(float value, float* warp_maxima) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    if (lane == 0U) {
        warp_maxima[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < 4U ? warp_maxima[threadIdx.x] : 0.0f;
    if (warp == 0U) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
        }
        if (lane == 0U) {
            warp_maxima[0] = value;
        }
    }
    __syncthreads();
    return warp_maxima[0];
}

__global__ void quantize_activation_kernel(
    const half* __restrict__ input,
    ElementA* __restrict__ quantized,
    float* __restrict__ scales,
    std::uint32_t rows,
    std::uint32_t input_cols,
    float scale_multiplier) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t k_block = blockIdx.y;
    const std::uint32_t col = k_block * kScaleGranularity + threadIdx.x;
    __shared__ float warp_maxima[4];
    const std::size_t index = static_cast<std::size_t>(row) * input_cols + col;
    const float value = __half2float(input[index]);
    const float max_abs = block_max(fabsf(value), warp_maxima);
    if (threadIdx.x == 0U) {
        const float scale = max_abs == 0.0f
            ? 1.0f
            : (max_abs / kE4m3Max) * scale_multiplier;
        warp_maxima[0] = scale;
        scales[static_cast<std::size_t>(k_block) * rows + row] = scale;
    }
    __syncthreads();
    const float scaled = value / warp_maxima[0];
    quantized[index] = ElementA(scaled);
}

__global__ void quantize_weight_kernel(
    const half* __restrict__ weight,
    ElementB* __restrict__ quantized,
    float* __restrict__ scales,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    float scale_multiplier) {
    const std::uint32_t n_block = blockIdx.x;
    const std::uint32_t k_block = blockIdx.y;
    __shared__ float warp_maxima[4];
    float local_max = 0.0f;
    constexpr std::uint32_t elements_per_tile = kScaleGranularity * kScaleGranularity;
    for (std::uint32_t local = threadIdx.x; local < elements_per_tile; local += blockDim.x) {
        const std::uint32_t local_k = local / kScaleGranularity;
        const std::uint32_t local_n = local % kScaleGranularity;
        const std::uint32_t k = k_block * kScaleGranularity + local_k;
        const std::uint32_t n = n_block * kScaleGranularity + local_n;
        const std::size_t index = static_cast<std::size_t>(k) * output_cols + n;
        local_max = fmaxf(local_max, fabsf(__half2float(weight[index])));
    }
    const float max_abs = block_max(local_max, warp_maxima);
    if (threadIdx.x == 0U) {
        const float scale = max_abs == 0.0f
            ? 1.0f
            : (max_abs / kE4m3Max) * scale_multiplier;
        warp_maxima[0] = scale;
        const std::uint32_t n_blocks = output_cols / kScaleGranularity;
        scales[static_cast<std::size_t>(k_block) * n_blocks + n_block] = scale;
    }
    __syncthreads();
    const float inverse_scale = 1.0f / warp_maxima[0];
    for (std::uint32_t local = threadIdx.x; local < elements_per_tile; local += blockDim.x) {
        const std::uint32_t local_k = local / kScaleGranularity;
        const std::uint32_t local_n = local % kScaleGranularity;
        const std::uint32_t k = k_block * kScaleGranularity + local_k;
        const std::uint32_t n = n_block * kScaleGranularity + local_n;
        const std::size_t index = static_cast<std::size_t>(k) * output_cols + n;
        const float scaled = __half2float(weight[index]) * inverse_scale;
        quantized[index] = ElementB(scaled);
    }
}

typename Gemm::Arguments make_arguments(
    const ElementA* input,
    const float* input_scales,
    const ElementB* weight,
    const float* weight_scales,
    ElementD* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols) {
    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(output_cols);
    const int k = static_cast<int>(input_cols);
    const auto problem_shape = cute::make_shape(m, n, k, 1);
    const StrideA stride_a = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(m, k, 1));
    const StrideB stride_b = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(n, k, 1));
    const StrideC stride_c = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(m, n, 1));
    const StrideD stride_d = cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape(m, n, 1));
    const LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    const LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {
            input,
            stride_a,
            weight,
            stride_b,
            input_scales,
            layout_sfa,
            weight_scales,
            layout_sfb,
        },
        {
            {},
            nullptr,
            stride_c,
            output,
            stride_d,
        },
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;
    return arguments;
}

#endif

}  // namespace

bool stream1_transformer_sm120_fp8_supported() {
#if BEAM_ENABLE_SM120_FP8 && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    int device = 0;
    cudaDeviceProp properties{};
    BEAM_CUDA_CHECK(cudaGetDevice(&device));
    BEAM_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return properties.major == 12 && properties.minor == 0;
#else
    return false;
#endif
}

std::size_t stream1_transformer_sm120_fp8_input_scale_elements(
    std::uint32_t rows,
    std::uint32_t input_cols) {
    if (rows == 0U || input_cols == 0U || input_cols % kScaleGranularity != 0U) {
        throw std::invalid_argument("SM120 FP8 activation shape must be rows x K with K divisible by 128");
    }
    return static_cast<std::size_t>(rows) * (input_cols / kScaleGranularity);
}

std::size_t stream1_transformer_sm120_fp8_weight_scale_elements(
    std::uint32_t input_cols,
    std::uint32_t output_cols) {
    validate_common_shape(1U, input_cols, output_cols);
    return static_cast<std::size_t>(input_cols / kScaleGranularity) *
        (output_cols / kScaleGranularity);
}

std::size_t stream1_transformer_sm120_fp8_workspace_bytes(
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols) {
    validate_common_shape(rows, input_cols, output_cols);
#if BEAM_ENABLE_SM120_FP8 && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    const auto arguments = make_arguments(
        nullptr, nullptr, nullptr, nullptr, nullptr,
        rows, input_cols, output_cols);
    return Gemm::get_workspace_size(arguments);
#else
    throw std::runtime_error("SM120 FP8 support was not compiled for sm_120a");
#endif
}

void stream1_transformer_sm120_fp8_quantize_weight_cuda(
    const half* weight,
    std::uint8_t* quantized_weight,
    float* weight_scales,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    float scale_multiplier,
    cudaStream_t stream) {
    validate_common_shape(1U, input_cols, output_cols);
    validate_scale_multiplier(scale_multiplier);
    if (weight == nullptr || quantized_weight == nullptr || weight_scales == nullptr) {
        throw std::invalid_argument("SM120 FP8 weight quantization received a null pointer");
    }
#if BEAM_ENABLE_SM120_FP8 && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    const dim3 grid(
        output_cols / kScaleGranularity,
        input_cols / kScaleGranularity);
    quantize_weight_kernel<<<grid, 128, 0, stream>>>(
        weight,
        reinterpret_cast<ElementB*>(quantized_weight),
        weight_scales,
        input_cols,
        output_cols,
        scale_multiplier);
    BEAM_CUDA_CHECK(cudaGetLastError());
#else
    (void)stream;
    throw std::runtime_error("SM120 FP8 support was not compiled for sm_120a");
#endif
}

void stream1_transformer_sm120_fp8_linear_cuda(
    const half* input,
    std::uint8_t* quantized_input,
    float* input_scales,
    const std::uint8_t* quantized_weight,
    const float* weight_scales,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    float activation_scale_multiplier,
    void* workspace,
    std::size_t workspace_bytes,
    cudaStream_t stream) {
    validate_common_shape(rows, input_cols, output_cols);
    validate_scale_multiplier(activation_scale_multiplier);
    if (input == nullptr || quantized_input == nullptr || input_scales == nullptr ||
        quantized_weight == nullptr || weight_scales == nullptr || output == nullptr) {
        throw std::invalid_argument("SM120 FP8 linear received a null tensor pointer");
    }
#if BEAM_ENABLE_SM120_FP8 && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    const std::size_t required_workspace =
        stream1_transformer_sm120_fp8_workspace_bytes(rows, input_cols, output_cols);
    if (workspace_bytes < required_workspace ||
        (required_workspace != 0U && workspace == nullptr)) {
        throw std::invalid_argument("SM120 FP8 linear workspace is smaller than required");
    }
    const dim3 quantize_grid(rows, input_cols / kScaleGranularity);
    quantize_activation_kernel<<<quantize_grid, 128, 0, stream>>>(
        input,
        reinterpret_cast<ElementA*>(quantized_input),
        input_scales,
        rows,
        input_cols,
        activation_scale_multiplier);
    BEAM_CUDA_CHECK(cudaGetLastError());

    auto arguments = make_arguments(
        reinterpret_cast<const ElementA*>(quantized_input),
        input_scales,
        reinterpret_cast<const ElementB*>(quantized_weight),
        weight_scales,
        reinterpret_cast<ElementD*>(output),
        rows,
        input_cols,
        output_cols);
    Gemm gemm;
    const cutlass::Status implementable = gemm.can_implement(arguments);
    if (implementable != cutlass::Status::kSuccess) {
        throw std::runtime_error("SM120 FP8 CUTLASS GEMM cannot implement this shape");
    }
    const cutlass::Status status = gemm(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("SM120 FP8 CUTLASS GEMM launch failed");
    }
#else
    (void)workspace;
    (void)workspace_bytes;
    (void)stream;
    throw std::runtime_error("SM120 FP8 support was not compiled for sm_120a");
#endif
}
