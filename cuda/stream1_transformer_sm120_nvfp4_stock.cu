#include "stream1_transformer_sm120_nvfp4_stock.hpp"
#include "stream1_transformer_sm120_nvfp4_ffn_policy.hpp"

#include "cuda_check.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#if BEAM_ENABLE_SM120_FP8
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#endif

namespace beam_sm120_nvfp4_stock_detail {

constexpr std::uint32_t kDModel = 256U;
constexpr std::uint32_t kFfDim = 1024U;
constexpr std::uint32_t kScaleVector = 16U;
constexpr std::uint32_t kRowsAlignment = 128U;

bool stock_profile_enabled() {
    const char* value = std::getenv("BEAM_STREAM1_SM120_NVFP4_STOCK_PROFILE");
    return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
}

std::uint32_t stock_profile_skip_calls() {
    const char* value = std::getenv(
        "BEAM_STREAM1_SM120_NVFP4_STOCK_PROFILE_SKIP_CALLS");
    if (value == nullptr || value[0] == '\0') {
        return 1U;
    }
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' ||
        parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(
            "BEAM_STREAM1_SM120_NVFP4_STOCK_PROFILE_SKIP_CALLS must be a uint32");
    }
    return static_cast<std::uint32_t>(parsed);
}

class StockFfnProfiler {
public:
    explicit StockFfnProfiler(cudaStream_t stream) : stream_(stream) {
        if (!stock_profile_enabled()) {
            return;
        }
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        BEAM_CUDA_CHECK(cudaStreamIsCapturing(stream_, &capture_status));
        if (capture_status != cudaStreamCaptureStatusNone) {
            throw std::invalid_argument(
                "BEAM_STREAM1_SM120_NVFP4_STOCK_PROFILE requires eager execution");
        }
        static std::atomic<std::uint32_t> call_count{0U};
        const std::uint32_t call_index =
            call_count.fetch_add(1U, std::memory_order_relaxed);
        if (call_index < stock_profile_skip_calls()) {
            return;
        }
        static std::atomic<bool> claimed{false};
        if (claimed.exchange(true, std::memory_order_relaxed)) {
            return;
        }
        enabled_ = true;
        BEAM_CUDA_CHECK(cudaEventCreate(&start_));
        BEAM_CUDA_CHECK(cudaEventCreate(&after_quantize_));
        BEAM_CUDA_CHECK(cudaEventCreate(&after_ff1_));
        BEAM_CUDA_CHECK(cudaEventCreate(&after_ff2_));
        BEAM_CUDA_CHECK(cudaEventRecord(start_, stream_));
    }

    StockFfnProfiler(const StockFfnProfiler&) = delete;
    StockFfnProfiler& operator=(const StockFfnProfiler&) = delete;

    ~StockFfnProfiler() {
        destroy(after_ff2_);
        destroy(after_ff1_);
        destroy(after_quantize_);
        destroy(start_);
    }

    void after_quantize() { record(after_quantize_); }
    void after_ff1() { record(after_ff1_); }

    void finish(std::uint32_t rows, const char* input_stage) {
        if (!enabled_) {
            return;
        }
        record(after_ff2_);
        BEAM_CUDA_CHECK(cudaEventSynchronize(after_ff2_));
        const float quantize_ms = elapsed(start_, after_quantize_);
        const float ff1_ms = elapsed(after_quantize_, after_ff1_);
        const float ff2_ms = elapsed(after_ff1_, after_ff2_);
        std::cout << "stream1_sm120_nvfp4_stock_profile"
                  << " rows=" << rows
                  << " input_stage=" << input_stage
                  << " input_stage_ms=" << quantize_ms
                  << " ff1_ms=" << ff1_ms
                  << " ff2_ms=" << ff2_ms
                  << " total_ms=" << (quantize_ms + ff1_ms + ff2_ms)
                  << "\n";
        std::cout.flush();
    }

private:
    void record(cudaEvent_t event) {
        if (enabled_) {
            BEAM_CUDA_CHECK(cudaEventRecord(event, stream_));
        }
    }

    static float elapsed(cudaEvent_t begin, cudaEvent_t end) {
        float milliseconds = 0.0F;
        BEAM_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
        return milliseconds;
    }

    static void destroy(cudaEvent_t event) {
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }

    cudaStream_t stream_ = nullptr;
    bool enabled_ = false;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t after_quantize_ = nullptr;
    cudaEvent_t after_ff1_ = nullptr;
    cudaEvent_t after_ff2_ = nullptr;
};

void validate_stock_arguments(
    const half* normalized_input_fp16,
    const half* residual_input_fp16,
    const std::uint8_t* ff1_weight_e2m1,
    const std::uint8_t* ff1_weight_ue4m3,
    const half* ff1_bias_fp16,
    const std::uint8_t* ff2_weight_e2m1,
    const std::uint8_t* ff2_weight_ue4m3,
    half* residual_output_fp16,
    void* input_scratch,
    std::size_t input_scratch_bytes,
    void* hidden_scratch,
    std::size_t hidden_scratch_bytes,
    std::uint32_t rows) {
    if (rows == 0U || rows % kRowsAlignment != 0U) {
        throw std::invalid_argument(
            "SM120 stock NVFP4 FFN rows must be a non-zero multiple of 128");
    }
    if (normalized_input_fp16 == nullptr || residual_input_fp16 == nullptr ||
        ff1_weight_e2m1 == nullptr || ff1_weight_ue4m3 == nullptr ||
        ff1_bias_fp16 == nullptr || ff2_weight_e2m1 == nullptr ||
        ff2_weight_ue4m3 == nullptr || residual_output_fp16 == nullptr ||
        input_scratch == nullptr || hidden_scratch == nullptr) {
        throw std::invalid_argument("SM120 stock NVFP4 FFN received a null tensor pointer");
    }
    const auto plan = stream1_sm120_nvfp4_stock_scratch_plan(rows);
    if (input_scratch_bytes < plan.input_total_bytes ||
        hidden_scratch_bytes < plan.hidden_total_bytes) {
        throw std::invalid_argument(
            "SM120 stock NVFP4 FFN scratch is smaller than the exact CUTLASS layout");
    }
    const auto aligned = [](const void* pointer) {
        return reinterpret_cast<std::uintptr_t>(pointer) % 128U == 0U;
    };
    if (!aligned(input_scratch) || !aligned(hidden_scratch) ||
        !aligned(ff1_weight_e2m1) || !aligned(ff1_weight_ue4m3) ||
        !aligned(ff2_weight_e2m1) || !aligned(ff2_weight_ue4m3)) {
        throw std::invalid_argument(
            "SM120 stock NVFP4 FFN requires 128-byte aligned values and scales");
    }
}

#if BEAM_ENABLE_SM120_FP8 && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementScale = cutlass::float_ue4m3_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementBias = cutlass::half_t;
using ElementResidual = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using LayoutSFD = cutlass::layout::RowMajor;
using ProblemShape = Shape<int, int, int, int>;
static_assert(
    Stream1Sm120Nvfp4FfnPolicy::kStockMainloopKTile == 128U,
    "stock NVFP4 mainloop specialization must match the measured K128 policy");
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;
using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVector>;
using ScaleLayoutSFA = decltype(
    ScaleConfig::tile_atom_to_shape_SFA(ProblemShape{}));

constexpr int kAlignmentA = 32;
constexpr int kAlignmentB = 32;
constexpr int kAlignmentResidual = 128 / cutlass::sizeof_bits<ElementResidual>::value;
constexpr int kAlignmentNvfp4 = 128 / cutlass::sizeof_bits<cutlass::float_e2m1_t>::value;

using Ff1Fusion = cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
    cutlass::epilogue::thread::ReLU,
    kScaleVector,
    cutlass::float_e2m1_t,
    ElementCompute,
    ElementScale,
    LayoutSFD,
    ElementBias,
    ElementResidual>;

using Ff1Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementResidual,
    LayoutC,
    1,
    cutlass::float_e2m1_t,
    LayoutD,
    kAlignmentNvfp4,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    Ff1Fusion>::CollectiveOp;

using Ff1Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Ff1Epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120>::CollectiveOp;

using Ff1Kernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, Ff1Mainloop, Ff1Epilogue, void>;
using Ff1Gemm = cutlass::gemm::device::GemmUniversalAdapter<Ff1Kernel>;

using Ff2Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementResidual,
    LayoutC,
    kAlignmentResidual,
    ElementResidual,
    LayoutD,
    kAlignmentResidual,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using Ff2Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Ff2Epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120>::CollectiveOp;

using Ff2Kernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, Ff2Mainloop, Ff2Epilogue, void>;
using Ff2Gemm = cutlass::gemm::device::GemmUniversalAdapter<Ff2Kernel>;

using QkvFusion = cutlass::epilogue::fusion::LinCombPerColBias<
    ElementResidual, ElementCompute, ElementBias, ElementResidual>;
using QkvEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator,
    ElementCompute,
    ElementResidual,
    LayoutC,
    kAlignmentResidual,
    ElementResidual,
    LayoutD,
    kAlignmentResidual,
    cutlass::epilogue::collective::EpilogueScheduleAuto,
    QkvFusion>::CollectiveOp;
using QkvMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename QkvEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120>::CollectiveOp;
using QkvKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, QkvMainloop, QkvEpilogue, void>;
using QkvGemm = cutlass::gemm::device::GemmUniversalAdapter<QkvKernel>;

__device__ float stream1_sm120_nvfp4_norm_constant = 1.0F;

__global__ void quantize_fp16_activation_nvfp4_kernel(
    const half* __restrict__ input,
    std::uint8_t* __restrict__ packed_values,
    ElementScale* __restrict__ physical_scales,
    ScaleLayoutSFA scale_layout,
    std::uint32_t rows,
    std::uint32_t columns) {
    constexpr std::uint32_t warps_per_block = 8U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t k16_groups = columns / kScaleVector;
    const std::uint64_t tile =
        static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    const std::uint64_t tile_count =
        static_cast<std::uint64_t>(rows) * k16_groups;
    if (tile >= tile_count) {
        return;
    }
    const std::uint32_t row = static_cast<std::uint32_t>(tile / k16_groups);
    const std::uint32_t group = static_cast<std::uint32_t>(tile % k16_groups);
    const std::uint32_t k_begin = group * kScaleVector;
    float max_abs = lane < kScaleVector
        ? fabsf(__half2float(input[static_cast<std::size_t>(row) * columns + k_begin + lane]))
        : 0.0F;
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffU, max_abs, offset));
    }
    float decoded_scale = 1.0F;
    if (lane == 0U) {
        const ElementScale encoded_scale(max_abs == 0.0F ? 1.0F : max_abs / 6.0F);
        decoded_scale = static_cast<float>(encoded_scale);
        auto scales = cute::make_tensor(physical_scales, scale_layout);
        scales(static_cast<int>(row), static_cast<int>(k_begin), 0) = encoded_scale;
    }
    decoded_scale = __shfl_sync(0xffffffffU, decoded_scale, 0);
    if (lane < 8U) {
        const std::size_t logical =
            static_cast<std::size_t>(row) * columns + k_begin + 2U * lane;
        const float first = __half2float(input[logical]) / decoded_scale;
        const float second = __half2float(input[logical + 1U]) / decoded_scale;
        const std::uint8_t first_raw = static_cast<std::uint8_t>(
            cutlass::float_e2m1_t(first).raw() & 0x0fU);
        const std::uint8_t second_raw = static_cast<std::uint8_t>(
            cutlass::float_e2m1_t(second).raw() & 0x0fU);
        packed_values[logical / 2U] = static_cast<std::uint8_t>(
            first_raw | (second_raw << 4U));
    }
}

__device__ __forceinline__ float stock_warp_reduce_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    return value;
}

template <bool ApplyResidualBiasAndStore, bool RegisterPack = false>
__global__ void bias_layernorm_quantize_fp16_activation_nvfp4_kernel(
    half* __restrict__ residual_input_output,
    const half* __restrict__ residual_bias,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    std::uint8_t* __restrict__ packed_values,
    ElementScale* __restrict__ physical_scales,
    ScaleLayoutSFA scale_layout,
    std::uint32_t rows) {
    constexpr std::uint32_t threads =
        Stream1Sm120Nvfp4FfnPolicy::kFusedLn2Threads;
    constexpr std::uint32_t values_per_thread =
        Stream1Sm120Nvfp4FfnPolicy::kFusedLn2ValuesPerThread;
    constexpr std::uint32_t threads_per_scale =
        Stream1Sm120Nvfp4FfnPolicy::kFusedLn2ThreadsPerScale;
    static_assert(threads == 128U && values_per_thread == 2U);
    static_assert(threads_per_scale == 8U);
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (row >= rows || tid >= threads) {
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    auto* rounded = reinterpret_cast<half*>(shared_bytes);
    auto* warp_scratch = reinterpret_cast<float*>(
        shared_bytes + kDModel * sizeof(half));
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint64_t base = static_cast<std::uint64_t>(row) * kDModel;

    // Preserve the exact production LayerNorm reduction mapping: each thread
    // owns columns tid and tid+128.  The rounded normalized values are staged
    // in shared memory only so the pack phase can regroup adjacent pairs.
    const std::uint32_t col0 = RegisterPack ? 2U * tid : tid;
    const std::uint32_t col1 = RegisterPack ? col0 + 1U : tid + threads;
    float x0 = __half2float(residual_input_output[base + col0]);
    float x1 = __half2float(residual_input_output[base + col1]);
    if constexpr (ApplyResidualBiasAndStore) {
        x0 += __half2float(residual_bias[col0]);
        x1 += __half2float(residual_bias[col1]);
        residual_input_output[base + col0] = __float2half_rn(x0);
        residual_input_output[base + col1] = __float2half_rn(x1);
    }

    const float warp_sum = stock_warp_reduce_sum(x0 + x1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_sum;
    }
    __syncthreads();
    float block_sum = tid < 4U ? warp_scratch[tid] : 0.0F;
    if (warp == 0U) {
        block_sum = stock_warp_reduce_sum(block_sum);
        if (lane == 0U) {
            warp_scratch[4] = block_sum * (1.0F / 256.0F);
        }
    }
    __syncthreads();
    const float mean = warp_scratch[4];

    const float centered0 = x0 - mean;
    const float centered1 = x1 - mean;
    const float warp_var_sum = stock_warp_reduce_sum(
        centered0 * centered0 + centered1 * centered1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_var_sum;
    }
    __syncthreads();
    float block_var_sum = tid < 4U ? warp_scratch[tid] : 0.0F;
    if (warp == 0U) {
        block_var_sum = stock_warp_reduce_sum(block_var_sum);
        if (lane == 0U) {
            warp_scratch[5] = rsqrtf(
                block_var_sum * (1.0F / 256.0F) + 1.0e-5F);
        }
    }
    __syncthreads();
    const float inv_std = warp_scratch[5];
    const float y0 = centered0 * inv_std * __half2float(gamma[col0]) +
        __half2float(beta[col0]);
    const float y1 = centered1 * inv_std * __half2float(gamma[col1]) +
        __half2float(beta[col1]);
    if constexpr (RegisterPack) {
        const float rounded0 = __half2float(__float2half_rn(y0));
        const float rounded1 = __half2float(__float2half_rn(y1));
        float max_abs = fmaxf(fabsf(rounded0), fabsf(rounded1));
        for (int offset = 4; offset > 0; offset >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffU, max_abs, offset, 8));
        }
        float decoded_scale = 1.0F;
        if ((lane & 7U) == 0U) {
            const ElementScale encoded_scale(max_abs == 0.0F ? 1.0F : max_abs / 6.0F);
            decoded_scale = static_cast<float>(encoded_scale);
            auto scales = cute::make_tensor(physical_scales, scale_layout);
            scales(static_cast<int>(row), static_cast<int>(col0), 0) = encoded_scale;
        }
        decoded_scale = __shfl_sync(0xffffffffU, decoded_scale, 0, 8);
        const std::uint8_t raw0 = static_cast<std::uint8_t>(cutlass::float_e2m1_t(rounded0 / decoded_scale).raw() & 0x0fU);
        const std::uint8_t raw1 = static_cast<std::uint8_t>(cutlass::float_e2m1_t(rounded1 / decoded_scale).raw() & 0x0fU);
        packed_values[(base + col0) / 2U] = static_cast<std::uint8_t>(raw0 | (raw1 << 4U));
    } else {
        rounded[col0] = __float2half_rn(y0);
        rounded[col1] = __float2half_rn(y1);
        __syncthreads();
        const std::uint32_t packed_col = tid * values_per_thread;
        const float first = __half2float(rounded[packed_col]);
        const float second = __half2float(rounded[packed_col + 1U]);
        float max_abs = fmaxf(fabsf(first), fabsf(second));
        for (int offset = 4; offset > 0; offset >>= 1) max_abs = fmaxf(max_abs, __shfl_down_sync(0xffffffffU, max_abs, offset, threads_per_scale));
        float decoded_scale = 1.0F;
        if ((tid & (threads_per_scale - 1U)) == 0U) {
            const ElementScale encoded_scale(max_abs == 0.0F ? 1.0F : max_abs / 6.0F);
            decoded_scale = static_cast<float>(encoded_scale);
            auto scales = cute::make_tensor(physical_scales, scale_layout);
            scales(static_cast<int>(row), static_cast<int>(packed_col), 0) = encoded_scale;
        }
        decoded_scale = __shfl_sync(0xffffffffU, decoded_scale, 0, threads_per_scale);
        const std::uint8_t first_raw = static_cast<std::uint8_t>(cutlass::float_e2m1_t(first / decoded_scale).raw() & 0x0fU);
        const std::uint8_t second_raw = static_cast<std::uint8_t>(cutlass::float_e2m1_t(second / decoded_scale).raw() & 0x0fU);
        packed_values[(base + packed_col) / 2U] = static_cast<std::uint8_t>(first_raw | (second_raw << 4U));
    }
}

float* norm_constant_device_pointer() {
    static int cached_device = -1;
    static float* cached_pointer = nullptr;
    int device = 0;
    BEAM_CUDA_CHECK(cudaGetDevice(&device));
    if (cached_pointer == nullptr || cached_device != device) {
        void* pointer = nullptr;
        BEAM_CUDA_CHECK(cudaGetSymbolAddress(
            &pointer, stream1_sm120_nvfp4_norm_constant));
        cached_pointer = static_cast<float*>(pointer);
        cached_device = device;
    }
    return cached_pointer;
}

typename Ff1Gemm::Arguments make_ff1_arguments(
    const std::uint8_t* input_values,
    const std::uint8_t* input_scales,
    const std::uint8_t* weight_values,
    const std::uint8_t* weight_scales,
    const half* bias,
    std::uint8_t* hidden_values,
    std::uint8_t* hidden_scales,
    std::uint32_t rows) {
    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(kFfDim);
    const int k = static_cast<int>(kDModel);
    const ProblemShape problem{m, n, k, 1};
    using StrideA = typename Ff1Kernel::StrideA;
    using StrideB = typename Ff1Kernel::StrideB;
    using StrideC = typename Ff1Kernel::StrideC;
    using StrideD = typename Ff1Kernel::StrideD;
    const StrideA stride_a = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(m, k, 1));
    const StrideB stride_b = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(n, k, 1));
    const StrideC stride_c = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(m, n, 1));
    const StrideD stride_d = cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape(m, n, 1));
    const auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem);
    const auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem);
    typename Ff1Kernel::MainloopArguments mainloop{
        reinterpret_cast<const typename Ff1Kernel::CollectiveMainloop::ArrayElementA*>(input_values),
        stride_a,
        reinterpret_cast<const typename Ff1Kernel::CollectiveMainloop::ArrayElementB*>(weight_values),
        stride_b,
        reinterpret_cast<const ElementScale*>(input_scales),
        layout_sfa,
        reinterpret_cast<const ElementScale*>(weight_scales),
        layout_sfb};
    typename Ff1Kernel::EpilogueArguments epilogue{
        {},
        nullptr,
        stride_c,
        reinterpret_cast<cutlass::float_e2m1_t*>(hidden_values),
        stride_d};
    epilogue.thread.alpha = 1.0F;
    epilogue.thread.beta = 0.0F;
    epilogue.thread.bias_ptr = reinterpret_cast<const ElementBias*>(bias);
    epilogue.thread.block_scale_factor_ptr =
        reinterpret_cast<ElementScale*>(hidden_scales);
    epilogue.thread.norm_constant_ptr = norm_constant_device_pointer();
    cutlass::KernelHardwareInfo hardware{};
    hardware.device_id = 0;
    hardware.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename Ff1Kernel::TileScheduler::Arguments scheduler{};
    return {
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        mainloop,
        epilogue,
        hardware,
        scheduler};
}

typename Ff2Gemm::Arguments make_ff2_arguments(
    const std::uint8_t* hidden_values,
    const std::uint8_t* hidden_scales,
    const std::uint8_t* weight_values,
    const std::uint8_t* weight_scales,
    const half* residual_input,
    half* residual_output,
    std::uint32_t rows) {
    const int m = static_cast<int>(rows);
    const int n = static_cast<int>(kDModel);
    const int k = static_cast<int>(kFfDim);
    const ProblemShape problem{m, n, k, 1};
    using StrideA = typename Ff2Kernel::StrideA;
    using StrideB = typename Ff2Kernel::StrideB;
    using StrideC = typename Ff2Kernel::StrideC;
    using StrideD = typename Ff2Kernel::StrideD;
    const StrideA stride_a = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(m, k, 1));
    const StrideB stride_b = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(n, k, 1));
    const StrideC stride_c = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(m, n, 1));
    const StrideD stride_d = cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape(m, n, 1));
    const auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem);
    const auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem);
    typename Ff2Kernel::MainloopArguments mainloop{
        reinterpret_cast<const typename Ff2Kernel::CollectiveMainloop::ArrayElementA*>(hidden_values),
        stride_a,
        reinterpret_cast<const typename Ff2Kernel::CollectiveMainloop::ArrayElementB*>(weight_values),
        stride_b,
        reinterpret_cast<const ElementScale*>(hidden_scales),
        layout_sfa,
        reinterpret_cast<const ElementScale*>(weight_scales),
        layout_sfb};
    typename Ff2Kernel::EpilogueArguments epilogue{
        {},
        reinterpret_cast<const ElementResidual*>(residual_input),
        stride_c,
        reinterpret_cast<ElementResidual*>(residual_output),
        stride_d};
    epilogue.thread.alpha = 1.0F;
    epilogue.thread.beta = 1.0F;
    cutlass::KernelHardwareInfo hardware{};
    hardware.device_id = 0;
    hardware.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename Ff2Kernel::TileScheduler::Arguments scheduler{};
    return {
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        mainloop,
        epilogue,
        hardware,
        scheduler};
}

void launch_prequantized_stock_ffn(
    const std::uint8_t* input_values,
    const std::uint8_t* input_scales,
    const half* residual_input_fp16,
    const std::uint8_t* ff1_weight_e2m1,
    const std::uint8_t* ff1_weight_ue4m3,
    const half* ff1_bias_fp16,
    const std::uint8_t* ff2_weight_e2m1,
    const std::uint8_t* ff2_weight_ue4m3,
    half* residual_output_fp16,
    std::uint8_t* hidden_values,
    std::uint8_t* hidden_scales,
    std::uint32_t rows,
    cudaStream_t stream,
    StockFfnProfiler& profiler,
    const char* input_stage) {
    auto ff1_arguments = make_ff1_arguments(
        input_values,
        input_scales,
        ff1_weight_e2m1,
        ff1_weight_ue4m3,
        ff1_bias_fp16,
        hidden_values,
        hidden_scales,
        rows);
    auto ff2_arguments = make_ff2_arguments(
        hidden_values,
        hidden_scales,
        ff2_weight_e2m1,
        ff2_weight_ue4m3,
        residual_input_fp16,
        residual_output_fp16,
        rows);
    const std::size_t ff1_workspace = Ff1Gemm::get_workspace_size(ff1_arguments);
    const std::size_t ff2_workspace = Ff2Gemm::get_workspace_size(ff2_arguments);
    if (ff1_workspace != 0U || ff2_workspace != 0U) {
        throw std::runtime_error(
            "SM120 stock NVFP4 FFN unexpectedly requires CUTLASS workspace");
    }
    Ff1Gemm ff1;
    Ff2Gemm ff2;
    if (ff1.can_implement(ff1_arguments) != cutlass::Status::kSuccess ||
        ff2.can_implement(ff2_arguments) != cutlass::Status::kSuccess) {
        throw std::runtime_error(
            "SM120 stock NVFP4 FFN cannot implement the production Cube4 shape");
    }
    if (ff1(ff1_arguments, nullptr, stream) != cutlass::Status::kSuccess) {
        throw std::runtime_error("SM120 stock NVFP4 FF1 launch failed");
    }
    profiler.after_ff1();
    if (ff2(ff2_arguments, nullptr, stream) != cutlass::Status::kSuccess) {
        throw std::runtime_error("SM120 stock NVFP4 FF2 launch failed");
    }
    profiler.finish(rows, input_stage);
}

#endif

}  // namespace beam_sm120_nvfp4_stock_detail

using namespace beam_sm120_nvfp4_stock_detail;

bool stream1_transformer_sm120_nvfp4_stock_supported() {
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

void stream1_transformer_sm120_nvfp4_stock_ffn_cuda(
    const half* normalized_input_fp16,
    const half* residual_input_fp16,
    const std::uint8_t* ff1_weight_e2m1,
    const std::uint8_t* ff1_weight_ue4m3,
    const half* ff1_bias_fp16,
    const std::uint8_t* ff2_weight_e2m1,
    const std::uint8_t* ff2_weight_ue4m3,
    half* residual_output_fp16,
    void* input_scratch,
    std::size_t input_scratch_bytes,
    void* hidden_scratch,
    std::size_t hidden_scratch_bytes,
    std::uint32_t rows,
    cudaStream_t stream) {
    validate_stock_arguments(
        normalized_input_fp16,
        residual_input_fp16,
        ff1_weight_e2m1,
        ff1_weight_ue4m3,
        ff1_bias_fp16,
        ff2_weight_e2m1,
        ff2_weight_ue4m3,
        residual_output_fp16,
        input_scratch,
        input_scratch_bytes,
        hidden_scratch,
        hidden_scratch_bytes,
        rows);
#if BEAM_ENABLE_SM120_FP8 && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    if (!stream1_transformer_sm120_nvfp4_stock_supported()) {
        throw std::runtime_error("SM120 stock NVFP4 FFN requires an sm_120 device");
    }
    const auto plan = stream1_sm120_nvfp4_stock_scratch_plan(rows);
    auto* input_bytes = static_cast<std::uint8_t*>(input_scratch);
    auto* hidden_bytes = static_cast<std::uint8_t*>(hidden_scratch);
    auto* input_values = input_bytes + plan.input_value_offset;
    auto* input_scales = input_bytes + plan.input_scale_offset;
    auto* hidden_values = hidden_bytes + plan.hidden_value_offset;
    auto* hidden_scales = hidden_bytes + plan.hidden_scale_offset;
    StockFfnProfiler profiler(stream);

    const ProblemShape input_problem{
        static_cast<int>(rows), static_cast<int>(kFfDim),
        static_cast<int>(kDModel), 1};
    const auto input_scale_layout = ScaleConfig::tile_atom_to_shape_SFA(input_problem);
    constexpr std::uint32_t warps_per_block = 8U;
    const std::uint64_t quantize_warps =
        static_cast<std::uint64_t>(rows) * (kDModel / kScaleVector);
    const std::uint32_t quantize_blocks = static_cast<std::uint32_t>(
        (quantize_warps + warps_per_block - 1U) / warps_per_block);
    quantize_fp16_activation_nvfp4_kernel<<<quantize_blocks, 256, 0, stream>>>(
        normalized_input_fp16,
        input_values,
        reinterpret_cast<ElementScale*>(input_scales),
        input_scale_layout,
        rows,
        kDModel);
    BEAM_CUDA_CHECK(cudaGetLastError());
    profiler.after_quantize();

    launch_prequantized_stock_ffn(
        input_values,
        input_scales,
        residual_input_fp16,
        ff1_weight_e2m1,
        ff1_weight_ue4m3,
        ff1_bias_fp16,
        ff2_weight_e2m1,
        ff2_weight_ue4m3,
        residual_output_fp16,
        hidden_values,
        hidden_scales,
        rows,
        stream,
        profiler,
        "fp16_quantize");
#else
    (void)stream;
    throw std::runtime_error("SM120 stock NVFP4 FFN was not compiled for sm_120a");
#endif
}

void stream1_transformer_sm120_nvfp4_stock_bias_layernorm_ffn_cuda(
    half* residual_input_output_fp16,
    const half* residual_bias_fp16,
    const half* layernorm_gamma_fp16,
    const half* layernorm_beta_fp16,
    const std::uint8_t* ff1_weight_e2m1,
    const std::uint8_t* ff1_weight_ue4m3,
    const half* ff1_bias_fp16,
    const std::uint8_t* ff2_weight_e2m1,
    const std::uint8_t* ff2_weight_ue4m3,
    void* input_scratch,
    std::size_t input_scratch_bytes,
    void* hidden_scratch,
    std::size_t hidden_scratch_bytes,
    std::uint32_t rows,
    cudaStream_t stream) {
    validate_stock_arguments(
        residual_input_output_fp16,
        residual_input_output_fp16,
        ff1_weight_e2m1,
        ff1_weight_ue4m3,
        ff1_bias_fp16,
        ff2_weight_e2m1,
        ff2_weight_ue4m3,
        residual_input_output_fp16,
        input_scratch,
        input_scratch_bytes,
        hidden_scratch,
        hidden_scratch_bytes,
        rows);
    if (residual_bias_fp16 == nullptr || layernorm_gamma_fp16 == nullptr ||
        layernorm_beta_fp16 == nullptr) {
        throw std::invalid_argument(
            "SM120 fused LN2 NVFP4 producer received a null parameter tensor");
    }
#if BEAM_ENABLE_SM120_FP8 && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    if (!stream1_transformer_sm120_nvfp4_stock_supported()) {
        throw std::runtime_error("SM120 stock NVFP4 FFN requires an sm_120 device");
    }
    const auto plan = stream1_sm120_nvfp4_stock_scratch_plan(rows);
    auto* input_bytes = static_cast<std::uint8_t*>(input_scratch);
    auto* hidden_bytes = static_cast<std::uint8_t*>(hidden_scratch);
    auto* input_values = input_bytes + plan.input_value_offset;
    auto* input_scales = input_bytes + plan.input_scale_offset;
    auto* hidden_values = hidden_bytes + plan.hidden_value_offset;
    auto* hidden_scales = hidden_bytes + plan.hidden_scale_offset;
    StockFfnProfiler profiler(stream);

    const ProblemShape input_problem{
        static_cast<int>(rows), static_cast<int>(kFfDim),
        static_cast<int>(kDModel), 1};
    const auto input_scale_layout = ScaleConfig::tile_atom_to_shape_SFA(input_problem);
    constexpr std::size_t shared_bytes =
        kDModel * sizeof(half) + 6U * sizeof(float);
    bias_layernorm_quantize_fp16_activation_nvfp4_kernel<true, false><<<
        rows,
        Stream1Sm120Nvfp4FfnPolicy::kFusedLn2Threads,
        shared_bytes,
        stream>>>(
        residual_input_output_fp16,
        residual_bias_fp16,
        layernorm_gamma_fp16,
        layernorm_beta_fp16,
        input_values,
        reinterpret_cast<ElementScale*>(input_scales),
        input_scale_layout,
        rows);
    BEAM_CUDA_CHECK(cudaGetLastError());
    profiler.after_quantize();

    launch_prequantized_stock_ffn(
        input_values,
        input_scales,
        residual_input_output_fp16,
        ff1_weight_e2m1,
        ff1_weight_ue4m3,
        ff1_bias_fp16,
        ff2_weight_e2m1,
        ff2_weight_ue4m3,
        residual_input_output_fp16,
        hidden_values,
        hidden_scales,
        rows,
        stream,
        profiler,
        "fused_ln2_quantize");
#else
    (void)stream;
    throw std::runtime_error("SM120 stock NVFP4 FFN was not compiled for sm_120a");
#endif
}

void stream1_transformer_sm120_nvfp4_stock_layernorm_qkv_cuda(
    half* residual_input_output_fp16,
    const half* residual_bias_fp16,
    const half* layernorm_gamma_fp16,
    const half* layernorm_beta_fp16,
    const std::uint8_t* qkv_weight_e2m1,
    const std::uint8_t* qkv_weight_ue4m3,
    const half* qkv_bias_fp16,
    half* qkv_output_fp16,
    void* input_scratch,
    std::size_t input_scratch_bytes,
    std::uint32_t rows,
    cudaStream_t stream) {
#if BEAM_ENABLE_SM120_FP8 && defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    if (!stream1_transformer_sm120_nvfp4_stock_supported()) {
        throw std::runtime_error("SM120 stock NVFP4 QKV requires an sm_120 device");
    }
    if (rows == 0U || rows % kRowsAlignment != 0U) {
        throw std::invalid_argument("SM120 stock NVFP4 QKV rows must be a non-zero multiple of 128");
    }
    if (residual_input_output_fp16 == nullptr || layernorm_gamma_fp16 == nullptr ||
        layernorm_beta_fp16 == nullptr || qkv_weight_e2m1 == nullptr ||
        qkv_weight_ue4m3 == nullptr || qkv_bias_fp16 == nullptr ||
        qkv_output_fp16 == nullptr || input_scratch == nullptr) {
        throw std::invalid_argument("SM120 stock NVFP4 QKV received a null required tensor pointer");
    }
    const auto plan = stream1_sm120_nvfp4_stock_scratch_plan(rows);
    if (input_scratch_bytes < plan.input_total_bytes) {
        throw std::invalid_argument("SM120 stock NVFP4 QKV scratch is smaller than the exact CUTLASS input layout");
    }
    auto* scratch_bytes = static_cast<std::uint8_t*>(input_scratch);
    auto* input_values = scratch_bytes + plan.input_value_offset;
    auto* input_scales = scratch_bytes + plan.input_scale_offset;
    const ProblemShape problem{
        static_cast<int>(rows), 3 * static_cast<int>(kDModel),
        static_cast<int>(kDModel), 1};
    const auto scale_layout = ScaleConfig::tile_atom_to_shape_SFA(problem);
    constexpr std::size_t shared_bytes = kDModel * sizeof(half) + 6U * sizeof(float);
    if (residual_bias_fp16 == nullptr) {
        bias_layernorm_quantize_fp16_activation_nvfp4_kernel<false, true><<<
            rows, Stream1Sm120Nvfp4FfnPolicy::kFusedLn2Threads,
            shared_bytes, stream>>>(
                residual_input_output_fp16, residual_input_output_fp16,
                layernorm_gamma_fp16, layernorm_beta_fp16,
                input_values, reinterpret_cast<ElementScale*>(input_scales),
                scale_layout, rows);
    } else {
        bias_layernorm_quantize_fp16_activation_nvfp4_kernel<true, true><<<
            rows, Stream1Sm120Nvfp4FfnPolicy::kFusedLn2Threads,
            shared_bytes, stream>>>(
                residual_input_output_fp16, residual_bias_fp16,
                layernorm_gamma_fp16, layernorm_beta_fp16,
                input_values, reinterpret_cast<ElementScale*>(input_scales),
                scale_layout, rows);
    }
    BEAM_CUDA_CHECK(cudaGetLastError());

    using StrideA = typename QkvKernel::StrideA;
    using StrideB = typename QkvKernel::StrideB;
    using StrideC = typename QkvKernel::StrideC;
    using StrideD = typename QkvKernel::StrideD;
    const auto stride_a = cutlass::make_cute_packed_stride(
        StrideA{}, cute::make_shape(static_cast<int>(rows), static_cast<int>(kDModel), 1));
    const auto stride_b = cutlass::make_cute_packed_stride(
        StrideB{}, cute::make_shape(3 * static_cast<int>(kDModel), static_cast<int>(kDModel), 1));
    const auto stride_c = cutlass::make_cute_packed_stride(
        StrideC{}, cute::make_shape(static_cast<int>(rows), 3 * static_cast<int>(kDModel), 1));
    const auto stride_d = cutlass::make_cute_packed_stride(
        StrideD{}, cute::make_shape(static_cast<int>(rows), 3 * static_cast<int>(kDModel), 1));
    typename QkvKernel::MainloopArguments mainloop{
        reinterpret_cast<const typename QkvKernel::CollectiveMainloop::ArrayElementA*>(input_values),
        stride_a,
        reinterpret_cast<const typename QkvKernel::CollectiveMainloop::ArrayElementB*>(qkv_weight_e2m1),
        stride_b,
        reinterpret_cast<const ElementScale*>(input_scales),
        ScaleConfig::tile_atom_to_shape_SFA(problem),
        reinterpret_cast<const ElementScale*>(qkv_weight_ue4m3),
        ScaleConfig::tile_atom_to_shape_SFB(problem)};
    typename QkvKernel::EpilogueArguments epilogue{
        {}, nullptr, stride_c,
        reinterpret_cast<ElementResidual*>(qkv_output_fp16), stride_d};
    epilogue.thread.alpha = 1.0F;
    epilogue.thread.beta = 0.0F;
    epilogue.thread.bias_ptr = reinterpret_cast<const ElementBias*>(qkv_bias_fp16);
    cutlass::KernelHardwareInfo hardware{};
    hardware.device_id = 0;
    hardware.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    typename QkvKernel::TileScheduler::Arguments scheduler{};
    typename QkvGemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm, problem, mainloop,
        epilogue, hardware, scheduler};
    if (QkvGemm::get_workspace_size(arguments) != 0U) {
        throw std::runtime_error("SM120 stock NVFP4 QKV unexpectedly requires CUTLASS workspace");
    }
    QkvGemm gemm;
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess ||
        gemm(arguments, nullptr, stream) != cutlass::Status::kSuccess) {
        throw std::runtime_error("SM120 stock NVFP4 QKV launch failed");
    }
#else
    (void)residual_input_output_fp16; (void)residual_bias_fp16;
    (void)layernorm_gamma_fp16; (void)layernorm_beta_fp16;
    (void)qkv_weight_e2m1; (void)qkv_weight_ue4m3;
    (void)qkv_bias_fp16; (void)qkv_output_fp16;
    (void)input_scratch; (void)input_scratch_bytes; (void)rows; (void)stream;
    throw std::runtime_error("SM120 stock NVFP4 QKV was not compiled for sm_120a");
#endif
}
