#include "cuda_check.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace sm120_mxfp8_bench {

using namespace cute;

using ElementAPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementBPair = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using ElementA = ElementAPair::DataType;
using ElementB = ElementBPair::DataType;
using ElementSFA = ElementAPair::ScaleFactorType;
using ElementSFB = ElementBPair::ScaleFactorType;
using ElementD = cutlass::half_t;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float,
    float,
    void,
    LayoutDTag,
    AlignmentD,
    ElementD,
    LayoutDTag,
    AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledTensorOp,
    ElementAPair,
    LayoutATag,
    AlignmentA,
    ElementBPair,
    LayoutBTag,
    AlignmentB,
    float,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename GemmKernel::StrideA;
using StrideB = typename GemmKernel::StrideB;
using StrideC = typename GemmKernel::StrideC;
using StrideD = typename GemmKernel::StrideD;
using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;
using ScaleConfig = typename GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

constexpr std::uint32_t kScaleVector = 32U;
constexpr float kE4m3Max = 448.0f;

struct DeviceBuffer {
    void* pointer = nullptr;
    explicit DeviceBuffer(std::size_t bytes) {
        if (bytes != 0U) BEAM_CUDA_CHECK(cudaMalloc(&pointer, bytes));
    }
    ~DeviceBuffer() { cudaFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
    }
    return __shfl_sync(0xffffffffU, value, 0);
}

__device__ __forceinline__ float safe_ue8m0_scale(float max_abs) {
    if (!(max_abs > 0.0f)) return 1.0f;
    const float required = max_abs / kE4m3Max;
    int exponent = 0;
    const float mantissa = frexpf(required, &exponent);
    // UE8M0 has no mantissa. Round upward so E4M3 never saturates because the
    // scale rounded down. Exact powers of two must remain unchanged.
    return ldexpf(1.0f, exponent - (mantissa == 0.5f ? 1 : 0));
}

__device__ __forceinline__ std::size_t sm120_scale_offset(
    std::uint32_t primary, std::uint32_t k, std::uint32_t padded_k) {
    // CUTLASS SM120 MX layouts interleave a 128x128 scale atom. This formula
    // is the flattened form of LayoutSFA/LayoutSFB printed by CUTLASS 4.5:
    // (((32,4),P/128),((32,4),K/128),(1,1)) with strides
    // (((16,4),4*K),((0,1),512),(0,...)). The k%32 coordinate is broadcast.
    return static_cast<std::size_t>(primary % 32U) * 16U +
        static_cast<std::size_t>((primary / 32U) % 4U) * 4U +
        static_cast<std::size_t>(primary / 128U) * (4U * padded_k) +
        static_cast<std::size_t>((k / 32U) % 4U) +
        static_cast<std::size_t>(k / 128U) * 512U;
}

__global__ void quantize_a_kernel(
    const half* __restrict__ source,
    ElementA* __restrict__ encoded,
    ElementSFA* __restrict__ scales,
    std::uint32_t rows,
    std::uint32_t cols) {
    constexpr std::uint32_t warps_per_block = 4U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t groups_per_row = cols / kScaleVector;
    const std::uint64_t group =
        static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    if (group >= static_cast<std::uint64_t>(rows) * groups_per_row) return;
    const std::uint32_t row = static_cast<std::uint32_t>(group / groups_per_row);
    const std::uint32_t k0 = static_cast<std::uint32_t>(group % groups_per_row) * kScaleVector;
    const std::size_t index = static_cast<std::size_t>(row) * cols + k0 + lane;
    const float value = __half2float(source[index]);
    const float scale = safe_ue8m0_scale(warp_max(fabsf(value)));
    encoded[index] = ElementA(value / scale);
    if (lane == 0U) {
        scales[sm120_scale_offset(row, k0, cols)] = ElementSFA(scale);
    }
}

__global__ void quantize_b_kernel(
    const half* __restrict__ source_kxn,
    ElementB* __restrict__ encoded_kxn,
    ElementSFB* __restrict__ scales,
    std::uint32_t input_cols,
    std::uint32_t output_cols) {
    constexpr std::uint32_t warps_per_block = 4U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t groups_per_col = input_cols / kScaleVector;
    const std::uint64_t group =
        static_cast<std::uint64_t>(blockIdx.x) * warps_per_block + warp;
    if (group >= static_cast<std::uint64_t>(output_cols) * groups_per_col) return;
    const std::uint32_t column = static_cast<std::uint32_t>(group / groups_per_col);
    const std::uint32_t k0 = static_cast<std::uint32_t>(group % groups_per_col) * kScaleVector;
    const std::size_t index = static_cast<std::size_t>(k0 + lane) * output_cols + column;
    const float value = __half2float(source_kxn[index]);
    const float scale = safe_ue8m0_scale(warp_max(fabsf(value)));
    encoded_kxn[index] = ElementB(value / scale);
    if (lane == 0U) {
        scales[sm120_scale_offset(column, k0, input_cols)] = ElementSFB(scale);
    }
}

std::size_t scale_elements(std::uint32_t primary, std::uint32_t k) {
    if (primary % 128U != 0U || k % 128U != 0U) {
        throw std::invalid_argument("native MXFP8 fixture requires 128-aligned dimensions");
    }
    return static_cast<std::size_t>(primary) * k / kScaleVector;
}

void quantize_a(
    const half* source, ElementA* encoded, ElementSFA* scales,
    std::uint32_t rows, std::uint32_t cols, cudaStream_t stream) {
    const std::uint64_t groups = static_cast<std::uint64_t>(rows) * cols / kScaleVector;
    quantize_a_kernel<<<static_cast<unsigned>((groups + 3U) / 4U), 128, 0, stream>>>(
        source, encoded, scales, rows, cols);
    BEAM_CUDA_CHECK(cudaGetLastError());
}

void quantize_b(
    const half* source, ElementB* encoded, ElementSFB* scales,
    std::uint32_t k, std::uint32_t n, cudaStream_t stream) {
    const std::uint64_t groups = static_cast<std::uint64_t>(n) * k / kScaleVector;
    quantize_b_kernel<<<static_cast<unsigned>((groups + 3U) / 4U), 128, 0, stream>>>(
        source, encoded, scales, k, n);
    BEAM_CUDA_CHECK(cudaGetLastError());
}

typename Gemm::Arguments make_arguments(
    const ElementA* a, const ElementSFA* sfa,
    const ElementB* b, const ElementSFB* sfb, ElementD* d,
    int m, int n, int k) {
    const auto problem = make_shape(m, n, k, 1);
    const StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, make_shape(m, k, 1));
    const StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, make_shape(n, k, 1));
    const StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, make_shape(m, n, 1));
    const StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, make_shape(m, n, 1));
    const LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem);
    const LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem);
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        {a, stride_a, b, stride_b, sfa, layout_sfa, sfb, layout_sfb},
        {{1.0f, 0.0f}, nullptr, stride_c, d, stride_d},
    };
    return arguments;
}

void launch_gemm(
    const ElementA* a, const ElementSFA* sfa,
    const ElementB* b, const ElementSFB* sfb, ElementD* d,
    int m, int n, int k, void* workspace, std::size_t workspace_bytes,
    cudaStream_t stream) {
    auto arguments = make_arguments(a, sfa, b, sfb, d, m, n, k);
    Gemm gemm;
    if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
        throw std::runtime_error("native MXFP8 GEMM cannot implement shape");
    }
    const std::size_t required = Gemm::get_workspace_size(arguments);
    if (workspace_bytes < required) throw std::runtime_error("native MXFP8 workspace too small");
    if (gemm.initialize(arguments, workspace, stream) != cutlass::Status::kSuccess ||
        gemm.run(stream) != cutlass::Status::kSuccess) {
        throw std::runtime_error("native MXFP8 GEMM launch failed");
    }
}

struct Result {
    double encode_ms = 0.0;
    double gemm_ms = 0.0;
    double end_to_end_ms = 0.0;
    double gemm_tflops = 0.0;
    double end_to_end_tflops = 0.0;
};

Result benchmark(std::uint32_t m, std::uint32_t n, std::uint32_t k) {
    constexpr int warmups = 10;
    constexpr int iterations = 100;
    const std::size_t a_elements = static_cast<std::size_t>(m) * k;
    const std::size_t b_elements = static_cast<std::size_t>(k) * n;
    const std::size_t d_elements = static_cast<std::size_t>(m) * n;
    DeviceBuffer source_a(a_elements * sizeof(half));
    DeviceBuffer source_b(b_elements * sizeof(half));
    DeviceBuffer encoded_a(a_elements * sizeof(ElementA));
    DeviceBuffer encoded_b(b_elements * sizeof(ElementB));
    DeviceBuffer sfa(scale_elements(m, k) * sizeof(ElementSFA));
    DeviceBuffer sfb(scale_elements(n, k) * sizeof(ElementSFB));
    DeviceBuffer output(d_elements * sizeof(ElementD));
    BEAM_CUDA_CHECK(cudaMemset(source_a.pointer, 0x31, a_elements * sizeof(half)));
    BEAM_CUDA_CHECK(cudaMemset(source_b.pointer, 0x2d, b_elements * sizeof(half)));
    quantize_b(static_cast<const half*>(source_b.pointer),
               static_cast<ElementB*>(encoded_b.pointer),
               static_cast<ElementSFB*>(sfb.pointer), k, n, nullptr);
    quantize_a(static_cast<const half*>(source_a.pointer),
               static_cast<ElementA*>(encoded_a.pointer),
               static_cast<ElementSFA*>(sfa.pointer), m, k, nullptr);
    auto arguments = make_arguments(
        static_cast<const ElementA*>(encoded_a.pointer), static_cast<const ElementSFA*>(sfa.pointer),
        static_cast<const ElementB*>(encoded_b.pointer), static_cast<const ElementSFB*>(sfb.pointer),
        static_cast<ElementD*>(output.pointer), m, n, k);
    DeviceBuffer workspace(Gemm::get_workspace_size(arguments));
    auto run_encode = [&]() {
        quantize_a(static_cast<const half*>(source_a.pointer),
                   static_cast<ElementA*>(encoded_a.pointer),
                   static_cast<ElementSFA*>(sfa.pointer), m, k, nullptr);
    };
    auto run_gemm = [&]() {
        launch_gemm(
            static_cast<const ElementA*>(encoded_a.pointer), static_cast<const ElementSFA*>(sfa.pointer),
            static_cast<const ElementB*>(encoded_b.pointer), static_cast<const ElementSFB*>(sfb.pointer),
            static_cast<ElementD*>(output.pointer), m, n, k,
            workspace.pointer, Gemm::get_workspace_size(arguments), nullptr);
    };
    for (int i = 0; i < warmups; ++i) { run_encode(); run_gemm(); }
    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start{}, stop{};
    BEAM_CUDA_CHECK(cudaEventCreate(&start));
    BEAM_CUDA_CHECK(cudaEventCreate(&stop));
    auto measure = [&](auto&& operation) {
        BEAM_CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) operation();
        BEAM_CUDA_CHECK(cudaEventRecord(stop));
        BEAM_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        BEAM_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        return static_cast<double>(elapsed) / iterations;
    };
    Result result;
    result.encode_ms = measure(run_encode);
    result.gemm_ms = measure(run_gemm);
    result.end_to_end_ms = measure([&]() { run_encode(); run_gemm(); });
    const double flops = 2.0 * static_cast<double>(m) * n * k;
    result.gemm_tflops = flops / (result.gemm_ms * 1.0e9);
    result.end_to_end_tflops = flops / (result.end_to_end_ms * 1.0e9);
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return result;
}

}  // namespace sm120_mxfp8_bench

using namespace sm120_mxfp8_bench;

int main(int argc, char** argv) {
    cudaDeviceProp properties{};
    BEAM_CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
    if (properties.major != 12 || properties.minor != 0) {
        std::cerr << "native MXFP8 benchmark requires SM120\n";
        return EXIT_FAILURE;
    }
    const std::uint32_t m = argc > 1 ? static_cast<std::uint32_t>(std::stoul(argv[1])) : 51072U;
    for (const auto [name, n, k] : std::vector<std::tuple<std::string, std::uint32_t, std::uint32_t>>{
             {"qkv", 768U, 256U}, {"ff1", 1024U, 256U},
             {"projection", 256U, 256U}, {"ff2", 256U, 1024U}}) {
        const Result result = benchmark(m, n, k);
        std::cout << "sm120_native_mxfp8"
                  << " name=" << name << " m=" << m << " n=" << n << " k=" << k
                  << " encode_ms=" << std::fixed << std::setprecision(6) << result.encode_ms
                  << " gemm_ms=" << result.gemm_ms
                  << " end_to_end_ms=" << result.end_to_end_ms
                  << " gemm_tflops=" << std::setprecision(3) << result.gemm_tflops
                  << " end_to_end_tflops=" << result.end_to_end_tflops << "\n";
    }
    return EXIT_SUCCESS;
}
