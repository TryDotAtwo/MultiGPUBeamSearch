#pragma once

#include "stream1_transformer_hopper_policy.hpp"

#if BEAM_HAS_CUTLASS
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#endif

namespace beam {

#if BEAM_HAS_CUTLASS && defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

template <template <class> class Activation>
__global__ void stream1_transformer_hopper_bias_activation_kernel(
    half* output, const half* bias, std::uint64_t elements, std::uint32_t cols) {
    const std::uint64_t idx = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= elements) return;
    Activation<float> activation;
    output[idx] = __float2half(activation(__half2float(output[idx]) + __half2float(bias[idx % cols])));
}

template <template <class> class Activation>
void stream1_transformer_hopper_fp16_bias_activation(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    cudaStream_t stream) {
    using namespace cute;
    using Element = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;
    // Hopper profile uses offline-packed physical KxN column-major weights.
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutD = cutlass::layout::RowMajor;
    constexpr int Alignment = 128 / cutlass::sizeof_bits<Element>::value;
    using TileShape = Shape<_128, _128, _64>;
    using ClusterShape = Shape<_1, _2, _1>;
    using Fusion = cutlass::epilogue::fusion::LinearCombination<Element, float, void, float>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        float, float,
        void, LayoutD, 1,
        Element, LayoutD, Alignment,
        cutlass::epilogue::TmaWarpSpecializedCooperative,
        Fusion>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
        Element, LayoutA, Alignment,
        Element, LayoutB, Alignment,
        float,
        TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
    using StrideA = typename Kernel::StrideA;
    using StrideB = typename Kernel::StrideB;
    using StrideD = typename Kernel::StrideD;

    const auto shape = make_shape(static_cast<int>(rows), static_cast<int>(output_cols),
                                  static_cast<int>(input_cols), 1);
    const auto stride_a = cutlass::make_cute_packed_stride(StrideA{},
        make_shape(static_cast<int>(rows), static_cast<int>(input_cols), 1));
    const auto stride_b = cutlass::make_cute_packed_stride(StrideB{},
        make_shape(static_cast<int>(output_cols), static_cast<int>(input_cols), 1));
    const auto stride_d = cutlass::make_cute_packed_stride(StrideD{},
        make_shape(static_cast<int>(rows), static_cast<int>(output_cols), 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        shape,
        {reinterpret_cast<Element const*>(input), stride_a,
         reinterpret_cast<Element const*>(weight), stride_b},
        {{1.0f, 0.0f}, nullptr, stride_d,
         reinterpret_cast<Element*>(output), stride_d}
    };
    const std::size_t workspace_bytes = Gemm::get_workspace_size(args);
    if (workspace_bytes != 0U) {
        throw std::runtime_error(
            "Hopper Stream1 FP16 TMA GEMM unexpectedly requires scheduler workspace");
    }
    Gemm gemm;
    const cutlass::Status status = gemm(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Hopper Stream1 FP16 TMA bias+activation GEMM failed");
    }
    const std::uint64_t elements = static_cast<std::uint64_t>(rows) * output_cols;
    stream1_transformer_hopper_bias_activation_kernel<Activation>
        <<<(elements + 255U) / 256U, 256, 0, stream>>>(output, bias, elements, output_cols);
}

#endif

}  // namespace beam
