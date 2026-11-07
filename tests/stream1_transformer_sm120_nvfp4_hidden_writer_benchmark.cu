#include "stream1_transformer_sm120_nvfp4_cutlass_persistent_dsm.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <type_traits>

using namespace stream1_sm120_nvfp4_cutlass_dsm;

template <class Policy>
struct EpiloguePolicyTraits;

template <int StagesC, int StagesD, int FragmentSize, bool Reuse, bool Delay>
struct EpiloguePolicyTraits<
    cutlass::epilogue::Sm90TmaWarpSpecialized<
        StagesC, StagesD, FragmentSize, Reuse, Delay>> {
    static constexpr int kStagesC = StagesC;
    static constexpr int kStagesD = StagesD;
    static constexpr bool kReuse = Reuse;
};

using Ff1EpiloguePolicy = EpiloguePolicyTraits<typename Ff1Epilogue::DispatchPolicy>;
using Ff1ReconstructedSmemLayoutD = decltype(cute::tile_to_shape(
    typename Ff1Epilogue::SmemLayoutAtomD{},
    cute::make_shape(
        cute::size<0>(typename Ff1Epilogue::EpilogueTile{}),
        cute::size<1>(typename Ff1Epilogue::EpilogueTile{}),
        cute::Int<Ff1EpiloguePolicy::kReuse ? Ff1EpiloguePolicy::kStagesC
                                            : Ff1EpiloguePolicy::kStagesD>{}),
    cute::conditional_t<
        cutlass::epilogue::collective::detail::is_m_major<
            typename Ff1Epilogue::StrideD>(),
        cute::Step<cute::_2, cute::_1, cute::_3>,
        cute::Step<cute::_1, cute::_2, cute::_3>>{}));

__global__ void fill_slot(std::uint32_t* checksum) {
    __shared__ DsmASlot slot;
    const std::uint32_t subgroup = threadIdx.x / 4U;
    const std::uint32_t lane = threadIdx.x & 3U;
    for (std::uint32_t work = subgroup; work < 128U * 8U;
         work += blockDim.x / 4U) {
        const std::uint32_t row = work / 8U;
        const std::uint32_t group = work % 8U;
        const std::uint8_t p0a = (row + group + lane * 2U) & 15U;
        const std::uint8_t p0b = (row + group + lane * 2U + 1U) & 15U;
        const std::uint8_t p1a = (row + group + 8U + lane * 2U) & 15U;
        const std::uint8_t p1b = (row + group + 8U + lane * 2U + 1U) & 15U;
        Ff1HiddenRingWriter::store_group_from_lane_pairs(
            slot, row, group,
            static_cast<std::uint8_t>(p0a | (p0b << 4U)),
            static_cast<std::uint8_t>(p1a | (p1b << 4U)),
            cutlass::float_ue4m3_t(1.0F).raw());
    }
    __syncthreads();
    if (threadIdx.x == 0U) atomicAdd(checksum, slot.values[0] + slot.scales[0]);
}

int main() {
    std::cout << "layout a_size=" << cute::size(Ff2StageLayoutA{})
              << " a_cosize=" << cute::cosize(Ff2StageLayoutA{})
              << " sfa_size=" << cute::size(Ff2StageLayoutSFA{})
              << " sfa_cosize=" << cute::cosize(Ff2StageLayoutSFA{})
              << " values_bytes=" << Contract::kValuesPerSlot
              << " scales_bytes=" << Contract::kScalesPerSlot
              << " alloc_a_sizeof=" << sizeof(PersistentBaseFf2Mainloop::SmemAllocTypeA)
              << " tensor_storage_bytes=" << sizeof(PersistentBaseFf2Mainloop::TensorStorage)
              << " ff1_epilogue_storage_bytes=" << sizeof(Ff1Epilogue::SharedStorage)
              << " ff1_d_size=" << cute::size(Ff1ReconstructedSmemLayoutD{})
              << " ff1_d_cosize=" << cute::cosize(Ff1ReconstructedSmemLayoutD{})
              << " ff1_ff2_layout_same="
              << std::is_same_v<
                     decltype(Ff1ReconstructedSmemLayoutD{}(_, _, cute::Int<0>{})),
                     Ff2StageLayoutA>
              << '\n';
    std::uint32_t* sink = nullptr;
    cudaMallocManaged(&sink, sizeof(*sink));
    *sink = 0;
    fill_slot<<<1, 128>>>(sink);
    auto correctness = cudaDeviceSynchronize();
    if (correctness != cudaSuccess || *sink == 0U) {
        std::cout << "correctness=fail cuda=" << cudaGetErrorString(correctness) << '\n';
        return 2;
    }
    for (int i = 0; i < 100; ++i) fill_slot<<<1, 128>>>(sink);
    auto warmup = cudaDeviceSynchronize();
    cudaEvent_t begin{}, end{};
    cudaEventCreate(&begin); cudaEventCreate(&end);
    cudaEventRecord(begin);
    constexpr int iterations = 10000;
    for (int i = 0; i < iterations; ++i) fill_slot<<<1, 128>>>(sink);
    cudaEventRecord(end);
    auto timed = cudaEventSynchronize(end);
    float ms = 0.0F;
    cudaEventElapsedTime(&ms, begin, end);
    const double us = ms * 1000.0 / iterations;
    std::cout << "correctness=pass checksum=" << *sink
              << " us=" << us
              << " effective_GBps=" << (Ff1HiddenRingWriter::kSlotBytes / (us * 1000.0))
              << " warmup=" << cudaGetErrorString(warmup)
              << " timed=" << cudaGetErrorString(timed) << '\n';
    return warmup == cudaSuccess && timed == cudaSuccess && ms > 0.0F ? 0 : 3;
}
