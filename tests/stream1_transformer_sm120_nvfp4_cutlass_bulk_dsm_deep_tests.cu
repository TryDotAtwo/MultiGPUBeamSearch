#include "stream1_transformer_sm120_nvfp4_cutlass_dsm_fused.cuh"

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/pipeline/pipeline.hpp>
#include <cuda/__ptx/instructions/cp_async_bulk.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace stream1_sm120_nvfp4_cutlass_dsm_deep {

using namespace cute;
using Base = stream1_sm120_nvfp4_cutlass_dsm::Contract;
using ElementAB = stream1_sm120_nvfp4_cutlass_dsm::ElementAB;
using ElementAccumulator =
    stream1_sm120_nvfp4_cutlass_dsm::ElementAccumulator;
using ElementCompute = stream1_sm120_nvfp4_cutlass_dsm::ElementCompute;
using ElementScale = stream1_sm120_nvfp4_cutlass_dsm::ElementScale;
using ElementOutput = stream1_sm120_nvfp4_cutlass_dsm::ElementOutput;
using LayoutA = stream1_sm120_nvfp4_cutlass_dsm::LayoutA;
using LayoutB = stream1_sm120_nvfp4_cutlass_dsm::LayoutB;
using LayoutD = stream1_sm120_nvfp4_cutlass_dsm::LayoutD;
using ArchTag = stream1_sm120_nvfp4_cutlass_dsm::ArchTag;
using ProblemShape = stream1_sm120_nvfp4_cutlass_dsm::ProblemShape;
using KernelSchedule = stream1_sm120_nvfp4_cutlass_dsm::KernelSchedule;
using EpilogueSchedule = stream1_sm120_nvfp4_cutlass_dsm::EpilogueSchedule;
using ClusterShape = Shape<_1, _1, _1>;
using LaunchClusterShape = Shape<_2, _1, _1>;
using DeepTileShape = Shape<_128, _128, _256>;

struct Contract {
    static constexpr std::uint32_t kRows = Base::kRows;
    static constexpr std::uint32_t kDModel = Base::kDModel;
    static constexpr std::uint32_t kFfDim = Base::kFfDim;
    static constexpr std::uint32_t kSliceColumns = 256U;
    static constexpr std::uint32_t kSlices = kFfDim / kSliceColumns;
    static constexpr std::uint32_t kClusterCtas = 2U;
    // Ownership alternates between the two cluster CTAs.  Each CTA therefore
    // stores one local slot; together the cluster still has the two physical
    // slots required to overlap producer and consumer work.
    static constexpr std::uint32_t kLocalRingSlots = 1U;
    static constexpr std::uint32_t kClusterRingSlots = 2U;
    static constexpr std::uint32_t kScaleVector = Base::kScaleVector;
    static constexpr std::uint32_t kValuesPerSlot =
        kRows * kSliceColumns / 2U;
    static constexpr std::uint32_t kScalesPerSlot =
        kRows * (kSliceColumns / kScaleVector);
    static constexpr std::uint32_t kRingBytes =
        kLocalRingSlots * (kValuesPerSlot + kScalesPerSlot);
};

using DeepEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp,
        DeepTileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementCompute,
        ElementOutput, LayoutD, 8,
        ElementOutput, LayoutD, 8,
        EpilogueSchedule>::CollectiveOp;

using DeepStages = cutlass::gemm::collective::StageCountAutoCarveout<
    static_cast<int>(Contract::kRingBytes)>;

using DeepMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassBlockScaledTensorOp,
        ElementAB, LayoutA, 32,
        ElementAB, LayoutB, 32,
        ElementAccumulator, DeepTileShape, ClusterShape,
        DeepStages, KernelSchedule>::CollectiveOp;

using DeepKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape, DeepMainloop, DeepEpilogue, void>;

struct alignas(1024) DsmASlot {
    alignas(1024) std::uint8_t values[Contract::kValuesPerSlot];
    alignas(16) std::uint8_t scales[Contract::kScalesPerSlot];
};

struct DeepCollective {
    using MainloopPipeline = typename DeepMainloop::MainloopPipeline;
    using PipelineState = typename DeepMainloop::PipelineState;
    struct SharedStorage {
        alignas(1024) typename DeepMainloop::TensorStorage tensors;
        alignas(16) typename MainloopPipeline::SharedStorage pipeline;
        DsmASlot hidden[Contract::kLocalRingSlots];
        alignas(8) std::uint64_t handoff_armed;
    };
    static constexpr std::uint32_t kDsmTransactionBytes =
        DeepMainloop::TmaTransactionBytesMK;
    static constexpr std::uint32_t kTmaTransactionBytes =
        DeepMainloop::TmaTransactionBytesNK;
    static constexpr std::uint32_t kTotalTransactionBytes =
        DeepMainloop::TmaTransactionBytes;
    static constexpr std::uint32_t kThreads = 256U;
};

static_assert(cute::thr_size(typename DeepMainloop::TiledMma{}) == 128);
static_assert(DeepCollective::kDsmTransactionBytes ==
              Contract::kValuesPerSlot + Contract::kScalesPerSlot);
static_assert(DeepCollective::kTmaTransactionBytes +
                  DeepCollective::kDsmTransactionBytes ==
              DeepCollective::kTotalTransactionBytes);
static_assert(sizeof(DeepCollective::SharedStorage) <=
              ArchTag::kSharedMemoryCapacityBytes);

template <class LoadInputs>
CUTLASS_DEVICE void issue_b_sfb_tma(
    typename DeepMainloop::Params const& params,
    typename DeepCollective::MainloopPipeline& pipeline,
    typename DeepCollective::PipelineState write_state,
    LoadInputs const& load_inputs,
    std::uint32_t n_tile,
    std::uint32_t k_tile,
    DeepCollective::SharedStorage& storage) {
    auto [gA_mkl, gB_nkl, gSFA_mkl, gSFB_nkl] = load_inputs;
    (void)gA_mkl;
    (void)gSFA_mkl;
    auto block_tma_b = params.tma_load_b.get_slice(0);
    auto block_tma_sfb = params.tma_load_sfb.get_slice(0);
    Tensor gB = gB_nkl(_, _, n_tile, _, 0);
    auto broadcast_n = make_layout(
        make_shape(
            Int<size<1>(typename DeepMainloop::TileShapeSFB{}) /
                size<1>(DeepTileShape{})>{},
            Int<cute::numeric_limits<int>::max()>{}),
        make_stride(
            _0{}, size<1>(typename DeepMainloop::TileShapeSFB{}) /
                size<1>(DeepTileShape{})));
    Tensor gSFB = gSFB_nkl(_, _, broadcast_n(n_tile), _, 0);
    Tensor sB = make_tensor(
        make_smem_ptr(storage.tensors.smem_B.begin()),
        typename DeepMainloop::SmemLayoutB{});
    Tensor sSFB = make_tensor(
        make_smem_ptr(storage.tensors.smem_SFB.begin()),
        typename DeepMainloop::SmemLayoutSFB{});
    Tensor tBgB = block_tma_b.partition_S(gB);
    Tensor tBsB = block_tma_b.partition_D(sB);
    Tensor tBgSFB = block_tma_sfb.partition_S(gSFB);
    Tensor tBsSFB = block_tma_sfb.partition_D(sSFB);
    using Barrier = typename DeepCollective::MainloopPipeline::
        ProducerBarrierType;
    Barrier* barrier = pipeline.producer_get_barrier(write_state);
    const int stage = write_state.index();
    copy(params.tma_load_b.with(*barrier),
         tBgB(_, _, _, k_tile), tBsB(_, _, _, stage));
    copy(params.tma_load_sfb.with(*barrier),
         tBgSFB(_, _, _, k_tile), tBsSFB(_, _, _, stage));
}

CUTLASS_DEVICE void issue_a_sfa_bulk_dsm(
    typename DeepCollective::MainloopPipeline& pipeline,
    typename DeepCollective::PipelineState write_state,
    DeepCollective::SharedStorage& storage,
    std::uint32_t slot) {
    namespace cg = cooperative_groups;
    const auto cluster = cg::this_cluster();
    const int stage = write_state.index();
    auto* local_a = reinterpret_cast<std::uint8_t*>(
        &storage.tensors.smem_A) +
        static_cast<std::size_t>(stage) * Contract::kValuesPerSlot;
    auto* local_sfa = reinterpret_cast<std::uint8_t*>(
        &storage.tensors.smem_SFA) +
        static_cast<std::size_t>(stage) * Contract::kScalesPerSlot;
    auto* local_barrier = pipeline.producer_get_barrier(write_state);
    constexpr std::uint32_t values_bytes = Contract::kValuesPerSlot;
    constexpr std::uint32_t scales_bytes = Contract::kScalesPerSlot;
    for (std::uint32_t target = 0; target < Contract::kClusterCtas;
         ++target) {
        auto* target_a = cluster.map_shared_rank(local_a, target);
        auto* target_sfa = cluster.map_shared_rank(local_sfa, target);
        auto* target_barrier = cluster.map_shared_rank(
            local_barrier, target);
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_cluster, cuda::ptx::space_shared,
            target_a, storage.hidden[slot].values,
            values_bytes, target_barrier);
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_cluster, cuda::ptx::space_shared,
            target_sfa, storage.hidden[slot].scales,
            scales_bytes, target_barrier);
    }
}

static __global__ __launch_bounds__(256, 1) __cluster_dims__(2, 1, 1)
void bulk_dsm_ff2_deep_kernel(
    CUTE_GRID_CONSTANT typename DeepMainloop::Params const params,
    float* rank_checksums) {
    namespace cg = cooperative_groups;
    extern __shared__ __align__(1024) unsigned char shared_bytes[];
    auto& storage = *reinterpret_cast<
        DeepCollective::SharedStorage*>(shared_bytes);
    const auto cluster = cg::this_cluster();
    const std::uint32_t rank = cluster.block_rank();
    const std::uint32_t warp_group = threadIdx.x / 128U;
    const std::uint32_t wg_thread = threadIdx.x % 128U;

    if (threadIdx.x == 0U) {
        cutlass::arch::ClusterBarrier::init(&storage.handoff_armed, 2U);
    }
    for (std::uint32_t slot = 0; slot < Contract::kLocalRingSlots; ++slot) {
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kValuesPerSlot; i += blockDim.x) {
            storage.hidden[slot].values[i] = 0x11U;
        }
        for (std::uint32_t i = threadIdx.x;
             i < Contract::kScalesPerSlot; i += blockDim.x) {
            storage.hidden[slot].scales[i] =
                ElementScale(1.0F).raw();
        }
    }

    using Pipeline = DeepCollective::MainloopPipeline;
    using PipelineState = DeepCollective::PipelineState;
    typename Pipeline::Params pipeline_params;
    pipeline_params.transaction_bytes =
        DeepCollective::kTotalTransactionBytes;
    pipeline_params.num_consumers = 128U;
    pipeline_params.is_leader = wg_thread == 0U;
    pipeline_params.role = warp_group == 0U
        ? Pipeline::ThreadCategory::Producer
        : Pipeline::ThreadCategory::Consumer;
    Pipeline pipeline(storage.pipeline, pipeline_params, LaunchClusterShape{});
    __syncthreads();
    cutlass::arch::fence_barrier_init();
    cute::cluster_arrive_relaxed();
    cute::cluster_wait();

    constexpr auto problem_shape = make_shape(128, 256, 1024, 1);
    DeepMainloop mainloop;
    const auto load_inputs = mainloop.load_init(problem_shape, params);
    if (warp_group == 0U && wg_thread == 0U) {
        PipelineState write_state =
            cutlass::make_producer_start_state<Pipeline>();
        for (std::uint32_t slice = 0; slice < Contract::kSlices; ++slice) {
            pipeline.producer_acquire(write_state);
            issue_b_sfb_tma(
                params, pipeline, write_state, load_inputs,
                rank, slice, storage);
            const std::uint32_t owner = slice & 1U;
            cutlass::arch::ClusterBarrier::arrive(
                &storage.handoff_armed, owner, 1U);
            if (rank == owner) {
                cutlass::arch::ClusterBarrier::wait(
                    &storage.handoff_armed, (slice / 2U) & 1U);
                issue_a_sfa_bulk_dsm(
                    pipeline, write_state, storage,
                    0U);
            }
            ++write_state;
        }
        pipeline.producer_tail(write_state);
    } else if (warp_group == 1U) {
        typename DeepMainloop::TiledMma tiled_mma;
        auto accum = partition_fragment_C(
            tiled_mma, take<0, 2>(DeepTileShape{}));
        PipelineState read_state;
        mainloop.mma(
            pipeline, read_state, accum, Contract::kSlices,
            static_cast<int>(wg_thread), storage.tensors,
            params, make_coord(0, rank, 0, 0));
        float local = 0.0F;
        for (int i = 0; i < size(accum); ++i) {
            local += accum(i) < 0.0F ? -accum(i) : accum(i);
        }
        const std::uint32_t cluster_index = blockIdx.x / 2U;
        atomicAdd(rank_checksums + cluster_index * 2U + rank, local);
    }
}

}  // namespace stream1_sm120_nvfp4_cutlass_dsm_deep

namespace {

using namespace cute;
using namespace stream1_sm120_nvfp4_cutlass_dsm_deep;

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main(int argc, char** argv) {
    constexpr int m = 128;
    constexpr int n = 256;
    constexpr int k = 1024;
    const int clusters = argc > 1 ? std::atoi(argv[1]) : 1;
    const int iterations = argc > 2 ? std::atoi(argv[2]) : 0;
    require(clusters > 0 && clusters <= 512,
            "clusters must be in [1, 512]");
    require(iterations >= 0, "iterations must be non-negative");
    const auto problem_shape = make_shape(m, n, k, 1);
    const typename DeepMainloop::StrideA stride_a =
        make_stride(std::int64_t{k}, _1{}, std::int64_t{m * k});
    const typename DeepMainloop::StrideB stride_b =
        make_stride(std::int64_t{k}, _1{}, std::int64_t{n * k});
    using ScaleConfig = typename DeepMainloop::Sm1xxBlkScaledConfig;
    const typename DeepMainloop::LayoutSFA layout_sfa =
        ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
    const typename DeepMainloop::LayoutSFB layout_sfb =
        ScaleConfig::tile_atom_to_shape_SFB(problem_shape);

    cutlass::float_e2m1_t *unused_a = nullptr, *b = nullptr;
    ElementScale *unused_sfa = nullptr, *sfb = nullptr;
    float* rank_checksums = nullptr;
    require(cudaMalloc(&unused_a, static_cast<std::size_t>(m) * k / 2U) ==
                cudaSuccess,
            "cudaMalloc unused A failed");
    require(cudaMalloc(&b, static_cast<std::size_t>(n) * k / 2U) ==
                cudaSuccess,
            "cudaMalloc B failed");
    require(cudaMalloc(
                &unused_sfa,
                static_cast<std::size_t>(size(filter_zeros(layout_sfa)))) ==
                cudaSuccess,
            "cudaMalloc unused SFA failed");
    require(cudaMalloc(
                &sfb,
                static_cast<std::size_t>(size(filter_zeros(layout_sfb)))) ==
                cudaSuccess,
            "cudaMalloc SFB failed");
    require(cudaMalloc(
                &rank_checksums,
                2U * clusters * sizeof(float)) == cudaSuccess,
            "cudaMalloc checksums failed");
    cudaMemset(unused_a, 0, static_cast<std::size_t>(m) * k / 2U);
    cudaMemset(b, 0x11, static_cast<std::size_t>(n) * k / 2U);
    cudaMemset(unused_sfa, 0, size(filter_zeros(layout_sfa)));
    cudaMemset(sfb, ElementScale(1.0F).raw(),
               size(filter_zeros(layout_sfb)));
    cudaMemset(rank_checksums, 0, 2U * clusters * sizeof(float));

    typename DeepMainloop::Arguments arguments{
        unused_a, stride_a, b, stride_b,
        unused_sfa, layout_sfa, sfb, layout_sfb};
    require(DeepMainloop::can_implement(problem_shape, arguments),
            "DeepMainloop rejected bulk-DSM descriptor inputs");
    const auto params = DeepMainloop::to_underlying_arguments(
        problem_shape, arguments, nullptr);
    require(cudaFuncSetAttribute(
                bulk_dsm_ff2_deep_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                sizeof(DeepCollective::SharedStorage)) == cudaSuccess,
            "deep bulk-DSM shared-memory attribute failed");
    bulk_dsm_ff2_deep_kernel<<<
        2 * clusters, DeepCollective::kThreads,
        sizeof(DeepCollective::SharedStorage)>>>(params, rank_checksums);
    require(cudaGetLastError() == cudaSuccess,
            "deep bulk-DSM launch failed");
    require(cudaDeviceSynchronize() == cudaSuccess,
            "deep bulk-DSM runtime failed");
    float host[1024] = {};
    cudaMemcpy(host, rank_checksums, 2U * clusters * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int cluster = 0; cluster < clusters; ++cluster) {
        require(host[2 * cluster] > 0.0F,
                "deep bulk-DSM checksum is zero");
        require(host[2 * cluster] == host[2 * cluster + 1],
                "deep bulk-DSM differs between output halves");
        require(host[2 * cluster] == host[0],
                "deep bulk-DSM differs between clusters");
    }
    std::cout << "stream1_transformer_sm120_nvfp4_cutlass_bulk_dsm_deep=pass"
              << " rank0_checksum=" << host[0]
              << " rank1_checksum=" << host[1]
              << " slice_columns=" << Contract::kSliceColumns
              << " slices=" << Contract::kSlices
              << " transaction_bytes="
              << DeepCollective::kTotalTransactionBytes
              << " shared_bytes=" << sizeof(DeepCollective::SharedStorage)
              << '\n';

    if (iterations > 0) {
        constexpr int warmups = 20;
        for (int iteration = 0; iteration < warmups; ++iteration) {
            bulk_dsm_ff2_deep_kernel<<<
                2 * clusters, DeepCollective::kThreads,
                sizeof(DeepCollective::SharedStorage)>>>(
                    params, rank_checksums);
        }
        require(cudaDeviceSynchronize() == cudaSuccess,
                "deep bulk-DSM warmup failed");
        cudaEvent_t start{}, stop{};
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            bulk_dsm_ff2_deep_kernel<<<
                2 * clusters, DeepCollective::kThreads,
                sizeof(DeepCollective::SharedStorage)>>>(
                    params, rank_checksums);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float total_ms = 0.0F;
        cudaEventElapsedTime(&total_ms, start, stop);
        const double launch_ms = total_ms / iterations;
        const double flops = 2.0 * clusters * m * n * k;
        const double tflops = flops / (launch_ms * 1.0e9);
        std::cout << "stream1_transformer_sm120_nvfp4_cutlass_bulk_dsm_deep_bench"
                  << " clusters=" << clusters
                  << " iterations=" << iterations
                  << " launch_ms=" << std::fixed << std::setprecision(6)
                  << launch_ms
                  << " ff2_tflops=" << std::setprecision(3) << tflops
                  << '\n';
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    cudaFree(rank_checksums);
    cudaFree(unused_a);
    cudaFree(b);
    cudaFree(unused_sfa);
    cudaFree(sfb);
    return EXIT_SUCCESS;
}
