#include "stream1_transformer_sm120_nvfp4_cutlass_dsm_fused.cuh"

#include <cstdlib>
#include <iostream>

int main() {
    using Plan = stream1_sm120_nvfp4_cutlass_dsm::Plan;
    using Storage = stream1_sm120_nvfp4_cutlass_dsm::SharedStorage;
    using Handoff =
        stream1_sm120_nvfp4_cutlass_dsm::BulkDsmFf2Collective;
    static_assert(Plan::kGlobalHiddenBytes == 0U);
    static_assert(Plan::kClusterCtas == 2U);
    static_assert(Plan::kRingSlots == 2U);
    static_assert(Plan::kSlices == 8U);
    static_assert(Plan::kPipelineStages == 9U);
    static_assert(Plan::kContractProducerWarps == 4U);
    static_assert(Plan::kContractConsumerWarps == 4U);
    static_assert(Plan::kDmaWarps == 4U);
    static_assert(Plan::kFf1MathWarps == 4U);
    static_assert(Plan::kFf2MathWarps == 4U);
    static_assert(Plan::kKernelThreads == 384U);
    static_assert(Handoff::kLoadsGlobalA == false);
    static_assert(Handoff::kLoadsGlobalSFA == false);
    static_assert(Handoff::kLoadsGlobalB == true);
    static_assert(Handoff::kLoadsGlobalSFB == true);
    static_assert(Handoff::kUsesCutlassConsumerMma == true);
    static_assert(Handoff::kUsesBulkDsm == true);
    static_assert(Handoff::kUsesPipelineTransactionBarrier == true);
    static_assert(Handoff::kUsesFullClusterSyncInHotLoop == false);
    static_assert(Handoff::kDsmTransactionBytes +
                      Handoff::kTmaTransactionBytes ==
                  Handoff::kTotalTransactionBytes);
    if (sizeof(Storage) >
        stream1_sm120_nvfp4_cutlass_dsm::ArchTag::kSharedMemoryCapacityBytes) {
        return EXIT_FAILURE;
    }
    std::cout << "stream1_transformer_sm120_nvfp4_cutlass_dsm_contract=pass"
              << " cluster_ctas=" << Plan::kClusterCtas
              << " shared_bytes=" << Plan::kSharedBytes
              << " ring_bytes=" << Plan::kRingBytes
              << " ff1_stages=" << Plan::kFf1Stages
              << " ff2_stages=" << Plan::kFf2Stages << '\n';

    std::uint64_t* checksums = nullptr;
    std::uint32_t* violations = nullptr;
    cudaMallocManaged(&checksums,
        Plan::kClusterCtas * Plan::kSlices * sizeof(std::uint64_t));
    cudaMallocManaged(&violations, sizeof(std::uint32_t));
    cudaMemset(checksums, 0,
        Plan::kClusterCtas * Plan::kSlices * sizeof(std::uint64_t));
    cudaMemset(violations, 0, sizeof(std::uint32_t));
    const cudaError_t attribute_status = cudaFuncSetAttribute(
        stream1_sm120_nvfp4_cutlass_dsm::dsm_ring_contract_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage));
    if (attribute_status != cudaSuccess) {
        std::cerr << "set attribute failed: "
                  << cudaGetErrorString(attribute_status) << '\n';
        return EXIT_FAILURE;
    }
    stream1_sm120_nvfp4_cutlass_dsm::dsm_ring_contract_kernel<<<
        Plan::kClusterCtas, 256, sizeof(Storage)>>>(checksums, violations);
    const cudaError_t run_status = cudaDeviceSynchronize();
    if (run_status != cudaSuccess || *violations != 0U) {
        std::cerr << "DSM ring failed: " << cudaGetErrorString(run_status)
                  << " violations=" << *violations << '\n';
        return EXIT_FAILURE;
    }
    for (std::uint32_t slice = 0; slice < Plan::kSlices; ++slice) {
        if (checksums[slice] == 0U ||
            checksums[slice] != checksums[Plan::kSlices + slice]) {
            std::cerr << "DSM checksum mismatch at slice " << slice << '\n';
            return EXIT_FAILURE;
        }
    }
    std::cout << "stream1_transformer_sm120_nvfp4_cutlass_dsm_runtime=pass"
              << " slices=" << Plan::kSlices
              << " violations=" << *violations << '\n';

    cudaMemset(checksums, 0,
        Plan::kClusterCtas * Plan::kSlices * sizeof(std::uint64_t));
    cudaMemset(violations, 0, sizeof(std::uint32_t));
    const cudaError_t overlap_attribute_status = cudaFuncSetAttribute(
        stream1_sm120_nvfp4_cutlass_dsm::dsm_pingpong_overlap_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage));
    if (overlap_attribute_status != cudaSuccess) {
        std::cerr << "set overlap attribute failed: "
                  << cudaGetErrorString(overlap_attribute_status) << '\n';
        return EXIT_FAILURE;
    }
    stream1_sm120_nvfp4_cutlass_dsm::dsm_pingpong_overlap_kernel<<<
        Plan::kClusterCtas, Plan::kContractThreads, sizeof(Storage)>>>(
            checksums, violations);
    const cudaError_t overlap_status = cudaDeviceSynchronize();
    if (overlap_status != cudaSuccess || *violations != 0U) {
        std::cerr << "DSM overlap failed: "
                  << cudaGetErrorString(overlap_status)
                  << " violations=" << *violations << '\n';
        return EXIT_FAILURE;
    }
    for (std::uint32_t slice = 0; slice < Plan::kSlices; ++slice) {
        if (checksums[slice] == 0U ||
            checksums[slice] != checksums[Plan::kSlices + slice]) {
            std::cerr << "DSM overlap checksum mismatch at slice "
                      << slice << '\n';
            return EXIT_FAILURE;
        }
    }
    std::cout << "stream1_transformer_sm120_nvfp4_cutlass_dsm_overlap=pass"
              << " pipeline_stages=" << Plan::kPipelineStages
              << " contract_producer_warps="
              << Plan::kContractProducerWarps
              << " contract_consumer_warps_per_cta="
              << Plan::kContractConsumerWarps
              << " production_kernel_threads=" << Plan::kKernelThreads
              << " violations=" << *violations << '\n';

    using RolePlan =
        stream1_sm120_nvfp4_cutlass_dsm::RoleSeparatedDsmPlan;
    using RoleStorage =
        stream1_sm120_nvfp4_cutlass_dsm::RoleSeparatedDsmStorage;
    static_assert(RolePlan::kProducerRank == 0U);
    static_assert(RolePlan::kConsumerRank == 1U);
    static_assert(RolePlan::kClusterCtas == 2U);
    static_assert(RolePlan::kRingSlots == 2U);
    static_assert(RolePlan::kSlices == 8U);
    static_assert(RolePlan::kHotLoopClusterSyncs == 0U);
    static_assert(RolePlan::kGlobalHiddenBytes == 0U);
    static_assert(RolePlan::kTransactionBytes == 9216U);
    cudaMemset(checksums, 0,
        Plan::kClusterCtas * Plan::kSlices * sizeof(std::uint64_t));
    cudaMemset(violations, 0, sizeof(std::uint32_t));
    const cudaError_t role_attribute_status = cudaFuncSetAttribute(
        stream1_sm120_nvfp4_cutlass_dsm::
            role_separated_dsm_pipeline_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(RoleStorage));
    if (role_attribute_status != cudaSuccess) {
        std::cerr << "set role-separated attribute failed: "
                  << cudaGetErrorString(role_attribute_status) << '\n';
        return EXIT_FAILURE;
    }
    stream1_sm120_nvfp4_cutlass_dsm::
        role_separated_dsm_pipeline_kernel<<<
            RolePlan::kClusterCtas,
            RolePlan::kContractThreads,
            sizeof(RoleStorage)>>>(checksums, violations);
    const cudaError_t role_status = cudaDeviceSynchronize();
    if (role_status != cudaSuccess || *violations != 0U) {
        std::cerr << "role-separated DSM pipeline failed: "
                  << cudaGetErrorString(role_status)
                  << " violations=" << *violations << '\n';
        return EXIT_FAILURE;
    }
    for (std::uint32_t slice = 0; slice < RolePlan::kSlices; ++slice) {
        std::uint64_t expected = 0U;
        for (std::uint32_t byte = 0;
             byte < RolePlan::kValuesPerSlot; ++byte) {
            expected += static_cast<std::uint8_t>(
                (slice * 29U + byte * 7U) & 0xffU);
        }
        for (std::uint32_t byte = 0;
             byte < RolePlan::kScalesPerSlot; ++byte) {
            expected += static_cast<std::uint8_t>(
                (slice * 13U + byte * 3U) & 0xffU);
        }
        if (checksums[slice] != expected) {
            std::cerr << "role-separated checksum mismatch at slice "
                      << slice << " expected=" << expected
                      << " actual=" << checksums[slice] << '\n';
            return EXIT_FAILURE;
        }
    }
    std::cout << "stream1_transformer_sm120_nvfp4_role_separated_dsm=pass"
              << " cluster_ctas=" << RolePlan::kClusterCtas
              << " slices=" << RolePlan::kSlices
              << " ring_slots=" << RolePlan::kRingSlots
              << " transaction_bytes=" << RolePlan::kTransactionBytes
              << " hot_loop_cluster_syncs="
              << RolePlan::kHotLoopClusterSyncs
              << " violations=" << *violations << '\n';
    cudaFree(checksums);
    cudaFree(violations);
    return EXIT_SUCCESS;
}
