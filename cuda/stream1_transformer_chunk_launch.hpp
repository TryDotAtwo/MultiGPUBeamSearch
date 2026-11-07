#pragma once
#include "stream1.hpp"
#include "stream1_chunk_plan.hpp"
#include "stream1_policy_snapshot.hpp"
#include "stream1_execution_contract.hpp"

namespace beam {
// Shared production entry for fixed and windowed jobs. No allocation/readback.
struct Stream1NoChunkObserver {
    void operator()(std::uint32_t, std::uint32_t) const noexcept {}
};
template<class Observer = Stream1NoChunkObserver>
inline void launch_stream1_transformer_chunks_cuda(
    const State128* frontier, const std::uint64_t* bases, const std::uint32_t* counts,
    const std::uint32_t* job_index, bool windowed,
    const Stream1TransformerNetworkView& network, const Stream1TransformerScratchView& scratch,
    std::uint32_t* scores, std::uint32_t outer, std::uint32_t inner, cudaStream_t stream,
    Observer observe = {}, const Stream1PolicySnapshot* policies = nullptr,
    const Stream1ExecutionContract* contract = nullptr) {
    if (contract && (outer!=contract->outer_microbatch() || inner!=contract->transformer_microbatch() ||
                     (windowed && !contract->graph_executor())))
        throw std::invalid_argument("Transformer chunk profile disagrees with execution contract");
    // Graph execution supports both fixed and windowed job addressing.
    // Only windowed addressing requires the graph executor; fixed addressing
    // is also used while capturing a non-windowed CUDA Graph.
    const auto& loaded_network=contract ? contract->network() : network;
    if (contract) policies=&contract->policies();
    std::optional<Stream1ResolvedGemmScope> gemm_scope;
    if (contract) gemm_scope.emplace(contract->gemm_choices());
    std::optional<Stream1ResolvedLaunchScope> launch_scope;
    if (contract) launch_scope.emplace(contract->launch_choices());
    std::optional<Stream1PolicyScope> policy_scope;
    if (policies) policy_scope.emplace(*policies);
    if (windowed != (job_index != nullptr))
        throw std::invalid_argument("Transformer chunk job addressing disagrees with metadata");
    for_each_stream1_transformer_chunk_observed(outer, inner, [&](std::uint32_t offset, std::uint32_t count) {
        if (windowed) {
            stream1_transformer_inference_graph_job_cuda(frontier, bases, counts, job_index,
                loaded_network, scratch, scores, count, outer, offset, stream);
        } else {
            stream1_transformer_inference_cuda(frontier, bases, counts, loaded_network, scratch,
                scores + static_cast<std::uint64_t>(offset) * MOVE_COUNT, count, offset, stream);
        }
    }, observe);
}
}
