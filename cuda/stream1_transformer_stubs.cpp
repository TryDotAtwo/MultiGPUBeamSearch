#include "stream1.hpp"

#include <stdexcept>

namespace beam {

namespace {

[[noreturn]] void native_transformer_disabled() {
    throw std::runtime_error(
        "native CUTLASS Stream1 transformer backend was disabled at build time");
}

}  // namespace

void stream1_transformer_inference_cuda(
    const State128*,
    const std::uint64_t*,
    const std::uint32_t*,
    const Stream1TransformerNetworkView&,
    const Stream1TransformerScratchView&,
    std::uint32_t*,
    std::uint32_t,
    std::uint32_t,
    cudaStream_t) {
    native_transformer_disabled();
}

void stream1_transformer_inference_graph_job_cuda(
    const State128*,
    const std::uint64_t*,
    const std::uint32_t*,
    const std::uint32_t*,
    const Stream1TransformerNetworkView&,
    const Stream1TransformerScratchView&,
    std::uint32_t*,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t,
    cudaStream_t) {
    native_transformer_disabled();
}

}  // namespace beam
