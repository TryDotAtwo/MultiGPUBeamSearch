#pragma once
#include "stream1.hpp"
namespace beam {
void stream1_transformer_build_input_layernorm256_generic_launch(
    const State128*,const std::uint64_t*,const std::uint32_t*,const std::uint32_t*,
    const Stream1TransformerNetworkView&,half*,std::uint32_t,std::uint32_t,cudaStream_t);
void stream1_transformer_build_input_dual_ln_cuda(
    const State128* states,const std::uint64_t* base,const std::uint32_t* count,const std::uint32_t* job,
    const Stream1TransformerNetworkView& network,half* tokens,half* normalized,
    std::uint32_t batch,std::uint32_t offset,cudaStream_t stream);
}
