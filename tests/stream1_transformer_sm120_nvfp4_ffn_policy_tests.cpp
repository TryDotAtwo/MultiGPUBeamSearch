#include "stream1_transformer_sm120_nvfp4_ffn_policy.hpp"
#include "stream1_transformer_sm120_nvfp4_ffn.hpp"

#include <cstdlib>
#include <iostream>

namespace {

Stream1Sm120Nvfp4FfnContract cube4_contract() {
    Stream1Sm120Nvfp4FfnContract contract{};
    contract.target_sm = 120U;
    contract.d_model = 256U;
    contract.ff_dim = 1024U;
    contract.seq_len = 57U;
    contract.padded_seq_len = 64U;
    contract.output_dim = 24U;
    contract.dtype = 0U;
    contract.activation = 0U;
    contract.has_next_layer = true;
    contract.weights_offline_immutable = true;
    return contract;
}

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    const auto accepted = cube4_contract();
    require(Stream1Sm120Nvfp4FfnPolicy::eligible(accepted), "Cube4 contract must be eligible");
    require(Stream1Sm120Nvfp4FfnPolicy::kScaleVector == 16U, "NVFP4 scale vector must be K16");
    const auto memory = stream1_sm120_nvfp4_ffn_memory_plan(64U);
    require(memory.global_hidden_bytes == 0U, "fused FFN must not materialize global hidden");
    require(memory.output_token_bytes == 64U * 256U * 2U, "FP16 residual output bytes mismatch");
    require(memory.next_activation_bytes == 64U * 256U / 2U, "packed E2M1 bytes mismatch");
    require(memory.next_scale_bytes == 64U * 16U, "UE4M3 K16 scale bytes mismatch");
    require(
        Stream1Sm120Nvfp4FfnPolicy::kProducerTiles *
            Stream1Sm120Nvfp4FfnPolicy::kColumnsPerProducerTile ==
            Stream1Sm120Nvfp4FfnPolicy::kFfDim,
        "eight sequential N128 producer tiles must own the complete hidden row");
    require(
        Stream1Sm120Nvfp4FfnPolicy::kProducerTileBuffers == 2U,
        "producer and consumers require ping-pong hidden slices");
    require(
        Stream1Sm120Nvfp4FfnPolicy::kSharedHiddenValueBytesPerTile == 8192U &&
            Stream1Sm120Nvfp4FfnPolicy::kSharedHiddenScaleBytesPerTile == 1024U,
        "one M128xK128 NVFP4 hidden slice must use 9 KiB shared memory");
    require(
        Stream1Sm120Nvfp4FfnPolicy::kSharedHiddenValueBytes == 16384U &&
            Stream1Sm120Nvfp4FfnPolicy::kSharedHiddenScaleBytes == 2048U,
        "double-buffered M128xK128 hidden slices must use 18 KiB shared memory");
    require(
        Stream1Sm120Nvfp4FfnPolicy::kConsumerGroups *
            Stream1Sm120Nvfp4FfnPolicy::kColumnsPerConsumer ==
            Stream1Sm120Nvfp4FfnPolicy::kDModel,
        "two N128 consumer groups must own the complete output row");

    auto rejected = accepted;
    rejected.activation = 1U;
    require(
        Stream1Sm120Nvfp4FfnPolicy::reject_reason(rejected) ==
            Stream1Sm120Nvfp4FfnRejectReason::WrongActivation,
        "SiLU must fail closed");
    rejected = accepted;
    rejected.output_dim = 1U;
    require(
        Stream1Sm120Nvfp4FfnPolicy::reject_reason(rejected) ==
            Stream1Sm120Nvfp4FfnRejectReason::WrongOutputHead,
        "output_dim=1 must fail closed");
    rejected = accepted;
    rejected.has_next_layer = false;
    require(
        Stream1Sm120Nvfp4FfnPolicy::reject_reason(rejected) ==
            Stream1Sm120Nvfp4FfnRejectReason::NoNextLayer,
        "next-layer packing must not be used for the final block");
    rejected = accepted;
    rejected.weights_offline_immutable = false;
    require(
        Stream1Sm120Nvfp4FfnPolicy::reject_reason(rejected) ==
            Stream1Sm120Nvfp4FfnRejectReason::MutableWeights,
        "runtime weight quantization must fail closed");

    std::cout << "stream1_transformer_sm120_nvfp4_ffn_policy_tests=pass\n";
    return EXIT_SUCCESS;
}
