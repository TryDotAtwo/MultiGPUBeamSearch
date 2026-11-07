#include "stream1_transformer_sm120_nvfp4_ffn_policy.hpp"
#include "stream1_transformer_sm120_nvfp4_ffn.hpp"
#include "stream1_transformer_sm120_nvfp4_stock.hpp"

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
    require(
        Stream1Sm120Nvfp4FfnPolicy::kStockMainloopKTile == 128U,
        "measured SM120 stock NVFP4 production mainloop must use K128");
    require(
        Stream1Sm120Nvfp4FfnPolicy::kFusedLn2Threads == 128U &&
            Stream1Sm120Nvfp4FfnPolicy::kFusedLn2ValuesPerThread == 2U &&
            Stream1Sm120Nvfp4FfnPolicy::kFusedLn2ThreadsPerScale == 8U,
        "fused LN2 producer must map one adjacent FP16 pair per thread and one K16 scale per 8-thread subgroup");
    require(
        Stream1Sm120Nvfp4FfnPolicy::kFusedLn2Threads *
                Stream1Sm120Nvfp4FfnPolicy::kFusedLn2ValuesPerThread ==
            Stream1Sm120Nvfp4FfnPolicy::kDModel,
        "fused LN2 producer must cover the complete K256 row exactly once");
    require(
        STREAM1_SM120_NVFP4_FFN_VALID_LAYER_MASK == 0x7U &&
            stream1_sm120_nvfp4_ffn_bit(3U) == 0U,
        "only complete non-final Cube4 FFN layers may be selected");
    const auto memory = stream1_sm120_nvfp4_ffn_memory_plan(64U);
    require(memory.global_hidden_bytes == 0U, "fused FFN must not materialize global hidden");
    require(memory.output_token_bytes == 64U * 256U * 2U, "FP16 residual output bytes mismatch");
    require(memory.next_activation_bytes == 64U * 256U / 2U, "packed E2M1 bytes mismatch");
    require(memory.next_scale_bytes == 64U * 16U, "UE4M3 K16 scale bytes mismatch");
    require(
        stream1_sm120_nvfp4_weight_value_bytes(256U, 1024U) == 131072U,
        "Cube4 FF1 packed E2M1 weight bytes mismatch");
    require(
        stream1_sm120_nvfp4_weight_scale_bytes(256U, 1024U) == 65536U,
        "Cube4 FF1 tiled UE4M3 scale bytes mismatch");
    require(
        stream1_sm120_nvfp4_weight_value_bytes(1024U, 256U) == 131072U,
        "Cube4 FF2 packed E2M1 weight bytes mismatch");
    require(
        stream1_sm120_nvfp4_weight_scale_bytes(1024U, 256U) == 65536U,
        "Cube4 FF2 tiled UE4M3 scale bytes mismatch");
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
    constexpr std::uint32_t production_rows = 384U * 64U;
    constexpr auto stock = stream1_sm120_nvfp4_stock_scratch_plan(production_rows);
    require(
        stock.input_value_bytes == 3145728U &&
            stock.input_scale_bytes == 1572864U &&
            stock.input_total_bytes == 4718592U,
        "M24576xK256 stock NVFP4 activation scratch mismatch");
    require(
        stock.hidden_value_bytes == 12582912U &&
            stock.hidden_scale_bytes == 6291456U &&
            stock.hidden_total_bytes == 18874368U,
        "M24576xK1024 stock NVFP4 hidden scratch mismatch");
    require(
        stock.input_scale_offset % 128U == 0U &&
            stock.hidden_scale_offset % 128U == 0U,
        "stock NVFP4 scale buffers must keep CUTLASS 128-byte alignment");
    require(
        stock.input_total_bytes <=
            static_cast<std::size_t>(production_rows) * 3U * 256U * sizeof(std::uint16_t),
        "stock input NVFP4 materialization must fit the retired QKV scratch");
    require(
        stock.input_total_bytes <=
            static_cast<std::size_t>(production_rows) * 1024U * sizeof(std::uint16_t),
        "fused LN1-QKV activation scratch must fit the reusable FF hidden allocation");
    require(
        stream1_sm120_nvfp4_weight_value_bytes(256U, 768U) == 98304U &&
            stream1_sm120_nvfp4_weight_scale_bytes(256U, 768U) == 49152U,
        "Cube4 fused QKV offline NVFP4 weight footprint mismatch");
    require(
        stock.hidden_total_bytes <=
            static_cast<std::size_t>(production_rows) * 1024U * sizeof(std::uint16_t),
        "stock hidden NVFP4 materialization must fit the FP16 FF hidden scratch");
    using Persistent = Stream1Sm120Nvfp4PersistentFfnPipeline;
    require(
        Persistent::kRowsPerCta == 128U && Persistent::kThreads == 256U,
        "persistent FFN must keep one M128 tile resident in one 8-warp CTA");
    require(
        Persistent::kProducerWarps == 4U && Persistent::kConsumerWarps == 4U,
        "persistent FFN must split the CTA into equal producer and consumer warp groups");
    require(
        Persistent::kHiddenSlices == 8U && Persistent::kRingBuffers == 2U,
        "FF1 K128 slices must stream through exactly two physical handoff buffers");
    require(
        Persistent::kReadyBarriers == Persistent::kRingBuffers &&
            Persistent::kFreeBarriers == Persistent::kRingBuffers,
        "each handoff buffer must have independent ready and free ownership barriers");
    require(
        Persistent::kGlobalHiddenBytes == 0U &&
            Persistent::kHandoffBytes == 18432U,
        "persistent handoff must use 18 KiB CTA storage and no global hidden tensor");
    for (std::uint32_t slice = 0; slice < Persistent::kHiddenSlices; ++slice) {
        require(
            Persistent::producer_buffer(slice) == slice % 2U,
            "producer must alternate physical handoff buffers");
        require(
            Persistent::consumer_buffer(Persistent::consumer_stage(slice)) == slice % 2U,
            "consumer must read the same buffer generation published by the producer");
    }

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
