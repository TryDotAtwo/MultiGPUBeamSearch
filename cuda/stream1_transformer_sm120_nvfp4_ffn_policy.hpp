#pragma once

#include "../src/config.hpp"

#include <cstdint>

enum class Stream1Sm120Nvfp4FfnRejectReason : std::uint32_t {
    None = 0,
    WrongTarget,
    WrongModelShape,
    WrongSequenceShape,
    WrongOutputHead,
    WrongDtype,
    WrongActivation,
    NoNextLayer,
    MutableWeights,
};

struct Stream1Sm120Nvfp4FfnContract {
    std::uint32_t target_sm = 0;
    std::uint32_t d_model = 0;
    std::uint32_t ff_dim = 0;
    std::uint32_t seq_len = 0;
    std::uint32_t padded_seq_len = 0;
    std::uint32_t output_dim = 0;
    std::uint32_t dtype = 0;
    std::uint32_t activation = 0;
    bool has_next_layer = false;
    bool weights_offline_immutable = false;
};

struct Stream1Sm120Nvfp4FfnPolicy {
    static constexpr std::uint32_t kTargetSm = 120U;
    static constexpr std::uint32_t kDModel = 256U;
    static constexpr std::uint32_t kFfDim = 1024U;
    static constexpr std::uint32_t kOutputDim = 24U;
    static constexpr std::uint32_t kFp16Dtype = beam::STREAM1_DTYPE_FP16;
    static constexpr std::uint32_t kReluActivation = beam::STREAM1_ACTIVATION_RELU;
    static constexpr std::uint32_t kScaleVector = 16U;
    static constexpr std::uint32_t kFusedLn2Threads = 128U;
    static constexpr std::uint32_t kFusedLn2ValuesPerThread = 2U;
    static constexpr std::uint32_t kFusedLn2ThreadsPerScale =
        kScaleVector / kFusedLn2ValuesPerThread;
    // Interleaved Molab SM120 measurements selected K128 over K256 for the
    // stock CUTLASS FF1->FF2 production control path.
    static constexpr std::uint32_t kStockMainloopKTile = 128U;
    static constexpr std::uint32_t kProducerTiles = 8U;
    static constexpr std::uint32_t kColumnsPerProducerTile = 128U;
    static constexpr std::uint32_t kProducerTileBuffers = 2U;
    static constexpr std::uint32_t kConsumerGroups = 2U;
    static constexpr std::uint32_t kColumnsPerConsumer = 128U;
    static constexpr std::uint32_t kInitialRowsPerCta = 128U;
    static constexpr std::uint32_t kSharedHiddenValueBytesPerTile =
        kInitialRowsPerCta * kColumnsPerProducerTile / 2U;
    static constexpr std::uint32_t kSharedHiddenScaleBytesPerTile =
        kInitialRowsPerCta * (kColumnsPerProducerTile / kScaleVector);
    static constexpr std::uint32_t kSharedHiddenValueBytes =
        kProducerTileBuffers * kSharedHiddenValueBytesPerTile;
    static constexpr std::uint32_t kSharedHiddenScaleBytes =
        kProducerTileBuffers * kSharedHiddenScaleBytesPerTile;

    static constexpr Stream1Sm120Nvfp4FfnRejectReason reject_reason(
        const Stream1Sm120Nvfp4FfnContract& contract) {
        if (contract.target_sm != kTargetSm) {
            return Stream1Sm120Nvfp4FfnRejectReason::WrongTarget;
        }
        if (contract.d_model != kDModel || contract.ff_dim != kFfDim) {
            return Stream1Sm120Nvfp4FfnRejectReason::WrongModelShape;
        }
        if (contract.seq_len == 0U || contract.padded_seq_len < contract.seq_len ||
            contract.padded_seq_len > 64U) {
            return Stream1Sm120Nvfp4FfnRejectReason::WrongSequenceShape;
        }
        if (contract.output_dim != kOutputDim) {
            return Stream1Sm120Nvfp4FfnRejectReason::WrongOutputHead;
        }
        if (contract.dtype != kFp16Dtype) {
            return Stream1Sm120Nvfp4FfnRejectReason::WrongDtype;
        }
        if (contract.activation != kReluActivation) {
            return Stream1Sm120Nvfp4FfnRejectReason::WrongActivation;
        }
        if (!contract.has_next_layer) {
            return Stream1Sm120Nvfp4FfnRejectReason::NoNextLayer;
        }
        if (!contract.weights_offline_immutable) {
            return Stream1Sm120Nvfp4FfnRejectReason::MutableWeights;
        }
        return Stream1Sm120Nvfp4FfnRejectReason::None;
    }

    static constexpr bool eligible(const Stream1Sm120Nvfp4FfnContract& contract) {
        return reject_reason(contract) == Stream1Sm120Nvfp4FfnRejectReason::None;
    }
};

// Ownership contract for the SM120 persistent FF1 -> ReLU/NVFP4 -> FF2
// handoff.  A producer publishes one K128 hidden slice into a two-generation
// CTA-local ring; the consumer releases that generation before the producer
// may reuse it.  No hidden payload is addressable in global memory.
struct Stream1Sm120Nvfp4PersistentFfnPipeline {
    static constexpr std::uint32_t kRowsPerCta = 128U;
    static constexpr std::uint32_t kThreads = 256U;
    static constexpr std::uint32_t kProducerWarps = 4U;
    static constexpr std::uint32_t kConsumerWarps = 4U;
    static constexpr std::uint32_t kHiddenSlices = 8U;
    static constexpr std::uint32_t kRingBuffers = 2U;
    static constexpr std::uint32_t kReadyBarriers = kRingBuffers;
    static constexpr std::uint32_t kFreeBarriers = kRingBuffers;
    static constexpr std::uint32_t kValuesPerSlice = kRowsPerCta * 128U;
    static constexpr std::uint32_t kValueBytesPerBuffer = kValuesPerSlice / 2U;
    static constexpr std::uint32_t kScaleBytesPerBuffer =
        kRowsPerCta * (128U / Stream1Sm120Nvfp4FfnPolicy::kScaleVector);
    static constexpr std::uint32_t kHandoffBytes =
        kRingBuffers * (kValueBytesPerBuffer + kScaleBytesPerBuffer);
    static constexpr std::uint32_t kGlobalHiddenBytes = 0U;

    static constexpr std::uint32_t producer_buffer(std::uint32_t slice) {
        return slice % kRingBuffers;
    }

    static constexpr std::uint32_t consumer_stage(std::uint32_t slice) {
        return slice + 1U;
    }

    static constexpr std::uint32_t consumer_buffer(std::uint32_t stage) {
        return (stage - 1U) % kRingBuffers;
    }
};

inline constexpr std::uint32_t stream1_sm120_nvfp4_ffn_bit(std::uint32_t layer) {
    return layer < 3U ? (1U << layer) : 0U;
}

inline constexpr std::uint32_t STREAM1_SM120_NVFP4_FFN_VALID_LAYER_MASK =
    stream1_sm120_nvfp4_ffn_bit(0U) |
    stream1_sm120_nvfp4_ffn_bit(1U) |
    stream1_sm120_nvfp4_ffn_bit(2U);
