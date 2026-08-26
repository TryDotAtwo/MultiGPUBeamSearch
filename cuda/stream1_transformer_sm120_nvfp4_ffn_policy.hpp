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
