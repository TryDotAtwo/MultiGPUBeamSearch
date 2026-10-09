#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace beam {

constexpr std::uint64_t STREAM1_TRANSFORMER_SCORE_DUMP_MAGIC = 0x31504D5544533153ULL;
constexpr std::uint32_t STREAM1_TRANSFORMER_SCORE_DUMP_VERSION = 1U;

struct Stream1TransformerScoreDumpHeader {
    std::uint64_t magic = STREAM1_TRANSFORMER_SCORE_DUMP_MAGIC;
    std::uint32_t version = STREAM1_TRANSFORMER_SCORE_DUMP_VERSION;
    std::uint32_t lanes = 0U;
    std::uint64_t values_per_lane = 0U;
};

static_assert(sizeof(Stream1TransformerScoreDumpHeader) == 24U);

inline std::uint64_t stream1_transformer_score_dump_value_count(
    const Stream1TransformerScoreDumpHeader& header) {
    if (header.magic != STREAM1_TRANSFORMER_SCORE_DUMP_MAGIC ||
        header.version != STREAM1_TRANSFORMER_SCORE_DUMP_VERSION ||
        header.lanes == 0U || header.values_per_lane == 0U) {
        throw std::invalid_argument("invalid Stream1 transformer score dump header");
    }
    if (header.values_per_lane >
        std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(header.lanes)) {
        throw std::invalid_argument("Stream1 transformer score dump value count overflow");
    }
    return static_cast<std::uint64_t>(header.lanes) * header.values_per_lane;
}

inline Stream1TransformerScoreDumpHeader make_stream1_transformer_score_dump_header(
    std::uint32_t lanes,
    std::uint64_t values_per_lane) {
    Stream1TransformerScoreDumpHeader header{};
    header.lanes = lanes;
    header.values_per_lane = values_per_lane;
    static_cast<void>(stream1_transformer_score_dump_value_count(header));
    return header;
}

} // namespace beam
