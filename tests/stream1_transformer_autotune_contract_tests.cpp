#include "stream1_transformer_score_dump.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace beam;

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void require_invalid(Fn&& fn, const char* message) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, message);
}

} // namespace

int main() {
    const Stream1TransformerScoreDumpHeader header =
        make_stream1_transformer_score_dump_header(2U, 512U * 24U);
    require(header.magic == STREAM1_TRANSFORMER_SCORE_DUMP_MAGIC, "score dump magic mismatch");
    require(header.version == STREAM1_TRANSFORMER_SCORE_DUMP_VERSION, "score dump version mismatch");
    require(header.lanes == 2U, "score dump lane count mismatch");
    require(header.values_per_lane == 512U * 24U, "score dump value count mismatch");
    require(stream1_transformer_score_dump_value_count(header) == 2ULL * 512ULL * 24ULL,
            "score dump total value count mismatch");

    require_invalid(
        [] { static_cast<void>(make_stream1_transformer_score_dump_header(0U, 1U)); },
        "zero lanes must be rejected");
    require_invalid(
        [] { static_cast<void>(make_stream1_transformer_score_dump_header(1U, 0U)); },
        "zero values per lane must be rejected");

    Stream1TransformerScoreDumpHeader overflow = header;
    overflow.lanes = std::numeric_limits<std::uint32_t>::max();
    overflow.values_per_lane = std::numeric_limits<std::uint64_t>::max();
    require_invalid(
        [&] { static_cast<void>(stream1_transformer_score_dump_value_count(overflow)); },
        "score dump value count overflow must be rejected");

    std::cout << "stream1_transformer_autotune_contract_tests=pass\n";
    return 0;
}
