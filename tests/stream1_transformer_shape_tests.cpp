#include "stream1_transformer_shape.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

int main() {
    const auto p51 = beam::make_stream1_transformer_sequence_plan(51U, 16U);
    require(p51.logical_seq_len == 51U && p51.padded_seq_len == 64U && p51.alignment == 16U, "51 shape");
    const auto p57 = beam::make_stream1_transformer_sequence_plan(57U, 16U);
    require(p57.logical_seq_len == 57U && p57.padded_seq_len == 64U, "57 shape");
    const auto p64 = beam::make_stream1_transformer_sequence_plan(64U, 16U);
    require(p64.padded_seq_len == 64U, "64 shape");
    bool zero_logical = false;
    try { (void)beam::make_stream1_transformer_sequence_plan(0U, 16U); } catch (const std::invalid_argument&) { zero_logical = true; }
    require(zero_logical, "zero logical must fail");
    bool zero_alignment = false;
    try { (void)beam::make_stream1_transformer_sequence_plan(57U, 0U); } catch (const std::invalid_argument&) { zero_alignment = true; }
    require(zero_alignment, "zero alignment must fail");
    bool overflow = false;
    try { (void)beam::make_stream1_transformer_sequence_plan(std::numeric_limits<std::uint32_t>::max(), 16U); } catch (const std::overflow_error&) { overflow = true; }
    require(overflow, "overflow must fail");
    return 0;
}
