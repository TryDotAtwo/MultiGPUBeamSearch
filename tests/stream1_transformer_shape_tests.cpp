#define BEAM_STREAM1_WEIGHT_IO_MANIFEST_ONLY
#include "stream1_transformer_shape.hpp"
#include "../tools/stream1_weight_io.hpp"
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
    beam::Stream1ModelConfig model51{};
    model51.backend = beam::STREAM1_BACKEND_PIECE_TRANSFORMER;
    model51.seq_len = 51U;
    model51.d_model = 256U;
    model51.nhead = 8U;
    model51.head_dim = 32U;
    model51.ff_dim = 1024U;
    model51.output_dim = 24U;
    const auto scratch51 = beam::stream1_weights::transformer_scratch_byte_plan(model51, 8U);
    require(scratch51.logical_seq_len == 51U && scratch51.padded_seq_len == 64U, "51 scratch shape");
    require(scratch51.token_bytes == beam::stream1_weights::fp16_bytes(8ULL * 64ULL * 256ULL), "51 token bytes");
    auto model57 = model51;
    model57.seq_len = 57U;
    const auto scratch57 = beam::stream1_weights::transformer_scratch_byte_plan(model57, 8U);
    require(scratch57.logical_seq_len == 57U && scratch57.padded_seq_len == 64U, "57 scratch shape");
    require(scratch57.qkv_bytes == beam::stream1_weights::fp16_bytes(8ULL * 64ULL * 3ULL * 256ULL), "57 qkv bytes");
    require(beam::stream1_weights::transformer_attention_score_stride(model51) ==
            beam::stream1_weights::transformer_attention_score_stride(model57), "aligned attention stride");
    require(scratch51.total_bytes() == scratch57.total_bytes(), "same physical shape must use same scratch bytes");

    require(
        beam::stream1_transformer_supports_generic_final_cls_only(
            57U, 64U, 256U, 8U, 32U, 4U, 1024U, 24U),
        "cube4 output24 shape must support generic final CLS-only");
    require(
        !beam::stream1_transformer_supports_generic_final_cls_only(
            57U, 64U, 256U, 8U, 32U, 4U, 1024U, 1U),
        "output1 shape must not enter the output24 generic final CLS-only path");
    require(
        !beam::stream1_transformer_supports_generic_final_cls_only(
            65U, 80U, 256U, 8U, 32U, 4U, 1024U, 24U),
        "sequence longer than the fused attention limit must not enter generic final CLS-only");
    require(
        beam::stream1_transformer_supports_compact_sequence57(
            57U, 256U, 8U, 32U, 4U, 1024U, 24U),
        "exact Cube4 output24 shape must support compact sequence57");
    require(
        !beam::stream1_transformer_supports_compact_sequence57(
            51U, 256U, 8U, 32U, 4U, 1024U, 24U),
        "Megaminx sequence must not enter the Cube4 compact sequence57 path");
    require(
        !beam::stream1_transformer_supports_compact_sequence57(
            57U, 256U, 8U, 32U, 4U, 1024U, 1U),
        "output1 must not enter the output24 compact sequence57 path");

    const auto compact57 = beam::make_stream1_transformer_sequence_plan(57U, 1U);
    require(compact57.padded_seq_len == 57U, "compact sequence57 must not allocate padded token rows");

    const auto tail57 = beam::make_stream1_transformer_padding_tail_plan(8U, 57U, 64U, 256U);
    require(tail57.row_count == 8U, "tail plan rows");
    require(tail57.tail_tokens_per_row == 7U, "tail plan token count");
    require(tail57.tail_elements == 14336ULL, "tail plan must launch only logical padding elements");
    require(tail57.row_stride_elements == 16384ULL, "tail plan must retain padded row stride");
    const auto no_tail = beam::make_stream1_transformer_padding_tail_plan(8U, 64U, 64U, 256U);
    require(no_tail.tail_elements == 0ULL, "aligned sequence must not launch a padding kernel");

    return 0;
}
