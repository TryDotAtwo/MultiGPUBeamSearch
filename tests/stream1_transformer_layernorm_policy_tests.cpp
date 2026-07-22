#include "stream1_transformer_layernorm_policy.hpp"

#include <cassert>
#include <stdexcept>

int main() {
    using Policy = beam::Stream1TransformerLayerNormRowsPolicy;
    if (beam::parse_stream1_transformer_layernorm_rows_policy(nullptr) != Policy::RowPerBlock ||
        beam::parse_stream1_transformer_layernorm_rows_policy("") != Policy::RowPerBlock ||
        beam::parse_stream1_transformer_layernorm_rows_policy("row") != Policy::RowPerBlock ||
        beam::parse_stream1_transformer_layernorm_rows_policy("block2") != Policy::TwoRowsPerBlock ||
        beam::parse_stream1_transformer_layernorm_rows_policy("persistent") != Policy::PersistentRows) {
        throw std::runtime_error("LayerNorm rows policy parse mismatch");
    }
    bool rejected = false;
    try { static_cast<void>(beam::parse_stream1_transformer_layernorm_rows_policy("unknown")); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("unknown LayerNorm rows policy must fail closed");
    if (!beam::stream1_transformer_layernorm_copy_policy_supported(Policy::RowPerBlock) ||
        !beam::stream1_transformer_layernorm_copy_policy_supported(Policy::PersistentRows) ||
        beam::stream1_transformer_layernorm_copy_policy_supported(Policy::TwoRowsPerBlock)) {
        throw std::runtime_error("LayerNorm copy policy support contract mismatch");
    }
    if (beam::stream1_transformer_layernorm_persistent_blocks_per_sm(nullptr, 12) != 12 ||
        beam::stream1_transformer_layernorm_persistent_blocks_per_sm("", 12) != 12 ||
        beam::stream1_transformer_layernorm_persistent_blocks_per_sm("1", 12) != 1 ||
        beam::stream1_transformer_layernorm_persistent_blocks_per_sm("6", 12) != 6 ||
        beam::stream1_transformer_layernorm_persistent_blocks_per_sm("12", 12) != 12) {
        throw std::runtime_error("LayerNorm persistent blocks-per-SM parse mismatch");
    }
    for (const char* value : {"0", "13", "abc", "2x"}) {
        bool blocks_rejected = false;
        try {
            static_cast<void>(beam::stream1_transformer_layernorm_persistent_blocks_per_sm(value, 12));
        } catch (const std::invalid_argument&) {
            blocks_rejected = true;
        }
        if (!blocks_rejected) {
            throw std::runtime_error("invalid LayerNorm persistent blocks-per-SM must fail closed");
        }
    }
    static_assert(beam::STREAM1_TRANSFORMER_LN256_WARP_SLOTS == 4U);
#if STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
    static_assert(beam::STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS_ENABLED);
    static_assert(beam::STREAM1_TRANSFORMER_LN256_MEAN_SLOT >= beam::STREAM1_TRANSFORMER_LN256_WARP_SLOTS);
    static_assert(beam::STREAM1_TRANSFORMER_LN256_INV_STD_SLOT != beam::STREAM1_TRANSFORMER_LN256_MEAN_SLOT);
    static_assert(beam::STREAM1_TRANSFORMER_LN256_SHARED_FLOATS > beam::STREAM1_TRANSFORMER_LN256_INV_STD_SLOT);
#else
    static_assert(!beam::STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS_ENABLED);
    static_assert(beam::STREAM1_TRANSFORMER_LN256_MEAN_SLOT == 0U);
    static_assert(beam::STREAM1_TRANSFORMER_LN256_INV_STD_SLOT == 0U);
    static_assert(beam::STREAM1_TRANSFORMER_LN256_SHARED_FLOATS == 4U);
#endif
    return 0;
}