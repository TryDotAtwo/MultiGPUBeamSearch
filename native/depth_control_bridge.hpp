#pragma once
#include "beam_native.h"
#include "engine_control.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace beam::native {
class DepthControlBridge {
    beam_depth_control_api_v1 api_{};

public:
    DepthControlBridge() {
        if (const auto* api = current_depth_control()) api_ = *api;
    }
    explicit operator bool() const { return api_.user_data != nullptr; }

    bool continue_after(
        std::uint32_t checkpoint,
        bool calibration_limit_reached,
        bool frontier_live,
        std::uint32_t tracked_path_stop) const {
        if (!*this) throw std::logic_error("Rust depth controller is not installed");
        beam_depth_control_input_v1 input{
            BEAM_NATIVE_ABI_VERSION,
            sizeof(beam_depth_control_input_v1),
            checkpoint,
            calibration_limit_reached ? 1U : 0U,
            frontier_live ? 1U : 0U,
            tracked_path_stop};
        beam_depth_control_output_v1 output{
            BEAM_NATIVE_ABI_VERSION,
            sizeof(beam_depth_control_output_v1),
            0U};
        const std::int32_t result = api_.decide(api_.user_data, &input, &output);
        if (result != BEAM_OK) {
            std::uint64_t required = 0;
            if (api_.error(api_.user_data, nullptr, 0, &required) != BEAM_BUFFER_TOO_SMALL ||
                required == 0 || required > (1U << 20U)) {
                throw std::runtime_error("Rust depth-control callback failed");
            }
            std::vector<char> message(static_cast<std::size_t>(required), 0);
            if (api_.error(api_.user_data, message.data(), message.size(), &required) != BEAM_OK) {
                throw std::runtime_error("Rust depth-control error retrieval failed");
            }
            throw std::runtime_error(std::string("Rust depth control: ") + message.data());
        }
        if (output.abi_version != BEAM_NATIVE_ABI_VERSION ||
            output.struct_size < sizeof(output) || output.continue_search > 1U) {
            throw std::runtime_error("Rust depth controller returned an invalid decision");
        }
        return output.continue_search != 0U;
    }
};
static_assert(sizeof(beam_depth_control_input_v1) == 24);
static_assert(sizeof(beam_depth_control_output_v1) == 12);
static_assert(sizeof(beam_depth_control_api_v1) == 32);
}
