#pragma once
#include "beam_native.h"
#include "engine_control.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace beam::native {
class DepthControlBridge {
    beam_depth_control_api_v1 api_v1_{};
    beam_depth_control_api_v2 api_v2_{};

    using ErrorFn = std::int32_t (*)(void*, char*, std::uint64_t, std::uint64_t*);

    [[noreturn]] static void throw_callback_error(void* user_data, ErrorFn get_error) {
        std::uint64_t required = 0;
        if (get_error(user_data, nullptr, 0, &required) != BEAM_BUFFER_TOO_SMALL ||
            required == 0 || required > (1U << 20U)) {
            throw std::runtime_error("Rust depth-control callback failed");
        }
        std::vector<char> message(static_cast<std::size_t>(required), 0);
        if (get_error(user_data, message.data(), message.size(), &required) != BEAM_OK) {
            throw std::runtime_error("Rust depth-control error retrieval failed");
        }
        throw std::runtime_error(std::string("Rust depth control: ") + message.data());
    }

    beam_depth_control_output_v1 decide_v1(
        const beam_depth_control_input_v1& input) const {
        beam_depth_control_output_v1 output{
            BEAM_NATIVE_ABI_VERSION,
            sizeof(beam_depth_control_output_v1),
            0U};
        const std::int32_t result = api_v1_.decide(api_v1_.user_data, &input, &output);
        if (result != BEAM_OK) throw_callback_error(api_v1_.user_data, api_v1_.error);
        if (output.abi_version != BEAM_NATIVE_ABI_VERSION ||
            output.struct_size < sizeof(output) || output.continue_search > 1U) {
            throw std::runtime_error("Rust ABI-1 depth controller returned an invalid decision");
        }
        return output;
    }

    beam_depth_control_output_v2 decide_v2(
        const beam_depth_control_input_v2& input) const {
        beam_depth_control_output_v2 output{
            2U,
            sizeof(beam_depth_control_output_v2),
            0U,
            0U};
        const std::int32_t result = api_v2_.decide(api_v2_.user_data, &input, &output);
        if (result != BEAM_OK) throw_callback_error(api_v2_.user_data, api_v2_.error);
        if (output.abi_version != 2U || output.struct_size < sizeof(output) ||
            output.continue_search > 1U ||
            output.stop_reason > static_cast<std::uint32_t>(BEAM_DEPTH_STOP_REASON_CAPACITY)) {
            throw std::runtime_error("Rust ABI-2 depth controller returned an invalid decision");
        }
        return output;
    }

public:
    DepthControlBridge() {
        if (const auto* api = current_depth_control_v1()) api_v1_ = *api;
        if (const auto* api = current_depth_control_v2()) api_v2_ = *api;
    }
    explicit operator bool() const {
        return api_v1_.user_data != nullptr || api_v2_.user_data != nullptr;
    }
    bool supports_solve_bucket() const { return api_v2_.user_data != nullptr; }

    bool continue_after(
        std::uint32_t checkpoint,
        bool calibration_limit_reached,
        bool frontier_live,
        std::uint32_t tracked_path_stop) const {
        if (api_v2_.user_data != nullptr) {
            beam_depth_control_input_v2 input{};
            input.abi_version = 2U;
            input.struct_size = sizeof(input);
            input.checkpoint = checkpoint == BEAM_DEPTH_BEFORE_FRONTIER_COLLECTIVE
                ? BEAM_DEPTH_V2_BEFORE_FRONTIER_COLLECTIVE
                : BEAM_DEPTH_V2_AFTER_FRONTIER_COLLECTIVE;
            input.calibration_limit_reached = calibration_limit_reached ? 1U : 0U;
            input.frontier_live = frontier_live ? 1U : 0U;
            input.tracked_path_stop = tracked_path_stop;
            const auto output = decide_v2(input);
            if (output.stop_reason != BEAM_DEPTH_STOP_REASON_NONE) {
                throw std::runtime_error("Rust depth controller returned a solve-bucket reason");
            }
            return output.continue_search != 0U;
        }
        if (api_v1_.user_data == nullptr) {
            throw std::logic_error("Rust depth controller is not installed");
        }
        beam_depth_control_input_v1 input{
            BEAM_NATIVE_ABI_VERSION,
            sizeof(beam_depth_control_input_v1),
            checkpoint,
            calibration_limit_reached ? 1U : 0U,
            frontier_live ? 1U : 0U,
            tracked_path_stop};
        return decide_v1(input).continue_search != 0U;
    }

    std::uint32_t solve_bucket_stop_reason(
        std::uint64_t max_solutions,
        std::uint64_t accepted_solutions,
        std::uint32_t completed_depths,
        std::uint32_t current_depth_index,
        std::uint32_t stop_depth,
        std::uint32_t first_found_depth_index,
        std::uint32_t extra_depths,
        bool found_any) const {
        if (!supports_solve_bucket()) {
            throw std::logic_error("ABI-2 solve-bucket policy is not installed");
        }
        beam_depth_control_input_v2 input{};
        input.abi_version = 2U;
        input.struct_size = sizeof(input);
        input.checkpoint = BEAM_DEPTH_V2_SOLVE_BUCKET_STOP;
        input.max_solutions = max_solutions;
        input.accepted_solutions = accepted_solutions;
        input.completed_depths = completed_depths;
        input.current_depth_index = current_depth_index;
        input.stop_depth = stop_depth;
        input.first_found_depth_index = first_found_depth_index;
        input.extra_depths = extra_depths;
        input.solve_bucket_found_any = found_any ? 1U : 0U;
        const auto output = decide_v2(input);
        if ((output.stop_reason == BEAM_DEPTH_STOP_REASON_NONE) !=
            (output.continue_search != 0U)) {
            throw std::runtime_error("Rust depth controller returned inconsistent bucket state");
        }
        return output.stop_reason;
    }
};

static_assert(sizeof(beam_depth_control_input_v1) == 24);
static_assert(sizeof(beam_depth_control_output_v1) == 12);
static_assert(sizeof(beam_depth_control_api_v1) == 32);
static_assert(sizeof(beam_depth_control_input_v2) == 64);
static_assert(alignof(beam_depth_control_input_v2) == 8);
static_assert(sizeof(beam_depth_control_output_v2) == 16);
static_assert(sizeof(beam_depth_control_api_v2) == 32);
}
