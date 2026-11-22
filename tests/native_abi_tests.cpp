#include "beam_native.h"
#include "depth_control_bridge.hpp"
#include "engine_control.hpp"
#include "history_bridge.hpp"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

struct DepthProbe {
    std::uint32_t calls = 0;
};

extern "C" int32_t depth_decide(
    void* user,
    const beam_depth_control_input_v1* input,
    beam_depth_control_output_v1* output) {
    if (!user || !input || !output || input->abi_version != 1 ||
        input->struct_size != sizeof(*input) || output->abi_version != 1 ||
        output->struct_size != sizeof(*output)) {
        return BEAM_INVALID;
    }
    ++static_cast<DepthProbe*>(user)->calls;
    if (input->checkpoint == BEAM_DEPTH_BEFORE_FRONTIER_COLLECTIVE) {
        output->continue_search = input->calibration_limit_reached == 0;
    } else if (input->checkpoint == BEAM_DEPTH_AFTER_FRONTIER_COLLECTIVE) {
        output->continue_search = input->frontier_live != 0 && input->tracked_path_stop == 0;
    } else {
        return BEAM_INVALID;
    }
    return BEAM_OK;
}

extern "C" int32_t depth_decide_v2(
    void* user,
    const beam_depth_control_input_v2* input,
    beam_depth_control_output_v2* output) {
    if (!user || !input || !output || input->abi_version != 2 ||
        input->struct_size != sizeof(*input) || output->abi_version != 2 ||
        output->struct_size != sizeof(*output)) {
        return BEAM_INVALID;
    }
    ++static_cast<DepthProbe*>(user)->calls;
    if (input->checkpoint == BEAM_DEPTH_V2_BEFORE_FRONTIER_COLLECTIVE) {
        output->continue_search = input->calibration_limit_reached == 0;
    } else if (input->checkpoint == BEAM_DEPTH_V2_AFTER_FRONTIER_COLLECTIVE) {
        output->continue_search = input->frontier_live != 0 && input->tracked_path_stop == 0;
    } else if (input->checkpoint == BEAM_DEPTH_V2_SOLVE_BUCKET_STOP) {
        if (input->max_solutions != 0 && input->accepted_solutions >= input->max_solutions) {
            output->stop_reason = static_cast<std::uint32_t>(BEAM_DEPTH_STOP_REASON_CAPACITY);
        } else if (input->stop_depth != 0 && input->completed_depths >= input->stop_depth) {
            output->stop_reason = static_cast<std::uint32_t>(BEAM_DEPTH_STOP_REASON_DEPTH);
        } else if (input->stop_depth == 0 && input->solve_bucket_found_any != 0 &&
                   input->current_depth_index >=
                       input->first_found_depth_index + input->extra_depths) {
            output->stop_reason = static_cast<std::uint32_t>(BEAM_DEPTH_STOP_REASON_DEPTH);
        }
        output->continue_search = output->stop_reason ==
            static_cast<std::uint32_t>(BEAM_DEPTH_STOP_REASON_NONE);
    } else {
        return BEAM_INVALID;
    }
    return BEAM_OK;
}

extern "C" int32_t depth_error(void*, char* out, uint64_t capacity, uint64_t* required) {
    if (!required) return BEAM_INVALID;
    *required = 1;
    if (capacity == 0) return BEAM_BUFFER_TOO_SMALL;
    if (!out) return BEAM_INVALID;
    out[0] = '\0';
    return BEAM_OK;
}

// Exercise the real ABI/control transport with a deterministic CPU engine.
// The production library supplies this symbol from its CUDA runner instead.
void beam_engine_layout(beam_layout_v1& layout) {
    layout = {1, sizeof(layout), 96, 112, 16, 24, 32, 32, 16, 16};
}
int run_production_runner(int argc, char** argv) {
    if(argc == 2 && std::string(argv[1]) == "fail-prepare")
        throw std::runtime_error("prepare failure preserved");
    auto* control = beam::native::current_control();
    if (argc == 2 && (std::string(argv[1]) == "depth-control-v1" ||
                      std::string(argv[1]) == "depth-control-v2")) {
        beam::native::DepthControlBridge depth_control;
        assert(depth_control);
        assert(!depth_control.continue_after(
            BEAM_DEPTH_BEFORE_FRONTIER_COLLECTIVE, true, false, 0));
        assert(depth_control.continue_after(
            BEAM_DEPTH_BEFORE_FRONTIER_COLLECTIVE, false, false, 0));
        assert(!depth_control.continue_after(
            BEAM_DEPTH_AFTER_FRONTIER_COLLECTIVE, false, false, 0));
        assert(!depth_control.continue_after(
            BEAM_DEPTH_AFTER_FRONTIER_COLLECTIVE, true, true, 1));
        assert(depth_control.continue_after(
            BEAM_DEPTH_AFTER_FRONTIER_COLLECTIVE, true, true, 0));
        if (std::string(argv[1]) == "depth-control-v2") {
            assert(depth_control.supports_solve_bucket());
            assert(depth_control.solve_bucket_stop_reason(0, 0, 1, 0, 0, 0, 0, false) ==
                   static_cast<std::uint32_t>(BEAM_DEPTH_STOP_REASON_NONE));
            assert(depth_control.solve_bucket_stop_reason(2, 2, 1, 0, 0, 0, 0, false) ==
                   static_cast<std::uint32_t>(BEAM_DEPTH_STOP_REASON_CAPACITY));
            assert(depth_control.solve_bucket_stop_reason(0, 0, 5, 4, 5, 4, 0, true) ==
                   static_cast<std::uint32_t>(BEAM_DEPTH_STOP_REASON_DEPTH));
        } else {
            assert(!depth_control.supports_solve_bucket());
        }
        return 0;
    }
    beam::native::HistoryBridge history;
    if (!control) {
        if(history) {
            history.initialize("{}");
            struct alignas(32) Meta {uint64_t lo,hi,parent;uint32_t score,route;};
            const Meta value{0,0,0,0,37};
            history.write(0,&value,1);
            assert(history.read(0,0).route==37);
            history.close();
        }
        return 7;
    }
    if (argc == 2 && std::string(argv[1]) == "expect-graph")
        assert(control->expected_graph_hash == std::string(64, 'a'));
    if(history) history.initialize("{}");
    beam::native::Job job;
    while(control->next(job)) {
        if(job.state.at(0) == 255) throw std::runtime_error("job failure preserved");
        if(job.state.at(0) == 254) {
            while(!control->cancelled()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            control->throw_if_cancelled();
        }
        if(job.state.at(0) == 253) std::this_thread::sleep_for(std::chrono::milliseconds(30));
        if(history) {
            struct alignas(32) Meta {uint64_t lo,hi,parent;uint32_t score,route;};
            const Meta value{0,0,job.sequence,0,37};
            history.write(static_cast<uint32_t>(job.sequence),&value,1);
            const auto restored=history.read(static_cast<uint32_t>(job.sequence),job.sequence);
            assert(restored.parent_idx==job.sequence && restored.route==37 && restored.pad==0);
            (void)history.stats();
        }
        control->complete(job.sequence);
    }
    if(history) history.close();
    return 0;
}
beam_context* prepared(const char* mode = "ok") {
    beam_context* ctx = nullptr;
    assert(beam_context_create(1, &ctx) == BEAM_OK);
    const char* args[] = {"test", mode};
    assert(beam_context_prepare(ctx, 2, args) == BEAM_OK);
    return ctx;
}
beam_job_v1 job(uint8_t* state, uint64_t sequence, const std::string& dir) {
    return {1, sizeof(beam_job_v1), sequence, state, 1, dir.data(), dir.size()};
}
int main() {
    beam_layout_v1 layout{};
    layout.abi_version = 1;
    layout.struct_size = sizeof(layout);
    assert(beam_native_layout(&layout) == BEAM_OK);
    assert(layout.state_len == 96 && layout.candidate_alignment == 32);
    const std::string dir = std::filesystem::temp_directory_path().string();
    beam_context* ctx = reinterpret_cast<beam_context*>(1);
    assert(beam_context_create(2, &ctx) == BEAM_INVALID && ctx == nullptr);
    assert(beam_context_create(1, nullptr) == BEAM_INVALID);
    assert(beam_context_create(1, &ctx) == BEAM_OK);
    DepthProbe depth_probe;
    beam_depth_control_api_v1 depth_api{
        1, sizeof(beam_depth_control_api_v1), &depth_probe, depth_decide, depth_error};
    assert(beam_context_set_depth_control(nullptr, &depth_api) == BEAM_INVALID);
    beam_depth_control_api_v1 invalid_depth_api = depth_api;
    invalid_depth_api.abi_version = 2;
    assert(beam_context_set_depth_control(ctx, &invalid_depth_api) == BEAM_INVALID);
    assert(beam_context_set_depth_control(ctx, &depth_api) == BEAM_OK);
    const char* depth_args_v1[] = {"test", "depth-control-v1"};
    const char* depth_args_v2[] = {"test", "depth-control-v2"};
    assert(beam_context_prepare(ctx, 2, depth_args_v1) == BEAM_OK);
    assert(beam_context_set_depth_control(ctx, &depth_api) == BEAM_BUSY);
    assert(beam_context_wait(ctx, 1000) == BEAM_OK);
    assert(depth_probe.calls == 5);
    assert(beam_context_destroy(ctx) == BEAM_OK);

    DepthProbe coarse_depth_probe;
    beam_depth_control_api_v1 coarse_depth_api{
        1, sizeof(beam_depth_control_api_v1), &coarse_depth_probe, depth_decide, depth_error};
    assert(beam_native_main_with_history_and_depth_control(
               2, depth_args_v1, nullptr, &coarse_depth_api) == 0);
    assert(coarse_depth_probe.calls == 5);

    ctx = nullptr;
    assert(beam_context_create(1, &ctx) == BEAM_OK);
    DepthProbe depth_probe_v2;
    beam_depth_control_api_v2 depth_api_v2{
        2, sizeof(beam_depth_control_api_v2), &depth_probe_v2, depth_decide_v2, depth_error};
    assert(beam_context_set_depth_control_v2(nullptr, &depth_api_v2) == BEAM_INVALID);
    beam_depth_control_api_v2 invalid_depth_api_v2 = depth_api_v2;
    invalid_depth_api_v2.abi_version = 1;
    assert(beam_context_set_depth_control_v2(ctx, &invalid_depth_api_v2) == BEAM_INVALID);
    assert(beam_context_set_depth_control_v2(ctx, &depth_api_v2) == BEAM_OK);
    assert(beam_context_prepare(ctx, 2, depth_args_v2) == BEAM_OK);
    assert(beam_context_set_depth_control_v2(ctx, &depth_api_v2) == BEAM_BUSY);
    assert(beam_context_wait(ctx, 1000) == BEAM_OK);
    assert(depth_probe_v2.calls == 8);
    assert(beam_context_destroy(ctx) == BEAM_OK);

    DepthProbe coarse_depth_probe_v2;
    beam_depth_control_api_v2 coarse_depth_api_v2{
        2, sizeof(beam_depth_control_api_v2), &coarse_depth_probe_v2,
        depth_decide_v2, depth_error};
    assert(beam_native_main_with_history_and_depth_control_v2(
               2, depth_args_v2, nullptr, &coarse_depth_api_v2) == 0);
    assert(coarse_depth_probe_v2.calls == 8);

    ctx = nullptr;
    assert(beam_context_create(1, &ctx) == BEAM_OK);
    std::string expected_hash(64, 'a');
    assert(beam_context_set_graph_hash(nullptr, expected_hash.data(), expected_hash.size()) == BEAM_INVALID);
    assert(beam_context_set_graph_hash(ctx, nullptr, 64) == BEAM_INVALID);
    assert(beam_context_set_graph_hash(ctx, expected_hash.data(), 63) == BEAM_INVALID);
    expected_hash[0] = 'A';
    assert(beam_context_set_graph_hash(ctx, expected_hash.data(), expected_hash.size()) == BEAM_INVALID);
    expected_hash[0] = 'a';
    assert(beam_context_set_graph_hash(ctx, expected_hash.data(), expected_hash.size()) == BEAM_OK);
    expected_hash[0] = 'b'; // the context must own a copy
    const char* graph_args[] = {"test", "expect-graph"};
    assert(beam_context_prepare(ctx, 2, graph_args) == BEAM_OK);
    assert(beam_context_set_graph_hash(ctx, expected_hash.data(), expected_hash.size()) == BEAM_BUSY);
    assert(beam_context_wait(ctx, 1000) == BEAM_OK);
    assert(beam_context_destroy(ctx) == BEAM_OK);
    ctx = prepared();
    beam_context* duplicate = nullptr;
    assert(beam_context_create(1, &duplicate) == BEAM_BUSY && duplicate == nullptr);
    assert(beam_context_wait(ctx, 1000) == BEAM_OK);
    uint8_t value = 253;
    auto task = job(&value, 0, dir);
    task.state_len = 0;
    assert(beam_context_submit(ctx, &task) == BEAM_INVALID);
    task.state_len = 1;
    assert(beam_context_submit(ctx, &task) == BEAM_OK);
    value = 255; // submit must have copied the borrowed view
    assert(beam_context_submit(ctx, &task) == BEAM_BUSY);
    assert(beam_context_wait(ctx, 1) == BEAM_TIMEOUT);
    assert(beam_context_wait(ctx, 1000) == BEAM_OK);
    beam_status_v1 status{};
    status.abi_version = 1;
    status.struct_size = sizeof(status);
    assert(beam_context_status(ctx, &status) == BEAM_OK);
    assert(status.phase == BEAM_READY && status.completed_jobs == 1);
    assert(beam_context_submit(ctx, &task) == BEAM_INVALID); // stale sequence
    value = 1; task.sequence = 1;
    assert(beam_context_submit(ctx, &task) == BEAM_OK);
    assert(beam_context_wait(ctx, 1000) == BEAM_OK);
    assert(beam_context_destroy(ctx) == BEAM_OK);

    ctx = prepared("fail-prepare");
    assert(beam_context_wait(ctx, 1000) == BEAM_FAILED);
    uint64_t required = 0;
    assert(beam_context_error(ctx, nullptr, 0, &required) == BEAM_BUFFER_TOO_SMALL);
    assert(required == 26);
    char message[64]{};
    assert(beam_context_error(ctx, message, sizeof message, &required) == BEAM_OK);
    assert(std::string(message) == "prepare failure preserved");
    assert(beam_context_destroy(ctx) == BEAM_OK);

    ctx = prepared();
    assert(beam_context_wait(ctx, 1000) == BEAM_OK);
    value = 255; task = job(&value, 0, dir);
    assert(beam_context_submit(ctx, &task) == BEAM_OK);
    assert(beam_context_wait(ctx, 1000) == BEAM_FAILED);
    assert(beam_context_error(ctx, message, sizeof message, &required) == BEAM_OK);
    assert(std::string(message) == "job failure preserved");
    assert(beam_context_destroy(ctx) == BEAM_OK);

    ctx = prepared();
    assert(beam_context_wait(ctx, 1000) == BEAM_OK);
    value = 254; task = job(&value, 0, dir);
    assert(beam_context_submit(ctx, &task) == BEAM_OK);
    assert(beam_context_cancel(ctx) == BEAM_OK);
    assert(beam_context_wait(ctx, 1000) == BEAM_CANCELLED);
    assert(beam_context_destroy(ctx) == BEAM_OK);
    assert(beam_context_destroy(nullptr) == BEAM_OK);
    const char* args[] = {"test"};
    assert(beam_native_main(1, args) == 7);
}
