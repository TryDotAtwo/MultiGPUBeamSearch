#include "beam_native.h"
#include "engine_control.hpp"
#include <cassert>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>

// Exercise the real ABI/control transport with a deterministic CPU engine.
// The production library supplies this symbol from its CUDA runner instead.
void beam_engine_layout(beam_layout_v1& layout) {
    layout = {1, sizeof(layout), 96, 112, 16, 24, 32, 32, 16, 16};
}
int run_production_runner(int argc, char** argv) {
    if(argc == 2 && std::string(argv[1]) == "fail-prepare")
        throw std::runtime_error("prepare failure preserved");
    auto* control = beam::native::current_control();
    if (!control) return 7;
    beam::native::Job job;
    while(control->next(job)) {
        if(job.state.at(0) == 255) throw std::runtime_error("job failure preserved");
        if(job.state.at(0) == 254) {
            while(!control->cancelled()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            control->throw_if_cancelled();
        }
        if(job.state.at(0) == 253) std::this_thread::sleep_for(std::chrono::milliseconds(30));
        control->complete(job.sequence);
    }
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
    beam_layout_v1 layout{1, sizeof(beam_layout_v1)};
    assert(beam_native_layout(&layout) == BEAM_OK);
    assert(layout.state_len == 96 && layout.candidate_alignment == 32);
    const std::string dir = std::filesystem::temp_directory_path().string();
    beam_context* ctx = reinterpret_cast<beam_context*>(1);
    assert(beam_context_create(2, &ctx) == BEAM_INVALID && ctx == nullptr);
    assert(beam_context_create(1, nullptr) == BEAM_INVALID);
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
    beam_status_v1 status{1, sizeof(beam_status_v1)};
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
