#include "engine_control.hpp"
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <thread>

int run_production_runner(int, char**);
void beam_engine_layout(beam_layout_v1&);
namespace {
std::atomic<bool> occupied{false};
thread_local beam::native::Control* active_control = nullptr;
std::vector<std::string> copy_arguments(int argc, const char* const* argv) {
    if(argc < 1 || argc > 4096 || !argv) throw std::invalid_argument("invalid native argv");
    std::vector<std::string> args;
    args.reserve(argc);
    for(int i=0; i<argc; ++i) {
        if(!argv[i]) throw std::invalid_argument("null native argument");
        args.emplace_back(argv[i]);
    }
    return args;
}
std::vector<char*> argument_views(std::vector<std::string>& args) {
    std::vector<char*> views;
    for(auto& arg : args) views.push_back(arg.data());
    views.push_back(nullptr);
    return views;
}
}
struct beam_context {
    beam::native::Control control;
    std::vector<std::string> args;
    std::thread worker;
};
beam::native::Control* beam::native::current_control() noexcept { return active_control; }

extern "C" {
uint32_t beam_native_abi_version() { return BEAM_NATIVE_ABI_VERSION; }
beam_result beam_native_layout(beam_layout_v1* layout) {
    if(!layout || layout->abi_version != BEAM_NATIVE_ABI_VERSION ||
       layout->struct_size < sizeof(*layout)) return BEAM_INVALID;
    try { beam_engine_layout(*layout); return BEAM_OK; }
    catch(...) { return BEAM_FAILED; }
}
beam_result beam_context_create(uint32_t version, beam_context** out) {
    if(!out) return BEAM_INVALID;
    *out = nullptr;
    if(version != BEAM_NATIVE_ABI_VERSION) return BEAM_INVALID;
    if(occupied.exchange(true)) return BEAM_BUSY;
    try { *out = new beam_context; return BEAM_OK; }
    catch(...) { occupied = false; return BEAM_FAILED; }
}
beam_result beam_context_prepare(beam_context* ctx, int argc, const char* const* argv) {
    if(!ctx) return BEAM_INVALID;
    try {
        auto args = copy_arguments(argc, argv);
        std::lock_guard lock(ctx->control.mutex);
        if(ctx->control.phase != BEAM_CREATED) return BEAM_BUSY;
        ctx->args = std::move(args);
        ctx->control.phase = BEAM_PREPARING;
        ctx->worker = std::thread([ctx] {
            active_control = &ctx->control;
            try {
                auto argv = argument_views(ctx->args);
                int result = run_production_runner(static_cast<int>(ctx->args.size()), argv.data());
                ctx->control.finish(result == 0 ? BEAM_OK : BEAM_FAILED,
                                    result == 0 ? "" : "native engine returned failure");
            } catch(const beam::native::Cancelled& error) {
                ctx->control.finish(BEAM_CANCELLED, error.what());
            } catch(const std::exception& error) {
                ctx->control.finish(BEAM_FAILED, error.what());
            } catch(...) { ctx->control.finish(BEAM_FAILED, "unknown native exception"); }
            active_control = nullptr;
        });
        return BEAM_OK;
    } catch(const std::invalid_argument&) { return BEAM_INVALID; }
      catch(...) { ctx->control.finish(BEAM_FAILED, "cannot start native worker"); return BEAM_FAILED; }
}
beam_result beam_context_submit(beam_context* ctx, const beam_job_v1* job) {
    if(!ctx || !job || job->abi_version != BEAM_NATIVE_ABI_VERSION ||
       job->struct_size < sizeof(*job) || !job->state || !job->state_len ||
       job->state_len > 65536 || !job->output_directory || !job->output_directory_len ||
       job->output_directory_len > 32768) return BEAM_INVALID;
    try {
        std::lock_guard lock(ctx->control.mutex);
        auto& control = ctx->control;
        if(control.phase != BEAM_READY || control.stop) return BEAM_BUSY;
        if(job->sequence != control.completed) return BEAM_INVALID;
        std::string directory(job->output_directory, job->output_directory_len);
        if(directory.find('\0') != std::string::npos ||
           !std::filesystem::path(directory).is_absolute() ||
           !std::filesystem::is_directory(directory)) return BEAM_INVALID;
        beam::native::Job copy;
        copy.sequence = job->sequence;
        copy.state.assign(job->state, job->state + job->state_len);
        copy.directory = std::move(directory);
        control.pending = std::move(copy);
        control.has_pending = true;
        control.phase = BEAM_RUNNING;
        control.changed.notify_all();
        return BEAM_OK;
    } catch(...) { return BEAM_FAILED; }
}
beam_result beam_context_wait(beam_context* ctx, uint64_t timeout_ms) {
    if(!ctx || timeout_ms > static_cast<uint64_t>(INT64_MAX)) return BEAM_INVALID;
    try {
        std::unique_lock lock(ctx->control.mutex);
        auto& control = ctx->control;
        if(control.phase == BEAM_CREATED) return BEAM_INVALID;
        if(!control.changed.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            return control.phase == BEAM_READY || control.phase == BEAM_STOPPED || control.phase == BEAM_ERROR;
        })) return BEAM_TIMEOUT;
        return control.stop ? BEAM_CANCELLED : control.result;
    } catch(...) { return BEAM_FAILED; }
}
beam_result beam_context_status(beam_context* ctx, beam_status_v1* status) {
    if(!ctx || !status || status->abi_version != BEAM_NATIVE_ABI_VERSION ||
       status->struct_size < sizeof(*status)) return BEAM_INVALID;
    try {
        std::lock_guard lock(ctx->control.mutex);
        status->phase = ctx->control.phase;
        status->result = ctx->control.result;
        status->completed_jobs = ctx->control.completed;
        return BEAM_OK;
    } catch(...) { return BEAM_FAILED; }
}
beam_result beam_context_cancel(beam_context* ctx) {
    if(!ctx) return BEAM_INVALID;
    try {
        std::lock_guard lock(ctx->control.mutex);
        ctx->control.stop = true;
        ctx->control.changed.notify_all();
        return BEAM_OK;
    } catch(...) { return BEAM_FAILED; }
}
beam_result beam_context_error(beam_context* ctx, char* dst, uint64_t capacity, uint64_t* required) {
    if(!ctx || !required || (!dst && capacity)) return BEAM_INVALID;
    try {
        std::lock_guard lock(ctx->control.mutex);
        const auto& message = ctx->control.error;
        *required = message.size() + 1;
        if(capacity < *required) return BEAM_BUFFER_TOO_SMALL;
        std::memcpy(dst, message.c_str(), *required);
        return BEAM_OK;
    } catch(...) { return BEAM_FAILED; }
}
beam_result beam_context_destroy(beam_context* ctx) {
    if(!ctx) return BEAM_OK;
    try {
        beam_context_cancel(ctx);
        if(ctx->worker.joinable()) ctx->worker.join();
        delete ctx;
        occupied = false;
        return BEAM_OK;
    } catch(...) { return BEAM_FAILED; }
}
int beam_native_main(int argc, const char* const* argv) {
    if(occupied.exchange(true)) return 1;
    int result = 1;
    try {
        auto args = copy_arguments(argc, argv);
        auto views = argument_views(args);
        result = run_production_runner(argc, views.data());
    } catch(const std::exception& error) {
        std::cerr << "production_runner_error=" << error.what() << '\n';
    } catch(...) { std::cerr << "production_runner_error=unknown exception\n"; }
    occupied = false;
    return result;
}
}
