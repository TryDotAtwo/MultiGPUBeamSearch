#pragma once
#include "beam_native.h"
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace beam::native {
struct Job {
    std::uint64_t sequence = 0;
    std::vector<std::uint8_t> state;
    std::string directory;
};
struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("native search cancelled") {}
};
// One engine thread and one controlling Rust thread. No device memory appears
// here; the engine's existing arena/stream/event ownership stays on that thread.
class Control {
public:
    std::mutex mutex;
    std::condition_variable changed;
    beam_phase phase = BEAM_CREATED;
    beam_result result = BEAM_OK;
    std::uint64_t completed = 0;
    std::string error;
    Job pending;
    bool has_pending = false;
    bool stop = false;
    beam_history_api_v1 history{};

    bool next(Job& job) {
        std::unique_lock lock(mutex);
        if(phase == BEAM_PREPARING) { phase = BEAM_READY; changed.notify_all(); }
        changed.wait(lock, [&] { return has_pending || stop; });
        if(stop) return false;
        job = std::move(pending);
        has_pending = false;
        return true;
    }
    void complete(std::uint64_t sequence) {
        std::lock_guard lock(mutex);
        if(sequence != completed || phase != BEAM_RUNNING)
            throw std::logic_error("native completion sequence mismatch");
        ++completed;
        phase = BEAM_READY;
        changed.notify_all();
    }
    bool cancelled() {
        std::lock_guard lock(mutex);
        return stop;
    }
    void throw_if_cancelled() { if(cancelled()) throw Cancelled(); }
    void finish(beam_result code, const char* message = "") {
        std::lock_guard lock(mutex);
        result = stop ? BEAM_CANCELLED : code;
        phase = result == BEAM_FAILED ? BEAM_ERROR : BEAM_STOPPED;
        error = message;
        changed.notify_all();
    }
};
Control* current_control() noexcept;
const beam_history_api_v1* current_history() noexcept;
inline void cancellation_point() {
    if(auto* control = current_control()) control->throw_if_cancelled();
}
}
