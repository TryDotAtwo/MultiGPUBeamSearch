// Diagnostic only: count ordinary C++ new/delete allocations made by the
// generator loader. It does not measure allocator overhead, RSS or GPU memory.
#include "../src/native_input.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>

namespace {
struct alignas(std::max_align_t) Header { std::size_t size; bool counted; };
std::atomic<bool> enabled{false};
std::atomic<std::size_t> count{0}, requested{0}, live{0}, peak{0};
void* allocate(std::size_t size) {
    if (size > static_cast<std::size_t>(-1) - sizeof(Header)) throw std::bad_alloc();
    auto* p = static_cast<Header*>(std::malloc(sizeof(Header) + size));
    if (!p) throw std::bad_alloc();
    p->size = size;
    p->counted = enabled.load(std::memory_order_relaxed);
    if (p->counted) {
        count.fetch_add(1, std::memory_order_relaxed);
        requested.fetch_add(size, std::memory_order_relaxed);
        const auto now = live.fetch_add(size, std::memory_order_relaxed) + size;
        auto old = peak.load(std::memory_order_relaxed);
        while (old < now && !peak.compare_exchange_weak(old, now, std::memory_order_relaxed)) {}
    }
    return p + 1;
}
void release(void* ptr) noexcept {
    if (!ptr) return;
    auto* p = static_cast<Header*>(ptr) - 1;
    if (p->counted) live.fetch_sub(p->size, std::memory_order_relaxed);
    std::free(p);
}
}
void* operator new(std::size_t size) { return allocate(size); }
void* operator new[](std::size_t size) { return allocate(size); }
void operator delete(void* ptr) noexcept { release(ptr); }
void operator delete[](void* ptr) noexcept { release(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { release(ptr); }
void operator delete[](void* ptr, std::size_t) noexcept { release(ptr); }

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const std::filesystem::path path(argv[1]);
    std::uint64_t checksum = 0;
    constexpr int repeats = 20;
    // Initialize any lazy standard-library state before enabling the counter.
    // Long-lived runtime caches are not per-load allocations.
    {
#ifdef BEAM_BENCH_COMBINED
        const auto warmup = beam::input::load_p900_moves(path);
        if (warmup.data.empty() || warmup.names.empty()) return 4;
#else
        const auto data = beam::input::load_p900_generators(path);
        const auto names = beam::input::load_p900_move_names(path);
        if (data.empty() || names.empty()) return 4;
#endif
    }
    const auto begin = std::chrono::steady_clock::now();
    enabled.store(true, std::memory_order_relaxed);
    for (int i = 0; i < repeats; ++i) {
#ifdef BEAM_BENCH_COMBINED
        const auto moves = beam::input::load_p900_moves(path);
        const auto& data = moves.data;
        const auto& names = moves.names;
#else
        const auto data = beam::input::load_p900_generators(path);
        const auto names = beam::input::load_p900_move_names(path);
#endif
        for (auto value : data) checksum += value;
        for (const auto& name : names) checksum += name.size();
    }
    enabled.store(false, std::memory_order_relaxed);
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    if (live.load() != 0) {
        std::cerr << "counted bytes still live after measured loads: " << live.load() << '\n';
        return 3;
    }
    std::cout << "{\"repeats\":" << repeats << ",\"seconds\":" << elapsed
              << ",\"ordinary_allocations\":" << count.load()
              << ",\"ordinary_requested_bytes\":" << requested.load()
              << ",\"ordinary_peak_live_bytes\":" << peak.load()
              << ",\"checksum\":" << checksum << "}\n";
}
