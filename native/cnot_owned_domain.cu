#include "cnot_owned_domain.h"
#include "../cuda/cnot_domain.hpp"
#include "../src/cnot_domain.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

static_assert(sizeof(beam_cnot_domain_config) == 48);
static_assert(alignof(beam_cnot_domain_config) == 8);
static_assert(offsetof(beam_cnot_domain_config, parent_capacity) == 24);
static_assert(sizeof(beam_cnot_device_view) == 24 && alignof(beam_cnot_device_view) == 8);
static_assert(sizeof(beam_cnot_unseen_workspace) == 120 && alignof(beam_cnot_unseen_workspace) == 8);

namespace {
constexpr int invalid = 1, capacity = 2, cuda_failure = 3, wrong_thread = 4, internal = 5;
thread_local char last_error[1024] = "";

struct Failure : std::runtime_error {
    int code;
    Failure(int c, const std::string& what) : std::runtime_error(what), code(c) {}
};

template<class F> int boundary(F&& f) noexcept {
    last_error[0] = '\0';
    try { f(); return 0; }
    catch (const Failure& e) { std::snprintf(last_error, sizeof(last_error), "%s", e.what()); return e.code; }
    catch (const std::exception& e) { std::snprintf(last_error, sizeof(last_error), "%s", e.what()); return internal; }
    catch (...) { std::snprintf(last_error, sizeof(last_error), "unknown CNOT domain exception"); return internal; }
}

void require(bool condition, int code, const std::string& message) {
    if (!condition) throw Failure(code, message);
}
void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw Failure(cuda_failure, std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
void room(std::uint64_t required, std::uint64_t available, const char* what) {
    if (required > available) {
        std::ostringstream message;
        message << what << ": required=" << required << " available=" << available;
        throw Failure(capacity, message.str());
    }
}
std::size_t multiply(std::uint64_t a, std::uint64_t b, const char* what) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a) {
        std::ostringstream message;
        message << what << " size overflow: required=" << a << "*" << b
                << " available=" << std::numeric_limits<std::size_t>::max();
        throw Failure(capacity, message.str());
    }
    return static_cast<std::size_t>(a * b);
}

struct DeviceGuard {
    int previous = -1;
    bool changed = false;
    explicit DeviceGuard(int target) {
        cuda_check(cudaGetDevice(&previous), "cudaGetDevice");
        if (previous != target) { cuda_check(cudaSetDevice(target), "cudaSetDevice"); changed = true; }
    }
    void restore() {
        if (changed) { cuda_check(cudaSetDevice(previous), "restore CUDA device"); changed = false; }
    }
    ~DeviceGuard() { if (changed) (void)cudaSetDevice(previous); }
};

// A borrowed host slice cannot expire before every queued copy has finished,
// even when a later enqueue or synchronization fails.
struct HostBorrow {
    cudaStream_t stream;
    bool pending = true;
    explicit HostBorrow(cudaStream_t value) : stream(value) {}
    void finish() { cuda_check(cudaStreamSynchronize(stream), "synchronize host transfer"); pending = false; }
    ~HostBorrow() { if (pending) (void)cudaStreamSynchronize(stream); }
};
} // namespace

struct beam_cnot_domain {
    beam_cnot_domain_config config{};
    std::thread::id thread;
    cudaStream_t stream = nullptr;
    std::int64_t *parents = nullptr, *closed = nullptr, *indices = nullptr, *moves = nullptr;
    std::int64_t *children = nullptr, *owners = nullptr, *selected = nullptr;
    int *ordinals = nullptr, *count = nullptr;
    std::uint8_t* flags = nullptr;
    void* temp = nullptr;
    std::size_t temp_bytes = 0, stride = 0;
    std::uint64_t parent_count = 0, closed_count = 0, child_count = 0, selected_count = 0;
    bool parents_valid = false, closed_valid = false, descriptors_valid = false;
    bool children_valid = false, owners_valid = false, selected_valid = false, poisoned = false;
    bool device_only = false;

    cudaError_t release() noexcept {
        cudaError_t first = cudaSuccess;
        auto record = [&](cudaError_t e) { if (first == cudaSuccess && e != cudaSuccess) first = e; };
        if (stream) record(cudaStreamSynchronize(stream));
        void* buffers[] = {parents, closed, indices, moves, children, owners, selected,
                           ordinals, flags, count, temp};
        for (void* ptr : buffers) if (ptr) record(cudaFree(ptr));
        if (stream) record(cudaStreamDestroy(stream));
        parents = closed = indices = moves = children = owners = selected = nullptr;
        ordinals = count = nullptr; flags = nullptr; temp = nullptr; stream = nullptr;
        return first;
    }
    ~beam_cnot_domain() { (void)release(); }
};

namespace {
template<class T> void allocate(T*& target, std::size_t bytes, const char* name) {
    if (!bytes) return;
    void* raw = nullptr;
    const auto status = cudaMalloc(&raw, bytes);
    if (status == cudaErrorMemoryAllocation) {
        std::size_t free_bytes = 0, total_bytes = 0;
        (void)cudaMemGetInfo(&free_bytes, &total_bytes);
        std::ostringstream message;
        message << name << " allocation: required=" << bytes << " available=" << free_bytes;
        throw Failure(cuda_failure, message.str());
    }
    cuda_check(status, name);
    target = static_cast<T*>(raw);
}

template<class F> void use(beam_cnot_domain* ctx, F&& f, bool allow_device_only = false) {
    require(ctx != nullptr, invalid, "null CNOT domain context");
    require(ctx->thread == std::this_thread::get_id(), wrong_thread, "CNOT domain used from a different thread");
    require(!ctx->poisoned, cuda_failure, "CNOT domain is invalid after a CUDA failure");
    require(allow_device_only || !ctx->device_only, invalid, "host stages unavailable on device-only CNOT domain");
    try {
        DeviceGuard guard(ctx->config.device);
        f();
        guard.restore();
    } catch (const Failure& e) {
        if (e.code == cuda_failure) ctx->poisoned = true;
        throw;
    }
}

void input(const void* ptr, std::uint64_t required, std::uint64_t available, const char* name) {
    room(required, available, name);
    require(required == 0 || ptr != nullptr, invalid, std::string(name) + " is null");
}
void upload(beam_cnot_domain* ctx, std::int64_t* destination, const std::int64_t* source,
            std::size_t bytes) {
    if (!bytes) return;
    HostBorrow borrow(ctx->stream);
    cuda_check(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyHostToDevice, ctx->stream), "upload CNOT data");
    borrow.finish();
}
void download(beam_cnot_domain* ctx, const std::int64_t* source, std::int64_t* destination,
              std::uint64_t required, std::uint64_t available, bool valid, const char* name) {
    require(valid, invalid, std::string(name) + " output is invalid");
    input(destination, required, available, name);
    if (!required) return;
    HostBorrow borrow(ctx->stream);
    cuda_check(cudaMemcpyAsync(destination, source, multiply(required, sizeof(std::int64_t), name),
                               cudaMemcpyDeviceToHost, ctx->stream), "download CNOT data");
    borrow.finish();
}

void device_view(beam_cnot_domain* ctx, const beam_cnot_device_view* view,
                 std::uint64_t required, const char* name,
                 std::size_t alignment = alignof(std::int64_t)) {
    require(view != nullptr, invalid, std::string(name) + " view is null");
    require(view->reserved == 0, invalid, std::string(name) + " reserved field must be zero");
    require(view->device == ctx->config.device, invalid, std::string(name) + " device differs from context");
    input(view->data, required, view->elements, name);
    if (!required) return;
    require(reinterpret_cast<std::uintptr_t>(view->data) % alignment == 0,
            invalid, std::string(name) + " address is not aligned");
    cudaPointerAttributes attrs{};
    const auto status = cudaPointerGetAttributes(&attrs, view->data);
    if (status == cudaErrorInvalidValue || status == cudaErrorInvalidDevicePointer) {
        (void)cudaGetLastError();
        throw Failure(invalid, std::string(name) + " is not a device allocation");
    }
    cuda_check(status, "query device pointer");
    require(attrs.type == cudaMemoryTypeDevice && attrs.device == ctx->config.device,
            invalid, std::string(name) + " is not on the context device");
}

void disjoint(const beam_cnot_device_view& first, std::uint64_t first_count, std::size_t first_width,
              const beam_cnot_device_view& second, std::uint64_t second_count, std::size_t second_width) {
    if (!first_count || !second_count) return;
    const auto first_bytes = multiply(first_count, first_width, "device view bytes");
    const auto second_bytes = multiply(second_count, second_width, "device view bytes");
    const auto a = reinterpret_cast<std::uintptr_t>(first.data);
    const auto b = reinterpret_cast<std::uintptr_t>(second.data);
    constexpr auto max = std::numeric_limits<std::uintptr_t>::max();
    require(first_bytes <= max - a && second_bytes <= max - b, invalid, "device view address range overflow");
    require(!(a < b + second_bytes && b < a + first_bytes), invalid, "device input and output overlap");
}

void same_stream_context(cudaStream_t owned_stream, cudaStream_t stream) {
    CUcontext active = nullptr, owner = nullptr, source = nullptr;
    const auto current_status = cuCtxGetCurrent(&active);
    if (current_status != CUDA_SUCCESS || !active) {
        throw Failure(cuda_failure, "cannot query the current CUDA context");
    }
    // A caller must supply a valid live stream; CUDA documents an arbitrary
    // invalid stream handle as undefined behavior even for this query.
    const auto owner_status = cuStreamGetCtx(reinterpret_cast<CUstream>(owned_stream), &owner);
    const auto stream_status = cuStreamGetCtx(reinterpret_cast<CUstream>(stream), &source);
    require(owner_status == CUDA_SUCCESS && stream_status == CUDA_SUCCESS && owner == active && source == owner,
            invalid,
            "CUDA stream is not associated with the context device");
}

std::size_t unseen_scratch(beam_cnot_domain* ctx, std::uint64_t closed, std::uint64_t children) {
    room(closed, ctx->config.closed_capacity, "device closed capacity");
    room(children, ctx->config.child_capacity, "device child capacity");
    if (!children) return 0;
    std::size_t bytes = 0;
    cuda_check(beam::cnot::first_unseen(nullptr, bytes, nullptr, static_cast<int>(closed),
        nullptr, static_cast<int>(children), ctx->config.n, nullptr, nullptr, nullptr,
        nullptr, ctx->stream), "query borrowed CUB scratch");
    // A non-null temporary pointer distinguishes execution from the CUB query.
    return std::max<std::size_t>(1, bytes);
}
} // namespace

extern "C" std::uint32_t beam_cnot_domain_abi_version() { return 1; }
extern "C" const char* beam_cnot_domain_error() { return last_error; }

namespace {
void create_context(const beam_cnot_domain_config* config, beam_cnot_domain** out, bool device_only) {
    require(out != nullptr, invalid, "null CNOT domain output pointer");
    *out = nullptr;
    require(config != nullptr, invalid, "null CNOT domain config");
    require(config->abi_version == 1 && config->struct_size == sizeof(*config), invalid,
            "CNOT domain ABI version or config size mismatch");
    require(config->reserved == 0, invalid, "CNOT domain reserved field must be zero");
    require(config->device >= 0 && config->n >= 2 && config->n <= 256, invalid,
            "invalid CNOT device or qubit count");
    require(config->world >= 1 && config->world <= 65536, invalid, "invalid CNOT world size");
    constexpr auto max_ordinal = static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    room(config->child_capacity, max_ordinal, "child ordinal capacity");
    room(config->closed_capacity, max_ordinal - config->child_capacity, "closed ordinal capacity");
    const std::size_t stride = static_cast<std::size_t>(config->n) * beam::cnot::words_per_row(config->n);
    const auto parent_bytes = multiply(multiply(config->parent_capacity, stride, "parents"), 8, "parents");
    const auto closed_bytes = multiply(multiply(config->closed_capacity, stride, "closed"), 8, "closed");
    const auto child_bytes = multiply(multiply(config->child_capacity, stride, "children"), 8, "children");
    const auto vec_bytes = multiply(config->child_capacity, 8, "child vector");
    const auto ord_bytes = multiply(config->closed_capacity + config->child_capacity, sizeof(int), "ordinals");
    DeviceGuard guard(config->device);
    auto ctx = std::make_unique<beam_cnot_domain>();
    ctx->config = *config; ctx->stride = stride; ctx->thread = std::this_thread::get_id();
    ctx->device_only = device_only;
    cuda_check(cudaStreamCreateWithFlags(&ctx->stream, cudaStreamNonBlocking), "create CNOT stream");
    if (!device_only) {
        allocate(ctx->parents, parent_bytes, "parents");
        allocate(ctx->closed, closed_bytes, "closed");
        allocate(ctx->children, child_bytes, "children");
        allocate(ctx->indices, vec_bytes, "indices");
        allocate(ctx->moves, vec_bytes, "moves");
        allocate(ctx->owners, vec_bytes, "owners");
        allocate(ctx->selected, vec_bytes, "selected");
        if (config->child_capacity) {
            allocate(ctx->ordinals, ord_bytes, "ordinals");
            allocate(ctx->flags, multiply(config->child_capacity, 1, "flags"), "flags");
            allocate(ctx->count, sizeof(int), "selected count");
            cuda_check(beam::cnot::first_unseen(nullptr, ctx->temp_bytes, ctx->closed,
                static_cast<int>(config->closed_capacity), ctx->children, static_cast<int>(config->child_capacity),
                config->n, ctx->ordinals, ctx->flags, ctx->selected, ctx->count, ctx->stream), "query CUB scratch");
            allocate(ctx->temp, std::max<std::size_t>(1, ctx->temp_bytes), "CUB scratch");
        }
    }
    guard.restore();
    *out = ctx.release();
}
} // namespace

extern "C" int beam_cnot_domain_create(const beam_cnot_domain_config* config, beam_cnot_domain** out) {
    return boundary([&] { create_context(config, out, false); });
}

extern "C" int beam_cnot_domain_create_device(const beam_cnot_domain_config* config, beam_cnot_domain** out) {
    return boundary([&] { create_context(config, out, true); });
}

extern "C" int beam_cnot_domain_destroy(beam_cnot_domain* ctx) {
    return boundary([&] {
        require(ctx != nullptr, invalid, "null CNOT domain context");
        require(ctx->thread == std::this_thread::get_id(), wrong_thread, "CNOT domain destroyed from a different thread");
        DeviceGuard guard(ctx->config.device);
        const auto status = ctx->release();
        delete ctx;
        cuda_check(status, "release CNOT domain");
        guard.restore();
    });
}

extern "C" int beam_cnot_domain_set_parents(beam_cnot_domain* ctx, const std::int64_t* data,
                                               std::uint64_t words, std::uint64_t states) {
    return boundary([&] { use(ctx, [&] {
        room(states, ctx->config.parent_capacity, "parent capacity");
        const auto required = multiply(states, ctx->stride, "parent words");
        input(data, required, words, "parent words");
        ctx->parents_valid = ctx->descriptors_valid = ctx->children_valid = ctx->owners_valid = ctx->selected_valid = false;
        upload(ctx, ctx->parents, data, multiply(required, 8, "parent words"));
        ctx->parent_count = states; ctx->parents_valid = true;
    }); });
}
extern "C" int beam_cnot_domain_set_closed(beam_cnot_domain* ctx, const std::int64_t* data,
                                              std::uint64_t words, std::uint64_t states) {
    return boundary([&] { use(ctx, [&] {
        room(states, ctx->config.closed_capacity, "closed capacity");
        const auto required = multiply(states, ctx->stride, "closed words");
        input(data, required, words, "closed words");
        ctx->closed_valid = ctx->selected_valid = false;
        upload(ctx, ctx->closed, data, multiply(required, 8, "closed words"));
        ctx->closed_count = states; ctx->closed_valid = true;
    }); });
}
extern "C" int beam_cnot_domain_set_descriptors(beam_cnot_domain* ctx, const std::int64_t* indices,
                                                   std::uint64_t indices_len, const std::int64_t* moves,
                                                   std::uint64_t moves_len) {
    return boundary([&] { use(ctx, [&] {
        require(ctx->parents_valid, invalid, "parents are invalid");
        require(indices_len == moves_len, invalid, "descriptor lengths differ");
        room(indices_len, ctx->config.child_capacity, "descriptor capacity");
        input(indices, indices_len, indices_len, "parent indices");
        input(moves, moves_len, moves_len, "moves");
        for (std::uint64_t i = 0; i < indices_len; ++i) {
            require(indices[i] >= 0 && static_cast<std::uint64_t>(indices[i]) < ctx->parent_count,
                    invalid, "invalid descriptor parent index");
            require(moves[i] >= 0 && moves[i] < beam::cnot::move_count(ctx->config.n), invalid,
                    "invalid descriptor move");
        }
        ctx->descriptors_valid = ctx->children_valid = ctx->owners_valid = ctx->selected_valid = false;
        const auto bytes = multiply(indices_len, 8, "descriptors");
        if (bytes) {
            HostBorrow borrow(ctx->stream);
            cuda_check(cudaMemcpyAsync(ctx->indices, indices, bytes, cudaMemcpyHostToDevice, ctx->stream), "upload indices");
            cuda_check(cudaMemcpyAsync(ctx->moves, moves, bytes, cudaMemcpyHostToDevice, ctx->stream), "upload moves");
            borrow.finish();
        }
        ctx->child_count = indices_len; ctx->descriptors_valid = true;
    }); });
}
extern "C" int beam_cnot_domain_expand(beam_cnot_domain* ctx) {
    return boundary([&] { use(ctx, [&] {
        require(ctx->parents_valid && ctx->descriptors_valid, invalid, "parents or descriptors are invalid");
        ctx->children_valid = ctx->owners_valid = ctx->selected_valid = false;
        cuda_check(beam::cnot::launch_children(ctx->parents, ctx->parent_count, ctx->config.n,
            ctx->indices, ctx->moves, false, ctx->child_count, ctx->children, ctx->stream), "expand CNOT children");
        ctx->children_valid = true;
    }); });
}
extern "C" int beam_cnot_domain_owners(beam_cnot_domain* ctx) {
    return boundary([&] { use(ctx, [&] {
        require(ctx->children_valid, invalid, "children are invalid");
        ctx->owners_valid = false;
        cuda_check(beam::cnot::launch_owners(ctx->children, ctx->child_count, ctx->config.n,
            ctx->config.world, ctx->owners, ctx->stream), "compute CNOT owners");
        ctx->owners_valid = true;
    }); });
}
extern "C" int beam_cnot_domain_first_unseen(beam_cnot_domain* ctx, std::uint64_t* selected_count) {
    return boundary([&] { use(ctx, [&] {
        require(selected_count != nullptr, invalid, "null CNOT selected count");
        require(ctx->children_valid && ctx->closed_valid, invalid, "children or closed states are invalid");
        ctx->selected_valid = false;
        if (!ctx->child_count) { ctx->selected_count = *selected_count = 0; ctx->selected_valid = true; return; }
        std::size_t required = 0;
        auto run = [&](void* temp, std::size_t& bytes) {
            return beam::cnot::first_unseen(temp, bytes, ctx->closed, static_cast<int>(ctx->closed_count),
                ctx->children, static_cast<int>(ctx->child_count), ctx->config.n, ctx->ordinals,
                ctx->flags, ctx->selected, ctx->count, ctx->stream);
        };
        cuda_check(run(nullptr, required), "query CUB scratch");
        room(required, ctx->temp_bytes, "CUB scratch bytes");
        auto available = ctx->temp_bytes;
        cuda_check(run(ctx->temp, available), "select unseen CNOT states");
        int result = 0;
        HostBorrow borrow(ctx->stream);
        cuda_check(cudaMemcpyAsync(&result, ctx->count, sizeof(result), cudaMemcpyDeviceToHost, ctx->stream), "download selected count");
        borrow.finish();
        require(result >= 0 && static_cast<std::uint64_t>(result) <= ctx->child_count, internal, "invalid CNOT selected count");
        ctx->selected_count = *selected_count = static_cast<std::uint64_t>(result);
        ctx->selected_valid = true;
    }); });
}
extern "C" int beam_cnot_domain_download_children(beam_cnot_domain* ctx, std::int64_t* out, std::uint64_t cap) {
    return boundary([&] { use(ctx, [&] { download(ctx, ctx->children, out,
        multiply(ctx->child_count, ctx->stride, "child words"), cap, ctx->children_valid, "children"); }); });
}
extern "C" int beam_cnot_domain_download_owners(beam_cnot_domain* ctx, std::int64_t* out, std::uint64_t cap) {
    return boundary([&] { use(ctx, [&] { download(ctx, ctx->owners, out, ctx->child_count, cap, ctx->owners_valid, "owners"); }); });
}
extern "C" int beam_cnot_domain_download_selected(beam_cnot_domain* ctx, std::int64_t* out, std::uint64_t cap) {
    return boundary([&] { use(ctx, [&] { download(ctx, ctx->selected, out, ctx->selected_count, cap, ctx->selected_valid, "selected"); }); });
}

extern "C" std::uint32_t beam_cnot_domain_device_abi_version() { return 1; }
extern "C" int beam_cnot_domain_owners_device(beam_cnot_domain* ctx, const beam_cnot_device_view* states,
                                                std::uint64_t count, const beam_cnot_device_view* owners,
                                                void* raw_stream) {
    return boundary([&] { use(ctx, [&] {
        room(count, ctx->config.child_capacity, "device state capacity");
        const auto words = multiply(count, ctx->stride, "device state words");
        device_view(ctx, states, words, "states");
        device_view(ctx, owners, count, "owners");
        disjoint(*states, words, sizeof(std::int64_t), *owners, count, sizeof(std::int64_t));
        const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
        same_stream_context(ctx->stream, stream);
        cuda_check(cudaStreamSynchronize(ctx->stream), "drain CNOT domain stream");
        HostBorrow completion(stream);
        cuda_check(beam::cnot::launch_owners(static_cast<const std::int64_t*>(states->data), count,
            ctx->config.n, ctx->config.world, static_cast<std::int64_t*>(owners->data), stream),
            "compute borrowed CNOT owners");
        completion.finish();
    }, true); });
}

extern "C" int beam_cnot_domain_children_device(
    beam_cnot_domain* ctx, const beam_cnot_device_view* parents, std::uint64_t parent_count,
    const beam_cnot_device_view* indices, const beam_cnot_device_view* moves,
    std::uint32_t moves_are_u16, std::uint64_t child_count, const beam_cnot_device_view* children,
    void* raw_stream) {
    return boundary([&] { use(ctx, [&] {
        room(parent_count, ctx->config.parent_capacity, "device parent capacity");
        room(parent_count, static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()),
             "device parent count");
        room(child_count, ctx->config.child_capacity, "device child capacity");
        require(moves_are_u16 <= 1, invalid, "invalid device move type");
        require(!child_count || parent_count, invalid, "cannot expand an empty parent frontier");
        const auto parent_words = multiply(parent_count, ctx->stride, "device parent words");
        const auto child_words = multiply(child_count, ctx->stride, "device child words");
        const auto move_width = moves_are_u16 ? sizeof(std::uint16_t) : sizeof(std::int64_t);
        device_view(ctx, parents, parent_words, "parents");
        device_view(ctx, indices, child_count, "indices");
        device_view(ctx, moves, child_count, "moves", move_width);
        device_view(ctx, children, child_words, "children");
        disjoint(*parents, parent_words, sizeof(std::int64_t), *children, child_words, sizeof(std::int64_t));
        disjoint(*indices, child_count, sizeof(std::int64_t), *children, child_words, sizeof(std::int64_t));
        disjoint(*moves, child_count, move_width, *children, child_words, sizeof(std::int64_t));
        const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
        same_stream_context(ctx->stream, stream);
        cuda_check(cudaStreamSynchronize(ctx->stream), "drain CNOT domain stream");
        HostBorrow completion(stream);
        cuda_check(beam::cnot::launch_children(static_cast<const std::int64_t*>(parents->data),
            static_cast<std::int64_t>(parent_count), ctx->config.n,
            static_cast<const std::int64_t*>(indices->data), moves->data, moves_are_u16 != 0,
            static_cast<std::int64_t>(child_count), static_cast<std::int64_t*>(children->data), stream),
            "expand borrowed CNOT children");
        completion.finish();
    }, true); });
}

extern "C" int beam_cnot_domain_unseen_scratch_bytes(
    beam_cnot_domain* ctx, std::uint64_t closed, std::uint64_t children, std::uint64_t* bytes) {
    return boundary([&] { use(ctx, [&] {
        require(bytes != nullptr, invalid, "null CNOT scratch size output");
        *bytes = unseen_scratch(ctx, closed, children);
    }, true); });
}

extern "C" int beam_cnot_domain_first_unseen_device(
    beam_cnot_domain* ctx, const beam_cnot_device_view* closed, std::uint64_t closed_count,
    const beam_cnot_device_view* children, std::uint64_t child_count,
    const beam_cnot_unseen_workspace* workspace, void* raw_stream) {
    return boundary([&] { use(ctx, [&] {
        require(workspace != nullptr, invalid, "null CNOT unseen workspace");
        const auto bytes = unseen_scratch(ctx, closed_count, child_count);
        const auto closed_words = multiply(closed_count, ctx->stride, "device closed words");
        const auto child_words = multiply(child_count, ctx->stride, "device child words");
        const auto ordinal_count = child_count ? closed_count + child_count : 0;
        device_view(ctx, closed, closed_words, "closed");
        device_view(ctx, children, child_words, "children");
        device_view(ctx, &workspace->ordinals, ordinal_count, "ordinals", alignof(int));
        device_view(ctx, &workspace->flags, child_count, "flags", alignof(std::uint8_t));
        device_view(ctx, &workspace->selected, child_count, "selected");
        device_view(ctx, &workspace->count, 1, "count", alignof(int));
        device_view(ctx, &workspace->scratch, bytes, "CUB scratch", 256);

        struct Region { const beam_cnot_device_view* view; std::uint64_t count; std::size_t width; };
        const Region inputs[] = {{closed, closed_words, sizeof(std::int64_t)},
                                 {children, child_words, sizeof(std::int64_t)}};
        const Region outputs[] = {{&workspace->ordinals, ordinal_count, sizeof(int)},
            {&workspace->flags, child_count, sizeof(std::uint8_t)},
            {&workspace->selected, child_count, sizeof(std::int64_t)},
            {&workspace->count, 1, sizeof(int)}, {&workspace->scratch, bytes, 1}};
        for (std::size_t i = 0; i < sizeof(outputs) / sizeof(outputs[0]); ++i) {
            for (const auto& input : inputs)
                disjoint(*input.view, input.count, input.width,
                         *outputs[i].view, outputs[i].count, outputs[i].width);
            for (std::size_t j = 0; j < i; ++j)
                disjoint(*outputs[j].view, outputs[j].count, outputs[j].width,
                         *outputs[i].view, outputs[i].count, outputs[i].width);
        }
        const auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
        same_stream_context(ctx->stream, stream);
        cuda_check(cudaStreamSynchronize(ctx->stream), "drain CNOT domain stream");
        HostBorrow completion(stream);
        if (!child_count) {
            cuda_check(cudaMemsetAsync(workspace->count.data, 0, sizeof(int), stream), "clear CNOT count");
        } else {
            auto available = bytes;
            cuda_check(beam::cnot::first_unseen(workspace->scratch.data, available,
                static_cast<const std::int64_t*>(closed->data), static_cast<int>(closed_count),
                static_cast<const std::int64_t*>(children->data), static_cast<int>(child_count),
                ctx->config.n, static_cast<int*>(workspace->ordinals.data),
                static_cast<std::uint8_t*>(workspace->flags.data),
                static_cast<std::int64_t*>(workspace->selected.data),
                static_cast<int*>(workspace->count.data), stream), "select borrowed unseen CNOT states");
        }
        completion.finish();
    }, true); });
}
