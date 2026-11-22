#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace beam {
// Host-side observation payload. Only actual launch sites may populate it.
// Presence/validation is not coverage, runtime attestation or numeric admission.
struct Stream1KernelDescription {
    std::string family, implementation, dtype, epilogue;
    std::uint32_t architecture;
    std::uint32_t rows, input_cols, output_cols;
    std::uint32_t tile_m, tile_n, tile_k;
    std::uint32_t warp_m, warp_n, warp_k;
    std::uint32_t instruction_m, instruction_n, instruction_k;
    std::uint32_t stages, swizzle;
    bool packed_weight;
    auto identity() const {
        return std::tie(family, implementation, dtype, epilogue, architecture,
            rows, input_cols, output_cols, tile_m, tile_n, tile_k,
            warp_m, warp_n, warp_k, instruction_m, instruction_n, instruction_k,
            stages, swizzle, packed_weight);
    }
    bool operator==(const Stream1KernelDescription& other) const {
        return identity()==other.identity();
    }
};
inline void validate_stream1_kernel_description(const Stream1KernelDescription& d) {
    if (d.family.empty() || d.implementation.empty() || d.dtype.empty() || d.epilogue.empty() ||
        d.architecture<75 || !d.rows || !d.input_cols || !d.output_cols ||
        !d.tile_m || !d.tile_n || !d.tile_k || !d.warp_m || !d.warp_n || !d.warp_k ||
        !d.instruction_m || !d.instruction_n || !d.instruction_k || !d.stages || !d.swizzle)
        throw std::invalid_argument("incomplete actual kernel description");
}
class Stream1KernelObservation {
    std::size_t capacity_;
    std::vector<Stream1KernelDescription> records_;
public:
    explicit Stream1KernelObservation(std::size_t capacity): capacity_(capacity) {
        if (!capacity) throw std::invalid_argument("kernel observation capacity must be positive");
    }
    void record(const Stream1KernelDescription& description) {
        validate_stream1_kernel_description(description);
        if (std::find(records_.begin(),records_.end(),description)!=records_.end()) return;
        if (records_.size()==capacity_)
            throw std::overflow_error("kernel observation capacity exhausted; incomplete trace forbidden");
        records_.push_back(description);
    }
    const std::vector<Stream1KernelDescription>& records() const noexcept { return records_; }
};
inline thread_local Stream1KernelObservation* stream1_kernel_observation = nullptr;
inline Stream1KernelObservation* active_stream1_kernel_observation() noexcept {
    return stream1_kernel_observation;
}
// Explicit diagnostic host scope; does not attest GPU completion or graph replay.
class Stream1KernelObservationScope {
    Stream1KernelObservation* previous_;
public:
    explicit Stream1KernelObservationScope(Stream1KernelObservation& observation) noexcept
        : previous_(stream1_kernel_observation) { stream1_kernel_observation=&observation; }
    ~Stream1KernelObservationScope() { stream1_kernel_observation=previous_; }
    Stream1KernelObservationScope(const Stream1KernelObservationScope&)=delete;
    Stream1KernelObservationScope& operator=(const Stream1KernelObservationScope&)=delete;
};
}
