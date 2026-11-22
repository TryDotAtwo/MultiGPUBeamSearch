#pragma once

#include "static_memory.hpp"

#include <cstdint>

namespace beam {

struct RuntimeConfigBuild {
    RuntimeConfig config;
    StaticMemoryPlan plan;
    bool manual_config = false;
    std::uint32_t stream3_ring_slots = 0;
    std::uint64_t gpu_headroom_bytes = 0;
    std::uint64_t gpu_budget_bytes = 0;
    std::uint64_t estimated_non_static_device_bytes = 0;
    std::uint64_t estimated_required_device_bytes = 0;
    std::uint64_t gross_candidates_per_depth_est = 0;
    std::uint64_t stream3_jobs_per_depth_est = 0;
    std::uint64_t stream3_sort_work_units_est = 0;
    std::uint32_t stream4_flow_scale_ppm = 0;
    std::uint64_t stream4_input_candidates_per_depth_est = 0;
    std::uint64_t stream4_jobs_per_shard_est = 0;
    std::uint64_t stream4_jobs_per_depth_est = 0;
    std::uint64_t stream4_waves_per_depth_est = 0;
    std::uint64_t stream4_sort_work_units_est = 0;
};

std::uint64_t ceil_div_u64(std::uint64_t numerator, std::uint64_t denominator);
std::uint64_t local_frontier_capacity(const RuntimeConfig& config);
std::uint64_t logical_shard_size_for(const RuntimeConfig& config);
std::uint64_t gross_generated_candidates_per_depth(const RuntimeConfig& config);
std::uint64_t estimated_stream3_jobs_per_depth(const RuntimeConfig& config);
std::uint64_t estimated_stream4_input_candidates_per_depth(
    const RuntimeConfig& config,
    std::uint32_t stream4_flow_scale_ppm);
std::uint64_t estimated_stream4_jobs_per_shard(
    const RuntimeConfig& config,
    std::uint32_t stream4_flow_scale_ppm);
std::uint64_t estimated_stream4_jobs_per_depth(
    const RuntimeConfig& config,
    std::uint32_t stream4_flow_scale_ppm);
std::uint64_t estimated_sort_work_units(std::uint64_t items, std::uint64_t batch);

RuntimeConfigBuild build_runtime_config_from_budget(
    std::uint64_t beam,
    std::uint32_t world_size,
    std::uint32_t local_rank,
    const Stream1ModelConfig& stream1_model,
    std::uint64_t free_before_bytes);

RuntimeConfigBuild build_runtime_config_from_budget(
    std::uint64_t beam,
    std::uint32_t world_size,
    std::uint32_t local_rank,
    std::uint64_t free_before_bytes);

} // namespace beam
