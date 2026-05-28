#include "runtime_config.hpp"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace beam {

namespace {

inline constexpr std::uint32_t STREAM1_MODEL_CLASSES = 120;
inline constexpr std::uint32_t STREAM1_HIDDEN1 = 1536;
inline constexpr std::uint32_t STREAM1_HIDDEN2 = 512;

std::uint64_t parse_u64(const char* text, const char* name) {
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid numeric argument: ") + name);
    }
    return static_cast<std::uint64_t>(value);
}

std::uint32_t env_u32(const char* name, std::uint32_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    const std::uint64_t parsed = parse_u64(value, name);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string("env value exceeds uint32: ") + name);
    }
    return static_cast<std::uint32_t>(parsed);
}

std::uint64_t env_u64(const char* name, std::uint64_t default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return parse_u64(value, name);
}

bool env_present(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

bool env_equals(const char* name, const char* expected) {
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, expected) == 0;
}

std::uint32_t required_env_u32(const char* name) {
    if (!env_present(name)) {
        throw std::invalid_argument(std::string("manual runtime config requires env: ") + name);
    }
    return env_u32(name, 0);
}

std::uint32_t checked_u32(std::uint64_t value, const char* name) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument(std::string("value exceeds uint32: ") + name);
    }
    return static_cast<std::uint32_t>(value);
}

std::uint32_t round_up_u32(std::uint64_t value, std::uint32_t alignment) {
    return checked_u32(round_up(value, alignment), "round_up_u32");
}

std::uint64_t scaled_round_up(std::uint64_t value, std::uint64_t ppm) {
    constexpr std::uint64_t scale = 1'000'000ULL;
    return (value * ppm + scale - 1ULL) / scale;
}

std::uint64_t ceil_log2_u64(std::uint64_t value) {
    if (value <= 1ULL) {
        return 0ULL;
    }
    std::uint64_t bits = 0;
    std::uint64_t current = value - 1ULL;
    while (current != 0ULL) {
        current >>= 1U;
        ++bits;
    }
    return bits;
}

std::uint64_t estimate_stream1_weight_bytes() {
    constexpr std::uint64_t fp16 = sizeof(std::uint16_t);
    std::uint64_t total = 0;
    total += static_cast<std::uint64_t>(STATE_LEN) * STREAM1_MODEL_CLASSES * STREAM1_HIDDEN1 * fp16;
    total += static_cast<std::uint64_t>(STREAM1_HIDDEN1) * fp16;
    total += static_cast<std::uint64_t>(STREAM1_HIDDEN1) * STREAM1_HIDDEN2 * fp16;
    total += static_cast<std::uint64_t>(STREAM1_HIDDEN2) * fp16;
    total += 2ULL * static_cast<std::uint64_t>(STREAM1_HIDDEN2) * STREAM1_HIDDEN2 * fp16;
    total += 2ULL * static_cast<std::uint64_t>(STREAM1_HIDDEN2) * fp16;
    total += 2ULL * static_cast<std::uint64_t>(STREAM1_HIDDEN2) * STREAM1_HIDDEN2 * fp16;
    total += 2ULL * static_cast<std::uint64_t>(STREAM1_HIDDEN2) * fp16;
    total += static_cast<std::uint64_t>(STREAM1_HIDDEN2) * MOVE_COUNT * fp16;
    total += static_cast<std::uint64_t>(MOVE_COUNT) * fp16;
    return total;
}

std::uint64_t estimate_read_only_table_bytes() {
    return static_cast<std::uint64_t>(MOVE_COUNT) * STATE_STORAGE_LEN * sizeof(std::uint8_t) +
           sizeof(State128) +
           static_cast<std::uint64_t>(STATE_STORAGE_LEN) * STATE_VALUE_PAD * sizeof(Hash128);
}

std::uint64_t estimate_stream1_scratch_bytes(std::uint32_t b_micro) {
    return static_cast<std::uint64_t>(b_micro) *
           (STREAM1_HIDDEN1 + 2ULL * STREAM1_HIDDEN2 + MOVE_COUNT) *
           sizeof(std::uint16_t);
}

std::uint64_t estimate_non_static_device_bytes(const RuntimeConfig& config) {
    return estimate_read_only_table_bytes() +
           estimate_stream1_weight_bytes() +
           static_cast<std::uint64_t>(config.inference_parallelism) *
               estimate_stream1_scratch_bytes(config.b_micro);
}

std::uint64_t beam_alignment_for(const RuntimeConfig& config) {
    return static_cast<std::uint64_t>(config.world_size) *
           static_cast<std::uint64_t>(config.shard_count) *
           static_cast<std::uint64_t>(config.stream4_batch_alignment);
}

std::uint64_t ring_slot_candidate_count(const RuntimeConfig& config) {
    return static_cast<std::uint64_t>(config.b_micro) * static_cast<std::uint64_t>(MOVE_COUNT);
}

std::uint64_t aligned_global_beam_width(const RuntimeConfig& config) {
    return round_up(config.user_global_beam_width, beam_alignment_for(config));
}

std::uint32_t set_stream3_batch_from_ring_slots(RuntimeConfig& config, std::uint32_t ring_slot_count) {
    const std::uint64_t slot_candidates = ring_slot_candidate_count(config);
    const std::uint64_t stream3_batch =
        static_cast<std::uint64_t>(ring_slot_count) * slot_candidates;
    if (ring_slot_count == 0U || stream3_batch == 0ULL) {
        throw std::invalid_argument("RING_SLOT_COUNT and STREAM3_BATCH_CANDIDATES must be nonzero");
    }
    config.stream3_batch_candidates = checked_u32(stream3_batch, "stream3_batch_candidates");
    return ring_slot_count;
}

void set_ring_count_from_logical_shard(RuntimeConfig& config) {
    const std::uint64_t slot_candidates = ring_slot_candidate_count(config);
    const std::uint64_t local_capacity = local_frontier_capacity(config);
    const std::uint64_t logical_shard_size = ceil_div_u64(local_capacity, config.shard_count);
    config.ring_count = checked_u32(ceil_div_u64(logical_shard_size, slot_candidates), "ring_count");
    if (config.ring_count == 0U) {
        throw std::invalid_argument("derived RING_COUNT must be nonzero");
    }
}

void set_shard_capacity_from_logical_shard(RuntimeConfig& config) {
    const std::uint64_t logical_shard_size = logical_shard_size_for(config);
    const std::uint64_t scaled_capacity =
        std::max<std::uint64_t>(
            logical_shard_size,
            scaled_round_up(logical_shard_size, config.shard_capacity_scale_ppm));
    if (env_present("BEAM_SHARD_CAPACITY_CANDIDATES")) {
        config.shard_capacity_candidates = env_u32("BEAM_SHARD_CAPACITY_CANDIDATES", 1);
        if (static_cast<std::uint64_t>(config.shard_capacity_candidates) < logical_shard_size) {
            throw std::invalid_argument("BEAM_SHARD_CAPACITY_CANDIDATES must be at least LOGICAL_SHARD_SIZE");
        }
    } else {
        config.shard_capacity_candidates = round_up_u32(scaled_capacity, config.stream4_batch_alignment);
    }
}

void set_global_spill_capacity(RuntimeConfig& config) {
    if (config.shard_buffer_count > 1U) {
        config.global_spill_capacity = env_u32("BEAM_GLOBAL_SPILL_CAPACITY", 0);
        return;
    }
    const std::uint64_t stream3_batch = static_cast<std::uint64_t>(config.stream3_batch_candidates);
    const std::uint64_t stream4_worker_count =
        static_cast<std::uint64_t>(config.stream4_active_sort_slots);
    const std::uint64_t worker_spill = stream4_worker_count * stream3_batch;
    const std::uint64_t worker_backlog_spill = stream4_worker_count * worker_spill;
    const std::uint64_t scaled_worker_backlog_spill =
        scaled_round_up(worker_backlog_spill, config.global_spill_scale_ppm);
    const std::uint64_t minimum_spill =
        std::max<std::uint64_t>(
            std::max<std::uint64_t>(stream3_batch, worker_spill),
            scaled_worker_backlog_spill);
    if (env_present("BEAM_GLOBAL_SPILL_CAPACITY")) {
        config.global_spill_capacity = env_u32("BEAM_GLOBAL_SPILL_CAPACITY", 1U << 20);
        if (config.global_spill_capacity < minimum_spill) {
            throw std::invalid_argument("BEAM_GLOBAL_SPILL_CAPACITY is below STREAM4 worker backlog spill bound");
        }
    } else {
        config.global_spill_capacity = round_up_u32(minimum_spill, config.stream4_batch_alignment);
    }
}

struct RuntimeConfigCandidate {
    RuntimeConfig config;
    StaticMemoryPlan plan;
    std::uint32_t ring_slot_count = 0;
    std::uint64_t local_frontier_capacity = 0;
    std::uint64_t logical_shard_size = 0;
    std::uint64_t gross_generated_candidates = 0;
    std::uint64_t stream3_jobs_per_depth = 0;
    std::uint64_t stream3_sort_work_units = 0;
    std::uint64_t stream4_input_candidates = 0;
    std::uint64_t stream4_jobs_per_shard = 0;
    std::uint64_t stream4_jobs_per_depth = 0;
    std::uint64_t stream4_waves_per_depth = 0;
    std::uint64_t stream4_sort_work_units = 0;
    std::uint64_t score = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t estimated_required_device_bytes = 0;
};

std::vector<std::uint32_t> ring_slot_count_candidates() {
    if (env_present("BEAM_STREAM3_RING_SLOTS")) {
        const std::uint32_t ring_slots = env_u32("BEAM_STREAM3_RING_SLOTS", 1);
        if (ring_slots == 0U) {
            throw std::invalid_argument("BEAM_STREAM3_RING_SLOTS must be nonzero");
        }
        return {ring_slots};
    }
    const std::uint32_t min_ring_slots = env_u32("BEAM_MIN_STREAM3_RING_SLOTS", 1);
    const std::uint32_t max_ring_slots = env_u32("BEAM_MAX_STREAM3_RING_SLOTS", 8);
    if (min_ring_slots == 0U || max_ring_slots == 0U || min_ring_slots > max_ring_slots) {
        throw std::invalid_argument("BEAM_MIN_STREAM3_RING_SLOTS/BEAM_MAX_STREAM3_RING_SLOTS invalid");
    }
    std::vector<std::uint32_t> ring_slots;
    for (std::uint32_t value = min_ring_slots; value <= max_ring_slots; ++value) {
        ring_slots.push_back(value);
    }
    return ring_slots;
}

std::vector<std::uint32_t> shard_count_candidates(std::uint32_t active_sort_slots) {
    if (env_present("BEAM_SHARD_COUNT")) {
        const std::uint32_t shard_count = env_u32("BEAM_SHARD_COUNT", 1);
        if (shard_count == 0U) {
            throw std::invalid_argument("BEAM_SHARD_COUNT must be nonzero");
        }
        return {shard_count};
    }
    const std::uint32_t max_shards = env_u32("BEAM_MAX_SHARD_COUNT", 128);
    const std::uint32_t min_shards = env_u32(
        "BEAM_MIN_SHARD_COUNT",
        std::max<std::uint32_t>(active_sort_slots, 1U));
    if (max_shards == 0U || min_shards == 0U || min_shards > max_shards) {
        throw std::invalid_argument("BEAM_MIN_SHARD_COUNT/BEAM_MAX_SHARD_COUNT invalid");
    }
    std::vector<std::uint32_t> shards;
    shards.reserve(max_shards - min_shards + 1U);
    for (std::uint32_t shard = min_shards; shard <= max_shards; ++shard) {
        shards.push_back(shard);
    }
    return shards;
}

void add_unique_batch(std::vector<std::uint32_t>& batches, std::uint64_t value, std::uint32_t unit) {
    if (value == 0ULL) {
        return;
    }
    const std::uint32_t batch = round_up_u32(value, unit);
    if (std::find(batches.begin(), batches.end(), batch) == batches.end()) {
        batches.push_back(batch);
    }
}

std::vector<std::uint32_t> stream4_batch_candidates(
    const RuntimeConfig& base,
    std::uint64_t logical_shard_size,
    std::uint32_t ring_slot_count) {
    const std::uint64_t stream3_batch =
        static_cast<std::uint64_t>(ring_slot_count) * ring_slot_candidate_count(base);
    if (env_present("BEAM_STREAM4_BATCH_CANDIDATES")) {
        const std::uint32_t batch = env_u32("BEAM_STREAM4_BATCH_CANDIDATES", 65536);
        if (batch == 0U) {
            throw std::invalid_argument("BEAM_STREAM4_BATCH_CANDIDATES must be nonzero");
        }
        if (batch % base.stream4_batch_alignment != 0U) {
            throw std::invalid_argument("BEAM_STREAM4_BATCH_CANDIDATES must be aligned to STREAM4_BATCH_ALIGNMENT");
        }
        if (static_cast<std::uint64_t>(batch) < stream3_batch) {
            throw std::invalid_argument("BEAM_STREAM4_BATCH_CANDIDATES must be at least STREAM3_BATCH_CANDIDATES");
        }
        return {batch};
    }

    const std::uint64_t min_batch = std::max<std::uint64_t>(
        env_u64("BEAM_MIN_STREAM4_BATCH_CANDIDATES", stream3_batch),
        stream3_batch);
    const std::uint64_t max_batch = std::min<std::uint64_t>(
        env_u64("BEAM_MAX_STREAM4_BATCH_CANDIDATES", std::max(logical_shard_size, stream3_batch)),
        static_cast<std::uint64_t>(std::numeric_limits<int>::max() / 2));
    if (min_batch > max_batch) {
        return {};
    }

    std::vector<std::uint32_t> batches;
    add_unique_batch(batches, min_batch, base.stream4_batch_alignment);
    for (std::uint32_t div : {1U, 2U, 3U, 4U, 6U, 8U, 12U, 16U, 24U, 32U}) {
        add_unique_batch(batches, ceil_div_u64(logical_shard_size, div), base.stream4_batch_alignment);
    }
    for (std::uint64_t batch = min_batch; batch <= max_batch / 2ULL; batch *= 2ULL) {
        add_unique_batch(batches, batch, base.stream4_batch_alignment);
    }
    std::sort(batches.begin(), batches.end());
    batches.erase(std::remove_if(
        batches.begin(),
        batches.end(),
        [&](std::uint32_t batch) {
            return static_cast<std::uint64_t>(batch) < min_batch ||
                   static_cast<std::uint64_t>(batch) > max_batch;
        }), batches.end());
    return batches;
}

bool try_make_candidate(
    const RuntimeConfig& base,
    std::uint32_t shard_count,
    std::uint32_t stream4_batch,
    std::uint32_t ring_slot_count,
    std::uint32_t min_stream4_jobs_per_shard,
    std::uint32_t target_stream4_jobs_per_shard,
    std::uint32_t target_shards_per_sort_slot,
    std::uint32_t stream4_flow_scale_ppm,
    std::uint64_t gpu_budget_bytes,
    std::uint64_t non_static_bytes,
    RuntimeConfigCandidate& out) {
    RuntimeConfig config = base;
    config.shard_count = shard_count;
    config.stream4_batch_candidates = stream4_batch;
    config.stream4_trigger_candidates = env_u32("BEAM_STREAM4_TRIGGER_CANDIDATES", stream4_batch);
    if (config.stream4_trigger_candidates == 0U) {
        throw std::invalid_argument("BEAM_STREAM4_TRIGGER_CANDIDATES must be nonzero");
    }
    set_stream3_batch_from_ring_slots(config, ring_slot_count);
    if (config.inference_parallelism > ring_slot_count) {
        return false;
    }
    set_ring_count_from_logical_shard(config);
    set_shard_capacity_from_logical_shard(config);
    set_global_spill_capacity(config);
    if (config.stream4_batch_candidates > config.shard_capacity_candidates ||
        config.stream4_trigger_candidates > config.shard_capacity_candidates) {
        return false;
    }

    const StaticMemoryPlan plan = make_static_memory_plan(config);
    const std::uint64_t required =
        static_cast<std::uint64_t>(plan.total_device_bytes) + non_static_bytes;
    if (required > gpu_budget_bytes) {
        return false;
    }

    const std::uint64_t local_capacity = local_frontier_capacity(config);
    const std::uint64_t logical_shard = ceil_div_u64(local_capacity, shard_count);
    const std::uint64_t gross_candidates = gross_generated_candidates_per_depth(config);
    const std::uint64_t stream3_jobs = estimated_stream3_jobs_per_depth(config);
    const std::uint64_t stream3_sort_work =
        estimated_sort_work_units(gross_candidates, config.stream3_batch_candidates);
    const std::uint64_t stream4_input =
        estimated_stream4_input_candidates_per_depth(config, stream4_flow_scale_ppm);
    const std::uint64_t stream4_jobs_per_shard =
        estimated_stream4_jobs_per_shard(config, stream4_flow_scale_ppm);
    if (stream4_jobs_per_shard < min_stream4_jobs_per_shard &&
        stream4_input > config.stream4_batch_candidates) {
        return false;
    }
    const std::uint64_t stream4_jobs = estimated_stream4_jobs_per_depth(config, stream4_flow_scale_ppm);
    const std::uint64_t stream4_waves = ceil_div_u64(stream4_jobs, config.stream4_active_sort_slots);
    const std::uint64_t stream4_batch_units =
        ceil_div_u64(stream4_batch, config.stream3_batch_candidates);
    const std::uint64_t stream4_sort_work =
        estimated_sort_work_units(stream4_input, config.stream4_batch_candidates);
    const std::uint64_t job_balance =
        stream4_jobs_per_shard > target_stream4_jobs_per_shard
            ? stream4_jobs_per_shard - target_stream4_jobs_per_shard
            : target_stream4_jobs_per_shard - stream4_jobs_per_shard;
    const std::uint64_t target_shards =
        std::max<std::uint64_t>(
            config.stream4_active_sort_slots,
            static_cast<std::uint64_t>(target_shards_per_sort_slot) * config.stream4_active_sort_slots);
    const std::uint64_t shard_balance =
        shard_count > target_shards ? shard_count - target_shards : target_shards - shard_count;
    const std::uint64_t score =
        stream4_waves * 1'000'000'000'000ULL +
        stream4_jobs * 1'000'000'000ULL +
        stream3_jobs * 10'000'000ULL +
        (stream4_sort_work / 1024ULL) * 10'000ULL +
        (stream3_sort_work / 1024ULL) * 1'000ULL +
        stream4_batch_units * 100'000ULL +
        job_balance * 1'000ULL +
        shard_balance * 10ULL +
        config.ring_count;

    out.config = config;
    out.plan = plan;
    out.ring_slot_count = ring_slot_count;
    out.local_frontier_capacity = local_capacity;
    out.logical_shard_size = logical_shard;
    out.gross_generated_candidates = gross_candidates;
    out.stream3_jobs_per_depth = stream3_jobs;
    out.stream3_sort_work_units = stream3_sort_work;
    out.stream4_input_candidates = stream4_input;
    out.stream4_jobs_per_shard = stream4_jobs_per_shard;
    out.stream4_jobs_per_depth = stream4_jobs;
    out.stream4_waves_per_depth = stream4_waves;
    out.stream4_sort_work_units = stream4_sort_work;
    out.score = score;
    out.estimated_required_device_bytes = required;
    return true;
}

void fill_runtime_config_build_estimates(
    RuntimeConfigBuild& build,
    std::uint32_t stream4_flow_scale_ppm,
    std::uint64_t free_before_bytes) {
    build.plan = make_static_memory_plan(build.config);
    build.stream4_flow_scale_ppm = stream4_flow_scale_ppm;
    build.gross_candidates_per_depth_est = gross_generated_candidates_per_depth(build.config);
    build.stream3_jobs_per_depth_est = estimated_stream3_jobs_per_depth(build.config);
    build.stream3_sort_work_units_est =
        estimated_sort_work_units(build.gross_candidates_per_depth_est, build.config.stream3_batch_candidates);
    build.stream4_input_candidates_per_depth_est =
        estimated_stream4_input_candidates_per_depth(build.config, stream4_flow_scale_ppm);
    build.stream4_jobs_per_shard_est =
        estimated_stream4_jobs_per_shard(build.config, stream4_flow_scale_ppm);
    build.stream4_jobs_per_depth_est =
        estimated_stream4_jobs_per_depth(build.config, stream4_flow_scale_ppm);
    build.stream4_waves_per_depth_est =
        ceil_div_u64(build.stream4_jobs_per_depth_est, build.config.stream4_active_sort_slots);
    build.stream4_sort_work_units_est =
        estimated_sort_work_units(
            build.stream4_input_candidates_per_depth_est,
            build.config.stream4_batch_candidates);
    build.estimated_required_device_bytes =
        static_cast<std::uint64_t>(build.plan.total_device_bytes) +
        build.estimated_non_static_device_bytes;
    if (build.estimated_required_device_bytes > build.gpu_budget_bytes) {
        throw std::runtime_error(
            "manual runtime config exceeds GPU budget: required=" +
            std::to_string(build.estimated_required_device_bytes) +
            " budget=" + std::to_string(build.gpu_budget_bytes) +
            " free_before=" + std::to_string(free_before_bytes) +
            " headroom=" + std::to_string(build.gpu_headroom_bytes));
    }
}

RuntimeConfigBuild build_manual_runtime_config(
    RuntimeConfig base,
    std::uint64_t free_before_bytes,
    std::uint64_t gpu_headroom_bytes,
    std::uint32_t stream4_flow_scale_ppm) {
    RuntimeConfigBuild build;
    build.manual_config = true;
    build.config = base;
    build.gpu_headroom_bytes = gpu_headroom_bytes;
    build.gpu_budget_bytes =
        free_before_bytes > build.gpu_headroom_bytes ? free_before_bytes - build.gpu_headroom_bytes : 0ULL;
    build.estimated_non_static_device_bytes = estimate_non_static_device_bytes(build.config);
    build.stream3_ring_slots = required_env_u32("BEAM_STREAM3_RING_SLOTS");
    build.config.shard_count = required_env_u32("BEAM_SHARD_COUNT");
    build.config.shard_buffer_count = env_u32("BEAM_SHARD_BUFFER_COUNT", 2);
    build.config.stream4_batch_candidates = required_env_u32("BEAM_STREAM4_BATCH_CANDIDATES");
    build.config.stream4_trigger_candidates =
        env_u32("BEAM_STREAM4_TRIGGER_CANDIDATES", build.config.stream4_batch_candidates);
    build.config.shard_capacity_candidates = required_env_u32("BEAM_SHARD_CAPACITY_CANDIDATES");
    build.config.global_spill_capacity =
        build.config.shard_buffer_count > 1U ? env_u32("BEAM_GLOBAL_SPILL_CAPACITY", 0) :
        required_env_u32("BEAM_GLOBAL_SPILL_CAPACITY");
    set_stream3_batch_from_ring_slots(build.config, build.stream3_ring_slots);
    if (build.config.inference_parallelism > build.stream3_ring_slots) {
        throw std::invalid_argument("manual BEAM_STREAM1_CONCURRENCY must be <= BEAM_STREAM3_RING_SLOTS");
    }
    if (env_present("BEAM_RING_COUNT")) {
        build.config.ring_count = env_u32("BEAM_RING_COUNT", 1);
    } else {
        set_ring_count_from_logical_shard(build.config);
    }
    if (build.config.stream4_batch_candidates % build.config.stream4_batch_alignment != 0U) {
        throw std::invalid_argument("manual BEAM_STREAM4_BATCH_CANDIDATES must be aligned to BEAM_STREAM4_BATCH_ALIGNMENT");
    }
    if (build.config.stream4_batch_candidates < build.config.stream3_batch_candidates) {
        throw std::invalid_argument("manual BEAM_STREAM4_BATCH_CANDIDATES must be at least STREAM3_BATCH_CANDIDATES");
    }
    if (build.config.stream4_trigger_candidates == 0U) {
        throw std::invalid_argument("manual BEAM_STREAM4_TRIGGER_CANDIDATES must be nonzero");
    }
    if (build.config.shard_capacity_candidates < logical_shard_size_for(build.config)) {
        throw std::invalid_argument("manual BEAM_SHARD_CAPACITY_CANDIDATES must be at least LOGICAL_SHARD_SIZE");
    }
    if (build.config.stream4_batch_candidates > build.config.shard_capacity_candidates) {
        throw std::invalid_argument("manual BEAM_STREAM4_BATCH_CANDIDATES must not exceed BEAM_SHARD_CAPACITY_CANDIDATES");
    }
    if (build.config.stream4_trigger_candidates > build.config.shard_capacity_candidates) {
        throw std::invalid_argument("manual BEAM_STREAM4_TRIGGER_CANDIDATES must not exceed BEAM_SHARD_CAPACITY_CANDIDATES");
    }
    if (build.config.shard_buffer_count != 2U) {
        throw std::invalid_argument("manual BEAM_SHARD_BUFFER_COUNT must be 2");
    }
    if (build.config.ring_count == 0U) {
        throw std::invalid_argument("manual RING_COUNT must be nonzero");
    }
    fill_runtime_config_build_estimates(build, stream4_flow_scale_ppm, free_before_bytes);
    return build;
}

} // namespace

std::uint64_t ceil_div_u64(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) {
        throw std::invalid_argument("ceil_div denominator is zero");
    }
    return (numerator + denominator - 1ULL) / denominator;
}

std::uint64_t local_frontier_capacity(const RuntimeConfig& config) {
    return ceil_div_u64(aligned_global_beam_width(config), config.world_size);
}

std::uint64_t logical_shard_size_for(const RuntimeConfig& config) {
    return ceil_div_u64(local_frontier_capacity(config), config.shard_count);
}

std::uint64_t gross_generated_candidates_per_depth(const RuntimeConfig& config) {
    return local_frontier_capacity(config) * static_cast<std::uint64_t>(MOVE_COUNT);
}

std::uint64_t estimated_stream3_jobs_per_depth(const RuntimeConfig& config) {
    return ceil_div_u64(gross_generated_candidates_per_depth(config), config.stream3_batch_candidates);
}

std::uint64_t estimated_stream4_input_candidates_per_depth(
    const RuntimeConfig& config,
    std::uint32_t stream4_flow_scale_ppm) {
    return scaled_round_up(gross_generated_candidates_per_depth(config), stream4_flow_scale_ppm);
}

std::uint64_t estimated_stream4_jobs_per_shard(
    const RuntimeConfig& config,
    std::uint32_t stream4_flow_scale_ppm) {
    const std::uint64_t per_shard_input =
        ceil_div_u64(estimated_stream4_input_candidates_per_depth(config, stream4_flow_scale_ppm), config.shard_count);
    if (config.stream4_trigger_candidates == 0U) {
        throw std::invalid_argument("STREAM4_TRIGGER_CANDIDATES must be nonzero");
    }
    const std::uint64_t trigger =
        std::min<std::uint64_t>(config.stream4_trigger_candidates, config.shard_capacity_candidates);
    return ceil_div_u64(per_shard_input, trigger);
}

std::uint64_t estimated_stream4_jobs_per_depth(
    const RuntimeConfig& config,
    std::uint32_t stream4_flow_scale_ppm) {
    return static_cast<std::uint64_t>(config.shard_count) *
           estimated_stream4_jobs_per_shard(config, stream4_flow_scale_ppm);
}

std::uint64_t estimated_sort_work_units(std::uint64_t items, std::uint64_t batch) {
    return items * std::max<std::uint64_t>(ceil_log2_u64(batch), 1ULL);
}

RuntimeConfigBuild build_runtime_config_from_budget(
    std::uint64_t beam,
    std::uint32_t world_size,
    std::uint32_t local_rank,
    std::uint64_t free_before_bytes) {
    RuntimeConfigBuild build;
    RuntimeConfig& config = build.config;
    config.user_global_beam_width = beam;
    config.world_size = world_size;
    config.local_rank = local_rank;
    config.b_micro = env_u32("BEAM_B_MICRO", 8192);
    config.stream4_batch_alignment =
        env_present("BEAM_STREAM4_BATCH_ALIGNMENT")
            ? env_u32("BEAM_STREAM4_BATCH_ALIGNMENT", 1024)
            : env_u32("BEAM_STREAM4_BATCH_CANDIDATES_PER_SHARD_UNIT", 1024);
    config.stream4_active_sort_slots = env_u32("BEAM_STREAM4_ACTIVE_SORT_SLOTS", 4);
    config.inference_parallelism = env_u32("BEAM_STREAM1_CONCURRENCY", 1);
    config.shard_buffer_count = env_u32("BEAM_SHARD_BUFFER_COUNT", 2);
    config.shard_capacity_scale_ppm = env_u32("BEAM_SHARD_CAPACITY_SCALE_PPM", 1'250'000);
    config.global_spill_scale_ppm = env_u32("BEAM_GLOBAL_SPILL_SCALE_PPM", 2'000'000);
    config.stream5_recv_capacity_scale_ppm = env_u32("BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM", 2'000'000);
    config.global_threshold_update_period_shards =
        env_u32("BEAM_GLOBAL_THRESHOLD_UPDATE_PERIOD_SHARDS", 64);
    config.solved_result_capacity = env_u32("BEAM_SOLVED_RESULT_CAPACITY", 1024);
    build.gpu_headroom_bytes = env_u64("BEAM_GPU_HEADROOM_BYTES", 768ULL * 1024ULL * 1024ULL);
    build.gpu_budget_bytes =
        free_before_bytes > build.gpu_headroom_bytes ? free_before_bytes - build.gpu_headroom_bytes : 0ULL;
    if (config.b_micro == 0U || config.inference_parallelism == 0U ||
        config.stream4_batch_alignment == 0U ||
        config.stream4_active_sort_slots == 0U || config.shard_buffer_count != 2U ||
        config.shard_capacity_scale_ppm == 0U ||
        config.global_spill_scale_ppm == 0U || config.stream5_recv_capacity_scale_ppm == 0U) {
        throw std::invalid_argument("B_MICRO, STREAM1_CONCURRENCY, STREAM4_ACTIVE_SORT_SLOTS, STREAM4_BATCH_ALIGNMENT, SHARD_BUFFER_COUNT, and capacity/spill scale values must be valid");
    }

    const std::uint32_t min_stream4_jobs_per_shard =
        env_u32("BEAM_MIN_STREAM4_JOBS_PER_SHARD", 1);
    const std::uint32_t target_stream4_jobs_per_shard =
        env_u32("BEAM_TARGET_STREAM4_JOBS_PER_SHARD", 4);
    const std::uint32_t target_shards_per_sort_slot =
        env_u32("BEAM_TARGET_SHARDS_PER_SORT_SLOT", 4);
    const std::uint32_t stream4_flow_scale_ppm =
        env_u32("BEAM_STREAM4_FLOW_SCALE_PPM", 2'500'000);
    if (min_stream4_jobs_per_shard == 0U || target_stream4_jobs_per_shard == 0U ||
        target_shards_per_sort_slot == 0U || stream4_flow_scale_ppm == 0U) {
        throw std::invalid_argument("BEAM_*STREAM4_JOBS_PER_SHARD, BEAM_TARGET_SHARDS_PER_SORT_SLOT, and BEAM_STREAM4_FLOW_SCALE_PPM must be nonzero");
    }
    if (env_equals("BEAM_RUNTIME_CONFIG_MODE", "manual") || env_present("BEAM_MANUAL_CONFIG")) {
        return build_manual_runtime_config(
            config,
            free_before_bytes,
            build.gpu_headroom_bytes,
            stream4_flow_scale_ppm);
    }
    build.estimated_non_static_device_bytes = estimate_non_static_device_bytes(config);
    const std::vector<std::uint32_t> ring_slots_options = ring_slot_count_candidates();

    RuntimeConfigCandidate best;
    bool found = false;
    const std::uint32_t requested_sort_slots = config.stream4_active_sort_slots;
    for (std::uint32_t sort_slots = requested_sort_slots; sort_slots >= 1U;) {
        RuntimeConfig base = config;
        base.stream4_active_sort_slots = sort_slots;
        for (std::uint32_t shard_count : shard_count_candidates(sort_slots)) {
            base.shard_count = shard_count;
            const std::uint64_t local_capacity = local_frontier_capacity(base);
            const std::uint64_t logical_shard = ceil_div_u64(local_capacity, shard_count);
            for (std::uint32_t ring_slot_count : ring_slots_options) {
                for (std::uint32_t batch : stream4_batch_candidates(base, logical_shard, ring_slot_count)) {
                    RuntimeConfigCandidate candidate;
                    if (!try_make_candidate(
                            base,
                            shard_count,
                            batch,
                            ring_slot_count,
                            min_stream4_jobs_per_shard,
                            target_stream4_jobs_per_shard,
                            target_shards_per_sort_slot,
                            stream4_flow_scale_ppm,
                            build.gpu_budget_bytes,
                            build.estimated_non_static_device_bytes,
                            candidate)) {
                        continue;
                    }
                    if (!found || candidate.score < best.score ||
                        (candidate.score == best.score &&
                         candidate.estimated_required_device_bytes < best.estimated_required_device_bytes)) {
                        best = candidate;
                        found = true;
                    }
                }
            }
        }
        if (found || env_present("BEAM_STREAM4_ACTIVE_SORT_SLOTS")) {
            break;
        }
        if (sort_slots == 1U) {
            break;
        }
        sort_slots = std::max<std::uint32_t>(1U, sort_slots / 2U);
    }

    if (!found) {
        throw std::runtime_error(
            "no runtime config fits GPU/final-layout budget: budget=" +
            std::to_string(build.gpu_budget_bytes) +
            " free_before=" + std::to_string(free_before_bytes) +
            " headroom=" + std::to_string(build.gpu_headroom_bytes));
    }
    build.config = best.config;
    build.plan = best.plan;
    build.stream3_ring_slots = best.ring_slot_count;
    build.estimated_required_device_bytes = best.estimated_required_device_bytes;
    build.gross_candidates_per_depth_est = best.gross_generated_candidates;
    build.stream3_jobs_per_depth_est = best.stream3_jobs_per_depth;
    build.stream3_sort_work_units_est = best.stream3_sort_work_units;
    build.stream4_flow_scale_ppm = stream4_flow_scale_ppm;
    build.stream4_input_candidates_per_depth_est = best.stream4_input_candidates;
    build.stream4_jobs_per_shard_est = best.stream4_jobs_per_shard;
    build.stream4_jobs_per_depth_est = best.stream4_jobs_per_depth;
    build.stream4_waves_per_depth_est = best.stream4_waves_per_depth;
    build.stream4_sort_work_units_est = best.stream4_sort_work_units;
    return build;
}

} // namespace beam
