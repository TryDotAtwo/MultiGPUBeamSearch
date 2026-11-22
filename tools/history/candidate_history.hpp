#pragma once
// Internal native history transport, included inside the runner namespace.
// Rust owns host storage when a context installs the history extension.

enum class CandidateHistoryMode : std::uint8_t {
    Ram,
    Disk,
    StaticHybrid
};

CandidateHistoryMode parse_history_mode() {
    const char* value = std::getenv("BEAM_HISTORY_MODE");
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "ram") == 0) {
        return CandidateHistoryMode::Ram;
    }
    if (std::strcmp(value, "disk") == 0) {
        return CandidateHistoryMode::Disk;
    }
    if (std::strcmp(value, "static_hybrid") == 0 || std::strcmp(value, "hybrid") == 0) {
        return CandidateHistoryMode::StaticHybrid;
    }
    throw std::runtime_error("BEAM_HISTORY_MODE must be ram, disk, static_hybrid, or hybrid");
}

const char* history_mode_name(CandidateHistoryMode mode) {
    switch (mode) {
    case CandidateHistoryMode::Ram:
        return "ram";
    case CandidateHistoryMode::Disk:
        return "disk";
    case CandidateHistoryMode::StaticHybrid:
        return "static_hybrid";
    }
    return "unknown";
}

struct HistoryEntry {
    std::uint64_t parent_idx = 0;
    std::uint32_t route_packed = 0;
    std::uint32_t pad = 0;
};

static_assert(sizeof(HistoryEntry) == 16);
static_assert(alignof(HistoryEntry) == 8);


struct CpuCandidateHistory {
    static constexpr std::uint32_t kWriteChunkEntries = 1U << 20U;
    beam::native::HistoryBridge rust_history;
    bool rust_history_open = false;
    ~CpuCandidateHistory() noexcept {
        try { destroy(); } catch(...) {}
    }

    struct DepthLocation {
        HistoryStorageLocation location = HistoryStorageLocation::None;
        std::uint64_t offset_entries = 0;
    };

    struct MaterializeResult {
        std::uint32_t depth_index = 0;
        HistoryStorageLocation location = HistoryStorageLocation::None;
        std::uint64_t offset_entries = 0;
        std::vector<HistoryEntry> ram_data;
    };

    struct Slot {
        CandidateMeta* host = nullptr;
        std::size_t registered_bytes = 0;
        std::vector<HistoryEntry> staging;
        std::uint32_t capacity = 0;
        std::uint32_t count = 0;
        std::uint32_t depth_index = 0;
        std::filesystem::path path;
        HistoryStorageLocation location = HistoryStorageLocation::None;
        std::uint64_t offset_entries = 0;
        cudaEvent_t copy_done = nullptr;
        bool copy_pending = false;
        bool free = true;
        std::future<MaterializeResult> writer;
    };

    struct PruneResult {
        std::uint32_t depth_index = 0;
        std::vector<HistoryEntry> previous;
        std::vector<HistoryEntry> current;
        std::uint64_t bytes_before = 0;
        std::uint64_t bytes_after = 0;
    };

    struct PruneJob {
        std::uint32_t depth_index = 0;
        std::future<PruneResult> worker;
    };

    CandidateHistoryMode mode = CandidateHistoryMode::Ram;
    std::filesystem::path dir;
    std::vector<std::filesystem::path> depth_files;
    std::vector<std::uint32_t> depth_counts;
    std::vector<DepthLocation> depth_locations;
    std::vector<std::vector<HistoryEntry>> ram_depths;
    std::vector<bool> ram_ready;
    std::vector<bool> prune_dirty;
    std::vector<PruneJob> prune_jobs;
    std::uint64_t bytes_received = 0;
    std::uint64_t bytes_stored = 0;
    std::uint64_t bytes_stored_ram = 0;
    std::uint64_t bytes_stored_disk = 0;
    std::uint64_t bytes_pruned = 0;
    std::uint64_t bytes_pinned_slots = 0;
    std::uint64_t bytes_slot_staging = 0;
    std::uint64_t bytes_static_ram_arena = 0;
    std::uint64_t bytes_static_disk_arena = 0;
    std::uint64_t history_required_bytes = 0;
    HistoryBudgetEstimate budget_estimate{};
    HistoryPlan static_plan;
    std::uint32_t worker_count = 1;
    bool prune_enabled = true;
    std::vector<Slot> slots;
    cudaStream_t copy_stream = nullptr;
    HistoryEntry* ram_arena = nullptr;
    std::uint64_t ram_arena_entries = 0;
    std::uint64_t ram_entries_used = 0;
    std::uint64_t disk_arena_entries = 0;
    std::uint64_t disk_entries_used = 0;
    std::filesystem::path static_disk_path;
    mutable std::mutex static_disk_mutex;
    mutable std::mutex storage_mutex;
    bool disk_write_failed = false;

    void sync_rust_stats() {
        if(!rust_history_open) return;
        const auto stats=rust_history.stats();
        bytes_received=stats.bytes_received; bytes_stored=stats.bytes_stored;
        bytes_stored_ram=stats.bytes_stored_ram; bytes_stored_disk=stats.bytes_stored_disk;
        bytes_pruned=stats.bytes_pruned;
        bytes_static_ram_arena=stats.bytes_static_ram_arena;
        bytes_static_disk_arena=stats.bytes_static_disk_arena;
        history_required_bytes=stats.required_entries*sizeof(HistoryEntry);
        budget_estimate={stats.effective_depth,stats.target_beam_depth,
            stats.states_before_target_beam,stats.required_entries};
        static_plan.ram_reserved_entries=stats.ram_reserved;
        static_plan.disk_reserved_entries=stats.disk_reserved;
    }

    void initialize(
        CandidateHistoryMode selected_mode,
        std::uint32_t capacity,
        std::uint32_t slot_count,
        std::uint32_t selected_worker_count,
        std::uint32_t depth_limit,
        std::uint32_t solved_neighborhood_radius,
        std::uint32_t stream2_suffix_radius,
        std::uint64_t history_ram_budget_bytes = 0,
        std::uint64_t history_disk_budget_bytes = 0,
        const std::filesystem::path& history_disk_path = {}) {
        if (copy_stream != nullptr) {
            throw std::runtime_error("candidate history already initialized");
        }
        if (capacity == 0U || slot_count == 0U || selected_worker_count == 0U) {
            throw std::runtime_error("candidate history capacity, slot count, and worker count must be nonzero");
        }
        mode = selected_mode;
        worker_count = selected_worker_count;
        std::filesystem::create_directories(dir);

        const std::uint64_t capacity_entries = static_cast<std::uint64_t>(capacity);
        const std::uint64_t slot_entries = static_cast<std::uint64_t>(slot_count);
        if (slot_entries > std::numeric_limits<std::uint64_t>::max() / capacity_entries ||
            slot_entries * capacity_entries > std::numeric_limits<std::uint64_t>::max() / sizeof(CandidateMeta)) {
            throw std::overflow_error("candidate history pinned slot bytes overflow");
        }
        bytes_pinned_slots = slot_entries * capacity_entries * sizeof(CandidateMeta);
        const std::uint64_t staging_entries_per_slot =
            std::min<std::uint64_t>(capacity_entries, kWriteChunkEntries);
        if (slot_entries > std::numeric_limits<std::uint64_t>::max() / staging_entries_per_slot ||
            slot_entries * staging_entries_per_slot > std::numeric_limits<std::uint64_t>::max() / sizeof(HistoryEntry)) {
            throw std::overflow_error("candidate history staging bytes overflow");
        }
        bytes_slot_staging = slot_entries * staging_entries_per_slot * sizeof(HistoryEntry);

        if(rust_history) {
            prune_enabled=selected_mode==CandidateHistoryMode::Ram && env_u32("WORLD_SIZE",1)==1;
            nlohmann::json config={{"mode",history_mode_name(selected_mode)},
                {"capacity",capacity},{"slots",slot_count},{"workers",selected_worker_count},
                {"max_depth",depth_limit},{"moves",MOVE_COUNT},
                {"ram_budget_bytes",history_ram_budget_bytes},{"disk_budget_bytes",history_disk_budget_bytes},
                {"directory",std::filesystem::absolute(dir).string()},{"prune",prune_enabled}};
            config["disk_path"]=history_disk_path.empty() ? nlohmann::json(nullptr) :
                nlohmann::json(std::filesystem::absolute(history_disk_path).string());
            rust_history.initialize(config.dump());
            rust_history_open=true;
            sync_rust_stats();
            if(selected_mode==CandidateHistoryMode::StaticHybrid && bytes_static_disk_arena)
                static_disk_path=history_disk_path.empty() ? dir/"history_static_arena.bin" : history_disk_path;
            std::cout<<"history_provider=rust\n";
        }

        if (!rust_history && selected_mode == CandidateHistoryMode::StaticHybrid) {
            if (history_ram_budget_bytes <= bytes_pinned_slots ||
                history_ram_budget_bytes - bytes_pinned_slots <= bytes_slot_staging) {
                throw std::runtime_error(
                    "static hybrid history RAM budget cannot fit pinned slots and staging: ram_budget=" +
                    std::to_string(history_ram_budget_bytes) +
                    " pinned_slots=" + std::to_string(bytes_pinned_slots) +
                    " staging=" + std::to_string(bytes_slot_staging));
            }
            budget_estimate = estimate_history_budget_entries(
                depth_limit,
                solved_neighborhood_radius,
                stream2_suffix_radius,
                capacity_entries);
            if (budget_estimate.required_entries >
                std::numeric_limits<std::uint64_t>::max() / sizeof(HistoryEntry)) {
                throw std::overflow_error("static hybrid history required bytes overflow");
            }
            history_required_bytes = budget_estimate.required_entries * sizeof(HistoryEntry);
            bytes_static_ram_arena = history_ram_budget_bytes - bytes_pinned_slots - bytes_slot_staging;
            ram_arena_entries = bytes_static_ram_arena / sizeof(HistoryEntry);
            bytes_static_ram_arena = ram_arena_entries * sizeof(HistoryEntry);
            disk_arena_entries = history_disk_budget_bytes / sizeof(HistoryEntry);
            bytes_static_disk_arena = disk_arena_entries * sizeof(HistoryEntry);
            // Validate placement, not only the sum of two arenas. All future
            // RAM layers are reserved before a disk-error fallback can run.
            static_plan = plan_history(depth_limit, capacity_entries,
                static_cast<std::uint32_t>(MOVE_COUNT), ram_arena_entries, disk_arena_entries);
            ram_entries_used = static_plan.ram_reserved_entries;
            disk_entries_used = static_plan.disk_reserved_entries;
            if (ram_arena_entries != 0ULL) {
                if (bytes_static_ram_arena > std::numeric_limits<std::size_t>::max())
                    throw std::overflow_error("static history RAM arena exceeds size_t");
                ram_arena = static_cast<HistoryEntry*>(std::malloc(bytes_static_ram_arena));
                if (ram_arena == nullptr) {
                    throw std::bad_alloc();
                }
                // Commit writable pages before any search work. This can fail
                // at startup under the host limit; it is not memory locking.
                host_memory::prefault_pages(ram_arena, static_cast<std::size_t>(bytes_static_ram_arena));
            }
            if (bytes_static_disk_arena != 0ULL) {
                static_disk_path = history_disk_path.empty() ? dir / "history_static_arena.bin" : history_disk_path;
                const std::filesystem::path parent = static_disk_path.parent_path();
                if (!parent.empty()) {
                    std::filesystem::create_directories(parent);
                }
                {
                    std::ofstream create_file(static_disk_path, std::ios::binary | std::ios::trunc);
                    if (!create_file) {
                        throw std::runtime_error("cannot create static history disk arena: " + static_disk_path.string());
                    }
                }
                std::filesystem::resize_file(static_disk_path, bytes_static_disk_arena);
#if defined(__linux__)
                if (bytes_static_disk_arena > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
                    throw std::overflow_error("static history disk reservation exceeds off_t");
                }
                const int descriptor = open(static_disk_path.c_str(), O_RDWR | O_CLOEXEC);
                if (descriptor < 0) {
                    throw std::runtime_error("cannot open static history disk reservation: " + static_disk_path.string());
                }
                int reserve_error;
                do {
                    reserve_error = posix_fallocate(descriptor, 0, static_cast<off_t>(bytes_static_disk_arena));
                } while (reserve_error == EINTR);
                const int close_error = close(descriptor);
                if (reserve_error != 0 || close_error != 0) {
                    throw std::runtime_error("static history physical disk reservation failed: " +
                        static_disk_path.string() + " error=" + std::to_string(reserve_error));
                }
#else
                throw std::runtime_error("static history physical disk reservation requires Linux");
#endif
            }
            prune_enabled = false;
        }

        BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
        slots.resize(slot_count);
        for (Slot& slot : slots) {
            slot.capacity = capacity;
            if(!rust_history) slot.staging.resize(static_cast<std::size_t>(staging_entries_per_slot));
            const std::size_t host_bytes = static_cast<std::size_t>(capacity) * sizeof(CandidateMeta);
            if (env_u32("BEAM_HISTORY_CHUNKED_PIN", 0) != 0) {
#if defined(__linux__)
                // Startup only. Keep one contiguous host address range; some hosts
                // reject large single cudaHostAlloc calls but accept bounded registrations.
                constexpr std::size_t page = 4096;
                constexpr std::size_t chunk = std::size_t{1} << 30;
                const std::size_t bytes = (host_bytes + page - 1) / page * page;
                void* base = nullptr;
                if (posix_memalign(&base, page, bytes) != 0) throw std::bad_alloc();
                slot.host = static_cast<CandidateMeta*>(base);
                while (slot.registered_bytes < bytes) {
                    const std::size_t count = std::min(chunk, bytes - slot.registered_bytes);
                    const cudaError_t rc = cudaHostRegister(
                        static_cast<char*>(base) + slot.registered_bytes, count, cudaHostRegisterPortable);
                    if (rc != cudaSuccess) {
                        for (std::size_t offset = 0; offset < slot.registered_bytes; offset += chunk)
                            cudaHostUnregister(static_cast<char*>(base) + offset);
                        std::free(base);
                        slot.host = nullptr;
                        slot.registered_bytes = 0;
                        BEAM_CUDA_CHECK(rc);
                    }
                    slot.registered_bytes += count;
                }
#else
                throw std::runtime_error("BEAM_HISTORY_CHUNKED_PIN requires Linux");
#endif
            } else {
                BEAM_CUDA_CHECK(cudaHostAlloc(
                    reinterpret_cast<void**>(&slot.host), host_bytes, cudaHostAllocPortable));
            }
            BEAM_CUDA_CHECK(cudaEventCreateWithFlags(&slot.copy_done, cudaEventDisableTiming));
        }
    }

    static HistoryEntry compress_candidate(const CandidateMeta& candidate) {
        return HistoryEntry{candidate.parent_idx, candidate.route_packed, 0U};
    }

    static std::vector<HistoryEntry> materialize_depth_ram_vector(
        const CandidateMeta* host,
        std::uint32_t count) {
        std::vector<HistoryEntry> entries(static_cast<std::size_t>(count));
        for (std::uint32_t i = 0; i < count; ++i) {
            entries[static_cast<std::size_t>(i)] = compress_candidate(host[i]);
        }
        return entries;
    }

    static void fill_history_staging(
        HistoryEntry* staging,
        const CandidateMeta* host,
        std::uint64_t begin,
        std::uint32_t count) {
        for (std::uint32_t i = 0; i < count; ++i) {
            staging[i] = compress_candidate(host[begin + i]);
        }
    }

    static void write_depth_file(
        const std::filesystem::path& path,
        const CandidateMeta* host,
        std::uint32_t count,
        HistoryEntry* staging,
        std::uint32_t staging_capacity) {
        std::ofstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open candidate history file for write: " + path.string());
        }
        for (std::uint64_t begin = 0; begin < count; begin += staging_capacity) {
            const std::uint32_t chunk =
                static_cast<std::uint32_t>(std::min<std::uint64_t>(staging_capacity, count - begin));
            fill_history_staging(staging, host, begin, chunk);
            file.write(
                reinterpret_cast<const char*>(staging),
                static_cast<std::streamsize>(static_cast<std::uint64_t>(chunk) * sizeof(HistoryEntry)));
        }
        file.flush();
        if (!file) {
            throw std::runtime_error("candidate history write failed: " + path.string());
        }
    }

    void write_static_disk(
        std::uint64_t offset_entries,
        const CandidateMeta* host,
        std::uint32_t count,
        HistoryEntry* staging,
        std::uint32_t staging_capacity) {
        std::lock_guard<std::mutex> lock(static_disk_mutex);
        std::fstream file(static_disk_path, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) {
            throw std::runtime_error("cannot open static history disk arena: " + static_disk_path.string());
        }
        file.seekp(static_cast<std::streamoff>(offset_entries * sizeof(HistoryEntry)), std::ios::beg);
        for (std::uint64_t begin = 0; begin < count; begin += staging_capacity) {
            const std::uint32_t chunk =
                static_cast<std::uint32_t>(std::min<std::uint64_t>(staging_capacity, count - begin));
            fill_history_staging(staging, host, begin, chunk);
            file.write(
                reinterpret_cast<const char*>(staging),
                static_cast<std::streamsize>(static_cast<std::uint64_t>(chunk) * sizeof(HistoryEntry)));
        }
        file.flush();
        if (!file) {
            throw std::runtime_error("static history disk arena write failed: " + static_disk_path.string());
        }
    }

    void write_static_ram(std::uint64_t offset_entries, const CandidateMeta* host, std::uint32_t count) {
        if (ram_arena == nullptr || offset_entries + count > ram_arena_entries) {
            throw std::runtime_error("static history RAM arena write exceeds capacity");
        }
        HistoryEntry* out = ram_arena + offset_entries;
        for (std::uint32_t i = 0; i < count; ++i) {
            out[i] = compress_candidate(host[i]);
        }
    }

    DepthLocation reserve_static_location(std::uint32_t depth_index, std::uint32_t count) {
        const auto& reservation = static_plan.reservation(depth_index, count);
        std::lock_guard<std::mutex> lock(storage_mutex);
        if (reservation.location == HistoryStorageLocation::Disk && disk_write_failed) {
            const auto offset = reserve_history_fallback_ram(count, ram_arena_entries, ram_entries_used);
            return DepthLocation{HistoryStorageLocation::Ram, offset};
        }
        return DepthLocation{reservation.location, reservation.offset_entries};
    }

    DepthLocation reserve_static_ram_fallback(std::uint32_t count, const std::string& disk_error) {
        const std::uint64_t count_entries = static_cast<std::uint64_t>(count);
        std::lock_guard<std::mutex> lock(storage_mutex);
        disk_write_failed = true;
        if (ram_entries_used <= ram_arena_entries && count_entries <= ram_arena_entries - ram_entries_used) {
            const auto offset = reserve_history_fallback_ram(count_entries, ram_arena_entries, ram_entries_used);
            return DepthLocation{HistoryStorageLocation::Ram, offset};
        }
        throw std::runtime_error(
            "static hybrid history disk write failed and RAM fallback is exhausted: disk_error=" + disk_error +
            " count=" + std::to_string(count) +
            " ram_used=" + std::to_string(ram_entries_used * sizeof(HistoryEntry)) +
            " ram_capacity=" + std::to_string(bytes_static_ram_arena));
    }

    MaterializeResult materialize_depth(
        CandidateHistoryMode selected_mode,
        const std::filesystem::path& path,
        const CandidateMeta* host,
        std::uint32_t count,
        std::uint32_t depth_index,
        HistoryStorageLocation location,
        std::uint64_t offset_entries,
        HistoryEntry* staging,
        std::uint32_t staging_capacity) {
        MaterializeResult result{depth_index, location, offset_entries, {}};
        if(rust_history) {
            rust_history.write(depth_index,host,count);
            return result;
        }
        if (selected_mode == CandidateHistoryMode::Ram) {
            result.ram_data = materialize_depth_ram_vector(host, count);
            return result;
        }
        if (selected_mode == CandidateHistoryMode::Disk) {
            write_depth_file(path, host, count, staging, staging_capacity);
            return result;
        }
        if (location == HistoryStorageLocation::Disk) {
            try {
                write_static_disk(offset_entries, host, count, staging, staging_capacity);
            } catch (const std::exception& ex) {
                const DepthLocation fallback = reserve_static_ram_fallback(count, ex.what());
                write_static_ram(fallback.offset_entries, host, count);
                result.location = fallback.location;
                result.offset_entries = fallback.offset_entries;
            }
            return result;
        }
        if (location == HistoryStorageLocation::Ram) {
            write_static_ram(offset_entries, host, count);
            return result;
        }
        throw std::runtime_error("static hybrid history materialize missing storage location");
    }

    static PruneResult prune_adjacent_depths(
        std::uint32_t depth_index,
        std::vector<HistoryEntry> previous,
        std::vector<HistoryEntry> current) {
        constexpr std::uint32_t invalid = std::numeric_limits<std::uint32_t>::max();
        const std::uint64_t bytes_before =
            (static_cast<std::uint64_t>(previous.size()) + static_cast<std::uint64_t>(current.size())) *
            sizeof(HistoryEntry);
        std::vector<std::uint32_t> remap(previous.size(), invalid);
        for (const HistoryEntry& entry : current) {
            if (entry.parent_idx >= previous.size()) {
                throw std::runtime_error("candidate history prune parent index out of range");
            }
            remap[static_cast<std::size_t>(entry.parent_idx)] = 1U;
        }

        std::vector<HistoryEntry> compact_previous;
        compact_previous.reserve(previous.size());
        for (std::size_t i = 0; i < previous.size(); ++i) {
            if (remap[i] != invalid) {
                remap[i] = static_cast<std::uint32_t>(compact_previous.size());
                compact_previous.push_back(previous[i]);
            }
        }
        for (HistoryEntry& entry : current) {
            entry.parent_idx = remap[static_cast<std::size_t>(entry.parent_idx)];
        }
        const std::uint64_t bytes_after =
            (static_cast<std::uint64_t>(compact_previous.size()) + static_cast<std::uint64_t>(current.size())) *
            sizeof(HistoryEntry);
        return PruneResult{depth_index, std::move(compact_previous), std::move(current), bytes_before, bytes_after};
    }

    bool prune_job_uses_depth(const PruneJob& job, std::uint32_t depth_index) const {
        return job.depth_index == depth_index || job.depth_index + 1U == depth_index;
    }

    bool any_prune_job_uses_pair(std::uint32_t depth_index) const {
        for (const PruneJob& job : prune_jobs) {
            if (prune_job_uses_depth(job, depth_index) || prune_job_uses_depth(job, depth_index + 1U)) {
                return true;
            }
        }
        return false;
    }

    void pump_prune_jobs(bool wait_all) {
        if(rust_history) return;
        if (!prune_enabled) {
            return;
        }
        bool progressed = true;
        while (progressed) {
            progressed = false;
            for (std::size_t i = 0; i < prune_jobs.size();) {
                PruneJob& job = prune_jobs[i];
                const bool ready = wait_all ||
                    job.worker.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                if (!ready) {
                    ++i;
                    continue;
                }
                PruneResult result = job.worker.get();
                if (result.depth_index + 1U >= ram_depths.size()) {
                    throw std::runtime_error("candidate history prune result depth index out of range");
                }
                ram_depths[result.depth_index] = std::move(result.previous);
                ram_depths[result.depth_index + 1U] = std::move(result.current);
                depth_counts[result.depth_index] =
                    static_cast<std::uint32_t>(ram_depths[result.depth_index].size());
                depth_counts[result.depth_index + 1U] =
                    static_cast<std::uint32_t>(ram_depths[result.depth_index + 1U].size());
                ram_ready[result.depth_index] = true;
                ram_ready[result.depth_index + 1U] = true;
                prune_dirty[result.depth_index] = false;
                if (result.bytes_before > result.bytes_after) {
                    bytes_stored -= (result.bytes_before - result.bytes_after);
                    bytes_pruned += (result.bytes_before - result.bytes_after);
                }
                if (result.depth_index > 0U) {
                    prune_dirty[result.depth_index - 1U] = true;
                }
                prune_jobs.erase(prune_jobs.begin() + static_cast<std::ptrdiff_t>(i));
                progressed = true;
            }

            while (mode == CandidateHistoryMode::Ram && prune_jobs.size() < worker_count) {
                bool launched = false;
                if (ram_depths.size() >= 2U) {
                    for (std::uint32_t depth = static_cast<std::uint32_t>(ram_depths.size() - 2U);
                         depth != std::numeric_limits<std::uint32_t>::max();
                         --depth) {
                        if (depth + 1U < ram_depths.size() &&
                            prune_dirty[depth] &&
                            ram_ready[depth] &&
                            ram_ready[depth + 1U] &&
                            !any_prune_job_uses_pair(depth)) {
                            std::vector<HistoryEntry> previous = std::move(ram_depths[depth]);
                            std::vector<HistoryEntry> current = std::move(ram_depths[depth + 1U]);
                            ram_ready[depth] = false;
                            ram_ready[depth + 1U] = false;
                            prune_jobs.push_back(PruneJob{
                                depth,
                                std::async(
                                    std::launch::async,
                                    &CpuCandidateHistory::prune_adjacent_depths,
                                    depth,
                                    std::move(previous),
                                    std::move(current))});
                            launched = true;
                            progressed = true;
                            break;
                        }
                        if (depth == 0U) {
                            break;
                        }
                    }
                }
                if (!launched) {
                    break;
                }
            }
            if (!wait_all) {
                break;
            }
        }
    }

    void pump_completed(bool wait_all) {
        bool progressed = true;
        while (progressed) {
            progressed = false;
            for (Slot& slot : slots) {
                if (slot.copy_pending) {
                    const cudaError_t status = wait_all ? cudaEventSynchronize(slot.copy_done) : cudaEventQuery(slot.copy_done);
                    if (status == cudaSuccess) {
                        slot.copy_pending = false;
                        const CandidateHistoryMode selected_mode = mode;
                        const std::filesystem::path path = slot.path;
                        const CandidateMeta* host = slot.host;
                        const std::uint32_t count = slot.count;
                        const std::uint32_t depth_index = slot.depth_index;
                        const HistoryStorageLocation location = slot.location;
                        const std::uint64_t offset_entries = slot.offset_entries;
                        HistoryEntry* staging = slot.staging.data();
                        const std::uint32_t staging_capacity =
                            static_cast<std::uint32_t>(slot.staging.size());
                        slot.writer = std::async(
                            std::launch::async,
                            [this,
                             selected_mode,
                             path,
                             host,
                             count,
                             depth_index,
                             location,
                             offset_entries,
                             staging,
                             staging_capacity]() {
                                return materialize_depth(
                                    selected_mode,
                                    path,
                                    host,
                                    count,
                                    depth_index,
                                    location,
                                    offset_entries,
                                    staging,
                                    staging_capacity);
                             });
                        progressed = true;
                    } else if (status != cudaErrorNotReady) {
                        BEAM_CUDA_CHECK(status);
                    }
                }
                if (!slot.copy_pending && !slot.free && slot.writer.valid()) {
                    const bool ready = wait_all ||
                        slot.writer.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                    if (ready) {
                        MaterializeResult result = slot.writer.get();
                        if(rust_history) {
                            sync_rust_stats();
                        } else if (mode == CandidateHistoryMode::Ram) {
                            if (slot.depth_index >= ram_depths.size()) {
                                throw std::runtime_error("candidate history RAM depth index missing");
                            }
                            ram_depths[slot.depth_index] = std::move(result.ram_data);
                            ram_ready[slot.depth_index] = true;
                            if (prune_enabled && slot.depth_index > 0U) {
                                prune_dirty[slot.depth_index - 1U] = true;
                            }
                        } else if (mode == CandidateHistoryMode::StaticHybrid) {
                            if (slot.depth_index >= ram_ready.size()) {
                                throw std::runtime_error("static history depth index missing");
                            }
                            if (result.depth_index >= depth_locations.size()) {
                                throw std::runtime_error("static history result depth index missing");
                            }
                            depth_locations[result.depth_index] =
                                DepthLocation{result.location, result.offset_entries};
                            depth_files[result.depth_index] =
                                result.location == HistoryStorageLocation::Disk ? static_disk_path : std::filesystem::path{};
                            const std::uint64_t stored_bytes =
                                static_cast<std::uint64_t>(slot.count) * sizeof(HistoryEntry);
                            if (result.location == HistoryStorageLocation::Disk) {
                                bytes_stored_disk += stored_bytes;
                            } else if (result.location == HistoryStorageLocation::Ram) {
                                bytes_stored_ram += stored_bytes;
                            }
                            ram_ready[slot.depth_index] = true;
                        }
                        slot.free = true;
                        progressed = true;
                    }
                }
            }
            if (!wait_all) {
                break;
            }
        }
        pump_prune_jobs(wait_all);
    }

    Slot& acquire_slot() {
        pump_completed(false);
        for (Slot& slot : slots) {
            if (slot.free) {
                slot.free = false;
                return slot;
            }
        }
        pump_completed(true);
        for (Slot& slot : slots) {
            if (slot.free) {
                slot.free = false;
                return slot;
            }
        }
        throw std::runtime_error("candidate history slot acquisition failed");
    }

    void commit_slot(Slot& slot, std::uint32_t depth_index, std::uint32_t count) {
        if (count > slot.capacity) {
            throw std::runtime_error("candidate history count exceeds slot capacity");
        }
        slot.count = count;
        slot.depth_index = depth_index;
        slot.path = dir / ("depth_" + std::to_string(depth_index) + ".candidate_meta.bin");
        slot.location = HistoryStorageLocation::None;
        slot.offset_entries = 0;
        slot.copy_pending = true;
        slot.free = false;
        if (depth_index != depth_counts.size()) {
            throw std::runtime_error("candidate history commits must be depth-ordered");
        }
        if(rust_history) {
            depth_files.push_back(slot.path);
            depth_counts.push_back(count);
            return;
        }
        if (mode == CandidateHistoryMode::StaticHybrid) {
            const DepthLocation location = reserve_static_location(depth_index, count);
            slot.location = location.location;
            slot.offset_entries = location.offset_entries;
            depth_locations.push_back(DepthLocation{slot.location, slot.offset_entries});
            depth_files.push_back(slot.location == HistoryStorageLocation::Disk ? static_disk_path : std::filesystem::path{});
        } else {
            depth_files.push_back(mode == CandidateHistoryMode::Disk ? slot.path : std::filesystem::path{});
        }
        depth_counts.push_back(count);
        if (mode == CandidateHistoryMode::Ram) {
            ram_depths.emplace_back();
            ram_ready.push_back(false);
            prune_dirty.push_back(false);
        } else if (mode == CandidateHistoryMode::StaticHybrid) {
            ram_ready.push_back(false);
            prune_dirty.push_back(false);
        }
        bytes_received += static_cast<std::uint64_t>(count) * sizeof(CandidateMeta);
        bytes_stored += static_cast<std::uint64_t>(count) * sizeof(HistoryEntry);
        if (mode == CandidateHistoryMode::Ram) {
            bytes_stored_ram += static_cast<std::uint64_t>(count) * sizeof(HistoryEntry);
        } else if (mode == CandidateHistoryMode::Disk) {
            bytes_stored_disk += static_cast<std::uint64_t>(count) * sizeof(HistoryEntry);
        }
    }

    void finish_all() {
        pump_completed(true);
        pump_prune_jobs(true);
    }

    void destroy() {
        if(slots.empty() && !ram_arena && !copy_stream && !rust_history_open) return;
        std::exception_ptr failure;
        try { finish_all(); } catch(...) { failure = std::current_exception(); }
        // A file writer may still be reading a pinned slot after its D2H event
        // completed. Join every reader before releasing either host allocation.
        if(copy_stream) cudaStreamSynchronize(copy_stream);
        for(auto& slot : slots) if(slot.writer.valid()) slot.writer.wait();
        for(auto& job : prune_jobs) if(job.worker.valid()) job.worker.wait();
        for (Slot& slot : slots) {
            if (slot.copy_done != nullptr) {
                cudaEventDestroy(slot.copy_done);
                slot.copy_done = nullptr;
            }
            if (slot.host != nullptr) {
                if (slot.registered_bytes != 0) {
                    constexpr std::size_t chunk = std::size_t{1} << 30;
                    for (std::size_t offset = 0; offset < slot.registered_bytes; offset += chunk)
                        cudaHostUnregister(reinterpret_cast<char*>(slot.host) + offset);
                    std::free(slot.host);
                    slot.registered_bytes = 0;
                } else {
                    cudaFreeHost(slot.host);
                }
                slot.host = nullptr;
            }
        }
        slots.clear();
        if (ram_arena != nullptr) {
            std::free(ram_arena);
            ram_arena = nullptr;
        }
        if (copy_stream != nullptr) {
            cudaStreamDestroy(copy_stream);
            copy_stream = nullptr;
        }
        if(rust_history_open) {
            try {rust_history.close();} catch(...) {if(!failure) failure=std::current_exception();}
            rust_history_open=false;
        }
        if(failure) std::rethrow_exception(failure);
    }

    HistoryEntry read_entry(std::uint32_t depth_index, std::uint64_t index) const {
        if(rust_history) {
            const auto entry=rust_history.read(depth_index,index);
            return {entry.parent_idx,entry.route,0};
        }
        const auto traced = [depth_index, index](HistoryEntry entry, const char* tier) {
#if BEAM_ENABLE_DEBUG_LOGS
            std::ostringstream line;
            line << "history_read_ok rank=" << env_u32("RANK", 0)
                 << " tier=" << tier << " depth_index=" << depth_index
                 << " entry_index=" << index << '\n';
            std::cout << line.str();
#else
            (void)depth_index; (void)index; (void)tier;
#endif
            return entry;
        };
        if (depth_index >= depth_counts.size()) {
            throw std::out_of_range("candidate history depth index out of range");
        }
        if (index >= depth_counts[depth_index]) {
            throw std::out_of_range("candidate history parent index out of range");
        }
        if (mode == CandidateHistoryMode::Ram) {
            if (depth_index >= ram_depths.size() || !ram_ready[depth_index]) {
                throw std::runtime_error("candidate history RAM depth not materialized");
            }
            return traced(ram_depths[depth_index][static_cast<std::size_t>(index)], "ram");
        }
        if (mode == CandidateHistoryMode::StaticHybrid) {
            if (depth_index >= depth_locations.size() || depth_index >= ram_ready.size() || !ram_ready[depth_index]) {
                throw std::runtime_error("static hybrid history depth not materialized");
            }
            const DepthLocation location = depth_locations[depth_index];
            if (location.location == HistoryStorageLocation::Ram) {
                if (ram_arena == nullptr || location.offset_entries + index >= ram_arena_entries) {
                    throw std::runtime_error("static hybrid history RAM read exceeds capacity");
                }
                return traced(ram_arena[location.offset_entries + index], "ram");
            }
            if (location.location == HistoryStorageLocation::Disk) {
                std::ifstream file(static_disk_path, std::ios::binary);
                if (!file) {
                    throw std::runtime_error("cannot open static history disk arena for read: " + static_disk_path.string());
                }
                const std::uint64_t offset = (location.offset_entries + index) * sizeof(HistoryEntry);
                file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
                HistoryEntry entry{};
                file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
                if (!file) {
                    throw std::runtime_error("static history disk arena read failed: " + static_disk_path.string());
                }
                return traced(entry, "disk");
            }
            throw std::runtime_error("static hybrid history read missing storage location");
        }
        std::ifstream file(depth_files[depth_index], std::ios::binary);
        if (!file) {
            throw std::runtime_error("cannot open candidate history file for read: " + depth_files[depth_index].string());
        }
        const std::uint64_t offset = index * sizeof(HistoryEntry);
        file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        HistoryEntry entry{};
        file.read(reinterpret_cast<char*>(&entry), sizeof(entry));
        if (!file) {
            throw std::runtime_error("candidate history read failed: " + depth_files[depth_index].string());
        }
        return traced(entry, "disk");
    }
};
