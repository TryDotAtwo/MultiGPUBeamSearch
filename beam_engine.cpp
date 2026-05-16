#include <torch/extension.h>
#include <torch/script.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nccl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "beam_engine_common.hpp"
#include "beam_config.hpp"
#include "beam_memory.hpp"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef BEAM_HISTORY_CPU
#define BEAM_HISTORY_CPU 0
#endif

#ifndef BEAM_DEBUG_ON
#define BEAM_DEBUG_ON 0
#endif


namespace py = pybind11;
using namespace beam_engine;

#define CUDA_CHECK(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    } \
} while (0)

#define NCCL_CHECK(expr) do { \
    ncclResult_t _res = (expr); \
    if (_res != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(_res)); \
    } \
} while (0)

struct DebugConfig {
    bool verbose = false;
    bool print_counters = false;
    int log_period = 8;
};
static DebugConfig debug_config;

extern "C" void launch_dummy_inference(const uint8_t*, const uint8_t*, uint16_t*, int, int, int, int, int64_t, int, cudaStream_t);
extern "C" void launch_uniform_inference(const uint8_t*, uint16_t*, int, int, int, int64_t, int, uint16_t, cudaStream_t);
extern "C" void launch_copy_i16_scores_to_ring(const int16_t*, uint16_t*, int, int, int, int, cudaStream_t);
extern "C" void launch_quantize_f32_scores_to_ring(const float*, uint16_t*, int, int, int, int, float, float, cudaStream_t);
extern "C" void launch_fullbeamnice_embed_relu(const uint8_t*, const uint8_t*, const half*, const half*, half*, int, int, int, int64_t, int, cudaStream_t);
extern "C" void launch_fullbeamnice_cutlass_gemm(const half*, const half*, half*, int, int, int, int, cudaStream_t);
extern "C" void launch_fullbeamnice_fill_bias(half*, const half*, int, int, cudaStream_t);
extern "C" void launch_fullbeamnice_fill_residual_bias(half*, const half*, const half*, int, int, cudaStream_t);
extern "C" void launch_fullbeamnice_bias_relu(half*, const half*, int, int, cudaStream_t);
extern "C" void launch_fullbeamnice_residual_bias_relu(half*, const half*, const half*, int, int, cudaStream_t);
extern "C" void launch_fullbeamnice_quantize_to_ring(const half*, const half*, const int32_t*, uint16_t*, int, int, int, int, float, float, cudaStream_t);
extern "C" void launch_reset_net_slot(CandidateRecord*, CandidateRecord*, int32_t*, int32_t*, int, int, int, cudaStream_t);
extern "C" void launch_process_score_slot(const uint8_t*, const uint8_t*, const uint16_t*, uint8_t*, BeamMeta*, HashSlot*, uint8_t*, int32_t*, int32_t*, uint32_t*, int32_t*, int32_t*, CandidateRecord*, int32_t*, const int32_t*, int, int, int, int, int, int, int, int64_t, int, int64_t, int, int, int, int, int, cudaStream_t);
extern "C" void launch_ingest_recv_slot(const CandidateRecord*, const int32_t*, uint8_t*, BeamMeta*, HashSlot*, uint8_t*, int32_t*, int32_t*, uint32_t*, int32_t*, int32_t*, const int32_t*, int, int, int, int, int, int, int, cudaStream_t);
extern "C" void launch_compute_threshold(const uint32_t*, int32_t*, int, int64_t, cudaStream_t);
extern "C" void launch_prune_by_threshold(BeamMeta*, HashSlot*, uint8_t*, int32_t*, int32_t*, uint32_t*, int32_t*, const int32_t*, int, int, int, cudaStream_t);
extern "C" void launch_compact_next_to_current(const uint8_t*, const BeamMeta*, const uint8_t*, uint8_t*, uint8_t*, int32_t*, uint8_t*, uint8_t*, uint8_t*, const int32_t*, int32_t*, int, int, int, cudaStream_t);
extern "C" void launch_clear_hash_table(HashSlot*, int, cudaStream_t);
extern "C" void launch_rebuild_hash_from_active(BeamMeta*, HashSlot*, const uint8_t*, int32_t*, int, int, int, cudaStream_t);
extern "C" void launch_clear_step_state(int32_t*, uint32_t*, int32_t*, uint8_t*, int32_t*, int32_t*, int, int, cudaStream_t);
extern "C" void launch_check_current_solved(const uint8_t*, const uint8_t*, int32_t*, int, int, cudaStream_t);
extern "C" void upload_action_permutation_table(const uint8_t*, int, int);
extern "C" void upload_central_state(const uint8_t*, int);
extern "C" void launch_v6_stream2_hash_goal(
    const beam_v6::State128*,
    const uint64_t*,
    const uint32_t*,
    const uint32_t*,
    beam_v6::Hash128*,
    const uint8_t*,
    const uint8_t*,
    const beam_v6::Hash128*,
    uint32_t*,
    uint32_t*,
    uint32_t*,
    uint32_t*,
    beam_v6::CandidateMeta*,
    uint32_t*,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    cudaStream_t);
extern "C" void launch_v6_final_materialize(
    const beam_v6::State128*,
    const beam_v6::FinalRequest*,
    const uint8_t*,
    beam_v6::FinalResponse*,
    uint32_t,
    cudaStream_t);
extern "C" void launch_v6_final_scatter_responses(
    const beam_v6::FinalResponse*,
    beam_v6::State128*,
    uint32_t,
    cudaStream_t);
namespace beam_v6 { struct Hash128Key; }
extern "C" void launch_v6_stream3_pack_threshold_compact(
    const uint32_t*,
    const beam_v6::Hash128*,
    const uint64_t*,
    const uint32_t*,
    beam_v6::Hash128Key*,
    uint64_t*,
    uint32_t*,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    cudaStream_t);
extern "C" void launch_v6_stream3_sort_pairs(
    void*,
    size_t&,
    const beam_v6::Hash128Key*,
    beam_v6::Hash128Key*,
    const uint64_t*,
    uint64_t*,
    int,
    cudaStream_t);
extern "C" void launch_v6_stream3_dedup_sorted(
    const beam_v6::Hash128Key*,
    const uint64_t*,
    beam_v6::Hash128*,
    uint64_t*,
    uint32_t*,
    uint32_t,
    cudaStream_t);
extern "C" void launch_v6_stream3_restore_split(
    const beam_v6::Hash128*,
    const uint64_t*,
    const uint64_t*,
    beam_v6::CandidateMeta*,
    beam_v6::CandidateMeta*,
    uint32_t*,
    uint32_t*,
    uint32_t*,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    uint32_t,
    cudaStream_t);
namespace beam_v6 { struct Stream4HashKey; }
extern "C" void launch_v6_stream4_threshold_compact(
    const beam_v6::CandidateMeta*,
    beam_v6::Stream4HashKey*,
    beam_v6::CandidateMeta*,
    uint32_t*,
    uint32_t,
    uint32_t,
    cudaStream_t);
extern "C" void launch_v6_stream4_sort_pairs(
    void*,
    size_t&,
    const beam_v6::Stream4HashKey*,
    beam_v6::Stream4HashKey*,
    const beam_v6::CandidateMeta*,
    beam_v6::CandidateMeta*,
    int,
    cudaStream_t);
extern "C" void launch_v6_stream4_dedup_sorted(
    const beam_v6::Stream4HashKey*,
    const beam_v6::CandidateMeta*,
    beam_v6::CandidateMeta*,
    uint32_t*,
    uint32_t,
    cudaStream_t);
extern "C" void launch_v6_stream4_write_clean(
    beam_v6::CandidateMeta*,
    const beam_v6::CandidateMeta*,
    uint32_t*,
    uint32_t*,
    uint8_t*,
    uint32_t,
    cudaStream_t);

// Helper: compute smallest power-of-2 >= x
static inline int64_t pow2_ceil(int64_t x) {
    if (x <= 1) return 1LL;
    // bit_position of (x-1): highest set bit position
    int bit_pos = 63 - __builtin_clzll(x - 1);
    return 1LL << (bit_pos + 1);
}

struct EngineConfig {
    int world_size = 1;
    int rank = 0;
    int fanout = FANOUT_FIXED;
    int state_size_bytes = STATE_SIZE_BYTES_FIXED;
    int b_micro = 131072;
    int score_ring_depth = 64;
    int net_ring_depth = 3;
    int probe_limit = 32;
    int bucket_cap_per_peer = 0;        // 0 = auto-derive (SAFE)
    int inference_parallelism = 1;
    int k_expand_tile = 0;
    float nn_score_scale = 1.0f;
    float nn_score_bias = 0.0f;
    int64_t global_beam_width = 1ll << 20;
    double gamma = 1.05;
    double beta = 1.10;
    double hash_load_factor = 0.60;
    int max_depth = 1;
    int64_t stream3_batch_candidates = 0;
    int64_t stream4_batch_candidates = 0;
    int64_t stream4_batch_candidates_per_shard_unit = 0;
    int ring_count = 2;
    int shard_count = 1;
    int64_t global_spill_capacity = 0;
    int64_t solved_result_capacity = 256;
    int64_t global_beam_width_max_safe = 0;
    int global_threshold_update_period_shards = 16;

    int64_t n_local = 0;
    int64_t k_keep = 0;
    int64_t k_work = 0;
    int64_t hash_capacity = 0;
    
    // Derived bucket sizes (for logging)
    int64_t bucket_cap_per_peer_safe = 0;
    double send_buckets_gib = 0.0;
    double recv_buckets_gib = 0.0;
    double total_bucket_gib = 0.0;

    void derive() {
        n_local = (global_beam_width + world_size - 1) / world_size;
        if (n_local < 1) n_local = 1;
        k_keep = static_cast<int64_t>(gamma * static_cast<double>(n_local) + 0.5);
        if (k_keep < n_local) k_keep = n_local;
        k_work = static_cast<int64_t>(beta * static_cast<double>(k_keep) + 0.5);
        if (k_work < k_keep) k_work = k_keep;
        hash_capacity = static_cast<int64_t>(static_cast<double>(k_work) / hash_load_factor + 0.5);
        if (hash_capacity < 1024) hash_capacity = 1024;

        // Formula: K_EXPAND_TILE = pow2_ceil(
        //     ceil(GLOBAL_BEAM_WIDTH * FANOUT / (WORLD_SIZE * TARGET_STREAM2_ROUNDS))
        // )
        // Auto-derive K_EXPAND_TILE if not explicitly set (0 = auto)
        if (k_expand_tile == 0) {
            const int TARGET_STREAM2_ROUNDS = 16;
            int64_t numerator = global_beam_width * static_cast<int64_t>(fanout);
            int64_t denominator = static_cast<int64_t>(world_size) * TARGET_STREAM2_ROUNDS;
            int64_t target_k_expand = (numerator + denominator - 1) / denominator;
            k_expand_tile = static_cast<int>(pow2_ceil(target_k_expand));
        }

        // Formula: SCORE_RING_DEPTH = pow2_ceil(
        //     ceil(K_EXPAND_TILE / (B_MICRO * FANOUT))
        // )
        // Auto-derive SCORE_RING_DEPTH if not explicitly set (0 = auto)
        if (score_ring_depth == 0) {
            int64_t numerator = static_cast<int64_t>(k_expand_tile);
            int64_t denominator = static_cast<int64_t>(b_micro) * static_cast<int64_t>(fanout);
            int64_t target_depth = (numerator + denominator - 1) / denominator;
            score_ring_depth = static_cast<int>(pow2_ceil(target_depth));
            if (score_ring_depth < 1) score_ring_depth = 1;
        }

        // Formula: BUCKET_CAP_PER_PEER = min(
        //     pow2_ceil(max(65536, K_EXPAND_TILE / 16)),
        //     2^20
        // )
        // BUCKET_CAP_PER_PEER_SAFE = min(
        //     pow2_ceil(max(131072, K_EXPAND_TILE / 8)),
        //     2^20
        // )
        const int64_t MAX_BUCKET_CAPACITY = 1LL << 20;  // 2^20 = 1M
        
        // Default (SAFE) derivation if bucket_cap_per_peer not explicitly set (0 = auto)
        if (bucket_cap_per_peer == 0) {
            int64_t base_safe = std::max<int64_t>(131072, static_cast<int64_t>(k_expand_tile) / 8);
            bucket_cap_per_peer_safe = static_cast<int64_t>(pow2_ceil(base_safe));
            if (bucket_cap_per_peer_safe > MAX_BUCKET_CAPACITY) {
                bucket_cap_per_peer_safe = MAX_BUCKET_CAPACITY;
            }
            bucket_cap_per_peer = static_cast<int>(bucket_cap_per_peer_safe);
        } else {
            // User-provided explicit value, compute SAFE variant for logging
            int64_t base_safe = std::max<int64_t>(131072, static_cast<int64_t>(k_expand_tile) / 8);
            bucket_cap_per_peer_safe = static_cast<int64_t>(pow2_ceil(base_safe));
            if (bucket_cap_per_peer_safe > MAX_BUCKET_CAPACITY) {
                bucket_cap_per_peer_safe = MAX_BUCKET_CAPACITY;
            }
        }

        // Compute bucket memory sizes for logging
        // Each peer has bucket_cap_per_peer slots, each slot is CandidateRecord (160 bytes)
        int64_t bytes_per_candidate = 160;  // CandidateRecord size
        int64_t send_buckets_bytes = static_cast<int64_t>(bucket_cap_per_peer) * bytes_per_candidate * static_cast<int64_t>(world_size - 1);
        int64_t recv_buckets_bytes = static_cast<int64_t>(bucket_cap_per_peer) * bytes_per_candidate * static_cast<int64_t>(world_size - 1);
        int64_t total_bucket_bytes = send_buckets_bytes + recv_buckets_bytes;
        
        send_buckets_gib = static_cast<double>(send_buckets_bytes) / (1024.0 * 1024.0 * 1024.0);
        recv_buckets_gib = static_cast<double>(recv_buckets_bytes) / (1024.0 * 1024.0 * 1024.0);
        total_bucket_gib = static_cast<double>(total_bucket_bytes) / (1024.0 * 1024.0 * 1024.0);
    }
};

static EngineConfig config_from_dict(const py::dict& d) {
    EngineConfig c;
    if (d.contains("world_size")) c.world_size = d["world_size"].cast<int>();
    if (d.contains("rank")) c.rank = d["rank"].cast<int>();
    if (d.contains("fanout")) c.fanout = d["fanout"].cast<int>();
    if (d.contains("state_size_bytes")) c.state_size_bytes = d["state_size_bytes"].cast<int>();
    if (d.contains("b_micro")) c.b_micro = d["b_micro"].cast<int>();
    if (d.contains("score_ring_depth")) c.score_ring_depth = d["score_ring_depth"].cast<int>();
    if (d.contains("net_ring_depth")) c.net_ring_depth = d["net_ring_depth"].cast<int>();
    if (d.contains("probe_limit")) c.probe_limit = d["probe_limit"].cast<int>();
    if (d.contains("bucket_cap_per_peer")) c.bucket_cap_per_peer = d["bucket_cap_per_peer"].cast<int>();
    if (d.contains("inference_parallelism")) c.inference_parallelism = d["inference_parallelism"].cast<int>();
    if (d.contains("k_expand_tile")) c.k_expand_tile = d["k_expand_tile"].cast<int>();
    if (d.contains("nn_score_scale")) c.nn_score_scale = d["nn_score_scale"].cast<float>();
    if (d.contains("nn_score_bias")) c.nn_score_bias = d["nn_score_bias"].cast<float>();
    if (d.contains("global_beam_width")) c.global_beam_width = d["global_beam_width"].cast<int64_t>();
    if (d.contains("gamma")) c.gamma = d["gamma"].cast<double>();
    if (d.contains("beta")) c.beta = d["beta"].cast<double>();
    if (d.contains("hash_load_factor")) c.hash_load_factor = d["hash_load_factor"].cast<double>();
    if (d.contains("max_depth")) c.max_depth = d["max_depth"].cast<int>();
    if (d.contains("stream3_batch_candidates")) c.stream3_batch_candidates = d["stream3_batch_candidates"].cast<int64_t>();
    if (d.contains("stream4_batch_candidates")) c.stream4_batch_candidates = d["stream4_batch_candidates"].cast<int64_t>();
    if (d.contains("stream4_batch_candidates_per_shard_unit")) c.stream4_batch_candidates_per_shard_unit = d["stream4_batch_candidates_per_shard_unit"].cast<int64_t>();
    if (d.contains("ring_count")) c.ring_count = d["ring_count"].cast<int>();
    if (d.contains("shard_count")) c.shard_count = d["shard_count"].cast<int>();
    if (d.contains("global_spill_capacity")) c.global_spill_capacity = d["global_spill_capacity"].cast<int64_t>();
    if (d.contains("solved_result_capacity")) c.solved_result_capacity = d["solved_result_capacity"].cast<int64_t>();
    if (d.contains("global_beam_width_max_safe")) c.global_beam_width_max_safe = d["global_beam_width_max_safe"].cast<int64_t>();
    if (d.contains("global_threshold_update_period_shards")) c.global_threshold_update_period_shards = d["global_threshold_update_period_shards"].cast<int>();
    if (c.max_depth < 1) c.max_depth = 1;
    if (c.inference_parallelism < 1) c.inference_parallelism = 1;
    if (c.inference_parallelism > c.score_ring_depth) c.inference_parallelism = c.score_ring_depth;
    if (c.k_expand_tile < 0) c.k_expand_tile = 0;
    if (c.fanout != FANOUT_FIXED || c.state_size_bytes != STATE_SIZE_BYTES_FIXED) {
        throw std::runtime_error("v1 supports fanout=24 and state_size_bytes=120 only");
    }
    c.derive();
    return c;
}

static beam_v6::TargetConfig target_config_from_engine(const EngineConfig& cfg) {
    beam_v6::TargetConfig target;
    target.b_micro = cfg.b_micro;
    target.inference_parallelism = cfg.inference_parallelism;
    target.stream3_batch_candidates = cfg.stream3_batch_candidates > 0
        ? cfg.stream3_batch_candidates
        : static_cast<int64_t>(cfg.score_ring_depth) * cfg.b_micro * cfg.fanout;
    target.stream4_batch_candidates = cfg.stream4_batch_candidates > 0
        ? cfg.stream4_batch_candidates
        : target.stream3_batch_candidates;
    target.stream4_batch_candidates_per_shard_unit = cfg.stream4_batch_candidates_per_shard_unit > 0
        ? cfg.stream4_batch_candidates_per_shard_unit
        : target.stream4_batch_candidates;
    target.ring_count = cfg.ring_count;
    target.world_size = cfg.world_size;
    target.local_rank = cfg.rank;
    target.shard_count = cfg.shard_count;
    target.global_spill_capacity = cfg.global_spill_capacity > 0
        ? cfg.global_spill_capacity
        : target.stream4_batch_candidates;
    target.solved_result_capacity = cfg.solved_result_capacity;
    target.user_global_beam_width = cfg.global_beam_width;
    target.global_beam_width_max_safe = cfg.global_beam_width_max_safe;
    target.global_threshold_update_period_shards = cfg.global_threshold_update_period_shards;
    return beam_v6::derive_target_config(target);
}

static void check_cuda_tensor(const torch::Tensor& t, const char* name) {
    if (!t.is_cuda()) throw std::runtime_error(std::string(name) + " must be a CUDA tensor");
    if (!t.is_contiguous()) throw std::runtime_error(std::string(name) + " must be contiguous");
}

static void check_cuda_half_tensor(const torch::Tensor& t, const char* name) {
    check_cuda_tensor(t, name);
    if (t.scalar_type() != at::kHalf) throw std::runtime_error(std::string(name) + " must be torch.float16");
}

static void check_cuda_i32_tensor(const torch::Tensor& t, const char* name) {
    check_cuda_tensor(t, name);
    if (t.scalar_type() != at::kInt) throw std::runtime_error(std::string(name) + " must be torch.int32");
}

static void check_cuda_i64_tensor(const torch::Tensor& t, const char* name) {
    check_cuda_tensor(t, name);
    if (t.scalar_type() != at::kLong) throw std::runtime_error(std::string(name) + " must be torch.int64");
}

static py::bytes nccl_unique_id_to_bytes(const ncclUniqueId& id) {
    return py::bytes(reinterpret_cast<const char*>(&id), sizeof(ncclUniqueId));
}

static ncclUniqueId nccl_unique_id_from_bytes(const py::bytes& b) {
    std::string s = b;
    if (s.size() != sizeof(ncclUniqueId)) throw std::runtime_error("bad ncclUniqueId size");
    ncclUniqueId id;
    std::memcpy(&id, s.data(), sizeof(ncclUniqueId));
    return id;
}

struct InferenceBackend {
    virtual ~InferenceBackend() = default;
    virtual void forward(const torch::Tensor& beam_current,
                         const torch::Tensor& current_active_flags,
                         torch::Tensor& score_ring,
                         int slot,
                         int lane,
                         int64_t start_state,
                         int micro_size,
                         const EngineConfig& cfg,
                         cudaStream_t stream) = 0;
};

struct DummyInferenceBackend final : public InferenceBackend {
    void forward(const torch::Tensor& beam_current,
                 const torch::Tensor& current_active_flags,
                 torch::Tensor& score_ring,
                 int slot,
                 int lane,
                 int64_t start_state,
                 int micro_size,
                 const EngineConfig& cfg,
                 cudaStream_t stream) override {
        launch_dummy_inference(
            reinterpret_cast<const uint8_t*>(beam_current.data_ptr<uint8_t>()),
            reinterpret_cast<const uint8_t*>(current_active_flags.data_ptr<uint8_t>()),
            reinterpret_cast<uint16_t*>(score_ring.data_ptr<int16_t>()),
            slot, cfg.state_size_bytes, cfg.fanout, cfg.b_micro, start_state, micro_size, stream);
        (void)lane;
        CUDA_CHECK(cudaGetLastError());
    }
};

struct UniformScoreBackend final : public InferenceBackend {
    uint16_t score_q = 1;
    explicit UniformScoreBackend(uint16_t score) : score_q(score) {}

    void forward(const torch::Tensor&,
                 const torch::Tensor& current_active_flags,
                 torch::Tensor& score_ring,
                 int slot,
                 int lane,
                 int64_t start_state,
                 int micro_size,
                 const EngineConfig& cfg,
                 cudaStream_t stream) override {
        launch_uniform_inference(
            reinterpret_cast<const uint8_t*>(current_active_flags.data_ptr<uint8_t>()),
            reinterpret_cast<uint16_t*>(score_ring.data_ptr<int16_t>()),
            slot, cfg.b_micro, cfg.fanout, start_state, micro_size, score_q, stream);
        (void)lane;
        CUDA_CHECK(cudaGetLastError());
    }
};

struct TorchScriptEnsembleBackend final : public InferenceBackend {
    std::vector<torch::jit::Module> modules;
    bool shared_module = false;

    explicit TorchScriptEnsembleBackend(const std::vector<std::string>& paths, const c10::Device& device) {
        if (paths.empty()) throw std::runtime_error("torchscript ensemble requires at least one module path");
        modules.reserve(paths.size());
        for (const auto& path : paths) {
            torch::jit::Module m = torch::jit::load(path, device);
            m.to(device);
            m.eval();
            modules.emplace_back(std::move(m));
        }
        shared_module = modules.size() == 1;
    }

    void forward(const torch::Tensor& beam_current,
                 const torch::Tensor&,
                 torch::Tensor& score_ring,
                 int slot,
                 int lane,
                 int64_t start_state,
                 int micro_size,
                 const EngineConfig& cfg,
                 cudaStream_t stream) override {
        if (modules.empty()) throw std::runtime_error("torchscript ensemble is empty");
        (void)lane;
        const int module_idx = shared_module ? 0 : slot % static_cast<int>(modules.size());
        auto torch_stream = c10::cuda::getStreamFromExternal(
            stream,
            static_cast<c10::DeviceIndex>(beam_current.device().index())
        );
        c10::cuda::CUDAStreamGuard guard(torch_stream);

        torch::NoGradGuard no_grad;
        torch::Tensor x = beam_current.narrow(0, start_state, micro_size);
        torch::Tensor y = modules[module_idx].forward({x}).toTensor();
        if (y.dim() != 2 || y.size(0) != micro_size || y.size(1) != cfg.fanout) {
            std::ostringstream oss;
            oss << "torchscript scorer must return [micro_size, fanout]; got [";
            for (int i = 0; i < y.dim(); ++i) {
                if (i) oss << ",";
                oss << y.size(i);
            }
            oss << "]";
            throw std::runtime_error(oss.str());
        }
        if (!y.is_contiguous()) y = y.contiguous();
        if (y.scalar_type() == at::kShort) {
            launch_copy_i16_scores_to_ring(
                reinterpret_cast<const int16_t*>(y.data_ptr<int16_t>()),
                reinterpret_cast<uint16_t*>(score_ring.data_ptr<int16_t>()),
                slot, cfg.b_micro, cfg.fanout, micro_size, stream);
        } else if (y.scalar_type() == at::kFloat) {
            launch_quantize_f32_scores_to_ring(
                reinterpret_cast<const float*>(y.data_ptr<float>()),
                reinterpret_cast<uint16_t*>(score_ring.data_ptr<int16_t>()),
                slot, cfg.b_micro, cfg.fanout, micro_size,
                cfg.nn_score_scale, cfg.nn_score_bias, stream);
        } else {
            throw std::runtime_error("torchscript scorer output dtype must be torch.int16 or torch.float32");
        }
        CUDA_CHECK(cudaGetLastError());
    }
};

struct TEInferenceBackend final : public InferenceBackend {
    void forward(const torch::Tensor&, const torch::Tensor&, torch::Tensor&, int, int, int64_t, int, const EngineConfig&, cudaStream_t) override {
        throw std::runtime_error("TEInferenceBackend placeholder: implement TE FP8 forward; optional TorchScript path requires ALLOW_TORCHSCRIPT_SCORER=1 in Python");
    }
};

struct FullBeamNiceStaticBackend final : public InferenceBackend {
    torch::Tensor embed_w_t, embed_bias;
    torch::Tensor hidden_w_t, hidden_bias;
    torch::Tensor res0_fc1_w_t, res0_fc1_bias, res0_fc2_w_t, res0_fc2_bias;
    torch::Tensor res1_fc1_w_t, res1_fc1_bias, res1_fc2_w_t, res1_fc2_bias;
    torch::Tensor out_w_t, out_bias, action_perm;
    torch::Tensor act1, act2, act3, out;
    float score_scale = 1024.0f;
    float score_bias = 65535.0f;
    int state_size = STATE_SIZE_BYTES_FIXED;
    int num_classes = 120;

    explicit FullBeamNiceStaticBackend(py::dict weights, py::dict buffers) {
        embed_w_t = weights["embed_w_t"].cast<torch::Tensor>();
        embed_bias = weights["embed_bias"].cast<torch::Tensor>();
        hidden_w_t = weights["hidden_w_t"].cast<torch::Tensor>();
        hidden_bias = weights["hidden_bias"].cast<torch::Tensor>();
        res0_fc1_w_t = weights["res0_fc1_w_t"].cast<torch::Tensor>();
        res0_fc1_bias = weights["res0_fc1_bias"].cast<torch::Tensor>();
        res0_fc2_w_t = weights["res0_fc2_w_t"].cast<torch::Tensor>();
        res0_fc2_bias = weights["res0_fc2_bias"].cast<torch::Tensor>();
        res1_fc1_w_t = weights["res1_fc1_w_t"].cast<torch::Tensor>();
        res1_fc1_bias = weights["res1_fc1_bias"].cast<torch::Tensor>();
        res1_fc2_w_t = weights["res1_fc2_w_t"].cast<torch::Tensor>();
        res1_fc2_bias = weights["res1_fc2_bias"].cast<torch::Tensor>();
        out_w_t = weights["out_w_t"].cast<torch::Tensor>();
        out_bias = weights["out_bias"].cast<torch::Tensor>();
        action_perm = weights["action_perm"].cast<torch::Tensor>();
        act1 = buffers["fb_act1"].cast<torch::Tensor>();
        act2 = buffers["fb_act2"].cast<torch::Tensor>();
        act3 = buffers["fb_act3"].cast<torch::Tensor>();
        out = buffers["fb_out"].cast<torch::Tensor>();
        if (weights.contains("score_scale")) score_scale = weights["score_scale"].cast<float>();
        if (weights.contains("score_bias")) score_bias = weights["score_bias"].cast<float>();
        if (weights.contains("state_size")) state_size = weights["state_size"].cast<int>();
        if (weights.contains("num_classes")) num_classes = weights["num_classes"].cast<int>();
        validate();
    }

    void validate() {
        check_cuda_half_tensor(embed_w_t, "embed_w_t");
        check_cuda_half_tensor(embed_bias, "embed_bias");
        check_cuda_half_tensor(hidden_w_t, "hidden_w_t");
        check_cuda_half_tensor(hidden_bias, "hidden_bias");
        check_cuda_half_tensor(res0_fc1_w_t, "res0_fc1_w_t");
        check_cuda_half_tensor(res0_fc1_bias, "res0_fc1_bias");
        check_cuda_half_tensor(res0_fc2_w_t, "res0_fc2_w_t");
        check_cuda_half_tensor(res0_fc2_bias, "res0_fc2_bias");
        check_cuda_half_tensor(res1_fc1_w_t, "res1_fc1_w_t");
        check_cuda_half_tensor(res1_fc1_bias, "res1_fc1_bias");
        check_cuda_half_tensor(res1_fc2_w_t, "res1_fc2_w_t");
        check_cuda_half_tensor(res1_fc2_bias, "res1_fc2_bias");
        check_cuda_half_tensor(out_w_t, "out_w_t");
        check_cuda_half_tensor(out_bias, "out_bias");
        check_cuda_i32_tensor(action_perm, "action_perm");
        check_cuda_half_tensor(act1, "fb_act1");
        check_cuda_half_tensor(act2, "fb_act2");
        check_cuda_half_tensor(act3, "fb_act3");
        check_cuda_half_tensor(out, "fb_out");
        if (state_size != STATE_SIZE_BYTES_FIXED || num_classes != 120) throw std::runtime_error("fullbeamnice_static supports state_size=120 and num_classes=120");
        if (embed_w_t.size(0) != state_size * num_classes || embed_w_t.size(1) != 1536) throw std::runtime_error("bad embed_w_t shape");
        if (hidden_w_t.size(0) != 1536 || hidden_w_t.size(1) != 512) throw std::runtime_error("bad hidden_w_t shape");
        if (out_w_t.size(0) != 512 || out_w_t.size(1) != FANOUT_FIXED) throw std::runtime_error("bad out_w_t shape");
    }

    void forward(const torch::Tensor& beam_current,
                 const torch::Tensor& current_active_flags,
                 torch::Tensor& score_ring,
                 int slot,
                 int lane,
                 int64_t start_state,
                 int micro_size,
                 const EngineConfig& cfg,
                 cudaStream_t stream) override {
        if (micro_size <= 0) return;
        const int lane_idx = lane % cfg.inference_parallelism;
        auto* act1_base = reinterpret_cast<half*>(act1.data_ptr<at::Half>());
        auto* act2_base = reinterpret_cast<half*>(act2.data_ptr<at::Half>());
        auto* act3_base = reinterpret_cast<half*>(act3.data_ptr<at::Half>());
        auto* out_base = reinterpret_cast<half*>(out.data_ptr<at::Half>());
        auto* act1_ptr = act1_base + static_cast<int64_t>(lane_idx) * cfg.b_micro * 1536;
        auto* act2_ptr = act2_base + static_cast<int64_t>(lane_idx) * cfg.b_micro * 512;
        auto* act3_ptr = act3_base + static_cast<int64_t>(lane_idx) * cfg.b_micro * 512;
        auto* out_ptr = out_base + static_cast<int64_t>(lane_idx) * cfg.b_micro * FANOUT_FIXED;
        launch_fullbeamnice_embed_relu(
            reinterpret_cast<const uint8_t*>(beam_current.data_ptr<uint8_t>()),
            reinterpret_cast<const uint8_t*>(current_active_flags.data_ptr<uint8_t>()),
            reinterpret_cast<const half*>(embed_w_t.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(embed_bias.data_ptr<at::Half>()),
            act1_ptr, cfg.b_micro, state_size, num_classes, start_state, micro_size, stream);
        CUDA_CHECK(cudaGetLastError());
        launch_fullbeamnice_fill_bias(act2_ptr, reinterpret_cast<const half*>(hidden_bias.data_ptr<at::Half>()), micro_size, 512, stream);
        launch_fullbeamnice_cutlass_gemm(act1_ptr, reinterpret_cast<const half*>(hidden_w_t.data_ptr<at::Half>()), act2_ptr, micro_size, 1536, 512, 1, stream);

        launch_fullbeamnice_fill_bias(act3_ptr, reinterpret_cast<const half*>(res0_fc1_bias.data_ptr<at::Half>()), micro_size, 512, stream);
        launch_fullbeamnice_cutlass_gemm(act2_ptr, reinterpret_cast<const half*>(res0_fc1_w_t.data_ptr<at::Half>()), act3_ptr, micro_size, 512, 512, 1, stream);
        launch_fullbeamnice_fill_residual_bias(act1_ptr, act2_ptr, reinterpret_cast<const half*>(res0_fc2_bias.data_ptr<at::Half>()), micro_size, 512, stream);
        launch_fullbeamnice_cutlass_gemm(act3_ptr, reinterpret_cast<const half*>(res0_fc2_w_t.data_ptr<at::Half>()), act1_ptr, micro_size, 512, 512, 1, stream);

        launch_fullbeamnice_fill_bias(act2_ptr, reinterpret_cast<const half*>(res1_fc1_bias.data_ptr<at::Half>()), micro_size, 512, stream);
        launch_fullbeamnice_cutlass_gemm(act1_ptr, reinterpret_cast<const half*>(res1_fc1_w_t.data_ptr<at::Half>()), act2_ptr, micro_size, 512, 512, 1, stream);
        launch_fullbeamnice_fill_residual_bias(act3_ptr, act1_ptr, reinterpret_cast<const half*>(res1_fc2_bias.data_ptr<at::Half>()), micro_size, 512, stream);
        launch_fullbeamnice_cutlass_gemm(act2_ptr, reinterpret_cast<const half*>(res1_fc2_w_t.data_ptr<at::Half>()), act3_ptr, micro_size, 512, 512, 1, stream);

        launch_fullbeamnice_fill_bias(out_ptr, reinterpret_cast<const half*>(out_bias.data_ptr<at::Half>()), micro_size, FANOUT_FIXED, stream);
        launch_fullbeamnice_cutlass_gemm(act3_ptr, reinterpret_cast<const half*>(out_w_t.data_ptr<at::Half>()), out_ptr, micro_size, 512, FANOUT_FIXED, 0, stream);
        launch_fullbeamnice_quantize_to_ring(
            out_ptr, nullptr, action_perm.data_ptr<int32_t>(),
            reinterpret_cast<uint16_t*>(score_ring.data_ptr<int16_t>()),
            slot, cfg.b_micro, FANOUT_FIXED, micro_size, score_scale, score_bias, stream);
        CUDA_CHECK(cudaGetLastError());
    }
};

class BeamEngine {
public:
    explicit BeamEngine(py::dict cfg_dict, py::dict buffers, std::string backend_name)
        : cfg_(config_from_dict(cfg_dict)) {
        buffers_ = buffers;
        beam_current_ = buffers["beam_current"].cast<torch::Tensor>();
        current_active_flags_ = buffers["current_active_flags"].cast<torch::Tensor>();
        next_state_pool_ = buffers["next_state_pool"].cast<torch::Tensor>();
        next_meta_ = buffers["next_meta"].cast<torch::Tensor>();
        hash_table_ = buffers["hash_table"].cast<torch::Tensor>();
        active_flags_ = buffers["active_flags"].cast<torch::Tensor>();
        free_indices_ = buffers["free_indices"].cast<torch::Tensor>();
        free_count_ = buffers["free_count"].cast<torch::Tensor>();
        score_ring_ = buffers["score_ring"].cast<torch::Tensor>();
        send_buckets_ = buffers["send_buckets"].cast<torch::Tensor>();
        recv_buckets_ = buffers["recv_buckets"].cast<torch::Tensor>();
        send_counts_ = buffers["send_counts"].cast<torch::Tensor>();
        recv_counts_ = buffers["recv_counts"].cast<torch::Tensor>();
        local_hist_ = buffers["local_hist"].cast<torch::Tensor>();
        global_hist_ = buffers["global_hist"].cast<torch::Tensor>();
        threshold_cell_ = buffers["threshold_cell"].cast<torch::Tensor>();
        counters_ = buffers["counters"].cast<torch::Tensor>();
        beam_status_ = buffers["beam_status"].cast<torch::Tensor>();
        history_parent_idx_ = buffers["history_parent_idx"].cast<torch::Tensor>();
        history_parent_rank_ = buffers["history_parent_rank"].cast<torch::Tensor>();
        history_action_ = buffers["history_action"].cast<torch::Tensor>();
        history_valid_ = buffers["history_valid"].cast<torch::Tensor>();
        history_depth_cell_ = buffers["history_depth_cell"].cast<torch::Tensor>();
        check_all_buffers();

        if (backend_name == "dummy" || backend_name == "central_hamming" || backend_name == "torchscript_ensemble" || backend_name == "fullbeamnice_static") inference_ = std::make_unique<DummyInferenceBackend>();
        else if (backend_name == "te") inference_ = std::make_unique<TEInferenceBackend>();
        else throw std::runtime_error("unknown inference backend: " + backend_name);

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_infer_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_ingest_, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_net_, cudaStreamNonBlocking));
        stream_infer_lanes_.resize(cfg_.inference_parallelism);
        for (int i = 0; i < cfg_.inference_parallelism; ++i) CUDA_CHECK(cudaStreamCreateWithFlags(&stream_infer_lanes_[i], cudaStreamNonBlocking));

        score_ready_.resize(cfg_.score_ring_depth);
        score_consumed_.resize(cfg_.score_ring_depth);
        send_ready_.resize(cfg_.net_ring_depth);
        recv_ready_.resize(cfg_.net_ring_depth);
        net_consumed_.resize(cfg_.net_ring_depth);
        for (int i = 0; i < cfg_.score_ring_depth; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&score_ready_[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&score_consumed_[i], cudaEventDisableTiming));
        }
        for (int i = 0; i < cfg_.net_ring_depth; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&send_ready_[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&recv_ready_[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&net_consumed_[i], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&start_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&clear_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&hist_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&threshold_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&compact_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&found_reduce_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreate(&timing_step_start_));
        CUDA_CHECK(cudaEventCreate(&timing_after_clear_));
        CUDA_CHECK(cudaEventCreate(&timing_after_micro_));
        CUDA_CHECK(cudaEventCreate(&timing_after_final_));
        
        // Log bucket configuration
        if (cfg_.rank == 0) {
            const beam_v6::TargetConfig target_cfg = target_config_from_engine(cfg_);
            std::cerr << "[BeamEngine] Config Summary:" << std::endl
                      << "  K_EXPAND_TILE: " << cfg_.k_expand_tile << std::endl
                      << "  SCORE_RING_DEPTH: " << cfg_.score_ring_depth << std::endl
                      << "  BUCKET_CAP_PER_PEER: " << cfg_.bucket_cap_per_peer << std::endl
                      << "  BUCKET_CAP_PER_PEER_SAFE: " << cfg_.bucket_cap_per_peer_safe << std::endl
                      << "  USER_GLOBAL_BEAM_WIDTH: " << target_cfg.user_global_beam_width << std::endl
                      << "  GLOBAL_BEAM_WIDTH_EFFECTIVE: " << target_cfg.global_beam_width_effective << std::endl
                      << "  GLOBAL_BEAM_WIDTH_MAX_SAFE: " << target_cfg.global_beam_width_max_safe << std::endl
                      << "  BEAM_WIDTH_ALIGNMENT: " << target_cfg.beam_width_alignment << std::endl
                      << "  SCORE_SCALE: " << target_cfg.score_scale << std::endl
                      << "  SCORE_MAX_KEY: " << target_cfg.score_max_key << std::endl
                      << "  SCORE_BIN_COUNT: " << target_cfg.score_bin_count << std::endl
                      << "  send_buckets: " << cfg_.send_buckets_gib << " GiB" << std::endl
                      << "  recv_buckets: " << cfg_.recv_buckets_gib << " GiB" << std::endl
                      << "  total_buckets: " << cfg_.total_bucket_gib << " GiB" << std::endl;
        }
    }

    ~BeamEngine() {
        if (nccl_inited_) ncclCommDestroy(comm_);
        if (cuda_graph_exec_) cudaGraphExecDestroy(cuda_graph_exec_);
        if (cuda_graph_) cudaGraphDestroy(cuda_graph_);
        for (auto e : score_ready_) cudaEventDestroy(e);
        for (auto e : score_consumed_) cudaEventDestroy(e);
        for (auto e : send_ready_) cudaEventDestroy(e);
        for (auto e : recv_ready_) cudaEventDestroy(e);
        for (auto e : net_consumed_) cudaEventDestroy(e);
        cudaEventDestroy(start_ready_);
        cudaEventDestroy(clear_ready_);
        cudaEventDestroy(hist_ready_);
        cudaEventDestroy(threshold_ready_);
        cudaEventDestroy(compact_ready_);
        cudaEventDestroy(found_reduce_ready_);
        cudaEventDestroy(timing_step_start_);
        cudaEventDestroy(timing_after_clear_);
        cudaEventDestroy(timing_after_micro_);
        cudaEventDestroy(timing_after_final_);
        for (auto st : stream_infer_lanes_) cudaStreamDestroy(st);
        cudaStreamDestroy(stream_infer_);
        cudaStreamDestroy(stream_ingest_);
        cudaStreamDestroy(stream_net_);
    }

    void init_nccl(py::bytes unique_id_bytes) {
        if (cfg_.world_size <= 1) return;
        ncclUniqueId id = nccl_unique_id_from_bytes(unique_id_bytes);
        NCCL_CHECK(ncclCommInitRank(&comm_, cfg_.world_size, id, cfg_.rank));
        nccl_inited_ = true;
    }

    void v6_stream5_exchange_candidate_meta(
        torch::Tensor remote_send_buffer,
        torch::Tensor remote_recv_buffer,
        torch::Tensor send_count,
        torch::Tensor send_offset,
        torch::Tensor recv_count,
        torch::Tensor recv_offset) {
        check_cuda_tensor(remote_send_buffer, "remote_send_buffer");
        check_cuda_tensor(remote_recv_buffer, "remote_recv_buffer");
        check_cuda_i32_tensor(send_count, "send_count");
        check_cuda_i32_tensor(send_offset, "send_offset");
        check_cuda_i32_tensor(recv_count, "recv_count");
        check_cuda_i32_tensor(recv_offset, "recv_offset");

        cudaStream_t stream = stream_net_;
        beam_v6::CandidateMeta* send_base = reinterpret_cast<beam_v6::CandidateMeta*>(remote_send_buffer.data_ptr<uint8_t>());
        beam_v6::CandidateMeta* recv_base = reinterpret_cast<beam_v6::CandidateMeta*>(remote_recv_buffer.data_ptr<uint8_t>());
        int32_t* send_count_base = send_count.data_ptr<int32_t>();
        int32_t* recv_count_base = recv_count.data_ptr<int32_t>();
        int32_t* send_offset_base = send_offset.data_ptr<int32_t>();
        int32_t* recv_offset_base = recv_offset.data_ptr<int32_t>();

        if (cfg_.world_size <= 1) {
            CUDA_CHECK(cudaMemcpyAsync(recv_count_base, send_count_base, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(recv_offset_base, send_offset_base, 2 * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
            int32_t host_count = 0;
            int32_t host_send_offset = 0;
            int32_t host_recv_offset = 0;
            CUDA_CHECK(cudaMemcpyAsync(&host_count, send_count_base, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&host_send_offset, send_offset_base, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(&host_recv_offset, recv_offset_base, sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (host_count > 0) {
                CUDA_CHECK(cudaMemcpyAsync(
                    recv_base + host_recv_offset,
                    send_base + host_send_offset,
                    static_cast<size_t>(host_count) * sizeof(beam_v6::CandidateMeta),
                    cudaMemcpyDeviceToDevice,
                    stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return;
        }

        if (!nccl_inited_) {
            throw std::runtime_error("v6 Stream5 exchange requires initialized NCCL for WORLD_SIZE > 1");
        }

        NCCL_CHECK(ncclGroupStart());
        for (int peer = 0; peer < cfg_.world_size; ++peer) {
            if (peer == cfg_.rank) continue;
            NCCL_CHECK(ncclSend(send_count_base + peer, 1, ncclInt32, peer, comm_, stream));
            NCCL_CHECK(ncclRecv(recv_count_base + peer, 1, ncclInt32, peer, comm_, stream));
        }
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        auto send_count_cpu = send_count.cpu();
        auto recv_count_cpu = recv_count.cpu();
        auto send_offset_cpu = send_offset.cpu();
        auto recv_offset_cpu = recv_offset.cpu();
        const int32_t* sc = send_count_cpu.data_ptr<int32_t>();
        const int32_t* rc = recv_count_cpu.data_ptr<int32_t>();
        const int32_t* so = send_offset_cpu.data_ptr<int32_t>();
        const int32_t* ro = recv_offset_cpu.data_ptr<int32_t>();

        NCCL_CHECK(ncclGroupStart());
        for (int peer = 0; peer < cfg_.world_size; ++peer) {
            if (peer == cfg_.rank) continue;
            if (sc[peer] > 0) {
                NCCL_CHECK(ncclSend(
                    send_base + so[peer],
                    static_cast<size_t>(sc[peer]) * sizeof(beam_v6::CandidateMeta),
                    ncclUint8,
                    peer,
                    comm_,
                    stream));
            }
            if (rc[peer] > 0) {
                NCCL_CHECK(ncclRecv(
                    recv_base + ro[peer],
                    static_cast<size_t>(rc[peer]) * sizeof(beam_v6::CandidateMeta),
                    ncclUint8,
                    peer,
                    comm_,
                    stream));
            }
        }
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void load_torchscript_ensemble(const std::vector<std::string>& module_paths) {
        invalidate_graph();
        inference_ = std::make_unique<TorchScriptEnsembleBackend>(module_paths, beam_current_.device());
        inference_warmed_ = false;
    }

    void load_fullbeamnice_static(py::dict weights) {
        invalidate_graph();
        inference_ = std::make_unique<FullBeamNiceStaticBackend>(weights, buffers_);
        inference_warmed_ = false;
    }

    void begin_uniform_score(uint16_t score_q) {
        invalidate_graph();
        if (saved_inference_) throw std::runtime_error("uniform score phase already active");
        saved_inference_ = std::move(inference_);
        saved_use_cuda_graphs_ = use_cuda_graphs_;
        use_cuda_graphs_ = false;
        inference_ = std::make_unique<UniformScoreBackend>(score_q);
        inference_warmed_ = false;
    }

    void set_uniform_score(uint16_t score_q) {
        auto* uniform = dynamic_cast<UniformScoreBackend*>(inference_.get());
        if (!uniform) throw std::runtime_error("uniform score phase is not active");
        uniform->score_q = score_q;
    }

    void end_uniform_score() {
        invalidate_graph();
        if (!saved_inference_) throw std::runtime_error("uniform score phase is not active");
        inference_ = std::move(saved_inference_);
        use_cuda_graphs_ = saved_use_cuda_graphs_;
        inference_warmed_ = false;
    }

    void warmup_inference(int repeats = 2) {
        if (repeats < 1) repeats = 1;
        const int micro_size = static_cast<int>(std::min<int64_t>(cfg_.b_micro, cfg_.n_local));
        if (micro_size <= 0) return;
        for (int r = 0; r < repeats; ++r) {
            for (int lane = 0; lane < cfg_.inference_parallelism; ++lane) {
                int slot = lane % cfg_.score_ring_depth;
                inference_->forward(beam_current_, current_active_flags_, score_ring_, slot, lane, 0, micro_size, cfg_, stream_infer_lanes_[lane]);
            }
        }
        for (auto st : stream_infer_lanes_) CUDA_CHECK(cudaStreamSynchronize(st));
        inference_warmed_ = true;
    }

    py::list benchmark_inference(int micro_size, int repeats, int warmup) {
        if (micro_size < 1) micro_size = 1;
        if (micro_size > cfg_.b_micro) micro_size = cfg_.b_micro;
        if (micro_size > cfg_.n_local) micro_size = static_cast<int>(cfg_.n_local);
        if (repeats < 1) repeats = 1;
        if (warmup < 0) warmup = 0;
        const int lanes = std::max(1, cfg_.inference_parallelism);
        std::vector<cudaEvent_t> lane_done(lanes, nullptr);
        for (int lane = 0; lane < lanes; ++lane) CUDA_CHECK(cudaEventCreateWithFlags(&lane_done[lane], cudaEventDisableTiming));
        cudaEvent_t bench_start = nullptr;
        cudaEvent_t bench_stop = nullptr;
        CUDA_CHECK(cudaEventCreate(&bench_start));
        CUDA_CHECK(cudaEventCreate(&bench_stop));

        auto run_once = [&](bool timed) -> float {
            CUDA_CHECK(cudaEventRecord(bench_start, stream_infer_));
            for (int lane = 0; lane < lanes; ++lane) {
                cudaStream_t st = stream_infer_lanes_[lane];
                CUDA_CHECK(cudaStreamWaitEvent(st, bench_start, 0));
                const int slot = lane % cfg_.score_ring_depth;
                inference_->forward(beam_current_, current_active_flags_, score_ring_, slot, lane, 0, micro_size, cfg_, st);
                CUDA_CHECK(cudaEventRecord(lane_done[lane], st));
            }
            for (int lane = 0; lane < lanes; ++lane) CUDA_CHECK(cudaStreamWaitEvent(stream_infer_, lane_done[lane], 0));
            CUDA_CHECK(cudaEventRecord(bench_stop, stream_infer_));
            CUDA_CHECK(cudaEventSynchronize(bench_stop));
            if (!timed) return 0.0f;
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, bench_start, bench_stop));
            return ms;
        };

        try {
            for (int i = 0; i < warmup; ++i) (void)run_once(false);
            py::list timings;
            for (int i = 0; i < repeats; ++i) timings.append(run_once(true));
            CUDA_CHECK(cudaEventDestroy(bench_start));
            CUDA_CHECK(cudaEventDestroy(bench_stop));
            for (auto e : lane_done) CUDA_CHECK(cudaEventDestroy(e));
            return timings;
        } catch (...) {
            if (bench_start) cudaEventDestroy(bench_start);
            if (bench_stop) cudaEventDestroy(bench_stop);
            for (auto e : lane_done) if (e) cudaEventDestroy(e);
            throw;
        }
    }

    void set_action_permutation_table(py::bytes table_bytes) {
        std::string table = table_bytes;
        const size_t expected = static_cast<size_t>(cfg_.fanout) * static_cast<size_t>(cfg_.state_size_bytes);
        if (table.size() != expected) throw std::runtime_error("action table must be fanout*state_size bytes");
        upload_action_permutation_table(reinterpret_cast<const uint8_t*>(table.data()), cfg_.fanout, cfg_.state_size_bytes);
        CUDA_CHECK(cudaGetLastError());
    }

    void set_central_state(py::bytes state_bytes) {
        std::string state = state_bytes;
        if (state.size() != static_cast<size_t>(cfg_.state_size_bytes)) throw std::runtime_error("central state must be exactly 120 bytes");
        upload_central_state(reinterpret_cast<const uint8_t*>(state.data()), cfg_.state_size_bytes);
        CUDA_CHECK(cudaGetLastError());
        central_loaded_ = true;
    }

    void reset_search(py::bytes initial_state_bytes, bool active_on_this_rank) {
        std::string state = initial_state_bytes;
        if (state.size() != static_cast<size_t>(cfg_.state_size_bytes)) throw std::runtime_error("initial state must be exactly 120 bytes");
        CUDA_CHECK(cudaMemsetAsync(beam_current_.data_ptr<uint8_t>(), 0, beam_current_.numel(), stream_ingest_));
        CUDA_CHECK(cudaMemsetAsync(current_active_flags_.data_ptr<uint8_t>(), 0, current_active_flags_.numel(), stream_ingest_));
        CUDA_CHECK(cudaMemsetAsync(beam_status_.data_ptr<int32_t>(), 0, beam_status_.numel() * sizeof(int32_t), stream_ingest_));
        CUDA_CHECK(cudaMemsetAsync(history_valid_.data_ptr<uint8_t>(), 0, history_valid_.numel(), stream_ingest_));
        CUDA_CHECK(cudaMemsetAsync(history_depth_cell_.data_ptr<int32_t>(), 0, sizeof(int32_t), stream_ingest_));
        current_history_depth_ = 0;
        if (active_on_this_rank) {
            CUDA_CHECK(cudaMemcpyAsync(beam_current_.data_ptr<uint8_t>(), state.data(), cfg_.state_size_bytes, cudaMemcpyHostToDevice, stream_ingest_));
            uint8_t one = 1;
            int32_t sz = 1;
            CUDA_CHECK(cudaMemcpyAsync(current_active_flags_.data_ptr<uint8_t>(), &one, sizeof(uint8_t), cudaMemcpyHostToDevice, stream_ingest_));
            CUDA_CHECK(cudaMemcpyAsync(beam_status_.data_ptr<int32_t>() + STATUS_CURRENT_SIZE, &sz, sizeof(int32_t), cudaMemcpyHostToDevice, stream_ingest_));
        }
        if (central_loaded_) {
            launch_check_current_solved(
                reinterpret_cast<const uint8_t*>(beam_current_.data_ptr<uint8_t>()),
                current_active_flags_.data_ptr<uint8_t>(),
                beam_status_.data_ptr<int32_t>(),
                cfg_.state_size_bytes,
                1,
                stream_ingest_);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_ingest_));
    }

    void clear_runtime_state() {
        clear_step_state_async(stream_ingest_, cfg_.k_work, cfg_.hash_capacity);
        CUDA_CHECK(cudaStreamSynchronize(stream_ingest_));
    }

    void enable_cuda_graphs(bool enable) {
        use_cuda_graphs_ = enable;
        if (!enable) invalidate_graph();
    }

    bool cuda_graph_captured() const { return cuda_graph_captured_; }

    void enable_step_timers(bool enable) {
        step_timers_enabled_ = enable;
        if (enable) invalidate_graph();
    }

    py::dict step_timing() const {
        py::dict d;
        if (!step_timers_enabled_ || !step_timing_valid_) {
            d["enabled"] = step_timers_enabled_;
            d["valid"] = false;
            return d;
        }
        float clear_ms = 0.0f;
        float micro_pipeline_ms = 0.0f;
        float final_prune_compact_ms = 0.0f;
        float total_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&clear_ms, timing_step_start_, timing_after_clear_));
        CUDA_CHECK(cudaEventElapsedTime(&micro_pipeline_ms, timing_after_clear_, timing_after_micro_));
        CUDA_CHECK(cudaEventElapsedTime(&final_prune_compact_ms, timing_after_micro_, timing_after_final_));
        CUDA_CHECK(cudaEventElapsedTime(&total_ms, timing_step_start_, timing_after_final_));
        d["enabled"] = true;
        d["valid"] = true;
        d["clear_and_solved_scan_ms"] = clear_ms;
        d["micro_pipeline_ms"] = micro_pipeline_ms;
        d["final_prune_compact_found_ms"] = final_prune_compact_ms;
        d["total_cuda_event_ms"] = total_ms;
        return d;
    }

    void set_active_limit(uint64_t active_limit) {
        int64_t limit = static_cast<int64_t>(active_limit);
        if (limit < 1) limit = 1;
        if (limit > cfg_.n_local) limit = cfg_.n_local;
        if (logical_active_limit_ != limit) invalidate_graph();
        logical_active_limit_ = limit;
    }

    void set_next_limit(uint64_t next_limit) {
        int64_t limit = static_cast<int64_t>(next_limit);
        if (limit < 1) limit = 1;
        if (limit > cfg_.k_work) limit = cfg_.k_work;
        const double load_factor = cfg_.hash_load_factor > 0.01f ? static_cast<double>(cfg_.hash_load_factor) : 0.45;
        int64_t hash_limit = static_cast<int64_t>(static_cast<double>(limit) / load_factor) + 1024;
        if (hash_limit < 1024) hash_limit = 1024;
        if (hash_limit > cfg_.hash_capacity) hash_limit = cfg_.hash_capacity;
        if (logical_next_limit_ != limit || logical_hash_capacity_ != hash_limit) invalidate_graph();
        logical_next_limit_ = limit;
        logical_hash_capacity_ = hash_limit;
    }

    void clear_logical_limits() {
        if (logical_active_limit_ != -1 || logical_next_limit_ != -1 || logical_hash_capacity_ != -1) invalidate_graph();
        logical_active_limit_ = -1;
        logical_next_limit_ = -1;
        logical_hash_capacity_ = -1;
    }

    void enable_debug(bool verbose, bool print_counters, int log_period) {
#if BEAM_DEBUG_ON
        debug_config.verbose = verbose;
        debug_config.print_counters = print_counters;
        debug_config.log_period = log_period;
#else
        (void)verbose;
        (void)print_counters;
        (void)log_period;
#endif
    }

    void step(int histogram_period_micro) {
        if (cfg_.world_size > 1 && !nccl_inited_) throw std::runtime_error("NCCL is not initialized");
        if (histogram_period_micro <= 0) histogram_period_micro = 8;
        if (use_cuda_graphs_) {
            if (!cuda_graph_captured_) capture_cuda_graph(histogram_period_micro);
            CUDA_CHECK(cudaGraphLaunch(cuda_graph_exec_, stream_infer_));
            CUDA_CHECK(cudaStreamSynchronize(stream_infer_));
        } else {
            enqueue_one_depth(histogram_period_micro);
            CUDA_CHECK(cudaStreamSynchronize(stream_infer_));
        }
        if (debug_config.verbose && debug_config.print_counters) print_debug_status();
    }

    void step_current(int histogram_period_micro) {
        if (cfg_.world_size > 1 && !nccl_inited_) throw std::runtime_error("NCCL is not initialized");
        if (histogram_period_micro <= 0) histogram_period_micro = 8;
        auto status_cpu = beam_status_.cpu();
        int64_t current_size = static_cast<int64_t>(status_cpu.data_ptr<int32_t>()[STATUS_CURRENT_SIZE]);
        if (current_size < 1) current_size = 1;
        if (current_size > cfg_.n_local) current_size = cfg_.n_local;
        const int64_t old_override = active_limit_override_;
        const int64_t old_logical = logical_active_limit_;
        active_limit_override_ = current_size;
        logical_active_limit_ = current_size;
        enqueue_one_depth(histogram_period_micro);
        CUDA_CHECK(cudaStreamSynchronize(stream_infer_));
        active_limit_override_ = old_override;
        logical_active_limit_ = old_logical;
        if (debug_config.verbose && debug_config.print_counters) print_debug_status();
    }

    void step_prepass_fast(uint16_t score_q) {
        begin_uniform_score(score_q);
        step_current(1);
        end_uniform_score();
    }

    void set_prepass_light_solved_scan(bool enable) {
        invalidate_graph();
        prepass_light_solved_scan_ = enable;
    }

    py::dict search(int max_depth, int histogram_period_micro) {
        py::dict result;
        if (max_depth < 0) max_depth = 0;
        for (int depth = 0; depth <= max_depth; ++depth) {
            py::dict st = status();
            if (st["found"].cast<int>() != 0) {
                result["found"] = true;
                result["depth"] = depth;
                result["status"] = st;
                return result;
            }
            if (depth == max_depth) break;
            if (cfg_.world_size <= 1 && st["current_size"].cast<int>() == 0) {
                result["found"] = false;
                result["depth"] = depth;
                result["status"] = st;
                return result;
            }
            step(histogram_period_micro);
        }
        result["found"] = false;
        result["depth"] = max_depth;
        result["status"] = status();
        return result;
    }

    py::dict status() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_infer_));
        for (auto st : stream_infer_lanes_) CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaStreamSynchronize(stream_ingest_));
        CUDA_CHECK(cudaStreamSynchronize(stream_net_));
        auto status_cpu = beam_status_.cpu();
        auto counters_cpu = counters_.cpu();
        auto threshold_cpu = threshold_cell_.cpu();
        const int32_t* s = status_cpu.data_ptr<int32_t>();
        const int32_t* c = counters_cpu.data_ptr<int32_t>();
        const int32_t* t = threshold_cpu.data_ptr<int32_t>();
        py::dict d;
        d["current_size"] = s[STATUS_CURRENT_SIZE];
        d["compacted_size"] = s[STATUS_COMPACTED_SIZE];
        d["found"] = s[STATUS_FOUND];
        d["local_found"] = s[STATUS_LOCAL_FOUND];
        d["found_local_index"] = s[STATUS_FOUND_LOCAL_INDEX];
        d["found_action"] = s[STATUS_FOUND_ACTION];
        d["cuda_graph_captured"] = cuda_graph_captured_ ? 1 : 0;
        d["threshold_valid"] = t[0];
        d["threshold_q"] = t[1];
        py::list counters;
        for (int i = 0; i < COUNTER_RESERVED; ++i) counters.append(c[i]);
        d["counters"] = counters;
        return d;
    }

    py::dict history_entry(int depth, int local_index) const {
        if (depth < 0 || depth >= cfg_.max_depth) throw std::runtime_error("history depth out of range");
        if (local_index < 0 || local_index >= cfg_.n_local) throw std::runtime_error("history local index out of range");
        CUDA_CHECK(cudaStreamSynchronize(stream_infer_));
#if BEAM_HISTORY_CPU
        int64_t pos = local_index;
#else
        int64_t pos = static_cast<int64_t>(depth) * cfg_.n_local + local_index;
#endif
        auto parent_cpu = history_parent_idx_.slice(0, pos, pos + 1).cpu();
        auto rank_cpu = history_parent_rank_.slice(0, pos, pos + 1).cpu();
        auto action_cpu = history_action_.slice(0, pos, pos + 1).cpu();
        auto valid_cpu = history_valid_.slice(0, pos, pos + 1).cpu();
        py::dict d;
        d["valid"] = static_cast<int>(valid_cpu.data_ptr<uint8_t>()[0]);
        d["parent_idx"] = parent_cpu.data_ptr<int32_t>()[0];
        d["parent_rank"] = static_cast<int>(rank_cpu.data_ptr<uint8_t>()[0]);
        d["action"] = static_cast<int>(action_cpu.data_ptr<uint8_t>()[0]);
        return d;
    }

    py::bytes current_state_bytes(int local_index) const {
        if (local_index < 0 || local_index >= cfg_.n_local) throw std::runtime_error("current state local index out of range");
        CUDA_CHECK(cudaStreamSynchronize(stream_infer_));
        for (auto st : stream_infer_lanes_) CUDA_CHECK(cudaStreamSynchronize(st));
        CUDA_CHECK(cudaStreamSynchronize(stream_ingest_));
        CUDA_CHECK(cudaStreamSynchronize(stream_net_));
        auto state_cpu = beam_current_.slice(0, local_index, local_index + 1).contiguous().cpu();
        const char* ptr = reinterpret_cast<const char*>(state_cpu.data_ptr<uint8_t>());
        return py::bytes(ptr, static_cast<size_t>(cfg_.state_size_bytes));
    }

    py::dict sizes() const {
        py::dict d;
        d["n_local"] = cfg_.n_local;
        d["k_keep"] = cfg_.k_keep;
        d["k_work"] = cfg_.k_work;
        d["hash_capacity"] = cfg_.hash_capacity;
        d["candidate_record_bytes"] = static_cast<int>(sizeof(CandidateRecord));
        d["beam_meta_bytes"] = static_cast<int>(sizeof(BeamMeta));
        d["hash_slot_bytes"] = static_cast<int>(sizeof(HashSlot));
        d["bucket_cap_per_peer"] = cfg_.bucket_cap_per_peer;
        d["inference_parallelism"] = cfg_.inference_parallelism;
        d["k_expand_tile"] = cfg_.k_expand_tile;
        return d;
    }

private:
    EngineConfig cfg_;
    std::unique_ptr<InferenceBackend> inference_;
    std::unique_ptr<InferenceBackend> saved_inference_;

    torch::Tensor beam_current_;
    torch::Tensor current_active_flags_;
    torch::Tensor next_state_pool_;
    torch::Tensor next_meta_;
    torch::Tensor hash_table_;
    torch::Tensor active_flags_;
    torch::Tensor free_indices_;
    torch::Tensor free_count_;
    torch::Tensor score_ring_;
    torch::Tensor send_buckets_;
    torch::Tensor recv_buckets_;
    torch::Tensor send_counts_;
    torch::Tensor recv_counts_;
    torch::Tensor local_hist_;
    torch::Tensor global_hist_;
    torch::Tensor threshold_cell_;
    torch::Tensor counters_;
    torch::Tensor beam_status_;
    torch::Tensor history_parent_idx_;
    torch::Tensor history_parent_rank_;
    torch::Tensor history_action_;
    torch::Tensor history_valid_;
    torch::Tensor history_depth_cell_;
    py::dict buffers_;

    cudaStream_t stream_infer_ = nullptr;
    cudaStream_t stream_ingest_ = nullptr;
    cudaStream_t stream_net_ = nullptr;
    std::vector<cudaStream_t> stream_infer_lanes_;
    std::vector<cudaEvent_t> score_ready_;
    std::vector<cudaEvent_t> score_consumed_;
    std::vector<cudaEvent_t> send_ready_;
    std::vector<cudaEvent_t> recv_ready_;
    std::vector<cudaEvent_t> net_consumed_;
    cudaEvent_t start_ready_ = nullptr;
    cudaEvent_t clear_ready_ = nullptr;
    cudaEvent_t hist_ready_ = nullptr;
    cudaEvent_t threshold_ready_ = nullptr;
    cudaEvent_t compact_ready_ = nullptr;
    cudaEvent_t found_reduce_ready_ = nullptr;
    cudaEvent_t timing_step_start_ = nullptr;
    cudaEvent_t timing_after_clear_ = nullptr;
    cudaEvent_t timing_after_micro_ = nullptr;
    cudaEvent_t timing_after_final_ = nullptr;

    ncclComm_t comm_{};
    bool nccl_inited_ = false;
    int current_history_depth_ = 0;
    bool use_cuda_graphs_ = true;
    bool saved_use_cuda_graphs_ = true;
    int64_t active_limit_override_ = -1;
    int64_t logical_active_limit_ = -1;
    int64_t logical_next_limit_ = -1;
    int64_t logical_hash_capacity_ = -1;
    bool prepass_light_solved_scan_ = false;
    bool step_timers_enabled_ = false;
    bool step_timing_valid_ = false;
    bool cuda_graph_captured_ = false;
    bool central_loaded_ = false;
    bool inference_warmed_ = false;
    cudaGraph_t cuda_graph_ = nullptr;
    cudaGraphExec_t cuda_graph_exec_ = nullptr;

    void check_all_buffers() {
        check_cuda_tensor(beam_current_, "beam_current");
        check_cuda_tensor(current_active_flags_, "current_active_flags");
        check_cuda_tensor(next_state_pool_, "next_state_pool");
        check_cuda_tensor(next_meta_, "next_meta");
        check_cuda_tensor(hash_table_, "hash_table");
        check_cuda_tensor(active_flags_, "active_flags");
        check_cuda_tensor(free_indices_, "free_indices");
        check_cuda_tensor(free_count_, "free_count");
        check_cuda_tensor(score_ring_, "score_ring");
        check_cuda_tensor(send_buckets_, "send_buckets");
        check_cuda_tensor(recv_buckets_, "recv_buckets");
        check_cuda_tensor(send_counts_, "send_counts");
        check_cuda_tensor(recv_counts_, "recv_counts");
        check_cuda_tensor(local_hist_, "local_hist");
        check_cuda_tensor(global_hist_, "global_hist");
        check_cuda_tensor(threshold_cell_, "threshold_cell");
        check_cuda_tensor(counters_, "counters");
        check_cuda_tensor(beam_status_, "beam_status");
        check_cuda_tensor(history_parent_idx_, "history_parent_idx");
        check_cuda_tensor(history_parent_rank_, "history_parent_rank");
        check_cuda_tensor(history_action_, "history_action");
        check_cuda_tensor(history_valid_, "history_valid");
        check_cuda_tensor(history_depth_cell_, "history_depth_cell");
    }

    void invalidate_graph() {
        if (cuda_graph_exec_) {
            cudaGraphExecDestroy(cuda_graph_exec_);
            cuda_graph_exec_ = nullptr;
        }
        if (cuda_graph_) {
            cudaGraphDestroy(cuda_graph_);
            cuda_graph_ = nullptr;
        }
        cuda_graph_captured_ = false;
    }

    void clear_step_state_async(cudaStream_t stream, int64_t next_limit, int64_t hash_limit) {
        launch_clear_hash_table(reinterpret_cast<HashSlot*>(hash_table_.data_ptr<uint8_t>()), static_cast<int>(hash_limit), stream);
        launch_clear_step_state(
            counters_.data_ptr<int32_t>(),
            reinterpret_cast<uint32_t*>(local_hist_.data_ptr<int32_t>()),
            threshold_cell_.data_ptr<int32_t>(),
            active_flags_.data_ptr<uint8_t>(),
            free_count_.data_ptr<int32_t>(),
            beam_status_.data_ptr<int32_t>(),
            SCORE_BINS,
            static_cast<int>(next_limit),
            stream);
        CUDA_CHECK(cudaGetLastError());
    }

    void do_fixed_all_to_all(int net_slot) {
        if (cfg_.world_size <= 1) {
            int32_t* send_counts_base = send_counts_.data_ptr<int32_t>() + static_cast<int64_t>(net_slot) * cfg_.world_size;
            int32_t* recv_counts_base = recv_counts_.data_ptr<int32_t>() + static_cast<int64_t>(net_slot) * cfg_.world_size;
            if (recv_counts_base != send_counts_base) {
                CUDA_CHECK(cudaMemcpyAsync(recv_counts_base, send_counts_base, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream_net_));
            }
            return;
        }
        CandidateRecord* send_base = reinterpret_cast<CandidateRecord*>(send_buckets_.data_ptr<uint8_t>());
        CandidateRecord* recv_base = reinterpret_cast<CandidateRecord*>(recv_buckets_.data_ptr<uint8_t>());
        int32_t* send_counts_base = send_counts_.data_ptr<int32_t>() + static_cast<int64_t>(net_slot) * cfg_.world_size;
        int32_t* recv_counts_base = recv_counts_.data_ptr<int32_t>() + static_cast<int64_t>(net_slot) * cfg_.world_size;
        const int64_t one_peer_records = cfg_.bucket_cap_per_peer;
        const int64_t one_peer_bytes = one_peer_records * static_cast<int64_t>(sizeof(CandidateRecord));
        const int64_t slot_offset_records = static_cast<int64_t>(net_slot) * cfg_.world_size * one_peer_records;

        NCCL_CHECK(ncclGroupStart());
        for (int peer = 0; peer < cfg_.world_size; ++peer) {
            NCCL_CHECK(ncclSend(send_counts_base + peer, 1, ncclInt32, peer, comm_, stream_net_));
            NCCL_CHECK(ncclRecv(recv_counts_base + peer, 1, ncclInt32, peer, comm_, stream_net_));
        }
        NCCL_CHECK(ncclGroupEnd());

        NCCL_CHECK(ncclGroupStart());
        for (int peer = 0; peer < cfg_.world_size; ++peer) {
            CandidateRecord* send_ptr = send_base + slot_offset_records + static_cast<int64_t>(peer) * one_peer_records;
            CandidateRecord* recv_ptr = recv_base + slot_offset_records + static_cast<int64_t>(peer) * one_peer_records;
            NCCL_CHECK(ncclSend(send_ptr, one_peer_bytes, ncclUint8, peer, comm_, stream_net_));
            NCCL_CHECK(ncclRecv(recv_ptr, one_peer_bytes, ncclUint8, peer, comm_, stream_net_));
        }
        NCCL_CHECK(ncclGroupEnd());
    }

    void enqueue_threshold_update(int64_t next_limit, int64_t hash_limit) {
        CUDA_CHECK(cudaEventRecord(hist_ready_, stream_ingest_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_net_, hist_ready_, 0));
        if (cfg_.world_size > 1) {
            NCCL_CHECK(ncclAllReduce(
                local_hist_.data_ptr<int32_t>(),
                global_hist_.data_ptr<int32_t>(),
                SCORE_BINS,
                ncclInt32,
                ncclSum,
                comm_,
                stream_net_));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(global_hist_.data_ptr<int32_t>(), local_hist_.data_ptr<int32_t>(), SCORE_BINS * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream_net_));
        }
        launch_compute_threshold(reinterpret_cast<const uint32_t*>(global_hist_.data_ptr<int32_t>()), threshold_cell_.data_ptr<int32_t>(), SCORE_BINS, cfg_.global_beam_width, stream_net_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(threshold_ready_, stream_net_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_ingest_, threshold_ready_, 0));
        launch_prune_by_threshold(
            reinterpret_cast<BeamMeta*>(next_meta_.data_ptr<uint8_t>()),
            reinterpret_cast<HashSlot*>(hash_table_.data_ptr<uint8_t>()),
            active_flags_.data_ptr<uint8_t>(),
            free_indices_.data_ptr<int32_t>(),
            free_count_.data_ptr<int32_t>(),
            reinterpret_cast<uint32_t*>(local_hist_.data_ptr<int32_t>()),
            counters_.data_ptr<int32_t>(),
            threshold_cell_.data_ptr<int32_t>(),
            static_cast<int>(next_limit),
            static_cast<int>(hash_limit),
            cfg_.probe_limit,
            stream_ingest_);
        CUDA_CHECK(cudaGetLastError());
        launch_clear_hash_table(reinterpret_cast<HashSlot*>(hash_table_.data_ptr<uint8_t>()), static_cast<int>(hash_limit), stream_ingest_);
        launch_rebuild_hash_from_active(
            reinterpret_cast<BeamMeta*>(next_meta_.data_ptr<uint8_t>()),
            reinterpret_cast<HashSlot*>(hash_table_.data_ptr<uint8_t>()),
            active_flags_.data_ptr<uint8_t>(),
            counters_.data_ptr<int32_t>(),
            static_cast<int>(next_limit),
            static_cast<int>(hash_limit),
            cfg_.probe_limit,
            stream_ingest_);
        CUDA_CHECK(cudaGetLastError());
    }

    void enqueue_found_allreduce_and_finish() {
        CUDA_CHECK(cudaEventRecord(compact_ready_, stream_ingest_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_net_, compact_ready_, 0));
        if (cfg_.world_size > 1) {
            NCCL_CHECK(ncclAllReduce(
                beam_status_.data_ptr<int32_t>() + STATUS_FOUND,
                beam_status_.data_ptr<int32_t>() + STATUS_FOUND,
                1,
                ncclInt32,
                ncclMax,
                comm_,
                stream_net_));
        }
        CUDA_CHECK(cudaEventRecord(found_reduce_ready_, stream_net_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_infer_, found_reduce_ready_, 0));
    }

    void enqueue_one_depth(int histogram_period_micro) {
        int64_t active_limit = cfg_.n_local;
        if (logical_active_limit_ > 0 && logical_active_limit_ < active_limit) active_limit = logical_active_limit_;
        if (active_limit_override_ > 0 && active_limit_override_ < active_limit) active_limit = active_limit_override_;
        int64_t next_limit = (logical_next_limit_ > 0 && logical_next_limit_ < cfg_.k_work) ? logical_next_limit_ : cfg_.k_work;
        int64_t hash_limit = (logical_hash_capacity_ > 0 && logical_hash_capacity_ < cfg_.hash_capacity) ? logical_hash_capacity_ : cfg_.hash_capacity;
        if (next_limit < 1) next_limit = 1;
        if (hash_limit < 1) hash_limit = 1;
        const int64_t current_output_cap = std::min<int64_t>(cfg_.n_local, next_limit);
        const int64_t num_micro = (active_limit + cfg_.b_micro - 1) / cfg_.b_micro;
        CUDA_CHECK(cudaEventRecord(start_ready_, stream_infer_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_ingest_, start_ready_, 0));
        for (auto st : stream_infer_lanes_) CUDA_CHECK(cudaStreamWaitEvent(st, start_ready_, 0));
        if (step_timers_enabled_) {
            step_timing_valid_ = false;
            CUDA_CHECK(cudaEventRecord(timing_step_start_, stream_ingest_));
        }

        clear_step_state_async(stream_ingest_, next_limit, hash_limit);
        if (central_loaded_) {
            int solved_scan_n = static_cast<int>(active_limit);
            if (prepass_light_solved_scan_) {
                int32_t cs = 1;
                CUDA_CHECK(cudaMemcpyAsync(
                    &cs,
                    beam_status_.data_ptr<int32_t>() + STATUS_CURRENT_SIZE,
                    sizeof(int32_t),
                    cudaMemcpyDeviceToHost,
                    stream_ingest_));
                CUDA_CHECK(cudaStreamSynchronize(stream_ingest_));
                if (cs < 1) cs = 1;
                const int64_t cap = std::min<int64_t>(active_limit, cs);
                solved_scan_n = static_cast<int>(cap);
            }
            launch_check_current_solved(
                reinterpret_cast<const uint8_t*>(beam_current_.data_ptr<uint8_t>()),
                current_active_flags_.data_ptr<uint8_t>(),
                beam_status_.data_ptr<int32_t>(),
                cfg_.state_size_bytes,
                solved_scan_n,
                stream_ingest_);
            CUDA_CHECK(cudaGetLastError());
        }
        if (step_timers_enabled_) CUDA_CHECK(cudaEventRecord(timing_after_clear_, stream_ingest_));
        CUDA_CHECK(cudaEventRecord(clear_ready_, stream_ingest_));
        CUDA_CHECK(cudaStreamWaitEvent(stream_infer_, clear_ready_, 0));
        for (auto st : stream_infer_lanes_) CUDA_CHECK(cudaStreamWaitEvent(st, clear_ready_, 0));

        for (int64_t mb = 0; mb < num_micro; ++mb) {
            const int score_slot = static_cast<int>(mb % cfg_.score_ring_depth);
            const int net_slot = static_cast<int>(mb % cfg_.net_ring_depth);
            const int infer_lane = static_cast<int>(mb % cfg_.inference_parallelism);
            cudaStream_t infer_stream = stream_infer_lanes_[infer_lane];
            const int64_t start = mb * cfg_.b_micro;
            int micro_size = cfg_.b_micro;
            if (start + micro_size > active_limit) micro_size = static_cast<int>(active_limit - start);

            if (mb >= cfg_.score_ring_depth) CUDA_CHECK(cudaStreamWaitEvent(infer_stream, score_consumed_[score_slot], 0));
            if (mb >= cfg_.net_ring_depth) CUDA_CHECK(cudaStreamWaitEvent(stream_ingest_, net_consumed_[net_slot], 0));

            inference_->forward(beam_current_, current_active_flags_, score_ring_, score_slot, infer_lane, start, micro_size, cfg_, infer_stream);
            CUDA_CHECK(cudaEventRecord(score_ready_[score_slot], infer_stream));

            CUDA_CHECK(cudaStreamWaitEvent(stream_ingest_, score_ready_[score_slot], 0));
            const int64_t total_candidate_lanes = static_cast<int64_t>(micro_size) * cfg_.fanout;
            int64_t tile_lanes_target = cfg_.k_expand_tile > 0 ? static_cast<int64_t>(cfg_.k_expand_tile) : total_candidate_lanes;
            if (tile_lanes_target < 1) tile_lanes_target = total_candidate_lanes;
            int tile_idx = 0;
            for (int64_t candidate_lane_offset = 0; candidate_lane_offset < total_candidate_lanes; candidate_lane_offset += tile_lanes_target, ++tile_idx) {
                const int candidate_lanes = static_cast<int>(std::min<int64_t>(tile_lanes_target, total_candidate_lanes - candidate_lane_offset));
                launch_reset_net_slot(
                    reinterpret_cast<CandidateRecord*>(send_buckets_.data_ptr<uint8_t>()),
                    reinterpret_cast<CandidateRecord*>(recv_buckets_.data_ptr<uint8_t>()),
                    send_counts_.data_ptr<int32_t>(),
                    recv_counts_.data_ptr<int32_t>(),
                    net_slot, cfg_.world_size, cfg_.bucket_cap_per_peer, stream_ingest_);
                CUDA_CHECK(cudaGetLastError());

                launch_process_score_slot(
                    reinterpret_cast<const uint8_t*>(beam_current_.data_ptr<uint8_t>()),
                    current_active_flags_.data_ptr<uint8_t>(),
                    reinterpret_cast<const uint16_t*>(score_ring_.data_ptr<int16_t>()),
                    reinterpret_cast<uint8_t*>(next_state_pool_.data_ptr<uint8_t>()),
                    reinterpret_cast<BeamMeta*>(next_meta_.data_ptr<uint8_t>()),
                    reinterpret_cast<HashSlot*>(hash_table_.data_ptr<uint8_t>()),
                    active_flags_.data_ptr<uint8_t>(),
                    free_indices_.data_ptr<int32_t>(),
                    free_count_.data_ptr<int32_t>(),
                    reinterpret_cast<uint32_t*>(local_hist_.data_ptr<int32_t>()),
                    counters_.data_ptr<int32_t>(),
                    beam_status_.data_ptr<int32_t>(),
                    reinterpret_cast<CandidateRecord*>(send_buckets_.data_ptr<uint8_t>()),
                    send_counts_.data_ptr<int32_t>(),
                    threshold_cell_.data_ptr<int32_t>(),
                    score_slot, net_slot, cfg_.world_size, cfg_.rank,
                    cfg_.state_size_bytes, cfg_.fanout, cfg_.b_micro, start, micro_size,
                    candidate_lane_offset, candidate_lanes,
                    cfg_.bucket_cap_per_peer, static_cast<int>(hash_limit),
                    static_cast<int>(next_limit), cfg_.probe_limit, stream_ingest_);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(send_ready_[net_slot], stream_ingest_));

                CUDA_CHECK(cudaStreamWaitEvent(stream_net_, send_ready_[net_slot], 0));
                do_fixed_all_to_all(net_slot);
                CUDA_CHECK(cudaEventRecord(recv_ready_[net_slot], stream_net_));

                CUDA_CHECK(cudaStreamWaitEvent(stream_ingest_, recv_ready_[net_slot], 0));
                launch_ingest_recv_slot(
                    reinterpret_cast<const CandidateRecord*>(recv_buckets_.data_ptr<uint8_t>()),
                    recv_counts_.data_ptr<int32_t>(),
                    reinterpret_cast<uint8_t*>(next_state_pool_.data_ptr<uint8_t>()),
                    reinterpret_cast<BeamMeta*>(next_meta_.data_ptr<uint8_t>()),
                    reinterpret_cast<HashSlot*>(hash_table_.data_ptr<uint8_t>()),
                    active_flags_.data_ptr<uint8_t>(),
                    free_indices_.data_ptr<int32_t>(),
                    free_count_.data_ptr<int32_t>(),
                    reinterpret_cast<uint32_t*>(local_hist_.data_ptr<int32_t>()),
                    counters_.data_ptr<int32_t>(),
                    beam_status_.data_ptr<int32_t>(),
                    threshold_cell_.data_ptr<int32_t>(),
                    net_slot, cfg_.world_size, cfg_.state_size_bytes,
                    cfg_.bucket_cap_per_peer, static_cast<int>(hash_limit),
                    static_cast<int>(next_limit), cfg_.probe_limit, stream_ingest_);
                CUDA_CHECK(cudaGetLastError());

                if (cfg_.k_expand_tile > 0 && ((tile_idx + 1) % histogram_period_micro == 0)) enqueue_threshold_update(next_limit, hash_limit);
            }
            CUDA_CHECK(cudaEventRecord(score_consumed_[score_slot], stream_ingest_));
            CUDA_CHECK(cudaEventRecord(net_consumed_[net_slot], stream_ingest_));

            if (cfg_.k_expand_tile <= 0 && ((mb + 1) % histogram_period_micro == 0)) enqueue_threshold_update(next_limit, hash_limit);
        }
        if (step_timers_enabled_) CUDA_CHECK(cudaEventRecord(timing_after_micro_, stream_ingest_));

        enqueue_threshold_update(next_limit, hash_limit);
        launch_prune_by_threshold(
            reinterpret_cast<BeamMeta*>(next_meta_.data_ptr<uint8_t>()),
            reinterpret_cast<HashSlot*>(hash_table_.data_ptr<uint8_t>()),
            active_flags_.data_ptr<uint8_t>(),
            free_indices_.data_ptr<int32_t>(),
            free_count_.data_ptr<int32_t>(),
            reinterpret_cast<uint32_t*>(local_hist_.data_ptr<int32_t>()),
            counters_.data_ptr<int32_t>(),
            threshold_cell_.data_ptr<int32_t>(),
            static_cast<int>(next_limit),
            static_cast<int>(hash_limit),
            cfg_.probe_limit,
            stream_ingest_);
        CUDA_CHECK(cudaGetLastError());
        launch_clear_hash_table(reinterpret_cast<HashSlot*>(hash_table_.data_ptr<uint8_t>()), static_cast<int>(hash_limit), stream_ingest_);
        launch_rebuild_hash_from_active(
            reinterpret_cast<BeamMeta*>(next_meta_.data_ptr<uint8_t>()),
            reinterpret_cast<HashSlot*>(hash_table_.data_ptr<uint8_t>()),
            active_flags_.data_ptr<uint8_t>(),
            counters_.data_ptr<int32_t>(),
            static_cast<int>(next_limit),
            static_cast<int>(hash_limit),
            cfg_.probe_limit,
            stream_ingest_);
        CUDA_CHECK(cudaGetLastError());

        int64_t flags_clear = cfg_.n_local;
        if (logical_next_limit_ > 0 || logical_active_limit_ > 0 || active_limit_override_ > 0) {
            flags_clear = std::min<int64_t>(cfg_.n_local, std::max<int64_t>(active_limit, current_output_cap));
        }
        CUDA_CHECK(cudaMemsetAsync(current_active_flags_.data_ptr<uint8_t>(), 0, static_cast<size_t>(flags_clear), stream_ingest_));
        CUDA_CHECK(cudaMemsetAsync(beam_status_.data_ptr<int32_t>() + STATUS_FOUND, 0, 3 * sizeof(int32_t), stream_ingest_));
        CUDA_CHECK(cudaMemsetAsync(beam_status_.data_ptr<int32_t>() + STATUS_LOCAL_FOUND, 0, sizeof(int32_t), stream_ingest_));
        launch_compact_next_to_current(
            reinterpret_cast<const uint8_t*>(next_state_pool_.data_ptr<uint8_t>()),
            reinterpret_cast<const BeamMeta*>(next_meta_.data_ptr<uint8_t>()),
            active_flags_.data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(beam_current_.data_ptr<uint8_t>()),
            current_active_flags_.data_ptr<uint8_t>(),
            history_parent_idx_.data_ptr<int32_t>(),
            history_parent_rank_.data_ptr<uint8_t>(),
            history_action_.data_ptr<uint8_t>(),
            history_valid_.data_ptr<uint8_t>(),
            history_depth_cell_.data_ptr<int32_t>(),
            beam_status_.data_ptr<int32_t>(),
            cfg_.state_size_bytes,
            static_cast<int>(next_limit),
            static_cast<int>(current_output_cap),
            stream_ingest_);
        CUDA_CHECK(cudaGetLastError());
        enqueue_found_allreduce_and_finish();
        if (step_timers_enabled_) {
            CUDA_CHECK(cudaEventRecord(timing_after_final_, stream_net_));
            step_timing_valid_ = true;
        }
        current_history_depth_ += 1;
    }

    void capture_cuda_graph(int histogram_period_micro) {
        if (cuda_graph_captured_) return;
        if (!inference_warmed_) warmup_inference(2);
#if BEAM_DEBUG_ON
        if (debug_config.verbose) std::cout << "[BeamEngine] capturing CUDA Graph" << std::endl;
#endif
        CUDA_CHECK(cudaStreamBeginCapture(stream_infer_, cudaStreamCaptureModeGlobal));
        enqueue_one_depth(histogram_period_micro);
        CUDA_CHECK(cudaStreamEndCapture(stream_infer_, &cuda_graph_));
        CUDA_CHECK(cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, nullptr, nullptr, 0));
        cuda_graph_captured_ = true;
        int32_t one = 1;
        CUDA_CHECK(cudaMemcpyAsync(beam_status_.data_ptr<int32_t>() + STATUS_GRAPH_CAPTURED, &one, sizeof(int32_t), cudaMemcpyHostToDevice, stream_ingest_));
        CUDA_CHECK(cudaStreamSynchronize(stream_ingest_));
    }

    void print_debug_status() const {
        py::dict st = status();
        std::cout << "[BeamEngine] rank=" << cfg_.rank
                  << " current_size=" << st["current_size"].cast<int>()
                  << " found=" << st["found"].cast<int>()
                  << " counters=" << py::str(st["counters"]).cast<std::string>()
                  << std::endl;
    }
};

py::bytes get_nccl_unique_id() {
    ncclUniqueId id;
    NCCL_CHECK(ncclGetUniqueId(&id));
    return nccl_unique_id_to_bytes(id);
}

py::dict v6_dispatcher_skeleton_single_gpu_smoke_contract();

py::dict derive_sizes(py::dict cfg_dict) {
    EngineConfig cfg = config_from_dict(cfg_dict);
    beam_v6::TargetConfig target_cfg = target_config_from_engine(cfg);
    beam_v6::ScratchLayouts target_layouts = beam_v6::derive_scratch_layouts(target_cfg);
    py::dict d;
    d["n_local"] = cfg.n_local;
    d["k_keep"] = cfg.k_keep;
    d["k_work"] = cfg.k_work;
    d["hash_capacity"] = cfg.hash_capacity;
    d["k_expand_tile"] = cfg.k_expand_tile;
    d["score_ring_depth"] = cfg.score_ring_depth;
    d["bucket_cap_per_peer"] = cfg.bucket_cap_per_peer;
    d["bucket_cap_per_peer_safe"] = cfg.bucket_cap_per_peer_safe;
    d["send_buckets_gib"] = cfg.send_buckets_gib;
    d["recv_buckets_gib"] = cfg.recv_buckets_gib;
    d["total_bucket_gib"] = cfg.total_bucket_gib;
    d["score_ring_elements"] = static_cast<int64_t>(cfg.score_ring_depth) * cfg.b_micro * cfg.fanout;
    d["candidate_record_bytes"] = static_cast<int>(sizeof(CandidateRecord));
    d["beam_meta_bytes"] = static_cast<int>(sizeof(BeamMeta));
    d["hash_slot_bytes"] = static_cast<int>(sizeof(HashSlot));
    d["send_recv_records"] = static_cast<int64_t>(cfg.net_ring_depth) * cfg.world_size * cfg.bucket_cap_per_peer;
#if BEAM_HISTORY_CPU
    d["history_records"] = cfg.n_local;
    d["history_backend_cpu"] = 1;
#else
    d["history_records"] = static_cast<int64_t>(cfg.max_depth) * cfg.n_local;
    d["history_backend_cpu"] = 0;
#endif
    d["inference_parallelism"] = cfg.inference_parallelism;
    d["state128_bytes"] = static_cast<int>(sizeof(beam_v6::State128));
    d["hash128_bytes"] = static_cast<int>(sizeof(beam_v6::Hash128));
    d["candidate_meta_bytes"] = static_cast<int>(sizeof(beam_v6::CandidateMeta));
    d["final_request_bytes"] = static_cast<int>(sizeof(beam_v6::FinalRequest));
    d["final_response_bytes"] = static_cast<int>(sizeof(beam_v6::FinalResponse));
    d["goal_score_key"] = beam_v6::GOAL_SCORE_KEY;
    d["score_scale_v6"] = beam_v6::SCORE_SCALE;
    d["score_max_key"] = beam_v6::SCORE_MAX_KEY;
    d["score_bin_count_v6"] = beam_v6::SCORE_BIN_COUNT;
    d["stream3_batch_candidates"] = target_cfg.stream3_batch_candidates;
    d["stream4_batch_candidates"] = target_cfg.stream4_batch_candidates;
    d["stream4_batch_candidates_per_shard_unit"] = target_cfg.stream4_batch_candidates_per_shard_unit;
    d["ring_count"] = target_cfg.ring_count;
    d["ring_slot_count"] = target_cfg.ring_slot_count;
    d["shard_count"] = target_cfg.shard_count;
    d["global_spill_capacity"] = target_cfg.global_spill_capacity;
    d["solved_result_capacity"] = target_cfg.solved_result_capacity;
    d["user_global_beam_width"] = target_cfg.user_global_beam_width;
    d["global_beam_width_effective"] = target_cfg.global_beam_width_effective;
    d["global_beam_width_max_safe"] = target_cfg.global_beam_width_max_safe;
    d["beam_width_alignment"] = target_cfg.beam_width_alignment;
    d["layout_streams_bytes"] = static_cast<int64_t>(target_layouts.streams.bytes);
    d["layout_final_bytes"] = static_cast<int64_t>(target_layouts.final.bytes);
    d["scratch_pool_bytes"] = static_cast<int64_t>(target_layouts.scratch_pool_bytes);
    d["current_frontier_state128_bytes"] = static_cast<int64_t>(target_layouts.current_frontier_bytes);
    d["solved_buffers_bytes"] = static_cast<int64_t>(target_layouts.solved_buffers_bytes);
    return d;
}

void v6_stream2_hash_goal_py(
    torch::Tensor current_frontier_states,
    torch::Tensor parent_base,
    torch::Tensor count,
    torch::Tensor score_ring,
    torch::Tensor hash_ring,
    torch::Tensor generators,
    torch::Tensor central_state,
    torch::Tensor zobrist,
    torch::Tensor solved_flag,
    torch::Tensor stop_flag,
    torch::Tensor solved_count,
    torch::Tensor solved_overflow,
    torch::Tensor solved_meta_list,
    torch::Tensor solved_depth_list,
    int solved_result_capacity,
    int depth,
    int local_rank,
    int ring,
    int ring_slot,
    int ring_slot_count,
    int b_micro) {
    check_cuda_tensor(current_frontier_states, "current_frontier_states");
    check_cuda_i64_tensor(parent_base, "parent_base");
    check_cuda_i32_tensor(count, "count");
    check_cuda_i32_tensor(score_ring, "score_ring");
    check_cuda_tensor(hash_ring, "hash_ring");
    check_cuda_tensor(generators, "generators");
    check_cuda_tensor(central_state, "central_state");
    check_cuda_tensor(zobrist, "zobrist");
    check_cuda_i32_tensor(solved_flag, "solved_flag");
    check_cuda_i32_tensor(stop_flag, "stop_flag");
    check_cuda_i32_tensor(solved_count, "solved_count");
    check_cuda_i32_tensor(solved_overflow, "solved_overflow");
    check_cuda_tensor(solved_meta_list, "solved_meta_list");
    check_cuda_i32_tensor(solved_depth_list, "solved_depth_list");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_stream2_hash_goal(
        reinterpret_cast<const beam_v6::State128*>(current_frontier_states.data_ptr<uint8_t>()),
        reinterpret_cast<const uint64_t*>(parent_base.data_ptr<int64_t>()),
        reinterpret_cast<const uint32_t*>(count.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(score_ring.data_ptr<int32_t>()),
        reinterpret_cast<beam_v6::Hash128*>(hash_ring.data_ptr<uint8_t>()),
        generators.data_ptr<uint8_t>(),
        central_state.data_ptr<uint8_t>(),
        reinterpret_cast<const beam_v6::Hash128*>(zobrist.data_ptr<uint8_t>()),
        reinterpret_cast<uint32_t*>(solved_flag.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(stop_flag.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(solved_count.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(solved_overflow.data_ptr<int32_t>()),
        reinterpret_cast<beam_v6::CandidateMeta*>(solved_meta_list.data_ptr<uint8_t>()),
        reinterpret_cast<uint32_t*>(solved_depth_list.data_ptr<int32_t>()),
        static_cast<uint32_t>(solved_result_capacity),
        static_cast<uint32_t>(depth),
        static_cast<uint32_t>(local_rank),
        static_cast<uint32_t>(ring),
        static_cast<uint32_t>(ring_slot),
        static_cast<uint32_t>(ring_slot_count),
        static_cast<uint32_t>(b_micro),
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void v6_final_materialize_py(
    torch::Tensor current_frontier_states,
    torch::Tensor final_request_buffer,
    torch::Tensor generators,
    torch::Tensor final_response_buffer,
    int request_count) {
    check_cuda_tensor(current_frontier_states, "current_frontier_states");
    check_cuda_tensor(final_request_buffer, "final_request_buffer");
    check_cuda_tensor(generators, "generators");
    check_cuda_tensor(final_response_buffer, "final_response_buffer");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_final_materialize(
        reinterpret_cast<const beam_v6::State128*>(current_frontier_states.data_ptr<uint8_t>()),
        reinterpret_cast<const beam_v6::FinalRequest*>(final_request_buffer.data_ptr<uint8_t>()),
        generators.data_ptr<uint8_t>(),
        reinterpret_cast<beam_v6::FinalResponse*>(final_response_buffer.data_ptr<uint8_t>()),
        static_cast<uint32_t>(request_count),
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void v6_final_scatter_responses_py(
    torch::Tensor final_response_buffer,
    torch::Tensor next_frontier_states_tmp,
    int response_count) {
    check_cuda_tensor(final_response_buffer, "final_response_buffer");
    check_cuda_tensor(next_frontier_states_tmp, "next_frontier_states_tmp");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_final_scatter_responses(
        reinterpret_cast<const beam_v6::FinalResponse*>(final_response_buffer.data_ptr<uint8_t>()),
        reinterpret_cast<beam_v6::State128*>(next_frontier_states_tmp.data_ptr<uint8_t>()),
        static_cast<uint32_t>(response_count),
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void v6_stream3_pack_threshold_compact_py(
    torch::Tensor score_ring,
    torch::Tensor hash_ring,
    torch::Tensor parent_base,
    torch::Tensor count,
    torch::Tensor stream3_key_a,
    torch::Tensor stream3_val_a,
    torch::Tensor compact_count,
    int current_threshold,
    int ring,
    int ring_slot_count,
    int b_micro,
    int stream3_batch_candidates) {
    check_cuda_i32_tensor(score_ring, "score_ring");
    check_cuda_tensor(hash_ring, "hash_ring");
    check_cuda_i64_tensor(parent_base, "parent_base");
    check_cuda_i32_tensor(count, "count");
    check_cuda_tensor(stream3_key_a, "stream3_key_a");
    check_cuda_i64_tensor(stream3_val_a, "stream3_val_a");
    check_cuda_i32_tensor(compact_count, "compact_count");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_stream3_pack_threshold_compact(
        reinterpret_cast<const uint32_t*>(score_ring.data_ptr<int32_t>()),
        reinterpret_cast<const beam_v6::Hash128*>(hash_ring.data_ptr<uint8_t>()),
        reinterpret_cast<const uint64_t*>(parent_base.data_ptr<int64_t>()),
        reinterpret_cast<const uint32_t*>(count.data_ptr<int32_t>()),
        reinterpret_cast<beam_v6::Hash128Key*>(stream3_key_a.data_ptr<uint8_t>()),
        reinterpret_cast<uint64_t*>(stream3_val_a.data_ptr<int64_t>()),
        reinterpret_cast<uint32_t*>(compact_count.data_ptr<int32_t>()),
        static_cast<uint32_t>(current_threshold),
        static_cast<uint32_t>(ring),
        static_cast<uint32_t>(ring_slot_count),
        static_cast<uint32_t>(b_micro),
        static_cast<uint32_t>(stream3_batch_candidates),
        stream);
    CUDA_CHECK(cudaGetLastError());
}

size_t v6_stream3_sort_temp_bytes_py(int item_count) {
    size_t temp_storage_bytes = 0;
    launch_v6_stream3_sort_pairs(nullptr, temp_storage_bytes, nullptr, nullptr, nullptr, nullptr, item_count, nullptr);
    return temp_storage_bytes;
}

void v6_stream3_sort_pairs_py(
    torch::Tensor temp_storage,
    torch::Tensor key_in,
    torch::Tensor key_out,
    torch::Tensor val_in,
    torch::Tensor val_out,
    int item_count) {
    check_cuda_tensor(temp_storage, "temp_storage");
    check_cuda_tensor(key_in, "key_in");
    check_cuda_tensor(key_out, "key_out");
    check_cuda_i64_tensor(val_in, "val_in");
    check_cuda_i64_tensor(val_out, "val_out");
    size_t temp_storage_bytes = static_cast<size_t>(temp_storage.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_stream3_sort_pairs(
        temp_storage.data_ptr<uint8_t>(),
        temp_storage_bytes,
        reinterpret_cast<const beam_v6::Hash128Key*>(key_in.data_ptr<uint8_t>()),
        reinterpret_cast<beam_v6::Hash128Key*>(key_out.data_ptr<uint8_t>()),
        reinterpret_cast<const uint64_t*>(val_in.data_ptr<int64_t>()),
        reinterpret_cast<uint64_t*>(val_out.data_ptr<int64_t>()),
        item_count,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void v6_stream3_dedup_sorted_py(
    torch::Tensor sorted_key,
    torch::Tensor sorted_val,
    torch::Tensor unique_key,
    torch::Tensor unique_val,
    torch::Tensor unique_count,
    int compact_count) {
    check_cuda_tensor(sorted_key, "sorted_key");
    check_cuda_i64_tensor(sorted_val, "sorted_val");
    check_cuda_tensor(unique_key, "unique_key");
    check_cuda_i64_tensor(unique_val, "unique_val");
    check_cuda_i32_tensor(unique_count, "unique_count");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_stream3_dedup_sorted(
        reinterpret_cast<const beam_v6::Hash128Key*>(sorted_key.data_ptr<uint8_t>()),
        reinterpret_cast<const uint64_t*>(sorted_val.data_ptr<int64_t>()),
        reinterpret_cast<beam_v6::Hash128*>(unique_key.data_ptr<uint8_t>()),
        reinterpret_cast<uint64_t*>(unique_val.data_ptr<int64_t>()),
        reinterpret_cast<uint32_t*>(unique_count.data_ptr<int32_t>()),
        static_cast<uint32_t>(compact_count),
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void v6_stream3_restore_split_py(
    torch::Tensor unique_key,
    torch::Tensor unique_val,
    torch::Tensor parent_base,
    torch::Tensor local_pending_buffer,
    torch::Tensor remote_send_buffer,
    torch::Tensor local_count,
    torch::Tensor send_count,
    torch::Tensor send_offset,
    int unique_count,
    int local_rank,
    int world_size,
    int ring,
    int ring_slot_count,
    int b_micro) {
    check_cuda_tensor(unique_key, "unique_key");
    check_cuda_i64_tensor(unique_val, "unique_val");
    check_cuda_i64_tensor(parent_base, "parent_base");
    check_cuda_tensor(local_pending_buffer, "local_pending_buffer");
    check_cuda_tensor(remote_send_buffer, "remote_send_buffer");
    check_cuda_i32_tensor(local_count, "local_count");
    check_cuda_i32_tensor(send_count, "send_count");
    check_cuda_i32_tensor(send_offset, "send_offset");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_stream3_restore_split(
        reinterpret_cast<const beam_v6::Hash128*>(unique_key.data_ptr<uint8_t>()),
        reinterpret_cast<const uint64_t*>(unique_val.data_ptr<int64_t>()),
        reinterpret_cast<const uint64_t*>(parent_base.data_ptr<int64_t>()),
        reinterpret_cast<beam_v6::CandidateMeta*>(local_pending_buffer.data_ptr<uint8_t>()),
        reinterpret_cast<beam_v6::CandidateMeta*>(remote_send_buffer.data_ptr<uint8_t>()),
        reinterpret_cast<uint32_t*>(local_count.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(send_count.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(send_offset.data_ptr<int32_t>()),
        static_cast<uint32_t>(unique_count),
        static_cast<uint32_t>(local_rank),
        static_cast<uint32_t>(world_size),
        static_cast<uint32_t>(ring),
        static_cast<uint32_t>(ring_slot_count),
        static_cast<uint32_t>(b_micro),
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void v6_stream4_threshold_compact_py(
    torch::Tensor survivor_shard,
    torch::Tensor stream4_key_a,
    torch::Tensor stream4_val_a,
    torch::Tensor compact_count,
    int input_count,
    int stream4_job_threshold) {
    check_cuda_tensor(survivor_shard, "survivor_shard");
    check_cuda_tensor(stream4_key_a, "stream4_key_a");
    check_cuda_tensor(stream4_val_a, "stream4_val_a");
    check_cuda_i32_tensor(compact_count, "compact_count");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_stream4_threshold_compact(
        reinterpret_cast<const beam_v6::CandidateMeta*>(survivor_shard.data_ptr<uint8_t>()),
        reinterpret_cast<beam_v6::Stream4HashKey*>(stream4_key_a.data_ptr<uint8_t>()),
        reinterpret_cast<beam_v6::CandidateMeta*>(stream4_val_a.data_ptr<uint8_t>()),
        reinterpret_cast<uint32_t*>(compact_count.data_ptr<int32_t>()),
        static_cast<uint32_t>(input_count),
        static_cast<uint32_t>(stream4_job_threshold),
        stream);
    CUDA_CHECK(cudaGetLastError());
}

size_t v6_stream4_sort_temp_bytes_py(int item_count) {
    size_t temp_storage_bytes = 0;
    launch_v6_stream4_sort_pairs(nullptr, temp_storage_bytes, nullptr, nullptr, nullptr, nullptr, item_count, nullptr);
    return temp_storage_bytes;
}

void v6_stream4_sort_pairs_py(
    torch::Tensor temp_storage,
    torch::Tensor key_in,
    torch::Tensor key_out,
    torch::Tensor val_in,
    torch::Tensor val_out,
    int item_count) {
    check_cuda_tensor(temp_storage, "temp_storage");
    check_cuda_tensor(key_in, "key_in");
    check_cuda_tensor(key_out, "key_out");
    check_cuda_tensor(val_in, "val_in");
    check_cuda_tensor(val_out, "val_out");
    size_t temp_storage_bytes = static_cast<size_t>(temp_storage.numel());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_stream4_sort_pairs(
        temp_storage.data_ptr<uint8_t>(),
        temp_storage_bytes,
        reinterpret_cast<const beam_v6::Stream4HashKey*>(key_in.data_ptr<uint8_t>()),
        reinterpret_cast<beam_v6::Stream4HashKey*>(key_out.data_ptr<uint8_t>()),
        reinterpret_cast<const beam_v6::CandidateMeta*>(val_in.data_ptr<uint8_t>()),
        reinterpret_cast<beam_v6::CandidateMeta*>(val_out.data_ptr<uint8_t>()),
        item_count,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void v6_stream4_dedup_sorted_py(
    torch::Tensor sorted_key,
    torch::Tensor sorted_val,
    torch::Tensor clean_tmp,
    torch::Tensor new_clean_count,
    int compact_count) {
    check_cuda_tensor(sorted_key, "sorted_key");
    check_cuda_tensor(sorted_val, "sorted_val");
    check_cuda_tensor(clean_tmp, "clean_tmp");
    check_cuda_i32_tensor(new_clean_count, "new_clean_count");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_stream4_dedup_sorted(
        reinterpret_cast<const beam_v6::Stream4HashKey*>(sorted_key.data_ptr<uint8_t>()),
        reinterpret_cast<const beam_v6::CandidateMeta*>(sorted_val.data_ptr<uint8_t>()),
        reinterpret_cast<beam_v6::CandidateMeta*>(clean_tmp.data_ptr<uint8_t>()),
        reinterpret_cast<uint32_t*>(new_clean_count.data_ptr<int32_t>()),
        static_cast<uint32_t>(compact_count),
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void v6_stream4_write_clean_py(
    torch::Tensor survivor_shard,
    torch::Tensor clean_tmp,
    torch::Tensor clean_count,
    torch::Tensor dirty_count,
    torch::Tensor processing_flag,
    int new_clean_count) {
    check_cuda_tensor(survivor_shard, "survivor_shard");
    check_cuda_tensor(clean_tmp, "clean_tmp");
    check_cuda_i32_tensor(clean_count, "clean_count");
    check_cuda_i32_tensor(dirty_count, "dirty_count");
    check_cuda_tensor(processing_flag, "processing_flag");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    launch_v6_stream4_write_clean(
        reinterpret_cast<beam_v6::CandidateMeta*>(survivor_shard.data_ptr<uint8_t>()),
        reinterpret_cast<const beam_v6::CandidateMeta*>(clean_tmp.data_ptr<uint8_t>()),
        reinterpret_cast<uint32_t*>(clean_count.data_ptr<int32_t>()),
        reinterpret_cast<uint32_t*>(dirty_count.data_ptr<int32_t>()),
        processing_flag.data_ptr<uint8_t>(),
        static_cast<uint32_t>(new_clean_count),
        stream);
    CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GPU-resident distributed beam search engine: CUDA Graph + streams + NCCL";
    m.def("get_nccl_unique_id", &get_nccl_unique_id, "Create ncclUniqueId");
    m.def("derive_sizes", &derive_sizes, "Derive static buffer sizes");
    m.def("v6_stream2_hash_goal", &v6_stream2_hash_goal_py, "Run v6 Stream2 hash/goal kernel");
    m.def("v6_final_materialize", &v6_final_materialize_py, "Run v6 final materialize kernel");
    m.def("v6_final_scatter_responses", &v6_final_scatter_responses_py, "Run v6 final scatter responses kernel");
    m.def("v6_stream3_pack_threshold_compact", &v6_stream3_pack_threshold_compact_py, "Run v6 Stream3 threshold/compact/pack kernel");
    m.def("v6_stream3_sort_temp_bytes", &v6_stream3_sort_temp_bytes_py, "Get v6 Stream3 CUB sort temp bytes");
    m.def("v6_stream3_sort_pairs", &v6_stream3_sort_pairs_py, "Run v6 Stream3 CUB sort pairs");
    m.def("v6_stream3_dedup_sorted", &v6_stream3_dedup_sorted_py, "Run v6 Stream3 sorted-key dedup");
    m.def("v6_stream3_restore_split", &v6_stream3_restore_split_py, "Run v6 Stream3 restore/split");
    m.def("v6_stream4_threshold_compact", &v6_stream4_threshold_compact_py, "Run v6 Stream4 threshold/compact");
    m.def("v6_stream4_sort_temp_bytes", &v6_stream4_sort_temp_bytes_py, "Get v6 Stream4 CUB sort temp bytes");
    m.def("v6_stream4_sort_pairs", &v6_stream4_sort_pairs_py, "Run v6 Stream4 CUB sort pairs");
    m.def("v6_stream4_dedup_sorted", &v6_stream4_dedup_sorted_py, "Run v6 Stream4 sorted-key dedup");
    m.def("v6_stream4_write_clean", &v6_stream4_write_clean_py, "Run v6 Stream4 clean writeback");
    m.def("v6_dispatcher_skeleton_single_gpu_smoke", &v6_dispatcher_skeleton_single_gpu_smoke_contract,
          "Return v6 dispatcher skeleton single-GPU smoke contract metadata");
    py::class_<BeamEngine>(m, "BeamEngine")
        .def(py::init<py::dict, py::dict, std::string>(), py::arg("cfg"), py::arg("buffers"), py::arg("backend") = "dummy")
        .def("init_nccl", &BeamEngine::init_nccl, py::arg("unique_id_bytes"))
        .def("v6_stream5_exchange_candidate_meta", &BeamEngine::v6_stream5_exchange_candidate_meta,
             py::arg("remote_send_buffer"), py::arg("remote_recv_buffer"),
             py::arg("send_count"), py::arg("send_offset"),
             py::arg("recv_count"), py::arg("recv_offset"))
        .def("load_torchscript_ensemble", &BeamEngine::load_torchscript_ensemble, py::arg("module_paths"))
        .def("load_fullbeamnice_static", &BeamEngine::load_fullbeamnice_static, py::arg("weights"))
        .def("begin_uniform_score", &BeamEngine::begin_uniform_score, py::arg("score_q"))
        .def("set_uniform_score", &BeamEngine::set_uniform_score, py::arg("score_q"))
        .def("end_uniform_score", &BeamEngine::end_uniform_score)
        .def("warmup_inference", &BeamEngine::warmup_inference, py::arg("repeats") = 2)
        .def("benchmark_inference", &BeamEngine::benchmark_inference, py::arg("micro_size"), py::arg("repeats") = 50, py::arg("warmup") = 10)
        .def("set_action_permutation_table", &BeamEngine::set_action_permutation_table, py::arg("table_bytes"))
        .def("set_central_state", &BeamEngine::set_central_state, py::arg("state_bytes"))
        .def("reset_search", &BeamEngine::reset_search, py::arg("initial_state_bytes"), py::arg("active_on_this_rank") = true)
        .def("clear_runtime_state", &BeamEngine::clear_runtime_state)
        .def("enable_cuda_graphs", &BeamEngine::enable_cuda_graphs, py::arg("enable") = true)
        .def("cuda_graph_captured", &BeamEngine::cuda_graph_captured)
        .def("enable_step_timers", &BeamEngine::enable_step_timers, py::arg("enable"))
        .def("step_timing", &BeamEngine::step_timing)
        .def("set_active_limit", &BeamEngine::set_active_limit, py::arg("active_limit"))
        .def("set_next_limit", &BeamEngine::set_next_limit, py::arg("next_limit"))
        .def("clear_logical_limits", &BeamEngine::clear_logical_limits)
        .def("enable_debug", &BeamEngine::enable_debug, py::arg("verbose") = true, py::arg("print_counters") = true, py::arg("log_period") = 8)
        .def("step", &BeamEngine::step, py::arg("histogram_period_micro") = 8)
        .def("step_current", &BeamEngine::step_current, py::arg("histogram_period_micro") = 8)
        .def("step_prepass_fast", &BeamEngine::step_prepass_fast, py::arg("score_q"))
        .def("set_prepass_light_solved_scan", &BeamEngine::set_prepass_light_solved_scan, py::arg("enable"))
        .def("search", &BeamEngine::search, py::arg("max_depth"), py::arg("histogram_period_micro") = 8)
        .def("status", &BeamEngine::status)
        .def("history_entry", &BeamEngine::history_entry, py::arg("depth"), py::arg("local_index"))
        .def("current_state_bytes", &BeamEngine::current_state_bytes, py::arg("local_index"))
        .def("sizes", &BeamEngine::sizes);
}
