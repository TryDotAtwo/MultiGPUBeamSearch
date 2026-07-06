#include "stream1_libtorch_ring_slot_launcher.hpp"

#include "cuda_check.hpp"
#include "stream1_transformer_libtorch_backend.hpp"
#include "../src/config.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/InferenceMode.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace beam::stream1_libtorch {
namespace {

void require_not_null(const void* ptr, const char* name) {
    if (ptr == nullptr) {
        throw std::invalid_argument(std::string("LibTorch Stream1 launcher missing pointer: ") + name);
    }
}

torch::Tensor score_keys_i32(const torch::Tensor& logits) {
    return torch::round(torch::clamp(logits.to(torch::kFloat32), 0.0, kScoreMaxQ) * kScoreScale)
        .to(torch::kInt32)
        .contiguous();
}

} // namespace

struct RingSlotLauncher::Impl {
    RingSlotLauncherConfig config;
    PieceTransformerLibTorch model;
    DispatcherRingSlotLauncher launcher{};
    std::vector<cudaEvent_t> score_ready;
    std::vector<cudaEvent_t> hash_ready;

    explicit Impl(RingSlotLauncherConfig cfg)
        : config(std::move(cfg)),
          model(config.weight_dir, torch::Device(torch::kCUDA, config.device_index)) {
        if (config.inference_parallelism == 0U) {
            throw std::invalid_argument("LibTorch Stream1 launcher requires nonzero inference_parallelism");
        }
        if (config.b_micro == 0U) {
            throw std::invalid_argument("LibTorch Stream1 launcher requires nonzero b_micro");
        }
        if (config.move_count != MOVE_COUNT) {
            throw std::invalid_argument("LibTorch Stream1 launcher move_count must match compile-time MOVE_COUNT");
        }
        if (model.output_dim != MOVE_COUNT || model.move_count != MOVE_COUNT) {
            throw std::runtime_error("LibTorch Stream1 launcher supports only multi-output piece_transformer with output_dim == MOVE_COUNT");
        }
        if (model.state_len != STATE_LEN) {
            throw std::runtime_error("LibTorch Stream1 launcher state_len must match compile-time STATE_LEN");
        }
        require_not_null(config.current_frontier_states, "current_frontier_states");
        require_not_null(config.parent_base, "parent_base");
        require_not_null(config.count, "count");
        require_not_null(config.score_ring, "score_ring");
        require_not_null(config.hash_ring, "hash_ring");
        require_not_null(config.generators, "generators");
        require_not_null(config.central_state, "central_state");
        require_not_null(config.zobrist, "zobrist");
        score_ready.resize(config.inference_parallelism, nullptr);
        hash_ready.resize(config.inference_parallelism, nullptr);
        try {
            for (std::uint32_t lane = 0; lane < config.inference_parallelism; ++lane) {
                BEAM_CUDA_CHECK(cudaEventCreateWithFlags(&score_ready[lane], cudaEventDisableTiming));
                BEAM_CUDA_CHECK(cudaEventCreateWithFlags(&hash_ready[lane], cudaEventDisableTiming));
            }
        } catch (...) {
            destroy_events();
            throw;
        }
        launcher.launch = &Impl::launch_static;
        launcher.user = this;
        launcher.name = "libtorch_eager";
    }

    ~Impl() {
        destroy_events();
    }

    void destroy_events() noexcept {
        for (cudaEvent_t& event : score_ready) {
            if (event != nullptr) {
                cudaEventDestroy(event);
                event = nullptr;
            }
        }
        for (cudaEvent_t& event : hash_ready) {
            if (event != nullptr) {
                cudaEventDestroy(event);
                event = nullptr;
            }
        }
    }

    static void launch_static(const DispatcherRingSlotLaunchContext& context, void* user) {
        if (user == nullptr) {
            throw std::invalid_argument("LibTorch Stream1 launcher missing user pointer");
        }
        static_cast<Impl*>(user)->launch(context);
    }

    void launch(const DispatcherRingSlotLaunchContext& context) {
        if (context.lane >= config.inference_parallelism) {
            throw std::invalid_argument("LibTorch Stream1 launcher lane exceeds inference_parallelism");
        }
        if (context.count > config.b_micro) {
            throw std::invalid_argument("LibTorch Stream1 launcher count exceeds b_micro");
        }
        if (context.b_micro != config.b_micro) {
            throw std::invalid_argument("LibTorch Stream1 launcher b_micro mismatch");
        }
        torch::NoGradGuard no_grad;
        c10::InferenceMode inference_mode;
        const auto stream = c10::cuda::getStreamFromExternal(context.stream1_lane, config.device_index);
        const c10::cuda::CUDAStreamGuard stream_guard(stream);

        auto state_options = torch::TensorOptions().device(torch::Device(torch::kCUDA, config.device_index)).dtype(torch::kUInt8);
        auto score_options = torch::TensorOptions().device(torch::Device(torch::kCUDA, config.device_index)).dtype(torch::kInt32);
        torch::Tensor states = torch::from_blob(
            const_cast<State128*>(config.current_frontier_states) + context.parent_base,
            {static_cast<std::int64_t>(context.count), static_cast<std::int64_t>(STATE_STORAGE_LEN)},
            state_options);
        torch::Tensor logits = model.forward(states);
        if (logits.dim() != 2 || logits.size(0) != static_cast<std::int64_t>(context.count) ||
            logits.size(1) != static_cast<std::int64_t>(MOVE_COUNT)) {
            throw std::runtime_error("LibTorch Stream1 logits shape does not match [count, MOVE_COUNT]");
        }
        torch::Tensor keys = score_keys_i32(logits);
        torch::Tensor score_out = torch::from_blob(
            config.score_ring + context.candidate_offset,
            {static_cast<std::int64_t>(context.count), static_cast<std::int64_t>(MOVE_COUNT)},
            score_options);
        score_out.copy_(keys, true);

        BEAM_CUDA_CHECK(cudaEventRecord(score_ready[context.lane], context.stream1_lane));
        BEAM_CUDA_CHECK(cudaStreamWaitEvent(context.stream2_lane, score_ready[context.lane], 0));
        stream2_hash_goal_cuda(
            config.current_frontier_states,
            config.parent_base + context.job,
            config.count + context.job,
            config.generators,
            config.central_state,
            config.zobrist,
            config.hash_ring + context.candidate_offset,
            0,
            0,
            config.b_micro,
            0,
            config.local_rank,
            config.solved,
            context.stream2_lane);
        BEAM_CUDA_CHECK(cudaEventRecord(hash_ready[context.lane], context.stream2_lane));
        BEAM_CUDA_CHECK(cudaStreamWaitEvent(context.stream1_lane, hash_ready[context.lane], 0));
    }
};

RingSlotLauncher::RingSlotLauncher(RingSlotLauncherConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

RingSlotLauncher::~RingSlotLauncher() = default;
RingSlotLauncher::RingSlotLauncher(RingSlotLauncher&&) noexcept = default;
RingSlotLauncher& RingSlotLauncher::operator=(RingSlotLauncher&&) noexcept = default;

const DispatcherRingSlotLauncher& RingSlotLauncher::dispatcher_launcher() const {
    if (!impl_) {
        throw std::runtime_error("moved-from LibTorch Stream1 launcher");
    }
    return impl_->launcher;
}

} // namespace beam::stream1_libtorch