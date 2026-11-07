#include "stream1_libtorch_ring_slot_launcher.hpp"

#include "cuda_check.hpp"
#include "stream1_transformer_libtorch_backend.hpp"
#include "stream1_mlp_libtorch_backend.hpp"
#include "cube444_blend_libtorch.hpp"
#include "stream1_ensemble_libtorch.hpp"
#include "../src/config.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/InferenceMode.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <iostream>
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
    std::unique_ptr<PieceTransformerLibTorch> transformer;
    std::unique_ptr<MlpLibTorch> mlp;
    std::unique_ptr<Cube444Blend> blend;
    std::unique_ptr<NativeEnsemble> ensemble;
    bool transformer_only=false;
    bool benchmark_mode=false;
    std::uint64_t submitted_parents=0;
    std::chrono::steady_clock::time_point progress_time=std::chrono::steady_clock::now();
    std::vector<cudaStream_t> mlp_streams;
    std::vector<cudaEvent_t> parents_ready, mlp_ready;
    std::vector<torch::Tensor> mlp_outputs;
    DispatcherRingSlotLauncher launcher{};
    std::vector<cudaEvent_t> score_ready;
    std::vector<cudaEvent_t> hash_ready;

    explicit Impl(RingSlotLauncherConfig cfg)
        : config(std::move(cfg)) {
        const auto device = torch::Device(torch::kCUDA, config.device_index);
        const auto manifest = config.blend_dir.empty() ? read_text_exact(config.weight_dir / "manifest.json") : std::string{};
        if (!config.blend_dir.empty() && fs::exists(config.blend_dir/"ensemble.json")) {
            if(!config.numeric_error) throw std::runtime_error("ensemble requires sticky numeric error flag");
            cudaDeviceProp properties{};
            BEAM_CUDA_CHECK(cudaGetDeviceProperties(&properties,config.device_index));
            ensemble=std::make_unique<NativeEnsemble>(config.blend_dir,device,config.b_micro,config.inference_parallelism);
            std::cout<<"ensemble_model_count="<<ensemble->heads.size()
                     <<" ensemble_schedule=serial execution_precision=fp16 accumulation_precision=fp32"<<std::endl;
        } else if (!config.blend_dir.empty()) {
            if (STATE_LEN!=96 || MOVE_COUNT!=24 || !config.numeric_error)
                throw std::runtime_error("Cube444 blend requires state96/moves24 and numeric flag");
            if(config.b_micro>256 || config.inference_parallelism!=1)
                throw std::runtime_error("initial blend memory admission requires b_micro<=256 and one lane");
            cudaDeviceProp properties{};
            BEAM_CUDA_CHECK(cudaGetDeviceProperties(&properties,config.device_index));
            if(properties.major<8) throw std::runtime_error("fused Cube444 blend requires SM80 or newer");
            benchmark_mode=std::getenv("BEAM_BENCHMARK_FRONTIER_FILE")!=nullptr;
            transformer_only=std::getenv("BEAM_BENCHMARK_TRANSFORMER_ONLY") && std::string(std::getenv("BEAM_BENCHMARK_TRANSFORMER_ONLY"))=="1";
            if(transformer_only && !std::getenv("BEAM_BENCHMARK_FRONTIER_FILE"))
                throw std::runtime_error("Transformer-only comparison requires explicit benchmark frontier");
            blend=std::make_unique<Cube444Blend>(config.blend_dir,device);
            std::cout << "blend_execution_precision=fp16 blend_accumulation_precision=fp32 blend_launch_order=transformer_first" << std::endl;
            if(std::getenv("BEAM_BENCHMARK_FRONTIER_FILE")) {
                auto input=torch::zeros({config.b_micro,96},torch::TensorOptions().device(device).dtype(torch::kLong));
                for(int i=0;i<3;++i) {
                    auto a=blend->transformer_cls(input); auto b=blend->mlp(input);
                    auto keys=torch::empty({config.b_micro,24},input.options().dtype(torch::kInt32));
                    blend->readout(a,b,reinterpret_cast<std::uint32_t*>(keys.data_ptr<int>()),config.b_micro,config.numeric_error,c10::cuda::getCurrentCUDAStream(config.device_index).stream(),transformer_only);
                    BEAM_CUDA_CHECK(cudaDeviceSynchronize());
                }
            }
        } else if (manifest.find("piece_transformer") != std::string::npos) {
            transformer = std::make_unique<PieceTransformerLibTorch>(config.weight_dir, device);
        } else {
            mlp = std::make_unique<MlpLibTorch>(config.weight_dir, device);
        }
        if (config.inference_parallelism == 0U) {
            throw std::invalid_argument("LibTorch Stream1 launcher requires nonzero inference_parallelism");
        }
        if (config.b_micro == 0U) {
            throw std::invalid_argument("LibTorch Stream1 launcher requires nonzero b_micro");
        }
        if (config.move_count != MOVE_COUNT) {
            throw std::invalid_argument("LibTorch Stream1 launcher move_count must match compile-time MOVE_COUNT");
        }
        const auto outputs = ensemble ? ensemble->output_dim : blend ? 24U : transformer ? transformer->output_dim : mlp->output_dim;
        if ((transformer && transformer->move_count != MOVE_COUNT) || (outputs != MOVE_COUNT && outputs != 1)) {
            throw std::runtime_error("LibTorch Stream1 requires scalar or MOVE_COUNT outputs");
        }
        if ((ensemble ? ensemble->state_len : blend ? 96U : transformer ? transformer->state_len : mlp->state_len) != STATE_LEN) {
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
        mlp_streams.resize(config.inference_parallelism,nullptr);
        parents_ready.resize(config.inference_parallelism,nullptr);
        mlp_ready.resize(config.inference_parallelism,nullptr);
        mlp_outputs.resize(config.inference_parallelism);
        try {
            for (std::uint32_t lane = 0; lane < config.inference_parallelism; ++lane) {
                BEAM_CUDA_CHECK(cudaEventCreateWithFlags(&score_ready[lane], cudaEventDisableTiming));
                BEAM_CUDA_CHECK(cudaEventCreateWithFlags(&hash_ready[lane], cudaEventDisableTiming));
                if(blend) {
                    BEAM_CUDA_CHECK(cudaStreamCreateWithFlags(&mlp_streams[lane],cudaStreamNonBlocking));
                    BEAM_CUDA_CHECK(cudaEventCreateWithFlags(&parents_ready[lane],cudaEventDisableTiming));
                    BEAM_CUDA_CHECK(cudaEventCreateWithFlags(&mlp_ready[lane],cudaEventDisableTiming));
                }
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
        for(auto s:mlp_streams) if(s) { cudaStreamSynchronize(s); cudaStreamDestroy(s); }
        for(auto e:parents_ready) if(e) cudaEventDestroy(e);
        for(auto e:mlp_ready) if(e) cudaEventDestroy(e);
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
        torch::Tensor logits;
        const auto outputs = ensemble ? ensemble->output_dim : blend ? 24U : transformer ? transformer->output_dim : mlp->output_dim;
        if (ensemble) {
            ensemble->score(states,config.score_ring+context.candidate_offset,
                            config.numeric_error,context.stream1_lane,context.lane,config.generators);
        } else if (blend) {
            auto lane=context.lane;
            states=states.narrow(1,0,96).to(torch::kLong).contiguous();
            if(!transformer_only) {
            BEAM_CUDA_CHECK(cudaEventRecord(parents_ready[lane],context.stream1_lane));
            BEAM_CUDA_CHECK(cudaStreamWaitEvent(mlp_streams[lane],parents_ready[lane],0));
            }
            // Queue the long backbone before submitting the short MLP. The
            // producer event still protects shared parents on the other stream.
            auto cls=blend->transformer_cls(states);
            if(!transformer_only) {
            {
                c10::cuda::CUDAStreamGuard mlp_guard(c10::cuda::getStreamFromExternal(mlp_streams[lane],config.device_index));
                mlp_outputs[lane]=blend->mlp(states).contiguous();
                BEAM_CUDA_CHECK(cudaEventRecord(mlp_ready[lane],mlp_streams[lane]));
            }
            }
            if(!transformer_only) BEAM_CUDA_CHECK(cudaStreamWaitEvent(context.stream1_lane,mlp_ready[lane],0));
            if(std::getenv("BEAM_BLEND_DIAGNOSTIC")) {
                if(!torch::isfinite(cls).all().item<bool>() || (!transformer_only && !torch::isfinite(mlp_outputs[lane]).all().item<bool>()))
                    throw std::runtime_error("blend nonfinite backbone, count="+std::to_string(context.count));
            }
            blend->readout(cls,mlp_outputs[lane],config.score_ring+context.candidate_offset,
                context.count,config.numeric_error,context.stream1_lane,transformer_only);
            if(std::getenv("BEAM_BLEND_DIAGNOSTIC")) {
                BEAM_CUDA_CHECK(cudaStreamSynchronize(context.stream1_lane));
                std::uint32_t error=0; BEAM_CUDA_CHECK(cudaMemcpy(&error,config.numeric_error,4,cudaMemcpyDeviceToHost));
                if(error) throw std::runtime_error("blend epilogue flagged finite heads, count="+std::to_string(context.count));
            }
        } else if (outputs == 1) {
            auto children = scalar_children(states, config.generators, MOVE_COUNT, STATE_LEN, STATE_STORAGE_LEN);
            logits = (transformer ? transformer->forward(children) : mlp->forward(children))
                .reshape({static_cast<std::int64_t>(context.count), MOVE_COUNT});
        } else {
            logits = transformer ? transformer->forward(states) : mlp->forward(states);
        }
        if (!blend && !ensemble) {
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
        }

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
        if(benchmark_mode) {
            submitted_parents+=context.count;
            const auto now=std::chrono::steady_clock::now();
            if(now-progress_time>=std::chrono::seconds(60)) {
                std::cout << "benchmark_scored_parents=" << submitted_parents << " rank=" << config.local_rank << std::endl;
                progress_time=now;
            }
        }
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
