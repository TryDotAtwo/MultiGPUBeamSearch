#include "stream1_transformer_libtorch_backend.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
    fs::path weight_dir;
    std::vector<std::int64_t> batches{384, 512, 768, 1024};
    std::int64_t warmup = 10;
    std::int64_t iters = 50;
    std::string device = "cuda:0";
    fs::path csv_path;
    bool cuda_graph = false;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };
        if (key == "--weight-dir") {
            args.weight_dir = require_value("--weight-dir");
        } else if (key == "--batches") {
            args.batches = beam::stream1_libtorch::parse_csv_i64(require_value("--batches"));
        } else if (key == "--warmup") {
            args.warmup = std::stoll(require_value("--warmup"));
        } else if (key == "--iters") {
            args.iters = std::stoll(require_value("--iters"));
        } else if (key == "--device") {
            args.device = require_value("--device");
        } else if (key == "--csv") {
            args.csv_path = require_value("--csv");
        } else if (key == "--cuda-graph") {
            args.cuda_graph = true;
        } else {
            throw std::runtime_error("unknown argument: " + key);
        }
    }
    if (args.weight_dir.empty()) {
        const char* env = std::getenv("BEAM_WEIGHT_DIR");
        if (env != nullptr && env[0] != '\0') {
            args.weight_dir = env;
        }
    }
    if (args.weight_dir.empty()) {
        throw std::runtime_error("--weight-dir or BEAM_WEIGHT_DIR is required");
    }
    if (args.warmup < 0 || args.iters <= 0) {
        throw std::runtime_error("--warmup must be >= 0 and --iters must be > 0");
    }
    return args;
}


std::string format_first_score_keys(const torch::Tensor& keys) {
    if (keys.size(0) <= 0) {
        return "";
    }
    torch::Tensor first = keys.index({0}).to(torch::kCPU).contiguous();
    auto accessor = first.accessor<std::int64_t, 1>();
    std::ostringstream out;
    for (std::int64_t i = 0; i < first.size(0); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << accessor[i];
    }
    return out.str();
}
std::int16_t cuda_device_index(const torch::Device& device) {
    if (!device.is_cuda()) {
        throw std::runtime_error("CUDA graph mode requires a CUDA device");
    }
    const std::int16_t index = device.index();
    return index >= 0 ? index : 0;
}

torch::Tensor run_eager(
    const beam::stream1_libtorch::PieceTransformerLibTorch& model,
    const torch::Tensor& states,
    std::int64_t warmup,
    std::int64_t iters,
    const torch::Device& device,
    double& elapsed_ms) {
    torch::Tensor logits;
    for (std::int64_t i = 0; i < warmup; ++i) {
        logits = model.forward(states);
    }
    beam::stream1_libtorch::synchronize_if_cuda(device);
    const auto start = std::chrono::steady_clock::now();
    for (std::int64_t i = 0; i < iters; ++i) {
        logits = model.forward(states);
    }
    beam::stream1_libtorch::synchronize_if_cuda(device);
    const auto stop = std::chrono::steady_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return logits;
}

torch::Tensor run_cuda_graph(
    const beam::stream1_libtorch::PieceTransformerLibTorch& model,
    const torch::Tensor& states,
    std::int64_t warmup,
    std::int64_t iters,
    const torch::Device& device,
    double& elapsed_ms) {
    const std::int16_t device_index = cuda_device_index(device);
    c10::cuda::CUDAStream graph_stream = c10::cuda::getStreamFromPool(false, device_index);
    torch::Tensor logits;

    {
        c10::cuda::CUDAStreamGuard guard(graph_stream);
        for (std::int64_t i = 0; i < warmup; ++i) {
            logits = model.forward(states);
        }
    }
    graph_stream.synchronize();

    at::cuda::CUDAGraph graph;
    {
        c10::cuda::CUDAStreamGuard guard(graph_stream);
        graph.capture_begin();
        logits = model.forward(states);
        graph.capture_end();
        graph.replay();
    }
    graph_stream.synchronize();

    const auto start = std::chrono::steady_clock::now();
    {
        c10::cuda::CUDAStreamGuard guard(graph_stream);
        for (std::int64_t i = 0; i < iters; ++i) {
            graph.replay();
        }
    }
    graph_stream.synchronize();
    const auto stop = std::chrono::steady_clock::now();
    elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return logits;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        c10::InferenceMode inference_mode(true);
        torch::Device device(args.device);
        if (device.is_cuda() && !torch::cuda::is_available()) {
            throw std::runtime_error("requested CUDA device but LibTorch CUDA is not available");
        }
        if (args.cuda_graph && !device.is_cuda()) {
            throw std::runtime_error("--cuda-graph requires a CUDA device");
        }
        beam::stream1_libtorch::PieceTransformerLibTorch model(args.weight_dir, device);
        const std::string mode = args.cuda_graph ? "cuda_graph" : "eager";

        std::ofstream csv;
        if (!args.csv_path.empty()) {
            csv.open(args.csv_path, std::ios::out | std::ios::trunc);
            csv << "mode,batch,iters,elapsed_ms,parents_per_sec,candidates_per_sec,device,dtype\n";
        }

        std::cout << "stream1_transformer_libtorch_backend=1"
                  << " mode=" << mode
                  << " device=" << device.str()
                  << " dtype=" << model.dtype_suffix.substr(1)
                  << " seq_len=" << model.seq_len
                  << " d_model=" << model.d_model
                  << " nhead=" << model.nhead
                  << " layers=" << model.num_layers
                  << " output_dim=" << model.output_dim
                  << " transposed_linear_weights=1"
                  << "\n";

        for (std::int64_t batch : args.batches) {
            torch::Tensor states = beam::stream1_libtorch::make_states(batch, model.state_len, model.num_classes, device);
            double elapsed_ms = 0.0;
            torch::Tensor logits = args.cuda_graph
                ? run_cuda_graph(model, states, args.warmup, args.iters, device, elapsed_ms)
                : run_eager(model, states, args.warmup, args.iters, device, elapsed_ms);
            const double parents_per_sec = (static_cast<double>(batch) * args.iters) / (elapsed_ms / 1000.0);
            const double candidates_per_sec = parents_per_sec * model.move_count;
            const torch::Tensor keys = beam::stream1_libtorch::score_keys(logits);
            const auto checksum = keys.sum().item<std::int64_t>();
            const std::string first_score_keys = format_first_score_keys(keys);
            std::cout << "stream1_transformer_libtorch_micro"
                      << " mode=" << mode
                      << " batch=" << batch
                      << " iters=" << args.iters
                      << " elapsed_ms=" << elapsed_ms
                      << " parents_per_sec=" << parents_per_sec
                      << " candidates_per_sec=" << candidates_per_sec
                      << " checksum=" << checksum
                      << " first_score_keys=" << first_score_keys
                      << "\n";
            if (csv) {
                csv << mode << ','
                    << batch << ','
                    << args.iters << ','
                    << elapsed_ms << ','
                    << parents_per_sec << ','
                    << candidates_per_sec << ','
                    << device.str() << ','
                    << model.dtype_suffix.substr(1) << '\n';
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "stream1_transformer_libtorch_error=" << e.what() << "\n";
        return 1;
    }
}