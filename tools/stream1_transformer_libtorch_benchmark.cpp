#include "stream1_transformer_libtorch_backend.hpp"

#include <c10/core/InferenceMode.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

} // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        c10::InferenceMode inference_mode(true);
        torch::Device device(args.device);
        if (device.is_cuda() && !torch::cuda::is_available()) {
            throw std::runtime_error("requested CUDA device but LibTorch CUDA is not available");
        }
        beam::stream1_libtorch::PieceTransformerLibTorch model(args.weight_dir, device);

        std::ofstream csv;
        if (!args.csv_path.empty()) {
            csv.open(args.csv_path, std::ios::out | std::ios::trunc);
            csv << "batch,iters,elapsed_ms,parents_per_sec,candidates_per_sec,device,dtype\n";
        }

        std::cout << "stream1_transformer_libtorch_backend=1"
                  << " device=" << device.str()
                  << " dtype=" << model.dtype_suffix.substr(1)
                  << " seq_len=" << model.seq_len
                  << " d_model=" << model.d_model
                  << " nhead=" << model.nhead
                  << " layers=" << model.num_layers
                  << " output_dim=" << model.output_dim
                  << "\n";

        for (std::int64_t batch : args.batches) {
            torch::Tensor states = beam::stream1_libtorch::make_states(batch, model.state_len, model.num_classes, device);
            torch::Tensor logits;
            for (std::int64_t i = 0; i < args.warmup; ++i) {
                logits = model.forward(states);
            }
            beam::stream1_libtorch::synchronize_if_cuda(device);
            const auto start = std::chrono::steady_clock::now();
            for (std::int64_t i = 0; i < args.iters; ++i) {
                logits = model.forward(states);
            }
            beam::stream1_libtorch::synchronize_if_cuda(device);
            const auto stop = std::chrono::steady_clock::now();
            const double elapsed_ms = std::chrono::duration<double, std::milli>(stop - start).count();
            const double parents_per_sec = (static_cast<double>(batch) * args.iters) / (elapsed_ms / 1000.0);
            const double candidates_per_sec = parents_per_sec * model.move_count;
            const auto checksum = beam::stream1_libtorch::score_keys(logits).sum().item<std::int64_t>();
            std::cout << "stream1_transformer_libtorch_micro"
                      << " batch=" << batch
                      << " iters=" << args.iters
                      << " elapsed_ms=" << elapsed_ms
                      << " parents_per_sec=" << parents_per_sec
                      << " candidates_per_sec=" << candidates_per_sec
                      << " checksum=" << checksum
                      << "\n";
            if (csv) {
                csv << batch << ','
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
