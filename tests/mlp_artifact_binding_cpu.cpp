#define BEAM_STREAM1_WEIGHT_IO_MANIFEST_ONLY
#include "../tools/stream1_weight_io.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    try {
        auto weights = beam::stream1_weights::load_stream1_weights(argv[1]);
        if (argc == 3) {
            if (std::string(argv[2]) != "owned" || !std::getenv("BEAM_MLP_GRAPH_HASH") ||
                !std::getenv("BEAM_MLP_ARTIFACT_HASH")) return 2;
            const std::filesystem::path directory(argv[1]);
            const auto manifest = beam::stream1_weights::read_text_exact(directory / "manifest.json");
            const auto suffix = beam::stream1_weights::weight_suffix(weights.model);
            std::fstream file(directory / (std::string("input_bias") + suffix),
                              std::ios::in | std::ios::out | std::ios::binary);
            char byte = 0;
            if (!file.get(byte)) throw std::runtime_error("cannot read test tensor");
            byte ^= 1;
            file.seekp(0);
            file.put(byte);
            file.close();
            if (!file) throw std::runtime_error("cannot mutate test tensor");
            // The file is no longer the source: only the already-owned vector
            // and the exact manifest captured by the loader may be hashed.
            beam::stream1_weights::bind_mlp_artifact(weights, manifest,
                std::getenv("BEAM_MLP_GRAPH_HASH"), std::getenv("BEAM_MLP_ARTIFACT_HASH"));
            weights.input_bias[0] ^= std::byte{1};
            try {
                beam::stream1_weights::bind_mlp_artifact(weights, manifest,
                    std::getenv("BEAM_MLP_GRAPH_HASH"), std::getenv("BEAM_MLP_ARTIFACT_HASH"));
                throw std::logic_error("changed owned buffer was accepted");
            } catch (const std::runtime_error& error) {
                if (std::string(error.what()).find("artifact hash mismatch") == std::string::npos) throw;
                if (!weights.verified_mlp_artifact.empty())
                    throw std::runtime_error("failed verification retained an old binding");
            }
            std::cout << "OWNED_BUFFER_BINDING_PASS\n";
        }
        std::cout << "loaded_mlp_bytes="
                  << beam::stream1_weights::total_host_weight_bytes(weights) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "load_error=" << error.what() << '\n';
        return 1;
    }
}
