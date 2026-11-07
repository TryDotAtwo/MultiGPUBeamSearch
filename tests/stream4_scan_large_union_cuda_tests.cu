#define BEAM_TEST_UNION_CAPACITY 65537
#define BEAM_TEST_UNION_TEMP_BYTES (64ULL << 20)
#define BEAM_TEST_LOGICAL_UNION 1
#define BEAM_TEST_UNION_GRAPH 1
#define main inherited_union_fixture_main
#include "stream4_logical_union_cuda_tests.cu"
#undef main

int main() {
    try {
        int devices = 0; check(cudaGetDeviceCount(&devices));
        if (devices != 2) throw std::runtime_error("requires two physical GPUs");
        for (int gpu = 0; gpu < devices; ++gpu) {
            check(cudaSetDevice(gpu)); Fixture f;
            const std::array<std::pair<unsigned, unsigned>, 7> cases{{
                {0, 0}, {1, 1}, {255, 256}, {257, 513},
                {32769, 65537}, {65537, 65537}, {0, 0}}};
            for (const auto sizes : cases) {
                std::vector<CandidateMeta> a, b;
                for (unsigned side = 0; side < 2; ++side) {
                    auto& values = side ? b : a;
                    const unsigned n = side ? sizes.second : sizes.first;
                    values.reserve(n);
                    for (unsigned i = 0; i < n; ++i) {
                        const std::uint64_t id = (i + side * 11U) / 2U;
                        values.push_back(CandidateMeta{{id * 11400714819323198485ULL, id % 13U},
                            (i + side) % 31U, (n - i) % 23U, (i + side) % 24U});
                    }
                }
                f.run(a, b, 17U);
                if (!f.verify(a, b, 17U)) throw std::runtime_error("large union oracle mismatch");
            }
            std::cout << "gpu=" << gpu << " large_union_scan_graph=pass\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n'; return 1;
    }
}
