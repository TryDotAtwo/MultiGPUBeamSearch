#include "../tools/native_cuda_device.hpp"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        int count = 0;
        BEAM_CUDA_CHECK(cudaGetDeviceCount(&count));
        if (count <= 0) throw std::runtime_error("no CUDA devices");
        const auto local = static_cast<std::uint32_t>(count == 1 ? 7 : count - 1);
        const int ordinal = beam::distributed::select_cuda_device(local);
        int active = -1;
        BEAM_CUDA_CHECK(cudaGetDevice(&active));
        if (ordinal != count - 1 || active != ordinal)
            throw std::runtime_error("search/score helper bound the wrong CUDA device");
        std::cout << "cluster_cuda_device_tests=pass visible=" << count
                  << " local_rank=" << local << " cuda_ordinal=" << active << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
