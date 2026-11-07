#include "stream1_mlp_libtorch_backend.hpp"
#include <algorithm>
#include <iostream>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDACachingAllocator.h>

int run_case(int argc, char** argv) {
    if (argc != 3) return 2;
    const auto center = beam::stream1_libtorch::parse_csv_i64(argv[2]);
    const auto n = std::int64_t(center.size());
    const auto storage = ((n + 4 + 15) / 16) * 16;
    std::vector<std::int64_t> state = center;
    std::sort(state.begin(),state.end());
    std::vector<std::uint8_t> host;
    do {
        for (const auto label : state) host.push_back(static_cast<std::uint8_t>(label));
        host.insert(host.end(),storage-n,255);
    } while (std::next_permutation(state.begin(),state.end()));
    const auto count = std::int64_t(host.size()) / storage;
    std::vector<std::uint8_t> generators(3*storage,255);
    for (std::int64_t p=0;p<n;++p) {
        generators[p]=(p+1)%n;
        generators[storage+p]=(p+n-1)%n;
        generators[2*storage+p]=p==0?1:p==1?0:p;
    }
    torch::NoGradGuard guard;
    const auto device=torch::Device(torch::kCUDA,0);
    auto states=torch::from_blob(host.data(),{count,storage},torch::kUInt8).clone().to(device);
    auto table=torch::from_blob(generators.data(),{3,storage},torch::kUInt8).clone().to(device);
    auto children=beam::stream1_libtorch::scalar_children(states,table.data_ptr<std::uint8_t>(),3,n,storage);
    beam::stream1_libtorch::MlpLibTorch model(argv[1],device);
    auto scores=model.forward(children).to(torch::kFloat32).cpu().contiguous();
    auto observed=children.cpu().contiguous();
    for (std::int64_t i=0;i<count;++i) for (std::int64_t move=0;move<3;++move) {
        int expected=0;
        for (std::int64_t p=0;p<n;++p) {
            const auto label=host[i*storage+generators[move*storage+p]];
            if (observed.data_ptr<std::uint8_t>()[(i*3+move)*n+p]!=label) return 3;
            expected+=label!=center[p];
        }
        if (scores.data_ptr<float>()[i*3+move]!=expected) return 4;
    }
    torch::cuda::synchronize();
    std::cout << "PASS states=" << count << " children=" << count*3 << " poisoned_padding=255\n";
    return 0;
}

int main(int argc, char** argv) {
    const int result = run_case(argc, argv);
    torch::cuda::synchronize();
    at::cuda::clearCublasWorkspaces();
    c10::cuda::CUDACachingAllocator::emptyCache();
    return result;
}
