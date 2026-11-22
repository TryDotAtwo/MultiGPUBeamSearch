#pragma once
#include "../src/types.hpp"
#include "cuda_check.hpp"
#include <filesystem>
#include <fstream>
#include <vector>
#include <stdexcept>
namespace beam::benchmark {
// Test fixture only: immutable rank-local full frontier, logical colors and zero padding.
inline std::uint64_t load_frontier(const std::filesystem::path& path,State128* destination,
                                   std::uint64_t capacity,std::uint32_t classes) {
    auto bytes=std::filesystem::file_size(path);
    if(bytes!=capacity*sizeof(State128)) throw std::runtime_error("benchmark frontier must fill exact local capacity");
    std::ifstream file(path,std::ios::binary);
    std::vector<State128> chunk(std::min<std::uint64_t>(capacity,65536));
    for(std::uint64_t offset=0;offset<capacity;) {
        auto count=std::min<std::uint64_t>(chunk.size(),capacity-offset);
        if(!file.read(reinterpret_cast<char*>(chunk.data()),count*sizeof(State128))) throw std::runtime_error("truncated benchmark frontier");
        for(std::size_t i=0;i<count;++i) {
            for(unsigned j=0;j<STATE_LEN;++j) if(chunk[i].v[j]>=classes) throw std::runtime_error("invalid benchmark class");
            for(unsigned j=STATE_LEN;j<STATE_STORAGE_LEN;++j) if(chunk[i].v[j]) throw std::runtime_error("nonzero benchmark padding");
        }
        BEAM_CUDA_CHECK(cudaMemcpy(destination+offset,chunk.data(),count*sizeof(State128),cudaMemcpyHostToDevice));
        offset+=count;
    }
    return capacity;
}
}
