#include "device_weight_verification.cuh"
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <iostream>

int main() {
    for (std::size_t size : {1U, 16U, 257U, 131073U}) {
        std::vector<std::byte> expected(size);
        for (std::size_t i=0; i<size; ++i) expected[i]=std::byte((i*37U+11U)%256U);
        void* device=nullptr;
        BEAM_CUDA_CHECK(cudaMalloc(&device,size));
        BEAM_CUDA_CHECK(cudaMemcpy(device,expected.data(),size,cudaMemcpyHostToDevice));
        beam::verify_device_weight_bytes(device,expected,"fixture");
        for (std::size_t offset : {std::size_t(0),size/2,size-1}) {
            auto changed=expected[offset]^std::byte{1};
            BEAM_CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(device)+offset,&changed,1,cudaMemcpyHostToDevice));
            bool rejected=false;
            try { beam::verify_device_weight_bytes(device,expected,"fixture"); }
            catch (const std::runtime_error&) { rejected=true; }
            if (!rejected) throw std::runtime_error("device corruption not rejected");
            BEAM_CUDA_CHECK(cudaMemcpy(static_cast<std::byte*>(device)+offset,&expected[offset],1,cudaMemcpyHostToDevice));
        }
        BEAM_CUDA_CHECK(cudaFree(device));
    }
    std::cout << "device_weight_verification=pass sizes=4 corruptions=12\n";
}
