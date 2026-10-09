#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

static void check(cudaError_t result) {
    if(result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
int main() {
    try {
        int count = 0;
        check(cudaGetDeviceCount(&count));
        if(count <= 0 || count > 1024) throw std::runtime_error("invalid visible CUDA device count");
        std::ostringstream json;
        std::set<std::string> identities;
        json << "{\"schema_version\":1,\"scope\":\"observed_cuda_visible_devices_not_quality\",\"devices\":[";
        for(int ordinal = 0; ordinal < count; ++ordinal) {
            cudaDeviceProp prop{};
            check(cudaGetDeviceProperties(&prop, ordinal));
            std::ostringstream uuid;
            uuid << std::hex << std::setfill('0');
            bool nonzero = false;
            for(unsigned char byte : prop.uuid.bytes) {
                nonzero |= byte != 0;
                uuid << std::setw(2) << static_cast<unsigned>(byte);
            }
            if(!nonzero || !identities.insert(uuid.str()).second)
                throw std::runtime_error("missing or duplicate visible CUDA UUID");
            if(ordinal) json << ',';
            json << "{\"device\":" << ordinal << ",\"uuid_hex\":\"" << uuid.str()
                 << "\",\"sm\":" << prop.major * 10 + prop.minor << '}';
        }
        json << "],\"production_quality_accepted\":false}";
        std::cout << json.str() << '\n';
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "CUDA device probe rejected: " << error.what() << '\n';
        return 2;
    }
}
