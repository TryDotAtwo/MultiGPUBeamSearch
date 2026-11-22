// Compile-time storage only: no CUDA context, allocation or kernel launch.
#include "../cuda/stream1.hpp"
#include "../cuda/stream1_transformer_layernorm_policy.hpp"
#include <iostream>
#include <cstddef>

int main() {
    std::cout << "{\"schema_version\":1,\"scope\":\"compiled_parameter_storage_not_admission\","
              << "\"pointer\":{\"size\":" << sizeof(void*) << ",\"alignment\":" << alignof(void*) << "},"
              << "\"u32\":{\"size\":" << sizeof(std::uint32_t) << ",\"alignment\":" << alignof(std::uint32_t) << "},"
              << "\"dims\":{\"size\":" << sizeof(beam::Stream1TransformerDims)
              << ",\"alignment\":" << alignof(beam::Stream1TransformerDims) << "},"
              << "\"network\":{\"size\":" << sizeof(beam::Stream1TransformerNetworkView)
              << ",\"alignment\":" << alignof(beam::Stream1TransformerNetworkView) << "},"
              << "\"ln_shared_bytes\":" << beam::STREAM1_TRANSFORMER_LN256_SHARED_FLOATS*sizeof(float)
              << "}\n";
}
