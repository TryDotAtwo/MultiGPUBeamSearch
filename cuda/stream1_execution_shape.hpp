#pragma once
#include <cstdint>
#include <ostream>
#include <stdexcept>
#include <iomanip>
#include <sstream>

namespace beam {
inline void write_stream1_device_identity(std::ostream& out, int device, int sm,
    const char (&bytes)[16]) {
    if(device < 0 || sm < 75) throw std::invalid_argument("invalid Stream1 device identity");
    std::ostringstream uuid;
    uuid << std::hex << std::setfill('0');
    bool nonzero = false;
    for(unsigned char byte : bytes) {
        nonzero |= byte != 0;
        uuid << std::setw(2) << static_cast<unsigned>(byte);
    }
    if(!nonzero) throw std::invalid_argument("missing Stream1 device UUID");
    out << "{\"schema_version\":1,\"device\":" << device << ",\"sm\":" << sm
        << ",\"uuid_hex\":\"" << uuid.str() << "\",\"production_quality_accepted\":false}";
}
// Observed allocation/execution shape only; not resolved-kernel admission.
inline void write_stream1_execution_shape(std::ostream& out, std::uint32_t outer,
    std::uint32_t inner, std::uint32_t lanes, int device, int sm) {
    if (!outer || !inner || inner > outer || !lanes || device < 0 || sm < 75)
        throw std::invalid_argument("invalid native Stream1 execution shape");
    out << "{\"schema_version\":1,\"scope\":\"execution_shape_not_kernel_admission\","
        << "\"outer_microbatch\":" << outer << ",\"transformer_microbatch\":" << inner
        << ",\"lanes\":" << lanes << ",\"device\":" << device << ",\"sm\":" << sm
        << ",\"production_quality_accepted\":false}";
}
}
