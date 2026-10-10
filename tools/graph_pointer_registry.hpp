#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace beam::score_mode {
// Diagnostic only. Resolve captured addresses in-process; never export them.
class GraphPointerRegistry {
    struct Span { std::string role; std::uintptr_t base; std::size_t bytes; };
    std::vector<Span> spans_;
public:
    struct Endpoint { std::string role; std::size_t offset; std::size_t available_bytes; };
    void add(const std::string& role,const void* pointer,std::size_t bytes) {
        const auto base=reinterpret_cast<std::uintptr_t>(pointer);
        if(role.empty() || role.size()>128 || !base || !bytes ||
           bytes>std::numeric_limits<std::uintptr_t>::max()-base || spans_.size()>=256)
            throw std::runtime_error("invalid diagnostic pointer span");
        for(const auto& s:spans_)
            if(role==s.role || (base<s.base+s.bytes && s.base<base+bytes))
                throw std::runtime_error("ambiguous diagnostic pointer span");
        spans_.push_back({role,base,bytes});
    }
    Endpoint resolve(const void* pointer,std::size_t required_bytes) const {
        const auto address=reinterpret_cast<std::uintptr_t>(pointer);
        if(!address || !required_bytes)
            throw std::runtime_error("invalid diagnostic pointer endpoint");
        for(const auto& s:spans_)
            if(address>=s.base && address-s.base<s.bytes &&
               required_bytes<=s.bytes-(address-s.base))
                return {s.role,address-s.base,s.bytes-(address-s.base)};
        throw std::runtime_error("captured pointer outside registered spans");
    }
};
}
