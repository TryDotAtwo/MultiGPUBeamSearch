#include "../tools/graph_pointer_registry.hpp"
#include <array>
#include <cassert>
#include <limits>
#include <stdexcept>

using beam::score_mode::GraphPointerRegistry;
template<class F> void rejects(F f) {
    bool rejected=false;
    try { f(); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
int main() {
    std::array<unsigned char,128> allocation{};
    GraphPointerRegistry registry;
    registry.add("tokens",allocation.data(),64);
    registry.add("logits",allocation.data()+64,64);
    const auto found=registry.resolve(allocation.data()+16,32);
    assert(found.role=="tokens" && found.offset==16 && found.available_bytes==48);
    assert(registry.resolve(allocation.data()+127,1).role=="logits");
    rejects([&]{ registry.resolve(allocation.data()+63,2); });
    rejects([&]{ registry.resolve(allocation.data()+128,1); });
    rejects([&]{ registry.resolve(nullptr,1); });
    rejects([&]{ registry.resolve(allocation.data(),0); });
    rejects([&]{ registry.add("overlap",allocation.data()+32,64); });
    rejects([&]{ registry.add("tokens",allocation.data()+128,1); });
    rejects([&]{ registry.add("",allocation.data()+128,1); });
    rejects([&]{ registry.add("null",nullptr,4); });
    rejects([&]{ registry.add("overflow",allocation.data(),std::numeric_limits<std::size_t>::max()); });
    // Failed registration must leave the resolver unchanged.
    assert(registry.resolve(allocation.data()+64,64).role=="logits");
}
