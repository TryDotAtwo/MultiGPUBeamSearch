#include "stream1_kernel_description.hpp"
#include <stdexcept>
#include <iostream>
int main() {
    using namespace beam;
    Stream1KernelDescription a{"qkv", "cutlass_sm75", "fp16", "identity", 75,
        8, 256, 768, 128, 64, 32, 64, 32, 32, 16, 8, 8, 2, 1, false};
    validate_stream1_kernel_description(a);
    if (active_stream1_kernel_observation()!=nullptr) throw std::runtime_error("unexpected observer");
    Stream1KernelObservation scoped(2), nested(1);
    {
        Stream1KernelObservationScope outer(scoped);
        active_stream1_kernel_observation()->record(a);
        { Stream1KernelObservationScope inner(nested);
          active_stream1_kernel_observation()->record(a); }
        if (active_stream1_kernel_observation()!=&scoped) throw std::runtime_error("scope restore lost");
    }
    if (active_stream1_kernel_observation()!=nullptr || scoped.records().size()!=1 || nested.records().size()!=1)
        throw std::runtime_error("observer ownership lost");
    auto b=a; b.stages=3;
    if (a==b) throw std::runtime_error("stage identity lost");
    b=a; b.instruction_k=16;
    if (a==b) throw std::runtime_error("instruction identity lost");
    b=a; b.packed_weight=true;
    if (a==b) throw std::runtime_error("layout identity lost");
    Stream1KernelObservation observations(2);
    observations.record(a); observations.record(a); observations.record(b);
    if (observations.records().size()!=2) throw std::runtime_error("duplicate handling");
    auto c=a; c.rows=7;
    bool rejected=false;
    try { observations.record(c); } catch (const std::overflow_error&) { rejected=true; }
    if (!rejected || observations.records().size()!=2) throw std::runtime_error("silent trace truncation");
    for (unsigned invalid=0;invalid<3;++invalid) {
        auto bad=a;
        if (invalid==0) bad.stages=0;
        if (invalid==1) bad.rows=0;
        if (invalid==2) bad.family.clear();
        rejected=false;
        try { validate_stream1_kernel_description(bad); }
        catch (const std::invalid_argument&) { rejected=true; }
        if (!rejected) throw std::runtime_error("incomplete identity accepted");
    }
    std::cout << "kernel_description=pass\n";
}
