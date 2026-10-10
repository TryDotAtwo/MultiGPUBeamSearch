#include "../cuda/stream1_executor_contract.hpp"
#include <stdexcept>
int main() {
    using beam::Stream1Executor;
    for (const char* value : {static_cast<const char*>(nullptr), "", "native_cuda_graph", "cuda_graph"})
        if (beam::resolve_stream1_executor(value) != Stream1Executor::NativeGraph) return 1;
    for (const char* value : {"native_eager", "native_no_graph"})
        if (beam::resolve_stream1_executor(value) != Stream1Executor::NativeEager) return 2;
    if (beam::resolve_stream1_executor("libtorch_eager") != Stream1Executor::LibTorchEager) return 3;
    for (const char* value : {"auto", "native_cuda_graph ", "CUDA_GRAPH", "0"}) {
        try { beam::resolve_stream1_executor(value); return 4; }
        catch (const std::invalid_argument&) {}
    }
    return 0;
}
