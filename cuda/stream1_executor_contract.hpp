#pragma once
#include <cstring>
#include <stdexcept>
namespace beam {
enum class Stream1Executor { NativeGraph, NativeEager, LibTorchEager };
inline Stream1Executor resolve_stream1_executor(const char* value) {
    if (value == nullptr || value[0] == '\0' ||
        std::strcmp(value, "native_cuda_graph") == 0 || std::strcmp(value, "cuda_graph") == 0)
        return Stream1Executor::NativeGraph;
    if (std::strcmp(value, "native_eager") == 0 || std::strcmp(value, "native_no_graph") == 0)
        return Stream1Executor::NativeEager;
    if (std::strcmp(value, "libtorch_eager") == 0) return Stream1Executor::LibTorchEager;
    throw std::invalid_argument("BEAM_STREAM1_EXECUTOR must be native_cuda_graph, cuda_graph, native_eager, native_no_graph, or libtorch_eager");
}
} // namespace beam
