// Compile-time expectations from the pinned CUTLASS implementation.
// No CUDA context, graph inspection, or kernel launch is performed.
#include <cuda_fp16.h>
#include <cutlass/numeric_types.h>
#include "kernel_forward.h"
#include <iostream>

template<class Arch, int Sm, int Q, int K, bool Aligned>
void emit(bool& first) {
    using A = AttentionKernel<cutlass::half_t, Arch, Aligned, Q, 64, K, false, false>;
    if (!first) std::cout << ',';
    first = false;
    std::cout << "{\"architecture\":" << Sm
              << ",\"queries_per_block\":" << Q
              << ",\"keys_per_block\":64,\"max_k\":" << K
              << ",\"aligned\":" << (Aligned ? "true" : "false")
#ifdef STREAM1_ATTENTION_PARAMETER_LAYOUT_PROBE
              << ",\"parameter_size\":" << sizeof(typename A::Params)
              << ",\"parameter_alignment\":" << alignof(typename A::Params)
#else
              << ",\"block\":[32," << A::kNumWarpsPerBlock
              << ",1],\"dynamic_shared_bytes\":" << sizeof(typename A::SharedStorage)
#endif
              << '}';
}
template<class Arch, int Sm>
void architecture(bool& first) {
    emit<Arch, Sm, 64, 64, true>(first);
    emit<Arch, Sm, 32, 64, true>(first);
    emit<Arch, Sm, 64, 32, true>(first);
    emit<Arch, Sm, 32, 32, true>(first);
    emit<Arch, Sm, 64, 64, false>(first);
    emit<Arch, Sm, 64, 32, false>(first);
}
int main() {
    bool first = true;
#ifdef STREAM1_ATTENTION_PARAMETER_LAYOUT_PROBE
    std::cout << "{\"schema_version\":1,\"scope\":\"compile_time_attention_parameter_storage_not_admission\",\"entries\":[";
#else
    std::cout << "{\"schema_version\":1,\"scope\":\"compile_time_attention_storage_not_admission\",\"entries\":[";
#endif
    architecture<cutlass::arch::Sm75, 75>(first);
    architecture<cutlass::arch::Sm80, 80>(first);
    std::cout << "]}\n";
    return std::cout.good() ? 0 : 1;
}
