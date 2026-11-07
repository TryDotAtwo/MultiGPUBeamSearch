// Independent pinned CUTLASS empty-argument exclusion for canonical Identity/ReLU.
// Does not execute CUDA or classify arbitrary activation types as empty.
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/numeric_types.h>
#include <type_traits>
#include <iostream>
using E=cutlass::half_t;
template<class Activation>using Op=cutlass::epilogue::thread::LinearCombinationBiasElementwise<
    E,float,float,E,E,8,Activation,cutlass::plus<float>,false,E>;
template<class Activation>void emit(const char* name,bool comma) {
    using A=typename Op<Activation>::ElementwiseArguments;
    using Empty=cutlass::epilogue::thread::detail::EmptyArguments;
    static_assert(std::is_same_v<A,Empty>,"canonical activation acquired arguments");
    static_assert(std::is_empty_v<A>,"canonical activation arguments acquired state");
    static_assert(std::is_trivially_copyable_v<A>);
    if(comma)std::cout<<',';
    std::cout<<"{\"activation\":\""<<name<<"\",\"empty\":true,\"pinned_empty_type\":true,"
        <<"\"storage_bytes\":"<<sizeof(A)<<",\"alignment\":"<<alignof(A)<<'}';
}
int main() {
    std::cout<<"{\"schema_version\":1,\"scope\":\"compiled_canonical_empty_elementwise_arguments_not_admission\",\"entries\":[";
    emit<cutlass::epilogue::thread::Identity<float>>("Identity",false);
    emit<cutlass::epilogue::thread::ReLu<float>>("ReLu",true);
    std::cout<<"]}\n";
}
