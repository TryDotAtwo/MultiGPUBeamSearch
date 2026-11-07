// Independent compiled classic SM75/SM80 geometry/signatures; no Params/capture calls.
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/activation.h>
#include <iostream>
#include <type_traits>
using E=cutlass::half_t;using L=cutlass::layout::RowMajor;
using S=cutlass::gemm::GemmShape<128,64,32>;
using W=cutlass::gemm::GemmShape<64,32,32>;
using I=cutlass::gemm::GemmShape<16,8,16>;
using R=cutlass::gemm::device::Gemm<E,L,E,L,E,L,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,S,W,I>;
using I75=cutlass::gemm::GemmShape<16,8,8>;
using R75=cutlass::gemm::device::Gemm<E,L,E,L,E,L,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm75,S,W,I75>;
template<class A>using Op=cutlass::epilogue::thread::LinearCombinationBiasElementwise<E,float,float,E,E,8,
    A,cutlass::plus<float>,false,E>;
template<class A>using B=cutlass::gemm::device::GemmUniversalWithBroadcast<E,L,E,L,E,L,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,S,W,I,Op<A>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,3>;
template<class A>using B75=cutlass::gemm::device::GemmUniversalWithBroadcast<E,L,E,L,E,L,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm75,S,W,I75,Op<A>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,2>;
template<class T,class=void>struct ProbeUnderlying:std::false_type{};
template<class T>struct ProbeUnderlying<T,std::void_t<typename T::UnderlyingIterator>>:std::true_type{};
template<class T,class=void>struct ProbeTileAccess:std::false_type{};
template<class T>struct ProbeTileAccess<T,std::void_t<typename T::TileAccessIterator>>:std::true_type{};
template<class T>void emit_mainloop_descriptor() {
    if constexpr(ProbeUnderlying<T>::value)emit_mainloop_descriptor<typename T::UnderlyingIterator>();
    else if constexpr(ProbeTileAccess<T>::value)emit_mainloop_descriptor<typename T::TileAccessIterator>();
    else {
        using Shape=typename T::Shape;using Map=typename T::ThreadMap;
        std::cout<<"{\"element_bits\":"<<cutlass::sizeof_bits<typename T::Element>::value
            <<",\"advance_rank\":"<<T::kAdvanceRank
            <<",\"shape_contiguous\":"<<Shape::kContiguous
            <<",\"shape_strided\":"<<Shape::kStrided
            <<",\"iterations_strided\":"<<Map::Iterations::kStrided
            <<",\"delta_strided\":"<<Map::Delta::kStrided<<'}';
    }
}
template<class G,bool Broadcast>void emit(const char* name) {
    using T=typename G::GemmKernel::Epilogue::OutputTileIterator::ThreadMap;
    if constexpr(Broadcast) {
        using Tensor=typename G::GemmKernel::Epilogue::TensorTileIterator;
        static_assert(std::is_same_v<T,typename Tensor::ThreadMap>,"Tensor thread map differs");
        static_assert(cutlass::sizeof_bits<typename Tensor::Element>::value==16,"Tensor element differs");
    }
    using S=typename T::Shape;using I=typename T::Iterations;
    using D=typename T::Delta;using C=typename T::Count;
    const char* symbol=nullptr;cudaError_t status;
    if constexpr(Broadcast)status=cudaFuncGetName(&symbol,cutlass::Kernel2<typename G::GemmKernel>);
    else status=cudaFuncGetName(&symbol,cutlass::Kernel<typename G::GemmKernel>);
    if(status!=cudaSuccess || !symbol)throw std::runtime_error("independent signature lookup failed");
    std::cout<<'"'<<name<<"\":{\"symbol\":\""<<symbol<<"\",\"descriptor\":{\"shape\":{\"row\":"<<S::kRow
        <<",\"group\":"<<S::kGroup<<",\"cluster\":"<<S::kCluster<<",\"tile\":"<<S::kTile
        <<"},\"iterations\":{\"row\":"<<I::kRow<<",\"group\":"<<I::kGroup
        <<"},\"delta\":{\"row\":"<<D::kRow<<",\"group\":"<<D::kGroup<<",\"cluster\":"<<D::kCluster
        <<"},\"count\":{\"row\":"<<C::kRow<<",\"group\":"<<C::kGroup<<"}},\"mainloop\":{\"A\":";
    emit_mainloop_descriptor<typename G::GemmKernel::Mma::IteratorA>();
    std::cout<<",\"B\":";
    emit_mainloop_descriptor<typename G::GemmKernel::Mma::IteratorB>();std::cout<<"}}";
}
int main(){std::cout<<'{';emit<R,false>("regular");std::cout<<',';
    emit<B<cutlass::epilogue::thread::Identity<float>>,true>("broadcast");std::cout<<',';
    emit<B<cutlass::epilogue::thread::ReLu<float>>,true>("relu");std::cout<<',';
    emit<R75,false>("regular75");std::cout<<',';
    emit<B75<cutlass::epilogue::thread::Identity<float>>,true>("broadcast75");std::cout<<',';
    emit<B75<cutlass::epilogue::thread::ReLu<float>>,true>("relu75");std::cout<<"}\n";}
