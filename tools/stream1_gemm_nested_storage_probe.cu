// Independent classic nested storage/accessor checks; no captured Params or kernel launch.
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/transform/threadblock/predicated_tile_access_iterator_params.h>
#include "../third_party/nlohmann/json.hpp"
#include <iostream>
#include <type_traits>
#include <cstring>
using E=cutlass::half_t;using L=cutlass::layout::RowMajor;
using S=cutlass::gemm::GemmShape<128,64,32>;using W=cutlass::gemm::GemmShape<64,32,32>;
using Base=cutlass::transform::threadblock::PredicatedTileAccessIteratorParams;
template<class T,class=void>struct Under:std::false_type{};
template<class T>struct Under<T,std::void_t<typename T::UnderlyingIterator>>:std::true_type{};
template<class T,class=void>struct Tile:std::false_type{};
template<class T>struct Tile<T,std::void_t<typename T::TileAccessIterator>>:std::true_type{};
template<class T>nlohmann::json wrappers() {
    using P=typename T::Params;
    static_assert(std::is_standard_layout_v<P> && std::is_trivially_copyable_v<P>);
    static_assert(sizeof(P)==sizeof(Base) && alignof(P)==alignof(Base));
    auto result=nlohmann::json::array();result.push_back({{"bytes",sizeof(P)},{"alignment",alignof(P)},
        {"standard_layout",true},{"trivially_copyable",true}});
    if constexpr(Under<T>::value)for(const auto& x:wrappers<typename T::UnderlyingIterator>())result.push_back(x);
    else if constexpr(Tile<T>::value)for(const auto& x:wrappers<typename T::TileAccessIterator>())result.push_back(x);
    else {static_assert(std::is_same_v<typename P::Base,Base> && std::is_base_of_v<Base,P>);}
    return result;
}
template<class T>nlohmann::json epilogue(T& value) {
    static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>);
    auto fields=nlohmann::json::array();
    auto add=[&](const char* name,const auto& member){fields.push_back({{"field",name},
        {"offset",reinterpret_cast<std::uintptr_t>(&member)-reinterpret_cast<std::uintptr_t>(&value)},{"bytes",sizeof(member)}});};
#define F(x) add(#x,value.x)
    F(stride);F(increment_row);F(increment_group);F(increment_cluster);
    F(advance_row);F(advance_group);F(advance_cluster);F(advance_tile);
#undef F
    return {{"bytes",sizeof(T)},{"alignment",alignof(T)},{"fields",fields}};
}
template<class T>nlohmann::json tensor_ref() {
    static_assert(std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T>);
    static_assert(std::is_same_v<typename T::Layout,L> && L::kStrideRank==1);
    E dummy{};T ref(&dummy,L(137));auto layout_offset=reinterpret_cast<std::uintptr_t>(&ref.layout())-reinterpret_cast<std::uintptr_t>(&ref);
    auto stride_offset=reinterpret_cast<std::uintptr_t>(&ref.stride(0))-reinterpret_cast<std::uintptr_t>(&ref);
    E* decoded=nullptr;std::memcpy(&decoded,&ref,sizeof(decoded));
    if(decoded!=ref.data() || layout_offset!=sizeof(void*) || stride_offset!=layout_offset ||
       sizeof(T)!=sizeof(void*)+sizeof(std::int64_t) || sizeof(L)!=sizeof(std::int64_t) || ref.stride(0)!=137)
        throw std::runtime_error("TensorRef accessor/storage decomposition changed");
    return {{"bytes",sizeof(T)},{"alignment",alignof(T)},{"pointer_offset",0},
        {"pointer_bytes",sizeof(void*)},{"layout_offset",layout_offset},{"layout_bytes",sizeof(L)},
        {"stride_offset",stride_offset},{"stride_bytes",sizeof(ref.stride(0))},{"accessor_roundtrip_checked",true}};
}
template<class G,bool B>nlohmann::json row(const char* name) {
    using K=typename G::GemmKernel;typename K::Params p{};
    nlohmann::json out={{"family",name},{"mainloop_A",wrappers<typename K::Mma::IteratorA>()},
        {"mainloop_B",wrappers<typename K::Mma::IteratorB>()},{"epilogue_C",epilogue(p.params_C)},
        {"epilogue_D",epilogue(p.params_D)}};
    if constexpr(B)out["tensor_iterator"]=epilogue(p.params_Tensor);
    else {out["ref_A"]=tensor_ref<decltype(p.ref_A)>();out["ref_B"]=tensor_ref<decltype(p.ref_B)>();
        out["ref_C"]=tensor_ref<decltype(p.ref_C)>();out["ref_D"]=tensor_ref<decltype(p.ref_D)>();}
    return out;
}
template<class Arch,class I,int Stages>void append(nlohmann::json& rows,const char* regular,const char* bias,const char* relu) {
    using G=cutlass::gemm::device::Gemm<E,L,E,L,E,L,float,cutlass::arch::OpClassTensorOp,Arch,S,W,I>;
    using O1=cutlass::epilogue::thread::LinearCombinationBiasElementwise<E,float,float,E,E,8,cutlass::epilogue::thread::Identity<float>,cutlass::plus<float>,false,E>;
    using O2=cutlass::epilogue::thread::LinearCombinationBiasElementwise<E,float,float,E,E,8,cutlass::epilogue::thread::ReLu<float>,cutlass::plus<float>,false,E>;
    using B1=cutlass::gemm::device::GemmUniversalWithBroadcast<E,L,E,L,E,L,float,cutlass::arch::OpClassTensorOp,Arch,S,W,I,O1,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,Stages>;
    using B2=cutlass::gemm::device::GemmUniversalWithBroadcast<E,L,E,L,E,L,float,cutlass::arch::OpClassTensorOp,Arch,S,W,I,O2,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,Stages>;
    rows.push_back(row<G,false>(regular));rows.push_back(row<B1,true>(bias));rows.push_back(row<B2,true>(relu));
}
int main(){try{auto rows=nlohmann::json::array();
    append<cutlass::arch::Sm80,cutlass::gemm::GemmShape<16,8,16>,3>(rows,"sm80_regular","sm80_bias","sm80_relu");
    append<cutlass::arch::Sm75,cutlass::gemm::GemmShape<16,8,8>,2>(rows,"sm75_regular","sm75_bias","sm75_relu");
    std::cout<<nlohmann::json{{"schema_version",1},{"scope","compiled_nested_gemm_storage_not_consumption_or_admission"},{"production_admitted",false},{"entries",rows}}.dump()<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
