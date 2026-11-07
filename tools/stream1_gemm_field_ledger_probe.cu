// Independent public top-level Params storage ledger; not value/admission proof.
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/activation.h>
#include "../third_party/nlohmann/json.hpp"
#include <type_traits>
#include <iostream>
using E=cutlass::half_t;using R=cutlass::layout::RowMajor;
using S=cutlass::gemm::GemmShape<128,64,32>;using W=cutlass::gemm::GemmShape<64,32,32>;
using I=cutlass::gemm::GemmShape<16,8,16>;
template<class Arch,class Inst>using Regular=cutlass::gemm::device::Gemm<E,R,E,R,E,R,float,cutlass::arch::OpClassTensorOp,
    Arch,S,W,Inst>;
template<class A>using Op=cutlass::epilogue::thread::LinearCombinationBiasElementwise<E,float,float,E,E,8,A,cutlass::plus<float>,false,E>;
template<class Arch,class Inst,int Stages,class A>using Broadcast=cutlass::gemm::device::GemmUniversalWithBroadcast<E,R,E,R,E,R,float,
    cutlass::arch::OpClassTensorOp,Arch,S,W,Inst,Op<A>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,Stages>;
template<class G,bool B>nlohmann::json ledger(const char* name) {
    using P=typename G::GemmKernel::Params;
    static_assert(std::is_trivially_copyable_v<P> && !std::is_polymorphic_v<P>);
    P p{};auto fields=nlohmann::json::array();
    auto add=[&](const char* label,const auto& member,const char* kind="observed") {
        const auto begin=reinterpret_cast<std::uintptr_t>(&p);
        const auto address=reinterpret_cast<std::uintptr_t>(&member);
        const auto bytes=sizeof(member);
        if(address<begin || address-begin+bytes>sizeof(P))throw std::runtime_error("field outside Params");
        fields.push_back({{"field",label},{"offset",address-begin},{"bytes",bytes},{"kind",kind}});
    };
#define FIELD(member) add(#member,p.member)
    FIELD(problem_size);FIELD(grid_tiled_shape);FIELD(swizzle_log_tile);
    FIELD(params_A);FIELD(params_B);FIELD(params_C);FIELD(params_D);
    FIELD(output_op.alpha);FIELD(output_op.beta);FIELD(output_op.alpha_ptr);FIELD(output_op.beta_ptr);
    FIELD(semaphore);FIELD(gemm_k_size);
    if constexpr(B) {
        using Arguments=typename G::GemmKernel::EpilogueOutputOp::ElementwiseArguments;
        static_assert(std::is_empty_v<Arguments> && std::is_same_v<Arguments,cutlass::epilogue::thread::detail::EmptyArguments>);
        add("output_op.elementwise",p.output_op.elementwise,"empty_activation_arguments");
        FIELD(mode);FIELD(batch_count);FIELD(batch_stride_D);FIELD(params_Tensor);
        FIELD(ptr_A);FIELD(ptr_B);FIELD(ptr_C);FIELD(ptr_D);FIELD(ptr_Vector);FIELD(ldr);FIELD(ptr_Tensor);
        FIELD(batch_stride_A);FIELD(batch_stride_B);FIELD(batch_stride_C);FIELD(batch_stride_Vector);FIELD(batch_stride_Tensor);
    } else {
        FIELD(output_op.alpha_ptr_array);FIELD(output_op.beta_ptr_array);
        FIELD(ref_A);FIELD(ref_B);FIELD(ref_C);FIELD(ref_D);
        FIELD(gather_A_indices);FIELD(gather_B_indices);FIELD(scatter_D_indices);
    }
#undef FIELD
    std::vector<unsigned char> covered(sizeof(P),0);
    for(const auto& field:fields) {
        auto offset=field.at("offset").get<std::size_t>();auto bytes=field.at("bytes").get<std::size_t>();
        for(std::size_t i=offset;i<offset+bytes;++i)if(covered[i]++)throw std::runtime_error("overlapping public field storage");
    }
    auto gaps=nlohmann::json::array();
    for(std::size_t i=0;i<covered.size();) {
        if(covered[i]){++i;continue;}const auto start=i;while(i<covered.size() && !covered[i])++i;
        gaps.push_back({{"offset",start},{"bytes",i-start}});
    }
    return {{"family",name},{"parameter_bytes",sizeof(P)},{"alignment",alignof(P)},
        {"fields",fields},{"unclassified_storage_gaps",gaps}};
}
int main() {
    try {
        nlohmann::json out={{"schema_version",1},{"scope","compiled_public_gemm_field_storage_not_consumption_or_admission"},
            {"production_admitted",false},{"entries",{
                ledger<Regular<cutlass::arch::Sm80,I>,false>("sm80_regular"),
                ledger<Broadcast<cutlass::arch::Sm80,I,3,cutlass::epilogue::thread::Identity<float>>,true>("sm80_bias"),
                ledger<Broadcast<cutlass::arch::Sm80,I,3,cutlass::epilogue::thread::ReLu<float>>,true>("sm80_relu"),
                ledger<Regular<cutlass::arch::Sm75,cutlass::gemm::GemmShape<16,8,8>>,false>("sm75_regular"),
                ledger<Broadcast<cutlass::arch::Sm75,cutlass::gemm::GemmShape<16,8,8>,2,cutlass::epilogue::thread::Identity<float>>,true>("sm75_bias"),
                ledger<Broadcast<cutlass::arch::Sm75,cutlass::gemm::GemmShape<16,8,8>,2,cutlass::epilogue::thread::ReLu<float>>,true>("sm75_relu")}}};
        std::cout<<out.dump()<<'\n';
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
