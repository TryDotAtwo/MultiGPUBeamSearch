// Independent public-field storage accounting; never launches attention.
#include "kernel_forward.h"
#include "../third_party/nlohmann/json.hpp"
#include <iostream>
#include <type_traits>
#include <vector>
#ifdef HAS_PYTORCH
#error This observer requires the standalone no-PyTorch kernel contract.
#endif
template<class Arch,int Q,int K,bool Aligned> nlohmann::json ledger() {
    using A=AttentionKernel<cutlass::half_t,Arch,Aligned,Q,64,K,false,false>;
    using P=typename A::Params;
    static_assert(std::is_standard_layout_v<P> && std::is_trivially_copyable_v<P>);
    P p{};auto fields=nlohmann::json::array();std::vector<unsigned char> covered(sizeof(P),0);
    auto add=[&](const char* name,const auto& member) {
        const auto offset=reinterpret_cast<std::uintptr_t>(&member)-reinterpret_cast<std::uintptr_t>(&p);
        if(offset+sizeof(member)>sizeof(P))throw std::runtime_error("field outside Params");
        for(std::size_t i=offset;i<offset+sizeof(member);++i)
            if(covered[i]++)throw std::runtime_error("overlapping field storage");
        fields.push_back({{"field",name},{"offset",offset},{"bytes",sizeof(member)}});
    };
#define FIELD(member) add(#member,p.member)
    FIELD(query_ptr);FIELD(key_ptr);FIELD(value_ptr);FIELD(attn_bias_ptr);
    FIELD(seqstart_q_ptr);FIELD(seqstart_k_ptr);FIELD(seqlen_k_ptr);FIELD(causal_diagonal_offset);
    FIELD(output_ptr);FIELD(output_accum_ptr);FIELD(logsumexp_ptr);FIELD(scale);
    FIELD(head_dim);FIELD(head_dim_value);FIELD(num_queries);FIELD(num_keys);FIELD(num_keys_absolute);
    FIELD(custom_mask_type);FIELD(q_strideM);FIELD(k_strideM);FIELD(v_strideM);FIELD(bias_strideM);
    FIELD(o_strideM);FIELD(q_strideH);FIELD(k_strideH);FIELD(v_strideH);FIELD(bias_strideH);
    FIELD(q_strideB);FIELD(k_strideB);FIELD(v_strideB);FIELD(bias_strideB);
    FIELD(num_batches);FIELD(num_heads);FIELD(use_dropout);FIELD(dropout_batch_head_rng_offset);FIELD(dropout_prob);
#undef FIELD
    auto gaps=nlohmann::json::array();
    for(std::size_t i=0;i<covered.size();) {
        if(covered[i]){++i;continue;}auto start=i;while(i<covered.size() && !covered[i])++i;
        gaps.push_back({{"offset",start},{"bytes",i-start}});
    }
    return {{"sm",int(Arch::kMinComputeCapability)},{"queries_per_block",Q},{"max_k",K},{"aligned",Aligned},
        {"parameter_bytes",sizeof(P)},{"alignment",alignof(P)},{"fields",fields},{"unclassified_storage_gaps",gaps}};
}
template<class Arch>void append(nlohmann::json& entries) {
    entries.push_back(ledger<Arch,64,64,true>());entries.push_back(ledger<Arch,32,64,true>());
    entries.push_back(ledger<Arch,64,32,true>());entries.push_back(ledger<Arch,32,32,true>());
    entries.push_back(ledger<Arch,64,64,false>());entries.push_back(ledger<Arch,64,32,false>());
}
int main() {
    try {
        auto entries=nlohmann::json::array();append<cutlass::arch::Sm75>(entries);append<cutlass::arch::Sm80>(entries);
        std::cout<<nlohmann::json{{"schema_version",1},{"scope","compiled_public_attention_storage_not_consumption_or_admission"},
            {"production_admitted",false},{"pytorch_rng_state_present",false},{"entries",entries}}.dump()<<'\n';
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
