#pragma once
#include <string>
#include <cstddef>
#include <vector>

namespace beam::score_mode {
inline std::vector<std::size_t> graph_diagnostic_u32_indices(const std::string& symbol) {
    if(symbol.rfind("_ZN4beam",0)!=0)return {};
    struct Entry {const char* token;std::vector<std::size_t> indices;};
    const Entry entries[]={
        {"stream1_transformer_build_input_layernorm256_generic_kernel",{6,7}},
        {"stream1_transformer_layernorm256_copy_kernel",{4,5}},
        {"stream1_transformer_bias_layernorm256_copy_kernel",{5,6}},
        {"stream1_transformer_gather_cls256_kernel",{3,4}},
        {"stream1_transformer_cls_bias_layernorm_kernel",{6}},
        {"stream1_transformer_build_input_kernel_graph_job",{6,7}},
        {"stream1_transformer_zero_padded_rows_kernel",{2,3}},
        {"stream1_transformer_zero_padded_rows_legacy_kernel",{2}},
        {"stream1_transformer_score_quantize_graph_job_kernel",{5,6,7}}};
    for(const auto& entry:entries)if(symbol.find(entry.token)!=std::string::npos)return entry.indices;
    return {};
}
// Known diagnostic signatures only. Do not discover argument count by making
// an intentionally invalid CUDA API call (strict memcheck treats that as error).
inline std::size_t graph_diagnostic_parameter_count(const std::string& symbol) {
    if(symbol.rfind("_ZN7cutlass6KernelI",0)==0 ||
       symbol.rfind("_ZN7cutlass7Kernel2I",0)==0 ||
       symbol.rfind("_Z29attention_kernel_batched_impl",0)==0) return 1;
    struct Entry {const char* token;std::size_t count;};
    const Entry entries[]={
        {"stream1_transformer_build_input_layernorm256_generic_kernel",11},
        {"stream1_transformer_layernorm256_copy_kernel",6},
        {"stream1_transformer_bias_layernorm256_copy_kernel",7},
        {"stream1_transformer_gather_cls256_kernel",5},
        {"stream1_transformer_cls_bias_layernorm_kernel",7},
        {"stream1_transformer_build_input_kernel_graph_job",8},
        {"stream1_transformer_zero_padded_rows_kernel",4},
        {"stream1_transformer_zero_padded_rows_legacy_kernel",3},
        {"stream1_transformer_score_quantize_graph_job_kernel",10}};
    if(symbol.rfind("_ZN4beam",0)!=0)return 0;
    for(const auto& entry:entries)if(symbol.find(entry.token)!=std::string::npos)return entry.count;
    return 0;
}
}
